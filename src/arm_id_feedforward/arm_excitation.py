#!/usr/bin/env python3
"""Offline FIGAROH/IPOPT excitation design, no torque constraints or hardware."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import time

import coal
import numpy as np
import pinocchio as pin
from scipy.interpolate import BSpline
from figaroh.tools.robotipopt import BaseOptimizationProblem, IPOPTConfig, RobotIPOPTSolver

if __package__:
    from . import arm_collect, arm_plan
    from .arm_dynamics import DynamicModel, information
else:
    import arm_collect
    import arm_plan
    from arm_dynamics import DynamicModel, information

DURATION=24.
SEGMENTS=3
CONTROL_POINTS=11
DEGREE=5
MAX_VELOCITY=.25
MAX_ACCELERATION=.5
OPT_FREQUENCY=5.
CONSTRAINT_FREQUENCY=3.
OUTPUT_FREQUENCY=100.


class ExactGeometry:
    """Convex-proxy self-collision and floor clearance for offline planning."""
    def __init__(self,model):
        self.model=model;self.data=model.createData()
        self.geometry=pin.buildGeomFromUrdf(model,str(arm_collect.URDF),pin.GeometryType.COLLISION,
                                           package_dirs=[str(arm_collect.URDF.parent)])
        original=len(self.geometry.geometryObjects)
        for a in range(original):
            for b in range(a+1,original):
                if abs(self.geometry.geometryObjects[a].parentJoint-self.geometry.geometryObjects[b].parentJoint)>1:
                    self.geometry.addCollisionPair(pin.CollisionPair(a,b))
        floor=self.geometry.addGeometryObject(pin.GeometryObject('floor',0,pin.SE3.Identity(),coal.Halfspace(np.array([0.,0.,1.]),0.)))
        for i in range(original):
            obj=self.geometry.geometryObjects[i]
            if obj.parentJoint==0:continue
            obj.geometry.buildConvexHull(True,'Qt')
            proxy=pin.GeometryObject(obj);proxy.name=obj.name+'_floor_support';proxy.geometry=obj.geometry.convex
            index=self.geometry.addGeometryObject(proxy)
            self.geometry.addCollisionPair(pin.CollisionPair(index,floor))
        self.geometry_data=self.geometry.createData()
        for request in self.geometry_data.collisionRequests:
            request.security_margin=arm_plan.CLEARANCE;request.enable_contact=False

    def clear(self,q):
        return not pin.computeCollisions(self.model,self.data,self.geometry,self.geometry_data,q,True)


class SplineMotion:
    """Quintic B-spline; fixed first/last 3 controls give zero endpoint v/a."""
    def __init__(self,duration=DURATION):
        self.duration=duration
        interior=np.linspace(0,duration,CONTROL_POINTS-DEGREE+1)[1:-1]
        self.knots=np.r_[np.zeros(DEGREE+1),interior,np.full(DEGREE+1,duration)]
        self.spline=BSpline(self.knots,np.eye(CONTROL_POINTS),DEGREE)
        self.free=np.arange(3,CONTROL_POINTS-3)

    def evaluate(self,x,frequency):
        t=np.linspace(0,self.duration,int(round(self.duration*frequency))+1)
        control=np.tile(arm_plan.HOME,(CONTROL_POINTS,1))
        control[self.free]=np.asarray(x).reshape(len(self.free),7)
        return t,*(self.spline(t,nu=k)@control for k in range(3))


class ExcitationProblem(BaseOptimizationProblem):
    def __init__(self,dynamics,basis,screen,seed,previous=None,*,duration=DURATION):
        super().__init__('ArmFullDynamicExcitation')
        self.dynamics=dynamics;self.basis=basis;self.screen=screen
        self.motion=SplineMotion(duration);self.lo,self.hi=arm_plan.planning_bounds(dynamics.model)
        self.previous=previous
        self.nvar=len(self.motion.free)*7
        self.initial=self._initial(seed)
        self.best=self.initial.copy();self.best_value=self._objective(self.initial)
        self.iteration_count=0;self.started=time.monotonic()
        self.best_updates=0

    def _initial(self,seed):
        rng=np.random.default_rng(seed)
        for attempt in range(60):
            controls=rng.uniform(self.lo,self.hi,(len(self.motion.free),7))
            # Keep the first guesses well inside the velocity budget, then
            # let IPOPT expand the independent control-point coordinates.
            scale=1.
            for _ in range(15):
                x=(arm_plan.HOME+scale*(controls-arm_plan.HOME)).reshape(-1)
                _,q,v,a=self.motion.evaluate(x,20.)
                if (np.max(abs(v))<MAX_VELOCITY*.88 and np.max(abs(a))<MAX_ACCELERATION*.88
                        and all(self.screen.clear(p) for p in q[::4])):
                    return x
                scale*=.8
        raise ValueError('不能找到可行激励初值')

    def get_variable_bounds(self):
        return np.tile(self.lo,len(self.motion.free)).tolist(),np.tile(self.hi,len(self.motion.free)).tolist()

    def get_initial_guess(self):return self.initial.tolist()

    def _objective(self,x):
        _,q,v,a=self.motion.evaluate(x,OPT_FREQUENCY)
        W=self.dynamics.matrix(q,v,a)@self.basis
        if self.previous is not None:W=np.vstack([self.previous,W])
        # Fixed reference whitening makes information comparable across runs.
        G=W.T@W+np.eye(self.basis.shape[1])*1e-8
        return -float(np.linalg.slogdet(G)[1])

    def objective(self,x):
        value=self._objective(x)
        if value<self.best_value:
            c=self.constraints(x)
            low,high=self.get_constraint_bounds()
            if np.all(c>=np.asarray(low)-1e-6) and np.all(c<=np.asarray(high)+1e-6):
                self.best=np.asarray(x).copy();self.best_value=value;self.best_updates+=1
        return value

    def gradient(self,x):
        x=np.asarray(x);gradient=np.empty_like(x);value=self._objective(x)
        for i in range(len(x)):
            perturbed=x.copy();perturbed[i]+=1e-5
            gradient[i]=(self._objective(perturbed)-value)/1e-5
        return gradient

    def constraints(self,x):
        _,q,v,a=self.motion.evaluate(x,CONSTRAINT_FREQUENCY)
        distances=[]
        for p in q:
            pin.updateGeometryPlacements(self.dynamics.model,self.screen.data,self.screen.geometry,self.screen.geometry_data,p)
            pin.computeDistances(self.screen.geometry,self.screen.geometry_data)
            distances.extend(result.min_distance for result in self.screen.geometry_data.distanceResults)
        return np.r_[v.reshape(-1),a.reshape(-1),distances]

    def get_constraint_bounds(self):
        n=int(round(self.motion.duration*CONSTRAINT_FREQUENCY))+1
        pairs=len(self.screen.geometry.collisionPairs)
        # Small discretization headroom for the 100 Hz CSV derivative checks.
        return (np.r_[np.full(n*7,-MAX_VELOCITY*.95),np.full(n*7,-MAX_ACCELERATION*.95),np.full(n*pairs,arm_plan.CLEARANCE)].tolist(),
                np.r_[np.full(n*7,MAX_VELOCITY*.95),np.full(n*7,MAX_ACCELERATION*.95),np.full(n*pairs,10.)].tolist())

    def jacobian(self,x):
        x=np.asarray(x);base=self.constraints(x);jac=np.empty((len(base),len(x)))
        for i in range(len(x)):
            p=x.copy();p[i]+=1e-5;jac[:,i]=(self.constraints(p)-base)/1e-5
        return jac.reshape(-1)

    def intermediate(self,*args):
        self.iteration_count+=1
        if self.iteration_count%10==0:
            print(f'  IPOPT 迭代 {self.iteration_count}，当前最好信息目标 {self.best_value:.3f}，耗时 {time.monotonic()-self.started:.1f}s',flush=True)
        return True


def check_motion(model,exact,t,q,v,a):
    lo,hi=arm_plan.planning_bounds(model)
    if np.any(q<lo-1e-8) or np.any(q>hi+1e-8):raise ValueError('生成轨迹超出规划范围')
    if np.max(abs(v))>MAX_VELOCITY or np.max(abs(a))>MAX_ACCELERATION:raise ValueError('生成轨迹速度/加速度超限')
    for p in q:
        if not exact.clear(p):raise ValueError('生成轨迹未通过代理几何/地面全路径复核')


def generate_trajectory(output,name,seed,dynamics,basis,screen,exact,*,problem_factory=ExcitationProblem):
    parts=[];projected=[];segments=[];initials=[]
    for index in range(SEGMENTS):
        print(f'{name}：优化第 {index+1}/{SEGMENTS} 段（不设力矩约束）',flush=True)
        previous=np.vstack(projected) if projected else None
        problem=problem_factory(dynamics,basis,screen,seed+index,previous)
        initials.append(problem.motion.evaluate(problem.initial,OPT_FREQUENCY))
        config=IPOPTConfig.for_trajectory_optimization()
        config.max_iterations=60;config.print_level=0;config.tolerance=1e-3;config.acceptable_tolerance=1e-2
        solver=RobotIPOPTSolver(problem,config)
        success,result=solver.solve()
        candidates=[problem.best]
        if result.get('x_opt') is not None:candidates.append(result['x_opt'])
        candidates.sort(key=problem._objective)
        selected=None
        for candidate in candidates+[problem.initial]:
            t,q,v,a=problem.motion.evaluate(candidate,OUTPUT_FREQUENCY)
            try:
                check_motion(dynamics.model,exact,t,q,v,a)
                if hasattr(problem,'check_candidate'):problem.check_candidate(candidate)
            except ValueError:continue
            selected=(candidate,t,q,v,a);break
        if selected is None:raise ValueError('优化段没有通过整段复核；未发布可执行计划')
        x,t,q,v,a=selected
        ts,qs,vs,aa=problem.motion.evaluate(x,OPT_FREQUENCY)
        projected.append(dynamics.matrix(qs,vs,aa)@basis)
        offset=parts[-1][0][-1] if parts else 0.
        parts.append((t+(offset),q,v,a))
        segments.append({'seed':seed+index,'solver_success':bool(success),'solver_status':int(result.get('status',-999)),
                         'initial_objective':problem._objective(problem.initial),'selected_objective':problem._objective(x),
                         'improved':bool(problem._objective(x)<problem._objective(problem.initial)-1e-6),
                         'control_points':np.asarray(x).reshape(-1,7).tolist(),'iterations':problem.iteration_count,
                         'selection':'best fully checked feasible iterate; no convergence claim for iteration-limit status'})
        print(f'  段复核通过：求解状态 {segments[-1]["solver_status"]}，信息目标 {segments[-1]["initial_objective"]:.2f} → {segments[-1]["selected_objective"]:.2f}',flush=True)
        if hasattr(problem,'on_selected'):problem.on_selected(x)
    arrays=[np.concatenate([p[k] if i==0 else p[k][1:] for i,p in enumerate(parts)]) for k in range(4)]
    t,q,v,a=arrays
    path=output/f'{name}.csv';arm_collect._write_trajectory(path,t,q)
    # Check the actual rounded command CSV too, not just the spline.
    saved=np.genfromtxt(path,delimiter=',',skip_header=1);sv=np.gradient(saved[:,1:],saved[:,0],axis=0,edge_order=2);sa=np.gradient(sv,saved[:,0],axis=0,edge_order=2)
    check_motion(dynamics.model,exact,saved[:,0],saved[:,1:],sv,sa)
    initialW=np.vstack([dynamics.matrix(p[1],p[2],p[3])@basis for p in initials])
    stride=int(round(OUTPUT_FREQUENCY/OPT_FREQUENCY))
    info=information(dynamics.matrix(q[::stride],v[::stride],a[::stride])@basis)
    before=information(initialW)
    if info['rank'] != basis.shape[1]:
        raise ValueError('激励未覆盖完整参考子空间；未发布筛选报告')
    metadata={'kind':'figaroh_full_dynamic_excitation','version':1,'backend':'FIGAROH RobotIPOPTSolver + RegressorBuilder; quintic B-spline adapter',
              'trajectory_sha256':arm_collect._sha256(path),'urdf_sha256':arm_collect._sha256(arm_collect.URDF),
              'duration_s':float(t[-1]),'sample_rate_hz':OUTPUT_FREQUENCY,'seed':seed,'segments':segments,
              'spline_degree':DEGREE,'control_points_per_segment':CONTROL_POINTS,
              'home_rad':arm_plan.HOME.tolist(),
              'joint_lower_rad':arm_plan.planning_bounds(dynamics.model)[0].tolist(),
              'joint_upper_rad':arm_plan.planning_bounds(dynamics.model)[1].tolist(),
              'joint_margin_rad':arm_plan.JOINT_LIMIT_MARGINS.tolist(),
              'maximum_velocity_rad_s':MAX_VELOCITY,'maximum_acceleration_rad_s2':MAX_ACCELERATION,
              'peak_velocity_rad_s':np.max(abs(sv),axis=0).tolist(),'peak_acceleration_rad_s2':np.max(abs(sa),axis=0).tolist(),
              'position_span_rad':np.ptp(q,axis=0).tolist(),'initial_information':before,'information':info,
              'torque_constraint':False,'geometry':'URDF convex proxies for self pairs and floor at 100 Hz, 5 mm margin',
              'objective':'D-optimal log determinant on fixed reference-whitened full regressor; rank and condition reported separately',
              'start_stop':'HOME, zero velocity/acceleration; C2 or better segment continuity'}
    arm_collect._write_metadata(path,metadata)
    # Planning derivatives are reference artifacts, never substitutes for measured q.
    with (output/f'{name}_derivatives.npz').open('wb') as stream:np.savez_compressed(stream,time_s=t,q=q,dq=v,ddq=a)
    return path,metadata


def prepare(output):
    output.mkdir(parents=True,exist_ok=True)
    if any(output.glob('*.csv')) or (output/'screening_report.json').exists():raise ValueError('请使用新的轨迹目录')
    dynamics=DynamicModel();basis,basis_meta=dynamics.reference_basis()
    screen=arm_plan.GeometryScreen(dynamics.model,arm_collect.URDF);exact=ExactGeometry(dynamics.model)
    paths={};reports={}
    for name,seed in [('dynamic',29101),('validation',29301)]:
        path,meta=generate_trajectory(output,name,seed,dynamics,basis,screen,exact)
        paths[name]={'file':path.name,'sha256':arm_collect._sha256(path)};reports[name]=meta
    # Compare with the former 120 s trajectory in exactly the same basis.
    old_args=arm_collect.build_parser().parse_args(['dynamic-plan','--output',str(output/'legacy_multisine_reference.csv'),'--seed','81','--limit-margin',str(arm_plan.LIMIT_MARGIN)])
    arm_collect.make_dynamic_trajectory(old_args)
    old=np.genfromtxt(old_args.output,delimiter=',',skip_header=1)[::20]
    ov=np.gradient(old[:,1:],old[:,0],axis=0,edge_order=2);oa=np.gradient(ov,old[:,0],axis=0,edge_order=2)
    oldinfo=information(dynamics.matrix(old[:,1:],ov,oa)@basis)
    report={'kind':'arm_stage0_screening','version':1,'profile':'figaroh_full_dynamic',
            'urdf_sha256':arm_collect._sha256(arm_collect.URDF),
            'geometry_sha256':arm_collect._sha256(arm_collect.URDF),
            'plans':paths,'basis':basis_meta,'dynamic':reports['dynamic'],'validation':reports['validation'],
            'legacy_multisine_information_same_basis':oldinfo,'torque_constraint':False,
            'limits':'existing tightened J2/J4 and all joint margins; v<=0.25, a<=0.5; no torque constraint',
            'scope':'offline design only; no prediction-accuracy guarantee, hardware actuation, or firmware modification'}
    (output/'screening_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,1,figsize=(13,10),constrained_layout=True)
    with np.load(output/'dynamic_derivatives.npz') as p:
        for ax,key,label in zip(axes,['q','dq','ddq'],['Position (rad)','Velocity (rad/s)','Acceleration (rad/s2)']):
            ax.plot(p['time_s'],p[key]);ax.set(xlabel='Time (s)',ylabel=label);ax.grid(alpha=.25)
        axes[0].legend([f'J{j}' for j in range(1,8)],ncol=7)
    fig.savefig(output/'excitation.png',dpi=150);plt.close(fig)
    (output/'report.md').write_text('\n'.join(['# FIGAROH 完整动力学激励轨迹','',
        f'参考可辨识秩：{basis_meta["rank"]}（含刚体、摩擦、转子惯量）。不设力矩约束。','',
        '| 轨迹 | 时长 s | 秩 | 同基底条件数 | 最小奇异值 |','|---|---:|---:|---:|---:|',
        *[f'| {name} | {reports[name]["duration_s"]:.1f} | {reports[name]["information"]["rank"]} | {reports[name]["information"]["condition_number"]:.1f} | {reports[name]["information"]["minimum_singular_value"]:.5f} |' for name in reports],
        f'| 旧多正弦参考 | 120 | {oldinfo["rank"]} | {oldinfo["condition_number"]:.1f} | {oldinfo["minimum_singular_value"]:.5f} |','',
        '![训练激励](excitation.png)','',
        '所有发布轨迹均完成规划范围、速度、加速度、逐点凸包代理自碰撞与地面检查。迭代上限时仅发布通过完整复核的可行解，不冒称收敛。',
        '轨迹优化指标不能保证实测参数精度；训练、验证必须分别采集。采集使用位控，不增加跟踪误差或反馈越界中断。','']))
    print(f'训练/验证激励轨迹已保存：{output}',flush=True)
    return report


def build_parser():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.set_defaults(handler=lambda a:(prepare(a.output),0)[1]);return p


if __name__=='__main__':
    args=build_parser().parse_args();raise SystemExit(args.handler(args))
