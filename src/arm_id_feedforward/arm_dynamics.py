#!/usr/bin/env python3
"""FIGAROH full dynamic regressor, offline processing and held-out validation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, savgol_filter
from figaroh.tools.regressor import RegressorBuilder, RegressorConfig

if __package__:
    from . import arm_collect, arm_plan
else:
    import arm_collect
    import arm_plan

Q = [f'q{j}_rad' for j in range(1,8)]
TAU = [f'tau{j}_nm' for j in range(1,8)]
CUTOFF_HZ = 4.0
RIDGES = (0., 1e-8, 1e-6, 1e-4, .01, .1)


class DynamicModel:
    def __init__(self, model=None):
        self.model = model if model is not None else arm_collect._load_urdf(arm_collect.URDF)
        self.robot = SimpleNamespace(model=self.model, data=self.model.createData())
        self.builder = RegressorBuilder(self.robot, RegressorConfig(
            has_friction=True, has_actuator_inertia=True, has_joint_offset=False))

    def matrix(self, q, v, a):
        """FIGAROH joint-major rows: all J1 times, then all J2 times, etc."""
        return self.builder.build_basic_regressor(q,v,a,{'act_idxv':list(range(7))})

    def reference_basis(self, seed=20261029):
        rng=np.random.default_rng(seed)
        lo,hi=arm_plan.planning_bounds(self.model)
        q=rng.uniform(lo,hi,(512,7));v=rng.uniform(-.25,.25,(512,7));a=rng.uniform(-.5,.5,(512,7))
        W=self.matrix(q,v,a)
        norms=np.linalg.norm(W,axis=0);active=norms>max(norms.max()*1e-10,1e-12)
        _,s,right=np.linalg.svd(W[:,active]/norms[active],full_matrices=False)
        rank=int(np.sum(s>s[0]*1e-8))
        transform=np.zeros((W.shape[1],rank))
        transform[active]=(right[:rank].T/norms[active,None])/s[:rank]
        return transform,{'seed':seed,'rank':rank,'columns':W.shape[1],
                          'components':['rigid_body_70','viscous_7','coulomb_7','actuator_inertia_7'],
                          'normalization':'fixed independent random-state reference whitening',
                          'rank_relative_tolerance':1e-8,'reference_samples':len(q),
                          'scope':'identifiable combinations, not unique individual physical parameters'}


def information(matrix):
    s=np.linalg.svd(matrix,compute_uv=False)
    rank=int(np.sum(s>s[0]*1e-8)) if len(s) and s[0]>0 else 0
    full=matrix.shape[1]
    return {'rank':rank,'reference_rank':full,
            'condition_number':float(s[0]/s[-1]) if len(s) and s[-1]>0 else None,
            'minimum_singular_value':float(s[-1]) if len(s) else 0.,
            'logdet_regularized':float(np.linalg.slogdet(matrix.T@matrix+np.eye(full)*1e-8)[1])}


def process_capture(path, cutoff=CUTOFF_HZ, *, required_status=15):
    """Process device time, never target motion; do not bridge feedback gaps."""
    df=pd.read_csv(path)
    required=['device_time_us','sample_sequence','status',*Q,*TAU]
    if any(k not in df for k in required):raise ValueError('动态 CSV 缺少反馈列')
    raw_count=len(df)
    valid=np.isfinite(df[Q+TAU+['device_time_us','sample_sequence']].to_numpy(dtype=float)).all(axis=1)
    def status_value(s):
        try:return int(str(s),0)
        except (TypeError,ValueError):return 0
    valid &= (df.status.map(status_value).to_numpy() & required_status)==required_status
    df=df[valid].drop_duplicates(['device_time_us','sample_sequence']).copy()
    metadata=json.loads(path.with_suffix('.json').read_text())
    if metadata.get('urdf_sha256') and metadata['urdf_sha256'] != arm_collect._sha256(arm_collect.URDF):
        raise ValueError('采集时 URDF 与当前辨识模型不一致')
    if 'host_elapsed_s' in df:
        df=df[df.host_elapsed_s>=float(metadata.get('transition_duration_s',0.))].copy()
    t=df.device_time_us.to_numpy(dtype=float)*1e-6
    if len(t)<200:raise ValueError('动态有效数据不足一秒')
    dt=np.diff(t);positive=dt[dt>0]
    if not len(positive):raise ValueError('没有递增的设备时间')
    step=float(np.median(positive));fs=1./step
    if cutoff<=0 or cutoff>=fs*.4:raise ValueError('滤波截止频率不适合实际采样率')
    splits=np.flatnonzero((dt<=0)|(dt>max(.03,step*3)))+1
    segments=np.split(np.arange(len(t)),splits)
    pieces=[];used=[]
    for ids in segments:
        if len(ids)<200 or t[ids[-1]]-t[ids[0]]<3:continue
        grid=np.arange(t[ids[0]],t[ids[-1]],step)
        q=np.column_stack([np.interp(grid,t[ids],df.iloc[ids][key]) for key in Q])
        tau=np.column_stack([np.interp(grid,t[ids],df.iloc[ids][key]) for key in TAU])
        sos=butter(4,cutoff,fs=fs,output='sos')
        q=sosfiltfilt(sos,q,axis=0);tau=sosfiltfilt(sos,tau,axis=0)
        window=max(7,int(round(.15/step))|1)
        v=savgol_filter(q,window,3,deriv=1,delta=step,axis=0)
        a=savgol_filter(q,window,3,deriv=2,delta=step,axis=0)
        edge=max(int(round(1./step)),window)
        stride=max(1,int(round(fs/50)))
        select=slice(edge,len(grid)-edge,stride)
        if len(grid[select])<30:continue
        pieces.append((grid[select],q[select],v[select],a[select],tau[select]))
        used.append({'start_device_s':float(grid[select][0]),'end_device_s':float(grid[select][-1]),'samples':len(grid[select])})
    if not pieces:raise ValueError('没有足够长的连续动态片段；原始数据保留')
    arrays=tuple(np.concatenate([p[k] for p in pieces]) for k in range(5))
    return arrays,{'raw_samples':raw_count,'valid_unique_body_samples':len(df),'processed_samples':len(arrays[0]),
                   'observed_body_device_duration_s':float(t[-1]-t[0]),
                   'device_sample_rate_hz':fs,'cutoff_hz':cutoff,'derivative':'filtered measured q, Savitzky-Golay order 3, 0.15 s',
                   'edge_trim_s':1.,'segments':used,'feedback_gaps_or_resets':len(splits),
                   'fit_row_rule':'exclude abs(measured-derived velocity)<0.01 rad/s per joint from sliding friction fit',
                   'torque_source':'raw_feedback, zero-phase filtered; no calibration or controller subtraction'}


def ridge_fit(X,y,ridge):
    u,s,vt=np.linalg.svd(X,full_matrices=False)
    keep=s>s[0]*1e-10
    factors=np.zeros_like(s);factors[keep]=s[keep]/(s[keep]**2+ridge*s[0]**2)
    return vt.T@(factors*(u.T@y))


def fit_capture(capture, output, validation=None, cutoff=CUTOFF_HZ):
    if output.exists():raise ValueError('请使用新的辨识输出目录')
    dynamics=DynamicModel();basis,basis_meta=dynamics.reference_basis()
    (t,q,v,a,tau),processing=process_capture(capture,cutoff)
    if processing['processed_samples']<200:raise ValueError('处理后动态数据不足')
    W=dynamics.matrix(q,v,a);X=W@basis;y=tau.T.reshape(-1)
    sliding=(abs(v).T.reshape(-1)>=.01)
    row_times=np.tile(t,7)
    edges=np.linspace(t[0],t[-1],6)
    cv=[]
    for ridge in RIDGES:
        pred=np.full_like(y,np.nan)
        for fold in range(5):
            lo,hi=edges[fold:fold+2]
            test=(row_times>=lo)&(row_times<=hi)&sliding
            train=((row_times<lo-1.)|(row_times>hi+1.))&sliding
            if train.sum()<basis.shape[1]*2 or not test.any():continue
            coefficients=ridge_fit(X[train],y[train],ridge)
            pred[test]=X[test]@coefficients
        use=np.isfinite(pred)
        if not use.any():raise ValueError('不能建立带间隔的时间块交叉验证')
        cv.append({'ridge_relative':ridge,'blocked_cv_rmse_nm':float(np.sqrt(np.mean((pred[use]-y[use])**2))),
                   'predicted_samples':int(use.sum())})
    chosen=min(cv,key=lambda r:r['blocked_cv_rmse_nm'])['ridge_relative']
    beta=ridge_fit(X[sliding],y[sliding],chosen);theta=basis@beta
    predicted=(W@theta).reshape(7,-1).T
    # Independent file is evaluated only after training and ridge are frozen.
    validation_result=None;validation_arrays=None
    if validation is not None:
        if capture.resolve()==validation.resolve():raise ValueError('训练与验证必须是不同文件')
        if arm_collect._sha256(capture)==arm_collect._sha256(validation):raise ValueError('验证文件内容与训练数据相同')
        (tv,qv,vv,av,tauv),vp=process_capture(validation,cutoff)
        pv=(dynamics.matrix(qv,vv,av)@theta).reshape(7,-1).T
        validation_result={'capture':str(validation.resolve()),'capture_sha256':arm_collect._sha256(validation),
                           'processing':vp,'rmse_nm':np.sqrt(np.mean((pv-tauv)**2,axis=0)).tolist(),
                           'bias_nm':np.mean(pv-tauv,axis=0).tolist(),
                           'peak_abs_error_nm':np.max(abs(pv-tauv),axis=0).tolist()}
        validation_arrays=(tv,tauv,pv)
    report={'kind':'arm_full_dynamic_identification','version':1,'backend':'FIGAROH RegressorBuilder / Pinocchio',
            'capture':str(capture.resolve()),'capture_sha256':arm_collect._sha256(capture),
            'urdf_sha256':arm_collect._sha256(arm_collect.URDF),'processing':processing,'basis':basis_meta,
            'observed_information':information(X[sliding]),'ridge_selection':cv,'chosen_ridge':chosen,
            'training_rmse_nm':np.sqrt(np.mean((predicted-tau)**2,axis=0)).tolist(),
            'training_sliding_rmse_nm':[float(np.sqrt(np.mean((predicted[abs(v[:,j])>=.01,j]-tau[abs(v[:,j])>=.01,j])**2))) if np.any(abs(v[:,j])>=.01) else None for j in range(7)],
            'validation':validation_result,'independent_validation_available':validation is not None,
            'verdict':'独立预测误差已报告，尚需实机验收' if validation is not None else '仅训练与时间块内部验证；缺少独立采集，不作为替换算法验收',
            'scope':'identifiable dynamic combinations; individual inertias/rotor/friction may be coupled; no physical-feasibility reconstruction or firmware export'}
    output.mkdir(parents=True)
    (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    (output/'dynamic_model.json').write_text(json.dumps({'kind':report['kind'],'version':1,'urdf_sha256':report['urdf_sha256'],
        'parameter_order':'70 Pinocchio m,hx,hy,hz,Ixx,Ixy,Iyy,Ixz,Iyz,Izz; fv J1..J7; fc J1..J7; rotor J1..J7',
        'parameters':theta.tolist(),'base_coefficients':beta.tolist(),'base_transform':basis.tolist(),
        'basis':basis_meta,'scope':report['scope']},indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    def plot(times,observed,pred,path,title):
        fig,axes=plt.subplots(4,2,figsize=(13,12),constrained_layout=True)
        sl=slice(None,None,5)
        for j,ax in enumerate(axes.flat[:7]):
            ax.plot(times[sl]-times[0],observed[sl,j],label='Filtered feedback')
            ax.plot(times[sl]-times[0],pred[sl,j],label='Full dynamic prediction')
            ax.set(title=f'J{j+1}',xlabel='Device time (s)',ylabel='N m');ax.grid(alpha=.25)
        axes.flat[0].legend(fontsize=8);axes.flat[7].axis('off');fig.suptitle(title)
        fig.savefig(path,dpi=150);plt.close(fig)
    plot(t,tau,predicted,output/'training.png','Training only: full dynamics + friction + rotor')
    if validation_arrays:plot(*validation_arrays,output/'validation.png','Independent validation, frozen coefficients')
    rows={'device_time_s':t}
    for j in range(7):
        rows.update({f'q{j+1}_rad':q[:,j],f'dq{j+1}_rad_s':v[:,j],f'ddq{j+1}_rad_s2':a[:,j],f'tau{j+1}_nm':tau[:,j],f'predicted_tau{j+1}_nm':predicted[:,j]})
    pd.DataFrame(rows).to_csv(output/'processed.csv',index=False)
    lines=['# 完整动态辨识', '',report['verdict'],'',
           f'回归矩阵：91 列；参考可辨识秩 {basis_meta["rank"]}；实测运动的观测秩 {report["observed_information"]["rank"]}。',
           '', '输入仅为实际位置、设备时间和原始反馈力矩；速度加速度由滤波后的实际位置获得。未扣除控制器力矩。', '',
           '| 轴 | 训练 RMSE（N·m） | 独立验证 RMSE（N·m） |','|---|---:|---:|']
    for j in range(7):
        value=f'{validation_result["rmse_nm"][j]:.4f}' if validation_result else '未采集'
        lines.append(f'| J{j+1} | {report["training_rmse_nm"][j]:.4f} | {value} |')
    lines+=['','![训练](training.png)','',report['scope'],'']
    if validation_result:lines+=['![独立验证](validation.png)','']
    (output/'report.md').write_text('\n'.join(lines))
    print(f'完整动态求解已保存：{output / "report.md"}',flush=True)
    print(report['verdict'],flush=True)
    return report


def build_parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('capture',type=Path);p.add_argument('--validation',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.set_defaults(handler=lambda a: (fit_capture(a.capture,a.output,a.validation),0)[1])
    return p


if __name__=='__main__':
    a=build_parser().parse_args();raise SystemExit(a.handler(a))
