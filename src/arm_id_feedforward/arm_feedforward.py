#!/usr/bin/env python3
"""Shared-model full or gravity feedforward: prepare, preview, run, analyze."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import pandas as pd
from scipy.interpolate import BSpline

if __package__:
    from . import arm_collect, arm_datasets, arm_host, arm_plan, arm_workflow
    from .arm_analysis import analyze_capture
    from .regressor import DynamicRegressor
else:
    import arm_collect, arm_datasets, arm_host, arm_plan, arm_workflow
    from arm_analysis import analyze_capture
    from regressor import DynamicRegressor

HOST = Path(__file__).resolve().parent
DEFAULT_MODEL = HOST/'models/identified/shared_full_model.json'
DEFAULT_TRAJECTORY = HOST/'models/controller/validation_plan.csv'
RATE = 200.
TORQUE_MODES = ('full', 'gravity')
Q = [f'q{j}_rad' for j in range(1, 8)]
V = [f'dq{j}_rad_s' for j in range(1, 8)]
A = [f'ddq{j}_rad_s2' for j in range(1, 8)]
T = [f'tau{j}_nm' for j in range(1, 8)]
TARGET_COLUMNS = ([f'target_{c}' for c in Q+V+A] +
                  [f'ff_tau{j}_nm' for j in range(1, 8)])


def quintic_spline(duration, home, free_controls):
    """Rebuild the 11-control spline shared by excitation and validation plans."""
    knots = np.r_[np.zeros(6), np.linspace(0, duration, 7)[1:-1],
                  np.full(6, duration)]
    controls = np.tile(np.asarray(home, dtype=float), (11, 1))
    controls[3:8] = np.asarray(free_controls, dtype=float)
    return BSpline(knots, controls, 5)


def write_json(path, document):
    arm_workflow.write_json(path, document)


class SharedModel:
    """Evaluate the saved 91 coefficients with the identification regressor.

    No physical inertia reconstruction, A/B gains, old J3 trim or URDF dynamic
    coefficients enter the prediction. URDF supplies kinematics and gravity.
    """
    def __init__(self, path=DEFAULT_MODEL):
        self.path = Path(path)
        self.document = json.loads(self.path.read_text())
        doc = self.document
        if (doc.get('kind') != 'arm_full_dynamic_identification' or doc.get('version') != 1
                or doc.get('urdf_sha256') != arm_collect._sha256(arm_collect.URDF)
                or doc.get('parameter_order') !=
                '70 Pinocchio m,hx,hy,hz,Ixx,Ixy,Iyy,Ixz,Iyz,Izz; fv J1..J7; fc J1..J7; rotor J1..J7'):
            raise ValueError('需要与当前 URDF 和参数顺序一致的完整共享模型')
        self.theta = np.asarray(doc['parameters'], dtype=float)
        if self.theta.shape != (91,) or not np.isfinite(self.theta).all():
            raise ValueError('共享模型需要 91 个有限系数')
        source = doc.get('provenance', {})
        if source.get('selected_parameter_field'):
            if source['selected_parameter_field'] != 'selection.shared_parameters':
                raise ValueError('本入口只执行共享模型，不接受 A/B 候选')
            saved = arm_datasets.resolve_provenance(source['source_model'], source['source_model_sha256'])
            expected = json.loads(saved.read_text())['selection']['shared_parameters']
            if not np.array_equal(self.theta, expected):
                raise ValueError('共享参数与冻结来源不一致')
        self.dynamics = DynamicRegressor()

    def predict(self, q, v, a, *, components=False):
        q, v, a = (np.asarray(x, dtype=float) for x in (q, v, a))
        if (q.ndim != 2 or q.shape[1] != 7 or q.shape != v.shape or q.shape != a.shape
                or not np.isfinite(np.concatenate((q, v, a))).all()):
            raise ValueError('逆动力学输入应为三个同形状 N×7 有限数组')
        result = []
        gravity = []
        # Keep the 7*N by 91 regressor bounded, even for long trajectories.
        for start in range(0, len(q), 256):
            qq, vv, aa = (x[start:start+256] for x in (q, v, a))
            result.append((self.dynamics.matrix(qq, vv, aa) @ self.theta).reshape(7, -1).T)
            if components:
                z = np.zeros_like(qq)
                gravity.append((self.dynamics.matrix(qq, z, z) @ self.theta).reshape(7, -1).T)
        total = np.concatenate(result)
        if not np.isfinite(total).all():
            raise ValueError('逆动力学输出包含非有限值')
        if not components:
            return total
        terms = dict(gravity=np.concatenate(gravity), viscous=v*self.theta[70:77],
                     coulomb=np.sign(v)*self.theta[77:84], rotor=a*self.theta[84:91])
        terms['rigid_motion'] = total-sum(terms.values())
        return total, terms


def predict_torque(model, q, v, a, mode):
    if mode == 'full':
        return model.predict(q, v, a)
    if mode == 'gravity':
        zero = np.zeros_like(q)
        return model.predict(q, zero, zero)
    raise ValueError(f'未知前馈类型：{mode}')


class Reference:
    """Recover analytic quintic reference from saved spline controls."""
    def __init__(self, path):
        self.path = Path(path)
        self.meta = json.loads(self.path.with_suffix('.json').read_text())
        m = self.meta
        if (m.get('kind') not in ('figaroh_full_dynamic_excitation', 'arm_validation_spline')
                or m.get('version') != 1
                or m.get('trajectory_sha256') != arm_collect._sha256(self.path)
                or m.get('urdf_sha256') != arm_collect._sha256(arm_collect.URDF)
                or m.get('spline_degree') != 5 or m.get('control_points_per_segment') != 11):
            raise ValueError('轨迹需为完整且哈希匹配的五次样条计划')
        self.duration = float(m['duration_s'])
        self.segment_duration = self.duration/len(m['segments'])
        if not np.isfinite(self.duration) or self.duration <= 0:
            raise ValueError('轨迹时长无效')
        if not np.isclose(m.get('segment_duration_s', self.segment_duration), self.segment_duration):
            raise ValueError('样条段时长不一致')
        self.splines = []
        for segment in m['segments']:
            free_controls = np.asarray(segment['control_points'], dtype=float)
            if free_controls.shape != (5, 7) or not np.isfinite(free_controls).all():
                raise ValueError('轨迹控制点包含非有限值')
            self.splines.append(quintic_spline(self.segment_duration,
                                               m['home_rad'], free_controls))
        saved = pd.read_csv(path)
        times = saved.time_s.to_numpy()
        if (len(times) < 2 or times[0] != 0 or not np.isclose(times[-1], self.duration)
                or not np.isfinite(times).all() or np.any(np.diff(times) <= 0)):
            raise ValueError('轨迹时间轴无效')
        if not np.allclose(self.evaluate(times)[0], saved[[f'q{j}' for j in range(1, 8)]], atol=5.1e-9, rtol=0):
            raise ValueError('解析样条与保存的实际轨迹不一致')

    def evaluate(self, times):
        times = np.asarray(times)
        indices = np.minimum((times/self.segment_duration).astype(int), len(self.splines)-1)
        arrays = [np.zeros((len(times), 7)) for _ in range(3)]
        for i, spline in enumerate(self.splines):
            mask = indices == i
            for derivative, array in enumerate(arrays):
                array[mask] = spline(times[mask]-i*self.segment_duration, nu=derivative)
        # Exact rest at joins/endpoints; avoid a floating-point Coulomb sign.
        for array in arrays[1:]:
            array[abs(array) < 1e-12] = 0.
        return arrays


def transition(start, end, duration=24.):
    delta = np.asarray(end)-np.asarray(start)
    duration = max(duration, 1.15*np.max(1.875*abs(delta)/.25),
                   1.15*np.max(np.sqrt(5.774*abs(delta)/.5)))
    duration = np.ceil(duration*RATE)/RATE
    t = np.arange(round(duration*RATE)+1)/RATE
    s = t/duration
    q = start + (10*s**3-15*s**4+6*s**5)[:, None]*delta
    v = ((30*s**2-60*s**3+30*s**4)/duration)[:, None]*delta
    a = ((60*s-180*s**2+120*s**3)/duration**2)[:, None]*delta
    return t, q, v, a


def scaled_reference(reference, speed):
    if not np.isfinite(speed) or speed <= 0:
        raise ValueError('轨迹倍速必须为正的有限值')
    duration = reference.duration/speed
    t = np.arange(0., duration, 1/RATE)
    # Preserve the exact final pose/rest even when duration is not a 5 ms multiple.
    if len(t) and np.isclose(t[-1], duration, atol=1e-12, rtol=0):
        t[-1] = duration
    else:
        t = np.r_[t, duration]
    q, v, a = reference.evaluate(np.minimum(t*speed, reference.duration))
    return t, q, v*speed, a*speed**2


def check_path(model, t, q, v, a, *, entry=False, speed=1.):
    # All checks happen before sending control commands. Actual feedback is
    # never rejected for position error or crossing a planning boundary.
    if not all(np.isfinite(x).all() for x in (t, q, v, a)):
        raise ValueError('生成轨迹含非有限数值')
    lo, hi = arm_plan.planning_bounds(model)
    if not entry and (np.any(q < lo-1e-8) or np.any(q > hi+1e-8)):
        raise ValueError('生成轨迹超出当前含余量的规划范围')
    if np.max(abs(v)) > .25*speed+1e-8 or np.max(abs(a)) > .5*speed**2+1e-8:
        raise ValueError('生成轨迹速度/加速度超出规划预算')
    if __package__:
        from .arm_excitation import ExactGeometry
    else:
        from arm_excitation import ExactGeometry
    geometry = ExactGeometry(model)
    # At 1.8x use every 200 Hz point, preserving source-path sampling density.
    for pose in q[::max(1, int(2/speed))]:
        if not geometry.clear(pose):
            raise ValueError('生成路径未通过代理几何自碰撞/地面检查，尚未执行')


def prepare(output, trajectory=DEFAULT_TRAJECTORY, model_path=DEFAULT_MODEL,
            speed=None, torque_mode=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = output/'session.json'
    if manifest.exists():
        doc = json.loads(manifest.read_text())
        if doc.get('kind') != 'arm_shared_feedforward' or doc.get('version') != 1:
            raise ValueError('不是受支持的完整前馈会话')
        if speed is not None and (not np.isfinite(speed) or speed <= 0
                                  or not np.isclose(speed, doc.get('speed', 1.), rtol=0, atol=1e-12)):
            raise ValueError('本会话倍速已冻结；更改倍速请使用新输出目录')
        if torque_mode is not None and torque_mode != doc.get('torque_mode', 'full'):
            raise ValueError('本会话前馈类型已冻结；更改类型请使用新输出目录')
        for name, expected in doc['files'].items():
            if arm_collect._sha256(output/name) != expected:
                raise ValueError(f'会话文件已变化：{name}')
        return doc
    if any(output.iterdir()):
        raise ValueError('新会话目录必须为空，避免覆盖已有数据')
    reference = Reference(trajectory)
    model = SharedModel(model_path)
    speed = 1. if speed is None else speed
    torque_mode = 'full' if torque_mode is None else torque_mode
    if torque_mode not in TORQUE_MODES:
        raise ValueError(f'未知前馈类型：{torque_mode}')
    t, q, v, a = scaled_reference(reference, speed)
    check_path(model.dynamics.model, t, q, v, a, speed=speed)
    full_torque, terms = model.predict(q, v, a, components=True)
    tau = full_torque if torque_mode == 'full' else terms['gravity']
    shutil.copyfile(trajectory, output/'source_plan.csv')
    shutil.copyfile(Path(trajectory).with_suffix('.json'), output/'source_plan.json')
    if speed == 1.:
        shutil.copyfile(trajectory, output/'plan.csv')
        shutil.copyfile(Path(trajectory).with_suffix('.json'), output/'plan.json')
    else:
        arm_collect._write_trajectory(output/'plan.csv', t, q)
        retimed = dict(reference.meta, duration_s=float(t[-1]),
                       segment_duration_s=reference.segment_duration/speed, speed=speed,
                       source_trajectory_sha256=arm_collect._sha256(trajectory),
                       trajectory_sha256=arm_collect._sha256(output/'plan.csv'),
                       sample_rate_hz=RATE, maximum_velocity_rad_s=.25*speed,
                       maximum_acceleration_rad_s2=.5*speed**2,
                       peak_velocity_rad_s=np.max(abs(v), axis=0).tolist(),
                       peak_acceleration_rad_s2=np.max(abs(a), axis=0).tolist(),
                       time_scaling=f'q(s*t), dq=s*dq_source, ddq=s^2*ddq_source; recomputed {torque_mode} torque')
        write_json(output/'plan.json', retimed)
    shutil.copyfile(model_path, output/'model.json')
    np.savez_compressed(output/'body_reference.npz', t=t, q=q, v=v, a=a, tau=tau)
    report = dict(kind='arm_shared_feedforward_plan', version=1, rate_hz=RATE,
                  duration_s=float(t[-1]), source_duration_s=reference.duration,
                  speed=speed, torque_mode=torque_mode,
                  planning_margin_rad=arm_plan.JOINT_LIMIT_MARGINS.tolist(),
                  torque_constraints=False, geometry_check_hz=RATE/max(1, int(2/speed)),
                  maximum_velocity_rad_s=.25*speed, maximum_acceleration_rad_s2=.5*speed**2,
                  peak_velocity_rad_s=np.max(abs(v), axis=0).tolist(),
                  peak_acceleration_rad_s2=np.max(abs(a), axis=0).tolist(),
                  reference='analytic original quintic B-spline, sampled at 200 Hz',
                  parameters=('all 91 shared coefficients' if torque_mode == 'full'
                              else 'shared rigid-body gravity Y(q,0,0); no motion or friction terms'),
                  peak_abs_torque=np.max(abs(tau), axis=0).tolist(),
                  peak_abs_full_torque=np.max(abs(full_torque), axis=0).tolist(),
                  peak_abs_omitted_torque=np.max(abs(full_torque-tau), axis=0).tolist(),
                  peak_abs_components={key: np.max(abs(value), axis=0).tolist() for key, value in terms.items()})
    write_json(output/'planning_report.json', report)
    doc = dict(kind='arm_shared_feedforward', version=1, phase='prepared',
               speed=speed, torque_mode=torque_mode,
               source_trajectory=str(Path(trajectory).resolve()), source_model=str(Path(model_path).resolve()),
               files={name: arm_collect._sha256(output/name) for name in
                      ('source_plan.csv', 'source_plan.json', 'plan.csv', 'plan.json',
                       'model.json', 'body_reference.npz', 'planning_report.json')})
    write_json(manifest, doc)
    return doc


def execution_reference(output, start):
    with np.load(output/'body_reference.npz') as saved:
        body = {name: saved[name].copy() for name in ('t', 'q', 'v', 'a', 'tau')}
    t, q, v, a = transition(np.asarray(start), body['q'][0])
    model = SharedModel(output/'model.json')
    check_path(model.dynamics.model, t, q, v, a, entry=True)
    mode = json.loads((output/'session.json').read_text()).get('torque_mode', 'full')
    entry = dict(t=t, q=q, v=v, a=a,
                 tau=predict_torque(model, q, v, a, mode))
    result = {name: np.concatenate((entry[name], body[name][1:]+t[-1] if name == 't'
                                    else body[name][1:])) for name in entry}
    return result, float(t[-1])


def save_reference(output, reference, entry):
    frame = {'time_s': reference['t']}
    for key, columns in [('q', Q), ('v', V), ('a', A), ('tau', T)]:
        frame.update({name: reference[key][:, j] for j, name in enumerate(columns)})
    pd.DataFrame(frame).to_csv(output/'reference.csv', index=False)
    # Pure q playback of the SAME entry/body reference, no extra viewer entry.
    arm_collect._write_trajectory(output/'preview.csv', reference['t'], reference['q'])


def stream_targets(client, reference, entry_duration, output):
    """No model work or file-wide loading on the 200 Hz command path."""
    references = {}
    started = time.monotonic()
    motion_start = None
    handshake_duration = None
    last_sample = started
    missing = False
    complete = False
    error = None
    session = json.loads((output/'session.json').read_text()) if (output/'session.json').exists() else {}
    metadata = dict(stage='dynamic', capture_mode='inverse_dynamics', torque_source='raw_feedback',
                    feedforward_torque_mode=session.get('torque_mode', 'full'),
                    urdf_sha256=arm_collect._sha256(arm_collect.URDF),
                    model_sha256=arm_collect._sha256(output/'model.json'),
                    trajectory_sha256=arm_collect._sha256(output/'plan.csv'),
                    reference_sha256=arm_collect._sha256(output/'reference.csv'),
                    command_rate_hz=RATE, transition_duration_s=entry_duration,
                    speed=session.get('speed', 1.),
                    start_position_rad=reference['q'][0].tolist())
    try:
        with (output/'capture.csv').open('x', newline='') as captures, \
             (output/'commands.csv').open('x', newline='') as commands:
            writer = csv.writer(captures)
            sent = csv.writer(commands)
            writer.writerow(['host_elapsed_s', 'device_time_us', 'sample_sequence',
                             'target_sequence', 'status', 'command_elapsed_s',
                             'reference_time_s', 'phase', *TARGET_COLUMNS, *Q, *V, *T])
            sent.writerow(['command_elapsed_s', 'target_sequence', 'reference_time_s',
                           'phase', *TARGET_COLUMNS])
            next_send = started
            next_flush = started+1
            next_progress = started+10
            while True:
                now = time.monotonic()
                if now >= next_send:
                    elapsed = max(0., now-motion_start) if motion_start is not None else 0.
                    index = max(0, int(np.searchsorted(reference['t'], elapsed, side='right'))-1)
                    phase = ('handshake' if motion_start is None else
                             'entry' if elapsed < entry_duration else 'body')
                    q, v, a, tau = (reference[key][index] for key in ('q', 'v', 'a', 'tau'))
                    seq = client.send_feedforward(q, v, tau)
                    record = (now-started, float(reference['t'][index]), phase, *q, *v, *a, *tau)
                    references[seq] = record
                    sent.writerow([record[0], seq, *record[1:]])
                    # Skip elapsed slots; never burst a backlog of stale targets.
                    next_send = started+(int((now-started)*RATE)+1)/RATE
                for message in client.read_messages():
                    if isinstance(message, arm_host.Ack):
                        if message.command_sequence not in references:
                            continue
                        if not arm_host.check_command_result(client, message):
                            continue
                        supported = message.active_mode == arm_host.MODE_INVERSE_DYNAMICS
                    else:
                        record = references.get(message.target_sequence)
                        writer.writerow([time.monotonic()-started, message.device_time_us,
                                         message.sample_sequence, message.target_sequence,
                                         f'0x{message.status:04x}',
                                         *(record or (['']*(3+28))),
                                         *message.position, *message.velocity, *message.torque])
                        supported = bool(record and (message.status & 31) == 31)
                        last_sample = time.monotonic()
                        if missing:
                            print('警告：反馈已恢复，继续记录。', flush=True)
                            missing = False
                    if motion_start is None and supported:
                        motion_start = time.monotonic()
                        handshake_duration = motion_start-started
                        print('模式 3 已确认，开始平滑入场和主体轨迹。', flush=True)
                now = time.monotonic()
                if motion_start is None and now-started > 2.:
                    raise RuntimeError('未收到模式 3 的确认或有效状态；未推进轨迹，请确认固件已更新')
                if now-last_sample > 1. and not missing:
                    print('警告：反馈中断超过 1 秒，仍继续发送轨迹；缺口会写入报告。', flush=True)
                    missing = True
                if now >= next_flush:
                    captures.flush(); commands.flush(); next_flush = now+1
                if now >= next_progress and motion_start is not None:
                    print(f'正在执行：{min(now-motion_start, reference["t"][-1]):.0f}/{reference["t"][-1]:.0f} 秒；指令与反馈持续保存。', flush=True)
                    next_progress = now+10
                if motion_start is not None and now-motion_start >= reference['t'][-1]+.1:
                    complete = True
                    break
                time.sleep(max(0., min(.001, next_send-time.monotonic())))
    except BaseException as exception:
        error = f'{type(exception).__name__}: {exception}'
        raise
    finally:
        # Metadata is written even on Ctrl+C, handshake failure or USB errors.
        metadata.update(complete=complete, error=error, commands_sent=len(references),
                        handshake_duration_s=handshake_duration,
                        transition_duration_s=entry_duration+(handshake_duration or 0.),
                        host_duration_s=time.monotonic()-started)
        write_json(output/'capture.json', metadata)


def plot_joint_errors(output, group, model_trace):
    """One readable time series per joint; model residual uses measured states."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = ['DejaVu Sans', 'AR PL UMing CN']
    plt.rcParams['axes.unicode_minus'] = False

    plotted = group.iloc[::max(1, len(group)//3000)]
    start_device_s = float(group.device_time_us.iloc[0])*1e-6
    start_reference_s = float(group.reference_time_s.iloc[0])
    x = plotted.reference_time_s.to_numpy()-start_reference_s
    for joint in range(1, 8):
        q_error = np.degrees(plotted[f'q{joint}_rad'].to_numpy()-
                             plotted[f'target_q{joint}_rad'].to_numpy())
        velocity_error = (plotted[f'dq{joint}_rad_s'].to_numpy()-
                          plotted[f'target_dq{joint}_rad_s'].to_numpy())
        feedforward_gap = (plotted[f'tau{joint}_nm'].to_numpy()-
                           plotted[f'ff_tau{joint}_nm'].to_numpy())
        count = 4 if model_trace is not None else 3
        fig, axes = plt.subplots(count, 1, figsize=(12, 2.45*count),
                                 sharex=True, constrained_layout=True)
        traces = ((q_error, '实测−目标位置 (°)', '#2369a8'),
                  (velocity_error, '反馈−目标速度 (rad/s)', '#bd6419'),
                  (feedforward_gap, '反馈力矩−前馈 (反馈单位)', '#7153a0'))
        for ax, (values, label, color) in zip(axes, traces):
            ax.plot(x, values, color=color, linewidth=.8)
            ax.axhline(0, color='0.3', linewidth=.7)
            ax.set_ylabel(label)
            ax.grid(alpha=.25)
        if model_trace is not None:
            device_times, residual = model_trace
            axes[3].plot(device_times-start_device_s, residual[:, joint-1],
                         color='#2b8066', linewidth=.8)
            axes[3].axhline(0, color='0.3', linewidth=.7)
            axes[3].set_ylabel('模型预测−滤波反馈 (反馈单位)')
            axes[3].grid(alpha=.25)
        axes[-1].set_xlabel('主体轨迹时间 (s)')
        axes[-1].set_xlim(0, max(0.01, float(group.reference_time_s.iloc[-1]-start_reference_s)))
        fig.suptitle(f'J{joint} 误差时间序列；力矩差包含电机 PD，末图使用真实状态')
        fig.savefig(output/f'errors_j{joint}.png', dpi=155)
        plt.close(fig)


def analyze(output):
    session = json.loads((output/'session.json').read_text()) if (output/'session.json').exists() else {}
    torque_mode = session.get('torque_mode', 'full')
    quality = analyze_capture(output/'capture.csv', 'dynamic', output/'plan.csv')
    df = pd.read_csv(output/'capture.csv')
    status = df.status.map(lambda x: int(str(x), 0)).to_numpy()
    if np.any((status & 31) != 31):
        quality['issues'].append('存在非完整前馈模式或未全部入队的反馈')
        quality['passed'] = False
    write_json(output/'quality_report.json', quality)
    finite = np.isfinite(df[Q+V+T+TARGET_COLUMNS].to_numpy(dtype=float)).all(axis=1)
    matched = finite & ((status & 31) == 31) & (df.phase == 'body')
    group = df.loc[matched].drop_duplicates(['device_time_us', 'sample_sequence'])
    report = dict(kind='arm_shared_feedforward_analysis', version=1,
                  complete=json.loads((output/'capture.json').read_text()).get('complete', False),
                  quality_issues=quality['issues'], matched_body_samples=len(group),
                  feedforward_torque_mode=torque_mode,
                  model_sha256=arm_collect._sha256(output/'model.json'),
                  comparison_scope='single run; no claim of improvement without same-path baseline',
                  torque_scope='feedback-minus-FF contains MIT PD and tracking effects, not pure model error')
    model_error = None
    model_trace = None
    if len(group):
        def metrics(error):
            return dict(rmse=np.sqrt(np.mean(error**2, axis=0)).tolist(),
                        bias=np.mean(error, axis=0).tolist(), peak_abs=np.max(abs(error), axis=0).tolist())
        report['position_error_rad'] = metrics(group[Q].to_numpy()-group[[f'target_{x}' for x in Q]].to_numpy())
        report['velocity_error_rad_s'] = metrics(group[V].to_numpy()-group[[f'target_{x}' for x in V]].to_numpy())
        report['feedback_minus_feedforward'] = metrics(group[T].to_numpy()-group[[f'ff_tau{j}_nm' for j in range(1, 8)]].to_numpy())
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(4, 2, figsize=(13, 11), constrained_layout=True)
        for j, ax in enumerate(axes.flat[:7], 1):
            ax.plot(group.reference_time_s, group[f'target_q{j}_rad'], label='Reference')
            ax.plot(group.reference_time_s, group[f'q{j}_rad'], label='Measured')
            ax.set(title=f'J{j}', xlabel='Reference time (s)', ylabel='rad'); ax.grid(alpha=.3)
        axes.flat[0].legend(); axes.flat[7].axis('off')
        fig.savefig(output/'tracking.png', dpi=140); plt.close(fig)
    try:
        if __package__:
            from . import arm_dynamics
        else:
            import arm_dynamics
        (processed_time, q, v, a, tau), processing = arm_dynamics.process_capture(
            output/'capture.csv', required_status=31)
        prediction = SharedModel(output/'model.json').predict(q, v, a)
        report['actual_state_prediction'] = dict(
            processing=processing, rmse=np.sqrt(np.mean((prediction-tau)**2, axis=0)).tolist(),
            bias=np.mean(prediction-tau, axis=0).tolist(),
            scope='frozen model evaluated on filtered MEASURED q/dq/ddq, not commanded motion')
        model_error = report['actual_state_prediction']['rmse']
        model_trace = (processed_time, prediction-tau)
    except ValueError as error:
        report['actual_state_prediction_unavailable'] = str(error)
    if len(group):
        plot_joint_errors(output, group, model_trace)
        report['joint_error_plots'] = [f'errors_j{joint}.png' for joint in range(1, 8)]
    write_json(output/'report.json', report)
    label = '完整动力学' if torque_mode == 'full' else '仅重力'
    lines = [f'# 共享模型{label}前馈试验', '',
             f'执行完整：{report["complete"]}；配对的主体样本：{len(group)}。', '',
             f'发送力矩：{label}前馈；位置、速度目标及电机 PD 与完整前馈试验相同。', '',
             '只做本轮跟踪与模型残差评估；未与同轨迹基线对比，不据此宣称优劣。', '',
             '反馈力矩减前馈包含电机 PD 和实际跟踪差异，不是纯模型误差。', '',
             '质量提示：'+'；'.join(quality['issues'] or ['无']), '']
    if len(group):
        lines += ['| 轴 | 位置 RMSE（度） | 速度 RMSE（rad/s） | 实际状态模型 RMSE（反馈力矩单位） | 误差图 |',
                  '|---|---:|---:|---:|---|']
        for j in range(7):
            value = f'{model_error[j]:.4f}' if model_error is not None else '不可用'
            lines.append(f'| J{j+1} | {np.degrees(report["position_error_rad"]["rmse"][j]):.3f} | {report["velocity_error_rad_s"]["rmse"][j]:.4f} | {value} | [J{j+1}](errors_j{j+1}.png) |')
        lines += ['', '![跟踪](tracking.png)', '']
    (output/'report.md').write_text('\n'.join(lines))
    return report


def run(args):
    output = args.output.resolve()
    if args.analyze_only:
        analyze(output); print(f'分析报告：{output/"report.md"}'); return 0
    if not (output/'session.json').exists():
        label = '仅重力' if args.torque_mode == 'gravity' else '完整动力学'
        speed = 1. if args.speed is None else args.speed
        arm_workflow.confirm(f'生成{label}前馈计划？轨迹 {args.trajectory.name}，'
                             f'主体 {speed:g} 倍速；检查规划范围和几何')
    doc = prepare(output, args.trajectory, args.model,
                  speed=args.speed, torque_mode=args.torque_mode)
    label = '仅重力' if doc.get('torque_mode', 'full') == 'gravity' else '完整动力学'
    print(f'计划及模型已就绪：{output}；轨迹 {Path(doc.get("source_trajectory", args.trajectory)).name}，'
          f'主体 {doc.get("speed", 1.):g} 倍速，'
          f'200 Hz，{label}前馈；位置、速度目标与电机 PD 不变。')
    if args.prepare_only:
        return 0
    if (output/'capture.csv').exists():
        print('已有采集文件，不覆盖、不自动重跑运动。')
    else:
        arm_workflow.confirm('打开 2.5 倍速预览？只读取当前姿态')
        port_name = arm_workflow.choose_port(args.port)
        with arm_host.open_port(port_name, 115200) as port:
            client = arm_host.ArmClient(port)
            initial = arm_host.read_current_state(client, 2.)
            preview, entry = execution_reference(output, initial.position)
            # A resumed, unexecuted session may have an older preview.
            (output/'preview.csv').unlink(missing_ok=True)
            save_reference(output, preview, entry)
            if __package__:
                from . import arm_sim
            else:
                import arm_sim
            speed = float(doc.get('speed', 1.))
            arm_workflow.run_tool(arm_sim, [
                'playback', str(output/'preview.csv'), '--speed', '2.5',
                '--maximum-velocity', str(.25*speed),
                '--maximum-acceleration', str(.5*speed**2),
            ], output/'details.log')
            print('预览结束。执行前会重读姿态，平滑入场；跟踪误差不触发中断。')
            arm_workflow.confirm(f'开始{label}前馈运动并采集？', motion=True)
            initial = arm_host.read_current_state(client, 2.)
            print(f'正在按当前姿态生成入场并计算{label}前馈，尚未运动…', flush=True)
            reference, entry = execution_reference(output, initial.position)
            (output/'preview.csv').unlink(missing_ok=True)
            save_reference(output, reference, entry)
            doc['phase'] = 'executing'; write_json(output/'session.json', doc)
            print('开始实机执行和采集。Ctrl+C 可中止并请求重力拖动。', flush=True)
            try:
                with arm_collect._gravity_on_exit(client):
                    stream_targets(client, reference, entry, output)
            finally:
                doc['phase'] = 'captured' if (output/'capture.csv').exists() else 'prepared'
                write_json(output/'session.json', doc)
        print('采集结束，原始反馈、发送指令和元数据已保存。')
    arm_workflow.confirm('开始跟踪和实际状态模型分析？详细数据只写入文件')
    analyze(output)
    doc['phase'] = 'analyzed'; write_json(output/'session.json', doc)
    print(f'分析报告：{output/"report.md"}')
    return 0


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', help='省略时交互选择 USB 串口')
    parser.add_argument('--output', type=Path, default=Path.cwd()/'captures'/datetime.now().strftime('feedforward_%Y%m%d_%H%M%S'))
    parser.add_argument('--trajectory', type=Path, default=DEFAULT_TRAJECTORY,
                        help='完整五次样条轨迹 CSV；默认使用已采验证轨迹')
    parser.add_argument('--model', type=Path, default=DEFAULT_MODEL)
    parser.add_argument('--speed', type=float, help='主体实机倍速，如 1.8；入场不加速，续用会话时默认保留已冻结倍速')
    parser.add_argument('--torque-mode', choices=TORQUE_MODES,
                        help='full 为完整动力学，gravity 为同一共享模型的重力项；续用会话时自动沿用')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--prepare-only', action='store_true', help='只离线生成并检查计划')
    mode.add_argument('--analyze-only', action='store_true', help='只分析指定输出目录已有记录')
    return parser


def main():
    args = build_parser().parse_args()
    try:
        return run(args)
    except (arm_workflow.Cancelled, KeyboardInterrupt, EOFError):
        print('\n已取消；已有文件保留，可用原输出目录继续。'); return 0
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        print(f'本步未完成：{error}；已有文件保留。', file=sys.stderr); return 2


if __name__ == '__main__':
    raise SystemExit(main())
