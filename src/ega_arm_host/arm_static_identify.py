#!/usr/bin/env python3
"""Offline gravity identification from measured static position/torque feedback.

No target position, firmware gravity output, or controller gains enter the fit.
The exported 70-vector is a nominal-prior representative, not unique link inertias.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pinocchio as pin

if __package__:
    from .arm_collect import URDF, _load_urdf
else:
    from arm_collect import URDF, _load_urdf

RIDGES = (0.0, 1e-8, 1e-6, 1e-4, 1e-2, 0.1, 1.0)
RANK_TOL = 1e-8


def fit_gravity(blocks, torque, prior, ridge=0.0):
    """Column-scaled observable SVD; retain the prior in the null space.

    Ridge is relative to the largest squared singular value. Scaling and SVD
    are recomputed from training poses in every validation fold.
    """
    matrix = np.asarray(blocks).reshape(-1, len(prior))
    residual = np.asarray(torque).reshape(-1) - matrix @ prior
    norms = np.linalg.norm(matrix, axis=0)
    active = norms > max(float(norms.max()) * 1e-10, 1e-12)
    if not np.any(active):
        raise ValueError("静态姿态没有重力可观测信息")
    scaled = matrix[:, active] / norms[active]
    left, singular, right = np.linalg.svd(scaled, full_matrices=False)
    rank = int(np.sum(singular > singular[0] * RANK_TOL))
    if not rank:
        raise ValueError("静态重力回归矩阵秩为零")
    s = singular[:rank]
    coordinates = (left[:, :rank].T @ residual) * s / (s*s + ridge*singular[0]**2)
    parameters = np.array(prior, copy=True)
    parameters[active] += (right[:rank].T @ coordinates) / norms[active]
    return parameters, {
        "rank": rank, "active_columns": np.flatnonzero(active).tolist(),
        "column_norms": norms[active].tolist(),
        "observable_basis_scaled": right[:rank].tolist(),
        "singular_values": singular.tolist(),
        "condition_observable_scaled": float(s[0] / s[-1]),
        "delta_observable_coordinates": coordinates.tolist(),
        "ridge_relative": ridge, "rank_relative_tolerance": RANK_TOL,
    }


def cross_validate(blocks, torque, prior):
    """Leave one complete pose out; never split correlated samples randomly."""
    predictions = []
    for ridge in RIDGES:
        predicted = np.empty_like(torque)
        for pose in range(len(torque)):
            train = np.arange(len(torque)) != pose
            params, _ = fit_gravity(blocks[train], torque[train], prior, ridge)
            predicted[pose] = blocks[pose] @ params
        predictions.append(predicted)
    scores = [float(np.mean((p-torque)**2)) for p in predictions]
    best = int(np.argmin(scores))
    return RIDGES[best], predictions[best], [
        {"ridge_relative": ridge, "leave_one_pose_out_rmse_nm": float(np.sqrt(score))}
        for ridge, score in zip(RIDGES, scores)
    ]


def process_windows(capture: Path, model, window_s=2.0):
    with capture.open(newline="", encoding="utf-8-sig") as stream:
        rows = [r for r in csv.DictReader(stream)
                if r.get("phase") in ("position_capture", "gravity_capture")]
    groups = {}
    invalid = duplicates = 0
    seen = set()
    for row in rows:
        state = int(row['status'], 0)
        expected = 9 if row['phase'] == 'gravity_capture' else 15
        values = np.array([float(row[f'{prefix}{j}_{unit}'])
                           for prefix, unit in [('q', 'rad'), ('dq', 'rad_s'), ('tau', 'nm')]
                           for j in range(1, 8)])
        if not np.all(np.isfinite(values)) or (state & 15) != expected:
            invalid += 1
            continue
        key = (int(row['device_time_us']), int(row['sample_sequence']))
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        groups.setdefault(int(row['pose_index']), []).append((key[0]*1e-6, values))
    if len(groups) < 4:
        raise ValueError("至少需要四个有效静态姿态以进行留出验证")
    data = model.createData()
    zero = np.zeros(7)
    blocks, torques, summaries = [], [], []
    for pose, records in sorted(groups.items()):
        t = np.array([r[0] for r in records]); a = np.array([r[1] for r in records])
        if np.any(np.diff(t) <= 0):
            raise ValueError(f"姿态 {pose} 时间戳倒退；请拆分不同设备时间周期后分析")
        # Fixed duration and selection based ONLY on measured motion, never fit error.
        # Disallow hidden interpolation across feedback gaps.
        candidates = []
        for start in np.arange(t[0]+0.5, t[-1]-window_s+1e-6, 0.25):
            use = (t >= start) & (t <= start+window_s)
            if np.sum(use) < 30 or np.max(np.diff(t[use])) > 0.1:
                continue
            qstd = np.std(a[use, :7], axis=0)
            vrms = np.sqrt(np.mean(a[use, 7:14]**2, axis=0))
            score = float(np.mean((qstd/0.01)**2 + (vrms/0.03)**2))
            candidates.append((score, use))
        if not candidates:
            raise ValueError(f"姿态 {pose} 没有连续的 {window_s:g} 秒分析窗口")
        score, use = min(candidates, key=lambda item: item[0])
        selected = a[use]
        # Average Y over measured positions, rather than substitute the target,
        # or approximate nonlinear E[Y(q)] by Y(E[q]).
        regressor = np.mean([
            pin.computeJointTorqueRegressor(model, data, q, zero, zero).copy()
            for q in selected[:, :7]
        ], axis=0)
        tau = np.mean(selected[:, 14:], axis=0)
        blocks.append(regressor); torques.append(tau)
        summaries.append({
            'pose_index': pose, 'available_samples': len(a), 'selected_samples': len(selected),
            'start_device_s': float(t[use][0]), 'end_device_s': float(t[use][-1]),
            'start_in_hold_s': float(t[use][0]-t[0]), 'motion_score': score,
            'position_mean_rad': np.mean(selected[:, :7], axis=0).tolist(),
            'position_std_rad': np.std(selected[:, :7], axis=0).tolist(),
            'velocity_rms_rad_s': np.sqrt(np.mean(selected[:, 7:14]**2, axis=0)).tolist(),
            'torque_mean_nm': tau.tolist(),
            'torque_std_nm': np.std(selected[:, 14:], axis=0).tolist(),
        })
    return np.array(blocks), np.array(torques), summaries, {
        'hold_samples': len(rows), 'invalid_samples': invalid, 'duplicates_removed': duplicates,
        'window_s': window_s, 'skip_initial_hold_s': 0.5, 'candidate_stride_s': 0.25,
        'selection': 'minimum measured-motion score only; equal weight per pose',
    }


def identify(capture: Path, urdf: Path, output: Path):
    model = _load_urdf(urdf)
    metadata_path = capture.with_suffix('.json')
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        expected_hash = metadata.get('urdf_sha256')
        if expected_hash and hashlib.sha256(urdf.read_bytes()).hexdigest() != expected_hash:
            raise ValueError('URDF 与采集时的模型哈希不一致')
    blocks, torque, poses, processing = process_windows(capture, model)
    prior = np.concatenate([inertia.toDynamicParameters() for inertia in model.inertias[1:]])
    ridge, _, candidates = cross_validate(blocks, torque, prior)
    parameters, basis = fit_gravity(blocks, torque, prior, ridge)
    nominal = blocks @ prior
    predicted = blocks @ parameters
    # Outer folds reselect regularization using only remaining poses. This avoids
    # presenting the hyperparameter-selection CV minimum as an unbiased estimate.
    outer = np.empty_like(torque)
    outer_ridges = []
    for pose in range(len(torque)):
        train = np.arange(len(torque)) != pose
        inner_ridge, _, _ = cross_validate(blocks[train], torque[train], prior)
        p, _ = fit_gravity(blocks[train], torque[train], prior, inner_ridge)
        outer[pose] = blocks[pose] @ p
        outer_ridges.append(inner_ridge)
    metrics = []
    for j in range(7):
        span = float(np.ptp(torque[:, j]))
        rmse = lambda p: float(np.sqrt(np.mean((p[:, j]-torque[:, j])**2)))
        metrics.append({
            'joint': j+1, 'measured_torque_span_nm': span,
            'nominal_gravity_span_nm': float(np.ptp(nominal[:, j])),
            'median_within_window_torque_std_nm': float(np.median([
                p['torque_std_nm'][j] for p in poses
            ])),
            'nominal_rmse_nm': rmse(nominal), 'fit_rmse_nm': rmse(predicted),
            'nested_leave_one_pose_out_rmse_nm': rmse(outer),
            'fit_bias_nm': float(np.mean(predicted[:, j]-torque[:, j])),
            'max_abs_fit_error_nm': float(np.max(np.abs(predicted[:, j]-torque[:, j]))),
            'gravity_observable': bool(np.linalg.norm(blocks[:, j]) > 1e-10),
        })
    report = {
        'kind': 'arm_static_gravity_identification', 'version': 1,
        'capture': str(capture.resolve()), 'capture_sha256': hashlib.sha256(capture.read_bytes()).hexdigest(),
        'urdf': str(urdf.resolve()), 'urdf_sha256': hashlib.sha256(urdf.read_bytes()).hexdigest(),
        'equation': 'mean(tau_feedback) = mean(Y(q_measured,0,0)) @ theta + residual',
        'torque_source': 'raw_feedback; no calibration or controller model',
        'processing': processing, 'poses': poses, 'basis': basis,
        'regularization_selection': candidates, 'outer_validation_ridges': outer_ridges,
        'joint_metrics': metrics,
        'limitations': [
            'Static data identify gravity combinations only, not inertia tensors or unique link masses/COMs.',
            'Exported parameters are an unconstrained nominal-prior representative; physical feasibility is not enforced.',
            'Residual includes static friction, remaining motion, payload/model and measurement errors; no controller torque is subtracted.',
            'One approach per pose cannot separate static friction from gravity or establish repeatability.',
            'Nested pose holdout is internal validation of this session, not independent experimental validation.',
        ],
    }
    output.mkdir(parents=True, exist_ok=False)
    (output/'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
    artifact = {k: report[k] for k in ('kind', 'version', 'urdf_sha256', 'equation', 'limitations')}
    artifact.update({
        'usage': 'gravity only: pin.computeJointTorqueRegressor(model,data,q,zeros,zeros) @ parameters',
        'parameter_order': 'URDF joint order; Pinocchio toDynamicParameters: m,m*cx,m*cy,m*cz,Ixx,Ixy,Iyy,Ixz,Iyz,Izz',
        'parameters': parameters.tolist(), 'nominal_parameters': prior.tolist(), 'basis': basis,
        'scope': 'gravity_only; do not use as identified full dynamic inertias or firmware configuration',
    })
    (output/'gravity_model.json').write_text(json.dumps(artifact, indent=2, ensure_ascii=False)+'\n')
    with (output/'pose_results.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['pose_index', 'joint', 'q_mean_rad', 'q_std_rad', 'dq_rms_rad_s',
                         'tau_mean_nm', 'tau_std_nm', 'nominal_nm', 'fit_nm', 'nested_holdout_nm'])
        for i, pose in enumerate(poses):
            for j in range(7):
                writer.writerow([pose['pose_index'], j+1, pose['position_mean_rad'][j],
                                 pose['position_std_rad'][j], pose['velocity_rms_rad_s'][j],
                                 torque[i,j], pose['torque_std_nm'][j], nominal[i,j], predicted[i,j], outer[i,j]])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 2, figsize=(13, 13), constrained_layout=True)
    x = [p['pose_index'] for p in poses]
    for j, ax in enumerate(axes.flat[:7]):
        ax.errorbar(x, torque[:,j], yerr=[p['torque_std_nm'][j] for p in poses], fmt='o', label='Measured mean +/- temporal std')
        ax.plot(x, nominal[:,j], ':', label='Nominal URDF')
        ax.plot(x, predicted[:,j], '-', label='Fit (all poses)')
        ax.plot(x, outer[:,j], 'x', label='Nested pose holdout')
        ax.set(title=f'J{j+1}', xlabel='Pose', ylabel='Torque (N m)'); ax.grid(alpha=.25)
    axes.flat[0].legend(fontsize=7)
    axes.flat[7].axis('off')
    fig.savefig(output/'torque_fit.png', dpi=160); plt.close(fig)
    lines = ['# 静态重力辨识', '',
             f'采用 {len(poses)} 个姿态，每姿态选择连续 2 秒低运动窗口；实际位置和原始力矩参与计算。', '',
             f'可观测秩：{basis["rank"]}；列归一化条件数：{basis["condition_observable_scaled"]:.2f}；正则化：{ridge:g}。', '',
             '| 轴 | 原 URDF RMSE | 拟合 RMSE | 嵌套留姿态验证 RMSE |',
             '|---|---:|---:|---:|']
    for m in metrics:
        lines.append(f'| J{m["joint"]} | {m["nominal_rmse_nm"]:.4f} | {m["fit_rmse_nm"]:.4f} | {m["nested_leave_one_pose_out_rmse_nm"]:.4f} |')
    lines += ['', '单位：N·m。验证按整个姿态留出，并在训练姿态内部重新选择正则化；没有随机拆分相邻样本。', '',
              '![力矩拟合与留出验证](torque_fit.png)', '',
              '仅辨识重力参数组合。导出的 70 维参数是保留 URDF 不可观测部分的数学代表，未约束物理可行性，不能解释成唯一连杆质量/质心，也不能作为完整动力学参数或直接写入固件。', '',
              '残差包含静摩擦、剩余运动、模型及反馈误差；本次没有扣除 PD 或使用下位机重力补偿输出来拟合。J1 在该模型中没有重力可观测信息。J7 的名义重力很小，不能把它的残余力矩直接解释为重力。', '',
              '单次单方向到达姿态无法分离静摩擦，也不能给出重复性结论。留出验证只覆盖本会话已有姿态，尚无独立新数据验证。', '']
    improved = [f'J{m["joint"]}' for m in metrics
                if m['gravity_observable'] and
                m['nested_leave_one_pose_out_rmse_nm'] < 0.8*m['nominal_rmse_nm']]
    lines += ['## 本组结果解读', '',
              f'留出误差比原 URDF 降低超过 20% 的轴：{", ".join(improved) or "无"}。其余轴尚未表现出稳定的辨识收益，不应仅凭训练误差采用整套参数。', '',
              f'所选窗口最大位置标准差约 {np.degrees(max(max(p["position_std_rad"]) for p in poses)):.3f}°。', '',
              '各轴窗口内力矩标准差的中位数（N·m）：' +
              '、'.join(f'{m["median_within_window_torque_std_nm"]:.4f}' for m in metrics) +
              '。若跨姿态残差明显大于此数值，延长同一姿态采样通常不能消除该系统性残差。', '',
              '本结果先作为离线重力模型候选保留，不修改控制器。后续若要分离静摩擦，应对相同实际姿态从两个方向到达并重复采集；本次没有执行补采。', '']
    (output/'report.md').write_text('\n'.join(lines))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture', type=Path)
    parser.add_argument('--output', type=Path, required=True, help='New output directory; never overwrite a prior fit')
    parser.add_argument('--urdf', type=Path, default=URDF)
    args = parser.parse_args()
    report = identify(args.capture, args.urdf, args.output)
    print(f'静态辨识完成：{len(report["poses"])} 个姿态，可观测秩 {report["basis"]["rank"]}')
    print(f'结果：{args.output / "report.md"}')


if __name__ == '__main__':
    main()
