#!/usr/bin/env python3
"""Generate a new coupled validation path for the shared-model feedforward run."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

if __package__:
    from . import arm_collect, arm_feedforward, arm_plan
    from .arm_excitation import ExactGeometry
else:
    import arm_collect
    import arm_feedforward
    import arm_plan
    from arm_excitation import ExactGeometry


HOST = Path(__file__).resolve().parent
DEFAULT_OUTPUT = Path.cwd()/'plans/complex_validation_001.csv'
SEGMENT_DURATION = 32.
SAMPLE_RATE = 100.
SEED = 31101
# Shoulder, wrist, and mixed segments; all seven joints still move together.
PATTERNS = np.asarray([
    [1.00, 1.00, 1.00, 0.85, 0.75, 0.75, 0.75],
    [0.65, 1.00, 1.00, 1.00, 0.55, 0.55, 0.55],
    [0.55, 0.65, 0.55, 0.65, 1.00, 1.00, 1.00],
    [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
    [0.85, 1.00, 0.85, 1.00, 0.85, 0.85, 0.85],
])


def evaluate(spline, frequency):
    t = np.arange(round(SEGMENT_DURATION*frequency)+1)/frequency
    return t, *(spline(t, nu=derivative) for derivative in range(3))


def feasible(q, v, a, bounds, geometry, stride):
    lo, hi = bounds
    return (np.all(q >= lo-1e-8) and np.all(q <= hi+1e-8)
            and np.max(abs(v)) <= .25+1e-8
            and np.max(abs(a)) <= .5+1e-8
            and all(geometry.clear(pose) for pose in q[::stride]))


def select_segment(rng, pattern, bounds, geometry):
    lo, hi = bounds
    candidates = []
    for _ in range(24):
        raw = rng.uniform(lo, hi, size=(5, 7))
        delta = (raw-arm_plan.HOME)*pattern
        for scale in (1., .85, .7, .55, .4):
            controls = arm_plan.HOME+delta*scale
            spline = arm_feedforward.quintic_spline(SEGMENT_DURATION,
                                                    arm_plan.HOME, controls)
            _, q, v, a = evaluate(spline, 10.)
            if feasible(q, v, a, bounds, geometry, 2):
                travel = float(np.abs(np.diff(q, axis=0)).sum())
                candidates.append((travel, controls, scale))
                break
    for travel, controls, scale in sorted(candidates, key=lambda item: -item[0]):
        spline = arm_feedforward.quintic_spline(SEGMENT_DURATION,
                                                arm_plan.HOME, controls)
        t, q, v, a = evaluate(spline, SAMPLE_RATE)
        if feasible(q, v, a, bounds, geometry, 1):
            return dict(t=t, q=q, v=v, a=a, controls=controls,
                        scale=scale, travel=travel)
    raise ValueError('没有找到通过全路径几何与规划范围检查的复杂轨迹段')


def write_overview(output, t, q):
    from matplotlib import pyplot as plt

    figure, axis = plt.subplots(figsize=(13, 5))
    for joint in range(7):
        axis.plot(t, np.degrees(q[:, joint]), lw=.8, label=f'J{joint+1}')
    for boundary in range(1, len(PATTERNS)):
        axis.axvline(boundary*SEGMENT_DURATION, color='0.75', lw=.8)
    axis.set(xlabel='Source trajectory time (s)', ylabel='Target angle (deg)',
             title='Complex validation path: five coupled spline segments')
    axis.grid(alpha=.25)
    axis.legend(ncol=7, loc='upper center', bbox_to_anchor=(.5, -.15))
    figure.tight_layout()
    figure.savefig(output.with_name(output.stem+'_overview.png'), dpi=140)
    plt.close(figure)


def generate(output=DEFAULT_OUTPUT, seed=SEED):
    output = Path(output)
    if output.exists() or output.with_suffix('.json').exists():
        raise FileExistsError(f'计划已存在，请使用新路径：{output}')
    model = arm_feedforward.SharedModel()
    geometry = ExactGeometry(model.dynamics.model)
    bounds = arm_plan.planning_bounds(model.dynamics.model)
    rng = np.random.default_rng(seed)
    selected = []
    for index, pattern in enumerate(PATTERNS):
        segment = select_segment(rng, pattern, bounds, geometry)
        selected.append(segment)
        print(f'第 {index+1}/{len(PATTERNS)} 段通过：源轨迹行程 {segment["travel"]:.2f} rad', flush=True)
    t = np.concatenate([part['t']+index*SEGMENT_DURATION if index == 0
                        else part['t'][1:]+index*SEGMENT_DURATION
                        for index, part in enumerate(selected)])
    q, v, a = (np.concatenate([part[key] if index == 0 else part[key][1:]
                               for index, part in enumerate(selected)])
               for key in ('q', 'v', 'a'))
    torque = model.predict(q, v, a)
    arm_collect._write_trajectory(output, t, q)
    metadata = dict(
        kind='arm_validation_spline', version=1,
        purpose='held-out complex coupled-path validation; not used for identification',
        trajectory_sha256=arm_collect._sha256(output),
        urdf_sha256=arm_collect._sha256(arm_collect.URDF),
        duration_s=float(t[-1]), segment_duration_s=SEGMENT_DURATION,
        sample_rate_hz=SAMPLE_RATE, spline_degree=5,
        control_points_per_segment=11, home_rad=arm_plan.HOME.tolist(),
        seed=seed, planning_margin_rad=arm_plan.JOINT_LIMIT_MARGINS.tolist(),
        geometry='URDF box proxies for self-collision and floor at 100 Hz, 5 mm clearance',
        torque_constraints=False,
        segments=[dict(pattern=PATTERNS[index].tolist(),
                       control_points=part['controls'].tolist(),
                       amplitude_scale=part['scale'])
                  for index, part in enumerate(selected)],
        position_span_rad=np.ptp(q, axis=0).tolist(),
        joint_travel_rad=np.abs(np.diff(q, axis=0)).sum(axis=0).tolist(),
        peak_velocity_rad_s=np.max(abs(v), axis=0).tolist(),
        peak_acceleration_rad_s2=np.max(abs(a), axis=0).tolist(),
        peak_model_torque=np.max(abs(torque), axis=0).tolist(),
    )
    arm_collect._write_metadata(output, metadata)
    arm_feedforward.Reference(output)  # Re-read the rounded CSV as the executor will.
    write_overview(output, t, q)
    print(f'新验证轨迹：{output}；源时长 {t[-1]:.1f} 秒；5 段，100 Hz。')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--seed', type=int, default=SEED)
    args = parser.parse_args()
    generate(args.output, args.seed)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
