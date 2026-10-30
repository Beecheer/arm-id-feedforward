#!/usr/bin/env python3
"""Prepare three complementary training runs and one fresh validation run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

if __package__:
    from . import arm_collect, arm_dynamics, arm_excitation as excitation, arm_plan
    from .arm_workflow import Workflow, Cancelled, confirm
else:
    import arm_collect
    import arm_dynamics
    import arm_excitation as excitation
    import arm_plan
    from arm_workflow import Workflow, Cancelled, confirm

# Mid-segment targets in degrees; J2/J4 remain inside the existing margins.
# Different coupled postures distinguish added coverage from repeated sampling.
TARGETS = {
    'train_01': [[0,65,-70,-70,15,60,0], [20,95,-65,-105,55,45,-30], [-25,75,-60,-90,-45,65,40]],
    'train_02': [[0,80,40,-90,-25,70,10], [-30,110,-25,-120,65,65,-45], [30,60,55,-65,-55,60,50]],
    'train_03': [[30,110,-65,-115,65,-45,-60], [-35,65,50,-75,-65,55,65], [10,90,-55,-100,45,70,-30]],
    'blind': [[-15,85,-60,-95,40,55,30], [25,100,45,-110,-40,60,-40], [-25,70,-40,-80,55,-35,50]],
}
DURATION = 32.
ANCHOR_TOLERANCE = np.radians(6.)


def row_bases(dynamics, basis):
    """Fixed kinematic reference only; no captured torque or fitted model used."""
    rng = np.random.default_rng(20261029)
    lo, hi = arm_plan.planning_bounds(dynamics.model)
    q = rng.uniform(lo, hi, (512, 7))
    v = rng.uniform(-.25, .25, (512, 7))
    a = rng.uniform(-.5, .5, (512, 7))
    reference = dynamics.matrix(q, v, a).reshape(7, 512, -1)
    result = []
    for W in reference:
        _, s, vt = np.linalg.svd(W @ basis, full_matrices=False)
        rank = int(np.sum(s > s[0] * 1e-8))
        result.append(basis @ vt[:rank].T / s[:rank])
    return result


class CoverageProblem(excitation.ExcitationProblem):
    def __init__(self, dynamics, basis, screen, seed, previous, *, target, rows, grams):
        self.target = np.radians(target)
        self.rows = rows
        self.grams = grams
        super().__init__(dynamics, basis, screen, seed, None, duration=DURATION)

    def blocks(self, x):
        _, q, v, a = self.motion.evaluate(x, excitation.OPT_FREQUENCY)
        W = self.dynamics.matrix(q, v, a).reshape(7, len(q), -1)
        return [Wj @ Tj for Wj, Tj in zip(W, self.rows)]

    def _objective(self, x):
        return -sum(float(np.linalg.slogdet(G + W.T @ W + np.eye(W.shape[1])*1e-8)[1])
                    for G, W in zip(self.grams, self.blocks(x)))

    def _initial(self, seed):
        rng = np.random.default_rng(seed)
        for _ in range(100):
            controls = np.clip(self.target + rng.normal(0, .055, (len(self.motion.free), 7)), self.lo, self.hi)
            x = controls.ravel()
            _, q, v, a = self.motion.evaluate(x, 20.)
            if (np.max(abs(v)) < excitation.MAX_VELOCITY*.94
                    and np.max(abs(a)) < excitation.MAX_ACCELERATION*.94
                    and np.max(abs(q[len(q)//2] - self.target)) < ANCHOR_TOLERANCE
                    and all(self.screen.clear(p) for p in q)):
                return x
        raise ValueError('补采目标没有可行初值；未发布该计划')

    def constraints(self, x):
        _, q, _, _ = self.motion.evaluate(x, excitation.CONSTRAINT_FREQUENCY)
        return np.r_[super().constraints(x), q[len(q)//2]]

    def get_constraint_bounds(self):
        lo, hi = super().get_constraint_bounds()
        return np.r_[lo, self.target-ANCHOR_TOLERANCE], np.r_[hi, self.target+ANCHOR_TOLERANCE]

    def check_candidate(self, x):
        _, q, _, _ = self.motion.evaluate(x, excitation.OUTPUT_FREQUENCY)
        if np.any(abs(q[len(q)//2]-self.target) > ANCHOR_TOLERANCE+1e-6):
            raise ValueError('优化候选未达到补采姿态范围')

    def on_selected(self, x):
        for G, W in zip(self.grams, self.blocks(x)):
            G += W.T @ W


def prepare(output):
    output.mkdir(parents=True, exist_ok=False)
    dynamics = arm_dynamics.DynamicModel()
    basis, meta = dynamics.reference_basis()
    rows = row_bases(dynamics, basis)
    screen = arm_plan.GeometryScreen(dynamics.model, arm_collect.URDF)
    exact = excitation.ExactGeometry(dynamics.model)
    training_grams = [np.zeros((T.shape[1], T.shape[1])) for T in rows]
    paths = {}
    reports = {}
    for index, (name, targets) in enumerate(TARGETS.items()):
        # Validation has a separate information objective, never updates training.
        grams = training_grams if name != 'blind' else [np.zeros_like(G) for G in training_grams]
        counter = iter(targets)

        def factory(*args):
            return CoverageProblem(*args, target=next(counter), rows=rows, grams=grams)

        directory = output/name/'plan'
        directory.mkdir(parents=True)
        stage = 'validation' if name == 'blind' else 'dynamic'
        path, report = excitation.generate_trajectory(directory, stage, 30101+index*100,
            dynamics, basis, screen, exact, problem_factory=factory)
        with np.load(directory/f'{stage}_derivatives.npz') as data:
            q, v, a = (data[k][::20] for k in ('q','dq','ddq'))
        W = dynamics.matrix(q, v, a).reshape(7, len(q), -1)
        report.update(objective='sum of per-joint D-optimal log determinants; cumulative across training runs only',
            segment_duration_s=DURATION,
            per_joint_information=[arm_dynamics.information(Wj @ Tj) for Wj, Tj in zip(W, rows)],
            midpoint_targets_deg=targets, midpoint_tolerance_deg=6.,
            position_min_deg=np.degrees(q.min(axis=0)).tolist(),
            position_max_deg=np.degrees(q.max(axis=0)).tolist())
        # Replace only metadata created by this invocation, never captured data.
        path.with_suffix('.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
        screening = dict(kind='arm_stage0_screening', version=1, profile='figaroh_full_dynamic',
            urdf_sha256=arm_collect._sha256(arm_collect.URDF),
            geometry_sha256=arm_collect._sha256(arm_collect.URDF),
            plans={stage:dict(file=path.name, sha256=arm_collect._sha256(path))},
            basis=meta, torque_constraint=False, **{stage:report})
        (directory/'screening_report.json').write_text(json.dumps(screening, ensure_ascii=False, indent=2)+'\n')
        paths[name] = str(path.relative_to(output))
        reports[name] = report
        (output/'progress.json').write_text(json.dumps(paths, indent=2)+'\n')
    manifest = dict(kind='arm_supplemental_collection', version=1, plans=paths,
        per_joint_reference_ranks=[T.shape[1] for T in rows], reports=reports,
        scope='Collect all three training runs before model selection. Freeze models before evaluating fresh validation torque. No automatic fitting or firmware deployment.')
    (output/'pack.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    write_report(output, manifest)
    print(f'三组训练和一组新验证已生成：{output}', flush=True)
    return manifest


def write_report(output, manifest):
    reports = manifest['reports']
    training = [reports[name] for name in TARGETS if name != 'blind']
    lo = np.min([item['position_min_deg'] for item in training], axis=0)
    hi = np.max([item['position_max_deg'] for item in training], axis=0)
    blind = reports['blind']
    lines = ['# 补充动态采集计划', '',
        '三组训练 + 一组新验证。每组 96 秒，另有原流程的 24 秒入场。',
        '逐轴信息优化，共 206 个参考方向；姿态带只约束离线计划。', '',
        '| 轴 | 三组训练规划范围 ° | 新验证规划范围 ° |', '|---|---:|---:|']
    for j in range(7):
        lines.append(f'| J{j+1} | {lo[j]:.1f} ～ {hi[j]:.1f} | '
            f'{blind["position_min_deg"][j]:.1f} ～ {blind["position_max_deg"][j]:.1f} |')
    lines += ['', '| 轨迹 | 逐轴秩 | 求解状态（逐段） |', '|---|---|---|']
    for name, item in reports.items():
        ranks = '/'.join(str(info['rank']) for info in item['per_joint_information'])
        status = '/'.join(str(seg['solver_status']) for seg in item['segments'])
        lines.append(f'| {name} | {ranks} | {status} |')
    lines += ['', '状态 -1 表示达到迭代上限，不表示收敛。发布的候选均已通过整段离线检查。',
        '保持现有 J2/J4 收紧限位，不设力矩约束。实际覆盖以后续反馈位置复核为准。',
        '采集时沿用位控，不因位置跟踪误差或反馈越界新增中断。', '',
        '在独立上位机仓库中运行：', '```bash',
        f'uv run ega-arm-collect-more collect --output captures/{output.name}', '```', '',
        '每组分别预览、输入执行确认、采集和质量分析，完成后可续采。',
        '此入口不辨识。合并旧、新训练数据，冻结候选模型后再评估新验证力矩；不改固件。', '']
    (output/'report.md').write_text('\n'.join(lines))


def collect(output, port=None):
    manifest = json.loads((output/'pack.json').read_text())
    if manifest.get('kind') != 'arm_supplemental_collection' or manifest.get('version') != 1:
        raise ValueError('补采计划格式无效')
    confirm('使用补采计划？按三组训练、一组新验证依次预览、确认执行和分析；暂不辨识')
    for name in TARGETS:
        workflow = Workflow(output/name, port)
        stage = 'validation' if name == 'blind' else 'dynamic'
        if stage in workflow.state['completed']:
            record = workflow.state.get('attempts', {}).get(stage, {})
            quality_path = workflow.output/record.get('directory', '')/'quality_report.json'
            incomplete = quality_path.is_file() and '动态轨迹采集时间覆盖不完整' in json.loads(quality_path.read_text()).get('issues', [])
            if incomplete and input(f'{name} 上次反馈覆盖不完整。输入 y 重采这一组，回车保留并继续：').strip().lower() in ('y', 'yes'):
                workflow.state['completed'].remove(stage)
                workflow.save()
            else:
                print(f'{name} 已保存，跳过。')
                continue
        workflow.state['preparation'] = 'plan/screening_report.json'
        workflow.save()
        print(f'\n补充采集：{name}')
        workflow.run_stage(stage)
        port = workflow.port
    print('补采结束。训练数据与新验证分开保存；暂不拟合、比较或部署参数。')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare','collect'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--port')
    args = parser.parse_args()
    try:
        if args.action == 'prepare': prepare(args.output.resolve())
        else: collect(args.output.resolve(), args.port)
    except (Cancelled, EOFError, KeyboardInterrupt):
        print('已退出，已有文件保留。')
        return 0
    except (OSError, RuntimeError, ValueError) as error:
        print(f'本次步骤未完成：{error}')
        print('恢复通信后重新运行原命令即可续采。')
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
