"""Stage 0: offline model-based screening of acquisition plans."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import tempfile

import coal
import numpy as np
import pinocchio as pin

if __package__:
    from . import arm_collect
else:
    import arm_collect


HOME = np.asarray([0.0, 0.9, 0.0, -0.8, 0.0, 0.25, 0.0])
LIMIT_MARGIN = 0.15
JOINT_LIMIT_MARGINS = np.asarray([
    LIMIT_MARGIN, math.radians(30), LIMIT_MARGIN, math.radians(30),
    LIMIT_MARGIN, LIMIT_MARGIN, LIMIT_MARGIN,
])
CLEARANCE = 0.005
PATH_STEP = 0.05
POOL_SIZE = 256
SEED = 2026
RATED_TORQUE_USAGE = 0.8
PROFILE = arm_collect.URDF.with_name("profile.json")


def planning_bounds(model: pin.Model) -> tuple[np.ndarray, np.ndarray]:
    return (
        model.lowerPositionLimit + JOINT_LIMIT_MARGINS,
        model.upperPositionLimit - JOINT_LIMIT_MARGINS,
    )


def motor_torque_limits() -> tuple[np.ndarray, dict[str, object]]:
    config = json.loads(PROFILE.read_text(encoding="utf-8"))
    models = config["motor_models"]
    rated = np.asarray(config["rated_torque_nm"], dtype=float)
    usage = float(config.get("rated_usage_fraction", RATED_TORQUE_USAGE))
    if len(models) != 7 or rated.shape != (7,) or not np.all(np.isfinite(rated)):
        raise ValueError("profile must provide seven motor models and rated torques")
    if np.any(rated <= 0) or not 0 < usage <= 1:
        raise ValueError("profile rated torque and usage must be positive")
    return rated * usage, {
        "motor_models": models, "rated_torque_nm": rated.tolist(),
        "rated_usage_fraction": usage,
        "planning_limit_nm": (rated * usage).tolist(),
        "rating_sources": config.get("rating_sources", {}),
        "profile_sha256": arm_collect._sha256(PROFILE),
    }


class TorqueScreen:
    """Planning only: inverse dynamics and an empirical gravity envelope."""

    def __init__(self, model: pin.Model, limits: np.ndarray, gravity_scale=None):
        self.model = model
        self.data = model.createData()
        self.limits = np.asarray(limits, dtype=float)
        self.gravity_scale = np.ones(7) if gravity_scale is None else np.asarray(gravity_scale)
        self.rejected_pose_joints = np.zeros(7, dtype=int)
        self.transition_cache = {}
        if self.limits.shape != (7,) or not np.all(np.isfinite(self.limits)) or np.any(self.limits <= 0):
            raise ValueError("规划力矩预算必须包含七个正的有限数值")

    def effort(self, q, dq=None, ddq=None) -> np.ndarray:
        zero = np.zeros(7)
        nominal = pin.rnea(
            self.model, self.data, q,
            zero if dq is None else dq, zero if ddq is None else ddq,
        ).copy()
        gravity = pin.computeGeneralizedGravity(self.model, self.data, q).copy()
        # The CAD model can underpredict loads. Also screen the existing
        # empirical gravity scale + nominal inertial/Coriolis increment.
        # This is a conservative planning proxy, not physical calibration.
        proxy = nominal + (self.gravity_scale - 1.0) * gravity
        return np.maximum(np.abs(nominal), np.abs(proxy))

    def pose_feasible(self, q) -> bool:
        exceeded = self.effort(q) > self.limits
        self.rejected_pose_joints += exceeded
        return not np.any(exceeded)

    def transition(self, start, end, minimum=3.0):
        key = (np.asarray(start).tobytes(), np.asarray(end).tobytes(), minimum)
        if key not in self.transition_cache:
            self.transition_cache[key] = self._transition(start, end, minimum)
        return self.transition_cache[key]

    def _transition(self, start, end, minimum):
        count = max(64, math.ceil(float(np.max(np.abs(end - start))) / PATH_STEP))
        phase = np.linspace(0, 1, count + 1)
        delta = end - start
        blend = 10 * phase**3 - 15 * phase**4 + 6 * phase**5
        q = start + blend[:, None] * delta
        # A static overload anywhere on the route cannot be fixed by slowing.
        if any(np.any(self.effort(pose) > self.limits) for pose in q):
            return None
        duration = arm_collect._move_duration(start, end, minimum, 0.25, 0.5)
        for _ in range(10):
            duration = math.ceil(duration * 100) / 100
            dq = (30 * phase**2 - 60 * phase**3 + 30 * phase**4)[:, None] * delta / duration
            ddq = (60 * phase - 180 * phase**2 + 120 * phase**3)[:, None] * delta / duration**2
            peaks = self.trajectory_peaks(q, dq, ddq)
            if np.all(peaks <= self.limits):
                return duration, peaks
            duration *= 1.25
        return None

    def trajectory_peaks(self, q, dq, ddq) -> np.ndarray:
        return np.max(np.asarray([
            self.effort(position, velocity, acceleration)
            for position, velocity, acceleration in zip(q, dq, ddq)
        ]), axis=0)


class GeometryScreen:
    """Conservative URDF convex collision proxies; excludes adjacent links."""

    def __init__(self, model: pin.Model, urdf: Path):
        self.model = model
        self.data = model.createData()
        self.geometry = pin.buildGeomFromUrdf(
            model, str(urdf), pin.GeometryType.COLLISION,
            package_dirs=[str(urdf.parent)],
        )
        count = len(self.geometry.geometryObjects)
        if count != 8:
            raise ValueError("stage 0 requires eight link collision proxies")
        for geometry in self.geometry.geometryObjects:
            mesh = geometry.geometry
            mesh.buildConvexHull(True, "Qt")
            if mesh.convex is None:
                raise ValueError("cannot build link collision convex hull")
            geometry.geometry = mesh.convex
        for first in range(count):
            for second in range(first + 1, count):
                a = self.geometry.geometryObjects[first].parentJoint
                b = self.geometry.geometryObjects[second].parentJoint
                if abs(a - b) > 1:
                    self.geometry.addCollisionPair(pin.CollisionPair(first, second))
        ground = self.geometry.addGeometryObject(pin.GeometryObject(
            "planning_ground", 0, pin.SE3.Identity(),
            coal.Halfspace(np.asarray([0.0, 0.0, 1.0]), 0.0),
        ))
        for index in range(count):
            if self.geometry.geometryObjects[index].parentJoint != 0:
                self.geometry.addCollisionPair(pin.CollisionPair(index, ground))
        self.geometry_data = self.geometry.createData()
        for request in self.geometry_data.collisionRequests:
            request.security_margin = CLEARANCE
            request.enable_contact = False

    def clear(self, q: np.ndarray) -> bool:
        return not pin.computeCollisions(
            self.model, self.data, self.geometry, self.geometry_data, q, True,
        )

    def path_clear(self, start: np.ndarray, end: np.ndarray) -> bool:
        count = max(1, math.ceil(float(np.max(np.abs(end - start))) / PATH_STEP))
        return all(self.clear(start + phase * (end - start)) for phase in np.linspace(0, 1, count + 1))


def regressor_blocks(model: pin.Model, poses: np.ndarray) -> np.ndarray:
    data = model.createData()
    zero = np.zeros(7)
    return np.asarray([
        pin.computeJointTorqueRegressor(model, data, q, zero, zero).copy()
        for q in poses
    ])


def observable_blocks(
    blocks: np.ndarray, reference: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Normalize columns and remove structural dependencies using a reference pool."""
    reference = blocks if reference is None else reference
    stacked = reference.reshape(-1, reference.shape[-1])
    norms = np.linalg.norm(stacked, axis=0)
    active = norms > max(float(np.max(norms)) * 1e-10, 1e-12)
    if not np.any(active):
        raise ValueError("候选姿态没有可辨识的重力参数组合")
    scaled_reference = reference[..., active] / norms[active]
    _, singular, right = np.linalg.svd(
        scaled_reference.reshape(-1, np.sum(active)), full_matrices=False,
    )
    rank = int(np.sum(singular > singular[0] * 1e-8))
    if rank == 0:
        raise ValueError("候选姿态没有可辨识的重力参数组合")
    # Whiten against the candidate pool so units and repeated parameter columns
    # cannot dominate the information objective.
    projected = (blocks[..., active] / norms[active]) @ right[:rank].T / singular[:rank]
    return projected, {
        "full_parameter_columns": blocks.shape[-1],
        "active_columns": int(np.sum(active)), "reference_rank": rank,
        "column_scaled_reference_condition": float(singular[0] / singular[rank - 1]),
        "rank_relative_tolerance": 1e-8,
        "basis": "column-scaled SVD, reference-pool whitening",
        "scope": "observable combinations, not individual link parameters",
    }


def select_information(blocks: np.ndarray, count: int) -> list[int]:
    if count < 1 or count > len(blocks) or not np.all(np.isfinite(blocks)):
        raise ValueError("信息筛选需要有限候选数据及有效姿态数量")
    information = np.eye(blocks.shape[-1]) * 1e-6
    selected: list[int] = []
    for _ in range(count):
        inverse = np.linalg.inv(information)
        small = np.eye(blocks.shape[1]) + (blocks @ inverse) @ blocks.transpose(0, 2, 1)
        _, scores = np.linalg.slogdet(small)
        scores[selected] = -np.inf
        index = int(np.argmax(scores))
        selected.append(index)
        information += blocks[index].T @ blocks[index]
    return selected


def information_report(blocks: np.ndarray) -> dict[str, object]:
    stacked = blocks.reshape(-1, blocks.shape[-1])
    singular = np.linalg.svd(stacked, compute_uv=False)
    rank = int(np.sum(singular > max(singular[0] * 1e-8, 1e-12)))
    return {
        "rank": rank, "reference_rank": blocks.shape[-1],
        "condition_number": float(singular[0] / singular[-1]) if rank == blocks.shape[-1] else None,
        "regularized_logdet": float(np.linalg.slogdet(stacked.T @ stacked + np.eye(blocks.shape[-1]) * 1e-6)[1]),
    }


def select_alignment(torque: np.ndarray, count: int) -> list[int]:
    spans = np.ptp(torque, axis=0)
    usable = spans >= 0.05
    if not np.any(usable):
        raise ValueError("候选姿态没有足够的重力输入力矩变化")
    normalized = torque[:, usable] / spans[usable]
    blocks = np.zeros((len(torque), np.sum(usable), np.sum(usable) * 2))
    for joint in range(np.sum(usable)):
        blocks[:, joint, 2 * joint] = normalized[:, joint]
        blocks[:, joint, 2 * joint + 1] = 1.0
    return select_information(blocks, count)


def order_poses(poses: np.ndarray, screen: GeometryScreen, torque: TorqueScreen):
    remaining = list(range(len(poses)))
    result = [HOME.copy()]
    durations = [3.0]
    peaks = torque.effort(HOME)
    current = HOME
    while remaining:
        ordered = sorted(remaining, key=lambda i: np.linalg.norm(poses[i] - current))
        direct = next((
            i for i in ordered if screen.path_clear(current, poses[i])
            and torque.transition(current, poses[i]) is not None
        ), None)
        if direct is None:
            # Every retained candidate has already passed HOME -> pose screening.
            via_home = torque.transition(current, HOME)
            if via_home is None:
                raise ValueError("参考姿态回程力矩未通过规划检查")
            result.append(HOME.copy())
            durations.append(via_home[0])
            peaks = np.maximum(peaks, via_home[1])
            current = HOME
            direct = ordered[0]
        result.append(poses[direct])
        transition = torque.transition(current, poses[direct])
        durations.append(transition[0])
        peaks = np.maximum(peaks, transition[1])
        current = poses[direct]
        remaining.remove(direct)
    return np.asarray(result), durations, peaks


def dynamic_plan(output: Path, screen: GeometryScreen, torque: TorqueScreen):
    lower, upper = planning_bounds(torque.model)
    with tempfile.TemporaryDirectory(prefix="arm-plan-") as directory:
        candidate = Path(directory) / "candidate.csv"
        args = arm_collect.build_parser().parse_args([
            "dynamic-plan", "--seed", "81", "--limit-margin", str(LIMIT_MARGIN),
            "--output", str(candidate),
        ])
        arm_collect.make_dynamic_trajectory(args)
        raw = np.genfromtxt(candidate, delimiter=",", skip_header=1)
        metadata = json.loads(candidate.with_suffix(".json").read_text())
    amplitude_scale = 1.0
    for _ in range(12):
        q = HOME + amplitude_scale * (raw[:, 1:] - HOME)
        if (
            np.all(q >= lower) and np.all(q <= upper)
            and all(np.all(torque.effort(pose) <= torque.limits) for pose in q)
            and all(screen.clear(pose) for pose in q)
        ):
            break
        amplitude_scale *= 0.85
    else:
        raise ValueError("动态姿态的重力/几何负载无法满足规划预算")
    stretch = 1.0
    for _ in range(10):
        times = raw[:, 0] * stretch
        dq = np.gradient(q, times, axis=0, edge_order=2)
        ddq = np.gradient(dq, times, axis=0, edge_order=2)
        peaks = torque.trajectory_peaks(q, dq, ddq)
        if np.all(peaks <= torque.limits):
            break
        stretch *= 1.25
    else:
        raise ValueError("动态运动力矩无法满足规划预算")
    path = output / "dynamic.csv"
    arm_collect._write_trajectory(path, times, q)
    metadata.update({
        "trajectory_csv": str(path.resolve()), "trajectory_sha256": arm_collect._sha256(path),
        "duration_s": float(times[-1]), "sample_rate_hz": args.rate / stretch,
        "torque_amplitude_scale": amplitude_scale, "torque_time_stretch": stretch,
        "torque_planning_limit_nm": torque.limits.tolist(),
        "estimated_peak_torque_nm": peaks.tolist(),
        "joint_margin_rad": JOINT_LIMIT_MARGINS.tolist(),
        "joint_lower_rad": lower.tolist(), "joint_upper_rad": upper.tolist(),
    })
    arm_collect._write_metadata(path, metadata)
    return path, times, q, dq, ddq, peaks


def prepare(output: Path) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    if (output / "screening_report.json").exists():
        raise ValueError("第0步结果已存在，请使用新的运行目录")
    urdf = arm_collect.URDF.resolve()
    model = arm_collect._load_urdf(urdf)
    screen = GeometryScreen(model, urdf)
    profile = json.loads(PROFILE.read_text(encoding="utf-8"))
    gravity_scale = np.asarray(profile["planning_gravity_scale"], dtype=float)
    if gravity_scale.shape != (7,) or not np.all(np.isfinite(gravity_scale)):
        raise ValueError("profile requires seven planning gravity scales")
    limits, torque_settings = motor_torque_limits()
    torque = TorqueScreen(model, limits, gravity_scale)
    lower, upper = planning_bounds(model)
    if np.any(HOME < lower) or np.any(HOME > upper):
        raise ValueError("参考姿态超出带余量的规划范围")
    if not screen.clear(HOME):
        raise ValueError("模型参考姿态未通过碰撞/间距检查")
    if not torque.pose_feasible(HOME):
        raise ValueError("参考姿态超过规划力矩预算，需要选择更轻负载的参考姿态")
    rng = np.random.default_rng(SEED)
    candidates = []
    attempts = 0
    while len(candidates) < POOL_SIZE and attempts < POOL_SIZE * 30:
        q = rng.uniform(lower, upper)
        attempts += 1
        if (
            torque.pose_feasible(q) and screen.clear(q) and screen.path_clear(HOME, q)
            and torque.transition(HOME, q) is not None
            and torque.transition(q, HOME) is not None
        ):
            candidates.append(q)
        if attempts % 100 == 0:
            print("正在筛选姿态、过渡路径和力矩负载…", flush=True)
    if len(candidates) < POOL_SIZE:
        raise ValueError("可用候选姿态不足；未保存可执行计划")
    candidates = np.asarray(candidates)
    print("正在比较姿态的重力覆盖与参数信息…", flush=True)
    projected, basis = observable_blocks(regressor_blocks(model, candidates))
    static_indices = select_information(projected, 16)
    report = {
        "kind": "arm_stage0_screening", "version": 1, "seed": SEED,
        "urdf": str(urdf), "urdf_sha256": arm_collect._sha256(urdf),
        "geometry_sha256": arm_collect._sha256(urdf),
        "planner_sha256": arm_collect._sha256(Path(__file__)),
        "profile_sha256": arm_collect._sha256(PROFILE),
        "candidate_count": len(candidates), "attempted_candidates": attempts,
        "torque_screening": {
            **torque_settings,
            "gravity_proxy_scale": torque.gravity_scale.tolist(),
            "estimate": "max(abs(URDF RNEA), abs(URDF dynamic increment + empirical scaled gravity))",
            "static_rejections_per_joint": torque.rejected_pose_joints.tolist(),
            "scope": "stage 0 only; no runtime torque aborts",
            "limitations": "nominal CAD inertia and empirical gravity proxy, not calibrated output torque; no unmodeled payload/friction",
        },
        "screening": {
            "joint_margin_rad": JOINT_LIMIT_MARGINS.tolist(),
            "joint_lower_rad": lower.tolist(), "joint_upper_rad": upper.tolist(),
            "minimum_proxy_distance_m": CLEARANCE,
            "path_max_joint_step_rad": PATH_STEP,
            "geometry": "conservative URDF convex proxies, nonadjacent links, floor at z=0",
            "limitations": [
                "sampled paths, not continuous collision certification",
                "adjacent links excluded; no cables, fixtures, or real environment",
                "live start pose/approach unknown; execution still requires preview",
                "nominal model information, not measured-data identifiability",
            ],
        },
        "static": {
            "basis": basis, "selected": information_report(projected[static_indices]),
            "random_reference": information_report(projected[:16]),
            "objective": "greedy regularized D-optimality on observable gravity combinations",
        },
        "plans": {},
    }
    ordered, durations, peaks = order_poses(candidates[static_indices], screen, torque)
    plan = output / "static.csv"
    arm_collect._write_pose_plan(plan, ordered, ["static"] * len(ordered), durations)
    arm_collect._write_metadata(plan, {
        "kind": "screened_pose_plan", "stage": "static",
        "urdf_sha256": report["urdf_sha256"],
        "pose_csv_sha256": arm_collect._sha256(plan),
        "selected_information_poses": len(static_indices),
        "capture_poses_including_home_waypoints": len(ordered),
        "home_rad": HOME.tolist(),
        "torque_planning_limit_nm": limits.tolist(),
        "estimated_peak_torque_nm": peaks.tolist(),
        "transition_durations_s": durations,
        "joint_margin_rad": JOINT_LIMIT_MARGINS.tolist(),
        "joint_lower_rad": lower.tolist(), "joint_upper_rad": upper.tolist(),
    })
    report["static"]["estimated_peak_torque_nm"] = peaks.tolist()
    report["plans"]["static"] = {"file": plan.name, "sha256": arm_collect._sha256(plan)}
    # Preserve the current conservative dynamic trajectory, but inspect its
    # nominal excitation rather than claiming it is an optimized experiment.
    dynamic, times, q, dq, ddq, dynamic_peaks = dynamic_plan(output, screen, torque)
    data = model.createData()
    dynamic_blocks = np.asarray([
        pin.computeJointTorqueRegressor(model, data, q[i], dq[i], ddq[i]).copy()
        for i in range(0, len(q), 20)
    ])
    reference_blocks = np.asarray([
        pin.computeJointTorqueRegressor(
            model, data, pose, rng.uniform(-0.25, 0.25, 7),
            rng.uniform(-0.5, 0.5, 7),
        ).copy() for pose in candidates
    ])
    dynamic_projected, dynamic_basis = observable_blocks(dynamic_blocks, reference_blocks)
    report["dynamic"] = {
        "optimized": False, "basis": dynamic_basis,
        "reference": "independent candidate poses and random velocity/acceleration within motion caps",
        "information": information_report(dynamic_projected),
        "peak_velocity_rad_s": np.max(np.abs(dq), axis=0).tolist(),
        "peak_acceleration_rad_s2": np.max(np.abs(ddq), axis=0).tolist(),
        "estimated_peak_torque_nm": dynamic_peaks.tolist(),
        "warning": "conservative multisine; inertia excitation versus actual noise is unverified",
    }
    report["plans"]["dynamic"] = {"file": dynamic.name, "sha256": arm_collect._sha256(dynamic)}
    report["static"]["full_reference_rank_reached"] = (
        report["static"]["selected"]["rank"] == basis["reference_rank"]
    )
    if not report["static"]["full_reference_rank_reached"]:
        raise ValueError("静态姿态未覆盖参考池可辨识子空间；第0步未完成")
    with (output / "screening_report.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="第0步：离线模型筛选采集计划，不连接硬件")
    parser.add_argument("--output", type=Path, required=True)
    parser.set_defaults(handler=generate)
    return parser


def generate(args: argparse.Namespace) -> int:
    prepare(args.output)
    print("离线筛选结果已保存。")
    return 0


def main() -> int:
    args = build_parser().parse_args()
    try:
        report = prepare(args.output)
        print(f"第0步完成：{args.output.resolve()}")
        print("动态计划沿用保守轨迹；已记录激励诊断，未宣称优化完成。")
        return 0
    except (OSError, RuntimeError, ValueError) as error:
        parser = build_parser()
        parser.exit(2, f"第0步未完成：{error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
