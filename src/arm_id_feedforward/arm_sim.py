#!/usr/bin/env python3
"""Kinematic MuJoCo playback for active-arm identification trajectories."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np

if __package__:
    from . import arm_host
else:
    import arm_host


JOINT_NAMES = (
    "Joint_01",
    "Joint_12",
    "Joint_23",
    "Joint_34",
    "Joint_45",
    "Joint_56",
    "Joint_67",
)
JOINT_COUNT = len(JOINT_NAMES)
DEFAULT_MODEL = Path(__file__).with_name("models") / "controller" / "mjmodel.xml"


class Playback:
    def __init__(self, times: np.ndarray, positions: np.ndarray) -> None:
        self.times = times
        self.positions = positions

    @property
    def duration(self) -> float:
        return float(self.times[-1])

    def position_at(self, elapsed: float) -> np.ndarray:
        if elapsed <= self.times[0]:
            return self.positions[0]
        right = int(np.searchsorted(self.times, elapsed, side="right"))
        if right >= len(self.times):
            return self.positions[-1]
        left = right - 1
        fraction = (elapsed - self.times[left]) / (
            self.times[right] - self.times[left]
        )
        return self.positions[left] + fraction * (
            self.positions[right] - self.positions[left]
        )


def _trajectory_csv(path: Path) -> Playback:
    times: list[float] = []
    positions: list[list[float]] = []
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = ["time_s", *(f"q{joint}" for joint in range(1, 8))]
        if reader.fieldnames is None or any(
            name not in reader.fieldnames for name in required
        ):
            raise ValueError(
                f"{path}: expected trajectory columns " + ",".join(required)
            )
        for row_number, row in enumerate(reader, start=2):
            try:
                timestamp = float(row["time_s"])
                q = [float(row[f"q{joint}"]) for joint in range(1, 8)]
            except (TypeError, ValueError) as error:
                raise ValueError(f"{path}: invalid values on row {row_number}") from error
            if not math.isfinite(timestamp) or timestamp < 0.0:
                raise ValueError(f"{path}: invalid time_s on row {row_number}")
            if not np.all(np.isfinite(q)):
                raise ValueError(f"{path}: non-finite joint value on row {row_number}")
            if times and timestamp <= times[-1]:
                raise ValueError(f"{path}: time_s values must increase strictly")
            times.append(timestamp)
            positions.append(q)
    if len(times) < 2:
        raise ValueError(f"{path}: trajectory needs at least two rows")
    return Playback(np.asarray(times), np.asarray(positions))


def _live_start(args: argparse.Namespace) -> np.ndarray | None:
    if args.port is None:
        return None
    with arm_host.open_port(args.port, args.baud) as port:
        sample = arm_host.read_current_state(
            arm_host.ArmClient(port), args.state_timeout
        )
    return np.asarray(arm_host.command_start_for_plan(sample.position, args.input), dtype=float)


def _pose_plan_csv(
    path: Path,
    rate: float,
    minimum_transition_duration: float,
    hold_duration: float,
    start_position: np.ndarray | None = None,
    maximum_velocity: float = 0.25,
    maximum_acceleration: float = 0.5,
) -> tuple[Playback, list[str]]:
    poses: list[list[float]] = []
    labels: list[str] = []
    planned_durations: list[float] = []
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = [*(f"q{joint}_rad" for joint in range(1, 8))]
        if reader.fieldnames is None or any(
            name not in reader.fieldnames for name in required
        ):
            raise ValueError(
                f"{path}: expected pose-plan columns " + ",".join(required)
            )
        for row_number, row in enumerate(reader, start=2):
            try:
                q = [float(row[f"q{joint}_rad"]) for joint in range(1, 8)]
            except (TypeError, ValueError) as error:
                raise ValueError(f"{path}: invalid pose on row {row_number}") from error
            if not np.all(np.isfinite(q)):
                raise ValueError(f"{path}: non-finite pose on row {row_number}")
            poses.append(q)
            duration = float(row.get("move_duration_s", "0"))
            if not math.isfinite(duration) or duration < 0:
                raise ValueError(f"{path}: invalid move_duration_s on row {row_number}")
            planned_durations.append(duration)
            labels.append(row.get("stage", row.get("split", "pose")) or "pose")
    if not poses:
        raise ValueError(f"{path}: pose plan contains no poses")
    hold_steps = round(hold_duration * rate)
    if (
        minimum_transition_duration <= 0.0
        or maximum_velocity <= 0.0
        or maximum_acceleration <= 0.0
    ):
        raise ValueError("transition duration and motion limits must be positive")
    if hold_steps < 1 or not math.isclose(
        hold_steps, hold_duration * rate, rel_tol=0.0, abs_tol=1e-6
    ):
        raise ValueError("hold-duration * rate must be a positive integer")

    times = [0.0]
    positions = [
        np.asarray(start_position).tolist() if start_position is not None else poses[0]
    ]
    current_time = 0.0
    current = (
        np.asarray(start_position, dtype=float)
        if start_position is not None
        else np.asarray(poses[0], dtype=float)
    )
    first_pose = 0 if start_position is not None else 1
    if start_position is None:
        for _ in range(hold_steps):
            current_time += 1.0 / rate
            times.append(current_time)
            positions.append(current.tolist())
    for index in range(first_pose, len(poses)):
        pose = np.asarray(poses[index], dtype=float)
        delta = np.abs(pose - current)
        transition_duration = max(
            minimum_transition_duration,
            planned_durations[index],
            1.15 * float(np.max(1.875 * delta / maximum_velocity)),
            1.15 * float(np.max(np.sqrt(5.774 * delta / maximum_acceleration))),
        )
        transition_steps = max(2, math.ceil(transition_duration * rate))
        for step in range(1, transition_steps + 1):
            phase = step / transition_steps
            blend = 10.0 * phase**3 - 15.0 * phase**4 + 6.0 * phase**5
            current_time += 1.0 / rate
            times.append(current_time)
            positions.append((current + blend * (pose - current)).tolist())
        for _ in range(hold_steps):
            current_time += 1.0 / rate
            times.append(current_time)
            positions.append(pose.tolist())
        current = pose
    return Playback(np.asarray(times), np.asarray(positions)), labels


def _load_model(path: Path) -> tuple[mujoco.MjModel, mujoco.MjData, np.ndarray]:
    if not path.is_file():
        raise ValueError(f"MuJoCo model does not exist: {path}")
    try:
        model = mujoco.MjModel.from_xml_path(str(path))
    except (ValueError, RuntimeError) as error:
        raise ValueError(f"cannot load MuJoCo model {path}: {error}") from error
    if model.nq != JOINT_COUNT or model.nv != JOINT_COUNT:
        raise ValueError(
            f"expected a fixed-base {JOINT_COUNT}-joint model, got nq={model.nq}, "
            f"nv={model.nv}"
        )

    qpos_addresses: list[int] = []
    for name in JOINT_NAMES:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"MuJoCo model is missing joint {name}")
        if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_HINGE:
            raise ValueError(f"MuJoCo joint {name} is not a hinge")
        qpos_addresses.append(int(model.jnt_qposadr[joint_id]))

    # This is a kinematic visualization. Keep proxy boxes visual-only: never
    # invoke MuJoCo's contact/collision pipeline for this playback.
    model.geom_contype[:] = 0
    model.geom_conaffinity[:] = 0
    return model, mujoco.MjData(model), np.asarray(qpos_addresses, dtype=int)


def _limit_report(
    model: mujoco.MjModel,
    positions: np.ndarray,
    margin: float,
) -> None:
    lower = np.empty(JOINT_COUNT)
    upper = np.empty(JOINT_COUNT)
    for index, name in enumerate(JOINT_NAMES):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if not model.jnt_limited[joint_id]:
            raise ValueError(f"MuJoCo joint {name} has no position limits")
        lower[index], upper[index] = model.jnt_range[joint_id]
    sampled_q = positions
    distances = np.minimum(sampled_q - lower, upper - sampled_q)
    sample, joint = np.unravel_index(np.argmin(distances), distances.shape)
    if np.any(sampled_q < lower) or np.any(sampled_q > upper):
        print(
            f"WARNING: preview crosses the URDF limit at sample {sample}, "
            f"J{joint + 1}={sampled_q[sample, joint]:.4f} rad; range "
            f"[{lower[joint]:.4f}, {upper[joint]:.4f}] rad. Preview continues."
        )
    elif margin > 0.0 and float(np.min(distances)) < margin:
        print(
            f"WARNING: preview comes within {margin:g} rad of a URDF limit "
            f"(minimum distance {float(np.min(distances)):.4f} rad). Preview continues."
        )


def _print_summary(playback: Playback, margin: float) -> tuple[np.ndarray, np.ndarray]:
    dt = 1.0 / 200.0
    sample_time = np.arange(playback.times[0], playback.duration + dt / 2.0, dt)
    sampled = np.vstack([playback.position_at(float(t)) for t in sample_time])
    edge_order = 2 if len(sampled) >= 3 else 1
    velocity = np.gradient(sampled, dt, axis=0, edge_order=edge_order)
    acceleration = np.gradient(velocity, dt, axis=0, edge_order=edge_order)
    print(f"Playback duration: {playback.duration:.3f} s at real-time scale")
    print(f"Limit margin requirement: {margin:.3f} rad")
    print("Joint       min q       max q      peak |dq|      peak |ddq|")
    for joint in range(JOINT_COUNT):
        print(
            f"J{joint + 1:<2} {np.min(sampled[:, joint]):10.4f} "
            f"{np.max(sampled[:, joint]):10.4f} "
            f"{np.max(np.abs(velocity[:, joint])):13.4f} "
            f"{np.max(np.abs(acceleration[:, joint])):14.4f}"
        )
    return np.max(np.abs(velocity), axis=0), np.max(np.abs(acceleration), axis=0)


def _write_gif(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qpos_addresses: np.ndarray,
    motion: Playback,
    output: Path,
    fps: float,
    width: int,
    height: int,
    speed: float = 1.0,
) -> None:
    try:
        import imageio.v2 as imageio
    except ImportError as error:
        raise ValueError(
            "GIF export requires imageio and Pillow; run `uv sync --locked`"
        ) from error
    if fps <= 0.0 or not math.isfinite(fps):
        raise ValueError("fps must be finite and positive")
    if width < 64 or height < 64:
        raise ValueError("GIF width and height must be at least 64 pixels")

    output.parent.mkdir(parents=True, exist_ok=True)
    frame_count = max(2, int(math.ceil(motion.duration * fps / speed)))
    frame_times = np.linspace(0.0, motion.duration, frame_count, endpoint=False)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, camera)
    camera.lookat[:] = (0.0, 0.0, 0.26)
    camera.distance = max(1.2, 3.2 * model.stat.extent)
    camera.azimuth = 135.0
    camera.elevation = -28.0
    renderer = mujoco.Renderer(model, height=height, width=width)
    try:
        with imageio.get_writer(
            output,
            mode="I",
            format="GIF",
            duration=1000.0 / fps,
            loop=0,
        ) as writer:
            for frame_time in frame_times:
                data.qpos[qpos_addresses] = motion.position_at(float(frame_time))
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=camera)
                writer.append_data(renderer.render())
    finally:
        renderer.close()
    print(f"GIF preview: {output} ({frame_count} frames at {fps:g} fps)")


def playback(args: argparse.Namespace) -> int:
    if args.speed <= 0.0 or not math.isfinite(args.speed):
        raise ValueError("speed must be finite and positive")
    if not math.isfinite(args.limit_margin) or args.limit_margin < 0.0:
        raise ValueError("limit-margin cannot be negative")
    if not math.isfinite(args.rate) or args.rate <= 0.0:
        raise ValueError("rate must be finite and positive")
    if (
        not math.isfinite(args.transition_duration)
        or not math.isfinite(args.hold_duration)
        or not math.isfinite(args.maximum_velocity)
        or not math.isfinite(args.maximum_acceleration)
        or args.maximum_velocity <= 0.0
        or args.maximum_acceleration <= 0.0
    ):
        raise ValueError("transition, hold, and motion limits must be finite")
    if args.check_only and args.gif is not None:
        raise ValueError("--check-only and --gif cannot be used together")

    model, data, qpos_addresses = _load_model(args.model)
    start_position = _live_start(args)
    mode = args.format
    if mode == "auto":
        with args.input.open("r", newline="", encoding="utf-8-sig") as stream:
            header = next(csv.reader(stream), [])
        mode = "pose-plan" if "q1_rad" in header else "trajectory"
    labels: list[str] = []
    if mode == "trajectory":
        motion = _trajectory_csv(args.input)
        if start_position is not None:
            delta = np.abs(motion.positions[0] - start_position)
            transition_duration = max(
                args.transition_duration,
                1.15 * float(np.max(1.875 * delta / args.maximum_velocity)),
                1.15
                * float(
                    np.max(np.sqrt(5.774 * delta / args.maximum_acceleration))
                ),
            )
            transition_steps = math.ceil(transition_duration * args.rate)
            transition_duration = transition_steps / args.rate
            if transition_steps < 2 or not math.isclose(
                transition_steps,
                transition_duration * args.rate,
                rel_tol=0.0,
                abs_tol=1e-6,
            ):
                raise ValueError("transition duration must be representable at preview rate")
            target = motion.positions[0]
            transition_time = np.arange(transition_steps + 1, dtype=float) / args.rate
            phase = transition_time / transition_duration
            blend = 10.0 * phase**3 - 15.0 * phase**4 + 6.0 * phase**5
            transition_q = start_position + blend[:, None] * (target - start_position)
            motion = Playback(
                np.concatenate(
                    (transition_time, motion.times[1:] + transition_duration)
                ),
                np.vstack((transition_q, motion.positions[1:])),
            )
    else:
        motion, labels = _pose_plan_csv(
            args.input,
            args.rate,
            args.transition_duration,
            args.hold_duration,
            start_position,
            args.maximum_velocity,
            args.maximum_acceleration,
        )

    positions = motion.positions[:, :]
    _limit_report(model, positions, args.limit_margin)
    peak_velocity, peak_acceleration = _print_summary(motion, args.limit_margin)
    if np.max(peak_velocity) > args.maximum_velocity * 1.05:
        raise ValueError(
            f"preview peak velocity {np.max(peak_velocity):.4f} rad/s exceeds "
            f"the {args.maximum_velocity:g} rad/s limit"
        )
    if np.max(peak_acceleration) > args.maximum_acceleration * 1.05:
        raise ValueError(
            f"preview peak acceleration {np.max(peak_acceleration):.4f} rad/s^2 "
            f"exceeds the {args.maximum_acceleration:g} rad/s^2 limit"
        )
    if labels:
        counts = {label: labels.count(label) for label in sorted(set(labels))}
        print("Pose plan splits: " + ", ".join(f"{key}={value}" for key, value in counts.items()))
    print("Kinematic playback only: gravity, motors, and contacts are not simulated.")
    if start_position is None:
        print("No USB port supplied; playback begins at the first pose/trajectory point.")
    if args.gif is not None:
        _write_gif(
            model,
            data,
            qpos_addresses,
            motion,
            args.gif,
            args.fps,
            args.width,
            args.height,
            args.speed,
        )
        return 0
    if args.check_only:
        return 0
    print("Close the viewer window to exit.")

    # Use qpos coordinates directly and call only mj_forward. The model's
    # collision masks are disabled above, and mj_step is intentionally omitted.
    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.lookat[:] = (0.0, 0.0, 0.26)
            viewer.cam.distance = max(1.2, 3.2 * model.stat.extent)
            viewer.cam.azimuth = 135.0
            viewer.cam.elevation = -28.0
            started = time.monotonic()
            last_reported_finish = False
            while viewer.is_running():
                wall_elapsed = time.monotonic() - started
                elapsed = wall_elapsed * args.speed
                if args.loop:
                    elapsed %= motion.duration
                elif elapsed >= motion.duration:
                    elapsed = motion.duration
                    if not last_reported_finish:
                        print("Playback complete; holding final pose.")
                        last_reported_finish = True
                data.qpos[qpos_addresses] = motion.position_at(elapsed)
                mujoco.mj_forward(model, data)
                viewer.sync()
                time.sleep(max(0.0, 1.0 / args.rate - 0.001))
    except Exception as error:
        raise ValueError(
            "cannot open the interactive MuJoCo viewer; use --gif with "
            "MUJOCO_GL=egl in a headless Linux session"
        ) from error
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preview active-arm identification motion in MuJoCo."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    preview = commands.add_parser(
        "playback", help="play a trajectory CSV or static pose plan in the viewer"
    )
    preview.add_argument("input", type=Path)
    preview.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    preview.add_argument(
        "--format", choices=("auto", "trajectory", "pose-plan"), default="auto"
    )
    preview.add_argument("--rate", type=float, default=100.0, help="viewer update rate")
    preview.add_argument("--speed", type=float, default=1.0, help="playback time scale")
    preview.add_argument("--loop", action="store_true")
    preview.add_argument("--limit-margin", type=float, default=0.0)
    preview.add_argument("--transition-duration", type=float, default=3.0)
    preview.add_argument("--hold-duration", type=float, default=1.0)
    preview.add_argument("--maximum-velocity", type=float, default=0.25)
    preview.add_argument("--maximum-acceleration", type=float, default=0.5)
    preview.add_argument(
        "--port",
        help="read the current pose from this USB CDC port and save an execution snapshot",
    )
    preview.add_argument("--baud", type=int, default=115200)
    preview.add_argument("--state-timeout", type=float, default=2.0)
    preview.add_argument(
        "--check-only", action="store_true", help="validate and report without opening a viewer"
    )
    preview.add_argument("--gif", type=Path, help="render a headless animated GIF")
    preview.add_argument("--fps", type=float, default=10.0, help="GIF frame rate")
    preview.add_argument("--width", type=int, default=640, help="GIF frame width")
    preview.add_argument("--height", type=int, default=480, help="GIF frame height")
    preview.set_defaults(handler=playback)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        return args.handler(args)
    except (OSError, RuntimeError, ValueError, np.linalg.LinAlgError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
