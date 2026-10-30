#!/usr/bin/env python3
"""Generate and execute the active-arm identification data collections."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import pinocchio as pin

if __package__:
    from . import arm_host
    from .arm_calibration import (
        CALIBRATION_EQUATION,
        CALIBRATION_KIND,
        TorqueCalibration,
        load_calibration,
    )
else:
    import arm_host
    from arm_calibration import (
        CALIBRATION_EQUATION,
        CALIBRATION_KIND,
        TorqueCalibration,
        load_calibration,
    )


JOINT_COUNT = 7
URDF = Path(__file__).with_name("models") / "controller" / "robot.urdf"
REFERENCE_GRAVITY_MODEL = URDF.with_name("reference_gravity.json")
REQUIRED_FEEDBACK = 1 << 0
DATA_MODE = 1 << 1
TARGET_VALID = 1 << 2
COMMANDS_QUEUED = 1 << 3
POSE_COLUMNS = tuple(f"q{index}_rad" for index in range(1, JOINT_COUNT + 1))
TRAJECTORY_COLUMNS = tuple(f"q{index}" for index in range(1, JOINT_COUNT + 1))


def _joint_array(values: list[float], label: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.shape != (JOINT_COUNT,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{label} must contain seven finite joint values")
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_metadata(output: Path, document: dict[str, object]) -> None:
    metadata_path = output.with_suffix(".json")
    with metadata_path.open("x", encoding="utf-8") as stream:
        json.dump(document, stream, indent=2)
        stream.write("\n")


def _load_urdf(path: Path) -> pin.Model:
    if not path.is_file():
        raise ValueError(f"URDF does not exist: {path}")
    model = pin.buildModelFromUrdf(str(path))
    expected = tuple(
        f"Joint_{a}{b}"
        for a, b in ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7))
    )
    if model.nq != JOINT_COUNT or model.nv != JOINT_COUNT:
        raise ValueError(f"expected a fixed-base seven-joint URDF, got {model.nq}/{model.nv}")
    if tuple(model.names[1:]) != expected:
        raise ValueError(f"URDF joint order must be {expected}")
    limits = np.column_stack((model.lowerPositionLimit, model.upperPositionLimit))
    if not np.allclose(limits, np.asarray(arm_host.ANGLE_LIMITS), atol=1e-5, rtol=0.0):
        raise ValueError("URDF limits do not match host protocol limits")
    return model


def _read_reference_gravity_model(
    path: Path = REFERENCE_GRAVITY_MODEL,
) -> dict[str, np.ndarray | float]:
    """Load a host-owned reference model for optional alignment and planning."""
    document = json.loads(path.read_text(encoding="utf-8"))
    shapes = {
        "parent_position": (JOINT_COUNT, 3),
        "center_of_mass": (JOINT_COUNT, 3),
        "joint_axis": (JOINT_COUNT, 3),
        "link_mass": (JOINT_COUNT,),
        "torque_scale": (JOINT_COUNT,),
    }
    model = {}
    for name, shape in shapes.items():
        values = np.asarray(document[name], dtype=float)
        if values.shape != shape or not np.all(np.isfinite(values)):
            raise ValueError(f"invalid reference gravity field {name}")
        model[name] = values
    model["gravity"] = float(document["gravity"])
    if not math.isfinite(model["gravity"]):
        raise ValueError("invalid reference gravity constant")
    return model


def reference_gravity(q: np.ndarray, model: dict[str, np.ndarray | float]) -> np.ndarray:
    """Evaluate a host-owned reference gravity model at a measured pose."""
    if q.shape != (JOINT_COUNT,) or not np.all(np.isfinite(q)):
        raise ValueError("gravity input must contain seven finite positions")
    parent = model["parent_position"]
    com = model["center_of_mass"]
    axes = model["joint_axis"]
    masses = model["link_mass"]
    scales = model["torque_scale"]
    gravity = float(model["gravity"])
    assert isinstance(parent, np.ndarray) and isinstance(com, np.ndarray)
    assert isinstance(axes, np.ndarray) and isinstance(masses, np.ndarray)
    assert isinstance(scales, np.ndarray)
    origin = np.zeros((JOINT_COUNT, 3))
    axis_world = np.zeros((JOINT_COUNT, 3))
    rotation_world = np.eye(3)
    position_world = np.zeros(3)
    torque = np.zeros(JOINT_COUNT)
    for index in range(JOINT_COUNT):
        position_world += rotation_world @ parent[index]
        origin[index] = position_world
        local_axis = axes[index]
        axis_world[index] = rotation_world @ local_axis
        x, y, z = local_axis
        skew = np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
        angle = q[index]
        joint_rotation = (
            np.eye(3)
            + math.sin(angle) * skew
            + (1.0 - math.cos(angle)) * (skew @ skew)
        )
        rotation_world = rotation_world @ joint_rotation
        center = position_world + rotation_world @ com[index]
        force = np.asarray([0.0, 0.0, masses[index] * gravity])
        for joint in range(index + 1):
            torque[joint] += axis_world[joint] @ np.cross(
                center - origin[joint], force
            )
    return torque * scales


def _write_pose_plan(
    path: Path, poses: np.ndarray, labels: list[str] | None = None,
    move_durations: list[float] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["pose_index", "stage", *POSE_COLUMNS, *(["move_duration_s"] if move_durations is not None else [])])
        for index, pose in enumerate(poses):
            writer.writerow(
                [index + 1, labels[index] if labels else "pose", *(f"{v:.8f}" for v in pose),
                 *([f"{move_durations[index]:.6f}"] if move_durations is not None else [])]
            )


def _read_pose_durations(path: Path) -> list[float]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        durations = [float(row.get("move_duration_s", "0")) for row in csv.DictReader(stream)]
    if not all(math.isfinite(value) and value >= 0 for value in durations):
        raise ValueError("pose transition durations must be finite and nonnegative")
    return durations


def _read_pose_plan(path: Path, expected_stage: str | None = None) -> np.ndarray:
    poses: list[list[float]] = []
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or any(name not in reader.fieldnames for name in POSE_COLUMNS):
            raise ValueError(f"{path}: pose plan needs columns {', '.join(POSE_COLUMNS)}")
        for row_number, row in enumerate(reader, start=2):
            if expected_stage is not None and "stage" in reader.fieldnames:
                if row.get("stage") != expected_stage:
                    raise ValueError(
                        f"{path}: row {row_number} is for stage {row.get('stage')!r}, "
                        f"not {expected_stage!r}"
                    )
            try:
                pose = [float(row[name]) for name in POSE_COLUMNS]
            except (TypeError, ValueError) as error:
                raise ValueError(f"{path}: invalid pose on row {row_number}") from error
            if not np.all(np.isfinite(pose)):
                raise ValueError(f"{path}: non-finite pose on row {row_number}")
            poses.append(pose)
    if not poses:
        raise ValueError(f"{path}: no poses found")
    return np.asarray(poses, dtype=float)


def generate_pose_plan(args: argparse.Namespace) -> int:
    if (
        args.poses < 1
        or args.candidates < args.poses
        or not math.isfinite(args.limit_margin)
        or args.limit_margin < 0.0
    ):
        raise ValueError("pose count, candidate count, or limit margin is invalid")
    model = _load_urdf(args.urdf)
    lower = model.lowerPositionLimit + args.limit_margin
    upper = model.upperPositionLimit - args.limit_margin
    if np.any(lower >= upper):
        raise ValueError("limit margin leaves an empty range")
    rng = np.random.default_rng(args.seed)
    candidates = rng.uniform(lower, upper, size=(args.candidates, JOINT_COUNT))
    selected = [int(rng.integers(len(candidates)))]
    min_distance = np.linalg.norm(candidates - candidates[selected[0]], axis=1)
    while len(selected) < args.poses:
        min_distance[np.asarray(selected)] = -1.0
        choice = int(np.argmax(min_distance))
        selected.append(choice)
        min_distance = np.minimum(
            min_distance, np.linalg.norm(candidates - candidates[choice], axis=1)
        )
    poses = candidates[np.asarray(selected)]
    _write_pose_plan(args.output, poses, [args.stage] * len(poses))
    _write_metadata(
        args.output,
        {
            "kind": "pose_plan",
            "stage": args.stage,
            "pose_csv": str(args.output.resolve()),
            "pose_csv_sha256": _sha256(args.output),
            "urdf": str(args.urdf.resolve()),
            "urdf_sha256": _sha256(args.urdf),
            "seed": args.seed,
            "candidate_count": args.candidates,
            "limit_margin_rad": args.limit_margin,
        },
    )
    print(f"Wrote {len(poses)} {args.stage} poses to {args.output}")
    print(f"Controller URDF: {args.urdf}; interior limit margin: {args.limit_margin:g} rad")
    return 0


def _quintic(start: np.ndarray, end: np.ndarray, phase: float) -> np.ndarray:
    s = 10.0 * phase**3 - 15.0 * phase**4 + 6.0 * phase**5
    return start + s * (end - start)


def _move_duration(
    start: np.ndarray,
    end: np.ndarray,
    minimum: float,
    velocity_limit: float,
    acceleration_limit: float,
) -> float:
    delta = np.abs(end - start)
    if (
        not np.all(np.isfinite(delta))
        or not math.isfinite(minimum)
        or not math.isfinite(velocity_limit)
        or not math.isfinite(acceleration_limit)
        or minimum < 0.0
        or velocity_limit <= 0.0
        or acceleration_limit <= 0.0
    ):
        raise ValueError("motion velocity and acceleration limits must be positive")
    velocity_time = float(np.max(1.875 * delta / velocity_limit))
    acceleration_time = float(np.max(np.sqrt(5.774 * delta / acceleration_limit)))
    return max(minimum, 1.15 * velocity_time, 1.15 * acceleration_time)


def _write_trajectory(path: Path, times: np.ndarray, positions: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["time_s", *TRAJECTORY_COLUMNS])
        for timestamp, q in zip(times, positions):
            writer.writerow([f"{timestamp:.6f}", *(f"{value:.8f}" for value in q)])


def make_dynamic_trajectory(args: argparse.Namespace) -> int:
    if (
        args.duration <= 0.0
        or args.rate <= 0.0
        or args.rate > 200.0
        or args.harmonics < 1
        or not math.isfinite(args.duration)
        or not math.isfinite(args.rate)
    ):
        raise ValueError("duration, rate, and harmonic count must be positive")
    if args.duration * args.rate < 3:
        raise ValueError("duration * rate must contain at least three samples")
    model = _load_urdf(args.urdf)
    center = _joint_array(args.center, "center")
    amplitude = _joint_array(args.amplitudes, "amplitudes")
    if (
        np.any(amplitude <= 0.0)
        or not math.isfinite(args.maximum_velocity)
        or args.maximum_velocity <= 0.0
        or not math.isfinite(args.maximum_acceleration)
        or args.maximum_acceleration <= 0.0
    ):
        raise ValueError("amplitudes and motion limits must be positive")
    lower = model.lowerPositionLimit + args.limit_margin
    upper = model.upperPositionLimit - args.limit_margin
    if (
        not math.isfinite(args.limit_margin)
        or args.limit_margin < 0.0
        or np.any(lower >= upper)
    ):
        raise ValueError("invalid joint limit margin")
    if np.any(center - amplitude < lower) or np.any(center + amplitude > upper):
        raise ValueError("center +/- amplitudes exceed URDF limits with margin")
    count = int(round(args.duration * args.rate))
    if not math.isclose(count, args.duration * args.rate, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError("duration * rate must be an integer")
    time_s = np.arange(count + 1, dtype=float) / args.rate
    envelope = np.sin(np.pi * time_s / args.duration) ** 2
    rng = np.random.default_rng(args.seed)
    q = np.empty((len(time_s), JOINT_COUNT))
    for joint in range(JOINT_COUNT):
        harmonics = np.arange(1, args.harmonics + 1, dtype=float)
        phases = rng.uniform(0.0, 2.0 * np.pi, size=args.harmonics)
        weights = 1.0 / np.sqrt(harmonics)
        signal = np.sum(
            weights[:, None]
            * np.sin(
                2.0 * np.pi * harmonics[:, None] * time_s[None, :] / args.duration
                + phases[:, None]
            ),
            axis=0,
        )
        signal /= max(float(np.max(np.abs(signal))), 1e-12)
        q[:, joint] = center[joint] + amplitude[joint] * envelope * signal
    dt = 1.0 / args.rate
    velocity = np.gradient(q, dt, axis=0, edge_order=2)
    acceleration = np.gradient(velocity, dt, axis=0, edge_order=2)
    scale = min(
        1.0,
        args.maximum_velocity / max(float(np.max(np.abs(velocity))), 1e-12),
        math.sqrt(args.maximum_acceleration / max(float(np.max(np.abs(acceleration))), 1e-12)),
    )
    if scale < 1.0:
        q = center + scale * (q - center)
    _write_trajectory(args.output, time_s, q)
    _write_metadata(
        args.output,
        {
            "kind": "dynamic_trajectory",
            "trajectory_csv": str(args.output.resolve()),
            "trajectory_sha256": _sha256(args.output),
            "urdf": str(args.urdf.resolve()),
            "urdf_sha256": _sha256(args.urdf),
            "seed": args.seed,
            "duration_s": args.duration,
            "sample_rate_hz": args.rate,
            "harmonics": args.harmonics,
            "center_rad": center.tolist(),
            "requested_amplitudes_rad": amplitude.tolist(),
            "applied_amplitude_scale": scale,
            "maximum_velocity_rad_s": args.maximum_velocity,
            "maximum_acceleration_rad_s2": args.maximum_acceleration,
            "limit_margin_rad": args.limit_margin,
        },
    )
    print(f"Wrote dynamic trajectory to {args.output}: {len(q)} points, {args.duration:g} s")
    print(f"Amplitude scale after velocity/acceleration caps: {scale:.4f}")
    print(f"Peak |dq|={np.max(np.abs(np.gradient(q, dt, axis=0))):.4f} rad/s")
    print(
        f"Peak |ddq|={np.max(np.abs(np.gradient(np.gradient(q, dt, axis=0), dt, axis=0))):.4f} rad/s^2"
    )
    return 0


def _wait_ack(client: arm_host.ArmClient, sequence: int, timeout: float) -> arm_host.Ack:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for message in client.read_messages():
            if isinstance(message, arm_host.Ack) and message.command_sequence == sequence:
                arm_host.check_command_result(client, message)
                return message
        time.sleep(0.001)
    raise RuntimeError("firmware command acknowledgement timed out")


def _read_state(client: arm_host.ArmClient, timeout: float) -> arm_host.Sample:
    return arm_host.read_current_state(client, timeout)


def _write_pose_sample(
    writer: csv.writer,
    elapsed: float,
    stage: str,
    pose_index: int,
    phase: str,
    sample: arm_host.Sample,
    target: np.ndarray | tuple[float, ...] | None,
    gravity: np.ndarray | None,
    calibration: TorqueCalibration | None = None,
) -> None:
    writer.writerow(
        [
            stage,
            pose_index,
            phase,
            f"{elapsed:.6f}",
            sample.sample_sequence,
            sample.target_sequence,
            sample.device_time_us,
            f"0x{sample.status:04x}",
            *(target.tolist() if isinstance(target, np.ndarray) else target or [""] * JOINT_COUNT),
            *sample.position,
            *sample.velocity,
            *sample.torque,
            *(gravity.tolist() if gravity is not None else [""] * JOINT_COUNT),
            *(calibration.apply(sample.torque) if calibration else ()),
        ]
    )


@contextmanager
def _gravity_on_exit(client: arm_host.ArmClient):
    try:
        yield
    finally:
        try:
            client.send_command(arm_host.MODE_GRAVITY_DRAG, (0.0,) * JOINT_COUNT)
        except (OSError, RuntimeError):
            pass


def collect_pose_sequence(args: argparse.Namespace) -> int:
    if args.stage == "alignment" and args.calibration is not None:
        raise ValueError("alignment uses raw feedback; omit --calibration")
    calibration = load_calibration(args.calibration) if args.calibration else None
    if not math.isfinite(args.rate) or args.rate <= 0.0 or args.rate > 200.0:
        raise ValueError("command rate must be in (0, 200] Hz")
    if (
        not math.isfinite(args.capture_seconds)
        or not math.isfinite(args.state_timeout)
        or args.capture_seconds <= 0.0
        or args.state_timeout <= 0.0
    ):
        raise ValueError("capture duration and state timeout must be positive")
    if (
        args.minimum_samples < 1
        or not math.isfinite(args.maximum_position_std)
        or args.maximum_position_std <= 0.0
        or not math.isfinite(args.maximum_velocity_rms)
        or args.maximum_velocity_rms <= 0.0
        or not math.isfinite(args.settle_window)
        or args.settle_window <= 0.0
        or not math.isfinite(args.settle_timeout)
        or args.settle_timeout <= 0.0
        or not math.isfinite(args.velocity_tolerance)
        or args.velocity_tolerance <= 0.0
        or not math.isfinite(args.minimum_move_duration)
        or args.minimum_move_duration < 0.0
        or not math.isfinite(args.state_timeout)
        or args.state_timeout <= 0.0
    ):
        raise ValueError("sample, stability, or settling limits must be positive")
    model = _load_urdf(args.urdf)
    poses = _read_pose_plan(args.poses, args.stage)
    planned_durations = _read_pose_durations(args.poses)
    margin = args.limit_margin
    if not math.isfinite(margin) or margin < 0.0:
        raise ValueError("limit margin cannot be negative")
    if np.any(poses < model.lowerPositionLimit + margin) or np.any(
        poses > model.upperPositionLimit - margin
    ):
        raise ValueError("a target pose violates URDF limits with the requested margin")
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    gravity_model = _read_reference_gravity_model() if args.stage == "alignment" else None
    quality_notes: list[dict[str, object]] = []

    with arm_host.open_port(args.port, args.baud) as port:
        client = arm_host.ArmClient(port)
        if args.stage == "alignment":
            gravity_sequence = client.send_command(
                arm_host.MODE_GRAVITY_DRAG, (0.0,) * JOINT_COUNT
            )
            _wait_ack(client, gravity_sequence, args.state_timeout)
        initial = _read_state(client, args.state_timeout)
        actual_start = np.asarray(initial.position)
        command_start = np.asarray(arm_host.command_start_for_plan(actual_start, args.poses))

        with _gravity_on_exit(client), output.open("x", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                [
                    "stage", "pose_index", "phase", "host_elapsed_s", "sample_sequence",
                    "target_sequence", "device_time_us", "status",
                    *(f"target_q{j}_rad" for j in range(1, 8)),
                    *(f"q{j}_rad" for j in range(1, 8)),
                    *(f"dq{j}_rad_s" for j in range(1, 8)),
                    *(f"tau{j}_nm" for j in range(1, 8)),
                    *(f"gravity_model_tau{j}_nm" for j in range(1, 8)),
                    *(f"calibrated_tau{j}_nm" for j in range(1, 8)
                      if calibration is not None),
                ]
            )
            session_start = time.monotonic()
            current = command_start.copy()
            latest_position = actual_start.copy()
            for pose_index, target in enumerate(poses, start=1):
                duration = _move_duration(
                    current,
                    target,
                    args.minimum_move_duration,
                    args.maximum_velocity,
                    args.maximum_acceleration,
                )
                # Keep offline timing and command continuity; do not add
                # runtime torque gates or rebuild targets from tracking error.
                duration = max(duration, planned_durations[pose_index - 1])
                steps = max(2, math.ceil(duration * args.rate))
                duration = steps / args.rate
                # Continue from the previous COMMAND target. Restarting from
                # tracking error can send rejected targets and expire the
                # firmware heartbeat even though USB traffic continues.
                mode_entry = client.send_command(
                    arm_host.MODE_DATA_COLLECTION, current
                )
                if args.stage == "alignment":
                    _wait_ack(client, mode_entry, args.state_timeout)
                start_time = time.monotonic()
                next_send = start_time
                while True:
                    now = time.monotonic()
                    if now < next_send:
                        time.sleep(next_send - now)
                        now = time.monotonic()
                    phase = min(1.0, (now - start_time) / duration)
                    command_position = _quintic(current, target, phase)
                    client.send_command(
                        arm_host.MODE_DATA_COLLECTION, command_position
                    )
                    next_send += 1.0 / args.rate
                    # Motion samples are intentionally retained for traceability.
                    for message in client.read_messages():
                        if isinstance(message, arm_host.Sample):
                            measured_q = np.asarray(message.position)
                            latest_position = measured_q
                            _write_pose_sample(
                                writer,
                                time.monotonic() - session_start,
                                args.stage,
                                pose_index,
                                "move",
                                message,
                                client.command_targets.get(message.target_sequence),
                                reference_gravity(np.asarray(message.position), gravity_model)
                                if gravity_model is not None
                                else None,
                                calibration,
                            )
                        elif isinstance(message, arm_host.Ack):
                            arm_host.check_command_result(client, message)
                    if phase >= 1.0:
                        break

                # Settling concerns measured motion, not reaching the commanded pose.
                stable_since: float | None = None
                settle_deadline = time.monotonic() + args.settle_timeout
                next_settle_send = 0.0
                while time.monotonic() < settle_deadline:
                    now = time.monotonic()
                    if now >= next_settle_send:
                        client.send_command(arm_host.MODE_DATA_COLLECTION, target)
                        next_settle_send = now + 1.0 / args.rate
                    for message in client.read_messages():
                        if isinstance(message, arm_host.Ack):
                            arm_host.check_command_result(client, message)
                        if not isinstance(message, arm_host.Sample):
                            continue
                        q = np.asarray(message.position)
                        latest_position = q
                        dq = np.asarray(message.velocity)
                        stable = float(np.max(np.abs(dq))) <= args.velocity_tolerance
                        now = time.monotonic()
                        if stable:
                            stable_since = stable_since or now
                            if now - stable_since >= args.settle_window:
                                break
                        else:
                            stable_since = None
                        _write_pose_sample(
                            writer,
                            now - session_start,
                            args.stage,
                            pose_index,
                            "settle",
                            message,
                            client.command_targets.get(message.target_sequence),
                            reference_gravity(q, gravity_model) if gravity_model is not None else None,
                            calibration,
                        )
                    if stable_since is not None and time.monotonic() - stable_since >= args.settle_window:
                        break
                    time.sleep(0.001)
                else:
                    print(f"Pose {pose_index}: still moving after settle wait; recording measured state anyway.")
                    quality_notes.append({"pose_index": pose_index, "note": "settle_wait_expired"})

                position_capture = args.stage == "static"
                if not position_capture:
                    drag_sequence = client.send_command(
                        arm_host.MODE_GRAVITY_DRAG, (0.0,) * JOINT_COUNT
                    )
                    _wait_ack(client, drag_sequence, args.state_timeout)
                phase_name = "position_capture" if position_capture else "gravity_capture"
                print(f"Pose {pose_index}/{len(poses)}: {phase_name} for {args.capture_seconds:g} s")
                capture_deadline = time.monotonic() + args.capture_seconds
                next_capture_send = 0.0
                capture_q: list[np.ndarray] = []
                capture_dq: list[np.ndarray] = []
                while time.monotonic() < capture_deadline:
                    if position_capture:
                        now = time.monotonic()
                        if now >= next_capture_send:
                            client.send_command(arm_host.MODE_DATA_COLLECTION, target)
                            next_capture_send = now + 1.0 / args.rate
                        samples = []
                        for message in client.read_messages():
                            if isinstance(message, arm_host.Ack):
                                arm_host.check_command_result(client, message)
                            elif isinstance(message, arm_host.Sample):
                                samples.append(message)
                    else:
                        samples = [_read_state(client, args.state_timeout)]
                    for sample in samples:
                        measured_q = np.asarray(sample.position)
                        latest_position = measured_q
                        capture_q.append(measured_q)
                        capture_dq.append(np.asarray(sample.velocity))
                        _write_pose_sample(
                            writer,
                            time.monotonic() - session_start,
                            args.stage,
                            pose_index,
                            phase_name,
                            sample,
                            target,
                            reference_gravity(measured_q, gravity_model)
                            if gravity_model is not None
                            else None,
                            calibration,
                        )
                    stream_flush = getattr(stream, "flush", None)
                    if stream_flush is not None:
                        stream_flush()
                    if position_capture:
                        time.sleep(0.001)
                if len(capture_q) < args.minimum_samples:
                    print(f"Pose {pose_index}: few capture samples; retaining data and continuing.")
                    quality_notes.append({"pose_index": pose_index, "note": "few_samples"})
                if not capture_q:
                    # An empty window provides no statistics; later poses may
                    # still be useful and the offline report records coverage.
                    current = (
                        target.copy() if position_capture
                        else np.asarray(arm_host.command_start_for_plan(latest_position, args.poses))
                    )
                    continue
                pose_std = np.std(np.asarray(capture_q), axis=0)
                velocity_rms = np.sqrt(
                    np.mean(np.square(np.asarray(capture_dq)), axis=0)
                )
                if float(np.max(pose_std)) > args.maximum_position_std:
                    joint = int(np.argmax(pose_std)) + 1
                    print(f"Pose {pose_index}: position varied during capture; retaining measured states.")
                    quality_notes.append({"pose_index": pose_index, "note": "position_variation", "joint": joint})
                if float(np.max(velocity_rms)) > args.maximum_velocity_rms:
                    joint = int(np.argmax(velocity_rms)) + 1
                    print(f"Pose {pose_index}: capture contains motion; retaining measured states.")
                    quality_notes.append({"pose_index": pose_index, "note": "velocity_variation", "joint": joint})
                current = (
                    target.copy() if position_capture
                    else np.asarray(arm_host.command_start_for_plan(latest_position, args.poses))
                )

    print(f"Saved {args.stage} pose sequence to {output}")
    metadata: dict[str, object] = {
        "stage": args.stage,
        "capture_csv": str(output.resolve()),
        "pose_plan_csv": str(args.poses.resolve()),
        "urdf": str(args.urdf.resolve()),
        "urdf_sha256": _sha256(args.urdf),
        "pose_plan_sha256": _sha256(args.poses),
        "start_position_rad": actual_start.tolist(),
        "command_start_position_rad": command_start.tolist(),
        "transition_start_policy": "continuous command targets; feedback retained unchanged",
        "command_events": {
            "position_rejections": client.position_rejections,
            "command_timeouts": client.command_timeouts,
        },
        "quality_notes": quality_notes,
        "tracking_policy": "use measured states; target error does not gate capture",
        "torque_source": "calibrated_feedback" if calibration else "raw_feedback",
        "capture_mode": "position_control" if args.stage == "static" else "gravity_drag",
        "options": {
            "command_rate_hz": args.rate,
            "capture_seconds_per_pose": args.capture_seconds,
            "maximum_velocity_rad_s": args.maximum_velocity,
            "maximum_acceleration_rad_s2": args.maximum_acceleration,
            "settle_velocity_tolerance_rad_s": args.velocity_tolerance,
            "stationary_max_position_std_rad": args.maximum_position_std,
            "stationary_max_velocity_rms_rad_s": args.maximum_velocity_rms,
        },
    }
    if args.stage == "alignment":
        metadata["reference_gravity_model"] = str(REFERENCE_GRAVITY_MODEL.resolve())
        metadata["reference_gravity_model_sha256"] = _sha256(REFERENCE_GRAVITY_MODEL)
    if calibration is not None:
        metadata["torque_calibration"] = calibration.metadata()
    _write_metadata(output, metadata)
    return 0


def collect_dynamic(args: argparse.Namespace) -> int:
    calibration = load_calibration(args.calibration) if args.calibration else None
    # Preview and execution are independent. Execution reads its own start pose
    # and computes the transition duration from that feedback.
    if not math.isfinite(args.rate) or args.rate <= 0.0 or args.rate > 200.0:
        raise ValueError("dynamic command rate must be in (0, 200] Hz")
    if not math.isfinite(args.transition_duration) or args.transition_duration <= 0.0:
        raise ValueError("transition duration must be positive")
    transition_steps = round(args.transition_duration * args.rate)
    if transition_steps < 3 or not math.isclose(
        transition_steps,
        args.transition_duration * args.rate,
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise ValueError("transition duration * command rate must be an integer >= 3")
    if (
        not math.isfinite(args.limit_margin)
        or args.limit_margin < 0.0
        or not math.isfinite(args.maximum_velocity)
        or args.maximum_velocity <= 0.0
        or not math.isfinite(args.maximum_acceleration)
        or args.maximum_acceleration <= 0.0
    ):
        raise ValueError("dynamic trajectory limits are invalid")
    model = _load_urdf(URDF)
    trajectory = arm_host.Trajectory(args.trajectory)
    if len(trajectory.times) < 3 or not math.isclose(
        trajectory.times[0], 0.0, rel_tol=0.0, abs_tol=1e-6
    ):
        raise ValueError("dynamic trajectory needs at least three points starting at time zero")
    q = np.asarray(trajectory.positions, dtype=float)
    if np.any(q < model.lowerPositionLimit + args.limit_margin) or np.any(
        q > model.upperPositionLimit - args.limit_margin
    ):
        raise ValueError("dynamic trajectory violates controller URDF limit margin")
    time_s = np.asarray(trajectory.times, dtype=float)
    velocity = np.gradient(q, time_s, axis=0, edge_order=2)
    acceleration = np.gradient(velocity, time_s, axis=0, edge_order=2)
    if float(np.max(np.abs(velocity))) > args.maximum_velocity * 1.05:
        raise ValueError("dynamic trajectory exceeds maximum velocity")
    if float(np.max(np.abs(acceleration))) > args.maximum_acceleration * 1.10:
        raise ValueError("dynamic trajectory exceeds maximum acceleration")
    host_args = argparse.Namespace(
        target=None,
        trajectory=args.trajectory,
        start_from_current=True,
        duration=None,
        output=args.output,
        port=args.port,
        baud=args.baud,
        state_timeout=args.state_timeout,
        start_transition_duration=args.transition_duration,
        start_velocity_limit=args.maximum_velocity,
        start_acceleration_limit=args.maximum_acceleration,
        rate=args.rate,
        calibration=args.calibration,
    )
    result = arm_host.run_collect(host_args)
    if result == 0:
        _write_metadata(
            args.output,
            {
                "stage": "dynamic",
                "capture_mode": "position_control",
                "torque_source": "calibrated_feedback" if calibration else "raw_feedback",
                **({"torque_calibration": calibration.metadata()} if calibration else {}),
                "capture_csv": str(args.output.resolve()),
                "trajectory_csv": str(args.trajectory.resolve()),
                "trajectory_sha256": _sha256(args.trajectory),
                "start_position_rad": getattr(host_args, "actual_start_position", None),
                "command_start_position_rad": getattr(host_args, "actual_command_start_position", None),
                "command_events": getattr(host_args, "command_events", {}),
                "command_rate_hz": args.rate,
                "transition_duration_s": getattr(
                    host_args,
                    "actual_transition_duration",
                    args.transition_duration,
                ),
                "urdf": str(URDF.resolve()),
                "urdf_sha256": _sha256(URDF),
                "limit_margin_rad": args.limit_margin,
                "maximum_velocity_rad_s": args.maximum_velocity,
                "maximum_acceleration_rad_s2": args.maximum_acceleration,
            },
        )
    return result


def collect_manual(args: argparse.Namespace) -> int:
    if (
        not math.isfinite(args.duration)
        or not math.isfinite(args.rate)
        or args.duration <= 0.0
        or args.rate <= 0.0
        or args.rate > 100.0
    ):
        raise ValueError("duration and sample rate must be positive")
    if (
        not math.isfinite(args.state_timeout)
        or args.state_timeout <= 0.0
    ):
        raise ValueError("state timeout must be positive")
    gravity_model = _read_reference_gravity_model()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with arm_host.open_port(args.port, args.baud) as port:
        client = arm_host.ArmClient(port)
        sequence = client.send_command(
            arm_host.MODE_GRAVITY_DRAG, (0.0,) * JOINT_COUNT
        )
        _wait_ack(client, sequence, args.state_timeout)
        start = time.monotonic()
        try:
            with args.output.open("x", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(
                    [
                        "stage", "pose_index", "phase", "host_elapsed_s", "sample_sequence",
                        "target_sequence", "device_time_us", "status",
                        *(f"target_q{j}_rad" for j in range(1, 8)),
                        *(f"q{j}_rad" for j in range(1, 8)),
                        *(f"dq{j}_rad_s" for j in range(1, 8)),
                        *(f"tau{j}_nm" for j in range(1, 8)),
                        *(f"gravity_model_tau{j}_nm" for j in range(1, 8)),
                    ]
                )
                next_sample = start
                while time.monotonic() - start < args.duration:
                    now = time.monotonic()
                    if now < next_sample:
                        time.sleep(next_sample - now)
                    sample = _read_state(client, args.state_timeout)
                    q = np.asarray(sample.position)
                    if sample.status & (REQUIRED_FEEDBACK | COMMANDS_QUEUED) != (
                        REQUIRED_FEEDBACK | COMMANDS_QUEUED
                    ):
                        raise RuntimeError("invalid feedback during manual gravity capture")
                    _write_pose_sample(
                        writer,
                        time.monotonic() - start,
                        "alignment",
                        0,
                        "manual_gravity",
                        sample,
                        None,
                        reference_gravity(q, gravity_model),
                    )
                    stream.flush()
                    next_sample += 1.0 / args.rate
        except KeyboardInterrupt:
            print("Manual alignment capture stopped by operator.")
        finally:
            try:
                sequence = client.send_command(
                    arm_host.MODE_GRAVITY_DRAG, (0.0,) * JOINT_COUNT
                )
                _wait_ack(client, sequence, args.state_timeout)
            except (OSError, RuntimeError):
                pass
    print(f"Saved manual alignment capture to {args.output}")
    _write_metadata(
        args.output,
        {
            "stage": "alignment",
            "acquisition": "manual_gravity_drag",
            "capture_csv": str(args.output.resolve()),
            "urdf": str(args.urdf.resolve()),
            "urdf_sha256": _sha256(args.urdf),
            "reference_gravity_model": str(REFERENCE_GRAVITY_MODEL.resolve()),
            "reference_gravity_model_sha256": _sha256(REFERENCE_GRAVITY_MODEL),
            "duration_s": args.duration,
            "sample_rate_hz": args.rate,
        },
    )
    return 0


def alignment_report(args: argparse.Namespace) -> int:
    if (
        not math.isfinite(args.maximum_velocity)
        or args.maximum_velocity <= 0.0
        or not math.isfinite(args.minimum_model_span)
        or args.minimum_model_span < 0.0
    ):
        raise ValueError("alignment report limits are invalid")
    with args.capture.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {
            "stage",
            "pose_index",
            "phase",
            "host_elapsed_s",
            "status",
            *(f"dq{joint}_rad_s" for joint in range(1, 8)),
            *(f"tau{joint}_nm" for joint in range(1, 8)),
            *(f"gravity_model_tau{joint}_nm" for joint in range(1, 8)),
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            missing = sorted(required - set(reader.fieldnames or ()))
            raise ValueError(f"capture is missing columns: {', '.join(missing)}")
        bins: dict[tuple[str, int], list[list[float]]] = {}
        for row_number, row in enumerate(reader, start=2):
            if row["stage"] != "alignment" or row["phase"] not in (
                "gravity_capture",
                "manual_gravity",
            ):
                continue
            try:
                status = int(row["status"], 0)
                elapsed = float(row["host_elapsed_s"])
                velocity = np.asarray(
                    [float(row[f"dq{joint}_rad_s"]) for joint in range(1, 8)]
                )
                measured = np.asarray(
                    [float(row[f"tau{joint}_nm"]) for joint in range(1, 8)]
                )
                predicted = np.asarray(
                    [float(row[f"gravity_model_tau{joint}_nm"]) for joint in range(1, 8)]
                )
                pose_index = int(row["pose_index"])
            except (TypeError, ValueError) as error:
                raise ValueError(f"row {row_number}: invalid alignment sample") from error
            if status & (REQUIRED_FEEDBACK | COMMANDS_QUEUED) != (
                REQUIRED_FEEDBACK | COMMANDS_QUEUED
            ) or status & DATA_MODE:
                continue
            if (
                not np.all(np.isfinite(velocity))
                or not math.isfinite(elapsed)
                or np.max(np.abs(velocity)) > args.maximum_velocity
                or not np.all(np.isfinite(measured))
                or not np.all(np.isfinite(predicted))
            ):
                continue
            # Pose captures get equal weight regardless of duration. Hand-guided
            # samples are reduced to one representative per second.
            key = (row["phase"], pose_index if pose_index else int(elapsed))
            bins.setdefault(key, []).append(
                np.concatenate((measured, predicted)).tolist()
            )
    if not bins:
        raise ValueError("no valid stationary gravity-mode samples found")
    points = np.asarray([np.median(values, axis=0) for values in bins.values()])
    measured = points[:, :JOINT_COUNT]
    predicted = points[:, JOINT_COUNT:]
    if len(points) < 3:
        raise ValueError(f"need at least three independent pose/time bins, found {len(points)}")

    report: dict[str, object] = {
        "kind": CALIBRATION_KIND,
        "version": 1,
        "equation": CALIBRATION_EQUATION,
        "capture": str(args.capture),
        "capture_sha256": _sha256(args.capture),
        "sample_bins": len(points),
        "maximum_velocity_rad_s": args.maximum_velocity,
        "joints": [],
    }
    joint_reports: list[dict[str, object]] = []
    print(f"Alignment report: {args.capture} ({len(points)} pose/time bins)")
    print("Joint  model span Nm  gain  offset Nm  RMSE Nm  correlation")
    for joint in range(JOINT_COUNT):
        x = predicted[:, joint]
        y = measured[:, joint]
        span = float(np.ptp(x))
        if span < args.minimum_model_span:
            result = {
                "joint": joint + 1,
                "model_span_nm": span,
                "gain": None,
                "offset_nm": None,
                "rmse_nm": None,
                "correlation": None,
            }
            print(f"J{joint + 1:<2} {span:13.4f}  insufficient model variation")
        else:
            design = np.column_stack((x, np.ones(len(x))))
            gain, offset = np.linalg.lstsq(design, y, rcond=None)[0]
            residual = y - design @ np.asarray((gain, offset))
            correlation = (
                float(np.corrcoef(x, y)[0, 1])
                if np.std(x) > 1e-12 and np.std(y) > 1e-12
                else None
            )
            result = {
                "joint": joint + 1,
                "model_span_nm": span,
                "gain": float(gain),
                "offset_nm": float(offset),
                "rmse_nm": float(np.sqrt(np.mean(np.square(residual)))),
                "correlation": correlation,
            }
            print(
                f"J{joint + 1:<2} {span:13.4f} {gain:6.3f} "
                f"{offset:10.4f} {result['rmse_nm']:8.4f} "
                f"{correlation if correlation is not None else 'n/a'}"
            )
        joint_reports.append(result)
    # Compare fits in motor coordinates: J1 is reversed, J5/J6/J7 are normal.
    # The affine gain is unchanged by reversing both input and feedback;
    # the offset changes sign.
    donor_reports = joint_reports[4:7]
    valid_donors = all(
        fit["gain"] is not None
        and math.isfinite(fit["gain"])
        and abs(fit["gain"]) >= 1e-8
        and fit["offset_nm"] is not None
        and math.isfinite(fit["offset_nm"])
        for fit in donor_reports
    )
    consistency: dict[str, object] = {"joints": [5, 6, 7], "available": valid_donors}
    if valid_donors:
        gains = np.asarray([fit["gain"] for fit in donor_reports])
        offsets = np.asarray([fit["offset_nm"] for fit in donor_reports])
        gain_spread = float(np.ptp(gains))
        offset_spread = float(np.ptp(offsets))
        mean_abs_gain = float(np.mean(np.abs(gains)))
        consistency.update({
            "gain_spread": gain_spread,
            "relative_gain_spread": gain_spread / mean_abs_gain,
            "offset_spread_nm": offset_spread,
            "same_gain_sign": bool(np.all(gains > 0) or np.all(gains < 0)),
            "maximum_rmse_nm": max(fit["rmse_nm"] for fit in donor_reports),
        })
        print(
            f"J5/J6/J7 consistency: gain spread {gain_spread:.4f} "
            f"({gain_spread / mean_abs_gain:.1%}), "
            f"offset spread {offset_spread:.4f} Nm; inspect individual residuals."
        )
    else:
        print("J5/J6/J7 consistency unavailable: one or more fits lack input variation.")
    report["same_motor_consistency"] = consistency

    j1_from = args.j1_from
    if j1_from is not None:
        donors = donor_reports if j1_from == "mean-567" else [joint_reports[int(j1_from) - 1]]
        if any(
            fit["gain"] is None or not math.isfinite(fit["gain"])
            or abs(fit["gain"]) < 1e-8
            or fit["offset_nm"] is None or not math.isfinite(fit["offset_nm"])
            for fit in donors
        ):
            raise ValueError("J1 transfer requires finite, invertible donor fits")
        donor_gains = np.asarray([fit["gain"] for fit in donors])
        if not (np.all(donor_gains > 0) or np.all(donor_gains < 0)):
            raise ValueError("J1 donor gains have inconsistent signs")
        direct_fit = joint_reports[0].copy()
        joint_reports[0].update({
            "gain": float(np.mean(donor_gains)),
            "offset_nm": -float(np.mean([fit["offset_nm"] for fit in donors])),
            "rmse_nm": None,
            "correlation": None,
            "calibration_source": {
                "method": "same_motor_transfer",
                "donor_joints": [fit["joint"] for fit in donors],
                "aggregation": "arithmetic_mean",
                "offset_direction_factor": -1,
                "assumption": "shared motor-coordinate gain and offset",
                "direct_fit": direct_fit,
            },
        })
        print(
            f"J1 uses {j1_from}: gain={joint_reports[0]['gain']:.4f}, "
            f"offset={joint_reports[0]['offset_nm']:.4f} Nm "
            "(offset sign reversed for J1 direction; transferred, not measured)."
        )
    report["joints"] = joint_reports
    print("These are alignment diagnostics, not absolute torque calibration.")
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2)
            stream.write("\n")
        print(f"Saved report to {args.output_json}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate and collect active-arm data; legacy alignment is optional."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    poses = commands.add_parser("pose-plan", help="generate safe multi-pose plans")
    poses.add_argument("--stage", choices=("alignment", "static"), required=True)
    poses.add_argument("--urdf", type=Path, default=URDF)
    poses.add_argument("--poses", type=int, default=12)
    poses.add_argument("--candidates", type=int, default=2000)
    poses.add_argument("--seed", type=int, default=51)
    poses.add_argument("--limit-margin", type=float, default=0.0)
    poses.add_argument("--output", type=Path, required=True)
    poses.set_defaults(handler=generate_pose_plan)

    dynamic = commands.add_parser("dynamic-plan", help="generate a bounded multisine trajectory")
    dynamic.add_argument("--urdf", type=Path, default=URDF)
    dynamic.add_argument("--output", type=Path, required=True)
    dynamic.add_argument("--duration", type=float, default=120.0)
    dynamic.add_argument("--rate", type=float, default=100.0)
    dynamic.add_argument("--harmonics", type=int, default=8)
    dynamic.add_argument("--seed", type=int, default=61)
    dynamic.add_argument("--center", type=float, nargs=7, default=(0.0, 0.9, 0.0, -0.8, 0.0, 0.25, 0.0))
    dynamic.add_argument("--amplitudes", type=float, nargs=7, default=(0.4, 0.35, 0.4, 0.3, 0.45, 0.45, 0.45))
    dynamic.add_argument("--limit-margin", type=float, default=0.0)
    dynamic.add_argument("--maximum-velocity", type=float, default=0.25)
    dynamic.add_argument("--maximum-acceleration", type=float, default=0.5)
    dynamic.set_defaults(handler=make_dynamic_trajectory)

    collect = commands.add_parser("collect-poses", help="execute alignment or static pose plan")
    collect.add_argument("--stage", choices=("alignment", "static"), required=True)
    collect.add_argument("--poses", type=Path, required=True)
    collect.add_argument("--port", required=True)
    collect.add_argument("--baud", type=int, default=115200)
    collect.add_argument("--urdf", type=Path, default=URDF)
    collect.add_argument("--rate", type=float, default=100.0)
    collect.add_argument("--minimum-move-duration", type=float, default=3.0)
    collect.add_argument("--maximum-velocity", type=float, default=0.25)
    collect.add_argument("--maximum-acceleration", type=float, default=0.5)
    collect.add_argument("--position-tolerance", type=float, default=0.025, help=argparse.SUPPRESS)
    collect.add_argument("--velocity-tolerance", type=float, default=0.025)
    collect.add_argument("--settle-window", type=float, default=0.5)
    collect.add_argument("--settle-timeout", type=float, default=3.0)
    collect.add_argument("--capture-seconds", type=float, default=3.0)
    collect.add_argument("--minimum-samples", type=int, default=30)
    collect.add_argument("--maximum-position-std", type=float, default=0.01)
    collect.add_argument("--maximum-velocity-rms", type=float, default=0.03)
    collect.add_argument("--state-timeout", type=float, default=1.0)
    collect.add_argument("--limit-margin", type=float, default=0.0)
    collect.add_argument("--output", type=Path, required=True)
    collect.add_argument("--calibration", type=Path, help="optional legacy alignment-report JSON")
    collect.set_defaults(handler=collect_pose_sequence)

    dynamic_collect = commands.add_parser("collect-dynamic", help="execute a previewed trajectory")
    dynamic_collect.add_argument("--trajectory", type=Path, required=True)
    dynamic_collect.add_argument("--port", required=True)
    dynamic_collect.add_argument("--baud", type=int, default=115200)
    dynamic_collect.add_argument("--rate", type=float, default=200.0)
    dynamic_collect.add_argument("--transition-duration", type=float, default=24.0)
    dynamic_collect.add_argument("--limit-margin", type=float, default=0.0)
    dynamic_collect.add_argument("--maximum-velocity", type=float, default=0.25)
    dynamic_collect.add_argument("--maximum-acceleration", type=float, default=0.5)
    dynamic_collect.add_argument("--state-timeout", type=float, default=2.0)
    dynamic_collect.add_argument("--output", type=Path, required=True)
    dynamic_collect.add_argument("--calibration", type=Path, help="optional legacy alignment-report JSON")
    dynamic_collect.set_defaults(handler=collect_dynamic)

    manual = commands.add_parser(
        "collect-manual", help="record hand-guided gravity-compensation alignment data"
    )
    manual.add_argument("--port", required=True)
    manual.add_argument("--baud", type=int, default=115200)
    manual.add_argument("--duration", type=float, default=120.0)
    manual.add_argument("--rate", type=float, default=50.0)
    manual.add_argument("--state-timeout", type=float, default=1.0)
    manual.add_argument("--output", type=Path, required=True)
    manual.set_defaults(handler=collect_manual)

    report = commands.add_parser(
        "alignment-report", help="compare reference gravity input with measured effort"
    )
    report.add_argument("capture", type=Path)
    report.add_argument("--maximum-velocity", type=float, default=0.03)
    report.add_argument("--minimum-model-span", type=float, default=0.05)
    report.add_argument("--output-json", type=Path)
    report.add_argument(
        "--j1-from", choices=("5", "6", "7", "mean-567"),
        help="explicitly transfer a same-model motor fit to J1 after checking consistency",
    )
    report.set_defaults(handler=alignment_report)

    parser.set_defaults(handler=None)
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
