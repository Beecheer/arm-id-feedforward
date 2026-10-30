#!/usr/bin/env python3
"""USB CDC host tool for the EGA active arm firmware."""

from __future__ import annotations

import argparse
import bisect
import csv
from dataclasses import dataclass
from datetime import datetime
import math
import json
import os
from pathlib import Path
import struct
import sys
import time
from typing import Iterable

import numpy as np
import serial
from serial.tools import list_ports

if __package__:
    from .arm_calibration import TorqueCalibration, load_calibration
else:
    from arm_calibration import TorqueCalibration, load_calibration


VERSION = 2
LINK_COMMAND = 0x01
LINK_FEEDBACK = 0x02
MAGIC = b"\x55\x53"
MAX_PAYLOAD = 128
COMMAND_STRUCT = struct.Struct("<BBI7f")
FEEDFORWARD_COMMAND_STRUCT = struct.Struct("<BBI21f")
ACK_STRUCT = struct.Struct("<BBIBB")
SAMPLE_STRUCT = struct.Struct("<BBIIQH21f")
COMMAND_SEQUENCE_MASK = 0xFFFFFFFF
COMMAND_SEQUENCE_BLOCK = 1 << 20
COMMAND_SEQUENCE_FILE = (
    Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state"))
    / "arm-id-host" / "command_sequence"
)

MODE_GRAVITY_DRAG = 0
MODE_DATA_COLLECTION = 1
MODE_READ_STATE = 2
MODE_INVERSE_DYNAMICS = 3
STATUS_INVERSE_DYNAMICS = 1 << 4
RESULT_POSITION_OUT_OF_RANGE = 3
RESULT_COMMAND_TIMEOUT = 7
MODE_NAMES = {
    MODE_GRAVITY_DRAG: "gravity_drag",
    MODE_DATA_COLLECTION: "data_collection",
    MODE_INVERSE_DYNAMICS: "inverse_dynamics",
}
ACK_NAMES = {
    0: "accepted",
    1: "invalid_mode",
    2: "invalid_position",
    3: "position_out_of_range",
    4: "motors_offline",
    5: "stale_sequence",
    6: "command_expired",
    7: "command_timeout",
    8: "invalid_feedforward",
}

# Protocol joint coordinates and limits for the supplied seven-axis profile.
ANGLE_LIMITS = (
    (-1.57, 1.57),
    (0.0, 3.1416),
    (-1.5708, 1.5708),
    (-3.1416, 0.0),
    (-3.1416, 3.1416),
    (-2.0944, 2.0944),
    (-3.1416, 3.1416),
)


def crc16(data: bytes) -> int:
    crc = 0xFFFF
    for value in data:
        crc ^= value
        for _ in range(8):
            if crc & 1:
                crc = (crc >> 1) ^ 0x8408
            else:
                crc >>= 1
    return crc & 0xFFFF


def make_link_frame(frame_type: int, sequence: int, payload: bytes) -> bytes:
    if len(payload) > MAX_PAYLOAD:
        raise ValueError(f"payload exceeds {MAX_PAYLOAD} bytes")
    header = MAGIC + bytes((frame_type, sequence & 0xFF, len(payload)))
    body = header[2:] + payload
    checksum = crc16(body)
    return header + payload + struct.pack("<H", checksum)


@dataclass(frozen=True)
class LinkMessage:
    frame_type: int
    sequence: int
    payload: bytes


class LinkParser:
    """Incremental parser for the firmware's 55 53 Link framing."""

    def __init__(self) -> None:
        self.buffer = bytearray()
        self.frame_errors = 0

    def feed(self, data: bytes) -> list[LinkMessage]:
        self.buffer.extend(data)
        messages: list[LinkMessage] = []

        while True:
            start = self.buffer.find(MAGIC)
            if start < 0:
                keep = 1 if self.buffer.endswith(MAGIC[:1]) else 0
                if len(self.buffer) > keep:
                    del self.buffer[: len(self.buffer) - keep]
                break
            if start:
                del self.buffer[:start]
            if len(self.buffer) < 5:
                break

            payload_size = self.buffer[4]
            if payload_size > MAX_PAYLOAD:
                self.frame_errors += 1
                del self.buffer[0]
                continue

            frame_size = 5 + payload_size + 2
            if len(self.buffer) < frame_size:
                break

            frame = bytes(self.buffer[:frame_size])
            received_crc = struct.unpack_from("<H", frame, frame_size - 2)[0]
            if crc16(frame[2:-2]) != received_crc:
                self.frame_errors += 1
                del self.buffer[0]
                continue

            messages.append(
                LinkMessage(frame[2], frame[3], frame[5 : 5 + payload_size])
            )
            del self.buffer[:frame_size]

        return messages


@dataclass(frozen=True)
class Ack:
    command_sequence: int
    active_mode: int
    result: int


@dataclass(frozen=True)
class Sample:
    sample_sequence: int
    target_sequence: int
    device_time_us: int
    status: int
    position: tuple[float, ...]
    velocity: tuple[float, ...]
    torque: tuple[float, ...]


def decode_feedback(payload: bytes) -> Ack | Sample | None:
    if len(payload) < 2 or payload[0] != VERSION:
        return None

    kind = payload[1]
    if kind == 1 and len(payload) == ACK_STRUCT.size:
        _, _, command_sequence, active_mode, result = ACK_STRUCT.unpack(payload)
        return Ack(command_sequence, active_mode, result)

    if kind == 2 and len(payload) == SAMPLE_STRUCT.size:
        fields = SAMPLE_STRUCT.unpack(payload)
        floats = fields[6:]
        return Sample(
            sample_sequence=fields[2],
            target_sequence=fields[3],
            device_time_us=fields[4],
            status=fields[5],
            position=tuple(floats[0:7]),
            velocity=tuple(floats[7:14]),
            torque=tuple(floats[14:21]),
        )

    return None


class ArmClient:
    def __init__(self, port: serial.Serial) -> None:
        self.port = port
        self.parser = LinkParser()
        self.link_sequence = 0
        self.command_sequence, self.sequence_remaining = reserve_command_sequences()
        self.recent_commands: list[int] = []
        self.command_targets: dict[int, tuple[float, ...]] = {}
        self.position_rejections = 0
        self.command_timeouts = 0

    def send_command(self, mode: int, position: Iterable[float]) -> int:
        if mode == MODE_INVERSE_DYNAMICS:
            raise ValueError("use send_feedforward(position, velocity, torque) for mode 3")
        positions = tuple(float(value) for value in position)
        if len(positions) != 7:
            raise ValueError("exactly seven joint positions are required")
        return self._send_target(mode, positions)

    def send_feedforward(
        self, position: Iterable[float], velocity: Iterable[float],
        torque: Iterable[float],
    ) -> int:
        """Send seven q/dq/tau targets in joint coordinates; no local gravity.

        Call at 200 Hz and keep reading ACK/sample messages. Firmware retains
        MIT gains, direction/zero mapping and encoding. No trajectory limits
        or feedback-error checks are imposed here.
        """
        vectors = []
        for name, values in (("position", position), ("velocity", velocity),
                             ("torque", torque)):
            vector = tuple(float(value) for value in values)
            if len(vector) != 7 or not all(math.isfinite(value) for value in vector):
                raise ValueError(f"{name} requires seven finite values")
            vectors.append(vector)
        return self._send_target(MODE_INVERSE_DYNAMICS, *vectors)

    def _send_target(
        self, mode: int, positions: tuple[float, ...],
        velocities: tuple[float, ...] = (), torques: tuple[float, ...] = (),
    ) -> int:
        if self.sequence_remaining == 0:
            self.command_sequence, self.sequence_remaining = (
                reserve_command_sequences()
            )
        command_sequence = self.command_sequence
        command_struct = (FEEDFORWARD_COMMAND_STRUCT if mode == MODE_INVERSE_DYNAMICS
                          else COMMAND_STRUCT)
        payload = command_struct.pack(
            VERSION, mode, command_sequence, *positions, *velocities, *torques,
        )
        frame = make_link_frame(LINK_COMMAND, self.link_sequence, payload)
        self.port.write(frame)
        self.link_sequence = (self.link_sequence + 1) & 0xFF
        self.command_sequence = (self.command_sequence + 1) & COMMAND_SEQUENCE_MASK
        self.sequence_remaining -= 1
        self.recent_commands.append(command_sequence)
        self.command_targets[command_sequence] = positions
        if len(self.recent_commands) > 64:
            expired = self.recent_commands[:32]
            del self.recent_commands[:32]
            for old_sequence in expired:
                self.command_targets.pop(old_sequence, None)
        return command_sequence

    def read_messages(self) -> list[Ack | Sample]:
        waiting = self.port.in_waiting
        data = self.port.read(max(1, min(waiting, 4096)))
        decoded: list[Ack | Sample] = []
        for message in self.parser.feed(data):
            if message.frame_type != LINK_FEEDBACK:
                continue
            feedback = decode_feedback(message.payload)
            if feedback is not None:
                decoded.append(feedback)
        return decoded


def reserve_command_sequences() -> tuple[int, int]:
    """Reserve a sequence block on disk so process restarts never reuse IDs."""
    try:
        next_sequence = int(COMMAND_SEQUENCE_FILE.read_text(encoding="ascii").strip())
    except FileNotFoundError:
        # Current Unix seconds are below 2^31 and let the first run move forward
        # from firmware instances that previously saw a small sequence value.
        next_sequence = int(time.time()) & COMMAND_SEQUENCE_MASK
    except (OSError, ValueError) as error:
        raise RuntimeError(
            f"cannot read command sequence state {COMMAND_SEQUENCE_FILE}: {error}"
        ) from error

    if not 0 <= next_sequence <= COMMAND_SEQUENCE_MASK:
        raise RuntimeError(f"invalid command sequence state in {COMMAND_SEQUENCE_FILE}")

    block_start = next_sequence
    next_unreserved = (block_start + COMMAND_SEQUENCE_BLOCK) & COMMAND_SEQUENCE_MASK
    temporary_path = COMMAND_SEQUENCE_FILE.with_suffix(".tmp")
    try:
        COMMAND_SEQUENCE_FILE.parent.mkdir(parents=True, exist_ok=True)
        temporary_path.write_text(f"{next_unreserved}\n", encoding="ascii")
        temporary_path.replace(COMMAND_SEQUENCE_FILE)
    except OSError as error:
        raise RuntimeError(
            f"cannot save command sequence state {COMMAND_SEQUENCE_FILE}: {error}"
        ) from error
    return block_start, COMMAND_SEQUENCE_BLOCK


def validate_position(position: Iterable[float]) -> tuple[float, ...]:
    """Validate a command numerically; planning limits do not reject commands."""
    values = tuple(float(value) for value in position)
    if len(values) != 7:
        raise ValueError("exactly seven joint positions are required")
    for index, value in enumerate(values):
        if not math.isfinite(value):
            raise ValueError(f"J{index + 1} position must be finite")
    return values


def validate_feedback_position(position: Iterable[float]) -> tuple[float, ...]:
    """Check feedback shape and numeric validity without applying position limits."""
    values = tuple(float(value) for value in position)
    if len(values) != 7:
        raise ValueError("exactly seven joint positions are required")
    for index, value in enumerate(values):
        if not math.isfinite(value):
            raise ValueError(f"J{index + 1} feedback position must be finite")
    return values


def command_start_for_plan(position: Iterable[float], plan: Path) -> tuple[float, ...]:
    """Choose a valid command anchor without altering measured feedback."""
    measured = validate_feedback_position(position)
    lower = np.asarray(ANGLE_LIMITS)[:, 0].copy()
    upper = np.asarray(ANGLE_LIMITS)[:, 1].copy()
    metadata_path = plan.with_suffix(".json")
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if "joint_lower_rad" in metadata and "joint_upper_rad" in metadata:
            planned_lower = np.asarray(metadata["joint_lower_rad"], dtype=float)
            planned_upper = np.asarray(metadata["joint_upper_rad"], dtype=float)
            if (
                planned_lower.shape != (7,) or planned_upper.shape != (7,)
                or not np.all(np.isfinite(planned_lower))
                or not np.all(np.isfinite(planned_upper))
            ):
                raise ValueError("plan bounds must contain seven finite joint limits")
            lower = np.maximum(lower, planned_lower)
            upper = np.minimum(upper, planned_upper)
    if np.any(lower > upper):
        raise ValueError("plan command bounds have an empty range")
    anchor = tuple(float(value) for value in np.clip(measured, lower, upper))
    if anchor != measured:
        print("起始反馈在规划范围外：仅将命令起点放入规划范围，实测反馈原样保存。")
    return anchor


def check_command_result(client: ArmClient, ack: Ack) -> bool:
    """Record recoverable firmware events without treating them as acceptance."""
    if ack.result == RESULT_POSITION_OUT_OF_RANGE:
        client.position_rejections += 1
        if client.position_rejections == 1:
            print("提示：固件拒绝了范围外的位置目标；上位机继续记录反馈并发送后续目标，不截断实测位置。")
        return False
    if ack.result == RESULT_COMMAND_TIMEOUT:
        client.command_timeouts += 1
        print("提示：固件报告位置命令超时并已回退；继续发送连续的规划目标，保留该事件。")
        return False
    if ack.result != 0:
        raise RuntimeError(
            f"firmware rejected command: {ACK_NAMES.get(ack.result, ack.result)}"
        )
    return True


def list_serial_ports() -> int:
    ports = sorted(list_ports.comports(), key=lambda item: item.device)
    if not ports:
        print("No serial ports found.")
        return 1
    for port in ports:
        description = port.description or "USB serial device"
        print(f"{port.device}\t{description}")
    return 0


def open_port(name: str, baud: int) -> serial.Serial:
    try:
        port = serial.Serial(name, baudrate=baud, timeout=0, write_timeout=0.5)
    except serial.SerialException as error:
        raise RuntimeError(f"cannot open {name}: {error}") from error
    port.reset_input_buffer()
    return port


def report_ack(ack: Ack) -> None:
    mode = MODE_NAMES.get(ack.active_mode, f"unknown({ack.active_mode})")
    result = ACK_NAMES.get(ack.result, f"unknown({ack.result})")
    print(f"ACK seq={ack.command_sequence} active={mode} result={result}")


def run_drag(args: argparse.Namespace) -> int:
    with open_port(args.port, args.baud) as port:
        client = ArmClient(port)
        sequence = client.send_command(MODE_GRAVITY_DRAG, (0.0,) * 7)
        print(f"Requested gravity drag mode (command seq {sequence}).")
        deadline = time.monotonic() + args.ack_timeout
        while time.monotonic() < deadline:
            for message in client.read_messages():
                if isinstance(message, Ack):
                    report_ack(message)
                    if message.command_sequence == sequence:
                        return 0 if message.result == 0 else 2
            time.sleep(0.002)
        print("No matching ACK received before timeout.", file=sys.stderr)
        return 3


def read_current_state(client: ArmClient, timeout: float) -> Sample:
    deadline = time.monotonic() + timeout
    next_request = 0.0
    requests: set[int] = set()
    accepted: set[int] = set()
    samples: dict[int, Sample] = {}
    messages_received = 0
    while time.monotonic() < deadline:
        now = time.monotonic()
        if now >= next_request:
            sequence = client.send_command(MODE_READ_STATE, (0.0,) * 7)
            requests.add(sequence)
            print(f"Requested current state (command seq {sequence}).")
            next_request = now + 0.5
        for message in client.read_messages():
            messages_received += 1
            if isinstance(message, Ack) and message.command_sequence in requests:
                report_ack(message)
                if message.result != 0:
                    raise RuntimeError(
                        "firmware rejected state request: "
                        f"{ACK_NAMES.get(message.result, message.result)}"
                    )
                accepted.add(message.command_sequence)
            elif (
                isinstance(message, Sample)
                and message.target_sequence in requests
            ):
                if not message.status & 0x0001:
                    raise RuntimeError("firmware returned an invalid state sample")
                validate_feedback_position(message.position)
                samples[message.target_sequence] = message
        for sequence in samples:
            if sequence in accepted:
                return samples[sequence]
        time.sleep(0.002)
    raise RuntimeError(
        f"状态读取超时：发送 {len(requests)} 次只读请求，收到 {messages_received} 条反馈、"
        f"{len(accepted)} 个匹配 ACK、{len(samples)} 个匹配状态。"
        "请检查 USB 通信，恢复后可重新运行原命令；已采集文件和预览计划保留。"
    )


def run_state(args: argparse.Namespace) -> int:
    with open_port(args.port, args.baud) as port:
        sample = read_current_state(ArmClient(port), args.ack_timeout)
        print("position_rad:", " ".join(f"{v:.6f}" for v in sample.position))
        print("velocity_rad_s:", " ".join(f"{v:.6f}" for v in sample.velocity))
        print("torque_nm:", " ".join(f"{v:.6f}" for v in sample.torque))
        print(f"status=0x{sample.status:04x} time_us={sample.device_time_us}")
        return 0


class Trajectory:
    def __init__(self, path: Path) -> None:
        self.times: list[float] = []
        self.positions: list[tuple[float, ...]] = []
        with path.open("r", newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            required = ["time_s", *(f"q{joint}" for joint in range(1, 8))]
            if reader.fieldnames is None or any(
                field not in reader.fieldnames for field in required
            ):
                raise ValueError(
                    "trajectory CSV needs columns: " + ",".join(required)
                )
            for row_number, row in enumerate(reader, start=2):
                try:
                    timestamp = float(row["time_s"])
                    position = validate_position(
                        row[f"q{joint}"] for joint in range(1, 8)
                    )
                except (TypeError, ValueError) as error:
                    raise ValueError(f"invalid trajectory row {row_number}: {error}") from error
                if not math.isfinite(timestamp) or timestamp < 0:
                    raise ValueError(f"invalid time_s on trajectory row {row_number}")
                if self.times and timestamp <= self.times[-1]:
                    raise ValueError("trajectory time_s values must increase")
                self.times.append(timestamp)
                self.positions.append(position)
        if not self.times:
            raise ValueError("trajectory CSV contains no data rows")

    @property
    def duration(self) -> float:
        return self.times[-1]

    def position_at(self, elapsed: float) -> tuple[float, ...]:
        if elapsed <= self.times[0]:
            return self.positions[0]
        right = bisect.bisect_right(self.times, elapsed)
        if right >= len(self.times):
            return self.positions[-1]
        left = right - 1
        fraction = (elapsed - self.times[left]) / (self.times[right] - self.times[left])
        return tuple(
            a + (b - a) * fraction
            for a, b in zip(self.positions[left], self.positions[right])
        )

    def prepend_transition(
        self, initial: tuple[float, ...], duration: float, rate: float
    ) -> None:
        steps = round(duration * rate)
        if (
            not math.isfinite(duration)
            or duration <= 0.0
            or steps < 3
            or not math.isclose(steps, duration * rate, rel_tol=0.0, abs_tol=1e-6)
        ):
            raise ValueError("start transition duration * rate must be an integer")

        final = self.positions[0]
        transition_times = [index / rate for index in range(steps + 1)]
        transition_positions = []
        for index in range(steps + 1):
            phase = index / steps
            blend = 10.0 * phase**3 - 15.0 * phase**4 + 6.0 * phase**5
            transition_positions.append(
                tuple(a + blend * (b - a) for a, b in zip(initial, final))
            )
        self.times = transition_times + [duration + value for value in self.times[1:]]
        self.positions = transition_positions + self.positions[1:]


def default_output_path() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path.cwd() / "captures" / f"manual_{stamp}.csv"


def write_sample(
    writer: csv.writer,
    start: float,
    sample: Sample,
    target_position: tuple[float, ...] | None,
    calibration: TorqueCalibration | None = None,
) -> None:
    writer.writerow(
        [
            f"{time.monotonic() - start:.6f}",
            sample.device_time_us,
            sample.sample_sequence,
            sample.target_sequence,
            f"0x{sample.status:04x}",
            *(target_position if target_position is not None else ("",) * 7),
            *sample.position,
            *sample.velocity,
            *sample.torque,
            *(calibration.apply(sample.torque) if calibration else ()),
        ]
    )


def run_collect(args: argparse.Namespace) -> int:
    calibration_path = getattr(args, "calibration", None)
    calibration = load_calibration(calibration_path) if calibration_path else None
    target = validate_position(args.target) if args.target is not None else None
    trajectory = Trajectory(args.trajectory) if args.trajectory else None
    if args.start_from_current and trajectory is None:
        raise ValueError("--start-from-current requires --trajectory")
    duration = args.duration
    if duration is not None and (not math.isfinite(duration) or duration <= 0):
        raise ValueError("duration must be positive")

    output_path = args.output or default_output_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open_port(args.port, args.baud) as port:
        client = ArmClient(port)
        if args.start_from_current:
            state = read_current_state(client, args.state_timeout)
            command_start = command_start_for_plan(state.position, args.trajectory)
            transition_duration = args.start_transition_duration
            if not math.isfinite(transition_duration) or transition_duration <= 0.0:
                raise ValueError("start transition duration must be finite and positive")
            velocity_limit = getattr(args, "start_velocity_limit", None)
            acceleration_limit = getattr(args, "start_acceleration_limit", None)
            if velocity_limit is not None and acceleration_limit is not None:
                if (
                    not math.isfinite(velocity_limit)
                    or not math.isfinite(acceleration_limit)
                    or velocity_limit <= 0.0
                    or acceleration_limit <= 0.0
                ):
                    raise ValueError("start transition limits must be finite and positive")
                delta = np.abs(
                    np.asarray(trajectory.positions[0])
                    - np.asarray(command_start)
                )
                transition_duration = max(
                    transition_duration,
                    1.15 * float(np.max(1.875 * delta / velocity_limit)),
                    1.15
                    * float(
                        np.max(np.sqrt(5.774 * delta / acceleration_limit))
                    ),
                )
                transition_steps = math.ceil(transition_duration * args.rate)
                transition_duration = transition_steps / args.rate
            args.actual_start_position = list(state.position)
            args.actual_command_start_position = list(command_start)
            args.actual_transition_duration = transition_duration
            trajectory.prepend_transition(
                command_start, transition_duration, args.rate
            )
            print(
                "Measured start position:",
                " ".join(f"{value:.6f}" for value in state.position),
            )
            print("Command start position:", " ".join(f"{value:.6f}" for value in command_start))
            print(
                f"Added {transition_duration:.2f} s smooth transition "
                f"to trajectory start."
            )
        if duration is None and trajectory is not None:
            duration = trajectory.duration + 1.0 / args.rate

        sample_count = 0
        interrupted = False
        try:
            with output_path.open("x", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(
                    [
                        "host_elapsed_s",
                        "device_time_us",
                        "sample_sequence",
                        "target_sequence",
                        "status",
                        *(f"target_q{joint}_rad" for joint in range(1, 8)),
                        *(f"q{joint}_rad" for joint in range(1, 8)),
                        *(f"dq{joint}_rad_s" for joint in range(1, 8)),
                        *(f"tau{joint}_nm" for joint in range(1, 8)),
                        *(f"calibrated_tau{joint}_nm" for joint in range(1, 8)
                          if calibration is not None),
                    ]
                )

                start = time.monotonic()
                send_period = 1.0 / args.rate
                next_send = start
                sent_count = 0
                sample_count = 0
                last_status = start
                handshake_complete = False
                last_sample = start
                feedback_missing = False
                print(f"Recording samples to {output_path}")
                print("Press Ctrl+C to stop and request gravity drag mode.")

                while duration is None or time.monotonic() - start < duration:
                    now = time.monotonic()
                    if now >= next_send:
                        elapsed = now - start
                        if trajectory is not None:
                            position = trajectory.position_at(elapsed)
                        else:
                            assert target is not None
                            position = target
                        client.send_command(MODE_DATA_COLLECTION, position)
                        sent_count += 1
                        next_send += send_period
                        if next_send < now:
                            next_send = now + send_period

                    for message in client.read_messages():
                        if isinstance(message, Ack):
                            if message.command_sequence not in client.recent_commands:
                                continue
                            accepted = check_command_result(client, message)
                            if not accepted:
                                # A rejection proves the link is responding;
                                # it does not mean the arm entered data mode.
                                handshake_complete = True
                                continue
                            report_ack(message)
                            if message.active_mode == MODE_DATA_COLLECTION:
                                handshake_complete = True
                        else:
                            validate_feedback_position(message.position)
                            handshake_complete = True
                            target_position = client.command_targets.get(
                                message.target_sequence
                            )
                            write_sample(writer, start, message, target_position, calibration)
                            sample_count += 1
                            if feedback_missing:
                                print("警告：反馈已恢复，继续记录；中断区间将保留在质量报告中。")
                                feedback_missing = False
                            last_sample = time.monotonic()

                    if now - last_sample > 1.0 and not feedback_missing:
                        print("警告：已超过 1 秒未收到状态反馈；仍继续发送轨迹，不自动中止。")
                        feedback_missing = True

                    if not handshake_complete and now - start > 2.0:
                        raise RuntimeError("no data-mode ACK or sample received")

                    if now - last_status >= 1.0:
                        print(
                            f"{sample_count} samples recorded; "
                            f"{sent_count} targets sent; "
                            f"{client.parser.frame_errors} bad frames"
                        )
                        stream.flush()
                        last_status = now
                    time.sleep(0.001)
        except KeyboardInterrupt:
            interrupted = True
            print("\nStop requested.")
        finally:
            try:
                sequence = client.send_command(MODE_GRAVITY_DRAG, (0.0,) * 7)
                print(f"Requested gravity drag mode (command seq {sequence}).")
                port.flush()
            except (serial.SerialException, OSError):
                pass

    print(f"Capture finished: {sample_count} samples saved to {output_path}")
    args.command_events = {
        "position_rejections": client.position_rejections,
        "command_timeouts": client.command_timeouts,
    }
    return 130 if interrupted else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="USB CDC control and data capture for the EGA active arm."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("ports", help="list available serial ports")

    drag = commands.add_parser("drag", help="request gravity compensation drag mode")
    drag.add_argument("--port", required=True, help="USB CDC serial port")
    drag.add_argument("--baud", type=int, default=115200, help="CDC line setting")
    drag.add_argument("--ack-timeout", type=float, default=2.0)

    state = commands.add_parser(
        "state", help="read current joint feedback without changing control mode"
    )
    state.add_argument("--port", required=True, help="USB CDC serial port")
    state.add_argument("--baud", type=int, default=115200, help="CDC line setting")
    state.add_argument("--ack-timeout", type=float, default=2.0)

    collect = commands.add_parser(
        "collect", help="stream position targets and record feedback"
    )
    collect.add_argument("--port", required=True, help="USB CDC serial port")
    collect.add_argument("--baud", type=int, default=115200, help="CDC line setting")
    source = collect.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--target", type=float, nargs=7, metavar="Q", help="fixed target in radians"
    )
    source.add_argument(
        "--trajectory", type=Path, help="CSV with time_s,q1,...,q7 columns"
    )
    collect.add_argument(
        "--rate", type=float, default=200.0, help="target command rate, max 300 Hz"
    )
    collect.add_argument("--duration", type=float, help="seconds; Ctrl+C if omitted")
    collect.add_argument(
        "--start-from-current",
        action="store_true",
        help="read feedback first and prepend a smooth transition to the trajectory",
    )
    collect.add_argument(
        "--start-transition-duration",
        type=float,
        default=10.0,
        help="seconds for the transition from current feedback to trajectory start",
    )
    collect.add_argument(
        "--state-timeout",
        type=float,
        default=2.0,
        help="seconds to wait for the current-state response",
    )
    collect.add_argument("--output", type=Path, help="CSV output path")
    collect.add_argument("--calibration", type=Path, help="alignment-report calibration JSON")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        if args.command == "ports":
            return list_serial_ports()
        if args.command == "drag":
            return run_drag(args)
        if args.command == "state":
            return run_state(args)
        if not math.isfinite(args.rate) or args.rate <= 0 or args.rate > 300:
            raise ValueError("rate must be in the range (0, 300] Hz")
        return run_collect(args)
    except (OSError, RuntimeError, ValueError, serial.SerialException) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
