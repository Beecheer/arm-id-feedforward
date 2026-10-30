"""Apply the affine feedback calibration fitted by alignment-report."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable


CALIBRATION_KIND = "arm_torque_calibration"
CALIBRATION_EQUATION = "feedback = gain * input + offset"


@dataclass(frozen=True)
class TorqueCalibration:
    path: Path
    sha256: str
    gains: tuple[float, ...]
    offsets: tuple[float, ...]
    sources: tuple[dict[str, object], ...]

    def apply(self, torque: Iterable[float]) -> tuple[float, ...]:
        feedback = tuple(torque)
        if len(feedback) != 7 or not all(math.isfinite(v) for v in feedback):
            raise ValueError("torque feedback must contain seven finite values")
        return tuple(
            (value - offset) / gain
            for value, gain, offset in zip(feedback, self.gains, self.offsets)
        )

    def metadata(self) -> dict[str, object]:
        return {
            "path": str(self.path.resolve()),
            "sha256": self.sha256,
            "equation": CALIBRATION_EQUATION,
            "gain": self.gains,
            "offset_nm": self.offsets,
            "joint_sources": self.sources,
            "application": "calibrated = (feedback - offset) / gain",
        }


def load_calibration(path: Path) -> TorqueCalibration:
    raw = path.read_bytes()
    document = json.loads(raw)
    if (
        not isinstance(document, dict)
        or document.get("kind") != CALIBRATION_KIND
        or document.get("version") != 1
        or document.get("equation") != CALIBRATION_EQUATION
    ):
        raise ValueError("expected a version 1 alignment-report calibration JSON")
    joints = document.get("joints")
    if not isinstance(joints, list) or len(joints) != 7:
        raise ValueError("calibration must contain seven joint fits")
    gains: list[float] = []
    offsets: list[float] = []
    sources: list[dict[str, object]] = []
    for index, joint in enumerate(joints, start=1):
        if not isinstance(joint, dict) or joint.get("joint") != index:
            raise ValueError("calibration joints must be ordered J1 through J7")
        gain, offset = joint.get("gain"), joint.get("offset_nm")
        if (
            isinstance(gain, bool)
            or not isinstance(gain, (int, float))
            or not math.isfinite(gain)
            or abs(gain) < 1e-8
            or isinstance(offset, bool)
            or not isinstance(offset, (int, float))
            or not math.isfinite(offset)
        ):
            raise ValueError(
                f"J{index} has no finite, invertible calibration; "
                "collect sufficient input-torque variation before using this fit"
            )
        gains.append(float(gain))
        offsets.append(float(offset))
        sources.append(joint.get("calibration_source", {"method": "direct_fit"}))
    return TorqueCalibration(
        path, hashlib.sha256(raw).hexdigest(), tuple(gains), tuple(offsets), tuple(sources)
    )
