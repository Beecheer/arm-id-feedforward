"""Offline acquisition quality reports; no position-controller model is used."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def analyze_capture(capture: Path, stage: str, plan: Path) -> dict[str, object]:
    with capture.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if stage == "static":
        rows = [row for row in rows if row.get("phase") in ("position_capture", "gravity_capture")]
    if not rows:
        raise ValueError("没有可分析的采集窗口；原始文件已保留")
    columns = [
        *(f"q{j}_rad" for j in range(1, 8)),
        *(f"dq{j}_rad_s" for j in range(1, 8)),
        *(f"tau{j}_nm" for j in range(1, 8)),
    ]
    try:
        values = np.asarray([[float(row[key]) for key in columns] for row in rows])
        timestamps = np.asarray([int(row["device_time_us"]) for row in rows], dtype=np.int64)
        statuses = np.asarray([int(row["status"], 0) for row in rows])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("采集文件缺少有效的实测状态或反馈力矩") from error
    finite = np.all(np.isfinite(values), axis=1)
    expected_status = np.asarray([
        9 if stage == "static" and row.get("phase") == "gravity_capture" else 15
        for row in rows
    ])
    valid_status = (statuses & 15) == expected_status
    valid = finite & valid_status
    issues: list[str] = []
    if not np.all(valid):
        issues.append("存在无效状态或非有限数值")
    if len(timestamps) < 2 or np.any(np.diff(timestamps) <= 0):
        issues.append("设备时间戳不足、重复或倒退")
    report: dict[str, object] = {
        "kind": "capture_quality", "stage": stage,
        "capture": str(capture.resolve()),
        "capture_sha256": hashlib.sha256(capture.read_bytes()).hexdigest(),
        "sample_count": len(rows), "valid_sample_count": int(np.sum(valid)),
        "torque_columns": columns[14:], "torque_source": "raw_feedback", "issues": issues,
        "scope": "acquisition_quality_only; no physical parameter fit",
        "capture_mode": (
            "legacy_gravity_drag" if stage == "static" and all(row.get("phase") == "gravity_capture" for row in rows)
            else "position_control"
        ),
    }
    if np.any(valid):
        measured = values[valid]
        report["joint_summary"] = {
            "position_min_rad": np.min(measured[:, :7], axis=0).tolist(),
            "position_max_rad": np.max(measured[:, :7], axis=0).tolist(),
            "velocity_rms_rad_s": np.sqrt(np.mean(measured[:, 7:14] ** 2, axis=0)).tolist(),
            "torque_min_nm": np.min(measured[:, 14:], axis=0).tolist(),
            "torque_max_nm": np.max(measured[:, 14:], axis=0).tolist(),
        }
    if stage == "static":
        with plan.open(newline="", encoding="utf-8-sig") as stream:
            planned = {int(row["pose_index"]) for row in csv.DictReader(stream)}
        observed = {int(row["pose_index"]) for row in rows}
        if planned != observed:
            issues.append("静态姿态采集不完整")
        poses = []
        for pose in sorted(observed):
            selected = np.asarray([int(row["pose_index"]) == pose for row in rows]) & valid
            group = values[selected]
            if len(group) < 30:
                issues.append(f"姿态 {pose} 有效样本不足")
            if not len(group):
                continue
            q_std = np.std(group[:, :7], axis=0)
            dq_rms = np.sqrt(np.mean(group[:, 7:14] ** 2, axis=0))
            if np.max(q_std) > 0.01 or np.max(dq_rms) > 0.03:
                issues.append(f"姿态 {pose} 稳定性未通过")
            poses.append({
                "pose_index": pose, "sample_count": len(group),
                "position_median_rad": np.median(group[:, :7], axis=0).tolist(),
                "position_std_rad": q_std.tolist(), "velocity_rms_rad_s": dq_rms.tolist(),
                "torque_median_nm": np.median(group[:, 14:], axis=0).tolist(),
                "torque_std_nm": np.std(group[:, 14:], axis=0).tolist(),
            })
        report["poses"] = poses
    else:
        sequences = np.asarray([int(row["sample_sequence"]) for row in rows], dtype=np.int64)
        increments = np.diff(sequences) % (1 << 32)
        if np.any((increments == 0) | (increments >= (1 << 31))):
            issues.append("样本序号重复或倒退")
        gaps = increments[(increments > 1) & (increments < (1 << 31))] - 1
        lost = int(np.sum(gaps))
        loss_fraction = lost / (len(rows) + lost)
        if loss_fraction > 0.01:
            issues.append("反馈丢样比例超过默认阈值")
        if len(timestamps) > 1 and np.max(np.diff(timestamps)) > 100_000:
            issues.append("存在较长的反馈中断")
        metadata_path = capture.with_suffix(".json")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        with plan.open(newline="", encoding="utf-8-sig") as stream:
            plan_rows = list(csv.DictReader(stream))
        expected_duration = float(plan_rows[-1]["time_s"]) + metadata["transition_duration_s"]
        host_times = np.asarray([float(row["host_elapsed_s"]) for row in rows])
        coverage_tolerance = min(1.0, 0.02 + expected_duration * 0.01)
        if (
            not np.all(np.isfinite(host_times))
            or np.any(np.diff(host_times) < 0)
            or host_times[0] > coverage_tolerance
            or host_times[-1] < expected_duration - coverage_tolerance
        ):
            issues.append("动态轨迹采集时间覆盖不完整")
        report.update({
            "missing_samples": lost, "loss_fraction": loss_fraction,
            "expected_duration_s": expected_duration,
            "device_duration_s": float((timestamps[-1] - timestamps[0]) / 1e6),
            "host_first_sample_s": float(host_times[0]),
            "host_last_sample_s": float(host_times[-1]),
            "missing_tail_s": max(0.0, float(expected_duration-host_times[-1])),
            "duplicate_sample_sequences": int(np.sum(increments == 0)),
        })
    report["passed"] = not issues
    return report


def calibration_consistent(report: dict[str, object]) -> bool:
    """Conservative workflow defaults; detailed metrics remain in the report."""
    consistency = report["same_motor_consistency"]
    if not (
        consistency.get("available") and consistency.get("same_gain_sign")
        and consistency["relative_gain_spread"] <= 0.15
        and consistency["offset_spread_nm"] <= 0.05
    ):
        return False
    return all(
        fit["gain"] is not None and np.isfinite(fit["gain"])
        and abs(fit["gain"]) >= 1e-8
        and fit["offset_nm"] is not None and np.isfinite(fit["offset_nm"])
        and fit["model_span_nm"] >= 0.05
        and fit["rmse_nm"] is not None
        and fit["rmse_nm"] / fit["model_span_nm"] <= 0.10
        for fit in report["joints"][1:]
    )
