"""Fit one shared 91-parameter model from several independent recordings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import arm_collect, arm_dynamics


PARAMETER_ORDER = (
    "70 Pinocchio m,hx,hy,hz,Ixx,Ixy,Iyy,Ixz,Iyz,Izz; "
    "fv J1..J7; fc J1..J7; rotor J1..J7"
)


def _ridge_path(matrix: np.ndarray, target: np.ndarray, ridges) -> np.ndarray:
    u, singular, vt = np.linalg.svd(matrix, full_matrices=False)
    if not len(singular) or singular[0] <= 1e-14:
        raise ValueError("training regressor is empty or unobservable")
    keep = singular > singular[0] * 1e-10
    factors = np.zeros((len(ridges), len(singular)))
    for index, ridge in enumerate(ridges):
        factors[index, keep] = (
            singular[keep] / (singular[keep] ** 2 + ridge * singular[0] ** 2)
        )
    return (factors * (u.T @ target)[None, :]) @ vt


def fit_many(
    training: list[Path], output: Path, validation: Path | None = None,
) -> dict[str, object]:
    training = [Path(path).resolve() for path in training]
    output = Path(output)
    if len(training) < 2:
        raise ValueError("multi-recording fit requires at least two training files")
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    if len(set(training)) != len(training):
        raise ValueError("training files must be distinct")
    hashes = [arm_collect._sha256(path) for path in training]
    if len(set(hashes)) != len(hashes):
        raise ValueError("training recordings have duplicate content")
    if validation is not None:
        validation = Path(validation).resolve()
        if arm_collect._sha256(validation) in hashes:
            raise ValueError("validation recording duplicates training data")

    dynamics = arm_dynamics.DynamicModel()
    basis, basis_meta = dynamics.reference_basis()
    pieces = []
    processing = []
    groups = []
    for index, capture in enumerate(training):
        arrays, metadata = arm_dynamics.process_capture(capture)
        pieces.append(arrays)
        processing.append(metadata)
        groups.append(np.full(len(arrays[0]), index, dtype=int))
    q, v, a, tau = (
        np.concatenate([piece[field] for piece in pieces])
        for field in (1, 2, 3, 4)
    )
    group_ids = np.tile(np.concatenate(groups), 7)
    matrix = dynamics.matrix(q, v, a) @ basis
    target = tau.T.reshape(-1)
    sliding = np.abs(v).T.reshape(-1) >= 0.01
    scores = np.zeros(len(arm_dynamics.RIDGES))
    counts = np.zeros(len(arm_dynamics.RIDGES), dtype=int)
    for held_out in range(len(training)):
        fit = sliding & (group_ids != held_out)
        test = sliding & (group_ids == held_out)
        if fit.sum() < 2 * basis.shape[1] or not test.any():
            raise ValueError("recording group has too few moving samples")
        candidates = _ridge_path(matrix[fit], target[fit], arm_dynamics.RIDGES)
        error = matrix[test] @ candidates.T - target[test, None]
        scores += np.sum(error ** 2, axis=0)
        counts += len(error)
    cv_rmse = np.sqrt(scores / counts)
    choice = int(np.argmin(cv_rmse))
    ridge = arm_dynamics.RIDGES[choice]
    coefficients = _ridge_path(matrix[sliding], target[sliding], (ridge,))[0]
    theta = basis @ coefficients
    predicted = (dynamics.matrix(q, v, a) @ theta).reshape(7, -1).T
    report = {
        "kind": "arm_shared_multi_recording_identification", "version": 1,
        "urdf_sha256": arm_collect._sha256(arm_collect.URDF),
        "training": [
            {"capture": str(path), "sha256": digest, "processing": meta}
            for path, digest, meta in zip(training, hashes, processing)
        ],
        "basis": basis_meta,
        "leave_one_recording_out_rmse_nm": dict(
            zip((str(value) for value in arm_dynamics.RIDGES), cv_rmse.tolist())
        ),
        "chosen_ridge": ridge,
        "training_rmse_nm": np.sqrt(np.mean((predicted - tau) ** 2, axis=0)).tolist(),
        "validation": None,
        "scope": "shared full inverse dynamics; uses measured q and feedback torque; "
        "selection uses training recordings only",
    }
    if validation is not None:
        (_, qv, vv, av, observed), validation_processing = arm_dynamics.process_capture(validation)
        predicted_validation = (
            dynamics.matrix(qv, vv, av) @ theta
        ).reshape(7, -1).T
        report["validation"] = {
            "capture": str(validation),
            "sha256": arm_collect._sha256(validation),
            "processing": validation_processing,
            "rmse_nm": np.sqrt(np.mean((predicted_validation - observed) ** 2, axis=0)).tolist(),
            "bias_nm": np.mean(predicted_validation - observed, axis=0).tolist(),
        }
    model = {
        "kind": "arm_full_dynamic_identification", "version": 1,
        "urdf_sha256": report["urdf_sha256"],
        "parameter_order": PARAMETER_ORDER,
        "parameters": theta.tolist(),
        "provenance": {"training_capture_sha256": hashes,
                       "validation_capture_sha256": report["validation"]["sha256"]
                       if report["validation"] else None},
    }
    output.mkdir(parents=True)
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    (output / "shared_full_model.json").write_text(json.dumps(model, ensure_ascii=False, indent=2) + "\n")
    lines = ["# 共享完整动力学辨识", "", f"训练组：{len(training)}；整组留出所选岭系数：{ridge:g}。", "",
             "| 关节 | 训练 RMSE (N·m) | 独立验证 RMSE (N·m) |", "|---|---:|---:|"]
    for joint in range(7):
        held = report["validation"]["rmse_nm"][joint] if report["validation"] else None
        lines.append(f"| J{joint + 1} | {report['training_rmse_nm'][joint]:.4f} | "
                     f"{held:.4f} |" if held is not None else
                     f"| J{joint + 1} | {report['training_rmse_nm'][joint]:.4f} | — |")
    (output / "report.md").write_text("\n".join(lines) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, action="append", required=True,
                        help="repeat for each independent training CSV")
    parser.add_argument("--validation", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fit_many(args.train, args.output, args.validation)
    print(f"Shared model saved to {args.output / 'shared_full_model.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
