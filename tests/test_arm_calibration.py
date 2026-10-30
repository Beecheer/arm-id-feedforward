"""Offline checks for fitting and reusing torque calibration."""

import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from ega_arm_host import arm_collect
from ega_arm_host import arm_host
from ega_arm_host.arm_calibration import load_calibration


class CalibrationTests(unittest.TestCase):
    def test_j1_transfer_with_zero_gravity_and_direction_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            capture = root / "alignment.csv"
            output = root / "transferred.json"
            with capture.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow([
                    "stage", "pose_index", "phase", "host_elapsed_s", "status",
                    *(f"dq{j}_rad_s" for j in range(1, 8)),
                    *(f"tau{j}_nm" for j in range(1, 8)),
                    *(f"gravity_model_tau{j}_nm" for j in range(1, 8)),
                ])
                for pose, value in enumerate((-1.0, 0.0, 1.0), start=1):
                    writer.writerow([
                        "alignment", pose, "gravity_capture", pose, "0x0009",
                        *([0.0] * 7),
                        0.0, *([value] * 3),
                        1.9 * value + 0.02, 2.0 * value + 0.03, 2.1 * value + 0.04,
                        0.0, *([value] * 6),
                    ])
            for donor in ("5", "6", "7", "mean-567"):
                donor_output = output.with_name(f"{donor}.json")
                args = arm_collect.build_parser().parse_args([
                    "alignment-report", str(capture), "--output-json", str(donor_output),
                    "--j1-from", donor,
                ])
                with contextlib.redirect_stdout(io.StringIO()):
                    arm_collect.alignment_report(args)
                calibration = load_calibration(donor_output)
                document = json.loads(donor_output.read_text())
                expected_gain = {"5": 1.9, "6": 2.0, "7": 2.1, "mean-567": 2.0}[donor]
                expected_offset = {"5": -0.02, "6": -0.03, "7": -0.04, "mean-567": -0.03}[donor]
                self.assertAlmostEqual(calibration.gains[0], expected_gain)
                self.assertAlmostEqual(calibration.offsets[0], expected_offset)
                self.assertAlmostEqual(
                    calibration.apply([expected_gain * 0.4 + expected_offset] + [0.0] * 6)[0], 0.4
                )
                self.assertIsNone(document["joints"][0]["calibration_source"]["direct_fit"]["gain"])
                self.assertIsNone(document["joints"][0]["rmse_nm"])
                self.assertAlmostEqual(document["same_motor_consistency"]["relative_gain_spread"], 0.1)
                self.assertEqual(calibration.metadata()["joint_sources"][0]["method"], "same_motor_transfer")
            args = arm_collect.build_parser().parse_args([
                "alignment-report", str(capture), "--output-json", str(output),
            ])
            with contextlib.redirect_stdout(io.StringIO()):
                arm_collect.alignment_report(args)
            with self.assertRaisesRegex(ValueError, "J1"):
                load_calibration(output)

    def test_fit_and_reuse_in_both_csv_formats(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            capture = root / "alignment.csv"
            output = root / "calibration.json"
            gains = np.array([1.2, -2.0, 0.8, 3.0, 1.5, 0.7, 2.5])
            offsets = np.arange(7) * 0.03
            with capture.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow([
                    "stage", "pose_index", "phase", "host_elapsed_s", "status",
                    *(f"dq{j}_rad_s" for j in range(1, 8)),
                    *(f"tau{j}_nm" for j in range(1, 8)),
                    *(f"gravity_model_tau{j}_nm" for j in range(1, 8)),
                ])
                for pose, value in enumerate((-1.0, 0.0, 1.0), start=1):
                    writer.writerow([
                        "alignment", pose, "gravity_capture", pose, "0x0009",
                        *([0.0] * 7), *(gains * value + offsets), *([value] * 7),
                    ])
                # Moving windows must not bias the calibration fit.
                writer.writerow([
                    "alignment", 4, "move", 4, "0x000f",
                    *([0.0] * 7), *([999.0] * 7), *([1.0] * 7),
                ])
            args = arm_collect.build_parser().parse_args([
                "alignment-report", str(capture), "--output-json", str(output),
            ])
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(arm_collect.alignment_report(args), 0)
            calibration = load_calibration(output)
            feedback = tuple(gains * 0.4 + offsets)
            np.testing.assert_allclose(calibration.apply(feedback), [0.4] * 7)
            sample = arm_host.Sample(1, 1, 1000, 15, (0.0,) * 7, (0.0,) * 7, feedback)
            for static in (False, True):
                stream = io.StringIO()
                writer = csv.writer(stream)
                if static:
                    arm_collect._write_pose_sample(
                        writer, 0.0, "static", 1, "gravity_capture", sample,
                        None, None, calibration,
                    )
                else:
                    arm_host.write_sample(writer, 0.0, sample, None, calibration)
                row = next(csv.reader(io.StringIO(stream.getvalue())))
                np.testing.assert_allclose(np.array(row[-7:], dtype=float), [0.4] * 7)
                raw = row[-21:-14] if static else row[-14:-7]
                np.testing.assert_allclose(np.array(raw, dtype=float), feedback)
            document = json.loads(output.read_text())
            for invalid in (None, 0.0, float("nan")):
                document["joints"][0]["gain"] = invalid
                output.write_text(json.dumps(document))
                with self.assertRaisesRegex(ValueError, "J1"):
                    load_calibration(output)

    def test_static_and_dynamic_calibration_is_optional(self):
        args = arm_collect.build_parser().parse_args([
            "collect-poses", "--stage", "static", "--poses", "unused.csv",
            "--port", "unused", "--output", "unused.csv",
        ])
        self.assertIsNone(args.calibration)
        dynamic = arm_collect.build_parser().parse_args([
            "collect-dynamic", "--trajectory", "unused.csv", "--port", "unused",
            "--output", "unused.csv",
        ])
        self.assertIsNone(dynamic.calibration)


if __name__ == "__main__":
    unittest.main()
