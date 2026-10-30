"""Independent recording groups determine one shared parameter vector."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from ega_arm_host import identify_many


class FakeDynamics:
    def reference_basis(self):
        basis = np.eye(91, 3)
        return basis, {"rank": 3}

    def matrix(self, q, v, a):
        samples = len(q)
        rows = np.zeros((samples * 7, 91))
        for joint in range(7):
            rows[joint * samples:(joint + 1) * samples, :3] = np.column_stack(
                (q[:, joint], v[:, joint], a[:, joint])
            )
        return rows


class MultiIdentificationTests(unittest.TestCase):
    def test_group_holdout_and_frozen_validation(self):
        rng = np.random.default_rng(2026)
        truth = np.array([0.7, -0.2, 0.1])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [root / f"recording_{index}.csv" for index in range(4)]
            captures = {}
            for index, path in enumerate(paths):
                path.write_text(f"independent group {index}\n")
                q = rng.normal(size=(60, 7))
                v = rng.uniform(0.1, 0.5, size=(60, 7))
                a = rng.normal(size=(60, 7))
                tau = truth[0] * q + truth[1] * v + truth[2] * a
                captures[path.resolve()] = (
                    (np.arange(60) * 0.02, q, v, a, tau),
                    {"processed_samples": 60},
                )
            with patch.object(identify_many.arm_dynamics, "DynamicModel", FakeDynamics), \
                 patch.object(identify_many.arm_dynamics, "process_capture",
                              side_effect=lambda path: captures[Path(path).resolve()]):
                report = identify_many.fit_many(paths[:3], root / "fit", paths[3])
            model = json.loads((root / "fit/shared_full_model.json").read_text())
            np.testing.assert_allclose(model["parameters"][:3], truth, atol=1e-10)
            self.assertEqual(len(model["parameters"]), 91)
            self.assertEqual(report["chosen_ridge"], 0.0)
            self.assertLess(max(report["validation"]["rmse_nm"]), 1e-10)


if __name__ == "__main__":
    unittest.main()
