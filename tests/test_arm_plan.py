"""Model-based checks for stage 0 information and geometry screening."""

import unittest
from pathlib import Path
import tempfile

import numpy as np
import pinocchio as pin

from arm_id_feedforward import arm_collect
from arm_id_feedforward import arm_plan
from arm_id_feedforward import arm_sim


class PlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = arm_collect._load_urdf(arm_collect.URDF.resolve())

    def test_static_regressor_reconstructs_nominal_gravity(self):
        poses = np.asarray([arm_plan.HOME, arm_plan.HOME + 0.1])
        blocks = arm_plan.regressor_blocks(self.model, poses)
        parameters = np.concatenate([
            inertia.toDynamicParameters() for inertia in self.model.inertias[1:]
        ])
        data = self.model.createData()
        for q, block in zip(poses, blocks):
            expected = pin.computeGeneralizedGravity(self.model, data, q)
            np.testing.assert_allclose(block @ parameters, expected, atol=1e-12)
        self.assertFalse(np.allclose(blocks[0], blocks[1]))

    def test_information_selection_covers_observable_gravity_space(self):
        rng = np.random.default_rng(2026)
        poses = rng.uniform(
            self.model.lowerPositionLimit + arm_plan.LIMIT_MARGIN,
            self.model.upperPositionLimit - arm_plan.LIMIT_MARGIN, size=(128, 7),
        )
        projected, basis = arm_plan.observable_blocks(arm_plan.regressor_blocks(self.model, poses))
        chosen = arm_plan.select_information(projected, 16)
        self.assertEqual(len(set(chosen)), 16)
        report = arm_plan.information_report(projected[chosen])
        baseline = arm_plan.information_report(projected[:16])
        self.assertLess(basis["reference_rank"], 70)
        self.assertEqual(report["rank"], basis["reference_rank"])
        self.assertGreater(report["regularized_logdet"], baseline["regularized_logdet"])
        np.testing.assert_array_equal(chosen, arm_plan.select_information(projected, 16))

    def test_affine_selection_covers_torque_range_without_zero_channels(self):
        torque = np.zeros((101, 7))
        torque[:, 1:6] = np.linspace(-1, 1, 101)[:, None]
        torque[:, 6] = 1e-6
        chosen = arm_plan.select_alignment(torque, 12)
        self.assertEqual(len(set(chosen)), 12)
        self.assertGreater(np.ptp(torque[chosen, 1]), 1.9)
        with self.assertRaises(ValueError):
            arm_plan.observable_blocks(np.zeros((4, 7, 70)))

    def test_geometry_rejects_colliding_pose_and_transition(self):
        screen = arm_plan.GeometryScreen(self.model, arm_collect.URDF.resolve())
        self.assertTrue(screen.clear(arm_plan.HOME))
        self.assertTrue(screen.path_clear(arm_plan.HOME, arm_plan.HOME + 0.1))
        rng = np.random.default_rng(0)
        candidates = rng.uniform(
            self.model.lowerPositionLimit, self.model.upperPositionLimit, size=(32, 7),
        )
        colliding = next(q for q in candidates if not screen.clear(q))
        self.assertFalse(screen.path_clear(arm_plan.HOME, colliding))

    def test_rated_torque_filters_j2_static_load_not_protocol_range(self):
        limits, metadata = arm_plan.motor_torque_limits()
        np.testing.assert_allclose(metadata["rated_torque_nm"], [0.8, 3.5, 3, 3, 0.8, 0.8, 0.8])
        self.assertAlmostEqual(limits[1], 2.8)
        scales = arm_collect._read_reference_gravity_model()["torque_scale"]
        torque = arm_plan.TorqueScreen(self.model, limits, scales)
        rng = np.random.default_rng(0)
        candidates = rng.uniform(
            self.model.lowerPositionLimit + arm_plan.LIMIT_MARGIN,
            self.model.upperPositionLimit - arm_plan.LIMIT_MARGIN, size=(128, 7),
        )
        overloaded = next(q for q in candidates if torque.effort(q)[1] > limits[1])
        self.assertFalse(torque.pose_feasible(overloaded))
        # Holding load cannot be repaired by a longer transition.
        self.assertIsNone(torque.transition(arm_plan.HOME, overloaded, minimum=100))

    def test_inertial_overload_retimes_instead_of_rejecting_pose(self):
        limits, _ = arm_plan.motor_torque_limits()
        scales = arm_collect._read_reference_gravity_model()["torque_scale"]
        start = arm_plan.HOME
        end = start + np.array([0.5, 0, 0, 0, 0, 0, 0])
        baseline = arm_plan.TorqueScreen(self.model, limits, scales).transition(start, end)
        tight = limits.copy()
        tight[0] = baseline[1][0] * 0.5
        torque = arm_plan.TorqueScreen(self.model, tight, scales)
        self.assertTrue(torque.pose_feasible(end))
        retimed = torque.transition(start, end)
        self.assertGreater(retimed[0], baseline[0])
        self.assertTrue(np.all(retimed[1] <= tight))

    def test_preview_preserves_saved_transition_time(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = Path(directory) / "poses.csv"
            poses = np.asarray([arm_plan.HOME, arm_plan.HOME + 0.01])
            arm_collect._write_pose_plan(plan, poses, ["static"] * 2, [0.0, 1.5])
            self.assertEqual(arm_collect._read_pose_durations(plan), [0.0, 1.5])
            playback, _ = arm_sim._pose_plan_csv(plan, 100, 0.1, 0.1)
            self.assertAlmostEqual(playback.duration, 1.7)


if __name__ == "__main__":
    unittest.main()
