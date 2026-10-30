"""Offline workflow checks. USB and viewer execution are replaced by fixtures."""

import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from arm_id_feedforward import arm_collect
from arm_id_feedforward import arm_host
from arm_id_feedforward.arm_analysis import analyze_capture, calibration_consistent
from arm_id_feedforward.arm_workflow import Cancelled, Workflow, run_tool


def write_capture(path, stage, pose_count=1, complete=True):
    columns = [
        "stage", "pose_index", "phase", "host_elapsed_s", "device_time_us",
        "sample_sequence", "target_sequence", "status",
        *(f"q{j}_rad" for j in range(1, 8)),
        *(f"dq{j}_rad_s" for j in range(1, 8)),
        *(f"tau{j}_nm" for j in range(1, 8)),
        *(f"gravity_model_tau{j}_nm" for j in range(1, 8)),
        *(f"calibrated_tau{j}_nm" for j in range(1, 8)),
    ]
    count = 201 if stage == "dynamic" else pose_count * 30
    if not complete:
        count = max(1, count // 2)
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        for index in range(count):
            pose = index // 30 + 1
            model_torque = [0.0] + [float(pose)] * 6
            writer.writerow([
                stage, pose, "position_capture", index / 200, index * 5000,
                index, index, "0x000f",
                *([0.0] * 7), *([0.0] * 7),
                *[2 * value + 0.01 for value in model_torque],
                *model_torque, *([0.0] * 7),
            ])
    path.with_suffix(".json").write_text(json.dumps({"transition_duration_s": 0.0}))


def write_preparation(directory):
    directory.mkdir(parents=True, exist_ok=True)
    report = {
        "kind": "arm_stage0_screening", "version": 1,
        "urdf_sha256": arm_collect._sha256(arm_collect.URDF),
        "reference_gravity_model_sha256": arm_collect._sha256(arm_collect.REFERENCE_GRAVITY_MODEL),
        "plans": {}, "alignment": {"insufficient_excitation_joints": [1]},
    }
    for stage, count in (("alignment", 12), ("static", 16), ("dynamic", 0), ("validation", 0)):
        plan = directory / f"{stage}.csv"
        if stage in ("dynamic", "validation"):
            arm_collect._write_trajectory(plan, np.array([0.0, 1.0]), np.zeros((2, 7)))
        else:
            arm_collect._write_pose_plan(plan, np.zeros((count, 7)), [stage] * count)
        arm_collect._write_metadata(plan, {"kind": "test_fixture"})
        report["plans"][stage] = {"file": plan.name, "sha256": arm_collect._sha256(plan)}
    (directory / "screening_report.json").write_text(json.dumps(report))


class WorkflowTests(unittest.TestCase):
    def test_feedback_outside_planning_limits_is_preserved(self):
        actual = (1.7, -0.1, 1.7, 0.1, 3.3, 2.2, 3.3)
        self.assertEqual(arm_host.validate_position(actual), actual)
        self.assertEqual(arm_host.validate_feedback_position(actual), actual)
        with self.assertRaises(ValueError):
            arm_host.validate_position([float("nan")] * 7)

    def test_firmware_range_rejection_is_advisory_but_not_acceptance(self):
        client = MagicMock()
        client.position_rejections = 0
        client.command_timeouts = 0
        ack = arm_host.Ack(17, arm_host.MODE_GRAVITY_DRAG, 3)
        client.read_messages.return_value = [ack]
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(arm_collect._wait_ack(client, 17, 1.0), ack)
            self.assertFalse(arm_host.check_command_result(client, ack))
        self.assertEqual(client.position_rejections, 2)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertFalse(arm_host.check_command_result(client, arm_host.Ack(18, 0, 7)))
        self.assertEqual(client.command_timeouts, 1)
        with self.assertRaisesRegex(RuntimeError, "motors_offline"):
            arm_host.check_command_result(client, arm_host.Ack(18, 0, 4))

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.workflow = Workflow(self.root, "unused")
        self.calls = []
        write_preparation(self.root / "prepared")
        self.workflow.state["preparation"] = "prepared/screening_report.json"
        self.workflow.state["completed"] = ["preparation"]
        self.workflow.save()

    def fake_tool(self, module, arguments, log):
        if module.__name__.endswith(("arm_plan", "arm_excitation")):
            args = module.build_parser().parse_args(arguments)
            write_preparation(args.output)
            return
        if module.__name__.endswith('arm_dynamics'):
            args=module.build_parser().parse_args(arguments)
            args.output.mkdir()
            (args.output/'report.json').write_text('{}')
            return
        command = arguments[0]
        self.calls.append(command)
        if command == "playback":
            return
        args = module.build_parser().parse_args(arguments)
        if command in ("pose-plan", "dynamic-plan"):
            if command == "dynamic-plan":
                args.duration = 1.0
            with contextlib.redirect_stdout(io.StringIO()):
                args.handler(args)
        elif command in ("collect-poses", "collect-dynamic"):
            self.assertIsNone(args.calibration)
            stage = args.stage if command == "collect-poses" else "dynamic"
            write_capture(args.output, stage, 12 if stage == "alignment" else 16)
        elif command == "alignment-report":
            with contextlib.redirect_stdout(io.StringIO()):
                args.handler(args)

    def run_stage(self, stage, answers):
        with patch("arm_id_feedforward.arm_workflow.run_tool", side_effect=self.fake_tool), \
             patch("builtins.input", side_effect=answers), \
             contextlib.redirect_stdout(io.StringIO()):
            return self.workflow.run_stage(stage)

    def test_static_and_dynamic_flow_without_calibration(self):
        self.assertTrue(self.run_stage("static", ["y", "y", "执行", "y"]))
        self.assertTrue(self.run_stage("dynamic", ["y", "y", "执行", "y"]))
        self.assertEqual(self.workflow.state["completed"], ["preparation", "static", "dynamic"])
        reopened = Workflow(self.root, "unused")
        self.assertEqual(reopened.state["torque_source"], "raw_feedback")
        self.assertNotIn("calibration", reopened.state)
        self.assertEqual(self.calls.count("playback"), 2)

    def test_direct_dynamic_training_validation_and_fit(self):
        self.assertTrue(self.run_stage('dynamic',['y','y','执行','y']))
        self.assertTrue(self.run_stage('validation',['y','y','执行','y']))
        calls=self.calls.copy()
        with patch('arm_id_feedforward.arm_workflow.choose_port',side_effect=AssertionError('fit must not open serial')):
            self.assertTrue(self.run_stage('identification',['y']))
        self.assertEqual(self.calls,calls)
        self.assertIn('identification',self.workflow.state['completed'])
        self.assertNotIn('static',self.workflow.state['completed'])

    def test_fit_requires_separate_validation_capture(self):
        self.assertTrue(self.run_stage('dynamic',['y','y','执行','y']))
        with self.assertRaisesRegex(ValueError,'独立验证'):
            self.run_stage('identification',[])

    def test_completed_dynamic_flow_does_not_suggest_optional_static(self):
        self.workflow.state['completed']=['preparation','dynamic','validation','identification']
        with patch('builtins.input',return_value='q') as prompt,contextlib.redirect_stdout(io.StringIO()):
            self.workflow.run()
        prompt.assert_called_once_with('选择阶段 [q]：')

    def test_declining_motion_never_collects_and_resume_previews_again(self):
        with self.assertRaises(Cancelled):
            self.run_stage("static", ["y", "y", ""])
        self.assertNotIn("collect-poses", self.calls)
        self.calls.clear()
        self.assertTrue(self.run_stage("static", ["y", "执行", "y"]))
        self.assertEqual(self.calls[0], "playback")
        self.assertNotIn("pose-plan", self.calls)

    def test_resume_after_capture_does_not_move_again(self):
        with self.assertRaises(Cancelled):
            self.run_stage("static", ["y", "y", "执行", ""])
        self.calls.clear()
        self.assertTrue(self.run_stage("static", ["y"]))
        self.assertEqual(self.calls, [])

    def test_new_preparation_replaces_pending_plan_without_overwriting(self):
        with self.assertRaises(Cancelled):
            self.run_stage("static", ["y", "y", ""])
        old = self.workflow.state["attempts"]["static"]["directory"]
        old_bytes = (self.root / old / "plan.csv").read_bytes()
        write_preparation(self.root / "updated_prepared")
        self.workflow.state["preparation"] = "updated_prepared/screening_report.json"
        self.assertTrue(self.run_stage("static", ["y", "y", "执行", "y"]))
        new = self.workflow.state["attempts"]["static"]
        self.assertNotEqual(new["directory"], old)
        self.assertEqual(new["preparation"], "updated_prepared/screening_report.json")
        self.assertEqual((self.root / old / "plan.csv").read_bytes(), old_bytes)

    def test_legacy_failed_alignment_and_calibration_do_not_block_or_apply(self):
        legacy = {"directory": "alignment/run_001", "phase": "rejected"}
        self.workflow.state["attempts"]["alignment"] = legacy.copy()
        self.workflow.state["calibration"] = "missing_legacy_calibration.json"
        self.workflow.save()
        self.workflow = Workflow(self.root, "unused")
        with patch("builtins.input", return_value="q") as prompt, contextlib.redirect_stdout(io.StringIO()):
            self.workflow.run()
        prompt.assert_called_once_with("选择阶段 [1]：")
        self.assertTrue(self.run_stage("static", ["y", "y", "执行", "y"]))
        self.assertEqual(self.workflow.state["attempts"]["alignment"], legacy)
        self.assertEqual(self.workflow.state["calibration"], "missing_legacy_calibration.json")

    def test_stage_zero_runs_without_serial_and_saved_plans_are_reused(self):
        self.workflow.port = None
        with patch("arm_id_feedforward.arm_workflow.choose_port", side_effect=AssertionError("USB must remain unopened")):
            self.assertTrue(self.run_stage("preparation", ["y", "y"]))
        self.assertEqual(self.calls, [])
        self.workflow.port = "unused"
        self.assertTrue(self.run_stage("static", ["y", "y", "执行", "y"]))
        self.assertNotIn("pose-plan", self.calls)

    def test_missing_preparation_and_changed_plan_block_motion(self):
        saved = self.workflow.state.pop("preparation")
        with self.assertRaisesRegex(ValueError, "第0步"):
            self.run_stage("static", [])
        self.workflow.state["preparation"] = saved
        source = self.root / "prepared/static.csv"
        source.write_text(source.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "计划已变化"):
            self.run_stage("static", [])
        self.assertEqual(self.calls, [])

    def test_dynamic_interrupt_is_not_success(self):
        parser = arm_collect.build_parser()
        args = parser.parse_args(["dynamic-plan", "--output", str(self.root / "unused.csv")])
        args.handler = lambda _: 130
        with patch.object(arm_collect, "build_parser") as build:
            build.return_value.parse_args.return_value = args
            with self.assertRaises(KeyboardInterrupt):
                run_tool(arm_collect, [], self.root / "details.log")

    def test_dynamic_capture_metadata_without_calibration(self):
        trajectory = self.root / "trajectory.csv"
        q = np.asarray([[0, 0.9, 0, -0.8, 0, 0.25, 0]] * 3)
        arm_collect._write_trajectory(trajectory, np.array([0, 0.01, 0.02]), q)
        output = self.root / "dynamic_raw.csv"
        args = arm_collect.build_parser().parse_args([
            "collect-dynamic", "--trajectory", str(trajectory), "--port", "unused",
            "--output", str(output),
        ])
        with patch.object(arm_collect, "load_calibration", side_effect=AssertionError("must use raw feedback")), \
             patch.object(arm_host, "run_collect", return_value=0) as collect:
            self.assertEqual(arm_collect.collect_dynamic(args), 0)
        self.assertIsNone(collect.call_args.args[0].calibration)
        metadata = json.loads(output.with_suffix(".json").read_text())
        self.assertEqual(metadata["torque_source"], "raw_feedback")
        self.assertNotIn("torque_calibration", metadata)

    def test_low_level_interrupt_requests_gravity_and_returns_130(self):
        args = arm_host.build_parser().parse_args([
            "collect", "--port", "unused", "--target",
            "0", "1", "0", "-1", "0", "0", "0",
            "--output", str(self.root / "interrupted.csv"),
        ])
        client = MagicMock()
        client.send_command.side_effect = [KeyboardInterrupt(), 99]
        with patch.object(arm_host, "open_port"), \
             patch.object(arm_host, "ArmClient", return_value=client), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(arm_host.run_collect(args), 130)
        self.assertEqual(client.send_command.call_args.args[0], arm_host.MODE_GRAVITY_DRAG)

    def test_pose_failure_requests_gravity(self):
        client = MagicMock()
        with self.assertRaises(KeyboardInterrupt):
            with arm_collect._gravity_on_exit(client):
                raise KeyboardInterrupt()
        client.send_command.assert_called_once_with(arm_host.MODE_GRAVITY_DRAG, (0.0,) * 7)

    def test_pose_collection_keeps_off_target_and_unsettled_feedback(self):
        for speed in (0.0, 1.0):
            plan = self.root / f"poses_{speed}.csv"
            capture = self.root / f"capture_{speed}.csv"
            target = [0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0]
            arm_collect._write_pose_plan(plan, np.asarray([target, target]), ["static"] * 2, [0.2, 0.2])
            args = arm_collect.build_parser().parse_args([
                "collect-poses", "--stage", "static", "--poses", str(plan),
                "--port", "unused", "--output", str(capture),
                "--settle-window", "0.01", "--settle-timeout", "0.1",
                "--capture-seconds", "0.3",
            ])
            clock = MagicMock()
            ticks = iter(np.arange(0, 100, 0.005))
            clock_now = [0.0]

            def now():
                clock_now[0] = float(next(ticks))
                return clock_now[0]

            clock.monotonic.side_effect = now
            actual = (0.0, 0.4, 0.0, -0.4, 0.0, 0.0, 0.0)
            sample = arm_host.Sample(1, 1, 1000, 1 if speed else 15, actual, (speed,) * 7, (0.0,) * 7)
            client = MagicMock()
            client.position_rejections = 0
            client.command_timeouts = 0
            client.read_messages.return_value = [sample]
            client.command_targets = {}
            command_times = []

            def send_command(mode, _position):
                if mode == arm_host.MODE_DATA_COLLECTION:
                    command_times.append(clock_now[0])
                return len(command_times)

            client.send_command.side_effect = send_command
            read_index = 0

            def read_state(*_):
                nonlocal read_index
                read_index += 1
                q = list(actual)
                q[0] = read_index * 0.03
                return arm_host.Sample(read_index, 1, read_index * 5000, 9, tuple(q), (speed,) * 7, (0.0,) * 7)

            with patch.object(arm_collect, "time", clock), \
                 patch.object(arm_collect.arm_host, "open_port"), \
                 patch.object(arm_collect.arm_host, "ArmClient", return_value=client), \
                 patch.object(arm_collect, "_wait_ack") as wait_ack, \
                 patch.object(arm_collect, "_read_state", side_effect=read_state) as read_current, \
                 patch.object(arm_collect, "_move_duration", return_value=0.01), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(arm_collect.collect_pose_sequence(args), 0)
            wait_ack.assert_not_called()
            read_current.assert_called_once()
            modes = [call.args[0] for call in client.send_command.call_args_list]
            self.assertEqual(modes[-1], arm_host.MODE_GRAVITY_DRAG)
            self.assertTrue(all(mode == arm_host.MODE_DATA_COLLECTION for mode in modes[:-1]))
            self.assertGreater(len(modes), 10)
            self.assertLess(float(np.max(np.diff(command_times))), 0.1)
            with capture.open() as stream:
                all_rows = list(csv.DictReader(stream))
                rows = [row for row in all_rows if row["phase"] == "position_capture"]
            for pose in ("1", "2"):
                times = [float(row["host_elapsed_s"]) for row in all_rows if row["pose_index"] == pose and row["phase"] == "move"]
                self.assertGreater(max(times) - min(times), 0.15)
            self.assertEqual({row["pose_index"] for row in rows}, {"1", "2"})
            self.assertTrue(all(float(row["q2_rad"]) == 0.4 for row in rows))
            self.assertNotIn("calibrated_tau1_nm", rows[0])
            metadata = json.loads(capture.with_suffix(".json").read_text())
            self.assertEqual(metadata["torque_source"], "raw_feedback")
            self.assertEqual(metadata["capture_mode"], "position_control")
            self.assertTrue(any(note["note"] == "few_samples" for note in metadata["quality_notes"]))
            if speed:
                self.assertTrue(any(note["note"] == "settle_wait_expired" for note in metadata["quality_notes"]))

    def test_quality_warning_does_not_reject_completed_stage(self):
        warning = {"passed": False, "issues": ["姿态稳定性提示"]}
        with patch("arm_id_feedforward.arm_workflow.analyze_capture", return_value=warning):
            self.assertTrue(self.run_stage("static", ["y", "y", "执行", "y"]))
        self.assertEqual(self.workflow.state["attempts"]["static"]["phase"], "complete")
        self.assertIn("static", self.workflow.state["completed"])

    def test_drift_outside_limits_never_becomes_next_command_start(self):
        plan = self.root / "drifting_poses.csv"
        poses = np.array([[0, 0.9, 0, -0.8, 0, 0.25, 0], [0, 1.1, 0, -0.9, 0, 0.25, 0]])
        arm_collect._write_pose_plan(plan, poses, ["static"] * 2, [0.2, 0.2])
        lower = np.array([-1.42, np.pi / 6, -1.42, -5 * np.pi / 6, -2.99, -1.94, -2.99])
        upper = np.array([1.42, 5 * np.pi / 6, 1.42, -np.pi / 6, 2.99, 1.94, 2.99])
        arm_collect._write_metadata(plan, {"joint_lower_rad": lower.tolist(), "joint_upper_rad": upper.tolist()})
        output = self.root / "drifting_capture.csv"
        args = arm_collect.build_parser().parse_args([
            "collect-poses", "--stage", "static", "--poses", str(plan), "--port", "unused",
            "--output", str(output), "--settle-window", "0.01", "--settle-timeout", "0.1",
            "--capture-seconds", "0.3",
        ])
        actual = (0.0, 3.2, 0.0, -0.3, 0.0, 0.25, 0.0)
        sample = arm_host.Sample(1, 1, 5000, 15, actual, (0.0,) * 7, (0.0,) * 7)
        client = MagicMock()
        client.position_rejections = 0
        client.command_timeouts = 0
        client.command_targets = {}
        client.read_messages.return_value = [sample]
        ticks = iter(np.arange(0, 100, 0.005))
        clock = MagicMock()
        clock.monotonic.side_effect = lambda: float(next(ticks))
        with patch.object(arm_collect, "time", clock), \
             patch.object(arm_host, "open_port"), \
             patch.object(arm_host, "ArmClient", return_value=client), \
             patch.object(arm_collect, "_read_state", return_value=sample), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(arm_collect.collect_pose_sequence(args), 0)
        commands = np.array([
            call.args[1] for call in client.send_command.call_args_list
            if call.args[0] == arm_host.MODE_DATA_COLLECTION
        ])
        self.assertTrue(np.all(commands >= lower) and np.all(commands <= upper))
        # At pose boundaries, the previous target remains the command anchor;
        # large measured tracking error must not reset it outside the bounds.
        self.assertLess(float(np.max(np.abs(np.diff(commands, axis=0)))), 0.15)
        with output.open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertTrue(all(float(row["q2_rad"]) == 3.2 for row in rows))
        metadata = json.loads(output.with_suffix(".json").read_text())
        self.assertEqual(metadata["start_position_rad"][1], 3.2)
        self.assertAlmostEqual(metadata["command_start_position_rad"][1], upper[1])

    def test_dynamic_start_is_bounded_without_clipping_feedback(self):
        path = self.root / "start_trajectory.csv"
        q = np.array([[0, 0.9, 0, -0.8, 0, 0.25, 0]] * 3)
        arm_collect._write_trajectory(path, np.array([0, 0.01, 0.02]), q)
        lower = np.array([-1.42, np.pi / 6, -1.42, -5 * np.pi / 6, -2.99, -1.94, -2.99])
        upper = np.array([1.42, 5 * np.pi / 6, 1.42, -np.pi / 6, 2.99, 1.94, 2.99])
        arm_collect._write_metadata(path, {"joint_lower_rad": lower.tolist(), "joint_upper_rad": upper.tolist()})
        actual = (0.0, 3.2, 0.0, -0.3, 0.0, 0.25, 0.0)
        with contextlib.redirect_stdout(io.StringIO()):
            anchor = arm_host.command_start_for_plan(actual, path)
        trajectory = arm_host.Trajectory(path)
        trajectory.prepend_transition(anchor, 0.2, 100)
        self.assertTrue(np.all(np.asarray(trajectory.positions) >= lower))
        self.assertTrue(np.all(np.asarray(trajectory.positions) <= upper))
        self.assertEqual(actual[1], 3.2)

    def test_quality_rejects_incomplete_and_invalid_captures(self):
        plan = self.root / "plan.csv"
        with plan.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["pose_index"])
            writer.writerows([[1], [2]])
        capture = self.root / "capture.csv"
        write_capture(capture, "static", pose_count=2)
        report = analyze_capture(capture, "static", plan)
        self.assertTrue(report["passed"])
        self.assertEqual(report["torque_source"], "raw_feedback")
        self.assertEqual(report["capture_mode"], "position_control")
        self.assertEqual(report["torque_columns"], [f"tau{j}_nm" for j in range(1, 8)])
        # Fixture calibrated columns are zero; raw feedback is deliberately different.
        np.testing.assert_allclose(report["poses"][0]["torque_median_nm"], [0.01] + [2.01] * 6)
        write_capture(capture, "static", pose_count=2, complete=False)
        self.assertFalse(analyze_capture(capture, "static", plan)["passed"])
        plan.write_text("time_s\n0\n1\n")
        write_capture(capture, "dynamic", complete=False)
        self.assertFalse(analyze_capture(capture, "dynamic", plan)["passed"])
        write_capture(capture, "dynamic")
        capture.write_text(capture.read_text().replace("0x000f", "0x0001"))
        self.assertFalse(analyze_capture(capture, "dynamic", plan)["passed"])

    def test_legacy_static_capture_remains_readable(self):
        plan = self.root / "legacy_plan.csv"
        plan.write_text("pose_index\n1\n")
        capture = self.root / "legacy_capture.csv"
        write_capture(capture, "static")
        capture.write_text(capture.read_text().replace("position_capture", "gravity_capture").replace("0x000f", "0x0009"))
        report = analyze_capture(capture, "static", plan)
        self.assertTrue(report["passed"])
        self.assertEqual(report["capture_mode"], "legacy_gravity_drag")

    def test_donor_consistency_gate(self):
        report = {
            "same_motor_consistency": {
                "available": True, "same_gain_sign": True,
                "relative_gain_spread": 0.1, "offset_spread_nm": 0.01,
            },
            "joints": [{"gain": None}] + [
                {"gain": 2.0, "offset_nm": 0.0, "model_span_nm": 1.0, "rmse_nm": 0.01}
                for _ in range(6)
            ],
        }
        self.assertTrue(calibration_consistent(report))
        report["same_motor_consistency"]["relative_gain_spread"] = 0.5
        self.assertFalse(calibration_consistent(report))


if __name__ == "__main__":
    unittest.main()
