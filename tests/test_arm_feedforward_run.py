"""Full shared dynamics, analytic references and timed execution without hardware."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pinocchio as pin

from ega_arm_host import arm_collect
from ega_arm_host import arm_dynamics
from ega_arm_host import arm_excitation
from ega_arm_host import arm_feedforward as ff
from ega_arm_host import arm_host
from ega_arm_host import arm_plan


class FakeClient:
    def __init__(self, *, replies=True, ack=True, interrupt=False):
        self.sent = []
        self.replies = replies
        self.ack = ack
        self.interrupt = interrupt
        self.received = 0
        self.position_rejections = self.command_timeouts = 0

    def send_feedforward(self, q, v, tau):
        if self.interrupt and len(self.sent) == 4:
            raise KeyboardInterrupt()
        seq = len(self.sent)+100
        self.sent.append((seq, np.array(q), np.array(v), np.array(tau)))
        return seq

    def read_messages(self):
        if not self.replies or self.received == len(self.sent):
            return []
        seq, q, v, tau = self.sent[-1]
        self.received = len(self.sent)
        # Large feedback error is retained, never a reason to stop streaming.
        sample = arm_host.Sample(self.received, seq, self.received*5000, 31,
                                 tuple(q+10), tuple(v+1), tuple(tau+2))
        return ([arm_host.Ack(seq, 3, 0)] if self.ack and self.received == 1 else [])+[sample]


class FeedforwardRunTests(unittest.TestCase):
    def test_full_shared_prediction_matches_identification_regressor_and_terms(self):
        model = ff.SharedModel()
        rng = np.random.default_rng(42)
        q, v, a = [rng.normal(size=(270, 7)) for _ in range(3)]  # two bounded chunks
        predicted, terms = model.predict(q, v, a, components=True)
        expected = (model.dynamics.matrix(q, v, a)@model.theta).reshape(7, -1).T
        np.testing.assert_allclose(predicted, expected, atol=2e-15)
        np.testing.assert_allclose(predicted, sum(terms.values()), atol=2e-15)
        np.testing.assert_allclose(terms['viscous'], v*model.theta[70:77])
        np.testing.assert_allclose(terms['coulomb'], np.sign(v)*model.theta[77:84])
        np.testing.assert_allclose(terms['rotor'], a*model.theta[84:91])
        # Independently verify the evaluator using physical URDF inertias and RNEA.
        model.theta = np.r_[np.concatenate([i.toDynamicParameters() for i in model.dynamics.model.inertias[1:]]),
                            np.full(7, .02), np.full(7, .04), np.full(7, .003)]
        data = model.dynamics.model.createData()
        expected = np.array([pin.rnea(model.dynamics.model, data, qq, vv, aa).copy()
                             for qq, vv, aa in zip(q, v, a)])+.02*v+.04*np.sign(v)+.003*a
        np.testing.assert_allclose(model.predict(q, v, a), expected, atol=2e-12)

    def make_plan(self, directory):
        path = directory/'plan.csv'
        motion = arm_excitation.SplineMotion(6.)
        controls = np.tile(arm_plan.HOME, (5, 1))
        controls[:, 0] = [.1, -.1, .2, -.2, .1]
        t, q, v, a = motion.evaluate(controls.reshape(-1), 100.)
        arm_collect._write_trajectory(path, t, q)
        document = dict(kind='figaroh_full_dynamic_excitation', version=1,
                        trajectory_sha256=arm_collect._sha256(path),
                        urdf_sha256=arm_collect._sha256(arm_collect.URDF),
                        spline_degree=5, control_points_per_segment=11, home_rad=arm_plan.HOME.tolist(),
                        duration_s=6., segments=[dict(control_points=controls.tolist())])
        path.with_suffix('.json').write_text(json.dumps(document))
        return path, t, q, v, a

    def test_reference_recovers_original_analytic_derivatives_and_rejects_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            path, t, q, v, a = self.make_plan(Path(directory))
            reference = ff.Reference(path)
            actual = reference.evaluate(t)
            for x, y in zip(actual, (q, v, a)):
                np.testing.assert_allclose(x, y, atol=1e-12)
            times = np.linspace(0, 6, 1000)
            epsilon = 1e-5
            derivative = (reference.evaluate(times+epsilon)[0]-reference.evaluate(times-epsilon)[0])/(2*epsilon)
            np.testing.assert_allclose(derivative[1:-1], reference.evaluate(times)[1][1:-1], atol=1e-9)
            path.write_text(path.read_text().replace('0.00000000', '0.01000000', 1))
            with self.assertRaisesRegex(ValueError, '哈希'):
                ff.Reference(path)

    def test_retimed_reference_and_full_torque_use_scaled_derivatives(self):
        with tempfile.TemporaryDirectory() as directory:
            path, _, _, _, _ = self.make_plan(Path(directory))
            source = ff.Reference(path)
            t, q, v, a = ff.scaled_reference(source, 1.8)
            original_q, original_v, original_a = source.evaluate(t*1.8)
            np.testing.assert_allclose(q, original_q, atol=1e-12)
            np.testing.assert_allclose(v, original_v*1.8, atol=1e-12)
            np.testing.assert_allclose(a, original_a*1.8**2, atol=1e-12)
            self.assertAlmostEqual(t[-1], 6/1.8)
            self.assertGreater(t[-1]-t[-2], 0)
            self.assertLess(t[-1]-t[-2], 1/ff.RATE)
            model = ff.SharedModel()
            selected = np.arange(0, len(t), 45)
            tau = model.predict(q[selected], v[selected], a[selected])
            reference = (model.dynamics.matrix(q[selected],v[selected],a[selected])@model.theta).reshape(7,-1).T
            np.testing.assert_allclose(tau, reference, atol=1e-12)
            # Time scaling changes rigid and friction terms; torque is not
            # simply multiplied by the speed factor.
            slow = model.predict(q[selected], original_v[selected], original_a[selected])
            self.assertGreater(np.max(abs(tau-slow)), 1e-5)

    def test_gravity_baseline_keeps_reference_and_freezes_torque_choice(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, _, _, _, _ = self.make_plan(root)
            full_path, gravity_path = root/'full', root/'gravity'
            with patch.object(ff, 'check_path'):
                ff.prepare(full_path, path, speed=3., torque_mode='full')
                ff.prepare(gravity_path, path, speed=3., torque_mode='gravity')
                entry, entry_duration = ff.execution_reference(gravity_path,
                                                                arm_plan.HOME+.1)
            with np.load(full_path/'body_reference.npz') as saved:
                full = {key: saved[key].copy() for key in ('t', 'q', 'v', 'a', 'tau')}
            with np.load(gravity_path/'body_reference.npz') as saved:
                gravity = {key: saved[key].copy() for key in ('t', 'q', 'v', 'a', 'tau')}
            for key in ('t', 'q', 'v', 'a'):
                np.testing.assert_array_equal(full[key], gravity[key])
            model = ff.SharedModel()
            zero = np.zeros_like(gravity['q'])
            np.testing.assert_allclose(gravity['tau'],
                                       model.predict(gravity['q'], zero, zero))
            self.assertGreater(np.max(abs(full['tau']-gravity['tau'])), .01)
            np.testing.assert_allclose(entry['tau'][0],
                                       model.predict(entry['q'][:1],
                                                     np.zeros((1, 7)),
                                                     np.zeros((1, 7)))[0])
            self.assertGreaterEqual(entry_duration, 24.)
            with self.assertRaisesRegex(ValueError, '前馈类型已冻结'):
                ff.prepare(gravity_path, path, torque_mode='full')

    def test_entry_is_smooth_and_preserves_actual_start_without_clipping(self):
        start = arm_plan.HOME.copy(); start[1] = -.2
        t, q, v, a = ff.transition(start, arm_plan.HOME)
        np.testing.assert_allclose(q[0], start)
        np.testing.assert_allclose(q[-1], arm_plan.HOME)
        np.testing.assert_allclose(v[[0, -1]], 0, atol=1e-14)
        np.testing.assert_allclose(a[[0, -1]], 0, atol=1e-14)
        self.assertLessEqual(np.max(abs(v)), .25)
        self.assertLessEqual(np.max(abs(a)), .5)
        self.assertGreaterEqual(t[-1], 24.)

    @staticmethod
    def tiny_reference():
        t = np.arange(9)/ff.RATE
        return dict(t=t, q=np.tile(t[:, None], (1, 7)),
                    v=np.ones((9, 7)), a=np.zeros((9, 7)), tau=np.full((9, 7), .3))

    def run_stream(self, client, *, fail=False, reference=None):
        now = [0.]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            for name in ('model.json', 'plan.csv', 'reference.csv'):
                (output/name).write_text('{}')
            def sleep(dt):
                now[0] += max(dt, .00001)
            with patch.object(ff.time, 'monotonic', side_effect=lambda: now[0]), \
                 patch.object(ff.time, 'sleep', side_effect=sleep):
                if fail:
                    with self.assertRaises((RuntimeError, KeyboardInterrupt)):
                        ff.stream_targets(client, reference or self.tiny_reference(), 0., output)
                else:
                    ff.stream_targets(client, reference or self.tiny_reference(), 0., output)
            with (output/'capture.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            return rows, json.loads((output/'capture.json').read_text())

    def test_execution_logs_exact_sequence_targets_and_does_not_gate_tracking_error(self):
        client = FakeClient()
        rows, metadata = self.run_stream(client)
        self.assertTrue(metadata['complete'])
        self.assertGreater(len(client.sent), 20)
        self.assertTrue(all(None not in row for row in rows))
        by_sequence = {seq: (q, v, tau) for seq, q, v, tau in client.sent}
        for row in rows:
            q, v, tau = by_sequence[int(row['target_sequence'])]
            self.assertAlmostEqual(float(row['target_q1_rad']), q[0])
            self.assertAlmostEqual(float(row['ff_tau1_nm']), tau[0])
            self.assertAlmostEqual(float(row['q1_rad']), q[0]+10)
        self.assertAlmostEqual(client.sent[-1][1][0], .04)

    def test_lost_entry_ack_can_be_confirmed_by_mode3_sequence_feedback(self):
        _, metadata = self.run_stream(FakeClient(ack=False))
        self.assertTrue(metadata['complete'])

    def test_old_firmware_never_advances_reference_and_failure_keeps_metadata(self):
        client = FakeClient(replies=False)
        rows, metadata = self.run_stream(client, fail=True)
        self.assertFalse(metadata['complete'])
        self.assertIn('模式 3', metadata['error'])
        self.assertEqual(rows, [])
        self.assertGreater(len(client.sent), 300)
        for _, q, _, _ in client.sent:
            np.testing.assert_allclose(q, 0.)

    def test_ctrl_c_preserves_partial_capture_and_metadata(self):
        rows, metadata = self.run_stream(FakeClient(interrupt=True), fail=True)
        self.assertFalse(metadata['complete'])
        self.assertIn('KeyboardInterrupt', metadata['error'])
        self.assertEqual(len(rows), 4)

    def test_feedback_loss_after_handshake_keeps_streaming_to_end(self):
        client = FakeClient()
        original = client.read_messages
        client.read_messages = lambda: original() if len(client.sent) == 1 else []
        reference = self.tiny_reference()
        reference['t'] = np.linspace(0, 2.2, 9)
        # Streaming normally uses 200 Hz references; here a long held target
        # is sufficient to exercise the no-feedback period and normal finish.
        _, metadata = self.run_stream(client, reference=reference)
        self.assertTrue(metadata['complete'])
        self.assertGreater(len(client.sent), 400)

    def test_analysis_separates_tracking_and_frozen_actual_state_prediction(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            t = np.arange(0, 6.005, .005)
            # Keep velocity positive: zero-phase filtering of discontinuous
            # Coulomb reversals is not an exact-model zero-residual test.
            q = np.tile(arm_plan.HOME, (len(t), 1))+.02*t[:, None]+.001*np.sin(t[:, None])
            v = np.tile(.02+.001*np.cos(t[:, None]), (1, 7))
            a = np.tile(-.001*np.sin(t[:, None]), (1, 7))
            model = ff.SharedModel()
            tau = model.predict(q, v, a)
            rows = dict(host_elapsed_s=t, device_time_us=np.rint(t*1e6).astype(int),
                        sample_sequence=np.arange(len(t)), target_sequence=np.arange(len(t))+100,
                        status=['0x001f']*len(t), reference_time_s=t, phase=['body']*len(t))
            for j in range(7):
                rows.update({ff.Q[j]: q[:, j], ff.V[j]: v[:, j], ff.T[j]: tau[:, j],
                             'target_'+ff.Q[j]: q[:, j]-.02,
                             'target_'+ff.V[j]: v[:, j]-.01,
                             'target_'+ff.A[j]: a[:, j],
                             f'ff_tau{j+1}_nm': tau[:, j]-.03})
            pd.DataFrame(rows).to_csv(output/'capture.csv', index=False)
            arm_collect._write_trajectory(output/'plan.csv', np.array([0., 6.]), q[[0, -1]])
            (output/'capture.json').write_text(json.dumps(dict(transition_duration_s=0., complete=True)))
            (output/'model.json').write_bytes(ff.DEFAULT_MODEL.read_bytes())
            report = ff.analyze(output)
            self.assertTrue((output/'report.md').is_file())
            self.assertTrue((output/'tracking.png').is_file())
            self.assertEqual(len(report['joint_error_plots']), 7)
            for joint in range(1, 8):
                self.assertTrue((output/f'errors_j{joint}.png').is_file())
            np.testing.assert_allclose(report['position_error_rad']['rmse'], .02, atol=1e-12)
            np.testing.assert_allclose(report['velocity_error_rad_s']['rmse'], .01, atol=1e-12)
            np.testing.assert_allclose(report['feedback_minus_feedforward']['bias'], .03, atol=1e-12)
            self.assertIn('actual_state_prediction', report)
            self.assertLess(max(report['actual_state_prediction']['rmse']), .01)

    def test_cancel_before_motion_never_sends_mode3(self):
        with tempfile.TemporaryDirectory() as directory:
            args = ff.build_parser().parse_args(['--output', directory])
            with patch.object(ff.arm_workflow, 'confirm', side_effect=ff.arm_workflow.Cancelled), \
                 patch.object(ff.arm_host, 'open_port') as serial:
                with self.assertRaises(ff.arm_workflow.Cancelled):
                    ff.run(args)
                serial.assert_not_called()

    def test_cancel_after_preview_never_enters_streaming(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            args = ff.build_parser().parse_args(['--output', directory, '--port', 'unused'])
            state = arm_host.Sample(1, 1, 1, 9, (0.,)*7, (0.,)*7, (0.,)*7)
            with patch.object(ff, 'prepare', return_value={'speed': 1.8}), \
                 patch.object(ff.arm_workflow, 'confirm', side_effect=[None, None, ff.arm_workflow.Cancelled]), \
                 patch.object(ff.arm_host, 'open_port'), \
                 patch.object(ff.arm_host, 'read_current_state', return_value=state) as read, \
                 patch.object(ff, 'execution_reference', return_value=(self.tiny_reference(), 0.)), \
                 patch.object(ff, 'save_reference'), \
                 patch.object(ff.arm_workflow, 'run_tool') as preview, \
                 patch.object(ff, 'stream_targets') as stream:
                with self.assertRaises(ff.arm_workflow.Cancelled):
                    ff.run(args)
                stream.assert_not_called()
                self.assertEqual(read.call_count, 1)
                arguments = preview.call_args.args[1]
                self.assertEqual(arguments[arguments.index('--maximum-velocity')+1], '0.45')
                self.assertEqual(float(arguments[arguments.index('--maximum-acceleration')+1]), 1.62)


if __name__ == '__main__':
    unittest.main()
