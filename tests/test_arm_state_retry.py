"""Read-only retries and visible feedback-loss warnings; no serial device."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from arm_id_feedforward import arm_host
from arm_id_feedforward.arm_workflow import ProgressLog


class StateRetryTests(unittest.TestCase):
    def run_request(self, reply, timeout=1.1):
        now = [0.]
        client = MagicMock()
        sent = []
        def send(mode, position):
            self.assertEqual(mode, arm_host.MODE_READ_STATE)
            sent.append(100+len(sent))
            return sent[-1]
        client.send_command.side_effect = send
        client.read_messages.side_effect = lambda: reply(now[0], sent)
        def sleep(dt): now[0] += dt
        with patch.object(arm_host.time, 'monotonic', side_effect=lambda:now[0]), \
             patch.object(arm_host.time, 'sleep', side_effect=sleep), \
             contextlib.redirect_stdout(io.StringIO()):
            return arm_host.read_current_state(client, timeout), sent

    @staticmethod
    def sample(sequence):
        return arm_host.Sample(1,sequence,1234,1,(0.,)*7,(0.,)*7,(0.,)*7)

    def test_lost_first_request_retries_and_allows_sample_before_ack(self):
        def reply(now, sent):
            if len(sent)<2: return []
            return [self.sample(sent[-1]), arm_host.Ack(sent[-1],0,0)]
        result, sent = self.run_request(reply)
        self.assertEqual(sent,[100,101])
        self.assertEqual(result.target_sequence,101)

    def test_unmatched_stale_sample_is_not_accepted(self):
        with self.assertRaisesRegex(RuntimeError, '收到.*0 个匹配 ACK、0 个匹配状态'):
            self.run_request(lambda now,sent:[self.sample(99)])

    def test_no_feedback_reports_attempt_count(self):
        with self.assertRaisesRegex(RuntimeError, '发送 3 次只读请求，收到 0 条反馈'):
            self.run_request(lambda now,sent:[])

    def test_missing_ack_does_not_accept_a_sample(self):
        with self.assertRaisesRegex(RuntimeError, '0 个匹配 ACK'):
            self.run_request(lambda now,sent:[self.sample(sent[-1])])

    def test_feedback_loss_warns_and_continues_sending_until_end(self):
        now=[0.]
        client=MagicMock()
        client.parser.frame_errors=0
        client.position_rejections=0; client.command_timeouts=0
        client.command_targets={}
        client.read_messages.side_effect=lambda:[self.sample(100)] if now[0]<.001 else []
        def sleep(dt): now[0]+=dt
        with tempfile.TemporaryDirectory() as directory:
            args=arm_host.build_parser().parse_args(['collect','--port','unused',
                '--target',*(['0']*7),'--duration','2.5','--output',str(Path(directory)/'raw.csv')])
            terminal=io.StringIO()
            with patch.object(arm_host,'open_port'), patch.object(arm_host,'ArmClient',return_value=client), \
                 patch.object(arm_host.time,'monotonic',side_effect=lambda:now[0]), \
                 patch.object(arm_host.time,'sleep',side_effect=sleep), contextlib.redirect_stdout(terminal):
                self.assertEqual(arm_host.run_collect(args),0)
            self.assertIn('仍继续发送轨迹',terminal.getvalue())
            self.assertGreater(now[0],2.49)
            modes=[call.args[0] for call in client.send_command.call_args_list]
            self.assertGreater(modes.count(arm_host.MODE_DATA_COLLECTION),400)
            self.assertEqual(modes[-1],arm_host.MODE_GRAVITY_DRAG)

    def test_warning_reaches_terminal_without_waiting_for_progress_tick(self):
        terminal=io.StringIO(); log=io.StringIO()
        sink=ProgressLog(log,terminal)
        sink.write('警告：反馈中断，继续采集。')
        self.assertIn('反馈中断',terminal.getvalue())
        self.assertIn('反馈中断',log.getvalue())


if __name__=='__main__': unittest.main()
