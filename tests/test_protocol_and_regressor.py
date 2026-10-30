"""Portable host protocol contract and shared-regressor checks."""

import struct
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from arm_id_feedforward import arm_host
from arm_id_feedforward.regressor import DynamicRegressor


MODE3_FRAME = bytes.fromhex(
    "555301095a02032a0000000000803f0000004000004040000080400000a040"
    "0000c0400000e040000000bf000080bf0000c0bf000000c0000020c00000"
    "40c0000060c00000803e0000003f0000403f0000803f0000a03f0000c03f"
    "0000e03fa8bf"
)


class ProtocolTests(unittest.TestCase):
    def test_mode3_golden_frame(self):
        payload = arm_host.FEEDFORWARD_COMMAND_STRUCT.pack(
            2, 3, 42, *range(1, 8), *[-i / 2 for i in range(1, 8)],
            *[i / 4 for i in range(1, 8)]
        )
        self.assertEqual(arm_host.make_link_frame(1, 9, payload), MODE3_FRAME)
        messages = arm_host.LinkParser().feed(MODE3_FRAME)
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0].payload, payload)

    def test_ack_and_sample_layout(self):
        ack = struct.pack("<BBIBB", 2, 1, 42, 3, 0)
        sample = struct.pack(
            "<BBIIQH21f", 2, 2, 17, 42, 500000, 31,
            *[float(i) for i in range(21)]
        )
        self.assertEqual(arm_host.decode_feedback(ack).command_sequence, 42)
        decoded = arm_host.decode_feedback(sample)
        self.assertEqual(decoded.position, tuple(float(i) for i in range(7)))
        self.assertEqual(decoded.torque, tuple(float(i) for i in range(14, 21)))

    def test_invalid_feedforward_never_writes(self):
        with patch.object(arm_host, "reserve_command_sequences", return_value=(42, 100)):
            port = MagicMock()
            client = arm_host.ArmClient(port)
        with self.assertRaises(ValueError):
            client.send_feedforward([0] * 7, [0] * 7, [float("nan")] * 7)
        port.write.assert_not_called()


class RegressorTests(unittest.TestCase):
    def test_host_control_matches_figaroh_columns(self):
        try:
            from arm_id_feedforward.arm_dynamics import DynamicModel
        except ImportError:
            self.skipTest("identify extra is not installed")
        q = np.asarray([[0.2, 1.0, 0.1, -0.8, 0.1, 0.3, -0.2],
                        [-0.1, 1.2, -0.2, -1.0, 0.2, 0.4, 0.3]])
        v = np.asarray([[0.1, -0.2, 0.3, -0.1, 0.4, -0.3, 0.1],
                        [0.2, -0.1, 0.1, -0.2, 0.3, -0.1, 0.2]])
        a = v * 0.4
        np.testing.assert_allclose(
            DynamicRegressor().matrix(q, v, a), DynamicModel().matrix(q, v, a),
            rtol=0, atol=1e-12,
        )


if __name__ == "__main__":
    unittest.main()
