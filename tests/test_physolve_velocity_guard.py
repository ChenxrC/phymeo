"""The scalar guard uses native DOF offsets and preserves the exact threshold."""
from types import SimpleNamespace
import unittest

import numpy as np
import tyx as tx

from phymeo.backends._physolve_validation import PreparedValidation


class VelocityGuard(unittest.TestCase):
    def test_guard_rejects_post_step_values_and_graph_reuses_flags(self):
        limits = np.asarray([2., 4.], np.float32) * np.float32(1.05)
        value = np.asarray([[9., limits[0], -limits[1]],
                            [9., -limits[0], limits[1]]], np.float32)
        with tx.use_device("cuda:0"):
            stream = tx.CudaStream()
            try:
                with tx.use_stream(stream):
                    first, second = [tx.Array(value, device="cuda:0") for _ in range(2)]
                    guard = PreparedValidation((), "cuda:0")
                    guard.prepare_joint_velocity_limits(
                        [SimpleNamespace(qvel=first), SimpleNamespace(qvel=second)],
                        1, limits, "cuda:0", stream
                    )
                    guard.enqueue_joint_velocity(0, stream)
                    guard.read(stream)
                    graph = tx.Graph(stream=stream, mode="cuda_stream")
                    try:
                        with graph:
                            guard.enqueue_joint_velocity(1, stream)
                        edited = value.copy()
                        edited[1, 2] = np.nextafter(limits[1], np.float32(np.inf))
                        second.assign(tx.Array(edited, device="cuda:0"), stream=stream)
                        graph.replay()
                        with self.assertRaisesRegex(RuntimeError, "joint velocity limit exceeded"):
                            guard.read(stream)
                        guard.clear(stream)
                        guard.enqueue_joint_velocity(0, stream)
                        guard.read(stream)
                    finally:
                        graph.destroy()
            finally:
                stream.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
