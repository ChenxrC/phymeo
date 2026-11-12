"""Public quaternion observations are the same accepted orientation as FK matrices."""

from dataclasses import replace

import numpy as np
import pytest
from test_resident_floating_read import H, cache, specification

import phymeo as ph


@pytest.mark.parametrize("mode", ["interpret", "cuda"])
def test_floating_quaternion_all_diagonals_and_endpoint(mode, tmp_path):
    spec = specification(True, eccentric=True)
    rotations = np.array(
        [[0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0.5, 0.5, 0.5, 0.5]], float
    )
    worlds = []
    for i, rotation in enumerate(rotations):
        q = spec.state.q.copy()
        q[3:7] = rotation
        worlds.append((i, replace(spec, state=ph.FloatingRobotState(q, spec.state.v))))
    with ph.TyxWorldBatch(worlds, mode, cache_root=cache(tmp_path)) as batch:
        with batch.prepare_body_read(fields=("rotation_xyzw", "rotation")) as plan:
            output = np.empty(plan.layout["shape"], np.float32)
            for advance in (False, True):
                if advance:
                    assert all(batch.step(H, require_all=True).values())
                before = batch.stats()
                plan.read_into(output)
                assert (
                    batch.stats()["floating_fk_kernel_launches"]
                    == before["floating_fk_kernel_launches"]
                )
                quats = output[:, :8].reshape(5, 2, 4)
                matrices = output[:, 8:].reshape(5, 2, 3, 3)
                np.testing.assert_allclose(np.linalg.norm(quats, axis=-1), 1, atol=1e-6, rtol=1e-5)
                if not advance:
                    np.testing.assert_allclose(quats[:, 0], rotations, atol=1e-6, rtol=1e-5)
                x, y, z, w = np.moveaxis(quats, -1, 0)
                expected = np.stack(
                    (
                        1 - 2 * (y * y + z * z),
                        2 * (x * y - z * w),
                        2 * (x * z + y * w),
                        2 * (x * y + z * w),
                        1 - 2 * (x * x + z * z),
                        2 * (y * z - x * w),
                        2 * (x * z - y * w),
                        2 * (y * z + x * w),
                        1 - 2 * (x * x + y * y),
                    ),
                    axis=-1,
                ).reshape(5, 2, 3, 3)
                np.testing.assert_allclose(matrices, expected, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("mode", ["interpret", "cuda"])
def test_sphere_and_fixed_quaternion_read(mode, tmp_path):
    from test_resident_body_read import model
    from test_resident_tree_step import _program

    fixed = ph.FixedRobotTree(*_program(model()))
    with ph.TyxWorldBatch(
        [(0, ph.FreeSphere(1, 0.125)), (1, fixed)], mode, cache_root=cache(tmp_path)
    ) as batch:
        with batch.prepare_body_read([0], fields=("rotation_xyzw",)) as plan:
            output = np.empty(plan.layout["shape"], np.float32)
            plan.read_into(output)
            np.testing.assert_array_equal(output, [[0, 0, 0, 1]])
        with batch.prepare_body_read([1], fields=("rotation_xyzw",)) as plan:
            output = np.empty(plan.layout["shape"], np.float32)
            plan.read_into(output)
            np.testing.assert_allclose(
                output.reshape(3, 4),
                [[0, 0, 0, 1], [2**-0.5, 0, 0, 2**-0.5], [0, 0, 0, 1]],
                atol=1e-6,
                rtol=1e-5,
            )
