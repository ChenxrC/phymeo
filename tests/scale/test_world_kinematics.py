"""Keyed typed position/velocity observations against the committed full checkpoint."""

from dataclasses import replace

import numpy as np
import pytest
from test_resident_step import resident_api, slider


def assert_frame(batch, keys):
    frame = batch.kinematics(keys)
    points = batch.snapshot(keys)
    assert frame.keys == tuple(points)
    assert frame.widths == tuple(len(point.position) for point in points.values())
    assert frame.values.shape == (len(points), 6) and frame.values.dtype == np.float32
    assert frame.ticks.shape == (len(points),) and frame.ticks.dtype == np.int32
    assert frame.values.flags.c_contiguous and frame.ticks.flags.c_contiguous
    assert not frame.values.flags.writeable and not frame.ticks.flags.writeable
    expected = np.zeros((len(points), 6), np.float32)
    for i, point in enumerate(points.values()):
        width = frame.widths[i]
        expected[i, :width] = point.position
        expected[i, 3 : 3 + width] = point.velocity
    np.testing.assert_array_equal(frame.values, expected)
    np.testing.assert_array_equal(frame.ticks, [p.tick for p in points.values()])
    return frame


@pytest.mark.parametrize("mode", ["interpret", "cuda"])
def test_frame_selection_rejection_history_and_local_replay(mode, tmp_path):
    api = resident_api()
    delayed, state = slider(delay=1, kp=8.0, kd=2.0, cap=3.0)
    ball = api.FreeSphere(
        2, 0.125, position=(1, -2, 2), velocity=(-0.25, 0.5, 0), gravity=(0, 0, -8)
    )
    other = api.FreeSphere(1, 0.125, gravity=(0, 0, 0))
    with api.TyxWorldBatch(
        [(91, ball), (7, api.PrismaticSlider(delayed, state)), (37, other)],
        mode,
        cache_root=tmp_path,
    ) as batch:
        initial = assert_frame(batch, None)
        assert initial.keys == (91, 7, 37) and initial.widths == (3, 1, 3)
        before = initial.values.copy()
        cmd = delayed.inputs(np.array([0.125]), np.array([-0.25]), feedforward=np.array([0.5]))
        batch.set_controls({7: cmd, 37: (3e38, 0, 0)})
        assert batch.step(2) == {91: True, 7: True, 37: False}
        selected = assert_frame(batch, [7, 37, 91])
        assert selected.widths == (1, 3, 3) and tuple(selected.ticks) == (1, 0, 1)
        np.testing.assert_array_equal(initial.values, before)
        with pytest.raises(ValueError, match="read-only"):
            selected.values[1, 0] = 9
        with pytest.raises(ValueError, match="read-only"):
            selected.ticks[0] = 9
        saved = batch.snapshot([7])
        assert saved[7].history == (0.125, -0.25, 0.5)
        batch.set_controls({37: (0.0, 0.0, 0.0)})
        assert all(batch.step(1 / 16).values())
        goal = batch.snapshot([7])
        batch.restore(saved)
        recovered = assert_frame(batch, [91, 7])
        assert tuple(recovered.ticks) == (2, 1)
        assert all(batch.step(1 / 16).values()) and batch.snapshot([7]) == goal
        point = batch.snapshot([37])[37]
        batch.restore({37: replace(point, tick=2**24 + 1)})
        assert assert_frame(batch, [37]).ticks[0] == 2**24 + 1
        batch.reset([7])
        assert assert_frame(batch, [7]).ticks[0] == 0 and batch.snapshot([7])[7].history == (
            0,
            0,
            0,
        )


@pytest.mark.parametrize("mode", ["interpret", "cuda"])
def test_invalid_keys_and_empty_selection_do_no_device_work(mode, tmp_path):
    api = resident_api()
    with api.TyxWorldBatch([(22, api.FreeSphere(2, 0.125))], mode, cache_root=tmp_path) as batch:
        before = batch.stats()
        empty = batch.kinematics([])
        assert empty.keys == empty.widths == () and empty.values.shape == (0, 6)
        assert len(empty.ticks) == 0
        for keys, error in (([22, 22], ValueError), ([22, 8], KeyError), ([True], ValueError)):
            with pytest.raises(error):
                batch.kinematics(keys)
        assert batch.stats() == before
    with pytest.raises(RuntimeError, match="closed"):
        batch.kinematics([])
    with api.TyxWorldBatch([], mode, cache_root=tmp_path) as batch:
        assert assert_frame(batch, None).keys == ()
        assert batch.stats()["device_allocations"] == 0


@pytest.mark.parametrize("mode", ["interpret", "cuda"])
def test_read_failure_retains_accepted_state_and_detached_older_frames(mode, tmp_path, monkeypatch):
    api = resident_api()
    with api.TyxWorldBatch(
        [(6, api.FreeSphere(2, 0.125, gravity=(0, 0, -8)))], mode, cache_root=tmp_path
    ) as batch:
        assert batch.step(1 / 128) == {6: True}
        expected = batch.snapshot()
        prior = assert_frame(batch, None)
        original = batch._read

        def fail(buffer, category):
            if category == "kinematics":
                raise RuntimeError("injected kinematics read failure")
            return original(buffer, category)

        with monkeypatch.context() as m:
            m.setattr(batch, "_read", fail)
            with pytest.raises(RuntimeError, match="injected"):
                batch.kinematics()
        assert batch.snapshot() == expected
        assert batch.stats()["recovery_sync_attempts"] == 1
        recovered = assert_frame(batch, None)
        assert not np.shares_memory(prior.values, recovered.values)
        assert not np.shares_memory(prior.ticks, recovered.ticks)
        assert batch.step(1 / 128) == {6: True}
        assert assert_frame(batch, None).ticks[0] == 2 and prior.ticks[0] == 1


@pytest.mark.parametrize("mode", ["interpret", "cuda"])
def test_nonempty_selection_downloads_one_pair_of_existing_buffers(mode, tmp_path):
    api = resident_api()
    ball = api.FreeSphere(2, 0.125)
    with api.TyxWorldBatch([(8, ball), (2, ball), (5, ball)], mode, cache_root=tmp_path) as batch:
        assert all(batch.step(1 / 128).values())
        before = batch.stats()
        if mode == "cuda":
            import tyx

            native = tyx.runtime_stats()
        selected = batch.kinematics([5, 8])
        after = batch.stats()
        assert selected.keys == (5, 8) and tuple(selected.ticks) == (1, 1)
        assert after["kinematics_downloaded_bytes"] - before["kinematics_downloaded_bytes"] == (
            3 * 52 if mode == "cuda" else 0
        )
        for name in (
            "device_allocations",
            "submitted_kernels",
            "synchronizations",
            "snapshot_downloaded_bytes",
            "completed_steps",
            "status_downloaded_bytes",
        ):
            assert after[name] == before[name]
        if mode == "cuda":
            observed = tyx.runtime_stats()
            assert observed.cuda_kernel_launches == native.cuda_kernel_launches
            assert observed.cuda_d2h_bytes - native.cuda_d2h_bytes == 3 * 52
