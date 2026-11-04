"""Control and geometry snapshots remain paired while another real command is published."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event, get_ident

import numpy as np
import pytest
from test_robot_payload_scene import rail_scene


class _HandoffAtUnlock:
    """Run a waiting writer at the first fully unlocked point in the reader.

    The wrapped lock is the scene's original RLock. Only the interleaving after
    the reader's outermost release is coordinated. Timeouts guard against deadlock.
    """

    def __init__(self, native):
        self.native = native
        self.reader = get_ident()
        self.depth = 0
        self.opened = Event()
        self.published = Event()

    def __enter__(self):
        self.native.acquire()
        if get_ident() == self.reader:
            self.depth += 1
        return self

    def __exit__(self, *exc):
        signal = False
        if get_ident() == self.reader:
            self.depth -= 1
            signal = self.depth == 0 and not self.opened.is_set()
        self.native.release()
        if signal:
            self.opened.set()
            assert self.published.wait(10), "second command did not finish"


def check_pose(scene, reading, geometry):
    assert reading.time_s == geometry.time_s
    robot_poses = scene.asset.collision_motion(reading.q)
    for key, motion in robot_poses.items():
        np.testing.assert_array_equal(geometry.poses[key].start, motion.start)
    # This fixture is a full box. Its eight registered material corners give its COM.
    np.testing.assert_allclose(
        geometry.poses[geometry.payload_key].start.mean(axis=0),
        reading.payload_position_m,
        atol=3e-15,
        rtol=0,
    )


def interleaved(monkeypatch, *, paired):
    scene = rail_scene()
    first = scene.step([0, 0.04])
    handoff = _HandoffAtUnlock(scene._lock)
    monkeypatch.setattr(scene, "_lock", handoff)

    def next_command():
        assert handoff.opened.wait(10)
        try:
            return scene.step([0, -0.04])
        finally:
            handoff.published.set()

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(next_command)
        reading, geometry = scene.observe_scene() if paired else (scene.observe(), scene.geometry())
        second = future.result(timeout=10)
    assert second.step_number == 2 and reading is first
    assert second.target_q[1] == -0.04 and first.target_q[1] == 0.04
    assert second.payload_position_m[1] != first.payload_position_m[1]
    return scene, reading, geometry, second


def test_atomic_observe_scene_returns_material_and_actual_force_from_one_publication(monkeypatch):
    scene, reading, geometry, second = interleaved(monkeypatch, paired=True)
    assert reading.time_s == 0.001 and reading.load_time_s == 0
    np.testing.assert_allclose(
        reading.structural_contacts[0]["force_on_payload_N"], [0, 2, 0], atol=1e-12
    )
    check_pose(scene, reading, geometry)
    latest, latest_geometry = scene.observe_scene()
    assert latest is second and latest.load_time_s == reading.time_s
    check_pose(scene, latest, latest_geometry)
    with pytest.raises(ValueError):
        geometry.poses[geometry.payload_key].start.setflags(write=True)
    with pytest.raises(ValueError):
        reading.drive_effort.setflags(write=True)


def test_separate_legacy_calls_can_span_two_real_publications(monkeypatch):
    scene, reading, geometry, second = interleaved(monkeypatch, paired=False)
    assert reading.time_s == 0.001 and geometry.time_s == 0.002
    check_pose(scene, second, geometry)
    assert not np.array_equal(reading.q, second.q)


def test_rejected_physical_step_preserves_both_snapshots_and_the_original_load_clock():
    scene = rail_scene(q=(0, 0.24935), v=(0, 0.08), center=(0.60, -0.22, 0.30), dt=0.005)
    start, g0 = scene.observe_scene()
    assert start.load_time_s is None and start.step_number == 0
    check_pose(scene, start, g0)
    accepted = scene.step([0, 0.24], effort_limit=[0, 0])
    before, geometry = scene.observe_scene()
    assert before is accepted and before.step_number == 1 and before.load_time_s == 0
    with pytest.raises(ValueError, match="joint position"):
        scene.step([0, 0.24], effort_limit=[0, 0])
    after, again = scene.observe_scene()
    assert after is before and after.time_s == 0.005 and again.model is geometry.model
    assert set(again.poses) == set(geometry.poses)
    for key in again.poses:
        np.testing.assert_array_equal(again.poses[key].start, geometry.poses[key].start)
    check_pose(scene, after, again)
