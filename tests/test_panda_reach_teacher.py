from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def test_axial_waypoints_follow_measured_pose_per_world(monkeypatch):
    torch = pytest.importorskip("torch")
    mj = pytest.importorskip("mujoco")
    examples = Path(__file__).resolve().parents[1] / "examples"
    monkeypatch.syspath_prepend(str(examples))
    from _panda_ppo_reach import ReachTask
    from _panda_reach_teacher import build_waypoints, waypoint_action

    path = examples / "checkpoints" / "panda-reach-s4000"
    model = mj.MjModel.from_xml_path(str(path / "scene.xml"))
    with np.load(path / "train-12.npz") as data:
        resets = data["qpos"][:2].copy()
    targets = torch.from_numpy(build_waypoints(model, resets))
    q = torch.tensor(resets, dtype=torch.float32)
    reach = ReachTask(torch, model, "cpu")
    reach.reset(q)
    stage = torch.zeros(2, dtype=torch.int64)
    task = SimpleNamespace(q=q, reach=reach, target=targets[:,0].clone(), reach_measurement=reach.measure(q))
    waypoint_action(torch, task, targets, stage)
    assert stage.tolist() == [0, 0]  # Commanded setpoint alone cannot unlock the descent.
    for index in range(4):
        q[0, :16] = targets[0, index]
        task.target[0] = q[0, :16]
        task.reach_measurement = reach.measure(q)
        action = waypoint_action(torch, task, targets, stage)
        assert stage.tolist() == [min(index+1, 3), 0]
        assert torch.isfinite(action).all() and action[:,:14].abs().max() <= .900001
        torch.testing.assert_close(action[:,14:], torch.ones((2,2)))
    assert reach.measure(q)["distance_m"][0] < .001


def test_recovery_rejoins_the_axis_only_with_measured_open_fingers(monkeypatch):
    torch = pytest.importorskip("torch")
    mj = pytest.importorskip("mujoco")
    examples = Path(__file__).resolve().parents[1] / "examples"
    monkeypatch.syspath_prepend(str(examples))
    from _panda_ppo_reach import ReachTask
    from _panda_reach_teacher import build_waypoints, recoverable_waypoint_action

    path = examples / "checkpoints" / "panda-reach-s4000"
    model = mj.MjModel.from_xml_path(str(path / "scene.xml"))
    with np.load(path / "train-12.npz") as data:
        resets = data["qpos"][:3].copy()
    q = torch.tensor(resets, dtype=torch.float32)
    reach = ReachTask(torch, model, "cpu")
    reach.reset(q)
    targets = torch.from_numpy(build_waypoints(model, resets))
    q[0,:16], q[1,:16], q[2,:16] = targets[0,2], targets[1,3], targets[2,2]
    q[2,14:16] = .005
    stage = torch.zeros(3, dtype=torch.int64)
    task = SimpleNamespace(q=q, reach=reach, target=q[:,:16].clone(), reach_measurement=reach.measure(q))
    action = recoverable_waypoint_action(torch, task, targets, stage)
    assert stage.tolist() == [3, 3, 0]
    torch.testing.assert_close(action[1,7:14], torch.zeros(7))
