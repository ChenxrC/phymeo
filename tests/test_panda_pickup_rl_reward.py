"""Regression tests for signed contact-slot reduction and pickup reward timing."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def runtime(monkeypatch):
    torch = pytest.importorskip("torch")
    mj = pytest.importorskip("mujoco")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    import _panda_pickup_rl_reward as rl
    from _panda_grasp_insert_scene import build_scene

    model = mj.MjModel.from_xml_string(build_scene().xml)
    return torch, mj, model, rl


def test_resident_reduction_uses_slot_identity_and_force_on_workpiece(runtime):
    torch, _, model, rl = runtime
    left, right, rail, other = [model.geom(n).id for n in
                                ("right_left_pad", "right_right_pad", "pickup_rail_left", "pickup_base")]
    handle, shoulder, shaft = [model.geom(n).id for n in rl.PARTS]
    pairs = [(left, handle), (handle, right), (rail, shoulder), (other, shaft), (left, right), (left, handle)]
    slots = np.asarray([(i, 0, *pair) for i, pair in enumerate(pairs)], np.int32)
    raw = dict(counts=np.array([[6, 5, 5], [0, 0, 0]], np.int32),
               contact_map=np.full((2, 7, 3), -1, np.int32),
               contact_wrench=np.full((2, 7, 6), np.nan, np.float32),
               normal_rows=np.zeros((2, len(slots), 32), np.float32))
    # Skip slot 2's force (detected but inactive). The empty second world
    # carries deliberately dirty padding. Neither may contribute to a load.
    for col, (slot, normal, shear) in enumerate([(5, 3., .5), (0, 2., 1.), (1, 4., -.7),
                                                 (2, 90., 60.), (3, 1., -2.), (4, 100., 100.)]):
        raw["contact_map"][0, col] = [slot, col if slot != 2 else -1, int(slot != 2)]
        raw["contact_wrench"][0, col] = [normal, shear, 0., 0., 0., 0.]
        # Normal along X, first friction axis along world Z.
        raw["normal_rows"][0, slot, 12:21] = [1, 0, 0, 0, 0, 1, 0, -1, 0]

    class View:
        def __init__(self, x):
            self.x, self.dtype, self.shape = x, x.dtype, x.shape

        def copy_to_torch(self, out):
            out.copy_(torch.from_numpy(self.x))

    bank = SimpleNamespace(contact_buffers=lambda: {"slot_ids": slots, **{k: View(v) for k, v in raw.items()}})
    reader = rl.ResidentPickupLoads(torch, model, bank, "cpu")
    normal, vertical = reader.read()
    expected_n, expected_up = np.zeros((2, 5, 3)), np.zeros((2, 5, 3))
    expected_n[0, 0, 0], expected_up[0, 0, 0] = 5., 1.5
    expected_n[0, 1, 0], expected_up[0, 1, 0] = 4., .7
    expected_n[0, 4, 2], expected_up[0, 4, 2] = 1., -2.
    np.testing.assert_allclose(normal.numpy(), expected_n, atol=1e-6)
    np.testing.assert_allclose(vertical.numpy(), expected_up, atol=1e-6)
    force = reader.world_force().numpy()
    np.testing.assert_allclose(force[..., 2], expected_up, atol=1e-6)
    expected_x = expected_n.copy()
    expected_x[0, 1, 0] = -4.
    np.testing.assert_allclose(force[..., 0], expected_x, atol=1e-6)
    assert not force[..., 1].any()
    # Reject a claimed active contact with an invalid slot instead of treating
    # it as a genuine zero reaction. Reusing output cannot leak between worlds.
    raw["contact_map"][0, 0, 0] = -1
    new_normal, new_up = reader.read()
    assert (normal.data_ptr(), vertical.data_ptr()) == (new_normal.data_ptr(), new_up.data_ptr())
    assert np.isnan(new_normal[0].numpy()).all() and np.isnan(new_up[0].numpy()).all()
    assert not new_normal[1].any()


def test_batched_reward_rejects_unloaded_flight_and_matches_independent_gate(runtime):
    torch, mj, model, rl = runtime
    from _panda_pickup_loads import PickupLoads

    scorer = PickupLoads(model)
    dt = float(model.opt.timestep)
    engine = rl.PickupRLReward(torch, scorer.weight_n, dt)
    # Rows: actual carry, ballistic, still on fork, other support,
    # slip-and-return, second pad on wrong material, interrupted bilateral grip.
    worlds, samples = 7, 90
    p0 = np.zeros((worlds, 3), np.float32)
    engine.reset(p0)
    data = mj.MjData(model)
    mj.mj_forward(model, data)
    base = scorer.sample(data)
    histories = [{key: [] for key in base} for _ in range(worlds)]
    successes = np.zeros(worlds, int)
    lift_total = np.zeros(worlds)
    for i in range(samples):
        n = np.zeros((worlds, 5, 3), np.float32)
        up = np.zeros_like(n)
        p, wrist, local = p0.copy(), p0.copy(), p0.copy()
        if i >= 1:
            n[:, :2, 0] = scorer.weight_n
            n[1, :2, 0] = 0.
            n[5, 1, 1], n[5, 1, 0] = n[5, 1, 0], 0.
        if i < 3:
            up[:, 2, 1] = n[:, 2, 1] = scorer.weight_n
        else:
            p[:, 2] = wrist[:, 2] = .030
            wrist[1] = 0.
            up[:, :2, 0] = scorer.weight_n * .5
            up[1] = 0.
            up[2, 2, 1] = n[2, 2, 1] = scorer.weight_n
            up[3, 4, 2] = n[3, 4, 2] = scorer.weight_n
            up[5, 1, 1], up[5, 1, 0] = up[5, 1, 0], 0.
            if i == 4:
                local[4, 1] = .006
            if i == 45:
                n[6, 1, 0] = up[6, 1, 0] = 0.
        reward, info = engine.step(n, up, p, wrist, local, np.zeros(worlds, np.float32))
        assert torch.isfinite(reward).all()
        if i >= 3:
            assert reward[1] <= 0.
        successes += info["new_success"].numpy()
        lift_total += info["lift_reward"].numpy()
        for world in range(worlds):
            row = dict(base, time_s=i*dt, normal_n=n[world], up_n=up[world], force_world_n=up[world, ..., None]*scorer.up,
                       part_xyz=p[world], wrist_xyz=wrist[world], part_in_hand_m=local[world])
            for key, value in row.items():
                histories[world][key].append(value)
    offline = [scorer.score({k: np.asarray(v) for k, v in h.items()})["success"] for h in histories]
    assert offline == info["success"].tolist() == [True, False, False, False, False, False, False]
    assert successes.tolist() == [1, 0, 0, 0, 0, 0, 0]
    assert lift_total[0] > 0.
    assert not lift_total[[1, 2, 3, 5]].any()
    engine.reset(p0)
    assert not engine.succeeded.any() and not engine.grip.any() and not engine.supported.any()
