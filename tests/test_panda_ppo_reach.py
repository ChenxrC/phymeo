"""Geometry/reward regressions for the state-only dual-Panda reaching task."""
from importlib import import_module
import os
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def fixture(monkeypatch):
    mj = pytest.importorskip("mujoco")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/"examples"))
    src = import_module("_panda_grasp_insert_scene").build_scene()
    return mj, src, mj.MjModel.from_xml_string(src.xml), import_module("_panda_ppo_reach")


def _check_batch(fixture, torch, device):
    mj,scene,model,mod = fixture
    task = mod.ReachTask(torch,model,device)
    rng = np.random.default_rng(23)
    clock, worlds, repeat = 3, 4, 2
    q0 = np.tile(scene.initial_qpos,(worlds,1)).astype(np.float32)
    q0[:,:14] += rng.uniform(-.15,.15,(worlds,14))
    q0[:,16:19] += rng.uniform(-.006,.006,(worlds,3))
    q = np.tile(q0,(clock,1,1))
    q[:,:,:14] += rng.uniform(-.28,.28,(clock,worlds,14))
    q[:,:,16:19] += rng.uniform(-.015,.015,(clock,worlds,3))
    quat = rng.normal(size=(clock,worlds,4))
    q[:,:,19:23] = quat/np.linalg.norm(quat,axis=-1,keepdims=True)
    v = rng.normal(0,.12,(clock,worlds,22)).astype(np.float32)
    forces = rng.uniform(-1,1,(clock,repeat,worlds,16)).astype(np.float32)
    limits = np.linspace(10,30,16,dtype=np.float32)
    def to_device(a):
        return torch.tensor(a,device=device)
    task.reset(to_device(q0))
    native = mod.audit_numpy(model,q0,q,v,forces,limits)
    d = mj.MjData(model)
    for i in range(clock):
        positions = to_device(q[i])
        poses = task.poses(positions)
        for world in range(worlds):
            d.qpos[:] = q[i,world]
            mj.mj_kinematics(model,d)
            for name,(xyz,rotation) in zip(mod.SITE_NAMES,poses,strict=True):
                site = d.site(name)
                np.testing.assert_allclose(xyz[world].cpu().numpy(),site.xpos,rtol=0,atol=4e-6)
                np.testing.assert_allclose(rotation[world].cpu().numpy(),site.xmat.reshape(3,3),rtol=0,atol=4e-6)
        m = task.measure(positions)
        rew = task.reward(positions,to_device(v[i]),to_device(forces[i]),to_device(limits),measurement=m)
        assert m["features"].shape == (worlds,21) and m["qualified"].dtype == torch.bool
        for key,value in {**m,"reward":rew}.items():
            if key=="qualified":
                np.testing.assert_array_equal(value.cpu().numpy(),native[key][i])
            else:
                np.testing.assert_allclose(value.cpu().numpy(),native[key][i],rtol=2e-5,atol=3e-5)
    assert task.contract["success"]["min_continuous_dwell_s"] == .1


def test_cpu_torch_fk_reward_matches_independent_native(fixture):
    _check_batch(fixture,pytest.importorskip("torch"),"cpu")


@pytest.mark.musa
@pytest.mark.skipif(os.getenv("PHYMEO_TEST_MUSA") != "1",reason="requires an actual MUSA GPU")
def test_musa_fk_reward_matches_independent_native(fixture):
    import torch
    import torch_musa  # noqa: F401 -- register the accelerator

    _check_batch(fixture,torch,"musa:0")


def test_original_reset_and_independent_board_and_pin_errors(fixture):
    torch = pytest.importorskip("torch")
    _,scene,model,mod = fixture
    reach = mod.ReachTask(torch,model,"cpu")
    q = torch.tensor(np.tile(scene.initial_qpos,(3,1)),dtype=torch.float32)
    reach.reset(q)
    start = reach.measure(q)
    assert torch.all(start["distance_m"] > .10)
    assert not torch.any(start["qualified"])
    # Moving only the left arm must not move the measured right hand or pin.
    q[1,0] += .20
    # Moving only the unactuated free body must not alter the measured board.
    q[2,16] += .025
    end = reach.measure(q)
    assert end["distance_m"][0] == end["distance_m"][1]
    assert end["board_tilt_rad"][1] > mod.BOARD_ROTATION_RAD
    assert end["part_drift_m"][1] == 0
    assert end["part_drift_m"][2] > mod.PART_DRIFT_M
    assert end["board_displacement_m"][2] < 1e-6
    assert not torch.any(end["qualified"])
