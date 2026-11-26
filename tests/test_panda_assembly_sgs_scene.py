"""Independent CPU geometry/optimization regressions; no GPU or rollout steps.

Native MuJoCo sites and mj_integratePos are the kinematic oracle. These tests
certify coordinate/optimizer contracts, not MUSA execution or grasp success.
"""
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    return import_module("_panda_assembly_scene"), import_module("_panda_assembly_sgs")


@pytest.fixture
def native(modules):
    mj = pytest.importorskip("mujoco")
    scene_module, _ = modules
    scene = scene_module.build_scene()
    model = mj.MjModel.from_xml_string(scene.xml)
    runner = import_module("panda_assembly_sgs_train")
    geometry = runner.Geometry(scene, SimpleNamespace(mj_model=model))
    return mj, scene_module, scene, model, runner, geometry


def _native_sites(mj, model, q):
    data = mj.MjData(model)
    data.qpos[:] = q
    mj.mj_forward(model, data)
    result = {}
    for name in ("assembly_hole", "assembly_entry", "assembly_driver", "assembly_pin_tip"):
        idx = model.site(name).id
        t = np.eye(4)
        t[:3, 3] = data.site_xpos[idx]
        t[:3, :3] = data.site_xmat[idx].reshape(3, 3)
        result[name] = t
    return result, data


def _integrate(mj, model, q, velocity, dt):
    result = np.array(q, dtype=float, copy=True)
    mj.mj_integratePos(model, result, velocity, dt)
    return result


def test_free_part_is_physical_and_not_welded_to_driver(native):
    mj, sm, scene, model, _, _ = native
    assert (model.nq, model.nv, model.nu) == (21, 20, 14)
    assert model.opt.integrator == mj.mjtIntegrator.mjINT_EULER
    assert model.neq == 0
    body = model.body("assembly_part").id
    assert model.body_parentid[body] == 0
    assert model.body_mass[body] == pytest.approx(.12)
    np.testing.assert_allclose(model.body_ipos[body], sm.PART_COM, atol=1e-15)
    assert np.all(model.body_inertia[body] > 1e-5)
    assert model.joint("assembly_part_free").type == mj.mjtJoint.mjJNT_FREE
    assert model.geom("assembly_pin").bodyid == body
    assert model.geom("assembly_pin_adapter").bodyid == body
    assert model.geom("assembly_pin_adapter").type == mj.mjtGeom.mjGEOM_CAPSULE
    assert model.site("assembly_driver").bodyid != body
    assert scene.initial_state().v.shape == (20,)


def test_part_and_arm_move_independently_without_state_rebinding(native):
    mj, _, scene, model, _, _ = native
    q = np.array(scene.initial_qpos)
    base, _ = _native_sites(mj, model, q)
    np.testing.assert_allclose(base["assembly_driver"], base["assembly_pin_tip"], atol=2e-14)
    v = np.zeros(20)
    v[14:17] = (.03, -.02, .01)
    v[17:] = (.12, -.19, .07)
    moved, _ = _native_sites(mj, model, _integrate(mj, model, q, v, 1.))
    np.testing.assert_array_equal(moved["assembly_driver"], base["assembly_driver"])
    np.testing.assert_array_equal(moved["assembly_hole"], base["assembly_hole"])
    assert np.linalg.norm(moved["assembly_pin_tip"]-base["assembly_pin_tip"]) > .01
    v[:] = 0
    v[9] = .13
    arm_moved, _ = _native_sites(mj, model, _integrate(mj, model, q, v, 1.))
    np.testing.assert_array_equal(arm_moved["assembly_pin_tip"], base["assembly_pin_tip"])
    assert np.linalg.norm(arm_moved["assembly_driver"]-base["assembly_driver"]) > .01


@pytest.mark.parametrize("perturbed", [False, True])
def test_urdf_and_free_part_fk_match_native_sites(native, perturbed):
    mj, sm, scene, model, _, _ = native
    q = np.asarray(scene.initial_qpos)
    if perturbed:
        q = _integrate(mj, model, q, np.linspace(-.18, .21, model.nv), 1.)
    sites, data = _native_sites(mj, model, q)
    fk = sm.forward_kinematics(scene, q)
    for name, expected in sites.items():
        np.testing.assert_allclose(fk[name], expected, atol=5e-13, rtol=0)
    expected_com = fk["assembly_part"] @ np.r_[sm.PART_COM, 1.]
    np.testing.assert_allclose(data.xipos[model.body("assembly_part").id], expected_com[:3], atol=5e-13)
    metrics = sm.assembly_metrics(scene, q)
    local = sites["assembly_hole"][:3, :3].T @ (sites["assembly_pin_tip"][:3, 3]-sites["assembly_hole"][:3, 3])
    np.testing.assert_allclose(metrics["tip_in_board_frame_m"], local, atol=5e-13)
    assert metrics["insertion_depth_m"] == pytest.approx(local[2]+.006, abs=5e-13)
    assert metrics["physical_success"] is None


def test_reset_helper_aligns_driver_but_metrics_require_actual_free_state(native):
    mj, sm, scene, model, _, _ = native
    q14 = sm.HOME.copy()
    q14[8] += .11
    initial = sm.initial_qpos_for_arms(scene, q14)
    assert initial.shape == (21,)
    np.testing.assert_array_equal(initial[:14], q14)
    sites, _ = _native_sites(mj, model, initial)
    np.testing.assert_allclose(sites["assembly_pin_tip"], sites["assembly_driver"], atol=5e-13)
    with pytest.raises(ValueError, match="q21"):
        sm.assembly_metrics(scene, q14)
    with pytest.raises(ValueError, match="q21"):
        sm.task_vector(scene, q14)
    assert "assembly_pin_tip" not in sm.forward_kinematics(scene, q14)
    np.testing.assert_allclose(sm.task_vector(scene, scene.initial_qpos), [0., 0., -.024, 0.], atol=5e-13)


@pytest.mark.parametrize("perturbed", [False, True])
def test_runner_relative_depth_and_angle_jacobian_matches_generic_integrate(native, perturbed):
    mj, _, scene, model, _, kin = native
    q = np.array(scene.initial_qpos)
    if perturbed:
        # Tilts both the board and independent part; do not test only the easy
        # aligned case where an incorrect angular-Z shortcut would pass.
        q = _integrate(mj, model, q, np.linspace(-.12, .23, model.nv), 1.)
    kin.update(q, np.zeros(model.nv))
    _, _, _, _, _, _, jp, ja, _ = kin.observe()
    home, _ = _native_sites(mj, model, scene.initial_qpos)
    zero = home["assembly_hole"][:3, :3].T @ home["assembly_driver"][:3, :3]

    def measured(state):
        sites, _ = _native_sites(mj, model, state)
        a, b = sites["assembly_hole"], sites["assembly_pin_tip"]
        p = a[:3, :3].T @ (b[:3, 3]-a[:3, 3])
        orientation = a[:3, :3].T @ b[:3, :3] @ zero.T
        return p, np.arctan2(orientation[1, 0], orientation[0, 0])

    numerical_p, numerical_angle = np.empty((3, model.nv)), np.empty(model.nv)
    h = 2e-7
    for dof in range(model.nv):
        velocity = np.eye(model.nv)[dof]
        plus, ap = measured(_integrate(mj, model, q, velocity, h))
        minus, am = measured(_integrate(mj, model, q, velocity, -h))
        numerical_p[:, dof] = (plus-minus)/(2*h)
        numerical_angle[dof] = np.arctan2(np.sin(ap-am), np.cos(ap-am))/(2*h)
    np.testing.assert_allclose(jp, numerical_p, atol=2e-8, rtol=2e-7)
    np.testing.assert_allclose(ja, numerical_angle, atol=2e-8, rtol=2e-7)
    # The depth coordinate differs from local z only by a constant entrance.
    np.testing.assert_allclose(jp[2], numerical_p[2], atol=2e-8, rtol=2e-7)
    assert np.linalg.norm(jp[:, :7]) > .01
    assert np.linalg.norm(jp[:, 14:]) > .1
    np.testing.assert_allclose(jp[:, 7:14], 0., atol=1e-14)


def test_mount_com_frame_matches_native_free_body_com_and_jacobian(native):
    mj, _, scene, model, runner, kin = native
    q = _integrate(mj, model, scene.initial_qpos, np.linspace(-.1, .2, model.nv), 1.)
    kin.update(q, np.zeros(model.nv))
    actual = runner.mount_frame(kin.frames()[2])
    body = model.body("assembly_part").id
    jp, jr = np.empty((3, model.nv)), np.empty((3, model.nv))
    mj.mj_jacBodyCom(model, kin.data, jp, jr, body)
    np.testing.assert_allclose(actual.x, kin.data.xipos[body], atol=5e-13)
    np.testing.assert_allclose(actual.jp, jp, atol=5e-13)
    np.testing.assert_allclose(actual.jr, jr, atol=5e-13)


@pytest.mark.parametrize("mode", ["alphanist", "sgs"])
def test_sampler_is_nonuniform_and_tracks_categories_independently(modules, mode):
    _, sgs = modules
    sampler = sgs.StateSampler(3, 4011, history=4, mode=mode)
    other_task = sgs.StateSampler(3, 4011, history=4, mode=mode)
    untouched = other_task.snapshot()
    sampler.update([0]*4+[1]*4+[2]*4, [0]*4+[0, 1, 0, 1]+[1]*4)
    np.testing.assert_array_equal(sampler.rates(), [0., .5, 1.])
    probabilities = sampler.probabilities()
    assert probabilities[1] > 2*probabilities[0]
    assert probabilities[0] == pytest.approx(probabilities[2])
    draws = np.bincount(sampler.sample(20000), minlength=3)/20000
    np.testing.assert_allclose(draws, probabilities, atol=.02, rtol=0)
    assert other_task.snapshot() == untouched
    # Repeated IDs wrap only their own rolling history; no class pooling.
    sampler.update([0]*5, [1, 1, 1, 1, 0])
    np.testing.assert_array_equal(sampler.rates(), [.75, .5, 1.])
    restored = sgs.StateSampler.restore(sampler.snapshot())
    np.testing.assert_array_equal(restored.sample(32), sampler.sample(32))


def test_validation_is_balanced_and_does_not_update_training_sampler(modules):
    _, sgs = modules
    sampler = sgs.StateSampler(3, 42)
    before = sampler.snapshot()
    indices = sgs.uniform_validation(6, 14, 8161)
    assert sorted(np.bincount(indices, minlength=6)) == [2, 2, 2, 2, 3, 3]
    np.testing.assert_array_equal(indices, sgs.uniform_validation(6, 14, 8161))
    assert sampler.snapshot() == before


@pytest.mark.parametrize("outcome", [0, 1])
def test_cem_all_equal_success_preserves_mean_and_exploration(modules, outcome):
    _, sgs = modules
    cem = sgs.TerminalCEM([.2, -.4], [.5, .7], 512, population=6)
    mean, std = cem.mean.copy(), cem.std.copy()
    pending = cem.ask()
    np.testing.assert_array_equal(cem.ask(), pending)
    scores = cem.tell(pending, np.full((6, 4), outcome))
    np.testing.assert_array_equal(scores, np.full(6, outcome))
    np.testing.assert_array_equal(cem.mean, mean)
    np.testing.assert_array_equal(cem.std, std)
    assert cem.generation == 1


def test_cem_cutoff_ties_share_elite_mass_instead_of_candidate_order(modules):
    _, sgs = modules
    cem = sgs.TerminalCEM([.2, -.4], [.5, .7], 512, population=6,
                          elite_fraction=1/3, smoothing=1., min_std=.001)
    candidates = cem.ask()
    # Elite mass two: one strictly better, plus half of each cutoff tie.
    wins = [[1, 1], [1, 0], [0, 1], [0, 0], [0, 0], [0, 0]]
    weights = np.array([.5, .25, .25, 0., 0., 0.])
    expected_mean = weights @ candidates
    expected_std = np.maximum(.001, np.sqrt(weights @ (candidates-expected_mean)**2))
    cem.tell(candidates, wins)
    np.testing.assert_allclose(cem.mean, expected_mean, atol=1e-15)
    np.testing.assert_allclose(cem.std, expected_std, atol=1e-15)


def test_optimizer_and_sampler_reject_nonterminal_rewards(modules):
    _, sgs = modules
    sampler = sgs.StateSampler(3, 0)
    before = sampler.snapshot()
    with pytest.raises(ValueError, match="binary"):
        sampler.update([0, 1], [0., .25])
    assert sampler.snapshot() == before
    cem = sgs.TerminalCEM([0.], [.2], 0, population=2)
    candidates = cem.ask()
    with pytest.raises(ValueError, match="binary"):
        cem.tell(candidates, [[.1], [1.]])
    np.testing.assert_array_equal(cem.ask(), candidates)
