"""Capsule loads use the whole shaft span and the actual triangle material chart."""

import numpy as np
import pytest
from capsule_load_cases import actual_step, create

import phymeo as ph


@pytest.mark.parametrize("reverse", [False, True])
def test_two_shaft_extremes_share_one_law_with_opposite_force_moment_and_work(reverse):
    system, state, _ = create(reverse=reverse, friction=0.5)
    state.rigid.velocity[0, 0] = 0.5
    with ph.CollisionWorkspace(system.model) as workspace:
        loads = system.sample(state, workspace)
    (pair,) = loads.contacts
    assert pair.force_n == pytest.approx(2.0)
    assert [s.fraction for s in pair.sites] == [0.5, 0.5]
    assert sorted(round(s.application_point_first_m[0], 6) for s in pair.sites) == [-0.25, 0.25]
    np.testing.assert_allclose(loads.inputs.particles.force, [[1 / 3, 0, -2 / 3]] * 3, atol=1e-14)
    np.testing.assert_allclose(loads.inputs.rigid.force, [[-1, 0, 2]], atol=1e-14)
    np.testing.assert_allclose(loads.inputs.rigid.torque, [[0, 0.25, 0]], atol=1e-14)
    assert pair.power_w == pytest.approx(-0.5)
    moment = np.cross(state.particles.position, loads.inputs.particles.force).sum(0)
    moment += np.cross(state.rigid.position, loads.inputs.rigid.force).sum(
        0
    ) + loads.inputs.rigid.torque.sum(0)
    np.testing.assert_allclose(moment, 0, atol=1e-14)


def test_unique_endcap_is_one_site_and_uses_off_axis_torque():
    system, state, _ = create(cap=True)
    state.rigid.position[0, 0] = 0.125
    with ph.CollisionWorkspace(system.model) as work:
        loads = system.sample(state, work)
        (pair,) = loads.contacts
        assert len(pair.sites) == 1 and pair.sites[0].fraction == 1
        assert pair.force_n == pytest.approx(2)
        np.testing.assert_allclose(loads.inputs.rigid.force, [[0, 0, 2]], atol=1e-14)
        np.testing.assert_allclose(loads.inputs.rigid.torque, [[0, 0.25, 0]], atol=1e-14)
        state.rigid.position[0, 2] = 1.0
        assert system.sample(state, work).contacts == ()


def test_actual_host_material_and_rigid_step_receive_the_same_load():
    f, _ = actual_step("host")
    assert f.state.rigid.velocity[0, 2] == pytest.approx(4 / 64)
    np.testing.assert_allclose(f.state.particles.velocity[:, 2], [-(2 / 3) / 64] * 3, atol=1e-13)
    assert f.path.status == "ok" and f.path.pairs[0].status == "separated"
    assert len(f.loads.contacts[0].sites) == 2


def test_capsule_axis_inside_cloth_prevents_publication():
    system, state, _ = create()
    state.rigid.position[0, 2] = 0.0
    with ph.CollisionWorkspace(system.model) as work:
        with pytest.raises(ph.SolverFailure, match="core"):
            system.sample(state, work)
        assert work.last_result is None
