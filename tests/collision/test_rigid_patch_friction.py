"""Multiple loaded rigid faces transmit both tangential force and torsional traction."""

from contextlib import ExitStack
from dataclasses import replace

import numpy as np
import pytest
from load_cases import scene

import phymeo as ph


def exercise(mode, cache_root, spin, *, reverse=False):
    dtype = np.float64 if mode == "host" else np.float32
    base, state, models, _ = scene("RR", dtype=dtype)
    shapes = base.model.shapes
    laws = {
        keys: replace(law, friction=0.5, damping_s_per_m=0, slip_speed_m_s=0.0625)
        for keys, law in base.layers.items()
    }
    bindings = base.bindings
    model = base.model
    if reverse:
        keys = {
            shape.key: ph.CollisionKey(shape.key.world, 100 - shape.key.body, shape.key.shape)
            for shape in shapes
        }
        model = ph.CollisionModel([replace(shape, key=keys[shape.key]) for shape in shapes])
        bindings = {keys[k]: value for k, value in bindings.items()}
        laws = {(keys[a], keys[b]): law for (a, b), law in laws.items()}
    if spin:
        state.rigid.angular_velocity[2, 2] = spin
    else:
        state.rigid.velocity[2, 0] = 0.5
    with ExitStack() as stack:
        workspace = stack.enter_context(ph.CollisionWorkspace(model, mode, cache_root=cache_root))
        solver = ph.RigidSemiImplicitSolver()
        layer = None
        if mode != "host":
            solver = stack.enter_context(
                ph.TyxRigidSemiImplicitSolver(mode=mode, cache_root=cache_root)
            )
            layer = stack.enter_context(ph.TyxNormalLayer(mode=mode, cache_root=cache_root))
        program = ph.CollisionLoadSystem(model, bindings, laws, layer_operator=layer)
        h = 1 / 512
        first = None
        stats = []
        for _ in range(4):
            result = program.step(
                state,
                ph.CoupledInputs.zeros(0, 3, dtype=dtype),
                models,
                (None, solver),
                h,
                workspace,
            )
            assert result.rigid_report.accepted
            assert result.path.status == "ok"
            forces = result.loads.inputs.rigid.force
            torque = result.loads.inputs.rigid.torque
            moment = np.cross(state.rigid.position, forces).sum(0) + torque.sum(0)
            np.testing.assert_allclose(forces.sum(0), 0, atol=1e-7)
            np.testing.assert_allclose(moment, 0, atol=1e-7)
            np.testing.assert_allclose(
                result.state.rigid.velocity[1:] - state.rigid.velocity[1:],
                h * (forces[1:] + models[1].gravity),
                atol=2e-7,
                rtol=0,
            )
            for c in result.loads.contacts:
                friction_power = sum(s.power_w - s.force_n * s.rate_m_per_s for s in c.sites)
                assert friction_power <= 1e-8
                for s in c.sites:
                    np.testing.assert_allclose(
                        s.application_point_first_m, s.application_point_second_m
                    )
            stats.append(
                (len(result.loads.contacts), [len(pair.sites) for pair in result.loads.contacts])
            )
            first = result if first is None else first
            state = result.state
        if layer is not None:
            assert layer.stats()["completed_launches"] >= 4
        assert first is not None and stats[0] == (2, [12, 12])
        return first, state, stats


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("spin", [0.25, -0.25, 0.0])
def test_sliding_or_spinning_stack_measures_both_body_reactions(tmp_path, spin, reverse):
    frame, final, _ = exercise("host", tmp_path, spin, reverse=reverse)
    loads = frame.loads.inputs.rigid
    if spin:
        # The regularized slip law is linear here. The square has <r^2> = 2*a^2/3.
        net = -0.5 * (2 / 3) * (1 / 8) ** 2 * spin / 0.0625
        np.testing.assert_allclose(loads.force[2], [0, 0, 1], atol=1e-10)
        assert loads.torque[2, 2] == pytest.approx(net)
        assert loads.torque[1, 2] == pytest.approx(-net)
        assert abs(final.rigid.angular_velocity[2, 2]) < abs(spin)
        assert final.rigid.angular_velocity[1, 2] * spin > 0
    else:
        np.testing.assert_allclose(loads.force[2], [-0.5, 0, 1], atol=1e-10)
        np.testing.assert_allclose(loads.force[1], [0.5, 0, 1], atol=1e-10)
        assert final.rigid.velocity[1, 0] > 0 and final.rigid.velocity[2, 0] < 0.5


@pytest.mark.tyx
def test_device_loaded_rigid_torsion(tmp_path, tyx_mode):
    frame, final, _ = exercise(tyx_mode, tmp_path, 0.25)
    assert frame.rigid_report.backend == "tyx:" + tyx_mode
    assert frame.loads.inputs.rigid.torque[2, 2] == pytest.approx(-1 / 48, abs=3e-6)
    assert final.rigid.angular_velocity[1, 2] > 0
