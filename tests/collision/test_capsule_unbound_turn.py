"""A rigid solver's endpoint poses alone cannot rule out a full turn through cloth."""

import numpy as np
import pytest
from capsule_load_cases import create

import phymeo as ph
from phymeo.core.result import StepReport, StepResult


class EndpointSolver:
    def __init__(self, declare, rotates):
        if declare:
            self.rigid_motion_contract = "semi_implicit_world_omega_v1"
        self.rotates = rotates

    def step(self, model, state, inputs, dt):
        returned = ph.RigidBodyState.copy(state)
        returned.angular_velocity[0] = [0.0, 2 * np.pi / dt if self.rotates else 0.0, 0.0]
        return StepResult(returned, StepReport(accepted=True))


@pytest.mark.parametrize(("declared", "rotates"), [(False, False), (False, True), (True, True)])
def test_unbound_motion_cannot_pass_as_a_static_capsule(declared, rotates):
    system, state, (_, rigid) = create()
    # Keep the triangle at z=0 so its material point is independent of the held force.
    cloth = ph.ParticleModel(np.zeros(3), np.zeros(3))
    center = state.rigid.position[0]
    assert center[2] == 0.25
    if rotates:
        # At a quarter turn about +y, the right end of the shaft is at the
        # triangle interior. At both published poses its axis is 0.25 m above.
        face_point = np.array([1 / 3, 1 / 3, 1 / 3]) @ state.particles.position
        quarter_turn_tip = center + [0.0, 0.0, -0.25]
        np.testing.assert_allclose(quarter_turn_tip, face_point, atol=1e-15)
    solver = EndpointSolver(declared, rotates)
    with ph.CollisionWorkspace(system.model) as workspace:
        with pytest.raises(ph.SolverFailure, match="continuous rigid motion|core path rejected"):
            system.step(
                state,
                ph.CoupledInputs.zeros(3, 1),
                (cloth, rigid),
                (ph.XPBDSolver(), solver),
                1 / 64,
                workspace,
            )
        assert workspace.last_result is None and system.last_frame is None
