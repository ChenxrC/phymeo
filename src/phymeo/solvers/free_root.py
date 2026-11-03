"""Semi-implicit evolution of a single free root with an eccentric body origin."""

import numpy as np

from ..core.result import StepReport, StepResult
from ..core.rigid import RigidBodyInputs, RigidBodyState
from ..dynamics.floating_robot import FloatingRobotModel, FloatingRobotState
from ..dynamics.rotation import quaternion_matrix
from .rigid_semi_implicit import RigidSemiImplicitSolver


def step_free_root(
    model: FloatingRobotModel, state: FloatingRobotState, inputs: RigidBodyInputs, dt: float
) -> StepResult[FloatingRobotState]:
    """Advance one isolated massive root using world force and world torque about its COM.

    Translation drifts with the updated COM velocity. Orientation advances by
    the exponential of the updated world angular velocity. The public XYZ is
    always the body origin. A rejected attempted step returns the input pose
    and velocity intact.
    """
    if not isinstance(model, FloatingRobotModel) or not isinstance(state, FloatingRobotState):
        raise TypeError("expected a FloatingRobotModel and FloatingRobotState")
    if model.children is not None:
        raise ValueError("step_free_root supports a single root without jointed children")
    if len(state.q) != 7:
        raise ValueError("root-only state must have seven coordinates")
    # Integrate in world-parallel axes translated to the initial COM. Rigid free
    # dynamics is translation invariant; the integrated position is its displacement.
    initial = RigidBodyState(
        np.zeros((1, 3)), state.q[None, 3:7], state.v[None, :3], state.v[None, 3:6]
    )
    result = RigidSemiImplicitSolver().step(model.root, initial, inputs, dt)
    if not result.report.accepted:
        return StepResult(state, result.report)
    candidate = result.state
    old_rotation = quaternion_matrix(initial.orientation)[0]
    new_rotation = quaternion_matrix(candidate.orientation)[0]
    with np.errstate(over="ignore", invalid="ignore"):
        offset_delta = (new_rotation - old_rotation) @ model.root_com_m
        origin = state.q[:3] + (candidate.position[0] - offset_delta)
        center = origin + new_rotation @ model.root_com_m
    if not np.isfinite(origin).all() or not np.isfinite(center).all():
        return StepResult(
            state,
            StepReport(accepted=False, reason="integrated root origin or COM exceeds FP64 range"),
        )
    q = np.concatenate((origin, candidate.orientation[0]))
    v = np.concatenate((candidate.velocity[0], candidate.angular_velocity[0]))
    return StepResult(FloatingRobotState(q, v), result.report)
