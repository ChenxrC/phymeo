"""Semi-implicit evolution of a single free root with an eccentric body origin."""

import numpy as np

from ..core.result import StepReport, StepResult
from ..core.rigid import RigidBodyInputs, RigidBodyState
from ..dynamics.floating_robot import FloatingRobotModel, FloatingRobotState
from ..dynamics.rotation import matrix_quaternion, quaternion_matrix
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
    old_rotation = quaternion_matrix(state.q[None, 3:7])[0]
    inertia_frame = None
    initial_orientation = state.q[None, 3:7]
    if np.any(model.root_inertia_frame_xyzw[:3]):
        inertia_frame = quaternion_matrix(model.root_inertia_frame_xyzw[None, :])[0]
        initial_orientation = matrix_quaternion(old_rotation @ inertia_frame)[None, :]
    initial = RigidBodyState(
        np.zeros((1, 3)), initial_orientation, state.v[None, :3], state.v[None, 3:6]
    )
    result = RigidSemiImplicitSolver().step(model.root, initial, inputs, dt)
    if not result.report.accepted:
        return StepResult(state, result.report)
    candidate = result.state
    new_rotation = quaternion_matrix(candidate.orientation)[0]
    new_orientation = candidate.orientation[0]
    if inertia_frame is not None:
        # The rigid solver advances the physical principal frame. Publish the
        # same body-origin frame as the input, with world COM velocity unchanged.
        new_rotation = new_rotation @ inertia_frame.T
        new_orientation = matrix_quaternion(new_rotation)
        if np.dot(new_orientation, state.q[3:7]) < 0:
            new_orientation = -new_orientation
    with np.errstate(over="ignore", invalid="ignore"):
        offset_delta = (new_rotation - old_rotation) @ model.root_com_m
        origin = state.q[:3] + (candidate.position[0] - offset_delta)
        center = origin + new_rotation @ model.root_com_m
    if not np.isfinite(origin).all() or not np.isfinite(center).all():
        return StepResult(
            state,
            StepReport(accepted=False, reason="integrated root origin or COM exceeds FP64 range"),
        )
    q = np.concatenate((origin, new_orientation))
    v = np.concatenate((candidate.velocity[0], candidate.angular_velocity[0]))
    return StepResult(FloatingRobotState(q, v), result.report)
