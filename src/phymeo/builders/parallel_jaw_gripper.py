"""Finite massive parallel-jaw model and bounded effort requests.

This builder does not simulate grasping, impose joint stops, or publish targets
as physical state. Contact and material response remain framework operations.
"""

from dataclasses import dataclass
from numbers import Real

import numpy as np

from ..constraints.robot_contact_patch import RobotContactPatch
from ..core.robot import RobotInputs, RobotModel, RobotState


def _scalar(value, name, *, positive=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar")
    try:
        result = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{name} is not representable") from error
    if not np.isfinite(result) or (positive and result <= 0):
        raise ValueError(f"{name} must be finite" + (" and positive" if positive else ""))
    if value != 0 and result == 0:
        raise ValueError(f"{name} underflows FP64")
    return result


def _array(value, name, shape, *, positive=False):
    raw = np.asarray(value)
    if raw.dtype.kind not in "fiu":
        raise TypeError(f"{name} must contain real numeric values")
    try:
        result = np.array(raw, dtype=np.float64, order="C", copy=True)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{name} is not representable") from error
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name} must have finite shape {shape}")
    if positive and np.any(result <= 0):
        raise ValueError(f"{name} must be positive")
    return result


@dataclass(frozen=True, eq=False)
class ParallelJawGripper:
    """Builder-owned geometry and a separate mutable initial state.

    Joint order is carriage +Z, left finger +Y, right finger -Y. Positive
    finger coordinates close the jaws. Limits bound requested efforts in N;
    they are not a joint-stop model or an actual velocity guarantee.
    """

    model: RobotModel
    initial_state: RobotState
    patches: tuple[RobotContactPatch, RobotContactPatch]
    effort_limits_N: np.ndarray
    half_gap_m: float

    def __post_init__(self):
        limits = _array(self.effort_limits_N, "effort_limits_N", (3,), positive=True)
        limits.flags.writeable = False
        object.__setattr__(self, "effort_limits_N", limits)
        object.__setattr__(
            self, "half_gap_m", _scalar(self.half_gap_m, "half_gap_m", positive=True)
        )

    def opening_m(self, configuration):
        """Return signed opening in metres without changing any state."""
        q = _array(configuration, "configuration", (3,))
        with np.errstate(over="raise", invalid="raise"):
            try:
                value = 2.0 * self.half_gap_m - q[1] - q[2]
            except FloatingPointError as error:
                raise ValueError("opening is not representable") from error
        if not np.isfinite(value):
            raise ValueError("opening is not representable")
        return float(value)


def make_parallel_jaw_gripper(
    *,
    half_gap_m,
    patch_half_extent_m,
    carriage_mass_kg,
    finger_mass_kg,
    inertia_diagonal_kg_m2,
    effort_limits_N,
    gravity_m_s2=(0.0, 0.0, -9.81),
):
    """Build three massive coordinate-prismatic links with finite patches.

    finger_mass_kg is a positive scalar or two positive values (left, right).
    patch_half_extent_m is (half_x, half_z), metres. Inertias are (3,3)
    positive COM principal values in kg*m^2; RobotModel validates them.
    Every configuration array is owned. No cloth nodes or fixed constraints
    are created here.
    """
    half_gap = _scalar(half_gap_m, "half_gap_m", positive=True)
    extent = _array(patch_half_extent_m, "patch_half_extent_m", (2,), positive=True)
    carriage = _scalar(carriage_mass_kg, "carriage_mass_kg", positive=True)
    if isinstance(finger_mass_kg, Real) and not isinstance(finger_mass_kg, (bool, np.bool_)):
        fingers = np.full(2, _scalar(finger_mass_kg, "finger_mass_kg", positive=True))
    else:
        fingers = _array(finger_mass_kg, "finger_mass_kg", (2,), positive=True)
    inertia = _array(inertia_diagonal_kg_m2, "inertia_diagonal_kg_m2", (3, 3), positive=True)
    limits = _array(effort_limits_N, "effort_limits_N", (3,), positive=True)
    gravity = _array(gravity_m_s2, "gravity_m_s2", (3,))
    positions = np.array([[0.0, 0.0, 0.0], [0.0, -half_gap, 0.0], [0.0, half_gap, 0.0]])
    model = RobotModel(
        parents=np.array([-1, 0, 0], dtype=np.int64),
        joint_types=("prismatic", "prismatic", "prismatic"),
        joint_position=positions,
        joint_orientation=np.tile([0.0, 0.0, 0.0, 1.0], (3, 1)),
        joint_axis=np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]]),
        mass=np.array([carriage, fingers[0], fingers[1]]),
        com=np.zeros((3, 3)),
        inertia_diagonal=inertia,
        gravity=gravity,
    )
    half_x, half_z = extent
    vertices = np.array(
        [
            [-half_x, 0.0, -half_z],
            [-half_x, 0.0, half_z],
            [half_x, 0.0, half_z],
            [half_x, 0.0, -half_z],
        ]
    )
    patches = (
        RobotContactPatch("left_finger", 1, vertices),
        RobotContactPatch("right_finger", 2, vertices),
    )
    return ParallelJawGripper(
        model, RobotState(np.zeros(3), np.zeros(3)), patches, limits, half_gap
    )


def gripper_inputs(gripper, *, lift_effort_N, closing_effort_N):
    """Return a new force request; negative finger efforts open the jaws.

    Requests outside a limit reject instead of silently clipping. Actual
    motion is calculated by the runtime, including gravity and contact.
    """
    if not isinstance(gripper, ParallelJawGripper):
        raise TypeError("gripper must be ParallelJawGripper")
    lift = _scalar(lift_effort_N, "lift_effort_N")
    closing = _array(closing_effort_N, "closing_effort_N", (2,))
    effort = np.array([lift, closing[0], closing[1]])
    if np.any(np.abs(effort) > gripper.effort_limits_N):
        raise ValueError("requested gripper effort exceeds its limit")
    return RobotInputs(effort, np.zeros((3, 3)), np.zeros((3, 3)))
