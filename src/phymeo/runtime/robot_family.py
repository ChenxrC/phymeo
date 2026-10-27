"""Explicit local registration for generalized robot coordinates."""

from ..core.robot import RobotInputs, RobotModel, RobotState, validate_robot_step
from .registry import StateFamily


def robot_family():
    return StateFamily(
        "robot",
        RobotModel,
        RobotState,
        RobotInputs,
        ("q", "v"),
        ("effort", "force", "torque"),
        "q",
        validate_robot_step,
    )
