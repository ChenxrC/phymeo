"""Accepted integration-state observation, without an endpoint CPU forward solve."""

from dataclasses import dataclass

import numpy as np

from .mujoco_runtime import COORDINATE_SYSTEM, MuJoCoControl


@dataclass(frozen=True)
class MuJoCoTyxIntegrationState:
    """Owned immutable arrays plus the original complete accepted-step receipt.

    No endpoint geometry, contact, sensor, acceleration or energy fields are
    provided. Forces in last_step belong to that receipt's evaluation phase,
    never to a newly evaluated endpoint. Use runtime.snapshot() for the latter.
    """

    world: int
    q: np.ndarray
    v: np.ndarray
    act: np.ndarray
    ctrl: np.ndarray
    time: float
    tick: int
    qfrc_applied: np.ndarray
    xfrc_applied: np.ndarray
    held_control: MuJoCoControl
    last_step: object
    coordinate_system: str = COORDINATE_SYSTEM
    observation_kind: str = "accepted-integration-state"
