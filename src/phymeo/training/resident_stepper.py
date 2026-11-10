"""Borrowed physical steppers and explicit per-world publication failures.

A backend's normal step must return exactly one boolean per environment world.
A backend that raises after some worlds advanced must report that fact through
PhysicsStepFailure. Unknown publication timing is explicitly conservative.
"""

from collections.abc import Mapping
from types import MappingProxyType

import numpy as np


def _acceptance_copy(value):
    if not isinstance(value, Mapping):
        raise TypeError("physical step acceptance must be a world-key mapping")
    result = {}
    for key, flag in value.items():
        if not isinstance(flag, (bool, np.bool_)):
            raise TypeError("physical step acceptance values must be boolean")
        result[key] = bool(flag)
    return MappingProxyType(result)


def _exact_acceptance(value, world_keys):
    owned = _acceptance_copy(value)
    if set(owned) != set(world_keys):
        raise ValueError("physical step acceptance must cover exactly the environment worlds")
    return MappingProxyType({key: owned[key] for key in world_keys})


class PhysicsStepFailure(RuntimeError):
    """Physical publication may have occurred, independently of an env transition.

    accepted_worlds is an owned immutable bool mapping. A known complete all-False
    map proves no world advanced. For uncertain publication timing, set
    acceptance_known=False; physics_accepted then conservatively returns True,
    meaning an advance cannot be ruled out, not that every world accepted.
    """

    def __init__(self, message, *, accepted_worlds, acceptance_known=True):
        super().__init__(message)
        if type(acceptance_known) is not bool:
            raise TypeError("acceptance_known must be boolean")
        self._accepted_worlds = _acceptance_copy(accepted_worlds)
        self._acceptance_known = acceptance_known

    @property
    def accepted_worlds(self):
        return self._accepted_worlds

    @property
    def acceptance_known(self):
        return self._acceptance_known

    @property
    def physics_accepted(self):
        return not self.acceptance_known or any(self.accepted_worlds.values())


class ResidentContactStepper:
    """Borrow one real batch/contact plan; call its existing public transaction.

    This object owns no runtime, plan or close operation. Closing the original
    owner or plan invalidates subsequent calls through the original public API.
    """

    __slots__ = ("_batch", "_plan")

    def __init__(self, batch, plan):
        from ..backends._resident_contact_bundle import ResidentContactBundle
        from ..backends._resident_contact_plan import ResidentContactPlan
        from ..backends.tyx_world_batch import TyxWorldBatch

        if not isinstance(batch, TyxWorldBatch):
            raise TypeError("contact stepper requires a TyxWorldBatch owner")
        if not isinstance(plan, (ResidentContactPlan, ResidentContactBundle)):
            raise TypeError(
                "contact stepper requires a ResidentContactPlan or ResidentContactBundle"
            )
        if plan._batch is not batch:
            raise ValueError("contact stepper plan belongs to a different batch owner")
        with batch._lock:
            batch._check_open()
            plan._check_open()
        self._batch, self._plan = batch, plan

    @property
    def batch(self):
        return self._batch

    @property
    def plan(self):
        return self._plan

    def __call__(self, h):
        return self.batch.step_contacts(self.plan, h)
