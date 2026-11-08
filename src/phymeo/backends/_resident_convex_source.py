"""Scoped accepted-pose loans for device convex queries.

This bridge does not compute FK, narrow geometry, forces or a physical step.
Its device handles are valid only inside the owning batch's existing lock and
runtime context. The query plan owns its own scratch and publication.
"""

from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np

from ._resident_collision_plan import ResidentCollisionResult, ResidentStaticBinding
from ._resident_tree_binding import ResidentTreeBinding


def _readonly_i32(value):
    array = np.asarray(value, dtype=np.int32)
    return np.frombuffer(array.tobytes(order="C"), dtype=np.int32).reshape(array.shape)


@dataclass
class _Lease:
    active: bool = True

    def require(self):
        if not self.active:
            raise RuntimeError("accepted convex source loan has expired")


@dataclass(frozen=True, eq=False)
class ResidentConvexSourceView:
    """One accepted bank and its original immutable geometry registrations."""

    accepted: object
    model: object
    bindings: tuple
    owner: object
    registration: object
    world_keys: tuple
    shape_keys: tuple
    shape_revisions: tuple
    allowed_pairs: tuple
    shape_rows: np.ndarray
    shape_groups: np.ndarray
    offsets: np.ndarray
    _plan: object = field(repr=False)
    _lease: _Lease = field(repr=False)

    def stage_broadphase(self):
        """Stage the old broad phase against this exact still-borrowed bank."""
        self._lease.require()
        plan = self._plan
        plan._check_open()
        if self.shape_keys:
            bounds, ticks, candidates, pending = plan._read_bounds(self.accepted)
        else:
            bounds = np.empty((0, 6), dtype=np.float32)
            ticks = np.empty(0, dtype=np.int32)
            candidates, pending = (), dict(plan._trees.layouts)
        result = ResidentCollisionResult(
            self.owner,
            self.registration,
            self.accepted.topology_generation,
            self.accepted.state_epoch,
            self.world_keys,
            ticks,
            self.shape_keys,
            self.shape_revisions,
            bounds,
            plan._registered_pairs,
            candidates,
        )
        return result, pending


@contextmanager
def borrow_convex_state(plan):
    """Borrow ordinary-tree/static sources without creating a second runtime."""
    batch = plan._batch
    with batch._lock:
        plan._check_open()
        bindings = plan._convex_bindings
        if len(bindings) != len(plan._shape_keys):
            raise ValueError("convex source registration is incomplete")
        # Affine subclasses have a different device layout and require a later
        # separately validated source adapter. Do not admit them by isinstance.
        if any(
            type(binding) not in (ResidentTreeBinding, ResidentStaticBinding)
            for binding in bindings
        ):
            raise ValueError("convex query currently requires ordinary tree or static bindings")
        allowed = ()
        if plan._space is not None:
            allowed = tuple(
                (plan._shape_keys[int(a)], plan._shape_keys[int(b)])
                for (a, b, _), flag in zip(plan._pairs, plan._space.defaults, strict=True)
                if int(flag) == 5
            )
        with batch._borrow_collision_state(
            owner=plan._owner, topology_generation=plan._topology_generation
        ) as accepted:
            for binding in bindings:
                if type(binding) is ResidentTreeBinding:
                    if accepted.tree is None or binding.world not in accepted.tree.rows:
                        raise ValueError("ordinary accepted tree geometry is unavailable")
            lease = _Lease()
            try:
                yield ResidentConvexSourceView(
                    accepted=accepted,
                    model=plan._model,
                    bindings=bindings,
                    owner=plan._owner,
                    registration=plan._registration,
                    world_keys=plan._world_keys,
                    shape_keys=plan._shape_keys,
                    shape_revisions=plan._shape_revisions,
                    allowed_pairs=allowed,
                    shape_rows=_readonly_i32(plan._tree_rows),
                    shape_groups=_readonly_i32(plan._tree_groups),
                    offsets=_readonly_i32(plan._offsets),
                    _plan=plan,
                    _lease=lease,
                )
            finally:
                lease.active = False
