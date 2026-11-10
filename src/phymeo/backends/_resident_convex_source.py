"""Scoped accepted-pose loans for device convex queries.

Geometry always comes from the owning accepted full physical FK bank. Affine
coordinates are never interpreted as body poses. This bridge computes neither
FK nor contact forces and retains no device handles beyond its scoped loan.
"""

from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np

from ._resident_affine_binding import ResidentAffineTreeBinding
from ._resident_collision_plan import ResidentCollisionResult, ResidentStaticBinding
from ._resident_floating_affine_binding import ResidentFloatingAffineTreeBinding
from ._resident_floating_binding import ResidentFloatingTreeBinding
from ._resident_tree_binding import ResidentTreeBinding

_DYNAMIC_BINDINGS = (
    ResidentTreeBinding,
    ResidentAffineTreeBinding,
    ResidentFloatingTreeBinding,
    ResidentFloatingAffineTreeBinding,
)
_LAYOUT_NAMES = ("ordinary", "affine", "floating", "floating_affine")
_WORLD_LAYOUT = {5: 0, 6: 0, 7: 1, 8: 1, 9: 2, 10: 2, 11: 3, 12: 3}


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
    shape_layouts: np.ndarray
    world_layouts: tuple
    sources: tuple = field(repr=False)
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


def _qualified_layouts(plan, accepted, bindings):
    sources = (
        accepted.tree,
        accepted.affine_tree,
        accepted.floating_tree,
        accepted.floating_affine_tree,
    )
    row_tables = (
        plan._tree_rows,
        plan._affine_tree_rows,
        plan._floating_tree_rows,
        plan._floating_affine_tree_rows,
    )
    world_layouts = []
    by_world = {}
    for world in plan._world_keys:
        if world not in accepted.slots:
            raise ValueError("convex source world is not owned by the accepted bank")
        slot = accepted.slots[world]
        kind = accepted.bindings[slot].kind
        if kind not in _WORLD_LAYOUT:
            raise ValueError("convex source requires a supported full physical tree world")
        layout = _WORLD_LAYOUT[kind]
        source = sources[layout]
        if (
            source is None
            or source.owner is not accepted.owner
            or source.topology_generation != accepted.topology_generation
            or source.state_epoch != accepted.state_epoch
            or source.active != accepted.active
            or world not in source.rows
            or world not in source.specs
        ):
            raise ValueError("convex source has an unavailable or stale physical layout")
        world_layouts.append(layout)
        by_world[world] = layout
    rows, layouts = [], []
    for index, binding in enumerate(bindings):
        world = binding.key.world
        if world not in by_world:
            raise ValueError("convex shape world differs from its accepted source")
        layout = by_world[world]
        source = sources[layout]
        if type(binding) is ResidentStaticBinding:
            if any(int(table[index]) != -1 for table in row_tables):
                raise ValueError("static convex shape aliases a physical source row")
            row = -1
        else:
            if type(binding) is not _DYNAMIC_BINDINGS[layout]:
                raise ValueError("convex binding type differs from its physical source layout")
            row = int(row_tables[layout][index])
            if row != source.rows[world] or row < 0:
                raise ValueError("convex binding row differs from its accepted source")
            if any(int(table[index]) != -1 for other, table in enumerate(row_tables) if other != layout):
                raise ValueError("convex binding aliases multiple physical layouts")
            spec = source.specs[world]
            body_count = spec.b if layout >= 2 else spec.n
            if not 0 <= binding.group_index < body_count:
                raise ValueError("convex binding group is outside the full physical body list")
            if int(plan._tree_groups[index]) != binding.group_index:
                raise ValueError("convex group differs from its registered material owner")
        rows.append(row)
        layouts.append(layout)
    return sources, tuple(world_layouts), _readonly_i32(rows), _readonly_i32(layouts)


@contextmanager
def borrow_convex_state(plan):
    """Borrow one of four explicit full physical layouts, never reduced poses."""
    batch = plan._batch
    with batch._lock:
        plan._check_open()
        bindings = plan._convex_bindings
        if len(bindings) != len(plan._shape_keys):
            raise ValueError("convex source registration is incomplete")
        if any(type(binding) not in (*_DYNAMIC_BINDINGS, ResidentStaticBinding) for binding in bindings):
            raise ValueError("convex query requires a supported full physical or static binding")
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
            sources, world_layouts, rows, layouts = _qualified_layouts(plan, accepted, bindings)
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
                    shape_rows=rows,
                    shape_groups=_readonly_i32(plan._tree_groups),
                    shape_layouts=layouts,
                    world_layouts=world_layouts,
                    sources=sources,
                    offsets=_readonly_i32(plan._offsets),
                    _plan=plan,
                    _lease=lease,
                )
            finally:
                lease.active = False
