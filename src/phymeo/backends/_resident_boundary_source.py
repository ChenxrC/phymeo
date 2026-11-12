"""Registered rigid boundary materials borrowed from one accepted physical bank.

No HOST FK, current bounds, closest-point calculation, or device ownership.
Only direct full-topology installations are admitted in the M1 source domain.
"""

from contextlib import contextmanager
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from ..constraints.collision_shapes import CollisionModel
from ._resident_affine_binding import ResidentAffineTreeBinding
from ._resident_collision_plan import ResidentStaticBinding
from ._resident_floating_affine_binding import ResidentFloatingAffineTreeBinding
from ._resident_floating_binding import ResidentFloatingTreeBinding
from ._resident_tree_binding import ResidentTreeBinding

DYNAMIC = (
    ResidentTreeBinding,
    ResidentAffineTreeBinding,
    ResidentFloatingTreeBinding,
    ResidentFloatingAffineTreeBinding,
)
LAYOUT = {5: 0, 6: 0, 7: 1, 8: 1, 9: 2, 10: 2, 11: 3, 12: 3}


def owned(value, dtype=None):
    data = np.ascontiguousarray(value, dtype=dtype)
    return np.frombuffer(data.tobytes(), dtype=data.dtype).reshape(data.shape)


def freeze(value):
    if isinstance(value, np.ndarray):
        return owned(value)
    if isinstance(value, dict) or isinstance(value, MappingProxyType):
        return MappingProxyType({key: freeze(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(freeze(v) for v in value)
    return value


def stored_interval(value):
    original = np.asarray(value, np.float64)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        stored = original.astype(np.float32)
    tiny = np.finfo(np.float32).tiny
    if not np.isfinite(stored).all() or np.any((original != 0) & (np.abs(stored) < tiny)):
        raise ValueError("boundary source requires finite normal FP32 or zero")
    lower = stored.copy()
    upper = stored.copy()
    low = stored.astype(np.float64) > original
    high = stored.astype(np.float64) < original
    lower[low] = np.nextafter(lower[low], np.float32(-np.inf))
    upper[high] = np.nextafter(upper[high], np.float32(np.inf))
    if not np.isfinite(lower).all() or not np.isfinite(upper).all():
        raise ValueError("boundary source has no finite outward FP32 bounds")
    return stored, lower, upper


@dataclass
class _Lease:
    active: bool = True

    def require(self):
        if not self.active:
            raise RuntimeError("boundary source loan expired")


@dataclass(frozen=True, eq=False)
class ResidentBoundarySourceView:
    accepted: object
    model: object
    bindings: tuple
    owner: object
    registration: object
    world_keys: tuple
    shape_keys: tuple
    shape_revisions: tuple
    world_layouts: tuple
    shape_rows: np.ndarray
    shape_groups: np.ndarray
    shape_layouts: np.ndarray
    sources: tuple
    _lease: object


def physical_sources(accepted):
    return (
        accepted.tree,
        accepted.affine_tree,
        accepted.floating_tree,
        accepted.floating_affine_tree,
    )


def check_accepted(plan, accepted):
    batch = plan._batch
    if accepted.owner is not plan._owner or accepted.owner is not batch._collision_owner:
        raise ValueError("boundary source owner differs")
    if (
        accepted.topology_generation != plan._topology_generation
        or accepted.topology_generation != batch._topology_generation
        or accepted.state_epoch != batch._state_epoch
        or accepted.active != batch._active
    ):
        raise ValueError("boundary source is stale or standby")
    if accepted.runtime is not getattr(batch, "_runtime", None) or accepted.mode != plan._mode:
        raise ValueError("boundary source runtime/mode differs")
    sources = physical_sources(accepted)
    layouts = []
    for world in plan._world_keys:
        if world not in accepted.slots:
            raise ValueError("boundary world is not owned")
        slot = accepted.slots[world]
        if accepted.bindings[slot].world != world:
            raise ValueError("boundary world-slot identity differs")
        layout = LAYOUT.get(accepted.bindings[slot].kind)
        if layout is None:
            raise ValueError("boundary world needs an accepted four-tree layout")
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
            raise ValueError("boundary physical layout source differs")
        layouts.append(layout)
    return sources, tuple(layouts)


def compile_registration(plan, model, bindings):
    if type(model) is not CollisionModel:
        raise TypeError("boundary query requires CollisionModel")
    bindings = tuple(bindings)
    if len(bindings) != len(model.shapes):
        raise ValueError("boundary query needs exactly one binding per shape")
    table = {}
    for binding in bindings:
        if type(binding) not in (*DYNAMIC, ResidentStaticBinding):
            raise TypeError("unsupported boundary binding type")
        if binding.key in table:
            raise ValueError("duplicate boundary binding")
        table[binding.key] = binding
    if set(table) != {shape.key for shape in model.shapes}:
        raise ValueError("boundary material key membership differs")
    plan._model = model
    plan._bindings = tuple(table[shape.key] for shape in model.shapes)
    plan._shape_keys = tuple(shape.key for shape in model.shapes)
    plan._world_keys = tuple(sorted({key.world for key in plan._shape_keys}))
    plan._shape_revisions = tuple(shape.revision for shape in model.shapes)
    plan._owner = plan._batch._collision_owner
    plan._topology_generation = plan._batch._topology_generation
    plan._registration = object()
    with plan._batch._borrow_boundary_collision_state(
        owner=plan._owner, topology_generation=plan._topology_generation
    ) as accepted:
        sources, world_layouts = check_accepted(plan, accepted)
        world_layout = dict(zip(plan._world_keys, world_layouts, strict=True))
        anchors = {
            w: np.asarray(sources[world_layout[w]].specs[w].anchor, np.float64)
            for w in plan._world_keys
        }
        meta = []
        local = []
        bounds = []
        radii = []
        authored = []
        errors = []
        ids = []
        groups = []
        rows = []
        layouts = []
        offsets = [0]
        body_owners = {}
        physical_owners = {}
        definitions = []
        for shape, binding in zip(model.shapes, plan._bindings, strict=True):
            if shape.kind not in ("triangle", "mesh", "sphere", "capsule"):
                raise ValueError("M1 supports only triangle/mesh against sphere/capsule")
            if shape.kind in ("triangle", "mesh") and shape.radius != 0:
                raise ValueError("boundary surface radius must be zero")
            world = shape.key.world
            layout = world_layout[world]
            source = sources[layout]
            slot = accepted.slots[world]
            body = (world, shape.key.body)
            if type(binding) is ResidentStaticBinding:
                if binding.motion.moving or binding.motion.body_orientation_xyzw is not None:
                    raise ValueError("boundary static installation requires a fixed explicit pose")
                row = group = -1
                points = np.asarray(binding.motion.start, np.float64)
                coordinates = points - anchors[world]
                owner = ("static",)
            else:
                if type(binding) is not DYNAMIC[layout]:
                    raise ValueError("boundary binding differs from physical layout")
                spec = source.specs[world]
                count = spec.b if layout >= 2 else spec.n
                group = binding.group_index
                if binding.world != world or not 0 <= group < count:
                    raise ValueError("boundary physical world/group differs")
                if binding.topology is None or binding.topology != spec.topology:
                    raise ValueError("M1 requires direct complete physical topology installation")
                owning = accepted.bindings[slot].tree_spec
                if layout == 3 and binding.graph_token != owning.graph_token:
                    raise ValueError("boundary floating affine graph differs")
                if getattr(binding, "_source_shape", None) is not None:
                    raise ValueError("M1 asset lowering is not part of direct installation")
                row = int(source.rows[world])
                points = np.asarray(binding.local_points_m, np.float64)
                coordinates = points
                owner = (layout, row, group)
                previous = physical_owners.setdefault((world, *owner), body)
                if previous != body:
                    raise ValueError("one physical group cannot alias different bodies")
            if body_owners.setdefault(body, owner) != owner:
                raise ValueError("one boundary body cannot alias static/dynamic or physical groups")
            if points.shape != (len(shape.vertex_ids), 3) or not np.isfinite(points).all():
                raise ValueError("boundary point order/arity/source differs")
            if (
                shape.kind == "sphere"
                and len(points) != 1
                or shape.kind == "capsule"
                and len(points) != 2
            ):
                raise ValueError("boundary round material arity differs")
            if shape.kind == "triangle" and len(points) != 3:
                raise ValueError("boundary triangle requires all three original points")
            stored, lo, hi = stored_interval(coordinates)
            radius, rlo, rhi = stored_interval(np.asarray([shape.radius]))
            del rlo
            begin = offsets[-1]
            end = begin + len(points)
            offsets.append(end)
            meta.append((layout, row, group, slot, begin, end))
            local.extend(stored)
            bounds.extend(np.c_[lo, hi])
            radii.append((radius[0], rhi[0]))
            authored.append(owned(points))
            errors.append(owned(stored.astype(np.float64) - coordinates))
            ids.append(tuple(shape.vertex_ids))
            rows.append(row)
            groups.append(group)
            layouts.append(layout)
            definitions.append(
                dict(
                    key=shape.key,
                    world=world,
                    body=shape.key.body,
                    layout=layout,
                    row=row,
                    group=group,
                    point_range=(begin, end),
                    point_ids=shape.vertex_ids,
                    topology=None if row < 0 else binding.topology,
                    graph_token=getattr(binding, "graph_token", None),
                    frame="static_anchor" if row < 0 else "body_local",
                )
            )
        plan._shape_rows = owned(rows, np.int32)
        plan._shape_groups = owned(groups, np.int32)
        plan._shape_layouts = owned(layouts, np.int32)
        plan._world_layouts = world_layouts
        plan._offsets = owned(offsets, np.int32)
        plan._constants = {
            "shape_meta": owned(np.asarray(meta, np.int32).reshape(-1, 6)),
            "local": owned(np.asarray(local, np.float32).reshape(-1, 3)),
            "local_bounds": owned(np.asarray(bounds, np.float32).reshape(-1, 6)),
            "radius_bounds": owned(np.asarray(radii, np.float32).reshape(-1, 2)),
        }
        plan._anchors = freeze(anchors)
        plan._source_base = dict(
            registration=plan._registration,
            model=model,
            bindings=tuple(definitions),
            shape_keys=plan._shape_keys,
            shape_revisions=plan._shape_revisions,
            point_ids=tuple(ids),
            facet_ids=tuple(
                tuple(range(1 if s.kind == "triangle" else len(s.triangles)))
                if s.kind in ("triangle", "mesh")
                else ()
                for s in model.shapes
            ),
            facet_vertices=tuple(
                (s.vertex_ids,) if s.kind == "triangle" else s.triangles if s.kind == "mesh" else ()
                for s in model.shapes
            ),
            world_layouts=world_layouts,
            shape_rows=plan._shape_rows,
            shape_groups=plan._shape_groups,
            static_flags=owned(np.asarray(rows) < 0),
            world_anchors=anchors,
            authored_constants=dict(
                points=tuple(authored), radii=owned([s.radius for s in model.shapes], np.float64)
            ),
            uploaded_constants=plan._constants,
            conversion_deltas=dict(
                points=tuple(errors),
                radii=owned(
                    plan._constants["radius_bounds"][:, 0].astype(np.float64)
                    - np.asarray([s.radius for s in model.shapes])
                ),
            ),
        )


@contextmanager
def borrow_boundary_source(plan):
    with plan._batch._borrow_boundary_collision_state(
        owner=plan._owner, topology_generation=plan._topology_generation
    ) as accepted:
        sources, layouts = check_accepted(plan, accepted)
        lease = _Lease()
        try:
            yield ResidentBoundarySourceView(
                accepted,
                plan._model,
                plan._bindings,
                plan._owner,
                plan._registration,
                plan._world_keys,
                plan._shape_keys,
                plan._shape_revisions,
                layouts,
                plan._shape_rows,
                plan._shape_groups,
                plan._shape_layouts,
                sources,
                lease,
            )
        finally:
            lease.active = False


def validate_source(plan, view):
    view._lease.require()
    sources, layouts = check_accepted(plan, view.accepted)
    if (
        view.model is not plan._model
        or view.bindings is not plan._bindings
        or view.owner is not plan._owner
        or view.registration is not plan._registration
        or view.world_keys != plan._world_keys
        or view.shape_keys != plan._shape_keys
        or view.shape_revisions != plan._shape_revisions
        or view.world_layouts != layouts
    ):
        raise ValueError("boundary registration or material source differs")
    for a, b in zip(view.sources, sources, strict=True):
        if a is not b:
            raise ValueError("boundary source is not the borrowed accepted bank")
    for a, b in (
        (view.shape_rows, plan._shape_rows),
        (view.shape_groups, plan._shape_groups),
        (view.shape_layouts, plan._shape_layouts),
    ):
        if not np.array_equal(a, b):
            raise ValueError("boundary material row map differs")
    for index, binding in enumerate(plan._bindings):
        if type(binding) is ResidentStaticBinding:
            continue
        layout = int(plan._shape_layouts[index])
        source = sources[layout]
        if int(plan._shape_rows[index]) != source.rows[binding.world]:
            raise ValueError("boundary current row differs from registered world")
    return view
