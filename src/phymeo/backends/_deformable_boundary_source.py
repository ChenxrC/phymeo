"""Open deformable material surfaces borrowed from accepted D1 node banks.

This adapter does not create a closed collision mesh or a rigid pose. It supplies
the existing boundary plan with current device points and complete source proof.
"""

from contextlib import contextmanager
from dataclasses import dataclass
from numbers import Integral

import numpy as np

from ..constraints.collision_shapes import CollisionKey, CollisionShape
from ..geometry._triangle_distance_dispatch import (
    BoundaryDistanceCapacityError,
    BoundaryDistanceLimits,
)
from ._resident_boundary_source import ResidentBoundarySourceView, freeze, owned, stored_interval
from ._resident_collision_plan import ResidentStaticBinding


def _integer(value, name):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Integral)
        or not 0 <= value <= 2**31 - 1
    ):
        raise ValueError(name + " requires a nonnegative int32")
    return int(value)


@dataclass(frozen=True)
class DeformableBoundarySurface:
    """One original D1 surface; its facets may form an open material patch."""

    key: CollisionKey
    surface_key: int
    group: int = 1
    mask: int = 2**31 - 1
    revision: int = 0

    def __post_init__(self):
        if type(self.key) is not CollisionKey:
            raise TypeError("deformable surface requires CollisionKey")
        for name in ("surface_key", "group", "mask", "revision"):
            object.__setattr__(self, name, _integer(getattr(self, name), name))


@dataclass(frozen=True)
class DeformableBoundaryModel:
    """Explicit open surfaces and fixed sphere/capsule material declarations."""

    shapes: tuple
    excluded: tuple = ()

    def __post_init__(self):
        shapes = tuple(self.shapes)
        if len(shapes) > 256:
            raise ValueError("D2 shape capacity is 256")
        if any(type(s) not in (DeformableBoundarySurface, CollisionShape) for s in shapes):
            raise TypeError("D2 shapes must be surfaces or static round declarations")
        if any(type(s) is CollisionShape and s.kind not in ("sphere", "capsule") for s in shapes):
            raise ValueError("D2 static shapes support only sphere/capsule")
        shapes = tuple(sorted(shapes, key=lambda s: s.key))
        keys = {s.key for s in shapes}
        if len(keys) != len(shapes):
            raise ValueError("duplicate D2 shape key")
        surfaces = [
            (s.key.world, s.surface_key) for s in shapes if type(s) is DeformableBoundarySurface
        ]
        if len(set(surfaces)) != len(surfaces):
            raise ValueError("one D1 surface cannot alias multiple shape keys")
        ownership = {}
        for shape in shapes:
            identity = (
                ("surface", shape.surface_key)
                if type(shape) is DeformableBoundarySurface
                else ("static",)
            )
            if ownership.setdefault((shape.key.world, shape.key.body), identity) != identity:
                raise ValueError("one D2 body cannot alias static/dynamic material owners")
        excluded = []
        for pair in self.excluded:
            if len(pair) != 2 or any(type(k) is not CollisionKey for k in pair):
                raise ValueError("excluded pair requires two CollisionKeys")
            a, b = sorted(pair)
            if a == b or a not in keys or b not in keys or a.world != b.world:
                raise ValueError("excluded pair must name two registered same-world shapes")
            excluded.append((a, b))
        if len(set(excluded)) != len(excluded):
            raise ValueError("duplicate excluded pair")
        object.__setattr__(self, "shapes", shapes)
        object.__setattr__(self, "excluded", tuple(sorted(excluded)))


@dataclass(frozen=True)
class _Surface:
    key: CollisionKey
    vertex_ids: tuple
    triangles: tuple
    group: int
    mask: int
    revision: int
    kind: str = "deformable_surface"
    radius: float = 0.0


@dataclass(frozen=True)
class _Model:
    shapes: tuple
    excluded: tuple


@dataclass(frozen=True, eq=False)
class _Accepted:
    owner: object
    topology_generation: int
    state_epoch: int
    active: int
    world_keys: tuple
    runtime: object
    mode: str


@dataclass
class _Lease:
    loan: object
    active: bool = True

    def require(self):
        if not self.active:
            raise RuntimeError("deformable boundary loan expired")
        self.loan.revision  # Checks actual owner, active bank, revision and scope.


class DeformableBoundaryAdapter:
    source_kind = "deformable-node-device-v1"

    def __init__(self, batch):
        self.batch = batch

    def compile_registration(self, plan, model, bindings):
        if plan._batch is not self.batch or type(model) is not DeformableBoundaryModel:
            raise TypeError("D2 requires its producer and DeformableBoundaryModel")
        bindings = tuple(bindings)
        table = {}
        for binding in bindings:
            if type(binding) is not ResidentStaticBinding:
                raise TypeError("D2 static rounds require ResidentStaticBinding")
            if binding.key in table:
                raise ValueError("duplicate D2 static binding")
            if binding.motion.moving or binding.motion.body_orientation_xyzw is not None:
                raise ValueError("D2 round installation must be a fixed explicit pose")
            table[binding.key] = binding
        rounds = {s.key for s in model.shapes if type(s) is CollisionShape}
        if set(table) != rounds:
            raise ValueError("D2 static binding key membership differs")
        plan._owner = self.batch._collision_owner
        plan._registration = object()
        plan._topology_generation = self.batch._topology_generation
        plan._world_keys = self.batch.world_keys
        slots = {w: i for i, w in enumerate(plan._world_keys)}
        plan._shape_keys = tuple(s.key for s in model.shapes)
        plan._shape_revisions = tuple(s.revision for s in model.shapes)
        plan._world_layouts = ("deformable_nodes",) * len(slots)
        definitions, resolved, selected, surfaces = [], [], [], {}
        meta, indices, local, bounds, radii, offsets = [], [], [], [], [], [0]
        authored, errors, point_ids, facets, triangles = [], [], [], [], []
        dynamic_worlds = set()
        with self.batch.borrow_vertices(expected_revision=self.batch.revision) as loan:
            for declaration in model.shapes:
                world = declaration.key.world
                if world not in slots:
                    raise ValueError("D2 world is not owned by this producer")
                slot = slots[world]
                if type(declaration) is DeformableBoundarySurface:
                    matches = [
                        (i, s)
                        for i, s in enumerate(loan.surfaces(world))
                        if s.key == declaration.surface_key
                    ]
                    if len(matches) != 1:
                        raise ValueError("D2 surface key is not registered by this producer")
                    group, surface = matches[0]
                    used = set(map(int, surface.oriented_point_ids.flat))
                    all_ids = tuple(map(int, loan.point_ids(world)))
                    ids = tuple(i for i in all_ids if i in used)
                    native = tuple(i for i, value in enumerate(all_ids) if value in used)
                    shape = _Surface(
                        declaration.key,
                        ids,
                        tuple(map(tuple, surface.oriented_point_ids.tolist())),
                        declaration.group,
                        declaration.mask,
                        declaration.revision,
                    )
                    row = slot
                    points = np.zeros((len(ids), 3), np.float64)
                    facet_ids = surface.facet_ids
                    oriented = surface.oriented_point_ids
                    dynamic_worlds.add(world)
                    surfaces[declaration.key] = dict(
                        surface_key=surface.key,
                        facet_ids=facet_ids,
                        oriented_point_ids=oriented,
                        native_indices=surface.native_indices,
                        gather_node_indices=owned(native, np.int32),
                        point_ids=owned(ids, np.int64),
                    )
                    binding = declaration
                else:
                    shape, binding = declaration, table[declaration.key]
                    ids, row, group = shape.vertex_ids, -1, -1
                    points = np.asarray(binding.motion.start, np.float64)
                    if points.shape != (len(ids), 3):
                        raise ValueError("D2 static material point arity differs")
                    native = (-1,) * len(ids)
                    facet_ids, oriented = (), ()
                end = offsets[-1] + len(ids)
                if end > 8192:
                    raise ValueError("D2 packed point capacity is 8192")
                stored, lo, hi = stored_interval(points)
                radius, _, upper = stored_interval(np.asarray([shape.radius]))
                meta.append((slot, row, group, slot, offsets[-1], end))
                indices.extend(native)
                local.extend(stored)
                bounds.extend(np.c_[lo, hi])
                radii.append((radius[0], upper[0]))
                offsets.append(end)
                resolved.append(shape)
                selected.append(binding)
                authored.append(owned(points))
                errors.append(owned(stored.astype(np.float64) - points))
                point_ids.append(tuple(ids))
                facets.append(tuple(map(int, facet_ids)))
                triangles.append(tuple(map(tuple, oriented)))
                definitions.append(
                    dict(
                        key=shape.key,
                        world=world,
                        body=shape.key.body,
                        frame="accepted_node_world" if row >= 0 else "fixed_world",
                        row=row,
                        group=group,
                        point_range=tuple(offsets[-2:]),
                        point_ids=tuple(ids),
                    )
                )
            # Check all topology limits before allocating any device storage.
            hard = BoundaryDistanceLimits()
            group_count = 0
            for shape in resolved:
                if shape.kind != "deformable_surface":
                    continue
                eligible = [
                    r
                    for r in resolved
                    if r.kind in ("sphere", "capsule")
                    and r.key.world == shape.key.world
                    and r.key.body != shape.key.body
                    and shape.group & r.mask
                    and r.group & shape.mask
                    and tuple(sorted((shape.key, r.key))) not in model.excluded
                ]
                if eligible:
                    group_count += 1
                    for name, required in (
                        ("vertices", len(shape.vertex_ids)),
                        ("faces", len(shape.triangles)),
                        ("segments", len(eligible)),
                    ):
                        if required > getattr(hard, name):
                            raise BoundaryDistanceCapacityError(name, required, getattr(hard, name))
            if group_count > hard.groups:
                raise BoundaryDistanceCapacityError("groups", group_count, hard.groups)
            borrowed = 12 * sum(int(loan.position(w).shape[0]) for w in dynamic_worlds) + 4 * len(
                slots
            )
            plan._counts.update(
                native_owned_limit=plan._memory_limit,
                borrowed_device_bytes=borrowed,
                actual_bank_host_readback_bytes=0,
                material_delta_host_bytes=24 * offsets[-1],
            )
        plan._model = _Model(tuple(resolved), model.excluded)
        plan._bindings = tuple(selected)
        plan._shape_rows = owned([m[1] for m in meta], np.int32)
        plan._shape_groups = owned([m[2] for m in meta], np.int32)
        plan._shape_layouts = owned([m[0] for m in meta], np.int32)
        plan._offsets = owned(offsets, np.int32)
        plan._anchors = freeze({w: np.zeros(3, np.float64) for w in slots})
        plan._constants = dict(
            shape_meta=owned(np.asarray(meta, np.int32).reshape(-1, 6)),
            node_indices=owned(indices, np.int32),
            local=owned(np.asarray(local, np.float32).reshape(-1, 3)),
            local_bounds=owned(np.asarray(bounds, np.float32).reshape(-1, 6)),
            radius_bounds=owned(np.asarray(radii, np.float32).reshape(-1, 2)),
        )
        plan._source_base = dict(
            registration=plan._registration,
            model=model,
            bindings=tuple(definitions),
            producer_binding_sha256=self.batch.binding_sha256,
            producer_topology_generation=0,
            shape_keys=plan._shape_keys,
            shape_revisions=plan._shape_revisions,
            point_ids=tuple(point_ids),
            facet_ids=tuple(facets),
            facet_vertices=tuple(triangles),
            surfaces=freeze(surfaces),
            world_layouts=plan._world_layouts,
            shape_rows=plan._shape_rows,
            shape_groups=plan._shape_groups,
            static_flags=owned(plan._shape_rows < 0),
            world_anchors=plan._anchors,
            dynamic_worlds=tuple(w for w in slots if w in dynamic_worlds),
            authored_constants=dict(
                points=tuple(authored), radii=owned([s.radius for s in resolved], np.float64)
            ),
            uploaded_constants=plan._constants,
            conversion_deltas=dict(
                points=tuple(errors),
                radii=owned(
                    plan._constants["radius_bounds"][:, 0].astype(np.float64)
                    - np.asarray([s.radius for s in resolved])
                ),
            ),
        )

    @contextmanager
    def borrow_source(self, plan):
        with self.batch.borrow_vertices(expected_revision=self.batch.revision) as loan:
            lease = _Lease(loan)
            accepted = _Accepted(
                plan._owner,
                plan._topology_generation,
                loan.revision,
                loan.active_bank,
                loan.world_keys,
                loan.runtime,
                loan.mode,
            )
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
                    plan._world_layouts,
                    plan._shape_rows,
                    plan._shape_groups,
                    plan._shape_layouts,
                    tuple(loan.position(w) for w in plan._world_keys),
                    lease,
                )
            finally:
                lease.active = False

    def validate_source(self, plan, view):
        view._lease.require()
        loan, accepted = view._lease.loan, view.accepted
        if (
            plan._batch is not self.batch
            or view.owner is not plan._owner
            or accepted.owner is not self.batch._collision_owner
            or view.registration is not plan._registration
            or view.model is not plan._model
            or view.bindings is not plan._bindings
            or view.world_keys != plan._world_keys
            or loan.world_keys != plan._world_keys
            or view.shape_keys != plan._shape_keys
            or view.shape_revisions != plan._shape_revisions
            or view.world_layouts != plan._world_layouts
            or accepted.state_epoch != loan.revision
            or accepted.active != loan.active_bank
            or accepted.topology_generation != self.batch._topology_generation
            or accepted.runtime is not plan._runtime
            or accepted.mode != plan._mode
            or loan.binding_sha256 != plan._source_base["producer_binding_sha256"]
        ):
            raise ValueError("D2 accepted source identity, epoch or registration differs")
        for w, array in zip(plan._world_keys, view.sources, strict=True):
            if array is not loan.position(w):
                raise ValueError("D2 source is not the borrowed accepted position bank")
        for actual, expected in (
            (view.shape_rows, plan._shape_rows),
            (view.shape_groups, plan._shape_groups),
            (view.shape_layouts, plan._shape_layouts),
        ):
            if actual is not expected:
                raise ValueError("D2 material mapping identity differs")
        return view

    def logical_ticks(self, plan, view, ticks):
        self.validate_source(plan, view)
        if ticks.dtype != np.int32 or ticks.shape != (len(plan._world_keys),) or np.any(ticks < 0):
            raise RuntimeError("D2 accepted ticks are malformed")
        accepted = self.batch.snapshot()
        if any(int(ticks[i]) != accepted[world].tick for i, world in enumerate(plan._world_keys)):
            raise RuntimeError("D2 device tick differs from the current accepted publication")
        return [int(t) for t in ticks]

    def source_prepass_bytes(self, plan):
        if not plan._shape_keys:
            return 0
        return 36 * int(plan._offsets[-1]) + 16 * len(plan._shape_keys) + 4 * len(plan._world_keys)

    def empty_source(self, plan, view):
        self.validate_source(plan, view)
        values = []
        for world in plan._world_keys:
            tick = plan._download(view._lease.loan.tick(world), "source/tick:" + str(world))
            if tick.shape != (1,) or tick.dtype != np.int32:
                raise RuntimeError("D2 empty source tick buffer differs")
            values.append(tick[0])
        ticks = np.asarray(values, np.int32)
        self.logical_ticks(plan, view, ticks)
        return (
            {},
            dict(
                points=np.empty((0, 3), np.float32),
                point_bounds=np.empty((0, 6), np.float32),
                stamps=np.empty((0, 4), np.int32),
                material_deltas=np.empty((0, 3), np.float64),
                actual_bank_positions={},
            ),
            ticks,
        )

    def gather(self, plan, view, ledger):
        from ._tyx_deformable_boundary import gather_deformable_points

        self.validate_source(plan, view)
        s, n, w = len(plan._shape_keys), int(plan._offsets[-1]), len(plan._world_keys)
        outputs = {
            name: plan._allocate(ledger, shape=shape, dtype=dtype)
            for name, shape, dtype in (
                ("points", (n, 3), np.float32),
                ("point_bounds", (n, 6), np.float32),
                ("stamps", (s, 4), np.int32),
                ("ticks", (w,), np.int32),
            )
        }
        loan = view._lease.loan
        for slot, world in enumerate(plan._world_keys):
            self.validate_source(plan, view)
            source = loan.position(world)
            plan._launch(
                "source/deformable:" + str(world),
                gather_deformable_points,
                s,
                slot,
                int(source.shape[0]),
                plan._buffers["shape_meta"],
                plan._buffers["node_indices"],
                plan._buffers["local"],
                plan._buffers["local_bounds"],
                source,
                loan.tick(world),
                outputs["points"],
                outputs["point_bounds"],
                outputs["stamps"],
                outputs["ticks"],
            )
        raw = {name: plan._download(array, "source/" + name) for name, array in outputs.items()}
        ticks = raw.pop("ticks")
        self.logical_ticks(plan, view, ticks)
        for name, shape, dtype in (
            ("points", (n, 3), np.float32),
            ("point_bounds", (n, 6), np.float32),
            ("stamps", (s, 4), np.int32),
        ):
            value = raw[name]
            if value.shape != shape or value.dtype != dtype or not np.isfinite(value).all():
                raise RuntimeError("D2 gathered " + name + " is malformed")
        meta = plan._constants["shape_meta"]
        if (
            np.any(raw["stamps"][:, 0] != 1)
            or not np.array_equal(raw["stamps"][:, 1], ticks[meta[:, 3]])
            or not np.array_equal(raw["stamps"][:, 2], meta[:, 3])
            or not np.array_equal(raw["stamps"][:, 3], meta[:, 1])
        ):
            raise RuntimeError("D2 source stamps differ")
        points, bounds = raw["points"], raw["point_bounds"]
        if np.any(bounds[:, :3] > points) or np.any(points > bounds[:, 3:]):
            raise RuntimeError("D2 bounds omit a gathered point")
        actual = {
            world: plan._download(loan.position(world), "source/actual_bank:" + str(world))
            for world in plan._source_base["dynamic_worlds"]
        }
        plan._counts["actual_bank_host_readback_bytes"] = sum(bank.nbytes for bank in actual.values())
        deltas = np.empty((n, 3), np.float64)
        for i, shape in enumerate(plan._model.shapes):
            begin, end = map(int, plan._offsets[i : i + 2])
            if plan._shape_rows[i] >= 0:
                bank = actual[shape.key.world]
                if (
                    bank.dtype != np.float32
                    or bank.shape != loan.position(shape.key.world).shape
                    or not np.isfinite(bank).all()
                ):
                    raise RuntimeError("D2 accepted source readback is malformed")
                expected = bank[plan._constants["node_indices"][begin:end]]
                if points[begin:end].tobytes() != expected.tobytes():
                    raise RuntimeError("D2 gathered point differs from its accepted native point")
                if (
                    bounds[begin:end, :3].tobytes() != expected.tobytes()
                    or bounds[begin:end, 3:].tobytes() != expected.tobytes()
                ):
                    raise RuntimeError("D2 native point interval differs")
            else:
                expected = plan._source_base["authored_constants"]["points"][i]
                if points[begin:end].tobytes() != plan._constants["local"][begin:end].tobytes():
                    raise RuntimeError("D2 static source differs from its registered upload")
                if bounds[begin:end].tobytes() != plan._constants["local_bounds"][begin:end].tobytes():
                    raise RuntimeError("D2 static interval differs from its registered upload")
            deltas[begin:end] = points[begin:end].astype(np.float64) - expected.astype(np.float64)
        raw.update(material_deltas=deltas, actual_bank_positions=actual)
        plan._raw_prefix["source/material_reconstruction_delta"] = owned(deltas)
        self.validate_source(plan, view)
        return outputs, raw, ticks

    def source_record(self, plan, raw, ticks, view):
        self.validate_source(plan, view)
        points, axes = {}, {}
        for i, shape in enumerate(plan._model.shapes):
            begin, end = map(int, plan._offsets[i : i + 2])
            points[shape.key] = raw["points"][begin:end]
            if shape.kind in ("sphere", "capsule"):
                axes[shape.key] = raw["points"][[begin, end - 1]]
        return freeze(
            dict(
                plan._source_base,
                state_epoch=view.accepted.state_epoch,
                active_bank=view.accepted.active,
                gathered_points=points,
                gathered_axes=axes,
                actual_bank_positions=raw["actual_bank_positions"],
                stamps=dict(
                    rows=raw["stamps"],
                    point_bounds=raw["point_bounds"],
                    accepted_ticks=ticks,
                    material_reconstruction_delta=raw["material_deltas"],
                ),
                **{
                    name: plan._counts[name]
                    for name in (
                        "native_owned_limit",
                        "borrowed_device_bytes",
                        "actual_bank_host_readback_bytes",
                        "material_delta_host_bytes",
                    )
                },
            )
        )

    def require_current(self, plan, result):
        if (
            result.source_kind != self.source_kind
            or result.world_keys != self.batch.world_keys
            or result.source["producer_binding_sha256"] != self.batch.binding_sha256
            or result.source["state_epoch"] != self.batch.revision
            or result.source["active_bank"] != self.batch._active
        ):
            raise ValueError("D2 publication belongs to another producer state")
