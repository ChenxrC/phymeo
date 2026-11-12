"""Logical participant ownership over the real shared accepted boundary banks.

Registration only: no HOST FK, closest-point search, device allocations, or
physical publication. The existing query plan owns its complete device pipeline.
"""

from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np

from ..constraints.collision_shapes import CollisionModel
from ._resident_boundary_asset import admit_boundary_asset
from ._resident_boundary_source import (
    DYNAMIC,
    LAYOUT,
    ResidentBoundarySourceView,
    freeze,
    owned,
    physical_sources,
    stored_interval,
)
from ._resident_collision_plan import ResidentStaticBinding
from ._resident_participant_layout import ResidentParticipantBinding, _freeze

_FAMILIES = ("fixed", "affine", "floating", "floating_affine")


@dataclass
class _SceneLease:
    mechanical: object
    active: bool = True

    def require(self):
        if not self.active:
            raise RuntimeError("boundary source loan expired")


class ResidentSceneBoundaryAdapter:
    source_kind = "resident-scene-rigid-device-v1"

    def __init__(self, scene):
        from .resident_scene import ResidentSceneBatch

        if type(scene) is not ResidentSceneBatch:
            raise TypeError("scene boundary requires ResidentSceneBatch")
        scene._check_open()
        self._scene = scene
        self._layout = scene.layout
        self._owner = scene._owner
        self._fingerprint = scene.layout.fingerprint
        self._registration = None
        self._signature = None
        self._mechanical = None
        self._wrappers = ()
        self._shape_participants = ()

    def _registry(self, plan):
        # This method is also called INSIDE the accepted source loan: never use
        # service.locked()/batch._check_open(), which correctly reject reentry.
        scene = self._scene
        if scene._closed or scene._service._closed:
            raise RuntimeError("resident scene boundary owner is closed")
        if (
            scene._owner is not self._owner
            or scene.layout is not self._layout
            or scene.layout.fingerprint != self._fingerprint
            or plan._batch is not scene._service._batch
        ):
            raise ValueError("boundary scene owner or registry differs")
        self._layout.require_current()
        if (
            self._signature is not None
            and _freeze((plan._model, self._wrappers)) != self._signature
        ):
            raise ValueError("boundary scene material registration changed")
        if self._registration is not None and plan._registration is not self._registration:
            raise ValueError("boundary scene registration identity differs")

    def _bank(self, plan, accepted, mechanical):
        self._registry(plan)
        batch = plan._batch
        if accepted.owner is not batch._collision_owner:
            raise ValueError("boundary native bank owner differs")
        if (
            accepted.topology_generation != plan._topology_generation
            or accepted.topology_generation != batch._topology_generation
            or accepted.state_epoch != batch._state_epoch
            or accepted.active != batch._active
        ):
            raise ValueError("boundary scene source is stale or standby")
        if accepted.runtime is not getattr(batch, "_runtime", None) or accepted.mode != plan._mode:
            raise ValueError("boundary scene runtime/mode differs")
        if set(mechanical) != set(self._layout.participant_keys) or len(
            set(mechanical.values())
        ) != len(mechanical):
            raise ValueError("boundary participant mechanical membership differs")
        if set(accepted.slots) != set(mechanical.values()):
            raise ValueError("boundary native bank contains different participants")
        if self._mechanical is not None and tuple(mechanical.items()) != self._mechanical:
            raise ValueError("boundary participant storage mapping changed")
        sources = physical_sources(accepted)
        addresses = {}
        participants = {item.key: item for item in self._layout.participants}
        for key in self._layout.participant_keys:
            native = mechanical[key]
            slot = accepted.slots[native]
            binding = accepted.bindings[slot]
            entry = self._layout.entries[key]
            layout = LAYOUT.get(binding.kind)
            if layout is None or _FAMILIES[layout] != entry.family or binding.world != native:
                raise ValueError("boundary participant physical layout differs")
            source = sources[layout]
            if (
                source is None
                or source.owner is not accepted.owner
                or source.topology_generation != accepted.topology_generation
                or source.state_epoch != accepted.state_epoch
                or source.active != accepted.active
                or native not in source.rows
                or native not in source.specs
            ):
                raise ValueError("boundary participant accepted source differs")
            spec = source.specs[native]
            owning = binding.tree_spec
            physical = getattr(owning, "physical", owning)
            item = participants[key]
            original = getattr(item.compiled, "physical", item.compiled)
            if (
                physical is not spec
                or spec.topology != original.topology
                or spec.model_token != entry.model_token
                or getattr(owning, "graph_token", None) != entry.graph_token
                or not np.array_equal(spec.anchor, original.anchor)
                or int(source.rows[native]) != entry.storage_row
            ):
                raise ValueError("boundary participant physical model/graph/row differs")
            addresses[key] = dict(
                participant=key,
                logical_world=key.world,
                mechanical_world=native,
                slot=slot,
                layout=layout,
                family=entry.family,
                row=int(source.rows[native]),
                body_ids=item.body_ids,
                physical_topology=spec.topology,
                model_token=spec.model_token,
                graph_token=getattr(owning, "graph_token", None),
                native_anchor=owned(spec.anchor, np.float64),
            )
        layouts = tuple(
            tuple(addresses[key]["layout"] for key in self._layout.worlds[w].participant_keys)
            for w in self._layout.world_keys
        )
        return sources, addresses, layouts

    def compile_registration(self, plan, model, bindings):
        if self._registration is not None:
            raise ValueError("scene boundary adapter already registered")
        if type(model) is not CollisionModel:
            raise TypeError("boundary query requires CollisionModel")
        wrappers = tuple(bindings)
        if len(wrappers) != len(model.shapes):
            raise ValueError("boundary query needs exactly one binding per shape")
        table = {}
        for wrapper in wrappers:
            if type(wrapper) is ResidentParticipantBinding:
                if wrapper.participant not in self._layout.entries:
                    raise ValueError("boundary binding has unknown participant")
                binding = wrapper.binding
            elif type(wrapper) is ResidentStaticBinding:
                binding = wrapper
            else:
                raise TypeError(
                    "scene boundary requires participant wrappers or declared static bindings"
                )
            if binding.key in table:
                raise ValueError("duplicate scene boundary binding")
            table[binding.key] = wrapper
        if set(table) != {shape.key for shape in model.shapes}:
            raise ValueError("boundary material key membership differs")
        self._wrappers = tuple(table[shape.key] for shape in model.shapes)
        self._shape_participants = tuple(
            w.participant if type(w) is ResidentParticipantBinding else None for w in self._wrappers
        )
        plan._model = model
        plan._bindings = tuple(
            w.binding if type(w) is ResidentParticipantBinding else w for w in self._wrappers
        )
        plan._shape_keys = tuple(shape.key for shape in model.shapes)
        plan._shape_revisions = tuple(shape.revision for shape in model.shapes)
        plan._world_keys = self._layout.world_keys
        plan._owner = self._owner
        plan._topology_generation = plan._batch._topology_generation
        plan._registration = object()
        with self._scene._service.borrow_boundary_source(self._owner, self._fingerprint) as (
            accepted,
            mechanical,
        ):
            sources, addresses, world_layouts = self._bank(plan, accepted, mechanical)
            anchors = {
                w: addresses[self._layout.worlds[w].participant_keys[0]]["native_anchor"]
                for w in plan._world_keys
            }
            meta = []
            local = []
            bounds = []
            radii = []
            shifts = []
            exact_shifts = []
            authored = []
            errors = []
            ids = []
            rows = []
            groups = []
            layouts = []
            offsets = [0]
            definitions = []
            for shape, wrapper, binding in zip(
                model.shapes, self._wrappers, plan._bindings, strict=True
            ):
                if shape.kind not in ("triangle", "mesh", "sphere", "capsule"):
                    raise ValueError("M1 supports only triangle/mesh against sphere/capsule")
                if shape.kind in ("triangle", "mesh") and shape.radius != 0:
                    raise ValueError("boundary surface radius must be zero")
                world = shape.key.world
                if world not in anchors:
                    raise ValueError("boundary shape world has no registered participants")
                admission = None
                participant = None
                spec = None
                if type(wrapper) is ResidentStaticBinding:
                    if (world, shape.key.body) not in self._layout.static_body_ids:
                        raise ValueError("boundary shape needs a declared static owner")
                    if binding.motion.moving or binding.motion.body_orientation_xyzw is not None:
                        raise ValueError(
                            "boundary static installation requires a fixed explicit pose"
                        )
                    first = self._layout.worlds[world].participant_keys[0]
                    layout = addresses[first]["layout"]
                    slot = addresses[first]["slot"]
                    row = group = -1
                    native = None
                    points = np.asarray(binding.motion.start, np.float64)
                    coordinates = points - anchors[world]
                    shift = np.zeros(3, np.float64)
                else:
                    participant = wrapper.participant
                    address = addresses[participant]
                    layout = address["layout"]
                    if type(binding) is not DYNAMIC[layout]:
                        raise ValueError("boundary binding differs from physical layout")
                    native = address["mechanical_world"]
                    slot = address["slot"]
                    source = sources[layout]
                    spec = source.specs[native]
                    owning = accepted.bindings[slot].tree_spec
                    group = binding.group_index
                    count = spec.b if layout >= 2 else spec.n
                    if (
                        binding.world != world
                        or participant.world != world
                        or not 0 <= group < count
                    ):
                        raise ValueError("boundary participant physical world/group differs")
                    if self._layout.body_owners.get((world, shape.key.body)) != (
                        participant,
                        group,
                    ):
                        raise ValueError(
                            "boundary registered body owner differs from participant/group"
                        )
                    if layout == 3 and binding.graph_token != owning.graph_token:
                        raise ValueError("boundary floating affine graph differs")
                    if getattr(binding, "_source_shape", None) is not None:
                        admission = admit_boundary_asset(
                            shape, binding, physical_spec=spec, owning_spec=owning
                        )
                        points = np.asarray(admission.local_points_m, np.float64)
                    else:
                        if binding.topology is None or binding.topology != spec.topology:
                            raise ValueError(
                                "M1 requires direct complete physical topology installation"
                            )
                        points = np.asarray(binding.local_points_m, np.float64)
                    row = int(source.rows[native])
                    coordinates = points
                    shift = np.asarray(spec.anchor, np.float64) - anchors[world]
                if points.shape != (len(shape.vertex_ids), 3) or not np.isfinite(points).all():
                    raise ValueError("boundary point order/arity/source differs")
                if (
                    shape.kind == "sphere"
                    and len(points) != 1
                    or shape.kind == "capsule"
                    and len(points) != 2
                    or shape.kind == "triangle"
                    and len(points) != 3
                ):
                    raise ValueError("boundary material arity differs")
                stored, lo, hi = stored_interval(coordinates)
                radius, _, rhi = stored_interval(np.asarray([shape.radius]))
                translated, tlo, thi = stored_interval(shift)
                begin = offsets[-1]
                end = begin + len(points)
                offsets.append(end)
                meta.append((layout, row, group, slot, begin, end))
                local.extend(stored)
                bounds.extend(np.c_[lo, hi])
                radii.append((radius[0], rhi[0]))
                shifts.append(np.r_[translated, tlo, thi])
                exact_shifts.append(shift)
                authored.append(owned(points))
                errors.append(owned(stored.astype(np.float64) - coordinates))
                ids.append(shape.vertex_ids)
                rows.append(row)
                groups.append(group)
                layouts.append(layout)
                definition = dict(
                    key=shape.key,
                    world=world,
                    body=shape.key.body,
                    participant=participant,
                    mechanical_world=native,
                    slot=slot,
                    layout=layout,
                    row=row,
                    group=group,
                    point_range=(begin, end),
                    point_ids=shape.vertex_ids,
                    topology=None if spec is None else spec.topology,
                    graph_token=getattr(binding, "graph_token", None),
                    frame="static_anchor" if row < 0 else "body_local",
                )
                if admission is not None:
                    definition.update(admission="asset", provenance=admission.provenance)
                definitions.append(definition)
            plan._shape_rows = owned(rows, np.int32)
            plan._shape_groups = owned(groups, np.int32)
            plan._shape_layouts = owned(layouts, np.int32)
            plan._world_layouts = world_layouts
            plan._offsets = owned(offsets, np.int32)
            plan._constants = dict(
                shape_meta=owned(np.asarray(meta, np.int32).reshape(-1, 6)),
                local=owned(np.asarray(local, np.float32).reshape(-1, 3)),
                local_bounds=owned(np.asarray(bounds, np.float32).reshape(-1, 6)),
                radius_bounds=owned(np.asarray(radii, np.float32).reshape(-1, 2)),
                anchor_shifts=owned(np.asarray(shifts, np.float32).reshape(-1, 9)),
            )
            plan._anchors = freeze(anchors)
            exact = owned(np.asarray(exact_shifts, np.float64).reshape(-1, 3))
            plan._source_base = dict(
                registration=plan._registration,
                model=model,
                bindings=tuple(definitions),
                shape_keys=plan._shape_keys,
                shape_revisions=plan._shape_revisions,
                point_ids=tuple(ids),
                facet_ids=tuple(
                    tuple(range(1 if shape.kind == "triangle" else len(shape.triangles)))
                    if shape.kind in ("triangle", "mesh")
                    else ()
                    for shape in model.shapes
                ),
                facet_vertices=tuple(
                    (shape.vertex_ids,)
                    if shape.kind == "triangle"
                    else shape.triangles
                    if shape.kind == "mesh"
                    else ()
                    for shape in model.shapes
                ),
                world_layouts=world_layouts,
                shape_rows=plan._shape_rows,
                shape_groups=plan._shape_groups,
                static_flags=owned(np.asarray(rows) < 0),
                world_anchors=anchors,
                authored_constants=dict(
                    points=tuple(authored),
                    radii=owned([shape.radius for shape in model.shapes], np.float64),
                ),
                uploaded_constants=plan._constants,
                conversion_deltas=dict(
                    points=tuple(errors),
                    radii=owned(
                        plan._constants["radius_bounds"][:, 0].astype(np.float64)
                        - np.asarray([shape.radius for shape in model.shapes])
                    ),
                    anchor_shifts=owned(
                        plan._constants["anchor_shifts"][:, :3].astype(np.float64) - exact
                    ),
                ),
                scene_owner=self._owner,
                scene_registry_fingerprint=self._fingerprint,
                participant_keys=self._layout.participant_keys,
                participant_addresses=addresses,
                participant_wrappers=self._wrappers,
                native_owner=accepted.owner,
                anchor_shifts_exact=exact,
            )
            self._mechanical = tuple(mechanical.items())
            self._signature = _freeze((model, self._wrappers))
            self._registration = plan._registration

    @contextmanager
    def borrow_source(self, plan):
        self._registry(plan)
        with self._scene._service.borrow_boundary_source(self._owner, self._fingerprint) as (
            accepted,
            mechanical,
        ):
            sources, _, layouts = self._bank(plan, accepted, mechanical)
            lease = _SceneLease(mechanical)
            try:
                yield ResidentBoundarySourceView(
                    accepted,
                    plan._model,
                    plan._bindings,
                    self._owner,
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

    def validate_source(self, plan, view):
        view._lease.require()
        sources, addresses, layouts = self._bank(plan, view.accepted, view._lease.mechanical)
        if (
            view.model is not plan._model
            or view.bindings is not plan._bindings
            or view.owner is not self._owner
            or view.registration is not plan._registration
            or view.world_keys != plan._world_keys
            or view.shape_keys != plan._shape_keys
            or view.shape_revisions != plan._shape_revisions
            or view.world_layouts != layouts
        ):
            raise ValueError("boundary scene registration/source view differs")
        if any(a is not b for a, b in zip(view.sources, sources, strict=True)):
            raise ValueError("boundary source is not this accepted scene loan")
        for a, b in (
            (view.shape_rows, plan._shape_rows),
            (view.shape_groups, plan._shape_groups),
            (view.shape_layouts, plan._shape_layouts),
        ):
            if not np.array_equal(a, b):
                raise ValueError("boundary scene material row map differs")
        for index, key in enumerate(self._shape_participants):
            if key is None:
                continue
            address = addresses[key]
            if (
                int(plan._shape_rows[index]) != address["row"]
                or int(plan._shape_layouts[index]) != address["layout"]
                or int(plan._constants["shape_meta"][index, 3]) != address["slot"]
            ):
                raise ValueError("boundary participant current row/slot differs")
        return view

    def logical_ticks(self, plan, view, accepted_ticks):
        self.validate_source(plan, view)
        if (
            accepted_ticks.shape != (len(view.accepted.bindings),)
            or accepted_ticks.dtype != np.int32
        ):
            raise RuntimeError("invalid accepted scene ticks")
        result = []
        for world in plan._world_keys:
            values = {
                int(accepted_ticks[view.accepted.slots[view._lease.mechanical[key]]])
                for key in self._layout.worlds[world].participant_keys
            }
            if len(values) != 1:
                raise RuntimeError("participant ticks differ inside a logical world")
            tick = values.pop()
            if tick < 0:
                raise RuntimeError("invalid accepted scene tick")
            result.append(tick)
        return result

    def require_current(self, plan, result):
        self._registry(plan)
        source = result.source
        if (
            source["scene_owner"] is not self._owner
            or source["scene_registry_fingerprint"] != self._fingerprint
            or source["participant_keys"] != self._layout.participant_keys
            or source["native_owner"] is not plan._batch._collision_owner
            or _freeze(source["participant_wrappers"]) != _freeze(self._wrappers)
        ):
            raise ValueError("boundary publication has different scene provenance")
        return result
