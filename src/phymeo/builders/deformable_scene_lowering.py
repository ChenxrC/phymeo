"""Asset-level deformable-normal lowering to the existing common D3 owner.

This optional compiler creates no independent rigid runtime, integrator, contact
law or collision kernel. Default participant and original scene routes are
unchanged. Array declarations are owned; checkpoint capsules remain owner-bound.
"""
from collections.abc import Mapping
from dataclasses import dataclass, field
from threading import RLock
from types import MappingProxyType

from ..backends._deformable_boundary_source import DeformableBoundaryModel, DeformableBoundarySurface
from ..backends._resident_floating_binding import compile_floating_tree_binding
from ..backends.tyx_deformable_batch import DeformableNodeWorld, TyxDeformableBatch
from ..backends.tyx_deformable_contact import TyxDeformableContactBatch
from ..constraints.collision_shapes import CollisionShape
from ..core.scene import StepSettings
from ..geometry._triangle_distance_dispatch import BoundaryDistanceLimits
from ..runtime.robot_scene import _asset_snapshot
from ..solvers.collision_loads import NormalLayer
from .floating_urdf import FloatingRobotAsset

_SCHEMA = "phymeo-deformable-participant-v1"


def _source(deformable):
    return (
        deformable.worlds, deformable.surfaces, deformable.settings, deformable.limits,
        deformable.memory_limit_bytes, deformable.node_memory_limit_bytes,
    )


@dataclass(frozen=True, eq=False)
class ParticipantDeformable:
    """Owned D1 material and installed open surfaces; not a solid-volume mesh.

    memory_limit_bytes covers the existing D3 query/response native reservation.
    node_memory_limit_bytes retains D1's separate device plus HOST array budget.
    Neither number is a process-RSS or caller-held historical evidence bound.
    """
    worlds: tuple[DeformableNodeWorld, ...]
    surfaces: tuple[DeformableBoundarySurface, ...]
    settings: StepSettings | None = field(default=None, kw_only=True)
    limits: BoundaryDistanceLimits | None = field(default=None, kw_only=True)
    memory_limit_bytes: int = field(default=256 * 1024 * 1024, kw_only=True)
    node_memory_limit_bytes: int = field(default=256 * 1024 * 1024, kw_only=True)
    _identity: str = field(init=False, repr=False)

    def __post_init__(self):
        from .participant_scene_compiler import _digest

        worlds = tuple(self.worlds)
        surfaces = tuple(self.surfaces)
        if not 1 <= len(worlds) <= 8 or any(type(w) is not DeformableNodeWorld for w in worlds):
            raise ValueError("deformable declaration requires 1..8 DeformableNodeWorlds")
        if len({w.key for w in worlds}) != len(worlds):
            raise ValueError("duplicate deformable world identity")
        if any(type(s) is not DeformableBoundarySurface for s in surfaces):
            raise TypeError("surfaces must be DeformableBoundarySurface declarations")
        if len(surfaces) > 256 or len({s.key for s in surfaces}) != len(surfaces):
            raise ValueError("surface capacity or unique installation identity differs")
        settings = StepSettings(substeps=1) if self.settings is None else self.settings
        limits = BoundaryDistanceLimits() if self.limits is None else self.limits
        if type(settings) is not StepSettings or type(limits) is not BoundaryDistanceLimits:
            raise TypeError("deformable settings/limits require their original public types")
        for name in ("memory_limit_bytes", "node_memory_limit_bytes"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(name + " requires a positive builtin int")
        # Original static admission validates fixed-node velocity, complete stable
        # IDs/topology and resource reservation before any TYX runtime is created.
        requirements = TyxDeformableBatch.requirements(worlds, settings=settings, trace=True)
        if requirements.total_bytes > self.node_memory_limit_bytes:
            raise MemoryError("deformable node reservation exceeds node_memory_limit_bytes")
        worlds = tuple(sorted(worlds, key=lambda w: w.key))
        declared = {(w.key, s.key): s for w in worlds for s in w.surfaces}
        installed = {(s.key.world, s.surface_key) for s in surfaces}
        if len(installed) != len(surfaces) or installed != set(declared):
            raise ValueError("installed surfaces must cover each original D1 surface exactly once")
        for name, value in (
            ("worlds", worlds), ("surfaces", tuple(sorted(surfaces, key=lambda s: s.key))),
            ("settings", settings), ("limits", limits),
        ):
            object.__setattr__(self, name, _asset_snapshot(value))
        object.__setattr__(self, "_identity", _digest(_source(self)))

    def assert_unchanged(self):
        from .participant_scene_compiler import _digest

        if _digest(_source(self)) != self._identity:
            raise ValueError("deformable material, topology or source changed")


def prepare_deformable_scene(declaration, rigid_prepared):
    """Pure lowering after the existing rigid asset preflight; no owner allocation."""
    from .participant_scene_compiler import ParticipantSceneDeclaration

    if type(declaration) is not ParticipantSceneDeclaration:
        raise TypeError("deformable lowering requires ParticipantSceneDeclaration")
    declaration.assert_unchanged()
    spec = declaration.deformable
    if declaration.contact_response != "deformable_normal" or type(spec) is not ParticipantDeformable:
        raise ValueError("deformable_normal requires ParticipantDeformable")
    spec.assert_unchanged()
    if declaration.static_shapes or declaration.static_body_ids or declaration.intra_participant_pairs:
        raise ValueError("D5-A supports no static installation or intra-participant response")
    if type(declaration.site_capacity) is not int or not 0 <= declaration.site_capacity <= 128:
        raise ValueError("D5-A site_capacity requires one integer in [0,128]")
    participants = rigid_prepared["participants"]
    worlds = tuple(w.key for w in spec.worlds)
    if tuple(sorted(p.key.world for p in participants)) != worlds or any(
        p.family != "floating" or p.body_count != 1 for p in participants
    ):
        raise ValueError("D5-A requires one ordinary single-body floating participant per node world")
    if any(type(row.asset) is not FloatingRobotAsset for row in declaration.articulations):
        raise ValueError("D5-A requires ordinary non-folded FloatingRobotAsset sources")
    rigid_model = rigid_prepared["model"]
    if rigid_model is None or any(
        type(shape) is not CollisionShape or shape.kind not in ("sphere", "capsule")
        for shape in rigid_model.shapes
    ):
        raise ValueError("D5-A moving geometry requires original sphere/capsule colliders")
    if {shape.key.world for shape in rigid_model.shapes} != set(worlds):
        raise ValueError("every D5-A world must retain an original moving round collider")
    by_participant = {p.key: p for p in participants}
    shapes = {s.key: s for s in rigid_model.shapes}
    for wrapper in rigid_prepared["bindings"]:
        participant = by_participant.get(wrapper.participant)
        if participant is None or wrapper.binding.group_index != 0:
            raise ValueError("D5-A collider is not owned by its original root body")
        # This is the original source/model-token gate, not a fake direct binding.
        compile_floating_tree_binding(
            shapes[wrapper.binding.key], wrapper.binding, participant.compiled
        )
    model = DeformableBoundaryModel(
        (*spec.surfaces, *rigid_model.shapes), rigid_model.excluded
    )
    expected = {
        tuple(sorted((surface.key, shape.key)))
        for surface in spec.surfaces for shape in rigid_model.shapes
        if surface.key.world == shape.key.world
        and surface.key.body != shape.key.body
        and surface.group & shape.mask and shape.group & surface.mask
        and tuple(sorted((surface.key, shape.key))) not in model.excluded
    }
    if len(expected) > declaration.candidate_capacity:
        raise ValueError("complete deformable candidate domain exceeds candidate_capacity")
    if set(declaration.materials_by_pair) != expected:
        raise ValueError("deformable materials must completely cover eligible original pairs")
    if any(law is not None and (type(law) is not NormalLayer or law.friction != 0)
           for law in declaration.materials_by_pair.values()):
        raise ValueError("D5-A requires NormalLayer with zero friction or explicit None")
    response_bytes = (
        116 * max(1, declaration.site_capacity)
        + 12 * sum(w.model.particle_count for w in spec.worlds)
        + 28 * len(worlds) + 4 * (len(worlds) + 1)
    )
    if response_bytes >= spec.memory_limit_bytes:
        raise MemoryError("response leaves no boundary query reservation")
    source_surfaces = {}
    for surface in spec.surfaces:
        world = next(w for w in spec.worlds if w.key == surface.key.world)
        native = next(s for s in world.surfaces if s.key == surface.surface_key)
        source_surfaces[surface.key] = dict(
            world=world.key, surface_key=surface.surface_key, point_ids=world.point_ids,
            facet_ids=native.facet_ids, oriented_point_ids=native.oriented_point_ids,
        )
    result = dict(rigid_prepared)
    result.update(
        node_worlds=spec.worlds, deformable_model=model, laws=declaration.materials_by_pair,
        settings=spec.settings, limits=spec.limits,
        memory_limit_bytes=spec.memory_limit_bytes,
        node_memory_limit_bytes=spec.node_memory_limit_bytes,
        source_map=_asset_snapshot(dict(
            participants=rigid_prepared["source_map"], surfaces=source_surfaces,
        )),
    )
    return result


class CompiledDeformableParticipantScene:
    """Public compiler result delegating all physics to one common D3 owner.

    Query results describe a particular accepted source revision; last_step.query
    instead describes that response's input. Checkpoints are owner-bound, and
    reset restores declared initial controls as well as both physical states.
    """
    def __init__(self, declaration, *, mode="cuda", cache_root=None):
        from .participant_scene_compiler import ParticipantSceneDeclaration, _input, _prepare

        if type(declaration) is not ParticipantSceneDeclaration:
            raise TypeError("compile requires ParticipantSceneDeclaration")
        if declaration.contact_response != "deformable_normal":
            raise ValueError("compiled deformable scene requires deformable_normal mode")
        if mode not in ("interpret", "cuda"):
            raise ValueError("deformable scene mode must be interpret or cuda")
        prepared = _prepare(declaration)
        self._lock = RLock()
        self._closed = False
        self._declaration = declaration
        self._rows = MappingProxyType({row.key: row for row in declaration.articulations})
        self._specs = prepared["specs"]
        self._source_map = prepared["source_map"]
        self._world_keys = tuple(w.key for w in prepared["node_worlds"])
        initial = {key: _input(row, self._specs[key], row.initial_control)
                   for key, row in self._rows.items()}
        self._response = TyxDeformableContactBatch(
            prepared["node_worlds"], prepared["participants"], prepared["deformable_model"],
            prepared["bindings"], prepared["laws"], mode=mode,
            settings=prepared["settings"], limits=prepared["limits"],
            contact_capacity=declaration.site_capacity,
            memory_limit_bytes=prepared["memory_limit_bytes"],
            node_memory_limit_bytes=prepared["node_memory_limit_bytes"], cache_root=cache_root,
        )
        try:
            self._response.set_controls(initial)
            self._initial = self._response.checkpoint()
        except BaseException as primary:
            try:
                self._response.close()
            except BaseException as cleanup:
                failure = BaseExceptionGroup(
                    "deformable assembly and owner cleanup failed", [primary, cleanup]
                )
                failure.contact_owner = self._response
                failure.cleanup_state = "unresolved"
                raise failure from None
            self._closed = True
            raise

    def _check(self, expected_revision=None):
        if self._closed:
            raise RuntimeError("compiled deformable scene is closed")
        self._declaration.assert_unchanged()
        self._declaration.deformable.assert_unchanged()
        if expected_revision is not None and (
            type(expected_revision) is not int or expected_revision != self._response.revision
        ):
            raise ValueError("stale compiled deformable revision")

    @property
    def declaration(self):
        return self._declaration

    @property
    def identity(self):
        return self._declaration.identity

    @property
    def source_map(self):
        return self._source_map

    @property
    def world_keys(self):
        return self._world_keys

    @property
    def participant_keys(self):
        return tuple(sorted(self._rows))

    @property
    def revision(self):
        with self._lock:
            self._check()
            return self._response.revision

    @property
    def state_epoch(self):
        return self.revision

    @property
    def last_step(self):
        with self._lock:
            self._check()
            return self._response.last_step

    @property
    def last_attempt(self):
        with self._lock:
            self._check()
            return self._response.last_attempt

    def set_controls(self, controls, *, expected_revision=None):
        from .participant_scene_compiler import _input, _validated_control

        with self._lock:
            self._check(expected_revision)
            if not isinstance(controls, Mapping) or any(key not in self._rows for key in controls):
                raise ValueError("controls must map registered participant identities")
            owned = {key: _validated_control(self._rows[key]._world, value)
                     for key, value in controls.items()}
            native = {key: _input(self._rows[key], self._specs[key], value)
                      for key, value in owned.items()}
            return self._response.set_controls(native)

    def step(self, inputs, h, *, expected_revision):
        with self._lock:
            self._check(expected_revision)
            return self._response.step(inputs, h, expected_revision=expected_revision)

    def query(self, *, expected_revision):
        with self._lock:
            self._check(expected_revision)
            return self._response.query(expected_revision=expected_revision)

    def require_query_current(self, result, *, expected_revision):
        with self._lock:
            self._check(expected_revision)
            return self._response.require_query_current(result, expected_revision=expected_revision)

    def snapshot(self):
        with self._lock:
            self._check()
            return self._response.snapshot()

    def checkpoint(self):
        with self._lock:
            self._check()
            return self._response.checkpoint()

    def restore(self, checkpoint, *, expected_revision, worlds=None):
        with self._lock:
            self._check(expected_revision)
            return self._response.restore(checkpoint, expected_revision=expected_revision, worlds=worlds)

    def reset(self, worlds=None, *, expected_revision):
        with self._lock:
            self._check(expected_revision)
            return self._response.restore(self._initial, expected_revision=expected_revision, worlds=worlds)

    def stats(self):
        with self._lock:
            self._check()
            return MappingProxyType(dict(
                self._response.stats(), assembly_schema=_SCHEMA,
                assembly_identity=self.identity, contact_response="deformable_normal",
            ))

    def close(self):
        with self._lock:
            if not self._closed:
                self._response.close()
                self._closed = True

    def __enter__(self):
        with self._lock:
            self._check()
            return self

    def __exit__(self, *_):
        self.close()
