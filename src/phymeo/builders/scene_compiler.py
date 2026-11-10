"""Owned assembly declarations compiled to the existing HOST or resident TYX path.

This module is an assembly layer, not another dynamics model or solver. Existing
RobotAsset/FloatingRobotAsset and affine assets retain their physical coordinates,
source collider identities, passive terms and static geometry. Each world has one
independent articulation and an explicitly declared finite static environment.
HOST and TYX use the same public RobotCommand/FloatingRobotCommand boundary.

Capabilities are explicit: self collision, CCD, moving environments and multiple
articulations within one world are outside this first compiler. TYX resident
mechanics are FP32 with their original FP64 anchors; finite contact narrow phase
remains HOST. No native MuJoCo model, contact law or device fallback is implied.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from itertools import combinations, product
from numbers import Integral
from threading import RLock
from types import MappingProxyType

import numpy as np

from ..constraints.collision_shapes import (
    MAX_SHAPES,
    MAX_VERTICES,
    CollisionKey,
    CollisionModel,
    CollisionMotion,
    CollisionShape,
)
from ..constraints.robot_contact import RobotContactMaterial
from ..core.robot import RobotState
from ..core.scene import fingerprint
from ..dynamics.floating_mimic import FloatingMimicRobotAsset
from ..dynamics.floating_robot import FloatingRobotState
from ..dynamics.joint_mimic import MimicRobotAsset
from ..dynamics.rotation import quaternion_matrix
from ..geometry._convex_hull import HullShape
from ..runtime._floating_robot_participant import FloatingRobotCommand
from ..runtime.robot_scene import RobotCommand, _asset_snapshot
from ..training.resident_stepper import PhysicsStepFailure
from ._robot_asset import RobotAsset
from .floating_urdf import FloatingRobotAsset

_ASSETS = (RobotAsset, FloatingRobotAsset, MimicRobotAsset, FloatingMimicRobotAsset)


def _integer(value, label):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Integral)
        or not 0 <= value < 2**31
    ):
        raise ValueError(f"{label} must be a nonnegative i32 integer")
    return int(value)


def _array(value, label, shape):
    raw = np.asarray(value)
    if raw.dtype.kind not in "fiu" or raw.shape != shape or not np.isfinite(raw).all():
        raise ValueError(f"{label} must be a finite real array with shape {shape}")
    array = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"{label} is not representable as finite FP64")
    return np.frombuffer(array.tobytes(order="C"), dtype=np.float64).reshape(shape)


def _physical(asset):
    return asset.tree if isinstance(asset, (MimicRobotAsset, FloatingMimicRobotAsset)) else asset


def _floating(asset):
    return isinstance(asset, (FloatingRobotAsset, FloatingMimicRobotAsset))


def _gravity(asset):
    model = _physical(asset).model
    return model.root.gravity if _floating(asset) else model.gravity


@dataclass(frozen=True, eq=False)
class FiniteStaticShape:
    """Complete original shape and fixed world-space material vertices.

    None material explicitly excludes contact with this world's articulation.
    Body IDs must be distinct within a world. Static-static pairs are excluded.
    """

    shape: CollisionShape
    points_m: np.ndarray
    material: RobotContactMaterial | None

    def __post_init__(self):
        if not isinstance(self.shape, CollisionShape) or not self.shape.rigid:
            raise TypeError("static geometry requires a rigid CollisionShape")
        if self.shape.kind not in ("sphere", "capsule", "hull"):
            raise ValueError("static geometry must be a finite sphere, capsule or hull")
        if self.material is not None and not isinstance(self.material, RobotContactMaterial):
            raise TypeError("static material must be RobotContactMaterial or explicit None")
        points = _array(self.points_m, "static material vertices", (len(self.shape.vertex_ids), 3))
        if self.shape.kind == "hull":
            if self.shape.radius != 0:
                raise ValueError("rounded static hull is unsupported")
            HullShape(self.shape, CollisionMotion(points))
        elif self.shape.radius <= 0:
            raise ValueError("static sphere/capsule requires positive radius")
        elif self.shape.kind == "capsule" and np.array_equal(points[0], points[1]):
            raise ValueError("static capsule requires distinct endpoints")
        object.__setattr__(self, "shape", _asset_snapshot(self.shape))
        object.__setattr__(self, "points_m", points)
        object.__setattr__(self, "material", _asset_snapshot(self.material))

    @classmethod
    def box(
        cls,
        world,
        body,
        half_extents,
        *,
        position=(0, 0, 0),
        orientation_xyzw=(0, 0, 0, 1),
        material=None,
    ):
        half = _array(half_extents, "box half extents", (3,))
        if np.any(half <= 0):
            raise ValueError("box half extents must be positive")
        position = _array(position, "box position", (3,))
        quat = _array(orientation_xyzw, "box orientation", (4,))
        if abs(float(np.linalg.norm(quat)) - 1) > 1e-10:
            raise ValueError("box orientation must be a unit XYZW quaternion")
        points = np.array(list(product(*[(-x, x) for x in half])))
        points = position + points @ quaternion_matrix(quat[None])[0].T
        shape = CollisionShape(CollisionKey(world, body, 0), "hull", tuple(range(8)))
        return cls(shape, points, material)

    @classmethod
    def sphere(cls, world, body, center, radius, *, material=None):
        shape = CollisionShape(CollisionKey(world, body, 0), "sphere", (0,), radius=radius)
        return cls(shape, _array(center, "sphere center", (3,))[None], material)

    @classmethod
    def capsule(cls, world, body, endpoints, radius, *, material=None):
        shape = CollisionShape(CollisionKey(world, body, 0), "capsule", (0, 1), radius=radius)
        return cls(shape, endpoints, material)


@dataclass(frozen=True, eq=False)
class ArticulationWorld:
    world: int
    asset: object
    initial_state: RobotState | FloatingRobotState
    kp: np.ndarray = field(kw_only=True)
    kd: np.ndarray = field(kw_only=True)
    effort_limit: np.ndarray = field(kw_only=True)
    static_shapes: tuple[FiniteStaticShape, ...] = field(default=(), kw_only=True)
    robot_body: int = field(default=1, kw_only=True)
    initial_control: RobotCommand | FloatingRobotCommand | None = field(default=None, kw_only=True)
    self_collision: bool = field(default=False, kw_only=True)

    def __post_init__(self):
        world, body = _integer(self.world, "world"), _integer(self.robot_body, "robot body")
        if not isinstance(self.asset, _ASSETS):
            raise TypeError(
                "world asset must be an existing fixed/floating ordinary or affine asset"
            )
        asset = _asset_snapshot(self.asset)
        n = len(asset.joint_names)
        expected = FloatingRobotState if _floating(asset) else RobotState
        if not isinstance(self.initial_state, expected):
            raise TypeError("initial state coordinate kind differs from the asset")
        state = _asset_snapshot(expected(self.initial_state.q, self.initial_state.v))
        if state.q.shape != (n + 7 if _floating(asset) else n,) or state.v.shape != (
            n + 6 if _floating(asset) else n,
        ):
            raise ValueError("initial state must use the asset's independent coordinates")
        if isinstance(asset, (MimicRobotAsset, FloatingMimicRobotAsset)):
            asset.full_state(state)
        joints = state.q[7:] if _floating(asset) else state.q
        rates = state.v[6:] if _floating(asset) else state.v
        if np.any(joints < np.asarray(asset.lower)) or np.any(joints > np.asarray(asset.upper)):
            raise ValueError("initial joint position outside asset metadata limits")
        speed = np.array([np.inf if v is None else v for v in asset.velocity_limit])
        if np.any(abs(rates) > speed):
            raise ValueError("initial joint velocity outside asset metadata limits")
        for name in ("kp", "kd", "effort_limit"):
            values = _array(getattr(self, name), name, (n,))
            if np.any(values < 0):
                raise ValueError(f"{name} must be nonnegative")
            object.__setattr__(self, name, values)
        rating = np.array([np.inf if v is None else v for v in asset.effort_limit])
        if np.any(self.effort_limit > rating):
            raise ValueError("declared effort cap exceeds original asset motor rating")
        static = tuple(self.static_shapes)
        if any(not isinstance(row, FiniteStaticShape) for row in static):
            raise TypeError("static_shapes must contain FiniteStaticShape declarations")
        if any(row.shape.key.world != world or row.shape.key.body == body for row in static):
            raise ValueError(
                "static world must agree and body IDs must differ from the articulation"
            )
        if len({row.shape.key.body for row in static}) != len(static):
            raise ValueError("one original finite collider per static body is required")
        if type(self.self_collision) is not bool:
            raise TypeError("self_collision must be boolean")
        object.__setattr__(self, "world", world)
        object.__setattr__(self, "robot_body", body)
        object.__setattr__(self, "asset", asset)
        object.__setattr__(self, "initial_state", state)
        object.__setattr__(self, "static_shapes", static)
        initial = self.initial_control
        if initial is None:
            joint = RobotCommand(joints) if n else None
            initial = FloatingRobotCommand(joint) if _floating(asset) else joint
        object.__setattr__(self, "initial_control", _validated_control(self, initial))


@dataclass(frozen=True)
class SceneDeclaration:
    """Frozen assembly of existing models; this is not a replacement SceneModel."""

    worlds: tuple[ArticulationWorld, ...]
    identity: str = field(init=False)

    def __post_init__(self):
        worlds = tuple(self.worlds)
        if not worlds or any(not isinstance(world, ArticulationWorld) for world in worlds):
            raise TypeError("scene needs nonempty ArticulationWorld declarations")
        if len({world.world for world in worlds}) != len(worlds):
            raise ValueError("duplicate independent world ID")
        worlds = tuple(sorted(worlds, key=lambda world: world.world))
        object.__setattr__(self, "worlds", worlds)
        object.__setattr__(self, "identity", fingerprint(("compiled-scene-assembly-v1", worlds)))

    def assert_unchanged(self):
        if fingerprint(("compiled-scene-assembly-v1", self.worlds)) != self.identity:
            raise ValueError("scene declaration changed; create a new declaration")


@dataclass(frozen=True)
class CompilationIssue:
    world: int | None
    feature: str
    reason: str


@dataclass(frozen=True)
class CompileReport:
    backend: str
    precision: str
    features: tuple[str, ...]
    issues: tuple[CompilationIssue, ...]
    unavailable: tuple[str, ...] = (
        "self_collision",
        "ccd",
        "moving_environment",
        "multi_articulation_world",
        "per_command_effort_cap",
        "joint_stop_law",
    )
    contact_law: str = "phymeo-compliant-velocity-friction"
    narrow_phase: str = "host"

    @property
    def supported(self):
        return not self.issues


class SceneCompilationError(ValueError):
    def __init__(self, report):
        super().__init__(
            "scene compilation unsupported: "
            + "; ".join(f"world={row.world} {row.feature}: {row.reason}" for row in report.issues)
        )
        self.report = report


@dataclass(frozen=True, eq=False)
class WorldSnapshot:
    world: int
    q: np.ndarray
    v: np.ndarray
    full_q: np.ndarray
    full_v: np.ndarray
    time_s: float
    tick: int
    raw: object

    def __post_init__(self):
        for name in ("q", "v", "full_q", "full_v"):
            value = np.asarray(getattr(self, name))
            object.__setattr__(self, name, _array(value, name, value.shape))
        object.__setattr__(self, "raw", _asset_snapshot(self.raw))


def _validated_control(world, control):
    n = len(world.asset.joint_names)
    floating = _floating(world.asset)
    if floating:
        if not isinstance(control, FloatingRobotCommand):
            raise TypeError("floating world needs a FloatingRobotCommand on every backend")
        joint = control.joints
        if control.external is not None:
            full_n = len(_physical(world.asset).joint_names)
            for name, shape in (
                ("effort", (full_n,)),
                ("force", (full_n + 1, 3)),
                ("torque", (full_n + 1, 3)),
            ):
                _array(getattr(control.external, name), "external " + name, shape)
    else:
        joint = control
    if n == 0 and floating:
        if joint is not None:
            raise ValueError("root-only articulation has no joint drive")
    else:
        if not isinstance(joint, RobotCommand):
            raise TypeError("fixed world or floating child drive needs RobotCommand")
        target = _array(joint.target_q, "joint target", (n,))
        if np.any(target < np.asarray(world.asset.lower)) or np.any(
            target > np.asarray(world.asset.upper)
        ):
            raise ValueError("command target outside asset metadata limits")
        if joint.target_velocity is not None:
            velocity = _array(joint.target_velocity, "target velocity", (n,))
            speed = np.array([np.inf if v is None else v for v in world.asset.velocity_limit])
            if np.any(abs(velocity) > speed):
                raise ValueError("command target velocity outside asset metadata limits")
        if joint.feedforward is not None:
            _array(joint.feedforward, "feedforward", (n,))
        if joint.effort_limit is not None:
            cap = _array(joint.effort_limit, "command effort cap", (n,))
            if not np.array_equal(cap, world.effort_limit):
                raise ValueError("dynamic effort cap differs from the frozen scene declaration")
    return _asset_snapshot(control)


def _world_spec(world):
    """Reuse existing static lowering; no runtime creation or physical advance."""
    from ..backends._resident_affine_tree import AffineRobotTree
    from ..backends._resident_floating_affine_tree import FloatingAffineRobotTree
    from ..backends._resident_floating_tree import FloatingRobotTree
    from ..backends._resident_tree_world import FixedRobotTree
    from ..solvers.driven_robot import DrivenRobotProgram

    common = dict(kp=world.kp, kd=world.kd, effort_limit=world.effort_limit)
    if isinstance(world.asset, FloatingMimicRobotAsset):
        return FloatingAffineRobotTree(world.asset, world.initial_state, **common)
    if isinstance(world.asset, MimicRobotAsset):
        return AffineRobotTree(world.asset, world.initial_state, **common)
    if _floating(world.asset):
        return FloatingRobotTree(world.asset.model, world.initial_state, **common)
    program = DrivenRobotProgram(world.asset.model, **common)
    return FixedRobotTree(program, program.initial_state(world.initial_state))


def _source_identity(world):
    """Same deterministic body allocator as the existing HOST scene compiler."""
    physical = _physical(world.asset)
    occupied = {world.robot_body, *(row.shape.key.body for row in world.static_shapes)}
    bodies, following = {}, 0
    for body in sorted({shape.key.body for shape in physical.collision_model.shapes}):
        while following in occupied:
            following += 1
        bodies[body] = following
        occupied.add(following)
    return MappingProxyType(
        {
            row.shape.key: CollisionKey(
                world.world, bodies[row.shape.key.body], row.shape.key.shape
            )
            for row in physical.collisions
        }
    )


def _tyx_geometry(declaration):
    from ..backends._resident_affine_binding import ResidentAffineTreeBinding
    from ..backends._resident_collision_plan import ResidentStaticBinding
    from ..backends._resident_floating_affine_binding import ResidentFloatingAffineTreeBinding
    from ..backends._resident_floating_binding import ResidentFloatingTreeBinding
    from ..backends._resident_tree_binding import ResidentTreeBinding

    shapes, bindings, laws, identities = [], [], {}, {}
    for world in declaration.worlds:
        asset, physical = world.asset, _physical(world.asset)
        identities[world.world] = _source_identity(world)
        robot_shapes = {}
        fixed_poses = None
        for collider in physical.collisions:
            key = identities[world.world][collider.shape.key]
            shape = replace(collider.shape, key=key)
            shapes.append(shape)
            if _floating(asset):
                kind = (
                    ResidentFloatingAffineTreeBinding
                    if isinstance(asset, FloatingMimicRobotAsset)
                    else ResidentFloatingTreeBinding
                )
                binding = kind.from_asset(asset, collider.shape.key, key=key, world=world.world)
            elif collider.link_index < 0:
                if fixed_poses is None:
                    fixed_poses = asset.collision_motion(world.initial_state.q)
                binding = ResidentStaticBinding(key, fixed_poses[collider.shape.key])
            else:
                kind = (
                    ResidentAffineTreeBinding
                    if isinstance(asset, MimicRobotAsset)
                    else ResidentTreeBinding
                )
                binding = kind.from_asset(asset, collider.shape.key, key=key, world=world.world)
                robot_shapes[key] = shape
            if _floating(asset):
                robot_shapes[key] = shape
            bindings.append(binding)
        for row in world.static_shapes:
            shapes.append(row.shape)
            bindings.append(ResidentStaticBinding(row.shape.key, CollisionMotion(row.points_m)))
            if row.material is not None:
                for key, shape in robot_shapes.items():
                    if shape.group & row.shape.mask and row.shape.group & shape.mask:
                        laws[tuple(sorted((key, row.shape.key)))] = row.material
    # Every omitted pair is an explicit filter, including self/static/cross-world.
    exclusions = tuple(
        tuple(sorted((first.key, second.key)))
        for first, second in combinations(shapes, 2)
        if tuple(sorted((first.key, second.key))) not in laws
    )
    return CollisionModel(tuple(shapes), exclusions), tuple(bindings), laws, identities


def scene_capabilities(declaration, backend="host"):
    """Report supported assembly before creating any runtime or device owner."""
    if not isinstance(declaration, SceneDeclaration):
        raise TypeError("expected a SceneDeclaration")
    if backend not in ("host", "interpret", "cuda"):
        raise ValueError("backend must be host, interpret or cuda; no implicit fallback")
    declaration.assert_unchanged()
    issues, features = (
        [],
        {"independent_worlds", "owned_declaration", "capped_pd", "explicit_static_pairs"},
    )
    total_shapes = total_vertices = 0
    for world in declaration.worlds:
        physical = _physical(world.asset)
        features.add("floating_root" if _floating(world.asset) else "fixed_root")
        if any(row.material is not None for row in world.static_shapes):
            features.add("finite_contact")
        if isinstance(world.asset, (MimicRobotAsset, FloatingMimicRobotAsset)):
            features.add("affine_coordinates")
        if world.self_collision:
            issues.append(
                CompilationIssue(
                    world.world,
                    "self_collision",
                    "not available in the common HOST/TYX assembly domain",
                )
            )
        source = tuple(physical.collision_model.shapes)
        count = len(source) + len(world.static_shapes)
        vertices = sum(len(shape.vertex_ids) for shape in source) + sum(
            len(row.shape.vertex_ids) for row in world.static_shapes
        )
        total_shapes += count
        total_vertices += vertices
        if count > MAX_SHAPES or vertices > MAX_VERTICES:
            issues.append(
                CompilationIssue(
                    world.world, "geometry_capacity", "world exceeds original shape/vertex bounds"
                )
            )
        table = {shape.key: shape for shape in source}
        if (
            len({row.shape.key for row in physical.collisions}) != len(physical.collisions)
            or (set(table) != {row.shape.key for row in physical.collisions})
            or any(table.get(row.shape.key) != row.shape for row in physical.collisions)
        ):
            issues.append(
                CompilationIssue(
                    world.world,
                    "asset_geometry",
                    "collider declarations must exactly cover the source shape table",
                )
            )
        if any(
            shape.kind not in ("sphere", "capsule", "hull")
            or (shape.kind == "hull" and shape.radius != 0)
            for shape in source
        ):
            issues.append(
                CompilationIssue(
                    world.world,
                    "asset_geometry",
                    "common domain admits sphere/capsule/unrounded hull only",
                )
            )
        if backend == "host":
            if any(
                row.material is not None
                and not (shape.group & row.shape.mask and row.shape.group & shape.mask)
                for shape in source
                for row in world.static_shapes
            ):
                issues.append(
                    CompilationIssue(
                        world.world,
                        "collision_filter",
                        "HOST body-pair robot contacts cannot discard an individual masked collider",
                    )
                )
            if any(row.shape.kind not in ("sphere", "hull") for row in world.static_shapes):
                issues.append(
                    CompilationIssue(
                        world.world,
                        "static_geometry",
                        "existing HOST robot scene does not support static capsules",
                    )
                )
        else:
            if (
                not _floating(world.asset)
                and any(row.link_index < 0 for row in physical.collisions)
                and any(row.material is not None for row in world.static_shapes)
            ):
                issues.append(
                    CompilationIssue(
                        world.world,
                        "fixed_root_contact",
                        "resident static-static root/environment reaction differs from HOST; no contact is silently discarded",
                    )
                )
            try:
                from ..backends._world_step_layout import bind_world

                bind_world(world.world, _world_spec(world))
            except (TypeError, ValueError, OverflowError) as error:
                issues.append(CompilationIssue(world.world, "tyx_lowering", str(error)))
    if backend != "host":
        if total_shapes > MAX_SHAPES or total_vertices > MAX_VERTICES:
            issues.append(
                CompilationIssue(
                    None,
                    "geometry_capacity",
                    "one resident plan exceeds original aggregate shape/vertex bounds",
                )
            )
        if not issues:
            try:
                _tyx_geometry(declaration)
            except (TypeError, ValueError, OverflowError) as error:
                issues.append(CompilationIssue(None, "tyx_geometry", str(error)))
    return CompileReport(
        backend,
        "float64" if backend == "host" else "float32-dynamics-with-original-fp64-anchors",
        tuple(sorted(features)),
        tuple(issues),
    )


class CompiledScene:
    """Owned independent runtimes with one common control and observation boundary.

    snapshot() retains backend-native detached records in .raw. HOST contact
    receipts are original scene readings; TYX receipts retain entry-time loads
    and the actual accepted device state. They are not relabelled as one law
    implementation or one precision. A partial/unknown advance requires full
    reset; successful worlds are never described as physically rolled back.
    """

    def __init__(self, declaration, backend, report, *, cache_root=None):
        if (
            not isinstance(report, CompileReport)
            or not report.supported
            or report.backend != backend
        ):
            raise ValueError("a supported matching compilation report is required")
        declaration.assert_unchanged()
        self.declaration, self.backend, self.report = declaration, backend, report
        self.identity = declaration.identity
        self.world_keys = tuple(world.world for world in declaration.worlds)
        self.assets = MappingProxyType({world.world: world.asset for world in declaration.worlds})
        self._worlds = {world.world: world for world in declaration.worlds}
        self._lock = RLock()
        self._closed = self._poisoned = False
        self._owners, self._plan = [], None
        self._specs, self._shape_keys, self._models = {}, {}, {}
        self._times = dict.fromkeys(self.world_keys, 0.0)
        self._receipts = {}
        self._requested_dt = self._effective_dt = None
        self._held = {world.world: world.initial_control for world in declaration.worlds}
        try:
            if backend == "host":
                from ..training.scene_backend import SceneBatchBackend

                self._raw = SceneBatchBackend(self.world_keys, self._host_factory)
                self._owners.append(self._raw)
            else:
                from ..backends.tyx_world_batch import TyxWorldBatch

                self._specs = {world.world: _world_spec(world) for world in declaration.worlds}
                model, bindings, materials, identities = _tyx_geometry(declaration)
                self._shape_keys.update(identities)
                self._models = {world: model for world in self.world_keys}
                self._raw = TyxWorldBatch(
                    tuple(self._specs.items()), backend, cache_root=cache_root
                )
                self._owners.append(self._raw)
                if materials:
                    self._plan = self._raw.prepare_contacts(
                        model, bindings, materials, candidate_capacity=4096, site_capacity=128
                    )
                    self._owners.append(self._plan)
            self.set_controls(self._held)
        except Exception as error:
            try:
                self.close()
            except Exception as cleanup_error:
                error.add_note("partial scene compilation cleanup failed: " + repr(cleanup_error))
            raise

    def _host_factory(self, key):
        from ..runtime.contact_scene import ContactSceneBuilder

        world = self._worlds[key]
        builder = ContactSceneBuilder(gravity=_gravity(world.asset))

        def loader(path, *, gravity):
            if not np.array_equal(np.asarray(gravity), _gravity(world.asset)):
                raise ValueError("compiled scene gravity differs from the original model")
            return world.asset

        add = builder.add_floating_urdf if _floating(world.asset) else builder.add_urdf
        add(
            world.robot_body,
            "frozen-scene-asset",
            robot_state=world.initial_state,
            kp=world.kp,
            kd=world.kd,
            effort_limit=world.effort_limit,
            loader=loader,
            world=key,
        )
        for row in world.static_shapes:
            # Reuse the existing typed rigid installation rather than recreate
            # primitive IDs through add_box/add_sphere. All authored vertex IDs,
            # filter bits, radius and revision remain in the actual runtime.
            sphere = row.shape.kind == "sphere"
            position = row.points_m[0] if sphere else np.zeros(3)
            points = np.zeros((1, 3)) if sphere else row.points_m
            builder._add_rigid(
                row.shape, points, None, None, position, (0, 0, 0, 1), (0, 0, 0), (0, 0, 0)
            )
            builder.pair(world.robot_body, row.shape.key.body, row.material, world=key)
        for first, second in combinations(world.static_shapes, 2):
            builder.pair(first.shape.key.body, second.shape.key.body, None, world=key)
        scene = builder.build(backend="host")
        try:
            source = tuple(row.shape.key for row in _physical(world.asset).collisions)
            compiled = tuple(scene.body_keys[key, world.robot_body])
            if len(source) != len(compiled):
                raise ValueError("HOST compiler did not preserve every original collider identity")
            actual = dict(zip(source, compiled, strict=True))
            if actual != dict(_source_identity(world)):
                raise ValueError("HOST source collider mapping differs from declared allocation")
            self._shape_keys[key] = MappingProxyType(actual)
            self._models[key] = scene.geometry().model
        except Exception:
            scene.close()
            raise
        return scene

    def _check_open(self, *, ready=False):
        if self._closed:
            raise RuntimeError("compiled scene is closed")
        if ready and self._poisoned:
            raise RuntimeError("reset all worlds after partial or uncertain physical publication")

    def _selected(self, keys):
        selected = (
            self.world_keys if keys is None else tuple(_integer(key, "world") for key in keys)
        )
        if len(set(selected)) != len(selected) or not set(selected) <= set(self.world_keys):
            raise ValueError("selected world keys must be unique members of this compiled scene")
        return selected

    @property
    def needs_reset(self):
        with self._lock:
            return self._poisoned

    @property
    def shape_keys(self):
        with self._lock:
            self._check_open()
            return MappingProxyType(dict(self._shape_keys))

    @property
    def collision_models(self):
        with self._lock:
            self._check_open()
            return MappingProxyType(dict(self._models))

    @property
    def contact_receipts(self):
        with self._lock:
            self._check_open()
            return MappingProxyType(dict(self._receipts))

    def snapshot(self, keys=None):
        with self._lock:
            self._check_open()
            selected = self._selected(keys)
            raw = self._raw.snapshot(selected)
            result = {}
            for key in selected:
                value = raw[key]
                if self.backend == "host":
                    world = self._worlds[key]
                    row = value.robots[key, world.robot_body]
                    q, v, full_q, full_v = row.q, row.v, row.full_q, row.full_v
                    time, tick = value.time_s, value.step_number
                else:
                    q, v = value.position, value.velocity
                    full_q = getattr(value, "full_position", q)
                    full_v = getattr(value, "full_velocity", v)
                    time, tick = self._times[key], value.tick
                result[key] = WorldSnapshot(
                    key,
                    np.asarray(q),
                    np.asarray(v),
                    np.asarray(full_q),
                    np.asarray(full_v),
                    float(time),
                    int(tick),
                    value,
                )
            return MappingProxyType(result)

    def _device_controls(self, controls):
        result = {}
        for key in self.world_keys:
            world, spec, command = self._worlds[key], self._specs[key], controls[key]
            joint = command.joints if _floating(world.asset) else command
            n = len(world.asset.joint_names)
            target = np.zeros(n) if joint is None else joint.target_q
            rate = (
                np.zeros(n)
                if joint is None or joint.target_velocity is None
                else joint.target_velocity
            )
            effort = (
                np.zeros(n) if joint is None or joint.feedforward is None else joint.feedforward
            )
            if _floating(world.asset):
                drive = np.column_stack((target, rate, effort))
                result[key] = spec.inputs(drive, external=command.external)
            elif isinstance(world.asset, MimicRobotAsset):
                result[key] = spec.inputs(target, rate, feedforward=effort)
            else:
                result[key] = spec.program.inputs(target, rate, feedforward=effort)
        return result

    def set_controls(self, controls):
        with self._lock:
            self._check_open(ready=True)
            if not isinstance(controls, Mapping) or set(controls) != set(self.world_keys):
                raise ValueError("controls must exactly cover every compiled world")
            for key in controls:
                _integer(key, "control world")
            owned = {
                key: _validated_control(self._worlds[key], controls[key]) for key in self.world_keys
            }
            if self.backend == "host":
                from ..training.scene_backend import SceneControl

                converted = {
                    key: SceneControl({self._worlds[key].robot_body: owned[key]})
                    for key in self.world_keys
                }
            else:
                converted = self._device_controls(owned)
            # Complete conversion/validation precedes any held-input replacement.
            self._raw.set_controls(converted)
            self._held = owned

    def _acceptance(self, mask):
        if not isinstance(mask, Mapping) or set(mask) != set(self.world_keys):
            raise ValueError("physical acceptance must cover exactly the compiled worlds")
        if any(not isinstance(mask[key], (bool, np.bool_)) for key in self.world_keys):
            raise TypeError("physical acceptance values must be boolean")
        return MappingProxyType({key: bool(mask[key]) for key in self.world_keys})

    def _record_publications(self, accepted, effective_h):
        selected = tuple(key for key in self.world_keys if accepted[key])
        for key in selected:
            self._times[key] += effective_h
        if self.backend == "host":
            self._receipts.update(self._raw.snapshot(selected))
        elif self._plan is not None:
            actual = self._plan.last_accepted
            self._receipts.update({key: actual[key] for key in selected if key in actual})

    def step(self, h):
        with self._lock:
            self._check_open(ready=True)
            if isinstance(h, (bool, np.bool_)) or np.ndim(h) != 0:
                raise ValueError("physical dt must be a positive finite scalar")
            requested = float(h)
            with np.errstate(over="ignore", under="ignore"):
                effective = requested if self.backend == "host" else float(np.float32(requested))
            if not np.isfinite((requested, effective)).all() or min(requested, effective) <= 0:
                raise ValueError("physical dt must be positive and finite in the selected backend")
            if any(
                not np.isfinite(time + effective) or time + effective <= time
                for time in self._times.values()
            ):
                raise ValueError("time must advance by a representable positive interval")
            self._requested_dt, self._effective_dt = requested, effective
            try:
                returned = (
                    self._raw.step_contacts(self._plan, effective)
                    if self._plan is not None
                    else self._raw.step(effective)
                )
            except PhysicsStepFailure as error:
                if not error.acceptance_known:
                    self._poisoned = True
                    raise
                try:
                    accepted = self._acceptance(error.accepted_worlds)
                except Exception as invalid:
                    self._poisoned = True
                    raise PhysicsStepFailure(
                        "backend failed with an invalid publication mask; reset all worlds",
                        accepted_worlds={},
                        acceptance_known=False,
                    ) from invalid
                self._poisoned = any(accepted.values())
                try:
                    self._record_publications(accepted, effective)
                except Exception as read_error:
                    self._poisoned = True
                    error.add_note("publication readback also failed: " + repr(read_error))
                raise
            except Exception as error:
                self._poisoned = True
                raise PhysicsStepFailure(
                    "physical publication is uncertain; reset all worlds",
                    accepted_worlds={},
                    acceptance_known=False,
                ) from error
            try:
                accepted = self._acceptance(returned)
            except Exception as error:
                self._poisoned = True
                raise PhysicsStepFailure(
                    "backend returned an invalid publication mask; reset all worlds",
                    accepted_worlds={},
                    acceptance_known=False,
                ) from error
            self._poisoned = any(accepted.values()) and not all(accepted.values())
            try:
                self._record_publications(accepted, effective)
            except Exception as error:
                self._poisoned = True
                raise PhysicsStepFailure(
                    "physics advanced but publication readback failed; reset all worlds",
                    accepted_worlds=accepted,
                ) from error
            if self._poisoned:
                raise PhysicsStepFailure(
                    "some worlds advanced; reset all worlds before another physical step",
                    accepted_worlds=accepted,
                )
            return accepted

    def reset(self, keys=None):
        with self._lock:
            self._check_open()
            selected = self._selected(keys)
            if self._poisoned and set(selected) != set(self.world_keys):
                raise RuntimeError("reset all worlds after partial or uncertain publication")
            if not selected:
                return
            held = {**self._held, **{key: self._worlds[key].initial_control for key in selected}}
            try:
                self._raw.reset(selected)
                if self.backend == "host":
                    from ..training.scene_backend import SceneControl

                    controls = {
                        key: SceneControl({self._worlds[key].robot_body: held[key]})
                        for key in self.world_keys
                    }
                else:
                    controls = self._device_controls(held)
                self._raw.set_controls(controls)
                published = self._raw.snapshot(selected)
                if any(
                    (published[key].step_number if self.backend == "host" else published[key].tick)
                    != 0
                    for key in selected
                ):
                    raise RuntimeError("reset did not publish the original step-zero worlds")
            except Exception:
                # A failed reset may already have installed new worlds. Retain
                # old bookkeeping and require complete successful recovery.
                self._poisoned = True
                raise
            self._held = held
            for key in selected:
                self._times[key] = 0.0
                self._receipts.pop(key, None)
            self._poisoned = False

    def stats(self):
        with self._lock:
            self._check_open()
            return MappingProxyType(
                dict(
                    backend=self.backend,
                    declaration_identity=self.identity,
                    world_count=len(self.world_keys),
                    needs_reset=self._poisoned,
                    last_requested_dt_s=self._requested_dt,
                    last_effective_dt_s=self._effective_dt,
                    batch=None if self.backend == "host" else self._raw.stats(),
                    contacts=None if self._plan is None else self._plan.stats(),
                )
            )

    def close(self):
        with self._lock:
            self._closed = True
            failed, errors = [], []
            for owner in reversed(self._owners):
                try:
                    owner.close()
                except Exception as error:
                    failed.append(owner)
                    errors.append(error)
            self._owners = list(reversed(failed))
            if errors:
                raise RuntimeError(
                    "compiled scene cleanup failed; close may be retried"
                ) from errors[0]

    def __enter__(self):
        with self._lock:
            self._check_open()
            return self

    def __exit__(self, kind, error, traceback):
        try:
            self.close()
        except Exception as cleanup_error:
            if error is None:
                raise
            error.add_note("compiled scene close failed: " + repr(cleanup_error))
        return False


def compile_scene(declaration, backend="host", *, cache_root=None):
    """Compile the same frozen assembly without changing solver or falling back."""
    report = scene_capabilities(declaration, backend)
    if not report.supported:
        raise SceneCompilationError(report)
    return CompiledScene(declaration, backend, report, cache_root=cache_root)


__all__ = [
    "ArticulationWorld",
    "CompilationIssue",
    "CompileReport",
    "CompiledScene",
    "FiniteStaticShape",
    "SceneCompilationError",
    "SceneDeclaration",
    "WorldSnapshot",
    "compile_scene",
    "scene_capabilities",
]
