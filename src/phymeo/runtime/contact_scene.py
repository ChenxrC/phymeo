"""Declare finite rigid tools and connected particle sheets for the native contact-load step."""

from collections.abc import Mapping
from contextlib import ExitStack
from dataclasses import dataclass, replace
from itertools import combinations, product
from threading import RLock
from types import MappingProxyType

import numpy as np

from ..constraints.collision_shapes import (
    MAX_CONVEX_VERTICES,
    MAX_SHAPES,
    MAX_VERTICES,
    CollisionKey,
    CollisionModel,
    CollisionMotion,
    CollisionShape,
)
from ..constraints.curved_solids import CurvedSolid
from ..constraints.distance import DistanceConstraints
from ..constraints.robot_contact import RobotContactMaterial
from ..core.coupled import CoupledInputs, CoupledState
from ..core.model import ParticleModel
from ..core.rigid import RigidBodyModel, RigidBodyState
from ..core.state import ParticleState
from ..core.validation import finite_real, validate_dt
from ..geometry._convex_hull import HullShape
from ..geometry.collision_pipeline import CollisionDetection, CollisionWorkspace
from ..solvers.collision_loads import (
    CollisionLoadSystem,
    NormalLayer,
    ParticleAttachment,
    RigidAttachment,
    SurfaceLoads,
)
from ..solvers.rigid_semi_implicit import RigidSemiImplicitSolver
from ..solvers.xpbd import XPBDSolver

_ZERO = (0.0, 0.0, 0.0)
_IDENTITY = (0.0, 0.0, 0.0, 1.0)


def _array(value, name, *, shape):
    a = np.asarray(value)
    if (
        a.dtype.kind not in "fiu"
        or a.ndim != len(shape)
        or any(n is not None and n != s for n, s in zip(shape, a.shape, strict=True))
    ):
        raise ValueError(f"{name} must be a real array with shape {shape}")
    a = np.array(a, dtype=float, copy=True, order="C")
    if not np.isfinite(a).all():
        raise ValueError(f"{name} must be finite")
    return a


def _address(world, body):
    key = CollisionKey(world, body, 0)
    return key.world, key.body


def _mass(value):
    mass = finite_real(value, "mass")
    if mass <= 0:
        raise ValueError("mass must be positive; use None for a fixed rigid body")
    return mass


@dataclass(frozen=True)
class _Body:
    shape: CollisionShape
    vertices: tuple
    inverse_mass: float
    inertia: np.ndarray
    position: np.ndarray
    orientation: np.ndarray
    velocity: np.ndarray
    omega: np.ndarray


@dataclass(frozen=True)
class _Sheet:
    position: np.ndarray
    velocity: np.ndarray
    inverse_mass: np.ndarray
    faces: tuple
    compliance: float | None


@dataclass(frozen=True)
class ContactSceneObservation:
    """Static finite geometry and material loads evaluated on the same current state."""

    geometry: CollisionDetection
    loads: SurfaceLoads


@dataclass(frozen=True, eq=False)
class ContactGeometrySnapshot:
    """Current, static world-space material poses at one accepted physical time in seconds."""

    time_s: float
    model: CollisionModel
    poses: Mapping[CollisionKey, CollisionMotion]


class ContactSceneBuilder:
    """Build isolated snapshots of explicitly calibrated, finite contact scenes.

    Body IDs are local to each integer world. All shape pairs in a world require
    an explicit material declaration or exclusion, including pairs of sheet faces.
    Gravity is shared. Body poses are COM position and body-to-world xyzw orientation.
    """

    def __init__(self, *, gravity=(0.0, 0.0, -9.81)):
        self._gravity = _array(gravity, "gravity", shape=(3,))
        self._bodies = {}
        self._pairs = {}
        self._robots = set()

    def add_urdf(
        self,
        body,
        path,
        *,
        robot_state,
        kp,
        kd,
        effort_limit=None,
        pads=(),
        loader=None,
        world=0,
    ):
        """Register a fixed-root asset for the HOST robot and multiple rigid body scene.

        This scene uses explicit RobotContactMaterial pairs. Unsupported shape,
        backend and participant combinations are rejected when build() is called.
        """
        from .robot_scene import make_robot_registration

        addr = _address(world, body)
        if addr in self._bodies:
            raise ValueError("duplicate world-local body ID")
        registration = make_robot_registration(
            path,
            robot_state=robot_state,
            kp=kp,
            kd=kd,
            effort_limit=effort_limit,
            pads=pads,
            loader=loader,
            gravity=self._gravity,
        )
        self._bodies[addr] = registration
        self._robots.add(addr)

    def add_floating_urdf(
        self,
        body,
        path,
        *,
        robot_state,
        kp,
        kd,
        effort_limit=None,
        pads=(),
        loader=None,
        world=0,
    ):
        """Register a physical free root and scalar-joint tree for HOST.

        Pass load_floating_mimic_urdf as loader for an affine mimic mechanism;
        its state, gains and motor limits use independent joint coordinates.
        Both forms retain the complete physical tree and every material owner.
        External body COM wrenches use full body order in FloatingRobotCommand,
        separately from independent joint drive effort.
        """
        from ._floating_robot_participant import make_floating_registration

        addr = _address(world, body)
        if addr in self._bodies:
            raise ValueError("duplicate world-local body ID")
        registration = make_floating_registration(
            path,
            robot_state=robot_state,
            kp=kp,
            kd=kd,
            effort_limit=effort_limit,
            pads=pads,
            loader=loader,
            gravity=self._gravity,
        )
        self._bodies[addr] = registration
        self._robots.add(addr)

    def add_box(
        self,
        body,
        half_extents,
        *,
        mass,
        position=_ZERO,
        orientation_xyzw=_IDENTITY,
        velocity=_ZERO,
        angular_velocity=_ZERO,
        inertia_diagonal=None,
        world=0,
    ):
        """Add a true finite box. A positive mass defaults to uniform-box principal moments."""
        d = _array(half_extents, "half extents", shape=(3,))
        if min(d) <= 0:
            raise ValueError("box half extents must be positive")
        if mass is not None and inertia_diagonal is None:
            dx, dy, dz = d
            inertia_diagonal = (
                _mass(mass)
                / 3
                * np.array([dy * dy + dz * dz, dx * dx + dz * dz, dx * dx + dy * dy])
            )
        verts = tuple(product(*[(-v, v) for v in d]))
        self.add_convex(
            body,
            verts,
            mass=mass,
            inertia_diagonal=inertia_diagonal,
            position=position,
            orientation_xyzw=orientation_xyzw,
            velocity=velocity,
            angular_velocity=angular_velocity,
            world=world,
        )

    def add_convex(
        self,
        body,
        local_vertices,
        *,
        mass,
        inertia_diagonal=None,
        position=_ZERO,
        orientation_xyzw=_IDENTITY,
        velocity=_ZERO,
        angular_velocity=_ZERO,
        world=0,
    ):
        """Register convex-hull points in the body's principal COM frame. Mass None fixes the body."""
        addr = _address(world, body)
        if addr in self._bodies:
            raise ValueError("duplicate world-local body ID")
        pts = _array(local_vertices, "local hull vertices", shape=(None, 3))
        if not 4 <= len(pts) <= MAX_CONVEX_VERTICES:
            raise ValueError("convex vertex count outside limits")
        shape = CollisionShape(CollisionKey(*addr, 0), "hull", tuple(range(len(pts))))
        HullShape(shape, CollisionMotion(pts))
        self._add_rigid(
            shape,
            pts,
            mass,
            inertia_diagonal,
            position,
            orientation_xyzw,
            velocity,
            angular_velocity,
        )

    def add_sphere(
        self,
        body,
        radius,
        *,
        mass,
        position=_ZERO,
        orientation_xyzw=_IDENTITY,
        velocity=_ZERO,
        angular_velocity=_ZERO,
        world=0,
    ):
        """Add a uniform solid sphere centered at its COM. A missing mass fixes it in space."""
        r = finite_real(radius, "sphere radius")
        if r <= 0:
            raise ValueError("sphere radius must be positive")
        shape = CollisionShape(CollisionKey(world, body, 0), "sphere", (0,), r)
        inertia = None if mass is None else np.full(3, (2 / 5) * _mass(mass) * r**2)
        self._add_rigid(
            shape, (_ZERO,), mass, inertia, position, orientation_xyzw, velocity, angular_velocity
        )

    def add_ellipsoid(
        self,
        body,
        semi_axes,
        *,
        mass,
        position=_ZERO,
        orientation_xyzw=_IDENTITY,
        velocity=_ZERO,
        angular_velocity=_ZERO,
        world=0,
    ):
        """Add a uniform ellipsoid. Its three semi-axes follow the body's principal frame."""
        d = _array(semi_axes, "ellipsoid semi axes", shape=(3,))
        asset = CurvedSolid("ellipsoid", tuple(d))
        shape = CollisionShape(CollisionKey(world, body, 0), "ellipsoid", (0,), curved=asset)
        inertia = None
        if mass is not None:
            x, y, z = d
            inertia = (_mass(mass) / 5) * np.array([y * y + z * z, x * x + z * z, x * x + y * y])
        self._add_rigid(
            shape, (_ZERO,), mass, inertia, position, orientation_xyzw, velocity, angular_velocity
        )

    def _add_rigid(
        self,
        shape,
        pts,
        mass,
        inertia_diagonal,
        position,
        orientation_xyzw,
        velocity,
        angular_velocity,
    ):
        addr = shape.key.world, shape.key.body
        if addr in self._bodies:
            raise ValueError("duplicate world-local body ID")
        if mass is None:
            inv = 0.0
            if inertia_diagonal is None:
                inertia_diagonal = _ZERO
        else:
            inv = 1.0 / _mass(mass)
            if inertia_diagonal is None:
                raise ValueError("a moving general convex body needs its principal inertia")
        inertia = _array(inertia_diagonal, "principal inertia", shape=(3,))
        x = _array(position, "body COM position", shape=(3,))
        q = _array(orientation_xyzw, "body orientation", shape=(4,))
        v = _array(velocity, "body COM velocity", shape=(3,))
        w = _array(angular_velocity, "body angular velocity", shape=(3,))
        RigidBodyModel(np.array([inv]), inertia[None, :], self._gravity)
        RigidBodyState(x[None, :], q[None, :], v[None, :], w[None, :])
        if inv == 0 and (v.any() or w.any()):
            raise ValueError("a fixed rigid body must have zero velocity")
        self._bodies[addr] = _Body(
            shape, RigidAttachment(0, pts).vertices, inv, inertia, x, q, v, w
        )

    def add_sheet(
        self,
        body,
        positions,
        triangles,
        *,
        node_masses,
        velocities=None,
        edge_compliance=0.0,
        world=0,
    ):
        """Register independent material nodes and finite triangles; each unique edge gets one XPBD constraint.

        Node masses are positive kg, scalar or one per node. A compliance of None disables stretch
        constraints. All triangles share the physical vertices named by their local node IDs.
        """
        addr = _address(world, body)
        if addr in self._bodies:
            raise ValueError("duplicate world-local body ID")
        x = _array(positions, "sheet positions", shape=(None, 3))
        n = len(x)
        if not 3 <= n <= MAX_VERTICES:
            raise ValueError("material node count outside limits")
        v = (
            np.zeros_like(x)
            if velocities is None
            else _array(velocities, "sheet velocities", shape=x.shape)
        )
        m = np.asarray(node_masses)
        masses = (
            np.full(n, _mass(node_masses)) if m.ndim == 0 else _array(m, "node masses", shape=(n,))
        )
        if np.any(masses <= 0):
            raise ValueError("node masses must be positive")
        f = np.asarray(triangles)
        if (
            f.ndim != 2
            or f.shape[1] != 3
            or not 0 < len(f) <= MAX_SHAPES
            or f.dtype.kind not in "iu"
            or np.any(f < 0)
            or np.any(f >= n)
        ):
            raise ValueError(
                "triangles must be a bounded array of triples of local material node IDs"
            )
        faces = tuple(tuple(map(int, row)) for row in f)
        distinct = {frozenset(row) for row in faces}
        if (
            len(distinct) != len(faces)
            or any(len(row) != 3 for row in distinct)
            or len(set(f.flat)) != n
        ):
            raise ValueError(
                "faces need distinct, referenced material nodes and unique triangle registrations"
            )
        u = x[f[:, 1]] - x[f[:, 0]]
        v_edge = x[f[:, 2]] - x[f[:, 0]]
        cross = np.cross(u, v_edge)
        if np.any(~np.isfinite(cross)) or np.any(~np.any(cross, axis=1)):
            raise ValueError("degenerate initial sheet triangle")
        compliance = (
            None if edge_compliance is None else finite_real(edge_compliance, "edge compliance")
        )
        if compliance is not None and compliance < 0:
            raise ValueError("edge compliance must be nonnegative")
        self._bodies[addr] = _Sheet(x, v, 1.0 / masses, faces, compliance)

    def pair(self, first, second, layer, *, world=0):
        """Assign an explicit material to a body pair, or exclude it with None.

        NormalLayer applies to ordinary contact scenes; RobotContactMaterial
        applies to registered robot scenes. build() rejects mixed material models.
        """
        a, b = sorted((_address(world, first), _address(world, second)))
        if a not in self._bodies or b not in self._bodies:
            raise ValueError("pair refers to an unregistered body")
        if a == b and (not isinstance(self._bodies[a], _Sheet) or len(self._bodies[a].faces) < 2):
            raise ValueError("a body needs distinct material faces for a self pair")
        if (a, b) in self._pairs:
            raise ValueError("duplicate material pair")
        if layer is not None and not isinstance(layer, (NormalLayer, RobotContactMaterial)):
            raise TypeError("expected NormalLayer, RobotContactMaterial or explicit exclusion None")
        self._pairs[a, b] = layer

    def build(self, *, backend="host", cache_root=None):
        """Compile a new independent runtime and initial state. The builder remains editable."""
        if backend not in ("host", "interpret", "cuda"):
            raise ValueError("expected host, interpret or cuda")
        if not self._bodies:
            raise ValueError("register at least one physical body")
        if self._robots:
            from ._floating_robot_participant import FloatingRobotRegistration
            from .robot_scene import build_robot_scene

            if any(isinstance(item, FloatingRobotRegistration) for item in self._bodies.values()):
                from .multi_robot_scene import build_multi_robot_scene

                return build_multi_robot_scene(
                    self._bodies,
                    self._pairs,
                    gravity=self._gravity,
                    backend=backend,
                    cache_root=cache_root,
                )

            return build_robot_scene(
                self._bodies,
                self._pairs,
                gravity=self._gravity,
                backend=backend,
                cache_root=cache_root,
            )
        if any(
            law is not None and not isinstance(law, NormalLayer) for law in self._pairs.values()
        ):
            raise TypeError("a scene without a robot requires NormalLayer materials")
        dtype = np.float64 if backend == "host" else np.float32
        shape_count = sum(
            len(b.faces) if isinstance(b, _Sheet) else 1 for b in self._bodies.values()
        )
        vertex_count = sum(
            3 * len(b.faces) if isinstance(b, _Sheet) else len(b.vertices)
            for b in self._bodies.values()
        )
        if shape_count > MAX_SHAPES or vertex_count > MAX_VERTICES:
            raise ValueError("collision model capacity exceeded")
        shapes = []
        bindings = {}
        rigid_rows = {}
        particle_rows = {}
        bodies = []
        points = []
        velocities = []
        mass = []
        edges = []
        compliances = []
        for addr, item in sorted(self._bodies.items()):
            if isinstance(item, _Body):
                rigid_rows[addr] = len(bodies)
                shape = item.shape
                bindings[shape.key] = RigidAttachment(len(bodies), item.vertices)
                shapes.append(shape)
                bodies.append(item)
                continue
            offset = len(points)
            rows = tuple(range(offset, offset + len(item.position)))
            particle_rows[addr] = rows
            points.extend(item.position)
            velocities.extend(item.velocity)
            mass.extend(item.inverse_mass)
            for i, face in enumerate(item.faces):
                shape = CollisionShape(CollisionKey(*addr, i), "triangle", face)
                bindings[shape.key] = ParticleAttachment(tuple(rows[node] for node in face))
                shapes.append(shape)
            if item.compliance is not None:
                unique = sorted(
                    {
                        tuple(sorted((rows[a], rows[b])))
                        for face in item.faces
                        for a, b in combinations(face, 2)
                    }
                )
                edges.extend(unique)
                compliances.extend([item.compliance] * len(unique))
        laws = {}
        exclusions = []
        for a, b in combinations(shapes, 2):
            if a.key.world != b.key.world:
                continue
            key = ((a.key.world, a.key.body), (b.key.world, b.key.body))
            if key not in self._pairs:
                raise ValueError(f"missing material pair or explicit exclusion for {key}")
            layer = self._pairs[key]
            pair = (a.key, b.key)
            if layer is None:
                exclusions.append(pair)
            else:
                laws[pair] = layer
        collision = CollisionModel(tuple(shapes), tuple(exclusions))

        def matrix(values, width):
            return np.asarray(values, dtype=dtype).reshape(-1, width)

        gravity = self._gravity.astype(dtype)
        particles = ParticleState(matrix(points, 3), matrix(velocities, 3))
        particle_inverse = np.asarray(mass, dtype=dtype)
        rigid_inverse = np.asarray([body.inverse_mass for body in bodies], dtype=dtype)
        dynamic = np.array([body.inverse_mass > 0 for body in bodies], dtype=bool)
        if np.any(particle_inverse == 0) or np.any(rigid_inverse[dynamic] == 0):
            raise ValueError("positive inverse mass is unrepresentable at backend precision")
        if backend != "host" and points:
            face_rows = [bindings[s.key].rows for s in shapes if s.kind == "triangle"]
            actual = particles.position[np.asarray(face_rows)].astype(np.float64)
            normals = np.cross(actual[:, 1] - actual[:, 0], actual[:, 2] - actual[:, 0])
            if np.any(~np.any(normals, axis=1)):
                raise ValueError("sheet triangle is degenerate at backend precision")
        pm = ParticleModel(particle_inverse, gravity)
        rm = RigidBodyModel(
            rigid_inverse,
            matrix([b.inertia for b in bodies], 3),
            gravity,
        )
        rigid = RigidBodyState(
            matrix([b.position for b in bodies], 3),
            matrix([b.orientation for b in bodies], 4),
            matrix([b.velocity for b in bodies], 3),
            matrix([b.omega for b in bodies], 3),
        )
        constraints = None
        if edges:
            ids = np.asarray(edges, dtype=np.int64)
            rest = np.linalg.norm(
                particles.position[ids[:, 0]] - particles.position[ids[:, 1]], axis=1
            )
            constraints = DistanceConstraints(ids, rest, np.asarray(compliances, dtype=dtype))
        return ContactScene(
            collision,
            bindings,
            laws,
            rigid_rows,
            particle_rows,
            (pm, rm),
            CoupledState(particles, rigid),
            constraints,
            backend,
            cache_root,
        )


class ContactScene:
    """Own a physical state and resources for a compiled declaration. Each step is one native substep."""

    def __init__(
        self,
        collision,
        bindings,
        laws,
        rigid_rows,
        particle_rows,
        models,
        state,
        constraints,
        backend,
        cache_root,
    ):
        self._lock = RLock()
        self._closed = False
        self._models = models
        self._state = state
        from .contact_history import ContactHistory

        self._time_s = 0.0
        self._contact_history = ContactHistory()
        self._backend = backend
        self._edge_count = 0 if constraints is None else len(constraints.indices)
        self.rigid_rows = MappingProxyType(rigid_rows)
        self.particle_rows = MappingProxyType(particle_rows)
        with ExitStack() as stack:
            self._work = stack.enter_context(
                CollisionWorkspace(collision, backend, cache_root=cache_root)
            )
            self._layer = None
            if backend == "host":
                ps = (
                    XPBDSolver(constraints=constraints, tolerance=5e-5)
                    if models[0].particle_count
                    else None
                )
                rs = RigidSemiImplicitSolver() if models[1].body_count else None
            else:
                from ..backends.tyx_normal_layer import TyxNormalLayer
                from ..backends.tyx_rigid import TyxRigidSemiImplicitSolver
                from ..backends.tyx_xpbd import TyxXPBDSolver

                self._layer = stack.enter_context(TyxNormalLayer(backend, cache_root=cache_root))
                ps = (
                    stack.enter_context(
                        TyxXPBDSolver(
                            constraints, mode=backend, tolerance=5e-5, cache_root=cache_root
                        )
                    )
                    if models[0].particle_count
                    else None
                )
                rs = (
                    stack.enter_context(
                        TyxRigidSemiImplicitSolver(mode=backend, cache_root=cache_root)
                    )
                    if models[1].body_count
                    else None
                )
            self._solvers = (ps, rs)
            self._system = CollisionLoadSystem(
                collision, bindings, laws, layer_operator=self._layer
            )
            self._resources = stack.pop_all()

    def _require_open(self):
        if self._closed:
            raise RuntimeError("contact scene is closed")

    @property
    def state(self):
        """Return an owned copy of the latest accepted physical state."""
        with self._lock:
            return CoupledState(self._state.particles, self._state.rigid)

    @property
    def last_frame(self):
        """An owned copy of the latest accepted substep, or None at entry."""
        with self._lock:
            frame = self._system.last_frame
            return None if frame is None else frame.owned_copy()

    @property
    def contact_history(self):
        """Immutable loaded-pair lifecycle from the last successful physical step.

        Its sample time is the force input time, distinct from the accepted end
        state time. Read-only observations never advance the lifecycle.
        """
        with self._lock:
            return self._contact_history

    @property
    def contact_events(self):
        """Unified immutable view of the last committed loaded-pair episodes."""
        with self._lock:
            from .contact_events import contact_events_from_history

            return contact_events_from_history(
                self._contact_history,
                layers=self._system.layers,
                bindings=self._system.bindings,
            )

    def inputs(self):
        """Allocate independently writable external forces and torques matching the compiled state."""
        return CoupledInputs.zeros(
            self._models[0].particle_count,
            self._models[1].body_count,
            dtype=self._state.particles.position.dtype,
        )

    def geometry(self):
        """Freeze every registered body/face at the accepted state for raycast_scene or DepthCamera."""
        with self._lock:
            self._require_open()
            poses = self._system._paths(self._state)
            return ContactGeometrySnapshot(
                self._time_s, self._system.model, MappingProxyType(poses)
            )

    def observe_scene(self, *, capacity=None):
        """Return (current contact sample, sensor geometry) under the same accepted-state lock."""
        with self._lock:
            return self.sample(capacity=capacity), self.geometry()

    def sample(self, *, capacity=None):
        """Observe loads and static material geometry without advancing either participant."""
        with self._lock:
            self._require_open()
            loads = self._system.sample(self._state, self._work, capacity=capacity)
            return ContactSceneObservation(self._work.last_result, loads)

    def enable_coupled_tangent(self, registry, **options):
        """Select explicit material-station spring/Coulomb friction before the first step."""
        with self._lock:
            self._require_open()
            from ._scene_coupled_tangent import configure

            configure(self, registry, **options)

    @property
    def tangent_history(self):
        with self._lock:
            return getattr(self, "_tangent_state", None)

    @property
    def tangent_receipt(self):
        with self._lock:
            value = getattr(self, "_tangent_receipt", None)
            return None if value is None else replace(value)

    def step(self, dt, external=None, *, capacity=None):
        """Advance both active participants and publish only after material and swept checks accept."""
        with self._lock:
            self._require_open()
            h = validate_dt(dt, self._state.particles.position.dtype)
            end_time = self._time_s + float(h)
            if not np.isfinite(end_time) or end_time <= self._time_s:
                raise ValueError("time must advance by a representable physical interval")
            inputs = self.inputs() if external is None else external
            tangent = None
            transform = None
            if getattr(self, "_tangent", None) is not None:
                from ._scene_coupled_tangent import ordinary_context, ordinary_events

                tangent = ordinary_context(self, inputs, h)

                def transform(loads):
                    return tangent.prepare(loads, ordinary_events(self, loads, h))

            with self._work.transaction():
                f = self._system.step(
                    self._state,
                    inputs,
                    self._models,
                    self._solvers,
                    h,
                    self._work,
                    capacity=capacity,
                    _load_transform=transform,
                )
                staged_tangent = None if tangent is None else tangent.finish(f.state.rigid)
                public = f.owned_copy()
                from .contact_history import advance_contact_history

                history = advance_contact_history(
                    self._contact_history,
                    f,
                    sample_time_s=self._time_s,
                    commit_time_s=end_time,
                    bindings=self._system.bindings,
                )
            tangent_state, tangent_receipt = (
                staged_tangent if staged_tangent is not None else (None, None)
            )
            publication = (f.state, end_time, history, tangent_state, tangent_receipt)
            (
                self._state,
                self._time_s,
                self._contact_history,
                self._tangent_state,
                self._tangent_receipt,
            ) = publication
            return public

    def stats(self):
        with self._lock:
            return dict(
                backend=self._backend,
                rigid_count=self._models[1].body_count,
                particle_count=self._models[0].particle_count,
                distance_edges=self._edge_count,
                geometry=self._work.stats(),
                normal_layer=None if self._layer is None else self._layer.stats(),
            )

    def close(self):
        with self._lock:
            if not self._closed:
                self._closed = True
                self._resources.close()

    def __enter__(self):
        self._require_open()
        return self

    def __exit__(self, *_):
        self.close()
