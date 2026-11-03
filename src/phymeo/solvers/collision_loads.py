"""Two-way contact loads on registered rigid and deformable surfaces."""

from dataclasses import dataclass
from numbers import Integral
from threading import Lock
from types import MappingProxyType

import numpy as np

from ..constraints.collision_shapes import (
    CollisionKey,
    CollisionModel,
    CollisionMotion,
    CollisionPolicy,
    RigidTurn,
)
from ..core.capabilities import SolverFailure
from ..core.coupled import CoupledInputs, CoupledState
from ..core.result import StepReport
from ..core.rigid import RigidBodyInputs
from ..core.state import ParticleState, StepInputs
from ..core.validation import finite_real, validate_dt
from ..dynamics.rotation import SEMI_IMPLICIT_WORLD_OMEGA, advance_world_orientation, quaternion_matrix
from ..geometry._capsule_triangle_sweep import capsule_material_sites
from ..geometry.collision_pipeline import CollisionDetection, CollisionWorkspace, detect
from ..geometry.collision_witness import ConvexContactPoint, ProximityWitness


def _row(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 0:
        raise ValueError("state row must be a nonnegative integer")
    return int(value)


@dataclass(frozen=True)
class RigidAttachment:
    """One COM state row and body-local collision vertices in registered ID order."""

    row: int
    vertices: tuple

    def __post_init__(self):
        object.__setattr__(self, "row", _row(self.row))
        a = np.asarray(self.vertices, dtype=float)
        if a.ndim != 2 or a.shape[1] != 3 or not a.size or not np.isfinite(a).all():
            raise ValueError("invalid body-local vertices")
        object.__setattr__(self, "vertices", tuple(tuple(p) for p in a))


@dataclass(frozen=True)
class ParticleAttachment:
    """Physical particle rows in registered material vertex order."""

    rows: tuple[int, ...]

    def __post_init__(self):
        rows = tuple(_row(i) for i in self.rows)
        if not rows or len(set(rows)) != len(rows):
            raise ValueError("shape needs distinct particle rows")
        object.__setattr__(self, "rows", rows)


@dataclass(frozen=True)
class NormalLayer:
    """Per-pair width [m], effective stiffness [N/m] and rate damping [s/m]."""

    width_m: float
    stiffness_n_per_m: float
    damping_s_per_m: float = 0.0
    friction: float = 0.0
    slip_speed_m_s: float = 0.01

    def __post_init__(self):
        for name in (
            "width_m",
            "stiffness_n_per_m",
            "damping_s_per_m",
            "friction",
            "slip_speed_m_s",
        ):
            value = finite_real(getattr(self, name), name)
            if value < 0 or (
                name in ("width_m", "stiffness_n_per_m", "slip_speed_m_s") and value == 0
            ):
                raise ValueError("invalid normal layer calibration")
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class LoadSite:
    material: ConvexContactPoint
    fraction: float
    rate_m_per_s: float
    force_n: float
    application_point_first_m: tuple
    application_point_second_m: tuple
    force_on_second_n: tuple
    power_w: float


@dataclass(frozen=True)
class PairLoad:
    first: CollisionKey
    second: CollisionKey
    normal: tuple
    gap_m: float
    force_n: float
    potential_j: float
    power_w: float
    sites: tuple[LoadSite, ...]


@dataclass(frozen=True)
class SurfaceLoads:
    inputs: CoupledInputs
    contacts: tuple[PairLoad, ...]


@dataclass(frozen=True)
class CollisionLoadStep:
    entry: CoupledState
    state: CoupledState
    external: CoupledInputs
    loads: SurfaceLoads
    path: CollisionDetection
    particle_report: StepReport | None
    rigid_report: StepReport | None


def _quadrature(points):
    """Lump a uniform patch measure at its material vertices. Integrate affine fields exactly."""
    n = len(points)
    if n <= 2:
        return (1 / n,) * n
    x = np.array([p.first_point for p in points])
    center = x.mean(axis=0)
    areas = np.linalg.norm(np.cross(x - center, np.roll(x, -1, axis=0) - center), axis=1)
    total = areas.sum()
    if not total > 0 or not np.isfinite(total):
        raise SolverFailure("nonpositive closest patch area")
    return (areas + np.roll(areas, 1)) / (3 * total) + 1 / (3 * n)


def _sites(pair, shapes):
    w = pair.witness
    if w.capsule_axis_span is not None:
        first = shapes[pair.first]
        cap_first = first.kind == "capsule"
        return capsule_material_sites(
            first if cap_first else shapes[pair.second], w, capsule_first=cap_first
        )
    if isinstance(w, ProximityWitness) and w.proximity_patch is not None:
        return w.proximity_patch.points
    if pair.plate is not None and pair.plate.end.patch is not None:
        return pair.plate.end.patch.points
    if pair.convex is not None:
        raise SolverFailure("missing finite rigid support patch")
    if pair.family == "DD":
        raise SolverFailure("missing deformable support patch")
    return (ConvexContactPoint(w.first_point, w.second_point, w.first_weights, w.second_weights),)


def _host_law(rows):
    rate = np.einsum("ij,ij->i", rows[:, :3], rows[:, 3:6])
    compression = np.maximum(0.0, rows[:, 6] - rows[:, 7])
    magnitude = rows[:, 8] * compression * rows[:, 9] * np.maximum(0.0, 1.0 - rows[:, 10] * rate)
    force = magnitude[:, None] * rows[:, :3]
    sliding = rows[:, 11] > 0
    if np.any(sliding):
        tangent = rows[sliding, 3:6] - rate[sliding, None] * rows[sliding, :3]
        speed = np.hypot.reduce(tangent, axis=1)
        factor = rows[sliding, 11] * magnitude[sliding] / np.maximum(speed, rows[sliding, 12])
        force[sliding] -= factor[:, None] * tangent
    return np.column_stack((rate, magnitude, force))


class CollisionLoadSystem:
    """Sample a common entry state and publish a step after both solvers and core CCD accept.

    Solvers and the caller-owned geometric workspace can be HOST or tyx. This
    adapter maps material loads on the host. Pass an explicit TyxNormalLayer
    to evaluate the per-site force on interpret or CUDA. Each explicitly listed pair has one
    effective stiffness, shared over its entire finite contact patch.
    """

    def __init__(self, model, bindings, layers, *, layer_operator=None):
        if not isinstance(model, CollisionModel):
            raise TypeError("expected a CollisionModel")
        self.model = model
        self._layer_operator = layer_operator
        shapes = {s.key: s for s in model.shapes}
        self._shapes = shapes
        if set(bindings) != set(shapes):
            raise ValueError("bindings must cover the collision model")
        self.bindings = MappingProxyType(dict(bindings))
        self._material = {}
        owners, identities, rigid_bodies = {}, {}, {}
        for key, b in self.bindings.items():
            s = shapes[key]
            owner = (key.world, key.body)
            if s.rigid:
                if not isinstance(b, RigidAttachment) or len(b.vertices) != len(s.vertex_ids):
                    raise ValueError("invalid rigid attachment")
                if (
                    rigid_bodies.setdefault(owner, b.row) != b.row
                    or owners.setdefault(("r", b.row), owner) != owner
                ):
                    raise ValueError("inconsistent rigid body row")
            else:
                if not isinstance(b, ParticleAttachment) or len(b.rows) != len(s.vertex_ids):
                    raise ValueError("invalid particle attachment")
                self._material[key] = dict(zip(s.vertex_ids, b.rows, strict=True))
                for node, row in self._material[key].items():
                    address = (*owner, node)
                    if (
                        identities.setdefault(address, row) != row
                        or owners.setdefault(("p", row), address) != address
                    ):
                        raise ValueError("inconsistent particle material identity")
        laws = {}
        for key, law in layers.items():
            a, b = sorted(key)
            if (
                a not in shapes
                or b not in shapes
                or not isinstance(law, NormalLayer)
                or (a, b) in laws
            ):
                raise ValueError("invalid collision layer pair")
            x, y = shapes[a], shapes[b]
            same_sheet = a != b and x.kind == y.kind == "triangle"
            if (
                a.world != b.world
                or (a.body == b.body and not same_sheet)
                or (a, b) in model.excluded
                or not (x.group & y.mask and y.group & x.mask)
            ):
                raise ValueError("layer pair is filtered by the collision model")
            kinds = tuple(sorted((x.kind, y.kind)))
            smooth_pair = "ellipsoid" in kinds and set(kinds) <= {
                "ellipsoid","sphere","triangle","particle","rounded_particle",
            }
            if (not smooth_pair and kinds not in {
                ("hull", "hull"),
                ("sphere", "triangle"),
                ("capsule", "triangle"),
                ("triangle", "triangle"),
                ("particle", "plate"),
                ("plate", "triangle"),
            }) or any(s.radius and s.kind not in ("sphere", "capsule", "plate", "rounded_particle") for s in (x, y)):
                raise ValueError("unsupported shape combination for contact layer")
            if law.friction and x.rigid == y.rigid:
                raise ValueError("sliding layer requires one rigid and one deformable surface")
            laws[a, b] = law
        self.layers = MappingProxyType(laws)
        self._margin = max((v.width_m for v in laws.values()), default=0.0)
        self._last = None
        self._lock = Lock()

    @property
    def last_frame(self):
        return self._last

    def _transaction(self, workspace):
        if not isinstance(workspace, CollisionWorkspace):
            raise ValueError("workspace is bound to a different collision model")
        return workspace.transaction()

    def _publish_frame(self, frame):
        with self._lock:
            self._last = frame

    def _paths(self, state, end=None, rigid_solver=None, dt=None):
        particles = state if isinstance(state, ParticleState) else state.particles
        end_particles = end if end is None or isinstance(end, ParticleState) else end.particles
        rotations = (
            None if isinstance(state, ParticleState) else quaternion_matrix(state.rigid.orientation)
        )
        out = {}
        for s in self.model.shapes:
            b = self.bindings[s.key]
            if isinstance(b, ParticleAttachment):
                rows = list(b.rows)
                if max(rows) >= particles.particle_count:
                    raise ValueError("particle attachment row exceeds state")
                out[s.key] = CollisionMotion(
                    particles.position[rows],
                    None if end_particles is None else end_particles.position[rows],
                )
                continue
            r = b.row
            if r >= state.rigid.body_count:
                raise ValueError("rigid attachment row exceeds state")
            local = np.array(b.vertices)
            start = local @ rotations[r].T + state.rigid.position[r]
            body_frame=None
            if s.curved is not None:
                q0=state.rigid.orientation[r].astype(float)
                body_frame=q0/np.hypot.reduce(q0)
            if end is None:
                out[s.key] = CollisionMotion(start,body_orientation_xyzw=body_frame)
                continue
            if body_frame is not None:
                if dt is None or getattr(rigid_solver,"rigid_motion_contract",None)!=SEMI_IMPLICIT_WORLD_OMEGA:
                    raise SolverFailure("attached curved solid requires a bound physical angular drift")
                omega=end.rigid.angular_velocity[r].astype(float)
                speed=float(np.hypot.reduce(omega))
                axis=(1.,0.,0.) if not speed else omega/speed
                expected=advance_world_orientation(body_frame[None,:],omega[None,:],dt)[0]
                q1=end.rigid.orientation[r]
                if min(np.linalg.norm(q1-expected),np.linalg.norm(q1+expected))>16*np.finfo(q1.dtype).eps:
                    raise SolverFailure("curved attachment orientation differs from physical angular drift")
                turn=RigidTurn(tuple(state.rigid.position[r]),tuple(end.rigid.position[r]),axis,speed*dt)
                out[s.key]=CollisionMotion(start,turn=turn,body_orientation_xyzw=body_frame)
                continue
            if s.kind == "capsule" and (
                getattr(rigid_solver, "rigid_motion_contract", None) != SEMI_IMPLICIT_WORLD_OMEGA
                or np.any(end.rigid.angular_velocity[r])
            ):
                raise SolverFailure(
                    "attached capsule requires bound continuous rigid motion for translation"
                )
            q0, q1 = state.rigid.orientation[r], end.rigid.orientation[r]
            if not (np.array_equal(q0, q1) or np.array_equal(q0, -q1)) and (
                s.kind != "sphere" or np.any(local)
            ):
                raise SolverFailure(
                    "rotating attached shape needs supported continuous rigid motion"
                )
            turn = RigidTurn(
                tuple(state.rigid.position[r]), tuple(end.rigid.position[r]), (1.0, 0.0, 0.0), 0.0
            )
            out[s.key] = CollisionMotion(start, turn=turn)
        return out

    def _query(self, state, end, work, capacity, rigid_solver=None, dt=None):
        policy = CollisionPolicy(
            margin=self._margin,
            proximity_patches=end is None,
            **({} if capacity is None else {"candidate_capacity": capacity}),
        )
        result = detect(self.model, self._paths(state, end, rigid_solver, dt), policy, work)
        if result.status != "ok":
            raise SolverFailure("collision " + result.status)
        return result

    def _velocity(self, key, point, weights, state):
        b = self.bindings[key]
        if isinstance(b, RigidAttachment):
            arm = np.asarray(point) - state.rigid.position[b.row]
            return state.rigid.velocity[b.row] + np.cross(state.rigid.angular_velocity[b.row], arm)
        rows = self._material[key]
        particles = state if isinstance(state, ParticleState) else state.particles
        return sum(
            weight * particles.velocity[rows[node]].astype(float) for node, weight in weights
        )

    def _apply(self, key, point, weights, f, state, particles, rigid, torques):
        b = self.bindings[key]
        if isinstance(b, RigidAttachment):
            rigid[b.row] += f
            torques[b.row] += np.cross(np.asarray(point) - state.rigid.position[b.row], f)
        else:
            for node, weight in weights:
                particles[self._material[key][node]] += weight * f

    def sample(self, state, workspace, *, capacity=None):
        """Evaluate internal loads without advancing state or replacing the last physical frame."""
        with self._transaction(workspace):
            return self._sample(state, workspace, capacity)

    def _require_particles(self, particles):
        if not isinstance(particles, ParticleState) or len(self._material) != len(self.bindings):
            raise ValueError("expected particle states and only particle surface bindings")

    def sample_particles(self, particles, workspace, *, capacity=None):
        """Sample surface forces directly from an external particle solver's state."""
        self._require_particles(particles)
        return self.sample(particles, workspace, capacity=capacity)

    def check_particle_segment(self, first, last, workspace, *, capacity=None):
        """Check the actual affine material segment, preserving legal shared addresses."""
        self._require_particles(first)
        self._require_particles(last)
        if (
            first.position.shape != last.position.shape
            or first.position.dtype != last.position.dtype
        ):
            raise ValueError("particle segment endpoints must have matching rows and precision")
        with self._transaction(workspace):
            return self._checked_path(first, last, workspace, capacity)

    def _checked_path(self, first, last, workspace, capacity, rigid_solver=None, dt=None):
        path = self._query(first, last, workspace, capacity, rigid_solver, dt)
        for pair in path.pairs:
            if (pair.first, pair.second) in self.layers and pair.status not in (
                "separated",
                "adjacent",
            ):
                raise SolverFailure("core path rejected: " + pair.reason)
        return path

    def _sample(self, state, workspace, capacity):
        result = self._query(state, None, workspace, capacity)
        particles = state if isinstance(state, ParticleState) else state.particles
        p = np.zeros(particles.position.shape, dtype=float)
        rigid_count = 0 if isinstance(state, ParticleState) else state.rigid.body_count
        r = np.zeros((rigid_count, 3), dtype=float)
        torque = np.zeros_like(r)
        pending = []
        rows = []
        for pair in result.pairs:
            law = self.layers.get((pair.first, pair.second))
            if law is None or pair.status == "adjacent":
                continue
            if pair.status not in ("separated", "touching", "overlap"):
                raise SolverFailure(pair.reason)
            w = pair.witness
            if w is None:
                if pair.status != "separated":
                    raise SolverFailure("overlapping material pair has no resolved load witness")
                continue
            if w.gap_m >= law.width_m:
                continue
            if w.gap_m <= 0 or w.normal is None:
                raise SolverFailure("hard core has no admissible layer state")
            points = _sites(pair, self._shapes)
            fractions = _quadrature(points)
            coordinates = []
            for site, fraction in zip(points, fractions, strict=True):
                pa, pb = site.first_point, site.second_point
                if law.friction:
                    # The traction acts at the cloth material point. The rigid
                    # pad supports a wrench at this same spatial location.
                    pa = pb = pb if isinstance(self.bindings[pair.first], RigidAttachment) else pa
                va = self._velocity(pair.first, pa, site.first_weights, state)
                vb = self._velocity(pair.second, pb, site.second_weights, state)
                coordinates.append((pa, pb, vb - va))
                rows.append(
                    (
                        *w.normal,
                        *(vb - va),
                        law.width_m,
                        w.gap_m,
                        law.stiffness_n_per_m,
                        fraction,
                        law.damping_s_per_m,
                        law.friction,
                        law.slip_speed_m_s,
                    )
                )
            pending.append((pair, law, points, fractions, coordinates))
        values = np.asarray(rows, dtype=float).reshape(-1, 13)
        response = (
            _host_law(values)
            if self._layer_operator is None
            else self._layer_operator.evaluate(values)
        )
        if (
            response.shape != (len(rows), 5)
            or not np.isfinite(response).all()
            or np.any(response[:, 1] < 0)
        ):
            raise SolverFailure("invalid normal layer response")
        contacts = []
        result_rows = iter(response.astype(float, copy=False))
        for pair, law, points, fractions, coordinates in pending:
            w = pair.witness
            sites = []
            for site, fraction, (pa, pb, velocity) in zip(
                points, fractions, coordinates, strict=True
            ):
                rate, magnitude, *directional = next(result_rows)
                f = np.asarray(directional)
                self._apply(pair.first, pa, site.first_weights, -f, state, p, r, torque)
                self._apply(pair.second, pb, site.second_weights, f, state, p, r, torque)
                sites.append(
                    LoadSite(
                        site,
                        float(fraction),
                        float(rate),
                        float(magnitude),
                        pa,
                        pb,
                        tuple(f),
                        float(f @ velocity),
                    )
                )
            contacts.append(
                PairLoad(
                    pair.first,
                    pair.second,
                    w.normal,
                    w.gap_m,
                    sum(s.force_n for s in sites),
                    law.stiffness_n_per_m * (law.width_m - w.gap_m) ** 2 / 2,
                    sum(s.power_w for s in sites),
                    tuple(sites),
                )
            )
        dtype = particles.position.dtype
        loads = CoupledInputs(
            StepInputs(p.astype(dtype)), RigidBodyInputs(r.astype(dtype), torque.astype(dtype))
        )
        return SurfaceLoads(loads, tuple(contacts))

    def step(self, state, external, models, solvers, dt, workspace, *, capacity=None):
        """Advance one substep with supplied particle-material and rigid solvers. No implicit retry."""
        with self._transaction(workspace), self._lock:
            h = validate_dt(dt, state.particles.position.dtype)
            if not isinstance(external, CoupledInputs):
                raise TypeError("expected CoupledInputs")
            pm, rm = models
            particle_solver, rigid_solver = solvers
            if (
                pm.particle_count != state.particles.particle_count
                or rm.body_count != state.rigid.body_count
                or external.particles.force.shape != state.particles.position.shape
                or external.rigid.force.shape != state.rigid.position.shape
                or external.particles.force.dtype != state.particles.position.dtype
            ):
                raise ValueError("state, model and external inputs must match")
            loads = self.sample(state, workspace, capacity=capacity)
            internal = loads.inputs
            combined = (
                StepInputs(external.particles.force + internal.particles.force),
                RigidBodyInputs(
                    external.rigid.force + internal.rigid.force,
                    external.rigid.torque + internal.rigid.torque,
                ),
            )
            output = []
            reports = []
            for model, solver, entry, forces, count in zip(
                (pm, rm),
                (particle_solver, rigid_solver),
                (state.particles, state.rigid),
                combined,
                (pm.particle_count, rm.body_count),
                strict=True,
            ):
                if count:
                    run = solver.step(model, entry, forces, h)
                    if not run.report.accepted:
                        raise SolverFailure("participant rejected: " + str(run.report.reason))
                    output.append(run.state)
                    reports.append(run.report)
                else:
                    output.append(entry)
                    reports.append(None)
            candidate = CoupledState(*output)
            path = self._checked_path(state, candidate, workspace, capacity, rigid_solver, h)
            frame = CollisionLoadStep(
                CoupledState(state.particles, state.rigid),
                candidate,
                CoupledInputs(external.particles, external.rigid),
                loads,
                path,
                *reports,
            )
            workspace._stage_publication(self._publish_frame, frame)
            return frame
