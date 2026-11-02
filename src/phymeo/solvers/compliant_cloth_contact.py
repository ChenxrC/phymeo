"""Sampled finite-patch contact and elastic cloth advanced together on the NumPy host."""

from dataclasses import dataclass
from numbers import Integral

import numpy as np

from ..constraints.bending import bending_angle_gradient
from ..constraints.cloth_material import ClothMaterial
from ..constraints.robot_contact_patch import RobotContactPatch
from ..constraints.triangle_prism_surface import TrianglePrismSurface
from ..core.capabilities import SolverFailure
from ..core.model import ParticleModel
from ..core.robot import RobotModel, RobotState
from ..core.scene import SceneModel, StageMetrics
from ..core.validation import finite_real
from ..coupling.offset_port import TriangleOffsetPointPort
from ..coupling.ports import ParticlePointPort
from ..dynamics.robot_contact import RobotContactSystem
from ..dynamics.robot_dynamics import robot_com_kinematics, robot_dynamics_terms
from ..geometry.pad_columns import check_column_segment, rounded_pad_columns
from ..runtime.program import ProgramCapabilities
from ..runtime.registry import StateRegistry
from ..runtime.robot_family import robot_family
from ..runtime.scene_stepper import SceneStepper
from ._compliant_motion import advance_state, check_robot_contact, optional_budget


@dataclass(frozen=True)
class CompliantPatch:
    """One-sided finite polygon. Set robot_domain=None for a stationary world patch.

    The signed canonical normal points into the allowed half-space. Stiffness
    is normal stress per metre of compression. A positive radius gives the
    finite pad a rounded rim and back behind its authored front. Friction is regularized below
    slip_speed_m_s. Both participants receive the force at the same point.
    """

    geometry: RobotContactPatch
    robot_domain: str | None = None
    normal_sign: int = 1
    width_m: float = 0.002
    stiffness_pa_per_m: float = 500.0
    damping_s_per_m: float = 0.0
    friction: float = 0.0
    slip_speed_m_s: float = 0.01
    radius_m: float | None = None

    def __post_init__(self):
        if self.radius_m is not None:
            radius = finite_real(self.radius_m, "pad radius")
            if radius <= 0:
                raise ValueError("finite pad radius must be positive")
            object.__setattr__(self, "radius_m", radius)
        if not isinstance(self.geometry, RobotContactPatch):
            raise TypeError("expected RobotContactPatch")
        if self.robot_domain is not None and (
            not isinstance(self.robot_domain, str) or not self.robot_domain
        ):
            raise ValueError("invalid robot domain")
        if (
            isinstance(self.normal_sign, (bool, np.bool_))
            or not isinstance(self.normal_sign, Integral)
            or self.normal_sign not in (-1, 1)
        ):
            raise ValueError("normal_sign must be -1 or 1")
        object.__setattr__(self, "normal_sign", int(self.normal_sign))
        for name in (
            "width_m",
            "stiffness_pa_per_m",
            "damping_s_per_m",
            "friction",
            "slip_speed_m_s",
        ):
            value = finite_real(getattr(self, name), name)
            if value < 0 or (
                name in ("width_m", "stiffness_pa_per_m", "slip_speed_m_s") and value == 0
            ):
                raise ValueError(f"invalid {name}")
            object.__setattr__(self, name, value)


class _RobotMechanics:
    def __init__(self, model):
        self.model = model
        self.translation = None
        if all(j == "prismatic" for j in model.joint_types):
            zero = np.zeros(model.dof_count)
            self.translation = self.geometry(RobotState(zero, zero))

    def geometry(self, state, *, dynamics=True):
        if self.translation is not None:
            pos, rot, com, jl, ja, m, b = self.translation
            shift = jl @ state.q
            return pos + shift, rot, com + shift, jl, ja, m, b
        if dynamics:
            geo, jl, ja, m, b = robot_dynamics_terms(self.model, state)
        else:
            geo, jl, ja = robot_com_kinematics(self.model, state.q)
            m, b = None, None
        return geo.position, geo.rotation, geo.com_position, jl, ja, m, b


def _elastic(material, x, v, damping):
    f = np.zeros_like(x)
    if material.distance is not None:
        rows = material.distance
        a, b = rows.indices.T
        edge = x[b] - x[a]
        length = np.linalg.norm(edge, axis=1)
        if np.any(length == 0):
            raise SolverFailure("zero elastic edge")
        grad = edge / length[:, None]
        rate = np.sum(grad * (v[b] - v[a]), axis=1)
        load = grad * ((length - rows.rest_length + damping * rate) / rows.compliance)[:, None]
        np.add.at(f, a, load)
        np.add.at(f, b, -load)
    if material.bending is not None:
        rows = material.bending
        for ids, rest, alpha in zip(rows.indices, rows.rest_angle, rows.compliance, strict=True):
            theta, grad = bending_angle_gradient(x[ids])
            diff = theta - rest
            if abs(diff) >= np.pi:
                raise SolverFailure("unsupported elastic hinge branch")
            rate = np.sum(grad * v[ids])
            np.add.at(f, ids, -grad * ((diff + damping * rate) / alpha))
    return f


class CompliantClothContactProgram:
    """Explicit common-motion cloth, finite planar patches and finite-mass robots.

    Each triangle contributes three thickness-offset quadrature points with
    reference-area weights. Patches load only their face interior. Time
    stepping samples the contact law; substeps set its temporal resolution.
    An optional hard penetration limit rejects an entire step at sampled states.
    """

    capabilities = ProgramCapabilities(
        "numpy", {"float64"}, {"particles", "robot"}, {"compliant_contact", "cloth_elasticity"}
    )
    history_fields = ()

    def __init__(
        self,
        scene,
        cloth_domain,
        material,
        surfaces,
        reference_positions,
        patches,
        *,
        material_damping_s=0.0,
        max_penetration_m=None,
        robot_contact=None,
        max_joint_violation=None,
    ):
        if not isinstance(scene, SceneModel) or not isinstance(material, ClothMaterial):
            raise TypeError("expected SceneModel and ClothMaterial")
        domains = {d.name: d for d in scene.domains}
        if (
            cloth_domain not in domains
            or domains[cloth_domain].family != "particles"
            or not isinstance(domains[cloth_domain].model, ParticleModel)
        ):
            raise ValueError("unbound cloth domain")
        self.scene, self.cloth_domain, self.cloth, self.material = (
            scene,
            cloth_domain,
            domains[cloth_domain].model,
            material,
        )
        material.validate_model(self.cloth)
        for law in (material.distance, material.bending):
            if law is not None and np.any(law.compliance <= 0):
                raise ValueError("elastic force path requires positive compliance")
        self.max_penetration_m = (
            None
            if max_penetration_m is None
            else finite_real(max_penetration_m, "max_penetration_m")
        )
        if self.max_penetration_m is not None and self.max_penetration_m <= 0:
            raise ValueError("penetration limit must be positive")
        self.damping = finite_real(material_damping_s, "material damping")
        if self.damping < 0:
            raise ValueError("negative material damping")
        self.robots = {}
        for name, dom in domains.items():
            if name == cloth_domain:
                continue
            if dom.family != "robot" or not isinstance(dom.model, RobotModel):
                raise ValueError("additional domains must be robots")
            self.robots[name] = _RobotMechanics(dom.model)
        if robot_contact is not None:
            if not isinstance(robot_contact, RobotContactSystem) or set(
                robot_contact.models
            ) != set(self.robots):
                raise ValueError("robot contact models must match the bound scene")
            if any(robot_contact.models[n] is not r.model for n, r in self.robots.items()):
                raise ValueError("robot contact uses different inertial models")
        self.robot_contact = robot_contact
        self.max_joint_violation = optional_budget(max_joint_violation, "max_joint_violation")
        self.surfaces = tuple(surfaces)
        if not self.surfaces or not all(
            isinstance(s, TrianglePrismSurface) and s.domain == cloth_domain for s in self.surfaces
        ):
            raise ValueError("expected nonempty cloth surface set")
        if len({s.key for s in self.surfaces}) != len(self.surfaces):
            raise ValueError("duplicate material face key")
        self.faces = np.array([s.node_indices for s in self.surfaces], dtype=np.int64)
        self.ids = np.repeat(self.faces, 3, axis=0)
        if self.faces.max() >= self.cloth.particle_count:
            raise ValueError("surface node outside particle model")
        self.weights = np.tile(np.eye(3), (len(self.faces), 1))
        self.rho = np.array([s.half_thickness_m for s in self.surfaces])
        ref = np.asarray(reference_positions, dtype=np.float64)
        if ref.shape != (self.cloth.particle_count, 3) or not np.isfinite(ref).all():
            raise ValueError("invalid reference positions")
        tri = ref[self.faces]
        self.area = np.repeat(
            np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) / 6, 3
        )
        if not np.isfinite(self.area).all() or np.any(self.area <= 0):
            raise ValueError("degenerate reference face")
        self.patches = tuple(patches)
        if not all(isinstance(p, CompliantPatch) for p in self.patches):
            raise TypeError("expected CompliantPatch")
        if len({p.geometry.key for p in self.patches}) != len(self.patches):
            raise ValueError("duplicate patch key")
        self.finite_pads = tuple(p for p in self.patches if p.radius_m is not None)
        self.normals = {}
        for p in self.patches:
            if p.robot_domain is not None and (
                p.robot_domain not in self.robots
                or p.geometry.link_index >= self.robots[p.robot_domain].model.dof_count
            ):
                raise ValueError("unbound robot patch")
            vertices = p.geometry.local_vertices
            edges = vertices[1:3] - vertices[0]
            scale = float(np.max(np.abs(edges)))
            if not np.isfinite(scale) or scale == 0:
                raise ValueError("patch edges are not representable")
            n = np.cross(*(edges / scale))
            length = np.hypot.reduce(n)
            if not np.isfinite(length) or length == 0:
                raise ValueError("patch normal is not representable")
            self.normals[p.geometry.key] = n / length
        for a in (self.faces, self.ids, self.weights, self.rho, self.area, *self.normals.values()):
            a.flags.writeable = False

    def problem_signature(self):
        return dict(
            semantics="compliant-finite-pad-midpoint-v1"
            if self.finite_pads
            else "compliant-patch-midpoint-v1",
            scene=self.scene.identity,
            cloth=self.cloth_domain,
            material=self.material,
            surfaces=self.surfaces,
            reference_area=self.area,
            patches=self.patches,
            material_damping=self.damping,
            max_penetration_m=self.max_penetration_m,
            robot_contact=None if self.robot_contact is None else self.robot_contact.signature(),
            max_joint_violation=self.max_joint_violation,
        )

    def stepper(self, settings=None):
        registry = StateRegistry()
        registry.register(robot_family())
        return SceneStepper(self.scene, self, settings, registry=registry)

    def create_workspace(self, scene, state, inputs, h, settings, context):
        if scene.identity != self.scene.identity:
            raise ValueError("different bound scene")
        return _Workspace(self, state, inputs, h)

    def close(self):
        pass

    def _loads(self, state, *, dynamics=True):
        cloth = state.domains[self.cloth_domain]
        base = ParticlePointPort(
            self.cloth, cloth, np.arange(self.cloth.particle_count), domain=self.cloth_domain
        )
        tri = cloth.position[self.faces]
        normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        geo = {n: r.geometry(state.domains[n], dynamics=dynamics) for n, r in self.robots.items()}
        nodal = np.zeros_like(cloth.position)
        reactions = {n: np.zeros_like(state.domains[n].v) for n in self.robots}
        details = {}
        ports = {}
        for patch in self.patches:
            p = patch.geometry
            local_normal = self.normals[p.key]
            robot = patch.robot_domain
            link = p.link_index
            if robot is None:
                origin, rotation = np.zeros(3), np.eye(3)
            else:
                pos, rot, com, jl, ja, _, _ = geo[robot]
                origin, rotation = pos[link], rot[link]
            normal = patch.normal_sign * (rotation @ local_normal)
            vertices = p.local_vertices
            pad_sites = None
            if patch.radius_m is None:
                signs = -np.sign(normals @ normal)
                key = signs.tobytes()
                if key not in ports:
                    port = TriangleOffsetPointPort(
                        base, self.ids, self.weights, np.repeat(signs * self.rho, 3)
                    )
                    ports[key] = port.positions(), port.velocities(), port.jacobians
                points, velocity, jac = ports[key]
                local = (points - origin) @ rotation
                edge = np.roll(vertices, -1, axis=0) - vertices
                sides = np.cross(edge[None, :, :], local[:, None, :] - vertices) @ local_normal
                inside = np.all(sides >= 0, axis=1)
                gap = (local - vertices[0]) @ (patch.normal_sign * local_normal)
                force_normal = normal
            else:
                lengths = np.linalg.norm(normals, axis=1)
                if np.any(lengths <= 0):
                    raise SolverFailure("degenerate material face at finite pad")
                centers = (tri.reshape(-1, 3) - origin) @ rotation
                half = np.repeat(normals * (self.rho / lengths)[:, None], 3, axis=0) @ rotation
                pad_sites = rounded_pad_columns(
                    vertices, patch.normal_sign * local_normal, patch.radius_m, centers, half
                )
                heights = pad_sites.fraction * np.repeat(self.rho, 3)
                key = heights.tobytes()
                # Separate the two cache namespaces even when all heights vanish.
                key = ("finite", key)
                if key not in ports:
                    port = TriangleOffsetPointPort(base, self.ids, self.weights, heights)
                    ports[key] = port.positions(), port.velocities(), port.jacobians
                points, velocity, jac = ports[key]
                gap = pad_sites.gap_m
                inside = np.ones(len(gap), dtype=bool)
                force_normal = pad_sites.normal @ rotation.T
            relative = velocity
            if robot is not None:
                speed = state.domains[robot].v
                arms = points - com[link]
                relative = velocity - (jl[link] @ speed + np.cross(ja[link] @ speed, arms))
            rate = (
                relative @ normal if pad_sites is None else np.sum(relative * force_normal, axis=1)
            )
            pressure = np.where(
                inside,
                self.area
                * patch.stiffness_pa_per_m
                * np.maximum(patch.width_m - gap, 0.0)
                * np.maximum(1 - patch.damping_s_per_m * rate, 0.0),
                0.0,
            )
            tangent = relative - rate[:, None] * force_normal
            ft = (
                -patch.friction
                * pressure[:, None]
                * tangent
                / np.maximum(patch.slip_speed_m_s, np.linalg.norm(tangent, axis=1))[:, None]
            )
            force = pressure[:, None] * force_normal + ft
            np.add.at(nodal, self.ids.ravel(), np.einsum("kaij,ki->kaj", jac, force).reshape(-1, 3))
            if robot is not None:
                reactions[robot] -= jl[link].T @ force.sum(axis=0) + ja[link].T @ np.cross(
                    arms, force
                ).sum(axis=0)
            details[p.key] = dict(
                points_m=points,
                force_n=force,
                normal=normal,
                normal_force_n=pressure,
                gap_m=gap,
                normal_speed_m_s=rate,
                in_patch=inside,
            )
            if pad_sites is not None:
                details[p.key].update(
                    site_normals=force_normal,
                    column_ends_local_m=pad_sites.endpoints_m,
                    middle_distance_m=gap + patch.radius_m,
                    material_fraction=pad_sites.fraction,
                )
        return nodal, reactions, details, geo

    def observe(self, state):
        """Sample endpoint contact forces; time-integrated loads are in substep diagnostics."""
        return self._loads(state, dynamics=False)[2]

    def observe_robot_contact(self, state):
        return None if self.robot_contact is None else self.robot_contact.evaluate(state.domains)

    def accelerations(self, state, inputs):
        nodal, reactions, details, geo = self._loads(state)
        body = None
        if self.robot_contact is not None:
            body = self.robot_contact._evaluate(state.domains, geo)
            for name in reactions:
                reactions[name] += body["efforts"][name]
        cloth = state.domains[self.cloth_domain]
        internal = _elastic(self.material, cloth.position, cloth.velocity, self.damping)
        accel = {
            self.cloth_domain: self.cloth.gravity
            + self.cloth.inverse_mass[:, None]
            * (inputs.domains[self.cloth_domain].force + internal + nodal)
        }
        for name, load in reactions.items():
            _, _, _, jl, ja, m, b = geo[name]
            f = inputs.domains[name]
            rhs = (
                f.effort
                + np.einsum("nki,nk->i", jl, f.force)
                + np.einsum("nki,nk->i", ja, f.torque)
                + load
                - b
            )
            try:
                accel[name] = np.linalg.solve(m, rhs)
            except np.linalg.LinAlgError as exc:
                raise SolverFailure("singular robot inertia") from exc
        if not all(np.isfinite(a).all() for a in accel.values()):
            raise SolverFailure("nonfinite coupled acceleration")
        return accel, details, internal, body


def _penetration(rows):
    return max(
        (
            float(np.max(np.maximum(-r["gap_m"][r["in_patch"]], 0.0), initial=0.0))
            for r in rows.values()
        ),
        default=0.0,
    )


class _Workspace:
    def __init__(self, program, state, inputs, h):
        self.program, self.state, self.inputs, self.h = program, state, inputs, h

    def predict(self):
        a0, initial, _, body0 = self.program.accelerations(self.state, self.inputs)
        mid = advance_state(self.state, a0, self.h / 2, midpoint=True)
        a, contact, internal, body = self.program.accelerations(mid, self.inputs)
        self.state = advance_state(self.state, a, self.h, midpoint=False)
        self.penetration = max(_penetration(initial), _penetration(contact))
        if (
            self.program.max_penetration_m is not None
            or body is not None
            or self.program.finite_pads
        ):
            _, _, endpoint, geo = self.program._loads(self.state, dynamics=False)
            self.penetration = max(self.penetration, _penetration(endpoint))
            for pad in self.program.finite_pads:
                key = pad.geometry.key
                for left, right in ((initial[key], contact[key]), (contact[key], endpoint[key])):
                    check_column_segment(
                        left["column_ends_local_m"],
                        left["middle_distance_m"],
                        right["column_ends_local_m"],
                        right["middle_distance_m"],
                    )
            if body is not None:
                end_body = self.program.robot_contact._evaluate(self.state.domains, geo)
                check_robot_contact(
                    (body0, body, end_body),
                    self.program.max_penetration_m,
                    self.program.max_joint_violation,
                )
                self.penetration = max(
                    self.penetration, *(r["penetration_m"] for r in (body0, body, end_body))
                )
            if (
                self.program.max_penetration_m is not None
                and self.penetration > self.program.max_penetration_m
            ):
                raise SolverFailure(
                    f"sampled hard penetration limit exceeded: {self.penetration:.6g} m"
                )
        self.report = dict(
            patches=contact,
            robot_contact=body,
            material_force_n=internal,
            sampled_penetration_m=self.penetration,
            force_sample="substep-midpoint",
        )

    def position_sweep(self):
        pass

    def position_metrics(self):
        return StageMetrics(penetration=self.penetration)

    def prepare_velocity(self):
        pass

    def velocity_sweep(self):
        pass

    def velocity_metrics(self):
        return StageMetrics()

    def result(self):
        return self.state

    def diagnostics(self):
        return self.report

    def close(self):
        pass
