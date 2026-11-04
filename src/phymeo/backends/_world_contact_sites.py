"""Original cloth/link material coordinates and energy accounting for grouped world steps."""

from dataclasses import dataclass
from numbers import Integral

import numpy as np

from ..core.capabilities import SolverFailure
from ..core.robot import RobotInputs, RobotState, validate_robot_step
from ..core.scene import SceneInputs, SceneState, fingerprint
from ..core.state import ParticleState, StepInputs
from ..geometry._capsule_triangle_sweep import capsule_material_sites
from ..solvers.compliant_cloth_contact import CompliantClothContactProgram, _elastic
from ._world_step_layout import as_fp32


def vector(array):
    return tuple(map(float, array))


def matrix(array):
    return tuple(map(vector, array))


@dataclass(frozen=True)
class ClothLinkWorld:
    """An original cloth/link program and its per-world initial scene state."""

    program: CompliantClothContactProgram
    initial: SceneState


@dataclass(frozen=True)
class WorldContactPoint:
    axis_parameter: float
    face_weights: tuple
    link_point_m: tuple
    cloth_point_m: tuple
    law_row: tuple
    normal_speed_m_s: float
    normal_force_n: float
    force_on_cloth_n: tuple


@dataclass(frozen=True)
class WorldContactSample:
    face: str
    link: str
    gap_m: float
    compression_m: float
    law_width_m: float
    area_m2: float
    stiffness_pa_per_m: float
    normal: tuple
    sites: tuple
    normal_force_n: float
    node_forces_n: tuple
    link_force_n: tuple
    joint_effort_n: float
    power_w: float


@dataclass(frozen=True)
class WorldContactAudit:
    sample: WorldContactSample
    kinetic_change_j: float
    external_work_j: float
    contact_work_j: float
    elastic_work_j: float
    layer_energy_before_j: float
    layer_energy_after_j: float
    spring_energy_before_j: float
    spring_energy_after_j: float
    momentum_change_ns: tuple
    external_impulse_ns: tuple
    support_impulse_ns: tuple


@dataclass(frozen=True)
class WorldContactState:
    world: int
    topology: str
    tick: int
    position: tuple
    velocity: tuple
    q: float
    qd: float
    last_audit: WorldContactAudit | None


@dataclass(frozen=True)
class WorldContactResult:
    accepted: bool
    required_sites: int
    reason: str | None
    audit: WorldContactAudit | None


@dataclass(frozen=True)
class FaceGeometry:
    gap: float
    normal: tuple
    sites: tuple


class WorldMaterial:
    def __init__(self, key, spec):
        if not isinstance(spec, ClothLinkWorld) or not isinstance(
            spec.program, CompliantClothContactProgram
        ):
            raise TypeError("expected ClothLinkWorld and CompliantClothContactProgram")
        p = spec.program
        p.scene.assert_unchanged()
        layer = p.cloth_links
        if (
            layer is None
            or len(layer.links) != 1
            or len(p.surfaces) != 1
            or len(p.robots) != 1
            or p.cloth.particle_count != 3
            or p.patches
            or p.cloth_self is not None
            or p.robot_contact.planes
            or p.robot_contact.limits
            or p.material.bending is not None
            or p.damping != 0
        ):
            raise ValueError(
                "expected one three-node face, one capsule and undamped distance elasticity"
            )
        link = layer.links[0]
        model = layer.models[link.domain]
        if model.joint_types != ("prismatic",) or link.link_index != 0:
            raise ValueError("the mobile link must have one fixed-base prismatic joint")
        self.key, self.p, self.layer = key, p, layer
        self.cloth, self.robot = p.cloth_domain, link.domain
        self.signature = fingerprint(p.problem_signature())
        self.mass = np.append(1 / p.cloth.inverse_mass, model.mass)
        # The scalar prismatic velocity Jacobian is a constant unit vector in world space.
        kin = p.robots[self.robot].geometry(RobotState(np.zeros(1), np.zeros(1)), dynamics=False)
        self.axis = np.array(kin[3][0, :, 0], copy=True)
        self.gravity = np.vstack((np.tile(p.cloth.gravity, (3, 1)), model.gravity))
        self.area = float(layer.areas[0])
        state = spec.initial
        if (
            not isinstance(state, SceneState)
            or set(state.domains) != {self.cloth, self.robot}
            or state.history
        ):
            raise ValueError("initial state does not match the material world")
        cloth, robot = state.domains[self.cloth], state.domains[self.robot]
        if not isinstance(cloth, ParticleState) or cloth.position.shape != (3, 3):
            raise ValueError("expected three material node states")
        validate_robot_step(model, robot, RobotInputs.zeros(1))
        self.initial = self.point(cloth.position, cloth.velocity, robot.q[0], robot.v[0], 0, None)

    def point(self, x, v, q, u, tick, audit):
        for values, name, size in (
            (x, "node positions", (3, 3)),
            (v, "node velocities", (3, 3)),
            (q, "joint q", ()),
            (u, "joint qd", ()),
        ):
            as_fp32(values, name, size)
        return WorldContactState(
            self.key, self.signature, tick, matrix(x), matrix(v), float(q), float(u), audit
        )

    def validate(self, state):
        if (
            not isinstance(state, WorldContactState)
            or state.world != self.key
            or state.topology != self.signature
        ):
            raise ValueError("checkpoint world or topology mismatch")
        if (
            isinstance(state.tick, (bool, np.bool_))
            or not isinstance(state.tick, Integral)
            or state.tick < 0
        ):
            raise ValueError("invalid checkpoint tick")
        if state.last_audit is not None and not isinstance(state.last_audit, WorldContactAudit):
            raise ValueError("invalid contact audit in checkpoint")
        return self.point(
            state.position, state.velocity, state.q, state.qd, int(state.tick), state.last_audit
        )

    def scene(self, x, v, q, qd):
        return SceneState(
            {
                self.cloth: ParticleState(np.asarray(x, dtype=float), np.asarray(v, dtype=float)),
                self.robot: RobotState(np.array([q], dtype=float), np.array([qd], dtype=float)),
            }
        )

    def loads(self, state, inputs):
        if not isinstance(inputs, SceneInputs) or set(inputs.domains) != {self.cloth, self.robot}:
            raise ValueError("external loads must match both material world domains")
        nodes, robot = inputs.domains[self.cloth], inputs.domains[self.robot]
        if (
            not isinstance(nodes, StepInputs)
            or nodes.force.shape != (3, 3)
            or not np.isfinite(nodes.force).all()
        ):
            raise ValueError("expected a finite three-node external force")
        validate_robot_step(self.layer.models[self.robot], state.domains[self.robot], robot)
        physical = self.mass[:, None] * self.gravity
        physical[:3] += nodes.force
        physical[3] += robot.force[0] + self.axis * robot.effort[0]
        # External torque has zero generalized power for a pure prismatic coordinate.
        return physical

    def geometry(self, state):
        layer = self.layer
        link, cap = layer.links[0], layer.capsules[0]
        data = self.p._link_geometry(state)
        ends = layer.endpoints(link, data)
        triangle = state.domains[self.cloth].position[layer.faces[0]]
        witness = layer._witness(cap, 0, ends, triangle)
        gap = float(witness.gap_m - layer.rho[0])
        if witness.normal is None:
            raise SolverFailure("capsule axis meets the cloth face")
        sites = (
            capsule_material_sites(cap, witness, capsule_first=True)
            if gap < layer.law.width_m
            else ()
        )
        return FaceGeometry(gap, vector(witness.normal), tuple(sites))

    def site_rows(self, geom):
        packed = np.zeros((2, 13), dtype=float)
        law = self.layer.law
        for index, site in enumerate(geom.sites):
            row = packed[index]
            row[:3] = geom.normal
            for node, weight in site.second_weights:
                row[3 + node] = weight
            row[6:] = (
                law.width_m,
                geom.gap,
                law.stiffness_pa_per_m * self.area,
                1 / len(geom.sites),
                law.damping_s_per_m,
                law.friction,
                law.slip_speed_m_s,
            )
        return packed

    def sample(self, geom, rows, forces, mapped):
        sites = []
        for i, site in enumerate(geom.sites):
            sites.append(
                WorldContactPoint(
                    float(dict(site.first_weights)[1]),
                    tuple(site.second_weights),
                    tuple(site.first_point),
                    tuple(site.second_point),
                    vector(rows[i]),
                    float(forces[i, 0]),
                    float(forces[i, 1]),
                    vector(forces[i, 2:]),
                )
            )
        law = self.layer.law
        n = len(sites)
        total = np.sum(forces[:n, 2:], axis=0, dtype=float)
        power = float(np.sum(rows[:n, 3:6] * forces[:n, 2:], dtype=float))
        return WorldContactSample(
            self.layer.face_names[0],
            self.layer.links[0].key,
            geom.gap,
            max(0.0, law.width_m - geom.gap),
            law.width_m,
            self.area,
            law.stiffness_pa_per_m,
            geom.normal,
            tuple(sites),
            float(np.sum(forces[:n, 1], dtype=float)),
            matrix(mapped[:3]),
            vector(-total),
            float(mapped[3, 0]),
            power,
        )

    def elasticity(self, state):
        sheet = state.domains[self.cloth]
        return _elastic(self.p.material, sheet.position, sheet.velocity, 0.0)

    def spring_energy(self, x):
        d = self.p.material.distance
        if d is None:
            return 0.0
        first, last = d.indices.T
        extension = np.linalg.norm(x[last] - x[first], axis=1) - d.rest_length
        return float(np.sum(0.5 * extension**2 / d.compliance))

    def audit(self, old, end, initial, final, sample, external, spring, h):
        x0, x1 = np.asarray(old.position), end.domains[self.cloth].position
        v0, v1 = np.asarray(old.velocity), end.domains[self.cloth].velocity
        arm = end.domains[self.robot]
        delta_x = np.vstack((x1 - x0, self.axis * (arm.q[0] - old.q)))
        vel0 = np.vstack((v0, old.qd * self.axis))
        vel1 = np.vstack((v1, arm.v[0] * self.axis))
        dkin = float(np.sum(self.mass[:, None] * (vel1 - vel0) * (vel1 + vel0)) / 2)
        layer_k = self.area * self.layer.law.stiffness_pa_per_m

        def potential(geom):
            return 0.5 * layer_k * max(0.0, self.layer.law.width_m - geom.gap) ** 2

        load = external[3] + sample.link_force_n
        support = h * ((self.axis @ load) * self.axis - load)
        return WorldContactAudit(
            sample,
            dkin,
            float(np.sum(external * delta_x)),
            float(
                np.sum(sample.node_forces_n * delta_x[:3])
                + sample.joint_effort_n * (arm.q[0] - old.q)
            ),
            float(np.sum(spring * delta_x[:3])),
            potential(initial),
            potential(final),
            self.spring_energy(x0),
            self.spring_energy(x1),
            vector(np.sum(self.mass[:, None] * (vel1 - vel0), axis=0)),
            vector(h * np.sum(external, axis=0)),
            vector(support),
        )
