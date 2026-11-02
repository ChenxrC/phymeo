"""Runnable NumPy cloth/patch cases. All recorded forces come from the public stepper."""

from collections.abc import Mapping
from dataclasses import dataclass, fields

import numpy as np

from ..constraints.bending import BendingConstraints
from ..constraints.cloth_material import ClothMaterial
from ..constraints.distance import DistanceConstraints
from ..constraints.robot_contact_patch import RobotContactPatch
from ..constraints.triangle_prism_surface import TrianglePrismSurface
from ..core.model import ParticleModel
from ..core.robot import RobotInputs, RobotModel, RobotState
from ..core.scene import DomainBinding, SceneInputs, SceneModel, SceneState, StepSettings
from ..core.state import ParticleState, StepInputs
from ..dynamics.robot_kinematics import forward_kinematics
from ..solvers.compliant_cloth_contact import CompliantClothContactProgram, CompliantPatch


@dataclass
class Case:
    program: CompliantClothContactProgram
    initial: SceneState
    inputs: SceneInputs
    dt: float
    substeps: int


def _slider():
    return RobotModel(
        np.array([-1]),
        ("prismatic",),
        np.zeros((1, 3)),
        np.array([[0.0, 0.0, 0.0, 1.0]]),
        np.array([[0.0, 0.0, 1.0]]),
        np.array([2.0]),
        np.zeros((1, 3)),
        np.full((1, 3), 2.0),
        np.zeros(3),
    )


def build_impact(*, support=False):
    mass, rho = 0.5, 1 / 64
    ref = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]])
    x = ref.copy()
    if not support:
        x[:, 2] += 3 / 1024
    v = np.zeros_like(x)
    if not support:
        v[:, 2] = [-1 / 8, -1 / 8, -1 / 8, -7 / 64]
    cloth = ParticleModel(np.full(4, 1 / mass), np.array([0.0, 0.0, -1 / 8 if support else 0.0]))
    material = ClothMaterial(
        DistanceConstraints(np.array([[0, 1], [0, 2], [0, 3]]), np.ones(3), np.ones(3)),
        BendingConstraints(
            np.array([[0, 1, 2, 3]]), np.array([1 / 1024 if support else 0.0]), np.ones(1)
        ),
    )
    robot = _slider()
    scene = SceneModel(
        (DomainBinding("cloth", "particles", cloth), DomainBinding("platform", "robot", robot))
    )
    faces = tuple(
        TrianglePrismSurface(name, "cloth", ids, half_thickness_m=rho)
        for name, ids in (("U", (0, 1, 2)), ("D", (1, 0, 3)))
    )
    verts = np.array([[-2.0, -2.0, -rho], [2.0, -2.0, -rho], [2.0, 2.0, -rho], [-2.0, 2.0, -rho]])
    p = CompliantPatch(
        RobotContactPatch("platform", 0, verts),
        "platform",
        width_m=0.002,
        stiffness_pa_per_m=500.0,
        damping_s_per_m=1000.0,
    )
    program = CompliantClothContactProgram(scene, "cloth", material, faces, ref, (p,))
    initial = SceneState(
        {"cloth": ParticleState(x, v), "platform": RobotState(np.zeros(1), np.zeros(1))}
    )
    inputs = SceneInputs({"cloth": StepInputs.zeros(4), "platform": RobotInputs.zeros(1)})
    return Case(program, initial, inputs, 1 / 64, 32)


def _patches(program, poses):
    world = {}
    for patch in program.patches:
        geom = patch.geometry
        vertices = geom.local_vertices
        if patch.robot_domain is not None:
            pose = poses[patch.robot_domain]
            vertices = vertices @ pose.rotation[geom.link_index].T + pose.position[geom.link_index]
        world[geom.key] = vertices.tolist()
    return world


def frame_record(program, state, t, sample=None, sample_t=None):
    particle = state.domains[program.cloth_domain]
    poses = {n: forward_kinematics(r.model, state.domains[n].q) for n, r in program.robots.items()}
    data = dict(
        t=t,
        cloth_position_m=particle.position.tolist(),
        cloth_velocity_m_s=particle.velocity.tolist(),
        robots={
            name: dict(
                q=state.domains[name].q.tolist(),
                v=state.domains[name].v.tolist(),
                links_m=poses[name].position.tolist(),
            )
            for name in program.robots
        },
        patch_vertices_m=_patches(program, poses),
    )
    if program.robot_contact is not None:
        data["body_capsules_m"] = {
            c.key: [
                (
                    poses[c.domain].position[c.link_index]
                    + poses[c.domain].rotation[c.link_index] @ p
                ).tolist()
                for p in (c.start, c.end)
            ]
            for c in program.robot_contact.capsules
        }
    if sample is not None:
        data["force_sample_s"] = sample_t
        data["contacts"] = {
            key: dict(
                points_m=val["points_m"].tolist(),
                force_n=val["force_n"].tolist(),
                normal_force_n=val["normal_force_n"].tolist(),
                normal=val["normal"].tolist(),
                **({"site_normals": val["site_normals"].tolist()} if "site_normals" in val else {}),
            )
            for key, val in sample.items()
        }
    return data


def _robot_record(model):
    record = {}
    for field in fields(model):
        value = getattr(model, field.name)
        record[field.name] = value.tolist() if isinstance(value, np.ndarray) else value
    return record


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def trace_start(case, name):
    p = case.program
    s = case.initial
    trace = dict(
        schema="phymeo-compliant-case-v1",
        case=name,
        backend="numpy",
        dt_s=case.dt,
        substeps=case.substeps,
        faces=p.faces.tolist(),
        robot_parents={n: r.model.parents.tolist() for n, r in p.robots.items()},
        robot_models={n: _robot_record(r.model) for n, r in p.robots.items()},
        owners=[f"{f.key}/{n}" for f in p.surfaces for n in f.node_indices],
        patches=[
            dict(
                name=a.geometry.key,
                robot=a.robot_domain,
                link_index=a.geometry.link_index,
                local_vertices_m=a.geometry.local_vertices.tolist(),
                normal_sign=a.normal_sign,
                width_m=a.width_m,
                stiffness_pa_per_m=a.stiffness_pa_per_m,
                **({"radius_m": a.radius_m} if a.radius_m is not None else {}),
                damping_s_per_m=a.damping_s_per_m,
                friction=a.friction,
                slip_speed_m_s=a.slip_speed_m_s,
            )
            for a in p.patches
        ],
        frames=[frame_record(p, s, 0.0, p.observe(s), 0.0)],
        samples=[],
        events=[],
    )
    if p.robot_contact is not None:
        system = p.robot_contact
        trace["robot_contact"] = dict(
            capsules=[_robot_record(c) for c in system.capsules],
            planes=[_robot_record(c) for c in system.planes],
            limits={n: _robot_record(limits) for n, limits in system.limits.items()},
            material=_robot_record(system.material),
            excluded_pairs=system.excluded_pairs,
        )
        trace["robot_samples"] = []
    return trace


def record_step(case, trace, result, t):
    if not result.report.accepted:
        trace.update(completed=False, failure=result.report.reason, time_s=t)
        return False
    h = case.dt / case.substeps
    for i, rep in enumerate(result.report.substeps):
        rows = rep.diagnostics["patches"]
        body = rep.diagnostics.get("robot_contact")
        if body is not None:
            trace["robot_samples"].append(dict(t=t + (i + 0.5) * h, **_jsonable(body)))
        for key, r in rows.items():
            trace["samples"].append(
                dict(
                    t=t + (i + 0.5) * h,
                    patch=key,
                    normal_n=float(np.sum(r["normal_force_n"])),
                    force_n=r["force_n"].sum(axis=0).tolist(),
                    normals_n=r["normal_force_n"].tolist(),
                    penetration_m=float(
                        np.max(np.maximum(-r["gap_m"][r["in_patch"]], 0.0), initial=0.0)
                    ),
                )
            )
    trace["frames"].append(
        frame_record(case.program, result.state, t + case.dt, rows, t + case.dt - h / 2)
    )
    if trace.get("robot_samples"):
        trace["frames"][-1]["robot_contact"] = trace["robot_samples"][-1]
    return True


def run_impact(*, support=False):
    case = build_impact(support=support)
    trace = trace_start(case, "support" if support else "impact")
    state = case.initial
    with case.program.stepper(
        StepSettings(substeps=case.substeps, position_iterations=1, velocity_iterations=1)
    ) as runtime:
        for i in range(8):
            result = runtime.step(state, case.inputs, case.dt)
            if not record_step(case, trace, result, i * case.dt):
                return trace
            state = result.state
    samples = trace["samples"]
    forces = np.array([r["normal_n"] for r in samples])
    peak = int(np.argmax(forces))
    loaded = np.flatnonzero(forces > 0)
    zero = np.flatnonzero((np.arange(len(samples)) > peak) & (forces == 0))
    end = case.program.observe(state)["platform"]
    end_force = float(end["normal_force_n"].sum())
    impulse = sum(r["normal_n"] * case.dt / case.substeps for r in samples)
    end_robot = state.domains["platform"]
    metrics = dict(
        duration_s=8 * case.dt,
        peak_force_n=float(forces[peak]),
        peak_sample_s=samples[peak]["t"],
        first_loaded_sample_s=samples[loaded[0]]["t"] if loaded.size else None,
        first_zero_after_peak_s=samples[zero[0]]["t"] if zero.size else None,
        terminal_normal_n=end_force,
        terminal_layer_gaps_m=end["gap_m"].tolist(),
        total_normal_impulse_ns=impulse,
        platform_end_momentum_ns=float(2 * end_robot.v[0]),
        maximum_sampled_penetration_m=max(r["penetration_m"] for r in samples),
    )
    ok = (
        (np.all(forces > 0) and end_force > 0)
        if support
        else (
            forces[0] == 0
            and 0 < peak < len(forces) - 1
            and loaded.size > 0
            and zero.size > 0
            and end_force == 0
        )
    )
    trace.update(completed=bool(ok), metrics=metrics)
    if not ok:
        trace["failure"] = "expected support or impact/load/release sequence not observed"
    return trace
