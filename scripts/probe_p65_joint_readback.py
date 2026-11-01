"""Probe the public B2/thick constructor and one prescribed-load articulated HOST replay."""

import argparse
import gzip
import json
import traceback
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from fractions import Fraction as Q
from pathlib import Path

import numpy as np
from verify_p65_contact_layer_screen import INPUT, sha

import phymeo as ph
from phymeo.constraints.bending import BendingConstraints
from phymeo.constraints.cloth_material import ClothMaterial, CompositeMaterialBudgets
from phymeo.core.scene import fingerprint
from phymeo.coupling.composite_cloth import CompositeClothParticipant
from phymeo.dynamics.robot_kinematics import point_jacobians
from phymeo.geometry.robot_patch_sweep import patch_geometry
from phymeo.validation.thick_contact_mapping import ThickContactMappingLedger

PLAN = INPUT.with_name("p65-b2-joint-readback-protocol.json")
PLAN_SHA = "1b70ba85d8255f6e390626952ac84c466083fb755f8033f83d3e54c0fb7ab1b4"


def encode(value):
    if isinstance(value, np.ndarray):
        return dict(dtype=value.dtype.str, shape=list(value.shape), values=encode(value.tolist()))
    if isinstance(value, np.generic):
        return encode(value.item())
    if isinstance(value, Q):
        return str(value)
    if is_dataclass(value):
        return {f.name: encode(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Mapping):
        return {k: encode(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [encode(v) for v in value]
    return value


def save(path, obj):
    if path.suffix == ".gz":
        with gzip.open(path, "xt", encoding="utf8") as file:
            json.dump(encode(obj), file, allow_nan=False, separators=(",", ":"))
    else:
        with path.open("x", encoding="utf8") as file:
            json.dump(encode(obj), file, indent=2, allow_nan=False)
            file.write("\n")


def fixture(plan):
    c = plan["cloth"]
    p = plan["platform"]
    bounds = plan["bounds"]
    m = float(Q(c["m_kg"]))
    rho = float(Q(c["half_thickness_m"]))
    X = np.array(c["nodes"], dtype=np.float64)
    cloth = ph.ParticleModel(np.full(4, 1 / m), np.array([0.0, 0.0, float(Q(c["gravity_z"]))]))
    material = ClothMaterial(
        ph.DistanceConstraints(np.array(c["distance_edges"]), np.full(3, 1.0), np.full(3, 1.0)),
        BendingConstraints(
            np.array([c["hinge"]]), np.array([float(Q(c["rest_angle_rad"]))]), np.array([1.0])
        ),
    )
    robot = ph.RobotModel(
        np.array([-1]),
        ("prismatic",),
        np.zeros((1, 3)),
        np.array([[0.0, 0.0, 0.0, 1.0]]),
        np.array([[0.0, 0.0, 1.0]]),
        np.array([4 * m]),
        np.zeros((1, 3)),
        np.full((1, 3), 4 * m),
        np.zeros(3),
    )
    patch = ph.RobotContactPatch(
        "platform", 0, np.array([[float(Q(v)) for v in row] for row in p["local_vertices"]])
    )
    surfaces = tuple(
        ph.TrianglePrismSurface(name, "cloth", tuple(ids), half_thickness_m=rho)
        for name, ids in zip(("U", "D"), c["faces"], strict=True)
    )
    scene = ph.SceneModel(
        (ph.DomainBinding("cloth", "particles", cloth), ph.DomainBinding("robot", "robot", robot))
    )
    state = ph.SceneState(
        {
            "cloth": ph.ParticleState(X, np.zeros_like(X)),
            "robot": ph.RobotState(np.zeros(1), np.zeros(1)),
        }
    )
    inputs = ph.SceneInputs(
        {
            "cloth": ph.StepInputs.zeros(4),
            "robot": ph.RobotInputs(np.zeros(1), np.zeros((1, 3)), np.zeros((1, 3))),
        }
    )
    participants = {
        "cloth": CompositeClothParticipant(
            cloth,
            material,
            budgets=CompositeMaterialBudgets(bounds["joint_energy_j"], bounds["joint_angular_nms"]),
        ),
        "robot": ph.RobotParticipant(robot),
    }
    base = ph.ArticulatedCouplingProgram(
        scene,
        participants,
        ph.RobotSphereAttachments(np.empty(0, dtype=int), np.empty((0, 3)), np.empty(0)),
        (),
    )
    return scene, state, inputs, base, material, robot, cloth, surfaces, patch


def observe(robot, cloth, state, surfaces, patch, world):
    x = state.domains["cloth"]
    r = state.domains["robot"]
    rho = surfaces[0].half_thickness_m
    faces = np.array([s.node_indices for s in surfaces])
    ids = np.repeat(faces, 3, axis=0)
    weights = np.tile(np.eye(3), (2, 1))
    offsets = np.full(6, -rho)
    port = ph.TriangleOffsetPointPort(
        ph.ParticlePointPort(cloth, x, np.arange(4), domain="cloth"), ids, weights, offsets
    )
    J = port.jacobians
    points = port.positions()
    samples = [patch_geometry(robot, r.q, patch, point) for point in points]
    local = np.array([s.local_point for s in samples])
    robJ, ang = point_jacobians(robot, r.q, np.zeros(6, dtype=int), local)
    blocks = np.einsum("rijk,rj->rik", J, world)
    node = np.zeros((4, 3))
    np.add.at(node, ids.reshape(-1), blocks.reshape(-1, 3))
    effort = -np.einsum("rjd,rj->d", robJ, world)
    ledger = ThickContactMappingLedger(x.position, ids, weights, offsets)
    try:
        mapping = dict(
            all_owners=True, report=ledger.require(J, world, node, x.velocity, tolerance=1e-8)
        )
    except ValueError as exc:
        if str(exc) != "different quadratic fields":
            raise
        by_face = []
        for first in (0, 3):
            group = slice(first, first + 3)
            contribution = np.zeros_like(node)
            np.add.at(contribution, ids[group].reshape(-1), blocks[group].reshape(-1, 3))
            local_ledger = ThickContactMappingLedger(
                x.position, ids[group], weights[group], offsets[group]
            )
            by_face.append(
                local_ledger.require(
                    J[group], world[group], contribution, x.velocity, tolerance=1e-8
                )
            )
        mapping = dict(all_owners=False, reason=str(exc), per_face=by_face)
    manifolds = []
    for face in faces:
        m = ph.triangle_prism_patch_manifold(
            robot, r.q, patch, x.position[face], half_thickness_m=rho, geometric_budget_m=0.0
        )
        manifolds.append(
            dict(
                status=m.status,
                gap=m.exact_gap_bounds,
                normal=m.normal,
                material=[p.material for p in m.points],
            )
        )
    return dict(
        triangles=ids,
        weights=weights,
        offsets=offsets,
        points=points,
        point_velocity=port.velocities(),
        jacobians=J,
        robot_local=local,
        robot_points=np.array([s.world_point for s in samples]),
        robot_j=robJ,
        robot_angular_j=ang,
        inside=[s.projection_inside for s in samples],
        world_error=[s.world_error_m for s in samples],
        node=node,
        robot_effort=effort,
        mapping=mapping,
        manifolds=manifolds,
    )


def calibrated_moments(plan):
    conf = plan["load_source"]
    file = INPUT.with_name(conf["report"])
    if sha(file, normalize=True) != conf["sha256_lf"]:
        raise ValueError("different calibration report")
    prior = json.loads(file.read_bytes())
    packet = INPUT.parents[2] / prior["raw_directory"] / "step-0" / "end.state.json"
    if (
        sha(packet) != conf["end_sha256"]
        or prior["steps"][0]["hashes"]["end.state.json"] != conf["end_sha256"]
    ):
        raise ValueError("different calibration endpoint")
    data = json.loads(packet.read_bytes())

    def scalar(key):
        word, rem = data[key]
        return Q(float.fromhex(word)) + Q(rem)

    h = Q(plan["h_s"])
    M = 2 * Q(plan["cloth"]["m_kg"])
    F = Q(plan["cloth"]["m_kg"]) / 4
    j = F * h - M * (scalar("y") - 1) / 1000
    b = F * h * h / 2 + M * scalar("gap")
    if (
        Q(data["t"]) != h
        or float(j) != prior["steps"][0]["contact_impulse_ns"]
        or float(b) != prior["steps"][0]["contact_time_moment_ns2"]
        or not 0 < b < h * j
    ):
        raise ValueError("unbound or inadmissible signal")
    return j, b


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--phase", choices=("automatic", "transport"), required=True)
    args = parser.parse_args()
    if (
        sha(PLAN, normalize=True) != PLAN_SHA
        or Path(ph.__file__).resolve().parent != (INPUT.parents[2] / "src" / "phymeo").resolve()
    ):
        raise ValueError("unexpected source or protocol")
    plan = json.loads(PLAN.read_bytes())
    root = args.root
    scene, state, inputs, base, mat, robot, cloth, surfaces, patch = fixture(plan)
    if args.phase == "automatic":
        try:
            ph.DynamicRobotThickContactProgram(
                scene,
                base,
                surfaces,
                (patch,),
                friction_by_collider={patch.key: 0.0},
                materials={"cloth": mat},
                budgets=ph.MaterialContactBudgets(1e-8, 1e-8),
                registry=base.registry,
            )
        except Exception as exc:
            save(
                root / "automatic.json",
                dict(
                    admitted=False,
                    exception=type(exc).__name__,
                    reason=str(exc),
                    frames=[
                        dict(file=f.filename, line=f.lineno, function=f.name)
                        for f in traceback.extract_tb(exc.__traceback__)
                    ],
                    base=type(base).__name__,
                    cloth=type(base.participants["cloth"]).__name__,
                    robot=type(base.participants["robot"]).__name__,
                    material=type(mat).__name__,
                    protocol_sha256_lf=PLAN_SHA,
                ),
            )
            print(f"{type(exc).__name__}: {exc}")
            return 1
        save(root / "automatic.json", dict(admitted=True, protocol_sha256_lf=PLAN_SHA))
        return 0
    h = float(Q(plan["h_s"]))
    j, b = calibrated_moments(plan)
    world = np.zeros((6, 3))
    world[:, 2] = float(j / 6)
    snapshot = fingerprint((state, inputs))
    inlet = observe(robot, cloth, state, surfaces, patch, world)
    request = {"cloth": {"force": inlet["node"]}, "robot": {"effort": inlet["robot_effort"]}}
    settings = ph.StepSettings(
        substeps=1,
        position_iterations=512,
        velocity_iterations=512,
        tolerance=1e-8,
        record_substeps=True,
    )
    save(
        root / "inlet.json.gz",
        dict(
            scene=scene,
            state=state,
            inputs=inputs,
            material=mat,
            surfaces=surfaces,
            patch=patch,
            h=h,
            settings=settings,
            signal_j=j,
            signal_b=b,
            world=world,
            request=request,
            mapping=inlet,
            initial_fingerprint=snapshot,
            protocol_sha256_lf=PLAN_SHA,
        ),
    )
    with ph.SceneLoadReplay(
        scene,
        base,
        state,
        inputs,
        h,
        channels={"cloth": ("force",), "robot": ("effort",)},
        settings=settings,
        context=ph.BackendContext("numpy"),
        registry=base.registry,
    ) as replay:
        try:
            replay.evaluate({"robot": {"effort_time_moment": np.array([-float(b)])}})
        except ValueError as exc:
            if str(exc) != "integrated load field is not a declared channel":
                raise
            save(root / "time-channel.json", dict(rejected=True, reason=str(exc), before_step=True))
        else:
            raise ValueError("undeclared channel was accepted")
        trial = replay.evaluate(request)
        save(root / "trial.json.gz", trial)
    if fingerprint((state, inputs)) != snapshot:
        raise ValueError("caller state or original inputs mutated")
    save(
        root / "bindings.json",
        dict(
            scene_identity=scene.identity,
            original_inputs=fingerprint(inputs),
            original_cloth=fingerprint(inputs.domains["cloth"]),
            effective_inputs=fingerprint(trial.effective_inputs),
            effective_cloth=fingerprint(trial.effective_inputs.domains["cloth"]),
            caller_unchanged=fingerprint((state, inputs)) == snapshot,
            readback_residual=trial.result.report.max_residual,
        ),
    )
    if not trial.result.report.accepted:
        print("inner trial rejected:", trial.result.report.reason)
        return 1
    end = trial.result.state
    endpoint = observe(robot, cloth, end, surfaces, patch, world)
    sweeps = []
    for surface in surfaces:
        idx = list(surface.node_indices)
        sweep = ph.triangle_prism_patch_sweep(
            robot,
            state.domains["robot"].q,
            end.domains["robot"].q,
            patch,
            state.domains["cloth"].position[idx],
            end.domains["cloth"].position[idx],
            half_thickness_m=surface.half_thickness_m,
            penetration_budget_m=0.0,
            max_intervals=4096,
            max_depth=32,
        )
        sweeps.append(sweep)
    save(root / "geometry.json.gz", dict(endpoint=endpoint, sweeps=sweeps))
    print(
        "HOST prescribed replay accepted; residual",
        trial.result.report.max_residual,
        "automatic geometry",
        [s.status for s in sweeps],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
