"""Connected cloth strips and independent force/kinematic checks for the multiface profile."""

from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass, replace

import numpy as np

import phymeo as ph
from phymeo.constraints.cloth_material import ClothMaterial
from phymeo.constraints.robot_contact import LinkCapsule
from phymeo.core.scene import fingerprint
from phymeo.dynamics.robot_contact import RobotContactSystem

H = 1 / 256
STEPS = 4


@dataclass
class Scene:
    program: object
    initial: object
    controls: tuple
    reference: np.ndarray
    faces: tuple
    keys: tuple
    areas: np.ndarray


def make_world(world, face_count, operator):
    """Refine one fixed physical rectangle, with one shaft reaching every real face."""
    if face_count not in (2, 8):
        raise ValueError("expected two or eight triangles")
    columns = face_count // 2
    shift = np.array([world / 32, -world / 64, 0.0])
    grid = np.array(
        [(x, y, 0) for x in np.linspace(-0.5, 0.5, columns + 1) for y in (-0.125, 0.125)]
    )
    x = grid + shift
    faces = []
    for j in range(columns):
        a = 2 * j
        faces.extend(((a, a + 2, a + 3), (a, a + 3, a + 1)))
    keys = tuple(f"strip-{world}-face-{i}" for i in range(face_count))
    tri = x[np.asarray(faces)]
    areas = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) / 2
    mass = np.zeros(len(x))
    for face, area in zip(faces, areas, strict=True):
        mass[list(face)] += (1.5 + (world % 4) / 32) * area / 3
    edges = sorted(
        {tuple(sorted((a, b))) for i, j, k in faces for a, b in ((i, j), (j, k), (k, i))}
    )
    indices = np.asarray(edges)
    rest = np.linalg.norm(x[indices[:, 0]] - x[indices[:, 1]], axis=1)
    elastic = ph.DistanceConstraints(indices, rest, np.full(len(edges), 1 / 64))
    cloth = ph.ParticleModel(1 / mass, np.array([0.0, 0.0, -0.25]))
    robot = ph.RobotModel(
        np.array([-1]),
        ("prismatic",),
        np.array([shift + [0, 0, 0.15625 + (world % 4) / 1024]]),
        np.array([[0.0, 0.0, 0.0, 1.0]]),
        np.array([[0.0, 0.0, 1.0]]),
        np.array([0.75 + (world % 4) / 64]),
        np.zeros((1, 3)),
        np.ones((1, 3)),
        np.zeros(3),
    )
    link = LinkCapsule(
        "shaft", "link", 0, np.array([-0.625, 0.0, 0.0]), np.array([0.625, 0.0, 0.0]), 1 / 16
    )
    scene = ph.SceneModel(
        (ph.DomainBinding("cloth", "particles", cloth), ph.DomainBinding("link", "robot", robot))
    )
    surfaces = tuple(
        ph.TrianglePrismSurface(k, "cloth", f, half_thickness_m=1 / 64)
        for k, f in zip(keys, faces, strict=True)
    )
    law = ph.ClothLinkMaterial(0.125, 96.0, 0.25, 0.25, 0.0625)
    program = ph.CompliantClothContactProgram(
        scene,
        "cloth",
        ClothMaterial(distance=elastic),
        surfaces,
        x,
        (),
        robot_contact=RobotContactSystem({"link": robot}, capsules=(link,)),
        cloth_link_material=law,
        cloth_link_operator=operator,
        max_penetration_m=1 / 64,
    )
    gx, gy = grid[:, 0], grid[:, 1]
    v = np.column_stack(
        (
            0.025 + 0.025 * gy,
            -0.015 * gx,
            0.07 * gx + 0.08 * gy + 0.15 * gx * gy + (world % 4) / 512,
        )
    )
    state = ph.SceneState(
        {"cloth": ph.ParticleState(x, v), "link": ph.RobotState(np.zeros(1), np.array([0.01]))}
    )
    controls = []
    for tick in range(STEPS):
        # Fixed, bounded load sequence. The cloth and finite-mass link both evolve freely.
        a = np.column_stack((0.03 + 0 * gx, 0 * gx, 0.04 * gx + (-1) ** tick * gy / 16))
        controls.append(
            ph.SceneInputs(
                {
                    "cloth": ph.StepInputs(mass[:, None] * a),
                    "link": ph.RobotInputs(
                        np.array([((world % 4) - 1.5 + tick) / 128]),
                        np.zeros((1, 3)),
                        np.zeros((1, 3)),
                    ),
                }
            )
        )
    return Scene(program, state, tuple(controls), x, tuple(faces), keys, areas)


def native(value):
    """Retain complete report, state, history and any accepted substep snapshots in the receipt."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value):
        return {f.name: native(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Mapping):
        return {k: native(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [native(v) for v in value]
    return value


def physical_hash(frame):
    substeps = tuple(
        replace(sub, timings={key: 0.0 for key in sub.timings}) for sub in frame.report.substeps
    )
    return fingerprint(replace(frame, report=replace(frame.report, substeps=substeps)))


def whole_window(worlds, steppers, *, isolated=False):
    states = [w.initial for w in worlds]
    frames = [[] for _ in worlds]
    schedule = (
        ((i, t) for i in range(len(worlds)) for t in range(STEPS))
        if isolated
        else ((i, t) for t in range(STEPS) for i in range(len(worlds)))
    )
    for i, t in schedule:
        out = steppers[i].step(states[i], worlds[i].controls[t], H)
        if not out.report.accepted:
            raise AssertionError((i, t, out.report.reason))
        frames[i].append(out)
        states[i] = out.state
    return frames


def certify(world, frames):
    evidence = []
    by_face = {k: [] for k in world.keys}
    for t, frame in enumerate(frames):
        d = frame.report.substeps[0].diagnostics["robot_contact"]
        rows = {row["face"]: row for row in d["cloth_links"]}
        assert set(rows) == set(world.keys) and len(d["cloth_links"]) == len(world.keys)
        assert d["cloth_link_candidates"] == len(world.keys)
        face_rows = []
        for i, (key, refarea) in enumerate(zip(world.keys, world.areas, strict=True)):
            row = rows[key]
            tri = row["face_positions_m"]
            n = row["normal"]
            sites = row["sites"]
            assert tuple(row["face_nodes"]) == world.faces[i]
            assert row["normal_force_n"] > 0.0 and 1 <= len(sites) <= 2
            av = np.cross(tri[1] - tri[0], tri[2] - tri[0])
            area = np.linalg.norm(av) / 2
            geometric_normal = av / (2 * area)
            force = np.zeros(3)
            normal_sum = 0.0
            for site in sites:
                ids, weights = zip(*site["face_weights"], strict=True)
                weights = np.asarray(weights)
                assert set(ids) == set(world.faces[i]) and abs(weights.sum() - 1) < 1e-12
                assert 0 <= site["axis_parameter"] <= 1 and min(weights) >= 0
                lookup = dict(zip(world.faces[i], tri, strict=True))
                np.testing.assert_allclose(
                    sum(w * lookup[k] for k, w in zip(ids, weights, strict=True)),
                    site["point_m"],
                    atol=3e-10,
                    rtol=0,
                )
                expected = (
                    96
                    * refarea
                    * max(0, 0.125 - row["gap_m"])
                    * max(0, 1 - 0.25 * site["normal_speed_m_s"])
                    / len(sites)
                )
                np.testing.assert_allclose(site["normal_force_n"], expected, rtol=4e-6, atol=1e-8)
                np.testing.assert_allclose(
                    site["force_on_cloth_n"] + site["force_on_link_n"], np.zeros(3), atol=1e-12
                )
                force += site["force_on_cloth_n"]
                normal_sum += site["normal_force_n"]
            np.testing.assert_allclose(n @ force, normal_sum, rtol=5e-6, atol=1e-8)
            np.testing.assert_allclose(normal_sum, row["normal_force_n"], rtol=0, atol=1e-12)
            center = tri.mean(0)
            by_face[key].append((area, geometric_normal, center))
            face_rows.append(
                dict(
                    face=key,
                    site_count=len(sites),
                    force_n=force.tolist(),
                    normal_force_n=normal_sum,
                    reference_area_m2=refarea,
                    actual_area_m2=area,
                    geometric_normal=geometric_normal.tolist(),
                    centroid_m=center.tolist(),
                )
            )
        expected_effort = -sum(f["force_n"][2] for f in face_rows)
        np.testing.assert_allclose(
            d["cloth_link_efforts"]["link"], [expected_effort], rtol=1e-11, atol=1e-12
        )
        evidence.append(
            dict(
                tick=t + 1,
                force_sample="substep-midpoint",
                faces=face_rows,
                joint_effort_n=expected_effort,
            )
        )
    changes = {}
    for key, poses in by_face.items():
        a0, n0, c0 = poses[0]
        changes[key] = dict(
            centroid_shift_m=float(max(np.linalg.norm(c - c0) for _, _, c in poses)),
            normal_change_l2=float(max(np.linalg.norm(n - n0) for _, n, _ in poses)),
            area_change_m2=float(max(abs(a - a0) for a, _, _ in poses)),
        )
        assert changes[key]["centroid_shift_m"] > 1e-7 and changes[key]["normal_change_l2"] > 1e-6
    assert abs(frames[-1].state.domains["link"].q[0] - world.initial.domains["link"].q[0]) > 1e-8
    return {"midpoints": evidence, "observed_changes": changes}
