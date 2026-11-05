"""Frozen C1 workloads and a scalar, independent semi-implicit material oracle."""

from dataclasses import dataclass

import numpy as np

H = 1 / 128
STEPS = 32
WORLD_COUNTS = (1, 32, 128)
CASES = ("free", "loaded", "birth")
BOX, BALL = 2, 9
HALF = np.array((1 / 2, 3 / 8, 1 / 8))
RADIUS, WIDTH, STIFFNESS, MASS = 1 / 8, 1 / 16, 128.0, 1.0
POSITION_ATOL_M = 5e-6
VELOCITY_ATOL_M_S = 5e-5
GAP_ATOL_M = 5e-6
FORCE_ATOL_N = 3e-4
MOMENTUM_ATOL_N_S = 2e-6
NORMAL_ATOL = 5e-4
CYCLIC = np.array(((0, 0, 1), (1, 0, 0), (0, 1, 0)), dtype=float)


@dataclass(frozen=True)
class Seed:
    world: int
    case: str
    feature: str
    mirror: int
    point: tuple
    normal: tuple
    center: tuple
    velocity: tuple


def matrix():
    return tuple((case, count) for case in CASES for count in WORLD_COUNTS)


def seeds(case, count, *, feature="edge"):
    if case not in CASES or count not in WORLD_COUNTS:
        raise ValueError("only the frozen nine C1 workloads may enter the cost matrix")
    if feature not in ("face", "edge", "corner"):
        raise ValueError("unknown independently defined finite-box feature")
    result = []
    for i in range(count):
        mirror = 1 if i % 2 == 0 else -1
        actual = feature if case == "birth" else "face"
        if actual == "face":
            point, normal = (0, 0, HALF[2]), (0, 0, 1)
        elif actual == "edge":
            point, normal = (mirror * HALF[0], HALF[1], 0), (mirror * 3 / 5, 4 / 5, 0)
        else:
            point = (mirror * HALF[0], HALF[1], HALF[2])
            normal = (mirror / 3, 2 / 3, 2 / 3)
        gap = {"free": 3 / 4, "loaded": 1 / 32, "birth": 3 / 32}[case]
        center = np.asarray(point) + (RADIUS + gap) * np.asarray(normal)
        velocity = (
            -0.5 * np.asarray(normal)
            if case == "birth"
            else np.array((mirror / 32, 0, 0))
        )
        result.append(
            Seed(17 + 3 * i, case, actual, mirror, point, normal, tuple(center), tuple(velocity))
        )
    return tuple(result)


def gravity(case):
    if case not in CASES:
        raise ValueError(case)
    return np.array((0, 0, 0 if case == "birth" else -4), dtype=float)


def builder(ph, case, count, *, feature="edge", transformed=False):
    """Public declarations only; two shapes/world, including at W128's 256-shape limit."""
    rotation = CYCLIC if transformed else np.eye(3)
    offset = np.array((3, -2, 1)) if transformed else np.zeros(3)
    orientation = (0.5,) * 4 if transformed else (0, 0, 0, 1)
    make = ph.ContactSceneBuilder(gravity=rotation @ gravity(case))
    for seed in seeds(case, count, feature=feature):
        make.add_box(
            BOX, HALF, mass=None, world=seed.world,
            position=offset, orientation_xyzw=orientation,
        )
        make.add_sphere(
            BALL, RADIUS, mass=MASS, world=seed.world,
            position=offset + rotation @ seed.center, velocity=rotation @ seed.velocity,
        )
        make.pair(BOX, BALL, ph.NormalLayer(WIDTH, STIFFNESS), world=seed.world)
    return make


def oracle(seed, *, transformed=False):
    """No Phymeo geometry, law or integrator supplies any expected value."""
    rotation = CYCLIC if transformed else np.eye(3)
    offset = np.array((3, -2, 1)) if transformed else np.zeros(3)
    normal = np.asarray(seed.normal)
    position, velocity = np.asarray(seed.center).copy(), np.asarray(seed.velocity).copy()
    contact_point = np.asarray(seed.point)
    points, velocities, entry_gaps, forces, points_on_box = [position.copy()], [velocity.copy()], [], [], []
    for _ in range(STEPS):
        if seed.feature == "face":
            gap = position[2] - HALF[2] - RADIUS
            point = np.array((position[0], position[1], HALF[2]))
        else:
            gap = float((position - contact_point) @ normal - RADIUS)
            point = contact_point.copy()
        force = max(0.0, STIFFNESS * (WIDTH - gap))
        entry_gaps.append(gap)
        forces.append(force * normal)
        points_on_box.append(point)
        velocity = velocity + H * (gravity(seed.case) + force * normal / MASS)
        position = position + H * velocity
        velocities.append(velocity.copy())
        points.append(position.copy())
    return dict(
        position_m=np.asarray(points) @ rotation.T + offset,
        velocity_m_s=np.asarray(velocities) @ rotation.T,
        entry_gap_m=np.asarray(entry_gaps),
        force_on_ball_n=np.asarray(forces) @ rotation.T,
        point_on_box_m=np.asarray(points_on_box) @ rotation.T + offset,
        normal=rotation @ normal,
        gravity_m_s2=rotation @ gravity(seed.case),
        fixed_box_position_m=offset,
    )


def certify_frame(frame, rows, case, count, step, *, feature="edge", transformed=False):
    """Validate actual entry loads and actual accepted state against the independent recurrence."""
    if not 0 <= step < STEPS:
        raise ValueError("oracle step outside the fixed window")
    contacts = {(c.first.world, c.first.body, c.second.body): c for c in frame.loads.contacts}
    errors = dict(position_m=0.0, velocity_m_s=0.0, force_n=0.0, momentum_n_s=0.0)
    for seed in seeds(case, count, feature=feature):
        expected = oracle(seed, transformed=transformed)
        box, ball = rows[seed.world, BOX], rows[seed.world, BALL]
        entry, state = frame.entry.rigid, frame.state.rigid
        force = frame.loads.inputs.rigid.force[ball].astype(float)
        want_force = expected["force_on_ball_n"][step]
        np.testing.assert_allclose(
            state.position[ball], expected["position_m"][step + 1], rtol=0, atol=POSITION_ATOL_M
        )
        np.testing.assert_allclose(
            entry.position[ball], expected["position_m"][step], rtol=0, atol=POSITION_ATOL_M
        )
        np.testing.assert_allclose(
            state.velocity[ball], expected["velocity_m_s"][step + 1],
            rtol=0, atol=VELOCITY_ATOL_M_S,
        )
        np.testing.assert_allclose(force, want_force, rtol=0, atol=FORCE_ATOL_N)
        np.testing.assert_allclose(frame.loads.inputs.rigid.force[box], -force, rtol=0, atol=1e-7)
        np.testing.assert_allclose(
            state.position[box], expected["fixed_box_position_m"], rtol=0, atol=POSITION_ATOL_M
        )
        np.testing.assert_array_equal(state.velocity[box], 0)
        np.testing.assert_allclose(state.angular_velocity[ball], 0, rtol=0, atol=1e-6)
        delta_p = MASS * (state.velocity[ball].astype(float) - entry.velocity[ball].astype(float))
        residual = delta_p - H * (force + MASS * expected["gravity_m_s2"])
        np.testing.assert_allclose(residual, 0, rtol=0, atol=MOMENTUM_ATOL_N_S)
        c = contacts.get((seed.world, BOX, BALL))
        if np.linalg.norm(want_force) > FORCE_ATOL_N:
            assert c is not None
            np.testing.assert_allclose(c.normal, expected["normal"], rtol=0, atol=NORMAL_ATOL)
            np.testing.assert_allclose(c.gap_m, expected["entry_gap_m"][step], rtol=0, atol=GAP_ATOL_M)
            assert c.gap_m > 0 and c.force_n > 0 and len(c.sites) == 1
            site = c.sites[0]
            np.testing.assert_allclose(
                site.application_point_first_m, expected["point_on_box_m"][step],
                rtol=0, atol=POSITION_ATOL_M,
            )
            np.testing.assert_allclose(
                site.application_point_second_m,
                entry.position[ball].astype(float) - RADIUS * expected["normal"],
                rtol=0, atol=POSITION_ATOL_M,
            )
        elif c is not None:
            assert abs(c.force_n) <= FORCE_ATOL_N
        errors["position_m"] = max(
            errors["position_m"],
            float(np.max(abs(state.position[ball] - expected["position_m"][step + 1]))),
        )
        errors["velocity_m_s"] = max(
            errors["velocity_m_s"],
            float(np.max(abs(state.velocity[ball] - expected["velocity_m_s"][step + 1]))),
        )
        errors["force_n"] = max(errors["force_n"], float(np.max(abs(force - want_force))))
        errors["momentum_n_s"] = max(errors["momentum_n_s"], float(np.max(abs(residual))))
    assert frame.rigid_report.accepted and frame.particle_report is None
    assert all(p.status == "filtered" for p in frame.path.pairs if p.first.world != p.second.world)
    return errors
