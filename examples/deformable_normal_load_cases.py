"""Fixed D4-N five-node membrane inputs and independent reference mechanics.

These are engineering qualification cases, not product solvers. Each facet is
one original NormalLayer spring. Inputs/gates are fixed before first execution.
"""

import math
from fractions import Fraction as Q
from itertools import combinations
from types import MappingProxyType

import numpy as np

import phymeo as ph

NAMES = ("n1_sphere", "n2_capsule", "n3_soft", "n1_fine")
T = 3.0
BASE_H = 1 / 1024
RADIUS, WIDTH, STIFFNESS, DAMPING, LOAD = 1 / 8, 1 / 16, 128.0, 0.5, 16.0
SHORT_ATOL, SHORT_RTOL = 5e-6, 5e-5
LONG_POSITION, LONG_VELOCITY, LONG_ANGLE = 2e-3, 1e-2, 2e-3
SUPPORT_WINDOWS = ((0.75, 1.0), (2.75, 3.0))
EVENT_DURATION = 1 / 128
SETTINGS = ph.StepSettings(substeps=1, position_iterations=200, tolerance=2e-6)


def owned(value, dtype):
    array = np.asarray(value, dtype=dtype)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def make_case(name, *, world=7):
    if name not in NAMES:
        raise ValueError("unknown fixed normal-load case")
    if type(world) is not int or world < 0:
        raise ValueError("world must be a nonnegative builtin int")
    h = BASE_H / 2 if name == "n1_fine" else BASE_H
    compliance = (2 if name == "n3_soft" else 1) / 16384
    points = owned(((0, 0, 0), (-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)), np.float32)
    ids = owned((1001, 1013, 1021, 1031, 1049), np.int64)
    faces = owned(((0, 1, 2), (0, 2, 3), (0, 3, 4), (0, 4, 1)), np.int64)
    facets = owned((503, 509, 521, 523), np.int64)
    edges = owned(((0, 1), (0, 2), (0, 3), (0, 4)), np.int64)
    rest = owned((math.sqrt(2),) * 4, np.float64)
    compliances = owned((compliance,) * 4, np.float64)
    weights = owned((2, 0, 0, 0, 0), np.float32)
    node = ph.DeformableNodeWorld(
        world,
        ph.ParticleModel(weights, np.zeros(3, np.float32)),
        ph.ParticleState(points, np.zeros_like(points)),
        ph.DistanceConstraints(edges, rest, compliances),
        ids,
        (ph.DeformableSurface(23, facets, ids[faces]),),
    )
    capsule = name == "n2_capsule"
    local = owned(((0, 0, -1 / 8), (0, 0, 1 / 8)) if capsule else ((0, 0, 0),), np.float64)
    root_position = RADIUS + WIDTH + 1 / 128 + (1 / 8 if capsule else 0)
    initial_root = ph.FloatingRobotState(
        np.array((0, 0, root_position, 0, 0, 0, 1), float),
        np.array((0, 0, -1 / 16, 0, 0, 0), float),
    )
    mass, inertia = 2.0, 0.25
    root_model = ph.FloatingRobotModel(
        ph.RigidBodyModel(np.array((1 / mass,)), np.full((1, 3), inertia), np.zeros(3)),
        np.zeros(3),
        None,
    )
    tree = ph.FloatingRobotTree(
        root_model, initial_root, kp=np.zeros(0), kd=np.zeros(0), effort_limit=np.zeros(0)
    )
    key = ph.ResidentParticipantKey(world, 19)
    participant = ph.ResidentParticipant(key, tree, (202,))
    surface, round_key = ph.CollisionKey(world, 101, 0), ph.CollisionKey(world, 202, 0)
    shape = ph.CollisionShape(
        round_key,
        "capsule" if capsule else "sphere",
        (41, 43) if capsule else (41,),
        RADIUS,
        revision=3,
    )
    binding = ph.ResidentParticipantBinding(
        key,
        ph.ResidentFloatingTreeBinding(
            round_key, world, 0, local, topology=participant.compiled.topology
        ),
    )
    law = ph.NormalLayer(WIDTH, STIFFNESS, DAMPING)
    return MappingProxyType(
        dict(
            name=name,
            world=world,
            h=h,
            steps=int(T / h),
            node=node,
            participant=participant,
            tree=tree,
            model=ph.DeformableBoundaryModel((ph.DeformableBoundarySurface(surface, 23), shape)),
            binding=binding,
            materials=MappingProxyType({(surface, round_key): law}),
            surface=surface,
            round=round_key,
            local=local,
            mass=mass,
            inertia=inertia,
            root_model=root_model,
            initial_root=initial_root,
            ids=ids,
            faces=faces,
            facets=facets,
            edges=edges,
            rest=rest,
            compliances=compliances,
            weights=weights,
            settings=SETTINGS,
            axis=owned((0, 0, 1), np.float64),
        )
    )


def manifest(case):
    return {
        key: case[key]
        for key in (
            "name",
            "world",
            "h",
            "steps",
            "node",
            "root_model",
            "initial_root",
            "model",
            "binding",
            "materials",
            "surface",
            "round",
            "local",
            "mass",
            "inertia",
            "ids",
            "faces",
            "facets",
            "edges",
            "rest",
            "compliances",
            "weights",
            "settings",
            "axis",
        )
    }


def load_at_step(case, step):
    if type(step) is not int or not 0 <= step < case["steps"]:
        raise ValueError("load step outside the fixed physical window")
    first, second = int(1 / case["h"]), int((9 / 8) / case["h"])
    sign = 1 if first <= step < second else -1
    return np.array((0, 0, sign * LOAD, 0, 0, 0), np.float64)


def controls(case, external):
    external = np.asarray(external, np.float64)
    return {
        case["participant"].key: case["tree"].inputs(
            np.zeros((0, 3)),
            ph.FloatingRobotInputs(np.zeros(0), external[None, :3], external[None, 3:]),
        )
    }


def create(case, mode, **options):
    return ph.TyxDeformableContactBatch(
        (case["node"],),
        (case["participant"],),
        case["model"],
        (case["binding"],),
        case["materials"],
        mode=mode,
        settings=case["settings"],
        contact_capacity=4,
        **options,
    )


def initial_reference(case):
    return (
        case["node"].initial_state.position.astype(float).copy(),
        case["node"].initial_state.velocity.astype(float).copy(),
        case["initial_root"].q.copy(),
        case["initial_root"].v.copy(),
    )


def root_state(state, case):
    return state.rigid.participants[case["participant"].key]


def dot(a, b):
    return sum(x * y for x, y in zip(a, b, strict=True))


def sub(a, b):
    return tuple(x - y for x, y in zip(a, b, strict=True))


def linear(matrix, rhs):
    n = len(rhs)
    rows = [list(matrix[i]) + [rhs[i]] for i in range(n)]
    for k in range(n):
        pivot = next((i for i in range(k, n) if rows[i][k]), None)
        if pivot is None:
            return None
        rows[k], rows[pivot] = rows[pivot], rows[k]
        value = rows[k][k]
        rows[k] = [x / value for x in rows[k]]
        for i in range(n):
            if i != k:
                value = rows[i][k]
                rows[i] = [a - value * b for a, b in zip(rows[i], rows[k], strict=True)]
    return tuple(rows[i][-1] for i in range(n))


def oracle(triangle, segment):
    t = [tuple(Q(float(x)) for x in row) for row in triangle]
    p = [tuple(Q(float(x)) for x in row) for row in segment]
    axis = sub(p[1], p[0])
    records = []
    for count in (1, 2, 3):
        for ids in combinations(range(3), count):
            base = t[ids[0]]
            edges = [sub(t[i], base) for i in ids[1:]]
            for state in (0, 1, 2):
                start = p[state] if state < 2 else p[0]
                columns = ([axis] if state == 2 else []) + [tuple(-x for x in e) for e in edges]
                offset = sub(start, base)
                rhs = [-dot(c, offset) for c in columns]
                gram = [[dot(a, b) for b in columns] for a in columns]
                solution = linear(gram, rhs) if columns else ()
                if solution is None:
                    continue
                s = solution[0] if state == 2 else Q(state)
                ws = solution[1:] if state == 2 else solution
                bary = [Q(0)] * 3
                bary[ids[0]] = 1 - sum(ws)
                for i, value in zip(ids[1:], ws, strict=True):
                    bary[i] = value
                if not (0 <= s <= 1 and min(bary) >= 0):
                    continue
                a = tuple(p[0][k] + s * axis[k] for k in range(3))
                b = tuple(sum(bary[i] * t[i][k] for i in range(3)) for k in range(3))
                d = sub(b, a)
                records.append((dot(d, d), s, tuple(bary), a, b))
    assert records
    best = min(x[0] for x in records)
    tied = [x for x in records if x[0] == best]
    return dict(
        distance2=best, axis_span=(min(x[1] for x in tied), max(x[1] for x in tied)), witnesses=tied
    )


def local_scale(triangle, segment, radius):
    return (
        max(float(np.linalg.norm(triangle[(i + 1) % 3] - triangle[i])) for i in range(3))
        if radius is None
        else max(
            *(float(np.linalg.norm(triangle[(i + 1) % 3] - triangle[i])) for i in range(3)),
            float(np.linalg.norm(segment[1] - segment[0])),
            float(radius),
        )
    )


def close(actual, expected, *, atol=SHORT_ATOL, rtol=SHORT_RTOL):
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)


def rotation(q):
    x, y, z, w = np.asarray(q, float) / np.linalg.norm(q)
    return np.array(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        )
    )


def rotate_step(q, omega, h):
    theta = h * np.linalg.norm(omega)
    a = np.r_[
        omega * (h / 2 if theta == 0 else math.sin(theta / 2) / np.linalg.norm(omega)),
        math.cos(theta / 2),
    ]
    b = np.asarray(q, float)
    result = np.r_[
        a[3] * b[:3] + b[3] * a[:3] + np.cross(a[:3], b[:3]), a[3] * b[3] - np.dot(a[:3], b[:3])
    ]
    return result / np.linalg.norm(result)


def reference_load(case, x, v, q, speed):
    """All finite facets from an independent exact active-face minimizer."""
    axis = case["local"] @ rotation(q[3:7]).T + q[:3]
    segment = np.vstack((axis, axis)) if len(axis) == 1 else axis
    total, wrench = np.zeros_like(x, dtype=float), np.zeros(6)
    records = []
    for ordinal, indices in enumerate(case["faces"]):
        exact = oracle(x[indices], segment)
        d2, s, bary, _, _ = exact["witnesses"][0]
        distance = math.sqrt(float(d2))
        if distance <= 0:
            raise ValueError("independent membrane distance has no normal")
        beta, s = np.asarray([float(a) for a in bary]), float(s)
        axis_point = (1 - s) * segment[0] + s * segment[1]
        node_point = beta @ x[indices]
        normal = (node_point - axis_point) / distance
        rigid_point = axis_point + RADIUS * normal
        gap = distance - RADIUS
        relative = beta @ v[indices] - speed[:3] - np.cross(speed[3:], rigid_point - q[:3])
        rate = float(normal @ relative)
        compression = max(WIDTH - gap, 0)
        magnitude = STIFFNESS * compression * max(1 - DAMPING * rate, 0)
        force = magnitude * normal
        total[indices] += beta[:, None] * force
        wrench[:3] -= force
        wrench[3:] += np.cross(rigid_point - q[:3], -force)
        oriented = np.cross(x[indices[1]] - x[indices[0]], x[indices[2]] - x[indices[0]])
        side = float((axis_point - node_point) @ oriented) / np.linalg.norm(oriented)
        records.append(
            dict(
                identity=(case["surface"], case["round"], int(case["facets"][ordinal])),
                ordinal=ordinal,
                distance=distance,
                gap=gap,
                bary=beta,
                s=s,
                normal=normal,
                node_point=node_point,
                rigid_point=rigid_point,
                axis_point=axis_point,
                relative=relative,
                rate=rate,
                compression=compression,
                magnitude=magnitude,
                force=force,
                side=side,
                local_scale=local_scale(x[indices], segment, RADIUS),
                tied_witnesses=tuple(
                    (
                        float(d),
                        float(a),
                        tuple(map(float, b)),
                        tuple(map(float, c)),
                        tuple(map(float, e)),
                    )
                    for d, a, b, c, e in exact["witnesses"]
                ),
            )
        )
    return total, wrench, records


def material_step(case, x, v, force, h, sweeps=None):
    weights = case["weights"].astype(float)
    fixed = weights == 0
    pv = v + h * weights[:, None] * force
    px = x + h * pv
    px[fixed], pv[fixed] = x[fixed], 0
    y, lam = px.copy(), np.zeros(len(case["edges"]))
    alpha = case["compliances"] / h / h
    count = case["settings"].position_iterations if sweeps is None else sweeps
    for _ in range(count):
        for e, (a, b) in enumerate(case["edges"]):
            delta = y[a] - y[b]
            length = np.linalg.norm(delta)
            direction = delta / length
            dl = -(length - case["rest"][e] + alpha[e] * lam[e]) / (
                weights[a] + weights[b] + alpha[e]
            )
            lam[e] += dl
            y[a] += weights[a] * dl * direction
            y[b] -= weights[b] * dl * direction
        if sweeps is None:
            lengths = np.linalg.norm(y[case["edges"][:, 0]] - y[case["edges"][:, 1]], axis=1)
            residual = np.max(
                abs(lengths - case["rest"] + alpha * lam)
                / np.maximum(case["rest"], case["settings"].reference_length)
            )
            if residual <= case["settings"].tolerance:
                break
    else:
        if sweeps is None:
            raise AssertionError("independent material reference exhausted original budget")
    velocity = pv + (y - px) / h
    velocity[fixed] = 0
    return y, velocity, lam


def rigid_step(case, q, v, wrench, h, external):
    force = wrench + external
    speed = v + h * np.r_[force[:3] / case["mass"], force[3:] / case["inertia"]]
    return np.r_[q[:3] + h * speed[:3], rotate_step(q[3:7], speed[3:], h)], speed


def continue_reference(case, state, external):
    x, v, q, speed = state
    force, wrench, rows = reference_load(case, x, v, q, speed)
    y, velocity, lam = material_step(case, x, v, force, case["h"])
    next_q, next_speed = rigid_step(case, q, speed, wrench, case["h"], external)
    return (y, velocity, next_q, next_speed), dict(
        force=force, wrench=wrench, rows=rows, multipliers=lam
    )
