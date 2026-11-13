"""Standalone M-B2 articulated/deformable public declaration consumer.

No source examples/tests or product geometry/mechanics helpers are imported.
The exact active-face oracle and FP64 ordered material recurrence below are
self-contained independent calculations, not device-state replacements.
"""

import argparse
import copy
import gzip
import hashlib
import json
import math
import sys
import traceback
from collections.abc import Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import fields, is_dataclass, replace
from fractions import Fraction as Q
from itertools import combinations, product
from pathlib import Path
from unittest.mock import patch

import numpy as np

H = 1 / 1024
WIDTH, STIFFNESS, DAMPING, RADIUS = 1 / 16, 128.0, 1 / 2, 1 / 8
ATOL, RTOL = 5e-6, 5e-5
IDS = np.array((101, 113, 127, 139), np.int64)
FACES = np.array(((0, 1, 2), (0, 2, 3)), np.int64)
FACETS = np.array((503, 509), np.int64)
EDGES = np.array(((0, 1), (1, 2), (2, 3), (3, 0), (0, 2)), np.int64)
REST = np.array((1, 1, 1, 1, math.sqrt(2)))
COMPLIANCE = 1 / 16384


def require(value, message):
    if not value:
        raise AssertionError(message)


def encode(value):
    if isinstance(value, BaseException):
        return dict(exception_type=type(value).__name__, message=str(value))
    if isinstance(value, np.ndarray):
        return dict(dtype=value.dtype.str, shape=value.shape, bytes_hex=value.tobytes().hex())
    if isinstance(value, np.generic):
        return encode(value.item())
    if isinstance(value, bytes):
        return dict(bytes_hex=value.hex())
    if type(value) is object:
        return dict(opaque_identity=id(value))
    if is_dataclass(value):
        return {f.name: encode(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Mapping):
        return dict(mapping_entries=[[encode(k), encode(v)] for k, v in value.items()])
    if isinstance(value, (tuple, list)):
        return [encode(v) for v in value]
    if type(value) is float and not math.isfinite(value):
        return dict(nonfinite=repr(value))
    if value is None or type(value) in (str, bool, int, float):
        return value
    raise TypeError("unrecorded D3 evidence " + type(value).__name__)


class Journal:
    def __init__(self, path):
        self.path, self.stream = path, gzip.open(path, "xb")
        self.digest, self.bytes, self.records = hashlib.sha256(), 0, 0

    def emit(self, event, **payload):
        raw = (
            json.dumps(
                dict(event=event, payload=encode(payload)), allow_nan=False, separators=(",", ":")
            )
            + "\n"
        ).encode()
        self.stream.write(raw)
        self.digest.update(raw)
        self.bytes += len(raw)
        self.records += 1

    def close(self):
        self.stream.close()
        return dict(
            logical_sha256=self.digest.hexdigest(),
            logical_bytes=self.bytes,
            records=self.records,
            compressed_bytes=self.path.stat().st_size,
        )


def provenance(root, *, preflight=False):
    def checkout(path):
        return ((path / "pyproject.toml").is_file() and (path / "src/phymeo").is_dir()) or (
            path.name.lower() == "src"
            and (path / "phymeo").is_dir()
            and (path.parent / "pyproject.toml").is_file()
        )

    require(
        not any(checkout(Path(p or Path.cwd()).resolve()) for p in sys.path),
        "checkout source entry in sys.path",
    )
    if preflight:
        require(
            not any(n == "phymeo" or n.startswith("phymeo.") for n in sys.modules),
            "product imported before installed preflight",
        )
        for origin in (Path.cwd().resolve(), Path(__file__).resolve().parent, root):
            require(
                not any(checkout(p) for p in (origin, *origin.parents)),
                "consumer, cwd and install must be outside checkouts",
            )
    import phymeo

    require(Path(phymeo.__file__).resolve().is_relative_to(root), "wrong installed package")
    result = {}
    for name, module in sorted(sys.modules.items()):
        if name == "phymeo" or name.startswith("phymeo."):
            location = getattr(module, "__file__", None)
            if location is not None:
                path = Path(location).resolve()
                require(path.is_relative_to(root), "mixed checkout module " + name)
                result[name] = dict(
                    path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()
                )
            for directory in getattr(module, "__path__", ()):
                require(
                    Path(directory).resolve().is_relative_to(root), "mixed package path " + name
                )
    return result


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


def close(actual, expected, *, atol=ATOL, rtol=RTOL):
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)



def skew(x):
    a, b, c = np.asarray(x, float)
    return np.array([[0.0, -c, b], [c, 0.0, -a], [-b, a, 0.0]])


def vee(x):
    return np.array([x[2, 1] - x[1, 2], x[0, 2] - x[2, 0], x[1, 0] - x[0, 1]]) / 2


def quat_matrix(q):
    x, y, z, w = np.asarray(q, float) / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def quat_product(a, b):
    av, bv = np.asarray(a[:3]), np.asarray(b[:3])
    return np.r_[a[3] * bv + b[3] * av + np.cross(av, bv), a[3] * b[3] - av @ bv]


def axis_rotation(axis, angle):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    K = skew(axis)
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def transform(rotation, position):
    result = np.eye(4)
    result[:3, :3], result[:3, 3] = rotation, position
    return result


def multiply(a, b):
    # These are derivatives, not Taylor coefficients.
    return a[0] @ b[0], a[1] @ b[0] + a[0] @ b[1], a[2] @ b[0] + 2 * a[1] @ b[1] + a[0] @ b[2]


def jets(data, q, v):
    q, v = np.asarray(q, float), np.asarray(v, float)
    root = quat_matrix(q[3:7])
    spin = skew(v[3:6])
    first, second = spin @ root, spin @ spin @ root
    T = transform(root, q[:3])
    D, DD = np.zeros((4, 4)), np.zeros((4, 4))
    D[:3, :3], DD[:3, :3] = first, second
    # The public root velocity is COM/world, not authored-origin velocity.
    D[:3, 3] = v[:3] - first @ data["root_com"]
    DD[:3, 3] = -second @ data["root_com"]
    result = [(T, D, DD)]
    children = data["children"]
    for i, parent in enumerate(children["parents"]):
        fixed = (
            transform(quat_matrix(children["orientations"][i]), children["offsets"][i]),
            np.zeros((4, 4)),
            np.zeros((4, 4)),
        )
        axis = children["axes"][i] / np.linalg.norm(children["axes"][i])
        rate = v[6 + i]
        local, local_d, local_dd = np.eye(4), np.zeros((4, 4)), np.zeros((4, 4))
        if children["kinds"][i] == "prismatic":
            local[:3, 3], local_d[:3, 3] = q[7 + i] * axis, rate * axis
        else:
            R = axis_rotation(axis, q[7 + i])
            K = skew(axis)
            local[:3, :3], local_d[:3, :3], local_dd[:3, :3] = (
                R,
                rate * K @ R,
                rate * rate * K @ K @ R,
            )
        result.append(multiply(multiply(result[parent + 1], fixed), (local, local_d, local_dd)))
    com_local = np.vstack((data["root_com"], children["com"]))
    positions, rotations, centers, velocity, acceleration, omega, alpha = [], [], [], [], [], [], []
    for (T, D, DD), c in zip(result, com_local, strict=True):
        point = np.r_[c, 1.0]
        positions.append(T[:3, 3])
        rotations.append(T[:3, :3])
        centers.append((T @ point)[:3])
        velocity.append((D @ point)[:3])
        acceleration.append((DD @ point)[:3])
        omega.append(vee(D[:3, :3] @ T[:3, :3].T))
        alpha.append(vee(DD[:3, :3] @ T[:3, :3].T + D[:3, :3] @ D[:3, :3].T))
    return dict(
        position=np.array(positions),
        rotation=np.array(rotations),
        com_position=np.array(centers),
        com_velocity=np.array(velocity),
        com_acceleration=np.array(acceleration),
        angular_velocity=np.array(omega),
        angular_acceleration=np.array(alpha),
    )


def mechanics(data, q, v):
    d = len(v)
    observed = jets(data, q, v)
    # Every column is an exact unit-velocity transform derivative, never FD.
    basis = [jets(data, q, column) for column in np.eye(d)]
    Jv = np.stack([item["com_velocity"] for item in basis], axis=2)
    Jw = np.stack([item["angular_velocity"] for item in basis], axis=2)
    masses = np.r_[data["root_mass"], data["children"]["mass"]]
    principal = np.vstack((data["root_inertia"], data["children"]["inertia"]))
    frames = observed["rotation"].copy()
    frames[0] = frames[0] @ quat_matrix(data["root_inertia_frame"])
    inertia = np.array(
        [
            R @ np.diag(principal_values) @ R.T
            for R, principal_values in zip(frames, principal, strict=True)
        ]
    )
    M, bias = np.zeros((d, d)), np.zeros(d)
    energy, momentum, angular = 0.0, np.zeros(3), np.zeros(3)
    for i, mass in enumerate(masses):
        spin = observed["angular_velocity"][i]
        F = mass * (observed["com_acceleration"][i] - data["gravity"])
        torque = inertia[i] @ observed["angular_acceleration"][i] + np.cross(
            spin, inertia[i] @ spin
        )
        M += mass * (Jv[i].T @ Jv[i]) + Jw[i].T @ inertia[i] @ Jw[i]
        bias += Jv[i].T @ F + Jw[i].T @ torque
        linear_momentum = mass * observed["com_velocity"][i]
        momentum += linear_momentum
        angular += np.cross(observed["com_position"][i], linear_momentum) + inertia[i] @ spin
        energy += (observed["com_velocity"][i] @ linear_momentum + spin @ inertia[i] @ spin) / 2
    return dict(
        **observed,
        mass=M,
        bias=bias,
        linear_jacobian=Jv,
        angular_jacobian=Jw,
        inertia_world=inertia,
        energy=energy,
        momentum=momentum,
        angular_momentum=angular,
    )


def external(data, q):
    effort, force, torque = (data[key].copy() for key in ("effort", "force", "torque"))
    if data.get("force_point") is not None:
        # A declared local application point and held world force; no product FK.
        arm = quat_matrix(q[3:7]) @ (data["force_point"] - data["root_com"])
        torque[0] = np.cross(arm, force[0])
    return effort, force, torque


def applied(terms, effort, force, torque):
    result = np.einsum("bci,bc->i", terms["linear_jacobian"], force)
    result += np.einsum("bci,bc->i", terms["angular_jacobian"], torque)
    result[6:] += effort
    return result


def bounded_drive(M, bias, load, q, v, drive, kp, kd, caps, h):
    n = len(kp)
    if not n:
        return np.zeros(0)
    free = np.linalg.solve(M, load - bias)[6:]
    mobility = np.linalg.inv(M)[6:, 6:]
    beta = h * h * kp + h * kd
    rhs = (
        drive[:, 2]
        + kp * (drive[:, 0] - q[7:] - h * v[6:])
        + kd * (drive[:, 1] - v[6:])
        - beta * free
    )
    fixed, moving = np.flatnonzero(beta == 0), np.flatnonzero(beta > 0)
    tau = np.clip(rhs, -caps, caps)
    if not len(moving):
        return tau
    if len(moving) > 4:
        raise ValueError("independent exhaustive PD reference limited to four nonzero-gain axes")
    A = mobility[np.ix_(moving, moving)] + np.diag(1 / beta[moving])
    b = rhs[moving] / beta[moving] - mobility[np.ix_(moving, fixed)] @ tau[fixed]
    # Exhaust every box face, solve its free block, check the full global KKT.
    for signs in product((-1, 0, 1), repeat=len(moving)):
        x = np.array(signs, float) * caps[moving]
        free_ids = np.flatnonzero(np.array(signs) == 0)
        pinned = np.flatnonzero(np.array(signs) != 0)
        if len(free_ids):
            x[free_ids] = np.linalg.solve(
                A[np.ix_(free_ids, free_ids)], b[free_ids] - A[np.ix_(free_ids, pinned)] @ x[pinned]
            )
        if np.any(np.abs(x) > caps[moving] + 1e-11):
            continue
        gradient = A @ x - b
        if any(
            (s == 0 and abs(g) > 1e-10) or (s < 0 and g < -1e-10) or (s > 0 and g > 1e-10)
            for s, g in zip(signs, gradient, strict=True)
        ):
            continue
        tau[moving] = x
        return tau
    raise AssertionError("no independently feasible PD active face")


def drift(data, q, velocity, h):
    q, velocity = np.asarray(q, float), np.asarray(velocity, float)
    rotation_vector = h * velocity[3:6]
    theta = np.linalg.norm(rotation_vector)
    inc = (
        np.r_[np.zeros(3), 1.0]
        if theta == 0
        else np.r_[rotation_vector * np.sin(theta / 2) / theta, np.cos(theta / 2)]
    )
    orientation = quat_product(inc, q[3:7])
    orientation /= np.linalg.norm(orientation)
    if orientation @ q[3:7] < 0:
        orientation *= -1
    origin = (
        q[:3]
        + h * velocity[:3]
        - (quat_matrix(orientation) - quat_matrix(q[3:7])) @ data["root_com"]
    )
    return np.r_[origin, orientation, q[7:] + h * velocity[6:]]


def tree_advance(data, q, v, drive, external_packet, h, *, kp=None, kd=None, caps=None):
    n = len(q) - 7
    kp = np.zeros(n) if kp is None else np.asarray(kp)
    kd = np.zeros(n) if kd is None else np.asarray(kd)
    caps = np.full(n, 64.0) if caps is None else np.asarray(caps)
    t = mechanics(data, q, v)
    load = applied(t, *external_packet)
    motor = bounded_drive(t["mass"], t["bias"], load, q, v, drive, kp, kd, caps, h)
    total = load.copy()
    total[6:] += motor
    a = np.linalg.solve(t["mass"], total - t["bias"])
    next_v = np.asarray(v) + h * a
    next_q = drift(data, q, next_v, h)
    return next_q, next_v, dict(**t, external=load, drive=motor, acceleration=a)



def material_step(item, x, v, force, h, sweeps=None):
    weights = item["node"].model.inverse_mass.astype(float)
    pv = v + h * weights[:, None] * force
    px = x + h * pv
    fixed = weights == 0
    px[fixed], pv[fixed] = x[fixed], 0
    y, lam = px.copy(), np.zeros(5)
    alpha = COMPLIANCE / h / h
    for _ in range(SETTINGS.position_iterations if sweeps is None else sweeps):
        for e, (a, b) in enumerate(EDGES):
            diff = y[a] - y[b]
            length = np.linalg.norm(diff)
            direction = diff / length
            dl = -(length - REST[e] + alpha * lam[e]) / (weights[a] + weights[b] + alpha)
            lam[e] += dl
            y[a] += weights[a] * dl * direction
            y[b] -= weights[b] * dl * direction
        if sweeps is None:
            lengths = np.linalg.norm(y[EDGES[:, 0]] - y[EDGES[:, 1]], axis=1)
            residual = np.max(
                abs(lengths - REST + alpha * lam) / np.maximum(REST, SETTINGS.reference_length)
            )
            if residual <= SETTINGS.tolerance:
                break
    else:
        if sweeps is None:
            raise AssertionError("independent material reference exhausted its fixed budget")
    speed = pv + (y - px) / h
    speed[fixed] = 0
    return y, speed, lam



def initialize():
    global ph, SETTINGS
    import phymeo as product

    ph = product
    SETTINGS = ph.StepSettings(substeps=1, position_iterations=200, tolerance=2e-6)


@contextmanager
def saved(path):
    journal = Journal(path)
    primary = None
    try:
        yield journal
    except BaseException as error:
        primary = error
        try:
            journal.emit("failure", error=error, traceback=traceback.format_exc())
        except BaseException as secondary:
            raise BaseExceptionGroup("M-B2 original and journal failure", [error, secondary])
        raise
    finally:
        try:
            journal.close()
        except BaseException as error:
            if primary is not None:
                raise BaseExceptionGroup("M-B2 original and close failure", [primary, error])
            raise



def _legacy_fixture(kind="A", *, world=7, extra=False):
    """Fixed before execution; anchors are original absolute source coordinates."""
    from phymeo.builders._robot_asset import LinkCollision

    require(kind in ("A", "B") and world in (7, 31), "fixed M-B1 source identity")
    origin = np.array((8, -4, 2) if world == 7 else (-3, 5, -1), float)
    x = np.array(((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)), np.float32) + origin
    weights = np.array((2, 0, 2, 2), np.float32)
    nodes = ph.DeformableNodeWorld(
        world,
        ph.ParticleModel(weights, np.zeros(3, np.float32)),
        ph.ParticleState(x.astype(np.float32), np.zeros((4, 3), np.float32)),
        ph.DistanceConstraints(EDGES, REST, np.full(5, COMPLIANCE)),
        IDS,
        (ph.DeformableSurface(23, FACETS, IDS[FACES]),),
    )
    surface = ph.CollisionKey(world, 101, 0)
    bodies, materials = [], {}
    for ordinal, (participant_id, body_id, mass) in enumerate(((19, 202, 2.0), (37, 303, 1.0))):
        key = ph.ResidentParticipantKey(world, participant_id)
        center = np.array(
            (0.5, 0.125, RADIUS + WIDTH / 2) if ordinal == 0 else (0.25, 0.875, RADIUS + WIDTH / 2),
            float,
        )
        velocity = np.array((0, 0, -1 / 16 if ordinal == 0 else -1 / 32, 0, 0, 0), float)
        external = np.array((0, 1 / 32, 0, 0, 0, 1 / 64), float)
        local = np.array(((0.25, 0, 0),) if ordinal == 0 else ((0, 0, 0),), float)
        shape_kind = "sphere"
        if kind == "B":
            if ordinal == 1:
                shape_kind = "capsule"
                local = np.array(((0, -1 / 16, 0), (1 / 16, 1 / 16, 1 / 8)), float)
                velocity[2] = -1 / 16
            if (world == 7 and ordinal == 0) or (world == 31 and ordinal == 1):
                center[2] = RADIUS + WIDTH + 1 / 16384
                velocity[2] = -1 / 8
                # Pre-execution hand proof: both entry/exit bodies use 32 m/s^2.
                # The mass-1 proposal +64 N only touched width at these h/v/delta;
                # it was never executed. Freeze +32 N here; no output-driven tuning.
                external[2] = 64 if mass == 2 else 32
            elif world == 31 and ordinal == 0:
                center[2] = RADIUS + WIDTH + 1 / 32
                velocity[2] = -1 / 16
        center += origin
        inertia = mass / 8
        model = ph.FloatingRobotModel(
            ph.RigidBodyModel(np.array((1 / mass,)), np.full((1, 3), inertia), np.zeros(3)),
            np.zeros(3),
        )
        initial = ph.FloatingRobotState(np.r_[center, (0, 0, 0, 1)], velocity)
        collider_specs = [(shape_kind, local)]
        if extra and ordinal == 0:
            collider_specs.append(("sphere", np.array(((0.125, 0.125, 0),), float)))
        collisions, original_shapes, mapping, collider_rows = [], [], {}, []
        for shape_index, (primitive, vertices) in enumerate(collider_specs):
            source = ph.CollisionKey(0, 42, 71 + shape_index)
            installed = ph.CollisionKey(world, body_id, shape_index)
            shape = ph.CollisionShape(
                source, primitive, tuple(range(41, 41 + len(vertices))), RADIUS, revision=3
            )
            collisions.append(
                LinkCollision(
                    f"original-{kind}-{ordinal}-{shape_index}",
                    "root",
                    -1,
                    (0.0, 0.0, 0.0),
                    (0.0, 0.0, 0.0, 1.0),
                    shape,
                    tuple(tuple(float(t) for t in row) for row in vertices),
                    True,
                )
            )
            original_shapes.append(shape)
            mapping[source] = installed
            collider_rows.append(
                dict(local=vertices, source=source, installed=installed, kind=primitive)
            )
            materials[(surface, installed)] = ph.NormalLayer(WIDTH, STIFFNESS, DAMPING)
        asset = ph.FloatingRobotAsset(
            f"m-b1-{kind}-{ordinal}",
            "root",
            model,
            (),
            (),
            (),
            (),
            (),
            (),
            tuple(collisions),
            ph.CollisionModel(tuple(original_shapes)),
        )
        command = ph.FloatingRobotCommand(
            None, ph.FloatingRobotInputs(np.zeros(0), external[None, :3], external[None, 3:])
        )
        row = ph.ParticipantArticulation(
            key,
            asset,
            initial,
            kp=np.zeros(0),
            kd=np.zeros(0),
            effort_limit=np.zeros(0),
            body_ids=(body_id,),
            collider_keys=mapping,
            initial_control=command,
        )
        bodies.append(
            dict(
                key=key,
                body_id=body_id,
                row=row,
                model=model,
                initial=initial,
                mass=mass,
                inertia=inertia,
                external=external,
                command=command,
                colliders=tuple(collider_rows),
            )
        )
    return dict(
        kind=kind,
        world=world,
        extra=extra,
        origin=origin,
        node=nodes,
        surface=surface,
        bodies=tuple(bodies),
        materials=materials,
    )



def declaration(items, *, reverse=False, site_capacity=None, memory_limit_bytes=256 * 1024 * 1024):
    rows = [body["row"] for item in items for body in item["bodies"]]
    if reverse:
        rows = [
            replace(row, collider_keys=dict(reversed(tuple(row.collider_keys.items()))))
            for row in reversed(rows)
        ]
    nodes = tuple(i["node"] for i in items)
    surfaces = tuple(ph.DeformableBoundarySurface(i["surface"], 23) for i in items)
    if reverse:
        nodes, surfaces = tuple(reversed(nodes)), tuple(reversed(surfaces))
    return ph.ParticipantSceneDeclaration(
        tuple(rows),
        contact_response="deformable_normal",
        deformable=ph.ParticipantDeformable(
            nodes, surfaces, settings=SETTINGS, memory_limit_bytes=memory_limit_bytes
        ),
        materials_by_pair={key: law for item in items for key, law in item["materials"].items()},
        site_capacity=(
            sum(len(b["colliders"]) * 2 for i in items for b in i["bodies"])
            if site_capacity is None
            else site_capacity
        ),
    )


def compile_scene(items, mode, **options):
    return ph.compile_participant_scene(declaration(items, **options), mode=mode)


_OWNER_CLOSE_RECORDS = []
_RETAINED_CLOSE_FAILURES = []


@contextmanager
def owned_scene(items, mode, journal, label):
    """Close each constructed owner once; retain uncertain owners, not fake cleanup."""
    scene = compile_scene(items, mode)
    record = dict(
        label=label, owner_identity=id(scene), close_attempts=0, close_succeeded=False, state="OPEN"
    )
    _OWNER_CLOSE_RECORDS.append(record)
    primary, errors = None, []
    try:
        journal.emit("owner-constructed", owner=record)
        yield scene
    except BaseException as error:
        primary = error
    try:
        journal.emit("owner-close-attempt", owner=record, primary=primary)
    except BaseException as error:
        errors.append(error)
    record["close_attempts"] = 1
    try:
        scene.close()
    except BaseException as error:
        record.update(state="UNKNOWN", close_error=repr(error))
        # Do not retry an opaque release or let its owner/runtime disappear.
        # This process-lifetime root deliberately outlives the context manager.
        _RETAINED_CLOSE_FAILURES.append(scene)
        errors.append(error)
    else:
        record.update(state="CLOSED", close_succeeded=True)
    try:
        journal.emit("owner-close-result", owner=record)
    except BaseException as error:
        errors.append(error)
    failures = ([primary] if primary is not None else []) + errors
    if len(failures) == 1:
        raise failures[0]
    if failures:
        raise BaseExceptionGroup("M-B2 primary and owner cleanup failures", failures)



def forces(items):
    return {i["world"]: ph.StepInputs(np.zeros((4, 3), np.float32)) for i in items}


def roots(state, item):
    return {b["key"]: state.rigid.participants[b["key"]] for b in item["bodies"]}


def reference_initial(items):
    return {
        i["world"]: (
            i["node"].initial_state.position.astype(float).copy(),
            i["node"].initial_state.velocity.astype(float).copy(),
            {
                b["key"]: (np.asarray(b["initial"].q).copy(), np.asarray(b["initial"].v).copy())
                for b in i["bodies"]
            },
        )
        for i in items
    }



def state_equal(left, right, items):
    for item in items:
        a, b = left[item["world"]], right[item["world"]]
        for name in ("position", "velocity", "clock"):
            x, y = getattr(a.node, name), getattr(b.node, name)
            require(
                x.dtype == y.dtype and x.shape == y.shape and x.tobytes() == y.tobytes(),
                "node raw " + name,
            )
        require(
            encode(a.node.last_step) == encode(b.node.last_step), "complete node material history"
        )
        require(
            a.node.tick == b.node.tick
            and a.node.time == b.node.time
            and a.rigid.tick == b.rigid.tick,
            "common raw clock and tick",
        )
        require(set(a.rigid.participants) == set(b.rigid.participants), "complete participant set")
        for body in item["bodies"]:
            require(
                a.rigid.participants[body["key"]] == b.rigid.participants[body["key"]],
                "complete participant q/v/control/generalized/history/raw-root",
            )


def check_sources(scene, items):
    for item in items:
        surface = scene.source_map["surfaces"][item["surface"]]
        np.testing.assert_array_equal(surface["facet_ids"], FACETS)
        np.testing.assert_array_equal(surface["oriented_point_ids"], IDS[FACES])
        for body in item["bodies"]:
            for collider in body["colliders"]:
                require(
                    scene.source_map["participants"][body["key"]][collider["source"]]
                    == collider["installed"],
                    "original asset to actual body installation",
                )



def query(scene, journal):
    revision = scene.revision
    journal.emit("query-attempt", expected_revision=revision, state=scene.snapshot())
    result = scene.query(expected_revision=revision)
    journal.emit("query-return", revision=revision, result=result)
    require(
        scene.require_query_current(result, expected_revision=revision) is result, "current query"
    )
    return result


# Independent tree transforms/J/M/PD/drift copied as local code from the accepted
# resident_floating_reference.py at base 9c7433d7; geometry/material/serialization
# from check_deformable_multibody.py at that same base. No runtime helper imports.

def command(body, phase=0):
    n = len(body["data"]["children"]["parents"])
    if not n:
        return body["command"]
    sign = 1 if phase == 0 else -1
    drive = sign * np.column_stack((
        np.array((1/32, -1/64, 1/64))[:n],
        np.array((1/64, -1/64, 1/128))[:n],
        np.array((1/16, -1/32, 1/64))[:n],
    ))
    return ph.FloatingRobotCommand(
        ph.RobotCommand(drive[:, 0], target_velocity=drive[:, 1], feedforward=drive[:, 2]),
        ph.FloatingRobotInputs(np.zeros(n), body["external"][:, :3], body["external"][:, 3:]),
    )


def drive_rows(body, phase=0):
    n = len(body["data"]["children"]["parents"])
    if not n:
        return np.zeros((0, 3))
    return (1 if phase == 0 else -1) * np.column_stack((
        np.array((1/32, -1/64, 1/64))[:n], np.array((1/64, -1/64, 1/128))[:n],
        np.array((1/16, -1/32, 1/64))[:n],
    ))


def fixture(kind="C", *, world=7):
    """The b2752ddc inputs, never selected from an execution result."""
    from phymeo.builders._robot_asset import LinkCollision

    require(kind in ("C", "D") and world in (7, 31), "fixed M-B2 identity")
    origin = np.array((8, -4, 2) if world == 7 else (-3, 5, -1), float)
    x = np.array(((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)), float) + origin
    node = ph.DeformableNodeWorld(
        world, ph.ParticleModel(np.array((2, 0, 2, 2), np.float32), np.zeros(3, np.float32)),
        ph.ParticleState(x.astype(np.float32), np.zeros((4, 3), np.float32)),
        ph.DistanceConstraints(EDGES, REST, np.full(5, COMPLIANCE)), IDS,
        (ph.DeformableSurface(23, FACETS, IDS[FACES]),),
    )
    n = 2 if kind == "C" else 3
    children = dict(
        parents=np.array((-1, 0) if kind == "C" else (-1, -1, -1), np.int64),
        kinds=("revolute", "prismatic") if kind == "C" else ("revolute", "prismatic", "revolute"),
        offsets=np.array(((.25, 0, 0), (.25, 0, 0)) if kind == "C" else
                         ((.25, -.375, 0), (-.25, .375, 0), (.125, .125, .5)), float),
        orientations=np.tile((0., 0., 0., 1.), (n, 1)),
        axes=np.array(((0, 1, 0), (0, 0, 1), (1, 0, 0)), float)[:n],
        mass=np.array((1., .75, .5))[:n],
        com=np.array(((1/32, -1/64, 1/64), (-1/64, 1/32, -1/64), (1/64, -1/64, 1/32)))[:n],
        inertia=np.array(((.25, .375, .5), (.125, .1875, .25), (.125, .1875, .25)))[:n],
    )
    data = dict(root_mass=2., root_com=np.array((1/32, -1/64, 1/128)),
                root_inertia=np.array((.5, .625, .75)), root_inertia_frame=np.full(4, .5),
                gravity=np.zeros(3), children=children)
    model = ph.FloatingRobotModel(
        ph.RigidBodyModel(np.array((.5,)), data["root_inertia"][None], np.zeros(3)),
        data["root_com"],
        ph.RobotModel(children["parents"], children["kinds"], children["offsets"],
                      children["orientations"], children["axes"], children["mass"],
                      children["com"], children["inertia"], np.zeros(3)),
        root_inertia_frame_xyzw=data["root_inertia_frame"],
    )
    center = origin + np.array((.25, .125, 5/32) if kind == "C" else (.5, .5, 5/32))
    initial = ph.FloatingRobotState(
        np.r_[center, (0., 0., 0., 1.), np.zeros(n)],
        np.r_[(1/128, -1/256, -1/16, 1/128, -1/256, 1/256), np.array((1/32, -1/64, 1/64))[:n]],
    )
    body_ids = (202, 211, 223) if kind == "C" else (202, 211, 223, 239)
    names = tuple("link" + str(i + 1) for i in range(n))
    collider_specs = (
        ((2, "capsule", ((0, 0, 0), (1/16, 1/16, 1/8))),) if kind == "C" else
        ((1, "sphere", ((0, 0, 0),)), (2, "capsule", ((0, 0, 0), (1/16, 0, 1/8))))
    )
    surface = ph.CollisionKey(world, 101, 0)
    collisions, original, mapping, colliders, materials = [], [], {}, [], {}
    for ordinal, (group, primitive, vertices) in enumerate(collider_specs):
        source = ph.CollisionKey(0, 42 + group, 71 + ordinal)
        installed = ph.CollisionKey(world, body_ids[group], ordinal)
        shape = ph.CollisionShape(source, primitive, tuple(range(41, 41 + len(vertices))), RADIUS, revision=3)
        collisions.append(LinkCollision("source-" + str(ordinal), names[group - 1], group - 1,
            (0., 0., 0.), (0., 0., 0., 1.), shape, vertices, True))
        original.append(shape)
        mapping[source] = installed
        colliders.append(dict(source=source, installed=installed, group=group, kind=primitive,
                              local=np.array(vertices, float)))
        materials[surface, installed] = ph.NormalLayer(WIDTH, STIFFNESS, DAMPING)
    asset = ph.FloatingRobotAsset(
        "m-b2-" + kind, "root", model, names, tuple("joint" + str(i) for i in range(n)),
        tuple([-2.] * n), tuple([2.] * n), tuple([4.] * n), tuple([4.] * n),
        tuple(collisions), ph.CollisionModel(tuple(original)),
    )
    external_wrench = np.zeros((n + 1, 6))
    external_wrench[0] = (1/64, -1/128, 0, 0, 0, 1/128)
    external_wrench[2] = (0, 1/64, 0, 1/128, 0, 0)
    body = dict(key=ph.ResidentParticipantKey(world, 19), body_ids=body_ids, model=model,
                initial=initial, data=data, external=external_wrench, colliders=tuple(colliders),
                kp=np.full(n, .25), kd=np.full(n, .125), caps=np.full(n, 4.))
    body["command"] = command(body)
    body["row"] = ph.ParticipantArticulation(
        body["key"], asset, initial, kp=body["kp"], kd=body["kd"], effort_limit=body["caps"],
        body_ids=body_ids, collider_keys=mapping, initial_control=body["command"],
    )
    return dict(kind=kind, world=world, origin=origin, node=node, surface=surface,
                bodies=(body,), materials=materials)


def legacy_fixture(*, world=7):
    """Original qualified A geometry/control, only independent reference metadata added."""
    item = _legacy_fixture("A", world=world)
    for body in item["bodies"]:
        body["body_ids"] = (body["body_id"],)
        body["external"] = body["external"][None, :]
        body["data"] = dict(root_mass=body["mass"], root_com=np.zeros(3),
            root_inertia=np.full(3, body["inertia"]), root_inertia_frame=np.array((0., 0., 0., 1.)),
            gravity=np.zeros(3), children=dict(parents=np.zeros(0, np.int64), kinds=(),
            offsets=np.zeros((0, 3)), orientations=np.zeros((0, 4)), axes=np.zeros((0, 3)),
            mass=np.zeros(0), com=np.zeros((0, 3)), inertia=np.zeros((0, 3))))
        body["kp"], body["kd"], body["caps"] = np.zeros(0), np.zeros(0), np.zeros(0)
        for collider in body["colliders"]:
            collider["group"] = 0
    return item


def participants(items):
    return tuple(ph.ResidentParticipant(b["key"], ph.FloatingRobotTree(
        b["model"], b["initial"], kp=b["kp"], kd=b["kd"], effort_limit=b["caps"]), b["body_ids"])
        for item in items for b in item["bodies"])


def reference_world(item, x, velocity, body_states):
    total, wrenches, records = np.zeros((4, 3)), {}, {}
    for body in item["bodies"]:
        q, speed = body_states[body["key"]]
        kin = jets(body["data"], q, speed)
        wrenches[body["key"]] = np.zeros((len(body["body_ids"]), 6))
        for collider in body["colliders"]:
            group = collider["group"]
            axis = collider["local"] @ kin["rotation"][group].T + kin["position"][group]
            segment = np.vstack((axis, axis)) if len(axis) == 1 else axis
            active_law = item["materials"][item["surface"], collider["installed"]] is not None
            for ordinal, indices in enumerate(FACES):
                exact = oracle(x[indices], segment)
                d2, s, bary, _, _ = exact["witnesses"][0]
                distance, s = math.sqrt(float(d2)), float(s)
                require(distance > 0, "independent undefined normal")
                beta = np.array([float(value) for value in bary])
                axis_point = (1-s) * segment[0] + s * segment[1]
                node_point = beta @ x[indices]
                normal = (node_point - axis_point) / distance
                point = axis_point + RADIUS * normal
                com = kin["com_position"][group]
                relative = beta @ velocity[indices] - kin["com_velocity"][group] - np.cross(
                    kin["angular_velocity"][group], point - com)
                rate = float(normal @ relative)
                gap = distance - RADIUS
                compression = max(WIDTH - gap, 0) if active_law else 0.
                force = STIFFNESS * compression * max(1-DAMPING*rate, 0) * normal
                total[indices] += beta[:, None] * force
                wrenches[body["key"]][group, :3] -= force
                wrenches[body["key"]][group, 3:] += np.cross(point-com, -force)
                records[collider["installed"], ordinal] = dict(
                    ordinal=ordinal, distance=distance, gap=gap, bary=beta, s=s, normal=normal,
                    node_point=node_point, rigid_point=point, relative=relative, force=force,
                    compression=compression, rate=rate, active_law=active_law,
                )
    return total, wrenches, records


def reference_step(body, q, v, contact, phase):
    packet = contact + body["external"]
    return tree_advance(body["data"], q, v, drive_rows(body, phase),
        (np.zeros(len(body["kp"])), packet[:, :3], packet[:, 3:]), H,
        kp=body["kp"], kd=body["kd"], caps=body["caps"])


def quaternion_close(actual, expected, *, atol=ATOL+RTOL):
    actual, expected = np.asarray(actual), np.asarray(expected)
    require(min(np.linalg.norm(actual-expected), np.linalg.norm(actual+expected)) <= atol,
            "independent quaternion")


def geometry_close(actual, terms):
    for field, independent in (("link_position_m", "position"), ("link_rotation", "rotation"),
            ("com_position_m", "com_position"), ("linear_velocity_m_s", "com_velocity"),
            ("angular_velocity_rad_s", "angular_velocity")):
        close(getattr(actual, field), terms[independent])
def audit_step(item, entry, after, receipt, phase=0):
    observed = receipt.worlds[item["world"]]
    row = receipt.node_receipt.worlds[item["world"]]
    x0, v0 = entry.node.position, entry.node.velocity
    old, new = roots(entry, item), roots(after, item)
    bodies = {b["key"]: b for b in item["bodies"]}
    colliders = {c["installed"]: (b, c) for b in item["bodies"] for c in b["colliders"]}
    body_states = {key: (np.asarray(s.position), np.asarray(s.velocity)) for key, s in old.items()}
    exact_force, exact_wrenches, expected = reference_world(item, x0.astype(float), v0.astype(float), body_states)
    k0 = {key: mechanics(b["data"], *body_states[key]) for key, b in bodies.items()}
    k1 = {key: mechanics(b["data"], np.asarray(new[key].position), np.asarray(new[key].velocity))
          for key, b in bodies.items()}
    require(receipt.schema == "tyx-deformable-articulated-contact-step-v1", "explicit articulated schema")
    require(set(observed["rigid_participants"]) == set(bodies), "complete participant publication")
    require(after.node.tick == entry.node.tick + 1 == after.rigid.tick, "one common tick")
    np.testing.assert_array_equal(row.input_position, x0)
    np.testing.assert_array_equal(row.input_velocity, v0)
    require(row.h == H and len(row.trace) == row.sweeps and row.residual <= SETTINGS.tolerance,
            "complete original material trace")
    total = np.zeros((4, 3))
    wrenches = {key: np.zeros((len(b["body_ids"]), 6)) for key, b in bodies.items()}
    seen = set()
    work = potential0 = potential1 = dissipation = power = 0.
    for site in observed["sites"]:
        require(site["surface"] == item["surface"] and site["round"] in colliders, "material owner")
        body, collider = colliders[site["round"]]
        key, group = body["key"], collider["group"]
        require(site["participant"] == key and site["body_group"] == group, "actual nonroot owner")
        identity = site["round"], int(site["facet_ordinal"])
        require(identity not in seen, "complete unique material address")
        seen.add(identity)
        target = expected[identity]
        indices = FACES[identity[1]]
        require(site["facet_id"] == FACETS[identity[1]], "stable facet identity")
        np.testing.assert_array_equal(site["point_ids"], IDS[indices])
        np.testing.assert_array_equal(site["node_indices"], indices)
        beta, normal = np.asarray(site["barycentric"]), np.asarray(site["normal"])
        require(beta.shape == (3,) and min(beta) >= 0, "nonnegative barycentric")
        close(beta.sum(), 1., atol=2e-6, rtol=0)
        kin, next_kin = k0[key], k1[key]
        local = collider["local"]
        segment = np.vstack((local, local)) if len(local) == 1 else local
        length = local_scale(x0[indices], segment, RADIUS)
        close(site["gap"], target["gap"], atol=1e-4*length, rtol=0)
        close(normal, target["normal"], atol=2e-3, rtol=0)
        close(site["node_point"], target["node_point"], atol=2e-4*length, rtol=0)
        close(site["rigid_point"], target["rigid_point"], atol=2e-4*length, rtol=0)
        close(site["axis_parameter"], target["s"], atol=2e-4, rtol=0)
        com = kin["com_position"][group]
        close(site["rigid_com"], com)
        point, node = np.asarray(site["rigid_point"]), np.asarray(site["node_point"])
        close(node, beta @ x0[indices], atol=2e-4*length, rtol=0)
        relative = beta @ v0[indices] - kin["com_velocity"][group] - np.cross(
            kin["angular_velocity"][group], point-com)
        close(site["relative_velocity"], relative)
        rate = float(normal @ relative)
        compression = max(WIDTH-site["gap"], 0)
        magnitude = STIFFNESS*compression*max(1-DAMPING*rate, 0)
        force = magnitude*normal
        close(site["law_input"][:11], np.r_[normal, relative, WIDTH, site["gap"], STIFFNESS, 1, DAMPING])
        close(site["law_output"], np.r_[rate, magnitude, force])
        close(site["node_forces"], beta[:, None]*force)
        close(site["rigid_force"], -force)
        close(site["rigid_torque"], np.cross(point-com, -force))
        dg, dr = abs(site["gap"]-target["gap"]), abs(rate-target["rate"])
        bound = STIFFNESS*(max(1-DAMPING*target["rate"], 0)*dg + target["compression"]*DAMPING*dr + DAMPING*dg*dr)
        bound += np.linalg.norm(target["force"])*np.linalg.norm(normal-target["normal"])
        require(np.linalg.norm(force-target["force"]) <= bound+ATOL+RTOL*max(1, magnitude),
                "independent finite-geometry force")
        total[indices] += site["node_forces"]
        wrenches[key][group] += np.r_[site["rigid_force"], site["rigid_torque"]]
        require(site["body_slot"] == observed["rigid_participants"][key]["body_slots"][group],
                "full physical body slot")
        local_point = kin["rotation"][group].T @ (point-kin["position"][group])
        local_axis = kin["rotation"][group].T @ (point-RADIUS*normal-kin["position"][group])
        next_point = next_kin["position"][group]+next_kin["rotation"][group]@local_point
        next_axis = next_kin["position"][group]+next_kin["rotation"][group]@local_axis
        next_node = beta @ after.node.position[indices]
        power += float(force@relative)
        work += float(force@(next_node-node)-force@(next_point-point))
        potential0 += .5*STIFFNESS*compression**2
        potential1 += .5*STIFFNESS*max(WIDTH-(np.linalg.norm(next_axis-next_node)-RADIUS),0)**2
        dissipation += (STIFFNESS*compression-magnitude)*rate
    require({key for key, target in expected.items() if target["compression"] > 0} <= seen,
            "all independently active addresses")
    close(observed["node_contact_force"], total)
    close(observed["node_total_force"], total)
    delivery = observed["node_force_delivery"]
    close(delivery.raw, total)
    np.testing.assert_array_equal(row.force, delivery.delivered)
    close(delivery.raw.astype(float)-delivery.delivered.astype(float), delivery.loss, atol=0, rtol=0)
    y, speed, multipliers = material_step(item, x0.astype(float), v0.astype(float), row.force.astype(float), H, row.sweeps)
    close(after.node.position, y, atol=2e-5, rtol=2e-5)
    close(after.node.velocity, speed, atol=2e-5, rtol=2e-5)
    close(row.multipliers, multipliers, atol=2e-5, rtol=2e-5)
    material = np.zeros((4, 3))
    for frame in row.trace:
        require(frame.shape == (5, 18), "complete ordered material rows")
        for e, (a, b) in enumerate(EDGES):
            direction, dl = frame[e, 14:17].astype(float), float(frame[e, 17])
            if dl != 0:
                separation = frame[e, :3].astype(float)-frame[e, 3:6]
                close(direction, separation/np.linalg.norm(separation), atol=2e-5, rtol=2e-5)
            material[a] += direction*dl/H
            material[b] -= direction*dl/H
    close(observed["material_impulse"], material)
    weights = item["node"].model.inverse_mass.astype(float)
    fixed = weights == 0
    reaction = np.zeros((4, 3))
    reaction[fixed] = -H*row.force[fixed]-material[fixed]
    close(observed["fixed_reaction_impulse"], reaction)
    np.testing.assert_array_equal(after.node.position[fixed], x0[fixed])
    np.testing.assert_array_equal(after.node.velocity[fixed], 0)
    impulse = (after.node.velocity[~fixed].astype(float)-v0[~fixed])/weights[~fixed, None]
    close(impulse, H*row.force[~fixed]+material[~fixed])
    resultant = total.sum(axis=0)
    moment = np.cross(x0, total).sum(axis=0)
    generalized_contact_power = 0.
    external_power = 0.
    audits = {}
    for key, body in bodies.items():
        record, kin = observed["rigid_participants"][key], k0[key]
        raw, delivered = record["rigid_wrench_delivery"], np.asarray(record["uploaded_rigid_wrench"])
        require(delivered.shape == (len(body["body_ids"]), 6), "no dropped unloaded body")
        require(tuple(record["body_ids"]) == body["body_ids"], "original body ids")
        close(raw.raw, wrenches[key])
        np.testing.assert_array_equal(record["rigid_wrench"], raw.delivered)
        np.testing.assert_array_equal(delivered, raw.delivered)
        close(raw.raw.astype(float)-raw.delivered.astype(float), raw.loss, atol=0, rtol=0)
        q0, speed0 = body_states[key]
        q1, speed1 = np.asarray(new[key].position), np.asarray(new[key].velocity)
        target_q, target_v, terms = reference_step(body, q0, speed0, delivered.astype(float), phase)
        diag = receipt.rigid_publication["diagnostics"][key]
        close(diag.entry_position, q0)
        close(diag.entry_velocity, speed0)
        for field in ("mass", "bias", "linear_jacobian", "angular_jacobian", "external", "drive", "acceleration"):
            close(getattr(diag, field), terms[field], atol=2e-5 if field == "mass" else ATOL)
        projected = applied(kin, np.zeros(len(body["kp"])), delivered[:, :3], delivered[:, 3:])
        close(receipt.rigid_publication["projected"][key], projected)
        close(diag.mass@diag.acceleration+diag.bias, diag.external+np.r_[np.zeros(6),diag.drive], atol=2e-5)
        require(diag.equation_residual <= 2e-5, "complete coupled mass equation")
        close(q1[:3], target_q[:3])
        close(q1[7:], target_q[7:])
        close(speed1, target_v)
        quaternion_close(q1[3:7], target_q[3:7])
        require(abs(q1[3:7]@q1[3:7]-1) <= 1e-5 and q1[3:7]@q0[3:7] >= 0, "root quaternion contract")
        require(new[key].tick == after.node.tick and new[key].control == old[key].control, "full tree accepted successor")
        geometry_close(receipt.rigid_publication["entry"][key]["geometry"], kin)
        geometry_close(receipt.rigid_publication["candidate"][key]["geometry"], k1[key])
        ext = applied(kin, np.zeros(len(body["kp"])), body["external"][:, :3], body["external"][:, 3:])
        close(record["rigid_external_entry_power"], ext@speed0)
        external_power += float(ext@speed0)
        generalized_contact_power += float(applied(kin, np.zeros(len(body["kp"])), wrenches[key][:,:3], wrenches[key][:,3:])@speed0)
        resultant += wrenches[key][:, :3].sum(axis=0)
        moment += (np.cross(kin["com_position"],wrenches[key][:,:3])+wrenches[key][:,3:]).sum(axis=0)
        contact_groups = {c["group"] for c in body["colliders"]}
        for group in set(range(len(body["body_ids"]))) - contact_groups:
            np.testing.assert_array_equal(delivered[group], 0)
        independent_end = mechanics(body["data"], target_q, target_v)
        close(k1[key]["momentum"], independent_end["momentum"])
        close(k1[key]["angular_momentum"], independent_end["angular_momentum"])
        audits[key] = dict(same_entry=terms, projected=projected, next_q=target_q, next_v=target_v,
            independent_wrench=exact_wrenches[key], actual_wrench=delivered, entry_geometry=kin,
            end_geometry=k1[key], independent_end=independent_end,
            contact_power=float(projected@speed0), external_power=float(ext@speed0),
            motor_power=float(terms["drive"]@speed0[6:]),
            # Full tree momentum change includes the declared finite drift; no
            # assertion that explicit gyro integration conserves L exactly.
            angular_change=k1[key]["angular_momentum"]-kin["angular_momentum"])
    close(power, float(np.sum(total*v0))+generalized_contact_power)
    close(resultant, 0)
    close(moment, 0)
    close(observed["rigid_external_entry_power"], external_power)
    close(observed["contact_work"], work)
    close(observed["layer_potential_before"], potential0)
    close(observed["layer_potential_after"], potential1)
    close(observed["entry_dissipation_rate"], dissipation)
    close(observed["work_balance"], work+potential1-potential0+H*dissipation)
    close(observed["node_external_work"], 0)
    return dict(independent_geometry=expected, independent_node_force=exact_force, per_body=audits,
        material_impulse=material, fixed_reaction=reaction, resultant=resultant, moment=moment,
        contact_entry_power=power)
def body_addresses(items):
    return tuple((body["key"], group, body_id)
        for item in sorted(items, key=lambda value: value["world"])
        for body in sorted(item["bodies"], key=lambda value: value["key"])
        for group, body_id in enumerate(body["body_ids"]))


def audit_source(receipt, items):
    source = receipt.query.source
    addresses = body_addresses(items)
    require(source["rigid_body_addresses"] == addresses, "complete canonical body address proof")
    expected_keys = {body["key"] for item in items for body in item["bodies"]}
    require(set(source["rigid_participant_sources"]) == expected_keys, "all participants retained")
    require(set(source["rigid_body_sources"]) == {(key, group) for key, group, _ in addresses},
            "full loaded and unloaded source proof")
    for item in items:
        world = item["world"]
        require(tuple(key for key, _ in source["world_rigid_participants"][world]) ==
                tuple(sorted(b["key"] for b in item["bodies"])), "all world source memberships")
        for body in item["bodies"]:
            key = body["key"]
            require(source["participant_body_counts"][key] == len(body["body_ids"]), "body count binding")
            first = source["rigid_participant_sources"][key]
            record = receipt.worlds[world]["rigid_participants"][key]
            slots = tuple(addresses.index((key, group, body_id)) for group, body_id in enumerate(body["body_ids"]))
            require(record["body_slots"] == slots and record["body_slot"] == slots[0], "canonical delivery slots")
            for group, body_id in enumerate(body["body_ids"]):
                proof = source["rigid_body_sources"][key, group]
                require(proof["participant"] == key and proof["logical_world"] == world
                    and proof["group"] == group and proof["body_id"] == body_id
                    and proof["body_slot"] == slots[group], "original body source identity")
                require(all(proof[field] == first[field] for field in
                            ("native", "slot", "row", "topology", "model_token", "graph_token")),
                        "same accepted tree source for every body")
                close(proof["anchor"], body["initial"].q[:3], atol=0, rtol=0)
                require(not proof["anchor"].flags.writeable, "immutable body source anchor")


def advance(scene, items, references, journal, identity, *, phase=0):
    entry, revision, stats = scene.snapshot(), scene.revision, scene.stats()
    inputs = forces(items)
    journal.emit("step-attempt", identity=identity, state=entry, expected_revision=revision,
                 inputs=inputs, h=H, stats=stats, independent_entry=references, phase=phase)
    try:
        import phymeo.constraints.distance as material
        import phymeo.dynamics.floating_robot as floating
        import phymeo.solvers.collision_loads as law

        def forbidden(*args, **kwargs):
            raise AssertionError("M-B2 used HOST current-state physics")

        with ExitStack() as guard:
            for module, name in ((material, "project_distances"), (floating, "floating_kinematics"), (law, "_host_law")):
                guard.enter_context(patch.object(module, name, forbidden))
            receipt = scene.step(inputs, H, expected_revision=revision)
        journal.emit("step-return", identity=identity, receipt=receipt)
        after, post = scene.snapshot(), scene.stats()
        journal.emit("step-state", identity=identity, state=after, stats=post)
        require(receipt.source_revision == revision and receipt.destination_revision == scene.revision,
                "one complete revision publication")
        require(scene.last_step is receipt, "same accepted receipt")
        require(post["response"]["launches"] == stats["response"]["launches"]+2, "one law and one transpose")
        keys = {b["key"] for item in items for b in item["bodies"]}
        require(all(set(receipt.rigid_publication[field]) == keys
                    for field in ("entry", "candidate", "diagnostics", "projected")), "all trees in one attempt")
        audit_source(receipt, items)
        continued, audits = {}, {}
        for item in items:
            world = item["world"]
            x, v, states = references[world]
            force, wrenches, geometry = reference_world(item, x, v, states)
            nx, nv, _ = material_step(item, x, v, force, H)
            next_bodies, independent_terms = {}, {}
            for body in item["bodies"]:
                key = body["key"]
                q, speed, terms = reference_step(body, *states[key], wrenches[key], phase)
                next_bodies[key] = q, speed
                independent_terms[key] = terms
            continued[world] = nx, nv, next_bodies
            journal.emit("independent-before-assert", identity=identity, world=world,
                         geometry=geometry, force=force, wrenches=wrenches,
                         state=continued[world], full_tree=independent_terms)
            audits[world] = audit_step(item, entry[world], after[world], receipt, phase)
            close(after[world].node.position, nx, atol=2e-4, rtol=0)
            close(after[world].node.velocity, nv, atol=2e-4, rtol=0)
            for body in item["bodies"]:
                key = body["key"]
                actual = after[world].rigid.participants[key]
                q, speed = next_bodies[key]
                close(actual.position[:3], q[:3], atol=2e-4, rtol=0)
                close(actual.position[7:], q[7:], atol=2e-4, rtol=0)
                close(actual.velocity, speed, atol=2e-4, rtol=0)
                quaternion_close(actual.position[3:7], q[3:7], atol=2e-4)
        journal.emit("step-audit", identity=identity, audits=audits, continuous=continued)
        return after, continued, receipt
    except BaseException as primary:
        try:
            journal.emit("step-failure", identity=identity, error=primary, last_attempt=scene.last_attempt)
        except BaseException as secondary:
            raise BaseExceptionGroup("M-B2 physics and capture failures", [primary, secondary])
        raise


def set_phase(scene, items, phase, journal):
    controls = {b["key"]: command(b, phase) for item in items for b in item["bodies"]}
    journal.emit("control-attempt", controls=controls, expected_revision=scene.revision)
    scene.set_controls(controls, expected_revision=scene.revision)
    journal.emit("control-return", phase=phase, checkpoint=scene.checkpoint())


def run_window(items, mode, journal, *, steps=8, require_loaded=True):
    journal.emit("construct-attempt", mode=mode, declaration=declaration(items), items=items)
    with owned_scene(items, mode, journal, "window") as scene:
        check_sources(scene, items)
        initial = scene.snapshot()
        refs = reference_initial(items)
        loaded = {b["key"]: set() for item in items for b in item["bodies"]}
        controls = 0
        for step in range(steps):
            phase = int(step >= 4)
            if step == 4:
                set_phase(scene, items, phase, journal)
                controls += 1
            current, refs, receipt = advance(scene, items, refs, journal, step, phase=phase)
            for item in items:
                for site in receipt.worlds[item["world"]]["sites"]:
                    if np.linalg.norm(site["rigid_force"]) > 0:
                        loaded[site["participant"]].add(site["body_group"])
        for item in items:
            world = item["world"]
            if require_loaded:
                require(np.max(abs(current[world].node.position-initial[world].node.position)) > 0,
                        "real node evolution")
            for body in item["bodies"]:
                key = body["key"]
                require(current[world].rigid.participants[key].tick == steps, "all actual participant steps")
                require(np.any(np.asarray(current[world].rigid.participants[key].position) !=
                               np.asarray(initial[world].rigid.participants[key].position)), "true tree motion")
                if require_loaded:
                    require(loaded[key] == {c["group"] for c in body["colliders"]}, "every declared loaded link carried force")
                else:
                    require(not loaded[key], "explicit None laws mean zero response sites")
        journal.emit("window-complete", final=current, loaded={key: tuple(sorted(value)) for key, value in loaded.items()}, stats=scene.stats())
    return dict(steps=steps, worldsteps=steps*len(items), participant_steps=steps*sum(len(i["bodies"]) for i in items),
                physical_body_steps=steps*sum(len(b["body_ids"]) for i in items for b in i["bodies"]),
                controls=controls, kind=tuple(i["kind"] for i in items))


def lifecycle(items, mode, journal):
    journal.emit("lifecycle-construct-attempt", declaration=declaration(items))
    with owned_scene(items, mode, journal, "lifecycle") as scene:
        initial, refs = scene.snapshot(), reference_initial(items)
        _, refs, _ = advance(scene, items, refs, journal, "warm")
        capsule, kept_refs = scene.checkpoint(), copy.deepcopy(refs)
        journal.emit("checkpoint", capsule=capsule, state=scene.snapshot())
        a, _, _ = advance(scene, items, copy.deepcopy(kept_refs), journal, "A")
        old_query = query(scene, journal)
        scene.restore(capsule, expected_revision=scene.revision)
        journal.emit("restore-return", checkpoint=scene.checkpoint(), state=scene.snapshot())
        try:
            scene.require_query_current(old_query, expected_revision=scene.revision)
        except ValueError as error:
            journal.emit("expected-stale", cause="restore", error=error)
        else:
            raise AssertionError("restore left old query current")
        b, refs, _ = advance(scene, items, copy.deepcopy(kept_refs), journal, "B")
        state_equal(a, b, items)
        before, selected = scene.snapshot(), items[0]
        old_query = query(scene, journal)
        scene.reset((selected["world"],), expected_revision=scene.revision)
        after = scene.snapshot()
        journal.emit("partial-reset-return", selected=selected["world"], state=after, checkpoint=scene.checkpoint())
        state_equal(initial, after, (selected,))
        state_equal(before, after, tuple(item for item in items if item is not selected))
        try:
            scene.require_query_current(old_query, expected_revision=scene.revision)
        except ValueError as error:
            journal.emit("expected-stale", cause="reset", error=error)
        else:
            raise AssertionError("reset left old query current")
        refs[selected["world"]] = reference_initial((selected,))[selected["world"]]
        advance(scene, items, refs, journal, "reset-successor")
    return dict(steps=4, worldsteps=4*len(items), participant_steps=4*sum(len(i["bodies"]) for i in items), queries=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("interpret", "cuda"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-install-root", "--installed-root", type=Path, required=True)
    args = parser.parse_args()
    require(args.output.is_absolute() and args.expected_install_root.is_absolute(), "absolute paths")
    args.output.mkdir(parents=True, exist_ok=False)
    root = args.expected_install_root.resolve(strict=True)
    journal = Journal(args.output / "deformable-articulated.jsonl.gz")
    summary = dict(status="RUNNING", mode=args.mode, physics_scope="M-B2 ordinary floating R/P tree",
                   consumer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    primary, errors = None, []
    try:
        summary["preflight"] = provenance(root, preflight=True)
        initialize()
        rows = [run_window(tuple(fixture(kind, world=w) for w in (31, 7)), args.mode, journal, steps=8)
                for kind in ("C", "D")]
        rows.append(lifecycle((fixture("C", world=7), fixture("D", world=31)), args.mode, journal))
        summary.update(status="PASS", cases=rows, worldsteps=sum(r["worldsteps"] for r in rows),
                       participant_steps=sum(r["participant_steps"] for r in rows))
        require(summary["worldsteps"] == 40 and summary["participant_steps"] == 40,
                "every declared actual world and tree step")
    except BaseException as error:
        primary = error
        summary.update(status="FAIL", error=repr(error), traceback=traceback.format_exc())
    finally:
        for label, operation in (("final_provenance", lambda: provenance(root)), ("trace", journal.close)):
            try:
                summary[label] = operation()
            except BaseException as error:
                errors.append(error)
                summary.update(status="FAIL")
                summary[label+"_failure"] = repr(error)
        summary["owner_cleanup"] = [dict(row) for row in _OWNER_CLOSE_RECORDS]
        summary["retained_close_failures"] = len(_RETAINED_CLOSE_FAILURES)
        summary["closed"] = all(row["close_succeeded"] for row in _OWNER_CLOSE_RECORDS) and "trace" in summary
        try:
            (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
        except BaseException as error:
            errors.append(error)
    if errors:
        raise BaseExceptionGroup("M-B2 consumer or finalization failed", ([primary] if primary else [])+errors)
    if primary is not None:
        raise primary


if __name__ == "__main__":
    main()