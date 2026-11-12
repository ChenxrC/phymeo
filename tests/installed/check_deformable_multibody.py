"""Standalone M-B1 multibody/deformable public declaration consumer.

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
from itertools import combinations
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


def reference_load(item, x, v, q, speed):
    axis = item["local"] @ rotation(q[3:7]).T + q[:3]
    segment = np.vstack((axis, axis)) if len(axis) == 1 else axis
    total = np.zeros((4, 3))
    wrench = np.zeros(6)
    records = []
    for ordinal, indices in enumerate(FACES):
        exact = oracle(x[indices], segment)
        d2, s, bary, _, point = exact["witnesses"][0]
        distance = math.sqrt(float(d2))
        beta = np.array([float(t) for t in bary])
        s = float(s)
        axis_point = (1 - s) * segment[0] + s * segment[1]
        node_point = beta @ x[indices]
        if distance == 0:
            raise ValueError("independent D3 reference has undefined normal")
        normal = (node_point - axis_point) / distance
        rigid_point = axis_point + RADIUS * normal
        gap = distance - RADIUS
        relative = beta @ v[indices] - speed[:3] - np.cross(speed[3:], rigid_point - q[:3])
        rate = np.dot(normal, relative)
        compression = max(WIDTH - gap, 0)
        force = STIFFNESS * compression * max(1 - DAMPING * rate, 0) * normal
        total[indices] += beta[:, None] * force
        wrench[:3] -= force
        wrench[3:] += np.cross(rigid_point - q[:3], -force)
        records.append(
            dict(
                ordinal=ordinal,
                distance=distance,
                gap=gap,
                bary=beta,
                s=s,
                normal=normal,
                node_point=node_point,
                rigid_point=rigid_point,
                relative=relative,
                force=force,
                compression=compression,
                rate=rate,
            )
        )
    return total, wrench, records


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


def rigid_step(item, q, v, wrench, h):
    wrench = wrench + item["external"]
    speed = v + h * np.r_[wrench[:3] / item["mass"], wrench[3:] / item["inertia"]]
    return np.r_[q[:3] + h * speed[:3], rotate_step(q[3:7], speed[3:], h)], speed


# Independent helpers above are copied from the accepted D5-A consumer
# 80a7897da7a57b24a4adfd1c035b894abde506b9726ff33853f23c6da354314d.
# They are local code, not imports of tests/examples or product mechanics.
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
            raise BaseExceptionGroup("M-B1 original and journal failure", [error, secondary])
        raise
    finally:
        try:
            journal.close()
        except BaseException as error:
            if primary is not None:
                raise BaseExceptionGroup("M-B1 original and close failure", [primary, error])
            raise


def fixture(kind="A", *, world=7, extra=False):
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
        raise BaseExceptionGroup("M-B1 primary and owner cleanup failures", failures)


def participants(items):
    return tuple(
        ph.ResidentParticipant(
            body["key"],
            ph.FloatingRobotTree(
                body["model"],
                body["initial"],
                kp=np.zeros(0),
                kd=np.zeros(0),
                effort_limit=np.zeros(0),
            ),
            (body["body_id"],),
        )
        for item in items
        for body in item["bodies"]
    )


def capacity_participants(count):
    body = fixture("A", world=7)["bodies"][0]
    return tuple(
        ph.ResidentParticipant(
            ph.ResidentParticipantKey(7, 100 + i),
            ph.FloatingRobotTree(
                body["model"],
                body["initial"],
                kp=np.zeros(0),
                kd=np.zeros(0),
                effort_limit=np.zeros(0),
            ),
            (300 + i,),
        )
        for i in range(count)
    )


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


def reference_world(item, x, v, body_states):
    total, wrenches, rows = np.zeros((4, 3)), {}, {}
    for body in item["bodies"]:
        q, speed = body_states[body["key"]]
        wrenches[body["key"]] = np.zeros(6)
        for collider in body["colliders"]:
            load, wrench, records = reference_load(
                dict(body, local=collider["local"]), x, v, q, speed
            )
            total += load
            wrenches[body["key"]] += wrench
            for record in records:
                rows[(collider["installed"], record["ordinal"])] = record
    return total, wrenches, rows


def audit_step(item, entry, after, receipt):
    observed = receipt.worlds[item["world"]]
    row = receipt.node_receipt.worlds[item["world"]]
    x0, v0 = entry.node.position, entry.node.velocity
    old_roots, new_roots = roots(entry, item), roots(after, item)
    body_by_key = {b["key"]: b for b in item["bodies"]}
    collider_by_key = {c["installed"]: (b, c) for b in item["bodies"] for c in b["colliders"]}
    body_states = {
        key: (np.asarray(s.position), np.asarray(s.velocity)) for key, s in old_roots.items()
    }
    exact_load, exact_wrenches, expected = reference_world(
        item, x0.astype(float), v0.astype(float), body_states
    )
    require(
        receipt.schema == "tyx-deformable-multibody-contact-step-v1", "explicit multibody schema"
    )
    require(
        set(observed["rigid_participants"]) == set(body_by_key), "complete participant publication"
    )
    require(
        not any(
            k in observed
            for k in ("rigid_wrench", "uploaded_rigid_wrench", "rigid_wrench_delivery")
        ),
        "no ambiguous world wrench",
    )
    require(after.node.tick == entry.node.tick + 1 == after.rigid.tick, "common tick")
    np.testing.assert_array_equal(row.input_position, x0)
    np.testing.assert_array_equal(row.input_velocity, v0)
    require(
        row.h == H and len(row.trace) == row.sweeps and row.residual <= SETTINGS.tolerance,
        "complete original material trace",
    )
    accumulated = np.zeros((4, 3))
    wrenches = {key: np.zeros(6) for key in body_by_key}
    seen = set()
    work = potential0 = potential1 = dissipation = entry_power = 0.0
    for site in observed["sites"]:
        require(
            site["surface"] == item["surface"] and site["round"] in collider_by_key,
            "source identity",
        )
        body, collider = collider_by_key[site["round"]]
        key = body["key"]
        require(site["participant"] == key and site["body_group"] == 0, "actual body owner")
        identity = (site["round"], int(site["facet_ordinal"]))
        require(identity not in seen, "no duplicate material site")
        seen.add(identity)
        target = expected[identity]
        indices = FACES[identity[1]]
        require(site["facet_id"] == FACETS[identity[1]], "stable facet id")
        np.testing.assert_array_equal(site["point_ids"], IDS[indices])
        np.testing.assert_array_equal(site["node_indices"], indices)
        beta, normal = np.asarray(site["barycentric"]), np.asarray(site["normal"])
        require(beta.shape == (3,) and min(beta) >= 0, "complete nonnegative barycentric")
        close(sum(beta), 1, atol=2e-6, rtol=0)
        q0, speed0 = body_states[key]
        q1, speed1 = np.asarray(new_roots[key].position), np.asarray(new_roots[key].velocity)
        local = collider["local"]
        segment = np.vstack((local, local)) if len(local) == 1 else local
        length = local_scale(x0[indices], segment, RADIUS)
        close(site["gap"], target["gap"], atol=1e-4 * length, rtol=0)
        close(normal, target["normal"], atol=2e-3, rtol=0)
        close(site["node_point"], target["node_point"], atol=2e-4 * length, rtol=0)
        close(site["rigid_point"], target["rigid_point"], atol=2e-4 * length, rtol=0)
        close(site["axis_parameter"], target["s"], atol=2e-4, rtol=0)
        close(site["rigid_com"], q0[:3])
        point, node = np.asarray(site["rigid_point"]), np.asarray(site["node_point"])
        close(node, beta @ x0[indices], atol=2e-4 * length, rtol=0)
        relative = beta @ v0[indices] - speed0[:3] - np.cross(speed0[3:], point - q0[:3])
        close(site["relative_velocity"], relative)
        rate = float(normal @ relative)
        compression = max(WIDTH - site["gap"], 0)
        magnitude = STIFFNESS * compression * max(1 - DAMPING * rate, 0)
        force = magnitude * normal
        close(
            site["law_input"][:11],
            np.r_[normal, relative, WIDTH, site["gap"], STIFFNESS, 1, DAMPING],
        )
        close(site["law_output"], np.r_[rate, magnitude, force])
        close(site["node_forces"], beta[:, None] * force)
        close(site["rigid_force"], -force)
        close(site["rigid_torque"], np.cross(point - q0[:3], -force))
        dg, dr = abs(site["gap"] - target["gap"]), abs(rate - target["rate"])
        bound = STIFFNESS * (
            max(1 - DAMPING * target["rate"], 0) * dg
            + target["compression"] * DAMPING * dr
            + DAMPING * dg * dr
        )
        bound += np.linalg.norm(target["force"]) * np.linalg.norm(normal - target["normal"])
        require(
            np.linalg.norm(force - target["force"]) <= bound + ATOL + RTOL * max(1, magnitude),
            "independent force from finite geometry",
        )
        accumulated[indices] += site["node_forces"]
        wrenches[key] += np.r_[site["rigid_force"], site["rigid_torque"]]
        close(site["body_slot"], observed["rigid_participants"][key]["body_slot"], atol=0, rtol=0)
        entry_power += float(force @ relative)
        next_point = q1[:3] + rotation(q1[3:]) @ (rotation(q0[3:]).T @ (point - q0[:3]))
        next_axis = q1[:3] + rotation(q1[3:]) @ (
            rotation(q0[3:]).T @ (point - RADIUS * normal - q0[:3])
        )
        next_node = beta @ after.node.position[indices]
        work += float(force @ (next_node - node) - force @ (next_point - point))
        potential0 += 0.5 * STIFFNESS * compression**2
        potential1 += (
            0.5 * STIFFNESS * max(WIDTH - (np.linalg.norm(next_axis - next_node) - RADIUS), 0) ** 2
        )
        dissipation += (STIFFNESS * compression - magnitude) * rate
    require(
        {identity for identity, record in expected.items() if record["compression"] > 0} <= seen,
        "all independently active material addresses present",
    )
    close(observed["node_contact_force"], accumulated)
    close(observed["node_total_force"], accumulated)
    close(row.force, observed["node_force_delivery"].delivered)
    close(observed["node_force_delivery"].raw, accumulated)
    close(
        observed["node_force_delivery"].raw.astype(float)
        - observed["node_force_delivery"].delivered.astype(float),
        observed["node_force_delivery"].loss,
        atol=0,
        rtol=0,
    )
    # Keep the original D5 same-entry ordered material gates and full row trace.
    y, speed, multipliers = material_step(
        item, x0.astype(float), v0.astype(float), row.force.astype(float), H, row.sweeps
    )
    close(after.node.position, y, atol=2e-5, rtol=2e-5)
    close(after.node.velocity, speed, atol=2e-5, rtol=2e-5)
    close(row.multipliers, multipliers, atol=2e-5, rtol=2e-5)
    material = np.zeros((4, 3))
    for frame in row.trace:
        require(frame.shape == (5, 18), "all ordered row columns")
        for edge, (a, b) in enumerate(EDGES):
            direction, delta = frame[edge, 14:17].astype(float), float(frame[edge, 17])
            if delta != 0:
                separation = frame[edge, :3].astype(float) - frame[edge, 3:6]
                close(direction, separation / np.linalg.norm(separation), atol=2e-5, rtol=2e-5)
            material[a] += direction * delta / H
            material[b] -= direction * delta / H
    close(observed["material_impulse"], material)
    weights = item["node"].model.inverse_mass.astype(float)
    fixed = weights == 0
    reaction = np.zeros((4, 3))
    reaction[fixed] = -H * row.force[fixed] - material[fixed]
    close(observed["fixed_reaction_impulse"], reaction)
    np.testing.assert_array_equal(after.node.position[fixed], x0[fixed])
    np.testing.assert_array_equal(after.node.velocity[fixed], 0)
    node_impulse = (after.node.velocity[~fixed].astype(float) - v0[~fixed]) / weights[~fixed, None]
    close(node_impulse, H * row.force[~fixed] + material[~fixed])
    resultant = accumulated.sum(axis=0)
    moment = np.cross(x0, accumulated).sum(axis=0)
    actual_total_impulse = node_impulse.sum(axis=0)
    external_total, external_power = np.zeros(3), 0.0
    p0 = v0[~fixed] / weights[~fixed, None]
    p1 = after.node.velocity[~fixed].astype(float) / weights[~fixed, None]
    angular = (np.cross(after.node.position[~fixed], p1) - np.cross(x0[~fixed], p0)).sum(axis=0)
    angular_expected = np.cross(x0[~fixed], H * row.force[~fixed] + material[~fixed]).sum(axis=0)
    angular_expected += np.cross(after.node.position[~fixed] - x0[~fixed], p1).sum(axis=0)
    body_audits = {}
    for body in item["bodies"]:
        key = body["key"]
        q0, speed0 = body_states[key]
        q1, speed1 = np.asarray(new_roots[key].position), np.asarray(new_roots[key].velocity)
        record = observed["rigid_participants"][key]
        delivery = record["rigid_wrench_delivery"]
        close(delivery.raw, wrenches[key][None, :])
        np.testing.assert_array_equal(record["rigid_wrench"], delivery.delivered)
        np.testing.assert_array_equal(record["uploaded_rigid_wrench"], delivery.delivered)
        close(
            delivery.raw.astype(float) - delivery.delivered.astype(float),
            delivery.loss,
            atol=0,
            rtol=0,
        )
        actual_wrench = np.asarray(record["uploaded_rigid_wrench"])[0].astype(float)
        diag = receipt.rigid_publication["diagnostics"][key]
        close(diag.mass, np.diag([body["mass"]] * 3 + [body["inertia"]] * 3), atol=2e-5)
        close(diag.bias, 0)
        close(diag.linear_jacobian, np.eye(3, 6)[None])
        close(diag.angular_jacobian, np.eye(3, 6, k=3)[None])
        close(receipt.rigid_publication["projected"][key], actual_wrench)
        total_wrench = actual_wrench + body["external"]
        acceleration = np.r_[total_wrench[:3] / body["mass"], total_wrench[3:] / body["inertia"]]
        close(diag.acceleration, acceleration)
        require(diag.equation_residual <= 2e-5, "per-body full mass equation")
        expected_q, expected_v = rigid_step(body, q0, speed0, actual_wrench, H)
        close(q1[:3], expected_q[:3])
        close(speed1, expected_v)
        require(
            min(np.linalg.norm(q1[3:] - expected_q[3:]), np.linalg.norm(q1[3:] + expected_q[3:]))
            <= ATOL + RTOL,
            "per-body physical orientation",
        )
        require(new_roots[key].tick == after.node.tick, "every participant actual successor")
        close(record["rigid_external_entry_power"], body["external"] @ speed0)
        external_power += float(body["external"] @ speed0)
        external_total += body["external"][:3]
        actual_total_impulse += body["mass"] * (speed1[:3] - speed0[:3])
        angular += np.cross(q1[:3], body["mass"] * speed1[:3]) - np.cross(
            q0[:3], body["mass"] * speed0[:3]
        )
        angular += body["inertia"] * (speed1[3:] - speed0[3:])
        angular_expected += np.cross(q0[:3], H * total_wrench[:3]) + H * total_wrench[3:]
        angular_expected += np.cross(q1[:3] - q0[:3], body["mass"] * speed1[:3])
        resultant += wrenches[key][:3]
        moment += np.cross(q0[:3], wrenches[key][:3]) + wrenches[key][3:]
        body_audits[key] = dict(
            independent_wrench=exact_wrenches[key],
            actual_wrench=actual_wrench,
            expected_q=expected_q,
            expected_v=expected_v,
            contact_power=float(actual_wrench @ speed0),
            external_power=float(body["external"] @ speed0),
        )
    close(
        entry_power,
        float(np.sum(accumulated * v0))
        + sum(float(wrenches[k] @ body_states[k][1]) for k in wrenches),
    )
    close(resultant, 0)
    close(moment, 0)
    close(angular, angular_expected)
    close(actual_total_impulse, reaction.sum(axis=0) + H * external_total)
    close(observed["rigid_external_entry_power"], external_power)
    close(observed["contact_work"], work)
    close(observed["layer_potential_before"], potential0)
    close(observed["layer_potential_after"], potential1)
    close(observed["entry_dissipation_rate"], dissipation)
    close(observed["work_balance"], work + potential1 - potential0 + H * dissipation)
    close(observed["node_external_work"], 0)
    return dict(
        independent_geometry=expected,
        independent_node_force=exact_load,
        per_body=body_audits,
        material_impulse=material,
        fixed_reaction=reaction,
        resultant=resultant,
        moment=moment,
        angular_change=angular,
        discrete_angular_expected=angular_expected,
        contact_entry_power=entry_power,
    )


def advance(scene, items, references, journal, identity):
    entry, revision, stats = scene.snapshot(), scene.revision, scene.stats()
    inputs = forces(items)
    journal.emit(
        "step-attempt",
        identity=identity,
        state=entry,
        expected_revision=revision,
        inputs=inputs,
        h=H,
        stats=stats,
        independent_entry=references,
    )
    try:
        import phymeo.constraints.distance as material
        import phymeo.dynamics.floating_robot as floating
        import phymeo.solvers.collision_loads as law

        def forbidden(*args, **kwargs):
            raise AssertionError("M-B1 used HOST current-state physics")

        with ExitStack() as guard:
            for module, name in (
                (material, "project_distances"),
                (floating, "floating_kinematics"),
                (law, "_host_law"),
            ):
                guard.enter_context(patch.object(module, name, forbidden))
            receipt = scene.step(inputs, H, expected_revision=revision)
        journal.emit("step-return", identity=identity, receipt=receipt)
        after, post = scene.snapshot(), scene.stats()
        journal.emit("step-state", identity=identity, state=after, stats=post)
        require(
            receipt.source_revision == revision and receipt.destination_revision == scene.revision,
            "one common revision publication",
        )
        require(scene.last_step is receipt, "same accepted receipt")
        require(
            post["response"]["launches"] == stats["response"]["launches"] + 2,
            "one law plus one multibody transpose",
        )
        all_keys = tuple(sorted(b["key"] for i in items for b in i["bodies"]))
        require(
            set(receipt.rigid_publication["entry"])
            == set(all_keys)
            == set(receipt.rigid_publication["candidate"])
            == set(receipt.rigid_publication["diagnostics"])
            == set(receipt.rigid_publication["projected"]),
            "every real participant in one attempt",
        )
        source = receipt.query.source
        require(
            set(source["rigid_participant_sources"]) == set(all_keys),
            "complete actual device source table",
        )
        for item in items:
            declared_keys = tuple(sorted(b["key"] for b in item["bodies"]))
            require(
                tuple(k for k, _ in source["world_rigid_participants"][item["world"]])
                == declared_keys,
                "all loaded and unloaded source members",
            )
        continued, audits = {}, {}
        for item in items:
            world = item["world"]
            x, v, body_states = references[world]
            force, wrenches, geometry = reference_world(item, x, v, body_states)
            next_x, next_v, _ = material_step(item, x, v, force, H)
            next_bodies = {
                b["key"]: rigid_step(b, *body_states[b["key"]], wrenches[b["key"]], H)
                for b in item["bodies"]
            }
            continued[world] = next_x, next_v, next_bodies
            journal.emit(
                "independent-before-assert",
                identity=identity,
                world=world,
                geometry=geometry,
                force=force,
                wrenches=wrenches,
                state=continued[world],
            )
            audits[world] = audit_step(item, entry[world], after[world], receipt)
            close(after[world].node.position, next_x, atol=2e-4, rtol=0)
            close(after[world].node.velocity, next_v, atol=2e-4, rtol=0)
            for body in item["bodies"]:
                key = body["key"]
                actual = after[world].rigid.participants[key]
                q, speed = next_bodies[key]
                close(actual.position[:3], q[:3], atol=2e-4, rtol=0)
                close(actual.velocity, speed, atol=2e-4, rtol=0)
                aq = np.asarray(actual.position[3:])
                require(
                    min(np.linalg.norm(aq - q[3:]), np.linalg.norm(aq + q[3:])) <= 2e-4,
                    "independently continuous orientation",
                )
                require(
                    receipt.worlds[world]["rigid_participants"][key]["body_slot"]
                    == all_keys.index(key),
                    "canonical physical body slot independent of publication",
                )
        journal.emit("step-audit", identity=identity, audits=audits, continuous=continued)
        return after, continued, receipt
    except BaseException as primary:
        try:
            journal.emit(
                "step-failure", identity=identity, error=primary, last_attempt=scene.last_attempt
            )
        except BaseException as secondary:
            raise BaseExceptionGroup("M-B1 physics and capture failures", [primary, secondary])
        raise


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


def run_window(items, mode, journal, *, steps):
    require(steps == (16 if items[0]["kind"] == "B" else 4), "fixed window length")
    journal.emit(
        "declaration",
        items=items,
        declaration=declaration(items),
        steps=steps,
        h=H,
        gates=dict(
            force_atol=ATOL,
            force_rtol=RTOL,
            geometry=1e-4,
            witness=2e-4,
            normal=2e-3,
            short_state=2e-5,
            continuous=2e-4,
        ),
    )
    with owned_scene(items, mode, journal, "window") as scene:
        check_sources(scene, items)
        initial = scene.snapshot()
        journal.emit("constructed", state=initial, source_map=scene.source_map, stats=scene.stats())
        references = reference_initial(items)
        histories = {b["key"]: [] for i in items for b in i["bodies"]}
        torques = {key: [] for key in histories}
        for step in range(steps):
            state, references, receipt = advance(scene, items, references, journal, step)
            for item in items:
                for body in item["bodies"]:
                    key = body["key"]
                    wrench = receipt.worlds[item["world"]]["rigid_participants"][key][
                        "uploaded_rigid_wrench"
                    ][0]
                    histories[key].append(float(np.linalg.norm(wrench[:3])))
                    torques[key].append(float(np.linalg.norm(wrench[3:])))
        for item in items:
            for body in item["bodies"]:
                key, world = body["key"], item["world"]
                old, actual = (
                    initial[world].rigid.participants[key],
                    state[world].rigid.participants[key],
                )
                require(
                    np.any(np.asarray(actual.position) != old.position),
                    "every participant really moved",
                )
                if item["kind"] == "A":
                    require(
                        all(value > 0 for value in histories[key]), "two distinct loaded bodies"
                    )
                elif (world == 7 and key.participant == 19) or (
                    world == 31 and key.participant == 37
                ):
                    values = histories[key]
                    require(
                        values[0] == 0
                        and any(value > 0 for value in values[1:-1])
                        and values[-1] == 0,
                        "same continuous trajectory enters and leaves the layer",
                    )
                elif world == 31:
                    require(
                        all(value == 0 for value in histories[key]),
                        "inactive body has no stale force",
                    )
                else:
                    require(
                        all(value > 0 for value in histories[key]), "other body remains supported"
                    )
            if item["kind"] == "A":
                require(
                    any(t > 0 for t in torques[item["bodies"][0]["key"]]), "off-COM reaction torque"
                )
        journal.emit(
            "window-complete",
            steps=steps,
            worldsteps=steps * len(items),
            participant_steps=steps * len(histories),
            force_history=histories,
            torque_history=torques,
            state=state,
            stats=scene.stats(),
        )
    return dict(
        steps=steps, worldsteps=steps * len(items), participant_steps=steps * len(histories)
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


def lifecycle(items, mode, journal):
    journal.emit("lifecycle-construct-attempt", declaration=declaration(items), items=items)
    with owned_scene(items, mode, journal, "lifecycle") as scene:
        check_sources(scene, items)
        initial, references = scene.snapshot(), reference_initial(items)
        _, references, _ = advance(scene, items, references, journal, "warm")
        capsule, saved_reference = scene.checkpoint(), copy.deepcopy(references)
        journal.emit("checkpoint", capsule=capsule, state=scene.snapshot())
        a, _, _ = advance(scene, items, copy.deepcopy(saved_reference), journal, "A")
        old_query = query(scene, journal)
        scene.restore(capsule, expected_revision=scene.revision)
        journal.emit("restore-return", state=scene.snapshot(), checkpoint=scene.checkpoint())
        try:
            scene.require_query_current(old_query, expected_revision=scene.revision)
        except ValueError as error:
            journal.emit("expected-stale", cause="restore", error=error)
        else:
            raise AssertionError("restore left old query current")
        b, refs_b, _ = advance(scene, items, copy.deepcopy(saved_reference), journal, "B")
        state_equal(a, b, items)
        selected = items[0]
        current = query(scene, journal)
        first_body = selected["bodies"][0]
        scene.set_controls(
            {
                first_body["key"]: ph.FloatingRobotCommand(
                    None, ph.FloatingRobotInputs(np.zeros(0), np.ones((1, 3)), np.zeros((1, 3)))
                )
            },
            expected_revision=scene.revision,
        )
        journal.emit("control-return", state=scene.snapshot(), checkpoint=scene.checkpoint())
        try:
            scene.require_query_current(current, expected_revision=scene.revision)
        except ValueError as error:
            journal.emit("expected-stale", cause="control", error=error)
        else:
            raise AssertionError("controls left old query current")
        before, checkpoint_before = scene.snapshot(), scene.checkpoint()
        scene.reset((selected["world"],), expected_revision=scene.revision)
        after = scene.snapshot()
        journal.emit(
            "reset-return",
            selected=selected["world"],
            before=checkpoint_before,
            state=after,
            checkpoint=scene.checkpoint(),
        )
        state_equal(initial, after, (selected,))
        state_equal(before, after, tuple(i for i in items if i is not selected))
        refs_b[selected["world"]] = reference_initial((selected,))[selected["world"]]
        advance(scene, items, refs_b, journal, "reset-successor")
    return dict(
        steps=4,
        worldsteps=4 * len(items),
        participant_steps=4 * sum(len(i["bodies"]) for i in items),
        queries=2,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("interpret", "cuda"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-install-root", "--installed-root", type=Path, required=True)
    args = parser.parse_args()
    require(
        args.output.is_absolute() and args.expected_install_root.is_absolute(), "absolute paths"
    )
    args.output.mkdir(parents=True, exist_ok=False)
    root = args.expected_install_root.resolve(strict=True)
    journal = Journal(args.output / "deformable-multibody.jsonl.gz")
    summary = dict(
        status="RUNNING",
        mode=args.mode,
        physics_scope="M-B1 independent free bodies",
        consumer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    primary, errors = None, []
    try:
        summary["preflight"] = provenance(root, preflight=True)
        initialize()
        rows = [
            run_window((fixture("A", world=7, extra=True),), args.mode, journal, steps=4),
            run_window(
                (fixture("B", world=7), fixture("B", world=31)), args.mode, journal, steps=16
            ),
            lifecycle(
                (fixture("A", world=7), fixture("A", world=31, extra=True)), args.mode, journal
            ),
        ]
        summary.update(
            status="PASS",
            cases=rows,
            worldsteps=sum(r["worldsteps"] for r in rows),
            participant_steps=sum(r["participant_steps"] for r in rows),
        )
        require(
            summary["worldsteps"] == 44 and summary["participant_steps"] == 88,
            "all declared high-level actual steps",
        )
    except BaseException as error:
        primary = error
        summary.update(status="FAIL", error=repr(error), traceback=traceback.format_exc())
    finally:
        for label, operation in (
            ("final_provenance", lambda: provenance(root)),
            ("trace", journal.close),
        ):
            try:
                summary[label] = operation()
            except BaseException as error:
                errors.append(error)
                summary.update(status="FAIL")
                summary[label + "_failure"] = repr(error)
        summary["owner_cleanup"] = [dict(row) for row in _OWNER_CLOSE_RECORDS]
        summary["retained_close_failures"] = len(_RETAINED_CLOSE_FAILURES)
        summary["closed"] = (
            all(row["close_succeeded"] for row in _OWNER_CLOSE_RECORDS) and "trace" in summary
        )
        try:
            (args.output / "summary.json").write_text(
                json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
            )
        except BaseException as error:
            errors.append(error)
    if errors:
        raise BaseExceptionGroup(
            "M-B1 consumer or finalization failed", ([primary] if primary else []) + errors
        )
    if primary is not None:
        raise primary


if __name__ == "__main__":
    main()
