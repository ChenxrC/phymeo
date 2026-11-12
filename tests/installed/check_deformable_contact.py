"""Standalone installed D3 consumer: common force/motion and honest finite scope.

No source examples/tests or product geometry/mechanics helpers are imported.
The exact active-face oracle and FP64 ordered material recurrence below are
self-contained independent calculations, not device-state replacements.
"""

import argparse
import gzip
import hashlib
import json
import math
import sys
import traceback
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from fractions import Fraction as Q
from itertools import combinations
from pathlib import Path

import numpy as np

H = 1/1024
WIDTH, STIFFNESS, DAMPING, RADIUS = 1/16, 128.0, 1/2, 1/8
ATOL, RTOL = 5e-6, 5e-5
IDS = np.array((101,113,127,139),np.int64)
FACES = np.array(((0,1,2),(0,2,3)),np.int64)
FACETS = np.array((503,509),np.int64)
EDGES = np.array(((0,1),(1,2),(2,3),(3,0),(0,2)),np.int64)
REST = np.array((1,1,1,1,math.sqrt(2)))
COMPLIANCE = 1/16384


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
    return np.array(((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)),
                     (2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)),
                     (2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y))))


def rotate_step(q, omega, h):
    theta = h * np.linalg.norm(omega)
    a = np.r_[omega * (h/2 if theta == 0 else math.sin(theta/2)/np.linalg.norm(omega)),
              math.cos(theta/2)]
    b = np.asarray(q, float)
    result = np.r_[a[3]*b[:3] + b[3]*a[:3] + np.cross(a[:3], b[:3]),
                   a[3]*b[3] - np.dot(a[:3], b[:3])]
    return result / np.linalg.norm(result)


def fixture(case="P", *, world=7, factor=1, variant=None, reverse=False):
    x = np.array(((0,0,0),(1,0,0),(1,1,0),(0,1,0)), np.float32)
    if case == "C":
        x = x[:, (2,0,1)].copy()
    weights = np.full(4, 2/factor, np.float32)
    if case == "E":
        weights[1] = 0
    spec = ph.DeformableNodeWorld(world, ph.ParticleModel(weights, np.zeros(3, np.float32)),
        ph.ParticleState(x, np.zeros_like(x)),
        ph.DistanceConstraints(EDGES, REST, np.full(5, COMPLIANCE)), IDS,
        (ph.DeformableSurface(23, FACETS, IDS[FACES]),))
    center = np.array((3/4,1/8,5/32), float)
    local = np.zeros((1,3))
    velocity = np.array((0,0,-1/16,0,0,0), float)
    external = np.zeros(6)
    kind = "sphere"
    if case == "E":
        center[0] = 1/2
        local[0,0] = 1/4
    elif case == "C":
        center = np.array((5/32,7/8,1/8))
        local = np.array(((0,-1/8,0),(1/8,1/8,0)), float)
        velocity = np.array((-1/16,1/8,0,0,0,0), float)
        kind = "capsule"
    if variant == "conditioning":
        center = np.array((1/2,1/2,5/32))
        local = np.array(((-1/8,-1/8,0),(1/8,1/8,0)), float)
        kind = "capsule"
    elif variant == "outside":
        center[2] = RADIUS + 3/32
    elif variant == "opening":
        velocity[2] = 4
    elif variant == "entry-exit":
        center[2] = RADIUS+WIDTH+1/16384
        velocity[2] = -1/8
        external[2] = 64
    root = ph.RigidBodyModel(np.array((1/(2*factor),)), np.full((1,3), factor/4), np.zeros(3))
    model = ph.FloatingRobotModel(root, np.zeros(3), None)
    initial = ph.FloatingRobotState(np.r_[center,(0,0,0,1)], velocity)
    tree = ph.FloatingRobotTree(model, initial, kp=np.zeros(0), kd=np.zeros(0),
                               effort_limit=np.zeros(0))
    key = ph.ResidentParticipantKey(world,19)
    participant = ph.ResidentParticipant(key,tree,(202,))
    sk, rk = ph.CollisionKey(world,101,0), ph.CollisionKey(world,202,0)
    shape = ph.CollisionShape(rk,kind,tuple(range(41,41+len(local))),RADIUS,revision=3)
    surface = ph.DeformableBoundarySurface(sk,23)
    shapes = (shape,surface) if reverse else (surface,shape)
    binding = ph.ResidentParticipantBinding(key,ph.ResidentFloatingTreeBinding(
        rk,world,0,local,topology=participant.compiled.topology))
    return dict(case=case, world=world, factor=factor, node=spec, participant=participant,
                model=ph.DeformableBoundaryModel(shapes), binding=binding,
                materials={(sk,rk):ph.NormalLayer(WIDTH,STIFFNESS,DAMPING)},
                surface=sk, round=rk, local=local, mass=2*factor, inertia=factor/4,
                root_initial=initial, root_model=model,
                external=external, command=tree.inputs(np.zeros((0,3)),
                    ph.FloatingRobotInputs(np.zeros(0),external[None,:3],external[None,3:])))


def create(items, mode, **options):
    shapes = tuple(s for item in items for s in item["model"].shapes)
    laws = {k:v for item in items for k,v in item["materials"].items()}
    return ph.TyxDeformableContactBatch(tuple(i["node"] for i in items),
        tuple(i["participant"] for i in items), ph.DeformableBoundaryModel(shapes),
        tuple(i["binding"] for i in items), laws, mode=mode, settings=SETTINGS, **options)


def manifest(item):
    return {name:item[name] for name in ("case","world","factor","node","surface","round",
        "local","mass","inertia","materials","external","root_initial","root_model")}

def forces(items):
    return {i["world"]:ph.StepInputs(np.zeros((4,3),np.float32)) for i in items}


def root_state(state, item):
    return state.rigid.participants[item["participant"].key]


def reference_load(item, x, v, q, speed):
    axis = item["local"] @ rotation(q[3:7]).T + q[:3]
    segment = np.vstack((axis,axis)) if len(axis) == 1 else axis
    total = np.zeros((4,3))
    wrench = np.zeros(6)
    records = []
    for ordinal, indices in enumerate(FACES):
        exact = oracle(x[indices],segment)
        d2,s,bary,_,point = exact["witnesses"][0]
        distance = math.sqrt(float(d2))
        beta = np.array([float(t) for t in bary])
        s = float(s)
        axis_point = (1-s)*segment[0] + s*segment[1]
        node_point = beta @ x[indices]
        if distance == 0:
            raise ValueError("independent D3 reference has undefined normal")
        normal = (node_point-axis_point)/distance
        rigid_point = axis_point + RADIUS*normal
        gap = distance-RADIUS
        relative = beta @ v[indices] - speed[:3] - np.cross(speed[3:],rigid_point-q[:3])
        rate = np.dot(normal,relative)
        compression = max(WIDTH-gap,0)
        force = STIFFNESS*compression*max(1-DAMPING*rate,0)*normal
        total[indices] += beta[:,None]*force
        wrench[:3] -= force
        wrench[3:] += np.cross(rigid_point-q[:3],-force)
        records.append(dict(ordinal=ordinal,distance=distance,gap=gap,bary=beta,s=s,
            normal=normal,node_point=node_point,rigid_point=rigid_point,relative=relative,
            force=force,compression=compression,rate=rate))
    return total,wrench,records


def material_step(item, x, v, force, h, sweeps=None):
    weights = item["node"].model.inverse_mass.astype(float)
    pv = v + h*weights[:,None]*force
    px = x + h*pv
    fixed = weights == 0
    px[fixed],pv[fixed] = x[fixed],0
    y,lam = px.copy(),np.zeros(5)
    alpha = COMPLIANCE/h/h
    for _ in range(SETTINGS.position_iterations if sweeps is None else sweeps):
        for e,(a,b) in enumerate(EDGES):
            diff = y[a]-y[b]
            length = np.linalg.norm(diff)
            direction = diff/length
            dl = -(length-REST[e]+alpha*lam[e])/(weights[a]+weights[b]+alpha)
            lam[e] += dl
            y[a] += weights[a]*dl*direction
            y[b] -= weights[b]*dl*direction
        if sweeps is None:
            lengths = np.linalg.norm(y[EDGES[:,0]]-y[EDGES[:,1]],axis=1)
            residual = np.max(abs(lengths-REST+alpha*lam)/np.maximum(REST,SETTINGS.reference_length))
            if residual <= SETTINGS.tolerance:
                break
    else:
        if sweeps is None:
            raise AssertionError("independent material reference exhausted its fixed budget")
    speed = pv+(y-px)/h
    speed[fixed] = 0
    return y,speed,lam


def rigid_step(item,q,v,wrench,h):
    wrench = wrench + item["external"]
    speed = v + h*np.r_[wrench[:3]/item["mass"],wrench[3:]/item["inertia"]]
    return np.r_[q[:3]+h*speed[:3],rotate_step(q[3:7],speed[3:],h)],speed


def audit_step(item, entry, after, receipt, h):
    w = item["world"]
    observed = receipt.worlds[w]
    row = receipt.node_receipt.worlds[w]
    root0,root1 = root_state(entry,item),root_state(after,item)
    q0,v0 = np.asarray(root0.position),np.asarray(root0.velocity)
    q1,v1 = np.asarray(root1.position),np.asarray(root1.velocity)
    x0,vs0 = entry.node.position,entry.node.velocity
    node_force,wrench,reference = reference_load(item,x0.astype(float),vs0.astype(float),q0,v0)
    assert observed["entry"].node.tick == entry.node.tick
    assert after.node.tick == entry.node.tick+1 == after.rigid.tick
    np.testing.assert_array_equal(row.input_position,x0)
    np.testing.assert_array_equal(row.input_velocity,vs0)
    assert row.input_tick == entry.node.tick and row.h == h
    assert row.residual <= SETTINGS.tolerance
    assert len(row.trace) == row.sweeps
    seen = set()
    accumulated = np.zeros((4,3))
    device_wrench = np.zeros(6)
    for site in observed["sites"]:
        assert site["surface"] == item["surface"] and site["round"] == item["round"]
        ordinal = int(site["facet_ordinal"])
        assert ordinal not in seen and ordinal in (0,1)
        seen.add(ordinal)
        assert site["facet_id"] == FACETS[ordinal]
        indices = FACES[ordinal]
        np.testing.assert_array_equal(site["point_ids"],IDS[indices])
        np.testing.assert_array_equal(site["node_indices"],indices)
        beta = np.asarray(site["barycentric"],float)
        assert beta.shape == (3,) and min(beta) >= 0
        close(sum(beta),1,atol=2e-6,rtol=0)
        target = reference[ordinal]
        length = local_scale(x0[indices],
            np.vstack((item["local"],item["local"])) if len(item["local"]) == 1 else item["local"],RADIUS)
        close(site["gap"],target["gap"],atol=1e-4*length,rtol=0)
        close(site["normal"],target["normal"],atol=2e-3,rtol=0)
        close(site["node_point"],target["node_point"],atol=2e-4*length,rtol=0)
        close(site["rigid_point"],target["rigid_point"],atol=2e-4*length,rtol=0)
        close(site["node_point"],beta @ x0[indices],atol=2e-4*length,rtol=0)
        close(site["rigid_com"],q0[:3])
        assert 0 <= site["axis_parameter"] <= 1
        close(site["axis_parameter"],target["s"],atol=2e-4,rtol=0)
        relative = beta @ vs0[indices] - v0[:3] - np.cross(v0[3:],np.asarray(site["rigid_point"])-q0[:3])
        close(site["relative_velocity"],relative)
        normal = np.asarray(site["normal"],float)
        rate = float(np.dot(normal,relative))
        compression = max(WIDTH-site["gap"],0)
        magnitude = STIFFNESS*compression*max(1-DAMPING*rate,0)
        force = magnitude*normal
        packed = np.asarray(site["law_input"])
        close(packed[:3],normal)
        close(packed[3:6],relative)
        close(packed[6:11],[WIDTH,site["gap"],STIFFNESS,1,DAMPING])
        close(site["law_output"],np.r_[rate,magnitude,force])
        expected_node = beta[:,None]*force
        close(site["node_forces"],expected_node)
        close(site["rigid_force"],-force)
        close(site["rigid_torque"],np.cross(np.asarray(site["rigid_point"])-q0[:3],-force))
        accumulated[indices] += np.asarray(site["node_forces"])
        device_wrench += np.r_[site["rigid_force"],site["rigid_torque"]]
        dg = abs(site["gap"]-target["gap"])
        dr = abs(rate-target["rate"])
        scalar_bound = STIFFNESS*(max(1-DAMPING*target["rate"],0)*dg +
            target["compression"]*DAMPING*dr + DAMPING*dg*dr)
        direction_bound = np.linalg.norm(target["force"])*np.linalg.norm(normal-target["normal"])
        assert np.linalg.norm(force-target["force"]) <= scalar_bound+direction_bound+ATOL+RTOL*max(1,magnitude)
    # All independently active facets must have actual sites; an inactive far
    # row may be retained as a zero-force site or remain only in the query.
    assert {r["ordinal"] for r in reference if r["compression"] > 0} <= seen
    close(observed["node_contact_force"],accumulated)
    close(observed["rigid_wrench"],device_wrench[None,:])
    close(observed["uploaded_rigid_wrench"],device_wrench[None,:])
    close(accumulated.sum(axis=0)+device_wrench[:3],0)
    close(np.cross(x0,accumulated).sum(axis=0)+np.cross(q0[:3],device_wrench[:3])+device_wrench[3:],0)
    total = np.asarray(observed["node_total_force"],float)
    close(total,accumulated)
    close(row.force,total)
    y,velocity,lam = material_step(item,x0.astype(float),vs0.astype(float),row.force.astype(float),h,row.sweeps)
    close(after.node.position,y,atol=2e-5,rtol=2e-5)
    close(after.node.velocity,velocity,atol=2e-5,rtol=2e-5)
    close(row.multipliers,lam,atol=2e-5,rtol=2e-5)
    diagnostics = receipt.rigid_publication["diagnostics"][item["participant"].key]
    independent_mass = np.diag([item["mass"]]*3+[item["inertia"]]*3)
    close(diagnostics.mass,independent_mass,atol=2e-5)
    close(diagnostics.bias,0)
    close(diagnostics.linear_jacobian,np.eye(3,6)[None,:,:])
    close(diagnostics.angular_jacobian,np.eye(3,6,k=3)[None,:,:])
    total_wrench = device_wrench+item["external"]
    close(diagnostics.acceleration,np.r_[total_wrench[:3]/item["mass"],total_wrench[3:]/item["inertia"]])
    assert diagnostics.equation_residual <= 2e-5
    expected_q,expected_v = rigid_step(item,q0,v0,device_wrench,h)
    close(q1[:3],expected_q[:3])
    # Physical orientation is sign-invariant, but the original raw q is saved.
    assert min(np.linalg.norm(q1[3:]-expected_q[3:]),np.linalg.norm(q1[3:]+expected_q[3:])) <= ATOL+RTOL
    close(v1,expected_v)
    material = np.zeros((4,3))
    for frame in row.trace:
        assert frame.shape == (5,18)
        for e,(a,b) in enumerate(EDGES):
            delta = float(frame[e,17])
            direction = frame[e,14:17].astype(float)
            separation = frame[e,:3].astype(float)-frame[e,3:6]
            if delta != 0:
                close(direction,separation/np.linalg.norm(separation),atol=2e-5,rtol=2e-5)
            impulse = direction*(delta/h)
            material[a] += impulse
            material[b] -= impulse
    close(observed["material_impulse"],material)
    weights = item["node"].model.inverse_mass.astype(float)
    fixed = weights == 0
    reaction = np.zeros((4,3))
    reaction[fixed] = -h*total[fixed]-material[fixed]
    close(observed["fixed_reaction_impulse"],reaction)
    np.testing.assert_array_equal(after.node.position[fixed],x0[fixed])
    np.testing.assert_array_equal(after.node.velocity[fixed],0)
    free = ~fixed
    impulse_actual = (after.node.velocity[free].astype(float)-vs0[free])/weights[free,None]
    close(impulse_actual,h*total[free]+material[free])
    root_impulse = item["mass"]*(v1[:3]-v0[:3])
    close(impulse_actual.sum(axis=0)+root_impulse,reaction.sum(axis=0)+h*item["external"][:3])
    # Account for application time explicitly instead of claiming that every
    # ordered central row has zero torque about the initial positions.
    p0 = vs0[free]/weights[free,None]
    p1 = after.node.velocity[free]/weights[free,None]
    angular = (np.cross(after.node.position[free],p1)-np.cross(x0[free],p0)).sum(axis=0)
    angular += np.cross(q1[:3],item["mass"]*v1[:3])-np.cross(q0[:3],item["mass"]*v0[:3])
    angular += item["inertia"]*(v1[3:]-v0[3:])
    expected_angular = np.cross(x0[free],h*total[free]+material[free]).sum(axis=0)
    expected_angular += np.cross(after.node.position[free]-x0[free],p1).sum(axis=0)
    expected_angular += np.cross(q0[:3],h*total_wrench[:3])+h*total_wrench[3:]
    expected_angular += np.cross(q1[:3]-q0[:3],item["mass"]*v1[:3])
    close(angular,expected_angular)
    close(observed["rigid_external_entry_power"],np.dot(item["external"],v0))
    work = potential_before = potential_after = dissipation = 0.0
    for site in observed["sites"]:
        beta = np.asarray(site["barycentric"],float)
        indices = np.asarray(site["node_indices"],int)
        force = np.asarray(site["law_output"][2:5],float)
        point = np.asarray(site["rigid_point"],float)
        node = np.asarray(site["node_point"],float)
        normal = np.asarray(site["normal"],float)
        # Advect the actual entry material addresses, not a new closest point.
        point_local = rotation(q0[3:]).T @ (point-q0[:3])
        axis_local = rotation(q0[3:]).T @ (point-RADIUS*normal-q0[:3])
        next_point = q1[:3]+rotation(q1[3:]) @ point_local
        next_axis = q1[:3]+rotation(q1[3:]) @ axis_local
        next_node = beta @ after.node.position[indices]
        work += float(force @ (next_node-node) - force @ (next_point-point))
        compression = max(WIDTH-site["gap"],0)
        potential_before += .5*STIFFNESS*compression**2
        potential_after += .5*STIFFNESS*max(WIDTH-(np.linalg.norm(next_axis-next_node)-RADIUS),0)**2
        rate,magnitude = np.asarray(site["law_output"][:2],float)
        dissipation += (STIFFNESS*compression-magnitude)*rate
    close(observed["contact_work"],work)
    close(observed["layer_potential_before"],potential_before)
    close(observed["layer_potential_after"],potential_after)
    close(observed["entry_dissipation_rate"],dissipation)
    close(observed["work_balance"],work+potential_after-potential_before+h*dissipation)
    close(observed["node_external_work"],0)
    power = sum(np.dot(r["force"],r["relative"]) for r in reference)
    storage_rate = sum(-STIFFNESS*r["compression"]*r["rate"] for r in reference)
    assert power+storage_rate <= 1e-10
    return dict(reference=reference,independent_force=node_force,independent_wrench=wrench,
        material_impulse=material,fixed_reaction=reaction,angular_change=angular,
        angular_expected=expected_angular,contact_power=power,storage_rate=storage_rate,
        contact_force_norm=float(np.linalg.norm(accumulated)))


def reference_initial(item):
    initial = item["root_initial"]
    return (item["node"].initial_state.position.astype(float).copy(),
            item["node"].initial_state.velocity.astype(float).copy(),
            np.asarray(initial.q,float).copy(),np.asarray(initial.v,float).copy())


def copy_references(values):
    return {world:tuple(array.copy() for array in state) for world,state in values.items()}


def advance(batch,items,references,journal,progress,identity):
    entry,revision,stats = batch.snapshot(),batch.revision,batch.stats()
    inputs = forces(items)
    progress["steps_attempted"] += 1
    journal.emit("step-attempt",identity=identity,entry=entry,expected_revision=revision,
                 inputs=inputs,h=H,stats=stats,reference_before=references)
    try:
        receipt = batch.step(inputs,H,expected_revision=revision)
        progress["steps_returned"] += 1
        progress["worldsteps"] += len(items)
        # Serialize actual return before snapshot, reference, or assertions.
        journal.emit("step-return",identity=identity,receipt=receipt)
        after,post = batch.snapshot(),batch.stats()
        journal.emit("step-state",identity=identity,state=after,stats=post)
        require(receipt.source_revision == revision and receipt.destination_revision == batch.revision,
                "common publication revision")
        require(batch.last_step is receipt,"last accepted common receipt")
        require(post["response"]["launches"] == stats["response"]["launches"]+2,
                "actual law and transpose kernel launches")
        audits,continued = {},{}
        for item in items:
            world = item["world"]
            audits[world] = audit_step(item,entry[world],after[world],receipt,H)
            x,v,q,rv = references[world]
            load,wrench,rows = reference_load(item,x,v,q,rv)
            x,v,_ = material_step(item,x,v,load,H)
            q,rv = rigid_step(item,q,rv,wrench,H)
            continued[world] = (x,v,q,rv)
            actual = root_state(after[world],item)
            close(after[world].node.position,x,atol=2e-4,rtol=0)
            close(after[world].node.velocity,v,atol=2e-4,rtol=0)
            close(actual.position[:3],q[:3],atol=2e-4,rtol=0)
            close(actual.velocity,rv,atol=2e-4,rtol=0)
            aq = np.asarray(actual.position[3:])
            require(min(np.linalg.norm(aq-q[3:]),np.linalg.norm(aq+q[3:])) <= 2e-4,
                    "independently continued orientation")
            require(after[world].node.tick == entry[world].node.tick+1 == after[world].rigid.tick,
                    "both physical owners advanced one step")
            progress["sites_checked"] += len(receipt.worlds[world]["sites"])
        journal.emit("step-audit",identity=identity,audits=audits,continuous_reference=continued)
        progress["steps_checked"] += 1
        return after,continued
    except BaseException as error:
        journal.emit("step-failure",identity=identity,error=error,last_attempt=batch.last_attempt)
        raise


def state_equal(a,b,items):
    for item in items:
        world = item["world"]
        for field in ("position","velocity","clock"):
            np.testing.assert_array_equal(getattr(a[world].node,field),getattr(b[world].node,field))
        require(a[world].node.tick == b[world].node.tick,"node replay tick")
        require(a[world].rigid.tick == b[world].rigid.tick,"rigid replay tick")
        require(root_state(a[world],item) == root_state(b[world],item),"full rigid replay capsule")


def run_batch(items,mode,journal,progress,*,restore_reset):
    batch = None
    primary,cleanup = None,[]
    try:
        journal.emit("construct-attempt",cases=[manifest(item) for item in items],
                     participants=tuple(item["participant"].key for item in items),
                     settings=SETTINGS,mode=mode,contact_capacity=2*len(items))
        batch = create(items,mode,contact_capacity=2*len(items))
        journal.emit("constructed",state=batch.snapshot(),stats=batch.stats())
        controls = {item["participant"].key:item["command"] for item in items}
        journal.emit("controls-attempt",controls=controls)
        batch.set_controls(controls)
        journal.emit("controls-return",checkpoint=batch.checkpoint())
        references = {item["world"]:reference_initial(item) for item in items}
        initial = batch.snapshot()
        first_checkpoint = first_reference = second_state = None
        peak_shape = 0.0
        for step in range(3):
            after,references = advance(batch,items,references,journal,progress,
                                      f"{items[0]['case']}-continuous-{step}")
            for item in items:
                points = after[item["world"]].node.position.astype(float)
                plane = np.cross(points[1]-points[0],points[2]-points[0])
                peak_shape = max(peak_shape,abs(float(plane @ (points[3]-points[0])))/np.linalg.norm(plane))
            if step == 0 and restore_reset:
                first_checkpoint,first_reference = batch.checkpoint(),copy_references(references)
                journal.emit("saved-checkpoint",checkpoint=first_checkpoint,
                             independent_reference=first_reference)
            if step == 1:
                second_state = after
        require(peak_shape > 1e-6,"actual nonplanar deformation during installed continuous steps")
        for item in items:
            world = item["world"]
            require(root_state(after[world],item).position != root_state(initial[world],item).position,
                    "real moving rigid body")
        if restore_reset:
            journal.emit("restore-attempt",checkpoint=first_checkpoint,
                         expected_revision=batch.revision)
            batch.restore(first_checkpoint,expected_revision=batch.revision)
            progress["restores"] += 1
            journal.emit("restore-return",state=batch.snapshot(),checkpoint=batch.checkpoint())
            replay,references = advance(batch,items,copy_references(first_reference),journal,progress,
                                        "PE-restored-successor")
            state_equal(replay,second_state,items)
            selected = items[0]["world"]
            journal.emit("reset-attempt",worlds=(selected,),expected_revision=batch.revision,
                         before=replay)
            batch.reset((selected,),expected_revision=batch.revision)
            progress["resets"] += 1
            reset = batch.snapshot()
            journal.emit("reset-return",state=reset,checkpoint=batch.checkpoint())
            state_equal(reset,initial,(items[0],))
            state_equal(reset,replay,items[1:])
            references[selected] = reference_initial(items[0])
            after,references = advance(batch,items,references,journal,progress,"PE-reset-successor")
            require(after[selected].node.tick == 1,"selected genuine next step")
            require(after[items[1]["world"]].node.tick == 3,"unselected continued third step")
            before,before_revision,old = batch.snapshot(),batch.revision,batch.last_step
            journal.emit("stale-revision-attempt",expected_revision=before_revision-1,
                         h=H,inputs=forces(items),before=before)
            try:
                batch.step(forces(items),H,expected_revision=before_revision-1)
            except (ValueError,RuntimeError) as error:
                progress["stale_rejections"] += 1
                journal.emit("stale-revision-rejected",error=error,state=batch.snapshot(),
                             last_attempt=batch.last_attempt)
            else:
                raise AssertionError("old common revision accepted")
            state_equal(before,batch.snapshot(),items)
            require(batch.revision == before_revision and batch.last_step is old,"stale attempt publication")
        journal.emit("batch-final",state=batch.snapshot(),checkpoint=batch.checkpoint(),
                     stats=batch.stats(),peak_nonplanarity=peak_shape,continuous_reference=references)
    except BaseException as error:
        primary = error
    finally:
        if batch is not None:
            try:
                journal.emit("close-attempt",stats=batch.stats())
            except BaseException as error:
                cleanup.append(error)
            try:
                batch.close()
                journal.emit("close-return")
            except BaseException as error:
                cleanup.append(error)
    if primary is not None:
        if cleanup:
            raise BaseExceptionGroup("D3 installed run and cleanup failed",[primary,*cleanup])
        raise primary
    if cleanup:
        raise BaseExceptionGroup("D3 installed cleanup failed",cleanup)


def run(mode,journal,progress):
    global ph, SETTINGS
    import phymeo as ph
    SETTINGS = ph.StepSettings(substeps=1,position_iterations=200,tolerance=2e-6)
    pair = (fixture("P",world=7),fixture("E",world=31))
    run_batch(pair,mode,journal,progress,restore_reset=True)
    run_batch((fixture("C",world=43),),mode,journal,progress,restore_reset=False)
    require(progress["steps_attempted"] == progress["steps_returned"] == progress["steps_checked"] == 8,
            "eight common accepted step calls")
    require(progress["worldsteps"] == 13,"thirteen actual world successors")
    require(progress["restores"] == progress["resets"] == progress["stale_rejections"] == 1,
            "complete checkpoint/reset/CAS control")
    return dict(steps=8,worldsteps=13,query_receipts=8,scope=
        "D3 entry-sampled mu0 discrete facet response; no CCD, friction or sustained-support qualification")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode",choices=("interpret","cuda"),required=True)
    parser.add_argument("--expected-install-root","--installed-root",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    require(args.output.is_absolute(),"absolute fresh output required")
    require(args.expected_install_root.is_absolute(),"absolute installed root required")
    root = args.expected_install_root.resolve(strict=True)
    args.output.mkdir(parents=True,exist_ok=False)
    journal = Journal(args.output/"deformable-contact.jsonl.gz")
    progress = dict(steps_attempted=0,steps_returned=0,steps_checked=0,worldsteps=0,
                    sites_checked=0,restores=0,resets=0,stale_rejections=0)
    summary = dict(status="RUNNING",mode=args.mode,
                   consumer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    errors = []
    def remember(stage,error):
        errors.append(error)
        summary["status"] = "FAIL"
        summary.setdefault("errors",[]).append(dict(stage=stage,type=type(error).__name__,
            message=str(error),traceback=traceback.format_exc()))
    try:
        summary["initial_provenance"] = provenance(root,preflight=True)
        summary.update(run(args.mode,journal,progress))
        summary["status"] = "PASS"
    except BaseException as error:
        remember("run",error)
    finally:
        summary["progress"] = progress
        for stage,callback in (
            ("final-provenance",lambda:summary.update(final_provenance=provenance(root))),
            ("journal-summary",lambda:journal.emit("summary",summary=summary)),
            ("journal-close",lambda:summary.update(trace=journal.close())),
        ):
            try:
                callback()
            except BaseException as error:
                remember(stage,error)
        try:
            (args.output/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False),encoding="utf-8")
        except BaseException as error:
            remember("summary-write",error)
    print(json.dumps(summary,allow_nan=False))
    if len(errors) == 1:
        raise errors[0]
    if errors:
        raise BaseExceptionGroup("D3 installed consumer/finalization failures",errors)


if __name__ == "__main__":
    main()