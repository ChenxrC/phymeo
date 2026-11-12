"""D3 tests first: actual entry forces, both candidates, independent mechanics.

Predeclared in p65-deformable-response-cases.md. AUTHORING_UNEXECUTED.
The parent owns every collection / project execution.
"""

import gzip
import importlib.util
import json
import math
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import fields, is_dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

import phymeo as ph

H = 1 / 1024
WIDTH, STIFFNESS, DAMPING, RADIUS = 1 / 16, 128.0, 1 / 2, 1 / 8
ATOL, RTOL = 5e-6, 5e-5
IDS = np.array((101, 113, 127, 139), np.int64)
FACES = np.array(((0, 1, 2), (0, 2, 3)), np.int64)
FACETS = np.array((503, 509), np.int64)
EDGES = np.array(((0, 1), (1, 2), (2, 3), (3, 0), (0, 2)), np.int64)
REST = np.array((1, 1, 1, 1, math.sqrt(2)))
COMPLIANCE = 1 / 16384
SETTINGS = ph.StepSettings(substeps=1, position_iterations=200, tolerance=2e-6)


@lru_cache(maxsize=1)
def finite_reference():
    path = Path(__file__).parents[1] / "reference_newton/rr/device_triangle_distance_reference.py"
    spec = importlib.util.spec_from_file_location("d3_independent_finite", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


@contextmanager
def saved(path, name, mode, **inputs):
    data = dict(schema="d3-response-evidence-v1", identity=name, mode=mode, inputs=inputs,
                attempts=[], complete=False)
    try:
        yield data
        data["complete"] = True
    except BaseException as error:
        data["failure"] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        with gzip.open(path / (name + ".json.gz"), "wt", encoding="utf-8") as stream:
            json.dump(encode(data), stream, sort_keys=True, allow_nan=False)


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
        "local","mass","inertia","materials","external")}

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
        exact = finite_reference().oracle(x[indices],segment)
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
        length = finite_reference().local_scale(x0[indices],
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
    power = sum(np.dot(r["force"],r["relative"]) for r in reference)
    storage_rate = sum(-STIFFNESS*r["compression"]*r["rate"] for r in reference)
    assert power+storage_rate <= 1e-10
    return dict(reference=reference,independent_force=node_force,independent_wrench=wrench,
        material_impulse=material,fixed_reaction=reaction,angular_change=angular,
        angular_expected=expected_angular,contact_power=power,storage_rate=storage_rate,
        contact_force_norm=float(np.linalg.norm(accumulated)))


def attempt(batch, items, h, evidence):
    record = dict(entry=batch.snapshot(),inputs=forces(items),h=h,revision=batch.revision,stats_before=batch.stats())
    evidence["attempts"].append(record)
    try:
        result = batch.step(record["inputs"],h,expected_revision=record["revision"])
        record["returned"] = result
        record["state"] = batch.snapshot()
        record["stats"] = batch.stats()
        assert record["stats"]["response"]["launches"] > record["stats_before"]["response"]["launches"]
        record["audit"] = {i["world"]:audit_step(i,record["entry"][i["world"]],
            record["state"][i["world"]],result,h) for i in items}
        assert result.source_revision == record["revision"]
        assert result.destination_revision == batch.revision
        assert batch.last_step is result
        return result,record
    except BaseException as error:
        record["error"] = dict(type=type(error).__name__,message=str(error))
        record["last_attempt"] = batch.last_attempt
        raise


@pytest.mark.tyx
@pytest.mark.parametrize("case",("P","E","C"))
@pytest.mark.parametrize("variant",("base","fine","mass"))
def test_three_physical_cases_continuous_reference(case,variant,tyx_mode,tmp_path):
    item = fixture(case,factor=4 if variant == "mass" else 1)
    h,count = (H/2,32) if variant == "fine" else (H,16)
    with saved(tmp_path,"response-"+case+"-"+variant,tyx_mode,
               case=manifest(item),h=h,steps=count) as evidence, create((item,),tyx_mode) as batch:
        batch.set_controls({item["participant"].key:item["command"]})
        initial = batch.snapshot()[7]
        root = root_state(initial,item)
        x,v = initial.node.position.astype(float),initial.node.velocity.astype(float)
        q,rv = np.asarray(root.position,float),np.asarray(root.velocity,float)
        nonplanarity,edge_change = 0.0,0.0
        for tick in range(count):
            # Independent trajectory is advanced from its OWN previous result.
            load,wrench,ref = reference_load(item,x,v,q,rv)
            result,record = attempt(batch,(item,),h,evidence)
            row = result.node_receipt.worlds[7]
            x,v,_ = material_step(item,x,v,load,h)
            q,rv = rigid_step(item,q,rv,wrench,h)
            state = record["state"][7]
            actual_root = root_state(state,item)
            close(state.node.position,x,atol=2e-4,rtol=0)
            close(state.node.velocity,v,atol=2e-4,rtol=0)
            close(actual_root.position[:3],q[:3],atol=2e-4,rtol=0)
            close(actual_root.velocity,rv,atol=2e-4,rtol=0)
            actual_q = np.asarray(actual_root.position[3:])
            assert min(np.linalg.norm(actual_q-q[3:]),np.linalg.norm(actual_q+q[3:])) <= 2e-4
            record["continuous_reference"] = dict(x=x.copy(),v=v.copy(),q=q.copy(),velocity=rv.copy())
            points = state.node.position.astype(float)
            n = np.cross(points[1]-points[0],points[2]-points[0])
            nonplanarity = max(nonplanarity,abs(np.dot(n,points[3]-points[0]))/np.linalg.norm(n))
            edge_change = max(edge_change,float(np.max(abs(np.linalg.norm(points[EDGES[:,0]]-points[EDGES[:,1]],axis=1)-REST))))
            if tick == 0:
                close(np.linalg.norm(result.worlds[7]["node_contact_force"].sum(axis=0)),33/8)
                if case == "P":
                    close(result.worlds[7]["rigid_wrench"][0,3:],0)
                elif case == "E":
                    close(result.worlds[7]["rigid_wrench"][0,4],-33/32)
                    close(result.worlds[7]["node_contact_force"][1,2],-165/64)
                    assert actual_root.velocity[4] < 0
                else:
                    close(result.worlds[7]["rigid_wrench"][0,5],33/64)
                    assert actual_root.position[1] > root.position[1]
        evidence["physical_change"] = dict(nonplanarity=nonplanarity,edge_change=edge_change)
        assert nonplanarity > 1e-6
        assert edge_change > 0
        assert batch.snapshot()[7].node.tick == count


@pytest.mark.tyx
@pytest.mark.parametrize("variant",("outside","opening"))
def test_zero_layer_load_is_real_motion_not_a_frozen_state(variant,tyx_mode,tmp_path):
    item = fixture("P",variant=variant)
    with saved(tmp_path,"zero-"+variant,tyx_mode,case=manifest(item)) as evidence, create((item,),tyx_mode) as batch:
        initial = batch.snapshot()[7]
        for _ in range(3):
            result,_ = attempt(batch,(item,),H,evidence)
            close(result.worlds[7]["node_contact_force"],0,atol=0,rtol=0)
            close(result.worlds[7]["rigid_wrench"],0,atol=0,rtol=0)
        assert root_state(batch.snapshot()[7],item).position != root_state(initial,item).position


@pytest.mark.tyx
def test_same_trajectory_enters_and_leaves_layer_under_declared_external_load(tyx_mode,tmp_path):
    item = fixture("P",variant="entry-exit")
    with saved(tmp_path,"entry-exit",tyx_mode,case=manifest(item),h=H,steps=16) as evidence, create((item,),tyx_mode) as batch:
        batch.set_controls({item["participant"].key:item["command"]})
        magnitudes,gaps = [],[]
        for _ in range(16):
            result,record = attempt(batch,(item,),H,evidence)
            magnitudes.append(float(np.linalg.norm(result.worlds[7]["node_contact_force"].sum(axis=0))))
            gaps.append(min(r["gap"] for r in record["audit"][7]["reference"]))
        evidence["layer_events"] = dict(entry_gaps=gaps,force_magnitudes=magnitudes)
        assert gaps[0] > WIDTH and magnitudes[0] == 0
        assert any(f > 1e-5 for f in magnitudes[1:-1])
        assert gaps[-1] > WIDTH and magnitudes[-1] == 0
        assert batch.snapshot()[7].node.tick == 16
