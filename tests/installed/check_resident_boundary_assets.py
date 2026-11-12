"""Source-external M1-A original asset boundary consumer; two worlds, three real steps.

Uses only installed product APIs. Original materials, candidates, accepted source binding
and raw TYX packets are saved; physics advances the accepted bank. HOST geometry is
used only by the independent result oracle, never as device source. No source test /
example import occurs. Caller prepares the installed wheel and dependency path.
"""

import argparse
import base64
import gzip
import hashlib
import json
import struct
import sys
import traceback
from collections.abc import Mapping
from dataclasses import fields, is_dataclass, replace
from pathlib import Path

import numpy as np

SCHEMA = "phymeo-installed-resident-boundary-assets-v1"
DISTANCE_GATE = 1e-4
WITNESS_GATE = 2e-4
NORMAL_GATE = 2e-3


def require(condition, message):
    if not condition:
        raise AssertionError(message)


_TOKENS = {}


def encode(value):
    if isinstance(value, np.ndarray):
        require(not value.dtype.hasobject, "object array in evidence")
        return dict(
            dtype=value.dtype.str,
            shape=list(value.shape),
            bytes_b64=base64.b64encode(value.tobytes()).decode(),
        )
    if isinstance(value, np.generic):
        return dict(dtype=value.dtype.str, scalar_b64=base64.b64encode(value.tobytes()).decode())
    if is_dataclass(value):
        return dict(
            type=type(value).__module__ + "." + type(value).__name__,
            fields={f.name: encode(getattr(value, f.name)) for f in fields(value)},
        )
    if isinstance(value, Mapping):
        return dict(mapping_entries=[[encode(k), encode(v)] for k, v in value.items()])
    if isinstance(value, tuple):
        return dict(tuple=[encode(v) for v in value])
    if isinstance(value, list):
        return [encode(v) for v in value]
    if type(value) is float:
        return dict(float64_bits=struct.pack(">d", value).hex())
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if type(value) is object:
        _TOKENS.setdefault(id(value), len(_TOKENS))
        return dict(opaque_token=_TOKENS[id(value)])
    raise TypeError("unhandled evidence value " + type(value).__name__)


class Journal:
    def __init__(self, path):
        self.path = path
        self.stream = gzip.open(path, "xb")
        self.sha = hashlib.sha256()
        self.logical_bytes = self.events = 0

    def emit(self, event, **payload):
        record = dict(schema=SCHEMA, ordinal=self.events, event=event, payload=encode(payload))
        line = (json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n").encode()
        self.stream.write(line)
        self.stream.flush()
        self.sha.update(line)
        self.logical_bytes += len(line)
        self.events += 1

    def close(self):
        self.stream.close()
        return dict(
            path=str(self.path),
            events=self.events,
            logical_bytes=self.logical_bytes,
            logical_sha256=self.sha.hexdigest(),
            compressed_bytes=self.path.stat().st_size,
            compressed_sha256=hashlib.sha256(self.path.read_bytes()).hexdigest(),
        )


def provenance(installed_root, *, preflight=False):
    def source_entry(path):
        return ((path / "pyproject.toml").is_file() and (path / "src" / "phymeo").is_dir()) or (
            path.name.lower() == "src"
            and (path / "phymeo").is_dir()
            and (path.parent / "pyproject.toml").is_file()
        )

    require(
        (installed_root / "phymeo" / "__init__.py").is_file(), "installed package root required"
    )
    paths = [Path(entry or Path.cwd()).resolve() for entry in sys.path]
    require(not any(source_entry(path) for path in paths), "checkout source entry in sys.path")
    if preflight:
        require(
            not any(name == "phymeo" or name.startswith("phymeo.") for name in sys.modules),
            "Phymeo imported before preflight",
        )
        for origin in (Path.cwd().resolve(), Path(__file__).resolve().parent, installed_root):
            require(
                not any(source_entry(parent) for parent in (origin, *origin.parents)),
                "consumer/cwd/install must be outside checkouts",
            )
    import phymeo

    require(
        Path(phymeo.__file__).resolve().is_relative_to(installed_root), "installed phymeo origin"
    )
    loaded = {}
    for name, module in sorted(sys.modules.items()):
        if not (name == "phymeo" or name.startswith("phymeo.")):
            continue
        filename = getattr(module, "__file__", None)
        if filename is not None:
            actual = Path(filename).resolve(strict=True)
            require(actual.is_relative_to(installed_root), "noninstalled product module " + name)
            loaded[name] = dict(
                path=str(actual), sha256=hashlib.sha256(actual.read_bytes()).hexdigest()
            )
        for directory in getattr(module, "__path__", ()):
            require(
                Path(directory).resolve(strict=True).is_relative_to(installed_root),
                "noninstalled package search path " + name,
            )
    return dict(
        installed_root=str(installed_root),
        cwd=str(Path.cwd().resolve()),
        consumer=str(Path(__file__).resolve()),
        sys_path=[str(path) for path in paths],
        loaded=loaded,
        numpy=dict(version=np.__version__, path=str(Path(np.__file__).resolve())),
    )


H = 1 / 1024
WORLDS = (7, 4096)
VERTICES = np.array(((0,0,0),(0,2,0),(0,0,2),(-1,0,0)),np.float64)
FACES = ((0,1,2),(0,2,3),(0,3,1),(1,3,2))
SPHERE = np.array(((-.25,.5,.5),),np.float64)
RADIUS = 1/8
MARGIN = .5


def exact(left, right, message):
    require(encode(left) == encode(right), message)


def immutable(value):
    if isinstance(value, np.ndarray):
        require(not value.flags.writeable, "mutable published array")
        try:
            value.setflags(write=True)
        except ValueError:
            pass
        else:
            raise AssertionError("published array permits writeable reenable")
    elif is_dataclass(value):
        for field in fields(value):
            immutable(getattr(value, field.name))
    elif isinstance(value, Mapping):
        for item in value.values():
            immutable(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            immutable(item)


def make_scene(output):
    import phymeo as ph

    # Original F0 mechanics and an actual OBJ asset, never a hull relabeled mesh.
    obj = "".join("v " + " ".join(str(float(x)) for x in row) + "\n" for row in VERTICES)
    obj += "".join("f " + " ".join(str(i+1) for i in face) + "\n" for face in FACES)
    (output / "surface.obj").write_text(obj,encoding="utf-8")
    xml = '''<robot name="m1_asset_f0">
<link name="base"><inertial><mass value="2"/><inertia ixx=".2" iyy=".3" izz=".4" ixy="0" ixz="0" iyz="0"/></inertial></link>
<link name="left_jaw"><inertial><mass value="1"/><inertia ixx=".0625" iyy=".078125" izz=".09375" ixy="0" ixz="0" iyz="0"/></inertial><collision name="surface"><geometry><mesh filename="surface.obj"/></geometry></collision></link>
<joint name="driver" type="prismatic"><parent link="base"/><child link="left_jaw"/><origin xyz="-.25 0 0"/><axis xyz="1 0 0"/><limit lower="-.1" upper=".1" effort="1.5" velocity=".2"/></joint>
<link name="right_jaw"><inertial><mass value="2"/><inertia ixx=".09375" iyy=".109375" izz=".125" ixy="0" ixz="0" iyz="0"/></inertial><collision name="round"><origin xyz="-.25 .5 .5"/><geometry><sphere radius=".125"/></geometry></collision></link>
<joint name="follower" type="prismatic"><parent link="base"/><child link="right_jaw"/><origin xyz=".25 0 0"/><axis xyz="1 0 0"/><limit lower="-.1" upper=".1" effort=".125" velocity=".08"/><mimic joint="driver" multiplier="-1" offset=".004"/></joint></robot>'''
    path=output / "input-f0.urdf"
    path.write_text(xml,encoding="utf-8")
    asset=ph.load_mimic_urdf(path,gravity=np.zeros(3)).tree
    require(asset.joint_names==("driver","follower"),"original full F0 joint identities")
    surface=next(c for c in asset.collisions if c.name=="surface")
    require(surface.shape.kind=="mesh" and surface.shape.triangles==FACES,"original OBJ facets")
    np.testing.assert_array_equal(surface.local_vertices,VERTICES)
    trees,initial={},{}
    for index,world in enumerate(WORLDS):
        q=np.array(((index+1)/512,.004-(index+1)/512),np.float64)
        v=np.array(((index+1)/1024,-(index+1)/1024),np.float64)
        program=ph.DrivenRobotProgram(asset.model,kp=np.zeros(2),kd=np.zeros(2),
                                     effort_limit=np.ones(2),delay_steps=0)
        trees[world]=ph.FixedRobotTree(program,program.initial_state(ph.RobotState(q,v)))
        initial[world]=dict(q=q,v=v)
    inputs=dict(xml=xml,obj=obj,path=str(path),
                xml_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                obj_sha256=hashlib.sha256((output/"surface.obj").read_bytes()).hexdigest())
    return trees,initial,inputs,asset


def register(asset):
    import phymeo as ph

    shapes,bindings=[],[]
    for world in WORLDS:
        for name,body,ordinal in (("surface",101,0),("round",202,1)):
            source=next(c.shape for c in asset.collisions if c.name==name)
            key=ph.CollisionKey(world,body,ordinal)
            shapes.append(replace(source,key=key))
            binding=ph.ResidentTreeBinding.from_asset(asset,source.key,key=key,world=world)
            require(binding.topology is None,"factory provenance, not fabricated direct topology")
            bindings.append(binding)
    return ph.CollisionModel(tuple(shapes)),tuple(bindings)


def closest_point(point,triangle):
    """Independent complete finite point/triangle candidates for this OBJ oracle.

    The interior orthogonal projection and all three closed edge projections are
    compared. No device candidate, product query or product geometry is reused.
    """
    a,b,c=triangle
    e,f=b-a,c-a
    ee,ef,ff=float(e@e),float(e@f),float(f@f)
    pe,pf=float((point-a)@e),float((point-a)@f)
    determinant=ee*ff-ef*ef
    require(determinant>0,"authored finite nondegenerate triangle")
    beta=(pe*ff-pf*ef)/determinant
    gamma=(pf*ee-pe*ef)/determinant
    candidates=[]
    if beta>=0 and gamma>=0 and beta+gamma<=1:
        weights=np.array((1-beta-gamma,beta,gamma))
        candidates.append((weights@triangle,weights))
    for first,second in ((0,1),(1,2),(2,0)):
        edge=triangle[second]-triangle[first]
        t=min(1.,max(0.,float((point-triangle[first])@edge)/float(edge@edge)))
        weights=np.zeros(3); weights[first]=1-t; weights[second]=t
        candidates.append((triangle[first]+t*edge,weights))
    nearest,weights=min(candidates,key=lambda item:float((item[0]-point)@(item[0]-point)))
    distance=float(np.linalg.norm(nearest-point))
    require(distance>0,"fixed consumer remains away from zero-direction band")
    return distance,nearest,weights,(nearest-point)/distance


def verify(packet,states,initial,tick,bindings):
    import phymeo as ph

    require(packet.schema=="phymeo-resident-boundary-distances-v1","M1 aggregate schema")
    require(packet.source_kind=="resident-rigid-device-v1","accepted device geometry source")
    require(packet.semantics=="threshold-limited-boundary-distance","boundary-only semantics")
    require(packet.world_keys==WORLDS and packet.margin==MARGIN,"world/margin binding")
    np.testing.assert_array_equal(packet.ticks,(tick,tick))
    require(len(packet.groups)==2 and packet.required_candidates==8,"complete two-world facets")
    require(len(packet.pairs)==2 and all(p.status=="boundary_candidates" for p in packet.pairs),
            "all within-world original pairs")
    require(len(packet.source_binding)==64,"full source binding")
    immutable(packet)
    definitions={item["key"]:item for item in packet.source["bindings"]}
    for binding in bindings:
        definition=definitions[binding.key]
        require(definition["admission"]=="asset","asset admission branch recorded")
        proof=definition["provenance"]
        require(proof["schema"]=="phymeo-boundary-asset-admission-v1","asset source schema")
        require(proof["binding_family"]=="ordinary" and proof["graph_token"] is None,
                "honest ordinary factory provenance")
        require(proof["source_shape"]==binding._source_shape,"full original asset source")
        require(proof["source_shape"].key!=binding.key,"original and registered keys distinguished")
        require(proof["model_token"]==binding._model_token,"original FP64 model binding")
        require(proof["physical_topology"]==definition["topology"] and definition["topology"] is not None,
                "validated full physical topology, not original None")
        require(proof["group_index"]==binding.group_index,"actual material owner group")
        np.testing.assert_array_equal(proof["local_points_m"],binding.local_points_m)
    checks={}
    for group in packet.groups:
        world=group.surface.world
        state=states[world]
        q,v=np.asarray(state.position),np.asarray(state.velocity)
        np.testing.assert_allclose(q,initial[world]["q"]+tick*H*initial[world]["v"],atol=3e-6,rtol=0)
        np.testing.assert_allclose(v,initial[world]["v"],atol=3e-6,rtol=0)
        require(state.tick==tick and np.any(v!=0),"three actual nonzero-velocity steps")
        surface=VERTICES+np.array((-.25+q[0],0,0))
        axis=SPHERE[0]+np.array((.25+q[1],0,0))
        anchor=np.asarray(packet.source["world_anchors"][world])
        np.testing.assert_array_equal(group.candidates,np.array(((0,0),(0,1),(0,2),(0,3)),np.int64))
        require(group.facet_vertices==FACES,"all original OBJ material facets")
        require(group.round_keys==(ph.CollisionKey(world,202,1),),"original sphere ownership")
        require(group.bvh_raw.shape==(1,4) and group.bvh_raw.dtype==np.int32,"complete K+3 BVH data")
        require(group.bvh_raw[0,0]==15 and np.all(group.bvh_raw[0,1:3]>=1),"all four selected bits and traversal")
        require(group.bvh_raw[0,3]==1,"all selected source rows valid")
        narrow=group.packet
        require(narrow.schema=="phymeo-resident-triangle-distance-v1","resident packed packet")
        require(narrow.source_binding==packet.source_binding,"same accepted source for every facet")
        np.testing.assert_array_equal(narrow.origin_m,anchor)
        require(len(narrow.rows)==4,"no facet result truncation")
        for name,shape,dtype in (("candidates",(4,18,11),np.float32),("valid",(4,18),np.int32),
                                 ("summary",(4,16),np.float32),("meta",(4,7),np.int32)):
            require(narrow.raw[name].shape==shape and narrow.raw[name].dtype==dtype,"complete raw "+name)
        np.testing.assert_array_equal(narrow.packed["owners"],group.candidates)
        facet_checks=[]
        for ordinal,row in enumerate(narrow.rows):
            face=FACES[ordinal]
            require(row.status=="RESOLVED" and row.segment_id==0 and row.facet_id==ordinal
                    and row.vertex_ids==face,"original complete finite ownership")
            triangle=surface[list(face)]
            length=max(float(np.linalg.norm(triangle[j]-triangle[i])) for i,j in ((0,1),(1,2),(2,0)))
            length=max(length,RADIUS)
            distance,nearest,weights,normal=closest_point(axis,triangle)
            require(abs(row.distance-distance)<=DISTANCE_GATE*length,"independent distance")
            require(abs(row.gap-(distance-RADIUS))<=DISTANCE_GATE*length,"original radius law")
            require(0<=row.axis_parameter<=1,"finite sphere's zero-length axis address")
            np.testing.assert_allclose(row.barycentric,weights,atol=WITNESS_GATE,rtol=0)
            np.testing.assert_allclose(np.asarray(row.axis_point_relative_m)+anchor,axis,
                                       atol=WITNESS_GATE*length,rtol=0)
            np.testing.assert_allclose(np.asarray(row.triangle_point_relative_m)+anchor,nearest,
                                       atol=WITNESS_GATE*length,rtol=0)
            np.testing.assert_allclose(row.normal,normal,atol=NORMAL_GATE,rtol=0)
            facet_checks.append(dict(facet=face,distance=distance,axis=axis,triangle=nearest,
                                     barycentric=weights,normal=normal,local_length=length))
        for key,expected in ((group.surface,surface),(group.round_keys[0],axis.reshape(1,3))):
            np.testing.assert_allclose(packet.source["gathered_points"][key],expected-anchor,
                                       atol=WITNESS_GATE*np.sqrt(8.),rtol=0)
        checks[world]=facet_checks
    require(packet.counters["live_bytes"]==packet.counters["persistent_bytes"],"temporary cleanup complete")
    require(packet.counters["launches"]==packet.counters["completed_launches"],"all submissions completed")
    if packet.backend=="cuda":
        require(packet.counters["downloads"]>0 and packet.counters["downloaded_bytes"]>0,"real transfers counted")
    return checks


def run(mode, journal, installed_root, output):
    import phymeo as ph

    owners = []
    attempts = []
    primary = None
    try:
        trees, initial, asset, source_asset = make_scene(output)
        journal.emit(
            "fixture",
            asset=asset,
            initial=initial,
            worlds=WORLDS,
            surface=VERTICES,
            facets=FACES,
            sphere=SPHERE,
            radius=RADIUS,
            margin=MARGIN,
            h=H,
        )
        batch = ph.TyxWorldBatch(list(trees.items())[::-1], mode, cache_root=output / "cache")
        owners.append(("batch", batch))
        model, bindings = register(source_asset)
        journal.emit(
            "registration",
            model=model,
            bindings=bindings,
            initial=batch.snapshot(),
            batch_stats=batch.stats(),
        )
        plan = batch.prepare_boundary_query(
            model, bindings, limits=ph.BoundaryDistanceLimits(), memory_limit_bytes=32 * 1024 * 1024
        )
        owners.append(("plan", plan))
        last = None
        for tick in (1, 2, 3):
            before = batch.snapshot()
            step_result = batch.step(H)
            states = batch.snapshot()
            journal.emit(
                "physical-step",
                tick=tick,
                h=H,
                before=before,
                result=step_result,
                after=states,
                batch_stats=batch.stats(),
            )
            require(all(step_result.values()), "real accepted physical step")
            if last is not None:
                rejection = dict(tick=tick, previous_sequence=last.sequence)
                try:
                    plan.require_current(last)
                except ValueError as error:
                    rejection.update(
                        status="rejected", error_type=type(error).__name__, message=str(error)
                    )
                else:
                    rejection.update(status="unexpected-success")
                journal.emit("stale-query-binding", **rejection)
                require(
                    rejection["status"] == "rejected",
                    "physical source change invalidates old packet",
                )
            attempt = dict(tick=tick, before=states, before_stats=plan.stats(), margin=MARGIN)
            attempts.append(attempt)
            query_error = None
            try:
                result = plan.query(margin=MARGIN)
                attempt["result"] = result
            except BaseException as error:
                query_error = error
                attempt["query_error"] = dict(
                    type=type(error).__name__, message=str(error), traceback=traceback.format_exc()
                )
            capture_errors = []
            for name, reader in (
                ("after", batch.snapshot),
                ("last_attempt", lambda: plan.last_attempt),
                ("last_result", lambda: plan.last_result),
                ("after_stats", plan.stats),
            ):
                try:
                    attempt[name] = reader()
                except BaseException as error:
                    capture_errors.append(error)
                    attempt[name + "_error"] = dict(
                        type=type(error).__name__,
                        message=str(error),
                        traceback=traceback.format_exc(),
                    )
            try:
                # Snapshot may be forbidden after UNKNOWN; retain the query's original error
                # and independently save every still-accessible raw/ownership diagnostic.
                journal.emit("query-outcome", **attempt)
            except BaseException as error:
                capture_errors.append(error)
            if query_error is not None:
                if capture_errors:
                    raise BaseExceptionGroup(
                        "query and outcome capture failed", [query_error, *capture_errors]
                    )
                raise query_error
            if capture_errors:
                raise BaseExceptionGroup("query outcome capture failed", capture_errors)
            exact(states, attempt["after"], "query cannot mutate accepted physical bank")
            require(plan.require_current(result) is result, "new current source accepted")
            checks = verify(result, states, initial, tick, bindings)
            journal.emit("query-checks", tick=tick, checks=checks)
            last = result
        modules = provenance(installed_root)
        journal.emit(
            "completed",
            accepted_world_steps=6,
            complete_queries=3,
            full_rows=24,
            modules=modules,
            plan_stats=plan.stats(),
            batch_stats=batch.stats(),
        )
        return dict(
            accepted_world_steps=6,
            complete_queries=3,
            rows=24,
            stale_rejections=2,
            final_ticks=last.ticks,
            modules=modules,
        )
    except BaseException as error:
        primary = error
        try:
            journal.emit(
                "attempt-failure",
                error_type=type(error).__name__,
                message=str(error),
                complete_attempt_prefix=attempts,
                owners=tuple(
                    dict(
                        label=label,
                        stats=owner.stats(),
                        last=getattr(owner, "last_result", None),
                        attempt=getattr(owner, "last_attempt", None),
                    )
                    for label, owner in owners
                ),
            )
        except BaseException as log_error:
            primary = BaseExceptionGroup(
                "original attempt and evidence-write errors", [error, log_error]
            )
        raise primary
    finally:
        failures = []
        for label, owner in reversed(owners):
            outcome = dict(label=label, status="closed")
            try:
                owner.close()
            except BaseException as error:
                failures.append(error)
                outcome.update(
                    status="close-failed", error_type=type(error).__name__, message=str(error)
                )
            try:
                journal.emit("owner-close", **outcome)
            except BaseException as error:
                failures.append(error)
        if failures:
            raise BaseExceptionGroup(
                "M1 consumer/cleanup errors (original retained)",
                ([primary] if primary is not None else []) + failures,
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("interpret", "cuda"), required=True)
    parser.add_argument("--installed-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(
        args.output.is_absolute() and args.installed_root.is_absolute(), "absolute paths required"
    )
    root = args.installed_root.resolve(strict=True)
    output = args.output.resolve()
    require(
        not any(
            (parent / "pyproject.toml").is_file() and (parent / "src" / "phymeo").is_dir()
            for parent in (output, *output.parents)
        ),
        "output must be outside checkouts",
    )
    output.mkdir(parents=True, exist_ok=False)
    journal = Journal(output / "resident-boundary-assets.jsonl.gz")
    summary = dict(
        status="RUNNING",
        mode=args.mode,
        schema=SCHEMA,
        consumer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope="installed original OBJ and URDF material provenance to resident device BVH and finite triangle distance",
        excluded="solid inside/outside, global nearest outside band, CCD, load response, deformable source bank",
        gates=dict(
            distance_over_local_L=DISTANCE_GATE,
            witness_over_local_L=WITNESS_GATE,
            normal=NORMAL_GATE,
        ),
    )
    primary = None
    try:
        summary["initial_provenance"] = provenance(root, preflight=True)
        result = run(args.mode, journal, root, output)
        summary["result"] = encode(result)
        summary["status"] = "PASS"
    except BaseException as error:
        primary = error
        summary["status"] = "FAIL"
        summary["exception"] = dict(
            type=type(error).__name__, message=str(error), traceback=traceback.format_exc()
        )
        try:
            journal.emit("failure", summary=summary)
        except BaseException as log_error:
            primary = BaseExceptionGroup(
                "original consumer and failure-write errors", [error, log_error]
            )
    final_errors = []
    try:
        summary["final_provenance"] = provenance(root)
    except BaseException as error:
        summary["status"] = "FAIL"
        summary["provenance_error"] = traceback.format_exc()
        final_errors.append(error)
    try:
        journal.emit("summary", summary=summary)
    except BaseException as error:
        final_errors.append(error)
    try:
        summary["trace"] = journal.close()
    except BaseException as error:
        final_errors.append(error)
    try:
        if final_errors:
            summary["status"] = "FAIL"
        (output / "summary.json").write_text(
            json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
        )
        print(json.dumps(summary, allow_nan=False))
    except BaseException as error:
        final_errors.append(error)
    if primary is not None or final_errors:
        errors = ([primary] if primary is not None else []) + final_errors
        if len(errors) == 1:
            raise errors[0]
        raise BaseExceptionGroup(
            "installed consumer and finalization errors (original retained)", errors
        )


if __name__ == "__main__":
    main()
