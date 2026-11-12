"""Standalone installed D1 producer consumer; no checkout imports or contact claim."""
import argparse
import base64
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
import gzip
import hashlib
import json
from pathlib import Path
import sys
import traceback
from unittest.mock import patch

import numpy as np

H = 1 / 128
KEYS = (7, 31)


def require(value, message):
    if not value:
        raise AssertionError(message)


def encode(value):
    if isinstance(value, np.ndarray):
        require(not value.dtype.hasobject, "numeric arrays only")
        return dict(dtype=value.dtype.str, shape=value.shape,
                    bytes_b64=base64.b64encode(value.tobytes()).decode())
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value):
        return dict(type=type(value).__module__ + "." + type(value).__name__,
                    fields={field.name: encode(getattr(value, field.name)) for field in fields(value)})
    if isinstance(value, Mapping):
        return dict(mapping_entries=[[encode(key), encode(item)] for key, item in value.items()])
    if isinstance(value, (tuple, list)):
        return [encode(item) for item in value]
    if type(value) is object:
        return dict(opaque_in_memory_owner=True)
    if value is None or type(value) in (str, bool, int, float):
        return value
    raise TypeError("unhandled evidence type " + type(value).__name__)


class Journal:
    def __init__(self, path):
        self.path = path
        self.stream = gzip.open(path, "xb")
        self.sha = hashlib.sha256()
        self.bytes = 0
        self.records = 0

    def emit(self, event, **payload):
        raw = (json.dumps(dict(event=event, payload=encode(payload)),
                          separators=(",", ":"), allow_nan=False) + "\n").encode()
        self.stream.write(raw)
        self.sha.update(raw)
        self.bytes += len(raw)
        self.records += 1

    def close(self):
        self.stream.close()
        return dict(logical_sha256=self.sha.hexdigest(), logical_bytes=self.bytes,
                    records=self.records, compressed_bytes=self.path.stat().st_size)


def provenance(root):
    import phymeo
    package = Path(phymeo.__file__).resolve()
    require(package.is_relative_to(root), "phymeo package is not from the expected install")
    modules = {}
    for name, module in sorted(sys.modules.items()):
        if name == "phymeo" or name.startswith("phymeo."):
            path = getattr(module, "__file__", None)
            if path is not None:
                path = Path(path).resolve()
                require(path.is_relative_to(root), "mixed source import: " + name)
                modules[name] = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return modules


def declarations():
    from phymeo.backends.tyx_deformable_batch import DeformableNodeWorld, DeformableSurface
    from phymeo.constraints.distance import DistanceConstraints
    from phymeo.core.model import ParticleModel
    from phymeo.core.state import ParticleState
    result = []
    for key in KEYS:
        square = key == 31
        x = np.array(((0,0,0),(1,0,0),(1,1,0),(0,1,0)) if square
                     else ((0,0,0),(1,0,0),(0,1,0)), np.float32)
        v = np.zeros_like(x)
        weights = np.ones(len(x), np.float32)
        if square:
            v[2:,0] = 1/16
            weights[0] = 0
            edges = np.array(((0,1),(1,2),(2,3),(3,0),(0,2)), np.int64)
            ids = np.array((101,113,127,139), np.int64)
            faces = np.array(((101,113,127),(101,127,139)), np.int64)
            facets = np.array((503,509), np.int64)
        else:
            v[:2,0] = (-1/16,1/16)
            edges = np.array(((0,1),), np.int64)
            ids = np.array((11,37,91), np.int64)
            faces = np.array(((11,37,91),), np.int64)
            facets = np.array((501,), np.int64)
        delta = x[edges[:,0]].astype(np.float64) - x[edges[:,1]]
        rest = np.linalg.norm(delta, axis=1)
        result.append(DeformableNodeWorld(
            key, ParticleModel(weights, np.zeros(3,np.float32)), ParticleState(x,v),
            DistanceConstraints(edges,rest,np.full(len(edges),H*H)), ids,
            (DeformableSurface(23,facets,faces),)))
    return tuple(result)


def host_step(position, velocity, model, constraints, force, sweeps):
    """Independent FP64 ordered XPBD recurrence; no device state is generated here."""
    weights = model.inverse_mass.astype(np.float64)
    dynamic = weights > 0
    pv = velocity + H * (model.gravity + weights[:,None]*force)
    px = position + H * pv
    px[~dynamic], pv[~dynamic] = position[~dynamic], 0
    x = px.copy()
    lam = np.zeros(len(constraints.indices))
    alpha = constraints.compliance / H / H
    for _ in range(sweeps):
        for row, (i,j) in enumerate(constraints.indices):
            delta = x[i] - x[j]
            length = np.linalg.norm(delta)
            normal = delta/length if length > 0 else np.array((1.,0.,0.))
            denominator = weights[i]+weights[j]+alpha[row]
            dl = (-(length-constraints.rest_length[row]+alpha[row]*lam[row])/denominator
                  if denominator > 0 else 0.)
            lam[row] += dl
            x[i] += weights[i]*dl*normal
            x[j] -= weights[j]*dl*normal
    v = pv+(x-px)/H
    v[~dynamic] = 0
    return x,v,lam,px,pv


def run(mode, journal, progress):
    from phymeo.backends.tyx_deformable_batch import TyxDeformableBatch
    from phymeo.core.scene import StepSettings
    from phymeo.core.state import StepInputs
    specs = declarations()
    settings = StepSettings(substeps=1, position_iterations=200, tolerance=2e-6)
    requirements = TyxDeformableBatch.requirements(specs,settings=settings,trace=True)
    oracle = {s.key:(s.initial_state.position.astype(float),s.initial_state.velocity.astype(float))
              for s in specs}
    original_lengths = {}
    for spec in specs:
        x = spec.initial_state.position.astype(float)
        original_lengths[spec.key] = np.linalg.norm(x[spec.constraints.indices[:,0]]
                                                   - x[spec.constraints.indices[:,1]],axis=1)
    cpu_calls = []
    def forbidden(*args, **kwargs):
        cpu_calls.append(True)
        raise AssertionError("HOST project_distances called by product step")
    with TyxDeformableBatch(specs,mode=mode,settings=settings,trace=True,
                           memory_limit_bytes=requirements.total_bytes) as owner:
        journal.emit("initial", declarations=specs, settings=settings, requirements=requirements,
                     state=owner.snapshot(), checkpoint=owner.checkpoint(), stats=owner.stats())
        def advance(tick):
            nonlocal oracle
            inputs = {}
            for spec in specs:
                force = np.zeros_like(spec.initial_state.position)
                if spec.key == 31:
                    force[:,1] = 1/32 if tick < 6 else -1/32
                inputs[spec.key] = StepInputs(force)
            before = owner.snapshot()
            counters = dict(owner.stats())
            progress["attempted"] += 1
            journal.emit("attempt", ordinal=progress["attempted"], source_revision=owner.revision,
                         input=inputs, h=H, state=before)
            with patch("phymeo.constraints.distance.project_distances",side_effect=forbidden):
                receipt = owner.step(inputs,H,expected_revision=owner.revision)
            progress["returned"] += 1
            journal.emit("raw-return",receipt=receipt,progress=dict(progress))
            after = owner.snapshot()
            journal.emit("accepted", receipt=receipt, state=after, stats=owner.stats())
            require(receipt.accepted and tuple(receipt.worlds)==KEYS,"complete atomic publication")
            require(owner.stats()["allocations"]==counters["allocations"],"no per-step allocation")
            require(owner.stats()["state_uploaded_bytes"]==counters["state_uploaded_bytes"],
                    "accepted device state was not uploaded again")
            independent = {}
            for spec in specs:
                key, row = spec.key, receipt.worlds[spec.key]
                same = host_step(before[key].position.astype(float),before[key].velocity.astype(float),
                                 spec.model,spec.constraints,inputs[key].force,row.sweeps)
                continuous = host_step(*oracle[key],spec.model,spec.constraints,inputs[key].force,row.sweeps)
                journal.emit("oracle-before-comparison",world=key,same_entry=same,continuous=continuous,
                             actual=after[key],row=row)
                np.testing.assert_allclose(after[key].position,same[0],rtol=2e-5,atol=2e-5)
                np.testing.assert_allclose(after[key].velocity,same[1],rtol=3e-3,atol=3e-3)
                np.testing.assert_allclose(row.multipliers,same[2],rtol=2e-5,atol=2e-5)
                np.testing.assert_allclose(row.predicted_position,same[3],rtol=2e-5,atol=2e-5)
                np.testing.assert_allclose(row.predicted_velocity,same[4],rtol=2e-5,atol=2e-5)
                np.testing.assert_allclose(after[key].position,continuous[0],rtol=2e-4,atol=2e-4)
                np.testing.assert_allclose(after[key].velocity,continuous[1],rtol=3e-3,atol=3e-3)
                require(row.residual<=settings.tolerance and len(row.trace)==row.sweeps,"full trace")
                require(after[key].tick==before[key].tick+1 and
                        after[key].time==before[key].time+H,"actual clock and tick")
                require(not after[key].position.flags.writeable,"immutable HOST publication")
                oracle[key] = continuous[:2]
                independent[key] = dict(same_entry=same,continuous=continuous)
            journal.emit("independent-checks", values=independent)
            progress["checked"] += 1
            return after

        for tick in range(12):
            advance(tick)
        for spec in specs:
            x = owner.snapshot()[spec.key].position.astype(float)
            lengths = np.linalg.norm(x[spec.constraints.indices[:,0]]-x[spec.constraints.indices[:,1]],axis=1)
            require(np.max(np.abs(lengths-original_lengths[spec.key]))>1e-6,"actual edge length changed")
        with owner.borrow_vertices(expected_revision=owner.revision) as loan:
            require(loan.world_keys==KEYS and loan.position(7).shape==(3,3),"accepted device vertex loan")
            journal.emit("loan", revision=loan.revision, active_bank=loan.active_bank,
                         point_ids={k:loan.point_ids(k) for k in KEYS},
                         surfaces={k:loan.surfaces(k) for k in KEYS})
            try:
                owner.reset((7,),expected_revision=owner.revision)
            except RuntimeError:
                pass
            else:
                raise AssertionError("mutation during loan accepted")
        try:
            loan.position(7)
        except RuntimeError:
            pass
        else:
            raise AssertionError("expired loan accepted")
        capsule = owner.checkpoint()
        saved_oracle = {k:tuple(a.copy() for a in v) for k,v in oracle.items()}
        branch_a = advance(12)
        owner.restore(capsule,expected_revision=owner.revision)
        oracle = saved_oracle
        branch_b = advance(12)
        for key in KEYS:
            for field in ("position","velocity","clock"):
                require(getattr(branch_a[key],field).tobytes()==getattr(branch_b[key],field).tobytes(),
                        "restored real successor differs")
        journal.emit("replay", checkpoint=capsule, first=branch_a, second=branch_b)
        untouched = owner.snapshot()[31]
        owner.reset((7,),expected_revision=owner.revision)
        require(owner.snapshot()[31] is untouched,"partial reset retained complete unselected state")
        oracle[7] = (specs[0].initial_state.position.astype(float),specs[0].initial_state.velocity.astype(float))
        journal.emit("partial-reset",state=owner.snapshot(),checkpoint=owner.checkpoint())
        advance(13)
        advance(14)
        final = owner.snapshot()
        bad = {s.key:StepInputs(np.zeros_like(s.initial_state.position)) for s in specs}
        bad[31].force[0,0] = np.nan
        revision = owner.revision
        try:
            owner.step(bad,H,expected_revision=revision)
        except ValueError as error:
            journal.emit("expected-rejection",error=str(error),input=bad,state=owner.snapshot())
        else:
            raise AssertionError("nonfinite force accepted")
        require(owner.revision==revision and owner.snapshot() is final,"rejection changed publication")
        journal.emit("final",states=final,stats=owner.stats(),checkpoint=owner.checkpoint())
    require(progress==dict(attempted=16,returned=16,checked=16) and not cpu_calls,"fixed actual counts")
    return dict(steps=16,worldsteps=32,worlds=KEYS,progress=progress,
                cpu_projection_calls=len(cpu_calls),scope="D1 node bank only; no query or contact response")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode",choices=("interpret","cuda"),required=True)
    parser.add_argument("--expected-install-root",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    require(args.output.is_absolute(),"absolute output required")
    args.output.mkdir(parents=True,exist_ok=False)
    import phymeo
    root = (args.expected_install_root.resolve(strict=True) if args.expected_install_root is not None
            else Path(phymeo.__file__).resolve().parent.parent)
    journal = Journal(args.output/"deformable-bank.jsonl.gz")
    progress = dict(attempted=0,returned=0,checked=0)
    summary = dict(status="RUNNING",mode=args.mode,
                   consumer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    errors=[]
    def remember(stage,error):
        errors.append(error)
        summary["status"]="FAIL"
        summary.setdefault("errors",[]).append(dict(stage=stage,type=type(error).__name__,
                                                    message=str(error),traceback=traceback.format_exc()))
    try:
        summary["initial_provenance"]=provenance(root)
        summary["result"]=run(args.mode,journal,progress)
        summary["worldsteps"]=summary["result"]["worldsteps"]
        summary["steps"]=summary["result"]["steps"]
        summary["status"]="PASS"
    except BaseException as error:
        remember("run",error)
    finally:
        summary["progress"]=progress
        try:
            summary["final_provenance"]=provenance(root)
        except BaseException as error:
            remember("final-provenance",error)
        try:
            journal.emit("summary",summary=summary)
        except BaseException as error:
            remember("journal-final",error)
        try:
            summary["trace"]=journal.close()
        except BaseException as error:
            remember("journal-close",error)
        try:
            (args.output/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False),encoding="utf-8")
        except BaseException as error:
            remember("summary-write",error)
    print(json.dumps(summary,allow_nan=False))
    if len(errors)==1:
        raise errors[0]
    if errors:
        raise BaseExceptionGroup("deformable consumer and finalization failures",errors)


if __name__=="__main__":
    main()
