"""Cross every new object/goal seed with both arm layouts using the frozen evaluation runner.

python examples/bimanual_layout_evaluation.py --policy <training-dir> --source-dir <frozen-sources> \
    --out <new-evaluation-dir> --library <Vulkan-library> --device-index 0

Run once per frozen task. An interrupted invocation resumes committed, hash-verified episodes.
"""

import argparse
import hashlib
import importlib
import json
import random
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from xml.etree import ElementTree as ET

import numpy as np

from bimanual_policy_replay import MODEL_DEFAULTS, _film_files, _load, _sources


SEEDS = tuple(range(70001,70009))
LAYOUTS = ((6,7),(7,6))
ORDER_SEED = 976841


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _name(seed,dofs):
    return f"{seed}-L{dofs[0]}R{dofs[1]}"


def _fixture(scene, xml_file, q):
    import mujoco as mj

    xml = xml_file.read_bytes()
    root = ET.fromstring(xml)
    world = root.find("worldbody")
    model = mj.MjModel.from_xml_string(xml.decode())
    if model.na or model.nmocap:
        raise ValueError("expected stateless actuators and a fully dynamic scene")
    actual = []
    for side in ("left","right"):
        joints = [model.joint(i).name for i in range(model.njnt) if model.joint(i).name.startswith(side+"_j")]
        if joints != [f"{side}_j{i}" for i in range(len(joints))]:
            raise ValueError("unexpected arm joint order")
        actual.append(len(joints))
    if tuple(actual)!=tuple(scene.arm_dofs):
        raise ValueError("physical XML and expected arm layout differ")
    expected = model.qpos0
    if getattr(scene,"stance","original")=="bent-elbow":
        key = mj.mj_name2id(model,mj.mjtObj.mjOBJ_KEY,"bent-elbow-start")
        if key<0:
            raise ValueError("expected the registered bent-elbow start keyframe")
        expected = model.key_qpos[key]
    if not np.all((q==expected) | (q==expected.astype(np.float32))):
        raise ValueError("actual first published q differs from the independently compiled source pose")
    data = mj.MjData(model)
    data.qpos[:] = q
    mj.mj_kinematics(model,data)
    def site(name):
        s = model.site(name).id
        return dict(xyz=data.site_xpos[s].tolist(),rotation=data.site_xmat[s].reshape(3,3).tolist())
    goal = site("task_goal")
    rotation = np.array(goal["rotation"])
    yaw = float(np.arctan2(rotation[1,0],rotation[0,0]))
    if not np.allclose(goal["xyz"][:2],scene.goal_xy,rtol=0,atol=2e-8) or abs(yaw-scene.goal_yaw)>2e-8:
        raise ValueError("compiled world-frame goal differs from the independent generator")
    payload = model.jnt_qposadr[model.joint("payload_free").id]
    stationary = [ET.tostring(c) for c in world if c.tag!="body" or c.get("name")=="payload"]
    names = [f"{s}_{site_name}" for s in ("left","right") for site_name in ("tool","wrist_camera")]
    return dict(physical_arm_dofs=actual,initial_payload_q=q[payload:payload+7].tolist(),
                initial_tool_and_camera_poses={n:site(n) for n in names},goal=goal,goal_yaw_rad=yaw,
                payload_table_and_goal_xml_sha256=_sha(b"\0".join(stationary)))


def _verify_pair(left,right):
    for k in ("initial_payload_q","goal","payload_table_and_goal_xml_sha256"):
        if left[k]!=right[k]:
            raise ValueError(f"matched-layout runs differ in {k}")
    for name,a in left["initial_tool_and_camera_poses"].items():
        b = right["initial_tool_and_camera_poses"][name]
        if not all(np.allclose(a[k],b[k],rtol=0,atol=2e-6) for k in ("xyz","rotation")):
            raise ValueError(f"matched-layout initial frames differ at {name}")


def _run_episode(args,trainer,envmod,cammod,policy,contract,folder,seed,dofs,protocol_hash):
    variant = policy.get("variant","centroid")
    options = {k:contract[k] for k in MODEL_DEFAULTS if k in contract}
    scene = envmod.make_scene(policy["task"],seed,arm_dofs=dofs,**options)
    frames, eyes = [],[]
    combined = hashlib.sha256()
    streams = [hashlib.sha256(),hashlib.sha256()]
    class StateRecorder(envmod.BimanualBatchEnv):
        def _states(self):
            states = super()._states()
            s = states[0]
            frames.append((int(s.tick),float(s.time),s.q.copy(),s.v.copy(),s.ctrl.copy()))
            return states
    class WristRecorder(cammod.BimanualWristRenderer):
        def observe(self,qpos,tick,time_s,**kwargs):
            images = super().observe(qpos,tick,time_s,**kwargs)
            if images.dtype!=np.uint8 or images.shape!=(2,contract["image_size"],contract["image_size"],3):
                raise ValueError("unexpected physical wrist format")
            hashes = []
            for stream,img in zip(streams,images,strict=True):
                raw = np.ascontiguousarray(img).tobytes()
                hashes.append(_sha(raw))
                stream.update(raw)
                combined.update(raw)
            eyes.append((int(tick),hashes))
            return images
    def assigned(input_seed):
        if input_seed!=seed:
            raise ValueError("runner requested a different environment seed")
        return dofs
    run = SimpleNamespace(**dict(contract,out=folder,batch=1,wall_seconds=None,policy_variant=variant,
                          **{k:contract.get(k,v) for k,v in MODEL_DEFAULTS.items()}),
                          library=args.library,device_index=args.device_index,cache_dir=args.cache_dir)
    job = trainer._job("frozen",policy["weights"],policy["arms"])
    # These scoped bindings run synchronously in this CLI's process. The recorder
    # classes return the exact state/images supplied by the frozen implementation.
    with (patch.object(trainer,"_dofs",assigned),
          patch.object(envmod,"BimanualBatchEnv",StateRecorder),
          patch.object(cammod,"BimanualWristRenderer",WristRecorder)):
        row, = trainer.Runner(run).episodes("replay",seed,[job],keep_trajectory=True)
    files = _film_files(folder,seed,policy,contract,scene)
    ticks,times,q,v,ctrl = map(np.asarray,zip(*frames,strict=True))
    eye_ticks = np.asarray([p[0] for p in eyes])
    eye_sha = np.asarray([p[1] for p in eyes])
    source = folder/files["files"]["npz"]
    with np.load(source,allow_pickle=False) as episode:
        decisions = episode["ticks"]
        if (not np.array_equal(ticks,np.arange(row["physics_ticks"]+1))
                or not np.all(np.diff(times)>0) or not np.array_equal(eye_ticks,decisions[:-1])
                or not np.array_equal(q[decisions],episode["qpos"])
                or not np.array_equal(times[decisions],episode["time_s"])
                or any(not np.isfinite(z).all() for z in (q,v,ctrl,times))
                or v.shape!=(len(ticks),q.shape[1]-1) or ctrl.shape!=(len(ticks),sum(dofs))
                or np.any(v[0]) or np.any(ctrl[0]) or combined.hexdigest()!=row["sensor_rgb_sha256"]):
            raise ValueError("physics/native wrist records failed their decision-time bindings")
    fixture = _fixture(scene,folder/files["files"]["xml"],q[0])
    state_path = folder/"physics-and-wrists.npz"
    np.savez_compressed(state_path,ticks=ticks,time_s=times,qpos=q,qvel=v,ctrl=ctrl,
                        eye_input_ticks=eye_ticks,eye_rgb_sha256=eye_sha,scene_sha256=scene.xml_sha256)
    d0,d1 = row["device_before"],row["device_after"]
    if (d1["backend"]!="musa" or "S4000" not in d1["device_name"].upper()
            or d1["completed_kernels"]<=d0["completed_kernels"] or d1["completed_kernels"]!=d1["launched_kernels"]):
        raise ValueError("missing complete S4000 kernel evidence")
    metrics = row["final_metrics"]
    result = dict(protocol_sha256=protocol_hash,seed=seed,layout=list(dofs),fixture=fixture,
                  success=row["success"],return_sum=row["return_sum"],decisions=row["decisions"],
                  physics_ticks=row["physics_ticks"],final_metrics=metrics,
                  wrist_stream_sha256=[s.hexdigest() for s in streams],combined_wrist_sha256=combined.hexdigest(),
                  files=files["files"]|{"states":state_path.name},
                  file_sha256=files["file_sha256"]|{"states":_sha(state_path.read_bytes())})
    trainer._save(folder/"result.json",result)
    return result


def _summary(rows):
    cells = {}
    for r in rows:
        a,b = r["layout"]
        yaw = round(np.rad2deg(r["fixture"]["goal_yaw_rad"]),4)
        for key in (f"L{a}R{b}",f"L{a}R{b} yaw {yaw:g} deg"):
            c = cells.setdefault(key,dict(trials=0,successes=0,return_sum=0.))
            c["trials"] += 1
            c["successes"] += int(r["success"])
            c["return_sum"] += r["return_sum"]
    return dict(completed=len(rows),total=16,cells=cells,
                failures=[dict(seed=r["seed"],layout=r["layout"],return_sum=r["return_sum"],
                               final_metrics=r["final_metrics"]) for r in rows if not r["success"]])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policy",type=Path,required=True)
    p.add_argument("--checkpoint",type=Path)
    p.add_argument("--source-dir",type=Path)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--cache-dir",type=Path,default=Path("artifacts/bimanual-musa/cache"))
    p.add_argument("--library",type=Path)
    p.add_argument("--device-index",type=int)
    args = p.parse_args(argv)
    path = (args.policy/"frozen-policy.json" if args.policy.is_dir() else args.policy).resolve()
    policy,contract,updates,policy_hash,cp_hash = _load(path,args.checkpoint or path.with_name("checkpoint.json"))
    if policy["arms"]!="both" or contract["mode"]!="musa" or policy["task"] not in ("carry","push_t"):
        p.error("expected a completed bilateral MUSA policy for carry or push_t")
    old = set(contract["holdout_seeds"])|set(contract["selection_seeds"])
    old.update(s for update in updates for s in update["seeds"])
    if old.intersection(SEEDS):
        p.error("evaluation seeds overlap the original learning or testing partitions")
    directory,trainer = _sources(contract["sources"],path,args.source_dir,policy.get("variant","centroid"))
    envmod,cammod = (importlib.import_module(n) for n in ("bimanual_musa_env","bimanual_wrist_render"))
    for mod in (envmod,cammod):
        if Path(mod.__file__).resolve()!=directory/(mod.__name__+".py"):
            raise ValueError("sources changed during evaluation setup")
    order = [(s,layout) for s in SEEDS for layout in LAYOUTS]
    random.Random(ORDER_SEED).shuffle(order)
    protocol = dict(schema="crossed-frozen-bimanual-layouts-v1", task=policy["task"],arms="both",
                    adapter_sha256=_sha(Path(__file__).read_bytes()),policy_sha256=policy_hash,
                    checkpoint_sha256=cp_hash,contract_sha256=policy["contract_sha256"],sources=contract["sources"],
                    order_seed=ORDER_SEED,order=[dict(seed=s,layout=list(x)) for s,x in order])
    protocol_hash = _sha(json.dumps(protocol,sort_keys=True,separators=(",",":")).encode())
    args.out.mkdir(parents=True,exist_ok=True)
    protocol_path = args.out/"protocol.json"
    if protocol_path.is_file():
        if json.loads(protocol_path.read_bytes())!=protocol:
            raise ValueError("output belongs to a different registered evaluation")
    elif any(args.out.iterdir()):
        p.error("use an empty output directory or resume the identical protocol")
    trainer._save(protocol_path,protocol)
    args.cache_dir.mkdir(parents=True,exist_ok=True)
    rows,paired = [],{}
    for seed,dofs in order:
        name = _name(seed,dofs)
        dest = args.out/name
        if dest.exists():
            r = json.loads((dest/"result.json").read_bytes())
            scene = envmod.make_scene(policy["task"],seed,arm_dofs=dofs,
                                      **{k:contract[k] for k in MODEL_DEFAULTS if k in contract})
            check = _film_files(dest,seed,policy,contract,scene)
            if (r["protocol_sha256"]!=protocol_hash or r["layout"]!=list(dofs) or r["seed"]!=seed
                    or check["success"]!=r["success"] or check["return_sum"]!=r["return_sum"]
                    or any(_sha((dest/f).read_bytes())!=r["file_sha256"][k] for k,f in r["files"].items())):
                raise ValueError("existing episode does not match its model, policy or artifact hashes")
            with np.load(dest/r["files"]["states"],allow_pickle=False) as archive:
                if _fixture(scene,dest/r["files"]["xml"],archive["qpos"][0])!=r["fixture"]:
                    raise ValueError("existing paired-scene evidence differs from the first measured state")
        else:
            staging = Path(tempfile.mkdtemp(prefix=f".pending-{name}-",dir=args.out))
            try:
                r = _run_episode(args,trainer,envmod,cammod,policy,contract,staging,seed,dofs,protocol_hash)
                if seed in paired:
                    _verify_pair(paired[seed],r["fixture"])
                staging.rename(dest)
            except Exception as exc:
                trainer._save(staging/"error.json",dict(seed=seed,layout=list(dofs),error=str(exc)))
                raise
        if seed in paired:
            _verify_pair(paired[seed],r["fixture"])
        paired[seed] = r["fixture"]
        rows.append(r)
        trainer._save(args.out/"evaluation.json",dict(protocol_sha256=protocol_hash,
                      episodes=[_name(x["seed"],x["layout"])+"/result.json" for x in rows],**_summary(rows)))


if __name__=="__main__":
    main()
