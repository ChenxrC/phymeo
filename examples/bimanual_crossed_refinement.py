"""Cross the original PushT workpiece seeds with both physical arm configurations.

python examples/bimanual_crossed_refinement.py train --base CEM_RUN --source-dir CEM_SOURCES --out NEW --library VULKAN
python examples/bimanual_crossed_refinement.py evaluate --out NEW --library VULKAN
python examples/bimanual_crossed_refinement.py replay --out NEW --library VULKAN

The deployment directory contains the source-bound seven weights, the original
CEM package and example sources. The original Phymeo and device runtime remain required.
"""

import argparse
import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from xml.etree import ElementTree as ET

import numpy as np

import bimanual_policy_replay as frozen


ARMS = ((6, 7), (7, 6))
TRAIN = ((81001, 81002), (81003, 81004))
SELECT = (82001, 82002)
HOLDOUT = (83001, 83002, 83003, 83004)
FILM = (84001, 84002)
PARAMETERS = ("closure", "lift", "translation", "visual_correction", "rotation", "release", "damping")


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _file(path):
    return _sha(path.read_bytes())


def _read(path):
    return json.loads(path.read_bytes())


def _bind_bytes(path, data):
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"existing source or freeze artifact differs: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix+".tmp")
        tmp.write_bytes(data)
        tmp.replace(path)


def _candidates(center, g):
    rng = np.random.default_rng(81703+g)
    # Fixed, paired perturbations; the centre competes on the same four scenes.
    directions = np.clip(rng.normal(size=(2,7)), -1.5, 1.5) * (.26, .18)[g]
    return np.clip(np.vstack((center, center+directions, center-directions)), -1.5, 1.5)


def _scene_pair(env, seed):
    scenes, fixtures = {}, []
    for dofs in ARMS:
        scene = env.make_scene("push_t", seed, arm_dofs=dofs)
        xml = ET.fromstring(scene.xml)
        joints = [e.get("name") for e in xml.findall("./worldbody//joint")]
        motors = [e.get("joint") for e in xml.findall("./actuator/motor")]
        expected = [f"{s}_j{i}" for s,n in zip(("left","right"),dofs,strict=True) for i in range(n)]
        body = xml.find("./worldbody/body[@name='payload']")
        if (scene.xml_sha256 != _sha(scene.xml.encode()) or tuple(scene.arm_dofs) != dofs
                or joints != expected or motors != expected or body.find("freejoint") is None
                or len(body.findall("geom")) != 2 or any(e.get("type")!="box" for e in body.findall("geom"))
                or abs(sum(float(e.get("mass")) for e in body.findall("geom"))-9.)>1e-9
                or scene.mass != 9. or scene.timestep != .002):
            raise ValueError(f"physical arm counts, free 9 kg T or native step did not bind: {seed}/{dofs}")
        fixtures.append((ET.tostring(body), tuple(scene.start_xyz), tuple(scene.goal_xy), scene.goal_yaw,
                         ET.tostring(xml.find("./worldbody/site[@name='task_goal']"))))
        scenes[dofs] = scene
    if fixtures[0] != fixtures[1] or scenes[ARMS[0]].xml_sha256 == scenes[ARMS[1]].xml_sha256:
        raise ValueError("the two assignments must change the arm geometry and preserve the workpiece and target")
    return scenes


def _cases(args, trainer, config, env, stage, seeds, jobs, *, film=False):
    rows = {job["label"]: [] for job in jobs}
    for seed in seeds:
        for dofs, scene in _scene_pair(env, seed).items():
            where = args.out/("films" if film else "runs")/f"{dofs[0]}x{dofs[1]}"
            options = dict(config, out=where, library=args.library, device_index=args.device_index,
                           cache_dir=args.cache_dir, batch=args.batch, wall_seconds=None)
            run = trainer.Runner(SimpleNamespace(**options))
            def assigned(s):
                if s != seed:
                    raise ValueError("runner requested an unexpected seed")
                return dofs
            # The frozen v1 runner uses this function both in the actual model
            # constructor and in the receipt. Verify against independent XML.
            with patch.object(trainer, "_dofs", assigned):
                result = run.episodes(stage, seed, jobs, keep_trajectory=film)
            for job, row in zip(jobs, result, strict=True):
                receipt = where/"episodes"/stage/f"{seed}-{job['label']}.json"
                b, a = row["device_before"], row["device_after"]
                kernels = a["completed_kernels"]-b["completed_kernels"]
                if (row["job"] != job or row["seed"] != seed or row["scene_sha256"] != scene.xml_sha256
                        or row["final_metrics"]["xml_sha256"] != scene.xml_sha256
                        or tuple(row["arm_dofs"]) != dofs or row["physics"] != "musa"
                        or not all("S4000" in str(x).upper() for x in
                                   (row["renderer"]["native_device_name"], a["device_name"]))
                        or kernels <= 0 or kernels != a["launched_kernels"]-b["launched_kernels"]
                        or len(bytes.fromhex(row["sensor_rgb_sha256"])) != 32
                        or not 0 < row["decisions"] <= 200 or type(row["success"]) is not bool
                        or not np.isfinite(row["return_sum"])):
                    raise ValueError(f"physical episode or real wrist evidence differs: {receipt}")
                item = dict(seed=seed, arm_dofs=list(dofs), goal_yaw_rad=scene.goal_yaw, success=row["success"],
                            return_sum=row["return_sum"], decisions=row["decisions"], kernels=kernels,
                            model_sha256=scene.xml_sha256, rgb_sha256=row["sensor_rgb_sha256"],
                            receipt=str(receipt.relative_to(args.out)), receipt_sha256=_file(receipt))
                if film:
                    item["film"] = frozen._film_files(where, seed,
                        dict(arms="both", weights=job["weights"], task="push_t"), config, scene)
                rows[job["label"]].append(item)
    return rows


def _score(cases):
    return sum(r["success"] for r in cases), float(np.mean([r["return_sum"] for r in cases]))


def _best(cases, labels):
    # Python max retains the earlier candidate on an exact tie (centre or base).
    return max(labels, key=lambda name: _score(cases[name]))


def load_frozen(deployment):
    """Load the selected policy after checking the selection, fitting and deployment bindings."""
    cert = _read(deployment/"binding.json")
    for name, digest in cert["sha256"].items():
        if _file(deployment/name) != digest:
            raise ValueError(f"frozen deployment mismatch: {name}")
    policy, history = _read(deployment/"crossed-policy.json"), _read(deployment/"selection.json")
    base = _read(deployment/"base"/"frozen-policy.json")
    centre = np.array(base["weights"], dtype=float)
    for i,g in enumerate(history["generations"]):
        candidates = _candidates(centre,i)
        labels = [f"c{k}" for k in range(len(candidates))]
        winner = _best(g["episodes"], labels)
        if g["winner"] != winner or not np.array_equal(candidates, g["population"]):
            raise ValueError("generation does not match its registered population or measured elite")
        centre = candidates[labels.index(winner)]
    chosen = _best(history["selection"], ("base","candidate"))
    weights = base["weights"] if chosen=="base" else centre.tolist()
    mode = "original-cem-fallback" if np.array_equal(weights, base["weights"]) else "crossed-elite-refinement"
    if (not np.array_equal(policy["weights"], weights) or policy["mode"] != mode
            or history["selected"] != chosen or policy["parameters"] != list(PARAMETERS)):
        raise ValueError("frozen policy is inconsistent with paired validation")
    return policy


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=("train","evaluate","replay"))
    p.add_argument("--base", type=Path, help="original frozen CEM directory or frozen-policy.json")
    p.add_argument("--source-dir", type=Path, help="original digest-matched examples")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--generations", type=int, choices=(1,2))
    p.add_argument("--batch", type=int, choices=(1,2,3,4,5))
    p.add_argument("--cache-dir", type=Path, default=Path("artifacts/bimanual-musa/cache"))
    p.add_argument("--library", type=Path)
    p.add_argument("--device-index", type=int)
    args = p.parse_args(argv)
    dep = args.out/"deployment"
    package = args.base or dep/"base"
    policy_path = (package/"frozen-policy.json" if package.is_dir() else package).resolve()
    if not policy_path.is_file():
        p.error("train with --base pointing at the extracted original PushT policy")
    checkpoint = policy_path.with_name("checkpoint.json")
    base, config, updates, base_sha, cp_sha = frozen._load(policy_path,checkpoint)
    if ((base["task"],base["arms"],base.get("variant"),config["mode"]) != ("push_t","both","registered","musa")
            or (config["image_size"],config["action_repeat"],config["horizon"]) != (96,8,200)
            or config["parameters"] != list(PARAMETERS)):
        p.error("expected the original 7-parameter, 96 px, 200-decision registered bilateral MUSA PushT policy")
    directory, trainer = frozen._sources(config["sources"],policy_path,args.source_dir,base["variant"])
    env = importlib.import_module("bimanual_musa_env")
    protocol_path = args.out/"protocol.json"
    previous = _read(protocol_path) if protocol_path.exists() else {}
    n = args.generations or previous.get("generations",2)
    args.batch = args.batch or previous.get("batch",2)
    protocol = dict(schema="crossed-pusht-elite-v1", base_sha256=base_sha, checkpoint_sha256=cp_sha,
                    sources=config["sources"], driver_sha256=_file(Path(__file__)), replay_sha256=_file(Path(frozen.__file__)),
                    generations=n, batch=args.batch, physical_step_s=.002, decision_steps=8, horizon=200,
                    arms=[list(a) for a in ARMS], train=[list(s) for s in TRAIN[:n]], selection=list(SELECT),
                    holdout=list(HOLDOUT), film=list(FILM), candidates=5, std=[.26,.18][:n],
                    bounds=[-1.5,1.5], ranking="success count, mean return, earlier centre/base on ties")
    historical = {s for u in updates for s in u["seeds"]} | set(config["selection_seeds"]) | set(config["holdout_seeds"])
    if historical.intersection((*TRAIN[0],*TRAIN[1],*SELECT,*HOLDOUT,*FILM)):
        raise ValueError("registered partitions overlap the original training package")
    if previous and previous != protocol:
        raise ValueError("existing refinement protocol differs")
    if not previous:
        if args.command != "train" or (args.out.exists() and any(args.out.iterdir())):
            p.error("begin with train in an empty output directory")
        trainer._save(protocol_path,protocol)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    if args.command == "train" and not (dep/"binding.json").exists():
        centre = np.array(base["weights"],dtype=float)
        generations = []
        for i, seeds in enumerate(TRAIN[:n]):
            candidates = _candidates(centre,i)
            jobs = [trainer._job(f"c{k}", weights) for k,weights in enumerate(candidates)]
            cases = _cases(args,trainer,config,env,f"train-{i+1}",seeds,jobs)
            labels = [j["label"] for j in jobs]
            win = _best(cases,labels)
            generations.append(dict(population=candidates.tolist(), winner=win, episodes=cases))
            centre = candidates[labels.index(win)]
            trainer._save(args.out/f"generation-{i+1}.json",generations[-1])
        jobs = [trainer._job("base",base["weights"]), trainer._job("candidate",centre)]
        selection = _cases(args,trainer,config,env,"selection",SELECT,jobs)
        choice = _best(selection, ("base","candidate"))
        weights = base["weights"] if choice=="base" else centre.tolist()
        mode = "original-cem-fallback" if np.array_equal(weights,base["weights"]) else "crossed-elite-refinement"
        files = {"base/frozen-policy.json": policy_path, "base/checkpoint.json": checkpoint,
                 Path(__file__).name: Path(__file__), Path(frozen.__file__).name: Path(frozen.__file__),
                 "protocol.json": protocol_path}
        files.update({f"base/sources/{name}": directory/name for name in config["sources"]})
        for name,src in files.items():
            _bind_bytes(dep/name,src.read_bytes())
        trainer._save(dep/"selection.json",dict(generations=generations, selection=selection, selected=choice))
        trainer._save(dep/"crossed-policy.json",dict(schema="crossed-pusht-policy-v1", task="push_t", arms="both",
                      variant="registered", parameters=list(PARAMETERS), weights=weights,
                      mode=mode, base_sha256=base_sha, source_sha256=config["sources"]))
        trainer._save(dep/"binding.json",dict(sha256={name:_file(dep/name) for name in
                      (*files,"crossed-policy.json","selection.json")}))
    if not (dep/"binding.json").exists():
        p.error("run train to completion before using the frozen policy")
    policy = load_frozen(dep)
    if _read(dep/"protocol.json") != protocol:
        raise ValueError("current experiment differs from the frozen deployment protocol")
    print(f"selected {policy['mode']}",flush=True)
    if args.command == "train":
        return 0
    if args.command == "evaluate":
        jobs = [trainer._job("base",base["weights"]),trainer._job("selected",policy["weights"])]
        rows = _cases(args,trainer,config,env,"holdout",HOLDOUT,jobs)
        trainer._save(args.out/"holdout.json", dict(binding_sha256=_file(dep/"binding.json"), mode=policy["mode"],
                      scores={k:_score(v) for k,v in rows.items()}, episodes=rows,
                      paired_return_change=[s["return_sum"]-b["return_sum"] for s,b in zip(rows["selected"],rows["base"],strict=True)]))
    else:
        rows = _cases(args,trainer,config,env,"replay",FILM,[trainer._job("frozen",policy["weights"])],film=True)
        trainer._save(args.out/"films.json",dict(binding_sha256=_file(dep/"binding.json"), mode=policy["mode"], episodes=rows["frozen"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
