"""Paired on-policy S4000 experiment on top of a digest-bound frozen bimanual controller.

train/evaluate both require --base (original completed training directory), --out,
and optionally --source-dir, --library, --device-index, --cache-dir.
For a host-only feature, update and checkpoint test use the self-test command.
"""

import argparse
import hashlib
import json
import os
import tempfile
from contextlib import ExitStack
from pathlib import Path

import numpy as np


HOLDOUT = tuple(range(63001, 63009))
DEVELOPMENT = (62001, 62002)
CONDITIONS = ("learned", "base_only", "frozen_init", "both_black", "first_frame", "residual_black")
ARCH = "pooled-rgb-proprio-tanh8-residual4-v1"


def _hash(data):
    return hashlib.sha256(data).hexdigest()


def _save(path, obj):
    data = json.dumps(obj, sort_keys=True, allow_nan=False, indent=2).encode()
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as f:
        tmp = Path(f.name)
        f.write(data)
    try:
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def features(rgb, proprio):
    if rgb.ndim != 4 or rgb.shape[0] != 2 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8 or min(rgb.shape[1:3]) < 3:
        raise ValueError("expected two actual HWC uint8 wrist frames")
    q = np.asarray(proprio, float)
    if q.shape != (42,) or not np.isfinite(q).all():
        raise ValueError("expected 14 joint positions, 14 velocities and 14 active flags")
    colours = rgb.astype(np.float32)/255.
    r, g, b = np.moveaxis(colours,-1,0)
    channels = np.concatenate((colours, np.maximum(r-g,0.)[...,None], np.maximum(g-b,0.)[...,None]), axis=-1)
    tiles = [np.stack([col.mean(axis=(1,2)) for col in np.array_split(row,3,axis=2)],axis=1)
             for row in np.array_split(channels,3,axis=1)]
    return np.r_[np.stack(tiles,axis=1).ravel(), np.clip(q[:14]/1.5,-2,2),
                 np.clip(q[14:28]/3.,-2,2), q[28:]]


class ResidualNet:
    """Eight fixed random nonlinear features and a 32-weight learned output head."""

    def __init__(self, encoder, bias, head):
        self.encoder, self.bias, self.head = (np.asarray(v,float) for v in (encoder,bias,head))
        for a, shape in zip((self.encoder,self.bias,self.head), ((132,8),(8,),(8,4)), strict=True):
            if a.shape != shape or not np.isfinite(a).all():
                raise ValueError("invalid residual network weights")

    @classmethod
    def initial(cls):
        rng = np.random.default_rng(51439)
        return cls(rng.normal(size=(132,8))/np.sqrt(132.), rng.normal(0,.12,8), rng.normal(0,.035,(8,4)))

    def with_head(self, head):
        return ResidualNet(self.encoder, self.bias, head)

    def predict(self, rgb, proprio):
        return np.tanh(np.tanh(features(rgb,proprio) @ self.encoder + self.bias) @ self.head)

    def save(self, path, contract):
        _save(path, dict(architecture=ARCH, contract=contract, encoder=self.encoder.tolist(),
                         bias=self.bias.tolist(), head=self.head.tolist()))

    @classmethod
    def load(cls, path, contract):
        doc = json.loads(path.read_bytes())
        if doc["architecture"] != ARCH or doc["contract"] != contract:
            raise ValueError("residual checkpoint and frozen base contract differ")
        return cls(doc["encoder"], doc["bias"], doc["head"])


def _action_delta(actor, latent, torque):
    forward, vertical, squeeze, twist = latent
    action = np.zeros((2,7))
    for i, ((_,va,_),(tool,_)) in enumerate(zip(actor.joints, actor.sites, strict=True)):
        sign = 1 if i == 0 else -1
        jp = np.zeros((3,actor.model.nv))
        actor.mj.mj_jacSite(actor.model,actor.data,jp,None,tool)
        j = jp[:,va]
        motion = np.array((forward-sign*twist, -sign*squeeze, vertical))
        if torque:
            tau = j.T @ (4.*motion)
            start = sum(len(x[0]) for x in actor.joints[:i])
            lo, hi = actor.model.actuator_ctrlrange[start:start+len(va)].T
            delta = tau / np.where(tau>=0,hi,-lo)
        else:
            # Frozen incremental policies use normalized joint displacement.
            delta = j.T @ np.linalg.solve(j@j.T + .002*np.eye(3), .0018*motion) / .045
        action[i,:len(va)] = np.clip(delta,-.12,.12)
    return action.ravel()


def _update(head, directions, differences):
    differences = np.asarray(differences,float)
    if not np.isfinite(differences).all():
        raise ValueError("invalid paired returns")
    gradient = np.einsum("i,ijk->jk",differences,directions)
    return np.clip(head + .07*gradient / max(1e-8, np.abs(differences).sum()),-.9,.9)


def rollout(trainer, base, config, args, seed, jobs):
    """All lanes use one native physics step per substep and their own actual wrist capture."""
    import inspect
    from bimanual_musa_env import BimanualBatchEnv
    from bimanual_wrist_render import BimanualWristRenderer

    variant = base.get("variant","centroid")
    torque = variant == "fiducial"
    keys = tuple(range(len(jobs)))
    options = dict(arm_dofs=trainer._dofs(seed), world_keys=keys, mode="musa", cache_root=args.cache_dir,
                   action_repeat=config["action_repeat"], horizon=config["horizon"])
    for k in ("stance","tool_profile","load_profile"):
        if k in inspect.signature(BimanualBatchEnv).parameters and k in config:
            options[k] = config[k]
    with ExitStack() as stack:
        env = stack.enter_context(BimanualBatchEnv(base["task"],seed=seed,**options))
        packets = env.reset(seed=seed, active_arms={k:base["arms"] for k in keys})
        cameras = dict(library_path=args.library,size=(config["image_size"],)*2,device_index=args.device_index)
        if "fiducials" in inspect.signature(BimanualWristRenderer).parameters:
            cameras["fiducials"] = torque
        visual = stack.enter_context(BimanualWristRenderer(env.model,args.cache_dir/"camera",**cameras))
        if "S4000" not in str(visual.device_report.get("native_device_name","")).upper():
            raise RuntimeError("wrist images must come from the native S4000 Vulkan renderer")
        actors = [trainer.VisualPolicy(base["task"],env.model,base["weights"],variant=variant,
                  decision_dt=config["action_repeat"]*env.scene.timestep,
                  servo_gains=(env.worlds[k].kp,env.worlds[k].kd)) for k in keys]
        start = env.runtime.musa_device_report()
        if start["backend"] != "musa" or "S4000" not in start["device_name"].upper():
            raise RuntimeError(f"expected S4000 physics, got {start['device_name']}")
        first, done = {}, set()
        initial = ResidualNet.initial()
        rows = [dict(condition=jobs[k][0],rewards=[],action_delta_l1_sum=0.,action_delta_max=0.,
                     changed_decisions=0, learned_vs_initial_peak=0., learned_vs_initial_changed=0,
                     scene_sha256=env.scene.xml_sha256) for k in keys]
        hashes = [[hashlib.sha256() for _ in range(4)] for _ in keys]
        for _ in range(config["horizon"]):
            actions = {k:np.zeros(14) for k in keys}
            for k in keys:
                if k in done:
                    continue
                obs, info = packets[k]
                actual = visual.observe(info["qpos"],info["tick"],info["time_s"],epoch=k)
                if k not in first:
                    first[k] = actual.copy()
                condition, net = jobs[k]
                image = (np.zeros_like(actual) if condition == "both_black" else
                         first[k] if condition == "first_frame" else actual)
                net_img = np.zeros_like(image) if condition == "residual_black" else image
                kwargs = dict(wrist_force_world_n=info["wrist_force_world_n"]) if torque else {}
                frozen = actors[k].act(obs,image,**kwargs)
                latent = net.predict(net_img,obs)
                residual = np.zeros(14) if condition == "base_only" else _action_delta(actors[k],latent,torque)
                actions[k] = np.asarray(obs[28:])*np.clip(frozen+residual,-1.,1.)
                diff = np.abs(actions[k]-frozen)
                rows[k]["changed_decisions"] += int(diff.max()>1e-7)
                rows[k]["action_delta_l1_sum"] += float(diff.sum())
                rows[k]["action_delta_max"] = max(rows[k]["action_delta_max"],float(diff.max()))
                if condition == "learned":
                    # Same physical state, same base, same RGB, same actuator
                    # saturation. This isolates the trained weights' action effect.
                    old_delta = _action_delta(actors[k],initial.predict(net_img,obs),torque)
                    old_action = np.asarray(obs[28:])*np.clip(frozen+old_delta,-1.,1.)
                    change = float(np.abs(actions[k]-old_action).max())
                    rows[k]["learned_vs_initial_peak"] = max(rows[k]["learned_vs_initial_peak"],change)
                    rows[k]["learned_vs_initial_changed"] += int(change>1e-7)
                hashes[k][0].update(np.ascontiguousarray(actual).tobytes())
                hashes[k][1].update(np.asarray(actions[k],dtype="<f8").tobytes())
                hashes[k][2].update(np.ascontiguousarray(image).tobytes())
                hashes[k][3].update(np.ascontiguousarray(net_img).tobytes())
            post = env.step_torque(actions) if torque else env.step(actions)
            for k in keys:
                if k in done:
                    continue
                obs,reward,terminated,truncated,info = post[k]
                packets[k] = obs,info
                rows[k]["rewards"].append(float(reward))
                if terminated or truncated:
                    done.add(k)
                    rows[k].update(success=bool(info["success"]),final_xy_error_m=float(info["goal_error_xy_m"]),
                                   final_yaw_error_rad=float(info["goal_yaw_error_rad"]), ticks=int(info["tick"]))
            if len(done)==len(keys):
                break
        end = env.runtime.musa_device_report()
        if (len(done)!=len(keys) or end["completed_kernels"] <= start["completed_kernels"]
                or end["completed_kernels"] != end["launched_kernels"]):
            raise RuntimeError("incomplete episode or device-kernel evidence")
        for k,row in enumerate(rows):
            row.update(seed=seed, arm_dofs=list(options["arm_dofs"]), return_sum=float(sum(row["rewards"])),
                       sensor_sha256=hashes[k][0].hexdigest(), action_sha256=hashes[k][1].hexdigest(),
                       base_image_sha256=hashes[k][2].hexdigest(), residual_image_sha256=hashes[k][3].hexdigest(),
                       completed_batch_kernels=end["completed_kernels"]-start["completed_kernels"],
                       physics_device=end["device_name"], renderer=visual.device_report)
        return rows


def experiment(args):
    from bimanual_policy_replay import _load, _sources

    path = (args.base/"frozen-policy.json" if args.base.is_dir() else args.base).resolve()
    base, config, updates, base_digest, checkpoint_digest = _load(path,path.with_name("checkpoint.json"))
    if base["arms"]!="both":
        raise ValueError("a completed bilateral frozen base is required")
    _,trainer = _sources(config["sources"],path,args.source_dir,base.get("variant","centroid"))
    seeds = [61001+i for i in range(2*args.generations)]
    historical = set(config["selection_seeds"]) | set(config["holdout_seeds"])
    historical.update(seed for update in updates for seed in update["seeds"])
    if historical.intersection((*seeds,*DEVELOPMENT,*HOLDOUT)):
        raise ValueError("new partitions overlap the original training or evaluation seeds")
    protocol = dict(architecture=ARCH, base_sha256=base_digest, base_checkpoint_sha256=checkpoint_digest,
                    base_contract_sha256=base["contract_sha256"], task=base["task"], source_sha256=_hash(Path(__file__).read_bytes()),
                    training=seeds, development=list(DEVELOPMENT), holdout=list(HOLDOUT), conditions=list(CONDITIONS),
                    sigma=.12, step=.07, generations=args.generations)
    args.out.mkdir(parents=True,exist_ok=True)
    args.cache_dir.mkdir(parents=True,exist_ok=True)
    checkpoint = args.out/"training.json"
    final = args.out/"frozen-network.json"
    init = ResidualNet.initial()
    if args.command=="train":
        if final.exists():
            raise ValueError("network already frozen in this directory")
        state = json.loads(checkpoint.read_bytes()) if checkpoint.is_file() else dict(protocol=protocol,updates=[])
        if state["protocol"]!=protocol:
            raise ValueError("resume contract differs")
        net = init.with_head(state["updates"][-1]["head"] if state["updates"] else init.head)
        for g in range(len(state["updates"]),args.generations):
            directions = np.random.default_rng(54123+g).choice([-1.,1.],size=(2,8,4))
            episodes, differences = [], []
            for j,direction in enumerate(directions):
                jobs = [(label,net.with_head(net.head+sign*.12*direction)) for sign,label in ((1,"plus"),(-1,"minus"))]
                pair = rollout(trainer,base,config,args,seeds[2*g+j],jobs)
                episodes.extend(pair)
                differences.append(pair[0]["return_sum"]-pair[1]["return_sum"])
            previous = net.head.copy()
            net = net.with_head(_update(previous,directions,differences))
            state["updates"].append(dict(generation=g, paired_differences=differences, head=net.head.tolist(),
                                         change_l2=float(np.linalg.norm(net.head-previous)),episodes=episodes))
            _save(checkpoint,state)
        if not np.any(net.head!=init.head) or not any(row["changed_decisions"] for u in state["updates"] for row in u["episodes"]):
            raise ValueError("no reward-driven weight change or realized residual action")
        state["development"] = [rollout(trainer,base,config,args,s,[("learned",net),("frozen_init",init)]) for s in DEVELOPMENT]
        _save(checkpoint,state)
        if not any(rows[0]["learned_vs_initial_changed"] for rows in state["development"]):
            raise ValueError("learned weights never changed a realized development action relative to initialization")
        net.save(final,protocol)
        _save(args.out/"binding.json",dict(network_sha256=_hash(final.read_bytes()),training_sha256=_hash(checkpoint.read_bytes())))
        return
    binding = json.loads((args.out/"binding.json").read_bytes())
    if binding!={"network_sha256":_hash(final.read_bytes()),"training_sha256":_hash(checkpoint.read_bytes())}:
        raise ValueError("frozen model/learning receipt binding mismatch")
    state = json.loads(checkpoint.read_bytes())
    net = ResidualNet.load(final,protocol)
    if state["protocol"]!=protocol or not np.array_equal(state["updates"][-1]["head"],net.head):
        raise ValueError("network differs from final on-policy update")
    all_rows = []
    for s in HOLDOUT:
        receipt = args.out/f"holdout-{s}.json"
        if receipt.is_file():
            stored = json.loads(receipt.read_bytes())
            if stored["binding"]!=binding or [r["condition"] for r in stored["episodes"]]!=list(CONDITIONS) or any(r["seed"]!=s for r in stored["episodes"]):
                raise ValueError("existing holdout receipt differs from the frozen experiment")
            rows = stored["episodes"]
        else:
            rows = []
            jobs = [(c,init if c=="frozen_init" else net) for c in CONDITIONS]
            for i in range(0,len(jobs),args.batch):
                rows.extend(rollout(trainer,base,config,args,s,jobs[i:i+args.batch]))
            _save(receipt,dict(binding=binding,episodes=rows))
        all_rows.append({r["condition"]:r for r in rows})
    summary = {}
    for c in CONDITIONS:
        summary[c] = dict(successes=sum(r[c]["success"] for r in all_rows), trials=len(all_rows),
                          mean_return=float(np.mean([r[c]["return_sum"] for r in all_rows])),
                          paired_learned_minus_condition=[r["learned"]["return_sum"]-r[c]["return_sum"] for r in all_rows],
                          changed_decisions=sum(r[c]["changed_decisions"] for r in all_rows))
    _save(args.out/"holdout.json",dict(binding=binding,seeds=list(HOLDOUT),summary=summary))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("train","evaluate","self-test"))
    parser.add_argument("--base",type=Path)
    parser.add_argument("--source-dir",type=Path)
    parser.add_argument("--out",type=Path)
    parser.add_argument("--cache-dir",type=Path,default=Path("artifacts/bimanual-musa/cache"))
    parser.add_argument("--library",type=Path)
    parser.add_argument("--device-index",type=int)
    parser.add_argument("--generations",type=int,default=3)
    parser.add_argument("--batch",type=int,default=2,help="maximum paired evaluation lanes per simulation batch")
    args = parser.parse_args(argv)
    if args.command=="self-test":
        net = ResidualNet.initial()
        q, rgb = np.zeros(42), np.zeros((2,15,17,3),np.uint8)
        before = net.predict(rgb,q)
        rgb[0,:5,:7,0] = 255
        assert np.max(abs(net.predict(rgb,q)-before))>1e-6
        directions = np.ones((2,8,4))
        assert (_update(net.head,directions,[2.,1.])>net.head).all()
        assert np.array_equal(_update(net.head,directions,[0.,0.]), net.head)
        with tempfile.TemporaryDirectory() as folder:
            file = Path(folder)/"net.json"
            net.save(file,{"test":True})
            assert np.array_equal(ResidualNet.load(file,{"test":True}).predict(rgb,q),net.predict(rgb,q))
        return
    if args.base is None or args.out is None or not 1<=args.generations<=50 or not 1<=args.batch<=len(CONDITIONS):
        parser.error("supply --base, --out, 1..50 generations and 1..6 evaluation lanes")
    experiment(args)


if __name__ == "__main__":
    main()
