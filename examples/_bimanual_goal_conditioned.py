"""Carry to a once-issued table XY command, using both wrist RGB images to track the load.

    python examples/_bimanual_goal_conditioned.py --current-prior --seed 701 --out artifacts/gc-dev \
        --stance bent-elbow --tool-profile twin-pad --load-profile side-lip --compare
    python examples/_bimanual_goal_conditioned.py --base /path/to/frozen-run --source-dir /path/to/sources \
        --seed 55001 --out artifacts/gc-frozen --compare --black-control

The command is the registered destination of the task, in metres. No destination
yaw is supplied. The carry controller maintains the initial box heading measured
by the cameras. This run has its own input contract and original physical scoring.
"""

import argparse
import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np


SOURCE = "registered_task_command_xy_once_per_episode"


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _registered_object_labels(tracker, wrist_module, rgb, cameras):
    """Count visible ink colours agreeing with the estimated object pose in each eye."""
    r, t = tracker.pose
    h,w = rgb.shape[1:3]
    f = .5*h/np.tan(np.deg2rad(39.))
    counts = [0,0]
    for i,(img,(eye,rotation)) in enumerate(zip(rgb,cameras,strict=True)):
        for m in tracker.markers:
            if m["role"]!="object":
                continue
            world = r@np.asarray(m["xyz"]) + t
            if (eye-world) @ (r@np.asarray(m["normal"])) <= 0.:
                continue
            x,y,z = rotation.T @ (world-eye)
            if z >= -.012:
                continue
            u,v = np.array([w/2,h/2]) + f/(-z)*np.array([x,-y])
            cx,cy = int(np.floor(u)),int(np.floor(v))
            if not (0<=cx<w and 0<=cy<h):
                continue
            area = img[max(0,cy-2):min(h,cy+3), max(0,cx-2):min(w,cx+3)]
            hue,chroma,hi = wrist_module._hue(area)
            colour = abs((hue-m["hue_deg"]+180.)%360.-180.) <= 17.
            if np.count_nonzero(colour & (chroma>28.) & (chroma>.40*hi) & (hi>42.)) >= 2:
                counts[i] += 1
    return counts


class _CommandObservation:
    def __init__(self, sensor, module, xy, dt):
        # Keep the complete hue palette for disambiguating object colours; the
        # inherited RGB detector still sees the same real images as the base.
        class ObjectSensor(type(sensor)):
            def _goal(self, obs):
                return None
        self.sensor = ObjectSensor("carry")
        self.module, self.xy = module, tuple(map(float,xy))
        self.heading = None
        self.age_limit = max(1,int(np.ceil(.25/dt)))
        self.eye_age = np.full(2,self.age_limit+1,int)
        self.diagnostics = {}

    def __getattr__(self,name):
        return getattr(self.sensor,name)

    @property
    def pose(self):
        if self.heading is None or max(self.eye_age)>self.age_limit:
            return None
        return self.sensor.pose

    def observe(self,rgb,cameras):
        obj,_,error,counts = self.sensor.observe(rgb,cameras)
        self.eye_age += 1
        marks = [0,0]
        if obj is not None:
            marks = _registered_object_labels(self.sensor,self.module,rgb,cameras)
            self.eye_age[np.asarray(marks)>0] = 0
            if self.heading is None and all(marks) and self.sensor.rotation_age==0:
                self.heading = float(obj[2])
        ready = self.pose is not None
        self.diagnostics = dict(self.sensor.diagnostics, goal_source=SOURCE, command_xy_m=list(self.xy),
                                goal_yaw_source="initial_bilaterally_observed_box_heading",
                                goal_triangle=dict(status="disabled_for_commanded_destination"),
                                stereo_ready=ready, object_colour_inliers_by_eye=marks,
                                object_view_age_decisions=self.eye_age.tolist())
        if not ready:
            return None,None,error,counts
        return obj,np.array([*self.xy,self.heading,0.]),error,counts


def bind_goal_xy(actor, goal_xy):
    """Bind a fresh fiducial carry actor to one immutable world-frame XY instruction."""
    import sys

    xy = np.asarray(goal_xy,float)
    if xy.shape!=(2,) or not np.isfinite(xy).all():
        raise ValueError("a finite registered goal XY in metres is required")
    if actor.task!="carry" or actor.variant!="fiducial" or actor.rest is not None:
        raise ValueError("bind once after reset, before the first observation of a fiducial carry actor")
    sensor = actor.fiducial.tracker
    if isinstance(sensor,_CommandObservation) or sensor.pose is not None:
        raise ValueError("the actor already has a command or observations")
    module = sys.modules[type(sensor).__module__]
    actor.fiducial.tracker = _CommandObservation(sensor,module,xy,actor.dt)
    return actor


def _scene_command(model):
    site = model.site("task_goal").id
    if int(model.site_bodyid[site])!=0:
        raise ValueError("the task command site must be registered to the stationary world body")
    return tuple(map(float,model.site_pos[site,:2]))


def _private_runner(trainer,args):
    """Reuse the exact frozen Runner, with a lane-specific actor factory local to this CLI."""
    spec = importlib.util.spec_from_file_location("_commanded_carry_runner",trainer.__file__)
    local = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(local)
    base = local.VisualPolicy

    class Runner(local.Runner):
        def _batch(self,seed,group,keep):
            jobs = iter(j for j,_ in group)
            command = None
            def create(task,model,weights,**kwargs):
                nonlocal command
                job = next(jobs)
                actor = base(task,model,weights,**kwargs)
                if job["goal_input"]==SOURCE:
                    # The environment's XML writer registers the externally issued
                    # scene.goal_xy at this static site. Copy only XY on reset.
                    if command is None:
                        command = _scene_command(model)
                    bind_goal_xy(actor,command)
                return actor
            local.VisualPolicy = create
            try:
                rows = super()._batch(seed,group,keep)
                if next(jobs,None) is not None:
                    raise RuntimeError("frozen runner did not construct one policy for every lane")
                return rows
            finally:
                local.VisualPolicy = base
    return Runner(args)


def main(argv=None):
    from bimanual_policy_replay import _load, _sources

    p = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    choice = p.add_mutually_exclusive_group(required=True)
    choice.add_argument("--base",type=Path,help="frozen training directory or frozen-policy.json")
    choice.add_argument("--current-prior",action="store_true",help="explicit untrained development control")
    p.add_argument("--source-dir",type=Path)
    p.add_argument("--seed",type=int,nargs="+",required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--library",type=Path)
    p.add_argument("--device-index",type=int)
    p.add_argument("--cache-dir",type=Path,default=Path("artifacts/bimanual-musa/cache"))
    p.add_argument("--compare",action="store_true",help="also run the original full-visual actor")
    p.add_argument("--black-control",action="store_true",help="also run the command with both wrist images black")
    p.add_argument("--batch",type=int,default=3)
    p.add_argument("--stance",choices=["original","bent-elbow"])
    p.add_argument("--tool-profile",choices=["single-pad","twin-pad"])
    p.add_argument("--load-profile",choices=["standard","side-lip"])
    p.add_argument("--image-size",type=int)
    p.add_argument("--horizon",type=int)
    p.add_argument("--action-repeat",type=int)
    args = p.parse_args(argv)
    if min(args.seed)<0 or len(set(args.seed))!=len(args.seed) or args.batch<1:
        p.error("choose distinct nonnegative seeds and a positive batch size")
    if args.out.exists() and (not args.out.is_dir() or any(args.out.iterdir())):
        p.error("use a new or empty output directory")
    defaults = dict(stance="original",tool_profile="single-pad",load_profile="standard",
                    image_size=96,horizon=200,action_repeat=8)
    if args.base:
        path = (args.base/"frozen-policy.json" if args.base.is_dir() else args.base).resolve()
        base,config,_,policy_hash,checkpoint_hash = _load(path,path.with_name("checkpoint.json"))
        if base["task"]!="carry" or base["arms"]!="both" or base.get("variant")!="fiducial":
            raise ValueError("the original frozen policy must be a bilateral fiducial carry policy")
        directory,trainer = _sources(config["sources"],path,args.source_dir,"fiducial")
        setup = {k:config.get(k,default) for k,default in defaults.items()}
        for k,v in setup.items():
            if getattr(args,k) is not None and getattr(args,k)!=v:
                p.error(f"--{k.replace('_','-')} conflicts with the frozen policy contract")
        weights,manifest = base["weights"],config["sources"]
    else:
        directory = (args.source_dir or Path(__file__).resolve().parent).resolve()
        files = ("bimanual_musa_train.py","bimanual_musa_env.py","bimanual_wrist_render.py",
                 "_bimanual_visual_policy.py","_bimanual_image_pose.py","_bimanual_fiducial.py")
        manifest = {f:_digest(directory/f) for f in files}
        directory,trainer = _sources(manifest,Path(__file__),directory,"fiducial")
        weights,policy_hash,checkpoint_hash = trainer.PRIOR,None,None
        setup = {k:default if getattr(args,k) is None else getattr(args,k) for k,default in defaults.items()}
    if min(setup[k] for k in ("image_size","action_repeat","horizon"))<1:
        p.error("image size, action repeat and horizon must be positive")
    args.out.mkdir(parents=True,exist_ok=True)
    args.cache_dir.mkdir(parents=True,exist_ok=True)
    run = SimpleNamespace(**setup,out=args.out,cache_dir=args.cache_dir,library=args.library,
                          device_index=args.device_index,batch=args.batch,wall_seconds=None,
                          policy_variant="fiducial",task="carry",mode="musa")
    jobs = [dict(trainer._job("commanded-xy",weights),goal_input=SOURCE)]
    if args.compare:
        jobs.append(dict(trainer._job("full-visual",weights),goal_input="wrist_rgb_triangle"))
    if args.black_control:
        jobs.append(dict(trainer._job("commanded-black",weights,black=True),goal_input=SOURCE))
    report = dict(schema="bimanual-commanded-xy-v1",policy_kind="untrained_prior" if args.current_prior else "frozen",
                  policy_sha256=policy_hash,checkpoint_sha256=checkpoint_hash,source_sha256=manifest,
                  adapter_sha256=_digest(Path(__file__)), command_frame="world table, metres",
                  yaw_input="initial stereo-measured box heading", object_input="two wrist RGB + robot calibration",
                  source_dir=str(directory),settings=setup,jobs=jobs,episodes=[])
    trainer._save(args.out/"commanded.json",report)
    runner = _private_runner(trainer,run)
    for seed in args.seed:
        rows = runner.episodes("commanded",seed,jobs,keep_trajectory=True)
        for row in rows:
            f = row["last_features"]
            report["episodes"].append(dict(seed=seed,job=row["job"]["label"],success=row["success"],
                        return_sum=row["return_sum"],goal_input=row["job"]["goal_input"],
                        command_xy_m=f.get("command_xy_m"),scene_sha256=row["scene_sha256"],
                        stereo_ready_decisions=sum(bool(x["visual"].get("stereo_ready")) for x in row["decision_diagnostics"]),
                        receipt=f"episodes/commanded/{seed}-{row['job']['label']}.json"))
        trainer._save(args.out/"commanded.json",report)


if __name__=="__main__":
    main()
