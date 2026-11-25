"""Carry the original 6 kg flat-sided box with a rectangular array of rubber contacts.

    python examples/bimanual_flat_gripper.py --out artifacts/flat-jaw-01 --seed 701 --compare

The alternative jaw has four physical sphere contacts per hand. The paired
control uses the original two spheres. Both use the original box, motors,
friction, visual policy and success test. Each run saves XML, trajectory and
contact evidence. A supplied frozen policy is a transfer to this mechanism.
"""

import argparse
import hashlib
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np


RADIUS = .0135
HALF_HEIGHT = .018
NAMES = ("a", "b", "c", "d")  # rear lower, front lower, rear upper, front upper


def _numbers(x):
    return " ".join(f"{v:.9g}" for v in x)


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def four_contact_scene(scene):
    """Replace just the wrist-side rubber/carrier geometry in one source scene."""
    if (scene.task,scene.load_profile,scene.tool_profile)!=("carry","standard","twin-pad"):
        raise ValueError("expected the original flat-box carry scene and two-pad source tool")
    root = ET.fromstring(scene.xml)
    payload = root.find(".//body[@name='payload']")
    geoms = payload.findall("geom")
    if len(geoms)!=1 or geoms[0].get("type")!="box" or float(geoms[0].get("mass"))!=6.:
        raise ValueError("expected the original unmodified 6 kg free box")
    for side,sign in (("left",1),("right",-1)):
        parent = root.find(f".//geom[@name='{side}_pad_a']/..")
        tcp = np.fromstring(parent.find(f"site[@name='{side}_tool']").get("pos"),sep=" ")
        pads = [parent.find(f"geom[@name='{side}_pad_{letter}']") for letter in ("a","b")]
        old_radius = float(pads[0].get("size"))
        mass = sum(float(p.get("mass")) for p in pads)/4.
        original_x = [float(p.get("pos").split()[0]) for p in pads]
        spec = pads[0].attrib.copy()
        for p in pads:
            parent.remove(p)
        for i,letter in enumerate(NAMES):
            # The smaller radius and inward offset cancel precisely at the
            # working face, retaining the initial 8 mm jaw-to-box gap.
            center = [original_x[i%2], tcp[1]-sign*(old_radius-RADIUS),
                      tcp[2]+(-HALF_HEIGHT if i<2 else HALF_HEIGHT)]
            ET.SubElement(parent,"geom",dict(spec,name=f"{side}_pad_{letter}",pos=_numbers(center),
                          size=str(RADIUS),mass=str(mass)))
        carrier = parent.find(f"geom[@name='{side}_carrier']")
        carrier.attrib.pop("fromto")
        # Recessed backing, as for the original non-contact carrier. The four
        # rubber caps are the physical faces and extend 20 mm ahead of the plate.
        carrier.attrib.update(type="box",pos=_numbers(tcp-np.array([0.,sign*.002,0.])),
                              size=".044 .004 .026")
    xml = ET.tostring(root,encoding="unicode")
    return replace(scene,xml=xml,xml_sha256=_sha(xml.encode()),tool_profile="four-contact")


class _ContactAudit:
    """Delegate every original physics/control operation and inspect step-input contacts."""
    def __init__(self,task,suffixes):
        self.task = task
        self.keys = [f"{s}_pad_{k}" for s in ("left","right") for k in suffixes]
        owner = {task.model.geom(n).id: i//len(suffixes) for i,n in enumerate(self.keys)}
        task.pad_owner = owner
        task.pad_geoms = tuple(owner)
        task.pad_slot = {idx:i for i,idx in enumerate(owner)}
        self.normal_impulse = np.zeros((2,2))  # per hand: signed z normal and signed z friction
        self.pad_peak = np.zeros(len(owner))
        self.all_loaded_air = 0.
        self.both_fore_aft_air = 0.
        self.table_load_impulse = 0.
        self.table_contact_steps = 0
        self.worst_table_distance = None
        self.hand_contact_time = np.zeros(2)
        self.hand_normal_impulse = np.zeros(2)
        self.hand_signed_up_impulse = np.zeros(2)
        self.hand_positive_up_impulse = np.zeros(2)
        self.hand_table_impulse = np.zeros(2)

    def __getattr__(self,name):
        return getattr(self.task,name)

    def evaluate(self,q,v,contacts,*,advance=True):
        state = self.task.evaluate(q,v,contacts,advance=advance)
        if not advance:
            return state
        dt = self.scene.timestep
        loads = np.asarray(state["individual_pad_payload_normal_n"])
        normal = np.asarray(state["pad_normal_n"])
        vertical = np.asarray(state["pad_vertical_n"])
        self.hand_contact_time += dt*(normal>0.)
        self.hand_normal_impulse += dt*normal
        self.hand_signed_up_impulse += dt*vertical
        self.hand_positive_up_impulse += dt*np.maximum(0.,vertical)
        self.pad_peak = np.maximum(self.pad_peak,loads)
        airborne = state["floor_vertical_n"] < .05*self.scene.mass*9.81 and state["clearance_m"]>=.003
        self.all_loaded_air += dt if airborne and np.all(loads>=1.) else 0.
        # Each pair of adjacent columns is rear/front on that horizontal row.
        columns = loads.reshape(2,-1,2).sum(axis=1)
        self.both_fore_aft_air += dt if airborne and np.all(columns>=1.) else 0.
        touched = False
        for c in contacts:
            a,b = map(int,c.geometry_ids)
            if a in self.pad_owner:
                pad,other,sign = a,b,1.
            elif b in self.pad_owner:
                pad,other,sign = b,a,-1.
            else:
                continue
            w = np.asarray(c.wrench_contact_frame)
            if other==self.floor:
                touched = True
                self.table_load_impulse += max(0.,float(w[0]))*dt
                self.hand_table_impulse[self.pad_owner[pad]] += max(0.,float(w[0]))*dt
                dist = float(c.distance)
                self.worst_table_distance = dist if self.worst_table_distance is None else min(self.worst_table_distance,dist)
            if airborne and other in self.payload_geoms and w[0]>0.:
                axes = np.asarray(c.frame).reshape(3,3)
                up = sign*w[:3]*axes[:,2]
                self.normal_impulse[self.pad_owner[pad]] += dt*np.array([up[0],up[1:].sum()])
        self.table_contact_steps += int(touched)
        return state

    def summary(self):
        hands = {}
        for i,side in enumerate(("left","right")):
            hands[side] = dict(motors_enabled=self.task.active_arms in ("both",side),
                               payload_contact_s=float(self.hand_contact_time[i]),
                               payload_normal_impulse_ns=float(self.hand_normal_impulse[i]),
                               payload_vertical_impulse_ns=float(self.hand_signed_up_impulse[i]),
                               payload_positive_vertical_impulse_ns=float(self.hand_positive_up_impulse[i]),
                               table_normal_impulse_ns=float(self.hand_table_impulse[i]))
        return dict(pad_order=self.keys, peak_normal_n=self.pad_peak.tolist(),
                    by_hand=hands,
                    all_contacts_loaded_airborne_s=self.all_loaded_air,
                    both_hands_front_and_rear_loaded_airborne_s=self.both_fore_aft_air,
                    airborne_upward_impulse_ns_by_hand_normal_then_friction=self.normal_impulse.tolist(),
                    pad_table_contact_steps=self.table_contact_steps, pad_table_normal_impulse_ns=self.table_load_impulse,
                    minimum_reported_pad_table_contact_distance_m=self.worst_table_distance)


class _FaceAlignment:
    """Orient the four-contact face around its TCP while retaining the base transport command."""

    def __init__(self):
        self.reference = None
        self.previous_delta = np.zeros((2,7))

    def correct(self,actor,observation,action):
        control = actor.fiducial
        pose = control.tracker.pose
        live = (pose is not None and actor.features["fresh_pose"] and
                control.phase in ("clamp","lift","transport","lower"))
        target_rate = np.zeros(3)
        if live:
            normal = pose[0][:,1]
            if self.reference is None:
                self.reference = normal.copy()
            previous = self.reference
            filtered = previous + (1.-np.exp(-actor.dt/.045))*(normal-previous)
            self.reference = filtered/max(1e-8,np.linalg.norm(filtered))
            target_rate = np.cross(previous,self.reference)/actor.dt
            target_rate *= min(1.,1.2/max(np.linalg.norm(target_rate),1e-8))
        else:
            self.reference = None
        # Arm Jacobians refer to the policy's own FK. RGB supplies the side-face
        # direction. Per-pad contact labels, simulator payload pose and target
        # position are deliberately absent from this servo.
        commands = np.asarray(action).reshape(2,7).copy()
        velocity = np.asarray(observation[14:28]).reshape(2,7)
        mask = np.asarray(observation[28:]).reshape(2,7)
        diagnostics = []
        for i,((_,va,_),(tool,_)) in enumerate(zip(actor.joints,actor.sites,strict=True)):
            n = len(va)
            change = np.zeros(n)
            angle,requested_moment = None,np.zeros(3)
            if live and mask[i,0]:
                jp,jr = np.zeros((3,actor.model.nv)),np.zeros((3,actor.model.nv))
                actor.mj.mj_jacSite(actor.model,actor.data,jp,jr,tool)
                j, angular = jp[:,va],jr[:,va]
                # The two rubber columns and two rows lie in tool-local XZ.
                jaw = actor.data.site_xmat[tool].reshape(3,3)[:,1]
                angle = float(np.arccos(np.clip(jaw@self.reference,-1.,1.)))
                error = np.cross(jaw,self.reference)
                rate = angular@velocity[i,:n] - target_rate
                rate -= (rate@jaw)*jaw
                requested_moment = 18.*error-.65*rate
                requested_moment *= min(1.,1.35/max(1e-8,np.linalg.norm(requested_moment)))
                # Remove the part of the new joint request along the TCP
                # position Jacobian. The original lift and transport continue.
                torque = angular.T@requested_moment
                change = torque - j.T@np.linalg.solve(j@j.T+1e-8*np.eye(3),j@torque)
            start = sum(len(a[0]) for a in actor.joints[:i])
            lo,hi = actor.model.actuator_ctrlrange[start:start+n].T
            base = commands[i,:n]*np.where(commands[i,:n]>=0.,hi,-lo)
            # Slew the additional wrist effort and fit it in the available
            # physical motor headroom without rescaling the original action.
            previous_delta = self.previous_delta[i,:n]
            change = previous_delta + np.clip(change-previous_delta,-15.*actor.dt,15.*actor.dt)
            limits = np.full(n,np.inf)
            pos,neg = change>1e-9,change< -1e-9
            limits[pos] = (hi[pos]-base[pos])/change[pos]
            limits[neg] = (lo[neg]-base[neg])/change[neg]
            scale = float(np.clip(limits.min(),0.,1.))
            change = scale*change*mask[i,:n]
            self.previous_delta[i,:n] = change
            total = np.clip(base+change,lo,hi)
            commands[i,:n] = total/np.where(total>=0.,hi,-lo)
            diagnostics.append(dict(normal_error_rad=angle,requested_moment_nm=requested_moment.tolist(),
                                    applied_joint_delta_nm=change.tolist(),headroom_fraction=scale))
        actor.features["flat_face_alignment"] = dict(live=bool(live),arms=diagnostics)
        return (commands*mask).ravel()


def _run(trainer,args,weights,seed,style):
    from bimanual_musa_env import BimanualBatchEnv
    from bimanual_wrist_render import BimanualWristRenderer, MARKER_LAYOUT_VERSION
    from phymeo.builders.mujoco_compiler import compile_mujoco_xml

    suffixes = NAMES if style=="four-contact" else ("a","b")
    stem = args.out/f"{seed}-{style}"
    with ExitStack() as stack:
        env = stack.enter_context(BimanualBatchEnv("carry",seed,world_keys=(0,),arm_dofs=trainer._dofs(seed),
                    mode=args.mode,cache_root=args.cache_dir,action_repeat=args.action_repeat,horizon=args.horizon,
                    stance="bent-elbow",load_profile="standard",tool_profile="twin-pad"))
        original_sha = env.scene.xml_sha256
        if style=="four-contact":
            env.scene = four_contact_scene(env.scene)
            env.tool_profile = env.scene.tool_profile
            env.compilation = compile_mujoco_xml(env.scene.xml,require_asset=False)
            env.model = env.compilation.mj_model
        obs,info = env.reset(seed=seed,active_arms=args.arms)[0]
        audit = _ContactAudit(env.worlds[0],suffixes)
        env.worlds[0] = audit
        model = env.model
        command_mask = np.zeros((2,7),dtype=bool)
        for i,(side,count) in enumerate(zip(("left","right"),env.scene.arm_dofs,strict=True)):
            if args.arms in ("both",side):
                command_mask[i,:count] = True
        if not np.array_equal(command_mask.ravel(),np.asarray(obs[28:],bool)):
            raise RuntimeError("environment actuator mask differs from selected arms")
        motor_enabled = np.asarray(audit.enabled,dtype=bool)
        motor_abs_impulse = np.zeros(model.nu)
        motor_peak = np.zeros(model.nu)
        vis = stack.enter_context(BimanualWristRenderer(model,args.cache_dir/"camera",size=(args.image_size,)*2,
                    fiducials=True,library_path=args.library,device_index=args.device_index))
        actor = trainer.VisualPolicy("carry",model,weights,variant="fiducial",
                    decision_dt=env.scene.timestep*args.action_repeat,servo_gains=(audit.kp,audit.kd))
        if style=="four-contact":
            class PatchControl(type(actor.fiducial)):
                def _carry_fixture(self,model):
                    super()._carry_fixture(model)
                    tcp = model.site_pos[model.site("left_tool").id,1]
                    pad = model.geom("left_pad_a").id
                    self.pad_radius = float(tcp-model.geom_pos[pad,1]+model.geom_size[pad,0])
            actor.fiducial = PatchControl("carry",actor.dt)
        if args.command_xy:
            from _bimanual_goal_conditioned import bind_goal_xy, _scene_command
            bind_goal_xy(actor,_scene_command(model))
        alignment = _FaceAlignment() if args.control=="face-alignment" and style=="four-contact" else None
        effective_control = "face-alignment" if alignment is not None else "original"
        start = env.runtime.musa_device_report() if args.mode=="musa" else None
        states = [info["qpos"].copy()]
        ticks,times = [int(info["tick"])],[float(info["time_s"])]
        actions,rewards,diagnostics = [],[],[]
        sensor,policy_input = hashlib.sha256(),hashlib.sha256()
        actual_nonzero,policy_nonzero = np.zeros(2,dtype=int),np.zeros(2,dtype=int)
        min_pad_height = None
        pads = np.array(audit.pad_geoms)
        for _ in range(args.horizon):
            rgb = vis.observe(info["qpos"],info["tick"],info["time_s"])
            height = float(np.min(vis.data.geom_xpos[pads,2]-model.geom_size[pads,0]))
            min_pad_height = height if min_pad_height is None else min(min_pad_height,height)
            sensor.update(np.ascontiguousarray(rgb).tobytes())
            image = np.zeros_like(rgb) if args.black_rgb else rgb
            actual_nonzero += np.any(rgb,axis=(1,2,3))
            policy_nonzero += np.any(image,axis=(1,2,3))
            policy_input.update(np.ascontiguousarray(image).tobytes())
            action = actor.act(obs,image,wrist_force_world_n=info["wrist_force_world_n"])
            if alignment is not None:
                action = alignment.correct(actor,obs,action)
            action = np.where(command_mask.ravel(),action,0.)
            obs,reward,terminated,truncated,info = env.step_torque({0:action})[0]
            applied = np.asarray(info["applied_ctrl"],float)
            if not np.isfinite(applied).all() or np.any(applied[~motor_enabled]!=0.):
                raise RuntimeError("invalid or nonzero parked-arm motor effort reported by the simulator")
            # The torque entry holds a constant motor command per decision.
            elapsed = float(info["time_s"])-times[-1]
            motor_abs_impulse += np.abs(applied)*elapsed
            motor_peak = np.maximum(motor_peak,np.abs(applied))
            actions.append(action.copy())
            rewards.append(float(reward))
            states.append(info["qpos"].copy())
            ticks.append(int(info["tick"]))
            times.append(float(info["time_s"]))
            diagnostics.append(dict(input_tick=ticks[-2],output_tick=ticks[-1],visual=actor.features,
                     measured={n:info[n] for n in ("payload_xyz_m","payload_yaw_rad","clearance_m","tilt_rad",
                     "individual_pad_payload_normal_n","goal_error_xy_m","goal_yaw_error_rad",
                     "pad_normal_n","pad_vertical_n","floor_vertical_n","both_loaded_airborne_s",
                     "carried_airborne","goal_dwell_s")}))
            if terminated or truncated:
                break
        if not (terminated or truncated):
            raise RuntimeError("episode did not complete")
        end = env.runtime.musa_device_report() if args.mode=="musa" else None
        if end and (end["backend"]!="musa" or "S4000" not in end["device_name"].upper() or
                    end["launched_kernels"]!=end["completed_kernels"] or end["completed_kernels"]<=start["completed_kernels"]):
            raise RuntimeError("missing completed S4000 physics kernel evidence")
        stem.with_suffix(".xml").write_bytes(env.scene.xml.encode())
        np.savez_compressed(stem.with_suffix(".npz"),qpos=states,ticks=ticks,time_s=times,
                            model_sha256=env.scene.xml_sha256,action=actions,rewards=rewards)
        if args.black_rgb and (np.any(policy_nonzero) or np.any(actual_nonzero==0)):
            raise RuntimeError("missing actual wrist images or nonblack policy input during blackout")
        motor_audit = {}
        start_index = 0
        for side,n in zip(("left","right"),env.scene.arm_dofs,strict=True):
            span = slice(start_index,start_index+n)
            motor_audit[side] = dict(enabled=bool(motor_enabled[span].all()),
                                     peak_abs_torque_nm=motor_peak[span].tolist(),
                                     integrated_abs_torque_nms=motor_abs_impulse[span].tolist())
            start_index += n
        row = dict(task="carry",seed=seed,arm_dofs=list(env.scene.arm_dofs),physics=args.mode,
                   job=trainer._job(style,weights,args.arms,black=args.black_rgb),
                   tool_profile=style,control=effective_control,
                   load_profile="standard",stance="bent-elbow",
                   original_scene_sha256=original_sha,scene_sha256=env.scene.xml_sha256,
                   trajectory_sha256=_sha(stem.with_suffix(".npz").read_bytes()),sensor_rgb_sha256=sensor.hexdigest(),
                   policy_input_rgb_sha256=policy_input.hexdigest(),actual_nonblack_frames_by_wrist=actual_nonzero.tolist(),
                   policy_nonblack_frames_by_wrist=policy_nonzero.tolist(),motor_audit=motor_audit,
                   fiducials=True,marker_layout_version=MARKER_LAYOUT_VERSION,decisions=len(actions),
                   physics_ticks=int(info["tick"]),rewards=rewards,
                   return_sum=float(sum(rewards)),success=bool(info["success"]),terminated=terminated,truncated=truncated,
                   last_features=actor.features,decision_diagnostics=diagnostics,device_before=start,device_after=end,
                   renderer=vis.device_report,min_decision_input_pad_clearance_m=min_pad_height,
                   native_contact_audit=audit.summary(),goal_input="registered_command_xy" if args.command_xy else "wrist_rgb",
                   final_metrics={k:v for k,v in info.items() if k not in ("qpos","qvel","applied_ctrl")})
        trainer._save(stem.with_suffix(".json"),row)
        return dict(seed=seed,tool=style,control=effective_control,active_arms=args.arms,black_rgb=args.black_rgb,
                    success=row["success"],receipt=stem.with_suffix(".json").name,motor_audit=motor_audit,
                    sensor_rgb_sha256=sensor.hexdigest(),policy_input_rgb_sha256=policy_input.hexdigest(),
                    original_scene_sha256=original_sha,carried_airborne=info["carried_airborne"],
                    max_clearance_m=info["max_clearance_m"],goal_error_xy_m=info["goal_error_xy_m"],
                    audit=audit.summary())


def main(argv=None):
    from bimanual_policy_replay import _load,_sources

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--seed",type=int,nargs="+",default=[701])
    p.add_argument("--compare",action="store_true")
    p.add_argument("--arms",choices=("both","left","right"),default="both",
                   help="active motors; the environment parks the other arm and retains its passive dynamics")
    p.add_argument("--black-rgb","--black-wrists",action="store_true",
                   help="render both actual wrist images, then give the policy two all-black frames")
    p.add_argument("--control",choices=("original","face-alignment"),default="original",
                   help="four-contact wrist controller; the two-pad comparator always uses the original controller")
    p.add_argument("--command-xy",action="store_true",help="use the same explicit XY in both jaws' comparison")
    p.add_argument("--base",type=Path,help="transfer a verified bilateral fiducial carry policy; default is the untrained prior")
    p.add_argument("--source-dir",type=Path)
    p.add_argument("--mode",choices=("musa","cpu"),default="musa")
    p.add_argument("--cache-dir",type=Path,default=Path("artifacts/bimanual-musa/cache"))
    p.add_argument("--image-size",type=int)
    p.add_argument("--horizon",type=int)
    p.add_argument("--action-repeat",type=int)
    p.add_argument("--library",type=Path)
    p.add_argument("--device-index",type=int)
    args = p.parse_args(argv)
    if min(args.seed)<0 or len(set(args.seed))!=len(args.seed):
        p.error("choose distinct nonnegative seeds")
    if args.out.exists() and (not args.out.is_dir() or any(args.out.iterdir())):
        p.error("use a new or empty output directory")
    if args.base:
        path = (args.base/"frozen-policy.json" if args.base.is_dir() else args.base).resolve()
        policy,config,_,policy_hash,cp_hash = _load(path,path.with_name("checkpoint.json"))
        if (policy["task"],policy["arms"],policy.get("variant"))!=("carry","both","fiducial"):
            raise ValueError("expected a frozen bilateral fiducial carry policy")
        manifest = config["sources"]
        directory,trainer = _sources(manifest,path,args.source_dir,"fiducial")
        weights = policy["weights"]
    else:
        directory = (args.source_dir or Path(__file__).resolve().parent).resolve()
        files = ("bimanual_musa_train.py","bimanual_musa_env.py","bimanual_wrist_render.py",
                 "_bimanual_visual_policy.py","_bimanual_image_pose.py","_bimanual_fiducial.py")
        manifest = {f:_sha((directory/f).read_bytes()) for f in files}
        directory,trainer = _sources(manifest,Path(__file__),directory,"fiducial")
        policy_hash=cp_hash=None
        weights,config = trainer.PRIOR,{}
    for k,default in (("image_size",96),("action_repeat",8),("horizon",400)):
        override = getattr(args,k)
        if args.base and override is not None and override!=config[k]:
            p.error(f"{k} differs from the supplied frozen controller")
        setattr(args,k,override if override is not None else config.get(k,default))
    if min(args.image_size,args.action_repeat,args.horizon)<1:
        p.error("image size, action repeat and horizon must be positive")
    args.out.mkdir(parents=True,exist_ok=True)
    args.cache_dir.mkdir(parents=True,exist_ok=True)
    report = dict(schema="flat-box-four-contact-v1",policy_kind="frozen_transfer" if args.base else "untrained_prior",
                  four_contact_control=args.control,active_arms=args.arms,black_rgb=args.black_rgb,
                  base_policy_sha256=policy_hash,base_checkpoint_sha256=cp_hash,source_dir=str(directory),
                  adapter_sha256=_sha(Path(__file__).read_bytes()),sources=manifest,episodes=[])
    if args.command_xy:
        import _bimanual_goal_conditioned as command_adapter
        report["command_adapter_sha256"] = _sha(Path(command_adapter.__file__).read_bytes())
    for seed in args.seed:
        for style in (("four-contact","twin-pad") if args.compare else ("four-contact",)):
            report["episodes"].append(_run(trainer,args,weights,seed,style))
            trainer._save(args.out/"flat-jaw.json",report)
            print(report["episodes"][-1],flush=True)


if __name__=="__main__":
    main()
