"""Run a small binary-reward SGS experiment on two Panda arms and a free part.

The policy tunes task-relative motion gains within an IK and motor-feedback
controller. Only the reset uses poses computed by IK. Controls and reciprocal
tool loads advance dynamics thereafter, including the independent free part.
"""
import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import mujoco as mj
import numpy as np

from _panda_assembly_scene import HOME, PART_COM, PIN_TIP_Z, build_scene, initial_qpos_for_arms
from _panda_assembly_sgs import StateSampler, TerminalCEM, uniform_validation
from _panda_thread_guide import LEAD_M_PER_RAD, implicit_turn_drag, thread_reaction
from phymeo.backends.mujoco_runtime import MuJoCoControl, MuJoCoInitialState, MuJoCoRuntime


def save(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf8")


def skew(x):
    a, b, c = x
    return np.array([[0., -c, b], [c, 0., -a], [-b, a, 0.]])


def zrot(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])


def rot_error(goal, current):
    return .5 * sum(np.cross(current[:,i], goal[:,i]) for i in range(3))


@dataclass
class Frame:
    x: np.ndarray
    r: np.ndarray
    jp: np.ndarray
    jr: np.ndarray


def frame(model, data, site):
    jp, jr = np.empty((3, model.nv)), np.empty((3, model.nv))
    mj.mj_jacSite(model, data, jp, jr, site)
    return Frame(data.site_xpos[site].copy(), data.site_xmat[site].reshape(3,3).copy(), jp, jr)


def relative(a, b):
    delta = b.x-a.x
    return (a.r.T@delta, a.r.T@b.r, a.r.T@(b.jp-a.jp+skew(delta)@a.jr),
            a.r.T@(b.jr-a.jr))


def mount_frame(tip):
    arm = tip.r @ (np.asarray(PART_COM)-np.array([0.,0.,PIN_TIP_Z]))
    return Frame(tip.x+arm,tip.r,tip.jp-skew(arm)@tip.jr,tip.jr)


class Geometry:
    def __init__(self, scene, comp):
        self.scene, self.model = scene, comp.mj_model
        self.data = mj.MjData(self.model)
        self.ids = [self.model.site(x).id for x in ("assembly_hole","assembly_driver","assembly_pin_tip")]
        self.home = np.asarray(scene.initial_qpos)
        self.update(self.home, np.zeros(20))
        a,b,_ = self.frames()
        self.zero_rot = a.r.T@b.r

    def update(self, q, v):
        self.data.qpos[:] = q
        self.data.qvel[:] = v
        mj.mj_fwdPosition(self.model, self.data)
        mj.mj_fwdVelocity(self.model, self.data)

    def frames(self):
        return [frame(self.model,self.data,i) for i in self.ids]

    def ik(self, q14, xyz, turn):
        q14 = np.array(q14,copy=True)
        goal = zrot(turn)@self.zero_rot
        for _ in range(64):
            self.update(np.r_[q14,self.home[14:]], np.zeros(20))
            a,b = [frame(self.model,self.data,i) for i in self.ids[:2]]
            p,r,jp,jr = relative(a,b)
            error = np.r_[xyz-p, rot_error(goal,r)]
            if np.linalg.norm(error) < 1.e-7:
                return q14
            j = np.vstack((jp[:,:14], jr[:,:14]))
            dq = j.T @ np.linalg.solve(j@j.T + np.eye(6)*.00001,error)
            q14 += dq / max(1.,np.max(np.abs(dq))/.10)
            q14 = np.clip(q14,self.model.jnt_range[:14,0]+.001,self.model.jnt_range[:14,1]-.001)
        raise RuntimeError("IK failed to construct the declared pose")

    def observe(self):
        board,driver,part = self.frames()
        p,r,jp,jr = relative(board,part)
        offset_r = r@self.zero_rot.T
        x = offset_r[:,0]
        denom = x[0]**2+x[1]**2
        if denom < .2:
            raise ValueError("turn coordinate singular")
        angle = np.arctan2(x[1], x[0])
        ja = np.array([-x[0]*x[2]/denom,-x[1]*x[2]/denom,1.])@jr
        tilt = np.arccos(np.clip(r[2,2],-1.,1.))
        # board's entrance is 6 mm in front of its middle plane
        return board,driver,part,p,p[2]+.006,angle,jp,ja,tilt


def reset_bank(task, seed, n):
    rng = np.random.default_rng(seed)
    bank=[]
    for index in range(n):
        tier=index%3
        if task=="pin":
            depth=rng.uniform(*[(-.018,-.008),(-.005,-.001),(.001,.004)][tier])
            radius=rng.uniform(0.,(.004,.002,.0005)[tier])
            turn=rng.uniform(-.09,.09)
        else:
            turn=rng.uniform(*[(0.,.15),(.2,.43),(.50,.68)][tier])
            depth=.005+LEAD_M_PER_RAD*turn
            radius=rng.uniform(.00002,.0004)
        az=rng.uniform(-np.pi,np.pi)
        bank.append(dict(id=index,xyz=[float(radius*np.cos(az)),float(radius*np.sin(az)),float(depth-.006)],
                         turn=float(turn),tier=tier))
    return bank


KP=np.array([270,300,240,220,95,85,65.]*2)
KD=np.array([26,29,22,18,6,5,3.]*2)


def run_batch(task, scene, comp, kin, specs, params, mode, cache, steps, monitor=None):
    n=len(specs)
    keys=tuple(range(n))
    init={}; starts=[]; targets=[]; ref=[]; records=[]
    for k,(s,par) in enumerate(zip(specs,params,strict=True)):
        pos=np.asarray(s["xyz"])
        q=kin.ik(HOME,pos,s["turn"])
        qfull=initial_qpos_for_arms(scene,q)
        init[k]=MuJoCoInitialState(q=qfull,v=np.zeros(20))
        starts.append(q)
        kin.update(qfull,np.zeros(20))
        _,_,_,_,depth,turn,_,_,_=kin.observe()
        ref.append((depth,turn))
        u=np.tanh(par)
        # Learned progress and centering gains act on goal-relative observations.
        nominal=np.array([0.,0.,(.010 if task=="pin" else .005+1.2*LEAD_M_PER_RAD)-.006])
        goal=pos+(nominal-pos)*np.array([1.+.65*u[0],1.+.65*u[0],1.+.45*u[1]])
        target_turn=s["turn"] if task=="pin" else s["turn"]+(1.2-s["turn"])*(1.+.45*u[2])
        targets.append(q if monitor=="hold" else kin.ik(q,goal,target_turn))
        records.append(dict(start=s,history=[],last=[],min_gap_m=0.,contact_steps=0,
                            peak_normal_n=0.,peak_thread_force_n=0.,drag_work_j=0.,peak_mount_error_m=0.))
    opts=dict(cache_root=cache,contact_capacity=32,constraint_capacity=128) if mode=="musa" else {}
    snap="integration_snapshot" if mode=="musa" else "snapshot"
    dt=float(scene.timestep)
    enabled=task=="screw_proxy"
    angles=[t for _,t in ref]
    previous=[float(t) for t in angles]
    previous_drag=np.zeros(n)
    with MuJoCoRuntime(comp,mode=mode,world_keys=keys,initial_states=init,**opts) as rt:
        before=rt.musa_device_report() if mode=="musa" else {}
        for step in range(steps+1):
            states=getattr(rt,snap)()
            controls={}
            for k in keys:
                info,st=records[k],states[k]
                q,v=np.asarray(st.q),np.asarray(st.v)
                kin.update(q,v)
                board,driver,part,p,depth,angle,jp,ja,tilt=kin.observe()
                increment=np.arctan2(np.sin(angle-previous[k]),np.cos(angle-previous[k]))
                angles[k]+=increment
                previous[k]=angle
                info["drag_work_j"]-=float(previous_drag[k]*increment)
                delta_angle=angles[k]-ref[k][1]
                # Compliant six-axis tool coupling. The Jacobian of relative
                # translation includes the moment arm at the driver's origin.
                offset,rd,jd,jw=relative(mount_frame(driver),mount_frame(part))
                f=-2400.*offset-22.*(jd@v)
                moment=1.6*rot_error(np.eye(3),rd)-.012*(jw@v)
                external=jd.T@f+jw.T@moment
                pitch_error=0.
                if enabled:
                    thread,loads=thread_reaction(depth-ref[k][0],delta_angle,jp[2],ja,v)
                    external+=thread
                    pitch_error=loads["pitch_error_m"]
                    info["peak_thread_force_n"]=max(info["peak_thread_force_n"],abs(loads["axial_force_n"]))
                info["peak_mount_error_m"]=max(info["peak_mount_error_m"],float(np.linalg.norm(offset)))
                contacts=st.last_step.contacts if st.last_step is not None else ()
                if contacts:
                    info["contact_steps"]+=1
                    info["min_gap_m"]=min(info["min_gap_m"],*(float(c.distance) for c in contacts))
                    info["peak_normal_n"]=max(info["peak_normal_n"],*(float(c.wrench_contact_frame[0]) for c in contacts))
                speed=float(np.linalg.norm(jp@v))
                angular_speed=abs(float(ja@v))
                lateral=float(np.max(np.abs(p[:2])))
                ok=lateral<=.0009 and tilt<=np.deg2rad(2.) and speed<=.008
                if enabled:
                    ok=ok and abs(angles[k]-1.2)<=.13 and abs(pitch_error)<=.00015 and angular_speed<=.22
                    ok=ok and depth-ref[k][0]>=.00007 and info["drag_work_j"]>.003
                else:
                    ok=ok and .008<=depth<=.012
                ok=ok and info["min_gap_m"]>=-.0007 and info["peak_mount_error_m"]<=.0025
                if step>=steps-24:
                    info["last"].append(bool(ok))
                if monitor is not None and (step%3==0 or step==steps):
                    info["history"].append((float(st.time),q.copy()))
                if step==steps:
                    info.update(depth_m=float(depth),advance_m=float(depth-ref[k][0]),turn_rad=float(angles[k]),
                                rotated_rad=float(delta_angle),lateral_m=lateral,tilt_rad=float(tilt),
                                pitch_error_m=float(pitch_error),speed_m_s=speed,turn_rate_rad_s=angular_speed,
                                left_motion_rad=float(np.linalg.norm(q[:7]-starts[k][:7])),
                                right_motion_rad=float(np.linalg.norm(q[7:14]-starts[k][7:14])))
                    continue
                blend=min(1.,(step+1)/(steps*.63))
                blend=blend**3*(10.+blend*(-15.+6.*blend))
                desired=starts[k]+blend*(targets[k]-starts[k])
                effort=KP*(desired-q[:14])-KD*v[:14]+kin.data.qfrc_bias[:14]
                ctrl=np.clip(effort,comp.mj_model.actuator_ctrlrange[:,0],comp.mj_model.actuator_ctrlrange[:,1])
                if enabled:
                    # Predict the relative spin without kinetic drag. The rank-one
                    # implicit solve preserves the original continuous friction law.
                    # Native unilateral contact is still solved by the real step.
                    external-=ja*loads["friction_torque_nm"]
                    no_drag=external+kin.data.qfrc_passive-kin.data.qfrc_bias
                    no_drag[:14]+=ctrl
                    inverse_j=np.empty((1,comp.mj_model.nv))
                    mj.mj_solveM(comp.mj_model,kin.data,inverse_j,np.ascontiguousarray(ja[None,:]))
                    mobility=float(inverse_j[0]@ja)
                    free_rate=float(ja@v+dt*(inverse_j[0]@no_drag))
                    _,previous_drag[k]=implicit_turn_drag(free_rate,mobility,dt)
                    external+=ja*previous_drag[k]
                controls[k]=MuJoCoControl(ctrl,qfrc_applied=external)
            if step<steps:
                rt.set_controls(controls)
                rt.step(dt)
        after=rt.musa_device_report() if mode=="musa" else {}
    if mode=="musa" and ("S4000" not in after["device_name"] or after["completed_kernels"]<=before["completed_kernels"]
                         or after["launched_kernels"]!=after["completed_kernels"]):
        raise RuntimeError("missing completed MTT S4000 kernel launches")
    for info in records:
        info["success"]=all(info.pop("last"))
    return records,dict(device=after.get("device_name","cpu"),completed=after.get("completed_kernels",0)-before.get("completed_kernels",0))


def experiment(args):
    out=args.out
    out.mkdir(parents=True,exist_ok=False)
    scene=build_scene(args.task,urdf_path=args.urdf)
    (out/"scene.xml").write_bytes(scene.xml.encode())
    comp=scene.compile()
    if (comp.mj_model.nq,comp.mj_model.nv,comp.mj_model.nu)!=(21,20,14):
        raise RuntimeError("expected two actuated Panda arms and a free workpiece")
    kin=Geometry(scene,comp)
    train=reset_bank(args.task,4011,12)
    hold=reset_bank(args.task,8161,6)
    sampler=StateSampler(len(train),909,mode=args.sampler)
    base=np.array([.25,-.42,-.42])
    cem=TerminalCEM(base,np.array([.56,.75,.75]),512,population=6)
    data=dict(task=args.task,mode=args.mode,train_bank=train,validation_bank=hold,steps=args.steps,
              timestep=scene.timestep,scene_sha256=scene.xml_sha256,updates=[],checks={},gpu_batches=[],
              sampler=args.sampler, reset_sampling=("Uniform over the same bank" if args.sampler=="uniform" else
                  "AlphaNIST eq.4 over a rolling buffer of the behavioral population's binary outcomes"),
              actuation="bounded IK motion targets and joint feedback; privileged observation, no RGB policy",
              thread="pre-engaged compliant lead & kinetic drag" if args.task=="screw_proxy" else None)
    for gen in range(args.generations):
        ids=sampler.sample(3)
        pop=cem.ask()
        wins=[]
        decomp=[]
        probs=sampler.probabilities().tolist()
        for idx in ids:
            rows,hw=run_batch(args.task,scene,comp,kin,[train[idx]]*len(pop),pop,args.mode,args.cache,args.steps)
            data["gpu_batches"].append(hw)
            wins.append([x["success"] for x in rows])
            decomp.append([{k:v for k,v in x.items() if k not in ("history","start")} for x in rows])
        binary=np.asarray(wins).T
        scores=cem.tell(pop,binary)
        sampler.update(np.repeat(ids,len(pop)),np.asarray(wins).ravel())
        data["updates"].append(dict(generation=gen,sampled=ids.tolist(),probabilities=probs,scores=scores.tolist(),
                                    result=binary.tolist(),candidates=pop.tolist(),mean_after=cem.mean.tolist(),metrics=decomp))
        save(out/"progress.json",data)
        print(args.task,"generation",gen,"scores",scores.tolist(),flush=True)
    save(out/"policy.json",dict(task=args.task,optimizer=cem.snapshot(),sampler=sampler.snapshot(),
                               scene_sha256=scene.xml_sha256,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    ids=uniform_validation(len(hold),len(hold),745)
    starts=[hold[i] for i in ids]
    for name,weights in (("hold",base),("initial",base),("trained",cem.mean)):
        rows,hw=run_batch(args.task,scene,comp,kin,starts,[weights]*len(starts),args.mode,args.cache,args.steps,
                          monitor=name)
        data["gpu_batches"].append(hw)
        data["checks"][name]=[]
        for idx,row in zip(ids,rows,strict=True):
            hist=row.pop("history")
            if name=="trained":
                np.savez_compressed(out/f"holdout-{idx}.npz",time_s=np.array([h[0] for h in hist]),
                                    qpos=np.stack([h[1] for h in hist]),model_sha256=scene.xml_sha256)
            data["checks"][name].append(row)
        print(name,"held out",sum(x["success"] for x in rows),"/",len(rows),flush=True)
    if args.audit_seed is not None:
        audit=reset_bank(args.task,args.audit_seed,12)
        data["fresh_validation_bank"]=audit
        for name,weights in (("initial",base),("trained",cem.mean)):
            data["checks"]["fresh_"+name]=[]
            for start in (0,6):
                rows,hw=run_batch(args.task,scene,comp,kin,audit[start:start+6],[weights]*6,
                                  args.mode,args.cache,args.steps)
                data["gpu_batches"].append(hw)
                for row in rows:
                    row.pop("history")
                data["checks"]["fresh_"+name].extend(rows)
            print(name,"fresh",sum(r["success"] for r in data["checks"]["fresh_"+name]),"/ 12",flush=True)
    save(out/"result.json",data)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task",choices=["pin","screw_proxy"],required=True)
    p.add_argument("--mode",choices=["musa","cpu"],default="musa")
    p.add_argument("--sampler",choices=["alphanist","uniform"],default="alphanist")
    p.add_argument("--urdf",type=Path)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--cache",type=Path)
    p.add_argument("--generations",type=int,default=4)
    p.add_argument("--steps",type=int,default=525)
    p.add_argument("--audit-seed",type=int,help="additional fixed bank, opened after freezing the policy")
    args=p.parse_args()
    if args.steps<26 or args.generations<1 or args.audit_seed in (4011,8161):
        p.error("need at least 26 steps and a positive generation count")
    experiment(args)
