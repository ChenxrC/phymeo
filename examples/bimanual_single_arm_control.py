"""Independent single-arm controls for the matched two-arm manipulation task.

The comparator is allowed exact payload pose and native robot kinematics. It
shares no policy weights, visual features or task-stage estimates with training.
Both parked-arm trials use the original MJCF, motor bounds, payload and success
predicate. Results describe these controllers at the recorded seeds.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np

from bimanual_musa_env import (
    ACTION_INCREMENT, BimanualBatchEnv, LOAD_PROFILES, PAD_RADIUS, SIDES, STANCES, TASKS, TIMESTEP, TOOL_PROFILES,
)


HOLDOUT = (43001, 43002, 43003, 43004)
PROBES = ("tracking", "saturated")
EPISODE_SECONDS = 3.5
SETTLE_TIME = 2.95


def _ramp(t, start, end):
    x = np.clip((t - start) / (end - start), 0., 1.)
    return x*x*(3. - 2.*x)


class Comparator:
    def __init__(self, world, index):
        import mujoco as mj

        self.mj, self.world, self.side = mj, world, index
        self.model, self.scene = world.model, world.scene
        self.data = mj.MjData(self.model)
        self.site = self.model.site(f"{SIDES[index]}_tool").id
        self.columns = world.slots // 7 == index
        self.qidx = world.q_indices[self.columns]
        self.vidx = world.v_indices[self.columns]
        self.sign = (1, -1)[index]
        self.contact_y = .082*self.sign
        self.last_stage = None

    def _carry(self, t, pos, rot):
        # Follow the moving side face, including its real tilt. Once seated,
        # ask a single frictional contact to lift and transport the whole mass.
        compression = .048*_ramp(t, .10, .58)
        release = _ramp(t, 2.75, SETTLE_TIME)
        local = np.array((0., self.sign*(.11 + PAD_RADIUS + .008 - compression
                                         + .09*release), -.009))
        p = pos + rot @ local
        lift = _ramp(t, .58, 1.24)*(1.-_ramp(t, 2.25, 2.78))
        transfer = _ramp(t, 1.22, 2.18)
        goal_error = np.asarray(self.scene.goal_xy) - pos[:2]
        p[:2] += transfer * np.clip(goal_error, -.075, .075)
        p[2] = max(PAD_RADIUS+.001, self.scene.start_xyz[2]-.009 + .10*lift)
        stages = ((.16, "approach"), (.58, "engage"), (1.24, "lift"),
                  (2.25, "transport"), (2.75, "lower"), (SETTLE_TIME, "release"))
        return p, next((name for end, name in stages if t < end), "settle")

    def _push(self, t, pos, rot, tool):
        # A single tool may cross the centre line to get a useful turning
        # moment. Moving behind the bar before traversing avoids pushing with
        # the wrong torque just because the arm started on one side.
        delta = np.asarray(self.scene.goal_xy)-pos[:2]
        error = self.scene.goal_yaw-np.arctan2(rot[1, 0], rot[0, 0])
        error = np.arctan2(np.sin(error), np.cos(error))
        contact_y = np.clip(-.32*error, -.105, .105)
        clearance = .023
        if t < .45:
            self.contact_y += float(np.clip(contact_y-self.contact_y, -.008, .008))
            press = -clearance
            stage = "position-behind-bar"
        else:
            self.contact_y += float(np.clip(contact_y-self.contact_y, -.003, .003))
            # Neutral backoff lets the floor accept the load during final dwell.
            at_pose = np.linalg.norm(delta) < .014 and abs(error) < np.deg2rad(4)
            press = (-clearance if at_pose else
                     np.clip(.8*float(delta @ rot[:2, 0])+.13*abs(error), .008, .060))
            stage = "unload" if at_pose else "push-and-turn"
        point = pos + rot @ np.array((-.028-PAD_RADIUS+press, self.contact_y, 0.))
        point[1] += np.clip(delta[1], -.025, .025)
        point[2] = max(self.scene.start_xyz[2]+.001, PAD_RADIUS+.001)
        if t < .45 and np.linalg.norm(tool[:2]-point[:2]) < .015:
            stage = "ready-behind-bar"
        return point, stage

    def commands(self, q, v, t):
        mj, model, data = self.mj, self.model, self.data
        data.qpos[:] = q
        # Zero velocity gives static robot gravity for this comparator's motor
        # feedforward. The live simulator receives neither qpos nor applied forces.
        mj.mj_kinematics(model, data)
        mj.mj_comPos(model, data)
        mj.mj_comVel(model, data)
        gravity = np.empty(model.nv)
        mj.mj_rne(model, data, 0, gravity)
        pos = data.body(self.scene.object_body).xpos
        rot = data.body(self.scene.object_body).xmat.reshape(3, 3)
        tool = data.site_xpos[self.site]
        if self.scene.task == "carry":
            target, stage = self._carry(t, pos, rot)
        else:
            target, stage = self._push(t, pos, rot, tool)
        jac = np.empty((3, model.nv))
        mj.mj_jacSite(model, data, jac, None, self.site)
        j = jac[:, self.vidx]
        velocity = j @ v[self.vidx]
        error = np.clip(target-tool, -.08, .08)
        dx = error - .018*velocity
        dq = j.T @ np.linalg.solve(j @ j.T + .018**2*np.eye(3), dx)
        # Joint-PD position equivalent of the motor gravity bias.
        desired = q[self.qidx] + dq + gravity[self.vidx]/self.world.kp[self.columns]
        action = np.zeros(14)
        action[self.world.slots[self.columns]] = np.clip(
            (desired-self.world.target[self.columns])/ACTION_INCREMENT, -1., 1.)
        directions = np.zeros((2, 3))
        # A velocity-damped, direction-changing maximum-motor-effort challenge.
        directions[self.side] = np.clip(14.*error - .45*velocity, -1., 1.)
        if self.scene.task == "carry" and stage in ("lift", "transport"):
            directions[self.side, 1] -= self.sign*.45
            directions[self.side, 2] += .75
        elif self.scene.task == "push_t" and stage == "push-and-turn":
            directions[self.side, :2] += .4*rot[:2, 0]
        self.last_stage = stage
        return action, directions, target.tolist(), tool.copy().tolist()


def _summarize(trace, final, side, full_protocol, weight, initial_xyz, probe):
    active = [sample for sample in trace if sample["normal_n"] >= 1.]
    stressed = [sample for sample in active if sample["control_mode"] == "saturated"
                and sample["peak_fraction"] >= .99]
    best = min(trace, key=lambda sample: sample["goal_error_xy_m"])
    travel = [float(np.linalg.norm(np.subtract(row["payload_xyz_m"][:2], initial_xyz[:2]))) for row in trace]
    parked_peak = final["peak_pad_normal_n"][1-side]
    engaged = any(row["stage"] in ("lift", "transport", "push-and-turn") for row in active)
    return {
        "success": final["success"], "complete_protocol_or_success": bool(full_protocol or final["success"]),
        "parked_arm_peak_contact_n": parked_peak, "passive_arm_contact_over_1n": parked_peak > 1.,
        "peak_active_normal_n": final["peak_pad_normal_n"][side],
        "peak_active_vertical_n": final["peak_pad_vertical_n"][side],
        "peak_active_vertical_over_weight": final["peak_pad_vertical_n"][side]/weight,
        "max_clearance_m": final["max_clearance_m"], "carried_airborne": final["carried_airborne"],
        "sampled_peak_translation_m": max(travel), "best_xy_error_m": best["goal_error_xy_m"],
        "yaw_error_at_best_xy_rad": best["goal_yaw_error_rad"],
        "final_xy_error_m": final["goal_error_xy_m"], "final_yaw_error_rad": final["goal_yaw_error_rad"],
        "longest_goal_dwell_sampled_s": max(row["goal_dwell_s"] for row in trace),
        "loaded_decision_endpoints": len(active), "loaded_saturated_decision_endpoints": len(stressed),
        "valid_loaded_negative_for_this_probe": bool(
            not final["success"] and full_protocol and parked_peak < 1. and engaged
            and (probe != "saturated" or bool(stressed))),
        "loaded_stages": sorted({row["stage"] for row in active}),
        "first_real_pad_load_s": active[0]["time_s"] if active else None,
        "modeled_force_scope": "native last-step contact wrench sampled at each decision; cumulative peak over source ticks",
    }


def run_trial(task, seed, probe, *, mode, cache_root=None, seconds=EPISODE_SECONDS, action_repeat=5,
              tool_profile="single-pad", load_profile="standard", stance="original"):
    if probe not in PROBES or seconds <= 0. or not np.isfinite(seconds):
        raise ValueError("invalid probe or duration")
    ticks = int(np.ceil(seconds/(TIMESTEP*action_repeat)))
    keys = (0, 1)
    started = time.perf_counter()
    with BimanualBatchEnv(task, seed, world_keys=keys, mode=mode, cache_root=cache_root,
                          horizon=ticks, action_repeat=action_repeat, tool_profile=tool_profile,
                          load_profile=load_profile, stance=stance) as env:
        states = env.reset(active_arms=dict(zip(keys, SIDES, strict=True)))
        actors = {k: Comparator(env.worlds[k], k) for k in keys}
        traces, final = {k: [] for k in keys}, {}
        for _ in range(ticks):
            t = float(states[0][-1]["time_s"])
            # Approach under control first. Keep the saturated challenge bounded
            # to a window where the task asks for substantial load. Later return
            # to tracking for steering, controlled unloading, and goal dwell.
            saturate = probe == "saturated" and .65 <= t < 1.55
            commands, targets, tools, stages = {}, {}, {}, {}
            for k in keys:
                info = states[k][-1]
                action, direction, targets[k], tools[k] = actors[k].commands(info["qpos"], info["qvel"], t)
                commands[k] = direction if saturate else action
                stages[k] = actors[k].last_stage
            states = env.step_saturated(commands) if saturate else env.step(commands)
            for k in keys:
                if k in final:
                    continue
                _, _, terminated, truncated, info = states[k]
                ctrl = info["applied_ctrl"][actors[k].columns]
                limits = env.model.actuator_ctrlrange[actors[k].columns, 1]
                sample = {key: info[key] for key in (
                    "tick", "time_s", "payload_xyz_m", "payload_yaw_rad", "goal_error_xy_m",
                    "goal_yaw_error_rad", "goal_dwell_s", "clearance_m", "floor_vertical_n",
                    "payload_com_speed_m_s", "payload_spin_rad_s")}
                sample.update(stage=stages[k], control_mode="saturated" if saturate else "tracking",
                              evaluation_input_time_s=info["contact_input_time_s"],
                              target_before_step_m=targets[k], tool_before_step_m=tools[k],
                              normal_n=info["pad_normal_n"][k], vertical_n=info["pad_vertical_n"][k],
                              motor_ctrl_nm=ctrl.tolist(), peak_fraction=float(np.max(abs(ctrl)/limits)))
                traces[k].append(sample)
                if terminated or truncated:
                    final[k] = info
            if len(final) == len(keys):
                break
        result = {"task": task, "seed": seed, "probe": probe, "mode": mode, "backend": final[0]["backend"],
                  "arm_dofs": env.scene.arm_dofs, "xml_sha256": env.scene.xml_sha256,
                  "mass_kg": env.scene.mass, "tool_profile": tool_profile, "load_profile": load_profile,
                  "stance": stance,
                  "gravity_m_s2": 9.81,
                  "payload_joint": "payload_free",
                  "timestep_s": TIMESTEP, "action_repeat": action_repeat, "requested_seconds": seconds,
                  "initial_xyz_m": env.scene.start_xyz, "goal_xy_m": env.scene.goal_xy,
                  "goal_yaw_rad": env.scene.goal_yaw, "contact_phase": final[0]["contact_phase"],
                  "elapsed_wall_s": time.perf_counter()-started, "arms": {}}
        for k, side in zip(keys, SIDES, strict=True):
            result["arms"][side] = {"summary": _summarize(traces[k], final[k], k,
                                                        seconds >= EPISODE_SECONDS, env.scene.mass*9.81,
                                                        env.scene.start_xyz, probe),
                                    "trace": traces[k]}
        if mode == "musa":
            result["device"] = env.runtime.musa_device_report()
        return result


def _capacity_at_pose(model, data, q):
    """Static native motor and Coulomb-pyramid feasibility at measured contact points."""
    import mujoco as mj
    from scipy.optimize import linprog

    data.qpos[:] = q
    data.qvel[:] = 0
    mj.mj_fwdPosition(model, data)
    mj.mj_comVel(model, data)
    bias = np.empty(model.nv)
    mj.mj_rne(model, data, 0, bias)
    body = model.body("payload").id
    accepted_pad_names = {f"{side}_{suffix}": side for side in SIDES
                          for suffix in ("pad", "pad_a", "pad_b")}
    pads = {i: accepted_pad_names[model.geom(i).name] for i in range(model.ngeom)
            if model.geom(i).name in accepted_pad_names}
    contacts = {side: [] for side in SIDES}
    arms = {}
    for contact in data.contact:
        a, b = map(int, contact.geom)
        if a in pads and model.geom_bodyid[b] == body:
            pad, sign = a, 1.
        elif b in pads and model.geom_bodyid[a] == body:
            pad, sign = b, -1.
        else:
            continue
        if contact.dist >= contact.includemargin:
            continue
        if contact.dim != 3:
            raise ValueError("capacity tool requires native condim=3 pad contacts")
        frame = np.asarray(contact.frame).reshape(3, 3)
        m1, m2 = contact.friction[:2]
        # Columns are forces ON the payload. Nonnegative combinations give
        # exactly MuJoCo's 4-ray condim3 pyramidal friction cone.
        rays = sign * np.column_stack((frame[0]+m1*frame[1], frame[0]-m1*frame[1],
                                       frame[0]+m2*frame[2], frame[0]-m2*frame[2]))
        pos = np.copy(contact.pos)
        jac = np.empty((3, model.nv))
        mj.mj_jac(model, data, jac, None, pos, int(model.geom_bodyid[pad]))
        contacts[pads[pad]].append(dict(rays=rays, jac=jac, point=pos, pad=model.geom(pad).name,
                                        normal=(sign*frame[0]).tolist(), mu=[float(m1), float(m2)]))
    for side, entries in contacts.items():
        if not entries:
            continue
        actuator_ids = [i for i in range(model.nu) if model.actuator(i).name.startswith(f"{side}_m")]
        joints = model.actuator_trnid[actuator_ids, 0]
        vid = model.jnt_dofadr[joints]
        gear = model.actuator_gear[actuator_ids, 0]
        if not np.allclose(gear, 1.) or not np.all(model.actuator_ctrllimited[actuator_ids]):
            raise ValueError("capacity tool requires direct unit-gear torque motors with native limits")
        if np.any(model.actuator_forcelimited[actuator_ids]) or np.any(model.jnt_actfrclimited[joints]):
            raise ValueError("additional native motor/joint force limits need explicit LP rows")
        bounds = model.actuator_ctrlrange[actuator_ids]
        joint_map = np.hstack([c["jac"][:, vid].T @ c["rays"] for c in entries])
        arm_a = np.vstack((joint_map, -joint_map))
        arm_b = np.r_[bounds[:, 1]-bias[vid], bias[vid]-bounds[:, 0]]
        arms[side] = dict(rays=np.hstack([c["rays"] for c in entries]),
                          moments=np.hstack([np.cross(c["point"]-data.xipos[body], c["rays"].T).T
                                             for c in entries]),
                          a=arm_a, b=arm_b, bias=bias[vid].copy(),
                          contacts=[dict(pad=c["pad"], contact_xyz_m=c["point"].tolist(),
                                         normal_into_payload=c["normal"], mu=c["mu"]) for c in entries])
    weight = float(-model.opt.gravity[2]*model.body_mass[body])

    def solve(objective, inequalities, capacity, equations=None, target=None):
        result = linprog(objective, A_ub=inequalities, b_ub=capacity,
                         A_eq=equations, b_eq=target, bounds=(0., None), method="highs")
        return {"status": int(result.status), "value": float(result.fun) if result.success else None,
                "x": result.x if result.success else None}

    def interval_result(result, sign=1.):
        return {"status": result["status"], "value_n": None if result["value"] is None else sign*result["value"]}

    reports = {}
    for side, arm in arms.items():
        up = arm["rays"][2]
        peak = solve(-up, arm["a"], arm["b"])
        nlo = solve(np.ones_like(up), arm["a"], arm["b"], up[None, :], [weight/2])
        nhi = solve(-np.ones_like(up), arm["a"], arm["b"], up[None, :], [weight/2])
        reports[side] = {
            "contacts": arm["contacts"], "gravity_bias_nm": arm["bias"].tolist(),
            "unbalanced_local_max_vertical": interval_result(peak, -1.),
            "min_normal_at_half_weight": interval_result(nlo),
            "max_normal_at_half_weight": interval_result(nhi, -1.),
        }
        if len(arm["contacts"]) == 1:
            reports[side].update(arm["contacts"][0])
    row = {"contacts": sum(len(values) for values in contacts.values()), "bilateral": len(arms) == 2,
           "weight_n": weight, "arms": reports}
    if len(arms) != 2:
        return row
    left, right = [arms[side] for side in SIDES]
    flat = np.hstack((left["rays"], right["rays"]))
    moment = np.hstack([arm["moments"] for arm in (left, right)])
    la, ra = left["a"], right["a"]
    nl, nr = la.shape[1], ra.shape[1]
    inequalities = np.block([[la, np.zeros((len(la), nr))], [np.zeros((len(ra), nl)), ra]])
    capacity = np.r_[left["b"], right["b"]]
    # No floor or merely convenient support reactions are available in these LPs.
    # All normal/tangent components remain jointly constrained by both arm motors.
    for label, eq in (("force_only", flat[:2]), ("force_and_moment", np.vstack((flat[:2], moment)))):
        optimum = solve(-flat[2], inequalities, capacity, eq, np.zeros(len(eq)))
        summary = interval_result(optimum, -1.)
        x = optimum["x"]
        if x is not None:
            summary.update(world_force_n=(flat @ x).tolist(), moment_about_com_nm=(moment @ x).tolist(),
                           normal_n=[float(x[:nl].sum()), float(x[nl:].sum())],
                           vertical_n=[float(left["rays"][2] @ x[:nl]), float(right["rays"][2] @ x[nl:])])
        row[label] = summary
    # If exact torque equilibrium is impossible, distinguish substantial
    # tipping from an almost-centred two-point grip by retaining the residual.
    eq = np.c_[flat, np.zeros(3)]
    torque_ineq = np.vstack((np.c_[moment, -np.ones(3)], np.c_[-moment, -np.ones(3)]))
    min_tipping = solve(np.r_[np.zeros(nl+nr), 1.],
                        np.vstack((np.c_[inequalities, np.zeros(len(capacity))], torque_ineq)),
                        np.r_[capacity, np.zeros(6)], eq, [0., 0., weight])
    row["smallest_peak_axis_moment_nm_at_weight"] = min_tipping["value"]
    row["at_weight_lp_status"] = min_tipping["status"]
    return row


def audit_capacity(path, *, stride=5, xml_path=None):
    import mujoco as mj

    if stride < 1:
        raise ValueError("capacity stride must be positive")
    xml_path = path.with_suffix(".xml") if xml_path is None else xml_path
    xml = xml_path.read_bytes()
    sha = hashlib.sha256(xml).hexdigest()
    with np.load(path, allow_pickle=False) as series:
        if str(series["model_sha256"].item()) != sha:
            raise ValueError("trajectory and original XML do not match")
        q, ticks, times = series["qpos"], series["ticks"], series["time_s"]
    model = mj.MjModel.from_xml_string(xml.decode("utf-8"))
    data = mj.MjData(model)
    if (q.ndim != 2 or q.shape != (len(ticks), model.nq) or not len(q)
            or times.shape != (len(q),) or not np.isfinite(q).all() or not np.isfinite(times).all()
            or np.any(np.diff(ticks) <= 0) or np.any(np.diff(times) <= 0)):
        raise ValueError("invalid native trajectory dimensions or timestamps")
    ids = list(range(0, len(q), stride))
    if ids[-1] != len(q)-1:
        ids.append(len(q)-1)
    rows = [dict(tick=int(ticks[i]), time_s=float(times[i]), **_capacity_at_pose(model, data, q[i]))
            for i in ids]
    return {"schema": "bimanual-single-arm-and-grasp-static-capacity-v1", "trajectory": str(path),
            "model_sha256": sha, "source_path": str(xml_path), "sampled_indices": ids,
            "assumptions": ["sampled geometry held fixed", "static arm bias and original unit-gear motor limits",
                            "native condim3 point contacts sharing the motor limits on each arm",
                            "payload supports its own weight without the table",
                            "tangent-pyramid coefficients are native contact values",
                            "local arm maximum allows unconstrained horizontal reactions; combined solve balances them",
                            "moment residual is minimized as a diagnostic only; exact feasibility shown independently"],
            "lp_status": {"0": "optimal", "2": "infeasible", "3": "unbounded"},
            "bilateral_samples": sum(row["bilateral"] for row in rows), "samples": rows}


def _save(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, mode="w", encoding="utf-8", delete=False) as stream:
        tmp = Path(stream.name)
        try:
            json.dump(report, stream, allow_nan=False, separators=(",", ":"))
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            stream.close()
            tmp.unlink(missing_ok=True)
            raise
    try:
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tasks", choices=TASKS, nargs="+", default=TASKS)
    p.add_argument("--seeds", type=int, nargs="+", default=HOLDOUT)
    p.add_argument("--probes", choices=PROBES, nargs="+", default=PROBES)
    p.add_argument("--mode", choices=("cpu", "musa"))
    p.add_argument("--tool-profile", choices=TOOL_PROFILES, default="single-pad")
    p.add_argument("--load-profile", choices=LOAD_PROFILES, default="standard")
    p.add_argument("--stance", choices=STANCES, default="original")
    p.add_argument("--seconds", type=float, default=EPISODE_SECONDS)
    p.add_argument("--action-repeat", type=int, default=5)
    p.add_argument("--cache-root", type=Path)
    p.add_argument("--capacity-trace", type=Path, help="audit a saved qpos/ticks/time_s NPZ without simulation")
    p.add_argument("--model-xml", type=Path, help="defaults to the trajectory's sibling XML")
    p.add_argument("--capacity-stride", type=int, default=5)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.capacity_trace is not None:
        report = audit_capacity(args.capacity_trace, xml_path=args.model_xml, stride=args.capacity_stride)
        _save(args.output, report)
        print(f"Audited {len(report['samples'])} postures, {report['bilateral_samples']} bilateral", flush=True)
        return
    if args.mode is None:
        p.error("--mode is required for simulation")
    if args.load_profile == "side-lip" and any(task != "carry" for task in args.tasks):
        p.error("--load-profile side-lip requires --tasks carry")
    if args.stance == "bent-elbow" and any(task != "carry" for task in args.tasks):
        p.error("--stance bent-elbow requires --tasks carry")
    report = {"schema": "bimanual-independent-single-arm-v1", "cases": [],
              "saturation_window_s": [.65, 1.55], "default_episode_s": EPISODE_SECONDS}
    for task in args.tasks:
        for seed in args.seeds:
            for probe in args.probes:
                case = run_trial(task, seed, probe, mode=args.mode, cache_root=args.cache_root,
                                 seconds=args.seconds, action_repeat=args.action_repeat,
                                 tool_profile=args.tool_profile, load_profile=args.load_profile,
                                 stance=args.stance)
                report["cases"].append(case)
                _save(args.output, report)
                print(task, seed, probe, {side: a["summary"] for side, a in case["arms"].items()}, flush=True)


if __name__ == "__main__":
    main()
