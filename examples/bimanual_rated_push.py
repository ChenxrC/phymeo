"""Rated, sustained straight push of a free 9 kg T by one or both wrist drives.

An additional machine-capacity experiment, separate from the rotating PushT
task. Start all three worlds in exactly the same pose. Both rotary arms hold
their mounting pose. Only cylinder power is switched between the three trials.
"""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from xml.etree import ElementTree as ET

import mujoco as mj
import numpy as np

from bimanual_musa_env import SIDES, _markers, _numbers, _rotation, make_scene
from phymeo.backends.mujoco_runtime import MuJoCoControl, MuJoCoInitialState, MuJoCoRuntime
from phymeo.builders.mujoco_compiler import compile_mujoco_xml


RATED_N = 28.
TIMESTEP = .0005
TIP_RADIUS = .006
ELBOW_RAD = -.30
STROKE_M = np.array((-.045, .32))
RUN_S = 3.20
MU = .52


def make_fixture(seed):
    original = make_scene("push_t", seed, tool_profile="single-pad")
    root = ET.fromstring(original.xml)
    root.set("model", f"rated-straight-push-{seed}")
    root.find("option").set("timestep", str(TIMESTEP))
    world, motors = root.find("worldbody"), root.find("actuator")
    for element in list(world):
        if element.get("name", "").startswith("goal_marker_") or element.get("name") == "task_goal":
            world.remove(element)
    x, y, _ = original.start_xyz
    _markers(world, (x + .12, y), 0., original.boxes)
    for side, count, height, spread in zip(SIDES, original.arm_dofs, (.012, .026), (.050, .095), strict=True):
        wrist = root.find(f".//body[@name='{side}_link{count - 1}']")
        mounting = wrist.find(f"geom[@name='{side}_pad']")
        flange = np.fromstring(mounting.attrib["pos"], sep=" ")
        wrist.remove(mounting)
        # Recess the whole arm. A thin lateral bracket places each actual tip
        # on the centre line while the arm housings stay on opposite sides.
        base = root.find(f".//body[@name='{side}_link0']")
        base.set("pos", _numbers(np.fromstring(base.attrib["pos"], sep=" ") - (.12, 0., 0.)))
        chain = [root.find(f".//body[@name='{side}_link{i}']") for i in range(count)]
        if any(any(tag in link.attrib for tag in ("quat", "axisangle", "xyaxes", "zaxis", "euler"))
               for link in chain):
            raise ValueError("the rated fixture expects unrotated link frames at the mounting pose")
        translations = [np.fromstring(link.attrib["pos"], sep=" ") for link in chain]
        nominal = flange + np.sum(translations, axis=0)
        # Use elbow flexion within the original hardware range to shorten
        # the vertical moment arm of rated horizontal loading. Mount the arm
        # lower to keep its wrist location and horizontally keyed tool axis.
        c, s = np.cos(ELBOW_RAD), np.sin(ELBOW_RAD)
        bent = np.array(((c,0.,s), (0.,1.,0.), (-s,0.,c)))
        elbow_joint = np.sum(translations[:3], axis=0)
        delta = (np.eye(3)-bent) @ (nominal-elbow_joint)
        base.set("pos", _numbers(translations[0] + delta))
        tip = np.array((x - .028 - TIP_RADIUS, y, height)) - nominal
        ET.SubElement(wrist, "geom", name=f"{side}_housing", type="box",
                      pos=_numbers(flange), size=".030 .015 .010", mass=".12",
                      rgba=".35 .39 .43 1", contype="0", conaffinity="0")
        rod = ET.SubElement(wrist, "body", name=f"{side}_plunger", pos=_numbers(flange),
                            euler=_numbers((0., -ELBOW_RAD, 0.)))
        ET.SubElement(rod, "joint", name=f"{side}_stroke", type="slide", axis="1 0 0",
                      limited="true", range=_numbers(STROKE_M), damping="0", frictionloss="0")
        # Each wrist now owns a symmetric fork. An off-yaw payload touches
        # its advanced tine first, producing a physical restoring yaw moment.
        tip_a, tip_b = tip - (0., spread, 0.), tip + (0., spread, 0.)
        heel_a, heel_b = tip_a - (.022,0.,0.), tip_b - (.022,0.,0.)
        closest = heel_a if abs(heel_a[1]) < abs(heel_b[1]) else heel_b
        for index, (a, b) in enumerate(((np.zeros(3), closest), (heel_a, heel_b),
                                        (heel_a, tip_a), (heel_b, tip_b))):
            ET.SubElement(rod, "geom", name=f"{side}_rod{index}", type="capsule",
                          fromto=_numbers((*a, *b)), size=".003", mass=".015",
                          rgba=".60 .65 .70 1", contype="0", conaffinity="0")
        for label, point in (("a", tip_a), ("b", tip_b)):
            ET.SubElement(rod, "geom", name=f"{side}_tip_{label}", type="sphere", pos=_numbers(point),
                          size=str(TIP_RADIUS), mass=".010", rgba=".12 .20 .26 1",
                          condim="1", priority="2", contype="4", conaffinity="3")
        ET.SubElement(rod, "site", name=f"{side}_fork_center", pos=_numbers(tip), size=".002", rgba="0 0 0 0")
        ET.SubElement(motors, "motor", name=f"{side}_drive", joint=f"{side}_stroke", gear="1",
                      ctrllimited="true", ctrlrange="0 28", forcelimited="true", forcerange="0 28")
    return ET.tostring(root, encoding="utf-8"), original


def _smooth(t):
    x = np.clip(t, 0., 1.)
    return x*x*(3.-2.*x)


class Fixture:
    def __init__(self, model, original):
        self.model, self.original = model, original
        self.data = mj.MjData(model)
        self.payload = model.body("payload").id
        joint = model.joint("payload_free").id
        self.q0 = model.jnt_qposadr[joint]
        self.v0 = model.jnt_dofadr[joint]
        self.payload_geoms = {model.geom(f"payload_{i}").id for i in range(len(original.boxes))}
        self.table = model.geom("table").id
        self.tips = [model.geom(f"{s}_tip_{label}").id for s in SIDES for label in ("a","b")]
        self.centers = [model.site(f"{s}_fork_center").id for s in SIDES]
        arm_shell = [i for i in range(model.ngeom) if model.geom_bodyid[i] not in (0, self.payload)
                     and i not in self.tips]
        self.capsules = np.array([i for i in arm_shell if model.geom_type[i] == mj.mjtGeom.mjGEOM_CAPSULE])
        self.boxes = np.array([i for i in arm_shell if model.geom_type[i] == mj.mjtGeom.mjGEOM_BOX])
        if len(self.capsules) + len(self.boxes) != len(arm_shell):
            raise ValueError("an arm surface lacks a signed separation check")
        self.slides = np.array([model.joint(s + "_stroke").id for s in SIDES])
        self.sl_q, self.sl_v = model.jnt_qposadr[self.slides], model.jnt_dofadr[self.slides]
        self.drives = [model.actuator(s + "_drive").id for s in SIDES]
        self.arm = []
        self.mount = model.qpos0.copy()
        for side, joints in zip(SIDES, original.arm_joint_names, strict=True):
            j = [model.joint(name).id for name in joints]
            motor = [model.actuator(f"{side}_m{i}").id for i in range(len(j))]
            self.arm.append((model.jnt_qposadr[j], model.jnt_dofadr[j], motor))
            self.mount[model.jnt_qposadr[j[2]]] = ELBOW_RAD
        self.start = np.array(original.start_xyz)
        self.weight = float(model.body_mass[self.payload] * 9.81)
        self.com_local = np.array(model.body_ipos[self.payload])

    def control(self, state, enabled):
        m, d = self.model, self.data
        q, v = np.asarray(state.q), np.asarray(state.v)
        d.qpos[:] = q
        d.qvel[:] = 0.
        mj.mj_kinematics(m, d)
        mj.mj_comPos(m, d)
        mj.mj_comVel(m, d)
        g = np.empty(m.nv)
        mj.mj_rne(m, d, 0, g)
        t = state.time
        envelope = _smooth((t-.15)/.30) * (1.-_smooth((t-2.65)/.25))
        axes = d.xaxis[self.slides].copy()
        tip_xyz = d.geom_xpos[self.tips].copy()
        ctrl = np.zeros(m.nu)
        relative_approach, demands = [], []
        quat = q[self.q0+3:self.q0+7]
        origin = q[self.q0:self.q0+3]
        rotation = _rotation(quat)
        omega = rotation @ v[self.v0+3:self.v0+6]
        # Project every (including collision-disabled) link, housing and rod
        # onto the world normal of the flat rear face of the free payload.
        # A positive margin proves those surfaces stay physically outside it.
        axis = rotation[:, 0]
        cap_rotation = d.geom_xmat[self.capsules].reshape(-1, 3, 3)
        cap_bound = m.geom_size[self.capsules, 0] + m.geom_size[self.capsules, 1]*abs(cap_rotation[:, :, 2] @ axis)
        box_rotation = d.geom_xmat[self.boxes].reshape(-1, 3, 3)
        box_bound = np.sum(m.geom_size[self.boxes]*abs(box_rotation.transpose(0, 2, 1) @ axis), axis=1)
        furthest = max(np.max((d.geom_xpos[self.capsules]-origin) @ axis + cap_bound),
                       np.max((d.geom_xpos[self.boxes]-origin) @ axis + box_bound))
        non_tip_gap = float(-self.original.boxes[0][1][0] - furthest)
        for i, (qs, vs, motor) in enumerate(self.arm):
            jac = np.empty((3, m.nv))
            mj.mj_jacSite(m, d, jac, None, self.centers[i])
            for k in (2*i, 2*i+1):
                pt_jac = np.empty_like(jac)
                mj.mj_jacGeom(m, d, pt_jac, None, self.tips[k])
                payload_velocity = v[self.v0:self.v0+3] + np.cross(omega, tip_xyz[k] - origin)
                relative_approach.append(float((pt_jac @ v - payload_velocity) @ axes[i]))
            # A nonnegative flow governor reduces cylinder force above 45 mm/s.
            # The controller never reads object pose/velocity or measured contacts.
            effort = max(0., min(RATED_N*envelope, RATED_N + 360.*(.045 - v[self.sl_v[i]])))
            effort *= float(enabled[i])
            ctrl[self.drives[i]] = effort
            tau = g[vs] + jac[:, vs].T @ (effort*axes[i]) + 220.*(self.mount[qs]-q[qs]) - 7.*v[vs]
            demands.append(tau.tolist())
            ctrl[motor] = np.clip(tau, *m.actuator_ctrlrange[motor].T)
        return ctrl, axes, tip_xyz, relative_approach, non_tip_gap, demands

    def measure(self, state, ctrl, axes, tip_xyz, approach, non_tip_gap, demands):
        step = state.last_step
        force = np.zeros((2, 3))
        floor = np.zeros(3)
        points = [[], []]
        tip_force = np.zeros((4,3))
        unexpected_pairs = {}
        foreign = 0.
        for contact in step.contacts:
            a, b = map(int, contact.geometry_ids)
            wrench = np.asarray(contact.wrench_contact_frame)
            if wrench[0] <= 0.:
                continue
            f = np.asarray(contact.frame).reshape(3, 3).T @ wrench[:3]
            if (a in self.payload_geoms) == (b in self.payload_geoms):
                foreign += float(np.linalg.norm(f))
                pair = "+".join(sorted((self.model.geom(a).name, self.model.geom(b).name)))
                unexpected_pairs[pair] = unexpected_pairs.get(pair, 0.) + float(np.linalg.norm(f))
                continue
            other = a if b in self.payload_geoms else b
            if a in self.payload_geoms:
                f = -f
            if other == self.table:
                floor += f
            elif other in self.tips:
                k = self.tips.index(other)
                i = k//2
                tip_force[k] += f
                force[i] += f
                points[i].append(list(map(float, contact.position)))
            else:
                foreign += float(np.linalg.norm(f))
                pair = "+".join(sorted((self.model.geom(a).name, self.model.geom(b).name)))
                unexpected_pairs[pair] = unexpected_pairs.get(pair, 0.) + float(np.linalg.norm(f))
        q = np.asarray(state.q)
        xyz, quat = q[self.q0:self.q0+3], q[self.q0+3:self.q0+7]
        rot = _rotation(quat)
        com_delta = xyz - self.start + (rot-np.eye(3)) @ self.com_local
        return dict(tick=int(state.tick), input_time_s=float(step.input_time),
                    time_s=float(state.time), contact_phase=step.contact_phase,
                    drive_command_n=ctrl[self.drives].tolist(),
                    actual_drive_n=np.asarray(step.qfrc_actuator)[self.sl_v].tolist(),
                    input_axes=axes.tolist(), input_tip_xyz_m=tip_xyz.tolist(),
                    input_non_tip_separation_m=non_tip_gap,
                    input_relative_axial_velocity_m_s=[float(np.mean(approach[i:i+2])) for i in (0,2)],
                    individual_approach_m_s=approach,
                    individual_axial_force_n=np.sum(tip_force*np.repeat(axes,2,axis=0),axis=1).tolist(),
                    rotary_torque_demand_nm=demands, rotary_torque_control_nm=[ctrl[motors].tolist() for _,_,motors in self.arm],
                    stroke_m=q[self.sl_q].tolist(), stroke_velocity_m_s=np.asarray(state.v)[self.sl_v].tolist(),
                    payload_xyz_m=xyz.tolist(), payload_quat_wxyz=quat.tolist(), delta_com_m=com_delta.tolist(),
                    tilt_rad=float(np.arccos(np.clip(rot[2, 2], -1., 1.))),
                    yaw_rad=float(np.arctan2(rot[1, 0], rot[0, 0])),
                    force_on_payload_by_tip_world_n=force.tolist(),
                    axial_tip_load_n=np.sum(force*axes, axis=1).tolist(),
                    contact_points_m=points, floor_on_payload_world_n=floor.tolist(),
                    unexpected_contact_force_n=foreign, unexpected_pairs_n=unexpected_pairs)


def _longest(mask):
    longest = run = 0
    for present in mask:
        run = run + 1 if present else 0
        longest = max(longest, run)
    return longest * TIMESTEP


def summarize(rows, active, weight):
    def col(name):
        return np.array([row[name] for row in rows])

    t = col("time_s")
    steady = (t >= .45 + TIMESTEP) & (t <= 2.65)
    x = col("delta_com_m")
    loads = col("axial_tip_load_n")
    drive = col("actual_drive_n")
    axes = col("input_axes")
    floor = col("floor_on_payload_world_n")
    strokes = col("stroke_m")
    enabled = np.array([active in ("both", side) for side in SIDES])
    loaded = loads >= (18. if active == "both" else 24.)
    contiguous = _longest(steady & np.all(loaded[:, enabled], axis=1))
    simultaneous = _longest(steady & np.all(loads >= 18., axis=1))
    onset = (loads > 1.) & ~np.vstack((np.zeros((1, 2), bool), loads[:-1] > 1.))
    speeds = np.maximum(0., col("input_relative_axial_velocity_m_s"))
    hit_speed = float(np.max(speeds[onset], initial=0.))
    individual_load = col("individual_axial_force_n")
    individual_onset = (individual_load > 1.) & ~np.vstack((np.zeros((1,4),bool), individual_load[:-1] > 1.))
    hit_speed = max(hit_speed, float(np.max(col("individual_approach_m_s")[individual_onset],initial=0.)))
    min_reserve = float(np.min(np.minimum(strokes - STROKE_M[0], STROKE_M[1] - strokes)))
    single_support_margin = (MU*floor[:, 2]/np.sqrt(2.) - np.sum(col("force_on_payload_by_tip_world_n")[:, :, 0], axis=1))
    gates = dict(continuous_rated_load=contiguous >= 1.,
                 well_inside_cylinder_stroke=min_reserve >= .01,
                 tips_aligned=float(np.min(axes[:, :, 0])) >= np.cos(np.deg2rad(2.)),
                 no_strike=hit_speed <= .025,
                 floor_support=float(np.min(floor[steady, 2])) >= .90*weight,
                 only_expected_contacts=float(col("unexpected_contact_force_n").max()) < .01,
                 non_tip_surfaces_clear=float(np.min(col("input_non_tip_separation_m"))) > .0005,
                 level=float(np.max(col("tilt_rad"))) < np.deg2rad(2.),
                 straight=float(np.max(abs(col("yaw_rad")))) < np.deg2rad(2.),
                 transverse=float(np.max(abs(x[:, 1]))) < .003,
                 actuator_rating=float(np.max(drive)) <= RATED_N + .001)
    if active == "both":
        gates["sustained_dual_transport"] = simultaneous >= 1. and x[-1, 0] >= .10
    else:
        gates["inactive_tip_unloaded"] = float(np.max(abs(loads[:, ~enabled]))) < 1.
        gates["single_stays_put"] = float(np.max(np.linalg.norm(x[:, :2], axis=1))) < .002
        gates["single_within_static_force_margin"] = float(np.min(single_support_margin[steady])) > 0.
    gates = {k: bool(v) for k, v in gates.items()}
    return dict(passed=all(gates.values()), gates=gates,
                max_planar_displacement_m=float(np.max(np.linalg.norm(x[:, :2], axis=1))),
                final_delta_com_m=x[-1].tolist(), axial_load_peak_n=loads.max(axis=0).tolist(),
                axial_load_mean_steady_n=loads[steady].mean(axis=0).tolist(),
                longest_active_loaded_s=contiguous, longest_simultaneous_loaded_s=simultaneous,
                min_stroke_reserve_m=min_reserve, max_onset_approach_m_s=hit_speed,
                min_non_tip_separation_m=float(np.min(col("input_non_tip_separation_m"))),
                min_floor_support_n=float(np.min(floor[steady, 2])),
                min_single_static_margin_n=(float(np.min(single_support_margin[steady])) if active != "both" else None))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["cpu", "musa"], default="cpu")
    parser.add_argument("--seed", type=int, default=701)
    parser.add_argument("--out", type=Path, required=True, help="fresh result directory")
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    xml, source = make_fixture(args.seed)
    (args.out/"rated.xml").write_bytes(xml)
    compilation = compile_mujoco_xml(xml.decode(), require_asset=False)
    model = compilation.mj_model
    fixture = Fixture(model, source)
    profiles = ("left", "right", "both")
    keys = tuple(range(len(profiles)))
    init = {i: MuJoCoInitialState(fixture.mount.copy()) for i in keys}
    options = dict(cache_root=args.cache_dir, contact_capacity=32, constraint_capacity=128) if args.mode == "musa" else {}
    rows = {i: [] for i in keys}
    with MuJoCoRuntime(compilation, mode=args.mode, world_keys=keys, initial_states=init, **options) as runtime:
        snapshot = runtime.integration_snapshot if args.mode == "musa" else runtime.snapshot
        states = snapshot()
        initial_q = np.array(states[0].q)
        initial_v = np.array(states[0].v)
        for st in states.values():
            if not np.array_equal(st.q, initial_q) or not np.array_equal(st.v, initial_v):
                raise RuntimeError("groups did not start from the same native state")
        for _ in range(round(RUN_S/TIMESTEP)):
            pending = {i: fixture.control(states[i], [profiles[i] in (s, "both") for s in SIDES]) for i in keys}
            runtime.set_controls({i: MuJoCoControl(pending[i][0]) for i in keys})
            runtime.step(TIMESTEP)
            states = snapshot()
            for i in keys:
                rows[i].append(fixture.measure(states[i], *pending[i]))
        device = runtime.musa_device_report() if args.mode == "musa" else None
        backend = runtime.capabilities["backend"]
    results = {}
    for i, name in enumerate(profiles):
        with gzip.open(args.out/f"{name}.jsonl.gz", "wt", encoding="utf-8") as stream:
            for row in rows[i]:
                stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
        results[name] = summarize(rows[i], name, fixture.weight)
    report = dict(passed=all(r["passed"] for r in results.values()), experiment="rated sustained straight push",
                  mode=args.mode, backend=backend, device=device, seed=args.seed, timestep_s=TIMESTEP,
                  end_time_s=RUN_S, arm_dofs=source.arm_dofs, rated_per_wrist_n=RATED_N,
                  weight_n=fixture.weight, friction_coefficient=MU, stroke_limits_m=STROKE_M.tolist(),
                  source_push_t_xml_sha256=source.xml_sha256, fixture_xml_sha256=hashlib.sha256(xml).hexdigest(),
                  initial_qpos=initial_q.tolist(), initial_qvel=initial_v.tolist(), groups=results)
    text = json.dumps(report, indent=2, allow_nan=False)
    (args.out/"summary.json").write_text(text, encoding="utf-8")
    print(text, flush=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
