"""Reconstruct right finger/workpiece contact geometry at saved paired reach endpoints."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import mujoco as mj
import numpy as np

from _panda_grasp_insert_scene import PAD_NAMES
from _panda_ppo_reach import DISTANCE_M, DWELL_S, PART_DRIFT_M, SITE_NAMES
from panda_reach_trace_diagnosis import LIMITS, angle, gate_counts, kinematics


def checked_inputs(trace, scene, summary):
    receipt = json.loads(summary.read_text(encoding="utf-8"))
    contract = receipt["contract"]
    records = [receipt.get("evaluation", {})] + [row["paired"] for row in receipt.get("updates", [])]
    trace_digest = hashlib.sha256(trace.read_bytes()).hexdigest()
    scene_digest = hashlib.sha256(scene.read_bytes()).hexdigest()
    if not any(row.get("sha256") == trace_digest and row.get("file") == trace.name for row in records):
        raise ValueError("paired trace name or SHA256 differs from run summary")
    if contract["model_sha256"] != scene_digest or contract["task"] != "reach":
        raise ValueError("reach task or original scene SHA256 differs from run summary")
    return contract, dict(trace_sha256=trace_digest, scene_sha256=scene_digest, summary=summary.name)


def load_states(trace, branch, contract):
    with np.load(trace, allow_pickle=False) as data:
        def read(field):
            return data[f"{branch}_{field}"]

        qs = read("substep_qpos")
        if qs.ndim != 4 or qs.shape[-1] != 23 or min(qs.shape[:3]) < 1:
            raise ValueError("expected [decision, repeat, world, 23] substep qpos")
        steps, repeat, worlds = qs.shape[:3]
        if (repeat, worlds) != (contract["action_repeat"], contract["worlds"]):
            raise ValueError("trace shape differs from recorded run dimensions")
        q, v = read("qpos"), read("qvel")
        if q.shape != (steps + 1, worlds, 23) or v.shape != (steps + 1, worlds, 22):
            raise ValueError("unexpected decision endpoint q/v shapes")
        np.testing.assert_array_equal(q[1:], qs[:, -1])
        q = np.concatenate([q[:1], qs.reshape(-1, worlds, 23)])
        stride = repeat
        if f"{branch}_substep_qvel" in data:
            vs = read("substep_qvel")
            if vs.shape != (steps, repeat, worlds, 22) or vs.dtype != v.dtype:
                raise ValueError("unexpected substep qvel shape or dtype")
            np.testing.assert_array_equal(v[1:], vs[:, -1])
            v = np.concatenate([v[:1], vs.reshape(-1, worlds, 22)])
            stride = 1
        if q.dtype != np.float32 or v.dtype != np.float32 or not (np.isfinite(q).all() and np.isfinite(v).all()):
            raise ValueError("original finite FP32 q/v endpoints required")
        mask = read("substep_qualified")
        dwell = read("reach_max_qualified_dwell_steps")
        if mask.shape != (steps, repeat, worlds) or mask.dtype != bool or dwell.shape != (worlds,):
            raise ValueError("unexpected saved qualification/dwell arrays")
        return q, v, stride, mask.reshape(-1, worlds), dwell


def contact_layout(model):
    pads = [model.geom(n).id for n in PAD_NAMES]
    finger_bodies = model.geom_bodyid[pads]
    finger = set(np.flatnonzero(np.isin(model.geom_bodyid, finger_bodies)))
    part = set(np.flatnonzero(model.geom_bodyid == model.body("assembly_part").id))
    if {model.geom(i).name for i in part} != {"assembly_pin", "part_handle", "part_shoulder"}:
        raise ValueError("unexpected free workpiece collision geometry")
    slides = [model.joint(f"right_panda_finger_joint{i}").id for i in (1, 2)]
    if not np.array_equal(model.jnt_bodyid[slides], finger_bodies):
        raise ValueError("finger slides and pad bodies differ")
    return finger, part, model.jnt_qposadr[slides]


def set_geometry(model, data, q):
    data.qpos[:] = q
    mj.mj_kinematics(model, data)


def first_contacts(model, data, q, fingers, parts):
    first, points, count = None, [], 0
    for step, pos in enumerate(q):
        set_geometry(model, data, pos)
        mj.mj_collision(model, data)
        touched = False
        for c in data.contact[:data.ncon]:
            a, b = map(int, c.geom)
            if a in fingers and b in parts:
                finger, part, sign = a, b, 1
            elif b in fingers and a in parts:
                finger, part, sign = b, a, -1
            else:
                continue
            if c.dist > 0:
                continue
            touched = True
            if first is None:
                first = step
            if step == first:
                points.append((finger, part, sign * c.frame[:3].copy(), c.pos.copy(), float(c.dist)))
        count += int(touched)
    if any(int(w.number) for w in data.warning):
        raise ValueError("MuJoCo raised a warning during endpoint collision queries")
    return first, points, count


def point_speed(model, data, bodies, local_points, normal, v):
    relative_jac = np.zeros((3, model.nv))
    jac = np.empty_like(relative_jac)
    for sign, body, local in zip((-1, 1), bodies, local_points, strict=True):
        point = data.xpos[body] + data.xmat[body].reshape(3, 3) @ local
        mj.mj_jac(model, data, jac, None, point, body)
        relative_jac += sign * jac
    return float(normal @ (relative_jac @ v))


def witness(model, data, q, v, stride, step, point):
    finger, part, normal, pos, dist = point
    bodies = model.geom_bodyid[[finger, part]]
    set_geometry(model, data, q[step])
    mj.mj_comPos(model, data)
    local = [data.xmat[b].reshape(3, 3).T @ (pos - data.xpos[b]) for b in bodies]
    recorded = (point_speed(model, data, bodies, local, normal, v[step // stride])
                if step % stride == 0 else None)
    incoming_estimate = None
    prev_dist = prev_speed = prev_estimate = None
    if step:
        dq = np.empty(model.nv)
        mj.mj_differentiatePos(model, dq, model.opt.timestep, q[step-1].astype(float), q[step].astype(float))
        incoming_estimate = point_speed(model, data, bodies, local, normal, dq)
        set_geometry(model, data, q[step-1])
        mj.mj_comPos(model, data)
        prev_dist = float(mj.mj_geomDistance(model, data, finger, part, 10., None))
        if (step-1) % stride == 0:
            prev_speed = point_speed(model, data, bodies, local, normal, v[(step-1) // stride])
        if step > 1:
            mj.mj_differentiatePos(model, dq, model.opt.timestep, q[step-2].astype(float), q[step-1].astype(float))
            prev_estimate = point_speed(model, data, bodies, local, normal, dq)
    brackets = []
    for k in sorted({step // stride, (step + stride - 1) // stride}):
        set_geometry(model, data, q[k * stride])
        mj.mj_comPos(model, data)
        brackets.append(dict(step=k * stride, time_s=k * stride * model.opt.timestep,
                             speed_m_s=point_speed(model, data, bodies, local, normal, v[k])))
    return dict(finger_geom=model.geom(finger).name, part_geom=model.geom(part).name,
                signed_distance_m=dist, preceding_endpoint_pair_distance_m=prev_dist,
                contact_point_world_m=pos.tolist(), normal_finger_to_part_world=normal.tolist(),
                recorded_normal_speed_m_s=recorded, incoming_qpos_difference_estimate_m_s=incoming_estimate,
                previous_recorded_normal_speed_m_s=prev_speed, previous_incoming_qpos_difference_estimate_m_s=prev_estimate,
                recorded_qvel_brackets=brackets)


def analyze_world(model, layout, q, v, stride, saved, recorded_dwell):
    dt = float(model.opt.timestep)
    fingers, parts, slides = layout
    # Original, independently reconstructed FK gates and duration at full physics resolution.
    _, _, _, m = kinematics(model, q)
    good = np.logical_and.reduce([m[key] <= lim for key, lim in LIMITS.items()])
    np.testing.assert_array_equal(good[1:], saved)
    dwell = gate_counts(good, dt)
    np.testing.assert_equal(dwell["longest_intervals"], recorded_dwell)
    needed = math.ceil(DWELL_S / dt)
    passing = np.flatnonzero(good)
    first_success = next((int(k) for k in passing if k >= needed and good[k-needed:k+1].all()), None)
    data = mj.MjData(model)
    first, points, count = first_contacts(model, data, q, fingers, parts)
    hand, pin = (model.site(n).id for n in SITE_NAMES[1:])

    def sample(step):
        if step is None:
            return None
        step = int(step)
        set_geometry(model, data, q[step])
        hand_r, pin_r = data.site_xmat[[hand, pin]].reshape(2, 3, 3)
        to_pin = pin_r.T @ (data.site_xpos[pin] - data.site_xpos[hand])
        return dict(step=step, time_s=step * dt, finger_joint_openings_m=q[step, slides].tolist(),
                    pin_minus_hand_in_pin_frame_m=to_pin.tolist(), axial_m=float(to_pin[2]),
                    lateral_m=float(np.linalg.norm(to_pin[:2])), orientation_rad=float(angle(hand_r.T @ pin_r)),
                    distance_m=float(m["distance_m"][step]), pin_drift_m=float(m["part_drift_m"][step]),
                    qualified=bool(good[step]))

    def first_true(mask):
        hits = np.flatnonzero(mask)
        return int(hits[0]) if len(hits) else None

    exceed = first_true(m["part_drift_m"] > PART_DRIFT_M)
    return dict(first_contact=sample(first), preceding_contact=sample(first - 1 if first else None),
                first_contact_witnesses=[witness(model, data, q, v, stride, first, p) for p in points],
                touching_endpoints=count, first_pin_drift_exceeded=sample(exceed),
                preceding_pin_drift_exceeded=sample(exceed - 1 if exceed else None),
                peak_pin_drift=sample(m["part_drift_m"].argmax()),
                first_distance_pass=sample(first_true(m["distance_m"] <= DISTANCE_M)),
                closest=sample(m["distance_m"].argmin()), final=sample(len(q)-1),
                first_dwell_success=sample(first_success), longest_qualified_dwell_s=dwell["longest_dwell_s"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--summary", type=Path, help="run receipt (defaults to summary.json beside trace)")
    parser.add_argument("--branch", default="after", choices=("before", "after", "motors_off"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    contract, provenance = checked_inputs(args.trace, args.scene, args.summary or args.trace.with_name("summary.json"))
    model = mj.MjModel.from_xml_path(str(args.scene))
    if (model.nq, model.nv, model.nu) != (23, 22, 16) or model.opt.timestep != contract["timestep"]:
        raise ValueError("expected original compiled reach model dimensions and timestep")
    layout = contact_layout(model)
    q, v, stride, mask, dwell = load_states(args.trace, args.branch, contract)
    rows = [dict(world=w, **analyze_world(model, layout, q[:, w], v[:, w], stride, mask[:, w], dwell[w]))
            for w in range(q.shape[1])]
    result = dict(schema="panda-reach-finger-contact-events-v1", branch=args.branch, **provenance,
                  timestep_s=model.opt.timestep, qvel_stride_physics_steps=stride, physics_intervals=len(q)-1,
                  definitions=dict(contact="First sampled native finger/workpiece contact with signed distance <= 0 m. "
                                           "Outputs describe endpoint geometry and kinematics.",
                                   normal_speed="(workpiece velocity - finger velocity) at the contact point projected "
                                                "onto the finger-to-workpiece unit normal. Negative means closing.",
                                   recorded_speed="Null when a co-timed recorded qvel is absent. Brackets use the "
                                                  "same material points and event normal at actual saved q/v pairs.",
                                   incoming_estimate="Previous physical interval generalized velocity from mj_differentiatePos, "
                                                     "projected with the endpoint contact-point Jacobians.",
                                   axial="(pin minus hand) along current positive pin Z; positive with hand above upright pin.",
                                   drift_threshold_m=PART_DRIFT_M, dwell_threshold_s=DWELL_S),
                  worlds_touching=sum(row["first_contact"] is not None for row in rows),
                  worlds_exceeding_pin_drift=sum(row["first_pin_drift_exceeded"] is not None for row in rows),
                  worlds_achieving_dwell=sum(row["first_dwell_success"] is not None for row in rows),
                  worlds=rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
