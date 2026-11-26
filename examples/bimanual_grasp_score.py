"""Score recorded flat-box grasp trials with MuJoCo forward kinematics only.

    python -B examples/bimanual_grasp_score.py runs/holdout --require-complete --output score.json

Each named input is a directory of original episode receipts or a single receipt.
Supply one experiment partition at a time. Paired comparisons use the same seed,
arm layout, enabled arms and XML. Only actual, colour-input episodes are scored.
"""

import argparse
import hashlib
import json
from collections import defaultdict
from io import BytesIO
from pathlib import Path

import mujoco as mj
import numpy as np


SIDES = ("left", "right")
MODES = ("face-alignment", "firm-grip", "stable-grasp")


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _relative_poses(model, qpos, diagnostics):
    """Return output-sample TCP-relative poses, and world up in each TCP frame."""
    data = mj.MjData(model)
    body = model.body("payload").id
    sites = [model.site(f"{s}_tool").id for s in SIDES]
    positions, rotations, up = [], [], []
    for q, step in zip(qpos, diagnostics, strict=True):
        data.qpos[:] = q
        mj.mj_kinematics(model, data)
        box = data.xpos[body]
        if not np.allclose(box, step["measured"]["payload_xyz_m"], atol=1e-6, rtol=0.):
            raise ValueError("saved output pose differs from decision endpoint")
        world_to_tool = data.site_xmat[sites].reshape(2, 3, 3).transpose(0, 2, 1)
        positions.append(np.einsum("sij,sj->si", world_to_tool, box - data.site_xpos[sites]))
        rotations.append(world_to_tool @ data.xmat[body].reshape(3, 3))
        up.append(world_to_tool[:, :, 2])
    return np.array(positions), np.array(rotations), np.array(up)


def _landing(phases, recovery, floor, weight, start, first, limit):
    """Classify a confirmed table load between this air bout and the next."""
    contact = np.flatnonzero(floor[first:limit] >= .05 * weight)
    if not len(contact):
        return {"kind": "no_confirmed_table_return"}
    i = first + int(contact[0])
    phase = phases[i]
    if any(recovery[start:i + 1]) and phase in ("lower", "unload", "clamp"):
        kind = "commanded_recovery_lowering"
    elif phase in ("lower", "unload", "release"):
        # lower is used for ordinary placement AND recovery. The recorded
        # post-touch decision sequence identifies a commanded re-clamp.
        after = phases[i:limit]
        next_command = next((p for p in after if p in ("clamp", "release")), None)
        if next_command == "clamp":
            kind = "commanded_lower_and_regrasp"
        elif next_command == "release":
            kind = "commanded_lower_and_release"
        else:
            kind = "commanded_lower"
    elif phase in ("lift", "transport"):
        kind = "unplanned_table_return"
    else:
        kind = "table_return_intent_unresolved"
    return {"kind": kind, "phase_at_table_contact": phase, "contact_output_index": i + 1,
            "measured_table_vertical_n": float(floor[i])}


def _sampled_bouts(mask, ticks, times, poses, enabled):
    """Use the first recorded output of each uninterrupted segment as its own reference."""
    positions, rotations, up = poses
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False])).reshape(-1, 2)
    for index, (start, end) in enumerate(edges):
        next_start = int(edges[index + 1, 0]) if index + 1 < len(edges) else len(mask)
        hands = {}
        for s, side in enumerate(SIDES):
            if side not in enabled or end - start < 2:
                continue
            r0 = rotations[start, s]
            delta = r0.T @ rotations[start:end, s]
            cosine = np.clip((np.trace(delta, axis1=1, axis2=2) - 1.) * .5, -1., 1.)
            angle = np.rad2deg(np.arccos(cosine))
            # A co-moving rigid grasp gives a constant TCP-relative vector.
            # Project the change onto gravity expressed in the first TCP frame.
            downward = (positions[start, s] - positions[start:end, s]) @ up[start, s]
            hands[side] = {"max_relative_rotation_deg": float(angle.max()),
                           "max_downward_slip_mm": 1000. * max(0., float(downward.max()))}
        bout = {"first_output_tick": int(ticks[start]), "last_output_tick": int(ticks[end - 1]),
                "first_time_s": float(times[start]), "last_time_s": float(times[end - 1]),
                "samples": int(end - start), "hands": hands}
        yield start, end, next_start, bout


def _bouts(air, phases, recovery, reasons, unsupported_descent, floor, weight, ticks, times, poses, enabled):
    bouts = []
    for start, end, next_start, bout in _sampled_bouts(air, ticks, times, poses, enabled):
        landing = _landing(phases, recovery, floor, weight, int(start), int(end), next_start)
        if "contact_output_index" in landing:
            landing["contact_tick"] = int(ticks[landing["contact_output_index"] - 1])
        bout.update(landing=landing, unsupported_descending_under_lift_or_transport_samples=
                    int(np.count_nonzero(unsupported_descent[start:end])),
                    controller_recovery_reasons=sorted({r for r in reasons[start:next_start] if r}))
        bouts.append(bout)
    return bouts


def _transport(air, phases, ticks, times, poses, enabled):
    moving = air & np.equal(phases, "transport")
    bouts = []
    for _, _, _, bout in _sampled_bouts(moving, ticks, times, poses, enabled):
        bout["duration_s"] = bout["last_time_s"] - bout["first_time_s"]
        bouts.append(bout)
    if not bouts:
        return None
    hands = [hand for bout in bouts for hand in bout["hands"].values()]
    return dict(bouts=bouts, bout_count=len(bouts), samples=sum(b["samples"] for b in bouts),
                duration_s=sum(b["duration_s"] for b in bouts),
                max_relative_rotation_deg=max((h["max_relative_rotation_deg"] for h in hands), default=None),
                max_downward_slip_mm=max((h["max_downward_slip_mm"] for h in hands), default=None))


def _score(path, row, raw):
    xml = path.with_suffix(".xml").read_bytes()
    archive = path.with_suffix(".npz").read_bytes()
    if _digest(xml) != row["scene_sha256"] or _digest(archive) != row["trajectory_sha256"]:
        raise ValueError("XML or NPZ SHA-256 mismatch")
    model = mj.MjModel.from_xml_string(xml.decode("utf-8"))
    final = row["final_metrics"]
    steps = row["decision_diagnostics"]
    n = row["decisions"]
    arms = row["job"]["arms"]
    if (not n or len(steps) != n or final["xml_sha256"] != row["scene_sha256"]
            or final["seed"] != row["seed"] or final["arm_dofs"] != row["arm_dofs"]
            or final["active_arms"] != arms or arms not in ("both", *SIDES)
            or final["success"] != row["success"] or not (row["terminated"] or row["truncated"])):
        raise ValueError("inconsistent or unfinished episode")
    with np.load(BytesIO(archive), allow_pickle=False) as saved:
        ticks, times, q = saved["ticks"], saved["time_s"], saved["qpos"]
        if (saved["model_sha256"].item() != row["scene_sha256"]
                or q.shape != (n + 1, model.nq) or ticks.shape != (n + 1,) or times.shape != ticks.shape
                or ticks[-1] != row["physics_ticks"] or not np.all(np.diff(ticks) > 0)
                or not np.all(np.diff(times) > 0) or not np.isfinite(times).all() or not np.isfinite(q).all()
                or not np.array_equal(saved["rewards"], row["rewards"])
                or not np.array_equal(ticks[:-1], [s["input_tick"] for s in steps])
                or not np.array_equal(ticks[1:], [s["output_tick"] for s in steps])):
            raise ValueError("trajectory model, samples, rewards or endpoint ticks disagree")
    poses = _relative_poses(model, q[1:], steps)
    enabled = SIDES if arms == "both" else (arms,)
    weight = float(model.body_mass[model.body("payload").id]) * float(-model.opt.gravity[2])
    if weight <= 0 or not np.allclose(model.opt.gravity[:2], 0.):
        raise ValueError("expected downward gravity and positive payload mass")
    floor = np.array([s["measured"]["floor_vertical_n"] for s in steps])
    clearance = np.array([s["measured"]["clearance_m"] for s in steps])
    if not (np.isfinite(floor).all() and np.isfinite(clearance).all()):
        raise ValueError("missing finite table-load or clearance observations")
    air = (floor < .05 * weight) & (clearance >= .003)
    phases = [s["visual"].get("phase") for s in steps]
    stable = [s["visual"].get("stable_grasp") or {} for s in steps]
    recovery = [s.get("stage") == "recovery" for s in stable]
    reasons = [s.get("recovery_reason") if active else None for s, active in zip(stable, recovery, strict=True)]
    # Native vertical forces are the upward load on the box from both hands.
    # These describe the final physics substep, while velocity here is the
    # saved box-origin displacement averaged over the decision interval.
    box_adr = int(model.jnt_qposadr[model.joint("payload_free").id])
    vz = np.diff(q[:, box_adr + 2]) / np.diff(times)
    hand_up = np.array([s["measured"]["pad_vertical_n"] for s in steps], dtype=float)
    if hand_up.shape != (n, 2) or not np.isfinite(hand_up).all():
        raise ValueError("expected finite left/right measured pad vertical loads")
    unsupported = (air & np.isin(phases, ("lift", "transport"))
                   & (np.maximum(0., hand_up).sum(axis=1) < .05 * weight) & (vz < -.01))
    bouts = _bouts(air, phases, recovery, reasons, unsupported, floor, weight, ticks[1:], times[1:], poses, enabled)
    hands = [metrics for bout in bouts for metrics in bout["hands"].values()]
    return dict(receipt=str(path), receipt_sha256=_digest(raw), trajectory_sha256=row["trajectory_sha256"],
                scene_sha256=row["scene_sha256"], seed=row["seed"], arm_dofs=row["arm_dofs"],
                arms=arms, control=row["control"], success=row["success"], physics=row["physics"],
                max_relative_rotation_deg=max((h["max_relative_rotation_deg"] for h in hands), default=None),
                max_downward_slip_mm=max((h["max_downward_slip_mm"] for h in hands), default=None),
                peak_pad_normal_n_by_hand=dict(zip(SIDES, final["peak_pad_normal_n"], strict=True)),
                unplanned_table_returns=sum(b["landing"]["kind"] == "unplanned_table_return" for b in bouts),
                unsupported_descending_under_lift_or_transport_samples=int(unsupported.sum()),
                recovery_entries=sum(active and (i == 0 or not recovery[i-1]) for i, active in enumerate(recovery)),
                completed_reclamps=sum(phases[i-1] == "unload" and p == "clamp" for i, p in enumerate(phases) if i),
                airborne_bouts=bouts, sampled_airborne_decisions=int(air.sum()),
                airborne_transport=_transport(air, phases, ticks[1:], times[1:], poses, enabled),
                max_clearance_mm=1000. * final["max_clearance_m"], carried_airborne=final["carried_airborne"],
                final_goal_xy_error_mm=1000. * final["goal_error_xy_m"],
                final_goal_yaw_error_deg=float(np.rad2deg(final["goal_yaw_error_rad"])),
                final_tilt_deg=float(np.rad2deg(final["tilt_rad"])), final_payload_xyz_m=final["payload_xyz_m"],
                initial_payload_xyz_quat_wxyz=q[0, box_adr:box_adr + 7].tolist(),
                final_payload_xyz_quat_wxyz=q[-1, box_adr:box_adr + 7].tolist(),
                final_goal_dwell_s=final["goal_dwell_s"], final_table_vertical_n=final["floor_vertical_n"])


def _stats(values):
    values = [v for v in values if v is not None]
    return dict(n=len(values), median=float(np.median(values)) if values else None, max=max(values, default=None))


def _summary(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["control"], tuple(row["arm_dofs"]), row["arms"], row["physics"])].append(row)
    out = []
    for (control, layout, arms, physics), group in sorted(groups.items()):
        summaries = {}
        for key in ("max_relative_rotation_deg", "max_downward_slip_mm", "final_goal_xy_error_mm"):
            summaries[key] = _stats(r[key] for r in group)
        transport = [r["airborne_transport"] for r in group if r["airborne_transport"] is not None]
        transport_stats = {k: _stats(t[k] for t in transport)
                           for k in ("bout_count", "samples", "duration_s", "max_relative_rotation_deg", "max_downward_slip_mm")}
        out.append(dict(control=control, arm_dofs=layout, arms=arms, physics=physics, trials=len(group),
                        seeds=sorted(r["seed"] for r in group), successes=sum(r["success"] for r in group),
                        unplanned_table_returns=sum(r["unplanned_table_returns"] for r in group),
                        recoveries=sum(r["recovery_entries"] for r in group),
                        completed_reclamps=sum(r["completed_reclamps"] for r in group),
                        airborne_transport=dict(trials_with_transport=len(transport), metrics=transport_stats),
                        peak_pad_normal_n_by_hand={s: max(r["peak_pad_normal_n_by_hand"][s] for r in group)
                                                   for s in SIDES}, metrics=summaries))
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, help="write JSON here (default: standard output)")
    parser.add_argument("--require-complete", action="store_true",
                        help="require all three bilateral controls at every supplied seed and arm layout")
    args = parser.parse_args(argv)
    files = set()
    for root in args.roots:
        if root.is_dir():
            files.update(p.resolve() for p in root.rglob("*.json"))
        elif root.is_file():
            files.add(root.resolve())
        else:
            parser.error(f"missing input: {root}")
    rows, cases = [], defaultdict(dict)
    for path in sorted(files):
        raw = path.read_bytes()
        row = json.loads(raw)
        if (not isinstance(row, dict) or row.get("tool_profile") != "four-contact" or row.get("task") != "carry"
                or row.get("control") not in MODES or "decision_diagnostics" not in row or row["job"]["black"]):
            continue
        try:
            result = _score(path, row, raw)
        except (KeyError, OSError, ValueError) as exc:
            raise ValueError(f"{path}: {exc}") from exc
        key = (result["seed"], tuple(result["arm_dofs"]), result["arms"], result["physics"])
        group = cases[key]
        if result["control"] in group:
            parser.error(f"duplicate trial for {key} / {result['control']}: {path}")
        if group and result["scene_sha256"] != next(iter(group.values()))["scene_sha256"]:
            parser.error(f"mismatched XML for paired controls at {key}: {path}")
        group[result["control"]] = result
        rows.append(result)
    if not rows:
        parser.error("no matching original four-contact episode receipts")
    if args.require_complete:
        for seed, layout, _, physics in list(cases):
            bilateral = cases.get((seed, layout, "both", physics), {})
            missing = set(MODES).difference(bilateral)
            if missing:
                parser.error(f"{seed} {layout} {physics}: missing bilateral controls {sorted(missing)}")
    paired = []
    for (seed, layout, arms, physics), group in sorted(cases.items()):
        keys = ("success", "max_relative_rotation_deg", "max_downward_slip_mm", "peak_pad_normal_n_by_hand",
                "unplanned_table_returns", "recovery_entries", "completed_reclamps",
                "unsupported_descending_under_lift_or_transport_samples", "max_clearance_mm", "final_goal_xy_error_mm")
        paired.append(dict(seed=seed, arm_dofs=layout, arms=arms, physics=physics, complete=set(group) == set(MODES),
                           controls={m: {k: group[m][k] for k in keys} for m in MODES if m in group}))
    report = dict(schema="bimanual-offline-grasp-score-v1",
                  measurement="Decision output poses, FK only. Air: clearance >= 3 mm and native last-substep "
                              "table force < 5% of weight. Each continuous sampled air bout has its own "
                              "reference; two samples required for drift. Peak force covers all physics steps. "
                              "Airborne transport uses the recorded transport command and the same air test. "
                              "Each continuous transport segment resets the tool-relative reference at its first output. "
                              "Duration sums last minus first output times per segment. Absent transport is null; "
                              "group n counts episodes with a measurement for that metric. "
                              "Table returns require observed support; a single decision can miss brief contacts. "
                              "Unsupported descent samples: lift/transport, airborne, positive upward hand "
                              "loads below 5% of weight, box dropping >10 mm/s over the decision. "
                              "Privileged box poses, contact labels and goal errors are scoring inputs only.",
                  episodes=rows, groups=_summary(rows), paired=paired)
    content = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        if args.output.resolve() in files and any(r["receipt"] == str(args.output.resolve()) for r in rows):
            parser.error("output would overwrite an input episode")
        args.output.write_text(content, encoding="utf-8")
    else:
        print(content, end="")


if __name__ == "__main__":
    main()
