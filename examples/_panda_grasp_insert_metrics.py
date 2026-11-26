"""Score saved physical states/contacts of the grasp-transfer-seat experiment."""

import numpy as np


PHASES = ("settle", "approach", "descend", "close", "lift", "transfer", "insert", "seat", "release", "clear")
SHAPES = dict(qpos=(23,), qvel=(22,), ctrl=(16,), pad_forces=(2,), wrist_xyz=(3,), part_xyz=(3,),
              pad_up_n=(), support_n=(), seat_n=(), other_up_n=(), depth_m=(), lateral_m=(),
              tilt_rad=(), hole_gap_m=())


def _trace(source, dt):
    t = np.asarray(source["time_s"], dtype=float)
    if (t.ndim != 1 or len(t) < 2 or not np.isfinite(t).all() or np.any(np.diff(t) <= 0.)
            or not np.allclose(np.diff(t), dt, rtol=.005, atol=1e-8)):
        raise ValueError("metrics require every ordered physics sample at dt")
    trace = {"time_s": t, "phase": np.asarray(source["phase"]).astype(str)}
    if trace["phase"].shape != t.shape:
        raise ValueError("one phase label is required per state")
    for name, shape in SHAPES.items():
        x = np.asarray(source[name], dtype=float)
        if x.shape != (len(t), *shape) or not np.isfinite(x).all():
            raise ValueError(f"missing, incomplete or nonfinite channel: {name}")
        trace[name] = x
    return trace


def _runs(mask):
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False].astype(int)))
    return zip(edges[::2], edges[1::2], strict=True)


def _tail(time, mask, seconds):
    """Indices of a complete trailing window inside the last continuous run."""
    hits = list(_runs(mask))
    if not hits:
        return np.array([], dtype=int)
    i, stop = hits[-1]
    end = time[stop-1]
    if end-time[i] < seconds-1e-7:
        return np.array([], dtype=int)
    first = max(i, np.searchsorted(time, end-seconds+1e-7, side="right")-1)
    return np.arange(first, stop)


def _range(x):
    return [float(x.min()), float(x.max())] if x.size else None


def _episode(d, weight):
    t, phase, p, w = d["time_s"], d["phase"], d["part_xyz"], d["wrist_xyz"]
    jaw = d["qpos"][:, 14:16]
    # The free joint follows both seven-axis arms and both right-hand slides.
    speed = np.linalg.norm(d["qvel"][:, 16:19], axis=1)
    resolved, unloaded = .02*weight, .05*weight
    settled = _tail(t, phase == "settle", .10)
    ref = np.median(p[settled], axis=0) if len(settled) else p[0]
    height = p[:, 2]-ref[2]
    xy = np.linalg.norm(p[:, :2]-ref[:2], axis=1)
    pad_peak = d["pad_forces"].max(axis=1)
    settle_ok = bool(len(settled) and np.all(d["support_n"][settled] >= .8*weight)
                     and np.all(speed[settled] <= .008) and np.all(pad_peak[settled] <= unloaded))

    # A single contiguous interval must contain the measured lift dwell and
    # subsequent transfer. Regrasping, returning to the fork, or losing a pad
    # breaks the interval; disjoint pieces cannot be combined into a pass.
    bearing = ((d["pad_forces"].min(axis=1) >= resolved) & (d["pad_up_n"] > 0.)
               & (np.abs(d["support_n"]) <= unloaded) & (np.abs(d["seat_n"]) <= unloaded)
               & (np.abs(d["other_up_n"]) <= unloaded) & (height >= .025))
    carry_windows = []
    for i, stop in _runs(bearing & np.isin(phase, ("lift", "transfer"))):
        duration = float(t[stop-1]-t[i])
        if phase[i] != "lift":
            continue
        eligible = np.arange(i, stop)
        eligible = eligible[(phase[eligible] == "transfer") & (t[eligible]-t[i] >= .150-1e-7)]
        distance = (float(np.linalg.norm(p[eligible, :2]-p[i, :2], axis=1).max())
                    if len(eligible) else 0.)
        carry_windows.append(dict(start_s=float(t[i]), end_s=float(t[stop-1]), duration_s=duration,
                                 transfer_from_lift_xy_m=distance,
                                 minimum_normal_by_pad_n=d["pad_forces"][i:stop].min(axis=0).tolist(),
                                 mean_pad_up_n=float(np.mean(d["pad_up_n"][i:stop]))))
    longest = max((x["duration_s"] for x in carry_windows), default=0.)
    distance = max((x["transfer_from_lift_xy_m"] for x in carry_windows), default=0.)

    geometry = ((d["depth_m"] >= .008) & (d["depth_m"] <= .012) & (d["lateral_m"] <= .0009)
                & (d["tilt_rad"] <= np.deg2rad(2.)))
    seat_indices = np.flatnonzero((phase == "seat") & geometry & (d["seat_n"] >= resolved))
    # CAD half-width of the handle is 12 mm. The inside face of each pad is
    # 0.02 mm inward of its rail displacement. Require positive jaw clearance.
    open_gap = jaw.min(axis=1) > .01202
    separated = open_gap & (pad_peak <= resolved) & (np.abs(d["pad_up_n"]) <= unloaded)
    release_indices = np.flatnonzero((phase == "release") & separated)
    terminal = _tail(t, phase == "clear", .10)
    final_ok = bool(len(terminal) and terminal[-1] == len(t)-1)
    seat_ok = bool(len(seat_indices))
    release_ok = bool(seat_ok and len(release_indices) and release_indices[-1] > seat_indices[-1])
    separation = np.linalg.norm(w-p, axis=1)
    retreat = None
    if final_ok and seat_ok:
        retreat = float(np.min(separation[terminal])-separation[seat_indices[-1]])
    final_support = (final_ok and np.all(geometry[terminal]) and np.all(speed[terminal] <= .008)
                     and np.all(d["seat_n"][terminal] >= .8*weight) and np.all(separated[terminal])
                     and np.all(np.abs(d["support_n"][terminal]) <= unloaded)
                     and np.all(np.abs(d["other_up_n"][terminal]) <= unloaded))
    minimum_gap = float(d["hole_gap_m"].min())
    spans = phase[np.r_[True, phase[1:] != phase[:-1]]].tolist()
    # The reset sample precedes the first commanded physics step.
    if phase[0] == "initial" and t[0] == 0. and np.count_nonzero(phase == "initial") == 1:
        spans = spans[1:]
    gates = dict(phase_order=spans == list(PHASES), settle_supported=settle_ok,
                 bilateral_lift=longest >= .150-1e-7, same_grasp_transfer=distance >= .080,
                 shoulder_seated=seat_ok, released=release_ok,
                 withdrawn=bool(release_ok and retreat is not None and retreat >= .040),
                 final_board_support=bool(final_support), penetration=minimum_gap >= -.0007)
    phases = {}
    for name in PHASES:
        idx = np.flatnonzero(phase == name)
        if len(idx):
            phases[name] = dict(time_s=_range(t[idx]), height_m=_range(height[idx]), xy_m=_range(xy[idx]),
                                pad_up_n=_range(d["pad_up_n"][idx]), support_n=_range(d["support_n"][idx]),
                                seat_n=_range(d["seat_n"][idx]), other_up_n=_range(d["other_up_n"][idx]),
                                minimum_normal_by_pad_n=d["pad_forces"][idx].min(axis=0).tolist(),
                                maximum_normal_by_pad_n=d["pad_forces"][idx].max(axis=0).tolist(),
                                depth_m=_range(d["depth_m"][idx]), lateral_m=_range(d["lateral_m"][idx]),
                                tilt_deg=_range(np.rad2deg(d["tilt_rad"][idx])), speed_m_s=_range(speed[idx]))
    return dict(gates=gates, phases=phases, carry_windows=carry_windows,
                longest_bilateral_lift_s=longest, same_grasp_transfer_m=distance,
                first_seat_s=float(t[seat_indices[0]]) if seat_ok else None,
                first_release_separation_s=float(t[release_indices[0]]) if len(release_indices) else None,
                final_separation_increase_m=retreat, minimum_hole_gap_m=minimum_gap,
                settled_xyz_m=ref.tolist(), peak_height_m=float(height.max()), peak_xy_m=float(xy.max()),
                final_window=dict(time_s=_range(t[terminal]), speed_m_s=_range(speed[terminal]),
                                  seat_n=_range(d["seat_n"][terminal]), pad_peak_n=_range(pad_peak[terminal]),
                                  jaw_inner_half_gap_m=_range(jaw[terminal]-.00002)))


def _comparison(closed, opened, dt, weight_n):
    if not np.isfinite(dt) or dt <= 0 or not np.isfinite(weight_n) or weight_n <= 0:
        raise ValueError("dt and weight_n must be finite and positive")
    c, o = _trace(closed, dt), _trace(opened, dt)
    pos, neg = _episode(c, weight_n), _episode(o, weight_n)
    same_initial = (np.array_equal(c["qpos"][0], o["qpos"][0])
                    and np.array_equal(c["qvel"][0], o["qvel"][0]))
    aligned = np.array_equal(c["time_s"], o["time_s"]) and np.array_equal(c["phase"], o["phase"])
    identical = bool(aligned and np.array_equal(c["ctrl"][:, :14], o["ctrl"][:, :14]))
    max_arm_delta = first_mismatch = max_departure = None
    path_by_phase = {}
    if aligned:
        delta = np.max(np.abs(c["ctrl"][:, :14]-o["ctrl"][:, :14]), axis=1)
        mismatches = np.flatnonzero(delta)
        max_arm_delta = float(delta.max())
        first_mismatch = float(c["time_s"][mismatches[0]]) if len(mismatches) else None
        path = np.linalg.norm(c["wrist_xyz"]-o["wrist_xyz"], axis=1)
        max_departure = float(path.max())
        for name in PHASES:
            idx = c["phase"] == name
            if idx.any():
                path_by_phase[name] = float(path[idx].max())
    jaws_open = bool(np.all(o["qpos"][:, 14:16]-.00002 > .012))
    no_open_carry = neg["peak_height_m"] < .025 and neg["peak_xy_m"] < .080
    gates = dict(pos["gates"], same_initial_state=bool(same_initial), aligned_controls=bool(aligned),
                 same_arm_torques=identical, open_jaws=jaws_open, no_open_carry=bool(no_open_carry),
                 open_start_supported=neg["gates"]["settle_supported"])
    report = dict(success=all(gates.values()), gates=gates, closed=pos, opened=neg,
                negative_control=dict(max_arm_torque_difference_nm=max_arm_delta,
                                      first_different_arm_torque_s=first_mismatch,
                                      max_wrist_path_difference_m=max_departure,
                                      wrist_path_difference_m_by_phase=path_by_phase,
                                      same_torques_and_over_40mm_wrist_divergence=bool(
                                          identical and max_departure is not None and max_departure > .040)),
                thresholds=dict(weight_n=float(weight_n), dt_s=float(dt), min_lift_m=.025,
                                min_lift_duration_s=.15, min_transfer_m=.08, min_withdrawal_m=.04,
                                support_dwell_s=.10, min_support_weight_fraction=.8, max_speed_m_s=.008,
                                depth_range_m=[.008, .012], max_lateral_m=.0009, max_tilt_deg=2.,
                                max_hole_penetration_m=.0007, contact_detection_weight_fraction=.02,
                                unloaded_weight_fraction=.05))
    return report, c, o


def evaluate(closed, opened, dt, weight_n):
    """Strict effort replay. Both runs must have identical submitted arm torques."""
    report, _, _ = _comparison(closed, opened, dt, weight_n)
    report["mode"] = "same-arm-torques/open-jaw"
    return report


def _pd_check(source, trace):
    n = len(trace["time_s"])
    target = np.asarray(source["arm_target_q"])
    if target.shape != (n, 14) or target.dtype.kind != "f" or not np.isfinite(target).all():
        raise ValueError("arm_target_q must contain a finite 14-joint target on every row")
    parameters = {}
    for name, shape in (("arm_kp", (14,)), ("arm_kd", (14,)), ("arm_ctrlrange", (14, 2))):
        values = np.asarray(source[name], dtype=float)
        if values.shape != shape or not np.isfinite(values).all():
            raise ValueError(f"invalid {name}")
        parameters[name] = values
    kp, kd, bounds = (parameters[key] for key in ("arm_kp", "arm_kd", "arm_ctrlrange"))
    if np.any(kp <= 0.) or np.any(kd < 0.) or np.any(bounds[:, 0] >= bounds[:, 1]):
        raise ValueError("expected positive stiffness, nonnegative damping and ordered effort limits")
    bias = np.asarray(source["arm_bias_nms"], dtype=float)
    if bias.shape != (n, 14) or not np.isfinite(bias).all():
        raise ValueError("arm_bias_nms must contain the input-state bias for each submitted arm command")
    # Row zero is the reset state. Each following row stores the command that
    # advanced the preceding q/v to the row's output state.
    predicted = np.clip(kp*(target[1:]-trace["qpos"][:-1, :14])
                        - kd*trace["qvel"][:-1, :14]+bias[1:], bounds[:, 0], bounds[:, 1])
    error = np.abs(predicted-trace["ctrl"][1:, :14])
    # Allow numerical storage of submitted MUSA controls in float32. This
    # tolerance applies only to the recomputed torque, never to the targets.
    tolerance = 2*np.finfo(np.float32).eps*np.maximum(1., np.abs(predicted))
    return target, parameters, bool(np.all(error <= tolerance)), float(np.max(error))


def evaluate_position_control(closed, opened, dt, weight_n):
    """Paired position targets with measured-state PD feedback in each world."""
    report, c, o = _comparison(closed, opened, dt, weight_n)
    ct, cp, closed_pd, closed_error = _pd_check(closed, c)
    ot, op, opened_pd, opened_error = _pd_check(opened, o)
    targets_equal = (ct.shape == ot.shape and ct.dtype == ot.dtype and ct.tobytes() == ot.tobytes())
    same_pd = all(np.array_equal(cp[key], op[key]) for key in cp)
    terminal = _tail(o["time_s"], o["phase"] == "clear", .10)
    complete = bool(len(terminal) and terminal[-1] == len(o["time_s"])-1)
    drift = np.linalg.norm(o["part_xyz"][terminal]-report["opened"]["settled_xyz_m"], axis=1)
    speed = np.linalg.norm(o["qvel"][terminal, 16:19], axis=1)
    remaining_on_fork = bool(complete and np.all(o["support_n"][terminal] >= .8*weight_n)
                             and np.all(np.abs(o["seat_n"][terminal]) <= .05*weight_n)
                             and np.all(np.abs(o["other_up_n"][terminal]) <= .05*weight_n)
                             and np.all(o["pad_forces"][terminal] <= .02*weight_n)
                             and np.all(np.abs(o["pad_up_n"][terminal]) <= .05*weight_n)
                             and np.all(speed <= .008))
    diag = report["negative_control"]
    departure = diag["max_wrist_path_difference_m"]
    gates = report["gates"]
    # Torque differences remain a diagnostic of closed-loop load response.
    # The separate effort-replay score continues to require actual equality.
    diag["observed_identical_arm_torques"] = gates.pop("same_arm_torques")
    del diag["same_torques_and_over_40mm_wrist_divergence"]
    gates.update(same_arm_position_targets=bool(targets_equal), same_arm_pd_parameters=bool(same_pd),
                 closed_arm_pd_verified=closed_pd, open_arm_pd_verified=opened_pd,
                 matched_wrist_path=departure is not None and departure <= .020,
                 open_final_fork_support=remaining_on_fork,
                 open_final_near_settled=bool(complete and np.all(drift <= .0025)))
    report.update(mode="same-arm-position-targets/independent-pd/open-jaw", success=all(gates.values()))
    diag.update(closed_max_pd_torque_residual_nm=closed_error, open_max_pd_torque_residual_nm=opened_error,
                open_final_fork_support_n=_range(o["support_n"][terminal]),
                open_final_speed_m_s=_range(speed), open_final_settled_displacement_m=_range(drift),
                closed_pd_parameters={k: a.tolist() for k, a in cp.items()},
                open_pd_parameters={k: a.tolist() for k, a in op.items()})
    report["thresholds"].update(max_open_wrist_path_difference_m=.020, max_final_settled_drift_m=.0025,
                                pd_torque_tolerance="2 * float32 epsilon * max(1 Nm, abs(expected torque))")
    return report
