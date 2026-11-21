"""Best-effort evidence from rejected public PushT device scratch.

Call inside ``except`` while the batch is open, save the returned dict, then use
bare ``raise``. This experimental reader neither publishes nor retries a step.
Scratch has no attempt identifier: even matching input q/v cannot establish that
an outer-loop buffer was written by the rejected attempt. HOST preflight failure
and unattempted buckets can leave older contents. No candidate q/v is reported.
"""

import math

import numpy as np

_STATUS = {
    0: "no_device_error_recorded_not_an_acceptance",
    1: "contact_or_constraint_capacity",
    2: "nonfinite_candidate",
    3: "joint_limit_domain",
    4: "nonpositive_mass_matrix",
    5: "normal_solver_not_converged",
    6: "nonpositive_normal_matrix",
    7: "constraint_solver_budget",
    8: "area_friction_solver_or_publication",
}
_NATIVE_COLUMNS = (
    "working_set_changes",
    "scaled_kkt_residual",
    "absolute_kkt_residual",
    "bound_violation",
    "native_rows",
    "free_rows",
    "converged",
    "budget",
    "scaled_physical_mass_residual",
    "kkt_scale",
)
_COLUMNS = {
    "area_native_meta": ("final_status", "last_outer_index", "native_calls", "converged"),
    "area_native_trace_meta": (
        "evaluation_phase_0_baseline_1_full_2_half",
        "status_before_area_evaluation",
        "converged",
    ),
    "trace_phase_values": (
        "0: baseline with zero area trial load",
        "1: this evaluation uses full updates if not converged",
        "2: this evaluation uses a half update if not converged; half mode persists within this solve",
    ),
    "solver_diagnostics": _NATIVE_COLUMNS,
    "constraint_counts": ("detected_contacts", "active_contacts", "native_rows_required"),
    "friction_sizes": ("nv", "area_samples", "outer_budget", "line_search_budget"),
    "friction_vectors_banks": (
        "A_inverse_source_load",
        "native_response_velocity",
        "area_evaluation_velocity",
        "area_load_delta",
        "area_trial_load",
        "evaluated_area_load",
        "forward_solve_scratch",
        "A_inverse_h_load_delta",
    ),
    "input_time": ("compensated_sum", "compensation"),
}


def _exception(error):
    try:
        message = str(error)
    except BaseException:
        message = "<exception message unavailable>"
    return {"type": type(error).__name__, "message": message[:4096]}


def _json(value):
    """Preserve diagnostic nonfinites explicitly without invalid JSON numbers."""
    if isinstance(value, np.generic):
        return _json(value.item())
    if isinstance(value, np.ndarray):
        return _json(value.tolist())
    if isinstance(value, (list, tuple)):
        return [_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else ("+Infinity" if value > 0 else "-Infinity")
    return value


def _read(device, buffer, shape):
    # Validate the small allocation before downloading, including the 25-row bound.
    if tuple(buffer.shape) != shape:
        raise ValueError(f"diagnostic buffer shape {tuple(buffer.shape)} != {shape}")
    result = np.asarray(device._get(buffer))
    if result.shape != shape or result.dtype.kind not in "iuf":
        raise ValueError("diagnostic download shape/dtype mismatch")
    return result


def _capture_owner(owner, bucket, single_bucket, limit, report):
    keys = tuple(owner.world_keys)
    if not keys or len(keys) > 32:
        raise ValueError("expected 1..32 resident area worlds")
    device, raw = owner._device, {}
    lanes = [
        dict(
            world=int(key),
            bucket=bucket,
            lane=i,
            publication="unpublished",
            freshness="unverified",
            availability="unavailable",
            layout_interpretable=False,
        )
        for i, key in enumerate(keys)
    ]
    report["lanes"].extend(lanes)
    nworld = len(keys)
    native = bool(getattr(device.plan, "resident_native_contacts", False))
    applicable = single_bucket and native and getattr(device, "mode", None) == "cuda"
    fields = [("status", "status", (nworld,))]
    if native:
        nv, nq, width, active = device.plan.nv, device.plan.nq, device.width, device.active
        if not 6 <= nv <= 8 or not 1 <= nq <= 16 or width != 8 or active not in (0, 1):
            raise ValueError("unsupported resident native/area diagnostic layout")
        packed = 6 if nv == 6 else 8
        fields.extend(
            [
                ("area_native_meta", "area_native_meta", (nworld, 4)),
                ("area_native_trace_meta", "area_native_trace_meta", (nworld, 25, 3)),
                ("area_native_trace", "area_native_trace", (nworld, 25, 14 + 2 * packed)),
                ("solver_diagnostics", "solver_diagnostics", (nworld, 10)),
                ("constraint_counts", "constraint_counts", (nworld, 3)),
                ("friction_sizes", "friction_sizes", (nworld, 4)),
                ("friction_pack_status", "friction_pack_status", (nworld,)),
                ("friction_vectors", "friction_vectors", (nworld, 8, packed)),
                ("area_native_applied", "area_native_applied", (nworld, width)),
                ("native_trial_physical_acc", "physical_acc", (nworld, width)),
                ("input_q", ("q", active), (nworld, max(8, nq))),
                ("input_v", ("v", active), (nworld, max(8, nv))),
                ("input_time", ("time", active), (nworld, 2)),
            ]
        )
        trace_columns = (
            *_NATIVE_COLUMNS,
            "scaled_area_load_delta",
            "scaled_response_velocity_delta",
            "absolute_area_load_delta",
            "absolute_response_velocity_delta",
            *(f"trial_velocity_{i}" for i in range(packed)),
            *(f"area_load_delta_{i}" for i in range(packed)),
        )
        for lane in lanes:
            lane.update(
                active_bank=int(active), nv=int(nv), nq=int(nq), trace_columns=list(trace_columns)
            )
    error_start = len(report["capture_errors"])
    try:
        with device.tx.use_runtime(device.runtime):
            device.runtime.sync()
            for name, source, shape in fields:
                try:
                    buffer = (
                        getattr(device, source[0])[source[1]]
                        if isinstance(source, tuple)
                        else getattr(device, source)
                    )
                    raw[name] = _read(device, buffer, shape)
                except BaseException as error:
                    report["capture_errors"].append(
                        dict(bucket=bucket, field=name, **_exception(error))
                    )
    except BaseException as error:
        report["capture_errors"].append(dict(bucket=bucket, field="runtime", **_exception(error)))
    complete = len(report["capture_errors"]) == error_start
    for i, lane in enumerate(lanes):
        lane["availability"] = "captured" if complete else ("partial" if raw else "unavailable")
        lane["layout_interpretable"] = applicable and complete
        lane["interpretation_scope"] = (
            "single-bucket-resident-native-area-outer-loop"
            if lane["layout_interpretable"]
            else "raw-scratch-only-no-current-attempt-inference"
        )
        for name, values in raw.items():
            if name not in ("area_native_trace_meta", "area_native_trace"):
                lane[name] = _json(values[i])
        if "status" in raw:
            lane["status_name"] = _STATUS.get(int(raw["status"][i]), "unknown_status_enum")
        if "area_native_meta" in raw:
            calls = int(raw["area_native_meta"][i, 2])
            keep = min(max(calls, 0), 25, limit)
            lane.update(
                trace_calls_recorded=calls,
                trace_rows_retained=keep,
                trace_truncated=calls > keep,
                trace_count_valid=0 <= calls <= 25,
            )
            for name in ("area_native_trace_meta", "area_native_trace"):
                if name in raw:
                    lane[name] = _json(raw[name][i, :keep])
            if not lane["trace_count_valid"]:
                lane["layout_interpretable"] = False
                lane["interpretation_scope"] = "invalid-trace-count-raw-scratch-only"


def capture_rejected_candidate(batch, error, *, max_trace_rows=25):
    """Return bounded JSON evidence; diagnostic failures never replace ``error``.

    Use only in a Scene.step rejection handler, before batch cleanup. Read under
    its lock and existing runtime; no stepping, bank swap, reset or observation
    publication occurs. Download accounting may increase. ``layout_interpretable``
    concerns column layout only, never freshness or acceptance. Status is an enum.
    A failed lane's native trial acceleration is scratch, not accepted qacc; the
    post-tail efc gradient and candidate integration state are deliberately absent.
    """
    report = dict(
        schema="phymeo-pusht-rejected-candidate-v1",
        publication="unpublished",
        freshness="unverified",
        original_error=_exception(error),
        lanes=[],
        capture_errors=[],
        columns=_json_columns(),
        nonfinite_encoding="NaN/+Infinity/-Infinity strings",
        limits=[
            "No attempt token: preflight failure or unattempted buckets may leave old scratch.",
            "Status zero and converged metadata do not establish batch acceptance.",
            "Trace status is captured before area evaluation, not the final stage status.",
            "Only active-bank input q/v/time are state; all other arrays are unpublished scratch.",
        ],
    )
    try:
        if type(max_trace_rows) is not int or not 0 <= max_trace_rows <= 25:
            raise ValueError("max_trace_rows must be an integer in 0..25")
        with batch._lock:
            batch._check()
            buckets = tuple(batch._buckets)
            for bucket, owner in enumerate(buckets):
                try:
                    _capture_owner(owner, bucket, len(buckets) == 1, max_trace_rows, report)
                except BaseException as capture_error:
                    report["capture_errors"].append(
                        dict(bucket=bucket, field="owner", **_exception(capture_error))
                    )
    except BaseException as capture_error:
        report["capture_errors"].append(dict(field="batch", **_exception(capture_error)))
    report["capture_complete"] = bool(report["lanes"]) and not report["capture_errors"]
    return report


def _json_columns():
    return {key: list(value) for key, value in _COLUMNS.items()}
