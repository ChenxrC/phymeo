"""Bounded two-push predictions using original native route/material checks.

The caller binds q to a real saved entry. The first route's robot endpoint and
original quasistatic object prediction create a separate synthetic configuration,
never a velocity, receipt, checkpoint or accepted physical state. The second
ordinary planner must withdraw from that complete configuration before approach.
Both layers share one original per-segment IK allowance, global native counters
and the caller's unchanged remaining physical-step budget.
"""

from dataclasses import asdict, is_dataclass
from math import atan2, cos, sin

import numpy as np

H = 0.001
SCHEMA = "phymeo-pusht-lookahead-pairs-v1"


def _require(ok, reason):
    if not ok:
        raise ValueError("push lookahead pairs: " + reason)


def _field(value, name):
    return value[name] if isinstance(value, dict) else getattr(value, name)


def _plain(value):
    if is_dataclass(value):
        return _plain(asdict(value))
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if hasattr(value, "__dict__"):
        return _plain(vars(value))
    return value


def _vector(value, shape, label):
    result = np.asarray(value, dtype=float)
    _require(result.shape == shape and np.isfinite(result).all(), label + " finite shape")
    return result


def _wrap(value):
    _require(np.isfinite(value), "finite world yaw")
    return atan2(sin(value), cos(value))


def _steps(route):
    nodes = _field(route, "nodes")
    _require(len(nodes) >= 2 and nodes[0][0] == 0, "complete original reference nodes")
    times = [float(node[0]) for node in nodes]
    _require(
        all(np.isfinite(t) for t in times) and all(a < b for a, b in zip(times, times[1:])),
        "finite increasing route times",
    )
    count = round(times[-1] / H)
    _require(
        count > 0 and abs(count * H - times[-1]) <= 1e-12,
        "original tick-aligned full route duration",
    )
    return count


def _route_source(driver, q, route):
    _require(
        _field(route, "status") == "sampled_reference_ready",
        "first or second route was not qualified",
    )
    source = _vector(q, (driver.model.nq,), "complete source q")
    _require(
        np.array_equal(_field(route, "source_q"), source),
        "route source differs from complete configuration",
    )
    projection = _vector(
        _field(route, "initial_reference_projection"), (len(driver.qa),), "route projection"
    )
    _require(
        np.all(projection == 0), "projected start is not the supplied real or predicted endpoint"
    )
    nodes = _field(route, "nodes")
    _steps(route)
    _require(
        np.array_equal(nodes[0][2], source[list(driver.qa)]),
        "route first robot node differs from source",
    )
    _vector(nodes[-1][2], (len(driver.qa),), "actual route robot endpoint")
    return source


def predicted_endpoint(driver, q, route):
    """Transport the observed root pose by the original heuristic, preserving tilt."""
    import mujoco as mj

    source = _route_source(driver, q, route)
    model, body = driver.model, int(driver.object_body)
    _require(
        int(model.body_parentid[body]) == 0 and int(model.body_jntnum[body]) == 1,
        "predicted object must be one original world-root free body",
    )
    joint = int(model.body_jntadr[body])
    _require(
        int(model.jnt_type[joint]) == int(mj.mjtJoint.mjJNT_FREE), "original object free joint"
    )
    adr = int(model.jnt_qposadr[joint])
    _require(
        not set(driver.qa).intersection(range(adr, adr + 7)),
        "robot and free-body coordinates are distinct",
    )
    proposal = _field(route, "proposal")
    _require(
        _field(proposal, "status") == "PROPOSAL_ONLY"
        and _field(proposal, "selection_policy") == "pose-cost",
        "unaltered original pose-cost prediction",
    )
    position, rotation, _ = driver.pose(np.asarray(q))
    position, rotation = (
        _vector(position, (3,), "source object position"),
        _vector(rotation, (3, 3), "source rotation"),
    )
    yaw = atan2(rotation[1, 0], rotation[0, 0])
    shift = _vector(
        _field(proposal, "predicted_translation_xy"), (2,), "original predicted translation"
    )
    predicted_xy = _vector(
        _field(proposal, "predicted_position_xy"), (2,), "original predicted position"
    )
    delta, target_yaw = (
        float(_field(proposal, "predicted_yaw_change")),
        float(_field(proposal, "predicted_yaw")),
    )
    _require(
        np.allclose(predicted_xy, position[:2] + shift, rtol=0, atol=1e-12)
        and abs(_wrap(target_yaw - yaw - delta)) <= 2e-12,
        "prediction belongs to current object pose",
    )
    _require(
        np.array_equal(source[adr : adr + 3], position),
        "free coordinates are the actual world body origin",
    )
    quaternion = source[adr + 3 : adr + 7]
    norm = float(np.linalg.norm(quaternion))
    _require(abs(norm * norm - 1) <= 4e-5, "source native free quaternion guard")
    # This normalization creates only a synthetic planning quaternion. Source q
    # remains byte-identical; the normalization amount is explicitly reported.
    w, x, y, z = quaternion / norm
    c, s = cos(delta / 2), sin(delta / 2)
    result = source.copy()
    result[list(driver.qa)] = _field(route, "nodes")[-1][2]
    result[adr : adr + 2] = predicted_xy
    result[adr + 3 : adr + 7] = (c * w - s * z, c * x - s * y, c * y + s * x, c * z + s * w)
    actual_position, actual_rotation, _ = driver.pose(result)
    actual_yaw = atan2(actual_rotation[1, 0], actual_rotation[0, 0])
    _require(
        np.allclose(actual_position[:2], predicted_xy, rtol=0, atol=1e-12)
        and actual_position[2] == position[2]
        and abs(_wrap(actual_yaw - target_yaw)) <= 2e-6,
        "synthetic FK matches original planar prediction",
    )
    return dict(
        kind="synthetic-predicted-after-first",
        q=result.tolist(),
        source_q=np.asarray(q).tolist(),
        object_position_world=np.asarray(actual_position).tolist(),
        object_yaw_rad=actual_yaw,
        source_free_quaternion_norm=norm,
        source_free_quaternion_normalization_max_delta=float(
            np.max(abs(quaternion / norm - quaternion))
        ),
        world_yaw_increment_rad=delta,
        free_height_preserved=True,
        tilt_preserved=True,
        reference_steps=_steps(route),
        velocity_prediction=None,
        physical_time=None,
        qualification="original heuristic XY/world-Z rotation plus actual reference robot endpoint; no dynamics, force, velocity or accepted-state claim",
    )


def pair_metrics(
    first,
    second,
    *,
    initial_position_xy,
    initial_yaw,
    target_xy,
    target_yaw,
    position_tolerance_m,
    yaw_tolerance_rad,
    baseline_prediction=None,
):
    """Filter by net XY and shortest-angle improvement at the original real input."""
    start, target = (
        _vector(initial_position_xy, (2,), "real input XY"),
        _vector(target_xy, (2,), "target XY"),
    )
    first_xy = _vector(_field(first, "predicted_position_xy"), (2,), "first predicted XY")
    final_xy = _vector(_field(second, "predicted_position_xy"), (2,), "second predicted XY")
    first_yaw, final_yaw = (
        float(_field(first, "predicted_yaw")),
        float(_field(second, "predicted_yaw")),
    )
    _require(
        np.isfinite((initial_yaw, target_yaw, position_tolerance_m, yaw_tolerance_rad)).all()
        and position_tolerance_m > 0
        and yaw_tolerance_rad > 0,
        "unchanged finite original goal",
    )
    _require(
        np.allclose(
            first_xy,
            start + _vector(_field(first, "predicted_translation_xy"), (2,), "first shift"),
            rtol=0,
            atol=1e-12,
        )
        and abs(_wrap(first_yaw - initial_yaw - float(_field(first, "predicted_yaw_change"))))
        <= 2e-12,
        "first prediction continues real input pose",
    )
    _require(
        np.allclose(
            final_xy,
            first_xy + _vector(_field(second, "predicted_translation_xy"), (2,), "second shift"),
            rtol=0,
            atol=1e-12,
        )
        and abs(_wrap(final_yaw - first_yaw - float(_field(second, "predicted_yaw_change"))))
        <= 2e-6,
        "second prediction continues first predicted pose",
    )

    def errors(xy, yaw):
        position = float(np.linalg.norm(target - xy))
        signed = _wrap(target_yaw - yaw)
        cost = (position / position_tolerance_m) ** 2 + (signed / yaw_tolerance_rad) ** 2
        _require(np.isfinite(cost), "finite original normalized pose cost")
        return dict(
            position_error_m=position,
            signed_yaw_error_rad=signed,
            absolute_yaw_error_rad=abs(signed),
            normalized_pose_cost=cost,
        )

    before, middle, after = (
        errors(start, initial_yaw),
        errors(first_xy, first_yaw),
        errors(final_xy, final_yaw),
    )
    position_gain = before["position_error_m"] - after["position_error_m"]
    yaw_gain = before["absolute_yaw_error_rad"] - after["absolute_yaw_error_rad"]
    reasons = []
    if not position_gain > 0:
        reasons.append("combined_position_not_improved")
    if not yaw_gain > 0:
        reasons.append("combined_absolute_yaw_not_improved")
    baseline = None
    if baseline_prediction is not None:
        predicted = errors(
            _vector(
                _field(baseline_prediction, "predicted_position_xy"), (2,), "baseline predicted XY"
            ),
            float(_field(baseline_prediction, "predicted_yaw")),
        )
        baseline = dict(
            predicted=predicted,
            position_advantage_m=predicted["position_error_m"] - after["position_error_m"],
            absolute_yaw_advantage_rad=predicted["absolute_yaw_error_rad"]
            - after["absolute_yaw_error_rad"],
            normalized_cost_advantage=predicted["normalized_pose_cost"]
            - after["normalized_pose_cost"],
            scope="saved baseline heuristic at the same entry; not the baseline's future measured outcome",
        )
    return dict(
        prediction_only=True,
        before=before,
        after_first=middle,
        after_second=after,
        first_yaw_worsens=middle["absolute_yaw_error_rad"] > before["absolute_yaw_error_rad"],
        position_improvement_m=position_gain,
        absolute_yaw_improvement_rad=yaw_gain,
        normalized_cost_improvement=before["normalized_pose_cost"] - after["normalized_pose_cost"],
        net_both_improve=not reasons,
        reasons=reasons,
        baseline_comparison=baseline,
    )


def transfer_summary(route):
    """Describe this original reverse-direction withdrawal and sampled approach."""
    attempts = _field(route, "attempts")
    withdrawal = next(
        (a for a in attempts if isinstance(a, dict) and a.get("kind") == "withdraw"), None
    )
    paths = []

    def collect(value):
        if isinstance(value, dict):
            paths.extend(value.get("paths", ()))
            for key, nested in value.items():
                if key != "paths":
                    collect(nested)
        elif isinstance(value, (tuple, list)):
            for nested in value:
                collect(nested)

    collect(attempts)
    withdraw_paths = (
        []
        if withdrawal is None
        else [a for label, a in withdrawal.get("paths", ()) if label == "withdraw"]
    )
    withdraw_clear = bool(withdraw_paths) and all(
        _field(a, "sampled_clear") for a in withdraw_paths
    )
    approach_clear = any(label == "approach" and _field(a, "sampled_clear") for label, a in paths)
    status = _field(route, "status")
    initial_contacts = []
    if withdraw_paths:
        samples = _field(withdraw_paths[0], "samples")
        if samples:
            initial_contacts = _plain(_field(samples[0], "contacts"))
    return dict(
        geometry_feasible=bool(
            status == "sampled_reference_ready" and withdraw_clear and approach_clear
        ),
        status=status,
        withdrawal_requested=withdrawal is not None,
        withdrawal_checked=bool(withdraw_paths),
        withdrawal_sampled_clear=bool(withdraw_clear),
        approach_sampled_clear=bool(approach_clear),
        reason=withdrawal.get("reason", status)
        if withdrawal is not None
        else "withdrawal_not_reached",
        initial_predicted_endpoint_contacts=initial_contacts,
        policy="original 25mm reverse-first-direction withdrawal, then original direct/orbit approach; unchanged sampled allowances, not CCD or strict zero penetration",
    )


def _budget(planner):
    return dict(
        ik_calls=planner.ik_calls,
        path_calls=planner.path_calls,
        max_ik=planner.max_ik,
        max_paths=planner.max_paths,
        max_ik_per_segment=planner.max_ik_per_segment,
        max_candidates=planner.max_candidates,
    )


def _identity(proposal):
    return [
        _field(proposal, "index"),
        _field(proposal, "distance_ordinal"),
        _field(proposal, "input_binding"),
    ]


def search_pairs(
    driver,
    q,
    material_at,
    *,
    remaining_steps,
    first_indices,
    push_distances_m=None,
    baseline_prediction=None,
):
    """Evaluate all requested first atoms once, then spend the shared remainder.

    material_at(complete_q) must freeze the same actual all-lateral asset through
    the existing admission helper. First indices are externally derived, never
    hard-coded. The second queue excludes mathematically ineligible net outcomes
    before native IK; all original proposals and filter reasons remain recorded.
    """
    from . import pusht_lookahead_candidates as candidates

    source = np.asarray(q)
    _require(
        source.shape == (driver.model.nq,)
        and source.dtype in (np.dtype("float32"), np.dtype("float64"))
        and np.isfinite(source).all(),
        "complete caller-bound real q",
    )
    _require(
        type(remaining_steps) is int and remaining_steps >= 0, "unchanged remaining physical steps"
    )
    first_indices = tuple(first_indices)
    _require(
        first_indices
        and len(first_indices) == len(set(first_indices))
        and all(type(index) is int for index in first_indices),
        "explicit distinct first contact indices",
    )
    planner = driver.planner
    _require(
        not planner.allow_loaded_withdrawal, "ordinary transfer only; no loaded-release exception"
    )
    before, old_max = _budget(planner), planner.max_ik
    limit = min(old_max, before["ik_calls"] + min(64, planner.max_ik_per_segment))
    position, rotation, _ = driver.pose(source)
    initial_xy = _vector(position, (3,), "real input body position")[:2]
    initial_yaw = atan2(rotation[1, 0], rotation[0, 0])
    goals = dict(
        initial_position_xy=initial_xy,
        initial_yaw=initial_yaw,
        target_xy=driver.target.position[:2],
        target_yaw=driver.target.yaw,
        position_tolerance_m=driver.goal.position_tolerance_m,
        yaw_tolerance_rad=driver.goal.yaw_tolerance_rad,
        baseline_prediction=baseline_prediction,
    )
    result = dict(
        schema=SCHEMA,
        real_input=dict(
            kind="caller-bound-real-entry",
            q=source.tolist(),
            object_position_world=np.asarray(position).tolist(),
            object_yaw_rad=initial_yaw,
        ),
        goal=_plain(
            {
                key: value
                for key, value in goals.items()
                if key not in ("initial_position_xy", "initial_yaw", "baseline_prediction")
            }
        ),
        first_indices=list(first_indices),
        first_layer=None,
        second_layers=[],
        pairs=[],
        selected_pair=None,
        best_geometry_pair=None,
        search_complete=True,
        material_sharing_scope="shared_material_admission paths are relative to each enclosing first_layer or second_layers[].evaluation object",
        qualification="bounded original heuristic and native sampled routes only; predicted state is not physical state; no force, success, CCD or global-optimum claim",
    )
    try:
        planner.max_ik = limit
        first_proposals = candidates.propose_candidates(
            driver, source, push_distances_m=push_distances_m, selection_policy="pose-cost"
        )
        first_material = material_at(source.copy())
        first = candidates.evaluate_candidates(
            driver,
            source,
            first_proposals,
            material_faces=first_material,
            indices=first_indices,
            previous_direction=None,
        )
        result["first_layer"] = candidates.compact_evaluation(first)
        result["search_complete"] = first["complete"]
        for atom in first["candidates"]:
            if not atom["reachable"]:
                continue
            key = _identity(atom["proposal"])
            branch = dict(
                first_identity=key,
                status="not_evaluated",
                synthetic_endpoint=None,
                prediction_filter=[],
                evaluation=None,
                truncation_reason=None,
            )
            result["second_layers"].append(branch)
            if planner.ik_calls >= limit or planner.path_calls >= planner.max_paths:
                branch["truncation_reason"] = (
                    "shared_ik_budget" if planner.ik_calls >= limit else "original_path_budget"
                )
                result["search_complete"] = False
                continue
            try:
                endpoint = predicted_endpoint(driver, source, atom["route"])
            except ValueError as error:
                branch.update(
                    status="predicted_endpoint_rejected",
                    errors=[dict(type=type(error).__name__, message=str(error))],
                )
                continue
            branch["synthetic_endpoint"] = endpoint
            predicted_q = np.asarray(endpoint["q"], dtype=np.float64)
            second_proposals = candidates.propose_candidates(
                driver, predicted_q, push_distances_m=push_distances_m, selection_policy="pose-cost"
            )
            eligible, by_key = [], {}
            for proposal in second_proposals:
                metrics = pair_metrics(atom["proposal"], proposal, **goals)
                identity = _identity(proposal)
                branch["prediction_filter"].append(
                    dict(
                        identity=identity,
                        proposal=_plain(proposal),
                        metrics=metrics,
                        requested_native_evaluation=metrics["net_both_improve"],
                        reason="net_pose_improvement"
                        if metrics["net_both_improve"]
                        else ";".join(metrics["reasons"]),
                    )
                )
                if metrics["net_both_improve"]:
                    eligible.append(proposal)
                    by_key[tuple(identity)] = metrics
            if not eligible:
                branch["status"] = "no_net_jointly_improving_prediction"
                continue
            material = material_at(predicted_q.copy())
            second = candidates.evaluate_candidates(
                driver,
                predicted_q,
                tuple(eligible),
                material_faces=material,
                previous_direction=_field(atom["proposal"], "direction_world"),
            )
            branch.update(status="evaluated", evaluation=candidates.compact_evaluation(second))
            if not second["complete"]:
                result["search_complete"] = False
                branch["truncation_reason"] = "native_evaluation_incomplete"
            for checked in second["candidates"]:
                identity = _identity(checked["proposal"])
                metrics = by_key[tuple(identity)]
                transfer = None if checked["route"] is None else transfer_summary(checked["route"])
                geometry = bool(
                    checked["reachable"] and transfer is not None and transfer["geometry_feasible"]
                )
                total = None
                if geometry:
                    try:
                        _route_source(driver, predicted_q, checked["route"])
                    except ValueError as error:
                        geometry = False
                        transfer.update(geometry_feasible=False, reason=str(error))
                    else:
                        total = endpoint["reference_steps"] + _steps(checked["route"])
                within = bool(total is not None and total <= remaining_steps)
                result["pairs"].append(
                    dict(
                        first_identity=key,
                        second_identity=identity,
                        metrics=metrics,
                        geometry_feasible=geometry,
                        transfer=transfer,
                        evaluated=checked["evaluated"],
                        status=checked["status"],
                        reference_steps=total,
                        remaining_steps=remaining_steps,
                        within_remaining_steps=within,
                        admissible_prediction=bool(
                            geometry and within and metrics["net_both_improve"]
                        ),
                    )
                )
    finally:
        planner.max_ik = old_max
    after = _budget(planner)
    _require(
        before["ik_calls"] <= after["ik_calls"] <= limit
        and before["path_calls"] <= after["path_calls"] <= before["max_paths"],
        "shared cumulative budget accounting",
    )
    result["budget"] = dict(
        before=before,
        after=after,
        shared_ik_limit=limit,
        new_ik_calls=after["ik_calls"] - before["ik_calls"],
        new_path_calls=after["path_calls"] - before["path_calls"],
        remaining_steps=remaining_steps,
        original_limits_restored=True,
        counters_reset=False,
        exhausted_ik=after["ik_calls"] >= limit,
        exhausted_paths=after["path_calls"] >= before["max_paths"],
    )

    def order(record):
        return (
            record["metrics"]["after_second"]["normalized_pose_cost"],
            record["first_identity"],
            record["second_identity"],
        )

    geometric = [
        row
        for row in result["pairs"]
        if row["geometry_feasible"] and row["metrics"]["net_both_improve"]
    ]
    admitted = [row for row in geometric if row["within_remaining_steps"]]
    if geometric:
        result["best_geometry_pair"] = min(geometric, key=order)
    if admitted:
        result["selected_pair"] = min(admitted, key=order)
    result["status"] = (
        "predicted_pair_within_budget"
        if admitted
        else "no_pair_within_remaining_steps"
        if geometric
        else "incomplete_search_no_qualified_pair"
        if not result["search_complete"]
        else "no_qualified_pair"
    )
    result["selection_scope"] = (
        "lowest original final normalized pose cost among evaluated net-improving admissible pairs; untested branches remain unknown"
    )
    return _plain(result)
