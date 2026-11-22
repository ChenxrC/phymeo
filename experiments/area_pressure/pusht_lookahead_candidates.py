"""Bounded ordinary-path evaluation for saved-state push lookahead experiments.

These are planning candidates, never accepted dynamics. All geometry, IK, native
sampled path and material gates are the existing implementations. The explicit
pose-cost proposal policy can temporarily worsen yaw, but still requires positive
original total pose-cost improvement. Caller-owned planner counters never reset.
"""

from copy import deepcopy
from hashlib import sha256

import numpy as np


def _require(ok, reason):
    if not ok:
        raise ValueError("push lookahead candidates: " + reason)


def propose_candidates(driver, q, *, push_distances_m=None, selection_policy="pose-cost"):
    """Generate original immutable proposals and bindings at this complete q."""
    from phymeo.control.manipulation_push_objective import propose_planar_pushes

    q = np.asarray(q)
    _require(
        q.shape == (driver.model.nq,) and np.isfinite(q).all(),
        "finite complete source configuration",
    )
    position, rotation, pusher = driver.pose(q)
    extra = {} if push_distances_m is None else dict(push_distances_m=tuple(push_distances_m))
    return propose_planar_pushes(
        driver.query,
        reference_position=driver.ref_position,
        reference_rotation=driver.ref_rotation,
        current_position=position,
        current_rotation=rotation,
        target_xy=driver.target.position[:2],
        target_yaw=driver.target.yaw,
        pusher_height_m=float(pusher[2]),
        position_tolerance_m=driver.goal.position_tolerance_m,
        yaw_tolerance_rad=driver.goal.yaw_tolerance_rad,
        selection_policy=selection_policy,
        **extra,
    )


def freeze_material_faces(driver, q, asset, *, model_key, asset_sha256):
    """Use the formal scene asset/FK admission; bind the original body roles."""
    from .pusht_pose_replan import freeze_declared_material

    planner, model = driver.planner, driver.model
    _require(
        planner.model is model and asset.identity.compliant_body == planner.body_name,
        "same source model and actual compliant pusher body",
    )
    rigid = int(model.body(asset.identity.rigid_body).id)
    _require(
        all(int(model.geom_bodyid[model.geom(name).id]) == rigid for name in planner.object_names),
        "native object geometries and finite rigid material have the same role",
    )
    return freeze_declared_material(
        model, q, asset, model_key=model_key, scene_asset_sha256=asset_sha256
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


def _errors(error):
    values = []
    while error is not None:
        values.append(dict(type=type(error).__name__, message=str(error)))
        error = error.__cause__
    return values


def evaluate_candidates(
    driver, q, proposals, *, material_faces, indices=None, previous_direction=None
):
    """Try each requested original (index, stroke) in supplied proposal order.

    The same planner handles every singleton, including withdrawal when the caller
    supplies ``previous_direction``. Across this call the original per-segment IK
    budget is shared, rather than renewed for each singleton. A caller evaluating
    multiple levels can additionally tighten ``planner.max_ik`` around the whole
    search; this function never widens it. Counters include failed attempts.
    """
    from phymeo.control.manipulation_push_material import NativePushMaterialFaces
    from phymeo.control.manipulation_push_objective import NativePlanarPushProposal

    planner = driver.planner
    source = np.asarray(q)
    _require(
        planner.model is driver.model
        and source.shape == (driver.model.nq,)
        and source.dtype in (np.dtype("float32"), np.dtype("float64"))
        and np.isfinite(source).all(),
        "one complete finite source q on the original model",
    )
    _require(
        not planner.allow_loaded_withdrawal,
        "ordinary sampled path gates; no loaded-release exception",
    )
    _require(
        isinstance(material_faces, NativePushMaterialFaces),
        "formal frozen finite material required",
    )
    material_faces.validate_source(source)
    _require(
        type(proposals) is tuple
        and 1 <= len(proposals) <= 384
        and all(
            isinstance(p, NativePlanarPushProposal) and p.status == "PROPOSAL_ONLY"
            for p in proposals
        ),
        "bounded original proposal tuple",
    )
    policy = {p.selection_policy for p in proposals}
    _require(
        len(policy) == 1 and policy <= {"pose-cost", "joint-pose-descent", "joint-pose-balanced"},
        "one original proposal policy",
    )
    policy = next(iter(policy))
    keys = [(p.index, p.distance_ordinal) for p in proposals]
    _require(len(keys) == len(set(keys)), "unique original index and stroke ordinal")
    if indices is not None:
        indices = tuple(indices)
        _require(
            bool(indices)
            and len(set(indices)) == len(indices)
            and all(type(i) is int and any(p.index == i for p in proposals) for i in indices),
            "requested existing distinct source indices",
        )
    selected = tuple(p for p in proposals if indices is None or p.index in indices)
    admission = material_faces.audit(selected)
    by_key = {(row["index"], row["distance_ordinal"]): row for row in admission["proposals"]}
    initial = _budget(planner)
    old_max_ik = planner.max_ik
    # Match one original plan's cumulative per-segment allowance even though
    # exhaustive comparison deliberately pays for independent path checks.
    shared_ik_limit = min(old_max_ik, planner.ik_calls + planner.max_ik_per_segment)
    report = dict(
        schema="phymeo-pusht-lookahead-candidates-v1",
        source_q=source.copy(),
        source_q_dtype=source.dtype.str,
        source_q_sha256=sha256(source.tobytes()).hexdigest(),
        policy=policy,
        requested_indices=indices,
        previous_direction=None if previous_direction is None else tuple(previous_direction),
        material_admission=admission,
        budget_before=initial,
        shared_ik_limit=shared_ik_limit,
        complete=True,
        candidates=[],
        qualification="Original singleton IK and sampled path audits, no physics; no reachability claim for untested or rejected rows. Finite material at frozen source only; predicted object poses are not observed states.",
    )
    admitted_groups = []
    blocked = None
    try:
        planner.max_ik = shared_ik_limit
        for proposal in selected:
            key = (proposal.index, proposal.distance_ordinal)
            material = by_key[key]
            record = dict(
                index=proposal.index,
                distance_ordinal=proposal.distance_ordinal,
                proposal=proposal,
                route=None,
                status="not_admitted",
                evaluated=False,
                reachable=False,
                material_admission=deepcopy(material),
                budget_before=_budget(planner),
            )
            reason = None
            if not material["eligible"]:
                reason = material["reason"]
            elif not proposal.reference_usable:
                reason = "reference_unusable"
            elif (
                not np.isfinite(proposal.predicted_improvement)
                or proposal.predicted_improvement <= 0
            ):
                reason = "nonpositive_pose_improvement"
            elif policy != "pose-cost" and not proposal.selection_eligible:
                reason = proposal.selection_reason
            if reason is None:
                if proposal.index not in admitted_groups:
                    admitted_groups.append(proposal.index)
                if admitted_groups.index(proposal.index) >= planner.max_candidates:
                    reason = "candidate_budget"
                elif blocked is not None:
                    reason = blocked
                elif planner.ik_calls >= shared_ik_limit:
                    reason = "ik_budget"
                elif planner.path_calls >= planner.max_paths:
                    reason = "path_budget"
                if reason is not None:
                    record["status"] = "not_evaluated"
                    report["complete"] = False
            if reason is None:
                try:
                    record["evaluated"] = True
                    route = planner.plan(
                        source,
                        (proposal,),
                        previous_direction=previous_direction,
                        material_faces=material_faces,
                    )
                    record.update(route=route, status=route.status)
                    if route.status == "sampled_reference_ready":
                        _require(
                            route.proposal is proposal or route.proposal == proposal,
                            "returned original singleton proposal",
                        )
                        _require(len(route.nodes) >= 2, "actual native path-checked route nodes")
                        record["reachable"] = True
                    elif route.status in ("ik_budget", "path_budget", "candidate_budget"):
                        report["complete"] = False
                except Exception as error:
                    record.update(status="evaluation_error", errors=_errors(error))
                    report["complete"] = False
                    blocked = "preceding_evaluation_error"
            else:
                record["reason"] = reason
            record["budget_after"] = _budget(planner)
            report["candidates"].append(record)
    finally:
        planner.max_ik = old_max_ik
    report["budget_after"] = _budget(planner)
    report["requested_count"] = len(selected)
    report["evaluated_count"] = sum(row["evaluated"] for row in report["candidates"])
    report["reachable_count"] = sum(row["reachable"] for row in report["candidates"])
    report["ordering"] = (
        "supplied original proposal order; independent singleton audits; no cross-stroke IK cache"
    )
    return report


def compact_evaluation(report):
    """JSON-ready complete diagnostics, sharing only repeated material matrices.

    The actual route objects remain intact in ``report``. Shared geometry/source
    arrays are stored once in top-level material_admission. Every IK result,
    sampled path result, rejection and per-proposal face decision is retained.
    """
    from scripts.run_manipulation_control import plain

    result = plain(report)
    binding = report["material_admission"]["binding"]
    for candidate in result["candidates"]:
        route = candidate["route"]
        if route is None:
            continue
        for attempt in route["attempts"]:
            if isinstance(attempt, dict) and attempt.get("kind") == "material_admission":
                _require(attempt["binding"] == binding, "same material audit for compact sharing")
                for key in ("triangles_world", "source_q", "face_ids"):
                    attempt.pop(key, None)
                attempt["shared_material_admission"] = "$.material_admission"
    return result
