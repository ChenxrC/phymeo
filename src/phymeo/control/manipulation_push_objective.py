"""Bounded current-pose push proposals, not current geometric or physical passes.

Material geometry is transported from a recorded strict planar query. The
quasistatic mobility approximation is a ranking heuristic, never a dynamics
oracle. A later full 3D path check and actual contact loads remain necessary.
"""

import json
from dataclasses import asdict, dataclass
from hashlib import sha256
from math import acos, atan2, cos, hypot, isfinite, sin

import numpy as np

from .manipulation_diagnostic import _require
from .manipulation_path import _items, _number
from .manipulation_push_rank import _validate_query


@dataclass(frozen=True)
class NativePlanarPushProposal:
    status: str
    input_binding: str
    index: int
    distance_ordinal: int
    distance_m: float
    reference_usable: bool
    point_world: tuple[float, float, float]
    outward_normal_world: tuple[float, float, float]
    object_com_world: tuple[float, float, float]
    contact_center_world: tuple[float, float, float]
    precontact_center_world: tuple[float, float, float]
    push_center_world: tuple[float, float, float]
    direction_world: tuple[float, float, float]
    nominal_yaw_moment: float
    mobility_radius_m: float
    predicted_translation_xy: tuple[float, float]
    predicted_yaw_change: float
    predicted_position_xy: tuple[float, float]
    predicted_yaw: float
    cost_before: float
    cost_after: float
    predicted_improvement: float
    sort_key: tuple[int, float, int, int]
    # Defaults also permit decoding archived proposals made before this policy.
    selection_policy: str = "pose-cost"
    selection_eligible: bool = True
    selection_reason: str = "legacy_pose_cost"
    position_error_before_m: float | None = None
    position_error_after_m: float | None = None
    yaw_error_before_rad: float | None = None
    yaw_error_after_rad: float | None = None


def _vector(value, width, name):
    return tuple(_number(item, name) for item in _items(value, name, width, width))


def _rotation(value, name):
    rows = _items(value, name, 3, 3, rank=2)
    result = np.array(tuple(_vector(row, 3, name) for row in rows), dtype=np.float64)
    _require(
        np.allclose(result.T @ result, np.eye(3), rtol=0, atol=2e-6)
        and abs(float(np.linalg.det(result)) - 1.0) <= 2e-6,
        name + " must be proper SO(3) within 2e-6; no repair is applied",
    )
    return result


def _tuple(value):
    return tuple(float(item) for item in value)


def _wrap(angle):
    return atan2(sin(angle), cos(angle))


def _cost(position, yaw, target, target_yaw, position_tolerance, yaw_tolerance):
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        xy = (np.asarray(target) - position) / position_tolerance
        angle = np.float64(_wrap(target_yaw - yaw)) / yaw_tolerance
        value = float(np.dot(xy, xy) + angle * angle)
    _require(isfinite(value), "nonfinite normalized pose cost")
    return value


def _joint_selection(
    usable,
    position_before,
    position_after,
    yaw_before,
    yaw_after,
    moment,
    position_tolerance,
    yaw_tolerance,
    improvement,
):
    """Finite-push heuristic admission, with no predicted/actual conflation."""
    if not usable:
        return False, "reference_unusable"
    position_met = position_before <= position_tolerance
    yaw_met = abs(yaw_before) <= yaw_tolerance
    if position_met and yaw_met:
        return False, "already_within_pose_tolerances"
    if position_met:
        if position_after > position_tolerance:
            return False, "position_tolerance_lost"
    elif not position_after < position_before:
        return False, "position_not_improved"
    if yaw_met:
        if abs(yaw_after) > yaw_tolerance:
            return False, "yaw_tolerance_lost"
    else:
        # Compare signs without an underflow-prone product. The zero of both
        # yaw angles is the same world +X axis used by the current rotation.
        if not ((moment > 0 and yaw_before > 0) or (moment < 0 and yaw_before < 0)):
            return False, "yaw_wrong_signed_moment"
        if not abs(yaw_after) < abs(yaw_before):
            return False, "yaw_not_improved"
    if not improvement > 0:
        return False, "nonpositive_pose_improvement"
    return True, "joint_pose_descent"


def propose_planar_pushes(
    query,
    *,
    reference_position,
    reference_rotation,
    current_position,
    current_rotation,
    target_xy,
    target_yaw,
    pusher_height_m,
    position_tolerance_m,
    yaw_tolerance_rad,
    push_distances_m=(0.005, 0.01, 0.02),
    stand_off_m=0.012,
    selection_policy="pose-cost",
):
    """Return every bounded proposal, sorted by reference usability and pose improvement.

    The caller binds the query to reference_position/reference_rotation and the
    current full body pose to its accepted physical observation. This function
    authenticates neither that external relationship nor the query's source SHA.
    It validates the public immutable query structure and hashes its complete
    payload as well as its claimed binding, all explicit inputs and this policy.

    Points and the COM use p_cur + R_cur R_ref.T (p_ref_material - p_ref);
    normals use R_cur R_ref.T without any rotation repair. Only the transported
    normal's XY projection is normalized for the proposed inward force. Original
    3D point, normal and COM are retained. Pusher centers have explicit fixed Z.
    Current body tilt relative to world +Z must not exceed .1 rad. This admission
    does not relax the old strict planar query or flatten actual physical state.

    r is the largest reference boundary-endpoint XY distance from the query COM.
    For unit inward f, Mz=((point-COM) cross f).z. A length l predicts body-origin
    translation l*f/(1+(Mz/r)^2) and yaw change l*Mz/(r*r+Mz*Mz). Their normalized
    squared XY/yaw target errors define improvement = cost_before - cost_after.
    The model omits friction, inertia, support, reachability and collision response.

    Lengths are 1..3 distinct explicit values in (0,.03] m; all <=384 combinations
    remain, including negative improvement and reference-unusable proposals.
    Sorting is (0 if reference_usable else 1, -improvement, index, distance_ordinal).
    The default pose-cost policy retains this ordering and its original input
    binding. Explicit joint-pose-descent instead puts eligible proposals first:
    every component outside its supplied Goal tolerance must strictly improve;
    an already qualified component must remain within that same tolerance.
    Unmet yaw additionally requires correctly signed world-Z moment. Total pose
    cost must still strictly improve. When both components already qualify no
    push is eligible. Rejected proposals and their reasons are retained, and
    input bindings include the new policy. Error fields are predictions only.
    No full 3D clearance, CCD, delivered force, actual motion or success is claimed.
    """
    _require(
        type(selection_policy) is str and selection_policy in ("pose-cost", "joint-pose-descent"),
        "unsupported selection policy",
    )
    _validate_query(query)
    reference = _vector(reference_position, 3, "reference position")
    current = _vector(current_position, 3, "current position")
    reference_r = _rotation(reference_rotation, "reference rotation")
    current_r = _rotation(current_rotation, "current rotation")
    _require(
        acos(max(-1.0, min(1.0, float(current_r[2, 2])))) <= 0.1, "current tilt exceeds .1 rad"
    )
    target = _vector(target_xy, 2, "target XY")
    target_yaw = _number(target_yaw, "target yaw")
    height = _number(pusher_height_m, "pusher height")
    pos_tol = _number(position_tolerance_m, "position tolerance")
    yaw_tol = _number(yaw_tolerance_rad, "yaw tolerance")
    stand_off = _number(stand_off_m, "stand off")
    _require(
        pos_tol > 0 and yaw_tol > 0 and stand_off > 0, "tolerances and stand off must be positive"
    )
    lengths = tuple(
        _number(value, "push length") for value in _items(push_distances_m, "push lengths", 1, 3)
    )
    _require(
        len(set(lengths)) == len(lengths) and all(0 < length <= 0.03 for length in lengths),
        "push lengths must be distinct and in (0,.03] metres",
    )
    _require(len(query.candidates) * len(lengths) <= 384, "proposal capacity exceeds 384")
    radius = max(
        hypot(
            float(point[0]) - float(query.object_com_world[0]),
            float(point[1]) - float(query.object_com_world[1]),
        )
        for edge in query.boundary
        for point in (edge.start_world, edge.end_world)
    )
    _require(
        isfinite(radius) and radius > 0, "reference mobility radius must be finite and positive"
    )
    rotation = current_r @ reference_r.T
    current_array, reference_array = np.asarray(current), np.asarray(reference)
    with np.errstate(over="ignore", invalid="ignore"):
        com = current_array + rotation @ (np.asarray(query.object_com_world) - reference_array)
    _require(np.isfinite(com).all(), "nonfinite transported COM")
    current_yaw = atan2(float(current_r[1, 0]), float(current_r[0, 0]))
    before = _cost(current_array[:2], current_yaw, target, target_yaw, pos_tol, yaw_tol)
    position_before = yaw_before = None
    if selection_policy == "joint-pose-descent":
        position_before = hypot(target[0] - current[0], target[1] - current[1])
        yaw_before = _wrap(target_yaw - current_yaw)
        _require(isfinite(position_before) and isfinite(yaw_before), "nonfinite pose error")

    saved_query = asdict(query)
    saved_query["q"] = query.q.tobytes().hex()
    saved_query["plane_axes_world"] = query.plane_axes_world.tobytes().hex()
    request = dict(
        schema="native-planar-push-objective-v1",
        query=saved_query,
        reference_position=reference,
        reference_rotation=tuple(_tuple(row) for row in reference_r),
        current_position=current,
        current_rotation=tuple(_tuple(row) for row in current_r),
        target_xy=target,
        target_yaw=target_yaw,
        pusher_height_m=height,
        position_tolerance_m=pos_tol,
        yaw_tolerance_rad=yaw_tol,
        push_distances_m=lengths,
        stand_off_m=stand_off,
        policy=dict(
            status="PROPOSAL_ONLY",
            rotation_tolerance=2e-6,
            maximum_tilt_rad=0.1,
            maximum_length_m=0.03,
            maximum_lengths=3,
            maximum_proposals=384,
            transport="Rcur-Rref-transpose-material-geometry; fixed-pusher-world-Z",
            normal="minus-unit-XY-projection-of-transported-material-normal",
            radius="max-reference-boundary-endpoint-XY-distance-to-query-COM",
            translation="l*f*r^2/(r^2+Mz^2)",
            yaw_change="l*Mz/(r^2+Mz^2)",
            prediction="body-origin-XY-plus-translation; wrapped-world-yaw-plus-yaw-change",
            objective="squared-XY/posTol plus squared-wrapped-yaw/yawTol",
            sorting="reference-unusable, negative-improvement, source-index, distance-ordinal",
        ),
    )
    if selection_policy == "joint-pose-descent":
        request["policy"].update(
            selection_policy=selection_policy,
            component_admission="strict-unmet-error-descent; preserve-met-Goal-tolerance",
            yaw_admission="same-world-zero; signed-Mz-matches-short-target-error",
            goal_admission="both-met-refuses-push; positive-total-improvement-required",
            sorting="eligible-first, negative-improvement, source-index, distance-ordinal",
        )
    binding = sha256(
        json.dumps(
            request,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
            default=lambda value: _number(value, "query scalar"),
        ).encode()
    ).hexdigest()
    records = []
    for candidate in query.candidates:
        with np.errstate(over="ignore", invalid="ignore"):
            point = current_array + rotation @ (np.asarray(candidate.point_world) - reference_array)
            normal = rotation @ np.asarray(candidate.outward_normal_world)
        _require(
            np.isfinite(point).all() and np.isfinite(normal).all(),
            "nonfinite transported material geometry",
        )
        projected = hypot(float(normal[0]), float(normal[1]))
        _require(
            isfinite(projected) and projected > 0, "material normal has no finite XY projection"
        )
        force = np.array((-normal[0] / projected, -normal[1] / projected, 0.0))
        with np.errstate(over="ignore", invalid="ignore"):
            moment = float((point[0] - com[0]) * force[1] - (point[1] - com[1]) * force[0])
            center = np.array(
                (
                    point[0] - query.pusher_radius_m * force[0],
                    point[1] - query.pusher_radius_m * force[1],
                    height,
                )
            )
            precontact = center - stand_off * force
        _require(
            isfinite(moment) and np.isfinite(center).all() and np.isfinite(precontact).all(),
            "nonfinite planar contact proposal",
        )
        # Scaling r and Mz together is algebraically identical to the stated
        # model, and avoids gratuitous overflow/underflow when squaring either.
        scale = max(radius, abs(moment))
        r_scaled, m_scaled = radius / scale, moment / scale
        denominator = r_scaled * r_scaled + m_scaled * m_scaled
        for ordinal, length in enumerate(lengths):
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                translation = length * (r_scaled * r_scaled / denominator) * force[:2]
                yaw_change = (
                    0.0
                    if moment == 0
                    else float((np.float64(length) / scale) * m_scaled / denominator)
                )
                predicted = current_array[:2] + translation
                push = center + length * force
            _require(
                np.isfinite(translation).all()
                and np.isfinite(predicted).all()
                and isfinite(yaw_change)
                and np.isfinite(push).all(),
                "nonfinite predicted proposal",
            )
            predicted_yaw = _wrap(current_yaw + yaw_change)
            after = _cost(predicted, predicted_yaw, target, target_yaw, pos_tol, yaw_tol)
            improvement = before - after
            _require(isfinite(improvement), "nonfinite predicted improvement")
            position_after = yaw_after = None
            eligible = candidate.usable and improvement > 0
            if not candidate.usable:
                reason = "reference_unusable"
            else:
                reason = "pose_cost_improvement" if eligible else "nonpositive_pose_improvement"
            admitted = candidate.usable
            if selection_policy == "joint-pose-descent":
                position_after = hypot(target[0] - predicted[0], target[1] - predicted[1])
                yaw_after = _wrap(target_yaw - predicted_yaw)
                _require(
                    isfinite(position_after) and isfinite(yaw_after),
                    "nonfinite predicted pose error",
                )
                eligible, reason = _joint_selection(
                    candidate.usable,
                    position_before,
                    position_after,
                    yaw_before,
                    yaw_after,
                    moment,
                    pos_tol,
                    yaw_tol,
                    improvement,
                )
                admitted = eligible
            key = (0 if admitted else 1, -improvement, candidate.index, ordinal)
            records.append(
                NativePlanarPushProposal(
                    "PROPOSAL_ONLY",
                    binding,
                    candidate.index,
                    ordinal,
                    length,
                    candidate.usable,
                    _tuple(point),
                    _tuple(normal),
                    _tuple(com),
                    _tuple(center),
                    _tuple(precontact),
                    _tuple(push),
                    _tuple(force),
                    moment,
                    radius,
                    _tuple(translation),
                    yaw_change,
                    _tuple(predicted),
                    predicted_yaw,
                    before,
                    after,
                    improvement,
                    key,
                    selection_policy,
                    eligible,
                    reason,
                    position_before,
                    position_after,
                    yaw_before,
                    yaw_after,
                )
            )
    return tuple(sorted(records, key=lambda record: record.sort_key))
