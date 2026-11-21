"""Bounded sampled release of initially overloaded declared geometry pairs.

This is a reference-path exception, not a physical contact allowance. The
ordinary audit remains unchanged and is retained even when it reports collision.
Only initial pusher/object pairs already beyond its 1 mm withdrawal cap may use
the additional monotone-release classification. No dynamics or CCD is performed.
"""

from dataclasses import dataclass

import numpy as np

from .manipulation_diagnostic import _require
from .manipulation_path import _items, _model_bytes, _number, audit_native_joint_path

WITHDRAW_DEPTH_M = 0.001
DISTANCE_LIMIT_M = 1.0


@dataclass(frozen=True)
class NativePushReleasePair:
    geometry_ids: tuple[int, int]
    geometry_names: tuple[str, str]
    initial_depth_m: float
    outward_normals_world: tuple


@dataclass(frozen=True)
class NativePushReleaseSample:
    time_s: float
    pair_distances_m: tuple
    pair_depths_m: tuple
    outward_step_projection_m: tuple


@dataclass(frozen=True)
class NativePushReleaseAudit:
    status: str
    sampled_release: bool
    reason: str
    original_audit: object
    eligible_pairs: tuple[NativePushReleasePair, ...]
    samples: tuple[NativePushReleaseSample, ...]
    separation_guard_m: float
    schema: str = "native-push-loaded-release-v1"
    distance_limit_m: float = DISTANCE_LIMIT_M
    qualification: str = "sampled frozen-object withdrawal only; not CCD, force or task success"


def audit_native_push_release(
    model,
    times,
    q_knots,
    joint_names,
    *,
    pusher_geom_name,
    object_geom_names,
    max_time_step=0.001,
    penetration_tolerance_m=1e-4,
    max_samples=4097,
):
    """Retain the ordinary audit and independently test a short outward release.

    Each eligible unordered geometry pair must be over the ordinary withdrawal
    cap at the first sample. Its maximum penetration cannot increase between
    any two samples, including after it first clears. Other pairs physically
    touching at the first sample keep their ordinary allowance. A previously
    noncontacting pair may never reach signed distance <= 0, even if its ordinary
    allowance would accept that shallow contact. The pusher's actual center displacement
    must project nonnegatively onto every initial eligible contact's outward
    normal, with positive net displacement for every pair. Each eligible pair
    must end at a positive native signed distance beyond an arithmetic guard.

    The grid is the original cubic-smoothstep audit, at most 250 ms and no
    coarser than 1 ms. Original filters, model bytes and unselected q coordinates
    stay unchanged. Missing/unsupported signed-distance evidence fails closed.
    """
    import mujoco as mj

    _require(isinstance(model, mj.MjModel), "original native model required")
    _require(
        type(pusher_geom_name) is str
        and pusher_geom_name
        and type(object_geom_names) is tuple
        and 1 <= len(object_geom_names) <= 2
        and all(type(name) is str and name for name in object_geom_names)
        and len(set(object_geom_names)) == len(object_geom_names)
        and pusher_geom_name not in object_geom_names,
        "release requires explicit distinct pusher/object geom names",
    )
    max_time_step = _number(max_time_step, "release max_time_step")
    _require(0 < max_time_step <= 0.001, "release grid may not exceed the original 1 ms step")
    stamps = tuple(_number(t, "release time") for t in _items(times, "release times", 2, 32))
    _require(stamps[-1] <= 0.25, "release duration must not exceed 250 ms")
    pusher = int(mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, pusher_geom_name))
    objects = tuple(
        int(mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, name)) for name in object_geom_names
    )
    _require(pusher >= 0 and all(g >= 0 for g in objects), "unknown release geometry")
    _require(
        all(model.geom_bodyid[g] != model.geom_bodyid[pusher] for g in objects),
        "release geometry must belong to distinct bodies",
    )
    binary = _model_bytes(mj, model)
    ordinary = audit_native_joint_path(
        model,
        stamps,
        q_knots,
        joint_names,
        allowed_contacts=tuple(
            (pusher_geom_name, name, WITHDRAW_DEPTH_M) for name in object_geom_names
        ),
        max_time_step=max_time_step,
        penetration_tolerance_m=penetration_tolerance_m,
        max_samples=max_samples,
    )
    declared = {tuple(sorted((pusher, other))) for other in objects}
    initial_physical_pairs = {
        tuple(sorted(c.geometry_ids)) for c in ordinary.samples[0].contacts if c.distance_m <= 0
    }
    eligible = tuple(
        sorted(
            {
                tuple(sorted(c.geometry_ids))
                for c in ordinary.samples[0].contacts
                if not c.sampled_clear and tuple(sorted(c.geometry_ids)) in declared
            }
        )
    )
    if not eligible:
        return NativePushReleaseAudit(
            "release_rejected",
            False,
            "no_initial_declared_violation",
            ordinary,
            (),
            (),
            0.0,
        )

    data = mj.MjData(model)
    data.qpos[:] = ordinary.samples[0].q
    mj.mj_kinematics(model, data)
    mj.mj_collision(model, data)
    _require(not any(int(item.number) for item in data.warning), "native release collision warning")
    normals = {pair: [] for pair in eligible}
    for contact in data.contact[: data.ncon]:
        ids = tuple(map(int, contact.geom))
        pair = tuple(sorted(ids))
        if pair not in normals:
            continue
        normal = np.asarray(contact.frame[:3], dtype=np.float64)
        _require(
            np.isfinite(normal).all() and abs(float(normal @ normal) - 1) <= 4e-6,
            "invalid release contact normal",
        )
        # Native normal points from geom1 toward geom2; release is away from
        # the object toward the pusher regardless of native collision ordering.
        normal = -normal if ids[0] == pusher else normal
        normals[pair].append(tuple(map(float, normal)))
    _require(all(normals.values()), "initial violating pairs lack native direction evidence")
    initial_center = data.geom_xpos[pusher].copy()
    previous_center = initial_center.copy()
    previous_depths, pairs, samples = None, (), []
    total_outward = np.zeros(len(eligible))
    maximum_coordinate = max(
        1.0, float(np.max(abs(data.geom_xpos))), float(np.max(abs(model.geom_size)))
    )
    reason = None
    # Ordinary records identify all collisions, including unselected geometry.
    # Only the initially violating identity set gets a different classification.
    for sample in ordinary.samples:
        if any(
            (not c.sampled_clear and tuple(sorted(c.geometry_ids)) not in eligible)
            or (c.distance_m <= 0 and tuple(sorted(c.geometry_ids)) not in initial_physical_pairs)
            for c in sample.contacts
        ):
            reason = reason or "new_or_other_pair_collision"
        data.qpos[:] = sample.q
        mj.mj_kinematics(model, data)
        _require(data.qpos.tobytes() == sample.q.tobytes(), "native release query changed q")
        center = data.geom_xpos[pusher].copy()
        _require(np.isfinite(center).all(), "nonfinite sampled release point")
        maximum_coordinate = max(maximum_coordinate, float(np.max(abs(data.geom_xpos))))
        distances, depths, projections = [], [], []
        for pair in eligible:
            distance = float(
                mj.mj_geomDistance(model, data, pair[0], pair[1], DISTANCE_LIMIT_M, None)
            )
            _require(np.isfinite(distance), "nonfinite native release distance")
            distances.append(distance)
            depths.append(max(0.0, -distance))
            projections.append(
                min(float(np.asarray(n) @ (center - previous_center)) for n in normals[pair])
            )
        depths = np.asarray(depths)
        if previous_depths is None:
            _require(
                np.all(depths > WITHDRAW_DEPTH_M),
                "initial violating pair signed distance is unsupported or inconsistent",
            )
            pairs = tuple(
                NativePushReleasePair(
                    pair,
                    tuple(model.geom(g).name for g in pair),
                    float(depth),
                    tuple(normals[pair]),
                )
                for pair, depth in zip(eligible, depths, strict=True)
            )
        elif np.any(depths > previous_depths):
            reason = reason or "initial_pair_depth_increased"
        if min(projections) < 0:
            reason = reason or "sample_direction_not_outward"
        total_outward += projections
        samples.append(
            NativePushReleaseSample(
                sample.time_s, tuple(distances), tuple(map(float, depths)), tuple(projections)
            )
        )
        previous_depths, previous_center = depths, center
    guard = 64 * np.finfo(np.float64).eps * maximum_coordinate
    if np.any(total_outward <= guard):
        reason = reason or "no_strict_outward_motion"
    if any(distance <= guard for distance in samples[-1].pair_distances_m):
        reason = reason or "initial_pair_not_strictly_separated"
    _require(_model_bytes(mj, model) == binary, "native model changed during release audit")
    accepted = reason is None
    return NativePushReleaseAudit(
        "sampled_release_ready" if accepted else "release_rejected",
        accepted,
        "initial_pairs_only_monotone_and_separated" if accepted else reason,
        ordinary,
        pairs,
        tuple(samples),
        float(guard),
    )
