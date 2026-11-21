"""Five frozen RR inputs: resident T4 rows with HOST area-load fixed-point trials.

This is an experiment, not Scene.step: one inactive integration candidate, no accepted-state publication,
checkpoint, or entirely resident area producer is claimed. The sole CUDA runtime
belongs to the ordinary TyxConstraintDevice. All M/J/native row coefficients used
by its solver are assembled there. CPU MuJoCo is used only by the independent
original-model reference. HOST downloads and extra-load uploads are counted.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco as mj
import numpy as np

from phymeo.backends.mujoco_area_pressure import (
    area_collision_ownership,
    compile_area_native_rows,
)
from phymeo.backends.mujoco_tyx._constraint_device import TyxConstraintDevice
from phymeo.training.manipulation_tasks import manipulation_scenes

from .pusht_entry_area import MU, SLIP_M_S, _quadrature, area_response, build_area
from .pusht_entry_rows import LABELS, assemble_entry, load_entries
from .pusht_entry_working_set import _reference
from .pusht_saved_contact import GRADIENT_PA_M, PushGeometry

CONTACT_CAPACITY = 64
ROW_CAPACITY = 512
MAX_TRIALS = 24
POINT_MATCH_M = 2e-5
JACOBIAN_MATCH = 5e-5
LIMITS = dict(
    coefficient_velocity=5e-5,
    coefficient_row_force=0.002,
    coefficient_area_force=2e-5,
    original_velocity=5e-5,
    original_row_force=0.002,
    original_area_force=2e-5,
    original_joint_torque=0.002,
    normalized_kkt=2e-5,
    normalized_momentum=2e-5,
    normalized_physical_mass=2e-5,
)


def _maxabs(value):
    return float(np.max(np.abs(value), initial=0.0))


def _plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    return value


def _download(device, names):
    with device.tx.use_runtime(device.runtime):
        device.runtime.sync()
        device.counts["synchronizations"] += 1
        return {name: device._get(getattr(device, name)) for name in names}


def _banks(device):
    with device.tx.use_runtime(device.runtime):
        device.runtime.sync()
        device.counts["synchronizations"] += 1
        return {
            f"{name}.{bank}": device._get(getattr(device, name)[bank])
            for name in ("q", "v", "time", *device.entry)
            for bank in (0, 1)
        }


def _require_unchanged_banks(device, before, active, *, allow_inactive_candidate=False):
    after = _banks(device)
    allowed = (
        {f"{name}.{1 - active}" for name in ("q", "v", "time")}
        if allow_inactive_candidate
        else set()
    )
    changed = [
        name
        for name in before
        if name not in allowed
        and (
            before[name].dtype != after[name].dtype
            or before[name].shape != after[name].shape
            or before[name].tobytes() != after[name].tobytes()
        )
    ]
    if device.active != active or changed:
        raise ArithmeticError(f"unpublished trial changed protected state/entry banks: {changed}")
    return after


def _audit_candidate(
    model, entry, result, world, packed_q, before, after, active, h, area, source, response
):
    """Independent CPU integration is an oracle only; it never writes GPU state."""
    new = 1 - active
    gpu_q = after[f"q.{new}"][world, : model.nq].astype(float)
    gpu_v = after[f"v.{new}"][world, : model.nv].astype(float)
    gpu_clock = after[f"time.{new}"][world].copy()
    if not all(np.isfinite(x).all() for x in (gpu_q, gpu_v, gpu_clock)):
        raise ArithmeticError("nonfinite GPU integration candidate")
    free = model.joint("object-free").id
    qa = int(model.jnt_qposadr[free]) + 3
    gpu_quat = gpu_q[qa : qa + 4]
    quat_norm = float(np.linalg.norm(gpu_quat))
    if quat_norm <= 0:
        raise ArithmeticError("zero GPU free-joint candidate quaternion")
    references, errors, limits = {}, {}, {}
    for name, reference_v, input_q, dt in (
        ("closed", result["velocity"], packed_q, h),
        ("coefficient_fp64", result["coefficient_reference_velocity"], packed_q, h),
        (
            "original_fp64",
            result["original_reference_velocity"],
            entry["q"],
            float(model.opt.timestep),
        ),
    ):
        qref = np.asarray(input_q, dtype=float).copy()
        vref = np.asarray(reference_v, dtype=float)
        mj.mj_integratePos(model, qref, vref, dt)
        mj.mj_normalizeQuat(model, qref)
        reference_quat = qref[qa : qa + 4]
        reference_quat = reference_quat / np.linalg.norm(reference_quat)
        unit = gpu_quat / quat_norm
        if np.dot(unit, reference_quat) < 0:
            unit = -unit
        angle = float(4 * np.arcsin(min(1.0, np.linalg.norm(unit - reference_quat) / 2)))
        references[name] = dict(q=qref, v=vref, h_s=dt)
        errors[name + "_velocity"] = _maxabs(gpu_v - vref)
        errors[name + "_configuration"] = _maxabs(gpu_q - qref)
        errors[name + "_free_quaternion_angle_rad"] = angle
        limits[name + "_velocity"] = 5e-5
        limits[name + "_configuration"] = 1e-6
        limits[name + "_free_quaternion_angle_rad"] = 2e-6
    # Re-evaluate the contact law at the actual FP32 velocity written by the integrator.
    observed_qforce, observed_point_forces = area_response(area, gpu_v, h)
    uploaded = result["uploaded_area_generalized_force"]
    scale = 1 + np.maximum(abs(uploaded), abs(observed_qforce))
    errors["integrated_area_load_scaled"] = _maxabs((observed_qforce - uploaded) / scale)
    limits["integrated_area_load_scaled"] = 1e-6
    errors["integrated_point_force_n"] = _maxabs(
        observed_point_forces - result["area_point_forces"]
    )
    limits["integrated_point_force_n"] = LIMITS["coefficient_area_force"]
    v0 = before[f"v.{active}"][world, : model.nv].astype(float)
    base = sum(
        sign * source[field][world, : model.nv].astype(float)
        for field, sign in (("applied", 1), ("passive", 1), ("bias", -1))
    )
    impulse = h * (base + result["native_generalized_force"] + observed_qforce)
    mass = source["mass"][world, : model.nv, : model.nv].astype(float)
    errors["integrated_momentum_scaled"] = _maxabs(response @ (gpu_v - v0) - impulse) / (
        1 + _maxabs(mass @ v0 + h * base)
    )
    limits["integrated_momentum_scaled"] = LIMITS["normalized_momentum"]
    errors["free_quaternion_norm"] = abs(quat_norm - 1)
    limits["free_quaternion_norm"] = 2e-6
    old_clock = before[f"time.{active}"][world]
    increment = np.float32(np.float32(h) - old_clock[1])
    total = np.float32(old_clock[0] + increment)
    compensation = np.float32(np.float32(total - old_clock[0]) - increment)
    expected_clock = np.array([total, compensation], np.float32)
    clock_bits_match = (
        gpu_clock.dtype == expected_clock.dtype and gpu_clock.tobytes() == expected_clock.tobytes()
    )
    effective = float(gpu_clock[0]) - float(gpu_clock[1])
    errors["clock_closed_s"] = abs(effective - (float(old_clock[0]) - float(old_clock[1]) + h))
    errors["clock_original_fp64_s"] = abs(
        effective - (float(entry["time"]) + float(model.opt.timestep))
    )
    limits["clock_closed_s"] = limits["clock_original_fp64_s"] = 1e-9
    failures = [
        "candidate_" + name
        for name, value in errors.items()
        if not np.isfinite(value) or value > limits[name]
    ]
    if not clock_bits_match:
        failures.append("candidate_compensated_clock_bits")
    return dict(
        status="PASS" if not failures else "FAILED",
        failures=failures,
        q=gpu_q,
        v=gpu_v,
        clock_words=gpu_clock,
        effective_time_s=effective,
        expected_clock_words=expected_clock,
        compensated_clock_bits_match=clock_bits_match,
        free_quaternion=gpu_quat,
        references=references,
        errors=errors,
        limits=limits,
        area_point_forces_at_integrated_velocity=observed_point_forces,
        generalized_area_force_at_integrated_velocity=observed_qforce,
        inactive_bank=new,
        active_bank=active,
        published=False,
    )


def _area_from_device(model, source, world):
    """Reuse geometric quadrature on actual device FK, with device point J.

    No MjData or mj_jac enters this producer. Geometry uses HOST float64 on
    downloaded FP32 poses. The independent CPU producer uses its own FK/J.
    """
    scene = SimpleNamespace(object_geom_names=("object-0", "object-1"), support_geom_name="support")
    geometry = PushGeometry(model, scene)
    body_ids = (int(geometry.body), int(geometry.tool))
    motion = source["motion"][world].astype(float)
    rotation, tool_rotation = source["rotation"][world, list(body_ids)].astype(float)
    origin, tool_origin = motion[list(body_ids), 0]
    ends = (geometry.local_ends @ tool_rotation.T + tool_origin - origin) @ rotation
    cells, metadata = _quadrature(ends, geometry.boxes, float(geometry.radius))
    n = len(cells)
    local = np.asarray([c["center"] for c in cells], dtype=float).reshape(n, 3)
    normals = -np.asarray([c["normal"] for c in cells], dtype=float).reshape(n, 3) @ rotation.T
    if n:
        normals /= np.linalg.norm(normals, axis=1)[:, None]
    points = local @ rotation.T + origin
    jacobians = []
    for body in body_ids:
        body_j = source["jacobian"][world, body, :, : model.nv].astype(float)
        offset = points - motion[body, 5]
        point_j = body_j[:3][None] + np.cross(body_j[3:].T[None], offset[:, None]).transpose(
            0, 2, 1
        )
        jacobians.append(point_j)
    weights = np.asarray([c["weight"] for c in cells], dtype=float)
    return dict(
        points=points,
        normals=normals,
        weights=weights,
        stiffness=GRADIENT_PA_M * weights,
        initial_gap=np.asarray([c["gap"] for c in cells], dtype=float),
        jac_a=jacobians[0],
        jac_b=jacobians[1],
        jac=jacobians[0] - jacobians[1],
        body_ids=body_ids,
        face_ids=np.asarray([c["face"] for c in cells], dtype=int),
        mu=MU,
        slip=SLIP_M_S,
        gradient_pa_m=GRADIENT_PA_M,
        quadrature=metadata,
    )


def _row_owners(model, plan, source, world):
    count = int(source["constraint_counts"][world, 2])
    owners = []
    for index in range(count):
        kind, identity, edge, slot, dof = map(int, source["efc_ids"][world, index])
        if kind == 3:
            owners.append(
                dict(
                    kind="limit",
                    device_row=index,
                    joint_id=identity,
                    joint_name=model.joint(identity).name,
                    dof=dof,
                    side=edge,
                )
            )
        elif kind == 6:
            pair, feature, ga, gb = map(int, plan.slot_ids[slot])
            position = source["normal_rows"][world, slot, 9:12].astype(float)
            owners.append(
                dict(
                    kind="contact",
                    device_row=index,
                    contact_index=identity,
                    geometry_slot=slot,
                    pair=pair,
                    feature=feature,
                    edge=edge,
                    geometry_ids=(ga, gb),
                    geometry_names=(model.geom(ga).name, model.geom(gb).name),
                    position=position,
                )
            )
        else:
            raise ValueError(f"RR experiment requires pyramidal/limit rows, got {kind}")
    return owners


def _match_rows(owners, jacobian, original):
    """One-to-one geometric matching; compact contact ids are never equated.

    At each matching point, compare J to identify the pyramid edge rather than
    assuming MuJoCo and T4 chose equal tangent-basis order. Missing/extra rows
    remain explicit, and a large J discrepancy does not count as a match.
    """
    unused = set(range(len(original["owners"])))
    aligned, differences = [], []
    for index, owner in enumerate(owners):
        choices = []
        for candidate in sorted(unused):
            other = original["owners"][candidate]
            if owner["kind"] != other["kind"]:
                continue
            distance = 0.0
            if owner["kind"] == "limit":
                if any(owner[k] != other[k] for k in ("joint_id", "side", "dof")):
                    continue
            else:
                if set(owner["geometry_ids"]) != set(other["geometry_ids"]):
                    continue
                point = original["contacts"][other["contact_index"]]["position"]
                distance = float(np.linalg.norm(owner["position"] - point))
                if distance > POINT_MATCH_M:
                    continue
            j_error = _maxabs(jacobian[index] - original["jacobian"][candidate])
            choices.append((j_error, distance, candidate))
        if not choices:
            differences.append(
                dict(device_row=index, reason="no original row at same geometry/point")
            )
            continue
        j_error, distance, candidate = min(choices)
        if j_error > JACOBIAN_MATCH:
            differences.append(
                dict(
                    device_row=index,
                    original_row=candidate,
                    reason="pyramid/limit Jacobian differs",
                    point_distance_m=distance,
                    jacobian_error=j_error,
                )
            )
            continue
        unused.remove(candidate)
        aligned.append(
            dict(
                device_row=index,
                original_row=candidate,
                point_distance_m=distance,
                jacobian_error=j_error,
            )
        )
    return dict(
        complete=not differences and not unused,
        matches=aligned,
        differences=differences,
        unmatched_original_rows=sorted(unused),
        point_tolerance_m=POINT_MATCH_M,
        jacobian_tolerance=JACOBIAN_MATCH,
    )


def _match_points(area, original, force, reference_force):
    unused = set(range(len(original["points"])))
    pairs, differences = [], []
    for i, point in enumerate(area["points"]):
        choices = [
            (float(np.linalg.norm(point - original["points"][j])), j)
            for j in sorted(unused)
            if area["face_ids"][i] == original["face_ids"][j]
        ]
        if not choices:
            differences.append(
                dict(device_point=i, reason="no original point on same exterior face")
            )
            continue
        distance, j = min(choices)
        area_error = abs(float(area["weights"][i] - original["weights"][j]))
        if distance > POINT_MATCH_M or area_error > 1e-10 + 1e-3 * original["weights"][j]:
            differences.append(
                dict(
                    device_point=i,
                    nearest_original_point=j,
                    point_distance_m=distance,
                    area_difference_m2=area_error,
                    reason="quadrature discretization differs",
                )
            )
            continue
        unused.remove(j)
        pairs.append(
            dict(
                device_point=i,
                original_point=j,
                point_distance_m=distance,
                area_difference_m2=area_error,
                force_error_n=_maxabs(force[i] - reference_force[j]),
            )
        )
    return dict(
        complete=not differences and not unused,
        matches=pairs,
        differences=differences,
        unmatched_original_points=sorted(unused),
        maximum_matched_force_error_n=max((x["force_error_n"] for x in pairs), default=0.0),
    )


def _audit_world(
    model,
    plan,
    entry,
    world,
    source,
    final,
    area,
    original,
    original_area,
    v0,
    response,
    h,
    velocity,
    final_q,
    point_force,
    packed_q,
    history,
):
    count = int(source["constraint_counts"][world, 2])
    mass = source["mass"][world, : model.nv, : model.nv].astype(float)

    # Add/subtract in FP64 from the individually packed fields, preserving the
    # independent physical-M audit instead of hiding cancellation in FP32.
    base = (
        source["applied"][world].astype(float)
        + source["passive"][world].astype(float)
        - source["bias"][world].astype(float)
    )[: model.nv]
    jac = source["efc_jacobian"][world, :count, : model.nv].astype(float)
    coeff = source["efc_data"][world, :count].astype(float)
    loads = final["efc_data"][world, :count, 9].astype(float)
    acc = final["physical_acc"][world, : model.nv].astype(float)
    rows = dict(
        mass=mass,
        base_load=base,
        jacobian=jac,
        regularization=coeff[:, 4],
        reference_acceleration=coeff[:, 3],
    )
    expected_v, expected_rows, expected_force = _reference(rows, area, v0, response, h)
    original_response = original["mass"] + float(model.opt.timestep) * np.diag(model.dof_damping)
    ref_v, ref_rows, ref_force = _reference(
        original,
        original_area,
        np.asarray(entry["v"]),
        original_response,
        float(model.opt.timestep),
    )
    owners = _row_owners(model, plan, source, world)
    alignment = _match_rows(owners, jac, original)
    matched_force_error = max(
        (
            abs(float(loads[p["device_row"]] - ref_rows[p["original_row"]]))
            for p in alignment["matches"]
        ),
        default=0.0,
    )
    points = _match_points(area, original_area, point_force, ref_force)
    native_q = jac.T @ loads
    original_q = original["jacobian"].T @ ref_rows + np.einsum(
        "sij,si->j", original_area["jac"], ref_force
    )
    gradient = jac @ acc + coeff[:, 4] * loads - coeff[:, 3]
    kkt = np.where(loads > 0, abs(gradient), np.maximum(0, -gradient))
    kkt /= np.maximum(1.0, abs(coeff[:, 4] * loads))
    momentum = response @ (velocity - v0) - h * (base + native_q + final_q)
    physical_residual = mass @ acc - (base + native_q + packed_q)
    object_q = np.einsum("sij,si->j", area["jac_a"], point_force)
    tool_q = np.einsum("sij,si->j", area["jac_b"], -point_force)
    errors = dict(
        coefficient_velocity=_maxabs(velocity - expected_v),
        coefficient_row_force=_maxabs(loads - expected_rows),
        coefficient_area_force=_maxabs(point_force - expected_force),
        original_velocity=_maxabs(velocity - ref_v),
        original_row_force=float(matched_force_error),
        original_area_force=_maxabs(point_force.sum(axis=0) - ref_force.sum(axis=0)),
        original_joint_torque=_maxabs((native_q + final_q - original_q)[:2]),
        normalized_kkt=_maxabs(kkt),
        normalized_momentum=_maxabs(momentum) / (1 + _maxabs(mass @ v0 + h * base)),
        normalized_physical_mass=_maxabs(physical_residual)
        / (1 + _maxabs(base + native_q + packed_q)),
    )
    failures = [name for name, value in errors.items() if value > LIMITS[name]]
    if not alignment["complete"]:
        failures.append("original_native_row_correspondence")
    if not points["complete"]:
        failures.append("original_area_point_correspondence")
    if points["maximum_matched_force_error_n"] > LIMITS["original_area_force"]:
        failures.append("original_per_point_area_force")
    if np.any(loads < 0) or not np.all(np.isfinite(velocity)):
        failures.append("force_or_velocity_domain")
    if _maxabs(object_q + tool_q - final_q) > 1e-10:
        failures.append("bilateral_generalized_sum")
    return dict(
        label=entry["label"],
        source_tick=entry["tick"],
        source_time=entry["time"],
        status="PASS" if not failures else "FAILED_REFERENCE",
        failures=failures,
        native_rows=count,
        original_native_rows=len(ref_rows),
        native_candidates=plan.npair,
        native_slots=plan.nslot,
        area_points=len(point_force),
        original_area_points=len(ref_force),
        velocity=velocity,
        original_reference_velocity=ref_v,
        coefficient_reference_velocity=expected_v,
        physical_acceleration=acc,
        native_row_forces=loads,
        original_reference_row_forces=ref_rows,
        native_owners=owners,
        original_native_owners=original["owners"],
        original_contacts=original["contacts"],
        row_alignment=alignment,
        area_point_alignment=points,
        area_points_world=area["points"],
        area_point_forces=point_force,
        original_area_points_world=original_area["points"],
        original_area_point_forces=ref_force,
        area_normal_n=float(np.sum(point_force * area["normals"])),
        area_force_world_n=point_force.sum(axis=0),
        area_generalized_force=final_q,
        uploaded_area_generalized_force=packed_q,
        native_generalized_force=native_q,
        area_on_object_generalized=object_q,
        area_on_tool_generalized=tool_q,
        total_joint_torque_nm=(native_q + final_q)[:2],
        original_total_joint_torque_nm=original_q[:2],
        errors=errors,
        limits=LIMITS,
        physical_mass_residual=_maxabs(physical_residual),
        mass_source_error=_maxabs(mass - original["mass"]),
        base_load_source_error=_maxabs(base - original["base_load"]),
        trial_history=history,
        solver_diagnostics=final["solver_diagnostics"][world],
        quadrature=area["quadrature"],
        original_quadrature=original_area["quadrature"],
    )


def run_trials(*, cache_root=None):
    """Return five JSON-ready results and one unpublished GPU integration candidate."""
    scene = next(s for s in manipulation_scenes() if s.key == "pusht-planar")
    model = mj.MjModel.from_xml_string(scene.mjcf)
    entries = load_entries()
    if (model.nq, model.nv, model.nu, model.nbody) != (9, 8, 2, 5):
        raise ValueError("requires the complete original RR model")
    plan = compile_area_native_rows(mj, model, (("tool", "object"),))
    ownership = area_collision_ownership(mj, model, (("tool", "object"),))
    if len(ownership.replaced_geometry_pairs[0]) != 2 or len(ownership.native_geometry_pairs) != 9:
        raise ValueError("original two replaced / nine retained candidate binding changed")
    if tuple(map(tuple, plan.pair_ids[: plan.npair, :2])) != ownership.native_geometry_pairs:
        raise ValueError("native trial compiler changed retained pair identity/order")
    if plan.nlimit != 2 or plan.nv != 8 or plan.nq != 9:
        raise ValueError("native trial compiler dropped original joint/free-body coordinates")
    originals = [assemble_entry(model, entries[label]) for label in LABELS]
    original_areas = [build_area(model, rows["data"]) for rows in originals]
    nworld, width = len(LABELS), max(8, plan.nv)
    q = np.zeros((nworld, max(8, plan.nq)), np.float32)
    v = np.zeros((nworld, width), np.float32)
    clock = np.zeros((nworld, 2), np.float32)
    controls = np.zeros((nworld, max(1, plan.nu)), np.float32)
    applied = np.zeros((nworld, width), np.float32)
    body_force = np.zeros((nworld, model.nbody, 6), np.float32)
    for i, label in enumerate(LABELS):
        entry = entries[label]
        q[i, : plan.nq], v[i, : plan.nv] = entry["q"], entry["v"]
        clock[i, 0] = entry["time"]
        clock[i, 1] = float(clock[i, 0]) - entry["time"]
        controls[i], applied[i], body_force[i] = (
            entry["ctrl"],
            entry["qfrc_applied"],
            entry["xfrc_applied"],
        )
    h = float(np.float32(model.opt.timestep))
    device = TyxConstraintDevice(
        plan,
        "cuda",
        cache_root,
        q,
        v,
        clock,
        model.nbody,
        h,
        (CONTACT_CAPACITY, ROW_CAPACITY),
    )
    extra = None
    transfers = dict(extra_device_allocations=0, extra_uploaded_bytes=0, extra_uploads=0)
    try:
        before = _banks(device)
        active = device.active
        with device.tx.use_runtime(device.runtime):
            extra = device.tx.cuda_array(np.zeros((nworld, width), np.float32))
            transfers["extra_device_allocations"] = 1
            transfers["extra_uploaded_bytes"] += nworld * width * 4
        with device.prepare_native_rows(controls, applied, body_force) as evaluate:
            evaluate(None)
            source = _download(
                device,
                (
                    "mass",
                    "vectors",
                    "applied",
                    "passive",
                    "bias",
                    "motion",
                    "rotation",
                    "jacobian",
                    "efc_ids",
                    "efc_data",
                    "efc_jacobian",
                    "normal_rows",
                    "contact_map",
                    "constraint_counts",
                    "status",
                    "physical_acc",
                    "solver_diagnostics",
                ),
            )
            if np.any(source["status"]) or np.any(source["solver_diagnostics"][:, 6] != 1):
                raise ArithmeticError(f"native baseline rejected: {source['status'].tolist()}")
            _require_unchanged_banks(device, before, active)
            areas = [_area_from_device(model, source, i) for i in range(nworld)]
            mass = source["mass"][:, : model.nv, : model.nv].astype(float)
            damping = source["vectors"][1, : model.nv].astype(float)
            response = mass + h * np.diag(damping)[None]
            v0 = v[:, : model.nv].astype(float)
            velocity = (
                v0
                + h
                * np.linalg.solve(
                    response,
                    np.einsum("wij,wj->wi", mass, source["physical_acc"][:, : model.nv])[..., None],
                )[..., 0]
            )
            histories = [[] for _ in LABELS]
            converged = np.zeros(nworld, dtype=bool)
            final_q, final_force = None, None
            packed = np.zeros((nworld, width), np.float32)
            for iteration in range(1, MAX_TRIALS + 1):
                for i, area in enumerate(areas):
                    packed[i, : model.nv] = area_response(area, velocity[i], h)[0]
                if not np.isfinite(packed).all():
                    raise ArithmeticError("nonfinite FP32 candidate area load")
                with device.tx.use_runtime(device.runtime):
                    extra.copy_from(packed)
                    transfers["extra_uploaded_bytes"] += packed.nbytes
                    transfers["extra_uploads"] += 1
                    evaluate(extra)
                final = _download(
                    device, ("physical_acc", "efc_data", "status", "solver_diagnostics")
                )
                if np.any(final["status"]) or np.any(final["solver_diagnostics"][:, 6] != 1):
                    raise ArithmeticError(
                        f"native trial {iteration} rejected: {final['status'].tolist()}"
                    )
                _require_unchanged_banks(device, before, active)
                candidate = (
                    v0
                    + h
                    * np.linalg.solve(
                        response,
                        np.einsum("wij,wj->wi", mass, final["physical_acc"][:, : model.nv])[
                            ..., None
                        ],
                    )[..., 0]
                )
                evaluated = [area_response(a, candidate[i], h) for i, a in enumerate(areas)]
                final_q = np.stack([a[0] for a in evaluated])
                final_force = [a[1] for a in evaluated]
                delta = final_q - packed[:, : model.nv].astype(float)
                defect = h * np.linalg.solve(response, delta[..., None])[..., 0]
                converged = np.all(
                    abs(delta) <= 1e-6 * (1 + np.maximum(abs(final_q), abs(packed))), axis=1
                )
                converged &= np.all(
                    abs(defect) <= 1e-7 * (1 + np.maximum(abs(v0), abs(candidate))), axis=1
                )
                for i in range(nworld):
                    histories[i].append(
                        dict(
                            trial=iteration,
                            load_defect=_maxabs(delta[i]),
                            velocity_defect=_maxabs(defect[i]),
                            closed=bool(converged[i]),
                            solver_diagnostics=final["solver_diagnostics"][i].copy(),
                        )
                    )
                velocity = candidate
                if np.all(converged):
                    break
            reports = []
            for i, label in enumerate(LABELS):
                result = _audit_world(
                    model,
                    plan,
                    entries[label],
                    i,
                    source,
                    final,
                    areas[i],
                    originals[i],
                    original_areas[i],
                    v0[i],
                    response[i],
                    h,
                    velocity[i],
                    final_q[i],
                    final_force[i],
                    packed[i, : model.nv].astype(float),
                    histories[i],
                )
                if not converged[i]:
                    result["status"] = "FAILED_CLOSURE"
                    result["failures"].append("area_load_iteration_budget")
                reports.append(result)
            # The load buffer still contains the exact final evaluated FP32 Q.
            # Every preceding trial left both physical banks and ALL entry banks
            # unchanged. Only this final invocation may write inactive q/v/clock.
            _require_unchanged_banks(device, before, active)
            candidate_integrations = 0
            if np.all(converged):
                with device.tx.use_runtime(device.runtime):
                    evaluate(extra, integrate=True)
                candidate_integrations = 1
                integrated = _download(
                    device, ("physical_acc", "efc_data", "status", "solver_diagnostics")
                )
                if np.any(integrated["status"]) or np.any(
                    integrated["solver_diagnostics"][:, 6] != 1
                ):
                    raise ArithmeticError(
                        f"GPU candidate integration rejected: {integrated['status'].tolist()}"
                    )
                for name in ("physical_acc", "efc_data"):
                    if (
                        integrated[name].dtype != final[name].dtype
                        or integrated[name].tobytes() != final[name].tobytes()
                    ):
                        raise ArithmeticError(
                            "final integration trial changed its converged native solution: " + name
                        )
                after_candidate = _require_unchanged_banks(
                    device, before, active, allow_inactive_candidate=True
                )
                for i, label in enumerate(LABELS):
                    checked = _audit_candidate(
                        model,
                        entries[label],
                        reports[i],
                        i,
                        q[i, : model.nq].astype(float),
                        before,
                        after_candidate,
                        active,
                        h,
                        areas[i],
                        source,
                        response[i],
                    )
                    reports[i]["integration_candidate"] = checked
                    if checked["failures"]:
                        reports[i]["failures"].extend(checked["failures"])
                        reports[i]["status"] = "FAILED_CANDIDATE"
        _require_unchanged_banks(
            device, before, active, allow_inactive_candidate=bool(candidate_integrations)
        )
        report = dict(
            schema="phymeo-pusht-resident-native-trials-v3",
            status="PASS" if all(r["status"] == "PASS" for r in reports) else "FAILED",
            scene=scene.key,
            source_sha256=scene.source_sha256,
            labels=LABELS,
            world_count=nworld,
            timestep_s=h,
            native_contact_capacity=CONTACT_CAPACITY,
            native_row_capacity=ROW_CAPACITY,
            replaced_geometry_pairs=ownership.replaced_geometry_pairs,
            retained_geometry_pairs=ownership.native_geometry_pairs,
            geometry_names=[model.geom(i).name for i in range(model.ngeom)],
            trial_physical_banks_unchanged=True,
            active_and_all_entry_banks_unchanged=True,
            candidate_integrations=candidate_integrations,
            published_steps=0,
            entry_bank_fields=tuple(device.entry),
            numerical_scope="resident T4 assembly/working-set; HOST quadrature, load iteration and FP64 audits",
            independent_reference="original CPU MuJoCo rows, M/J and area geometry; independent FP64 coupled root",
            device_transport_counters=dict(device.counts),
            experiment_transfers=transfers,
            results=reports,
        )
        return _plain(report)
    finally:
        # The separately owned candidate-load buffer must be gone before its
        # borrowed device runtime. If freeing it raises, do not close that runtime.
        if extra is not None:
            with device.tx.use_runtime(device.runtime):
                extra.free()
        device.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path)
    args = parser.parse_args()
    report = run_trials(cache_root=args.cache_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            dict(
                status=report["status"],
                output=str(args.output),
                worlds=[
                    dict(label=r["label"], status=r["status"], failures=r["failures"])
                    for r in report["results"]
                ],
            ),
            allow_nan=False,
        )
    )
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
