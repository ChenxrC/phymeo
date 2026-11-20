"""Eight-DOF RR entry experiment with a CUDA native working set and HOST area coupling."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import mujoco as mj
import numpy as np

from phymeo.backends.mujoco_area_pressure import area_collision_ownership
from phymeo.training.manipulation_tasks import manipulation_scenes

from .native_row_device import NativeRowDevice
from .pusht_entry_area import area_response, build_area
from .pusht_entry_rows import assemble_entry, load_entries


def _reference(rows, area, v0, response, h):
    """Independent FP64 root in physical acceleration, with no device-force seed."""
    mass, base = rows["mass"], rows["base_load"]
    j, reg, aref = (rows[k] for k in ("jacobian", "regularization", "reference_acceleration"))
    point_j = area["jac"]
    normal = area["normals"]
    dv_da = h * np.linalg.solve(response, mass)
    relative_j = np.einsum("sij,jk->sik", point_j, dv_da)
    vn_j = np.einsum("si,sij->sj", normal, relative_j)
    tangent_j = relative_j - normal[:, :, None] * vn_j[:, None, :]

    def evaluate(a):
        velocity = v0 + dv_da @ a
        relative = point_j @ velocity
        vn = np.sum(relative * normal, axis=1)
        vt = relative - normal * vn[:, None]
        speed = np.sqrt(np.sum(vt**2, axis=1) + 0.01**2)
        pressure_load = np.maximum(0, -area["stiffness"] * (area["initial_gap"] + h * vn))
        direction = normal - 0.7 * vt / speed[:, None]
        force = direction * pressure_load[:, None]
        row_force = np.maximum(0, (aref - j @ a) / reg)
        residual = mass @ a - base - j.T @ row_force - np.einsum("sij,si->j", point_j, force)
        return residual, velocity, row_force, pressure_load, direction, vt, speed, force

    acceleration = np.linalg.solve(mass, base)
    for _ in range(80):
        residual, v, loads, pressure, direction, vt, speed, force = evaluate(acceleration)
        norm = np.linalg.norm(residual)
        if norm <= 1e-10 * (1 + np.linalg.norm(base)):
            return v, loads, force
        active = loads > 0
        derivative = mass + j[active].T @ (j[active] / reg[active, None])
        dfn = -h * (area["stiffness"] * (pressure > 0))[:, None] * vn_j
        tangential = (
            tangent_j / speed[:, None, None]
            - vt[:, :, None]
            * (np.einsum("si,sij->sj", vt, tangent_j) / speed[:, None] ** 3)[:, None, :]
        )
        df = direction[:, :, None] * dfn[:, None, :] - (0.7 * pressure)[:, None, None] * tangential
        derivative -= np.einsum("sij,sik->jk", point_j, df)
        delta = np.linalg.solve(derivative, -residual)
        for power in range(20):
            candidate = acceleration + delta / (2**power)
            if np.linalg.norm(evaluate(candidate)[0]) < norm:
                acceleration = candidate
                break
        else:
            raise ArithmeticError("FP64 coupled reference stalled")
    raise ArithmeticError("FP64 coupled reference exceeded the iteration budget")


def _entry_solution(model, entry):
    start = perf_counter()
    rows = assemble_entry(model, entry)
    data = rows["data"]
    area = build_area(model, data)
    mass, base = rows["mass"], rows["base_load"]
    j, reg, aref = (rows[k] for k in ("jacobian", "regularization", "reference_acceleration"))
    v0 = data.qvel.copy()
    h = float(model.opt.timestep)
    response = mass + h * np.diag(model.dof_damping)
    dv_da = h * np.linalg.solve(response, mass)
    setup_s = perf_counter() - start
    start = perf_counter()
    with NativeRowDevice(mass, j, reg, aref) as device:
        baseline_a, baseline_rows, diag = device.evaluate(base)
        no_area = v0 + dv_da @ baseline_a
        velocity = no_area.copy()
        work = [int(diag["iterations"])]
        for iteration in range(1, 25):
            trial_q, _ = area_response(area, velocity, h)
            if iteration == 1 and not np.any(trial_q):
                a, loads = baseline_a, baseline_rows
            else:
                a, loads, diag = device.evaluate(base + trial_q)
                work.append(int(diag["iterations"]))
            candidate = v0 + dv_da @ a
            final_q, point_force = area_response(area, candidate, h)
            # Compare the final constitutive load to the one actually solved.
            # The FP32 working set can alternate at adjacent input loads even
            # when they imply indistinguishable motion. Test the residual in
            # motion units instead of imposing FP64 equality between trials.
            delta_load = final_q - trial_q
            motion_defect = h * np.linalg.solve(response, delta_load)
            # Reserve at least one order of magnitude below the final 2e-5
            # momentum/KKT and 5e-5 velocity checks. Scales use SI units for
            # each generalized coordinate and are independent of the fixture.
            load_scale = 1.0 + np.maximum(np.abs(trial_q), np.abs(final_q))
            velocity_scale = 1.0 + np.maximum(np.abs(v0), np.abs(candidate))
            if np.all(abs(delta_load) <= 1e-6 * load_scale) and np.all(
                abs(motion_defect) <= 1e-7 * velocity_scale
            ):
                break
            velocity = candidate
        else:
            raise ArithmeticError(
                "area load and integrated velocity did not agree within 24 trials"
            )
    solve_s = perf_counter() - start
    ref_v, ref_rows, ref_point_force = _reference(rows, area, v0, response, h)
    owners = rows["owners"]
    support = [
        i
        for i, row in enumerate(owners)
        if "support" in row.get("geometry_names", ())
        and any(name in ("object-0", "object-1") for name in row["geometry_names"])
    ]
    gradient = j @ a + reg * loads - aref
    kkt = np.where(loads > 0, abs(gradient), np.maximum(0, -gradient)) / np.maximum(
        1, abs(reg * loads)
    )
    constraint_force = j.T @ loads + final_q
    momentum = response @ (candidate - v0) - h * (base + constraint_force)
    area_force = point_force.sum(axis=0)
    object_q = np.einsum("sij,si->j", area["jac_a"], point_force)
    tool_q = np.einsum("sij,si->j", area["jac_b"], -point_force)
    if not np.allclose(object_q + tool_q, final_q, atol=1e-12, rtol=1e-12):
        raise ArithmeticError("bilateral area force does not match the coupled generalized load")
    normal_n = np.sum(point_force * area["normals"])
    q1 = data.qpos.copy()
    mj.mj_integratePos(model, q1, candidate, h)
    point_force_error = np.max(np.abs(point_force - ref_point_force), initial=0.0)
    if point_force_error > 2e-5 or np.any(loads < 0) or not np.isfinite(q1).all():
        raise ArithmeticError("unilateral force, area reference, or integrated pose failed")
    report = dict(
        tick=entry["tick"],
        native_rows=len(loads),
        replaced_native_rows=len(rows["excluded_indices"]),
        native_positive_rows=int(np.count_nonzero(loads)),
        native_limit_rows=sum(row["kind"] == "limit" for row in owners),
        owners=owners,
        native_forces=loads.tolist(),
        working_set_iterations=work,
        outer_iterations=iteration,
        support_normal_n=float(loads[support].sum()),
        area_points=len(point_force),
        area_normal_n=float(normal_n),
        area_force_world_n=area_force.tolist(),
        area_generalized_force=final_q.tolist(),
        quadrature=area["quadrature"],
        area_tool_generalized=tool_q.tolist(),
        area_object_generalized=object_q.tolist(),
        area_hinge_torque_nm=tool_q[:2].tolist(),
        area_moment_on_object_world_nm=np.cross(
            area["points"] - data.xipos[model.body("object").id], point_force
        )
        .sum(axis=0)
        .tolist(),
        no_area_support_normal_n=float(baseline_rows[support].sum()),
        no_area_velocity=no_area.tolist(),
        velocity=candidate.tolist(),
        next_q=q1.tolist(),
        max_velocity_change_vs_no_area=float(np.max(abs(candidate - no_area))),
        max_velocity_error=float(np.max(abs(candidate - ref_v))),
        max_native_row_force_error=float(np.max(abs(loads - ref_rows), initial=0.0)),
        max_area_point_force_error_n=float(point_force_error),
        native_kkt=float(np.max(kkt, initial=0.0)),
        scaled_momentum_residual=float(
            np.linalg.norm(momentum) / (1 + np.linalg.norm(mass @ v0 + h * base))
        ),
        area_load_iteration_error=float(np.max(abs(final_q - trial_q))),
        area_velocity_defect=float(np.max(abs(motion_defect))),
        host_assembly_s=setup_s,
        staged_cuda_coupling_s=solve_s,
        entry_q=data.qpos.tolist(),
        entry_v=v0.tolist(),
        object_com_world=data.xipos[model.body("object").id].tolist(),
    )
    limits = {
        "max_velocity_error": 5e-5,
        "max_native_row_force_error": 0.002,
        "scaled_momentum_residual": 2e-5,
        "native_kkt": 2e-5,
    }
    for name, cap in limits.items():
        if not np.isfinite(report[name]) or report[name] > cap:
            raise ArithmeticError(f"tick {entry['tick']}: {name}={report[name]} exceeds {cap}")
    return report


def run_comparison():
    scene = next(s for s in manipulation_scenes() if s.key == "pusht-planar")
    model = mj.MjModel.from_xml_string(scene.mjcf)
    ownership = area_collision_ownership(mj, model, (("tool", "object"),))
    frames = {name: _entry_solution(model, entry) for name, entry in load_entries().items()}
    return dict(
        schema="phymeo-rr-entry-working-set-v1",
        source_sha256=scene.source_sha256,
        mode="cuda-native-rows-host-area",
        nv=model.nv,
        retained_candidates=len(ownership.native_geometry_pairs),
        frames=frames,
    )


def advance_entries(model, initial, commands, save_step):
    """Advance from one initial q/v. Subsequent command records supply forces only."""
    tick, time = initial["tick"], initial["time"]
    q, v = initial["q"], initial["v"]
    for command in commands:
        if command["tick"] != tick:
            raise ValueError(f"control tick {command['tick']} does not follow {tick}")
        entry = {key: command[key] for key in ("ctrl", "qfrc_applied", "xfrc_applied")}
        entry.update(q=q, v=v, time=time, tick=tick)
        report = _entry_solution(model, entry)
        report["entry_control"] = {
            key: np.asarray(entry[key]).tolist() for key in ("ctrl", "qfrc_applied", "xfrc_applied")
        }
        if "trace_line_sha256" in command:
            report["trace_line_sha256"] = command["trace_line_sha256"]
        save_step(report)
        q, v = report["next_q"], report["velocity"]
        tick += 1
        time += model.opt.timestep
    return dict(tick=tick, time=time, q=q, v=v)


def run_recorded_window(directory, output, steps=None):
    from .pusht_entry_rows import iter_controls

    entries = load_entries()
    initial = entries["first_load"]
    length = entries["withdrawn"]["tick"] - initial["tick"] + 1
    steps = length if steps is None else steps
    if not 1 <= steps <= length:
        raise ValueError(f"expected 1..{length} archived motor commands")
    scene = next(s for s in manipulation_scenes() if s.key == "pusht-planar")
    model = mj.MjModel.from_xml_string(scene.mjcf)
    commands = iter_controls(directory, initial["tick"], initial["tick"] + steps)
    stats = dict(
        requested=steps,
        accepted=0,
        start_tick=initial["tick"],
        scene_sha256=scene.source_sha256,
        mode="cuda-native-rows-host-area-open-loop",
        gradient_pa_m=1e6,
        source_trace_sha256=initial["saved_trace_sha256"],
    )
    trace_path = output.with_suffix(".jsonl")
    if trace_path == output:
        raise ValueError("summary path and JSONL trace must differ")
    with output.open("x", encoding="utf-8") as summary:
        with trace_path.open("x", encoding="utf-8") as trace:

            def record(step):
                trace.write(json.dumps(step, allow_nan=False) + "\n")
                trace.flush()
                stats["accepted"] += 1

            try:
                stats["final_state"] = advance_entries(model, initial, commands, record)
                if stats["accepted"] != steps:
                    raise ValueError("archived control window ended early")
                stats["status"] = "complete"
            except Exception as error:
                stats.update(
                    status="rejected",
                    rejected_tick=initial["tick"] + stats["accepted"],
                    error_type=type(error).__name__,
                    error=str(error),
                )
                raise
            finally:
                json.dump(stats, summary, indent=2, allow_nan=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--trajectory", type=Path, help="original CUDA archive for a continuous control replay"
    )
    parser.add_argument(
        "--steps", type=int, help="prefix length, starting at the first recorded load"
    )
    args = parser.parse_args()
    if args.trajectory is not None:
        run_recorded_window(args.trajectory, args.output, args.steps)
        return
    if args.steps is not None:
        parser.error("--steps requires --trajectory")
    report = run_comparison()
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
