"""Run a detached-sheet and a gravity-preserving rotated twin of a stored physical handoff frame."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from benchmark_planar_handoff import physical, read

import phymeo as ph
from phymeo.dynamics.robot_contact import RobotContactSystem
from phymeo.experiments.compliant_handoff import build_handoff


def rotated_program(case, rotation):
    source = case.program
    models = {}
    for name, grip in case.grippers.items():
        model = grip.model
        position = model.joint_position.copy()
        orientation = model.joint_orientation.copy()
        roots = np.flatnonzero(model.parents == -1)
        np.testing.assert_array_equal(orientation[roots], np.tile([0, 0, 0, 1], (len(roots), 1)))
        position[roots] = position[roots] @ rotation.T
        orientation[roots] = [0, 0, 1, 0]
        models[name] = replace(model, joint_position=position, joint_orientation=orientation)
    scene = ph.SceneModel(
        (
            ph.DomainBinding("cloth", "particles", source.cloth),
            *(ph.DomainBinding(n, "robot", m) for n, m in models.items()),
        )
    )
    body = source.robot_contact
    collision = RobotContactSystem(
        models,
        limits=body.limits,
        capsules=body.capsules,
        excluded_pairs=body.excluded_pairs,
        planes=tuple(replace(plane, normal=rotation @ plane.normal) for plane in body.planes),
        material=body.material,
    )
    pads = [
        p
        if p.robot_domain is not None
        else replace(
            p, geometry=replace(p.geometry, local_vertices=p.geometry.local_vertices @ rotation.T)
        )
        for p in source.patches
    ]
    return ph.CompliantClothContactProgram(
        scene,
        "cloth",
        source.material,
        source.surfaces,
        case.initial.domains["cloth"].position @ rotation.T,
        pads,
        material_damping_s=source.damping,
        max_penetration_m=source.max_penetration_m,
        robot_contact=collision,
        max_joint_violation=source.max_joint_violation,
        cloth_link_material=source.cloth_links.law,
        cloth_self_layer=dict(source.cloth_self.calibrations),
    )


def check(source, observed, output):
    src = read(source)
    trace = read(observed)
    pad = next(p for p in src["patches"] if p["name"] == "B_left")
    case = build_handoff(
        robot=src["robot_layout"],
        fabric=src["fabric"]["name"],
        grid=src["fabric"]["grid"],
        initial_tilt_deg=src["initial_tilt_deg"],
        finger_pad_radius_m=pad["radius_m"],
        finger_backing_clearance_m=pad["backing_clearance_m"],
        self_contact=True,
    )
    before, end = trace["frames"][:2]
    expected = physical(end)
    limits = trace["origin"]["force_limits"]
    drives = {}
    for n, g in case.grippers.items():
        drives[n] = ph.RobotInputs.zeros(g.model.dof_count)
        drives[n].effort[:] = end["actuator_effort"][n]
        assert np.all(np.abs(drives[n].effort) <= limits[n])
    inputs = ph.SceneInputs({"cloth": case.inputs.domains["cloth"], **drives})
    rotation = np.diag([-1.0, -1.0, 1.0])
    result = {}
    for mode in ("mirror", "empty"):
        state = physical(before)
        cloth = state.domains["cloth"]
        if mode == "mirror":
            cloth.position[:] = cloth.position @ rotation.T
            cloth.velocity[:] = cloth.velocity @ rotation.T
            program = rotated_program(case, rotation)
        else:
            cloth.position[:] += [0, 1, 1]
            program = case.program
        with program.stepper(
            ph.StepSettings(substeps=case.substeps, position_iterations=1, velocity_iterations=1)
        ) as runtime:
            out = runtime.step(state, inputs, case.dt)
        assert out.report.accepted, out.report.reason
        stepped = out.state.domains["cloth"]
        observed_peak = err_force = 0.0
        names = [p["name"] for p in trace["patches"]]
        for i, rep in enumerate(out.report.substeps):
            body = rep.diagnostics["robot_contact"]
            assert rep.diagnostics["self_contact"]["registered_pairs"] == 276
            group = trace["samples"][i * len(names) : (i + 1) * len(names)]
            for old in group:
                new = np.asarray(rep.diagnostics["patches"][old["patch"]]["force_n"]).sum(axis=0)
                observed_peak = max(observed_peak, float(np.linalg.norm(new)))
                if mode == "mirror":
                    err_force = max(
                        err_force, float(np.max(np.abs(new - rotation @ old["force_n"])))
                    )
                else:
                    np.testing.assert_array_equal(new, 0)
                    assert not body["cloth_links"]
        if mode == "mirror":
            ex = float(
                np.max(abs(stepped.position - expected.domains["cloth"].position @ rotation.T))
            )
            ev = float(
                np.max(abs(stepped.velocity - expected.domains["cloth"].velocity @ rotation.T))
            )
            eq = max(
                float(np.max(abs(out.state.domains[n].q - expected.domains[n].q))) for n in drives
            )
            assert ex < 5e-8 and ev < 1e-5 and eq < 5e-8 and err_force < 1e-4, (
                ex,
                ev,
                eq,
                err_force,
            )
            result[mode] = dict(
                max_node_x_m=ex,
                max_node_v_m_s=ev,
                max_joint_q=eq,
                max_patch_force_n=err_force,
                actual_peak_force_n=observed_peak,
            )
        else:
            dv = (stepped.velocity - cloth.velocity).mean(0)
            err = float(np.max(abs(dv - case.dt * np.array([0.0, 0.0, -9.81]))))
            assert err < 1e-10 and observed_peak == 0
            result[mode] = dict(peak_surface_force_n=observed_peak, center_gravity_dv_error_m_s=err)
    result["dt_s"] = case.dt
    result["substeps_per_case"] = case.substeps
    result["source_t"] = before["t"]
    output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf8")
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("observed", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    check(args.source, args.observed, args.output)
