"""Two independently driven hinge/slide arms, with joint stops and dry friction."""

import json
import os
import sys
from xml.etree import ElementTree as ET

import numpy as np
import pytest
import test_mujoco_tyx_constraints as t4
from test_mujoco_tyx_runtime import compile_source

from phymeo.backends.mujoco_resource_budget import MuJoCoTyxResourceBudget
from phymeo.backends.mujoco_runtime import MuJoCoControl, MuJoCoInitialState, MuJoCoRuntime

pytestmark = [
    pytest.mark.musa,
    pytest.mark.skipif(os.environ.get("PHYMEO_TEST_MUSA") != "1", reason="requires a MUSA GPU"),
]

PUSH = 140
RETRACT = 140
# Initial shoulder angle, shoulder motor command (gear 2), extension motor command (gear 1).
WORLDS = {4: (0.165, 0.80, 1.20), 91: (-0.165, -0.68, -0.95)}


def _arm_xml():
    # T4's serial arm retains its joint ranges/margins, damping, armatures,
    # dry-friction bounds, tip COM offset and full inertias. Isolate joint loads.
    root = ET.fromstring(t4.mixed_xml())
    world = root.find("worldbody")
    world.remove(world.find("geom[@name='floor']"))
    root.find("option").set("gravity", "0 0 0")
    motors = ET.SubElement(root, "actuator")
    ET.SubElement(motors, "motor", name="shoulder-drive", joint="turn", gear="2")
    ET.SubElement(motors, "motor", name="extension-drive", joint="slide", gear="1")
    return ET.tostring(root, encoding="unicode")


def _command(key, tick):
    values = WORLDS[key][1:]
    return values if tick <= PUSH else tuple(-v for v in values)


def _native_reference(mj, model, tip):
    traces = {}
    for key, (angle, _, _) in WORLDS.items():
        data = mj.MjData(model)
        data.qpos[:] = [angle, 0.0]
        samples = []
        for tick in range(1, PUSH + RETRACT + 1):
            qin, vin, tin = data.qpos.copy(), data.qvel.copy(), float(data.time)
            data.ctrl[:] = _command(key, tick)
            mj.mj_step(model, data)
            # Native EFC, actuator and body transforms remain at the Euler input;
            # integrated coordinates and velocity are the accepted endpoint.
            jac = data.efc_J.reshape(data.nefc, model.nv)
            rows = []
            for i in range(data.nefc):
                kind, source = int(data.efc_type[i]), int(data.efc_id[i])
                if kind == int(mj.mjtConstraint.mjCNSTR_FRICTION_DOF):
                    dof, side = source, -1
                else:
                    assert kind == int(mj.mjtConstraint.mjCNSTR_LIMIT_JOINT)
                    dof = int(model.jnt_dofadr[source])
                    side = -1 if jac[i, dof] > 0 else 1
                rows.append(
                    dict(
                        kind=kind,
                        source=source,
                        side=side,
                        dof=dof,
                        jac=jac[i].tolist(),
                        force=float(data.efc_force[i]),
                    )
                )
            assert data.ncon == 0
            samples.append(
                dict(
                    tick=tick,
                    input_time=tin,
                    output_time=float(data.time),
                    q_in=qin.tolist(),
                    v_in=vin.tolist(),
                    q=data.qpos.tolist(),
                    v=data.qvel.tolist(),
                    ctrl=data.ctrl.tolist(),
                    qacc=data.qacc.tolist(),
                    integrated_acc=((data.qvel - vin) / model.opt.timestep).tolist(),
                    actuator=data.qfrc_actuator.tolist(),
                    constraint=data.qfrc_constraint.tolist(),
                    tip_in=data.xpos[tip].tolist(),
                    rotation_in=data.xmat[tip].reshape(3, 3).tolist(),
                    rows=rows,
                )
            )
        traces[key] = samples
    return traces


def _sample(state, tip):
    receipt = state.last_step
    assert receipt.backend == "mujoco-tyx-musa" and not receipt.contacts
    assert receipt.tick == state.tick and receipt.output_time == state.time
    assert receipt.contact_phase == "tyx-euler-input-constraint-evaluation"
    assert receipt.contact_time == receipt.dynamics.time == receipt.input_time < state.time
    np.testing.assert_array_equal(receipt.dynamics.input_q, receipt.input_q)
    np.testing.assert_array_equal(receipt.dynamics.input_v, receipt.input_v)
    return dict(
        tick=state.tick,
        input_time=receipt.input_time,
        output_time=state.time,
        q_in=receipt.input_q.tolist(),
        v_in=receipt.input_v.tolist(),
        q=state.q.tolist(),
        v=state.v.tolist(),
        ctrl=receipt.control.ctrl.tolist(),
        qacc=receipt.qacc.tolist(),
        integrated_acc=receipt.integration_acceleration.tolist(),
        actuator=receipt.qfrc_actuator.tolist(),
        constraint=receipt.qfrc_constraint.tolist(),
        tip_in=receipt.dynamics.body_position[tip].tolist(),
        rotation_in=receipt.dynamics.body_rotation[tip].tolist(),
        rows=[
            dict(
                kind=r.constraint_type,
                source=r.source_id,
                side=r.side_or_edge,
                dof=r.dof,
                jac=r.jacobian.tolist(),
                force=r.force,
            )
            for r in receipt.constraints
        ],
    )


def _verify(key, native, actual, model):
    def column(rows, name):
        return np.asarray([r[name] for r in rows])

    def keyed(rows):
        return {(r["kind"], r["source"], r["side"]): r for r in rows}

    # T4 short-path tolerances and the original force, acceleration and FK gates.
    tolerances = {
        "q": (0, 2e-4),
        "v": (0, 3e-3),
        "q_in": (0, 2e-4),
        "v_in": (0, 3e-3),
        "input_time": (0, 2e-6),
        "output_time": (0, 2e-6),
        "tip_in": (2e-5, 3e-5),
        "rotation_in": (2e-5, 3e-5),
        "qacc": (3e-3, 1e-2),
        "integrated_acc": (3e-3, 1e-2),
        "actuator": (3e-3, 3e-3),
        "constraint": (3e-3, 3e-3),
    }
    errors = {}
    for field, (rtol, atol) in tolerances.items():
        a, b = column(actual, field), column(native, field)
        np.testing.assert_allclose(a, b, rtol=rtol, atol=atol, err_msg=f"world {key}, {field}")
        errors[field] = float(np.max(np.abs(a - b)))
    np.testing.assert_allclose(column(actual, "ctrl"), column(native, "ctrl"), rtol=0, atol=1e-6)
    np.testing.assert_array_equal(column(actual, "q_in")[1:], column(actual, "q")[:-1])
    np.testing.assert_array_equal(column(actual, "v_in")[1:], column(actual, "v")[:-1])
    np.testing.assert_array_equal(column(actual, "tick"), np.arange(1, len(actual) + 1))

    direction = int(np.sign(WORLDS[key][0]))
    dry_peak = np.zeros(2)
    dry_work = np.zeros(2)
    limit_ticks, loaded_ticks, limit_forces = [], [], []
    # Match by native EFC type, source joint/DOF and stop side, independently
    # of array order. Account for every row on every step, including release.
    for ref, gpu in zip(native, actual, strict=True):
        a, b = keyed(gpu["rows"]), keyed(ref["rows"])
        assert len(a) == len(gpu["rows"]) and len(b) == len(ref["rows"]) and a.keys() == b.keys()
        assert (1, 0, -1) in a and (1, 1, -1) in a
        generalized = np.zeros(2)
        for identity, row in a.items():
            other = b[identity]
            assert row["dof"] == other["dof"]
            np.testing.assert_allclose(row["jac"], other["jac"], rtol=2e-4, atol=5e-5)
            np.testing.assert_allclose(row["force"], other["force"], **t4.FORCE_TOL)
            projected = np.asarray(row["jac"]) * row["force"]
            generalized += projected
            if row["kind"] == 1:
                dof = row["dof"]
                dry_peak[dof] = max(dry_peak[dof], abs(row["force"]))
                dry_work[dof] += projected @ gpu["v_in"] * model.opt.timestep
            else:
                assert identity == (3, 0, direction)
                limit_ticks.append(gpu["tick"])
                limit_forces.append(row["force"])
                assert direction * projected[0] <= 3e-3
                if row["force"] > t4.FORCE_TOL["atol"]:
                    loaded_ticks.append(gpu["tick"])
        np.testing.assert_allclose(generalized, gpu["constraint"], rtol=5e-5, atol=5e-5)
    assert loaded_ticks and 1 < limit_ticks[0] <= loaded_ticks[0] <= PUSH
    assert PUSH in limit_ticks and limit_ticks[-1] < len(actual)
    assert np.all(dry_peak > t4.FORCE_TOL["atol"])
    assert np.all(dry_peak <= model.dof_frictionloss + t4.FORCE_TOL["atol"])
    assert np.all(dry_work < 0)
    q = column(actual, "q")
    qin = column(actual, "q_in")
    # Reaction entry is geometric, and positive force is reported separately.
    # At the end both joints have moved; the shoulder has crossed back into the interior.
    upper_margin_edge = abs(model.jnt_range[0, 1]) - model.jnt_margin[0]
    assert direction * qin[0, 0] < upper_margin_edge
    assert direction * qin[limit_ticks[0] - 1, 0] >= upper_margin_edge - 2e-4
    assert direction * q[-1, 0] < upper_margin_edge - 2e-4
    assert direction * actual[-1]["v"][0] < -3e-3
    assert np.ptp(q[:, 1]) > 2e-4
    return dict(
        shoulder_drive_Nm=float(actual[0]["actuator"][0]),
        shoulder_stop_side=direction,
        drive_input_ticks=[1, PUSH],
        reverse_input_ticks=[PUSH + 1, len(actual)],
        first_stop_tick=limit_ticks[0],
        first_loaded_stop_tick=loaded_ticks[0],
        last_loaded_stop_tick=loaded_ticks[-1],
        first_released_tick=limit_ticks[-1] + 1,
        peak_shoulder_stop_Nm=float(max(limit_forces)),
        dry_peak_Nm_N=dry_peak.tolist(),
        dry_work_J=dry_work.tolist(),
        shoulder_range_rad=[float(q[:, 0].min()), float(q[:, 0].max())],
        extension_range_m=[float(q[:, 1].min()), float(q[:, 1].max())],
        max_abs_error=errors,
    )


def test_musa_two_joint_arm_stops_and_reverses(tmp_path, monkeypatch):
    import mujoco as mj

    from phymeo.backends.mujoco_tyx import _kernels

    xml = _arm_xml()
    compilation = compile_source(xml)
    model = compilation.mj_model
    assert model.nq == model.nv == model.nu == 2
    tip = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "tip")
    assert tip > 0
    native = _native_reference(mj, model, tip)
    starts = {k: MuJoCoInitialState([params[0], 0.0], [0.0, 0.0]) for k, params in WORLDS.items()}
    actual = {k: [] for k in WORLDS}
    budget = MuJoCoTyxResourceBudget(64 * 1024 * 1024)
    monkeypatch.setitem(sys.modules, "warp", None)
    monkeypatch.setitem(sys.modules, "mujoco_warp", None)

    def forbid(*args, **kwargs):
        pytest.fail("MUSA joint-step execution used a host simulation path")

    with MuJoCoRuntime(
        compilation,
        mode="musa",
        world_keys=tuple(WORLDS),
        initial_states=starts,
        resource_budget=budget,
        cache_root=tmp_path / "cache",
        contact_capacity=0,
        constraint_capacity=4,
    ) as runtime:
        assert runtime.capabilities["gpu"] and runtime.capabilities["implementation"] == "tyx"
        caps = runtime.capabilities
        assert caps["domain"] == "fixed-scalar-joint-constraints-v1"
        assert caps["contact_law"] == "none" and caps["contact_dimensions"] == ()
        assert caps["native_dry_dof_ids"] == (0, 1)
        assert caps["native_scalar_limit_joint_ids"] == (0, 1)
        assert caps["dry_friction_supported"] and caps["scalar_joint_limits_supported"]
        start = runtime.musa_device_report()
        assert start["device_name"] and start["arch"] and start["backend"] == "musa"
        with monkeypatch.context() as guard:
            guard.setattr(mj, "mj_step", forbid)
            guard.setattr(mj, "mj_forward", forbid)
            guard.setattr(type(_kernels.advance), "__call__", forbid)
            for tick in range(1, PUSH + RETRACT + 1):
                if tick in (1, PUSH + 1):
                    runtime.set_controls({k: MuJoCoControl(_command(k, tick)) for k in WORLDS})
                assert runtime.step(model.opt.timestep) == dict.fromkeys(WORLDS, True)
                for key, state in runtime.integration_snapshot().items():
                    actual[key].append(_sample(state, tip))
        end = runtime.musa_device_report()
        counters = dict(runtime.counters)
        assert end["live_payload_bytes"] == budget.snapshot()["current_bytes"] > 0
    assert budget.snapshot()["current_bytes"] == budget.snapshot()["reserved_bytes"] == 0
    evidence = dict(
        source_xml=xml,
        world_commands=WORLDS,
        push_steps=PUSH,
        retract_steps=RETRACT,
        timestep=model.opt.timestep,
        cpu=native,
        musa=actual,
        device_before=start,
        device_after=end,
        counters=counters,
    )
    (tmp_path / "joint-arm-traces.json").write_text(
        json.dumps(evidence, allow_nan=False), encoding="utf8"
    )
    launches = end["launched_kernels"] - start["launched_kernels"]
    # The shared T4 step still dispatches contact geometry when there are no
    # eligible pairs. FK, mechanics, contact geometry, rows, solve, decode,
    # integration and the two entry writers all run once per batch.
    assert launches == end["completed_kernels"] - start["completed_kernels"] == 9 * (PUSH + RETRACT)
    assert counters["completed_kernels"] == launches
    assert end["downloaded_bytes"] > start["downloaded_bytes"]
    assert end["failure"] is None and end["release_error"] is None
    summary = {k: _verify(k, native[k], actual[k], model) for k in WORLDS}
    (tmp_path / "joint-arm-summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf8"
    )
