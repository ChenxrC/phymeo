"""Ground reactions transmitted through two moving children of a free base."""

import json
import os
import sys
from xml.etree import ElementTree as ET

import numpy as np
import pytest
import test_mujoco_tyx_constraints as t4
import test_mujoco_tyx_floating as floating

from phymeo.backends.mujoco_runtime import MuJoCoControl, MuJoCoRuntime
from phymeo.builders.mujoco_compiler import compile_mujoco_xml

pytestmark = [
    pytest.mark.musa,
    pytest.mark.skipif(os.environ.get("PHYMEO_TEST_MUSA") != "1", reason="requires a MUSA GPU"),
]
COMMANDS = {7: (0.16, -1.5), 41: (-0.12, 0.5)}  # ankle Nm, telescoping leg N
STEPS = 32


def _two_feet_xml():
    root = ET.fromstring(t4.normal._two_feet())
    root.find("option").set("cone", "pyramidal")
    default_geom = root.find("./default/geom")
    default_geom.set("condim", "3")
    default_geom.set("friction", ".6 .02 .003")
    for geom in root.findall(".//worldbody//geom"):
        floor = geom.get("name") == "floor"
        geom.set("contype", "1" if floor else "2")
        geom.set("conaffinity", "2" if floor else "1")
    motors = ET.SubElement(root, "actuator")
    for joint in ("ankle", "leg"):
        ET.SubElement(motors, "motor", name=f"drive-{joint}", joint=joint)
    return ET.tostring(root, encoding="unicode")


def _contact(jacobian, wrench, on_foot):
    # Jacobian rows are in the contact frame at the physical contact point.
    # Its first six columns map free-root rates; the last two map ankle and leg.
    return dict(
        generalized=(jacobian.T @ wrench).tolist(),
        on_foot=on_foot.tolist(),
        contact_wrench=wrench.tolist(),
    )


def _reference(mj, model, packets):
    traces = {}
    for key, packet in packets.items():
        data = mj.MjData(model)
        records = []
        for _ in range(STEPS):
            # This existing independent oracle builds point Jacobians with
            # mj_jac and reads mj_contactForce. It never inspects the GPU state.
            oracle = t4.native(mj, model, data.qpos, data.qvel, packet, data.time)
            contacts = {}
            for c in oracle["contacts"]:
                floor, foot = c["geometry_ids"]
                assert floor == 0 and c["dimension"] == 3 and foot not in contacts
                wrench = c["wrench_contact_frame"]
                contacts[foot] = _contact(c["jacobian"], wrench, c["frame"].T @ wrench[:3])
            floating._set_control(data, packet)
            mj.mj_step(model, data)
            records.append(
                dict(
                    q=data.qpos.tolist(),
                    v=data.qvel.tolist(),
                    time=data.time,
                    qacc=data.qacc.tolist(),
                    actuator=data.qfrc_actuator.tolist(),
                    constraint=data.qfrc_constraint.tolist(),
                    contacts=contacts,
                )
            )
        traces[key] = records
    return traces


def _sample(state):
    step = state.last_step
    assert (
        step.backend == "mujoco-tyx-musa"
        and step.contact_phase == "tyx-euler-input-constraint-evaluation"
    )
    assert (
        step.tick == state.tick
        and step.output_time == state.time
        and step.contact_time == step.input_time
    )
    assert step.input_time < step.output_time
    contacts = {}
    for c in step.contacts:
        floor, foot = map(int, c.geometry_ids)
        assert floor == 0 and c.dimension == 3 and foot not in contacts
        contacts[foot] = _contact(c.jacobian, c.wrench_contact_frame, c.force_on_geometry2_world)
        np.testing.assert_allclose(
            c.force_on_geometry1_world, -c.force_on_geometry2_world, rtol=0, atol=1e-6
        )
    return dict(
        q=state.q.tolist(),
        v=state.v.tolist(),
        time=state.time,
        qacc=step.qacc.tolist(),
        actuator=step.qfrc_actuator.tolist(),
        constraint=step.qfrc_constraint.tolist(),
        contacts=contacts,
    )


def _compare(actual, reference, toes):
    errors = dict(q=0.0, v=0.0, constraint=0.0)
    peaks = np.zeros((2, 2))
    both_loaded_ticks = []
    for tick, (a, b) in enumerate(zip(actual, reference, strict=True), 1):
        for field, atol in (
            ("q", t4.normal.Q_ATOL),
            ("v", t4.normal.V_ATOL),
            ("time", floating.TIME_ATOL),
        ):
            np.testing.assert_allclose(a[field], b[field], rtol=0, atol=atol)
        for field in ("actuator", "constraint"):
            np.testing.assert_allclose(a[field], b[field], **t4.FORCE_TOL)
        np.testing.assert_allclose(a["qacc"], b["qacc"], **t4.normal.ACC_TOL)
        for field in errors:
            errors[field] = max(
                errors[field], float(np.max(np.abs(np.subtract(a[field], b[field]))))
            )
        assert a["contacts"].keys() == b["contacts"].keys()
        total = np.zeros(8)
        support = np.zeros(3)
        loaded = []
        for joint, toe in enumerate(toes):
            if toe not in a["contacts"]:
                continue
            measured, native = a["contacts"][toe], b["contacts"][toe]
            for field in ("on_foot", "contact_wrench", "generalized"):
                np.testing.assert_allclose(measured[field], native[field], **t4.FORCE_TOL)
            projected = np.asarray(measured["generalized"])
            total += projected
            support += measured["on_foot"]
            root_force = np.linalg.norm(projected[:3])
            joint_effort = abs(projected[6 + joint])
            peaks[joint] = np.maximum(peaks[joint], [root_force, joint_effort])
            if min(root_force, joint_effort) > 10 * t4.FORCE_TOL["atol"]:
                loaded.append(toe)
        if len(loaded) == len(toes):
            both_loaded_ticks.append(tick)
        np.testing.assert_allclose(total, a["constraint"], rtol=4e-5, atol=5e-5)
        np.testing.assert_allclose(total[:3], support, rtol=4e-5, atol=5e-5)
    assert both_loaded_ticks, "Neither contact may be a decorative or unloaded foot"
    return dict(
        simultaneously_loaded_ticks=both_loaded_ticks,
        max_abs_error=errors,
        peak_ground_force_on_base_N=peaks[:, 0].tolist(),
        peak_ankle_Nm=float(peaks[0, 1]),
        peak_leg_N=float(peaks[1, 1]),
        first_contact_generalized=actual[0]["constraint"],
        final_root_q=actual[-1]["q"][:7],
        final_ankle_leg_q=actual[-1]["q"][7:],
    )


def test_floating_base_and_both_joints_receive_ground_reaction(monkeypatch, tmp_path):
    from phymeo.backends.mujoco_tyx import _kernels

    mj = floating._deps()
    xml = _two_feet_xml()
    compilation = compile_mujoco_xml(xml, require_asset=False)
    model = compilation.mj_model
    assert (model.nq, model.nv, model.nu) == (9, 8, 2)
    toes = [mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, name) for name in ("toe0", "toe1")]
    assert len(set(toes)) == 2 and min(toes) > 0
    packets = {
        k: floating._stored(
            MuJoCoControl(
                cmd, qfrc_applied=np.zeros(model.nv), xfrc_applied=np.zeros((model.nbody, 6))
            )
        )
        for k, cmd in COMMANDS.items()
    }
    reference = _reference(mj, model, packets)
    samples = {k: [] for k in COMMANDS}

    def forbidden(*args, **kwargs):
        pytest.fail("The MUSA floating step called a host physics, Jacobian or solve path")

    with MuJoCoRuntime(
        compilation,
        mode="musa",
        world_keys=tuple(COMMANDS),
        cache_root=tmp_path / "cache",
        contact_capacity=2,
        constraint_capacity=8,
    ) as runtime:
        runtime.set_controls(packets)
        start = runtime.musa_device_report()
        assert runtime.capabilities["gpu"] and start["device_name"] and start["arch"]
        assert runtime.capabilities["articulation_root_scope"] == "single-free-root"
        assert runtime.capabilities["contact_dimensions"] == (3,)
        assert runtime.capabilities["g1_supported"] is False
        with monkeypatch.context() as guard:
            for name in (
                "mj_step",
                "mj_forward",
                "mj_fullM",
                "mj_rne",
                "mj_jac",
                "mj_contactForce",
            ):
                guard.setattr(mj, name, forbidden)
            for name in ("solve", "cholesky", "inv"):
                guard.setattr(np.linalg, name, forbidden)
            guard.setattr(type(_kernels.advance), "__call__", forbidden)
            for module in ("mujoco_warp", "warp"):
                guard.setitem(sys.modules, module, None)
            for tick in range(1, STEPS + 1):
                assert runtime.step(model.opt.timestep) == dict.fromkeys(COMMANDS, True)
                for key, state in runtime.integration_snapshot().items():
                    assert state.tick == tick
                    samples[key].append(_sample(state))
        end = runtime.musa_device_report()
        assert end["revision"] == start["revision"] + STEPS
        assert end["active_bank"] == (start["active_bank"] ^ (STEPS % 2))
        completed = end["completed_kernels"] - start["completed_kernels"]
        assert completed == end["launched_kernels"] - start["launched_kernels"] >= STEPS
        assert end["downloaded_bytes"] > start["downloaded_bytes"]
        assert end["failure"] is None and end["release_error"] is None
    traces = dict(
        xml=xml,
        timestep=model.opt.timestep,
        controls=COMMANDS,
        device_before=start,
        device_after=end,
        native=reference,
        musa=samples,
    )
    (tmp_path / "floating-ground-traces.json").write_text(
        json.dumps(traces, allow_nan=False), encoding="utf8"
    )
    summary = {k: _compare(samples[k], reference[k], toes) for k in COMMANDS}
    # The two worlds started from the same configuration. Different child motor
    # loads must change the actual base acceleration, joint acceleration and reaction.
    a, b = (samples[k][0] for k in COMMANDS)
    response = np.subtract(a["qacc"], b["qacc"])
    reaction = np.subtract(a["constraint"], b["constraint"])
    assert np.max(np.abs(response[:6])) > t4.normal.ACC_TOL["atol"]
    assert np.max(np.abs(response[6:])) > t4.normal.ACC_TOL["atol"]
    assert np.max(np.abs(reaction)) > t4.FORCE_TOL["atol"]
    summary.update(
        motor_command_difference_effect=dict(
            root_acc=response[:6].tolist(),
            joint_acc=response[6:].tolist(),
            contact_generalized=reaction.tolist(),
        ),
        completed_device_kernels=completed,
    )
    (tmp_path / "floating-ground-summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf8"
    )
