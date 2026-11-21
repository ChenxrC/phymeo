"""Portable causal/mapping checks for previous-receipt load compensation."""

import json
from dataclasses import asdict, replace
from types import SimpleNamespace

import mujoco as mj
import numpy as np
import pytest

from experiments.area_pressure.pusht_load_compensation import PushLoadCompensation
from experiments.area_pressure.pusht_public_scene import IDENTITY
from phymeo.control.manipulation_diagnostic import NativeJointPD
from phymeo.control.manipulation_push_objective import NativePlanarPushProposal
from phymeo.control.manipulation_push_planner import NativePushRoute
from phymeo.geometry.area_pressure import (
    AreaPressureStamp,
    AreaPressureSurface,
    AreaPressureTriangle,
    quadrature_area_pressure,
)
from phymeo.solvers.area_friction import LAW


def _case(*, force=(0.0, -12.0, 0.0), raw=(1.0, 2.0), stage="push", swap_motors=False):
    motors = (
        '<motor joint="j1"/><motor joint="j0"/>'
        if swap_motors
        else '<motor joint="j0"/><motor joint="j1"/>'
    )
    model = mj.MjModel.from_xml_string(
        """<mujoco><compiler angle="radian"/>
      <option gravity="0 0 0"/><default><motor ctrllimited="true" ctrlrange="-20 20"/>
      <joint limited="true" range="-3 3"/></default><worldbody>
      <body name="link"><joint name="j0" axis="0 0 1"/>
      <geom type="capsule" fromto="0 0 0 .4 0 0" size=".02"/>
      <body name="tool" pos=".4 0 0"><joint name="j1" axis="0 0 1"/>
      <geom name="pusher" type="sphere" size=".02" pos=".3 0 0"/></body></body>
      <body name="object" pos="1 1 0"><geom type="sphere" size=".1"/></body>
      </worldbody><actuator>"""
        + motors
        + "</actuator></mujoco>"
    )
    pd = NativeJointPD(model, ("j0", "j1"), 25.0, 3.0)
    q, v = np.array((0.3, -0.4)), np.zeros(2)
    target = q + np.asarray(raw) / 25.0
    _, pd_record = pd.command(q, v, target)
    proposal = NativePlanarPushProposal(
        "PROPOSAL_ONLY",
        "portable-test",
        0,
        0,
        0.01,
        True,
        (0.0, 0.0, 0.0),
        (0.0, -1.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, -0.01, 0.0),
        (0.0, 0.01, 0.0),
        (0.0, 1.0, 0.0),
        0.0,
        0.1,
        (0.0, 0.01),
        0.0,
        (0.0, 0.01),
        0.0,
        2.0,
        1.0,
        1.0,
        (0, -1.0, 0, 0),
    )
    route = NativePushRoute(
        "sampled_reference_ready",
        proposal,
        q.copy(),
        ((0.0, stage, target.copy()), (1.0, stage, target.copy())),
        (),
        1,
        1,
        (0.0, 0.0),
    )
    driver = SimpleNamespace(model=model, active=(100, route), tick=107, last_q=q.copy())
    entry = SimpleNamespace(
        model=model,
        pd=pd,
        driver=driver,
        scene=SimpleNamespace(robot_joint_names=("j0", "j1"), timestep_s=0.001),
    )
    stamp = AreaPressureStamp("6", 6, 0.006, "source-accepted-input")
    triangle = AreaPressureTriangle(
        "patch",
        "cell",
        "face",
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        (1.0, 1.0, 1.0),
        (0.0, 0.0, 1.0),
    )
    quad = quadrature_area_pressure(AreaPressureSurface(IDENTITY, stamp, (triangle,)))
    point_forces = np.asarray(force)[None, :] * np.array((1 / 6, 2 / 6, 3 / 6))[:, None]
    pair = dict(
        identity=IDENTITY,
        body_ids=(model.body("tool").id, model.body("object").id),
        stamp=stamp,
        source_quadrature=quad,
        sample_ids=quad.sample_ids,
        source_ids=quad.source_ids,
        law=LAW,
        force_on_a=point_forces,
        force_on_b=-point_forces,
        normal_force=np.ones(3),
    )
    solver = dict(
        schema="mujoco-tyx-area-friction-resident-native-v1",
        precision="actual-packed-point-fp32",
        law=LAW,
        solver_status=0,
        output_velocity=v.copy(),
    )
    receipt = SimpleNamespace(
        world=6,
        tick=7,
        input_time=0.006,
        output_time=0.007,
        h=0.001,
        contact_time=0.006,
        contact_phase="tyx-euler-input-constraint-evaluation",
        state_precision="fp32-device-integration",
        backend="mujoco-tyx-cuda",
        input_q=np.zeros(2),
        input_v=np.zeros(2),
        raw_contact_fields=dict(area_friction=(pair,), area_friction_solver=solver),
    )
    snapshot = SimpleNamespace(
        world=6,
        tick=7,
        time=0.007,
        q=q.copy(),
        v=v.copy(),
        last_step=receipt,
        observation_kind="accepted-integration-state",
    )
    diagnostic = dict(
        input_tick=107,
        local_tick=7,
        input_time=0.007,
        measurement_q=q.tolist(),
        measurement_v=v.tolist(),
        stage=stage,
        target_q=target.tolist(),
        target_v=[0.0, 0.0],
        route=dict(route_start_tick=100, route_elapsed_s=0.007),
        pd=asdict(pd_record),
    )
    return entry, snapshot, diagnostic, pair


def _analytic_j(q):
    a, b = q
    return np.array(
        (
            (-0.4 * np.sin(a) - 0.3 * np.sin(a + b), -0.3 * np.sin(a + b)),
            (0.4 * np.cos(a) + 0.3 * np.cos(a + b), 0.3 * np.cos(a + b)),
            (0.0, 0.0),
        )
    )


@pytest.mark.parametrize("stage", ("entry", "push"))
@pytest.mark.parametrize("swap_motors", (False, True))
def test_previous_point_force_projects_through_current_geometry_in_native_motor_order(
    stage, swap_motors, monkeypatch
):
    entry, snapshot, diagnostic, pair = _case(stage=stage, swap_motors=swap_motors)
    controller = PushLoadCompensation(entry)
    before_q, before_v = snapshot.q.tobytes(), snapshot.v.tobytes()

    def forbidden(*args, **kwargs):
        raise AssertionError("load compensation must only use kinematics and the saved PD effort")

    for name in ("mj_forward", "mj_inverse", "mj_step", "mj_step1", "mj_step2", "mj_rne"):
        monkeypatch.setattr(mj, name, forbidden)
    effort, result = controller(entry, snapshot, diagnostic)
    order = np.argsort(entry.pd.actuator_ids)
    jacobian = _analytic_j(snapshot.q)[:, order]
    correction = jacobian.T @ np.array((0.0, 6.0, 0.0))
    np.testing.assert_allclose(result["jacobian_world"], jacobian, rtol=0, atol=2e-15)
    np.testing.assert_allclose(result["correction_nm"], correction, rtol=0, atol=2e-15)
    expected_raw = np.asarray(diagnostic["pd"]["unclipped_effort"])[order] + correction
    np.testing.assert_allclose(effort, expected_raw, rtol=0, atol=3e-15)
    np.testing.assert_array_equal(
        result["previous_force_on_tool_world_n"], pair["force_on_a"].sum(axis=0)
    )
    assert result["source_local_tick"] == snapshot.tick
    assert result["source_output_time"] == snapshot.time
    assert result["gain"] == 0.5 and result["enabled"] is True and result["applied"] is True
    assert result["resisting_force_n"] == 12.0
    assert snapshot.q.tobytes() == before_q and snapshot.v.tobytes() == before_v
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize(
    "stage,force,enabled,resisting",
    (
        ("approach", (0.0, -12.0, 0.0), False, 12.0),
        ("push", (0.0, 0.0, 0.0), True, 0.0),
        ("push", (0.0, 12.0, 0.0), True, 0.0),
        ("push", (12.0, 0.0, 8.0), True, 0.0),
    ),
)
def test_phase_and_resisting_projection_never_turn_unloaded_or_assisting_force_into_correction(
    stage, force, enabled, resisting
):
    entry, snapshot, diagnostic, _ = _case(stage=stage, force=force)
    effort, result = PushLoadCompensation(entry)(entry, snapshot, diagnostic)
    assert result["gain"] == 0.5 and result["enabled"] is enabled
    assert result["resisting_force_n"] == resisting and result["applied"] is False
    np.testing.assert_array_equal(result["correction_nm"], np.zeros(2))
    np.testing.assert_array_equal(effort, diagnostic["pd"]["effort"])


def test_adds_to_unclipped_pd_before_exact_original_motor_limit():
    entry, snapshot, diagnostic, _ = _case(raw=(-22.0, 19.0))
    effort, result = PushLoadCompensation(entry)(entry, snapshot, diagnostic)
    correction = _analytic_j(snapshot.q).T @ np.array((0.0, 6.0, 0.0))
    raw = np.asarray(diagnostic["pd"]["unclipped_effort"]) + correction
    np.testing.assert_allclose(result["combined_unclipped_nm"], raw, rtol=0, atol=3e-15)
    np.testing.assert_allclose(effort, np.clip(raw, -19.9, 19.9), rtol=0, atol=3e-15)
    assert effort[1] == 19.9
    assert not np.array_equal(
        effort, np.clip(np.asarray(diagnostic["pd"]["effort"]) + correction, -19.9, 19.9)
    )


def test_a_bound_empty_previous_surface_is_valid_zero_load_not_a_missing_receipt():
    entry, snapshot, diagnostic, pair = _case(stage="approach")
    quad = quadrature_area_pressure(AreaPressureSurface(IDENTITY, pair["stamp"], ()))
    pair.update(
        source_quadrature=quad,
        source_ids=(),
        sample_ids=(),
        force_on_a=np.empty((0, 3)),
        force_on_b=np.empty((0, 3)),
        normal_force=np.empty(0),
    )
    effort, result = PushLoadCompensation(entry)(entry, snapshot, diagnostic)
    assert result["source_local_tick"] == snapshot.tick and result["resisting_force_n"] == 0
    np.testing.assert_array_equal(result["previous_force_on_tool_world_n"], np.zeros(3))
    np.testing.assert_array_equal(result["correction_nm"], np.zeros(2))
    np.testing.assert_array_equal(effort, diagnostic["pd"]["effort"])


@pytest.mark.parametrize(
    "fault",
    (
        "missing",
        "tick",
        "time",
        "world",
        "phase",
        "schema",
        "solver-status",
        "swapped-bodies",
        "material",
        "extra-pair",
        "source-stamp",
        "nan-force",
        "opposite-force",
        "stale-pd",
        "mapping",
        "direction",
    ),
)
def test_bad_or_stale_source_is_rejected_even_in_approach(fault):
    entry, snapshot, diagnostic, pair = _case(stage="approach")
    controller = PushLoadCompensation(entry)
    receipt = snapshot.last_step
    if fault == "missing":
        snapshot.last_step = None
    elif fault == "tick":
        receipt.tick -= 1
    elif fault == "time":
        receipt.output_time -= 0.001
    elif fault == "world":
        receipt.world = 5
    elif fault == "phase":
        receipt.contact_phase = "endpoint-cpu-forward-recomputed"
    elif fault == "schema":
        receipt.raw_contact_fields["area_friction_solver"]["schema"] = "foreign"
    elif fault == "solver-status":
        receipt.raw_contact_fields["area_friction_solver"]["solver_status"] = 8
    elif fault == "swapped-bodies":
        pair["body_ids"] = tuple(reversed(pair["body_ids"]))
    elif fault == "material":
        pair["identity"] = replace(IDENTITY, rigid_material="other")
    elif fault == "extra-pair":
        receipt.raw_contact_fields["area_friction"] = (pair, pair)
    elif fault == "source-stamp":
        pair["stamp"] = replace(pair["stamp"], step_index=5)
    elif fault == "nan-force":
        pair["force_on_a"][0, 0] = np.nan
    elif fault == "opposite-force":
        pair["force_on_b"][0, 0] += 1.0
    elif fault == "stale-pd":
        diagnostic["measurement_q"][0] += 0.1
    elif fault == "mapping":
        diagnostic["pd"]["actuator_ids"] = (1, 0)
    else:
        start, route = entry.driver.active
        entry.driver.active = (
            start,
            replace(route, proposal=replace(route.proposal, direction_world=(0.0, 2.0, 0.0))),
        )
    with pytest.raises(ValueError):
        controller(entry, snapshot, diagnostic)


@pytest.mark.parametrize("fault", ("gear", "gain", "narrow-limit"))
def test_nonunit_or_incompatible_motor_mapping_is_not_silently_reinterpreted(fault):
    entry, _, _, _ = _case()
    if fault == "gear":
        entry.model.actuator_gear[0, 0] = -1.0
    elif fault == "gain":
        entry.model.actuator_gainprm[0, 0] = 2.0
    else:
        entry.model.actuator_ctrlrange[0] = (-10.0, 10.0)
    with pytest.raises(ValueError):
        PushLoadCompensation(entry)
