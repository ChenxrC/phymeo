"""Admission and independent geometry checks for mixed free-root area scenes."""

from types import SimpleNamespace

import mujoco as mj
import numpy as np
import pytest

from phymeo.backends.area_friction_law import AreaFrictionPairLaw
from phymeo.backends.mujoco_area_pressure import compile_area_plan
from phymeo.backends.mujoco_tyx._allocation_manifest import allocation_manifest
from phymeo.geometry.area_pressure import AreaPressureIdentity
from phymeo.training.manipulation_tasks import manipulation_scenes


@pytest.mark.cuda
@pytest.mark.parametrize("nv", [6, 7])
def test_free_body_native_support_without_area_at_smaller_widths(nv):
    from dataclasses import replace

    from test_mujoco_area_friction import _declarations
    from test_mujoco_area_friction_wide import _wide

    from phymeo.backends.mujoco_heterogeneous import MuJoCoTyxHeterogeneousBatch
    from phymeo.backends.mujoco_heterogeneous_area_scene import (
        MuJoCoTyxHeterogeneousAreaPressureScene,
    )
    from phymeo.backends.mujoco_runtime import MuJoCoInitialState
    from phymeo.builders.mujoco_compiler import compile_mujoco_xml

    if nv == 7:
        spec, world, assets, _ = _wide(7)
    else:
        specs, worlds, assets = _declarations()
        spec, world = specs[0], worlds[0]
        assets = {spec.key: assets[spec.key]}
        identity = assets[spec.key][0].identity
        spec = replace(spec, area_friction_laws=(AreaFrictionPairLaw(identity, 0.5, 0.001),))
    xml = spec.compilation.source.primary_xml.decode("utf-8")
    xml = xml.replace(
        "<worldbody>",
        '<worldbody><geom name="floor" type="plane" pos="0 0 .099" size="1 1 .05" condim="3" friction=".5 0 0" contype="2" conaffinity="1"/>',
    )
    xml = xml.replace(
        '<geom name="pressure-boundary"',
        '<geom name="foot" type="sphere" size=".002" condim="3" friction=".5 0 0" contype="1" conaffinity="2"/><geom name="pressure-boundary"',
    )
    spec = replace(spec, compilation=compile_mujoco_xml(xml, require_asset=False))
    q = list(world.initial_state.q)
    q[2] = 0.1
    world = replace(world, initial_state=MuJoCoInitialState(q, np.zeros(nv)))
    with MuJoCoTyxHeterogeneousBatch((spec,), (world,), mode="cuda") as batch:
        scene = MuJoCoTyxHeterogeneousAreaPressureScene(batch, assets)
        scene.step(0.001, expected_revision=batch.checkpoint_revision)
        (step,) = batch.integration_snapshot().values()
        assert step.tick == 1 and step.v.shape == (nv,)
        (support,) = step.last_step.contacts
        assert support.normal_force > 0 and len(step.last_step.constraints) == 4
        (area,) = step.last_step.raw_contact_fields["area_friction"]
        assert len(area["sample_ids"]) == 0


def test_floating_native_manifest_is_budgeted_for_each_world():
    scene = next(s for s in manipulation_scenes() if s.key == "pusht-planar")
    model = mj.MjModel.from_xml_string(scene.mjcf)
    identity = AreaPressureIdentity("tool", "object", "elastic", "rigid")
    law = AreaFrictionPairLaw(identity, 0.7, 0.01)
    plan = compile_area_plan(mj, model, (("tool", "object"),), (law,))
    assert plan.resident_native_contacts and plan.kind == 3
    assert (plan.npair, plan.nlimit, plan.nfriction) == (9, 2, 0)
    manifest = allocation_manifest(plan, 5, model.nbody, (64, 512))
    shapes = {e.name: e.shape for e in manifest.entries if e.name.startswith("area_native_")}
    assert shapes == {
        "area_native_applied": (5, 8),
        "area_native_meta": (5, 4),
        "area_native_trace_meta": (5, 25, 3),
        "area_native_trace": (5, 25, 30),
    }


@pytest.mark.parametrize("kind,features", [(2, 1), (3, 2)])
def test_plane_round_features_bind_distance_point_activity_and_friction_frame(kind, features):
    from phymeo.backends.mujoco_area_native_geometry import validate_geometry

    geom = np.zeros((2, 16), np.float32)
    geom[:, 3:12] = np.eye(3).ravel()
    geom[1, 12:14] = 0.25, 0.5
    # Capsule along X, centered .125 m above a Z-up plane.
    rotation = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    dynamics = SimpleNamespace(
        body_position=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.125]]),
        body_rotation=np.stack((np.eye(3), rotation)),
    )
    law = np.zeros((1, 13), np.float32)
    law[0, 10] = 1e-12
    plan = SimpleNamespace(
        geom_ids=np.array([[0, 0], [1, kind]]),
        geom_data=geom,
        pair_ids=np.array([[0, 1, 0, features]]),
        npair=1,
        nslot=features,
        slot_ids=np.array([(0, i, 0, 1) for i in range(features)]),
        pair_law=law,
    )
    rows = np.zeros((features, 32))
    n = np.array([0.0, 0.0, 1.0])
    tangent = np.eye(3)[0 if kind == 3 else 1]
    for i, row in enumerate(rows):
        row[:3] = 1, 1, -0.125
        row[9:12] = (0.5 * (1 - 2 * i) if kind == 3 else 0.0, 0.0, -0.0625)
        row[12:21] = np.stack((n, tangent, np.cross(n, tangent))).ravel()
    validate_geometry(plan, rows, dynamics)
    for column, value in ((1, 0), (2, -0.1), (9, 0.2), (15, 0.5)):
        forged = rows.copy()
        forged[0, column] = value
        with pytest.raises(ValueError):
            validate_geometry(plan, forged, dynamics)


def test_missing_box_plane_contacts_and_unverified_overlapping_boxes_reject():
    from phymeo.backends.mujoco_area_native_geometry import validate_geometry

    geom = np.zeros((3, 16), dtype=np.float32)
    geom[:, 3:12] = np.eye(3).ravel()
    geom[1:, 12:15] = (0.1, 0.2, 0.3)
    dynamics = SimpleNamespace(
        body_position=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.1], [1.0, 0.0, 0.1]]),
        body_rotation=np.tile(np.eye(3), (3, 1, 1)),
    )
    plan = SimpleNamespace(
        geom_ids=np.array([[0, 0], [1, 6], [2, 6]]),
        geom_data=geom,
        pair_ids=np.array([[0, 1, 0, 8], [1, 2, 8, 1]]),
        npair=2,
        nslot=9,
        slot_ids=np.array([(0, i, 0, 1) for i in range(8)] + [(1, 0, 1, 2)]),
        pair_law=np.zeros((2, 13), np.float32),
    )
    rows = np.zeros((9, 32))
    with pytest.raises(ValueError):
        validate_geometry(plan, rows, dynamics)
    # Four bottom corners are in the plane's detection margin. Each has its
    # own material feature and a midpoint between the corner and the plane.
    for index in range(4):
        corner = np.array([0.1 if index & 1 else -0.1, 0.2 if index & 2 else -0.2, -0.2])
        rows[index, :3] = 1, 1, -0.2
        rows[index, 9:12] = corner + [0.0, 0.0, 0.1]
        rows[index, 12:21] = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]).ravel()
    validate_geometry(plan, rows, dynamics)
    swapped = rows.copy()
    swapped[[0, 1]] = swapped[[1, 0]]
    with pytest.raises(ValueError, match="contact point"):
        validate_geometry(plan, swapped, dynamics)
    rows[:] = 0
    dynamics.body_position[1:, 2] = 1.0
    validate_geometry(plan, rows, dynamics)
    dynamics.body_position[2] = dynamics.body_position[1]
    with pytest.raises(ValueError):
        validate_geometry(plan, rows, dynamics)
