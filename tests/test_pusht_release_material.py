"""Analytic material evidence checks; no CUDA, archived run or dynamics solve."""

from dataclasses import replace
from types import SimpleNamespace

import mujoco as mj
import numpy as np
import pytest

from experiments.area_pressure.pusht_public_scene import build_side_asset
from experiments.area_pressure.pusht_release_material import material_evidence
from phymeo.backends.mujoco_area_pressure_scene import BodyAreaPressureAsset
from phymeo.control.manipulation_release_monitor import ReleaseMonitor, ReleaseStep
from phymeo.geometry.area_pressure import (
    AreaPressureIdentity,
    AreaPressureStamp,
    quadrature_area_pressure,
)
from phymeo.geometry.area_pressure_intersection import (
    RigidPressureMesh,
    TetraPressureMesh,
    intersect_area_pressure,
)
from phymeo.training.manipulation_tasks import manipulation_scenes


def _fixture():
    model = mj.MjModel.from_xml_string("""<mujoco><worldbody>
      <body name="tool"><freejoint/><geom type="sphere" size=".01" mass="1"/></body>
      <body name="object"><geom type="sphere" size=".01" mass="1"/></body>
    </worldbody></mujoco>""")
    pressure = TetraPressureMesh(
        ("o", "x", "y", "z", "c"),
        ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (0.25, 0.25, 0.25)),
        ((0, 1, 2, 4), (0, 2, 3, 4), (0, 3, 1, 4), (1, 3, 2, 4)),
        (0, 0, 0, 0, 1),
        ("base", "x-side", "y-side", "slant"),
    )
    # Both finite patches lie entirely in the base cell, where p=4z.
    # The last face is clear, but remains in every evidence catalogue.
    vertices = tuple(
        (x + dx, y, 0.05) for dx in (0.0, 0.2, 2.0) for x, y in ((0.1, 0.1), (0.2, 0.1), (0.1, 0.2))
    )
    rigid = RigidPressureMesh(
        tuple(f"r{i}" for i in range(9)),
        vertices,
        ((0, 1, 2), (3, 4, 5), (6, 7, 8)),
        ("z-first", "a-second", "clear-third"),
    )
    asset = BodyAreaPressureAsset(
        AreaPressureIdentity("tool", "object", "analytic-PL", "finite-faces"),
        pressure,
        rigid,
    )
    state = dict(q=model.qpos0.copy(), v=np.zeros(model.nv), time=0.0)
    return model, asset, state


def _after(before, *, z=0.025):
    state = dict(q=np.array(before["q"]), v=np.array(before["v"]), time=before["time"] + 0.001)
    state["q"][2] = z
    return state


def _geometry(model, asset, state):
    data = mj.MjData(model)
    data.qpos[:] = state["q"]
    mj.mj_kinematics(model, data)
    meshes = []
    for name, local in (("tool", asset.pressure_local), ("object", asset.rigid_local)):
        body = model.body(name).id
        vertices = local.world_vertices @ data.xmat[body].reshape(3, 3).T + data.xpos[body]
        meshes.append(replace(local, world_vertices=vertices))
    return intersect_area_pressure(
        *meshes,
        identity=asset.identity,
        stamp=AreaPressureStamp("world-6", 7, state["time"], "actual-input"),
    )


def _live(model, asset, before, after, *, geometry=None):
    if geometry is None:
        geometry = _geometry(model, asset, before)
    quad = quadrature_area_pressure(geometry.surface)
    count = len(quad.points)
    normal = np.arange(1.0, count + 1.0)
    force = normal[:, None] * quad.normals + np.array((0.125, 0.25, 0.0))
    pair = dict(
        source_quadrature=quad,
        source_ids=quad.source_ids,
        sample_ids=quad.sample_ids,
        identity=asset.identity,
        stamp=quad.stamp,
        body_ids=(model.body("tool").id, model.body("object").id),
        normal_force=normal,
        force_on_a=force,
        force_on_b=-force,
    )
    receipt = SimpleNamespace(
        input_q=np.array(before["q"]),
        input_v=np.array(before["v"]),
        input_time=before["time"],
        raw_contact_fields={"area_friction": (pair,)},
    )
    return geometry, receipt, pair


def _previous(before, after):
    return dict(
        input_state=before,
        output_state=after,
        area=dict(normal_load_n=17.0, total_force_on_pusher_world_n=[1.0, 2.0, 3.0]),
        measurement=dict(
            phase="accepted-step-input",
            time_s=before["time"],
            native_contacts=0,
            group_force_world=[-1.0, -2.0, -3.0],
        ),
    )


def test_real_side_catalogue_is_fixed_including_clear_faces():
    source = next(x for x in manipulation_scenes() if x.key == "pusht-planar")
    model = mj.MjModel.from_xml_string(source.mjcf)
    asset = build_side_asset(model)
    before = dict(q=model.qpos0.copy(), v=np.zeros(model.nv), time=0.0)
    # Place the actual pressure body far from the T for a portable clear seed.
    before["q"][0] -= 2.0
    after = dict(q=before["q"].copy(), v=before["v"].copy(), time=0.001)
    previous = _previous(before, after)
    previous["area"].update(normal_load_n=0.0, total_force_on_pusher_world_n=[0.0, 0.0, 0.0])
    previous["measurement"]["group_force_world"] = [0.0, 0.0, 0.0]
    rows = material_evidence(model, asset, before, after, previous=previous)
    assert len(asset.rigid_local.face_ids) == 6
    assert len(rows) == 7
    assert rows[0].key.identity[-1] == "whole"
    assert tuple(r.key.identity[-1] for r in rows[1:]) == asset.rigid_local.face_ids
    assert all(r.output_strictly_separated is True for r in rows)


def test_live_source_face_mapping_depth_speed_and_bilateral_forces(monkeypatch):
    model, asset, before = _fixture()
    # v_z=.5-x: the contact patches have x<=.4, whereas unrelated tetra
    # vertices include x=1. Using every tetra vertex would falsely report -.5.
    before["v"][[2, 4]] = (0.5, 1.0)
    after = _after(before)
    geometry, receipt, pair = _live(model, asset, before, after)

    def forbidden(*args, **kwargs):
        raise AssertionError("material evidence must not execute dynamics")

    for name in ("mj_forward", "mj_inverse", "mj_step", "mj_step1", "mj_step2"):
        monkeypatch.setattr(mj, name, forbidden)
    q0, v0 = before["q"].tobytes(), before["v"].tobytes()
    rows = material_evidence(model, asset, before, after, geometry=geometry, receipt=receipt)
    assert [r.key.identity[-1] for r in rows] == ["whole", *asset.rigid_local.face_ids]
    assert rows[0].normal_load_n == float(np.sum(pair["normal_force"]))
    assert rows[0].normal_load_n == sum(r.normal_load_n for r in rows[1:])
    for row, fid in zip(rows[1:], asset.rigid_local.face_ids, strict=True):
        mask = np.array([s[1] == fid for s in pair["source_quadrature"].source_ids])
        assert row.normal_load_n == float(np.sum(pair["normal_force"][mask]))
        np.testing.assert_array_equal(row.force_on_a_world, pair["force_on_a"][mask].sum(axis=0))
        np.testing.assert_array_equal(row.force_on_b_world, pair["force_on_b"][mask].sum(axis=0))
    for row in rows[:3]:
        assert row.input_depth_m == pytest.approx(0.05, abs=2e-15)
        assert row.output_depth_m == pytest.approx(0.025, abs=2e-15)
        assert row.input_contact and row.output_contact
    assert rows[1].input_normal_speed_m_s == pytest.approx(0.3, abs=2e-15)
    assert rows[2].input_normal_speed_m_s == pytest.approx(0.1, abs=2e-15)
    assert rows[0].input_normal_speed_m_s == pytest.approx(0.1, abs=2e-15)
    assert rows[3].input_contact is False and rows[3].normal_load_n == 0
    assert rows[3].input_normal_speed_m_s is None
    assert before["q"].tobytes() == q0 and before["v"].tobytes() == v0


@pytest.mark.parametrize("z,strict,depth", [(0.05, False, 0.0), (0.1, True, 0.0)])
def test_zero_pressure_touch_is_not_strict_separation(z, strict, depth):
    model, asset, before = _fixture()
    after = _after(before, z=z)
    geometry, receipt, _ = _live(model, asset, before, after)
    rows = material_evidence(model, asset, before, after, geometry=geometry, receipt=receipt)
    assert rows[0].output_strictly_separated is strict
    assert rows[1].output_strictly_separated is strict
    assert rows[1].output_depth_m == pytest.approx(depth, abs=2e-15)
    assert rows[1].output_contact is (not strict)


def test_legacy_seed_preserves_unknown_face_forces_and_old_whole_load():
    model, asset, before = _fixture()
    after = _after(before)
    rows = material_evidence(model, asset, before, after, previous=_previous(before, after))
    assert rows[0].normal_load_n == 17.0
    assert rows[0].force_on_a_world == (1.0, 2.0, 3.0)
    assert rows[0].force_on_b_world == (-1.0, -2.0, -3.0)
    assert all(
        r.normal_load_n is None and r.force_on_a_world is None and r.force_on_b_world is None
        for r in rows[1:]
    )
    # That same record cannot be rebound as the following input's current load.
    with pytest.raises(ValueError, match="previous|seed|state"):
        material_evidence(
            model, asset, after, _after(after, z=0.04), previous=_previous(before, after)
        )


@pytest.mark.parametrize("corruption", ["state", "face", "sources", "count", "negative", "both"])
def test_rejects_wrong_state_source_mapping_and_ambiguous_force_phase(corruption):
    model, asset, before = _fixture()
    after = _after(before)
    geometry, receipt, pair = _live(model, asset, before, after)
    options = dict(geometry=geometry, receipt=receipt)
    if corruption == "state":
        receipt.input_q[0] += 0.01
    elif corruption == "face":
        quad = pair["source_quadrature"]
        pair["source_quadrature"] = replace(
            quad, source_ids=tuple((c, "foreign-face") for c, _ in quad.source_ids)
        )
    elif corruption == "sources":
        pair["source_ids"] = tuple(reversed(pair["source_ids"]))
    elif corruption == "count":
        pair["normal_force"] = pair["normal_force"][:-1]
    elif corruption == "negative":
        pair["normal_force"][0] = -1.0
    else:
        options["previous"] = _previous(before, after)
    with pytest.raises(ValueError):
        material_evidence(model, asset, before, after, **options)


# Measured same-q CPU/output versus CUDA/input depth gap in stage 20. This
# analytic fixture injects that FK translation discrepancy; it does not pretend
# to execute the saved CUDA case or to alter the monitor's zero tolerances.
FK_DEPTH_GAP = 0.002985447623637378 - 0.002985436762843509


def _continuous_pair(*, output_shift=0.005):
    model, asset, earlier = _fixture()
    before = _after(earlier)
    after = _after(before, z=before["q"][2] + output_shift)
    previous = _previous(earlier, before)
    previous["area"]["normal_load_n"] = 100.0
    seed = material_evidence(model, asset, earlier, before, previous=previous)
    # Same accepted input q/v, but device FK puts the pressure body 10.86 nm
    # farther inward. All generated source quadrature and loads still bind to
    # this actual device geometry; only the declared monitor geometry is CPU.
    device_pose = dict(q=before["q"].copy(), v=before["v"].copy(), time=before["time"])
    device_pose["q"][2] -= FK_DEPTH_GAP
    geometry = _geometry(model, asset, device_pose)
    geometry, receipt, pair = _live(model, asset, before, after, geometry=geometry)
    return model, asset, earlier, before, after, seed, geometry, receipt, pair


def _step(before, after, rows, tick):
    return ReleaseStep(
        6,
        tick,
        tick + 1,
        before["time"],
        after["time"],
        tuple(before["q"]),
        tuple(before["v"]),
        tuple(after["q"]),
        tuple(after["v"]),
        rows,
    )


def test_same_q_cpu_depth_is_continuous_despite_device_fk_gap():
    model, asset, earlier, before, after, seed, geometry, receipt, pair = _continuous_pair()
    rows = material_evidence(model, asset, before, after, geometry=geometry, receipt=receipt)
    assert rows[0].input_depth_m == seed[0].output_depth_m
    assert rows[0].input_depth_m == pytest.approx(0.025, abs=2e-15)
    assert rows[0].output_depth_m < rows[0].input_depth_m
    assert rows[0].normal_load_n == pair["normal_force"].sum()
    monitor = ReleaseMonitor()
    monitor.start(_step(earlier, before, seed, 1699))
    decision = monitor.observe(_step(before, after, rows, 1700))
    assert decision.admissible and not decision.violations
    assert monitor.policy.depth_tolerance_m == monitor.policy.load_tolerance_n == 0


def test_real_cpu_deepening_remains_rejected_at_ten_nanometres():
    model, asset, earlier, before, after, seed, geometry, receipt, _ = _continuous_pair(
        output_shift=-FK_DEPTH_GAP
    )
    rows = material_evidence(model, asset, before, after, geometry=geometry, receipt=receipt)
    monitor = ReleaseMonitor()
    monitor.start(_step(earlier, before, seed, 1699))
    decision = monitor.observe(_step(before, after, rows, 1700))
    assert not decision.admissible
    assert any(v.code == "penetration_increased" for v in decision.violations)


def test_device_and_cpu_geometry_numbers_are_reported_separately():
    model, asset, _, before, after, _, geometry, receipt, _ = _continuous_pair()
    diagnostics = {}
    rows = material_evidence(
        model, asset, before, after, geometry=geometry, receipt=receipt, diagnostics=diagnostics
    )
    assert diagnostics["schema"] == "release-material-geometries-v1"
    assert diagnostics["monitor_geometry"] == "cpu-fk-exact-clip"
    assert diagnostics["input_face_contact_agreement"] is True
    assert diagnostics["device_only_input_faces"] == []
    assert diagnostics["cpu_only_input_faces"] == []
    whole = diagnostics["whole"]
    assert whole["cpu_input"]["depth_m"] == rows[0].input_depth_m
    assert whole["cpu_output"]["depth_m"] == rows[0].output_depth_m
    assert whole["device_input"]["depth_m"] - whole["cpu_input"]["depth_m"] == pytest.approx(
        FK_DEPTH_GAP, rel=0, abs=2e-15
    )
    assert whole["device_input"]["strictly_separated"] is None
    assert [r["rigid_face_id"] for r in diagnostics["faces"]] == list(asset.rigid_local.face_ids)
    for row, detail in zip(rows[1:], diagnostics["faces"], strict=True):
        assert detail["cpu_input"]["depth_m"] == row.input_depth_m
        assert detail["cpu_input"]["normal_speed_m_s"] == row.input_normal_speed_m_s
        assert detail["cpu_output"]["normal_speed_m_s"] == row.output_normal_speed_m_s


@pytest.mark.parametrize("device_contact", [True, False])
def test_cpu_device_face_disagreement_cannot_silently_drop_new_contact(device_contact):
    model, asset, before = _fixture()
    # Opposite sides of the zero-pressure boundary, less than the measured FK
    # gap apart. Positive-area face existence differs, even with tiny pressure.
    before["q"][2] = 0.05 + (FK_DEPTH_GAP / 4 if device_contact else -FK_DEPTH_GAP / 4)
    after = _after(before, z=0.051)
    device_pose = dict(q=before["q"].copy(), v=before["v"].copy(), time=before["time"])
    device_pose["q"][2] = 0.05 + (-FK_DEPTH_GAP / 4 if device_contact else FK_DEPTH_GAP / 4)
    geometry = _geometry(model, asset, device_pose)
    geometry, receipt, _ = _live(model, asset, before, after, geometry=geometry)
    diagnostics = {}
    with pytest.raises(ValueError, match="CPU/device.*face|face.*disagreement"):
        material_evidence(
            model, asset, before, after, geometry=geometry, receipt=receipt, diagnostics=diagnostics
        )
    assert diagnostics["input_face_contact_agreement"] is False
    changed = diagnostics["device_only_input_faces" if device_contact else "cpu_only_input_faces"]
    assert changed == list(asset.rigid_local.face_ids[:2])
