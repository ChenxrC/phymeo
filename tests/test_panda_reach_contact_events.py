"""Native contact signs, saved velocity timing and physical crossed reset banks."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def native(monkeypatch):
    mj = pytest.importorskip("mujoco")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    import panda_reach_contact_events as events

    return mj, events


@pytest.fixture
def spheres(native):
    mj, _ = native
    model = mj.MjModel.from_xml_string("""
        <mujoco>
          <option timestep=".01"/>
          <worldbody>
            <body name="left">
              <joint type="slide" axis="1 0 0"/>
              <joint type="hinge" axis="0 0 1"/>
              <geom name="left_ball" type="sphere" pos="0 .03 0" size=".05"/>
            </body>
            <body name="right" pos=".11 .03 0">
              <joint type="slide" axis="1 0 0"/>
              <geom name="right_ball" type="sphere" size=".05"/>
            </body>
          </worldbody>
        </mujoco>
    """)
    q = np.array([[0., 0., 0.], [.012, 0., 0.], [.013, 0., 0.]], dtype=np.float32)
    v = np.array([[.4, 2., .05], [.3, 1., .08], [.1, 0., .2]], dtype=np.float32)
    return model, mj.MjData(model), q, v


@pytest.mark.parametrize("finger_on_left", [True, False])
def test_contact_normal_and_point_speed_under_role_exchange(native, spheres, finger_on_left):
    _, events = native
    model, data, q, v = spheres
    left, right = (model.geom(name).id for name in ("left_ball", "right_ball"))
    finger, part = (left, right) if finger_on_left else (right, left)
    step, points, count = events.first_contacts(model, data, q, {finger}, {part})
    assert (step, count, len(points)) == (1, 2, 1)
    result = events.witness(model, data, q, v, 1, step, points[0])
    assert result["finger_geom"] == model.geom(finger).name
    assert result["part_geom"] == model.geom(part).name
    np.testing.assert_allclose(result["normal_finger_to_part_world"],
                               [1. if finger_on_left else -1., 0., 0.], atol=1e-8)
    assert result["preceding_endpoint_pair_distance_m"] == pytest.approx(.010)
    assert result["signed_distance_m"] == pytest.approx(-.002)
    # The left ball's contact point has y=.03 relative to its hinge.
    # Its x velocity is .3 - 1*.03; the right ball's is .08.
    assert result["recorded_normal_speed_m_s"] == pytest.approx(-.19)
    assert result["previous_recorded_normal_speed_m_s"] == pytest.approx(-.29)


def test_sparse_trace_does_not_invent_velocity_at_contact(native, spheres, tmp_path):
    _, events = native
    model, data, q, v = spheres
    # Embed the moving joints in the actual reach trace format. One decision
    # covers two physics steps; contact first appears between its endpoints.
    full_q = np.zeros((3, 1, 23), dtype=np.float32)
    full_v = np.zeros((3, 1, 22), dtype=np.float32)
    full_q[:, 0, :3], full_v[:, 0, :3] = q, v
    full_q[:, 0, 19] = 1.
    fields = dict(after_qpos=full_q[::2], after_qvel=full_v[::2],
                  after_substep_qpos=full_q[None, 1:],
                  after_substep_qualified=np.zeros((1, 2, 1), dtype=bool),
                  after_reach_max_qualified_dwell_steps=np.zeros(1, dtype=np.int64))
    step, points, _ = events.first_contacts(model, data, q, {model.geom("left_ball").id},
                                            {model.geom("right_ball").id})
    assert step == 1
    contract = dict(action_repeat=2, worlds=1)
    for dense in (False, True):
        if dense:
            fields["after_substep_qvel"] = full_v[None, 1:]
        path = tmp_path / f"recorded-{dense}.npz"
        np.savez(path, **fields)
        qs, vs, stride, _, _ = events.load_states(path, "after", contract)
        assert stride == (1 if dense else 2)
        result = events.witness(model, data, qs[:, 0, :3], vs[:, 0, :3], stride, step, points[0])
        brackets = result["recorded_qvel_brackets"]
        if dense:
            assert result["recorded_normal_speed_m_s"] == pytest.approx(-.19)
            assert [b["step"] for b in brackets] == [1]
        else:
            assert result["recorded_normal_speed_m_s"] is None
            assert [b["step"] for b in brackets] == [0, 2]
            np.testing.assert_allclose([b["speed_m_s"] for b in brackets], [-.29, .10], atol=1e-7)
        assert result["incoming_qpos_difference_estimate_m_s"] == pytest.approx(-1.2)


def test_generated_crossed_banks_pair_arm_poses_across_both_physical_axes(native, tmp_path):
    mj, _ = native
    from panda_reach_crossed_resets import build_scene, generate_bank

    scene = build_scene()
    model = mj.MjModel.from_xml_string(scene.xml)
    data = mj.MjData(model)
    pin, fork = (model.site(name).id for name in ("part_grasp_center", "pickup_grasp_center"))
    seen_arms = set()
    for split, yaw, cross, along in (("crossed", 26., .001, -.009), ("validation", 22., .0008, .007)):
        out = tmp_path / split
        generate_bank(scene, model, split, out)
        metadata = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
        path = out / metadata["file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == metadata["file_sha256"]
        with np.load(path, allow_pickle=False) as bank:
            q = bank["qpos"]
            assert bank["model_sha256"].item() == scene.xml_sha256
            assert not np.any(bank["qvel"])
        grouped = {}
        for pos in q:
            data.qpos[:] = pos
            mj.mj_kinematics(model, data)
            fork_rot = data.site_xmat[fork].reshape(3, 3)
            xyz = fork_rot.T @ (data.site_xpos[pin] - data.site_xpos[fork])
            rel_rot = data.site_xmat[pin].reshape(3, 3) @ fork_rot.T
            measured_yaw = np.rad2deg(np.arctan2(rel_rot[1, 0], rel_rot[0, 0]))
            np.testing.assert_allclose([abs(xyz[0]), xyz[1], xyz[2], abs(measured_yaw)],
                                       [cross, along, 0., yaw], atol=1e-8, rtol=0.)
            cell = (np.sign(xyz[0]), np.sign(measured_yaw))
            grouped.setdefault(cell, []).append(tuple(pos[:14]))
        assert set(grouped) == {(-1, -1), (-1, 1), (1, -1), (1, 1)}
        paired = set(grouped[(-1, -1)])
        assert len(paired) == 3
        assert all(len(rows) == 3 and set(rows) == paired for rows in grouped.values())
        assert paired.isdisjoint(seen_arms)
        seen_arms.update(paired)
