"""Native contact wrench conventions and continuous load-transfer evidence."""

from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def native(monkeypatch):
    mj = pytest.importorskip("mujoco")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    from _panda_pickup_loads import PickupLoads

    return mj, PickupLoads


def _fixture(native, *, part_sphere):
    mj, pickup = native
    # Deliberately tilt gravity onto -X. Sphere/box interchange exercises
    # both native contact orderings with the same physical load on the part.
    part = 'type="sphere" size=".02"' if part_sphere else 'type="box" size=".02 .02 .02"'
    rail = 'type="box" size=".05 .08 .08"' if part_sphere else 'type="sphere" size=".05"'
    model = mj.MjModel.from_xml_string(f"""
      <mujoco><option timestep=".002" gravity="-9.81 0 0"/>
      <worldbody>
        <site name="right_grasp_center" pos=".03 0 0"/>
        <body pos="0 1 0"><joint name="right_panda_finger_joint1" type="slide" axis="0 1 0"/>
          <geom name="right_left_pad" type="sphere" size=".01"/></body>
        <body pos="0 -1 0"><joint name="right_panda_finger_joint2" type="slide" axis="0 1 0"/>
          <geom name="right_right_pad" type="sphere" size=".01"/></body>
        <geom name="pickup_rail_left" {rail} pos="-.05 0 0"/>
        <geom name="pickup_rail_right" type="sphere" size=".01" pos="0 2 0"/>
        <body name="assembly_part" pos=".0199 0 0"><freejoint/>
          <geom name="part_handle" {part} mass=".114"/>
          <geom name="part_shoulder" type="sphere" pos=".03 0 0" size=".001" mass=".003"/>
          <geom name="assembly_pin" type="sphere" pos=".04 0 0" size=".001" mass=".003"/>
          <site name="part_grasp_center"/>
        </body>
      </worldbody></mujoco>""")
    return model, mj.MjData(model), pickup(model)


def test_solved_contact_force_is_on_the_part_and_opposes_actual_gravity(native):
    mj, _ = native
    part_contact_sides = set()
    for sphere in (True, False):
        model, data, loads = _fixture(native, part_sphere=sphere)
        for _ in range(350):
            mj.mj_step(model, data)
        mj.mj_forward(model, data)
        row = loads.sample(data)
        assert data.ncon == 1
        handle = model.geom("part_handle").id
        part_contact_sides.add(list(data.contact[0].geom).index(handle))
        assert loads.weight_n == pytest.approx(.120 * 9.81)
        expected = np.zeros((5, 3, 3))
        expected[2, 0, 0] = loads.weight_n
        np.testing.assert_allclose(row["force_world_n"], expected, atol=1e-5)
        np.testing.assert_allclose(row["normal_world_n"], expected, atol=1e-5)
        np.testing.assert_allclose(row["up_n"], expected[:, :, 0], atol=1e-5)
        np.testing.assert_allclose(row["normal_n"], expected[:, :, 0], atol=1e-5)
        assert row["contact_count"][2, 0] == 1
    assert part_contact_sides == {0, 1}


def test_score_requires_same_grip_to_leave_fork_and_follow_hand(native):
    mj, _ = native
    model, data, loads = _fixture(native, part_sphere=True)
    mj.mj_forward(model, data)
    sample = loads.sample(data)
    steps = 180
    d = {key: np.repeat(np.asarray(val)[None], steps, axis=0) for key, val in sample.items()}
    t = d["time_s"] = np.arange(steps) * model.opt.timestep
    d["normal_n"][:] = 0.
    d["up_n"][:] = 0.
    w = loads.weight_n
    # Construct analytical records to test the temporal gate independently
    # of solver tuning. Native force extraction is exercised above.
    d["normal_n"][:20, 2, 1] = d["up_n"][:20, 2, 1] = w
    d["normal_n"][10:, :2, 0] = 2.*w
    d["up_n"][20:, :2, 0] = .5*w
    d["force_world_n"][:] = d["up_n"][..., None] * loads.up
    rise = np.clip((t-t[20])/.10, 0., 1.) * .030
    for key in ("part_xyz", "wrist_xyz"):
        d[key] += rise[:, None] * loads.up
    score = loads.score(d)
    assert score["success"]
    assert score["contact_source"] == "native_mujoco_cpu"
    with pytest.raises(ValueError, match="contact_source"):
        loads.score(d, contact_source="musa_device_same_solve")
    unlabeled = {k: v for k, v in d.items() if k != "contact_solver_code"}
    with pytest.raises(ValueError, match="contact_source"):
        loads.score(unlabeled)
    labeled = loads.score(unlabeled, contact_source="musa_device_same_solve")
    assert labeled["loaded_windows"] == score["loaded_windows"]
    assert labeled["contact_source"] == "musa_device_same_solve"
    assert score["first_bilateral_still_on_fork_s"] == pytest.approx(t[10])
    assert score["first_fork_unloaded_after_support_s"] == pytest.approx(t[20])
    assert score["loaded_windows"][0]["peak_slip_since_grasp_m"] == pytest.approx(0.)

    def copy():
        return {k: v.copy() for k, v in d.items()}

    single = copy()
    single["normal_n"][:, 1] = single["up_n"][:, 1] = single["force_world_n"][:, 1] = 0.
    # The remaining pad can supply the entire weight yet cannot prove a
    # bilateral grasp. Leave the part on the fork in this negative trajectory.
    single["up_n"][20:, 0, 0] = w
    single["up_n"][20:, 2, 1] = single["normal_n"][20:, 2, 1] = w
    single["force_world_n"][:] = single["up_n"][..., None] * loads.up
    single["part_xyz"][:] = single["part_xyz"][0]
    single["part_in_hand_m"][:] = single["part_xyz"] - single["wrist_xyz"]
    comparison = loads.compare(d, single, control_mode="one_sided")
    assert comparison["success"] and not comparison["negative"]["success"]

    for failure in ("second_pad_only_touches_shoulder", "fork_still_loaded", "slip_then_recapture", "direct_force"):
        trial = copy()
        if failure == "second_pad_only_touches_shoulder":
            trial["normal_n"][:, 1, [0, 1]] = trial["normal_n"][:, 1, [1, 0]]
        elif failure == "fork_still_loaded":
            trial["up_n"][20:, 2, 1] = trial["normal_n"][20:, 2, 1] = w
            trial["force_world_n"][:] = trial["up_n"][..., None] * loads.up
        elif failure == "slip_then_recapture":
            trial["part_in_hand_m"][40, 1] += .012
        else:
            trial["part_noncontact_applied"][40, 0] = w
        assert not loads.score(trial)["success"], failure
