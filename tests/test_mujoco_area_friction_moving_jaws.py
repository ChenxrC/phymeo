"""Movable jaws: independently verify the accepted CUDA trajectory in both materials."""

import json

import numpy as np
import pytest

from experiments.area_pressure.moving_jaw_grasp import H, run


def _check_step(r):
    import mujoco

    read = r["wrenches"]
    forces = np.array([read[n]["force"] for n in ("object", "left", "right")])
    moments = np.array([read[n]["moment"] for n in ("object", "left", "right")])
    np.testing.assert_allclose(forces.sum(axis=0), 0.0, atol=2e-7)
    np.testing.assert_allclose(moments.sum(axis=0), 0.0, atol=2e-8)
    assert r["native_rows"] == 0
    q0, q = np.array(r["input_q"]), np.array(r["q"])
    v0, v = np.array(r["input_v"]), np.array(r["v"])
    rot = np.empty(9)
    mujoco.mju_quat2Mat(rot, q0[3:7])
    torque = rot.reshape(3, 3).T @ moments[0]
    jaw_forces = np.array([forces[1, 0], -forces[2, 0]])
    np.testing.assert_allclose(r["q_contact"], [*forces[0], *torque, *jaw_forces], atol=3e-6)
    # Independent Newton/Euler balance about the free box COM and both inward slides.
    linear_residual = 0.3 * (v[:3] - v0[:3]) - H * (forces[0] + (0, 0, r["load_z_n"]))
    inertia = np.array([0.00005, 0.00005, 0.00008])
    angular = inertia * (v[3:6] - v0[3:6]) / H + np.cross(v0[3:6], inertia * v0[3:6])
    np.testing.assert_allclose(angular, torque, atol=2e-6)
    slide_balance = 0.05 * (v[6:] - v0[6:]) / H + 8 * v[6:] - r["drive_n"] - jaw_forces
    np.testing.assert_allclose(slide_balance, 0.0, atol=5e-5)
    # The live solver bounds impulse in the inverse-response norm. Reconstruct
    # its physical inputs from the fixture and three observed body wrenches.
    response = np.r_[[0.3] * 3, inertia, [0.05 + H * 8] * 2]
    residual = np.r_[linear_residual, H * (angular - torque), H * slide_balance]
    momentum_free = np.r_[
        0.3 * v0[:3] + H * np.array((0, 0, r["load_z_n"])),
        inertia * v0[3:6] - H * np.cross(v0[3:6], inertia * v0[3:6]),
        0.05 * v0[6:] + H * r["drive_n"],
    ]
    scale = 1 + np.linalg.norm(momentum_free / np.sqrt(response))
    assert np.linalg.norm(residual / np.sqrt(response)) <= 2e-5 * scale
    assert np.all(np.abs(residual / response) <= 1e-5 * (1 + np.abs(v)))
    np.testing.assert_allclose(q[:3], q0[:3] + H * v[:3], atol=1e-8)
    np.testing.assert_allclose(q[7:], q0[7:] + H * v[6:], atol=1e-8)
    expected_rot = q0[3:7].copy()
    mujoco.mju_quatIntegrate(expected_rot, v[3:6], H)
    np.testing.assert_allclose(q[3:7], expected_rot, atol=1e-6)
    for index, name in enumerate(("left", "right")):
        fn = read[name]["normal"][0]
        assert fn == pytest.approx(read["object"]["normal"][index], abs=2e-7)
        assert -jaw_forces[index] == pytest.approx(fn, abs=2e-6)
        if not read[name]["samples"]:
            np.testing.assert_array_equal(read[name]["force"], 0)


@pytest.mark.cuda
def test_moving_jaws_clamp_load_and_open(tmp_path, monkeypatch):
    import mujoco

    def no_cpu_step(*args, **kwargs):
        pytest.fail("the full trajectory must use CUDA dynamics")

    monkeypatch.setattr(mujoco, "mj_step", no_cpu_step)
    result = run()
    (tmp_path / "grasp.json").write_text(json.dumps(result, allow_nan=False), encoding="utf-8")
    slip = {}
    for name, rows in result["records"].items():
        assert [r["tick"] for r in rows] == list(range(1, 36))
        assert [r["phase"] for r in rows] == ["close"] * 15 + ["load"] * 8 + ["open"] * 12
        for prev, cur in zip(rows, rows[1:]):
            np.testing.assert_array_equal(prev["q"], cur["input_q"])
            np.testing.assert_array_equal(prev["v"], cur["input_v"])
        for row in rows:
            _check_step(row)
        first = rows[0]
        assert first["wrenches"]["object"]["samples"] == 0
        np.testing.assert_allclose(first["v"][6:], H * 0.45 / (0.05 + H * 8), atol=2e-7)
        assert min(rows[14]["q"][7:]) > 0.00025
        loaded = rows[15:23]
        for r in loaded:
            assert min(r["wrenches"]["object"]["normal"]) > 0.2
            assert all(r["wrenches"][j]["samples"] > 0 for j in ("left", "right"))
        applied_impulse = H * sum(r["load_z_n"] for r in loaded)
        support_impulse = H * sum(r["wrenches"]["object"]["force"][2] for r in loaded)
        delta_p = 0.3 * (loaded[-1]["v"][2] - loaded[0]["input_v"][2])
        assert abs(delta_p - applied_impulse - support_impulse) < 0.01 * abs(applied_impulse)
        slip[name] = loaded[0]["input_q"][2] - loaded[-1]["q"][2]
        if name == "smooth":
            for r in loaded:
                assert abs(r["wrenches"]["object"]["force"][2]) < 1e-7
            ballistic = 0.22 / 0.3 * H**2 * 8 * 9 / 2 - 8 * H * loaded[0]["input_v"][2]
            assert slip[name] == pytest.approx(ballistic, abs=1e-8)
        else:
            for jaw in ("left", "right"):
                assert loaded[-1]["wrenches"][jaw]["force"][2] < -0.05
            assert loaded[-1]["wrenches"]["object"]["force"][2] > 0.1
        for r in rows[-3:]:
            assert r["load_z_n"] == -0.22
            assert r["wrenches"]["object"]["samples"] == 0
            np.testing.assert_array_equal(r["q_contact"], 0.0)
            assert (r["v"][2] - r["input_v"][2]) / H == pytest.approx(-0.22 / 0.3, abs=2e-5)
        assert max(rows[-1]["q"][7:]) < 0.00025
        assert rows[-1]["time_s"] == pytest.approx(0.14, abs=1e-7)
    assert 0 < slip["grip"] < 0.5 * slip["smooth"]
