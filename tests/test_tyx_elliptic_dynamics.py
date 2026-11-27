"""Acceleration operator tests; they do not qualify the original M16 engine."""

from pathlib import Path

import mujoco as mj
import numpy as np
import pytest

from phymeo.backends.mujoco_tyx._elliptic_dynamics import EllipticDynamicsWorkspace


def blank(work):
    return {
        name: np.zeros(shape, dtype)
        for name, (shape, dtype) in work.layouts.items()
        if name in work.INPUTS
    }


@pytest.mark.tyx
def test_scalar_equality_friction_loss_and_limit(tyx_mode, tmp_path):
    with EllipticDynamicsWorkspace(1, 3, 1, 3, mode=tyx_mode, cache_root=tmp_path) as work:
        inputs = blank(work)
        inputs["mass"][0] = np.eye(3)
        inputs["smooth"][0] = [3, -2, -1]
        inputs["scalar_counts"][0] = 3
        inputs["sj"][0] = [[1, 1, 0], [1, 0, 0], [0, 0, 1]]
        inputs["sd"][0] = [10, 5, 20]
        inputs["sl"][0] = [0, 0.2, 0]
        inputs["sk"][0] = [0, 1, 3]
        result = work.solve(inputs)
        raw = result.read()
        np.testing.assert_array_equal(raw["status"], 1)
        np.testing.assert_allclose(
            raw["acc"][0], [2.8 - 8 / 21, -2 - 8 / 21, -1 / 21], atol=2e-6, rtol=0
        )
        with result.device_view() as view:
            assert view["acc"] is work._buffers["acc"]
            with pytest.raises(RuntimeError, match="borrowed"):
                work.solve(inputs)
            with pytest.raises(RuntimeError, match="borrowed"):
                work.close()
        work.solve(inputs)
        with pytest.raises(RuntimeError, match="expired"):
            result.read()


@pytest.mark.tyx
def test_native_sphere_contact_coupled_acceleration(tyx_mode, tmp_path):
    model = mj.MjModel.from_xml_string("""<mujoco><option cone="elliptic" tolerance="1e-12"/>
    <worldbody><geom type="plane" size="1 1 .1"/>
    <body pos="0 0 .099"><freejoint/><geom type="sphere" size=".1" mass="1"/></body>
    </worldbody></mujoco>""")
    data = mj.MjData(model)
    data.qvel[0] = 0.2
    mj.mj_forward(model, data)
    assert data.nefc == 3
    with EllipticDynamicsWorkspace(1, 6, 1, 1, mode=tyx_mode, cache_root=tmp_path) as work:
        inputs = blank(work)
        M = np.zeros((6, 6))
        mj.mj_fullM(model, M, data.qM)
        inputs["mass"][0] = M
        inputs["smooth"][0] = data.qacc_smooth
        inputs["counts"][0] = 1
        inputs["jac"][0] = data.efc_J.reshape(3, 6)
        inputs["aref"][0] = data.efc_aref
        inputs["D"][0] = data.efc_D
        inputs["friction"][0] = data.contact[0].friction[:2]
        inputs["mu"][0] = data.contact[0].mu
        inputs["eligible"][0] = 1
        result = work.solve(inputs).read()
        np.testing.assert_array_equal(result["status"], 1)
        np.testing.assert_allclose(result["acc"][0], data.qacc, rtol=2e-5, atol=2e-5)
        np.testing.assert_allclose(result["force"][0], data.efc_force, rtol=2e-5, atol=2e-5)


@pytest.mark.tyx
@pytest.mark.parametrize("bad", ["count", "mass", "jacobian", "eligibility", "kind", "loss"])
def test_invalid_dynamics_never_publish_to_consumer(bad, tyx_mode, tmp_path):
    with EllipticDynamicsWorkspace(1, 3, 1, 1, mode=tyx_mode, cache_root=tmp_path) as work:
        inputs = blank(work)
        inputs["mass"][0] = np.eye(3)
        inputs["counts"][0] = 1
        inputs["eligible"][0] = 1
        inputs["D"][0] = 1
        inputs["mu"][0] = 1
        inputs["friction"][0] = 1
        inputs["scalar_counts"][0] = 1
        inputs["sd"][0] = 1
        if bad == "count":
            inputs["counts"][0] = 2
        elif bad == "mass":
            inputs["mass"][0, 0, 0] = -1
        elif bad == "jacobian":
            inputs["jac"][0, 0, 0] = np.nan
        elif bad == "eligibility":
            inputs["eligible"][0] = 0
        elif bad == "kind":
            inputs["sk"][0] = 9
        else:
            inputs["sl"][0] = np.nan
        result = work.solve(inputs)
        assert result.read()["status"][0] != 1
        with pytest.raises(ValueError, match="unconverged/invalid"):
            with result.device_view():
                pass


def test_workspace_preflight():
    with pytest.raises(MemoryError):
        EllipticDynamicsWorkspace(8, 15, 600, 3, memory_limit_bytes=4)
    with pytest.raises(ValueError):
        EllipticDynamicsWorkspace(1, 36, 1, 1)


@pytest.mark.tyx
def test_saved_high_acceleration_requires_retained_low_bits(tyx_mode, tmp_path):
    with np.load(Path(__file__).parent / "fixtures/m16_coupled_acceleration.npz") as fixture:
        inputs = {name: fixture[name].copy() for name in EllipticDynamicsWorkspace.INPUTS}
        native = fixture["native_acc"].copy()
    cap = len(inputs["D"]) // 2
    with EllipticDynamicsWorkspace(2, 15, cap, 3, mode=tyx_mode, cache_root=tmp_path) as work:
        result = work.solve(inputs)
        raw = result.read()
        np.testing.assert_array_equal(raw["status"], 1)
        assert np.max(raw["balance"]) <= 1e-6
        reconstructed = raw["acc"].astype(float) + raw["acc_low"].astype(float)
        np.testing.assert_allclose(reconstructed, native, rtol=5e-4, atol=0.005)
        assert np.any(raw["acc_low"] != 0)
        np.testing.assert_array_equal(raw["result_counts"], inputs["counts"])
        with result.device_view() as view:
            assert view["acc_low"] is work._buffers["acc_low"]
        # Independent consumers must not replace the paired acceleration with
        # a single rounded array; the full-scene precision-floor artifact
        # demonstrates why, without turning it into a looser physics gate.
        inputs["smooth_low"][0, 0] = np.nan
        failed = work.solve(inputs)
        assert failed.read()["status"][0] != 1
        with pytest.raises(ValueError, match="unconverged/invalid"):
            with failed.device_view():
                pass


@pytest.mark.tyx
def test_iteration_exhaustion_is_not_success(tyx_mode, tmp_path):
    with np.load(Path(__file__).parent / "fixtures/m16_coupled_acceleration.npz") as fixture:
        inputs = {name: fixture[name].copy() for name in EllipticDynamicsWorkspace.INPUTS}
    with EllipticDynamicsWorkspace(
        2, 15, len(inputs["D"]) // 2, 3, iterations=1, mode=tyx_mode, cache_root=tmp_path
    ) as work:
        result = work.solve(inputs)
        assert np.all(result.read()["status"] != 1)
        with pytest.raises(ValueError, match="unconverged/invalid"):
            with result.device_view():
                pass
