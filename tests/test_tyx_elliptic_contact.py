"""Actual CUDA constitutive force/cost/Hessian against the native cone operator."""

import mujoco as mj
import numpy as np
import pytest

from phymeo.backends.mujoco_tyx._elliptic_contact import EllipticContactWorkspace


def native_inputs():
    model = mj.MjModel.from_xml_string("""<mujoco><option cone="elliptic" impratio="2"/>
      <worldbody><geom type="plane" size="1 1 .1" friction=".8 .01 .01"/>
      <body pos="0 0 .095"><freejoint/><geom type="sphere" size=".1" mass="1" friction=".8 .01 .01"/></body>
      </worldbody></mujoco>""")
    data = mj.MjData(model)
    mj.mj_forward(model, data)
    assert data.nefc == 3 and data.contact[0].dim == 3
    residual = np.array(
        [[2, 0, 0], [-2, 0, 0], [-0.1, 4, 1], [0.2, -4, 3], [0, 0, 0], [-3, 0.01, 0.02]], np.float32
    )
    count = len(residual)
    D = np.repeat(data.efc_D[None].astype(np.float32), count, axis=0)
    friction = np.repeat(np.asarray(data.contact[0].friction[:2], np.float32)[None], count, axis=0)
    mu = np.full(count, data.contact[0].mu, np.float32)
    expected = []
    for row in residual:
        cost = np.zeros(1)
        mj.mj_constraintUpdate(model, data, row.astype(float), cost, 1)
        state = int(data.efc_state[0])
        hessian = np.zeros((3, 3))
        if state == 1:
            hessian = np.diag(data.efc_D)
        elif state == 4:
            hessian = np.asarray(data.contact[0].H).reshape(-1)[:9].reshape(3, 3).copy()
        expected.append((data.efc_force.copy(), hessian, float(cost[0]), state))
    return (residual, D, friction, mu, np.ones(count, np.int32)), expected


@pytest.mark.tyx
def test_native_cone_force_hessian_cost(tyx_mode, tmp_path):
    inputs, expected = native_inputs()
    with EllipticContactWorkspace(len(inputs[0]), mode=tyx_mode, cache_root=tmp_path) as w:
        first = w.evaluate(*inputs)
        result = first.read()
        assert not result["force"].flags.writeable
        np.testing.assert_array_equal(result["status"], 0)
        for row, (force, hessian, cost, state) in enumerate(expected):
            np.testing.assert_allclose(result["force"][row], force, rtol=3e-5, atol=2e-5)
            np.testing.assert_allclose(result["hessian"][row], hessian, rtol=3e-5, atol=2e-5)
            assert result["cost"][row] == pytest.approx(cost, rel=3e-5, abs=2e-5)
            assert result["state"][row] == state
            np.testing.assert_allclose(result["hessian"][row], result["hessian"][row].T, atol=1e-6)
            assert np.linalg.eigvalsh(result["hessian"][row].astype(float)).min() > -2e-4
            assert (
                np.linalg.norm(result["force"][row, 1:] / inputs[2][row])
                <= result["force"][row, 0] + 2e-4
            )
        w.evaluate(*inputs)
        with pytest.raises(RuntimeError, match="expired"):
            first.read()
    with pytest.raises(RuntimeError, match="closed"):
        first.read()


@pytest.mark.tyx
def test_invalid_inputs_do_not_publish_forces(tyx_mode, tmp_path):
    inputs, _ = native_inputs()
    residual, D, friction, mu, eligible = (x.copy() for x in inputs)
    residual[0, 0] = np.nan
    D[1, 1] = -1
    friction[2, 0] = 0
    mu[3] = 0
    eligible[4] = 0
    D[5, 1] *= 2
    with EllipticContactWorkspace(len(residual), mode=tyx_mode, cache_root=tmp_path) as w:
        raw = w.evaluate(residual, D, friction, mu, eligible).read()
        assert np.all(raw["status"] != 0)
        np.testing.assert_array_equal(raw["force"], 0)
        np.testing.assert_array_equal(raw["hessian"], 0)
        np.testing.assert_array_equal(raw["state"], -1)


def test_response_capacity_before_allocation():
    with pytest.raises(MemoryError):
        EllipticContactWorkspace(1024, memory_limit_bytes=4)
    with pytest.raises(ValueError):
        EllipticContactWorkspace(True)
    with pytest.raises(ValueError):
        EllipticContactWorkspace(0)
