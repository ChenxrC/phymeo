"""Native-input comparison for a driven capsule tip and a moving free box on MUSA."""

import os

import numpy as np
import pytest

from phymeo.backends._mujoco_tyx_runtime import MuJoCoTyxRuntime, MuJoCoTyxUnsupported
from phymeo.backends.mujoco_runtime import MuJoCoControl, MuJoCoInitialState, MuJoCoRuntime
from phymeo.builders.mujoco_compiler import compile_mujoco_xml


pytestmark = [
    pytest.mark.musa,
    pytest.mark.skipif(os.environ.get("PHYMEO_TEST_MUSA") != "1", reason="requires a MUSA device"),
]


def _source(dimension, *, box_tool=False):
    tool = ('type="box" pos=".10 0 -.14" size=".026 .015 .020"' if box_tool else
            'type="capsule" fromto=".020 0 -.090 .125 0 -.150" size=".015"')
    # The inactive hinge-limit candidates also select the existing T4 row
    # assembly for condim 1. The free box has a noncentral point of contact.
    return f"""<mujoco><compiler angle="radian"/>
      <option timestep=".001" gravity="0 0 0" solver="Newton" cone="pyramidal"
              jacobian="dense" iterations="64" tolerance="1e-8"/>
      <default><geom condim="{dimension}" friction=".55 .005 .001"/></default>
      <worldbody>
        <body name="arm" pos="-.179 .025 .200">
          <joint name="pitch" axis="0 1 0" range="-1.1 1.1" limited="true" armature=".01"/>
          <geom name="pad" {tool} mass=".35"/>
        </body>
        <body name="payload" pos="0 0 .05"><freejoint name="load"/>
          <geom name="box" type="box" size=".04 .060 .045" mass="1.2"/>
        </body>
      </worldbody><actuator><motor joint="pitch"/></actuator></mujoco>"""


@pytest.mark.parametrize("dimension", (1, 3), ids=("normal", "pyramidal"))
def test_moving_capsule_tip_and_box_against_native(dimension, tmp_path, monkeypatch):
    mj = pytest.importorskip("mujoco")
    asset = compile_mujoco_xml(_source(dimension), require_asset=False)
    model, h = asset.mj_model, .001
    pitch = model.joint("pitch").dofadr[0]
    load = model.joint("load")
    pv, px = load.dofadr[0], load.qposadr[0]
    active, separated = 4, 91
    states = {}
    for key in (active, separated):
        q, v = model.qpos0.copy(), np.zeros(model.nv)
        if key == separated:
            q[px] += .06
        v[pitch], v[pv+1] = -.05, -.04
        states[key] = MuJoCoInitialState(q, v)
    def forbidden(*args, **kwargs):
        pytest.fail("a MUSA step called the CPU contact or integration oracle")

    with MuJoCoRuntime(asset, mode="musa", world_keys=tuple(states), initial_states=states,
                       cache_root=tmp_path/"kernels", contact_capacity=2, constraint_capacity=8) as runtime:
        cpu = {}
        for key, state in runtime.integration_snapshot().items():
            data = mj.MjData(model)
            data.qpos[:], data.qvel[:], data.ctrl[:] = state.q, state.v, [-.16]
            cpu[key] = data
        runtime.set_controls({k: MuJoCoControl([-.16]) for k in states})
        start = runtime.musa_device_report()
        for tick in range(1, 5):
            # Native mj_step preserves contacts and qfrc_constraint at the step
            # input while qpos and qvel advance. Both native worlds run continuously.
            for data in cpu.values():
                mj.mj_step(model, data)
            with monkeypatch.context() as guard:
                for fn in ("mj_step", "mj_forward", "mj_contactForce"):
                    guard.setattr(mj, fn, forbidden)
                assert runtime.step(h) == dict.fromkeys(states, True)
                published = runtime.integration_snapshot()
            for key, state in published.items():
                data, row = cpu[key], state.last_step
                assert row.backend == "mujoco-tyx-musa"
                assert row.contact_time == pytest.approx((tick-1)*h, abs=2e-7)
                assert state.tick == tick and state.time == pytest.approx(data.time, abs=2e-7)
                np.testing.assert_allclose(state.q, data.qpos, rtol=0, atol=2e-4)
                np.testing.assert_allclose(state.v, data.qvel, rtol=0, atol=3e-3)
                np.testing.assert_allclose(row.qfrc_constraint, data.qfrc_constraint, rtol=3e-3, atol=3e-3)
                assert len(row.contacts) == data.ncon == int(key == active)
                if key == separated:
                    np.testing.assert_allclose(row.qfrc_constraint, 0., atol=1e-7)
                    continue
                c, native = row.contacts[0], data.contact[0]
                assert tuple(c.geometry_ids) == tuple(native.geom) == (model.geom("pad").id, model.geom("box").id)
                assert c.dimension == native.dim == dimension
                frame = native.frame.reshape(3, 3)
                np.testing.assert_allclose(c.position, native.pos, rtol=0, atol=4e-5)
                np.testing.assert_allclose(c.frame[0], frame[0], rtol=0, atol=1e-4)
                wrench = np.zeros(6)
                mj.mj_contactForce(model, data, 0, wrench)
                force = frame.T @ wrench[:3]
                assert force[0] > .05 and abs(data.qfrc_constraint[pitch]) > .005
                if dimension == 3:
                    assert np.linalg.norm(wrench[1:3]) > .005
                np.testing.assert_allclose(c.force_on_geometry2_world, force, rtol=3e-3, atol=3e-3)
                np.testing.assert_allclose(c.force_on_geometry1_world, -force, rtol=3e-3, atol=3e-3)
                # Native body kinematics still describe the same input pose.
                jac = np.zeros((3, model.nv))
                for sign, geom in zip((-1, 1), native.geom, strict=True):
                    local = np.empty_like(jac)
                    mj.mj_jac(model, data, local, None, native.pos, model.geom_bodyid[geom])
                    jac += sign*local
                np.testing.assert_allclose(c.jacobian[0], frame[0] @ jac, rtol=2e-4, atol=5e-5)
        end = runtime.musa_device_report()
        completed = end["completed_kernels"] - start["completed_kernels"]
        assert end["backend"] == "musa" and end["device_name"] and end["arch"]
        assert completed == end["launched_kernels"] - start["launched_kernels"] >= 4


def test_box_tool_box_payload_condim6_rejected_before_allocation(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("box-box condim 6 reached MUSA device allocation")

    # Normal/pyramidal box faces now have their own native-wrench regression;
    # rolling/torsional box contact remains outside the MUSA admission scope.
    asset = compile_mujoco_xml(_source(6, box_tool=True), require_asset=False)
    monkeypatch.setattr(MuJoCoTyxRuntime, "_build_device", forbidden)
    with pytest.raises(MuJoCoTyxUnsupported, match="musa-domain"):
        MuJoCoRuntime(asset, mode="musa")
