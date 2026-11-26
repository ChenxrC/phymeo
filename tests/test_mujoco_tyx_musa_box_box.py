"""MUSA box face manifolds: native motion and net body wrench, not point ordering.

Four coplanar contacts can have redundant force distributions. Compare their
resultant about each input COM, retaining the established native force/motion
tolerances. No CPU contact implementation may run inside the device step.
"""

import os

import numpy as np
import pytest

from phymeo.backends._mujoco_tyx_runtime import MuJoCoTyxRuntime, MuJoCoTyxUnsupported
from phymeo.backends.mujoco_runtime import MuJoCoControl, MuJoCoInitialState, MuJoCoRuntime
from phymeo.builders.mujoco_compiler import compile_mujoco_xml


def _source(dimension):
    # An eccentric box pad overlaps the free box by 0.5 mm at q=0. The
    # rectangular pad face is wholly inside the payload face. The other world
    # retains the same eligible pair, but has a 59.5 mm separation.
    return f"""<mujoco><compiler angle="radian"/>
      <option timestep=".001" gravity="0 0 0" solver="Newton" cone="pyramidal"
              jacobian="dense" iterations="64" tolerance="1e-8"/>
      <default><geom condim="{dimension}" friction=".55 .005 .001"/></default>
      <worldbody>
        <body name="arm" pos="-.1655 .025 .200">
          <joint name="pitch" axis="0 1 0" range="-1.1 1.1" limited="true" armature=".01"/>
          <geom name="pad" type="box" pos=".10 0 -.14" size=".026 .015 .020" mass=".35"/>
        </body>
        <body name="payload" pos="0 0 .05"><freejoint name="load"/>
          <geom name="box" type="box" size=".04 .060 .045" mass="1.2"/>
        </body>
      </worldbody><actuator><motor joint="pitch"/></actuator></mujoco>"""


def _native_wrenches(mj, model, data):
    """mj_step leaves its contact kinematics at the step input configuration."""
    result = np.zeros((2, 6))
    for index in range(data.ncon):
        contact = data.contact[index]
        local = np.zeros(6)
        mj.mj_contactForce(model, data, index, local)
        rotation = contact.frame.reshape(3, 3).T
        force, couple = rotation @ local[:3], rotation @ local[3:]
        for side, sign in enumerate((-1., 1.)):
            body = model.geom_bodyid[contact.geom[side]]
            result[side, :3] += sign * force
            result[side, 3:] += sign * (couple + np.cross(contact.pos - data.xipos[body], force))
    return result


@pytest.mark.musa
@pytest.mark.skipif(os.environ.get("PHYMEO_TEST_MUSA") != "1", reason="requires a MUSA device")
@pytest.mark.parametrize("dimension", (1, 3), ids=("normal-face", "pyramidal-face"))
def test_box_face_manifold_native_wrench_and_motion(dimension, tmp_path, monkeypatch):
    mj = pytest.importorskip("mujoco")
    from phymeo.backends.mujoco_tyx._device import TyxDevice

    asset = compile_mujoco_xml(_source(dimension), require_asset=False)
    model, h = asset.mj_model, .001
    pitch = int(model.joint("pitch").dofadr[0])
    load = model.joint("load")
    pv, px = int(load.dofadr[0]), int(load.qposadr[0])
    pair = (model.geom("pad").id, model.geom("box").id)
    bodies = tuple(int(model.geom_bodyid[g]) for g in pair)
    states = {}
    for key in (4, 91):
        q, v = model.qpos0.copy(), np.zeros(model.nv)
        if key == 91:
            q[px] += .06
        v[pitch], v[pv + 1] = -.05, -.04
        states[key] = MuJoCoInitialState(q, v)

    lowered = []
    original_kernel = TyxDevice._musa_kernel

    def observed_kernel(self, kernel):
        lowered.append(kernel.name)
        return original_kernel(self, kernel)

    def forbidden(*args, **kwargs):
        pytest.fail("MUSA box contact step called a CPU contact/integration oracle")

    monkeypatch.setattr(TyxDevice, "_musa_kernel", observed_kernel)
    with MuJoCoRuntime(asset, mode="musa", world_keys=tuple(states), initial_states=states,
                       cache_root=tmp_path / "kernels", contact_capacity=24,
                       constraint_capacity=96) as runtime:
        cpu = {}
        for key, state in runtime.integration_snapshot().items():
            data = mj.MjData(model)
            data.qpos[:], data.qvel[:], data.ctrl[:] = state.q, state.v, [-.16]
            cpu[key] = data
        runtime.set_controls({k: MuJoCoControl([-.16]) for k in states})
        start = runtime.musa_device_report()
        initial = {k: (d.qpos.copy(), d.qvel.copy()) for k, d in cpu.items()}
        for tick in range(1, 5):
            for data in cpu.values():
                mj.mj_step(model, data)
            with monkeypatch.context() as guard:
                for name in ("mj_step", "mj_forward", "mj_contactForce"):
                    guard.setattr(mj, name, forbidden)
                assert runtime.step(h) == dict.fromkeys(states, True)
                published = runtime.integration_snapshot()
            for key, state in published.items():
                data, row = cpu[key], state.last_step
                assert row.backend == "mujoco-tyx-musa"
                assert row.contact_time == pytest.approx((tick - 1) * h, abs=2e-7)
                assert state.tick == tick and state.time == pytest.approx(data.time, abs=2e-7)
                np.testing.assert_allclose(state.q, data.qpos, rtol=0, atol=2e-4)
                np.testing.assert_allclose(state.v, data.qvel, rtol=0, atol=3e-3)
                np.testing.assert_allclose(row.qfrc_constraint, data.qfrc_constraint,
                                           rtol=3e-3, atol=3e-3)
                if key == 91:
                    assert data.ncon == len(row.contacts) == 0
                    np.testing.assert_allclose(row.qfrc_constraint, 0., atol=1e-7)
                    continue
                assert data.ncon >= 4 and len(row.contacts) >= 4
                assert all(tuple(data.contact[i].geom) == pair for i in range(data.ncon))
                # Redundant/interior face points are legal, but none may leave
                # the native contact face bounds or change identity/normal.
                rotation = data.geom_xmat[pair[0]].reshape(3, 3)
                native_points = np.array([data.contact[i].pos for i in range(data.ncon)])
                local_points = (native_points - data.geom_xpos[pair[0]]) @ rotation
                native_distances = [data.contact[i].dist for i in range(data.ncon)]
                expected = _native_wrenches(mj, model, data)
                observed = np.zeros((2, 6))
                normal = data.contact[0].frame.reshape(3, 3)[0]
                for contact in row.contacts:
                    assert tuple(contact.geometry_ids) == pair
                    assert tuple(contact.body_ids) == bodies
                    assert contact.dimension == dimension
                    assert min(native_distances) - 4e-5 <= contact.distance <= max(native_distances) + 4e-5
                    point = (contact.position - data.geom_xpos[pair[0]]) @ rotation
                    assert np.all(point >= local_points.min(axis=0) - 4e-5)
                    assert np.all(point <= local_points.max(axis=0) + 4e-5)
                    np.testing.assert_allclose(contact.frame[0], normal, rtol=0, atol=1e-4)
                    np.testing.assert_allclose(contact.force_on_geometry1_world,
                                               -contact.force_on_geometry2_world, rtol=0, atol=1e-7)
                    # condim 1/3 has no independent contact couple. Validate
                    # published COM moments as well as reconstructing a common
                    # input-COM resultant without assuming a point permutation.
                    for side, force in enumerate((contact.force_on_geometry1_world,
                                                  contact.force_on_geometry2_world)):
                        wrench = np.r_[force, np.cross(contact.position - data.xipos[bodies[side]], force)]
                        observed[side] += wrench
                np.testing.assert_allclose(observed[:, :3], expected[:, :3], rtol=3e-3, atol=3e-3)
                np.testing.assert_allclose(observed[:, 3:], expected[:, 3:], rtol=3e-3, atol=3e-4)
                published_wrench = np.array([
                    np.sum([c.wrench_on_body1_world for c in row.contacts], axis=0),
                    np.sum([c.wrench_on_body2_world for c in row.contacts], axis=0),
                ])
                np.testing.assert_allclose(published_wrench[:, :3], expected[:, :3], rtol=3e-3, atol=3e-3)
                np.testing.assert_allclose(published_wrench[:, 3:], expected[:, 3:], rtol=3e-3, atol=3e-4)
                assert expected[1, 0] > .05
                assert np.linalg.norm(expected[1, 3:]) > .001
                assert abs(data.qfrc_constraint[pitch]) > .005
                if dimension == 3:
                    assert np.linalg.norm(expected[1, 1:3]) > .005
        # Prevent an immobile or no-load implementation passing a loose q gate.
        assert np.linalg.norm(cpu[4].qpos - initial[4][0]) > 1e-5
        assert abs(cpu[4].qvel[pv] - initial[4][1][pv]) > 1e-4
        assert np.linalg.norm(published[4].q - initial[4][0]) > 1e-5
        assert abs(published[4].v[pv] - initial[4][1][pv]) > 1e-4
        assert "extended_contacts" in lowered
        end = runtime.musa_device_report()
        assert end["device_name"] and end["arch"] and end["backend"] == "musa"
        assert end["completed_kernels"] - start["completed_kernels"] == end["launched_kernels"] - start["launched_kernels"] >= 4


def test_box_box_condim6_rejected_before_device_allocation(monkeypatch):
    pytest.importorskip("mujoco")
    asset = compile_mujoco_xml(_source(6), require_asset=False)

    def forbidden(*args, **kwargs):
        pytest.fail("unsupported box-box condim 6 reached device allocation")

    monkeypatch.setattr(MuJoCoTyxRuntime, "_build_device", forbidden)
    with pytest.raises(MuJoCoTyxUnsupported, match="musa-domain"):
        MuJoCoRuntime(asset, mode="musa")
