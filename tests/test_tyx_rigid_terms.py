"""Original tree terms against native forward, including moving free roots."""

import mujoco as mj
import numpy as np
import pytest

from phymeo.backends.mujoco_tyx._rigid_terms import RigidWorkspace, compile_rigid_model


def scene():
    m = mj.MjModel.from_xml_string("""<mujoco>
      <option gravity="1 -9.81 .4"/>
      <default><joint armature=".003"/><geom type="box" size=".12 .07 .04" mass=".8"/></default>
      <worldbody><body mocap="true" pos=".2 -.1 .3" euler="30 15 20"><geom/></body>
      <body pos=".5 .1 -.2" euler="20 40 10"><freejoint/>
        <inertial pos=".07 -.04 .02" mass="2" diaginertia=".02 .03 .04" euler="10 20 30"/><geom/>
        <body pos=".13 .05 -.04" euler="15 -30 10">
          <joint type="hinge" axis="1 2 -1" pos=".03 -.02 .05" ref="12"/><geom pos=".02 .05 -.03"/>
          <body pos=".11 -.08 .02" euler="-20 10 40">
            <joint type="slide" axis="-.3 .5 1" ref=".02"/><geom pos=".01 -.03 .02"/>
            <body pos=".12 .03 .04"><geom/></body>
          </body></body></body></worldbody></mujoco>""")
    return m


def reference(m, q, v, p, r):
    d = mj.MjData(m)
    d.qpos[:] = q
    d.qvel[:] = v
    d.mocap_pos[:] = p
    d.mocap_quat[:] = r
    mj.mj_forward(m, d)
    mass = np.zeros((m.nv, m.nv))
    mj.mj_fullM(m, mass, d.qM)
    jac = np.zeros((m.nbody, 6, m.nv))
    for b in range(m.nbody):
        mj.mj_jacBodyCom(m, d, jac[b, :3], jac[b, 3:], b)
    return d, mass, jac


@pytest.mark.tyx
@pytest.mark.parametrize("angle", [0.0, -2.4, 4.1, 31.9, -31.9])
def test_moving_tree_pose_mass_coriolis(tyx_mode, tmp_path, angle):
    m = scene()
    d = mj.MjData(m)
    q = d.qpos.copy()
    q[:3] += [0.003, -0.01, 0.02]
    q[3:7] = [0.7, 0.3, -0.1, 0.4]  # deliberately nonunit; input must remain untouched
    q[7] = m.qpos0[7] + angle
    q[8] = -0.1
    v = np.linspace(-0.7, 0.9, m.nv)
    p = d.mocap_pos.copy() + [0.01, 0.02, -0.03]
    r = np.array([[0.5, 0.2, -0.3, 0.1]])
    before = [x.copy() for x in (q, v, p, r)]
    d, mass, jac = reference(m, q, v, p, r)
    with RigidWorkspace(compile_rigid_model(mj, m), mode=tyx_mode, cache_root=tmp_path) as work:
        got = work.evaluate(q[None], v[None], p[None], r[None]).read()
    for value, old in zip((q, v, p, r), before, strict=True):
        np.testing.assert_array_equal(value, old)
    for name, expected in (
        ("positions", d.geom_xpos),
        ("rotations", d.geom_xmat.reshape(-1, 3, 3)),
        ("com", d.xipos),
    ):
        np.testing.assert_allclose(
            got[name][0].astype(float) + got[name + "_low"][0], expected, atol=2e-11, rtol=0
        )
    np.testing.assert_allclose(got["body_jac"][0], jac, atol=2e-6, rtol=0)
    np.testing.assert_allclose(
        got["body_jac"][0].astype(float) + got["body_jac_low"][0], jac, atol=2e-11, rtol=0
    )
    np.testing.assert_allclose(
        got["mass"][0].astype(float) + got["mass_low"][0], mass, atol=2e-11, rtol=0
    )
    assert np.all(np.abs(got["mass"][0] - mass) <= 2e-6 * (1 + np.abs(mass)))
    assert np.all(np.abs(got["bias"][0] - d.qfrc_bias) <= 2e-6 * (1 + np.abs(d.qfrc_bias)))
    np.testing.assert_array_equal(got["mass"][0], got["mass"][0].T)
    np.linalg.cholesky(got["mass"][0])
    assert abs(v @ (got["mass"][0] - mass) @ v) <= 2e-6 * (1 + v @ mass @ v)
    # Nonzero Coriolis is actually exercised, not just static gravity.
    zero, _, _ = reference(m, q, np.zeros(m.nv), p, r)
    assert np.max(np.abs(d.qfrc_bias - zero.qfrc_bias)) > 0.01


@pytest.mark.tyx
def test_lifetime_and_resident_state(tyx_mode, tmp_path):
    m = scene()
    d = mj.MjData(m)
    work = RigidWorkspace(compile_rigid_model(mj, m), mode=tyx_mode, cache_root=tmp_path)
    args = (d.qpos[None], d.qvel[None], d.mocap_pos[None], d.mocap_quat[None])
    result = work.evaluate(*args)
    host = result.read()
    host["mass"][0, 0, 0] = 999  # detached read cannot alter resident data
    with result.device_view() as view:
        assert set(view) == set(work.OUTPUT)
        with pytest.raises(RuntimeError, match="borrowed"):
            work.evaluate(*args)
        with pytest.raises(RuntimeError, match="borrowed"):
            work.close()
    old = work.stats.copy()
    current = work.evaluate_device({k: work._buffers[k] for k in work.inputs})
    assert work.stats["uploaded_bytes"] == old["uploaded_bytes"]
    assert work.stats["downloaded_bytes"] - old["downloaded_bytes"] == 4  # status only
    assert current.read()["mass"][0, 0, 0] != 999
    # Producer reuse cannot change the velocity epoch of an already computed result.
    replacement = np.ones((1, m.nv), np.float32)
    if tyx_mode == "cuda":
        work._buffers["qvel"].copy_from(replacement)
    else:
        work._buffers["qvel"][...] = replacement
    np.testing.assert_array_equal(current.read()["qvel"], d.qvel[None].astype(np.float32))
    saved_q = current.read()["qpos"]
    changed_q = saved_q.copy()
    changed_q[:, 0, 0] += 5
    if tyx_mode == "cuda":
        work._buffers["qpos"].copy_from(changed_q)
    else:
        work._buffers["qpos"][...] = changed_q
    np.testing.assert_array_equal(current.read()["qpos"], saved_q)
    with pytest.raises(RuntimeError, match="expired"):
        result.read()
    with pytest.raises(ValueError, match="schema"):
        work.evaluate_device({})
    with pytest.raises(RuntimeError, match="expired"):
        current.read()
    current = work.evaluate(*args)
    work.close()
    work.close()
    with pytest.raises(RuntimeError, match="expired"):
        current.read()


@pytest.mark.tyx
@pytest.mark.parametrize(
    "bad", ["qnan", "vnan", "zeroquat", "mocapquat", "angle", "wrongshape", "device_dtype"]
)
def test_bad_state_rejects(tyx_mode, tmp_path, bad):
    m = scene()
    d = mj.MjData(m)
    q, v, p, r = (
        d.qpos[None].copy(),
        d.qvel[None].copy(),
        d.mocap_pos[None].copy(),
        d.mocap_quat[None].copy(),
    )
    if bad == "qnan":
        q[0, 0] = np.nan
    if bad == "vnan":
        v[0, 0] = np.nan
    if bad == "zeroquat":
        q[0, 3:7] = 0
    if bad == "mocapquat":
        r[:] = 0
    if bad == "angle":
        q[0, 7] = m.qpos0[7] + 33
    if bad == "wrongshape":
        q = q[:, :-1]
    with RigidWorkspace(compile_rigid_model(mj, m), mode=tyx_mode, cache_root=tmp_path) as work:
        if bad == "device_dtype":
            with pytest.raises(ValueError):
                work.evaluate_device(dict(qpos=q, qvel=v, mocap=r))
        else:
            with pytest.raises(ValueError):
                work.evaluate(q, v, p, r)


@pytest.mark.parametrize(
    "joint", ['<joint type="ball"/>', '<joint axis="1 0 0"/><joint axis="0 1 0"/>']
)
def test_unsupported_tree(joint):
    m = mj.MjModel.from_xml_string(
        f'<mujoco><worldbody><body>{joint}<geom size=".1"/></body></worldbody></mujoco>'
    )
    with pytest.raises(ValueError):
        compile_rigid_model(mj, m)


def test_resource_and_static_copy():
    m = scene()
    cm = compile_rigid_model(mj, m)
    old = cm.static["gravity"].copy()
    m.opt.gravity[:] = 0
    np.testing.assert_array_equal(cm.static["gravity"], old)
    with pytest.raises(ValueError):
        cm.static["gravity"][0] = 99
    with pytest.raises(MemoryError):
        RigidWorkspace(cm, memory_limit_bytes=1)


@pytest.mark.tyx
def test_no_mocap_batch_epoch(tyx_mode, tmp_path):
    m = mj.MjModel.from_xml_string(
        '<mujoco><worldbody><body><joint axis="0 0 1"/><geom size=".1" pos=".2 0 0"/></body></worldbody></mujoco>'
    )
    cm = compile_rigid_model(mj, m)
    q = np.array([[0.4], [-0.9]])
    v = np.array([[0.3], [-0.2]])
    with RigidWorkspace(cm, worlds=2, mode=tyx_mode, cache_root=tmp_path) as work:
        result = work.evaluate(q, v, np.zeros((2, 0, 3)), np.zeros((2, 0, 4)))
        got = result.read()
        for w in range(2):
            d, M, J = reference(m, q[w], v[w], np.zeros((0, 3)), np.zeros((0, 4)))
            np.testing.assert_allclose(
                got["positions"][w].astype(float) + got["positions_low"][w],
                d.geom_xpos,
                atol=2e-11,
                rtol=0,
            )
        # Malformed device state is rejected, and the previous epoch expires.
        bad = np.zeros((2, m.nq, 2), np.float32)
        bad[1, 0, 1] = np.nan
        if tyx_mode == "cuda":
            work._buffers["qpos"].copy_from(bad)
        else:
            work._buffers["qpos"][...] = bad
        with pytest.raises(ValueError, match="device state"):
            work.evaluate_device({k: work._buffers[k] for k in work.inputs})
        with pytest.raises(RuntimeError, match="expired"):
            result.read()
