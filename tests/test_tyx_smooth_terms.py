"""Native fixed-state force/scalar reference; no integration or learned data."""

import mujoco as mj
import numpy as np
import pytest

from phymeo.backends.mujoco_tyx._rigid_terms import RigidWorkspace, compile_rigid_model
from phymeo.backends.mujoco_tyx._smooth_terms import SmoothWorkspace, compile_smooth_model


def scene(direct=False, constant=False):
    ref = "-200 -12" if direct else ".0001 .8"  # exercises refsafety
    second = "" if constant else 'joint2="b"'
    return mj.MjModel.from_xml_string(f'''<mujoco>
      <compiler angle="radian"/><option gravity=".2 -9.81 .4" timestep=".001" jacobian="dense"/>
      <default><geom size=".08 .06 .04" type="box" mass=".7" contype="0" conaffinity="0"/>
       <joint damping=".4" armature=".02" stiffness="1.3" springref=".03"
        solreflimit="{ref}" solimplimit=".7 .93 .3 .4 3"/>
      </default><worldbody>
       <body pos=".1 .3 -.2"><joint name="a" axis="1 2 -.4" ref=".1" range="-.3 .6" margin=".04"
         actuatorfrcrange="-1.2 1.4" frictionloss=".2" solreffriction="{ref}" solimpfriction=".7 .93 .3 .4 3"/>
         <geom pos=".1 .02 -.03"/>
        <body pos=".2 -.1 .1"><joint name="b" type="slide" axis=".3 .2 1" ref="-.02" range="-.2 .2" margin=".03"/>
         <geom pos=".04 .02 .01"/>
         <body pos=".1 .1 .1"><joint name="c" axis="1 0 1" frictionloss=".1"/>
          <geom pos=".04 -.02 .01"/></body></body></body></worldbody>
      <equality><joint joint1="a" {second} polycoef=".01 .7 -.2 .3 -.1" solref="{ref}" solimp=".7 .93 .3 .4 3"/></equality>
      <actuator>
       <general joint="a" gear="-2" gainprm="2" biastype="affine" biasprm=".3 -1 -.4" ctrlrange="-.2 .3" forcerange="-.7 .8"/>
       <motor joint="a" gear="1.5" forcerange="-2 2"/>
       <general joint="b" gear=".7" gainprm="3" biastype="affine" biasprm=".2 -.8 -.3"/>
      </actuator></mujoco>''')


def arguments(m, position=0, active=1):
    d = mj.MjData(m)
    d.qpos[:] = ([0.2, 0.04, -0.1], [-0.32, 0.23, 0.4], [0.63, -0.22, -0.3])[position]
    d.qvel[:] = [0.4, -0.5, 0.6]
    d.ctrl[:] = [3.0, -2.0, 0.6] if position != 2 else [-3.0, 2.0, -0.6]
    d.qfrc_applied[:] = [0.3, -0.1, 0.2]
    d.xfrc_applied[:] = np.arange(m.nbody * 6).reshape(m.nbody, 6) * 0.01 - 0.08
    d.eq_active[:] = active
    mj.mj_forward(m, d)
    rigid = (
        d.qpos[None].copy(),
        d.qvel[None].copy(),
        d.mocap_pos[None].copy(),
        d.mocap_quat[None].copy(),
    )
    force = (
        d.ctrl[None].copy(),
        d.qfrc_applied[None].copy(),
        d.xfrc_applied[None].copy(),
        d.eq_active[None].copy(),
    )
    return d, rigid, force


def check(m, d, got):
    def near(value, ref):
        assert np.all(np.abs(value - ref) <= 2e-5 * (1 + np.abs(ref))), (value, ref)

    near(got["actuator_force"][0], d.actuator_force)
    near(got["actuator"][0].astype(float) + got["actuator_low"][0], d.qfrc_actuator)
    near(got["passive"][0], d.qfrc_passive)
    external = d.qfrc_applied.copy()
    for b in range(1, m.nbody):
        jp = np.zeros((3, m.nv))
        jr = jp.copy()
        mj.mj_jacBodyCom(m, d, jp, jr, b)
        external += jp.T @ d.xfrc_applied[b, :3] + jr.T @ d.xfrc_applied[b, 3:]
    near(got["external"][0], external)
    rhs = d.qfrc_passive - d.qfrc_bias + external + d.qfrc_actuator
    near(got["rhs"][0].astype(float) + got["rhs_low"][0], rhs)
    near(got["smooth"][0].astype(float) + got["smooth_low"][0], d.qacc_smooth)
    n = int(got["scalar_counts"][0])
    assert n == d.nefc
    np.testing.assert_array_equal(got["sk"][0, :n], d.efc_type)
    np.testing.assert_array_equal(got["owner"][0, :n], d.efc_id)
    np.testing.assert_allclose(got["sj"][0, :n], d.efc_J.reshape(n, m.nv), rtol=0, atol=2e-6)
    for key, ref in [
        ("sa", d.efc_aref),
        ("sl", d.efc_frictionloss),
        ("position", d.efc_pos),
        ("margin", d.efc_margin),
        ("velocity", d.efc_vel),
    ]:
        near(got[key][0, :n], ref)
    np.testing.assert_allclose(got["sd"][0, :n], d.efc_D, rtol=2e-5, atol=0)
    assert got["balance"][0] <= 2e-6


@pytest.mark.tyx
@pytest.mark.parametrize(
    "position,active,direct,constant",
    [
        (0, 1, False, False),
        (1, 1, False, False),
        (2, 1, True, False),
        (1, 0, True, False),
        (2, 1, False, True),
        (0, 0, False, True),
    ],
)
def test_native_forces_and_scalar_rows(tyx_mode, tmp_path, position, active, direct, constant):
    m = scene(direct, constant)
    d, rargs, fargs = arguments(m, position, active)
    rm = compile_rigid_model(mj, m)
    sm = compile_smooth_model(mj, m, rm)
    with (
        RigidWorkspace(rm, mode=tyx_mode, cache_root=tmp_path / "rigid") as rw,
        SmoothWorkspace(sm, mode=tyx_mode, cache_root=tmp_path / "smooth") as sw,
    ):
        got = sw.evaluate(rw.evaluate(*rargs), *fargs).read()
        check(m, d, got)
    assert abs(d.qfrc_actuator[0]) in (1.2, 1.4)  # combined joint clamp exercised
    if position:
        assert np.count_nonzero(d.efc_type == mj.mjtConstraint.mjCNSTR_LIMIT_JOINT) == 2
    assert np.max(np.abs(d.qfrc_passive)) > 0.1
    if active and not constant:
        assert abs(d.efc_J.reshape(d.nefc, m.nv)[0, 1]) > 0.5


@pytest.mark.tyx
def test_resident_snapshot_lifetime_and_schema(tyx_mode, tmp_path):
    m = scene()
    d, rargs, fargs = arguments(m)
    rm = compile_rigid_model(mj, m)
    sm = compile_smooth_model(mj, m, rm)
    with (
        RigidWorkspace(rm, mode=tyx_mode, cache_root=tmp_path / "rigid") as rw,
        SmoothWorkspace(sm, mode=tyx_mode, cache_root=tmp_path / "smooth") as sw,
    ):
        rigid = rw.evaluate(*rargs)
        first = sw.evaluate(rigid, *fargs)
        first.read()["mass"][:] = 999
        before = sw.stats.copy()
        current = sw.evaluate_device(rigid, {k: sw._buffers[k] for k in sw.inputs})
        assert sw.stats["uploaded_bytes"] == before["uploaded_bytes"]
        assert sw.stats["downloaded_bytes"] - before["downloaded_bytes"] == 4
        with pytest.raises(RuntimeError, match="expired"):
            first.read()
        check(m, d, current.read())
        with current.device_view():
            with pytest.raises(RuntimeError, match="borrowed"):
                sw.evaluate(rigid, *fargs)
            with pytest.raises(RuntimeError, match="borrowed"):
                sw.close()
        # The new result owns M and force/row arrays, so upstream reuse is safe.
        changed = list(rargs)
        changed[0] = changed[0] + 0.01
        rw.evaluate(*changed)
        check(m, d, current.read())
        with pytest.raises(ValueError, match="schema"):
            sw.evaluate_device(rigid, {})
        with pytest.raises(RuntimeError, match="expired"):
            current.read()
        with pytest.raises(RuntimeError, match="expired"):
            sw.evaluate(rigid, *fargs)
        rigid = rw.evaluate(*rargs)
        other = RigidWorkspace(
            compile_rigid_model(mj, m), mode=tyx_mode, cache_root=tmp_path / "rigid"
        )
        try:
            with pytest.raises(ValueError, match="same original"):
                sw.evaluate(other.evaluate(*rargs), *fargs)
        finally:
            other.close()


@pytest.mark.tyx
@pytest.mark.parametrize(
    "name,index,value",
    [
        ("ctrl", (0, 0, 1), np.nan),
        ("applied", (0, 1, 0), np.inf),
        ("body_force", (0, 0, 0, 1), np.nan),
        ("body_force", (0, 2, 3, 0), np.nan),
        ("eq_active", (0, 0), 2),
    ],
)
def test_bad_resident_input_rejected(tyx_mode, tmp_path, name, index, value):
    m = scene()
    _, rargs, fargs = arguments(m)
    rm = compile_rigid_model(mj, m)
    with (
        RigidWorkspace(rm, mode=tyx_mode, cache_root=tmp_path / "rigid") as rw,
        SmoothWorkspace(
            compile_smooth_model(mj, m, rm), mode=tyx_mode, cache_root=tmp_path / "smooth"
        ) as sw,
    ):
        rr = rw.evaluate(*rargs)
        old = sw.evaluate(rr, *fargs)
        arr = sw._buffers[name]
        host = arr.to_numpy() if tyx_mode == "cuda" else arr.copy()
        host[index] = value
        if tyx_mode == "cuda":
            arr.copy_from(host)
        else:
            arr[...] = host
        with pytest.raises(ValueError, match="device state"):
            sw.evaluate_device(rr, {k: sw._buffers[k] for k in sw.inputs})
        with pytest.raises(RuntimeError, match="expired"):
            old.read()


@pytest.mark.tyx
def test_empty_actuation_constraints(tyx_mode, tmp_path):
    m = mj.MjModel.from_xml_string(
        '<mujoco><worldbody><body><joint/><geom size=".1" pos=".2 0 0"/></body></worldbody></mujoco>'
    )
    d = mj.MjData(m)
    mj.mj_forward(m, d)
    rm = compile_rigid_model(mj, m)
    with (
        RigidWorkspace(rm, mode=tyx_mode, cache_root=tmp_path / "rigid") as rw,
        SmoothWorkspace(
            compile_smooth_model(mj, m, rm), mode=tyx_mode, cache_root=tmp_path / "smooth"
        ) as sw,
    ):
        rr = rw.evaluate(d.qpos[None], d.qvel[None], d.mocap_pos[None], d.mocap_quat[None])
        got = sw.evaluate(
            rr, d.ctrl[None], d.qfrc_applied[None], d.xfrc_applied[None], d.eq_active[None]
        ).read()
        assert got["scalar_counts"][0] == 0
        np.testing.assert_allclose(got["smooth"][0], d.qacc_smooth, atol=1e-7)


@pytest.mark.tyx
def test_host_validation_and_corrupted_mass(tyx_mode, tmp_path):
    m = scene()
    _, rargs, fargs = arguments(m)
    rm = compile_rigid_model(mj, m)
    with (
        RigidWorkspace(rm, mode=tyx_mode, cache_root=tmp_path / "rigid") as rw,
        SmoothWorkspace(
            compile_smooth_model(mj, m, rm), mode=tyx_mode, cache_root=tmp_path / "smooth"
        ) as sw,
    ):
        rr = rw.evaluate(*rargs)
        for index in range(4):
            broken = [a.copy() for a in fargs]
            broken[index] = broken[index][:, :0]
            with pytest.raises(ValueError, match="shape"):
                sw.evaluate(rr, *broken)
        broken = [a.copy() for a in fargs]
        broken[3][0, 0] = 2
        with pytest.raises(ValueError, match="binary"):
            sw.evaluate(rr, *broken)
        for name in ["mass", "mass_low"]:
            rr = rw.evaluate(*rargs)
            with rr.device_view() as view:
                buf = view[name]
                bad = buf.to_numpy() if tyx_mode == "cuda" else buf.copy()
                bad[0, 0, 0] = np.nan if name == "mass_low" else -1
                if tyx_mode == "cuda":
                    buf.copy_from(bad)
                else:
                    buf[...] = bad
            with pytest.raises(ValueError, match="device state"):
                sw.evaluate(rr, *fargs)


@pytest.mark.parametrize(
    "bad",
    [
        "gain",
        "bias",
        "transmission",
        "flags",
        "fluid",
        "gravcomp",
        "eq",
        "mixed_ref",
        "identity",
        "negative",
    ],
)
def test_unsupported_semantics_rejected(bad):
    m = scene()
    rm = compile_rigid_model(mj, m)
    if bad == "gain":
        m.actuator_gaintype[0] = 1
    if bad == "bias":
        m.actuator_biastype[0] = 2
    if bad == "transmission":
        m.actuator_trntype[0] = 4
    if bad == "flags":
        m.opt.disableactuator = 1
    if bad == "fluid":
        m.opt.viscosity = 0.1
    if bad == "gravcomp":
        m.body_gravcomp[1] = 1
    if bad == "eq":
        m.eq_type[0] = 0
    if bad == "mixed_ref":
        m.eq_solref[0] = [0.02, -1]
    if bad == "identity":
        m.body_mass[1] *= 2
    if bad == "negative":
        m.dof_damping[0] = -1
    with pytest.raises(ValueError):
        compile_smooth_model(mj, m, rm)


def test_static_copy_resource():
    m = scene()
    rm = compile_rigid_model(mj, m)
    sm = compile_smooth_model(mj, m, rm)
    saved = sm.static["params"].copy()
    m.actuator_gainprm[:] = 99
    np.testing.assert_array_equal(sm.static["params"], saved)
    with pytest.raises(ValueError):
        sm.static["params"][0, 0, 0] = 99
    with pytest.raises(MemoryError):
        SmoothWorkspace(sm, memory_limit_bytes=1)
