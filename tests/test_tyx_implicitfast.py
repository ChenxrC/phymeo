"""Native implicitfast references and process-local transaction failure tests."""

from contextlib import contextmanager

import mujoco as mj
import numpy as np
import pytest

from phymeo.backends.mujoco_tyx._implicitfast import ImplicitWorkspace, compile_implicit_model
from phymeo.backends.mujoco_tyx._rigid_terms import _split, compile_rigid_model
from phymeo.backends.mujoco_tyx._smooth_terms import compile_smooth_model


def scene():
    return mj.MjModel.from_xml_string("""<mujoco><compiler angle="radian"/>
     <option integrator="implicitfast" timestep=".001" gravity=".2 -9.81 .3"/>
     <default><geom type="box" size=".12 .08 .05" mass="1" contype="0" conaffinity="0"/>
      <joint damping=".2" armature=".01"/></default>
     <worldbody><body pos=".2 .3 .4" euler=".2 -.4 .1"><freejoint/><geom pos=".02 -.01 .04"/>
      <body pos=".25 .03 -.02"><joint name="a" axis="1 .3 -.2" stiffness="2" springref=".1" actuatorfrcrange="-1 1"/><geom pos=".06 .02 .01"/>
       <body pos=".1 -.05 .04"><joint name="b" type="slide" axis=".3 .5 1"/><geom pos=".04 .02 -.03"/>
       </body></body></body></worldbody>
     <actuator><general joint="a" gear="-2" gainprm="30" biastype="affine" biasprm="0 -40 -8" forcerange="-2 2"/>
       <general joint="a" gear=".7" gainprm="2" biastype="affine" biasprm="0 -2 -3"/>
       <general joint="b" gear="1" gainprm="20" biastype="affine" biasprm="0 -30 -4"/></actuator>
    </mujoco>""")


def compiled(m):
    rm = compile_rigid_model(mj, m)
    sm = compile_smooth_model(mj, m, rm)
    return compile_implicit_model(mj, m, sm)


def inputs(m, d):
    mj.mj_forward(m, d)
    M = np.zeros((m.nv, m.nv))
    mj.mj_fullM(m, M, d.qM)
    force = d.qfrc_smooth + d.qfrc_constraint
    packed = _split(M)[None]
    f = _split(force)[None]
    return dict(
        mass=packed[..., 0].copy(),
        mass_low=packed[..., 1].copy(),
        force=f[..., 0].copy(),
        force_low=f[..., 1].copy(),
    )


@contextmanager
def resident(work, values):
    import tyx as tx

    with work.runtime.activate():
        data = {k: tx.cuda.array(v) if work.mode == "cuda" else v.copy() for k, v in values.items()}
    try:
        yield data
    finally:
        for a in data.values():
            if hasattr(a, "free"):
                a.free()


@pytest.mark.tyx
@pytest.mark.parametrize(
    "spin,scale", [(0.0, 1.0), (1e-15, 1.0), (2.0, 1.0), (-3.0, 2.0), (80.0, 1.0)]
)
def test_original_implicitfast_and_quaternion(tyx_mode, tmp_path, spin, scale):
    m = scene()
    d = mj.MjData(m)
    d.qpos[3:7] *= scale
    d.qpos[7:] = [0.2, 0.1]
    d.qvel[:] = np.linspace(-0.3, 0.2, m.nv)
    d.qvel[3:6] = [spin, -spin * 0.3, spin * 0.7]
    d.ctrl[:] = [4.0, -3.0, 0.6]
    d.qfrc_applied[:] = 0.07
    q0 = d.qpos.copy()
    v0 = d.qvel.copy()
    values = inputs(m, d)
    model = compiled(m)
    with ImplicitWorkspace(model, mode=tyx_mode, cache_root=tmp_path) as work:
        work.initialize(q0[None], v0[None], np.array([0.123]))
        with resident(work, values) as device:
            candidate = work.prepare_device(device, expected_generation=0)
            got = candidate.read()
        np.testing.assert_array_equal(work.read_state()["qvel"], v0[None].astype(np.float32))
        d.time = 0.123
        mj.mj_implicit(m, d)
        np.testing.assert_allclose(
            got["qpos"][0, :, 0].astype(float) + got["qpos"][0, :, 1], d.qpos, atol=2e-8, rtol=0
        )
        np.testing.assert_allclose(got["qvel"][0], d.qvel, atol=2e-6, rtol=2e-6)
        assert abs(got["time"][0].astype(float).sum() - d.time) < 2e-12
        assert got["balance"][0] < 2e-6
        deriv = np.zeros((m.nv, m.nv))
        for i in range(m.nv):
            a = int(m.D_rowadr[i])
            n = int(m.D_rownnz[i])
            deriv[i, m.D_colind[a : a + n]] = d.qDeriv[a : a + n]
        expected = np.diag(
            model.static["derivative"][:, 0].astype(float) + model.static["derivative"][:, 1]
        )
        np.testing.assert_allclose(deriv, expected, atol=2e-12, rtol=0)
        # Native derivative is still active at the saturated joint.
        assert abs(d.qfrc_actuator[6]) == 1 and deriv[6, 6] < -30
        corrected = got["modified"][0].astype(float) + got["modified_low"][0]
        np.testing.assert_allclose(
            corrected,
            values["mass"][0].astype(float) + values["mass_low"][0] - m.opt.timestep * deriv,
            atol=2e-12,
            rtol=0,
        )
        assert work.commit(candidate) == 1
        with pytest.raises(RuntimeError, match="expired"):
            candidate.read()


@pytest.mark.tyx
def test_bounded_sequence_and_clock(tyx_mode, tmp_path):
    m = scene()
    d = mj.MjData(m)
    d.qvel[:] = np.linspace(-0.2, 0.3, m.nv)
    d.ctrl[:] = [0.01, 0.02, -0.03]
    with ImplicitWorkspace(compiled(m), mode=tyx_mode, cache_root=tmp_path) as work:
        work.initialize(d.qpos[None], d.qvel[None], np.zeros(1))
        for step in range(32):
            with resident(work, inputs(m, d)) as device:
                candidate = work.prepare_device(device, expected_generation=step)
            work.commit(candidate)
            mj.mj_implicit(m, d)
            got = work.read_state()
            np.testing.assert_allclose(
                got["qpos"][0, :, 0].astype(float) + got["qpos"][0, :, 1], d.qpos, atol=2e-6, rtol=0
            )
            np.testing.assert_allclose(got["qvel"][0], d.qvel, atol=2e-5, rtol=0)
        assert abs(got["time"][0].astype(float).sum() - 0.032) < 2e-12
        assert work.stats["committed"] == 32


@pytest.mark.tyx
@pytest.mark.parametrize(
    "bad",
    ["force_nan", "mass_negative", "low_nan", "shape", "generation", "quat_zero", "spin_domain"],
)
def test_failure_preserves_active_state(tyx_mode, tmp_path, bad):
    m = scene()
    d = mj.MjData(m)
    values = inputs(m, d)
    if bad == "quat_zero":
        d.qpos[3:7] = 0
    if bad == "spin_domain":
        d.qvel[3] = 3000
    with ImplicitWorkspace(compiled(m), mode=tyx_mode, cache_root=tmp_path) as work:
        work.initialize(d.qpos[None], d.qvel[None], np.zeros(1))
        before = work.read_state()
        if bad == "force_nan":
            values["force"][0, 1] = np.nan
        if bad == "mass_negative":
            values["mass"][0, 0, 0] = -10
        if bad == "low_nan":
            values["mass_low"][0, 0, 0] = np.nan
        if bad == "shape":
            values["force"] = values["force"][:, :-1]
        with resident(work, values) as device, pytest.raises((RuntimeError, ValueError)):
            work.prepare_device(device, expected_generation=1 if bad == "generation" else 0)
        after = work.read_state()
        for k in ("qpos", "qvel", "time"):
            np.testing.assert_array_equal(before[k], after[k])
        assert after["generation"] == 0 and work.stats["committed"] == 0


@pytest.mark.tyx
def test_generation_loans_discard(tyx_mode, tmp_path):
    m = scene()
    d = mj.MjData(m)
    with ImplicitWorkspace(compiled(m), mode=tyx_mode, cache_root=tmp_path) as work:
        work.initialize(d.qpos[None], d.qvel[None], np.zeros(1))
        with resident(work, inputs(m, d)) as device:
            with work.state_view() as state:
                candidate = work.prepare_device(device, expected_generation=state["generation"])
                with pytest.raises(RuntimeError, match="borrowed"):
                    work.commit(candidate)
            with candidate.device_view():
                with pytest.raises(RuntimeError, match="borrowed"):
                    work.prepare_device(device, expected_generation=0)
                with pytest.raises(RuntimeError, match="borrowed"):
                    work.close()
            work.discard(candidate)
            with pytest.raises(RuntimeError, match="expired"):
                work.commit(candidate)
            old = work.prepare_device(device, expected_generation=0)
            new = work.prepare_device(device, expected_generation=0)
            with pytest.raises(RuntimeError, match="expired"):
                work.commit(old)
            work.commit(new)
            with pytest.raises(RuntimeError, match="generation"):
                work.prepare_device(device, expected_generation=0)
            with pytest.raises(RuntimeError, match="initialized"):
                work.initialize(d.qpos[None], d.qvel[None], np.zeros(1))
        assert work.generation == 1


@pytest.mark.parametrize("bad", ["euler", "positive_derivative", "identity", "resource"])
def test_model_domain(bad):
    m = scene()
    rm = compile_rigid_model(mj, m)
    sm = compile_smooth_model(mj, m, rm)
    if bad == "euler":
        m.opt.integrator = mj.mjtIntegrator.mjINT_EULER
    if bad == "positive_derivative":
        m.actuator_biasprm[0, 2] = 100
        sm = compile_smooth_model(mj, m, rm)
    if bad == "identity":
        m.opt.timestep *= 2
    if bad == "resource":
        with pytest.raises(MemoryError):
            ImplicitWorkspace(compile_implicit_model(mj, m, sm), memory_limit_bytes=1)
    else:
        with pytest.raises(ValueError):
            compile_implicit_model(mj, m, sm)


@pytest.mark.tyx
@pytest.mark.parametrize("bad", [False, True])
def test_actual_contact_and_scalar_force_assembly(tyx_mode, tmp_path, bad):
    from phymeo.backends.mujoco_tyx import _implicitfast_kernels as k

    m = scene()
    # Equality gives -4, saturated dry friction gives -0.5, active lower limit gives +3.
    values = dict(
        counts=np.array([1], np.int32),
        result_counts=np.array([0 if bad else 1], np.int32),
        jac=np.array([[[1.0, 0.0], [0.0, 1.0], [1.0, -1.0]]], np.float32),
        force=np.array([[2.0, 3.0, 4.0]], np.float32),
        scalar_counts=np.array([3], np.int32),
        sj=np.array([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]], np.float32),
        sa=np.array([[0.0, 0.0, 4.0]], np.float32),
        sd=np.array([[4.0, 2.0, 3.0]], np.float32),
        sl=np.array([[0.0, 0.5, 0.0]], np.float32),
        sk=np.array([[0, 1, 3]], np.int32),
        acc=np.array([[1.0, 2.0]], np.float32),
        acc_low=np.zeros((1, 2), np.float32),
        smooth_rhs=np.array([[10.0, 20.0]], np.float32),
        smooth_rhs_low=np.zeros((1, 2), np.float32),
        total=np.zeros((1, 2), np.float32),
        total_low=np.zeros((1, 2), np.float32),
    )
    with (
        ImplicitWorkspace(compiled(m), mode=tyx_mode, cache_root=tmp_path) as work,
        resident(work, values) as device,
        work.runtime.activate(),
    ):
        k.coupled_force(1, 2, 1, 3, *device.values(), mode=tyx_mode)
        work.runtime.sync()
        result = {
            n: a.to_numpy() if tyx_mode == "cuda" else a.copy()
            for n, a in device.items()
            if n.startswith("total")
        }
        if bad:
            assert not np.isfinite(result["total"]).any()
        else:
            np.testing.assert_array_equal(
                result["total"].astype(float) + result["total_low"], [[15.0, 21.5]]
            )


@pytest.mark.tyx
def test_resident_force_integration_sequence(tyx_mode, tmp_path):
    from phymeo.backends.mujoco_tyx._rigid_terms import RigidWorkspace
    from phymeo.backends.mujoco_tyx._smooth_terms import SmoothWorkspace

    m = scene()
    model = compiled(m)
    d = mj.MjData(m)
    d.qvel[:] = np.linspace(-0.2, 0.3, m.nv)
    d.ctrl[:] = [0.01, 0.02, -0.03]
    # No contacts or scalar rows in this small model. All moving forces below
    # are produced from the current resident state, not the native reference.
    assert m.neq == 0 and not m.dof_frictionloss.any() and not m.jnt_limited.any()
    with (
        ImplicitWorkspace(model, mode=tyx_mode, cache_root=tmp_path / "i") as work,
        RigidWorkspace(model.smooth.rigid, mode=tyx_mode, cache_root=tmp_path / "r") as rigid,
        SmoothWorkspace(model.smooth, mode=tyx_mode, cache_root=tmp_path / "s") as smooth,
    ):
        values = dict(
            mocap=np.zeros((1, 0, 7, 2), np.float32),
            ctrl=_split(d.ctrl[None]),
            applied=_split(d.qfrc_applied[None]),
            body_force=_split(d.xfrc_applied[None]),
            eq_active=d.eq_active[None].astype(np.int32),
        )
        work.initialize(d.qpos[None], d.qvel[None], np.zeros(1))
        with resident(work, values) as device:
            for step in range(32):
                with work.state_view() as state:
                    rr = rigid.evaluate_device(
                        dict(qpos=state["qpos"], qvel=state["qvel"], mocap=device["mocap"])
                    )
                    sr = smooth.evaluate_device(rr, {k: device[k] for k in smooth.inputs})
                    with sr.device_view() as s:
                        candidate = work.prepare_device(
                            dict(
                                mass=s["mass"],
                                mass_low=s["mass_low"],
                                force=s["rhs"],
                                force_low=s["rhs_low"],
                            ),
                            expected_generation=step,
                        )
                work.commit(candidate)
                mj.mj_forward(m, d)
                mj.mj_implicit(m, d)
                got = work.read_state()
                np.testing.assert_allclose(
                    got["qpos"][0, :, 0].astype(float) + got["qpos"][0, :, 1],
                    d.qpos,
                    atol=2e-6,
                    rtol=0,
                )
                np.testing.assert_allclose(got["qvel"][0], d.qvel, atol=2e-5, rtol=0)
        assert work.generation == 32


@pytest.mark.tyx
def test_one_bad_world_keeps_entire_batch(tyx_mode, tmp_path):
    m = scene()
    d = mj.MjData(m)
    values = {k: np.repeat(a, 2, axis=0) for k, a in inputs(m, d).items()}
    values["force"][1, 0] = np.nan
    with ImplicitWorkspace(compiled(m), worlds=2, mode=tyx_mode, cache_root=tmp_path) as work:
        work.initialize(
            np.repeat(d.qpos[None], 2, axis=0), np.repeat(d.qvel[None], 2, axis=0), np.zeros(2)
        )
        before = work.read_state()
        with resident(work, values) as device, pytest.raises(ValueError, match="candidate"):
            work.prepare_device(device, expected_generation=0)
        for k in ("qpos", "qvel", "time"):
            np.testing.assert_array_equal(before[k], work.read_state()[k])
        assert work.generation == 0


@pytest.mark.parametrize("field", ["capacity", "scalar_capacity"])
def test_coupled_capacity_mismatch_fails_before_access(tmp_path, field):
    from types import SimpleNamespace as NS

    m = scene()
    model = compiled(m)
    with ImplicitWorkspace(model, mode="interpret", cache_root=tmp_path) as work:
        work.initialize(m.qpos0[None], np.zeros((1, m.nv)), np.zeros(1))
        s = NS(workspace=NS(model=model.smooth, worlds=1, mode="interpret"))
        c = NS(workspace=NS(worlds=1, mode="interpret", capacity=2))
        a = NS(
            workspace=NS(
                worlds=1,
                mode="interpret",
                nv=m.nv,
                capacity=2,
                scalar_capacity=model.smooth.capacity,
            )
        )
        setattr(a.workspace, field, getattr(a.workspace, field) + 1)
        with pytest.raises(ValueError, match="capacity"):
            work.prepare_coupled(s, c, a, expected_generation=0)
        assert work.generation == 0
