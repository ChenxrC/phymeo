"""Actual device manifold assembly; no integration or learning samples."""

from itertools import product

import mujoco as mj
import numpy as np
import pytest

from phymeo.backends.mujoco_tyx._mesh_catalog import compile_mesh_query_catalog
from phymeo.backends.mujoco_tyx._mesh_contacts import MeshContactWorkspace
from phymeo.backends.mujoco_tyx._mesh_query import MeshQueryWorkspace


def scene(mesh=False, variation=0):
    vertices = " ".join(str(x * 0.1) for v in product((-1, 1), repeat=3) for x in v)
    shape = 'type="mesh" mesh="cube"' if mesh else 'type="box" size=".1 .1 .1"'
    model = mj.MjModel.from_xml_string(f'''<mujoco>
      <option cone="elliptic" timestep=".001" impratio="1000"/>
      <asset><mesh name="cube" vertex="{vertices}"/></asset><worldbody>
      <body><freejoint/><geom {shape} friction=".3 .005 .0001" solref=".008 1"/></body>
      <body pos=".195 .013 -.017"><freejoint/><geom {shape} friction=".7 .005 .0001" solref=".012 .8"/></body>
      </worldbody></mujoco>''')
    if variation == 1:
        model.geom_priority[0] = 3
    elif variation == 2:
        model.geom_solref[0] = [-1000, -20]
    elif variation == 3:
        model.geom_solimp[:, :3] = [0.7, 0.7, 0]
    elif variation == 4:
        model.geom_solmix[:] = 0
        model.geom_solimp[:, 4] = 1
    d = mj.MjData(model)
    d.qvel[:] = np.linspace(-0.03, 0.04, model.nv)
    mj.mj_forward(model, d)
    comjac = np.zeros((model.nbody, 6, model.nv))
    for b in range(model.nbody):
        mj.mj_jacBodyCom(model, d, comjac[b, :3], comjac[b, 3:], b)
    motion = dict(
        com=d.xipos[None].astype(np.float32),
        body_jac=comjac[None].astype(np.float32),
        qvel=d.qvel[None].astype(np.float32),
    )
    return model, d, motion


@pytest.mark.tyx
@pytest.mark.parametrize("variation", range(5))
def test_native_box_manifold_and_rows(tyx_mode, tmp_path, variation):
    m, d, motion = scene(variation=variation)
    cat = compile_mesh_query_catalog(mj, m)
    with (
        MeshQueryWorkspace(cat, capacity=4, mode=tyx_mode, cache_root=tmp_path) as q,
        MeshContactWorkspace(mj, m, cat, capacity=24, mode=tyx_mode, cache_root=tmp_path) as c,
    ):
        query = q.query(d.geom_xpos[None], d.geom_xmat.reshape(1, m.ngeom, 3, 3))
        result = c.assemble(query, motion)
        raw = result.read()
        assert raw["status"].tolist() == [0]
        assert raw["counts"].tolist() == [d.ncon]
        assert d.ncon > 1
        for i in range(d.ncon):
            native = min(d.contact, key=lambda x: np.linalg.norm(x.pos - raw["geometry"][i, 1:4]))
            adr = native.efc_address
            np.testing.assert_allclose(
                raw["geometry"][i, :4], np.r_[native.dist, native.pos], atol=5e-6
            )
            np.testing.assert_allclose(raw["frame"][i], native.frame.reshape(3, 3), atol=2e-4)
            np.testing.assert_allclose(
                raw["jac"][i], d.efc_J.reshape(-1, m.nv)[adr : adr + 3], atol=2e-5
            )
            np.testing.assert_allclose(raw["R"][i], d.efc_R[adr : adr + 3], rtol=2e-5)
            np.testing.assert_allclose(raw["D"][i], d.efc_D[adr : adr + 3], rtol=2e-5)
            np.testing.assert_allclose(
                raw["aref"][i], d.efc_aref[adr : adr + 3], rtol=2e-5, atol=2e-4
            )
            np.testing.assert_allclose(raw["friction"][i], native.friction[:2], rtol=2e-5)
            assert raw["mu"][i] == pytest.approx(native.mu, rel=2e-5)
        with result.device_view() as view:
            assert view["jac"] is c._buffers["jac"]
            with pytest.raises(RuntimeError, match="borrowed"):
                c.close()
            with pytest.raises(RuntimeError, match="borrowed"):
                c.assemble(query, motion)
        with pytest.raises(ValueError, match="schema"):
            c.assemble(query, {})
        with pytest.raises(RuntimeError, match="expired"):
            result.read()


@pytest.mark.tyx
def test_mesh_single_point_and_bad_motion(tyx_mode, tmp_path):
    m, d, motion = scene(mesh=True)
    cat = compile_mesh_query_catalog(mj, m)
    with (
        MeshQueryWorkspace(cat, capacity=4, mode=tyx_mode, cache_root=tmp_path) as q,
        MeshContactWorkspace(mj, m, cat, capacity=4, mode=tyx_mode, cache_root=tmp_path) as c,
    ):
        query = q.query(d.geom_xpos[None], d.geom_xmat.reshape(1, m.ngeom, 3, 3))
        raw = c.assemble(query, motion).read()
        assert raw["counts"].tolist() == [1]
        assert raw["status"].tolist() == [0]
        assert raw["geometry"][0, 0] == pytest.approx(d.contact[0].dist, abs=5e-6)
        # Independently use native Jacobian at the GPU contact point, since a
        # face/face contact has nonunique witness midpoint.
        point = raw["geometry"][0, 1:4].astype(float)
        js = []
        for b in m.geom_bodyid:
            jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
            mj.mj_jac(m, d, jp, jr, point, b)
            js.append(jp)
        np.testing.assert_allclose(raw["jac"][0], raw["frame"][0] @ (js[1] - js[0]), atol=2e-5)
        motion["qvel"][0, 0] = np.nan
        invalid = c.assemble(query, motion)
        with pytest.raises(ValueError, match="invalid"):
            with invalid.device_view():
                pass


@pytest.mark.tyx
def test_contact_overflow_not_truncated(tyx_mode, tmp_path):
    m, d, motion = scene()
    cat = compile_mesh_query_catalog(mj, m)
    with (
        MeshQueryWorkspace(cat, capacity=4, mode=tyx_mode, cache_root=tmp_path) as q,
        MeshContactWorkspace(mj, m, cat, capacity=1, mode=tyx_mode, cache_root=tmp_path) as c,
    ):
        query = q.query(d.geom_xpos[None], d.geom_xmat.reshape(1, m.ngeom, 3, 3))
        result = c.assemble(query, motion)
        assert result.read()["status"][0] != 0
        with pytest.raises(ValueError, match="overflow"):
            with result.device_view():
                pass


@pytest.mark.parametrize("field,value", [("margin", 0.001), ("gap", 0.001), ("condim", 4)])
def test_unsupported_law_is_rejected_before_device(field, value):
    m, _, _ = scene()
    getattr(m, "geom_" + field)[:] = value
    cat = compile_mesh_query_catalog(mj, m)
    with pytest.raises(ValueError, match="zero-margin"):
        MeshContactWorkspace(mj, m, cat)


def test_resource_preflight_and_multiccd_rejection():
    m, _, _ = scene()
    cat = compile_mesh_query_catalog(mj, m)
    with pytest.raises(MemoryError):
        MeshContactWorkspace(mj, m, cat, memory_limit_bytes=1)
    m.opt.enableflags = int(mj.mjtEnableBit.mjENBL_MULTICCD)
    with pytest.raises(ValueError, match="default flags"):
        MeshContactWorkspace(mj, m, cat)


@pytest.mark.tyx
def test_resident_motion_mixed_pair_and_empty_world(tyx_mode, tmp_path):
    import tyx as tx

    m, d, single = scene(mesh=True)
    m.geom_type[0] = 6
    m.geom_size[0] = [0.1, 0.1, 0.1]
    cat = compile_mesh_query_catalog(mj, m)
    positions = np.repeat(d.geom_xpos[None], 2, axis=0)
    rotations = np.repeat(d.geom_xmat.reshape(1, m.ngeom, 3, 3), 2, axis=0)
    positions[1, 1, 0] = 1
    motion = {name: np.repeat(value, 2, axis=0) for name, value in single.items()}
    with (
        MeshQueryWorkspace(cat, worlds=2, capacity=4, mode=tyx_mode, cache_root=tmp_path) as q,
        MeshContactWorkspace(
            mj, m, cat, worlds=2, capacity=4, mode=tyx_mode, cache_root=tmp_path
        ) as c,
    ):
        query = q.query(positions, rotations)
        with c.runtime.activate():
            device = {
                name: tx.cuda.array(v) if tyx_mode == "cuda" else v for name, v in motion.items()
            }
            try:
                result = c.assemble_device(query, device)
                raw = result.read()
                np.testing.assert_array_equal(raw["counts"], [1, 0])
                np.testing.assert_array_equal(raw["eligible"], [1, 0, 0, 0, 0, 0, 0, 0])
                np.testing.assert_array_equal(raw["status"], 0)
                with result.device_view() as view:
                    assert view["jac"] is c._buffers["jac"]
                q.query(positions, rotations)
                with pytest.raises(RuntimeError, match="expired"):
                    c.assemble_device(query, device)
                with pytest.raises(RuntimeError, match="expired"):
                    result.read()
            finally:
                for v in device.values():
                    if hasattr(v, "free"):
                        v.free()
    with pytest.raises(RuntimeError, match="closed"):
        result.read()
