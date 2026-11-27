"""Geometry-only device integration, distinct from dynamics/learning admission."""

import numpy as np
import pytest

from phymeo.backends.mujoco_tyx._mesh_catalog import compile_mesh_query_catalog
from phymeo.backends.mujoco_tyx._mesh_query import MeshQueryWorkspace


def boxes(count=2):
    import mujoco as mj

    bodies = "".join(
        f'<body pos="{i * 0.195} 0 0"><freejoint/><geom type="box" size=".1 .1 .1" margin=".01"/></body>'
        for i in range(count)
    )
    model = mj.MjModel.from_xml_string(f"<mujoco><worldbody>{bodies}</worldbody></mujoco>")
    data = mj.MjData(model)
    mj.mj_forward(model, data)
    return mj, model, data


def test_catalog_is_owned_and_preserves_vertices_and_filters():
    mj, model, _ = boxes(3)
    cat = compile_mesh_query_catalog(mj, model, geom_ids=[2, 0])
    np.testing.assert_array_equal(cat.geom_ids, [2, 0])
    np.testing.assert_array_equal(cat.pairs, [[1, 0]])
    assert cat.points.shape == (16, 3)
    model.geom_size[:] *= 2
    np.testing.assert_allclose(np.abs(cat.points), 0.1)
    with pytest.raises(ValueError):
        cat.points.setflags(write=True)


@pytest.mark.parametrize("selection", [[0, 0], [-1], [9], [0.1], []])
def test_invalid_geometry_selection(selection):
    mj, model, _ = boxes()
    with pytest.raises(ValueError):
        compile_mesh_query_catalog(mj, model, geom_ids=selection)


def test_explicit_pairs_and_collision_flags_are_not_silently_ignored():
    mj, model, _ = boxes()
    model.opt.disableflags = int(mj.mjtDisableBit.mjDSBL_FILTERPARENT)
    with pytest.raises(ValueError, match="default collision"):
        compile_mesh_query_catalog(mj, model)
    model.opt.disableflags = 0
    model.opt.enableflags = int(mj.mjtEnableBit.mjENBL_OVERRIDE)
    with pytest.raises(ValueError, match="default collision"):
        compile_mesh_query_catalog(mj, model)


def test_resource_check_precedes_device_allocation():
    mj, model, _ = boxes()
    cat = compile_mesh_query_catalog(mj, model)
    with pytest.raises(MemoryError):
        MeshQueryWorkspace(cat, worlds=256, capacity=65536, memory_limit_bytes=1024)
    with pytest.raises(ValueError):
        MeshQueryWorkspace(cat, worlds=True)
    with pytest.raises(ValueError):
        MeshQueryWorkspace(cat, tile=4097)
    with pytest.raises(MemoryError):
        MeshQueryWorkspace(cat, tile=4096, memory_limit_bytes=1024)


@pytest.mark.tyx
@pytest.mark.parametrize("separation", [0.195, 0.205, 0.5])
def test_device_box_gap_and_world_isolation(tyx_mode, tmp_path, separation):
    mj, model, data = boxes()
    cat = compile_mesh_query_catalog(mj, model)
    positions = np.repeat(data.geom_xpos[None], 2, axis=0)
    rotations = np.repeat(data.geom_xmat.reshape(1, 2, 3, 3), 2, axis=0)
    positions[0, 1, 0] = separation
    positions[1, 1, 0] = 0.5
    with MeshQueryWorkspace(cat, worlds=2, capacity=4, mode=tyx_mode, cache_root=tmp_path) as query:
        result = query.query(positions, rotations)
        raw = result.read()
        assert result.counts == ((1, 0) if separation < 0.22 else (0, 0))
        if len(raw["world"]):
            assert raw["world"].tolist() == [0]
            assert raw["meta"][0, 0] == 1  # RESOLVED, not UNKNOWN mapped to no contact.
            assert raw["gap"][0, 0] == pytest.approx(separation - 0.2, abs=2e-6)
            np.testing.assert_allclose(raw["normal"][0], [1, 0, 0], atol=2e-5)
            assert not raw["gap"].flags.writeable
        query.query(positions, rotations)
        with pytest.raises(RuntimeError, match="expired"):
            result.read()
    with pytest.raises(RuntimeError, match="closed"):
        result.read()


@pytest.mark.tyx
def test_overflow_invalid_pose_and_stale_result(tyx_mode, tmp_path):
    mj, model, data = boxes(3)
    cat = compile_mesh_query_catalog(mj, model)
    positions = data.geom_xpos[None].copy()
    rotations = data.geom_xmat.reshape(1, 3, 3, 3).copy()
    with MeshQueryWorkspace(cat, capacity=1, mode=tyx_mode, cache_root=tmp_path) as query:
        positions[0, :, 0] = [0, 0.5, 1]
        result = query.query(positions, rotations)
        positions[0, :, 0] = [0, 0.01, 0.02]
        with pytest.raises(OverflowError, match="capacity"):
            query.query(positions, rotations)
        with pytest.raises(RuntimeError, match="expired"):
            result.read()
        rotations[0, 0, 0, 0] = -1
        with pytest.raises(ValueError, match="rigid"):
            query.query(positions, rotations)
        positions[0, 0, 0] = np.nan
        with pytest.raises(ValueError, match="finite"):
            query.query(positions, rotations)


@pytest.mark.tyx
def test_device_pose_entry_reads_counts_only_before_explicit_download(tyx_mode, tmp_path):
    import tyx as tx

    mj, model, data = boxes()
    cat = compile_mesh_query_catalog(mj, model)
    p = data.geom_xpos[None].astype(np.float32)
    r = data.geom_xmat.reshape(1, 2, 3, 3).astype(np.float32)
    if tyx_mode == "cuda":
        p, r = tx.cuda.array(p), tx.cuda.array(r)
    try:
        with MeshQueryWorkspace(cat, capacity=2, mode=tyx_mode, cache_root=tmp_path) as query:
            uploaded = query.stats["uploaded_bytes"]
            result = query.query_device(p, r)
            assert result.counts == (1,)
            assert query.stats["uploaded_bytes"] == uploaded
            assert query.stats["downloaded_bytes"] == (8 if tyx_mode == "cuda" else 0)
            assert result.read()["meta"][0, 0] == 1
    finally:
        if tyx_mode == "cuda":
            p.free()
            r.free()


@pytest.mark.tyx
@pytest.mark.parametrize("length", [1.0, 0.01, 0.001])
def test_small_geometry_preserves_resolved_separation(tyx_mode, tmp_path, length):
    mj, model, data = boxes()
    model.geom_size[:] *= length
    cat = compile_mesh_query_catalog(mj, model)
    positions = np.zeros((1, 2, 3), dtype=np.float32)
    positions[0, 1, 0] = 0.205 * length
    rotations = np.tile(np.eye(3, dtype=np.float32), (1, 2, 1, 1))
    with MeshQueryWorkspace(cat, capacity=2, mode=tyx_mode, cache_root=tmp_path) as query:
        raw = query.query(positions, rotations).read()
        assert raw["meta"][0, 0] == 1
        assert raw["gap"][0, 0] == pytest.approx(0.005 * length, abs=2e-7 * length)


@pytest.mark.tyx
def test_compiled_mesh_rotation_translation_and_release(tyx_mode, tmp_path):
    import mujoco as mj

    xml = """<mujoco><asset><mesh name="poly" vertex="-.001 -.001 -.001 .001 -.001 -.001 -.001 .001 -.001 .001 .001 -.001 -.001 -.001 .001 .001 -.001 .001 -.001 .001 .001 .001 .001 .001"/></asset><worldbody>
    <body><freejoint/><geom type="mesh" mesh="poly" margin=".0001"/></body>
    <body pos=".00203 0 0"><freejoint/><geom type="mesh" mesh="poly" margin=".0001"/></body>
    </worldbody></mujoco>"""
    model = mj.MjModel.from_xml_string(xml)
    data = mj.MjData(model)
    mj.mj_kinematics(model, data)
    cat = compile_mesh_query_catalog(mj, model)
    # Keep the compiler's mesh-local frame and PCA correction; rotate the whole scene.
    turn = np.array([[0.6, -0.8, 0], [0.8, 0.6, 0], [0, 0, 1.0]])
    positions = (data.geom_xpos @ turn.T + [0.48, 0.4, 0.1])[None]
    rotations = (turn[None] @ data.geom_xmat.reshape(2, 3, 3))[None]
    query = MeshQueryWorkspace(cat, capacity=2, mode=tyx_mode, cache_root=tmp_path)
    with query:
        arrays = tuple(query._buffers.values())
        raw = query.query(positions, rotations).read()
        assert raw["meta"][0, 0] == 1
        assert raw["gap"][0, 0] == pytest.approx(0.00003, abs=1e-7)
        np.testing.assert_allclose(raw["normal"][0], [0.6, 0.8, 0], atol=2e-3)
        np.testing.assert_allclose(
            raw["surface"][0, 1] - raw["surface"][0, 0],
            raw["gap"][0, 0] * raw["normal"][0],
            atol=1e-7,
        )
    if tyx_mode == "cuda":
        assert all(value._freed for value in arrays)
