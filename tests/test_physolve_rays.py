"""Native endpoint pose -> selected device rays, with independent geometry references."""

import numpy as np
import pytest
from test_physolve_runtime import payload as payload_fixture

from phymeo import PhySolveRuntime
from phymeo.backends.physolve_initial import NativeInitialState
from phymeo.builders.mujoco_compiler import compile_mujoco_xml
from phymeo.builders.native_assembly import NativeTemplate

payload = payload_fixture
SCENE = """<mujoco><option timestep=".01" gravity="0 0 0"/>
<worldbody><body name="ball"><freejoint/>
<geom name="ball" type="sphere" size=".5" mass="1"/></body>
<geom name="box" type="box" pos="3 0 0" size=".5 .5 .5"/>
<geom name="ground" type="plane" pos="0 0 -2" size="0 0 1"/>
</worldbody></mujoco>"""


def template(xml=SCENE):
    import mujoco as mj

    record = compile_mujoco_xml(xml, require_asset=False)
    model = record.fresh_native_model()
    return NativeTemplate(
        record,
        NativeInitialState.from_data(record, mj.MjData(model)),
        (),
        tuple((model.body(i).name or "world", i) for i in range(model.nbody)),
        tuple((model.geom(i).name, i) for i in range(model.ngeom)),
        (),
    )


def inputs(tx, worlds, origins, directions):
    n = len(worlds)
    return (
        tx.Array(np.asarray(worlds, np.int32), device="cuda:0"),
        tx.Array(np.asarray([n], np.int32), device="cuda:0"),
        tx.Array(np.zeros(1, np.int32), device="cuda:0"),
        tx.Array(
            np.broadcast_to(origins, (n, *np.shape(origins))).copy().astype(np.float32),
            device="cuda:0",
        ),
        tx.Array(
            np.broadcast_to(directions, (n, *np.shape(directions))).copy().astype(np.float32),
            device="cuda:0",
        ),
    )


@pytest.mark.cuda
@pytest.mark.parametrize("nworld", [1, 8, 64])
def test_selected_rays_endpoint_step_reset(nworld, payload, monkeypatch):
    import tyx as tx

    source = template()
    scene = source.compile_ray_scene()
    with PhySolveRuntime(
        source.compilation, payload=payload, nworld=nworld, body_observations=True
    ) as runtime:
        plan = runtime.prepare_ray_query(scene, max_rays=3)
        worlds = [nworld - 1, 0] if nworld > 1 else [0]
        args = inputs(
            tx, worlds, [[0, 0, 2], [3, 0, 2], [8, 0, 2]], [[0, 0, -1], [0, 0, -1], [0, 0, 1]]
        )

        def sample():
            plan.sample(*args, rows=len(worlds), far=10.0)
            plan.storage.stream.synchronize()
            assert plan.errors.numpy()[0] == 0
            return plan.hits.numpy()[: len(worlds) * 3].reshape(len(worlds), 3, 4).copy()

        baseline = sample()
        np.testing.assert_allclose(baseline[:, :2, 0], 1.5, atol=1e-6, rtol=0)
        assert np.isposinf(baseline[:, 2, 0]).all()
        np.testing.assert_array_equal(
            baseline[:, :2, 1:], np.tile([[[0, 0, 1]]], (len(worlds), 2, 1))
        )
        labels = plan.labels.numpy()[: len(worlds) * 3].reshape(len(worlds), 3)
        assert scene.surfaces[labels[0, 0]].name == "ball"
        assert scene.surfaces[labels[0, 1]].name == "box"
        np.testing.assert_array_equal(labels[:, 2], -1)
        runtime.prepare_step_graph()
        ticket = runtime.begin_execution()
        state = runtime.storage_binding().borrow("qvel", runtime.writable_bank, write=True)
        velocity = np.zeros(state.shape, np.float32)
        velocity[:, 2] = np.arange(nworld, dtype=np.float32) + 1
        state.assign(velocity, stream=plan.storage.stream)
        runtime.enqueue_step(ticket, steps=4, execution_mode="cuda_graph")
        runtime.complete_execution(ticket)
        runtime.accept_execution(ticket)
        observed = sample()
        center_z = 0.04 * (np.asarray(worlds) + 1)
        entry, exit = 2 - center_z - 0.5, 2 - center_z + 0.5
        expected = np.where(entry >= 0, entry, np.where(exit >= 0, exit, 4.0))
        np.testing.assert_allclose(observed[:, 0, 0], expected, atol=1e-6, rtol=0)
        with monkeypatch.context() as patch:
            patch.setattr(
                tx, "empty", lambda *a, **k: (_ for _ in ()).throw(AssertionError("hot allocation"))
            )
            patch.setattr(
                tx.Array,
                "numpy",
                lambda *a, **k: (_ for _ in ()).throw(AssertionError("hot snapshot")),
            )
            plan.sample(*args, rows=len(worlds), far=10.0)
        plan.storage.stream.synchronize()
        mask = np.arange(nworld) == 0
        runtime.reset(mask)
        after = sample()
        np.testing.assert_allclose(after[-1], baseline[-1], atol=1e-6, rtol=0)
        if nworld > 1:
            np.testing.assert_array_equal(after[0], observed[0])
        assert plan.costs()["dispatched_rays"] == 4 * len(worlds) * 3


@pytest.mark.cuda
def test_analytic_surfaces_and_native_mesh(payload):
    import mujoco as mj
    import tyx as tx

    shapes = [
        ("sphere", ".5"),
        ("capsule", ".5 .5"),
        ("ellipsoid", ".4 .3 .5"),
        ("cylinder", ".5 .5"),
        ("box", ".5 .5 .5"),
        ("mesh", ""),
    ]
    geoms = "".join(
        f'<geom name="g{i}" type="{kind}" pos="{3 * i} 0 0" '
        + (f'size="{size}"' if kind != "mesh" else 'mesh="tetra"')
        + "/>"
        for i, (kind, size) in enumerate(shapes)
    )
    xml = (
        """<mujoco><asset><mesh name="tetra" vertex="0 0 0 1 0 0 0 1 0 0 0 1"
        face="0 2 1 0 1 3 0 3 2 1 2 3"/></asset><worldbody>"""
        + geoms
        + """</worldbody></mujoco>"""
    )
    source = template(xml)
    scene = source.compile_ray_scene()
    origins = [[3 * i + 0.1, 0.1, 2.0] for i in range(len(shapes))]
    directions = [[0, 0, -1]] * len(shapes)
    model = source.compilation.fresh_native_model()
    data = mj.MjData(model)
    mj.mj_kinematics(model, data)
    expected, ids = [], []
    for origin, direction in zip(origins, directions):
        geom = np.full(1, -1, np.int32)
        expected.append(
            mj.mj_ray(
                model,
                data,
                np.asarray(origin),
                np.asarray(direction, np.float64),
                None,
                True,
                -1,
                geom,
            )
        )
        ids.append(int(geom[0]))
    with PhySolveRuntime(source.compilation, payload=payload, body_observations=True) as runtime:
        plan = runtime.prepare_ray_query(scene, max_rays=len(shapes))
        args = inputs(tx, [0], origins, directions)
        plan.sample(*args, rows=1, far=10.0)
        plan.storage.stream.synchronize()
        assert plan.errors.numpy()[0] == 0
        np.testing.assert_allclose(plan.hits.numpy()[:, 0], expected, atol=2e-6, rtol=2e-6)
        np.testing.assert_array_equal(plan.labels.numpy(), ids)
        # Near clipping must find the second sphere root; far endpoint is included.
        args = inputs(tx, [0], [[0, 0, 2]], [[0, 0, -1]])
        plan.sample(*args, rows=1, near=2.0, far=2.5)
        plan.storage.stream.synchronize()
        np.testing.assert_allclose(plan.hits.numpy()[0, 0], 2.5, atol=1e-6, rtol=0)
        # Interior exits share the same ray-facing normal policy for analytic and mesh geometry.
        args = inputs(tx, [0], [[0, 0, 0], [12, 0, 0]], [[0, 0, 1], [0, 0, 1]])
        plan.sample(*args, rows=1, far=10.0)
        plan.storage.stream.synchronize()
        np.testing.assert_allclose(plan.hits.numpy()[:2, 0], 0.5, atol=1e-6, rtol=0)
        np.testing.assert_allclose(
            plan.hits.numpy()[:2, 1:], [[0, 0, -1], [0, 0, -1]], atol=1e-6, rtol=0
        )


@pytest.mark.cuda
def test_ray_errors_capture_stream_and_cleanup(payload, monkeypatch):
    import gc

    import tyx as tx

    from phymeo.backends.prepared_rays import PreparedRayQuery

    source = template()
    runtime = PhySolveRuntime(source.compilation, payload=payload, nworld=8, body_observations=True)
    graph, external = None, None
    try:
        with pytest.raises(ValueError, match="compilation"):
            runtime.prepare_ray_query(template().compile_ray_scene(), max_rays=1)
        plan = runtime.prepare_ray_query(source.compile_ray_scene(), max_rays=1)
        args = inputs(tx, [7, 0], [[0, 0, 2]], [[0, 0, -1]])
        plan.sample(*args, rows=2)
        plan.storage.stream.synchronize()
        before = plan.hits.numpy().copy()
        args[2].fill_(1, stream=plan.storage.stream)
        plan.sample(*args, rows=2)
        plan.storage.stream.synchronize()
        np.testing.assert_array_equal(plan.hits.numpy(), before)
        args[2].zero_(stream=plan.storage.stream)
        args[0].assign(np.asarray([8, 0], np.int32), stream=plan.storage.stream)
        plan.sample(*args, rows=2)
        plan.storage.stream.synchronize()
        assert plan.errors.numpy()[0] == 1
        args[0].assign(np.asarray([7, 0], np.int32), stream=plan.storage.stream)
        args[4].zero_(stream=plan.storage.stream)
        plan.sample(*args, rows=2)
        plan.storage.stream.synchronize()
        assert plan.errors.numpy()[0] == 1
        args[3].assign(
            np.tile([[[3.0e38, 0, 2]]], (2, 1, 1)).astype(np.float32), stream=plan.storage.stream
        )
        args[4].assign(
            np.tile([[[1, 0, 0]]], (2, 1, 1)).astype(np.float32), stream=plan.storage.stream
        )
        plan.sample(*args, rows=2, near=2.0e38, far=3.0e38)
        plan.storage.stream.synchronize()
        assert plan.errors.numpy()[0] == 1  # finite inputs overflow the clipped local origin
        args[3].assign(
            np.tile([[[0, 0, 2]]], (2, 1, 1)).astype(np.float32), stream=plan.storage.stream
        )
        args[4].assign(
            np.tile([[[0, 0, -1]]], (2, 1, 1)).astype(np.float32), stream=plan.storage.stream
        )
        plan.sample(*args, rows=2)
        plan.storage.stream.synchronize()
        with tx.Graph(stream=plan.storage.stream, mode="cuda_stream") as graph:
            plan.sample(*args, rows=2)
        gc.collect()
        graph.replay()
        plan.storage.stream.synchronize()
        assert graph.backend == "cuda"
        assert plan.errors.numpy()[0] == 0
        np.testing.assert_allclose(plan.hits.numpy()[:2, 0], 1.5, atol=1e-6, rtol=0)
        graph.destroy()
        graph = None
        external = tx.CudaStream()
        runtime.bind_stream(int(external.handle), owner=external)
        plan.sample(*args, rows=2)
        plan.storage.stream.synchronize()
        assert plan.errors.numpy()[0] == 0
        runtime.bind_stream()
        allocate = PreparedRayQuery._array

        def fail(plan, *args):
            allocate(plan, *args)
            raise MemoryError("ray upload submitted")

        with monkeypatch.context() as patch:
            patch.setattr(PreparedRayQuery, "_array", fail)
            with pytest.raises(MemoryError, match="upload submitted"):
                runtime.prepare_ray_query(source.compile_ray_scene(), max_rays=1)
        retained = runtime._ray_queries[-1]
        with monkeypatch.context() as patch:
            patch.setattr(
                runtime,
                "_synchronize_stream",
                lambda *a: (_ for _ in ()).throw(RuntimeError("ray wait")),
            )
            with pytest.raises(RuntimeError, match="ray wait"):
                runtime.close()
        assert retained.resources[0].ptr
        runtime.close()
        assert not retained.resources
        with pytest.raises(RuntimeError, match="stale|closed"):
            plan.sample(*args, rows=2)
    finally:
        if graph is not None:
            graph.destroy()
        runtime.close()
        if external is not None:
            external.destroy()


def test_ray_surface_and_source_validation():
    from phymeo.geometry.ray_surface import RaySurface

    with pytest.raises(ValueError, match="positive"):
        RaySurface("bad", "world", "sphere", (-1,))
    with pytest.raises(ValueError, match="positive"):
        RaySurface("bad", "world", "sphere", (1.0e-50,))
    with pytest.raises(ValueError, match="degenerate"):
        RaySurface(
            "bad",
            "world",
            "mesh",
            vertices=((1.0, 0, 0), (1.0 + 1.0e-8, 0, 0), (0, 1, 0)),
            triangles=((0, 1, 2),),
        )
    with pytest.raises(ValueError, match="triangular"):
        RaySurface(
            "bad",
            "world",
            "mesh",
            vertices=((0, 0, 0), (1, 0, 0), (0, 1, 0)),
            triangles=((0, 1, 3),),
        )
    assert template().compile_ray_scene().contract_version == 1


@pytest.mark.cuda
@pytest.mark.parametrize("phase", ["mesh", "bvh"])
def test_nested_ray_prepare_cleanup_failure(payload, monkeypatch, phase):
    import tyx as tx
    from tyx.cuda_array import CudaArray

    source = template()
    runtime = PhySolveRuntime(source.compilation, payload=payload, body_observations=True)
    zero, copy = tx.Array.zero_, CudaArray.copy_from

    def fail_zero(array, *args, **kwargs):
        zero(array, *args, **kwargs)
        if runtime._ray_queries[-1].meshes and array.dtype == tx.vec3f:
            raise MemoryError("nested mesh upload")

    def fail_copy(array, host, *args, **kwargs):
        copy(array, host, *args, **kwargs)
        if runtime._ray_queries[-1].meshes and host.dtype == np.dtype("int32"):
            raise MemoryError("nested bvh upload")

    try:
        with monkeypatch.context() as patch:
            if phase == "mesh":
                patch.setattr(tx.Array, "zero_", fail_zero)
            else:
                patch.setattr(CudaArray, "copy_from", fail_copy)
            with pytest.raises(MemoryError, match="nested"):
                runtime.prepare_ray_query(source.compile_ray_scene(), max_rays=1)
        retained = runtime._ray_queries[-1]
        mesh = retained.meshes[-1]
        submitted = mesh.velocities if phase == "mesh" else mesh.bvh.order
        with monkeypatch.context() as patch:
            patch.setattr(
                runtime,
                "_synchronize_stream",
                lambda *a: (_ for _ in ()).throw(RuntimeError("nested wait")),
            )
            with pytest.raises(RuntimeError, match="nested wait"):
                runtime.close()
        pointer = submitted.ptr if phase == "mesh" else submitted.device_ptr
        assert retained.meshes[-1] is mesh and pointer
        runtime.close()
        assert not retained.meshes and not retained.resources
    finally:
        runtime.close()


@pytest.mark.cuda
@pytest.mark.parametrize("robot", ["panda", "fanuc"])
def test_robot_mesh_source_and_selected_geometry(payload, robot):
    import mujoco as mj
    import tyx as tx
    from test_physolve_models import robot_path

    from phymeo.builders.native_assembly import NativeAssetInstance, compile_native_template

    source = compile_native_template(
        (NativeAssetInstance(robot, str(robot_path(robot)), drive_mode="scalar_pd"),),
        timestep=0.01,
        gravity=(0, 0, -9.81),
    )
    scene = source.compile_ray_scene()
    if robot == "fanuc":
        assert any(s.kind == "mesh" for s in scene.surfaces)
    model = source.compilation.fresh_native_model()
    data = mj.MjData(model)
    source.initial_state.apply(source.compilation, data)
    mj.mj_kinematics(model, data)
    origins = data.geom_xpos + np.asarray([0, 0, 3.0])
    directions = np.tile([0.0, 0.0, -1.0], (model.ngeom, 1))
    expected, labels = [], []
    for origin, direction in zip(origins, directions):
        out = np.full(1, -1, np.int32)
        expected.append(mj.mj_ray(model, data, origin, direction, None, True, -1, out))
        labels.append(int(out[0]))
    expected = np.asarray(expected)
    assert np.any(expected >= 0)
    with PhySolveRuntime(
        source.compilation,
        payload=payload,
        nworld=8,
        initial_state=source.initial_state,
        body_observations=True,
    ) as runtime:
        plan = runtime.prepare_ray_query(scene, max_rays=model.ngeom)
        args = inputs(tx, [7, 0], origins, directions)
        plan.sample(*args, rows=2, far=10.0)
        plan.storage.stream.synchronize()
        assert plan.errors.numpy()[0] == 0
        values = plan.hits.numpy().reshape(8, model.ngeom, 4)[:2]
        actual = plan.labels.numpy().reshape(8, model.ngeom)[:2]
        np.testing.assert_array_equal(actual, np.tile(labels, (2, 1)))
        hit = expected >= 0
        np.testing.assert_allclose(
            values[:, hit, 0], np.tile(expected[hit], (2, 1)), atol=2e-6, rtol=2e-6
        )
        assert np.isposinf(values[:, ~hit, 0]).all()
