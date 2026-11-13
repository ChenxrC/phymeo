"""Current forward must neither integrate nor replace accepted solver observations."""

import numpy as np
import pytest

from phymeo import PhySolveRuntime
from phymeo.builders.mujoco_compiler import compile_mujoco_xml

pytest_plugins = ["test_physolve_runtime"]


@pytest.mark.cuda
@pytest.mark.parametrize("nworld", [1, 8, 64])
def test_current_frame_preserves_authority_and_pending_candidate(payload, nworld):
    scene = compile_mujoco_xml(
        """<mujoco><option timestep=".002" gravity="0 0 0"/>
    <worldbody><body name="link"><joint type="slide" axis="1 0 0"/>
    <geom size=".1" mass="2"/><site pos=".1 0 0"/></body></worldbody></mujoco>""",
        require_asset=False,
    )
    with PhySolveRuntime(scene, payload=payload, nworld=nworld) as runtime:
        frame = runtime.prepare_current_observations(("qM", "qfrc_bias", "site_xpos"))
        before = runtime.snapshot()
        frame.refresh()
        with pytest.raises(RuntimeError, match="complete the idle"):
            runtime.begin_execution()
        frame.complete()
        for key, value in before.items():
            np.testing.assert_array_equal(runtime.snapshot()[key], value)
        assert frame.arrays["qM"].shape[0] == nworld
        # Frontend qM is tiled/padded storage, not a public dense nv-by-nv matrix.
        mass = frame.arrays["qM"].numpy().reshape(nworld, -1)
        np.testing.assert_array_equal(mass[:, 0], 2.0)
        np.testing.assert_array_equal(mass[:, 1:], 0.0)
        ticket = runtime.begin_execution()
        storage = runtime.storage_binding()
        storage.borrow("qfrc_applied", runtime.writable_bank, write=True).assign(
            np.full((nworld, 1), 6.0, np.float32), stream=storage.stream
        )
        runtime.enqueue_step(ticket)
        frame.refresh()
        runtime.complete_execution(ticket)
        with pytest.raises(RuntimeError, match="after execution completion"):
            frame.refresh()
        np.testing.assert_allclose(
            frame.arrays["site_xpos"].numpy()[:, 0, 0], 0.100012, rtol=1e-6, atol=1e-8
        )
        for key, value in before.items():
            np.testing.assert_array_equal(runtime.snapshot()[key], value)
        runtime.discard_execution(ticket)
        frame.refresh()
        frame.complete()
        np.testing.assert_allclose(
            frame.arrays["site_xpos"].numpy()[:, 0, 0], 0.1, rtol=1e-6, atol=1e-8
        )
        frame.close()
        with pytest.raises(RuntimeError, match="closed"):
            frame.refresh()


@pytest.mark.cuda
def test_complete_scene_ft_admission_does_not_expand_default_sensor_profile(payload, tmp_path):
    from test_native_scene import SCENE

    from phymeo.builders.native_scene import compile_native_scene

    source = tmp_path / "scene.xml"
    source.write_text(SCENE)
    template = compile_native_scene(source, name="task")
    with pytest.raises(NotImplementedError, match="sensors"):
        PhySolveRuntime(template.compilation, payload=payload)
    with template.bind(1).create_runtime(payload=payload) as runtime:
        frame = runtime.prepare_current_observations(("sensordata",))
        frame.refresh()
        frame.complete()
        assert frame.arrays["sensordata"].shape == (1, 6)
        assert np.isfinite(frame.arrays["sensordata"].numpy()).all()
        assert (
            runtime._cpu_model.opt.jacobian
            == template.compilation.fresh_native_model().opt.jacobian
        )
    source.write_text(SCENE.replace('<force name="force"', '<accelerometer name="force"'))
    template = compile_native_scene(source, name="task")
    with pytest.raises(NotImplementedError, match="sensors"):
        template.bind(1).create_runtime(payload=payload)


@pytest.mark.cuda
def test_current_forward_error_and_close_retry(payload, monkeypatch):
    from contextlib import closing

    import tyx as tx
    from test_physolve_runtime import SLIDER

    runtime = PhySolveRuntime(compile_mujoco_xml(SLIDER, require_asset=False), payload=payload)
    try:
        frame = runtime.prepare_current_observations(("qvel",))
        storage = runtime.storage_binding()
        with closing(tx.CudaStream(device=0)) as other:
            with pytest.raises(NotImplementedError, match="fixed stream"):
                runtime.bind_stream(int(other.handle), owner=other)
        # Fault injection bypasses the public input precheck deliberately.
        source = storage.borrow("qvel", runtime.active_bank)
        source.assign(np.array([[np.nan]], np.float32), stream=storage.stream)
        frame.refresh()
        with pytest.raises(RuntimeError, match="current-forward"):
            frame.complete()
        np.testing.assert_array_equal(frame.arrays["qvel"].numpy(), 0.0)
        source.assign(np.array([[0]], np.float32), stream=storage.stream)
        frame.refresh()
        frame.complete()
        ticket = runtime.begin_execution()
        storage.borrow("qfrc_applied", runtime.writable_bank, write=True).assign(
            np.array([[6]], np.float32), stream=storage.stream
        )
        runtime.enqueue_step(ticket)
        runtime.complete_execution(ticket)
        runtime.accept_execution(ticket)
        frame.refresh()
        frame.complete()
        np.testing.assert_allclose(frame.arrays["qvel"].numpy(), 0.006, rtol=1e-6, atol=1e-8)
        assert runtime.active_bank == 1
        frame.close()
        bad = runtime.begin_execution()
        storage.borrow("qvel", runtime.writable_bank, write=True).assign(
            np.array([[np.nan]], np.float32), stream=storage.stream
        )
        runtime.discard_execution(bad)
        frame = runtime.prepare_current_observations(("qvel",))
        np.testing.assert_allclose(frame.arrays["qvel"].numpy(), 0.006, rtol=1e-6, atol=1e-8)
        original = runtime._native._close_runtime

        def fail(handle, completed):
            if handle is frame.handle:
                raise RuntimeError("injected frame close failure")
            return original(handle, completed)

        with monkeypatch.context() as patch:
            patch.setattr(runtime._native, "_close_runtime", fail)
            with pytest.raises(RuntimeError, match="injected frame close"):
                frame.close()
            with pytest.raises(RuntimeError, match="closing"):
                runtime.bind_stream(0, owner=None)
            assert frame.handle is not None and not frame.closed
    finally:
        runtime.close()
    assert frame.closed
