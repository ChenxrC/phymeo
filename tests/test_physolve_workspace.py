"""Prepared native workspaces: installed before/after and real Host execution."""

import os
import time
from pathlib import Path

import numpy as np
import pytest
from test_native_exclusions import connection_plan, fanuc_template
from test_physolve_runtime import CONTACT, FREE

from phymeo import PhySolveRuntime
from phymeo.builders.mujoco_compiler import compile_mujoco_xml

pytest_plugins = ["test_physolve_runtime"]


def capture(payload, kind, nworld, *, inspect_workspace=False):
    """Capture public values and valid contacts; never compare unused pool entries."""
    if kind == "fanuc":
        source = fanuc_template()
        template = source.with_collision_exclusions(connection_plan(source))
        runtime = template.bind(nworld).create_runtime(payload=payload, nconmax=128)
    else:
        record = compile_mujoco_xml(FREE if kind == "free" else CONTACT, require_asset=False)
        runtime = PhySolveRuntime(record, payload=payload, nworld=nworld)
    results, windows = {}, []
    with runtime:
        def save(label):
            results.update({f"{label}_{k}": v for k, v in runtime.snapshot().items()})
            data = runtime._banks[runtime._active]
            count = int(data.nacon.numpy()[0])
            worlds = data.contact.worldid.numpy()[:count]
            geoms = data.contact.geom.numpy()[:count]
            order = np.lexsort((geoms[:, 1], geoms[:, 0], worlds))
            for field in ("worldid", "geom", "dist", "pos", "frame"):
                results[f"{label}_contact_{field}"] = getattr(data.contact, field).numpy()[:count][order]

        if inspect_workspace:
            spaces = [data._execution_workspace for data in runtime._banks]
            pointers = [space.moment_nnz.ptr for space in spaces]
            assert len(set(pointers)) == 2
        save("initial")
        force = np.full((nworld, runtime._cpu_model.nv), .03, np.float32) if kind == "fanuc" else None
        for _ in range(4):
            runtime.step(qfrc_applied=force)
        save("step")
        runtime.forward()
        save("forward")
        runtime.reset(np.arange(nworld) % 2 == 0)
        save("masked")
        runtime.step()
        save("continued")
        runtime.reset()
        save("reset")
        # Explicit completion is inside step. Timing is Runtime-only, without GUI.
        # The B1c FANUC long-window run failed with CUDA 719 before this change.
        # Keep its short qualified trajectory; no FANUC timing qualification.
        for _ in range(0 if kind == "fanuc" else 5):
            start = time.perf_counter()
            for _ in range(5):
                runtime.step(qfrc_applied=force)
            windows.append((time.perf_counter() - start) * 1000 / 5)
        if inspect_workspace:
            assert pointers == [data._execution_workspace.moment_nnz.ptr for data in runtime._banks]
    return results, windows


@pytest.mark.cuda
@pytest.mark.parametrize("nworld", [1, 8, 64])
@pytest.mark.parametrize("kind", ["free", "contact", "fanuc"])
def test_installed_workspace_matches_frozen_trajectory(payload, nworld, kind):
    location = os.environ.get("PHYMEO_WORKSPACE_BASELINE")
    if not location:
        pytest.skip("explicit installed B1c reference required")
    expected = np.load(Path(location) / f"{kind}-{nworld}.npz")
    actual, windows = capture(payload, kind, nworld, inspect_workspace=True)
    assert set(actual) == set(expected.files)
    for key, value in actual.items():
        np.testing.assert_allclose(value, expected[key], atol=1e-6, rtol=0, err_msg=key)
    print("workspace Runtime step ms", kind, nworld, windows)


@pytest.mark.cuda
@pytest.mark.parametrize("nworld", [1, 8, 64])
def test_prepared_host_no_array_factories_and_repeated_reset(payload, monkeypatch, nworld):
    record = compile_mujoco_xml(CONTACT, require_asset=False)
    with PhySolveRuntime(record, payload=payload, nworld=nworld) as runtime:
        runtime.step()
        initial = runtime.snapshot()
        spaces = [d._execution_workspace for d in runtime._banks]
        ids = [id(s) for s in spaces]

        def forbidden(*args, **kwargs):
            raise AssertionError("steady native Host must reuse its prepared workspace")

        for name in ("zeros", "full", "empty", "clone"):
            monkeypatch.setattr(runtime._tx, name, forbidden)
        for _ in range(3):
            for space in spaces:
                space.moment_nnz.fill_(123)  # must be cleared on every transmission
            runtime.step()
            runtime.forward()
            runtime.reset(np.arange(nworld) % 2 == 0)
            runtime.reset()
        assert ids == [id(d._execution_workspace) for d in runtime._banks]
        assert np.isfinite(runtime.snapshot()["qvel"]).all()
        assert initial["qpos"].shape == runtime.snapshot()["qpos"].shape


@pytest.mark.cuda
def test_collision_workspace_reuse_contact_disappear_return(payload):
    """Valid-contact counters, not old scratch contents, delimit each new collision."""
    record = compile_mujoco_xml(CONTACT, require_asset=False)
    with PhySolveRuntime(record, payload=payload, nworld=8) as prepared, \
            PhySolveRuntime(record, payload=payload, nworld=8) as original:
        spaces = [d._execution_workspace for r in (prepared, original) for d in r._banks]
        assert len({s.moment_nnz.ptr for s in spaces}) == 4
        # Reference retains the original per-call collision/transmission factories.
        for bank in original._banks:
            del bank._execution_workspace
        for height in (.08, 1., .08):
            for runtime in (prepared, original):
                with runtime._scope():
                    data = runtime._banks[runtime._active]
                    qpos = data.qpos.numpy()
                    qpos[:, 2] = height
                    source = runtime._tx.Array(qpos)
                    runtime._tx.copy(data.qpos, source)
                    runtime._complete()  # cold test input retains source until copy completion
                runtime.forward()
            for field, value in prepared.snapshot().items():
                np.testing.assert_allclose(value, original.snapshot()[field], atol=1e-6, rtol=0)
            counts = [int(r._banks[r._active].nacon.numpy()[0]) for r in (prepared, original)]
            assert counts == ([0, 0] if height == 1. else [8, 8])
            for field in ("dist", "pos", "frame", "geom", "worldid"):
                left, right = [getattr(r._banks[r._active].contact, field).numpy()[:counts[0]]
                               for r in (prepared, original)]
                # Contacts may be atomically appended in a different world order.
                order = [np.argsort(r._banks[r._active].contact.worldid.numpy()[:counts[0]])
                         for r in (prepared, original)]
                np.testing.assert_allclose(left[order[0]], right[order[1]], atol=1e-6, rtol=0)


@pytest.mark.cuda
def test_workspace_owner_survives_host_prepare_and_cleanup_failure(payload, monkeypatch):
    native = payload.load()
    prepare, close = native._prepare_runtime, native._close_runtime

    def failed_prepare(handle):
        prepare(handle)
        raise RuntimeError("injected prepared Host failure")

    def failed_close(*args):
        raise RuntimeError("injected Host cleanup failure")

    monkeypatch.setattr(native, "_prepare_runtime", failed_prepare)
    monkeypatch.setattr(native, "_close_runtime", failed_close)
    with pytest.raises(RuntimeError, match="prepared Host failure") as captured:
        PhySolveRuntime(compile_mujoco_xml(FREE, require_asset=False), payload=payload)
    retained = captured.value.runtime
    try:
        assert retained._handles[0] is not None
        assert retained._banks[0]._execution_workspace._data() is retained._banks[0]
        assert retained._stream is not None
        with pytest.raises(RuntimeError, match="closing"):
            retained.forward()
    finally:
        monkeypatch.setattr(native, "_close_runtime", close)
        retained.close()
    assert retained._closed and not retained._banks and retained._model is None
