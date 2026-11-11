"""Published solver-time observations must survive another world's reset."""

import traceback

import numpy as np
import pytest
from test_physolve_runtime import CONTACT, FREE

from phymeo import PhySolveRuntime
from phymeo.builders.mujoco_compiler import compile_mujoco_xml

pytest_plugins = ["test_physolve_runtime"]


@pytest.mark.cuda
@pytest.mark.parametrize("nworld", [1, 8, 64])
@pytest.mark.parametrize("xml", [FREE, CONTACT], ids=["freefall", "contact"])
def test_masked_reset_keeps_all_published_observations(payload, nworld, xml):
    record = compile_mujoco_xml(xml, require_asset=False)
    with (
        PhySolveRuntime(record, payload=payload, nworld=nworld) as runtime,
        PhySolveRuntime(record, payload=payload, nworld=nworld) as reference,
    ):
        initial = runtime.snapshot()
        for _ in range(3):
            runtime.step()
            reference.step()
        # Deliberately no forward here: xpos/force belong to the step's solve
        # time, not to the new qpos endpoint. The previous test missed this.
        before = runtime.snapshot()
        mask = np.arange(nworld) % 2 == 0
        runtime.reset(mask)
        after = runtime.snapshot()
        for field in before:
            np.testing.assert_array_equal(after[field][~mask], before[field][~mask], err_msg=field)
            np.testing.assert_array_equal(after[field][mask], initial[field][mask], err_msg=field)
        for _ in range(2):
            runtime.step()
            reference.step()
        actual, expected = runtime.snapshot(), reference.snapshot()
        for field in actual:
            np.testing.assert_allclose(
                actual[field][~mask], expected[field][~mask], atol=1e-6, rtol=0, err_msg=field
            )
        assert np.isfinite(runtime.snapshot()["qvel"]).all()


@pytest.mark.cuda
def test_reset_does_not_hide_unselected_forward_error(payload, monkeypatch):
    record = compile_mujoco_xml(FREE, require_asset=False)
    with PhySolveRuntime(record, payload=payload, nworld=8) as runtime:
        runtime.step()
        before = runtime.snapshot()
        native_forward = runtime._native._forward_runtime

        def invalid(handle):
            native_forward(handle)
            index = runtime._handles.index(handle)
            data = runtime._banks[index]
            bad = np.zeros((8, 6), np.float32)
            bad[1, 0] = np.nan  # this world is not selected for reset
            data.qacc.assign(bad, stream=runtime._stream)

        monkeypatch.setattr(runtime._native, "_forward_runtime", invalid)
        with pytest.raises(RuntimeError, match="non-finite.*qacc"):
            runtime.reset(np.arange(8) % 2 == 0)
        for field in before:
            np.testing.assert_array_equal(runtime.snapshot()[field], before[field])
        monkeypatch.setattr(runtime._native, "_forward_runtime", native_forward)
        runtime.reset(np.arange(8) % 2 == 0)
        runtime.step()


@pytest.mark.cuda
def test_reset_copy_failure_keeps_previous_publication(payload, monkeypatch):
    record = compile_mujoco_xml(FREE, require_asset=False)
    with PhySolveRuntime(record, payload=payload, nworld=8) as runtime:
        runtime.step()
        before = runtime.snapshot()
        preserve = runtime._reset.preserve_observations

        def rejected(index, stream):
            preserve(index, stream)
            raise RuntimeError("injected observation copy failure")

        monkeypatch.setattr(runtime._reset, "preserve_observations", rejected)
        with pytest.raises(RuntimeError, match="copy failure"):
            runtime.reset(np.arange(8) % 2 == 0)
        for field in before:
            np.testing.assert_array_equal(runtime.snapshot()[field], before[field])
        monkeypatch.setattr(runtime._reset, "preserve_observations", preserve)
        runtime.reset(np.arange(8) % 2 == 0)


@pytest.mark.cuda
def test_reset_prepared_bindings_and_completion_cost(payload, monkeypatch):
    record = compile_mujoco_xml(FREE, require_asset=False)
    with PhySolveRuntime(record, payload=payload, nworld=8) as runtime:
        runtime.step()
        bindings = tuple(
            id(view)
            for plans in runtime._reset.preserve_plans
            for source, target, _ in plans
            for view in (source, target)
        )
        count = 0
        complete = runtime._validation.read
        factories = {name: getattr(runtime._tx, name) for name in ("zeros", "full", "empty")}
        allocations = []

        def tracked_factory(name):
            def tracked(*args, **kwargs):
                result = factories[name](*args, **kwargs)
                origin = next(
                    frame.name
                    for frame in reversed(traceback.extract_stack())
                    if "physolve_tyx" in frame.filename and "_src" in frame.filename
                )
                allocations.append((name, tuple(result.shape), str(result.dtype), origin))
                return result

            return tracked

        def counted(stream):
            nonlocal count
            count += 1
            complete(stream)

        def allocation_forbidden(*args, **kwargs):
            raise AssertionError("prepared reset cannot allocate a state bank or scratch")

        def without_allocations(operation):
            def checked(*args, **kwargs):
                with monkeypatch.context() as guard:
                    for name in (*factories, "clone"):
                        guard.setattr(runtime._tx, name, allocation_forbidden)
                    return operation(*args, **kwargs)

            return checked

        monkeypatch.setattr(runtime._validation, "read", counted)
        monkeypatch.setattr(runtime._tx, "clone", allocation_forbidden)
        for name in factories:
            monkeypatch.setattr(runtime._tx, name, tracked_factory(name))
        for name in ("restore", "preserve_observations"):
            monkeypatch.setattr(
                runtime._reset, name, without_allocations(getattr(runtime._reset, name))
            )
        # Prepared common scratch and no-output sensor dispatch remove the B1c
        # factory calls. This is not a whole-path low-level allocator claim.
        runtime.forward()
        assert count == 1
        baseline = tuple(allocations)
        print("native forward factory baseline", baseline)
        assert baseline == ()
        allocations.clear()
        count = 0
        runtime.reset(np.arange(8) % 2 == 0)
        assert count == 1  # one compact read completes forward and observation preservation
        assert tuple(allocations) == baseline
        allocations.clear()
        runtime.reset()
        assert count == 2  # full reset also has one compact completion
        assert tuple(allocations) == baseline
        print("reset inherited native forward allocations", baseline)
        assert bindings == tuple(
            id(view)
            for plans in runtime._reset.preserve_plans
            for source, target, _ in plans
            for view in (source, target)
        )
