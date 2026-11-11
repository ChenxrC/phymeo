"""N1 CUDA Torch inference qualification; fixtures frozen before execution.

No old test helpers are imported. CUDA absence is a qualification failure,
not a CPU fallback or a silently skipped positive matrix.
"""

import hashlib
import importlib.util
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import pytest

from phymeo.backends.mujoco_runtime import MuJoCoRuntime

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "n1_cuda_policy_example", ROOT / "examples" / "mujoco_tyx_cuda_policy.py"
)
example = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(example)

pytestmark = [pytest.mark.cuda, pytest.mark.tyx]
CASES = tuple(
    ("hinge-slide", count, stream) for count in (1, 8, 32) for stream in ("default", "nondefault")
) + tuple(("support-slide", 8, stream) for stream in ("default", "nondefault"))
STEPS = 8
XML_SHA256 = {
    "hinge-slide": "8596be527027c85f5c6f4f17d64ababf5820a75771943da71e18509347cb16e7",
    "support-slide": "bf4eb00b8d0eb00ac26be3f5fa31c2972fcde895ce5290bc5bdde60d45d11e92",
}
EXPECTED_LAYOUT = {"hinge-slide": (2, 2, 1, 3), "support-slide": (1, 1, 1, 2)}
EXPECTED_H = {"hinge-slide": 0.002, "support-slide": 0.001}
Q_ATOL, V_ATOL = 2e-4, 3e-3
FORCE_TOL = dict(rtol=3e-3, atol=3e-3)


def resources(stack, case="support-slide", count=2):
    torch = example.require_cuda()
    source = example.fixture_source(case)
    model = source.fresh_native_model()
    keys = example.world_keys(count)
    initial = example.initial_states(case, model, keys)
    runtime = stack.enter_context(
        MuJoCoRuntime(source, mode="cuda", world_keys=keys, initial_states=initial)
    )
    assert (model.nq, model.nv, model.nu, model.nbody) == EXPECTED_LAYOUT[case]
    return torch, source, model, initial, runtime


def actions(torch, model, count=2):
    return (
        torch.full((count, model.nu), 0.015, dtype=torch.float32, device="cuda"),
        torch.full((count, model.nv), 0.008, dtype=torch.float32, device="cuda"),
        torch.full((count, model.nbody, 6), 0.003, dtype=torch.float32, device="cuda"),
    )


def raw_submit(runtime, values, stream=None, revision=None):
    return runtime.set_controls_from_device(
        values[0],
        qfrc_applied=values[1],
        xfrc_applied=values[2],
        expected_revision=runtime.checkpoint_revision if revision is None else revision,
        producer_stream=stream,
    )


def borrow(stack, tensors):
    from tyx.torch import from_torch

    result = []
    for tensor in tensors:
        view = from_torch(tensor, backend="cuda")
        stack.callback(view.free)
        result.append(view)
    return tuple(result)


def unchanged(runtime, before, revision):
    assert runtime.checkpoint_revision == revision
    assert runtime.checkpoint().to_bytes() == before


@pytest.mark.parametrize("case,count,stream", CASES)
def test_actual_cuda_mlp_continuous_physics_and_complete_host_route(case, count, stream, tmp_path):
    assert hashlib.sha256(example.XML[case].encode()).hexdigest() == XML_SHA256[case]
    summary = example.run_case(case, count, stream, tmp_path / "trajectory.jsonl")
    assert summary["steps"] == STEPS
    assert summary["world_steps"] == count * STEPS
    assert summary["h"] == EXPECTED_H[case]
    assert summary["state_and_receipt_comparison"] == "exact-complete-same-FP32-input"
    assert summary["policy_device"] == "cuda:0"
    assert summary["all_parameters_cuda"]
    assert summary["all_inputs_outputs_cuda"]
    assert summary["device_counters"]["device_control_submissions"] == STEPS
    assert summary["device_counters"]["accepted_world_steps"] == count * STEPS
    native_width = sum(
        (EXPECTED_LAYOUT[case][2], EXPECTED_LAYOUT[case][1], 6 * EXPECTED_LAYOUT[case][3])
    )
    assert (
        summary["device_counters"]["control_audit_downloaded_bytes"]
        == STEPS * count * native_width * 4
    )
    assert summary["task_counters"]["output_d2d_bytes"] > 0
    assert summary["diagnostic_action_readback_bytes"] == STEPS * count * native_width * 4
    if case == "support-slide":
        assert all(value > 0 for value in summary["loaded_contact_steps_by_world"])
    assert all(value > 0 for value in summary["moved_worlds"])


@pytest.mark.parametrize(
    "bad", ("cpu", "dtype", "shape", "stride", "nan", "inf", "subnormal", "negative_subnormal")
)
def test_torch_bad_input_preserves_previous_accepted_packet(bad):
    with ExitStack() as stack:
        torch, _, model, _, runtime = resources(stack)
        good = actions(torch, model)
        stream = torch.cuda.default_stream()
        example.submit_torch_controls(
            runtime, good, stream, expected_revision=runtime.checkpoint_revision
        )
        before, revision = runtime.checkpoint().to_bytes(), runtime.checkpoint_revision
        invalid = list(actions(torch, model))
        if bad == "cpu":
            invalid[0] = invalid[0].cpu()
        elif bad == "dtype":
            invalid[0] = invalid[0].to(dtype=torch.float64)
        elif bad == "shape":
            invalid[0] = torch.zeros((2, model.nu + 1), device="cuda")
        elif bad == "stride":
            invalid[0] = torch.zeros((2, 2 * model.nu), device="cuda")[:, ::2]
            assert not invalid[0].is_contiguous()
        else:
            word = {
                "nan": 0x7FC00001,
                "inf": 0x7F800000,
                "subnormal": 1,
                "negative_subnormal": -2147483647,
            }[bad]
            # Integer storage writes preserve the invalid FP32 word without arithmetic FTZ.
            invalid[0].view(torch.int32)[0, 0] = word
        with pytest.raises((ValueError, TypeError, RuntimeError)):
            example.submit_torch_controls(
                runtime, tuple(invalid), stream, expected_revision=revision
            )
        unchanged(runtime, before, revision)
        assert dict(runtime.step(EXPECTED_H["support-slide"])) == dict.fromkeys(
            runtime.world_keys, True
        )
        for index, state in enumerate(runtime.snapshot().values()):
            np.testing.assert_array_equal(
                state.last_step.stored_control.ctrl, good[0][index].cpu().numpy()
            )


def test_torch_channel_alias_rejected_before_commit():
    with ExitStack() as stack:
        torch, _, model, _, runtime = resources(stack)
        values = list(actions(torch, model))
        values[1] = values[0]  # Native nu == nv == 1, so shape checks alone cannot catch this.
        before, revision = runtime.checkpoint().to_bytes(), runtime.checkpoint_revision
        with pytest.raises(ValueError, match="overlap"):
            example.submit_torch_controls(
                runtime, tuple(values), torch.cuda.default_stream(), expected_revision=revision
            )
        unchanged(runtime, before, revision)


@pytest.mark.parametrize("closed", ("view", "stream"))
def test_closed_borrowed_torch_handle_preserves_accepted_state(closed):
    from tyx.backends.cuda_runtime import CudaBackendError
    from tyx.torch import stream_from_torch

    with ExitStack() as stack:
        torch, _, model, _, runtime = resources(stack)
        tensors = actions(torch, model)
        views = borrow(stack, tensors)
        producer = torch.cuda.Stream()
        producer.wait_stream(torch.cuda.current_stream())
        wrapped = stream_from_torch(producer, backend="cuda")
        stack.callback(wrapped.close)
        if closed == "view":
            views[0].free()
        else:
            wrapped.close()
        before, revision = runtime.checkpoint().to_bytes(), runtime.checkpoint_revision
        with pytest.raises(
            (ValueError, RuntimeError, CudaBackendError), match="destroyed|freed|closed"
        ):
            raw_submit(runtime, views, wrapped)
        unchanged(runtime, before, revision)
        # Closing the borrowed handle never owns or destroys the original Torch allocation/stream.
        with torch.cuda.stream(producer):
            result = tensors[0] + 1
        producer.synchronize()
        assert torch.all(result > 1).item()


def test_stale_revision_and_frame_refused_then_fresh_step():
    with ExitStack() as stack:
        torch, _, model, _, runtime = resources(stack)
        workspace = stack.enter_context(runtime.prepare_task_workspace(example.task_spec(model)))
        frame = workspace.refresh(expected_revision=runtime.checkpoint_revision)
        workspace.to_torch(frame)
        revision = runtime.checkpoint_revision
        values = actions(torch, model)
        example.submit_torch_controls(
            runtime, values, torch.cuda.default_stream(), expected_revision=revision
        )
        before = runtime.checkpoint().to_bytes()
        with pytest.raises(ValueError, match="stale"):
            example.submit_torch_controls(
                runtime, values, torch.cuda.default_stream(), expected_revision=revision
            )
        unchanged(runtime, before, revision + 1)
        with pytest.raises(ValueError, match="stale"):
            workspace.to_torch(frame)
        assert dict(runtime.step(EXPECTED_H["support-slide"])) == dict.fromkeys(
            runtime.world_keys, True
        )
        fresh = workspace.refresh(expected_revision=runtime.checkpoint_revision)
        assert workspace.to_torch(fresh)[0].is_cuda


def test_owned_observation_and_submitted_actions_outlive_borrowed_owners():
    with ExitStack() as stack:
        torch, source, model, initial, runtime = resources(stack)
        reference = stack.enter_context(
            MuJoCoRuntime(
                source, mode="cuda", world_keys=runtime.world_keys, initial_states=initial
            )
        )
        workspace = runtime.prepare_task_workspace(example.task_spec(model))
        stack.callback(workspace.close)
        frame = workspace.refresh(expected_revision=runtime.checkpoint_revision)
        observation, reward, terminated = workspace.to_torch(frame)
        saved_observation = observation.clone()
        workspace.close()
        producer = torch.cuda.Stream()
        policy = example.make_policy(model, torch.device("cuda:0"))
        with torch.cuda.stream(producer), torch.inference_mode():
            output = policy(observation)
            values = example.native_channels(output, model)
        receipt = example.submit_torch_controls(
            runtime, values, producer, expected_revision=runtime.checkpoint_revision
        )
        producer.synchronize()
        host_values = tuple(value.cpu().numpy().copy() for value in values)
        example.host_submit(reference, host_values)
        with torch.cuda.stream(producer), torch.inference_mode():
            for value in values:
                value.fill_(17)
        producer.synchronize()
        del values, output
        for _ in range(2):
            assert dict(runtime.step(EXPECTED_H["support-slide"])) == dict.fromkeys(
                runtime.world_keys, True
            )
            assert dict(reference.step(EXPECTED_H["support-slide"])) == dict.fromkeys(
                reference.world_keys, True
            )
            example.assert_exact(runtime.snapshot(), reference.snapshot())
        assert torch.equal(observation, saved_observation)
        assert reward.is_cuda and terminated.is_cuda
        for i, key in enumerate(runtime.world_keys):
            np.testing.assert_array_equal(receipt.controls[key].ctrl, host_values[0][i])


def test_negative_zero_word_is_preserved_in_native_control_receipt():
    with ExitStack() as stack:
        torch, _, model, _, runtime = resources(stack)
        values = actions(torch, model)
        values[0].view(torch.int32)[0, 0] = -2147483648
        receipt = example.submit_torch_controls(
            runtime,
            values,
            torch.cuda.default_stream(),
            expected_revision=runtime.checkpoint_revision,
        )
        stored = np.asarray(receipt.controls[runtime.world_keys[0]].ctrl, dtype=np.float32)
        assert stored.view(np.uint32)[0] == 0x80000000
        assert dict(runtime.step(EXPECTED_H["support-slide"])) == dict.fromkeys(
            runtime.world_keys, True
        )


def test_real_second_device_input_refused_when_hardware_exists():
    with ExitStack() as stack:
        torch, _, model, _, runtime = resources(stack)
        if torch.cuda.device_count() < 2:
            pytest.skip("UNTESTED_CROSS_DEVICE: only one CUDA device; not an interop pass")
        values = list(actions(torch, model))
        values[0] = values[0].to("cuda:1")
        before, revision = runtime.checkpoint().to_bytes(), runtime.checkpoint_revision
        with pytest.raises(ValueError, match="device"):
            example.submit_torch_controls(
                runtime, tuple(values), torch.cuda.default_stream(), expected_revision=revision
            )
        unchanged(runtime, before, revision)
