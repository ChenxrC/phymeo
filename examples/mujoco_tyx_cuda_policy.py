"""Actual CUDA Torch MLP -> public MuJoCo-TYX actions -> continuous physics.

This is inference qualification, not PPO. The independent comparison runtime
uses the same CUDA physics with the public HOST control entry, and never copies
state from the device-action runtime. Complete physics HOST validation remains.
"""

import argparse
import hashlib
import json
import sys
import traceback
from collections.abc import Mapping
from contextlib import ExitStack
from dataclasses import fields, is_dataclass
from pathlib import Path

import numpy as np

from phymeo.backends.mujoco_runtime import MuJoCoControl, MuJoCoInitialState, MuJoCoRuntime
from phymeo.backends.mujoco_task_workspace import MuJoCoTyxTaskSpec
from phymeo.builders.mujoco_compiler import compile_mujoco_xml

STEPS = 8
H = {"hinge-slide": 0.002, "support-slide": 0.001}
MATRIX = {"hinge-slide": (1, 8, 32), "support-slide": (8,)}
XML = {
    "hinge-slide": """<mujoco><compiler angle="radian" autolimits="true"/>
<option timestep=".002" gravity="0 0 -9.81" integrator="Euler" cone="pyramidal"
 iterations="128" tolerance="1e-10"/>
<default><geom condim="3" friction=".6 .02 .003" solref=".02 1"/></default>
<worldbody><body pos=".2 0 .1"><joint name="a" type="hinge" axis="0 1 0" damping=".03" armature=".02"/>
<inertial pos=".02 -.01 .03" mass="1.2" fullinertia=".04 .05 .06 .001 -.001 .001"/>
<geom type="sphere" size=".04" contype="0" conaffinity="0"/>
<body pos=".1 .1 0"><joint name="b" type="slide" axis="1 0 0" damping=".03" stiffness=".4" springref=".04"/>
<inertial pos=".02 -.01 .03" mass="1.2" fullinertia=".04 .05 .06 .001 -.001 .001"/>
<geom type="sphere" size=".04" contype="0" conaffinity="0"/>
</body></body></worldbody><actuator><general joint="a" gaintype="fixed" gainprm="1.2"
 biastype="affine" biasprm="0 -.2 -.04" ctrllimited="true" ctrlrange="-.05 .05"
 forcelimited="true" forcerange="-.08 .08"/></actuator></mujoco>""",
    "support-slide": """<mujoco model="t3-normal"><compiler angle="radian" fusestatic="false"/>
<option timestep=".001" gravity="0 0 -9.81" integrator="Euler" solver="Newton"
 iterations="128" tolerance="1e-10" jacobian="dense"/>
<default><geom condim="1" solref=".02 1" solimp=".8 .9 .002 .5 2"/>
<joint armature=".01" damping=".04"/></default>
<worldbody><geom name="platform" type="box" size=".24 .18 .05"/>
<body name="root" pos="0 0 .109"><joint name="vertical" type="slide" axis="0 0 1" armature="0" damping="0"/>
<inertial mass="2.0" pos="0 0 0" fullinertia=".08 .10 .12 .002 -.001 .003"/>
<geom name="moving" type="box" size=".12 .08 .06"/></body></worldbody>
<actuator><motor joint="vertical"/></actuator></mujoco>""",
}


def require_cuda():
    import torch

    if torch.version.cuda is None or not torch.cuda.is_available():
        raise RuntimeError("N1 requires qualified CUDA PyTorch; CPU fallback is forbidden")
    torch.cuda.set_device(0)
    return torch


def fixture_source(case):
    return compile_mujoco_xml(XML[case], require_asset=False)


def world_keys(count):
    return tuple(5 + 36 * i for i in range(count))


def initial_states(case, model, keys):
    result = {}
    for index, key in enumerate(keys):
        q = model.qpos0.copy()
        if case == "hinge-slide":
            v = np.array([0.015, 0.02 if index == 0 else 0.08])
            q[-1] = 0.03 if index == 0 else 0.04
        else:
            v = np.array([-0.02])
        result[key] = MuJoCoInitialState(q, v)
    return result


def task_spec(model):
    width = model.nq + model.nv
    return MuJoCoTyxTaskSpec(
        tuple(range(model.nq)), tuple(range(model.nv)),
        scale=[1 + i / 8 for i in range(width)], bias=[0.05] * width,
        target=[0] * width, weight=[1] * width, lower=[-2] * width, upper=[2] * width,
    )


def make_policy(model, device):
    import torch

    network = torch.nn.Sequential(
        torch.nn.Linear(model.nq + model.nv, 8), torch.nn.Tanh(),
        torch.nn.Linear(8, model.nu + model.nv + 6), torch.nn.Tanh(),
    ).to(device=device, dtype=torch.float32).eval()
    # Fixed small nonzero weights, independent of runtime results or random seeds.
    with torch.no_grad():
        for layer in (network[0], network[2]):
            sequence = torch.arange(layer.weight.numel(), device=device, dtype=torch.float32)
            layer.weight.copy_(((sequence.remainder(5) - 2) * 0.01).reshape_as(layer.weight))
            layer.bias.copy_(0.01 + 0.001 * torch.arange(layer.bias.numel(), device=device, dtype=torch.float32))
    # Construction may occur outside the later producer-stream context.
    torch.cuda.current_stream(device).synchronize()
    return network


def native_channels(output, model):
    import torch

    if not isinstance(output, torch.Tensor) or not output.is_cuda or output.dtype != torch.float32:
        raise ValueError("policy output must be a CUDA FP32 tensor")
    if output.ndim != 2 or output.shape[1] != model.nu + model.nv + 6 or output.shape[0] <= 0:
        raise ValueError("policy output width/world count differs from native channel layout")
    ctrl = (0.01 + 0.02 * output[:, :model.nu]).contiguous()
    force = (0.005 + 0.01 * output[:, model.nu:model.nu + model.nv]).contiguous()
    wrench = torch.zeros((output.shape[0], model.nbody, 6), dtype=torch.float32, device=output.device)
    wrench[:, -1, :] = 0.001 * (torch.arange(6, device=output.device, dtype=torch.float32) + 1) + 0.01 * output[:, -6:]
    return ctrl, force, wrench


def submit_torch_controls(runtime, tensors, producer_stream, *, expected_revision):
    """Borrow contiguous Torch storage until the public copy-owned submit returns.

    All three original tensors and the producer stream remain strongly owned in
    this call. Action publication and a subsequent physics step are separate.
    This convenience function never runs policy scaling or a HOST fallback.
    """
    import torch
    from tyx.torch import from_torch, stream_from_torch

    if not isinstance(tensors, (tuple, list)) or len(tensors) != 3:
        raise ValueError("exactly ctrl, qfrc_applied and xfrc_applied are required")
    if runtime.capabilities.get("implementation") != "tyx" or runtime.capabilities.get("mode") != "cuda":
        raise ValueError("policy input requires the public TYX CUDA runtime")
    if not isinstance(producer_stream, torch.cuda.Stream):
        raise ValueError("an explicit Torch CUDA producer stream is required")
    count, info = len(runtime.world_keys), runtime.capabilities
    shapes = ((count, info["nu"]), (count, info["nv"]), (count, info["nbody"], 6))
    for tensor, shape in zip(tensors, shapes, strict=True):
        if not isinstance(tensor, torch.Tensor) or not tensor.is_cuda:
            raise ValueError("policy native channels must be CUDA tensors")
        if tuple(tensor.shape) != shape or tensor.device != producer_stream.device:
            raise ValueError("native channel shape/device differs from the declared runtime/stream")
        if tensor.dtype != torch.float32 or not tensor.is_contiguous():
            raise ValueError("policy native channels require contiguous FP32 storage")
    with ExitStack() as owners:
        views = []
        for tensor in tensors:
            view = from_torch(tensor, backend="cuda")
            owners.callback(view.free)
            views.append(view)
        stream = stream_from_torch(producer_stream, backend="cuda")
        owners.callback(stream.close)
        return runtime.set_controls_from_device(
            views[0], qfrc_applied=views[1], xfrc_applied=views[2],
            expected_revision=expected_revision, producer_stream=stream,
        )


def host_submit(runtime, values):
    runtime.set_controls({
        key: MuJoCoControl(values[0][i], qfrc_applied=values[1][i], xfrc_applied=values[2][i])
        for i, key in enumerate(runtime.world_keys)
    })


def assert_exact(left, right, path="state"):
    """Complete same-backend comparison, retaining all arrays and receipt fields."""
    if isinstance(left, np.ndarray):
        assert isinstance(right, np.ndarray), path
        assert left.dtype == right.dtype and left.shape == right.shape, path
        assert left.tobytes() == right.tobytes(), path
    elif is_dataclass(left):
        assert type(left) is type(right), path
        for field in fields(left):
            assert_exact(getattr(left, field.name), getattr(right, field.name), path + "." + field.name)
    elif isinstance(left, Mapping):
        assert set(left) == set(right), path
        for key in left:
            assert_exact(left[key], right[key], path + "." + str(key))
    elif isinstance(left, (tuple, list)):
        assert type(left) is type(right) and len(left) == len(right), path
        for i, value in enumerate(left):
            assert_exact(value, right[i], path + f"[{i}]")
    else:
        assert left == right, (path, left, right)


def encoded(value):
    if isinstance(value, np.ndarray):
        raw = value.tobytes()
        return {"dtype": value.dtype.str, "shape": list(value.shape), "hex": raw.hex(), "sha256": hashlib.sha256(raw).hexdigest()}
    if is_dataclass(value):
        return {"type": type(value).__module__ + "." + type(value).__name__, "fields": {field.name: encoded(getattr(value, field.name)) for field in fields(value)}}
    if isinstance(value, Mapping):
        return {str(key): encoded(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [encoded(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def append(stream, value):
    stream.write(json.dumps(encoded(value), allow_nan=False, separators=(",", ":")) + "\n")
    stream.flush()


def task_oracle(states, spec, actual):
    raw = np.array([np.r_[state.q[list(spec.q_indices)], state.v[list(spec.v_indices)]] for state in states.values()], dtype=np.float32)
    observation = raw * spec.scale + spec.bias
    reward = -np.sum(spec.weight * (raw - spec.target) ** 2, axis=1)
    terminated = np.any((raw < spec.lower) | (raw > spec.upper), axis=1).astype(np.int32)
    np.testing.assert_allclose(actual[0], observation, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(actual[1], reward, rtol=2e-5, atol=2e-6)
    np.testing.assert_array_equal(actual[2], terminated)


def module_origins():
    return {name: str(module.__file__) for name, module in sorted(sys.modules.items()) if (name == "torch" or name == "mujoco" or name == "tyx" or name.startswith("phymeo.")) and getattr(module, "__file__", None)}


def run_case(case, count, stream_kind, output):
    """Run the predeclared eight-step qualification, saving full public evidence."""
    if type(count) is not int or case not in MATRIX or count not in MATRIX[case] or stream_kind not in ("default", "nondefault"):
        raise ValueError("case/world/stream is outside the frozen N1 matrix")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    phase = "qualification"
    runtime = reference = None
    last_public = {}
    action_accepted = state_accepted = reference_accepted = False
    with output.open("x", encoding="utf-8") as log:
        try:
            torch = require_cuda()
            import mujoco

            if mujoco.__version__ != "3.3.7":
                raise RuntimeError("N1 native compilation requires MuJoCo 3.3.7")
            source = fixture_source(case)
            model = source.fresh_native_model()
            keys = world_keys(count)
            initial = initial_states(case, model, keys)
            spec = task_spec(model)
            network = make_policy(model, torch.device("cuda:0"))
            producer = torch.cuda.default_stream(0) if stream_kind == "default" else torch.cuda.Stream(device=0)
            parameters = {name: parameter.detach().cpu().numpy().copy() for name, parameter in network.named_parameters()}
            append(log, {"event": "configuration", "case": case, "world_keys": keys, "h": H[case], "steps": STEPS,
                "xml": XML[case], "xml_sha256": hashlib.sha256(XML[case].encode()).hexdigest(),
                "compiled_model_sha256": hashlib.sha256(source.source.model_binary).hexdigest(),
                "source_archive_sha256": hashlib.sha256(source.source.archive_bytes).hexdigest(),
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "native_layout": (model.nq, model.nv, model.nu, model.nbody), "initial": initial, "task": spec,
                "policy": {"architecture": str(network), "parameters": parameters, "devices": {name: str(p.device) for name, p in network.named_parameters()}},
                "torch_version": torch.__version__, "cuda_build": torch.version.cuda, "device_name": torch.cuda.get_device_name(0),
                "producer_stream": stream_kind, "producer_handle": int(producer.cuda_stream),
                "reference_route": "independent-TYX-CUDA-with-HOST-controls", "scope": "inference-not-PPO; complete-physics-HOST-diagnostics-retained"})
            with ExitStack() as owners:
                runtime = owners.enter_context(MuJoCoRuntime(source, mode="cuda", world_keys=keys, initial_states=initial))
                reference = owners.enter_context(MuJoCoRuntime(source, mode="cuda", world_keys=keys, initial_states=initial))
                workspace = owners.enter_context(runtime.prepare_task_workspace(spec))
                start = runtime.snapshot()
                assert_exact(start, reference.snapshot())
                last_public = {"device": start, "reference": reference.snapshot()}
                loaded = [0] * count
                all_cuda = True
                action_readback = task_readback = 0
                for tick in range(STEPS):
                    action_accepted = state_accepted = reference_accepted = False
                    phase = "task-and-policy"
                    before = runtime.snapshot()  # Detached diagnostic reference, never policy input.
                    revision = runtime.checkpoint_revision
                    frame = workspace.refresh(expected_revision=revision)
                    with torch.cuda.stream(producer), torch.inference_mode():
                        exported = workspace.to_torch(frame)
                        raw_action = network(exported[0])
                        values = native_channels(raw_action, model)
                    all_cuda = all_cuda and all(value.is_cuda and value.device.index == 0 for value in (*exported, raw_action, *values))
                    phase = "action-publication"
                    control_receipt = submit_torch_controls(runtime, values, producer, expected_revision=revision)
                    action_accepted = True
                    # These explicit diagnostic copies are separate from the GPU action ingress.
                    host_values = tuple(value.detach().cpu().numpy().copy() for value in values)
                    task_values = tuple(value.detach().cpu().numpy().copy() for value in exported)
                    action_readback += sum(value.nbytes for value in host_values)
                    task_readback += sum(value.nbytes for value in task_values)
                    task_oracle(before, spec, task_values)
                    for i, key in enumerate(keys):
                        for name, expected in zip(("ctrl", "qfrc_applied", "xfrc_applied"), host_values, strict=True):
                            actual = np.asarray(getattr(control_receipt.controls[key], name), dtype=np.float32)
                            assert actual.tobytes() == expected[i].tobytes(), name
                    phase = "host-control-publication"
                    host_submit(reference, host_values)
                    append(log, {"event": "attempt", "tick": tick + 1, "before": before, "task_frame": frame,
                        "task_outputs": task_values, "native_action": host_values, "action_receipt": control_receipt,
                        "raw_policy_output": raw_action.detach().cpu().numpy(), "action_accepted": True})
                    phase = "device-physics"
                    accepted = runtime.step(H[case])
                    state_accepted = bool(accepted) and all(accepted.values())
                    assert dict(accepted) == dict.fromkeys(keys, True)
                    last_public["device"] = runtime.snapshot()
                    phase = "reference-physics"
                    accepted_reference = reference.step(H[case])
                    reference_accepted = bool(accepted_reference) and all(accepted_reference.values())
                    assert dict(accepted_reference) == dict.fromkeys(keys, True)
                    last_public["reference"] = reference.snapshot()
                    phase = "complete-comparison"
                    after, expected = runtime.snapshot(), reference.snapshot()
                    assert_exact(after, expected)
                    for i, key in enumerate(keys):
                        assert after[key].tick == tick + 1
                        assert abs(after[key].time - (tick + 1) * float(np.float32(H[case]))) <= 2e-6
                        loaded[i] += int(any(contact.normal_force > 0 for contact in after[key].last_step.contacts))
                    append(log, {"event": "step", "tick": tick + 1, "before": before, "task_frame": frame,
                        "task_outputs": task_values, "raw_policy_output": raw_action.detach().cpu().numpy(),
                        "native_action": host_values, "action_receipt": control_receipt, "device": after, "reference": expected,
                        "device_counters": runtime.counters, "reference_counters": reference.counters, "task_counters": workspace.counters})
                moved = [int(not np.array_equal(start[key].q, after[key].q) or not np.array_equal(start[key].v, after[key].v)) for key in keys]
                assert all(moved)
                if case == "support-slide":
                    assert all(loaded), "support fixture must carry actual device contact load"
                summary = {"status": "PASS_N1_INFERENCE_ONLY", "case": case, "steps": STEPS, "world_steps": count * STEPS,
                    "h": H[case], "producer_stream": stream_kind, "policy_device": "cuda:0",
                    "all_parameters_cuda": all(parameter.is_cuda for parameter in network.parameters()), "all_inputs_outputs_cuda": all_cuda,
                    "state_and_receipt_comparison": "exact-complete-same-FP32-input", "moved_worlds": moved,
                    "loaded_contact_steps_by_world": loaded, "device_counters": dict(runtime.counters),
                    "reference_counters": dict(reference.counters), "task_counters": dict(workspace.counters),
                    "diagnostic_action_readback_bytes": action_readback, "diagnostic_task_readback_bytes": task_readback,
                    "module_origins": module_origins(), "cross_device": "separate-hardware-conditional-negative-test"}
                assert summary["all_parameters_cuda"] and all_cuda
                phase = "owner-cleanup"
            append(log, {"event": "summary", **summary, "owners_closed": True})
            return summary
        except BaseException as error:
            evidence = {"event": "failure", "phase": phase, "type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
            evidence.update(last_public=last_public, action_accepted=action_accepted, state_accepted=state_accepted, reference_accepted=reference_accepted)
            for name, owner in (("device", runtime), ("reference", reference)):
                if owner is not None:
                    try:
                        evidence[name] = owner.snapshot()
                    except Exception as snapshot_error:
                        evidence[name + "_snapshot_error"] = repr(snapshot_error)
            append(log, evidence)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=tuple(MATRIX), default="hinge-slide")
    parser.add_argument("--worlds", type=int, default=8)
    parser.add_argument("--stream", choices=("default", "nondefault"), default="nondefault")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_case(args.case, args.worlds, args.stream, args.output)
    print(json.dumps({key: result[key] for key in ("status", "case", "steps", "world_steps")}, sort_keys=True))


if __name__ == "__main__":
    main()
