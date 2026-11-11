"""N4a fixed declarations before execution; expensive nodes are explicit learning/scale CUDA entries."""

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def api():
    path = ROOT / "examples" / "evaluate_mujoco_tyx_learning.py"
    spec = importlib.util.spec_from_file_location("n4a_public_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "name,seeds",
    (
        ("J", (14011, 14012, 14013)),
        ("S", (14111, 14112, 14113)),
        ("R", (14211, 14212, 14213)),
    ),
)
def test_fixed_contract_has_disjoint_grids_and_declared_budget(name, seeds, api):
    tasks = api.tasks()
    task = tasks.task(name)
    assert task.seeds == seeds
    assert tasks.UPDATES == 128 and tasks.ROLLOUT == 32
    assert tasks.LEARNING_WORLDS == 8 and tasks.HORIZON == 256
    assert tasks.SCALE_WORLDS == (1, 8, 32, 64, 128, 256)
    train = tasks.templates(name, held_out=False)
    held = tasks.templates(name, held_out=True)
    assert len(train) == len(held) == 8

    def identities(rows):
        return {tuple(q) + tuple(v) for q, v in rows}

    assert len(identities(train)) == len(identities(held)) == 8
    assert identities(train).isdisjoint(identities(held))
    for q, v in train + held:
        assert np.any(np.asarray(v) != 0)
        assert np.all(np.asarray(q) > task.q_lower) and np.all(np.asarray(q) < task.q_upper)
        assert np.all(np.abs(v) < task.v_bound)
    assert task.h == (0.002 if name == "J" else 0.001)
    assert task.action_scale == (20.0 if name == "S" else 2.0)


def test_support_reuses_original_source_and_target_not_rejected_proposal(api):
    original = ROOT / "examples" / "mujoco_tyx_cuda_policy.py"
    # XML binding is compared as authored text without importing that backend module.
    tree = ast.parse(original.read_text(encoding="utf-8-sig"))
    literal = next(
        node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "XML" for target in node.targets)
    )
    original_xml = ast.literal_eval(literal)["support-slide"]
    assert api.tasks().task("S").xml == original_xml
    task = api.tasks().task("S")
    assert task.q_target == (0.00075,) and task.h == 0.001
    assert 'fullinertia=".08 .10 .12 .002 -.001 .003"' in task.xml
    assert 'size=".24 .18 .05"' in task.xml and 'size=".12 .08 .06"' in task.xml
    assert 'iterations="128"' in task.xml and 'solimp=".8 .9 .002 .5 2"' in task.xml


def test_independent_native_cost():
    from phymeo.training.mujoco_tyx_evaluation import QuadraticCost, native_cost

    cost = QuadraticCost((0, 0), (0.2, 0.2), (1, 1), 0.01)
    states = {
        5: SimpleNamespace(q=np.array([0.2, 0]), v=np.array([1.0, 0])),
        41: SimpleNamespace(q=np.array([0.0, 0.2]), v=np.array([0.0, 2.0])),
    }
    np.testing.assert_allclose(
        native_cost(states, (5, 41), cost), [0.505, 0.52], atol=1e-15, rtol=0
    )
    states[41].q[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        native_cost(states, (5, 41), cost)


@pytest.mark.parametrize(
    "baseline,learned,passed",
    (
        ((1, 2, 3), (0.7, 1.5, 3.0), True),
        ((1, 2, 3), (0.7, 2.0, 3.0), False),
        ((1, 2, 3), (0.7, 1.5, 3.2), False),
        ((0, 2, 3), (0, 1, 1), False),
        ((1, 2, 3), (0.7, float("nan"), 1), False),
    ),
)
def test_three_seed_gate_never_selects_best_or_divides_zero(baseline, learned, passed):
    from phymeo.training.mujoco_tyx_evaluation import learning_outcome

    result = learning_outcome(baseline, learned)
    assert result["passed"] is passed
    assert len(result["baseline"]) == len(result["learned"]) == 3
    if baseline[0] == 0:
        assert result["classification"] == "BASELINE_DEGENERATE"
        assert result["ratios"] is None


@pytest.mark.parametrize("fault", ("terminated", "early-timeout", "missing-world", "reject-world"))
def test_complete_window_rejects_shortening(fault):
    from phymeo.training.mujoco_tyx_evaluation import check_window_transition

    transition = SimpleNamespace(
        accepted_worlds={5: True, 41: True},
        states={5: object(), 41: object()},
        terminated=np.array([False, False]),
        time_outs=np.array([False, False]),
    )
    if fault == "terminated":
        transition.terminated[1] = True
    elif fault == "early-timeout":
        transition.time_outs[1] = True
    elif fault == "missing-world":
        del transition.states[41]
    else:
        transition.accepted_worlds[41] = False
    with pytest.raises(ValueError):
        check_window_transition(transition, (5, 41), 17, 256)


def test_final_timeout_and_named_counter_accounting():
    from phymeo.training.mujoco_tyx_evaluation import check_window_transition, counter_delta

    transition = SimpleNamespace(
        accepted_worlds={5: True},
        states={5: object()},
        terminated=np.array([False]),
        time_outs=np.array([True]),
    )
    check_window_transition(transition, (5,), 256, 256)
    result = counter_delta(
        {"runtime": {"downloaded_bytes": 10}, "task": {"output_d2d_bytes": 4}},
        {"runtime": {"downloaded_bytes": 26}, "task": {"output_d2d_bytes": 12}},
    )
    assert result == {"runtime": {"downloaded_bytes": 16}, "task": {"output_d2d_bytes": 8}}
    # No sum is invented across overlapping ownership categories.
    assert "total_bytes" not in result


@pytest.mark.parametrize("name", ("J", "S", "R"))
@pytest.mark.cuda
@pytest.mark.tyx
def test_fixed_first_preflight(name, api, tmp_path):
    result = api.run_preflight(name, tmp_path / name)
    assert result["status"] == "PASS_N4A_PREFLIGHT"
    assert result["accepted_world_steps"] == 256
    assert result["steps"] == 32 and result["worlds"] == 8
    if name == "S":
        assert all(result["contact_seen"])


@pytest.mark.parametrize(
    "name,seed",
    (
        ("J", 14011),
        ("J", 14012),
        ("J", 14013),
        ("S", 14111),
        ("S", 14112),
        ("S", 14113),
        ("R", 14211),
        ("R", 14212),
        ("R", 14213),
    ),
)
@pytest.mark.cuda
@pytest.mark.tyx
def test_fixed_learning_seed(name, seed, api, tmp_path):
    result = api.run_learning(name, seed, tmp_path / (name + str(seed)))
    assert result["status"] == "COMPLETE_N4A_SEED"
    assert result["updates"] == 128 and result["accepted_world_steps"] == 32768
    assert result["baseline"]["world_steps"] == result["learned"]["world_steps"] == 2048
    assert result["parameter_hash_before"] != result["parameter_hash_after"]
    # The three-seed learning gate is evaluated by summarize_task, not by picking
    # an individual lucky seed; a completed weak seed remains visible evidence.


@pytest.mark.parametrize("name", ("J", "S", "R"))
@pytest.mark.parametrize("worlds", (1, 8, 32, 64, 128, 256))
@pytest.mark.cuda
@pytest.mark.tyx
def test_fixed_scale_cell(name, worlds, api, tmp_path):
    result = api.run_scale(
        name, api.tasks().task(name).seeds[0], worlds, tmp_path / (name + str(worlds))
    )
    assert result["status"] == "PASS_N4A_SCALE_CELL"
    assert result["measured_world_steps"] == 96 * worlds
    assert len(result["measured"]) == 3
    assert result["timing"]["compute_and_mandatory_diagnostics_s"] > 0
    assert result["timing"]["evidence_write_s"] >= 0
    assert result["coverage"]["unobserved_torch_internal_sync"] == "UNKNOWN"


def test_second_public_step_failure_retains_owned_prefix_and_restores_entry():
    from phymeo.training.mujoco_tyx_evaluation import capture_adapter_steps

    class Action:
        def __init__(self, value):
            self.value = np.array(value, np.float32)

        def detach(self):
            return self

        def clone(self):
            return Action(self.value.copy())

    class Adapter:
        def __init__(self):
            self.calls, self.last_transition = 0, None

        def step(self, action):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("injected second physical publication refusal")
            self.last_transition = {"sequence": self.calls, "value": action.value.copy()}
            return self.last_transition

    adapter, action = Adapter(), Action([[0.1]])
    original = adapter.step
    with pytest.raises(RuntimeError, match="second physical"):
        with capture_adapter_steps(adapter) as attempts:
            first = adapter.step(action)
            action.value[:] = 0.2
            adapter.step(action)
    action.value[:] = 0.9
    assert adapter.calls == 2 and len(attempts) == 2
    assert attempts[0]["transition"] is first
    assert attempts[0]["status"] == "accepted" and attempts[1]["status"] == "failed"
    np.testing.assert_array_equal(
        attempts[0]["requested_action"].value, np.array([[0.1]], np.float32)
    )
    np.testing.assert_array_equal(
        attempts[1]["requested_action"].value, np.array([[0.2]], np.float32)
    )
    assert "step" not in vars(adapter) and adapter.step == original
    adapter.step(action)
    assert adapter.calls == 3 and len(attempts) == 2


def test_lossless_evidence_preserves_array_bits_and_full_mapping(tmp_path):
    import gzip
    import json
    from base64 import b64decode
    from hashlib import sha256

    from phymeo.training.mujoco_tyx_evaluation import EvidenceWriter

    bits = np.array([0x80000000, 0x3F800000], np.uint32)
    path = tmp_path / "complete.jsonl.gz"
    writer = EvidenceWriter(path)
    writer.write("full", {5: bits.view(np.float32), "5": {"failure": "retained"}})
    writer.close()
    receipt = writer.receipt()
    raw = gzip.decompress(path.read_bytes())
    assert sha256(raw).hexdigest() == receipt["logical_sha256"]
    payload = json.loads(raw)["payload"]["mapping"]
    assert payload[0][0] == 5 and payload[1][0] == "5"
    assert b64decode(payload[0][1]["bytes_base64"]) == bits.tobytes()
    assert receipt["records"] == 1 and receipt["logical_bytes"] == len(raw)


@pytest.mark.parametrize("inputs", (("a", "a", "b"), ("a", "b")))
def test_summary_requires_exact_distinct_seed_inputs(api, tmp_path, inputs):
    with pytest.raises(ValueError, match="three distinct"):
        api.summarize_task("J", [tmp_path / name for name in inputs], tmp_path / "result")
    assert not (tmp_path / "result").exists()


def test_summary_rejects_changed_publication_before_cost_aggregation(api, tmp_path):
    import json

    directory = tmp_path / "saved"
    directory.mkdir()
    (directory / "summary.json").write_text(json.dumps({"task": "J"}), encoding="utf-8")
    (directory / "manifest.json").write_text("{}", encoding="utf-8")
    (directory / "publication.json").write_text(
        json.dumps({"summary_sha256": "0" * 64}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="publication hash"):
        api._read_summary(directory)


def _counter_fixture():
    return {
        "counts": {
            "transitions": 0,
            "physical_steps": 0,
            "resets": 0,
            "termination_readback_bytes": 0,
            "action_finite_scalar_readbacks": 0,
            "runtime": {
                "completed_steps": 0,
                "accepted_world_steps": 0,
                "failed_batches": 0,
                "device_control_submissions": 0,
                "device_control_refusals": 0,
                "control_copy_d2d_bytes": 0,
                "control_audit_downloaded_bytes": 0,
                "control_status_downloaded_bytes": 0,
                "control_initial_uploaded_bytes": 0,
                "control_pack_kernels": 0,
                "completed_kernels": 100,
                "synchronizations": 80,
                "device_allocations": 20,
                "static_uploaded_bytes": 200,
                "uploaded_bytes": 400,
                "downloaded_bytes": 800,
                "device_control_d2d_bytes": 128,
            },
            "task": {
                "task_kernels": 0,
                "source_synchronizations": 0,
                "task_status_downloaded_bytes": 0,
                "explicit_output_readback_bytes": 0,
                "output_d2d_bytes": 0,
                "published_frames": 0,
                "failed_refreshes": 0,
            },
        },
        "owners": {"environment": 1, "runtime": 2, "workspace": "task-owner"},
        "frame": {"revision": 60, "epoch": 30, "ticks": [30, 30], "times": [0.06, 0.06]},
    }


def test_resettable_device_counter_drop_is_signed_snapshot_not_work():
    from copy import deepcopy

    from phymeo.training.mujoco_tyx_evaluation import counter_accounting

    before = _counter_fixture()
    after = deepcopy(before)
    after["counts"].update(transitions=2, physical_steps=2, resets=2)
    after["counts"]["runtime"].update(
        completed_steps=2, accepted_world_steps=4, completed_kernels=0, downloaded_bytes=64
    )
    del after["counts"]["runtime"]["device_control_d2d_bytes"]
    after["counts"]["task"].update(task_kernels=3, published_frames=3)
    after["frame"] = {"revision": 65, "epoch": 33, "ticks": [0, 0], "times": [0.0, 0.0]}
    transitions = tuple(
        SimpleNamespace(
            sequence=index,
            world_keys=(5, 41),
            accepted_worlds={5: True, 41: True},
            states={key: SimpleNamespace(tick=30 + index) for key in (5, 41)},
            output_frame=SimpleNamespace(
                ticks=(31, 31) if index == 1 else (0, 0), revision=62 if index == 1 else 65
            ),
        )
        for index in (1, 2)
    )
    result = counter_accounting(before, after, transitions)
    assert result["monotonic_delta"]["environment"]["resets"] == 2
    assert result["monotonic_delta"]["runtime"]["accepted_world_steps"] == 4
    assert result["device_snapshot"]["signed_difference_not_work"]["completed_kernels"] == -100
    assert (
        result["device_snapshot"]["signed_difference_not_work"]["device_control_d2d_bytes"] is None
    )
    assert result["device_snapshot"]["cumulative_device_work"] == "UNKNOWN"
    assert result["receipt_events"]["reset_calls"] == 1
    assert result["receipt_events"]["reset_worlds"] == 2
    assert result["receipt_events"]["resets"][0]["world_keys"] == (5, 41)
    assert before["counts"]["runtime"]["completed_kernels"] == 100


def test_unknown_counter_field_cannot_be_silently_called_monotonic():
    from copy import deepcopy

    from phymeo.training.mujoco_tyx_evaluation import counter_accounting

    before = _counter_fixture()
    after = deepcopy(before)
    after["counts"]["runtime"]["new_unclassified_counter"] = 1
    with pytest.raises(ValueError, match="unclassified"):
        counter_accounting(before, after, ())


@pytest.mark.parametrize("fault", ("post-counter-read", "final-cuda-completion"))
def test_post_update_failure_retains_all_32_public_step_attempts(monkeypatch, fault):
    import sys

    from phymeo.training import mujoco_tyx_evaluation as module

    class Action:
        def __init__(self, value):
            self.value = value

        def detach(self):
            return self

        def clone(self):
            return Action(self.value)

    class Adapter:
        def __init__(self):
            self.calls, self.last_transition = 0, None

        def step(self, action):
            self.calls += 1
            self.last_transition = {"sequence": self.calls, "action": action.value}
            return self.last_transition

    adapter = Adapter()
    record = {"transitions": None, "loss": {"value": 0.0}, "iteration": 0, "next_iteration": 1}

    def update():
        record["transitions"] = tuple(adapter.step(Action(i)) for i in range(32))
        return record

    session = SimpleNamespace(env=SimpleNamespace(device="cuda:0"), adapter=adapter, update=update)
    reads, syncs = [], []

    def snapshot(_):
        reads.append(1)
        if len(reads) == 2 and fault == "post-counter-read":
            raise RuntimeError("injected post-update counter read")
        return _counter_fixture()

    def synchronize(_):
        syncs.append(1)
        if len(syncs) == 2 and fault == "final-cuda-completion":
            raise RuntimeError("injected final CUDA completion")

    cuda = SimpleNamespace(
        synchronize=synchronize,
        mem_get_info=lambda _: (1000, 2000),
        reset_peak_memory_stats=lambda _: None,
        max_memory_allocated=lambda _: 1,
        max_memory_reserved=lambda _: 2,
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    monkeypatch.setattr(module, "measurement_counter_snapshot", snapshot)
    with pytest.raises(module.UpdateCaptureError) as caught:
        module.measure_update(session)
    error = caught.value
    assert adapter.calls == 32 and len(error.attempts) == 32
    assert error.completed_record is record
    assert "step" not in vars(adapter)
    for i, entry in enumerate(error.attempts):
        assert entry["status"] == "accepted" and entry["requested_action"].value == i
        assert entry["transition"] is record["transitions"][i]


def test_post_update_finite_audit_writes_complete_record_before_refusal():
    from phymeo.training.mujoco_tyx_evaluation import validate_update

    record = {
        "iteration": 0,
        "next_iteration": 1,
        "transitions": tuple(range(32)),
        "optimization": {"optimizer_step": 1, "finite_gradient": False},
        "loss": {"value": 0.0},
    }
    written = []
    writer = SimpleNamespace(write=lambda kind, payload: written.append((kind, payload)))
    with pytest.raises(ValueError, match="nonfinite optimization"):
        validate_update(record, None, expected_iteration=0, cost=None, writer=writer)
    assert written == [("update", record)]


def test_failure_context_captures_public_counters_before_owner_close(api, monkeypatch):
    from contextlib import contextmanager

    closed, reads = [], []
    env = SimpleNamespace(last_transition={"sequence": 32})

    @contextmanager
    def make_environment(*args, **kwargs):
        try:
            yield env
        finally:
            closed.append(True)

    def snapshot(value):
        assert value is env and not closed
        reads.append(True)
        return _counter_fixture()

    monkeypatch.setattr(api.tasks(), "make_environment", make_environment)
    monkeypatch.setattr(api.evaluation, "measurement_counter_snapshot", snapshot)
    context = {}
    with pytest.raises(RuntimeError, match="late measurement"):
        with api._observed_environment(context, "J"):
            raise RuntimeError("late measurement")
    assert closed == [True] and reads == [True]
    assert context["env"] is None
    assert context["owner_exit"]["last_transition"] is env.last_transition
    assert (
        context["owner_exit"]["counter_snapshot"]["counts"]["runtime"]["completed_kernels"] == 100
    )


def test_scale_delivery_uses_full_wall_region_not_sum_of_subtimers(api, tmp_path, monkeypatch):
    events = []
    clock = iter((110.0, 111.0, 114.0, 115.0, 117.0))
    monkeypatch.setattr(api.time, "perf_counter", lambda: next(clock))

    class Writer:
        write_seconds = 5.0

        def write(self, kind, payload):
            events.append(kind)

        def close(self):
            events.append("closed")

        def receipt(self):
            assert events[-1] == "closed"
            events.append("hash")
            return {"scope": "fake file IO for timing boundary only"}

    context = {
        "writer": Writer(),
        "path": tmp_path,
        "scale_write_start": 1.0,
        "scale_wall_start": 100.0,
    }
    result = {
        "status": "PASS_N4A_SCALE_CELL",
        "task": "J",
        "measured_world_steps": 768,
        "timing": {
            "compute_and_mandatory_diagnostics_s": 2.0,
            "post_update_independent_audit_s": 1.0,
        },
    }
    api._finish(context, result)
    assert result["timing"]["measured_loop_to_trace_close_s"] == 10.0
    assert result["timing"]["transitions_per_s_with_full_trace_delivery"] == 76.8
    assert result["timing"]["evidence_write_s"] == 4.0
    assert result["final_evidence_hash_io_s"] == 3.0
    assert events == ["completion", "closed", "hash"]
