"""End-to-end paired observation benchmark; no learning-quality assertion."""

import importlib.util
from pathlib import Path

import pytest


def api():
    path = (
        Path(__file__).resolve().parents[2] / "examples/compare_mujoco_tyx_observation_training.py"
    )
    spec = importlib.util.spec_from_file_location("paired_training_observation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.cuda
@pytest.mark.parametrize("worlds", (8, 128, 256))
@pytest.mark.parametrize("name", ("J", "S", "R"))
def test_complete_paired_ppo(name, worlds, tmp_path):
    result = api().run_pair(name, worlds, tmp_path / "result")
    assert result["status"] == "PASS_PAIRED_PPO_OBSERVATION"
    assert result["physics_and_learner_identical"]
    assert result["measured_pairs"] == 2
    for mode in ("full", "integration"):
        row = result["modes"][mode]
        assert row["measured_world_steps"] == 32 * worlds
        assert len(row["signatures"]) == 3
        assert row["signatures"][0]["learner"] != row["initial_policy_hash"]
        assert row["counter_end"]["physical_steps"] == 48
        assert row["counter_end"]["transitions"] == 48
        assert row["loop_to_trace_close_s"] > row["compute_and_mandatory_diagnostics_s"] > 0


def task_binding(owner="owner"):
    return {
        "schema": "mujoco-tyx-task-workspace-v1",
        "workspace_owner": owner,
        "runtime_owner": id(owner),
        "spec_sha256": "task",
        "source_sha256": "source",
        "archive_sha256": "archive",
        "model_sha256": "model",
        "world_keys": (5, 41),
        "mode": "cuda",
        "device": 0,
        "native_layout": (2, 2),
        "output_layout": (2, 4),
        "precision": "FP32-device-native-features",
        "actions": "existing-HOST-control-and-receipt-path",
        "physics_diagnostics": "existing-complete-HOST-validation",
    }


def environment_binding(mode="full", owner="owner"):
    return {
        "schema": "mujoco-tyx-vector-env-v1",
        "world_keys": (5, 41),
        "task": "task",
        "action": "action",
        "task_source": task_binding(owner),
        "h": 0.002,
        "horizons": (64, 72),
        "decimation": 1,
        "runtime_ownership": "borrowed-exclusive-mutation",
        "physics_diagnostics": "complete-HOST-validation-retained",
        "state_observation": mode,
        "episode_bookkeeping": "HOST-lengths-and-done-mask",
        "restore": "explicit-N3-update-boundary-composition",
    }


@pytest.mark.parametrize(
    "field", ("source_sha256", "spec_sha256", "native_layout", "world_keys", "future_contract")
)
def test_stable_bindings_exclude_only_declared_identity_and_output(field):
    import copy

    module = api()
    left = environment_binding("full", "first")
    right = environment_binding("integration", "second")
    assert module.stable_environment_binding(left) == module.stable_environment_binding(right)
    changed = copy.deepcopy(right)
    changed["task_source"][field] = "different"
    assert module.stable_environment_binding(left) != module.stable_environment_binding(changed)
    del changed["task_source"]["model_sha256"]
    with pytest.raises(ValueError, match="binding"):
        module.stable_environment_binding(changed)


def test_summary_only_refuses_retained_owner():
    module = api()
    with pytest.raises(TypeError):
        module.summary_only({"owner": object()})


def test_single_mode_releases_learner_before_next_factory(monkeypatch, tmp_path):
    import weakref
    from contextlib import contextmanager
    from types import SimpleNamespace

    module = api()
    owners_seen = []

    class Writer:
        def __init__(self, path):
            self.write_seconds = 0.0

        def write(self, *args):
            pass

        def close(self):
            pass

        def receipt(self):
            return {"records": 3}

    class Env:
        def __init__(
            self,
            runtime=None,
            task_spec=None,
            action_spec=None,
            *,
            h=0.002,
            horizons=(64, 72),
            observation_mode="full",
        ):
            self.runtime = runtime or SimpleNamespace(capabilities={"backend": "fixture"})
            self.task_spec, self.action_spec = task_spec, action_spec
            self.h, self.horizons = h, horizons
            self.binding = environment_binding(observation_mode)
            self.device = "cuda:0"
            self.closed = False
            self.counters = {"physical_steps": 48, "transitions": 48}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

        def close(self):
            self.closed = True

        def get_observations(self):
            return [0.0, 1.0]

    class Session:
        def __init__(self, env, *, steps):
            self.env, self.next_iteration, self.config = env, 0, {"steps": steps}
            self.algorithm = SimpleNamespace(
                **{
                    name: SimpleNamespace(state_dict=lambda: {"value": 1.0})
                    for name in ("actor", "critic", "optimizer")
                }
            )
            owners_seen.append(weakref.ref(self))

    @contextmanager
    def make_environment(*args, **kwargs):
        assert all(ref() is None for ref in owners_seen)
        with Env() as env:
            yield env

    tasks = SimpleNamespace(
        make_environment=make_environment,
        seed_all=lambda seed: None,
        task=lambda name: SimpleNamespace(seeds=(14011,), cost=lambda: None),
    )
    torch = SimpleNamespace(
        get_rng_state=lambda: [1], cuda=SimpleNamespace(get_rng_state=lambda device: [2])
    )

    def measure(session, **kwargs):
        index = session.next_iteration
        session.next_iteration += 1
        return {"transitions": tuple(range(16)), "iteration": index}, {
            "compute_and_mandatory_diagnostics_s": 0.001
        }

    monkeypatch.setattr(module.evidence, "EvidenceWriter", Writer)
    monkeypatch.setattr(module.evidence, "measure_update", measure)
    monkeypatch.setattr(
        module.evidence, "validate_update", lambda *args, **kwargs: {"accepted": 16}
    )
    monkeypatch.setattr(module, "common_update", lambda record: record)
    monkeypatch.setattr(module, "_training_factories", lambda: (Env, lambda env: env, Session))
    for mode in ("full", "integration"):
        result = module._run_mode("J", 8, mode, tmp_path, tasks, torch, 14011)
        assert all(ref() is None for ref in owners_seen)
        assert set(result) == {"summary", "comparable"}


@pytest.mark.parametrize("fails", (False, True))
def test_memory_copy_facade_delegates_and_restores_without_native_owners(fails, tmp_path):
    import copy
    import json
    import weakref
    from types import SimpleNamespace

    module = api()

    class Data:
        def __init__(self):
            self.narena, self.nbuffer, self.maxuse_arena = 4096, 512, 128
            self.model = SimpleNamespace(narena=4096, nbuffer=256)

    native = Data()
    original_calls = []

    def native_copy(value):
        original_calls.append(weakref.ref(value))
        if fails:
            raise MemoryError("injected native allocation")
        return copy.copy(value)

    original_module = SimpleNamespace(copy=native_copy)
    target = SimpleNamespace(copy=original_module)
    observer = module.MemoryObservation(
        tmp_path, 2, process_sample=lambda: {"private_commit_bytes": 1, "rss_bytes": 2}
    )
    with observer.install(target, Data):
        observer.begin_mode("integration")
        observer.begin_step()
        if fails:
            with pytest.raises(MemoryError, match="injected native"):
                target.copy.copy(native)
        else:
            result = target.copy.copy(native)
            assert result is not native
            assert result.model is native.model
            del result
        observer.end_step(failed=fails)
        observer.event("mode-close-complete")
    assert target.copy is original_module
    assert len(original_calls) == 1
    ref = weakref.ref(native)
    del native
    assert ref() is None
    snapshot = observer.snapshot()
    assert snapshot["copy_calls"] == 1
    assert snapshot["copy_failures"] == int(fails)
    assert (
        json.loads(json.dumps(snapshot))["process_metric_scope"]
        == "whole-process-not-exclusive-native"
    )
    assert any(row.get("native", {}).get("data_narena_bytes") == 4096 for row in snapshot["events"])
    if fails:
        assert any(
            row["stage"] == "copy-failed" and row["copy_ordinal"] == 1 for row in snapshot["events"]
        )


@pytest.mark.parametrize("name,worlds", (("J", 256), ("S", 128)))
def test_memory_diagnostic_only_failed_s256(name, worlds, tmp_path):
    with pytest.raises(ValueError, match="S256"):
        api().run_pair(name, worlds, tmp_path / "refused", memory_diagnostic=True)
    assert not (tmp_path / "refused").exists()


def test_common_transition_hash_includes_stable_frame_binding():
    from types import SimpleNamespace

    module = api()
    frame = SimpleNamespace(
        binding=task_binding(), revision=1, epoch=2, ticks=(1, 1), times=(0.002, 0.002)
    )
    values = dict.fromkeys(module.TRANSITION_FIELDS)
    values.update(states={}, input_frame=frame, output_frame=frame)
    transition = SimpleNamespace(**values)
    record = {
        key: None
        for key in (
            "iteration",
            "next_iteration",
            "loss",
            "optimization",
            "recurrent_masks",
            "step_requests",
        )
    }
    record["transitions"] = (transition,)
    first = module.digest(module.common_update(record))
    frame.binding = {**frame.binding, "workspace_owner": "new-opaque", "runtime_owner": 999}
    assert module.digest(module.common_update(record)) == first
    frame.binding = {**frame.binding, "source_sha256": "wrong-model-source"}
    assert module.digest(module.common_update(record)) != first


def native_fixture():
    from types import SimpleNamespace

    class Data:
        def __init__(self, model, *, marker=None):
            if model.fail:
                raise MemoryError("original public constructor failed")
            self.model, self.marker = model, marker
            self.narena, self.nbuffer = model.narena, 512
            self.maxuse_arena, self.maxuse_stack = 128, 32
            self.maxuse_threadstack, self.maxuse_con, self.maxuse_efc = (16, 32), 2, 6

    model = SimpleNamespace(narena=4096, nbuffer=256, fail=False)
    return Data, model


@pytest.mark.parametrize("fails", (False, True))
def test_memory_public_constructor_preserves_real_type_failure_and_restoration(fails, tmp_path):
    import copy
    import weakref
    from types import SimpleNamespace

    module = api()
    data_type, model = native_fixture()
    native_module = SimpleNamespace(MjData=data_type)
    original_copy = SimpleNamespace(copy=copy.copy)
    target = SimpleNamespace(copy=original_copy)
    observer = module.MemoryObservation(tmp_path, 2, process_sample=lambda: {})
    model.fail = fails
    with observer.install(target, data_type, native_module=native_module):
        observer.begin_mode("full")
        if fails:
            with pytest.raises(MemoryError, match="original public constructor"):
                native_module.MjData(model, marker="unchanged")
        else:
            data = native_module.MjData(model, marker="unchanged")
            assert type(data) is data_type
            assert data.model is model and data.marker == "unchanged"
            reference = weakref.ref(data)
            del data
            assert reference() is None
        observer.finish_construction()
    assert native_module.MjData is data_type
    assert target.copy is original_copy
    saved = observer.snapshot()
    assert saved["schema"] == "bounded-native-allocation-observation-v2"
    assert saved["constructor_calls"] == 1
    assert saved["constructor_failures"] == int(fails)
    assert saved["copy_calls"] == 0
    assert saved["phase_accounting"]["full"]["construct"]["constructor_calls"] == 1
    assert saved["performance_eligible"] is False
    if not fails:
        assert saved["native_ranges"]["data_maxuse_stack_bytes"] == [32, 32]
        assert saved["native_ranges"]["data_maxuse_threadstack_bytes/1"] == [32, 32]


def test_memory_physics_interval_excludes_full_observer_and_retains_no_runtime(tmp_path):
    import copy
    import weakref
    from types import SimpleNamespace

    module = api()
    data_type, model = native_fixture()
    native_module = SimpleNamespace(MjData=data_type)
    target = SimpleNamespace(copy=SimpleNamespace(copy=copy.copy))

    class Runtime:
        def __init__(self):
            self._data = {7: SimpleNamespace(), 91: SimpleNamespace()}
            self._observer_data = None

        def step(self, h):
            assert h == 0.002
            return {7: True, 91: True}

        def snapshot(self):
            if self._observer_data is None:
                self._observer_data = native_module.MjData(model)
            self._observer_data.maxuse_arena = 192
            return {"marker": "actual-observer-result"}

        def integration_snapshot(self):
            return {"marker": "integration-result"}

    runtime = Runtime()

    class Adapter:
        def __init__(self, owner):
            self.runtime = owner

        def step(self, actions):
            accepted = self.runtime.step(0.002)
            full = self.runtime.snapshot()
            return accepted, full, actions

    adapter = Adapter(runtime)
    observer = module.MemoryObservation(tmp_path, 2, process_sample=lambda: {})
    with observer.install(target, data_type, native_module=native_module):
        observer.begin_mode("full")
        initial = native_module.MjData(model)
        del initial
        with observer.observe_runtime(runtime, data_type), observer.observe_steps(adapter):
            observer.finish_construction()
            for _ in range(2):
                result = adapter.step((1.0, -2.0))
                assert result[1] == {"marker": "actual-observer-result"}
            runtime.integration_snapshot()
        assert "step" not in vars(runtime) and "snapshot" not in vars(runtime)
        assert "integration_snapshot" not in vars(runtime) and "step" not in vars(adapter)
    saved = observer.snapshot()
    ledger = saved["phase_accounting"]["full"]
    assert ledger["construct"]["constructor_calls"] == 1
    assert ledger["full-observer"]["constructor_calls"] == 1
    assert ledger["full-observer"]["calls"] == 2
    assert ledger["physics-step"]["calls"] == 2
    assert ledger["physics-step"]["constructor_calls"] == 0
    assert ledger["physics-step"]["copy_calls"] == 0
    assert len(saved["physics_steps"]) == 2
    assert all(row["zero_native_allocation"] for row in saved["physics_steps"])
    assert saved["native_ranges"]["data_maxuse_arena_bytes"] == [128, 192]
    runtime_ref, scratch_ref = weakref.ref(runtime), weakref.ref(runtime._observer_data)
    del runtime, adapter
    assert runtime_ref() is None and scratch_ref() is None


@pytest.mark.parametrize("operation", ("construct", "copy", "throw"))
def test_memory_step_allocation_gate_and_original_failure_restore_wrappers(operation, tmp_path):
    import copy
    from types import SimpleNamespace

    module = api()
    data_type, model = native_fixture()
    native_module = SimpleNamespace(MjData=data_type)
    target = SimpleNamespace(copy=SimpleNamespace(copy=copy.copy))
    seed = data_type(model)

    class Runtime:
        def step(self, h):
            if operation == "construct":
                native_module.MjData(model)
            elif operation == "copy":
                target.copy.copy(seed)
            else:
                raise LookupError("original step error")

        def snapshot(self):
            return {}

        def integration_snapshot(self):
            return {}

    runtime = Runtime()
    observer = module.MemoryObservation(tmp_path, 1, process_sample=lambda: {})
    expected = LookupError if operation == "throw" else AssertionError
    match = "original step error" if operation == "throw" else "native allocation"
    with pytest.raises(expected, match=match):
        with observer.install(target, data_type, native_module=native_module):
            observer.begin_mode("integration")
            observer.finish_construction()
            with observer.observe_runtime(runtime, data_type):
                observer.begin_step()
                runtime.step(0.002)
    assert native_module.MjData is data_type
    assert "step" not in vars(runtime)
    row = observer.snapshot()["physics_steps"][0]
    assert row["original_call_failed"] is (operation == "throw")
    assert row["constructor_calls"] == int(operation == "construct")
    assert row["copy_calls"] == int(operation == "copy")


def test_memory_constructor_failure_unwinds_both_public_proxies(tmp_path):
    import copy
    from types import SimpleNamespace

    module = api()
    data_type, model = native_fixture()
    model.fail = True
    native_module = SimpleNamespace(MjData=data_type)
    original_copy = SimpleNamespace(copy=copy.copy)
    target = SimpleNamespace(copy=original_copy)
    observer = module.MemoryObservation(tmp_path, 1, process_sample=lambda: {})
    with pytest.raises(MemoryError, match="original public constructor"):
        with observer.install(target, data_type, native_module=native_module):
            observer.begin_mode("integration")
            native_module.MjData(model)
    observer.finish_construction(failed=True)
    assert native_module.MjData is data_type and target.copy is original_copy
    snapshot = observer.snapshot()
    assert snapshot["constructor_failures"] == 1
    assert snapshot["phase_accounting"]["integration"]["construct"]["failures"] == 1


def test_memory_layout_records_only_owned_reference_counts(tmp_path):
    import weakref
    from types import SimpleNamespace

    module = api()
    data_type, model = native_fixture()
    compact_type = type("NativeMirror", (), {"__module__": "phymeo.backends._mujoco_tyx_mirror"})
    compact, scratch = compact_type(), data_type(model)
    runtime = SimpleNamespace(_data={7: compact}, _observer_data=scratch)
    observer = module.MemoryObservation(tmp_path, 1, process_sample=lambda: {})
    observer.begin_mode("full")
    row = observer.runtime_layout(runtime, data_type)
    assert row["compact_worlds"] == 1 and row["accepted_native_records"] == 0
    assert row["observer_scratch_present"] == 1 and row["observer_is_native"]
    compact_ref, scratch_ref = weakref.ref(compact), weakref.ref(scratch)
    del compact, scratch, runtime
    assert compact_ref() is None and scratch_ref() is None
    observer.finish_construction()
    assert observer.snapshot()["runtime_layouts"][0] == row
