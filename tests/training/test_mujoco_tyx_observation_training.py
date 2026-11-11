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
        "workspace_owner": owner, "runtime_owner": id(owner),
        "spec_sha256": "task", "source_sha256": "source",
        "archive_sha256": "archive", "model_sha256": "model",
        "world_keys": (5, 41), "mode": "cuda", "device": 0,
        "native_layout": (2, 2), "output_layout": (2, 4),
        "precision": "FP32-device-native-features",
        "actions": "existing-HOST-control-and-receipt-path",
        "physics_diagnostics": "existing-complete-HOST-validation",
    }


def environment_binding(mode="full", owner="owner"):
    return {
        "schema": "mujoco-tyx-vector-env-v1", "world_keys": (5, 41),
        "task": "task", "action": "action", "task_source": task_binding(owner),
        "h": .002, "horizons": (64, 72), "decimation": 1,
        "runtime_ownership": "borrowed-exclusive-mutation",
        "physics_diagnostics": "complete-HOST-validation-retained",
        "state_observation": mode, "episode_bookkeeping": "HOST-lengths-and-done-mask",
        "restore": "explicit-N3-update-boundary-composition",
    }


@pytest.mark.parametrize("field", ("source_sha256", "spec_sha256", "native_layout", "world_keys", "future_contract"))
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
        def __init__(self, runtime=None, task_spec=None, action_spec=None, *, h=.002,
                     horizons=(64, 72), observation_mode="full"):
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
            self.algorithm = SimpleNamespace(**{
                name: SimpleNamespace(state_dict=lambda: {"value": 1.0})
                for name in ("actor", "critic", "optimizer")
            })
            owners_seen.append(weakref.ref(self))

    @contextmanager
    def make_environment(*args, **kwargs):
        assert all(ref() is None for ref in owners_seen)
        with Env() as env:
            yield env

    tasks = SimpleNamespace(
        make_environment=make_environment, seed_all=lambda seed: None,
        task=lambda name: SimpleNamespace(seeds=(14011,), cost=lambda: None),
    )
    torch = SimpleNamespace(get_rng_state=lambda: [1], cuda=SimpleNamespace(get_rng_state=lambda device: [2]))

    def measure(session, **kwargs):
        index = session.next_iteration
        session.next_iteration += 1
        return {"transitions": tuple(range(16)), "iteration": index}, {"compute_and_mandatory_diagnostics_s": .001}

    monkeypatch.setattr(module.evidence, "EvidenceWriter", Writer)
    monkeypatch.setattr(module.evidence, "measure_update", measure)
    monkeypatch.setattr(module.evidence, "validate_update", lambda *args, **kwargs: {"accepted": 16})
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
    observer = module.MemoryObservation(tmp_path, 2, process_sample=lambda: {"private_commit_bytes": 1, "rss_bytes": 2})
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
    assert json.loads(json.dumps(snapshot))["process_metric_scope"] == "whole-process-not-exclusive-native"
    assert any(row.get("native", {}).get("data_narena_bytes") == 4096 for row in snapshot["events"])
    if fails:
        assert any(row["stage"] == "copy-failed" and row["copy_ordinal"] == 1 for row in snapshot["events"])


@pytest.mark.parametrize("name,worlds", (("J", 256), ("S", 128)))
def test_memory_diagnostic_only_failed_s256(name, worlds, tmp_path):
    with pytest.raises(ValueError, match="S256"):
        api().run_pair(name, worlds, tmp_path / "refused", memory_diagnostic=True)
    assert not (tmp_path / "refused").exists()
