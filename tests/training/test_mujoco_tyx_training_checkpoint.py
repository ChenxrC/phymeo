"""N3 exact continuation; original N2 models and actual upstream CUDA PPO."""
import base64
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

pytestmark = [pytest.mark.cuda, pytest.mark.tyx]
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def api():
    spec = importlib.util.spec_from_file_location(
        "n3_restore_example", ROOT / "examples" / "restore_mujoco_tyx_ppo.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.n2().require_cuda()
    return module


@pytest.mark.parametrize("case", ("hinge-slide", "support-slide", "gru"))
def test_actual_update_boundary_continuation(case, api, tmp_path):
    result = api.compare_case(case, tmp_path)
    assert result["status"] == "PASS_N3_EXACT_CONTINUATION"
    assert result["next_iteration"] == (2 if case == "gru" else 3)
    assert result["compared_world_steps"] == (64 if case == "gru" else 128)
    assert result["physics_and_training_exact"]
    assert result["old_frame_refused"]
    assert result["destination_revision_increased"]
    if case == "gru":
        assert result["saved_nonzero_actor_hidden"] and result["saved_nonzero_critic_hidden"]
        assert result["partial_done_steps"] >= 2


@pytest.fixture(scope="module")
def prepared(api, tmp_path_factory):
    with api.make_session("hinge-slide") as session:
        session.update()
        session.update()
        path = tmp_path_factory.mktemp("n3-base") / "bundle"
        session.save_checkpoint(path)
        yield session, path


def _copy_bundle(source, target):
    target.mkdir()
    for name in ("manifest.json", "physics.bin", "training.json"):
        (target / name).write_bytes((source / name).read_bytes())


def _rewrite_training(path, body):
    raw = json.dumps(body, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()
    (path / "training.json").write_bytes(raw)
    manifest = json.loads((path / "manifest.json").read_text())
    manifest["files"]["training.json"] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _first_array(value):
    if isinstance(value, dict):
        if "$array" in value:
            return value
        for child in value.values():
            found = _first_array(child)
            if found is not None:
                return found
    if isinstance(value, list):
        for child in value:
            found = _first_array(child)
            if found is not None:
                return found
    return None


@pytest.mark.parametrize("fault", (
    "hash", "truncate", "model", "world", "task", "action", "h", "config",
    "policy-size", "missing-rng", "missing-rnn", "missing-generator", "filename",
))
def test_bad_bundle_is_rejected_before_physical_cas(fault, api, prepared, tmp_path):
    session, original = prepared
    bundle = tmp_path / "bad"
    _copy_bundle(original, bundle)
    if fault == "hash":
        value = bytearray((bundle / "training.json").read_bytes())
        value[-1] ^= 1
        (bundle / "training.json").write_bytes(value)
    elif fault == "truncate":
        value = (bundle / "physics.bin").read_bytes()
        (bundle / "physics.bin").write_bytes(value[:-1])
    elif fault in {"model", "world", "task", "action", "h", "config", "filename"}:
        manifest = json.loads((bundle / "manifest.json").read_text())
        binding = manifest["binding"]
        if fault == "model":
            binding["physics"]["model_sha256"] = "0" * 64
        elif fault == "world":
            binding["environment"]["world_keys"].reverse()
        elif fault in {"task", "action"}:
            binding["environment"][fault] = "0" * 64
        elif fault == "h":
            binding["environment"]["h"] *= 2
        elif fault == "config":
            binding["config"]["actor"]["hidden_dims"] = [17, 17]
        else:
            manifest["files"]["../training.json"] = manifest["files"].pop("training.json")
        (bundle / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    else:
        body = json.loads((bundle / "training.json").read_text())
        if fault == "policy-size":
            array = _first_array(body["learner"]["actor_state_dict"])
            assert array is not None and array["dtype"] == "<f4"
            array["shape"] = [1]
            array["data"] = base64.b64encode(b"\0" * 4).decode()
        elif fault == "missing-rng":
            del body["rng"]
        elif fault == "missing-rnn":
            del body["hidden"]
        else:
            del body["rng"]["named"]["torch-cuda"]
        _rewrite_training(bundle, body)
    before = session.training_state()
    physical = session.env.runtime.checkpoint().to_bytes()
    revision = session.env.runtime.checkpoint_revision
    learner = session.algorithm
    with pytest.raises((ValueError, RuntimeError, KeyError)):
        session.restore_checkpoint(bundle, expected_revision=revision)
    assert session.algorithm is learner
    assert session.env.runtime.checkpoint_revision == revision
    assert session.env.runtime.checkpoint().to_bytes() == physical
    api.exact(before, session.training_state())
    assert not session.needs_recovery


def test_real_other_model_and_stale_cas_refused(api, prepared):
    session, bundle = prepared
    with api.make_session("support-slide") as other:
        before = other.env.runtime.checkpoint().to_bytes()
        with pytest.raises(ValueError, match="binding"):
            other.restore_checkpoint(bundle, expected_revision=other.env.runtime.checkpoint_revision)
        assert other.env.runtime.checkpoint().to_bytes() == before
    revision = session.env.runtime.checkpoint_revision
    before = session.training_state()
    with pytest.raises(ValueError, match="revision|CAS|stale"):
        session.restore_checkpoint(bundle, expected_revision=revision - 1)
    api.exact(before, session.training_state())


def test_candidate_construction_rng_effects_are_rolled_back(api, prepared, monkeypatch):
    session, bundle = prepared
    torch = api.n2().require_cuda()
    import numpy as np
    import random

    before = session.training_state()
    physical = session.env.runtime.checkpoint().to_bytes()
    algorithm = session.algorithm
    def fail(observation):
        torch.rand(9, device=session.env.device)
        torch.rand(7)
        np.random.random(5)
        random.random()
        raise RuntimeError("injected candidate load failure after RNG use")
    monkeypatch.setattr(session, "_construct_algorithm", fail)
    with pytest.raises(RuntimeError, match="candidate load"):
        session.restore_checkpoint(bundle, expected_revision=session.env.runtime.checkpoint_revision)
    assert session.algorithm is algorithm
    assert session.env.runtime.checkpoint().to_bytes() == physical
    api.exact(before, session.training_state())


def test_existing_target_not_overwritten(prepared):
    session, path = prepared
    before = {name: (path / name).read_bytes()
              for name in ("manifest.json", "physics.bin", "training.json")}
    with pytest.raises(FileExistsError):
        session.save_checkpoint(path)
    assert before == {name: (path / name).read_bytes() for name in before}


def test_nonempty_pending_and_mid_update_boundaries(api, tmp_path, monkeypatch):
    torch = api.n2().require_cuda()
    with api.make_session("hinge-slide") as session:
        with pytest.raises(RuntimeError, match="completed update"):
            session.save_checkpoint(tmp_path / "initial")
        original = session.algorithm.update
        seen = []
        def update():
            with pytest.raises(RuntimeError, match="boundary|progress"):
                session.save_checkpoint(tmp_path / "during")
            seen.append(True)
            return original()
        monkeypatch.setattr(session.algorithm, "update", update)
        session.update()
        assert seen
        with torch.inference_mode():
            obs = session.adapter.get_observations()
            action = session.algorithm.act(obs)
            with pytest.raises(RuntimeError, match="Transition|pending"):
                session.save_checkpoint(tmp_path / "pending")
            obs, rewards, dones, extras = session.adapter.step(action)
            session.algorithm.process_env_step(obs, rewards, dones, extras)
        assert session.algorithm.storage.step == 1
        with pytest.raises(RuntimeError, match="storage|rollout"):
            session.save_checkpoint(tmp_path / "partial")


@pytest.mark.parametrize("fault", ("task-refresh", "rng-install"))
def test_post_cas_failure_is_honest_and_poisoned(fault, api, tmp_path, monkeypatch):
    from phymeo.training.mujoco_tyx_checkpoint import MuJoCoTyxTrainingRestoreError

    with api.make_session("hinge-slide") as session:
        session.update()
        saved_states = session.env.runtime.snapshot()
        path = tmp_path / "bundle"
        session.save_checkpoint(path)
        session.update()
        revision = session.env.runtime.checkpoint_revision
        if fault == "task-refresh":
            def fail(frame):
                raise RuntimeError("injected post-CAS task export failure")
            monkeypatch.setattr(session.env.workspace, "to_torch", fail)
        else:
            def fail(state):
                raise RuntimeError("injected post-CAS RNG installation failure")
            monkeypatch.setattr(session, "_install_rng", fail)
        with pytest.raises(MuJoCoTyxTrainingRestoreError) as caught:
            session.restore_checkpoint(path, expected_revision=revision)
        assert caught.value.physics_restored
        assert session.env.runtime.checkpoint_revision > revision
        api.n2().load_n1().assert_exact(saved_states, session.env.runtime.snapshot())
        assert session.needs_recovery and session.adapter.needs_recovery
        assert session.env.needs_recovery
        with pytest.raises(RuntimeError, match="recovery|failed"):
            session.update()
