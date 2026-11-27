"""Compact pickup handoff binding and replay dispatch."""
from argparse import Namespace
from importlib import import_module
import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def pickup(monkeypatch):
    pytest.importorskip("mujoco")
    examples = Path(__file__).resolve().parents[1] / "examples"
    monkeypatch.syspath_prepend(str(examples))
    runner = import_module("panda_pickup_from_reach")
    return runner, examples / "checkpoints/panda-pickup-from-reach-s4000"


def test_published_banks_bind_every_ordered_row(pickup, tmp_path):
    runner, published = pickup
    scene = runner.PUBLISHED / "scene.xml"
    model = runner.mj.MjModel.from_xml_path(str(scene))
    results = json.loads((published / "results.json").read_text(encoding="utf8"))
    assert results["student_sha256"] == runner.sha(published / "student.npz") == runner.STUDENT_SHA256
    for split, field in (("factor-update", "train_bank_sha256"), ("validation", "eval_bank_sha256")):
        path = published / (split + ".npz")
        assert results[field] == runner.sha(path) == runner.HANDOFF_SHA256[split]
        rows = runner.read_handoffs(path, model, expected_split=split)
        with np.load(path, allow_pickle=False) as bank:
            assert len(rows) == 12
            np.testing.assert_array_equal(np.stack([r.qpos for r in rows]), bank["qpos"])
    validation = published / "validation.npz"
    with pytest.raises(ValueError, match="split"):
        runner.read_handoffs(validation, model, expected_split="factor-update")
    with np.load(validation, allow_pickle=False) as original:
        bank = dict(original)
    # Carry all self-reported provenance forward while permuting complete worlds.
    for field in ("qpos", "qvel", "bias", "target"):
        bank[field] = bank[field][::-1]
    forged = tmp_path / "validation.npz"
    np.savez_compressed(forged, **bank)
    with pytest.raises(ValueError, match="checksum"):
        runner.read_handoffs(forged, model, expected_split="validation")


def test_replay_only_dispatches_validation_and_two_controls(pickup, tmp_path, monkeypatch):
    runner, published = pickup
    args = Namespace(scene=runner.PUBLISHED / "scene.xml", out=tmp_path / "run", handoff_dir=published,
                     student=published / "student.npz", training_trace=None, validation_trace=None)
    monkeypatch.setattr(runner, "train_student", lambda *a: pytest.fail("replay invoked training"))
    calls = []

    def fake_rollout(model, start, *, student=None, reference=None, negative=None, save_trace=False):
        calls.append((start.qpos.copy(), student is not None, negative))
        assert save_trace
        assert (reference is None) == (student is not None)
        return dict(score=dict(success=True), commands=np.zeros((runner.DECISIONS, 16)), trace={})

    monkeypatch.setattr(runner, "rollout", fake_rollout)
    monkeypatch.setattr(runner.PickupLoads, "compare", lambda *a, **kw: dict(success=True, gates={}))
    runner.run(args)
    with np.load(published / "validation.npz", allow_pickle=False) as bank:
        np.testing.assert_array_equal(np.stack([q for q, _, _ in calls]), np.repeat(bank["qpos"], 3, axis=0))
    assert [(deployed, branch) for _, deployed, branch in calls] == [
        (True, None), (False, "open_jaw"), (False, "one_sided")] * 12
    result = json.loads((args.out / "results.json").read_text(encoding="utf8"))
    assert result["student_sha256"] == runner.STUDENT_SHA256
    assert result["teacher"] == [] and result["training_candidates"] == []
    assert not (args.out / "training.json").exists()


def test_replay_rejects_substituted_student(pickup, tmp_path):
    runner, published = pickup
    student = tmp_path / "other.npz"
    student.write_bytes((published / "student.npz").read_bytes() + b"modified")
    args = Namespace(scene=runner.PUBLISHED / "scene.xml", out=tmp_path / "run", handoff_dir=published,
                     student=student, training_trace=None, validation_trace=None)
    with pytest.raises(ValueError, match="student checkpoint checksum"):
        runner.run(args)


def test_compact_training_uses_original_training_rows(pickup, tmp_path, monkeypatch):
    runner, published = pickup
    args = Namespace(scene=runner.PUBLISHED / "scene.xml", out=tmp_path / "run", handoff_dir=published,
                     student=None, training_trace=None, validation_trace=None)

    class TrainingReached(Exception):
        pass

    def inspect_training_input(model, starts, out, report):
        with np.load(published / "factor-update.npz", allow_pickle=False) as bank:
            for field in ("qpos", "qvel", "target", "bias"):
                np.testing.assert_array_equal(np.stack([getattr(s, field) for s in starts]), bank[field])
        assert report["train_bank_sha256"] == runner.HANDOFF_SHA256["factor-update"]
        raise TrainingReached

    monkeypatch.setattr(runner, "train_student", inspect_training_input)
    with pytest.raises(TrainingReached):
        runner.run(args)
