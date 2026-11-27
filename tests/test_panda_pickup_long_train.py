from importlib import import_module
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def trainer(monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("mujoco")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    return import_module("panda_pickup_long_train")


def test_resume_after_latest_commit_reproduces_next_update_and_scores(trainer, monkeypatch, tmp_path):
    """No dynamics calls: exercise real CPU sampling, optimizer/RNG persistence and committed projections."""
    torch = trainer.torch
    def domain(xml, handoffs, split, group, worlds, *, batch):
        return SimpleNamespace(manifest=dict(resets_sha256=f"{split}/{group}/{batch}/{worlds}", xml_sha256="unit"),
                               worlds=worlds, split=split, group=group, batch=batch)
    calls = []
    def rollout(args, d, actor, mode):
        calls.append((d.split, d.batch, d.group, mode))
        n = args.decisions * d.worlds
        stats = dict(mean_return=1., physics_world_steps=n * args.repeat)
        if mode != "train":
            return {}, [dict(success=False, longest_loaded_dwell_s=0.) for _ in range(d.worlds)], stats
        with torch.no_grad():
            obs = torch.randn(args.decisions, d.worlds, 43)
            _, raw, logp, val = actor.sample(obs)
            reward = torch.randn(args.decisions, d.worlds)
        return dict(obs=obs, raw=raw, logp=logp, value=val, reward=reward), [], stats
    monkeypatch.setattr(trainer, "make_batch", domain)
    monkeypatch.setattr(trainer, "_rollout", rollout)

    def args(name):
        return trainer.arguments(["--out", str(tmp_path / name), "--backend", "native", "--updates", "2",
                                  "--decisions", "2", "--worlds", "2", "--repeat", "1", "--eval-worlds", "2",
                                  "--eval-every", "1", "--epochs", "1", "--minibatch", "3"])

    reference = args("reference")
    trainer.run(reference)
    ref = trainer._load(reference.out / "latest.pt")
    interrupt = args("interrupted")
    publish = trainer._json
    def crash(path, data):
        if path.name == "update-00001.json":
            raise OSError("interrupted after durable checkpoint")
        publish(path, data)
    monkeypatch.setattr(trainer, "_json", crash)
    with pytest.raises(OSError, match="interrupted after"):
        trainer.run(interrupt)
    assert trainer._load(interrupt.out / "latest.pt")["next_update"] == 2
    monkeypatch.setattr(trainer, "_json", publish)
    calls.clear()
    final = trainer.run(interrupt)
    restored = trainer._load(interrupt.out / "latest.pt")
    assert trainer._actor_hash(restored["actor"]) == trainer._actor_hash(ref["actor"])
    assert torch.equal(restored["rng"]["torch_cpu"], ref["rng"]["torch_cpu"])
    assert restored["completed_train_world_steps"] == 8
    assert restored["next_train_batch"] == 1002
    assert [(r[0], r[1]) for r in calls if r[3] == "train"] == [("train", 1001)]
    assert sum(r[0] == "select" and r[3] == "prior" for r in calls) == 2
    assert sum(r[0] == "blind" and r[3] == "candidate" for r in calls) == 2
    assert final["select"]["update"] == 1  # equal scores, earliest frozen candidate
    assert final["total_world_steps"] == 48 and not final["blind_improved"]
    assert len(list(interrupt.out.glob("update-*.json"))) == 2
    calls.clear()
    assert trainer.run(interrupt) == final and not calls
    interrupt.seed += 1
    with pytest.raises(ValueError, match="immutable result"):
        trainer.run(interrupt)


def test_writer_excludes_colliding_driver_and_reopens(trainer, tmp_path):
    with trainer._writer(tmp_path):
        with pytest.raises(RuntimeError, match="writer lock"):
            with trainer._writer(tmp_path):
                pytest.fail("second trainer acquired same directory")
    with trainer._writer(tmp_path):
        trainer._json(tmp_path / "test.json", {"complete": True})
    assert json.loads((tmp_path / "test.json").read_text(encoding="utf8"))["complete"]


def test_conditional_ppo_indexes_exact_rollout_history(trainer):
    torch = trainer.torch
    student = trainer.HERE / "checkpoints/panda-pickup-from-reach-s4000/student.npz"
    actor = trainer.ResidualPickupActorCritic.from_student(student, "cpu", arm_std=.01, finger_std=.10, arm_rho=.8)
    torch.manual_seed(9384)
    obs = torch.randn(3, 2, 43)
    prev = torch.zeros(2, 16)
    rows = {k: [] for k in ("raw", "prev_raw", "has_prev", "logp", "value")}
    with torch.no_grad():
        for i in range(len(obs)):
            valid = torch.full((2,), i > 0)
            _, raw, logp, value = actor.sample(obs[i], prev_raw=prev, has_prev=valid)
            for k, x in (("raw", raw), ("prev_raw", prev), ("has_prev", valid), ("logp", logp), ("value", value)):
                rows[k].append(x)
            prev = raw
    records = dict(obs=obs, reward=torch.randn(3, 2), **{k: torch.stack(v) for k, v in rows.items()})
    # A frozen learning rate isolates the old/new conditional-density replay under shuffled minibatches.
    opt = torch.optim.Adam(actor.parameters(), lr=0.)
    metrics = trainer.update_policy(torch, actor, opt, records, epochs=1, minibatch=2)
    assert metrics["approx_kl"] == pytest.approx(0., abs=1e-6)
    assert metrics["training_clip_fraction"] == 0.
    records["prev_raw"][1] = 0.
    with pytest.raises(ValueError, match="conditional history"):
        trainer.update_policy(torch, actor, opt, records)
