"""All-world publication is a consumer contract, not a new physical solver."""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from phymeo.backends.tyx_world_batch import BatchStepRejected, SpherePlaneWorld, TyxWorldBatch


def cache(tmp_path):
    return Path(os.environ.get("PHYMEO_RESIDENT_CACHE", tmp_path))


@pytest.mark.parametrize("mode", ["interpret", "cuda"])
@pytest.mark.parametrize("count", [1, 8, 64])
def test_all_world_rejection_keeps_accepted_state_and_support_impulse(
    mode, count, tmp_path, monkeypatch
):
    sphere = SpherePlaneWorld(mass=1, radius=0.125, position=(0, 0, 0.125), gravity=(0, 0, -8))
    with TyxWorldBatch(
        [(w, sphere) for w in range(count)], mode, cache_root=cache(tmp_path)
    ) as batch:
        assert all(batch.step(1 / 8, require_all=True).values())
        batch.set_controls({count - 1: (3e38, 0, 0)})
        before = batch.snapshot()
        stats = batch.stats()
        assert stats["publication_contract_version"] == 1
        with monkeypatch.context() as patch:
            patch.setattr(batch, "snapshot", lambda *a, **k: pytest.fail("step downloaded state"))
            with pytest.raises(BatchStepRejected) as error:
                batch.step(2, require_all=True)
        assert error.value.rejected_worlds == (count - 1,)
        after_stats = batch.stats()
        assert after_stats["state_epoch"] == stats["state_epoch"]
        assert after_stats["completed_steps"] == stats["completed_steps"]
        assert after_stats["aborted_batch_steps"] == stats["aborted_batch_steps"] + 1
        assert after_stats["failed_steps"] == stats["failed_steps"] + 1
        assert after_stats["device_allocations"] == stats["device_allocations"]
        assert after_stats["snapshot_downloaded_bytes"] == stats["snapshot_downloaded_bytes"]
        assert after_stats["synchronizations"] == stats["synchronizations"] + 1
        assert after_stats["recovery_sync_attempts"] == stats["recovery_sync_attempts"]
        print(
            "RESIDENT_PUBLICATION "
            + json.dumps(
                dict(
                    mode=mode,
                    worlds=count,
                    deltas={
                        key: after_stats[key] - stats[key]
                        for key in (
                            "device_allocations",
                            "snapshot_downloaded_bytes",
                            "status_downloaded_bytes",
                            "synchronizations",
                            "recovery_sync_attempts",
                            "state_epoch",
                            "completed_steps",
                            "aborted_batch_steps",
                        )
                    },
                )
            )
        )
        if mode == "cuda":
            assert (
                after_stats["status_downloaded_bytes"] - stats["status_downloaded_bytes"]
                == 4 * count
            )
        assert batch.snapshot() == before
        batch.set_controls({count - 1: (1, 0, 0)})
        assert all(batch.step(1 / 8, require_all=True).values())
        accepted = batch.snapshot()
        assert accepted[count - 1].velocity[0] == 0.125
        assert accepted[count - 1].position[0] == 1 / 64
        assert all(s.tick == 2 for s in accepted.values())
        assert all(s.normal_impulse_n_s == 1 for s in accepted.values())
        batch.reset([count - 1])
        reset = batch.snapshot()
        assert reset[count - 1].tick == 0 and reset[count - 1].normal_impulse_n_s == 0
        assert all(reset[w] == accepted[w] for w in range(count - 1))


@pytest.mark.parametrize("mode", ["interpret", "cuda"])
def test_default_partial_acceptance_is_unchanged(mode, tmp_path):
    sphere = SpherePlaneWorld(mass=1, radius=0.125)
    with TyxWorldBatch([(3, sphere), (7, sphere)], mode, cache_root=cache(tmp_path)) as batch:
        batch.set_controls({7: (3e38, 0, 0)})
        assert batch.step(2) == {3: True, 7: False}
        states = batch.snapshot()
        assert states[3].tick == 1 and states[7].tick == 0
        assert batch.stats()["aborted_batch_steps"] == 0


@pytest.mark.parametrize("value", [0, 1, None, "all", np.bool_(True)])
def test_all_world_policy_is_explicit_even_for_empty_batch(value):
    with TyxWorldBatch([], "interpret") as batch:
        with pytest.raises(TypeError, match="require_all"):
            batch.step(0.01, require_all=value)
        assert batch.step(0.01, require_all=True) == {}


@pytest.mark.parametrize("mode", ["interpret", "cuda"])
def test_tree_contact_receipt_and_delayed_history_publish_together(mode, tmp_path):
    from test_resident_tree_contacts import H, _action, _json, _law, _r01_bound, _slider

    from phymeo.backends.tyx_world_batch import FreeSphere

    program, tree = _slider(kp=8, kd=2, delay=1)
    fragile = FreeSphere(mass=1e-30, radius=0.125, gravity=(0, 0, 0))
    with TyxWorldBatch([(17, tree), (88, fragile)], mode, cache_root=cache(tmp_path)) as batch:
        model, bindings, _, pair = _r01_bound(batch)
        with batch.prepare_contacts(model, bindings, {pair: _law()}) as plan:
            batch.set_controls({17: _action(program, target=[0.5])})
            assert all(batch.step_contacts(plan, H, require_all=True).values())
            batch.set_controls({88: (3e38, 0, 0)})
            before = _json(batch.snapshot())
            receipt = plan.last_accepted
            epoch = batch.stats()["state_epoch"]
            with pytest.raises(BatchStepRejected) as error:
                batch.step_contacts(plan, H, require_all=True)
            assert error.value.rejected_worlds == (88,)
            assert batch.stats()["state_epoch"] == epoch
            assert _json(batch.snapshot()) == before
            assert plan.last_accepted is receipt
            batch.set_controls({88: (0, 0, 0)})
            assert all(batch.step_contacts(plan, H, require_all=True).values())
            assert plan.last_accepted is not receipt
            current = _json(batch.snapshot())
            assert current["17"]["tick"] == current["88"]["tick"] == 2
            batch.reset([17])
            restored = _json(batch.snapshot())
            assert restored["17"]["tick"] == 0
            assert restored["88"] == current["88"]
