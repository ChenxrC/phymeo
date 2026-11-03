"""A mixed physical population exercises two device solvers in one publication."""

import json
from dataclasses import replace

import numpy as np
import pytest
from test_resident_step import slider
from test_world_contact_loads import fixture

import phymeo as ph


def population():
    driven, drive_state = slider(delay=1)
    sheet, clear = fixture(), fixture(lift=0.5)
    worlds = [
        (19, ph.ClothLinkWorld(*sheet[:2])),
        (73, ph.FreeSphere(2.0, 0.125, position=(0, 0, 2), gravity=(0, 0, -8))),
        (4, ph.ClothLinkWorld(*clear[:2])),
        (31, ph.PrismaticSlider(driven, drive_state)),
    ]
    return worlds, {19: sheet[2], 4: clear[2]}, driven


@pytest.mark.tyx
def test_local_rejection_and_cross_bucket_state_on_one_real_clock(tyx_mode, tmp_path):
    worlds, loads, drive = population()
    h = 1 / 256
    with ph.TyxBatchedScene(worlds, tyx_mode, cache_root=tmp_path) as group:
        group.set_controls({31: drive.inputs(np.array([1.0]))})
        before = group.stats()
        first = group.step(h, loads)
        after = group.stats()
        assert tuple(first.accepted) == (19, 73, 4, 31) and all(first.accepted.values())
        assert first.contact_results[19].audit.sample.normal_force_n == pytest.approx(2)
        assert first.contact_results[4].audit.sample.normal_force_n == 0
        assert (
            after["resident"]["snapshot_downloaded_bytes"]
            == before["resident"]["snapshot_downloaded_bytes"]
        )
        assert after["resident"]["submitted_kernels"] - before["resident"]["submitted_kernels"] == 1
        assert after["material"]["completed_kernels"] > 0 and after["completed_steps"] == 1
        state = group.snapshot()
        assert all(s.tick == 1 for s in state.values())
        assert state[73].position == (0.0, 0.0, 2 - 8 * h * h)
        assert state[73].velocity == (0.0, 0.0, -8 * h)
        assert state[31].applied_effort == (0.0,) and state[31].history[0] == 1
        assert state[19].qd == pytest.approx(1 / 64) and state[4].qd == 0
        np.testing.assert_allclose(np.array(state[19].velocity)[:, 2], [-1 / 384] * 3, atol=2e-9)
        blocked = group.step(h, loads, site_capacity={19: 1})
        assert blocked.accepted == {19: False, 73: True, 4: True, 31: True}
        assert blocked.contact_results[19].required_sites == 2
        second = group.snapshot()
        assert second[19] == state[19] and second[31].applied_effort == (4.0,)
        assert second[31].velocity == (2 * h,) and second[73].position[2] == 2 - 24 * h * h
        group.restore({73: state[73], 19: state[19]})
        assert group.snapshot([4, 19, 73, 31]) == {
            4: second[4],
            19: state[19],
            73: state[73],
            31: second[31],
        }
        replay = group.step(h, loads)
        assert (
            replay.contact_results[19].audit.sample.normal_force_n > 0
            and group.snapshot([73])[73] == second[73]
        )
        group.reset([19, 31])
        reset = group.snapshot()
        assert reset[19].tick == reset[31].tick == 0 and reset[4].tick == 3 and reset[73].tick == 2
        with pytest.raises(TypeError):
            first.accepted[19] = False
        (tmp_path / "mixed-physical-step.json").write_text(
            json.dumps(
                dict(
                    mode=tyx_mode,
                    first_force_n=first.contact_results[19].audit.sample.normal_force_n,
                    cloth_velocity_m_s=state[19].velocity,
                    link_velocity_m_s=state[19].qd,
                    accepted=dict(blocked.accepted),
                    ticks_after_reset={k: v.tick for k, v in reset.items()},
                    launches=[
                        after["resident"]["submitted_kernels"],
                        after["material"]["completed_kernels"],
                    ],
                    implicit_full_state_readback_bytes=after["resident"][
                        "snapshot_downloaded_bytes"
                    ]
                    - before["resident"]["snapshot_downloaded_bytes"],
                )
            ),
            encoding="utf8",
        )


@pytest.mark.tyx
def test_late_device_failure_discards_contact_result_and_retries_at_same_input(
    tyx_mode, tmp_path, monkeypatch
):
    from phymeo.backends.tyx_world_batch import TyxWorldBatch

    worlds, loads, _ = population()
    with ph.TyxBatchedScene(worlds, tyx_mode, cache_root=tmp_path) as group:
        group.step(1 / 256, loads)
        initial = group.snapshot()
        accepted = group.step(1 / 256, loads)
        expected = group.snapshot()
        group.restore(initial)

        def fail_status(batch):
            raise RuntimeError("injected late resident status download")

        with monkeypatch.context() as patch:
            patch.setattr(TyxWorldBatch, "_read_status", fail_status)
            with pytest.raises(RuntimeError, match="late resident status"):
                group.step(1 / 256, loads)
        assert group.snapshot() == initial and group.stats()["failed_steps"] == 1
        actual = group.step(1 / 256, loads)
        assert actual == accepted and group.snapshot() == expected
        # A foreign checkpoint cannot partially replace a valid cloth checkpoint.
        with pytest.raises(ValueError):
            group.restore({19: initial[19], 73: replace(initial[73], world=31)})
        assert group.snapshot() == expected
        # A rejected input on an early bucket cannot advance the later bucket.
        with pytest.raises(ValueError):
            group.step(1 / 256, {19: loads[19]})
        assert group.snapshot() == expected
        with pytest.raises(KeyError):
            group.reset([999])
        assert group.snapshot() == expected


def test_mixed_registration_and_empty_scene():
    ball = ph.FreeSphere(1, 0.125)
    with pytest.raises(ValueError, match="duplicate"):
        ph.TyxBatchedScene([(0, ball), (0, ball)], "interpret")
    with pytest.raises(TypeError):
        ph.TyxBatchedScene([(3, object())], "interpret")
    with ph.TyxBatchedScene([], "interpret") as group:
        assert group.step(1 / 64).accepted == {} and group.snapshot() == {}
    with pytest.raises(RuntimeError, match="closed"):
        group.step(1 / 64)
