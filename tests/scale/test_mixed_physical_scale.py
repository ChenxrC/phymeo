"""Independent physical-window checks for the production mixed-world scene."""

import json
from contextlib import ExitStack
from dataclasses import asdict

import numpy as np
import pytest
from mixed_physical_cases import WINDOW, H, HostReferences, MixedRun, population, signature


@pytest.mark.parametrize("count", [4, 16])
@pytest.mark.parametrize("mode", ["interpret", "cuda"])
def test_group_matches_all_original_laws_and_complete_separate_reports(count, mode, tmp_path):
    case = population(count)
    with ExitStack() as stack:
        scene = MixedRun(case, True, mode, tmp_path / "cache", stack)
        singles = MixedRun(case, False, mode, tmp_path / "cache", stack)
        host = HostReferences(case, stack)
        frames = []
        for index in range(WINDOW):
            observed = scene.advance(index)
            assert observed == singles.advance(index)
            host.advance_and_check(index, observed)
            frames.append(observed)
        k = scene.keys[0]
        start = dict(frames[0].material)[k].audit
        np.testing.assert_allclose(
            start.sample.node_forces_n, np.tile([0, 0, -2 / 3], (3, 1)), atol=2e-7
        )
        np.testing.assert_allclose(start.sample.link_force_n, [0, 0, 2], atol=2e-7)
        assert start.sample.gap_m == 3 / 32 and start.sample.compression_m == 1 / 32
        assert abs(start.kinetic_change_j - 7 / 98304) < 2e-10
        assert all(len(s.law_row) == 13 for s in start.sample.sites)
        finish = dict(frames[-1].material)[k].audit
        assert 0 < finish.sample.normal_force_n < start.sample.normal_force_n
        if count == 16:
            end = dict(frames[-1].checkpoints)
            assert (
                len({end[key].velocity[2] for key, role in case.roles.items() if role == "ball"})
                == 4
            )
        stats = scene.counts()
        assert (
            stats["resident_submitted_kernels"] == WINDOW
            and stats["material_completed_kernels"] == 6 * WINDOW
        )
        receipt = dict(
            mode=mode,
            worlds=count,
            h_s=H,
            frames=WINDOW,
            sha256=signature(frames),
            original_host_max_differences=host.max_diffs,
            max_work_residual_j=host.max_work_residual,
            max_momentum_residual_ns=host.max_momentum,
            first_audit=asdict(start),
            last_audit=asdict(finish),
            site_counts=[
                {k: len(v.audit.sample.sites) for k, v in frame.material} for frame in frames
            ],
            group_workspace_counters={k: v for k, v in stats.items() if not k.startswith("cuda_")},
        )
        (tmp_path / "mixed-scale-qualification.json").write_text(
            json.dumps(receipt, indent=2), encoding="utf8"
        )


@pytest.mark.parametrize("mode", ["interpret", "cuda"])
def test_capacity_rejection_preserves_only_that_world_and_selective_reset_replays(mode, tmp_path):
    case = population(16)
    with ExitStack() as stack:
        scene = MixedRun(case, True, mode, tmp_path / "cache", stack)
        singles = MixedRun(case, False, mode, tmp_path / "cache", stack)
        hit, _, _, rail = scene.keys[:4]
        first = scene.advance(0)
        assert first == singles.advance(0)
        before = dict(first.checkpoints)
        blocked = scene.advance(1, {hit: 1})
        assert blocked == singles.advance(1, {hit: 1})
        flags = dict(blocked.flags)
        assert not flags[hit] and sum(flags.values()) == 15
        state = dict(blocked.checkpoints)
        assert state[hit] == before[hit] and all(state[k].tick == 2 for k in flags if k != hit)
        fail = dict(blocked.material)[hit]
        assert (
            fail.reason == "site_capacity"
            and fail.required_sites == 2
            and fail.audit == before[hit].last_audit
        )
        for runner in (scene, singles):
            runner.reset([hit, rail])
        rows = scene.snapshot()
        assert rows == singles.snapshot() and rows[hit].tick == rows[rail].tick == 0
        assert rows[rail].history == (0.0, 0.0, 0.0) and rows[hit].last_audit is None
        assert all(rows[k] == state[k] for k in rows if k not in (hit, rail))
        again = scene.advance(0, {hit: 2})
        assert again == singles.advance(0, {hit: 2}) and all(dict(again.flags).values())
        current = dict(again.checkpoints)
        assert current[hit] == before[hit] and current[rail] == before[rail]
        assert all(current[k].tick == 3 for k in current if k not in (hit, rail))
