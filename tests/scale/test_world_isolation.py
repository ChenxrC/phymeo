"""Co-located worlds retain local IDs, contacts, material witnesses and independent motion."""

from contextlib import ExitStack

import pytest
from cases import SCALES, PhymeoGroup, world_ids


@pytest.mark.parametrize("count", SCALES)
def test_single_calls_and_reordered_combined_model(count, tmp_path):
    ids = world_ids(count)
    held = ids[len(ids) // 2]
    with ExitStack() as stack:

        def group(worlds, shuffle):
            result = PhymeoGroup(worlds, "host", tmp_path, shuffle=shuffle)
            stack.callback(result.close)
            return result

        ordered, shuffled = group(ids, False), group(ids, True)
        single = [group((w,), True) for w in ids]
        reference = None
        for moved in (None, held, None):
            answers = []
            for api in (ordered, shuffled, *single):
                api.select(moved)
                answers.append(api.inspect(api.query(), moved))
            a, b, *per_world = answers
            assert a == b
            assert a["contacts"] == [row for one in per_world for row in one["contacts"]]
            assert a["candidate_pairs"] == sum(s["candidate_pairs"] for s in per_world)
            assert (
                a["same_world_rows"]
                == sum(s["published_pair_rows"] for s in per_world)
                == 6 * count
            )
            assert a["published_pair_rows"] == (4 * count) * (4 * count - 1) // 2
            if reference is None:
                reference = a["contacts"]
            if moved is None:
                assert a["contacts"] == reference
            else:
                assert a["contacts"] == [row for row in reference if row["world"] != held]
                assert shuffled.near_control(held)[0]["gap_m"] == pytest.approx(1 / 16)


@pytest.mark.tyx
def test_native_world_filters_have_real_completed_work(tyx_mode, tmp_path):
    ids = world_ids(2)
    api = PhymeoGroup(ids, tyx_mode, tmp_path / "cache")
    try:
        for moved in (None, ids[0], None):
            api.select(moved)
            observed = api.inspect(api.query(), moved)
            assert observed["cross_world_filtered_rows"] == 16
        stats = api.stats()
        assert stats["backend"] == tyx_mode and stats["completed_launches"] >= 6
        if tyx_mode == "cuda":
            assert stats["synchronizations"] > 0 and stats["downloaded_bytes"] > 0
    finally:
        api.close()


def test_above_existing_complete_model_limit_is_explicit(tmp_path):
    with pytest.raises(ValueError, match="shape count"):
        PhymeoGroup(tuple(range(65)), "host", tmp_path)
