import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from profile_multiface_stage_cost import WallTimeline  # noqa: E402


def test_one_wall_clock_excludes_parked_world_and_overlapping_caller_wait():
    # Main waits for A, performs device work while both worlds are parked, then resumes A.
    profile = WallTimeline()
    profile.main = 0
    profile.records = [
        (0, 100, 11, "world_stack"),
        (10, 90, 11, "parked"),
        (15, 85, 22, "world_stack"),
        (20, 80, 22, "parked"),
        (0, 20, 0, "await_world"),
        (30, 70, 0, "law_wrapper"),
        (40, 60, 0, "kernel_submit"),
        (80, 100, 0, "await_world"),
    ]
    result = profile.partition(0, 100)
    assert result["exclusive_ns"] == {
        "world_stack": 30,
        "handoff_idle": 10,
        "scheduler_or_other_caller": 20,
        "law_wrapper": 20,
        "kernel_submit": 20,
    }
    assert result["inclusive_wait_ns"] == {
        "caller_await_inclusive": 40,
        "parked_world_stack_sum": 140,
    }
    assert sum(result["exclusive_ns"].values()) == 100


def test_real_worker_handoff_overlap_is_reported_once():
    profile = WallTimeline()
    profile.main = 0
    profile.records = [
        (0, 8, 111, "world_stack"),
        (5, 15, 222, "world_stack"),
        (0, 15, 0, "await_world"),
    ]
    assert profile.partition(0, 15)["exclusive_ns"] == {"world_stack": 12, "handoff_overlap": 3}
