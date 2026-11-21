"""Accepted progress stops are successful termination, never planner-error aliases."""

import sys

import pytest

from experiments.area_pressure import pusht_public_feedback as feedback


@pytest.mark.parametrize("reason", ("success", "no_progress", "segment_budget", "step_budget"))
def test_accepted_progress_terminal_keeps_original_reason(reason):
    assert (
        feedback._accepted_controller_stop(
            stopped=True, reason=reason, monitor_status=reason, accepted_steps=1
        )
        == reason
    )


def test_accepted_running_controller_continues():
    assert (
        feedback._accepted_controller_stop(
            stopped=False, reason="continue", monitor_status="continue", accepted_steps=1
        )
        is None
    )


@pytest.mark.parametrize(
    "stopped,reason,monitor_status,accepted_steps",
    (
        (True, "unsupported_tilt", "continue", 1),
        (True, "no_sampled_route", "continue", 1),
        (True, "success", "continue", 1),
        (False, "continue", "success", 1),
        (True, "success", "success", 0),
        (True, "success", "success", True),
    ),
)
def test_no_success_alias_for_path_errors_inconsistent_or_unaccepted_state(
    stopped, reason, monitor_status, accepted_steps
):
    with pytest.raises(ValueError):
        feedback._accepted_controller_stop(
            stopped=stopped,
            reason=reason,
            monitor_status=monitor_status,
            accepted_steps=accepted_steps,
        )


@pytest.mark.parametrize("status", ("complete", "controller_stopped", "partial_failure"))
def test_cli_distinguishes_clean_terminal_from_fault(monkeypatch, status):
    monkeypatch.setattr(
        sys,
        "argv",
        ["feedback", "--trajectory", "archive", "--source", "source.json", "--output", "new.json"],
    )
    monkeypatch.setattr(feedback, "run_window", lambda *args, **kwargs: dict(status=status))
    if status == "partial_failure":
        with pytest.raises(SystemExit) as caught:
            feedback.main()
        assert caught.value.code == 1
    else:
        feedback.main()
