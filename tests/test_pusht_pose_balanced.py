"""Opt-in A policy plumbing and saved-audit isolation; no physical rollout here."""

import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import pytest
from test_manipulation_push_objective import propose

POLICY = "joint-pose-balanced"


@pytest.mark.parametrize("policy", [None, "joint-pose-descent", POLICY])
def test_pose_runner_cli_preserves_default_and_explicit_A_policy(monkeypatch, policy):
    from experiments.area_pressure import pusht_pose_replan as runner

    calls = []

    def fake_run(*args, **kwargs):
        calls.append(kwargs)
        return dict(cases={"A": dict(summary=dict(status="route_complete"))})

    monkeypatch.setattr(runner, "run", fake_run)
    argv = ["pose"]
    for flag in ("summary", "release", "source", "audit", "checkpoint", "output"):
        argv += ["--" + flag, flag + ".json"]
    argv += ["--case", "A"]
    if policy is not None:
        argv += ["--selection-policy", policy]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as exit_info:
        runner.main()
    assert exit_info.value.code == 0
    assert calls[0]["selection_policy"] == (policy or "joint-pose-descent")


@pytest.mark.parametrize("case", ["B", "both"])
def test_balanced_run_cannot_change_B_or_start_dual_case_work(case):
    from experiments.area_pressure.pusht_pose_replan import run

    with pytest.raises(ValueError, match="A"):
        run(*([Path("absent.json")] * 6), case=case, selection_policy=POLICY)


def test_single_case_audit_cli_retains_existing_case_validator(tmp_path, monkeypatch):
    from experiments.area_pressure import pusht_pose_replan_comparison as reader

    source, output = tmp_path / "A.json", tmp_path / "audit.json"
    seen = []

    def fake_case(path):
        seen.append(path)
        return dict(schema="test-only", selection_policy=POLICY)

    monkeypatch.setattr(reader, "compare_case", fake_case)
    monkeypatch.setattr(
        sys, "argv", ["audit", "--case-report", str(source), "--output", str(output)]
    )
    reader.main()
    assert seen == [source]
    assert json.loads(output.read_text("utf-8"))["selection_policy"] == POLICY


def test_saved_balanced_planner_is_bound_to_its_case_policy(tmp_path):
    from experiments.area_pressure import pusht_pose_replan_comparison as reader

    p = asdict(
        replace(
            propose()[0],
            selection_policy=POLICY,
            selection_eligible=True,
            selection_reason="joint_pose_balanced",
        )
    )
    state = dict(q=[0.0] * 9, v=[0.0] * 8)
    path = tmp_path / "plan.json"
    path.write_text(
        json.dumps(
            dict(
                tick=4450,
                q=state["q"],
                v=state["v"],
                route=dict(status="sampled_reference_ready", proposal=p),
            )
        ),
        encoding="utf-8",
    )
    payload = dict(
        route_file=path.name,
        route_sha256=reader._digest(path),
        tick=4450,
        source_q=state["q"],
        source_v=state["v"],
        status="sampled_reference_ready",
        selection_policy=POLICY,
    )
    result, saved = reader._planner(tmp_path, payload, state, 4450, selection_policy=POLICY)
    assert saved["selection_policy"] == POLICY
    assert result["heuristic_prediction"]["selection_policy"] == POLICY
    with pytest.raises(ValueError, match="policy"):
        reader._planner(tmp_path, payload, state, 4450, selection_policy="joint-pose-descent")


def test_balanced_case_cannot_be_merged_with_old_descent_case(tmp_path, monkeypatch):
    from test_pusht_pose_replan_comparison import _split_case_files

    reader, paths = _split_case_files(tmp_path)
    a = reader._read(paths["A"])
    a["selection_policy"] = POLICY
    paths["A"].write_text(json.dumps(a), encoding="utf-8")
    monkeypatch.setattr(reader, "compare_case", lambda path: dict(actual=dict(trajectory=[])))
    with pytest.raises(ValueError, match="policy"):
        reader.compare_cases(paths["A"], paths["B"])
