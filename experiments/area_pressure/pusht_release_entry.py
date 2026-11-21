"""Resume the RR controller at a published route boundary using its saved physical state."""

import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from shutil import copyfile

import mujoco
import numpy as np

from phymeo.backends.mujoco_runtime import MuJoCoInitialState
from phymeo.training.manipulation_tasks import (
    SuccessFrame,
    SuccessHistory,
    evaluate_goal,
    manipulation_scenes,
)
from scripts.run_manipulation_control import plain
from scripts.run_manipulation_push import SOURCE_SHA, tuple_values
from scripts.run_manipulation_push_feedback import PushFeedbackDriver, hold_plan

from .pusht_entry_rows import _digest


def _require(ok, reason):
    if not ok:
        raise ValueError("release entry: " + reason)


def public_release_initial(model, state):
    """Find a unit FP64 declaration in the saved quaternion's FP32 rounding cells."""
    q = np.array(state["q"], dtype=np.float64)
    v = np.asarray(state["v"], dtype=np.float64)
    _require(
        q.shape == (model.nq,)
        and v.shape == (model.nv,)
        and np.isfinite(q).all()
        and np.isfinite(v).all(),
        "invalid saved state",
    )
    _require(
        np.array_equal(q.astype(np.float32), q) and np.array_equal(v.astype(np.float32), v),
        "state must carry published FP32 values",
    )
    start = int(model.jnt_qposadr[model.joint("object-free").id]) + 3
    rotation = q[start : start + 4].copy()
    norm = np.linalg.norm(rotation)
    _require(abs(norm - 1) < 1e-5, "invalid saved rotation")
    candidate = rotation / norm
    target = rotation.astype(np.float32)
    if candidate.astype(np.float32).tobytes() != target.tobytes():
        # Bounds just inside each cell avoid guessing how a tie rounds. Along
        # this diagonal the squared norm increases monotonically.
        mag = abs(target)
        lo = np.maximum(0.0, (mag.astype(float) + np.nextafter(mag, np.float32(-np.inf))) / 2)
        hi = (mag.astype(float) + np.nextafter(mag, np.float32(np.inf))) / 2
        lo = np.nextafter(lo, mag.astype(float))
        hi = np.nextafter(hi, mag.astype(float))
        _require(lo @ lo <= 1 <= hi @ hi, "unit rotation cannot reproduce saved FP32 bytes")
        for _ in range(56):
            mid = (lo + hi) / 2
            if mid @ mid < 1:
                lo = mid
            else:
                hi = mid
        candidate = np.copysign((lo + hi) / 2, rotation)
    _require(
        candidate.astype(np.float32).tobytes() == target.tobytes()
        and abs(candidate @ candidate - 1) <= 64 * np.finfo(float).eps,
        "public rotation packing",
    )
    q[start : start + 4] = candidate
    return MuJoCoInitialState(q, v, time=state["time"])


def require_release_seed(entry, snapshot):
    def same_words(value, saved):
        return (
            np.array_equal(value, saved)
            and np.asarray(value, dtype=np.float32).tobytes()
            == np.asarray(saved, dtype=np.float32).tobytes()
        )

    _require(
        snapshot.tick == 0
        and snapshot.time == entry.state["time"]
        and same_words(snapshot.q, entry.state["q"])
        and same_words(snapshot.v, entry.state["v"])
        and same_words(snapshot.q, entry.driver.last_q),
        "owner differs from committed source",
    )


@dataclass
class ReleaseEntry:
    scene: object
    model: object
    driver: PushFeedbackDriver
    pd: object
    history: SuccessHistory
    state: dict
    previous: dict
    tick: int
    initial: MuJoCoInitialState
    provenance: dict


def _bound_file(directory, name, digest):
    path = (directory / name).resolve()
    _require(path.is_relative_to(directory) and _digest(path) == digest, "source file binding")
    return path


def _goal_history(values):
    return SuccessHistory(**{key: tuple_values(value) for key, value in values.items()})


def prepare_release_entry(summary_path, route_output_dir):
    """Load the last completed feedback segment, its Goal history and original policy.

    The old owner and its checkpoint were not archived. Restore q/v and scalar
    time by public initialization; local revisions start at zero on the new owner.
    """
    path = Path(summary_path).resolve()
    old = json.loads(path.read_text(encoding="utf-8"))
    comparison = json.loads(path.with_name(path.stem + "-comparison.json").read_text("utf-8"))
    count, tick = old["recorded_pair_steps"], old["failed_original_input_tick"]
    _require(
        old["status"] == "partial_failure"
        and old["failure_stage"] == "feedback_reference"
        and old["controller_status"] == "withdraw_rejected"
        and count == old["accepted_pair_steps"] == comparison["verified_pair_steps"]
        and count >= 2
        and old["start_tick"] + count == tick
        and comparison["source"]["summary_sha256"] == _digest(path)
        and comparison["source"]["journal_sha256"] == old["journal"]["sha256"],
        "expected a verified, unpublished withdrawal attempt after a complete prefix",
    )
    journal = _bound_file(path.parent, old["journal"]["file"], old["journal"]["sha256"])
    # The previous line carries the causal history needed to re-evaluate the
    # last Goal. No physics or prior accepted command is executed during restore.
    ordinal = 0
    penultimate = last = None
    with journal.open("rb") as stream:
        for ordinal, line in enumerate(stream, 1):
            penultimate, last = last, line
    _require(ordinal == count, "journal record count")
    row = json.loads(last)
    previous = row["lanes"]["feedback"]
    state = old["final_states"]["feedback"]
    summary = previous["driver_state_after"]
    _require(
        row["original_output_tick"] == tick
        and state == previous["output_state"]
        and state == comparison["verified_final_states"]["feedback"]
        and summary == old["controller_final"]
        and summary["partial_route"] is None
        and summary["tick"] == tick
        and summary["status"] == "continue"
        and previous["feedback_accepted"]["completed_route"],
        "expected the exact endpoint of a completed route",
    )
    original = next(scene for scene in manipulation_scenes() if scene.key == "pusht-planar")
    provenance = old["provenance"]
    _require(original.source_sha256 == provenance["source_sha256"], "original physical model")
    model = mujoco.MjModel.from_xml_string(original.mjcf)
    data_path = Path(provenance["candidate_source"])
    _require(
        _digest(data_path) == SOURCE_SHA == provenance["candidate_source_sha256"],
        "original candidate data",
    )
    dataset = json.loads(data_path.read_text("utf-8"))
    # The original archive declares world 5. World 6 is the feedback copy of
    # that model in the paired pressure experiment.
    saved = next(r for r in dataset["rows"] if r["summary"]["world"] == old["worlds"]["open"])
    driver = PushFeedbackDriver(original, model, Path(route_output_dir), saved)
    pd, _, _ = hold_plan(original, model)
    init = public_release_initial(model, state)
    goal_row = previous["goal_result"]
    before_history = _goal_history(
        json.loads(penultimate)["lanes"]["feedback"]["goal_result"]["history"]
    )
    goal = evaluate_goal(
        original, driver.goal, SuccessFrame(**previous["goal_frame"]), before_history
    )
    _require(
        plain(goal) == goal_row and not goal.success and goal.history.tick == tick - 1,
        "preceding accepted Goal binding",
    )
    # The solver stopped before issuing another physical command. Only completed
    # routes contribute planner state, direction, and the progress budget.
    routes = summary["routes"]
    _require(len(routes) == summary["completed_routes"] > 0, "incomplete route history")
    root = path.with_suffix(".routes").resolve()
    for record in routes:
        route_path = _bound_file(root, record["file"], record["sha256"])
        _require(
            record["status"] == "sampled_reference_ready" and route_path.name == record["file"],
            "completed plan identity",
        )
        copyfile(route_path, driver.output / route_path.name)
    with gzip.open(route_path, "rt", encoding="utf-8") as stream:
        plan = json.load(stream)
    _require(
        plan["tick"] == routes[-1]["tick"] and plan["route"]["proposal"] is not None,
        "accepted push direction",
    )
    driver.previous_direction = tuple(plan["route"]["proposal"]["direction_world"])
    driver.routes.extend(routes)
    driver.completed = len(routes)
    driver.planner.ik_calls = routes[-1]["ik_calls"]
    driver.planner.path_calls = routes[-1]["path_calls"]
    driver.tick = tick
    driver.last_q = np.asarray(state["q"], dtype=float)
    driver.last_cost = driver.cost(driver.last_q)
    driver.initial_cost = summary["initial_cost"]
    driver.total_impulse = summary["total_contact_impulse_ns"]
    driver.monitor.start(0, driver.initial_cost)
    for record in summary["progress"]:
        observed = driver.monitor.observe(
            record["end_step"],
            record["cost_after"],
            contact_impulse_ns=record["contact_impulse_ns"],
            goal_success=False,
        )
        _require(plain(observed) == record, "progress history reconstruction")
    _require(plain(driver.summary()) == summary, "restored controller differs from source")
    binding = dict(
        source_summary_sha256=_digest(path),
        journal_sha256=old["journal"]["sha256"],
        source_sha256=original.source_sha256,
        scene_asset_sha256=provenance["scene_asset_sha256"],
        completed_route_sha256=routes[-1]["sha256"],
        source_tick=tick,
        source_local_tick=state["local_tick"],
        source_time=state["time"],
        initializer="unit FP64 declaration matching committed FP32 q/v; original scalar clock",
        owner_scope="fresh local owner, no historical checkpoint, hidden state or clock-word identity",
        source_controller="last physical receipt before rejected reference",
    )
    return ReleaseEntry(
        original, model, driver, pd, goal.history, state, previous, tick, init, binding
    )
