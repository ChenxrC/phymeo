"""Read saved same-episode lateral replanning; never step or restore physics.

Predictions, release trials, committed feedback and unrecorded physical tails
remain separate. Geometry release evidence is checked as a fixed per-pair ledger;
this reader does not independently reconstruct a swept collision certificate.
"""

import argparse
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from .pusht_active_comparison import (
    _array,
    _channels,
    _close,
    _control,
    _digest,
    _load,
    _LoadCompensationAudit,
    _pose_errors,
    _read,
    _receipt,
    _reference,
    _require,
    _state,
)
from .pusht_lateral_readback import _EventBinding
from .pusht_lateral_resume_audit import (
    _checkpoint,
    _events,
    bind_source,
    iter_committed,
)
from .pusht_pose_replan_comparison import (
    _goal_row,
    _planner,
    _wrench,
    summarize_motion,
    validate_counts,
    verify_separation,
)

SCHEMA = "phymeo-pusht-lateral-replan-v1"
EVENT = "phymeo-pusht-lateral-replan-event-v1"
AUDIT = "phymeo-pusht-lateral-replan-audit-v1"


def bind_terminal_source(summary, parent, audited, metadata, seed):
    """A completed prior route may replan, but cannot reset state or history."""
    _require(
        parent["schema"] == "phymeo-pusht-lateral-resume-v1"
        and audited["schema"] == "phymeo-pusht-lateral-resume-audit-v1"
        and parent["status"] == "route_complete"
        and summary["world"] == parent["world"] == metadata["world"] == 6,
        "audited completed lateral source",
    )
    # The archived route remains explicitly named previous_route. This view
    # applies the existing strict source binding without relabeling a new plan.
    bind_source(
        dict(summary, active_route=summary["previous_route"]), parent, audited, metadata, seed
    )
    _require(
        summary["previous_route"] == parent["active_route"] == metadata["active_route"]
        and parent["active_route"]["end_tick"] == summary["start_tick"]
        and metadata["controller"]["partial_route"] is None
        and metadata["controller"]["tick"] == summary["start_tick"]
        and metadata["controller"]["status"] == "continue",
        "completed route and current controller boundary before replanning",
    )


def compare_prediction(prediction, initial, final, goal, *, regular_steps):
    """Only actual regular callbacks measure whether a selected push helped."""
    _require(type(regular_steps) is int and regular_steps >= 0, "regular measured count")
    result = dict(
        measured_regular_steps=regular_steps,
        prediction_available=prediction is not None,
        predicted_position_improvement_m=None,
        predicted_absolute_yaw_improvement_rad=None,
        measured_position_improvement_m=None,
        measured_absolute_yaw_improvement_rad=None,
        actual_both_improved=None,
        scope="selected heuristic versus complete regular callbacks; neither implies original Goal success",
    )
    if prediction is not None:
        values = _array(
            [
                prediction[k]
                for k in (
                    "position_error_before_m",
                    "position_error_after_m",
                    "yaw_error_before_rad",
                    "yaw_error_after_rad",
                )
            ],
            (4,),
            "finite predicted errors",
        )
        result.update(
            predicted_position_improvement_m=float(values[0] - values[1]),
            predicted_absolute_yaw_improvement_rad=float(abs(values[2]) - abs(values[3])),
        )
    if regular_steps:
        first, last = _pose_errors(initial, goal), _pose_errors(final, goal)
        position = first["position_error_m"] - last["position_error_m"]
        yaw = abs(first["signed_yaw_error_rad"]) - abs(last["signed_yaw_error_rad"])
        result.update(
            measured_position_improvement_m=position,
            measured_absolute_yaw_improvement_rad=yaw,
            actual_both_improved=bool(position > 0 and yaw > 0),
        )
    return result


def _bound(base, name, digest, label):
    path = (base / name).resolve()
    _require(_digest(path) == digest, label + " bytes")
    return path, _read(path)


def _source(base, summary):
    p = summary["provenance"]
    source_path, parent = _bound(
        base, p["parent_report_path"], p["parent_report_sha256"], "parent report"
    )
    _, audited = _bound(base, p["parent_audit_path"], p["parent_audit_sha256"], "parent audit")
    metadata_path, metadata = _bound(
        base, p["checkpoint_metadata_path"], p["checkpoint_metadata_sha256"], "parent metadata"
    )
    checked_metadata, proof = _checkpoint(
        source_path.parent, dict(parent, _filename=source_path.name)
    )
    _require(
        metadata == checked_metadata
        and metadata_path
        == (source_path.parent / parent["final_checkpoint"]["metadata_file"]).resolve()
        and audited["source_sha256"] == p["parent_report_sha256"]
        and audited["journal_sha256"] == parent["journal"]["sha256"],
        "parent audit and capsule provenance",
    )
    seed, count = None, 0
    for row in iter_committed(
        _events(source_path.parent / parent["journal"]["file"], parent), parent
    ):
        seed, count = row, count + 1
    _require(
        seed is not None and count == parent["recorded_steps"], "complete parent feedback stream"
    )
    bind_terminal_source(summary, parent, audited, metadata, seed)
    # Independently sum the bound report ancestry, including rejected attempts.
    cursor, report, visited, attempts = source_path, parent, set(), 0
    while True:
        _require(cursor not in visited and len(visited) < 64, "bounded acyclic source ancestry")
        visited.add(cursor)
        _require(
            type(report["physical_attempts"]) is int and report["physical_attempts"] >= 0,
            "parent attempt count",
        )
        attempts += report["physical_attempts"]
        if report["schema"] == "phymeo-pusht-lateral-route-v1":
            budget = report["step_budget"]
            break
        _require(report["schema"] == "phymeo-pusht-lateral-resume-v1", "bound ancestor schema")
        old = report["provenance"]
        cursor, report = _bound(
            cursor.parent, old["parent_report_path"], old["parent_report_sha256"], "ancestor report"
        )
    _require(
        summary["original_step_budget"] == budget
        and summary["parent_physical_attempts"] == attempts == audited["original_attempts_consumed"]
        and type(summary["step_budget"]) is int
        and 0 < summary["step_budget"] <= budget - attempts,
        "unchanged total physical-attempt budget",
    )
    return parent, audited, metadata, seed, proof


def point_loads(value):
    """Read actual source identities, never infer a loaded face from a proposal."""
    ids, fn = value["source_ids"], np.asarray(value["raw_normal_force_n"], dtype=float)
    fa = np.asarray(value["raw_force_on_tool_world_n"], dtype=float).reshape(-1, 3)
    fb = np.asarray(value["raw_force_on_object_world_n"], dtype=float).reshape(-1, 3)
    _require(
        fn.ndim == 1
        and len(ids) == len(fn) == len(fa) == len(fb)
        and np.isfinite(fn).all()
        and np.isfinite(fa).all()
        and np.isfinite(fb).all()
        and np.all(fn >= 0)
        and np.array_equal(fb, -fa),
        "actual per-point bilateral loads",
    )
    _require(
        all(len(key) == 2 and all(type(v) is str and v for v in key) for key in ids),
        "actual cell/face identities",
    )
    samples = value["sample_ids"]
    _require(
        len(samples) == len(fn)
        and len({json.dumps(key, sort_keys=True) for key in samples}) == len(fn),
        "unique complete actual sample identities",
    )
    faces = {}
    for face in sorted({key[1] for key in ids}):
        mask = np.asarray([key[1] == face for key in ids], dtype=bool)
        faces[face] = dict(
            normal_load_n=float(fn[mask].sum()),
            samples=int(mask.sum()),
            force_on_tool_world_n=fa[mask].sum(axis=0).tolist(),
            force_on_object_world_n=fb[mask].sum(axis=0).tolist(),
        )
    return dict(
        samples=len(fn),
        normal_load_n=float(fn.sum()),
        force_on_tool_world_n=fa.sum(axis=0).tolist(),
        force_on_object_world_n=fb.sum(axis=0).tolist(),
        faces=faces,
    )


def _plain(value):
    return json.loads(json.dumps(asdict(value), allow_nan=False))


def _release_step(value):
    from phymeo.control.manipulation_release_monitor import (
        ReleasePairEvidence,
        ReleasePairKey,
        ReleaseStep,
    )

    return ReleaseStep(
        **dict(
            value,
            pairs=tuple(
                ReleasePairEvidence(**dict(p, key=ReleasePairKey(**p["key"])))
                for p in value["pairs"]
            ),
        )
    )


def _bind_evidence(evidence, row, world, *, seed=False):
    before, after = row["input_state"], row["output_state"]
    _require(
        evidence["world"] == world
        and evidence["complete_pair_catalog"] is True
        and evidence["input_tick"] == row["original_input_tick"]
        and evidence["output_tick"] == row["original_output_tick"]
        and evidence["force_phase"] == "accepted-step-input-geometry"
        and evidence["geometry_phase"] == "input-and-output-configuration",
        "release evidence receipt phase",
    )
    for prefix, state in (("input", before), ("output", after)):
        _require(
            all(evidence[prefix + "_" + key] == state[key] for key in ("q", "v", "time")),
            "release evidence " + prefix + " state",
        )
    whole = [
        p
        for p in evidence["pairs"]
        if p["key"]["kind"] == "area-material" and p["key"]["identity"][-1] == "whole"
    ]
    _require(len(whole) == 1, "one declared whole-material evidence")
    area, item = row["area"], whole[0]
    _close(
        item["normal_load_n"],
        area["normal_load_n"],
        "actual whole material normal load",
        rtol=0,
        atol=0,
    )
    _close(
        item["force_on_a_world"],
        area["total_force_on_pusher_world_n"],
        "actual whole tool force",
        rtol=0,
        atol=0,
    )
    _close(
        item["force_on_b_world"],
        row["measurement"]["group_force_world"],
        "actual whole object force",
        rtol=0,
        atol=0,
    )
    native = row["native"]
    _require(
        native["detected_contacts"] == len(native["supports"]),
        "saved native forces include every detected contact",
    )
    for pair in evidence["pairs"]:
        if pair["key"]["kind"] != "native-geometry":
            continue
        names = pair["key"]["identity"]
        normal, fa, fb = 0.0, np.zeros(3), np.zeros(3)
        for support in native["supports"]:
            _require(
                support["geometry_names"].count("support") == 1, "original native support receiver"
            )
            if set(support["geometry_names"]) != set(names):
                continue
            force = np.asarray(support["force_on_supported_world_n"], dtype=float)
            normal += support["normal_n"]
            force_a = -force if names[0] == "support" else force
            fa += force_a
            fb -= force_a
        _close(pair["normal_load_n"], normal, "actual native pair normal load", rtol=0, atol=0)
        _close(pair["force_on_a_world"], fa, "actual native pair first force", rtol=0, atol=0)
        _close(pair["force_on_b_world"], fb, "actual native pair second force", rtol=0, atol=0)
    if not seed:
        loads = point_loads(area)
        for face in evidence["pairs"]:
            identity = face["key"]["identity"]
            if (
                face["key"]["kind"] != "area-material"
                or len(identity) < 4
                or identity[-2] != "face"
            ):
                continue
            actual = loads["faces"].get(
                identity[-1],
                dict(
                    normal_load_n=0.0,
                    force_on_tool_world_n=[0.0] * 3,
                    force_on_object_world_n=[0.0] * 3,
                ),
            )
            for target, key in (
                ("normal_load_n", "normal_load_n"),
                ("force_on_a_world", "force_on_tool_world_n"),
                ("force_on_b_world", "force_on_object_world_n"),
            ):
                _close(face[target], actual[key], "actual face " + target, rtol=0, atol=0)
    return _release_step(evidence)


class ReleaseLedger:
    """Replay only the public per-pair safety ledger, without geometry or dynamics."""

    def __init__(self, seed, previous_row, world, *, max_steps=320):
        from phymeo.control.manipulation_release_monitor import ReleaseMonitor

        self.world = world
        step = _bind_evidence(seed, previous_row, world, seed=True)
        supports = tuple(
            p.key
            for p in step.pairs
            if p.key.kind == "native-geometry"
            and "support" in p.key.identity
            and (p.input_contact or p.output_contact or (p.normal_load_n or 0) > 0)
        )
        self.monitor = ReleaseMonitor(support_keys=supports, max_steps=max_steps)
        self.monitor.start(step)
        self.trials, self.committed = [], []

    def candidate(self, row):
        step = _bind_evidence(row["evidence"], row, self.world)
        decision = self.monitor.assess(replace(step, publication="trial"))
        _require(
            _plain(decision) == row["release_decision"] and row["selected"] is decision.admissible,
            "saved release trial decision and selection",
        )
        self.trials.append(
            dict(
                trial_index=row["trial_index"],
                trial_mode=row["trial_mode"],
                selected=row["selected"],
                decision=row["release_decision"],
                rollback_verification=row["rollback_verification"],
            )
        )

    def accepted(self, row):
        decision = self.monitor.observe(_release_step(row["evidence"]))
        _require(
            _plain(decision) == row["committed_release_decision"] and decision.admissible,
            "saved committed release ledger decision",
        )
        self.committed.append(dict(output_tick=row["original_output_tick"], status=decision.status))


class _ReplanBinding(_EventBinding):
    def __init__(self, summary):
        super().__init__(summary)
        self.start = summary["source_reference_tick_origin"]
        self.probed = self.release_requested = False

    def check(self, kind, payload, index, pending):
        s = self.summary
        if kind == "initialization":
            keys = (
                "schema",
                "world",
                "side_coverage",
                "publication_frame",
                "control_variant",
                "selection_policy",
                "material_admission",
                "start_tick",
                "local_tick_origin",
                "source_reference_tick_origin",
                "goal_tick_origin",
                "initial_state",
                "controller_initial",
                "goal_history_initial",
                "goal",
                "previous_route",
                "provenance",
                "checkpoint_restore",
                "inherited_point_loads",
                "step_budget",
                "original_step_budget",
                "parent_physical_attempts",
            )
            _require(
                not self.initialized
                and index is None
                and all(payload.get(k) == s.get(k) for k in keys)
                and payload["status"] == "running"
                and payload["final_state"] == self.retained
                and all(
                    payload[k] == 0
                    for k in (
                        "physical_attempts",
                        "physical_trial_steps",
                        "accepted_steps",
                        "recorded_steps",
                        "rejected_trials",
                    )
                ),
                "replan initialization/source identity",
            )
            self.initialized = True
        elif kind == "planner_probe":
            _require(
                self.initialized
                and not self.probed
                and pending is None
                and self.candidates == 0
                and index is None
                and payload == s["entry_probe"]
                and payload["input_state"] == self.retained
                and payload["tick"] == s["start_tick"]
                and payload["purpose"] == "entry-separation-gate"
                and payload["selected"] is False
                and payload["planner_requested"] is False,
                "entry separation probe is not an executed route",
            )
            self.probed = True
        elif kind == "release_reference":
            _require(
                self.probed
                and not self.release_requested
                and not self.planned
                and pending is None
                and index is None
                and payload["reference"] == s["release_reference"]
                and payload["seed"] == s["seed_evidence"] == s["entry_probe"]["seed"],
                "one actual release qualification request",
            )
            self.release_requested = True
        elif kind == "planner":
            _require(
                self.probed
                and not self.planned
                and pending is None
                and index is None
                and payload == s["ordinary_route"]
                and payload["tick"] == self.start + self.retained["local_tick"]
                and payload["source_q"] == self.retained["q"]
                and payload["source_v"] == self.retained["v"],
                "one ordinary plan from current accepted endpoint",
            )
            self.planned = True
        elif kind == "candidate":
            from .pusht_lateral_readback import _command_phase, _one_step, _rollback

            _require(
                index == self.candidates == payload["index"] == payload["trial_index"]
                and payload["input_state"] == self.retained,
                "candidate index and retained prestate",
            )
            _one_step(payload["input_state"], payload["output_state"])
            _command_phase(payload["control"], self.retained, self.start)
            _require(
                payload["original_input_tick"] == self.start + self.retained["local_tick"]
                and payload["original_output_tick"] == payload["original_input_tick"] + 1
                and type(payload["selected"]) is bool
                and type(payload["restored"]) is bool
                and payload["restored"] is not payload["selected"],
                "candidate publication and absolute phase",
            )
            if payload["selected"]:
                self.retained = payload["output_state"]
            else:
                _rollback(payload["rollback_verification"], self.world)
            self.candidates += 1
            self.selected += int(payload["selected"])
            self.last = payload
        elif kind == "final":
            _require(
                self.initialized
                and index is None
                and payload == {k: v for k, v in s.items() if k != "journal"}
                and self.retained == s["final_state"],
                "final replan report and physical endpoint",
            )
            tail = int(self.selected > self.candidates)
            validate_counts(
                s,
                candidate_count=self.candidates + tail,
                selected_count=self.selected,
                callback_count=self.callbacks,
                failed_attempts=self.failed_attempts - tail,
                failed_physical=self.failed_physical - tail,
            )
            _require(
                s["planner_calls"] == int(self.planned)
                and s["release_planner_calls"] == int(self.release_requested),
                "saved actual planner request counts",
            )
            status = s["status"]
            _require(
                (status == "partial_failure") == self.failure, "replan failure/status boundary"
            )
            _require(
                status
                in (
                    "partial_failure",
                    "planning_rejected",
                    "release_qualification_rejected",
                    "release_candidate_rejected",
                    "release_step_budget",
                    "controller_stopped",
                    "route_complete",
                    "step_budget",
                ),
                "replan terminal classification",
            )
            if status == "planning_rejected":
                _require(
                    self.planned and s["ordinary_route"]["status"] != "sampled_reference_ready",
                    "actual ordinary planning rejection",
                )
            if status == "release_qualification_rejected":
                _require(
                    self.release_requested and not self.planned and self.candidates == 0,
                    "release qualification stops before physics",
                )
            if status == "release_candidate_rejected":
                _require(
                    self.last is not None
                    and not self.last["selected"]
                    and self.last["phase"] == "withdraw",
                    "actual rejected withdrawal candidate",
                )
            if status == "step_budget":
                _require(
                    s["physical_attempts"] == s["step_budget"], "exhausted physical-attempt budget"
                )
            if status == "route_complete":
                _require(
                    self.planned
                    and self.callbacks == self.selected
                    and self.start + self.retained["local_tick"] == s["active_route"]["end_tick"],
                    "complete new reference endpoint",
                )
            if status == "controller_stopped":
                _require(
                    s["controller_final"]["status"] != "continue", "actual original controller stop"
                )
        else:
            super().check(kind, payload, index, pending)


def iter_bound_events(events, summary):
    """Validate ordering, including rejected trials and callback-failure tails."""
    binding, phase, pending, next_index = _ReplanBinding(summary), "start", None, 0
    for event in events:
        kind, payload, index = event["event"], event["payload"], event.get("trial_index")
        _require(event["schema"] == EVENT and phase != "final", "replan event schema/order")
        binding.check(kind, payload, index, pending)
        if kind == "initialization":
            _require(phase == "start", "first initialization")
            phase = "initial"
        elif kind == "planner_probe":
            _require(phase == "initial", "one initial probe")
            phase = "probe"
        elif kind == "release_reference":
            _require(phase == "probe", "release follows rejected entry gate")
            phase = "withdraw"
        elif kind == "planner":
            _require(
                phase in ("probe", "withdraw") and pending is None,
                "ordinary plan follows complete entry/release evidence",
            )
            phase = "regular"
        elif kind == "candidate":
            _require(
                phase in ("withdraw", "regular")
                and payload["phase"] == phase
                and index == next_index
                and pending is None,
                "candidate execution phase",
            )
            if payload["selected"]:
                pending = (index, payload)
            else:
                next_index += 1
        elif kind == "accepted_callback":
            _require(
                pending is not None
                and index == pending[0]
                and not set(payload).intersection(pending[1])
                and payload["feedback_accepted"]["measurement"] == pending[1]["measurement"],
                "complete callback binds candidate wrench",
            )
            pending, next_index = None, next_index + 1
        elif kind == "failure":
            _require(phase != "failed", "one terminal failure")
            phase, pending = "failed", None
        elif kind == "final":
            _require(phase != "start" and pending is None, "complete final publication boundary")
            phase = "final"
        else:
            raise ValueError("unknown replan event " + str(kind))
        yield event
    _require(phase == "final", "complete replan journal")


def _model_and_catalog(summary):
    import mujoco as mj

    from phymeo.backends.mujoco_tyx._constraint_model import constraint_pairs
    from phymeo.training.manipulation_tasks import manipulation_scenes

    from .pusht_public_scene import build_side_asset

    source = next(s for s in manipulation_scenes() if s.key == "pusht-planar")
    _require(
        source.source_sha256 == summary["provenance"]["source_model_sha256"],
        "original RR source model",
    )
    model = mj.MjModel.from_xml_string(source.mjcf)
    asset = build_side_asset(model, side_coverage="all-lateral")
    identity = asdict(asset.identity)
    _require(
        identity == summary["provenance"]["material_identity"],
        "unchanged declared lateral identity",
    )
    keys = set()
    for raw in constraint_pairs(model):
        pair = tuple(sorted(map(int, raw)))
        keys.add(
            (
                "native-geometry",
                tuple(model.geom(i).name for i in pair),
                tuple(int(model.geom_bodyid[i]) for i in pair),
            )
        )
    bodies = (
        int(model.body(asset.identity.compliant_body).id),
        int(model.body(asset.identity.rigid_body).id),
    )
    materials = (asset.identity.compliant_material, asset.identity.rigid_material)
    keys.add(("area-material", materials + ("whole",), bodies))
    keys.update(
        ("area-material", materials + ("face", face), bodies) for face in asset.rigid_local.face_ids
    )
    return mj, model, mj.MjData(model), asset, keys


def _catalogue(evidence, expected):
    keys = [
        (p["key"]["kind"], tuple(p["key"]["identity"]), tuple(p["key"]["body_ids"]))
        for p in evidence["pairs"]
    ]
    _require(
        len(keys) == len(set(keys)) and set(keys) == expected,
        "full source-geometry and declared-material catalogue",
    )


def _bias(mj, model, data, row):
    data.qpos[:], data.qvel[:] = row["input_state"]["q"], row["input_state"]["v"]
    mj.mj_kinematics(model, data)
    mj.mj_comPos(model, data)
    mj.mj_comVel(model, data)
    bias = np.zeros(model.nv)
    mj.mj_rne(model, data, 0, bias)
    _close(row["controller"]["pd"]["bias"], bias[:2], "independent actual-state PD bias")


def _withdraw_control(row, nodes, start_local):
    c, before, command = row["controller"], row["input_state"], row["control"]
    elapsed = (before["local_tick"] - start_local) * 0.001
    mode = row["trial_mode"]
    _require(
        c["mode"] == mode
        and mode in ("smooth", "constant_rate")
        and c["elapsed_s"] == elapsed
        and c["measurement_q"] == before["q"]
        and c["measurement_v"] == before["v"]
        and c["stage"] == "withdraw"
        and "load_compensation" not in c,
        "withdrawal actual state/reference mode",
    )
    _require(
        len(nodes) == 2
        and nodes[0][0] == 0
        and nodes[1][0] > 0
        and nodes[0][1] == nodes[1][1] == "withdraw",
        "original finite withdrawal nodes",
    )
    if mode == "smooth":
        target, velocity, _ = _reference(nodes, elapsed)
    else:
        a, b, duration = np.asarray(nodes[0][2]), np.asarray(nodes[1][2]), nodes[1][0]
        target = a + min(elapsed / duration, 1) * (b - a)
        velocity = (b - a) / duration if elapsed < duration else np.zeros_like(a)
    for key, value in (("target_q", target), ("target_v", velocity)):
        _close(c[key], value, "independent withdrawal " + key)
    pd = c["pd"]
    _require(
        all(pd[k] == [0, 1] for k in ("qpos_addresses", "dof_addresses", "actuator_ids"))
        and pd["position"] == before["q"][:2]
        and pd["velocity"] == before["v"][:2],
        "withdrawal original RR PD axes",
    )
    _close(pd["target"], target, "withdrawal PD target")
    _close(pd["target_velocity"], velocity, "withdrawal PD target velocity")
    raw = (
        np.asarray(pd["bias"]) + 25 * (target - before["q"][:2]) + 3 * (velocity - before["v"][:2])
    )
    _close(pd["unclipped_effort"], raw, "withdrawal original PD arithmetic")
    effort = np.clip(raw, -19.9, 19.9)
    _close(pd["effort"], effort, "withdrawal original effort limit")
    channels, transport = _channels(command), c["transport"]
    direct = np.asarray(effort, np.float32)
    _require(
        transport["kind"] == "original-env-fp32-atanh-cuda-tanh"
        and transport["device"].startswith("cuda")
        and transport["scale"] == 20.0
        and transport["bias"] == 0.0
        and transport["operator"] == "bias + scale * torch.tanh(raw)"
        and np.array_equal(transport["direct_pd_ctrl"], direct)
        and np.array_equal(transport["raw_action"], np.arctanh(direct / np.float32(20)))
        and transport["transported_ctrl"] == channels["ctrl"],
        "withdrawal original CUDA command transport",
    )
    _close(
        transport["direct_to_transported_delta"],
        np.asarray(channels["ctrl"]) - direct,
        "withdrawal transport difference",
        atol=0,
        rtol=0,
    )


def _point_totals(area, measurement):
    values = point_loads(area)
    _require(values["samples"] == area["samples"], "all real samples retained")
    _close(values["normal_load_n"], area["normal_load_n"], "raw normal force sum", atol=0, rtol=0)
    _close(
        values["force_on_tool_world_n"],
        area["total_force_on_pusher_world_n"],
        "raw tool force sum",
        atol=0,
        rtol=0,
    )
    _close(
        values["force_on_object_world_n"],
        measurement["group_force_world"],
        "raw object force sum",
        atol=0,
        rtol=0,
    )
    return values


def audit(path):
    path = Path(path).resolve()
    s, base = _read(path), path.parent
    _require(
        s["schema"] == SCHEMA
        and s["side_coverage"] == "all-lateral"
        and s["publication_frame"] == "rigid-origin"
        and s["control_variant"] == "load-compensation-half"
        and s["selection_policy"] == "joint-pose-balanced"
        and s["material_admission"] == "declared-finite-faces-v1",
        "same-episode lateral replan contract",
    )
    _state(s["initial_state"], s["local_tick_origin"])
    parent, parent_audit, metadata, seed_row, source_proof = _source(base, s)
    mj, model, data, asset, catalogue = _model_and_catalog(s)
    inherited = s["inherited_point_loads"]
    _require(
        inherited["identity"] == asdict(asset.identity)
        and inherited["receipt_local_tick"] == s["local_tick_origin"]
        and inherited["stamp"]["step_index"] == seed_row["input_state"]["local_tick"]
        and inherited["output_time"] == s["initial_state"]["time"]
        and all(inherited["input_" + k] == seed_row["input_state"][k] for k in ("q", "v", "time"))
        and inherited["raw_normal_force_n"] == seed_row["area"]["raw_normal_force_n"],
        "inherited load is the prior actual input geometry",
    )
    old_loads = point_loads(inherited)
    _close(
        old_loads["normal_load_n"],
        seed_row["area"]["normal_load_n"],
        "inherited normal load",
        atol=0,
        rtol=0,
    )
    _close(
        old_loads["force_on_tool_world_n"],
        seed_row["area"]["total_force_on_pusher_world_n"],
        "inherited actual tool force",
        atol=0,
        rtol=0,
    )
    _close(
        old_loads["force_on_object_world_n"],
        seed_row["measurement"]["group_force_world"],
        "inherited actual object force",
        atol=0,
        rtol=0,
    )
    planning = dict(probe=None, release_reference=None, ordinary=None)
    rejected, face_history = [], []
    previous, previous_row = s["initial_state"], seed_row
    latest_evidence = release_ref = ledger = pending = compensation = regular_summary = None
    regular_initial = regular_final = s["initial_state"]
    regular_steps = count = 0
    fixed_mode = None
    last_trial = None
    history = s["goal_history_initial"]
    prediction = None
    selected_face_ids = []

    def rows():
        nonlocal \
            previous, \
            previous_row, \
            latest_evidence, \
            release_ref, \
            ledger, \
            pending, \
            compensation, \
            regular_summary
        nonlocal \
            regular_initial, \
            regular_final, \
            regular_steps, \
            count, \
            fixed_mode, \
            last_trial, \
            history, \
            prediction, \
            selected_face_ids
        for event in iter_bound_events(_events(base / s["journal"]["file"], s), s):
            kind, p = event["event"], event["payload"]
            tick = s["source_reference_tick_origin"] + previous["local_tick"]
            if kind == "planner_probe":
                evidence = p["seed"]
                _catalogue(evidence, catalogue)
                _bind_evidence(evidence, previous_row, s["world"], seed=True)
                try:
                    verify_separation(evidence, previous, tick)
                    clear = True
                except ValueError:
                    clear = False
                _require(
                    p["status"] == ("strictly_separated" if clear else "entry_separation_rejected")
                    and isinstance(p["errors"], list)
                    and bool(p["errors"]) is not clear,
                    "saved entry separation result",
                )
                planning["probe"] = p
                latest_evidence = evidence
            elif kind == "release_reference":
                ref, evidence = p["reference"], p["seed"]
                _require(
                    planning["probe"]["status"] == "entry_separation_rejected",
                    "release only after actual gate rejection",
                )
                route_path, saved = _bound(
                    base, ref["file"], ref["sha256"], "original release request"
                )
                route = saved["route"]
                _require(
                    saved["tick"] == tick
                    and saved["q"] == previous["q"]
                    and saved["v"] == previous["v"]
                    and route["nodes"] == ref["nodes"]
                    and route["status"] == ref["route_status"]
                    and ref["max_steps"] == 320,
                    "actual release reference source and bound",
                )
                before_budget, after_budget = (
                    saved["planning_budget_before"],
                    saved["planning_budget_after"],
                )
                _require(
                    all(
                        type(before_budget[k]) is int
                        and type(after_budget[k]) is int
                        and before_budget[k] == s["controller_initial"][k]
                        and after_budget[k] >= before_budget[k]
                        for k in ("ik_calls", "path_calls")
                    ),
                    "inherited release planning counters",
                )
                supports = [
                    p["key"]
                    for p in evidence["pairs"]
                    if p["key"]["kind"] == "native-geometry"
                    and "support" in p["key"]["identity"]
                    and (p["input_contact"] or p["output_contact"] or (p["normal_load_n"] or 0) > 0)
                ]
                _require(
                    ref["support_keys"] == supports, "only real original native support exemptions"
                )
                planning["release_reference"] = dict(
                    file=str(route_path),
                    sha256=ref["sha256"],
                    status=route["status"],
                    proposals=saved["proposals"],
                    route=route,
                    planning_budget_before=before_budget,
                    planning_budget_after=after_budget,
                    dynamic_unloading_proven=False,
                )
                release_ref = dict(ref, start_local=previous["local_tick"])
                if route["status"] == "sampled_release_ready" and route["proposal"] is None:
                    ledger = ReleaseLedger(
                        evidence, previous_row, s["world"], max_steps=ref["max_steps"]
                    )
                else:
                    _require(
                        s["status"] == "release_qualification_rejected"
                        and s["release_rejection"]["actual_status"] == route["status"]
                        and s["release_rejection"]["required_status"] == "sampled_release_ready"
                        and s["release_rejection"]["attempts"] == route["attempts"],
                        "retain original release qualification rejection",
                    )
            elif kind == "planner":
                verify_separation(latest_evidence, previous, tick)
                if release_ref is not None:
                    _require(
                        ledger is not None
                        and ledger.monitor.status == "released"
                        and (previous["local_tick"] - release_ref["start_local"]) * 0.001
                        >= release_ref["nodes"][-1][0],
                        "ordinary replan follows complete same-path actual release",
                    )
                record, proposal = _planner(
                    base,
                    p,
                    previous,
                    tick,
                    selection_policy=s["selection_policy"],
                    material_admission=s["material_admission"],
                    material_asset_sha256=s["provenance"]["scene_asset_sha256"],
                    side_coverage="all-lateral",
                )
                saved = _read((base / p["route_file"]).resolve())
                planning["ordinary"] = dict(
                    record, proposals=saved["proposals"], route=saved["route"]
                )
                prediction = record["heuristic_prediction"]
                regular_initial = regular_final = previous
                if saved["route"]["status"] == "sampled_reference_ready":
                    active = s["active_route"]
                    duration = saved["route"]["nodes"][-1][0]
                    _require(
                        active["start_tick"] == tick
                        and active["end_tick"] == tick + round(duration / 0.001)
                        and abs(round(duration / 0.001) * 0.001 - duration) <= 1e-12
                        and active["nodes"] == saved["route"]["nodes"]
                        and active["sha256"] == p["route_sha256"]
                        and (base / active["file"]).resolve() == (base / p["route_file"]).resolve()
                        and proposal in saved["proposals"],
                        "unchanged selected new reference",
                    )
                    remaining = s["step_budget"] - (
                        event["payload"]["tick"] - s["start_tick"] + len(rejected)
                    )
                    budget = s["reference_budget"]
                    _require(
                        budget["route_steps"] == active["end_tick"] - active["start_tick"]
                        and budget["available_steps"] == remaining
                        and budget["complete_route_within_budget"]
                        is (budget["route_steps"] <= remaining),
                        "new route prefix budget without changing reference duration",
                    )
                    actual_before = _pose_errors(previous, s["goal"])
                    _close(
                        prediction["position_error_before_m"],
                        actual_before["position_error_m"],
                        "proposal actual initial XY error",
                        atol=2e-6,
                        rtol=2e-6,
                    )
                    _close(
                        prediction["yaw_error_before_rad"],
                        actual_before["signed_yaw_error_rad"],
                        "proposal actual initial yaw error",
                        atol=2e-6,
                        rtol=2e-6,
                    )
                    material = next(
                        a
                        for a in saved["route"]["attempts"]
                        if a.get("kind") == "material_admission"
                    )
                    selected = next(
                        a
                        for a in material["proposals"]
                        if all(
                            a[k] == proposal[k]
                            for k in ("index", "distance_ordinal", "input_binding")
                        )
                    )
                    selected_face_ids = selected["face_ids"]
                    regular_summary = dict(
                        s,
                        start_tick=tick,
                        initial_state=previous,
                        first_command_binding=s.get("first_regular_command_binding"),
                    )
                    compensation = _LoadCompensationAudit(
                        regular_summary, {"trials": [previous_row]}, saved
                    )
            elif kind == "candidate":
                _receipt(p)
                _wrench(p, p["measurement"], s["world"])
                _load(dict(p, feedback_accepted={"measurement": p["measurement"]}))
                _require(
                    p["side_coverage"] == "all-lateral"
                    and p["material_identity"] == asdict(asset.identity)
                    and p["stored_control_matches_submitted"] is True,
                    "actual candidate material and control publication",
                )
                channels = _channels(p["control"])
                digest = hashlib.sha256(
                    json.dumps(
                        channels, sort_keys=True, separators=(",", ":"), allow_nan=False
                    ).encode()
                ).hexdigest()
                _require(
                    p["control"]["channels_sha256"] == digest, "derived submitted channel hash"
                )
                loads = _point_totals(p["area"], p["measurement"])
                _require(
                    all(
                        k[0] in asset.pressure_local.cell_ids and k[1] in asset.rigid_local.face_ids
                        for k in p["area"]["source_ids"]
                    ),
                    "actual load source belongs to declared material",
                )
                _bias(mj, model, data, p)
                if p["phase"] == "withdraw":
                    _require(ledger is not None, "no physical release without qualified reference")
                    _catalogue(p["evidence"], catalogue)
                    _withdraw_control(p, release_ref["nodes"], release_ref["start_local"])
                    if fixed_mode is not None:
                        _require(
                            p["trial_mode"] == fixed_mode,
                            "same selected withdrawal mode throughout",
                        )
                    elif p["trial_mode"] == "constant_rate":
                        codes = (
                            set(v["code"] for v in last_trial["release_decision"]["violations"])
                            if last_trial
                            else set()
                        )
                        _require(
                            last_trial is not None
                            and not last_trial["selected"]
                            and last_trial["trial_mode"] == "smooth"
                            and last_trial["input_state"] == p["input_state"]
                            and p["input_state"]["local_tick"] == release_ref["start_local"]
                            and codes
                            and codes
                            <= {"inward_motion", "penetration_increased", "normal_load_increased"},
                            "only registered initial smooth-to-constant retry",
                        )
                    else:
                        _require(
                            p["trial_mode"] == "smooth" and last_trial is None,
                            "original first withdrawal candidate",
                        )
                    ledger.candidate(p)
                    if p["selected"]:
                        fixed_mode = p["trial_mode"]
                else:
                    _require(
                        regular_summary is not None
                        and p["selected"]
                        and p["evidence"] is None
                        and p["release_decision"] is None,
                        "ordinary admitted candidate",
                    )
                    index = p["input_state"]["local_tick"] - regular_initial["local_tick"]
                    _control(
                        dict(p, submitted_control=p["control"]),
                        regular_summary,
                        index,
                        base,
                        compensation,
                    )
                if p["selected"]:
                    pending = p
                else:
                    rejected.append(
                        dict(
                            trial_index=p["trial_index"],
                            phase=p["phase"],
                            trial_mode=p["trial_mode"],
                            input_state=p["input_state"],
                            candidate_output_state=p["output_state"],
                            load=loads,
                            measurement=p["measurement"],
                            release_decision=p["release_decision"],
                            evidence=p["evidence"],
                            geometry_comparison=p["geometry_comparison"],
                            rollback_verification=p["rollback_verification"],
                        )
                    )
                last_trial = p
            elif kind == "accepted_callback":
                row = dict(pending, **p)
                _goal_row(row, s["goal"], s["world"], goal_tick=row["input_state"]["local_tick"])
                _require(
                    row["driver_state_after"]["tick"] == row["original_output_tick"],
                    "accepted controller advances exactly once",
                )
                if row["phase"] == "withdraw":
                    ledger.accepted(row)
                    latest_evidence = row["evidence"]
                else:
                    _require(
                        row["committed_release_decision"] is None,
                        "regular callback has no release decision",
                    )
                    compensation.accepted(row)
                    regular_steps += 1
                    regular_final = row["output_state"]
                    row["planned_push_direction_world"] = planning["ordinary"]["route"]["proposal"][
                        "direction_world"
                    ]
                face_history.append(
                    dict(
                        input_tick=row["original_input_tick"],
                        phase=row["phase"],
                        faces=point_loads(row["area"])["faces"],
                    )
                )
                previous, previous_row, history = (
                    row["output_state"],
                    row,
                    row["goal_result"]["history"],
                )
                count += 1
                pending = None
                yield row
            elif kind == "failure":
                pending = None

    actual = summarize_motion(s["initial_state"], s["goal"], rows())
    _require(
        count == s["recorded_steps"] and history == s["goal_history_final"],
        "complete feedback/Goal history prefix",
    )
    if not s.get("unrecorded_accepted_step"):
        _require(
            previous == s["final_state"]
            and s["controller_final"]["tick"]
            == s["source_reference_tick_origin"] + previous["local_tick"],
            "recorded controller at physical endpoint",
        )
    checkpoint = None
    if s.get("final_checkpoint") is not None:
        _, checkpoint = _checkpoint(
            base, dict(s, _filename=path.name, active_route=s.get("active_route"))
        )
    loaded_faces = sorted(
        {
            face
            for row in face_history
            if row["phase"] == "regular"
            for face, value in row["faces"].items()
            if value["normal_load_n"] > 0
        }
    )
    old_faces = sorted(
        face for face, value in old_loads["faces"].items() if value["normal_load_n"] > 0
    )
    release_complete = ledger is not None and ledger.monitor.status == "released"
    if planning["release_reference"] is not None:
        planning["release_reference"]["dynamic_unloading_proven"] = release_complete
    return dict(
        schema=AUDIT,
        status=s["status"],
        source_sha256=_digest(path),
        journal_sha256=s["journal"]["sha256"],
        physical_final_state=s["final_state"],
        actual=actual,
        goal=s["goal"],
        unrecorded_accepted_step=s.get("unrecorded_accepted_step"),
        accepted_steps=s["accepted_steps"],
        recorded_steps=count,
        physical_attempts=s["physical_attempts"],
        budget=dict(
            original=s["original_step_budget"],
            prior_attempts=s["parent_physical_attempts"],
            allowance=s["step_budget"],
            used=s["physical_attempts"],
            remaining=s["original_step_budget"]
            - s["parent_physical_attempts"]
            - s["physical_attempts"],
        ),
        parent=dict(
            report_sha256=s["provenance"]["parent_report_sha256"],
            audit_sha256=s["provenance"]["parent_audit_sha256"],
            physical_final_state=parent["final_state"],
            checkpoint=source_proof,
            verified_steps=parent["recorded_steps"],
        ),
        final_checkpoint=checkpoint,
        control=dict(variant=s["control_variant"], verified_steps=count),
        original_attempts_consumed=s["parent_physical_attempts"] + s["physical_attempts"],
        combined_verified_steps=parent_audit["combined_verified_steps"] + count,
        planning=planning,
        prediction_vs_actual=compare_prediction(
            prediction, regular_initial, regular_final, s["goal"], regular_steps=regular_steps
        ),
        release=dict(
            verified_committed_steps=0 if ledger is None else len(ledger.committed),
            per_pair_monitor_status=None if ledger is None else ledger.monitor.status,
            release_complete_observed=release_complete,
            rejected_trials=rejected,
            trial_decisions=[] if ledger is None else ledger.trials,
            committed_decisions=[] if ledger is None else ledger.committed,
        ),
        face_contact=dict(
            previous_input_loads=old_loads,
            previous_positive_load_faces=old_faces,
            proposed_face_ids=selected_face_ids,
            regular_positive_load_faces=loaded_faces,
            actual_new_loaded_faces=sorted(set(loaded_faces) - set(old_faces)),
            accepted_face_history=face_history,
        ),
        termination={
            k: s.get(k)
            for k in (
                "status",
                "failure_stage",
                "errors",
                "failed_candidate",
                "release_status",
                "release_rejection",
                "checkpoint_save_error",
            )
        },
        scope="saved source/capsule/history, actual trial and complete-callback control/receipt/force/integration checks; per-pair release ledger rechecked from supplied geometry evidence, not independently regenerated or swept; selected predictions are not measured success; Goal phase/pose metrics checked without replaying the full Goal evaluator; no physics step or capsule restore",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.input)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, allow_nan=False, indent=2)
        stream.write("\n")
    print(result["status"], result["actual"]["verified_steps"])


if __name__ == "__main__":
    main()
