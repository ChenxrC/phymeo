"""Bounded, stdlib-only audit of the frozen C5 saved evidence (not a rerun).

The membership below comes from jobs 264--269 and 276--279 plus their bound
producer sources. Failed prefixes are retained, never promoted to complete runs.
No Phymeo, NumPy, Newton, pytest, oracle, or producer is imported/executed.

Authoring status: UNEXECUTED. The parent owns the eventual serial audit run.
"""

import argparse
import gzip
import hashlib
import io
import json
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

MIB = 1024 * 1024
MAX_COMPRESSED = 16 * MIB
MAX_PAYLOAD = 128 * MIB
MAX_TOTAL_PAYLOAD = 512 * MIB
MAX_METADATA = 16 * MIB
MAX_TOTAL_FILES = 2048
H = 1 / 1024
STEP = "tests/scale/test_resident_affine_step.py"
COLLISION = "tests/scale/test_resident_affine_collision.py"
LOSS = "tests/scale/test_resident_affine_device_loss.py"
REFERENCE = "tests/reference_newton/robot/resident_affine_reference.py"
EXACT = ("A01F", "A02", "A03", "A04", "A05", "A05G", "A05T")
WINDOWS = ("A01", "A02", "A03", "A04", "A05", "A05G", "A05T", "N9", "N16")
NATIVE = ("A01", "A02", "A04", "A05G", "A03-flattened")


@dataclass(frozen=True)
class Member:
    source: str
    function: str
    group: str
    parameter: tuple = ()
    files: tuple = ()
    pytest_id: str = ""

    @property
    def node(self):
        return (
            self.source
            + "::"
            + self.function
            + ("[" + self.pytest_id + "]" if self.pytest_id else "")
        )

    @property
    def identity(self):
        return self.group + ":" + json.dumps(self.parameter, separators=(",", ":"))


def membership():
    """Explicit authored parametrization/order, independent of found files."""
    step, collision, loss, host = [], [], [], []

    def add(out, source, function, group, variants, filenames):
        for parameter, param_id in variants:
            names = filenames(parameter) if callable(filenames) else filenames
            out.append(Member(source, function, group, parameter, names, param_id))

    add(
        step,
        STEP,
        "test_a01_public_affine_tree_keeps_follower_inertia_and_real_reaction",
        "public",
        [((), "")],
        ("a01-public-input", "a01-public-result"),
    )
    add(
        step,
        STEP,
        "test_a01_a05_independent_kkt_exact_first_steps",
        "exact",
        [((name,), name) for name in EXACT],
        lambda p: (p[0] + "-exact",),
    )
    add(
        step,
        STEP,
        "test_a01_a07_forty_accepted_steps_without_reseeding",
        "window",
        [((name,), name) for name in WINDOWS],
        lambda p: (p[0] + "-window",),
    )
    add(
        step,
        STEP,
        "test_a04_a05_covariant_installation_and_current_com_load",
        "covariance",
        [
            ((placement, name), placement + "-" + name)
            for placement in ("cycle", "large")
            for name in ("A04", "A05G")
        ],
        ("covariance",),
    )
    add(
        step,
        STEP,
        "test_a02_source_reorder_keeps_named_axes_and_full_bodies",
        "reorder",
        [((), "")],
        ("named-reorder",),
    )
    add(
        step,
        STEP,
        "test_a03_chain_and_flattened_graph_keep_motion_but_not_row_forces",
        "chain",
        [((), "")],
        ("chain-versus-flatten",),
    )
    for function, group, values, file in (
        ("test_a06_scalar_implicit_pd_cap_and_exact_nonsaturated_drive", "pd", (1.0, 16.0), "pd"),
        (
            "test_a06_wide_cap_and_large_terms_do_not_hide_small_effort",
            "small",
            (False, True),
            "small-effort",
        ),
        ("test_a06_delay_only_drive_triplet_external_load_is_current", "delay", (0, 1), "delay"),
    ):
        add(step, STEP, function, group, [((value,), str(value)) for value in values], (file,))
    add(
        step,
        STEP,
        "test_a06_one_sweep_rejects_handderived_cross_residual_then_full_budget_solves",
        "budget",
        [((), "")],
        ("budget",),
    )
    add(
        step,
        STEP,
        "test_a07_position_intersection_candidate_reject_retains_all_rows",
        "domain",
        [((value,), str(value)) for value in (-1, 1)],
        ("domain-rejection",),
    )
    add(
        step,
        STEP,
        "test_a07_mixed_bank_failure_retry_is_atomic",
        "atomic",
        [
            ((fault, warm), fault + "-" + str(warm))
            for fault in ("partial_upload", "step_status", "restore_sync")
            for warm in (0, 1)
        ],
        ("atomic-retry",),
    )
    add(
        step,
        STEP,
        "test_a07_restore_rejects_graph_full_state_and_history_mismatch",
        "restore",
        [((warm,), str(warm)) for warm in (0, 1)],
        ("restore-schema",),
    )
    add(
        step,
        STEP,
        "test_a07_owned_model_input_and_readbacks_and_old_scalar_reader_rejection",
        "owned",
        [((), "")],
        ("owned",),
    )
    add(
        host,
        STEP,
        "test_a07_invalid_host_graph_rejected_before_device",
        "host-graph",
        [
            ((value,), value)
            for value in (
                "cycle",
                "duplicate_follower",
                "unknown",
                "type",
                "zero",
                "nonfinite",
                "empty_interval",
                "duplicate_name",
            )
        ],
        (),
    )
    add(
        host,
        STEP,
        "test_a07_static_device_domain_rejects_without_silently_changing_graph",
        "host-domain",
        [
            ((value,), value)
            for value in ("N17", "gain_underflow", "gain_overflow", "delay2", "budget0")
        ],
        (),
    )
    # Float parameter text is not inferred from pytest's version-dependent repr.
    angles = (-256 * math.pi, -2 * math.pi, 0.0, 0.37, 2 * math.pi, 256 * math.pi)
    add(
        collision,
        COLLISION,
        "test_a08_all_materials_use_full_accepted_mimic_pose",
        "shapes",
        [
            ((placement, angle), placement + "-angle" + str(i))
            for placement in ("base", "cycle", "large")
            for i, angle in enumerate(angles)
        ],
        ("all-shapes",),
    )
    add(
        collision,
        COLLISION,
        "test_a08_predeclared_current_box_near_touch_and_clear_miss",
        "near",
        [((gap,), str(gap)) for gap in (-1 / 65536, 0.25)],
        ("near-miss",),
    )
    add(
        collision,
        COLLISION,
        "test_a08_capsule_axis_is_finite_and_keeps_both_skin_ends",
        "capsule",
        [((zone,), zone) for zone in ("axis", "end_skin", "beyond_end")],
        ("finite-capsule",),
    )
    add(
        collision,
        COLLISION,
        "test_a05_original_asset_sites_and_fixed_tool_materials_survive_projection",
        "asset",
        [((name,), name) for name in ("A03", "A05T")],
        ("asset-source",),
    )
    add(
        collision,
        COLLISION,
        "test_a07_affine_graph_owner_and_source_identity_rejected_before_allocation",
        "identity",
        [
            ((fault,), fault)
            for fault in (
                "wrong_graph",
                "wrong_model",
                "revision",
                "vertex_ids",
                "body_alias",
                "missing_exclusion",
            )
        ],
        (),
    )
    for function, group, stem in (
        ("test_a08_complete_heterogeneous_forty_step_output", "full", "full-w"),
        ("test_a08_warm_no_snapshot_envelope_upload_or_device_allocation", "warm", "warm-w"),
    ):
        add(
            collision,
            COLLISION,
            function,
            group,
            [((n,), str(n)) for n in (1, 4, 32)],
            lambda p, prefix=stem: (prefix + str(p[0]),),
        )
    add(
        collision,
        COLLISION,
        "test_a07_a08_epoch_owner_restore_reset_and_numeric_rejection",
        "epoch",
        [((warm,), str(warm)) for warm in (0, 1)],
        ("epoch",),
    )
    for function, group, file in (
        (
            "test_a08_capacity_and_failed_query_publish_no_partial_result",
            "query-failure",
            "query-failure",
        ),
        (
            "test_a08_close_callback_failure_still_releases_other_plans_and_runtime",
            "closed",
            "closed",
        ),
        ("test_a08_empty_batch_and_empty_collision_plan_own_no_device_state", "empty", "empty"),
    ):
        add(collision, COLLISION, function, group, [((), "")], (file,))
    add(
        collision,
        COLLISION,
        "test_a07_a08_exact_full_coordinate_domain_edge_and_candidate_rejection",
        "edge",
        [((direction,), str(direction)) for direction in (-1, 1)],
        ("exact-domain-edge",),
    )
    add(
        loss,
        LOSS,
        "test_permanent_restore_sync_loss_requires_checkpoint_rebuild",
        "loss",
        [((warm,), str(warm)) for warm in (0, 1)],
        ("permanent-device-loss-input", "permanent-device-loss-result"),
    )
    add(
        loss,
        LOSS,
        "test_ordinary_affine_and_scalar_share_control_restore_and_retry_transaction",
        "mixed",
        [((warm,), str(warm)) for warm in (0, 1)],
        ("mixed-tree-transaction",),
    )
    require((len(step), len(collision), len(loss), len(host)) == (41, 44, 4, 13), "manifest size")
    return step, collision, loss, host


class AuditError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise AuditError(message)


def fields(value, names, label="record"):
    require(isinstance(value, dict), label + " must be a mapping")
    missing = set(names.split() if isinstance(names, str) else names) - value.keys()
    require(not missing, label + " missing " + str(sorted(missing)))


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def array(value, *dimensions):
    if not dimensions:
        require(number(value), "nonfinite/non-numeric array element")
    else:
        require(
            isinstance(value, list) and len(value) == dimensions[0],
            "array shape mismatch: expected " + str(dimensions),
        )
        for item in value:
            array(item, *dimensions[1:])


def close(actual, expected, atol=5e-6, rtol=5e-5):
    """Compare saved numbers only; never reevaluate a physical oracle."""
    if isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), "saved comparison shape")
        for a, e in zip(actual, expected):
            close(a, e, atol, rtol)
    else:
        require(number(actual) and number(expected), "saved comparison nonfinite")
        require(
            abs(actual - expected) <= atol + rtol * abs(expected),
            "saved comparison exceeds original gate",
        )


def dimensions(data):
    fields(
        data,
        "name model P c C rows independent s u motor effort force torque lower upper speed rating",
        "input",
    )
    n, k = len(data["c"]), len(data["s"])
    require(1 <= k < n <= 16, "affine input dimensions")
    m = data["model"]
    fields(
        m,
        "parents kinds offsets orientations axes mass com inertia gravity names joint_names",
        "model",
    )
    nb = len(m["mass"])
    require(sum(kind != "fixed" for kind in m["kinds"]) == n, "full rows vs authored bodies")
    for key in ("parents", "kinds", "names", "joint_names"):
        require(len(m[key]) == nb, "model field length " + key)
    for key, width in (
        ("offsets", 3),
        ("orientations", 4),
        ("axes", 3),
        ("com", 3),
        ("inertia", 3),
    ):
        array(m[key], nb, width)
    array(m["mass"], nb)
    require(all(x > 0 for x in m["mass"]), "nonpositive authored mass")
    array(m["gravity"], 3)
    array(data["P"], n, k)
    array(data["C"], n - k, n)
    array(data["rows"], n - k, 4)
    for key in ("c", "effort", "lower", "upper", "speed", "rating"):
        array(data[key], n)
    for key in ("s", "u", "motor"):
        array(data[key], k)
    for key in ("force", "torque"):
        array(data[key], nb, 3)
    require(
        len(data["independent"]) == k and len(set(data["independent"])) == k, "independent row IDs"
    )
    return n, k


def checkpoint(value):
    fields(
        value, "world topology tick position velocity applied_effort history control", "checkpoint"
    )
    require(
        type(value["world"]) is int and type(value["tick"]) is int and value["tick"] >= 0,
        "world/tick",
    )
    require(
        isinstance(value["topology"], list) and bool(value["topology"]), "missing model/graph token"
    )
    k = len(value["position"])
    array(value["position"], k)
    array(value["velocity"], k)
    if "full_position" in value:
        n = len(value["full_position"])
        require(1 <= k < n <= 16, "checkpoint K/N")
        for name in ("applied_effort", "generalized_applied"):
            array(value[name], k)
        for name in (
            "full_position",
            "full_velocity",
            "full_applied_effort",
            "full_generalized_applied",
        ):
            array(value[name], n)
        require(len(value["control"]) == 2, "drive/full-load control blocks")
        array(value["control"][0], k, 3)
        array(value["control"][1], n, 7)
        if value["history"]:
            array(value["history"], k, 3)
        return n, k
    if "generalized_applied" in value:
        array(value["control"], k, 10)
        array(value["generalized_applied"], k)
        if value["history"]:
            array(value["history"], k, 3)
    return k, k


def kinematics(value):
    fields(
        value,
        "tick joint_types position velocity link_position_m link_rotation com_position_m",
        "FK",
    )
    n = len(value["position"])
    array(value["position"], n)
    array(value["velocity"], n)
    array(value["link_position_m"], n, 3)
    array(value["link_rotation"], n, 3, 3)
    array(value["com_position_m"], n, 3)
    require(len(value["joint_types"]) == n, "FK full joint types")
    if "independent_position" in value:
        k = len(value["independent_position"])
        array(value["independent_velocity"], k)
        require(
            len(value["joint_names"]) == k and len(value["full_joint_names"]) == n, "K/N named rows"
        )


def diagnostic(value):
    fields(
        value,
        "entry_tick exit_tick dt_s entry_position entry_velocity mass bias external drive acceleration "
        "linear_jacobian angular_jacobian iterations control_residual equation_residual",
        "diagnostic",
    )
    k, n = len(value["entry_position"]), len(value["mass"])
    require(
        value["exit_tick"] == value["entry_tick"] + 1 and value["dt_s"] == H,
        "diagnostic sample time",
    )
    array(value["mass"], n, n)
    for name in ("bias", "external"):
        array(value[name], n)
    for name in ("entry_position", "entry_velocity", "drive", "acceleration"):
        array(value[name], k)
    for name in ("linear_jacobian", "angular_jacobian"):
        array(value[name], n, 3, n)
    require(number(value["control_residual"]), "control residual missing/nonfinite")
    require(
        number(value["equation_residual"]) and value["equation_residual"] <= 2e-5,
        "mass equation residual",
    )
    if "reduced_mass" in value:
        array(value["reduced_mass"], k, k)
        for name in ("reduced_bias", "reduced_external"):
            array(value[name], k)
        for name in (
            "full_entry_position",
            "full_entry_velocity",
            "full_acceleration",
            "constraint_effort_full",
        ):
            array(value[name], n)
        for name in ("reduced_linear_jacobian", "reduced_angular_jacobian"):
            array(value[name], n, 3, k)
        for name in ("row_multipliers", "row_powers_w"):
            array(value[name], n - k)
        require(all(abs(x) <= 2e-5 for x in value["row_powers_w"]), "saved row power")
        require(abs(value["constraint_power_w"]) <= 2e-5, "saved constraint power")


def key(value):
    fields(value, "world body shape", "collision key")
    require(all(type(value[name]) is int for name in ("world", "body", "shape")), "collision IDs")
    return tuple(value[name] for name in ("world", "body", "shape"))


def geometry(value):
    fields(
        value,
        "topology_generation state_epoch world_keys ticks shape_keys shape_revisions bounds registered_pairs candidates",
        "geometry",
    )
    keys = [key(item) for item in value["shape_keys"]]
    require(len(keys) == len(set(keys)), "duplicate shape key")
    require(len(value["shape_revisions"]) == len(keys), "shape revision coverage")
    array(value["bounds"], len(keys), 6)
    require(
        all(all(row[i] <= row[i + 3] for i in range(3)) for row in value["bounds"]),
        "inverted bounds",
    )
    require(len(value["ticks"]) == len(value["world_keys"]), "world/tick coverage")
    require(len(set(value["world_keys"])) == len(value["world_keys"]), "duplicate world")
    require(all(type(t) is int and t >= 0 for t in value["ticks"]), "tick encoding")
    pairs = []
    for pair in value["registered_pairs"]:
        require(len(pair) == 2, "pair arity")
        a, b = map(key, pair)
        require(a in keys and b in keys and a[0] == b[0] and a < b, "owner/pair registration")
        pairs.append((a, b))
    hits = [tuple(map(key, pair)) for pair in value["candidates"]]
    require(len(pairs) == len(set(pairs)) and len(hits) == len(set(hits)), "duplicate pairs")
    require(set(hits) <= set(pairs), "unregistered candidate")


def scan(value, label="payload", depth=0):
    """Schema traversal only; rejected fixtures and immutable tokens are opaque."""
    require(depth <= 64, "payload nesting limit")
    if isinstance(value, float):
        require(math.isfinite(value), label + ": nonfinite JSON number")
    elif isinstance(value, list):
        for item in value:
            scan(item, label, depth + 1)
    elif isinstance(value, dict):
        require("nonfinite" not in value, label + ": nonfinite runtime value")
        if {"world", "topology", "tick", "control", "position"} <= value.keys():
            checkpoint(value)
        if {"link_position_m", "com_position_m", "joint_types"} <= value.keys():
            kinematics(value)
        if {"entry_tick", "exit_tick", "equation_residual"} <= value.keys():
            diagnostic(value)
        if {"shape_keys", "registered_pairs", "bounds"} <= value.keys():
            geometry(value)
        if {"P", "C", "model", "independent", "motor"} <= value.keys():
            dimensions(value)
        for name, item in value.items():
            if name not in (
                "topology",
                "graph_token",
                "rejected",
                "invalid_restore",
                "native_model",
                "parameters_before_zero",
            ):
                scan(item, label + "." + name, depth + 1)


def accepted(record, tick, worlds=None):
    fields(record, "tick accepted")
    require(record["tick"] == tick, "trace prefix/tick")
    require(
        record["accepted"] and all(x is True for x in record["accepted"].values()),
        "unaccepted complete row",
    )
    if worlds is not None:
        require(set(record["accepted"]) == set(worlds), "accepted owner coverage")


def paired(before, after, read=None, diag=None):
    n, k = checkpoint(after)
    checkpoint(before)
    require(
        before["world"] == after["world"] and before["topology"] == after["topology"],
        "checkpoint owner/model changed",
    )
    require(after["tick"] == before["tick"] + 1, "not a single accepted step")
    if read is not None:
        kinematics(read)
        require(read["tick"] == after["tick"], "end FK tick")
        if "full_position" in after:
            require(
                read["position"] == after["full_position"]
                and read["velocity"] == after["full_velocity"],
                "FK/full checkpoint mismatch",
            )
            require(
                read["independent_position"] == after["position"]
                and read["independent_velocity"] == after["velocity"],
                "FK/K checkpoint mismatch",
            )
    if diag is not None:
        diagnostic(diag)
        require(
            (diag["entry_tick"], diag["exit_tick"]) == (before["tick"], after["tick"]),
            "entry diagnostic tick",
        )
        require(
            diag["entry_position"] == before["position"]
            and diag["entry_velocity"] == before["velocity"],
            "entry diagnostic state",
        )
    return n, k


def check_oracle(diag, saved):
    """Use already stored independent values; do not compute M, FK, KKT or steps."""
    fields(
        saved,
        "full q v mass bias external external_full acceleration full_acceleration motor_full multipliers reaction",
    )
    for actual, expected in (
        ("mass", saved["full"]["mass"]),
        ("bias", saved["full"]["bias"]),
        ("external", saved["external_full"]),
        ("reduced_mass", saved["mass"]),
        ("reduced_bias", saved["bias"]),
        ("reduced_external", saved["external"]),
        ("acceleration", saved["acceleration"]),
        ("full_acceleration", saved["full_acceleration"]),
        ("constraint_effort_full", saved["reaction"]),
        ("row_multipliers", saved["multipliers"]),
    ):
        close(diag[actual], expected)


def physical_payload(member, records):
    for record in records.values():
        scan(record)
    group, p = member.group, member.parameter
    r = next(iter(records.values()))
    if group == "public":
        entry, r = records["a01-public-input"], records["a01-public-result"]
        require(entry["input"]["name"] == "A01" and entry["h_s"] == H, "public input")
        paired(r["before"], r["after"], r["kinematics"], r["diagnostic"])
        check_oracle(r["diagnostic"], r["oracle"])
    elif group in ("exact", "covariance"):
        require(r["input"]["name"] == p[-1], "case identity")
        if group == "covariance":
            origin = [4096, -2048, 1024] if p[0] == "large" else [0.5, -0.25, 0.75]
            require(r["input"]["model"]["offsets"][0] == origin, "installation identity")
        else:
            require(r["h_s"] == H, "exact h")
            fields(r, "independent_kkt actual_entry_oracle")
        paired(r["before"], r["after"], r["kinematics"], r["diagnostic"])
        check_oracle(r["diagnostic"], r["actual_entry_oracle"] if group == "exact" else r["oracle"])
    elif group == "window":
        require(
            r["input"]["name"] == p[0] and r["h_s"] == H and r["requested_steps"] == 40,
            "window input",
        )
        require(len(r["trace"]) == 40 and r["initial"]["tick"] == 0, "complete window")
        before = r["initial"]
        for tick, row in enumerate(r["trace"], 1):
            accepted(row, tick, ["7"])
            fields(
                row,
                "actual diagnostic kinematics oracle independent_window_s independent_window_u stats",
            )
            paired(before, row["actual"], row["kinematics"], row["diagnostic"])
            check_oracle(row["diagnostic"], row["oracle"])
            close(row["actual"]["position"], row["independent_window_s"], 2e-4)
            close(row["actual"]["velocity"], row["independent_window_u"], 2e-4)
            before = row["actual"]
        require(
            {row["stats"]["current_buffer"] for row in r["trace"]} == {0, 1}, "both state buffers"
        )
        fields(r, "final_stats")
    elif group == "reorder":
        require(r["input"]["name"] == "A02" and len(r["named_states"]) == 2, "named reorder")
        close(r["named_states"][0], r["named_states"][1])
    elif group == "chain":
        require(len(r) == 2, "chain vs flat records")
        close(r[0]["state"]["full_position"], r[1]["state"]["full_position"])
        close(r[0]["reaction"], r[1]["reaction"])
        close(r[0]["row"], [2.5, -1])
        close(r[1]["row"], [2, -1])
    elif group == "pd":
        paired(r["before"], r["after"], diag=r["diagnostic"]["7"])
        close(r["after"]["applied_effort"], [r["expected"]])
        require((r["expected"] == 1) == (p[0] == 1), "cap parameter identity")
    elif group == "small":
        paired(r["before"], r["after"])
        require(r["input"]["s"] == ([0.5] if p[0] else [0]), "wide-cap/cancellation identity")
        for actual, expected in zip(r["after"]["applied_effort"], r["original_clipped"]):
            require(
                abs(actual - expected) <= 5e-5 * max(1, abs(actual), abs(expected)),
                "original clipped effort gate",
            )
    elif group == "delay":
        require(r["delay"] == p[0], "delay identity")
        paired(r["initial"], r["first"])
        paired(r["first"], r["second"], diag=r["diagnostic"]["7"])
        require(r["first"]["history"] == ([[0, 0, 3]] if p[0] else []), "delay history")
        require(r["second"]["control"][0][0][2] == -3, "current held command")
    elif group == "budget":
        require([x["budget"] for x in r["outputs"]] == [1, 512], "budget order")
        first, second = r["outputs"]
        require(
            first["accepted"] == {"7": False} and first["after"] == first["before"],
            "one-sweep no publication",
        )
        require(first["diagnostic"] == {"7": None}, "failed scratch diagnostic")
        require(second["accepted"] == {"7": True}, "full-budget acceptance")
        paired(second["before"], second["after"], diag=second["diagnostic"]["7"])
        close(second["after"]["applied_effort"], r["exact"])
        require(r["first_row_residual"] > 0.06, "predeclared finite-budget negative")
    elif group == "domain":
        require(
            r["after"] == r["before"] and r["input"]["u"] == [p[0] / 8],
            "domain rejection retained state",
        )
    elif group == "atomic":
        require((r["failure"], r["warm_steps"]) == p, "fault/buffer identity")
        require(set(r["before"]) == set(r["after"]) == {"7", "23"}, "two-bank owners")
        require(r["uploads"] or p[0] == "step_status", "fault injection evidence")
        for world in r["after"]:
            checkpoint(r["after"][world])
        fields(r, "commands stats_before stats_after")
    elif group == "restore":
        require(r["before"]["tick"] == p[0] and r["after"] == r["before"], "same-tick restore")
        require(len(r["rejected"]) == 3, "three rejected checkpoint schemas")
    elif group == "owned":
        paired(r["before"], r["after"], r["read"], r["diagnostic"])
        require(r["before"]["control"][0] == [[0, 0, 3]], "owned held drive")
    elif group == "shapes":
        placement, angle = p
        require(
            r["input"]["name"] == "A04" and r["input"]["s"] == [angle], "material pose identity"
        )
        origin = {"base": [0, 0, 0], "cycle": [0.5, -0.25, 0.75], "large": [4096, -2048, 1024]}[
            placement
        ]
        require(r["input"]["model"]["offsets"][0] == origin, "material installation")
        require(
            r["result"]["candidates"] == [] and r["result"]["ticks"] == [0],
            "current-pose clear shape set",
        )
        ids = {11: [101], 12: [201, 202], 13: list(range(301, 309)), 14: [401, 402, 403, 404, 405]}
        for shape in r["model"]["shapes"]:
            if shape["key"]["body"] != 900:
                require(
                    shape["vertex_ids"] == ids[shape["key"]["shape"]] and shape["revision"] == 7,
                    "original material IDs/revision",
                )
        require(
            len(r["materials"]) == 4 and len(r["result"]["bounds"]) == 5, "finite material coverage"
        )
    elif group == "near":
        require(r["gap_m"] == p[0], "near/miss frozen gap")
        require(len(r["result"]["candidates"]) == (1 if p[0] < 0 else 0), "near/miss candidate")
        array(r["oracle_bounds"], 2, 3)
    elif group == "capsule":
        require(
            r["zone"] == p[0] and r["original_points"] == [[-8, 0, 0], [8, 0, 0]],
            "finite capsule identity",
        )
        array(r["oracle_world"], 2, 3)
        require(
            len(r["result"]["candidates"]) == (0 if p[0] == "beyond_end" else 1),
            "finite endpoint candidate",
        )
    elif group == "asset":
        require(r["input"]["name"] == p[0] and len(r["trace"]) == 3, "asset original source")
        require(
            "<robot" in r["source_urdf"] and len(r["original_material"]) == 3,
            "saved original URDF/materials",
        )
        for tick, row in enumerate(r["trace"], 1):
            require(
                row["state"]["tick"] == tick and row["result"]["ticks"] == [tick],
                "asset-source tick",
            )
            require(row["result"]["candidates"] == [], "asset-source clear")
    elif group in ("full", "warm"):
        matrix_payload(r, p[0], warm=group == "warm")
    elif group == "epoch":
        require(r["checkpoint"]["7"]["tick"] == p[0], "epoch warm identity")
        require(r["accepted"] == {"7": False, "23": True}, "per-world numeric rejection")
        require(
            r["final"]["7"]["tick"] == 0 and r["final"]["23"]["tick"] == p[0] + 1,
            "selected reset/spectator",
        )
    elif group == "query-failure":
        fields(r, "actual retained_result stats")
        require(
            r["actual"]["tick"] == 0 and r["retained_result"]["ticks"] == [0],
            "query failure accepted result",
        )
    elif group == "closed":
        require(
            r["device_owned_bytes"] == 0 and r["collision_plan_live_count"] == 0,
            "all resources released",
        )
    elif group == "empty":
        require(
            r["result"]["shape_keys"] == r["result"]["candidates"] == r["result"]["ticks"] == [],
            "empty result",
        )
        require(r["batch_stats"]["device_allocations"] == 0, "empty no allocation")
    elif group == "edge":
        require(r["before"]["full_position"][1] == p[0] * 4096, "exact full-coordinate boundary")
        require(
            r["entry"]["bounds"] == r["after"]["bounds"]
            and r["entry"]["ticks"] == r["after"]["ticks"],
            "rejected boundary geometry",
        )
    elif group == "loss":
        loss_payload(records, p[0])
    elif group == "mixed":
        require(set(r["before"]) == {"101", "7", "23"}, "ordinary/affine/scalar owner coverage")
        require(
            r["before"]["7"]["tick"] == p[0] and len(r["partial_upload_calls"]) == 3,
            "mixed bank fault identity",
        )
        require(len(r["trace"]) == 2, "mixed continuation")
        for index, row in enumerate(r["trace"], 1):
            require(row["actual"] == row["control"], "mixed independent control mismatch")
            require(
                all(x["tick"] == p[0] + index for x in row["actual"].values()),
                "mixed continuation tick",
            )
            require(set(row["diagnostics"]) == {"101", "7"}, "both tree diagnostics")
    else:
        raise AuditError("unhandled saved payload group " + group)


def matrix_payload(r, count, *, warm):
    owners = {str(7 + 13 * i) for i in range(count)}
    fields(r, "commands initial materials trace")
    require(
        set(r["initial"]) == set(r["commands"]) == set(r["materials"]) == owners,
        "matrix owner coverage",
    )
    inputs = r["input"] if warm else r["inputs"]
    require(set(inputs) == owners and len(r["trace"]) == 40, "matrix complete inputs/window")
    kinds = (
        ["A04"]
        if count == 1
        else ["sphere", "A01", "A03", "A05"]
        if count == 4
        else ["sphere", "slider", "rotor", "ordinary", "A02", "A03", "N9", "N16"] * 4
    )
    expected_pairs = []
    for index, kind in enumerate(kinds):
        owner = str(7 + 13 * index)
        data = inputs[owner]
        if kind == "sphere":
            require(data is None, "scalar sphere input identity")
        elif kind in ("slider", "rotor"):
            require(
                data["model"]["kinds"] == ["prismatic" if kind == "slider" else "revolute"],
                "old scalar kind",
            )
        else:
            require(
                data["name"] == ("A04" if kind == "ordinary" else kind),
                "matrix affine input identity",
            )
        if not warm:
            shape = [
                item
                for item in r["model"]["shapes"]
                if item["key"] == r["materials"][owner]["moving"]
            ]
            require(len(shape) == 1, "original model shape owner")
            expected_ids = (
                [201, 202]
                if kind in ("A03", "rotor")
                else (list(range(301, 309)) if kind in ("A02", "N16", "ordinary") else [101])
            )
            require(
                shape[0]["vertex_ids"] == expected_ids and shape[0]["revision"] == 7,
                "complete matrix original material IDs/revision",
            )
            require(
                shape[0]["radius"] == r["materials"][owner]["radius"], "actual finite skin radius"
            )
        if index % 2 == 0:
            expected_pairs.append(
                (key(r["materials"][owner]["moving"]), key(r["materials"][owner]["wall"]))
            )
    start_tick = 3 if warm else 1
    previous = r["actual_warm_entry"] if warm else r["initial"]
    if warm:
        require(all(x["tick"] == 2 for x in previous.values()), "two accepted bootstrap steps")
        fields(r, "actual_warm_end batch_before plan_before final_stats")
    for tick, row in enumerate(r["trace"], start_tick):
        accepted(row, tick, owners)
        fields(row, "result batch_stats plan_stats")
        result = row["result"]
        require(
            set(map(str, result["world_keys"])) == owners and result["ticks"] == [tick] * count,
            "all world ticks",
        )
        require(
            len(result["shape_keys"]) == 2 * count and len(result["registered_pairs"]) == count,
            "all finite shapes/pairs",
        )
        require(
            set(tuple(map(key, pair)) for pair in result["candidates"]) == set(expected_pairs),
            "same-world near/far candidates",
        )
        if warm:
            require(
                not ({"actual", "kinematics", "diagnostic", "independent_affine_s_u"} & row.keys()),
                "warm row falsely includes state readback",
            )
            for name in r["batch_before"]:
                if name in ("device_allocations", "tree_diagnostic_reads") or (
                    name.endswith("_downloaded_bytes")
                    and any(t in name for t in ("state", "snapshot", "kinematic"))
                ):
                    require(
                        row["batch_stats"][name] == r["batch_before"][name],
                        "warm forbidden counter " + name,
                    )
            require(
                row["plan_stats"]["device_allocations"] == r["plan_before"]["device_allocations"],
                "warm plan allocation",
            )
            require(
                row["plan_stats"]["dynamic_envelope_uploaded_bytes"] == 0,
                "warm HOST envelope upload",
            )
        else:
            fields(row, "actual kinematics diagnostic independent_affine_s_u")
            require(set(row["actual"]) == owners, "complete state coverage")
            affine = {
                owner
                for owner, data in inputs.items()
                if data is not None and "P" in data and "full_position" in row["actual"][owner]
            }
            require(
                set(row["kinematics"])
                == set(row["diagnostic"])
                == set(row["independent_affine_s_u"])
                == affine,
                "complete affine FK/diagnostic coverage",
            )
            for owner in owners:
                require(row["actual"][owner]["tick"] == tick, "actual state tick")
            for owner in affine:
                paired(
                    previous[owner],
                    row["actual"][owner],
                    row["kinematics"][owner],
                    row["diagnostic"][owner],
                )
                s, u = row["independent_affine_s_u"][owner]
                close(row["actual"][owner]["position"], s, 2e-4)
                close(row["actual"][owner]["velocity"], u, 2e-4)
            previous = row["actual"]
    require(
        {row["batch_stats"]["current_buffer"] for row in r["trace"]} == {0, 1},
        "matrix both buffers",
    )
    if warm:
        require(
            set(r["actual_warm_end"]) == owners
            and all(x["tick"] == 42 for x in r["actual_warm_end"].values()),
            "warm accepted end readback",
        )
        final = r["trace"][-1]["plan_stats"]
        require(final["state_buffer_directions"] == [0, 1], "warm bounds both directions")
        require(
            final["copied_to_host_bytes"] > r["plan_before"]["copied_to_host_bytes"],
            "honest warm result readback",
        )


def loss_payload(records, warm):
    initial, r = records["permanent-device-loss-input"], records["permanent-device-loss-result"]
    require(initial["warm_steps"] == r["warm_steps"] == warm, "permanent loss identity")
    for name, value in initial.items():
        require(r[name] == value, "permanent loss pre-failure record changed " + name)
    require(r["sync_calls"] == ["restore", "recovery"], "permanent sync/recovery evidence")
    require(r["restored_checkpoint"] == r["before_checkpoint"], "new owner full checkpoint restore")
    require(r["restored_tree_kinematics"] == r["before_tree_kinematics"], "restored full FK")
    require(
        r["restored_scalar_kinematics"] == r["before_scalar_kinematics"], "restored scalar state"
    )
    for name in ("current_buffer", "state_epoch"):
        require(r["failed_stats"][name] == r["before_stats"][name], "failure published bank/epoch")
    for name in (
        "current_buffer",
        "state_epoch",
        "launch_attempts",
        "synchronizations",
        "recovery_sync_attempts",
    ):
        require(
            r["failed_stats"][name] == r["after_blocked_stats"][name],
            "poisoned operation side effect",
        )
    operations = {
        "snapshot",
        "tree_kinematics",
        "tree_diagnostics",
        "scalar_kinematics",
        "summary",
        "step",
        "restore",
        "set_controls",
        "reset",
    }
    require(set(r["blocked_operations"]) == operations, "all unavailable public paths")
    require(
        all(
            "rebuild from a saved checkpoint" in x["message"]
            for x in r["blocked_operations"].values()
        ),
        "poison must persist",
    )
    require(
        r["closed_stats"]["device_owned_bytes"] == 0 and len(r["continuation"]) == 2,
        "close then independent continuation",
    )
    for index, row in enumerate(r["continuation"], warm + 2):
        require(row["rebuilt"] == row["independent_control"], "new owner/control replay mismatch")
        require(
            all(x["tick"] == index for x in row["rebuilt"].values()), "rebuild continuation tick"
        )
        paired(row["before"]["7"], row["rebuilt"]["7"], row["kinematics"], row["diagnostic"])
        check_oracle(row["diagnostic"], row["oracle"])


def strict_object(pairs):
    result = {}
    for name, value in pairs:
        require(name not in result, "duplicate JSON key " + name)
        result[name] = value
    return result


def normalized(path):
    return str(path).replace("\\", "/").casefold()


class Audit:
    def __init__(self, root):
        self.root = root.resolve(strict=True)
        self.files = {}
        self.payload_bytes = 0
        self.report = {
            "scope": "C5 saved evidence only; no producer, solver or oracle execution",
            "status": "incomplete",
            "jobs": [],
            "errors": [],
            "limits": {
                "compressed_file_bytes": MAX_COMPRESSED,
                "uncompressed_payload_bytes": MAX_PAYLOAD,
                "total_uncompressed_payload_bytes": MAX_TOTAL_PAYLOAD,
                "files": MAX_TOTAL_FILES,
            },
            "saving_limits": [
                "HOST 13 and collision identity 6 per mode have no _save; only source-bound pytest receipts certify those nodes.",
                "Named reorder saves named q/v/drive arrays, not complete checkpoints or FK.",
                "PD, delay, chain and several geometry/fault controls save their authored subsets; missing fields are not invented.",
                "Warm frames contain actual bounds/ticks/acceptance/counters only; state is saved at warm entry and end.",
                "Numerical comparisons below use already saved oracle values; no mechanics/FK/material reconstruction is rerun.",
                "Readonly ownership and injected callback behavior not encoded as data remain source-bound test assertions.",
                "Job 276 contains 15 non-C5 tests; their physics/payloads are outside this audit.",
                "A03-flattened is a separate native graph; native original-chain rejection is retained.",
            ],
        }

    def read(self, path, limit=MAX_METADATA):
        path = path.resolve(strict=True)
        require(path.is_relative_to(self.root), "file escapes evidence root")
        require(
            path.is_file() and path.stat().st_size <= limit, "file size/type limit: " + str(path)
        )
        raw = path.read_bytes()
        require(len(raw) <= limit, "file grew beyond bound")
        relative = path.relative_to(self.root).as_posix()
        digest = hashlib.sha256(raw).hexdigest()
        if relative in self.files:
            require(
                self.files[relative]["sha256"] == digest,
                "evidence changed during audit: " + relative,
            )
        self.files[relative] = {"sha256": digest, "bytes": len(raw)}
        require(len(self.files) <= MAX_TOTAL_FILES, "file count bound")
        return raw

    def text(self, path):
        return self.read(path).decode("utf-8-sig")

    def json(self, path, *, compressed=False):
        raw = self.read(path, MAX_COMPRESSED if compressed else MAX_METADATA)
        if compressed:
            with gzip.GzipFile(fileobj=io.BytesIO(raw), mode="rb") as stream:
                raw = stream.read(MAX_PAYLOAD + 1)
            require(len(raw) <= MAX_PAYLOAD, "uncompressed per-payload bound")
            self.payload_bytes += len(raw)
            require(self.payload_bytes <= MAX_TOTAL_PAYLOAD, "uncompressed cumulative bound")
            self.files[path.resolve().relative_to(self.root).as_posix()]["uncompressed_bytes"] = (
                len(raw)
            )
        return json.loads(
            raw.decode("utf-8-sig"),
            object_pairs_hook=strict_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                AuditError("non-JSON numeric constant " + token)
            ),
        )

    def receipt(self, name, raw_exit, mode):
        directory = self.root / name
        config = self.json(self.root / (name + ".json"))
        start, final = self.json(directory / "start.json"), self.json(directory / "final.json")
        require(start["task"] == final["task"] == name, "receipt task identity")
        for field in ("owner", "exe", "args"):
            require(start[field] == final[field] == config[field], "receipt/config " + field)
        require(normalized(start["cwd"]) == normalized(config["workdir"]), "receipt cwd")
        require(start["head"] == final["head"] and bool(start["head"]), "source HEAD")
        require(start["sources"] == final["sources"], "bound source manifest changed")
        require(
            final["raw_exit"] == raw_exit
            and int(self.text(directory / "RAW_EXIT.txt").strip()) == raw_exit,
            "original raw exit mismatch",
        )
        before, after, sources = {}, {}, {}
        for item in start["sources"]:
            identity = normalized(item["path"])
            require(identity not in before, "duplicate source binding")
            before[identity] = item["sha256"].lower()
            snapshot = directory / item["snapshot"]
            require(snapshot.parent == directory, "source snapshot must be job local")
            contents = self.read(snapshot)
            require(
                hashlib.sha256(contents).hexdigest() == before[identity], "source snapshot digest"
            )
            sources[identity] = (before[identity], contents.decode("utf-8-sig"))
        for item in final["sources_after"]:
            identity = normalized(item["path"])
            require(identity not in after, "duplicate after source")
            after[identity] = item["sha256"].lower()
        require(before == after, "source before/after mismatch")
        require(
            set(before) == {normalized(path) for path in config["files"]}, "config source coverage"
        )
        args = config["args"]
        option = "--device" if mode in ("cpu", "cuda:0") else "--tyx-mode"
        if mode != "host":
            require(
                option in args and args[args.index(option) + 1] == mode, "requested device/mode"
            )
        else:
            require("not tyx" in args, "HOST marker")
        stdout, stderr = self.text(directory / "stdout.log"), self.text(directory / "stderr.log")
        return (
            directory,
            args,
            sources,
            stdout,
            {
                "job": name,
                "raw_exit": raw_exit,
                "head": start["head"],
                "mode": mode,
                "source_before_after_equal": True,
                "source_files": len(before),
                "stderr_bytes": len(stderr.encode("utf-8")),
            },
        )

    @staticmethod
    def source(sources, suffix):
        rows = [
            (digest, text)
            for path, (digest, text) in sources.items()
            if path.endswith("/" + suffix.casefold())
        ]
        require(len(rows) == 1, "missing/ambiguous bound source " + suffix)
        return rows[0]

    def pytest_job(
        self, name, members, *, passed, deselected=0, failed=None, mode="interpret", extra=0
    ):
        raw_exit = 1 if failed is not None else 0
        directory, args, sources, stdout, row = self.receipt(name, raw_exit, mode)
        summary = [line for line in stdout.splitlines() if re.search(r"\d+ passed", line)]
        require(bool(summary), "pytest result summary absent")
        summary = summary[-1]
        for tag, expected in (
            ("passed", passed),
            ("failed", int(failed is not None)),
            ("deselected", deselected),
        ):
            match = re.search(r"(\d+) " + tag, summary)
            require((int(match.group(1)) if match else 0) == expected, "pytest " + tag + " count")
        require(passed == len(members) + extra, "logical test vs receipt count")
        if failed is not None:
            require("FAILED " + failed.node in stdout, "first failure identity")
        excluded = {args[i + 1] for i, value in enumerate(args[:-1]) if value == "--deselect"}
        steps, collisions, losses, hosts = membership()
        expected_sources = (
            {STEP}
            if mode == "host"
            else {STEP, COLLISION, LOSS}
            if mode == "cuda"
            else {
                LOSS,
                "tests/test_persistent_grasp_holdouts.py",
                "tests/test_sampled_payload_material_events.py",
            }
            if extra
            else {STEP, COLLISION}
        )
        require(
            {value for value in args if value.endswith(".py")} == expected_sources,
            "pytest exact entry points",
        )
        universe = (
            hosts
            if mode == "host"
            else losses
            if extra
            else steps + collisions + (losses if mode == "cuda" else [])
        )
        selected = [
            member
            for member in universe
            if member.node not in excluded
            and member.source + "::" + member.function not in excluded
        ]
        require(selected[: len(members)] == members, "selected authored success prefix")
        if failed is not None:
            require(
                len(selected) > len(members) and selected[len(members)] == failed,
                "selected first failure prefix",
            )
        else:
            require(len(selected) == len(members), "complete run has unsatisfied logical members")
        for member in members + ([failed] if failed is not None else []):
            require(
                member.node not in excluded
                and member.source + "::" + member.function not in excluded,
                "required member deselected",
            )
            _, source = self.source(sources, member.source)
            require("def " + member.function + "(" in source, "member missing in bound producer")
        ordinals, expected_files, successes = Counter(), set(), []
        for member in members:
            # pytest truncates the original function name to 30 chars for tmp_path.
            ordinal = ordinals[member.function]
            ordinals[member.function] += 1
            test_dir = directory / "basetemp" / (member.function[:30] + str(ordinal))
            require(test_dir.is_dir(), "missing original test directory: " + str(test_dir))
            records = {}
            for filename in member.files:
                path = test_dir / (filename + ".json.gz")
                expected_files.add(path.relative_to(directory).as_posix())
                records[filename] = self.json(path, compressed=True)
            if records:
                physical_payload(member, records)
            successes.append(
                {
                    "logical_id": member.identity,
                    "node": member.node,
                    "payloads": [
                        str((test_dir / (f + ".json.gz")).relative_to(directory)).replace("\\", "/")
                        for f in member.files
                    ],
                    "evidence": "saved-schema-and-bound-receipt"
                    if records
                    else "bound-test-receipt-only",
                }
            )
        if failed is not None:
            failed_dir = (
                directory / "basetemp" / (failed.function[:30] + str(ordinals[failed.function]))
            )
            require(failed_dir.is_dir(), "first failure directory absent")
            require(
                not list(failed_dir.glob("*.json.gz")),
                "failure unexpectedly saved a completion payload",
            )
        found = set()
        directories = list((directory / "basetemp").iterdir())
        require(len(directories) <= 256, "bounded pytest directory inventory")
        for test_dir in directories:
            if test_dir.is_dir() and not test_dir.is_symlink():
                for path in test_dir.glob("*.json.gz"):
                    found.add(path.relative_to(directory).as_posix())
        require(
            found == expected_files,
            "payload membership mismatch; missing="
            + str(sorted(expected_files - found))
            + "; extra="
            + str(sorted(found - expected_files)),
        )
        row.update(
            passed=passed,
            c5_passed=len(members),
            excluded_non_c5=extra,
            failed_node=None if failed is None else failed.node,
            prefix_complete=failed is None,
            payload_count=len(found),
            members=successes,
            summary=summary,
        )
        self.report["jobs"].append(row)

    def native_job(self, name, device):
        directory, args, sources, _, row = self.receipt(name, 0, device)
        require(REFERENCE in args, "native reference entry point")
        outputs = directory / "outputs"
        summary = self.json(outputs / "summary.json")
        require(
            summary["device"] == device
            and summary["versions"]["newton"] == "1.7.0"
            and summary["versions"]["warp-lang"] == "1.17.0",
            "native pinned version/device",
        )
        require(isinstance(summary["versions"].get("numpy"), str), "native numpy version")
        expected_sources = {
            "_src/sim/state.py",
            "_src/sim/articulation.py",
            "_src/sim/joint_mimic.py",
            "_src/sim/builder.py",
            "_src/utils/import_urdf.py",
            "_src/solvers/featherstone/kernels.py",
            "_src/solvers/featherstone/solver_featherstone.py",
            "driver",
        }
        require(set(summary["sources"]) == expected_sources, "native source coverage")
        for relative, info in summary["sources"].items():
            actual = (
                outputs
                / "source_snapshot"
                / ("resident_affine_reference.py" if relative == "driver" else relative)
            )
            digest = hashlib.sha256(self.read(actual)).hexdigest()
            require(digest == info["sha256"].lower(), "native source snapshot digest")
            source_digest, _ = self.source(sources, REFERENCE if relative == "driver" else relative)
            require(digest == source_digest, "native executed source vs receipt")
        chain = summary["original_chain"]
        require(
            chain["accepted"] is False
            and chain["expected_rejection"] is True
            and "mimic" in chain["error"].lower(),
            "native original chain boundary",
        )
        original = self.read(outputs / "original-chain.urdf")
        require(
            hashlib.sha256(original).hexdigest() == chain["source"]["sha256"]
            and original.decode("utf-8") == chain["source"]["text"],
            "original chain source",
        )
        dimensions(chain["source"]["input"])
        require([x["case"] for x in summary["results"]] == list(NATIVE), "native membership/order")
        require(
            {x.name for x in outputs.glob("*.json.gz")} == {x + ".json.gz" for x in NATIVE},
            "native payload membership",
        )
        cases = []
        for result in summary["results"]:
            label = result["case"]
            require(
                result["passed"] is True
                and result["steps"] == 40
                and result["execution_failure"] is None,
                "native summary acceptance",
            )
            require(
                normalized(result["path"]).endswith(
                    "/" + name + "/outputs/" + label.casefold() + ".json.gz"
                ),
                "native output path identity",
            )
            report = self.json(outputs / (label + ".json.gz"), compressed=True)
            native_payload(report, label, device)
            urdf = self.read(outputs / (label + ".urdf"))
            require(
                hashlib.sha256(urdf).hexdigest() == report["urdf"]["sha256"]
                and urdf.decode("utf-8") == report["urdf"]["text"],
                "native actual model source",
            )
            cases.append(
                {
                    "case": label,
                    "actual_device": report["native_model"]["device"],
                    "steps": len(report["trace"]),
                    "checks": sum(len(x["checks"]) for x in [report, *report["trace"]]),
                }
            )
        row.update(
            versions=summary["versions"], original_chain="rejected", cases=cases, payload_count=5
        )
        self.report["jobs"].append(row)


def native_checks(record, expected_names, *, per_step=False):
    checks = record["checks"]
    require(
        len(checks) == len(expected_names) and {x["name"] for x in checks} == set(expected_names),
        "native complete check membership",
    )
    for check in checks:
        require(
            check["passed"] is True
            and number(check["max_abs_error"])
            and check["max_abs_error"] >= 0,
            "native saved gate failed/missing",
        )
        tolerance = (
            3e-4
            if check["name"] == "derived_reaction"
            else (2e-4 if check["name"] in ("window_position", "window_velocity") else 5e-6)
        )
        require(
            check["atol"] == tolerance and check["rtol"] == 5e-5, "native predeclared tolerance"
        )
    if per_step:
        require(
            number(record["kinetic_energy_j"]) and record["kinetic_energy_j"] >= 0,
            "native kinetic receipt",
        )


def native_state(record, n, authored_bodies, native_bodies):
    for name in ("q", "v"):
        array(record[name], n)
    for name in ("joint_q", "joint_qd"):
        require(len(record[name]) >= n, "native full joint state")
        array(record[name], len(record[name]))
    array(record["body_q"], native_bodies, 7)
    for name in ("body_qd", "body_f"):
        array(record[name], native_bodies, 6)
    for name in (
        "authored_position_m",
        "com_world_m",
        "com_velocity_world_m_s",
        "angular_velocity_world_rad_s",
    ):
        array(record[name], authored_bodies, 3)
    array(record["authored_rotation"], authored_bodies, 3, 3)


def native_payload(report, label, device):
    scan(report)
    require(
        report["case"] == label and report["requested_device"] == device,
        "native case/device identity",
    )
    require(
        report["h_s"] == H and report["steps"] == 40 and len(report["trace"]) == 40,
        "native complete 40-step window",
    )
    require(
        report["execution_failure"] is None and report["passed"] is True, "native actual completion"
    )
    data, meta = report["input"], report["native_model"]
    n, _ = dimensions(data)
    require(data["name"] == ("A03" if label == "A03-flattened" else label), "native original input")
    fields(report, "parameters_before_zero urdf conventions")
    require(
        report["conventions"]["body_qd"] == "world COM linear first, world angular last"
        and report["conventions"]["body_f"] == "world force at COM first, world torque at COM last"
        and report["conventions"]["contacts"] is None,
        "native reference-point/load convention",
    )
    zeros = (
        "joint_armature",
        "joint_damping",
        "joint_friction",
        "joint_target_ke",
        "joint_target_kd",
        "joint_limit_ke",
        "joint_limit_kd",
    )
    names = (
        "body_mass body_com body_inertia joint_type joint_parent joint_child joint_q_start joint_qd_start "
        "joint_X_p joint_X_c joint_axis joint_dof_dim joint_articulation articulation_start articulation_end "
        "joint_mimic_joint joint_mimic_coeffs joint_limit_lower joint_limit_upper joint_effort_limit "
        "joint_velocity_limit shape_type shape_body shape_transform shape_scale gravity body_label joint_label shape_label device"
    ).split()
    fields(meta, (*names, *zeros), "native actual metadata")
    require(meta["device"] == device, "native no fallback")
    for name in zeros:
        require(name in report["parameters_before_zero"], "prezero parameter provenance")
        require(
            isinstance(meta[name], list) and all(value == 0 for value in meta[name]),
            "native unwanted passive/limit/target load",
        )
    nb, ns, nj = len(meta["body_label"]), len(meta["shape_label"]), len(meta["joint_label"])
    authored = len(data["model"]["mass"])
    require(nb >= authored and ns == authored and nj >= n, "native original bodies/shapes/joints")
    array(meta["body_mass"], nb)
    array(meta["body_com"], nb, 3)
    array(meta["body_inertia"], nb, 3, 3)
    array(meta["shape_transform"], ns, 7)
    array(meta["shape_scale"], ns, 3)
    require(
        len(meta["shape_type"]) == len(meta["shape_body"]) == ns,
        "native shape metadata completeness",
    )
    for name in ("joint_type", "joint_parent", "joint_child", "joint_mimic_joint"):
        require(len(meta[name]) == nj, "native joint metadata completeness")
    array(meta["joint_mimic_coeffs"], nj, 2)
    top_checks = [
        f"body{i}_{field}" for i in range(authored) for field in ("mass", "com", "inertia")
    ]
    top_checks += [f"joint{i}_mimic_coefficients" for i in range(n)]
    native_checks(report, top_checks)
    step_checks = (
        "full_native_mass",
        "same_entry_velocity",
        "same_entry_position",
        "full_affine_position",
        "full_affine_velocity",
        "window_position",
        "window_velocity",
        "derived_reaction",
        "authored_position_m",
        "authored_rotation",
        "com_world_m",
        "com_velocity_world_m_s",
        "angular_velocity_world_rad_s",
    )
    previous = None
    for tick, row in enumerate(report["trace"], 1):
        require(
            row["step"] == tick
            and row["entry_time_s"] == (tick - 1) * H
            and row["commit_time_s"] == tick * H,
            "native same-step sample time",
        )
        fields(
            row,
            "entry actual applied_body_f applied_joint_f native_full_mass native_raw_mass oracle "
            "oracle_window_s oracle_window_u derived_full_reaction derived_reaction_reference native_H native_H_mimic",
        )
        native_state(row["entry"], n, authored, nb)
        native_state(row["actual"], n, authored, nb)
        if previous is not None:
            for name in ("joint_q", "joint_qd", "body_q", "body_qd", "q", "v"):
                require(
                    row["entry"][name] == previous[name], "native continuous own-state trajectory"
                )
        previous = row["actual"]
        array(row["applied_body_f"], nb, 6)
        array(row["applied_joint_f"], len(row["entry"]["joint_qd"]))
        array(row["native_full_mass"], n, n)
        array(row["derived_full_reaction"], n)
        require(
            row["derived_reaction_reference"] == "KKT at actual input; not native sensor",
            "derived/native distinction",
        )
        native_checks(row, step_checks, per_step=True)


def run(audit):
    step, collision, loss, host = membership()
    jobs = (
        ("264-c5-host", host, dict(passed=13, deselected=41, mode="host")),
        ("265-c5-interpret", step[:7], dict(passed=7, deselected=13, failed=step[7])),
        ("266-c5-interpret-remaining", step[7:23], dict(passed=16, deselected=20, failed=step[23])),
        ("267-c5-interpret-later", step[23:25], dict(passed=2, deselected=36, failed=step[25])),
        ("268-c5-interpret-rest", step[25:36], dict(passed=11, deselected=38, failed=step[36])),
        ("269-c5-interpret-tail", step[36:] + collision, dict(passed=49, deselected=49)),
        ("276-c5-grasp-adjacent", loss, dict(passed=19, extra=15)),
        ("277-c5-cuda", step + collision + loss, dict(passed=89, deselected=13, mode="cuda")),
    )
    audit.report["predeclared_membership"] = {
        "host_tests_without_payload": [member.node for member in host],
        "each_device_logical_tests": 89,
        "each_device_payload_files": 86,
        "device_tests_without_payload": [member.node for member in collision if not member.files],
        "interpret_success_partition": [7, 16, 2, 11, 49, 4],
        "interpret_payload_partition": [8, 16, 2, 11, 43, 6],
        "native_cases_each_device": list(NATIVE),
        "native_steps_each_case": 40,
        "first_failures": [
            {
                "job": "265-c5-interpret",
                "node": step[7].node,
                "reason": "A05T integer v failed the existing state dtype guard",
            },
            {
                "job": "266-c5-interpret-remaining",
                "node": step[23].node,
                "reason": "test abs(tuple), before PD payload save",
            },
            {
                "job": "267-c5-interpret-later",
                "node": step[25].node,
                "reason": "integer fixture motor truncated 1/128 to zero",
            },
            {
                "job": "268-c5-interpret-rest",
                "node": step[36].node,
                "reason": "always-throw sync mock also defeated recovery; separate permanent-loss tests retained",
            },
        ],
    }
    # This list itself is checked, not reconstructed from whichever jobs exist.
    interpreted = step[:7] + step[7:23] + step[23:25] + step[25:36] + step[36:] + collision + loss
    require(
        Counter(x.identity for x in interpreted)
        == Counter(x.identity for x in step + collision + loss),
        "logical union duplicate/omission",
    )
    for name, members, options in jobs:
        try:
            audit.pytest_job(name, members, **options)
        except (AuditError, KeyError, TypeError, ValueError, OSError, EOFError) as error:
            audit.report["errors"].append(
                {"job": name, "type": type(error).__name__, "message": str(error)}
            )
    for name, device in (("278-c5-newton-cpu", "cpu"), ("279-c5-newton-cuda", "cuda:0")):
        try:
            audit.native_job(name, device)
        except (AuditError, KeyError, TypeError, ValueError, OSError, EOFError) as error:
            audit.report["errors"].append(
                {"job": name, "type": type(error).__name__, "message": str(error)}
            )
    audit.report["status"] = (
        "saved_evidence_audited"
        if not audit.report["errors"] and len(audit.report["jobs"]) == 10
        else "audit_failed"
    )
    audit.report["files"] = audit.files
    audit.report["uncompressed_payload_bytes_read"] = audit.payload_bytes
    return audit.report["status"] == "saved_evidence_audited"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, required=True, help="new JSON audit receipt; never overwrites"
    )
    args = parser.parse_args()
    require(
        not args.output.exists() and args.output.parent.is_dir(),
        "output must be new under an existing directory",
    )
    audit = Audit(args.evidence_root)
    audit.report["audit_source"] = {
        "path": str(Path(__file__).resolve()),
        "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    audit.report["evidence_root"] = str(audit.root)
    passed = run(audit)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(audit.report, stream, indent=2, allow_nan=False)
    print(
        json.dumps(
            {
                "status": audit.report["status"],
                "jobs": len(audit.report["jobs"]),
                "files": len(audit.files),
                "errors": audit.report["errors"],
                "output": str(args.output),
            },
            allow_nan=False,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
