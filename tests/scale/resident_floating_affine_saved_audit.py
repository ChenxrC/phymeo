"""C11 saved-only evidence observer; AUTHORING_UNEXECUTED.

Only stdlib file/JSON/gzip/hash inspection. No producer import, subprocess,
collection, geometry evaluation, dynamics, matrix solve or device access.
"""

import argparse
import gzip
import hashlib
import io
import json
import math
import re
import struct
import traceback
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import url2pathname
from zipfile import ZipFile

STEP = "tests/scale/test_resident_floating_affine_step.py"
CONTACT = "tests/scale/test_resident_floating_affine_contacts.py"
TRANSACTIONS = "tests/scale/test_resident_floating_affine_transactions.py"
REFERENCE = "tests/reference_newton/robot/resident_floating_affine_reference.py"
NATIVE = "tests/reference_newton/robot/resident_floating_affine_native.py"
WHEEL = "tests/scale/resident_floating_affine_wheel_consumer.py"
WORLDS = (503, 91, 1207, 6)
MAX_COMPRESSED, MAX_PAYLOAD = 64 << 20, 256 << 20
MAX_TOTAL, MAX_METADATA, MAX_FILES, MAX_DEPTH = 2 << 30, 16 << 20, 8192, 96
ATOL, RTOL, WINDOW_ATOL = 5e-6, 5e-5, 2e-4


@dataclass(frozen=True)
class Member:
    source: str
    function: str
    parameter: str
    family: str
    files: tuple
    steps: int = 0
    h: float = 1 / 1024
    replay: bool = False
    optional: bool = False

    @property
    def node(self):
        suffix = "[" + self.parameter + "]" if self.parameter else ""
        return self.source + "::" + self.function + suffix


def membership():
    """Authored collection order, fixed before reading ANY saved output."""
    rows = []

    def add(
        source, function, variants, family, names, steps=0, h=1 / 1024, replay=False, optional=False
    ):
        for parameter, files in zip(variants, names, strict=True):
            rows.append(
                Member(
                    source, function, parameter, family, tuple(files), steps, h, replay, optional
                )
            )

    add(
        STEP,
        "test_a01_public_affine_floating_root_reaction",
        ("",),
        "public",
        (("a01-public.json.gz",),),
        1,
    )
    add(
        STEP,
        "test_a01_cycle_translation_preserves_full_response",
        ("cycle",),
        "step",
        (("a01-cycle.json.gz",),),
        4,
    )
    add(
        STEP,
        "test_a03_original_graph_rows_and_offset",
        ("fan", "chain"),
        "step",
        tuple(("a03-" + x + ".json.gz",) for x in ("fan", "chain")),
        4,
    )
    variants = ("p0", "p1", "p2", "p1-cycle")
    add(
        STEP,
        "test_a04_raw_parts_rotating_tool_response",
        variants,
        "step",
        tuple(("a04-" + x + ".json.gz",) for x in variants),
        1,
        1 / 4096,
    )
    add(
        STEP,
        "test_a04_rotating_full_reduced_checkpoint_replay",
        ("p1",),
        "step",
        (("a04-p1-window.json.gz",),),
        40,
        1 / 4096,
        True,
    )
    variants = ("d0-c1p5", "d1-c1p5", "d0-c0p25", "d1-c0p25")
    add(
        STEP,
        "test_a01_pd_delay_and_current_com_load",
        variants,
        "pd",
        tuple(("pd-" + x + ".json.gz",) for x in variants),
        4,
        1 / 128,
    )
    variants = ("n16-k1", "n16-k15")
    add(
        STEP,
        "test_full_body_and_reduced_motor_counts",
        variants,
        "counts",
        tuple(("counts-" + x + ".json.gz",) for x in variants),
        1,
    )
    variants = ("forward", "reverse")
    add(
        CONTACT,
        "test_a02_follower_surface_load_projects_once",
        variants,
        "contact",
        tuple(("a02-" + x + ".json.gz",) for x in variants),
        2,
    )
    variants = ("forward", "reverse", "cycle")
    add(
        CONTACT,
        "test_a05_same_tree_bilateral_response",
        variants,
        "contact",
        tuple(("a05-" + x + ".json.gz",) for x in variants),
        2,
    )
    for category, function, steps, replay in (
        ("core", "test_w1_w4_affine_core_same_input", 4, False),
        ("contact", "test_w1_w4_affine_contact_replay", 40, True),
    ):
        files = tuple(category + "/w1-" + str(w) + "/trace.json.gz" for w in WORLDS) + (
            category + "/w4/trace.json.gz",
        )
        add(TRANSACTIONS, function, (category,), "mixed", (files,), steps, replay=replay)
    add(
        TRANSACTIONS,
        "test_w32_bounded_affine_contact_batch",
        ("contact",),
        "mixed",
        (("contact/w32/trace.json.gz",),),
        4,
        optional=True,
    )
    variants = (
        "fixed_asset",
        "full_width",
        "joint_position",
        "joint_velocity",
        "nonfinite_gain",
        "negative_cap",
        "model_token",
        "graph_token",
    )
    add(
        TRANSACTIONS,
        "test_x01_affine_spec_and_source_preflight",
        variants,
        "nosave",
        ((),) * len(variants),
    )
    variants = ("large_anchor", "full_q_mismatch", "full_v_mismatch")
    add(
        TRANSACTIONS,
        "test_x02_local_origin_full_reduced_restore",
        variants,
        "restore",
        tuple(("restore-" + x + ".json.gz",) for x in variants),
    )
    variants = ("bank0", "bank1")
    add(
        TRANSACTIONS,
        "test_x03_contact_capacity_preserves_both_banks",
        variants,
        "capacity",
        tuple(("capacity-" + x + ".json.gz",) for x in variants),
    )
    variants = tuple(
        phase + "-b" + str(bank) for phase in ("upload", "candidate", "receipt") for bank in (0, 1)
    )
    add(
        TRANSACTIONS,
        "test_x04_late_failure_preserves_graph_state_and_history",
        variants,
        "late",
        tuple(("late-" + x + ".json.gz",) for x in variants),
    )
    variants = ("stale", "close_rebuild")
    add(
        TRANSACTIONS,
        "test_x05_stale_restore_and_closed_rebuild",
        variants,
        "owner",
        tuple(("owner-" + x + ".json.gz",) for x in variants),
    )
    variants = (
        "body17-n16",
        "body16-n15",
        "drive_full_n",
        "external_joint_effort",
        "wrong_reduced_width",
        "cross_independent_tree",
    )
    add(
        TRANSACTIONS,
        "test_x06_body_motor_and_domain_preflight",
        variants,
        "nosave",
        ((),) * len(variants),
    )
    variants = ("full_nan", "reduced_nan", "stale_tick")
    add(
        TRANSACTIONS,
        "test_x07_candidate_diagnostics_reject_corrupted_readback",
        variants,
        "diagnostic",
        tuple(("diagnostic-" + x + ".json.gz",) for x in variants),
    )
    return tuple(rows)


class AuditError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise AuditError(message)


def fields(value, names, label):
    require(isinstance(value, dict), label + " must be an object")
    require(
        set(names) <= value.keys(), label + " missing " + str(sorted(set(names) - value.keys()))
    )


def number(value):
    require(type(value) in (int, float) and math.isfinite(value), "finite saved number required")
    return value


def array(value, *shape):
    if not shape:
        number(value)
        return
    require(isinstance(value, list) and len(value) == shape[0], "saved shape " + str(shape))
    for item in value:
        array(item, *shape[1:])


def exact(actual, expected, label):
    # Preserve every value, including -0.0. Object ordering is not physical.
    a = json.dumps(actual, sort_keys=True, separators=(",", ":"), allow_nan=False)
    b = json.dumps(expected, sort_keys=True, separators=(",", ":"), allow_nan=False)
    require(a == b, label + " differs")


def same(actual, expected, label):
    require(actual == expected, label + ": " + str(actual)[:160] + " != " + str(expected)[:160])


def close(actual, expected, label, *, atol=ATOL, rtol=RTOL):
    """Compare saved actual/independent expected; generate no physical oracle."""
    if isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), label + " shape")
        for a, b in zip(actual, expected, strict=True):
            close(a, b, label, atol=atol, rtol=rtol)
    else:
        number(actual)
        number(expected)
        require(abs(actual - expected) <= atol + rtol * abs(expected), label + " saved discrepancy")


def scan(value, *, tags=False, depth=0):
    require(depth <= MAX_DEPTH, "JSON nesting bound")
    if isinstance(value, dict):
        if set(value) == {"nonfinite"}:
            require(
                tags and value["nonfinite"] in ("nan", "inf", "-inf"), "unexpected nonfinite tag"
            )
        else:
            for item in value.values():
                scan(item, tags=tags, depth=depth + 1)
    elif isinstance(value, list):
        for item in value:
            scan(item, tags=tags, depth=depth + 1)
    elif isinstance(value, float):
        require(math.isfinite(value), "bare nonfinite JSON")
    else:
        require(value is None or type(value) in (str, int, bool), "unsupported saved type")


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def hex_sha(value):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{64}", value), "explicit SHA256")
    return value.lower()


def normalized(value):
    return str(value).replace("\\", "/").casefold()


def strict_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key " + key)
        result[key] = value
    return result


def strict_load(raw):
    def invalid(value):
        raise AuditError("invalid JSON constant " + value)

    return json.loads(
        raw.decode("utf-8-sig"), object_pairs_hook=strict_object, parse_constant=invalid
    )


def checkpoint(value):
    fields(
        value,
        (
            "world",
            "topology",
            "tick",
            "position",
            "velocity",
            "applied_effort",
            "history",
            "control",
        ),
        "checkpoint",
    )
    require(
        type(value["world"]) is int and type(value["tick"]) is int and value["tick"] >= 0,
        "world/tick",
    )
    floating, affine = "local_origin_m" in value, "full_position" in value
    q, v = value["position"], value["velocity"]
    if "generalized_applied" not in value:
        array(q, 3)
        array(v, 3)
        array(value["applied_effort"], 3)
        array(value["control"], 3)
        same(value["history"], [], "original free-sphere history")
        require(value["topology"], "original scalar topology")
        return dict(kind="scalar", n=0, k=0, b=1, d=3)
    fields(value, ("topology", "generalized_applied"), "tree checkpoint")
    require(value["topology"], "original topology retained")
    k = len(v) - (6 if floating else 0)
    n = len(value["full_velocity"]) - (6 if floating else 0) if affine else k
    require(
        0 <= n <= 16 and (1 <= k < n if affine else k == n), "physical/full and independent counts"
    )
    b, d = n + int(floating), n + 6 * int(floating)
    array(q, k + 7 * int(floating))
    array(v, k + 6 * int(floating))
    array(value["applied_effort"], k)
    array(value["generalized_applied"], k + 6 * int(floating))
    if floating:
        array(value["local_origin_m"], 3)
    if affine:
        fields(
            value,
            ("full_position", "full_velocity", "full_applied_effort", "full_generalized_applied"),
            "full affine checkpoint",
        )
        array(value["full_position"], n + 7 * int(floating))
        array(value["full_velocity"], d)
        array(value["full_applied_effort"], n)
        array(value["full_generalized_applied"], d)
        if floating:
            exact(q[:7], value["full_position"][:7], "one actual full/reduced root pose")
            exact(v[:6], value["full_velocity"][:6], "one actual full/reduced root twist")
    control = value["control"]
    if floating:
        same(len(control), 5 if affine else 4, "floating held blocks")
        blocks = control[1:] if affine else control
        if affine:
            require(control[0], "held original graph identity")
        for block, shape in zip(blocks, ((k, 3), (n,), (b, 3), (b, 3)), strict=True):
            array(block, *shape)
        same(blocks[1], [0.0] * n, "full joint effort stays zero")
    elif affine:
        same(len(control), 2, "fixed affine held blocks")
        array(control[0], k, 3)
        array(control[1], n, 7)
    else:
        array(control, n, 10)
    if value["history"]:
        array(value["history"], k, 3)
    return dict(
        kind=("floating_" if floating else "") + ("affine" if affine else "tree"),
        n=n,
        k=k,
        b=b,
        d=d,
    )


def state_map(values, worlds=None):
    require(isinstance(values, dict) and values, "nonempty complete state mapping")
    if worlds is not None:
        same(set(values), {str(w) for w in worlds}, "all declared world states")
    result = {}
    for key, value in values.items():
        same(str(value["world"]), key, "world map ownership")
        result[key] = checkpoint(value)
    return result


def geometry(value, state):
    spec = checkpoint(state)
    fields(
        value,
        (
            "tick",
            "joint_types",
            "position",
            "velocity",
            "link_position_m",
            "link_rotation",
            "com_position_m",
        ),
        "full FK",
    )
    same(value["tick"], state["tick"], "same accepted FK tick")
    b = spec["b"]
    for name in ("link_position_m", "com_position_m"):
        array(value[name], b, 3)
    if "local_origin_m" in state:
        for name in ("linear_velocity_m_s", "angular_velocity_rad_s"):
            array(value[name], b, 3)
    array(value["link_rotation"], b, 3, 3)
    close(
        value["position"],
        state.get("full_position", state["position"]),
        "full FK q",
        atol=0,
        rtol=0,
    )
    close(
        value["velocity"],
        state.get("full_velocity", state["velocity"]),
        "full FK v",
        atol=0,
        rtol=0,
    )
    if "local_origin_m" in state:
        same(
            [
                value[name]
                for name in ("joint_count", "body_count", "velocity_count", "configuration_count")
            ],
            [spec["n"], b, spec["d"], spec["n"] + 7],
            "FK N/B/D/Q",
        )
    if "full_position" in state:
        close(value["independent_position"], state["position"], "independent FK q", atol=0, rtol=0)
        close(value["independent_velocity"], state["velocity"], "independent FK v", atol=0, rtol=0)
        same(len(value["joint_names"]), spec["k"], "independent names")
        same(len(value["full_joint_names"]), spec["n"], "full names")


def diagnostic(value, state):
    spec = checkpoint(state)
    n, k, b, d = (spec[name] for name in ("n", "k", "b", "d"))
    floating, affine = "local_origin_m" in state, "full_position" in state
    reduced = k + 6 * int(floating)
    fields(
        value,
        (
            "entry_tick",
            "exit_tick",
            "dt_s",
            "entry_position",
            "entry_velocity",
            "mass",
            "bias",
            "external",
            "linear_jacobian",
            "angular_jacobian",
            "drive",
            "acceleration",
            "iterations",
            "control_residual",
            "equation_residual",
        ),
        "complete mechanics",
    )
    same(value["exit_tick"], value["entry_tick"] + 1, "one diagnostic attempt")
    same(value["exit_tick"], state["tick"], "latest actual accepted attempt")
    array(value["entry_position"], k + 7 * int(floating))
    array(value["entry_velocity"], reduced)
    array(value["mass"], d, d)
    array(value["bias"], d)
    array(value["external"], d)
    array(value["linear_jacobian"], b, 3, d)
    array(value["angular_jacobian"], b, 3, d)
    array(value["drive"], k)
    array(value["acceleration"], reduced)
    require(
        0 <= number(value["control_residual"]) <= 5e-5
        and 0 <= number(value["equation_residual"]) <= 2e-5,
        "original solver residual gates",
    )
    if affine:
        array(value["full_entry_position"], n + 7 * int(floating))
        array(value["full_entry_velocity"], d)
        array(value["reduced_mass"], reduced, reduced)
        for name in ("reduced_bias", "reduced_external"):
            array(value[name], reduced)
        for name in ("reduced_linear_jacobian", "reduced_angular_jacobian"):
            array(value[name], b, 3, reduced)
        array(value["full_acceleration"], d)
        array(value["constraint_effort_full"], d)
        array(value["row_multipliers"], n - k)
        array(value["row_powers_w"], n - k)
        number(value["constraint_power_w"])
        if floating:
            array(value["root_constraint_residual"], 6)


def held_matches(state, command):
    if "local_origin_m" in state:
        fields(command, ("drive", "external"), "original full COM command")
        load = command["external"]
        expected = [command["drive"], load["effort"], load["force"], load["torque"]]
        if "full_position" in state:
            expected.insert(0, command["graph_token"])
        # The held representation is the declared packet rounded to FP32.
        actual = state["control"]
        if "full_position" in state:
            exact(actual[0], expected[0], "original graph held token")
            actual, expected = actual[1:], expected[1:]
        exact(actual, fp32_values(expected), "complete held floating control")
    elif "full_position" in state:
        fields(command, ("graph_token", "drive", "loads"), "original fixed-affine inputs")
        exact(command["graph_token"], state["topology"][1], "same original fixed-affine graph")
        exact(
            state["control"],
            fp32_values([command["drive"], command["loads"]]),
            "all fixed-affine held input columns",
        )
    elif "generalized_applied" in state:
        fields(command, ("domains",), "original ordinary SceneInputs")
        same(len(command["domains"]), 1, "one declared ordinary drive domain")
        command = next(iter(command["domains"].values()))
        fields(
            command,
            ("target", "target_velocity", "feedforward", "effort", "force", "torque"),
            "original ordinary tree inputs",
        )
        n = len(state["position"])
        expected = [
            [
                command["target"][i],
                command["target_velocity"][i],
                command["feedforward"][i],
                command["effort"][i],
                *command["force"][i],
                *command["torque"][i],
            ]
            for i in range(n)
        ]
        exact(state["control"], fp32_values(expected), "complete ordinary held ten columns")
    else:
        array(command, 3)
        exact(state["control"], fp32_values(command), "complete held free-sphere COM force")


def fp32_values(value):
    if isinstance(value, list):
        return [fp32_values(item) for item in value]
    number(value)
    return struct.unpack("<f", struct.pack("<f", value))[0]


def key_tuple(value):
    fields(value, ("world", "body", "shape"), "CollisionKey")
    require(
        all(type(value[k]) is int for k in ("world", "body", "shape")), "integer material identity"
    )
    return tuple(value[k] for k in ("world", "body", "shape"))


def shape_schema(value):
    fields(
        value,
        (
            "key",
            "kind",
            "vertex_ids",
            "radius",
            "group",
            "mask",
            "revision",
            "normal",
            "triangles",
            "heightfield",
            "curved",
        ),
        "complete source shape",
    )
    key = key_tuple(value["key"])
    require(value["kind"] in ("sphere", "capsule", "hull"), "frozen finite shape")
    ids = value["vertex_ids"]
    require(
        ids and len(set(ids)) == len(ids) and all(type(x) is int for x in ids),
        "original vertex IDs",
    )
    number(value["radius"])
    return key


def bindings_schema(model, bindings):
    fields(model, ("shapes", "excluded"), "full collision model")
    shapes = {shape_schema(row): row for row in model["shapes"]}
    same(len(shapes), len(model["shapes"]), "unique shape identity")
    same({key_tuple(b["key"]) for b in bindings}, set(shapes), "all registered bindings")
    same(len(bindings), len(shapes), "one binding per shape")
    result = {}
    for bound in bindings:
        key = key_tuple(bound["key"])
        if "group_index" in bound:
            same(bound["world"], key[0], "same-world material owner")
            array(bound["local_points_m"], len(shapes[key]["vertex_ids"]), 3)
            require(
                type(bound["group_index"]) is int and bound["group_index"] >= 0, "real body group"
            )
            if bound.get("_source_shape") is None:
                require(bound["topology"], "direct full topology provenance")
            else:
                original = bound["_source_shape"]
                shape_schema(original)
                require(bound.get("_model_token"), "from_asset full model provenance")
                exact(
                    dict(original, key=bound["key"]),
                    shapes[key],
                    "source shape unchanged except key",
                )
        else:
            array(bound["motion"]["start"], len(shapes[key]["vertex_ids"]), 3)
        result[key] = bound
    return shapes, result


def frames_schema(values, world, geo, shapes, bindings):
    require(isinstance(values, dict), "all material source frames")
    result = {}
    for frame in values.values():
        fields(
            frame,
            (
                "shape",
                "world_points_m",
                "group_index",
                "position_m",
                "rotation",
                "reference_point_m",
                "linear_velocity_m_s",
                "angular_velocity_rad_s",
                "rotation_orthogonality_error",
            ),
            "actual source frame",
        )
        key = shape_schema(frame["shape"])
        same(key[0], world, "frame world")
        require(key not in result, "unique source frame")
        result[key] = frame
        array(frame["world_points_m"], len(frame["shape"]["vertex_ids"]), 3)
        for name in (
            "position_m",
            "reference_point_m",
            "linear_velocity_m_s",
            "angular_velocity_rad_s",
        ):
            array(frame[name], 3)
        array(frame["rotation"], 3, 3)
        number(frame["rotation_orthogonality_error"])
        if shapes is not None:
            exact(frame["shape"], shapes[key], "original shape packet")
            same(frame["group_index"], bindings[key].get("group_index"), "full physical group")
        group = frame["group_index"]
        if group is not None:
            require(
                type(group) is int and 0 <= group < len(geo["link_position_m"]), "full B bounds"
            )
            for name, source in (
                ("position_m", "link_position_m"),
                ("rotation", "link_rotation"),
                ("reference_point_m", "com_position_m"),
            ):
                exact(frame[name], geo[source][group], "stored geometry/packet " + name)
    if shapes is not None:
        same(
            set(result),
            {key for key in shapes if key[0] == world},
            "complete world source frame set",
        )
    return result


def receipt(value, state=None, *, shapes=None, bindings=None):
    fields(
        value,
        (
            "world",
            "entry_tick",
            "exit_tick",
            "dt_s",
            "entry_state",
            "accepted_state",
            "entry_geometry",
            "accepted_geometry",
            "input_external",
            "selected_drive",
            "law_wrench_fp64",
            "uploaded_wrench_fp32",
            "contact_generalized",
            "actual_drive",
            "sites",
            "static_reactions",
            "diagnostics",
        ),
        "complete contact receipt",
    )
    entry, end, meta = value["entry_state"], value["accepted_state"], value["diagnostics"]
    spec = checkpoint(entry)
    checkpoint(end)
    n, k, b, d = (spec[name] for name in ("n", "k", "b", "d"))
    same(value["world"], entry["world"], "entry owner")
    same(end["world"], value["world"], "accepted owner")
    same(value["entry_tick"], entry["tick"], "entry tick")
    same(value["exit_tick"], end["tick"], "exit tick")
    same(end["tick"], entry["tick"] + 1, "one accepted contact step")
    same(value["dt_s"], 1 / 1024, "contact fixed h")
    exact(entry["control"], end["control"], "temporary force does not replace held controls")
    if state is not None:
        exact(end, state, "same accepted receipt checkpoint")
    for name, shape in (
        ("input_external", (d,)),
        ("contact_generalized", (d,)),
        ("selected_drive", (k, 3)),
        ("actual_drive", (k,)),
        ("law_wrench_fp64", (b, 6)),
        ("uploaded_wrench_fp32", (b, 6)),
    ):
        array(value[name], *shape)
    geometry(value["entry_geometry"], entry)
    geometry(value["accepted_geometry"], end)
    if "full_position" in entry:
        nested_name = (
            "floating_affine_diagnostic" if "local_origin_m" in entry else "affine_diagnostic"
        )
        same(
            meta["coordinate_kind"],
            "floating_affine" if "local_origin_m" in entry else "affine",
            "affine receipt type",
        )
        array(meta["constraint_rows"], n - k, n)
        array(meta["offset_vector"], n)
        array(meta["projection_matrix"], d, k + (6 if "local_origin_m" in entry else 0))
        nested = meta[nested_name]
        diagnostic(nested, end)
        exact(nested["full_entry_position"], entry["full_position"], "full sampled q")
        exact(nested["full_entry_velocity"], entry["full_velocity"], "full sampled v")
        exact(
            meta["graph_token"],
            entry["control"][0] if "local_origin_m" in entry else entry["topology"][1],
            "original graph identity",
        )
        if "local_origin_m" in entry:
            same(
                [
                    meta[name]
                    for name in (
                        "joint_count",
                        "body_count",
                        "velocity_count",
                        "position_count",
                        "full_dof_count",
                        "reduced_dof_count",
                        "independent_dof_count",
                    )
                ],
                [n, b, d, n + 7, d, k + 6, k],
                "full/reduced counts",
            )
    elif "local_origin_m" in entry:
        nested = meta["floating_diagnostic"]
        diagnostic(nested, end)
    else:
        nested = None
    fields(
        meta,
        (
            "entry_frames",
            "accepted_frames",
            "mass",
            "bias",
            "linear_jacobian",
            "angular_jacobian",
            "acceleration",
            "total_external",
            "equation_residual",
            "control_residual",
            "entry_kinetic_energy_j",
            "accepted_kinetic_energy_j",
            "kinetic_energy_change_j",
            "entry_power_times_h_j",
            "application_work_j",
            "material_work_j",
            "wrench_quantization_error",
        ),
        "same-attempt contact mechanics/work",
    )
    if nested is not None:
        for name in ("mass", "bias", "linear_jacobian", "angular_jacobian"):
            close(meta[name], nested[name], "same nested " + name, atol=0, rtol=0)
        close(
            meta["acceleration"],
            nested.get("full_acceleration", nested["acceleration"]),
            "same full acceleration",
            atol=0,
            rtol=0,
        )
        close(meta["total_external"], nested["external"], "same full external", atol=0, rtol=0)
        close(value["actual_drive"], nested["drive"], "same motor", atol=0, rtol=0)
    prior = frames_schema(
        meta["entry_frames"], value["world"], value["entry_geometry"], shapes, bindings
    )
    later = frames_schema(
        meta["accepted_frames"], value["world"], value["accepted_geometry"], shapes, bindings
    )
    same(set(prior), set(later), "same full material source set")
    for key, bound in (bindings or {}).items():
        if key[0] == value["world"] and "group_index" in bound:
            if bound.get("_source_shape") is None:
                exact(bound["topology"], entry["topology"], "direct full topology")
            if "local_origin_m" in entry and "full_position" in entry:
                exact(bound["graph_token"], entry["control"][0], "floating-affine material graph")
                if bound.get("_source_shape") is not None:
                    exact(
                        bound["_model_token"],
                        entry["control"][0][1],
                        "from_asset full floating model",
                    )
    for site in value["sites"]:
        fields(
            site,
            ("pair", "geometry", "law", "application_work_j", "material_work_j"),
            "complete site",
        )
        pair = tuple(key_tuple(key) for key in site["pair"])
        require(
            len(pair) == 2 and pair[0] < pair[1] and pair[0][0] == pair[1][0] == value["world"],
            "canonical same-world pair",
        )
        require(set(pair) <= prior.keys(), "site owners in original frame set")
        packet = site["geometry"]
        fields(packet, ("witness", "load_point", "weight"), "ContactSite")
        array(packet["load_point"], 3)
        require(number(packet["weight"]) > 0, "positive patch weight")
        witness = packet["witness"]
        fields(
            witness,
            (
                "gap_m",
                "normal",
                "first_point",
                "second_point",
                "first_weights",
                "second_weights",
                "source",
                "axis_parameter_span",
                "first_offset",
                "second_offset",
                "intersection_points",
                "shared_material_points",
                "first_height_m",
                "second_height_m",
                "capsule_axis_span",
            ),
            "unflattened witness",
        )
        number(witness["gap_m"])
        array(witness["normal"], 3)
        for index, side in enumerate(("first", "second")):
            array(witness[side + "_point"], 3)
            if witness[side + "_offset"] is not None:
                array(witness[side + "_offset"], 3)
            weights = witness[side + "_weights"]
            require(
                weights and len({a[0] for a in weights}) == len(weights),
                "complete material weights",
            )
            for address in weights:
                require(
                    len(address) == 2
                    and address[0] in prior[pair[index]]["shape"]["vertex_ids"]
                    and number(address[1]) >= 0,
                    "source vertex ID/weight",
                )
            for suffix in (
                "velocity_m_s",
                "force_n",
                "reference_point_m",
                "material_displacement_m",
                "application_displacement_m",
            ):
                array(site[side + "_" + suffix], 3)
            exact(
                site[side + "_reference_point_m"],
                prior[pair[index]]["reference_point_m"],
                "same COM/static reference",
            )
            if prior[pair[index]]["group_index"] is None:
                for suffix in ("material_displacement_m", "application_displacement_m"):
                    same(site[side + "_" + suffix], [0, 0, 0], "static motion zero")
        same(site["law"]["friction"], 0, "original frictionless law")
        number(site["application_work_j"])
        number(site["material_work_j"])
    for reaction in value["static_reactions"].values():
        for name in ("reference_point_m", "force_n", "torque_nm"):
            array(reaction[name], 3)


def record_frame(
    row, worlds=None, *, shapes=None, bindings=None, receipt_worlds=None, allow_old_receipts=False
):
    fields(row, ("state", "geometry", "diagnostics", "summary"), "complete public frame")
    specs = state_map(row["state"], worlds)
    trees = {w for w, spec in specs.items() if spec["kind"] != "scalar"}
    same(set(row["geometry"]), trees, "full FK map coverage")
    same(set(row["diagnostics"]), trees, "full diagnostic map coverage")
    same(set(row["summary"]), set(specs), "full summary map coverage")
    for w in trees:
        geometry(row["geometry"][w], row["state"][w])
        if row["diagnostics"][w] is not None:
            diagnostic(row["diagnostics"][w], row["state"][w])
    for value in row["summary"].values():
        number(value["kinetic_energy_j"])
    if receipt_worlds is not None:
        fields(row, ("receipts",), "published contact records")
        same(
            set(row["receipts"]), {str(w) for w in receipt_worlds}, "full accepted receipt coverage"
        )
    for w, value in row.get("receipts", {}).items():
        require(w in trees, "receipt actual tree owner")
        receipt(
            value, None if allow_old_receipts else row["state"][w], shapes=shapes, bindings=bindings
        )
    return specs


def input_schema(data):
    fields(
        data,
        (
            "name",
            "variant",
            "placement",
            "root_mass",
            "root_com",
            "root_inertia",
            "root_inertia_frame",
            "children",
            "gravity",
            "h",
            "q",
            "v",
            "drive",
            "kp",
            "kd",
            "caps",
            "force",
            "torque",
            "effort",
            "P",
            "offset",
            "C",
            "S",
            "E",
            "drivers",
            "parts",
            "contacts",
        ),
        "complete original affine input",
    )
    n, k = len(data["P"]), len(data["P"][0])
    require(1 <= k < n <= 16, "declared physical/independent input dimensions")
    array(data["P"], n, k)
    array(data["offset"], n)
    array(data["C"], n - k, n + 6)
    array(data["S"], n + 6, k + 6)
    array(data["E"], n + 6, k)
    array(data["q"], k + 7)
    array(data["v"], k + 6)
    array(data["drive"], k, 3)
    for name in ("kp", "kd", "caps"):
        array(data[name], k)
    array(data["force"], n + 1, 3)
    array(data["torque"], n + 1, 3)
    same(data["effort"], [0] * n, "no second full joint effort channel")
    array(data["root_com"], 3)
    array(data["root_inertia"], 3)
    array(data["root_inertia_frame"], 4)
    children = data["children"]
    fields(
        children,
        ("parents", "kinds", "offsets", "orientations", "axes", "mass", "com", "inertia"),
        "all original physical children",
    )
    for name in ("parents", "kinds", "mass"):
        same(len(children[name]), n, "all input physical rows")
    for name in ("offsets", "axes", "com", "inertia"):
        array(children[name], n, 3)
    array(children["orientations"], n, 4)
    return n, k


def input_identity(member, value):
    data = value.get("input")
    if data is None:
        return
    n, k = input_schema(data)
    if member.family in ("step", "pd", "counts", "contact"):
        same(data["h"], member.h, "frozen authored h")
        function = member.function
        if "a04" in function:
            same((data["name"], data["variant"]), ("A04", member.parameter), "A04 original pose")
            same(len(data["parts"]), 4, "all authored inertial components")
        elif "a03" in function:
            same(
                (data["name"], data["variant"]),
                ("A03", member.parameter),
                "original fan/chain rows",
            )
            same((n, k), (3, 1), "fan/chain full and reduced")
        elif member.family == "counts":
            same(
                (data["name"], data["variant"], n, k),
                ("counts", member.parameter, 16, 1 if member.parameter.endswith("k1") else 15),
                "N16 motor capacity",
            )
        elif member.family == "pd":
            same(data["name"], "A01", "PD original physical model")
            same(
                value["compiled"]["503"]["delay"], int(member.parameter[1]), "original actual delay"
            )
            same(data["kp"], [100], "original kp")
            same(data["kd"], [2], "original kd")
            same(data["caps"], [0.25 if "c0p25" in member.parameter else 1.5], "original cap")
            same(data["drive"], [[0.01, 0, 0]], "original PD target")
            same(data["force"][0], [1, 0, 0], "instantaneous COM load")
        elif member.family == "contact":
            name = "A02" if "a02" in function else "A05"
            same(data["name"], name, "original contact identity")
            same(value["reverse"], member.parameter == "reverse", "authored registration order")
            same(
                data["placement"],
                "cycle" if member.parameter == "cycle" else "base",
                "authored placement",
            )
        else:
            same((data["name"], data["placement"]), ("A01", "cycle"), "cycle control identity")


def compiled_schema(values, states):
    require(isinstance(values, dict), "all compiled tree declarations")
    trees = {w for w, state in states.items() if "generalized_applied" in state}
    same(set(values), trees, "compiled tree/world coverage")
    for w, value in values.items():
        require(isinstance(value, dict) and value, "nonempty actual compilation")
        state = states[w]
        exact(value["topology"], state["topology"], "same complete compiled topology")
        if "local_origin_m" in state and "full_position" in state:
            spec = checkpoint(state)
            same(
                (value["n"], value["k"], value["r"]),
                (spec["n"], spec["k"], spec["n"] - spec["k"]),
                "compiled full graph counts",
            )
            exact(value["graph_token"], state["control"][0], "original compiled graph")
            for name, shape in (
                ("matrix", (spec["n"], spec["k"])),
                ("projection", (spec["d"], spec["k"] + 6)),
                ("offset", (spec["n"],)),
                ("constraints", (spec["n"] - spec["k"], spec["n"])),
            ):
                array(value[name], *shape)
            fields(
                value,
                (
                    "physical",
                    "row_indices",
                    "row_values",
                    "reverse_rows",
                    "full_limits",
                    "limits",
                    "independent",
                    "initial_state",
                    "initial_loads",
                ),
                "compiled original physical/row tables",
            )


def saved_comparison(actual, expected):
    require(isinstance(expected, dict), "saved independent mechanics")
    names = (
        "mass",
        "bias",
        "external",
        "linear_jacobian",
        "angular_jacobian",
        "drive",
        "acceleration",
        "reduced_mass",
        "reduced_bias",
        "reduced_external",
        "reduced_linear_jacobian",
        "reduced_angular_jacobian",
        "full_acceleration",
        "constraint_effort_full",
        "row_multipliers",
        "row_powers_w",
        "root_constraint_residual",
    )
    for name in names:
        if name in actual:
            require(name in expected, "saved independent " + name)
            close(
                actual[name],
                expected[name],
                "saved independent " + name,
                atol=2e-5
                if name
                in (
                    "mass",
                    "reduced_mass",
                    "constraint_effort_full",
                    "row_multipliers",
                    "row_powers_w",
                    "root_constraint_residual",
                )
                else ATOL,
            )


def audit_saved_row(row, worlds, *, bound=(), shapes=None, bindings=None, require_audit=True):
    record_frame(
        row, worlds, shapes=shapes, bindings=bindings, receipt_worlds=bound if bound else None
    )
    if require_audit:
        fields(row, ("audit",), "saved independent comparison")
        audits = (
            row["audit"]
            if set(row["audit"]) <= {str(w) for w in worlds}
            else {str(worlds[0]): row["audit"]}
        )
        same(set(audits), {str(w) for w in worlds}, "all authored per-world audit records")
        for w, audit in audits.items():
            if checkpoint(row["state"][w])["kind"] == "scalar":
                same(
                    audit,
                    None,
                    "scalar original check is assert-only, not an invented audit record",
                )
                continue
            require(
                audit is not None and row["diagnostics"].get(w) is not None,
                "accepted tree has its actual diagnostic and saved audit",
            )
            if "same_entry_reference" in audit:
                saved_comparison(row["diagnostics"][w], audit["same_entry_reference"])
                close(
                    row["state"][w]["position"],
                    audit["predicted_position"],
                    "saved ordinary floating prediction q",
                )
                close(
                    row["state"][w]["velocity"],
                    audit["predicted_velocity"],
                    "saved ordinary floating prediction v",
                )
            if "same_entry" in audit:
                saved_comparison(row["diagnostics"][w], audit["same_entry"])
                close(row["state"][w]["position"], audit["next_q"], "saved same-entry q")
                close(row["state"][w]["velocity"], audit["next_v"], "saved same-entry v")
                if "receipts" in row and w in row["receipts"]:
                    receipt_value = row["receipts"][w]
                    truth = audit["same_entry"]
                    close(
                        receipt_value["law_wrench_fp64"],
                        truth["independent_wrench"],
                        "saved independently sampled law",
                    )
                    for name in ("contact_generalized", "input_external"):
                        close(receipt_value[name], truth[name], "saved independent contact " + name)
                    works = audit.get("original_work", audit.get("material_work"))
                    if works is not None:
                        for name, expected in works.items():
                            close(
                                receipt_value["diagnostics"][name],
                                expected,
                                "saved work comparison",
                                atol=1e-12,
                                rtol=1e-10,
                            )


def bound_source(record):
    if "model" in record:
        return bindings_schema(record["model"], record["bindings"])
    if "collision_model" in record:
        return bindings_schema(record["collision_model"], record["bindings"])
    return None, None


def compare_replay(rows, replay, fields_to_compare):
    same(len(rows), 40, "original full window")
    same(len(replay), 21, "full nineteen-to-forty replay")
    for index, row in enumerate(replay, start=19):
        for name in fields_to_compare:
            exact(row[name], rows[index][name], "full replay " + name)


def same_retained(before, after, *, epoch=True):
    for name in ("state", "geometry", "summary", "receipts", "status"):
        exact(after[name], before[name], "failed whole attempt retained " + name)
    same(set(after["diagnostics"]), set(before["diagnostics"]), "failure diagnostic owner coverage")
    require(
        all(value is None for value in after["diagnostics"].values()),
        "failed transient diagnostics invalidated",
    )
    if epoch:
        exact(after["epoch"], before["epoch"], "failed epoch retained")
        exact(after["bank"], before["bank"], "failed active bank retained")


def snapshots(record, names, worlds, shapes=None, bindings=None, bound=()):
    for name in names:
        fields(record, (name,), "saved transaction")
        frame = record[name]
        wanted = [w for w in bound if frame["state"][str(w)]["tick"] > 0]
        record_frame(frame, worlds, shapes=shapes, bindings=bindings, receipt_worlds=wanted)


def x07(record):
    variant = record["variant"]
    fields(
        record,
        (
            "initial",
            "warm",
            "warm_audit",
            "before",
            "injections",
            "rejected",
            "failed_retained",
            "retry",
            "retry_audit",
            "retained",
        ),
        "actual corrupted readback evidence",
    )
    same(len(record["injections"]), 1, "one actual diagnostic corruption")
    injection = record["injections"][0]
    same(injection["world"], 503, "selected world")
    same(injection["selected_worlds"], [503], "only selected readback")
    fields(
        injection,
        (
            "storage_row",
            "original_full",
            "original_reduced",
            "modified_full",
            "modified_reduced",
            "modified_array",
        ),
        "both actual diagnostic banks",
    )
    for side in ("full", "reduced"):
        scan(injection["original_" + side])
        same(
            set(injection["modified_" + side]),
            set(injection["original_" + side]),
            "all readback arrays retained",
        )
    change = injection["modified_array"]
    owner, field = (
        ("reduced", "mass")
        if variant == "reduced_nan"
        else ("full", "mass")
        if variant == "full_nan"
        else ("full", "info")
    )
    same((change["owner"], change["field"]), (owner, field), "declared injection target")
    row = injection["storage_row"]
    require(
        type(row) is int and 0 <= row < len(injection["original_" + owner][field]),
        "actual selected storage row",
    )
    same(change["shape"], value_shape(change["before"]), "actual selected readback shape")
    for other in range(len(injection["original_" + owner][field])):
        if other != row:
            exact(
                injection["modified_" + owner][field][other],
                injection["original_" + owner][field][other],
                "unselected actual readback row unchanged",
            )
    exact(
        change["before"],
        injection["original_" + owner][field][row],
        "actual original selected array",
    )
    exact(
        change["after"],
        injection["modified_" + owner][field][row],
        "actual modified selected array",
    )
    for side in ("full", "reduced"):
        for name, before in injection["original_" + side].items():
            if (side, name) != (owner, field):
                exact(before, injection["modified_" + side][name], "unmodified actual readback")
    original_info = injection["original_full"]["info"][row]
    same(original_info[:3], [1, 2, 0], "successful actual candidate before observer corruption")
    if variant == "stale_tick":
        same(change["dtype"], "int32", "actual integer tick array")
        exact(change["before_int32_values"], change["before"], "saved original int32 values")
        exact(change["after_int32_values"], change["after"], "saved modified int32 values")
        expected = list(change["before"])
        expected[1] = expected[0]
        exact(change["after"], expected, "only exit tick changed")
        require("ticks" in record["rejected"]["message"], "normal constructor rejects stale tick")
    else:
        same(change["dtype"], "float32", "actual FP32 readback")
        exact(change["after"][0][0], {"nonfinite": "nan"}, "declared NaN only")
        for phase in ("before", "after"):
            values, bits = change[phase], change[phase + "_uint32_bits"]
            same(len(values), len(bits), "all raw rows retained")
            for i, (value_row, bit_row) in enumerate(zip(values, bits, strict=True)):
                same(len(value_row), len(bit_row), "all raw columns retained")
                for j, (value, word) in enumerate(zip(value_row, bit_row, strict=True)):
                    require(type(word) is int and 0 <= word < 2**32, "uint32 actual bit pattern")
                    decoded = struct.unpack("<f", struct.pack("<I", word))[0]
                    if phase == "after" and (i, j) == (0, 0):
                        require(math.isnan(decoded), "saved bits encode actual NaN")
                    else:
                        number(value)
                        same(decoded, value, "all original raw FP32 words")
                        if phase == "after":
                            exact(
                                value, change["before"][i][j], "only first matrix element corrupted"
                            )
        require(
            "nonfinite" in record["rejected"]["message"], "normal constructor rejects nonfinite"
        )
    same(record["rejected"]["type"], "ValueError", "actual normal constructor refusal")
    same_retained(record["before"], record["failed_retained"])
    exact(
        record["retry"]["receipts"]["503"]["entry_state"],
        record["before"]["state"]["503"],
        "retry same full entry",
    )
    same(record["retry"]["state"]["503"]["tick"], 2, "one warm plus one accepted retry")
    # No nonfinite marker outside the specifically saved modified arrays.
    scan({key: value for key, value in record.items() if key != "injections"})


def physical_payload(member, relative, record, *, partial=False):
    if partial:
        scan(record, tags=True)
        return dict(
            status="failed-prefix",
            rows=len(record.get("rows", record.get("frames", []))),
            failure=record.get("failure"),
        )
    require("failure" not in record, "failed prefix is not complete")
    scan(record, tags=member.family == "diagnostic")
    input_identity(member, record)
    shapes, bindings = bound_source(record)
    if member.family == "public":
        fields(
            record,
            (
                "asset",
                "fixture",
                "initial",
                "full_initial",
                "external",
                "drive",
                "host_premise",
                "host_step",
                "packet",
                "compiled",
                "frames",
                "retained",
            ),
            "A01 full old HOST premise and device record",
        )
        same(record["case"], "A01", "public first node")
        same(len(record["frames"]), 1, "public one actual step")
        frame = record["frames"][0]
        state = frame["accepted"]
        checkpoint(frame["entry"])
        checkpoint(state)
        compiled_schema(record["compiled"], {"503": frame["entry"]})
        geometry(frame["geometry"], state)
        diagnostic(frame["diagnostic"], state)
        close(
            frame["diagnostic"]["full_acceleration"],
            record["literal_acceleration"],
            "saved literal acceleration",
        )
        close(
            frame["diagnostic"]["constraint_effort_full"],
            record["literal_constraint"],
            "saved literal original reaction",
            atol=2e-5,
        )
        same(set(frame["summary"]), {"503"}, "complete public first-step summary")
        number(frame["summary"]["503"]["kinetic_energy_j"])
        state_map(record["retained"], (503,))
        exact(record["retained"]["503"], state, "actual retained public first step")
        held_matches(state, record["packet"])
        return dict(rows=1, premise="same-node HOST; no extra test count")
    if member.family in ("step", "pd", "counts", "contact", "mixed"):
        fields(record, ("rows", "initial", "retained", "stats", "compiled"), "full original window")
        if member.family == "mixed":
            category = member.parameter
            same(record["category"], category, "mixed original category")
            worlds = [int(relative.split("/")[1][3:])] if "/w1-" in relative else list(WORLDS)
            if member.optional:
                worlds += [10001 + 97 * i for i in range(28)]
            same(set(record["inputs"]), {str(w) for w in worlds}, "all mixed original inputs")
            same(set(record["commands"]), set(record["inputs"]), "all original mixed commands")
            bound = worlds if category == "contact" else ()
        else:
            worlds, bound = [503], [503] if member.family == "contact" else []
        compiled_schema(record["compiled"], record["initial"]["state"])
        if member.family == "contact":
            require(
                "receipts" not in record["initial"],
                "standalone contact initial precedes plan creation",
            )
        record_frame(
            record["initial"],
            worlds,
            shapes=shapes,
            bindings=bindings,
            receipt_worlds=[] if bound and member.family != "contact" else None,
        )
        same(len(record["rows"]), member.steps, "complete original step budget")
        previous = record["initial"]["state"]
        for index, row in enumerate(record["rows"]):
            audit_saved_row(row, worlds, bound=bound, shapes=shapes, bindings=bindings)
            for w in map(str, worlds):
                same(row["state"][w]["tick"], index + 1, "every actual physical step")
                if "entry" in row:
                    exact(row["entry"][w], previous[w], "full entry continuity")
                exact(row["state"][w]["control"], previous[w]["control"], "held command continuity")
                held_matches(
                    row["state"][w],
                    record["commands"][w] if member.family == "mixed" else record["command"],
                )
            if "trajectory_reference" in row:
                close(
                    row["state"]["503"]["position"],
                    row["trajectory_reference"]["q"],
                    "saved independent window q",
                    atol=WINDOW_ATOL,
                )
                close(
                    row["state"]["503"]["velocity"],
                    row["trajectory_reference"]["v"],
                    "saved independent window v",
                    atol=WINDOW_ATOL,
                )
            for w, values in row.get("independent", {}).items():
                close(
                    row["state"][w]["position"],
                    values[0],
                    "saved mixed independent q",
                    atol=WINDOW_ATOL,
                )
                close(
                    row["state"][w]["velocity"],
                    values[1],
                    "saved mixed independent v",
                    atol=WINDOW_ATOL,
                )
            previous = row["state"]
        if member.replay:
            exact(record["saved"], record["rows"][18]["state"], "exact tick19 checkpoint")
            compare_replay(
                record["rows"],
                record["replay"],
                ("state", "geometry", "diagnostics", "summary", *(("receipts",) if bound else ())),
            )
            for row in record["replay"]:
                record_frame(
                    row,
                    worlds,
                    shapes=shapes,
                    bindings=bindings,
                    receipt_worlds=bound if bound else None,
                )
        elif "replay" in record:
            same(record["replay"], [], "no unregistered replay")
        record_frame(
            record["retained"],
            worlds,
            shapes=shapes,
            bindings=bindings,
            receipt_worlds=bound if bound else None,
        )
        if member.family == "counts":
            fields(record, ("body16_current_bounds",), "N16 actual original body16 geometry")
            saved = record["body16_current_bounds"]
            fields(
                saved,
                ("binding", "shape", "result", "stored_center", "independent_center"),
                "complete body16 bounds record",
            )
            count_shapes, _ = bindings_schema(
                dict(shapes=[saved["shape"]], excluded=[]), [saved["binding"]]
            )
            broad_schema(saved["result"], record["rows"][0]["state"], count_shapes)
            fields(
                saved["result"], ("owner", "registration"), "complete counts process-local fields"
            )
            for name in ("owner", "registration"):
                exact(
                    saved["result"][name],
                    {
                        "opaque_process_local_type": "builtins.object",
                        "identity_serialized": False,
                    },
                    "counts opaque process-local " + name,
                )
            close(
                saved["stored_center"],
                saved["independent_center"],
                "saved body16 independent location",
            )
            same(saved["binding"]["group_index"], 16, "full body16 owner, not K bound")
        if (
            member.family == "mixed"
            and member.parameter == "contact"
            and not member.optional
            and "503" in record["inputs"]
        ):
            values = record["lifecycle"]["503"]
            same(len(values), 40, "all lifecycle samples")
            require(
                values[0] is True and False in values and any(values[values.index(False) + 1 :]),
                "original loaded/open/reentry",
            )
        return dict(rows=member.steps, replay=21 if member.replay else 0, worlds=worlds)
    if member.family == "restore":
        same(record["problem"], member.parameter, "restore case")
        checkpoint(record["initial"])
        state_map(record["entry"], (503,))
        for name in ("first", "replay", "retained"):
            record_frame(record[name], [503])
        exact(record["first"], record["replay"], "entire complete first/replay")
        exact(record["saved"], record["first"]["state"], "saved accepted checkpoint")
        if member.parameter == "large_anchor":
            same(record["initial"]["position"][0], 2**60, "original large common anchor")
            same(
                record["entry"]["503"]["local_origin_m"],
                [1 / 1024, 0, 0],
                "lossless local checkpoint",
            )
            same(
                record["entry"]["503"]["position"][0],
                record["initial"]["position"][0],
                "documented world addition rounding",
            )
        return dict(accepted=2, saved_negative="assert-only invalid restore attempt")
    if member.family == "capacity":
        warm = 0 if member.parameter == "bank0" else 1
        limited = "503" if warm == 0 else "91"
        same(
            (record["warm"], str(record["limited_world"])),
            (warm, limited),
            "both actual bank identities",
        )
        before, rejected, retry = record["before"], record["rejected"], record["retry"]
        compiled_schema(record["compiled"], before["state"])
        record_frame(
            before, WORLDS, shapes=shapes, bindings=bindings, receipt_worlds=WORLDS if warm else ()
        )
        same(len(record["rows"]), warm, "complete original warm prefix")
        for row in record["rows"]:
            record_frame(row, WORLDS, shapes=shapes, bindings=bindings, receipt_worlds=WORLDS)
        exact(
            rejected["state"][limited], before["state"][limited], "all failed world fields retained"
        )
        exact(rejected["geometry"][limited], before["geometry"][limited], "failed FK retained")
        exact(rejected["summary"][limited], before["summary"][limited], "failed energy retained")
        same(record["limited_status"][limited], "SITE_CAPACITY_EXCEEDED", "real capacity status")
        same(
            set(rejected["receipts"]),
            {str(w) for w in WORLDS} - {limited},
            "other worlds really publish",
        )
        record_frame(
            rejected,
            WORLDS,
            shapes=shapes,
            bindings=bindings,
            receipt_worlds=set(WORLDS) - {int(limited)},
        )
        record_frame(retry, WORLDS, shapes=shapes, bindings=bindings, receipt_worlds=WORLDS)
        exact(
            retry["receipts"][limited]["entry_state"],
            before["state"][limited],
            "capacity same-state retry",
        )
        for w in map(str, WORLDS):
            same(
                rejected["state"][w]["tick"], warm + int(w != limited), "selective capacity retain"
            )
            same(
                retry["state"][w]["tick"],
                rejected["state"][w]["tick"] + 1,
                "all retry worlds advance",
            )
        return dict(warm=warm, rejected_world=int(limited), retry=1)
    if member.family == "late":
        phase, bank = member.parameter.rsplit("-b", 1)
        same((record["phase"], record["warm"]), (phase, int(bank)), "actual late phase/bank")
        compiled_schema(record["compiled"], record["before"]["state"])
        same_retained(record["before"], record["failed_retained"])
        fields(
            record,
            {
                "upload": ("actual_uploaded_array",),
                "candidate": ("actual_candidate_packets", "actual_candidate_groups"),
                "receipt": ("actual_prepublication_receipts", "actual_prepublication_status"),
            }[phase],
            "actual call-through result",
        )
        exact(record["retry"], record["control"], "whole independent control/retry public record")
        snapshots(record, ("retry", "control"), (503, 91), shapes, bindings, (91,))
        return dict(phase=phase, bank=int(bank), saved_failure="actual late callback result")
    if member.family == "owner":
        same(record["variant"], member.parameter, "owner variant")
        same(record["closed_stats"]["device_owned_bytes"], 0, "closed batch resource")
        if member.parameter == "stale":
            record_frame(
                record["retry"], [503], shapes=shapes, bindings=bindings, receipt_worlds=[503]
            )
            exact(
                record["retry"]["receipts"]["503"]["entry_state"],
                record["before"]["state"]["503"],
                "stale rejected then actual retry",
            )
        else:
            exact(
                record["rebuilt"],
                record["expected"],
                "closed/new batch full same-state continuation",
            )
            snapshots(record, ("rebuilt", "expected"), (503,), shapes, bindings, (503,))
        return dict(variant=member.parameter)
    if member.family == "diagnostic":
        same(record["variant"], member.parameter, "actual diagnostic corruption identity")
        x07(record)
        snapshots(
            record, ("initial", "warm", "retry", "retained"), (503,), shapes, bindings, (503,)
        )
        saved_comparison(record["warm"]["diagnostics"]["503"], record["warm_audit"]["same_entry"])
        saved_comparison(record["retry"]["diagnostics"]["503"], record["retry_audit"]["same_entry"])
        return dict(accepted=2, rejected=1, corrupt=member.parameter)
    raise AuditError("unknown authored payload schema " + member.family)


REQUIRED_SOURCE = (
    STEP,
    CONTACT,
    TRANSACTIONS,
    REFERENCE,
    "tests/reference_newton/robot/resident_floating_reference.py",
    "tests/reference_newton/robot/resident_affine_reference.py",
    "tests/scale/test_resident_floating_step.py",
    "tests/scale/test_resident_affine_contacts.py",
    "tests/scale/test_resident_affine_contact_transactions.py",
    "tests/fixtures/urdf/floating_mimic_jaws.urdf",
    "tests/fixtures/urdf/floating_mimic_elbow.urdf",
    "src/phymeo/__init__.py",
    "src/phymeo/backends/tyx_world_batch.py",
    "src/phymeo/backends/_world_step_layout.py",
    "src/phymeo/backends/_tyx_world_kernels.py",
    "src/phymeo/backends/_resident_collision_plan.py",
    "src/phymeo/backends/_resident_contact_plan.py",
    "src/phymeo/backends/_resident_contact_bridge.py",
    "src/phymeo/backends/_resident_floating_contact.py",
    "src/phymeo/backends/_tyx_resident_floating_contact.py",
    "src/phymeo/backends/_resident_floating_tree.py",
    "src/phymeo/backends/_resident_floating_storage.py",
    "src/phymeo/backends/_tyx_resident_floating_kernels.py",
    "src/phymeo/backends/_tyx_floating_numeric.py",
    "src/phymeo/backends/_resident_floating_binding.py",
    "src/phymeo/backends/_tyx_resident_floating_bounds.py",
    "src/phymeo/backends/_resident_floating_affine_tree.py",
    "src/phymeo/backends/_resident_floating_affine_storage.py",
    "src/phymeo/backends/_tyx_resident_floating_affine.py",
    "src/phymeo/backends/_resident_floating_affine_binding.py",
    "src/phymeo/backends/_resident_tree_storage.py",
    "src/phymeo/backends/_resident_affine_storage.py",
)


def option(args, name, *, required=True):
    found = []
    for i, token in enumerate(args):
        if token == name:
            require(i + 1 < len(args), "missing argument to " + name)
            found.append(args[i + 1])
        elif token.startswith(name + "="):
            found.append(token[len(name) + 1 :])
    require(len(found) <= 1 and (bool(found) or not required), "exactly one original " + name)
    return found[0] if found else None


def keyword_predicate(expression, member):
    """Small closed grammar of exact authored function names, never eval."""
    if expression is None:
        return True
    tokens = re.findall(r"[A-Za-z_][A-Za-z_0-9]*|[()]", expression)
    same("".join(tokens), re.sub(r"\s+", "", expression), "unsupported -k syntax")
    require(len(tokens) <= 128, "bounded -k expression")
    names = {m.function for m in membership()}
    cursor = 0

    def atom():
        nonlocal cursor
        require(cursor < len(tokens), "incomplete -k expression")
        token = tokens[cursor]
        cursor += 1
        if token == "not":
            return not atom()
        if token == "(":
            value = either()
            require(cursor < len(tokens) and tokens[cursor] == ")", "unclosed -k group")
            cursor += 1
            return value
        require(token in names, "-k operands must be complete frozen function names")
        return token == member.function

    def both():
        nonlocal cursor
        value = atom()
        while cursor < len(tokens) and tokens[cursor] == "and":
            cursor += 1
            other = atom()
            value = value and other
        return value

    def either():
        nonlocal cursor
        value = both()
        while cursor < len(tokens) and tokens[cursor] == "or":
            cursor += 1
            other = both()
            value = value or other
        return value

    value = either()
    same(cursor, len(tokens), "complete exact-name -k expression")
    return value


def selection(args):
    members = membership()
    excluded, targets = [], []
    for i, arg in enumerate(args):
        if arg == "--deselect":
            require(i + 1 < len(args), "complete deselect")
            excluded.append(args[i + 1])
        elif arg.startswith("--deselect="):
            excluded.append(arg.partition("=")[2])
        elif (
            arg.startswith("tests/")
            and (arg.endswith(".py") or ".py::" in arg)
            and (i == 0 or args[i - 1] != "--deselect")
        ):
            targets.append(arg)
    valid = {value for m in members for value in (m.source, m.source + "::" + m.function, m.node)}
    require(
        targets and set(targets) <= valid and set(excluded) <= valid,
        "only original exact file/function/node targets",
    )
    keyword = option(args, "-k", required=False)
    result = []
    for target in targets:
        for member in members:
            identities = (member.source, member.source + "::" + member.function, member.node)
            if (
                target in identities
                and not set(identities).intersection(excluded)
                and keyword_predicate(keyword, member)
            ):
                result.append(member.node)
    same(len(result), len(set(result)), "no duplicated target expansion")
    return result


def inventory(root, suffix):
    """Inventory verifies declared membership, never selects identities."""
    result, pending, count = set(), [root], 0
    while pending:
        for path in pending.pop().iterdir():
            count += 1
            require(count <= MAX_FILES, "bounded directory inventory")
            if path.is_symlink():
                # pytest's current aliases are not additional evidence roots.
                require(path.name.endswith("current"), "unexpected evidence symlink")
                continue
            if path.is_dir():
                pending.append(path)
            elif path.is_file() and path.name.endswith(suffix):
                result.add(path.relative_to(root).as_posix())
    return result


class Auditor:
    def __init__(self, manifest):
        self.manifest, self.files, self.total = manifest, {}, 0
        self.coverage = {mode: {} for mode in ("interpret", "cuda")}
        self.external, self.jobs = {}, set()
        self.report = dict(
            status="incomplete",
            observer_only=True,
            runs=[],
            errors=[],
            files=self.files,
            scope="Saved source/structure/protocol/comparison consistency; no new physics pass.",
            limits=dict(
                compressed=MAX_COMPRESSED,
                payload=MAX_PAYLOAD,
                total_decoded=MAX_TOTAL,
                metadata=MAX_METADATA,
                files=MAX_FILES,
                depth=MAX_DEPTH,
            ),
            saving_boundaries=[
                "14 preflights/backend have no gzip; original test outcome only.",
                "Read-only flags and callback timing are original test assertions, not reconstructable from JSON.",
                "Invalid restore/control attempts and some intermediate failure reads are not saved separately.",
                "No FK/J/M/geometry/dynamics/constraint solve or producer import is performed.",
                "Failed-node artifacts and NOT_RUN identities do not fill successful coverage.",
            ],
        )

    def read(self, path, root, limit=MAX_METADATA):
        path, root = Path(path).resolve(strict=True), Path(root).resolve(strict=True)
        require(
            path.is_relative_to(root) and path.is_file(), "input outside declared evidence root"
        )
        require(path.stat().st_size <= limit, "input size bound " + str(path))
        with path.open("rb") as stream:
            raw = stream.read(limit + 1)
        require(len(raw) <= limit, "growing input bound")
        digest, key = hashlib.sha256(raw).hexdigest(), str(path)
        if key in self.files:
            same(self.files[key]["sha256"], digest, "same-run input changed during observation")
        else:
            self.files[key] = dict(bytes=len(raw), sha256=digest)
            self.total += len(raw)
            require(self.total <= MAX_TOTAL, "total bounded input bytes")
        require(len(self.files) <= MAX_FILES, "total processed files bound")
        return raw

    def text(self, path, root):
        return self.read(path, root).decode("utf-8-sig")

    def json(self, path, root, *, compressed=False):
        raw = self.read(path, root, MAX_COMPRESSED if compressed else MAX_METADATA)
        if compressed:
            with gzip.GzipFile(fileobj=io.BytesIO(raw)) as stream:
                raw = stream.read(MAX_PAYLOAD + 1)
            require(len(raw) <= MAX_PAYLOAD, "per-file decoded bound")
            key = str(Path(path).resolve())
            if "uncompressed_bytes" not in self.files[key]:
                self.total += len(raw) - self.files[key]["bytes"]
                require(self.total <= MAX_TOTAL, "total expanded input bound")
            self.files[key].update(
                uncompressed_bytes=len(raw), uncompressed_sha256=hashlib.sha256(raw).hexdigest()
            )
        return strict_load(raw)

    @staticmethod
    def source(sources, path):
        matches = [v for k, v in sources.items() if k.endswith("/" + path.casefold())]
        require(
            len(matches) == 1 and isinstance(matches[0], str), "missing/ambiguous source " + path
        )
        return matches[0]

    def receipt(self, spec):
        fields(
            spec,
            ("id", "kind", "mode", "directory", "config", "head", "raw_exit", "source_sha256"),
            "original ended run",
        )
        require(spec["id"] and spec["id"] not in self.jobs, "unique run identity")
        self.jobs.add(spec["id"])
        root, config_path = Path(spec["directory"]), Path(spec["config"])
        require(
            root.is_absolute() and config_path.is_absolute(), "absolute run and original job config"
        )
        root = root.resolve(strict=True)
        config = self.json(config_path, config_path.parent)
        start, final = self.json(root / "start.json", root), self.json(root / "final.json", root)
        same(start["task"], final["task"], "one ended original task")
        for name in ("owner", "exe", "args"):
            exact(start[name], final[name], "same start/final " + name)
            exact(start[name], config[name], "same original command " + name)
        same(normalized(start["cwd"]), normalized(config["workdir"]), "explicit actual cwd")
        same(start["head"], final["head"], "fixed source HEAD")
        same(start["head"], spec["head"], "declared original HEAD")
        exact(start["sources"], final["sources"], "same initial source list")
        code = int(self.text(root / "RAW_EXIT.txt", root).strip())
        same(code, final["raw_exit"], "original RAW/final exit")
        same(code, spec["raw_exit"], "manifest raw exit")
        before, after, texts = {}, {}, {}
        for row in start["sources"]:
            key = normalized(row["path"])
            require(key not in before, "unique source identity")
            raw = self.read(root / row["snapshot"], root, MAX_COMPRESSED)
            before[key] = hex_sha(row["sha256"])
            same(hashlib.sha256(raw).hexdigest(), before[key], "source snapshot original bytes")
            try:
                texts[key] = raw.decode("utf-8-sig")
            except UnicodeDecodeError:
                texts[key] = None
        for row in final["sources_after"]:
            key = normalized(row["path"])
            require(key not in after, "unique final source identity")
            after[key] = hex_sha(row["sha256"])
        exact(before, after, "all source before/after unchanged")
        exact(
            before,
            {normalized(k): hex_sha(v) for k, v in spec["source_sha256"].items()},
            "explicit complete original source set",
        )
        same(
            set(before),
            {normalized((Path(start["cwd"]) / name).resolve()) for name in config["files"]},
            "config sources relative to actual original cwd",
        )
        stdout, stderr = self.text(root / "stdout.log", root), self.text(root / "stderr.log", root)
        job = dict(
            id=spec["id"],
            kind=spec["kind"],
            mode=spec["mode"],
            head=spec["head"],
            directory=str(root),
            raw_exit=code,
            source_sha256=before,
            source_set_sha256=fingerprint(before),
            original_command=dict(exe=start["exe"], args=start["args"], workdir=start["cwd"]),
            environment=config.get("env", {}),
            stderr_bytes=len(stderr.encode()),
        )
        return root, start["args"], texts, stdout, job

    def embedded(self, value, job, root, depth=0):
        require(depth <= MAX_DEPTH, "bounded embedded source walk")
        if isinstance(value, dict):
            if {"path", "sha256"} <= value.keys() and ("utf8" in value or "text" in value):
                raw = value["utf8" if "utf8" in value else "text"].encode("utf-8")
                if "bytes" in value:
                    same(len(raw), value["bytes"], "actual complete generated source byte count")
                same(
                    hashlib.sha256(raw).hexdigest(),
                    hex_sha(value["sha256"]),
                    "complete authored URDF bytes",
                )
                path = Path(job["original_command"]["workdir"]) / value["path"]
                key = normalized(path.resolve())
                if key in job["source_sha256"]:
                    same(
                        job["source_sha256"][key],
                        value["sha256"],
                        "original static fixture source snapshot",
                    )
                else:
                    require(
                        path.resolve().is_relative_to(root), "generated fixture in actual run root"
                    )
                    same(self.read(path, root), raw, "actual generated input bytes")
            for item in value.values():
                self.embedded(item, job, root, depth + 1)
        elif isinstance(value, list):
            for item in value:
                self.embedded(item, job, root, depth + 1)

    def pytest_run(self, spec):
        root, args, sources, stdout, job = self.receipt(spec)
        require(spec["mode"] in self.coverage, "no standalone HOST group in this contract")
        same(option(args, "--tyx-mode"), spec["mode"], "actual backend argument")
        selected, completed = spec["selected"], spec["completed"]
        same(selection(args), selected, "CLI selection from fixed authored membership")
        require(selected, "nonempty actual selection")
        same(completed, selected[: len(completed)], "exact successful first prefix")
        failed = spec.get("failed")
        if failed is None:
            same(completed, selected, "complete successful group")
            same(spec["raw_exit"], 0, "successful RAW0")
        else:
            require(
                len(completed) < len(selected) and failed == selected[len(completed)],
                "exact first failure identity",
            )
            same(spec["raw_exit"], 1, "first functional failure RAW1")
            require(
                "FAILED " + failed in stdout
                and ("-x" in args or option(args, "--maxfail", required=False) == "1"),
                "actual fail-fast prefix",
            )
        members = {m.node: m for m in membership()}
        opaque = spec.get("failed_opaque_artifacts", [])
        require(isinstance(opaque, list), "explicit failed opaque artifact list")
        require(
            not opaque or failed is not None, "opaque artifacts require the actual RAW1 failed node"
        )
        opaque_by_path = {}
        for artifact in opaque:
            fields(
                artifact,
                ("path", "bytes", "sha256", "reason", "log_contains"),
                "failed opaque artifact",
            )
            same(
                set(artifact),
                {"path", "bytes", "sha256", "reason", "log_contains"},
                "exact failed opaque artifact schema",
            )
            relative = artifact["path"]
            require(
                isinstance(relative, str) and relative in members[failed].files,
                "opaque artifact is an authored failed-node path",
            )
            require(
                relative not in opaque_by_path and relative not in spec.get("failed_payloads", []),
                "no duplicate or decoded/opaque overlap",
            )
            require(
                type(artifact["bytes"]) is int and 0 <= artifact["bytes"] <= MAX_COMPRESSED,
                "bounded original opaque byte count",
            )
            hex_sha(artifact["sha256"])
            require(
                isinstance(artifact["reason"], str) and artifact["reason"].strip(),
                "opaque history reason",
            )
            exception = artifact["log_contains"]
            require(
                isinstance(exception, str)
                and 0 < len(exception.strip()) <= 4096
                and "\n" not in exception
                and "\r" not in exception,
                "one bounded original exception line",
            )
            require(
                re.match(r"^[A-Za-z_][A-Za-z_0-9.]*(?:Error|Exception):\s*\S", exception)
                is not None,
                "opaque history must name the original exception",
            )
            require(
                exception in stdout,
                "opaque artifact exception exists in original failed pytest log",
            )
            opaque_by_path[relative] = artifact
        for path in REQUIRED_SOURCE:
            self.source(sources, path)
        for node in selected:
            require(
                "def " + members[node].function + "(" in self.source(sources, members[node].source),
                "original authored function source",
            )
        lines = [line for line in stdout.splitlines() if re.search(r"\d+ (?:passed|failed)", line)]
        require(lines, "original pytest summary")
        for word, wanted in (("passed", len(completed)), ("failed", int(failed is not None))):
            match = re.search(r"(\d+) " + word, lines[-1])
            same(int(match.group(1)) if match else 0, wanted, "original pytest " + word)
        require(
            not re.search(r"[1-9]\d* (?:error|skipped|xfailed|xpassed)", lines[-1]),
            "no skipped/error coverage",
        )
        verbose_outcomes = []
        for line_number, line in enumerate(stdout.splitlines(), 1):
            outcome = re.fullmatch(
                r"(?P<node>\S+\.py::\S+)\s+(?P<outcome>PASSED|FAILED)(?:\s+\[\s*\d+%\])?",
                line.strip(),
            )
            if outcome is not None:
                verbose_outcomes.append(
                    dict(node=outcome["node"], outcome=outcome["outcome"], stdout_line=line_number)
                )
        expected_outcomes = [(node, "PASSED") for node in completed]
        if failed is not None:
            expected_outcomes.append((failed, "FAILED"))
        same(
            [(row["node"], row["outcome"]) for row in verbose_outcomes],
            expected_outcomes,
            "exact ordered unique original verbose node outcomes",
        )
        prior = self.coverage[spec["mode"]]
        repeats = set(prior).intersection(completed)
        qualifications = spec.get("qualifies_completed", {})
        same(set(qualifications), repeats, "exact necessary repeat qualifications")
        for node, link in qualifications.items():
            fields(link, ("prior_run", "reason"), "repeat qualification")
            require(
                link["prior_run"] in [r["run"] for r in prior[node]]
                and bool(link["reason"].strip()),
                "prior success and necessary repair reason",
            )
        basetemp = (Path(job["original_command"]["workdir"]) / option(args, "--basetemp")).resolve()
        same(
            normalized(basetemp),
            normalized((root / spec["basetemp"]).resolve()),
            "original basetemp resolved from original cwd",
        )
        require(basetemp.is_relative_to(root), "original pytest evidence root")
        roots, ordinals, expected_roots = spec["payload_roots"], {}, {}
        for node in selected:
            prefix = re.sub(r"[\W]", "_", node.rsplit("::", 1)[-1])[:30]
            ordinal = ordinals.get(prefix, 0)
            ordinals[prefix] = ordinal + 1
            expected_roots[node] = basetemp / (prefix + str(ordinal))
        expected_nodes = {node for node in completed if members[node].files}
        if failed and (spec.get("failed_payloads") or opaque_by_path):
            expected_nodes.add(failed)
        same(set(roots), expected_nodes, "all and only original saved node roots")
        for node, name in roots.items():
            same(
                (root / name).resolve(strict=True),
                expected_roots[node],
                "pytest prefix/ordinal identity " + node,
            )
        details, expected_files, cross = [], set(), {}
        for node in completed + ([failed] if failed else []):
            member, partial = members[node], node == failed
            files = spec.get("failed_payloads", []) if partial else list(member.files)
            require(
                len(set(files)) == len(files) and set(files) <= set(member.files),
                "original exact failure-prefix files",
            )
            detail = dict(
                node=node,
                family=member.family,
                status="failed-prefix"
                if partial
                else "pytest-only-no-save"
                if not files
                else "complete-saved",
                files=[],
            )
            for relative in files:
                base = (root / roots[node]).resolve(strict=True)
                path = (base / relative).resolve(strict=True)
                require(path.is_relative_to(base), "payload within its authored node root")
                name = path.relative_to(basetemp).as_posix()
                require(name not in expected_files, "distinct identity cannot reuse one payload")
                expected_files.add(name)
                record = self.json(path, root, compressed=True)
                result = physical_payload(member, relative, record, partial=partial)
                if "mode" in record:
                    same(record["mode"], spec["mode"], "saved original backend")
                if "stats" in record:
                    same(record["stats"]["mode"], spec["mode"], "actual batch mode")
                if not partial:
                    self.embedded(record, job, root)
                    if member.optional:
                        self.optional_reference(spec, job, record, member)
                    for source, digest in record.get("source_sha256", {}).items():
                        match = [
                            value
                            for key, value in job["source_sha256"].items()
                            if key.endswith("/" + source.casefold())
                        ]
                        same(match, [digest], "same-run helper source byte binding")
                    if member.family == "mixed" and not member.optional:
                        for world in record["initial"]["state"]:
                            physical = (
                                "state",
                                "geometry",
                                "diagnostics",
                                "summary",
                                *(("receipts",) if member.parameter == "contact" else ()),
                            )
                            tokens = [
                                fingerprint(
                                    {key: row[key][world] for key in physical if world in row[key]}
                                )
                                for row in record["rows"]
                            ]
                            key = (node, world)
                            if key in cross:
                                exact(tokens, cross[key], "complete W1/W4 same-world output")
                            cross[key] = tokens
                detail["files"].append(dict(path=str(path), result=result))
            if partial:
                detail["opaque_artifacts"] = []
                for relative, artifact in opaque_by_path.items():
                    base = (root / roots[node]).resolve(strict=True)
                    path = (base / relative).resolve(strict=True)
                    require(
                        path.is_relative_to(base),
                        "opaque artifact stays in original failed-node root",
                    )
                    name = path.relative_to(basetemp).as_posix()
                    require(
                        name not in expected_files,
                        "opaque artifact cannot replace any decoded payload",
                    )
                    expected_files.add(name)
                    raw = self.read(path, root, MAX_COMPRESSED)
                    same(len(raw), artifact["bytes"], "original failed artifact bytes")
                    same(
                        hashlib.sha256(raw).hexdigest(),
                        hex_sha(artifact["sha256"]),
                        "original failed artifact SHA",
                    )
                    detail["opaque_artifacts"].append(
                        dict(
                            node=node,
                            path=str(path),
                            status="opaque-failed-node-artifact",
                            bytes=len(raw),
                            sha256=hex_sha(artifact["sha256"]),
                            reason=artifact["reason"],
                            log_contains=artifact["log_contains"],
                            decoded=False,
                            payload_pass=False,
                            coverage_contribution=0,
                        )
                    )
            details.append(detail)
        if expected_files or basetemp.exists():
            same(
                inventory(basetemp, ".json.gz"),
                expected_files,
                "exact original saved gzip inventory",
            )
        job.update(
            records=details,
            selected=selected,
            completed=completed,
            failed=failed,
            qualifications=qualifications,
            prefix_evidence=dict(
                summary=lines[-1],
                authored_order=selected,
                successful_prefix=completed,
                first_failed=failed,
                raw_exit=spec["raw_exit"],
                verbose_outcomes=verbose_outcomes,
            ),
        )
        # Complete prior nodes require original fail-fast/log/source binding and
        # all their full payloads. The failed node, decoded or opaque, never counts.
        for node in completed:
            prior.setdefault(node, []).append(
                dict(
                    run=spec["id"],
                    source_set_sha256=job["source_set_sha256"],
                    files=len(members[node].files),
                )
            )
        self.report["runs"].append(job)

    def optional_reference(self, spec, job, record, member):
        fields(spec, ("w4_reference",), "optional original W32 dependency")
        ref = spec["w4_reference"]
        fields(ref, ("run", "path", "sha256"), "explicit same-source original W4")
        prior = [
            item
            for item in self.report["runs"]
            if item["id"] == ref["run"] and item["mode"] == spec["mode"]
        ]
        require(len(prior) == 1, "W32 references one already audited same-mode W4 run")
        source_member = next(
            m for m in membership() if m.function == "test_w1_w4_affine_contact_replay"
        )
        declared = [
            row
            for row in prior[0].get("records", ())
            if row["node"] == source_member.node and row["status"] == "complete-saved"
        ]
        require(len(declared) == 1, "W32 dependency is a complete original W1/W4 identity")
        expected = [
            f["path"]
            for f in declared[0]["files"]
            if f["path"].replace("\\", "/").endswith("/contact/w4/trace.json.gz")
        ]
        require(len(expected) == 1, "exact original W4 trace")
        path = Path(ref["path"]).resolve(strict=True)
        same(path, Path(expected[0]).resolve(), "no output-selected substitute W4")
        same(
            normalized(job["environment"].get("PHYMEO_C11_W4_TRACE", "")),
            normalized(path),
            "original explicit W32 environment binding",
        )
        reference = self.json(path, Path(prior[0]["directory"]), compressed=True)
        same(ref["sha256"], self.files[str(path)]["sha256"], "original W4 complete gzip hash")
        exact(
            reference["source_sha256"], record["source_sha256"], "same authored helper source bytes"
        )
        same(reference["mode"], spec["mode"], "same W4/W32 backend")
        same(len(reference["rows"]), 40, "complete W4 before optional four-step W32")
        same(len(reference["replay"]), 21, "complete W4 replay before W32")
        for index, row in enumerate(record["rows"]):
            for w in map(str, WORLDS):
                for name in ("state", "geometry", "diagnostics", "summary", "receipts"):
                    exact(
                        row[name][w],
                        reference["rows"][index][name][w],
                        "same original W4 first four physical records",
                    )

    def external_run(self, spec):
        root, args, sources, stdout, job = self.receipt(spec)
        require(spec["kind"] in ("native", "wheel"), "declared external producer")
        cwd = Path(job["original_command"]["workdir"])
        output = Path(spec["output_directory"])
        require(output.is_absolute(), "external original output root is absolute")
        output = output.resolve(strict=True)
        same((cwd / option(args, "--output")).resolve(), output, "actual CLI external output root")
        # The producer may run outside the repository and outside the log root.
        summary = self.json(output / "summary.json", output)
        historical = spec.get("historical", False)
        require(type(historical) is bool, "explicit history classification")
        if historical:
            require(
                spec["raw_exit"] != 0 or summary.get("status") != "passed",
                "history must retain an actual incomplete/failed attempt",
            )
            files = (
                [label + "-native.json.gz" for label, _, _, _ in NATIVE_CASES]
                if spec["kind"] == "native"
                else [WHEEL_TRACE]
            )
            saved = spec.get("saved_prefix_files", [])
            require(
                len(set(saved)) == len(saved) and set(saved) <= set(files),
                "declared original failed external prefix",
            )
            for name in saved:
                value = self.json(output / name, output, compressed=True)
                scan(value, tags=True)
            same(
                inventory(output, ".json.gz"),
                set(saved),
                "all failed external prefix files retained",
            )
            job.update(
                status="historical-prefix-only",
                output_directory=str(output),
                saved_prefix_files=saved,
                summary=summary,
                coverage_contribution=0,
            )
        else:
            same(spec["raw_exit"], 0, "external actual RAW0")
            same(summary["status"], "passed", "complete external producer")
            key = (spec["kind"], spec["mode"])
            require(
                key not in self.external, "one complete external identity per declared device/mode"
            )
            if spec["kind"] == "native":
                detail = self.native_run(spec, output, args, summary, job, sources)
            else:
                detail = self.wheel_run(spec, output, args, summary, job, sources)
            job.update(status="complete-saved", output_directory=str(output), detail=detail)
            self.external[key] = spec["id"]
        self.report["runs"].append(job)

    def native_run(self, spec, output, args, summary, job, sources):
        require(spec["mode"] in ("cpu", "cuda:0"), "pinned native actual device")
        same(option(args, "--device"), spec["mode"], "native CLI device")
        fields(
            summary,
            (
                "device",
                "channel",
                "h_s",
                "declared_cases",
                "declared_total_steps",
                "cases",
                "checks",
                "execution_error",
                "actual_cwd",
                "versions",
                "module_paths",
                "sources_before",
                "sources_after",
                "forbidden_loaded_modules",
                "gates",
                "limits",
            ),
            "complete native summary",
        )
        same(summary["device"], spec["mode"], "native actual device")
        same(summary["channel"], "native", "not matched-law integration")
        same(
            summary["declared_cases"],
            [list(item) for item in NATIVE_CASES],
            "predeclared native cases/order",
        )
        same(summary["declared_total_steps"], 9, "nine total native steps")
        same(summary["h_s"], 1 / 1024, "original native h")
        same(summary["execution_error"], None, "no native runner failure")
        same(
            normalized(summary["actual_cwd"]),
            normalized(job["original_command"]["workdir"]),
            "original native cwd",
        )
        same(summary["versions"]["newton"], "1.7.0", "pinned Newton")
        same(summary["versions"]["warp"], "1.17.0", "pinned Warp")
        same(summary["forbidden_loaded_modules"], [], "no Phymeo/TYX native oracle import")
        same(
            summary["gates"],
            dict(
                same_entry_atol=ATOL,
                rtol=RTOL,
                window_atol=WINDOW_ATOL,
                acceleration_reaction_atol=3e-4,
            ),
            "native frozen gates",
        )
        same(
            summary["limits"],
            dict(per_case_decoded_bytes=MAX_PAYLOAD, per_case_compressed_bytes=MAX_COMPRESSED),
            "native original caps",
        )
        check_records(summary["checks"], NATIVE_SUMMARY_CHECKS)
        expected_sources = {"newton/" + name for name in NATIVE_SOURCES} | {
            "driver/" + Path(NATIVE).name,
            "independent/" + Path(REFERENCE).name,
            "independent/resident_floating_reference.py",
        }
        same(
            set(summary["sources_before"]),
            expected_sources,
            "all 16 native + three independent/driver sources",
        )
        same(set(summary["sources_after"]), expected_sources, "complete native final source list")
        source_digests = {}
        for name, entry in summary["sources_before"].items():
            fields(
                entry, ("path", "snapshot", "size_bytes", "sha256"), "native original source bytes"
            )
            raw = self.read(Path(entry["snapshot"]), output)
            same(len(raw), entry["size_bytes"], "native snapshot byte count")
            digest = hashlib.sha256(raw).hexdigest()
            same(digest, hex_sha(entry["sha256"]), "native source hash")
            same(summary["sources_after"][name], digest, "native source unchanged after steps")
            same(
                job["source_sha256"].get(normalized(entry["path"])),
                digest,
                "same-run outer source binding",
            )
            source_digests[name] = digest
        for name in (
            NATIVE,
            REFERENCE,
            "tests/reference_newton/robot/resident_floating_reference.py",
        ):
            self.source(sources, name)
        same(len(summary["cases"]), 3, "all three native observations")
        details = []
        for declared, item in zip(NATIVE_CASES, summary["cases"], strict=True):
            label, name, variant, steps = declared
            same(item["case"], label, "native case identity")
            same(item["status"], "passed", "complete native case")
            same(
                [
                    item[key]
                    for key in ("declared_steps", "native_steps_completed", "complete_steps")
                ],
                [steps] * 3,
                "native and observer exact completion",
            )
            same(item["failed_checks"], [], "no failed native observation")
            same(item["execution_error"], None, "no native case exception")
            path = output / (label + "-native.json.gz")
            same(Path(item["path"]).resolve(), path.resolve(), "original fixed native payload path")
            value = self.json(path, output, compressed=True)
            file_info = self.files[str(path.resolve())]
            same(item["decoded_bytes"], file_info["uncompressed_bytes"], "native decoded bytes")
            same(item["compressed_bytes"], file_info["bytes"], "native compressed bytes")
            same(item["gzip_sha256"], file_info["sha256"], "native original gzip")
            native_record(value, declared, spec["mode"], source_digests)
            details.append(dict(case=label, path=str(path), actual_steps=steps))
        same(
            inventory(output, ".json.gz"),
            {label + "-native.json.gz" for label, _, _, _ in NATIVE_CASES},
            "exact native gzip membership",
        )
        return dict(
            records=details,
            actual_native_steps=9,
            scope="Native Featherstone/contact observation; no matched contact force and no new numerical solve.",
        )

    def archive(self, identity, expected, package):
        fields(
            identity,
            (
                "path",
                "bytes",
                "sha256",
                "direct_url",
                "installed_from",
                "version",
                "installer_recorded_sha256",
                "installed_product_files_verified",
                "members",
                "metadata_members",
            ),
            "actual installed archive identity",
        )
        expected, package = Path(expected).resolve(strict=True), Path(package).resolve(strict=True)
        same(Path(identity["path"]).resolve(), expected, "explicit expected wheel archive")
        raw = self.read(expected, expected.parent, MAX_COMPRESSED)
        same(identity["bytes"], len(raw), "archive original byte count")
        same(identity["sha256"], hashlib.sha256(raw).hexdigest(), "archive actual SHA")
        url = identity["direct_url"]["url"]
        same(identity["installed_from"], url, "installer exact URL")
        parsed = urlsplit(url)
        require(
            parsed.scheme == "file" and parsed.netloc in ("", "localhost"),
            "local explicit archive URL",
        )
        same(Path(url2pathname(parsed.path)).resolve(), expected, "direct URL actual archive")
        if identity["installer_recorded_sha256"] is not None:
            same(
                identity["installer_recorded_sha256"],
                identity["sha256"],
                "optional installer hash when supplied",
            )
        with ZipFile(io.BytesIO(raw)) as archive:
            names = archive.namelist()
            same(len(names), len(set(names)), "unique wheel members")
            require(len(names) <= MAX_FILES, "bounded wheel inventory")
            product = {
                name for name in names if name.startswith("phymeo/") and not name.endswith("/")
            }
            recorded = {row["name"]: row for row in identity["members"]}
            same(len(recorded), len(identity["members"]), "unique recorded package members")
            same(set(recorded), product, "complete wheel product members")
            metadata = {row["name"]: row for row in identity["metadata_members"]}
            require(
                metadata
                and all(
                    name.rsplit("/", 1)[-1] in ("METADATA", "WHEEL", "top_level.txt")
                    for name in metadata
                ),
                "recorded distribution metadata",
            )
            require(
                any(name.endswith("/METADATA") for name in metadata)
                and any(name.endswith("/WHEEL") for name in metadata),
                "required wheel metadata",
            )
            for name, row in {**recorded, **metadata}.items():
                info = archive.getinfo(name)
                require(info.file_size <= MAX_METADATA, "bounded installed member")
                content = archive.read(info)
                same(row["bytes"], len(content), "original member bytes")
                same(row["sha256"], hashlib.sha256(content).hexdigest(), "original member digest")
                installed = package.parent / name
                same(
                    self.read(installed, package.parent),
                    content,
                    "actual installed/archive member parity",
                )
        actual = inventory(package, "")
        actual = {"phymeo/" + name for name in actual}
        same(actual, product, "no missing or extra installed product files")
        same(
            identity["installed_product_files_verified"],
            len(product),
            "full installed member count",
        )
        return recorded

    def wheel_run(self, spec, output, args, summary, job, sources):
        require(spec["mode"] in ("interpret", "cuda"), "actual wheel mode")
        same(option(args, "--mode"), spec["mode"], "wheel CLI actual mode")
        require("-I" in args and "-B" in args, "isolated no-bytecode original wheel invocation")
        package = Path(option(args, "--expected-package-root")).resolve(strict=True)
        archive_path = Path(option(args, "--expected-wheel")).resolve(strict=True)
        forbidden = Path(option(args, "--forbidden-checkout")).resolve(strict=True)
        same(summary["protocol"], WHEEL_PROTOCOL, "original wheel protocol")
        same(summary["raw_exit"], 0, "wheel original raw exit")
        same(summary["mode"], spec["mode"], "actual saved mode")
        same(summary["errors"], [], "no physics/IO/cleanup error")
        trace = summary["trace"]
        fields(
            trace,
            (
                "path",
                "complete",
                "prefix_only",
                "uncompressed_bytes",
                "uncompressed_sha256",
                "compressed_bytes",
                "sha256",
            ),
            "complete saved wheel trace",
        )
        require(
            trace["complete"] is True and trace["prefix_only"] is False,
            "archive completeness is not inferred from RAW alone",
        )
        path = output / WHEEL_TRACE
        same(Path(trace["path"]).resolve(), path.resolve(), "fixed wheel gzip identity")
        value = self.json(path, output, compressed=True)
        info = self.files[str(path.resolve())]
        for key in ("uncompressed_bytes", "uncompressed_sha256"):
            same(trace[key], info[key], "wheel decoded trace " + key)
        same(trace["compressed_bytes"], info["bytes"], "wheel compressed trace bytes")
        same(trace["sha256"], info["sha256"], "wheel original gzip bytes")
        scan(value)
        same(value["protocol"], WHEEL_PROTOCOL, "trace protocol")
        same(value["status"], "passed", "whole wheel protocol passed")
        same(value["mode"], spec["mode"], "trace actual backend")
        same(value["errors"], [], "no hidden wheel error")
        same(value["world_order"], list(WORLDS), "original mixed-world ordering")
        same(value["expected_package_root"], str(package), "explicit installed root")
        same(
            normalized(value["forbidden_checkout"]),
            normalized(forbidden),
            "checkout exclusion declaration",
        )
        same(
            normalized(value["cwd"]),
            normalized(job["original_command"]["workdir"]),
            "real outside cwd",
        )
        for candidate in (Path(value["cwd"]), output, Path(value["script"]["path"]), package):
            require(
                not candidate.resolve().is_relative_to(forbidden), "outside-checkout provenance"
            )
        require(not output.is_relative_to(package), "outputs do not modify installed package")
        same(
            Path(value["script"]["path"]).resolve().parent,
            Path(value["cwd"]).resolve(),
            "original copied consumer cwd",
        )
        same(
            Path(option(args, "--wheel-site")).resolve(),
            package.parent,
            "actual explicit wheel installation",
        )
        for directory in (Path(value["cwd"]), output, package.parent, package):
            require(
                not any((parent / ".git").exists() for parent in (directory, *directory.parents)),
                "all declared execution/install paths outside any checkout",
            )
        script = value["script"]
        raw = script["source_utf8"].encode("utf-8")
        same(len(raw), script["bytes"], "complete external script bytes")
        same(hashlib.sha256(raw).hexdigest(), script["sha256"], "complete consumer source hash")
        same(
            job["source_sha256"].get(normalized(script["path"])),
            script["sha256"],
            "same-run copied consumer source",
        )
        same(
            self.source(sources, Path(WHEEL).name),
            script["source_utf8"],
            "original copied consumer source text",
        )
        fields(
            value["versions"],
            ("phymeo", "tyx", "numpy", "python"),
            "actual installed runtime versions",
        )
        same(
            value["versions"]["phymeo"],
            value["wheel_before"]["version"],
            "installed distribution version",
        )
        exact(value["wheel_before"], value["wheel_after"], "unchanged installed archive identity")
        exact(summary["wheel"], value["wheel_before"], "summary same original wheel")
        members = self.archive(value["wheel_before"], archive_path, package)
        same(
            job["source_sha256"].get(normalized(archive_path)),
            value["wheel_before"]["sha256"],
            "same-run binary wheel source snapshot",
        )
        for key in ("modules_before", "modules_after"):
            rows = value[key]
            require(
                rows and len({row["module"] for row in rows}) == len(rows),
                "complete unique loaded-module list",
            )
            for row in rows:
                fields(
                    row,
                    (
                        "module",
                        "path",
                        "origin",
                        "package_paths",
                        "wheel_member",
                        "bytes",
                        "sha256",
                    ),
                    "loaded wheel module",
                )
                require(
                    row["module"] == "phymeo" or row["module"].startswith("phymeo."),
                    "only package modules",
                )
                name = row["wheel_member"]
                require(name in members, "loaded module is a verified archive member")
                path = Path(row["path"]).resolve(strict=True)
                require(path.is_relative_to(package), "loaded module installed origin")
                same(path.relative_to(package.parent).as_posix(), name, "exact module/archive path")
                same(Path(row["origin"]).resolve(), path, "actual import spec origin")
                for directory in row["package_paths"]:
                    require(
                        Path(directory).resolve().is_relative_to(package),
                        "package search path within wheel",
                    )
                for field in ("bytes", "sha256"):
                    same(row[field], members[name][field], "loaded member " + field)
        same(
            summary["verified_module_count"],
            len(value["modules_after"]),
            "actual loaded module count",
        )
        same(value["gates"], WHEEL_GATES, "all original wheel engineering gates")
        same(value["fixed_membership"], WHEEL_MEMBERSHIP, "fixed wheel subwindows")
        exact(summary["gates"], value["gates"], "summary gates")
        exact(summary["fixed_membership"], value["fixed_membership"], "summary membership")
        result = wheel_payload(value)
        for name, detail in result.items():
            saved = summary["windows"][name]
            for key, number_value in detail.items():
                same(saved[key], number_value, "same complete window summary " + name + "/" + key)
        self.embedded(value["w4"]["sources"], job, output)
        self.embedded(value["a04_p1"]["source"], job, output)
        same(inventory(output, ".json.gz"), {WHEEL_TRACE}, "one complete wheel gzip")
        return dict(
            windows=result,
            modules=len(value["modules_after"]),
            installed_files=len(members),
            scope="Complete public installed consumer records; not a new benchmark or solver run.",
        )


NATIVE_CASES = (
    ("A01-motor", "A01", None, 1),
    ("A03-fan", "A03", "fan", 4),
    ("A05-native-contact", "A05", None, 4),
)
NATIVE_SOURCES = (
    "_src/sim/builder.py",
    "_src/sim/model.py",
    "_src/sim/state.py",
    "_src/sim/articulation.py",
    "_src/sim/collide.py",
    "_src/sim/contacts.py",
    "_src/sim/joint_mimic.py",
    "_src/sim/control.py",
    "_src/sim/enums.py",
    "_src/geometry/broad_phase_common.py",
    "_src/geometry/collision_primitive.py",
    "_src/geometry/contact_data.py",
    "_src/geometry/narrow_phase.py",
    "_src/solvers/semi_implicit/kernels_contact.py",
    "_src/solvers/featherstone/solver_featherstone.py",
    "_src/solvers/featherstone/kernels.py",
)
NATIVE_SUMMARY_CHECKS = (
    "all_sources_unchanged",
    "no_Phymeo_TYX_import",
    "exact_declared_case_order",
    "all_nine_native_steps_completed",
)
NATIVE_RAW = ("joint_q", "joint_qd", "body_q", "body_qd", "body_f", "body_parent_f")
NATIVE_MODEL = tuple(
    (
        "body_mass body_inv_mass body_com body_inertia body_inv_inertia body_flags body_world "
        "joint_type joint_parent joint_child joint_X_p joint_X_c joint_axis joint_dof_dim joint_q_start joint_qd_start "
        "joint_armature joint_damping joint_friction joint_target_ke joint_target_kd joint_limit_ke joint_limit_kd "
        "joint_limit_lower joint_limit_upper joint_enabled joint_world joint_articulation articulation_start articulation_end gravity "
        "joint_mimic_joint joint_mimic_coeffs joint_effort_limit joint_velocity_limit joint_target_mode joint_target_q_start "
        "body_label joint_label shape_label body_count joint_count joint_dof_count joint_coord_count shape_count articulation_count world_count actual_device"
    ).split()
)
NATIVE_SHAPES = tuple(
    (
        "shape_type shape_body shape_world shape_transform shape_scale shape_flags shape_margin shape_gap "
        "shape_collision_group shape_collision_filter_pairs shape_contact_pairs shape_material_ke shape_material_kd shape_material_kf "
        "shape_material_ka shape_material_mu shape_material_mu_rolling shape_material_mu_torsional"
    ).split()
)
NATIVE_SCRATCH = tuple(
    (
        "joint_qdd joint_tau joint_qd_internal_in joint_qd_internal_out joint_f_internal joint_S_s body_qd_fk "
        "body_q_com body_solve_origin body_I_s body_v_s body_a_s body_f_s body_f_ext body_ft_s H J P L M H_mimic joint_tau_mimic "
        "articulation_J_start articulation_J_rows articulation_J_cols articulation_H_start articulation_H_rows articulation_dof_start "
        "articulation_M_start articulation_M_rows joint_armature_effective joint_armature_zero"
    ).split()
)
NATIVE_CONTACT = tuple(
    (
        "rigid_contact_point_id rigid_contact_shape0 rigid_contact_shape1 rigid_contact_point0 rigid_contact_point1 "
        "rigid_contact_offset0 rigid_contact_offset1 rigid_contact_normal rigid_contact_margin0 rigid_contact_margin1 "
        "rigid_contact_surface_velocity rigid_contact_stiffness rigid_contact_damping rigid_contact_friction rigid_contact_tids rigid_contact_force"
    ).split()
)
NATIVE_OBSERVATION_CHECKS = tuple(
    (
        "full_M full_J body_positions body_rotations body_COM body_linear_velocity body_angular_velocity "
        "linear_momentum angular_momentum kinetic_energy public_configuration_identity_anchors public_velocity_basis full_q_graph "
        "full_v_graph original_rows_position original_rows_velocity all_angular_components_zero root_quaternion"
    ).split()
)
NATIVE_MODEL_CHECKS = tuple(
    (
        "frozen_h root_inertia_authored_axes zero_independent_PD zero_external_joint_effort actual_device_exact "
        "counts_B_joints_D_Q_articulations_worlds full_body_mass full_body_com full_body_inertia full_types full_parent full_child "
        "one_articulation_all_rows full_q_starts full_qd_starts full_parent_offsets identity_parent_rotations identity_child_anchors "
        "child_axes mimic_leaders mimic_coefficients zero_joint_armature zero_joint_damping zero_joint_friction zero_joint_target_ke "
        "zero_joint_target_kd zero_joint_limit_ke zero_joint_limit_kd declared_joint_limit_lower declared_joint_limit_upper "
        "declared_joint_velocity_limit declared_joint_effort_limit gravity body_world_identity joint_world_identity actual_shape_count "
        "initial_full_q initial_full_v completed_native_steps completed_observer_steps step_device_exact"
    ).split()
)
NATIVE_STEP_CHECKS = tuple(
    (
        "only_authored_motor no_root_motor only_authored_external_wrench native_contact_capacity "
        "native_scratch_entry_twists native_scratch_entry_COM native_solve_origin_root_COM native_internal_velocity_basis "
        "native_output_internal_velocity_basis actual_post_RNEA_wrench full_load_native_vs_independent_J full_load_native_vs_declared "
        "internal_full_M_basis internal_full_J_basis internal_full_tau_basis native_mimic_full_padded_H native_mimic_full_padded_tau "
        "actual_delta_v_acceleration decoded_internal_acceleration actual_full_reaction actual_original_row_lambda actual_original_row_representation "
        "actual_root_six_balance actual_reduced_balance actual_original_row_acceleration actual_original_row_power independent_window_full_q "
        "independent_window_full_v independent_window_kinetic_energy actual_linear_momentum_balance actual_angular_momentum_balance "
        "actual_discrete_kinetic_balance native_contact_count native_contact_body_wrench"
    ).split()
)


def value_shape(value):
    if not isinstance(value, list):
        number(value)
        return []
    if not value:
        return [0]
    tail = value_shape(value[0])
    require(all(value_shape(item) == tail for item in value), "rectangular saved check array")
    return [len(value), *tail]


def check_records(values, required=()):
    fields(values, required, "required independent native checks")
    require(values, "nonempty native check library")
    for name, row in values.items():
        fields(
            row,
            (
                "passed",
                "actual",
                "expected",
                "atol",
                "rtol",
                "max_abs_error",
                "actual_shape",
                "expected_shape",
            ),
            "full native check " + name,
        )
        require(row["passed"] is True, "original native check failed: " + name)
        require(
            0 <= number(row["atol"]) <= 3e-4 and 0 <= number(row["rtol"]) <= RTOL,
            "original native engineering tolerance bound",
        )
        same(value_shape(row["actual"]), row["actual_shape"], "native actual saved shape")
        same(value_shape(row["expected"]), row["expected_shape"], "native expected saved shape")
        same(row["actual_shape"], row["expected_shape"], "native comparison dimensions")
        close(
            row["actual"],
            row["expected"],
            "saved native independent comparison " + name,
            atol=row["atol"],
            rtol=row["rtol"],
        )
        require(number(row["max_abs_error"]) >= 0, "saved native error is quantitative")


def required_values(value, names, label):
    fields(value, names, label)
    require(
        all(value[name] is not None for name in names),
        label + " cannot silently substitute missing metadata",
    )


def native_raw(value, n):
    fields(value, NATIVE_RAW, "complete native public raw state")
    b, d = n + 1, n + 6
    for name, shape in (
        ("joint_q", (n + 7,)),
        ("joint_qd", (d,)),
        ("body_q", (b, 7)),
        ("body_qd", (b, 6)),
        ("body_f", (b, 6)),
    ):
        array(value[name], *shape)
    if value["body_parent_f"] is not None:
        array(value["body_parent_f"], b, 6)


def native_observation(value, n, mode):
    fields(
        value,
        (
            "raw",
            "raw_dtypes",
            "actual_device",
            "centers_world_m",
            "parent_anchor_position_m",
            "parent_anchor_rotation",
            "public_native_jacobian",
            "public_native_mass",
            "world_com_to_public_joint_velocity",
            "world_com_to_internal_anchor_velocity",
            "jacobian_world_com",
            "mass_world_com",
            "full_position",
            "full_velocity",
            "linear_momentum_kg_m_s",
            "angular_momentum_world_origin_kg_m2_s",
            "kinetic_energy_j",
        ),
        "complete native observation",
    )
    same(value["actual_device"], mode, "native actual state device")
    native_raw(value["raw"], n)
    d, b = n + 6, n + 1
    for name, shape in (
        ("centers_world_m", (b, 3)),
        ("parent_anchor_position_m", (3,)),
        ("parent_anchor_rotation", (3, 3)),
        ("public_native_jacobian", (b, 6, d)),
        ("jacobian_world_com", (b, 6, d)),
        ("full_position", (n + 7,)),
        ("full_velocity", (d,)),
        ("linear_momentum_kg_m_s", (3,)),
        ("angular_momentum_world_origin_kg_m2_s", (3,)),
    ):
        array(value[name], *shape)
    for name in (
        "public_native_mass",
        "mass_world_com",
        "world_com_to_public_joint_velocity",
        "world_com_to_internal_anchor_velocity",
    ):
        array(value[name], d, d)
    number(value["kinetic_energy_j"])
    required_values(value["raw_dtypes"], NATIVE_RAW[:-1], "actual native state dtypes")


def native_record(value, declared, mode, source_digests):
    scan(value)
    label, name, variant, steps = declared
    fields(
        value,
        (
            "case",
            "channel",
            "status",
            "h_s",
            "declared_steps",
            "native_steps_completed",
            "complete_steps",
            "trace",
            "checks",
            "execution_error",
            "source_binding",
            "semantics",
            "declaration",
            "model",
            "role_ids",
            "solver",
            "pipeline",
            "initial_state",
            "final_state",
            "failed_checks",
            "first_failed_check",
        ),
        "complete native case",
    )
    same(value["case"], label, "native label")
    same(value["channel"], "native", "native contact path")
    same(value["status"], "passed", "native actual completed case")
    same(value["h_s"], 1 / 1024, "native h")
    same(
        [value[field] for field in ("declared_steps", "native_steps_completed", "complete_steps")],
        [steps] * 3,
        "exact native counts",
    )
    same(value["execution_error"], None, "no native exception")
    same(value["failed_checks"], [], "no native failed checks")
    same(value["first_failed_check"], None, "no unreported first observation failure")
    exact(value["source_binding"]["sources"], source_digests, "same native source bytes")
    same(value["source_binding"]["device"], mode, "same native source device")
    same(
        value["semantics"]["zero_matched_contact_force"], True, "no matched force native substitute"
    )
    data = value["declaration"]
    n, k = input_schema(data)
    same(
        (data["name"], data["variant"]), (name, variant), "original native declared physical input"
    )
    same(data["h"], 1 / 1024, "native input h")
    same((n, k), (3, 1) if name == "A03" else (2, 1), "native physical graph")
    meta = value["model"]
    required_values(meta, NATIVE_MODEL, "full native model metadata")
    fields(meta, NATIVE_SHAPES, "native shape metadata including optional fields")
    if name == "A05":
        required_values(
            meta,
            tuple(key for key in NATIVE_SHAPES if key != "shape_contact_pairs"),
            "native contact metadata",
        )
    same(
        [
            meta[field]
            for field in (
                "body_count",
                "joint_count",
                "joint_dof_count",
                "joint_coord_count",
                "articulation_count",
                "world_count",
            )
        ],
        [n + 1, n + 1, n + 6, n + 7, 1, 1],
        "native complete coordinate/body counts",
    )
    same(meta["actual_device"], mode, "native model actual device")
    same(meta["shape_count"], 2 if name == "A05" else 0, "real native shape count")
    solver = value["solver"]
    for key, expected in dict(
        name="SolverFeatherstone",
        angular_damping=0,
        update_mass_matrix_interval=1,
        use_tile_gemm=False,
        fuse_cholesky=True,
        requires_grad=False,
    ).items():
        same(solver[key], expected, "original native solver " + key)
    number(solver["friction_smoothing"])
    same(
        value["pipeline"],
        dict(
            active=name == "A05",
            broad_phase="nxn" if name == "A05" else None,
            rigid_contact_max=8 if name == "A05" else 0,
            soft_contact_max=0,
            deterministic=True,
            contact_matching="disabled",
        ),
        "actual native collision channel",
    )
    native_observation(value["initial_state"], n, mode)
    native_observation(value["final_state"], n, mode)
    checks = list(NATIVE_MODEL_CHECKS) + ["initial_" + key for key in NATIVE_OBSERVATION_CHECKS]
    if name == "A05":
        checks += [
            "actual_sphere_types",
            "actual_shape_owners",
            "original_radii",
            "sphere_local_pose",
            "shape_margin",
            "shape_gap",
            "siblings_not_filtered",
            "shape_collision_enabled",
        ]
        checks += [
            "material_" + key
            for key in ("ke", "kd", "kf", "ka", "mu", "mu_rolling", "mu_torsional")
        ]
    check_records(value["checks"], checks)
    same(len(value["trace"]), steps, "complete native trace")
    previous = value["initial_state"]
    for index, row in enumerate(value["trace"], 1):
        fields(
            row,
            (
                "step",
                "entry_time_s",
                "exit_time_s",
                "entry",
                "exit",
                "original_control",
                "original_external_body_wrench",
                "matched_contact_wrench_supplied",
                "independent_same_entry",
                "independent_window",
                "native_step_submitted",
                "native_step_completed",
                "checks_complete",
                "native_contacts",
                "native_return",
                "poststep_input_raw",
                "poststep_output_raw",
                "solver_scratch",
                "contact_observation",
                "internal_basis_expected",
                "full_balance_observation",
                "energy_observation",
                "checks",
            ),
            "native full step",
        )
        same(row["step"], index, "native contiguous step")
        same(
            (row["entry_time_s"], row["exit_time_s"]),
            ((index - 1) / 1024, index / 1024),
            "actual native sampling times",
        )
        require(
            row["native_step_submitted"] is True
            and row["native_step_completed"] is True
            and row["checks_complete"] is True,
            "both actual advance and observer completed",
        )
        same(row["matched_contact_wrench_supplied"], False, "no injected contact replacement")
        native_observation(row["entry"], n, mode)
        native_observation(row["exit"], n, mode)
        for field in ("full_position", "full_velocity"):
            exact(row["entry"][field], previous[field], "native state continuity")
        for field in ("joint_q", "joint_qd", "body_q", "body_qd"):
            exact(row["entry"]["raw"][field], previous["raw"][field], "native raw state continuity")
        for field in ("poststep_input_raw", "poststep_output_raw"):
            native_raw(row[field], n)
        fields(
            row["original_control"],
            ("joint_f", "joint_target_q", "joint_target_qd", "joint_act"),
            "complete original native control",
        )
        array(row["original_external_body_wrench"], n + 1, 6)
        terms = row["independent_same_entry"]
        for field in (
            "mass",
            "bias",
            "external",
            "linear_jacobian",
            "angular_jacobian",
            "reduced_mass",
            "reduced_bias",
            "reduced_external",
            "full_acceleration",
            "acceleration",
            "drive",
            "constraint_effort_full",
            "row_multipliers",
            "row_powers_w",
            "constraint_power_w",
            "root_constraint_residual",
        ):
            require(field in terms, "native saved independent " + field)
        fields(
            row["independent_window"],
            ("entry_q", "entry_v", "exit_q", "exit_v", "entry_terms"),
            "independent native window",
        )
        scratch = row["solver_scratch"]
        required_values(scratch["raw"], NATIVE_SCRATCH, "all actual native solver scratch")
        fields(
            scratch,
            (
                "raw_dtypes",
                "decoded_com_world_wrench",
                "decoded_public_world_com_acceleration",
                "decoded_com_world_jacobian",
                "decoder",
                "timing",
                "body_parent_f_semantics",
            ),
            "complete RNEA decoder",
        )
        same(
            scratch["decoder"],
            "F=-rawF; tauCOM=-rawTau-(body_q_com.xyz-body_solve_origin) cross F",
            "actual solve-origin COM decoder",
        )
        same(
            scratch["body_parent_f_semantics"],
            "recursive joint wrench, never original-row mimic lambda",
            "recursive wrench is not original-row reaction",
        )
        array(scratch["decoded_com_world_wrench"], n + 1, 6)
        array(scratch["decoded_public_world_com_acceleration"], n + 6)
        array(scratch["decoded_com_world_jacobian"], n + 1, 6, n + 6)
        contact = row["native_contacts"]
        same(
            (
                contact["count"],
                contact["capacity"],
                contact["pipeline_active"],
                contact["valid_capacity"],
            ),
            (1, 8, True, True) if name == "A05" else (0, 0, False, True),
            "real native finite contact rows",
        )
        fields(contact["rows"], NATIVE_CONTACT, "all native contact buffers")
        if name == "A05":
            required_values(
                contact["rows"],
                (
                    "rigid_contact_shape0",
                    "rigid_contact_shape1",
                    "rigid_contact_point0",
                    "rigid_contact_point1",
                    "rigid_contact_normal",
                    "rigid_contact_margin0",
                    "rigid_contact_margin1",
                ),
                "measured native contact geometry",
            )
            same(
                contact["force_buffer_semantics"],
                "unused by Featherstone; not measured contact force",
                "unused force buffer not promoted",
            )
        fields(
            row["contact_observation"],
            ("rows", "dynamic_body_wrench_world_com", "attribution"),
            "decoded actual native contact force",
        )
        array(row["contact_observation"]["dynamic_body_wrench_world_com"], n + 1, 6)
        fields(
            row["internal_basis_expected"],
            (
                "full_mass",
                "full_rhs",
                "padded_reduced_mass",
                "padded_reduced_rhs",
                "active_indices",
                "inactive_follower_indices",
                "acceleration_correction",
            ),
            "explicit native basis comparison",
        )
        balance = row["full_balance_observation"]
        for field, shape in (
            ("acceleration_from_actual_delta_v", (n + 6,)),
            ("full_generalized_applied", (n + 6,)),
            ("full_reaction", (n + 6,)),
            ("row_multipliers", (n - k,)),
            ("row_powers_w", (n - k,)),
            ("root_constraint_residual", (6,)),
        ):
            array(balance[field], *shape)
        number(balance["constraint_power_w"])
        require(balance["source"], "independent original-row balance source")
        for field in (
            "actual_delta_kinetic_j",
            "endpoint_force_work_j",
            "implicit_loss_j",
            "balance_j",
        ):
            number(row["energy_observation"][field])
        required_checks = list(NATIVE_STEP_CHECKS) + [
            prefix + "_" + key for prefix in ("entry", "exit") for key in NATIVE_OBSERVATION_CHECKS
        ]
        if name == "A05":
            required_checks += [
                "native_pair_owners",
                "native_shape_identity",
                "native_normal",
                "native_gap",
                "native_force_magnitude",
                "native_surface_0",
                "native_surface_1",
                "native_skin_radius_0",
                "native_skin_radius_1",
            ]
        check_records(row["checks"], required_checks)
        if index == 1:
            require(row.get("literal_first_step"), "independent literal first-step anchors")
        previous = row["exit"]
    for field in ("full_position", "full_velocity"):
        exact(value["final_state"][field], previous[field], "native true final state")


WHEEL_PROTOCOL = "c11-installed-floating-affine-v1"
WHEEL_TRACE = "resident-floating-affine-wheel.trace.json.gz"
WHEEL_GATES = dict(
    h_w4_s=1 / 1024,
    h_a04_s=1 / 4096,
    all_delay_steps=0,
    state_atol=ATOL,
    state_rtol=RTOL,
    window_atol=WINDOW_ATOL,
    full_mass_reaction_atol=2e-5,
    equation_residual=2e-5,
    control_residual=5e-5,
    material_record_construction_atol_m=1e-10,
    material_work_atol_j=1e-12,
    material_work_rtol=1e-10,
    payload_bytes_cap=MAX_PAYLOAD,
    compressed_bytes_cap=MAX_COMPRESSED,
)
WHEEL_MEMBERSHIP = dict(
    w4=dict(
        worlds=list(WORLDS),
        forward_steps=40,
        restore_tick=19,
        replay_steps=21,
        capacity_banks=[0, 1],
        capacity_worlds=[503, 91],
        maximum_alignment_steps=2,
    ),
    a04_p1=dict(worlds=[503], steps=4),
)
BROAD_PHYSICAL = (
    "world_keys",
    "ticks",
    "shape_keys",
    "shape_revisions",
    "bounds",
    "registered_pairs",
    "candidates",
)


def material_key_string(key):
    world, body, shape = key
    return (
        "CollisionKey(world=" + str(world) + ", body=" + str(body) + ", shape=" + str(shape) + ")"
    )


def broad_schema(value, states, shapes, *, saved_oracle=None):
    fields(
        value,
        ("topology_generation", "state_epoch", *BROAD_PHYSICAL),
        "complete current finite broad phase",
    )
    worlds = [int(w) for w in states]
    same(set(value["world_keys"]), set(worlds), "broad-phase world ownership")
    same(len(value["world_keys"]), len(worlds), "unique broad-phase worlds")
    same(len(value["ticks"]), len(worlds), "all broad-phase ticks")
    for w, tick in zip(value["world_keys"], value["ticks"], strict=True):
        same(tick, states[str(w)]["tick"], "bounds actual accepted tick")
    keys = [key_tuple(key) for key in value["shape_keys"]]
    same(set(keys), set(shapes), "all original finite source shapes")
    same(len(keys), len(shapes), "unique shape rows")
    same(len(value["shape_revisions"]), len(keys), "all original shape revisions")
    array(value["bounds"], len(keys), 6)
    for index, key in enumerate(keys):
        same(value["shape_revisions"][index], shapes[key]["revision"], "source revision")
        require(
            all(value["bounds"][index][i] <= value["bounds"][index][i + 3] for i in range(3)),
            "finite bounds interval",
        )

    def pairs(rows):
        result = []
        for row in rows:
            same(len(row), 2, "whole original pair")
            a, b = (key_tuple(key) for key in row)
            require(
                a < b and a in shapes and b in shapes and a[0] == b[0] and a[1] != b[1],
                "canonical same-world different-body pair",
            )
            result.append((a, b))
        same(len(result), len(set(result)), "unique full pair list")
        return set(result)

    registered, candidates = pairs(value["registered_pairs"]), pairs(value["candidates"])
    require(candidates <= registered, "candidate membership in explicit registered pairs")
    if saved_oracle is not None:
        fields(
            saved_oracle,
            ("material_bounds", "true_pairs"),
            "saved independent original material bounds",
        )
        same(
            set(saved_oracle["material_bounds"]),
            {material_key_string(key) for key in shapes},
            "all source-material independent bounds",
        )
        for index, key in enumerate(keys):
            low, high = saved_oracle["material_bounds"][material_key_string(key)]
            array(low, 3)
            array(high, 3)
            require(
                all(
                    value["bounds"][index][i] <= low[i] and value["bounds"][index][i + 3] >= high[i]
                    for i in range(3)
                ),
                "stored device bounds contain saved independent material bounds",
            )
        true_pairs = pairs(saved_oracle["true_pairs"])
        require(true_pairs and true_pairs <= candidates, "no missing saved true candidate")


def wheel_capture(
    value, worlds, *, shapes=None, bindings=None, wanted_receipts=(), old_receipts=False
):
    fields(
        value,
        ("state", "geometry", "diagnostics", "energy", "batch_stats", "receipts", "status"),
        "complete installed public capture",
    )
    # The producer names its unchanged batch.summary() mapping 'energy'. This
    # validation view does not remove/normalize any archived field or value.
    view = dict(value, summary=value["energy"])
    specs = record_frame(
        view,
        worlds,
        shapes=shapes,
        bindings=bindings,
        receipt_worlds=wanted_receipts,
        allow_old_receipts=old_receipts,
    )
    if shapes is not None:
        fields(value, ("broad", "plan_stats"), "complete installed contact capture")
        broad_schema(value["broad"], value["state"], shapes)
        same(
            value["broad"]["state_epoch"],
            value["batch_stats"]["state_epoch"],
            "current device epoch",
        )
    for state in value["state"].values():
        same(state["history"], [], "all frozen wheel delays are zero")
    return specs


def wheel_row(row, worlds, controls, *, h, shapes=None, bindings=None, rejected_world=None):
    fields(
        row,
        (
            "label",
            "input",
            "entry",
            "entry_geometry",
            "entry_energy",
            "entry_stats",
            "entry_receipts",
            "h_s",
            "expected_acceptance",
            "acceptance",
            "audit",
            "independent",
            "checks_complete",
        ),
        "all original installed step fields",
    )
    require(
        "failure" not in row and row["checks_complete"] is True,
        "no accepted-but-unchecked row promoted to complete",
    )
    same(row["h_s"], h, "original wheel step h")
    world_keys = {str(w) for w in worlds}
    flags = {str(w): w != rejected_world for w in worlds}
    exact(row["acceptance"], flags, "actual per-world acceptance")
    exact(row["expected_acceptance"], flags, "declared per-world expectation")
    exact(row["input"], controls, "same full original commands")
    state_map(row["entry"], worlds)
    same(set(row["entry_geometry"]), world_keys, "all wheel entry geometry")
    same(set(row["entry_energy"]), world_keys, "all wheel entry energy")
    wanted = [w for w in worlds if w != rejected_world] if shapes is not None else []
    wheel_capture(row, worlds, shapes=shapes, bindings=bindings, wanted_receipts=wanted)
    accepted = {w for w, yes in flags.items() if yes}
    same(set(row["audit"]), accepted, "independent same-entry audit for every accepted world")
    same(set(row["independent"]), accepted, "independent trajectory for every accepted world")
    for w in world_keys:
        before, after = row["entry"][w], row["state"][w]
        geometry(row["entry_geometry"][w], before)
        number(row["entry_energy"][w]["kinetic_energy_j"])
        exact(after["control"], before["control"], "temporary loads never replace held input")
        same(before["history"], [], "zero-delay entry history")
        held_matches(before, controls[w])
        held_matches(after, controls[w])
        if not flags[w]:
            exact(after, before, "capacity retains full independent/root state")
            exact(row["geometry"][w], row["entry_geometry"][w], "capacity retains accepted full FK")
            exact(row["energy"][w], row["entry_energy"][w], "capacity retains accepted energy")
            same(row["diagnostics"][w], None, "rejected attempt has no accepted diagnostic")
            same(row["status"][w], "SITE_CAPACITY_EXCEEDED", "exact capacity refusal")
            continue
        same(after["tick"], before["tick"] + 1, "one accepted installed drift")
        require(row["diagnostics"][w] is not None, "same-attempt full diagnostic cannot be absent")
        audit = row["audit"][w]
        fields(audit, ("same_entry", "next_q", "next_v"), "saved independent same-entry result")
        saved_comparison(row["diagnostics"][w], audit["same_entry"])
        close(after["position"], audit["next_q"], "saved same-entry next q")
        close(after["velocity"], audit["next_v"], "saved same-entry next v")
        independent = row["independent"][w]
        fields(
            independent, ("position", "velocity", "full"), "complete independent trajectory entry"
        )
        close(after["position"], independent["position"], "independent window q", atol=WINDOW_ATOL)
        close(after["velocity"], independent["velocity"], "independent window v", atol=WINDOW_ATOL)
        if shapes is not None:
            load = row["receipts"][w]
            exact(load["entry_state"], before, "same entry contact/physics")
            truth = audit["same_entry"]
            for field in ("contact_generalized", "input_external"):
                close(load[field], truth[field], "same independently saved " + field)
            close(load["actual_drive"], truth["drive"], "same actual motor")
            close(
                load["law_wrench_fp64"],
                truth["independent_wrench"],
                "same independent finite normal law",
            )
            fields(audit, ("material_work",), "both actual motion/work records")
            for field, expected in audit["material_work"].items():
                close(
                    load["diagnostics"][field],
                    expected,
                    "saved independent actual work " + field,
                    atol=1e-12,
                    rtol=1e-10,
                )
    if shapes is not None:
        same(set(row["status"]), world_keys, "whole contact status map")
        broad_schema(row["entry_broad"], row["entry"], shapes)
        broad_schema(row["broad"], row["state"], shapes, saved_oracle=row["bounds_audit"])
        same(
            row["entry_broad"]["state_epoch"],
            row["entry_stats"]["state_epoch"],
            "original entry epoch",
        )
    return sum(flags.values()), len(flags) - sum(flags.values())


def resource_closed(section, *, contact):
    require(
        section.get("failure") is None and section.get("cleanup_failure") is None,
        "no hidden physics/cleanup failure",
    )
    fields(section, ("closed_batch_stats",), "actual final resource counters")
    same(section["closed_batch_stats"]["device_owned_bytes"], 0, "closed batch owned bytes")
    same(section["closed_batch_stats"]["collision_plan_live_count"], 0, "all batch plans closed")
    if contact:
        fields(section, ("closed_plan_stats",), "actual contact close counters")
        same(section["closed_plan_stats"]["device_owned_bytes"], 0, "closed plan bounds scratch")
        same(
            section["closed_plan_stats"]["contact_device_owned_bytes"],
            0,
            "closed contact wrench scratch",
        )


def wheel_payload(value):
    result = {}
    for label, worlds, h, count in (("w4", WORLDS, 1 / 1024, 40), ("a04_p1", (503,), 1 / 4096, 4)):
        section = value[label]
        fields(
            section,
            ("status", "world_order", "h_s", "trace", "initial", "final", "independent_final"),
            "complete installed subwindow",
        )
        same(section["status"], "passed", "actual installed subwindow")
        same(section["world_order"], list(worlds), "declared subwindow worlds")
        same(section["h_s"], h, "fixed subwindow h")
        same(len(section["trace"]), count, "complete installed forward window")
        shapes = bindings = None
        if label == "w4":
            fields(
                section,
                (
                    "inputs",
                    "declarations",
                    "sources",
                    "assets",
                    "controls",
                    "literal_first_anchors",
                    "collision_model",
                    "bindings",
                    "material_sources",
                    "laws",
                    "replay",
                    "capacity",
                    "alignments",
                    "lifecycle",
                    "before_restore",
                    "after_restore",
                    "restore_checkpoint",
                    "retained_first_owned",
                    "checks",
                ),
                "full mixed input/transaction record",
            )
            for name in (
                "inputs",
                "declarations",
                "sources",
                "assets",
                "controls",
                "material_sources",
            ):
                same(set(section[name]), {str(w) for w in worlds}, "all mixed original " + name)
            shapes, bindings = bindings_schema(section["collision_model"], section["bindings"])
            controls = section["controls"]
            for w, name in (("503", "lifecycle"), ("91", "A05")):
                input_schema(section["inputs"][w])
                same(section["inputs"][w]["name"], name, "fixed mixed physical identity")
            close(
                section["literal_first_anchors"]["a05"]["acceleration"][0],
                2 / 7,
                "original A05 literal root anchor",
                atol=2e-13,
                rtol=0,
            )
            close(
                section["literal_first_anchors"]["a05"]["acceleration"][6],
                -10 / 7,
                "original A05 literal motor anchor",
                atol=2e-13,
                rtol=0,
            )
            close(
                section["literal_first_anchors"]["a05"]["row_multipliers"],
                [-2 / 7],
                "original A05 row anchor",
                atol=2e-13,
                rtol=0,
            )
            close(
                section["literal_first_anchors"]["fixed_a03"]["acceleration"],
                [4 / 7],
                "original fixed affine anchor",
                atol=2e-13,
                rtol=0,
            )
            close(
                section["literal_first_anchors"]["fixed_a03"]["row_multipliers"],
                [41 / 14, -25 / 7],
                "original fixed affine rows",
                atol=2e-13,
                rtol=0,
            )
        else:
            fields(
                section,
                ("input", "source", "asset", "declaration", "command"),
                "complete A04 input",
            )
            input_schema(section["input"])
            same(
                (section["input"]["name"], section["input"]["variant"]),
                ("A04", "p1"),
                "frozen A04-p1 pose",
            )
            same(len(section["input"]["parts"]), 4, "all raw A04 inertial components")
            controls = {"503": section["command"]}
        wheel_capture(
            section["initial"], worlds, shapes=shapes, bindings=bindings, wanted_receipts=()
        )
        same(
            {w: s["tick"] for w, s in section["initial"]["state"].items()},
            {str(w): 0 for w in worlds},
            "true initial accepted state",
        )
        previous = section["initial"]["state"]
        rows, total_accepted, total_rejected = [], 0, 0
        for row in section["trace"]:
            exact(row["entry"], previous, "complete original forward continuity")
            a, r = wheel_row(row, worlds, controls, h=h, shapes=shapes, bindings=bindings)
            total_accepted += a
            total_rejected += r
            rows.append(row)
            previous = row["state"]
        if label == "w4":
            same(len(section["replay"]), 21, "complete installed restore replay")
            exact(
                section["restore_checkpoint"],
                section["trace"][18]["state"],
                "original tick19 checkpoint",
            )
            exact(section["before_restore"]["state"], previous, "real forward state before restore")
            wheel_capture(
                section["before_restore"],
                worlds,
                shapes=shapes,
                bindings=bindings,
                wanted_receipts=worlds,
            )
            wheel_capture(
                section["after_restore"],
                worlds,
                shapes=shapes,
                bindings=bindings,
                wanted_receipts=worlds,
                old_receipts=True,
            )
            exact(
                section["after_restore"]["state"],
                section["restore_checkpoint"],
                "complete original restore",
            )
            require(
                all(v is None for v in section["after_restore"]["diagnostics"].values()),
                "restore clears accepted-attempt diagnostics",
            )
            same(
                section["after_restore"]["batch_stats"]["state_epoch"],
                section["before_restore"]["batch_stats"]["state_epoch"] + 1,
                "same-tick restore has a new epoch",
            )
            previous = section["after_restore"]["state"]
            for original, row in zip(section["trace"][19:], section["replay"], strict=True):
                exact(row["entry"], previous, "restored replay entry continuity")
                a, r = wheel_row(row, worlds, controls, h=h, shapes=shapes, bindings=bindings)
                total_accepted += a
                total_rejected += r
                rows.append(row)
                require(row["complete_replay_equal"] is True, "producer replay completion")
                for key in (
                    "input",
                    "entry",
                    "entry_geometry",
                    "entry_energy",
                    "acceptance",
                    "state",
                    "geometry",
                    "diagnostics",
                    "energy",
                    "receipts",
                    "status",
                ):
                    exact(row[key], original[key], "full original replay " + key)
                for key in ("entry_broad", "broad"):
                    for field in BROAD_PHYSICAL:
                        exact(
                            row[key][field],
                            original[key][field],
                            "all replay bounds/pairs " + field,
                        )
                    same(
                        row[key]["topology_generation"],
                        original[key]["topology_generation"],
                        "same original registered generation",
                    )
                    require(
                        row[key]["state_epoch"] > original[key]["state_epoch"],
                        "replay uses current accepted epoch",
                    )
                previous = row["state"]
            same(len(section["capacity"]), 2, "both capacity banks")
            require(0 <= len(section["alignments"]) <= 2, "bounded actual alignment")
            tail = []
            for row in section["alignments"]:
                same(row["label"], "bank-alignment", "real alignment only")
                tail.append(row)
                a, r = wheel_row(row, worlds, controls, h=h, shapes=shapes, bindings=bindings)
                total_accepted += a
                total_rejected += r
            for bank, target, transaction in zip(
                (0, 1), (503, 91), section["capacity"], strict=True
            ):
                same(
                    (transaction["bank"], transaction["limited_world"]),
                    (bank, target),
                    "fixed two-bank rejection identity",
                )
                same(
                    transaction["capacities"],
                    {str(w): 0 if w == target else 128 for w in worlds},
                    "fixed per-world capacity inputs",
                )
                same(len(transaction["limited_rows"]), 1, "one real rejection attempt")
                same(len(transaction["retry_rows"]), 1, "one full-capacity same-input retry")
                require(transaction["checks_complete"] is True, "complete atomic rejection/retry")
                rejected, retry = transaction["limited_rows"][0], transaction["retry_rows"][0]
                same(
                    rejected["entry_stats"]["current_buffer"], bank, "actual requested active bank"
                )
                exact(
                    rejected["entry"],
                    transaction["before"]["state"],
                    "actual pre-refusal full state",
                )
                wheel_capture(
                    transaction["before"],
                    worlds,
                    shapes=shapes,
                    bindings=bindings,
                    wanted_receipts=worlds,
                )
                exact(
                    transaction["full_plan_receipts_after_rejection"],
                    transaction["before"]["receipts"],
                    "other existing plan receipts remain owned and unchanged",
                )
                for row, refused in ((rejected, target), (retry, None)):
                    a, r = wheel_row(
                        row,
                        worlds,
                        controls,
                        h=h,
                        shapes=shapes,
                        bindings=bindings,
                        rejected_world=refused,
                    )
                    total_accepted += a
                    total_rejected += r
                    tail.append(row)
                exact(
                    retry["entry"],
                    rejected["state"],
                    "retry uses actual retained/advanced mixed state",
                )
                exact(
                    retry["receipts"][str(target)]["entry_state"],
                    rejected["entry"][str(target)],
                    "refused world retries the same original state",
                )
                for field in ("device_owned_bytes", "contact_device_owned_bytes"):
                    same(
                        transaction["closed_plan_stats"][field],
                        0,
                        "closed short-lived plan scratch",
                    )
            # Ordering verifies every fixed record; it does not select evidence.
            tail.sort(key=lambda row: row["entry_stats"]["state_epoch"])
            same(
                len({row["entry_stats"]["state_epoch"] for row in tail}),
                len(tail),
                "distinct actual tail attempts",
            )
            for row in tail:
                exact(row["entry"], previous, "full tail attempt chronology")
                previous = row["state"]
            rows += tail
            exact(
                section["retained_first_owned"],
                {
                    key: section["trace"][0][key]
                    for key in ("state", "geometry", "diagnostics", "receipts")
                },
                "old owned public records unchanged",
            )
            lifecycle = section["lifecycle"]
            same(len(lifecycle["loaded"]), 40, "complete lifecycle record")
            require(
                all(type(flag) is bool for flag in lifecycle["loaded"]), "actual lifecycle booleans"
            )
            first_open = lifecycle["first_open_index"]
            require(
                type(first_open) is int
                and 0 < first_open < 39
                and lifecycle["loaded"][0]
                and not lifecycle["loaded"][first_open]
                and all(lifecycle["loaded"][:first_open])
                and any(lifecycle["loaded"][first_open + 1 :]),
                "loaded/open/reentry original lifecycle",
            )
            for flag, row in zip(lifecycle["loaded"], section["trace"], strict=True):
                same(
                    flag,
                    any(
                        x != 0 for force in row["receipts"]["503"]["law_wrench_fp64"] for x in force
                    ),
                    "lifecycle uses actual saved force",
                )
            same(
                section["checks"],
                dict(
                    complete_forward_40=True,
                    complete_restore_21=True,
                    lifecycle_loaded_open_reenter=True,
                    both_capacity_banks=True,
                    owned_initial_unchanged=True,
                ),
                "all fixed wheel checks",
            )
        exact(section["final"]["state"], previous, "complete final state")
        wheel_capture(
            section["final"],
            worlds,
            shapes=shapes,
            bindings=bindings,
            wanted_receipts=worlds if shapes is not None else (),
        )
        for w in worlds:
            state = section["final"]["state"][str(w)]
            wanted_tick = (
                40 + len(section["alignments"]) + 4 - int(w in (503, 91)) if label == "w4" else 4
            )
            same(state["tick"], wanted_tick, "actual final independent tick")
            final_q, final_v = section["independent_final"][str(w)]
            close(
                state["position"], final_q, "whole installed final independent q", atol=WINDOW_ATOL
            )
            close(
                state["velocity"], final_v, "whole installed final independent v", atol=WINDOW_ATOL
            )
        resource_closed(section, contact=label == "w4")
        result[label] = dict(
            status="passed",
            completed_forward=count,
            completed_replay=21 if label == "w4" else 0,
            completed_capacity_banks=2 if label == "w4" else 0,
            real_alignment_steps=len(section["alignments"]) if label == "w4" else 0,
            attempted_batch_steps=len(rows),
            returned_acceptance_steps=len(rows),
            accepted_world_steps=total_accepted,
            rejected_world_steps=total_rejected,
            checks=section.get("checks", {}),
            closed_batch_stats=section["closed_batch_stats"],
            closed_plan_stats=section.get("closed_plan_stats"),
            failure=None,
            cleanup_failure=None,
        )
    same(result["w4"]["rejected_world_steps"], 2, "two real capacity-refused world attempts")
    accepted = sum(item["accepted_world_steps"] for item in result.values())
    same(
        accepted,
        262 + 4 * result["w4"]["real_alignment_steps"],
        "actual complete accepted count without reset/replay inflation",
    )
    require(262 <= accepted <= 270, "declared complete installed range")
    return result


def audit_manifest(manifest, path):
    fields(
        manifest,
        ("schema", "pytest_runs", "external_runs", "historical_runs", "w32"),
        "explicit ended-run manifest",
    )
    same(manifest["schema"], "c11-saved-affine-v1", "manifest protocol")
    scan(manifest)
    observer = Auditor(manifest)
    observer.report["manifest"] = dict(
        path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()
    )
    observer.report["authoring_basis"] = dict(
        tests="be4f1a052a13c2d3227fc4a4ab8599c044e64a42",
        native="5b34f120519afec9ea4772b04c237c3a14be0ed6",
        contract="2966aa9 section 11",
    )
    all_members = membership()
    core = tuple(member for member in all_members if not member.optional)
    same(
        (
            len(core),
            sum(len(member.files) for member in core),
            sum(not member.files for member in core),
        ),
        (52, 46, 14),
        "fixed pre-output authored identity table",
    )
    observer.report["fixed_membership"] = [
        dict(
            node=member.node,
            family=member.family,
            payloads=member.files,
            steps=member.steps,
            h_s=member.h,
            replay=member.replay,
            optional=member.optional,
        )
        for member in all_members
    ]
    for section in ("historical_runs", "pytest_runs", "external_runs"):
        require(isinstance(manifest[section], list), "explicit ordered run list")
        for spec in manifest[section]:
            try:
                if section == "historical_runs":
                    root, args, sources, stdout, job = observer.receipt(spec)
                    require(
                        spec.get("reason"),
                        "history has an explicit original failure/source qualification",
                    )
                    files = spec.get("saved_prefix_files", [])
                    require(len(set(files)) == len(files), "unique historical prefix paths")
                    history_root = Path(spec.get("output_directory", root)).resolve(strict=True)
                    if "output_directory" in spec:
                        same(
                            (
                                Path(job["original_command"]["workdir"]) / option(args, "--output")
                            ).resolve(),
                            history_root,
                            "original historical output CLI",
                        )
                    for name in files:
                        value = observer.json(
                            history_root / name, history_root, compressed=name.endswith(".gz")
                        )
                        scan(value, tags=True)
                    job.update(
                        status="historical-only",
                        reason=spec["reason"],
                        saved_prefix_files=files,
                        coverage_contribution=0,
                    )
                    observer.report["runs"].append(job)
                elif section == "pytest_runs":
                    same(spec["kind"], "pytest", "actual pytest run kind")
                    observer.pytest_run(spec)
                else:
                    observer.external_run(spec)
            except Exception as error:
                observer.report["errors"].append(
                    dict(
                        run=spec.get("id"),
                        type=type(error).__name__,
                        message=str(error),
                        traceback=traceback.format_exc(),
                    )
                )
    core_nodes = {member.node for member in core}
    optional = next(member for member in all_members if member.optional)
    coverage = {}
    for mode, actual in observer.coverage.items():
        complete = core_nodes.intersection(actual)
        coverage[mode] = dict(
            complete_nodes=len(complete),
            expected_nodes=52,
            complete_payload_identities=sum(
                len(member.files) for member in core if member.node in complete
            ),
            expected_payload_identities=46,
            pytest_only_no_save=sum(not member.files for member in core if member.node in complete),
            expected_no_save=14,
            missing=sorted(core_nodes - complete),
            identities={member.node: actual.get(member.node, []) for member in core},
        )
        if complete != core_nodes:
            observer.report["errors"].append(
                dict(
                    scope=mode,
                    message="incomplete fixed core coverage",
                    missing=coverage[mode]["missing"],
                )
            )
    observer.report["coverage"] = coverage
    same(set(manifest["w32"]), {"interpret", "cuda"}, "explicit optional result in both modes")
    observer.report["optional_w32"] = {}
    for mode, declared in manifest["w32"].items():
        status = declared["status"]
        require(status in ("NOT_RUN", "PASSED", "FAILED"), "explicit optional W32 status")
        completed = observer.coverage[mode].get(optional.node, [])
        if status == "PASSED":
            require(
                completed and declared["run"] in [item["run"] for item in completed],
                "optional completed identity was actually audited",
            )
        else:
            require(not completed, "NOT_RUN/FAILED cannot discard a completed W32")
            if status == "FAILED":
                matches = [job for job in observer.report["runs"] if job["id"] == declared["run"]]
                require(
                    len(matches) == 1 and matches[0].get("failed") == optional.node,
                    "optional original first failure",
                )
        observer.report["optional_w32"][mode] = dict(
            status=status, runs=completed, declaration=declared
        )
    expected_external = {
        ("native", "cpu"),
        ("native", "cuda:0"),
        ("wheel", "interpret"),
        ("wheel", "cuda"),
    }
    observer.report["external_coverage"] = [
        {
            "kind": kind,
            "mode": mode,
            "run": observer.external.get((kind, mode)),
            "complete": (kind, mode) in observer.external,
        }
        for kind, mode in sorted(expected_external)
    ]
    missing_external = expected_external - set(observer.external)
    if missing_external:
        observer.report["errors"].append(
            dict(
                scope="external",
                message="missing original native/wheel complete records",
                missing=sorted(missing_external),
            )
        )
    observer.report["decoded_bytes"] = observer.total
    observer.report["processed_files"] = len(observer.files)
    observer.report["status"] = (
        "complete-saved-evidence"
        if not observer.report["errors"]
        else "incomplete-or-invalid-evidence"
    )
    return observer.report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(
        args.manifest.is_absolute() and args.output.is_absolute(),
        "absolute manifest and fresh output directory required",
    )
    manifest_path = args.manifest.resolve(strict=True)
    require(
        manifest_path.is_file() and manifest_path.stat().st_size <= MAX_METADATA,
        "bounded manifest file",
    )
    output = args.output.resolve()
    require(not output.exists(), "output must be a new directory; old evidence is never replaced")
    manifest = strict_load(manifest_path.read_bytes())
    # The manifest and every original input remain untouched. Only this new
    # report directory is created, including when a source/schema check fails.
    output.mkdir(parents=True, exist_ok=False)
    try:
        report = audit_manifest(manifest, manifest_path)
    except Exception as error:
        report = dict(
            status="incomplete-or-invalid-evidence",
            observer_only=True,
            errors=[
                dict(
                    type=type(error).__name__, message=str(error), traceback=traceback.format_exc()
                )
            ],
            scope="No physics, import, producer or project command has been invoked.",
        )
    target = output / "audit.json"
    target.write_text(
        json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            dict(status=report["status"], audit=str(target), errors=len(report.get("errors", ()))),
            allow_nan=False,
        )
    )
    return 0 if report["status"] == "complete-saved-evidence" else 1


if __name__ == "__main__":
    raise SystemExit(main())
