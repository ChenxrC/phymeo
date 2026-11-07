"""C9 saved-record audit: standard library only, never a physics rerun.

The authored membership is fixed below before any evidence is read. A manifest
names ended runs, exact node whitelists, source hashes and original tmp roots.
No glob decides which logical cases should exist. AUTHORING_UNEXECUTED.
"""

import argparse
import gzip
import hashlib
import io
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

MIB = 1024 * 1024
MAX_COMPRESSED = 64 * MIB
MAX_PAYLOAD = 256 * MIB
MAX_TOTAL = 2 * 1024 * MIB
MAX_METADATA = 16 * MIB
MAX_FILES = 8192
MAX_DEPTH = 96
H = 1 / 1024
STEP = "tests/scale/test_resident_floating_step.py"
COLLISION = "tests/scale/test_resident_floating_collision.py"
REFERENCE = "tests/reference_newton/robot/resident_floating_reference.py"
NATIVE_SOURCE = "tests/reference_newton/robot/resident_floating_native.py"
WHEEL_SOURCE = "tests/scale/resident_floating_wheel_consumer.py"
NUMERIC = "src/phymeo/backends/_tyx_floating_numeric.py"
NUMERIC_TEST = "tests/scale/_floating_numeric_kernel.py"
KEYS = (503, 91, 1207, 6) + tuple(2003 + 17 * i for i in range(28))
CASES = ("P01", "P02_origin", "P02_frame", "P03", "P03_external", "P04", "P04_eccentric")
NATIVE_CASES = (
    "P01_freefall",
    "P02_origin_force",
    "P02_principal_torque",
    "P03_internal",
    "P03_child_force",
    "P04_internal",
    "P04_eccentric",
)


class AuditError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise AuditError(message)


@dataclass(frozen=True)
class Member:
    source: str
    function: str
    group: str
    parameter: tuple = ()
    suffix: str = ""
    files: tuple = ()

    @property
    def node(self):
        return self.source + "::" + self.function + ("[" + self.suffix + "]" if self.suffix else "")


def membership():
    """Literal author contract, not inferred from found results or their hashes."""
    device, host = [], []

    def add(source, function, group, variants, files, out=device):
        for params, suffix in variants:
            names = files(params) if callable(files) else files
            out.append(
                Member(
                    source,
                    function,
                    group,
                    params,
                    suffix,
                    tuple(name + ".json.gz" for name in names),
                )
            )

    def simple(source, function, group, files):
        add(source, function, group, [((), "")], files)

    simple(
        STEP,
        "test_p03_public_floating_tree_keeps_complete_mass_and_internal_recoil",
        "public",
        ("p03-public-floating",),
    )
    add(
        STEP,
        "test_p01_p04_full_physical_windows",
        "window",
        [((x,), x) for x in CASES],
        ("floating-window",),
    )
    add(
        STEP,
        "test_p03_p05_nonaxis_installation_boost_and_large_common_origin",
        "covariance",
        [
            ((a, b, c), f"{a}-{b}-{c}")
            for a, b, c in (
                ("P01", "cycle", True),
                ("P03", "reverse", True),
                ("P04_eccentric", "cycle", False),
                ("P05", "large", True),
            )
        ],
        ("floating-window",),
    )
    add(
        STEP,
        "test_p05_independent_full_root_and_branch_dimensions",
        "dimension",
        [((n,), str(n)) for n in (0, 1, 9, 16)],
        ("floating-window",),
    )
    simple(
        STEP,
        "test_p05_real_folded_root_tool_mass_frame_and_load_window",
        "tool",
        ("floating-window",),
    )
    add(
        STEP,
        "test_p06_delayed_drive_external_immediate_and_bounded_full_response",
        "pd",
        [((cap, delay), f"{cap}-{delay}") for cap in (0.125, 8.0) for delay in (0, 1)],
        ("floating-window",),
    )
    simple(
        STEP,
        "test_p05_w1_w4_w32_all_full_states_and_21_step_replay",
        "worlds",
        tuple(f"w1-{k}/floating-worlds" for k in KEYS[:4])
        + ("w4/floating-worlds", "w32/floating-worlds"),
    )
    simple(
        STEP,
        "test_p05_mixed_existing_fixed_affine_and_free_worlds_share_one_publish",
        "mixed",
        ("floating-existing-mixed",),
    )
    add(
        STEP,
        "test_p06_real_late_failure_keeps_both_banks_and_same_request_retries",
        "late",
        [
            ((warm, phase), f"{warm}-{phase}")
            for warm in (0, 1)
            for phase in ("upload", "candidate", "restore")
        ],
        ("floating-late-fault",),
    )
    add(
        STEP,
        "test_p06_invalid_restore_is_atomic_for_mixed_full_dimensions",
        "invalid",
        [
            ((s,), s)
            for s in (
                "q_length",
                "v_length",
                "zero_quat",
                "quat_norm",
                "nan",
                "local_position",
                "joint_position",
                "foreign_topology",
            )
        ],
        ("floating-invalid-restore",),
    )
    add(
        STEP,
        "test_p06_candidate_domain_failure_is_world_local_and_does_not_keep_old_diagnostic",
        "domain",
        [((s,), s) for s in ("root_translation", "joint_translation", "rotation_vector")],
        ("floating-domain-retain",),
    )
    simple(
        STEP,
        "test_p06_full_cross_motor_mobility_low_budget_rejects_then_full_budget_agrees",
        "budget",
        ("floating-low-budget", "budget512/floating-window"),
    )
    simple(
        STEP,
        "test_p06_owned_outputs_controls_and_invalid_mixed_update_are_atomic",
        "owned",
        ("floating-owned-update",),
    )
    simple(
        STEP,
        "test_p06_large_anchor_checkpoint_retains_actual_local_bank_and_replays",
        "large",
        ("floating-large-anchor-local",),
    )
    add(
        STEP,
        "test_p06_checkpoint_local_and_world_origin_must_describe_same_state",
        "origin",
        [((s,), s) for s in ("world", "local", "width", "domain", "nan")],
        ("floating-origin-binding-rejection",),
    )
    add(
        STEP,
        "test_p06_pi_neighbourhood_preserves_real_pose_and_quaternion_hemisphere",
        "hemisphere",
        [((a,), str(a)) for a in (math.pi - 2.0**-20, math.pi)],
        ("floating-pi-hemisphere",),
    )
    simple(
        STEP,
        "test_p06_partial_floating_storage_allocation_closes_original_runtime_and_rebuilds",
        "allocation",
        ("floating-partial-allocation",),
    )
    add(
        STEP,
        "test_p06_floating_fp32_arithmetic_flushes_underflow_but_retains_nonfinite",
        "numeric",
        [
            ((s,), s)
            for s in (
                "product_underflow",
                "division_underflow",
                "cancellation_underflow",
                "overflow",
                "invalid_division",
            )
        ],
        ("floating-fp32-underflow-policy",),
    )
    add(
        COLLISION,
        "test_p07_all_finite_materials_cover_actual_root_and_child_rotation",
        "bounds",
        [
            ((n, p), f"{n}-{p}")
            for n, p in ((0, "base"), (1, "base"), (3, "cycle"), (9, "large"), (16, "base"))
        ],
        ("floating-current-bounds",),
    )
    add(
        COLLISION,
        "test_p07_warm_bounds_consume_real_banks_without_host_snapshot_or_dynamic_upload",
        "warm",
        [((n,), str(n)) for n in (1, 4, 32)],
        ("floating-warm-bounds",),
    )
    simple(
        COLLISION,
        "test_p07_folded_asset_root_and_tool_material_is_transformed_once",
        "folded",
        ("floating-folded-source",),
    )
    add(
        COLLISION,
        "test_p07_same_full_tree_branch_domain_filters_before_adjacency",
        "filter",
        [
            ((s,), s)
            for s in ("none", "same_body", "mask", "excluded", "parent_mask", "parent_excluded")
        ],
        ("floating-branch-domain",),
    )
    add(
        COLLISION,
        "test_p07_mismatched_owner_material_source_and_parent_reject_before_allocation",
        "identity-no-save",
        [
            ((s,), s)
            for s in (
                "parent",
                "group",
                "world",
                "body_alias",
                "two_bodies_one_group",
                "foreign_topology",
                "source_radius",
                "source_vertices",
                "source_revision",
                "source_model",
            )
        ],
        (),
    )
    simple(
        COLLISION, "test_p07_floating_contacts_stay_explicitly_unsupported", "contact-no-save", ()
    )
    simple(
        COLLISION,
        "test_p07_old_fixed_and_affine_contacts_advance_unbound_floating_world",
        "unbound",
        ("floating-unbound-contact-mixed",),
    )
    simple(
        COLLISION,
        "test_p07_floating_bounds_capacity_late_readback_and_same_tick_restore_are_atomic",
        "bounds-atomic",
        ("floating-bounds-transaction",),
    )
    simple(
        COLLISION,
        "test_p07_floating_runtime_close_releases_all_owned_plans_and_rebuilds",
        "close",
        ("floating-bounds-close",),
    )
    simple(
        COLLISION,
        "test_p07_folded_moving_child_tool_uses_authored_nonaxis_mount",
        "child-tool",
        ("floating-folded-child-material",),
    )
    add(
        STEP,
        "test_p01_p04_time_jet_oracle_has_literal_physical_anchors",
        "host-anchor",
        [((s,), s) for s in CASES[:6]],
        (),
        host,
    )
    add(
        STEP,
        "test_p06_declared_input_dimensions_and_resources_reject_before_runtime",
        "host-domain",
        [
            ((s,), s)
            for s in (
                "n17",
                "gain_width",
                "negative_gain",
                "delay2",
                "bool_iterations",
                "zero_iterations",
                "nonunit_initial",
                "load_effort_width",
                "load_body_width",
            )
        ],
        (),
        host,
    )
    require(
        len(device) == 86 and len(host) == 15 and sum(len(m.files) for m in device) == 81,
        "fixed authored membership count",
    )
    require(len({m.node for m in device + host}) == 101, "duplicate authored node")
    return device, host


def fields(value, names, label="record"):
    require(isinstance(value, dict), label + " is not an object")
    require(
        set(names) <= value.keys(), label + " missing " + str(sorted(set(names) - value.keys()))
    )


def number(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def array(value, *shape):
    if not shape:
        require(number(value), "nonfinite/non-numeric array scalar")
        return
    require(isinstance(value, list) and len(value) == shape[0], "array shape " + str(shape))
    for part in value:
        array(part, *shape[1:])


def same(left, right, label):
    require(left == right, label + " saved values differ")


def exact(left, right, label):
    # Same producer's complete JSON-value identity: preserve every field and
    # signed zero/type spelling; sort object keys only, as the wheel consumer does.
    def digest(value):
        result = hashlib.sha256()
        for token in json.JSONEncoder(
            sort_keys=True, allow_nan=False, separators=(",", ":")
        ).iterencode(value):
            result.update(token.encode("utf-8"))
        return result.digest()

    require(digest(left) == digest(right), label + " complete saved bytes differ")


def close(left, right, atol=5e-6, rtol=5e-5):
    """Compare two saved values only; never recompute a physical oracle."""
    if isinstance(left, list) or isinstance(right, list):
        require(
            isinstance(left, list) and isinstance(right, list) and len(left) == len(right),
            "saved comparison shape",
        )
        for a, b in zip(left, right, strict=True):
            close(a, b, atol, rtol)
    else:
        require(number(left) and number(right), "saved comparison nonfinite")
        require(abs(left - right) <= atol + rtol * abs(right), "saved reference mismatch")


def checkpoint(value):
    fields(
        value,
        ("world", "tick", "position", "velocity", "applied_effort", "history", "control"),
        "checkpoint",
    )
    require(
        type(value["world"]) is int and type(value["tick"]) is int and value["tick"] >= 0,
        "checkpoint identity/tick",
    )
    if "local_origin_m" in value:
        fields(value, ("topology", "generalized_applied"), "floating checkpoint")
        n = len(value["position"]) - 7
        require(0 <= n <= 16, "floating N domain")
        array(value["position"], n + 7)
        array(value["velocity"], n + 6)
        array(value["local_origin_m"], 3)
        array(value["applied_effort"], n)
        array(value["generalized_applied"], n + 6)
        require(len(value["control"]) == 4, "all four floating control blocks")
        for block, shape in zip(
            value["control"], ((n, 3), (n,), (n + 1, 3), (n + 1, 3)), strict=True
        ):
            array(block, *shape)
        require(value["history"] == [] or n > 0, "root-only cannot contain motor history")
        if value["history"]:
            array(value["history"], n, 3)
        require(value["topology"], "full topology binding absent")
        return n
    # The unbound mixed participants remain old public checkpoint types.
    k = len(value["position"])
    array(value["position"], k)
    array(value["velocity"], k)
    if "full_position" in value:
        n = len(value["full_position"])
        for name in (
            "full_position",
            "full_velocity",
            "full_applied_effort",
            "full_generalized_applied",
        ):
            array(value[name], n)
        require(len(value["control"]) == 2, "affine held blocks")
        array(value["control"][0], k, 3)
        array(value["control"][1], n, 7)
    elif "generalized_applied" in value:
        array(value["control"], k, 10)
    return None


def geometry(value):
    fields(
        value,
        ("tick", "position", "velocity", "link_position_m", "link_rotation", "com_position_m"),
        "tree geometry",
    )
    b = len(value["link_position_m"])
    for name in ("link_position_m", "com_position_m"):
        array(value[name], b, 3)
    array(value["link_rotation"], b, 3, 3)
    if "configuration_count" in value:
        n = value["joint_count"]
        for name in ("linear_velocity_m_s", "angular_velocity_rad_s"):
            array(value[name], b, 3)
        same(
            (value["body_count"], value["velocity_count"], value["configuration_count"]),
            (n + 1, n + 6, n + 7),
            "B/D/Q distinct dimensions",
        )
        array(value["position"], n + 7)
        array(value["velocity"], n + 6)
        require(len(value["joint_types"]) == n, "full joint kinds")


def diagnostic(value, *, check_residual=True):
    fields(
        value,
        (
            "entry_tick",
            "exit_tick",
            "dt_s",
            "entry_position",
            "entry_velocity",
            "linear_jacobian",
            "angular_jacobian",
            "mass",
            "bias",
            "external",
            "drive",
            "acceleration",
            "iterations",
            "control_residual",
            "equation_residual",
        ),
        "diagnostic",
    )
    d, b = len(value["mass"]), len(value["linear_jacobian"])
    array(value["mass"], d, d)
    for name in ("bias", "external"):
        array(value[name], d)
    if "reduced_mass" in value:
        k = len(value["entry_position"])
        array(value["acceleration"], k)
        array(value["full_acceleration"], d)
        array(value["constraint_effort_full"], d)
        array(value["reduced_mass"], k, k)
        for name in ("reduced_bias", "reduced_external", "drive"):
            array(value[name], k)
        for name in ("reduced_linear_jacobian", "reduced_angular_jacobian"):
            array(value[name], b, 3, k)
    else:
        array(value["acceleration"], d)
    for name in ("linear_jacobian", "angular_jacobian"):
        array(value[name], b, 3, d)
    same(value["exit_tick"], value["entry_tick"] + 1, "diagnostic attempt")
    same(value["dt_s"], H, "fixed physical substep")
    if check_residual:
        require(
            0 <= value["control_residual"] <= 5e-5 and 0 <= value["equation_residual"] <= 2e-5,
            "saved original residual gates",
        )
    if len(value["entry_position"]) == d + 1:
        n = d - 6
        require(0 <= n <= 16 and b == n + 1, "floating diagnostic B/D")
        array(value["drive"], n)
        array(value["entry_position"], n + 7)
        array(value["entry_velocity"], n + 6)


def collision_key(value):
    fields(value, ("world", "body", "shape"), "collision key")
    require(
        all(type(value[k]) is int for k in ("world", "body", "shape")), "integer collision identity"
    )
    return value["world"], value["body"], value["shape"]


def broad(value):
    fields(
        value,
        (
            "topology_generation",
            "state_epoch",
            "world_keys",
            "ticks",
            "shape_keys",
            "shape_revisions",
            "bounds",
            "registered_pairs",
            "candidates",
        ),
        "resident bounds",
    )
    keys = [collision_key(x) for x in value["shape_keys"]]
    require(len(set(keys)) == len(keys), "duplicate full shape identity")
    require(value["world_keys"] == sorted(set(value["world_keys"])), "canonical worlds")
    require(
        len(value["ticks"]) == len(value["world_keys"])
        and len(value["shape_revisions"]) == len(keys),
        "complete bounds ticks/revisions",
    )
    array(value["bounds"], len(keys), 6)
    require(
        all(all(row[j] <= row[j + 3] for j in range(3)) for row in value["bounds"]),
        "inverted saved AABB",
    )
    pairs = []
    for pair in value["registered_pairs"]:
        require(len(pair) == 2, "pair length")
        a, b = map(collision_key, pair)
        require(a < b and a[0] == b[0] and a in keys and b in keys, "same-world canonical pair")
        pairs.append((a, b))
    require(len(set(pairs)) == len(pairs), "duplicate registered pair")
    candidates = [tuple(map(collision_key, p)) for p in value["candidates"]]
    require(
        len(set(candidates)) == len(candidates) and set(candidates) <= set(pairs),
        "candidate ownership",
    )


def scan(value, depth=0, *, failure_prefix=False):
    """Validate every retained field, including nested old world observations."""
    require(depth <= MAX_DEPTH, "JSON nesting bound")
    if isinstance(value, float):
        require(math.isfinite(value), "nonfinite retained scalar")
    elif isinstance(value, list):
        for item in value:
            scan(item, depth + 1, failure_prefix=failure_prefix)
    elif isinstance(value, dict):
        require("nonfinite" not in value, "nonfinite marker in retained record")
        if {"world", "tick", "position", "velocity", "control", "history"} <= value.keys():
            checkpoint(value)
        if {"configuration_count", "link_position_m"} <= value.keys():
            geometry(value)
        if {"entry_tick", "mass", "control_residual"} <= value.keys():
            diagnostic(value, check_residual=not failure_prefix)
        if {"shape_keys", "registered_pairs", "bounds"} <= value.keys():
            broad(value)
        for item in value.values():
            scan(item, depth + 1, failure_prefix=failure_prefix)
    else:
        require(value is None or isinstance(value, (str, bool, int)), "unsupported JSON type")


def state_map(value, expected=None):
    require(isinstance(value, dict) and value, "full world state mapping")
    if expected is not None:
        same(set(value), {str(k) for k in expected}, "world set")
    for key, state in value.items():
        checkpoint(state)
        same(str(state["world"]), key, "outer/inner world identity")


def advance(entry, accepted):
    checkpoint(entry)
    checkpoint(accepted)
    same(accepted["world"], entry["world"], "attempt world")
    same(accepted["tick"], entry["tick"] + 1, "one accepted physical step")
    same(accepted["control"], entry["control"], "complete held inputs")
    if "topology" in entry:
        same(accepted["topology"], entry["topology"], "model/source identity")


def data_schema(data):
    fields(
        data,
        (
            "name",
            "root_mass",
            "root_com",
            "root_inertia",
            "root_inertia_frame",
            "gravity",
            "children",
            "q",
            "v",
            "drive",
            "effort",
            "force",
            "torque",
            "placement",
            "force_point",
        ),
        "original model input",
    )
    n = len(data["q"]) - 7
    array(data["q"], n + 7)
    array(data["v"], n + 6)
    array(data["drive"], n, 3)
    array(data["effort"], n)
    array(data["force"], n + 1, 3)
    array(data["torque"], n + 1, 3)
    for name, width in (
        ("root_com", 3),
        ("root_inertia", 3),
        ("root_inertia_frame", 4),
        ("gravity", 3),
    ):
        array(data[name], width)
    child = data["children"]
    fields(
        child,
        ("parents", "kinds", "offsets", "orientations", "axes", "mass", "com", "inertia"),
        "full child declaration",
    )
    for name in ("parents", "kinds", "mass"):
        require(len(child[name]) == n, "child field width " + name)
    for name in ("offsets", "axes", "com", "inertia"):
        array(child[name], n, 3)
    array(child["orientations"], n, 4)
    return n


def binding_schema(record):
    fields(record, ("model", "bindings"), "collision source packet")
    shapes = record["model"]["shapes"]
    shape_keys = {collision_key(s["key"]) for s in shapes}
    bindings = record["bindings"]
    require(len(shape_keys) == len(shapes) == len(bindings), "complete shape binding coverage")
    require({collision_key(b["key"]) for b in bindings} == shape_keys, "binding identities")
    for shape in shapes:
        fields(
            shape, ("key", "kind", "vertex_ids", "radius", "revision"), "original material shape"
        )
        require(len(set(shape["vertex_ids"])) == len(shape["vertex_ids"]), "original material IDs")
    for binding in bindings:
        fields(binding, ("key",), "original binding")
        if "local_points_m" in binding:
            same(binding["world"], binding["key"]["world"], "binding world")
            require(
                type(binding["group_index"]) is int and binding["group_index"] >= 0,
                "body group index",
            )
            shape = next(s for s in shapes if s["key"] == binding["key"])
            array(binding["local_points_m"], len(shape["vertex_ids"]), 3)


def full_frame(row, worlds=None):
    fields(row, ("state", "geometry", "diagnostic", "energy", "stats"), "complete accepted frame")
    state_map(row["state"], worlds)
    for key, geo in row["geometry"].items():
        require(key in row["state"], "geometry world")
        geometry(geo)
        same(geo["tick"], row["state"][key]["tick"], "geometry accepted tick")
    for key, diag in row["diagnostic"].items():
        require(key in row["state"], "diagnostic world")
        if diag is not None:
            diagnostic(diag)
            same(diag["exit_tick"], row["state"][key]["tick"], "diagnostic accepted attempt")


def original_bounds(row, name="independent_material_bounds"):
    fields(row, ("result", name), "material bounds frame")
    result = row["result"]
    broad(result)
    wanted = row[name]
    require(len(wanted) == len(result["shape_keys"]), "all original materials saved")
    # Keys are the source serializer's full CollisionKey repr; no rematching by world point.
    for key, bound in zip(result["shape_keys"], result["bounds"], strict=True):
        label = f"CollisionKey(world={key['world']}, body={key['body']}, shape={key['shape']})"
        fields(
            wanted[label],
            ("material_points", "radius", "low", "high"),
            "independent original material bounds",
        )
        item = wanted[label]
        require(item["material_points"], "empty finite material")
        array(item["material_points"], len(item["material_points"]), 3)
        array(item["low"], 3)
        array(item["high"], 3)
        require(
            all(bound[j] <= item["low"][j] and bound[j + 3] >= item["high"][j] for j in range(3)),
            "saved FP64 material not enclosed",
        )


def check_window(member, record, *, budget=False, partial=False):
    fields(
        record,
        (
            "input",
            "mode",
            "h_s",
            "steps",
            "delay",
            "kp",
            "kd",
            "cap",
            "rows",
            "failure",
            "retained",
            "stats",
        ),
        "floating window",
    )
    n = data_schema(record["input"])
    same(record["h_s"], H, "window dt")
    steps = 2 if budget else 4 if member.group == "pd" else 40
    same(record["steps"], steps, "declared full window")
    if not partial:
        same(len(record["rows"]), steps, "completed full window")
        same(record["failure"], None, "window failure retained")
    else:
        require(len(record["rows"]) <= steps, "failure prefix length")
    if member.group == "window":
        same(record["input"]["name"], member.parameter[0], "window input case")
    if member.group == "dimension":
        same(n, member.parameter[0], "full N")
    if member.group == "covariance":
        same(
            (record["input"]["name"], record["input"]["placement"]),
            member.parameter[:2],
            "covariance input",
        )
    if member.group == "pd":
        same(record["delay"], member.parameter[1], "delay contract")
        same(record["cap"], [member.parameter[0]] * 3, "original effort cap")
    previous = None
    for index, row in enumerate(record["rows"]):
        fields(
            row,
            (
                "entry",
                "accepted",
                "geometry",
                "diagnostic",
                "summary",
                "packet",
                "selected_drive",
                "expected_position",
                "expected_velocity",
                "trajectory_reference",
                "audit",
                "stats",
            ),
            "full window row",
        )
        advance(row["entry"], row["accepted"])
        same(row["entry"]["tick"], index, "entry tick")
        if previous is not None:
            # Dynamic world load packets can change at set_controls; all physical fields stay continuous.
            for field in (
                "position",
                "velocity",
                "local_origin_m",
                "applied_effort",
                "generalized_applied",
                "history",
                "topology",
            ):
                same(row["entry"][field], previous[field], "physical continuity " + field)
        fields(row["packet"], ("drive", "external"), "actual held request")
        packet = row["packet"]
        same(
            row["entry"]["control"],
            [
                packet["drive"],
                packet["external"]["effort"],
                packet["external"]["force"],
                packet["external"]["torque"],
            ],
            "all held input blocks",
        )
        same(
            len(row["accepted"]["history"]),
            n if record["delay"] else 0,
            "complete delayed motor history",
        )
        geometry(row["geometry"])
        diagnostic(row["diagnostic"])
        same(row["diagnostic"]["entry_position"], row["entry"]["position"], "diagnostic entry Q")
        same(row["diagnostic"]["entry_velocity"], row["entry"]["velocity"], "diagnostic entry D")
        close(row["accepted"]["position"][3:], row["expected_position"][3:], atol=2e-4)
        close(row["accepted"]["velocity"], row["expected_velocity"], atol=2e-4)
        fields(
            row["audit"],
            (
                "same_entry_reference",
                "actual_end_reference",
                "predicted_position",
                "predicted_velocity",
                "generalized_entry_power",
                "actual_kinetic_energy",
            ),
            "all independent saved audit fields",
        )
        for name in (
            "mass",
            "bias",
            "external",
            "acceleration",
            "drive",
            "linear_jacobian",
            "angular_jacobian",
        ):
            close(
                row["diagnostic"][name],
                row["audit"]["same_entry_reference"][name],
                atol=2e-5 if name == "mass" else 5e-6,
            )
        previous = row["accepted"]
    state_map(record["retained"], [503])
    if record["rows"] and not partial:
        same(record["retained"]["503"], previous, "retained complete checkpoint")


def check_worlds(record, count, replay, *, partial=False):
    fields(
        record,
        (
            "mode",
            "h_s",
            "inputs",
            "controls",
            "delay",
            "rows",
            "checkpoint_19",
            "replay",
            "failure",
            "retained",
            "stats",
        ),
        "multiworld record",
    )
    require(len(record["inputs"]) == count, "all worlds retained")
    worlds = tuple(map(int, record["inputs"]))
    require(set(record["inputs"]) == set(record["controls"]), "full controls/inputs worlds")
    for data in record["inputs"].values():
        data_schema(data)
    same(record["h_s"], H, "multiworld h")
    same(record["delay"], 1, "multiworld motor history")
    if not partial:
        same(
            (len(record["rows"]), len(record["replay"])),
            (40, 21 if replay else 0),
            "full/replay lengths",
        )
        same(record["failure"], None, "full run failure")
    for i, row in enumerate(record["rows"]):
        full_frame(row, worlds)
        fields(row, ("entry", "audits"), "all per-world audits")
        same(set(row["audits"]), set(record["inputs"]), "allbody audit world coverage")
        state_map(row["entry"], worlds)
        for key in row["state"]:
            advance(row["entry"][key], row["state"][key])
            same(row["state"][key]["tick"], i + 1, "world tick")
            n = len(record["inputs"][key]["q"]) - 7
            same(len(row["state"][key]["history"]), n, "full motor delay history")
    if len(record["rows"]) >= 19:
        same(record["checkpoint_19"], record["rows"][18]["state"], "exact checkpoint19")
    for i, row in enumerate(record["replay"]):
        full_frame(row, worlds)
        for field in ("state", "geometry", "diagnostic", "energy"):
            exact(row[field], record["rows"][19 + i][field], "full replay field " + field)
    if record["rows"] and not partial:
        same(record["retained"], record["rows"][-1]["state"], "multiworld retained")


def physical_payload(member, relative, record, *, partial=False):
    """Read only the subset that the bound producer really saved."""
    scan(record, failure_prefix=partial)
    group = member.group
    if partial:
        # Validate the available prefix's shapes/types but do not demand that a
        # preserved first failure pass the physics assertions that actually failed.
        # It is never credited to complete coverage.
        if relative.endswith("floating-window.json.gz"):
            fields(
                record, ("input", "mode", "h_s", "steps", "rows", "failure", "retained", "stats")
            )
            data_schema(record["input"])
            require(len(record["rows"]) <= record["steps"], "first failure prefix length")
            require(record["failure"] is not None, "missing original first failure")
        elif group == "worlds":
            fields(record, ("inputs", "rows", "replay", "failure", "retained"))
            require(
                len(record["rows"]) <= 40 and len(record["replay"]) <= 21,
                "first failure world prefix length",
            )
            require(record["failure"] is not None, "missing original first failure")
        elif group == "public":
            fields(record, ("case", "model", "initial", "host_premise", "failure"))
        return
    if relative.endswith("floating-window.json.gz"):
        check_window(member, record, budget=group == "budget", partial=partial)
        return
    if group == "worlds":
        first = relative.split("/")[0]
        count = 1 if first.startswith("w1-") else int(first[1:])
        expected_worlds = (int(first[3:]),) if count == 1 else KEYS[:count]
        same(set(record["inputs"]), set(map(str, expected_worlds)), "fixed multiworld installation")
        for key, data in record["inputs"].items():
            same(
                len(data["q"]) - 7,
                (0, 1, 9, 16)[KEYS.index(int(key)) % 4],
                "original world N binding",
            )
        check_worlds(record, count, count != 1, partial=partial)
        return

    if group == "public":
        fields(
            record,
            (
                "case",
                "mode",
                "dt_s",
                "model",
                "initial",
                "literal_mass",
                "literal_acceleration",
                "host_premise",
                "drive",
                "external",
                "entry",
                "accepted",
                "geometry",
                "diagnostic",
                "energy",
                "stats",
            ),
        )
        require("failure" not in record, "public first-node incomplete")
        same(record["case"], "P03", "public input")
        same(record["dt_s"], H, "public h")
        advance(record["entry"], record["accepted"])
        close(record["diagnostic"]["mass"], record["literal_mass"], atol=2e-5)
        close(record["diagnostic"]["acceleration"], record["literal_acceleration"])
    elif group in ("mixed", "unbound"):
        fields(record, ("input", "affine_input", "controls", "rows", "stats"))
        same(len(record["rows"]), 40 if group == "mixed" else 2, "mixed full window")
        data_schema(record["input"])
        for i, row in enumerate(record["rows"]):
            full_frame(row, KEYS[:4])
            for key in row["state"]:
                advance(row["entry"][key], row["state"][key])
                same(row["state"][key]["tick"], i + 1, "mixed tick")
            if group == "unbound":
                fields(row, ("receipts",), "old-contact full receipt")
    elif group in ("invalid", "origin"):
        fields(record, ("problem", "input", "before", "retained", "stats"))
        same(record["problem"], member.parameter[0], "negative identity")
        same(record["retained"], record["before"]["state"], "invalid restore no publication")
    elif group == "late":
        fields(
            record,
            (
                "phase",
                "warm",
                "input",
                "affine_input",
                "before",
                "restore_target",
                "retried",
                "reference",
                "geometry",
                "diagnostic",
                "stats",
            ),
        )
        same((record["warm"], record["phase"]), member.parameter, "fault bank/phase")
        exact(record["retried"], record["reference"], "full retry checkpoints")
        state_map(record["retried"], KEYS[:4])
        if record["phase"] == "restore":
            same(record["retried"], record["restore_target"], "restore retry target")
            require(
                all(x is None for x in record["diagnostic"].values()),
                "restore invalidates all attempt diagnostics",
            )
    elif group == "domain":
        fields(record, ("kind", "input", "before", "recovered", "geometry", "stats"))
        same(record["kind"], member.parameter[0], "domain identity")
        state_map(record["recovered"], (503, 91))
    elif group == "budget":
        fields(record, ("input", "before", "retained", "stats"))
        same(
            record["retained"]["503"],
            record["before"]["503"],
            "low budget entire floating state retained",
        )
        advance(record["before"]["91"], record["retained"]["91"])
    elif group == "owned":
        fields(
            record,
            (
                "input",
                "packet",
                "before",
                "first",
                "first_geometry",
                "first_diagnostic",
                "retained",
                "stats",
            ),
        )
        for key in ("503", "91"):
            advance(record["before"][key], record["first"][key])
            advance(record["first"][key], record["retained"][key])
        same(
            record["first"]["503"]["velocity"],
            record["first"]["91"]["velocity"],
            "owned command two-world agreement",
        )
    elif group == "large":
        fields(
            record,
            ("input", "h_s", "initial", "first", "second", "second_diagnostic", "replay", "stats"),
        )
        same(record["input"]["q"][0], 2.0**60, "large anchor declaration")
        same(
            record["first"]["position"][:3],
            record["initial"]["position"][:3],
            "rounded global position",
        )
        same(record["first"]["local_origin_m"], [H, 0.0, 0.0], "actual local bank preserved")
        same(
            record["second"]["state"]["503"]["local_origin_m"],
            [2 * H, 0.0, 0.0],
            "second local bank",
        )
        for field in ("state", "geometry", "energy"):
            exact(record["second"][field], record["replay"][field], "large-anchor replay " + field)
    elif group == "hemisphere":
        fields(record, ("input", "declared_angle", "h_s", "rows", "stats"))
        same(record["declared_angle"], member.parameter[0], "pi neighbourhood input")
        same(len(record["rows"]), 2, "both pi-neighbourhood steps")
        for row in record["rows"]:
            fields(row, ("entry", "state", "geometry", "diagnostic", "expected_position"))
            advance(row["entry"], row["state"])
    elif group == "allocation":
        fields(
            record,
            (
                "input",
                "allocated_before_failure",
                "closed_stats",
                "entry",
                "state",
                "geometry",
                "diagnostic",
                "audit",
                "stats",
            ),
        )
        same(len(record["allocated_before_failure"]), 2, "real allocation fault boundary")
        same(
            record["closed_stats"]["device_owned_bytes"],
            0,
            "failed construction resources released",
        )
        advance(record["entry"], record["state"])
    elif group == "numeric":
        fields(record, ("case", "input", "mode", "stats"))
        same(record["case"], member.parameter[0], "numeric negative identity")
        array(record["input"], 1, 2)
        if member.parameter[0] in ("overflow", "invalid_division"):
            require(
                isinstance(record.get("actual_rejection"), str)
                and "non-finite" in record["actual_rejection"],
                "nonfinite must be an actual rejection, not an accepted NaN",
            )
            require("actual_result" not in record, "rejected arithmetic cannot be accepted output")
        else:
            fields(record, ("actual_result",))
            array(record["actual_result"], 1, 3)
            require("actual_rejection" not in record, "FTZ positive rejected")
    elif group in ("bounds", "folded", "child-tool"):
        fields(record, ("model", "bindings", "rows", "stats"))
        binding_schema(record)
        same(len(record["rows"]), 40, "finite materials all40")
        for index, row in enumerate(record["rows"]):
            fields(row, ("state", "result"))
            state_map(row["state"], (503,))
            same(row["state"]["503"]["tick"], index + 1, "material state tick")
            same(row["result"]["ticks"], [index + 1], "accepted-bounds tick")
            if group == "child-tool":
                fields(row, ("geometry", "diagnostic", "independent_authored_material"))
                require(
                    len(row["independent_authored_material"]) == len(record["model"]["shapes"]),
                    "all authored child-tool surfaces",
                )
            else:
                original_bounds(row)
        if group == "bounds":
            same(
                (len(record["input"]["q"]) - 7, record["input"]["placement"]),
                member.parameter,
                "finite geometry fixture",
            )
            same(
                record["stats"]["dynamic_envelope_uploaded_bytes"], 0, "no dynamic envelope upload"
            )
    elif group == "warm":
        fields(
            record,
            (
                "input",
                "model",
                "bindings",
                "entry",
                "bootstrap",
                "before",
                "plan_before",
                "rows",
                "final",
                "geometry",
                "independent_end_bounds",
                "stats",
            ),
        )
        binding_schema(record)
        count = member.parameter[0]
        state_map(record["entry"], KEYS[:count])
        state_map(record["final"], KEYS[:count])
        same((len(record["bootstrap"]), len(record["rows"])), (2, 40), "warm sample membership")
        for key in record["input"]:
            data_schema(record["input"][key])
            same(record["entry"][key]["tick"], 2, "warm entry tick")
            same(record["final"][key]["tick"], 42, "warm end tick")
        frozen_counts = (
            "snapshot_downloaded_bytes",
            "kinematics_downloaded_bytes",
            "tree_diagnostic_reads",
            "floating_state_downloaded_bytes",
            "floating_fk_downloaded_bytes",
            "floating_diagnostic_downloaded_bytes",
            "floating_state_reads",
            "floating_fk_reads",
            "floating_diagnostic_reads",
        )
        for index, row in enumerate(record["rows"], 3):
            same(
                set(row),
                {"result", "stats", "batch_stats"},
                "warm intentionally has no intermediate state",
            )
            same(row["result"]["ticks"], [index] * count, "warm accepted ticks")
            same(
                row["stats"]["device_allocations"],
                record["plan_before"]["device_allocations"],
                "no warm plan allocation",
            )
            same(
                row["batch_stats"]["device_allocations"],
                record["before"]["device_allocations"],
                "no warm batch allocation",
            )
            same(row["stats"]["dynamic_envelope_uploaded_bytes"], 0, "no warm dynamic envelope")
            for field in frozen_counts:
                same(row["batch_stats"][field], record["before"][field], "warm counter " + field)
        same(
            record["rows"][-1]["stats"]["floating_bounds_kernel_launches"]
            - record["plan_before"]["floating_bounds_kernel_launches"],
            40,
            "real warm launch count",
        )
        original_bounds(
            {
                "result": record["rows"][-1]["result"],
                "independent_material_bounds": record["independent_end_bounds"],
            }
        )
    elif group == "filter":
        fields(record, ("filter_kind", "input", "model", "bindings", "result", "stats"))
        binding_schema(record)
        same(record["filter_kind"], member.parameter[0], "filter identity")
        same(record["result"]["ticks"], [0], "current-pose query does not advance")
        same(
            len(record["result"]["candidates"]),
            1 if member.parameter[0] == "none" else 0,
            "declared filter result",
        )
    elif group == "bounds-atomic":
        fields(
            record,
            (
                "input",
                "model",
                "bindings",
                "before",
                "limited",
                "original",
                "retry",
                "restored",
                "current",
                "state",
                "geometry",
                "independent_bounds",
                "stats",
                "batch_stats",
            ),
        )
        binding_schema(record)
        same(record["original"], record["retry"], "query retry full values")
        for field in ("bounds", "ticks", "candidates"):
            same(
                record["original"][field], record["restored"][field], "same-tick geometry " + field
            )
        require(
            record["restored"]["state_epoch"] > record["original"]["state_epoch"],
            "same-tick restore invalidates old query",
        )
        original_bounds(
            {
                "result": record["current"],
                "independent_material_bounds": record["independent_bounds"],
            }
        )
    elif group == "close":
        fields(
            record,
            (
                "input",
                "original",
                "closed_stats",
                "closed_plans",
                "rebuilt",
                "state",
                "independent_bounds",
                "stats",
            ),
        )
        same(record["closed_stats"]["device_owned_bytes"], 0, "closed batch bytes")
        same(record["closed_stats"]["resident_collision_plans"], 0, "closed plan count")
        same(len(record["closed_plans"]), 2, "both plans retained")
        require(
            all(p["closed"] and p["device_owned_bytes"] == 0 for p in record["closed_plans"]),
            "all plan resources released",
        )
        for field in ("bounds", "candidates"):
            same(
                record["original"][field],
                record["rebuilt"][field],
                "fresh batch exact geometry " + field,
            )
    else:
        raise AuditError("no saved schema for " + group)


def strict_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key " + key)
        result[key] = value
    return result


def strict_load(raw):
    return json.loads(
        raw.decode("utf-8-sig"),
        object_pairs_hook=strict_object,
        parse_constant=lambda x: (_ for _ in ()).throw(AuditError("non-JSON constant " + x)),
    )


def normalized(path):
    return str(path).replace("\\", "/").casefold()


def hex_sha(value):
    require(
        isinstance(value, str) and re.fullmatch(r"[a-fA-F0-9]{64}", value),
        "explicit SHA256 required",
    )
    return value.lower()


class Auditor:
    def __init__(self, manifest):
        self.manifest = manifest
        self.files = {}
        self.total = 0
        self.coverage = {mode: set() for mode in ("host", "interpret", "cuda")}
        self.external = set()
        self.report = dict(
            status="incomplete",
            scope="saved records only; no producer/oracle/solver imported or rerun",
            jobs=[],
            errors=[],
            files=self.files,
            limits=dict(
                compressed=MAX_COMPRESSED,
                payload=MAX_PAYLOAD,
                total=MAX_TOTAL,
                files=MAX_FILES,
                depth=MAX_DEPTH,
            ),
            saving_boundaries=[
                "HOST15 and device11/mode have no saved payload; only bound pytest results cover them.",
                "Invalid restore tests save retained valid states, not the rejected malformed/NaN input.",
                "Domain and low-budget tests omit the instantaneous rejected diagnostic; that assertion remains receipt-only.",
                "Owned/read-only and injected callback behavior cannot be re-established from JSON.",
                "Warm middle frames intentionally have only bounds/ticks/counters; complete states exist at entry/end.",
                "Failure prefixes are retained separately and never complete missing tests.",
                "Numerical checks compare already-saved fields/oracles only; no physics or geometry is reconstructed.",
                "Native normalized-Euler orientation is not the device exp-map oracle; native decoder/source fields remain distinct.",
            ],
        )

    def read(self, path, root, limit=MAX_METADATA):
        path, root = Path(path).resolve(strict=True), Path(root).resolve(strict=True)
        require(
            path.is_relative_to(root) and path.is_file(),
            "input escapes declared root or is not a file",
        )
        require(path.stat().st_size <= limit, "file size bound " + str(path))
        with path.open("rb") as stream:
            raw = stream.read(limit + 1)
        require(len(raw) <= limit, "growing input exceeded file bound")
        digest = hashlib.sha256(raw).hexdigest()
        name = str(path)
        if name in self.files:
            same(self.files[name]["sha256"], digest, "evidence changed within audit")
        self.files[name] = dict(bytes=len(raw), sha256=digest)
        require(len(self.files) <= MAX_FILES, "total evidence file bound")
        return raw

    def text(self, path, root):
        return self.read(path, root).decode("utf-8-sig")

    def json(self, path, root, compressed=False):
        raw = self.read(path, root, MAX_COMPRESSED if compressed else MAX_METADATA)
        if compressed:
            with gzip.GzipFile(fileobj=io.BytesIO(raw)) as stream:
                raw = stream.read(MAX_PAYLOAD + 1)
            require(len(raw) <= MAX_PAYLOAD, "decompressed per-file bound")
            self.total += len(raw)
            require(self.total <= MAX_TOTAL, "cumulative decompression bound")
            self.files[str(Path(path).resolve())]["uncompressed_bytes"] = len(raw)
        return strict_load(raw)

    def receipt(self, spec):
        directory = Path(spec["directory"])
        config_path = Path(spec["config"])
        require(
            directory.is_absolute() and config_path.is_absolute(),
            "run/config must be explicit absolute paths",
        )
        directory = directory.resolve(strict=True)
        config = self.json(config_path, config_path.parent)
        start = self.json(directory / "start.json", directory)
        final = self.json(directory / "final.json", directory)
        same(start["task"], final["task"], "same ended run task")
        for field in ("owner", "exe", "args"):
            same(start[field], final[field], "start/final " + field)
            same(start[field], config[field], "config " + field)
        same(normalized(start["cwd"]), normalized(config["workdir"]), "working directory binding")
        same(start["head"], final["head"], "start/final HEAD")
        same(start["head"], spec["head"], "explicit expected HEAD")
        same(start["sources"], final["sources"], "source manifest fixed during run")
        raw_exit = int(self.text(directory / "RAW_EXIT.txt", directory).strip())
        same(raw_exit, final["raw_exit"], "final raw exit")
        same(raw_exit, spec["raw_exit"], "declared original raw exit")
        expected = {normalized(k): hex_sha(v) for k, v in spec["source_sha256"].items()}
        require(expected, "full explicit source SHA set required")
        before, after, text = {}, {}, {}
        for item in start["sources"]:
            key = normalized(item["path"])
            require(key not in before, "duplicate bound source")
            before[key] = hex_sha(item["sha256"])
            snapshot = directory / item["snapshot"]
            require(snapshot.resolve().parent == directory, "snapshot must be job local")
            raw = self.read(snapshot, directory)
            same(hashlib.sha256(raw).hexdigest(), before[key], "source snapshot hash")
            try:
                text[key] = raw.decode("utf-8-sig")
            except UnicodeDecodeError:
                # Binary artifacts remain fully hash-bound, never parsed as source.
                text[key] = None
        for item in final["sources_after"]:
            key = normalized(item["path"])
            require(key not in after, "duplicate after source")
            after[key] = hex_sha(item["sha256"])
        same(before, after, "all sources before/after")
        same(before, expected, "caller-declared complete source SHA set")
        same(set(before), {normalized(p) for p in config["files"]}, "config source coverage")
        stdout = self.text(directory / "stdout.log", directory)
        stderr = self.text(directory / "stderr.log", directory)
        return (
            directory,
            config["args"],
            text,
            stdout,
            dict(
                directory=str(directory),
                mode=spec["mode"],
                kind=spec["kind"],
                raw_exit=raw_exit,
                head=spec["head"],
                source_sha256=before,
                stderr_bytes=len(stderr.encode("utf-8")),
            ),
        )

    @staticmethod
    def source(sources, path):
        matches = [v for k, v in sources.items() if k.endswith("/" + path.casefold())]
        require(
            len(matches) == 1 and isinstance(matches[0], str),
            "missing/ambiguous/non-text bound source " + path,
        )
        return matches[0]

    def pytest_run(self, spec):
        directory, args, sources, stdout, summary = self.receipt(spec)
        device, host = membership()
        mode = spec["mode"]
        require(mode in self.coverage, "pytest declared mode")
        universe = host if mode == "host" else device
        by_node = {m.node: m for m in universe}
        selected, completed = spec["selected"], spec["completed"]
        require(
            selected and len(set(selected)) == len(selected) and set(selected) <= by_node.keys(),
            "explicit authored node whitelist",
        )
        require(completed == selected[: len(completed)], "successes must be selected first prefix")
        failed = spec.get("failed")
        if failed is None:
            same(completed, selected, "successful job complete whitelist")
            same(spec["raw_exit"], 0, "successful job exit")
        else:
            require(
                len(completed) < len(selected) and selected[len(completed)] == failed,
                "original first failure node",
            )
            same(spec["raw_exit"], 1, "failed prefix original exit")
            require("FAILED " + failed in stdout, "stdout first failure identity")
        require(
            not self.coverage[mode].intersection(completed),
            "same-mode duplicate successful node coverage",
        )
        self.source(sources, STEP)
        self.source(sources, REFERENCE)
        if mode != "host":
            self.source(sources, NUMERIC)
            self.source(sources, NUMERIC_TEST)
            require(
                "--tyx-mode" in args and args[args.index("--tyx-mode") + 1] == mode,
                "actual requested backend",
            )
        else:
            require("not tyx" in args, "HOST-only marker")
        # Bind selection to actual CLI, not merely the manifest or success count.
        entrypoints = [
            arg
            for i, arg in enumerate(args)
            if arg.startswith("tests/")
            and (arg.endswith(".py") or ".py::" in arg)
            and (i == 0 or args[i - 1] != "--deselect")
        ]
        require(
            entrypoints and "-k" not in args, "explicit pytest entrypoints, no opaque -k selector"
        )
        excluded = {args[i + 1] for i, arg in enumerate(args[:-1]) if arg == "--deselect"}
        actual_selection = []
        for target in entrypoints:
            for member in universe:
                if target in (
                    member.source,
                    member.source + "::" + member.function,
                    member.node,
                ) and not any(
                    value in excluded
                    for value in (member.node, member.source + "::" + member.function)
                ):
                    actual_selection.append(member.node)
        same(actual_selection, selected, "actual CLI ordered authored selection")
        for node in selected:
            member = by_node[node]
            require(
                "def " + member.function + "(" in self.source(sources, member.source),
                "authored test missing from exact snapshot",
            )
        lines = [line for line in stdout.splitlines() if re.search(r"\d+ (?:passed|failed)", line)]
        require(lines, "pytest final summary missing")
        final_line = lines[-1]
        for tag, count in (("passed", len(completed)), ("failed", int(failed is not None))):
            match = re.search(r"(\d+) " + tag, final_line)
            same(int(match.group(1)) if match else 0, count, "pytest original " + tag)
        require(
            not re.search(r"[1-9]\d* (?:error|skipped|xfailed|xpassed)", final_line),
            "skips/errors cannot complete membership",
        )
        roots = spec["payload_roots"]
        expected_root_nodes = {n for n in completed if by_node[n].files} | (
            {failed} if failed and by_node[failed].files else set()
        )
        same(set(roots), expected_root_nodes, "explicit saved-node roots")
        expected_paths, rows = set(), []
        all_nodes = completed + ([failed] if failed else [])
        for node in all_nodes:
            member = by_node[node]
            partial = node == failed
            row = dict(
                node=node,
                group=member.group,
                parameter=member.parameter,
                completion="first-failure-prefix"
                if partial
                else "saved-and-receipt"
                if member.files
                else "receipt-only-no-save",
                files=[],
            )
            filenames = spec.get("failed_payloads", []) if partial else list(member.files)
            require(set(filenames) <= set(member.files), "unrecognized first-failure payload path")
            for relative in filenames:
                base = directory / roots[node]
                require(base.resolve().is_relative_to(directory), "test root escapes run")
                path = base / relative
                require(
                    path.resolve().is_relative_to(base.resolve()), "payload escapes fixed test root"
                )
                normalized_path = path.resolve().relative_to(directory).as_posix()
                require(
                    normalized_path not in expected_paths, "payload reused across logical nodes"
                )
                expected_paths.add(normalized_path)
                record = self.json(path, directory, compressed=True)
                physical_payload(member, relative, record, partial=partial)
                if "mode" in record:
                    same(record["mode"], mode, "saved actual backend label")
                row["files"].append(normalized_path)
            rows.append(row)
        # Inventory only the explicit pytest payload root; it cannot choose membership.
        basetemp = directory / spec["basetemp"]
        require(basetemp.resolve().is_relative_to(directory), "basetemp root")
        found, entries = set(), 0
        pending = [basetemp]
        while pending:
            folder = pending.pop()
            for path in folder.iterdir():
                entries += 1
                require(entries <= MAX_FILES, "bounded saved-payload inventory")
                if path.is_symlink():
                    # pytest's *current convenience links are aliases, not original records.
                    require(path.name.endswith("current"), "unexpected payload symlink")
                    continue
                if path.is_dir():
                    # Do not walk runtime caches; only declared test/subcase paths can hold saved JSON.
                    if path.name in ("cache", "tyx", "compiled"):
                        continue
                    pending.append(path)
                elif path.name.endswith(".json.gz"):
                    found.add(path.resolve().relative_to(directory).as_posix())
        same(found, expected_paths, "complete gzip membership (including first-failure subset)")
        self.coverage[mode].update(completed)
        summary.update(
            selected=selected,
            completed=completed,
            failed=failed,
            members=rows,
            original_pytest_summary=final_line,
        )
        self.report["jobs"].append(summary)


def native_observation(value, b):
    fields(
        value,
        (
            "raw",
            "raw_dtypes",
            "actual_device",
            "position",
            "velocity",
            "centers",
            "linear_momentum",
            "angular_momentum_world_origin",
            "kinetic_energy_j",
            "public_native_jacobian",
            "public_native_mass",
            "world_to_native_velocity",
            "jacobian_world_com",
            "mass_world_com",
        ),
        "complete native observation",
    )
    d, q = b + 5, b + 6
    for name, shape in (
        ("joint_q", (q,)),
        ("joint_qd", (d,)),
        ("body_q", (b, 7)),
        ("body_qd", (b, 6)),
        ("body_f", (b, 6)),
    ):
        array(value["raw"][name], *shape)
        require(name in value["raw_dtypes"], "actual native dtype")
    array(value["position"], q)
    array(value["velocity"], d)
    array(value["centers"], b, 3)
    for name in ("public_native_jacobian", "jacobian_world_com"):
        array(value[name], b, 6, d)
    for name in ("public_native_mass", "world_to_native_velocity", "mass_world_com"):
        array(value[name], d, d)


def check_flags(value, required):
    fields(value, required, "complete named checks")
    for name, row in value.items():
        fields(
            row,
            ("passed", "max_abs_error", "atol", "rtol", "actual_shape", "expected_shape"),
            "quantitative check " + name,
        )
        require(row["passed"] is True, "saved failed check " + name)
        require(number(row["max_abs_error"]) and row["max_abs_error"] >= 0, "saved check error")
        same(row["actual_shape"], row["expected_shape"], "check compared full shape")


def native_payload(record, summary, mode):
    fields(
        record,
        (
            "status",
            "device",
            "h_s",
            "declared_steps",
            "cases",
            "errors",
            "gates",
            "script",
            "semantics",
            "versions",
            "sources",
        ),
        "native full trace",
    )
    same(record["status"], "passed", "native execution status")
    same(record["errors"], [], "native execution errors")
    same(record["device"], mode, "actual native device")
    same((record["h_s"], record["declared_steps"]), (H, 40), "native frozen window")
    same(record["versions"]["newton"], "1.7.0", "pinned Newton")
    same(record["versions"]["warp"], "1.17.0", "pinned Warp")
    same(
        tuple(x["id"] for x in record["cases"]), NATIVE_CASES, "native complete seven-case identity"
    )
    source_names = (
        "_src/sim/builder.py",
        "_src/sim/model.py",
        "_src/sim/state.py",
        "_src/sim/articulation.py",
        "_src/solvers/featherstone/kernels.py",
        "_src/solvers/featherstone/solver_featherstone.py",
    )
    same(set(record["sources"]), set(source_names), "native primary source snapshots")
    for source in record["sources"].values():
        fields(source, ("path", "sha256", "text"))
        hex_sha(source["sha256"])
        require(bool(source["text"]), "empty primary-source text")
    base_checks = (
        "original_mass",
        "original_com",
        "original_inertia",
        "initial_world_q",
        "initial_world_v",
        "joint_armature_zero",
        "joint_damping_zero",
        "joint_target_ke_zero",
        "joint_target_kd_zero",
        "joint_limit_ke_zero",
        "joint_limit_kd_zero",
    )
    step_checks = (
        "full_mass_world_com",
        "full_linear_J",
        "full_angular_J",
        "body_pose_from_declared_geometry",
        "body_twists_from_public_J",
        "public_parent_anchor_qd",
        "kinetic_mass_identity",
        "decoded_force_is_original",
        "decoded_acceleration",
        "actual_velocity_step",
        "root_COM_drift",
        "native_normalized_Euler_quaternion",
        "cumulative_linear_impulse",
    )
    for case in record["cases"]:
        fields(case, ("declaration", "metadata", "checks", "initial", "rows", "complete_steps"))
        same(case["complete_steps"], 40, "native completed count")
        same(len(case["rows"]), 40, "native actual rows")
        require("execution_error" not in case, "native case error")
        same(case["declaration"]["id"], case["id"], "native declaration identity")
        b = 1 if case["id"].startswith("P02") else 2
        meta = case["metadata"]
        fields(
            meta,
            (
                "body_count",
                "joint_count",
                "joint_dof_count",
                "joint_coord_count",
                "articulation_count",
                "device",
                "body_label",
                "joint_label",
                "shape_collision_filter_pairs",
                "body_mass",
                "body_com",
                "body_inertia",
                "joint_type",
                "joint_parent",
                "joint_child",
                "joint_X_p",
                "joint_X_c",
                "joint_axis",
                "joint_q_start",
                "joint_qd_start",
                "joint_dof_dim",
            ),
            "native actual model metadata",
        )
        same(
            (meta["body_count"], meta["joint_dof_count"], meta["joint_coord_count"]),
            (b, b + 5, b + 6),
            "actual native B/D/Q",
        )
        same(meta["device"], mode, "native model device")
        array(meta["body_mass"], b)
        array(meta["body_com"], b, 3)
        array(meta["body_inertia"], b, 3, 3)
        check_flags(case["checks"], base_checks)
        native_observation(case["initial"], b)
        previous = case["initial"]
        for index, row in enumerate(case["rows"]):
            fields(
                row,
                (
                    "index",
                    "entry_time_s",
                    "exit_time_s",
                    "entry",
                    "independent_entry",
                    "joint_f_input",
                    "checks",
                    "state_body_f_before_step",
                    "solver",
                    "accepted",
                    "expmap_alternative_not_native_oracle",
                    "completed",
                ),
                "native complete physical step",
            )
            same(
                (row["index"], row["entry_time_s"], row["exit_time_s"], row["completed"]),
                (index, index * H, (index + 1) * H, True),
                "native actual step identity",
            )
            same(row["entry"], previous, "native continuous own state")
            native_observation(row["entry"], b)
            native_observation(row["accepted"], b)
            array(row["joint_f_input"], b + 5)
            array(row["state_body_f_before_step"], b, 6)
            solver = row["solver"]
            fields(
                solver,
                (
                    "raw",
                    "raw_dtypes",
                    "decoded_com_world_wrench",
                    "decoded_world_com_acceleration",
                    "original_com_world_wrench",
                    "decoder",
                ),
                "post-RNEA reference-point receipt",
            )
            for name, width in (("body_q_com", 7), ("body_solve_origin", 3), ("body_f_ext", 6)):
                array(solver["raw"][name], b, width)
            for name in ("decoded_com_world_wrench", "original_com_world_wrench"):
                array(solver[name], b, 6)
            close(solver["decoded_com_world_wrench"], solver["original_com_world_wrench"])
            fields(
                row["independent_entry"],
                (
                    "mass",
                    "bias",
                    "external",
                    "acceleration",
                    "force",
                    "torque",
                    "linear",
                    "angular",
                ),
                "independent same-native-entry data",
            )
            required = list(step_checks)
            if b == 2:
                required.append("child_scalar_drift")
            if index == 0 and "literal_first_acceleration" in case["declaration"]:
                required.extend(
                    ("literal_independent_first_acceleration", "literal_native_first_acceleration")
                )
            if case["id"] in ("P01_freefall", "P03_internal", "P03_child_force", "P04_internal"):
                required.extend(
                    ("closed_window_velocity", "closed_window_root_COM", "closed_window_child_q")
                )
            check_flags(row["checks"], required)
            previous = row["accepted"]
    same(tuple(x["id"] for x in summary["cases"]), NATIVE_CASES, "native summary case membership")
    require(
        all(
            x["completed"] == 40
            and not x["failed_checks"]
            and not x["failed_steps"]
            and x["execution_error"] is None
            for x in summary["cases"]
        ),
        "native summary not all complete",
    )


def wheel_frame(frame):
    fields(
        frame,
        ("state", "geometry", "diagnostics", "energy", "broad", "batch_stats", "plan_stats"),
        "complete wheel frame",
    )
    state_map(frame["state"], KEYS[:4])
    same(set(frame["geometry"]), {"503", "91", "1207"}, "wheel three actual tree observations")
    for key, value in frame["geometry"].items():
        geometry(value)
        same(value["tick"], frame["state"][key]["tick"], "wheel accepted geometry")
    for value in frame["diagnostics"].values():
        if value is not None:
            diagnostic(value)
    broad(frame["broad"])
    same(
        dict(zip(map(str, frame["broad"]["world_keys"]), frame["broad"]["ticks"], strict=True)),
        {k: v["tick"] for k, v in frame["state"].items()},
        "wheel bounds current world ticks",
    )


def wheel_payload(record, summary, mode):
    fields(
        record,
        (
            "status",
            "mode",
            "world_order",
            "errors",
            "checks",
            "script",
            "expected_package_root",
            "gates",
            "units",
            "limits",
            "wheel",
            "modules_before",
            "modules_after",
            "versions",
            "declarations",
            "trace",
            "replay",
            "faults",
            "initial",
            "model",
            "bindings",
            "original_sources",
            "initial_observation",
            "floating_contacts_rejected",
            "checkpoint_19",
            "stale_query_rejected",
            "restored",
            "retained_at_exit",
            "before_close_stats",
            "before_close_plan_stats",
            "after_close_stats",
            "after_close_plan_stats",
            "closed_rejection",
        ),
        "complete external wheel trace",
    )
    same(record["status"], "passed", "wheel producer completion")
    same(record["errors"], [], "wheel execution/IO errors")
    same(record["mode"], mode, "wheel actual mode")
    same(record["world_order"], [1207, 6, 91, 503], "declared W4 installation")
    same(
        (len(record["trace"]), len(record["replay"]), len(record["faults"])),
        (40, 21, 2),
        "complete wheel windows",
    )
    same(
        (
            summary["complete_forward_steps"],
            summary["complete_replay_steps"],
            summary["complete_fault_banks"],
        ),
        (40, 21, 2),
        "wheel summary coverage",
    )
    required_checks = (
        "W4_continuous_40_independent_physics",
        "full_19_to_21_replay",
        "both_banks_invalid_and_retained_domain_retry",
        "owned_old_observations_unchanged",
        "floating_contact_remains_unsupported",
        "closed_resources_released",
    )
    fields(record["checks"], required_checks)
    require(all(x is True for x in record["checks"].values()), "wheel original checks failed")
    same(record["checks"], summary["checks"], "wheel summary checks")
    binding_schema(record)
    wheel_frame(record["initial_observation"])
    wheel_frame(record["restored"])
    same(record["checkpoint_19"], record["trace"][18]["state"], "wheel complete tick19 checkpoint")
    same(record["restored"]["state"], record["checkpoint_19"], "wheel legitimate restore")
    for index, row in enumerate(record["trace"]):
        wheel_frame(row)
        fields(
            row,
            (
                "input_packet",
                "entry",
                "independent_predicted",
                "checks_complete",
                "acceptance",
                "independent_material_bounds",
                "independent_true_pairs",
            ),
            "complete wheel forward row",
        )
        same(row["checks_complete"], True, "wheel whole row checked")
        same(row["acceptance"], dict.fromkeys(map(str, KEYS[:4]), True), "all W4 advance")
        for key in row["state"]:
            advance(row["entry"][key], row["state"][key])
            same(row["state"][key]["tick"], index + 1, "wheel forward tick")
            close(row["state"][key]["position"], row["independent_predicted"][key][0], atol=2e-4)
            close(row["state"][key]["velocity"], row["independent_predicted"][key][1], atol=2e-4)
    for index, row in enumerate(record["replay"]):
        wheel_frame(row)
        same(row["checks_complete"], True, "wheel replay complete")
        original = record["trace"][19 + index]
        for field in ("input_packet", "entry", "state", "geometry", "diagnostics", "energy"):
            exact(row[field], original[field], "exact full wheel replay " + field)
        for field in (
            "world_keys",
            "ticks",
            "shape_keys",
            "shape_revisions",
            "bounds",
            "registered_pairs",
            "candidates",
        ):
            exact(row["broad"][field], original["broad"][field], "exact material replay " + field)
    for bank, row in enumerate(record["faults"]):
        fields(
            row,
            (
                "target_bank",
                "entry",
                "checks_complete",
                "invalid_restore_input",
                "invalid_restore",
                "invalid_control_input",
                "invalid_control",
                "domain_entry",
                "domain_acceptance",
                "domain_exit",
                "retry_entry",
                "retry_acceptance",
                "retry_exit",
            ),
            "both-bank fault receipt",
        )
        same((row["target_bank"], row["checks_complete"]), (bank, True), "both banks exercised")
        for name in ("entry", "domain_entry", "domain_exit", "retry_exit"):
            wheel_frame(row[name])
        same(
            row["domain_acceptance"],
            {"503": False, "91": True, "1207": True, "6": True},
            "world-local domain rejection",
        )
        same(
            row["domain_exit"]["state"]["503"],
            row["domain_entry"]["state"]["503"],
            "entire rejected checkpoint unchanged",
        )
        same(
            row["domain_exit"]["diagnostics"]["503"],
            None,
            "no failed-attempt diagnostic publication",
        )
        same(row["retry_acceptance"], dict.fromkeys(map(str, KEYS[:4]), True), "all-world recovery")
        for key in row["retry_entry"]:
            advance(row["retry_entry"][key], row["retry_exit"]["state"][key])
    for field in ("after_close_stats", "after_close_plan_stats"):
        same(record[field]["device_owned_bytes"], 0, "real closed resource balance")
    require(record["after_close_plan_stats"]["closed"] is True, "plan closed")
    package = normalized(record["expected_package_root"]).rstrip("/") + "/"
    require(record["modules_before"] and record["modules_after"], "actual loaded module provenance")
    for group in ("modules_before", "modules_after"):
        names = set()
        for row in record[group]:
            fields(
                row, ("module", "path", "origin", "wheel_member", "sha256"), "loaded wheel module"
            )
            require(
                row["module"] not in names
                and (row["module"] == "phymeo" or row["module"].startswith("phymeo.")),
                "unique product modules",
            )
            names.add(row["module"])
            require(normalized(row["path"]).startswith(package), "non-wheel source path")
            same(normalized(row["path"]), normalized(row["origin"]), "module origin")
            require(row["wheel_member"].startswith("phymeo/"), "wheel member path")
            hex_sha(row["sha256"])
    fields(
        record["wheel"],
        (
            "path",
            "sha256",
            "installed_from",
            "version",
            "installer_recorded_sha256",
            "installed_product_files_verified",
        ),
        "archive provenance",
    )
    hex_sha(record["wheel"]["sha256"])
    same(record["wheel"], summary["wheel"], "original wheel binding")
    same(
        len(record["modules_after"]),
        summary["verified_module_count"],
        "all imported modules retained",
    )


def external_run(audit, spec):
    directory, args, sources, _stdout, job = audit.receipt(spec)
    kind, mode = spec["kind"], spec["mode"]
    require(kind in ("native", "wheel"), "external entrypoint kind")
    if kind == "native":
        require(mode in ("cpu", "cuda:0"), "canonical native device")
        source, filename, option = NATIVE_SOURCE, "native-floating.trace.json.gz", "--device"
        supplied = args[args.index(option) + 1] if option in args else None
        same("cuda:0" if supplied == "cuda" else supplied, mode, "actual native requested device")
    else:
        require(mode in ("interpret", "cuda"), "wheel mode")
        source, filename, option = WHEEL_SOURCE, "resident-floating-wheel.trace.json.gz", "--mode"
        require(option in args and args[args.index(option) + 1] == mode, "actual external backend")
    require(
        any(path.endswith("/" + source.rsplit("/", 1)[-1]) for path in sources),
        "bound original or copied external driver source",
    )
    declared_root = Path(spec["payload_directory"])
    root = (declared_root if declared_root.is_absolute() else directory / declared_root).resolve(
        strict=True
    )
    require("--output" in args, "external actual output option missing")
    actual_root = Path(args[args.index("--output") + 1])
    require(actual_root.is_absolute(), "external output must be explicitly absolute")
    same(root, actual_root.resolve(strict=True), "payload root bound to original external CLI")
    document = audit.json(root / filename, root, compressed=True)
    summary = audit.json(root / "summary.json", root)
    scan(document)
    scan(summary)
    saved = audit.files[str((root / filename).resolve())]
    same(summary["trace"]["sha256"], saved["sha256"], "same-run compressed trace SHA")
    same(
        summary["trace"]["uncompressed_bytes"],
        saved["uncompressed_bytes"],
        "original complete JSON byte count",
    )
    same(
        normalized(summary["trace"]["path"]),
        normalized((root / filename).resolve()),
        "summary original trace path",
    )
    same(summary["raw_exit"], spec["raw_exit"], "producer original exit vs receipt")
    require(
        spec["raw_exit"] == 0 and summary["status"] == "passed",
        "external incomplete run cannot provide full coverage",
    )
    # Installed copies of the driver may live outside the repository: bind by explicit hash.
    require(
        document["script"]["sha256"].lower()
        in {hex_sha(v) for v in spec["source_sha256"].values()},
        "same-run actual driver source hash",
    )
    if kind == "native":
        native_payload(document, summary, mode)
    else:
        wheel_payload(document, summary, mode)
    identity = kind + ":" + mode
    require(identity not in audit.external, "duplicate external complete membership")
    audit.external.add(identity)
    job.update(
        payloads=[filename, "summary.json"],
        membership=NATIVE_CASES
        if kind == "native"
        else ["40-forward", "21-replay", "bank0", "bank1"],
        scope="saved observations/checks/provenance only",
    )
    audit.report["jobs"].append(job)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(
        args.manifest.is_absolute() and args.output.is_absolute(),
        "explicit absolute manifest/output paths",
    )
    require(not args.output.exists(), "audit output must be a new file")
    require(args.manifest.stat().st_size <= MAX_METADATA, "manifest size bound")
    manifest_bytes = args.manifest.read_bytes()
    require(len(manifest_bytes) <= MAX_METADATA, "manifest grew beyond bound")
    manifest = strict_load(manifest_bytes)
    fields(manifest, ("schema", "required_modes", "required_external", "runs"), "audit manifest")
    same(manifest["schema"], "c9-saved-audit-v1", "manifest version")
    require(
        set(manifest["required_modes"]) <= {"host", "interpret", "cuda"}, "required public modes"
    )
    require(
        set(manifest["required_external"])
        <= {"native:cpu", "native:cuda:0", "wheel:interpret", "wheel:cuda"},
        "required external modes",
    )
    require(manifest["required_modes"] or manifest["required_external"], "empty acceptance request")
    require(
        isinstance(manifest["runs"], list) and 0 < len(manifest["runs"]) <= 64,
        "bounded explicit runs",
    )
    audit = Auditor(manifest)
    device, host = membership()
    audit.report["contract"] = dict(
        device_nodes=86,
        device_payloads=81,
        host_nodes=15,
        nodes=[
            dict(node=m.node, group=m.group, parameter=m.parameter, files=m.files)
            for m in host + device
        ],
    )
    audit.report["manifest_sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
    for spec in manifest["runs"]:
        try:
            if spec["kind"] == "pytest":
                audit.pytest_run(spec)
            elif spec["kind"] == "historical":
                _directory, _args, _sources, _stdout, row = audit.receipt(spec)
                require(bool(spec.get("reason")), "historical exclusion must name the reason")
                row.update(reason=spec["reason"], coverage="none; historical raw receipt only")
                audit.report["jobs"].append(row)
            else:
                external_run(audit, spec)
        except (
            AuditError,
            OSError,
            ValueError,
            TypeError,
            KeyError,
            IndexError,
            EOFError,
            RecursionError,
            OverflowError,
        ) as error:
            audit.report["errors"].append(
                dict(
                    directory=spec.get("directory"),
                    kind=spec.get("kind"),
                    type=type(error).__name__,
                    message=str(error),
                )
            )
    for mode in manifest["required_modes"]:
        expected = {m.node for m in (host if mode == "host" else device)}
        if audit.coverage[mode] != expected:
            audit.report["errors"].append(
                dict(
                    mode=mode,
                    missing=sorted(expected - audit.coverage[mode]),
                    unexpected=sorted(audit.coverage[mode] - expected),
                )
            )
    if audit.external != set(manifest["required_external"]):
        audit.report["errors"].append(
            dict(
                external_missing=sorted(set(manifest["required_external"]) - audit.external),
                external_unrequested=sorted(audit.external - set(manifest["required_external"])),
            )
        )
    audit.report["coverage"] = {mode: sorted(nodes) for mode, nodes in audit.coverage.items()}
    audit.report["external_coverage"] = sorted(audit.external)
    audit.report["decompressed_bytes"] = audit.total
    audit.report["status"] = (
        "saved_records_consistent" if not audit.report["errors"] else "incomplete_or_inconsistent"
    )
    audit.report["new_physics_pass_count"] = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(audit.report, stream, indent=2, allow_nan=False)
    print(
        json.dumps(
            dict(
                status=audit.report["status"],
                output=str(args.output),
                jobs=len(audit.report["jobs"]),
                errors=audit.report["errors"],
                new_physics_pass_count=0,
            ),
            allow_nan=False,
        )
    )
    return int(bool(audit.report["errors"]))


if __name__ == "__main__":
    raise SystemExit(main())
