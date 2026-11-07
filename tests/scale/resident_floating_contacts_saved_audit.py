"""C10 saved-evidence observer: authored membership, never a physics runner.

UNEXECUTED author candidate. Standard library only. This file does not import the
package, producers, tests, numerical libraries, or a solver. Read the companion
plan for manifest, same-run source binding, partial-prefix, and saving limits.
"""

import argparse
from dataclasses import dataclass
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import re
import zipfile

STEP = "tests/scale/test_resident_floating_contacts.py"
TRANSACTIONS = "tests/scale/test_resident_floating_contact_transactions.py"
REFERENCE = "tests/reference_newton/robot/resident_floating_contact_reference.py"
BASE_REFERENCE = "tests/reference_newton/robot/resident_floating_reference.py"
NATIVE = "tests/reference_newton/robot/resident_floating_contacts_native.py"
WHEEL = "tests/scale/resident_floating_contact_wheel_consumer.py"
H = 1 / 1024
ATOL, RTOL, WINDOW_ATOL = 5e-6, 5e-5, 2e-4
MAX_COMPRESSED, MAX_PAYLOAD = 64 << 20, 256 << 20
MAX_TOTAL, MAX_METADATA, MAX_FILES, MAX_DEPTH = 2 << 30, 16 << 20, 8192, 96
WORLDS = (503, 91, 1207, 6)
WINDOW = ("floating-contact-window.json.gz",)


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
    parameter: object
    suffix: str | None
    files: tuple

    @property
    def node(self):
        result = self.source + "::" + self.function
        return result if self.suffix is None else result + "[" + self.suffix + "]"


def membership():
    """Literal authored order/parameters, fixed before any output was read.

    Tyx mode is a CLI fixture, not a pytest parameter. NumPy's bool_(True) is
    the non-builtin value in the flag parameter and receives fallback ID flag1.
    No payload discovery, producer import, AST execution or pytest collection.
    """
    rows = []

    def add(source, function, group, values, files):
        for suffix, parameter in values:
            rows.append(Member(source, function, group, parameter, suffix, files))

    add(STEP, "test_f01_public_root_contact_keeps_full_six_coordinate_response", "f01",
        [(None, {})], ("f01-public-floating-contact.json.gz",))
    add(STEP, "test_f02_root_child_and_internal_motor_use_one_full_matrix", "f02",
        [(v, {"variant": v}) for v in ("child", "motor", "root")], WINDOW)
    add(STEP, "test_f03_two_dynamic_siblings_react_and_transform", "f03", [
        ("base-False-False", {"placement": "base", "reverse": False, "boost": False}),
        ("base-True-False", {"placement": "base", "reverse": True, "boost": False}),
        ("cycle-True-False", {"placement": "cycle", "reverse": True, "boost": False}),
        ("base-False-True", {"placement": "base", "reverse": False, "boost": True}),
    ], WINDOW)
    add(STEP, "test_f04_rotating_tool_uses_entry_point_velocity_and_true_arc", "f04", [(None, {})], WINDOW)
    add(STEP, "test_f05_com_offset_and_rotated_principal_inertia_react", "f05", [
        (v + "-" + p, {"variant": v, "placement": p})
        for v, p in (("origin", "base"), ("frame", "base"), ("frame", "cycle"))
    ], WINDOW)
    add(STEP, "test_f06_authored_fixed_tool_is_folded_once_and_reacts", "f06", [(None, {})], WINDOW)
    add(STEP, "test_f07_finite_shape_geometry_anchor_then_full_response", "f07",
        [(v, {"variant": v}) for v in ("hull", "cap_side", "cap_end", "cap_miss", "CC", "HH")],
        ("floating-shape-response.json.gz",))
    add(STEP, "test_f08_nonaxis_full_tree_and_last_physical_body", "f08", [
        (str(n) + "-" + str(v), {"n": n, "variant": v})
        for n, v in ((9, None), (16, None), (16, "body15"), (16, "both"))
    ], WINDOW)
    add(STEP, "test_f08_shared_revolute_ancestor_keeps_complete_cross_coupling", "branch", [(None, {})], WINDOW)
    add(STEP, "test_f09_loaded_separated_and_reentered_without_reset", "f09", [(None, {})], WINDOW)
    add(TRANSACTIONS, "test_f10_single_four_thirtytwo_worlds_and_full_restore21", "batch", [(None, {})],
        tuple(folder + "/floating-contact-batch.json.gz" for folder in ("w1-503", "w1-91", "w1-1207", "w1-6", "w4", "w32")))
    add(TRANSACTIONS, "test_f10_capacity_retains_root_and_last_body_all_banks", "capacity", [
        (str(warm) + "-" + str(world), {"warm": warm, "limited_world": world})
        for warm in (0, 1) for world in (503, 6)
    ], ("floating-contact-capacity.json.gz",))
    add(TRANSACTIONS, "test_f10_drive_delay_cap_and_immediate_external_full_response", "delay", [
        (str(delay) + "-" + str(cap), {"delay": delay, "cap": cap})
        for delay in (0, 1) for cap in (0.125, 8.0)
    ], WINDOW)
    add(TRANSACTIONS, "test_f10_actual_late_fault_retains_every_bank_and_retry_matches", "late", [
        (phase + "-" + str(warm), {"phase": phase, "warm": warm})
        for phase in ("upload", "candidate", "receipt") for warm in (0, 1)
    ], ("floating-contact-late-fault.json.gz",))
    add(TRANSACTIONS, "test_f10_restore_after_real_candidate_rejects_stale_attempt", "stale",
        [(str(w), {"warm": w}) for w in (0, 1)], ("floating-contact-stale.json.gz",))
    add(TRANSACTIONS, "test_f10_partial_allocation_and_global_candidate_cap_clean_up", "resource", [(None, {})],
        ("floating-contact-resource.json.gz",))
    add(TRANSACTIONS, "test_f10_owned_receipt_foreign_batch_and_close_rebuild", "close",
        [(n, {"name": n}) for n in ("F01", "F03")], ("floating-contact-close.json.gz",))
    add(TRANSACTIONS, "test_f10_old_fixed_affine_scalar_and_unbound_floating_publish_together", "mixed", [(None, {})],
        ("floating-contact-old-mixed.json.gz",))
    add(TRANSACTIONS, "test_f10_explicit_pair_filters_precede_actual_parent_gate", "filter",
        [(k, {"kind": k}) for k in ("default", "same_body", "mask", "excluded", "parent_mask", "parent_excluded")],
        ("floating-contact-domain.json.gz",))
    add(TRANSACTIONS, "test_f10_invalid_graph_material_source_and_domain_reject_before_allocation", "invalid_graph",
        [(p, {"problem": p}) for p in ("parent", "body_alias", "two_bodies_one_group", "world", "missing", "extra",
                                      "friction", "law", "sampling", "foreign_topology", "source_revision")], ())
    add(TRANSACTIONS, "test_f10_dynamic_opt_in_requires_builtin_bool", "invalid_bool", [
        ("1", {"type": "int", "value": 1}), ("flag1", {"type": "numpy.bool_", "value": True}),
        ("true", {"type": "str", "value": "true"}), ("None", {"type": "NoneType", "value": None}),
    ], ())
    add(TRANSACTIONS, "test_f10_physical_body_count_is_not_motor_or_padded_count", "invalid_body", [
        (str(n) + "-" + str(g), {"n": n, "invalid_group": g}) for n, g in ((16, 17), (15, 16), (0, 1))
    ], ())
    add(TRANSACTIONS, "test_f10_larger_floating_body_limit_does_not_invent_fixed_bodies", "old_body",
        [(k, {"kind": k}) for k in ("ordinary", "affine")], ())
    add(TRANSACTIONS, "test_f10_huge_anchor_losing_nonzero_material_offset_rejects_without_repair", "huge_anchor",
        [(None, {})], ("floating-contact-source-representability.json.gz",))
    add(TRANSACTIONS, "test_f10_actual_candidate_material_packet_mismatch_is_not_published", "source_fault",
        [(k, {"field": k}) for k in ("shape", "group", "world")], ("floating-contact-candidate-source.json.gz",))
    require(len(rows) == 76 and len({m.node for m in rows}) == 76, "authored device membership must remain 76")
    require(sum(len(m.files) for m in rows) == 61, "authored payload membership must remain 61")
    require(sum(not m.files for m in rows) == 20, "20 no-save nodes are receipt-only")
    return tuple(rows)


def fields(value, names, label):
    require(isinstance(value, dict), label + " must be an object")
    require(set(names) <= value.keys(), label + " missing fields: " + str(sorted(set(names) - value.keys())))


def number(value, label="number"):
    require(type(value) in (int, float) and math.isfinite(value), label + " must be finite numeric")
    return value


def array(value, *shape):
    if not shape:
        number(value)
        return
    require(isinstance(value, list) and len(value) == shape[0], "saved array dimensions " + str(shape))
    for element in value:
        array(element, *shape[1:])


def exact(actual, expected, label):
    # Key order is nonphysical; no fields/values (including -0.0) are removed.
    a = json.dumps(actual, sort_keys=True, separators=(",", ":"), allow_nan=False)
    b = json.dumps(expected, sort_keys=True, separators=(",", ":"), allow_nan=False)
    require(a == b, label + " differs in saved complete values")


def same(actual, expected, label):
    if actual != expected:
        raise AuditError(label + ": " + str(actual)[:160] + " != " + str(expected)[:160])


def close(actual, expected, label, *, atol=ATOL, rtol=RTOL):
    """Compare already saved numbers; never produce a dynamics/geometry oracle."""
    if isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), label + " shape")
        for a, b in zip(actual, expected, strict=True):
            close(a, b, label, atol=atol, rtol=rtol)
    else:
        number(actual, label)
        number(expected, label)
        require(abs(actual - expected) <= atol + rtol * abs(expected), label + " saved discrepancy")


def scan(value, *, partial=False, depth=0):
    require(depth <= MAX_DEPTH, "bounded JSON nesting")
    if isinstance(value, dict):
        if set(value) == {"nonfinite"}:
            require(partial and value["nonfinite"] in ("nan", "inf", "-inf"), "nonfinite only in a declared failed prefix")
        else:
            for item in value.values():
                scan(item, partial=partial, depth=depth + 1)
    elif isinstance(value, list):
        for item in value:
            scan(item, partial=partial, depth=depth + 1)
    elif isinstance(value, float):
        require(math.isfinite(value), "unwrapped nonfinite JSON")
    else:
        require(value is None or type(value) in (str, int, bool), "unexpected JSON type")


def checkpoint(value):
    fields(value, ("world", "tick", "position", "velocity", "applied_effort", "history", "control"), "checkpoint")
    require(type(value["world"]) is int and type(value["tick"]) is int and value["tick"] >= 0, "checkpoint world/tick")
    if "local_origin_m" in value:
        fields(value, ("topology", "generalized_applied"), "floating checkpoint")
        n = len(value["position"]) - 7
        require(0 <= n <= 16, "floating N")
        array(value["position"], n + 7)
        array(value["velocity"], n + 6)
        array(value["applied_effort"], n)
        array(value["generalized_applied"], n + 6)
        array(value["local_origin_m"], 3)
        require(value["topology"], "full original topology token")
        require(len(value["control"]) == 4, "floating drive/effort/force/torque held blocks")
        for block, shape in zip(value["control"], ((n, 3), (n,), (n + 1, 3), (n + 1, 3)), strict=True):
            array(block, *shape)
        if value["history"]:
            array(value["history"], n, 3)
        return n
    k = len(value["position"])
    array(value["position"], k)
    array(value["velocity"], k)
    if "full_position" in value:
        n = len(value["full_position"])
        for name in ("full_position", "full_velocity", "full_applied_effort", "full_generalized_applied"):
            array(value[name], n)
        require(len(value["control"]) == 2, "affine drive/full loads")
        array(value["control"][0], k, 3)
        array(value["control"][1], n, 7)
        array(value["generalized_applied"], k)
    elif "generalized_applied" in value:
        array(value["control"], k, 10)
        array(value["generalized_applied"], k)
    return None


def state_map(values, worlds=None):
    require(isinstance(values, dict) and values, "nonempty complete state map")
    if worlds is not None:
        same(set(values), {str(w) for w in worlds}, "full world membership")
    for key, value in values.items():
        checkpoint(value)
        same(key, str(value["world"]), "saved world identity")


def geometry(value, state=None):
    fields(value, ("tick", "position", "velocity", "link_position_m", "link_rotation", "com_position_m"), "kinematics")
    b = len(value["link_position_m"])
    array(value["link_position_m"], b, 3)
    array(value["com_position_m"], b, 3)
    array(value["link_rotation"], b, 3, 3)
    if "configuration_count" in value:
        n = value["joint_count"]
        same([value["body_count"], value["velocity_count"], value["configuration_count"]], [n + 1, n + 6, n + 7], "N/B/D/Q")
        same(b, n + 1, "all physical body FK")
        same(len(value["joint_types"]), n, "all joint kinds")
        array(value["position"], n + 7)
        array(value["velocity"], n + 6)
        array(value["linear_velocity_m_s"], b, 3)
        array(value["angular_velocity_rad_s"], b, 3)
    if state is not None:
        same(value["tick"], state["tick"], "accepted FK tick")
        q = state.get("full_position", state["position"])
        v = state.get("full_velocity", state["velocity"])
        close(value["position"], q, "FK vs full accepted coordinates", atol=0, rtol=0)
        close(value["velocity"], v, "FK vs full accepted velocity", atol=0, rtol=0)


def diagnostic(value):
    fields(value, ("entry_tick", "exit_tick", "dt_s", "entry_position", "entry_velocity", "mass", "bias", "external",
                   "linear_jacobian", "angular_jacobian", "drive", "acceleration", "iterations", "equation_residual", "control_residual"), "mechanics diagnostic")
    d, b = len(value["mass"]), len(value["linear_jacobian"])
    array(value["mass"], d, d)
    for name in ("bias", "external"):
        array(value[name], d)
    for name in ("linear_jacobian", "angular_jacobian"):
        array(value[name], b, 3, d)
    if "reduced_mass" in value:
        k = len(value["entry_position"])
        array(value["reduced_mass"], k, k)
        for name in ("reduced_bias", "reduced_external", "drive", "acceleration"):
            array(value[name], k)
        for name in ("full_acceleration", "constraint_effort_full"):
            array(value[name], d)
        for name in ("reduced_linear_jacobian", "reduced_angular_jacobian"):
            array(value[name], b, 3, k)
        fields(value, ("row_multipliers", "row_powers_w", "constraint_power_w"), "original affine reactions")
    else:
        array(value["acceleration"], d)
    if len(value["entry_position"]) == d + 1:
        n = d - 6
        require(0 <= n <= 16 and b == n + 1, "floating full mechanics dimensions")
        array(value["drive"], n)
        array(value["entry_position"], n + 7)
        array(value["entry_velocity"], n + 6)
    same(value["exit_tick"], value["entry_tick"] + 1, "same mechanics attempt")
    same(value["dt_s"], H, "original h")
    require(0 <= number(value["equation_residual"]) <= 2e-5, "saved mass equation residual")
    require(0 <= number(value["control_residual"]) <= 5e-5, "saved original PD residual")


def key_tuple(value):
    fields(value, ("world", "body", "shape"), "CollisionKey")
    require(all(type(value[k]) is int for k in ("world", "body", "shape")), "integer collision identity")
    return tuple(value[k] for k in ("world", "body", "shape"))


def shape_schema(value):
    fields(value, ("key", "kind", "vertex_ids", "radius", "group", "mask", "revision", "normal", "triangles", "heightfield", "curved"), "complete source shape")
    key = key_tuple(value["key"])
    require(value["kind"] in ("sphere", "capsule", "hull"), "frozen C10 finite shape domain")
    ids = value["vertex_ids"]
    require(ids and all(type(i) is int for i in ids) and len(set(ids)) == len(ids), "original material IDs")
    number(value["radius"])
    return key


def bindings_schema(model, bindings):
    fields(model, ("shapes", "excluded"), "original collision model")
    shapes = {shape_schema(s): s for s in model["shapes"]}
    require(len(shapes) == len(model["shapes"]) == len(bindings), "one complete binding per source shape")
    require({key_tuple(v["key"]) for v in bindings} == set(shapes), "binding-key coverage")
    result = {}
    for binding in bindings:
        key = key_tuple(binding["key"])
        source = shapes[key]
        if "group_index" in binding:
            same(binding["world"], key[0], "binding actual world")
            require(type(binding["group_index"]) is int and binding["group_index"] >= 0, "true physical group")
            array(binding["local_points_m"], len(source["vertex_ids"]), 3)
            if binding.get("_source_shape") is None:
                require(binding["topology"], "explicit installation needs the owned full topology")
            else:
                # from_asset retains original source shape/model independently
                # of per-world gains/anchor topology; topology=None is valid.
                original = binding["_source_shape"]
                shape_schema(original)
                require(binding.get("_model_token"), "asset binding needs its original full model")
                exact(dict(original, key=binding["key"]), source, "asset shape with only the declared key remapped")
        else:
            fields(binding, ("motion",), "finite static binding")
            array(binding["motion"]["start"], len(source["vertex_ids"]), 3)
        result[key] = binding
    return shapes, result


def binding_matches_state(binding, state):
    """Read original saved identities; no model/asset compilation or FK."""
    same(binding["world"], state["world"], "binding physical checkpoint owner")
    topology = state["topology"]
    require(isinstance(topology, list) and topology, "owned actual checkpoint topology")
    if binding.get("_source_shape") is None:
        exact(binding["topology"], topology, "explicit binding full checkpoint topology")
        return
    kind = topology[0]
    if kind == "floating_tree_v1":
        model = topology[1]
    elif kind == "fixed_tree_v1":
        model = topology[3]
    elif kind == "affine_tree_v1":
        graph = topology[1]
        require(graph[0] == "affine_graph_v1", "original full affine graph")
        exact(binding["_graph_token"], graph, "asset graph matches owned affine coordinates")
        model = graph[1]
    else:
        raise AuditError("unsupported saved asset topology " + str(kind))
    exact(binding["_model_token"], model, "asset original physical model matches accepted world")


def frames_schema(values, world, geometry_value, shapes=None, bindings=None):
    require(isinstance(values, dict), "complete source frame mapping")
    result = {}
    for frame in values.values():
        fields(frame, ("shape", "world_points_m", "group_index", "position_m", "rotation", "reference_point_m",
                       "linear_velocity_m_s", "angular_velocity_rad_s", "rotation_orthogonality_error"), "source frame")
        key = shape_schema(frame["shape"])
        same(key[0], world, "packet world")
        require(key not in result, "duplicate packet owner")
        result[key] = frame
        array(frame["world_points_m"], len(frame["shape"]["vertex_ids"]), 3)
        for field in ("position_m", "reference_point_m", "linear_velocity_m_s", "angular_velocity_rad_s"):
            array(frame[field], 3)
        array(frame["rotation"], 3, 3)
        number(frame["rotation_orthogonality_error"])
        if shapes is not None:
            require(key in shapes, "frame belongs to declared original shape")
            exact(frame["shape"], shapes[key], "same complete source shape")
        if bindings is not None:
            same(frame["group_index"], bindings[key].get("group_index"), "same real material owner")
        group = frame["group_index"]
        if group is not None:
            require(type(group) is int and 0 <= group < len(geometry_value["link_position_m"]), "physical group inside actual B")
            for target, source in (("position_m", "link_position_m"), ("reference_point_m", "com_position_m"), ("rotation", "link_rotation")):
                close(frame[target], geometry_value[source][group], "stored packet/FK " + target, atol=0, rtol=0)
        else:
            close(frame["position_m"], [0, 0, 0], "static origin", atol=0, rtol=0)
            close(frame["linear_velocity_m_s"], [0, 0, 0], "static linear velocity", atol=0, rtol=0)
            close(frame["angular_velocity_rad_s"], [0, 0, 0], "static angular velocity", atol=0, rtol=0)
    return result


def receipt(value, state=None, *, shapes=None, bindings=None):
    fields(value, ("world", "entry_tick", "exit_tick", "dt_s", "entry_state", "accepted_state", "entry_geometry", "accepted_geometry",
                   "input_external", "selected_drive", "law_wrench_fp64", "uploaded_wrench_fp32", "contact_generalized", "actual_drive",
                   "sites", "static_reactions", "diagnostics"), "complete contact receipt")
    entry, end, diag = value["entry_state"], value["accepted_state"], value["diagnostics"]
    n = checkpoint(entry)
    checkpoint(end)
    same(value["world"], entry["world"], "receipt owner")
    same(value["world"], end["world"], "accepted owner")
    same(value["entry_tick"], entry["tick"], "receipt entry tick")
    same(value["exit_tick"], end["tick"], "receipt accepted tick")
    same(end["tick"], entry["tick"] + 1, "one accepted contact drift")
    same(value["dt_s"], H, "contact h")
    exact(end["control"], entry["control"], "held not overwritten by temporary contact")
    geometry(value["entry_geometry"], entry)
    geometry(value["accepted_geometry"], end)
    exact(entry["topology"], end["topology"], "same physical model over accepted step")
    if bindings is not None:
        for key, binding in bindings.items():
            if key[0] == value["world"] and "group_index" in binding:
                binding_matches_state(binding, entry)
    if state is not None:
        exact(end, state, "receipt accepted complete checkpoint")
    b, d = len(value["law_wrench_fp64"]), len(value["contact_generalized"])
    array(value["law_wrench_fp64"], b, 6)
    array(value["uploaded_wrench_fp32"], b, 6)
    array(value["input_external"], d)
    array(value["contact_generalized"], d)
    if n is not None:
        same((b, d), (n + 1, n + 6), "full body vs velocity rows")
        array(value["selected_drive"], n, 3)
        array(value["actual_drive"], n)
        same(diag["coordinate_kind"], "floating", "floating receipt kind")
        same([diag[k] for k in ("joint_count", "body_count", "velocity_count", "position_count")], [n, n + 1, n + 6, n + 7], "receipt N/B/D/Q")
        nested = diag["floating_diagnostic"]
        diagnostic(nested)
        for name in ("mass", "bias", "linear_jacobian", "angular_jacobian", "acceleration", "equation_residual"):
            exact(diag[name], nested[name], "same-attempt diagnostic " + name)
        exact(diag["total_external"], nested["external"], "same total external")
        exact(value["actual_drive"], nested["drive"], "same actual motor")
        same(nested["entry_tick"], entry["tick"], "same nested sample time")
        close(nested["entry_position"], entry["position"], "actual entry q", atol=0, rtol=0)
        close(nested["entry_velocity"], entry["velocity"], "actual entry v", atol=0, rtol=0)
    fields(diag, ("entry_frames", "accepted_frames", "mass", "bias", "linear_jacobian", "angular_jacobian", "acceleration",
                  "equation_residual", "control_residual", "entry_kinetic_energy_j", "accepted_kinetic_energy_j",
                  "entry_power_times_h_j", "application_work_j", "material_work_j"), "complete contact mechanics/work")
    before_frames = frames_schema(diag["entry_frames"], value["world"], value["entry_geometry"], shapes, bindings)
    after_frames = frames_schema(diag["accepted_frames"], value["world"], value["accepted_geometry"], shapes, bindings)
    same(set(before_frames), set(after_frames), "same material source coverage")
    for key in before_frames:
        exact(before_frames[key]["shape"], after_frames[key]["shape"], "same source over accepted motion")
        same(before_frames[key]["group_index"], after_frames[key]["group_index"], "same body owner over motion")
    for site in value["sites"]:
        fields(site, ("pair", "geometry", "law", "application_work_j", "material_work_j"), "site")
        pair = tuple(key_tuple(key) for key in site["pair"])
        require(len(pair) == 2 and pair[0] < pair[1] and pair[0][0] == pair[1][0] == value["world"], "canonical same-world pair")
        require(set(pair) <= before_frames.keys(), "site material registered in source packet")
        fields(site["geometry"], ("witness", "load_point", "weight"), "complete ContactSite")
        array(site["geometry"]["load_point"], 3)
        require(number(site["geometry"]["weight"]) > 0, "positive quadrature weight")
        witness = site["geometry"]["witness"]
        fields(witness, ("gap_m", "normal", "first_point", "second_point", "first_weights", "second_weights", "source",
                         "axis_parameter_span", "first_offset", "second_offset", "intersection_points", "shared_material_points",
                         "first_height_m", "second_height_m", "capsule_axis_span"), "original complete witness")
        number(witness["gap_m"])
        array(witness["normal"], 3)
        for index, side in enumerate(("first", "second")):
            array(witness[side + "_point"], 3)
            if witness[side + "_offset"] is not None:
                array(witness[side + "_offset"], 3)
            ids = before_frames[pair[index]]["shape"]["vertex_ids"]
            weights = witness[side + "_weights"]
            require(weights and len({item[0] for item in weights}) == len(weights), "full unambiguous material weights")
            for address in weights:
                require(len(address) == 2 and address[0] in ids and number(address[1]) >= 0, "original material vertex address")
            for suffix in ("velocity_m_s", "force_n", "reference_point_m", "material_displacement_m", "application_displacement_m"):
                array(site[side + "_" + suffix], 3)
            close(site[side + "_reference_point_m"], before_frames[pair[index]]["reference_point_m"], "actual COM/static reference", atol=0, rtol=0)
            if before_frames[pair[index]]["group_index"] is None:
                for suffix in ("material_displacement_m", "application_displacement_m"):
                    close(site[side + "_" + suffix], [0, 0, 0], "static actual displacement", atol=0, rtol=0)
        same(site["law"]["friction"], 0, "frozen frictionless material")
        for name in ("application_work_j", "material_work_j"):
            number(site[name])
    for reaction in value["static_reactions"].values():
        for name in ("reference_point_m", "force_n", "torque_nm"):
            array(reaction[name], 3)
    for name in ("entry_kinetic_energy_j", "accepted_kinetic_energy_j", "entry_power_times_h_j", "application_work_j", "material_work_j"):
        number(diag[name])


def record_frame(row, worlds=None, *, shapes=None, bindings=None):
    fields(row, ("state", "geometry", "receipts", "summary"), "complete saved frame")
    state_map(row["state"], worlds)
    tree_worlds = {w for w, state in row["state"].items() if "generalized_applied" in state}
    same(set(row["geometry"]), tree_worlds, "complete public FK tree-world coverage")
    same(set(row["summary"]), set(row["state"]), "complete public summary world coverage")
    if "diagnostics" in row:
        same(set(row["diagnostics"]), tree_worlds, "complete public diagnostic tree-world coverage")
    bound_worlds = (tree_worlds if bindings is None else
                    {str(key[0]) for key, binding in bindings.items() if "group_index" in binding})
    require(bound_worlds <= tree_worlds, "contact worlds are actual declared trees")
    if "acceptance" in row:
        # A newly prepared limited-capacity plan may have no old receipt even
        # though its rejected world's checkpoint tick is already positive.
        wanted_receipts = {w for w in bound_worlds if row["acceptance"][w] or w in row["entry_receipts"]}
    else:
        # Original test snapshots are either fresh tick0 or an already stepped
        # plan; no mid-restore plan publication is silently inferred here.
        wanted_receipts = {w for w in bound_worlds if row["state"][w]["tick"] > 0}
    same(set(row["receipts"]), wanted_receipts, "complete actual bound-world receipt coverage")
    for w, geo in row["geometry"].items():
        require(w in row["state"], "FK owner in accepted map")
        geometry(geo, row["state"][w])
    for w, measured in row["receipts"].items():
        require(w in row["state"], "receipt owner in accepted map")
        receipt(measured, row["state"][w], shapes=shapes, bindings=bindings)
    for w, energy in row["summary"].items():
        require(w in row["state"], "summary owner")
        number(energy["kinetic_energy_j"])
    for w, value in row.get("diagnostics", {}).items():
        if value is not None:
            diagnostic(value)
            same(value["exit_tick"], row["state"][w]["tick"], "latest accepted diagnostic")


def input_schema(data):
    fields(data, ("name", "root_mass", "root_com", "root_inertia", "root_inertia_frame", "gravity", "children", "q", "v",
                  "drive", "effort", "force", "torque"), "full original floating input")
    n = len(data["q"]) - 7
    require(0 <= n <= 16, "input joint count")
    array(data["q"], n + 7)
    array(data["v"], n + 6)
    array(data["drive"], n, 3)
    array(data["effort"], n)
    array(data["force"], n + 1, 3)
    array(data["torque"], n + 1, 3)
    for name, size in (("root_com", 3), ("root_inertia", 3), ("root_inertia_frame", 4), ("gravity", 3)):
        array(data[name], size)
    child = data["children"]
    for name in ("parents", "kinds", "mass"):
        same(len(child[name]), n, "all original child rows")
    for name in ("offsets", "axes", "com", "inertia"):
        array(child[name], n, 3)
    array(child["orientations"], n, 4)
    require(number(data["root_mass"]) > 0, "original root mass")
    if "shapes" in data:
        for item in data["shapes"]:
            require(item["group"] is None or 0 <= item["group"] <= n, "declared body source")
            require(item["kind"] in ("sphere", "capsule", "hull"), "original finite shape family")
            array(item["points"], len(item["points"]), 3)
            number(item["radius"])
    return n


def command_schema(value, n):
    fields(value, ("drive", "external"), "unchanged floating input packet")
    array(value["drive"], n, 3)
    for name, shape in (("effort", (n,)), ("force", (n + 1, 3)), ("torque", (n + 1, 3))):
        array(value["external"][name], *shape)


def held_matches(state, command):
    if "local_origin_m" in state:
        wanted = [command["drive"], command["external"]["effort"], command["external"]["force"], command["external"]["torque"]]
        close(state["control"], wanted, "full held input packet", atol=0, rtol=0)


def saved_independent(measured, audit):
    fields(audit, ("same_entry",), "saved independent same-entry audit")
    truth = audit["same_entry"]
    diag = measured["diagnostics"]
    for name in ("mass", "bias", "linear_jacobian", "angular_jacobian", "acceleration"):
        close(diag[name], truth[name], "recorded independent " + name, atol=2e-5 if name == "mass" else ATOL)
    for actual, expected in (("contact_generalized", "contact_generalized"), ("input_external", "input_external"), ("actual_drive", "drive")):
        close(measured[actual], truth[expected], "saved independent " + actual)
    if "wrench" in truth:
        close(measured["law_wrench_fp64"], truth["wrench"], "saved independent body wrench")
    observed = audit.get("accepted_mechanics", audit.get("accepted"))
    require(observed is not None, "saved complete accepted independent mechanics")
    for actual, expected in (("link_position_m", "position"), ("link_rotation", "rotation"), ("com_position_m", "com_position"),
                             ("linear_velocity_m_s", "com_velocity"), ("angular_velocity_rad_s", "angular_velocity")):
        close(measured["accepted_geometry"][actual], observed[expected], "recorded accepted independent " + actual)
    work = audit.get("work", audit.get("material_and_work"))
    require(work is not None, "saved independent two-work audit")
    for name in ("material_work_j", "application_work_j"):
        close(diag[name], work[name], "recorded work " + name, atol=1e-12, rtol=1e-10)
    close(diag["entry_kinetic_energy_j"], truth["energy"], "entry kinetic observation")
    close(diag["accepted_kinetic_energy_j"], observed["energy"], "accepted kinetic observation")


def input_identity(member, data):
    n = input_schema(data)
    wanted = {
        "f02": "F02", "f03": "F03", "f04": "F04", "f05": "F05", "f06": "F06", "f07": "F07",
        "f08": "F08", "branch": "F08_branch", "delay": "F08_branch", "f09": "F09", "stale": "F04",
        "resource": "F01", "source_fault": "F02", "huge_anchor": "F01", "filter": "F03",
    }.get(member.group, member.parameter.get("name"))
    if wanted is not None:
        same(data["name"], wanted, "declared case identity")
    for key in ("variant", "placement", "boost"):
        if key in member.parameter:
            same(data[key], member.parameter[key], "declared " + key)
    if member.group == "f08":
        same(n, member.parameter["n"], "N9/N16 explicit source")
    if member.group == "delay":
        same(data["delay"], member.parameter["delay"], "physical delay")
        close(data["caps"], [member.parameter["cap"]] * 3, "original cap", atol=0, rtol=0)
        close(data["kp"], [16, 8, 4], "original kp", atol=0, rtol=0)
        close(data["kd"], [2, 1, 0.5], "original kd", atol=0, rtol=0)
    return n


def source_input_binding(data, model, bindings):
    shapes, bound = bindings_schema(model, bindings)
    if "shapes" not in data:
        return shapes, bound
    require(len(data["shapes"]) == len(shapes), "full source declaration includes every collider")
    for index, original in enumerate(data["shapes"]):
        # Standard declared shape IDs survive reordering. F06 carries imported
        # authored IDs instead, so topology/source equality is used there.
        if data["name"] == "F06" and index == 0:
            matches = [k for k, v in bound.items() if v.get("group_index") == 0]
        else:
            ids = [1000 * index + 11 + 3 * j for j in range(len(original["points"]))]
            matches = [k for k, s in shapes.items() if s["vertex_ids"] == ids]
        require(len(matches) == 1, "independent original material ID/source mapping")
        key = matches[0]
        same(shapes[key]["kind"], original["kind"], "original finite geometry kind")
        same(shapes[key]["radius"], original["radius"], "original physical skin radius")
        same(bound[key].get("group_index"), original["group"], "physical source group")
        actual_points = bound[key]["local_points_m"] if original["group"] is not None else bound[key]["motion"]["start"]
        if data["name"] == "F06" and index == 0:
            # The authored URDF box can order its vertices differently. The
            # producer compares the original point sets; actual ID/order is
            # retained exactly in binding, shape, both frames and each witness.
            close(sorted(actual_points), sorted(original["points"]), "authored fixed-tool point set", atol=1e-12, rtol=0)
        else:
            close(actual_points, original["points"], "original complete material points", atol=1e-12, rtol=0)
    return shapes, bound


def window_payload(member, record):
    fields(record, ("input", "command", "mode", "dt_s", "rows", "retained", "stats", "model", "bindings", "laws"), "window")
    data = record["input"]
    n = input_identity(member, data)
    command_schema(record["command"], n)
    same(record["dt_s"], H, "original window h")
    steps = 2 if member.group == "f07" else 4 if member.group == "delay" else 40
    if member.group != "f07":
        same(record["steps"], steps, "declared complete window")
        same(record["failure"], None, "successful window has no failure")
    shapes, bindings = source_input_binding(data, record["model"], record["bindings"])
    if member.group == "f03":
        # CollisionModel canonicalizes shapes, but the saved registration tuple
        # retains the actual input order supplied by _geometry(reverse=...).
        declared = [(503, 101 + item["group"] if item["group"] is not None else 900 + i, 7 + i)
                    for i, item in enumerate(data["shapes"])]
        if member.parameter["reverse"]:
            declared.reverse()
        same([key_tuple(binding["key"]) for binding in record["bindings"]], declared,
             "original F03 forward/reverse registration order")
    same(len(record["rows"]), steps, "complete real forward steps")
    previous = None
    for i, row in enumerate(record["rows"]):
        fields(row, ("receipt", "state", "geometry", "summary", "audit"), "completed physical row")
        frame = dict(row, receipts={str(row["receipt"]["world"]): row["receipt"]})
        record_frame(frame, (503,), shapes=shapes, bindings=bindings)
        measured = row["receipt"]
        same(measured["entry_tick"], i, "consecutive original entry tick")
        same(measured["exit_tick"], i + 1, "consecutive original accepted tick")
        if "entry" in row:
            exact(row["entry"], measured["entry_state"], "saved sampled entry")
        if previous is not None:
            exact(measured["entry_state"], previous, "no reset between actual steps")
        held_matches(measured["entry_state"], record["command"])
        selected = [[0, 0, 0] for _ in range(n)] if i < data.get("delay", 0) else record["command"]["drive"]
        close(measured["selected_drive"], selected, "actual delayed request", atol=0, rtol=0)
        saved_independent(measured, row["audit"])
        if member.group != "f07":
            fields(row, ("independent_window", "independent_q", "independent_v"), "independent own trajectory")
            close(measured["accepted_state"]["position"], row["independent_q"], "complete independent window q", atol=WINDOW_ATOL)
            close(measured["accepted_state"]["velocity"], row["independent_v"], "complete independent window v", atol=WINDOW_ATOL)
        previous = measured["accepted_state"]
    exact(record["retained"], record["rows"][-1]["state"], "retained full state at end")
    if "retained_geometry" in record:
        exact(record["retained_geometry"], record["rows"][-1]["geometry"], "retained FK at end")


def check_snapshot(value, worlds=None):
    fields(value, ("state", "geometry", "summary", "receipts", "status", "epoch", "bank"), "transaction before snapshot")
    record_frame(value, worlds)
    require(value["bank"] in (0, 1), "actual active bank")
    require(type(value["epoch"]) is int and value["epoch"] >= 0, "actual state epoch")


def batch_payload(relative, record):
    fields(record, ("inputs", "commands", "mode", "dt_s", "steps", "restore_tick", "replay_steps", "rows", "replay", "reverse",
                   "model", "bindings", "laws", "saved", "retained", "retained_geometry", "stats", "plan_stats", "failure"), "multiworld complete archive")
    name = relative.split("/", 1)[0]
    worlds = [int(name[3:])] if name.startswith("w1-") else list(WORLDS)
    if name == "w32":
        worlds.extend(10001 + 97 * i for i in range(28))
    same(set(record["inputs"]), {str(w) for w in worlds}, "authored W identities")
    same(set(record["commands"]), {str(w) for w in worlds}, "all world control bindings")
    shapes, bindings = bindings_schema(record["model"], record["bindings"])
    for index, w in enumerate(worlds):
        data = record["inputs"][str(w)]
        n = input_schema(data)
        source_index = WORLDS.index(w) if w in WORLDS else (index - 4) % 4
        same(data["name"], ("F01", "F02", "F08", "F08")[source_index], "world declared case")
        same(n, (0, 1, 9, 16)[source_index], "world full physical dimensions")
        same(data["delay"], 1, "matrix input delay")
        command_schema(record["commands"][str(w)], n)
        subset = {"shapes": [s for key, s in shapes.items() if key[0] == w], "excluded": []}
        source_input_binding(data, subset, [v for key, v in bindings.items() if key[0] == w])
    same((record["dt_s"], record["steps"], record["restore_tick"], record["replay_steps"]), (H, 40, 19, 21), "full authored W window")
    same((len(record["rows"]), len(record["replay"])), (40, 21), "complete matrix frames")
    same(record["failure"], None, "no matrix failure")
    for index, row in enumerate(record["rows"]):
        record_frame(row, worlds, shapes=shapes, bindings=bindings)
        same(set(row["audit"]), {str(w) for w in worlds}, "all world independent audits")
        for w in worlds:
            key = str(w)
            measured = row["receipts"][key]
            same(row["state"][key]["tick"], index + 1, "matrix actual tick")
            held_matches(row["state"][key], record["commands"][key])
            if index:
                exact(measured["entry_state"], record["rows"][index - 1]["state"][key], "continuous world checkpoint")
            saved_independent(measured, row["audit"][key])
            for a, b in (("position", "q"), ("velocity", "v")):
                close(row["state"][key][a], row["audit"][key]["window"][b], "independent full world window", atol=WINDOW_ATOL)
    exact(record["saved"], record["rows"][18]["state"], "actual tick19 saved checkpoint")
    for index, row in enumerate(record["replay"]):
        record_frame(row, worlds, shapes=shapes, bindings=bindings)
        for field in ("state", "geometry", "receipts", "summary"):
            exact(row[field], record["rows"][index + 19][field], "full checkpoint replay " + field)
    exact(record["retained"], record["replay"][-1]["state"], "after replay retained state")
    exact(record["retained_geometry"], record["replay"][-1]["geometry"], "after replay retained geometry")
    require(record["plan_stats"]["contact_kernel_launches"] > 0 and record["plan_stats"]["floating_bounds_kernel_launches"] > 0, "actual contact and bounds counters")


def partial_payload(record):
    # Do not reject a real first physical failure merely because its failing
    # field violates a success gate. Preserve/type-scan every original field;
    # only fully available state blocks are required to be structurally sound.
    scan(record, partial=True)
    rows = record.get("rows", record.get("trace", []))
    require(isinstance(rows, list), "failed prefix row list")
    require(len(rows) <= 40, "original failed prefix bound")
    return {"rows_present": len(rows), "fields_present": sorted(record),
            "failure": record.get("failure", record.get("error")),
            "scope": "all raw bytes/types bound; success physics gates not asserted for a failed node"}


def physical_payload(member, relative, record, *, partial=False):
    scan(record, partial=partial)
    if partial:
        return partial_payload(record)
    group = member.group
    if group in ("f02", "f03", "f04", "f05", "f06", "f07", "f08", "branch", "f09", "delay"):
        window_payload(member, record)
    elif group == "f01":
        fields(record, ("case", "dt_s", "model", "initial", "shapes", "wall_points", "local_ball", "law", "host_patch",
                       "host_loads", "host_terms", "literal_mass", "literal_acceleration", "drive", "external", "entry", "entry_geometry",
                       "bindings", "receipt", "accepted", "geometry", "plan_stats", "batch_stats"), "F01 complete initial/actual evidence")
        same(record["case"], "F01-public", "F01 identity")
        same(record["dt_s"], H, "F01 h")
        close(record["literal_acceleration"], [0, 0, 1, 0, 0, 0], "F01 independent acceleration literal", atol=0, rtol=0)
        close(record["host_terms"]["acceleration"], record["literal_acceleration"], "saved HOST prerequisite", atol=1e-12, rtol=0)
        close(record["host_terms"]["mass_matrix"], record["literal_mass"], "saved HOST full mass", atol=1e-12, rtol=0)
        model = {"shapes": record["shapes"], "excluded": []}
        shapes, bindings = bindings_schema(model, record["bindings"])
        receipt(record["receipt"], record["accepted"], shapes=shapes, bindings=bindings)
        exact(record["receipt"]["entry_state"], record["entry"], "F01 original entry")
        exact(record["receipt"]["accepted_geometry"], record["geometry"], "F01 actual geometry")
        same((record["entry"]["tick"], record["accepted"]["tick"]), (0, 1), "F01 one physical step")
    elif group == "batch":
        batch_payload(relative, record)
    elif group == "capacity":
        fields(record, ("inputs", "commands", "warm", "limited_world", "before", "limited_status", "retry", "audit", "retained", "geometry", "stats"), "capacity evidence")
        for name in ("warm", "limited_world"):
            same(record[name], member.parameter[name], "capacity original " + name)
        check_snapshot(record["before"], WORLDS)
        same(record["before"]["bank"], record["warm"], "tested bank")
        limited = str(record["limited_world"])
        same(record["limited_status"][limited], "SITE_CAPACITY_EXCEEDED", "real capacity reject status")
        state_map(record["retained"], WORLDS)
        for w in map(str, WORLDS):
            input_schema(record["inputs"][w])
            command_schema(record["commands"][w], len(record["inputs"][w]["drive"]))
            measured = record["retry"][w]
            receipt(measured, record["retained"][w])
            saved_independent(measured, record["audit"][w])
            same(measured["exit_tick"], record["warm"] + (1 if w == limited else 2), "capacity world-specific real advance")
            if w == limited:
                exact(measured["entry_state"], record["before"]["state"][w], "rejected entire world becomes unchanged retry entry")
    elif group in ("late", "stale", "source_fault"):
        fields(record, ("before", "retry", "stats"), "failure and retry evidence")
        check_snapshot(record["before"])
        record_frame(record["retry"], record["before"]["state"])
        if "inputs" in record:
            for w, data in record["inputs"].items():
                input_schema(data)
                command_schema(record["commands"][w], len(data["drive"]))
        else:
            n = input_identity(member, record["input"])
            command_schema(record["command"], n)
        for w, measured in record["retry"]["receipts"].items():
            exact(measured["entry_state"], record["before"]["state"][w], "whole rejected attempt unchanged at retry")
        if group == "late":
            same(record["phase"], member.parameter["phase"], "real injected operation")
            same(record["warm"], member.parameter["warm"], "fault active direction")
            exact(record["retry"], record["control"], "independent control full retry result")
        elif group == "stale":
            same(record["warm"], member.parameter["warm"], "stale active direction")
            require(record["stats"]["state_epoch"] >= record["before"]["epoch"] + 2, "real restore epoch and retry")
        else:
            same(record["field"], member.parameter["field"], "wrong source dimension")
            same(len(record["injected"]), 1, "one real candidate packet alteration")
            original, changed = record["injected"][0]["original"], record["injected"][0]["changed"]
            require(original != changed, "actual changed source packet retained")
            if record["field"] == "shape":
                same(changed["shape"]["revision"], original["shape"]["revision"] + 1, "wrong revision evidence")
            elif record["field"] == "group":
                same(changed["group_index"], 0, "wrong actual group evidence")
            else:
                same(changed["shape"]["key"]["world"], 999, "foreign actual world evidence")
    elif group == "resource":
        input_identity(member, record["input"])
        state_map(record["before"], (503,))
        same(record["scratch_closed"]["closed"], True, "actual partial scratch owner closed")
        same(record["scratch_closed"]["device_owned_bytes"], 0, "partial allocation reclaimed")
        record_frame(record["retry"], (503,))
        exact(record["retry"]["receipts"]["503"]["entry_state"], record["before"]["503"], "allocation/global-cap fail did not drift")
    elif group == "close":
        input_identity(member, record["input"])
        state_map(record["saved"], (503,))
        receipt(record["original_receipt"], record["saved"]["503"])
        record_frame(record["expected"], (503,))
        record_frame(record["rebuilt"], (503,))
        exact(record["expected"], record["rebuilt"], "fresh-owner restore complete response")
        exact(record["rebuilt"]["receipts"]["503"]["entry_state"], record["saved"]["503"], "restored saved first state")
    elif group == "filter":
        same(record["kind"], member.parameter["kind"], "declared geometric filter")
        input_identity(member, record["input"])
        state_map(record["before"], (503,))
        record_frame(record["accepted"], (503,))
        measured = record["accepted"]["receipts"]["503"]
        if record["kind"] != "default":
            same(measured["sites"], [], "filtered pair has no contact")
            close(measured["law_wrench_fp64"], [[0] * 6 for _ in range(3)], "filtered full body wrench", atol=0, rtol=0)
            exact(measured["accepted_state"]["position"], record["before"]["503"]["position"], "filtered configuration unchanged")
        else:
            same(len(measured["sites"]), 1, "explicit dynamic opt-in real site")
    elif group == "huge_anchor":
        input_identity(member, record["input"])
        same(record["input"]["q"][0], 2**60, "original world anchor")
        close(record["local_points"], [[1 / 8, 0, 0]], "nonzero local material retained", atol=0, rtol=0)
        bindings_schema(record["model"], record["bindings"])
        require(record["broad_candidates"], "real broad-phase candidate before material failure")
        same(record["error"]["type"], "ValueError", "explicit material representability reject")
        exact(record["before"], record["retained"], "no silent world/local repair")
        state_map(record["retained"], (503,))
    elif group == "mixed":
        fields(record, ("old_inputs", "floating_input", "commands", "model", "bindings", "laws", "saved", "rows", "replay", "retained", "stats"), "old/new mixed publication")
        input_schema(record["floating_input"])
        same(record["floating_input"]["name"], "P03", "unbound true floating world")
        shapes, bindings = bindings_schema(record["model"], record["bindings"])
        same((len(record["rows"]), len(record["replay"])), (40, 21), "mixed complete window")
        for index, row in enumerate(record["rows"]):
            record_frame(row, WORLDS, shapes=shapes, bindings=bindings)
            same(set(row["receipts"]), {"503", "91"}, "unbound scalar/floating get no fabricated receipt")
            for w in map(str, WORLDS):
                same(row["state"][w]["tick"], index + 1, "all mixed worlds same physical step")
            for actual, expected in (("position", "q"), ("velocity", "v")):
                close(row["state"]["1207"][actual], row["independent_floating"][expected], "unbound floating independent window", atol=WINDOW_ATOL)
        exact(record["saved"], record["rows"][18]["state"], "mixed tick19 complete checkpoint")
        for index, row in enumerate(record["replay"]):
            record_frame(row, WORLDS, shapes=shapes, bindings=bindings)
            for field in ("state", "geometry", "receipts", "summary", "diagnostics"):
                exact(row[field], record["rows"][index + 19][field], "mixed replay " + field)
        exact(record["retained"], record["replay"][-1]["state"], "mixed final retained")
    else:
        raise AuditError("no saved-payload schema for " + group)
    return {"schema": group, "completion": "complete saved structure"}


def strict_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key " + key)
        result[key] = value
    return result


def strict_load(raw):
    def invalid(value):
        raise AuditError("non-JSON numeric constant " + value)
    result = json.loads(raw.decode("utf-8-sig"), object_pairs_hook=strict_object, parse_constant=invalid)
    return result


def normalized(path):
    return str(path).replace("\\", "/").casefold()


def hex_sha(value):
    require(isinstance(value, str) and re.fullmatch(r"[a-fA-F0-9]{64}", value), "explicit SHA256 required")
    return value.lower()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


REQUIRED_SOURCE = (
    STEP, TRANSACTIONS, REFERENCE, BASE_REFERENCE,
    "tests/scale/test_resident_floating_step.py",
    "tests/scale/test_resident_affine_contacts.py",
    "src/phymeo/backends/tyx_world_batch.py",
    "src/phymeo/backends/_resident_contact_plan.py",
    "src/phymeo/backends/_resident_contact_bridge.py",
    "src/phymeo/backends/_resident_collision_plan.py",
    "src/phymeo/backends/_resident_floating_contact.py",
    "src/phymeo/backends/_tyx_resident_floating_contact.py",
    "src/phymeo/backends/_resident_floating_tree.py",
    "src/phymeo/backends/_resident_floating_storage.py",
    "src/phymeo/backends/_tyx_resident_floating_kernels.py",
    "src/phymeo/backends/_tyx_floating_numeric.py",
    "src/phymeo/backends/_resident_floating_binding.py",
    "src/phymeo/backends/_tyx_resident_floating_bounds.py",
)


class Auditor:
    def __init__(self, manifest):
        self.manifest, self.files, self.total = manifest, {}, 0
        self.coverage = {mode: {} for mode in ("interpret", "cuda")}
        self.external = {}
        self.job_ids = set()
        self.report = dict(status="incomplete", observer_only=True, jobs=[], errors=[], files=self.files,
            limits=dict(compressed=MAX_COMPRESSED, decoded_payload=MAX_PAYLOAD, total_decoded=MAX_TOTAL,
                        metadata=MAX_METADATA, files=MAX_FILES, depth=MAX_DEPTH),
            saving_boundaries=[
                "20 nodes/backend have no payload; bound original pytest assertions are the only saved result.",
                "F01 HOST prerequisites belong to its device node; no standalone HOST node is declared.",
                "Late-fault/capacity/owned tests do not save every intermediate retained read or mutability attempt.",
                "Read-only flags and exact injected callback timing cannot be re-proved from JSON.",
                "F07 proves literal first geometry plus response to uploaded wrench, not independent rolling geometry.",
                "All ordinary full windows are complete; no warm-state snapshot omissions are invented here.",
                "Failed prefixes retain raw bytes/types and original status, never complete successful coverage.",
                "Saved independent values are compared, not regenerated by this observer.",
                "No import, physics, geometry, matrix solve, producer/verifier subprocess or device query.",
            ])

    def read(self, path, root, limit=MAX_METADATA):
        path, root = Path(path).resolve(strict=True), Path(root).resolve(strict=True)
        require(path.is_relative_to(root) and path.is_file(), "input escapes declared root")
        require(path.stat().st_size <= limit, "input file bound " + str(path))
        with path.open("rb") as stream:
            raw = stream.read(limit + 1)
        require(len(raw) <= limit, "growing input size bound")
        digest, name = hashlib.sha256(raw).hexdigest(), str(path)
        if name in self.files:
            same(self.files[name]["sha256"], digest, "evidence changed during audit")
        else:
            self.files[name] = dict(bytes=len(raw), sha256=digest)
        require(len(self.files) <= MAX_FILES, "file-count limit")
        return raw

    def text(self, path, root):
        return self.read(path, root).decode("utf-8-sig")

    def json(self, path, root, *, compressed=False):
        raw = self.read(path, root, MAX_COMPRESSED if compressed else MAX_METADATA)
        if compressed:
            with gzip.GzipFile(fileobj=io.BytesIO(raw)) as stream:
                raw = stream.read(MAX_PAYLOAD + 1)
            require(len(raw) <= MAX_PAYLOAD, "per-payload decoded bound")
            self.total += len(raw)
            require(self.total <= MAX_TOTAL, "cumulative decoded limit")
            self.files[str(Path(path).resolve())].update(uncompressed_bytes=len(raw), uncompressed_sha256=hashlib.sha256(raw).hexdigest())
        return strict_load(raw)

    @staticmethod
    def source(sources, path):
        matches = [v for k, v in sources.items() if k.endswith("/" + path.casefold())]
        require(len(matches) == 1 and isinstance(matches[0], str), "missing/ambiguous/non-text source " + path)
        return matches[0]

    def run_receipt(self, spec):
        fields(spec, ("id", "kind", "mode", "directory", "config", "head", "raw_exit", "source_sha256"), "original run declaration")
        require(isinstance(spec["id"], str) and spec["id"] and spec["id"] not in self.job_ids, "unique run id")
        self.job_ids.add(spec["id"])
        root, config_path = Path(spec["directory"]), Path(spec["config"])
        require(root.is_absolute() and config_path.is_absolute(), "absolute original run/config")
        root = root.resolve(strict=True)
        config = self.json(config_path, config_path.parent)
        start, final = self.json(root / "start.json", root), self.json(root / "final.json", root)
        same(start["task"], final["task"], "same ended run task")
        for key in ("owner", "exe", "args"):
            exact(start[key], final[key], "start/final " + key)
            exact(start[key], config[key], "same original command " + key)
        same(normalized(start["cwd"]), normalized(config["workdir"]), "explicit original cwd")
        same(start["head"], final["head"], "HEAD fixed across run")
        same(start["head"], spec["head"], "declared original HEAD")
        exact(start["sources"], final["sources"], "same source-before list")
        raw_exit = int(self.text(root / "RAW_EXIT.txt", root).strip())
        same(raw_exit, spec["raw_exit"], "original RAW_EXIT")
        same(raw_exit, final["raw_exit"], "final receipt raw exit")
        before, after, texts = {}, {}, {}
        for item in start["sources"]:
            key = normalized(item["path"])
            require(key not in before, "duplicate source identity")
            before[key] = hex_sha(item["sha256"])
            path = root / item["snapshot"]
            require(path.resolve().is_relative_to(root), "same-run snapshot root")
            raw = self.read(path, root, MAX_COMPRESSED)
            same(hashlib.sha256(raw).hexdigest(), before[key], "actual original snapshot bytes")
            try:
                texts[key] = raw.decode("utf-8-sig")
            except UnicodeDecodeError:
                texts[key] = None  # Binary wheel stays hash-bound, never text-decoded.
        for item in final["sources_after"]:
            key = normalized(item["path"])
            require(key not in after, "duplicate after source")
            after[key] = hex_sha(item["sha256"])
        exact(before, after, "full source before/after equality")
        expected = {normalized(k): hex_sha(v) for k, v in spec["source_sha256"].items()}
        require(expected, "explicit full source hash binding")
        exact(before, expected, "manifest complete source set")
        configured = {normalized((Path(start["cwd"]) / k).resolve()) for k in config["files"]}
        same(set(before), configured, "all config files resolved from actual original cwd")
        stdout, stderr = self.text(root / "stdout.log", root), self.text(root / "stderr.log", root)
        summary = dict(id=spec["id"], kind=spec["kind"], mode=spec["mode"], directory=str(root), head=spec["head"],
                       raw_exit=raw_exit, source_sha256=before, source_set_sha256=fingerprint(before),
                       original_command=dict(exe=config["exe"], args=config["args"], workdir=config["workdir"]),
                       stderr_bytes=len(stderr.encode()))
        return root, config["args"], texts, stdout, summary

    def pytest_run(self, spec):
        root, args, sources, stdout, summary = self.run_receipt(spec)
        require(spec["mode"] in self.coverage, "there are no standalone HOST nodes in this contract")
        members = membership()
        by_node = {m.node: m for m in members}
        selected, completed = spec["selected"], spec["completed"]
        require(selected and len(set(selected)) == len(selected) and set(selected) <= by_node.keys(), "explicit authored selection")
        same(completed, selected[:len(completed)], "successful first prefix in actual authored order")
        failed = spec.get("failed")
        if failed is None:
            same(completed, selected, "successful group completed selection")
            same(spec["raw_exit"], 0, "successful group raw exit")
        else:
            require(len(completed) < len(selected) and failed == selected[len(completed)], "exact first failed node")
            same(spec["raw_exit"], 1, "original first failure exit")
            require("FAILED " + failed in stdout, "actual first failure log identity")
        for path in {by_node[node].source for node in selected}:
            self.source(sources, path)
        if completed:
            for path in REQUIRED_SOURCE:
                self.source(sources, path)
        if any(by_node[node].group in ("f06", "invalid_graph") for node in completed):
            self.source(sources, "tests/fixtures/urdf/floating_fixed_composite.urdf")
        if any(by_node[node].group in ("mixed", "old_body") for node in completed):
            self.source(sources, "tests/scale/test_resident_affine_contact_transactions.py")
        require("--tyx-mode" in args and args[args.index("--tyx-mode") + 1] == spec["mode"], "same actual backend")
        require("-k" not in args, "declare explicit entrypoints/deselect, not an opaque -k selector")
        targets, exclusions = [], set()
        for i, arg in enumerate(args):
            if arg == "--deselect":
                exclusions.add(args[i + 1])
            elif arg.startswith("--deselect="):
                exclusions.add(arg.partition("=")[2])
            elif arg.startswith("tests/") and (arg.endswith(".py") or ".py::" in arg) and (i == 0 or args[i - 1] != "--deselect"):
                targets.append(arg)
        actual_selection = []
        for target in targets:
            for member in members:
                if target in (member.source, member.source + "::" + member.function, member.node) and not any(
                    value in exclusions for value in (member.node, member.source + "::" + member.function)
                ):
                    actual_selection.append(member.node)
        same(actual_selection, selected, "actual CLI vs predeclared membership")
        for node in selected:
            member = by_node[node]
            require("def " + member.function + "(" in self.source(sources, member.source), "actual authored function bound")
        lines = [line for line in stdout.splitlines() if re.search(r"\d+ (?:passed|failed)", line)]
        require(lines, "original pytest result line")
        result_line = lines[-1]
        for word, count in (("passed", len(completed)), ("failed", int(failed is not None))):
            found = re.search(r"(\d+) " + word, result_line)
            same(int(found.group(1)) if found else 0, count, "actual pytest " + word)
        require(not re.search(r"[1-9]\d* (?:error|skipped|xfailed|xpassed)", result_line), "errors/skips cannot complete declared nodes")
        existing = self.coverage[spec["mode"]]
        repeated = set(existing).intersection(completed)
        qualifies = spec.get("qualifies_completed", {})
        same(set(qualifies), repeated, "only actual completed-node repeats qualified")
        for node, link in qualifies.items():
            fields(link, ("prior_run", "reason"), "repeat qualification")
            require(link["prior_run"] in [v["run"] for v in existing[node]], "repeat references an earlier successful original run")
            require(isinstance(link["reason"], str) and link["reason"].strip(), "explicit necessary change/retest reason")
        roots = spec["payload_roots"]
        expect_roots = {node for node in completed if by_node[node].files}
        if failed and by_node[failed].files:
            expect_roots.add(failed)
        same(set(roots), expect_roots, "explicit saved-test root mapping")
        basetemp = (root / spec["basetemp"]).resolve(strict=True)
        require(basetemp.is_relative_to(root), "original basetemp under run")
        require("--basetemp" in args, "explicit original pytest basetemp")
        actual_basetemp = Path(summary["original_command"]["workdir"]) / args[args.index("--basetemp") + 1]
        same(normalized(actual_basetemp.resolve()), normalized(basetemp), "pytest basetemp resolved from actual original cwd")
        # Pinned pytest _mk_tmp uses sanitized node.name[:30] and increments
        # that prefix's ordinal in a fresh --basetemp. Bind identities BEFORE
        # inspecting payloads, including no-save nodes that consume an ordinal.
        ordinals, root_names = {}, {}
        for node in selected:
            prefix = re.sub(r"[\W]", "_", node.rsplit("::", 1)[-1])[:30]
            ordinal = ordinals.get(prefix, 0)
            root_names[node] = prefix + str(ordinal)
            ordinals[prefix] = ordinal + 1
        for node, name in roots.items():
            original_root = (root / name).resolve(strict=True)
            same(original_root, basetemp / root_names[node], "authored pytest tmp prefix/ordinal for " + node)
        expected_paths, details = set(), []
        for node in completed + ([failed] if failed else []):
            member, partial = by_node[node], node == failed
            files = spec.get("failed_payloads", []) if partial else list(member.files)
            require(len(set(files)) == len(files) and set(files) <= set(member.files), "fixed original failure-prefix filenames")
            row = dict(node=node, group=member.group, parameter=member.parameter,
                       status="first-failure-prefix" if partial else "complete-saved" if member.files else "pytest-only-no-save", files=[])
            cross = {}
            for relative in files:
                base = (root / roots[node]).resolve(strict=True)
                require(base.is_relative_to(root), "pytest original payload root inside ended run")
                path = (base / relative).resolve(strict=True)
                require(path.is_relative_to(base), "fixed saved file inside original test root")
                name = path.relative_to(root).as_posix()
                require(name not in expected_paths, "one physical payload cannot fill multiple identities")
                expected_paths.add(name)
                record = self.json(path, root, compressed=True)
                result = physical_payload(member, relative, record, partial=partial)
                if "mode" in record:
                    same(record["mode"], spec["mode"], "saved original backend")
                for stats_name in ("stats", "batch_stats"):
                    if stats_name in record:
                        same(record[stats_name]["mode"], spec["mode"], "actual batch backend")
                row["files"].append(dict(path=name, result=result))
                if member.group == "batch" and not partial:
                    for world in map(str, WORLDS):
                        if world not in record["inputs"]:
                            continue
                        token = [fingerprint({field: frame[field][world] for field in ("state", "geometry", "receipts", "summary")})
                                 for frame in record["rows"]]
                        if world in cross:
                            same(token, cross[world], "W1/W4/W32 full original-world records")
                        else:
                            cross[world] = token
            details.append(row)
        # Inventory is bounded and checks the predefined set; it cannot select it.
        found, seen, pending = set(), 0, [basetemp]
        while pending:
            for path in pending.pop().iterdir():
                seen += 1
                require(seen <= MAX_FILES, "bounded payload directory entries")
                if path.is_symlink():
                    require(path.name.endswith("current"), "unexpected evidence symlink")
                elif path.is_dir():
                    if path.name not in ("cache", "tyx", "compiled", "__pycache__") and not path.name.endswith("-cache"):
                        pending.append(path)
                elif path.name.endswith(".json.gz"):
                    found.add(path.resolve().relative_to(root).as_posix())
        same(found, expected_paths, "all original saved payloads including declared failure subset")
        for node in completed:
            existing.setdefault(node, []).append(dict(run=spec["id"], source_set_sha256=summary["source_set_sha256"]))
        summary.update(selected=selected, completed=completed, failed=failed, original_pytest_summary=result_line,
                       members=details, qualifies_completed=qualifies)
        self.report["jobs"].append(summary)


NATIVE_STATE = ("joint_q", "joint_qd", "body_q", "body_qd", "body_f", "body_parent_f")
NATIVE_AUX = (
    "joint_qdd", "joint_tau", "joint_qd_internal_in", "joint_qd_internal_out", "joint_f_internal", "joint_S_s", "body_qd_fk",
    "body_q_com", "body_solve_origin", "body_I_s", "body_v_s", "body_a_s", "body_f_s", "body_f_ext", "body_ft_s", "H",
)
NATIVE_CONTACT = (
    "rigid_contact_point_id", "rigid_contact_shape0", "rigid_contact_shape1", "rigid_contact_point0", "rigid_contact_point1",
    "rigid_contact_offset0", "rigid_contact_offset1", "rigid_contact_normal", "rigid_contact_margin0", "rigid_contact_margin1",
    "rigid_contact_surface_velocity", "rigid_contact_stiffness", "rigid_contact_damping", "rigid_contact_friction",
    "rigid_contact_tids", "rigid_contact_force",
)
NATIVE_MODEL = (
    "body_mass", "body_inv_mass", "body_com", "body_inertia", "body_inv_inertia", "body_flags", "body_world", "joint_type",
    "joint_parent", "joint_child", "joint_X_p", "joint_X_c", "joint_axis", "joint_dof_dim", "joint_q_start", "joint_qd_start",
    "joint_armature", "joint_damping", "joint_friction", "joint_target_ke", "joint_target_kd", "joint_limit_ke", "joint_limit_kd",
    "joint_limit_lower", "joint_limit_upper", "joint_enabled", "joint_world", "joint_articulation", "articulation_start", "articulation_end",
    "gravity", "shape_type", "shape_body", "shape_world", "shape_transform", "shape_scale", "shape_flags", "shape_margin", "shape_gap",
    "shape_collision_group", "shape_collision_filter_pairs", "shape_contact_pairs", "shape_material_ke", "shape_material_kd", "shape_material_kf",
    "shape_material_ka", "shape_material_mu", "shape_material_mu_rolling", "shape_material_mu_torsional",
)


def native_checks(value, *, success):
    require(isinstance(value, dict) and value, "original checks retained")
    for result in value.values():
        fields(result, ("passed", "max_abs_error", "atol", "rtol", "actual_shape", "expected_shape"), "native quantitative check")
        require(type(result["passed"]) is bool, "native check boolean")
        require(result["atol"] in (0, ATOL, 2e-5, WINDOW_ATOL), "frozen native absolute gates")
        require(result["rtol"] in (0, RTOL), "frozen native relative gate")
        if success:
            same(result["passed"], True, "native original assertion")
            same(result["actual_shape"], result["expected_shape"], "native compared dimensions")
            number(result["max_abs_error"])


def native_observation(value, case, mode):
    b, d, q = (1, 6, 7) if case == "F01" else (3, 8, 9)
    fields(value, ("raw", "raw_dtypes", "actual_device", "centers_world_m", "public_native_jacobian", "public_native_mass",
                   "world_to_native_velocity", "jacobian_world_com", "mass_world_com", "full_velocity_world_com",
                   "linear_momentum_kg_m_s", "angular_momentum_world_origin_kg_m2_s", "kinetic_energy_j"), "native complete observation")
    same(value["actual_device"], mode, "actual native device")
    fields(value["raw"], NATIVE_STATE, "all native state buffers")
    array(value["raw"]["joint_q"], q)
    array(value["raw"]["joint_qd"], d)
    array(value["raw"]["body_q"], b, 7)
    for key in ("body_qd", "body_f", "body_parent_f"):
        if value["raw"][key] is not None:
            array(value["raw"][key], b, 6)
    array(value["centers_world_m"], b, 3)
    for key in ("public_native_jacobian", "jacobian_world_com"):
        array(value[key], b, 6, d)
    for key in ("public_native_mass", "mass_world_com", "world_to_native_velocity"):
        array(value[key], d, d)
    array(value["full_velocity_world_com"], d)
    array(value["linear_momentum_kg_m_s"], 3)
    array(value["angular_momentum_world_origin_kg_m2_s"], 3)
    number(value["kinetic_energy_j"])


def native_payload(record, summary, case, mode, *, success):
    scan(record, partial=not success)
    same(record["case"], case, "native authored case")
    same(record["channel"], "native", "not matched-law external force")
    same((record["declared_steps"], record["h_s"]), (40, H), "native original window")
    same(record["semantics"]["zero_matched_force"], True, "zero matched force declaration")
    trace = record["trace"]
    require(isinstance(trace, list) and len(trace) <= 40, "native fixed prefix")
    if not success:
        # Original failures may include all 40 native steps with a failed check.
        # Preserve all raw fields without relabelling that dynamics as accepted.
        return dict(rows=len(trace), native_completed=record["native_steps_completed"], complete=record["complete_steps"],
                    first_failed_check=record["first_failed_check"], execution_error=record["execution_error"])
    same(record["status"], "passed", "native successful original status")
    same((len(trace), record["native_steps_completed"], record["complete_steps"]), (40, 40, 40), "real native 40-step completion")
    same(record["execution_error"], None, "native no execution error")
    same(record["failed_checks"], [], "native original checks all succeeded")
    same(record["first_failed_check"], None, "native no hidden first check failure")
    native_checks(record["checks"], success=True)
    meta = record["model"]
    fields(meta, (*NATIVE_MODEL, "body_count", "joint_count", "joint_dof_count", "joint_coord_count", "shape_count", "body_label", "joint_label", "shape_label", "actual_device"), "full native metadata")
    b, d, q = (1, 6, 7) if case == "F01" else (3, 8, 9)
    same((meta["body_count"], meta["joint_dof_count"], meta["joint_coord_count"], meta["shape_count"]), (b, d, q, 2), "native original topology")
    same(meta["actual_device"], mode, "native model device")
    close(meta["body_mass"], [2] if case == "F01" else [2, 2, 1], "literal native physical masses", atol=0, rtol=0)
    for name in ("kd", "kf", "ka", "mu", "mu_rolling", "mu_torsional"):
        close(meta["shape_material_" + name], [0, 0], "frozen native zero " + name, atol=0, rtol=0)
    close(meta["shape_material_ke"], [64, 64], "frozen native stiffness", atol=0, rtol=0)
    same(record["solver"]["angular_damping"], 0, "no native damping retune")
    same(record["pipeline"]["rigid_contact_max"], 8, "native original contact capacity")
    same(record["literal_first_step"]["normal_force_n"], 2, "independent original 2N literal")
    same(record["literal_first_step"]["gap_m"], -1 / 32, "independent original penetration literal")
    literal_a = [0, 0, 1, 0, 0, 0] if case == "F01" else [0, 0, 0, 0, 0, 0, -1, 2]
    close(record["literal_first_step"]["acceleration"], literal_a, "literal full acceleration", atol=0, rtol=0)
    native_observation(record["initial_state"], case, mode)
    previous = record["initial_state"]
    for index, row in enumerate(trace):
        fields(row, ("step", "entry_time_s", "exit_time_s", "entry", "exit", "original_control", "original_external_body_wrench",
                     "matched_contact_wrench_supplied", "independent_same_entry", "independent_window_entry", "independent_window_exit",
                     "native_contacts", "native_step_scratch", "native_row_reconstruction", "decoded_contact_generalized_force",
                     "net_contact_force_n", "net_contact_world_origin_moment_nm", "energy", "checks", "native_step_completed", "checks_complete"), "full native step")
        same((row["step"], row["entry_time_s"], row["exit_time_s"]), (index + 1, index * H, (index + 1) * H), "native original time")
        require(row["native_step_submitted"] is True and row["native_step_completed"] is True and row["checks_complete"] is True, "real native completion not submission only")
        same(row["matched_contact_wrench_supplied"], False, "native not matched-law injection")
        native_observation(row["entry"], case, mode)
        native_observation(row["exit"], case, mode)
        for field in ("joint_q", "joint_qd", "body_q", "body_qd"):
            exact(row["entry"]["raw"][field], previous["raw"][field], "continuous native accepted state")
        close(row["original_external_body_wrench"], [[0] * 6 for _ in range(b)], "native original zero external wrench", atol=0, rtol=0)
        close(row["original_control"]["joint_f"], [0] * d, "native original zero joint force", atol=0, rtol=0)
        fields(row["original_control"], ("joint_f", "joint_target_q", "joint_target_qd", "joint_act"), "original native control")
        contacts = row["native_contacts"]
        same((contacts["count"], contacts["capacity"], contacts["valid_capacity"]), (1, 8, True), "one real finite native contact")
        fields(contacts["rows"], NATIVE_CONTACT, "all actual active native contact channels")
        for value in contacts["rows"].values():
            require(value is None or isinstance(value, list) and len(value) == 1, "all raw active native rows retained")
        scratch = row["native_step_scratch"]
        fields(scratch["raw"], NATIVE_AUX, "same-step native raw RNEA/mechanics")
        array(scratch["raw"]["body_q_com"], b, 7)
        array(scratch["raw"]["body_solve_origin"], b, 3)
        array(scratch["raw"]["body_f_ext"], b, 6)
        array(scratch["decoded_com_world_wrench"], b, 6)
        array(scratch["decoded_public_world_com_acceleration"], d)
        require("body_solve_origin" in scratch["decoder"] and "body_q_com" in scratch["decoder"], "actual solve-origin decoder retained")
        rebuilt, truth = row["native_row_reconstruction"], row["independent_same_entry"]
        close(scratch["decoded_com_world_wrench"], rebuilt["dynamic_body_wrench_world_com"], "decoded native vs independent raw-row observation")
        close(scratch["decoded_com_world_wrench"], truth["body_wrench_world_com"], "saved independent native contact load")
        close(row["entry"]["mass_world_com"], truth["full_mass"], "saved independent native full mass", atol=2e-5)
        close(row["entry"]["jacobian_world_com"], truth["full_jacobian"], "saved independent native full Jacobian")
        close(scratch["decoded_public_world_com_acceleration"], truth["acceleration"], "saved native acceleration")
        close(row["decoded_contact_generalized_force"], truth["generalized_force"], "saved native generalized force")
        window = row["independent_window_exit"]
        close(row["exit"]["full_velocity_world_com"], window["full_velocity"], "native independent window velocity", atol=WINDOW_ATOL)
        close([x[:3] for x in row["exit"]["raw"]["body_q"]], window["origins_m"], "native independent window origin", atol=WINDOW_ATOL)
        close(row["net_contact_force_n"], [0, 0, 0], "saved native force balance")
        close(row["net_contact_world_origin_moment_nm"], [0, 0, 0], "saved native moment balance")
        require(len(rebuilt["rows"]) == 1, "complete native row reconstruction")
        for detail in rebuilt["rows"]:
            fields(detail, ("shape_ids", "normal_a_to_b", "effective_surface_points_world_m", "signed_gap_m", "mixed_material",
                            "force_on_sides_n", "wrench_on_sides_about_reference"), "native actual material/owner detail")
            same(sorted(detail["shape_ids"]), sorted(record["role_ids"]["shapes"]), "actual native pair owners")
            close(detail["signed_gap_m"], truth["gap_m"], "saved native physical surface gap")
        native_checks(row["checks"], success=True)
        previous = row["exit"]
    same(summary["complete_steps"], 40, "native per-case summary completion")
    return dict(rows=40, native_completed=40, complete=40)


BROAD_FIELDS = ("world_keys", "ticks", "shape_keys", "shape_revisions", "bounds", "registered_pairs", "candidates")


def broad_schema(value, states, shapes=None, observed=None):
    fields(value, (*BROAD_FIELDS, "state_epoch", "topology_generation"), "complete accepted broad phase")
    keys = [key_tuple(k) for k in value["shape_keys"]]
    require(len(set(keys)) == len(keys), "distinct original collision keys")
    array(value["bounds"], len(keys), 6)
    same(len(value["shape_revisions"]), len(keys), "every original material revision")
    same(len(value["world_keys"]), len(value["ticks"]), "all world ticks")
    for w, tick in zip(value["world_keys"], value["ticks"], strict=True):
        same(tick, states[str(w)]["tick"], "broad pose comes from same accepted state")
    registered = {tuple(key_tuple(k) for k in p) for p in value["registered_pairs"]}
    candidates = {tuple(key_tuple(k) for k in p) for p in value["candidates"]}
    require(candidates <= registered, "only registered candidates")
    for a, b in registered:
        require(a < b and a[0] == b[0] and a[1] != b[1] and a in keys and b in keys, "no cross-world/self-body pair")
    if shapes is not None:
        same(set(keys), set(shapes), "all original finite collision shapes")
        for index, key in enumerate(keys):
            same(value["shape_revisions"][index], shapes[key]["revision"], "original material revision")
    if observed is not None:
        fields(observed, ("material_bounds", "true_pairs"), "saved independent original-material bounds")
        expected_keys = {"CollisionKey(world=" + str(k[0]) + ", body=" + str(k[1]) + ", shape=" + str(k[2]) + ")": k for k in keys}
        same(set(observed["material_bounds"]), set(expected_keys), "full material-bound source map")
        for text_key, bounds in observed["material_bounds"].items():
            array(bounds, 2, 3)
            actual = value["bounds"][keys.index(expected_keys[text_key])]
            require(all(actual[i] <= bounds[0][i] and actual[i + 3] >= bounds[1][i] for i in range(3)), "actual device envelope contains saved original material bounds")
        true_pairs = {tuple(key_tuple(k) for k in p) for p in observed["true_pairs"]}
        require(true_pairs and true_pairs <= candidates, "no omitted saved true candidate")


def wheel_row(row, section, shapes, bindings, *, expected=None):
    worlds = section["world_order"]
    fields(row, ("input", "entry", "entry_geometry", "entry_energy", "entry_receipts", "entry_status", "entry_stats", "entry_broad",
                "acceptance", "state", "geometry", "diagnostics", "energy", "receipts", "status", "broad", "batch_stats", "plan_stats",
                "checks_complete", "audit", "bounds_audit"), "wheel complete physical row")
    same(row["checks_complete"], True, "actual checked wheel step")
    exact(row["input"], section["controls"], "original held request every actual step")
    state_map(row["entry"], worlds)
    state_map(row["state"], worlds)
    entry_trees = {w for w, state in row["entry"].items() if "generalized_applied" in state}
    same(set(row["entry_geometry"]), entry_trees, "complete entry public FK tree-world coverage")
    same(set(row["entry_energy"]), set(row["entry"]), "complete entry public energy coverage")
    record_frame(dict(row, summary=row["energy"]), worlds, shapes=shapes, bindings=bindings)
    broad_schema(row["entry_broad"], row["entry"], shapes)
    broad_schema(row["broad"], row["state"], shapes, row["bounds_audit"])
    same(row["batch_stats"]["state_epoch"], row["entry_stats"]["state_epoch"] + 1, "single shared publish epoch")
    same(row["batch_stats"]["current_buffer"], 1 - row["entry_stats"]["current_buffer"], "one shared active-bank swap")
    if expected is not None:
        exact(row["acceptance"], expected, "declared capacity mask")
    for w in map(str, worlds):
        before, after = row["entry"][w], row["state"][w]
        if not row["acceptance"][w]:
            exact(after, before, "complete rejected world retained")
            for field, prior in (("geometry", "entry_geometry"), ("energy", "entry_energy")):
                exact(row[field][w], row[prior][w], "rejected " + field)
            exact(row["receipts"].get(w), row["entry_receipts"].get(w), "rejected prior receipt retained")
            same(row["status"][w], "SITE_CAPACITY_EXCEEDED", "real reject reason")
            continue
        same(after["tick"], before["tick"] + 1, "one actual accepted substep")
        exact(after["control"], before["control"], "temporary contacts do not change held command")
        exact(after["history"], before["history"], "frozen zero-delay histories")
        if "local_origin_m" in before:
            held_matches(before, section["controls"][w])
            saved_independent(row["receipts"][w], row["audit"][w])
            exact(row["diagnostics"][w], row["receipts"][w]["diagnostics"]["floating_diagnostic"], "same recorded mechanics attempt")
        independent = row["independent_window"][w]
        close(after["position"], independent["q"], "installed independent own trajectory q", atol=WINDOW_ATOL)
        close(after["velocity"], independent["v"], "installed independent own trajectory v", atol=WINDOW_ATOL)


def wheel_window(section, label, *, success):
    if not success:
        scan(section, partial=True)
        return dict(forward_rows=len(section.get("trace", [])), replay_rows=len(section.get("replay", [])),
                    capacity_records=len(section.get("capacity", [])), status=section.get("status"), failure=section.get("failure"))
    fields(section, ("inputs", "controls", "world_order", "h_s", "steps", "trace", "replay", "capacity", "alignments", "checks", "first_literal_anchors",
                     "collision_model", "bindings", "source_shapes", "laws", "initial_state", "initial_geometry", "initial_broad", "initial_stats",
                     "retained_initial_owned", "final", "independent_final_state", "closed_plan_stats", "closed_batch_stats", "read_after_close", "status"), "installed complete window")
    same(section["status"], "passed", "installed original status")
    worlds = [503] if label == "f03" else list(WORLDS)
    same(section["world_order"], worlds, "fixed installed world order")
    same(set(section["inputs"]), {str(w) for w in worlds}, "all installed physical input declarations")
    same((section["h_s"], section["steps"], len(section["trace"])), (H, 40, 40), "installed original window")
    shapes, bindings = bindings_schema(section["collision_model"], section["bindings"])
    state_map(section["initial_state"], worlds)
    initial_trees = {w for w, state in section["initial_state"].items() if "generalized_applied" in state}
    same(set(section["initial_geometry"]), initial_trees, "complete initial public tree geometry")
    record_frame(dict(section["final"], summary=section["final"]["energy"]), worlds, shapes=shapes, bindings=bindings)
    broad_schema(section["initial_broad"], section["initial_state"], shapes)
    same(section["inputs"]["503"]["name"], "F03" if label == "f03" else "F01", "installed main frozen input")
    if label == "w4":
        same(section["inputs"]["91"]["name"], "F02-motor", "installed real internal motor input")
        same(section["inputs"]["1207"]["name"], "ordinary-P", "mixed ordinary tree")
        same(section["inputs"]["6"]["name"], "FreeSphere", "mixed old scalar")
    previous = section["initial_state"]
    for index, row in enumerate(section["trace"]):
        exact(row["entry"], previous, "installed no-reset continuous original window")
        wheel_row(row, section, shapes, bindings, expected={str(w): True for w in worlds})
        for state in row["state"].values():
            same(state["tick"], index + 1, "installed actual step tick")
        previous = row["state"]
    kept = section["retained_initial_owned"]
    for field in ("receipts", "state", "geometry", "diagnostics"):
        exact(kept[field], section["trace"][0][field], "old owned readback unchanged")
    if label == "w4":
        same((len(section["replay"]), len(section["capacity"])), (21, 2), "actual replay and both capacity directions")
        exact(section["restore_checkpoint"], section["trace"][18]["state"], "actual complete tick19 checkpoint")
        exact(section["after_restore"]["state"], section["restore_checkpoint"], "actual restore all fields")
        same(section["after_restore"]["stats"]["state_epoch"], section["before_restore"]["batch_stats"]["state_epoch"] + 1, "same-tick restore advances epoch")
        previous = section["restore_checkpoint"]
        for index, row in enumerate(section["replay"]):
            exact(row["entry"], previous, "replay continuous accepted entry")
            wheel_row(row, section, shapes, bindings, expected={str(w): True for w in worlds})
            original = section["trace"][index + 19]
            for field in ("input", "entry", "entry_geometry", "entry_energy", "acceptance", "state", "geometry", "diagnostics", "energy", "receipts", "status"):
                exact(row[field], original[field], "installed full replay " + field)
            for field in ("entry_broad", "broad"):
                for part in BROAD_FIELDS:
                    exact(row[field][part], original[field][part], "installed replay original material " + field + "." + part)
                require(row[field]["state_epoch"] > original[field]["state_epoch"], "replay source epoch explicitly distinct")
                same(row[field]["topology_generation"], original[field]["topology_generation"], "same source topology")
            same(row["complete_replay_equal"], True, "original complete replay check")
            previous = row["state"]
        require(len(section["alignments"]) <= 2, "only declared true alignment steps")
        pending = list(section["alignments"])
        for bank, transaction in enumerate(section["capacity"]):
            same(transaction["bank"], bank, "actual two-bank order")
            limited = str((503, 91)[bank])
            same(str(transaction["limited_world"]), limited, "original limited world")
            same((len(transaction["limited_rows"]), len(transaction["retry_rows"])), (1, 1), "one reject and one retry each")
            rejected_row, retry = transaction["limited_rows"][0], transaction["retry_rows"][0]
            if rejected_row["entry"] != previous:
                require(pending, "missing true bank alignment")
                alignment = pending.pop(0)
                exact(alignment["entry"], previous, "real alignment continuous entry")
                wheel_row(alignment, section, shapes, bindings, expected={str(w): True for w in worlds})
                previous = alignment["state"]
            exact(rejected_row["entry"], previous, "capacity actual prior accepted state")
            same(rejected_row["entry_stats"]["current_buffer"], bank, "actual capacity bank")
            wheel_row(rejected_row, section, shapes, bindings, expected={str(w): str(w) != limited for w in worlds})
            exact(transaction["entry_full_plan_receipts"], transaction["full_plan_receipts_after_rejection"], "full plan publication unchanged by another plan's capacity failure")
            exact(retry["entry"], rejected_row["state"], "retry after actual mixed retain/advance")
            wheel_row(retry, section, shapes, bindings, expected={str(w): True for w in worlds})
            exact(retry["receipts"][limited]["entry_state"], rejected_row["entry"][limited], "rejected full world unchanged at retry")
            same(transaction["closed_plan_stats"]["device_owned_bytes"], 0, "limited scratch reclaimed")
            same(transaction["checks_complete"], True, "original transaction checks complete")
            previous = retry["state"]
        same(pending, [], "all actual alignment rows consumed")
        exact(section["final"]["state"], previous, "installed final same physical trajectory")
        same(section["foreign_closed_stats"]["device_owned_bytes"], 0, "foreign owner resources closed")
    else:
        same((section["replay"], section["capacity"], section["alignments"]), ([], [], []), "F03 is not a second replay/capacity claim")
        exact(section["final"]["state"], section["trace"][-1]["state"], "F03 complete final state")
    for field in ("closed_plan_stats", "closed_batch_stats"):
        same(section[field]["device_owned_bytes"], 0, "actual resource release " + field)
        same(section[field]["closed"], True, "actual closed owner " + field)
    same(section["closed_plan_stats"]["contact_device_owned_bytes"], 0, "contact scratch closed")
    same(section["closed_batch_stats"]["collision_plan_live_count"], 0, "no surviving plans")
    same(section["read_after_close"]["type"], "RuntimeError", "public step after real close")
    require(section["checks"] and all(v is True for v in section["checks"].values()), "original installed checks")
    return dict(forward=40, replay=21 if label == "w4" else 0, capacity_banks=2 if label == "w4" else 0,
                actual_alignment_steps=len(section["alignments"]))


PIN_SOURCES = (
    "_src/sim/builder.py", "_src/sim/model.py", "_src/sim/state.py", "_src/sim/articulation.py", "_src/sim/collide.py", "_src/sim/contacts.py",
    "_src/geometry/broad_phase_common.py", "_src/geometry/collision_primitive.py", "_src/geometry/contact_data.py", "_src/geometry/narrow_phase.py",
    "_src/solvers/semi_implicit/kernels_contact.py", "_src/solvers/featherstone/solver_featherstone.py", "_src/solvers/featherstone/kernels.py",
)


def producer_sources(audit, summary, root, spec):
    fields(summary, ("sources_before", "sources_after", "versions", "module_paths", "forbidden_loaded_modules"), "native actual pin/source provenance")
    same((summary["versions"]["newton"], summary["versions"]["warp"]), ("1.7.0", "1.17.0"), "actual pinned distributions")
    same(summary["forbidden_loaded_modules"], [], "native never imported product/TYX")
    same(set(summary["sources_before"]), set(PIN_SOURCES), "all frozen native source files")
    same(set(summary["sources_after"]), set(PIN_SOURCES), "all native final sources")
    declared = spec["native_source_sha256"]
    same(set(declared), set(PIN_SOURCES), "manifest full native pin snapshot")
    for name, row in summary["sources_before"].items():
        same(hex_sha(row["sha256"]), hex_sha(declared[name]), "predeclared native source hash")
        same(hex_sha(row["sha256"]), hex_sha(summary["sources_after"][name]), "native source unchanged")
        path = Path(row["snapshot"])
        require(path.is_absolute() and path.resolve().is_relative_to(root), "original native source snapshot")
        raw = audit.read(path, root)
        same(len(raw), row["size_bytes"], "native source bytes")
        same(hashlib.sha256(raw).hexdigest(), row["sha256"], "native snapshot actual bytes")


def installed_provenance(audit, record, summary, args):
    fields(record, ("wheel_before", "wheel_after", "modules_before", "modules_after", "versions", "expected_package_root", "forbidden_checkout"), "installed provenance")
    exact(record["wheel_before"], record["wheel_after"], "complete original wheel before/after")
    exact(record["wheel_before"], summary["wheel"], "original installed summary binding")
    archive = record["wheel_before"]
    fields(archive, ("path", "bytes", "sha256", "direct_url", "installed_from", "version", "installer_recorded_sha256",
                     "installed_product_files_verified", "members", "metadata_members"), "full installed archive identity")
    for flag in ("--wheel-site", "--expected-package-root", "--expected-wheel", "--forbidden-checkout"):
        require(flag in args, "actual installed CLI " + flag)
    site = Path(args[args.index("--wheel-site") + 1]).resolve(strict=True)
    package = Path(args[args.index("--expected-package-root") + 1]).resolve(strict=True)
    wheel = Path(args[args.index("--expected-wheel") + 1]).resolve(strict=True)
    forbidden = Path(args[args.index("--forbidden-checkout") + 1]).resolve(strict=True)
    same(package, site / "phymeo", "actual expected installed package")
    same(normalized(record["expected_package_root"]), normalized(package), "recorded package root")
    same(normalized(record["forbidden_checkout"]), normalized(forbidden), "original excluded checkout")
    same(normalized(archive["path"]), normalized(wheel), "actual archive CLI binding")
    require(not site.is_relative_to(forbidden), "installed site not checkout")
    raw = audit.read(wheel, wheel.parent, MAX_COMPRESSED)
    same(len(raw), archive["bytes"], "wheel compressed bytes")
    same(hashlib.sha256(raw).hexdigest(), hex_sha(archive["sha256"]), "actual wheel SHA")
    same(archive["installed_from"], archive["direct_url"]["url"], "saved local installer URL")
    # Pure URL parsing only, never an HTTP request or download.
    from urllib.parse import unquote, urlsplit
    uri = urlsplit(archive["direct_url"]["url"])
    require(uri.scheme == "file" and uri.netloc in ("", "localhost"), "only actual local wheel URL")
    local_name = unquote(uri.path)
    if re.match(r"^/[A-Za-z]:", local_name):
        local_name = local_name[1:]
    same(Path(local_name).resolve(strict=True), wheel, "direct URL exact archive")
    installer_hash = archive["installer_recorded_sha256"]
    if installer_hash is not None:
        same(hex_sha(installer_hash), hex_sha(archive["sha256"]), "optional installer digest")
    saved = {v["name"]: v for v in archive["members"]}
    require(len(saved) == len(archive["members"]), "unique installed product membership")
    same(len(saved), archive["installed_product_files_verified"], "all installed members counted")
    with zipfile.ZipFile(io.BytesIO(raw)) as zipped:
        names = zipped.namelist()
        require(len(names) <= MAX_FILES and len(set(names)) == len(names), "bounded unique wheel members")
        product = {name for name in names if name.startswith("phymeo/") and not name.endswith("/")}
        same(product, set(saved), "full saved installed/wheel member identity")
        actual, pending, count = set(), [package], 0
        while pending:
            for path in pending.pop().iterdir():
                count += 1
                require(count <= MAX_FILES and not path.is_symlink(), "bounded non-aliased installed files")
                if path.is_dir():
                    pending.append(path)
                elif path.is_file():
                    actual.add(path.relative_to(site).as_posix())
        same(actual, product, "complete installed directory vs actual wheel")
        all_rows = [*archive["members"], *archive["metadata_members"]]
        expected_metadata = {name for name in names if ".dist-info/" in name and name.rsplit("/", 1)[-1] in ("METADATA", "WHEEL", "top_level.txt")}
        same({v["name"] for v in archive["metadata_members"]}, expected_metadata, "all original wheel metadata")
        for row in all_rows:
            name = row["name"]
            require(not Path(name).is_absolute() and ".." not in Path(name).parts, "safe declared archive member")
            info = zipped.getinfo(name)
            require(info.file_size <= MAX_METADATA, "bounded wheel member decoding")
            with zipped.open(info) as stream:
                content = stream.read(MAX_METADATA + 1)
            require(len(content) <= MAX_METADATA, "growing wheel member bound")
            audit.total += len(content)
            require(audit.total <= MAX_TOTAL, "total decoded archive/evidence budget")
            actual_bytes = audit.read(site / name, site)
            same(actual_bytes, content, "actual installed member bytes")
            same(len(content), row["bytes"], "saved installed member length")
            same(hashlib.sha256(content).hexdigest(), hex_sha(row["sha256"]), "saved installed member hash")
    for phase in ("modules_before", "modules_after"):
        rows, seen = record[phase], set()
        require(rows, "loaded public modules retained")
        for row in rows:
            name = row["module"]
            require(name not in seen and (name == "phymeo" or name.startswith("phymeo.")), "unique real public module")
            seen.add(name)
            path = Path(row["path"]).resolve(strict=True)
            require(path.is_relative_to(package) and not path.is_relative_to(forbidden), "actual loaded module belongs to wheel")
            same(Path(row["origin"]).resolve(strict=True), path, "loaded spec origin")
            for entry in row["package_paths"]:
                require(Path(entry).resolve(strict=True).is_relative_to(package), "loaded package path inside wheel")
            member = path.relative_to(site).as_posix()
            same(member, row["wheel_member"], "loaded original wheel member")
            same(row["bytes"], saved[member]["bytes"], "loaded member bytes")
            same(row["sha256"], saved[member]["sha256"], "loaded member hash")
    same(len(record["modules_after"]), summary["verified_module_count"], "all actually loaded modules counted")


def gzip_inventory(root):
    result = set()
    for index, path in enumerate(root.iterdir(), start=1):
        require(index <= MAX_FILES and not path.is_symlink(), "bounded original external inventory")
        if path.is_file() and path.name.endswith(".json.gz"):
            result.add(path.name)
    return result

def external_run(audit, spec):
    root, args, sources, _stdout, job = audit.run_receipt(spec)
    kind, mode = spec["kind"], spec["mode"]
    require(kind in ("native", "wheel"), "external runner type")
    source = NATIVE if kind == "native" else WHEEL
    require(any(key.endswith("/" + source.rsplit("/", 1)[-1]) for key in sources), "actual original/copied external driver snapshot")
    option = "--device" if kind == "native" else "--mode"
    require(option in args and "--output" in args, "actual external CLI")
    requested = args[args.index(option) + 1]
    if kind == "native":
        same("cuda:0" if requested == "cuda" else requested, mode, "native actual device argument")
        require(mode in ("cpu", "cuda:0"), "native fixed device identities")
    else:
        same(requested, mode, "wheel actual backend argument")
        require(mode in ("interpret", "cuda") and "-I" in args and "-B" in args, "actual isolated installed execution")
    output = Path(args[args.index("--output") + 1])
    require(output.is_absolute(), "actual external absolute output root")
    output = output.resolve(strict=True)
    declared = Path(spec["payload_directory"])
    declared = declared if declared.is_absolute() else root / declared
    same(declared.resolve(strict=True), output, "same-run original outside-checkout output")
    summary = audit.json(output / "summary.json", output)
    complete = spec["raw_exit"] == 0
    same(summary["status"], "passed" if complete else "failed", "producer status separate from audit")
    scan(summary, partial=not complete)
    summaries = []
    if kind == "native":
        same(summary["device"], mode, "native recorded device")
        same((summary["declared_cases"], summary["steps_per_case"], summary["h_s"]), (["F01", "F03"], 40, H), "native full declared identity")
        original = summary["cases"]
        require([v["case"] for v in original] == ["F01", "F03"] or not complete, "native both fixed cases attempted")
        if complete:
            producer_sources(audit, summary, output, spec)
            native_checks(summary["checks"], success=True)
        files = ["F01-native.json.gz", "F03-native.json.gz"] if complete else spec.get("failed_payloads", [])
        require(len(set(files)) == len(files) and set(files) <= {"F01-native.json.gz", "F03-native.json.gz"}, "original native prefix membership")
        for filename in files:
            case = filename.split("-", 1)[0]
            record = audit.json(output / filename, output, compressed=True)
            matches = [v for v in original if v["case"] == case]
            require(len(matches) == 1, "native case summary identity")
            case_summary = matches[0]
            saved = audit.files[str((output / filename).resolve())]
            if "output" in case_summary:
                same(normalized(case_summary["output"]), normalized(output / filename), "native original payload location")
                same(case_summary["compressed_sha256"], saved["sha256"], "native same-run payload SHA")
                same(case_summary["compressed_bytes"], saved["bytes"], "native same-run compressed length")
                same(case_summary["uncompressed_bytes"], saved["uncompressed_bytes"], "native same-run raw length")
            if complete:
                same(record["source_binding"]["device"], mode, "native per-case actual source device")
                exact(record["source_binding"]["newton_sources"], {k: v["sha256"] for k, v in summary["sources_before"].items()}, "native same source set")
                same(record["source_binding"]["script_sha256"], summary["script"]["sha256"], "native same driver")
            summaries.append(dict(case=case, file=filename, **native_payload(record, case_summary, case, mode, success=complete)))
        script_hash = summary["script"]["sha256"]
        same(gzip_inventory(output), set(files), "complete native gzip inventory")
    else:
        same(summary["mode"], mode, "wheel recorded mode")
        same(summary["raw_exit"], spec["raw_exit"], "wheel original exit")
        filename = "resident-floating-contact-wheel.trace.json.gz"
        files = [filename] if complete else spec.get("failed_payloads", [])
        require(len(set(files)) == len(files) and set(files) <= {filename}, "wheel original prefix membership")
        if files:
            record = audit.json(output / filename, output, compressed=True)
            scan(record, partial=not complete)
            saved = audit.files[str((output / filename).resolve())]
            for original_field, saved_field in (("sha256", "sha256"), ("compressed_bytes", "bytes"), ("uncompressed_bytes", "uncompressed_bytes"), ("uncompressed_sha256", "uncompressed_sha256")):
                if complete or original_field in summary["trace"]:
                    same(summary["trace"][original_field], saved[saved_field], "original installed payload " + original_field)
            same(normalized(summary["trace"]["path"]), normalized(output / filename), "original external trace path")
            if complete:
                same(record["status"], "passed", "full installed record passed")
                same(record["errors"], [], "no erased installed error")
                same(summary["errors"], [], "same installed summary errors")
                same(record["mode"], mode, "actual installed mode")
                installed_provenance(audit, record, summary, args)
                same(summary["trace"]["complete"], True, "complete written gzip")
                for name, value in (("h_s", H), ("forward_steps", 40), ("replay_from", 19), ("replay_steps", 21), ("f03_steps", 40),
                                    ("state_atol", ATOL), ("state_rtol", RTOL), ("window_atol", WINDOW_ATOL), ("mass_atol", 2e-5),
                                    ("equation_residual", 2e-5), ("control_residual", 5e-5), ("material_record_construction_atol_m", 1e-10)):
                    same(record["gates"][name], value, "unchanged installed gate " + name)
            for label in ("w4", "f03"):
                section = record.get(label)
                if section is None:
                    require(not complete, "missing installed fixed window")
                    continue
                result = wheel_window(section, label, success=complete)
                summaries.append(dict(window=label, **result))
                if complete:
                    same(section["final"]["batch_stats"]["mode"], mode, "actual installed batch backend")
                    same(section["closed_batch_stats"]["mode"], mode, "same closed installed backend")
                    entry = summary["windows"][label]
                    same((entry["completed_forward"], entry["completed_replay"], entry["completed_capacity_banks"]),
                         (40, 21 if label == "w4" else 0, 2 if label == "w4" else 0), "same installed actual counts")
                    same(entry["real_alignment_steps"], len(section["alignments"]), "same true alignment count")
            script_hash = record["script"]["sha256"]
        else:
            require(not complete and summary["trace"]["complete"] is False, "failed persistence cannot claim a full trace")
            script_hash = None
        same(gzip_inventory(output), set(files), "complete wheel gzip inventory")
    if script_hash is not None:
        require(hex_sha(script_hash) in {hex_sha(v) for v in spec["source_sha256"].values()}, "same-run driver source hash")
    identity = kind + ":" + mode
    if complete:
        if identity in audit.external:
            qualified = spec.get("qualifies_external", {})
            fields(qualified, ("prior_run", "reason"), "necessary external retest")
            require(qualified["prior_run"] in audit.external[identity] and bool(qualified["reason"]), "explicit prior external completion and reason")
        audit.external.setdefault(identity, []).append(spec["id"])
    job.update(payload_directory=str(output), producer_complete=complete, records=summaries,
               scope="original saved values/checks only; no new physical pass", qualifies_external=spec.get("qualifies_external"))
    audit.report["jobs"].append(job)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(args.manifest.is_absolute() and args.output.is_absolute(), "explicit absolute manifest/new output directory")
    require(not args.output.exists(), "audit output directory must be new")
    require(args.manifest.stat().st_size <= MAX_METADATA, "manifest bound")
    raw = args.manifest.read_bytes()
    require(len(raw) <= MAX_METADATA, "manifest grew beyond bound")
    manifest = strict_load(raw)
    fields(manifest, ("schema", "required_modes", "required_external", "runs"), "audit manifest")
    same(manifest["schema"], "c10-contact-saved-audit-v1", "observer contract version")
    require(len(set(manifest["required_modes"])) == len(manifest["required_modes"]) and set(manifest["required_modes"]) <= {"interpret", "cuda"}, "fixed device modes; HOST count is zero")
    require(len(set(manifest["required_external"])) == len(manifest["required_external"]) and set(manifest["required_external"]) <= {"native:cpu", "native:cuda:0", "wheel:interpret", "wheel:cuda"}, "external original mode identity")
    require(manifest["required_modes"] or manifest["required_external"], "explicit nonempty audit scope")
    require(isinstance(manifest["runs"], list) and 0 < len(manifest["runs"]) <= 64, "bounded original run set")
    audit = Auditor(manifest)
    members = membership()
    audit.report.update(contract=dict(device_nodes_per_mode=76, payloads_per_mode=61, standalone_host_nodes=0,
                                     no_save_nodes_per_mode=20, authored_source="RD d7bed1556fe630f5ace96bc7c251f4cd4e056fae",
                                     members=[dict(node=m.node, group=m.group, parameter=m.parameter, files=m.files) for m in members]),
                        manifest_sha256=hashlib.sha256(raw).hexdigest())
    for spec in manifest["runs"]:
        try:
            if spec["kind"] == "pytest":
                audit.pytest_run(spec)
            elif spec["kind"] == "historical":
                root, _args, _sources, _stdout, job = audit.run_receipt(spec)
                require(isinstance(spec.get("reason"), str) and spec["reason"].strip(), "historical exclusion reason")
                files = []
                for name in spec.get("payloads", []):
                    record = audit.json(root / name, root, compressed=True)
                    files.append(dict(path=name, preserved_prefix=partial_payload(record)))
                job.update(reason=spec["reason"], coverage="none; original historical evidence", payloads=files)
                audit.report["jobs"].append(job)
            else:
                external_run(audit, spec)
        except (AuditError, OSError, ValueError, TypeError, KeyError, IndexError, EOFError, RecursionError, OverflowError, zipfile.BadZipFile) as error:
            audit.report["errors"].append(dict(id=spec.get("id"), directory=spec.get("directory"), kind=spec.get("kind"),
                                               type=type(error).__name__, message=str(error), category="observer_incomplete_or_inconsistent"))
    wanted = {m.node for m in members}
    for mode in manifest["required_modes"]:
        actual = set(audit.coverage[mode])
        if actual != wanted:
            audit.report["errors"].append(dict(mode=mode, missing=sorted(wanted - actual), unexpected=sorted(actual - wanted)))
    if set(audit.external) != set(manifest["required_external"]):
        audit.report["errors"].append(dict(external_missing=sorted(set(manifest["required_external"]) - set(audit.external)),
                                           external_unrequested=sorted(set(audit.external) - set(manifest["required_external"]))))
    audit.report["coverage"] = {mode: dict(unique_nodes=len(rows), nodes=rows) for mode, rows in audit.coverage.items()}
    # A source set identifies actual run bindings, not a fabricated same-source
    # green union. Every membership entry keeps its own original source group.
    source_groups = {}
    for job in audit.report["jobs"]:
        if job["kind"] == "pytest":
            key = job["mode"] + ":" + job["source_set_sha256"]
            source_groups.setdefault(key, []).append(dict(run=job["id"], completed=job["completed"]))
    audit.report.update(source_group_coverage=source_groups, external_coverage=audit.external, decoded_bytes=audit.total,
                        new_physics_pass_count=0, status="saved_records_consistent" if not audit.report["errors"] else "incomplete_or_inconsistent")
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / "audit.json").open("x", encoding="utf-8") as stream:
        json.dump(audit.report, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(status=audit.report["status"], output=str(args.output / "audit.json"), original_runs=len(audit.report["jobs"]),
                          errors=audit.report["errors"], new_physics_pass_count=0), allow_nan=False))
    return int(bool(audit.report["errors"]))


if __name__ == "__main__":
    raise SystemExit(main())
