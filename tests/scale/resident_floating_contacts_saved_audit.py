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
