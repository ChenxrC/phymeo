"""Source-external ROB1d installed consumer; no test or private-runtime imports.

Copy this file outside every checkout. Supply the new installation/dependency
paths through the caller's environment, then run with -B and the three CLI
arguments documented in the accompanying report. Nothing here modifies sys.path.
The checks below were authored before the two public scenarios. No producer,
CPU integrator, numerical optimizer, or Newton backend is used as an oracle.
"""

import argparse
import base64
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
import gzip
import hashlib
from itertools import combinations, product
import json
from pathlib import Path
import struct
import sys
import traceback

import numpy as np

H = 1 / 1024
FAMILIES = ("fixed", "affine", "floating", "floating_affine")
STORAGES = dict(fixed="tree", affine="affine", floating="floating_tree",
                floating_affine="floating_affine")
REQUIRED_MODULES = (
    "phymeo.backends.resident_scene",
    "phymeo.backends._resident_participant_layout",
    "phymeo.backends._resident_participant_contacts",
    "phymeo.backends._resident_batch_transaction",
    "phymeo.backends._tyx_participant_transaction",
    "phymeo.backends.tyx_world_batch",
)


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def encode(value):
    """Full typed payload; process owner tokens are labelled, never serialized as state."""
    if isinstance(value, np.ndarray):
        require(not value.dtype.hasobject, "object arrays are not evidence values")
        return dict(dtype=value.dtype.str, shape=list(value.shape),
                    bytes_b64=base64.b64encode(value.tobytes(order="C")).decode())
    if isinstance(value, np.generic):
        return dict(dtype=value.dtype.str, scalar_b64=base64.b64encode(value.tobytes()).decode())
    if type(value) is object:
        return dict(process_owner_token=id(value))
    if is_dataclass(value):
        return dict(type=type(value).__module__ + "." + type(value).__name__,
                    fields={f.name: encode(getattr(value, f.name)) for f in fields(value)})
    if isinstance(value, Mapping):
        return dict(mapping_entries=[[encode(k), encode(v)] for k, v in value.items()])
    if isinstance(value, tuple):
        return dict(tuple=[encode(v) for v in value])
    if isinstance(value, list):
        return [encode(v) for v in value]
    if type(value) is float:
        return dict(float64_bits=struct.pack(">d", value).hex())
    if isinstance(value, bytes):
        return dict(bytes_b64=base64.b64encode(value).decode())
    if value is None or isinstance(value, (str, bool, int)):
        return value
    raise TypeError("unsupported complete evidence value: " + type(value).__name__)


def exact(a, b, label):
    require(encode(a) == encode(b), label)


def close(a, b, *, atol=5e-6, rtol=5e-5):
    a, b = np.asarray(a, float), np.asarray(b, float)
    require(np.isfinite(a).all() and np.isfinite(b).all(), "nonfinite comparison")
    np.testing.assert_allclose(a, b, atol=atol, rtol=rtol)


def checkout_ancestor(path):
    return next((p for p in (path, *path.parents) if (p / ".git").exists()), None)


def venv_dependency(path):
    for p in (path, *path.parents):
        if p.name.lower() == "site-packages":
            return any((q / "pyvenv.cfg").is_file() for q in p.parents)
    return False


def reject_source_entry(path, install):
    if checkout_ancestor(path) is not None:
        require(path.is_relative_to(install) or venv_dependency(path),
                "source checkout entry in sys.path: " + str(path))


def provenance(install, *, final=False):
    rows = {}
    for name, module in sorted(sys.modules.items()):
        if name == "phymeo" or name.startswith("phymeo."):
            source = getattr(module, "__file__", None)
            if source is not None:
                path = Path(source).resolve()
                require(path.is_relative_to(install), "Phymeo source leakage: " + str(path))
                rows[name] = str(path)
            for item in getattr(module, "__path__", ()):
                require(Path(item).resolve().is_relative_to(install),
                        "Phymeo package path escapes installation")
    if final:
        require(set(REQUIRED_MODULES).issubset(rows), "new installed modules were not exercised")
    return rows


def preflight(install, output):
    require(install.is_dir(), "expected installation root must exist")
    require(checkout_ancestor(Path.cwd().resolve()) is None, "cwd is inside a checkout")
    require(checkout_ancestor(Path(__file__).resolve()) is None, "copy consumer outside source")
    require(checkout_ancestor(output) is None, "output must be outside source")
    require(not any(n == "phymeo" or n.startswith("phymeo.") for n in sys.modules),
            "Phymeo was imported before the provenance guard")
    rows = []
    for entry in sys.path:
        path = Path(entry or Path.cwd()).resolve()
        reject_source_entry(path, install)
        rows.append(str(path))
    # Pure path-guard negative; no checkout is imported or inserted in sys.path.
    fake = output / "guard-only-checkout"
    (fake / "src").mkdir(parents=True)
    (fake / ".git").write_text("guard fixture; not a Git repository\n", encoding="utf-8")
    rejected = None
    try:
        reject_source_entry(fake / "src", install)
    except AssertionError as error:
        rejected = str(error)
    require(rejected is not None, "checkout/src negative did not reject")
    return dict(cwd=str(Path.cwd().resolve()), script=str(Path(__file__).resolve()),
                sys_path=rows, source_path_negative=rejected,
                dependency_rule="Existing virtualenv site-packages may be checkout descendants.")


def normalized_pd(spec, item):
    floating = spec["family"].startswith("floating")
    q0 = np.asarray(item.entry_state.position)[7 if floating else 0:]
    q1 = np.asarray(item.accepted_state.position)[7 if floating else 0:]
    v1 = np.asarray(item.accepted_state.velocity)[6 if floating else 0:]
    kp, kd, cap = (np.asarray(spec[k], np.float32).astype(float)
                   for k in ("kp", "kd", "caps"))
    held = np.asarray(spec["drive"], np.float32).astype(float)
    tau = np.asarray(item.diagnostic.drive)
    require(q0.shape == q1.shape == v1.shape == kp.shape == tau.shape, "PD dimensions")
    require(np.isfinite(tau).all() and np.all(abs(tau) <= cap), "actuator cap or finiteness")
    target, rate, ff = held.T
    first = np.clip(ff + kp * (target - q0 - H * v1) + kd * (rate - v1), -cap, cap)
    second = np.clip(ff + kp * (target - q1) + kd * (rate - v1), -cap, cap)
    residual = np.maximum(abs(tau - first), abs(tau - second))
    scale = np.maximum(1.0, np.maximum(abs(tau), np.maximum(abs(first), abs(second))))
    result = float(np.max(residual / scale, initial=0.0))
    require(result <= 5e-5, "normalized PD publication residual")
    return result


def check_receipt(built, world, before, after, receipt):
    require(receipt.entry_tick == before.tick and receipt.exit_tick == after.tick
            == before.tick + 1, "continuous common-entry publication")
    require(tuple(receipt.participants) == tuple(before.participants)
            == tuple(after.participants), "complete participant membership")
    require(not receipt.static_reactions, "no dynamic participant may become static")
    require(len(receipt.sites) == len(built["data"]["pairs"]), "all authored contact sites")
    wrench = {key: np.zeros_like(item.law_wrench_fp64)
              for key, item in receipt.participants.items()}
    site_checks = []
    for site in receipt.sites:
        require(site.pair[0].world == site.pair[1].world == world, "cross-world contact")
        close(site.first_force_n + site.second_force_n, np.zeros(3))
        point = np.asarray(site.geometry.load_point)
        n = np.asarray(site.geometry.witness.normal)
        gap = float(site.geometry.witness.gap_m)
        material = built["laws"][site.pair]
        relative = site.first_velocity_m_s - site.second_velocity_m_s
        tangent = relative - np.dot(relative, n) * n
        normal_load = material.stiffness * max(0.0, -gap)
        expected = -normal_load * n - material.friction * normal_load * tangent / max(
            material.slip_speed, float(np.linalg.norm(tangent)))
        close(site.first_force_n, expected)
        require(normal_load > 0, "loaded installed smoke lost its real contact")
        for collision, force, com in (
            (site.pair[0], site.first_force_n, site.first_reference_point_m),
            (site.pair[1], site.second_force_n, site.second_reference_point_m),
        ):
            key, body = built["shape_owners"][collision]
            item = receipt.participants[key]
            close(com, item.entry_geometry.com_position_m[body])
            moment = np.cross(point - com, force)
            wrench[key][body, :3] += force
            wrench[key][body, 3:] += moment
        site_checks.append(dict(pair=site.pair, normal_load=normal_load,
                                force_expected=expected, tangent=tangent))
    checks = {}
    for key, item in receipt.participants.items():
        exact(item.entry_state, before.participants[key], "receipt entry binding")
        exact(item.accepted_state, after.participants[key], "receipt successor binding")
        spec = built["data"]["specs"][key.participant]
        n = len(spec["model"]["mass"])
        d = n + (6 if spec["family"].startswith("floating") else 0)
        require(item.entry_mass.shape == (d, d)
                and item.entry_bias.shape == item.contact_generalized.shape == (d,),
                "full physical dimensions, including dependent joints")
        close(item.law_wrench_fp64, wrench[key])
        exact(item.uploaded_wrench_fp32, np.asarray(wrench[key], np.float32),
              "complete body wrench FP32 upload")
        jtw = np.einsum("bij,bi->j", item.entry_linear, item.uploaded_wrench_fp32[:, :3])
        jtw += np.einsum("bij,bi->j", item.entry_angular, item.uploaded_wrench_fp32[:, 3:])
        close(item.contact_generalized, jtw, atol=2e-5)
        diag = item.diagnostic
        require(diag.equation_residual <= 2e-5 and np.isfinite(diag.control_residual),
                "actual mechanical/actuator diagnostics")
        checks[key] = dict(wrench=wrench[key], full_contact_projection=jtw,
                           normalized_pd=normalized_pd(spec, item))
        if spec["family"] in ("affine", "floating_affine"):
            close(spec["S"].T @ diag.constraint_effort_full,
                  np.zeros(spec["S"].shape[1]), atol=2e-5)
        for value in (item.entry_mass, item.entry_linear, item.entry_angular,
                      item.law_wrench_fp64, item.uploaded_wrench_fp32):
            require(not value.flags.writeable, "published receipt is mutable")
    return dict(sites=site_checks, participants=checks)


def check_stages(before, after, families):
    used = {STORAGES[f] for f in families}
    for storage, old in before["storage_stage_counts"].items():
        for name in ("entry_mechanics_stages", "candidate_solve_stages",
                     "candidate_finalize_stages"):
            require(after["storage_stage_counts"][storage][name] - old[name]
                    == int(storage in used), "storage advanced other than once per step")
