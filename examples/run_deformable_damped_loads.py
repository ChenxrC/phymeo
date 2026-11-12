"""Run a separately named D6-B damped membrane window, preserving D4 gates.

Outside-source use requires this file, deformable_damped_load_cases.py, and the
exact unchanged deformable_normal_load_cases.py / deformable_impulse_audit.py
whose hashes are bound below. No tests or product oracle is imported.

Derived from the parent D4 runner (SHA below); damping audit is copied from the
reviewed D6-A installed consumer, with only its independent recurrence routed to
our sibling case module. Complete 3 s runs are explicit CLI actions only.
"""

import argparse
import gzip
import hashlib
import importlib
import json
import math
import sys
import time
import traceback
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from pathlib import Path

import numpy as np

ref = None
impulse_ref = None


def require(value, message):
    if not value:
        raise AssertionError(message)


def encode(value):
    if isinstance(value, BaseException):
        return dict(exception_type=type(value).__name__, message=str(value))
    if isinstance(value, np.ndarray):
        return dict(dtype=value.dtype.str, shape=value.shape, bytes_hex=value.tobytes().hex())
    if isinstance(value, np.generic):
        return encode(value.item())
    if isinstance(value, bytes):
        return dict(bytes_hex=value.hex())
    if type(value) is object:
        return dict(opaque_identity=id(value))
    if is_dataclass(value):
        return {f.name: encode(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Mapping):
        return dict(mapping_entries=[[encode(k), encode(v)] for k, v in value.items()])
    if isinstance(value, (tuple, list)):
        return [encode(v) for v in value]
    if type(value) is float and not math.isfinite(value):
        return dict(nonfinite=repr(value))
    if value is None or type(value) in (str, bool, int, float):
        return value
    raise TypeError("unrecorded D6-B evidence " + type(value).__name__)


class Journal:
    def __init__(self, path):
        self.path, self.stream = path, gzip.open(path, "xb")
        self.digest, self.bytes, self.records = hashlib.sha256(), 0, 0

    def emit(self, event, **payload):
        raw = (
            json.dumps(
                dict(ordinal=self.records, event=event, payload=encode(payload)),
                allow_nan=False,
                separators=(",", ":"),
            )
            + "\n"
        ).encode()
        self.stream.write(raw)
        self.digest.update(raw)
        self.bytes += len(raw)
        self.records += 1

    def close(self):
        self.stream.close()
        return dict(
            logical_sha256=self.digest.hexdigest(),
            logical_bytes=self.bytes,
            records=self.records,
            compressed_bytes=self.path.stat().st_size,
        )


def provenance(root, *, preflight=False):
    def checkout(path):
        return ((path / "pyproject.toml").is_file() and (path / "src/phymeo").is_dir()) or (
            path.name.lower() == "src"
            and (path / "phymeo").is_dir()
            and (path.parent / "pyproject.toml").is_file()
        )

    require(
        not any(checkout(Path(p or Path.cwd()).resolve()) for p in sys.path),
        "checkout source entry in sys.path",
    )
    if preflight:
        require(
            not any(n == "phymeo" or n.startswith("phymeo.") for n in sys.modules),
            "product imported before installed preflight",
        )
        for origin in (Path.cwd().resolve(), Path(__file__).resolve().parent, root):
            require(
                not any(checkout(p) for p in (origin, *origin.parents)),
                "consumer, cwd and install must be outside checkouts",
            )
    import phymeo

    require(Path(phymeo.__file__).resolve().is_relative_to(root), "wrong installed package")
    result = {}
    for name, module in sorted(sys.modules.items()):
        if name == "phymeo" or name.startswith("phymeo."):
            location = getattr(module, "__file__", None)
            if location is not None:
                path = Path(location).resolve()
                require(path.is_relative_to(root), "mixed checkout module " + name)
                result[name] = dict(
                    path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()
                )
            for directory in getattr(module, "__path__", ()):
                require(
                    Path(directory).resolve().is_relative_to(root), "mixed package path " + name
                )
    return result


# Original byte snapshots at parent 7bb1037755f287f9934188fcb92683be78605cc2.
DERIVED_RUNNER_SHA256 = "25079b2294e239adf92e80f81dccf92b9f006034ea417638cfba687d4b775e8d"
PHASE_AUDIT_SOURCE_SHA256 = "640cd87f35d39e23792e2dbf35096adb995efa1da6fc3f3bce4b824210c8fe6c"
EXPECTED_DEPENDENCIES = {
    "deformable_normal_load_cases": "fd27afe75005e6ffb5e4ca3336952f453423ebfc852f13905aac6add8ec2a830",
    "deformable_impulse_audit": "6ca37c4fa5c0a91a99d9d2b98484a7ae48980dd19cba4672f86b2770d757cc90",
}


def close(actual, expected, *, atol=5e-6, rtol=5e-5):
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)


def reference_provenance():
    load_reference()
    result = {}
    for module in (ref, ref.base, impulse_ref):
        path = Path(module.__file__).resolve()
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        expected = EXPECTED_DEPENDENCIES.get(module.__name__)
        if expected is not None:
            require(digest == expected, "original dependency bytes changed: " + module.__name__)
        result[module.__name__] = dict(path=str(path), sha256=digest, bytes=len(raw))
    return result


def audit_dissipation(spec, position, phase, h):
    edges, params = spec.constraints.indices, spec.dissipation
    weights = np.asarray(spec.model.inverse_mass, np.float32).astype(float)
    require(phase.schema == "distance-dissipation-step-v1", "phase schema")
    np.testing.assert_array_equal(phase.ordered_point_ids, spec.point_ids[edges])
    np.testing.assert_array_equal(phase.coefficients_ns_per_m, params.coefficients_ns_per_m)
    require(
        phase.trace.dtype == np.float32 and phase.trace.shape == (len(edges), 32),
        "complete phase rows",
    )
    for name in (
        "ordered_point_ids",
        "coefficients_ns_per_m",
        "reconstructed_velocity",
        "dissipated_velocity",
        "trace",
        "node_impulse",
        "node_velocity_impulse",
        "kinetic_change",
        "ideal_work",
        "rounding_work",
    ):
        array = getattr(phase, name)
        require(
            not array.flags.writeable and np.isfinite(array).all(), "immutable finite phase " + name
        )
    chain = phase.reconstructed_velocity.copy()
    requested = np.zeros(chain.shape, float)
    changes, ideals, rounds, energy_bounds = [], [], [], []
    gamma = (8 * 2**-24) / (1 - 8 * 2**-24)
    for e, ((i, j), coefficient) in enumerate(
        zip(edges, params.coefficients_ns_per_m, strict=True)
    ):
        row = phase.trace[e].astype(float)
        np.testing.assert_array_equal(row[6:12], np.r_[chain[i], chain[j]])
        np.testing.assert_array_equal(row[28:32], 0)
        close(
            row[18:22], [h, float(np.float32(coefficient)), weights[i], weights[j]], atol=0, rtol=0
        )
        if coefficient == 0:
            np.testing.assert_array_equal(row[:6], 0)
            np.testing.assert_array_equal(row[22:23], 0)
            np.testing.assert_array_equal(row[24:28], 0)
            require(row[23] == 1, "zero-coefficient denominator")
            np.testing.assert_array_equal(row[12:18], row[6:12])
        else:
            separation = position[i].astype(float) - position[j]
            length = np.linalg.norm(separation)
            normal = separation / length
            rate = float(normal @ (chain[i].astype(float) - chain[j]))
            denom = 1 + h * float(np.float32(coefficient)) * (weights[i] + weights[j]) * (
                normal @ normal
            )
            impulse = -h * float(np.float32(coefficient)) * rate / denom
            close(row[:3], normal, atol=2e-6, rtol=2e-5)
            close(row[3:5], [rate, impulse], atol=2e-6, rtol=2e-5)
            close(row[22:24], [length, denom], atol=2e-6, rtol=2e-5)
            close(row[27], row[:3] @ row[:3], atol=2e-6, rtol=2e-5)
            close(
                row[12:15],
                chain[i].astype(float) + weights[i] * impulse * normal,
                atol=2e-6,
                rtol=2e-5,
            )
            close(
                row[15:18],
                chain[j].astype(float) - weights[j] * impulse * normal,
                atol=2e-6,
                rtol=2e-5,
            )
            np.testing.assert_array_equal(
                phase.trace[e, 24:27], np.float32(phase.trace[e, 4] * phase.trace[e, :3])
            )
            close(row[5], row[:3] @ (row[12:15] - row[15:18]), atol=2e-6, rtol=2e-5)
        kinetic = bound = 0.0
        for node, pre, post, sign in ((i, row[6:9], row[12:15], 1), (j, row[9:12], row[15:18], -1)):
            if weights[node] == 0:
                np.testing.assert_array_equal(post, pre)
                np.testing.assert_array_equal(post, 0)
                continue
            ideal_velocity = pre + sign * weights[node] * row[4] * row[:3]
            # Three primitive products/addition are bounded before examining post.
            error_bound = gamma * (np.abs(pre) + np.abs(weights[node] * row[4] * row[:3]))
            require(
                np.all(np.abs(post - ideal_velocity) <= error_bound + np.finfo(float).eps),
                "per-component FP32 velocity rounding bound",
            )
            bound += float(
                np.sum(np.abs(ideal_velocity) * error_bound + 0.5 * error_bound**2) / weights[node]
            )
            kinetic += 0.5 * float((post @ post - pre @ pre) / weights[node])
        ideal = row[4] * row[3] + 0.5 * (weights[i] + weights[j]) * row[27] * row[4] ** 2
        require(ideal <= 0.0, "implicit axial work is nonpositive")
        # Dot-product rounding in recorded s/g2 is separately bounded from operands.
        bound += abs(row[4]) * gamma * float(np.sum(np.abs(row[:3] * (row[6:9] - row[9:12]))))
        bound += 0.5 * (weights[i] + weights[j]) * row[4] ** 2 * gamma * float(row[:3] @ row[:3])
        require(
            abs(kinetic - ideal) <= bound + 1e-14,
            "raw kinetic change within independently bounded rounding work",
        )
        close(phase.kinetic_change[e], kinetic, atol=1e-14, rtol=1e-12)
        close(phase.ideal_work[e], ideal, atol=1e-14, rtol=1e-12)
        close(phase.rounding_work[e], kinetic - ideal, atol=1e-14, rtol=1e-12)
        requested[i] += row[24:27]
        requested[j] -= row[24:27]
        chain[i], chain[j] = phase.trace[e, 12:15], phase.trace[e, 15:18]
        changes.append(kinetic)
        ideals.append(ideal)
        rounds.append(kinetic - ideal)
        energy_bounds.append(bound)
    np.testing.assert_array_equal(chain, phase.dissipated_velocity)
    close(
        chain,
        ref.ordered_velocity(spec, position, phase.reconstructed_velocity, h),
        atol=2e-6,
        rtol=2e-5,
    )
    np.testing.assert_array_equal(requested, phase.node_impulse)
    actual = np.zeros(chain.shape, float)
    free = weights > 0
    actual[free] = (chain[free].astype(float) - phase.reconstructed_velocity[free]) / weights[
        free, None
    ]
    np.testing.assert_array_equal(actual, phase.node_velocity_impulse)
    velocity_rounding = np.zeros(chain.shape, float)
    velocity_rounding[free] = actual[free] - requested[free]
    return dict(
        requested=requested,
        actual=actual,
        velocity_rounding_impulse=velocity_rounding,
        kinetic_change=changes,
        ideal_work=ideals,
        rounding_work=rounds,
        rounding_bound=energy_bounds,
        raw_linear_impulse=actual.sum(axis=0),
        fixed_reaction_impulse=-requested[~free],
        raw_angular_impulse=np.cross(position.astype(float), actual).sum(axis=0),
        requested_angular_impulse=np.cross(position.astype(float), requested).sum(axis=0),
        interpretation="signed observed energy and prior arithmetic bound; no normalization",
    )


def audit_step(case, entry, after, receipt, h, external, emit, *, short_state_diagnostic=None):
    close, root_state, reference_load = ref.close, ref.root_state, ref.reference_load
    local_scale, material_step, rigid_step = ref.local_scale, ref.material_step, ref.rigid_step
    rotation = ref.rotation
    RADIUS, WIDTH, STIFFNESS, DAMPING = ref.RADIUS, ref.WIDTH, ref.STIFFNESS, ref.DAMPING
    ATOL, RTOL = ref.SHORT_ATOL, ref.SHORT_RTOL
    w = case["world"]
    observed = receipt.worlds[w]
    row = receipt.node_receipt.worlds[w]
    phase = row.dissipation
    # Raw phase is already in step-return; save independent audit before later gates.
    damping = audit_dissipation(case["node"], after.node.position, phase, h)
    np.testing.assert_array_equal(phase.dissipated_velocity, after.node.velocity)
    np.testing.assert_array_equal(observed["damping_impulse"], phase.node_impulse)
    require(encode(observed["damping_step"]) == encode(phase), "same complete damping phase")
    emit("damping-phase-audit", tick=entry.node.tick, phase=phase, audit=damping)
    root0, root1 = root_state(entry, case), root_state(after, case)
    q0, v0 = np.asarray(root0.position), np.asarray(root0.velocity)
    q1, v1 = np.asarray(root1.position), np.asarray(root1.velocity)
    x0, vs0 = entry.node.position, entry.node.velocity
    node_force, wrench, reference = reference_load(
        case, x0.astype(float), vs0.astype(float), q0, v0
    )
    emit(
        "same-input-reference",
        tick=entry.node.tick,
        force=node_force,
        wrench=wrench,
        rows=reference,
    )
    assert observed["entry"].node.tick == entry.node.tick
    assert after.node.tick == entry.node.tick + 1 == after.rigid.tick
    np.testing.assert_array_equal(row.input_position, x0)
    np.testing.assert_array_equal(row.input_velocity, vs0)
    assert row.input_tick == entry.node.tick and row.h == h
    assert row.residual <= case["settings"].tolerance
    assert len(row.trace) == row.sweeps
    seen = set()
    accumulated = np.zeros_like(x0, dtype=float)
    device_wrench = np.zeros(6)
    for site in observed["sites"]:
        assert site["surface"] == case["surface"] and site["round"] == case["round"]
        ordinal = int(site["facet_ordinal"])
        identity = (site["surface"], site["round"], site["facet_id"])
        assert identity not in seen and 0 <= ordinal < len(case["faces"])
        seen.add(identity)
        assert site["facet_id"] == case["facets"][ordinal]
        indices = case["faces"][ordinal]
        np.testing.assert_array_equal(site["point_ids"], case["ids"][indices])
        np.testing.assert_array_equal(site["node_indices"], indices)
        beta = np.asarray(site["barycentric"], float)
        assert beta.shape == (3,) and min(beta) >= 0
        close(sum(beta), 1, atol=2e-6, rtol=0)
        target = reference[ordinal]
        length = local_scale(
            x0[indices],
            np.vstack((case["local"], case["local"])) if len(case["local"]) == 1 else case["local"],
            RADIUS,
        )
        close(site["gap"], target["gap"], atol=1e-4 * length, rtol=0)
        close(site["normal"], target["normal"], atol=2e-3, rtol=0)
        close(site["node_point"], target["node_point"], atol=2e-4 * length, rtol=0)
        close(site["rigid_point"], target["rigid_point"], atol=2e-4 * length, rtol=0)
        close(site["node_point"], beta @ x0[indices], atol=2e-4 * length, rtol=0)
        close(site["rigid_com"], q0[:3])
        assert 0 <= site["axis_parameter"] <= 1
        close(site["axis_parameter"], target["s"], atol=2e-4, rtol=0)
        relative = (
            beta @ vs0[indices]
            - v0[:3]
            - np.cross(v0[3:], np.asarray(site["rigid_point"]) - q0[:3])
        )
        close(site["relative_velocity"], relative)
        normal = np.asarray(site["normal"], float)
        rate = float(np.dot(normal, relative))
        compression = max(WIDTH - site["gap"], 0)
        magnitude = STIFFNESS * compression * max(1 - DAMPING * rate, 0)
        force = magnitude * normal
        packed = np.asarray(site["law_input"])
        close(packed[:3], normal)
        close(packed[3:6], relative)
        close(packed[6:11], [WIDTH, site["gap"], STIFFNESS, 1, DAMPING])
        close(site["law_output"], np.r_[rate, magnitude, force])
        expected_node = beta[:, None] * force
        close(site["node_forces"], expected_node)
        close(site["rigid_force"], -force)
        close(site["rigid_torque"], np.cross(np.asarray(site["rigid_point"]) - q0[:3], -force))
        accumulated[indices] += np.asarray(site["node_forces"])
        device_wrench += np.r_[site["rigid_force"], site["rigid_torque"]]
        dg = abs(site["gap"] - target["gap"])
        dr = abs(rate - target["rate"])
        scalar_bound = STIFFNESS * (
            max(1 - DAMPING * target["rate"], 0) * dg
            + target["compression"] * DAMPING * dr
            + DAMPING * dg * dr
        )
        direction_bound = np.linalg.norm(target["force"]) * np.linalg.norm(
            normal - target["normal"]
        )
        assert np.linalg.norm(
            force - target["force"]
        ) <= scalar_bound + direction_bound + ATOL + RTOL * max(1, magnitude)
    # All independently active facets must have actual sites; an inactive far
    # row may be retained as a zero-force site or remain only in the query.
    assert {r["identity"] for r in reference if r["compression"] > 0} <= seen
    close(observed["node_contact_force"], accumulated)
    close(observed["rigid_wrench"], device_wrench[None, :])
    close(observed["uploaded_rigid_wrench"], device_wrench[None, :])
    close(accumulated.sum(axis=0) + device_wrench[:3], 0)
    close(
        np.cross(x0, accumulated).sum(axis=0)
        + np.cross(q0[:3], device_wrench[:3])
        + device_wrench[3:],
        0,
    )
    total = np.asarray(observed["node_total_force"], float)
    close(total, accumulated)
    close(row.force, total)
    y, velocity, lam = material_step(
        case, x0.astype(float), vs0.astype(float), row.force.astype(float), h, row.sweeps
    )
    for field, actual, expected in (
        ("node_position", after.node.position, y),
        ("elastic_reconstructed_velocity", phase.reconstructed_velocity, velocity),
        ("multipliers", row.multipliers, lam),
    ):
        try:
            close(actual, expected, atol=2e-5, rtol=2e-5)
        except AssertionError as error:
            if short_state_diagnostic is None:
                raise
            short_state_diagnostic(field, actual, expected, error)
    diagnostics = receipt.rigid_publication["diagnostics"][case["participant"].key]
    independent_mass = np.diag([case["mass"]] * 3 + [case["inertia"]] * 3)
    close(diagnostics.mass, independent_mass, atol=2e-5)
    close(diagnostics.bias, 0)
    close(diagnostics.linear_jacobian, np.eye(3, 6)[None, :, :])
    close(diagnostics.angular_jacobian, np.eye(3, 6, k=3)[None, :, :])
    total_wrench = device_wrench + external
    close(
        diagnostics.acceleration,
        np.r_[total_wrench[:3] / case["mass"], total_wrench[3:] / case["inertia"]],
    )
    assert diagnostics.equation_residual <= 2e-5
    expected_q, expected_v = rigid_step(case, q0, v0, device_wrench, h, external)
    close(q1[:3], expected_q[:3])
    # Physical orientation is sign-invariant, but the original raw q is saved.
    assert (
        min(np.linalg.norm(q1[3:] - expected_q[3:]), np.linalg.norm(q1[3:] + expected_q[3:]))
        <= ATOL + RTOL
    )
    close(v1, expected_v)
    material = np.zeros_like(x0, dtype=float)
    for frame in row.trace:
        assert frame.shape == (len(case["edges"]), 18)
        for e, (a, b) in enumerate(case["edges"]):
            delta = float(frame[e, 17])
            direction = frame[e, 14:17].astype(float)
            separation = frame[e, :3].astype(float) - frame[e, 3:6]
            if delta != 0:
                close(direction, separation / np.linalg.norm(separation), atol=2e-5, rtol=2e-5)
            impulse = direction * (delta / h)
            material[a] += impulse
            material[b] -= impulse
    close(observed["material_impulse"], material)
    weights = case["node"].model.inverse_mass.astype(float)
    fixed = weights == 0
    reaction = np.zeros_like(x0, dtype=float)
    reaction[fixed] = -h * total[fixed] - material[fixed] - phase.node_impulse[fixed]
    close(observed["fixed_reaction_impulse"], reaction)
    np.testing.assert_array_equal(after.node.position[fixed], x0[fixed])
    np.testing.assert_array_equal(after.node.velocity[fixed], 0)
    free = ~fixed
    arithmetic = impulse_ref.audit_deformable_impulse(
        input_position=x0,
        input_velocity=vs0,
        force=row.force,
        inverse_mass=case["node"].model.inverse_mass,
        gravity=case["node"].model.gravity,
        h=h,
        predicted_position=row.predicted_position,
        predicted_velocity=row.predicted_velocity,
        trace=row.trace,
        edges=case["edges"],
        final_position=after.node.position,
        final_velocity=phase.reconstructed_velocity,
        sweeps=row.sweeps,
    )
    emit("stored-arithmetic-audit", tick=entry.node.tick, audit=arithmetic)
    numerical = arithmetic["total_remainder"][free]
    impulse_actual = (after.node.velocity[free].astype(float) - vs0[free]) / weights[free, None]
    close(
        impulse_actual - numerical,
        h * total[free] + material[free] + phase.node_velocity_impulse[free],
    )
    root_impulse = case["mass"] * (v1[:3] - v0[:3])
    close(
        impulse_actual.sum(axis=0) + root_impulse,
        reaction.sum(axis=0)
        + h * external[:3]
        + numerical.sum(axis=0)
        + damping["velocity_rounding_impulse"].sum(axis=0),
    )
    # Account for application time explicitly instead of claiming that every
    # ordered central row has zero torque about the initial positions.
    p0 = vs0[free] / weights[free, None]
    p1 = after.node.velocity[free] / weights[free, None]
    angular = (np.cross(after.node.position[free], p1) - np.cross(x0[free], p0)).sum(axis=0)
    angular += np.cross(q1[:3], case["mass"] * v1[:3]) - np.cross(q0[:3], case["mass"] * v0[:3])
    angular += case["inertia"] * (v1[3:] - v0[3:])
    # Elastic movement ends at candidate x with v_reconstructed. Damping acts
    # afterwards at that fixed candidate, not at entry x or at a moved position.
    reconstructed_momentum = phase.reconstructed_velocity[free].astype(float) / weights[free, None]
    expected_angular = np.cross(x0[free], h * total[free] + material[free] + numerical).sum(axis=0)
    expected_angular += np.cross(after.node.position[free] - x0[free], reconstructed_momentum).sum(
        axis=0
    )
    damping_angular = np.cross(after.node.position.astype(float), phase.node_velocity_impulse).sum(
        axis=0
    )
    expected_angular += damping_angular
    expected_angular += np.cross(q0[:3], h * total_wrench[:3]) + h * total_wrench[3:]
    expected_angular += np.cross(q1[:3] - q0[:3], case["mass"] * v1[:3])
    close(angular, expected_angular)
    close(observed["rigid_external_entry_power"], np.dot(external, v0))
    work = potential_before = potential_after = dissipation = 0.0
    for site in observed["sites"]:
        beta = np.asarray(site["barycentric"], float)
        indices = np.asarray(site["node_indices"], int)
        force = np.asarray(site["law_output"][2:5], float)
        point = np.asarray(site["rigid_point"], float)
        node = np.asarray(site["node_point"], float)
        normal = np.asarray(site["normal"], float)
        # Advect the actual entry material addresses, not a new closest point.
        point_local = rotation(q0[3:]).T @ (point - q0[:3])
        axis_local = rotation(q0[3:]).T @ (point - RADIUS * normal - q0[:3])
        next_point = q1[:3] + rotation(q1[3:]) @ point_local
        next_axis = q1[:3] + rotation(q1[3:]) @ axis_local
        next_node = beta @ after.node.position[indices]
        work += float(force @ (next_node - node) - force @ (next_point - point))
        compression = max(WIDTH - site["gap"], 0)
        potential_before += 0.5 * STIFFNESS * compression**2
        potential_after += (
            0.5 * STIFFNESS * max(WIDTH - (np.linalg.norm(next_axis - next_node) - RADIUS), 0) ** 2
        )
        rate, magnitude = np.asarray(site["law_output"][:2], float)
        dissipation += (STIFFNESS * compression - magnitude) * rate
    close(observed["contact_work"], work)
    close(observed["layer_potential_before"], potential_before)
    close(observed["layer_potential_after"], potential_after)
    close(observed["entry_dissipation_rate"], dissipation)
    close(observed["work_balance"], work + potential_after - potential_before + h * dissipation)
    close(observed["node_external_work"], 0)
    power = sum(np.dot(r["force"], r["relative"]) for r in reference)
    storage_rate = sum(-STIFFNESS * r["compression"] * r["rate"] for r in reference)
    assert power + storage_rate <= 1e-10
    return dict(
        reference=reference,
        independent_force=node_force,
        independent_wrench=wrench,
        material_impulse=material,
        damping=damping,
        damping_angular_at_candidate=damping_angular,
        elastic_arithmetic=arithmetic,
        fixed_reaction=reaction,
        angular_change=angular,
        angular_expected=expected_angular,
        contact_power=power,
        storage_rate=storage_rate,
        contact_force_norm=float(np.linalg.norm(accumulated)),
    )


def load_reference():
    global ref, impulse_ref
    if ref is None:
        ref = importlib.import_module("deformable_damped_load_cases")
    if impulse_ref is None:
        impulse_ref = importlib.import_module("deformable_impulse_audit")
    return ref


def source_origins():
    result = {}
    for name, module in sorted(sys.modules.items()):
        if name == "phymeo" or name.startswith("phymeo."):
            path = getattr(module, "__file__", None)
            if path is not None:
                path = Path(path).resolve()
                result[name] = dict(
                    path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()
                )
    return result


def angle_error(actual, expected):
    a, b = np.asarray(actual, float), np.asarray(expected, float)
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    if a @ b < 0:
        b = -b
    return float(4 * math.atan2(np.linalg.norm(a - b), np.linalg.norm(a + b)))


def mechanical_energy(case, state):
    root = ref.root_state(state, case)
    weights = case["weights"].astype(float)
    free = weights > 0
    speed = state.node.velocity.astype(float)
    q, v = np.asarray(root.position), np.asarray(root.velocity)
    lengths = np.linalg.norm(
        state.node.position[case["edges"][:, 0]].astype(float)
        - state.node.position[case["edges"][:, 1]].astype(float),
        axis=1,
    )
    return dict(
        node_kinetic=float(0.5 * np.sum(speed[free] ** 2 / weights[free, None])),
        rigid_kinetic=float(
            0.5 * case["mass"] * (v[:3] @ v[:3]) + 0.5 * case["inertia"] * (v[3:] @ v[3:])
        ),
        material_potential=float(np.sum((lengths - case["rest"]) ** 2 / (2 * case["compliances"]))),
        rigid_com=q[:3].copy(),
    )


class Observations:
    """Fixed-size statistics only; complete samples remain in the journal."""

    def __init__(self, case):
        self.case = case
        self.minimum = int(ref.EVENT_DURATION / case["h"])
        self.windows = [
            dict(begin=a, end=b, count=0, force_sum=0.0, rigid_speed2=0.0, node_speed2=0.0)
            for a, b in ref.SUPPORT_WINDOWS
        ]
        self.events = {
            name: dict(run=0, maximum=0, start=None, qualified_start=None, qualified_end=None)
            for name in ("initial_outside", "entry", "release", "recontact")
        }
        self.prefix_open = True
        self.peak, self.minimum_gap, self.minimum_side = 0.0, math.inf, math.inf
        self.steps = 0

    def event(self, name, condition, t):
        row = self.events[name]
        if condition:
            if row["run"] == 0:
                row["start"] = t
            row["run"] += 1
            row["maximum"] = max(row["maximum"], row["run"])
            if row["run"] >= self.minimum and row["qualified_start"] is None:
                row["qualified_start"], row["qualified_end"] = row["start"], t + self.case["h"]
        else:
            row["run"], row["start"] = 0, None

    def add(self, step, entry, receipt, audit):
        case, h = self.case, self.case["h"]
        require(step == self.steps, "observation order must cover the complete window")
        t = step * h
        world = receipt.worlds[case["world"]]
        force = float(np.asarray(world["rigid_wrench"])[0, :3] @ case["axis"])
        rows = audit["reference"]
        gap = min(row["gap"] for row in rows)
        side = min(row["side"] for row in rows)
        zero = bool(
            np.all(np.asarray(world["node_contact_force"]) == 0)
            and np.all(np.asarray(world["rigid_wrench"]) == 0)
        )
        outside, positive = gap >= ref.WIDTH and zero, force > 0
        root = ref.root_state(entry, case)
        rigid_speed = float(np.asarray(root.velocity[:3]) @ case["axis"])
        node_speed = float(entry.node.velocity[0] @ case["axis"])
        self.peak = max(self.peak, force)
        self.minimum_gap, self.minimum_side = (
            min(self.minimum_gap, gap),
            min(self.minimum_side, side),
        )
        require(side > 0, "independent accepted-entry geometry crossed the membrane side")
        require(gap >= -ref.RADIUS / 2, "normal penetration exceeds the frozen radius/2 budget")
        require(force <= 8 * ref.LOAD, "normal support exceeds the frozen 8L peak")
        if step == 0:
            require(outside, "first real entry must be outside and unloaded")
        self.event("initial_outside", self.prefix_open and outside, t)
        if not outside:
            self.prefix_open = False
        self.event("entry", t < ref.SUPPORT_WINDOWS[0][0] and positive, t)
        self.event("release", t >= 1 and outside, t)
        released = self.events["release"]["qualified_end"] is not None
        self.event("recontact", t >= 9 / 8 and released and positive, t)
        for row in self.windows:
            if row["begin"] <= t < row["end"]:
                row["count"] += 1
                row["force_sum"] += force
                row["rigid_speed2"] += rigid_speed * rigid_speed
                row["node_speed2"] += node_speed * node_speed
        self.steps += 1
        return dict(
            step=step,
            time=t,
            support_force=force,
            gap=gap,
            oriented_side=side,
            rigid_normal_speed=rigid_speed,
            center_normal_speed=node_speed,
            outside=outside,
            positive=positive,
        )

    def summary(self):
        windows = []
        for row in self.windows:
            count = row["count"]
            windows.append(
                dict(
                    row,
                    mean_force=row["force_sum"] / count if count else None,
                    rigid_velocity_rms=math.sqrt(row["rigid_speed2"] / count) if count else None,
                    center_velocity_rms=math.sqrt(row["node_speed2"] / count) if count else None,
                )
            )
        return dict(
            steps=self.steps,
            peak_normal_force=self.peak,
            minimum_gap=self.minimum_gap if math.isfinite(self.minimum_gap) else None,
            minimum_oriented_side=self.minimum_side if math.isfinite(self.minimum_side) else None,
            events=self.events,
            windows=windows,
        )

    def finish(self):
        result = self.summary()
        require(self.steps == self.case["steps"], "no truncated normal-load window")
        for name, row in self.events.items():
            require(row["maximum"] >= self.minimum, "missing sustained physical event " + name)
        require(self.events["initial_outside"]["qualified_start"] == 0, "initial outside prefix")
        require(self.events["entry"]["qualified_end"] <= 0.75, "entry before first support window")
        require(
            self.events["release"]["qualified_start"] >= 1, "release follows first support window"
        )
        require(
            self.events["recontact"]["qualified_end"] <= 2.75,
            "recontact before final support window",
        )
        require(
            self.events["release"]["qualified_end"] <= self.events["recontact"]["qualified_start"],
            "release and recontact are distinct continuous events",
        )
        for row in result["windows"]:
            require(
                row["count"] == int((row["end"] - row["begin"]) / self.case["h"]),
                "full support statistics",
            )
            require(
                0.8 * ref.LOAD <= row["mean_force"] <= 1.2 * ref.LOAD,
                "mean support load outside [0.8,1.2]L",
            )
            require(
                row["rigid_velocity_rms"] <= 0.1, "rigid COM has not met normal support RMS gate"
            )
            require(
                row["center_velocity_rms"] <= 0.1, "free center has not met normal support RMS gate"
            )
        return result


def run_window(name, mode, output, *, expected_install_root=None, diagnose_short_state=False):
    """One complete immutable case; raises after writing any failed prefix."""
    output = Path(output)
    require(output.is_absolute(), "absolute fresh output required")
    output.mkdir(parents=True, exist_ok=False)
    journal = Journal(output / "damped-normal-load.jsonl.gz")
    summary = dict(
        schema="d6b-damped-normal-window-v1",
        derived_runner_sha256=DERIVED_RUNNER_SHA256,
        phase_audit_source_sha256=PHASE_AUDIT_SOURCE_SHA256,
        original_d4_results="unchanged; independent optional material model",
        status="RUNNING",
        case=name,
        mode=mode,
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        qualification="DIAGNOSTIC_ONLY" if diagnose_short_state else "STRICT",
        short_state_violations={},
        continuous_max_errors={},
    )
    progress = dict(
        steps_attempted=0,
        steps_returned=0,
        steps_checked=0,
        worldsteps=0,
        control_submissions=0,
        construct_returned=0,
        close_returned=0,
    )
    costs = dict(
        physics_seconds=0.0,
        same_input_reference_seconds=0.0,
        continued_reference_seconds=0.0,
        save_seconds=0.0,
    )
    batch, observations, root = None, None, None
    errors = []
    started = time.perf_counter()

    def emit(event, **payload):
        begin = time.perf_counter()
        journal.emit(event, **payload)
        costs["save_seconds"] += time.perf_counter() - begin

    def remember(stage, error):
        errors.append(error)
        summary["status"] = "FAIL"
        summary.setdefault("errors", []).append(
            dict(
                stage=stage,
                type=type(error).__name__,
                message=str(error),
                traceback=traceback.format_exc(),
            )
        )

    def short_state_diagnostic(field, actual, expected, error):
        difference = float(np.max(np.abs(np.asarray(actual, float) - expected)))
        counts = summary["short_state_violations"].setdefault(
            field, dict(count=0, first_returned_step=progress["steps_returned"], maximum=0.0)
        )
        counts["count"] += 1
        counts["maximum"] = max(counts["maximum"], difference)
        emit(
            "short-state-qualification-failure",
            returned_step=progress["steps_returned"],
            field=field,
            actual=actual,
            expected=expected,
            original_atol=2e-5,
            original_rtol=2e-5,
            error=error,
        )
        if not diagnose_short_state:
            raise error

    try:
        if expected_install_root is not None:
            root = Path(expected_install_root)
            require(root.is_absolute(), "absolute installed root required")
            root = root.resolve(strict=True)
            summary["initial_origins"] = provenance(root, preflight=True)
        else:
            root = None
        load_reference()
        summary["reference_provenance"] = reference_provenance()
        case = ref.make_case(name)
        summary.update(
            h=case["h"],
            planned_steps=case["steps"],
            T=ref.T,
            cases_sha256=hashlib.sha256(Path(ref.__file__).read_bytes()).hexdigest(),
        )
        emit(
            "declaration",
            case=ref.manifest(case),
            mode=mode,
            source_origins=source_origins(),
            reference_provenance=summary["reference_provenance"],
            gates=dict(
                position=ref.LONG_POSITION,
                velocity=ref.LONG_VELOCITY,
                angle=ref.LONG_ANGLE,
                support=(0.8 * ref.LOAD, 1.2 * ref.LOAD),
                RMS=0.1,
                peak=8 * ref.LOAD,
                minimum_gap=-ref.RADIUS / 2,
                event_duration=ref.EVENT_DURATION,
            ),
        )
        emit("construct-attempt", contact_capacity=4, settings=case["settings"])
        batch = ref.create(case, mode)
        progress["construct_returned"] += 1
        emit("construct-return", state=batch.snapshot(), stats=batch.stats())
        reference_state = ref.initial_reference(case)
        observations = Observations(case)
        last_external = None
        for step in range(case["steps"]):
            external = ref.load_at_step(case, step)
            if last_external is None or not np.array_equal(external, last_external):
                command = ref.controls(case, external)
                emit(
                    "control-attempt",
                    step=step,
                    time=step * case["h"],
                    command=command,
                    external=external,
                    revision=batch.revision,
                )
                batch.set_controls(command)
                progress["control_submissions"] += 1
                emit("control-return", step=step, revision=batch.revision)
                last_external = external.copy()
            entry = batch.snapshot()[case["world"]]
            revision, stats = batch.revision, batch.stats()
            inputs = {case["world"]: ref.ph.StepInputs(np.zeros_like(entry.node.position))}
            emit(
                "step-attempt",
                step=step,
                entry=entry,
                inputs=inputs,
                h=case["h"],
                expected_revision=revision,
                external=external,
                reference_entry=reference_state,
                stats=stats,
            )
            progress["steps_attempted"] += 1
            begin = time.perf_counter()
            receipt = batch.step(inputs, case["h"], expected_revision=revision)
            costs["physics_seconds"] += time.perf_counter() - begin
            progress["steps_returned"] += 1
            progress["worldsteps"] += 1
            emit("step-return", step=step, receipt=receipt)
            after, post = batch.snapshot()[case["world"]], batch.stats()
            emit("step-state", step=step, state=after, stats=post)
            require(
                receipt.source_revision == revision
                and receipt.destination_revision == batch.revision,
                "full common revision binding",
            )
            require(batch.last_step is receipt, "last accepted receipt identity")
            require(
                entry.node.tick == step and after.node.tick == step + 1 == after.rigid.tick,
                "complete real node/rigid trajectory",
            )
            require(
                post["response"]["launches"] == stats["response"]["launches"] + 2,
                "actual normal-law and transpose kernel work",
            )
            begin = time.perf_counter()
            saved_before = costs["save_seconds"]
            audit = audit_step(
                case,
                entry,
                after,
                receipt,
                case["h"],
                external,
                emit,
                short_state_diagnostic=short_state_diagnostic,
            )
            costs["same_input_reference_seconds"] += (
                time.perf_counter() - begin - (costs["save_seconds"] - saved_before)
            )
            emit("same-input-audit", step=step, audit=audit)
            begin = time.perf_counter()
            next_reference, reference_load = ref.continue_reference(case, reference_state, external)
            costs["continued_reference_seconds"] += time.perf_counter() - begin
            emit("continued-reference", step=step, state=next_reference, load=reference_load)
            x, v, q, velocity = next_reference
            actual_root = ref.root_state(after, case)
            errors_now = dict(
                node_position=float(np.max(abs(after.node.position - x))),
                node_velocity=float(np.max(abs(after.node.velocity - v))),
                rigid_position=float(np.max(abs(np.asarray(actual_root.position[:3]) - q[:3]))),
                rigid_velocity=float(np.max(abs(np.asarray(actual_root.velocity) - velocity))),
                orientation=angle_error(actual_root.position[3:], q[3:]),
            )
            for field, value in errors_now.items():
                summary["continuous_max_errors"][field] = max(
                    summary["continuous_max_errors"].get(field, 0.0), value
                )
            emit("continuous-errors", step=step, errors=errors_now)
            require(
                max(errors_now["node_position"], errors_now["rigid_position"]) <= ref.LONG_POSITION,
                "long-window position reference gate",
            )
            require(
                max(errors_now["node_velocity"], errors_now["rigid_velocity"]) <= ref.LONG_VELOCITY,
                "long-window velocity reference gate",
            )
            require(
                errors_now["orientation"] <= ref.LONG_ANGLE,
                "long-window orientation reference gate",
            )
            reference_state = next_reference
            observation = observations.add(step, entry, receipt, audit)
            root0, root1 = ref.root_state(entry, case), ref.root_state(after, case)
            emit(
                "observation",
                sample=observation,
                energy_before=mechanical_energy(case, entry),
                energy_after=mechanical_energy(case, after),
                material_damping_kinetic_change=float(
                    np.sum(receipt.node_receipt.worlds[case["world"]].dissipation.kinetic_change)
                ),
                material_damping_ideal_work=float(
                    np.sum(receipt.node_receipt.worlds[case["world"]].dissipation.ideal_work)
                ),
                material_damping_rounding_work=float(
                    np.sum(receipt.node_receipt.worlds[case["world"]].dissipation.rounding_work)
                ),
                external_translation_work=float(
                    external[:3] @ (np.asarray(root1.position[:3]) - np.asarray(root0.position[:3]))
                ),
            )
            progress["steps_checked"] += 1
        # Last accepted endpoint is not the entry of another step. Check it once.
        final = batch.snapshot()[case["world"]]
        final_root = ref.root_state(final, case)
        _, _, final_rows = ref.reference_load(
            case,
            final.node.position.astype(float),
            final.node.velocity.astype(float),
            np.asarray(final_root.position),
            np.asarray(final_root.velocity),
        )
        emit("terminal-geometry", state=final, rows=final_rows)
        require(min(row["side"] for row in final_rows) > 0, "terminal membrane side")
        require(
            min(row["gap"] for row in final_rows) >= -ref.RADIUS / 2, "terminal penetration budget"
        )
        summary["observations"] = observations.summary()
        emit("window-statistics", statistics=summary["observations"])
        observations.finish()
        require(
            progress["steps_attempted"]
            == progress["steps_returned"]
            == progress["steps_checked"]
            == case["steps"],
            "full accepted/audited physical window",
        )
        require(progress["control_submissions"] == 3, "original three-phase control schedule")
        emit("window-final", state=final, stats=batch.stats(), reference=reference_state)
        summary["status"] = "DIAGNOSTIC_COMPLETED" if diagnose_short_state else "PASS"
    except BaseException as error:
        remember("window", error)
        try:
            emit(
                "failure",
                error=error,
                last_attempt=None if batch is None else batch.last_attempt,
                state=None if batch is None else batch.snapshot(),
                progress=progress,
            )
        except BaseException as capture_error:
            remember("failure-capture", capture_error)
    finally:
        if observations is not None:
            summary["observations"] = observations.summary()
        if batch is not None:
            try:
                emit("close-attempt", stats=batch.stats())
            except BaseException as error:
                remember("close-prefix", error)
            try:
                batch.close()
                progress["close_returned"] += 1
                emit("close-return")
            except BaseException as error:
                remember("close", error)
        summary["strict_elastic_precision"] = dict(
            status="FAIL" if summary["short_state_violations"] else "NO_RECORDED_VIOLATION",
            complete=progress["steps_checked"] == summary.get("planned_steps"),
            violations=summary["short_state_violations"],
            qualification="separate original short gate; diagnostic completion is never strict PASS",
        )
        summary.update(
            progress=progress, costs=costs, elapsed_seconds=time.perf_counter() - started
        )
        try:
            summary["final_origins"] = provenance(root) if root is not None else source_origins()
        except BaseException as error:
            remember("final-origins", error)
        try:
            emit("summary", summary=summary)
        except BaseException as error:
            remember("summary-record", error)
        try:
            summary["trace"] = journal.close()
        except BaseException as error:
            remember("journal-close", error)
        try:
            (output / "summary.json").write_text(
                json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
            )
        except BaseException as error:
            remember("summary-write", error)
    if len(errors) == 1:
        raise errors[0]
    if errors:
        raise BaseExceptionGroup("D6-B run and finalization failed", errors)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        choices=(
            "d6b_n1_sphere_c4096",
            "d6b_n2_capsule_c4096",
            "d6b_n3_soft_c4096",
            "d6b_n1_fine_c4096",
        ),
        required=True,
    )
    parser.add_argument("--mode", choices=("interpret", "cuda"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-install-root", "--installed-root", type=Path)
    parser.add_argument(
        "--diagnose-short-state",
        action="store_true",
        help="Record failed original short state gates and continue; never returns qualification PASS.",
    )
    args = parser.parse_args()
    summary = run_window(
        args.case,
        args.mode,
        args.output,
        expected_install_root=args.expected_install_root,
        diagnose_short_state=args.diagnose_short_state,
    )
    print(json.dumps(summary, allow_nan=False))


if __name__ == "__main__":
    main()
