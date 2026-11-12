"""D3 pair-frame regression: actual joint steps, never synthetic query rows.

Predeclared matrix: E sphere / C capsule x translation (0,0,0)/(3,-2,5),
two successive steps per case, h=1/1024. Original D3 gates are unchanged.
This regression qualifies world frames, loads and the actual stored recurrence.
The separately recorded real-arithmetic material velocity qualification may
fail; its original threshold is not a coordinate-frame correctness criterion.
The original first surface point is displaced 1/64 along its normal before
construction, producing a genuinely nonplanar facet and a nonzero pair origin
at tick zero. Real ordered material/contact evolution must change that origin.

D2/D3 currently use zero packet.origin_m. The nonzero world anchor tested here
is the actual resident floating-body FP64 anchor; the facet pair origin is
independently nonzero. This test does not fake a nonzero packet anchor.

Raw candidate/summary witnesses are pair-local. Public TriangleDistanceRow
witnesses already include pair_origins_relative_m and need ONLY packet.origin_m
to obtain a world point. No expected geometry is derived from the query frames.
"""

import importlib.util
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest
from test_deformable_contact_response import (
    ATOL,
    DAMPING,
    EDGES,
    FACES,
    FACETS,
    RADIUS,
    RTOL,
    STIFFNESS,
    WIDTH,
    H,
    audit_step,
    close,
    create,
    finite_reference,
    fixture,
    forces,
    manifest,
    reference_load,
    root_state,
    rotation,
    saved,
)

import phymeo as ph

pytestmark = pytest.mark.tyx


@lru_cache(maxsize=1)
def impulse_auditor():
    path = Path(__file__).parents[2] / "examples" / "deformable_impulse_audit.py"
    spec = importlib.util.spec_from_file_location("coordinate_impulse_audit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.audit_deformable_impulse


def translated_deformed_fixture(case, translation):
    item = fixture(case)
    shift = np.asarray(translation, np.float64)
    position = item["node"].initial_state.position.copy()
    normal_axis = 0 if case == "C" else 2
    position[0, normal_axis] += np.float32(1 / 64)
    position = (position.astype(np.float64) + shift).astype(np.float32)
    item["node"] = replace(
        item["node"],
        initial_state=ph.ParticleState(position, item["node"].initial_state.velocity),
    )
    old_tree = item["participant"].system
    q = old_tree.state.q.copy()
    q[:3] += shift
    velocity = old_tree.state.v.copy()
    velocity[3:] = (1 / 32, -1 / 64, 1 / 16)
    tree = replace(old_tree, state=ph.FloatingRobotState(q, velocity))
    participant = ph.ResidentParticipant(item["participant"].key, tree, (202,))
    item["participant"] = participant
    item["binding"] = ph.ResidentParticipantBinding(
        participant.key,
        ph.ResidentFloatingTreeBinding(
            item["round"],
            item["world"],
            0,
            item["local"],
            topology=participant.compiled.topology,
        ),
    )
    item["command"] = tree.inputs(
        np.zeros((0, 3)),
        ph.FloatingRobotInputs(np.zeros(0), item["external"][None, :3], item["external"][None, 3:]),
    )
    return item


def independent_entry(item, entry):
    """Own original material arrays and public rigid state, not packed query data."""
    root = root_state(entry, item)
    q, speed = np.asarray(root.position), np.asarray(root.velocity)
    x, v = entry.node.position.astype(float), entry.node.velocity.astype(float)
    load, wrench, rows = reference_load(item, x, v, q, speed)
    axis = item["local"] @ rotation(q[3:]).T + q[:3]
    segment = np.vstack((axis, axis)) if len(axis) == 1 else axis
    return dict(q=q, speed=speed, x=x, v=v, axis=segment, load=load, wrench=wrench, rows=rows)


def check_coordinate_contract(item, entry, after, receipt, expected):
    world = item["world"]
    original_x = expected["x"]
    groups = [g for g in receipt.query.groups if g.surface == item["surface"]]
    assert len(groups) == 1
    group = groups[0]
    # This is an explicit domain statement, not a fabricated translated packet.
    np.testing.assert_array_equal(group.packet.origin_m, np.zeros(3))
    by_facet, pair_origins = {}, {}
    for index, row in enumerate(group.packet.rows):
        assert group.round_keys[row.segment_id] == item["round"]
        assert row.status == "RESOLVED"
        assert row.facet_id not in by_facet
        ordinal = row.facet_id
        assert ordinal in (0, 1)
        target = expected["rows"][ordinal]
        tri = original_x[FACES[ordinal]]
        length = finite_reference().local_scale(tri, expected["axis"], RADIUS)
        source_first = tri[0]
        pair_origins[ordinal] = group.packet.pair_origins_relative_m[index].copy()
        close(
            group.packet.origin_m + pair_origins[ordinal], source_first, atol=2e-4 * length, rtol=0
        )
        # Independent exact finite oracle gives the expected selected witnesses.
        expected_axis = target["rigid_point"] - RADIUS * target["normal"]
        actual_axis = np.asarray(row.axis_point_relative_m) + group.packet.origin_m
        actual_triangle = np.asarray(row.triangle_point_relative_m) + group.packet.origin_m
        close(actual_axis, expected_axis, atol=2e-4 * length, rtol=0)
        close(actual_triangle, target["node_point"], atol=2e-4 * length, rtol=0)
        by_facet[ordinal] = (row, length, actual_axis, actual_triangle)
    active = {r["ordinal"] for r in expected["rows"] if r["compression"] > 0}
    assert active and active <= set(by_facet)
    sites = receipt.worlds[world]["sites"]
    assert {int(site["facet_ordinal"]) for site in sites} == active
    material_load = np.zeros_like(original_x)
    rigid_force = np.zeros(3)
    rigid_torque = np.zeros(3)
    nodal_power = rigid_power = relative_power = 0.0
    for site in sites:
        ordinal = int(site["facet_ordinal"])
        row, length, axis_point, triangle_point = by_facet[ordinal]
        target = expected["rows"][ordinal]
        indices = FACES[ordinal]
        assert site["facet_id"] == FACETS[ordinal]
        np.testing.assert_array_equal(site["node_indices"], indices)
        # These world points are checked independently, not just against each other.
        close(site["node_point"], target["node_point"], atol=2e-4 * length, rtol=0)
        close(site["rigid_point"], target["rigid_point"], atol=2e-4 * length, rtol=0)
        close(site["node_point"], triangle_point, atol=2e-4 * length, rtol=0)
        close(
            site["rigid_point"],
            axis_point + RADIUS * np.asarray(row.normal),
            atol=2e-4 * length,
            rtol=0,
        )
        close(site["rigid_com"], expected["q"][:3])
        beta = np.asarray(site["barycentric"])
        normal = np.asarray(site["normal"])
        point = np.asarray(site["rigid_point"])
        point_velocity = expected["speed"][:3] + np.cross(
            expected["speed"][3:], point - expected["q"][:3]
        )
        relative = beta @ expected["v"][indices] - point_velocity
        close(site["relative_velocity"], relative)
        rate = float(normal @ relative)
        compression = max(WIDTH - site["gap"], 0)
        magnitude = STIFFNESS * compression * max(1 - DAMPING * rate, 0)
        force = magnitude * normal
        close(site["law_output"], np.r_[rate, magnitude, force])
        close(site["node_forces"], beta[:, None] * force)
        close(site["rigid_force"], -force)
        torque = np.cross(point - expected["q"][:3], -force)
        close(site["rigid_torque"], torque)
        material_load[indices] += beta[:, None] * force
        rigid_force -= force
        rigid_torque += torque
        nodal_power += float(np.sum((beta[:, None] * force) * expected["v"][indices]))
        rigid_power += float((-force) @ expected["speed"][:3] + torque @ expected["speed"][3:])
        relative_power += float(force @ relative)
        # Correct world points make this independently assembled work pairing agree.
        close(
            float(force @ relative),
            float(force @ (beta @ expected["v"][indices]) - force @ point_velocity),
        )
    close(receipt.worlds[world]["node_contact_force"], material_load)
    close(receipt.worlds[world]["rigid_wrench"], np.r_[rigid_force, rigid_torque][None, :])
    close(nodal_power + rigid_power, relative_power)
    # Original D3 independent geometry error propagation, complete material
    # recurrence, rigid M/J/a, both momenta and all fixed reactions remain checked.
    node_step = receipt.node_receipt.worlds[world]
    arithmetic = impulse_auditor()(
        input_position=entry.node.position,
        input_velocity=entry.node.velocity,
        force=node_step.force,
        inverse_mass=item["node"].model.inverse_mass,
        gravity=item["node"].model.gravity,
        h=H,
        predicted_position=node_step.predicted_position,
        predicted_velocity=node_step.predicted_velocity,
        trace=node_step.trace,
        edges=EDGES,
        final_position=after.node.position,
        final_velocity=after.node.velocity,
        sweeps=node_step.sweeps,
    )
    velocity_failures = []
    full_audit = audit_step(
        item,
        entry,
        after,
        receipt,
        H,
        numerical_impulse=arithmetic["total_remainder"],
        material_velocity_diagnostic=velocity_failures.append,
    )
    return dict(
        pair_origins=pair_origins,
        node_power=nodal_power,
        rigid_power=rigid_power,
        relative_power=relative_power,
        full_audit=full_audit,
        stored_arithmetic=arithmetic,
        separate_material_velocity_qualification=dict(
            status="FAIL" if velocity_failures else "PASS", failures=velocity_failures
        ),
    )


@pytest.mark.parametrize("case", ("E", "C"), ids=("sphere-offset", "capsule"))
@pytest.mark.parametrize("translation", ((0, 0, 0), (3, -2, 5)), ids=("base", "translated"))
def test_public_witness_world_frame_in_joint_response(case, translation, tyx_mode, tmp_path):
    item = translated_deformed_fixture(case, translation)
    label = "coordinate-" + case + ("-translated" if any(translation) else "-base")
    with saved(
        tmp_path,
        label,
        tyx_mode,
        case=manifest(item),
        translation=translation,
        initial_surface_displacement=1 / 64,
        initial_angular_velocity=(1 / 32, -1 / 64, 1 / 16),
        tolerances=dict(atol=ATOL, rtol=RTOL),
    ) as evidence:
        with create((item,), tyx_mode) as batch:
            initial = batch.snapshot()[item["world"]]
            anchor = np.asarray(root_state(initial, item).position[:3])
            assert np.all(anchor != 0)
            evidence["declared_floating_anchor"] = anchor.copy()
            records = []
            for tick in range(2):
                entry = batch.snapshot()[item["world"]]
                expected = independent_entry(item, entry)
                record = dict(
                    tick=tick,
                    entry=entry,
                    expected=expected,
                    inputs=forces((item,)),
                    h=H,
                    revision=batch.revision,
                )
                evidence["attempts"].append(record)
                try:
                    receipt = batch.step(record["inputs"], H, expected_revision=record["revision"])
                    record["returned"] = receipt
                    after = batch.snapshot()[item["world"]]
                    record["after"] = after
                    assert receipt.source_revision == record["revision"]
                    assert receipt.destination_revision == batch.revision
                    assert after.node.tick == after.rigid.tick == entry.node.tick + 1
                    record["audit"] = check_coordinate_contract(
                        item, entry, after, receipt, expected
                    )
                    records.append(record)
                except BaseException:
                    record["last_attempt"] = batch.last_attempt
                    raise
            first = records[0]["audit"]["pair_origins"][0]
            second = records[1]["audit"]["pair_origins"][0]
            assert np.linalg.norm(first) > 0
            assert not np.array_equal(first, second), "actual pair frame must evolve"
            before = initial.node.position
            after = records[-1]["after"].node.position
            before_edges = np.linalg.norm(before[FACES[0]][1:] - before[FACES[0]][0], axis=1)
            after_edges = np.linalg.norm(after[FACES[0]][1:] - after[FACES[0]][0], axis=1)
            assert not np.array_equal(before_edges, after_edges), (
                "material changed, not rigid FK only"
            )
            evidence["initial_edges"], evidence["final_edges"] = before_edges, after_edges
