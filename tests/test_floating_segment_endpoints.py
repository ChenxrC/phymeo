"""Two actual cylinder/plane endpoints constrain a complete floating kick/impact/drift."""

from dataclasses import replace

import numpy as np
import pytest

import phymeo as ph
from phymeo.coupling.curved_manifold import select_static_support

UP = np.array([0.0, 1.0, 0.0])
IDENTITY = np.array([0.0, 0.0, 0.0, 1.0])
H = 1 / 64


def detect_support(*, curved_first=False, center=(0.0, 0.5, 0.0), normal=UP):
    low, high = ph.CollisionKey(9, 1, 0), ph.CollisionKey(9, 7, 0)
    cylinder, plane = (low, high) if curved_first else (high, low)
    shapes = ph.CollisionModel(
        [
            ph.CollisionShape(
                cylinder, "cylinder", (31,), curved=ph.CurvedSolid("cylinder", (0.5, 1.0))
            ),
            ph.CollisionShape(plane, "plane", (17,), normal=tuple(normal)),
        ]
    )
    paths = {
        plane: ph.CollisionMotion([np.zeros(3)]),
        cylinder: ph.CollisionMotion([center], body_orientation_xyzw=IDENTITY),
    }
    with ph.CollisionWorkspace(shapes) as workspace:
        (pair,) = ph.detect(shapes, paths, ph.CollisionPolicy(margin=0.25), workspace).pairs
    registered = {shape.key: shape for shape in shapes.shapes}
    return cylinder, plane, pair, (registered[pair.first], registered[pair.second])


def physical_step(indices, *, curved_first=False, speed=-1.0, gravity=0.0, transverse=19 / 48):
    center = np.array([0.0, 0.5, 0.0])
    cylinder, plane, pair, _ = detect_support(curved_first=curved_first)
    ends = np.array([[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]])
    assert pair.status == "touching" and pair.curved.locus == "segment"
    np.testing.assert_array_equal(pair.curved.points_m, ends)
    rigid = ph.RigidBodyModel(np.ones(1), np.array([[transverse, transverse, 1 / 8]]), gravity * UP)
    floating = ph.FloatingRobotModel(rigid, np.zeros(3))
    state = ph.FloatingRobotState(np.r_[center, IDENTITY], np.r_[speed * UP, np.zeros(3)])
    inputs = ph.FloatingRobotInputs(np.empty(0), np.zeros((1, 3)), np.zeros((1, 3)))
    bed = ph.RigidBodyModel(np.zeros(1), np.zeros((1, 3)), np.zeros(3))
    rest = ph.RigidBodyState(
        np.zeros((1, 3)), IDENTITY[None, :], np.zeros((1, 3)), np.zeros((1, 3))
    )
    ids = np.zeros(2, dtype=np.int64)
    sites = {"cylinder": (ids, ends - center), "plane": (ids, ends)}
    rows = [(pair, {cylinder: ("cylinder", i), plane: ("plane", i)}, i) for i in indices]

    def step(at):
        return ph.step_floating_contacts(
            {"cylinder": (floating, at, inputs)},
            sites,
            rows,
            [0.0] * len(rows),
            [0.0] * len(rows),
            H,
            fixed_bodies={"plane": (bed, rest)},
            local_normals={plane: UP},
        )

    return floating, state, rest, sites, step


@pytest.mark.parametrize("curved_first,order", [(False, (0, 1)), (True, (1, 0))])
def test_complete_two_endpoint_impact_stops_the_whole_cylinder(curved_first, order):
    floating, state, rest, sites, step = physical_step(order, curved_first=curved_first)
    before_q, before_v = state.q.copy(), state.v.copy()
    out = step(state)
    assert out.report.accepted, out.report.reason
    # A centered 1 kg body loses 1 m/s. Equal arms require equal impulses for zero torque.
    np.testing.assert_allclose(out.impulses.normal_impulses, [0.5, 0.5], atol=2e-9, rtol=0)
    support = -UP if curved_first else UP  # returned world impulses act on the second shape
    np.testing.assert_allclose(
        out.impulses.world_impulses, np.tile(0.5 * support, (2, 1)), atol=2e-9, rtol=0
    )
    observed = out.states["cylinder"]
    np.testing.assert_allclose(observed.q, before_q, atol=2e-10, rtol=0)
    np.testing.assert_allclose(observed.v, np.zeros(6), atol=2e-9, rtol=0)
    port = ph.FloatingRobotPointPort(floating, observed, *sites["cylinder"])
    np.testing.assert_allclose(port.velocities(), np.zeros((2, 3)), atol=2e-9, rtol=0)
    assert out.impulses.energy_change == pytest.approx(-0.5, abs=2e-9)
    np.testing.assert_array_equal(state.q, before_q)
    np.testing.assert_array_equal(state.v, before_v)
    np.testing.assert_array_equal(rest.velocity, np.zeros((1, 3)))


def test_two_endpoints_support_the_uniform_gravity_kick():
    # A nonuniform 1 kg cylinder with I_transverse = m * half_length**2 makes
    # its two normal response rows independent and preserves exact horizontal contact.
    _, state, _, _, step = physical_step((1, 0), speed=0.0, gravity=-8.0, transverse=1.0)
    original = state.q.copy()
    for _ in range(4):
        out = step(state)
        assert out.report.accepted, out.report.reason
        np.testing.assert_allclose(out.predicted["cylinder"].v[:3], -8 * H * UP, atol=2e-9, rtol=0)
        np.testing.assert_allclose(out.impulses.normal_impulses, [4 * H, 4 * H], atol=2e-9, rtol=0)
        state = out.states["cylinder"]
        np.testing.assert_allclose(state.v, np.zeros(6), atol=2e-9, rtol=0)
        np.testing.assert_allclose(state.q, original, atol=2e-10, rtol=0)


@pytest.mark.parametrize("indices", [(1,), (0, 0)])
def test_incomplete_or_repeated_support_leaves_the_original_state(indices):
    _, state, rest, _, step = physical_step(indices)
    q, v, ground = state.q.copy(), state.v.copy(), rest.copy()
    try:
        out = step(state)
    except ValueError:
        pass
    else:
        assert not out.report.accepted and not out.predicted and out.impulses is None
        np.testing.assert_array_equal(out.states["cylinder"].q, q)
        np.testing.assert_array_equal(out.states["cylinder"].v, v)
    np.testing.assert_array_equal(state.q, q)
    np.testing.assert_array_equal(state.v, v)
    for field in ("position", "orientation", "velocity", "angular_velocity"):
        np.testing.assert_array_equal(getattr(rest, field), getattr(ground, field))


@pytest.mark.parametrize("curved_first", [False, True])
def test_selection_translates_each_owner_offset_preserving_a_nonzero_gap(curved_first):
    center = np.array([0.0, 0.625, 0.0])
    cylinder, plane, pair, registered = detect_support(curved_first=curved_first, center=center)
    old = pair.witness
    assert pair.status == "separated" and old.gap_m == 0.125
    assert select_static_support(pair, old, None) is old
    # Public separated diagnostics have no binding. The internal fresh-pose checker
    # already owns the two registered shapes used by this query.
    with pytest.raises(ValueError):
        select_static_support(replace(pair, witness_shapes=()), old, 1)
    pair = replace(pair, witness_shapes=registered)
    chosen = select_static_support(pair, old, np.int64(1))
    assert chosen.normal == old.normal and chosen.gap_m == old.gap_m
    assert chosen.first_weights == old.first_weights and chosen.second_weights == old.second_weights
    for key, point, offset in (
        (pair.first, chosen.first_point, chosen.first_offset),
        (pair.second, chosen.second_point, chosen.second_offset),
    ):
        if key == cylinder:
            np.testing.assert_array_equal(point, [0.0, 0.125, 1.0])
            np.testing.assert_array_equal(offset, np.asarray(point) - center)
        else:
            assert key == plane
            np.testing.assert_array_equal(point, [0.0, 0.0, 1.0])
            np.testing.assert_array_equal(offset, point)
    for invalid in (-1, 2, 0.0, True, np.bool_(False)):
        with pytest.raises(ValueError):
            select_static_support(pair, old, invalid)
    with pytest.raises(ValueError):
        select_static_support(replace(pair, status="hit"), old, 1)
    _, _, disk, _ = detect_support(center=(0.0, 0.0, 1.0), normal=(0.0, 0.0, 1.0))
    assert disk.curved.locus == "disk"
    with pytest.raises(ValueError):
        select_static_support(disk, disk.witness, 0)
