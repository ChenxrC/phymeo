"""C6 HOST bridge tests-first; author execution is forbidden.

These tests reuse the original finite patch and normal law. They do not build a
second scene, integrate a tree, or duplicate the resident device acceptance suite.
"""

from itertools import product

import numpy as np
import pytest

from phymeo.constraints.collision_shapes import CollisionKey, CollisionShape
from phymeo.constraints.robot_contact import RobotContactMaterial
from phymeo.geometry.contact_patch_adapter import contact_patch_sample
from phymeo.solvers.contact_patch_loads import sample_patch_loads

WORLD = 17


def _box(low, high):
    return np.array(list(product(*zip(low, high, strict=True))), dtype=float)


def _r01_inputs():
    moving = CollisionShape(CollisionKey(WORLD, 100, 11), "hull", tuple(range(101, 109)))
    fixed = CollisionShape(CollisionKey(WORLD, 900, 1), "hull", tuple(range(201, 209)))
    position = np.array([0.0, 0.0, 3 / 32])
    vertices = _box([-1 / 8] * 3, [1 / 8] * 3) + position
    support = _box([-1 / 2, -1 / 2, -1 / 4], [1 / 2, 1 / 2, 0])
    return moving, fixed, position, vertices, support


def _frames(moving, fixed, position, vertices, support, *, velocity=(0, 0, 0), omega=(0, 0, 0)):
    from phymeo.backends._resident_contact_bridge import ContactShapeFrame

    return {
        moving.key: ContactShapeFrame(
            moving, vertices, 0, position, np.eye(3), position, velocity, omega
        ),
        fixed.key: ContactShapeFrame(
            fixed,
            support,
            None,
            np.zeros(3),
            np.eye(3),
            support.mean(axis=0),
            np.zeros(3),
            np.zeros(3),
        ),
    }


def _sample(frames, law=None, *, capacity=128, groups=1, candidates=None):
    from phymeo.backends._resident_contact_bridge import sample_tree_contacts

    pair = tuple(sorted(frames))
    return sample_tree_contacts(
        world=WORLD,
        frames=frames,
        materials_by_pair={pair: RobotContactMaterial(64, 0, 0) if law is None else law},
        candidates=(pair,) if candidates is None else candidates,
        group_count=groups,
        site_capacity=capacity,
    )


def test_a01_real_finite_patch_two_newtons_reaches_owned_bridge():
    moving, fixed, position, vertices, support = _r01_inputs()
    law = RobotContactMaterial(64, 0, 0)
    patch = contact_patch_sample(moving, fixed, vertices, support)
    assert len(patch.sites) == 4
    assert sum(site.weight for site in patch.sites) == pytest.approx(1, abs=1e-14)
    original = sample_patch_loads(patch, law, np.zeros((4, 3)), np.zeros((4, 3)))
    np.testing.assert_allclose(original["first_force"].sum(axis=0), (0, 0, 2), atol=1e-12, rtol=0)
    # Only after the old geometric/law precondition do we consume the new bridge.
    loads = _sample(_frames(moving, fixed, position, vertices, support), law)
    assert loads.world == WORLD
    np.testing.assert_allclose(loads.wrench_fp64, [[0, 0, 2, 0, 0, 0]], atol=1e-12, rtol=0)
    reaction = loads.static_reactions[fixed.key]
    np.testing.assert_allclose(reaction.reference_point_m, (0, 0, -1 / 8), atol=1e-14, rtol=0)
    np.testing.assert_allclose(reaction.force_n, (0, 0, -2), atol=1e-12, rtol=0)
    np.testing.assert_allclose(reaction.torque_nm, 0, atol=1e-12, rtol=0)
    assert len(loads.sites) == 4
    for site in loads.sites:
        assert site.pair == (moving.key, fixed.key)
        assert site.geometry.witness.gap_m == pytest.approx(-1 / 32, abs=1e-12)
        assert site.law == law
        np.testing.assert_array_equal(site.first_force_n, -site.second_force_n)
        assert site.application_work_j is None and site.material_work_j is None
        assert site.first_material_displacement_m is None
        assert site.second_application_displacement_m is None


def _frame(
    shape,
    points,
    *,
    group=0,
    origin=(0, 0, 0),
    rotation=None,
    com=None,
    velocity=(0, 0, 0),
    omega=(0, 0, 0),
):
    from phymeo.backends._resident_contact_bridge import ContactShapeFrame

    points = np.asarray(points, dtype=float)
    if group is None:
        origin, rotation, com, velocity, omega = (
            np.zeros(3),
            np.eye(3),
            points.mean(axis=0),
            np.zeros(3),
            np.zeros(3),
        )
    return ContactShapeFrame(
        shape,
        points,
        group,
        origin,
        np.eye(3) if rotation is None else rotation,
        origin if com is None else com,
        velocity,
        omega,
    )


def _sphere_pair(*, capsule=False, x=0.0):
    moving = CollisionShape(CollisionKey(WORLD, 100, 7), "sphere", (71,), radius=1 / 8)
    fixed = CollisionShape(
        CollisionKey(WORLD, 900, 4),
        "capsule" if capsule else "sphere",
        (501, 509) if capsule else (501,),
        radius=1 / 8,
    )
    position = (x, 0, 3 / 32)
    support = ((-1 / 4, 0, -1 / 8), (1 / 4, 0, -1 / 8)) if capsule else ((0, 0, -1 / 8),)
    return {
        moving.key: _frame(moving, (position,), origin=position),
        fixed.key: _frame(fixed, support, group=None),
    }


def _rotor_frames(*, damping_case=False):
    moving = CollisionShape(CollisionKey(WORLD, 100, 7), "sphere", (71,), radius=1 / 8)
    fixed = CollisionShape(CollisionKey(WORLD, 900, 4), "hull", tuple(range(501, 509)))
    return {
        moving.key: _frame(
            moving, ((1 / 2, 3 / 32, 0),), omega=(0, 0, 1 / 2) if damping_case else (0, 0, 0)
        ),
        fixed.key: _frame(fixed, _box((-1, -1 / 4, -1 / 2), (1, 0, 1 / 2)), group=None),
    }


@pytest.mark.parametrize(
    "capsule,x,normal_force", [(False, 0, 2), (True, 0, 2), (True, 1 / 4, 2), (True, 1, 0)]
)
def test_a02_finite_sphere_capsule_shaft_endpoint_and_beyond(capsule, x, normal_force):
    frames = _sphere_pair(capsule=capsule, x=x)
    loads = _sample(frames)
    assert len(loads.sites) == 1
    np.testing.assert_allclose(loads.wrench_fp64[0, :3], (0, 0, normal_force), atol=1e-12, rtol=0)
    np.testing.assert_allclose(loads.wrench_fp64[0, 3:], 0, atol=1e-12, rtol=0)
    if normal_force:
        assert loads.sites[0].geometry.witness.gap_m == pytest.approx(-1 / 32, abs=1e-12)
    else:
        assert loads.sites[0].geometry.witness.gap_m > 0
    # Original IDs and world-space skin offsets, not an infinite cylinder proxy.
    site = loads.sites[0]
    for key, weights, skin, point in (
        (
            site.pair[0],
            site.geometry.witness.first_weights,
            site.geometry.witness.first_offset,
            site.geometry.witness.first_point,
        ),
        (
            site.pair[1],
            site.geometry.witness.second_weights,
            site.geometry.witness.second_offset,
            site.geometry.witness.second_point,
        ),
    ):
        by_id = dict(zip(frames[key].shape.vertex_ids, frames[key].world_points_m, strict=True))
        reconstructed = sum(weight * by_id[node] for node, weight in weights)
        reconstructed += np.zeros(3) if skin is None else skin
        np.testing.assert_allclose(reconstructed, point, atol=1e-12, rtol=0)


@pytest.mark.parametrize("speed,expected", [(1 / 8, 1.5), (-1 / 8, 2.5)])
def test_a03_law_uses_entry_com_velocity_and_weights_only_once(speed, expected):
    frames = _frames(*_r01_inputs(), velocity=(0, 0, speed))
    loads = _sample(frames, RobotContactMaterial(64, 4, 0))
    assert len(loads.sites) == 4
    np.testing.assert_allclose(loads.wrench_fp64[0], (0, 0, expected, 0, 0, 0), atol=1e-12, rtol=0)
    assert sum(site.geometry.weight for site in loads.sites) == pytest.approx(1)
    for site in loads.sites:
        np.testing.assert_allclose(
            site.first_force_n, (0, 0, expected * site.geometry.weight), atol=1e-12, rtol=0
        )
        np.testing.assert_array_equal(site.first_velocity_m_s, (0, 0, speed))


def test_a04_same_group_two_shapes_preserve_com_torque_and_static_reaction():
    from phymeo.backends._resident_contact_bridge import sample_tree_contacts

    fixed = CollisionShape(CollisionKey(WORLD, 900, 4), "hull", tuple(range(501, 509)))
    support = _frame(fixed, _box((-1 / 2, -1 / 4, -1 / 2), (1 / 2, 0, 1 / 2)), group=None)
    frames = {fixed.key: support}
    pairs = []
    for number, z in enumerate((-1 / 4, 1 / 4)):
        shape = CollisionShape(
            CollisionKey(WORLD, 100, 10 + number), "sphere", (70 + number,), radius=1 / 8
        )
        frames[shape.key] = _frame(
            shape, ((0, 7 / 64, z),), group=1, origin=(1, 0, 0), com=(1, 0, 0)
        )
        pairs.append((shape.key, fixed.key))
    loads = sample_tree_contacts(
        world=WORLD,
        frames=frames,
        materials_by_pair={pair: RobotContactMaterial(64, 0, 0) for pair in pairs},
        candidates=tuple(reversed(pairs)),
        group_count=3,
        site_capacity=2,
    )
    # Each independent finite sphere contributes 1 N at x=0. Their opposite z
    # moment arms cancel, while each COM at x=1 contributes -1 N m about z.
    np.testing.assert_allclose(
        loads.wrench_fp64,
        ((0, 0, 0, 0, 0, 0), (0, 2, 0, 0, 0, -2), (0, 0, 0, 0, 0, 0)),
        atol=1e-12,
        rtol=0,
    )
    reaction = loads.static_reactions[fixed.key]
    np.testing.assert_allclose(reaction.force_n, (0, -2, 0), atol=1e-12, rtol=0)
    np.testing.assert_allclose(reaction.torque_nm, 0, atol=1e-12, rtol=0)
    # Sum moments about one world origin, not the sum of unlike COM torques.
    dynamic_torque = loads.wrench_fp64[1, 3:] + np.cross((1, 0, 0), loads.wrench_fp64[1, :3])
    static_torque = reaction.torque_nm + np.cross(reaction.reference_point_m, reaction.force_n)
    np.testing.assert_allclose(dynamic_torque + static_torque, 0, atol=1e-12, rtol=0)
    assert tuple(site.pair for site in loads.sites) == tuple(pairs)


def test_a05_angular_velocity_at_common_point_controls_original_damping():
    loads = _sample(_rotor_frames(damping_case=True), RobotContactMaterial(64, 4, 0))
    site = loads.sites[0]
    # Common p=(1/2,-1/64,0), omega=(0,0,1/2): v=(1/128,1/4,0).
    np.testing.assert_allclose(site.geometry.load_point, (1 / 2, -1 / 64, 0), atol=1e-12, rtol=0)
    np.testing.assert_allclose(site.first_velocity_m_s, (1 / 128, 1 / 4, 0), atol=1e-12, rtol=0)
    np.testing.assert_allclose(loads.wrench_fp64[0], (0, 1, 0, 0, 0, 1 / 2), atol=1e-12, rtol=0)


def test_a06_key_order_and_rigid_axis_cycling_do_not_change_physical_load():
    from dataclasses import replace

    original = _frames(*_r01_inputs())
    base = _sample(original)
    cycle = np.array(((0, 1, 0), (0, 0, 1), (1, 0, 0)), dtype=float)
    shift = np.array((3, -2, 5), dtype=float)
    moved = {}
    for key, frame in original.items():
        shape = replace(
            frame.shape,
            key=CollisionKey(WORLD, 1000 if frame.group_index is not None else 1, key.shape),
        )
        points = frame.world_points_m @ cycle.T + shift
        moved[shape.key] = _frame(
            shape,
            points,
            group=frame.group_index,
            origin=cycle @ frame.position_m + shift,
            rotation=cycle @ frame.rotation,
            com=cycle @ frame.reference_point_m + shift,
            velocity=cycle @ frame.linear_velocity_m_s,
            omega=cycle @ frame.angular_velocity_rad_s,
        )
    loads = _sample(dict(reversed(tuple(moved.items()))))
    assert loads.sites[0].pair[0].body == 1  # Static is now the first ordered owner.
    np.testing.assert_allclose(
        loads.wrench_fp64[0, :3], cycle @ base.wrench_fp64[0, :3], atol=1e-12, rtol=0
    )
    np.testing.assert_allclose(
        loads.wrench_fp64[0, 3:], cycle @ base.wrench_fp64[0, 3:], atol=1e-12, rtol=0
    )
    for site in loads.sites:
        np.testing.assert_array_equal(site.first_force_n, -site.second_force_n)
        assert all(node >= 201 for node, _ in site.geometry.witness.first_weights)
        assert all(101 <= node < 109 for node, _ in site.geometry.witness.second_weights)


def test_a07_rotated_box_on_actual_nonbox_finite_hull():
    moving, fixed, position, vertices, _ = _r01_inputs()
    cosine, sine = np.cos(0.37), np.sin(0.37)
    rotation = np.array(((cosine, -sine, 0), (sine, cosine, 0), (0, 0, 1)))
    points = (vertices - position) @ rotation.T + position
    support = np.array(
        [
            (x, y, z)
            for z, half in ((-1 / 4, 3 / 4), (0, 1 / 2))
            for x, y in product((-half, half), repeat=2)
        ]
    )
    frames = {
        moving.key: _frame(moving, points, origin=position, rotation=rotation),
        fixed.key: _frame(fixed, support, group=None),
    }
    loads = _sample(frames)
    assert len(loads.sites) == 4
    np.testing.assert_allclose(loads.wrench_fp64[0], (0, 0, 2, 0, 0, 0), atol=1e-12, rtol=0)
    for site in loads.sites:
        assert set(node for node, _ in site.geometry.witness.second_weights) <= set(
            fixed.vertex_ids[4:]
        )


def test_b01_zero_force_geometry_counts_towards_capacity_and_retry():
    from phymeo.backends._resident_contact_bridge import ContactSiteCapacityExceeded

    frames = _sphere_pair(capsule=True, x=1)
    with pytest.raises(ContactSiteCapacityExceeded) as raised:
        _sample(frames, capacity=0)
    assert (raised.value.count, raised.value.limit) == (1, 0)
    loads = _sample(frames, capacity=1)
    assert len(loads.sites) == 1
    np.testing.assert_array_equal(loads.wrench_fp64, 0)
    assert loads.sites[0].geometry.witness.gap_m > 0
    with pytest.raises(ContactSiteCapacityExceeded) as raised:
        _sample(_frames(*_r01_inputs()), capacity=3)
    assert (raised.value.count, raised.value.limit) == (4, 3)


def test_b02_empty_candidates_and_empty_world_preserve_real_group_dimensions():
    from phymeo.backends._resident_contact_bridge import finish_contact_loads, sample_tree_contacts

    frames = _frames(*_r01_inputs())
    loads = _sample(frames, groups=16, candidates=())
    assert not loads.sites
    np.testing.assert_array_equal(loads.wrench_fp64, np.zeros((16, 6)))
    empty = sample_tree_contacts(
        world=WORLD, frames={}, materials_by_pair={}, candidates=(), group_count=3, site_capacity=0
    )
    assert not empty.frames and not empty.static_reactions and not empty.sites
    np.testing.assert_array_equal(empty.wrench_fp64, np.zeros((3, 6)))
    assert finish_contact_loads(empty, {}).finalized
    pair = tuple(sorted(frames))
    with pytest.raises(ValueError, match="absent"):
        sample_tree_contacts(
            world=WORLD,
            frames={},
            materials_by_pair={pair: RobotContactMaterial(64, 0, 0)},
            candidates=(),
            group_count=3,
            site_capacity=0,
        )
    with pytest.raises(ValueError):
        sample_tree_contacts(
            world=WORLD,
            frames={},
            materials_by_pair={},
            candidates=(pair,),
            group_count=3,
            site_capacity=0,
        )


@pytest.mark.parametrize(
    "fault",
    [
        "reverse",
        "duplicate",
        "missing_law",
        "world",
        "dynamic_dynamic",
        "static_static",
        "group_range",
        "group_source",
        "friction",
    ],
)
def test_b03_source_owner_law_and_candidate_contract_rejects(fault):
    from dataclasses import replace

    from phymeo.backends._resident_contact_bridge import sample_tree_contacts

    frames = _sphere_pair()
    first, second = tuple(sorted(frames))
    pair = (first, second)
    laws, candidates, world, count = {pair: RobotContactMaterial(64, 0, 0)}, (pair,), WORLD, 1
    if fault == "reverse":
        candidates = ((second, first),)
    elif fault == "duplicate":
        candidates = (pair, pair)
    elif fault == "missing_law":
        laws = {}
    elif fault == "world":
        world += 1
    elif fault == "dynamic_dynamic":
        frames[second] = replace(
            frames[second],
            group_index=0,
            position_m=frames[first].position_m,
            reference_point_m=frames[first].reference_point_m,
        )
    elif fault == "static_static":
        frames[first] = _frame(frames[first].shape, frames[first].world_points_m, group=None)
    elif fault == "group_range":
        frames[first] = replace(frames[first], group_index=1)
    elif fault == "group_source":
        extra = replace(frames[first].shape, key=CollisionKey(WORLD, 100, 99))
        frames[extra.key] = replace(frames[first], shape=extra, linear_velocity_m_s=(1, 0, 0))
    elif fault == "friction":
        laws[pair] = RobotContactMaterial(64, 0, 0.1)
    with pytest.raises(ValueError):
        sample_tree_contacts(
            world=world,
            frames=frames,
            materials_by_pair=laws,
            candidates=candidates,
            group_count=count,
            site_capacity=128,
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("group_count", 0),
        ("group_count", 34),
        ("group_count", True),
        ("site_capacity", -1),
        ("site_capacity", 4097),
        ("site_capacity", False),
    ],
)
def test_b04_existing_resource_limits_are_explicit(field, value):
    from phymeo.backends._resident_contact_bridge import sample_tree_contacts

    args = dict(
        world=WORLD, frames={}, materials_by_pair={}, candidates=(), group_count=1, site_capacity=0
    )
    args[field] = value
    with pytest.raises(ValueError):
        sample_tree_contacts(**args)


@pytest.mark.parametrize("groups", (17, 30, 33))
def test_high_tree_group_retains_force_moment_and_finished_work(groups):
    """The last real body owns the load; preceding bodies receive no ghost load."""
    from dataclasses import replace

    from phymeo.backends._resident_contact_bridge import finish_contact_loads

    frames = _rotor_frames()
    first = min(frames)
    frames[first] = replace(frames[first], group_index=groups - 1)
    loads = _sample(frames, groups=groups)
    assert loads.wrench_fp64.shape == (groups, 6)
    np.testing.assert_array_equal(loads.wrench_fp64[:-1], 0)
    # F=(0,2,0) at x=1/2 gives a real +z COM moment of 1 Nm.
    np.testing.assert_allclose(loads.wrench_fp64[-1], (0, 2, 0, 0, 0, 1), atol=1e-12, rtol=0)
    finished = finish_contact_loads(loads, _rotated_candidate(frames))
    assert finished.finalized
    np.testing.assert_array_equal(finished.wrench_fp64, loads.wrench_fp64)
    assert finished.sites[0].application_work_j == pytest.approx(13 / 16, abs=1e-12)
    assert finished.sites[0].material_work_j == pytest.approx(33 / 40, abs=1e-12)
    with pytest.raises(ValueError, match="group count"):
        replace(loads, wrench_fp64=np.zeros((34, 6)))


def test_b05_unsupported_and_invalid_finite_geometry_cannot_become_zero_load():
    from phymeo.core.capabilities import SolverFailure

    plane = CollisionShape(CollisionKey(WORLD, 100, 1), "plane", (1,), normal=(0, 0, 1))
    with pytest.raises(ValueError, match="supports"):
        _frame(plane, ((0, 0, 0),))
    moving, _, position, points, _ = _r01_inputs()
    flat = points.copy()
    flat[:, 2] = 0
    with pytest.raises((ValueError, SolverFailure)):
        _frame(moving, flat, origin=position)
    # Inside-hull round-axis ambiguity is the original geometry's rejection,
    # not a separation or a fabricated zero normal.
    frames = _rotor_frames()
    first = min(frames)
    frames[first] = _frame(frames[first].shape, ((0, -1 / 8, 0),))
    with pytest.raises((ValueError, SolverFailure)):
        _sample(frames)


def test_b06_original_law_failure_propagates_without_mutating_entry(monkeypatch):
    from phymeo.backends import _resident_contact_bridge as bridge

    frames = _frames(*_r01_inputs())
    saved = {key: frame.world_points_m.tobytes() for key, frame in frames.items()}
    original = bridge.sample_patch_loads

    def failed(*args, **kwargs):
        raise RuntimeError("injected original law failure")

    monkeypatch.setattr(bridge, "sample_patch_loads", failed)
    with pytest.raises(RuntimeError, match="original law failure"):
        _sample(frames)
    assert saved == {key: frame.world_points_m.tobytes() for key, frame in frames.items()}
    monkeypatch.setattr(bridge, "sample_patch_loads", original)
    np.testing.assert_allclose(_sample(frames).wrench_fp64[0, 2], 2, atol=1e-12, rtol=0)


def test_c01_packets_own_all_source_arrays_and_mapping_values():
    from dataclasses import FrozenInstanceError

    moving, fixed, position, points, support = _r01_inputs()
    frames = _frames(moving, fixed, position, points, support)
    loads = _sample(frames)
    position[:] = 99
    points[:] = 99
    support[:] = 99
    frames.clear()
    assert len(loads.frames) == 2 and len(loads.sites) == 4
    np.testing.assert_allclose(loads.wrench_fp64[0, 2], 2, atol=1e-12, rtol=0)
    assert loads.frames[moving.key].position_m[2] == 3 / 32
    with pytest.raises(TypeError):
        loads.frames[moving.key] = None
    with pytest.raises(TypeError):
        loads.static_reactions[fixed.key] = None
    with pytest.raises(FrozenInstanceError):
        loads.sites[0].law = RobotContactMaterial(32, 0, 0)
    arrays = [loads.wrench_fp64]
    for frame in loads.frames.values():
        arrays.extend(
            getattr(frame, name)
            for name in (
                "world_points_m",
                "position_m",
                "rotation",
                "reference_point_m",
                "linear_velocity_m_s",
                "angular_velocity_rad_s",
            )
        )
    site = loads.sites[0]
    arrays.extend(
        getattr(site, name)
        for name in (
            "first_velocity_m_s",
            "second_velocity_m_s",
            "first_force_n",
            "second_force_n",
            "first_reference_point_m",
            "second_reference_point_m",
        )
    )
    reaction = loads.static_reactions[fixed.key]
    arrays.extend((reaction.reference_point_m, reaction.force_n, reaction.torque_nm))
    for array in arrays:
        assert array.dtype == np.float64 and not array.flags.writeable
        with pytest.raises(ValueError):
            array.setflags(write=True)


def _rotated_candidate(frames, *, cosine=3 / 5, sine=4 / 5):
    from dataclasses import replace

    rotation = np.array(((cosine, -sine, 0), (sine, cosine, 0), (0, 0, 1)))
    result = dict(frames)
    for key, entry in frames.items():
        if entry.group_index is not None:
            result[key] = replace(
                entry,
                world_points_m=entry.world_points_m @ rotation.T,
                position_m=rotation @ entry.position_m,
                rotation=rotation @ entry.rotation,
                reference_point_m=rotation @ entry.reference_point_m,
                linear_velocity_m_s=(21, -13, 8),
                angular_velocity_rad_s=(3, 7, 11),
            )
    return result


def test_c02_actual_rotated_material_and_application_work_are_distinct():
    from phymeo.backends._resident_contact_bridge import finish_contact_loads

    frames = _rotor_frames()
    entry = _sample(frames)
    final = finish_contact_loads(entry, _rotated_candidate(frames))
    site = final.sites[0]
    # Entry material point=(1/2,-1/32,0), common point=(1/2,-1/64,0).
    # Rz has cos=3/5, sin=4/5. Explicit (R-I)p, with no endpoint force resample.
    np.testing.assert_allclose(
        site.first_material_displacement_m, (-7 / 40, 33 / 80, 0), atol=1e-12, rtol=0
    )
    np.testing.assert_allclose(
        site.first_application_displacement_m, (-3 / 16, 13 / 32, 0), atol=1e-12, rtol=0
    )
    np.testing.assert_array_equal(site.second_material_displacement_m, 0)
    np.testing.assert_array_equal(site.second_application_displacement_m, 0)
    assert site.material_work_j == pytest.approx(33 / 40, abs=1e-12)
    assert site.application_work_j == pytest.approx(13 / 16, abs=1e-12)
    assert site.material_work_j != site.application_work_j
    np.testing.assert_array_equal(site.first_force_n, entry.sites[0].first_force_n)
    np.testing.assert_array_equal(site.first_velocity_m_s, entry.sites[0].first_velocity_m_s)
    assert not entry.finalized and entry.sites[0].application_work_j is None
    assert final.finalized and final.frames is not entry.frames
    with pytest.raises(ValueError, match="transient"):
        finish_contact_loads(final, _rotated_candidate(frames))
    # Pure finish can be retried against the unchanged original packet.
    retry = finish_contact_loads(entry, _rotated_candidate(frames))
    np.testing.assert_array_equal(
        retry.sites[0].first_material_displacement_m, site.first_material_displacement_m
    )
    for name in (
        "first_material_displacement_m",
        "second_material_displacement_m",
        "first_application_displacement_m",
        "second_application_displacement_m",
    ):
        with pytest.raises(ValueError):
            getattr(site, name).setflags(write=True)


def test_c03_stored_near_rigid_matrix_is_preserved_and_inverted_not_transposed():
    from dataclasses import replace

    from phymeo.backends._resident_contact_bridge import finish_contact_loads

    frames = _rotor_frames()
    first = min(frames)
    stretch = 1 + 2e-7
    stored = np.diag((stretch, 1, 1))
    frames[first] = replace(frames[first], rotation=stored)
    entry = _sample(frames)
    error = frames[first].rotation_orthogonality_error
    assert 1e-10 < error < 5e-5
    np.testing.assert_array_equal(entry.frames[first].rotation, stored)
    end = dict(frames)
    end[first] = replace(
        frames[first], rotation=np.eye(3), world_points_m=((0.5 / stretch, 3 / 32, 0),)
    )
    site = finish_contact_loads(entry, end).sites[0]
    expected = (0.5 / stretch - 0.5, 0, 0)
    np.testing.assert_allclose(site.first_material_displacement_m, expected, atol=1e-12, rtol=0)
    np.testing.assert_allclose(site.first_application_displacement_m, expected, atol=1e-12, rtol=0)
    assert site.application_work_j == pytest.approx(0, abs=1e-12)


@pytest.mark.parametrize(
    "rotation", [np.diag((1.001, 1, 1)), np.diag((-1, 1, 1)), np.zeros((3, 3))]
)
def test_c04_invalid_stored_rotation_is_rejected_without_repair(rotation):
    from dataclasses import replace

    entry = _rotor_frames()
    with pytest.raises(ValueError, match="near-rigid"):
        replace(entry[min(entry)], rotation=rotation)


@pytest.mark.parametrize(
    "fault", ["shape", "radius", "material_point", "group", "static_move", "missing", "key"]
)
def test_c05_candidate_source_change_rejects_and_original_retry_is_unchanged(fault):
    from dataclasses import replace

    from phymeo.backends._resident_contact_bridge import finish_contact_loads

    frames = _rotor_frames()
    loads = _sample(frames)
    good = _rotated_candidate(frames)
    first, second = sorted(frames)
    bad = dict(good)
    if fault == "shape":
        bad[first] = replace(good[first], shape=replace(good[first].shape, vertex_ids=(999,)))
    elif fault == "radius":
        bad[first] = replace(good[first], shape=replace(good[first].shape, radius=1 / 4))
    elif fault == "material_point":
        bad[first] = replace(
            good[first], world_points_m=good[first].world_points_m + (1 / 1024, 0, 0)
        )
    elif fault == "group":
        bad[first] = replace(good[first], group_index=1)
    elif fault == "static_move":
        points = good[second].world_points_m + (1 / 1024, 0, 0)
        bad[second] = _frame(good[second].shape, points, group=None)
    elif fault == "missing":
        del bad[second]
    elif fault == "key":
        other_key = CollisionKey(WORLD + 1, first.body, first.shape)
        bad[other_key] = bad.pop(first)
    with pytest.raises(ValueError):
        finish_contact_loads(loads, bad)
    assert not loads.finalized and loads.sites[0].application_work_j is None
    np.testing.assert_allclose(loads.wrench_fp64[0], (0, 2, 0, 0, 0, 1), atol=1e-12, rtol=0)
    assert finish_contact_loads(loads, good).finalized


@pytest.mark.parametrize("center_z,expected_force", [(3 / 32, 2), (1 / 4, 0)])
def test_a08_dynamic_finite_capsule_has_two_actual_line_sites(center_z, expected_force):
    moving = CollisionShape(CollisionKey(WORLD, 100, 12), "capsule", (17, 43), radius=1 / 8)
    fixed = CollisionShape(CollisionKey(WORLD, 900, 1), "hull", tuple(range(201, 209)))
    frames = {
        moving.key: _frame(
            moving, ((-1 / 4, 0, center_z), (1 / 4, 0, center_z)), origin=(0, 0, center_z)
        ),
        fixed.key: _frame(fixed, _box((-1 / 2, -1 / 2, -1 / 4), (1 / 2, 1 / 2, 0)), group=None),
    }
    loads = _sample(frames)
    assert len(loads.sites) == 2
    assert {site.geometry.witness.first_weights for site in loads.sites} == {
        ((17, 1.0), (43, 0.0)),
        ((17, 0.0), (43, 1.0)),
    }
    np.testing.assert_allclose(
        loads.wrench_fp64[0], (0, 0, expected_force, 0, 0, 0), atol=1e-12, rtol=0
    )
    for site in loads.sites:
        assert site.geometry.weight == pytest.approx(1 / 2)
        assert site.first_force_n[2] == pytest.approx(expected_force / 2, abs=1e-12)


def test_c06_candidate_actual_com_is_not_reconstructed_from_independently_rounded_fk():
    from dataclasses import replace

    from phymeo.backends._resident_contact_bridge import finish_contact_loads

    frames = _rotor_frames()
    first = min(frames)
    loads = _sample(frames)
    candidate = _rotated_candidate(frames)
    # This input represents a separately rounded stored COM. Source/model proof
    # belongs to the resident parent plan, not an invented 1e-10 FK/COM identity.
    candidate[first] = replace(candidate[first], reference_point_m=(2**-24, 0, 0))
    final = finish_contact_loads(loads, candidate)
    assert final.finalized
    assert candidate[first].reference_point_m[0] == 2**-24
    np.testing.assert_array_equal(final.wrench_fp64, loads.wrench_fp64)
    assert final.sites[0].application_work_j == pytest.approx(13 / 16, abs=1e-12)


@pytest.mark.parametrize(
    "field,value",
    [
        ("world_points_m", ((np.nan, 0, 0),)),
        ("reference_point_m", (0, np.inf, 0)),
        ("linear_velocity_m_s", (0, 0)),
        ("angular_velocity_rad_s", (False, False, False)),
    ],
)
def test_c07_invalid_packet_vectors_reject_before_sampling(field, value):
    from dataclasses import replace

    frame = _sphere_pair()[CollisionKey(WORLD, 100, 7)]
    with pytest.raises(ValueError):
        replace(frame, **{field: value})
