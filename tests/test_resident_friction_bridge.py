"""Nonzero-mu HOST bridge oracles; authoring only until the parent serial run."""

import json
from collections.abc import Mapping
from dataclasses import fields, is_dataclass, replace
from itertools import product

import numpy as np
import pytest

from phymeo.backends._resident_contact_bridge import (
    ContactShapeFrame,
    ContactSiteCapacityExceeded,
    finish_contact_loads,
    sample_tree_contacts,
)
from phymeo.constraints.collision_shapes import CollisionKey, CollisionShape
from phymeo.constraints.robot_contact import RobotContactMaterial
from phymeo.geometry.contact_patch_adapter import contact_patch_sample
from phymeo.solvers.contact_patch_loads import sample_patch_loads

WORLD = 67
LAW = RobotContactMaterial(64, 4, 1 / 2, 1 / 8)


def _json(value):
    if is_dataclass(value):
        return {f.name: _json(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _json(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (tuple, list)):
        return [_json(item) for item in value]
    return value


def _box(low, high):
    return np.array(list(product(*zip(low, high, strict=True))), dtype=float)


def _inputs(speed=1 / 4, *, height=3 / 32, normal_speed=-1 / 8, dynamic_second=False):
    first = CollisionShape(CollisionKey(WORLD, 101, 3), "hull", tuple(range(10, 18)))
    second = CollisionShape(CollisionKey(WORLD, 909, 8), "hull", tuple(range(30, 38)))
    center = np.array([0.0, 0, height])
    points = _box([-1 / 8] * 3, [1 / 8] * 3) + center
    support = _box([-1 / 2, -1 / 2, -1 / 4], [1 / 2, 1 / 2, 0])
    shared_speed = 1 / 8 if dynamic_second else 0
    frames = {
        first.key: ContactShapeFrame(
            first, points, 0, center, np.eye(3), center,
            [speed + shared_speed, 0, normal_speed], np.zeros(3),
        ),
        second.key: ContactShapeFrame(
            second, support, 1 if dynamic_second else None, np.zeros(3), np.eye(3),
            support.mean(axis=0), [shared_speed, 0, 0], np.zeros(3),
        ),
    }
    return frames, (first.key, second.key)


def _sample(frames, pair, *, capacity=4, dynamic_second=False):
    return sample_tree_contacts(
        world=WORLD,
        frames=frames,
        materials_by_pair={pair: LAW},
        candidates=(pair,),
        group_count=2 if dynamic_second else 1,
        site_capacity=capacity,
        tree_dynamic_pairs=dynamic_second,
    )


def _close(actual, expected):
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)


@pytest.mark.parametrize("speed,fx", ((0, 0), (1 / 16, -5 / 8), (1 / 4, -5 / 4)))
def test_mu_half_regularized_force_weight_and_reciprocal_moment(speed, fx, tmp_path):
    frames, pair = _inputs(speed)
    patch = contact_patch_sample(
        frames[pair[0]].shape, frames[pair[1]].shape,
        frames[pair[0]].world_points_m, frames[pair[1]].world_points_m,
    )
    assert len(patch.sites) == 4 and patch.dimension == 2
    _close([site.weight for site in patch.sites], np.full(4, 1 / 4))
    # Literal spring/dashpot force: 64/32 - 4*(-1/8) = 5/2.
    force = np.array([fx, 0, 5 / 2])
    original = sample_patch_loads(
        patch, LAW, np.tile([speed, 0, -1 / 8], (4, 1)), np.zeros((4, 3))
    )
    _close(original["first_force"].sum(axis=0), force)
    record = dict(frames=frames, law=LAW, literal_force=force, host_precondition=original)
    try:
        loads = _sample(frames, pair)
        record["bridge"] = loads
        _close(loads.wrench_fp64[0, :3], force)
        # Common midpoint z=-1/64; moving COM z=3/32 => lever z=-7/64.
        _close(loads.wrench_fp64[0, 3:], [0, -7 * fx / 64, 0])
        reaction = loads.static_reactions[pair[1]]
        _close(reaction.force_n, -force)
        moment = (
            np.cross(frames[pair[0]].reference_point_m, loads.wrench_fp64[0, :3])
            + loads.wrench_fp64[0, 3:]
            + np.cross(reaction.reference_point_m, reaction.force_n)
            + reaction.torque_nm
        )
        _close(moment, np.zeros(3))
        assert sum(site.geometry.weight for site in loads.sites) == pytest.approx(1)
        for site in loads.sites:
            _close(site.first_force_n, force / 4)
            _close(site.second_force_n, -force / 4)
            _close(site.first_velocity_m_s, [speed, 0, -1 / 8])
            _close(site.second_velocity_m_s, np.zeros(3))
            assert site.geometry.witness.gap_m == pytest.approx(-1 / 32)
            assert site.material_work_j is None
            with pytest.raises(ValueError):
                site.first_force_n.flags.writeable = True
        friction_power = fx * speed
        assert friction_power <= 0
        _close(
            sum(
                site.first_force_n @ site.first_velocity_m_s
                + site.second_force_n @ site.second_velocity_m_s
                for site in loads.sites
            ),
            friction_power - 5 / 16,
        )
    finally:
        (tmp_path / ("mu-half-" + str(speed) + ".json")).write_text(
            json.dumps(_json(record), allow_nan=False), encoding="utf-8"
        )


@pytest.mark.parametrize("normal_speed,fn", ((1 / 8, 3 / 2), (1, 0)))
def test_opening_dashpot_clamps_without_adhesion(normal_speed, fn):
    frames, pair = _inputs(1 / 4, normal_speed=normal_speed)
    loads = _sample(frames, pair)
    _close(loads.wrench_fp64[0, :3], [-fn / 2, 0, fn])


def test_separated_finite_patch_has_no_normal_or_friction_load():
    frames, pair = _inputs(1 / 4, height=5 / 32, normal_speed=-10)
    loads = _sample(frames, pair)
    assert loads.sites
    assert all(site.geometry.witness.gap_m > 0 for site in loads.sites)
    _close(loads.wrench_fp64, np.zeros((1, 6)))
    _close(loads.static_reactions[pair[1]].force_n, np.zeros(3))


def test_finish_keeps_input_force_and_records_actual_material_and_application_work(tmp_path):
    frames, pair = _inputs()
    entry = _sample(frames, pair)
    shift = np.array([1 / 64, 0, -1 / 128])
    candidate = dict(frames)
    original = frames[pair[0]]
    candidate[pair[0]] = replace(
        original,
        world_points_m=original.world_points_m + shift,
        position_m=original.position_m + shift,
        reference_point_m=original.reference_point_m + shift,
        linear_velocity_m_s=np.array([10.0, 20, 30]),
    )
    final = finish_contact_loads(entry, candidate)
    _close(sum(site.material_work_j for site in final.sites), -5 / 128)
    _close(sum(site.application_work_j for site in final.sites), -5 / 128)
    for before, after in zip(entry.sites, final.sites, strict=True):
        _close(after.first_force_n, before.first_force_n)
        _close(after.first_velocity_m_s, [1 / 4, 0, -1 / 8])
        _close(after.first_material_displacement_m, shift)
        _close(after.first_application_displacement_m, shift)
        _close(after.second_material_displacement_m, np.zeros(3))
        _close(after.second_application_displacement_m, np.zeros(3))
    assert not entry.finalized and final.finalized
    (tmp_path / "friction-input-and-finish.json").write_text(
        json.dumps(_json(dict(entry=entry, candidate=candidate, finished=final)), allow_nan=False),
        encoding="utf-8",
    )


def test_both_moving_owners_use_relative_velocity_and_equal_world_moment():
    frames, pair = _inputs(dynamic_second=True)
    loads = _sample(frames, pair, dynamic_second=True)
    _close(loads.wrench_fp64[0, :3], [-5 / 4, 0, 5 / 2])
    _close(loads.wrench_fp64[:, :3].sum(axis=0), np.zeros(3))
    total_moment = sum(
        np.cross(frames[key].reference_point_m, loads.wrench_fp64[index, :3])
        + loads.wrench_fp64[index, 3:]
        for index, key in enumerate(pair)
    )
    _close(total_moment, np.zeros(3))
    assert not loads.static_reactions
    for site in loads.sites:
        _close(site.first_velocity_m_s - site.second_velocity_m_s, [1 / 4, 0, -1 / 8])


def test_negative_mu_wrong_group_and_site_capacity_are_still_rejected():
    with pytest.raises(ValueError, match="friction"):
        RobotContactMaterial(64, 4, -1 / 2, 1 / 8)
    frames, pair = _inputs()
    bad = dict(frames)
    bad[pair[0]] = replace(frames[pair[0]], group_index=1)
    with pytest.raises(ValueError, match="group"):
        _sample(bad, pair)
    with pytest.raises(ContactSiteCapacityExceeded) as caught:
        _sample(frames, pair, capacity=3)
    assert (caught.value.count, caught.value.limit) == (4, 3)
    with pytest.raises(ValueError, match="world|key|source"):
        sample_tree_contacts(
            world=WORLD + 1, frames=frames, materials_by_pair={pair: LAW},
            candidates=(pair,), group_count=1, site_capacity=4,
        )


def test_rotation_retains_distinct_material_and_common_point_work():
    # Packet transport only; no dynamics step or energy acceptance is asserted.
    frames, pair = _inputs()
    entry = _sample(frames, pair)
    original = frames[pair[0]]
    rotation = np.array([[0.0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    candidate = dict(frames)
    candidate[pair[0]] = replace(
        original,
        world_points_m=(
            original.world_points_m - original.position_m
        ) @ rotation.T + original.position_m,
        rotation=rotation,
        linear_velocity_m_s=np.zeros(3),
    )
    finished = finish_contact_loads(entry, candidate)
    # Literal mean material lever is -1/8 ez, mean application lever -7/64 ez.
    # R_y(pi/2) maps each to negative x; F=(-5/4,0,5/2) is the frozen entry force.
    _close(sum(site.material_work_j for site in finished.sites), 15 / 32)
    _close(sum(site.application_work_j for site in finished.sites), 105 / 256)
    assert finished.finalized
    for site in finished.sites:
        _close(site.first_velocity_m_s, [1 / 4, 0, -1 / 8])
