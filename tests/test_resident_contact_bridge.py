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
            fixed, support, None, np.zeros(3), np.eye(3), support.mean(axis=0),
            np.zeros(3), np.zeros(3),
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
