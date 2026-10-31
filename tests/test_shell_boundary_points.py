"""Full-star exact material addresses and independent flat analytic Jacobians."""

from dataclasses import replace
from fractions import Fraction as F

import numpy as np
import pytest
from test_shell_boundary import control, product_tets, save

from phymeo.constraints.conforming_shell import ShellUnresolved
from phymeo.constraints.shell_boundary import BoundaryProofLimits, ExactBoundaryAddress
from phymeo.geometry.shell_boundary import shell_exact_boundary_point
from phymeo.validation.shell_boundary import verify_shell_exact_boundary_point


def address(ref, simplex, weights):
    tets = product_tets(ref.face_ids, ref.oriented_nodes)
    reps = tuple(
        (key, tuple(weights[simplex.index(v)] if v in simplex else (F(0),) for v in vertices))
        for key, vertices in sorted(tets.items())
        if set(simplex) <= set(vertices)
    )
    return ExactBoundaryAddress(ref.reference_id, 0, tuple(simplex), tuple(weights), reps, ())


def test_complete_flat_corner_jacobian_independent_translation_rotation_and_work():
    ref, x, _, _, _ = control("C2")
    point = shell_exact_boundary_point(ref, x, address(ref, ((0, 1),), ((F(1),),)))
    expected = np.zeros((4, 3, 3), dtype=np.float64)
    expected[0] = np.eye(3)
    expected[:, 0, 2] += np.array([2, -2, -2, 2]) / 128
    expected[:, 1, 2] += np.array([1, 1, -1, -1]) / 128
    assert point.node_ids == (0, 1, 2, 3)
    assert np.array_equal(point.point.values, [-2, -1, 1 / 32])
    assert np.array_equal(point.jacobian.values, expected)
    assert np.array_equal(expected.sum(axis=0), np.eye(3))
    impulse = np.array([2, -1, 3], dtype=np.float64)
    loads = np.einsum("nji,j->ni", expected, impulse)
    assert np.array_equal(loads.sum(axis=0), impulse)
    assert np.array_equal(np.cross(x[:4], loads).sum(axis=0), np.cross([-2, -1, 1 / 32], impulse))
    virtual = np.array([[1, 2, 3], [2, 3, -1], [-3, 1, 2], [4, -1, 2]], dtype=np.float64)
    expected_velocity = np.array([1, 2, 3], dtype=np.float64) + np.array([8, -2, 0]) / 128
    assert np.array_equal(np.einsum("nij,nj->i", expected, virtual), expected_velocity)
    assert np.sum(loads * virtual) == impulse @ expected_velocity
    assert verify_shell_exact_boundary_point(
        point, ref, x, point.address, limits=BoundaryProofLimits()
    )


def test_rational_third_address_is_exact_separate_from_old_dyadic_type(tmp_path):
    ref, x, _, _, _ = control("C2")
    material = address(ref, ((0, 1), (1, 1)), ((F(1, 3),), (F(2, 3),)))
    receipt = shell_exact_boundary_point(ref, x, material)
    assert receipt.status == "exact_boundary_map_only"
    assert receipt.exact_point == ((F(-4, 3),), (F(-1),), (F(1, 32),))
    checked = verify_shell_exact_boundary_point(
        receipt, ref, x, material, limits=BoundaryProofLimits()
    )
    assert checked and checked.verification_kind == "point" and checked.created_cells == 0
    save(tmp_path / "exact-rational-third.json", (ref, x, material, receipt, checked))
    assert len(receipt.node_ids) == 4
    assert max(receipt.point.errors + receipt.jacobian.errors) < F(1, 2**40)


@pytest.mark.parametrize(
    "bad", ("missing_rep", "wrong_rep", "internal_simplex", "weight", "identity", "bool")
)
def test_invalid_material_address_cannot_be_repaired_from_world_coordinates(bad):
    ref, x, _, _, _ = control("C2")
    material = address(ref, ((0, 1),), ((F(1),),))
    if bad == "missing_rep":
        material = replace(material, representatives=material.representatives[:-1])
    elif bad == "wrong_rep":
        key, weights = material.representatives[0]
        material = replace(
            material, representatives=((key, weights[::-1]),) + material.representatives[1:]
        )
    elif bad == "internal_simplex":
        t = next(iter(product_tets(ref.face_ids, ref.oriented_nodes).values()))
        material = address(ref, tuple(sorted(t)), ((F(1, 4),),) * 4)
    elif bad == "weight":
        material = replace(material, weights=((F(2),),))
    elif bad == "identity":
        material = replace(material, reference_id="0" * 64)
    else:
        with pytest.raises((ValueError, TypeError)):
            replace(material, generation=True)
        return
    with pytest.raises((ShellUnresolved, ValueError, TypeError)):
        shell_exact_boundary_point(ref, x, material)


def test_query_receipt_owns_inputs_and_verifier_rechecks_all_neighbors():
    ref, x, _, _, _ = control("C2")
    material = address(ref, ((0, 1),), ((F(1),),))
    receipt = shell_exact_boundary_point(ref, x, material)
    x[:] = 100
    assert np.array_equal(receipt.point.values, [-2, -1, 1 / 32])
    with pytest.raises(ValueError):
        receipt.jacobian.values.flags.writeable = True
    bad = replace(receipt, node_ids=receipt.node_ids[:-1])
    _, original, _, _, _ = control("C2")
    assert not verify_shell_exact_boundary_point(
        bad, ref, original, material, limits=BoundaryProofLimits()
    )
