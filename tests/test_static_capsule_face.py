"""Exact material equivalence between a single pose and the original affine capsule/face query."""

import json
from dataclasses import asdict
from fractions import Fraction as F

import numpy as np
import pytest
from test_cloth_link_contact import make_scene

import phymeo as ph
from phymeo.geometry import _capsule_triangle_sweep as original

FACE = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
AXES = (
    ((-1, 0.5, 0.25), (3, 0.5, 0.25)),  # projected interval clipped at two distinct edges
    ((0.25, 0.5, 0.25), (0.5, 1, 0.25)),  # whole shaft has the same distance
    ((-4, -0.5, 0.25), (-1, -0.5, 0.25)),  # strict closest corner
    ((-1, 0.5, -0.5), (1, 0.5, 0.5)),  # open-face piercing
    ((0.5, -1, 0.5), (0.5, 1, 1.5)),  # skew edge/shaft interiors
    ((-1, 0, 0), (3, 0, 0)),  # exactly coplanar with a shared line
    ((0.5, 0.5, 0.125), (0.5, 0.5, 0.125)),  # coincident ends tangent to the face
    ((3, 2, 0), (3, 2, 0)),  # coincident ends outside the triangle
    ((0.25, 0.25, 0.125), (0.5, 0.25, np.nextafter(0.125, 1.0))),
)


def registered(axis, face, reverse=False, end=None):
    cap = ph.CollisionShape(ph.CollisionKey(4, 7, 0), "capsule", (91, 23), 0.125)
    tri = ph.CollisionShape(ph.CollisionKey(4, 2, 0), "triangle", (23, 59, 17))
    mc, mf = ph.CollisionMotion(axis), ph.CollisionMotion(face, end)
    return (tri, cap, mf, mc) if reverse else (cap, tri, mc, mf)


def both(args):
    from phymeo.geometry._static_capsule_face import static_capsule_face_sample

    old = original.capsule_face_query(*args)
    expected = None if old is None else old.sample(0)
    result = static_capsule_face_sample(*args)
    if expected is not None:
        assert result[1] == expected[1] and type(result[1]) is type(expected[1])
        assert json.dumps(asdict(result[0]), sort_keys=True) == json.dumps(
            asdict(expected[0]), sort_keys=True
        )
    else:
        assert result is None
    return result


@pytest.mark.parametrize(
    "axis",
    AXES,
    ids=(
        "clipped",
        "parallel",
        "corner",
        "pierce",
        "skew_edge",
        "collinear",
        "point_tangent",
        "point_beyond",
        "one_ulp_tilt",
    ),
)
@pytest.mark.parametrize("reverse", (False, True))
def test_static_extremes_and_original_material_ids_are_identical(axis, reverse):
    both(registered(axis, FACE, reverse))


def test_known_shaft_and_face_addresses_not_just_the_gap():
    w, signed = both(registered(AXES[0], FACE))
    assert signed == F(3, 64) and w.gap_m == 0.125 and w.normal == (0.0, 0.0, -1.0)
    start, finish = w.capsule_axis_span
    assert (start.axis_parameter, finish.axis_parameter) == (0.25, 0.625)
    assert dict(start.face_weights) == {23: 0.75, 59: 0.0, 17: 0.25}
    assert dict(finish.face_weights) == {23: 0.0, 59: 0.75, 17: 0.25}


def test_off_axis_scales_storage_permutations_and_distinct_consecutive_poses():
    rng = np.random.default_rng(10473)
    for k in range(54):
        points = np.ldexp(rng.integers(-24, 25, (5, 3)).astype(float), (k % 11) - 5)
        # A large common shift must not change the exact Voronoi region or addresses.
        if k % 4 == 0:
            points += np.array([2.0**38, -(2.0**31), 2.0**29])
        face = points[:3][np.roll(np.arange(3), k % 3)]
        if np.all(np.cross(face[1] - face[0], face[2] - face[0]) == 0):
            continue
        both(registered(points[3:][:: -1 if k % 2 else 1], face, reverse=bool(k % 2)))


def test_intrinsic_degeneracy_and_an_interior_time_collapse_keep_the_original_failure():
    from phymeo.geometry._static_capsule_face import static_capsule_face_sample

    flat = np.array([[0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0]])
    for args in (registered(AXES[0], flat), registered(AXES[0], FACE, end=FACE[[0, 2, 1]])):
        with pytest.raises(ph.SolverFailure, match="triangle_degenerate_on_path"):
            original.capsule_face_query(*args).sample(0)
        with pytest.raises(ph.SolverFailure, match="triangle_degenerate_on_path"):
            static_capsule_face_sample(*args)
    cap, tri, mc, mf = registered(AXES[0], FACE)
    turn = ph.CollisionMotion(mc.start, turn=ph.RigidTurn((0, 0, 0), (0, 0, 0), (0, 0, 1), 0.25))
    both((cap, tri, turn, mf))
    rounded = ph.CollisionShape(tri.key, "triangle", tri.vertex_ids, 0.125)
    both((cap, rounded, mc, mf))


def test_public_cloth_load_uses_the_pose_only_route(monkeypatch):
    program, state, inputs = make_scene()

    def unexpected(*args):
        pytest.fail("single-pose contact rebuilt an affine triangle path")

    monkeypatch.setattr(original, "_FacePath", unexpected)
    _, _, _, observed = program.accelerations(state, inputs)
    assert sum(row["normal_force_n"] for row in observed["cloth_links"]) == pytest.approx(2.0)
