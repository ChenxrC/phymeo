"""Independent geometry oracles for the frozen I0 area-pressure port.

These tests consume prescribed triangles only. They do not certify clipping,
volume-mesh continuity, a collision producer, or a dynamic contact solve.
"""

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from phymeo.geometry.area_pressure import (
    AreaPressureIdentity,
    AreaPressureStamp,
    AreaPressureSurface,
    AreaPressureTriangle,
    quadrature_area_pressure,
)


RTOL = ATOL = 1e-12


def _identity():
    return AreaPressureIdentity("soft-body", "rigid-body", "soft-material", "rigid-material")


def _stamp():
    return AreaPressureStamp("world-3", 7, 0.125, "accepted-configuration-7")


def _triangle(**changes):
    values = {
        "triangle_id": "triangle-0",
        "compliant_cell_id": "cell-4",
        "rigid_face_id": "face-9",
        "vertices": ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        "vertex_pressure": (1.0, 2.0, 4.0),
        "pressure_gradient": (0.5, 3.0, 5.0),
    }
    values.update(changes)
    return AreaPressureTriangle(**values)


def _surface(triangles=None):
    if triangles is None:
        triangles = (_triangle(),)
    return AreaPressureSurface(_identity(), _stamp(), tuple(triangles))


def _wrench(quadrature):
    # Independent per-point force/cross-product integration, not a port helper.
    force = quadrature.dA[:, None] * quadrature.pressure[:, None] * quadrature.normals
    return np.r_[force.sum(axis=0), np.cross(quadrature.points, force).sum(axis=0)]


def _assert_bytes_backed(array):
    assert isinstance(array, np.ndarray)
    assert array.dtype == np.dtype(np.float64)
    owner = array
    while isinstance(owner, np.ndarray):
        assert not owner.flags.writeable
        with pytest.raises(ValueError):
            owner.setflags(write=True)
        owner = owner.base
    while isinstance(owner, memoryview):
        assert owner.readonly
        owner = owner.obj
    assert isinstance(owner, bytes)
    if array.size:
        with pytest.raises(ValueError):
            array.flat[0] = 0.0


def test_linear_pressure_triangle_has_true_area_and_exact_first_moment():
    surface = _surface()
    quad = quadrature_area_pressure(surface)
    # Integral x^i y^j over this triangle is 2^(i+1) i! j! / (i+j+2)!.
    # For p=1+x/2+3y: int p=7/3, int y*p=11/12, int x*p=3/2.
    np.testing.assert_allclose(
        quad.points, ((1 / 3, 1 / 6, 0), (4 / 3, 1 / 6, 0), (1 / 3, 2 / 3, 0)),
        rtol=RTOL, atol=ATOL,
    )
    np.testing.assert_allclose(quad.normals, ((0, 0, 1),) * 3, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(quad.dA, (1 / 3,) * 3, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(quad.pressure, (5 / 3, 13 / 6, 19 / 6), rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(quad.gradient, (5.0,) * 3, rtol=RTOL, atol=ATOL)
    assert quad.total_area_m2 == pytest.approx(1.0, rel=RTOL, abs=ATOL)
    np.testing.assert_allclose(_wrench(quad), (0, 0, 7 / 3, 11 / 12, -3 / 2, 0),
                               rtol=RTOL, atol=ATOL)
    assert quad.identity == surface.identity and quad.stamp == surface.stamp
    assert quad.sample_ids == tuple(("triangle-0", slot) for slot in range(3))
    assert quad.source_ids == (("cell-4", "face-9"),) * 3
    # A geometric-centroid force would give (7/9, -14/9), which is not this moment.
    assert not np.allclose(_wrench(quad)[3:5], (7 / 9, -14 / 9), rtol=RTOL, atol=ATOL)


def test_subdivision_and_cyclic_vertices_preserve_linear_pressure_integral():
    vertices = np.array(((0, 0, 0), (2, 0, 0), (0, 1, 0), (1, 0, 0), (1, 0.5, 0),
                         (0, 0.5, 0)), dtype=float)
    faces = ((0, 3, 5), (3, 1, 4), (5, 4, 2), (3, 4, 5))
    triangles = []
    for index, face in enumerate(faces):
        points = vertices[list(face)]
        pressure = 1 + points[:, 0] / 2 + 3 * points[:, 1]
        triangles.append(_triangle(triangle_id=f"sub-{index}", vertices=points,
                                   vertex_pressure=pressure))
    quad = quadrature_area_pressure(_surface(triangles))
    assert quad.total_area_m2 == pytest.approx(1.0, rel=RTOL, abs=ATOL)
    np.testing.assert_allclose(_wrench(quad), (0, 0, 7 / 3, 11 / 12, -3 / 2, 0),
                               rtol=RTOL, atol=ATOL)
    cyclic = _triangle(vertices=((2, 0, 0), (0, 1, 0), (0, 0, 0)),
                       vertex_pressure=(2, 4, 1))
    np.testing.assert_allclose(_wrench(quadrature_area_pressure(_surface((cyclic,)))),
                               (0, 0, 7 / 3, 11 / 12, -3 / 2, 0), rtol=RTOL, atol=ATOL)
    reordered = quadrature_area_pressure(_surface(tuple(reversed(triangles))))
    np.testing.assert_allclose(_wrench(reordered), _wrench(quad), rtol=RTOL, atol=ATOL)
    assert reordered.sample_ids[0] == ("sub-3", 0)


def test_area_is_not_normalized_and_scales_with_geometric_size():
    # Uniform pressure and the same gradient make geometric scaling independent
    # of an unrelated pressure change: area x4, force x4, first moment x8.
    small = _triangle(vertex_pressure=(6, 6, 6), pressure_gradient=(0, 0, 5))
    large = _triangle(vertices=((0, 0, 0), (4, 0, 0), (0, 2, 0)),
                      vertex_pressure=(6, 6, 6), pressure_gradient=(0, 0, 5))
    q_small = quadrature_area_pressure(_surface((small,)))
    q_large = quadrature_area_pressure(_surface((large,)))
    assert q_large.total_area_m2 == pytest.approx(4.0, rel=RTOL, abs=ATOL)
    np.testing.assert_allclose(q_large.dA, (4 / 3,) * 3, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(_wrench(q_small), (0, 0, 6, 2, -4, 0), rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(_wrench(q_large), (0, 0, 24, 16, -32, 0), rtol=RTOL, atol=ATOL)


def test_rigid_rotation_and_translation_preserve_area_and_transform_wrench():
    # Rational proper rotation: Rx(arccos(3/5)) Ry(arccos(12/13)).
    rotation = np.array(((12 / 13, 0, 5 / 13), (4 / 13, 3 / 5, -48 / 65),
                         (-3 / 13, 4 / 5, 36 / 65)))
    translation = np.array((0.7, -1.1, 0.4))
    vertices = np.array(((0, 0, 0), (2, 0, 0), (0, 1, 0)))
    triangle = _triangle(vertices=vertices @ rotation.T + translation,
                         pressure_gradient=rotation @ np.array((0.5, 3.0, 5.0)))
    quad = quadrature_area_pressure(_surface((triangle,)))
    expected_force = rotation @ np.array((0, 0, 7 / 3))
    expected_moment = rotation @ np.array((11 / 12, -3 / 2, 0))
    expected_moment += np.cross(translation, expected_force)
    assert quad.total_area_m2 == pytest.approx(1.0, rel=RTOL, abs=ATOL)
    np.testing.assert_allclose(quad.gradient, 5.0, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(quad.normals, np.tile(rotation[:, 2], (3, 1)),
                               rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(_wrench(quad), np.r_[expected_force, expected_moment],
                               rtol=RTOL, atol=ATOL)
    assert not np.isclose(quad.total_area_m2 * abs(rotation[2, 2]), quad.total_area_m2)


def test_non_coplanar_surfaces_keep_normals_and_a_true_pure_couple():
    # Both yz triangles have area one. Their centroids are (0,+/-1,0),
    # normals are +/-x and p=3: total F=0, total Mz=-6.
    positive = _triangle(
        triangle_id="right", vertices=((0, 1 / 3, -1 / 3), (0, 7 / 3, -1 / 3),
                                       (0, 1 / 3, 2 / 3)),
        vertex_pressure=(3, 3, 3), pressure_gradient=(5, 0, 0),
    )
    negative = _triangle(
        triangle_id="left", vertices=((0, -5 / 3, -1 / 3), (0, -5 / 3, 2 / 3),
                                      (0, 1 / 3, -1 / 3)),
        vertex_pressure=(3, 3, 3), pressure_gradient=(-5, 0, 0),
    )
    # A third tilted patch makes the surface genuinely non-coplanar; integrate
    # it separately against the same closed-form triangle oracle.
    tilt = np.array(((0, 1, 0), (0, 0, 1), (1, 0, 0)))
    third = _triangle(triangle_id="third", vertices=np.array(((0, 0, 0), (2, 0, 0),
                                                            (0, 1, 0))) @ tilt.T,
                      pressure_gradient=tilt @ np.array((0.5, 3, 5)))
    couple = quadrature_area_pressure(_surface((positive, negative)))
    np.testing.assert_allclose(_wrench(couple), (0, 0, 0, 0, 0, -6), rtol=RTOL, atol=ATOL)
    both = quadrature_area_pressure(_surface((positive, negative, third)))
    expected = np.r_[tilt @ np.array((0, 0, 7 / 3)),
                     tilt @ np.array((11 / 12, -3 / 2, 0)) + (0, 0, -6)]
    np.testing.assert_allclose(_wrench(both), expected, rtol=RTOL, atol=ATOL)
    assert both.total_area_m2 == pytest.approx(3.0, rel=RTOL, abs=ATOL)


def test_zero_pressure_and_empty_contact_have_distinct_areas_and_exact_shapes():
    zero = _triangle(vertex_pressure=(0, 0, 0), pressure_gradient=(0, 0, 5))
    quad = quadrature_area_pressure(_surface((zero,)))
    assert quad.total_area_m2 == 1.0
    np.testing.assert_array_equal(quad.pressure, np.zeros(3))
    np.testing.assert_array_equal(_wrench(quad), np.zeros(6))
    empty = quadrature_area_pressure(_surface(()))
    assert empty.total_area_m2 == 0.0
    assert empty.sample_ids == () and empty.source_ids == ()
    assert empty.points.shape == empty.normals.shape == (0, 3)
    for name in ("dA", "pressure", "gradient"):
        assert getattr(empty, name).shape == (0,)
    assert empty.identity == _identity() and empty.stamp == _stamp()


def test_triangle_and_quadrature_own_truly_immutable_copies():
    vertices = np.array(((0, 0, 0), (2, 0, 0), (0, 1, 0)), dtype=float)
    pressure = np.array((1, 2, 4), dtype=float)
    gradient = np.array((0.5, 3, 5), dtype=float)
    original = [array.copy() for array in (vertices, pressure, gradient)]
    triangle = _triangle(vertices=vertices, vertex_pressure=pressure, pressure_gradient=gradient)
    vertices[:] = pressure[:] = gradient[:] = -999
    for name, expected in zip(("vertices", "vertex_pressure", "pressure_gradient"), original):
        array = getattr(triangle, name)
        np.testing.assert_array_equal(array, expected)
        _assert_bytes_backed(array)
    surface = _surface((triangle,))
    quad = quadrature_area_pressure(surface)
    for name in ("points", "normals", "dA", "pressure", "gradient"):
        _assert_bytes_backed(getattr(quad, name))
    for obj, name, value in ((triangle, "triangle_id", "new"), (surface, "triangles", ()),
                             (quad, "total_area_m2", 99), (surface.stamp, "step_index", 8),
                             (surface.identity, "rigid_body", "other")):
        with pytest.raises(FrozenInstanceError):
            setattr(obj, name, value)


@pytest.mark.parametrize("changes", [
    {"vertices": ((0, 0, 0), (1, 0, 0), (2, 0, 0))},
    {"vertices": ((0, 0, 0), (np.nan, 0, 0), (0, 1, 0))},
    {"vertex_pressure": (1, -2, 4)},
    {"vertex_pressure": (1, np.inf, 4)},
    {"vertex_pressure": (1, 2)},
    {"vertex_pressure": (1, 2, 4.000001)},
    {"pressure_gradient": (0.5, 3, 0)},
    {"pressure_gradient": (0.5, 3, -5)},
    {"pressure_gradient": (0.5, 3)},
    {"triangle_id": ""},
])
def test_invalid_triangle_is_rejected_wholly(changes):
    with pytest.raises(ValueError):
        quadrature_area_pressure(_surface((_triangle(**changes),)))


def test_pressure_consistency_uses_the_frozen_absolute_residual_budget():
    accepted = _triangle(vertex_pressure=(1, 2, 4 + 5e-11))
    quadrature_area_pressure(_surface((accepted,)))
    with pytest.raises(ValueError):
        quadrature_area_pressure(_surface((_triangle(vertex_pressure=(1, 2, 4 + 2e-10)),)))


def test_reversed_winding_requires_reversing_the_normal_pressure_gradient():
    reverse = ((0, 0, 0), (0, 1, 0), (2, 0, 0))
    with pytest.raises(ValueError):
        quadrature_area_pressure(_surface((_triangle(vertices=reverse,
                                                     vertex_pressure=(1, 4, 2)),)))
    valid = _triangle(vertices=reverse, vertex_pressure=(1, 4, 2),
                      pressure_gradient=(0.5, 3, -5))
    quad = quadrature_area_pressure(_surface((valid,)))
    np.testing.assert_allclose(_wrench(quad), (0, 0, -7 / 3, -11 / 12, 3 / 2, 0),
                               rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("changes", [
    {"world_id": ""}, {"step_index": -1},
    {"step_index": True}, {"step_index": 1.5}, {"sample_time_s": -0.1},
    {"sample_time_s": np.inf},
])
def test_stamp_rejects_invalid_binding(changes):
    values = dict(world_id="world", step_index=0, sample_time_s=0, configuration_id="initial")
    values.update(changes)
    with pytest.raises(ValueError):
        AreaPressureStamp(**values)


def test_body_material_ids_duplicates_and_triangle_capacity():
    for values in (("body", "body", "a", "b"), ("", "body", "a", "b"),
                   ("a", "b", "", "b"), ("a", "b", "a", "b" * 129)):
        with pytest.raises(ValueError):
            AreaPressureIdentity(*values)
    with pytest.raises(ValueError):
        quadrature_area_pressure(_surface((_triangle(), _triangle())))
    triangles = tuple(_triangle(triangle_id=f"triangle-{index}") for index in range(1024))
    quad = quadrature_area_pressure(_surface(triangles))
    assert quad.points.shape == (3072, 3)
    assert quad.total_area_m2 == pytest.approx(1024.0, rel=RTOL, abs=ATOL)
    with pytest.raises(ValueError):
        quadrature_area_pressure(_surface(triangles + (_triangle(triangle_id="extra"),)))
    # Geometric overlap is intentionally not an I0 collision certification gate.
    assert quad.source_ids[0] == quad.source_ids[-1] == ("cell-4", "face-9")


def test_derived_overflow_and_malformed_surface_do_not_mutate_valid_input():
    triangle = _triangle()
    before = triangle.vertices.tobytes(), triangle.vertex_pressure.tobytes()
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(ValueError):
            quadrature_area_pressure(_surface((_triangle(
                vertices=((0, 0, 0), (1e308, 0, 0), (0, 1e308, 0)),
                vertex_pressure=(1, 1, 1), pressure_gradient=(0, 0, 5),
            ),)))
    with pytest.raises(ValueError):
        quadrature_area_pressure(replace(_surface(), identity=None))
    with pytest.raises(ValueError):
        quadrature_area_pressure(_surface((triangle, "not-a-triangle")))
    assert before == (triangle.vertices.tobytes(), triangle.vertex_pressure.tobytes())
    np.testing.assert_allclose(_wrench(quadrature_area_pressure(_surface((triangle,)))),
                               (0, 0, 7 / 3, 11 / 12, -3 / 2, 0), rtol=RTOL, atol=ATOL)
