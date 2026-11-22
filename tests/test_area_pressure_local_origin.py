"""Translated I0 representation keeps world API semantics and original gates.

The portable rational fixture is the actual tick-5836 failing triangle from
repro-01.json, source SHA 5299ef56547535638d085949dac09c27a7b485e75489df885cc0b50a565d8bf1.
No saved archive, device runtime or physics step is needed by these tests.
"""

import json
from dataclasses import replace
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest

from phymeo.geometry._area_pressure_clipping import AreaPressureIntersectionError
from phymeo.geometry.area_pressure import (
    AreaPressureIdentity,
    AreaPressureStamp,
    AreaPressureSurface,
    AreaPressureTriangle,
    quadrature_area_pressure,
)
from phymeo.geometry.area_pressure_intersection import (
    RigidPressureMesh,
    TetraPressureMesh,
    _publish_triangle,
    intersect_area_pressure,
)

CASE = json.loads(r"""
{
  "exact_points": [
    [
      "70490106882344483217659021932560653442871867/140946668788467000153346362510224687655550976",
      "-39640714160310630441633378978867845567144355/9020586802461888009814167200654380009955262464",
      "-2414163834598243290457009508561247733440377/54123520814771328058885003203926280059731574784"
    ],
    [
      "7999812153341733924381038559243225535697057361/15982859014332335022854927699985423140292395008",
      "-102907537766792281801533400792118736745595275/127862872114658680182839421599883385122339160064",
      "17296128454384086594761579024756720726080209039/1022902976917269441462715372799067080978713280512"
    ],
    [
      "1333276593540866019473791430144293135342336719/2663761153497265496120004711992588070773850112",
      "-17131959941440300213967111387670964176934875/21310089227978123968960037695940704566190800896",
      "-22822447361629547439489756357962078146335637/511442141471474975255040904702576909588579221504"
    ]
  ],
  "exact_pressure": [
    "230366066830143291535489570230000/125185789546537584862792125149",
    "0",
    "0"
  ],
  "exact_gradient": [
    "-211899777946022463233233067704320000/232277763564533022796276633691",
    "-95139011561809318658748507488256000/232277763564533022796276633691",
    "0"
  ],
  "origin": [
    0.520556628704071,
    0.01619611866772175,
    0.014954747632145882
  ],
  "cell_id": "cell1",
  "face_id": "T-all-lateral-exterior-7"
}
""")
BARY = np.array(((2 / 3, 1 / 6, 1 / 6), (1 / 6, 2 / 3, 1 / 6), (1 / 6, 1 / 6, 2 / 3)))


def _identity():
    return AreaPressureIdentity("soft", "rigid", "material-a", "material-b")


def _stamp():
    return AreaPressureStamp("test", 0, 0.0, "configuration")


def _surface(triangle):
    return AreaPressureSurface(_identity(), _stamp(), (triangle,))


def _case():
    points = tuple(tuple(map(Fraction, p)) for p in CASE["exact_points"])
    pressure = tuple(map(Fraction, CASE["exact_pressure"]))
    gradient = tuple(map(Fraction, CASE["exact_gradient"]))

    def field(point):
        return pressure[0] + sum(
            g * (x - a) for g, x, a in zip(gradient, point, points[0], strict=True)
        )

    assert tuple(field(p) for p in points) == pressure
    cell = SimpleNamespace(gradient=gradient, pressure=field)
    return cell, points, np.array(pressure, dtype=float), np.array(gradient, dtype=float)


def _local_case():
    cell, points, pressure, gradient = _case()
    origin = np.asarray(CASE["origin"])
    local = np.array(
        [[float(x - Fraction(float(a))) for x, a in zip(p, origin, strict=True)] for p in points]
    )
    return local, pressure, gradient, origin


def test_actual_5836_exact_fragment_localizes_before_rounding_and_preserves_world_semantics():
    cell, points, pressure, gradient = _case()
    with pytest.raises(AreaPressureIntersectionError, match="publication_failure"):
        _publish_triangle(cell, points, CASE["cell_id"], CASE["face_id"], 0)
    local, _, _, origin = _local_case()
    triangle = _publish_triangle(
        cell, points, CASE["cell_id"], CASE["face_id"], 0, origin_world=origin
    )
    np.testing.assert_array_equal(triangle.local_vertices, local)
    np.testing.assert_array_equal(triangle.origin_world, origin)
    np.testing.assert_array_equal(triangle.vertices, local + origin)
    np.testing.assert_array_equal(triangle.vertex_pressure, pressure)
    np.testing.assert_array_equal(triangle.pressure_gradient, gradient)
    assert not np.array_equal(local, np.array(points, dtype=float) - origin)
    quad = quadrature_area_pressure(_surface(triangle))
    np.testing.assert_array_equal(quad.points, BARY @ local + origin)
    np.testing.assert_array_equal(quad.pressure, BARY @ pressure)
    for i, j in ((0, 1), (1, 2), (2, 0)):
        change = float(gradient @ (local[j] - local[i]))
        residual = abs(float(pressure[j] - pressure[i]) - change)
        bound = 1e-10 + 1e-12 * max(abs(pressure[i]), abs(pressure[j]), abs(change))
        assert residual <= bound


def test_origin_is_preserved_by_surface_revalidation_replace_and_owned_arrays():
    local, pressure, gradient, origin = _local_case()
    triangle = AreaPressureTriangle.from_local(
        "t", "c", "f", local, pressure, gradient, origin_world=origin
    )
    original = triangle.local_vertices.copy()
    local[:] = origin[:] = 999
    clone = replace(triangle, triangle_id="copy")
    for item in (triangle, clone, _surface(triangle).triangles[0]):
        np.testing.assert_array_equal(item.local_vertices, original)
        for value in (item.local_vertices, item.origin_world, item.vertices):
            assert not value.flags.writeable
            with pytest.raises(ValueError):
                value.setflags(write=True)
        np.testing.assert_array_equal(item.vertices, item.local_vertices + item.origin_world)
        quadrature_area_pressure(_surface(item))
    with pytest.raises(ValueError, match="origin|local|world"):
        replace(triangle, origin_world=triangle.origin_world + 1.0)
    wrong = triangle.vertices.copy()
    wrong[0, 0] += 1.0
    with pytest.raises(ValueError, match="origin|local|world"):
        replace(triangle, vertices=wrong)


@pytest.mark.parametrize("fault", ("boundary-pressure", "pressure", "gradient", "degenerate"))
def test_local_origin_does_not_excuse_bad_pressure_gradient_or_area(fault):
    local, pressure, gradient, origin = _local_case()
    if fault == "boundary-pressure":
        pressure[1] = 2e-10  # Same zero-pressure outer edge; original 1e-10 Pa gate.
    elif fault == "pressure":
        pressure[0] += 1.0
    elif fault == "gradient":
        gradient = -gradient
    else:
        local[1] = local[0]
    with pytest.raises(ValueError):
        AreaPressureTriangle.from_local(
            "bad", "c", "f", local, pressure, gradient, origin_world=origin
        )


@pytest.mark.parametrize("origin", ((0.5, -0.2, 0.7), (0, 0, 0)))
def test_original_absolute_pressure_gate_is_unchanged_in_translated_representation(origin):
    vertices = np.array(((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 1.0, 0.0)))
    accepted = AreaPressureTriangle.from_local(
        "t", "c", "f", vertices, (1, 2, 4 + 5e-11), (0.5, 3, 5), origin_world=origin
    )
    quadrature_area_pressure(_surface(accepted))
    with pytest.raises(ValueError, match="pressure differences"):
        AreaPressureTriangle.from_local(
            "t", "c", "f", vertices, (1, 2, 4 + 2e-10), (0.5, 3, 5), origin_world=origin
        )


def _meshes(translation):
    shift = np.asarray(translation)
    pressure = TetraPressureMesh(
        ("a", "b", "c", "d"),
        np.array(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))) + shift,
        ((0, 1, 2, 3),),
        (0.0, 0.0, 0.0, 1.0),
        ("cell",),
        pressure_domain="local",
    )
    rigid = RigidPressureMesh(
        ("r0", "r1", "r2"),
        np.array(((0.0, 0.0, 0.25), (0.5, 0.0, 0.25), (0.0, 0.5, 0.25))) + shift,
        ((0, 1, 2),),
        ("face",),
    )
    return pressure, rigid


def _wrench(quad):
    forces = (quad.dA * quad.pressure)[:, None] * quad.normals
    return np.r_[forces.sum(axis=0), np.cross(quad.points, forces).sum(axis=0)]


def test_public_intersection_translates_points_and_world_moment_but_not_pressure_or_area():
    shift = np.array((1024.0, -2048.0, 4096.0))
    a = intersect_area_pressure(
        *_meshes((0, 0, 0)), identity=_identity(), stamp=_stamp(), origin_world=(0, 0, 0)
    )
    b = intersect_area_pressure(
        *_meshes(shift), identity=_identity(), stamp=_stamp(), origin_world=shift
    )
    qa, qb = quadrature_area_pressure(a.surface), quadrature_area_pressure(b.surface)
    for field in ("dA", "normals", "pressure", "gradient"):
        np.testing.assert_array_equal(getattr(qa, field), getattr(qb, field))
    np.testing.assert_array_equal(qb.points, qa.points + shift)
    wa, wb = _wrench(qa), _wrench(qb)
    np.testing.assert_allclose(
        wb, np.r_[wa[:3], wa[3:] + np.cross(shift, wa[:3])], rtol=1e-12, atol=1e-12
    )
    legacy = intersect_area_pressure(*_meshes(shift), identity=_identity(), stamp=_stamp())
    assert all(
        t.origin_world is None and t.local_vertices is None for t in legacy.surface.triangles
    )
    old = quadrature_area_pressure(legacy.surface)
    np.testing.assert_array_equal(old.points, BARY @ legacy.surface.triangles[0].vertices)


def test_world_vertex_and_sample_collapse_still_fail_closed():
    # Local coordinates can be healthy even when the public world triangle is not.
    with pytest.raises(ValueError, match="world|area|represent"):
        AreaPressureTriangle.from_local(
            "t",
            "c",
            "f",
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
            (1, 1, 1),
            (0, 0, 1),
            origin_world=(1e300, 1e300, 0),
        )
    triangle = AreaPressureTriangle.from_local(
        "t",
        "c",
        "f",
        ((0.49, 0.49, 0), (0.51, 0.49, 0), (0.49, 0.5001, 0)),
        (1, 1, 1),
        (0, 0, 1),
        origin_world=(2.0**52, 2.0**52, 0),
    )
    assert len({tuple(x) for x in triangle.vertices}) == 3
    with pytest.raises(ValueError, match="world|sample|represent|area"):
        quadrature_area_pressure(_surface(triangle))


def test_exact_positive_sliver_is_not_laundered_by_local_validation():
    from test_area_pressure_intersection import _rigid_input, _unit_input

    small, large = 2.0**-200, 2.0**200
    pressure = TetraPressureMesh(**_unit_input(), pressure_domain="local")
    rigid = RigidPressureMesh(
        **_rigid_input(((3 / 4, small, 1 / 4), (-1 / 4, -large, 1 / 4), (1, small, 1 / 4)))
    )
    with pytest.raises(AreaPressureIntersectionError, match="publication_failure"):
        intersect_area_pressure(
            pressure, rigid, identity=_identity(), stamp=_stamp(), origin_world=(0.75, 0, 0.25)
        )


@pytest.mark.parametrize("bad", ((np.nan, 0, 0), (0, 0), (True, False, True)))
def test_public_origin_validation_does_not_depend_on_contact_existence(bad):
    with pytest.raises(ValueError, match="origin_world"):
        intersect_area_pressure(
            *_meshes((0, 0, 0)), identity=_identity(), stamp=_stamp(), origin_world=bad
        )


def test_geometry_request_owns_origin_without_changing_builder_four_tuple():
    from test_area_pressure_intersection import _rigid_input, _unit_input

    from phymeo.geometry.area_pressure_intersection import (
        _AreaPressureGeometryInput,
        _build_geometry_inputs,
    )

    pressure = dict(_unit_input(), pressure_domain="local")
    rigid = _rigid_input(((0, 0, 0.25), (0.5, 0, 0.25), (0, 0.5, 0.25)))
    origin = np.array((0.1, 0.2, 0.3))
    request = _AreaPressureGeometryInput(
        pressure, rigid, _identity(), _stamp(), origin_world=origin
    )
    origin[:] = 9
    np.testing.assert_array_equal(request.origin_world, (0.1, 0.2, 0.3))
    assert not request.origin_world.flags.writeable
    assert len(tuple(_build_geometry_inputs((request,)))[0]) == 4
