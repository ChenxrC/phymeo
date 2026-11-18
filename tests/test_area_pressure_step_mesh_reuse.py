"""Each checked side can be shared; all contact owners keep their own publication."""

from dataclasses import replace

import numpy as np
import pytest
from test_area_pressure_intersection import _case, _identity, _stamp

import phymeo.geometry.area_pressure_intersection as geom
from phymeo.geometry.area_pressure import _identity as copy_identity
from phymeo.geometry.area_pressure import _stamp as copy_stamp


def uncached_inputs(requests):
    """Reproduce the immediately preceding HOST preparation for paired benchmarks."""
    for r in requests:
        assert type(r) is geom._AreaPressureGeometryInput
        identity, stamp = copy_identity(r.identity), copy_stamp(r.stamp)
        pressure = geom.TetraPressureMesh(**geom._raw_mesh_fields(r.pressure, pressure=True))
        rigid = geom.RigidPressureMesh(**geom._raw_mesh_fields(r.rigid, pressure=False))
        yield pressure, rigid, identity, stamp


def _inputs():
    p, r, domain, *_ = _case("full_body")
    return dict(p, pressure_domain=domain), r


@pytest.mark.parametrize(
    "side,field",
    [
        ("pressure", "node_ids"),
        ("pressure", "cell_ids"),
        ("pressure", "cells"),
        ("pressure", "world_vertices"),
        ("pressure", "signed_zero"),
        ("pressure", "pressure_domain"),
        ("pressure", "nodal_pressure"),
        ("pressure", "body"),
        ("pressure", "material"),
        ("rigid", "node_ids"),
        ("rigid", "face_ids"),
        ("rigid", "faces"),
        ("rigid", "world_vertices"),
        ("rigid", "body"),
        ("rigid", "material"),
    ],
)
def test_each_side_distinguishes_all_mesh_inputs(side, field):
    p, r = _inputs()
    identity, stamp = _identity(), _stamp()
    first = geom._AreaPressureGeometryInput(p, r, identity, stamp)
    chosen = p if side == "pressure" else r
    if field == "signed_zero":
        coords = np.array(chosen["world_vertices"], dtype=float)
        coords[0, 0] = -0.0
        chosen["world_vertices"] = coords
    elif field in ("node_ids", "cell_ids", "face_ids"):
        ids = chosen[field]
        chosen[field] = (*ids[:-1], "renamed")
    elif field in ("cells", "faces"):
        indices = np.asarray(chosen[field]).copy()
        indices[0] = np.roll(indices[0], 1)
        chosen[field] = indices
    elif field == "world_vertices":
        chosen[field] = np.asarray(chosen[field]) + [1.0, 0.0, 0.0]
    elif field == "pressure_domain":
        chosen[field] = "local"
    elif field == "nodal_pressure":
        chosen[field] = (*chosen[field][:-1], 2.0)
    else:
        prefix = "compliant" if side == "pressure" else "rigid"
        identity = replace(identity, **{f"{prefix}_{field}": "renamed"})
    other_stamp = replace(stamp, world_id="other")
    second = geom._AreaPressureGeometryInput(p, r, identity, other_stamp)
    a, b = geom._build_geometry_inputs((first, second))
    assert (a[0] is b[0]) == (side != "pressure")
    assert (a[1] is b[1]) == (side != "rigid")
    assert b[2:] == (identity, other_stamp)
    (expected,) = uncached_inputs((second,))
    for mesh, rebuilt, fields in (
        (b[0], expected[0], second.pressure),
        (b[1], expected[1], second.rigid),
    ):
        for name in fields:
            value, original = getattr(mesh, name), getattr(rebuilt, name)
            if isinstance(value, np.ndarray):
                assert (value.dtype, value.shape, value.tobytes()) == (
                    original.dtype,
                    original.shape,
                    original.tobytes(),
                )
            else:
                assert value == original


def test_hit_preserves_metadata_and_failure_expires_the_batch():
    p, r = _inputs()
    a = geom._AreaPressureGeometryInput(p, r, _identity(), _stamp())
    b = geom._AreaPressureGeometryInput(p, r, _identity(), replace(_stamp(), step_index=12))
    bad = geom._AreaPressureGeometryInput(
        dict(p, nodal_pressure=(2, 0, 0, 0, 1)), r, _identity(), _stamp()
    )
    iterator = geom._build_geometry_inputs((a, b, bad))
    first, hit = next(iterator), next(iterator)
    assert first[0] is hit[0] and first[1] is hit[1]
    assert first[3] != hit[3]
    with pytest.raises(geom.AreaPressureIntersectionError, match="zero boundary pressure"):
        next(iterator)
    retry, repeat = geom._build_geometry_inputs((a, b))
    assert retry[0] is not first[0] and retry[1] is not first[1]
    assert repeat[0] is retry[0] and repeat[1] is retry[1]


@pytest.mark.parametrize(
    "field,detail",
    [("identity", "AreaPressureIdentity"), ("stamp", "AreaPressureStamp"), ("shape", "shape")],
)
def test_malformed_request_is_rejected_after_a_confirmed_hit(field, detail):
    p, r = _inputs()
    first, second, third = (
        geom._AreaPressureGeometryInput(p, r, _identity(), _stamp()) for _ in range(3)
    )
    if field == "shape":
        vertices = third.pressure["world_vertices"]
        object.__setattr__(
            third, "pressure", dict(third.pressure, world_vertices=vertices.reshape(-1, 1))
        )
    else:
        object.__setattr__(third, field, None)
    batch = geom._build_geometry_inputs((first, second, third))
    a, b = next(batch), next(batch)
    assert a[0] is b[0] and a[1] is b[1]
    with pytest.raises(ValueError, match=detail):
        next(batch)
