"""Material-interior rigid/cloth controls with independent face, edge and vertex oracles."""

from math import hypot

import numpy as np
import pytest

import phymeo as ph


FACE = np.array(((-2., -1., 0.), (2., -1., 0.), (0., 2., 0.)))
IDS = (19, 7, 41)


def observe(asset, center, face=FACE, *, swapped=False, ids=IDS, radius=0.0, end=None):
    first, second = ph.CollisionKey(8, 5, 2), ph.CollisionKey(8, 12, 4)
    rigid, cloth = (second, first) if swapped else (first, second)
    shape = ph.CollisionShape(rigid, asset.kind, (73,), curved=asset)
    triangle = ph.CollisionShape(cloth, "triangle", ids, radius=radius)
    model = ph.CollisionModel((triangle, shape))
    middle = ph.CollisionMotion(face, end)
    paths = {rigid: ph.CollisionMotion([center]), cloth: middle}
    with ph.CollisionWorkspace(model) as workspace:
        result = ph.detect(model, paths, ph.CollisionPolicy(margin=0.5), workspace)
    row, = result.pairs
    assert row.candidate and row.family == "RD"
    return row, rigid, cloth


@pytest.mark.parametrize("swapped", [False, True])
@pytest.mark.parametrize("gap", [1/16, 0., -1/32])
@pytest.mark.parametrize("kind, dims, quat", [
    ("ellipsoid", (0.375, 0.25, 0.125), (0, 0, 0, 1)),
    ("cylinder", (0.25, 0.375), (0, 0, 0, 1)),
    ("cone", (0.25, 0.375), (1, 0, 0, 0)),
])
def test_large_triangle_meets_finite_body_while_all_cloth_nodes_stay_clear(kind, dims, quat, gap, swapped):
    asset = ph.CurvedSolid(kind, dims, quat)
    reach = dims[-1]
    center = np.array((0.0625, 0.125, reach + gap))
    # Each node is clear even of the enclosing bounding sphere.
    assert min(np.linalg.norm(p - center) for p in FACE) > hypot(*dims[:2]) + 0.5
    row, rk, ck = observe(asset, center, FACE[::-1] if swapped else FACE, swapped=swapped,
                           ids=IDS[::-1] if swapped else IDS)
    assert row.status == ("separated" if gap > 0 else "overlap" if gap < 0 else "touching"), row.reason
    w = row.witness
    assert w and w.source == "host-convex-support" and row.sweep is None
    assert row.support_map.gap_lower_m - 1e-10 <= gap <= row.support_map.gap_upper_m + 1e-10
    cloth_first = ck == row.first
    p, coeff = (w.first_point, w.first_weights) if cloth_first else (w.second_point, w.second_weights)
    rigid_p = w.second_point if cloth_first else w.first_point
    assert {k for k,_ in coeff} == set(IDS) and all(weight > 0 for _,weight in coeff)
    pt = sum(weight * FACE[IDS.index(k)] for k, weight in coeff)
    np.testing.assert_allclose(p, pt, atol=3e-7, rtol=0)
    assert abs(p[2]) < 3e-7 and abs(rigid_p[2] - gap) < 2e-5
    assert abs(w.gap_m - gap) < 2e-5
    normal = [0, 0, 1] if cloth_first else [0, 0, -1]
    np.testing.assert_allclose(w.normal, normal, atol=3e-3, rtol=0)
    np.testing.assert_allclose(np.subtract(rigid_p,p), [0,0,gap], atol=2e-5, rtol=0)


@pytest.mark.parametrize("gap", [1/16, -1/32])
@pytest.mark.parametrize("contact, axis, owners", [
    ((0,-1,0), (0,-0.6,0.8), [(19,.5),(7,.5)]),
    ((0,2,0), (0,0.6,0.8), [(41,1.)]),
])
def test_finite_cloth_stops_at_its_own_edge_or_vertex(contact, axis, owners, gap):
    normal = np.array(axis)
    center = np.asarray(contact) + (0.25 + gap) * normal
    row,_,_ = observe(ph.CurvedSolid("ellipsoid",(.25,.25,.25)), center)
    assert row.status == ("overlap" if gap < 0 else "separated"), row.reason
    w=row.witness
    assert abs(w.gap_m-gap) < 2e-5
    np.testing.assert_allclose(w.normal,-normal,rtol=0,atol=3e-3)
    np.testing.assert_allclose(w.second_point,contact,atol=2e-4,rtol=0)
    np.testing.assert_allclose(w.first_point,center-.25*normal,atol=2e-4,rtol=0)
    got = dict(w.second_weights)
    for k, weight in owners:
        assert abs(got.get(k,0.)-weight) < 2e-4
    assert sum(v for k,v in got.items() if k not in dict(owners)) < 2e-4


def test_curved_cloth_domain_outcomes_preserve_finite_geometry():
    asset=ph.CurvedSolid("ellipsoid",(.3,.25,.125))
    center=(0.,0.,.125)
    bad = np.array(((0.,0.,0.),(1.,0.,0.),(-1.,0.,0.)))
    result,_,_=observe(asset,center,bad)
    assert result.status == "invalid_geometry" and result.witness is None
    for kwargs in (dict(radius=.0625), dict(end=FACE+np.array(((0.,0.,.1),(0,0,-.1),(0,0,0))))):
        row,_,_=observe(asset,center,**kwargs)
        assert row.status == "unsupported" and row.witness is None and row.sweep is None
