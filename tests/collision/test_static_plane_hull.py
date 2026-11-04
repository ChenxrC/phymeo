"""Independent half-space supports through ordinary detect and finite material impulse rows."""

from dataclasses import asdict
from itertools import product
import json

import numpy as np
import pytest

import phymeo as ph

CORNERS = np.array(list(product((-0.5, 0.5), repeat=3)))
Q = np.array([0.0, 0.0, 0.0, 1.0])


def fixture(normal=(0,0,1), *, lift=0.0, reverse=False, transformed=False, hull_radius=0.0):
    plane, hull = (ph.CollisionKey(5, int(reverse), 0), ph.CollisionKey(5, int(not reverse), 0))
    frame = np.array(((0,0,1),(1,0,0),(0,1,0))) if transformed else np.eye(3)
    origin = np.array([2.25,-1.5,0.375]) if transformed else np.zeros(3)
    canonical = np.vstack((CORNERS, [0.0,0.0,-0.5], [0.0,0.0,0.0], [0.0,-0.5,-0.5]))
    ids = tuple(37+7*i for i in range(len(canonical)))
    order = list(reversed(range(len(ids)))) if reverse else list(range(len(ids)))
    n = np.asarray(normal, dtype=float)
    anchor = CORNERS[0].copy()
    if tuple(normal) == (0,0,1):
        anchor += [1.25,-0.75,0]
    center = np.array([0.0,0.0,lift])
    positions = (canonical+center) @ frame.T + origin
    plane_anchor = frame @ anchor + origin
    shapes = ph.CollisionModel((
        ph.CollisionShape(plane,"plane",(37,),normal=tuple(frame@n)),
        ph.CollisionShape(hull,"hull",tuple(ids[i] for i in order),radius=hull_radius),
    ))
    paths = {plane:ph.CollisionMotion([plane_anchor]),hull:ph.CollisionMotion(positions[order])}
    n = frame @ (n/np.linalg.norm(n))
    projected = CORNERS @ np.asarray(normal,dtype=float)
    extreme = np.flatnonzero(projected==projected.min())
    return shapes, paths, plane, hull, n, {ids[i]:positions[i] for i in extreme}


def detect(model,paths,*,backend="host",cache_root=None):
    with ph.CollisionWorkspace(model,backend,cache_root=cache_root) as work:
        result=ph.detect(model,paths,ph.CollisionPolicy(margin=2),work)
        return result.pairs[0], work.stats()


@pytest.mark.parametrize("normal,dim,lift", [
    ((0,0,1),2,0.0),((0,0,1),2,0.125),((0,0,1),2,-0.25),
    ((3,0,4),1,0.0),((1,2,3),0,0.0)])
@pytest.mark.parametrize("reverse,transformed",[(False,False),(True,True)])
def test_signed_halfspace_gap_and_all_actual_support_materials(normal,dim,lift,reverse,transformed,tmp_path):
    model,paths,plane,hull,n,extreme=fixture(normal,lift=lift,reverse=reverse,transformed=transformed)
    row,_=detect(model,paths)
    assert row.status==("separated" if lift>0 else "overlap" if lift<0 else "touching"),row
    assert row.candidate and row.family=="RR" and row.sweep is None
    assert row.witness.gap_m==pytest.approx(lift,abs=1e-14)
    (patch,)=row.convex.end.patches
    sign=1 if row.first==plane else -1
    np.testing.assert_allclose(patch.normal,sign*n,atol=1e-14,rtol=0)
    assert patch.dimension==dim and patch.area_m2==pytest.approx(1.0 if dim==2 else 0)
    expected=dict(extreme)
    for corner in patch.points:
        pp,hp,pw,hw,offset=corner.first_point,corner.second_point,corner.first_weights,corner.second_weights,corner.first_offset
        if row.first!=plane:
            pp,hp,pw,hw,offset=corner.second_point,corner.first_point,corner.second_weights,corner.first_weights,corner.second_offset
        assert pw==((37,1.0),) and len(hw)==1 and hw[0][1]==1
        np.testing.assert_allclose(expected.pop(hw[0][0]),hp,atol=1e-14,rtol=0)
        np.testing.assert_allclose(np.asarray(hp)-lift*n,pp,atol=1e-14,rtol=0)
        np.testing.assert_allclose(paths[plane].start[0]+offset,pp,atol=1e-14,rtol=0)
        assert abs(n@offset)<1e-13
    assert not expected
    (tmp_path/"plane-hull.json").write_text(json.dumps(asdict(row)),encoding="utf8")


def test_motion_and_rounded_hulls_do_not_publish_a_partial_static_result():
    model,paths,_,hull,_,_=fixture()
    p=paths[hull].start
    paths[hull]=ph.CollisionMotion(p,p+[0,0,0.125])
    row,_=detect(model,paths)
    assert row.status=="unsupported" and row.sweep is None and row.witness is None
    model,paths,*_=fixture(hull_radius=0.125)
    assert detect(model,paths)[0].status=="unsupported"


def test_unrepresentable_anchor_offset_cannot_claim_the_plane_material_point():
    model,paths,plane,_,_,_=fixture()
    paths[plane]=ph.CollisionMotion([[float(2**54),0,-0.5]])
    row,_=detect(model,paths)
    assert row.status=="unresolved" and row.witness is None


def test_polygon_carries_eccentric_gravity_without_false_roll_and_allows_separation(tmp_path):
    model,paths,plane,hull,_,_=fixture()
    pair,_=detect(model,paths)
    assert pair.status=="touching"
    patch=pair.convex.end.patches[0]
    xy=np.array([p.first_point for p in patch.points])
    ids=np.zeros(4,dtype=np.int64)
    body=ph.RigidBodyModel(np.ones(1),np.ones((1,3)),np.array([0.0,0.0,-8.0]))
    arm=ph.FloatingRobotModel(body,np.array([0.1875,0.0,0.0]))
    state=ph.FloatingRobotState(np.r_[np.zeros(3),Q],np.zeros(6))
    inputs=ph.FloatingRobotInputs(np.empty(0),np.zeros((1,3)),np.zeros((1,3)))
    floor=ph.RigidBodyModel(np.zeros(1),np.zeros((1,3)),np.zeros(3))
    resting=ph.RigidBodyState(np.zeros((1,3)),Q[None,:],np.zeros((1,3)),np.zeros((1,3)))
    rows=[(pair,{hull:("box",i),plane:("floor",i)},i) for i in range(4)]
    attachments={"box":(ids,xy),"floor":(ids,xy)}
    h=1/64
    def step(start, ordered):
        return ph.step_floating_contacts({"box":(arm,start,inputs)},attachments,ordered,
            [0.4]*len(ordered),[0.3]*len(ordered),h,fixed_bodies={"floor":(floor,resting)},
            local_vertices={hull:paths[hull].start, plane:paths[plane].start},local_normals={plane:(0,0,1)})
    out=step(state,rows)
    assert out.report.accepted,out.report.reason
    j=out.impulses.world_impulses
    np.testing.assert_allclose(j.sum(0),[0,0,8*h],atol=2e-9,rtol=0)
    np.testing.assert_allclose(np.cross(xy-arm.root_com_m,j).sum(0),np.zeros(3),atol=2e-9,rtol=0)
    assert (xy[:,0]@j[:,2])/j[:,2].sum()==pytest.approx(0.1875,abs=2e-8)
    np.testing.assert_allclose(out.states["box"].v,np.zeros(6),atol=2e-9,rtol=0)
    incomplete=step(state,rows[:-1])
    assert not incomplete.report.accepted and incomplete.impulses is None
    upward=ph.FloatingRobotState(state.q,np.array([0,0,1,0,0,0],dtype=float))
    exited=step(upward,rows)
    assert exited.report.accepted and not np.any(exited.impulses.world_impulses)
    assert exited.states["box"].q[2]>0
    (tmp_path/"eccentric.json").write_text(json.dumps({"world_impulses":j.tolist(),
        "center_of_pressure_x_m":(xy[:,0]@j[:,2])/j[:,2].sum(),"remaining_v":out.states["box"].v.tolist()}),encoding="utf8")
