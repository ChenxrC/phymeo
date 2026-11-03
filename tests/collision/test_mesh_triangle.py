"""Exact closed-solid sections of a complete finite cloth triangle on the public RD path."""

import json
import runpy
from pathlib import Path

import numpy as np
import pytest

import phymeo as ph
from mesh_triangle_oracles import clipped_fractions
from test_mesh_mesh_contact import box

ROOT=Path(__file__).resolve().parents[1]/"reference_newton"
CATALOG=json.loads((ROOT/"rd"/"mesh_triangle_cases.json").read_text("utf-8"))
CASES=CATALOG["cases"]
CYCLE=np.array([[0,0,1],[1,0,0],[0,1,0]])


def build(case,swap=False,transform=False,radius=0):
    asset=case["mesh"]
    if asset=="u":
        verts,faces=runpy.run_path(str(ROOT/"rr"/"baseline.py"))["_u_geometry"]([1,1,1],swap)
        faces=faces.reshape(-1,3)
    else:
        verts,faces=box(*CATALOG["closed_boxes_m"][asset],flip=swap)
    ids=tuple(17+5*i for i in range(len(verts)))
    mk,ck=(ph.CollisionKey(0,1+int(swap),0),ph.CollisionKey(0,2-int(swap),0))
    mesh=ph.CollisionShape(mk,"mesh",ids,triangles=tuple(tuple(ids[i] for i in f) for f in faces))
    cloth_nodes=(53,17,8)
    xyz=case["triangle_m"]
    tri=ph.CollisionShape(ck,"triangle",cloth_nodes[::-1] if swap else cloth_nodes,radius=radius)
    matrix=CYCLE if transform else np.eye(3)
    origin=np.array([1.25,-0.5,0.875]) if transform else np.zeros(3)
    paths={
        mk:ph.CollisionMotion(np.asarray(verts)@matrix.T+origin),
        ck:ph.CollisionMotion(np.asarray(xyz[::-1] if swap else xyz)@matrix.T+origin),
    }
    return ph.CollisionModel([tri,mesh]), paths,mesh,tri,matrix,origin


def detect(model,paths):
    with ph.CollisionWorkspace(model) as workspace:
        (row,)=ph.detect(model,paths,ph.CollisionPolicy(margin=0.5),workspace).pairs
        return row


def materials(witness,model,paths,time):
    for shape,ww,point in zip(model.shapes,(witness.first_weights,witness.second_weights),
                              (witness.first_point,witness.second_point),strict=True):
        path=paths[shape.key]
        nodes=path.start if path.end is None else path.start+time*(path.end-path.start)
        verts=dict(zip(shape.vertex_ids,nodes,strict=True))
        assert min(v for _,v in ww)>=0 and abs(sum(v for _,v in ww)-1)<1e-12
        np.testing.assert_allclose(sum(v*verts[i] for i,v in ww),point,rtol=0,atol=1e-9)
    if witness.normal is not None:
        assert np.linalg.norm(witness.normal)==pytest.approx(1)
        assert np.subtract(witness.second_point,witness.first_point)@witness.normal==pytest.approx(witness.gap_m,abs=1e-9)


@pytest.mark.parametrize("case",CASES,ids=[c["id"] for c in CASES])
@pytest.mark.parametrize("swap",[False,True])
def test_closed_mesh_against_filled_triangle(case,swap,tmp_path):
    model,paths,mesh,tri,matrix,origin=build(case,swap,swap)
    row=detect(model,paths)
    relation=case["relation"]
    status={"separated":"separated","touching":"touching"}.get(relation,"overlap")
    assert row.status==status and row.family=="RD",row.reason
    p=row.mesh_triangle.end
    assert row.mesh_triangle.start is p and p.relation==relation and row.sweep is None
    boxes=CATALOG["u_closed_pieces_m"] if case["mesh"]=="u" else [CATALOG["closed_boxes_m"][case["mesh"]]]
    expected=clipped_fractions(case["triangle_m"],boxes)
    np.testing.assert_allclose((p.inside_fraction,p.boundary_fraction,p.outside_fraction),
                               list(map(float,expected)),atol=1e-12,rtol=0)
    assert p.boundary_gap_m==pytest.approx(case["boundary_gap_m"],abs=1e-10)
    assert row.witness==p.witness
    assert bool(p.boundary)==(case["boundary_gap_m"]==0)
    assert p.witness.gap_m==pytest.approx(-p.boundary_gap_m if relation=="contained" else p.boundary_gap_m,abs=1e-10)
    materials(p.witness,model,paths,0)
    patches=p.boundary or p.closest
    assert patches
    touched=set()
    for patch in patches:
        assert patch.mesh_face in mesh.triangles and patch.dimension in (0,1,2)
        assert len(patch.points)>=patch.dimension+1
        touched.add(patch.mesh_face)
        for w in patch.points:
            materials(w,model,paths,0)
            mw=w.first_weights if row.first==mesh.key else w.second_weights
            assert {i for i,v in mw if v}<=set(patch.mesh_face)
            if p.boundary:
                np.testing.assert_allclose(w.first_point,w.second_point,atol=1e-12,rtol=0)
    if case["id"] in ("contained","parallel-nearest","whole-on-surface"):
        local=paths[mesh.key].start if not swap else (paths[mesh.key].start-origin)@matrix
        top={i for i,xyz in zip(mesh.vertex_ids,local,strict=True) if xyz[2]==1}
        assert touched=={face for face in mesh.triangles if set(face)<=top}
        assert all(patch.dimension==2 for patch in patches)
    if case["id"]=="all-nodes-outside":
        assert expected[0]==1/8 and all(max(map(abs,p))>1 for p in case["triangle_m"])
    if case["id"]=="corner":
        assert all(patch.dimension==0 for patch in patches) and p.witness.normal is None
    (tmp_path/"mesh-triangle.json").write_text(json.dumps(dict(case=case["id"],swap=swap,relation=p.relation,
        fractions=[p.inside_fraction,p.boundary_fraction,p.outside_fraction],oracle=[str(x) for x in expected],
        boundary_gap_m=p.boundary_gap_m,patches=[dict(mesh_face=a.mesh_face,dimension=a.dimension,
        points=[dict(first=w.first_point,second=w.second_point,first_weights=w.first_weights,
                     second_weights=w.second_weights) for w in a.points]) for a in patches]),allow_nan=False),encoding="utf-8")


def test_reject_finite_thickness_and_invalid_pose_before_reporting_solid_contact():
    model,paths,mesh,tri,_,_=build(CASES[0],radius=0.0625)
    row=detect(model,paths)
    assert row.status=="unsupported" and row.mesh_triangle is None
    model,paths,mesh,tri,_,_=build(CASES[0])
    paths[tri.key]=ph.CollisionMotion([[0,0,0],[1,0,0],[2,0,0]])
    row=detect(model,paths)
    assert row.status=="unresolved" and row.mesh_triangle is None
    paths[tri.key]=ph.CollisionMotion(CASES[0]["triangle_m"])
    paths[mesh.key]=ph.CollisionMotion(paths[mesh.key].start,turn=ph.RigidTurn((0,0,0),(0,0,0),(0,0,1),0.5))
    row=detect(model,paths)
    assert row.status=="unsupported"
    exclude=ph.CollisionModel(model.shapes,excluded=((mesh.key,tri.key),))
    assert detect(exclude,paths).status=="filtered"
