"""Independent exhaustive finite-feature distances. Coordinates retain their binary rationals."""
from fractions import Fraction as F
from itertools import combinations
from math import sqrt

import numpy as np


def sub(a,b):
    return tuple(x-y for x,y in zip(a,b,strict=True))


def dot(a,b):
    return sum(x*y for x,y in zip(a,b,strict=True))


def mix(x,y,t):
    return tuple(a+t*(b-a) for a,b in zip(x,y,strict=True))


def length2(p,q):
    d=sub(q,p)
    return dot(d,d)


def segment(p,a,b):
    u=sub(b,a)
    t=max(F(0),min(F(1),dot(sub(p,a),u)/dot(u,u)))
    return mix(a,b,t),t


def vt(p,tri):
    a,b,c=tri
    u,v,r=sub(b,a),sub(c,a),sub(p,a)
    aa,ab,bb=dot(u,u),dot(u,v),dot(v,v)
    det=aa*bb-ab*ab
    if not det:
        raise ValueError("degenerate frozen material triangle")
    xb=(bb*dot(r,u)-ab*dot(r,v))/det
    xc=(aa*dot(r,v)-ab*dot(r,u))/det
    if min(xb,xc,1-xb-xc)>=0:
        z=tuple(x+xb*y+xc*w for x,y,w in zip(a,u,v,strict=True))
        return length2(p,z)
    return min(length2(p,segment(p,tri[i],tri[j])[0]) for i,j in ((0,1),(1,2),(2,0)))


def ee(edge_a,edge_b):
    a,b=edge_a
    c,d=edge_b
    u,v,r=sub(b,a),sub(d,c),sub(c,a)
    aa,ab,bb=dot(u,u),dot(u,v),dot(v,v)
    det=aa*bb-ab*ab
    values=[length2(p,segment(p,*edge_b)[0]) for p in edge_a]
    values.extend(length2(p,segment(p,*edge_a)[0]) for p in edge_b)
    if det:
        s=(bb*dot(u,r)-ab*dot(v,r))/det
        t=(ab*dot(u,r)-aa*dot(v,r))/det
        if 0<=s<=1 and 0<=t<=1:
            values.append(length2(mix(a,b,s),mix(c,d,t)))
    return min(values)


def materials(inputs):
    coords={}
    faces=[]
    for sheet in inputs["sheets"]:
        for ident,*xyz in sheet["vertices"]:
            if ident in coords:
                raise ValueError("frozen material IDs must be globally unambiguous")
            coords[ident]=tuple(map(F,xyz))
        faces.extend(tuple(sorted(face)) for face in sheet["faces"])
    return coords,faces


def feature_sets(inputs):
    xyz,faces=materials(inputs)
    edges=sorted({tuple(sorted(edge)) for tri in faces for edge in combinations(tri,2)})
    connected={v:{v} for v in xyz}
    for a,b in edges:
        connected[a].add(b)
        connected[b].add(a)
    assert inputs["topological_ring"]==2
    limit=(F(inputs["self_margin_m"])+F(inputs["detection_gap_m"]))**2
    vs={}
    es={}
    for face in faces:
        tri=tuple(xyz[i] for i in face)
        for node,p in xyz.items():
            if not connected[node].isdisjoint(face):
                continue
            d2=vt(p,tri)
            if d2<limit: vs[node,face]=sqrt(d2)
    for a,b in combinations(edges,2):
        if not (connected[a[0]]|connected[a[1]]).isdisjoint(b):
            continue
        d2=ee(tuple(xyz[i] for i in a),tuple(xyz[i] for i in b))
        if d2<limit: es[a,b]=sqrt(d2)
    return vs,es


def weights_check(observed,expected,tol):
    a={int(i):float(w) for i,w in observed if abs(w)>tol}
    b=dict(expected)
    assert a.keys()==b.keys(),(a,b)
    assert all(abs(a[i]-v)<=tol for i,v in b.items()),(a,b)


def check_forces(data,case,engine):
    inp,exp,tol=case["inputs"],case["expected"],case["tolerances"]
    xyz,_=materials(inp)
    forces={int(i):np.asarray(f,dtype=float) for i,f in data["forces_n"].items()}
    assert forces.keys()==xyz.keys()
    for f in forces.values():
        assert f.shape==(3,) and np.isfinite(f).all()
    residual=np.sum(list(forces.values()),axis=0)
    np.testing.assert_allclose(residual,0,rtol=0,atol=tol["force_n"])
    magnitude=exp[engine+"_unique_force_n"]
    if magnitude is None:
        assert any(np.linalg.norm(f)>tol["force_n"] for f in forces.values())
        return
    expected={i:np.zeros(3) for i in xyz}
    for sign,weights in ((-1,exp["focus_weights_a"]),(1,exp["focus_weights_b"])):
        for i,w in weights:
            expected[i][2]=sign*w*magnitude
    np.testing.assert_allclose([forces[i] for i in sorted(xyz)],[expected[i] for i in sorted(xyz)],rtol=0,atol=tol["force_n"])
