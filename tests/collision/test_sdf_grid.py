"""Distances in metres and analytic trilinear derivatives in their requested frame."""

import json
import runpy
from pathlib import Path

import numpy as np
import pytest

from test_mesh_mesh_contact import box

REF=Path(__file__).resolve().parents[1]/"reference_newton"


def api():
    from phymeo.geometry.sdf_grid import SignedDistanceGrid,bake_mesh_sdf
    return SignedDistanceGrid,bake_mesh_sdf


def lattice(shape,origin,spacing):
    return np.stack(np.meshgrid(*[origin[k]+spacing[k]*np.arange(shape[k]) for k in (2,1,0)],indexing="ij"),axis=-1)[...,::-1]


def polynomial(points):
    x,y,z=np.moveaxis(np.asarray(points),-1,0)
    return (0.25+0.75*x-0.5*y+0.25*z+0.5*x*y-0.25*y*z+0.375*x*z+0.125*x*y*z,
            np.stack((0.75+0.5*y+0.375*z+0.125*y*z,-0.5+0.5*x-0.25*z+0.125*x*z,
                      0.25-0.25*y+0.375*x+0.125*x*y),axis=-1))


def external():
    grid,_=api()
    shape=(5,4,3)
    origin=(-0.5,0.25,-0.75)
    h=(0.25,0.375,0.5)
    values=polynomial(lattice(shape,origin,h))[0]
    return grid(values,origin_m=origin,spacing_m=h),values


def asset(kind):
    import phymeo as ph
    if kind=="u":
        p,f=runpy.run_path(str(REF/"rr"/"baseline.py"))["_u_geometry"]([1,1,1],False)
        f=f.reshape(-1,3)
    elif kind=="thin":
        p,f=box((7/64,-0.5,-0.5),(9/64,0.5,0.5))
    else:
        p,f=box((-0.5,-0.375,-0.25),(0.5,0.375,0.25))
        if kind=="oblique":
            p=p@np.array([[0.6,0.8,0],[-0.8,0.6,0],[0,0,1]])+[0.125,-0.25,0]
    ids=tuple(3+7*i for i in range(len(p)))
    return ph.CollisionShape(ph.CollisionKey(0,0,0),"mesh",ids,triangles=tuple(tuple(ids[i] for i in t) for t in f)),p


def exact_distance(kind,p):
    p=np.asarray(p)
    if kind=="u":
        fn=runpy.run_path(str(REF/"rd"/"mesh_triangle_sdf.py"))["exact_distance"]
        return np.array([fn(x,"u") for x in p])
    if kind=="oblique":
        p=(p-[0.125,-0.25,0])@np.array([[0.6,-0.8,0],[0.8,0.6,0],[0,0,1]])
    center=np.array([0.125,0,0]) if kind=="thin" else 0
    half=np.array([1/64,0.5,0.5] if kind=="thin" else [0.5,0.375,0.25])
    d=np.abs(p-center)-half
    return np.linalg.norm(np.maximum(d,0),axis=1)+np.minimum(d.max(1),0)


def test_external_values_gradients_terminal_nodes_and_world_frame():
    grid,original=external()
    original.fill(-91)
    points=np.array([[-0.5,0.25,-0.75],[0.5,1.375,0.25],[-0.125,0.8125,-0.375],[0,1,-0.25]])
    distances,gradients=polynomial(points)
    rows=grid.sample_local(points)
    assert grid.shape_xyz==(5,4,3) and grid.geometry_error_bound_m is None
    assert [r.cell_xyz for r in rows]==[(0,0,0),(3,2,1),(1,1,0),(2,2,1)]
    for row,value,grad in zip(rows,distances,gradients,strict=True):
        assert row.status=="valid" and row.distance_m==pytest.approx(value,abs=1e-12)
        np.testing.assert_allclose(row.gradient,grad,rtol=0,atol=1e-12)
        assert not hasattr(row,"surface_point") and not hasattr(row,"material_weights")
    q=(0.5,0.5,0.5,0.5)
    shift=np.array([4,-2,0.75])
    world=np.roll(points,1,axis=1)+shift
    transformed=grid.sample_world(world,translation_m=shift,orientation_xyzw=q)
    for a,b in zip(rows,transformed,strict=True):
        assert a.distance_m==pytest.approx(b.distance_m,abs=1e-12)
        np.testing.assert_allclose(b.gradient,np.roll(a.gradient,1),rtol=0,atol=1e-12)
    outside=grid.sample_local([[np.nextafter(0.5,np.inf),1,0],[-0.51,1,0]])
    assert all(r.status=="outside_grid" and r.distance_m is r.gradient is r.cell_xyz is None for r in outside)
    assert grid.sample_local([])==()


def test_band_mask_crease_and_flat_gradient_are_distinct_from_a_mesh_normal():
    cls,_=api()
    pts=lattice((5,2,2),(-1,0,0),(0.5,1,1))
    band=cls(pts[...,0],origin_m=(-1,0,0),spacing_m=(0.5,1,1),band_m=0.75)
    rows=band.sample_local([[-0.25,0.5,0.5],[0.875,0.5,0.5],[2,0.5,0.5]])
    assert [r.status for r in rows]==["valid","outside_band","outside_grid"]
    assert rows[1].cell_xyz==(3,0,0) and rows[1].distance_m is rows[1].gradient is None
    crease=cls(np.abs(pts[...,0]),origin_m=(-1,0,0),spacing_m=(0.5,1,1))
    left,knot,right=crease.sample_local([[-0.125,0.5,0.5],[0,0.5,0.5],[0.125,0.5,0.5]])
    assert left.gradient==(-1,0,0) and knot.gradient==right.gradient==(1,0,0)
    flat=cls(np.zeros((2,2,2)),origin_m=(0,0,0),spacing_m=(1,1,1)).sample_local([[0.5,0.5,0.5]])[0]
    assert flat.status=="valid" and flat.gradient==(0,0,0)


@pytest.mark.parametrize("kind",["box","oblique","u","thin"])
def test_baked_nodes_independent_geometry_and_interpolation_bound(kind,tmp_path):
    _,bake=api()
    mesh,p=asset(kind)
    layout=((13,13,7),(-3,-3,-1.5),(0.5,0.5,0.5)) if kind=="u" else ((9,9,9),(-1,-1,-1),(0.25,0.25,0.25))
    shape,origin,h=layout
    grid=bake(mesh,p,origin_m=origin,spacing_m=h,shape_xyz=shape)
    nodes=lattice(*layout).reshape(-1,3)
    true=exact_distance(kind,nodes)
    np.testing.assert_allclose(grid.values_zyx_m.ravel(),true,atol=2e-12,rtol=0)
    bound=grid.geometry_error_bound_m
    assert np.linalg.norm(h)/2-1e-15<=bound<=np.linalg.norm(h)/2+1e-12
    assert grid.bake_stats["validated_meshes"]==1 and grid.bake_stats["node_count"]==len(nodes)
    points=np.array([[0.125,0,0],[-0.1875,0.375,-0.0625],[0.53125,0.15625,0.1875],[0.168,0.013,0.095]])
    if kind=="u":
        points=np.array([[0,0,0],[1.5,0,0],[-1.5,0,0],[0,-1.5,0],[0.825,0.225,0.25]])
    observations=grid.sample_local(points)
    assert all(s.status=="valid" for s in observations)
    exact=exact_distance(kind,points)
    measured=np.array([s.distance_m for s in observations])
    assert np.max(np.abs(measured-exact))<=bound+1e-12
    if kind=="u":
        np.testing.assert_allclose(measured[:4],[1,-0.5,-0.5,-0.5],atol=1e-12,rtol=0)
    if kind=="thin":
        assert exact[0]==-1/64 and measured[0]>0 and observations[0].gradient==(0,0,0)
    (tmp_path/"sdf-grid.json").write_text(json.dumps(dict(kind=kind,shape_xyz=shape,
        maximum_node_error_m=float(np.max(np.abs(grid.values_zyx_m.ravel()-true))),
        bound_m=bound,measured_m=measured.tolist(),exact_m=exact.tolist(),bake=grid.bake_stats)),encoding="utf-8")


def test_validation_closed_topology_capacity_and_grid_ownership():
    import phymeo as ph
    cls,bake=api()
    mesh,p=asset("box")
    open_mesh=ph.CollisionShape(mesh.key,"mesh",mesh.vertex_ids,triangles=mesh.triangles[:-1])
    with pytest.raises(ValueError,match="mesh_"):
        bake(open_mesh,p,origin_m=(0,0,0),spacing_m=(1,1,1),shape_xyz=(3,3,3))
    for v,h in [(np.zeros((1,2,2)),(1,1,1)),(np.zeros((2,2,2)),(0,1,1)),(np.full((2,2,2),np.nan),(1,1,1))]:
        with pytest.raises(ValueError):
            cls(v,origin_m=(0,0,0),spacing_m=h)
    with pytest.raises(ValueError,match="capacity"):
        bake(mesh,p,origin_m=(0,0,0),spacing_m=(1,1,1),shape_xyz=(1024,1024,1024))
    grid,_=external()
    with pytest.raises(ValueError):
        grid.values_zyx_m.setflags(write=True)
    with pytest.raises(ValueError):
        grid.sample_local([[np.inf,0,0]])
    with pytest.raises(ValueError):
        grid.sample_world([[0,0,0]],translation_m=(0,0,0),orientation_xyzw=(0,0,0,0))


def test_sdf_bake_validates_topology_once_and_keeps_narrow_band_explicit(monkeypatch):
    from phymeo.geometry import _sdf_mesh_bake
    _,bake=api()
    original=_sdf_mesh_bake.ClosedMesh
    seen=[]
    def counted(*args):
        seen.append(args[0].key)
        return original(*args)
    monkeypatch.setattr(_sdf_mesh_bake,"ClosedMesh",counted)
    mesh,p=asset("box")
    grid=bake(mesh,p,shape_xyz=(9,9,9),origin_m=(-1,-1,-1),spacing_m=(0.25,)*3,band_m=0.4)
    assert seen==[mesh.key]
    valid,off=grid.sample_local([[0.625,0,0],[0.99,0,0]])
    assert valid.distance_m==pytest.approx(0.125) and off.status=="outside_band" and off.gradient is None
    assert grid.geometry_error_bound_m is not None
