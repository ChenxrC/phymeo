"""Robot asset primitives exercised through actual FK, mixed public collision and a moving joint."""

import json
from pathlib import Path

import numpy as np
import pytest

import phymeo as ph
from phymeo.dynamics.rotation import quaternion_matrix


ASSETS=Path(__file__).parent/"reference_newton"/"robot"
URDF=ASSETS/"solid_arm.urdf"
MJCF=ASSETS/"solid_arm.mjcf.xml"
PIVOT=np.array([0.25,-0.125,0.25])


def ry(q):
    c,s=np.cos(q),np.sin(q)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]])


def load(fmt):
    return ph.load_urdf(URDF,gravity=(0.,0.,0.)) if fmt=="urdf" else ph.load_mjcf(MJCF,gravity=(0.,0.,0.))


def native_colliders(asset):
    return {v.name:v for v in asset.collisions}


def plane_model(asset):
    plane=ph.CollisionShape(ph.CollisionKey(0,9,90),"plane",(80,),normal=(0,0,1))
    model=ph.CollisionModel((*asset.collision_model.shapes,plane),asset.collision_model.excluded)
    return plane,model


def sample(asset,q,plane,model,ws):
    paths=asset.collision_motion(np.array([q]))
    paths[plane.key]=ph.CollisionMotion([[0.,0.,0.]])
    result=ph.detect(model,paths,ph.CollisionPolicy(margin=0.75),ws)
    assert result.status=="ok"
    contacts={}
    for r in result.pairs:
        if plane.key not in (r.first,r.second):
            continue
        solid=r.first if r.first!=plane.key else r.second
        point=r.witness.first_point if r.first==solid else r.witness.second_point
        contacts[solid]=r.witness.gap_m
        np.testing.assert_allclose(point[2],r.witness.gap_m,rtol=0,atol=3e-13)
    return paths,contacts


def expected(q):
    c,s=np.cos(q),np.sin(q)
    return dict(base_barrel=0.4375,
                barrel=0.25-0.125*s-0.25*abs(s)-0.125*abs(c),
                capsule=0.25-0.1875*c-0.25*abs(s)-0.0625,
                oval=0.25+0.0625*s-0.125*c-np.hypot(0.25*c,0.0625*s))


@pytest.mark.parametrize("fmt",["urdf","mjcf"])
def test_imported_authored_dimensions_material_owner_and_inertia(fmt):
    a=load(fmt);col=native_colliders(a)
    assert a.root_link=="base" and a.joint_names==("tilt",) and a.link_names==("carrier",)
    assert set(col)==({"base_barrel","barrel"} if fmt=="urdf" else {"base_barrel","barrel","capsule","oval"})
    np.testing.assert_array_equal(a.model.mass,[2])
    np.testing.assert_array_equal(a.model.com,[[0.0625,-0.03125,0.125]])
    np.testing.assert_array_equal(a.model.inertia_diagonal,[[0.125,0.1875,0.25]])
    np.testing.assert_array_equal(a.model.joint_axis,[[0,1,0]])
    np.testing.assert_allclose([a.lower[0],a.upper[0]],[-np.pi,np.pi],rtol=0,atol=1e-14)
    base,barrel=col["base_barrel"],col["barrel"]
    assert (base.link_index,base.shape.key)==(-1,ph.CollisionKey(0,0,0))
    assert (barrel.link_index,barrel.shape.key)==(0,ph.CollisionKey(0,1,0))
    assert barrel.shape.kind=="cylinder" and barrel.shape.curved.dimensions_m==(0.125,0.25)
    assert base.shape.curved.dimensions_m==(0.0625,0.125)
    np.testing.assert_array_equal(barrel.local_vertices,[[0.125,0,0]])
    np.testing.assert_allclose(quaternion_matrix(np.array([barrel.origin_xyzw]))[0][:,2],[1,0,0],atol=2e-15)
    if fmt=="mjcf":
        cap,ell=col["capsule"],col["oval"]
        assert (cap.shape.key,ell.shape.key)==(ph.CollisionKey(0,1,2),ph.CollisionKey(0,1,3))
        assert cap.shape.kind=="capsule" and cap.shape.radius==0.0625 and cap.shape.curved is None
        assert ell.shape.kind=="ellipsoid" and ell.shape.curved.dimensions_m==(0.25,0.125,0.0625)
        np.testing.assert_allclose(cap.local_vertices,[[-0.25,-0.125,-0.1875],[0.25,-0.125,-0.1875]],rtol=0,atol=3e-15)
        np.testing.assert_allclose(quaternion_matrix(np.array([cap.origin_xyzw]))[0][:,2],[1,0,0],atol=2e-15)


@pytest.mark.parametrize("fmt",["urdf","mjcf"])
def test_live_fk_world_pose_and_solid_plane_contact_at_multiple_poses(fmt):
    asset=load(fmt); colliders=native_colliders(asset)
    plane,model=plane_model(asset)
    poses=(0.,np.arctan2(0.6,0.8),np.pi/2,-np.pi/2,np.pi-2.0**-24,np.pi)
    reads=[]
    with ph.CollisionWorkspace(model) as ws:
        for q in poses:
            paths,gaps=sample(asset,q,plane,model,ws)
            values={}
            for name,collider in colliders.items():
                key=collider.shape.key
                local=np.asarray(collider.local_vertices)
                poses_world=local if name=="base_barrel" else PIVOT+local@ry(q).T
                np.testing.assert_allclose(paths[key].start,poses_world,rtol=0,atol=4e-14)
                if collider.shape.curved is not None:
                    body=quaternion_matrix(np.array([paths[key].body_orientation_xyzw]))[0]
                    np.testing.assert_allclose(body,np.eye(3) if name=="base_barrel" else ry(q),rtol=0,atol=4e-14)
                else: assert paths[key].body_orientation_xyzw is None
                np.testing.assert_allclose(gaps[key],expected(q)[name],rtol=0,atol=4e-13)
                values[name]=gaps[key]
            reads.append(dict(q=q,gaps=values))
    print(json.dumps(dict(format=fmt,poses=reads)))


def edit(tmp_path,data,old,new,*,fmt="mjcf"):
    assert old in data
    p=tmp_path/("fixture.xml" if fmt=="mjcf" else "fixture.urdf")
    p.write_text(data.replace(old,new,1),encoding="utf-8")
    return p


def test_mjcf_root_pose_capsule_conventional_size_and_cylinder_fromto_are_not_double_applied(tmp_path):
    text=MJCF.read_text()
    text=text.replace('<body name="base">','<body name="base" pos="0.375 -0.25 0.5" quat="0.5 0.5 0.5 0.5">')
    text=text.replace('size="0.0625" fromto="0.25 -0.125 -0.1875 -0.25 -0.125 -0.1875"',
                      'size="0.0625 0.25" pos="0 -0.125 -0.1875" euler="0 90 0"')
    f=edit(tmp_path,text,'size="0.125 0.25" pos="0.125 0 0" euler="0 90 0"',
           'size="0.125" fromto="0.375 0 0 -0.125 0 0"')
    a=ph.load_mjcf(f)
    original=load("mjcf")
    q=np.array([np.pi-2**-24]); rot=np.array([[0.,0,1],[1,0,0],[0,1,0]])
    original_paths=original.collision_motion(q); paths=a.collision_motion(q)
    for v in a.collisions:
        key=v.shape.key
        np.testing.assert_allclose(paths[key].start,[0.375,-0.25,0.5]+original_paths[key].start@rot.T,rtol=0,atol=3e-14)
        if v.shape.curved is not None:
            local=quaternion_matrix(np.array([v.shape.curved.orientation_xyzw]))[0]
            body=quaternion_matrix(np.array([paths[key].body_orientation_xyzw]))[0]
            original_local=quaternion_matrix(np.array([native_colliders(original)[v.name].shape.curved.orientation_xyzw]))[0]
            expected_body=np.eye(3) if v.link_index==-1 else ry(q[0])
            np.testing.assert_allclose(body@local,rot@expected_body@original_local,rtol=0,atol=4e-14)


def test_joint_effort_moves_imported_cylinder_from_clearance_to_contact():
    asset=load("urdf"); key=native_colliders(asset)["barrel"].shape.key
    plane,registered=plane_model(asset)
    state=ph.SceneState({"robot":ph.RobotState(np.zeros(1),np.zeros(1))})
    load_in=ph.SceneInputs({"robot":ph.RobotInputs(np.array([29/64]),np.zeros((1,3)),np.zeros((1,3)))})
    h=1/32;first=None
    with ph.make_robot_runtime(asset.model,settings=ph.StepSettings(substeps=1)) as rt, ph.CollisionWorkspace(registered) as ws:
        _,initial=sample(asset,0.,plane,registered,ws)
        for k in range(1,25):
            accepted=rt.step(state,load_in,h)
            assert accepted.report.accepted
            state=accepted.state
            q=state.domains["robot"].q[0]
            np.testing.assert_allclose(q,k*(k+1)*h*h,rtol=0,atol=3e-14)
            _,gaps=sample(asset,q,plane,registered,ws)
            np.testing.assert_allclose(gaps[key],expected(q)["barrel"],rtol=0,atol=3e-13)
            if gaps[key]<0 and first is None: first=k
    assert initial[key]>0 and first is not None and gaps[key]<0
    print(json.dumps(dict(actual_steps=24,first_negative_step=first,q=q,end_gap=gaps[key])))


@pytest.mark.parametrize(("fmt","old","new","match"),[
    ("urdf",'radius="0.125" length="0.5"','length="0.5"',"radius"),
    ("urdf",'radius="0.125" length="0.5"','radius="0.125"',"length"),
    ("urdf",'radius="0.125" length="0.5"','radius="-1" length="0.5"',"positive|radius"),
    ("urdf",'radius="0.125" length="0.5"','radius="0.125" length="nan"',"finite|length"),
    ("mjcf",'size="0.125 0.25"','size="0.125"',"size"),
    ("mjcf",'size="0.25 0.125 0.0625"','size="0.25 0.125 0"',"positive|size"),
    ("mjcf",'fromto="0.25 -0.125 -0.1875 -0.25 -0.125 -0.1875"','fromto="0 0 0 0 0 0"',"fromto"),
    ("mjcf",'fromto="0.25 -0.125 -0.1875 -0.25 -0.125 -0.1875"','fromto="0 0 nan 0 0 1"',"fromto|finite"),
    ("mjcf",'fromto="0.25 -0.125 -0.1875 -0.25 -0.125 -0.1875"','pos="1 2 3" fromto="0 0 0 0 0 1"',"fromto"),
    ("mjcf",'name="oval" type="ellipsoid"','name="oval" type="ellipsoid" fromto="0 0 0 0 0 1"',"fromto"),
])
def test_rejects_missing_dimensions_unusable_segments_and_ambiguous_poses(tmp_path,fmt,old,new,match):
    data=(URDF if fmt=="urdf" else MJCF).read_text()
    with pytest.raises(ValueError,match=match):
        (ph.load_urdf if fmt=="urdf" else ph.load_mjcf)(edit(tmp_path,data,old,new,fmt=fmt))
