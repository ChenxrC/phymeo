"""Imported closed triangle meshes use authored geometry through real FK and public collision."""

import shutil
from pathlib import Path

import numpy as np
import pytest

import phymeo as ph

ASSETS=Path(__file__).parent/"reference_newton"/"robot"/"mesh_assets"
FILES={"urdf":"mesh_arm.urdf","mjcf":"mesh_arm.xml"}
PIVOT=np.array([.5,.25,-.125])
ORIGIN=np.array([.125,-.25,.0625])
SCALE=np.array([.125,.25,.25])
RZ=np.array([[0,-1,0],[1,0,0],[0,0,1]])
XY=((0,0),(2,0),(2,1),(1,1),(1,2),(0,2))
BRACKET=np.array([(x,y,z) for z in (-.5,.5) for x,y in XY])
TETRA=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]])


def load(fmt,root=ASSETS):
    return (ph.load_urdf if fmt=="urdf" else ph.load_mjcf)(root/FILES[fmt],gravity=(0.,0.,0.))


def world(raw,q):
    c,s=np.cos(q),np.sin(q)
    ry=np.array([[c,0,s],[0,1,0],[-s,0,c]])
    return PIVOT+(ORIGIN+(np.asarray(raw)*SCALE)@RZ.T)@ry.T


def branch(tmp_path,fmt):
    local=tmp_path/"asset"
    shutil.copytree(ASSETS,local)
    return local,local/FILES[fmt]


def change(path,old,new):
    text=path.read_text("utf8")
    assert old in text
    path.write_text(text.replace(old,new),encoding="utf8")


def query(asset,collider,center,q):
    shape=collider.shape
    probe=ph.CollisionShape(ph.CollisionKey(0,11,0),"sphere",(71,),1/64)
    model=ph.CollisionModel((*asset.collision_model.shapes,probe),asset.collision_model.excluded)
    paths=asset.collision_motion(np.array([q]))
    paths[probe.key]=ph.CollisionMotion(np.array([center]))
    with ph.CollisionWorkspace(model) as ws:
        result=ph.detect(model,paths,ph.CollisionPolicy(margin=.5),ws)
    assert result.status=="ok"
    return next(row for row in result.pairs if {row.first,row.second}=={shape.key,probe.key})


@pytest.mark.parametrize("fmt",FILES)
def test_two_formats_import_the_authored_vertices_shapes_material_faces_and_owners(fmt):
    asset=load(fmt)
    assert (asset.root_link,asset.link_names,asset.joint_names)==("base",("carrier",),("tilt",))
    assert [c.name for c in asset.collisions]==["root_marker","bracket"]
    root,moving=asset.collisions
    assert (root.link_index,root.link_name,root.shape.key)==(-1,"base",ph.CollisionKey(0,0,0))
    assert (moving.link_index,moving.link_name,moving.shape.key)==(0,"carrier",ph.CollisionKey(0,1,int(fmt=="mjcf")))
    assert root.shape.kind==moving.shape.kind=="mesh"
    assert len(root.shape.triangles)==4 and len(moving.shape.triangles)==20
    assert moving.shape.vertex_ids==tuple(range(12))
    assert set(root.shape.triangles)=={(0,2,1),(0,1,3),(0,3,2),(1,2,3)}
    np.testing.assert_allclose(root.local_vertices,[-1,0,0]+TETRA*[.125,.25,.5],atol=0,rtol=0)
    np.testing.assert_allclose(moving.local_vertices,ORIGIN+(BRACKET*SCALE)@RZ.T,rtol=0,atol=2e-15)
    np.testing.assert_allclose(asset.model.mass,[2],atol=0,rtol=0)
    np.testing.assert_allclose(asset.model.com,[[.0625,-.03125,.125]],atol=0,rtol=0)
    for q in (0.,np.arctan2(.6,.8),-np.pi/2):
        paths=asset.collision_motion(np.array([q]))
        np.testing.assert_allclose(paths[moving.shape.key].start,world(BRACKET,q),rtol=0,atol=2e-14)
        np.testing.assert_allclose(paths[root.shape.key].start,root.local_vertices,atol=0,rtol=0)


@pytest.mark.parametrize("fmt",FILES)
@pytest.mark.parametrize("q",[0.,np.arctan2(.6,.8)])
def test_imported_notch_is_empty_and_leg_is_solid_after_joint_motion(fmt,q):
    asset=load(fmt);root,moving=asset.collisions
    for raw,relation,gap in (([1.375,1.375,0],"separated",.03125),
                              ([.5,1.5,0],"mesh_contains_round",.046875),
                              ([1.0625,1.5,0],"boundary_overlap",0)):
        row=query(asset,moving,world(raw,q),q)
        pose=row.mesh_round.end
        assert pose.relation==relation
        assert pose.boundary_gap_m==pytest.approx(gap,abs=2e-12)
        for support in pose.supports:
            assert support.face in moving.shape.triangles
    root_probe=query(asset,root,[-1.03125,.03125,.0625],q)
    assert root_probe.mesh_round.end.relation=="separated"
    assert root_probe.mesh_round.end.boundary_gap_m==pytest.approx(.015625,abs=2e-12)


def test_mjcf_reference_frame_then_signed_nonuniform_scale_uses_outward_triangles(tmp_path):
    local,path=branch(tmp_path,"mjcf")
    change(path,'file="tetra.obj" scale="0.125 0.25 0.5"',
           'file="tetra.obj" refpos="0.25 -0.5 0" refquat="2 0 0 2" scale="-2 3 0.5"')
    asset=load("mjcf",local);root=asset.collisions[0]
    expected=np.array([[-1,.75,0],[-1,-2.25,0],[-3,.75,0],[-1,.75,.5]])+[-1,0,0]
    np.testing.assert_allclose(root.local_vertices,expected,atol=2e-14,rtol=0)
    assert set(root.shape.triangles)=={(0,1,2),(0,3,1),(0,2,3),(1,3,2)}
    row=query(asset,root,[-2.4,.15,.1],0.)
    assert row.mesh_round.end.relation=="mesh_contains_round"


@pytest.mark.parametrize("fmt",FILES)
def test_visual_only_mesh_does_not_open_a_missing_asset(fmt,tmp_path):
    local,path=branch(tmp_path,fmt)
    if fmt=="urdf":
        change(path,'<visual name="display"><geometry><mesh filename="models/tetra.obj"',
                    '<visual name="display"><geometry><mesh filename="package://missing/render.obj"')
    else:
        change(path,"</asset>",'<mesh name="unused" file="absent.obj"/></asset>')
        change(path,'name="display" type="mesh" mesh="root_part"','name="display" type="mesh" mesh="unused"')
    assert [c.name for c in load(fmt,local).collisions]==["root_marker","bracket"]


@pytest.mark.parametrize("transform,reason",[
    ("open","mesh_open_or_misoriented_edge|closed mesh face count"),("inward","mesh_nonpositive_or_inward"),
    ("duplicate","mesh_duplicate_face"),("quad","triangle"),("zero","index"),
    ("out_of_range","index"),("nan","finite"),("line","unsupported"),
    ("duplicate_point","mesh_coincident_nodes"),("bad_texture","index"),
])
def test_invalid_referenced_obj_fails_at_asset_load(tmp_path,transform,reason):
    local,path=branch(tmp_path,"urdf")
    file=local/"models"/"tetra.obj"
    content=file.read_text(encoding="utf8")
    if transform=="open": content=content.replace("f 2 3 4","")
    if transform=="inward": content="\n".join(
        " ".join(["f",*reversed(line.split()[1:])]) if line.startswith("f ") else line for line in content.splitlines())
    if transform=="duplicate": content+="\nf 2 3 4\n"
    if transform=="quad": content=content.replace("f 2 3 4","f 1 2 3 4")
    if transform=="zero": content=content.replace("f 2 3 4","f 0 3 4")
    if transform=="out_of_range": content=content.replace("f 2 3 4","f 77 3 4")
    if transform=="nan": content=content.replace("v 0 0 0","v 0 nan 0")
    if transform=="line": content+="\nl 1 2\n"
    if transform=="duplicate_point": content+="\nv 0 0 0\n"
    if transform=="bad_texture": content=content.replace("3/3/1","3/88/1")
    file.write_text(content,encoding="utf8")
    with pytest.raises(ValueError,match=reason):
        load("urdf",local)


@pytest.mark.parametrize("fmt",FILES)
@pytest.mark.parametrize("replacement,reason",[("missing.obj","missing|not found"),("../escape.obj","outside|escape"),
    ("package://unresolved/part.obj","URI|relative"),("https://example.invalid/part.obj","URI|relative"),
    ("part.stl","OBJ|obj")])
def test_collision_file_resolution_is_local_and_explicit(fmt,replacement,reason,tmp_path):
    local,path=branch(tmp_path,fmt)
    if fmt=="urdf":
        change(path,'filename="models/bracket.obj"',f'filename="{replacement}"')
    else:
        change(path,'file="bracket.obj"',f'name="bracket" file="{replacement}"')
    with pytest.raises(ValueError,match=reason):
        load(fmt,local)


@pytest.mark.parametrize("fmt",FILES)
def test_zero_mesh_scale_fails(fmt,tmp_path):
    local,path=branch(tmp_path,fmt);change(path,'scale="0.125 0.25 0.25"','scale="0.125 0 0.25"')
    with pytest.raises(ValueError,match="scale"):
        load(fmt,local)


@pytest.mark.parametrize("old,new,reason",[
    ('file="bracket.obj"', 'name="root_part" file="bracket.obj"', "duplicate"),
    ('name="bracket" mesh="bracket"', 'name="bracket" mesh="absent"', "unknown|missing|undefined"),
    ('name="bracket" mesh="bracket"', 'name="bracket" type="box" mesh="bracket"', "mesh|fit"),
    ('file="bracket.obj"', 'file="bracket.obj" maxhullvert="4"', "unsupported"),
    ('meshdir="models"', 'meshdir="models" strippath="true"', "unsupported"),
    ('file="bracket.obj"', 'file="bracket.obj" refquat="0 0 0 0"', "refquat"),
])
def test_mjcf_unknown_names_fitting_and_unsupported_semantics_fail(tmp_path,old,new,reason):
    local,path=branch(tmp_path,"mjcf");change(path,old,new)
    with pytest.raises(ValueError,match=reason):
        load("mjcf",local)
