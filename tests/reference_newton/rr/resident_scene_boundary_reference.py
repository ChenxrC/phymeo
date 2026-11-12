"""S1/S2 fixed original materials and independent rigid FK; no product import."""
from pathlib import Path
import numpy as np

FAMILIES=("ordinary","affine","floating","floating_affine")
PIDS=(7,29,53,71)
WORLDS=(83,17)
H=1/1024
MARGIN=.5
RADIUS=1/8
VERTICES=np.array(((0,0,0),(0,2,0),(0,0,2),(-1,0,0)),np.float64)
FACES=((0,1,2),(0,2,3),(0,3,1),(1,3,2))
ROUND=np.array(((-.25,.5,.5),),np.float64)
TOOL_R=np.array(((1,0,0),(0,0,-1),(0,1,0)),np.float64)
TOOL_P=np.array((0,1.75,0),np.float64)
CYCLE_R=np.array(((0,0,1),(1,0,0),(0,1,0)),np.float64)


def rotation(q):
    x,y,z,w=np.asarray(q,np.float64)/np.linalg.norm(q)
    return np.array(((1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)),
                     (2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)),
                     (2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y))))


def write_asset(directory,*,anchor,floating,cycle=False,folded=False):
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=True)
    frame=CYCLE_R if cycle else np.eye(3)
    origin=np.asarray(anchor,np.float64)
    rpy="0 0 0" if floating or not cycle else "1.5707963267948966 0 1.5707963267948966"
    positions=[np.array((x,0,0)) if floating else origin+frame@np.array((x,0,0)) for x in (-.25,.25)]
    xyz=[" ".join(str(float(x)) for x in row) for row in positions]
    obj="".join("v "+" ".join(str(float(x)) for x in row)+"\n" for row in VERTICES)
    obj+="".join("f "+" ".join(str(i+1) for i in face)+"\n" for face in FACES)
    (directory/"surface.obj").write_text(obj,encoding="utf-8")
    mesh='<collision name="surface"><geometry><mesh filename="surface.obj"/></geometry></collision>'
    tool=(f'<link name="surface_tool">{mesh}</link><joint name="tool_fixed" type="fixed">'
          '<parent link="left_jaw"/><child link="surface_tool"/><origin xyz="0 1.75 0" rpy="1.5707963267948966 0 0"/></joint>') if folded else ''
    xml=f'''<robot name="scene_source_f0">
<link name="base"><inertial><mass value="2"/><inertia ixx=".2" iyy=".3" izz=".4" ixy="0" ixz="0" iyz="0"/></inertial></link>
<link name="left_jaw"><inertial><mass value="1"/><inertia ixx=".0625" iyy=".078125" izz=".09375" ixy="0" ixz="0" iyz="0"/></inertial>{'' if folded else mesh}</link>
<joint name="driver" type="prismatic"><parent link="base"/><child link="left_jaw"/><origin xyz="{xyz[0]}" rpy="{rpy}"/><axis xyz="1 0 0"/><limit lower="-.1" upper=".1" effort="1.5" velocity=".2"/></joint>
<link name="right_jaw"><inertial><mass value="2"/><inertia ixx=".09375" iyy=".109375" izz=".125" ixy="0" ixz="0" iyz="0"/></inertial><collision name="round"><origin xyz="-.25 .5 .5"/><geometry><sphere radius=".125"/></geometry></collision></link>
<joint name="follower" type="prismatic"><parent link="base"/><child link="right_jaw"/><origin xyz="{xyz[1]}" rpy="{rpy}"/><axis xyz="1 0 0"/><limit lower="-.1" upper=".1" effort=".125" velocity=".08"/><mimic joint="driver" multiplier="-1" offset=".004"/></joint>{tool}</robot>'''
    path=directory/"f0.urdf"; path.write_text(xml,encoding="utf-8")
    return path,dict(anchor=origin,cycle=cycle,folded=folded,xml=xml,obj=obj)


def points(checkpoint,family,description,which,local):
    q=np.asarray(checkpoint.full_position if "affine" in family else checkpoint.position,np.float64)
    if family.startswith("floating") or family=="free":
        frame=rotation(q[3:7]); origin=q[:3]; scalar=q[7:]
    else:
        frame=CYCLE_R if description["cycle"] else np.eye(3)
        origin=np.asarray(description["anchor"]); scalar=q
    if which=="root": return origin+np.asarray(local)@frame.T
    index=0 if which=="driver" else 1
    offset=-.25 if index==0 else .25
    return origin+(np.asarray(local)+np.array((offset+scalar[index],0,0)))@frame.T


def local_surface(folded):
    return VERTICES@TOOL_R.T+TOOL_P if folded else VERTICES.copy()


def memory(shapes,points,groups,candidates):
    persistent=68*shapes+36*points
    global_query=112*shapes+36*points
    group_bytes=sum(40*f+48*(2*f-1)+(52+4*((f+29)//30))*r for f,r in groups)
    prepass=persistent+global_query+group_bytes
    return dict(persistent_bytes=persistent,prepass_required_bytes=prepass,
                peak_required_bytes=prepass+1060*sum(candidates))
