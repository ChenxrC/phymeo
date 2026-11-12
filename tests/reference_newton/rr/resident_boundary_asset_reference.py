"""M1-A predeclared authored geometry; no Phymeo/TYX import or product FK."""

from pathlib import Path

import numpy as np

LAYOUTS = ("ordinary", "affine", "floating", "floating_affine")
VERTICES = np.array(((0, 0, 0), (0, 2, 0), (0, 0, 2), (-1, 0, 0)), np.float64)
FACES = ((0, 1, 2), (0, 3, 1), (0, 2, 3), (1, 3, 2))
ROUND = np.array(((-0.25, 0.5, 0.5),), np.float64)
CAPSULE = np.array(((-0.25, 0.5, 0.5), (0, 0.5, 0.5)), np.float64)
FIXED_CAPSULE = np.array(((0, 0.5, 0.5), (-0.25, 0.5, 0.5)), np.float64)
# Fixed MJCF uses +z=(start-end)/2 and material IDs at center-/+half_axis:
# IDs 0,1 therefore denote end,start. Floating fromto preserves start,end.
# Keep the two original importer address conventions; the geometric axis agrees.
TOOL_R = np.array(((1, 0, 0), (0, 0, -1), (0, 1, 0)), np.float64)
TOOL_P = np.array((0, 1.75, 0), np.float64)
H = 1 / 1024
RADIUS = 1 / 8
MARGIN = 0.5


def write_urdf(directory, *, folded=False, mass="1", multiplier="-1"):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    obj = "".join("v " + " ".join(str(float(x)) for x in row) + "\n" for row in VERTICES)
    obj += "".join("f " + " ".join(str(i + 1) for i in face) + "\n" for face in FACES)
    (directory / "surface.obj").write_text(obj, encoding="utf-8")
    mesh = (
        '<collision name="surface"><geometry><mesh filename="surface.obj"/></geometry></collision>'
    )
    tool = (
        (
            f'<link name="surface_tool">{mesh}</link><joint name="tool_fixed" type="fixed">'
            '<parent link="left_jaw"/><child link="surface_tool"/>'
            '<origin xyz="0 1.75 0" rpy="1.5707963267948966 0 0"/></joint>'
        )
        if folded
        else ""
    )
    xml = f'''<robot name="m1_asset_f0">
<link name="base"><inertial><mass value="2"/><inertia ixx=".2" iyy=".3" izz=".4" ixy="0" ixz="0" iyz="0"/></inertial></link>
<link name="left_jaw"><inertial><mass value="{mass}"/><inertia ixx=".0625" iyy=".078125" izz=".09375" ixy="0" ixz="0" iyz="0"/></inertial>{"" if folded else mesh}</link>
<joint name="driver" type="prismatic"><parent link="base"/><child link="left_jaw"/><origin xyz="-.25 0 0"/><axis xyz="1 0 0"/><limit lower="-.1" upper=".1" effort="1.5" velocity=".2"/></joint>
<link name="right_jaw"><inertial><mass value="2"/><inertia ixx=".09375" iyy=".109375" izz=".125" ixy="0" ixz="0" iyz="0"/></inertial><collision name="round"><origin xyz="-.25 .5 .5"/><geometry><sphere radius=".125"/></geometry></collision></link>
<joint name="follower" type="prismatic"><parent link="base"/><child link="right_jaw"/><origin xyz=".25 0 0"/><axis xyz="1 0 0"/><limit lower="-.1" upper=".1" effort=".125" velocity=".08"/><mimic joint="driver" multiplier="{multiplier}" offset=".004"/></joint>{tool}</robot>'''
    path = directory / "f0.urdf"
    path.write_text(xml, encoding="utf-8")
    return path


def material_points(folded=False, triangle=False):
    surface = VERTICES[:3] if triangle else VERTICES
    return surface @ TOOL_R.T + TOOL_P if folded else surface.copy()


def rotation(q):
    q = np.asarray(q, np.float64)
    x, y, z, w = q / np.linalg.norm(q)
    return np.array(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        )
    )


def pose(state, layout, group):
    q = np.asarray(state.full_position if "affine" in layout else state.position, np.float64)
    root, frame, scalar = (
        (q[:3], rotation(q[3:7]), q[7:])
        if layout.startswith("floating")
        else (np.zeros(3), np.eye(3), q)
    )
    offset = -0.25 if group == 0 else 0.25
    return root + frame @ np.array((offset + scalar[group], 0, 0)), frame


def world_points(state, layout, group, points):
    position, frame = pose(state, layout, group)
    return position + np.asarray(points, np.float64) @ frame.T


def write_mjcf(directory, *, floating=False):
    directory = Path(directory)
    write_urdf(directory)
    if floating:
        xml = """<mujoco model="asset_capsule_free"><compiler angle="radian"/>
<worldbody><body name="base"><freejoint/><inertial pos="0 0 0" mass="2" diaginertia=".2 .3 .4"/>
<body name="arm"><joint name="hinge" type="hinge" axis="0 0 1" range="-1 1"/><inertial pos="0 0 0" mass="1" diaginertia=".0625 .078125 .09375"/>
<geom name="round" type="capsule" fromto="-.25 .5 .5 0 .5 .5" size=".125"/></body></body></worldbody>
<actuator><motor joint="hinge" ctrlrange="-1 1" gear="1"/></actuator></mujoco>"""
    else:
        xml = """<mujoco model="asset_capsule_fixed"><compiler angle="radian"/><asset><mesh name="surface_mesh" file="surface.obj"/></asset>
<worldbody><body name="base"><body name="left_jaw" pos="-.25 0 0"><joint name="driver" type="slide" axis="1 0 0" range="-.1 .1"/><inertial pos="0 0 0" mass="1" diaginertia=".0625 .078125 .09375"/><geom name="surface" type="mesh" mesh="surface_mesh"/></body>
<body name="right_jaw" pos=".25 0 0"><joint name="follower" type="slide" axis="1 0 0" range="-.1 .1"/><inertial pos="0 0 0" mass="2" diaginertia=".09375 .109375 .125"/><geom name="round" type="capsule" fromto="-.25 .5 .5 0 .5 .5" size=".125"/></body></body></worldbody></mujoco>"""
    path = directory / "capsule.xml"
    path.write_text(xml, encoding="utf-8")
    return path
