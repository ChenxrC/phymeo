"""Autonomous floating import keeps an independent frame for child COM tensors."""

import numpy as np
import pytest

from phymeo.builders.native_assembly import NativeAssetInstance
from phymeo.builders.scalar_physolve import compile_scalar_template
from phymeo.dynamics.rotation import quaternion_matrix

URDF = """<robot name="asymmetric">
  <link name="base">
    <inertial><mass value="2"/><inertia ixx=".4" iyy=".5" izz=".6"
            ixy="0" ixz="0" iyz="0"/></inertial>
  </link>
  <link name="hinge_link">
    <inertial><origin xyz=".125 -.0625 .03125" rpy="{inertia_rpy}"/><mass value="1"/>
      <inertia ixx=".375" iyy=".375" izz=".5" ixy="{cross}" ixz="0" iyz="0"/>
    </inertial>
    <collision name="probe"><origin xyz="0 .125 0"/>
      <geometry><sphere radius=".03125"/></geometry>
    </collision>
  </link>
  <joint name="pivot" type="revolute">
    <parent link="base"/><child link="hinge_link"/>
    <origin xyz=".25 .125 -.0625" rpy="0 0 .25"/><axis xyz="0 0 1"/>
    <limit lower="-2" upper="2" effort="20" velocity="20"/>
    <dynamics damping=".1" friction=".2"/>
  </joint>
</robot>"""


def compile_articulation(tmp_path, cross, inertia_yaw=0.):
    asset = NativeAssetInstance(
        "robot", str(tmp_path / "in-memory.urdf"), floating=True,
        drive_mode="scalar_pd", urdf_xml=URDF.format(cross=cross, inertia_rpy=f"0 0 {inertia_yaw}"),
    )
    return compile_scalar_template(
        asset, timestep=.002, gravity=(0, -9.81, 0), statics=(),
        integrator="implicitfast", capture_visuals=True, capture_frames=True,
    )


@pytest.mark.parametrize("inertia_yaw", [0., .25])
def test_child_products_keep_link_com_collision_and_passive_parameters(tmp_path, inertia_yaw):
    model = compile_articulation(tmp_path, "-.125", inertia_yaw).compilation
    assert model.extended
    link = model.model.bodies[1]
    assert link.name == "robot::hinge_link" and link.parent == 1 and link.dof == 0
    assert link.mass == 1
    assert link.com == (.125, -.0625, .03125)
    axes = quaternion_matrix(np.asarray([link.inertia_frame]))[0]
    source_tensor = [[.375, -.125, 0], [-.125, .375, 0], [0, 0, .5]]
    c, s = np.cos(inertia_yaw), np.sin(inertia_yaw)
    authored_axes = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    expected = authored_axes @ source_tensor @ authored_axes.T
    np.testing.assert_allclose((axes * link.inertia) @ axes.T, expected, atol=5e-16, rtol=0)
    assert link.position == (.25, .125, -.0625)
    np.testing.assert_allclose(link.rotation, (0, 0, np.sin(.125), np.cos(.125)),
                               atol=5e-16, rtol=0)
    assert model.model.joint_axis == ((0, 0, 1),)
    assert model.geom_rows[0][1] == 2
    assert model.geom_rows[0][4] == (0, .125, 0)
    assert model.joints[0].damping == .1 and model.joints[0].frictionloss == .2


def test_diagonal_link_retains_the_compact_path(tmp_path):
    assert not compile_articulation(tmp_path, "0").compilation.extended


def test_invalid_full_tensor_does_not_bypass_physical_checks(tmp_path):
    with pytest.raises(ValueError, match="invalid mass or COM inertia"):
        compile_articulation(tmp_path, ".5")

def test_continuous_joint_keeps_drive_limits_without_position_stops(tmp_path):
    urdf = URDF.format(cross=0, inertia_rpy="0 0 0")
    urdf = urdf.replace('type="revolute"', 'type="continuous"')
    urdf = urdf.replace('lower="-2" upper="2" ', "")
    asset = NativeAssetInstance(
        "wheel", str(tmp_path / "wheel.urdf"), floating=True,
        drive_mode="scalar_pd", urdf_xml=urdf,
    )
    model = compile_scalar_template(
        asset, timestep=.002, gravity=(0, -9.81, 0), statics=(),
        integrator="implicitfast", capture_visuals=False, capture_frames=True,
    ).compilation
    assert model.has_free_root and model.extended
    assert model.model.joint_types == ("continuous",)
    assert model.limits == ((None, None, 20.),)
    assert model.joints[0].effort_limit == 20.
    assert model.joints[0].damping == .1
    assert model.joints[0].frictionloss == .2
