"""Independent sphere/link surface witnesses, two-body wrenches and coupled motion."""

from itertools import product

import numpy as np
import pytest

from phymeo.constraints.robot_contact import RobotContactMaterial

LAW=RobotContactMaterial(2000.,10.,.5,.04)


def binding(name="rail",kind="capsule", owner=0, vertices=((0.,0.,-.25),(0.,0.,.25)), radius=.1):
    from phymeo.solvers.sphere_link_contact import LinkSolid
    return LinkSolid(name,name,owner,kind,vertices,radius)


def evaluate(shapes, center, speed=(0.,0.,0.), omega=(0.,0.,0.), law=LAW,
             com=(0.,0.,0.), twist=(0.,)*6,rotation=None):
    from phymeo.solvers.sphere_link_contact import SphereLinkContacts
    c=np.array(center);rot=np.eye(3) if rotation is None else np.array(rotation)
    return SphereLinkContacts(shapes).evaluate(np.zeros((1,3)), rot[None],np.array([com]),
        np.array([twist]), c,.05,np.array(speed),np.array(omega),law)


def test_finite_capsule_side_wrench_damping_power_and_com():
    result=evaluate([binding()],(.149,0.,.075),(-.1,.06,0.),com=(0,0,.02))
    row,=result.contacts
    assert row["feature"]=="shaft" and row["gap_m"]==pytest.approx(-.001)
    np.testing.assert_allclose(row["normal_world"],[1,0,0])
    np.testing.assert_allclose(row["sphere_surface_m"],[.099,0,.075])
    np.testing.assert_allclose(row["link_surface_m"],[.1,0,.075])
    np.testing.assert_allclose(row["point_world_m"],[.0995,0,.075])
    np.testing.assert_allclose(row["link_com_world_m"],[0,0,.02])
    np.testing.assert_allclose(row["force_on_payload_N"],[3,-1.5,0],atol=1e-12)
    np.testing.assert_allclose(row["force_on_link_N"],[-3,1.5,0],atol=1e-12)
    np.testing.assert_allclose(row["torque_on_payload_Nm"],[0,0,.07425],atol=1e-12)
    np.testing.assert_allclose(row["torque_on_link_Nm"],[-.0825,-.165,.14925],atol=1e-12)
    assert row["pair_power_W"]==pytest.approx(-.39)
    # Compression contributes -k*delta*delta_dot; damping and slip dissipate 0.19 W.
    assert row["pair_power_W"]+.2==pytest.approx(-.19)
    np.testing.assert_allclose(result.payload_wrench,[3,-1.5,0,0,0,.07425],atol=1e-12)


def test_finite_capsule_end_and_separation_do_not_extend_its_shaft():
    cap=binding()
    n=np.array([.6,0,.8])
    tip=np.array([0,0,.25])
    row,=evaluate([cap],tip+.149*n).contacts
    assert row["feature"]=="end1"
    np.testing.assert_allclose(row["normal_world"],n)
    np.testing.assert_allclose(row["point_world_m"],tip+.0995*n)
    assert not evaluate([cap],tip+[0,0,.151]).contacts
    release,=evaluate([cap],tip+.149*n,speed=2*n).contacts
    assert release["normal_load_N"]==0 and not np.any(release["force_on_payload_N"])


def test_spinning_structure_relative_velocity_and_registration_order():
    first=binding("left");second=binding("right","sphere",vertices=((.299,0,.075),),radius=.102)
    one=evaluate([first,second],[.149,0,.075],twist=[0,0,0,0,0,2])
    other=evaluate([second,first],[.149,0,.075],twist=[0,0,0,0,0,2])
    np.testing.assert_allclose(one.payload_wrench,other.payload_wrench,rtol=0,atol=0)
    assert [r["shape"] for r in one.contacts]==[r["shape"] for r in other.contacts]
    row=one.contacts[0]
    np.testing.assert_allclose(row["relative_velocity_m_s"],[0,-.199,0])
    assert row["pair_power_W"]<0


def test_oriented_box_corner_and_ambiguous_geometry():
    corners=np.array(list(product((-1,1),repeat=3)))*[.1,.2,.3]
    box=binding("box","box",vertices=corners,radius=0.)
    rot=np.array([[0,-1,0],[1,0,0],[0,0,1]])
    n=np.array([.6,.8,0]);boundary=rot@np.array([.1,.2,0])
    row,=evaluate([box],boundary+.049*(rot@n),rotation=rot).contacts
    np.testing.assert_allclose(row["force_on_payload_N"],2*(rot@n),atol=1e-12)
    assert row["feature"]=="edge"
    assert not evaluate([box],boundary+.051*(rot@n),rotation=rot).contacts
    with pytest.raises(ValueError,match="normal"):
        evaluate([binding()], [0,0,.1])
    with pytest.raises(ValueError,match="box"):
        binding("bad","box",vertices=corners[[0,1,2,3,4,6,5,7]],radius=0.)
