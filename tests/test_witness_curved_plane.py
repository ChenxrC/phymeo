"""Published oriented ellipsoid/half-space witnesses applied to their two live rigid bodies."""

import numpy as np
import pytest

import phymeo as ph


CYCLE = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
CYCLE_Q = (0.5, 0.5, 0.5, 0.5)
S = np.sqrt(0.5)
RZ90 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
# Homogeneous 2 kg ellipsoid moments, permuted from its shape into the body axes.
SOLID_INERTIA = (1 / 8, 1 / 32, 17 / 160)
# Both poses share this material point. The second rotates the entire system about it.
CONTACT = np.array([1.25, -0.5, 0.75])
PLANE = ph.CollisionKey(9, 2, 4)
SOLID = ph.CollisionKey(9, 8, 3)
LOCAL_NORMAL = (0.0, 0.0, 1.0)
BINDINGS = {PLANE: ("plane", 0), SOLID: ("solid", 0)}


def rigid_port(domain, mass, inertia, center, q, velocity, local_point):
    model = ph.RigidBodyModel(np.array([1.0 / mass]), np.array([inertia]), np.zeros(3))
    state = ph.RigidBodyState(
        center[None, :], np.array([q]), velocity[None, :], np.zeros((1, 3))
    )
    return ph.RigidPointPort(model, state, np.array([0]), np.array([local_point]), domain=domain)


def collision_scene(cycled):
    turn = CYCLE if cycled else np.eye(3)
    # In the base pose the solid body is +90 degrees about z. Its local shape
    # maps x to body y, so the 1/2 m semiaxis faces the plane along world -x.
    # Multiplication by CYCLE on the left represents the second physical pose.
    solid_q = (S, 0.0, S, 0.0) if cycled else (0.0, 0.0, S, S)
    plane_q = (0.5, 0.5, 0.5, -0.5) if cycled else CYCLE_Q
    normal = turn @ [1.0, 0.0, 0.0]
    plane_center = CONTACT - turn @ [0.0, 0.25, 0.0]
    solid_center = CONTACT + 0.5 * normal
    asset = ph.CurvedSolid("ellipsoid", (0.5, 0.25, 0.125), CYCLE_Q)
    model = ph.CollisionModel(
        [
            ph.CollisionShape(SOLID, "ellipsoid", (47,), curved=asset),
            ph.CollisionShape(PLANE, "plane", (91,), normal=tuple(normal)),
        ]
    )
    paths = {
        SOLID: ph.CollisionMotion([solid_center], body_orientation_xyzw=solid_q),
        PLANE: ph.CollisionMotion([plane_center]),
    }
    with ph.CollisionWorkspace(model) as work:
        (pair,) = ph.detect(model, paths, ph.CollisionPolicy(), work).pairs
    assert (pair.first, pair.second, pair.status) == (PLANE, SOLID, "touching")
    assert pair.curved.owner == SOLID and pair.witness.gap_m == 0.0
    np.testing.assert_allclose(pair.witness.first_point, CONTACT, atol=1e-14, rtol=0)
    np.testing.assert_allclose(pair.witness.second_point, CONTACT, atol=1e-14, rtol=0)
    np.testing.assert_allclose(pair.witness.normal, normal, atol=1e-14, rtol=0)
    ports = {
        "plane": rigid_port(
            "plane", 4.0, (0.25, 0.25, 0.25), plane_center, plane_q, np.zeros(3), (0.25, 0.0, 0.0)
        ),
        "solid": rigid_port(
            "solid", 2.0, SOLID_INERTIA, solid_center, solid_q, -normal, (0.0, 0.5, 0.0)
        ),
    }
    for port in ports.values():
        np.testing.assert_allclose(port.positions()[0], CONTACT, atol=1e-14, rtol=0)
    return turn, pair, ports


def check_unchanged(ports, before):
    for name, old in before.items():
        observed = ports[name].snapshot()
        for field in ("position", "orientation", "velocity", "angular_velocity"):
            np.testing.assert_array_equal(getattr(observed, field), getattr(old, field))


@pytest.mark.parametrize("cycled", [False, True], ids=["base", "axis-cycle"])
def test_live_curved_plane_witness_rotates_impulse_and_reaction(cycled):
    turn, pair, ports = collision_scene(cycled)
    before = {name: port.snapshot() for name, port in ports.items()}
    out = ph.solve_witness_impulses(
        ports, [(pair, BINDINGS)], [0.0], [0.0], local_normals={PLANE: LOCAL_NORMAL}
    )
    normal = turn @ [1.0, 0.0, 0.0]
    # Closing speed is 1 m/s. The two translations contribute 1/2 + 1/4
    # kg^-1 and the offset plane contributes (1/4)^2 / (1/4) kg^-1.
    np.testing.assert_allclose(out.normal_impulses, [1.0], atol=2e-10, rtol=0)
    np.testing.assert_allclose(out.world_impulses[0], normal, atol=2e-10, rtol=0)
    plane, solid = out.states["plane"], out.states["solid"]
    np.testing.assert_allclose(solid.velocity[0], -0.5 * normal, atol=2e-10, rtol=0)
    np.testing.assert_allclose(plane.velocity[0], -0.25 * normal, atol=2e-10, rtol=0)
    np.testing.assert_allclose(solid.angular_velocity[0], np.zeros(3), atol=2e-10, rtol=0)
    np.testing.assert_allclose(plane.angular_velocity[0], turn @ [0, 0, 1], atol=2e-10, rtol=0)
    np.testing.assert_allclose(out.relative_velocity_after[0], np.zeros(3), atol=2e-10, rtol=0)
    assert out.energy_change == pytest.approx(-0.5, abs=2e-10)
    momentum = 2.0 * solid.velocity[0] + 4.0 * plane.velocity[0]
    arm = solid.position[0] - plane.position[0]
    angular = np.cross(arm, 2.0 * solid.velocity[0])
    solid_rotation = turn @ RZ90
    solid_spin = solid_rotation @ (np.asarray(SOLID_INERTIA) * (solid_rotation.T @ solid.angular_velocity[0]))
    angular += solid_spin + 0.25 * plane.angular_velocity[0]
    np.testing.assert_allclose(momentum, -2.0 * normal, atol=2e-10, rtol=0)
    np.testing.assert_allclose(angular, turn @ [0, 0, 0.5], atol=2e-10, rtol=0)
    check_unchanged(ports, before)


def test_old_plane_normal_is_rejected_at_the_same_live_material_point():
    _, old, _ = collision_scene(False)
    _, current, ports = collision_scene(True)
    np.testing.assert_array_equal(old.witness.first_point, current.witness.first_point)
    np.testing.assert_array_equal(old.witness.second_point, current.witness.second_point)
    assert np.dot(old.witness.normal, current.witness.normal) == 0.0
    before = {name: port.snapshot() for name, port in ports.items()}
    with pytest.raises(ValueError):
        ph.solve_witness_impulses(
            ports, [(old, BINDINGS)], [0.0], [0.0], local_normals={PLANE: LOCAL_NORMAL}
        )
    check_unchanged(ports, before)


@pytest.mark.parametrize("options", [{}, {"local_normals": {PLANE: (0.0, 0.0, -1.0)}}])
def test_half_space_requires_its_correct_oriented_body_local_normal(options):
    _, pair, ports = collision_scene(False)
    before = {name: port.snapshot() for name, port in ports.items()}
    with pytest.raises(ValueError):
        ph.solve_witness_impulses(ports, [(pair, BINDINGS)], [0.0], [0.0], **options)
    check_unchanged(ports, before)
