"""Analytic effective joint inertia, passive damping and MJCF admission cases."""

from dataclasses import replace

import numpy as np
import pytest

from phymeo.builders.floating_mjcf import load_floating_mjcf
from phymeo.core.rigid import RigidBodyModel
from phymeo.core.robot import RobotModel
from phymeo.dynamics.floating_robot import (
    FloatingRobotModel,
    FloatingRobotState,
    floating_kinematics,
)
from phymeo.solvers.floating_tree import (
    FloatingRobotInputs,
    floating_dynamics_terms,
    floating_kinetic_energy,
    floating_mass_matrix,
    floating_step_mass_matrix,
    step_floating_robot,
)


def system(kind="revolute", *, armature=None, damping=None, rate=1.0):
    gravity = np.zeros(3)
    root = RigidBodyModel(np.array([0.5]), np.full((1, 3), 2.0), gravity)
    children = RobotModel(
        np.array([-1]),
        (kind,),
        np.zeros((1, 3)),
        np.array([[0.0, 0.0, 0.0, 1.0]]),
        np.array([[0.0, 0.0, 1.0] if kind == "revolute" else [1.0, 0.0, 0.0]]),
        np.ones(1),
        np.zeros((1, 3)),
        np.ones((1, 3)),
        gravity,
    )
    model = FloatingRobotModel(
        root,
        np.zeros(3),
        children,
        joint_armature=armature,
        joint_damping=damping,
    )
    velocity = np.zeros(7)
    velocity[5 if kind == "revolute" else 0] = -rate / 3
    velocity[6] = rate
    state = FloatingRobotState(np.array([0.0, 0, 0, 0, 0, 0, 1, 0.4]), velocity)
    return model, state


def body_totals(model, state):
    geometry = floating_kinematics(model, state)
    momenta = np.array([2.0, 1.0])[:, None] * geometry.com_velocity
    spins = np.array([2.0, 1.0])[:, None] * geometry.angular_velocity
    kinetic = 0.5 * (
        np.sum(momenta * geometry.com_velocity) + np.sum(spins * geometry.angular_velocity)
    )
    angular = np.cross(geometry.com_position, momenta).sum(axis=0) + spins.sum(axis=0)
    return momenta.sum(axis=0), angular, kinetic


@pytest.mark.parametrize("kind", ("revolute", "prismatic"))
def test_armature_changes_only_joint_inertia_and_effective_energy(kind):
    model, state = system(kind, armature=np.array([1.0]))
    baseline = replace(model, joint_armature=None)
    mass = floating_mass_matrix(model, state)
    original = floating_mass_matrix(baseline, state)
    expected_delta = np.zeros((7, 7))
    expected_delta[6, 6] = 1
    np.testing.assert_allclose(mass - original, expected_delta, rtol=0, atol=1e-15)
    root_index = 5 if kind == "revolute" else 0
    np.testing.assert_allclose(mass[np.ix_([root_index, 6], [root_index, 6])], [[3, 1], [1, 2]])
    _, _, body_energy = body_totals(model, state)
    np.testing.assert_allclose(floating_kinetic_energy(model, state), body_energy + 0.5)
    # Added joint energy never changes measured body poses or COM mass.
    np.testing.assert_array_equal(
        floating_kinematics(model, state).com_position,
        floating_kinematics(baseline, state).com_position,
    )


@pytest.mark.parametrize("kind", ("revolute", "prismatic"))
def test_continuous_equation_retains_root_joint_coupling(kind):
    model, state = system(kind, armature=np.array([1.0]), damping=np.array([4.0]))
    inputs = FloatingRobotInputs.zeros(1)
    terms = floating_dynamics_terms(model, state, inputs)
    expected = np.zeros(7)
    expected[5 if kind == "revolute" else 0] = 4 / 5
    expected[6] = -12 / 5
    np.testing.assert_allclose(terms.acceleration, expected, rtol=0, atol=2e-14)
    expected_bias = np.zeros(7)
    expected_bias[6] = 4
    np.testing.assert_allclose(terms.bias, expected_bias, rtol=0, atol=2e-14)
    np.testing.assert_allclose(
        terms.mass_matrix @ terms.acceleration + terms.bias,
        terms.applied,
        rtol=0,
        atol=2e-14,
    )


@pytest.mark.parametrize("kind", ("revolute", "prismatic"))
def test_implicit_damping_has_analytic_decay_and_discrete_work_balance(kind):
    model, state = system(kind, armature=np.array([1.0]), damping=np.array([4.0]))
    h = 0.5
    initial_energy = floating_kinetic_energy(model, state)
    initial_p, initial_l, _ = body_totals(model, state)
    result = step_floating_robot(model, state, FloatingRobotInputs.zeros(1), h)
    assert result.report.accepted, result.report.reason
    after = result.state
    expected_rate = 5 / 11  # effective inertia 5/3, D=4: rate/(1+h*D/(5/3)).
    np.testing.assert_allclose(after.v[6], expected_rate, rtol=0, atol=2e-15)
    np.testing.assert_allclose(
        after.v[5 if kind == "revolute" else 0], -expected_rate / 3, rtol=0, atol=2e-15
    )
    mass = floating_mass_matrix(model, state)
    effective = floating_step_mass_matrix(model, state, h)
    expected_effective = mass.copy()
    expected_effective[6, 6] += h * 4
    np.testing.assert_array_equal(effective, expected_effective)
    delta = after.v - state.v
    final_energy = floating_kinetic_energy(model, after)
    passive_loss = h * 4 * after.v[6] ** 2
    integration_loss = 0.5 * delta @ mass @ delta
    np.testing.assert_allclose(
        initial_energy - final_energy, passive_loss + integration_loss, rtol=0, atol=2e-15
    )
    final_p, final_l, _ = body_totals(model, after)
    np.testing.assert_allclose(final_p, initial_p, atol=2e-15)
    np.testing.assert_allclose(final_l, initial_l, atol=2e-15)


def test_large_positive_damping_does_not_reverse_or_explode_velocity():
    model, state = system(armature=np.array([1.0]), damping=np.array([1e8]))
    result = step_floating_robot(model, state, FloatingRobotInputs.zeros(1), 1.0)
    assert result.report.accepted
    expected = (5 / 3) / (5 / 3 + 1e8)
    np.testing.assert_allclose(result.state.v[6], expected, rtol=2e-8, atol=1e-15)
    assert 0 < result.state.v[6] < state.v[6]
    assert floating_kinetic_energy(model, result.state) < floating_kinetic_energy(model, state)


def test_zero_fields_retain_default_step_and_are_owned_read_only():
    default, state = system()
    explicit = replace(default, joint_armature=np.zeros(1), joint_damping=np.zeros(1))
    inputs = FloatingRobotInputs(np.array([0.75]), np.zeros((2, 3)), np.zeros((2, 3)))
    left = step_floating_robot(default, state, inputs, 1 / 64)
    right = step_floating_robot(explicit, state, inputs, 1 / 64)
    assert left.report.accepted and right.report.accepted
    np.testing.assert_array_equal(left.state.q, right.state.q)
    np.testing.assert_array_equal(left.state.v, right.state.v)
    for name in ("joint_armature", "joint_damping"):
        with pytest.raises(ValueError):
            getattr(default, name).flags.writeable = True
    values = np.array([0.25])
    isolated = replace(default, joint_armature=values)
    values[0] = 100
    assert isolated.joint_armature[0] == 0.25


@pytest.mark.parametrize("field", ("joint_armature", "joint_damping"))
@pytest.mark.parametrize(
    "value,error",
    (
        (np.array([-1.0]), ValueError),
        (np.array([np.inf]), ValueError),
        (np.array([np.nan]), ValueError),
        (np.array([0.0, 1.0]), ValueError),
        (np.array([0.25], dtype=np.float32), TypeError),
        (np.array([True]), TypeError),
    ),
)
def test_actuator_fields_reject_invalid_domain(field, value, error):
    model, _ = system()
    with pytest.raises(error):
        replace(model, **{field: value})


def test_root_only_model_has_no_scalar_actuator_slots():
    root = RigidBodyModel(np.array([0.5]), np.full((1, 3), 2.0), np.zeros(3))
    model = FloatingRobotModel(root, np.zeros(3))
    assert model.joint_armature.shape == model.joint_damping.shape == (0,)
    with pytest.raises(ValueError, match="length 0"):
        FloatingRobotModel(root, np.zeros(3), joint_armature=np.ones(1))


XML = """<mujoco model="actuator-defaults"><compiler angle="radian"/>
<default><joint armature=".5" damping="2"/>
  <default class="robot"><joint damping="4"/>
    <default class="fast"><joint armature=".75"/></default>
  </default>
</default>
<worldbody><body name="root" childclass="robot"><freejoint/>
<inertial mass="2" pos="0 0 0" diaginertia="2 2 2"/>
<body name="arm"><joint name="hinge" class="fast" axis="0 0 1" range="-4 4"/>
<inertial mass="1" pos="0 0 0" diaginertia="1 1 1"/>
<geom type="sphere" size=".1"/></body></body></worldbody>
<actuator><motor joint="hinge" ctrlrange="-8 8"/></actuator></mujoco>"""


def load(tmp_path, text=XML):
    path = tmp_path / "actuator.xml"
    path.write_text(text)
    return load_floating_mjcf(path, gravity=(0, 0, 0))


@pytest.mark.parametrize(
    "change,armature,damping",
    (
        ("", 0.75, 4),
        ('damping=".125"', 0.75, 0.125),
        ('armature="0" damping="0"', 0, 0),
    ),
)
def test_mjcf_resolves_defaults_and_explicit_overrides(tmp_path, change, armature, damping):
    text = XML.replace('name="hinge"', f'name="hinge" {change}')
    model = load(tmp_path, text).model
    np.testing.assert_array_equal(model.joint_armature, [armature])
    np.testing.assert_array_equal(model.joint_damping, [damping])


@pytest.mark.parametrize(
    "old,new,match",
    (
        ('name="hinge"', 'name="hinge" armature="-1"', "nonnegative"),
        ('name="hinge"', 'name="hinge" damping="-1"', "nonnegative"),
        ('name="hinge"', 'name="hinge" damping="nan"', "finite"),
        ('name="hinge"', 'name="hinge" damping="1 2 3"', "1 finite"),
        ('class="fast" axis=', 'class="missing" axis=', "default class"),
        ('damping="4"', 'damping="4" frictionloss=".2"', "frictionloss"),
        ("<freejoint/>", '<freejoint damping=".1"/>', "freejoint"),
        ("<motor joint=", '<motor armature="1" joint=', "joint properties"),
        ("</actuator>", '<motor joint="hinge" ctrlrange="-8 8"/></actuator>', "one named"),
        ("</actuator>", '<motor joint="missing" ctrlrange="-8 8"/></actuator>', "motor mapping"),
    ),
)
def test_mjcf_rejects_unsupported_or_invalid_actuator_data(tmp_path, old, new, match):
    with pytest.raises(ValueError, match=match):
        load(tmp_path, XML.replace(old, new))
