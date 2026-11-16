"""Independent virtual-work and force-map tests for the frozen I0 rows.

No dynamics are advanced. Supplied Jacobians use one shared coordinate basis;
world-origin action/reaction is distinct from total generalized-force balance.
"""

from dataclasses import FrozenInstanceError, fields, replace

import numpy as np
import pytest

from phymeo.geometry.area_pressure import (
    AreaPressureIdentity,
    AreaPressureStamp,
    AreaPressureSurface,
    AreaPressureTriangle,
    quadrature_area_pressure,
)
from phymeo.solvers.area_pressure_rows import (
    AreaPressureKinematics,
    assemble_area_pressure_rows,
    map_area_pressure_normal_forces,
)


RTOL = ATOL = 1e-12


def _triangle(index=0, **changes):
    values = dict(
        triangle_id=f"triangle-{index}", compliant_cell_id="cell-4", rigid_face_id="face-9",
        vertices=((0, 0, 0), (2, 0, 0), (0, 1, 0)), vertex_pressure=(1, 2, 4),
        pressure_gradient=(0.5, 3, 5),
    )
    values.update(changes)
    return AreaPressureTriangle(**values)


def _quad(triangles=None, stamp=None):
    identity = AreaPressureIdentity("soft", "rigid", "soft-material", "rigid-material")
    if stamp is None:
        stamp = AreaPressureStamp("world-3", 7, 0.125, "accepted-configuration-7")
    if triangles is None:
        triangles = (_triangle(),)
    return quadrature_area_pressure(AreaPressureSurface(identity, stamp, tuple(triangles)))


def _fixed(quad, dofs=3):
    count = len(quad.sample_ids)
    jacobian_a = np.zeros((count, 3, dofs))
    for axis in range(min(dofs, 3)):
        jacobian_a[:, axis, axis] = 1.0
    return AreaPressureKinematics(
        quad.stamp, quad.sample_ids, tuple(f"dof-{index}" for index in range(dofs)),
        jacobian_a, np.zeros_like(jacobian_a),
    )


def _free_body_kinematics(quad, com_a, com_b):
    count = len(quad.sample_ids)
    jacobian_a, jacobian_b = np.zeros((count, 3, 12)), np.zeros((count, 3, 12))
    for row, point in enumerate(quad.points):
        jacobian_a[row, :, :3] = np.eye(3)
        jacobian_b[row, :, 6:9] = np.eye(3)
        for axis, unit in enumerate(np.eye(3)):
            jacobian_a[row, :, 3 + axis] = np.cross(unit, point - com_a)
            jacobian_b[row, :, 9 + axis] = np.cross(unit, point - com_b)
    return AreaPressureKinematics(
        quad.stamp, quad.sample_ids, tuple(f"free-dof-{index}" for index in range(12)),
        jacobian_a, jacobian_b,
    )


def _forge(value, **changes):
    """Bypass frozen constructors to exercise the consumer's trust boundary."""
    forged = object.__new__(type(value))
    for field in fields(value):
        object.__setattr__(forged, field.name, changes.get(field.name, getattr(value, field.name)))
    return forged


def _assert_bytes_backed(array):
    assert isinstance(array, np.ndarray) and array.dtype == np.dtype(np.float64)
    owner = array
    while isinstance(owner, np.ndarray):
        assert not owner.flags.writeable
        with pytest.raises(ValueError):
            owner.setflags(write=True)
        owner = owner.base
    while isinstance(owner, memoryview):
        assert owner.readonly
        owner = owner.obj
    assert isinstance(owner, bytes)
    if array.size:
        with pytest.raises(ValueError):
            array.flat[0] = 999


def test_local_stiffness_gap_and_elastic_force_match_independent_pressure_integral():
    quad = _quad()
    rows = assemble_area_pressure_rows(quad, _fixed(quad))
    np.testing.assert_allclose(rows.stiffness, (5 / 3,) * 3, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(rows.surrogate_gap, (-1 / 3, -13 / 30, -19 / 30),
                               rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(rows.elastic_force, (5 / 9, 13 / 18, 19 / 18),
                               rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(rows.normal_jacobian, ((0, 0, 1),) * 3, rtol=RTOL, atol=ATOL)
    loads = map_area_pressure_normal_forces(rows, rows.elastic_force, stamp=quad.stamp)
    np.testing.assert_allclose(loads.wrench_a, (0, 0, 7 / 3, 11 / 12, -3 / 2, 0),
                               rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(loads.wrench_b, (0, 0, -7 / 3, -11 / 12, 3 / 2, 0),
                               rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(loads.generalized_a, (0, 0, 7 / 3), rtol=RTOL, atol=ATOL)
    np.testing.assert_array_equal(loads.generalized_b, np.zeros(3))
    np.testing.assert_allclose(loads.generalized_total, (0, 0, 7 / 3), rtol=RTOL, atol=ATOL)
    for result in (rows, loads):
        assert result.identity == quad.identity and result.stamp == quad.stamp
        assert result.sample_ids == quad.sample_ids and result.source_ids == quad.source_ids
        assert result.dof_ids == ("dof-0", "dof-1", "dof-2")


def test_separation_decreases_pressure_with_directional_not_gradient_norm_stiffness():
    quad = _quad()
    rows = assemble_area_pressure_rows(quad, _fixed(quad))
    separation = 0.01
    # A's advected field loses g*delta pressure at a fixed B witness. Its large
    # tangential gradient is not part of the normal pressure derivative.
    shifted = _quad((_triangle(vertex_pressure=(0.95, 1.95, 3.95)),))
    shifted_rows = assemble_area_pressure_rows(shifted, _fixed(shifted))
    np.testing.assert_allclose(shifted_rows.surrogate_gap - rows.surrogate_gap, separation,
                               rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(shifted_rows.elastic_force - rows.elastic_force, -1 / 60,
                               rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(shifted_rows.stiffness, rows.stiffness, rtol=RTOL, atol=ATOL)
    assert not np.isclose(rows.stiffness[0], np.sqrt(0.5**2 + 3**2 + 5**2) / 3)
    zero = _quad((_triangle(vertex_pressure=(0, 0, 0), pressure_gradient=(0, 0, 5)),))
    zero_rows = assemble_area_pressure_rows(zero, _fixed(zero))
    np.testing.assert_array_equal(zero_rows.surrogate_gap, np.zeros(3))
    np.testing.assert_array_equal(zero_rows.elastic_force, np.zeros(3))
    np.testing.assert_allclose(zero_rows.stiffness, (5 / 3,) * 3, rtol=RTOL, atol=ATOL)


def test_all_twelve_free_body_columns_power_and_common_rigid_motion():
    # Rotate the triangle so every world component of its normal is exercised.
    rotation = np.array(((12 / 13, 0, 5 / 13), (4 / 13, 3 / 5, -48 / 65),
                         (-3 / 13, 4 / 5, 36 / 65)))
    translation = np.array((0.2, -0.3, 0.4))
    vertices = np.array(((0, 0, 0), (2, 0, 0), (0, 1, 0)))
    quad = _quad((_triangle(vertices=vertices @ rotation.T + translation,
                            pressure_gradient=rotation @ np.array((0.5, 3, 5))),))
    com_a, com_b = np.array((0.3, -0.4, 0.2)), np.array((-0.7, 0.5, -0.1))
    kinematics = _free_body_kinematics(quad, com_a, com_b)
    rows = assemble_area_pressure_rows(quad, kinematics)
    expected_rows = []
    for point, normal in zip(quad.points, quad.normals):
        expected_rows.append(np.r_[normal, np.cross(point - com_a, normal),
                                   -normal, -np.cross(point - com_b, normal)])
    np.testing.assert_allclose(rows.normal_jacobian, expected_rows, rtol=RTOL, atol=ATOL)
    velocity = np.array((0.2, -0.6, 0.5, 0.8, -0.3, 0.4, -0.7, 0.1, 0.3, -0.2, 0.9, 0.6))
    forces = np.array((2.0, 3.0, 7.0))
    power = 0.0
    for point, normal, force in zip(quad.points, quad.normals, forces):
        va = velocity[:3] + np.cross(velocity[3:6], point - com_a)
        vb = velocity[6:9] + np.cross(velocity[9:12], point - com_b)
        power += force * np.dot(normal, va - vb)
    loads = map_area_pressure_normal_forces(rows, forces, stamp=quad.stamp)
    assert float(loads.generalized_total @ velocity) == pytest.approx(power, rel=RTOL, abs=ATOL)
    np.testing.assert_allclose(loads.generalized_total, loads.generalized_a + loads.generalized_b,
                               rtol=RTOL, atol=ATOL)
    force_a, moment_a = loads.wrench_a[:3], loads.wrench_a[3:]
    force_b, moment_b = loads.wrench_b[:3], loads.wrench_b[3:]
    np.testing.assert_allclose(loads.generalized_a,
                               np.r_[force_a, moment_a - np.cross(com_a, force_a), np.zeros(6)],
                               rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(loads.generalized_b,
                               np.r_[np.zeros(6), force_b, moment_b - np.cross(com_b, force_b)],
                               rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(loads.wrench_a + loads.wrench_b, np.zeros(6), rtol=RTOL, atol=ATOL)
    assert np.linalg.norm(loads.generalized_total) > 0
    common_v, common_w = np.array((0.7, 0.4, -0.2)), np.array((-0.3, 0.5, 0.9))
    common = np.r_[common_v + np.cross(common_w, com_a), common_w,
                   common_v + np.cross(common_w, com_b), common_w]
    np.testing.assert_allclose(rows.normal_jacobian @ common, np.zeros(3), rtol=RTOL, atol=ATOL)


def test_shared_coordinates_and_joint_type_jacobians_preserve_both_sides():
    quad = _quad()
    ja, jb = np.zeros((3, 3, 4)), np.zeros((3, 3, 4))
    ja[:, 2, 0] = jb[:, 2, 0] = 1.0  # One shared translation, not two free coordinates.
    ja[:, 2, 1], jb[:, 2, 1] = 2.0, -1.0
    for row, point in enumerate(quad.points):
        ja[row, :, 2] = np.cross((0, 1, 0), point - (0.2, 0, 0))
        jb[row, :, 3] = np.cross((1, 0, 0), point - (0, -0.3, 0))
    kin = AreaPressureKinematics(quad.stamp, quad.sample_ids,
                                 ("shared-slide", "opposed-slide", "hinge-a", "hinge-b"), ja, jb)
    rows = assemble_area_pressure_rows(quad, kin)
    np.testing.assert_allclose(rows.normal_jacobian,
                               ((0, 3, -2 / 15, -7 / 15), (0, 3, -17 / 15, -7 / 15),
                                (0, 3, -2 / 15, -29 / 30)), rtol=RTOL, atol=ATOL)
    loads = map_area_pressure_normal_forces(rows, (2, 3, 7), stamp=quad.stamp)
    # Hand sums: sum f=12, sum x*f=7, sum y*f=11/2.
    np.testing.assert_allclose(loads.generalized_a, (12, 24, -4.6, 0), rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(loads.generalized_b, (-12, 12, 0, -9.1), rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(loads.generalized_total, (0, 36, -4.6, -9.1), rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(loads.wrench_a, (0, 0, 12, 5.5, -7, 0), rtol=RTOL, atol=ATOL)
    # Reindexing the shared basis reindexes generalized loads, not world wrench.
    order = (3, 1, 0, 2)
    permuted = AreaPressureKinematics(quad.stamp, quad.sample_ids,
                                      tuple(kin.dof_ids[index] for index in order),
                                      ja[:, :, order], jb[:, :, order])
    permuted_rows = assemble_area_pressure_rows(quad, permuted)
    permuted_loads = map_area_pressure_normal_forces(permuted_rows, (2, 3, 7), stamp=quad.stamp)
    np.testing.assert_allclose(permuted_loads.generalized_total,
                               loads.generalized_total[list(order)], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(permuted_loads.wrench_a, loads.wrench_a, rtol=RTOL, atol=ATOL)


def test_variable_normals_retain_zero_resultant_nonzero_couple():
    positive = _triangle(0, vertices=((0, 1 / 3, -1 / 3), (0, 7 / 3, -1 / 3),
                                      (0, 1 / 3, 2 / 3)),
                         vertex_pressure=(3, 3, 3), pressure_gradient=(5, 0, 0))
    negative = _triangle(1, vertices=((0, -5 / 3, -1 / 3), (0, -5 / 3, 2 / 3),
                                      (0, 1 / 3, -1 / 3)),
                         vertex_pressure=(3, 3, 3), pressure_gradient=(-5, 0, 0))
    quad = _quad((positive, negative))
    kin = _free_body_kinematics(quad, np.zeros(3), np.zeros(3))
    rows = assemble_area_pressure_rows(quad, kin)
    loads = map_area_pressure_normal_forces(rows, rows.elastic_force, stamp=quad.stamp)
    np.testing.assert_allclose(loads.wrench_a, (0, 0, 0, 0, 0, -6), rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(loads.wrench_b, (0, 0, 0, 0, 0, 6), rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(loads.generalized_total,
                               (0, 0, 0, 0, 0, -6, 0, 0, 0, 0, 0, 6), rtol=RTOL, atol=ATOL)


def test_empty_contact_and_zero_force_keep_shapes_identity_and_phase():
    empty = _quad(())
    kin = _fixed(empty, dofs=4)
    rows = assemble_area_pressure_rows(empty, kin)
    assert rows.normal_jacobian.shape == (0, 4)
    for name in ("stiffness", "surrogate_gap", "elastic_force"):
        assert getattr(rows, name).shape == (0,)
    loads = map_area_pressure_normal_forces(rows, (), stamp=empty.stamp)
    assert loads.sample_ids == loads.source_ids == ()
    assert loads.identity == empty.identity and loads.stamp == empty.stamp
    for name in ("generalized_a", "generalized_b", "generalized_total"):
        np.testing.assert_array_equal(getattr(loads, name), np.zeros(4))
    np.testing.assert_array_equal(loads.wrench_a, np.zeros(6))
    np.testing.assert_array_equal(loads.wrench_b, np.zeros(6))
    quad = _quad()
    rows = assemble_area_pressure_rows(quad, _fixed(quad))
    zero = map_area_pressure_normal_forces(rows, np.zeros(3), stamp=quad.stamp)
    np.testing.assert_array_equal(zero.wrench_a, np.zeros(6))
    np.testing.assert_array_equal(zero.generalized_total, np.zeros(3))


def test_kinematics_rows_and_loads_are_owned_frozen_and_bytes_backed():
    quad = _quad()
    ja, jb = np.tile(np.eye(3), (3, 1, 1)), np.zeros((3, 3, 3))
    kin = AreaPressureKinematics(quad.stamp, quad.sample_ids, ("x", "y", "z"), ja, jb)
    ja[:] = jb[:] = 99
    np.testing.assert_array_equal(kin.jacobian_a, np.tile(np.eye(3), (3, 1, 1)))
    np.testing.assert_array_equal(kin.jacobian_b, np.zeros((3, 3, 3)))
    rows = assemble_area_pressure_rows(quad, kin)
    forces = np.array((2.0, 3.0, 7.0))
    loads = map_area_pressure_normal_forces(rows, forces, stamp=quad.stamp)
    forces[:] = 0
    np.testing.assert_allclose(loads.wrench_a, (0, 0, 12, 5.5, -7, 0), rtol=RTOL, atol=ATOL)
    for name in ("jacobian_a", "jacobian_b"):
        _assert_bytes_backed(getattr(kin, name))
    for name in ("points", "normals", "jacobian_a", "jacobian_b", "normal_jacobian",
                 "stiffness", "surrogate_gap", "elastic_force"):
        _assert_bytes_backed(getattr(rows, name))
    for name in ("generalized_a", "generalized_b", "generalized_total", "wrench_a", "wrench_b"):
        _assert_bytes_backed(getattr(loads, name))
    for value, name in ((kin, "dof_ids"), (rows, "sample_ids"), (loads, "source_ids")):
        with pytest.raises(FrozenInstanceError):
            setattr(value, name, ())


@pytest.mark.parametrize("change", [
    {"world_id": "other-world"}, {"step_index": 8}, {"sample_time_s": 0.126},
    {"configuration_id": "trial-not-accepted"},
])
def test_every_stamp_component_must_match_without_implicit_rebinding(change):
    quad = _quad()
    kin = _fixed(quad)
    other = replace(quad.stamp, **change)
    with pytest.raises(ValueError):
        assemble_area_pressure_rows(quad, replace(kin, stamp=other))
    rows = assemble_area_pressure_rows(quad, kin)
    with pytest.raises(ValueError):
        map_area_pressure_normal_forces(rows, (1, 1, 1), stamp=other)
    # A rejected phase does not poison these immutable, stateless inputs.
    valid = map_area_pressure_normal_forces(rows, (1, 1, 1), stamp=quad.stamp)
    np.testing.assert_allclose(valid.wrench_a[:3], (0, 0, 3), rtol=RTOL, atol=ATOL)


def test_sample_order_cannot_be_silently_repaired_and_dof_names_are_unique():
    quad = _quad()
    kin = _fixed(quad)
    with pytest.raises(ValueError):
        assemble_area_pressure_rows(quad, replace(kin, sample_ids=tuple(reversed(quad.sample_ids))))
    with pytest.raises(ValueError):
        assemble_area_pressure_rows(quad, replace(kin, dof_ids=("same", "same", "other")))
    with pytest.raises(ValueError):
        assemble_area_pressure_rows(quad, replace(kin, dof_ids=("x", "", "z")))
    with pytest.raises(ValueError):
        assemble_area_pressure_rows(quad, replace(kin, dof_ids=("x", "y", "z" * 129)))


@pytest.mark.parametrize("field,value", [
    ("points", np.zeros((2, 3))), ("normals", np.tile((0, 0, 2), (3, 1))),
    ("dA", np.array((-1.0, 1 / 3, 1 / 3))), ("pressure", np.array((1, np.nan, 2))),
    ("gradient", np.array((5.0, 0.0, 5.0))),
    ("sample_ids", (("triangle-0", 0),) * 3),
    ("source_ids", (("cell-4", "face-9"),)),
])
def test_assembler_revalidates_forged_quadrature(field, value):
    quad = _quad()
    with pytest.raises(ValueError):
        assemble_area_pressure_rows(_forge(quad, **{field: value}), _fixed(quad))


@pytest.mark.parametrize("field,value", [
    ("jacobian_a", np.zeros((3, 3))), ("jacobian_b", np.full((3, 3, 3), np.nan)),
    ("dof_ids", ("dup", "dup", "z")),
])
def test_assembler_revalidates_forged_kinematics(field, value):
    quad = _quad()
    with pytest.raises(ValueError):
        assemble_area_pressure_rows(quad, _forge(_fixed(quad), **{field: value}))


@pytest.mark.parametrize("field,value", [
    ("normals", np.zeros((3, 3))), ("jacobian_b", np.full((3, 3, 3), np.nan)),
    ("stiffness", np.full(3, np.inf)), ("source_ids", (("cell-4", "face-9"),)),
])
def test_force_map_revalidates_forged_rows(field, value):
    quad = _quad()
    rows = assemble_area_pressure_rows(quad, _fixed(quad))
    with pytest.raises(ValueError):
        map_area_pressure_normal_forces(_forge(rows, **{field: value}), (1, 2, 3), stamp=quad.stamp)


@pytest.mark.parametrize("forces", [
    (-1, 2, 3), (1, np.nan, 3), (1, 2), ((1,), (2,), (3,)),
])
def test_force_preflight_is_atomic_and_does_not_broadcast(forces):
    quad = _quad()
    rows = assemble_area_pressure_rows(quad, _fixed(quad))
    before = rows.normal_jacobian.tobytes(), rows.elastic_force.tobytes()
    with pytest.raises(ValueError):
        map_area_pressure_normal_forces(rows, forces, stamp=quad.stamp)
    assert before == (rows.normal_jacobian.tobytes(), rows.elastic_force.tobytes())
    loads = map_area_pressure_normal_forces(rows, (2, 3, 7), stamp=quad.stamp)
    np.testing.assert_allclose(loads.wrench_a, (0, 0, 12, 5.5, -7, 0), rtol=RTOL, atol=ATOL)


def test_dof_and_jacobian_element_caps_are_independent():
    quad = _quad()
    rows = assemble_area_pressure_rows(quad, _fixed(quad, dofs=256))
    assert rows.normal_jacobian.shape == (3, 256)
    for dofs in (0, 257):
        with pytest.raises(ValueError):
            assemble_area_pressure_rows(quad, _fixed(quad, dofs=dofs))
    # 434*3*3*256=999936 entries per J is allowed; 435 gives 1002240.
    below = _quad(tuple(_triangle(index) for index in range(434)))
    below_rows = assemble_area_pressure_rows(below, _fixed(below, dofs=256))
    assert below_rows.normal_jacobian.shape == (1302, 256)
    above = _quad(tuple(_triangle(index) for index in range(435)))
    with pytest.raises(ValueError):
        assemble_area_pressure_rows(above, _fixed(above, dofs=256))


def test_finite_inputs_with_nonfinite_derived_rows_or_loads_are_rejected():
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        with pytest.raises(ValueError):
            quad = _quad((_triangle(vertices=((0, 0, 0), (8, 0, 0), (0, 3, 0)),
                                    vertex_pressure=(1, 1, 1), pressure_gradient=(0, 0, 1e308)),))
            assemble_area_pressure_rows(quad, _fixed(quad))
        with pytest.raises(ValueError):
            quad = _quad((_triangle(vertex_pressure=(1e308,) * 3,
                                    pressure_gradient=(0, 0, 1e-308)),))
            assemble_area_pressure_rows(quad, _fixed(quad))
        quad = _quad()
        rows = assemble_area_pressure_rows(quad, _fixed(quad))
        with pytest.raises(ValueError):
            map_area_pressure_normal_forces(rows, (1e308, 1e308, 1e308), stamp=quad.stamp)
    valid = map_area_pressure_normal_forces(rows, (1, 1, 1), stamp=quad.stamp)
    np.testing.assert_allclose(valid.wrench_a[:3], (0, 0, 3), rtol=RTOL, atol=ATOL)
