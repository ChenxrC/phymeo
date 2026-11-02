"""A compliant elastic sheet receives self loads in the same substep as its material forces."""

from itertools import combinations

import numpy as np
import pytest

import phymeo as ph
from phymeo.constraints.bending import BendingConstraints, bending_angle_gradient
from phymeo.constraints.cloth_material import ClothMaterial


def setup_sheet(
    *, active=True, native=None, mode="host", hinge_cross=False, layers=None, backed=False
):
    if hinge_cross:
        x = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.5, -0.5]])
        faces = ((0, 1, 2), (1, 0, 3))
        mat = ClothMaterial()
    else:
        bottom = np.array([[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 2.0, 0.0]])
        x = np.concatenate((bottom, bottom + [0.0, 0.0, 0.125]))
        faces = ((0, 1, 2), (3, 5, 4), (1, 0, 3), (1, 3, 4))
        ids = np.array([[0, 1, 2, 3]])
        rest = bending_angle_gradient(x[ids[0]])[0] + 0.005
        mat = ClothMaterial(
            ph.DistanceConstraints(np.array([[0, 1]]), np.array([2.0]), np.array([1.0])),
            BendingConstraints(ids, np.array([rest]), np.array([1.0])),
        )
    model = ph.ParticleModel(np.ones(len(x)), np.zeros(3))
    scene = ph.SceneModel((ph.DomainBinding("sheet", "particles", model),))
    surfaces = tuple(
        ph.TrianglePrismSurface("face_" + str(i), "sheet", face, half_thickness_m=1 / 32)
        for i, face in enumerate(faces)
    )
    layer = ph.NormalLayer(3 / 16, 16.0) if layers is None else layers
    kwargs = (
        dict(cloth_self_layer=layer, cloth_self_operator=native, cloth_self_geometry=mode)
        if active
        else {}
    )
    patches = ()
    if backed:
        boundary = np.array(
            [[-3.0, -3.0, -1 / 32], [3.0, -3.0, -1 / 32], [3.0, 3.0, -1 / 32], [-3.0, 3.0, -1 / 32]]
        )
        patches = (
            ph.CompliantPatch(
                ph.RobotContactPatch("table", 0, boundary),
                width_m=0.02,
                stiffness_pa_per_m=100.0,
                radius_m=0.01,
                backing_clearance_m=0.0025,
            ),
        )
    program = ph.CompliantClothContactProgram(scene, "sheet", mat, surfaces, x, patches, **kwargs)
    v = np.zeros_like(x)
    if hinge_cross:
        v[3, 2] = 1.5 * 64
    state = ph.SceneState({"sheet": ph.ParticleState(x, v)})
    inputs = ph.SceneInputs({"sheet": ph.StepInputs.zeros(len(x))})
    return program, state, inputs


def test_real_material_and_two_self_faces_share_one_particle_state():
    on, s, inputs = setup_sheet()
    off, _, _ = setup_sheet(active=False)
    sample = on.observe_self_contact(s)
    (contact,) = sample["contacts"]
    assert contact["faces"] == ("face_0", "face_1")
    assert contact["force_n"] == pytest.approx(2.0) and len(contact["sites"]) == 3
    assert contact["middle_gap_m"] == pytest.approx(0.125)
    np.testing.assert_allclose(sample["nodal_force_n"][:, 2], [-2 / 3] * 3 + [2 / 3] * 3)
    a_on, _, internal, _ = on.accelerations(s, inputs)
    a_off, _, base_internal, _ = off.accelerations(s, inputs)
    np.testing.assert_array_equal(internal, base_internal)
    assert np.linalg.norm(internal) > 0.001
    np.testing.assert_allclose(a_on["sheet"] - a_off["sheet"], sample["nodal_force_n"], atol=1e-13)
    with on.stepper(
        ph.StepSettings(substeps=1, position_iterations=1, velocity_iterations=1)
    ) as rt:
        out = rt.step(s, inputs, 1 / 256)
    assert out.report.accepted, out.report.reason
    info = out.report.substeps[0].diagnostics["self_contact"]
    assert info["contacts"][0]["force_n"] > 0 and info["geometry_backend"] == "host"
    assert set(out.state.domains) == {"sheet"}
    np.testing.assert_allclose(out.state.domains["sheet"].velocity.sum(0), 0, atol=1e-12)


def test_incident_remote_crossing_rejects_complete_frame_with_zero_initial_self_load():
    on, s, inputs = setup_sheet(hinge_cross=True)
    assert on.observe_self_contact(s)["contacts"] == ()
    with on.stepper(
        ph.StepSettings(substeps=1, position_iterations=1, velocity_iterations=1)
    ) as rt:
        out = rt.step(s, inputs, 1 / 64)
    assert not out.report.accepted and "core path" in out.report.reason
    np.testing.assert_array_equal(out.state.domains["sheet"].position, s.domains["sheet"].position)
    np.testing.assert_array_equal(out.state.domains["sheet"].velocity, s.domains["sheet"].velocity)


def test_pair_calibration_requires_every_pair_and_binds_the_signature():
    names = ["face_" + str(i) for i in range(4)]
    law = ph.NormalLayer(3 / 16, 16.0)
    layers = {pair: law for pair in combinations(names, 2)}
    a, s, _ = setup_sheet(layers=layers)
    different = {**layers, ("face_0", "face_1"): ph.NormalLayer(3 / 16, 32.0)}
    b, _, _ = setup_sheet(layers=different)
    assert b.observe_self_contact(s)["contacts"][0]["force_n"] == pytest.approx(4.0)
    assert (
        a.problem_signature()["cloth_self_contact"] != b.problem_signature()["cloth_self_contact"]
    )
    del layers[("face_2", "face_3")]
    with pytest.raises(ValueError, match="every|complete"):
        setup_sheet(layers=layers)


def test_backed_drift_uses_the_same_self_and_external_table_force_sample():
    program, state, inputs = setup_sheet(backed=True)
    h = 1 / 256
    with program.stepper(
        ph.StepSettings(substeps=1, position_iterations=1, velocity_iterations=1)
    ) as rt:
        out = rt.step(state, inputs, h)
    assert out.report.accepted, out.report.reason
    diag = out.report.substeps[0].diagnostics
    assert diag["force_sample"] == "substep-entry"
    assert diag["self_contact"]["contacts"][0]["force_n"] == pytest.approx(2.0)
    external = np.sum(diag["patches"]["table"]["force_n"], axis=0)
    assert external[2] > 0
    np.testing.assert_allclose(out.state.domains["sheet"].velocity.sum(0), h * external, atol=1e-11)
