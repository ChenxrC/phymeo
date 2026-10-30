"""Frozen rational A/B/C controls: full windows, fresh rebuilds and covariance.

These fixtures/oracles were written before execution. No production normal,
Jacobian, response matrix or pressure solve constructs an expected answer.
"""

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from hashlib import sha256
import json

import numpy as np
import pytest

from phymeo.constraints.cloth_self_contact import ClothSelfContactTopology
from phymeo.constraints.cloth_self_response import SelfContactResponseLaw, SelfContactResponseLimits, SelfContactBudgets
from phymeo.constraints.distance import DistanceConstraints
from phymeo.core.model import ParticleModel
from phymeo.core.state import ParticleState, StepInputs
from phymeo.core.scene import SceneModel, DomainBinding, SceneState, SceneInputs, StepSettings, BackendContext, fingerprint
from phymeo.coupling.participant import ClothParticipant
from phymeo.runtime.point_participant_program import PointParticipantProgram
from phymeo.runtime.registry import StateRegistry
from phymeo.runtime.scene_stepper import SceneStepper
from phymeo.solvers.cloth_self_contact import ClothSelfContactProgram

ROTATION = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
TRANSLATION = np.array([4., -2., 1.])


def serialize(value):
    if isinstance(value, np.ndarray):
        return dict(dtype=value.dtype.str, shape=value.shape, values=value.tolist(), bytes_sha256=sha256(value.tobytes()).hexdigest())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): serialize(item) for key, item in value.items()}
    if is_dataclass(value):
        return dict(type=type(value).__module__+"."+type(value).__qualname__, fields={f.name: serialize(getattr(value, f.name)) for f in fields(value)})
    if isinstance(value, (tuple, list)):
        return [serialize(item) for item in value]
    return value


def save_evidence(path, *, result, attempts, inputs, label):
    payload = serialize(dict(label=label, result=result, attempts=attempts, original_inputs=inputs))
    path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")), encoding="utf8")
    path.with_suffix(".sha256").write_text(sha256(path.read_bytes()).hexdigest()+"  "+path.name+"\n", encoding="ascii")


def fixture(case, *, material=False, u=35/64, variant="identity"):
    if case == "A":
        xy = np.array([[0., 0.], [3., 0.], [0., 4.]])
        x = np.vstack((np.column_stack((xy, np.full(3, 1/8))), np.column_stack((xy, np.full(3, -1/4)))))
        v = np.zeros_like(x)
        w, rho = np.array([.5]*3+[.25]*3), np.array([1/8, 1/4])
        force = np.array([[0., 0., -2.]]*3+[[0., 0., 2.]]*3)
        edges = np.array([[0, 1], [1, 2], [2, 0], [3, 4], [4, 5], [5, 3]], np.int64)
        law_material = DistanceConstraints(edges, np.array([3., 5., 4.]*2), np.full(6, 1/16)) if material else None
        duration = 1/4
    else:
        xy = np.array([[-1., -1.], [1., -1.], [0., 1.]])
        if case == "B":
            duration = 1/16
            xa = np.column_stack((xy, np.full(3, .25+duration)))
            xb = np.column_stack((xy, np.full(3, -.25-duration)))
            xa[:, 0] -= u*duration/2
            xb[:, 0] += u*duration/2
            x = np.vstack((xa, xb))
            v = np.array([[u/2, 0., -1.]]*3+[[-u/2, 0., 1.]]*3)
        else:
            duration = 1/64
            x = np.vstack([np.column_stack((xy, np.full(3, z))) for z in (0., -.5, .5)])
            v = np.array([[0., 0., 0.]]*3+[[0., 0., 1.]]*3+[[0., 0., -1.]]*3)
        w, rho, force, law_material = np.ones(len(x)), np.full(len(x)//3, .25), np.zeros_like(x), None
    faces = np.arange(len(x), dtype=np.int64).reshape(-1, 3)
    rotation, translation = np.eye(3), np.zeros(3)
    if variant == "rigid":
        rotation, translation = ROTATION, TRANSLATION
        x, v, force = x @ rotation.T+translation, v @ rotation.T, force @ rotation.T
    elif variant == "exchange_cycle":
        faces, rho = np.roll(faces[::-1], 1, axis=1).copy(), rho[::-1].copy()
    model = ParticleModel(w, np.zeros(3))
    return dict(case=case, model=model, state=ParticleState(x, v), inputs=StepInputs(force),
                topology=ClothSelfContactTopology("cloth", faces, rho), material=law_material,
                duration=duration, u=u, rotation=rotation, translation=translation)


def configured(data, h, *, backend="numpy", outer_substeps=1):
    model, material = data["model"], data["material"]
    scene = SceneModel((DomainBinding("cloth", "particles", model),))
    registry = StateRegistry()
    if backend == "numpy":
        factory = ClothParticipant(model, material, record_material_path=True)
    else:
        from phymeo.backends.tyx_participants import TyxClothParticipant
        factory = TyxClothParticipant(model, material, backend.removeprefix("tyx:"), record_material_path=True)
    base = PointParticipantProgram(scene, {"cloth": factory}, registry=registry)
    tau, physical = (1e-8, 1e-8) if backend == "numpy" else (2e-5, 1e-5)
    inner = StepSettings(substeps=1, position_iterations=512, velocity_iterations=512, tolerance=tau,
                         reference_length=1., reference_speed=1., record_substeps=True)
    budgets = SelfContactBudgets(0., 2.**-40, tau, tau, physical, physical)
    program = ClothSelfContactProgram(scene, base, data["topology"],
        law=SelfContactResponseLaw(.5, .5, normal_domain="unique_exposed_ray_v1"), materials=material,
        budgets=budgets, limits=SelfContactResponseLimits(), registry=registry, inner_settings=inner)
    state = SceneState({"cloth": data["state"]}, program.initial_history())
    inputs = SceneInputs({"cloth": data["inputs"]})
    from dataclasses import replace
    runtime = SceneStepper(scene, program, replace(inner, substeps=outer_substeps), BackendContext(backend), registry)
    return runtime, program, state, inputs


def kinetic(model, state):
    return float(np.sum(state.velocity**2/model.inverse_mass[:, None])/2)


def angular(model, state):
    return np.sum(np.cross(state.position, state.velocity/model.inverse_mass[:, None]), axis=0)


def run_window(data, h, tmp_path, *, backend="numpy"):
    runtime, program, state, inputs = configured(data, h, backend=backend)
    original_token = fingerprint((state, inputs))
    initial = state
    records = []
    held_total = 0.
    try:
        steps = int(round(data["duration"]/h))
        for index in range(steps):
            before_token = fingerprint((state, inputs))
            result = runtime.step(state, inputs, h)
            save_evidence(tmp_path/f"step-{index:04d}.json", result=result,
                          attempts=program.last_attempt_receipts, inputs=inputs, label=dict(case=data["case"], backend=backend, h=h, index=index))
            assert fingerprint((state, inputs)) == before_token
            assert result.report.accepted, result.report.reason
            record = result.report.substeps[0].diagnostics["self_contact"]
            records.append(record)
            held_total += sum(row["normal_ns"] for row in record["physical"]["held_rows"])
            assert record["physical"]["positive_energy_upper_j"] <= program.budgets.energy_j
            assert record["physical"]["angular_absolute_nms"] <= program.budgets.angular_impulse_nms
            assert record["native_fallback"] is False
            state = result.state
        assert fingerprint((initial, inputs)) == original_token
        assert state.history[program.history_prefix+"/window_positive"][0] <= program.budgets.energy_j
        assert state.history[program.history_prefix+"/window_positive"][1] <= program.budgets.angular_impulse_nms
        return initial, state, tuple(records), held_total, program.budgets
    finally:
        runtime.close()
        data_path = tmp_path/"last-attempt.json"
        data_path.write_text(json.dumps(serialize(program.last_attempt_receipts), sort_keys=True), encoding="utf8")


def assert_a(data, result, h):
    initial, final, records, held_total, budgets = result
    a, b = initial.domains["cloth"], final.domains["cloth"]
    np.testing.assert_allclose(b.position, a.position, atol=budgets.velocity_mps*h, rtol=0)
    np.testing.assert_allclose(b.velocity, 0., atol=budgets.velocity_mps, rtol=0)
    assert abs(held_total-1.5) <= budgets.impulse_ns
    for record in records:
        np.testing.assert_allclose([row["normal_ns"] for row in record["physical"]["held_rows"]], 2*h, atol=budgets.impulse_ns, rtol=0)
        np.testing.assert_allclose(record["terminal_plan"].node_impulses, 0., atol=budgets.impulse_ns, rtol=0)
        assert record["replay"]["material_path"]["schema"] == "xpbd-distance-path-v1"
        assert len(record["replay"]["material_path"]["frames"]) >= 1
    assert abs(kinetic(data["model"], b)) <= budgets.energy_j


def assert_b(data, result, t, slip, delta_energy):
    initial, final, records, held_total, budgets = result
    a, b = initial.domains["cloth"], final.domains["cloth"]
    assert held_total <= budgets.impulse_ns
    for record in records[:-1]:
        np.testing.assert_allclose(record["terminal_plan"].node_impulses, 0., atol=budgets.impulse_ns, rtol=0)
    g = np.array([[-.5, -.25], [.5, -.25], [0., .5]])
    expected = np.zeros((6, 3))
    expected[:3, 0], expected[3:, 0] = data["u"]/2+t, -data["u"]/2-t
    expected[:3, 2] = 3*.25*t*g[:, 0]
    expected[3:, 2] = 3*.25*t*g[:, 0]
    expected = expected @ data["rotation"].T
    np.testing.assert_allclose(b.velocity, expected, atol=budgets.velocity_mps, rtol=0)
    assert abs(kinetic(data["model"], b)-kinetic(data["model"], a)-delta_energy) <= budgets.energy_j
    np.testing.assert_allclose(angular(data["model"], b), angular(data["model"], a), atol=budgets.angular_impulse_nms, rtol=0)
    np.testing.assert_allclose(angular(data["model"], b), np.array([0., 3*.25*data["u"], 0.]) @ data["rotation"].T, atol=budgets.angular_impulse_nms, rtol=0)
    rows = records[-1]["physical"]["terminal_rows"]
    np.testing.assert_allclose([row["normal_ns"] for row in rows], 1., atol=budgets.impulse_ns, rtol=0)
    # Independently reconstruct the material-point tangent speed from nodal
    # translation plus rho times the normal derivative, using explicit g.
    local = b.velocity @ data["rotation"]
    dn_a = -np.array([np.dot(g[:, j], local[:3, 2]) for j in range(2)])
    dn_b = -np.array([np.dot(g[:, j], local[3:, 2]) for j in range(2)])
    relative = local[:3, :2]-.25*dn_a-(local[3:, :2]+.25*dn_b)
    np.testing.assert_allclose(relative[:, 0], slip, atol=budgets.velocity_mps, rtol=0)
    if t == -.25:
        np.testing.assert_allclose(local[:3, 0].mean(), 3/128, atol=budgets.velocity_mps, rtol=0)
        np.testing.assert_allclose(local[3:, 0].mean(), -3/128, atol=budgets.velocity_mps, rtol=0)


def assert_c(data, result):
    initial, final, records, _, budgets = result
    a, b = initial.domains["cloth"], final.domains["cloth"]
    np.testing.assert_allclose(b.velocity, 0., atol=budgets.velocity_mps, rtol=0)
    np.testing.assert_allclose(b.position, a.position, atol=budgets.velocity_mps/64, rtol=0)
    np.testing.assert_allclose(records[0]["held_plan"].node_impulses[:3], 0., atol=budgets.impulse_ns, rtol=0)
    np.testing.assert_allclose([row["normal_ns"] for row in records[0]["physical"]["held_rows"]], 1., atol=budgets.impulse_ns, rtol=0)
    assert abs(kinetic(data["model"], b)-kinetic(data["model"], a)+3) <= budgets.energy_j


@pytest.mark.parametrize("h", [1/64, 1/128, 1/256])
@pytest.mark.parametrize("material", [False, True])
@pytest.mark.parametrize("rebuild", [0, 1])
@pytest.mark.parametrize("variant", ["identity", "rigid", "exchange_cycle"])
def test_a_sustained_pressure_full_window(h, material, rebuild, variant, tmp_path):
    data = fixture("A", material=material, variant=variant)
    assert_a(data, run_window(data, h, tmp_path), h)


@pytest.mark.parametrize("h", [1/16, 1/32, 1/64])
@pytest.mark.parametrize("u,t,slip,delta", [(35/64, -1/4, 0., -3-105/512), (35/16, -1/2, 35/32, -3-315/128)])
@pytest.mark.parametrize("rebuild", [0, 1])
@pytest.mark.parametrize("variant", ["identity", "rigid", "exchange_cycle"])
def test_b_forcefree_terminal_stick_slip(h, u, t, slip, delta, rebuild, variant, tmp_path):
    data = fixture("B", u=u, variant=variant)
    assert_b(data, run_window(data, h, tmp_path), t, slip, delta)


@pytest.mark.parametrize("rebuild", [0, 1])
@pytest.mark.parametrize("variant", ["identity", "rigid", "exchange_cycle"])
def test_c_shared_base_cross_response(rebuild, variant, tmp_path):
    data = fixture("C", variant=variant)
    assert_c(data, run_window(data, 1/64, tmp_path))
