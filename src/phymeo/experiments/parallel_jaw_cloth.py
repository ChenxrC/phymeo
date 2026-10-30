"""Finite-force gripper configurations and unmodified public-runtime records.

This module contains no trajectory oracle. Case names select physical input
configuration only; the shared runtime determines contact, motion and rejection.
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np

import phymeo as ph
from phymeo.builders.parallel_jaw_gripper import gripper_inputs, make_parallel_jaw_gripper
from phymeo.core.scene import fingerprint
from phymeo.core.validation import finite_real
from phymeo.runtime.program import own_diagnostics


@dataclass(frozen=True)
class GripperClothCase:
    scene: ph.SceneModel
    initial: ph.SceneState
    inputs: ph.SceneInputs
    program: object
    settings: ph.StepSettings
    context: ph.BackendContext
    parameters: Mapping
    h: float


@dataclass(frozen=True)
class GripperClothRun:
    scene: ph.SceneModel
    initial: ph.SceneState
    inputs: ph.SceneInputs
    settings: ph.StepSettings
    parameters: Mapping
    program_signature: object
    h: float
    requested_steps: int
    records: tuple

    @property
    def accepted(self):
        return len(self.records) == self.requested_steps and all(
            item.report.accepted for item in self.records
        )

    @property
    def final_state(self):
        return self.records[-1].state if self.records else self.initial


def _positive(value, name):
    value = finite_real(value, name)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def build_gripper_case(
    case,
    *,
    mode,
    h,
    mass_scale=1.0,
    patch_half_extent_m=(4.0, 4.0),
    vertical_shift_m=0.0,
    reverse_declarations=False,
):
    """Build an owned finite-grasp input bundle; do not execute a timestep.

    Modes are explicit. Existing native participants own their normal cache
    policy; this adapter neither changes the environment nor substitutes state.
    reverse_declarations changes only scene/patch declaration order.
    """
    if not isinstance(case, str) or case not in ("hold", "slip", "capture", "material"):
        raise ValueError("unknown gripper case")
    if not isinstance(mode, str) or mode not in ("numpy", "interpret", "cuda", "cpu_cpp"):
        raise ValueError("mode must select numpy or an explicit supported tyx backend")
    if type(reverse_declarations) is not bool:
        raise TypeError("reverse_declarations must be boolean")
    h = _positive(h, "h")
    scale = _positive(mass_scale, "mass_scale")
    shift = finite_real(vertical_shift_m, "vertical_shift_m")
    gravity = (0.0, 0.0, 0.0 if case == "material" else -1.0)
    half_gap = 0.125 + (0.0625 if case == "capture" else 0.0)
    mu = {"hold": 1.0, "slip": 0.25, "capture": 2.0, "material": 0.0}[case]
    lift = {"hold": 9.0, "slip": 9.0, "capture": 12.0, "material": 0.0}[case]
    gripper = make_parallel_jaw_gripper(
        half_gap_m=half_gap,
        patch_half_extent_m=patch_half_extent_m,
        carriage_mass_kg=2.0 * scale,
        finger_mass_kg=2.0 * scale,
        inertia_diagonal_kg_m2=np.full((3, 3), scale),
        effort_limits_N=np.array([16.0, 4.0, 4.0]) * scale,
        gravity_m_s2=gravity,
    )
    nodes = ph.ParticleModel(np.full(3, 1.0 / scale), np.array(gravity))
    constraints = None
    if case == "material":
        constraints = ph.DistanceConstraints(
            np.array([[0, 2]], dtype=np.int64),
            np.array([2.0 - 1.0 / 64.0]),
            np.array([1.0 / (16.0 * scale)]),
        )
    domains = (
        ph.DomainBinding("robot", "robot", gripper.model),
        ph.DomainBinding("nodes", "particles", nodes),
    )
    scene = ph.SceneModel(domains[::-1] if reverse_declarations else domains)
    position = np.array([[-1.0, 0.0, -1.0], [0.0, 0.0, 2.0], [1.0, 0.0, -1.0]])
    position[:, 2] += shift
    initial = ph.SceneState(
        {
            "robot": ph.RobotState(np.array([shift, 0.0, 0.0]), np.zeros(3)),
            "nodes": ph.ParticleState(position, np.zeros((3, 3))),
        }
    )
    inputs = ph.SceneInputs(
        {
            "robot": gripper_inputs(
                gripper,
                lift_effort_N=lift * scale,
                closing_effort_N=np.full(2, 2.0 * scale),
            ),
            "nodes": ph.StepInputs.zeros(3),
        }
    )
    if mode == "numpy":
        participants = {
            "robot": ph.RobotParticipant(gripper.model),
            "nodes": ph.ClothParticipant(nodes, constraints, record_material_path=True),
        }
        base_factory = ph.ArticulatedCouplingProgram
    else:
        participants = {
            "robot": ph.TyxRobotParticipant(gripper.model, mode),
            "nodes": ph.TyxClothParticipant(nodes, constraints, mode, record_material_path=True),
        }
        base_factory = ph.TyxArticulatedCouplingProgram
    base = base_factory(
        scene,
        participants,
        ph.RobotSphereAttachments(np.empty(0, dtype=np.int64), np.empty((0, 3)), np.empty(0)),
        (),
    )
    patches = gripper.patches[::-1] if reverse_declarations else gripper.patches
    budget = 1e-8 if mode == "numpy" else 1e-5
    program = ph.DynamicRobotThickContactProgram(
        scene,
        base,
        (ph.TrianglePrismSurface("sheet", "nodes", (0, 1, 2), half_thickness_m=0.125),),
        patches,
        friction_by_collider={patch.key: mu for patch in patches},
        materials={"nodes": constraints},
        budgets=ph.MaterialContactBudgets(budget, budget),
    )
    initial = own_diagnostics(
        ph.SceneState(
            initial.domains,
            {field.name: np.zeros(field.shape, field.dtype) for field in program.history_fields},
        )
    )
    inputs = own_diagnostics(inputs)
    settings = ph.StepSettings(
        substeps=1,
        position_iterations=512,
        velocity_iterations=512,
        tolerance=1e-8 if mode == "numpy" else 2e-5,
        record_substeps=True,
    )
    parameters = own_diagnostics(
        {
            "case": case,
            "mode": mode,
            "mass_scale": scale,
            "patch_half_extent_m": np.array(patch_half_extent_m, dtype=np.float64),
            "vertical_shift_m": shift,
            "reverse_declarations": reverse_declarations,
            "friction": mu,
            "half_thickness_m": 0.125,
            "initial_half_gap_m": half_gap,
            "lift_effort_N": lift * scale,
            "closing_effort_N": np.full(2, 2.0 * scale),
            "max_energy_gain_j": budget,
            "max_material_angular_defect_nms": budget,
        }
    )
    return GripperClothCase(
        scene, initial, inputs, program, settings,
        ph.BackendContext(program.capabilities.backend), parameters, h,
    )


def run_gripper_case(case, *, mode, h, duration_s, **configuration):
    """Record each actual step, stopping at the first rejection without repair."""
    h = _positive(h, "h")
    duration = _positive(duration_s, "duration_s")
    count = Fraction.from_float(duration) / Fraction.from_float(h)
    if count.denominator != 1:
        raise ValueError("duration_s must be an exact integral number of stored h values")
    if case == "material" and count != 1:
        raise ValueError("the material control is exactly one step")
    problem = build_gripper_case(case, mode=mode, h=h, **configuration)
    original = fingerprint((problem.initial, problem.inputs))
    signature = own_diagnostics(problem.program.problem_signature())
    state = problem.initial
    records = []
    with ph.SceneStepper(
        problem.scene, problem.program, problem.settings, problem.context, problem.program.registry
    ) as runtime:
        for _ in range(count.numerator):
            with np.errstate(all="raise"):
                result = runtime.step(state, problem.inputs, h)
            records.append(own_diagnostics(result))
            if not result.report.accepted:
                break
            state = result.state
    problem.scene.assert_unchanged()
    if fingerprint((problem.initial, problem.inputs)) != original:
        raise RuntimeError("runtime changed the original experiment snapshot")
    return GripperClothRun(
        problem.scene, problem.initial, problem.inputs, problem.settings,
        problem.parameters, signature, h, count.numerator, tuple(records),
    )


def _encode(value):
    """Lossless typed JSON representation; raw timing observations are retained."""
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return {
            "type": "ndarray", "dtype": array.dtype.str,
            "shape": list(array.shape), "bytes_hex": array.tobytes().hex(),
        }
    if isinstance(value, np.generic):
        return _encode(np.asarray(value))
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "type": type(value).__module__ + "." + type(value).__qualname__,
            "fields": {field.name: _encode(getattr(value, field.name)) for field in fields(value)},
        }
    if isinstance(value, Mapping):
        return {key: _encode(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_encode(item) for item in value]
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        return {"type": "float", "hex": value.hex()}
    raise TypeError(f"unsupported artifact value: {type(value).__name__}")


def save_gripper_run(run, destination, *, source):
    """Save full actual records and provenance without overwriting evidence."""
    if not isinstance(run, GripperClothRun):
        raise TypeError("run must be GripperClothRun")
    if not isinstance(source, Mapping):
        raise TypeError("source provenance must be a mapping")
    run.scene.assert_unchanged()
    payload = {
        "schema": "phymeo-finite-gripper-run-v1",
        "accepted": run.accepted,
        "source": _encode(source),
        "run": _encode(run),
    }
    serialized = json.dumps(payload, indent=2, allow_nan=False)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    with (destination / "run.json").open("x", encoding="utf-8") as output:
        output.write(serialized + "\n")
    return destination