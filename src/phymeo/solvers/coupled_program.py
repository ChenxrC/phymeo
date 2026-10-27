"""Phased distance/vertex-sphere programs and a legacy-shaped result adapter."""

import numpy as np

from ..constraints.distance import distance_residual, project_distances
from ..constraints.particle_sphere import (
    particle_sphere_residual,
    particle_sphere_velocity_residual,
    project_particle_sphere_velocities,
    project_particle_spheres,
)
from ..core.capabilities import SolverFailure
from ..core.coupled import (
    CoupledInputs,
    CoupledState,
    CoupledStepReport,
    CoupledStepResult,
    CoupledSubstepReport,
)
from ..core.scene import (
    BackendContext,
    DomainBinding,
    SceneInputs,
    SceneModel,
    SceneState,
    StageMetrics,
    StepSettings,
)
from ..geometry.particle_sphere import particle_sphere_geometry
from ..runtime.program import ProgramCapabilities
from ..runtime.scene_stepper import SceneStepper
from .colored_coupled import ColoredCoupledXPBDSolver
from .coupled_xpbd import CoupledXPBDSolver, _impulse_ledger, _reconstruct
from .rigid_semi_implicit import RigidSemiImplicitSolver
from .semi_implicit import SemiImplicitSolver


def coupled_signature(config):
    """A distance multiset is physical; its color order is an execution choice."""
    rows = (
        ()
        if config.constraints is None
        else tuple(
            sorted(
                (int(i), int(j), float(rest), float(compliance))
                for (i, j), rest, compliance in zip(
                    config.constraints.indices,
                    config.constraints.rest_length,
                    config.constraints.compliance,
                    strict=True,
                )
            )
        )
    )
    return {
        "law": "xpbd-distance-vertex-sphere-frictionless-v1",
        "distance": rows,
        "spheres": config.spheres.radii,
    }


def coupled_scene(config):
    return SceneModel(
        (
            DomainBinding("particles", "particles", config.particle_model),
            DomainBinding("rigid", "rigid_bodies", config.rigid_model),
        )
    )


class NumpyCoupledProgram:
    capabilities = ProgramCapabilities(
        "numpy",
        {"float32", "float64"},
        {"particles", "rigid_bodies"},
        {"distance", "point_sphere", "rotation"},
    )
    history_fields = ()

    def __init__(self, particle_model, rigid_model, constraints, spheres, *, colored=False):
        if not isinstance(colored, (bool, np.bool_)):
            raise TypeError("colored must be boolean")
        # Reuse physical validation and stable coloring, never the old time loop.
        cls = ColoredCoupledXPBDSolver if colored else CoupledXPBDSolver
        self._config = cls(particle_model, rigid_model, constraints, spheres)
        self.scene = coupled_scene(self._config)

    def problem_signature(self):
        return coupled_signature(self._config)

    def create_workspace(self, scene, state, inputs, h, settings, context):
        self.scene.assert_unchanged()
        if scene.identity != self.scene.identity:
            raise ValueError("coupled program must use its bound physical models")
        return _NumpyCoupledWorkspace(self._config, state, inputs, h, settings)

    def close(self):
        pass


class _NumpyCoupledWorkspace:
    def __init__(self, config, state, inputs, h, settings):
        self.config, self.h, self.settings = config, h, settings
        self.initial = CoupledState(state.domains["particles"], state.domains["rigid"])
        self.inputs = inputs
        self.wp, self.wb = config.particle_model.inverse_mass, config.rigid_model.inverse_mass
        self.position_transfers, self.velocity_transfers = [], []

    def predict(self):
        config = self.config
        p = SemiImplicitSolver().step(
            config.particle_model, self.initial.particles, self.inputs.domains["particles"], self.h
        )
        b = RigidSemiImplicitSolver().step(
            config.rigid_model, self.initial.rigid, self.inputs.domains["rigid"], self.h
        )
        if not p.report.accepted or not b.report.accepted:
            raise SolverFailure("non-finite coupled free prediction")
        self.predicted = CoupledState(p.state, b.state)
        self.candidate = CoupledState(p.state, b.state)
        constraints = config.constraints
        count = 0 if constraints is None else len(constraints.indices)
        self.alpha, self.edge_lambda = np.zeros(count), np.zeros(count)
        self.contact_lambda = np.zeros(len(self.wp) * len(self.wb))
        if constraints is not None:
            np.divide(
                constraints.compliance,
                np.float64(self.h),
                out=self.alpha,
                where=constraints.compliance > 0,
            )
            np.divide(self.alpha, np.float64(self.h), out=self.alpha)

    def position_sweep(self):
        constraints = self.config.constraints
        if constraints is not None:
            project_distances(
                self.candidate.particles.position,
                self.wp,
                constraints,
                self.alpha,
                self.edge_lambda,
            )
        self.position_transfers.append(
            project_particle_spheres(
                self.candidate.particles.position,
                self.candidate.rigid.position,
                self.wp,
                self.wb,
                self.config.spheres,
                self.contact_lambda,
            )
        )

    def position_metrics(self):
        p, b, constraints = self.candidate.particles, self.candidate.rigid, self.config.constraints
        distance, geometric = (
            (0.0, 0.0)
            if constraints is None
            else distance_residual(
                p.position,
                constraints,
                self.alpha,
                self.edge_lambda,
                self.settings.reference_length,
            )
        )
        contact = particle_sphere_residual(
            p.position,
            b.position,
            self.wp,
            self.wb,
            self.config.spheres,
            self.contact_lambda,
            reference_length=self.settings.reference_length,
        )
        return StageMetrics({"distance": distance, "contact": contact}, geometric)

    def prepare_velocity(self):
        p, b = self.candidate.particles, self.candidate.rigid
        _reconstruct(p, self.predicted.particles, self.initial.particles, self.wp, float(self.h))
        _reconstruct(b, self.predicted.rigid, self.initial.rigid, self.wb, float(self.h))
        self.geometry = particle_sphere_geometry(p.position, b.position, self.config.spheres)
        self.penetration = float(np.max(np.maximum(-self.geometry.gap, 0), initial=0))
        self.impulses = np.zeros_like(self.contact_lambda)
        self.band = np.float64(self.settings.tolerance) * self.settings.reference_length

    def velocity_sweep(self):
        self.velocity_transfers.append(
            project_particle_sphere_velocities(
                self.candidate.particles.velocity,
                self.candidate.rigid.velocity,
                self.wp,
                self.wb,
                self.geometry,
                self.impulses,
                contact_tolerance=self.band,
            )
        )

    def velocity_metrics(self):
        residual = particle_sphere_velocity_residual(
            self.candidate.particles.velocity,
            self.candidate.rigid.velocity,
            self.wp,
            self.wb,
            self.geometry,
            self.impulses,
            contact_tolerance=self.band,
            reference_speed=self.settings.reference_speed,
        )
        return StageMetrics({"velocity": residual}, penetration=self.penetration)

    def result(self):
        return SceneState({"particles": self.candidate.particles, "rigid": self.candidate.rigid})

    def diagnostics(self):
        return {
            "position_impulses": _impulse_ledger(
                self.position_transfers, float(self.h), len(self.wp), len(self.wb)
            ),
            "velocity_impulses": _impulse_ledger(
                self.velocity_transfers, 1.0, len(self.wp), len(self.wb)
            ),
        }

    def close(self):
        pass


class CoupledRuntimeAdapter:
    """Convert only data; all time, iterations, validation and commit live in runtime."""

    def __init__(self, program, settings, context):
        self.program = program
        self.runtime = SceneStepper(program.scene, program, settings, context)
        self.last_scene_result = None

    @staticmethod
    def _state(value):
        return CoupledState(value.domains["particles"], value.domains["rigid"])

    def step(self, state, inputs, dt):
        self.last_scene_result = None
        if not isinstance(state, CoupledState) or not isinstance(inputs, CoupledInputs):
            raise TypeError("expected CoupledState and CoupledInputs")
        result = self.runtime.step(
            SceneState({"particles": state.particles, "rigid": state.rigid}),
            SceneInputs({"particles": inputs.particles, "rigid": inputs.rigid}),
            dt,
        )
        self.last_scene_result = result
        reports = []
        for r in result.report.substeps:
            position = r.position or StageMetrics({"missing": 0})
            velocity = r.velocity or StageMetrics({"missing": 0})
            reports.append(
                CoupledSubstepReport(
                    r.accepted,
                    r.position_iterations,
                    r.velocity_iterations,
                    position.residuals.get("distance", float("inf")),
                    position.residuals.get("contact", float("inf")),
                    velocity.residuals.get("velocity", float("inf")),
                    position.geometric_error,
                    velocity.penetration,
                    r.diagnostics.get("position_impulses"),
                    r.diagnostics.get("velocity_impulses"),
                    r.reason,
                )
            )
        report = CoupledStepReport(
            result.report.accepted,
            result.report.completed_substeps,
            tuple(reports),
            result.report.reason,
            result.report.backend,
        )
        return CoupledStepResult(
            self._state(result.state), report, tuple(self._state(s) for s in result.substep_states)
        )

    def close(self):
        self.runtime.close()

    def __enter__(self):
        self.runtime.__enter__()
        return self

    def __exit__(self, *_):
        self.close()


def make_coupled_runtime(
    particle_model,
    rigid_model,
    constraints,
    spheres,
    *,
    mode="numpy",
    substeps=4,
    iterations=200,
    tolerance=None,
    reference_length=1.0,
    reference_speed=1.0,
    record_substeps=False,
    cache_root=None,
):
    """Build a phased runtime with explicit backend selection and no silent fallback."""
    numpy_mode = mode in ("numpy", "numpy_colored")
    if tolerance is None:
        tolerance = 1e-7 if numpy_mode else 1e-6
    settings = StepSettings(
        substeps,
        iterations,
        iterations,
        tolerance,
        reference_length,
        reference_speed,
        record_substeps,
    )
    if numpy_mode:
        program = NumpyCoupledProgram(
            particle_model, rigid_model, constraints, spheres, colored=mode == "numpy_colored"
        )
    else:
        from ..backends.tyx_coupled_program import TyxCoupledProgram

        program = TyxCoupledProgram(
            particle_model, rigid_model, constraints, spheres, mode=mode, cache_root=cache_root
        )
    try:
        context = BackendContext(program.capabilities.backend, particle_model.inverse_mass.dtype)
        return CoupledRuntimeAdapter(program, settings, context)
    except BaseException:
        program.close()
        raise
