"""A real mixed free-dynamics program consuming the generic phased runtime."""

from ..core.capabilities import SolverFailure
from ..core.scene import SceneState, StageMetrics
from ..runtime.program import ProgramCapabilities
from .rigid_semi_implicit import RigidSemiImplicitSolver
from .semi_implicit import SemiImplicitSolver


class FreeSceneProgram:
    capabilities = ProgramCapabilities(
        "numpy",
        {"float32", "float64"},
        {"particles", "rigid_bodies"},
        {"free_dynamics", "rotation"},
    )
    history_fields = ()

    def problem_signature(self):
        return {"semantics": "free-scene-force-torque-gravity-v1"}

    def create_workspace(self, scene, state, inputs, h, settings, context):
        return _FreeWorkspace(scene, state, inputs, h)

    def close(self):
        pass


class _FreeWorkspace:
    def __init__(self, scene, state, inputs, h):
        self.scene, self.state, self.inputs, self.h = scene, state, inputs, h
        self.solvers = {
            "particles": SemiImplicitSolver(),
            "rigid_bodies": RigidSemiImplicitSolver(),
        }

    def predict(self):
        states = {}
        for domain in self.scene.domains:
            result = self.solvers[domain.family].step(
                domain.model,
                self.state.domains[domain.name],
                self.inputs.domains[domain.name],
                self.h,
            )
            if not result.report.accepted:
                raise SolverFailure(result.report.reason or "free prediction failed")
            states[domain.name] = result.state
        self.state = SceneState(states)

    def position_sweep(self):
        pass

    def position_metrics(self):
        return StageMetrics()

    def prepare_velocity(self):
        pass

    def velocity_sweep(self):
        pass

    def velocity_metrics(self):
        return StageMetrics()

    def result(self):
        return self.state

    def diagnostics(self):
        return {}

    def close(self):
        pass
