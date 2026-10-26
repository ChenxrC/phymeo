"""FP32 colored XPBD using real tyx kernels and independent host residual checks."""

from contextlib import ExitStack
from dataclasses import replace

import numpy as np

from ..constraints.coloring import color_edges
from ..core.numerics import normalized_error
from ..core.result import StepReport, StepResult
from ..core.state import ParticleState
from ..core.validation import validate_step
from ..solvers.xpbd import XPBDSolver
from .tyx import TyxSemiImplicitSolver


class TyxXPBDSolver(TyxSemiImplicitSolver):
    """Own a tyx runtime with colored distance and frictionless y-plane projection.

    Each color executes in parallel without shared position writers. Multipliers
    are FP32 and reset per step. Host residual checks synchronize each iteration;
    this initial adapter does not promise a speedup or persistent cross-step buffers.
    """

    def __init__(
        self,
        constraints=None,
        *,
        mode="interpret",
        iterations=20,
        tolerance=1e-5,
        plane=None,
        reference_length=1.0,
        cache_root=None,
        cuda_graph=False,
    ):
        if not isinstance(cuda_graph, bool):
            raise ValueError("cuda_graph must be bool")
        if cuda_graph and mode != "cuda":
            raise ValueError("cuda_graph requires mode='cuda'")
        self._cuda_graph = cuda_graph
        self._config = XPBDSolver(constraints, iterations, tolerance, plane, reference_length)
        if constraints is not None and constraints.rest_length.dtype != np.float32:
            raise ValueError("the tyx adapter supports FP32 constraints only")
        indices = np.empty((0, 2), np.int64) if constraints is None else constraints.indices
        if len(indices) > np.iinfo(np.int32).max or np.any(indices > np.iinfo(np.int32).max):
            raise ValueError("constraints exceed the tyx i32 kernel interface")
        self._colors = color_edges(indices)
        super().__init__(mode, cache_root=cache_root)
        from ._tyx_kernels import (
            project_distance_color,
            project_y_plane,
            reconstruct_particle_velocity,
        )

        self._project = project_distance_color
        self._plane = project_y_plane
        self._velocity = reconstruct_particle_velocity

    @property
    def color_count(self):
        return len(self._colors.offsets) - 1

    @property
    def capabilities(self):
        return replace(
            self._config.capabilities,
            backend=f"tyx:{self.mode}",
            dtypes=frozenset(["float32"]),
        )

    def problem_signature(self):
        return self._config.problem_signature()

    def step(self, model, state_in, inputs, dt) -> StepResult:
        if self._runtime.closed:
            raise RuntimeError("tyx solver is closed")
        candidate, force, h = validate_step(model, state_in, inputs, dt)
        if candidate.position.dtype != np.float32:
            raise ValueError("the tyx adapter supports FP32 state only")
        count = model.particle_count
        if count > np.iinfo(np.int32).max:
            raise ValueError("particle count exceeds the tyx i32 kernel interface")
        config, constraints = self._config, self._config.constraints
        if constraints is not None and np.any(constraints.indices >= count):
            raise ValueError("constraint index exceeds the model particle count")
        if count == 0:
            return StepResult(candidate, StepReport(accepted=True, backend=f"tyx:{self.mode}"))
        edge_count = len(self._colors.order)
        alpha64 = np.zeros(edge_count, np.float64)
        height = np.float32(0)
        with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
            if edge_count:
                alpha64[:] = constraints.compliance.astype(np.float64) / float(h) / float(h)
            alpha = alpha64.astype(np.float32)
            if not np.isfinite(alpha).all() or np.any((alpha == 0) & (alpha64 > 0)):
                return self._failure(state_in, 0, "scaled compliance is outside FP32 range")
            if config.plane is not None:
                height = np.float32(config.plane.height)
                if float(height) < config.plane.height:
                    height = np.nextafter(height, np.float32(np.inf))
                if not np.isfinite(height):
                    return self._failure(state_in, 0, "plane height is outside FP32 range")
            with self._tx.use_runtime(self._runtime), ExitStack() as resources:
                return self._execute(model, candidate, force, h, alpha, alpha64, height, resources)

    def _execute(self, model, state_in, force, h, alpha, alpha64, height, resources):
        config, constraints = self._config, self._config.constraints
        count, edge_count = model.particle_count, len(alpha)

        def buffer(value):
            if self.mode != "cuda":
                return np.array(value, order="C", copy=True)
            device = self._tx.cuda_array(value)
            resources.callback(device.free)
            return device

        def read(value):
            return value.to_numpy() if self.mode == "cuda" else value

        original = buffer(state_in.position)
        previous_velocity = buffer(state_in.velocity)
        external = buffer(force)
        weights = buffer(model.inverse_mass)
        gravity = buffer(model.gravity)
        position = buffer(np.empty_like(state_in.position))
        velocity = buffer(np.empty_like(state_in.velocity))
        self._kernel(
            count,
            h,
            original,
            previous_velocity,
            external,
            weights,
            gravity,
            position,
            velocity,
            mode=self.mode,
        )
        self._runtime.sync()
        # Check prediction before a plane projection could hide a negative infinity.
        if not (np.isfinite(read(position)).all() and np.isfinite(read(velocity)).all()):
            return self._failure(state_in, 0, "non-finite prediction")
        multipliers = np.zeros(0, np.float32)
        if edge_count:
            order = buffer(self._colors.order.astype(np.int32))
            edges = buffer(constraints.indices.astype(np.int32))
            rest = buffer(constraints.rest_length)
            scaled = buffer(alpha)
            multipliers = buffer(np.zeros(edge_count, np.float32))
            failed = buffer(np.zeros(edge_count, np.int32))

        def sweep(stream=None):
            if edge_count:
                for begin, end in zip(self._colors.offsets[:-1], self._colors.offsets[1:]):
                    self._project(
                        int(end - begin),
                        int(begin),
                        order,
                        edges,
                        rest,
                        scaled,
                        weights,
                        position,
                        multipliers,
                        failed,
                        mode=self.mode,
                        stream=stream,
                    )
            if config.plane is not None:
                self._plane(count, height, weights, position, mode=self.mode, stream=stream)

        graph = None
        if self._cuda_graph and (edge_count or config.plane is not None):
            # Registered after every array: synchronize and release graph references
            # before buffers, including capture/replay/readback exception paths.
            graph_resources = resources.enter_context(ExitStack())
            stream = self._tx.cuda_stream()
            graph_resources.callback(stream.destroy)
            graph = self._tx.Graph(stream=stream, mode="cuda", optimize=False)
            graph_resources.callback(graph.destroy)
            graph_resources.callback(stream.synchronize)
            with graph:
                sweep(stream)
            if graph.backend != "cuda":
                raise RuntimeError("explicit CUDA graph capture did not produce a CUDA graph")

        for iteration in range(1, config.iterations + 1):
            if graph is None:
                sweep()
            else:
                graph.replay()
            self._runtime.sync()
            host_position = read(position)
            host_lambda = read(multipliers) if edge_count else multipliers
            if (
                (edge_count and np.any(read(failed)))
                or not np.isfinite(host_position).all()
                or not np.isfinite(host_lambda).all()
            ):
                return self._failure(state_in, iteration, "non-finite distance projection")
            residual, geometric = self._residual(host_position, alpha64, host_lambda)
            if not (np.isfinite(residual) and np.isfinite(geometric)):
                return self._failure(state_in, iteration, "non-finite XPBD residual")
            if residual <= config.tolerance:
                break
        self._velocity(
            count,
            h,
            int(config.plane is not None),
            height,
            weights,
            original,
            position,
            velocity,
            mode=self.mode,
        )
        self._runtime.sync()
        host_velocity = read(velocity)
        if not np.isfinite(host_velocity).all():
            return self._failure(state_in, iteration, "non-finite reconstructed velocity")
        converged = residual <= config.tolerance
        return StepResult(
            ParticleState(host_position, host_velocity),
            StepReport(
                accepted=converged,
                iterations=iteration,
                max_residual=residual,
                max_constraint_error=geometric,
                reason=None if converged else "iteration budget exhausted",
                backend=f"tyx:{self.mode}",
            ),
        )

    def _residual(self, position, alpha, multipliers):
        # Original physical parameters and FP64 geometry provide an independent
        # acceptance check; do not merely echo the rounded device residual.
        geometry = position.astype(np.float64)
        constraints, config = self._config.constraints, self._config
        residual, geometric = 0.0, 0.0
        if constraints is not None:
            delta = geometry[constraints.indices[:, 0]] - geometry[constraints.indices[:, 1]]
            rest = constraints.rest_length.astype(np.float64)
            c = np.hypot.reduce(delta, axis=1) - rest
            scale = np.maximum(rest, config.reference_length)
            residual = float(
                np.max(normalized_error(np.abs(c + alpha * multipliers), scale), initial=0.0)
            )
            geometric = float(np.max(np.abs(c), initial=0.0))
        if config.plane is not None:
            penetration = float(np.max(config.plane.height - geometry[:, 1], initial=0.0))
            residual = max(residual, float(normalized_error(penetration, config.reference_length)))
            geometric = max(geometric, penetration)
        return residual, geometric

    def _failure(self, state, iteration, reason):
        return StepResult(
            ParticleState.copy(state),
            StepReport(
                accepted=False,
                iterations=iteration,
                max_residual=float("inf"),
                max_constraint_error=float("inf"),
                reason=reason,
                backend=f"tyx:{self.mode}",
            ),
        )
