"""Actual FP32 displacement execution; FP64 world decoding is an explicit codec."""

import numpy as np

from ..constraints.material_representation import (
    NativeRepresentationOperation,
    RepresentationBoundary,
    RepresentedMaterialBatch,
    RepresentedMaterialPath,
    RepresentedMaterialSweep,
    decode,
    encode_values,
    exact,
    serialize_evidence,
)
from ..core.capabilities import SolverFailure
from ..core.scene import BackendContext, fingerprint
from ..core.state import ParticleState
from ..runtime.registry import StateRegistry
from ..solvers.composite_cloth import CompositeClothWorkspace
from ..validation.material_representation import RepresentedMaterialPathLedger
from .tyx_bending import TyxBendingMath
from .tyx_composite_cloth import TyxCompositeClothWorkspace, TyxCompositeParticipantWorkspace
from .tyx_math import as_index


class RepresentedMath(TyxBendingMath):
    def __init__(self, mode, *, cache_root=None):
        super().__init__(mode, cache_root=cache_root)
        self.terminal_operations = []
        self.material_operations = []
        self.failed_operations = []

    def _operation(self, name, inputs, kernel, *arguments, outputs):
        before = self.stats()
        try:
            values = self.call(kernel, *arguments, outputs=outputs)
        except BaseException:
            after = self.stats()
            self.failed_operations.append(
                NativeRepresentationOperation(
                    "failed/" + name,
                    tuple(inputs),
                    (),
                    after["launches"] - before["launches"],
                    after["completed_launches"] - before["completed_launches"],
                )
            )
            raise
        after = self.stats()
        return values, NativeRepresentationOperation(
            name,
            tuple(inputs),
            tuple(values),
            after["launches"] - before["launches"],
            after["completed_launches"] - before["completed_launches"],
        )

    def _material_call(self, name, kernel, *arguments, outputs):
        before = self.stats()
        inputs = tuple(np.array(v, copy=True) for v in arguments if isinstance(v, np.ndarray))
        try:
            result = self.call(kernel, *arguments, outputs=outputs)
        except BaseException:
            after = self.stats()
            self.failed_operations.append(
                NativeRepresentationOperation(
                    "failed/" + name,
                    inputs,
                    (),
                    after["launches"] - before["launches"],
                    after["completed_launches"] - before["completed_launches"],
                )
            )
            raise
        after = self.stats()
        operation = NativeRepresentationOperation(
            name,
            inputs,
            tuple(result),
            after["launches"] - before["launches"],
            after["completed_launches"] - before["completed_launches"],
        )
        self.material_operations.append(operation)
        if np.any(result[3]) or np.any(result[4]):
            self.failed_operations.append(operation)
        return result

    def predict_displacement(self, velocity, force, weights, gravity, h):
        from ._tyx_represented_material_kernels import predict_represented

        inputs = (velocity, force, weights, gravity, np.asarray(h, np.float32))
        zero = np.zeros_like(velocity)
        if not len(velocity):
            return (
                zero,
                zero.copy(),
                NativeRepresentationOperation("predict", inputs, (zero, zero.copy()), 0, 0),
            )
        values, operation = self._operation(
            "predict",
            inputs,
            predict_represented,
            len(velocity),
            float(h),
            velocity,
            force,
            weights,
            gravity,
            zero,
            zero.copy(),
            outputs=(6, 7),
        )
        return values[0], values[1], operation

    def alpha_scale(self, name, value, inverse_h):
        before = self.stats()
        zero = np.zeros_like(value)
        result = super().axpy(zero, value, float(inverse_h))
        after = self.stats()
        return result, NativeRepresentationOperation(
            name,
            (zero, value, np.asarray(inverse_h, np.float32)),
            (result,),
            after["launches"] - before["launches"],
            after["completed_launches"] - before["completed_launches"],
        )

    def reconstruct_displacement(self, predicted_d, predicted_v, displacement, h):
        from ._tyx_represented_material_kernels import reconstruct_represented

        inputs = (predicted_d, predicted_v, displacement, np.asarray(h, np.float32))
        if not len(displacement):
            output = np.zeros_like(displacement)
            return output, NativeRepresentationOperation("reconstruct", inputs, (output,), 0, 0)
        values, operation = self._operation(
            "reconstruct",
            inputs,
            reconstruct_represented,
            len(displacement),
            float(h),
            predicted_d,
            predicted_v,
            displacement,
            np.zeros_like(displacement),
            outputs=(5,),
        )
        return values[0], operation

    def project_bending(
        self, displacement, weights, constraints, alpha, multipliers, reference_vectors
    ):
        from ._tyx_represented_material_kernels import project_bending_represented

        count = len(constraints.indices)
        return self._material_call(
            "bending",
            project_bending_represented,
            count,
            as_index(constraints.indices, "hinges"),
            constraints.rest_angle,
            alpha,
            weights,
            displacement,
            multipliers,
            np.zeros((count, 45), np.float32),
            np.zeros(count, np.int32),
            np.zeros(1, np.int32),
            np.zeros((6, 3), np.float32),
            reference_vectors,
            np.zeros((count, 4, 3), np.float32),
            np.zeros((count, 3, 3), np.float32),
            np.zeros((count, 3, 3), np.float32),
            outputs=(5, 6, 7, 8, 9, 12, 13, 14),
        )

    def project_distance(
        self, displacement, weights, constraints, alpha, multipliers, reference_vectors
    ):
        from ._tyx_represented_material_kernels import project_distance_represented

        count = len(constraints.indices)
        return self._material_call(
            "distance",
            project_distance_represented,
            count,
            as_index(constraints.indices, "edges"),
            constraints.rest_length,
            alpha,
            weights,
            displacement,
            multipliers,
            np.zeros((count, 18), np.float32),
            np.zeros(count, np.int32),
            np.zeros(1, np.int32),
            reference_vectors,
            np.zeros((count, 2, 3), np.float32),
            np.zeros((count, 1, 3), np.float32),
            np.zeros((count, 1, 3), np.float32),
            outputs=(5, 6, 7, 8, 9, 11, 12, 13),
        )

    def apply_load(self, state, indices, loads, weights):
        before = self.stats()
        value = super().apply_load(state, indices, loads, weights)
        after = self.stats()
        self.terminal_operations.append(
            NativeRepresentationOperation(
                "terminal_response",
                (
                    np.asarray(state, np.float32),
                    np.asarray(indices, np.int32),
                    np.asarray(loads, np.float32),
                    np.asarray(weights, np.float32),
                ),
                (np.asarray(value, np.float32),),
                after["launches"] - before["launches"],
                after["completed_launches"] - before["completed_launches"],
            )
        )
        return value

    def axpy(self, x, y, a):
        before = self.stats()
        value = super().axpy(x, y, a)
        after = self.stats()
        self.terminal_operations.append(
            NativeRepresentationOperation(
                "terminal_apply",
                (np.asarray(x, np.float32), np.asarray(y, np.float32), np.asarray(a, np.float32)),
                (value,),
                after["launches"] - before["launches"],
                after["completed_launches"] - before["completed_launches"],
            )
        )
        return value


class RepresentedCompositeClothWorkspace(TyxCompositeClothWorkspace):
    def __init__(self, program, domain, state, inputs, h, settings):
        CompositeClothWorkspace.__init__(self, program, domain, state, inputs, h, settings)
        self.family = StateRegistry().get("particles")
        self.context = BackendContext("tyx:" + program.mode)
        self.reference = self.initial.position.copy()
        self.reference.flags.writeable = False
        self.reference_id = fingerprint(
            (
                self.reference,
                self.material.identity,
                program.representation,
                domain,
                self.h,
                fingerprint(self.initial),
                fingerprint(self.inputs),
            )
        )
        self.displacement = np.zeros_like(self.reference, dtype=np.float32)
        self.predicted_displacement = None
        self.reconstructed_velocity = None
        self.prediction_operation = self.reconstruction_operation = None
        self.alpha_operations = []
        self.reference_vectors = {}
        self.native_rest = {}
        self.native_weights = None
        self.ledger = RepresentedMaterialPathLedger(
            self.model,
            self.material,
            self.initial,
            self.inputs,
            h,
            settings,
            audit_settings=program.audit_settings,
            backend=program.capabilities.backend,
            representation=program.representation,
            domain=domain,
        )
        self.engine = RepresentedMath(program.mode, cache_root=program.cache_root)

    def _convert(self, name, value):
        original = np.asarray(value)
        computed, witnesses = encode_values((exact(v) for v in original.flat), original.shape)
        self.compute_boundaries.append(RepresentationBoundary(name, computed, witnesses))
        return computed

    def _reference_table(self, kind, law):
        count = self.material.count(kind)
        values = []
        for ids in () if law is None else law.indices:
            pairs = (
                ((ids[0], ids[1]),)
                if kind == "distance"
                else ((ids[1], ids[0]), (ids[2], ids[0]), (ids[3], ids[1]))
            )
            for a, b in pairs:
                for c in range(3):
                    values.append(exact(self.reference[a, c]) - exact(self.reference[b, c]))
        shape = (count, 1 if kind == "distance" else 3, 3)
        computed, witnesses = encode_values(values, shape)
        self.compute_boundaries.append(
            RepresentationBoundary("reference/" + kind, computed, witnesses)
        )
        return computed

    def predict(self):
        if self.closed or self.candidate is not None:
            raise RuntimeError("composite prediction must occur exactly once")
        boundaries = list(self.compute_boundaries)
        operations = list(self.alpha_operations)
        try:
            velocity = self._convert("predict/velocity", self.initial.velocity)
            force = self._convert("predict/force", self.inputs.force)
            weights = self._convert("predict/weights", self.model.inverse_mass)
            gravity = self._convert("predict/gravity", self.model.gravity)
            h = self._convert("predict/h", np.asarray(self.h))
            d, v, prediction = self.engine.predict_displacement(
                velocity, force, weights, gravity, h
            )
            x, _ = decode(self.reference, d)
            alphas, rests, tables = {}, {}, {}
            inverse, inverse_witnesses = encode_values((1 / exact(self.h),), ())
            self.compute_boundaries.append(
                RepresentationBoundary("alpha/inverse_h", inverse, inverse_witnesses)
            )
            for kind in self.material.order:
                law = getattr(self.material, kind)
                rests[kind] = self._convert(
                    kind + "/rest",
                    np.zeros(0)
                    if law is None
                    else (law.rest_length if kind == "distance" else law.rest_angle),
                )
                compliance = self._convert(
                    kind + "/compliance", np.zeros(0) if law is None else law.compliance
                )
                tables[kind] = self._reference_table(kind, law)
                if self.material.count(kind):
                    first, op1 = self.engine.alpha_scale(kind + "/alpha/0", compliance, inverse)
                    last, op2 = self.engine.alpha_scale(kind + "/alpha/1", first, inverse)
                    self.alpha_operations.extend((op1, op2))
                    alphas[kind] = last.astype(np.float64)
                else:
                    alphas[kind] = np.zeros(0)
            self.ledger.domain(self.initial.position, x)
        except BaseException:
            self.compute_boundaries[:] = boundaries
            self.alpha_operations[:] = operations
            raise
        self.native_weights, self.native_h = weights, h
        self.native_rest, self.reference_vectors, self.alphas = rests, tables, alphas
        self.displacement, self.predicted_displacement = d.copy(), d.copy()
        self.prediction_operation = prediction
        self.candidate = ParticleState(x, v.astype(np.float64))
        self.predicted = self.candidate.copy()
        self.multipliers = {
            kind: np.zeros(self.material.count(kind)) for kind in self.material.order
        }

    def _represented_block(self, kind, displacement, multipliers):
        from types import SimpleNamespace

        current_reference_id = fingerprint(
            (
                self.reference,
                self.material.identity,
                self.program.representation,
                self.domain,
                self.h,
                fingerprint(self.initial),
                fingerprint(self.inputs),
            )
        )
        if current_reference_id != self.reference_id:
            raise SolverFailure("immutable represented reference binding changed")
        count = self.material.count(kind)
        law = getattr(self.material, kind)
        arity, wire, vectors = (2, 18, 1) if kind == "distance" else (4, 45, 3)
        ids = np.empty((0, arity), np.int64) if law is None else law.indices
        start, old_lambda = displacement.copy(), np.asarray(multipliers, np.float32).copy()
        alpha = np.asarray(self.alphas[kind], np.float32)
        before = self.engine.stats()
        if count:
            native_law = SimpleNamespace(
                indices=ids, rest_length=self.native_rest[kind], rest_angle=self.native_rest[kind]
            )
            method = (
                self.engine.project_distance if kind == "distance" else self.engine.project_bending
            )
            result = method(
                start,
                self.native_weights,
                native_law,
                alpha,
                old_lambda,
                self.reference_vectors[kind],
            )
            end, new_lambda, rows, status, failed, products, geometry_before, geometry_after = (
                result
            )
            if np.any(status) or failed[0]:
                raise SolverFailure("represented native row rejected: " + str(int(failed[0])))
        else:
            end, new_lambda = start.copy(), old_lambda.copy()
            rows = np.empty((0, wire), np.float32)
            status = np.zeros(0, np.int32)
            products = np.empty((0, arity, 3), np.float32)
            geometry_before = np.empty((0, vectors, 3), np.float32)
            geometry_after = geometry_before.copy()
        after = self.engine.stats()
        x_in, _ = decode(self.reference, displacement)
        x_out, _ = decode(self.reference, end)
        world_before, world_after, witnesses_before, witnesses_after = [], [], [], []
        for k, nodes in enumerate(ids):
            size = arity * 3
            x0, first = decode(self.reference[nodes], rows[k, :size].reshape((arity, 3)))
            x1, last = decode(self.reference[nodes], rows[k, size : 2 * size].reshape((arity, 3)))
            world_before.append(x0)
            world_after.append(x1)
            witnesses_before.extend(first)
            witnesses_after.extend(last)
        batch = RepresentedMaterialBatch(
            kind,
            ids,
            self.reference_id,
            displacement,
            end,
            x_in,
            x_out,
            old_lambda,
            new_lambda,
            self.native_weights,
            self.native_rest[kind],
            alpha,
            self.reference_vectors[kind],
            rows,
            products,
            geometry_before,
            geometry_after,
            np.asarray(world_before, np.float64).reshape((count, arity, 3)),
            np.asarray(world_after, np.float64).reshape((count, arity, 3)),
            tuple(witnesses_before),
            tuple(witnesses_after),
            status,
            after["launches"] - before["launches"],
            after["completed_launches"] - before["completed_launches"],
        )
        return end, new_lambda.astype(np.float64), batch

    def position_sweep(self):
        self._ready()
        if self.velocity_ready:
            raise RuntimeError("material projection after reconstruction")
        displacement = self.displacement.copy()
        multipliers = {kind: values.copy() for kind, values in self.multipliers.items()}
        blocks = []
        for kind in self.material.order:
            displacement, multipliers[kind], batch = self._represented_block(
                kind, displacement, multipliers[kind]
            )
            blocks.append(batch)
        sweep = RepresentedMaterialSweep(len(self.frames), tuple(blocks))
        domains, bounds = self.ledger._domains.copy(), self.ledger._bounds.copy()
        try:
            self.ledger.audit_sweep(
                sweep,
                self.candidate.position,
                self.multipliers,
                self.alphas,
                displacement=self.displacement,
            )
            world, _ = decode(self.reference, displacement)
        except BaseException:
            self.ledger._domains, self.ledger._bounds = domains, bounds
            raise
        self.displacement = displacement.copy()
        self.candidate.position = world
        self.multipliers = multipliers
        self.frames.append(sweep)

    def _reconstruct(self):
        world, _ = decode(self.reference, self.displacement)
        if fingerprint(world) != fingerprint(self.candidate.position):
            raise SolverFailure("represented hidden displacement/world mismatch")
        value, operation = self.engine.reconstruct_displacement(
            self.predicted_displacement,
            self.predicted.velocity.astype(np.float32),
            self.displacement,
            self.native_h,
        )
        return value, operation

    def prepare_velocity(self):
        self._ready()
        if self.velocity_ready:
            raise RuntimeError("composite velocity already reconstructed")
        value, operation = self._reconstruct()
        if (
            value.dtype != np.float32
            or value.shape != self.candidate.velocity.shape
            or not np.isfinite(value).all()
        ):
            raise SolverFailure("represented velocity reconstruction failed")
        published = value.astype(np.float64)
        self.reconstructed_velocity, self.reconstruction_operation = value.copy(), operation
        self.candidate.velocity, self.velocity_ready = published, True

    def _path(self):
        return RepresentedMaterialPath(
            self.material.identity,
            fingerprint(self.program.audit_settings),
            fingerprint(self.inputs),
            fingerprint(self.initial),
            self.program.capabilities.backend,
            self.h,
            self.program.representation,
            self.domain,
            self.reference,
            self.reference_id,
            self.predicted_displacement,
            self.displacement,
            self.predicted.position,
            self.predicted.velocity,
            self.reconstructed_velocity,
            np.asarray(self.alphas["distance"], np.float32),
            np.asarray(self.alphas["bending"], np.float32),
            tuple(self.frames),
            tuple(self.compute_boundaries),
            self.prediction_operation,
            self.reconstruction_operation,
            tuple(self.alpha_operations),
        )

    def _audit(self):
        self._ready()
        if not self.velocity_ready:
            raise RuntimeError("cannot publish before velocity reconstruction")
        base = self.candidate if self.terminal_pre is None else self.terminal_pre
        data = dict(
            material_path=self._path(),
            distance_multipliers=self.multipliers["distance"],
            bending_multipliers=self.multipliers["bending"],
            angular_error_rad=getattr(self, "angular_error_rad", 0.0),
            represented_displacement=self.displacement.copy(),
            represented_reference=self.reference.copy(),
            terminal_operations=tuple(self.engine.terminal_operations),
        )
        data["material_audit"] = self.ledger.require(
            base, self.inputs, np.zeros_like(base.position), data, sweeps=len(self.frames)
        )
        data["physical_audit"] = self.energy_ledger.require(base)
        if self.terminal_pre is not None:
            if not np.array_equal(self.candidate.position, base.position):
                raise SolverFailure("terminal configuration mutation")
            error = (
                self.candidate.velocity
                - base.velocity
                - self.model.inverse_mass[:, None] * self.terminal_requests
            )
            if (
                np.max(np.abs(error), initial=0) / self.settings.reference_speed
                > self.settings.tolerance
            ):
                raise SolverFailure("terminal velocity mutation or unrecorded impulse")
            self.ledger.audit_terminal(
                self.engine.terminal_operations, base.velocity, self.candidate.velocity
            )
            data["terminal_pre"], data["terminal_requests"] = (
                base.copy(),
                self.terminal_requests.copy(),
            )
        data["represented_wire"] = serialize_evidence(data["material_path"]).decode("utf-8")
        return data


class RepresentedCompositeParticipantWorkspace(TyxCompositeParticipantWorkspace):
    def apply_velocity_impulses(self, indices, local_points, impulses):
        native = self.native
        state = native.candidate.copy() if native.candidate is not None else None
        pre = None if native.terminal_pre is None else native.terminal_pre.copy()
        requests = native.terminal_requests.copy()
        checkpoint = len(native.engine.terminal_operations)
        try:
            super().apply_velocity_impulses(indices, local_points, impulses)
        except BaseException:
            native.candidate, native.terminal_pre, native.terminal_requests = state, pre, requests
            native.engine.failed_operations.extend(native.engine.terminal_operations[checkpoint:])
            del native.engine.terminal_operations[checkpoint:]
            raise
