"""Persistent scalar drive commands; native ctrl is their derived execution input."""

import numpy as np
import tyx as tx

from ._physolve_reset import _restore_float, _preserve_float


@tx.kernel(gradient="unsupported")
def _compose(
    position: tx.Array[tx.f32],
    velocity: tx.Array[tx.f32],
    coefficients: tx.Array[tx.f32],
    actuators: tx.Array[tx.i32],
    sources: tx.Array[tx.i32],
    affine: tx.Array[tx.f32],
    ctrl: tx.Array[tx.f32],
    command_width: tx.i32,
    native_width: tx.i32,
    error: tx.Array[tx.i32],
    failure_bit: tx.i32,
):
    for i in tx.range(position.shape[0] // command_width * actuators.shape[0]):
        joint = i % actuators.shape[0]
        world = i // actuators.shape[0]
        source = world * command_width + sources[joint]
        q = affine[2 * joint] * position[source] + affine[2 * joint + 1]
        v = affine[2 * joint] * velocity[source]
        value = coefficients[2 * joint] * q + coefficients[2 * joint + 1] * v
        if tx.isfinite(q) and tx.isfinite(v) and tx.isfinite(value):
            ctrl[world * native_width + actuators[joint]] = value
        else:
            # Reject the input; retain finite candidate ctrl for native work.
            tx.atomic_or(error[0], failure_bit)


class PreparedControls:
    """Two command banks and reset baseline, owned only by the physical runtime."""

    def initialize(self, rows, banks, model, nworld, device, error, followers=()):
        self.error = error
        self.names = tuple(row[0] for row in rows)
        drives = [(r[0], r[1], i, 1.0, 0.0, r[2], r[3]) for i, r in enumerate(rows)]
        drives.extend(followers)
        self.drive_names = tuple(row[0] for row in drives)
        ids = [row[1] for row in drives]
        self.native_actuators = tuple(ids)
        self.recompose_reset = False
        if len(set(self.names)) != len(rows) or len(set(self.drive_names)) != len(drives) or len(set(ids)) != len(ids):
            raise ValueError("scalar control bindings must be unique")
        if not np.isfinite(np.asarray([r[4:6] for r in rows], dtype=np.float32)).all():
            raise ValueError("invalid scalar control targets")
        for name, actuator, source, ratio, offset, kp, kd in drives:
            if (
                not isinstance(name, str)
                or type(actuator) is not int
                or not 0 <= actuator < model.nu
                or kp < 0
                or kd < 0
                or type(source) is not int or not 0 <= source < len(rows)
                or not np.isfinite(np.asarray([kp, kd, ratio, offset], dtype=np.float32)).all()
                or np.float32(ratio) == 0
            ):
                raise ValueError("invalid scalar control binding")
            if not (
                model.actuator_gainprm[actuator, 0] == 1
                and np.array_equal(model.actuator_biasprm[actuator, :3], [0, -kp, -kd])
            ):
                raise ValueError("scalar control binding differs from compiled actuator")
        self.width = len(rows)
        self.coefficients = tx.Array(
            np.asarray([[r[5], r[6]] for r in drives], np.float32).reshape(-1), device=device
        )
        self.actuators = tx.Array(np.asarray(ids, np.int32), device=device)
        self.sources = tx.Array(np.asarray([r[2] for r in drives], np.int32), device=device)
        self.affine = tx.Array(np.asarray([[r[3], r[4]] for r in drives], np.float32).reshape(-1), device=device)
        self.banks = []
        for _ in banks:
            bank = {}
            self.banks.append(bank)
            for name, column in (("target_position", 4), ("target_velocity", 5)):
                bank[name] = tx.Array(
                    np.broadcast_to(
                        np.asarray([row[column] for row in rows], np.float32),
                        (nworld, len(rows)),
                    ).copy(), device=device,
                )
        self.initial = {}
        for name, array in self.banks[0].items():
            self.initial[name] = tx.clone(array)
        self.flat = [{name: array.flatten() for name, array in bank.items()} for bank in self.banks]
        self.initial_flat = {name: array.flatten() for name, array in self.initial.items()}
        self.native = tuple(bank.ctrl.view(tx.f32).flatten() for bank in banks)
        self.native_width = model.nu
        self.validation_scratch = tx.empty_like(self.native[0])

    def prepare(self, stream):
        # Native initial ctrl was compiled in double; warming must not alter it.
        scratch = self.validation_scratch
        _compose(
            self.flat[0]["target_position"],
            self.flat[0]["target_velocity"],
            self.coefficients,
            self.actuators,
            self.sources,
            self.affine,
            scratch,
            self.width,
            self.native_width,
            self.error,
            2097152,
            mode="cuda",
            stream=stream,
        )
        stream.synchronize()

    def copy(self, source, target, stream):
        for name in self.banks[source]:
            self.banks[target][name].assign(self.banks[source][name], stream=stream)

    def compose(self, index, stream):
        _compose(
            self.flat[index]["target_position"],
            self.flat[index]["target_velocity"],
            self.coefficients,
            self.actuators,
            self.sources,
            self.affine,
            self.native[index],
            self.width,
            self.native_width,
            self.error,
            2097152,
            mode="cuda",
            stream=stream,
        )

    def validate_commands(self, commands, stream):
        _compose(commands['target_position'], commands['target_velocity'],
                 self.coefficients, self.actuators, self.sources, self.affine,
                 self.validation_scratch, self.width, self.native_width, self.error, 32,
                 mode='cuda', stream=stream)

    def reset(self, index, mask, stream, *, compose=True):
        for name, array in self.flat[index].items():
            _restore_float(
                mask, self.initial_flat[name], array, self.width, mode="cuda", stream=stream
            )
        if self.recompose_reset and compose:
            # Model gain edits persist across reset. Initial targets are still
            # authoritative, whereas the original compiled ctrl is now stale.
            self.compose(index, stream)
            _preserve_float(mask, self.native[1-index], self.native[index],
                            self.native_width, mode="cuda", stream=stream)
