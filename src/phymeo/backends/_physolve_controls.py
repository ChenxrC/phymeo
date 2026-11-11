"""Persistent scalar drive commands; native ctrl is their derived execution input."""

import numpy as np
import tyx as tx

from ._physolve_reset import _restore_float


@tx.kernel(gradient="unsupported")
def _compose(
    position: tx.Array[tx.f32],
    velocity: tx.Array[tx.f32],
    coefficients: tx.Array[tx.f32],
    actuators: tx.Array[tx.i32],
    ctrl: tx.Array[tx.f32],
    native_width: tx.i32,
):
    for i in tx.range(position.shape[0]):
        joint = i % actuators.shape[0]
        world = i // actuators.shape[0]
        ctrl[world * native_width + actuators[joint]] = (
            coefficients[2 * joint] * position[i] + coefficients[2 * joint + 1] * velocity[i]
        )


class PreparedControls:
    """Two command banks and reset baseline, owned only by the physical runtime."""

    def __init__(self, rows, banks, model, nworld, device):
        self.names = tuple(row[0] for row in rows)
        ids = [row[1] for row in rows]
        if len(set(self.names)) != len(rows) or len(set(ids)) != len(ids):
            raise ValueError("scalar control bindings must be unique")
        for name, actuator, kp, kd, q, qd in rows:
            if (
                not isinstance(name, str)
                or type(actuator) is not int
                or not 0 <= actuator < model.nu
                or kp < 0
                or kd < 0
                or not np.isfinite(np.asarray([kp, kd, q, qd], dtype=np.float32)).all()
            ):
                raise ValueError("invalid scalar control binding")
            if not (
                model.actuator_gainprm[actuator, 0] == 1
                and np.array_equal(model.actuator_biasprm[actuator, :3], [0, -kp, -kd])
            ):
                raise ValueError("scalar control binding differs from compiled actuator")
        self.width = len(rows)
        self.coefficients = tx.Array(
            np.asarray([[r[2], r[3]] for r in rows], np.float32).reshape(-1), device=device
        )
        self.actuators = tx.Array(np.asarray(ids, np.int32), device=device)
        self.banks = []
        for _ in banks:
            self.banks.append(
                {
                    name: tx.Array(
                        np.broadcast_to(
                            np.asarray([row[column] for row in rows], np.float32),
                            (nworld, len(rows)),
                        ).copy(),
                        device=device,
                    )
                    for name, column in (("target_position", 4), ("target_velocity", 5))
                }
            )
        self.initial = {name: tx.clone(array) for name, array in self.banks[0].items()}
        self.flat = [{name: array.flatten() for name, array in bank.items()} for bank in self.banks]
        self.initial_flat = {name: array.flatten() for name, array in self.initial.items()}
        self.native = tuple(bank.ctrl.view(tx.f32).flatten() for bank in banks)
        self.native_width = model.nu

    def prepare(self, stream):
        # Native initial ctrl was compiled in double; warming must not alter it.
        scratch = tx.empty_like(self.native[0])
        _compose(
            self.flat[0]["target_position"],
            self.flat[0]["target_velocity"],
            self.coefficients,
            self.actuators,
            scratch,
            self.native_width,
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
            self.native[index],
            self.native_width,
            mode="cuda",
            stream=stream,
        )

    def reset(self, index, mask, stream):
        for name, array in self.flat[index].items():
            _restore_float(
                mask, self.initial_flat[name], array, self.width, mode="cuda", stream=stream
            )
