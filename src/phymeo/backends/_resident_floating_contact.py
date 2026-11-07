"""Specified-bank floating packets for the existing resident contact attempt.

These helpers only read and own observations. The caller holds the original
batch lock and device transaction, controls all candidate stages, and performs
its single publication after every receipt has been constructed successfully.
"""

from collections.abc import Mapping
from numbers import Integral
from types import MappingProxyType

import numpy as np

from ._resident_collision_plan import ResidentStaticBinding
from ._resident_contact_bridge import _CONSTRUCTION_M, ContactShapeFrame
from ._resident_floating_binding import ResidentFloatingTreeBinding
from ._resident_floating_tree import FloatingTreeDiagnostic, FloatingTreeKinematics


def _worlds(storage, worlds):
    storage._check()
    values = tuple(worlds)
    if any(isinstance(w, (bool, np.bool_)) or not isinstance(w, Integral) for w in values):
        raise ValueError("floating contact worlds must be integer identities")
    values = tuple(int(w) for w in values)
    if len(set(values)) != len(values) or any(
        w not in storage.specs or w not in storage.rows for w in values
    ):
        raise ValueError("floating contact worlds must be unique members of this storage")
    return values


def _array(value, shape, dtype, name):
    result = np.asarray(value)
    if result.shape != shape or result.dtype != np.dtype(dtype):
        raise ValueError(f"floating contact {name} must retain stored {np.dtype(dtype)} {shape}")
    return result


def _finite(*values):
    if any(not np.isfinite(value).all() for value in values):
        raise FloatingPointError("nonfinite floating contact readback")


def _material_points(position, rotation, local_points):
    # An input-material representability gate, not a contact/physics tolerance.
    # Invert the actual stored R; R.T would silently assume exact orthogonality.
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            points = position + local_points @ rotation.T
            recovered = np.linalg.solve(rotation, (points - position).T).T
            error = np.max(abs(recovered - local_points))
    except (FloatingPointError, np.linalg.LinAlgError) as error:
        raise ValueError(
            "floating contact material cannot be represented in its stored frame"
        ) from error
    if (
        not np.isfinite(points).all()
        or not np.isfinite(recovered).all()
        or not np.isfinite(error)
        or error > _CONSTRUCTION_M
    ):
        raise ValueError(
            "floating contact material lost its declared local source beyond construction consistency"
        )
    return points


def read_floating_contact_packet(plan, storage, worlds, bank):
    """Return selected worlds from one explicit root/joint/FK/twist bank.

    No use of active/snapshot/kinematics, and no reconstruction of FK or COM
    velocities on HOST. Full bounded storage arrays are read through the plan's
    existing transfer counters, then selected by storage row and global tick
    slot. Empty world selection performs no device read or synchronization.
    """
    worlds = _worlds(storage, worlds)
    batch = plan._batch
    if storage.batch is not batch:
        raise ValueError("floating contact storage belongs to a different batch")
    if isinstance(bank, (bool, np.bool_)) or not isinstance(bank, Integral) or bank not in (0, 1):
        raise ValueError("floating contact bank must be zero or one")
    bank = int(bank)
    if any(w not in plan._contact_worlds or w not in batch._slots for w in worlds):
        raise ValueError("floating contact packet requested an unregistered world")
    if not worlds:
        return {}
    with batch._device():
        batch._sync()
        roots, states, applied, fk, twist, energy, ticks = [
            plan._read_contact(buffer)
            for buffer in (
                storage.roots[bank],
                storage.states[bank],
                storage.applied[bank],
                storage.fk[bank],
                storage.twist[bank],
                storage.energy[bank],
                batch._ticks[bank],
            )
        ]
    count, body_max, velocity_max = storage.count, storage.bmax, storage.dmax
    roots = _array(roots, (count, 13), np.float32, "root bank")
    states = _array(states, (count, storage.npad, 7), np.float32, "joint bank")
    applied = _array(applied, (count, velocity_max), np.float32, "generalized load bank")
    fk = _array(fk, (count, body_max, 18), np.float32, "FK bank")
    twist = _array(twist, (count, body_max, 6), np.float32, "COM twist bank")
    energy = _array(energy, (count,), np.float32, "kinetic energy bank")
    ticks = np.asarray(ticks)
    if ticks.ndim != 1 or ticks.dtype != np.dtype(np.int32):
        raise ValueError("floating contact ticks must retain the batch int32 vector")
    packets = {}
    for world in worlds:
        row, spec = storage.rows[world], storage.specs[world]
        n, bodies, dofs = spec.n, spec.b, spec.d
        slot = batch._slots[world]
        if not (0 <= n <= 16 and bodies == n + 1 and dofs == n + 6):
            raise ValueError("floating contact physical B/D/N mismatch")
        if not 0 <= slot < len(ticks) or int(ticks[slot]) < 0:
            raise ValueError("floating contact world has an invalid global tick slot")
        body = fk[row, :bodies]
        velocity = twist[row, :bodies]
        _finite(roots[row], states[row, :n], applied[row, :dofs], body, velocity, energy[row])
        tick = int(ticks[slot])
        checkpoint = spec.checkpoint(
            world, roots[row], states[row], applied[row], tick, storage._packet(world)
        )
        # The actual bank's local origin remains in checkpoint.local_origin_m.
        # World positions are FP64 expansions; never invert their rounded sum
        # to reconstruct the local root state or normalize stored rotations.
        with np.errstate(over="raise", invalid="raise"):
            position = body[:, :3].astype(np.float64) + spec.anchor
            rotation = body[:, 3:12].astype(np.float64).reshape(bodies, 3, 3)
            com = body[:, 15:18].astype(np.float64) + spec.anchor
        _finite(position, rotation, com)
        geometry = FloatingTreeKinematics(
            tick,
            spec.joint_types,
            checkpoint.position,
            checkpoint.velocity,
            position,
            rotation,
            com,
            velocity[:, :3],
            velocity[:, 3:],
            n,
            bodies,
            dofs,
            spec.q_count,
        )
        frames = {}
        for shape in plan._model.shapes:
            if shape.key.world != world:
                continue
            binding = plan._contact_bindings[shape.key]
            if isinstance(binding, ResidentFloatingTreeBinding):
                group = binding.group_index
                if binding.world != world or binding.key != shape.key or not 0 <= group < bodies:
                    raise ValueError("floating material binding does not identify a real body")
                p, r = position[group], rotation[group]
                points = _material_points(p, r, binding.local_points_m)

                frame = ContactShapeFrame(
                    shape, points, group, p, r, com[group], velocity[group, :3], velocity[group, 3:]
                )
            elif isinstance(binding, ResidentStaticBinding):
                points = binding.motion.start
                with np.errstate(over="raise", invalid="raise"):
                    reference = np.mean(points, axis=0)
                frame = ContactShapeFrame(
                    shape, points, None, np.zeros(3), np.eye(3), reference, np.zeros(3), np.zeros(3)
                )
            else:
                raise ValueError(
                    "floating contact packet only accepts its own body or static bindings"
                )
            frames[shape.key] = frame
        packets[world] = dict(
            state=checkpoint,
            geometry=geometry,
            frames=MappingProxyType(frames),
            energy=float(energy[row]),
        )
    return packets


def floating_contact_diagnostics(storage, worlds, arrays):
    """Build original diagnostics from this attempt's pre-swap readback only.

    ``arrays`` has linear/angular/mass/vector/entry_pose/info/diag, with the
    complete storage row dimension. Caller selects accepted worlds after the
    existing finalize/status checks. No active-state read, validity mutation,
    new device transfer, or mechanics occurs here.
    """
    worlds = _worlds(storage, worlds)
    if not isinstance(arrays, Mapping):
        raise TypeError("floating contact diagnostics need a same-attempt array mapping")
    if not worlds:
        return {}
    count, bodies, dofs = storage.count, storage.bmax, storage.dmax
    shapes = {
        "linear": (count, bodies, 3, dofs),
        "angular": (count, bodies, 3, dofs),
        "mass": (count, dofs, dofs),
        "vector": (count, dofs, 14),
        "entry_pose": (count, 7),
        "info": (count, 4),
        "diag": (count, 3),
    }
    if not set(shapes) <= arrays.keys():
        raise ValueError("floating contact diagnostic readback is incomplete")
    values = {
        name: _array(arrays[name], shape, np.int32 if name == "info" else np.float32, name)
        for name, shape in shapes.items()
    }
    result = {}
    for world in worlds:
        row, spec = storage.rows[world], storage.specs[world]
        n, b, d = spec.n, spec.b, spec.d
        if not (0 <= n <= 16 and b == n + 1 and d == n + 6):
            raise ValueError("floating contact diagnostic B/D/N mismatch")
        info, diag = values["info"][row], values["diag"][row]
        if int(info[2]) != 0:
            raise ValueError("rejected floating attempt cannot publish an accepted diagnostic")
        if int(info[0]) < 0 or int(info[1]) != int(info[0]) + 1:
            raise ValueError("floating contact diagnostic ticks are not one physical step")
        vector = values["vector"][row, :d]
        pose = values["entry_pose"][row]
        linear = values["linear"][row, :b, :, :d]
        angular = values["angular"][row, :b, :, :d]
        mass = values["mass"][row, :d, :d]
        _finite(vector, pose, linear, angular, mass, diag)
        with np.errstate(over="raise", invalid="raise"):
            q = np.r_[spec.anchor + pose[:3].astype(np.float64), pose[3:7], vector[6:, 4]]
        _finite(q)
        result[world] = FloatingTreeDiagnostic(
            int(info[0]),
            int(info[1]),
            float(diag[0]),
            q,
            vector[:, 5],
            linear,
            angular,
            mass,
            vector[:, 0],
            vector[:, 1],
            vector[6:, 2],
            vector[:, 3],
            int(info[3]),
            float(diag[1]),
            float(diag[2]),
        )
    return result
