"""Cold shared-model scalar drive edits, preserving native array/Graph owners."""

from numbers import Real

import numpy as np


def parameters(runtime):
    control = runtime._controls_binding
    if control is None:
        return {}
    cpu = runtime._cpu_model
    result = {}
    for name, actuator in zip(control.drive_names, control.native_actuators):
        if cpu.actuator_forcelimited[actuator] and (
                cpu.actuator_forcerange[actuator, 0] != -cpu.actuator_forcerange[actuator, 1]
                or cpu.actuator_forcerange[actuator, 1] <= 0):
            raise NotImplementedError('scalar drive parameter contract requires symmetric effort limits')
        result[name] = dict(kp=float(-cpu.actuator_biasprm[actuator, 1]),
                            kd=float(-cpu.actuator_biasprm[actuator, 2]),
                            effort_limit=float(cpu.actuator_forcerange[actuator, 1])
                            if cpu.actuator_forcelimited[actuator] else 0.)
    return result


def update(runtime, updates):
    """All worlds share these model parameters. Targets and reset commands stay intact."""
    runtime._require_idle()
    if runtime._stream.is_capturing:
        raise RuntimeError('drive parameter edits are unavailable during capture')
    current = parameters(runtime)
    if not isinstance(updates, dict) or not updates:
        raise ValueError('drive updates require a nonempty mapping')
    for name, values in updates.items():
        if name not in current:
            raise ValueError(f'unknown prepared scalar drive: {name}')
        if not isinstance(values, dict) or not values or set(values) - {'kp', 'kd', 'effort_limit'}:
            raise ValueError('drive parameter fields require kp, kd or effort_limit')
        for field, value in values.items():
            if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                    or not np.isfinite(value) or not 0 <= value <= float(np.finfo(np.float32).max)):
                raise ValueError(f'{name}.{field} requires nonnegative finite FP32')
    effective = {name: {**values, **updates.get(name, {})} for name, values in current.items()}
    if effective == current:
        return effective
    cpu, model, control = runtime._cpu_model, runtime._model, runtime._controls_binding
    originals = {name: getattr(cpu, name).copy() for name in (
        'actuator_biasprm', 'actuator_forcelimited', 'actuator_forcerange')}
    proposed = {name: array.copy() for name, array in originals.items()}
    coefficients = []
    old_coefficients = []
    for name, actuator in zip(control.drive_names, control.native_actuators):
        row = effective[name]
        proposed['actuator_biasprm'][actuator, 1:3] = [-row['kp'], -row['kd']]
        if 'effort_limit' in updates.get(name, {}):
            proposed['actuator_forcelimited'][actuator] = bool(row['effort_limit'])
            proposed['actuator_forcerange'][actuator] = [-row['effort_limit'], row['effort_limit']]
        coefficients.extend((row['kp'], row['kd']))
        old_coefficients.extend((current[name]['kp'], current[name]['kd']))
    tx = runtime._tx

    def upload(values, gains):
        for name, array in values.items():
            boolean = name == 'actuator_forcelimited'
            data = np.asarray(array, dtype=np.bool_ if boolean else np.float32).reshape(-1)
            target = getattr(model, name).view(tx.bool1 if boolean else tx.f32).flatten()
            target.assign(data, stream=runtime._stream)
            runtime._drive_parameter_upload_bytes += data.nbytes
        data = np.asarray(gains, np.float32)
        control.coefficients.assign(data, stream=runtime._stream)
        runtime._drive_parameter_upload_bytes += data.nbytes

    try:
        upload(proposed, coefficients)
        # A finite gain can overflow when multiplied by a finite target. Check
        # current AND reset commands before invoking the native forward. Only a
        # compact flag crosses the explicit cold completion boundary.
        runtime._validation.clear(runtime._stream)
        for commands in (control.flat[runtime._active], control.initial_flat):
            control.validate_commands(commands, runtime._stream)
        runtime._validation.read(runtime._stream)
        # Publish refreshed observations only after successful completion. The
        # fixed Graph retains the same model/coefficient pointers throughout.
        with runtime._synchronous_execution() as ticket:
            runtime.enqueue_update(ticket, refresh_state=True, compose_controls=True)
    except Exception as primary:
        try:
            runtime._complete()
            upload(originals, old_coefficients)
            runtime._validation.clear(runtime._stream)
            runtime._complete()
            runtime._drive_parameter_rollbacks += 1
        except Exception as rollback:
            runtime._closing = True  # Keep all owners; only explicit close retry is safe.
            raise RuntimeError('drive parameter update and rollback failed; close required') from rollback
        raise primary
    for name, array in proposed.items():
        getattr(cpu, name)[:] = array
    if any(effective[name][field] != current[name][field]
           for name in current for field in ('kp', 'kd')):
        control.recompose_reset = True
    runtime._drive_parameter_updates += 1
    return effective
