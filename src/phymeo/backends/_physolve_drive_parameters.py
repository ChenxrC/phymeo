"""Cold shared-model scalar drive edits, preserving native array/Graph owners."""

from numbers import Real

import numpy as np


def parameters(runtime):
    control = runtime._controls_binding
    if control is None:
        return {}
    cpu = runtime._cpu_model
    if cpu is None:
        current = runtime._autonomous_drive_parameters
        if current is not None:
            return {name: values.copy() for name, values in current.items()}
        compilation = runtime._compilation
        return {
            name: dict(kp=row.kp, kd=row.kd, effort_limit=row.effort_limit, armature=row.armature)
            for name, row in zip(compilation.joint_names, compilation.joints, strict=True)
        }
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


def _armature_edits(runtime, effective):
    """Prepare a model-only journal for scalar and derived inertia slices."""
    model, compilation = runtime._model, runtime._compilation
    armatures = [effective[name]["armature"] for name in compilation.joint_names]
    mean, body_weights, dof_weights, actuator_acc = compilation.reference_inertia(armatures)
    # Independent free objects retain their existing inverse weights. Include only
    # their unchanged diagonal inertia in the aggregate solver scale.
    trace = mean * len(dof_weights) + sum(
        3 * mass + sum(inertia) for _, mass, inertia in compilation.free_objects
    )
    with np.errstate(over="ignore", under="ignore"):
        mean = np.float32(trace / model.nv)
        scale = mean * np.float32(model.nv)
    # Native scales residuals with the rounded mean multiplied by the DOF count.
    if not np.isfinite(scale) or scale <= 0.:
        raise ValueError("armature update requires a finite FP32 solver inertia scale")

    _, _, dof_start = compilation.scalar_offsets
    first_body = 1 if compilation.extended or compilation.has_free_root else 2
    edits = (
        (model.dof_armature, dof_start, armatures),
        (model.body_invweight0, 2 * first_body, body_weights),
        (model.dof_invweight0, 0, dof_weights),
        (model.actuator_acc0, 0, actuator_acc),
        (model.stat.meaninertia, 0, [mean]),
    )
    journal = []
    for array, start, values in edits:
        with np.errstate(over="ignore", under="ignore"):
            proposed = np.asarray(values, dtype=np.float32).reshape(-1)
        if not np.isfinite(proposed).all():
            raise ValueError("armature reference inertia must be representable in FP32")
        target = array.view(runtime._tx.f32).flatten()[start:start + proposed.size]
        journal.append((target, target.numpy(stream=runtime._stream), proposed))
    return journal


def update(runtime, updates):
    """All worlds share these model parameters. Targets and reset commands stay intact."""
    runtime._require_idle()
    if runtime._stream.is_capturing:
        raise RuntimeError('drive parameter edits are unavailable during capture')
    current = parameters(runtime)
    if not isinstance(updates, dict) or not updates:
        raise ValueError('drive updates require a nonempty mapping')
    allowed = {'kp', 'kd', 'effort_limit'}
    if runtime.supports_scalar_armature_updates:
        allowed.add('armature')
    for name, values in updates.items():
        if name not in current:
            raise ValueError(f'unknown prepared scalar drive: {name}')
        if not isinstance(values, dict) or not values or set(values) - allowed:
            raise ValueError(f"unsupported drive parameter patch, expected fields {sorted(allowed)}")
        for field, value in values.items():
            if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                    or not np.isfinite(value) or not 0 <= value <= float(np.finfo(np.float32).max)):
                raise ValueError(f'{name}.{field} requires nonnegative finite FP32')
    effective = {name: {**values, **updates.get(name, {})} for name, values in current.items()}
    if effective == current:
        return effective
    armature = []
    if any('armature' in patch and effective[name]['armature'] != current[name]['armature']
           for name, patch in updates.items()):
        armature = _armature_edits(runtime, effective)
    cpu, model, control = runtime._cpu_model, runtime._model, runtime._controls_binding
    fields = ('actuator_biasprm', 'actuator_forcelimited', 'actuator_forcerange')
    if cpu is None:
        # One cold transaction snapshots only the model arrays it will edit.
        originals = {name: getattr(model, name).numpy(stream=runtime._stream) for name in fields}
        originals['actuator_biasprm'] = originals['actuator_biasprm'][0]
        originals['actuator_forcerange'] = originals['actuator_forcerange'][0]
    else:
        originals = {name: getattr(cpu, name).copy() for name in fields}
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

    def upload(values, gains, *, rollback=False):
        for name, array in values.items():
            boolean = name == 'actuator_forcelimited'
            data = np.asarray(array, dtype=np.bool_ if boolean else np.float32).reshape(-1)
            target = getattr(model, name).view(tx.bool1 if boolean else tx.f32).flatten()
            target.assign(data, stream=runtime._stream)
            runtime._drive_parameter_upload_bytes += data.nbytes
        data = np.asarray(gains, np.float32)
        control.coefficients.assign(data, stream=runtime._stream)
        runtime._drive_parameter_upload_bytes += data.nbytes
        for target, original, proposed in armature:
            data = original if rollback else proposed
            target.assign(data, stream=runtime._stream)
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
            upload(originals, old_coefficients, rollback=True)
            runtime._validation.clear(runtime._stream)
            runtime._complete()
            runtime._drive_parameter_rollbacks += 1
        except Exception as rollback:
            runtime._closing = True  # Keep all owners; only explicit close retry is safe.
            raise RuntimeError('drive parameter update and rollback failed; close required') from rollback
        raise primary
    if cpu is None:
        runtime._autonomous_drive_parameters = {name: row.copy() for name, row in effective.items()}
    else:
        for name, array in proposed.items():
            getattr(cpu, name)[:] = array
    if any(effective[name][field] != current[name][field]
           for name in current for field in ('kp', 'kd')):
        control.recompose_reset = True
    runtime._drive_parameter_updates += 1
    return effective
