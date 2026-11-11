# Copyright 2026 Phymeo contributors
# Native semantics: MuJoCo 3.3.7 engine_forward.c, Apache-2.0.
"""One actual device activation/force stage, between unchanged mechanics and solve."""
import tyx as tx


def update(
    count: tx.i32, nu: tx.i32, na: tx.i32, nv: tx.i32, h: tx.f32,
    ids: tx.ndarray(tx.i32, 2), values: tx.ndarray(tx.f32, 2),
    q: tx.ndarray(tx.f32, 2), v: tx.ndarray(tx.f32, 2), controls: tx.ndarray(tx.f32, 2),
    activation: tx.ndarray(tx.f32, 2), next_activation: tx.ndarray(tx.f32, 2),
    derivative: tx.ndarray(tx.f32, 2), integrated_derivative: tx.ndarray(tx.f32, 2),
    fields: tx.ndarray(tx.f32, 3), actuator: tx.ndarray(tx.f32, 2),
    applied: tx.ndarray(tx.f32, 2), status: tx.ndarray(tx.i32, 1),
):
    for w in tx.range(count):
        for index in range(na):
            next_activation[w, index] = activation[w, index]
            derivative[w, index] = 0.0
            integrated_derivative[w, index] = 0.0
        for actor in range(nu):
            qa = ids[actor, 0]
            dof = ids[actor, 1]
            adr = ids[actor, 3]
            ctrl = controls[w, actor]
            if ids[actor, 6] != 0:
                ctrl = tx.min(values[actor, 6], tx.max(values[actor, 5], ctrl))
            force_input = ctrl
            if adr >= 0:
                entry = activation[w, adr]
                tau = values[actor, 9]
                raw_derivative = (ctrl - entry) / tau
                derivative[w, adr] = raw_derivative
                actual_derivative = raw_derivative
                if ids[actor, 5] != 0:
                    actual_derivative = 0.0
                integrated_derivative[w, adr] = actual_derivative
                candidate = entry
                if ids[actor, 5] == 0:
                    ratio = h / tau
                    # Stable evaluation of 1-exp(-ratio). This small-x series
                    # has truncation below FP32 rounding at the branch point.
                    factor = 0.0
                    if ratio < 0.01:
                        factor = ratio * (1.0 + ratio * (-0.5 + ratio * (1.0/6.0 + ratio * (-1.0/24.0 + ratio/120.0))))
                    else:
                        factor = 1.0 - tx.exp(-ratio)
                    candidate = entry + (ctrl-entry) * factor
                    if not tx.isfinite(ratio) or not tx.isfinite(factor):
                        status[w] = 2
                if ids[actor, 7] != 0:
                    candidate = tx.min(values[actor, 11], tx.max(values[actor, 10], candidate))
                next_activation[w, adr] = candidate
                force_input = entry
                if ids[actor, 4] != 0:
                    force_input = candidate
                if not tx.isfinite(raw_derivative) or not tx.isfinite(candidate):
                    status[w] = 2
            raw_force = 0.0
            gear = values[actor, 0]
            if ids[actor, 5] == 0:
                raw_force = (values[actor, 1]*force_input + values[actor, 2]
                             + values[actor, 3]*gear*q[w, qa]
                             + values[actor, 4]*gear*v[w, dof])
            force = raw_force
            if ids[actor, 8] != 0:
                force = tx.min(values[actor, 8], tx.max(values[actor, 7], force))
            fields[w, actor, 0] = ctrl
            fields[w, actor, 1] = force_input
            fields[w, actor, 2] = raw_force
            fields[w, actor, 3] = force
            actuator[w, dof] = actuator[w, dof] + gear*force
            if (not tx.isfinite(ctrl) or not tx.isfinite(force_input)
                    or not tx.isfinite(raw_force) or not tx.isfinite(force)):
                status[w] = 2
        for dof in range(nv):
            applied[w, dof] = applied[w, dof] + actuator[w, dof]
            if not tx.isfinite(actuator[w, dof]) or not tx.isfinite(applied[w, dof]):
                status[w] = 2


update_other = tx.kernel(update)
update = tx.kernel(update)
