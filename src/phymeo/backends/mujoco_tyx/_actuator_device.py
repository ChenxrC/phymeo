"""Shared activation bank/stage transport for T1/T2/T3/T4.

The kernel preserves the accepted filter law and composes native JOINT moments. This
module only validates/allocates its banks and composes it with each existing
mechanical route; it never advances mechanical state or publishes a bank.
"""

import numpy as np


def validate(plan, activation, world_count):
    enabled = hasattr(plan, "activation")
    if enabled:
        if (
            not isinstance(activation, np.ndarray)
            or activation.dtype != np.float32
            or activation.shape != (world_count, plan.activation.na)
            or not np.isfinite(activation).all()
        ):
            raise ValueError("stateful TYX requires the complete FP32 activation bank")
    elif activation is not None:
        raise ValueError("stateless TYX cannot receive an activation bank")
    return enabled


def allocate(device, activation):
    """Called in the owning device runtime/resource transaction, once."""
    if not device.activation_enabled:
        return
    from . import _actuator_kernels

    device.activation_kernels = (_actuator_kernels.update, _actuator_kernels.update_other)
    device.act = (device._allocate("act.0", activation), device._allocate("act.1", activation))
    for name in (
        "activation_ids",
        "activation_values",
        "activation_derivative",
        "integration_derivative",
        "actuator_fields",
    ):
        setattr(device, name, device._allocate(name))


def advance(device, old, new):
    """After original mechanics(nu=0), before original constraints/solve."""
    if not device.activation_enabled:
        return
    device.activation_kernels[old](
        device.nworld,
        device.plan.nu,
        device.plan.activation.na,
        device.plan.nv,
        device.h,
        device.activation_ids,
        device.activation_values,
        device.q[old],
        device.v[old],
        device.controls,
        device.act[old],
        device.act[new],
        device.activation_derivative,
        device.integration_derivative,
        device.actuator_fields,
        device.actuator,
        device.applied,
        device.status,
        int(getattr(device, "transmission_enabled", False)),
        getattr(device, "transmission_offsets", device.status),
        getattr(device, "transmission_dofs", device.status),
        getattr(device, "transmission_moment", device.gravity),
        getattr(device, "transmission_fields", device.actuator_fields),
        mode=device.mode,
    )
    device.counts["completed_kernels"] += 1


def download(device, result, old, target):
    """Return actual owned bytes; accepted publication stays in the runtime."""
    if device.activation_enabled:
        result.update(
            {
                name: device._get(buffer)
                for name, buffer in (
                    ("act", device.act[target]),
                    ("input_act", device.act[old]),
                    ("activation_derivative", device.activation_derivative),
                    ("integration_derivative", device.integration_derivative),
                    ("actuator_fields", device.actuator_fields),
                )
            }
        )
