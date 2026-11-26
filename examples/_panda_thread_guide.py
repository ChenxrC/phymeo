"""Reduced passive guide for a pair already engaged in a single-start thread.

Positive unwrapped relative rotation advances the pin into the board. Depth and
turn are increments from the same engaged reset. Both Jacobians map the full
model's generalized velocities to the relative rates between part and hole.
"""

import math

import numpy as np


PITCH_M_PER_TURN = .0015
LEAD_M_PER_RAD = PITCH_M_PER_TURN / math.tau
STIFFNESS_N_PER_M = 12_000.
DAMPING_NS_PER_M = 12.
# Assumed kinetic turning drag for this reduced model. Smoothing at zero
# speed gives a continuous reversal; no static breakaway torque is imposed.
TURN_DRAG_NM = .025
DRAG_TRANSITION_RAD_S = .05


def implicit_turn_drag(free_rate, inverse_inertia, dt):
    """Return end-of-step relative spin and the drag torque producing it.

    free_rate includes the step's other forces. inverse_inertia is the scalar
    relative rotational mobility J M^-1 J^T. Use the returned torque once,
    in place of the explicit friction term in thread_reaction.
    """
    values = np.asarray([free_rate, inverse_inertia, dt], dtype=float)
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError("free rate, inverse inertia and step duration must be finite scalars")
    free, mobility, step = map(float, values)
    if mobility < 0. or step < 0.:
        raise ValueError("inverse inertia and step duration must be nonnegative")
    if free == 0.:
        return 0., 0.
    drop = (step*TURN_DRAG_NM)*mobility
    if not math.isfinite(drop):
        raise ValueError("maximum drag-induced rate change is not finite")
    speed = abs(free)
    # The scalar equation is odd and strictly increasing. Its positive root
    # lies between the fully saturated prediction and the no-drag speed.
    lo, hi = max(0., speed-drop), speed
    x = max(lo, speed/(1.+drop/DRAG_TRANSITION_RAD_S))
    for _ in range(64):
        tangent = math.tanh(x/DRAG_TRANSITION_RAD_S)
        residual = (x-speed)+drop*tangent
        if abs(residual) <= 4.*math.ulp(speed):
            break
        if residual < 0.:
            lo = x
        else:
            hi = x
        slope = 1.+drop*(1.-tangent*tangent)/DRAG_TRANSITION_RAD_S
        trial = x-residual/slope
        x = trial if lo < trial < hi else lo+.5*(hi-lo)
    predicted = math.copysign(x, free)
    drag = -TURN_DRAG_NM*math.tanh(predicted/DRAG_TRANSITION_RAD_S)
    return predicted, drag


def thread_reaction(depth_change, turn_change, depth_jac, turn_jac, qvel):
    """Return generalized external force conjugate to qvel and signed diagnostics.

    Entries conjugate to translation have units N, and entries conjugate to
    rotation have units N m. All three vectors use the model's DOF order.

    The zero-backlash spring energy is 0.5*k*(depth - lead*turn)**2.
    Its gradient pulls the coordinates toward the helix. The dashpot acts
    on that same mismatch rate, independently of rotational friction.
    """
    depth, turn = np.asarray(depth_change, dtype=float), np.asarray(turn_change, dtype=float)
    if depth.ndim or turn.ndim or not np.isfinite(depth) or not np.isfinite(turn):
        raise ValueError("depth change (m) and unwrapped turn (rad) must be finite scalars")
    jz, jt, v = (np.asarray(a, dtype=float) for a in (depth_jac, turn_jac, qvel))
    if (v.ndim != 1 or v.size == 0 or jz.shape != v.shape or jt.shape != v.shape
            or not all(np.isfinite(a).all() for a in (jz, jt, v))):
        raise ValueError("both relative Jacobians and qvel must be equal-length finite nonempty vectors")
    error = float(depth - LEAD_M_PER_RAD*turn)
    error_jac = jz - LEAD_M_PER_RAD*jt
    error_rate = float(error_jac@v)
    angular_rate = float(jt@v)
    axial_force = -STIFFNESS_N_PER_M*error - DAMPING_NS_PER_M*error_rate
    helix_torque = -LEAD_M_PER_RAD*axial_force
    friction = -TURN_DRAG_NM*math.tanh(angular_rate/DRAG_TRANSITION_RAD_S)
    tau = axial_force*error_jac + friction*jt
    dissipation = DAMPING_NS_PER_M*error_rate**2 - friction*angular_rate
    metrics = dict(
        pitch_error_m=error,
        pitch_error_rate_m_s=error_rate,
        axial_force_n=axial_force,
        helix_torque_nm=helix_torque,
        friction_torque_nm=friction,
        total_turn_torque_nm=helix_torque+friction,
        elastic_energy_j=.5*STIFFNESS_N_PER_M*error**2,
        elastic_energy_rate_w=STIFFNESS_N_PER_M*error*error_rate,
        dissipation_w=dissipation,
        external_power_w=float(tau@v),
    )
    return tau, metrics
