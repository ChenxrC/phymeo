"""Scene-specific checks using state differences rather than solver geometry code."""

import numpy as np


def audit_substep(scene, before, after, report, h, *, tolerance, reaction):
    """Audit every accepted sample, including external gravity and prescribed motion.

    Distance constitutive convergence is explicitly a reported solver metric;
    penetration, normal velocity, fixed poses and rigid momentum are recomputed.
    Pinned cloth exchanges external momentum, so no closed-system claim is made.
    """
    state_arrays = (
        after.particles.position,
        after.particles.velocity,
        after.rigid.position,
        after.rigid.orientation,
        after.rigid.velocity,
        after.rigid.angular_velocity,
    )
    if not all(np.isfinite(value).all() for value in state_arrays):
        raise AssertionError("non-finite accepted state")
    residuals = np.array(
        [report.distance_residual, report.contact_residual, report.velocity_residual]
    )
    if not report.accepted or not np.isfinite(residuals).all() or np.max(residuals) > tolerance:
        raise AssertionError("reported residual failed acceptance")
    p = after.particles.position.astype(np.float64)
    c = after.rigid.position.astype(np.float64)
    pin_error = float(
        np.max(
            np.abs(p[scene.pinned] - scene.initial_state.particles.position[scene.pinned]),
            initial=0,
        )
    )
    if pin_error != 0:
        raise AssertionError("pinned cloth moved")
    if np.any(after.particles.velocity[scene.pinned] != 0):
        raise AssertionError("pinned cloth acquired velocity")
    quaternion_error = float(
        np.max(
            np.abs(np.linalg.norm(after.rigid.orientation.astype(np.float64), axis=1) - 1),
            initial=0,
        )
    )
    if quaternion_error > 1e-6:
        raise AssertionError("rigid quaternion is not unit length")
    if np.any(after.rigid.angular_velocity != 0) or not np.array_equal(
        after.rigid.orientation, scene.initial_state.rigid.orientation
    ):
        raise AssertionError(
            "sphere rotated despite zero initial spin, external torque and normal contact torque"
        )
    separation = p[:, None, :] - c[None, :, :]
    distances = np.sqrt(np.sum(separation * separation, axis=2))
    if np.any(distances == 0):
        raise AssertionError("accepted point coincides with sphere center")
    gaps = distances - scene.spheres.radii.astype(np.float64)[None, :]
    penetration = float(np.max(np.maximum(-gaps, 0), initial=0))
    if penetration > 2e-6:
        raise AssertionError("independent vertex penetration exceeds 2e-6 m")
    normals = separation / distances[:, :, None]
    relative = (
        after.particles.velocity.astype(np.float64)[:, None, :]
        - after.rigid.velocity.astype(np.float64)[None, :, :]
    )
    normal_speed = np.sum(relative * normals, axis=2)
    active = gaps <= tolerance  # Both reference length and speed are 1 in this scene.
    violation = float(np.max(np.maximum(-normal_speed[active], 0), initial=0))
    eps = np.finfo(after.particles.position.dtype).eps
    velocity_roundoff = 64 * eps * max(1.0, float(np.max(np.abs(relative), initial=0)))
    if violation > tolerance + velocity_roundoff:
        raise AssertionError("independent normal velocity residual exceeds tolerance")
    position_load, velocity_load = report.position_impulses, report.velocity_impulses
    if position_load is None or velocity_load is None:
        raise AssertionError("accepted substep has no contact impulse records")
    for load in (position_load, velocity_load):
        if not all(
            np.isfinite(value).all()
            for value in (load.particle, load.rigid_linear, load.rigid_angular)
        ):
            raise AssertionError("non-finite contact impulse record")
        if np.any(load.rigid_angular != 0):
            raise AssertionError("normal sphere contact produced torque about its center")
    contact_p = position_load.particle + velocity_load.particle
    contact_b = position_load.rigid_linear + velocity_load.rigid_linear
    ledger_error = float(np.linalg.norm(contact_p.sum(axis=0) + contact_b.sum(axis=0)))
    if ledger_error > 1e-10:
        raise AssertionError("contact ledger does not balance")
    mass = 1.0 / scene.rigid_model.inverse_mass.astype(np.float64)
    observed = mass[:, None] * (
        after.rigid.velocity.astype(np.float64) - before.rigid.velocity.astype(np.float64)
    )
    external = h * (
        mass[:, None] * scene.rigid_model.gravity.astype(np.float64) + scene.inputs.rigid.force
    )
    expected_contact = contact_b if reaction else np.zeros_like(contact_b)
    balance_error = float(np.max(np.abs(observed - external - expected_contact), initial=0))
    # FP32 corrected-position reconstruction quantizes velocity by position ULP/h.
    balance_limit = max(
        1e-10, 64 * eps * float(np.max(mass)) * max(1.0, float(np.max(np.abs(c)))) / h
    )
    if balance_error > balance_limit:
        raise AssertionError("independent rigid momentum balance failed")
    return {
        "vertex_penetration_m": penetration,
        "pin_displacement_m": pin_error,
        "quaternion_norm_error": quaternion_error,
        "reported_residual": float(np.max(residuals)),
        "normal_velocity_violation_mps": violation,
        "contact_ledger_imbalance_kg_mps": ledger_error,
        "rigid_motion_balance_error_kg_mps": balance_error,
        "rigid_balance_limit_kg_mps": balance_limit,
        "contact": bool(np.any(contact_p != 0)),
        "position_iterations": report.iterations,
        "velocity_iterations": report.velocity_iterations,
    }
