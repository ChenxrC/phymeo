"""One weighted compliant law for geometry-independent bilateral contact sites."""

import numpy as np

from ..constraints.robot_contact import RobotContactMaterial
from ..geometry.contact_patch_adapter import ContactPatchSample
from .sampled_box_contact import sample_loads


def sample_patch_loads(sample, law, first_site_velocity, second_site_velocity):
    """Return each owner's force at the site's common load point, in world N.

    The law's N/m and Ns/m constants belong to the whole patch. Site weights
    enter exactly once after evaluation. Callers supply true v + omega x r (or
    the corresponding material Jacobian velocity); this function never advances
    or publishes any owner.
    """
    if not isinstance(sample, ContactPatchSample) or not isinstance(law, RobotContactMaterial):
        raise TypeError("expected a contact patch and RobotContactMaterial")
    count = len(sample.sites)
    first, second = (
        np.asarray(v, dtype=float) for v in (first_site_velocity, second_site_velocity)
    )
    if (
        first.shape != (count, 3)
        or second.shape != first.shape
        or not np.isfinite(first).all()
        or not np.isfinite(second).all()
    ):
        raise ValueError("one finite world velocity per material contact site is required")
    depths = np.array([max(0.0, -s.witness.gap_m) for s in sample.sites])
    # Existing law normals point outward onto the participant receiving the force.
    normals = -np.array([s.witness.normal for s in sample.sites])
    _, force = sample_loads(depths, normals, first - second, law)
    force *= np.array([s.weight for s in sample.sites])[:, None]
    if not np.isfinite(force).all():
        raise ValueError("nonfinite weighted contact load")
    first_force = np.frombuffer(force.tobytes(), dtype=float).reshape(count, 3)
    second_force = np.frombuffer((-force).tobytes(), dtype=float).reshape(count, 3)
    return {"first_force": first_force, "second_force": second_force}
