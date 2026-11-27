# Copyright 2026 Phymeo contributors. Apache-2.0; see LICENSE and NOTICE.
"""Explicit CPU oracle using certified geometry and independent native mechanics.

Never imported by resident production workspaces. No GPU Jacobian, impedance,
reference acceleration, force, or solved state is used to build this reference.
The candidate acceleration is only a test vector for the balance equation.
"""

import numpy as np

from .convex_contact import certify_contact, match_contact_pairs


def certify_world(model, data, catalog, query, contacts, world):
    """Certify the full active single-point convex set at this independent pose.

    Box-box multimanifolds require their own certificate and reject here. This
    limited domain is explicit, including when a new moving state leaves it.
    """
    count = len(contacts["counts"])
    if type(world) is not int or not 0 <= world < count:
        raise ValueError("invalid world")
    capacity, remainder = divmod(len(contacts["pairs"]), count)
    ncontact = int(contacts["counts"][world])
    if remainder or not 0 <= ncontact <= capacity or contacts["status"][world] != 0:
        raise ValueError("contact status/capacity rejected")
    wi = np.flatnonzero(query["world"] == world)
    if np.any(query["meta"][wi, 0] != 1):
        raise ValueError("unresolved query")
    active = wi[query["gap"][wi, 0] < 0]
    pair = query["geom_ids"][active]
    if np.any(np.all(model.geom_type[pair] == 6, axis=1)):
        raise ValueError("box-box multimanifold certificate not implemented")
    raw_ix = world * capacity + np.arange(ncontact)
    rowmap = match_contact_pairs(pair, contacts["pairs"][raw_ix])
    native_pairs = np.asarray([c.geom for c in data.contact], dtype=np.int64).reshape(-1, 2)
    nativemap = match_contact_pairs(pair, native_pairs)
    certificates, diagnostics = [], []
    for qi, row, ni in zip(active, raw_ix[rowmap], nativemap, strict=True):
        ids = query["material_ids"][qi].copy()
        # Query material IDs are world-global. Validate this world's ownership
        # before converting, so an ID from a previous world cannot become -1.
        used = ids >= 0
        if np.any(ids[used] < world * len(catalog.points)) or np.any(
            ids[used] >= (world + 1) * len(catalog.points)
        ):
            raise ValueError("cross-world material ID")
        ids[used] -= world * len(catalog.points)
        if np.any(query["skin_offset"][qi] != 0):
            raise ValueError("nonzero skin unsupported")
        cert = certify_contact(
            catalog.points,
            catalog.offsets,
            data.geom_xpos,
            data.geom_xmat.reshape(-1, 3, 3),
            query["geom_ids"][qi],
            ids,
            query["barycentric"][qi],
            query["surface"][qi].astype(float) + query["surface_low"][qi],
            query["normal"][qi],
            float(query["gap"][qi, 0]),
            query["witness_receipt"][qi],
        )
        geo = contacts["geometry"][row].astype(float)
        point = geo[1:4] + contacts["point_low"][row]
        if (
            not np.isfinite(geo).all()
            or not np.isfinite(point).all()
            or (
                np.linalg.norm(point - cert.point) > 2e-9
                or abs(geo[0] - cert.gap) > 2e-9
                or np.linalg.norm(geo[4:7] - cert.normal) > 1e-6
            )
        ):
            raise ValueError("assembled contact disagrees with certified geometry")
        native = data.contact[int(ni)]
        # An unmatched physical normal/depth is still a failure, rather than
        # papering over it by replacing an arbitrary reference contact.
        if (
            abs(native.dist - cert.gap) > 2e-9
            or np.linalg.norm(native.frame[:3] - cert.normal) > 1e-6
        ):
            raise ValueError("native normal/gap disagreement")
        certificates.append((int(ni), cert))
        diagnostics.append(
            dict(
                pair=list(cert.pair),
                unique=cert.unique,
                kind=cert.kind,
                native_point_distance=float(np.linalg.norm(native.pos - cert.point)),
                **{
                    key: getattr(cert, key)
                    for key in ("surface_error", "point_error", "closure_error", "support_error")
                },
            )
        )
    return certificates, diagnostics


def install_certified_geometry(mj, model, data, certificates):
    """Rebuild and solve native constraints from independent certified inputs.

    Caller must use a fresh native forward at the exact tested q/v. Material,
    mass, scalar rows and contact law remain native. All active contacts must
    be accounted for; no partial ID-specific replacements are accepted.
    """
    indices = [i for i, _ in certificates]
    if sorted(indices) != list(range(data.ncon)):
        raise ValueError("complete one-to-one contact certificates required")
    for i, cert in certificates:
        c = data.contact[i]
        if tuple(c.geom) != cert.pair or c.dim != 3 or c.friction[0] != c.friction[1]:
            raise ValueError("same-pair isotropic condim3 reference required")
        normal = np.asarray(cert.normal)
        # Keep the old tangent direction after orthogonal projection. Isotropic
        # friction makes this basis choice irrelevant to generalized response.
        tangent = c.frame[3:6] - np.dot(c.frame[3:6], normal) * normal
        size = np.linalg.norm(tangent)
        if not np.isfinite(size) or size <= 1e-12:
            raise ValueError("invalid reference tangent")
        tangent /= size
        c.pos[:] = cert.point
        c.dist = cert.gap
        c.frame[:] = np.concatenate((normal, tangent, np.cross(normal, tangent)))
    mj.mj_makeConstraint(model, data)
    # Rebuild per-island constraint/Jacobian copies as mj_fwdPosition does.
    # Keeping the old copies can solve the old contact lever arms while the
    # public efc_J already describes the newly certified points.
    mj.mj_island(model, data)
    mj.mj_projectConstraint(model, data)
    mj.mj_referenceConstraint(model, data)
    mj.mj_fwdConstraint(model, data)
    self_check, _ = evaluate_candidate(mj, model, data, data.qacc.copy())
    if not self_check["balance_passed"]:
        raise ValueError("independent reference fails its own force balance")


def evaluate_candidate(mj, model, data, acceleration):
    """Read-only candidate comparison on a private solved native reference.

    The constraintUpdate below changes efc_force, not the independently solved
    reference qacc. Restore forces before a later reference integration.
    """
    a = np.asarray(acceleration, dtype=float)
    if a.shape != (model.nv,) or not np.isfinite(a).all():
        raise ValueError("finite candidate acceleration required")
    original = data.qacc.copy()
    original_force = data.efc_force.copy()
    original_qfrc = data.qfrc_constraint.copy()
    M = np.zeros((model.nv, model.nv))
    mj.mj_fullM(model, M, data.qM)
    J = data.efc_J.reshape(data.nefc, model.nv).copy()
    aref = data.efc_aref.copy()
    smooth = data.qacc_smooth.copy()
    try:
        mj.mj_constraintUpdate(model, data, J @ a - aref, np.zeros(1), 1)
        f = data.efc_force.copy()
        delta = a - smooth
        scale = 1 + np.abs(M * delta[None]).sum(1) + np.abs(J * f[:, None]).sum(0)
        balance = float(np.max(np.abs(M @ delta - J.T @ f) / scale))
    finally:
        data.efc_force[:] = original_force
        data.qfrc_constraint[:] = original_qfrc
    error = np.abs(a - original)
    return dict(
        balance=balance,
        balance_passed=balance <= 2e-6,
        acceleration_error=float(error.max()),
        acceleration_passed=bool(np.all(error <= 0.002 + 5e-4 * np.abs(original))),
    ), dict(
        M=M,
        J=J,
        aref=aref,
        smooth=smooth,
        force_at_candidate=f,
        independently_solved_acceleration=original,
        candidate_acceleration=a,
    )
