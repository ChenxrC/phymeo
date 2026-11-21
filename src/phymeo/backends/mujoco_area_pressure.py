"""Opt-in supplied-surface area contact for a shared TYX dynamics step.

Geometry validity is the producer's responsibility. A stamp binds consumption to
one accepted runtime state; it is not a collision or pressure-field certificate.
"""

from dataclasses import dataclass
from hashlib import sha256
from types import MappingProxyType

import numpy as np

from ..geometry.area_pressure import AreaPressureSurface, quadrature_area_pressure
from .mujoco_runtime import _frozen


@dataclass(frozen=True)
class _AreaPlan:
    base: object
    area_pair_names: tuple
    area_body_ids: np.ndarray
    area_friction_laws: tuple = ()
    area_friction_parameters: np.ndarray | None = None
    damped_contact_dofs: int = 0
    resident_native_contacts: bool = False

    @property
    def scalar_damped_contacts(self):
        return self.damped_contact_dofs == 1

    @property
    def coupled_damped_contacts(self):
        return self.damped_contact_dofs == 2

    @property
    def two_native_supports(self):
        return self.coupled_damped_contacts and self.npair == 2

    @property
    def coupled_contact_variant(self):
        """Version suffix shared by the coupled solver, checkpoint and capability names."""
        if self.two_native_supports:
            return "two-supports-v1"
        return ("v1", "limit-v1", "two-limits-v1")[self.nlimit]

    def __getattr__(self, name):
        return getattr(self.base, name)


@dataclass(frozen=True)
class AreaCollisionOwnership:
    """Geometry candidates from the compiled native model, before any contact detection."""

    body_ids: tuple
    replaced_geometry_pairs: tuple
    native_geometry_pairs: tuple


def area_collision_ownership(mj, model, pairs):
    """Resolve candidates per declared body pair, preserving native order in both routes."""
    from .mujoco_tyx._constraint_model import constraint_pairs

    if not isinstance(pairs, tuple) or not 1 <= len(pairs) <= 16:
        raise ValueError("area_pressure_pairs requires 1..16 ordered body pairs")
    ids, owners = [], {}
    for pair in pairs:
        if (
            not isinstance(pair, tuple)
            or len(pair) != 2
            or any(not isinstance(name, str) or not name for name in pair)
        ):
            raise ValueError("area pairs require two body names")
        bodies = tuple(int(mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, name)) for name in pair)
        if min(bodies) < 0 or bodies[0] == bodies[1]:
            raise ValueError("area pair bodies must exist and differ")
        key = tuple(sorted(bodies))
        if key in owners or model.body_weldid[bodies[0]] == model.body_weldid[bodies[1]]:
            raise ValueError("duplicate or welded area pair")
        owners[key] = len(ids)
        ids.append(bodies)
    # Ownership is exact body identity, not a weld component: two distinct
    # static tables may share body_weldid=0 but need independent contact laws.
    replaced = [[] for _ in ids]
    remaining = []
    for pair in constraint_pairs(model):
        owner = owners.get(tuple(sorted(map(int, model.geom_bodyid[list(pair)]))))
        if owner is None:
            remaining.append(pair)
        else:
            replaced[owner].append(pair)
    return AreaCollisionOwnership(tuple(ids), tuple(map(tuple, replaced)), tuple(remaining))


def compile_area_native_rows(mj, model, pairs):
    """Compile the native half of a replacement-contact calculation.

    The supplied model is left intact. Only native collision candidates owned by
    a declared body pair are removed. Joint rows and other contacts still come
    from the original dynamics. The caller owns the replacement load.
    """
    from .mujoco_tyx._constraint_model import compile_constraint_model

    ownership = area_collision_ownership(mj, model, pairs)
    return compile_constraint_model(mj, model, ownership.native_geometry_pairs)


def compile_area_plan(mj, model, pairs, friction_laws=()):
    from .area_friction_law import capture_pair_laws
    from .mujoco_tyx._constraint_model import compile_constraint_model

    friction_laws = capture_pair_laws(pairs, friction_laws)
    ownership = area_collision_ownership(mj, model, pairs)
    ids, remaining = ownership.body_ids, ownership.native_geometry_pairs
    if model.na or np.any(model.jnt_type == mj.mjtJoint.mjJNT_BALL):
        raise ValueError("first area route supports stateless hinge/slide or free-root trees")
    limit_count = np.count_nonzero(model.jnt_limited)
    joint = mj.mjtJoint
    resident_native_contacts = bool(
        remaining
        and friction_laws
        and 6 <= model.nv <= 8
        and np.count_nonzero(model.jnt_type == joint.mjJNT_FREE) == 1
        and np.isin(model.jnt_type, (joint.mjJNT_FREE, joint.mjJNT_HINGE, joint.mjJNT_SLIDE)).all()
        and not (model.neq or model.ntendon or np.any(model.dof_frictionloss))
        and not np.any(model.jnt_limited[model.jnt_type == joint.mjJNT_FREE])
    )
    damped_contact_dofs = 0
    if (
        remaining
        and model.nv == model.njnt
        and (model.nv == 1 or (model.nv == 2 and friction_laws))
        and np.isin(model.jnt_type, (mj.mjtJoint.mjJNT_HINGE, mj.mjtJoint.mjJNT_SLIDE)).all()
        and np.isfinite(model.dof_damping).all()
        and np.all(model.dof_damping >= 0)
        and np.any(model.dof_damping > 0)
        and not (model.neq or model.ntendon or np.any(model.dof_frictionloss))
        and (limit_count == 0 or model.nv == 2)
    ):
        damped_contact_dofs = model.nv
    if (
        not (damped_contact_dofs or resident_native_contacts)
        and np.any(model.dof_damping)
        and (remaining or np.any(model.jnt_limited) or np.any(model.dof_frictionloss))
    ):
        raise ValueError("damped area route cannot mix native constraints in this version")
    if friction_laws:
        from .mujoco_tyx._area_friction_layout import friction_width

        friction_width(model.nv)
        if model.nbody > 128 or model.ngeom > 128:
            raise ValueError("live area-friction capacity: 128 bodies/geoms")
        if (
            model.neq
            or model.ntendon
            or (remaining and not (damped_contact_dofs or resident_native_contacts))
        ):
            raise ValueError(
                "live area friction requires area-only constraints or supported scalar joint rows"
            )
        dry_count = np.count_nonzero(model.dof_frictionloss)
        if (
            (dry_count or limit_count)
            and not resident_native_contacts
            and not (
                1 <= model.nv == model.njnt <= 2
                and np.isin(
                    model.jnt_type, (mj.mjtJoint.mjJNT_SLIDE, mj.mjtJoint.mjJNT_HINGE)
                ).all()
                and dry_count <= 1
                and limit_count <= (2 if damped_contact_dofs == 2 else 1)
            )
        ):
            raise ValueError(
                "mixed area friction requires a supported scalar joint limit and dry-friction layout"
            )
        if not np.isfinite(model.dof_damping).all() or np.any(model.dof_damping < 0):
            raise ValueError("live area friction requires nonnegative finite damping")
    base = compile_constraint_model(mj, model, remaining)
    if resident_native_contacts:
        # Native geometry is independently checked against the same entry at publication.
        if base.kind != 3 or np.any(base.pair_condim[: base.npair] != 3):
            raise ValueError(
                "resident mixed contacts require a free root and native condim-3 pairs"
            )
        if not np.isin(model.geom_type[np.array(remaining)], (0, 2, 3, 6)).all():
            raise ValueError(
                "resident mixed contacts require plane, sphere, capsule or box geometry"
            )
    if friction_laws and damped_contact_dofs:
        max_pairs = 2 if damped_contact_dofs == 2 and not limit_count else 1
        plane_sphere = (mj.mjtGeom.mjGEOM_PLANE, mj.mjtGeom.mjGEOM_SPHERE)
        if (
            len(remaining) > max_pairs
            or base.nslot != len(remaining)
            or any(tuple(model.geom_type[list(pair)]) != plane_sphere for pair in remaining)
            or np.any(base.pair_condim[: base.npair] != 3)
        ):
            raise ValueError(
                f"live damped contact requires at most {max_pairs} native condim-3 plane/sphere supports"
            )
    if friction_laws and base.nlimit:
        lo, hi, margin = base.limit_law[: base.nlimit, :3].astype(np.float64).T
        if np.any(lo >= hi) or np.any(hi - lo < 2.0 * margin):
            raise ValueError(
                "mixed area friction requires distinct limits with nonoverlapping margins"
            )
    body_ids = np.frombuffer(np.asarray(ids, dtype=np.int32).tobytes(), dtype=np.int32).reshape(
        -1, 2
    )
    parameters = None
    if friction_laws:
        values = np.asarray(
            [(law.mu, law.slip_speed_m_s) for law in friction_laws], dtype=np.float64
        )
        with np.errstate(over="raise", invalid="raise"):
            parameters = values.astype(np.float32)
        if not np.isfinite(parameters).all() or np.any((values > 0) & (parameters <= 0)):
            raise ValueError("area-friction parameters are not positive finite FP32")
        parameters = np.frombuffer(parameters.tobytes(), dtype=np.float32).reshape(-1, 2)
    return _AreaPlan(
        base,
        tuple(pairs),
        body_ids,
        friction_laws,
        parameters,
        damped_contact_dofs,
        resident_native_contacts,
    )


def configuration_id(runtime, key):
    runtime._check_open()
    if not hasattr(runtime._plan, "area_pair_names"):
        raise ValueError("runtime has no area-pressure pair declaration")
    if key not in runtime._world_keys:
        raise KeyError(key)
    state = runtime._data[key]
    digest = sha256()
    digest.update(str(id(runtime)).encode("ascii"))
    digest.update(
        repr((key, runtime._checkpoint_revision, runtime._ticks[key], state.time)).encode()
    )
    digest.update(state.qpos.tobytes())
    digest.update(state.qvel.tobytes())
    return digest.hexdigest()


def pack(runtime, mapping, expected_revision):
    if type(expected_revision) is not int or expected_revision != runtime._checkpoint_revision:
        raise ValueError("stale area-pressure revision")
    if not isinstance(mapping, dict) or set(mapping) != set(runtime._world_keys):
        raise ValueError("area surfaces must cover exactly all runtime worlds")
    from .mujoco_tyx._model import fp32

    plan, cap = runtime._plan, runtime._constraint_capacity
    if not hasattr(plan, "area_pair_names"):
        raise ValueError("runtime has no area-pressure declaration")
    storage = max(1, cap)
    data = np.zeros((len(runtime._world_keys), storage, 8), dtype=np.float32)
    measures = np.zeros((len(runtime._world_keys), storage, 2), dtype=np.int32)
    pair_index = np.zeros((len(runtime._world_keys), storage), dtype=np.int32)
    counts = np.zeros(len(runtime._world_keys), dtype=np.int32)
    metadata = []
    for wi, key in enumerate(runtime._world_keys):
        native_rows = plan.nfriction if plan.area_friction_laws else 0
        if plan.area_friction_laws and plan.nlimit:
            from .mujoco_area_limit import expected_limit_sides, expected_resident_limit_rows

            limits = (
                expected_resident_limit_rows
                if plan.resident_native_contacts
                else expected_limit_sides
            )
            native_rows += len(limits(plan, runtime._data[key].qpos))
        surfaces = mapping[key]
        if not isinstance(surfaces, tuple) or len(surfaces) != len(plan.area_pair_names):
            raise ValueError("every declared area pair requires a surface, including empty")
        row = []
        cursor = 0
        for pi, (names, surface) in enumerate(zip(plan.area_pair_names, surfaces, strict=True)):
            if not isinstance(surface, AreaPressureSurface):
                raise ValueError("expected AreaPressureSurface")
            if (surface.identity.compliant_body, surface.identity.rigid_body) != names:
                raise ValueError("area pair order/identity mismatch")
            stamp = surface.stamp
            if (
                stamp.world_id != str(key)
                or stamp.step_index != runtime._ticks[key]
                or stamp.sample_time_s != runtime._data[key].time
                or stamp.configuration_id != configuration_id(runtime, key)
            ):
                raise ValueError("stale area-pressure state binding")
            quad = quadrature_area_pressure(surface)
            n = len(quad.dA)
            if (
                runtime._area_friction_laws
                and surface.identity != runtime._area_friction_laws[pi].identity
            ):
                raise ValueError("area-friction body/material identity differs")
            if cursor + n + native_rows > cap:
                raise ValueError("area-pressure samples exceed constraint capacity")
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                k = quad.dA * quad.gradient
                phi = -quad.pressure / quad.gradient
                values = np.column_stack((quad.points, quad.normals, k, phi))
                converted = fp32(values, "area-pressure samples")
            if not np.isfinite(converted).all() or np.any(converted[:, 6] <= 0):
                raise ValueError("area-pressure coefficients not representable in FP32")
            data[wi, cursor : cursor + n] = converted
            pair_index[wi, cursor : cursor + n] = pi
            measures[wi, cursor : cursor + n] = (
                np.asarray(quad.dA, dtype="<f8").view("<i4").reshape(n, 2)
            )
            row.append((quad, cursor, n))
            cursor += n
        counts[wi] = cursor
        metadata.append(tuple(row))
    return data, pair_index, counts, tuple(metadata), measures


def publish(runtime, result, index, fields, native_rows):
    """Validate device point Jacobians and forces before the common bank commit."""
    packet = runtime._area_packet
    quad_meta = packet[3][index]
    count = int(packet[2][index])
    inputs = packet[0][index, :count].astype(np.float64)
    start = int(fields["counts"][2]) - count
    if start != native_rows:
        raise RuntimeError("area row offset does not follow exact native rows")
    jac = fields["efc_J"][start : start + count, : runtime._plan.nv].astype(np.float64)
    rows = fields["efc_data"][start : start + count].astype(np.float64)
    masses = result["mass_matrix"][index, : runtime._plan.nv, : runtime._plan.nv]
    acc = result["integration_acc"][index, : runtime._plan.nv].astype(np.float64)
    vin = result["input_v"][index, : runtime._plan.nv].astype(np.float64)
    h = float(np.float32(runtime._timestep))
    vout = result["v"][index, : runtime._plan.nv].astype(np.float64)
    if not np.allclose(vout, vin + h * acc, rtol=2e-6, atol=2e-7):
        raise RuntimeError("area response did not produce the published velocity")
    answer = []
    for pi, (quad, offset, n) in enumerate(quad_meta):
        sl = slice(offset, offset + n)
        points, normals = inputs[sl, :3], inputs[sl, 3:6]
        k, phi = inputs[sl, 6], inputs[sl, 7]
        ba, bb = map(int, runtime._plan.area_body_ids[pi])
        body_j = result["jacobian_com"][index, :, :, : runtime._plan.nv].astype(np.float64)
        com = result["body_com"][index].astype(np.float64)
        expected_j = []
        for point, normal in zip(points, normals, strict=True):
            ja = body_j[ba, :3] + np.cross(body_j[ba, 3:].T, point - com[ba]).T
            jb = body_j[bb, :3] + np.cross(body_j[bb, 3:].T, point - com[bb]).T
            expected_j.append(normal @ (ja - jb))
        expected_j = np.asarray(expected_j).reshape(n, runtime._plan.nv)
        if not np.allclose(jac[sl], expected_j, rtol=2e-5, atol=2e-6):
            raise RuntimeError("area Jacobian differs from actual body point motion")
        force = rows[sl, 9]
        predicted = np.maximum(0, -k * (phi + h * (jac[sl] @ vout)))
        if np.any(force < 0) or not np.allclose(force, predicted, rtol=5e-4, atol=5e-4):
            raise RuntimeError("area force and actual integration motion disagree")
        force_a = normals * force[:, None]
        moment_a = np.cross(points, force_a)
        answer.append(
            MappingProxyType(
                dict(
                    source_quadrature=quad,
                    identity=quad.identity,
                    stamp=quad.stamp,
                    sample_ids=quad.sample_ids,
                    source_ids=quad.source_ids,
                    body_ids=(ba, bb),
                    dA=quad.dA,
                    points=_frozen(points),
                    normals=_frozen(normals),
                    stiffness=_frozen(k),
                    surrogate_gap=_frozen(phi),
                    normal_jacobian=_frozen(jac[sl]),
                    normal_force=_frozen(force),
                    force_on_a=_frozen(force_a),
                    force_on_b=_frozen(-force_a),
                    torque_on_a_world_origin=_frozen(moment_a),
                    torque_on_b_world_origin=_frozen(-moment_a),
                )
            )
        )
    if masses.shape != (runtime._plan.nv, runtime._plan.nv):
        raise RuntimeError("area dynamics mass layout mismatch")
    return tuple(answer)
