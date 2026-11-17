"""I3v: one real normal-only entry, three unintegrated friction counterfactuals.

Native MuJoCo evaluates M/J/bias at the saved entry; it never steps physics.
The frictionless oracle is an independently assembled linear active-set solve.
The frictional certificate substitutes actual device velocity/forces into the
complete equations. Neither oracle supplies a state to the production scene.
"""

import gzip
import json
from collections.abc import Mapping
from dataclasses import fields, is_dataclass, replace

import numpy as np
import pytest

import experiments.area_pressure.free_body_support as original
from phymeo.backends.mujoco_area_friction import (
    AreaFrictionPairLaw,
    MuJoCoAreaFrictionAdapter,
)
from phymeo.backends.mujoco_heterogeneous import (
    MuJoCoTyxHeterogeneousBatch,
    MuJoCoTyxModelSpec,
    MuJoCoTyxWorldSpec,
)
from phymeo.backends.mujoco_heterogeneous_area_scene import MuJoCoTyxHeterogeneousAreaPressureScene
from phymeo.backends.mujoco_runtime import MuJoCoInitialState
from phymeo.backends.tyx_area_friction import TyxAreaFrictionWorkspace
from phymeo.builders.mujoco_compiler import compile_mujoco_xml
from phymeo.geometry.area_pressure import AreaPressureIdentity
from phymeo.solvers.area_friction import LAW

H = 0.001
SCOPE = "historical-same-entry-counterfactual-no-integration"
VELOCITY = (0.02, -0.01, -0.005, 0.1, 0.2, -0.1)
COM = (0.001, -0.002, 0.001)


def _xml(free):
    mass = 0.3 if free else 0.5
    joint = (
        '<freejoint name="free"/>'
        if free
        else ('<joint name="slide" type="slide" axis="1 0 0" damping="40"/>')
    )
    com = ".001 -.002 .001" if free else "0 0 0"
    gravity = "0 0 -9.81" if free else "0 0 0"
    return f'''<mujoco model="historical-area-friction-{"free" if free else "empty"}">
      <compiler angle="radian" fusestatic="false"/>
      <option timestep=".001" gravity="{gravity}" integrator="Euler"
        solver="Newton" iterations="128" tolerance="1e-10" cone="pyramidal" jacobian="dense"/>
      <default><geom condim="1" friction="0 0 0" contype="0" conaffinity="0"/>
        <joint damping="0" frictionloss="0" armature="0"/></default>
      <worldbody><body name="pressure" pos="0 0 .011">{joint}
        <inertial pos="{com}" mass="{mass}"
          diaginertia="{mass / 6000} {mass / 6000} {mass / 3750}"/>
        <geom name="pressure-boundary" type="box" size=".02 .02 .01"/>
      </body><body name="plate" pos="0 0 -.0025">
        <geom name="plate-boundary" type="box" size=".01 .01 .0025"/>
      </body></worldbody></mujoco>'''


def _declarations():
    specs = tuple(
        MuJoCoTyxModelSpec(
            name,
            compile_mujoco_xml(_xml(free), require_asset=False),
            contact_capacity=16,
            constraint_capacity=64,
            area_pressure_pairs=(("pressure", "plate"),),
        )
        for name, free in (("free-com", True), ("empty-damped", False))
    )
    # The already registered complete pressure volume and finite plate are
    # reused as geometry, never as a solver or numerical oracle.
    asset = original._asset(original.MODEL, original.PARAMETERS)
    assets = {spec.key: (asset,) for spec in specs}
    cp, sp = np.cos(0.02 / 2), np.sin(0.02 / 2)
    cy, sy = np.cos(0.35 / 2), np.sin(0.35 / 2)
    q = (0, 0, (0.01 - 0.00073575) / np.cos(0.02), cy * cp, -sy * sp, cy * sp, sy * cp)
    worlds = (
        MuJoCoTyxWorldSpec(7, specs[0].key, MuJoCoInitialState(q, VELOCITY)),
        MuJoCoTyxWorldSpec(41, specs[1].key, MuJoCoInitialState((0,), (0.02,))),
    )
    return specs, worlds, assets


def _plain(value):
    if isinstance(value, bytes):
        return dict(bytes_hex=value.hex())
    if isinstance(value, np.ndarray):
        return dict(dtype=value.dtype.str, shape=list(value.shape), hex=value.tobytes().hex())
    if isinstance(value, np.generic):
        return _plain(np.asarray(value))
    if is_dataclass(value):
        return {field.name: _plain(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(type(value).__name__)


def _owned(value):
    if isinstance(value, np.ndarray):
        assert not value.flags.writeable
        with pytest.raises(ValueError):
            value.setflags(write=True)
    elif is_dataclass(value):
        for field in fields(value):
            _owned(getattr(value, field.name))
    elif isinstance(value, Mapping):
        for item in value.values():
            _owned(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _owned(item)


def _close(actual, expected, absolute, relative):
    a, b = np.asarray(actual), np.asarray(expected)
    assert a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
    assert np.all(np.abs(a - b) <= absolute + relative * np.abs(b)), (a, b)


def _native(spec, receipt, quadratures):
    import mujoco as mj

    model = spec.compilation.fresh_native_model()
    data = mj.MjData(model)
    data.qpos[:], data.qvel[:], data.time = receipt.input_q, receipt.input_v, receipt.input_time
    control = receipt.stored_control
    data.ctrl[:] = control.ctrl
    if control.qfrc_applied is not None:
        data.qfrc_applied[:] = control.qfrc_applied
    if control.xfrc_applied is not None:
        data.xfrc_applied[:] = control.xfrc_applied
    mj.mj_forward(model, data)
    assert data.ncon == data.nefc == 0
    mass = np.empty((model.nv, model.nv))
    mj.mj_fullM(model, mass, data.qM)
    com_j = np.zeros((model.nbody, 6, model.nv))
    applied = data.qfrc_actuator.copy() + data.qfrc_applied
    for body in range(model.nbody):
        mj.mj_jac(model, data, com_j[body, :3], com_j[body, 3:], data.xipos[body], body)
        applied += com_j[body].T @ data.xfrc_applied[body]
    ja, jb = [], []
    for quad in quadratures:
        a = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, quad.identity.compliant_body)
        b = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, quad.identity.rigid_body)
        for point in quad.points:
            left, right, angular = (np.zeros((3, model.nv)) for _ in range(3))
            mj.mj_jac(model, data, left, angular, point, a)
            mj.mj_jac(model, data, right, angular, point, b)
            ja.append(left)
            jb.append(right)
    return dict(
        mass=mass,
        ja=np.asarray(ja).reshape(-1, 3, model.nv),
        jb=np.asarray(jb).reshape(-1, 3, model.nv),
        com_j=com_j,
        bias=data.qfrc_bias.copy(),
        passive=data.qfrc_passive.copy(),
        applied=applied,
        body_position=data.xpos.copy(),
        body_com=data.xipos.copy(),
        body_rotation=data.xmat.reshape(model.nbody, 3, 3).copy(),
        damping=model.dof_damping.copy(),
        body_ipos=model.body_ipos.copy(),
    )


def _entry_oracle(entry, receipt, native):
    p, dyn = entry.problem, receipt.dynamics
    if len(p.input_velocity) == 6:
        _close(p.mass_matrix[:3, :3], native["mass"][:3, :3], 2e-6, 3e-4)
        _close(p.mass_matrix[3:, 3:], native["mass"][3:, 3:], 2e-9, 3e-4)
        _close(p.mass_matrix[:3, 3:], native["mass"][:3, 3:], 2e-8, 3e-4)
        _close(p.mass_matrix[3:, :3], native["mass"][3:, :3], 2e-8, 3e-4)
        assert np.max(np.abs(native["mass"][:3, 3:])) > 1e-4
    else:
        _close(p.mass_matrix, native["mass"], 2e-6, 3e-4)
    for name in ("bias", "passive", "applied"):
        _close(getattr(dyn, name), native[name], 2e-5, 3e-4)
    _close(dyn.jacobian_com, native["com_j"], 2e-6, 2e-5)
    _close(dyn.body_com, native["body_com"], 2e-6, 2e-5)
    expected_force = (
        native["applied"]
        + native["passive"]
        - native["bias"]
        + (native["damping"] * receipt.input_v)
    )
    _close(p.applied_force, expected_force, 2e-5, 3e-4)
    assert p.h_s == float(np.float32(receipt.h))
    np.testing.assert_array_equal(p.damping, native["damping"].astype(np.float32).astype(float))
    np.testing.assert_array_equal(p.input_velocity, receipt.input_v)
    index = 0
    for quad in entry.quadratures:
        for j, point in enumerate(quad.points):
            sample, catalog = p.samples[index], entry.sample_catalog[index]
            np.testing.assert_array_equal(sample.point_world, point)
            _close(sample.jacobian_a, native["ja"][index], 2e-6, 2e-5)
            _close(sample.jacobian_b, native["jb"][index], 2e-6, 2e-5)
            assert sample.stiffness_n_m == quad.dA[j] * quad.gradient[j]
            assert sample.phi_m == -quad.pressure[j] / quad.gradient[j]
            assert catalog["sample_id"] == quad.sample_ids[j]
            assert catalog["source_id"] == quad.source_ids[j]
            assert catalog["identity"] == quad.identity and catalog["stamp"] == quad.stamp
            assert catalog["dA"] == quad.dA[j] and catalog["pair_index"] == 0
            index += 1
    assert index == len(entry.sample_catalog) == len(p.samples)
    assert entry.law == LAW and entry.scope == SCOPE
    _owned(entry)
    return dict(native_force=expected_force, force_error=p.applied_force - expected_force)


def _equations(problem, velocity):
    """Independent final-normal law and full, generally nonsymmetric Jacobian."""
    v = np.asarray(velocity, dtype=np.float64)
    m, d = np.asarray(problem.mass_matrix, dtype=float), np.asarray(problem.damping, dtype=float)
    h = float(problem.h_s)
    a = m + h * np.diag(d)
    rhs = m @ np.asarray(problem.input_velocity, dtype=float) + h * np.asarray(
        problem.applied_force, dtype=float
    )
    free = np.linalg.solve(a, rhs)
    total, jacobian = np.zeros(len(v)), a.copy()
    normal, tangent, force, moment, normal_rows = [], [], [], [], []
    for sample in problem.samples:
        n = np.asarray(sample.normal_world, dtype=float)
        g = np.asarray(sample.jacobian_a, dtype=float) - np.asarray(sample.jacobian_b, dtype=float)
        projector = np.eye(3) - np.outer(n, n) / (n @ n)
        vt, row = projector @ g @ v, n @ g
        k, mu = float(sample.stiffness_n_m), float(sample.mu)
        fn = max(0.0, -k * (float(sample.phi_m) + h * (row @ v)))
        speed = np.sqrt(vt @ vt + float(sample.slip_speed_m_s) ** 2)
        ft = -mu * fn * vt / speed
        f = fn * n + ft
        dfn = -k * h * row if fn > 0 else np.zeros(len(v))
        derivative = (
            np.outer(n - mu * vt / speed, dfn)
            - mu * fn * (np.eye(3) / speed - np.outer(vt, vt) / speed**3) @ projector @ g
        )
        total += g.T @ f
        jacobian -= h * g.T @ derivative
        normal.append(fn)
        tangent.append(ft)
        force.append(f)
        moment.append(np.cross(np.asarray(sample.point_world, dtype=float), f))
        normal_rows.append(row)
    residual = a @ v - rhs - h * total
    mass_norm = np.sqrt(max(0.0, float(residual @ np.linalg.solve(a, residual))))
    count = len(problem.samples)
    return dict(
        response=a,
        rhs=rhs,
        free=free,
        generalized=total,
        residual=residual,
        residual_norm=mass_norm,
        scale=1 + np.sqrt(max(0.0, float(free @ a @ free))),
        jacobian=jacobian,
        normal=np.array(normal),
        tangent=np.asarray(tangent).reshape(count, 3),
        force=np.asarray(force).reshape(count, 3),
        moment=np.asarray(moment).reshape(count, 3),
        normal_rows=np.asarray(normal_rows).reshape(count, len(v)),
    )


def _certificate(result):
    p = result.packed_problem
    e = _equations(p, result.velocity)
    assert e["residual_norm"] <= 2e-5 * e["scale"]
    assert result.diagnostics["device_residual_norm"] <= result.diagnostics["device_tolerance"]
    _close(result.free_velocity, e["free"], 5e-5, 5e-5)
    _close(result.normal_force, e["normal"], 0.002, 0.002)
    _close(result.tangential_force, e["tangent"], 0.002, 0.002)
    _close(result.force_on_a, e["force"], 0.002, 0.002)
    _close(result.torque_on_a_world_origin, e["moment"], 1e-5, 0.002)
    np.testing.assert_array_equal(result.force_on_b, -result.force_on_a)
    np.testing.assert_array_equal(result.torque_on_b_world_origin, -result.torque_on_a_world_origin)
    np.testing.assert_allclose(result.residual_jacobian, e["jacobian"], rtol=2e-5, atol=2e-7)
    actual_q = np.zeros(len(result.velocity))
    for sample, fn, ft, f in zip(
        p.samples, result.normal_force, result.tangential_force, result.force_on_a, strict=True
    ):
        assert fn >= 0 and np.linalg.norm(ft.astype(float)) <= float(sample.mu) * float(fn) + 1e-5
        g = np.asarray(sample.jacobian_a, dtype=float) - np.asarray(sample.jacobian_b, dtype=float)
        actual_q += g.T @ f.astype(float)
    stored_residual = (
        e["response"] @ result.velocity.astype(float) - e["rhs"] - float(p.h_s) * actual_q
    )
    stored_norm = np.sqrt(
        max(0.0, float(stored_residual @ np.linalg.solve(e["response"], stored_residual)))
    )
    assert stored_norm <= 2e-5 * e["scale"]
    _owned(result)
    e.update(
        stored_force_residual=stored_residual,
        stored_force_residual_norm=stored_norm,
        strict_host_gate=e["residual_norm"] <= 1e-10 * e["scale"],
        jacobian_max_error=float(np.max(np.abs(result.residual_jacobian - e["jacobian"]))),
    )
    return e


def _normal_linear(problem):
    """Solve the all-active normal system, and prove its active-set premise."""
    e = _equations(problem, np.zeros(len(problem.input_velocity)))
    j = e["normal_rows"]
    k = np.array([float(s.stiffness_n_m) for s in problem.samples])
    phi = np.array([float(s.phi_m) for s in problem.samples])
    h = float(problem.h_s)
    v = np.linalg.solve(
        e["response"] + h * h * j.T @ (k[:, None] * j), e["rhs"] - h * j.T @ (k * phi)
    )
    force = -k * (phi + h * j @ v)
    assert np.all(force > 0), "registered first-entry all-active premise did not hold"
    return v, force


def test_pair_law_owned_schema_and_no_native_evaluation(monkeypatch):
    import mujoco as mj

    specs, _, assets = _declarations()
    identity = replace(assets[specs[0].key][0].identity)
    law = AreaFrictionPairLaw(identity, 0.5, 0.001)
    saved = law.identity
    object.__setattr__(identity, "compliant_material", "caller-mutated")
    assert law.identity == saved and law.identity != identity
    for mu, slip in (
        (True, 0.001),
        (-1, 0.001),
        (np.nan, 0.001),
        (0.5, 0),
        (0.5, np.inf),
        (0.5, True),
    ):
        with pytest.raises(ValueError):
            AreaFrictionPairLaw(saved, mu, slip)

    def forbidden(*args, **kwargs):
        raise AssertionError("adapter performed native physics evaluation")

    for name in ("mj_forward", "mj_step", "mj_kinematics", "mj_inverse"):
        monkeypatch.setattr(mj, name, forbidden)
    adapter = MuJoCoAreaFrictionAdapter(specs[0], (law,))
    assert adapter.law == LAW and adapter.scope == SCOPE
    assert adapter.pair_laws[0].identity == saved
    _owned(adapter.binding)
    with pytest.raises(TypeError):
        adapter.binding["model_key"] = "other"
    with pytest.raises((AttributeError, TypeError)):
        law.mu = 0


def test_static_source_and_unrepresented_native_constraints_rejected():
    specs, _, assets = _declarations()
    spec = specs[0]
    law = AreaFrictionPairLaw(assets[spec.key][0].identity, 0.5, 0.001)
    for laws in (
        (),
        [law],
        (law, law),
        (replace(law, identity=AreaPressureIdentity("plate", "pressure", "a", "b")),),
    ):
        with pytest.raises(ValueError):
            MuJoCoAreaFrictionAdapter(spec, laws)
    for name in ("primary_sha256", "archive_sha256", "model_sha256"):
        source = replace(spec.compilation.source, **{name: "0" * 64})
        bad = replace(spec, compilation=replace(spec.compilation, source=source))
        with pytest.raises(ValueError):
            MuJoCoAreaFrictionAdapter(bad, (law,))
    with pytest.raises(ValueError):
        MuJoCoAreaFrictionAdapter(replace(spec, area_pressure_pairs=()), (law,))
    native = (
        _xml(True)
        .replace('contype="0" conaffinity="0"', 'contype="1" conaffinity="1"')
        .replace(
            "</worldbody>",
            '<body name="wall" pos=".1 0 0"><geom type="box" size=".01 .01 .01"/></body></worldbody>',
        )
    )
    variants = (
        native,
        _xml(False).replace('axis="1 0 0"', 'axis="1 0 0" limited="true" range="-1 1"'),
        _xml(False).replace('frictionloss="0"', 'frictionloss=".1"'),
        _xml(True).replace('<freejoint name="free"/>', '<joint name="ball" type="ball"/>'),
        _xml(False).replace(
            "</mujoco>",
            '<actuator><general joint="slide" dyntype="filter" dynprm=".01"/></actuator></mujoco>',
        ),
        _xml(False).replace(
            "</mujoco>", '<equality><joint joint1="slide" active="false"/></equality></mujoco>'
        ),
        _xml(False).replace(
            "</mujoco>",
            '<tendon><fixed name="unloaded"><joint joint="slide" coef="1"/></fixed></tendon></mujoco>',
        ),
    )
    for xml in variants:
        bad = replace(spec, compilation=compile_mujoco_xml(xml, require_asset=False))
        with pytest.raises(ValueError):
            MuJoCoAreaFrictionAdapter(bad, (law,))


def _reject_entries(adapter, receipt, layout, quad):
    """Same one accepted entry supplies every negative; never step again."""
    refused = []

    def reject(label, row=receipt, slot=layout, quads=None):
        with pytest.raises(ValueError):
            adapter.prepare_from_receipt(
                row, layout=slot, quadratures=(quad,) if quads is None else quads
            )
        refused.append(label)

    for name, value in (
        ("world", 123),
        ("nv", layout.nv - 1),
        ("model_key", "wrong"),
        ("primary_sha256", "0" * 64),
        ("archive_sha256", "0" * 64),
        ("model_sha256", "0" * 64),
    ):
        reject("layout-" + name, slot=replace(layout, **{name: value}))
    for name, value in (
        ("contact_phase", "endpoint"),
        ("contact_time", receipt.input_time + 0.1),
        ("input_time", receipt.input_time + 0.1),
        ("tick", receipt.tick + 1),
        ("backend", "native"),
        ("h", receipt.h * 2),
    ):
        reject("receipt-" + name, row=replace(receipt, **{name: value}))
    for name in ("input_q", "input_v"):
        changed = getattr(receipt, name).copy()
        changed[0] += 0.001
        reject("entry-" + name, row=replace(receipt, **{name: changed}))
    reject("dynamics-time", row=replace(receipt, dynamics=replace(receipt.dynamics, time=0.1)))
    reject(
        "dynamics-phase", row=replace(receipt, dynamics=replace(receipt.dynamics, phase="output"))
    )
    nonfinite = receipt.dynamics.bias.copy()
    nonfinite[0] = np.nan
    reject(
        "nonfinite-dynamics",
        row=replace(receipt, dynamics=replace(receipt.dynamics, bias=nonfinite)),
    )
    reject("missing-pair", quads=())
    reject("extra-pair", quads=(quad, quad))
    reject(
        "material",
        quads=(replace(quad, identity=replace(quad.identity, compliant_material="wrong")),),
    )
    reject(
        "pair-order",
        quads=(replace(quad, identity=AreaPressureIdentity("plate", "pressure", "a", "b")),),
    )
    for name, value in (
        ("world_id", "41"),
        ("step_index", 1),
        ("sample_time_s", 0.1),
        ("configuration_id", "wrong"),
    ):
        reject("stamp-" + name, quads=(replace(quad, stamp=replace(quad.stamp, **{name: value})),))
    reject("native-contact", row=replace(receipt, contacts=(object(),)))
    reject(
        "native-row",
        row=replace(
            receipt,
            constraints=(replace(receipt.constraints[0], constraint_type=0),)
            + receipt.constraints[1:],
        ),
    )
    # A genuine, well-formed 43-triangle quadrature has 129 samples. The old
    # row capacity and raw receipt differ too; the explicit sample cap must win.
    many = replace(
        quad,
        sample_ids=tuple((f"cap-{i}", j) for i in range(43) for j in range(3)),
        source_ids=tuple(quad.source_ids[:3]) * 43,
        points=np.tile(quad.points[:3], (43, 1)),
        normals=np.tile(quad.normals[:3], (43, 1)),
        dA=np.tile(quad.dA[:3], 43),
        pressure=np.tile(quad.pressure[:3], 43),
        gradient=np.tile(quad.gradient[:3], 43),
        total_area_m2=float(np.tile(quad.dA[:3], 43).sum()),
    )
    with pytest.raises(ValueError, match="128"):
        adapter.prepare_from_receipt(receipt, layout=layout, quadratures=(many,))
    refused.append("129-full-samples")
    return refused


@pytest.mark.cuda
def test_actual_entry_native_oracle_and_three_counterfactuals(tmp_path, monkeypatch):
    evidence = dict(schema="mujoco-area-friction-entry-v1", qualification=SCOPE, raw={})
    try:
        specs, worlds, assets = _declarations()
        evidence.update(
            xml={"A": _xml(True), "B": _xml(False)},
            worlds=worlds,
            assets=assets,
            sources={spec.key: spec.compilation.source for spec in specs},
        )
        identity = assets[specs[0].key][0].identity
        adapters = (
            MuJoCoAreaFrictionAdapter(specs[0], (AreaFrictionPairLaw(identity, 0.5, 0.001),)),
            MuJoCoAreaFrictionAdapter(specs[0], (AreaFrictionPairLaw(identity, 0.0, 0.001),)),
            MuJoCoAreaFrictionAdapter(specs[1], (AreaFrictionPairLaw(identity, 0.5, 0.001),)),
        )
        with MuJoCoTyxHeterogeneousBatch(specs, worlds, mode="cuda") as batch:
            scene = MuJoCoTyxHeterogeneousAreaPressureScene(batch, assets)
            evidence["initial"] = batch.integration_snapshot()
            scene.step(H, expected_revision=batch.checkpoint_revision)  # Sole physical call.
            evidence.update(
                receipt=batch.last_step,
                geometry=scene.last_geometry,
                after_normal=batch.integration_snapshot(),
                layout=batch.layout,
            )
            receipt_a, receipt_b = batch.last_step.receipts[7], batch.last_step.receipts[41]
            quads_a = tuple(
                row["source_quadrature"] for row in receipt_a.raw_contact_fields["area_pressure"]
            )
            quads_b = tuple(
                row["source_quadrature"] for row in receipt_b.raw_contact_fields["area_pressure"]
            )
            assert len(quads_a[0].sample_ids) > 0 and quads_b[0].sample_ids == ()
            before = _plain(batch.integration_snapshot())
            revision, last = batch.checkpoint_revision, batch.last_step
            entries = tuple(
                adapter.prepare_from_receipt(row, layout=batch.layout[key], quadratures=quads)
                for adapter, row, key, quads in (
                    (adapters[0], receipt_a, 7, quads_a),
                    (adapters[1], receipt_a, 7, quads_a),
                    (adapters[2], receipt_b, 41, quads_b),
                )
            )
            evidence["prepared"] = entries
            native_a, native_b = (
                _native(specs[0], receipt_a, quads_a),
                _native(specs[1], receipt_b, quads_b),
            )
            evidence["native"] = (native_a, native_b)
            body = adapters[0].binding["body_names"].index("pressure")
            np.testing.assert_array_equal(native_a["body_ipos"][body], COM)
            np.testing.assert_array_equal(
                receipt_a.input_v, np.asarray(VELOCITY, dtype=np.float32).astype(float)
            )
            evidence["entry_oracles"] = tuple(
                _entry_oracle(e, r, n)
                for e, r, n in (
                    (entries[0], receipt_a, native_a),
                    (entries[1], receipt_a, native_a),
                    (entries[2], receipt_b, native_b),
                )
            )
            for entry in entries[:2]:
                assert not np.shares_memory(
                    entry.problem.mass_matrix, receipt_a.dynamics.mass_matrix
                )
                assert not np.shares_memory(entry.quadratures[0].points, quads_a[0].points)
            problems = tuple(entry.problem for entry in entries)
            evidence["problem_dimensions"] = tuple(
                dict(nv=len(p.input_velocity), samples=len(p.samples)) for p in problems
            )
            assert tuple(len(p.input_velocity) for p in problems) == (6, 6, 1)
            with TyxAreaFrictionWorkspace(capacity=3, mode="cuda") as solver:
                get = solver._download
                prepare = solver._prepare

                def capture_inputs(*args, **kwargs):
                    prepared, arrays = prepare(*args, **kwargs)
                    evidence["actual_uploads"] = {
                        name: np.array(a, copy=True) for name, a in arrays.items()
                    }
                    return prepared, arrays

                def capture(name):
                    array = get(name)
                    evidence["raw"][name] = np.array(array, copy=True)
                    return array

                monkeypatch.setattr(solver, "_download", capture)
                monkeypatch.setattr(solver, "_prepare", capture_inputs)
                evidence["stats_before"] = solver.stats
                try:
                    results = solver.solve_many(problems, max_iterations=24, max_backtracks=12)
                finally:
                    evidence["stats_after"] = solver.stats
                evidence["results"] = results
                assert solver.stats["launches"] == evidence["stats_before"]["launches"] + 1
                evidence["certificates"] = tuple(_certificate(result) for result in results)
                normal_v, normal_force = _normal_linear(results[1].packed_problem)
                evidence["normal_linear_oracle"] = dict(velocity=normal_v, force=normal_force)
                _close(results[1].velocity, normal_v, 5e-5, 5e-5)
                _close(results[1].normal_force, normal_force, 0.002, 0.002)
                # This is engineering parity, not exact packed-point equivalence.
                _close(results[1].velocity, evidence["after_normal"][7].v, 5e-5, 5e-5)
                b = results[2].packed_problem
                m, v0, d, h = (
                    float(b.mass_matrix[0, 0]),
                    float(b.input_velocity[0]),
                    float(b.damping[0]),
                    float(b.h_s),
                )
                correct, doubled = m * v0 / (m + h * d), (m - h * d) * v0 / (m + h * d)
                evidence["damping_oracle"] = dict(
                    correct=correct,
                    repeated_damping=doubled,
                    cancelled_force=entries[2].problem.applied_force,
                    stored_passive=receipt_b.dynamics.passive,
                )
                assert abs(correct - doubled) > 1e-3
                _close(results[2].velocity, np.array([correct]), 5e-5, 5e-5)
                _close(evidence["after_normal"][41].v, np.array([correct]), 5e-5, 5e-5)
                assert abs(float(results[2].velocity[0]) - doubled) > 1e-3
                assert np.max(np.abs(entries[2].problem.applied_force)) <= 2e-5
                assert results[2].normal_force.shape == (0,) and results[2].force_on_a.shape == (
                    0,
                    3,
                )
            evidence["negative_cases"] = _reject_entries(
                adapters[0], receipt_a, batch.layout[7], quads_a[0]
            )
            assert batch.checkpoint_revision == revision and batch.last_step is last
            assert _plain(batch.integration_snapshot()) == before
            evidence.update(
                snapshot_after_counterfactual=batch.integration_snapshot(),
                unchanged_revision=revision,
                status="PASS",
            )
    except BaseException as error:
        evidence.update(status="FAIL", error_type=type(error).__name__, error=str(error))
        if hasattr(error, "diagnostics"):
            evidence["failure_diagnostics"] = error.diagnostics
        raise
    finally:
        with gzip.open(tmp_path / "entry-counterfactual.json.gz", "xt", encoding="utf-8") as stream:
            json.dump(_plain(evidence), stream, allow_nan=False)
