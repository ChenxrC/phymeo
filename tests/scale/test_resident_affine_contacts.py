"""C7 frozen affine resident-contact cases; authoring only, UNEXECUTED.

The existing C5 reference is an independent full-component/KKT oracle. Product
M/J/loads never define expected dynamics. All execution belongs to the parent.
"""

import copy
import gzip
import importlib.util
import json
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from functools import lru_cache
from itertools import product
from pathlib import Path

import numpy as np
import pytest

import phymeo as ph
from phymeo.geometry.contact_patch_adapter import contact_patch_sample
from phymeo.solvers.contact_patch_loads import sample_patch_loads

H = 1 / 1024
ATOL = 5e-6
RTOL = 5e-5
WINDOW_ATOL = 2e-4
WORLD = 503


@lru_cache(maxsize=1)
def _ref():
    path = (
        Path(__file__).resolve().parents[1] / "reference_newton/robot/resident_affine_reference.py"
    )
    spec = importlib.util.spec_from_file_location("c7_frozen_independent_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _json(value):
    if is_dataclass(value):
        return {field.name: _json(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _json(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return _json(value.tolist())
    if isinstance(value, np.generic):
        return _json(value.item())
    if isinstance(value, (tuple, list)):
        return [_json(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return {"nonfinite": repr(value)}
    return value


def _save(directory, name, payload):
    with gzip.open(directory / (name + ".json.gz"), "wt", encoding="utf-8") as stream:
        json.dump(_json(payload), stream, allow_nan=False, separators=(",", ":"))


def _close(actual, expected, *, atol=ATOL, rtol=RTOL):
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)


def _box(low, high):
    return np.array(list(product(*zip(low, high, strict=True))), dtype=float)


def _case(name):
    data = copy.deepcopy(_ref().case(name))
    # Replace the original imposed 3 N by actual finite contact. Never add both.
    if name in ("A01F", "A05G"):
        data["force"][:] = 0
    return data


def _asset(directory, data, *, reorder=False):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "mechanism.urdf"
    path.write_text(_ref().urdf(data, reorder=reorder), encoding="utf-8")
    return ph.load_mimic_urdf(path, gravity=data["model"]["gravity"])


def _orders(asset, data):
    names = tuple(data["model"]["joint_names"][i] for i in _ref().coordinates(data["model"]))
    independent = tuple(names[i] for i in data["independent"])
    return [names.index(name) for name in asset.tree.joint_names], [
        independent.index(name) for name in asset.joint_names
    ]


def _world(asset, data, *, kp=None, kd=None, cap=None, delay=0, iterations=512):
    _, order = _orders(asset, data)
    k = len(order)

    def vector(value, default):
        return np.full(k, default) if value is None else np.asarray(value, dtype=float)[order]

    return ph.AffineRobotTree(
        asset,
        ph.RobotState(np.asarray(data["s"], float)[order], np.asarray(data["u"], float)[order]),
        kp=vector(kp, 0),
        kd=vector(kd, 0),
        effort_limit=vector(cap, 16),
        delay_steps=delay,
        control_max_iterations=iterations,
    )


def _control(world, asset, data, *, target=None, rate=None, ff=None):
    full_order, order = _orders(asset, data)
    active = list(_ref().coordinates(data["model"]))

    def vector(value, default):
        return np.asarray(default if value is None else value, dtype=float)[order]

    return world.inputs(
        vector(target, data["s"]),
        vector(rate, np.zeros(len(order))),
        feedforward=vector(ff, data["motor"]),
        loads=ph.RobotInputs(
            np.asarray(data["effort"], float)[full_order],
            np.asarray(data["force"], float)[active][full_order],
            np.asarray(data["torque"], float)[active][full_order],
        ),
    )


def _a01_geometry(batch, asset, *, world=WORLD):
    group = asset.tree.sites["link1"].link_index
    moving, fixed = ph.CollisionKey(world, 100 + group, 11), ph.CollisionKey(world, 900, 1)
    first = ph.CollisionShape(moving, "sphere", (71,), radius=1 / 8)
    second = ph.CollisionShape(fixed, "hull", tuple(range(201, 209)))
    local = np.zeros((1, 3))
    support = _box((-13 / 64, 1 / 4, -1 / 4), (3 / 64, 3 / 4, 1 / 4))
    checkpoint = batch.snapshot([world])[world]
    bindings = (
        ph.ResidentAffineTreeBinding(moving, world, group, local, topology=checkpoint.topology),
        ph.ResidentStaticBinding(fixed, ph.CollisionMotion(support)),
    )
    return ph.CollisionModel((first, second)), bindings, (moving, fixed), local, support


@pytest.mark.tyx
def test_c701_public_affine_contact_replaces_follower_three_newtons(tyx_mode, tmp_path):
    data = _case("A01F")
    asset = _asset(tmp_path, data)
    assert asset.tree.model.dof_count == 2 and len(asset.joint_names) == 1
    np.testing.assert_array_equal(asset.matrix, ((1,), (2,)))
    np.testing.assert_array_equal(asset.offset_vector, (0, 1 / 8))
    np.testing.assert_array_equal(
        asset.full_state(ph.RobotState(np.zeros(1), np.zeros(1))).q, (0, 1 / 8)
    )
    np.testing.assert_array_equal(data["force"], 0)
    world = _world(asset, data)
    record = dict(input=data, h_s=H, backend=tyx_mode)
    with ph.TyxWorldBatch([(WORLD, world)], tyx_mode, cache_root=tmp_path) as batch:
        batch.set_controls({WORLD: _control(world, asset, data)})
        before = batch.snapshot()
        model, bindings, pair, local, support = _a01_geometry(batch, asset)
        geometry = batch.tree_kinematics([WORLD])[WORLD]
        group = bindings[0].group_index
        _close(geometry.com_position_m[group], (1 / 8, 1 / 2, 0))
        points = geometry.link_position_m[group] + local @ geometry.link_rotation[group].T
        law = ph.RobotContactMaterial(64, 0, 0)
        patch = contact_patch_sample(model.shapes[0], model.shapes[1], points, support)
        assert len(patch.sites) == 1 and patch.sites[0].weight == 1
        assert patch.sites[0].witness.gap_m == pytest.approx(-3 / 64, abs=1e-12)
        sampled = sample_patch_loads(patch, law, np.zeros((1, 3)), np.zeros((1, 3)))
        _close(sampled["first_force"], ((3, 0, 0),), atol=1e-12, rtol=0)
        with batch.prepare_collision(model, bindings) as collision:
            result = collision.query()
            assert result.candidates == (pair,)
            assert batch.snapshot() == before
        record.update(
            before=before,
            geometry=geometry,
            collision_model=model,
            bindings=bindings,
            original_patch=patch,
            original_force=sampled,
        )
        try:
            # The accepted asset/full FK/binding and actual old 3 N law precede
            # the real missing capability: ordinary-only prepare_contacts.
            with batch.prepare_contacts(model, bindings, {pair: law}) as plan:
                assert batch.step_contacts(plan, H) == {WORLD: True}
                after = batch.snapshot()[WORLD]
                diagnostic = batch.tree_diagnostics([WORLD])[WORLD]
                receipt = plan.last_accepted[WORLD]
                record.update(after=after, diagnostic=diagnostic, receipt=receipt)
                _close(after.velocity, (H,))
                _close(after.position, (H * H,))
                _close(after.full_velocity, (H, 2 * H))
                _close(receipt.input_external, (0, 0))
                _close(receipt.contact_generalized, (0, 3))
                _close(diagnostic.reduced_mass, ((6,),))
                _close(diagnostic.reduced_external, (6,))
                _close(diagnostic.acceleration, (1,))
                _close(diagnostic.full_acceleration, (1, 2))
                _close(diagnostic.row_multipliers, (-1,))
                _close(diagnostic.constraint_effort_full, (2, -1))
                assert after.control == before[WORLD].control
        except Exception as error:
            record["error"] = dict(type=type(error).__name__, message=str(error))
            raise
        finally:
            record["retained"] = batch.snapshot()
            record["stats"] = batch.stats()
            _save(tmp_path, "c701-public-contact", record)


_INITIAL_CENTERS = {
    "A01F": (1 / 8, 1 / 2, 0),
    "A02": (-1 / 16, -1 / 2, 0),
    "A03": (1 / 8, -1 / 2, 0),
    "A03_flat": (1 / 8, -1 / 2, 0),
    "A04": (1, 1 / 2, 0),
    "A05": (1 / 8, 1, 0),
    "A05G": (1 / 8, 1, 0),
}
EXACT = {
    "A01F": ((1,), (-1,), (2, -1)),
    "A02": ((11 / 14,), (11 / 7, -32 / 7), (-38 / 7, 11 / 7, -32 / 7)),
    "A03": ((4 / 7,), (41 / 14, -25 / 7), (-41 / 7, 8 / 7, -25 / 7)),
    "A03_flat": ((4 / 7,), (8 / 7, -25 / 7), (-41 / 7, 8 / 7, -25 / 7)),
    "A04": ((27 / 20,), (27 / 40,), (27 / 20, 27 / 40)),
    "A05": ((-512 / 741, 935 / 741), (53 / 247,), (0, -106 / 247, 53 / 247)),
    "A05G": ((-1360 / 741, 469 / 741), (25 / 247,), (0, -50 / 247, 25 / 247)),
}


def _placed_case(name, placement="base"):
    data = _ref().transformed(_case(name), placement)
    data["placement"] = placement
    return data


def _placement(data):
    if data.get("placement", "base") == "base":
        return np.eye(3), np.zeros(3)
    assert data["placement"] == "cycle"
    return np.array(((0, 0, 1), (1, 0, 0), (0, 1, 0)), dtype=float), np.array((0.5, -0.25, 0.75))


def _contact_geometry(batch, asset, data, *, world=WORLD, kind="sphere"):
    body = len(data["model"]["mass"]) - 1
    source_name = data["model"]["names"][body]
    site = asset.tree.sites[source_name]
    group = site.link_index
    center_local = np.asarray(data["model"]["com"][body])
    if kind == "sphere":
        authored, ids, radius = center_local[None, :], (71,), 1 / 8
    elif kind == "capsule":
        authored, ids, radius = (
            center_local + np.array(((0, 0, -1 / 16), (0, 0, 1 / 16))),
            (17, 43),
            1 / 8,
        )
    else:
        assert kind == "hull"
        authored = center_local + _box((-1 / 8, -1 / 16, -1 / 16), (1 / 8, 1 / 16, 1 / 16))
        ids, radius = tuple(range(101, 109)), 0
    local = (
        np.asarray(site.local_position_m)
        + authored @ _ref().quaternion_matrix(site.local_orientation_xyzw).T
    )
    first, second = ph.CollisionKey(world, 100 + group, 11), ph.CollisionKey(world, 900, 1)
    a = ph.CollisionShape(first, kind, ids, radius, revision=7)
    b = ph.CollisionShape(second, "hull", tuple(range(201, 209)), revision=3)
    initial = np.asarray(_INITIAL_CENTERS[data["name"]])
    depth = 1 / 32 if data["name"] == "A04" else 3 / 64
    face = initial[0] - 1 / 8 + depth
    low = (face - 1 / 4, initial[1] - 1 / 4, -1 / 4)
    high = (face, initial[1] + 1 / 4, 1 / 4)
    Q, shift = _placement(data)
    support = _box(low, high) @ Q.T + shift
    bindings = (
        ph.ResidentAffineTreeBinding(
            first, world, group, local, topology=batch.snapshot([world])[world].topology
        ),
        ph.ResidentStaticBinding(second, ph.CollisionMotion(support)),
    )
    sources = {first: (a, group, local), second: (b, None, support)}
    return ph.CollisionModel((a, b)), bindings, sources, (first, second)


def _expected(data, s, u, *, motor=None, damping=0.0):
    """Independent component mechanics plus the declared finite +x face formula."""
    q, v = data["P"] @ s + data["c"], data["P"] @ u
    terms = _ref().oracle(data["model"], q, v)
    body = len(data["model"]["mass"]) - 1
    Q, shift = _placement(data)
    center = terms["com_position"][body]
    base_center = Q.T @ (center - shift)
    initial = np.asarray(_INITIAL_CENTERS[data["name"]])
    depth0 = 1 / 32 if data["name"] == "A04" else 3 / 64
    face = initial[0] - 1 / 8 + depth0
    # Frozen windows stay on this actual finite face. Do not silently replace an
    # escaped finite face by an infinite plane in the independent reference.
    assert abs(base_center[1] - initial[1]) < 1 / 4
    assert abs(base_center[2]) < 1 / 4
    assert base_center[0] > face
    depth = 1 / 8 + face - base_center[0]
    normal = Q[:, 0]
    speed = float(terms["com_velocity"][body] @ normal)
    magnitude = max(0.0, 64 * depth - damping * speed) if depth > 0 else 0.0
    applied = copy.deepcopy(data)
    applied["force"][body] += magnitude * normal
    truth = _ref().kkt(applied, s, u, motor)
    truth["contact_force"] = magnitude * normal
    truth["contact_generalized"] = terms["linear_jacobian"][body].T @ (magnitude * normal)
    truth["input_external"] = _ref().external(terms, data["effort"], data["force"], data["torque"])
    truth["contact_magnitude"] = magnitude
    return truth


def _state(asset, data, checkpoint):
    full_order, order = _orders(asset, data)
    inverse_full, inverse = np.argsort(full_order), np.argsort(order)
    s, u = np.asarray(checkpoint.position)[inverse], np.asarray(checkpoint.velocity)[inverse]
    q, v = (
        np.asarray(checkpoint.full_position)[inverse_full],
        np.asarray(checkpoint.full_velocity)[inverse_full],
    )
    _close(q, data["P"] @ s + data["c"], atol=2e-5)
    _close(v, data["P"] @ u, atol=2e-5)
    _close(data["C"] @ q, [row[3] for row in data["rows"]], atol=2e-5)
    _close(data["C"] @ v, np.zeros(len(data["rows"])), atol=2e-5)
    return s, u


def _row_order(asset, data):
    names = data["model"]["joint_names"]
    declared = [(names[d], names[f], a, b) for d, f, a, b in data["rows"]]
    return [declared.index((r.driver, r.follower, r.multiplier, r.offset)) for r in asset.relations]


def _audit_sites(receipt, diagnostic, sources):
    full_velocity = np.asarray(
        getattr(receipt.entry_state, "full_velocity", receipt.entry_state.velocity), dtype=float
    )
    wrench = np.zeros_like(receipt.law_wrench_fp64, dtype=float)
    static, total_work, total_material_work = {}, 0.0, 0.0
    for item in receipt.sites:
        witness = item.geometry.witness
        point = np.asarray(item.geometry.load_point, float)
        np.testing.assert_array_equal(item.first_force_n, -item.second_force_n)
        work = material_work = 0.0
        for side, key in zip(("first", "second"), item.pair, strict=True):
            shape, group, local = sources[key]
            weights = getattr(witness, side + "_weights")
            by_id = dict(zip(shape.vertex_ids, local, strict=True))
            material_local = sum(weight * by_id[node] for node, weight in weights)
            skin = getattr(witness, side + "_offset")
            surface = np.asarray(getattr(witness, side + "_point"), float)
            if group is None:
                rebuilt = material_local
                reference, velocity = local.mean(axis=0), np.zeros(3)
                displacement = application = np.zeros(3)
            else:
                entry, end = receipt.entry_geometry, receipt.accepted_geometry
                p0, p1 = entry.link_position_m[group], end.link_position_m[group]
                r0, r1 = entry.link_rotation[group], end.link_rotation[group]
                rebuilt = p0 + r0 @ material_local
                reference = entry.com_position_m[group]
                omega = diagnostic.angular_jacobian[group] @ full_velocity
                velocity = diagnostic.linear_jacobian[group] @ full_velocity + np.cross(
                    omega, point - reference
                )
                displacement = p1 + r1 @ np.linalg.solve(r0, surface - p0) - surface
                application = p1 + r1 @ np.linalg.solve(r0, point - p0) - point
            rebuilt = rebuilt + (np.zeros(3) if skin is None else skin)
            _close(rebuilt, surface, atol=1e-10, rtol=0)
            _close(getattr(item, side + "_reference_point_m"), reference, atol=1e-10, rtol=0)
            _close(getattr(item, side + "_velocity_m_s"), velocity)
            _close(
                getattr(item, side + "_material_displacement_m"), displacement, atol=1e-10, rtol=0
            )
            _close(
                getattr(item, side + "_application_displacement_m"), application, atol=1e-10, rtol=0
            )
            force = np.asarray(getattr(item, side + "_force_n"))
            work += float(force @ application)
            material_work += float(force @ displacement)
            contribution = np.r_[force, np.cross(point - reference, force)]
            if group is None:
                static.setdefault(key, np.zeros(6))
                static[key] += contribution
            else:
                wrench[group] += contribution
        _close(item.application_work_j, work, atol=1e-12, rtol=1e-10)
        _close(item.material_work_j, material_work, atol=1e-12, rtol=1e-10)
        total_work += work
        total_material_work += material_work
    _close(receipt.law_wrench_fp64, wrench)
    np.testing.assert_array_equal(receipt.uploaded_wrench_fp32, np.asarray(wrench, np.float32))
    projected = np.einsum("bij,bi->j", diagnostic.linear_jacobian, wrench[:, :3]) + np.einsum(
        "bij,bi->j", diagnostic.angular_jacobian, wrench[:, 3:]
    )
    _close(receipt.contact_generalized, projected)
    for key, reaction in receipt.static_reactions.items():
        _close(reaction.reference_point_m, sources[key][2].mean(axis=0), atol=1e-12, rtol=0)
        _close(reaction.force_n, static.get(key, np.zeros(6))[:3])
        _close(reaction.torque_nm, static.get(key, np.zeros(6))[3:])
    return dict(application_work_j=total_work, material_work_j=total_material_work)


def _audit_step(batch, world, asset, data, entry, receipt, sources, *, h=H, motor=None):
    full_order, order = _orders(asset, data)
    rows = _row_order(asset, data)
    s, u = _state(asset, data, entry)
    truth = _expected(data, s, u, motor=motor)
    after = receipt.accepted_state
    actual_s, actual_u = _state(asset, data, after)
    _close(actual_u, u + h * truth["acceleration"])
    _close(actual_s, s + h * (u + h * truth["acceleration"]))
    diag = batch.tree_diagnostics([world])[world]
    assert (receipt.entry_tick, receipt.exit_tick, receipt.dt_s) == (entry.tick, entry.tick + 1, h)
    assert receipt.entry_state == entry and after == batch.snapshot([world])[world]
    _close(diag.mass, truth["full"]["mass"][np.ix_(full_order, full_order)])
    _close(diag.bias, truth["full"]["bias"][full_order])
    _close(diag.linear_jacobian, truth["full"]["linear_jacobian"][full_order][:, :, full_order])
    _close(diag.angular_jacobian, truth["full"]["angular_jacobian"][full_order][:, :, full_order])
    _close(diag.reduced_mass, truth["mass"][np.ix_(order, order)])
    _close(diag.reduced_bias, truth["bias"][order])
    _close(diag.reduced_external, truth["external"][order])
    _close(diag.external, truth["external_full"][full_order])
    _close(diag.acceleration, truth["acceleration"][order])
    _close(diag.full_acceleration, truth["full_acceleration"][full_order])
    _close(diag.constraint_effort_full, truth["reaction"][full_order])
    _close(diag.row_multipliers, truth["multipliers"][rows])
    _close(diag.row_powers_w, np.zeros(len(rows)), atol=2e-5)
    _close(diag.constraint_power_w, 0, atol=2e-5)
    _close(receipt.input_external, truth["input_external"][full_order])
    _close(receipt.contact_generalized, truth["contact_generalized"][full_order])
    _close(receipt.actual_drive, after.applied_effort)
    meta = receipt.diagnostics
    assert meta["coordinate_kind"] == "affine"
    assert (meta["full_dof_count"], meta["independent_dof_count"]) == data["P"].shape
    assert isinstance(meta["graph_token"], tuple) and meta["graph_token"]
    _close(meta["projection_matrix"], data["P"][full_order][:, order])
    _close(meta["offset_vector"], data["c"][full_order])
    _close(meta["constraint_rows"], data["C"][rows][:, full_order])
    assert _json(meta["affine_diagnostic"]) == _json(diag)
    assert meta["equation_residual"] == diag.equation_residual
    assert after.control == entry.control
    work = _audit_sites(receipt, diag, sources)
    next_q, next_v = data["P"] @ actual_s + data["c"], data["P"] @ actual_u
    next_mass = _ref().oracle(data["model"], next_q, next_v)["mass"]
    entry_ke = float(truth["v"] @ truth["full"]["mass"] @ truth["v"] / 2)
    accepted_ke = float(next_v @ next_mass @ next_v / 2)
    _close(meta["entry_kinetic_energy_j"], entry_ke)
    _close(meta["accepted_kinetic_energy_j"], accepted_ke)
    _close(meta["kinetic_energy_change_j"], accepted_ke - entry_ke)
    _close(meta["application_work_j"], work["application_work_j"], atol=1e-12, rtol=1e-10)
    _close(meta["material_work_j"], work["material_work_j"], atol=1e-12, rtol=1e-10)
    full_power = float(receipt.contact_generalized @ np.asarray(entry.full_velocity))
    reduced_power = float(
        (meta["projection_matrix"].T @ receipt.contact_generalized) @ np.asarray(entry.velocity)
    )
    _close(full_power, reduced_power, atol=2e-5)
    return dict(
        independent=truth,
        diagnostic=diag,
        row_multipliers_source=np.asarray(diag.row_multipliers)[np.argsort(rows)],
        work=work,
        entry_virtual_power_w=full_power,
    )


@pytest.mark.tyx
@pytest.mark.parametrize("name", tuple(EXACT))
def test_c702_c705_full_coupling_and_original_row_reaction(name, tyx_mode, tmp_path):
    data = _placed_case(name)
    truth = _expected(data, data["s"], data["u"])
    for actual, exact in zip(
        (truth["acceleration"], truth["multipliers"], truth["reaction"]), EXACT[name], strict=True
    ):
        _close(actual, exact, atol=2e-13, rtol=0)
    asset = _asset(tmp_path, data)
    world = _world(asset, data)
    record = dict(input=data, exact=EXACT[name], oracle=truth, h_s=H, backend=tyx_mode)
    with ph.TyxWorldBatch([(WORLD, world)], tyx_mode, cache_root=tmp_path) as batch:
        batch.set_controls({WORLD: _control(world, asset, data)})
        model, bindings, sources, pair = _contact_geometry(batch, asset, data)
        before = batch.snapshot()[WORLD]
        with batch.prepare_contacts(
            model, bindings, {pair: ph.RobotContactMaterial(64, 0, 0)}
        ) as plan:
            try:
                assert batch.step_contacts(plan, H) == {WORLD: True}
                receipt = plan.last_accepted[WORLD]
                record.update(
                    receipt=receipt,
                    audit=_audit_step(batch, WORLD, asset, data, before, receipt, sources),
                )
            finally:
                record.update(
                    retained=batch.snapshot(), stats=batch.stats(), plan_stats=plan.stats()
                )
                _save(tmp_path, name + "-exact-contact", record)


@pytest.mark.tyx
@pytest.mark.parametrize("kind,sites", [("capsule", 2), ("hull", 4)])
def test_c701_finite_patch_normalization_does_not_multiply_follower_load(
    kind, sites, tyx_mode, tmp_path
):
    data = _placed_case("A01F")
    asset = _asset(tmp_path, data)
    with ph.TyxWorldBatch([(WORLD, _world(asset, data))], tyx_mode, cache_root=tmp_path) as batch:
        model, bindings, sources, pair = _contact_geometry(batch, asset, data, kind=kind)
        before = batch.snapshot()[WORLD]
        with batch.prepare_contacts(
            model, bindings, {pair: ph.RobotContactMaterial(64, 0, 0)}
        ) as plan:
            assert batch.step_contacts(plan, H) == {WORLD: True}
            receipt = plan.last_accepted[WORLD]
            assert len(receipt.sites) == sites
            assert sum(site.geometry.weight for site in receipt.sites) == pytest.approx(1)
            _close(receipt.law_wrench_fp64[-1], (3, 0, 0, 0, 0, 0))
            audit = _audit_step(batch, WORLD, asset, data, before, receipt, sources)
            _save(
                tmp_path,
                kind + "-normalization",
                dict(input=data, receipt=receipt, audit=audit, stats=batch.stats()),
            )


def _window(directory, data, mode, *, reorder=False, steps=40):
    asset = _asset(directory, data, reorder=reorder)
    world = _world(asset, data)
    expected_s, expected_u = data["s"].copy(), data["u"].copy()
    rows = []
    with ph.TyxWorldBatch([(WORLD, world)], mode, cache_root=directory) as batch:
        batch.set_controls({WORLD: _control(world, asset, data)})
        model, bindings, sources, pair = _contact_geometry(batch, asset, data)
        with batch.prepare_contacts(
            model, bindings, {pair: ph.RobotContactMaterial(64, 0, 0)}
        ) as plan:
            try:
                for index in range(steps):
                    expected = _expected(data, expected_s, expected_u)
                    expected_u = expected_u + H * expected["acceleration"]
                    expected_s = expected_s + H * expected_u
                    before = batch.snapshot()[WORLD]
                    assert batch.step_contacts(plan, H) == {WORLD: True}
                    receipt = plan.last_accepted[WORLD]
                    audit = _audit_step(batch, WORLD, asset, data, before, receipt, sources)
                    actual_s, actual_u = _state(asset, data, receipt.accepted_state)
                    _close(actual_s, expected_s, atol=WINDOW_ATOL)
                    _close(actual_u, expected_u, atol=WINDOW_ATOL)
                    assert receipt.exit_tick == index + 1
                    rows.append(
                        dict(
                            receipt=receipt,
                            audit=audit,
                            actual_state=(actual_s.copy(), actual_u.copy()),
                            oracle_state=(expected_s.copy(), expected_u.copy()),
                            stats=batch.stats(),
                        )
                    )
            finally:
                _save(
                    directory,
                    "continuous-contact",
                    dict(
                        input=data,
                        h_s=H,
                        steps=steps,
                        rows=rows,
                        last_accepted=plan.last_accepted,
                        diagnostic=batch.tree_diagnostics(),
                        retained=batch.snapshot(),
                        plan_stats=plan.stats(),
                        stats=batch.stats(),
                    ),
                )
    return rows


@pytest.mark.tyx
@pytest.mark.parametrize("name", ("A01F", "A02", "A03", "A04", "A05", "A05G"))
def test_c701_c705_forty_continuous_steps_use_current_geometry(name, tyx_mode, tmp_path):
    rows = _window(tmp_path, _placed_case(name), tyx_mode)
    assert len(rows) == 40
    if name == "A04":
        first = rows[0]["receipt"]
        assert not np.array_equal(
            first.entry_geometry.link_rotation, first.accepted_geometry.link_rotation
        )
        assert any(
            np.linalg.norm(site.first_application_displacement_m) > 0 for site in first.sites
        )


@pytest.mark.tyx
@pytest.mark.parametrize("name", ("A02", "A05", "A05G"))
def test_c704_reordered_asset_and_rigid_coordinate_change_are_covariant(name, tyx_mode, tmp_path):
    base = _window(tmp_path / "base", _placed_case(name), tyx_mode)
    moved = _window(tmp_path / "cycle-reverse", _placed_case(name, "cycle"), tyx_mode, reorder=True)
    for a, b in zip(base, moved, strict=True):
        _close(a["oracle_state"][0], b["oracle_state"][0], atol=2e-13, rtol=0)
        _close(a["actual_state"][0], b["actual_state"][0], atol=WINDOW_ATOL)
        _close(a["actual_state"][1], b["actual_state"][1], atol=WINDOW_ATOL)
        _close(a["audit"]["row_multipliers_source"], b["audit"]["row_multipliers_source"])


@pytest.mark.tyx
def test_c703_chain_and_flatten_keep_motion_but_preserve_distinct_original_rows(tyx_mode, tmp_path):
    chain = _window(tmp_path / "chain", _placed_case("A03"), tyx_mode, steps=2)
    flat = _window(tmp_path / "flatten", _placed_case("A03_flat"), tyx_mode, steps=2)
    for a, b in zip(chain, flat, strict=True):
        _close(a["receipt"].accepted_state.position, b["receipt"].accepted_state.position)
        _close(a["receipt"].accepted_state.full_velocity, b["receipt"].accepted_state.full_velocity)
        assert a["receipt"].diagnostics["graph_token"] != b["receipt"].diagnostics["graph_token"]
        assert not np.allclose(
            a["audit"]["diagnostic"].row_multipliers, b["audit"]["diagnostic"].row_multipliers
        )


@pytest.mark.tyx
@pytest.mark.parametrize("delay", (0, 1))
def test_c7_pd_delay_retains_current_contact_and_real_independent_history(
    delay, tyx_mode, tmp_path
):
    data = _placed_case("A01F")
    asset = _asset(tmp_path, data)
    world = _world(asset, data, kd=[6], cap=[1 / 4], delay=delay)
    rows = []
    with ph.TyxWorldBatch([(WORLD, world)], tyx_mode, cache_root=tmp_path) as batch:
        batch.set_controls({WORLD: _control(world, asset, data, rate=[1 / 4], ff=[1 / 8])})
        model, bindings, sources, pair = _contact_geometry(batch, asset, data)
        with batch.prepare_contacts(
            model, bindings, {pair: ph.RobotContactMaterial(64, 0, 0)}
        ) as plan:
            for index in range(2):
                before = batch.snapshot()[WORLD]
                tau = -6 * H / (1 + H) if delay and index == 0 else 1 / 4
                expected_selected = (0, 0, 0) if delay and index == 0 else (0, 1 / 4, 1 / 8)
                assert batch.step_contacts(plan, H) == {WORLD: True}
                receipt = plan.last_accepted[WORLD]
                _close(receipt.actual_drive, [tau])
                _close(receipt.selected_drive, [expected_selected])
                audit = _audit_step(
                    batch, WORLD, asset, data, before, receipt, sources, motor=[tau]
                )
                assert receipt.entry_state.control == receipt.accepted_state.control
                if delay:
                    _close(receipt.accepted_state.history, [(0, 1 / 4, 1 / 8)])
                rows.append(dict(receipt=receipt, audit=audit))
            _save(
                tmp_path,
                "pd-delay-contact",
                dict(input=data, delay=delay, rows=rows, stats=batch.stats()),
            )


@pytest.mark.tyx
def test_c7_low_motor_cap_does_not_clip_the_external_contact(tyx_mode, tmp_path):
    data = _placed_case("A01F")
    asset = _asset(tmp_path, data)
    world = _world(asset, data, kd=[6], cap=[1 / 1024])
    with ph.TyxWorldBatch([(WORLD, world)], tyx_mode, cache_root=tmp_path) as batch:
        model, bindings, sources, pair = _contact_geometry(batch, asset, data)
        before = batch.snapshot()[WORLD]
        with batch.prepare_contacts(
            model, bindings, {pair: ph.RobotContactMaterial(64, 0, 0)}
        ) as plan:
            assert batch.step_contacts(plan, H) == {WORLD: True}
            receipt = plan.last_accepted[WORLD]
            _close(receipt.actual_drive, [-1 / 1024])
            _close(receipt.contact_generalized, [0, 3])
            _close(batch.tree_diagnostics([WORLD])[WORLD].acceleration, [1 - 1 / 6144])
            audit = _audit_step(
                batch, WORLD, asset, data, before, receipt, sources, motor=[-1 / 1024]
            )
            _save(
                tmp_path,
                "low-cap-contact",
                dict(input=data, receipt=receipt, audit=audit, stats=batch.stats()),
            )


@pytest.mark.tyx
@pytest.mark.parametrize("h", (1 / 1024, 1 / 2048))
def test_c706_actual_unload_and_reentry_without_reset_or_control_change(h, tyx_mode, tmp_path):
    data = _placed_case("A01F")
    data["u"][:] = 1 / 4
    data["motor"][:] = -6
    asset = _asset(tmp_path, data)
    world = _world(asset, data)
    s, u = 0.0, 1 / 4
    rows, departed, returned = [], None, None
    with ph.TyxWorldBatch([(WORLD, world)], tyx_mode, cache_root=tmp_path) as batch:
        batch.set_controls({WORLD: _control(world, asset, data)})
        model, bindings, sources, pair = _contact_geometry(batch, asset, data)
        with batch.prepare_contacts(
            model, bindings, {pair: ph.RobotContactMaterial(64, 0, 0)}
        ) as plan:
            try:
                for index in range(round((7 / 8) / h)):
                    independent_n = max(0.0, 64 * (3 / 64 - 2 * s))
                    u += h * (-6 + 2 * independent_n) / 6
                    s += h * u
                    before = batch.snapshot()[WORLD]
                    assert batch.step_contacts(plan, h) == {WORLD: True}
                    receipt = plan.last_accepted[WORLD]
                    audit = _audit_step(batch, WORLD, asset, data, before, receipt, sources, h=h)
                    actual_s, actual_u = _state(asset, data, receipt.accepted_state)
                    _close(actual_s, [s], atol=WINDOW_ATOL)
                    _close(actual_u, [u], atol=WINDOW_ATOL)
                    normal = float(receipt.contact_generalized[1])
                    if departed is None and normal == 0:
                        departed = index + 1
                    elif departed is not None and normal > 0:
                        returned = index + 1
                    rows.append(
                        dict(
                            receipt=receipt,
                            audit=audit,
                            independent=(s, u, independent_n),
                            departed=departed,
                            returned=returned,
                            stats=batch.stats(),
                        )
                    )
                    if returned is not None:
                        break
                assert departed is not None and returned is not None and returned > departed
                assert rows[-1]["receipt"].exit_tick * h <= 7 / 8
            finally:
                _save(
                    tmp_path,
                    "unload-reentry",
                    dict(
                        input=data,
                        h_s=h,
                        maximum_time_s=7 / 8,
                        departed=departed,
                        returned=returned,
                        rows=rows,
                        last_accepted=plan.last_accepted,
                        diagnostic=batch.tree_diagnostics(),
                        retained=batch.snapshot(),
                        stats=batch.stats(),
                    ),
                )
