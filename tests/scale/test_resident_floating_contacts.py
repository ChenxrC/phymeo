"""C10 public floating contact inputs and independent physical anchors.

AUTHORING_UNEXECUTED. Parent alone executes these tests. The first node proves
an old HOST finite patch/law and a literal root-only mass matrix before calling
the already-public prepare_contacts entry whose floating domain is missing.
"""

import gzip
import json
from collections.abc import Mapping
from dataclasses import fields, is_dataclass, replace
from functools import lru_cache
from itertools import combinations, product
from pathlib import Path

import numpy as np
import pytest
from test_resident_affine_contacts import _audit_sites
from test_resident_floating_step import _model as _floating_model

import phymeo as ph
from phymeo.core.rigid import RigidBodyModel
from phymeo.dynamics.floating_robot import FloatingRobotModel, FloatingRobotState
from phymeo.geometry.contact_patch_adapter import contact_patch_sample
from phymeo.solvers.contact_patch_loads import sample_patch_loads
from phymeo.solvers.floating_tree import FloatingRobotInputs, floating_dynamics_terms

H = 1 / 1024
ATOL, RTOL, WINDOW_ATOL = 5e-6, 5e-5, 2e-4
WORLD = 503


def _json(value):
    if is_dataclass(value):
        return {f.name: _json(getattr(value, f.name)) for f in fields(value)}
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
    directory.mkdir(parents=True, exist_ok=True)
    with gzip.open(directory / (name + ".json.gz"), "wt", encoding="utf-8") as stream:
        json.dump(_json(payload), stream, allow_nan=False, separators=(",", ":"))


def _close(actual, expected, *, atol=ATOL, rtol=RTOL):
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)


def _box(low, high):
    return np.array(list(product(*zip(low, high, strict=True))), dtype=float)


@pytest.mark.tyx
def test_f01_public_root_contact_keeps_full_six_coordinate_response(tyx_mode, tmp_path):
    root = RigidBodyModel(np.array([0.5]), np.full((1, 3), 0.5), np.zeros(3))
    model = FloatingRobotModel(root, np.zeros(3), None)
    initial = FloatingRobotState(np.array([0.0, 0, 3 / 32, 0, 0, 0, 1]), np.zeros(6))
    ball_key, wall_key = ph.CollisionKey(WORLD, 101, 7), ph.CollisionKey(WORLD, 900, 2)
    ball = ph.CollisionShape(ball_key, "sphere", (101,), 1 / 8, revision=7)
    wall = ph.CollisionShape(wall_key, "hull", tuple(501 + 3 * i for i in range(8)), revision=3)
    wall_points = _box([-1, -1, -1 / 4], [1, 1, 0])
    patch = contact_patch_sample(ball, wall, [[0, 0, 3 / 32]], wall_points)
    assert len(patch.sites) == 1 and patch.dimension == 0
    site = patch.sites[0]
    _close(site.witness.gap_m, -1 / 32, atol=1e-12, rtol=0)
    _close(site.witness.normal, [0, 0, -1], atol=1e-12, rtol=0)
    assert site.weight == 1
    law = ph.RobotContactMaterial(64, 0, 0)
    measured = sample_patch_loads(patch, law, np.zeros((1, 3)), np.zeros((1, 3)))
    _close(measured["first_force"], [[0, 0, 2]], atol=1e-12, rtol=0)
    _close(measured["second_force"], [[0, 0, -2]], atol=1e-12, rtol=0)
    _close(
        np.cross(np.array(site.load_point) - initial.q[:3], measured["first_force"][0]),
        0,
        atol=1e-12,
        rtol=0,
    )
    known = FloatingRobotInputs(np.zeros(0), np.array([[0.0, 0, 2]]), np.zeros((1, 3)))
    host = floating_dynamics_terms(model, initial, known)
    literal_mass = np.diag([2.0, 2, 2, 0.5, 0.5, 0.5])
    literal_acceleration = np.array([0.0, 0, 1, 0, 0, 0])
    _close(host.mass_matrix, literal_mass, atol=1e-12, rtol=0)
    _close(host.bias, np.zeros(6), atol=1e-12, rtol=0)
    _close(host.applied, [0, 0, 2, 0, 0, 0], atol=1e-12, rtol=0)
    _close(host.acceleration, literal_acceleration, atol=1e-12, rtol=0)
    record = dict(
        case="F01-public",
        mode=tyx_mode,
        dt_s=H,
        model=model,
        initial=initial,
        shapes=(ball, wall),
        wall_points=wall_points,
        local_ball=np.zeros((1, 3)),
        law=law,
        host_patch=patch,
        host_loads=measured,
        host_terms=host,
        literal_mass=literal_mass,
        literal_acceleration=literal_acceleration,
        drive=np.zeros((0, 3)),
        external=None,
    )
    try:
        tree = ph.FloatingRobotTree(
            model, initial, kp=np.zeros(0), kd=np.zeros(0), effort_limit=np.zeros(0)
        )
        with ph.TyxWorldBatch([(WORLD, tree)], tyx_mode, cache_root=tmp_path) as batch:
            batch.set_controls({WORLD: tree.inputs(np.zeros((0, 3)))})
            before = batch.snapshot()[WORLD]
            before_geometry = batch.tree_kinematics()[WORLD]
            assert (before_geometry.body_count, before_geometry.velocity_count) == (1, 6)
            _close(before_geometry.link_position_m, [[0, 0, 3 / 32]])
            collision = ph.CollisionModel((ball, wall))
            bindings = (
                ph.ResidentFloatingTreeBinding(
                    ball_key, WORLD, 0, np.zeros((1, 3)), topology=before.topology
                ),
                ph.ResidentStaticBinding(wall_key, ph.CollisionMotion(wall_points)),
            )
            record.update(entry=before, entry_geometry=before_geometry, bindings=bindings)
            # Effective RED: all physical and original resident prerequisites
            # above must pass before the current floating-domain guard is met.
            try:
                with batch.prepare_contacts(
                    collision, bindings, {(ball_key, wall_key): law}
                ) as plan:
                    assert batch.step_contacts(plan, H) == {WORLD: True}
                    receipt = plan.last_accepted[WORLD]
                    after = batch.snapshot()[WORLD]
                    geometry = batch.tree_kinematics()[WORLD]
                    record.update(
                        receipt=receipt,
                        accepted=after,
                        geometry=geometry,
                        plan_stats=plan.stats(),
                        batch_stats=batch.stats(),
                    )
                    assert receipt.entry_state == before and receipt.accepted_state == after
                    assert (receipt.entry_tick, receipt.exit_tick) == (0, 1)
                    assert after.control == before.control and not after.history
                    assert receipt.selected_drive.shape == (
                        0,
                        3,
                    ) and receipt.actual_drive.shape == (0,)
                    assert receipt.law_wrench_fp64.shape == (1, 6)
                    _close(receipt.law_wrench_fp64, [[0, 0, 2, 0, 0, 0]])
                    _close(receipt.input_external, np.zeros(6))
                    _close(receipt.contact_generalized, [0, 0, 2, 0, 0, 0])
                    _close(receipt.static_reactions[wall_key].force_n, [0, 0, -2])
                    _close(receipt.static_reactions[wall_key].torque_nm, np.zeros(3))
                    d = receipt.diagnostics
                    assert d["coordinate_kind"] == "floating"
                    assert tuple(
                        d[k]
                        for k in ("joint_count", "body_count", "velocity_count", "position_count")
                    ) == (0, 1, 6, 7)
                    _close(d["mass"], literal_mass, atol=2e-5)
                    _close(d["acceleration"], literal_acceleration)
                    _close(after.velocity, H * literal_acceleration)
                    _close(after.position[:3], [0, 0, 3 / 32 + H * H])
                    _close(after.position[3:], [0, 0, 0, 1])
                    _close(geometry.linear_velocity_m_s, [[0, 0, H]])
                    assert d["floating_diagnostic"].equation_residual == d["equation_residual"]
                    assert batch.stats()["floating_world_count"] == 1
                    assert batch.stats()["submitted_kernels"] > 0
            except Exception:
                record.update(
                    retained=batch.snapshot(),
                    retained_geometry=batch.tree_kinematics(),
                    batch_stats=batch.stats(),
                )
                raise
    except Exception as error:
        record["failure"] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        _save(tmp_path, "f01-public-floating-contact", record)


# C10 complete frozen matrix. Existing F01 above is unchanged.


@lru_cache(maxsize=1)
def _reference():
    import importlib.util

    path = (
        Path(__file__).resolve().parents[1]
        / "reference_newton/robot/resident_floating_contact_reference.py"
    )
    spec = importlib.util.spec_from_file_location("c10_independent_contacts", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _item(data, *, delay=0):
    import copy

    data = copy.deepcopy(data)
    data["delay"] = delay
    asset = None
    if data["name"] == "F06":
        asset = ph.load_floating_urdf(
            Path(__file__).resolve().parents[1] / "fixtures/urdf/floating_fixed_composite.urdf",
            gravity=(0.0, 0.0, 0.0),
            collapse_fixed_joints=True,
        )
    tree = ph.FloatingRobotTree(
        _floating_model(data) if asset is None else asset.model,
        FloatingRobotState(data["q"], data["v"]),
        kp=data["kp"],
        kd=data["kd"],
        effort_limit=data["caps"],
        delay_steps=delay,
    )
    external = _reference().base().external(data, data["q"])
    command = tree.inputs(data["drive"], FloatingRobotInputs(*external))
    return dict(data=data, tree=tree, command=command, external=external, asset=asset, delay=delay)


def _geometry(batch, item, *, world=WORLD, reverse=False):
    data, asset = item["data"], item["asset"]
    topology = batch.snapshot([world])[world].topology
    shapes, bindings, sources, keys = [], [], {}, []
    for i, declared in enumerate(data["shapes"]):
        group, points = declared["group"], declared["points"]
        key = ph.CollisionKey(world, 101 + group if group is not None else 900 + i, 7 + i)
        if asset is not None and group is not None:
            collider = next(c for c in asset.collisions if c.link_name == "tool")
            shape = replace(collider.shape, key=key)
            binding = ph.ResidentFloatingTreeBinding.from_asset(
                asset,
                collider.shape.key,
                key=key,
                world=world,
            )
            _close(
                sorted(binding.local_points_m.tolist()), sorted(points.tolist()), atol=1e-12, rtol=0
            )
            points = np.asarray(binding.local_points_m)
            assert binding.group_index == 0
        else:
            ids = tuple(1000 * i + 11 + 3 * k for k in range(len(points)))
            shape = ph.CollisionShape(key, declared["kind"], ids, declared["radius"], revision=7)
            binding = (
                ph.ResidentStaticBinding(key, ph.CollisionMotion(points))
                if group is None
                else ph.ResidentFloatingTreeBinding(key, world, group, points, topology=topology)
            )
        shapes.append(shape)
        bindings.append(binding)
        sources[key] = (shape, group, points)
        keys.append(key)
    laws = {
        tuple(sorted((keys[r["first"]], keys[r["second"]]))): ph.RobotContactMaterial(
            r["stiffness"], r["damping"], 0.0
        )
        for r in data["rules"]
    }
    excluded = tuple(
        (a.key, b.key)
        for a, b in combinations(shapes, 2)
        if a.key.body != b.key.body and tuple(sorted((a.key, b.key))) not in laws
    )
    if reverse:
        shapes.reverse()
        bindings.reverse()
    return ph.CollisionModel(tuple(shapes), excluded), tuple(bindings), sources, laws


def _selected(item, tick):
    if tick < item["delay"]:
        # Original C9 delay history holds the declared initial joint target.
        # Build it from authored input, independently of a product checkpoint.
        initial = np.zeros_like(item["data"]["drive"])
        initial[:, 0] = np.asarray(item["data"]["q"][7:], dtype=np.float32)
        return initial
    return item["data"]["drive"]


def _first_anchors(data):
    """Zero-delay literal mechanics; actual delayed rows have their own oracle."""
    if data["name"] == "F07":
        return None
    ref = _reference()
    q, v, truth = ref.advance(data, data["q"], data["v"], h=H)
    name, variant = data["name"], data["variant"]
    a, w = truth["acceleration"], truth["wrench"]
    if data["placement"] == "base":
        if name == "F01":
            _close(w, [[0, 0, 2, 0, 0, 0]], atol=1e-12, rtol=0)
            _close(a, [0, 0, 1, 0, 0, 0], atol=1e-12, rtol=0)
        elif name == "F02":
            expected = {"child": (0.0, 2.0), "motor": (-1.5, 6.5), "root": (1.0, -1.0)}[variant]
            _close(a[[0, 6]], expected, atol=1e-12, rtol=0)
            _close(truth["mass"][np.ix_([0, 6], [0, 6])], [[3, 1], [1, 1]], atol=1e-12, rtol=0)
        elif name == "F03":
            _close(w[:, :3], [[0, 0, 0], [-2, 0, 0], [2, 0, 0]], atol=1e-12, rtol=0)
            _close(a[[0, 6, 7]], [0, -1, 2], atol=1e-12, rtol=0)
        elif name == "F04":
            _close(w[1, 0], 65 / 32, atol=1e-12, rtol=0)
            _close(a[[0, 5, 6]], [65 / 96, -2, 127 / 32], atol=1e-12, rtol=0)
        elif name == "F05" and variant == "origin":
            _close(w, [[0, 2, 0, 0, 0, -1 / 4]], atol=1e-12, rtol=0)
            _close(a, [0, 1, 0, 0, 0, -0.5], atol=1e-12, rtol=0)
        elif name == "F05":
            _close(w, [[0, 0, 0, 1, 0, 0]], atol=1e-12, rtol=0)
            _close(a, [0, 0, 0, 3, 1, 0], atol=1e-12, rtol=0)
        elif name == "F06":
            _close(w, [[0, 3, 0, 0, 0, 0.5]], atol=1e-12, rtol=0)
            _close(a, [0, 1, 0, 0, 0, 12 / 13], atol=1e-12, rtol=0)
        elif name == "F08_branch":
            _close(truth["contact_generalized"], [0, 0, 0, 0, 0, 0, 0, -2, 0], atol=1e-12, rtol=0)
            assert abs(truth["mass"][6, 7]) > 0
        elif name == "F08":
            group = data["shapes"][0]["group"]
            _close(w[group, :3], [2, 0, 0], atol=1e-12, rtol=0)
        elif name == "F09":
            _close(w[0, 2], 1 / 64, atol=1e-12, rtol=0)
            _close(a[2], -1023 / 128, atol=1e-12, rtol=0)
    return dict(q=q, v=v, truth=truth)


def _audit(batch, item, receipt, sources, *, world=WORLD, h=H):
    data, ref = item["data"], _reference()
    entry, after = receipt.entry_state, receipt.accepted_state
    n, b, d = len(data["drive"]), len(data["drive"]) + 1, len(data["drive"]) + 6
    assert receipt.world == world and receipt.entry_tick == entry.tick
    assert receipt.exit_tick == after.tick == entry.tick + 1
    assert receipt.dt_s == h and after == batch.snapshot([world])[world]
    diagnostics = receipt.diagnostics
    assert diagnostics["coordinate_kind"] == "floating"
    assert tuple(
        diagnostics[k] for k in ("joint_count", "body_count", "velocity_count", "position_count")
    ) == (n, b, d, n + 7)
    assert receipt.law_wrench_fp64.shape == receipt.uploaded_wrench_fp32.shape == (b, 6)
    assert receipt.input_external.shape == receipt.contact_generalized.shape == (d,)
    assert receipt.selected_drive.shape == (n, 3) and receipt.actual_drive.shape == (n,)
    selected = _selected(item, entry.tick)
    _close(receipt.selected_drive, selected, atol=0, rtol=0)
    if data["name"] == "F07":
        q, v, expected = ref.response(
            data,
            np.array(entry.position),
            np.array(entry.velocity),
            np.asarray(receipt.uploaded_wrench_fp32, float),
            h=h,
            drive=selected,
            external_packet=item["external"],
        )
    else:
        q, v, expected = ref.advance(
            data,
            np.array(entry.position),
            np.array(entry.velocity),
            h=h,
            drive=selected,
            external_packet=item["external"],
        )
        _close(receipt.law_wrench_fp64, expected["wrench"])
    for field in ("mass", "bias", "linear_jacobian", "angular_jacobian", "acceleration"):
        _close(diagnostics[field], expected[field], atol=2e-5 if field == "mass" else ATOL)
    _close(receipt.contact_generalized, expected["contact_generalized"])
    _close(receipt.input_external, expected["input_external"])
    _close(receipt.actual_drive, expected["drive"])
    _close(
        diagnostics["total_external"], expected["input_external"] + expected["contact_generalized"]
    )
    _close(after.position, q)
    _close(after.velocity, v)
    observed = ref.base().mechanics(data, np.array(after.position), np.array(after.velocity))
    for public, independent in (
        ("link_position_m", "position"),
        ("link_rotation", "rotation"),
        ("com_position_m", "com_position"),
        ("linear_velocity_m_s", "com_velocity"),
        ("angular_velocity_rad_s", "angular_velocity"),
    ):
        _close(getattr(receipt.accepted_geometry, public), observed[independent])
    nested = diagnostics["floating_diagnostic"]
    assert nested.entry_tick == entry.tick and nested.exit_tick == after.tick
    for field in ("mass", "bias", "linear_jacobian", "angular_jacobian", "acceleration"):
        _close(getattr(nested, field), diagnostics[field], atol=0, rtol=0)
    _close(nested.external, diagnostics["total_external"], atol=0, rtol=0)
    _close(nested.drive, receipt.actual_drive, atol=0, rtol=0)
    assert diagnostics["equation_residual"] == nested.equation_residual <= 2e-5
    assert diagnostics["control_residual"] <= 5e-5
    actual_total = np.asarray(receipt.input_external) + receipt.contact_generalized
    actual_total = actual_total.copy()
    actual_total[6:] += receipt.actual_drive
    _close(
        expected["mass"] @ np.asarray(diagnostics["acceleration"]) + expected["bias"],
        actual_total,
        atol=2e-5,
    )
    works = _audit_sites(receipt, nested, sources)
    for field, value in works.items():
        _close(diagnostics[field], value, atol=1e-12, rtol=1e-10)
    power = sum(
        float(
            np.asarray(s.first_force_n) @ s.first_velocity_m_s
            + np.asarray(s.second_force_n) @ s.second_velocity_m_s
        )
        for s in receipt.sites
    )
    _close(diagnostics["entry_power_times_h_j"], h * power, atol=1e-12, rtol=1e-10)
    _close(diagnostics["entry_kinetic_energy_j"], expected["energy"])
    _close(diagnostics["accepted_kinetic_energy_j"], observed["energy"])
    # Work and kinetic-energy change are independent observations, not an invented
    # conservation equality for the existing penalty/semi-implicit model.
    assert after.control == entry.control
    for array in (
        receipt.input_external,
        receipt.law_wrench_fp64,
        receipt.uploaded_wrench_fp32,
        receipt.contact_generalized,
        receipt.actual_drive,
        diagnostics["mass"],
    ):
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.setflags(write=True)
    return dict(same_entry=expected, accepted_mechanics=observed, work=works)


def _window(item, mode, directory, *, steps=40, reverse=False, world=WORLD):
    data, ref = item["data"], _reference()
    rows, failure = [], None
    q, v = data["q"].copy(), data["v"].copy()
    record = dict(
        input=data,
        command=item["command"],
        dt_s=H,
        mode=mode,
        steps=steps,
        first_independent_zero_delay=_first_anchors(data),
        rows=rows,
    )
    with ph.TyxWorldBatch([(world, item["tree"])], mode, cache_root=directory) as batch:
        batch.set_controls({world: item["command"]})
        model, bindings, sources, laws = _geometry(batch, item, world=world, reverse=reverse)
        record.update(model=model, bindings=bindings, laws=laws)
        try:
            with batch.prepare_contacts(model, bindings, laws, tree_dynamic_pairs=True) as plan:
                for index in range(steps):
                    before = batch.snapshot([world])[world]
                    assert batch.step_contacts(plan, H) == {world: True}
                    receipt = plan.last_accepted[world]
                    row = dict(
                        entry=before,
                        receipt=receipt,
                        state=batch.snapshot(),
                        geometry=batch.tree_kinematics(),
                        summary=batch.summary(),
                    )
                    rows.append(row)
                    row["audit"] = _audit(batch, item, receipt, sources, world=world)
                    q, v, independent = ref.advance(
                        data,
                        q,
                        v,
                        h=H,
                        drive=_selected(item, index),
                        external_packet=item["external"],
                    )
                    row.update(
                        independent_window=independent,
                        independent_q=q.copy(),
                        independent_v=v.copy(),
                    )
                    _close(receipt.accepted_state.position, q, atol=WINDOW_ATOL)
                    _close(receipt.accepted_state.velocity, v, atol=WINDOW_ATOL)
                record.update(last_status=plan.last_status, stats=batch.stats())
        except BaseException as exc:
            failure = dict(type=type(exc).__name__, message=str(exc))
            raise
        finally:
            record.update(
                retained=batch.snapshot(),
                retained_geometry=batch.tree_kinematics(),
                stats=batch.stats(),
                failure=failure,
            )
            _save(directory, "floating-contact-window", record)
    return rows


@pytest.mark.tyx
@pytest.mark.parametrize("variant", ("child", "motor", "root"))
def test_f02_root_child_and_internal_motor_use_one_full_matrix(variant, tyx_mode, tmp_path):
    _window(_item(_reference().case("F02", variant=variant)), tyx_mode, tmp_path)


@pytest.mark.tyx
@pytest.mark.parametrize(
    "placement,reverse,boost",
    (("base", False, False), ("base", True, False), ("cycle", True, False), ("base", False, True)),
)
def test_f03_two_dynamic_siblings_react_and_transform(
    placement, reverse, boost, tyx_mode, tmp_path
):
    data = _reference().case("F03", placement=placement, boost=boost)
    rows = _window(_item(data), tyx_mode, tmp_path, reverse=reverse)
    initial = _reference().base().mechanics(data, data["q"], data["v"])["momentum"]
    mass = data["root_mass"] + sum(data["children"]["mass"])
    for row in rows:
        _close(
            (row["audit"]["accepted_mechanics"]["momentum"] - initial) / mass,
            np.zeros(3),
            atol=ATOL,
            rtol=0,
        )
        receipt = row["receipt"]
        _close(np.sum(receipt.law_wrench_fp64[:, :3], axis=0), np.zeros(3), atol=1e-12, rtol=0)
        assert not receipt.static_reactions


@pytest.mark.tyx
def test_f04_rotating_tool_uses_entry_point_velocity_and_true_arc(tyx_mode, tmp_path):
    rows = _window(_item(_reference().case("F04")), tyx_mode, tmp_path)
    first = rows[0]["receipt"]
    _close(first.law_wrench_fp64[1, 0], 65 / 32)
    assert any(
        abs(s.material_work_j - s.application_work_j) > 0
        for row in rows
        for s in row["receipt"].sites
    )
    assert not np.array_equal(rows[-1]["receipt"].accepted_geometry.link_rotation[1], np.eye(3))


@pytest.mark.tyx
@pytest.mark.parametrize(
    "variant,placement", (("origin", "base"), ("frame", "base"), ("frame", "cycle"))
)
def test_f05_com_offset_and_rotated_principal_inertia_react(variant, placement, tyx_mode, tmp_path):
    _window(
        _item(_reference().case("F05", variant=variant, placement=placement)), tyx_mode, tmp_path
    )


@pytest.mark.tyx
def test_f06_authored_fixed_tool_is_folded_once_and_reacts(tyx_mode, tmp_path):
    item = _item(_reference().case("F06"))
    assert item["asset"].joint_names == () and item["asset"].model.body_count == 1
    _close(item["asset"].sites["tool"].local_position_m, [1 / 4, 0, 0], atol=1e-12, rtol=0)
    _window(item, tyx_mode, tmp_path)


@pytest.mark.tyx
@pytest.mark.parametrize("variant", ("hull", "cap_side", "cap_end", "cap_miss", "CC", "HH"))
def test_f07_finite_shape_geometry_anchor_then_full_response(variant, tyx_mode, tmp_path):
    data = _reference().case("F07", variant=variant)
    item, rows = _item(data), []
    entry_gap = data["first_gap"]
    anchor = None
    if variant in ("CC", "HH"):
        # Identity-pose support arithmetic independently binds both original
        # decimal geometry and the declared FP32 child-offset representation.
        first, second = data["shapes"]
        normal = data["first_normal"]
        local_gap = (
            np.min(second["points"] @ normal)
            - np.max(first["points"] @ normal)
            - first["radius"]
            - second["radius"]
        )
        offsets = data["children"]["offsets"]
        authored_gap = float((offsets[1] - offsets[0]) @ normal + local_gap)
        _close(authored_gap, data["first_gap"], atol=1e-10, rtol=0)
        stored_offsets = offsets.astype(np.float32).astype(float)
        entry_gap = float((stored_offsets[1] - stored_offsets[0]) @ normal + local_gap)
        anchor = dict(
            authored_gap=authored_gap,
            stored_offsets=stored_offsets,
            expected_entry_gap=entry_gap,
            local_support_gap=float(local_gap),
        )
    record = dict(
        first_geometry_anchor=anchor,
        input=data,
        command=item["command"],
        mode=tyx_mode,
        dt_s=H,
        rows=rows,
        oracle_scope=data["reference_scope"],
    )
    with ph.TyxWorldBatch([(WORLD, item["tree"])], tyx_mode, cache_root=tmp_path) as batch:
        batch.set_controls({WORLD: item["command"]})
        model, bindings, sources, laws = _geometry(batch, item)
        record.update(model=model, bindings=bindings, laws=laws)
        try:
            with batch.prepare_contacts(model, bindings, laws, tree_dynamic_pairs=True) as plan:
                for index in range(2):
                    assert batch.step_contacts(plan, H) == {WORLD: True}
                    receipt = plan.last_accepted[WORLD]
                    row = dict(
                        receipt=receipt,
                        state=batch.snapshot(),
                        geometry=batch.tree_kinematics(),
                        summary=batch.summary(),
                    )
                    rows.append(row)
                    row["audit"] = _audit(batch, item, receipt, sources)
                    if index == 0:
                        magnitude = max(0.0, -64 * entry_gap)
                        _close(receipt.law_wrench_fp64[1, :3], -magnitude * data["first_normal"])
                        _close(receipt.law_wrench_fp64[2, :3], magnitude * data["first_normal"])
                        if variant == "cap_miss":
                            assert not receipt.sites
                        else:
                            _close(
                                sum(s.geometry.weight for s in receipt.sites),
                                1.0,
                                atol=1e-12,
                                rtol=0,
                            )
                            for site in receipt.sites:
                                _close(
                                    site.geometry.witness.gap_m,
                                    entry_gap,
                                    atol=1e-10,
                                    rtol=0,
                                )
                            centroid = np.zeros(3)
                            if variant == "HH":
                                ring = np.array(
                                    [[-0.5, -0.1], [0.5, -0.1], [0.6, 0], [0.3, 0.1], [-0.5, 0.1]]
                                )
                                nxt = np.roll(ring, -1, axis=0)
                                cross = ring[:, 0] * nxt[:, 1] - nxt[:, 0] * ring[:, 1]
                                centroid[:2] = np.sum((ring + nxt) * cross[:, None], axis=0) / (
                                    3 * sum(cross)
                                )
                            for group, sign in ((1, -1.0), (2, 1.0)):
                                expected_torque = np.cross(
                                    centroid, sign * magnitude * data["first_normal"]
                                )
                                _close(receipt.law_wrench_fp64[group, 3:], expected_torque)
        finally:
            record.update(retained=batch.snapshot(), stats=batch.stats())
            _save(tmp_path, "floating-shape-response", record)


@pytest.mark.tyx
@pytest.mark.parametrize("n,variant", ((9, None), (16, None), (16, "body15"), (16, "both")))
def test_f08_nonaxis_full_tree_and_last_physical_body(n, variant, tyx_mode, tmp_path):
    rows = _window(_item(_reference().case("F08", n=n, variant=variant)), tyx_mode, tmp_path)
    assert rows[0]["receipt"].law_wrench_fp64.shape == (n + 1, 6)
    assert rows[0]["receipt"].contact_generalized.shape == (n + 6,)
    if variant == "both":
        _close(rows[0]["receipt"].law_wrench_fp64[0, :3], [0, 2, 0])
        _close(rows[0]["receipt"].law_wrench_fp64[n, :3], [2, 0, 0])


@pytest.mark.tyx
def test_f08_shared_revolute_ancestor_keeps_complete_cross_coupling(tyx_mode, tmp_path):
    _window(_item(_reference().case("F08_branch")), tyx_mode, tmp_path)


@pytest.mark.tyx
def test_f09_loaded_separated_and_reentered_without_reset(tyx_mode, tmp_path):
    rows = _window(_item(_reference().case("F09")), tyx_mode, tmp_path)
    loaded = [float(r["receipt"].law_wrench_fp64[0, 2]) > 0 for r in rows]
    assert loaded[0] and not all(loaded)
    first_open = loaded.index(False)
    assert any(loaded[first_open + 1 :])
    assert [r["receipt"].accepted_state.tick for r in rows] == list(range(1, 41))
