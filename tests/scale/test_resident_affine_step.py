"""C5 T1: independent affine/KKT physical tests; authoring only, UNEXECUTED."""

import copy
import gzip
import importlib.util
import json
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from dataclasses import fields, is_dataclass, replace
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

import phymeo as ph
from phymeo.backends import tyx_world_batch as resident

H = 1 / 1024
ATOL = 5e-6
RTOL = 5e-5
WINDOW_ATOL = 2e-4
CONSTRAINT_ATOL = 2e-5


@lru_cache(maxsize=1)
def _ref():
    path = Path(__file__).resolve().parents[1] / "reference_newton/robot/resident_affine_reference.py"
    spec = importlib.util.spec_from_file_location("c5_independent_reference", path)
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


def _save(path, name, record):
    with gzip.open(path / (name + ".json.gz"), "wt", encoding="utf-8") as stream:
        json.dump(_json(record), stream, indent=2, allow_nan=False)


def _close(actual, expected, *, atol=ATOL, rtol=RTOL):
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)


def _load(path, data, *, reorder=False):
    path.mkdir(parents=True, exist_ok=True)
    source = path / "mechanism.urdf"
    source.write_text(_ref().urdf(data, reorder=reorder), encoding="utf-8")
    asset = ph.load_mimic_urdf(source, gravity=data["model"]["gravity"])
    return asset


def _permutations(asset, data):
    names = tuple(data["model"]["joint_names"][i] for i in _ref().coordinates(data["model"]))
    independent = tuple(names[i] for i in data["independent"])
    return ([names.index(name) for name in asset.tree.joint_names],
            [independent.index(name) for name in asset.joint_names])


def _world(asset, data, *, s=None, u=None, kp=None, kd=None, cap=None, delay=0, iterations=512):
    _, p = _permutations(asset, data)
    k = len(p)
    def vector(value, default):
        return np.full(k, default) if value is None else np.asarray(value, float)[p]
    initial = ph.RobotState(np.asarray(data["s"] if s is None else s)[p], np.asarray(data["u"] if u is None else u)[p])
    return ph.AffineRobotTree(asset, initial, kp=vector(kp, 0), kd=vector(kd, 0),
                              effort_limit=vector(cap, 16), delay_steps=delay,
                              control_max_iterations=iterations)


def _action(world, asset, data, *, target=None, rate=None, ff=None, effort=None, force=None, torque=None):
    f, p = _permutations(asset, data)
    active = _ref().coordinates(data["model"])
    def vector(value, fallback):
        return np.asarray(fallback if value is None else value, float)[p]
    applied = data["effort"] if effort is None else np.asarray(effort, float)
    forces = data["force"] if force is None else np.asarray(force, float)
    torques = data["torque"] if torque is None else np.asarray(torque, float)
    # A05T applies no body wrench: its independent fixed-tool oracle retains all
    # parts, while the public load has one row per actual folded physical group.
    loads = ph.RobotInputs(np.asarray(applied)[f], np.asarray(forces)[list(active)][f], np.asarray(torques)[list(active)][f])
    return world.inputs(vector(target, data["s"]), vector(rate, np.zeros(len(p))),
                        feedforward=vector(ff, data["motor"]), loads=loads)


def _groups(data, truth):
    m, active = data["model"], _ref().coordinates(data["model"])
    counts = len(active)
    owner, masses = {}, np.zeros(counts)
    com = np.zeros((counts, 3))
    J = np.zeros((counts, 3, counts))
    W = truth["angular_jacobian"][list(active)]
    for body in range(len(m["mass"])):
        parent = body
        while parent not in active:
            parent = int(m["parents"][parent])
        group = active.index(parent)
        owner[body] = group
        masses[group] += m["mass"][body]
        com[group] += m["mass"][body]*truth["com_position"][body]
        J[group] += m["mass"][body]*truth["linear_jacobian"][body]
    return com/masses[:, None], J/masses[:, None, None], W


def _check_state(asset, data, checkpoint):
    f, p = _permutations(asset, data)
    inv_f, inv_p = np.argsort(f), np.argsort(p)
    s, u = np.asarray(checkpoint.position)[inv_p], np.asarray(checkpoint.velocity)[inv_p]
    _close(np.asarray(checkpoint.full_position)[inv_f], data["P"] @ s+data["c"], atol=CONSTRAINT_ATOL)
    _close(np.asarray(checkpoint.full_velocity)[inv_f], data["P"] @ u, atol=CONSTRAINT_ATOL)
    _close(data["C"] @ np.asarray(checkpoint.full_position)[inv_f], [row[3] for row in data["rows"]], atol=CONSTRAINT_ATOL)
    _close(data["C"] @ np.asarray(checkpoint.full_velocity)[inv_f], np.zeros(len(data["rows"])), atol=CONSTRAINT_ATOL)
    return s, u


def _check_readback(batch, key, asset, data, entry, after, h=H):
    # All ordinary fixtures have the frozen source joint order. Reorder control
    # compares by name separately; no product output defines the oracle.
    names = tuple(data["model"]["joint_names"][i] for i in _ref().coordinates(data["model"]))
    assert asset.tree.joint_names == names
    s, u = _check_state(asset, data, entry)
    next_s, next_u = _check_state(asset, data, after)
    actual_motor = np.asarray(after.applied_effort)
    expected = _ref().kkt(data, s, u, actual_motor)
    diag = batch.tree_diagnostics([key])[key]
    read = batch.tree_kinematics([key])[key]
    assert diag is not None and (diag.entry_tick, diag.exit_tick, diag.dt_s) == (entry.tick, after.tick, h)
    for name, truth in (("entry_position", s), ("entry_velocity", u), ("full_entry_position", expected["q"]),
                        ("full_entry_velocity", expected["v"]), ("mass", expected["full"]["mass"]),
                        ("bias", expected["full"]["bias"]), ("external", expected["external_full"]),
                        ("reduced_mass", expected["mass"]), ("reduced_bias", expected["bias"]),
                        ("reduced_external", expected["external"]), ("drive", actual_motor),
                        ("acceleration", expected["acceleration"]), ("full_acceleration", expected["full_acceleration"]),
                        ("constraint_effort_full", expected["reaction"]), ("row_multipliers", expected["multipliers"])):
        _close(getattr(diag, name), truth)
        assert not getattr(diag, name).flags.writeable
    _, J, W = _groups(data, expected["full"])
    for name, truth in (("linear_jacobian", J), ("angular_jacobian", W),
                        ("reduced_linear_jacobian", J @ data["P"]), ("reduced_angular_jacobian", W @ data["P"])):
        _close(getattr(diag, name), truth)
    _close(after.full_applied_effort, expected["motor_full"])
    _close(after.generalized_applied, actual_motor+expected["external"])
    _close(after.full_generalized_applied, expected["motor_full"]+expected["external_full"])
    _close(next_u, u+h*expected["acceleration"])
    _close(next_s, s+h*next_u)
    _close(diag.row_powers_w, np.zeros(len(data["rows"])), atol=CONSTRAINT_ATOL)
    _close(diag.constraint_power_w, 0., atol=CONSTRAINT_ATOL)
    assert diag.equation_residual <= 2e-5
    assert np.isfinite(diag.control_residual)
    assert read.tick == after.tick and read.joint_names == asset.joint_names and read.full_joint_names == names
    _close(read.position, after.full_position)
    _close(read.velocity, after.full_velocity)
    _close(read.independent_position, after.position)
    _close(read.independent_velocity, after.velocity)
    final = _ref().oracle(data["model"], np.asarray(after.full_position), np.asarray(after.full_velocity))
    com, _, _ = _groups(data, final)
    _close(read.com_position_m, com)
    for body, name in enumerate(data["model"]["names"]):
        site = asset.tree.sites[name]
        group = site.link_index
        R = read.link_rotation[group]
        authored_R = R @ _ref().quaternion_matrix(site.local_orientation_xyzw)
        authored_p = read.link_position_m[group] + R @ site.local_position_m
        _close(authored_p, final["position"][body])
        _close(authored_R, final["rotation"][body])
    _close(batch.summary()[key]["kinetic_energy_j"], .5*np.asarray(after.full_velocity) @ final["mass"] @ after.full_velocity)
    return diag, read, expected


@pytest.mark.tyx
def test_a01_public_affine_tree_keeps_follower_inertia_and_real_reaction(tyx_mode, tmp_path):
    data = _ref().case("A01")
    asset = _load(tmp_path, data)
    assert asset.tree.model.dof_count == 2 and len(asset.joint_names) == 1
    np.testing.assert_array_equal(asset.matrix, [[1], [2]])
    np.testing.assert_array_equal(asset.offset_vector, [0, 1/8])
    existing = asset.full_state(ph.RobotState(np.zeros(1), np.zeros(1)))
    np.testing.assert_array_equal(existing.q, [0, 1/8])
    _save(tmp_path, "a01-public-input", dict(input=data, h_s=H, full_initial=existing))
    # Valid old HOST asset established before touching the missing public API.
    world = _world(asset, data)
    with resident.TyxWorldBatch([(7, world)], tyx_mode, cache_root=tmp_path) as batch:
        batch.set_controls({7: _action(world, asset, data)})
        before = batch.snapshot()[7]
        assert batch.step(H) == {7: True}
        after = batch.snapshot()[7]
        diag, read, expected = _check_readback(batch, 7, asset, data, before, after)
        _save(tmp_path, "a01-public-result", dict(before=before, after=after, diagnostic=diag, kinematics=read, oracle=expected, stats=batch.stats()))
        _close(after.velocity, [H/2])
        _close(after.position, [H*H/2])
        _close(diag.mass, [[2, 0], [0, 1]])
        _close(diag.reduced_mass, [[6]])
        _close(diag.row_multipliers, [1])
        _close(diag.constraint_effort_full, [-2, 1])


EXACT = {
    "A01F": ([1], [-1]), "A02": ([1], [2, -2]), "A03": ([1], [2.5, -1]),
    "A04": ([.75], [-.125]), "A05": ([64/741, 161/741], None),
    "A05G": ([-1360/741, 469/741], None), "A05T": ([1], None),
}


@pytest.mark.tyx
@pytest.mark.parametrize("name", tuple(EXACT))
def test_a01_a05_independent_kkt_exact_first_steps(name, tyx_mode, tmp_path):
    data = _ref().case(name)
    expected = _ref().kkt(data, data["s"], data["u"])
    np.testing.assert_allclose(expected["acceleration"], EXACT[name][0], atol=2e-13, rtol=0)
    if EXACT[name][1] is not None:
        np.testing.assert_allclose(expected["multipliers"], EXACT[name][1], atol=2e-13, rtol=0)
    asset = _load(tmp_path, data)
    world = _world(asset, data)
    with resident.TyxWorldBatch([(7, world)], tyx_mode, cache_root=tmp_path) as batch:
        batch.set_controls({7: _action(world, asset, data)})
        before = batch.snapshot()[7]
        assert batch.step(H) == {7: True}
        after = batch.snapshot()[7]
        diag, read, actual_oracle = _check_readback(batch, 7, asset, data, before, after)
        _save(tmp_path, name+"-exact", dict(input=data, h_s=H, before=before, after=after,
              diagnostic=diag, kinematics=read, independent_kkt=expected, actual_entry_oracle=actual_oracle, stats=batch.stats()))


@pytest.mark.tyx
@pytest.mark.parametrize("name", ("A01", "A02", "A03", "A04", "A05", "A05G", "A05T", "N9", "N16"))
def test_a01_a07_forty_accepted_steps_without_reseeding(name, tyx_mode, tmp_path):
    data = _ref().case(name)
    asset = _load(tmp_path, data)
    world = _world(asset, data)
    record = dict(input=data, h_s=H, requested_steps=40, trace=[])
    with resident.TyxWorldBatch([(7, world)], tyx_mode, cache_root=tmp_path) as batch:
        batch.set_controls({7: _action(world, asset, data)})
        previous = batch.snapshot()[7]
        s, u = np.asarray(previous.position), np.asarray(previous.velocity)
        record["initial"] = previous
        try:
            for tick in range(1, 41):
                accepted = batch.step(H)
                actual = batch.snapshot()[7]
                row = dict(tick=tick, accepted=accepted, actual=actual)
                record["trace"].append(row)
                assert accepted == {7: True} and actual.tick == tick
                diag, read, oracle = _check_readback(batch, 7, asset, data, previous, actual)
                s, u, _ = _ref().advance(data, s, u)
                row.update(diagnostic=diag, kinematics=read, oracle=oracle,
                           independent_window_s=s.copy(), independent_window_u=u.copy(), stats=batch.stats())
                _close(actual.position, s, atol=WINDOW_ATOL)
                _close(actual.velocity, u, atol=WINDOW_ATOL)
                if name == "A01":
                    _close(actual.position, [H*H*tick*(tick+1)/4])
                    _close(actual.velocity, [tick*H/2])
                previous = actual
            assert {row["stats"]["current_buffer"] for row in record["trace"]} == {0, 1}
        finally:
            record["final_stats"] = batch.stats()
            _save(tmp_path, name+"-window", record)


@pytest.mark.tyx
@pytest.mark.parametrize("name", ("A04", "A05G"))
@pytest.mark.parametrize("placement", ("cycle", "large"))
def test_a04_a05_covariant_installation_and_current_com_load(name, placement, tyx_mode, tmp_path):
    data = _ref().transformed(_ref().case(name), placement)
    asset = _load(tmp_path, data)
    world = _world(asset, data)
    with resident.TyxWorldBatch([(7, world)], tyx_mode, cache_root=tmp_path) as batch:
        batch.set_controls({7: _action(world, asset, data)})
        before = batch.snapshot()[7]
        assert batch.step(H) == {7: True}
        after = batch.snapshot()[7]
        diag, read, oracle = _check_readback(batch, 7, asset, data, before, after)
        _save(tmp_path, "covariance", dict(input=data, before=before, after=after, diagnostic=diag, kinematics=read, oracle=oracle))


@pytest.mark.tyx
def test_a02_source_reorder_keeps_named_axes_and_full_bodies(tyx_mode, tmp_path):
    data = _ref().case("A02")
    snapshots = []
    for reorder in (False, True):
        asset = _load(tmp_path / str(reorder), data, reorder=reorder)
        world = _world(asset, data)
        with resident.TyxWorldBatch([(7, world)], tyx_mode, cache_root=tmp_path) as batch:
            batch.set_controls({7: _action(world, asset, data)})
            assert batch.step(H) == {7: True}
            state = batch.snapshot()[7]
            _check_state(asset, data, state)
            f, p = _permutations(asset, data)
            snapshots.append((np.asarray(state.full_position)[np.argsort(f)], np.asarray(state.full_velocity)[np.argsort(f)], np.asarray(state.applied_effort)[np.argsort(p)]))
    for left, right in zip(*snapshots, strict=True):
        _close(left, right)
    _save(tmp_path, "named-reorder", dict(input=data, named_states=snapshots))


@pytest.mark.tyx
def test_a03_chain_and_flattened_graph_keep_motion_but_not_row_forces(tyx_mode, tmp_path):
    outputs = []
    for name in ("A03", "A03_flat"):
        data = _ref().case(name)
        asset = _load(tmp_path / name, data)
        world = _world(asset, data)
        with resident.TyxWorldBatch([(7, world)], tyx_mode, cache_root=tmp_path) as batch:
            batch.set_controls({7: _action(world, asset, data)})
            before = batch.snapshot()[7]
            assert batch.step(H) == {7: True}
            after = batch.snapshot()[7]
            diag, _, _ = _check_readback(batch, 7, asset, data, before, after)
            outputs.append(dict(state=after, row=diag.row_multipliers.copy(), reaction=diag.constraint_effort_full.copy()))
    _close(outputs[0]["state"].full_position, outputs[1]["state"].full_position)
    _close(outputs[0]["reaction"], outputs[1]["reaction"])
    _close(outputs[0]["row"], [2.5, -1])
    _close(outputs[1]["row"], [2, -1])
    _save(tmp_path, "chain-versus-flatten", outputs)


@pytest.mark.tyx
@pytest.mark.parametrize("cap", (1., 16.))
def test_a06_scalar_implicit_pd_cap_and_exact_nonsaturated_drive(cap, tyx_mode, tmp_path):
    data = _ref().case("A01")
    data["motor"][:] = 0
    asset = _load(tmp_path, data)
    world = _world(asset, data, kp=[64], kd=[8], cap=[cap])
    with resident.TyxWorldBatch([(7, world)], tyx_mode, cache_root=tmp_path) as batch:
        batch.set_controls({7: _action(world, asset, data, target=[1/8], ff=[0])})
        before = batch.snapshot()[7]
        assert batch.step(H) == {7: True}
        after = batch.snapshot()[7]
        expected = min(cap, 8/(1+129/98304))
        _close(after.applied_effort, [expected])
        _check_readback(batch, 7, asset, data, before, after)
        raw_expected = np.clip(64*(1/8-np.asarray(before.position)-H*np.asarray(after.velocity))-8*np.asarray(after.velocity), -cap, cap)
        assert np.all(abs(np.asarray(after.applied_effort)-raw_expected) <= 5e-5*np.maximum(1, np.maximum(abs(after.applied_effort), abs(raw_expected))))
        _save(tmp_path, "pd", dict(before=before, after=after, expected=expected, diagnostic=batch.tree_diagnostics()))


@pytest.mark.tyx
@pytest.mark.parametrize("cancel", (False, True))
def test_a06_wide_cap_and_large_terms_do_not_hide_small_effort(cancel, tyx_mode, tmp_path):
    data = _ref().case("A01")
    data["rating"][:] = 1e6
    data["s"][:] = .5 if cancel else 0
    data["motor"][:] = 512+1/128 if cancel else 1/128
    asset = _load(tmp_path, data)
    world = _world(asset, data, kp=[1024 if cancel else 0], cap=[1e6])
    with resident.TyxWorldBatch([(7, world)], tyx_mode, cache_root=tmp_path) as batch:
        batch.set_controls({7: _action(world, asset, data, target=[0])})
        before = batch.snapshot()[7]
        assert batch.step(H) == {7: True}
        after = batch.snapshot()[7]
        expected = (1/128)/(1+H/6) if cancel else 1/128
        _close(after.applied_effort, [expected])
        raw = np.clip((1024 if cancel else 0)*(0-np.asarray(before.position)-H*np.asarray(after.velocity))+data["motor"], -1e6, 1e6)
        assert np.all(abs(np.asarray(after.applied_effort)-raw) <= 5e-5*np.maximum(1, np.maximum(abs(after.applied_effort), abs(raw))))
        _check_readback(batch, 7, asset, data, before, after)
        _save(tmp_path, "small-effort", dict(input=data, before=before, after=after, original_clipped=raw))


@pytest.mark.tyx
@pytest.mark.parametrize("delay", (0, 1))
def test_a06_delay_only_drive_triplet_external_load_is_current(delay, tyx_mode, tmp_path):
    data = _ref().case("A01F")
    data["motor"][:] = 3
    asset = _load(tmp_path, data)
    world = _world(asset, data, delay=delay)
    with resident.TyxWorldBatch([(7, world)], tyx_mode, cache_root=tmp_path) as batch:
        first_action = _action(world, asset, data)
        batch.set_controls({7: first_action})
        before = batch.snapshot()[7]
        assert batch.step(H) == {7: True}
        first = batch.snapshot()[7]
        _close(first.applied_effort, [0 if delay else 3])
        _close(first.velocity, [H*(1 if delay else 1.5)])
        assert first.history == (((0., 0., 3.),) if delay else ())
        _check_readback(batch, 7, asset, data, before, first)
        data2 = copy.deepcopy(data)
        data2["motor"][:] = -3
        data2["force"][1, 0] = -3
        batch.set_controls({7: _action(world, asset, data2)})
        assert batch.step(H) == {7: True}
        second = batch.snapshot()[7]
        _close(second.applied_effort, [3 if delay else -3])
        _check_readback(batch, 7, asset, data2, first, second)
        _close(np.asarray(second.control[0])[:, 2], [-3])
        _close(np.asarray(second.control[1])[:, 1:4], data2["force"])
        _save(tmp_path, "delay", dict(delay=delay, initial=before, first=first, second=second, diagnostic=batch.tree_diagnostics()))


@pytest.mark.tyx
def test_a06_one_sweep_rejects_handderived_cross_residual_then_full_budget_solves(tyx_mode, tmp_path):
    data = _ref().case("A05")
    data["effort"][:] = 0
    tau0 = 741/1061
    tau1 = 741*997/(1061*902)
    assert 64*tau1/741 > .06
    exact = np.array([838*741, 997*741])/(1061*902-4096)
    outputs = []
    for budget in (1, 512):
        asset = _load(tmp_path / str(budget), data)
        world = _world(asset, data, kd=[1024, 1024], iterations=budget)
        with resident.TyxWorldBatch([(7, world)], tyx_mode, cache_root=tmp_path) as batch:
            batch.set_controls({7: _action(world, asset, data, rate=[H, H], ff=[0, 0])})
            before = batch.snapshot()[7]
            accepted = batch.step(H)
            after = batch.snapshot()[7]
            outputs.append(dict(budget=budget, before=before, after=after, accepted=accepted, stats=batch.stats(), diagnostic=batch.tree_diagnostics()))
            if budget == 1:
                assert accepted == {7: False} and after == before
                assert batch.tree_diagnostics() == {7: None}
            else:
                assert accepted == {7: True}
                _close(after.applied_effort, exact)
                _check_readback(batch, 7, asset, data, before, after)
    _save(tmp_path, "budget", dict(input=data, first_sweep=[tau0, tau1], first_row_residual=64*tau1/741, exact=exact, outputs=outputs))


@pytest.mark.parametrize("fault", ("cycle", "duplicate_follower", "unknown", "type", "zero", "nonfinite", "empty_interval", "duplicate_name"))
def test_a07_invalid_host_graph_rejected_before_device(fault, tmp_path):
    data = _ref().case("A03")
    root = ET.fromstring(_ref().urdf(data))
    joints = {joint.get("name"): joint for joint in root.findall("joint")}
    if fault == "cycle":
        ET.SubElement(joints["joint0"], "mimic", joint="joint2", multiplier="1", offset="0")
    elif fault == "duplicate_follower":
        ET.SubElement(joints["joint1"], "mimic", joint="joint0", multiplier="2", offset="0")
    elif fault == "unknown":
        joints["joint1"].find("mimic").set("joint", "missing")
    elif fault == "type":
        joints["joint1"].set("type", "revolute")
    elif fault == "zero":
        joints["joint1"].find("mimic").set("multiplier", "0")
    elif fault == "nonfinite":
        joints["joint1"].find("mimic").set("offset", "nan")
    elif fault == "empty_interval":
        joints["joint1"].find("limit").set("lower", "100")
        joints["joint1"].find("limit").set("upper", "101")
    else:
        joints["joint2"].set("name", "joint1")
    path = tmp_path / "invalid.urdf"
    path.write_text(ET.tostring(root, encoding="unicode"), encoding="utf-8")
    with pytest.raises((ValueError, TypeError)):
        ph.load_mimic_urdf(path, gravity=(0, 0, 0))


@pytest.mark.parametrize("fault", ("N17", "gain_underflow", "gain_overflow", "delay2", "budget0"))
def test_a07_static_device_domain_rejects_without_silently_changing_graph(fault, tmp_path):
    data = _ref().case("N17" if fault == "N17" else "A01")
    if fault.startswith("gain"):
        a = 1e-50 if fault == "gain_underflow" else 1e40
        data["rows"] = ((0, 1, a, 0.),)
        data["P"] = np.array([[1.], [a]])
        data["c"][:] = 0
    asset = _load(tmp_path, data)
    with pytest.raises((ValueError, TypeError), match="16|count|capacity|represent|FP32|gain|multiplier|delay|iteration|budget"):
        world = _world(asset, data, delay=2 if fault == "delay2" else 0, iterations=0 if fault == "budget0" else 512)
        # Static compilation must fail before needing a TYX runtime.
        with resident.TyxWorldBatch([(7, world)], "interpret", cache_root=tmp_path):
            pass


@pytest.mark.tyx
@pytest.mark.parametrize("direction", (-1, 1))
def test_a07_position_intersection_candidate_reject_retains_all_rows(direction, tyx_mode, tmp_path):
    data = _ref().case("A01")
    data["lower"], data["upper"] = np.array([-1., 0.]), np.array([1., 1.])
    data["s"][:] = -1/16 if direction < 0 else 7/16
    data["u"][:] = direction/8
    asset = _load(tmp_path, data)
    _close(asset.lower, [-1/16])
    _close(asset.upper, [7/16])
    world = _world(asset, data, delay=1)
    with resident.TyxWorldBatch([(7, world)], tyx_mode, cache_root=tmp_path) as batch:
        batch.set_controls({7: _action(world, asset, data)})
        before = batch.snapshot()[7]
        fk = batch.tree_kinematics()[7]
        assert batch.step(H) == {7: False}
        assert batch.snapshot()[7] == before and batch.tree_diagnostics() == {7: None}
        np.testing.assert_array_equal(batch.tree_kinematics()[7].link_position_m, fk.link_position_m)
        # Same accepted entry can proceed only with a valid opposite external load;
        # this test intentionally does not overwrite the physical entry velocity.
        _save(tmp_path, "domain-rejection", dict(input=data, before=before, after=batch.snapshot()[7], stats=batch.stats()))


@pytest.mark.tyx
@pytest.mark.parametrize("warm_steps", (0, 1))
@pytest.mark.parametrize("failure", ("partial_upload", "step_status", "restore_sync"))
def test_a07_mixed_bank_failure_retry_is_atomic(warm_steps, failure, tyx_mode, tmp_path, monkeypatch):
    data = _ref().case("A03")
    asset = _load(tmp_path, data)
    world = _world(asset, data, delay=1)
    worlds = [(7, world), (23, resident.FreeSphere(2, 1/16, gravity=(0, 0, 0)))]
    actions = {7: _action(world, asset, data), 23: (1., 0., 0.)}
    with resident.TyxWorldBatch(worlds, tyx_mode, cache_root=tmp_path / "trial") as batch, resident.TyxWorldBatch(worlds, tyx_mode, cache_root=tmp_path / "control") as control:
        batch.set_controls(actions)
        control.set_controls(actions)
        for _ in range(warm_steps):
            assert batch.step(H) == control.step(H) == {7: True, 23: True}
        restore_to = batch.snapshot()
        if failure == "restore_sync":
            assert batch.step(H) == control.step(H) == {7: True, 23: True}
        else:
            actions = {7: _action(world, asset, data, ff=[-3]), 23: (0., 1., 0.)}
            batch.set_controls(actions)
            control.set_controls(actions)
        before, fk, stats = batch.snapshot(), batch.tree_kinematics()[7], batch.stats()
        calls = []
        with monkeypatch.context() as patch:
            if failure == "partial_upload":
                original = batch._upload
                def injected(buffer, value, category):
                    original(buffer, value, category)
                    calls.append(category)
                    if len(calls) == 2:
                        raise RuntimeError("C5 injected partial upload")
                patch.setattr(batch, "_upload", injected)
            elif failure == "step_status":
                original = batch._read_status
                def injected(*args, **kwargs):
                    original(*args, **kwargs)
                    raise RuntimeError("C5 injected status read")
                patch.setattr(batch, "_read_status", injected)
            else:
                original = batch._sync
                def injected(*args, **kwargs):
                    original(*args, **kwargs)
                    raise RuntimeError("C5 injected restore sync")
                patch.setattr(batch, "_sync", injected)
            with pytest.raises(RuntimeError, match="C5 injected"):
                batch.restore(restore_to) if failure == "restore_sync" else batch.step(H)
        assert batch.snapshot() == before
        assert batch.stats()["state_epoch"] == stats["state_epoch"]
        assert batch.stats()["current_buffer"] == stats["current_buffer"]
        np.testing.assert_array_equal(batch.tree_kinematics()[7].link_position_m, fk.link_position_m)
        assert batch.tree_diagnostics() == {7: None}
        if failure == "restore_sync":
            batch.restore(restore_to)
            control.restore(restore_to)
        assert batch.step(H) == control.step(H) == {7: True, 23: True}
        assert batch.snapshot() == control.snapshot()
        _save(tmp_path, "atomic-retry", dict(failure=failure, warm_steps=warm_steps, input=data, commands=actions,
              before=before, after=batch.snapshot(), uploads=calls, stats_before=stats, stats_after=batch.stats()))


@pytest.mark.tyx
@pytest.mark.parametrize("warm_steps", (0, 1))
def test_a07_restore_rejects_graph_full_state_and_history_mismatch(warm_steps, tyx_mode, tmp_path):
    data = _ref().case("A03")
    asset = _load(tmp_path, data)
    world = _world(asset, data, delay=1)
    foreign_data = _ref().case("A03_flat")
    foreign_asset = _load(tmp_path / "foreign", foreign_data)
    with resident.TyxWorldBatch([(7, world)], tyx_mode, cache_root=tmp_path) as batch, resident.TyxWorldBatch([(7, _world(foreign_asset, foreign_data, delay=1))], tyx_mode, cache_root=tmp_path) as foreign:
        batch.set_controls({7: _action(world, asset, data)})
        for _ in range(warm_steps):
            assert batch.step(H) == {7: True}
        before = batch.snapshot()[7]
        bad = (foreign.snapshot()[7], replace(before, full_position=(0., 0., 0.)), replace(before, history=((0., 0.),)))
        epoch = batch.stats()["state_epoch"]
        for checkpoint in bad:
            with pytest.raises((ValueError, TypeError)):
                batch.restore({7: checkpoint})
            assert batch.snapshot()[7] == before and batch.stats()["state_epoch"] == epoch
        batch.restore({7: before})
        assert batch.snapshot()[7] == before and batch.stats()["state_epoch"] == epoch+1
        assert batch.tree_diagnostics() == {7: None}
        _save(tmp_path, "restore-schema", dict(before=before, rejected=bad, after=batch.snapshot()[7], stats=batch.stats()))


@pytest.mark.tyx
def test_a07_owned_model_input_and_readbacks_and_old_scalar_reader_rejection(tyx_mode, tmp_path):
    data = _ref().case("A01")
    asset = _load(tmp_path, data)
    world = _world(asset, data)
    ff = np.array([3.])
    load = ph.RobotInputs.zeros(2)
    with resident.TyxWorldBatch([(7, world)], tyx_mode, cache_root=tmp_path) as batch:
        action = world.inputs(np.zeros(1), feedforward=ff, loads=load)
        batch.set_controls({7: action})
        ff[:] = 100
        load.effort[:] = 100
        asset.tree.model.mass.setflags(write=True)
        asset.tree.model.mass[:] = 100
        before = batch.snapshot()[7]
        assert batch.step(H) == {7: True}
        after = batch.snapshot()[7]
        _close(after.velocity, [H/2])
        _close(before.control[0], [[0, 0, 3]])
        assert isinstance(before.control, tuple) and isinstance(before.control[0], tuple)
        read, diag = batch.tree_kinematics()[7], batch.tree_diagnostics()[7]
        for value in (read.position, read.velocity, read.independent_position, read.link_rotation,
                      diag.mass, diag.reduced_mass, diag.constraint_effort_full, diag.row_multipliers):
            with pytest.raises(ValueError):
                value.setflags(write=True)
        with pytest.raises((ValueError, TypeError), match="tree|affine|unsupported|scalar"):
            batch.prepare_readback([7])
        _save(tmp_path, "owned", dict(before=before, after=after, read=read, diagnostic=diag, stats=batch.stats()))
