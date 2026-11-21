"""Original RR T4 resident rows, without integration or an area-contact solve."""

from types import SimpleNamespace

import mujoco as mj
import numpy as np
import pytest
import test_mujoco_tyx_constraints as mixed
import test_mujoco_tyx_normal_contacts as normal

from experiments.area_pressure.pusht_entry_rows import assemble_entry, load_entries
from phymeo.backends.mujoco_area_pressure import compile_area_native_rows
from phymeo.backends.mujoco_tyx._constraint_device import TyxConstraintDevice
from phymeo.training.manipulation_tasks import manipulation_scenes

LABELS = ("initial", "first_load", "peak", "last_push_entry", "withdrawn")
SOURCE_FIELDS = (
    "mass",
    "bias",
    "passive",
    "applied",
    "actuator",
    "motion",
    "rotation",
    "jacobian",
    "normal_rows",
    "normal_jacobian",
    "efc_ids",
    "efc_jacobian",
    "contact_map",
    "contact_jacobian",
    "constraint_counts",
)
RESULT_FIELDS = (
    "status",
    "efc_data",
    "physical_acc",
    "mixed_constraint",
    "total_applied",
    "solver_diagnostics",
)


def _fixture(labels=LABELS):
    scene = next(scene for scene in manipulation_scenes() if scene.key == "pusht-planar")
    model = mj.MjModel.from_xml_string(scene.mjcf)
    saved = load_entries()
    return model, [saved[label] for label in labels]


def _make_device(model, entries, constraint_capacity=256):
    plan = compile_area_native_rows(mj, model, (("tool", "object"),))
    assert (plan.nq, plan.nv, plan.nu, plan.nlimit, plan.nfriction) == (9, 8, 2, 2, 0)
    pusher = model.geom("pusher").id
    excluded = {frozenset((pusher, model.geom(name).id)) for name in ("object-0", "object-1")}
    retained = {frozenset(pair[:2]) for pair in plan.pair_ids[: plan.npair]}
    assert plan.npair == len(retained) == 9 and not retained.intersection(excluded)
    support = model.geom("support").id
    assert all(
        frozenset((support, model.geom(name).id)) in retained for name in ("object-0", "object-1")
    )
    q = np.asarray([entry["q"] for entry in entries], np.float32)
    v = np.asarray([entry["v"] for entry in entries], np.float32)
    time = np.zeros((len(entries), 2), np.float32)
    for lane, entry in enumerate(entries):
        time[lane, 0] = entry["time"]
        time[lane, 1] = float(time[lane, 0]) - entry["time"]
    controls = np.asarray([entry["ctrl"] for entry in entries], np.float32)
    applied = np.asarray([entry["qfrc_applied"] for entry in entries], np.float32)
    body_force = np.asarray([entry["xfrc_applied"] for entry in entries], np.float32)
    device = TyxConstraintDevice(
        plan,
        "cuda",
        None,
        q,
        v,
        time,
        model.nbody,
        model.opt.timestep,
        (64, constraint_capacity),
    )
    return device, (controls, applied, body_force)


def _read(device, names):
    with device.tx.use_runtime(device.runtime):
        device.runtime.sync()
        return {name: device._get(getattr(device, name)) for name in names}


def _banks(device):
    with device.tx.use_runtime(device.runtime):
        device.runtime.sync()
        return device.active, tuple(
            device._get(bank) for field in (device.q, device.v, device.time) for bank in field
        )


def _same_banks(device, before):
    active, values = _banks(device)
    assert active == before[0]
    for value, wanted in zip(values, before[1], strict=True):
        np.testing.assert_array_equal(value, wanted)


def _forbidden(*args, **kwargs):
    raise AssertionError("resident evaluation repeated CPU forwarding, assembly, or integration")


def _check_source(model, device, lane, source, rows, oracle):
    """Match geometry before mapping compact native IDs to original CPU rows."""
    data, plan = oracle["data"], device.plan
    for name, expected in (
        ("mass", oracle["mass"]),
        ("bias", data.qfrc_bias),
        ("passive", data.qfrc_passive),
        ("applied", oracle["base_load"] - data.qfrc_passive + data.qfrc_bias),
    ):
        np.testing.assert_allclose(source[name][lane], expected, rtol=4e-4, atol=5e-5, err_msg=name)
    nc, active, nr = map(int, source["constraint_counts"][lane])
    expected_contacts = []
    for contact in oracle["contacts"]:
        if contact["retained"]:
            ci = contact["contact_index"]
            expected_contacts.append(
                dict(
                    native_index=ci,
                    geometry_ids=contact["geometry_ids"],
                    position=contact["position"],
                    frame=contact["frame"],
                    dimension=int(data.contact[ci].dim),
                    active=contact["efc_address"] >= 0,
                )
            )
    assert nc == len(expected_contacts) and nr == len(oracle["retained_indices"])
    assert active == sum(contact["active"] for contact in expected_contacts)
    compact = source["contact_map"][lane, :nc]
    actual = []
    for slot, first, count in compact:
        pair, _feature, ga, gb = map(int, plan.slot_ids[slot])
        row = source["normal_rows"][lane, slot]
        assert row[0] == 1 and count == (4 if row[1] else 0)
        assert (first >= 0) == bool(count)
        actual.append(
            SimpleNamespace(
                geometry_ids=(ga, gb),
                position=row[9:12],
                frame=row[12:21].reshape(3, 3),
                dimension=int(plan.pair_condim[pair]),
                active=bool(row[1]),
            )
        )
    matched = mixed.match_contacts(actual, expected_contacts)
    for contact, expected in zip(actual, matched, strict=True):
        assert contact.dimension == expected["dimension"] == 3
        assert contact.active == expected["active"]
        np.testing.assert_allclose(contact.frame, expected["frame"], rtol=3e-5, atol=4e-5)
    source_rows = []
    joint_rows = {}
    for owner in oracle["owners"]:
        if owner["kind"] == "limit":
            joint_rows[(owner["joint_id"], owner["side"])] = owner["source_row"]
    for row_index, ids in enumerate(source["efc_ids"][lane, :nr]):
        kind, owner, side, slot, dof = map(int, ids)
        if kind == 3:
            index = joint_rows[(owner, side)]
            assert slot == -1 and dof == model.jnt_dofadr[owner]
        else:
            assert kind == 6 and dof == -1 and slot == compact[owner, 0]
            original_ci = matched[owner]["native_index"]
            index = int(data.contact[original_ci].efc_address) + side
            assert 0 <= side < 4 and data.efc_id[index] == original_ci
        assert data.efc_type[index] == kind
        source_rows.append(index)
        expected = np.array(
            [
                data.efc_pos[index],
                data.efc_margin[index],
                data.efc_vel[index],
                data.efc_aref[index],
                data.efc_R[index],
                data.efc_D[index],
                data.efc_KBIP[index, 2],
                data.efc_diagApprox[index],
            ]
        )
        np.testing.assert_allclose(
            rows[row_index, [0, 1, 2, 3, 4, 5, 6, 13]], expected, **mixed.ROW_TOL
        )
        np.testing.assert_allclose(
            source["efc_jacobian"][lane, row_index],
            data.efc_J.reshape(-1, 8)[index],
            rtol=2e-4,
            atol=5e-5,
        )
        np.testing.assert_array_equal(rows[row_index, [7, 8, 11, 12]], [0, 0, 1, 0])
    assert len(set(source_rows)) == nr and set(source_rows) == set(oracle["retained_indices"])
    assert not np.any(rows[nr:])
    assert not np.any(source["efc_jacobian"][lane, nr:])
    return np.asarray(source_rows, dtype=int)


def _root(mass, load, jac, reg, aref):
    """Independent FP64 positive-part equation in eight physical accelerations."""

    def residual(a):
        force = np.maximum(0.0, (aref - jac @ a) / reg)
        return mass @ a - load - jac.T @ force, force

    a = np.linalg.solve(mass, load)
    for _ in range(80):
        value, force = residual(a)
        if np.linalg.norm(value, np.inf) <= 1e-11 * (1 + np.linalg.norm(load, np.inf)):
            return a, force
        active = force > 0
        derivative = mass + jac[active].T @ (jac[active] / reg[active, None])
        delta = np.linalg.solve(derivative, -value)
        for power in range(30):
            candidate = a + delta / 2**power
            if np.linalg.norm(residual(candidate)[0]) < np.linalg.norm(value):
                a = candidate
                break
        else:
            raise AssertionError("independent native positive-part root stalled")
    raise AssertionError("independent native positive-part root exceeded its budget")


def _check_solution(source, result, extra, oracles, row_order):
    np.testing.assert_array_equal(result["status"], 0)
    for lane, oracle in enumerate(oracles):
        nr = int(source["constraint_counts"][lane, 2])
        row = result["efc_data"][lane, :nr].astype(float)
        force, jac = row[:, 9], source["efc_jacobian"][lane, :nr].astype(float)
        mass = source["mass"][lane].astype(float)
        base = (
            source["applied"][lane].astype(float) + source["passive"][lane] - source["bias"][lane]
        )
        load = base + extra[lane]
        expected_a, expected_force = _root(mass, load, jac, row[:, 4], row[:, 3])
        actual_a = result["physical_acc"][lane].astype(float)
        assert np.all(force >= 0)
        np.testing.assert_allclose(actual_a, expected_a, **normal.ACC_TOL)
        np.testing.assert_allclose(force, expected_force, **normal.FORCE_TOL)
        np.testing.assert_allclose(
            result["mixed_constraint"][lane], jac.T @ force, rtol=5e-5, atol=5e-5
        )
        np.testing.assert_allclose(
            result["total_applied"][lane],
            source["applied"][lane] + extra[lane] + jac.T @ force,
            rtol=5e-5,
            atol=5e-5,
        )
        momentum = mass @ actual_a - load - jac.T @ force
        assert np.linalg.norm(momentum, np.inf) <= 4e-5 * (
            1 + np.linalg.norm(load, np.inf) + np.linalg.norm(jac.T @ force, np.inf)
        )
        gradient = jac @ actual_a + row[:, 4] * force - row[:, 3]
        violation = np.where(force > 0, abs(gradient), np.maximum(0.0, -gradient))
        scale = max(1.0, np.max(abs(row[:, 3]), initial=0), np.max(abs(jac @ actual_a), initial=0))
        assert np.max(violation, initial=0) <= 4e-5 * scale
        # Also solve the independently forwarded original-model coefficients.
        data, order = oracle["data"], row_order[lane]
        native_a, native_force = _root(
            oracle["mass"],
            oracle["base_load"] + extra[lane],
            data.efc_J.reshape(-1, 8)[order],
            data.efc_R[order],
            data.efc_aref[order],
        )
        np.testing.assert_allclose(actual_a, native_a, **normal.ACC_TOL)
        np.testing.assert_allclose(force, native_force, **normal.FORCE_TOL)
        diagnostic = result["solver_diagnostics"][lane]
        assert diagnostic[6] == 1 and diagnostic[4] == nr
        assert not np.any(result["efc_data"][lane, nr:, 9:11])


@pytest.mark.cuda
def test_five_original_rr_entries_reuse_live_native_rows_for_three_loads(monkeypatch):
    model, entries = _fixture()
    oracles = [assemble_entry(model, entry) for entry in entries]
    assert [len(oracle["retained_indices"]) for oracle in oracles] == [0, 32, 32, 28, 32]
    device, controls = _make_device(model, entries)
    extra_buffer = None
    try:
        before = _banks(device)
        with monkeypatch.context() as guard:
            guard.setattr(mj, "mj_step", _forbidden)
            guard.setattr(mj, "mj_forward", _forbidden)
            with device.prepare_native_rows(*controls) as evaluate:
                source = _read(device, SOURCE_FIELDS)
                prepared = _read(device, ("status", "efc_data"))
                np.testing.assert_array_equal(prepared["status"], 0)
                row_order = [
                    _check_source(model, device, lane, source, prepared["efc_data"][lane], oracle)
                    for lane, oracle in enumerate(oracles)
                ]
                for name in ("geometry", "mechanics", "contacts"):
                    guard.setattr(device, name, (_forbidden, _forbidden))
                guard.setattr(device, "_assemble_rows", _forbidden)
                guard.setattr(device, "solve", (_forbidden, _forbidden))
                with device.tx.use_runtime(device.runtime):
                    extra_buffer = device.tx.cuda_empty((5, 8), np.float32)
                loads = (
                    np.zeros((5, 8), np.float32),
                    np.tile([0.02, -0.01, 0.03, -0.04, -0.05, 0.001, -0.001, 0.002], (5, 1)).astype(
                        np.float32
                    ),
                    np.tile(
                        [-0.015, 0.025, -0.05, 0.02, 0.04, -0.001, 0.002, -0.003], (5, 1)
                    ).astype(np.float32),
                )
                baseline = None
                for index, extra in enumerate((*loads, loads[0])):
                    argument = None
                    if index in (1, 2):
                        with device.tx.use_runtime(device.runtime):
                            extra_buffer.copy_from(extra)
                        argument = extra_buffer
                    evaluate(argument)
                    result = _read(device, RESULT_FIELDS)
                    _check_solution(source, result, extra, oracles, row_order)
                    _same_banks(device, before)
                    for name, value in _read(device, SOURCE_FIELDS).items():
                        np.testing.assert_array_equal(value, source[name], err_msg=name)
                    np.testing.assert_array_equal(
                        result["efc_data"][:, :, [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13]],
                        prepared["efc_data"][:, :, [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13]],
                    )
                    if baseline is None:
                        baseline = result
                    elif index == 3:
                        for name in RESULT_FIELDS:
                            np.testing.assert_array_equal(result[name], baseline[name])
                    else:
                        assert np.max(abs(result["physical_acc"] - baseline["physical_acc"])) > 1e-3
        _same_banks(device, before)
    finally:
        if extra_buffer is not None:
            extra_buffer.free()
        device.close()


@pytest.mark.cuda
def test_original_rr_both_joint_limit_owners_survive_area_pair_exclusion():
    model, entries = _fixture(("withdrawn",))
    entry = dict(entries[0], q=entries[0]["q"].copy())
    entry["q"][:2] = np.asarray(
        [model.jnt_range[0, 1] + 0.002, model.jnt_range[1, 0] - 0.002], np.float32
    )
    oracle = assemble_entry(model, entry)
    assert {
        (row["joint_id"], row["side"]) for row in oracle["owners"] if row["kind"] == "limit"
    } == {(0, 1), (1, -1)}
    device, controls = _make_device(model, [entry])
    try:
        before = _banks(device)
        with device.prepare_native_rows(*controls) as evaluate:
            source = _read(device, SOURCE_FIELDS)
            prepared = _read(device, ("status", "efc_data"))
            np.testing.assert_array_equal(prepared["status"], 0)
            order = _check_source(model, device, 0, source, prepared["efc_data"][0], oracle)
            evaluate(None)
            _check_solution(
                source, _read(device, RESULT_FIELDS), np.zeros((1, 8)), [oracle], [order]
            )
        _same_banks(device, before)
    finally:
        device.close()


@pytest.mark.cuda
def test_rr_native_row_capacity_rejection_preserves_both_physical_banks():
    model, entries = _fixture(("peak",))
    device, controls = _make_device(model, entries, constraint_capacity=31)
    try:
        before = _banks(device)
        try:
            with device.prepare_native_rows(*controls) as evaluate:
                evaluate(None)
        except (ValueError, RuntimeError):
            pass
        result = _read(device, ("status", "constraint_counts"))
        assert result["status"].tolist() == [1]
        assert result["constraint_counts"][0, 2] == 32
        _same_banks(device, before)
    finally:
        device.close()


@pytest.mark.cuda
def test_rr_native_trial_integrates_floating_tail_once_without_publication(monkeypatch):
    """One bilateral trial load, one native FREE tail, and no accepted-state swap."""
    from experiments.area_pressure.pusht_entry_area import build_area

    model, entries = _fixture(("peak",))
    oracle = assemble_entry(model, entries[0])
    area = build_area(model, oracle["data"])
    assert len(area["points"]) == 237
    # Apply equal and opposite finite world forces at one real shared area point.
    # Both the two RR hinges and the FREE object receive generalized load.
    force = np.array([0.03, -0.04, 0.02])
    extra = ((area["jac_a"][0] - area["jac_b"][0]).T @ force)[None].astype(np.float32)
    assert np.any(extra[0, :2]) and np.any(extra[0, 2:])
    device, controls = _make_device(model, entries)
    extra_buffer = None

    def entry_bytes():
        with device.tx.use_runtime(device.runtime):
            device.runtime.sync()
            return {
                (name, bank): device._get(values[bank]).tobytes()
                for name, values in device.entry.items()
                for bank in (0, 1)
            }

    try:
        before, entry_before = _banks(device), entry_bytes()
        old, new = before[0], 1 - before[0]
        q0 = before[1][old][0].astype(float)
        v0 = before[1][2 + old][0].astype(float)
        old_clock = before[1][4 + old][0]
        with device.tx.use_runtime(device.runtime):
            extra_buffer = device.tx.cuda_empty(extra.shape, np.float32)
            extra_buffer.copy_from(extra)
        with monkeypatch.context() as guard:
            guard.setattr(mj, "mj_step", _forbidden)
            guard.setattr(mj, "mj_forward", _forbidden)
            with device.prepare_native_rows(*controls) as evaluate:
                source = _read(device, SOURCE_FIELDS)
                prepared = _read(device, ("status", "efc_data", "vectors"))
                np.testing.assert_array_equal(prepared["status"], [0])
                order = _check_source(model, device, 0, source, prepared["efc_data"][0], oracle)
                nr = int(source["constraint_counts"][0, 2])
                assert nr == 32
                row = prepared["efc_data"][0, :nr].astype(float)
                mass = source["mass"][0].astype(float)
                load = (
                    source["applied"][0].astype(float)
                    + source["passive"][0].astype(float)
                    - source["bias"][0].astype(float)
                    + extra[0].astype(float)
                )
                acceleration, _ = _root(
                    mass,
                    load,
                    source["efc_jacobian"][0, :nr].astype(float),
                    row[:, 4],
                    row[:, 3],
                )
                h = float(np.float32(device.h))
                damping = prepared["vectors"][1, :8].astype(float)
                integration = np.linalg.solve(mass + h * np.diag(damping), mass @ acceleration)
                expected_v = v0 + h * integration
                expected_q = q0.copy()
                mj.mj_integratePos(model, expected_q, expected_v, h)
                assert np.linalg.norm(integration - acceleration, np.inf) > 1e-5
                increment = np.float32(np.float32(h) - old_clock[1])
                total = np.float32(old_clock[0] + increment)
                compensation = np.float32(np.float32(total - old_clock[0]) - increment)
                expected_clock = np.array([[total, compensation]], np.float32)
                for name in ("geometry", "mechanics", "contacts"):
                    guard.setattr(device, name, (_forbidden, _forbidden))
                guard.setattr(device, "_assemble_rows", _forbidden)

                evaluate(extra_buffer)
                _check_solution(source, _read(device, RESULT_FIELDS), extra, [oracle], [order])
                _same_banks(device, before)
                assert entry_bytes() == entry_before

                evaluate(extra_buffer, integrate=True)
                result = _read(device, RESULT_FIELDS + ("acceleration", "integration_acc"))
                _check_solution(source, result, extra, [oracle], [order])
                np.testing.assert_allclose(
                    result["acceleration"][0], acceleration, **normal.ACC_TOL
                )
                np.testing.assert_allclose(
                    result["integration_acc"][0], integration, **normal.ACC_TOL
                )
                candidate = _banks(device)
                assert candidate[0] == old
                for offset in (0, 2, 4):
                    np.testing.assert_array_equal(
                        candidate[1][offset + old], before[1][offset + old]
                    )
                np.testing.assert_allclose(
                    candidate[1][2 + new][0], expected_v, rtol=5e-5, atol=2e-5
                )
                np.testing.assert_allclose(candidate[1][new][0], expected_q, rtol=5e-6, atol=2e-7)
                np.testing.assert_array_equal(candidate[1][4 + new], expected_clock)
                elapsed = (
                    float(expected_clock[0, 0])
                    - float(expected_clock[0, 1])
                    - float(old_clock[0])
                    + float(old_clock[1])
                )
                assert abs(elapsed - h) <= 2e-12
                free_joint = int(np.flatnonzero(model.jnt_type == mj.mjtJoint.mjJNT_FREE)[0])
                qa = int(model.jnt_qposadr[free_joint])
                assert np.any(candidate[1][new][0, qa : qa + 7] != q0[qa : qa + 7])
                assert abs(np.linalg.norm(candidate[1][new][0, qa + 3 : qa + 7]) - 1) < 2e-6
                assert entry_bytes() == entry_before

                with pytest.raises(RuntimeError):
                    evaluate(None)
                _same_banks(device, candidate)
                with pytest.raises(RuntimeError):
                    evaluate(extra_buffer, integrate=True)
                _same_banks(device, candidate)
                np.testing.assert_array_equal(_read(device, ("status",))["status"], [0])
                assert entry_bytes() == entry_before
        _same_banks(device, candidate)
        assert entry_bytes() == entry_before
    finally:
        if extra_buffer is not None:
            extra_buffer.free()
        device.close()
