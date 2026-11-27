"""Compare the compact recorder's scoring inputs with the published device receipts."""
from importlib import import_module
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def receipts(monkeypatch):
    mj = pytest.importorskip("mujoco")
    root = Path(__file__).resolve().parents[1] / "examples"
    monkeypatch.syspath_prepend(str(root))
    evaluate = import_module("_panda_pickup_rl_evaluate")
    cal = import_module("panda_pickup_resident_load_check")
    model = mj.MjModel.from_xml_path(str(root / "checkpoints/panda-reach-factors-s4000/scene.xml"))
    loads = evaluate.PickupLoads(model)
    directory = root / "checkpoints/panda-pickup-from-reach-s4000/s4000"
    rows = []
    for branch in ("student", "open"):
        with np.load(directory / f"world-000-{branch}.npz", allow_pickle=False) as bank:
            rows.append({k: bank[k] for k in bank.files})
    return evaluate, cal, loads, rows


def test_compact_worlds_recover_published_device_scores(receipts):
    evaluate, _, loads, rows = receipts
    evaluate._check_motor_model(loads.model, loads)
    arrays = {k: np.stack([r[k] for r in rows], axis=1) for k in evaluate.CHANNEL_SHAPES}
    arrays["clock"] = np.stack([r["time_s"] for r in rows], axis=1)
    traces, scores = evaluate._score_resident_arrays(loads, arrays, 0)
    for row, trace, score in zip(rows, traces, scores, strict=True):
        expected = loads.score(row, contact_source=evaluate.CONTACT_SOURCE)
        assert {k: score[k] for k in expected} == expected
        np.testing.assert_array_equal(row["part_noncontact_applied"], trace["part_noncontact_applied"])
        np.testing.assert_array_equal(row["input_tick"], trace["input_tick"])
        np.testing.assert_array_equal(row["finger_opening_m"], trace["finger_opening_m"])


def test_resident_calibration_requires_supported_contact_and_matched_states(receipts):
    _, cal, loads, rows = receipts
    row = rows[0]
    keys = ("time_s", "qpos", "qvel", "ctrl", "end_qpos", "end_qvel", "normal_n", "up_n", "force_world_n")
    reference = {k: row[k].copy() for k in keys}
    resident = {k: v.copy() for k, v in reference.items()}
    assert cal.compare_arrays(reference, resident, loads.weight_n)["supported_steps"] > 0
    resident["normal_n"][:, 0, 0] += .01
    with pytest.raises(AssertionError, match="normal_n"):
        cal.compare_arrays(reference, resident, loads.weight_n)
    resident["normal_n"][:] = reference["normal_n"]
    resident["qvel"][0, 0] += .01
    with pytest.raises(AssertionError, match="qvel"):
        cal.compare_arrays(reference, resident, loads.weight_n)
    reference["up_n"][:] = 0.
    with pytest.raises(RuntimeError, match="fork-supported"):
        cal.compare_arrays(reference, reference, loads.weight_n)


def test_resident_contact_buffers_require_fresh_accepted_native_solve():
    from threading import RLock
    from types import SimpleNamespace
    from unittest.mock import Mock

    from phymeo.backends.mujoco_musa_resident import MusaResidentSession, PhysicsStepFailure

    def buffer(value):
        a = np.asarray(value)
        return SimpleNamespace(shape=a.shape, dtype=a.dtype, to_numpy=a.copy, copy_from_device=Mock())

    state = [buffer([0.]), buffer([0.])]
    ids = np.array([[0, 1, 3, 7], [1, 0, 2, 4]], dtype=np.int32)
    plan = SimpleNamespace(slot_ids=ids, nslot=1, area_friction_laws=())
    device = SimpleNamespace(plan=plan, active=0, nworld=1, q=state, v=state, time=state, controls=buffer([0.]),
                             body_force=buffer([0.]), advance=Mock(), launch_kernel=Mock(),
                             status=np.zeros(1, np.int32), _get=np.copy, counts={"completed_kernels": 0},
                             constraint_counts=buffer([[1, 1, 4]]), contact_map=buffer([[[0, 0, 4]]]),
                             contact_wrench=buffer([[[1., 0., 0., 0., 0., 0.]]]), normal_rows=buffer([[[2.]]]))
    device.publish = lambda: setattr(device, "active", 1 - device.active)
    owner = SimpleNamespace(_lock=RLock(), _device=device, world_keys=(0,), timestep=.002,
                            _model=SimpleNamespace(nq=23, nv=22, nu=16))
    session = MusaResidentSession(owner)
    owner._musa_resident_owner = session
    # Stand in for device allocations at _start; exercise the real step/reset publication methods.
    session._applied, session._check_ctrl, session._initial = buffer([0.]), Mock(), [state[0]] * 4
    with pytest.raises(RuntimeError, match="accepted step"):
        session.contact_buffers()
    session.step()
    first = session.contact_buffers()
    assert set(first) == {"counts", "contact_map", "contact_wrench", "normal_rows", "slot_ids"}
    np.testing.assert_array_equal(first["slot_ids"], ids[:1])
    assert not np.shares_memory(ids, first["slot_ids"])
    with pytest.raises(ValueError):
        first["slot_ids"].setflags(write=True)
    for name in ("counts", "contact_map", "contact_wrench", "normal_rows"):
        source = device.constraint_counts if name == "counts" else getattr(device, name)
        np.testing.assert_array_equal(first[name].to_numpy(), source.to_numpy())
        with pytest.raises(TypeError, match="writable"):
            first[name].copy_from_torch(None)
    session.step()
    with pytest.raises(RuntimeError, match="expired"):
        first["counts"].to_numpy()
    second = session.contact_buffers()["contact_wrench"]
    device.status[0] = 1
    with pytest.raises(PhysicsStepFailure):
        session.step()
    with pytest.raises(RuntimeError, match="accepted step"):
        session.contact_buffers()
    with pytest.raises(RuntimeError, match="expired"):
        second.to_numpy()
    device.status[0] = 0
    session.step()
    third = session.contact_buffers()["contact_map"]
    session.reset()
    with pytest.raises(RuntimeError, match="accepted step"):
        session.contact_buffers()
    with pytest.raises(RuntimeError, match="expired"):
        third.to_numpy()
    device.plan.area_friction_laws = ("mixed",)
    session.step()
    with pytest.raises(NotImplementedError, match="native mixed constraint"):
        session.contact_buffers()
