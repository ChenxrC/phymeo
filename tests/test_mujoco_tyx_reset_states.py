"""N5 public reset-state transaction cases, authored before execution."""

import base64
import hashlib
import importlib.util
import json
from dataclasses import fields, is_dataclass
from pathlib import Path
from collections.abc import Mapping

import numpy as np
import pytest

from phymeo.backends.mujoco_runtime import MuJoCoControl, MuJoCoInitialState, MuJoCoRuntime
from phymeo.builders.mujoco_compiler import compile_mujoco_xml

MODES = ("interpret", pytest.param("cuda", marks=pytest.mark.cuda))
KEYS = (7, 31, 103, 4096)


def tasks():
    path = Path(__file__).resolve().parents[1] / "examples/mujoco_tyx_learning_tasks.py"
    spec = importlib.util.spec_from_file_location("n5_reset_tasks", path)
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def source(case):
    assert pytest.importorskip("mujoco").__version__ == "3.3.7"
    pytest.importorskip("tyx")
    return compile_mujoco_xml(tasks().task(case).xml, require_asset=False)


def state(case, index, *, time=0.0):
    return MuJoCoInitialState(
        [0.02 + index * 0.005, -0.03] if case == "J" else [0.0007 + index * 0.000025],
        [0.025, -0.02] if case == "J" else [0.001],
        time=time,
    )


def controls(case):
    return {
        key: MuJoCoControl([0.15, -0.12] if case == "J" else [19.62])
        for key in KEYS
    }


def signature(value):
    if isinstance(value, np.ndarray):
        return (value.dtype.str, value.shape, value.tobytes())
    if is_dataclass(value):
        return (type(value).__name__, tuple(
            (f.name, signature(getattr(value, f.name))) for f in fields(value)
        ))
    if isinstance(value, Mapping):
        return tuple((key, signature(item)) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return tuple(signature(item) for item in value)
    return value


def payload(engine):
    raw = engine.checkpoint().to_bytes()
    magic, digest, body = raw.split(b"\n", 2)
    assert magic == b"PHYMEOTYXCP1"
    assert hashlib.sha256(body).hexdigest().encode() == digest
    return json.loads(body)["state"]


def bank_row(record, row):
    value = np.frombuffer(base64.b64decode(record["data"]), dtype=record["$array"])
    return value.reshape(record["shape"])[row].tobytes()


@pytest.mark.tyx
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("case", ("J", "S"))
def test_public_subset_reset_keeps_full_unselected_state(case, mode, tmp_path):
    compiled = source(case)
    initial = {key: state(case, i, time=0.125) for i, key in enumerate(KEYS)}
    h = tasks().task(case).h
    with MuJoCoRuntime(compiled, world_keys=KEYS, mode=mode, initial_states=initial) as engine:
        engine.set_controls(controls(case))
        for _ in range(3):
            assert all(engine.step(h).values())
        before = payload(engine)
        snapshots = engine.snapshot()
        revision = engine.checkpoint_revision
        requested = {KEYS[3]: state(case, 1), KEYS[1]: state(case, 2)}
        receipt = engine.reset_states(requested, expected_revision=revision)
        assert receipt.schema == "mujoco-tyx-reset-v1"
        assert receipt.world_keys == (KEYS[1], KEYS[3])
        assert receipt.source_revision == revision
        assert receipt.destination_revision == revision + 1
        assert len(receipt.binding_sha256) == 64
        assert not receipt.cleanup_errors
        assert receipt.clock.shape == (2, 2) and receipt.clock.dtype == np.float32
        assert receipt.clock.tobytes() == bytes(16)
        with pytest.raises(ValueError):
            receipt.clock.setflags(write=True)
        after = payload(engine)
        current = engine.snapshot()
        for index in (0, 2):
            assert after["worlds"][index] == before["worlds"][index]
            for name in ("q", "v", "clock"):
                assert bank_row(after[name], index) == bank_row(before[name], index)
            assert signature(current[KEYS[index]]) == signature(snapshots[KEYS[index]])
        for key in receipt.world_keys:
            for name in ("q", "v"):
                expected = getattr(requested[key], name).astype(np.float32).astype(np.float64)
                assert getattr(receipt.stored[key], name).tobytes() == expected.tobytes()
                assert getattr(current[key], name).tobytes() == expected.tobytes()
                with pytest.raises(ValueError):
                    getattr(receipt.stored[key], name).setflags(write=True)
            assert current[key].tick == 0 and current[key].time == 0
            assert current[key].last_step is None
            assert not np.any(current[key].held_control.ctrl)
        with pytest.raises(TypeError):
            receipt.requested[KEYS[0]] = initial[KEYS[0]]
        engine.set_controls(controls(case))
        for _ in range(2):
            assert all(engine.step(h).values())
        if case == "S":
            assert all(any(c.normal_force > 0 for c in value.last_step.contacts)
                       for value in engine.snapshot().values())
        # reset_states never replaces the original reset declaration.
        assert engine.reset() is None
        for key in KEYS:
            assert engine.snapshot()[key].time == float(np.float32(0.125))
            assert engine.snapshot()[key].q.tobytes() == initial[key].q.astype(
                np.float32
            ).astype(np.float64).tobytes()
        (tmp_path / f"reset-{case}-{mode}.json").write_text(json.dumps({
            "schema": receipt.schema, "keys": receipt.world_keys,
            "binding": receipt.binding_sha256,
            "before": before, "after": after,
        }), encoding="utf-8")


@pytest.mark.tyx
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("fault", ("stale", "time", "unknown", "late-world", "build"))
def test_reset_states_precommit_refusal_is_atomic(mode, fault, monkeypatch):
    compiled = source("J")
    with MuJoCoRuntime(compiled, mode=mode, world_keys=KEYS) as engine:
        engine.set_controls(controls("J"))
        assert all(engine.step(0.002).values())
        before = engine.checkpoint().to_bytes()
        revision = engine.checkpoint_revision
        wanted = {key: state("J", i) for i, key in enumerate(KEYS)}
        if fault == "stale":
            revision -= 1
        elif fault == "time":
            wanted[KEYS[-1]] = state("J", 3, time=1)
        elif fault == "unknown":
            wanted[5000] = state("J", 0)
        elif fault == "late-world":
            wanted[KEYS[-1]] = MuJoCoInitialState([0.1], [0.1])
        else:
            def fail(*args, **kwargs):
                raise RuntimeError("injected replacement build failure")
            monkeypatch.setattr(engine, "_build_device", fail)
        with pytest.raises((ValueError, RuntimeError, KeyError)):
            engine.reset_states(wanted, expected_revision=revision)
        assert engine.checkpoint().to_bytes() == before


@pytest.mark.tyx
@pytest.mark.parametrize("mode", MODES)
def test_empty_reset_and_retired_cleanup_do_not_fake_refusal(mode, monkeypatch):
    compiled = source("J")
    with MuJoCoRuntime(compiled, mode=mode, world_keys=KEYS) as engine:
        before = engine.checkpoint().to_bytes()
        receipt = engine.reset_states({}, expected_revision=engine.checkpoint_revision)
        assert receipt.world_keys == ()
        assert receipt.source_revision == receipt.destination_revision
        assert engine.checkpoint().to_bytes() == before
        with pytest.raises(ValueError):
            engine.reset_states({}, expected_revision=engine.checkpoint_revision - 1)
        old = engine._device  # Fault injection only, never a state-write shortcut.
        close = old.close
        calls = []

        def fail_once():
            calls.append(True)
            if len(calls) == 1:
                raise RuntimeError("injected old-owner close failure")
            return close()

        monkeypatch.setattr(old, "close", fail_once)
        receipt = engine.reset_states(
            {KEYS[0]: state("J", 1)}, expected_revision=engine.checkpoint_revision
        )
        assert receipt.cleanup_errors
        assert engine.snapshot()[KEYS[0]].q.tobytes() == receipt.stored[KEYS[0]].q.tobytes()
        engine.set_controls(controls("J"))
        assert all(engine.step(0.002).values())
    assert len(calls) >= 2


@pytest.mark.tyx
@pytest.mark.parametrize("mode", MODES)
def test_bad_free_anchor_is_not_normalized_or_written(mode):
    pytest.importorskip("mujoco")
    compiled = compile_mujoco_xml(
        '<mujoco><option timestep=".002" gravity="0 0 0"/>'
        '<worldbody><body><freejoint/><inertial pos="0 0 0" mass="1" '
        'diaginertia=".02 .03 .04"/><geom type="box" size=".1 .1 .1" '
        'contype="0" conaffinity="0"/></body></worldbody></mujoco>',
        require_asset=False,
    )
    with MuJoCoRuntime(compiled, mode=mode, world_keys=KEYS) as engine:
        before = engine.checkpoint().to_bytes()
        with pytest.raises(ValueError):
            engine.reset_states(
                {KEYS[0]: MuJoCoInitialState([0, 0, 0, 1, 0.1, 0, 0], np.zeros(6))},
                expected_revision=engine.checkpoint_revision,
            )
        assert engine.checkpoint().to_bytes() == before
