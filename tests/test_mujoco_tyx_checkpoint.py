"""Exact public TYX checkpoint continuation; authored before any A1 execution.

The expectations are complete bit equality on repeated accepted execution,
not a checkpoint producer/verifier self-comparison or a fitted tolerance.
CPU snapshots are detached observations; no CPU step drives this trajectory.
"""

import hashlib
import json
import struct
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass, replace

import numpy as np
import pytest

from phymeo.backends import mujoco_checkpoint as capsule
from phymeo.backends.mujoco_checkpoint import MuJoCoTyxCheckpoint
from phymeo.backends.mujoco_runtime import MuJoCoControl, MuJoCoInitialState, MuJoCoRuntime
from phymeo.builders.mujoco_compiler import compile_mujoco_xml
from phymeo.training.resident_stepper import PhysicsStepFailure

MODES = ("interpret", pytest.param("cuda", marks=pytest.mark.cuda))
KEYS = (4, 31)
H = 0.002
SHAPE = '<geom type="box" size=".04 .03 .02" contype="0" conaffinity="0"/>'
INERTIA = '<inertial pos=".02 -.01 .03" mass="1.2" fullinertia=".04 .05 .06 .001 -.002 .001"/>'
SCALAR = f"""<body name="hinged" pos=".2 0 .1"><joint name="a" type="hinge" axis="0 1 0"
  damping=".04" armature=".03"/>{INERTIA}{SHAPE}
  <body name="slide" pos=".1 .1 0"><joint name="b" type="slide" axis="1 0 0"
  damping=".02" armature=".01"/>{INERTIA}{SHAPE}</body></body>"""
MOTOR = '<actuator><motor name="drive" joint="a" gear=".7"/></actuator>'
FREE = f"""<body name="root" pos=".1 -.1 .7"><freejoint/>{INERTIA}{SHAPE}{SCALAR}</body>"""
MIXED = """<geom name="floor" type="plane" size="2 2 .1"/>
  <body name="root" pos="0 0 .099"><freejoint/>
    <inertial pos=".01 -.01 .02" mass="1.4" fullinertia=".06 .07 .08 .001 -.001 .001"/>
    <geom type="sphere" pos="-.12 0 0" size=".1"/><geom type="sphere" pos=".12 0 0" size=".1"/>
    <body name="joint" pos="0 0 .25"><joint name="a" type="hinge" axis="0 1 0"
      limited="true" range="-.2 .2" margin=".02" frictionloss=".03" armature=".02" damping=".04"/>
      <inertial pos="0 0 .03" mass=".2" diaginertia=".003 .004 .005"/>
      <geom type="sphere" size=".02" contype="0" conaffinity="0"/>
    </body>
  </body>"""
SLIDER = f"""<body name="slide" pos=".2 0 .4">
  <joint name="a" type="slide" axis="1 0 0" damping=".03"/>
  <joint name="b" type="slide" axis="0 1 0" damping=".04"/>
  <joint name="c" type="slide" axis="0 0 1" damping=".05"/>
  {INERTIA}{SHAPE}</body>"""


def xml(body, *, mixed=False):
    default = """<default><geom condim="6" friction=".6 .02 .003"
      solref=".02 1" solimp=".8 .9 .002 .5 2"/></default>""" if mixed else ""
    return f"""<mujoco model="checkpoint"><compiler angle="radian" autolimits="true"
      fusestatic="false"/><option timestep=".002" gravity=".1 -.2 -9.81" integrator="Euler"
      cone="pyramidal" solver="Newton" iterations="128" tolerance="1e-10"/>
      {default}<worldbody>{body}</worldbody>{MOTOR}</mujoco>"""


CASES = {
    "prismatic": xml(SLIDER),
    "fixed-tree": xml(SCALAR),
    "floating-tree": xml(FREE),
    "floating-mixed": xml(MIXED, mixed=True),
}


def compile_case(name):
    assert pytest.importorskip("mujoco").__version__ == "3.3.7"
    pytest.importorskip("tyx")
    return compile_mujoco_xml(CASES[name], require_asset=False)


def initial(compiled, key):
    model = compiled.fresh_native_model()
    q = model.qpos0.copy()
    free = int(model.jnt_type[0]) == 0
    if free:
        quat = np.array([1.0, .02, -.015, .01 + key * .00001])
        q[3:7] = quat / np.linalg.norm(quat)
        if model.nq > 7:
            q[7:] = .19 if model.nq == 8 else [.07, -.025]
        v = np.array([.08, -.03, -.015, .12, -.09, .08] + [.045] * (model.nv - 6))
    else:
        q[:] += np.linspace(.02, .06, model.nq)
        v = np.linspace(.04, -.035, model.nv)
    return MuJoCoInitialState(q, v * (1 + key * .001), time=.123)


def runtime(compiled, mode, *, keys=KEYS, contact_capacity=16, initial_states=None):
    return MuJoCoRuntime(
        compiled, world_keys=keys, mode=mode, contact_capacity=contact_capacity,
        constraint_capacity=64,
        initial_states=({key: initial(compiled, key) for key in keys}
                        if initial_states is None else initial_states),
    )


def packet(compiled, tick, key):
    model = compiled.fresh_native_model()
    direction = -1 if (tick + key) % 3 == 0 else 1
    qforce = np.linspace(.004, .014, model.nv) * direction
    wrench = np.zeros((model.nbody, 6))
    wrench[-1] = np.array([.01, -.02, .025, .003, -.002, .004]) * direction
    return MuJoCoControl([direction * .03], qfrc_applied=qforce, xfrc_applied=wrench)


def exact(value):
    """Independent recursive bit signature; includes every dataclass field."""
    if isinstance(value, np.ndarray):
        return ("array", value.dtype.str, value.shape, value.tobytes())
    if is_dataclass(value):
        return (type(value).__name__, tuple((f.name, exact(getattr(value, f.name))) for f in fields(value)))
    if isinstance(value, Mapping):
        return tuple((key, exact(value[key])) for key in sorted(value))
    if isinstance(value, (tuple, list)):
        return tuple(exact(item) for item in value)
    if type(value) is float:
        return ("float64", struct.pack("<d", value))
    return value


def save(path, checkpoint):
    path.write_bytes(checkpoint.to_bytes())
    path.with_suffix(path.suffix + ".sha256").write_text(
        hashlib.sha256(path.read_bytes()).hexdigest() + "  " + path.name + "\n", encoding="utf-8"
    )
    return MuJoCoTyxCheckpoint.from_bytes(path.read_bytes())


def advance(engine, compiled, tick, *, update=True):
    if update:
        engine.set_controls({key: packet(compiled, tick, key) for key in engine.world_keys})
    assert all(engine.step(H).values())
    return exact(engine.snapshot())


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("case", tuple(CASES))
def test_exact_accepted_batch_restores_and_continues(case, mode, tmp_path):
    compiled = compile_case(case)
    with runtime(compiled, mode) as engine:
        for tick in range(5):
            advance(engine, compiled, tick)
        # Held control may differ from the last applied receipt; both must survive.
        engine.set_controls({key: packet(compiled, 11, key) for key in KEYS})
        before = exact(engine.snapshot())
        cp = save(tmp_path / "accepted.checkpoint", engine.checkpoint())
        revision = engine.checkpoint_revision
        assert cp.source_revision == revision
        assert cp.binding["world_keys"] == KEYS
        assert cp.binding["mode"] == mode
        expected = [advance(engine, compiled, 11, update=False)]
        expected += [advance(engine, compiled, tick) for tick in range(12, 17)]
        engine.restore(cp, expected_revision=engine.checkpoint_revision)
        assert engine.checkpoint_revision > revision
        assert exact(engine.snapshot()) == before
        actual = [advance(engine, compiled, 11, update=False)]
        actual += [advance(engine, compiled, tick) for tick in range(12, 17)]
        assert actual == expected
        # Exact clock bytes, all receipts and both held/control channels survive
        # a public restore into a distinct compatible device owner.
        with runtime(compiled, mode, initial_states={key: MuJoCoInitialState() for key in KEYS}) as fresh:
            fresh.restore(cp, expected_revision=fresh.checkpoint_revision)
            assert exact(fresh.snapshot()) == before
            observed = [advance(fresh, compiled, 11, update=False)]
            observed += [advance(fresh, compiled, tick) for tick in range(12, 17)]
            assert observed == expected
            # Restore also restores the checkpoint owner's original reset declaration.
            engine.reset()
            fresh.reset()
            assert exact(fresh.snapshot()) == exact(engine.snapshot())


@pytest.mark.parametrize("mode", MODES)
def test_free_quaternion_is_accepted_bytes_not_a_new_initial_declaration(mode, tmp_path):
    compiled = compile_case("floating-tree")
    with runtime(compiled, mode) as engine:
        for tick in range(4):
            advance(engine, compiled, tick)
        before = engine.snapshot()
        q = before[KEYS[0]].q
        # FP32 integration's normalized result is not a mathematical FP64 unit declaration.
        assert abs(float(q[3:7] @ q[3:7]) - 1) > 64 * np.finfo(float).eps
        declarations = {key: MuJoCoInitialState(s.q, s.v, time=s.time) for key, s in before.items()}
        with pytest.raises(ValueError, match="unit norm"):
            runtime(compiled, mode, initial_states=declarations)
        cp = save(tmp_path / "free.checkpoint", engine.checkpoint())
        advance(engine, compiled, 7)
        engine.restore(cp, expected_revision=engine.checkpoint_revision)
        restored = engine.snapshot()
        for key in KEYS:
            assert restored[key].q.tobytes() == before[key].q.tobytes()
            assert restored[key].v.tobytes() == before[key].v.tobytes()
        assert exact(restored) == exact(before)


@pytest.mark.parametrize("mode", MODES)
def test_revision_cas_and_invalid_controls_or_step_leave_accepted_state(mode):
    compiled = compile_case("fixed-tree")
    with runtime(compiled, mode) as engine:
        advance(engine, compiled, 0)
        cp = engine.checkpoint()
        stale = engine.checkpoint_revision
        engine.set_controls({key: packet(compiled, 4, key) for key in KEYS})
        before, revision = exact(engine.snapshot()), engine.checkpoint_revision
        with pytest.raises(ValueError, match="stale"):
            engine.restore(cp, expected_revision=stale)
        assert engine.checkpoint_revision == revision and exact(engine.snapshot()) == before
        with pytest.raises((ValueError, TypeError)):
            engine.set_controls({KEYS[0]: packet(compiled, 2, KEYS[0])})
        with pytest.raises(ValueError):
            engine.step(H * 2)
        assert engine.checkpoint_revision == revision and exact(engine.snapshot()) == before
        engine.restore(cp, expected_revision=revision)
        assert engine.checkpoint_revision == revision + 1


def rewrite(cp, change):
    """Independent envelope author for corruption/admission negative cases."""
    payload = json.loads(cp.to_bytes().split(b"\n", 2)[2])
    change(payload)
    body = json.dumps(payload, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()
    return MuJoCoTyxCheckpoint.from_bytes(
        b"PHYMEOTYXCP1\n" + hashlib.sha256(body).hexdigest().encode() + b"\n" + body
    )


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("mismatch", ("source", "world-order", "capacity", "backend", "layout"))
def test_wrong_binding_rejected_before_device_allocation(mode, mismatch, monkeypatch):
    compiled = compile_case("fixed-tree")
    with runtime(compiled, mode) as engine:
        advance(engine, compiled, 0)
        cp = engine.checkpoint()
        field, replacement = {
            "source": ("model_sha256", "0" * 64),
            "world-order": ("world_keys", list(reversed(KEYS))),
            "capacity": ("capacities", [15, 64]),
            "backend": ("mode", "cuda" if mode == "interpret" else "interpret"),
            "layout": ("layout", [9, 8, 2]),
        }[mismatch]
        bad = rewrite(cp, lambda value: value["binding"].__setitem__(field, replacement))
        before, revision = exact(engine.snapshot()), engine.checkpoint_revision
        def forbidden(*args, **kwargs):
            raise AssertionError("wrong binding allocated a device")
        monkeypatch.setattr(engine, "_build_device", forbidden)
        with pytest.raises(ValueError, match="binding"):
            engine.restore(bad, expected_revision=revision)
        assert engine.checkpoint_revision == revision
        assert exact(engine.snapshot()) == before


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("bad", ("shape", "nonfinite", "clock", "receipt-tick", "duplicate-key"))
def test_bounded_payload_validation_is_transactional(mode, bad, monkeypatch):
    compiled = compile_case("fixed-tree")
    with runtime(compiled, mode) as engine:
        advance(engine, compiled, 0)
        cp = engine.checkpoint()
        before, revision = exact(engine.snapshot()), engine.checkpoint_revision
        if bad == "duplicate-key":
            raw = cp.to_bytes().split(b"\n", 2)[2]
            body = b'{"revision":0,' + raw[1:]
            broken = MuJoCoTyxCheckpoint(b"PHYMEOTYXCP1\n" + hashlib.sha256(body).hexdigest().encode() + b"\n" + body)
        else:
            def change(value):
                state = value["state"]
                if bad == "shape":
                    state["q"]["shape"] = [2**31, 8]
                elif bad == "nonfinite":
                    # Exact FP32 +inf in a one-item declared array; rejected before shape use.
                    state["q"] = {"$array": "<f4", "shape": [1], "data": "AACAfw=="}
                elif bad == "clock":
                    state["worlds"][0]["time"] += 1
                else:
                    state["worlds"][0]["receipt"]["fields"]["tick"] += 1
            broken = rewrite(cp, change)
        def forbidden(*args, **kwargs):
            raise AssertionError("invalid payload allocated a device")
        monkeypatch.setattr(engine, "_build_device", forbidden)
        with pytest.raises(ValueError):
            engine.restore(broken, expected_revision=revision)
        assert engine.checkpoint_revision == revision
        assert exact(engine.snapshot()) == before


@pytest.mark.parametrize("mode", MODES)
def test_late_restore_readback_failure_preserves_batch_and_future(mode, monkeypatch):
    compiled = compile_case("floating-mixed")
    with runtime(compiled, mode) as engine:
        for tick in range(4):
            advance(engine, compiled, tick)
        cp = engine.checkpoint()
        before, revision = exact(engine.snapshot()), engine.checkpoint_revision
        expected = advance(engine, compiled, 4)
        engine.restore(cp, expected_revision=engine.checkpoint_revision)
        revision = engine.checkpoint_revision
        def late(*args):
            raise RuntimeError("authored replacement readback fault")
        with monkeypatch.context() as patch:
            patch.setattr(capsule, "_verify_bank", late)
            with pytest.raises(RuntimeError, match="authored replacement"):
                engine.restore(cp, expected_revision=revision)
        assert engine.checkpoint_revision == revision
        assert exact(engine.snapshot()) == before
        assert advance(engine, compiled, 4) == expected


@pytest.mark.parametrize("mode", MODES)
def test_rejected_candidate_bank_is_never_captured(mode, monkeypatch):
    compiled = compile_case("fixed-tree")
    with runtime(compiled, mode) as engine:
        advance(engine, compiled, 0)
        before = exact(engine.snapshot())
        revision = engine.checkpoint_revision
        original = engine._finite_state
        def reject(candidate):
            original(candidate)
            raise RuntimeError("authored post-device rejection")
        # step has already run the real kernel when host validation rejects.
        with monkeypatch.context() as patch:
            patch.setattr(engine, "_finite_state", reject)
            with pytest.raises(PhysicsStepFailure):
                engine.step(H)
        assert engine.checkpoint_revision == revision and exact(engine.snapshot()) == before
        cp = engine.checkpoint()
        engine.restore(cp, expected_revision=revision)
        assert exact(engine.snapshot()) == before


@dataclass(frozen=True)
class ControllerHistory:
    tick: int
    memory: tuple
    held: float
    physics_sha256: str | None = None

    def proposal(self, state):
        if self.tick % 2:
            return self, self.held
        memory = tuple(.75 * x + .25 * float(v) for x, v in zip(self.memory, state.v, strict=True))
        held = .02 + .01 * sum(memory)
        return replace(self, memory=memory, held=held), held

    def commit(self):
        return replace(self, tick=self.tick + 1)


@pytest.mark.parametrize("mode", MODES)
def test_explicit_policy_history_and_clock_composition(mode):
    compiled = compile_case("floating-tree")
    with runtime(compiled, mode, keys=(4,)) as engine:
        history = ControllerHistory(0, tuple(0.0 for _ in range(compiled.fresh_native_model().nv)), 0.0)
        def one(owner):
            proposal, action = owner.proposal(engine.snapshot()[4])
            engine.set_controls({4: MuJoCoControl([action])})
            engine.step(H)
            return proposal.commit(), (action, exact(engine.snapshot()))
        for _ in range(5):
            history, _ = one(history)
        cp = engine.checkpoint()
        saved = replace(history, physics_sha256=cp.sha256)
        expected = []
        for _ in range(7):
            history, result = one(history)
            expected.append(result)
        engine.restore(cp, expected_revision=engine.checkpoint_revision)
        assert saved.physics_sha256 == cp.sha256
        history = replace(saved, physics_sha256=None)
        actual = []
        for _ in range(7):
            history, result = one(history)
            actual.append(result)
        assert actual == expected
        # An arbitrary PolicyRunner/ONNX hidden state was not silently cloned.


def test_capsule_bytes_are_owned_immutable_and_checksum_checked():
    # Pure schema test needs no optional native/backend import.
    body = b'{"binding":{},"revision":0,"state":{}}'
    buffer = bytearray(b"PHYMEOTYXCP1\n" + hashlib.sha256(body).hexdigest().encode() + b"\n" + body)
    cp = MuJoCoTyxCheckpoint(buffer)
    original = cp.to_bytes()
    buffer[-2] ^= 1
    assert cp.to_bytes() == original
    with pytest.raises(ValueError, match="checksum"):
        MuJoCoTyxCheckpoint(buffer)
    with pytest.raises((AttributeError, TypeError)):
        cp._bytes = b"changed"


@pytest.mark.parametrize("mode", MODES)
def test_actual_different_compiled_source_rejected(mode, monkeypatch):
    compiled = compile_case("fixed-tree")
    changed = compile_mujoco_xml(CASES["fixed-tree"].replace('mass="1.2"', 'mass="1.3"'), require_asset=False)
    with runtime(compiled, mode) as source, runtime(changed, mode) as target:
        advance(source, compiled, 0)
        cp = source.checkpoint()
        before, revision = exact(target.snapshot()), target.checkpoint_revision
        def forbidden(*args, **kwargs):
            raise AssertionError("different compiled source allocated a replacement")
        monkeypatch.setattr(target, "_build_device", forbidden)
        with pytest.raises(ValueError, match="binding"):
            target.restore(cp, expected_revision=revision)
        assert exact(target.snapshot()) == before and target.checkpoint_revision == revision


@pytest.mark.parametrize("mode", MODES)
def test_subset_reset_has_distinct_world_ticks_and_original_clock_bits(mode):
    compiled = compile_case("fixed-tree")
    with runtime(compiled, mode) as engine:
        for tick in range(3):
            advance(engine, compiled, tick)
        engine.reset(keys=(KEYS[0],))
        snapshot = engine.snapshot()
        assert (snapshot[KEYS[0]].tick, snapshot[KEYS[1]].tick) == (0, 3)
        cp = engine.checkpoint()
        before = exact(snapshot)
        expected = advance(engine, compiled, 5)
        engine.restore(cp, expected_revision=engine.checkpoint_revision)
        assert exact(engine.snapshot()) == before
        again = engine.checkpoint()
        original = json.loads(cp.to_bytes().split(b"\n", 2)[2])
        restored = json.loads(again.to_bytes().split(b"\n", 2)[2])
        assert restored["state"] == original["state"]
        assert advance(engine, compiled, 5) == expected


@pytest.mark.parametrize("mode", MODES)
def test_postcommit_cleanup_failure_does_not_report_false_rollback(mode, monkeypatch):
    compiled = compile_case("fixed-tree")
    with runtime(compiled, mode) as engine:
        advance(engine, compiled, 0)
        cp = engine.checkpoint()
        expected = advance(engine, compiled, 1)
        revision = engine.checkpoint_revision
        old = engine._device
        def cleanup_failure():
            raise RuntimeError("authored retired-owner cleanup failure")
        with monkeypatch.context() as patch:
            patch.setattr(old, "close", cleanup_failure)
            engine.restore(cp, expected_revision=revision)
        assert engine.checkpoint_revision == revision + 1
        assert advance(engine, compiled, 1) == expected
        # Public close retries the old owner, while the new owner was already valid.
