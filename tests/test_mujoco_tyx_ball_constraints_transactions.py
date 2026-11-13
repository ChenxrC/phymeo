"""BALL-S actual mixed-row whole-batch failure and continuation tests.

Authoring only. Fixed cases and all numerical gates live in the standalone
consumer. Unknown release retains the original resource contract unchanged.
"""

import base64
import hashlib
import json

import numpy as np
import pytest
from test_mujoco_tyx_ball_constraints import MODES, case

ROUTES = ("fixed-lower", "free-filter-limits", "mixed-pyramid-filter")
FAULTS = tuple(
    (name, fault)
    for name in ROUTES
    for fault in (
        ("status", "readback", "constraints", "restore")
        if name == "fixed-lower"
        else ("status", "readback", "constraints", "restore", "actuation", "transmission")
    )
)


def retained(runtime, before, raw, revision, journal):
    actual = runtime.integration_snapshot()
    checkpoint = runtime.checkpoint().to_bytes()
    journal.emit(
        "retained", state=actual, checkpoint=checkpoint, revision=runtime.checkpoint_revision
    )
    case.exact(actual, before, "complete rejected q/v/act/control/force/history")
    assert checkpoint == raw and runtime.checkpoint_revision == revision
    assert all(actual[k].last_step is before[k].last_step for k in runtime.world_keys)


def retry(runtime, clone, mj, model, before, journal):
    held = {k: s.held_control for k, s in before.items()}
    actual = case.advance(
        runtime, mj, model, held, case.references(mj, model, before), journal, "retry"
    )
    expected = case.advance(
        clone, mj, model, held, case.references(mj, model, before), journal, "untouched-clone"
    )
    case.exact(actual, expected, "complete actual retry successor")


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("name", ROUTES)
def test_ball_scalar_checkpoint_clone_partial_reset_real_successors(name, mode, tmp_path):
    with case.saved(tmp_path / "lifecycle.jsonl.gz") as journal:
        case.lifecycle(name, mode, journal, output_path=tmp_path)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("name,fault", FAULTS)
def test_ball_scalar_late_fault_retains_both_worlds_and_retries(
    name, fault, mode, monkeypatch, tmp_path
):
    import phymeo.backends.mujoco_actuation as actuation
    import phymeo.backends.mujoco_checkpoint as checkpoints
    import phymeo.backends.mujoco_joint_transmission as transmission
    import phymeo.backends.mujoco_tyx._constraint_publication as constraints
    from phymeo.training.resident_stepper import PhysicsStepFailure

    with case.saved(tmp_path / "late-failure.jsonl.gz") as journal:
        journal.emit("fixture-attempt", name=name, fault=fault, source=case.source(name))
        mj, model, _, _, runtime = case.fixture(name, mode)
        with case.owned(runtime, journal):
            refs = case.references(mj, model, runtime.integration_snapshot())
            case.advance(
                runtime,
                mj,
                model,
                {k: case.control(model, 0, i, name) for i, k in enumerate(case.KEYS)},
                refs,
                journal,
                "warm",
            )
            runtime.set_controls(
                {k: case.control(model, 8, i, name) for i, k in enumerate(case.KEYS)}
            )
            before = runtime.integration_snapshot()
            capsule = runtime.checkpoint()
            raw, revision = capsule.to_bytes(), runtime.checkpoint_revision
            journal.emit("fault-attempt", state=before, checkpoint=raw, revision=revision)
            with case.owned(runtime.clone(), journal) as clone:
                reached = []
                with monkeypatch.context() as change:
                    if fault == "restore":
                        original = checkpoints._verify_bank

                        def refuse(*args, **kwargs):
                            original(*args, **kwargs)
                            reached.append("verified-bank")
                            journal.emit("verified-replacement-bank", expected=args[1])
                            raise RuntimeError("BALL-S verified restore refused")

                        change.setattr(checkpoints, "_verify_bank", refuse)
                        with pytest.raises(RuntimeError, match="verified restore") as caught:
                            runtime.restore(capsule, expected_revision=revision)
                        assert reached == ["verified-bank"]
                    else:
                        if fault in ("constraints", "actuation", "transmission"):
                            module, attribute = {
                                "constraints": (constraints, "publish"),
                                "actuation": (actuation, "publish_actuation"),
                                "transmission": (transmission, "publish_joint_transmission"),
                            }[fault]
                            original = getattr(module, attribute)

                            def refuse(*args, **kwargs):
                                value = original(*args, **kwargs)
                                index = args[2]
                                reached.append(index)
                                journal.emit(
                                    "actual-publication-prefix",
                                    kind=fault,
                                    index=index,
                                    value=value,
                                )
                                if index == len(case.KEYS) - 1:
                                    raise RuntimeError("BALL-S late publication refused")
                                return value

                            change.setattr(module, attribute, refuse)
                        else:
                            original = runtime._device.download

                            def refuse():
                                value = original()
                                reached.append("download")
                                journal.emit("complete-real-download", raw=value)
                                if fault == "readback":
                                    raise RuntimeError("BALL-S completed readback refused")
                                value["status"][-1] = 2
                                journal.emit("injected-last-status", raw=value)
                                return value

                            change.setattr(runtime._device, "download", refuse)
                        with pytest.raises(PhysicsStepFailure) as caught:
                            runtime.step(case.H)
                        assert caught.value.acceptance_known and not caught.value.physics_accepted
                        assert caught.value.accepted_worlds == dict.fromkeys(case.KEYS, False)
                        assert reached == (
                            list(range(2))
                            if fault in ("constraints", "actuation", "transmission")
                            else ["download"]
                        )
                    journal.emit("rejected", error=repr(caught.value), reached=reached)
                    retained(runtime, before, raw, revision, journal)
                retry(runtime, clone, mj, model, before, journal)


def _change_array(record, transform):
    value = (
        np.frombuffer(base64.b64decode(record["data"]), dtype=record["$array"])
        .reshape(record["shape"])
        .copy()
    )
    transform(value)
    record["data"] = base64.b64encode(value.tobytes()).decode()


def forge(raw, corruption):
    header, digest, body = raw.split(b"\n", 2)
    assert hashlib.sha256(body).hexdigest().encode() == digest
    data = json.loads(body)
    fields = data["state"]["worlds"][-1]["receipt"]["fields"]
    if corruption == "foreign-transmission-schema":
        tr = fields["transmission"]["fields"]
        tr["schema"] = (
            "native-joint-motor-transmission-v1"
            if "free-ball" in tr["schema"]
            else "native-free-ball-joint-motor-transmission-v1"
        )
    elif corruption == "foreign-activation-pair":
        fields["actuation"]["fields"]["schema"] = "native-joint-filterexact-actuation-v1"
        fields["transmission"]["fields"]["schema"] = "native-joint-motor-transmission-v1"
    elif corruption == "force-binding":
        _change_array(fields["qfrc_constraint"], lambda a: a.__setitem__(-1, a[-1] + 0.125))
    else:
        raise AssertionError(corruption)
    original = json.loads(body)
    assert data["binding"] == original["binding"] and data["revision"] == original["revision"]
    assert data["state"]["worlds"][:-1] == original["state"]["worlds"][:-1]
    changed = json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return header + b"\n" + hashlib.sha256(changed).hexdigest().encode() + b"\n" + changed


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("name", ROUTES)
def test_ball_scalar_checksum_valid_checkpoint_cannot_cross_schema_or_force(name, mode, tmp_path):
    from phymeo.backends.mujoco_checkpoint import MuJoCoTyxCheckpoint

    with case.saved(tmp_path / "checkpoint-corruption.jsonl.gz") as journal:
        mj, model, _, _, runtime = case.fixture(name, mode)
        with case.owned(runtime, journal):
            state = runtime.integration_snapshot()
            case.advance(
                runtime,
                mj,
                model,
                {k: case.control(model, 0, i, name) for i, k in enumerate(case.KEYS)},
                case.references(mj, model, state),
                journal,
                "warm",
            )
            before = runtime.integration_snapshot()
            raw = runtime.checkpoint().to_bytes()
            revision = runtime.checkpoint_revision
            corruptions = (
                ("foreign-transmission-schema", "force-binding")
                if name == "fixed-lower"
                else ("foreign-transmission-schema", "foreign-activation-pair", "force-binding")
            )
            with case.owned(runtime.clone(), journal) as clone:
                for corruption in corruptions:
                    changed = forge(raw, corruption)
                    capsule = MuJoCoTyxCheckpoint.from_bytes(changed)
                    assert capsule.to_bytes() == changed
                    journal.emit(
                        "restore-attempt", corruption=corruption, original=raw, changed=changed
                    )
                    with pytest.raises(ValueError) as caught:
                        runtime.restore(capsule, expected_revision=revision)
                    journal.emit("restore-rejected", error=repr(caught.value))
                    retained(runtime, before, raw, revision, journal)
                retry(runtime, clone, mj, model, before, journal)


def capacity_starts(mj, model, starts):
    from phymeo.backends.mujoco_runtime import MuJoCoInitialState

    out = dict(starts)
    first = out[case.KEYS[0]]
    q = first.q.copy()
    q[2] += 1.0
    joint = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, "hinge")
    q[model.jnt_qposadr[joint]] = 1 / 16
    out[case.KEYS[0]] = MuJoCoInitialState(q, first.v, act=first.act)
    return out


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("kind", ("contact", "constraint"))
def test_ball_scalar_actual_capacity_exact_overflow_and_fresh_owner_successor(kind, mode, tmp_path):
    import mujoco as mj

    from phymeo.backends.mujoco_runtime import MuJoCoRuntime
    from phymeo.builders.mujoco_compiler import compile_mujoco_xml
    from phymeo.training.resident_stepper import PhysicsStepFailure

    name = "mixed-normal-filter"
    xml = case.source(name)
    model = mj.MjModel.from_xml_string(xml)
    starts = capacity_starts(
        mj, model, {k: case.initial(mj, model, name, i) for i, k in enumerate(case.KEYS)}
    )
    packets = {k: case.control(model, 0, i, name) for i, k in enumerate(case.KEYS)}
    oracle = {k: case.native_fields(mj, model, state, packets[k]) for k, state in starts.items()}
    contacts = max(len(x["contacts"]) for x in oracle.values())
    rows = max(len(x["rows"]) for x in oracle.values())
    assert contacts == 1 and len(oracle[case.KEYS[0]]["contacts"]) == 0
    assert len(oracle[case.KEYS[0]]["rows"]) < rows
    caps = dict(contact_capacity=contacts, constraint_capacity=rows)
    compiled = compile_mujoco_xml(xml, require_asset=False)
    with case.saved(tmp_path / "actual-capacity.jsonl.gz") as journal:
        journal.emit(
            "predeclared-native-counts", source=xml, initial=starts, native=oracle, exact=caps
        )
        small = dict(caps)
        small[kind + "_capacity"] -= 1
        failed = MuJoCoRuntime(
            compiled,
            mode=mode,
            implementation="tyx",
            world_keys=case.KEYS,
            initial_states=starts,
            **small,
        )
        with case.owned(failed, journal):
            failed.set_controls(packets)
            before = failed.integration_snapshot()
            raw = failed.checkpoint().to_bytes()
            revision = failed.checkpoint_revision
            journal.emit("overflow-attempt", capacities=small, state=before)
            with pytest.raises(PhysicsStepFailure) as caught:
                failed.step(case.H)
            journal.emit(
                "overflow-return", error=repr(caught.value), scratch=failed._device.download()
            )
            assert caught.value.accepted_worlds == dict.fromkeys(case.KEYS, False)
            retained(failed, before, raw, revision, journal)
        fresh = MuJoCoRuntime(
            compiled,
            mode=mode,
            implementation="tyx",
            world_keys=case.KEYS,
            initial_states=starts,
            **caps,
        )
        with case.owned(fresh, journal):
            assert fresh.capabilities["conservative_efc_bound"] > rows
            case.advance(
                fresh,
                mj,
                model,
                packets,
                case.references(mj, model, fresh.integration_snapshot()),
                journal,
                "exact-capacity-real-successor",
            )


def budget_source():
    # Exact T4 deep/shallow slider declaration, plus a disconnected masked BALL.
    joints = "".join(
        f'<joint name="j{i}" type="slide" axis="{axis}" armature="0" damping="0" frictionloss="0.0"'
        + (' limited="true" range="-.2 .2" margin=".01"' if i == 0 else "")
        + "/>"
        for i, axis in enumerate(("1 0 0", "0 1 0", "0 0 1"))
    )
    return f"""<mujoco model="t4-mixed"><compiler angle="radian" autolimits="true" fusestatic="false"/>
      <option timestep="0.001" gravity="0 0 -9.81" integrator="Euler" solver="Newton" iterations="1" tolerance="1e-10" jacobian="dense" cone="pyramidal"/>
      <default><geom condim="1" friction=".6 .02 .003" solref=".02 1" solimp=".8 .9 .002 .5 2"/><joint armature=".01" damping=".04"/></default>
      <worldbody><geom name="floor" type="plane" size="2 2 .1"/>
      <body name="root" pos="0 0 0.099">{joints}<inertial pos="0 0 0" mass="1.0" fullinertia="0.04 0.05 0.06 0.001 -0.0005 0.0015"/>
      <geom name="deep" type="sphere" size=".1" pos="-.12 0 -.008"/><geom name="shallow" type="sphere" size=".1" pos=".12 0 .0009"/></body>
      <body name="uncoupled-ball" pos="2 0 1"><joint name="ball" type="ball" limited="false"/>{case.INERTIA}<geom type="sphere" size=".035" contype="0" conaffinity="0"/></body>
      </worldbody></mujoco>"""


@pytest.mark.parametrize("mode", MODES)
def test_ball_scalar_source_budget_one_needs_reduced_face_and_actual_retry(mode, tmp_path):
    import mujoco as mj

    from phymeo.backends.mujoco_runtime import MuJoCoControl, MuJoCoInitialState, MuJoCoRuntime
    from phymeo.builders.mujoco_compiler import compile_mujoco_xml
    from phymeo.training.resident_stepper import PhysicsStepFailure

    xml = budget_source()
    model = mj.MjModel.from_xml_string(xml)
    q0 = model.qpos0.copy()
    raised = q0.copy()
    raised[2] += 0.1
    starts = {
        case.KEYS[0]: MuJoCoInitialState(raised, np.zeros(model.nv)),
        case.KEYS[1]: MuJoCoInitialState(q0, np.zeros(model.nv)),
    }
    zero = MuJoCoControl(
        np.zeros(model.nu), qfrc_applied=np.zeros(model.nv), xfrc_applied=np.zeros((model.nbody, 6))
    )
    oracle = case.native_rows(
        mj, model, q0, np.zeros(model.nv), np.zeros(model.na), zero, tiny=True
    )
    assert len(oracle["rows"]) == 2
    unrestricted = np.linalg.solve(oracle["matrix"], -oracle["linear"])
    reduced = case.tiny_box_qp(
        oracle["matrix"], oracle["linear"], oracle["lower"], np.full(2, np.inf)
    )
    assert (
        unrestricted[0] > 0.01 and unrestricted[1] < -0.01 and reduced[0] > 0.01 and reduced[1] == 0
    )
    with case.saved(tmp_path / "budget.jsonl.gz") as journal:
        journal.emit(
            "budget-oracle",
            source=xml,
            starts=starts,
            unrestricted=unrestricted,
            reduced=reduced,
            native=oracle,
        )
        runtime = MuJoCoRuntime(
            compile_mujoco_xml(xml, require_asset=False),
            mode=mode,
            implementation="tyx",
            world_keys=case.KEYS,
            initial_states=starts,
        )
        with case.owned(runtime, journal):
            runtime.set_controls(dict.fromkeys(case.KEYS, zero))
            before = runtime.integration_snapshot()
            raw = runtime.checkpoint().to_bytes()
            revision = runtime.checkpoint_revision
            with case.owned(runtime.clone(), journal) as clone:
                with pytest.raises(PhysicsStepFailure) as caught:
                    runtime.step(case.H)
                scratch = runtime._device.download()
                journal.emit("budget-failed", error=repr(caught.value), scratch=scratch)
                assert scratch["status"][-1] == 7 and scratch["solver_diagnostics"][-1, 0] == 1
                assert caught.value.accepted_worlds == dict.fromkeys(case.KEYS, False)
                retained(runtime, before, raw, revision, journal)
                force = np.zeros(model.nv)
                force[2] = 256.0
                recovery = MuJoCoControl(
                    np.zeros(model.nu), qfrc_applied=force, xfrc_applied=np.zeros((model.nbody, 6))
                )
                packets = dict.fromkeys(case.KEYS, recovery)
                actual = case.advance(
                    runtime,
                    mj,
                    model,
                    packets,
                    case.references(mj, model, before),
                    journal,
                    "budget-one-recovery",
                )
                expected = case.advance(
                    clone,
                    mj,
                    model,
                    packets,
                    case.references(mj, model, before),
                    journal,
                    "budget-one-clone",
                )
                case.exact(actual, expected, "budget unchanged actual retry")
