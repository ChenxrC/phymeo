"""B2a accepted-entry persistence contracts; author lane is UNEXECUTED.

The diagnostic reader observes accepted device bytes. Native MuJoCo is only an
independent contact oracle; neither receipts nor oracle values produce entry
buffers. Public workspace contact/body/wrench consumption belongs to B2b.
"""

from contextlib import ExitStack

import numpy as np
import pytest
import test_mujoco_tyx_activation as activation
import test_mujoco_tyx_constraints as mixed
import test_mujoco_tyx_forest as forest
import test_mujoco_tyx_forest_composition as composition
import test_mujoco_tyx_normal_contacts as normal
import test_mujoco_tyx_runtime as legacy
import test_mujoco_tyx_tree as tree
from test_mujoco_tyx_checkpoint import exact

from phymeo.backends.mujoco_checkpoint import (
    MuJoCoTyxCheckpoint,
    _accepted_bank,
    _make_checkpoint,
)
from phymeo.backends.mujoco_entry_checkpoint import _accepted_entry
from phymeo.backends.mujoco_heterogeneous import (
    MuJoCoTyxHeterogeneousBatch,
    MuJoCoTyxModelSpec,
    MuJoCoTyxWorldSpec,
)
from phymeo.backends.mujoco_runtime import MuJoCoControl, MuJoCoInitialState, MuJoCoRuntime
from phymeo.builders.mujoco_compiler import compile_mujoco_xml
from phymeo.training.resident_stepper import PhysicsStepFailure

MODES = normal.MODES
H = 0.001
KEYS = (41, 7, 305, 11, 203, 59, 91, 2)
KINDS = dict(zip(KEYS, (
    "legacy-contact", "legacy-empty", "tree", "normal", "mixed1",
    "mixed3", "floating", "multi-free",
), strict=True))
FIELDS = {
    "entry_q", "entry_v", "entry_act", "entry_clock", "entry_meta",
    "entry_contact_ids", "entry_contact_data",
}


def record(owner):
    return {name: value.copy() for name, value in _accepted_entry(owner._device).items()}


def bank(owner):
    return tuple(value.copy() for value in _accepted_bank(owner._device))


def row_bytes(values, index):
    return exact({name: value[index] for name, value in values.items()})


def command(model, tick=0):
    return MuJoCoControl(
        np.array([0.025 * (-1 if (i + tick) % 2 else 1) for i in range(model.nu)]),
        qfrc_applied=np.zeros(model.nv),
        xfrc_applied=np.zeros((model.nbody, 6)),
    )


def runtime(compiled, mode, starts):
    return MuJoCoRuntime(
        compiled, world_keys=tuple(starts), initial_states=starts,
        mode=mode, implementation="tyx",
    )


def declarations(mj, keys):
    sources = {
        "legacy-contact": legacy.source(free=True, angle=True),
        "legacy-empty": legacy.SLIDER,
        "tree": tree.MIXED_XML,
        "normal": normal._two_feet(),
        "mixed1": mixed.slider_xml(dim=1, dry=0.1, limits=True),
        "mixed3": mixed.slider_xml(dim=3, shape="box", dry=0.1, limits=True),
        "floating": forest.source("two-free", object_first=False),
        "multi-free": composition.free_pair_source(),
    }
    specs = tuple(
        MuJoCoTyxModelSpec(KINDS[key], compile_mujoco_xml(sources[KINDS[key]], require_asset=False))
        for key in keys
    )
    models = {spec.key: spec.compilation.fresh_native_model() for spec in specs}
    worlds = []
    for key in keys:
        kind, model = KINDS[key], models[KINDS[key]]
        q, v = model.qpos0.copy(), np.linspace(0.012, -0.009, model.nv)
        act = np.zeros(model.na)
        if kind == "normal":
            joint = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, "ankle")
            q[int(model.jnt_qposadr[joint])] = -0.4  # Candidate toe0 is above the plane.
        elif kind in ("mixed1", "mixed3"):
            q[0], v[:] = 0.195, (0.1, -0.07, -0.01)
        elif kind in ("floating", "multi-free"):
            start = forest.initial(mj, model, contact=kind == "multi-free")
            q, v = start.q.copy(), start.v.copy()
        worlds.append(MuJoCoTyxWorldSpec(
            key, kind, MuJoCoInitialState(q, v, act=act, time=0.125),
        ))
    return specs, tuple(worlds), models, sources


def assert_initial(values):
    assert set(values) == FIELDS
    for value in values.values():
        assert value.tobytes() == np.zeros_like(value).tobytes()


def assert_entry(owner, model, before, old_bank, values):
    """Check the fixed ABI against input banks and the same accepted full receipt."""
    assert set(values) == FIELDS
    count = len(owner.world_keys)
    shapes = {
        "entry_q": (count, max(8, model.nq)),
        "entry_v": (count, max(8, model.nv)),
        "entry_act": (count, max(1, model.na)),
        "entry_clock": (count, 4),
        "entry_meta": (count, 6),
        "entry_contact_ids": (count, 64, 8),
        "entry_contact_data": (count, 64, 20),
    }
    for name, value in values.items():
        assert value.shape == shapes[name]
        assert value.dtype == np.dtype(np.int32 if name in (
            "entry_meta", "entry_contact_ids",
        ) else np.float32)
        assert np.isfinite(value).all()
    assert exact(values["entry_q"]) == exact(old_bank[0])
    assert exact(values["entry_v"]) == exact(old_bank[1])
    expected_act = old_bank[3] if model.na else np.zeros((count, 1), np.float32)
    assert exact(values["entry_act"]) == exact(expected_act)
    new_bank, after = bank(owner), owner.integration_snapshot()
    expected_clock = np.concatenate((old_bank[2], new_bank[2]), axis=1)
    assert exact(values["entry_clock"]) == exact(expected_clock)
    for index, key in enumerate(owner.world_keys):
        start, state = before[key], after[key]
        receipt, contacts = state.last_step, state.last_step.contacts
        np.testing.assert_array_equal(values["entry_meta"][index], [
            1, start.tick, state.tick, len(contacts), sum(c.active for c in contacts), 0,
        ])
        assert state.tick == start.tick + 1 == receipt.tick
        stamp = values["entry_clock"][index]
        assert float(stamp[0]) - float(stamp[1]) == start.time == receipt.input_time
        assert float(stamp[2]) - float(stamp[3]) == state.time == receipt.output_time
        for name, width in (("q", model.nq), ("v", model.nv)):
            expected = np.asarray(getattr(receipt, "input_" + name), np.float32)
            assert exact(values["entry_" + name][index, :width]) == exact(expected)
        if model.na:
            assert exact(values["entry_act"][index, :model.na]) == exact(
                np.asarray(receipt.actuation.input_act, np.float32)
            )
        ids, data = values["entry_contact_ids"][index], values["entry_contact_data"][index]
        for ci, contact in enumerate(contacts):
            g1, g2 = contact.geometry_ids
            expected_ids = np.array([
                1, int(contact.active), g1, g2,
                int(model.geom_bodyid[g1]), int(model.geom_bodyid[g2]),
                int(getattr(contact, "feature_index", 0)), contact.dimension,
            ], dtype=np.int32)
            expected_data = np.concatenate((
                [contact.distance], contact.position, np.asarray(contact.frame).reshape(-1),
                contact.wrench_contact_frame, [contact.include_margin],
            )).astype(np.float32)
            assert exact(ids[ci]) == exact(expected_ids)
            assert exact(data[ci]) == exact(expected_data)
        for padding in (ids[len(contacts):], data[len(contacts):]):
            assert padding.tobytes() == np.zeros_like(padding).tobytes()


def native_contacts(mj, model, start, control):
    data = mj.MjData(model)
    data.qpos[:], data.qvel[:], data.act[:], data.time = start.q, start.v, start.act, start.time
    normal._set(data, control)
    mj.mj_forward(model, data)
    result = []
    for index in range(data.ncon):
        contact, wrench = data.contact[index], np.zeros(6)
        mj.mj_contactForce(model, data, index, wrench)
        result.append(dict(
            native_index=index, geometry_ids=tuple(map(int, contact.geom)),
            dimension=int(contact.dim), active=contact.efc_address >= 0,
            distance=float(contact.dist), position=contact.pos.copy(),
            frame=contact.frame.reshape(3, 3).copy(),
            include_margin=float(contact.includemargin), wrench_contact_frame=wrench,
        ))
    return result


def assert_native(receipt, expected):
    for actual, oracle in zip(
        receipt.contacts, mixed.match_contacts(receipt.contacts, expected), strict=True,
    ):
        assert tuple(actual.geometry_ids) == oracle["geometry_ids"]
        assert actual.dimension == oracle["dimension"] and actual.active == oracle["active"]
        for name in ("distance", "position", "frame", "include_margin"):
            np.testing.assert_allclose(getattr(actual, name), oracle[name], rtol=3e-5, atol=4e-5)
        np.testing.assert_allclose(
            actual.wrench_contact_frame, oracle["wrench_contact_frame"], **normal.FORCE_TOL,
        )


def forbid_entry_upload(patch, owners):
    """Ordinary steps must pack entry on device, not upload HOST receipt arrays."""
    for owner in owners:
        device, original = owner._device, owner._device._put

        def put(target, value, device=device, original=original):
            assert all(target is not array for banks in device.entry.values() for array in banks), (
                "ordinary step uploaded an entry array from HOST"
            )
            return original(target, value)

        patch.setattr(device, "_put", put)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("reverse", (False, True))
def test_w8_routes_input_binding_receipt_bytes_and_w1_permutation(mode, reverse, tmp_path, monkeypatch):
    mj = normal._deps()
    keys = KEYS[::-1] if reverse else KEYS
    specs, worlds, models, sources = declarations(mj, keys)
    history = []
    with ExitStack() as stack:
        batch = stack.enter_context(MuJoCoTyxHeterogeneousBatch(specs, worlds, mode=mode))
        singles = {
            world.world: stack.enter_context(runtime(
                next(s.compilation for s in specs if s.key == world.model_key), mode,
                {world.world: world.initial_state},
            )) for world in worlds
        }
        for owner in (*batch._buckets, *singles.values()):
            assert_initial(record(owner))
        for tick in range(3):
            before = [owner.integration_snapshot() for owner in batch._buckets]
            old_banks = [bank(owner) for owner in batch._buckets]
            unchanged = [record(owner) for owner in batch._buckets]
            packets = {key: command(models[KINDS[key]], tick) for key in keys}
            batch.set_controls(packets)
            for owner, old in zip(batch._buckets, unchanged, strict=True):
                assert exact(record(owner)) == exact(old)  # Controls-only CAS is not physics history.
            for key, owner in singles.items():
                owner.set_controls({key: packets[key]})
            with monkeypatch.context() as patch:
                forbid_entry_upload(patch, (*batch._buckets, *singles.values()))
                assert batch.step(H) == dict.fromkeys(keys, True)
                for key, owner in singles.items():
                    assert owner.step(H) == {key: True}
            values = [record(owner) for owner in batch._buckets]
            states = batch.integration_snapshot()
            oracles = {
                key: native_contacts(
                    mj, models[KINDS[key]], before[batch.layout[key].bucket][key], packets[key],
                ) for key in keys
            }
            history.append(dict(
                sources=sources if tick == 0 else None, before=before, input_banks=old_banks,
                receipt=batch.last_step, entries=values, native_contacts=oracles,
                singles={key: record(owner) for key, owner in singles.items()},
            ))
            activation.save(tmp_path / "entry-w8.json.gz", history)
            for index, owner in enumerate(batch._buckets):
                model = models[KINDS[owner.world_keys[0]]]
                assert_entry(owner, model, before[index], old_banks[index], values[index])
            for key, owner in singles.items():
                layout, reference = batch.layout[key], owner.integration_snapshot()[key]
                assert row_bytes(values[layout.bucket], layout.row) == row_bytes(record(owner), 0)
                for name in ("q", "v", "act", "ctrl", "time", "tick"):
                    assert exact(getattr(states[key], name)) == exact(getattr(reference, name))
                assert_native(states[key].last_step, oracles[key])
            if tick == 0:
                normal_row = states[11].last_step
                assert len(normal_row.contacts) == 1
                assert normal_row.contacts[0].geometry_ids[-1] == mj.mj_name2id(
                    models["normal"], mj.mjtObj.mjOBJ_GEOM, "toe1",
                )
                for key, dimension in ((203, 1), (59, 3)):
                    row = states[key].last_step
                    assert row.contacts and all(c.dimension == dimension for c in row.contacts)
                    assert {1, 3} <= {r.constraint_type for r in row.constraints}
                    assert len(row.constraints) > len(row.contacts)
                assert np.linalg.norm(states[59].last_step.contacts[0].wrench_contact_frame[1:3]) > 1e-6
                assert len(states[2].last_step.contacts) == 1
                assert states[2].last_step.contacts[0].normal_force > 0.01
                assert all(not states[key].last_step.contacts for key in (7, 305, 91))
        # Restore all leaf layouts, including nonempty contacts, through the
        # public heterogeneous capsule and reproduce the next accepted record.
        cp = batch.checkpoint()
        (tmp_path / "entry-w8.checkpoint").write_bytes(cp.to_bytes())
        checkpoint_entries = [record(owner) for owner in batch._buckets]
        checkpoint_banks = [bank(owner) for owner in batch._buckets]
        checkpoint_states = batch.integration_snapshot()
        assert batch.step(H) == dict.fromkeys(keys, True)
        next_entries = [record(owner) for owner in batch._buckets]
        next_banks, next_states = [bank(owner) for owner in batch._buckets], batch.integration_snapshot()
        batch.restore(cp, expected_revision=batch.checkpoint_revision)
        assert exact([record(owner) for owner in batch._buckets]) == exact(checkpoint_entries)
        assert exact([bank(owner) for owner in batch._buckets]) == exact(checkpoint_banks)
        assert exact(batch.integration_snapshot()) == exact(checkpoint_states)
        assert batch.step(H) == dict.fromkeys(keys, True)
        repeated = [record(owner) for owner in batch._buckets]
        history.append(dict(checkpoint_entries=checkpoint_entries, next_entries=next_entries,
                            repeated_entries=repeated, repeated_receipt=batch.last_step))
        activation.save(tmp_path / "entry-w8.json.gz", history)
        assert exact(repeated) == exact(next_entries)
        assert exact([bank(owner) for owner in batch._buckets]) == exact(next_banks)
        assert exact(batch.integration_snapshot()) == exact(next_states)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("dimension", (4, 6))
def test_true_contact_couple_uses_compacted_contact_index_and_candidate_slot(mode, dimension, tmp_path):
    mj = normal._deps()
    # Reuse the original two-sphere fixture and budget. A small root tilt raises
    # sphere a above the plane while b remains loaded: contact 0 is candidate 1.
    xml = mixed.free_xml(dim=dimension, offset=False, two=True)
    compiled = compile_mujoco_xml(xml, require_asset=False)
    model = compiled.fresh_native_model()
    q = model.qpos0.copy()
    q[3:7] = (np.cos(0.01), 0, np.sin(0.01), 0)
    start = MuJoCoInitialState(q, [0.02, -0.01, -0.005, 1.5, 0, 2], time=0.125)
    history = []
    with runtime(compiled, mode, {41: start}) as owner:
        for tick in range(2):
            before, old_bank = owner.integration_snapshot(), bank(owner)
            packet = command(model, tick)
            oracle = native_contacts(mj, model, before[41], packet)
            owner.set_controls({41: packet})
            assert owner.step(H) == {41: True}
            values, state = record(owner), owner.integration_snapshot()[41]
            history.append(dict(source=xml, input=before, input_bank=old_bank, entry=values,
                                receipt=state.last_step, oracle=oracle))
            activation.save(tmp_path / "entry-couple.json.gz", history)
            assert_entry(owner, model, before, old_bank, values)
            assert_native(state.last_step, oracle)
            row = state.last_step
            assert len(row.contacts) == 1 and row.contacts[0].dimension == dimension
            assert int(row.raw_contact_fields["contact_map"][0, 0]) == 1
            assert row.contacts[0].geometry_ids[-1] == mj.mj_name2id(
                model, mj.mjtObj.mjOBJ_GEOM, "b",
            )
            assert np.linalg.norm(row.contacts[0].wrench_contact_frame[3:dimension]) > 1e-6


@pytest.mark.parametrize("mode", MODES)
def test_active_inactive_empty_records_and_same_bucket_partial_reset(mode, tmp_path):
    normal._deps()
    compiled = compile_mujoco_xml(normal._slider_pair(z=".11171875", inactive=True), require_asset=False)
    model = compiled.fresh_native_model()
    keys = (41, 7, 305)
    starts = {key: MuJoCoInitialState([offset], [0], time=0.125)
              for key, offset in zip(keys, (-0.02, 0, 0.1), strict=True)}
    with runtime(compiled, mode, starts) as owner:
        assert_initial(record(owner))
        before, old_bank = owner.integration_snapshot(), bank(owner)
        owner.set_controls({key: command(model) for key in keys})
        assert owner.step(H) == dict.fromkeys(keys, True)
        accepted, state = record(owner), owner.integration_snapshot()
        activation.save(tmp_path / "entry-validity.json.gz", dict(
            initial=before, input_bank=old_bank, accepted=accepted, states=state,
        ))
        assert_entry(owner, model, before, old_bank, accepted)
        np.testing.assert_array_equal(accepted["entry_meta"][:, [0, 3, 4]], [
            [1, 2, 2], [1, 2, 0], [1, 0, 0],
        ])
        assert not np.any(accepted["entry_contact_data"][1, :, 13:19])
        assert np.all(accepted["entry_contact_data"][1, :2, 19] > 0)
        old_device, raw = owner._device, bank(owner)
        owner.reset_states({7: MuJoCoInitialState([0.1], [0])}, expected_revision=owner.checkpoint_revision)
        reset = record(owner)
        assert owner._device is not old_device
        for index in (0, 2):
            assert row_bytes(reset, index) == row_bytes(accepted, index)
            for previous, current in zip(raw, bank(owner), strict=True):
                assert exact(current[index]) == exact(previous[index])
            assert exact(owner.integration_snapshot()[keys[index]]) == exact(state[keys[index]])
        assert_initial({name: value[1:2] for name, value in reset.items()})
        assert owner.integration_snapshot()[7].tick == 0
        before, old_bank = owner.integration_snapshot(), bank(owner)
        assert owner.step(H) == dict.fromkeys(keys, True)
        assert_entry(owner, model, before, old_bank, record(owner))
        np.testing.assert_array_equal(record(owner)["entry_meta"][:, :3], [[1, 1, 2], [1, 0, 1], [1, 1, 2]])
        owner.reset()
        assert_initial(record(owner))


@pytest.mark.parametrize("mode", MODES)
def test_activation_unordered_clone_raw_clock_checkpoint_and_legacy_migration(mode, tmp_path, monkeypatch):
    mj = normal._deps()
    compiled = compile_mujoco_xml(composition.constrained_source(), require_asset=False)
    model, keys = compiled.fresh_native_model(), (41, 7, 305)
    starts = {}
    for index, key in enumerate(keys):
        start = forest.initial(mj, model, slot=index)
        q = start.q.copy()
        q[0] = 0.0195
        starts[key] = MuJoCoInitialState(q, start.v, act=[0.07 + index * 0.01], time=0.125)
    history = []
    with runtime(compiled, mode, starts) as owner:
        for tick in range(3):
            before, old_bank = owner.integration_snapshot(), bank(owner)
            owner.set_controls({key: command(model, tick + index) for index, key in enumerate(keys)})
            assert owner.step(H) == dict.fromkeys(keys, True)
            history.append(dict(input=before, input_bank=old_bank, entry=record(owner),
                                states=owner.integration_snapshot()))
            activation.save(tmp_path / "entry-lifecycle.json.gz", history)
            assert_entry(owner, model, before, old_bank, record(owner))
        accepted, raw, states = record(owner), bank(owner), owner.integration_snapshot()
        assert np.any(raw[2][:, 1] != 0)  # Preserve compensation, not only decoded float64 time.
        assert np.any(accepted["entry_clock"][:, [1, 3]] != 0)
        assert not np.array_equal(accepted["entry_act"], raw[3])
        with owner.clone((305, 41)) as clone:
            assert clone.world_keys == (305, 41)
            cloned, cloned_bank = record(clone), bank(clone)
            history.append(dict(clone_entry=cloned, clone_bank=cloned_bank,
                                source_entry=accepted, source_bank=raw))
            activation.save(tmp_path / "entry-lifecycle.json.gz", history)
            for new_index, old_index in enumerate((2, 0)):
                assert row_bytes(cloned, new_index) == row_bytes(accepted, old_index)
                for actual, expected in zip(cloned_bank, raw, strict=True):
                    assert exact(actual[new_index]) == exact(expected[old_index])
                assert exact(clone.integration_snapshot()[keys[old_index]]) == exact(states[keys[old_index]])
            # Source reset must not alias either accepted clone bank.
            owner.reset_states({7: MuJoCoInitialState()}, expected_revision=owner.checkpoint_revision)
            assert exact(record(clone)) == exact(cloned)
            clone_before, clone_input = clone.integration_snapshot(), bank(clone)
            assert clone.step(H) == {305: True, 41: True}
            clone_next = record(clone)
            history.append(dict(clone_input=clone_input, clone_next=clone_next,
                                clone_states=clone.integration_snapshot()))
            activation.save(tmp_path / "entry-lifecycle.json.gz", history)
            assert_entry(clone, model, clone_before, clone_input, clone_next)
        # A new accepted step supplies a valid record for every row after partial reset.
        assert owner.step(H) == dict.fromkeys(keys, True)
        for clone_index, source_index in enumerate((2, 0)):
            assert row_bytes(clone_next, clone_index) == row_bytes(record(owner), source_index)
        held = {key: command(model, 9 + index) for index, key in enumerate(keys)}
        saved_entry = record(owner)
        owner.set_controls(held)
        assert exact(record(owner)) == exact(saved_entry)
        cp = owner.checkpoint()
        checkpoint_path = tmp_path / "accepted-entry.checkpoint"
        checkpoint_path.write_bytes(cp.to_bytes())
        cp = MuJoCoTyxCheckpoint.from_bytes(checkpoint_path.read_bytes())
        expected_entry, expected_bank = record(owner), bank(owner)
        expected_state = owner.integration_snapshot()
        with runtime(compiled, mode, starts) as fresh:
            fresh.restore(cp, expected_revision=fresh.checkpoint_revision)
            assert exact(record(fresh)) == exact(expected_entry)
            assert exact(bank(fresh)) == exact(expected_bank)
            assert exact(fresh.integration_snapshot()) == exact(expected_state)
            history.append(dict(checkpoint_entry=expected_entry, restored_entry=record(fresh),
                                source_revision=cp.source_revision,
                                destination_revision=fresh.checkpoint_revision))
            activation.save(tmp_path / "entry-lifecycle.json.gz", history)
            # Historical physics ticks are independent of this fresh owner's CAS revision.
            assert owner.step(H) == fresh.step(H) == dict.fromkeys(keys, True)
            assert exact(record(fresh)) == exact(record(owner))
            assert exact(bank(fresh)) == exact(bank(owner))
            assert exact(fresh.integration_snapshot()) == exact(owner.integration_snapshot())
            stable_entry, stable_state = record(fresh), fresh.integration_snapshot()
            revision, original_device = fresh.checkpoint_revision, fresh._device
            for field in ("entry_q", "entry_meta"):
                damaged_state = dict(cp._read(cp.binding))
                damaged_entry = {name: value.copy() for name, value in damaged_state["accepted_entry"].items()}
                damaged_entry[field][0, 0 if field == "entry_q" else 1] += 1
                damaged_state["accepted_entry"] = damaged_entry
                damaged = _make_checkpoint(dict(cp.binding), cp.source_revision, damaged_state)
                with monkeypatch.context() as patch:
                    patch.setattr(fresh, "_build_device", lambda *a, **kw: pytest.fail(
                        "damaged entry reached replacement allocation"
                    ))
                    with pytest.raises(ValueError, match="entry"):
                        fresh.restore(damaged, expected_revision=revision)
                assert fresh._device is original_device and fresh.checkpoint_revision == revision
                assert exact(record(fresh)) == exact(stable_entry)
                assert exact(fresh.integration_snapshot()) == exact(stable_state)
            # Explicit synthetic migration fixture: retain all original v1/v2
            # fields and remove only the new extension, not an old producer rerun.
            legacy_binding, legacy_state = dict(cp.binding), dict(cp._read(cp.binding))
            legacy_binding.pop("accepted_entry")
            legacy_state.pop("accepted_entry")
            old_capsule = _make_checkpoint(legacy_binding, cp.source_revision, legacy_state)
            fresh.restore(old_capsule, expected_revision=fresh.checkpoint_revision)
            invalid = record(fresh)
            for name, value in invalid.items():
                expected = np.zeros_like(value)
                if name == "entry_meta":
                    expected[:, 2] = [expected_state[key].tick for key in keys]
                assert exact(value) == exact(expected)
            assert exact(fresh.integration_snapshot()) == exact(expected_state)
            before, old_bank = fresh.integration_snapshot(), bank(fresh)
            assert fresh.step(H) == dict.fromkeys(keys, True)
            assert_entry(fresh, model, before, old_bank, record(fresh))
            assert exact(record(fresh)) == exact(record(owner))
            history.append(dict(legacy_invalid=invalid, next_valid=record(fresh),
                                next_state=fresh.integration_snapshot()))
        activation.save(tmp_path / "entry-lifecycle.json.gz", history)


@pytest.mark.parametrize("mode", MODES)
def test_success_then_failed_advance_and_final_publication_preserve_entry(mode, tmp_path, monkeypatch):
    mj = normal._deps()
    specs, worlds, models, sources = declarations(mj, (41, 91, 2))
    history = []
    with MuJoCoTyxHeterogeneousBatch(specs, worlds, mode=mode) as batch:
        packets = {world.world: command(models[world.model_key]) for world in worlds}
        batch.set_controls(packets)
        assert batch.step(H) == dict.fromkeys(batch.world_keys, True)
        for phase in ("advance", "publication"):
            before, last = batch.integration_snapshot(), batch.last_step
            revision = batch.checkpoint_revision
            accepted = [record(owner) for owner in batch._buckets]
            raw = [bank(owner) for owner in batch._buckets]
            active = [owner._device.active for owner in batch._buckets]
            with monkeypatch.context() as patch:
                if phase == "advance":
                    target = batch._buckets[-1]._device
                    original = target.advance

                    def fail(*args, **kwargs):
                        original(*args, **kwargs)
                        raise RuntimeError("authored failure after final device advance")

                    patch.setattr(target, "advance", fail)
                else:
                    def fail(operation, pending):
                        assert operation == "step"
                        raise RuntimeError("authored rejection after every bucket prepared")

                    patch.setattr(batch, "_before_commit", fail)
                with pytest.raises(PhysicsStepFailure):
                    batch.step(H)
            rejected = [record(owner) for owner in batch._buckets]
            history.append(dict(phase=phase, source=sources, accepted=accepted, rejected=rejected,
                                input_bank=raw, last_receipt=last))
            activation.save(tmp_path / "entry-failures.json.gz", history)
            assert exact(rejected) == exact(accepted)
            assert exact([bank(owner) for owner in batch._buckets]) == exact(raw)
            assert [owner._device.active for owner in batch._buckets] == active
            assert batch.checkpoint_revision == revision and batch.last_step is last
            assert exact(batch.integration_snapshot()) == exact(before)
            assert batch.step(H) == dict.fromkeys(batch.world_keys, True)
            for index, owner in enumerate(batch._buckets):
                subset = {key: before[key] for key in owner.world_keys}
                assert_entry(owner, models[KINDS[owner.world_keys[0]]], subset, raw[index], record(owner))
