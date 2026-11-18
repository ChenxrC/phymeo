"""B2 short CUDA lifecycle and bilateral motion; no old trajectory rerun."""

import gzip
import json
from collections.abc import Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import fields, is_dataclass, replace
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest
from test_mujoco_area_friction import _close, _equations, _native, _owned, _plain
from test_mujoco_area_friction_live import H, _cases, _published, _surface
from test_mujoco_tyx_area_pressure_device import _bytes_equal, _capture, _host, _outputs
from test_mujoco_tyx_area_pressure_heterogeneous_scene import H as BILATERAL_H
from test_mujoco_tyx_area_pressure_heterogeneous_scene import _declarations as _bilateral_cases

from phymeo.backends.area_friction_readback import AreaFrictionEntryReader
from phymeo.backends.area_pressure_readback import AreaPressureEntryReader
from phymeo.backends.mujoco_area_friction import AreaFrictionPairLaw
from phymeo.backends.mujoco_heterogeneous import MuJoCoTyxHeterogeneousBatch, MuJoCoTyxWorldSpec
from phymeo.backends.mujoco_heterogeneous_area_scene import (
    MuJoCoTyxHeterogeneousAreaPressureScene,
    MuJoCoTyxHeterogeneousAreaSceneCheckpoint,
)
from phymeo.backends.mujoco_runtime import MuJoCoControl
from phymeo.geometry.area_pressure import AreaPressureStamp
from phymeo.training.resident_stepper import PhysicsStepFailure


@contextmanager
def _evidence(path):
    record = dict(schema="mujoco-area-friction-lifecycle-test-v1", records=[])
    try:
        yield record
        record["status"] = "PASS"
    except BaseException as error:
        record.update(status="FAIL", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        with gzip.open(path, "xt", encoding="utf-8") as stream:
            json.dump(_plain(record), stream, allow_nan=False)


def _same(actual, expected, *, fresh_stamp=False):
    """All numerical bytes; only a verified fresh stamp's owner token may vary."""
    if isinstance(expected, np.ndarray):
        assert actual.dtype == expected.dtype and actual.shape == expected.shape
        assert actual.tobytes() == expected.tobytes()
    elif is_dataclass(expected):
        assert type(actual) is type(expected)
        for field in fields(expected):
            if (
                fresh_stamp
                and type(expected) is AreaPressureStamp
                and field.name == "configuration_id"
            ):
                continue
            _same(
                getattr(actual, field.name), getattr(expected, field.name), fresh_stamp=fresh_stamp
            )
    elif isinstance(expected, Mapping):
        assert isinstance(actual, Mapping) and set(actual) == set(expected)
        for key in expected:
            _same(actual[key], expected[key], fresh_stamp=fresh_stamp)
    elif isinstance(expected, (tuple, list)):
        assert type(actual) is type(expected) and len(actual) == len(expected)
        for a, b in zip(actual, expected, strict=True):
            _same(a, b, fresh_stamp=fresh_stamp)
    else:
        assert type(actual) is type(expected) and actual == expected


def _make(stack):
    specs, worlds, assets = _cases()
    worlds += (MuJoCoTyxWorldSpec(11, specs[0].key, worlds[0].initial_state),)
    batch = stack.enter_context(MuJoCoTyxHeterogeneousBatch(specs, worlds, mode="cuda"))
    scene = MuJoCoTyxHeterogeneousAreaPressureScene(batch, assets)
    ws = stack.enter_context(batch.prepare_device_workspace(include_area_pressure=True))
    assert batch.layout[7].bucket == batch.layout[11].bucket
    return batch, scene, ws, specs, worlds, assets


def _step(scene, h=H):
    batch = scene.batch
    configs = {
        key: batch.area_pressure_configuration_id(key)
        for key in batch.world_keys
        if batch.layout[key].model_key != "native"
    }
    assert scene.step(h, expected_revision=batch.checkpoint_revision) == dict.fromkeys(
        batch.world_keys, True
    )
    for key, rows in scene.last_geometry.items():
        for row in rows:
            assert row.surface.stamp.configuration_id == configs[key]
    return batch.integration_snapshot()


def _capture_all(stack, batch):
    histories, copies = {}, {}
    for i, owner in enumerate(batch._buckets):
        history = owner.area_pressure_snapshot()
        assert all(row["source_revision"] == owner.checkpoint_revision for row in history.values())
        histories.update(history)
        binding, arrays = _capture(stack, owner, "cuda")
        friction = bool(owner._area_friction_laws)
        assert (
            binding["schema"] == "mujoco-tyx-area-device-copy-v2"
            if friction
            else binding["schema"] == "mujoco-tyx-area-device-copy-v1"
        )
        assert arrays["area_data"].shape[-1] == (23 if friction else 15)
        assert binding["revision"] == owner.checkpoint_revision
        for row, key in enumerate(owner.world_keys):
            valid = histories[key]["valid"]
            assert arrays["area_meta"][row, 0] == ((2 if friction else 1) if valid else 0)
            if valid:
                pair = histories[key]["pairs"][0]
                n = len(pair["normal_force"])
                expected = np.column_stack(
                    (
                        pair["points"],
                        pair["normals"],
                        pair["stiffness"],
                        pair["surrogate_gap"],
                        pair["normal_force"],
                        pair["force_on_a"],
                        pair["torque_on_a_world_origin"],
                    )
                )
                np.testing.assert_allclose(
                    arrays["area_data"][row, :n, :15], expected, rtol=3e-5, atol=2e-6
                )
                if friction:
                    tail = np.column_stack(
                        (
                            pair["tangential_force"],
                            pair["relative_velocity"],
                            np.full(n, pair["mu"]),
                            np.full(n, pair["slip_speed_m_s"]),
                        )
                    )
                    np.testing.assert_array_equal(
                        arrays["area_data"][row, :n, 15:], tail.astype(np.float32)
                    )
                words = np.ascontiguousarray(arrays["area_measure_words"][row, :n], dtype="<i4")
                assert (
                    words.view("<f8").reshape(n).tobytes()
                    == np.asarray(pair["dA"], dtype="<f8").tobytes()
                )
        copies[i] = dict(binding=binding, arrays=arrays)
    _owned(histories)
    return dict(state=batch.integration_snapshot(), history=histories, copies=copies)


def _same_capture(actual, expected, *, fresh_stamp=False):
    _same(actual["state"], expected["state"], fresh_stamp=fresh_stamp)
    for key, row in expected["history"].items():
        # Each source_revision was separately checked against its actual owner.
        _same(
            {k: v for k, v in actual["history"][key].items() if k != "source_revision"},
            {k: v for k, v in row.items() if k != "source_revision"},
            fresh_stamp=fresh_stamp,
        )
    for bucket, row in expected["copies"].items():
        got = actual["copies"][bucket]
        _bytes_equal(got["arrays"], row["arrays"])
        _same(got["binding"]["catalogs"], row["binding"]["catalogs"], fresh_stamp=fresh_stamp)


def _retained(batch, scene, ws, frame, arrays, before, revision, raw, geometry):
    assert batch.checkpoint_revision == revision and scene.last_geometry is geometry
    _same(batch.integration_snapshot(), before)
    assert scene.checkpoint().to_bytes() == raw
    _bytes_equal(ws.readback(frame), arrays)
    for key in before:
        assert batch.integration_snapshot()[key].last_step is before[key].last_step


@pytest.mark.cuda
def test_same_bucket_selected_reset_restore_and_bitwise_replay(tmp_path):
    with _evidence(tmp_path / "reset-replay.json.gz") as evidence, ExitStack() as stack:
        batch, scene, ws, _, _, assets = _make(stack)
        _step(scene)
        saved = _capture_all(stack, batch)
        capsule = scene.checkpoint()
        restored = MuJoCoTyxHeterogeneousAreaSceneCheckpoint.from_bytes(capsule.to_bytes())
        assert restored.to_bytes() == capsule.to_bytes()
        evidence.update(checkpoint=restored.to_bytes(), saved=saved)
        frame = ws.refresh(expected_revision=batch.checkpoint_revision)
        ws_arrays = ws.readback(frame)
        identities = {
            key: tuple(asset.identity for asset in values) for key, values in assets.items()
        }
        with pytest.raises(ValueError, match="ABI differs"):
            AreaPressureEntryReader(ws, identities)
        decoded = AreaFrictionEntryReader(ws, identities).validate(ws_arrays, frame)
        for key, entry in decoded.items():
            receipt = saved["state"][key].last_step
            raw = receipt.raw_contact_fields
            if "area_friction" in raw:
                rows = raw["area_friction"]
                assert len(entry.samples) == sum(len(row["sample_ids"]) for row in rows)
                for pair_index, row in enumerate(rows):
                    samples = [
                        sample for sample in entry.samples if sample.pair_index == pair_index
                    ]
                    np.testing.assert_array_equal(
                        np.asarray(
                            [sample.tangential_force_on_a_world for sample in samples]
                        ).reshape(-1, 3),
                        row["tangential_force"],
                    )
                    np.testing.assert_array_equal(
                        np.asarray([sample.relative_velocity_world for sample in samples]).reshape(
                            -1, 3
                        ),
                        row["relative_velocity"],
                    )
                    assert sum(sample.area_m2 for sample in samples) == pytest.approx(
                        float(row["dA"].sum()), rel=1e-14
                    )
                    assert all(
                        sample.identity == row["identity"] and sample.stamp == row["stamp"]
                        for sample in samples
                    )
            else:
                assert all(
                    sample.mu is None and sample.relative_velocity_world is None
                    for sample in entry.samples
                )
        evidence["actual_v2_readback"] = decoded
        owner = batch._buckets[batch.layout[7].bucket]
        retained_outputs = _outputs(stack, owner, "cuda")
        owner.copy_area_pressure(retained_outputs, expected_revision=owner.checkpoint_revision)
        retained = _host(retained_outputs, "cuda")
        _step(scene)
        expected = _capture_all(stack, batch)
        evidence["expected"] = expected
        controls = {key: row["current_control"] for key, row in expected["history"].items()}
        controls[11] = MuJoCoControl([], qfrc_applied=[0.001, 0, 0, 0, 0, 0])
        batch.set_controls(controls)
        changed = _capture_all(stack, batch)
        np.testing.assert_array_equal(changed["history"][11]["applied_control"].qfrc_applied, 0)
        np.testing.assert_array_equal(
            changed["history"][11]["current_control"].qfrc_applied, [0.001, 0, 0, 0, 0, 0]
        )
        scene.reset(keys=(7,))
        reset = _capture_all(stack, batch)
        evidence["reset"] = reset
        assert reset["state"][7].tick == 0 and not reset["history"][7]["valid"]
        assert reset["history"][7]["pairs"] == () and reset["state"][7].last_step is None
        for key in (11, 73, 305, 41):
            _same(reset["state"][key], changed["state"][key])
        _bytes_equal(_host(retained_outputs, "cuda"), retained)
        with pytest.raises(ValueError):
            ws.readback(frame)
        with pytest.raises(ValueError):
            ws.refresh(expected_revision=frame.revision)
        owners = tuple(batch._buckets)
        scene.restore(restored, expected_revision=batch.checkpoint_revision)
        assert all(a is not b for a, b in zip(owners, batch._buckets, strict=True))
        assert not scene.last_geometry
        restored_state = _capture_all(stack, batch)
        evidence["restored"] = restored_state
        _same_capture(restored_state, saved)
        restored_frame = ws.refresh(expected_revision=batch.checkpoint_revision)
        _bytes_equal(ws.readback(restored_frame), ws_arrays)
        _step(scene)
        replay = _capture_all(stack, batch)
        evidence["replay"] = replay
        _same_capture(replay, expected, fresh_stamp=True)
        for key in (7, 73, 41, 11):
            a = replay["state"][key].last_step.raw_contact_fields["area_friction"][0]["stamp"]
            b = expected["state"][key].last_step.raw_contact_fields["area_friction"][0]["stamp"]
            assert a.configuration_id != b.configuration_id


@pytest.mark.cuda
def test_rechecksummed_law_load_and_diagnostic_tamper_rejected_before_writes(tmp_path, monkeypatch):
    import phymeo.backends.mujoco_checkpoint as codec

    with _evidence(tmp_path / "capsule-negative.json.gz") as evidence, ExitStack() as stack:
        batch, scene, ws, specs, worlds, assets = _make(stack)
        _step(scene)
        before, revision, geometry = (
            batch.integration_snapshot(),
            batch.checkpoint_revision,
            scene.last_geometry,
        )
        frame = ws.refresh(expected_revision=revision)
        arrays, capsule = ws.readback(frame), scene.checkpoint()
        raw = capsule.to_bytes()
        owner = batch._buckets[batch.layout[7].bucket]
        leaf = owner.checkpoint()
        evidence.update(before=before, capsule=raw, refused=[])
        for fault in ("normal-force", "tangent-force", "output-velocity", "diagnostics"):
            state = dict(leaf._read(leaf.binding))
            saved_worlds = list(state["worlds"])
            world = dict(saved_worlds[0])
            receipt = world["receipt"]
            raw_fields = dict(receipt.raw_contact_fields)
            pairs, solver = (
                list(raw_fields["area_friction"]),
                dict(raw_fields["area_friction_solver"]),
            )
            pair = dict(pairs[0])
            if fault in ("normal-force", "tangent-force"):
                name = "normal_force" if fault == "normal-force" else "tangential_force"
                value = pair[name].copy()
                value.flat[0] += 0.125
                pair[name] = value
            elif fault == "output-velocity":
                value = solver["output_velocity"].copy()
                value[0] += 0.01
                solver["output_velocity"] = value
            else:
                diag = dict(solver["raw_diagnostics"])
                value = diag["meta"].copy()
                value[0] = 99
                diag["meta"] = value
                solver["raw_diagnostics"] = MappingProxyType(diag)
            pairs[0] = MappingProxyType(pair)
            raw_fields.update(
                area_friction=tuple(pairs), area_friction_solver=MappingProxyType(solver)
            )
            world["receipt"] = replace(receipt, raw_contact_fields=MappingProxyType(raw_fields))
            saved_worlds[0] = world
            state["worlds"] = tuple(saved_worlds)
            forged = codec._make_checkpoint(leaf.binding, leaf.source_revision, state)
            with monkeypatch.context() as patch:

                def forbidden(*args, **kwargs):
                    pytest.fail("invalid friction capsule reached replacement writes")

                patch.setattr(owner, "_build_device", forbidden)
                with pytest.raises(ValueError):
                    owner.restore(forged, expected_revision=owner.checkpoint_revision)
            evidence["refused"].append(fault)
            _retained(batch, scene, ws, frame, arrays, before, revision, raw, geometry)
        bad_specs = tuple(
            replace(spec, area_friction_laws=(replace(spec.area_friction_laws[0], mu=0.25),))
            if spec.key == specs[0].key
            else spec
            for spec in specs
        )
        other = stack.enter_context(MuJoCoTyxHeterogeneousBatch(bad_specs, worlds, mode="cuda"))
        wrong = MuJoCoTyxHeterogeneousAreaPressureScene(other, assets)
        original = wrong.checkpoint().to_bytes()
        with monkeypatch.context() as patch:

            def no_clone(*args, **kwargs):
                pytest.fail("different law capsule reached a replacement owner")

            patch.setattr(other, "_clone_bucket", no_clone)
            with pytest.raises(ValueError):
                wrong.restore(capsule, expected_revision=other.checkpoint_revision)
        assert wrong.checkpoint().to_bytes() == original
        _retained(batch, scene, ws, frame, arrays, before, revision, raw, geometry)


@pytest.mark.cuda
def test_late_prepare_is_atomic_but_postcommit_observation_does_not_rollback(tmp_path, monkeypatch):
    with _evidence(tmp_path / "failure-phases.json.gz") as evidence, ExitStack() as stack:
        batch, scene, ws, _, _, _ = _make(stack)
        _step(scene)
        before, revision, geometry = (
            batch.integration_snapshot(),
            batch.checkpoint_revision,
            scene.last_geometry,
        )
        frame = ws.refresh(expected_revision=revision)
        arrays, raw = ws.readback(frame), scene.checkpoint().to_bytes()
        evidence.update(before=before, arrays=arrays, faults=[])
        last = batch._buckets[-1]
        # Real pack/solve have run. Inject their failure status at the narrow
        # boundary, then let apply_load/native stage propagate it normally.
        from phymeo.backends.mujoco_tyx import _area_friction_live_kernels as live_kernels

        apply_load = live_kernels.apply_load
        for fault in ("pack-status", "common-solve-status"):
            touched = []
            with monkeypatch.context() as patch:

                def failed_status(*args, **launch):
                    if args[-1] is last._device.status:
                        device = last._device
                        with device.tx.use_runtime(device.runtime):
                            if fault == "pack-status":
                                status = device._get(device.friction_pack_status)
                                evidence["records"].append(
                                    dict(fault=fault, original=status.copy())
                                )
                                status[:] = 1
                                device._put(device.friction_pack_status, status)
                            else:
                                status = device._get(device.friction_meta)
                                evidence["records"].append(
                                    dict(fault=fault, original=status.copy())
                                )
                                status[:, 0] = 1
                                device._put(device.friction_meta, status)
                        touched.append(True)
                    return apply_load(*args, **launch)

                patch.setattr(live_kernels, "apply_load", failed_status)
                with pytest.raises(PhysicsStepFailure) as rejected:
                    _step(scene)
            assert touched == [True]
            assert rejected.value.accepted_worlds == dict.fromkeys(batch.world_keys, False)
            _retained(batch, scene, ws, frame, arrays, before, revision, raw, geometry)
            evidence["faults"].append(fault)
        original = last._device.download
        seen = []
        with monkeypatch.context() as patch:

            def fail_download():
                original()
                seen.append(True)
                raise RuntimeError("last friction bucket download")

            patch.setattr(last._device, "download", fail_download)
            with pytest.raises(PhysicsStepFailure) as caught:
                _step(scene)
        assert seen == [True] and caught.value.accepted_worlds == dict.fromkeys(
            batch.world_keys, False
        )
        _retained(batch, scene, ws, frame, arrays, before, revision, raw, geometry)
        evidence["faults"].append("last-bucket-download")
        with monkeypatch.context() as patch:

            def fail_commit(operation, publication):
                assert operation == "step"
                raise RuntimeError("whole-batch-before-commit")

            patch.setattr(batch, "_before_commit", fail_commit)
            with pytest.raises(PhysicsStepFailure) as caught:
                _step(scene)
        assert caught.value.accepted_worlds == dict.fromkeys(batch.world_keys, False)
        _retained(batch, scene, ws, frame, arrays, before, revision, raw, geometry)
        evidence["faults"].append("before-all-bucket-commit")
        _step(scene)
        accepted = batch.integration_snapshot()
        physics = scene.checkpoint().to_bytes()
        committed_geometry = scene.last_geometry
        revision = batch.checkpoint_revision
        refresh = ws._area_observer.refresh_bucket
        with monkeypatch.context() as patch:

            def fail_observation(index, owner, destination, options):
                refresh(index, owner, destination, options)
                if index == len(batch._buckets) - 1:
                    raise RuntimeError("postcommit area export")

            patch.setattr(ws._area_observer, "refresh_bucket", fail_observation)
            with pytest.raises(RuntimeError, match="postcommit"):
                ws.refresh(expected_revision=revision)
        assert ws._frame is frame and batch.checkpoint_revision == revision
        assert (
            scene.last_geometry is committed_geometry and scene.checkpoint().to_bytes() == physics
        )
        _same(batch.integration_snapshot(), accepted)
        with pytest.raises(ValueError):
            ws.readback(frame)
        fresh = ws.refresh(expected_revision=revision)  # Retry observation only.
        assert fresh.ticks == tuple(accepted[key].tick for key in batch.world_keys)
        assert batch.checkpoint_revision == revision
        evidence.update(
            accepted=accepted, fresh_arrays=ws.readback(fresh), final=_capture_all(stack, batch)
        )


@pytest.mark.cuda
def test_current_hinge_and_moving_plate_have_full_bilateral_response(tmp_path, monkeypatch):
    import mujoco as mj

    with _evidence(tmp_path / "bilateral-two-step.json.gz") as evidence, ExitStack() as stack:
        declarations, old_worlds, all_assets = _bilateral_cases()
        selected = {"hinge", "native"}
        specs = tuple(
            replace(
                spec,
                area_friction_laws=(
                    AreaFrictionPairLaw(all_assets[spec.key][0].identity, 0.5, 0.001),
                ),
            )
            if spec.key == "hinge"
            else spec
            for spec in declarations
            if spec.key in selected
        )
        worlds = tuple(world for world in old_worlds if world.world in (91, 7))
        assets = {"hinge": all_assets["hinge"]}
        batch = stack.enter_context(MuJoCoTyxHeterogeneousBatch(specs, worlds, mode="cuda"))
        scene = MuJoCoTyxHeterogeneousAreaPressureScene(batch, assets)
        native_spec = next(spec for spec in specs if spec.key == "native")
        native_world = next(world for world in worlds if world.world == 7)
        baseline = stack.enter_context(
            MuJoCoTyxHeterogeneousBatch((native_spec,), (native_world,), mode="cuda")
        )
        spec = next(spec for spec in specs if spec.key == "hinge")

        def no_cpu_step(*args, **kwargs):
            pytest.fail("bilateral oracle cannot advance native MuJoCo")

        monkeypatch.setattr(mj, "mj_step", no_cpu_step)
        for step in range(2):
            before = batch.integration_snapshot()
            _step(scene, BILATERAL_H)
            baseline.step(BILATERAL_H)
            after = batch.integration_snapshot()
            record = dict(step=step, before=before, after=after, geometry=scene.last_geometry)
            evidence["records"].append(record)
            _same(after[7], baseline.integration_snapshot()[7])
            receipt = after[91].last_step
            rows, solver, problem = _published(receipt)
            row = rows[0]
            record["surface"] = _surface(row, scene.last_geometry[91][0], receipt)
            native = _native(
                spec, receipt, (SimpleNamespace(identity=row["identity"], points=row["points"]),)
            )
            record["native"] = native
            _close(problem.mass_matrix, native["mass"], 2e-6, 3e-4)
            _close(row["jacobian_a"], native["ja"], 2e-6, 2e-5)
            _close(row["jacobian_b"], native["jb"], 2e-6, 2e-5)
            assert np.linalg.norm(row["jacobian_a"]) > 0 and np.linalg.norm(row["jacobian_b"]) > 0
            assert len(row["normal_force"]) > 0 and np.sum(row["normal_force"]) > 0
            e = _equations(problem, after[91].v)
            record["equations"] = e
            assert e["residual_norm"] <= 2e-5 * e["scale"]
            _close(row["force_on_a"], e["force"], 0.002, 0.002)
            _close(row["tangential_force"], e["tangent"], 0.002, 0.002)
            assert np.linalg.norm(row["tangential_force"]) > 1e-6
            np.testing.assert_array_equal(row["force_on_b"], -row["force_on_a"])
            np.testing.assert_array_equal(
                row["torque_on_b_world_origin"], -row["torque_on_a_world_origin"]
            )
            force = row["force_on_a"].astype(float)
            _close(row["torque_on_a_world_origin"], np.cross(row["points"], force), 1e-5, 0.002)
            qa = np.einsum("nij,ni->j", native["ja"], force)
            qb = np.einsum("nij,ni->j", native["jb"], -force)
            assert np.linalg.norm(qa) > 1e-6 and np.linalg.norm(qb) > 1e-6
            _close(receipt.qfrc_constraint, qa + qb, 0.002, 0.002)
            record.update(generalized_a=qa, generalized_b=qb)
            v = after[91].v
            relative = np.einsum("nij,j->ni", native["ja"] - native["jb"], v)
            np.testing.assert_allclose(
                (qa + qb) @ v, np.sum(force * relative), rtol=1e-12, atol=1e-12
            )
            acceleration = np.linalg.solve(
                native["mass"], native["applied"] + native["passive"] - native["bias"] + qa + qb
            )
            _close(v, before[91].v + problem.h_s * acceleration, 5e-5, 5e-5)
            model = spec.compilation.fresh_native_model()
            q = before[91].q.copy()
            mj.mj_integratePos(model, q, v, problem.h_s)
            np.testing.assert_allclose(after[91].q, q, rtol=0, atol=2e-7)
