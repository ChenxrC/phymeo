"""S3 tests first: declared assets to the accepted device boundary source.

No geometry producer or FK implementation is an oracle. Independent references
retain the original material literals and full finite-triangle candidate set.
Parent alone executes this authored matrix.
"""

import gzip
import importlib.util
import json
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import fields, is_dataclass, replace
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

import phymeo as ph
from phymeo.builders._robot_asset import LinkCollision
from phymeo.builders.floating_urdf import FloatingRobotAsset

H = 1 / 1024
BUDGET = 64 * 1024 * 1024
FAMILIES = ("ordinary", "affine", "floating", "floating_affine")
WORLDS = (83, 17)


@lru_cache(maxsize=None)
def reference(name):
    path = Path(__file__).parent / "reference_newton/rr" / (name + ".py")
    spec = importlib.util.spec_from_file_location("s3_" + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def encode(value):
    if isinstance(value, np.ndarray):
        return dict(dtype=value.dtype.str, shape=value.shape, bytes_hex=value.tobytes().hex())
    if isinstance(value, np.generic):
        return encode(value.item())
    if isinstance(value, bytes):
        return dict(bytes_hex=value.hex())
    if type(value) is object:
        return dict(opaque_identity=id(value))
    if is_dataclass(value):
        return {f.name: encode(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Mapping):
        return dict(mapping_entries=[[encode(k), encode(v)] for k, v in value.items()])
    if isinstance(value, (tuple, list)):
        return [encode(v) for v in value]
    if type(value) is float and not np.isfinite(value):
        return dict(nonfinite=repr(value))
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise TypeError("unrecorded S3 evidence: " + type(value).__name__)


def canonical(value):
    return json.dumps(encode(value), sort_keys=True, allow_nan=False)


@contextmanager
def saved(path, identity, mode, declaration):
    record = dict(schema="participant-scene-boundary-v1", identity=identity, mode=mode,
                  declaration=declaration, attempts=[], complete=False)
    try:
        yield record
        record["complete"] = True
    except BaseException as error:
        record["failure"] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        with gzip.open(path / (identity + ".json.gz"), "wt", encoding="utf-8") as stream:
            json.dump(encode(record), stream, sort_keys=True, allow_nan=False)


def free_asset(*, empty=False, masked=False):
    source = ph.CollisionKey(0, 42, 71)
    shape = ph.CollisionShape(source, "capsule", (40, 41), radius=1 / 8,
                              revision=3, mask=0 if masked else (1 << 31) - 1)
    local = ((0.0, 0.0, 0.0), (0.25, 0.0, 0.0))
    collider = LinkCollision("free-tool", "root", -1, (0., 0., 0.), (0., 0., 0., 1.),
                             shape, local, True)
    model = ph.FloatingRobotModel(
        ph.RigidBodyModel(np.array((0.5,)), np.full((1, 3), 1 / 8), np.zeros(3)),
        np.zeros(3))
    asset = FloatingRobotAsset("s3-free-tool", "root", model, (), (), (), (), (), (),
                               () if empty else (collider,),
                               ph.CollisionModel(() if empty else (shape,)))
    return asset, np.asarray(local)


def assembly(path, *, families=("ordinary",), worlds=(83,), reverse=False,
             folded=False, cycle=False, empty=False, masked=False, response="none"):
    ref = reference("resident_scene_boundary_reference")
    rows, definitions = [], {}
    for wi, world in enumerate(worlds):
        frame = ref.CYCLE_R if cycle else np.eye(3)
        base = np.array((1 + 2 * wi, -0.75, 0.125))
        for index, family in enumerate(families):
            pid = ref.PIDS[index]
            key = ph.ResidentParticipantKey(world, pid)
            floating = family.startswith("floating")
            anchor = base + frame @ np.array((index / 8, 0., 0.))
            source, desc = ref.write_asset(path / f"{world}-{pid}", anchor=anchor,
                                          floating=floating, cycle=cycle,
                                          folded=folded and index == 0)
            raw = (ph.load_floating_mimic_urdf(source, gravity=np.zeros(3),
                                              collapse_fixed_joints=folded and index == 0)
                   if floating else ph.load_mimic_urdf(source, gravity=np.zeros(3)))
            asset = raw if "affine" in family else raw.tree
            full = raw.tree
            n = 1 if "affine" in family else 2
            q = np.array((1 / 512, 0.004 - 1 / 512))[:n]
            v = np.array((1 / 1024, -1 / 1024))[:n]
            if floating:
                root = np.r_[anchor, (0.5, 0.5, 0.5, 0.5) if cycle else (0., 0., 0., 1.)]
                state = ph.FloatingRobotState(np.r_[root, q],
                    np.r_[frame @ np.array((1 / 128, 0., 0.)), np.zeros(3), v])
            else:
                state = ph.RobotState(q, v)
            joint = ph.RobotCommand(q, np.zeros(n), feedforward=np.full(n, 1 / 64))
            command = (ph.FloatingRobotCommand(joint,
                ph.FloatingRobotInputs(np.zeros(2),
                    np.tile(frame @ np.array((1 / 128, 0., 0.)), (3, 1)), np.zeros((3, 3))))
                       if floating else joint)
            body_ids = tuple(100 * pid + i for i in range(2 + int(floating)))
            mapping = {}
            desc.update(family=family)
            for collider in full.collisions:
                which = "driver" if collider.name == "surface" else "follower"
                group = int(which == "follower") + int(floating)
                target = ph.CollisionKey(world, body_ids[group], 0)
                mapping[collider.shape.key] = target
                definitions[target] = dict(owner=key, family=family, description=desc,
                    which=which, local=ref.local_surface(desc["folded"])
                    if which == "driver" else ref.ROUND,
                    shape=replace(collider.shape, key=target))
            rows.append(ph.ParticipantArticulation(key, asset, state,
                kp=np.zeros(n), kd=np.zeros(n), effort_limit=np.full(n, 1 / 16),
                body_ids=body_ids, collider_keys=mapping, initial_control=command))
        key = ph.ResidentParticipantKey(world, 97)
        asset, local = free_asset(empty=empty, masked=masked)
        position = base + frame @ np.array((0.125, 0.5, 0.5))
        state = ph.FloatingRobotState(
            np.r_[position, (0.5, 0.5, 0.5, 0.5) if cycle else (0., 0., 0., 1.)],
            np.r_[frame @ np.array((-1 / 64, 0., 0.)), np.zeros(3)])
        mapping = {}
        if not empty:
            source = asset.collisions[0].shape
            target = ph.CollisionKey(world, 9700, 0)
            mapping[source.key] = target
            definitions[target] = dict(owner=key, family="free", which="root", local=local,
                description=dict(anchor=position, cycle=cycle, folded=False),
                shape=replace(source, key=target))
        rows.append(ph.ParticipantArticulation(key, asset, state,
            kp=np.zeros(0), kd=np.zeros(0), effort_limit=np.zeros(0),
            body_ids=(9700,), collider_keys=mapping,
            initial_control=ph.FloatingRobotCommand(None,
                ph.FloatingRobotInputs(np.zeros(0),
                    np.asarray([frame @ np.array((1 / 64, 0., 0.))]), np.zeros((1, 3))))))
    if reverse:
        rows.reverse()
    declaration = ph.ParticipantSceneDeclaration(tuple(rows), contact_response=response)
    return declaration, definitions


def prepare(scene, **options):
    return scene.prepare_boundary_query(
        limits=options.get("limits", ph.BoundaryDistanceLimits()),
        memory_limit_bytes=options.get("memory_limit_bytes", BUDGET))


def query(scene, plan, record):
    attempt = dict(operation="query", before=scene.snapshot(), stats=plan.stats())
    record["attempts"].append(attempt)
    try:
        result = plan.query(margin=0.5)
        attempt["result"] = result
        return result
    finally:
        attempt.update(after=scene.snapshot(), last_attempt=plan.last_attempt,
                       final_stats=plan.stats())
        assert canonical(attempt["before"]) == canonical(attempt["after"])


def step(scene, record):
    attempt = dict(operation="step", before=scene.snapshot(), h=H)
    record["attempts"].append(attempt)
    attempt["result"] = scene.step(H)
    attempt["receipts"] = scene.last_accepted
    attempt["status"] = scene.last_status
    attempt["after"] = scene.snapshot()
    assert attempt["result"] == dict.fromkeys(scene.world_keys, True)
    assert all(attempt["after"][w].tick == attempt["before"][w].tick + 1 for w in scene.world_keys)
    assert set(scene.last_accepted) == set(scene.world_keys)
    return attempt["after"]


def check(scene, plan, packet, definitions):
    ref = reference("resident_scene_boundary_reference")
    distance = reference("resident_boundary_source_reference")
    states = scene.snapshot()
    assert packet.world_keys == scene.world_keys
    assert packet.state_epoch == scene.state_epoch
    assert packet.source_kind == "resident-scene-rigid-device-v1"
    assert plan.require_current(packet) is packet
    assert scene.require_boundary_result(packet) is packet
    assert scene.last_boundary_result is packet
    np.testing.assert_array_equal(packet.ticks, [states[w].tick for w in scene.world_keys])
    material = {}
    for key, entry in definitions.items():
        material[key] = ref.points(states[key.world].participants[entry["owner"]],
            entry["family"], entry["description"], entry["which"], entry["local"])
        anchor = np.asarray(packet.source["world_anchors"][key.world])
        length = max(1.0, float(np.max(np.linalg.norm(entry["local"] - entry["local"][0], axis=1))))
        np.testing.assert_allclose(packet.source["gathered_points"][key],
                                   material[key] - anchor, atol=2e-4 * length, rtol=0)
    assert packet.source["participant_keys"] == scene.participant_keys
    assert packet.source["scene_registry_fingerprint"] == scene.layout.fingerprint
    assert len(packet.source["participant_addresses"]) == len(scene.participant_keys)
    assert packet.source["stamps"]["native_poses"].shape == (len(definitions), 12)
    shapes = {key: entry["shape"] for key, entry in definitions.items()}
    count = 0
    for group in packet.groups:
        shape = shapes[group.surface]
        vertices = material[group.surface]
        faces = tuple(tuple(shape.vertex_ids.index(i) for i in f) for f in group.facet_vertices)
        axes = [distance.axis(material[key]) for key in group.round_keys]
        radii = [shapes[key].radius for key in group.round_keys]
        expected = distance.aabb_candidates(vertices, faces, axes, radii, packet.margin)
        assert tuple(map(tuple, group.candidates.tolist())) == expected
        assert group.bvh_raw.shape == (len(axes), (len(faces) + 29) // 30 + 3)
        assert np.all(group.bvh_raw[:, -1] == 2 * len(faces) - 1)
        assert group.packet.source_binding == packet.source_binding
        assert len(group.packet.rows) == len(expected)
        for row, (segment, facet) in zip(group.packet.rows, expected, strict=True):
            assert (row.segment_id, row.facet_id) == (segment, facet)
            assert row.vertex_ids == group.facet_vertices[facet]
            distance.assert_row(row, vertices[list(faces[facet])], axes[segment],
                                radii[segment], group.packet.origin_m)
        count += len(expected)
    assert count == packet.required_candidates
    assert all(pair.first.world == pair.second.world for pair in packet.pairs)
    assert plan.stats()["live_bytes"] == plan.stats()["persistent_bytes"]
    return count


@pytest.mark.tyx
@pytest.mark.parametrize("family", FAMILIES)
def test_declared_four_families_query_after_three_real_steps(family, tyx_mode, tmp_path):
    declaration, definitions = assembly(tmp_path / "assets", families=(family,))
    with saved(tmp_path, family, tyx_mode, declaration) as record:
        with ph.compile_participant_scene(declaration, mode=tyx_mode, cache_root=tmp_path) as scene:
            plan = prepare(scene)
            previous = None
            initial = canonical(scene.snapshot())
            for tick in (1, 2, 3):
                step(scene, record)
                if previous is not None:
                    with pytest.raises(ValueError):
                        scene.require_boundary_result(previous)
                previous = query(scene, plan, record)
                assert check(scene, plan, previous, definitions) > 0
                assert all(s.tick == tick for s in scene.snapshot().values())
            assert canonical(scene.snapshot()) != initial
            assert scene.stats()["contact_response"] == "none"
            record["source_map"] = scene.source_map
        assert plan.stats()["closed"] and plan.stats()["live_bytes"] == 0


@pytest.mark.tyx
@pytest.mark.parametrize("reverse", (False, True))
def test_declared_heterogeneous_sparse_worlds(reverse, tyx_mode, tmp_path):
    declaration, definitions = assembly(tmp_path / "assets", families=FAMILIES,
                                       worlds=WORLDS, reverse=reverse)
    with saved(tmp_path, "heterogeneous-" + str(reverse), tyx_mode, declaration) as record:
        with ph.compile_participant_scene(declaration, mode=tyx_mode, cache_root=tmp_path) as scene:
            plan = prepare(scene)
            for _ in range(3):
                step(scene, record)
                assert check(scene, plan, query(scene, plan, record), definitions) > 0
            assert len(scene.participant_keys) == 10
            assert len(scene.source_map) == 10


@pytest.mark.tyx
@pytest.mark.parametrize("cycle", (False, True))
def test_public_fixed_tool_and_free_object(cycle, tyx_mode, tmp_path):
    declaration, definitions = assembly(tmp_path / "assets", families=("affine",),
                                       worlds=WORLDS, folded=True, cycle=cycle)
    with saved(tmp_path, "tool-object-" + str(cycle), tyx_mode, declaration) as record:
        with ph.compile_participant_scene(declaration, mode=tyx_mode, cache_root=tmp_path) as scene:
            plan = prepare(scene)
            for _ in range(3):
                step(scene, record)
                assert check(scene, plan, query(scene, plan, record), definitions) > 0


@pytest.mark.tyx
def test_masks_and_original_same_body_filters_are_not_law_defaults(tyx_mode, tmp_path):
    declaration, definitions = assembly(tmp_path / "assets", masked=True)
    with saved(tmp_path, "filters", tyx_mode, declaration) as record:
        with ph.compile_participant_scene(declaration, mode=tyx_mode, cache_root=tmp_path) as scene:
            plan = prepare(scene)
            packet = query(scene, plan, record)
            check(scene, plan, packet, definitions)
            for group in packet.groups:
                assert all(key.body != 9700 for key in group.round_keys)
            record["pairs"] = packet.pairs


@pytest.mark.tyx
def test_partial_reset_restore_then_actual_next_step(tyx_mode, tmp_path):
    declaration, definitions = assembly(tmp_path / "assets", families=("affine",),
                                       worlds=WORLDS, folded=True)
    with saved(tmp_path, "reset-restore", tyx_mode, declaration) as record:
        with ph.compile_participant_scene(declaration, mode=tyx_mode, cache_root=tmp_path) as scene:
            plan = prepare(scene)
            step(scene, record)
            checkpoint = scene.snapshot()
            old = query(scene, plan, record)
            scene.reset((17,))
            reset = scene.snapshot()
            assert reset[17].tick == 0
            assert canonical(reset[83]) == canonical(checkpoint[83])
            with pytest.raises(ValueError):
                scene.require_boundary_result(old)
            scene.restore({17: checkpoint[17]}, expected_epoch=scene.state_epoch)
            step(scene, record)
            packet = query(scene, plan, record)
            assert check(scene, plan, packet, definitions) > 0
            assert all(value.tick == 2 for value in scene.snapshot().values())
            record["checkpoint"] = checkpoint


@pytest.mark.tyx
def test_multiple_plans_current_entry_is_last_successful_prepare(tyx_mode, tmp_path):
    declaration, definitions = assembly(tmp_path / "assets")
    with saved(tmp_path, "plans", tyx_mode, declaration) as record:
        with ph.compile_participant_scene(declaration, mode=tyx_mode, cache_root=tmp_path) as scene:
            assert scene.last_boundary_result is None
            with pytest.raises(RuntimeError, match="no boundary"):
                scene.require_boundary_result(None)
            first = prepare(scene)
            old = query(scene, first, record)
            check(scene, first, old, definitions)
            with pytest.raises(MemoryError):
                prepare(scene, memory_limit_bytes=1)
            assert scene.last_boundary_result is old
            second = prepare(scene)
            assert scene.last_boundary_result is None
            assert first.require_current(old) is old
            with pytest.raises(ValueError):
                scene.require_boundary_result(old)
            check(scene, second, query(scene, second, record), definitions)
        assert first.stats()["closed"] and second.stats()["closed"]


@pytest.mark.tyx
@pytest.mark.parametrize("fault", ("packet", "last-read", "source"))
def test_late_query_failure_preserves_new_physics_and_prior_result(fault, tyx_mode, tmp_path, monkeypatch):
    declaration, definitions = assembly(tmp_path / "assets", worlds=WORLDS, folded=True,
                                       families=("affine",))
    with saved(tmp_path, "late-" + fault, tyx_mode, declaration) as record:
        with ph.compile_participant_scene(declaration, mode=tyx_mode, cache_root=tmp_path) as scene:
            plan = prepare(scene)
            old = query(scene, plan, record)
            step(scene, record)
            before = canonical(scene.snapshot())
            read, build = plan._download, plan._build_packet
            fired = []
            def bad_read(array, label):
                value = read(array, label)
                if fault == "last-read" and label == "group:1/meta":
                    fired.append(label)
                    raise RuntimeError("S3 late read")
                if fault == "source" and label == "source/native_poses":
                    fired.append(label)
                    value[-1, 0] = np.nan
                return value
            def bad_build(*args, **kwargs):
                build(*args, **kwargs)
                fired.append("packet")
                raise RuntimeError("S3 late packet")
            with monkeypatch.context() as patch:
                patch.setattr(plan, "_download", bad_read)
                if fault == "packet":
                    patch.setattr(plan, "_build_packet", bad_build)
                with pytest.raises((ValueError, RuntimeError)):
                    query(scene, plan, record)
            assert fired
            assert scene.last_boundary_result is old
            assert canonical(scene.snapshot()) == before
            with pytest.raises(ValueError):
                scene.require_boundary_result(old)
            current = query(scene, plan, record)
            check(scene, plan, current, definitions)
            assert current.sequence == old.sequence + 1


@pytest.mark.tyx
@pytest.mark.parametrize("resource", ("groups", "candidates", "memory"))
def test_resource_refusal_does_not_replace_current_plan(resource, tyx_mode, tmp_path):
    declaration, definitions = assembly(tmp_path / "assets")
    with saved(tmp_path, "resource-" + resource, tyx_mode, declaration) as record:
        with ph.compile_participant_scene(declaration, mode=tyx_mode, cache_root=tmp_path) as scene:
            plan = prepare(scene)
            packet = query(scene, plan, record)
            check(scene, plan, packet, definitions)
            budget = plan.stats()["peak_required_bytes"]
            limits = ph.BoundaryDistanceLimits()
            if resource == "memory":
                small = prepare(scene, memory_limit_bytes=budget - 1)
            else:
                small = prepare(scene, limits=replace(limits, **{resource: 0}))
            with pytest.raises((ph.BoundaryDistanceCapacityError, MemoryError)):
                query(scene, small, record)
            assert small.last_result is None
            assert plan.last_result is packet
            assert plan.require_current(packet) is packet
            assert small.stats()["live_bytes"] == small.stats()["persistent_bytes"]
            assert all(s.tick == 0 for s in scene.snapshot().values())


@pytest.mark.tyx
def test_empty_original_collisions_still_bind_real_ticks(tyx_mode, tmp_path):
    declaration, definitions = assembly(tmp_path / "assets", families=(), worlds=WORLDS, empty=True)
    with saved(tmp_path, "empty", tyx_mode, declaration) as record:
        with ph.compile_participant_scene(declaration, mode=tyx_mode, cache_root=tmp_path) as scene:
            plan = prepare(scene, memory_limit_bytes=1)
            step(scene, record)
            packet = query(scene, plan, record)
            assert check(scene, plan, packet, definitions) == 0
            assert packet.world_keys == (17, 83)
            np.testing.assert_array_equal(packet.ticks, (1, 1))


@pytest.mark.tyx
def test_mode_identity_and_cross_owner_checkpoint_refusal(tyx_mode, tmp_path):
    declaration, _ = assembly(tmp_path / "assets", families=(), empty=True)
    legacy = replace(declaration, contact_response="host_patch")
    assert declaration.identity != legacy.identity
    with ph.compile_participant_scene(declaration, mode=tyx_mode, cache_root=tmp_path / "none") as first:
        with ph.compile_participant_scene(legacy, mode=tyx_mode, cache_root=tmp_path / "host") as second:
            first.step(H)
            checkpoint = first.snapshot()
            before = canonical(second.snapshot())
            with pytest.raises(ValueError, match="another owner"):
                second.restore(checkpoint, expected_epoch=second.state_epoch)
            assert canonical(second.snapshot()) == before
            first_plan = prepare(first, memory_limit_bytes=1)
            second_plan = prepare(second, memory_limit_bytes=1)
            foreign = first_plan.query(margin=0.5)
            second_plan.query(margin=0.5)
            with pytest.raises(ValueError, match="current publication"):
                second.require_boundary_result(foreign)


@pytest.mark.tyx
def test_compiled_close_retains_owner_until_delegated_close_succeeds(tyx_mode, tmp_path, monkeypatch):
    declaration, _ = assembly(tmp_path / "assets", families=(), empty=True)
    scene = ph.compile_participant_scene(declaration, mode=tyx_mode, cache_root=tmp_path)
    plan = prepare(scene, memory_limit_bytes=1)
    original = scene._batch.close
    calls = []
    def pending():
        calls.append("pending")
        raise RuntimeError("S3 delegated close pending")
    try:
        with monkeypatch.context() as patch:
            patch.setattr(scene._batch, "close", pending)
            with pytest.raises(RuntimeError, match="delegated close pending"):
                scene.close()
            assert not scene.stats()["assembly_closed"]
            assert not plan.stats()["closed"]
        scene.close()
        assert calls == ["pending"] and scene.stats()["assembly_closed"]
        assert plan.stats()["closed"] and plan.stats()["live_bytes"] == 0
        with pytest.raises(RuntimeError, match="closed"):
            scene.prepare_boundary_query(limits=ph.BoundaryDistanceLimits(), memory_limit_bytes=1)
    finally:
        original()


@pytest.mark.parametrize("response", ("", "device", None, True, 0))
def test_invalid_response_mode_is_rejected_before_owner(response, tmp_path, monkeypatch):
    from phymeo.backends import resident_scene
    declaration, _ = assembly(tmp_path / "assets", families=(), empty=True)
    def forbidden(*args, **kwargs):
        raise AssertionError("invalid declaration reached runtime allocation")
    monkeypatch.setattr(resident_scene, "ResidentSceneBatch", forbidden)
    with pytest.raises((TypeError, ValueError), match="contact_response"):
        replace(declaration, contact_response=response)


@pytest.mark.parametrize("law", (None, "force"))
def test_none_never_silently_ignores_a_material_entry(law, tmp_path, monkeypatch):
    from phymeo.backends import resident_scene
    declaration, _ = assembly(tmp_path / "assets")
    shapes = [target for row in declaration.articulations for target in row.collider_keys.values()]
    pair = tuple(sorted((shapes[0], shapes[-1])))
    value = None if law is None else ph.RobotContactMaterial(64, 0, 0)
    def forbidden(*args, **kwargs):
        raise AssertionError("nonempty no-response laws reached runtime allocation")
    monkeypatch.setattr(resident_scene, "ResidentSceneBatch", forbidden)
    with pytest.raises(ValueError, match="empty materials_by_pair"):
        replace(declaration, materials_by_pair={pair: value})
