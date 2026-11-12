"""Asset-to-participant assembly contracts; tests first, authoring unexecuted.

Only literal ROB1 scenarios and independent reference helpers are reused.
No old test runner or production FK/contact/integrator is an expected result.
"""

import importlib.util
import sys
from contextlib import ExitStack
from dataclasses import FrozenInstanceError, replace
from itertools import combinations, product
from pathlib import Path

import numpy as np
import pytest

import phymeo as ph
from phymeo.backends import _resident_batch_transaction as transaction
from phymeo.backends import resident_scene
from phymeo.builders._robot_asset import LinkCollision, RobotAsset
from phymeo.builders.floating_urdf import FloatingRobotAsset
from phymeo.builders.participant_scene_compiler import (
    ParticipantArticulation,
    ParticipantSceneDeclaration,
    compile_participant_scene,
)
from phymeo.builders.scene_compiler import ArticulationWorld, FiniteStaticShape, SceneDeclaration
from phymeo.dynamics.floating_mimic import FloatingMimicRobotAsset
from phymeo.dynamics.joint_mimic import MimicRobotAsset

H = 1 / 1024
WORLDS = (503, 91)
CASES = ("R01", "R02", "R02-friction", "fixed", "affine", "floating", "floating_affine")


def references():
    """Load existing independent oracle modules, not their pytest entrypoints."""
    base = Path(__file__).parent / "scale"
    modules = []
    for name in ("test_resident_multi_participant", "test_resident_multi_participant_robots"):
        path = base / (name + ".py")
        if name in sys.modules:
            module = sys.modules[name]
            assert Path(module.__file__).resolve() == path.resolve()
        else:
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            try:
                spec.loader.exec_module(module)
            except BaseException:
                sys.modules.pop(name, None)
                raise
        modules.append(module)
    return tuple(modules)


def scenario(case):
    _, ref = references()
    if case == "R01":
        specs = {}
        for pid, side, mass in ((7, -1, 2), (29, 1, 1)):
            s = ref.spec(
                "floating", ref.model([], [], [], [], [], []),
                [side * 7 / 64, 0, 0, 0, 0, 0, 1],
                [-side / 64, 0, 0, 0, 0, 0], root_mass=mass,
            )
            s["shapes"] = [dict(body=0, kind="sphere", center=np.zeros(3), radius=1 / 8)]
            specs[pid] = s
        return dict(name=case, specs=specs, pairs=((7, 0, 29, 0),), stiffness=64.,
                    friction=0., slip_speed=1 / 16, steps=40, switch=None)
    return ref.r02(1 / 4 if case.endswith("friction") else 0) if case.startswith("R02") else ref.r04(case)


def asset_from_spec(s):
    _, ref = references()
    floating = s["family"].startswith("floating")
    n = len(s["model"]["mass"])
    model = ref.robot_model(s["model"]) if n else None
    if floating:
        model = ph.FloatingRobotModel(
            ph.RigidBodyModel(np.array([1 / s["root_mass"]]), s["root_inertia"][None], np.zeros(3)),
            s["root_com"], model,
        )
    colliders = []
    for si, row in enumerate(s["shapes"]):
        group = row["body"]
        link = group - int(floating)
        local = (row["center"][None] if row["kind"] == "sphere" else
                 np.array(list(product((-1, 1), repeat=3))) * row["half"] + row["center"])
        shape = ph.CollisionShape(ph.CollisionKey(0, 42 + si, 71 + si), row["kind"],
                                  tuple(range(31, 31 + len(local))), row.get("radius", 0.), revision=1)
        colliders.append(LinkCollision(
            f"original-{si}", "base" if link < 0 else f"link{link}", link,
            (0., 0., 0.), (0., 0., 0., 1.), shape,
            tuple(tuple(float(x) for x in p) for p in local), True,
        ))
    cls = FloatingRobotAsset if floating else RobotAsset
    asset = cls("assembly-" + s["family"], "base", model,
                tuple(f"link{i}" for i in range(n)), tuple(f"joint{i}" for i in range(n)),
                (-2.,) * n, (2.,) * n, (64.,) * n, (10.,) * n,
                tuple(colliders), ph.CollisionModel(tuple(c.shape for c in colliders)))
    if s["family"] == "affine":
        asset = MimicRobotAsset(asset, "joint0", "joint1", -1., 0.)
    elif s["family"] == "floating_affine":
        asset = FloatingMimicRobotAsset(asset, (("joint0", "joint1", -1., 0.),))
    return asset


def high_control(s, drive=None):
    drive = s["drive"] if drive is None else drive
    joint = ph.RobotCommand(drive[:, 0], drive[:, 1], feedforward=drive[:, 2]) if len(drive) else None
    if s["family"].startswith("floating"):
        return ph.FloatingRobotCommand(joint, ph.FloatingRobotInputs(s["effort"], s["force"], s["torque"]))
    assert not np.any(s["force"]) and not np.any(s["torque"]) and not np.any(s["effort"])
    return joint


def assembly(case="R02", *, worlds=WORLDS, reverse=False):
    data = scenario(case)
    rows, keys, controls = [], {}, {}
    for world in worlds:
        for pid, s in data["specs"].items():
            asset = asset_from_spec(s)
            physical = asset.tree if s["family"] in ("affine", "floating_affine") else asset
            body_ids = tuple(100 * pid + i for i in range(len(s["force"])))
            mapping = {}
            for si, collider in enumerate(physical.collisions):
                target = ph.CollisionKey(world, body_ids[s["shapes"][si]["body"]], 10 + si)
                mapping[collider.shape.key] = target
                keys[world, pid, si] = target
            key = ph.ResidentParticipantKey(world, pid)
            state_cls = ph.FloatingRobotState if s["family"].startswith("floating") else ph.RobotState
            controls[key] = high_control(s)
            rows.append(ParticipantArticulation(
                key, asset, state_cls(s["q"], s["v"]), kp=s["kp"], kd=s["kd"],
                effort_limit=s["caps"], body_ids=body_ids, collider_keys=mapping,
                initial_control=controls[key],
            ))
    wanted = {tuple(sorted((keys[w, a, ai], keys[w, b, bi]))) for w in worlds
              for a, ai, b, bi in data["pairs"]}
    materials = {}
    for left, right in combinations(rows, 2):
        if left.key.world != right.key.world:
            continue
        for a in left.collider_keys.values():
            for b in right.collider_keys.values():
                pair = tuple(sorted((a, b)))
                materials[pair] = (ph.RobotContactMaterial(data["stiffness"], 0, data["friction"], data["slip_speed"])
                                   if pair in wanted else None)
    if reverse:
        rows.reverse()
        materials = dict(reversed(tuple(materials.items())))
    declaration = ParticipantSceneDeclaration(tuple(rows), materials_by_pair=materials,
                                             candidate_capacity=len(wanted), site_capacity=128)
    return declaration, data, dict(keys=keys, worlds=tuple(sorted(worlds))), controls


def capture(scene):
    return dict(states=scene.snapshot(), receipts=scene.last_accepted, status=scene.last_status,
                epoch=scene.state_epoch, bank=scene.stats()["current_buffer"], stats=scene.stats())


def retained(scene, previous):
    assert scene.snapshot() == previous["states"]
    assert scene.last_accepted is previous["receipts"]
    assert scene.last_status is previous["status"]
    assert scene.state_epoch == previous["epoch"]
    assert scene.stats()["current_buffer"] == previous["bank"]


def audited_step(scene, data, built, history):
    _, ref = references()
    before = scene.snapshot()
    assert scene.step(H) == dict.fromkeys(built["worlds"], True)
    after = scene.snapshot()
    checks = {w: ref.audit_receipt(data, built, w, before[w], after[w], scene.last_accepted[w],
                                   ref.drives(data, before[w].tick)) for w in built["worlds"]}
    history.append(dict(entry=before, accepted=after, receipts=scene.last_accepted,
                        status=scene.last_status, independent=checks, epoch=scene.state_epoch,
                        controls={k: p.control for w in after.values() for k, p in w.participants.items()}))


def save_case(path, declaration, data, built, history, scene, failure):
    base, _ = references()
    base.save(path, "participant-assembly", dict(declaration=declaration, inputs=data, identities=built,
                                               source_map=scene.source_map, identity=scene.identity,
                                               history=history, final=capture(scene), failure=failure))

@pytest.mark.tyx
@pytest.mark.parametrize("case", CASES)
def test_original_multicase_asset_lowering_complete_windows(case, tyx_mode, tmp_path):
    declaration, data, built, _ = assembly(case)
    _, ref = references()
    oracle = {pid: (s["q"].copy(), s["v"].copy()) for pid, s in data["specs"].items()}
    history, failure = [], None
    with compile_participant_scene(declaration, mode=tyx_mode, cache_root=tmp_path) as scene:
        try:
            assert scene.world_keys == built["worlds"]
            assert scene.participant_keys == tuple(row.key for row in sorted(declaration.articulations, key=lambda r: r.key))
            for row in declaration.articulations:
                assert scene.source_map[row.key] == row.collider_keys
                assert scene.layout.entries[row.key].family == data["specs"][row.key.participant]["family"]
            for tick in range(data["steps"]):
                terms, wrench, sites = ref.independent_contacts(data, oracle)
                selected = ref.drives(data, tick)
                expected = {pid: ref.advance_one(s, *oracle[pid], terms[pid], wrench[pid], selected[pid])[0]
                            for pid, s in data["specs"].items()}
                audited_step(scene, data, built, history)
                for world in scene.world_keys:
                    for pid, (q, v) in expected.items():
                        actual = scene.snapshot()[world].participants[ph.ResidentParticipantKey(world, pid)]
                        ref.close(actual.position, q, atol=2e-4)
                        ref.close(actual.velocity, v, atol=2e-4)
                history[-1]["continuous_oracle"] = dict(states=expected, sites=sites)
                oracle = expected
        except BaseException as exc:
            failure = dict(type=type(exc).__name__, message=str(exc))
            raise
        finally:
            save_case(tmp_path, declaration, data, built, history, scene, failure)


@pytest.mark.tyx
def test_world_and_registration_permutation_keeps_sources_and_motion(tyx_mode, tmp_path):
    first, data, built, _ = assembly("floating_affine")
    second, _, _, _ = assembly("floating_affine", worlds=WORLDS[::-1], reverse=True)
    with ExitStack() as stack:
        a = stack.enter_context(compile_participant_scene(first, mode=tyx_mode, cache_root=tmp_path / "a"))
        b = stack.enter_context(compile_participant_scene(second, mode=tyx_mode, cache_root=tmp_path / "b"))
        assert a.identity == b.identity
        assert a.source_map == b.source_map
        history = []
        for _ in range(3):
            audited_step(a, data, built, history)
            assert b.step(H) == dict.fromkeys(b.world_keys, True)
            for w in a.world_keys:
                assert a.snapshot()[w].participants == b.snapshot()[w].participants
                for key in a.last_accepted[w].participants:
                    left, right = a.last_accepted[w].participants[key], b.last_accepted[w].participants[key]
                    for field in ("entry_mass", "entry_bias", "entry_linear", "entry_angular", "input_external",
                                  "law_wrench_fp64", "uploaded_wrench_fp32", "contact_generalized"):
                        np.testing.assert_array_equal(getattr(left, field), getattr(right, field))
            history[-1]["permuted"] = capture(b)
        save_case(tmp_path, first, data, built, history, a, None)


@pytest.mark.tyx
def test_checkpoint_replay_selected_reset_and_real_successor(tyx_mode, tmp_path):
    declaration, data, built, controls = assembly("floating_affine")
    history = []
    with compile_participant_scene(declaration, mode=tyx_mode, cache_root=tmp_path) as scene:
        for _ in range(3):
            audited_step(scene, data, built, history)
        checkpoint = scene.snapshot()
        for _ in range(3):
            audited_step(scene, data, built, history)
        expected = scene.snapshot()
        selected, other = scene.world_keys
        scene.restore({selected: checkpoint[selected]}, expected_epoch=scene.state_epoch)
        assert scene.snapshot()[selected] == checkpoint[selected]
        assert scene.snapshot()[other] == expected[other]
        for _ in range(3):
            audited_step(scene, data, built, history)
        assert scene.snapshot()[selected] == expected[selected]
        before = scene.snapshot()
        scene.reset((selected,))
        assert scene.snapshot()[selected].tick == 0
        assert scene.snapshot()[other] == before[other]
        scene.set_controls({k: v for k, v in controls.items() if k.world == selected})
        audited_step(scene, data, built, history)
        assert scene.snapshot()[selected].tick == 1
        history.append(dict(checkpoint=checkpoint, expected=expected, before_reset=before))
        save_case(tmp_path, declaration, data, built, history, scene, None)


@pytest.mark.tyx
@pytest.mark.parametrize("fault", ("missing_participant", "wrong_world", "stale_epoch", "foreign_owner"))
def test_restore_identity_rejects_before_publication(fault, tyx_mode, tmp_path):
    declaration, data, built, _ = assembly("floating_affine")
    history = []
    with ExitStack() as stack:
        scene = stack.enter_context(compile_participant_scene(declaration, mode=tyx_mode, cache_root=tmp_path / "a"))
        audited_step(scene, data, built, history)
        previous = capture(scene)
        world = scene.world_keys[-1]
        point, epoch = previous["states"][world], scene.state_epoch
        if fault == "missing_participant":
            subset = dict(point.participants)
            subset.pop(next(reversed(subset)))
            point = replace(point, participants=subset)
        elif fault == "wrong_world":
            world = scene.world_keys[0]
        elif fault == "stale_epoch":
            epoch -= 1
        else:
            other = stack.enter_context(compile_participant_scene(declaration, mode=tyx_mode, cache_root=tmp_path / "b"))
            point = other.snapshot()[world]
        with pytest.raises((ValueError, RuntimeError)):
            scene.restore({world: point}, expected_epoch=epoch)
        retained(scene, previous)
        audited_step(scene, data, built, history)
        history.append(dict(rejected=fault, point=point, expected_epoch=epoch, previous=previous))
        save_case(tmp_path, declaration, data, built, history, scene, None)


@pytest.mark.tyx
@pytest.mark.parametrize("fault", ("publication", "restore", "participant"))
def test_actual_late_failure_preserves_complete_world_and_retries(fault, tyx_mode, tmp_path, monkeypatch):
    declaration, data, built, _ = assembly("floating_affine")
    history, reached = [], []
    with compile_participant_scene(declaration, mode=tyx_mode, cache_root=tmp_path) as scene:
        audited_step(scene, data, built, history)
        checkpoint = scene.snapshot()
        audited_step(scene, data, built, history)
        previous = capture(scene)
        target_world = scene.world_keys[-1]
        target = ph.ResidentParticipantKey(target_world, 7)
        with monkeypatch.context() as patch:
            if fault == "publication":
                original = transaction.ResidentBatchTransaction._prepare_publication
                def fail(service, *args):
                    candidate = original(service, *args)
                    assert set(candidate.receipts) == set(scene.world_keys)
                    reached.append(candidate)
                    raise RuntimeError("after complete assembly receipt")
                patch.setattr(transaction.ResidentBatchTransaction, "_prepare_publication", fail)
                with pytest.raises(RuntimeError, match="complete assembly"):
                    scene.step(H)
            elif fault == "restore":
                def fail(service):
                    reached.append(service.state_epoch)
                    raise RuntimeError("after actual restore preparation")
                patch.setattr(transaction.ResidentBatchTransaction, "_before_restore_commit", fail)
                with pytest.raises(RuntimeError, match="restore preparation"):
                    scene.restore(checkpoint, expected_epoch=scene.state_epoch)
            else:
                def fail(service, attempt):
                    assert set(attempt["rows"]) == set(scene.participant_keys)
                    reached.append(tuple(attempt["rows"]))
                    return {target: 6}
                patch.setattr(transaction.ResidentBatchTransaction, "_before_finalize", fail)
                assert scene.step(H) == {w: w != target_world for w in scene.world_keys}
        assert len(reached) == 1
        if fault == "participant":
            assert scene.snapshot()[target_world] == previous["states"][target_world]
            assert scene.last_accepted[target_world] is previous["receipts"][target_world]
            for w in scene.world_keys:
                assert scene.snapshot()[w].tick == previous["states"][w].tick + int(w != target_world)
        else:
            retained(scene, previous)
        history.append(dict(fault=fault, reached=reached, previous=previous, observed=capture(scene)))
        audited_step(scene, data, built, history)
        save_case(tmp_path, declaration, data, built, history, scene, None)


@pytest.mark.tyx
def test_late_invalid_control_cannot_partly_commit(tyx_mode, tmp_path):
    declaration, data, built, controls = assembly("R02")
    with compile_participant_scene(declaration, mode=tyx_mode, cache_root=tmp_path) as scene:
        history = []
        audited_step(scene, data, built, history)
        previous = capture(scene)
        valid = ph.ResidentParticipantKey(scene.world_keys[0], 7)
        invalid = ph.ResidentParticipantKey(scene.world_keys[-1], 7)
        changed = dict(controls)
        changed[valid] = ph.RobotCommand(np.array([1 / 8, 1 / 16]))
        changed[invalid] = ph.RobotCommand(np.array([3., 0.]))
        with pytest.raises(ValueError, match="limit"):
            scene.set_controls(changed)
        retained(scene, previous)
        audited_step(scene, data, built, history)
        save_case(tmp_path, declaration, data, built, history, scene, None)


def test_declaration_owns_assets_commands_source_and_material_identity():
    declaration, _, _, _ = assembly("floating_affine")
    row = declaration.articulations[0]
    identity = declaration.identity
    with pytest.raises(FrozenInstanceError):
        row.key = ph.ResidentParticipantKey(0, 0)
    for array in (row.kp, row.initial_state.q, row.asset.tree.model.children.mass):
        with pytest.raises(ValueError):
            array.flags.writeable = True
    with pytest.raises(TypeError):
        row.collider_keys[next(iter(row.collider_keys))] = ph.CollisionKey(1, 1, 1)
    with pytest.raises(TypeError):
        declaration.materials_by_pair[next(iter(declaration.materials_by_pair))] = None
    declaration.assert_unchanged()
    assert declaration.identity == identity
    assert row.asset.tree.collisions[0].shape.key == ph.CollisionKey(0, 42, 71)
    assert row.asset.joint_names == ("joint0",)


@pytest.mark.parametrize("fault", ("missing", "extra", "world", "duplicate", "body", "metadata"))
def test_bad_asset_installation_rejected_before_runtime(fault, monkeypatch):
    declaration, _, _, _ = assembly("R02", worlds=(91,))
    row = declaration.articulations[0]
    called = []
    monkeypatch.setattr(resident_scene, "ResidentSceneBatch", lambda *a, **kw: called.append(True))
    def attempt():
        mapping = dict(row.collider_keys)
        source = next(iter(mapping))
        target = mapping[source]
        if fault == "missing":
            mapping.clear()
        elif fault == "extra":
            mapping[ph.CollisionKey(0, 900, 901)] = ph.CollisionKey(91, 700, 999)
        elif fault == "world":
            mapping[source] = replace(target, world=503)
        elif fault == "duplicate":
            # A different source collider cannot alias another participant's key.
            mapping[source] = next(iter(declaration.articulations[-1].collider_keys.values()))
        elif fault == "body":
            mapping[source] = replace(target, body=row.body_ids[0])
        else:
            asset = replace(row.asset, collision_model=ph.CollisionModel(()))
            row2 = replace(row, asset=asset)
            return compile_participant_scene(replace(declaration, articulations=(row2, *declaration.articulations[1:])))
        row2 = replace(row, collider_keys=mapping)
        return compile_participant_scene(replace(declaration, articulations=(row2, *declaration.articulations[1:])))
    with pytest.raises((ValueError, TypeError)):
        attempt()
    assert not called


@pytest.mark.parametrize("fault", ("missing", "foreign", "reverse", "static_default", "static_alias", "schema", "body_alias", "capacity"))
def test_incomplete_scene_is_not_silently_repaired(fault, monkeypatch):
    declaration, _, _, _ = assembly("R02", worlds=(91,))
    called = []
    monkeypatch.setattr(resident_scene, "ResidentSceneBatch", lambda *a, **kw: called.append(True))
    def attempt():
        if fault in ("missing", "foreign", "reverse"):
            materials = dict(declaration.materials_by_pair)
            pair, law = next(iter(materials.items()))
            if fault == "missing":
                materials.clear()
            elif fault == "foreign":
                materials[(pair[0], ph.CollisionKey(91, 999, 0))] = law
            else:
                materials = {pair[::-1]: law}
            bad = replace(declaration, materials_by_pair=materials)
        elif fault == "static_default":
            static = FiniteStaticShape.sphere(91, 990, (10, 0, 0), 1 / 8, material=ph.RobotContactMaterial(64, 0, 0))
            bad = replace(declaration, static_shapes=(static,), static_body_ids=((91, 990),))
        elif fault == "static_alias":
            bad = replace(declaration, static_body_ids=((91, declaration.articulations[0].body_ids[0]),))
        elif fault == "schema":
            bad = replace(declaration, schema="old-default")
        elif fault == "body_alias":
            a, b = declaration.articulations
            bad = replace(declaration, articulations=(a, replace(b, body_ids=(a.body_ids[0],))))
        else:
            bad = replace(declaration, candidate_capacity=0)
        return compile_participant_scene(bad)
    with pytest.raises((ValueError, TypeError)):
        attempt()
    assert not called


@pytest.mark.tyx
def test_explicit_none_pair_and_empty_geometry_are_real_no_contact(tyx_mode, tmp_path):
    declaration, data, built, _ = assembly("R01", worlds=(91,))
    empty_laws = dict.fromkeys(declaration.materials_by_pair, None)
    excluded = replace(declaration, materials_by_pair=empty_laws, candidate_capacity=0)
    rows = []
    for row in declaration.articulations:
        source = replace(row.asset, collisions=(), collision_model=ph.CollisionModel(()))
        rows.append(replace(row, asset=source, collider_keys={}))
    empty = ParticipantSceneDeclaration(tuple(rows), candidate_capacity=0)
    with ExitStack() as stack:
        a = stack.enter_context(compile_participant_scene(excluded, mode=tyx_mode, cache_root=tmp_path / "excluded"))
        b = stack.enter_context(compile_participant_scene(empty, mode=tyx_mode, cache_root=tmp_path / "empty"))
        before = a.snapshot()
        assert a.step(H) == b.step(H) == {91: True}
        assert a.snapshot()[91].participants == b.snapshot()[91].participants
        assert not a.last_accepted[91].sites and not b.last_accepted[91].sites
        for key, point in before[91].participants.items():
            after = a.snapshot()[91].participants[key]
            np.testing.assert_allclose(after.position[:3], np.asarray(point.position[:3]) + H * np.asarray(point.velocity[:3]), atol=2e-7)
            np.testing.assert_array_equal(after.velocity, point.velocity)
        references()[0].save(tmp_path, "explicit-no-contact", dict(excluded=excluded, empty=empty,
                                  before=before, a=capture(a), b=capture(b), inputs=data, identities=built))


@pytest.mark.tyx
def test_local_urdf_root_static_installation_and_old_declaration_unchanged(tyx_mode, tmp_path):
    path = tmp_path / "asset.urdf"
    path.write_text('''<robot name="assembly_source"><link name="base"><collision name="root_marker"><origin xyz="10 0 0"/><geometry><sphere radius=".125"/></geometry></collision></link><link name="slider"><inertial><mass value="2"/><inertia ixx=".125" iyy=".125" izz=".125" ixy="0" ixz="0" iyz="0"/></inertial><collision name="moving_tip"><geometry><sphere radius=".125"/></geometry></collision></link><joint name="slide" type="prismatic"><parent link="base"/><child link="slider"/><axis xyz="1 0 0"/><limit lower="-2" upper="2" effort="64" velocity="10"/></joint></robot>''', encoding="utf-8")
    asset = ph.load_urdf(path, gravity=(0, 0, 0))
    mapping = {c.shape.key: ph.CollisionKey(91, 990 if c.link_index < 0 else 700, 20 + i)
               for i, c in enumerate(asset.collisions)}
    row = ParticipantArticulation(ph.ResidentParticipantKey(91, 7), asset,
            ph.RobotState(np.array([1 / 16]), np.array([1 / 64])), kp=np.zeros(1), kd=np.zeros(1),
            effort_limit=np.array([64.]), body_ids=(700,), collider_keys=mapping)
    pair = tuple(sorted(mapping.values()))
    declaration = ParticipantSceneDeclaration((row,), static_body_ids=((91, 990),), materials_by_pair={pair: None})
    with compile_participant_scene(declaration, mode=tyx_mode, cache_root=tmp_path) as scene:
        assert scene.source_map[row.key] == mapping
        assert scene.layout.static_body_ids == ((91, 990),)
        for tick in range(3):
            assert scene.step(H) == {91: True}
            point = scene.snapshot()[91].participants[row.key]
            np.testing.assert_allclose(point.position, [1 / 16 + (tick + 1) * H / 64], atol=2e-7)
        references()[0].save(tmp_path, "urdf-source", dict(xml=path.read_text(), declaration=declaration, final=capture(scene)))
    old = ArticulationWorld(91, asset, row.initial_state, kp=row.kp, kd=row.kd, effort_limit=row.effort_limit)
    assert SceneDeclaration((old,)).worlds[0].world == 91
    with pytest.raises(ValueError, match="duplicate"):
        SceneDeclaration((old, old))


def test_plan_construction_failure_closes_allocated_public_owner(monkeypatch):
    declaration, _, _, _ = assembly("R02", worlds=(91,))
    calls = []
    class Owner:
        def __init__(self, *args, **kwargs):
            calls.append("construct")
        def prepare_contacts(self, *args, **kwargs):
            calls.append("prepare")
            raise RuntimeError("deliberate late prepare")
        def close(self):
            calls.append("close")
    monkeypatch.setattr(resident_scene, "ResidentSceneBatch", Owner)
    with pytest.raises(RuntimeError, match="late prepare"):
        compile_participant_scene(declaration)
    assert calls == ["construct", "prepare", "close"]
