"""D5-A tests first: asset declaration to the accepted common D3 transaction.

New assembly qualification only; original D3 independent mechanics are reused
as a test oracle, not as an implementation or a second product owner.
"""
import importlib.util
from contextlib import ExitStack
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

import phymeo as ph
from phymeo.builders._robot_asset import LinkCollision
from phymeo.builders.deformable_scene_lowering import (
    CompiledDeformableParticipantScene,
    ParticipantDeformable,
)
from phymeo.builders.floating_urdf import FloatingRobotAsset
from phymeo.builders.participant_scene_compiler import (
    ParticipantArticulation,
    ParticipantSceneDeclaration,
    compile_participant_scene,
)


@lru_cache(maxsize=1)
def reference():
    path = Path(__file__).parent / "collision/test_deformable_contact_response.py"
    spec = importlib.util.spec_from_file_location("d5_original_independent_d3", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture(case="E", world=7):
    ref = reference()
    item = ref.fixture(case, world=world)
    item["external"] = np.array((0, 1 / 32, 0, 0, 0, 1 / 64), float)
    tree = item["participant"].system
    source = ph.CollisionKey(0, 42, 71)
    shape = replace(next(s for s in item["model"].shapes if type(s) is ph.CollisionShape),
                    key=source)
    collider = LinkCollision(
        "original-" + case, "root", -1, (0., 0., 0.), (0., 0., 0., 1.),
        shape, tuple(tuple(float(v) for v in row) for row in item["local"]), True,
    )
    asset = FloatingRobotAsset(
        "d5-" + case, "root", tree.model, (), (), (), (), (), (), (collider,),
        ph.CollisionModel((shape,)),
    )
    command = ph.FloatingRobotCommand(
        None, ph.FloatingRobotInputs(np.zeros(0), item["external"][None, :3],
                                    item["external"][None, 3:]),
    )
    row = ParticipantArticulation(
        item["participant"].key, asset, tree.state,
        kp=np.zeros(0), kd=np.zeros(0), effort_limit=np.zeros(0),
        body_ids=(202,), collider_keys={source: item["round"]}, initial_control=command,
    )
    item.update(row=row, original_source=source, public_command=command)
    return item


def declaration(items, **options):
    ref = reference()
    node = ParticipantDeformable(
        tuple(i["node"] for i in items),
        tuple(ph.DeformableBoundarySurface(i["surface"], 23) for i in items),
        settings=ref.SETTINGS,
    )
    return ParticipantSceneDeclaration(
        tuple(i["row"] for i in items),
        deformable=node, contact_response="deformable_normal",
        materials_by_pair={pair: law for i in items for pair, law in i["materials"].items()},
        site_capacity=2 * len(items), **options,
    )


def checked_step(scene, items, evidence):
    return reference().attempt(scene, items, reference().H, evidence)


def assert_physical_equal(a, b, items):
    for item in items:
        world = item["world"]
        for name in ("position", "velocity", "clock"):
            np.testing.assert_array_equal(getattr(a[world].node, name), getattr(b[world].node, name))
        assert a[world].node.tick == b[world].node.tick
        assert a[world].rigid.tick == b[world].rigid.tick
        assert reference().root_state(a[world], item) == reference().root_state(b[world], item)


@pytest.mark.tyx
@pytest.mark.parametrize("cases", (("E", "P"), ("C",)))
def test_asset_to_common_response_short_windows(cases, tyx_mode, tmp_path):
    ref = reference()
    items = tuple(fixture(case, world=7 + 24 * i) for i, case in enumerate(cases))
    decl = declaration(items)
    with ref.saved(tmp_path, "assembly-" + "".join(cases), tyx_mode,
                   declaration=decl) as evidence:
        with compile_participant_scene(decl, mode=tyx_mode) as scene:
            assert isinstance(scene, CompiledDeformableParticipantScene)
            assert scene.identity == decl.identity
            assert scene.world_keys == tuple(sorted(i["world"] for i in items))
            for item in items:
                assert scene.source_map["participants"][item["row"].key][
                    item["original_source"]] == item["round"]
                surface = scene.source_map["surfaces"][item["surface"]]
                np.testing.assert_array_equal(surface["facet_ids"], ref.FACETS)
                np.testing.assert_array_equal(surface["oriented_point_ids"], ref.IDS[ref.FACES])
            continuous = {}
            for item in items:
                initial = item["participant"].system.state
                continuous[item["world"]] = (
                    item["node"].initial_state.position.astype(float).copy(),
                    item["node"].initial_state.velocity.astype(float).copy(),
                    initial.q.copy(), initial.v.copy(),
                )
            for _ in range(16):
                result, record = checked_step(scene, items, evidence)
                expected = {}
                for item in items:
                    w = item["world"]
                    x, v, q, rv = continuous[w]
                    force, wrench, _ = ref.reference_load(item, x, v, q, rv)
                    x, v, _ = ref.material_step(item, x, v, force, ref.H)
                    q, rv = ref.rigid_step(item, q, rv, wrench, ref.H)
                    expected[w] = (x, v, q, rv)
                record["independent_continuous"] = expected
                for item in items:
                    w = item["world"]
                    x, v, q, rv = expected[w]
                    current = ref.root_state(record["state"][w], item)
                    ref.close(record["state"][w].node.position, x, atol=2e-4, rtol=0)
                    ref.close(record["state"][w].node.velocity, v, atol=2e-4, rtol=0)
                    ref.close(current.position[:3], q[:3], atol=2e-4, rtol=0)
                    ref.close(current.velocity, rv, atol=2e-4, rtol=0)
                    assert min(np.linalg.norm(np.asarray(current.position[3:]) - q[3:]),
                               np.linalg.norm(np.asarray(current.position[3:]) + q[3:])) <= 2e-4
                continuous = expected
                assert result.query.source_kind == "deformable-floating-entry-device-v1"


@pytest.mark.tyx
def test_query_revision_control_restore_and_initial_held_reset(tyx_mode, tmp_path):
    ref = reference()
    items = (fixture("E", 31), fixture("P", 7))
    with ref.saved(tmp_path, "lifecycle", tyx_mode, declaration=declaration(items)) as evidence:
        with compile_participant_scene(declaration(items), mode=tyx_mode) as scene:
            initial = scene.snapshot()
            query = scene.query(expected_revision=scene.revision)
            evidence["initial_query"] = query
            assert scene.require_query_current(query, expected_revision=scene.revision) is query
            checked_step(scene, items, evidence)
            with pytest.raises(ValueError, match="stale|current|revision"):
                scene.require_query_current(query, expected_revision=scene.revision)
            saved = scene.checkpoint()
            state_a = checked_step(scene, items, evidence)[1]["state"]
            scene.restore(saved, expected_revision=scene.revision)
            state_b = checked_step(scene, items, evidence)[1]["state"]
            assert_physical_equal(state_a, state_b, items)
            current = scene.snapshot()
            with pytest.raises(ValueError, match="stale|current|revision|query"):
                scene.require_query_current(scene.last_step.query, expected_revision=scene.revision)
            control_query = scene.query(expected_revision=scene.revision)
            scene.set_controls({items[0]["row"].key: ph.FloatingRobotCommand(
                None, ph.FloatingRobotInputs(np.zeros(0), np.ones((1, 3)), np.zeros((1, 3))))})
            with pytest.raises(ValueError, match="stale|current|revision"):
                scene.require_query_current(control_query, expected_revision=scene.revision)
            query = scene.query(expected_revision=scene.revision)
            scene.reset((31,), expected_revision=scene.revision)
            with pytest.raises(ValueError, match="stale|current|revision"):
                scene.require_query_current(query, expected_revision=scene.revision)
            reset = scene.snapshot()
            assert_physical_equal(initial, reset, (items[0],))
            assert_physical_equal(current, reset, (items[1],))
            checked_step(scene, items, evidence)  # all original held channels restored
            stale = scene.revision - 1
            before = scene.checkpoint()
            with pytest.raises(ValueError, match="stale|revision"):
                scene.reset(expected_revision=stale)
            assert ref.encode(before) == ref.encode(scene.checkpoint())


@pytest.mark.tyx
@pytest.mark.parametrize("seam", ("_query_entry", "_prepare_response_publication"))
def test_late_common_failure_and_real_retry(seam, tyx_mode, tmp_path, monkeypatch):
    ref = reference()
    items = (fixture("E"), fixture("P", 31))
    with ref.saved(tmp_path, "assembly-late-" + seam, tyx_mode) as evidence:
        with compile_participant_scene(declaration(items), mode=tyx_mode) as scene:
            checked_step(scene, items, evidence)
            before, old, revision = scene.checkpoint(), scene.last_step, scene.revision
            original = getattr(scene._response, seam)
            completed = []
            def late(*args, **kwargs):
                value = original(*args, **kwargs)
                completed.append(value)
                raise RuntimeError("D5 after real candidate")
            with monkeypatch.context() as patch:
                patch.setattr(scene._response, seam, late)
                with pytest.raises(RuntimeError, match="D5 after real candidate"):
                    checked_step(scene, items, evidence)
            assert len(completed) == 1
            assert ref.encode(before) == ref.encode(scene.checkpoint())
            assert scene.last_step is old and scene.revision == revision
            evidence["last_failed_attempt"] = scene.last_attempt
            checked_step(scene, items, evidence)


@pytest.mark.parametrize("mode", ("host_patch", "none"))
def test_old_modes_refuse_deformable_without_changing_defaults(mode):
    item = fixture()
    decl = declaration((item,))
    with pytest.raises(ValueError, match="deformable|mode|response"):
        replace(decl, contact_response=mode, materials_by_pair={})


@pytest.mark.parametrize("fault", (
    "missing_layer", "extra_layer", "friction", "wrong_world", "unknown_surface",
    "duplicate_world", "static_shape", "site_mapping", "capacity", "node_memory", "pair_capacity",
))
def test_full_static_admission_before_owner(fault, monkeypatch):
    from phymeo.builders import deformable_scene_lowering as lowering
    item = fixture()
    decl = declaration((item,))
    called = []
    def forbidden(*args, **kwargs):
        called.append(True)
        raise AssertionError("allocated before static rejection")
    monkeypatch.setattr(lowering, "TyxDeformableContactBatch", forbidden)
    with pytest.raises((ValueError, TypeError, MemoryError)):
        if fault == "missing_layer":
            decl = replace(decl, materials_by_pair={})
        elif fault == "extra_layer":
            extra = (item["surface"], ph.CollisionKey(7, 999, 0))
            decl = replace(decl, materials_by_pair={**decl.materials_by_pair,
                                                  extra: ph.NormalLayer(1/16, 128, .5)})
        elif fault == "friction":
            decl = replace(decl, materials_by_pair={
                next(iter(decl.materials_by_pair)): ph.NormalLayer(1/16, 128, .5, friction=.1)})
        elif fault == "wrong_world":
            decl = replace(decl, deformable=replace(decl.deformable, worlds=(
                replace(item["node"], key=31),)))
        elif fault == "unknown_surface":
            decl = replace(decl, deformable=replace(decl.deformable, surfaces=(
                ph.DeformableBoundarySurface(item["surface"], 24),)))
        elif fault == "duplicate_world":
            decl = replace(decl, deformable=replace(decl.deformable,
                                                  worlds=(item["node"], item["node"])))
        elif fault == "static_shape":
            decl = replace(decl, static_shapes=(ph.FiniteStaticShape.sphere(7, 999, (5,5,5), .1),),
                           static_body_ids=((7, 999),))
        elif fault == "site_mapping":
            decl = replace(decl, site_capacity={7: 2})
        elif fault == "pair_capacity":
            decl = replace(decl, candidate_capacity=0)
        elif fault == "capacity":
            decl = replace(decl, site_capacity=129)
        elif fault == "node_memory":
            decl = replace(decl, deformable=replace(decl.deformable, node_memory_limit_bytes=1))
        compile_participant_scene(decl, mode="interpret")
    assert called == []


def test_owned_model_surface_and_source_content():
    item = fixture()
    decl = declaration((item,))
    arrays = (decl.deformable.worlds[0].initial_state.position,
              decl.deformable.worlds[0].point_ids,
              decl.deformable.worlds[0].surfaces[0].facet_ids)
    for array in arrays:
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.setflags(write=True)
    before = decl.identity
    item["node"].initial_state.position[0, 0] = 1 / 16
    assert decl.identity == before
    decl.assert_unchanged()


def test_source_mutation_is_rejected_before_allocation(monkeypatch):
    from phymeo.builders import deformable_scene_lowering as lowering
    item = fixture()
    decl = declaration((item,))
    object.__setattr__(decl.articulations[0].asset.model, "root_com_m", np.array((.25, 0, 0)))
    monkeypatch.setattr(lowering, "TyxDeformableContactBatch",
                        lambda *a, **k: pytest.fail("allocation after source mutation"))
    with pytest.raises(ValueError, match="source|changed"):
        compile_participant_scene(decl, mode="interpret")


@pytest.mark.tyx
def test_wrong_checkpoint_owner_and_closed_lifecycle(tyx_mode):
    items = (fixture(),)
    with ExitStack() as stack:
        first = stack.enter_context(compile_participant_scene(declaration(items), mode=tyx_mode))
        second = stack.enter_context(compile_participant_scene(declaration(items), mode=tyx_mode))
        before = first.checkpoint()
        with pytest.raises(ValueError, match="owner|binding"):
            first.restore(second.checkpoint(), expected_revision=first.revision)
        assert reference().encode(before) == reference().encode(first.checkpoint())
        first.close()
        with pytest.raises(RuntimeError, match="closed"):
            first.snapshot()


@pytest.mark.parametrize("cleanup_fails", (False, True))
def test_constructor_control_failure_retains_cleanup_owner(monkeypatch, cleanup_fails):
    from phymeo.builders import deformable_scene_lowering as lowering
    primary, cleanup = RuntimeError("initial control rejected"), RuntimeError("release unknown")
    owners = []
    class FailedOwner:
        def __init__(self, *args, **kwargs):
            self.closed = 0
            owners.append(self)
        def set_controls(self, controls):
            raise primary
        def close(self):
            self.closed += 1
            if cleanup_fails:
                raise cleanup
    monkeypatch.setattr(lowering, "TyxDeformableContactBatch", FailedOwner)
    with pytest.raises(BaseException) as failure:
        compile_participant_scene(declaration((fixture(),)), mode="interpret")
    assert len(owners) == 1 and owners[0].closed == 1
    if cleanup_fails:
        assert isinstance(failure.value, BaseExceptionGroup)
        assert failure.value.exceptions == (primary, cleanup)
        assert failure.value.contact_owner is owners[0]
        assert failure.value.cleanup_state == "unresolved"
    else:
        assert failure.value is primary


@pytest.mark.parametrize("value", ("object", "none", "host_patch"))
def test_direct_compiled_constructor_refuses_wrong_declaration(value):
    if value == "object":
        decl = object()
    else:
        decl = ParticipantSceneDeclaration((fixture()["row"],), contact_response=value)
    with pytest.raises((TypeError, ValueError), match="ParticipantSceneDeclaration|deformable_normal"):
        CompiledDeformableParticipantScene(decl, mode="interpret")
