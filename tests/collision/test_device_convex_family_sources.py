"""COL1a accepted bank, four layouts, conservative broad bounds and atomic failure."""

from contextlib import closing, contextmanager
from dataclasses import replace

import numpy as np
import pytest
import test_device_convex_query as k0
import test_device_convex_sources as old_sources
import test_device_convex_support_family as family

import phymeo as ph
from phymeo.dynamics.floating_robot import FloatingRobotState

LAYOUTS = old_sources.LAYOUTS
WORLDS = (83, 17, 107, 31)
ASSET_Q = ((0, 0, 0, 1), (.5, .5, .5, .5), (0, .6, 0, .8), (0, 0, 1, 0))
H = 1/1024


def declaration(layout, index):
    floating = layout.startswith("floating")
    asset = (ph.load_floating_mimic_urdf if floating else ph.load_mimic_urdf)(old_sources.ASSET, gravity=np.zeros(3))
    independent_q, independent_v = (index+1)/512, (index+1)/1024
    full_q, full_v = np.array((independent_q, .004-independent_q)), np.array((independent_v, -independent_v))
    gains = dict(kp=np.zeros(1), kd=np.zeros(1), effort_limit=np.ones(1), delay_steps=0)
    root_q = ASSET_Q[index]
    root = np.r_[np.array((2+index/4, -3, .25)), root_q]
    root_v = np.r_[family.reference().rotation(root_q)@np.array((1/128, 0, 0)), np.zeros(3)]
    if layout == "affine":
        return ph.AffineRobotTree(asset, ph.RobotState([independent_q], [independent_v]), **gains)
    if layout == "floating_affine":
        return ph.FloatingAffineRobotTree(asset, FloatingRobotState(np.r_[root, independent_q], np.r_[root_v, independent_v]), **gains)
    if layout == "ordinary":
        program = ph.DrivenRobotProgram(asset.tree.model, kp=np.zeros(2), kd=np.zeros(2), effort_limit=np.ones(2), delay_steps=0)
        return ph.FixedRobotTree(program, program.initial_state(ph.RobotState(full_q, full_v)))
    return ph.FloatingRobotTree(asset.tree.model, FloatingRobotState(np.r_[root, full_q], np.r_[root_v, full_v]),
                               kp=np.zeros(2), kd=np.zeros(2), effort_limit=np.ones(2), delay_steps=0)


def independent_last_pose(checkpoint, layout):
    # Explicit URDF premise: two +X sibling sliders at +/- .25; follower is last.
    q = np.asarray(checkpoint.full_position if "affine" in layout else checkpoint.position)
    if layout.startswith("floating"):
        R = family.reference().rotation(q[3:7])
        return q[:3]+R@np.array((.25+q[-1], 0, 0)), R, 2
    return np.array((.25+q[-1], 0, 0)), np.eye(3), 1


def install(batch, tree, layout, kind, world, index):
    checkpoint = batch.snapshot()[world]
    position, body_R, group = independent_last_pose(checkpoint, layout)
    ref = family.reference()
    local_center = np.array((index/64, -1/32, 1/64))
    asset_q = ASSET_Q[(index+1)%4]
    effective_R = body_R@ref.rotation(asset_q)
    normal = body_R@np.array((1.0, 0, 0))
    center = position+body_R@local_center
    support = center+effective_R@ref.support(kind, effective_R.T@normal)
    static_center = support+(1/8+1/32)*normal
    keys = (ph.CollisionKey(world, 21, 3), ph.CollisionKey(world, 39, 7))
    first = ph.CollisionShape(keys[0], family.KIND[kind], (101,), curved=ph.CurvedSolid(family.KIND[kind], ref.primitive(kind)[2], asset_q))
    second = ph.CollisionShape(keys[1], "sphere", (201,), radius=1/8)
    binding_type = (ph.ResidentTreeBinding, ph.ResidentAffineTreeBinding, ph.ResidentFloatingTreeBinding, ph.ResidentFloatingAffineTreeBinding)[LAYOUTS.index(layout)]
    options = dict(topology=checkpoint.topology)
    if layout == "floating_affine":
        options["graph_token"] = tree.inputs(np.zeros((1, 3))).graph_token
    bindings = (binding_type(keys[0], world, group, local_center[None], **options),
                ph.ResidentStaticBinding(keys[1], ph.CollisionMotion(static_center[None])))
    return (first, second), bindings, dict(local_center=local_center, asset_q=asset_q,
                                            static_center=static_center, normal=normal, world=world, kind=kind)


def current_case(checkpoint, layout, declared):
    ref = family.reference()
    position, body_R, _ = independent_last_pose(checkpoint, layout)
    center = position+body_R@declared["local_center"]
    R = body_R@ref.rotation(declared["asset_q"])
    normal = declared["normal"]
    point = center+R@ref.support(declared["kind"], R.T@normal)
    static_center = declared["static_center"]
    # Collinear, zero applied loads and zero gravity keep body orientation fixed.
    gap = float((static_center-point)@normal)-1/8
    assert np.linalg.norm(static_center-point-(gap+1/8)*normal) <= 2e-6
    assert gap > 1/64
    world = declared["world"]
    return dict(family=declared["kind"]+"S", kinds=(declared["kind"], "S"),
                phase="sep", placement_m=gap, expected_gap_m=gap, normal=normal,
                keys=((world, 21, 3), (world, 39, 7)), ids=((101,), (201,)),
                centers=(center, static_center), frames=(R, np.eye(3)),
                points=(center[None], static_center[None]), requested_reverse=False, oblique=True)


@pytest.mark.tyx
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("kind", ("Y", "N", "E"))
def test_four_layouts_four_sparse_worlds_continue_three_real_steps(layout, kind, tyx_mode, tmp_path):
    trees = tuple(declaration(layout, i) for i in range(4))
    with family.saved(tmp_path, "source-"+layout+"-"+kind, tyx_mode, dict(layout=layout, kind=kind, worlds=WORLDS)) as record:
        with ph.TyxWorldBatch(list(zip(WORLDS, trees, strict=True))[::-1], tyx_mode, cache_root=tmp_path) as batch:
            shapes, bindings, declarations, pairs = [], [], {}, []
            for index, (world, tree) in enumerate(zip(WORLDS, trees, strict=True)):
                shape, binding, declared = install(batch, tree, layout, kind, world, index)
                shapes.extend(shape)
                bindings.extend(binding)
                declarations[world] = declared
                pairs.append(tuple(item.key for item in shape))
            record.update(declarations=declarations, initial=batch.snapshot(), steps=[])
            with batch.prepare_collision(ph.CollisionModel(tuple(shapes)), tuple(bindings)) as broad:
                with closing(batch.prepare_convex_query(broad, pairs=tuple(pairs))) as plan:
                    for tick in range(3):
                        assert all(batch.step(H).values())
                        state = batch.snapshot()
                        result = k0.query_attempt(batch, plan, record, "continuing-"+str(tick))
                        family.check_raw(result)
                        family.check_source(result)
                        checked = {}
                        for row in result.rows:
                            world = row.pair[0].world
                            data = current_case(state[world], layout, declarations[world])
                            # check_result's raw checks use global row ordinal; check geometry directly here.
                            truth = family.reference().oracle(data)
                            assert row.status == "RESOLVED" and row.relation == "SEPARATED"
                            length = truth["scale_m"]
                            assert abs(row.gap_m-truth["gap_m"]) <= 1e-4*length
                            assert np.linalg.norm(np.asarray(row.normal)-truth["normal"]) <= 2e-3
                            checked[world] = tuple(family.reference().witness_check(data, side, witness, np.asarray(row.origin_m), truth["normal"], 2e-4*length) for side, witness in enumerate((row.first, row.second)))
                            shape_row = result.shape_keys.index(row.pair[0])
                            np.testing.assert_allclose(result.source["effective_shape_rotation"][shape_row], data["frames"][0], atol=2e-6, rtol=0)
                            assert result.ticks[result.world_keys.index(world)] == state[world].tick == tick+1
                        record["steps"].append(dict(state=state, query=result, checks=checked))
                    batch.reset([WORLDS[1]])
                    with pytest.raises(ValueError, match="stale"):
                        plan.require_current(result)
                    assert all(batch.step(H).values())
                    retry = k0.query_attempt(batch, plan, record, "selected-reset-next-real-step")
                    assert retry.ticks[retry.world_keys.index(WORLDS[1])] == 1


@pytest.mark.tyx
@pytest.mark.parametrize("kind", ("Y", "N", "E"))
@pytest.mark.parametrize("dynamic_second", (False, True))
def test_extended_surface_broad_phase_keeps_nonoverlapping_centers(kind, dynamic_second, tyx_mode, tmp_path):
    ref = family.reference()
    asset_q = ref.TILT_Q if kind == "E" else (0.0, 0.0, 0.0, 1.0)
    asset_R = ref.rotation(asset_q)
    direction = asset_R@np.array((1.0, 0, 0))
    surface = asset_R@ref.support(kind, asset_R.T@direction)
    sphere_center = surface+(1/8-1/64)*direction
    world = 17
    tree = k0.ordinary_forest(n=2 if dynamic_second else 1)
    with family.saved(tmp_path, "broad-"+kind+"-"+str(dynamic_second), tyx_mode, dict(kind=kind, asset_q=asset_q, sphere_center=sphere_center, dynamic_second=dynamic_second)) as record:
        with ph.TyxWorldBatch([(world, tree)], tyx_mode, cache_root=tmp_path) as batch:
            state = batch.snapshot()[world]
            keys = (ph.CollisionKey(world, 11, 1), ph.CollisionKey(world, 22, 2))
            shapes = (ph.CollisionShape(keys[0], family.KIND[kind], (101,), curved=ph.CurvedSolid(family.KIND[kind], ref.primitive(kind)[2], asset_q)),
                      ph.CollisionShape(keys[1], "sphere", (201,), radius=1/8))
            second = ph.ResidentTreeBinding(keys[1], world, 1, sphere_center[None], topology=state.topology) if dynamic_second else ph.ResidentStaticBinding(keys[1], ph.CollisionMotion(sphere_center[None]))
            bindings = (ph.ResidentTreeBinding(keys[0], world, 0, np.zeros((1, 3)), topology=state.topology), second)
            with batch.prepare_collision(ph.CollisionModel(shapes), bindings, tree_dynamic_pairs=dynamic_second) as broad:
                with closing(batch.prepare_convex_query(broad)) as plan:
                    result = k0.query_attempt(batch, plan, record, "broad")
                    assert broad.last_result.candidates == (keys,)
                    row = result.rows[0]
                    # The second sphere is itself an exact inscribed ball; A has the support point.
                    assert row.status == "RESOLVED" and row.relation == "OVERLAP"
                    assert abs(row.gap_m+1/64) <= 1e-4
                    assert np.linalg.norm(np.asarray(row.normal)-direction) <= 2e-3


@pytest.mark.tyx
@pytest.mark.parametrize("fault", ("owner", "world", "epoch", "layout", "late_read"))
def test_source_failure_keeps_every_publication_and_retries(fault, tyx_mode, tmp_path, monkeypatch):
    data = family.reference().case("NE", "sep")
    with family.saved(tmp_path, "source-failure-"+fault, tyx_mode, data) as record:
        with family.prepared(data, tyx_mode, tmp_path, pairs=None) as (batch, broad, plan):
            # Use a nonzero overlapping broad candidate for the atomic source test.
            accepted = k0.query_attempt(batch, plan, record, "before")
            old_broad, before = broad.last_result, batch.snapshot()
            original_loan, original_read = batch._borrow_collision_state, plan._read_raw
            @contextmanager
            def invalid(*args, **kwargs):
                with original_loan(*args, **kwargs) as loan:
                    source = loan.tree
                    if fault == "owner":
                        source = replace(source, owner=object())
                    elif fault == "world":
                        source = replace(source, rows={999: 0})
                    elif fault == "epoch":
                        source = replace(source, state_epoch=source.state_epoch+1)
                    else:
                        source = None
                    yield replace(loan, tree=source)
            if fault == "late_read":
                def bad_read():
                    original_read()
                    raise RuntimeError("late curved readback failure")
                monkeypatch.setattr(plan, "_read_raw", bad_read)
            else:
                monkeypatch.setattr(batch, "_borrow_collision_state", invalid)
            with pytest.raises((ValueError, RuntimeError)):
                k0.query_attempt(batch, plan, record, "failure")
            assert plan.last_result is accepted and broad.last_result is old_broad
            assert k0.canonical(before) == k0.canonical(batch.snapshot())
            monkeypatch.setattr(plan, "_read_raw", original_read)
            monkeypatch.setattr(batch, "_borrow_collision_state", original_loan)
            retry = k0.query_attempt(batch, plan, record, "retry")
            assert retry.state_epoch == accepted.state_epoch


@pytest.mark.tyx
def test_new_buffers_count_towards_exact_memory_and_pair_capacity(tyx_mode, tmp_path):
    data = family.reference().case("NE", "overlap")
    with family.saved(tmp_path, "exact-memory-and-pair-cap", tyx_mode, data) as record:
        with family.prepared(data, tyx_mode, tmp_path) as (batch, broad, plan):
            required = plan.stats["live_bytes"]
            with closing(batch.prepare_convex_query(broad, pairs=(tuple(ph.CollisionKey(*k) for k in data["keys"]),), memory_limit_bytes=required)) as exact:
                assert k0.query_attempt(batch, exact, record, "exact-memory").rows[0].status == "RESOLVED"
            with pytest.raises(MemoryError):
                batch.prepare_convex_query(broad, memory_limit_bytes=required-1)
            with closing(batch.prepare_convex_query(broad, pair_capacity=0)) as small:
                before, last = batch.snapshot(), broad.last_result
                with pytest.raises(ValueError, match="capacity"):
                    k0.query_attempt(batch, small, record, "pair-overflow")
                assert small.last_result is None and broad.last_result is last
                assert k0.canonical(batch.snapshot()) == k0.canonical(before)



@pytest.mark.tyx
def test_copied_checkpoint_next_step_and_owner_do_not_alias(tyx_mode, tmp_path):
    data = family.reference().case("NE", "sep")
    with family.saved(tmp_path, "copied-checkpoint-next-step", tyx_mode, data) as record:
        with family.prepared(data, tyx_mode, tmp_path/"first") as (batch, broad, plan):
            assert batch.step(H) == {17: True}
            checkpoint = batch.snapshot()
            old = k0.query_attempt(batch, plan, record, "initial")
            with family.prepared(data, tyx_mode, tmp_path/"second") as (clone, clone_broad, clone_plan):
                clone.restore(checkpoint)
                with pytest.raises(ValueError, match="another|stale"):
                    clone_plan.require_current(old)
                assert batch.step(H) == clone.step(H) == {17: True}
                a = k0.query_attempt(batch, plan, record, "original-next")
                b = k0.query_attempt(clone, clone_plan, record, "clone-next")
                np.testing.assert_array_equal(batch.snapshot()[17].position, clone.snapshot()[17].position)
                np.testing.assert_array_equal(batch.snapshot()[17].velocity, clone.snapshot()[17].velocity)
                np.testing.assert_array_equal(a.raw["gap"], b.raw["gap"])



@pytest.mark.tyx
def test_curved_required_load_rejects_without_host_fallback_or_partial_step(tyx_mode, tmp_path, monkeypatch):
    from phymeo.backends import _resident_contact_bridge
    data = family.reference().case("SE", "overlap")
    with family.saved(tmp_path, "required-load-rejected", tyx_mode, data) as record:
        with family.prepared(data, tyx_mode, tmp_path) as (batch, broad, query):
            before = batch.snapshot()
            def forbidden(*args, **kwargs):
                raise AssertionError("curved device patch fell back to HOST")
            monkeypatch.setattr(_resident_contact_bridge, "contact_patch_sample", forbidden)
            pair = tuple(ph.CollisionKey(*key) for key in data["keys"])
            created = None
            with pytest.raises((ValueError, RuntimeError), match="patch|unsupported|UNSUPPORTED|curved"):
                with batch.prepare_contacts(broad._model, broad._convex_bindings,
                                            {pair: ph.RobotContactMaterial(stiffness=64, damping=0, friction=0)},
                                            narrow_phase="device_patch") as plan:
                    created = plan
                    batch.step_contacts(plan, H)
            if created is not None:
                assert not created.last_accepted and created.last_device_query is None
            assert k0.canonical(batch.snapshot()) == k0.canonical(before)



@pytest.mark.tyx
def test_independent_static_body_frame_is_rejected_before_query_allocation(tyx_mode, tmp_path):
    data = family.reference().case("SE", "sep")
    with family.saved(tmp_path, "static-body-frame-rejected", tyx_mode, data) as record:
        with family.prepared(data, tyx_mode, tmp_path) as (batch, broad, plan):
            bindings = list(broad._convex_bindings)
            second = bindings[1]
            motion = ph.CollisionMotion(second.motion.start, body_orientation_xyzw=(.5, .5, .5, .5))
            bindings[1] = ph.ResidentStaticBinding(second.key, motion)
            with batch.prepare_collision(broad._model, tuple(bindings)) as explicit_frame:
                before = dict(batch.stats())
                with pytest.raises(ValueError, match="independent static curved body orientation"):
                    batch.prepare_convex_query(explicit_frame)
                after = dict(batch.stats())
                assert after["convex_query_plan_live_count"] == before["convex_query_plan_live_count"]
                assert after["convex_query_live_bytes"] == before["convex_query_live_bytes"]



@pytest.mark.tyx
@pytest.mark.parametrize("field", ("declared_dimensions_m", "uploaded_asset_rotation", "support_family_schema", "pair_origins_relative_m"))
def test_curved_metadata_cannot_be_rebound(field, tyx_mode, tmp_path):
    data = family.reference().case("NE", "sep")
    with family.saved(tmp_path, "metadata-rejected-"+field, tyx_mode, data) as record:
        with family.prepared(data, tyx_mode, tmp_path) as (batch, broad, plan):
            result = k0.query_attempt(batch, plan, record, "original")
            changed = dict(result.source)
            if field == "support_family_schema":
                changed[field] = "unrelated-source"
            else:
                changed[field] = changed[field].copy()
                changed[field].flat[0] += .25
            with pytest.raises(ValueError, match="source binding"):
                plan.require_current(replace(result, source=changed))



@pytest.mark.tyx
def test_last_effective_frame_failure_is_atomic(tyx_mode, tmp_path, monkeypatch):
    data = family.reference().case("NE", "overlap")
    with family.saved(tmp_path, "late-effective-frame", tyx_mode, data) as record:
        with family.prepared(data, tyx_mode, tmp_path, pairs=None) as (batch, broad, plan):
            accepted = k0.query_attempt(batch, plan, record, "before")
            before, previous_broad = batch.snapshot(), broad.last_result
            original = plan._read
            def broken(buffer):
                value = original(buffer)
                if buffer is plan._buffers["shape_rotation"]:
                    value[-1, 0, 0] += .25
                return value
            monkeypatch.setattr(plan, "_read", broken)
            with pytest.raises(RuntimeError, match="rotation"):
                k0.query_attempt(batch, plan, record, "bad-frame")
            assert plan.last_result is accepted and broad.last_result is previous_broad
            assert k0.canonical(batch.snapshot()) == k0.canonical(before)
            monkeypatch.setattr(plan, "_read", original)
            retry = k0.query_attempt(batch, plan, record, "retry")
            family.check_result(retry, data)
