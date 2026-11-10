"""B1 accepted full-source layouts and identity failures; authoring only."""
from contextlib import closing, contextmanager
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import test_device_convex_patch_qualification as patch
import test_device_convex_query as k0

import phymeo as ph
from phymeo.dynamics.floating_robot import FloatingRobotState

LAYOUTS = ("ordinary", "affine", "floating", "floating_affine")
ASSET = Path(__file__).resolve().parents[1] / "fixtures/urdf/floating_mimic_jaws.urdf"
CYCLE = np.asarray([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])


def declaration(layout):
    """Full physical input differs from K=1 control coordinates by construction."""
    floating = layout.startswith("floating")
    asset = (ph.load_floating_mimic_urdf if floating else ph.load_mimic_urdf)(ASSET, gravity=np.zeros(3))
    assert asset.tree.joint_names == ("driver", "follower")
    full_q, full_v = np.asarray([.01, -.006]), np.asarray([.002, -.002])
    gains = dict(kp=np.zeros(1), kd=np.zeros(1), effort_limit=np.ones(1), delay_steps=0)
    root = np.asarray([2., -3., .25, .5, .5, .5, .5])
    if layout == "affine":
        tree = ph.AffineRobotTree(asset, ph.RobotState([.01], [.002]), **gains)
    elif layout == "floating_affine":
        tree = ph.FloatingAffineRobotTree(
            asset, FloatingRobotState(np.r_[root, .01], np.r_[np.zeros(6), .002]), **gains)
    elif layout == "ordinary":
        program = ph.DrivenRobotProgram(asset.tree.model, kp=np.zeros(2), kd=np.zeros(2),
                                        effort_limit=np.ones(2), delay_steps=0)
        tree = ph.FixedRobotTree(program, program.initial_state(ph.RobotState(full_q, full_v)))
    else:
        tree = ph.FloatingRobotTree(asset.tree.model,
                                   FloatingRobotState(np.r_[root, full_q], np.r_[np.zeros(6), full_v]),
                                   kp=np.zeros(2), kd=np.zeros(2), effort_limit=np.ones(2), delay_steps=0)
    return tree


def installation(batch, tree, layout, family, world, root=False):
    base = k0.reference().family_case(family, -.01)
    spec = getattr(patch.reference(), family.lower() + "_overlap")()
    floating = layout.startswith("floating")
    rotation = CYCLE if floating else np.eye(3)
    anchor = np.asarray([2., -3., .25]) if floating else np.asarray([-.25, 0., 0.])
    group = 0 if root else (2 if floating else 1)
    body_position = np.asarray([2., -3., .25]) if floating else np.zeros(3)
    if not root:
        body_position = body_position + rotation @ np.asarray([.244, 0., 0.])
    checkpoint = batch.snapshot([world])[world]
    case = dict(base)
    case["keys"] = ((world, 20 + group, 11), (world, 90, 21))
    case["points"] = tuple(np.asarray(points) @ rotation.T + body_position for points in base["points"])
    case["positions"] = (body_position, body_position)
    binding_type = (ph.ResidentTreeBinding, ph.ResidentAffineTreeBinding,
                    ph.ResidentFloatingTreeBinding, ph.ResidentFloatingAffineTreeBinding)[LAYOUTS.index(layout)]
    shapes, bindings = [], []
    for side in range(2):
        key = ph.CollisionKey(*case["keys"][side])
        shape = ph.CollisionShape(key, {"S": "sphere", "C": "capsule", "B": "hull", "H": "hull"}[case["kinds"][side]],
                                  case["ids"][side], case["radii"][side], revision=7 + side)
        shapes.append(shape)
        if side == 0:
            options = dict(topology=checkpoint.topology)
            if layout == "floating_affine":
                options["graph_token"] = tree.inputs(np.zeros((1, 3))).graph_token
            bindings.append(binding_type(key, world, group, case["local"][0], **options))
        else:
            bindings.append(ph.ResidentStaticBinding(key, ph.CollisionMotion(case["points"][side])))
    expected = dict(spec)
    expected["normal"] = rotation @ spec["normal"]
    expected["sites"] = tuple(dict(site, surfaces=tuple(np.asarray(point) @ rotation.T + body_position - anchor
                                                        for point in site["surfaces"]))
                              for site in spec["sites"])
    return case, expected, shapes, bindings, anchor, body_position, rotation, group


@pytest.mark.tyx
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("family", ("SS", "CC", "HH"))
def test_full_physical_last_body_supplies_patch(layout, family, tyx_mode, tmp_path):
    tree, world = declaration(layout), 83
    with patch.saved(tmp_path, f"B1-source-{layout}-{family}", tyx_mode, dict(layout=layout, family=family)) as record:
        with ph.TyxWorldBatch([(world, tree)], tyx_mode, cache_root=tmp_path) as batch:
            case, expected, shapes, bindings, anchor, position, rotation, group = installation(batch, tree, layout, family, world)
            record.update(case=case, expected=expected, before=batch.snapshot())
            with batch.prepare_collision(ph.CollisionModel(tuple(shapes)), tuple(bindings)) as broad:
                pairs = (tuple(shape.key for shape in shapes),)
                with closing(batch.prepare_convex_query(broad, pairs=pairs, site_capacity=16)) as plan:
                    result = k0.query_attempt(batch, plan, record, "full-last-body")
                    patch.assert_patch(result.rows[0], expected, case)
                    assert tuple(result.source["world_layouts"]) == (layout,)
                    assert tuple(result.source["shape_layouts"]) == (LAYOUTS.index(layout),) * 2
                    assert tuple(result.source["shape_rows"]) == (0, -1)
                    assert result.source["shape_groups"][0] == group
                    np.testing.assert_allclose(result.rows[0].origin_m, anchor, atol=0, rtol=0)
                    np.testing.assert_allclose(result.source["pose"][0, :3], position - anchor, atol=2e-6, rtol=0)
                    np.testing.assert_allclose(result.source["pose"][0, 3:].reshape(3, 3), rotation, atol=2e-6, rtol=0)
                    assert k0.canonical(batch.snapshot()) == k0.canonical(record["before"])
                    if "affine" in layout:
                        checkpoint = record["before"][world]
                        assert len(checkpoint.full_position) > len(checkpoint.position)


@pytest.mark.tyx
@pytest.mark.parametrize("layout", ("floating", "floating_affine"))
@pytest.mark.parametrize("family", ("SS", "CC", "HH"))
def test_free_root_body_zero_is_not_dropped(layout, family, tyx_mode, tmp_path):
    tree, world = declaration(layout), 107
    with patch.saved(tmp_path, f"B1-root-{layout}-{family}", tyx_mode, dict(layout=layout, family=family)) as record:
        with ph.TyxWorldBatch([(world, tree)], tyx_mode, cache_root=tmp_path) as batch:
            case, expected, shapes, bindings, anchor, position, rotation, group = installation(batch, tree, layout, family, world, root=True)
            with batch.prepare_collision(ph.CollisionModel(tuple(shapes)), tuple(bindings)) as broad:
                with closing(batch.prepare_convex_query(broad, pairs=(tuple(shape.key for shape in shapes),), site_capacity=16)) as plan:
                    result = k0.query_attempt(batch, plan, record, "root")
                    patch.assert_patch(result.rows[0], expected, case)
                    assert result.source["shape_groups"][0] == 0
                    np.testing.assert_allclose(result.source["pose"][0, :3], 0, atol=0, rtol=0)
                    np.testing.assert_allclose(result.rows[0].origin_m, anchor, atol=0, rtol=0)


@pytest.mark.tyx
def test_sparse_mixed_layout_batch_and_epoch(tyx_mode, tmp_path):
    worlds = (83, 17, 107, 31)
    trees = tuple(declaration(layout) for layout in LAYOUTS)
    entries = list(zip(worlds, trees, strict=True))
    with patch.saved(tmp_path, "B1-mixed-layouts", tyx_mode, dict(worlds=worlds, layouts=LAYOUTS)) as record:
        with ph.TyxWorldBatch(entries[::-1], tyx_mode, cache_root=tmp_path) as batch:
            shapes, bindings, expected_by_pair = [], [], {}
            for world, tree, layout in zip(worlds, trees, LAYOUTS, strict=True):
                case, expected, shape, binding, *_ = installation(batch, tree, layout, "HH", world)
                shapes.extend(shape)
                bindings.extend(binding)
                expected_by_pair[tuple(s.key for s in shape)] = (case, expected)
            with batch.prepare_collision(ph.CollisionModel(tuple(shapes[::-1])), tuple(bindings[::-1])) as broad:
                with closing(batch.prepare_convex_query(broad, pairs=tuple(expected_by_pair), site_capacity=32)) as plan:
                    first = k0.query_attempt(batch, plan, record, "mixed")
                    assert len(first.rows) == 4
                    for row in first.rows:
                        case, expected = expected_by_pair[row.pair]
                        patch.assert_patch(row, expected, case)
                    for world, layout in zip(worlds, LAYOUTS, strict=True):
                        assert first.source["world_layouts"][first.world_keys.index(world)] == layout
                    batch.reset([17])
                    with pytest.raises(ValueError, match="stale"):
                        plan.require_current(first)
                    second = k0.query_attempt(batch, plan, record, "after-reset")
                    assert second.state_epoch != first.state_epoch
                    for row in second.rows:
                        case, expected = expected_by_pair[row.pair]
                        patch.assert_patch(row, expected, case)
                    assert plan.require_current(second) is second


@pytest.mark.tyx
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("fault", ("owner", "world", "epoch", "layout"))
def test_source_loan_identity_rejects_before_query(layout, fault, tyx_mode, tmp_path, monkeypatch):
    tree, world = declaration(layout), 17
    with patch.saved(tmp_path, f"B1-source-fault-{layout}-{fault}", tyx_mode, dict(layout=layout, fault=fault)) as record:
        with ph.TyxWorldBatch([(world, tree)], tyx_mode, cache_root=tmp_path) as batch:
            case, expected, shapes, bindings, *_ = installation(batch, tree, layout, "SS", world)
            with batch.prepare_collision(ph.CollisionModel(tuple(shapes)), tuple(bindings)) as broad:
                with closing(batch.prepare_convex_query(broad, pairs=(tuple(s.key for s in shapes),), site_capacity=4)) as plan:
                    accepted = k0.query_attempt(batch, plan, record, "before-fault")
                    before, broad_before = batch.snapshot(), broad.last_result
                    original = batch._borrow_collision_state
                    attribute = ("tree", "affine_tree", "floating_tree", "floating_affine_tree")[LAYOUTS.index(layout)]

                    @contextmanager
                    def invalid(*args, **kwargs):
                        with original(*args, **kwargs) as loan:
                            source = getattr(loan, attribute)
                            if fault == "owner":
                                source = replace(source, owner=object())
                            elif fault == "world":
                                source = replace(source, rows={world + 1: 0})
                            elif fault == "epoch":
                                source = replace(source, state_epoch=source.state_epoch + 1)
                            else:
                                source = None
                            yield replace(loan, **{attribute: source})

                    monkeypatch.setattr(batch, "_borrow_collision_state", invalid)
                    with pytest.raises(ValueError, match="source|layout"):
                        k0.query_attempt(batch, plan, record, "bad-source")
                    assert plan.last_result is accepted and broad.last_result is broad_before
                    assert k0.canonical(batch.snapshot()) == k0.canonical(before)
                    monkeypatch.setattr(batch, "_borrow_collision_state", original)
                    retry = k0.query_attempt(batch, plan, record, "retry")
                    patch.assert_patch(retry.rows[0], expected, case)


@pytest.mark.tyx
def test_loan_expires_and_result_cannot_cross_owner(tyx_mode, tmp_path):
    case, spec = patch.reference().polygon_case("partial_bb")
    with patch.prepared(case, tyx_mode, tmp_path) as (batch, broad, plan, before):
        result = batch.convex_query(plan)
        with broad._borrow_convex_state() as loan:
            assert loan._lease.active
        with pytest.raises(RuntimeError, match="expired"):
            loan.stage_broadphase()
        with patch.prepared(case, tyx_mode, tmp_path) as (other, other_broad, other_plan, _):
            with pytest.raises(ValueError, match="another|stale"):
                other_plan.require_current(result)
            with pytest.raises(ValueError, match="another batch"):
                other.convex_query(plan)
