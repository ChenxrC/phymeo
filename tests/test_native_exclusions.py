"""Explicit rules lower exactly; unconfigured native physics stays unchanged."""

from dataclasses import replace

import mujoco
import numpy as np
import pytest
from test_physolve_models import robot_path

from phymeo.builders.native_assembly import (
    NativeAssetInstance,
    NativeStaticShape,
    compile_native_template,
)


def fanuc_template(*, obstacle=False):
    shapes = (NativeStaticShape("obstacle", "box", (2, 2, 0.01), (0, 0, 0.27)),) if obstacle else ()
    return compile_native_template(
        (NativeAssetInstance("fanuc", str(robot_path("fanuc")), drive_mode="scalar_pd"),),
        timestep=0.002,
        gravity=(0, 0, -9.81),
        static_shapes=shapes,
    )


def connection_plan(template):
    source = template.compilation.collision_source
    keys = []
    for name in ("fanuc::base_link", "fanuc::link_1"):
        rows = [row for row in source.geoms if row.body_name == name]
        assert len(rows) == 1
        keys.append(rows[0].key)
    return source.plan_exclusions((tuple(keys),), source_identity=source.identity)


def eligibility(model):
    # Native structural policy, independent of PHYMEO's lowering implementation.
    a, b = np.triu_indices(model.ngeom, 1)
    ba, bb = model.geom_bodyid[a], model.geom_bodyid[b]
    wa, wb = model.body_weldid[ba], model.body_weldid[bb]
    pa = model.body_weldid[model.body_parentid[wa]]
    pb = model.body_weldid[model.body_parentid[wb]]
    parent = (wa != 0) & (wb != 0) & ((wa == pb) | (wb == pa))
    mask = (
        (model.geom_contype[a] & model.geom_conaffinity[b])
        | (model.geom_contype[b] & model.geom_conaffinity[a])
    ) != 0
    active = mask & (wa != wb)
    if not model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_FILTERPARENT:
        active &= ~parent
    active &= ~np.isin((ba << 16) + bb, model.exclude_signature)
    pairs = {tuple(sorted((int(x), int(y)))) for x, y in zip(a[active], b[active])}
    pairs.update(
        tuple(sorted((int(x), int(y)))) for x, y in zip(model.pair_geom1, model.pair_geom2)
    )
    return pairs


def test_fanuc_exact_rule_retains_geometry_parameters_and_obstacle_contacts():
    template = fanuc_template(obstacle=True)
    plan = connection_plan(template)
    filtered = template.with_collision_exclusions(plan)
    before, after = template.compilation.mj_model, filtered.compilation.mj_model
    assert eligibility(after) == eligibility(before) - set(plan.native_geom_pairs)
    # Inspect all native numeric arrays, not only nq/nv or a chosen pose subset.
    for name in dir(before):
        # The new exclude adds a name-table slot; resolved names are invariant,
        # byte offsets/hash buckets are not physical numeric fields.
        if name.startswith(("_", "name_")) or name in {"exclude_signature", "names", "names_map"}:
            continue
        value = getattr(before, name)
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(getattr(after, name), value, err_msg=name)
    for name in dir(before.opt):
        if not name.startswith("_"):
            value = getattr(before.opt, name)
            if isinstance(value, (int, float, np.ndarray)):
                np.testing.assert_array_equal(getattr(after.opt, name), value, err_msg=name)

    def contacts(model):
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        return {tuple(sorted(c.geom)) for c in data.contact}

    original = contacts(before)
    current = contacts(after)
    assert current == original - set(plan.native_geom_pairs)
    obstacle = after.geom("obstacle").id
    link = plan.native_geom_pairs[0][1]
    assert tuple(sorted((obstacle, link))) in current
    assert filtered.initial_state.model_sha256 == filtered.compilation.source.model_sha256
    for (_, old), (_, new) in zip(template.initial_state.fields, filtered.initial_state.fields):
        np.testing.assert_array_equal(old, new)
    assert template.compilation.mj_model.nexclude == 0
    assert filtered.collision_plans == (plan,)


def test_no_rules_identity_stale_and_forged_rules():
    template = fanuc_template()
    source = template.compilation.collision_source
    empty = source.plan_exclusions((), source_identity=source.identity)
    assert template.with_collision_exclusions(empty) is template
    plan = connection_plan(template)
    with pytest.raises(ValueError, match="identity mismatch"):
        template.with_collision_exclusions(replace(plan, source_identity="stale"))
    with pytest.raises(ValueError, match="content"):
        template.with_collision_exclusions(replace(plan, native_geom_pairs=((0, 2),)))
    other = compile_native_template(
        (NativeAssetInstance("fanuc", str(robot_path("fanuc")), drive_mode="scalar_pd"),),
        timestep=0.003,
        gravity=(0, 0, -9.81),
    )
    assert other.compilation.mj_model.nq == template.compilation.mj_model.nq
    foreign = replace(template, initial_state=other.initial_state)
    with pytest.raises(ValueError, match="another compiled model"):
        foreign.with_collision_exclusions(plan)


def test_partial_multishape_body_rule_rejected(tmp_path):
    path = tmp_path / "two.xml"
    path.write_text(
        '<mujoco><worldbody><body name="a"><joint/><geom name="a0" size=".1"/>'
        '<geom name="a1" size=".1" pos="0 0 1"/></body><body name="b">'
        '<joint/><geom name="b0" size=".1"/></body></worldbody></mujoco>'
    )
    template = compile_native_template(
        (NativeAssetInstance("x", str(path)),), timestep=0.002, gravity=(0, 0, -9.81)
    )
    source = template.compilation.collision_source
    keys = {row.geom_name: row.key for row in source.geoms}
    plan = source.plan_exclusions(
        ((keys["x::a0"], keys["x::b0"]),), source_identity=source.identity
    )
    with pytest.raises(NotImplementedError, match="undeclared"):
        template.with_collision_exclusions(plan)
    complete = source.plan_exclusions(
        ((keys["x::a0"], keys["x::b0"]), (keys["x::a1"], keys["x::b0"])),
        source_identity=source.identity,
    )
    filtered = template.with_collision_exclusions(complete)
    assert eligibility(filtered.compilation.mj_model) == eligibility(
        template.compilation.mj_model
    ) - set(complete.native_geom_pairs)


def test_owned_exact_spec_survives_public_mutation_and_resource_removal(tmp_path):
    mesh = tmp_path / "tetra.obj"
    mesh.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\nf 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n")
    path = tmp_path / "asset.xml"
    path.write_text(
        '<mujoco><asset><mesh name="m" file="tetra.obj"/></asset><worldbody>'
        '<body name="a" pos="0.12345678901234567 0 0"><joint/>'
        '<geom name="g0" type="mesh" mesh="m"/></body><body name="b">'
        '<joint/><geom name="g1" size=".1"/></body></worldbody></mujoco>'
    )
    template = compile_native_template(
        (NativeAssetInstance("x", str(path)),), timestep=0.002, gravity=(0, 0, -9.81)
    )
    source = template.compilation.collision_source
    plan = source.plan_exclusions(
        ((source.geoms[0].key, source.geoms[1].key),), source_identity=source.identity
    )
    expected = template.compilation.fresh_native_model()
    template.compilation.mj_spec.body("x::a").pos = [999, 0, 0]
    template.compilation.mj_model.body_pos[:] = 999
    path.unlink()
    mesh.unlink()
    filtered = template.with_collision_exclusions(plan)
    np.testing.assert_array_equal(filtered.compilation.mj_model.body_pos, expected.body_pos)
    np.testing.assert_array_equal(filtered.compilation.mj_model.mesh_vert, expected.mesh_vert)


def test_instance_scope_and_world_rule_rejection():
    path = str(robot_path("fanuc"))
    template = compile_native_template(
        (
            NativeAssetInstance("fanuc", path, drive_mode="scalar_pd"),
            NativeAssetInstance("second", path, position=(3, 0, 0), drive_mode="scalar_pd"),
        ),
        timestep=0.002,
        gravity=(0, 0, -9.81),
        static_shapes=(NativeStaticShape("ground", "box", (4, 4, 0.01), (0, 0, -0.1)),),
    )
    plan = connection_plan(template)
    filtered = template.with_collision_exclusions(plan)
    assert eligibility(filtered.compilation.mj_model) == eligibility(
        template.compilation.mj_model
    ) - set(plan.native_geom_pairs)
    source = template.compilation.collision_source
    ground = next(row.key for row in source.geoms if row.geom_name == "ground")
    link = next(row.key for row in source.geoms if row.body_name == "fanuc::link_1")
    invalid = source.plan_exclusions(((ground, link),), source_identity=source.identity)
    with pytest.raises(NotImplementedError, match="non-world"):
        template.with_collision_exclusions(invalid)
