"""ROB1a registration/layout contract. Authoring only; no device runtime required."""

from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import numpy as np
import pytest

import phymeo as ph
from phymeo.backends.resident_scene import (
    ResidentParticipant,
    ResidentParticipantBinding,
    ResidentParticipantKey,
    ResidentParticipantLayout,
)

ASSET = Path(__file__).parent / "fixtures/urdf/floating_mimic_jaws.urdf"


def fixed():
    model = ph.RobotModel(
        np.array([-1, 0]),
        ("revolute", "prismatic"),
        np.array([[0.0, 0, 0], [0.25, 0, 0]]),
        np.tile([0.0, 0, 0, 1], (2, 1)),
        np.array([[0.0, 0, 1], [1.0, 0, 0]]),
        np.array([2.0, 1.0]),
        np.array([[0.125, 0, 0], [0.0625, 0, 0]]),
        np.full((2, 3), 0.125),
        np.zeros(3),
    )
    program = ph.DrivenRobotProgram(
        model,
        kp=np.array([3.0, 4.0]),
        kd=np.array([0.5, 0.75]),
        effort_limit=np.array([8.0, 9.0]),
        delay_steps=1,
    )
    state = program.initial_state(
        ph.RobotState(np.array([0.125, 0.03125]), np.array([0.25, -0.125]))
    )
    return ph.FixedRobotTree(program, state)


def floating():
    model = ph.FloatingRobotModel(
        ph.RigidBodyModel(np.array([0.5]), np.full((1, 3), 0.125), np.zeros(3)),
        np.array([0.03125, 0, 0]),
    )
    state = ph.FloatingRobotState(
        np.array([0.5, 0.25, 1.0, 0, 0, 0, 1]),
        np.array([0.125, 0, 0, 0, 0.25, 0]),
    )
    return ph.FloatingRobotTree(
        model,
        state,
        kp=np.zeros(0),
        kd=np.zeros(0),
        effort_limit=np.zeros(0),
    )


def affine():
    asset = ph.load_mimic_urdf(ASSET, gravity=(0.0, 0.0, 0.0))
    return ph.AffineRobotTree(
        asset,
        ph.RobotState(np.array([0.001]), np.array([0.01])),
        np.array([2.0]),
        np.array([0.25]),
        np.array([1.0]),
        delay_steps=1,
    )


def floating_affine():
    asset = ph.load_floating_mimic_urdf(ASSET, gravity=(0.0, 0.0, 0.0))
    state = ph.FloatingRobotState(
        np.array([0.5, 0.25, 1.0, 0, 0, 0, 1, 0.001]),
        np.array([0.125, 0, 0, 0, 0.25, 0, 0.01]),
    )
    return ph.FloatingAffineRobotTree(
        asset,
        state,
        kp=np.array([2.0]),
        kd=np.array([0.25]),
        effort_limit=np.array([1.0]),
        delay_steps=1,
    )


def participant(kind="fixed", *, world=503, identity=7, bodies=None):
    system = globals()[kind]()
    counts = {"fixed": 2, "affine": 2, "floating": 1, "floating_affine": 3}
    if bodies is None:
        bodies = tuple(101 + i for i in range(counts[kind]))
    return ResidentParticipant(ResidentParticipantKey(world, identity), system, bodies)


def sphere_binding(owner, *, group=0, shape=7, body=None, topology=None):
    body = owner.body_ids[group] if body is None else body
    key = ph.CollisionKey(owner.key.world, body, shape)
    collision = ph.CollisionShape(key, "sphere", (11,), radius=0.125)
    cls = {
        "fixed": ph.ResidentTreeBinding,
        "affine": ph.ResidentAffineTreeBinding,
        "floating": ph.ResidentFloatingTreeBinding,
        "floating_affine": ph.ResidentFloatingAffineTreeBinding,
    }[owner.family]
    kwargs = {"topology": owner.compiled.topology if topology is None else topology}
    if owner.family == "floating_affine":
        kwargs["graph_token"] = owner.compiled.graph_token
    bound = cls(key, owner.key.world, group, np.zeros((1, 3)), **kwargs)
    return collision, ResidentParticipantBinding(owner.key, bound)


@pytest.mark.parametrize(
    "world,identity", [(True, 1), (1, False), (-1, 1), (1, -1), (1.0, 1), (1, 2**31)]
)
def test_key_rejects_non_i32_identity(world, identity):
    with pytest.raises(ValueError):
        ResidentParticipantKey(world, identity)


def test_key_is_ordered_and_immutable():
    first = ResidentParticipantKey(np.int64(503), np.int32(7))
    assert first == ResidentParticipantKey(503, 7)
    assert type(first.world) is type(first.participant) is int
    assert first < ResidentParticipantKey(503, 9)
    with pytest.raises(FrozenInstanceError):
        first.world = 8


@pytest.mark.parametrize(
    "kind,counts",
    [
        ("fixed", (2, 2, 2, 2, 2)),
        ("affine", (1, 1, 2, 2, 2)),
        ("floating", (7, 6, 1, 7, 6)),
        ("floating_affine", (8, 7, 3, 9, 8)),
    ],
)
def test_all_families_keep_independent_and_physical_dimensions(kind, counts):
    owner = participant(kind)
    layout = ResidentParticipantLayout([owner])
    row = layout.entries[owner.key]
    assert (
        row.configuration_count,
        row.velocity_count,
        row.body_count,
        row.physical_configuration_count,
        row.physical_velocity_count,
    ) == counts
    assert row.family == kind and row.storage_row == 0
    assert row.configuration_offset == row.velocity_offset == row.body_offset == 0
    assert row.topology == owner.compiled.topology
    assert row.model_token == owner.model_token
    assert row.graph_token == owner.graph_token
    assert layout.require_current() is None


def test_sparse_worlds_have_separate_q_v_body_offsets_and_storage_rows():
    a = participant("fixed", identity=3, bodies=(10, 11))
    b = participant("floating", identity=9, bodies=(20,))
    c = participant("affine", world=901, identity=3, bodies=(10, 11))
    d = participant("floating_affine", world=901, identity=8, bodies=(20, 21, 22))
    layout = ResidentParticipantLayout([d, b, c, a], static_body_ids=[(901, 90), (503, 90)])
    assert layout.participant_keys == (a.key, b.key, c.key, d.key)
    assert layout.world_keys == (503, 901)
    rows = [layout.entries[key] for key in layout.participant_keys]
    assert [(r.configuration_offset, r.velocity_offset, r.body_offset) for r in rows] == [
        (0, 0, 0),
        (2, 2, 2),
        (9, 8, 3),
        (10, 9, 5),
    ]
    assert (layout.configuration_count, layout.velocity_count, layout.body_count) == (18, 16, 8)
    assert layout.worlds[503].participant_keys == (a.key, b.key)
    assert (
        layout.worlds[901].configuration_offset,
        layout.worlds[901].velocity_offset,
        layout.worlds[901].body_offset,
    ) == (9, 8, 3)
    assert layout.body_owners[(901, 21)] == (d.key, 1)
    assert layout.body_keys[(d.key, 1)] == (901, 21)
    assert layout.body_slot(d.key, 1) == 3
    assert layout.family_rows["fixed"] == (a.key,)
    reverse = ResidentParticipantLayout([a, c, b, d], static_body_ids=[(503, 90), (901, 90)])
    assert layout.fingerprint == reverse.fingerprint
    assert tuple(layout.entries.values()) == tuple(reverse.entries.values())
    with pytest.raises(TypeError):
        layout.entries[a.key] = rows[0]
    with pytest.raises(FrozenInstanceError):
        rows[0].body_offset = 99


def test_family_rows_include_multiple_worlds_and_participants_without_alias():
    owners = [
        participant(world=503, identity=7, bodies=(10, 11)),
        participant(world=503, identity=8, bodies=(20, 21)),
        participant(world=901, identity=7, bodies=(10, 11)),
    ]
    layout = ResidentParticipantLayout(owners[::-1])
    assert [layout.entries[p.key].storage_row for p in owners] == [0, 1, 2]
    assert layout.body_owners[(503, 10)] != layout.body_owners[(901, 10)]


@pytest.mark.parametrize("body_ids", [(1,), (1, 1), (1, True), (1, -1), (1, 2**31)])
def test_participant_requires_complete_unique_body_ids(body_ids):
    with pytest.raises(ValueError):
        participant(bodies=body_ids)


def test_duplicate_participants_cross_owner_bodies_and_empty_layout_rejected():
    a = participant()
    with pytest.raises(ValueError, match="duplicate participant"):
        ResidentParticipantLayout([a, a])
    with pytest.raises(ValueError, match="body"):
        ResidentParticipantLayout([a, participant(identity=8)])
    with pytest.raises(ValueError):
        ResidentParticipantLayout([])
    with pytest.raises(TypeError):
        ResidentParticipant(a.key, object(), a.body_ids)


def test_capacity_is_per_world_and_static_owner_is_distinct():
    owners = [participant(identity=i, bodies=(10 * i, 10 * i + 1)) for i in range(9)]
    layout = ResidentParticipantLayout(owners[:8], static_body_ids=[(503, 1000)])
    assert len(layout.participant_keys) == 8 and layout.body_count == 16
    with pytest.raises(ValueError, match="eight"):
        ResidentParticipantLayout(owners)
    with pytest.raises(ValueError, match="static"):
        ResidentParticipantLayout(owners[:1], static_body_ids=[(503, 0)])
    with pytest.raises(ValueError):
        ResidentParticipantLayout(owners[:1], static_body_ids=[(901, 99)])
    with pytest.raises(ValueError):
        ResidentParticipantLayout(owners[:1], static_body_ids=[(503, 99), (503, 99)])


def test_compiled_snapshot_is_owned_and_source_changes_are_rejected():
    system = fixed()
    bodies = [10, 11]
    owner = ResidentParticipant(ResidentParticipantKey(503, 7), system, bodies)
    before = owner.compiled.initial_state.tobytes()
    bodies[0] = 100
    assert owner.body_ids == (10, 11)
    with pytest.raises(ValueError):
        owner.compiled.mass.flags.writeable = True
    with pytest.raises(ValueError):
        owner.compiled.initial_state[0, 0] = 99
    system.state.domains[system.program.domain].q[0] += 1 / 1024
    assert owner.compiled.initial_state.tobytes() == before
    with pytest.raises(ValueError, match="changed"):
        owner.require_current()
    with pytest.raises(ValueError):
        ResidentParticipantLayout([owner])


def test_original_model_and_affine_graph_remain_bound():
    ordinary = participant()
    ordinary.system.program.model.mass.flags.writeable = True
    ordinary.system.program.model.mass[0] += 0.25
    with pytest.raises(ValueError, match="changed"):
        ordinary.require_current()
    source = affine()
    owner = ResidentParticipant(ResidentParticipantKey(503, 8), source, (30, 31))
    object.__setattr__(source, "_graph_token", ("wrong-graph",))
    with pytest.raises(ValueError):
        owner.require_current()


@pytest.mark.parametrize("kind", ["fixed", "affine", "floating", "floating_affine"])
def test_geometry_binding_preserves_family_source_and_full_body(kind):
    owner = participant(kind)
    group = len(owner.body_ids) - 1
    shape, binding = sphere_binding(owner, group=group)
    layout = ResidentParticipantLayout([owner])
    registration = layout.bind_geometry(ph.CollisionModel((shape,)), (binding,))
    assert registration.shape_keys == (shape.key,)
    assert registration.owners[shape.key] == (owner.key, group)
    assert registration.compiled_bindings[shape.key].group_index == group
    assert registration.require_current(layout) is None
    with pytest.raises(TypeError):
        registration.owners[shape.key] = None
    with pytest.raises(ValueError):
        registration.compiled_bindings[shape.key].points_m.flags.writeable = True


def test_same_local_group_across_participants_is_not_same_body():
    a = participant(identity=7, bodies=(10, 11))
    b = participant(identity=8, bodies=(20, 21))
    sa, ba = sphere_binding(a)
    sb, bb = sphere_binding(b)
    layout = ResidentParticipantLayout([b, a])
    result = layout.bind_geometry(ph.CollisionModel((sb, sa)), (bb, ba))
    assert result.owners[sa.key] == (a.key, 0)
    assert result.owners[sb.key] == (b.key, 0)
    assert result.shape_keys == (sa.key, sb.key)
    assert result.require_current(layout) is None


def test_geometry_rejects_wrong_participant_group_body_and_topology():
    a = participant()
    layout = ResidentParticipantLayout([a])
    shape, bound = sphere_binding(a)
    with pytest.raises(ValueError, match="participant"):
        layout.bind_geometry(
            ph.CollisionModel((shape,)),
            (ResidentParticipantBinding(ResidentParticipantKey(503, 99), bound.binding),),
        )
    with pytest.raises(ValueError, match="body"):
        wrong = replace(bound.binding, group_index=1)
        layout.bind_geometry(
            ph.CollisionModel((shape,)), (ResidentParticipantBinding(a.key, wrong),)
        )
    bad_shape, bad_binding = sphere_binding(a, topology=("wrong",))
    with pytest.raises(ValueError, match="topology"):
        layout.bind_geometry(ph.CollisionModel((bad_shape,)), (bad_binding,))
    with pytest.raises(ValueError, match="world"):
        ResidentParticipantBinding(ResidentParticipantKey(901, 7), bound.binding)


def test_geometry_requires_exact_binding_membership_and_family():
    a = participant()
    layout = ResidentParticipantLayout([a])
    shape, bound = sphere_binding(a)
    model = ph.CollisionModel((shape,))
    with pytest.raises(ValueError):
        layout.bind_geometry(model, ())
    with pytest.raises(ValueError):
        layout.bind_geometry(model, (bound, bound))
    wrong = ph.ResidentFloatingTreeBinding(
        shape.key, 503, 0, np.zeros((1, 3)), topology=a.compiled.topology
    )
    with pytest.raises(TypeError, match="family"):
        layout.bind_geometry(model, (ResidentParticipantBinding(a.key, wrong),))


def test_static_shapes_have_explicit_owners_and_immutable_source():
    a = participant()
    dynamic, db = sphere_binding(a)
    key = ph.CollisionKey(503, 90, 8)
    static = ph.CollisionShape(key, "sphere", (21,), radius=0.125)
    points = np.array([[0.5, 0.0, 0.0]])
    sb = ph.ResidentStaticBinding(key, ph.CollisionMotion(points))
    layout = ResidentParticipantLayout([a], static_body_ids=[(503, 90)])
    result = layout.bind_geometry(ph.CollisionModel((static, dynamic)), (sb, db))
    points[0, 0] = 99
    assert result.owners[key] is None
    np.testing.assert_array_equal(result.compiled_bindings[key].start, [[0.5, 0, 0]])
    with pytest.raises(ValueError):
        result.compiled_bindings[key].start.flags.writeable = True
    with pytest.raises(ValueError, match="static"):
        ResidentParticipantLayout([a]).bind_geometry(ph.CollisionModel((static, dynamic)), (sb, db))
    moving = ph.ResidentStaticBinding(key, ph.CollisionMotion([[0.5, 0, 0]], [[0.75, 0, 0]]))
    with pytest.raises(ValueError, match="static"):
        layout.bind_geometry(ph.CollisionModel((static, dynamic)), (moving, db))


def test_geometry_registration_rejects_foreign_layout_and_source_revision_change():
    a = participant()
    shape, bound = sphere_binding(a)
    model = ph.CollisionModel((shape,))
    layout = ResidentParticipantLayout([a])
    result = layout.bind_geometry(model, (bound,))
    with pytest.raises(ValueError, match="owner"):
        result.require_current(ResidentParticipantLayout([a]))
    object.__setattr__(shape, "revision", shape.revision + 1)
    with pytest.raises(ValueError, match="changed"):
        result.require_current(layout)


def test_empty_geometry_is_registration_only_not_fake_step():
    layout = ResidentParticipantLayout([participant()])
    registered = layout.bind_geometry(ph.CollisionModel(()), ())
    assert registered.shape_keys == () and dict(registered.owners) == {}
    assert not hasattr(layout, "step")


def test_from_asset_retains_original_shape_and_graph_proof():
    owner = participant("floating_affine")
    asset = owner.system.asset
    collider = next(item for item in asset.tree.collisions if item.link_index == -1)
    key = ph.CollisionKey(owner.key.world, owner.body_ids[0], collider.shape.key.shape)
    original = ph.ResidentFloatingAffineTreeBinding.from_asset(
        asset,
        collider.shape.key,
        key=key,
        world=owner.key.world,
    )
    shape = replace(collider.shape, key=key)
    layout = ResidentParticipantLayout([owner])
    wrapped = ResidentParticipantBinding(owner.key, original)
    valid = layout.bind_geometry(ph.CollisionModel((shape,)), (wrapped,))
    assert valid.owners[key] == (owner.key, 0)
    assert wrapped.binding._source_shape == original._source_shape
    assert wrapped.binding.graph_token == original.graph_token
    with pytest.raises(ValueError, match="source shape"):
        layout.bind_geometry(
            ph.CollisionModel((replace(shape, radius=shape.radius * 2),)), (wrapped,)
        )
    wrong = replace(original, graph_token=("foreign-graph",))
    with pytest.raises(ValueError, match="graph"):
        layout.bind_geometry(
            ph.CollisionModel((shape,)), (ResidentParticipantBinding(owner.key, wrong),)
        )
