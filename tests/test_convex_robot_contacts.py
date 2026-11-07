"""Convex families drive actual fixed and floating owners in one scene transaction."""

from dataclasses import replace
from itertools import combinations_with_replacement, product

import numpy as np
import pytest

from phymeo.builders.floating_urdf import load_floating_urdf
from phymeo.builders.urdf import load_urdf
from phymeo.constraints.collision_shapes import CollisionModel
from phymeo.constraints.robot_contact import RobotContactMaterial
from phymeo.core.robot import RobotState
from phymeo.dynamics.floating_robot import FloatingRobotState
from phymeo.runtime._floating_robot_participant import FloatingRobotCommand
from phymeo.runtime.contact_scene import ContactSceneBuilder
from phymeo.runtime.robot_scene import RobotCommand

H = 0.001
KINDS = ("sphere", "capsule", "box", "hull")
PAIRS = tuple(combinations_with_replacement(KINDS, 2))
LAW = RobotContactMaterial(100, 0, 0)


def _vertices(kind):
    if kind == "sphere":
        return np.array([[0.0, 0.0, 0.0]]), 0.05
    if kind == "capsule":
        return np.array([[-0.5, 0, 0], [0.5, 0, 0]]), 0.05
    if kind == "box":
        return np.array(list(product((-0.6, 0.6), (-0.1, 0.1), (-0.1, 0.1)))), 0.0
    ring = ((-0.6, 0), (-0.4, -0.1), (0.4, -0.1), (0.6, 0), (0.4, 0.1), (-0.4, 0.1))
    return np.array([(x, y, z) for x, y in ring for z in (-0.1, 0.1)]), 0.0


def _asset(path, kind, mass, floating, z, offset=(0, 0, 0)):
    inertia = f'<inertial><mass value="{mass}"/><inertia ixx="1" iyy="1" izz="1" ixy="0" ixz="0" iyz="0"/></inertial>'
    collision = '<collision name="shape"><geometry><sphere radius=".05"/></geometry></collision>'
    if floating:
        source = f'<robot name="convex"><link name="base">{inertia}{collision}</link></robot>'
    else:
        source = f'<robot name="convex"><link name="base"/><link name="tool">{inertia}{collision}</link><joint name="slide" type="prismatic"><parent link="base"/><child link="tool"/><origin xyz="0 0 {z}"/><axis xyz="0 0 1"/><limit lower="-1" upper="1" effort="128" velocity="10"/></joint></robot>'
    path.write_text(source, encoding="utf-8")
    asset = (load_floating_urdf if floating else load_urdf)(path, gravity=(0, 0, 0))
    original = asset.collisions[0]
    vertices, radius = _vertices(kind)
    vertices = vertices + offset
    shape = replace(
        original.shape,
        kind="hull" if kind == "box" else kind,
        vertex_ids=tuple(range(17, 17 + len(vertices))),
        radius=radius,
    )
    collider = replace(original, shape=shape, local_vertices=tuple(map(tuple, vertices)))
    return replace(asset, collisions=(collider,), collision_model=CollisionModel((shape,)))


def _scene(
    tmp_path, kinds, owners, gap=-0.01, velocity=(0.0, 0.0), reverse=False, offset=(0, 0, 0)
):
    builder = ContactSceneBuilder(gravity=(0, 0, 0))
    height = sum(0.05 if k in ("sphere", "capsule") else 0.1 for k in kinds) + gap
    actions = {}
    order = (1, 0) if reverse else (0, 1)
    for i in order:
        body, mass, z, floating = (10, 20)[i], (2.0, 3.0)[i], (0.0, height)[i], owners[i]
        path = tmp_path / f"body{body}.urdf"
        asset = _asset(path, kinds[i], mass, floating, z, offset)

        def loader(path, *, gravity, asset=asset):
            return asset

        if floating:
            builder.add_floating_urdf(
                body,
                path,
                robot_state=FloatingRobotState(
                    np.array([0, 0, z, 0, 0, 0, 1.0]), np.array([0, 0, velocity[i], 0, 0, 0.0])
                ),
                kp=[],
                kd=[],
                loader=loader,
            )
            actions[body] = FloatingRobotCommand(None)
        else:
            builder.add_urdf(
                body,
                path,
                robot_state=RobotState(np.zeros(1), np.array([velocity[i]])),
                kp=[0],
                kd=[0],
                loader=loader,
            )
            actions[body] = RobotCommand([0])
    builder.pair(10, 20, LAW)
    return builder, actions


@pytest.mark.parametrize("kinds", PAIRS)
@pytest.mark.parametrize("owners", ((False, False), (False, True), (True, True)))
def test_family_patch_accelerates_both_actual_owners(tmp_path, kinds, owners):
    builder, actions = _scene(tmp_path, kinds, owners)
    with builder.build() as scene:
        out = scene.step(H, actions=actions)
        assert out.loads.contacts
        assert sum(row.force_n for row in out.loads.contacts) == pytest.approx(1.0, abs=1e-10)
        for i, body in enumerate((10, 20)):
            obs = out.robots[0, body]
            force = (-1.0, 1.0)[i]
            np.testing.assert_allclose(obs.link_wrenches_world[0, :3], (0, 0, force), atol=1e-10)
            assert obs.v[2 if owners[i] else 0] == pytest.approx(
                H * force / (2.0, 3.0)[i], abs=1e-12
            )
            np.testing.assert_allclose(obs.fixed_root_contact_wrench, 0, atol=1e-12)
        force = sum(
            np.array(row["force_on_first_N"]) + row["force_on_second_N"] for row in out.contacts
        )
        np.testing.assert_allclose(force, 0, atol=1e-12)


@pytest.mark.parametrize("kinds", (("capsule", "capsule"), ("capsule", "box"), ("box", "hull")))
def test_separation_touch_velocity_and_registration_order(tmp_path, kinds):
    for gap in (0.02, 0.0):
        folder = tmp_path / str(gap)
        folder.mkdir()
        builder, actions = _scene(folder, kinds, (False, True), gap=gap, velocity=(0.1, -0.2))
        with builder.build() as scene:
            out = scene.step(H, actions=actions)
            assert not out.loads.contacts
            assert out.robots[0, 10].v[0] == pytest.approx(0.1)
            assert out.robots[0, 20].v[2] == pytest.approx(-0.2)
    a = tmp_path / "forward"
    b = tmp_path / "reverse"
    a.mkdir()
    b.mkdir()
    first, act1 = _scene(a, kinds, (False, True), velocity=(0.1, -0.2))
    second, act2 = _scene(b, kinds, (False, True), velocity=(0.1, -0.2), reverse=True)
    with first.build() as one, second.build() as two:
        out1, out2 = one.step(H, actions=act1), two.step(H, actions=act2)
        for body in (10, 20):
            np.testing.assert_array_equal(out1.robots[0, body].q, out2.robots[0, body].q)
            np.testing.assert_array_equal(out1.robots[0, body].v, out2.robots[0, body].v)


def test_offset_convex_load_accelerates_real_root_rotation(tmp_path):
    builder, actions = _scene(tmp_path, ("box", "box"), (True, True), offset=(0.2, 0, 0))
    with builder.build() as scene:
        out = scene.step(H, actions=actions)
        for i, body in enumerate((10, 20)):
            sign = (-1, 1)[i]
            obs = out.robots[0, body]
            np.testing.assert_allclose(
                obs.link_wrenches_world[0], (0, 0, sign, 0, -0.2 * sign, 0), atol=1e-10
            )
            np.testing.assert_allclose(
                obs.v, (0, 0, sign * H / (2, 3)[i], 0, -0.2 * sign * H, 0), atol=1e-12
            )
        total = sum(
            np.cross(row["application_point_first_m"], row["force_on_first_N"])
            + np.cross(row["application_point_second_m"], row["force_on_second_N"])
            for row in out.contacts
        )
        np.testing.assert_allclose(total, 0, atol=1e-12)


@pytest.mark.parametrize("failure", ("second_candidate", "event"))
def test_non_sphere_trial_failure_keeps_both_owners_and_retry(tmp_path, monkeypatch, failure):
    builder, actions = _scene(tmp_path, ("capsule", "hull"), (False, True))
    with builder.build() as scene, builder.build() as fresh:
        old, geometry = scene.observe_scene()
        events = scene.contact_events
        target = (
            scene._participants[0, 20] if failure == "second_candidate" else scene._event_adapter
        )
        method = "integrate" if failure == "second_candidate" else "stage"

        def fail(*args, **kwargs):
            raise RuntimeError("injected common convex failure")

        with monkeypatch.context() as patcher:
            patcher.setattr(target, method, fail)
            with pytest.raises(RuntimeError, match="common convex"):
                scene.step(H, actions=actions)
        current, after_geometry = scene.observe_scene()
        assert current.step_number == old.step_number == 0 and scene.contact_events is events
        assert current.loads.contacts == old.loads.contacts and current.contacts == old.contacts
        for owner in current.robots:
            np.testing.assert_array_equal(current.robots[owner].q, old.robots[owner].q)
            np.testing.assert_array_equal(current.robots[owner].v, old.robots[owner].v)
        for key in geometry.poses:
            np.testing.assert_array_equal(
                after_geometry.poses[key].start, geometry.poses[key].start
            )
        retried = scene.step(H, actions=actions)
        expected = fresh.step(H, actions=actions)
        for owner in retried.robots:
            np.testing.assert_array_equal(retried.robots[owner].q, expected.robots[owner].q)
            np.testing.assert_array_equal(retried.robots[owner].v, expected.robots[owner].v)
        assert (
            retried.contacts == expected.contacts and scene.contact_events == fresh.contact_events
        )


@pytest.mark.parametrize("kinds", (("capsule", "capsule"), ("capsule", "box"), ("hull", "hull")))
def test_continuous_approach_compression_and_release_from_real_velocity(tmp_path, kinds):
    import json

    builder, actions = _scene(tmp_path, kinds, (False, False), gap=0.02, velocity=(0.08, -0.12))
    q = np.array([0.0, 0.0])
    v = np.array([0.08, -0.12])
    mass = np.array([2.0, 3.0])
    # Relative speed .2 and omega=sqrt(100*(1/2+1/3)) give
    # max layer depth .2/omega < .022 < radius .05. Contact lasts
    # half a period, not a quarter; birth .1 + pi/omega < .445 s.
    dt = 1 / 256
    rows = []
    with builder.build() as scene:
        for step in range(128):
            gap = 0.02 + q[1] - q[0]
            force = 100 * max(0, -gap)
            v += dt * np.array([-force, force]) / mass
            q += dt * v
            out = scene.step(dt, actions=actions)
            actual_force = sum(pair.force_n for pair in out.loads.contacts)
            assert actual_force == pytest.approx(force, abs=1e-9)
            for i, body in enumerate((10, 20)):
                assert out.robots[0, body].v[0] == pytest.approx(v[i], abs=1e-11)
                assert out.robots[0, body].q[0] == pytest.approx(q[i], abs=1e-11)
            rows.append(
                {
                    "step": step + 1,
                    "time_s": out.time_s,
                    "entry_gap_m": gap,
                    "force_n": actual_force,
                    "q": [float(out.robots[0, b].q[0]) for b in (10, 20)],
                    "v": [float(out.robots[0, b].v[0]) for b in (10, 20)],
                    "events": [event.kind for event in scene.contact_events.events],
                }
            )
        forces = [r["force_n"] for r in rows]
        assert forces[0] == 0 and max(forces) > 1 and all(f == 0 for f in forces[-4:])
        assert any("begin" in r["events"] for r in rows)
        assert any("end" in r["events"] for r in rows)
        assert rows[-1]["time_s"] == 0.5
    (tmp_path / "continuous-convex.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


@pytest.mark.parametrize("robot_kind", KINDS)
@pytest.mark.parametrize("body_kind", ("sphere", "box", "hull"))
@pytest.mark.parametrize("floating", (False, True))
def test_same_convex_load_map_drives_robot_and_free_body(tmp_path, robot_kind, body_kind, floating):
    builder = ContactSceneBuilder(gravity=(0, 0, 0))
    path = tmp_path / "robot.urdf"
    asset = _asset(path, robot_kind, 2.0, floating, 0.0)

    def loader(path, *, gravity):
        return asset

    if floating:
        builder.add_floating_urdf(
            10,
            path,
            robot_state=FloatingRobotState(np.array([0, 0, 0, 0, 0, 0, 1.0]), np.zeros(6)),
            kp=[],
            kd=[],
            loader=loader,
        )
        action = FloatingRobotCommand(None)
    else:
        builder.add_urdf(
            10,
            path,
            robot_state=RobotState(np.zeros(1), np.zeros(1)),
            kp=[0],
            kd=[0],
            loader=loader,
        )
        action = RobotCommand([0])
    height = (
        (0.05 if robot_kind in ("sphere", "capsule") else 0.1)
        + (0.05 if body_kind == "sphere" else 0.1)
        - 0.01
    )
    if body_kind == "sphere":
        builder.add_sphere(20, 0.05, mass=3, position=(0, 0, height))
    elif body_kind == "box":
        builder.add_box(
            20, (0.6, 0.1, 0.1), mass=3, position=(0, 0, height), inertia_diagonal=(1, 1, 1)
        )
    else:
        builder.add_convex(
            20, _vertices("hull")[0], mass=3, position=(0, 0, height), inertia_diagonal=(1, 1, 1)
        )
    builder.pair(10, 20, LAW)
    with builder.build() as scene:
        out = scene.step(H, actions={10: action})
        obs = out.robots[0, 10]
        assert obs.v[2 if floating else 0] == pytest.approx(-H / 2, abs=1e-12)
        np.testing.assert_allclose(out.rigid.velocity[0], (0, 0, H / 3), atol=1e-12)
        np.testing.assert_allclose(obs.link_wrenches_world[0, :3], (0, 0, -1), atol=1e-10)
        np.testing.assert_allclose(out.loads.inputs.rigid.force[0], (0, 0, 1), atol=1e-10)
        assert sum(c.force_n for c in out.loads.contacts) == pytest.approx(1, abs=1e-10)


@pytest.mark.parametrize("reverse", (False, True))
def test_existing_unique_sphere_inside_obb_response_is_preserved(tmp_path, reverse):
    builder, actions = _scene(
        tmp_path, ("box", "sphere"), (False, True), gap=-0.07, reverse=reverse
    )
    with builder.build() as scene:
        out = scene.step(H, actions=actions)
        assert sum(c.force_n for c in out.loads.contacts) == pytest.approx(7, abs=1e-10)
        np.testing.assert_allclose(
            out.robots[0, 10].link_wrenches_world[0, :3], (0, 0, -7), atol=1e-10
        )
        np.testing.assert_allclose(
            out.robots[0, 20].link_wrenches_world[0, :3], (0, 0, 7), atol=1e-10
        )
        assert out.robots[0, 10].v[0] == pytest.approx(-7 * H / 2, abs=1e-12)
        assert out.robots[0, 20].v[2] == pytest.approx(7 * H / 3, abs=1e-12)


@pytest.mark.parametrize("enabled", (False, True))
def test_far_flat_hull_cannot_hide_behind_broad_rejection(tmp_path, enabled):
    builder = ContactSceneBuilder(gravity=(0, 0, 0))
    for i, body in enumerate((10, 20)):
        path = tmp_path / f"flat{body}.urdf"
        asset = _asset(path, "hull" if i == 0 else "sphere", 2.0, False, 100.0 * i)
        if i == 0:
            col = asset.collisions[0]
            shape = replace(col.shape, vertex_ids=(17, 18, 19, 20))
            flat = ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0))
            asset = replace(
                asset,
                collisions=(replace(col, shape=shape, local_vertices=flat),),
                collision_model=CollisionModel((shape,)),
            )

        def loader(path, *, gravity, asset=asset):
            return asset

        builder.add_urdf(
            body,
            path,
            robot_state=RobotState(np.zeros(1), np.zeros(1)),
            kp=[0],
            kd=[0],
            loader=loader,
        )
    builder.pair(10, 20, LAW if enabled else None)
    if enabled:
        with pytest.raises(ValueError):
            builder.build()
    else:
        with builder.build() as scene:
            out = scene.step(H, actions={10: RobotCommand([0]), 20: RobotCommand([0])})
            assert not out.loads.contacts


@pytest.mark.parametrize("pad", (False, True))
def test_excluded_general_hull_does_not_change_existing_pair_sampling(tmp_path, pad):
    from phymeo.runtime.robot_payload_scene import PadFace

    outputs = []
    for extra in (False, True):
        builder = ContactSceneBuilder(gravity=(0, 0, 0))
        path = tmp_path / f"excluded-{extra}.urdf"
        asset = _asset(path, "box", 2.0, False, 0.0)

        def loader(path, *, gravity, asset=asset):
            return asset

        pads = (PadFace.named("tool", "shape", 2, 1, ((0.0, 0.0),)),) if pad else ()
        builder.add_urdf(
            10,
            path,
            robot_state=RobotState(np.zeros(1), np.zeros(1)),
            kp=[0],
            kd=[0],
            loader=loader,
            pads=pads,
        )
        builder.add_box(
            20,
            (0.6, 0.1, 0.1),
            mass=3,
            position=(0, 0, 0.19),
            angular_velocity=(0, 0, 0.001),
            inertia_diagonal=(1, 1, 1),
        )
        builder.pair(10, 20, RobotContactMaterial(100, 0, 0.25, slip_speed_m_s=0.01))
        if extra:
            builder.add_convex(
                30, _vertices("hull")[0], mass=3, position=(0, 0, 100), inertia_diagonal=(1, 1, 1)
            )
            builder.pair(10, 30, None)
            builder.pair(20, 30, None)
        with builder.build() as scene:
            out = scene.step(H, actions={10: RobotCommand([0])})
            outputs.append(
                (
                    out.rigid.velocity[0].copy(),
                    out.rigid.angular_velocity[0].copy(),
                    out.robots[0, 10].v.copy(),
                    out.contacts,
                )
            )
    for a, b in zip(outputs[0][:3], outputs[1][:3], strict=True):
        np.testing.assert_array_equal(a, b)
    assert outputs[0][3] == outputs[1][3]


@pytest.mark.parametrize("enabled", (False, True))
def test_rounded_hull_is_rejected_before_enabled_broad_skip(tmp_path, enabled):
    builder = ContactSceneBuilder(gravity=(0, 0, 0))
    for i, body in enumerate((10, 20)):
        path = tmp_path / f"rounded{body}.urdf"
        asset = _asset(path, "hull" if i == 0 else "sphere", 2.0, False, 100.0 * i)
        if i == 0:
            col = asset.collisions[0]
            shape = replace(col.shape, radius=0.01)
            asset = replace(
                asset,
                collisions=(replace(col, shape=shape),),
                collision_model=CollisionModel((shape,)),
            )

        def loader(path, *, gravity, asset=asset):
            return asset

        builder.add_urdf(
            body,
            path,
            robot_state=RobotState(np.zeros(1), np.zeros(1)),
            kp=[0],
            kd=[0],
            loader=loader,
        )
    builder.pair(10, 20, LAW if enabled else None)
    if enabled:
        with pytest.raises(ValueError, match="rounded hull"):
            builder.build()
    else:
        with builder.build() as scene:
            assert not scene.step(
                H, actions={10: RobotCommand([0]), 20: RobotCommand([0])}
            ).loads.contacts


@pytest.mark.parametrize("named", (False, True))
def test_hybrid_filter_preserves_full_asset_pad_identity(tmp_path, named):
    from phymeo.runtime.robot_payload_scene import PadFace

    builder = ContactSceneBuilder(gravity=(0, 0, 0))
    path = tmp_path / "ambiguous.urdf"
    asset = _asset(path, "box", 2.0, False, 0.0)
    col = asset.collisions[0]
    vertices = _vertices("hull")[0] + (3.0, 0.0, 0.0)
    shape = replace(
        col.shape,
        key=replace(col.shape.key, shape=col.shape.key.shape + 1),
        vertex_ids=tuple(range(100, 100 + len(vertices))),
    )
    other = replace(col, shape=shape, local_vertices=tuple(map(tuple, vertices)))
    asset = replace(
        asset, collisions=(col, other), collision_model=CollisionModel((col.shape, shape))
    )

    def loader(path, *, gravity):
        return asset

    pad = (
        PadFace.named("tool", "shape", 2, 1, ((0.0, 0.0),))
        if named
        else PadFace(col.shape.key, 2, 1, ((0.0, 0.0),))
    )
    builder.add_urdf(
        10,
        path,
        robot_state=RobotState(np.zeros(1), np.zeros(1)),
        kp=[0],
        kd=[0],
        loader=loader,
        pads=(pad,),
    )
    builder.add_box(20, (0.6, 0.1, 0.1), mass=3, position=(0, 0, 0.19), inertia_diagonal=(1, 1, 1))
    builder.pair(10, 20, LAW)
    if named:
        with pytest.raises(ValueError, match="ambiguous"):
            builder.build()
    else:
        with builder.build() as scene:
            out = scene.step(H, actions={10: RobotCommand([0])})
            assert out.contacts and all(row["feature"] == "sampled-pad" for row in out.contacts)
