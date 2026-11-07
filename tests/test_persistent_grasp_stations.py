"""Tests-first: declare original grasp material stations before contact is born.

UNEXECUTED by the author. The parent owns all project execution.
"""

import gzip
import json
import sys
from dataclasses import FrozenInstanceError, replace
from importlib.util import module_from_spec, spec_from_file_location
from itertools import product
from math import ceil
from pathlib import Path

import numpy as np
import pytest

import phymeo as ph
from phymeo.dynamics.robot_kinematics import forward_kinematics, point_jacobians
from phymeo.runtime import _multi_robot_tangent
from phymeo.runtime.material_stations import MaterialStationRegistry
from phymeo.runtime.robot_payload_scene import PadFace
from phymeo.solvers.tangent_history import StationKey

OFFSETS = tuple(product((-0.005, 0.005), repeat=2))
TOKENS = ((101, 107, 131, 149), (211, 239, 263, 307))


def _acquisition():
    name = "phymeo_original_persistent_grasp_acquisition"
    if name not in sys.modules:
        path = Path(__file__).resolve().parents[1] / "examples" / "robot_force_acquisition.py"
        spec = spec_from_file_location(name, path)
        module = module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def _declarations(case):
    if case == "panda-sphere":
        names = tuple(
            (f"panda_{side}finger", f"panda_{side}finger_pad") for side in ("left", "right")
        )
        axis, signs = 1, (-1, 1)
        counter = ((0,), (0,))
    else:
        names = (("left", "left_rubber"), ("right", "right_rubber"))
        axis, signs = 0, (1, -1)
        counter = ((0, 1, 2, 3), (4, 5, 6, 7))
    return dict(
        robot=(0, 10),
        pads=tuple(
            PadFace.named(*name, axis, sign, OFFSETS)
            for name, sign in zip(names, signs, strict=True)
        ),
        station_keys={
            name: tuple(StationKey(0, token) for token in tokens)
            for name, tokens in zip(names, TOKENS, strict=True)
        },
        counter_features=dict(zip(names, counter, strict=True)),
        chart_ids={name: (1001 + i, 2001 + i) for i, name in enumerate(names)},
    )


def _named(pad):
    return pad.shape_key.link_name, pad.shape_key.collision_name


def _registry(scene, info, declaration):
    # Deliberately resolve the new public function only after each real premise.
    return ph.material_stations_from_pads(scene, payload_key=info["primary_key"], **declaration)


def _assert_mimic_open_premise(scene, info, declaration):
    reading, geometry = scene.observe_scene()
    asset = scene.assets[0, 10]
    assert asset.joint_names == ("lift", "screw")
    assert asset.tree.joint_names == ("lift", "screw", "idler")
    np.testing.assert_array_equal(asset.matrix, ((1, 0), (0, 1), (0, -1)))
    np.testing.assert_array_equal(reading.robots[0, 10].q, (0, 0))
    np.testing.assert_array_equal(reading.robots[0, 10].v, (0, 0))
    np.testing.assert_allclose(reading.robots[0, 10].full_q, (0, 0, 0.004), atol=1e-15, rtol=0)
    np.testing.assert_array_equal(reading.robots[0, 10].full_v, (0, 0, 0))
    assert info["dt"] == 0.001
    assert info["masses"] == (0.1, 0.03)
    np.testing.assert_allclose(info["center"], (0.25, -0.125, 0.25), atol=1e-15, rtol=0)
    assert reading.step_number == 0
    assert reading.time_s == geometry.time_s == 0.0
    assert reading.load_time_s is None
    assert scene.tangent_history is None
    assert scene.tangent_receipt is None

    # Independent authored URDF positions/box dimensions, not sampled contact rows.
    world_anchors = []
    for pad, origin_x in zip(declaration["pads"], (0.215, 0.285), strict=True):
        name = _named(pad)
        key = scene.shape_key(10, *name)
        col = next(c for c in asset.tree.collisions if (c.link_name, c.name) == name)
        origin, rotation = asset.authored_link_pose(name[0], reading.robots[0, 10].q)
        np.testing.assert_allclose(origin, (origin_x, -0.125, 0.25), atol=1e-14, rtol=0)
        np.testing.assert_allclose(rotation, np.eye(3), atol=1e-14, rtol=0)
        vertices = np.asarray(geometry.poses[key].start)
        np.testing.assert_allclose(
            vertices, origin + np.asarray(col.local_vertices) @ rotation.T, atol=1e-14, rtol=0
        )
        np.testing.assert_allclose(
            np.ptp(vertices, axis=0), (0.006, 0.024, 0.018), atol=1e-14, rtol=0
        )
        face_x = origin_x + pad.side * 0.003
        for u, v in pad.offsets_m:
            point = np.array((face_x, -0.125 + u, 0.25 + v))
            assert point[0] == pytest.approx(
                vertices[:, 0].max() if pad.side > 0 else vertices[:, 0].min(), abs=1e-14
            )
            assert vertices[:, 1].min() < point[1] < vertices[:, 1].max()
            assert vertices[:, 2].min() < point[2] < vertices[:, 2].max()
            world_anchors.append(point)
    assert len(world_anchors) == 8
    assert len({tuple(p) for p in world_anchors}) == 8
    pairs = {tuple(sorted((key, info["primary_key"]))) for key in info["pad_keys"]}
    assert not any(
        load.force_n > 0 for load in scene.sample().contacts if (load.first, load.second) in pairs
    )
    assert scene.observe().step_number == 0
    return reading


def test_named_pad_registry_predeclares_real_open_mimic_sites():
    scene, _controller, info = _acquisition().make_case("mimic-box")
    with scene:
        declaration = _declarations("mimic-box")
        _assert_mimic_open_premise(scene, info, declaration)
        events = scene.contact_events

        registry = _registry(scene, info, declaration)

        assert isinstance(registry, MaterialStationRegistry)
        assert len(registry.registrations) == 8
        assert tuple(row.key for row in registry.registrations) == tuple(
            sorted(key for keys in declaration["station_keys"].values() for key in keys)
        )
        assert set(registry.pairs) == {
            tuple(sorted((key, info["primary_key"]))) for key in info["pad_keys"]
        }
        assert scene.contact_events is events
        _assert_mimic_open_premise(scene, info, declaration)


def _public(value):
    # Reuse the existing lossless public serializer, never a private solver oracle.
    name = "phymeo_persistent_grasp_public_values"
    if name not in sys.modules:
        path = Path(__file__).resolve().parents[1] / "examples" / "general_scene_friction.py"
        spec = spec_from_file_location(name, path)
        module = module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name].public_payload(value)


def _snapshot(scene):
    reading, geometry = scene.observe_scene()
    return _public(
        dict(
            state=scene.state,
            reading=reading,
            geometry=geometry,
            events=scene.contact_events,
            history=scene.tangent_history,
            receipt=scene.tangent_receipt,
            last_frame=scene.last_frame,
        )
    )


def _authored_points(case, pad):
    # Original URDF dimensions/origins; offsets stay in the author's declared order.
    if case == "panda-sphere":
        center_y = 0.00758 if _named(pad)[0] == "panda_leftfinger" else -0.00758
        return np.array([(u, center_y + pad.side * 0.0076, 0.04525 + v) for u, v in pad.offsets_m])
    return np.array([(pad.side * 0.003, u, v) for u, v in pad.offsets_m])


def _assert_declaration_geometry(scene, info, declaration, registry):
    case = info["case"]
    reading, geometry = scene.observe_scene()
    asset = scene.assets[0, 10]
    tree = getattr(asset, "tree", asset)
    full = reading.robots[0, 10].full_q
    fk = forward_kinematics(tree.model, full)
    shapes = {shape.key: shape for shape in geometry.model.shapes}
    rows = {row.key: row for row in registry.registrations}
    expected_world = []
    for pad in declaration["pads"]:
        name = _named(pad)
        key = scene.shape_key(10, *name)  # Single RobotScene accepts a body integer.
        collider = next(c for c in tree.collisions if (c.link_name, c.name) == name)
        link = tree.sites[name[0]].link_index
        assert link == collider.link_index and 0 <= link < len(full)
        points = _authored_points(case, pad)
        model_points = tree.authored_to_model_points(name[0], points)
        authored_origin, authored_rotation = asset.authored_link_pose(
            name[0], reading.robots[0, 10].q
        )
        world = authored_origin + points @ authored_rotation.T
        np.testing.assert_allclose(
            fk.position[link] + model_points @ fk.rotation[link].T, world, atol=2e-13, rtol=0
        )
        np.testing.assert_allclose(
            geometry.poses[key].start,
            authored_origin + np.asarray(collider.local_vertices) @ authored_rotation.T,
            atol=2e-13,
            rtol=0,
        )
        axis = np.eye(3)[next(i for i in range(3) if i != pad.axis)]
        local_axis = tree.authored_to_model_points(name[0], axis) - tree.authored_to_model_points(
            name[0], np.zeros(3)
        )
        local_axis /= np.linalg.norm(local_axis)
        face_indices = (
            ((4, 5, 6, 7) if pad.side > 0 else (0, 1, 2, 3))
            if pad.axis == 0
            else ((2, 3, 6, 7) if pad.side > 0 else (0, 1, 4, 5))
        )
        feature = tuple(shapes[key].vertex_ids[i] for i in face_indices)
        counter = declaration["counter_features"][name]
        charts = declaration["chart_ids"][name]
        for token, local, actual_world in zip(
            declaration["station_keys"][name], model_points, world, strict=True
        ):
            row = rows[token]
            assert (row.first, row.second) == tuple(sorted((key, info["primary_key"])))
            side = "first" if row.first == key else "second"
            other = "second" if side == "first" else "first"
            assert row.anchor_side == side
            np.testing.assert_allclose(row.anchor_local_m, local, atol=2e-13, rtol=0)
            np.testing.assert_allclose(row.tangent_axis_local, local_axis, atol=2e-13, rtol=0)
            np.testing.assert_allclose(
                fk.position[link] + fk.rotation[link] @ row.anchor_local_m,
                actual_world,
                atol=2e-13,
                rtol=0,
            )
            assert getattr(row, side + "_feature") == feature
            assert getattr(row, other + "_feature") == counter
            assert getattr(row, side + "_chart_id") == charts[0]
            assert getattr(row, other + "_chart_id") == charts[1]
            assert row.match_tolerance_m == 1e-8
            assert row.topology_revision == 0
            expected_world.append(actual_world)
    assert len(expected_world) == 8
    return np.array(expected_world)


@pytest.mark.parametrize("case", ["mimic-box", "panda-sphere"])
def test_pad_declarations_survive_mimic_projection_and_panda_fixed_frames(case, monkeypatch):
    scene, _controller, info = _acquisition().make_case(case)
    with scene:
        declaration = _declarations(case)
        before = _snapshot(scene)

        def forbidden(*args, **kwargs):
            raise AssertionError("declaration must not sample forces or enable/advance a scene")

        with monkeypatch.context() as patch:
            patch.setattr(scene, "sample", forbidden)
            patch.setattr(scene, "step", forbidden)
            patch.setattr(scene, "enable_coupled_tangent", forbidden)
            registry = _registry(scene, info, declaration)
        _assert_declaration_geometry(scene, info, declaration, registry)
        assert _snapshot(scene) == before
        assert scene.tangent_history is None
        if case == "panda-sphere":
            # This is the original nine-coordinate Panda asset, not a mimic surrogate.
            assert len(scene.assets[0, 10].joint_names) == 9
            assert "panda_hand_tcp" in scene.assets[0, 10].sites


@pytest.mark.parametrize("case", ["mimic-box", "panda-sphere"])
def test_declaration_reorder_and_source_mutation_do_not_change_owned_registry(case):
    scene, _controller, info = _acquisition().make_case(case)
    with scene:
        declaration = _declarations(case)
        original = _registry(scene, info, declaration)
        frozen = _public(original)
        reordered = dict(declaration)
        reordered["pads"] = tuple(
            replace(pad, offsets_m=pad.offsets_m[::-1]) for pad in reversed(declaration["pads"])
        )
        reordered["station_keys"] = {
            name: tuple(reversed(keys))
            for name, keys in reversed(tuple(declaration["station_keys"].items()))
        }
        again = _registry(scene, info, reordered)
        assert again is not original
        assert _public(again) == frozen
        _assert_declaration_geometry(scene, info, reordered, again)
        declaration["station_keys"].clear()
        declaration["counter_features"].clear()
        declaration["chart_ids"].clear()
        reordered["station_keys"].clear()
        assert _public(original) == _public(again) == frozen
        with pytest.raises((FrozenInstanceError, AttributeError, TypeError)):
            original.registrations = ()
        assert scene.observe().step_number == 0
        assert scene.tangent_history is None


@pytest.mark.parametrize(
    "invalid",
    [
        "robot-world",
        "robot-body",
        "station-world",
        "payload-world",
        "duplicate-key",
        "missing-chart",
        "unknown-feature",
        "whole-box",
        "missing-pad",
    ],
)
def test_invalid_material_declaration_rejects_without_scene_publication(invalid):
    scene, _controller, info = _acquisition().make_case("mimic-box")
    with scene:
        declaration = _declarations("mimic-box")
        info = dict(info)
        left, right = tuple(declaration["station_keys"])
        if invalid == "robot-world":
            declaration["robot"] = (97, 10)
        elif invalid == "robot-body":
            declaration["robot"] = (0, 20)
        elif invalid == "station-world":
            declaration["station_keys"][left] = tuple(
                StationKey(97, key.token) for key in declaration["station_keys"][left]
            )
        elif invalid == "payload-world":
            info["primary_key"] = replace(info["primary_key"], world=97)
        elif invalid == "duplicate-key":
            declaration["station_keys"][right] = declaration["station_keys"][left]
        elif invalid == "missing-chart":
            del declaration["chart_ids"][left]
        elif invalid == "unknown-feature":
            declaration["counter_features"][left] = (0, 1, 2, 999)
        elif invalid == "whole-box":
            declaration["counter_features"][left] = tuple(range(8))
        else:
            declaration["pads"] = (
                PadFace.named("left", "not-authored", 0, 1, OFFSETS),
                declaration["pads"][1],
            )
        before = _snapshot(scene)
        with pytest.raises((ValueError, TypeError)):
            _registry(scene, info, declaration)
        assert _snapshot(scene) == before
        assert scene.tangent_history is None


def _assert_real_selection(scene, info, registry, selected, events, frames):
    assert selected.samples
    assert len(selected.samples) == len(selected.bindings) <= 8
    assert tuple(s.key for s in selected.samples) == tuple(sorted(s.key for s in selected.samples))
    pairs = set(registry.pairs)
    positive = sum(
        site.force_n > 0
        for record in events.active
        if (record.identity.first, record.identity.second) in pairs
        for component in record.components
        for site in component.load.sites
    )
    assert positive == len(selected.samples)
    asset = scene.assets[0, 10]
    tree = getattr(asset, "tree", asset)
    cols = {
        scene.shape_key(10, c.link_name, c.name): c
        for c in tree.collisions
        if (c.link_name, c.name) in _declarations(info["case"])["station_keys"]
    }
    declared = {row.key: row for row in registry.registrations}
    for sample, binding in zip(selected.samples, selected.bindings, strict=True):
        row = declared[sample.key]
        assert sample.key == binding.key
        assert sample.normal_force_n == binding.site.force_n > 0
        np.testing.assert_array_equal(sample.normal, binding.component.load.normal)
        assert binding.component.law.friction == info["friction"]
        for side in ("first", "second"):
            material = getattr(sample, side)
            owner = getattr(binding, side + "_owner")
            assert material.owner == owner
            assert owner.world == 0
            point = getattr(binding.site.material, side + "_point")
            pose = frames[owner]
            np.testing.assert_allclose(
                np.asarray(pose.position_m)
                + np.asarray(pose.rotation_world) @ material.local_point_m,
                point,
                atol=1e-10,
                rtol=0,
            )
            assert material.weights == getattr(binding.site.material, side + "_weights")
            assert set(node for node, _weight in material.weights).issubset(material.feature)
            assert material.feature == getattr(row, side + "_feature")
            if material.shape in cols:
                col = cols[material.shape]
                assert (owner.kind, owner.local_frame, owner.body) == (
                    "robot_link",
                    "robot_model_link",
                    10,
                )
                assert owner.link_index == col.link_index
                assert owner.link_name == tree.link_names[col.link_index]
                np.testing.assert_allclose(
                    material.local_point_m, row.anchor_local_m, atol=1e-8, rtol=0
                )
            else:
                assert material.shape == info["primary_key"]
                assert owner.kind == "rigid" and owner.body == 20
        np.testing.assert_allclose(
            binding.application_point_first_m,
            binding.application_point_second_m,
            atol=1e-14,
            rtol=0,
        )


def _bilateral_readout(selection, frames):
    rows = []
    for sample, binding in zip(selection.samples, selection.bindings, strict=True):
        row = dict(key=sample.key, normal=sample.normal, normal_force_n=sample.normal_force_n)
        for side in ("first", "second"):
            owner = getattr(binding, side + "_owner")
            pose = frames[owner]
            application = np.asarray(getattr(binding, "application_point_" + side + "_m"))
            row[side] = dict(
                owner=owner,
                material=getattr(sample, side),
                application_world_m=application,
                application_local_m=(application - pose.position_m)
                @ np.asarray(pose.rotation_world),
            )
        rows.append(row)
    return tuple(rows)


def _assert_first_load_balance(scene, info, before, after, raw_loads, registry):
    # The old law still supplies every N, gap and site; only selected traction changes.
    assert len(raw_loads.contacts) == len(after.loads.contacts)
    for raw, revised in zip(raw_loads.contacts, after.loads.contacts, strict=True):
        assert (raw.first, raw.second) == (revised.first, revised.second)
        if (raw.first, raw.second) not in registry.pairs:
            assert raw == revised
            continue
        assert raw.force_n == revised.force_n
        assert raw.gap_m == revised.gap_m
        assert raw.normal == revised.normal
        assert len(raw.sites) == len(revised.sites)
        for a, b in zip(raw.sites, revised.sites, strict=True):
            assert a.force_n == b.force_n
            assert a.fraction == b.fraction
            assert a.material == b.material
            assert a.application_point_first_m == b.application_point_first_m
            assert a.application_point_second_m == b.application_point_second_m

    for body, mass in zip((20, 30), info["masses"], strict=True):
        row = scene.rigid_rows[0, body]
        owned = scene.body_keys[0, body]
        force, torque = np.zeros(3), np.zeros(3)
        for load in after.loads.contacts:
            for side, sign in (("first", -1.0), ("second", 1.0)):
                if getattr(load, side) in owned:
                    for site in load.sites:
                        applied = sign * np.asarray(site.force_on_second_n)
                        point = getattr(site, "application_point_" + side + "_m")
                        force += applied
                        torque += np.cross(np.asarray(point) - before.rigid.position[row], applied)
        np.testing.assert_allclose(force, after.loads.inputs.rigid.force[row], atol=1e-9, rtol=0)
        np.testing.assert_allclose(torque, after.loads.inputs.rigid.torque[row], atol=1e-10, rtol=0)
        impulse = mass * (after.rigid.velocity[row] - before.rigid.velocity[row])
        np.testing.assert_allclose(
            impulse, info["dt"] * (force + mass * np.array((0, 0, -9.81))), atol=3e-12, rtol=0
        )
    asset = scene.assets[0, 10]
    tree = getattr(asset, "tree", asset)
    robot = after.robots[0, 10]
    linear, angular = point_jacobians(
        tree.model, before.robots[0, 10].full_q, np.arange(tree.model.dof_count), tree.model.com
    )
    wrench = robot.link_wrenches_world
    full = np.einsum("nij,ni->j", linear, wrench[:, :3]) + np.einsum(
        "nij,ni->j", angular, wrench[:, 3:]
    )
    np.testing.assert_allclose(robot.contact_effort_full, full, atol=1e-9, rtol=0)
    projection = asset.matrix if hasattr(asset, "matrix") else np.eye(len(robot.q))
    np.testing.assert_allclose(robot.contact_effort, projection.T @ full, atol=1e-9, rtol=0)
    assert np.max(abs(robot.drive_effort) - info["cap"]) <= 1e-9


@pytest.mark.parametrize("case", ["mimic-box", "panda-sphere"])
def test_first_loaded_pad_sites_and_retry_publish_once(case, tmp_path, monkeypatch):
    # Only the original acquire prefix through its first positive-normal step.
    # No lift/hold/release loop, state overwrite, changed command or warm-start pose.
    scene, controller, info = _acquisition().make_case(case)
    archive = dict(case=case, inputs=info, scope="first positive pad load and same-command retry")

    class InjectedBirthFailure(MemoryError):
        pass

    with scene:
        declaration = _declarations(case)
        registry = _registry(scene, info, declaration)
        archive["declaration"] = declaration
        archive["registry"] = registry
        scene.enable_coupled_tangent(registry, stiffness_n_per_m=200)
        pending = {}
        original_bind = MaterialStationRegistry.bind
        original_finish = _multi_robot_tangent.finish_tangent_step

        def capture_binding(self, events, frames, *, sample_time_s):
            if self is registry and any(
                site.force_n > 0
                for record in events.active
                if (record.identity.first, record.identity.second) in registry.pairs
                for component in record.components
                for site in component.load.sites
            ):
                archive["positive_binding_input"] = dict(events=events, owner_frames=frames)
            selection = original_bind(self, events, frames, sample_time_s=sample_time_s)
            if self is registry:
                pending.clear()
                pending.update(selection=selection, events=events, frames=frames)
            return selection

        def reject_first_positive(*args, **kwargs):
            result = original_finish(*args, **kwargs)
            if pending["selection"].samples:
                archive["unpublished_candidate"] = result
                raise InjectedBirthFailure("first real material birth publication")
            return result

        try:
            with monkeypatch.context() as patch:
                patch.setattr(MaterialStationRegistry, "bind", capture_binding)
                patch.setattr(_multi_robot_tangent, "finish_tangent_step", reject_first_positive)
                for _ in range(ceil(info["controller_inputs"]["timeout_s"] / info["dt"]) + 1):
                    before = scene.observe()
                    geometry = scene.geometry()
                    saved = _snapshot(scene)
                    history, receipt, events = (
                        scene.tangent_history,
                        scene.tangent_receipt,
                        scene.contact_events,
                    )
                    proposal = controller.propose(
                        before, dt_s=info["dt"], posture_target=info["q"], intent="acquire"
                    )
                    archive["last_entry"] = dict(
                        reading=before,
                        geometry=geometry,
                        history=history,
                        receipt=receipt,
                        events=events,
                        proposal=proposal,
                    )
                    assert proposal.phase not in {"lost", "empty", "timeout"}
                    try:
                        scene.step(info["dt"], actions={10: proposal.command})
                    except InjectedBirthFailure:
                        archive["first_positive"] = dict(pending)
                        break
                    assert not scene.tangent_history.active
                else:
                    pytest.fail("original acquisition prefix produced no positive target pad load")

                assert scene.tangent_history is history
                assert scene.tangent_receipt is receipt
                assert scene.contact_events is events
                assert _snapshot(scene) == saved
                assert not history.active
                again = controller.propose(
                    scene.observe(), dt_s=info["dt"], posture_target=info["q"], intent="acquire"
                )
                assert _public(again) == _public(proposal)
                selected = pending["selection"]
                _assert_real_selection(
                    scene, info, registry, selected, pending["events"], pending["frames"]
                )
                archive["bilateral_sites"] = _bilateral_readout(selected, pending["frames"])

                # Keep original physical input; a different declared face cannot inherit it.
                if case == "mimic-box":
                    bad = _declarations(case)
                    left, right = tuple(bad["counter_features"])
                    bad["counter_features"][left], bad["counter_features"][right] = (
                        bad["counter_features"][right],
                        bad["counter_features"][left],
                    )
                    wrong = _registry(scene, info, bad)
                    with pytest.raises(ValueError, match="declared.*feature"):
                        original_bind(
                            wrong, pending["events"], pending["frames"], sample_time_s=before.time_s
                        )
                binding = selected.bindings[0]
                owner = (
                    binding.first_owner
                    if binding.first_owner.kind == "robot_link"
                    else binding.second_owner
                )
                bad_frames = dict(pending["frames"])
                bad_frames[owner] = replace(
                    bad_frames[owner],
                    owner=replace(owner, link_index=0, link_name="wrong-material-owner"),
                )
                with pytest.raises(ValueError, match="owner"):
                    original_bind(
                        registry, pending["events"], bad_frames, sample_time_s=before.time_s
                    )
                assert _snapshot(scene) == saved

                raw_loads = scene.sample()
                archive["raw_entry_loads"] = raw_loads
                patch.setattr(_multi_robot_tangent, "finish_tangent_step", original_finish)
                after = scene.step(info["dt"], actions={10: again.command})
                archive["accepted"] = dict(
                    reading=after,
                    geometry=scene.geometry(),
                    history=scene.tangent_history,
                    receipt=scene.tangent_receipt,
                    events=scene.contact_events,
                    selection=pending["selection"],
                    owner_frames=pending["frames"],
                )
                assert _public(pending["selection"]) == _public(selected)
                assert _public((scene.tangent_history, scene.tangent_receipt)) == _public(
                    archive["unpublished_candidate"]
                )
                assert after.step_number == before.step_number + 1
                assert after.load_time_s == before.time_s
                assert after.time_s == pytest.approx(before.time_s + info["dt"], abs=1e-14)
                assert scene.tangent_history.step_index == history.step_index + 1
                assert {r.identity.key for r in scene.tangent_history.active} == {
                    s.key for s in selected.samples
                }
                assert all(r.identity.generation == 1 for r in scene.tangent_history.active)
                assert scene.tangent_receipt.accepted
                assert scene.tangent_receipt.material_motion_error_m <= 1e-6
                assert scene.tangent_receipt.application_motion_error_m <= 1e-6
                assert scene.tangent_receipt.material_application_error_m <= 1e-6
                assert abs(scene.tangent_receipt.energy_residual_j) <= 1e-8
                _assert_first_load_balance(scene, info, before, after, raw_loads, registry)
                committed = _snapshot(scene)
                scene.observe_scene()
                scene.sample()
                assert _snapshot(scene) == committed
        except Exception as error:
            archive["failure"] = dict(type=type(error).__name__, message=str(error))
            raise
        finally:
            archive["final_public"] = _snapshot(scene)
            with gzip.open(
                tmp_path / (case + "-first-load.json.gz"), "xt", encoding="utf-8"
            ) as stream:
                json.dump(
                    _public(archive),
                    stream,
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                )
