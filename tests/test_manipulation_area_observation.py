"""Explicit area roles on a real FREE-object protocol Scene, not a trained task.

The unchanged numerical fixture uses three undamped unlimited P joints and
native motors. Supplied pressure triangles exercise shared TYX dynamics and
the reader; they are not an automatic collision or grasp-success oracle.
"""

import json
from contextlib import ExitStack
from dataclasses import replace
from types import MappingProxyType

import numpy as np
import pytest
from test_manipulation_observation import loads as native_loads
from test_mujoco_tyx_area_pressure_lifecycle import _same

from phymeo.backends.mujoco_heterogeneous import (
    MuJoCoTyxHeterogeneousBatch,
    MuJoCoTyxModelSpec,
    MuJoCoTyxWorldSpec,
)
from phymeo.backends.mujoco_runtime import MuJoCoControl
from phymeo.builders.mujoco_compiler import compile_mujoco_xml
from phymeo.geometry.area_pressure import (
    AreaPressureIdentity,
    AreaPressureStamp,
    AreaPressureSurface,
    AreaPressureTriangle,
    quadrature_area_pressure,
)
from phymeo.training.manipulation_observation import (
    AreaPressureRoleBinding,
    ManipulationSuccessReader,
)
from phymeo.training.manipulation_tasks import Goal, Primitive, Scene, Target

MODES = ("interpret", pytest.param("cuda", marks=pytest.mark.cuda))
H = 0.0005
AREA = 1e-4
GRADIENT = 2e6
DEPTH = 0.001
KEYS = (41, 7, 305, 11)
MODEL_KEYS = ("area-a", "native", "area-b", "area-split")
PAIRS_A = (
    ("object", "left-finger-body"),
    ("right-finger-body", "object"),
    ("support-body", "object"),
)
PAIRS_B = tuple(tuple(reversed(pair)) for pair in PAIRS_A)
ROLES = ("finger0", "finger1", "support")
OBJECT_NORMALS = ((0, -1, 0), (0, 1, 0), (0, 0, 1))
CENTERS = ((0, 0.049, 0.049), (0, -0.049, 0.049), (0, 0, 0))


def _scene():
    xml = f'''<mujoco model="area-success-protocol">
      <compiler angle="radian" fusestatic="false"/>
      <option timestep="{H}" gravity="0 0 -9.81" integrator="Euler"
        cone="pyramidal" solver="Newton" iterations="128" tolerance="1e-10"/>
      <default><joint damping="0" armature="0" limited="false"/>
        <geom condim="1" friction="0 0 0" solref=".02 1"/>
      </default>
      <worldbody>
        <body name="support-body"><geom name="support" type="plane" size="1 1 .05"/></body>
        <body name="arm" pos="0 0 .1">
          <joint name="arm-slide" type="slide" axis="1 0 0"/>
          <inertial mass=".1" pos="0 0 0" diaginertia=".01 .01 .01"/>
          <geom name="arm-geom" type="sphere" size=".01" pos=".30 0 0"/>
          <body name="left-finger-body" pos="0 .060 -.051">
            <joint name="left-finger" type="slide" axis="0 -1 0"/>
            <geom name="left-pad" type="box" size=".015 .011 .04" mass=".04"/>
          </body>
          <body name="right-finger-body" pos="0 -.060 -.051">
            <joint name="right-finger" type="slide" axis="0 1 0"/>
            <geom name="right-pad" type="box" size=".015 .011 .04" mass=".04"/>
          </body>
        </body>
        <body name="object" pos="0 0 .049">
          <freejoint name="object-free"/>
          <geom name="object-0" type="sphere" size=".05" mass=".08"/>
        </body>
        <site name="target-center" type="box" pos="0 0 .001" size=".06 .06 .001"/>
      </worldbody>
      <actuator>
        <motor name="motor-arm" joint="arm-slide" gear="1"/>
        <motor name="motor-left" joint="left-finger" gear="1"/>
        <motor name="motor-right" joint="right-finger" gear="1"/>
      </actuator>
    </mujoco>'''
    return Scene(
        key="area-success-protocol",
        robot_kinds="P",
        mjcf=xml,
        object_shapes=(Primitive("sphere", (0.05,)),),
        object_mass_kg=0.08,
        initial_object_position=(0, 0, 0.049),
        targets=(Target("center", "中心", "center", (0, 0, 0), 0, (0.06, 0.06)),),
        goals=(
            Goal(
                "grasp",
                "object",
                "center",
                ("抬起球体", "把球提起"),
                ("Lift the sphere", "Raise the ball"),
            ),
        ),
        default_goal=0,
        robot_joint_names=("arm-slide", "left-finger", "right-finger"),
        actuator_names=("motor-arm", "motor-left", "motor-right"),
        finger_geom_names=("left-pad", "right-pad"),
        timestep_s=H,
        object_label_zh="协议测试球体",
        object_label_en="protocol sphere",
    )


def _roles(pairs, material):
    return tuple(
        AreaPressureRoleBinding(
            AreaPressureIdentity(*pair, material + "-pressure", material + "-rigid"),
            "A" if pair[0] == "object" else "B",
            role,
        )
        for pair, role in zip(pairs, ROLES, strict=True)
    )


def _make(stack, mode):
    scene = _scene()
    compilation = compile_mujoco_xml(scene.mjcf, require_asset=False)
    roles_a, roles_b = _roles(PAIRS_A, "protocol"), _roles(PAIRS_B, "alternative")
    bindings = {"area-a": roles_a, "area-b": roles_b, "area-split": roles_a}
    specs = tuple(
        MuJoCoTyxModelSpec(
            key,
            compilation,
            contact_capacity=16,
            constraint_capacity=64,
            area_pressure_pairs=() if key == "native" else PAIRS_B if key == "area-b" else PAIRS_A,
        )
        for key in MODEL_KEYS
    )
    worlds = tuple(
        MuJoCoTyxWorldSpec(key, model) for key, model in zip(KEYS, MODEL_KEYS, strict=True)
    )
    batch = stack.enter_context(MuJoCoTyxHeterogeneousBatch(specs, worlds, mode=mode))
    workspace = stack.enter_context(batch.prepare_device_workspace(include_area_pressure=True))
    scenes = dict.fromkeys(KEYS, scene)
    reader = ManipulationSuccessReader(workspace, scenes, area_roles=bindings)
    return batch, workspace, reader, scenes, bindings, compilation, worlds


def _surface(batch, key, binding, pair_index, *, split):
    state = batch.integration_snapshot()[key]
    stamp = AreaPressureStamp(
        str(key), state.tick, state.time, batch.area_pressure_configuration_id(key)
    )
    normal = np.array(OBJECT_NORMALS[pair_index], dtype=np.float64)
    u = np.array((1.0, 0.0, 0.0))
    v = np.cross(normal, u)
    # Base=2a and height=a give actual area a^2=1e-4. The centroid
    # is CENTER; these are physical coordinates, not normalized weights.
    a = 0.01
    offsets = ((-a, -a / 3), (a, -a / 3), (0, 2 * a / 3))
    vertices = np.array([np.array(CENTERS[pair_index]) + x * u + y * v for x, y in offsets])
    if binding.object_side == "B":
        # Swap ownership and winding while retaining the identical physical
        # patch and points; do not mirror the patch in its tangent plane.
        normal = -normal
        vertices = vertices[[0, 2, 1]]
    pieces = (vertices,)
    if split:
        p, q, r = vertices
        pq, qr, rp = (p + q) / 2, (q + r) / 2, (r + p) / 2
        pieces = (
            np.array((p, pq, rp)),
            np.array((pq, q, qr)),
            np.array((rp, qr, r)),
            np.array((pq, qr, rp)),
        )
    triangles = tuple(
        AreaPressureTriangle(
            f"pair-{pair_index}-triangle-{index}",
            f"cell-{pair_index}",
            f"face-{pair_index}",
            piece,
            np.full(3, GRADIENT * DEPTH),
            GRADIENT * normal,
        )
        for index, piece in enumerate(pieces)
    )
    result = AreaPressureSurface(binding.identity, stamp, triangles)
    quad = quadrature_area_pressure(result)
    assert quad.total_area_m2 == pytest.approx(AREA, rel=1e-12, abs=1e-15)
    np.testing.assert_allclose(quad.dA, AREA / (12 if split else 3), rtol=1e-12, atol=1e-15)
    return result


def _step(batch, workspace, roles):
    mapping = {
        key: tuple(
            _surface(batch, key, binding, pi, split=batch.layout[key].model_key == "area-split")
            for pi, binding in enumerate(roles[batch.layout[key].model_key])
        )
        for key in KEYS
        if batch.layout[key].model_key in roles
    }
    assert batch.step_area_pressure(
        H, mapping, expected_revision=batch.checkpoint_revision
    ) == dict.fromkeys(KEYS, True)
    frame = workspace.refresh(expected_revision=batch.checkpoint_revision)
    return frame, workspace.readback(frame)


def _area_loads(receipt, bindings):
    fingers, support = [0.0, 0.0], 0.0
    for pair, binding in zip(receipt.raw_contact_fields["area_pressure"], bindings, strict=True):
        assert pair["identity"] == binding.identity
        force = float(np.sum(pair["normal_force"], dtype=np.float64))
        if binding.role in ("finger0", "finger1"):
            fingers[int(binding.role[-1])] += force
        if binding.role == "support":
            sign = 1 if binding.object_side == "A" else -1
            support += sign * float(np.sum(pair["force_on_a"][:, 2], dtype=np.float64))
    return np.array(fingers), max(0.0, support)


def _initial(reader, batch, workspace):
    frame = workspace.refresh(expected_revision=batch.checkpoint_revision)
    assert not reader.read(frame, episodes=dict.fromkeys(KEYS, 0))
    assert all(len(record) == 6 and record[2] == -1 for record in reader.checkpoint().records)


def _copied_arrays(arrays):
    return {name: value.copy() for name, value in arrays.items()}


def _catalog_frame(frame, world, pair, **changes):
    catalogs = dict(frame.area_catalogs)
    items = list(catalogs[world])
    item = dict(items[pair])
    item.update(changes)
    items[pair] = MappingProxyType(item)
    catalogs[world] = tuple(items)
    return replace(frame, area_catalogs=MappingProxyType(catalogs))


def _reject_read(monkeypatch, ws, reader, frame, arrays, cursor, *, match=None):
    # Corrupt a consumer-owned public export, not accepted physical storage.
    # Bypass only the workspace token guard to isolate reader validation of
    # the supplied frame/catalog; all legitimate payloads came from TYX above.
    def readback(selected):
        assert selected is frame
        return MappingProxyType(arrays)

    with monkeypatch.context() as patch:
        patch.setattr(ws, "readback", readback)
        with pytest.raises(ValueError, match=match):
            reader.read(frame, episodes=dict.fromkeys(KEYS, 0))
    assert reader.checkpoint() == cursor


@pytest.mark.parametrize("mode", MODES)
def test_area_sides_split_measures_and_native_contacts_use_actual_forces_in_newtons(mode, tmp_path):
    with ExitStack() as stack:
        batch, ws, reader, scenes, roles, compilation, _ = _make(stack, mode)
        _initial(reader, batch, ws)
        frame, arrays = _step(batch, ws, roles)
        observed = reader.read(frame, episodes=dict.fromkeys(KEYS, 0))
        assert set(observed) == set(KEYS)
        native_model = compilation.fresh_native_model()
        expected = {}
        for key in KEYS:
            model_key = batch.layout[key].model_key
            receipt = batch.last_step.receipts[key]
            loads = (
                _area_loads(receipt, roles[model_key])
                if model_key in roles
                else native_loads(scenes[key], native_model, receipt)
            )
            expected[key] = loads
            np.testing.assert_allclose(
                observed[key].finger_normal_forces_n, loads[0], rtol=3e-3, atol=3e-3
            )
            assert observed[key].support_up_force_n == pytest.approx(loads[1], rel=3e-3, abs=3e-3)
            assert observed[key].tick == 0 and observed[key].phase == "accepted-entry"
            assert observed[key].time_s == receipt.input_time
        for key in (41, 305, 11):
            assert min(expected[key][0]) > 0.01 and expected[key][1] > 0.01
        assert expected[7][1] > 0
        for key in (305, 11):
            np.testing.assert_allclose(expected[key][0], expected[41][0], rtol=3e-3, atol=3e-3)
            assert expected[key][1] == pytest.approx(expected[41][1], rel=3e-3, abs=3e-3)
        for pi in range(3):
            coarse = batch.last_step.receipts[41].raw_contact_fields["area_pressure"][pi]
            refined = batch.last_step.receipts[11].raw_contact_fields["area_pressure"][pi]
            assert len(coarse["normal_force"]) == 3 and len(refined["normal_force"]) == 12
            np.testing.assert_allclose(refined["dA"], coarse["dA"][0] / 4, rtol=1e-12, atol=1e-15)
            assert float(np.sum(coarse["dA"])) == pytest.approx(AREA, rel=1e-12, abs=1e-15)
            correct = float(np.sum(refined["normal_force"]))
            incorrectly_area_weighted = float(np.sum(refined["dA"] * refined["normal_force"]))
            assert np.isfinite(correct) and correct > 0.01
            assert incorrectly_area_weighted < 0.001 * correct
        summary = {
            "mode": mode,
            "scope": "accepted-entry-load-reader-not-task-success",
            "h": H,
            "area_m2_per_pair": AREA,
            "worlds": [
                {
                    "world": key,
                    "model": batch.layout[key].model_key,
                    "tick": observed[key].tick,
                    "time_s": observed[key].time_s,
                    "finger_loads_n": [float(v) for v in observed[key].finger_normal_forces_n],
                    "support_up_n": float(observed[key].support_up_force_n),
                    "oracle_finger_n": expected[key][0].tolist(),
                    "oracle_support_n": float(expected[key][1]),
                }
                for key in KEYS
            ],
        }
        (tmp_path / "accepted-loads.json").write_text(
            json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
        )
        assert arrays["area_declared_mask"][:, 0].tolist() == [1, 0, 1, 1]
        assert frame.area_catalogs[7] == ()
        cursor = reader.checkpoint()
        assert not reader.read(frame, episodes=dict.fromkeys(KEYS, 0))
        assert reader.checkpoint() == cursor


@pytest.mark.parametrize("mode", MODES)
def test_area_configuration_and_late_catalog_rejections_are_cursor_atomic(mode, monkeypatch):
    with ExitStack() as stack:
        batch, ws, reader, scenes, roles, _, _ = _make(stack, mode)
        original = roles["area-a"][0]
        with pytest.raises(ValueError, match="explicit body-pair"):
            ManipulationSuccessReader(ws, scenes)
        plain = stack.enter_context(batch.prepare_device_workspace(include_contacts=True))
        with pytest.raises(ValueError, match="area observation workspace"):
            ManipulationSuccessReader(plain, scenes, area_roles=roles)
        for invalid in ({"area-a": roles["area-a"]}, {**roles, "native": ()}):
            with pytest.raises(ValueError, match="exactly cover"):
                ManipulationSuccessReader(ws, scenes, area_roles=invalid)
        for replacement in (
            replace(original, object_side="B"),
            replace(original, role="finger1"),
            replace(original, role="other"),
        ):
            changed = dict(roles)
            changed["area-a"] = (replacement, *roles["area-a"][1:])
            with pytest.raises(ValueError):
                ManipulationSuccessReader(ws, scenes, area_roles=changed)
        ambiguous = {
            key: replace(scene, support_geom_name="left-pad") for key, scene in scenes.items()
        }
        with pytest.raises(ValueError, match="unambiguous|ambiguous"):
            ManipulationSuccessReader(ws, ambiguous, area_roles=roles)
        for side, role in ((np.array("A"), "finger0"), ("A", np.array("support"))):
            with pytest.raises(ValueError):
                AreaPressureRoleBinding(original.identity, side, role)
        supplied_identity = replace(original.identity)
        owned = AreaPressureRoleBinding(supplied_identity, "A", "finger0")
        object.__setattr__(supplied_identity, "compliant_material", "mutated-caller-material")
        assert owned.identity == original.identity
        assert owned.identity is not supplied_identity
        caller_roles = dict(roles)
        owned_reader = ManipulationSuccessReader(ws, scenes, area_roles=caller_roles)
        caller_roles.clear()
        _initial(owned_reader, batch, ws)
        _initial(reader, batch, ws)
        cursor = reader.checkpoint()
        frame, arrays = _step(batch, ws, roles)
        # All four models deliberately reuse native numeric IDs. Material and
        # model declarations must still bind each directory independently.
        meta_a, meta_b = ws.model_metadata["area-a"], ws.model_metadata["area-b"]
        np.testing.assert_array_equal(meta_a["geom_bodyid"], meta_b["geom_bodyid"])
        assert meta_a["body_names"] == meta_b["body_names"]
        row = KEYS.index(11)  # Last world: a rejection cannot advance earlier cursors.
        arm = tuple(meta_a["body_names"]).index("arm")
        for field, index, value in (
            ("area_meta", (row, 0), 0),
            ("area_meta", (row, 1), int(arrays["area_meta"][row, 1]) - 1),
            ("area_ids", (row, 0, 1), 1),
            ("area_ids", (row, 0, 2), 1),
            ("area_ids", (row, 0, 3), arm),
            ("area_declared_mask", (row, 0), 0),
            ("area_measure_words", (row, 0, slice(None)), 0),
            ("area_data", (row, 0, 8), np.nan),
            ("entry_meta", (row, 1), 1),
            ("entry_clock", (row, 0), 4 * H),
        ):
            changed = _copied_arrays(arrays)
            changed[field][index] = value
            _reject_read(monkeypatch, ws, reader, frame, changed, cursor)
        item = frame.area_catalogs[11][0]
        different_material = replace(
            item["identity"], compliant_material=roles["area-b"][0].identity.compliant_material
        )
        for changes in (
            {"identity": different_material},
            {"stamp": replace(item["stamp"], step_index=item["stamp"].step_index + 1)},
            {"stamp": replace(item["stamp"], sample_time_s=item["stamp"].sample_time_s + 1e-6)},
            {"sample_ids": item["sample_ids"][:-3]},
            {"source_ids": item["source_ids"][:-1]},
        ):
            changed_frame = _catalog_frame(frame, 11, 0, **changes)
            _reject_read(monkeypatch, ws, reader, changed_frame, _copied_arrays(arrays), cursor)
        observed = reader.read(frame, episodes=dict.fromkeys(KEYS, 0))
        assert set(observed) == set(KEYS)
        _same(owned_reader.read(frame, episodes=dict.fromkeys(KEYS, 0)), observed)


@pytest.mark.parametrize("mode", MODES)
def test_controls_reset_and_paired_restore_preserve_area_history_and_cursor(mode):
    with ExitStack() as stack:
        batch, ws, reader, _, roles, _, _ = _make(stack, mode)
        episodes = dict.fromkeys(KEYS, 0)
        _initial(reader, batch, ws)
        first, _ = _step(batch, ws, roles)
        reader.read(first, episodes=episodes)
        cursor, physics = reader.checkpoint(), batch.checkpoint()
        historical_catalogs = first.area_catalogs
        controls = {key: MuJoCoControl(np.full(3, 0.01)) for key in KEYS}
        batch.set_controls(controls)
        controlled = ws.refresh(expected_revision=batch.checkpoint_revision)
        assert controlled.area_catalogs == historical_catalogs
        assert not reader.read(controlled, episodes=episodes)
        assert reader.checkpoint() == cursor
        second, _ = _step(batch, ws, roles)
        expected = reader.read(second, episodes=episodes)
        expected_state = batch.integration_snapshot()
        batch.restore(physics, expected_revision=batch.checkpoint_revision)
        restored = ws.refresh(expected_revision=batch.checkpoint_revision)
        assert restored.area_catalogs == historical_catalogs
        with pytest.raises(ValueError):
            reader.read(restored, episodes=episodes)
        reader.restore(cursor, frame=restored)
        assert not reader.read(restored, episodes=episodes)
        batch.set_controls(controls)
        replay, _ = _step(batch, ws, roles)
        _same(reader.read(replay, episodes=episodes), expected)
        _same(batch.integration_snapshot(), expected_state)
        before_reset = reader.checkpoint()
        batch.reset((41,))
        reset = ws.refresh(expected_revision=batch.checkpoint_revision)
        with pytest.raises(ValueError):
            reader.read(reset, episodes=episodes)
        assert reader.checkpoint() == before_reset
        episodes[41] = 1
        assert not reader.read(reset, episodes=episodes)
        records = {row[0]: row for row in reader.checkpoint().records}
        assert records[41][1:3] == (1, -1)
        for key in KEYS[1:]:
            assert records[key] == next(row for row in before_reset.records if row[0] == key)


@pytest.mark.parametrize("mode", MODES)
def test_same_tick_area_changes_and_native_double_counting_are_rejected(mode, monkeypatch):
    with ExitStack() as stack:
        batch, ws, reader, _, roles, _, _ = _make(stack, mode)
        _initial(reader, batch, ws)
        frame, arrays = _step(batch, ws, roles)
        assert set(reader.read(frame, episodes=dict.fromkeys(KEYS, 0))) == set(KEYS)
        cursor = reader.checkpoint()
        row = KEYS.index(11)
        changed = _copied_arrays(arrays)
        # Coherent force/moment changes still cannot rewrite accepted history.
        changed["area_data"][row, 0, 8:15] *= 2
        _reject_read(monkeypatch, ws, reader, frame, changed, cursor, match="same tick")
        changed = _copied_arrays(arrays)
        original_area = np.frombuffer(
            arrays["area_measure_words"][row, 0].astype("<i4").tobytes(), dtype="<f8"
        )[0]
        changed["area_measure_words"][row, 0] = np.frombuffer(
            np.array([2 * original_area], dtype="<f8").tobytes(), dtype="<i4"
        )
        _reject_read(monkeypatch, ws, reader, frame, changed, cursor, match="same tick")
        item = frame.area_catalogs[11][0]
        renamed_samples = (
            tuple(("renamed-triangle", slot) for slot in range(3)) + item["sample_ids"][3:]
        )
        renamed_sources = (("renamed-cell", "renamed-face"),) * 3 + item["source_ids"][3:]
        for changes in (
            {"sample_ids": renamed_samples},
            {"source_ids": renamed_sources},
            {"stamp": replace(item["stamp"], configuration_id="changed-historical-configuration")},
        ):
            changed_frame = _catalog_frame(frame, 11, 0, **changes)
            _reject_read(
                monkeypatch,
                ws,
                reader,
                changed_frame,
                _copied_arrays(arrays),
                cursor,
                match="same tick",
            )
        # Copy a genuine native support contact from the same initial scene,
        # including its original wrenches; this is a policy-violation export,
        # not invented contact-force evidence or a mutation of the solver.
        native_row = KEYS.index(7)
        geometry = tuple(ws.model_metadata["native"]["geom_names"])
        native_pair = {geometry.index("object-0"), geometry.index("support")}
        count = int(arrays["entry_meta"][native_row, 3])
        matches = [
            index
            for index, contact in enumerate(arrays["entry_contact_ids"][native_row, :count])
            if set(map(int, contact[2:4])) == native_pair
        ]
        assert matches and arrays["entry_meta"][row, 3] == 0
        index = matches[0]
        changed = _copied_arrays(arrays)
        for name in changed:
            if name.startswith("entry_contact_"):
                changed[name][row, 0] = arrays[name][native_row, index]
        changed["entry_meta"][row, 3] = 1
        changed["entry_meta"][row, 4] = int(changed["entry_contact_ids"][row, 0, 1])
        _reject_read(
            monkeypatch,
            ws,
            reader,
            frame,
            changed,
            cursor,
            match="duplicates a declared area body pair",
        )
        assert not reader.read(frame, episodes=dict.fromkeys(KEYS, 0))
        assert reader.checkpoint() == cursor
