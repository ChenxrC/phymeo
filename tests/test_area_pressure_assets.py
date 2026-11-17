"""Portable full pressure assets: HOST protocol and one short CUDA comparison.

Capsule mutations below use the frozen public JSON/array representation, not
the new implementation's decoder. No malformed capsule is a physics oracle.
"""

import base64
import json
from collections.abc import Mapping
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256

import numpy as np
import pytest
from test_mujoco_tyx_area_pressure_lifecycle import _same

from phymeo.backends.area_pressure_assets import AreaPressureAssetBundle
from phymeo.backends.mujoco_area_pressure_scene import BodyAreaPressureAsset
from phymeo.geometry import area_pressure_intersection as geometry

MAGIC = b"PHYMEOBODYAREAASSETS1\n"
MAX_BYTES = 16 * 1024 * 1024


@pytest.fixture(scope="module")
def declared():
    # CPU compilation only; no test helper starts a runtime at import time.
    from test_manipulation_area_env import _declarations

    return _declarations()


def _decode(raw):
    assert raw.startswith(MAGIC)
    digest, payload = raw[len(MAGIC) :].split(b"\n", 1)
    assert digest == sha256(payload).hexdigest().encode("ascii")
    return json.loads(payload)


def _envelope(value):
    payload = json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")).encode(
        "utf-8"
    )
    return _raw_envelope(payload)


def _raw_envelope(payload):
    return MAGIC + sha256(payload).hexdigest().encode("ascii") + b"\n" + payload


def _array(value, dtype):
    array = np.asarray(value, dtype=dtype)
    return {
        "$array": dtype,
        "shape": list(array.shape),
        "data": base64.b64encode(array.tobytes(order="C")).decode("ascii"),
    }


def _assets(declared):
    return {
        key: tuple(BodyAreaPressureAsset(a.identity, a.pressure_local, a.rigid_local) for a in rows)
        for key, rows in declared["assets"].items()
    }


def _bundle(declared):
    return AreaPressureAssetBundle.from_assets(declared["specs"], declared["assets"])


def _mesh_arrays(asset):
    return (
        asset.pressure_local.world_vertices,
        asset.pressure_local.cells,
        asset.pressure_local.nodal_pressure,
        asset.rigid_local.world_vertices,
        asset.rigid_local.faces,
    )


def _geometry_spy(monkeypatch):
    calls = []
    original_tet, original_rigid = geometry._tet_geometry, geometry._rigid_geometry

    def tetra(*args, **kwargs):
        calls.append("pressure")
        return original_tet(*args, **kwargs)

    def rigid(*args, **kwargs):
        calls.append("rigid")
        return original_rigid(*args, **kwargs)

    monkeypatch.setattr(geometry, "_tet_geometry", tetra)
    monkeypatch.setattr(geometry, "_rigid_geometry", rigid)
    return calls


def test_exact_signed_zero_roundtrip_owned_bytes_and_fresh_immutable_exports(declared):
    source_assets = _assets(declared)
    original = _bundle(declared)
    asset = source_assets["area-soft"][0]
    vertices, pressure = (
        asset.pressure_local.world_vertices.copy(),
        asset.pressure_local.nodal_pressure.copy(),
    )
    vertices[8, 0] = -0.0
    pressure[0] = -0.0
    signed = replace(asset.pressure_local, world_vertices=vertices, nodal_pressure=pressure)
    source_assets["area-soft"] = (BodyAreaPressureAsset(asset.identity, signed, asset.rigid_local),)
    bundle = AreaPressureAssetBundle.from_assets(declared["specs"], source_assets)
    raw = bundle.to_bytes()
    assert type(raw) is bytes and len(raw) < MAX_BYTES
    assert _decode(raw)["asset_sha256"] != _decode(original.to_bytes())["asset_sha256"]
    mutable = bytearray(raw)
    loaded = AreaPressureAssetBundle.from_bytes(memoryview(mutable))
    mutable[:] = b"x" * len(mutable)
    assert loaded.to_bytes() == raw
    first, second = loaded.assets_for(declared["specs"]), loaded.assets_for(declared["specs"])
    _same(first, source_assets)
    _same(second, source_assets)
    assert isinstance(first, Mapping) and set(first) == {"area-soft", "area-stiff"}
    with pytest.raises(TypeError):
        first["forbidden"] = ()
    for key in first:
        assert type(first[key]) is tuple
        assert first[key][0] is not second[key][0]
        for left, right in zip(
            _mesh_arrays(first[key][0]), _mesh_arrays(second[key][0]), strict=True
        ):
            assert left.tobytes() == right.tobytes() and not np.shares_memory(left, right)
            with pytest.raises(ValueError):
                left.setflags(write=True)
    assert np.signbit(first["area-soft"][0].pressure_local.world_vertices[8, 0])
    assert np.signbit(first["area-soft"][0].pressure_local.nodal_pressure[0])
    # Deliberate frozen-object bypass must affect only this export/caller copy.
    object.__setattr__(first["area-soft"][0].identity, "compliant_material", "tampered-export")
    object.__setattr__(source_assets["area-stiff"][0].pressure_local, "nodal_pressure", np.zeros(9))
    source_assets.clear()
    _same(loaded.assets_for(declared["specs"]), second)
    assert loaded.to_bytes() == raw and bundle.to_bytes() == raw


def test_source_bytes_model_binding_origin_independence_and_extra_native(declared):
    bundle = _bundle(declared)
    payload = _decode(bundle.to_bytes())
    specs = declared["specs"]
    assert set(payload) == {
        "schema",
        "coordinates",
        "units",
        "pressure_domain",
        "law",
        "friction",
        "models",
        "asset_sha256",
    }
    assert payload["schema"] == "phymeo-body-area-assets-v1"
    assert payload["coordinates"] == "mujoco-body-local"
    assert payload["units"] == {"length": "m", "pressure": "Pa"}
    assert payload["pressure_domain"] == "full-zero-boundary"
    assert (
        payload["law"] == "frozen-surface-elastic-backward-euler-v1"
        and payload["friction"] == "none"
    )
    assert set(payload["models"]) == {"area-soft", "area-stiff"}
    for spec in specs:
        if not spec.area_pressure_pairs:
            continue
        source = spec.compilation.source
        assert payload["models"][spec.key]["source"] == dict(
            primary_sha256=sha256(source.primary_xml).hexdigest(),
            archive_sha256=sha256(source.archive_bytes).hexdigest(),
            model_sha256=sha256(source.model_binary).hexdigest(),
            mujoco_version=source.mujoco_version,
        )
    moved = tuple(
        replace(
            spec,
            compilation=replace(
                spec.compilation,
                source=replace(spec.compilation.source, origin="Z:/unavailable/original.xml"),
            ),
        )
        for spec in specs
    )
    _same(bundle.assets_for(moved), declared["assets"])
    assert (
        AreaPressureAssetBundle.from_assets(moved, declared["assets"]).to_bytes()
        == bundle.to_bytes()
    )
    native = next(spec for spec in specs if not spec.area_pressure_pairs)
    area_only = tuple(spec for spec in specs if spec.area_pressure_pairs)
    _same(bundle.assets_for(area_only), declared["assets"])
    _same(bundle.assets_for((*specs, replace(native, key="another-native"))), declared["assets"])
    soft = area_only[0]
    source = soft.compilation.source
    for field in ("primary_xml", "archive_bytes", "model_binary"):
        damaged = replace(source, **{field: getattr(source, field) + b"\n"})
        bad = replace(soft, compilation=replace(soft.compilation, source=damaged))
        changed = tuple(bad if spec.key == soft.key else spec for spec in specs)
        with pytest.raises(ValueError):
            AreaPressureAssetBundle.from_assets(changed, declared["assets"])
        with pytest.raises(ValueError):
            bundle.assets_for(changed)
    # Even a self-consistent source digest/version change is a different binding.
    new_primary = source.primary_xml + b"\n"
    for changed_source in (
        replace(source, primary_xml=new_primary, primary_sha256=sha256(new_primary).hexdigest()),
        replace(source, mujoco_version="another-version"),
    ):
        bad = replace(soft, compilation=replace(soft.compilation, source=changed_source))
        with pytest.raises(ValueError):
            bundle.assets_for(tuple(bad if spec.key == soft.key else spec for spec in specs))
    for changed in (
        (soft,),
        (*specs, soft),
        tuple(
            replace(spec, key="renamed-area") if spec.key == soft.key else spec for spec in specs
        ),
    ):
        with pytest.raises(ValueError):
            bundle.assets_for(changed)


def test_pair_order_exact_coverage_and_legal_tetra_reordering(declared):
    soft = next(spec for spec in declared["specs"] if spec.key == "area-soft")
    first = declared["assets"]["area-soft"][0]
    second = BodyAreaPressureAsset(
        replace(first.identity, rigid_body="left-finger-body", rigid_material="second-rigid"),
        first.pressure_local,
        first.rigid_local,
    )
    spec = replace(
        soft, area_pressure_pairs=(("object", "support-body"), ("object", "left-finger-body"))
    )
    bundle = AreaPressureAssetBundle.from_assets((spec,), {spec.key: (first, second)})
    exported = bundle.assets_for((spec,))[spec.key]
    assert tuple(a.identity for a in exported) == (first.identity, second.identity)
    with pytest.raises(ValueError):
        AreaPressureAssetBundle.from_assets((spec,), {spec.key: (second, first)})
    with pytest.raises(ValueError):
        bundle.assets_for(
            (replace(spec, area_pressure_pairs=tuple(reversed(spec.area_pressure_pairs))),)
        )
    for supplied in ({}, {spec.key: (first,)}, {spec.key: (first, second), "native": ()}):
        with pytest.raises(ValueError):
            AreaPressureAssetBundle.from_assets((spec,), supplied)
    mesh = first.pressure_local
    reversed_mesh = replace(mesh, cells=mesh.cells[::-1], cell_ids=tuple(reversed(mesh.cell_ids)))
    reordered = BodyAreaPressureAsset(first.identity, reversed_mesh, first.rigid_local)
    changed = AreaPressureAssetBundle.from_assets((soft,), {soft.key: (reordered,)})
    (actual,) = AreaPressureAssetBundle.from_bytes(changed.to_bytes()).assets_for((soft,))[soft.key]
    _same(actual, reordered)
    assert actual.pressure_local.cells.tobytes() == mesh.cells[::-1].tobytes()


def test_rechecksummed_schema_source_identity_and_array_mutations_rejected(declared):
    bundle = _bundle(declared)
    original_bytes = bundle.to_bytes()
    payload = _decode(original_bytes)
    corruptions = []
    for name, value in (
        ("schema", "future-schema"),
        ("coordinates", "world"),
        ("units", {"length": "mm", "pressure": "Pa"}),
        ("pressure_domain", "local"),
        ("law", "different-law"),
        ("friction", "coulomb"),
        ("asset_sha256", "0" * 64),
    ):
        bad = deepcopy(payload)
        bad[name] = value
        corruptions.append(bad)
    bad = deepcopy(payload)
    bad["unknown"] = "not silently ignored"
    corruptions.append(bad)
    for name in ("primary_sha256", "archive_sha256", "model_sha256", "mujoco_version"):
        bad = deepcopy(payload)
        bad["models"]["area-soft"]["source"][name] = True
        corruptions.append(bad)
    for change in ("record", "body", "extra", "dtype", "shape-bool", "length", "nonfinite"):
        bad = deepcopy(payload)
        pair = bad["models"]["area-stiff"]["pairs"][0]
        if change == "record":
            pair["identity"] = {"$record": "AreaPressureIdentity", "fields": pair["identity"]}
        elif change == "body":
            pair["identity"]["rigid_body"] = pair["identity"]["compliant_body"]
        elif change == "extra":
            pair["pressure"]["trusted"] = True
        elif change == "dtype":
            pair["pressure"]["vertices_local"]["$array"] = "<f4"
        elif change == "shape-bool":
            pair["pressure"]["cells"]["shape"][1] = True
        elif change == "length":
            pair["rigid"]["vertices_local"]["data"] = ""
        else:
            values = declared["assets"]["area-stiff"][0].pressure_local.nodal_pressure.copy()
            values[0] = np.nan
            pair["pressure"]["nodal_pressure"] = _array(values, "<f8")
        corruptions.append(bad)
    for bad in corruptions:
        with pytest.raises(ValueError):
            AreaPressureAssetBundle.from_bytes(_envelope(bad))
    assert bundle.to_bytes() == original_bytes


def test_whole_bundle_counts_array_extents_and_byte_limit_precede_geometry(declared, monkeypatch):
    payload = _decode(_bundle(declared).to_bytes())
    corruptions = []
    bad = deepcopy(payload)
    template = bad["models"]["area-stiff"]
    bad["models"] = {f"model-{i}": deepcopy(template) for i in range(33)}
    corruptions.append(bad)
    bad = deepcopy(payload)
    bad["models"]["area-stiff"]["pairs"] *= 17
    corruptions.append(bad)
    for kind in ("nodes", "cells", "faces", "last-dtype"):
        bad = deepcopy(payload)
        pair = bad["models"]["area-stiff"]["pairs"][0]
        if kind == "nodes":
            pair["pressure"]["node_ids"] = [f"node-{i}" for i in range(129)]
            pair["pressure"]["vertices_local"] = _array(np.zeros((129, 3)), "<f8")
            pair["pressure"]["nodal_pressure"] = _array(np.zeros(129), "<f8")
        elif kind == "cells":
            pair["pressure"]["cell_ids"] = [f"cell-{i}" for i in range(33)]
            pair["pressure"]["cells"] = _array(np.tile((0, 1, 2, 8), (33, 1)), "<i8")
        elif kind == "faces":
            pair["rigid"]["face_ids"] = [f"face-{i}" for i in range(65)]
            pair["rigid"]["faces"] = _array(np.tile((0, 1, 2), (65, 1)), "<i8")
        else:
            pair["rigid"]["faces"]["$array"] = "<f8"
        corruptions.append(bad)
    calls = _geometry_spy(monkeypatch)
    for bad in corruptions:
        with pytest.raises(ValueError):
            AreaPressureAssetBundle.from_bytes(_envelope(bad))
        assert calls == []
    encoded = json.dumps(payload, sort_keys=True, allow_nan=False, separators=(",", ":")).encode(
        "utf-8"
    )
    # Still valid JSON/checksum; only the total envelope size is over its cap.
    oversized = _raw_envelope(encoded + b" " * (MAX_BYTES + 1 - len(MAGIC) - 65 - len(encoded)))
    assert len(oversized) == MAX_BYTES + 1
    with pytest.raises(ValueError):
        AreaPressureAssetBundle.from_bytes(oversized)
    assert calls == []


def test_geometry_is_rebuilt_and_invalid_public_assets_never_trust_old_caches(
    declared, monkeypatch
):
    bundle = _bundle(declared)
    calls = _geometry_spy(monkeypatch)
    loaded = AreaPressureAssetBundle.from_bytes(bundle.to_bytes())
    assert "pressure" in calls and "rigid" in calls
    calls.clear()
    loaded.assets_for(declared["specs"])
    assert "pressure" in calls and "rigid" in calls
    for kind in ("nonzero-boundary", "overlapping-tetra", "degenerate-rigid", "local-domain"):
        assets = loaded.assets_for(declared["specs"])
        target = assets["area-stiff"][0]
        if kind == "nonzero-boundary":
            values = target.pressure_local.nodal_pressure.copy()
            values[0] = 1.0
            object.__setattr__(target.pressure_local, "nodal_pressure", values)
        elif kind == "overlapping-tetra":
            cells = target.pressure_local.cells.copy()
            cells[1] = cells[0]
            object.__setattr__(target.pressure_local, "cells", cells)
        elif kind == "degenerate-rigid":
            vertices = target.rigid_local.world_vertices.copy()
            vertices[2] = (vertices[0] + vertices[1]) / 2
            object.__setattr__(target.rigid_local, "world_vertices", vertices)
        else:
            object.__setattr__(target.pressure_local, "pressure_domain", "local")
        with pytest.raises(ValueError):
            AreaPressureAssetBundle.from_assets(declared["specs"], assets)
        assert loaded.to_bytes() == bundle.to_bytes()
    _same(loaded.assets_for(declared["specs"]), declared["assets"])


def test_framing_duplicate_keys_nonfinite_and_bounded_json_structure(declared):
    raw = _bundle(declared).to_bytes()
    for damaged in (
        b"wrong-header\n" + raw,
        raw[:-1],
        raw[: len(MAGIC)] + b"0" * 64 + raw[len(MAGIC) + 64 :],
    ):
        with pytest.raises(ValueError):
            AreaPressureAssetBundle.from_bytes(damaged)
    payload = raw[len(MAGIC) + 65 :]
    for malformed in (
        b'{"schema":"duplicate",' + payload[1:],
        b'{"unexpected":NaN,' + payload[1:],
        b'{"unexpected":Infinity,' + payload[1:],
    ):
        with pytest.raises(ValueError):
            AreaPressureAssetBundle.from_bytes(_raw_envelope(malformed))
    bad = _decode(raw)
    nested = 0
    for _ in range(17):
        nested = [nested]
    bad["models"]["area-stiff"]["pairs"][0]["pressure"]["node_ids"] = nested
    with pytest.raises(ValueError):
        AreaPressureAssetBundle.from_bytes(_envelope(bad))
    bad = _decode(raw)
    bad["models"]["area-stiff"]["pairs"][0]["pressure"]["node_ids"] = [0] * 262145
    with pytest.raises(ValueError):
        AreaPressureAssetBundle.from_bytes(_envelope(bad))


@pytest.mark.cuda
def test_original_and_loaded_assets_two_steps_same_state_and_paired_scene_restore(
    declared, tmp_path
):
    from test_manipulation_area_env import KEYS, H, _make, _native_controls
    from test_manipulation_env import evidence

    from phymeo.backends.mujoco_heterogeneous_area_scene import (
        MuJoCoTyxHeterogeneousAreaSceneCheckpoint,
    )

    bundle = _bundle(declared)
    path = tmp_path / "body-local-pressure.assets"
    path.write_bytes(bundle.to_bytes())
    loaded = AreaPressureAssetBundle.from_bytes(path.read_bytes())
    with (
        ExitStack() as stack,
        evidence(tmp_path / "loaded-assets-two-step-comparison.json.gz") as records,
    ):
        _, original_batch, original_scene = _make(stack, declared, as_env=False)
        _, loaded_batch, loaded_scene = _make(
            stack, declared, as_env=False, assets=loaded.assets_for(declared["specs"])
        )
        pairs = ((original_batch, original_scene), (loaded_batch, loaded_scene))
        workspaces = tuple(
            stack.enter_context(
                batch.prepare_device_workspace(include_area_pressure=True, include_bodies=True)
            )
            for batch, _ in pairs
        )
        assert (
            original_scene.asset_sha256
            == loaded_scene.asset_sha256
            == _decode(loaded.to_bytes())["asset_sha256"]
        )
        for tick in range(2):
            outputs = []
            for (batch, scene), workspace in zip(pairs, workspaces, strict=True):
                batch.set_controls(_native_controls(batch))
                assert scene.step(H, expected_revision=batch.checkpoint_revision) == dict.fromkeys(
                    KEYS, True
                )
                frame = workspace.refresh(expected_revision=batch.checkpoint_revision)
                outputs.append(
                    (
                        batch.integration_snapshot(),
                        scene.last_geometry,
                        workspace.readback(frame),
                        frame.area_catalogs,
                    )
                )
            records.append(dict(tick=tick + 1, original=outputs[0], loaded=outputs[1]))
            _same(outputs[0], outputs[1])
            assert all(state.tick == tick + 1 for state in outputs[0][0].values())
        # Restore the ORIGINAL scene capsule into the scene built from bytes.
        # No extra physical step or substituted asset reconstruction is needed.
        capsule = MuJoCoTyxHeterogeneousAreaSceneCheckpoint.from_bytes(
            original_scene.checkpoint().to_bytes()
        )
        loaded_scene.reset((41,))
        loaded_scene.restore(capsule, expected_revision=loaded_batch.checkpoint_revision)
        frame = workspaces[1].refresh(expected_revision=loaded_batch.checkpoint_revision)
        restored = (
            loaded_batch.integration_snapshot(),
            workspaces[1].readback(frame),
            frame.area_catalogs,
        )
        expected = (outputs[0][0], outputs[0][2], outputs[0][3])
        records.append(dict(restored=restored, expected=expected))
        _same(restored, expected)
        assert loaded_scene.last_geometry == {}
