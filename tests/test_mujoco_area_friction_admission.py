"""B1/B2 remaining admission and last-replacement rollback contracts.

Plan: docs/superpowers/plans/2026-11-18-area-friction-production-integration.md.
The CUDA boundary node reuses the
existing empty D40 slider; it does not introduce a new contact-law trajectory.
Private fault seam: batch._clone_bucket returns a real candidate whose restore
is allowed to finish before the LAST replacement raises. No precommit shortcut.
"""

import xml.etree.ElementTree as ET
from contextlib import ExitStack
from dataclasses import replace

import numpy as np
import pytest
from test_mujoco_area_friction import _xml
from test_mujoco_area_friction_guide import _declarations as guide_declarations
from test_mujoco_area_friction_lifecycle import (
    _capture_all,
    _evidence,
    _retained,
    _same,
    _same_capture,
)
from test_mujoco_area_friction_live import H, _cases

from phymeo.backends.area_friction_law import AreaFrictionPairLaw, law_binding
from phymeo.backends.area_pressure_assets import AreaPressureAssetBundle
from phymeo.backends.mujoco_area_pressure_scene import BodyAreaPressureAsset
from phymeo.backends.mujoco_heterogeneous import (
    MuJoCoTyxHeterogeneousBatch,
    MuJoCoTyxWorldSpec,
)
from phymeo.backends.mujoco_heterogeneous_area_scene import (
    MuJoCoTyxHeterogeneousAreaPressureScene,
    MuJoCoTyxHeterogeneousAreaSceneCheckpoint,
)
from phymeo.builders.mujoco_compiler import compile_mujoco_xml


@pytest.fixture(scope="module")
def declarations():
    return _cases()


def _no_leaf_allocation(monkeypatch):
    import phymeo.backends.mujoco_heterogeneous as heterogeneous

    def forbidden(*args, **kwargs):
        pytest.fail("invalid live declaration reached a runtime allocation")

    monkeypatch.setattr(heterogeneous, "MuJoCoRuntime", forbidden)


def test_interpret_refused_before_any_leaf_allocation(declarations, monkeypatch):
    specs, worlds, _ = declarations
    _no_leaf_allocation(monkeypatch)
    with pytest.raises(ValueError, match="requires CUDA"):
        MuJoCoTyxHeterogeneousBatch(specs[:1], worlds[:1], mode="interpret")


def test_v1_friction_none_bundle_cannot_acquire_live_law(declarations):
    specs, _, assets = declarations
    live = specs[0]
    normal = replace(live, area_friction_laws=())
    source = {live.key: assets[live.key]}
    bundle = AreaPressureAssetBundle.from_assets((normal,), source)
    raw = bundle.to_bytes()
    loaded = AreaPressureAssetBundle.from_bytes(raw)
    _same(loaded.assets_for((normal,)), source)
    with pytest.raises(ValueError, match="friction none"):
        loaded.assets_for((live,))
    with pytest.raises(ValueError, match="friction none"):
        AreaPressureAssetBundle.from_assets((live,), source)
    assert loaded.to_bytes() == raw


@pytest.mark.parametrize("invalid", ("nv9", "remaining-native", "limit"))
def test_live_domain_refused_before_device_allocation(declarations, monkeypatch, invalid):
    specs, _, _ = declarations
    xml = _xml(invalid != "limit")
    if invalid == "nv9":
        extra = """<body name="extra" pos=".2 0 0">
          <joint name="extra-x" type="slide" axis="1 0 0"/>
          <joint name="extra-y" type="slide" axis="0 1 0"/>
          <joint name="extra-z" type="slide" axis="0 0 1"/>
          <inertial pos="0 0 0" mass=".1" diaginertia=".001 .001 .001"/>
          <geom type="box" size=".01 .01 .01"/>
        </body>"""
        xml = xml.replace("</worldbody>", extra + "</worldbody>")
        expected = "1..8 DOF"
    elif invalid == "remaining-native":
        xml = xml.replace('contype="0" conaffinity="0"', 'contype="1" conaffinity="1"')
        wall = """<body name="wall" pos=".1 0 0">
          <geom name="wall-boundary" type="box" size=".01 .01 .01"/>
        </body>"""
        xml = xml.replace("</worldbody>", wall + "</worldbody>")
        expected = "area-only constraints"
    else:
        xml = xml.replace('damping="40"', 'damping="0" limited="true" range="-1 1"')
        expected = "area-only constraints"
    compilation = compile_mujoco_xml(xml, require_asset=False)
    model = compilation.fresh_native_model()
    if invalid == "nv9":
        assert model.nv == 9
    elif invalid == "limit":
        assert model.nv == 1 and np.any(model.jnt_limited) and not np.any(model.dof_damping)
    else:
        assert model.nv == 6 and model.ngeom == 3
        assert np.all(model.geom_contype == 1) and np.all(model.geom_conaffinity == 1)
    spec = replace(specs[0], key=invalid, compilation=compilation)
    _no_leaf_allocation(monkeypatch)
    with pytest.raises(ValueError, match=expected):
        MuJoCoTyxHeterogeneousBatch((spec,), (MuJoCoTyxWorldSpec(9, spec.key),), mode="cuda")


@pytest.mark.parametrize("invalid", ("two-dry-rows", "three-slides", "hinge", "damped"))
def test_mixed_guide_refuses_out_of_domain_before_allocation(monkeypatch, invalid):
    base, _, _ = guide_declarations(0.25, 41)
    tree = ET.fromstring(base.compilation.source.primary_xml)
    parent = tree.find("./worldbody/body[@name='pressure']")
    for index in range(2 if invalid == "three-slides" else 1):
        child = ET.SubElement(parent, "body", name=f"link{index}")
        ET.SubElement(child, "inertial", pos="0 0 0", mass="1", diaginertia="1 1 1")
        ET.SubElement(
            child,
            "joint",
            type="hinge" if invalid == "hinge" else "slide",
            axis="0 0 1",
            damping="1" if invalid == "damped" else "0",
            frictionloss="1" if invalid == "two-dry-rows" else "0",
        )
        parent = child
    spec = replace(
        base,
        compilation=compile_mujoco_xml(ET.tostring(tree, encoding="unicode"), require_asset=False),
    )
    _no_leaf_allocation(monkeypatch)
    reason = "damped area route" if invalid == "damped" else "one dry row"
    with pytest.raises(ValueError, match=reason):
        MuJoCoTyxHeterogeneousBatch((spec,), (MuJoCoTyxWorldSpec(41, spec.key),), mode="cuda")


@pytest.mark.cuda
def test_law_buckets_material_guard_and_last_restored_candidate_are_atomic(
    declarations, tmp_path, monkeypatch
):
    old_specs, old_worlds, old_assets = declarations
    template, initial = old_specs[-1], old_worlds[-1].initial_state
    asset = old_assets[template.key][0]
    laws = ((0.5, 0.001), (0.5, 0.001), (0.0, 0.001), (0.5, 0.002))
    keys = (7, 11, 73, 305)
    specs = tuple(
        replace(
            template,
            key=f"empty-law-{i}",
            area_friction_laws=(AreaFrictionPairLaw(asset.identity, mu, vs),),
        )
        for i, (mu, vs) in enumerate(laws)
    )
    worlds = tuple(
        MuJoCoTyxWorldSpec(key, spec.key, initial) for key, spec in zip(keys, specs, strict=True)
    )
    assets = {spec.key: (asset,) for spec in specs}
    # Exact same source, capacities, pairs and state; only law parameters vary.
    assert all(
        spec.compilation.source.model_binary == template.compilation.source.model_binary
        for spec in specs
    )
    assert len({law_binding(spec.area_friction_laws) for spec in specs}) == 3
    with _evidence(tmp_path / "admission-last-restore.json.gz") as evidence, ExitStack() as stack:
        batch = stack.enter_context(MuJoCoTyxHeterogeneousBatch(specs, worlds, mode="cuda"))
        assert batch.layout[7].bucket == batch.layout[11].bucket
        assert len({batch.layout[key].bucket for key in (7, 73, 305)}) == 3
        assert len(batch._buckets) == 3
        for spec, world in zip(specs, worlds, strict=True):
            binding = batch._binding["buckets"][batch.layout[world.world].bucket]
            assert binding["area_friction"] == law_binding(spec.area_friction_laws)
        evidence.update(
            sources={spec.key: spec.compilation.source for spec in specs},
            worlds=worlds,
            assets=assets,
            layout=batch.layout,
            bucket_bindings=batch._binding["buckets"],
            identity_refusals=[],
        )
        # Same body pair but different material must not reuse the declared law.
        for field in ("compliant_material", "rigid_material", "rigid_body"):
            wrong_identity = replace(asset.identity, **{field: "different-" + field})
            wrong_asset = BodyAreaPressureAsset(
                wrong_identity, asset.pressure_local, asset.rigid_local
            )
            wrong = {**assets, specs[0].key: (wrong_asset,)}
            before = batch.checkpoint_revision
            with pytest.raises(ValueError, match="identities|declared area pairs"):
                MuJoCoTyxHeterogeneousAreaPressureScene(batch, wrong)
            assert batch.checkpoint_revision == before and batch.last_step is None
            evidence["identity_refusals"].append(field)
        scene = MuJoCoTyxHeterogeneousAreaPressureScene(batch, assets)
        ws = stack.enter_context(batch.prepare_device_workspace(include_area_pressure=True))
        initial_capture = _capture_all(stack, batch)
        capsule = MuJoCoTyxHeterogeneousAreaSceneCheckpoint.from_bytes(
            scene.checkpoint().to_bytes()
        )
        evidence.update(initial=initial_capture, restore_capsule=capsule.to_bytes())
        # One established empty-domain step gives a genuine accepted receipt,
        # device record and geometry publication that the failed restore must keep.
        assert scene.step(H, expected_revision=batch.checkpoint_revision) == dict.fromkeys(
            keys, True
        )
        accepted = batch.integration_snapshot()
        evidence["accepted"] = accepted
        assert all(row.tick == 1 and row.last_step is not None for row in accepted.values())
        for row in accepted.values():
            assert all(
                len(pair["sample_ids"]) == 0
                for pair in row.last_step.raw_contact_fields["area_friction"]
            )
        revision, geometry, publication = (
            batch.checkpoint_revision,
            scene.last_geometry,
            batch.last_step,
        )
        frame = ws.refresh(expected_revision=revision)
        arrays, raw = ws.readback(frame), scene.checkpoint().to_bytes()
        owners = tuple(batch._buckets)
        owner_revisions = tuple(owner.checkpoint_revision for owner in owners)
        evidence.update(
            initial=initial_capture,
            restore_capsule=capsule.to_bytes(),
            accepted=accepted,
            geometry=geometry,
            arrays=arrays,
            accepted_capsule=raw,
            completed_replacements=[],
        )
        original_clone = batch._clone_bucket
        candidates = []
        with monkeypatch.context() as patch:

            def clone_then_fail_after_restore(owner):
                index = next(i for i, old in enumerate(owners) if old is owner)
                candidate = original_clone(owner)
                candidates.append(candidate)
                restore = candidate.restore

                def restored_then_fault(*args, **kwargs):
                    result = restore(*args, **kwargs)
                    restored = candidate.integration_snapshot()
                    assert all(row.tick == 0 and row.last_step is None for row in restored.values())
                    evidence["completed_replacements"].append(dict(index=index, state=restored))
                    if index == len(owners) - 1:
                        raise RuntimeError("last replacement restored, before batch publication")
                    return result

                patch.setattr(candidate, "restore", restored_then_fault)
                return candidate

            patch.setattr(batch, "_clone_bucket", clone_then_fail_after_restore)
            with pytest.raises(RuntimeError, match="last replacement restored"):
                scene.restore(capsule, expected_revision=revision)
        assert [row["index"] for row in evidence["completed_replacements"]] == [0, 1, 2]
        assert all(
            current is original for current, original in zip(batch._buckets, owners, strict=True)
        )
        assert tuple(owner.checkpoint_revision for owner in owners) == owner_revisions
        assert all(candidate._closed for candidate in candidates)
        assert batch.last_step is publication and ws._frame is frame
        _retained(batch, scene, ws, frame, arrays, accepted, revision, raw, geometry)
        result = scene.restore(capsule, expected_revision=revision)
        assert result.operation == "restore" and result.revision_after == revision + 1
        assert all(
            current is not original
            for current, original in zip(batch._buckets, owners, strict=True)
        )
        assert not scene.last_geometry and batch.last_step is None
        restored = _capture_all(stack, batch)
        _same_capture(restored, initial_capture)
        fresh = ws.refresh(expected_revision=batch.checkpoint_revision)
        assert fresh.ticks == (0,) * len(keys)
        evidence.update(
            retried=restored,
            restored_arrays=ws.readback(fresh),
            resources=batch.resource_snapshot(),
        )
