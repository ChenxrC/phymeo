"""Recover a missing public scene checkpoint from an exact accepted PushT prefix.

New runs persist the native capsule. This legacy prefix only archived the FP32
states and commands. Replay the shortest available prefix and compare every
published state, control, contact and solver observation before exporting it.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from phymeo.backends.mujoco_heterogeneous import MuJoCoTyxHeterogeneousBatch, MuJoCoTyxWorldSpec
from phymeo.backends.mujoco_heterogeneous_area_scene import (
    MuJoCoTyxHeterogeneousAreaPressureScene,
    MuJoCoTyxHeterogeneousAreaSceneCheckpoint,
)
from phymeo.backends.mujoco_runtime import MuJoCoControl

from .pusht_public_scene import H, _step_record, build_scene_spec, build_side_asset
from .pusht_release_entry import prepare_release_entry

SCHEMA = "phymeo-pusht-reconstructed-scene-v1"


def _require(ok, message):
    if not ok:
        raise ValueError("active checkpoint: " + message)


def _sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require_saved_state(snapshot, state):
    _require(
        snapshot.tick == state["local_tick"] and snapshot.time == state["time"],
        "local tick or clock differs",
    )
    for name in ("q", "v"):
        actual = np.asarray(getattr(snapshot, name))
        expected = np.asarray(state[name])
        _require(
            np.array_equal(actual, expected)
            and actual.astype(np.float32).tobytes() == expected.astype(np.float32).tobytes(),
            "accepted " + name + " differs",
        )


def selected_steps(rows, first_tick, count):
    tick = first_tick
    for row in rows:
        _require(
            type(row["selected"]) is bool and row["original_input_tick"] == tick,
            "trial order or publication flag",
        )
        if not row["selected"]:
            _require(
                row["restored"] is True and row["rollback_verification"]["passed"] is True,
                "unverified rejected trial",
            )
            continue
        _require(row["original_output_tick"] == tick + 1, "nonadjacent trial endpoint")
        yield row
        tick += 1
    _require(tick == first_tick + count, "selected count")


def recover(prior, release, output):
    output, release = Path(output).resolve(), Path(release).resolve()
    capsule = output.with_suffix(".scene")
    _require(not output.exists() and not capsule.exists(), "fresh output required")
    saved = json.loads(release.read_text("utf-8"))
    entry = prepare_release_entry(prior, output.with_suffix(".source-routes"))
    _require(
        saved["provenance"] == entry.provenance
        and saved["status"] == "regular_step_budget"
        and saved["regular_plan_ready"]
        and saved["start_tick"] == entry.tick,
        "release source binding",
    )
    result = dict(
        schema=SCHEMA,
        world=6,
        release_sha256=_sha(release),
        release_journal_sha256=saved["journal"]["sha256"],
        checked_steps=0,
        asset_sha256=entry.provenance["scene_asset_sha256"],
    )
    spec, asset = build_scene_spec(entry.scene.mjcf), build_side_asset(entry.model)
    qa = int(entry.model.jnt_qposadr[entry.model.joint("object-free").id])
    declaration = MuJoCoTyxWorldSpec(6, spec.key, entry.initial)
    with MuJoCoTyxHeterogeneousBatch((spec,), (declaration,), mode="cuda") as batch:
        scene = MuJoCoTyxHeterogeneousAreaPressureScene(batch, {spec.key: (asset,)})
        before = batch.integration_snapshot()[6]
        require_saved_state(before, saved["initial_state"])
        _require(scene.asset_sha256 == result["asset_sha256"], "wrong scene asset")
        for row in selected_steps(saved["trials"], entry.tick, saved["selected_steps"]):
            tick = row["original_input_tick"]
            require_saved_state(before, row["input_state"])
            c = row["control"]
            _require(c["tick"] == tick and c["output_tick"] == tick + 1, "command tick")
            batch.set_controls(
                {
                    6: MuJoCoControl(
                        c["ctrl"], qfrc_applied=c["qfrc_applied"], xfrc_applied=c["xfrc_applied"]
                    )
                }
            )
            _require(
                dict(scene.step(H, expected_revision=batch.checkpoint_revision)) == {6: True},
                f"step {tick} not published",
            )
            after = batch.integration_snapshot()[6]
            require_saved_state(after, row["output_state"])
            command = {
                name: c[name]
                for name in ("tick", "output_tick", "ctrl", "qfrc_applied", "xfrc_applied")
            }
            observed = _step_record(
                entry.model, asset, before, after, scene.last_geometry[6][0], command, qa, 0.0
            )
            for name in (
                "input_state",
                "output_state",
                "control",
                "geometry",
                "area",
                "native",
                "solver",
            ):
                # The archived JSON encodes native geometry-name tuples as arrays.
                current = json.loads(json.dumps(observed[name], allow_nan=False))
                _require(row[name] == current, f"step {tick} differs in {name}")
            before = after
            result["checked_steps"] += 1
        require_saved_state(before, saved["final_state"])
        raw = scene.checkpoint().to_bytes()
    with capsule.open("xb") as f:
        f.write(raw)
    result.update(
        capsule=capsule.name,
        sha256=hashlib.sha256(raw).hexdigest(),
        size=len(raw),
        source_tick=saved["controller_final"]["tick"],
        state=saved["final_state"],
    )
    with output.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    return result


def restore_bound_entry(scene, batch, entry, release_report_path, checkpoint_metadata_path):
    meta = Path(checkpoint_metadata_path).resolve()
    doc = json.loads(meta.read_text("utf-8"))
    _require(
        doc["schema"] == SCHEMA
        and doc["world"] == 6
        and doc["release_sha256"] == _sha(release_report_path)
        and doc["state"] == entry.state
        and doc["source_tick"] == entry.tick
        and doc["asset_sha256"] == scene.asset_sha256,
        "checkpoint source/state/model",
    )
    leaf = Path(doc["capsule"])
    _require(leaf.name == doc["capsule"] and not leaf.is_absolute(), "capsule must be a sibling")
    path = meta.parent / leaf
    _require(path.stat().st_size == doc["size"] and _sha(path) == doc["sha256"], "capsule bytes")
    capsule = MuJoCoTyxHeterogeneousAreaSceneCheckpoint.from_bytes(path.read_bytes())
    scene.restore(capsule, expected_revision=batch.checkpoint_revision)
    snap = batch.integration_snapshot()[6]
    require_saved_state(snap, entry.state)
    _require(
        snap.last_step is not None
        and snap.last_step.tick == snap.tick
        and snap.last_step.output_time == snap.time,
        "restored prior receipt",
    )
    return dict(
        checkpoint_sha256=doc["sha256"],
        checked_prefix_steps=doc["checked_steps"],
        restored_local_tick=snap.tick,
        absolute_input_tick=entry.tick,
        time=snap.time,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary", required=True)
    p.add_argument("--release", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    out = recover(args.summary, args.release, args.output)
    print(json.dumps({k: out[k] for k in ("checked_steps", "sha256", "source_tick", "size")}))
