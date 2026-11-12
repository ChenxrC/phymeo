"""M-B1 S8/M3 capacity consumer using the same-directory qualified A oracle.

Copy this file and the exact tested check_deformable_multibody.py together to an
outside scripts directory. No source checkout test/example import is performed.
These are four-step capacity and mixed-layout checks, not a performance claim.
"""

import argparse
import hashlib
import importlib.util
import json
import traceback
from dataclasses import replace
from pathlib import Path

HELPER_PATH = Path(__file__).with_name("check_deformable_multibody.py").resolve(strict=True)
_SPEC = importlib.util.spec_from_file_location("multibody_scale_original_oracle", HELPER_PATH)
base = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(base)


def _world_key(key, world):
    return base.ph.CollisionKey(world, key.body, key.shape)


def remap(original, world):
    """Remap installation identity only; retain all original physical declarations."""
    bodies = []
    for body in original["bodies"]:
        key = base.ph.ResidentParticipantKey(world, body["key"].participant)
        colliders = tuple(
            dict(c, installed=_world_key(c["installed"], world)) for c in body["colliders"]
        )
        mapping = {c["source"]: c["installed"] for c in colliders}
        # Public replacement invokes the original dataclass admission, rebuilding
        # its derived ArticulationWorld and digest with the new logical world.
        row = replace(body["row"], key=key, collider_keys=mapping)
        bodies.append(dict(body, key=key, row=row, colliders=colliders))
    return dict(
        original,
        world=world,
        node=replace(original["node"], key=world),
        surface=_world_key(original["surface"], world),
        bodies=tuple(bodies),
        materials={
            tuple(_world_key(k, world) for k in pair): law
            for pair, law in original["materials"].items()
        },
    )


def verify_remap(original, remapped):
    """Independent static identity check; no geometry or physics substitution."""
    world = remapped["world"]
    base.require(
        remapped["node"].key == world == remapped["surface"].world,
        "complete node/surface logical world",
    )
    for name in ("kind", "extra", "origin"):
        base.require(
            base.encode(remapped[name]) == base.encode(original[name]),
            "unchanged original A " + name,
        )
    for name in ("model", "initial_state", "constraints", "point_ids", "surfaces", "dissipation"):
        base.require(
            base.encode(getattr(remapped["node"], name))
            == base.encode(getattr(original["node"], name)),
            "unchanged node " + name,
        )
    for old, body in zip(original["bodies"], remapped["bodies"], strict=True):
        base.require(
            body["key"] == base.ph.ResidentParticipantKey(world, old["key"].participant),
            "participant world remap",
        )
        base.require(body["row"].key == body["key"], "public declaration participant identity")
        for name in ("body_id", "model", "initial", "mass", "inertia", "external", "command"):
            base.require(
                base.encode(body[name]) == base.encode(old[name]), "unchanged body physics " + name
            )
        for name in (
            "asset",
            "initial_state",
            "kp",
            "kd",
            "effort_limit",
            "body_ids",
            "initial_control",
        ):
            base.require(
                base.encode(getattr(body["row"], name)) == base.encode(getattr(old["row"], name)),
                "unchanged public row " + name,
            )
        for previous, collider in zip(old["colliders"], body["colliders"], strict=True):
            for name in ("source", "local", "kind"):
                base.require(
                    base.encode(collider[name]) == base.encode(previous[name]),
                    "unchanged asset collider " + name,
                )
            base.require(
                collider["installed"] == _world_key(previous["installed"], world),
                "installed round world remap",
            )
            base.require(
                body["row"].collider_keys[collider["source"]] == collider["installed"],
                "complete original source-to-install mapping",
            )
    expected = {
        tuple(_world_key(k, world) for k in pair): law
        for pair, law in original["materials"].items()
    }
    base.require(
        base.encode(remapped["materials"]) == base.encode(expected),
        "unchanged laws on remapped original pairs",
    )


def cases(identity, journal):
    base.require(identity in ("S8", "M3"), "predeclared capacity identity")
    originals = {world: base.fixture("A", world=world) for world in (7, 31)}
    aliases = ((7, 7), (31, 31), (43, 7), (97, 31)) if identity == "S8" else ((7, 7), (31, 31))
    items = []
    for world, source in aliases:
        item = remap(originals[source], world)
        journal.emit(
            "exact-world-remap", identity=identity, original=originals[source], result=item
        )
        verify_remap(originals[source], item)
        if identity == "M3" and world == 7:
            first = item["bodies"][0]
            rounds = {c["installed"] for c in first["colliders"]}
            item = dict(
                item,
                bodies=(first,),
                materials={
                    pair: law
                    for pair, law in item["materials"].items()
                    if any(k in rounds for k in pair)
                },
            )
            journal.emit("mixed-world-subset", source_participant=first["key"], result=item)
            base.require(
                tuple(b["key"].participant for b in item["bodies"]) == (19,),
                "M3 selects only original first A body",
            )
        items.append(item)
    counts = tuple(len(i["bodies"]) for i in items)
    base.require(
        counts == ((2, 2, 2, 2) if identity == "S8" else (1, 2)),
        "complete predeclared participant counts",
    )
    keys = tuple(b["key"] for i in items for b in i["bodies"])
    base.require(len(set(keys)) == sum(counts), "all physical participants uniquely installed")
    return tuple(items)


def run_case(identity, mode, journal):
    items = cases(identity, journal)
    journal.emit(
        "capacity-case",
        identity=identity,
        worlds=tuple(i["world"] for i in items),
        participant_keys=tuple(b["key"] for i in items for b in i["bodies"]),
        h=base.H,
        steps=4,
        helper_path=str(HELPER_PATH),
        helper_sha256=hashlib.sha256(HELPER_PATH.read_bytes()).hexdigest(),
    )
    # Reuse every original geometry, force, COM moment, source, full J/M,
    # material/fixed reaction and continuous-reference assertion unchanged.
    # In M3 this also requires multibody schema and per-body maps in BOTH worlds.
    result = base.run_window(items, mode, journal, steps=4)
    expected = (16, 32) if identity == "S8" else (8, 12)
    base.require(
        (result["worldsteps"], result["participant_steps"]) == expected,
        "all worlds and participants actually advanced",
    )
    return dict(result, identity=identity)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("interpret", "cuda"), required=True)
    parser.add_argument("--expected-install-root", "--installed-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base.require(
        args.output.is_absolute() and args.expected_install_root.is_absolute(), "absolute paths"
    )
    args.output.mkdir(parents=True, exist_ok=False)
    root = args.expected_install_root.resolve(strict=True)
    journal = base.Journal(args.output / "deformable-multibody-scale.jsonl.gz")
    summary = dict(
        status="RUNNING",
        mode=args.mode,
        cases=[],
        consumer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        helper_path=str(HELPER_PATH),
        helper_sha256=hashlib.sha256(HELPER_PATH.read_bytes()).hexdigest(),
    )
    primary, errors = None, []
    try:
        base.require(
            HELPER_PATH.parent == Path(__file__).resolve().parent
            and Path(base.__file__).resolve() == HELPER_PATH,
            "same-directory original oracle",
        )
        summary["preflight"] = base.provenance(root, preflight=True)
        base.initialize()
        for identity in ("S8", "M3"):
            summary["cases"].append(run_case(identity, args.mode, journal))
        summary.update(
            worldsteps=sum(r["worldsteps"] for r in summary["cases"]),
            participant_steps=sum(r["participant_steps"] for r in summary["cases"]),
        )
        base.require(
            summary["worldsteps"] == 24 and summary["participant_steps"] == 44,
            "both complete predeclared capacity windows",
        )
        summary["status"] = "PASS"
    except BaseException as error:
        primary = error
        summary.update(status="FAIL", error=repr(error), traceback=traceback.format_exc())
        try:
            journal.emit("failure", error=error, traceback=traceback.format_exc())
        except BaseException as capture:
            errors.append(capture)
    finally:
        for label, operation in (
            ("final_provenance", lambda: base.provenance(root)),
            ("trace", journal.close),
        ):
            try:
                summary[label] = operation()
            except BaseException as error:
                errors.append(error)
                summary.update(status="FAIL")
                summary[label + "_failure"] = repr(error)
        summary["owner_cleanup"] = [dict(row) for row in base._OWNER_CLOSE_RECORDS]
        summary["retained_close_failures"] = len(base._RETAINED_CLOSE_FAILURES)
        summary["closed"] = (
            all(row["close_succeeded"] for row in base._OWNER_CLOSE_RECORDS) and "trace" in summary
        )
        try:
            (args.output / "summary.json").write_text(
                json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
            )
        except BaseException as error:
            errors.append(error)
    if errors:
        raise BaseExceptionGroup(
            "M-B1 scale primary or cleanup failure", ([primary] if primary else []) + errors
        )
    if primary is not None:
        raise primary


if __name__ == "__main__":
    main()
