"""Fixed PushT boundary routes: sampled references, then real CUDA motor steps.

Qualification consumes the prior bounded candidate/IK dataset; it never repeats
those endpoints or edits the object state. Dynamics remains owned by the Env.
"""

import argparse
import gzip
import hashlib
import json
import sys
from dataclasses import fields, is_dataclass
from pathlib import Path

import mujoco as mj
import numpy as np

from phymeo.control.manipulation_diagnostic import NativeJointPD, solve_native_point_ik
from phymeo.control.manipulation_path import audit_native_joint_path
from phymeo.control.manipulation_push import (
    NativePlanarPushContactCandidate,
    NativePlanarPushContacts,
    PlanarPushBoundaryEdge,
    PlanarPushRectangle,
)
from phymeo.control.manipulation_push_rank import rank_planar_push_candidates
from phymeo.training.manipulation_tasks import manipulation_scenes

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_manipulation_control import run  # noqa: E402

SOURCE_SHA = "8d9a660cd966c4449d0cbde6cd09b57765af1d40e11186e4f54f1b5d8c3fe3c5"
TIMES = (0.0, 0.2, 0.8, 1.1, 1.8, 2.0)
STAGES = ("initial", "settle", "approach", "contact", "push", "hold")


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value):
        return {f.name: plain(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    return value


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def model_digest(model):
    binary = np.empty(mj.mj_sizeModel(model), dtype=np.uint8)
    mj.mj_saveModel(model, None, binary)
    return hashlib.sha256(binary.tobytes()).hexdigest()


def tuple_values(value):
    return tuple(tuple_values(v) for v in value) if isinstance(value, list) else value


def restore_query(raw):
    values = dict(raw)
    for name in ("q", "plane_axes_world"):
        array = np.asarray(values[name], dtype=np.float64)
        values[name] = np.frombuffer(array.tobytes(), dtype=np.float64).reshape(array.shape)
    for name, cls in (
        ("rectangles", PlanarPushRectangle),
        ("boundary", PlanarPushBoundaryEdge),
        ("candidates", NativePlanarPushContactCandidate),
    ):
        values[name] = tuple(
            cls(**{k: tuple_values(v) for k, v in item.items()}) for item in raw[name]
        )
    for name in ("object_com_world", "plane_origin_world"):
        values[name] = tuple(values[name])
    return NativePlanarPushContacts(**values)


def write(path, value):
    Path(path).write_text(json.dumps(plain(value), indent=2, allow_nan=False), encoding="utf-8")


def qualify(source, output):
    require(source.is_file() and digest(source) == SOURCE_SHA, "candidate source binding differs")
    require(output.is_absolute() and not output.exists(), "fresh absolute qualification output")
    saved = json.loads(source.read_text(encoding="utf-8"))
    require(saved["schema"] == "original-push-boundary-endpoints-v1", "candidate schema differs")
    require([r["summary"]["world"] for r in saved["rows"]] == [5, 6, 7], "world membership differs")
    output.mkdir(parents=True)
    scenes, plans = manipulation_scenes(), []
    for row in saved["rows"]:
        world = row["summary"]["world"]
        scene = scenes[world]
        model = mj.MjModel.from_xml_string(scene.mjcf)
        initial_digest = model_digest(model)
        require(initial_digest == row["summary"]["model_sha256"], "native model differs")
        query = restore_query(row["query"])
        require(np.array_equal(query.q, model.qpos0), "candidate q differs from original q0")
        require(
            [c.index for c in query.candidates] == list(range(len(query.candidates))),
            "source candidate order differs",
        )
        target = next(
            t for t in scene.targets if t.key == scene.goals[scene.default_goal].target_key
        )
        desired = np.subtract(target.position[:2], scene.initial_object_position[:2])
        ranked = rank_planar_push_candidates(query, desired, target.yaw)
        prior = {item["index"]: item for item in row["ik"]}
        require(set(prior) == {c.index for c in query.candidates}, "saved IK membership differs")
        addresses = [int(model.jnt_qposadr[model.joint(n).id]) for n in scene.robot_joint_names]
        q0 = model.qpos0.copy()
        selected, attempts = None, []
        for rank in ranked:
            item = dict(ranking=rank, reason=None, push_ik=None, paths=None)
            if selected is not None:
                item["reason"] = "not-attempted-after-first-qualified"
            elif not rank.usable:
                item["reason"] = "geometric-candidate-rejected"
            elif not prior[rank.index]["both_converged"]:
                item["reason"] = "previous-endpoints-not-both-converged"
            else:
                candidate = query.candidates[rank.index]
                pre = np.asarray(prior[rank.index]["precontact"]["configuration"], dtype=float)
                contact = np.asarray(prior[rank.index]["contact"]["configuration"], dtype=float)
                push_target = np.asarray(candidate.contact_center_world) - 0.02 * np.asarray(
                    candidate.outward_normal_world
                )
                push = solve_native_point_ik(
                    model,
                    push_target,
                    contact,
                    scene.robot_joint_names,
                    local_point=(0.0, 0.0, -0.16),
                )
                item["push_ik"] = push
                item["push_seed"] = contact
                item["push_target"] = push_target
                if not push.converged:
                    item["reason"] = "push-endpoint-not-converged"
                else:
                    pairs = tuple(("pusher", name, 0.0005) for name in scene.object_geom_names)
                    push_pairs = tuple(("pusher", name, 0.021) for name in scene.object_geom_names)
                    approach = audit_native_joint_path(
                        model, (0.0, 0.2, 0.8), (q0, q0, pre), scene.robot_joint_names
                    )
                    entry = audit_native_joint_path(
                        model,
                        (0.0, 0.3),
                        (pre, contact),
                        scene.robot_joint_names,
                        allowed_contacts=pairs,
                    )
                    pushing = audit_native_joint_path(
                        model,
                        (0.0, 0.7, 0.9),
                        (contact, push.configuration, push.configuration),
                        scene.robot_joint_names,
                        allowed_contacts=push_pairs,
                    )
                    item["paths"] = dict(approach=approach, entry=entry, push=pushing)
                    clear = all(p.sampled_clear for p in (approach, entry, pushing))
                    item["reason"] = "qualified" if clear else "sampled-reference-collision"
                    if clear:
                        knots = (q0, q0, pre, contact, push.configuration, push.configuration)
                        selected = dict(
                            candidate=plain(candidate),
                            direction_world=list(-np.asarray(candidate.outward_normal_world)),
                            q_knots=plain(knots),
                            nodes=[
                                (t, label, q[addresses].tolist())
                                for t, label, q in zip(TIMES, STAGES, knots, strict=True)
                            ],
                            path_bindings=[p.input_binding for p in (approach, entry, pushing)],
                            sample_counts=[p.sample_count for p in (approach, entry, pushing)],
                            allowances=dict(approach=[], entry=pairs, push=push_pairs),
                        )
            attempts.append(item)
            print(
                json.dumps(dict(world=world, index=rank.index, reason=item["reason"])), flush=True
            )
        require(
            model_digest(model) == initial_digest and np.array_equal(q0, model.qpos0),
            "query changed original state/model",
        )
        evidence = output / f"world-{world}-attempts.json.gz"
        with gzip.open(evidence, "xt", encoding="utf-8") as stream:
            json.dump(
                plain(
                    dict(
                        world=world,
                        query_binding=query.input_binding,
                        attempts=attempts,
                        selected=selected,
                    )
                ),
                stream,
                allow_nan=False,
            )
        plan = dict(
            world=world,
            model_sha256=initial_digest,
            source_sha256=SOURCE_SHA,
            selected=selected,
            goal=plain(scene.goals[scene.default_goal]),
            target=plain(target),
            attempts_file=evidence.name,
            attempts_sha256=digest(evidence),
        )
        plans.append(plan)
        write(
            output / "plans.json",
            dict(schema="registered-push-routes-v1", source_sha256=SOURCE_SHA, plans=plans),
        )
    return plans


def frozen_builder(plan, qualified):
    def build(scene, model):
        require(plan["selected"] is not None, "no qualified reference")
        require(model_digest(model) == plan["model_sha256"], "execution model differs")
        require(
            digest(qualified / plan["attempts_file"]) == plan["attempts_sha256"],
            "path evidence differs",
        )
        with gzip.open(qualified / plan["attempts_file"], "rt", encoding="utf-8") as stream:
            evidence = json.load(stream)
        require(evidence["selected"] == plan["selected"], "frozen nodes differ from checked path")
        names = scene.robot_joint_names
        kp = [250.0 if kind == "P" else 25.0 for kind in scene.robot_kinds]
        kd = [12.0 if kind == "P" else 3.0 for kind in scene.robot_kinds]
        nodes = [(t, label, np.asarray(q)) for t, label, q in plan["selected"]["nodes"]]
        audit = dict(
            valid=True,
            goal=scene.goals[scene.default_goal],
            kp=kp,
            kd=kd,
            robot_joint_names=names,
            reference_knots=nodes,
            selected=plan["selected"],
            source_sha256=SOURCE_SHA,
            model_sha256=plan["model_sha256"],
            qualification="three phase-specific sampled reference paths; no dynamic object/CCD guarantee",
        )
        return NativeJointPD(model, names, kp, kd), nodes, audit

    return build


def execute(qualified, output):
    require(output.is_absolute() and not output.exists(), "fresh absolute CUDA output")
    data = json.loads((qualified / "plans.json").read_text(encoding="utf-8"))
    require(
        data["schema"] == "registered-push-routes-v1" and data["source_sha256"] == SOURCE_SHA,
        "plan source differs",
    )
    require([p["world"] for p in data["plans"]] == [5, 6, 7], "incomplete qualified worlds")
    output.mkdir(parents=True)
    attempts = []
    for plan in data["plans"]:
        world = plan["world"]
        if plan["selected"] is None:
            attempts.append(dict(world=world, status="PREFLIGHT_REJECTED", physical_steps=0))
        else:
            run(
                output / f"world-{world}",
                2000,
                (world,),
                plan_builder=frozen_builder(plan, qualified),
                record_entry=True,
            )
            manifest = json.loads(
                (output / f"world-{world}" / "manifest.json").read_text(encoding="utf-8")
            )
            attempts.append(
                dict(
                    world=world,
                    status=manifest["status"],
                    physical_steps=manifest["steps"],
                    recorded_steps=manifest["recorded_steps"],
                    success=manifest["worlds"][0]["task_success"],
                )
            )
        write(output / "attempts.json", attempts)
    print(json.dumps(attempts), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("qualify", "execute"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "qualify":
        qualify(args.source, args.output)
    else:
        execute(args.source, args.output)
