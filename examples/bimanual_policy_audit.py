"""Paired counterfactual targets, wrist-image interventions and per-pad contact impulses."""

import argparse
import hashlib
import importlib
import json
import math
import sys
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np


CONDITIONS = ("normal", "frozen_first", "left_masked", "right_masked", "both_masked")
SIDES = ("left", "right")
MODEL_DEFAULTS = {"tool_profile": "single-pad", "load_profile": "standard", "stance": "original"}


def _hash(data):
    return hashlib.sha256(data).hexdigest()


def _canonical(value):
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()


def _arrays(values):
    digest = hashlib.sha256()
    for value in values:
        arr = np.ascontiguousarray(value)
        digest.update(_canonical([str(arr.dtype), arr.shape]))
        digest.update(arr.tobytes())
    return digest.hexdigest()


def load_frozen(path, source_dir):
    """Bind actor code and weights to the actual trained/selected checkpoint before importing it."""
    directory = Path(source_dir).resolve()
    policy = json.loads(path.read_bytes())
    checkpoint = json.loads(path.with_name("checkpoint.json").read_bytes())
    contract = checkpoint["contract"]
    if (policy["schema"] != "bimanual-visual-policy-v1" or policy["contract_sha256"] != _hash(_canonical(contract))
            or policy["task"] != contract["task"] or policy["arms"] != contract["arms"] or policy["arms"] != "both"
            or policy.get("variant", "centroid") != contract.get("policy_variant", "centroid")
            or any(policy[key] != contract.get(key, default)
                   for key, default in MODEL_DEFAULTS.items() if key in policy)
            or policy["parameters"] != contract["parameters"]
            or not 0 < policy["generations"] == checkpoint["generation"] == len(checkpoint["updates"])):
        raise ValueError("frozen policy and its originating training checkpoint disagree")
    selection = np.asarray(policy["selection_returns"], dtype=float)
    if selection.shape != (len(contract["selection_seeds"]), 2) or not np.isfinite(selection).all():
        raise ValueError("missing paired model-selection returns")
    winner = int(np.argmax(selection.mean(axis=0)))
    weights = np.asarray(policy["weights"], dtype=float)
    selected_weights = (checkpoint["mean"], checkpoint["updates"][-1]["best"])[winner]
    if (policy["selected"] != ("mean", "latest-elite")[winner] or weights.shape != (len(policy["parameters"]),)
            or not np.isfinite(weights).all() or not np.array_equal(weights, selected_weights)):
        raise ValueError("frozen weights disagree with selected pre-holdout candidate")
    required = {"bimanual_musa_train.py", "_bimanual_visual_policy.py", "bimanual_musa_env.py",
                "bimanual_wrist_render.py"}
    admitted = required | {"_bimanual_image_pose.py", "_bimanual_fiducial.py"}
    if policy.get("variant") == "fiducial":
        required |= {"_bimanual_fiducial.py"}
    names = set(contract["sources"])
    if not required <= names <= admitted:
        raise ValueError("unexpected frozen source manifest")
    for name, expected in contract["sources"].items():
        source = directory / name
        if _hash(source.read_bytes()) != expected:
            raise ValueError(f"source hash mismatch for {source}")
        loaded = sys.modules.get(source.stem)
        if loaded is not None and Path(loaded.__file__).resolve() != source:
            raise ValueError(f"another copy of {name} is already imported")
    sys.path.insert(0, str(directory))
    visual = importlib.import_module("_bimanual_visual_policy")
    environment = importlib.import_module("bimanual_musa_env")
    camera = importlib.import_module("bimanual_wrist_render")
    for module in (visual, environment, camera):
        if Path(module.__file__).resolve().parent != directory:
            raise ValueError(f"import did not resolve to verified frozen code: {module.__name__}")
    pose_module = sys.modules.get("_bimanual_image_pose")
    if pose_module is not None and (
        "_bimanual_image_pose.py" not in names or Path(pose_module.__file__).resolve().parent != directory
    ):
        raise ValueError("visual registration helper is outside the frozen source manifest")
    if list(visual.PARAMETERS) != policy["parameters"]:
        raise ValueError("actor parameter order mismatch")
    return contract, policy, visual, environment, camera


def _marker(node):
    name = node.get("name", "")
    return (node.tag == "site" and name == "task_goal") or (node.tag == "geom" and name.startswith("goal_marker_"))


def target_scene(original, environment, sign):
    """Replace only goal-site/zero-collision marker coordinates; keep element and ID order."""
    xy = np.array(original.goal_xy)
    if original.task == "carry":
        xy[1] = original.start_xyz[1] + sign * .040
        yaw = original.goal_yaw
    else:
        yaw = sign * math.radians(15)
    if _hash(original.xml.encode()) != original.xml_sha256:
        raise ValueError("original scene does not match its source model digest")
    xml = ET.fromstring(original.xml)
    world = xml.find("worldbody")
    originals = [n for n in world if _marker(n)]
    source_markers = {n.get("name"): n for n in originals}
    if len(source_markers) != len(originals):
        raise ValueError("duplicate target element in source model")
    # Physical side lips need not have a target outline. The original XML
    # specifies exactly which components contribute edges and their ID order.
    outlined = [box for i, box in enumerate(original.boxes) if f"goal_marker_{i}_0" in source_markers]
    baseline = ET.Element("worldbody")
    environment._markers(baseline, original.goal_xy, original.goal_yaw, outlined)
    if not outlined or [(n.tag, n.attrib) for n in baseline] != [(n.tag, n.attrib) for n in originals]:
        raise ValueError("frozen goal generator does not reproduce the original complete outline")
    markers = ET.Element("worldbody")
    environment._markers(markers, xy, yaw, outlined)
    if set(source_markers) != {n.get("name") for n in markers}:
        raise ValueError("source and replacement targets have different element names")
    variables = {"geom": ("fromto",), "site": ("pos", "euler")}
    for new in markers:
        old = source_markers[new.get("name")]
        if old.tag == "geom" and (old.get("contype"), old.get("conaffinity")) != ("0", "0"):
            raise ValueError("counterfactual target unexpectedly has collision response")
        for attribute in variables[old.tag]:
            old.set(attribute, new.attrib[attribute])
        if old.attrib != new.attrib:
            raise ValueError("target replacement would change non-pose attributes")
    result = ET.tostring(xml, encoding="unicode")
    for old in source_markers.values():
        for attribute in variables[old.tag]:
            old.set(attribute, "counterfactual")
    material_and_robot_hash = _hash(ET.tostring(xml, encoding="utf-8"))
    return replace(original, xml=result, xml_sha256=_hash(result.encode()), goal_xy=tuple(xy),
                   goal_yaw=float(yaw)), material_and_robot_hash


def intervene(condition, image, first):
    if image.shape != first.shape or image.shape[0] != 2 or image.dtype != np.uint8:
        raise ValueError("expected two synchronised uint8 wrist views")
    if condition == "normal":
        return image
    if condition == "frozen_first":
        return first
    if condition not in ("left_masked", "right_masked", "both_masked"):
        raise ValueError(f"unknown wrist intervention {condition}")
    masked = image.copy()
    if condition in ("left_masked", "both_masked"):
        masked[0] = 0
    if condition in ("right_masked", "both_masked"):
        masked[1] = 0
    return masked


class Impulses:
    """Sample native contact force and pure couple at each original constraint input."""

    def __init__(self, task, rotation):
        self.task, self.rotation = task, rotation
        self.samples = []
        self.body = int(task.model.body(task.scene.object_body).id)

    def observe(self, before, after):
        task = self.task
        receipt = after.last_step
        h = float(receipt.h)
        if not (receipt.tick == after.tick == before.tick + 1 and abs(receipt.input_time - before.time) < 1e-7):
            raise ValueError("contact receipt phase or native step order changed")
        addr = task.root_q
        origin = before.q[addr:addr+3]
        rot = self.rotation(before.q[addr+3:addr+7])
        com = origin + rot @ task.model.body_ipos[self.body]
        clearance = min(origin[2] + rot[2] @ offset - np.abs(rot[2]) @ half for offset, half in task.scene.boxes)
        forces, torques = np.zeros((2, 3)), np.zeros((2, 3))
        normal = np.zeros(2)
        floor_up = 0.
        for contact in receipt.contacts:
            a, b = map(int, contact.geometry_ids)
            if (a in task.payload_geoms) == (b in task.payload_geoms):
                continue
            sign, counterpart = (-1., b) if a in task.payload_geoms else (1., a)
            local = np.asarray(contact.wrench_contact_frame)
            if local[0] <= 0:
                continue
            frame = np.asarray(contact.frame).reshape(3, 3)
            force = sign * (frame.T @ local[:3])
            if counterpart == task.floor:
                floor_up += force[2]
            elif counterpart in task.pad_geoms:
                side = task.pad_owner[counterpart] if hasattr(task, "pad_owner") else task.pad_geoms.index(counterpart)
                normal[side] += local[0]
                forces[side] += force
                torques[side] += sign * (frame.T @ local[3:]) + np.cross(contact.position - com, force)
        airborne = clearance >= .003 and floor_up < .05 * task.scene.mass * 9.81
        self.samples.append((before.time, h, forces, torques, normal, floor_up, clearance, airborne))

    def finish(self, path, goal_xy, start_xy):
        times, dt, forces, torques, normals, floor, clear, air = map(np.asarray, zip(*self.samples, strict=True))
        forward = np.subtract(goal_xy, start_xy).astype(float)
        forward /= np.linalg.norm(forward)
        impulse = np.einsum("t,tac->ac", dt, forces)
        angular = np.einsum("t,tac->ac", dt, torques)
        up = forces[:, :, 2]
        loaded = normals >= 1.
        air_s = float(dt[air].sum())
        np.savez_compressed(path, input_time_s=times, h_s=dt, force_on_payload_n=forces,
                            torque_about_payload_com_nm=torques, pad_normal_n=normals,
                            floor_vertical_n=floor, input_clearance_m=clear, airborne=air)
        return dict(contact_npz=path.name, contact_npz_sha256=_hash(path.read_bytes()), physical_steps=len(dt),
                    total_impulse_world_ns=impulse.tolist(), total_angular_impulse_world_nms=angular.tolist(),
                    forward_axis_world_xy=forward.tolist(), forward_impulse_ns=(impulse[:, :2] @ forward).tolist(),
                    world_x_impulse_ns=impulse[:, 0].tolist(), yaw_impulse_nms=angular[:, 2].tolist(),
                    airborne_duration_s=air_s, both_loaded_duration_s=float(dt[loaded.all(axis=1)].sum()),
                    airborne_both_loaded_ratio=(float(dt[air & loaded.all(axis=1)].sum()/air_s) if air_s else None),
                    airborne_vertical_impulse_ns=(dt[air] @ up[air]).tolist(),
                    airborne_gravity_impulse_magnitude_ns=air_s*self.task.scene.mass*9.81)


def _run_goal(args, contract, policy, actor_module, environment, camera_module, seed, sign):
    class ObservedBatch(environment.BimanualBatchEnv):
        observers = None

        def _states(self):
            states = super()._states()
            if self.observers is not None:
                for key, state in states.items():
                    if not self.finished[key]:
                        self.observers[key].observe(self.states[key], state)
            return states

    keys = tuple(range(len(CONDITIONS)))
    fiducials = policy.get("variant") == "fiducial"
    tool_config = {key: contract[key] for key in MODEL_DEFAULTS if key in contract}
    with ExitStack() as stack:
        env = stack.enter_context(ObservedBatch(policy["task"], seed, world_keys=keys, arm_dofs=args.arm_dofs,
                                                mode=args.mode, cache_root=args.cache_dir, action_repeat=contract["action_repeat"],
                                                horizon=contract["horizon"], **tool_config))
        original_sha256 = env.scene.xml_sha256
        env.scene, scene_invariant = target_scene(env.scene, environment, sign)
        env.compilation = environment.compile_mujoco_xml(env.scene.xml, require_asset=False)
        env.model = env.compilation.mj_model
        initial = env.reset()
        visual = stack.enter_context(camera_module.BimanualWristRenderer(
            env.model, args.cache_dir / "camera", library_path=args.library, size=(contract["image_size"],)*2,
            device_index=args.device_index, **({"fiducials": True} if fiducials else {})))
        env.observers = {k: Impulses(env.worlds[k], environment._rotation) for k in keys}
        actors = [actor_module.VisualPolicy(policy["task"], env.model, policy["weights"],
                                            variant=policy.get("variant", "centroid"),
                                            decision_dt=env.scene.timestep*contract["action_repeat"],
                                            servo_gains=(env.worlds[k].kp, env.worlds[k].kd)) for k in keys]
        observations = {k: initial[k][0] for k in keys}
        info = {k: initial[k][1] for k in keys}
        initial_hashes = {_arrays([p["qpos"], p["qvel"]]) for p in info.values()}
        if len(initial_hashes) != 1:
            raise ValueError("paired worlds have different initial states")
        initial_hash = initial_hashes.pop()
        first, logs, ended = {}, {k: [] for k in keys}, set()
        before = env.runtime.musa_device_report() if args.mode == "musa" else None
        for decision in range(contract["horizon"]):
            actions = {k: np.zeros(14) for k in keys}
            for k in keys:
                if k in ended:
                    continue
                p = info[k]
                raw = visual.observe(p["qpos"], p["tick"], p["time_s"], epoch=k)
                if k not in first:
                    first[k] = raw.copy()
                pixels = intervene(CONDITIONS[k], raw, first[k])
                tactile = {"wrist_force_world_n": p["wrist_force_world_n"]} if fiducials else {}
                actions[k] = actors[k].act(observations[k], pixels, **tactile)
                logs[k].append(dict(input_tick=p["tick"], raw_rgb_sha256=[_arrays([x]) for x in raw],
                                    actor_rgb_sha256=[_arrays([x]) for x in pixels], action=actions[k].tolist()))
            next_states = env.step_torque(actions) if fiducials else env.step(actions)
            for k in keys:
                if k in ended:
                    continue
                observations[k], reward, done, truncated, info[k] = next_states[k]
                logs[k][-1].update(reward=float(reward), output_tick=info[k]["tick"],
                                   payload_xyz_m=info[k]["payload_xyz_m"], payload_yaw_rad=info[k]["payload_yaw_rad"])
                if done or truncated:
                    ended.add(k)
            if len(ended) == len(keys):
                break
        if len(ended) != len(keys):
            raise ValueError("original episode horizon did not terminate all lanes")
        if len({tuple(entry[0]["raw_rgb_sha256"]) for entry in logs.values()}) != 1:
            raise ValueError("paired interventions started from different raw wrist images")
        after = env.runtime.musa_device_report() if args.mode == "musa" else None
        if after is not None and not (after["backend"] == "musa" and after["device_name"] and after["arch"]
                                      and after["completed_kernels"] > before["completed_kernels"]
                                      and after["completed_kernels"] == after["launched_kernels"]):
            raise ValueError("missing MUSA completion evidence")
        prefix = f"{seed}-{'minus' if sign < 0 else 'plus'}"
        (args.out / f"{prefix}.xml").write_bytes(env.scene.xml.encode())
        condition_results = {}
        for k, name in enumerate(CONDITIONS):
            p = info[k]
            metrics = {field: p[field] for field in ("success", "goal_error_xy_m", "goal_yaw_error_rad",
                       "goal_dwell_s", "max_clearance_m", "carried_airborne", "payload_xyz_m", "payload_yaw_rad")}
            impulse = env.observers[k].finish(args.out / f"{prefix}-{name}.npz", env.scene.goal_xy, env.scene.start_xyz[:2])
            log_path = args.out / f"{prefix}-{name}.json"
            log_path.write_bytes(_canonical(logs[k]))
            condition_results[name] = dict(**metrics, impulses=impulse, decisions=len(logs[k]),
                                           return_sum=sum(row["reward"] for row in logs[k]),
                                           trace=log_path.name, trace_sha256=_hash(log_path.read_bytes()))
        return dict(seed=seed, sign=sign, initial_sha256=initial_hash, invariant_scene_sha256=scene_invariant,
                    original_scene_sha256=original_sha256,
                    scene_sha256=env.scene.xml_sha256, goal_xy=list(env.scene.goal_xy), goal_yaw_rad=env.scene.goal_yaw,
                    renderer=visual.device_report, backend_start=before, backend_end=after, lanes=condition_results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--source-dir", required=True, type=Path, help="original, digest-matched training examples")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--device-index", type=int)
    parser.add_argument("--mode", choices=("cpu", "musa"), default="musa")
    parser.add_argument("--seeds", nargs="+", type=int, default=[52001])
    parser.add_argument("--arm-dofs", type=int, nargs=2, choices=(6, 7), default=[6, 7])
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 50000:
        parser.error("supply unique supplemental seeds >=50000 (separate from all registered data)")
    if args.arm_dofs[0] == args.arm_dofs[1]:
        parser.error("select one six-DOF and one seven-DOF arm")
    if args.out.exists() and any(args.out.iterdir()):
        parser.error("audit requires a fresh output directory")
    contract, policy, actor, env, camera = load_frozen(args.policy, args.source_dir)
    args.out.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    report = dict(schema="bimanual-frozen-policy-audit-v1", task=policy["task"], mode=args.mode,
                  policy_variant=policy.get("variant", "centroid"),
                  **{key: contract.get(key, default) for key, default in MODEL_DEFAULTS.items()},
                  task_scoring="original frozen environment, supplemental targets only", arm_dofs=args.arm_dofs,
                  frozen_policy_file_sha256=_hash(args.policy.read_bytes()), contract_sha256=policy["contract_sha256"],
                  source_sha256=contract["sources"], auditor_sha256=_hash(Path(__file__).read_bytes()),
                  side_order=SIDES, air_definition="input clearance >= 3mm and floor Fz < 0.05 mg",
                  shared_loading_definition="each pad normal force >= 1 N", seeds=args.seeds,
                  runs=[], paired_target_response=[])
    for seed in args.seeds:
        previous = None
        for sign in (-1, 1):
            result = _run_goal(args, contract, policy, actor, env, camera, seed, sign)
            if previous is not None and any(previous[key] != result[key] for key in
                                            ("original_scene_sha256", "initial_sha256", "invariant_scene_sha256")):
                raise ValueError("target intervention changed robot, physical scene or initial state")
            if previous is not None:
                command_y = result["goal_xy"][1] - previous["goal_xy"][1]
                command_yaw = result["goal_yaw_rad"] - previous["goal_yaw_rad"]
                if policy["task"] == "carry" and command_y <= 2*.025:
                    raise ValueError("lateral targets do not separate the frozen tolerance regions")
                if policy["task"] == "push_t" and command_yaw <= 2*math.radians(6):
                    raise ValueError("yaw targets do not separate the frozen tolerance regions")
                outcomes = {}
                for condition in CONDITIONS:
                    low, high = previous["lanes"][condition], result["lanes"][condition]
                    yaw_change = high["payload_yaw_rad"] - low["payload_yaw_rad"]
                    outcomes[condition] = dict(success_minus=low["success"], success_plus=high["success"],
                                               final_y_change_m=high["payload_xyz_m"][1] - low["payload_xyz_m"][1],
                                               final_yaw_change_rad=math.atan2(math.sin(yaw_change), math.cos(yaw_change)))
                report["paired_target_response"].append(dict(seed=seed, command_y_change_m=command_y,
                                                            command_yaw_change_rad=command_yaw, lanes=outcomes))
            previous = result
            report["runs"].append(result)
            (args.out / "audit.json").write_bytes(_canonical(report))
            print(f"{seed} target {sign:+d}: " + ", ".join(f"{name}={int(row['success'])}" for name, row in result["lanes"].items()), flush=True)


if __name__ == "__main__":
    main()
