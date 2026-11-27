"""Fit and evaluate pickup from physical endpoints of the published MUSA reach policy."""
import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path

import mujoco as mj
import numpy as np

from _panda_grasp_insert_control import ARM_KD, ARM_KP, _rotation_vector
from _panda_pickup_loads import PickupLoads
from _panda_pickup_student import PickupStudent
from _panda_pickup_teacher import PickupTeacher


DT, REPEAT, DECISIONS = .002, 8, 160
PUBLISHED = Path(__file__).resolve().parent / "checkpoints/panda-reach-factors-s4000"
# Original CPU run results.json. Each digest covers the ordered rows and
# complete archive, independently of provenance strings inside supplied files.
HANDOFF_SHA256 = {
    "factor-update": "53050f202490c55ff01848b658d1d5e123f368520b3b2d00ac0db5814aef8622",
    "validation": "48dc7744b89f0fec42df1c936a94f4bd2370c5478b8233b90a83c8c2c6b68135",
}
STUDENT_SHA256 = "4c8f50d08495017c7bd52715ccbe913e64fde9bbcfffde8ad453914dae8a835a"
OBSERVATION_CONTRACT = "pickup-v1: hand-frame pin xyz and rotation vector, pin yaw xy, pin and hand height from handoff pin, finger positions, finger targets, right joint target error; state only"
KP = np.r_[ARM_KP, 1400., 1400.]
KD = np.r_[ARM_KD, 12., 12.]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf8")


@dataclass
class Handoff:
    qpos: np.ndarray
    qvel: np.ndarray
    target: np.ndarray
    bias: np.ndarray


def apply_action(model, target, action):
    a = np.asarray(action, dtype=np.float32)
    if a.shape != (16,) or not np.isfinite(a).all() or np.any(np.abs(a) > 1):
        raise ValueError("expected sixteen bounded motor actions")
    out = np.array(target, dtype=np.float32, copy=True)
    limits = model.jnt_range[:16].copy()
    limits[:14] += [.001, -.001]
    out[:14] += np.float32(.02) * a[:14]
    out[14:] = np.float32(.0175) * (a[14:] + np.float32(1.))
    return np.clip(out, limits[:, 0].astype(np.float32), limits[:, 1].astype(np.float32))


def effort(model, q, v, target, bias):
    cmd = KP * (target - q[:16]) - KD * v[:16] + bias[:16]
    lo, hi = model.actuator_ctrlrange.T.copy()
    lo[14:], hi[14:] = np.maximum(lo[14:], -20.), np.minimum(hi[14:], 20.)
    return np.clip(cmd, lo, hi)


def bind_handoffs(trace, split, scene, out):
    release = json.loads((PUBLISHED / "results.json").read_text(encoding="utf8"))
    binding = release["experiments"][split]
    scene_hash, ck, source = sha(scene), sha(PUBLISHED / "candidate.pt"), sha(trace)
    if (scene_hash != release["model_sha256"] or ck != binding["after_checkpoint_sha256"]
            or source != binding["trace_sha256"]):
        raise ValueError("trace, checkpoint or scene differs from the published paired run")
    model = mj.MjModel.from_xml_path(str(scene))
    with np.load(trace, allow_pickle=False) as f:
        q, v = f["after_qpos"][-1], f["after_qvel"][-1]
        actions = f["after_action"]
        qs, vs = f["after_substep_qpos"], f["after_substep_qvel"]
        if (actions.shape != (128, 12, 16) or qs.shape != (128, 8, 12, 23)
                or not np.array_equal(qs[-1, -1], q) or not np.array_equal(vs[-1, -1], v)):
            raise ValueError("not a complete reach endpoint")
        targets, biases = [], []
        d = mj.MjData(model)
        for world in range(len(q)):
            target = model.key("home").qpos[:16]
            for a in actions[:, world]:
                target = apply_action(model, target, a)
            targets.append(target)
            d.qpos[:], d.qvel[:] = qs[-1, -2, world], vs[-1, -2, world]
            mj.mj_forward(model, d)
            biases.append(d.qfrc_bias[:16].copy())
    with out.open("xb") as stream:
        np.savez_compressed(stream, qpos=q, qvel=v, target=targets, bias=biases,
                            source_sha256=source, weights_sha256=ck, scene_sha256=scene_hash,
                            split=split, episode_time_s=2.048)


def read_handoffs(path, model, weights_path=None, *, expected_split=None, from_verified_trace=False):
    release = json.loads((PUBLISHED / "results.json").read_text(encoding="utf8"))
    payload = Path(path).read_bytes()
    with np.load(io.BytesIO(payload), allow_pickle=False) as f:
        split = str(f["split"].item())
        if split not in HANDOFF_SHA256 or (expected_split is not None and split != expected_split):
            raise ValueError("handoff split does not match its assigned role")
        if not from_verified_trace and hashlib.sha256(payload).hexdigest() != HANDOFF_SHA256[split]:
            raise ValueError(f"{split} handoff checksum differs from the published pickup bank")
        b = release["experiments"][split]
        weights_sha = sha(weights_path or PUBLISHED / "candidate.pt")
        if (model.nq, model.nv, model.nu) != (23, 22, 16) or (
                weights_sha != b["after_checkpoint_sha256"] or str(f["weights_sha256"].item()) != weights_sha
                or str(f["source_sha256"].item()) != b["trace_sha256"]
                or str(f["scene_sha256"].item()) != release["model_sha256"]
                or float(f["episode_time_s"].item()) != 2.048):
            raise ValueError("wrong handoff source or model")
        arrays = [f[k] for k in ("qpos", "qvel", "target", "bias")]
        for a, n in zip(arrays, (23, 22, 16, 16), strict=True):
            if a.shape != (12, n) or not np.isfinite(a).all():
                raise ValueError("invalid handoff arrays")
        return [Handoff(*rows) for rows in zip(*arrays, strict=True)]


class PickupFeatures:
    def __init__(self, model, initial_qpos):
        self.model = model
        self.hand, self.pin = [model.site(k).id for k in ("right_grasp_center", "part_grasp_center")]
        d = mj.MjData(model)
        d.qpos[:] = initial_qpos
        mj.mj_kinematics(model, d)
        self.initial_z = d.site_xpos[self.pin, 2]

    def observe(self, data, target):
        h, p = data.site_xpos[[self.hand, self.pin]]
        rh, rp = data.site_xmat[[self.hand, self.pin]].reshape(2, 3, 3)
        return np.r_[rh.T @ (p-h), _rotation_vector(mj, rh.T @ rp), rp[:2, 0],
                     p[2]-self.initial_z, h[2]-self.initial_z, data.qpos[14:16], target[14:16],
                     target[7:14]-data.qpos[7:14]]


def rollout(model, start, *, student=None, reference=None, negative=None, label=False, save_trace=False):
    data = mj.MjData(model)
    data.qpos[:], data.qvel[:] = start.qpos, start.qvel
    target, bias = start.target.copy(), start.bias.copy()
    sensor = PickupLoads(model)
    observer = PickupFeatures(model, start.qpos)
    teacher = PickupTeacher(model, start.qpos) if student is None and reference is None or label else None
    record = defaultdict(list)
    obs_rows, labels, commands, phases = [], [], [], Counter()
    for tick in range(DECISIONS):
        mj.mj_kinematics(model, data)
        obs = observer.observe(data, target)
        if teacher is not None:
            desired = teacher.action(data.qpos, data.qvel, target)
            phases[teacher.phase] += 1
            obs_rows.append(obs)
            labels.append(desired)
        if reference is not None:
            target = reference[tick].copy()
            if negative == "open_jaw":
                target[14:] = .035
            elif negative == "one_sided":
                target[15] = .035
            else:
                raise ValueError("unrecognized control branch")
        else:
            target = apply_action(model, target, desired if student is None else student.act(obs))
        commands.append(target.copy())
        for _ in range(REPEAT):
            data.ctrl[:] = effort(model, data.qpos, data.qvel, target, bias)
            # Euler integrates precisely the constraint solve whose forces we sample.
            mj.mj_forward(model, data)
            for key, value in sensor.sample(data).items():
                record[key].append(value)
            bias = data.qfrc_bias[:16].copy()
            mj.mj_Euler(model, data)
    trace = {k: np.asarray(v) for k,v in record.items()}
    score = sensor.score(trace)
    return dict(score=score, phases=dict(phases), obs=np.asarray(obs_rows), labels=np.asarray(labels),
                commands=np.asarray(commands), trace=trace if save_trace else None)


def train_student(model, train, out, report):
    obs, labels = [], []
    for i, start in enumerate(train):
        r = rollout(model, start)
        obs.append(r["obs"]); labels.append(r["labels"])
        report["teacher"].append(dict(world=i, score=r["score"], decisions_by_phase=r["phases"]))
        print("teacher", i, r["score"]["success"], r["phases"], flush=True)
    write_json(out/"training.json", report)
    if not any(x["score"]["success"] for x in report["teacher"]):
        raise RuntimeError("teacher never transferred the load on the training bank")
    best = None
    best_rank = None
    for iteration in range(2):
        student = PickupStudent.fit(np.concatenate(obs), np.concatenate(labels),
                                    observation_contract=OBSERVATION_CONTRACT)
        candidate = []
        for start in train:
            r = rollout(model, start, student=student, label=iteration == 0)
            candidate.append(r["score"])
            if iteration == 0:
                obs.append(r["obs"]); labels.append(r["labels"])
        rank = (sum(s["success"] for s in candidate),
                sum(s["longest_loaded_dwell_s"] for s in candidate))
        report["training_candidates"].append(dict(iteration=iteration, fit=student.fit_summary,
                                                  passed=rank[0], total_loaded_s=rank[1], scores=candidate))
        if best_rank is None or rank > best_rank:
            best, best_rank, chosen = student, rank, iteration
        print("student training", iteration, rank, flush=True)
    best.save(out/"student.npz")
    student = PickupStudent.load(out/"student.npz")
    report.update(selected_iteration=chosen, student_sha256=sha(out/"student.npz"))
    return student


def run(args):
    compact = args.handoff_dir is not None
    if compact and (args.training_trace is not None or args.validation_trace is not None):
        raise ValueError("choose --handoff-dir or the two source traces")
    if not compact and (args.training_trace is None or args.validation_trace is None):
        raise ValueError("supply both --training-trace and --validation-trace, or --handoff-dir")
    scene = args.scene
    release = json.loads((PUBLISHED / "results.json").read_text(encoding="utf8"))
    if sha(scene) != release["model_sha256"]:
        raise ValueError("expected the exact scene used by the reach policy")
    model = mj.MjModel.from_xml_path(str(scene))
    if model.opt.integrator != mj.mjtIntegrator.mjINT_EULER or model.opt.timestep != DT:
        raise ValueError("requires the original 2 ms MuJoCo Euler scene")
    if args.student is not None:
        payload = args.student.read_bytes()
        if hashlib.sha256(payload).hexdigest() != STUDENT_SHA256:
            raise ValueError("student checkpoint checksum differs from the published pickup weights")
        student = PickupStudent.load(io.BytesIO(payload))
        if student.observation_contract != OBSERVATION_CONTRACT:
            raise ValueError("student observation contract differs")
    bank_dir = args.handoff_dir if compact else args.out
    if not compact:
        args.out.mkdir(parents=True, exist_ok=False)
        for split, trace in (("factor-update", args.training_trace), ("validation", args.validation_trace)):
            bind_handoffs(trace, split, scene, bank_dir / (split + ".npz"))
    train = read_handoffs(bank_dir / "factor-update.npz", model, expected_split="factor-update",
                          from_verified_trace=not compact)
    validation = read_handoffs(bank_dir / "validation.npz", model, expected_split="validation",
                               from_verified_trace=not compact)
    if compact:
        args.out.mkdir(parents=True, exist_ok=False)
    report = dict(schema="dual-panda-learned-pickup-v1", backend="native MuJoCo CPU Euler",
                  predecessor_checkpoint_sha256=sha(PUBLISHED / "candidate.pt"), scene_sha256=sha(scene),
                  native_version=mj.__version__, train_bank_sha256=sha(bank_dir/"factor-update.npz"),
                  eval_bank_sha256=sha(bank_dir/"validation.npz"), dt=DT, decisions=DECISIONS, repeat=REPEAT,
                  teacher=[], training_candidates=[], evaluation=[])
    if args.student is None:
        student = train_student(model, train, args.out, report)
    else:
        report.update(mode="published student validation replay", student_sha256=STUDENT_SHA256)
    sensor = PickupLoads(model)
    for i, start in enumerate(validation):
        r = rollout(model, start, student=student, save_trace=True)
        results = dict(world=i, student=r["score"], controls={})
        for mode in ("open_jaw", "one_sided"):
            neg = rollout(model, start, reference=r["commands"], negative=mode, save_trace=True)
            comparison = sensor.compare(r["trace"], neg["trace"], control_mode=mode)
            results["controls"][mode] = dict(success=comparison["success"], gates=comparison["gates"],
                                              negative=neg["score"], shared_arm_targets=bool(
                                                  np.array_equal(r["commands"][:, :14], neg["commands"][:, :14])))
            if i == 0:
                np.savez_compressed(args.out/(mode+"-world0.npz"), **neg["trace"], targets=neg["commands"])
        if i == 0:
            np.savez_compressed(args.out/"student-world0.npz", **r["trace"], targets=r["commands"])
        report["evaluation"].append(results)
        print("heldout", i, results["student"]["success"], flush=True)
    write_json(args.out/"results.json", report)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene", type=Path, default=PUBLISHED / "scene.xml")
    inputs = p.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--training-trace", type=Path, help="original factor-update reach trace")
    inputs.add_argument("--handoff-dir", type=Path, help="compact factor-update.npz and validation.npz directory")
    p.add_argument("--validation-trace", type=Path, help="companion validation trace for --training-trace")
    p.add_argument("--student", type=Path, help="replay the published pickup student on validation worlds only")
    p.add_argument("--out", type=Path, required=True)
    run(p.parse_args())
