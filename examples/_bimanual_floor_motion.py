"""Index measured S4000 episodes and sample them for a two-task training floor."""

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from xml.etree import ElementTree

import numpy as np


STAGES = ("training-candidate", "training-candidate-replay", "frozen-evaluation")


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _quaternions(xml):
    """Read generalized-coordinate order from the compiled scene's joint tree."""
    at, rotations = 0, []
    for element in ElementTree.parse(xml).getroot().find("worldbody").iter():
        if element.tag not in ("joint", "freejoint"):
            continue
        kind = "free" if element.tag == "freejoint" else element.get("type", "hinge")
        if kind in ("free", "ball"):
            rotations.append(at+3 if kind == "free" else at)
        at += {"free": 7, "ball": 4, "hinge": 1, "slide": 1}[kind]
    return at, tuple(rotations)


@dataclass(eq=False)
class Clip:
    xml: Path
    trajectory: Path
    receipt: Path
    task: str
    stage: str
    seed: int
    job: str
    success: bool
    times: np.ndarray
    qpos: np.ndarray
    rotations: tuple
    motion_id: str
    provenance: dict

    @property
    def duration(self):
        return float(self.times[-1]-self.times[0])

    def bracket(self, seconds):
        """Bracketing sample indices and fraction for a time relative to the episode start."""
        t = float(seconds)
        if not np.isfinite(t):
            raise ValueError("playback time must be finite")
        t = np.clip(t+self.times[0], self.times[0], self.times[-1])
        i = min(len(self.times)-2, max(0, np.searchsorted(self.times, t, side="right")-1))
        u = (t-self.times[i])/(self.times[i+1]-self.times[i])
        return int(i), int(i+1), float(u)

    def sample(self, seconds):
        """Interpolate within this episode. Never blend across an episode reset."""
        i, j, u = self.bracket(seconds)
        a, b = self.qpos[i], self.qpos[j]
        state = a+u*(b-a)
        for start in self.rotations:
            region = slice(start, start+4)
            q, r = a[region], b[region]
            dot = float(q@r)
            if dot < 0:
                r, dot = -r, -dot
            if dot > .9995:
                quat = q+u*(r-q)
            else:
                theta = np.arccos(np.clip(dot, 0., 1.))
                quat = (np.sin((1-u)*theta)*q+np.sin(u*theta)*r)/np.sin(theta)
            state[region] = quat/np.linalg.norm(quat)
        return state


def _load(entry, root):
    xml, trajectory, receipt = ((root/entry[k]).resolve() for k in ("xml", "trajectory", "receipt"))
    record = json.loads(receipt.read_bytes())
    model_sha, trajectory_sha, receipt_sha = (_digest(p) for p in (xml, trajectory, receipt))
    if model_sha != record["scene_sha256"] or trajectory_sha != record.get("trajectory_sha256", trajectory_sha):
        raise ValueError(f"episode digest mismatch: {receipt}")
    for key, actual in (("model_sha256", model_sha), ("trajectory_sha256", trajectory_sha), ("receipt_sha256", receipt_sha)):
        if key in entry and entry[key] != actual:
            raise ValueError(f"index digest mismatch for {key}: {receipt}")
    device, before = record.get("device_after") or {}, record.get("device_before") or {}
    completed = device.get("completed_kernels", 0)
    if (record.get("physics") != "musa" or "S4000" not in device.get("device_name", "").upper()
            or completed <= before.get("completed_kernels", 0) or completed != device.get("launched_kernels")):
        raise ValueError(f"missing completed S4000 kernel evidence: {receipt}")
    task, stage = record["task"], entry["stage"]
    if task not in ("carry", "push_t") or stage not in STAGES:
        raise ValueError(f"unknown floor task or source stage: {receipt}")
    job = record["job"]
    if job.get("arms") != "both" or job.get("black", False):
        raise ValueError(f"expected a two-arm full-vision episode: {receipt}")
    provenance = dict(model_sha256=model_sha, trajectory_sha256=trajectory_sha, receipt_sha256=receipt_sha,
                      receipt_binds_trajectory_hash="trajectory_sha256" in record,
                      device=device["device_name"], completed_kernels_in_batch=completed-before["completed_kernels"])
    if stage.startswith("training-"):
        historic = (root/entry.get("training_receipt", entry["receipt"])).resolve()
        origin = json.loads(historic.read_bytes())
        if (not re.fullmatch(r"train-\d+", historic.parent.name)
                or not re.fullmatch(r"candidate-\d+", job["label"])
                or origin["job"] != job or origin["task"] != task or origin["seed"] != record["seed"]):
            raise ValueError(f"candidate does not match its training receipt: {historic}")
        if (stage == "training-candidate-replay") == (historic == receipt):
            raise ValueError(f"candidate capture/replay provenance conflicts: {historic}")
        provenance.update(generation=int(historic.parent.name.split("-")[1]),
                          training_receipt=str(historic), training_receipt_sha256=_digest(historic))
        checkpoint = historic.parents[2]/"checkpoint.json"
        if checkpoint.is_file():
            update = next((u for u in json.loads(checkpoint.read_bytes())["updates"]
                           if u["generation"] == provenance["generation"]), None)
            if update is not None and origin["seed"] in update["seeds"]:
                provenance["completed_generation_checkpoint_sha256"] = _digest(checkpoint)
    elif re.fullmatch(r"candidate-\d+", job["label"]):
        raise ValueError(f"candidate supplied as frozen evaluation: {receipt}")
    nq, rotations = _quaternions(xml)
    with np.load(trajectory, allow_pickle=False) as stored:
        if str(stored["model_sha256"].item()) != model_sha:
            raise ValueError(f"trajectory-XML mismatch: {trajectory}")
        q = np.asarray(stored["qpos"], dtype=np.float64)
        times = np.asarray(stored["time_s"], dtype=np.float64)
        ticks, rewards = stored["ticks"], stored["rewards"]
    count = int(record["decisions"])+1
    if (count < 2 or q.shape != (count, nq) or times.shape != (count,) or ticks.shape != times.shape
            or np.any(np.diff(times) <= 0) or np.any(np.diff(ticks) <= 0)
            or ticks[-1] != record["physics_ticks"] or not np.array_equal(rewards, record["rewards"])
            or not np.isfinite(q).all() or not np.isfinite(times).all()):
        raise ValueError(f"invalid episode samples or decision receipts: {trajectory}")
    for index in rotations:
        lengths = np.linalg.norm(q[:, index:index+4], axis=1)
        if np.any(np.abs(lengths-1.) > 1e-3):
            raise ValueError(f"nonunit recorded orientations: {trajectory}")
    # Identify repackaged copies independently of ZIP compression and filenames.
    digest = hashlib.sha256(model_sha.encode("ascii"))
    digest.update(times.astype("<f8", copy=False).tobytes())
    digest.update(q.astype("<f8", copy=False).tobytes())
    return Clip(xml, trajectory, receipt, task, stage, int(record["seed"]), job["label"],
                bool(record["success"]), times, q, rotations, digest.hexdigest(), provenance)


def load_clips(manifest_path):
    path = Path(manifest_path).resolve()
    manifest = json.loads(path.read_bytes())
    clips = [_load(e, path.parent) for e in manifest["clips"]]
    if not clips:
        raise ValueError("empty floor motion manifest")
    return clips


@dataclass(frozen=True)
class Station:
    row: int
    col: int
    clip: Clip
    phase_s: float
    playback_rate: float = 1.
    sampled: bool = False

    def source_time(self, seconds):
        return (seconds*self.playback_rate+self.phase_s) % self.clip.duration

    def sample_index(self, seconds):
        i, j, u = self.clip.bracket(self.source_time(seconds))
        return i if u < .5 else j

    def sample(self, seconds):
        if self.sampled:
            return self.clip.qpos[self.sample_index(seconds)]
        return self.clip.sample(self.source_time(seconds))


def build_stations(clips, rows=16, columns=16, *, playback_rate=1., sampled=False):
    if rows < 1 or columns < 2 or columns % 2:
        raise ValueError("the two task areas require a positive row count and even columns")
    if not np.isfinite(playback_rate) or playback_rate <= 0:
        raise ValueError("playback rate must be finite and positive")
    lanes = rows*(columns//2)
    assignments = []
    for task in ("carry", "push_t"):
        candidates = sorted((c for c in clips if c.task == task),
                            key=lambda c: (STAGES.index(c.stage), c.motion_id))
        unique = {}
        for clip in candidates:
            unique.setdefault(clip.motion_id, clip)
        if not unique:
            raise ValueError(f"no S4000 episodes for {task}")
        pool = list(unique.values())[:lanes]
        assignments.append(pool)
    result = []
    for row in range(rows):
        for col in range(columns):
            half = columns//2
            lane = row*half+col%half
            pool = assignments[int(col >= half)]
            clip = pool[lane%len(pool)]
            # Deterministically stagger both the first and subsequent uses of
            # a capture. Offsets change playback phase only, never the sample.
            phase = ((lane*.6180339887498949) % 1.)*clip.duration
            result.append(Station(row, col, clip, phase, float(playback_rate), bool(sampled)))
    return result


def floor_manifest(stations):
    counts = Counter(s.clip.motion_id for s in stations)
    used = {s.clip.motion_id: s.clip for s in stations}
    return dict(stations=len(stations), unique_episodes=len(used),
                repeats_beyond_first=sum(counts.values())-len(used),
                unique_by_stage=dict(Counter(c.stage for c in used.values())),
                stations_by_task=dict(Counter(s.clip.task for s in stations)),
                playback="Source time is (video time * playback rate + phase) modulo recorded duration. "
                         "Nearest recorded state or linear positions and shortest-arc unit quaternion interpolation.",
                sources=[dict(motion_id=k, task=c.task, stage=c.stage, seed=c.seed, job=c.job, success=c.success,
                              duration_s=c.duration, stations=counts[k], xml=str(c.xml), trajectory=str(c.trajectory),
                              receipt=str(c.receipt), **c.provenance) for k,c in used.items()],
                layout=[dict(row=s.row, col=s.col, motion_id=s.clip.motion_id, phase_s=s.phase_s,
                             playback_rate=s.playback_rate, nearest_recorded_state=s.sampled) for s in stations])


def _index(args):
    entries = []
    for stage, roots in (("training-candidate", args.training), ("frozen-evaluation", args.frozen)):
        for root in roots:
            for trajectory in sorted(root.rglob("*.npz")):
                receipt, xml = trajectory.with_suffix(".json"), trajectory.with_suffix(".xml")
                if not (receipt.is_file() and xml.is_file()):
                    continue
                rec = json.loads(receipt.read_bytes())
                job = rec.get("job", {})
                if job.get("arms") != "both" or job.get("black", False):
                    continue
                if stage == "training-candidate" and not re.fullmatch(r"train-\d+", receipt.parent.name):
                    continue
                if stage == "frozen-evaluation" and job.get("label") not in ("frozen", "trained"):
                    continue
                paths = dict(xml=xml, trajectory=trajectory, receipt=receipt)
                entry = {k: os.path.relpath(p.resolve(), args.out.parent.resolve()) for k,p in paths.items()}
                entry["stage"] = stage
                clip = _load(entry, args.out.parent)
                entry.update({k: clip.provenance[k] for k in ("model_sha256","trajectory_sha256","receipt_sha256")})
                entries.append(entry)
    if not entries:
        raise ValueError("no complete XML, NPZ and receipt triples found")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(schema="s4000-bimanual-floor-input-v1", clips=entries), indent=2),
                        encoding="utf-8")


def _capture_train(argv):
    """Use the existing CEM selection/update exactly as written; retain its candidate histories."""
    import bimanual_musa_train as trainer

    original = trainer.Runner
    class RecordingRunner(original):
        def __init__(self, args):
            if args.stage != "train" or args.mode != "musa" or args.arms != "both":
                raise ValueError("capture-train needs --stage train --mode musa --arms both")
            super().__init__(args)

        def episodes(self, stage, seed, jobs, *, keep_trajectory=False):
            capture = stage.startswith("train-")
            if capture:
                for job in jobs:
                    path = self.args.out/"episodes"/stage/f"{seed}-{job['label']}.json"
                    if path.is_file() and not all(path.with_suffix(s).is_file() for s in (".xml", ".npz")):
                        raise ValueError(f"training receipt exists without a capture; use a new output run: {path}")
            return super().episodes(stage, seed, jobs, keep_trajectory=keep_trajectory or capture)

    trainer.Runner = RecordingRunner
    try:
        trainer.main(argv)
    finally:
        trainer.Runner = original


if __name__ == "__main__":
    if sys.argv[1:2] == ["capture-train"]:
        _capture_train(sys.argv[2:])
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        sub = parser.add_subparsers(dest="command", required=True)
        index = sub.add_parser("index", help="index candidate captures and optional frozen fallback directories")
        index.add_argument("--training", action="append", type=Path, default=[])
        index.add_argument("--frozen", action="append", type=Path, default=[])
        index.add_argument("--out", type=Path, required=True)
        _index(parser.parse_args())
