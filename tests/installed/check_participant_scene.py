"""Standalone installed asset-to-participant consumer (R01, sparse W2).

Original 2/1 kg roots, 1/8 m spheres, 64 N/m contact and h=1/1024.
No checkout examples/tests are imported. The independent spring/mass recurrence
checks both current-entry response and uninterrupted motion, then real replay.
"""

import argparse
import base64
import gzip
import hashlib
import json
import struct
import sys
import traceback
from collections.abc import Mapping
from contextlib import ExitStack
from dataclasses import fields, is_dataclass
from pathlib import Path
from unittest.mock import patch

import numpy as np

H = 1 / 1024
WORLDS = (91, 503)
PIDS = (7, 29)
MASS = np.array([2., 1.])


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def near(actual, expected, *, atol=5e-6):
    np.testing.assert_allclose(actual, expected, rtol=5e-5, atol=atol)


def encode(value):
    if isinstance(value, np.ndarray):
        require(not value.dtype.hasobject, "numeric evidence required")
        return dict(dtype=value.dtype.str, shape=value.shape,
                    bytes_b64=base64.b64encode(value.tobytes(order="C")).decode())
    if isinstance(value, np.generic):
        return encode(value.item())
    if is_dataclass(value):
        return dict(type=type(value).__module__ + "." + type(value).__name__,
                    fields={item.name: encode(getattr(value, item.name)) for item in fields(value)})
    if isinstance(value, Mapping):
        return dict(mapping_entries=[(encode(k), encode(v)) for k, v in value.items()])
    if isinstance(value, (tuple, list)):
        return [encode(item) for item in value]
    if isinstance(value, bytes):
        return dict(bytes_b64=base64.b64encode(value).decode())
    if type(value) is float:
        return dict(float64_bits=struct.pack(">d", value).hex())
    if type(value) is object:
        return dict(opaque_process_identity=True)
    if value is None or type(value) in (str, bool, int):
        return value
    raise TypeError("unhandled evidence " + type(value).__name__)


class Journal:
    def __init__(self, path):
        self.path, self.stream = path, gzip.open(path, "xb")
        self.sha, self.size, self.count = hashlib.sha256(), 0, 0

    def emit(self, event, **payload):
        data = (json.dumps(encode(dict(event=event, ordinal=self.count, **payload)),
                           separators=(",", ":"), allow_nan=False) + "\n").encode()
        self.stream.write(data)
        self.stream.flush()
        self.sha.update(data)
        self.size += len(data)
        self.count += 1

    def close(self):
        self.stream.close()
        with self.path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        return dict(events=self.count, logical_bytes=self.size, logical_sha256=self.sha.hexdigest(),
                    compressed_bytes=self.path.stat().st_size, compressed_sha256=digest)


def provenance(root, *, preflight=False):
    def checkout(path):
        return ((path / "pyproject.toml").is_file() and (path / "src/phymeo").is_dir()) or (
            path.name.lower() == "src" and (path / "phymeo").is_dir() and (path.parent / "pyproject.toml").is_file())
    require((root / "phymeo/__init__.py").is_file(), "installed site root required")
    require(not any(checkout(Path(value or Path.cwd()).resolve()) for value in sys.path), "checkout in sys.path")
    if preflight:
        require(not any(n == "phymeo" or n.startswith("phymeo.") for n in sys.modules), "early product import")
        for path in (Path.cwd().resolve(), Path(__file__).resolve().parent, root):
            require(not any(checkout(p) for p in (path, *path.parents)), "consumer/cwd must be outside checkout")
    import phymeo
    require(Path(phymeo.__file__).resolve().is_relative_to(root), "wrong package origin")
    result = {}
    for name, module in sorted(sys.modules.items()):
        if (name == "phymeo" or name.startswith("phymeo.")) and getattr(module, "__file__", None):
            path = Path(module.__file__).resolve(strict=True)
            require(path.is_relative_to(root), "wrong module origin: " + name)
            result[name] = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return result


def declaration():
    import phymeo as ph
    from phymeo.builders._robot_asset import LinkCollision
    from phymeo.builders.floating_urdf import FloatingRobotAsset
    from phymeo.builders.participant_scene_compiler import ParticipantArticulation, ParticipantSceneDeclaration
    rows, laws = [], {}
    for world in WORLDS[::-1]:
        keys = []
        for index, (pid, sign) in enumerate(zip(PIDS, (-1, 1), strict=True)):
            source = ph.CollisionKey(0, 42, 71)
            shape = ph.CollisionShape(source, "sphere", (41,), radius=1 / 8, revision=3)
            collider = LinkCollision("original-sphere", "root", -1, (0., 0., 0.), (0., 0., 0., 1.),
                                     shape, ((0., 0., 0.),), True)
            model = ph.FloatingRobotModel(ph.RigidBodyModel(np.array([1 / MASS[index]]),
                                            np.full((1, 3), 1 / 8), np.zeros(3)), np.zeros(3))
            asset = FloatingRobotAsset(f"original-{pid}", "root", model, (), (), (), (), (), (),
                                       (collider,), ph.CollisionModel((shape,)))
            key = ph.ResidentParticipantKey(world, pid)
            installed = ph.CollisionKey(world, 100 * pid, 11)
            keys.append(installed)
            rows.append(ParticipantArticulation(key, asset,
                ph.FloatingRobotState(np.array([sign * 7 / 64, 0., 0., 0., 0., 0., 1.]),
                                      np.array([-sign / 64, 0., 0., 0., 0., 0.])),
                kp=np.zeros(0), kd=np.zeros(0), effort_limit=np.zeros(0),
                body_ids=(100 * pid,), collider_keys={source: installed}))
        laws[tuple(sorted(keys))] = ph.RobotContactMaterial(64, 0, 0)
    return ParticipantSceneDeclaration(tuple(rows), materials_by_pair=laws, candidate_capacity=2)


def command(sign):
    import phymeo as ph
    return ph.FloatingRobotCommand(None, ph.FloatingRobotInputs(np.zeros(0),
                                      np.array([[0., sign / 32, 0.]]), np.zeros((1, 3))))


def recurrence(position, velocity, external):
    delta = position[1] - position[0]
    length = np.linalg.norm(delta)
    require(length > 0, "separated original centers")
    normal = delta / length
    load = 64 * max(0., 1 / 4 - length)
    forces = np.array([-load * normal, load * normal])
    acceleration = (forces + external) / MASS[:, None]
    v = velocity + H * acceleration
    return position + H * v, v, forces, acceleration


def step(scene, sign, oracle, journal):
    import phymeo as ph
    controls = {key: command(sign) for key in scene.participant_keys}
    scene.set_controls(controls)
    before = scene.snapshot()
    external = np.tile([0., sign / 32, 0.], (2, 1))
    require(scene.step(H) == dict.fromkeys(WORLDS, True), "all physical worlds accepted")
    after = scene.snapshot()
    expected = {}
    for world in WORLDS:
        keys = tuple(ph.ResidentParticipantKey(world, pid) for pid in PIDS)
        entry = before[world]
        positions = np.array([entry.participants[k].position[:3] for k in keys])
        velocities = np.array([entry.participants[k].velocity[:3] for k in keys])
        q, v, force, acceleration = recurrence(positions, velocities, external)
        whole_q, whole_v, _, _ = recurrence(*oracle[world], external)
        receipt = scene.last_accepted[world]
        require(receipt.entry_tick == entry.tick and receipt.exit_tick == after[world].tick == entry.tick + 1, "tick binding")
        require(receipt.dt_s == H and receipt.state_epoch == scene.state_epoch, "clock/source binding")
        require(len(receipt.sites) == 1 and not receipt.static_reactions, "one genuine sphere contact")
        near(receipt.sites[0].first_force_n, force[0])
        near(receipt.sites[0].second_force_n, force[1])
        for i, key in enumerate(keys):
            point, item = after[world].participants[key], receipt.participants[key]
            require(item.entry_state == entry.participants[key] and item.accepted_state == point, "receipt/state binding")
            near(point.position[:3], q[i])
            near(point.velocity[:3], v[i])
            near(point.position[:3], whole_q[i], atol=2e-4)
            near(point.velocity[:3], whole_v[i], atol=2e-4)
            near(point.position[3:], [0, 0, 0, 1])
            near(point.velocity[3:], np.zeros(3))
            near(item.entry_mass, np.diag([MASS[i]] * 3 + [1 / 8] * 3), atol=2e-5)
            near(item.entry_bias, np.zeros(6))
            near(item.entry_linear, np.eye(6)[:3][None])
            near(item.entry_angular, np.eye(6)[3:][None])
            near(item.input_external, np.r_[external[i], np.zeros(3)])
            near(item.law_wrench_fp64, np.r_[force[i], np.zeros(3)][None])
            near(item.uploaded_wrench_fp32, item.law_wrench_fp64)
            near(item.contact_generalized, np.r_[force[i], np.zeros(3)])
            near(item.diagnostic.acceleration, np.r_[acceleration[i], np.zeros(3)], atol=2e-5)
        expected[world] = (whole_q, whole_v)
    journal.emit("step", requested_controls=controls, entry=before, accepted=after,
                 receipts=scene.last_accepted, status=scene.last_status, stats=scene.stats(),
                 independent_continuous=expected)
    return expected


def run(mode, journal):
    import phymeo as ph
    from phymeo.builders.participant_scene_compiler import compile_participant_scene
    from phymeo.runtime.scene_stepper import SceneStepper
    from phymeo.solvers import floating_tree
    source = declaration()
    journal.emit("source", declaration=source, original_case="ROB1 R01", h=H,
                 tolerances=dict(relative=5e-5, same_entry=5e-6, continuous=2e-4))
    forbidden_calls = []
    def forbidden(*args, **kwargs):
        forbidden_calls.append("CPU integration")
        raise AssertionError("installed resident consumer cannot use a CPU integrator")
    initial = (np.array([[-7 / 64, 0., 0.], [7 / 64, 0., 0.]]),
               np.array([[1 / 64, 0., 0.], [-1 / 64, 0., 0.]]))
    oracle = {w: tuple(x.copy() for x in initial) for w in WORLDS}
    completed = 0
    with ExitStack() as stack:
        stack.enter_context(patch.object(SceneStepper, "step", forbidden))
        stack.enter_context(patch.object(floating_tree, "step_floating_robot", forbidden))
        stack.enter_context(patch.object(floating_tree, "step_free_root", forbidden))
        scene = stack.enter_context(compile_participant_scene(source, mode=mode))
        require(scene.world_keys == WORLDS, "canonical sparse logical worlds")
        for row in source.articulations:
            require(scene.source_map[row.key] == row.collider_keys, "original asset installation binding")
        journal.emit("compiled", identity=scene.identity, source_map=scene.source_map,
                     initial=scene.snapshot(), stats=scene.stats())
        def advance(sign, current):
            nonlocal completed
            try:
                result = step(scene, sign, current, journal)
                completed += 1
                return result
            except BaseException:
                journal.emit("first-failure", requested_force_sign=sign, attempted_ordinal=completed,
                             states=scene.snapshot(), receipts=scene.last_accepted,
                             status=scene.last_status, stats=scene.stats(), traceback=traceback.format_exc())
                raise
        for _ in range(4):
            oracle = advance(1, oracle)
        checkpoint = scene.snapshot()
        saved_oracle = {w: tuple(x.copy() for x in values) for w, values in oracle.items()}
        for _ in range(3):
            oracle = advance(-1, oracle)
        branch_a = scene.snapshot()
        epoch = scene.state_epoch
        scene.restore(checkpoint, expected_epoch=epoch)
        require(scene.snapshot() == checkpoint, "complete world restore includes held controls/history")
        oracle = saved_oracle
        for _ in range(3):
            oracle = advance(-1, oracle)
        require(scene.snapshot() == branch_a, "exact replay from accepted complete worlds")
        journal.emit("restore-replay", checkpoint=checkpoint, branch_a=branch_a,
                     branch_b=scene.snapshot(), expected_epoch=epoch, actual_epoch=scene.state_epoch)
        before_reset = scene.snapshot()
        scene.reset((WORLDS[0],))
        require(scene.snapshot()[WORLDS[0]].tick == 0, "selected complete world reset")
        require(scene.snapshot()[WORLDS[1]] == before_reset[WORLDS[1]], "unselected complete world retained")
        oracle[WORLDS[0]] = tuple(x.copy() for x in initial)
        journal.emit("partial-reset", before=before_reset, after=scene.snapshot())
        for _ in range(2):
            oracle = advance(1, oracle)
        require(scene.snapshot()[WORLDS[0]].tick == 2, "real post-reset successor")
        require(scene.snapshot()[WORLDS[1]].tick == 9, "unselected history continues")
        journal.emit("final", states=scene.snapshot(), receipts=scene.last_accepted,
                     status=scene.last_status, stats=scene.stats())
    require(completed == 12 and not forbidden_calls, "actual bounded call count and CPU sentinel")
    return dict(physical_batch_steps=completed, physical_world_steps=2 * completed,
                world_count=2, participant_count=4, cpu_integrator_calls=len(forbidden_calls),
                scope="asset assembly and explicit HOST contact sampling; resident mechanics", identity=source.identity)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("interpret", "cuda"), required=True)
    parser.add_argument("--expected-install-root", "--installed-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(args.expected_install_root.is_absolute() and args.output.is_absolute(), "absolute paths required")
    root, output = args.expected_install_root.resolve(strict=True), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    journal = Journal(output / "participant-scene.jsonl.gz")
    summary = dict(status="RUNNING", mode=args.mode, consumer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    errors = []
    def remember(stage, error):
        errors.append(error)
        summary.setdefault("errors", []).append(dict(stage=stage, type=type(error).__name__,
                                                      message=str(error), traceback=traceback.format_exc()))
        summary["status"] = "FAIL"
    try:
        summary["initial_provenance"] = provenance(root, preflight=True)
        summary["result"] = run(args.mode, journal)
        summary["status"] = "PASS"
    except BaseException as error:
        remember("run", error)
    finally:
        try:
            summary["final_provenance"] = provenance(root)
        except BaseException as error:
            remember("final-provenance", error)
        try:
            journal.emit("summary", summary=summary)
        except BaseException as error:
            remember("final-journal", error)
        try:
            summary["trace"] = journal.close()
        except BaseException as error:
            remember("journal-close", error)
        try:
            (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
        except BaseException as error:
            remember("summary-write", error)
    print(json.dumps(summary, allow_nan=False))
    if len(errors) == 1:
        raise errors[0]
    if errors:
        raise BaseExceptionGroup("participant consumer and finalization failures", errors)


if __name__ == "__main__":
    main()
