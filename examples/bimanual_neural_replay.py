"""Record a fresh S4000 episode of the frozen CEM controller plus its learned SPSA residual.

python examples/bimanual_neural_replay.py --base CEM_RUN --network SPSA_RUN \
    --source-dir CEM_SOURCES --seed 65001 --out NEW_DIR --library VULKAN_DLL

Pass NEW_DIR/65001/episode.{xml,npz,json} to bimanual_wrist_render.py as
--model, --trajectory and --episode-receipt respectively. The RGB in the NPZ
is the original pair of wrist exposures used to choose each recorded action.
"""

import argparse
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from bimanual_policy_replay import MODEL_DEFAULTS, _load, _sources


def _sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read(path):
    return json.loads(path.read_bytes())


def _network(args, base, base_sha, checkpoint_sha):
    files = {name: args.network/name for name in ("frozen-network.json", "training.json", "binding.json")}
    state = _read(files["training.json"])
    binding = _read(files["binding.json"])
    if binding != {"network_sha256": _sha(files["frozen-network.json"]), "training_sha256": _sha(files["training.json"])}:
        raise ValueError("SPSA frozen network and training journal differ from binding.json")
    protocol = state["protocol"]
    source = (args.neural_source or Path(__file__).with_name("bimanual_neural_residual.py")).resolve(strict=True)
    if _sha(source) != protocol["source_sha256"]:
        raise ValueError("provide the training-time bimanual_neural_residual.py with --neural-source")
    spec = importlib.util.spec_from_file_location("bimanual_bound_residual", source)
    residual = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(residual)
    g = protocol["generations"]
    if type(g) is not int or not 1 <= g <= 50:
        raise ValueError("invalid SPSA generation count")
    expected = dict(architecture=residual.ARCH, base_sha256=base_sha, base_checkpoint_sha256=checkpoint_sha,
                    base_contract_sha256=base["contract_sha256"], task=base["task"], source_sha256=_sha(source),
                    training=list(range(61001, 61001+2*g)), development=list(residual.DEVELOPMENT),
                    holdout=list(residual.HOLDOUT), conditions=list(residual.CONDITIONS),
                    sigma=.12, step=.07, generations=g)
    if protocol != expected or len(state["updates"]) != g:
        raise ValueError("SPSA training contract does not match the CEM model or the recorded protocol")
    net = residual.ResidualNet.load(files["frozen-network.json"], expected)
    initial = residual.ResidualNet.initial()
    head = initial.head
    for i, update in enumerate(state["updates"]):
        episodes = update["episodes"]
        if (update["generation"] != i or len(episodes) != 4
                or [r["condition"] for r in episodes] != ["plus","minus"]*2
                or [r["seed"] for r in episodes] != [61001+2*i,61001+2*i,61002+2*i,61002+2*i]
                or any(r["completed_batch_kernels"] <= 0 for r in episodes)):
            raise ValueError(f"incomplete paired device episodes in SPSA generation {i}")
        diff = [episodes[j]["return_sum"]-episodes[j+1]["return_sum"] for j in (0, 2)]
        directions = np.random.default_rng(54123+i).choice([-1.,1.], size=(2,8,4))
        head = residual._update(head, directions, diff)
        if not np.array_equal(diff, update["paired_differences"]) or not np.array_equal(head, update["head"]):
            raise ValueError(f"stored SPSA update {i} does not follow the original reward differences")
    if (not np.array_equal(net.head, head) or np.array_equal(head, initial.head)
            or not np.array_equal(net.encoder, initial.encoder) or not np.array_equal(net.bias, initial.bias)):
        raise ValueError("frozen network weights differ from the fitted residual model")
    return residual, net, protocol, {**binding, "binding_sha256": _sha(files["binding.json"])}


def _record(args, seed, residual, net, trainer, base, config, provenance, folder):
    """Observe the original rollout's call boundaries. All control and physics calls delegate unchanged."""
    environment = importlib.import_module("bimanual_musa_env")
    wrist = importlib.import_module("bimanual_wrist_render")
    states, inputs, images, latent, frozen, commands, rewards, diagnostics, counters = ([] for _ in range(9))
    seen = {}

    class RecordingEnv(environment.BimanualBatchEnv):
        def snapshot_output(self, info):
            velocity = info["qvel"] if "qvel" in info else self.states[0].v
            states.append((int(info["tick"]), float(info["time_s"]), np.array(info["qpos"], copy=True),
                           np.array(velocity, copy=True)))
            counters.append(int(self.runtime.musa_device_report()["completed_kernels"]))

        def reset(self, *a, **kw):
            packet = super().reset(*a, **kw)
            if set(packet) != {0}:
                raise ValueError("expected one complete physical environment per recorded episode")
            seen.update(scene=self.scene, model=self.model, before=self.runtime.musa_device_report())
            self.snapshot_output(packet[0][1])
            return packet

        def capture_post(self, actions, post):
            _, reward, terminated, truncated, info = post[0]
            commands.append(np.array(actions[0], copy=True))
            rewards.append(float(reward))
            measured = {k: v for k, v in info.items() if k not in ("qpos", "qvel")}
            diagnostics.append(dict(input_tick=states[-1][0], output_tick=int(info["tick"]),
                                    visual=seen["features"], measured=measured))
            self.snapshot_output(info)
            seen.update(final=measured, terminated=bool(terminated), truncated=bool(truncated),
                        after=self.runtime.musa_device_report())
            return post

        def step_torque(self, actions):
            return self.capture_post(actions, super().step_torque(actions))

        def step(self, actions):
            return self.capture_post(actions, super().step(actions))

    class RecordingPolicy(trainer.VisualPolicy):
        def act(self, obs, image, **kwargs):
            inputs.append(np.array(obs, copy=True))
            images.append(np.array(image, copy=True))
            command = super().act(obs, image, **kwargs)
            frozen.append(np.array(command, copy=True))
            seen["features"] = json.loads(trainer._bytes(self.features))
            if "wrist_force_world_n" in kwargs:
                seen.setdefault("force", []).append(np.array(kwargs["wrist_force_world_n"], copy=True))
            return command

    class RecordingNet:
        def predict(self, image, obs):
            result = net.predict(image, obs)
            latent.append(result.copy())
            return result

    # rollout imports its environment inside the function. This scoped alias
    # only records the real returned states. The policy, Jacobian conversion,
    # clipping, action dispatch, and device kernels all run in the bound source.
    with patch.object(environment, "BimanualBatchEnv", RecordingEnv):
        policy_api = SimpleNamespace(VisualPolicy=RecordingPolicy, _dofs=trainer._dofs)
        [result] = residual.rollout(policy_api, base, config, args, seed, [("learned", RecordingNet())])
    count = len(commands)
    if (count < 1 or not all(len(x) == count for x in (inputs, images, latent, frozen, rewards, diagnostics))
            or len(states) != count+1 or list(result["rewards"]) != rewards
            or result["scene_sha256"] != seen["scene"].xml_sha256 or result["ticks"] != states[-1][0]
            or not (seen["terminated"] ^ seen["truncated"])):
        raise RuntimeError("recorded channels do not describe the original completed episode")
    raw = np.stack(images)
    action = np.array(commands, dtype="<f8")
    rgb_sha = hashlib.sha256(raw.tobytes()).hexdigest()
    if (not all(rgb_sha == result[k] for k in ("sensor_sha256", "base_image_sha256", "residual_image_sha256"))
            or hashlib.sha256(action.tobytes()).hexdigest() != result["action_sha256"]
            or counters[-1]-counters[0] != result["completed_batch_kernels"]):
        raise RuntimeError("saved RGB, applied actions, or device kernel counters differ from the bound rollout")
    ticks, times, qpos, qvel = (np.asarray(x) for x in zip(*states, strict=True))
    if (ticks[0] != 0 or not np.all(np.diff(ticks)>0) or not np.all(np.diff(times)>0)
            or not np.isfinite(qpos).all() or not np.isfinite(qvel).all()):
        raise RuntimeError("invalid physical trajectory clock or state")
    xml, archive, receipt = (folder/f"episode.{ext}" for ext in ("xml", "npz", "json"))
    xml.write_bytes(seen["scene"].xml.encode())
    optional = {"wrist_force_world_n": np.array(seen["force"])} if "force" in seen else {}
    if optional and len(optional["wrist_force_world_n"]) != count:
        raise RuntimeError("incomplete force input stream")
    if all("applied_ctrl" in d["measured"] for d in diagnostics):
        optional["post_actuator_ctrl"] = np.array([d["measured"]["applied_ctrl"] for d in diagnostics])
    np.savez_compressed(archive, ticks=ticks, time_s=times, qpos=qpos, qvel=qvel, action=action,
                        rewards=np.array(rewards), model_sha256=seen["scene"].xml_sha256,
                        wrist_rgb=raw, proprio=np.array(inputs), residual_latent=np.array(latent),
                        frozen_action=np.array(frozen), completed_kernels=np.array(counters), **optional)
    final = seen["final"]
    variant = base.get("variant", "centroid")
    row = dict(schema="bimanual-spsa-replay-episode-v1", algorithm=residual.ARCH,
               condition="learned", policy_binding=provenance, seed_partition="new", task=base["task"],
               seed=seed, job=trainer._job("cem-plus-learned-spsa", base["weights"], base["arms"]),
               physics="musa", physics_ticks=int(ticks[-1]), arm_dofs=result["arm_dofs"],
               fiducials=variant=="fiducial", marker_layout_version=(
                   getattr(wrist, "MARKER_LAYOUT_VERSION", "split-floor-v1") if variant=="fiducial" else None),
               **{k: config.get(k, v) for k,v in MODEL_DEFAULTS.items()},
               action_semantics="bounded_motor_torque_fraction" if variant=="fiducial" else "bounded_joint_increment",
               success=bool(final["success"]), terminated=seen["terminated"], truncated=seen["truncated"],
               return_sum=result["return_sum"], decisions=count, rewards=rewards, final_metrics=final,
               device_before=seen["before"], device_after=seen["after"], renderer=result["renderer"],
               decision_diagnostics=diagnostics, sensor_rgb_sha256=rgb_sha, last_features=seen["features"],
               rollout=result, scene_sha256=_sha(xml), trajectory_sha256=_sha(archive))
    trainer._save(receipt, row)
    if hasattr(wrist, "_episode_result"):
        wrist._episode_result(receipt, archive, seen["model"], _sha(xml), _sha(archive), ticks, times, qpos, rewards)
    return dict(seed=seed, success=row["success"], decisions=count, physical_ticks=row["physics_ticks"],
                simulated_seconds=float(times[-1]), completed_kernels=result["completed_batch_kernels"],
                seed_directory=str(seed), xml_sha256=row["scene_sha256"], npz_sha256=row["trajectory_sha256"],
                receipt_sha256=_sha(receipt))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", type=Path, required=True, help="CEM training directory or frozen-policy.json")
    p.add_argument("--network", type=Path, required=True, help="SPSA directory with frozen-network, training and binding JSON")
    p.add_argument("--source-dir", type=Path, help="digest-matched original CEM examples")
    p.add_argument("--neural-source", type=Path, help="digest-matched original bimanual_neural_residual.py")
    p.add_argument("--seed", "--seeds", type=int, nargs="+", default=[65001])
    p.add_argument("--library", type=Path)
    p.add_argument("--cache-dir", type=Path, default=Path("artifacts/bimanual-musa/cache"))
    p.add_argument("--device-index", type=int)
    p.add_argument("--out", type=Path, required=True, help="new or empty output directory")
    args = p.parse_args(argv)
    path = (args.base/"frozen-policy.json" if args.base.is_dir() else args.base).resolve()
    base, config, updates, policy_sha, checkpoint_sha = _load(path, path.with_name("checkpoint.json"))
    if base["arms"] != "both":
        p.error("the residual experiment requires a bilateral frozen CEM model")
    residual, net, protocol, binding = _network(args, base, policy_sha, checkpoint_sha)
    historical = {s for u in updates for s in u["seeds"]} | set(config["selection_seeds"]) | set(config["holdout_seeds"])
    for partition in ("training", "development", "holdout"):
        historical.update(protocol[partition])
    if min(args.seed)<0 or len(set(args.seed))!=len(args.seed) or historical.intersection(args.seed):
        p.error("replay seeds must be distinct, nonnegative and new to both training and evaluation protocols")
    _, trainer = _sources(config["sources"], path, args.source_dir, base.get("variant","centroid"))
    args.out.mkdir(parents=True, exist_ok=True)
    if any(args.out.iterdir()):
        p.error("choose a new or empty output directory")
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    provenance = dict(**binding, frozen_cem_sha256=policy_sha, cem_checkpoint_sha256=checkpoint_sha,
                      neural_source_sha256=protocol["source_sha256"], cem_sources=config["sources"],
                      cem_contract_sha256=base["contract_sha256"], replay_source_sha256=_sha(Path(__file__)))
    report = dict(schema="bimanual-neural-replay-v1", algorithm=residual.ARCH, task=base["task"],
                  policy_binding=provenance, episodes=[])
    for seed in args.seed:
        with TemporaryDirectory(dir=args.out) as tmp:
            folder = Path(tmp)/str(seed)
            folder.mkdir()
            episode = _record(args, seed, residual, net, trainer, base, config, provenance, folder)
            folder.rename(args.out/str(seed))
        report["episodes"].append(episode)
        trainer._save(args.out/"replay.json", report)
        print(f"seed={seed} success={episode['success']} decisions={episode['decisions']} files={args.out/str(seed)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
