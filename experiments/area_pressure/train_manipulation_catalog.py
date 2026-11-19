"""Train one goal-conditioned CUDA policy on the eight original manipulation scenes."""

import argparse
import json
from contextlib import ExitStack

import numpy as np

from phymeo.backends.mujoco_heterogeneous import (
    MuJoCoTyxHeterogeneousBatch,
    MuJoCoTyxModelSpec,
    MuJoCoTyxWorldSpec,
)
from phymeo.builders.mujoco_compiler import compile_mujoco_xml
from phymeo.training.goal_language import FrozenMiniLM
from phymeo.training.manipulation_env import HeterogeneousManipulationEnv
from phymeo.training.manipulation_tasks import manipulation_scenes
from phymeo.training.mujoco_tyx_env import MuJoCoTyxActionSpec

WORLD_KEYS = (41, 7, 305, 11, 203, 59, 91, 2)
STEPS = 32


def make_env(stack, encoder, language="zh", *, min_horizon=1000):
    if type(min_horizon) is not int or min_horizon < 1:
        raise ValueError("min_horizon must be a positive step count")
    scenes = dict(zip(WORLD_KEYS, manipulation_scenes(), strict=True))
    models = tuple(
        MuJoCoTyxModelSpec(scene.key, compile_mujoco_xml(scene.mjcf, require_asset=False))
        for scene in scenes.values()
    )
    worlds = tuple(MuJoCoTyxWorldSpec(key, scene.key) for key, scene in scenes.items())
    batch = stack.enter_context(MuJoCoTyxHeterogeneousBatch(models, worlds, mode="cuda"))
    specs = {}
    for key in scenes:
        width = batch.layout[key].nu
        specs[key] = MuJoCoTyxActionSpec(
            tuple(range(width)),
            scale=np.full(width, 0.1, np.float32),
            bias=np.full(width, 0.025, np.float32),
        )
    env = stack.enter_context(
        HeterogeneousManipulationEnv(
            batch,
            scenes,
            action_specs=specs,
            goals={key: scene.goals[scene.default_goal] for key, scene in scenes.items()},
            horizons={key: min_horizon + row for row, key in enumerate(scenes)},
            instruction_encoder=encoder,
            instruction_language=language,
            native_contact_features=True,
        )
    )
    return env, scenes


def train_epoch(session, scenes):
    from phymeo.training.rsl_mujoco_tyx import optimization_evidence

    rows = []
    for key, scene in scenes.items():
        layout = session.env.batch.layout[key]
        goal = scene.goals[scene.default_goal]
        rows.append(
            dict(
                world=key,
                scene=scene.key,
                target=goal.target_key,
                task=goal.kind,
                nv=layout.nv,
                nu=layout.nu,
                accepted=0,
                timeouts=0,
                successes=0,
                reward=0.0,
                support_steps=0,
                peak_support_n=0.0,
                peak_robot_normal_n=0.0,
            )
        )
    for _ in range(STEPS):
        transition = session.collect_step()
        # Summarize the accepted entry, including the last one before each reset.
        roles = transition.final_observation["native_object_contact"].tolist()
        rewards = transition.reward.flatten().tolist()
        timeouts = transition.truncated.flatten().tolist()
        terminated = transition.terminated.flatten().tolist()
        for i, row in enumerate(rows):
            row["accepted"] += int(transition.accepted_worlds[row["world"]])
            row["timeouts"] += int(timeouts[i])
            row["successes"] += int(terminated[i])
            row["reward"] += rewards[i]
            support = roles[i][2][4]
            row["support_steps"] += int(support > 0)
            row["peak_support_n"] = max(row["peak_support_n"], support)
            robot = sum(roles[i][j][4] for j in (0, 1, 3))
            row["peak_robot_normal_n"] = max(row["peak_robot_normal_n"], robot)
    losses = session.update()
    return dict(
        completed_updates=session.completed_updates,
        worlds=rows,
        losses=losses,
        backend="mujoco-tyx-cuda",
        contact_schema="native-object-contact-v1",
        model_buckets=len(session.env.workspace.binding["source"]["buckets"]),
        horizons=session.env.binding["horizons"],
        optimizer=optimization_evidence(session.algorithm),
    )


def main():
    import torch

    from phymeo.training.rsl_manipulation import ManipulationPpo

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_directory")
    parser.add_argument("--language", choices=("zh", "en"), default="zh")
    parser.add_argument("--updates", type=int, default=1)
    parser.add_argument(
        "--min-horizon",
        type=int,
        default=1000,
        help="Steps before timeout in the first world, increased by one per world",
    )
    parser.add_argument("--save-to")
    parser.add_argument("--resume-from")
    args = parser.parse_args()
    if args.updates < 1:
        parser.error("--updates must be positive")
    torch.manual_seed(720)
    with ExitStack() as stack:
        env, scenes = make_env(
            stack, FrozenMiniLM(args.model_directory), args.language, min_horizon=args.min_horizon
        )
        session = ManipulationPpo(env, steps=STEPS)
        if args.resume_from:
            session.restore_checkpoint(
                args.resume_from, expected_revision=env.batch.checkpoint_revision
            )
        for _ in range(args.updates):
            print(json.dumps(train_epoch(session, scenes), allow_nan=False))
        if args.save_to:
            session.save_checkpoint(args.save_to)


if __name__ == "__main__":
    main()
