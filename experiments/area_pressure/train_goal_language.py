"""Train two identical CUDA paddles with different instructions and the same visible targets."""

import argparse
import json
from contextlib import ExitStack
from dataclasses import replace

from experiments.area_pressure.train_gripper import _gripper, _paddle
from phymeo.backends.mujoco_heterogeneous import MuJoCoTyxHeterogeneousBatch, MuJoCoTyxWorldSpec
from phymeo.backends.mujoco_heterogeneous_area_scene import MuJoCoTyxHeterogeneousAreaPressureScene
from phymeo.training.area_pressure_success import AreaPressureRoleBinding
from phymeo.training.goal_language import FrozenMiniLM
from phymeo.training.manipulation_env import HeterogeneousManipulationEnv
from phymeo.training.manipulation_tasks import Goal, Target
from phymeo.training.mujoco_tyx_env import MuJoCoTyxActionSpec


def make_env(stack, encoder, language="zh"):
    grip, _, surfaces, task = _gripper(0.6)
    model, assets, push = _paddle(grip, surfaces, task)
    near = Goal(
        "pusht",
        "object",
        "near",
        ("把方块推到近处的蓝区", "将方块移入较近的蓝色区域"),
        ("Push the box into the near blue zone", "Move the block to the closer blue area"),
    )
    far = Goal(
        "pusht",
        "object",
        "far",
        ("把方块推到远处的橙区", "将方块移入较远的橙色区域"),
        ("Push the box into the far orange zone", "Move the block to the more distant orange area"),
    )
    half_xy = push.object_shapes[0].size[:2]
    task = replace(
        push,
        goals=(near, far),
        targets=(
            Target("near", "近处的蓝区", "near blue zone", (0.008, 0, 0), 0, half_xy),
            Target("far", "远处的橙区", "far orange zone", (0.018, 0, 0), 0, half_xy),
        ),
    )
    keys = (41, 7)
    worlds = tuple(MuJoCoTyxWorldSpec(k, model.key) for k in keys)
    batch = stack.enter_context(MuJoCoTyxHeterogeneousBatch((model,), worlds, mode="cuda"))
    scene = MuJoCoTyxHeterogeneousAreaPressureScene(batch, {model.key: assets})
    roles = tuple(
        AreaPressureRoleBinding(a.identity, side, role)
        for a, side, role in zip(assets, ("B", "A"), ("support", "other"), strict=True)
    )
    return stack.enter_context(
        HeterogeneousManipulationEnv(
            batch,
            dict.fromkeys(keys, task),
            goals=dict(zip(keys, (near, far), strict=True)),
            horizons={41: 7, 7: 10},
            action_specs=dict.fromkeys(
                keys, MuJoCoTyxActionSpec((0,), scale=(0.08,), bias=(0.04,))
            ),
            area_scene=scene,
            area_roles={model.key: roles},
            instruction_encoder=encoder,
            instruction_language=language,
        )
    )


def main():
    import torch

    from phymeo.training.rsl_manipulation import ManipulationPpo
    from phymeo.training.rsl_mujoco_tyx import optimization_evidence

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_directory", help="Local pinned ONNX model and tokenizer directory")
    parser.add_argument("--language", choices=("zh", "en"), default="zh")
    args = parser.parse_args()
    torch.manual_seed(720)
    with ExitStack() as stack:
        env = make_env(stack, FrozenMiniLM(args.model_directory), args.language)
        session = ManipulationPpo(env, steps=12)
        for _ in range(12):
            session.collect_step()
        losses = session.update()
        print(
            json.dumps(
                dict(
                    worlds=env.world_keys,
                    steps=12,
                    language=env.binding["goal_language"],
                    losses=losses,
                    optimizer=optimization_evidence(session.algorithm),
                ),
                allow_nan=False,
            )
        )


if __name__ == "__main__":
    main()
