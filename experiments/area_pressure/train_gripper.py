"""One bounded CUDA PPO update on two grippers under different external loads.

Run python -m experiments.area_pressure.train_gripper to print the update.
"""

import json
from contextlib import ExitStack
from dataclasses import replace
from xml.etree import ElementTree as ET

from experiments.area_pressure.moving_jaw_grasp import JAWS, H, _declarations
from phymeo.backends.mujoco_area_pressure_scene import BodyAreaPressureAsset
from phymeo.backends.mujoco_heterogeneous import MuJoCoTyxHeterogeneousBatch
from phymeo.backends.mujoco_heterogeneous_area_scene import MuJoCoTyxHeterogeneousAreaPressureScene
from phymeo.builders.mujoco_compiler import compile_mujoco_xml
from phymeo.geometry.area_pressure_intersection import RigidPressureMesh
from phymeo.training.area_friction_policy import ROLES
from phymeo.training.area_pressure_success import AreaPressureRoleBinding
from phymeo.training.manipulation_env import HeterogeneousManipulationEnv
from phymeo.training.manipulation_tasks import Goal, Primitive, Scene, Target
from phymeo.training.mujoco_tyx_env import MuJoCoTyxActionSpec


def make_env(stack, *, mu=0.6, action_specs=None):
    specs, worlds, assets = _declarations()
    model = specs[0]
    xml = ET.fromstring(model.compilation.source.primary_xml)
    for body, geom in (("object", "object-0"), ("left", "left-pad"), ("right", "right-pad")):
        xml.find(f".//body[@name='{body}']/geom").set("name", geom)
    # The two jaws and the table each own their finite area contact surface.
    table = ET.SubElement(xml.find("worldbody"), "body", name="table", pos="0 0 -.017")
    ET.SubElement(table, "geom", name="support", type="box", size=".04 .04 .005")
    jaws = assets[model.key]
    identity = replace(jaws[0].identity, rigid_body="table", rigid_material="table-top")
    face = RigidPressureMesh(
        ("sw", "se", "ne", "nw"),
        ((-0.04, -0.04, 0.005), (0.04, -0.04, 0.005), (0.04, 0.04, 0.005), (-0.04, 0.04, 0.005)),
        ((0, 1, 2), (0, 2, 3)),
        ("lower", "upper"),
    )
    asset = (*jaws, BodyAreaPressureAsset(identity, jaws[0].pressure_local, face))
    source = ET.tostring(xml, encoding="unicode")
    model = replace(
        model,
        compilation=compile_mujoco_xml(source, require_asset=False),
        area_pressure_pairs=tuple(
            (a.identity.compliant_body, a.identity.rigid_body) for a in asset
        ),
        area_friction_laws=tuple(
            replace(model.area_friction_laws[0], identity=a.identity, mu=mu) for a in asset
        ),
    )
    worlds = tuple(replace(w, model_key=model.key) for w in worlds)
    goal = Goal(
        "grasp", "object", "center", ("夹起方块", "抬起方块"), ("Lift the box", "Raise the box")
    )
    task = Scene(
        "fixed-gripper",
        "",
        source,
        (Primitive("box", (0.02, 0.02, 0.01)),),
        0.30,
        (0, 0, 0),
        (Target("center", "中央", "center", (0, 0, 0), 0),),
        (goal,),
        0,
        tuple(j + "-slide" for j in JAWS),
        tuple(j + "-close" for j in JAWS),
        ("left-pad", "right-pad"),
        timestep_s=H,
        support_top_m=-0.012,
    )
    batch = stack.enter_context(MuJoCoTyxHeterogeneousBatch((model,), worlds, mode="cuda"))
    scene = MuJoCoTyxHeterogeneousAreaPressureScene(batch, {model.key: asset})
    keys = batch.world_keys
    loads = {41: 0.22, 7: 0.11}
    env = stack.enter_context(
        HeterogeneousManipulationEnv(
            batch,
            dict.fromkeys(keys, task),
            goals=dict.fromkeys(keys, goal),
            action_specs=action_specs
            if action_specs is not None
            else {
                k: MuJoCoTyxActionSpec(
                    (0, 1), (2,), scale=(1.0, 1.0, 0.0), bias=(0.0, 0.0, loads[k])
                )
                for k in keys
            },
            horizons={41: 19, 7: 100},
            area_scene=scene,
            area_roles={
                model.key: tuple(
                    AreaPressureRoleBinding(a.identity, "A", ROLES[i]) for i, a in enumerate(asset)
                )
            },
        )
    )
    return env


def training_actions():
    # Both grippers can open or close around a mild inward actuator bias. World
    # 41 also permits a small horizontal disturbance. Z loads are constant.
    return {
        41: MuJoCoTyxActionSpec(
            (0, 1), (2, 0), scale=(1.0, 1.0, 0.0, 0.01), bias=(0.48, 0.48, 0.22, 0.0)
        ),
        7: MuJoCoTyxActionSpec((0, 1), (2,), scale=(1.0, 1.0, 0.0), bias=(0.42, 0.42, 0.11)),
    }


def main():
    import torch

    from phymeo.training.rsl_manipulation import ManipulationPpo
    from phymeo.training.rsl_mujoco_tyx import optimization_evidence

    torch.manual_seed(720)
    steps = 24
    with ExitStack() as stack:
        env = make_env(stack, action_specs=training_actions())
        session = ManipulationPpo(env, steps=steps)
        for _ in range(steps):
            session.collect_step()
        losses = session.update()
        print(
            json.dumps(
                dict(
                    worlds=env.world_keys,
                    steps=steps,
                    losses=losses,
                    optimizer=optimization_evidence(session.algorithm),
                ),
                allow_nan=False,
            )
        )


if __name__ == "__main__":
    main()
