"""One shared CUDA PPO update on two grippers or on a gripper and a rotary paddle.

Run python -m experiments.area_pressure.train_gripper to print the update.
"""

import argparse
import json
from contextlib import ExitStack
from dataclasses import replace
from xml.etree import ElementTree as ET

import numpy as np

from experiments.area_pressure.moving_jaw_grasp import JAWS, H, _declarations
from phymeo.backends.mujoco_area_pressure_scene import BodyAreaPressureAsset
from phymeo.backends.mujoco_heterogeneous import MuJoCoTyxHeterogeneousBatch, MuJoCoTyxWorldSpec
from phymeo.backends.mujoco_heterogeneous_area_scene import MuJoCoTyxHeterogeneousAreaPressureScene
from phymeo.builders.mujoco_compiler import compile_mujoco_xml
from phymeo.geometry.area_pressure import AreaPressureIdentity
from phymeo.geometry.area_pressure_intersection import RigidPressureMesh, TetraPressureMesh
from phymeo.training.area_pressure_success import AreaPressureRoleBinding
from phymeo.training.manipulation_env import HeterogeneousManipulationEnv
from phymeo.training.manipulation_tasks import Goal, Primitive, Scene, Target
from phymeo.training.mujoco_tyx_env import MuJoCoTyxActionSpec


def _gripper(mu):
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
    return model, worlds, asset, task


def _paddle(gripper, gripper_assets, grasp_task):
    # The hinge is 30 mm from the pad centre. A positive motor torque sweeps
    # the pad toward +X. The hinge precedes the FREE box in native coordinates.
    xml = ET.fromstring(grasp_task.mjcf)
    xml.set("model", "rotary-paddle")
    xml.find("option").set("gravity", "0 0 -9.81")
    bodies = xml.find("worldbody")
    bodies.remove(bodies.find("body[@name='right']"))
    paddle = bodies.find("body[@name='left']")
    bodies.remove(paddle)
    bodies.insert(0, paddle)
    paddle.set("name", "paddle")
    paddle.find("joint").attrib.update(
        name="sweep", type="hinge", pos="0 .03 0", axis="0 0 1", damping=".0072"
    )
    paddle.find("geom").set("name", "paddle-face")
    motors = xml.find("actuator")
    motors.remove(motors.find("motor[@name='right-close']"))
    motors.find("motor").attrib.update(name="paddle-drive", joint="sweep")
    source = ET.tostring(xml, encoding="unicode")
    pad, _, table = gripper_assets
    pad = replace(
        pad, identity=replace(pad.identity, rigid_body="paddle", rigid_material="paddle-face")
    )
    # The broad support indents a pressure field belonging to the table.
    # The lower box face winds toward the table (negative object-local Z).
    volume = table.pressure_local
    table_volume = TetraPressureMesh(
        volume.node_ids,
        volume.world_vertices * np.array((2.0, 2.0, 0.5)),
        volume.cells,
        volume.nodal_pressure,
        volume.cell_ids,
        pressure_domain=volume.pressure_domain,
    )
    bottom = RigidPressureMesh(
        ("sw", "se", "ne", "nw"),
        ((-0.02, -0.02, -0.01), (0.02, -0.02, -0.01), (0.02, 0.02, -0.01), (-0.02, 0.02, -0.01)),
        ((0, 2, 1), (0, 3, 2)),
        ("bottom-0", "bottom-1"),
    )
    support = BodyAreaPressureAsset(
        AreaPressureIdentity("table", "object", "table-volume", "box-bottom"),
        table_volume,
        bottom,
    )
    assets = (support, pad)
    model = replace(
        gripper,
        key="rotary-paddle",
        compilation=compile_mujoco_xml(source, require_asset=False),
        area_pressure_pairs=tuple(
            (a.identity.compliant_body, a.identity.rigid_body) for a in assets
        ),
        area_friction_laws=tuple(
            replace(gripper.area_friction_laws[0], identity=a.identity) for a in assets
        ),
    )
    goal = Goal(
        "pusht",
        "object",
        "forward",
        ("把方块向前推", "将方块推到前方"),
        ("Push the box forward", "Move the box ahead"),
    )
    task = replace(
        grasp_task,
        key="paddle-push",
        mjcf=source,
        robot_kinds="R",
        robot_joint_names=("sweep",),
        actuator_names=("paddle-drive",),
        finger_geom_names=(),
        goals=(goal,),
        targets=(Target("forward", "前方", "forward", (0.02, 0, 0), 0),),
    )
    return model, assets, task


def make_env(stack, *, mu=0.6, action_specs=None, mixed_tools=False):
    model, worlds, asset, task = _gripper(mu)
    models = [model]
    tasks = dict.fromkeys((w.world for w in worlds), task)
    surfaces = {model.key: asset}
    roles = {model.key: ("finger0", "finger1", "support")}
    horizons = {41: 19, 7: 100}
    if mixed_tools:
        paddle, paddle_assets, push_task = _paddle(model, asset, task)
        models.append(paddle)
        worlds = (worlds[0], MuJoCoTyxWorldSpec(7, paddle.key))
        tasks[7] = push_task
        surfaces[paddle.key] = paddle_assets
        roles[paddle.key] = ("support", "other")
        horizons[7] = 13
        if action_specs is None:
            action_specs = training_actions(mixed_tools=True)
    batch = stack.enter_context(MuJoCoTyxHeterogeneousBatch(tuple(models), worlds, mode="cuda"))
    scene = MuJoCoTyxHeterogeneousAreaPressureScene(batch, surfaces)
    keys = batch.world_keys
    loads = {41: 0.22, 7: 0.11}
    env = stack.enter_context(
        HeterogeneousManipulationEnv(
            batch,
            tasks,
            goals={k: tasks[k].goals[tasks[k].default_goal] for k in keys},
            action_specs=action_specs
            if action_specs is not None
            else {
                k: MuJoCoTyxActionSpec(
                    (0, 1), (2,), scale=(1.0, 1.0, 0.0), bias=(0.0, 0.0, loads[k])
                )
                for k in keys
            },
            horizons=horizons,
            area_scene=scene,
            area_roles={
                key: tuple(
                    AreaPressureRoleBinding(
                        a.identity, "A" if a.identity.compliant_body == "object" else "B", role
                    )
                    for a, role in zip(rows, roles[key], strict=True)
                )
                for key, rows in surfaces.items()
            },
        )
    )
    return env


def training_actions(*, mixed_tools=False):
    # Both grippers can open or close around a mild inward actuator bias. World
    # 41 also permits a small horizontal disturbance. Z loads are constant.
    return {
        41: MuJoCoTyxActionSpec(
            (0, 1), (2, 0), scale=(1.0, 1.0, 0.0, 0.01), bias=(0.48, 0.48, 0.22, 0.0)
        ),
        7: MuJoCoTyxActionSpec((0,), scale=(0.08,), bias=(0.04,))
        if mixed_tools
        else MuJoCoTyxActionSpec((0, 1), (2,), scale=(1.0, 1.0, 0.0), bias=(0.42, 0.42, 0.11)),
    }


def main():
    import torch

    from phymeo.training.rsl_manipulation import ManipulationPpo
    from phymeo.training.rsl_mujoco_tyx import optimization_evidence

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mixed-tools",
        action="store_true",
        help="Train the parallel gripper and rotary pusher together",
    )
    mixed = parser.parse_args().mixed_tools
    torch.manual_seed(720)
    steps = 24
    with ExitStack() as stack:
        env = make_env(stack, action_specs=training_actions(mixed_tools=mixed), mixed_tools=mixed)
        session = ManipulationPpo(env, steps=steps)
        for _ in range(steps):
            session.collect_step()
        losses = session.update()
        print(
            json.dumps(
                dict(
                    worlds=env.world_keys,
                    models=[env.batch.layout[k].model_key for k in env.world_keys],
                    steps=steps,
                    losses=losses,
                    optimizer=optimization_evidence(session.algorithm),
                ),
                allow_nan=False,
            )
        )


if __name__ == "__main__":
    main()
