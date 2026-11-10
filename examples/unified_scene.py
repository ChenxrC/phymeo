"""One CPU-compiled source, explicit Phymeo or MuJoCo execution.

The default is a force-driven slider, not a robot controller. Phymeo backends
use the lowered asset and their original physics; MuJoCo backends retain the
original MuJoCo model and contact law. This does not export arbitrary Phymeo
scene declarations into MuJoCo.
"""

import argparse
import json

import numpy as np

from phymeo.builders.mujoco_compiler import compile_mujoco, compile_mujoco_xml

SLIDER_XML = """
<mujoco model="unified_slider">
  <option gravity="0 0 0" timestep="0.001"/>
  <worldbody><body name="base"><body name="carriage">
    <joint name="slide" type="slide" axis="1 0 0" limited="true" range="-2 2"/>
    <geom name="carriage_shape" type="sphere" size="0.05" mass="0.5"/>
  </body></body></worldbody>
  <actuator><motor name="drive" joint="slide" ctrllimited="true" ctrlrange="-5 5"/></actuator>
</mujoco>
"""


def run(*, backend="host", source=None, steps=20, dt=None, force=1.0):
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")
    if (dt is not None and (not np.isfinite(dt) or dt <= 0)) or not np.isfinite(force):
        raise ValueError("dt must be positive and force finite")
    native = backend in {"mujoco-cpu", "mujoco-cuda", "mujoco-warp", "mujoco-tyx-interpret"}
    compilation = (
        compile_mujoco(source, require_asset=not native)
        if source is not None
        else compile_mujoco_xml(SLIDER_XML, require_asset=not native)
    )
    dt = float(compilation.mj_model.opt.timestep) if dt is None else dt
    if not native and compilation.unlowered_world_geom_ids:
        raise ValueError(
            "source contains world geometry; assemble it explicitly in SceneDeclaration "
            "or select a native MuJoCo backend"
        )
    keys = (4, 19)
    if native:
        from phymeo.backends.mujoco_runtime import MuJoCoControl, MuJoCoRuntime

        runtime = MuJoCoRuntime(
            compilation,
            world_keys=keys,
            mode="cpu"
            if backend == "mujoco-cpu"
            else "interpret"
            if backend == "mujoco-tyx-interpret"
            else "cuda",
            implementation="warp" if backend == "mujoco-warp" else None,
        )
        controls = {}
        for key, sign in zip(keys, (1, -1), strict=True):
            control = np.zeros(compilation.mj_model.nu)
            if len(control):
                control[0] = sign * force
            controls[key] = MuJoCoControl(control)
        coordinates = "MuJoCo qpos/qvel; native actuator control"
    else:
        from phymeo import FloatingRobotCommand, RobotCommand
        from phymeo.builders.scene_compiler import (
            ArticulationWorld,
            SceneDeclaration,
            compile_scene,
        )
        from phymeo.core.robot import RobotState
        from phymeo.dynamics.floating_robot import FloatingRobotModel, FloatingRobotState

        asset = compilation.asset
        floating = isinstance(asset.model, FloatingRobotModel)
        state_type = FloatingRobotState if floating else RobotState
        size = len(asset.joint_names)
        initial = state_type(compilation.initial_q, compilation.initial_v)
        declaration = SceneDeclaration(
            tuple(
                ArticulationWorld(
                    key,
                    asset,
                    initial,
                    kp=np.zeros(size),
                    kd=np.zeros(size),
                    effort_limit=np.full(size, 5.0),
                )
                for key in keys
            )
        )
        runtime = compile_scene(declaration, backend=backend)
        controls = {}
        for key, sign in zip(keys, (1, -1), strict=True):
            effort = np.zeros(size)
            if size:
                effort[0] = sign * force
            target = compilation.initial_q[7:] if floating else compilation.initial_q
            command = RobotCommand(target, feedforward=effort) if size else None
            controls[key] = FloatingRobotCommand(command) if floating else command
        coordinates = "Phymeo public q/v; floating root velocity at COM in world frame"
    with runtime:
        runtime.set_controls(controls)
        for _ in range(steps):
            accepted = runtime.step(dt)
            if not all(accepted.values()):
                raise RuntimeError(f"physical step refused: {dict(accepted)}")
        states = runtime.snapshot()
        return {
            "backend": backend,
            "coordinates": coordinates,
            "steps": steps,
            "worlds": {
                str(key): {
                    "q": state.q.tolist(),
                    "v": state.v.tolist(),
                    "time": state.time if native else state.time_s,
                }
                for key, state in states.items()
            },
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        default="host",
        choices=(
            "host",
            "interpret",
            "cuda",
            "mujoco-cpu",
            "mujoco-cuda",
            "mujoco-warp",
            "mujoco-tyx-interpret",
        ),
    )
    parser.add_argument("--source", help="optional MuJoCo XML/URDF file")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--dt", type=float, help="default: compiled source timestep")
    parser.add_argument("--force", type=float, default=1.0)
    print(json.dumps(run(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
