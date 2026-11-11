"""Public ROB1c example: a real R/P arm pushes an eccentric free box.

No private banks, prescribed motion, test imports or HOST integrator. The
resident transaction samples one common entry, advances each existing storage
once and publishes the world once. Current contact sampling is HOST finite
patches; mechanics runs on the selected TYX backend.
"""

import argparse
import gzip
import json
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from itertools import product
from pathlib import Path

import numpy as np

import phymeo as ph


def encode(value):
    if isinstance(value, np.ndarray):
        return dict(
            dtype=value.dtype.str,
            shape=value.shape,
            bytes_hex=value.tobytes(order="C").hex(),
            values=value.tolist(),
        )
    if isinstance(value, np.generic):
        return encode(value.item())
    if isinstance(value, bytes):
        return dict(bytes_hex=value.hex())
    if type(value) is object:
        return dict(opaque_process_identity=True)
    if is_dataclass(value):
        return {item.name: encode(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Mapping):
        return dict(mapping_entries=[(encode(k), encode(v)) for k, v in value.items()])
    if isinstance(value, (list, tuple)):
        return [encode(x) for x in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError("unsupported evidence value: " + type(value).__name__)


def build():
    world = 503
    arm_key, box_key = ph.ResidentParticipantKey(world, 7), ph.ResidentParticipantKey(world, 29)
    q, v = np.array([1 / 16, 1 / 32]), np.array([1 / 32, -1 / 64])
    model = ph.RobotModel(
        np.array([-1, 0]),
        ("revolute", "prismatic"),
        np.array([[0, 0, 0], [0, 1 / 4, 0]], float),
        np.tile([0.0, 0, 0, 1], (2, 1)),
        np.array([[0, 0, 1], [1, 0, 0]], float),
        np.array([2.0, 1.0]),
        np.array([[1 / 8, 0, 0], [1 / 16, 1 / 32, 0]]),
        np.full((2, 3), 1 / 8),
        np.zeros(3),
    )
    program = ph.DrivenRobotProgram(
        model, kp=np.full(2, 4.0), kd=np.full(2, 0.5), effort_limit=np.full(2, 64.0)
    )
    arm = ph.FixedRobotTree(program, program.initial_state(ph.RobotState(q, v)))
    arm_owner = ph.ResidentParticipant(arm_key, arm, (700, 701))
    angle = q[0]
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    sphere_center = R @ np.array([q[1] + 1 / 8, 3 / 8, 0])
    box_origin = sphere_center + R @ np.array([7 / 32, -1 / 16, 0])
    box_model = ph.FloatingRobotModel(
        ph.RigidBodyModel(np.array([1.0]), np.full((1, 3), 1 / 8), np.zeros(3)),
        np.zeros(3),
    )
    box_q = np.r_[box_origin, [0, 0, np.sin(angle / 2), np.cos(angle / 2)]]
    box_v = np.array([-1 / 64, 1 / 32, 0, 0, 0, 1 / 128])
    box = ph.FloatingRobotTree(
        box_model,
        ph.FloatingRobotState(box_q, box_v),
        kp=np.zeros(0),
        kd=np.zeros(0),
        effort_limit=np.zeros(0),
    )
    box_owner = ph.ResidentParticipant(box_key, box, (2900,))
    first, second = ph.CollisionKey(world, 701, 10), ph.CollisionKey(world, 2900, 10)
    local_sphere = np.array([[1 / 8, 1 / 8, 0]])
    local_box = np.array(list(product((-1, 1), repeat=3))) * [1 / 8, 3 / 16, 1 / 8]
    local_box += [0, 1 / 16, 0]
    shapes = ph.CollisionModel(
        (
            ph.CollisionShape(first, "sphere", (31,), 1 / 8, revision=1),
            ph.CollisionShape(second, "hull", tuple(range(31, 39)), revision=1),
        )
    )
    bindings = (
        ph.ResidentParticipantBinding(
            arm_key,
            ph.ResidentTreeBinding(
                first, world, 1, local_sphere, topology=arm_owner.compiled.topology
            ),
        ),
        ph.ResidentParticipantBinding(
            box_key,
            ph.ResidentFloatingTreeBinding(
                second, world, 0, local_box, topology=box_owner.compiled.topology
            ),
        ),
    )
    control = {
        arm_key: program.inputs(
            q + 1 / 64,
            np.array([1 / 64, -1 / 128]),
            feedforward=np.array([1 / 32, 1 / 16]),
            loads=ph.RobotInputs.zeros(2),
        ),
        box_key: box.inputs(
            np.zeros((0, 3)),
            ph.FloatingRobotInputs(np.zeros(0), np.array([[0, 1 / 32, 0]]), np.zeros((1, 3))),
        ),
    }
    inputs = dict(
        world=world,
        arm_model=model,
        arm_q=q,
        arm_v=v,
        box_model=box_model,
        box_q=box_q,
        box_v=box_v,
        local_sphere=local_sphere,
        local_box=local_box,
        controls=control,
        h=1 / 1024,
    )
    return (arm_owner, box_owner), shapes, bindings, control, (first, second), inputs


def run(mode, steps, friction, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    participants, model, bindings, controls, pair, inputs = build()
    law = ph.RobotContactMaterial(64.0, 0.0, friction, 1 / 16)
    rows, failure = [], None
    scene = ph.ResidentSceneBatch(participants, mode=mode, cache_root=output / "cache")
    try:
        with scene:
            scene.set_controls(controls)
            with scene.prepare_contacts(
                model,
                bindings,
                {pair: law},
                candidate_capacity=1,
                site_capacity=128,
                narrow_phase="host_patch",
            ) as plan:
                initial = scene.snapshot()
                try:
                    for tick in range(steps):
                        entry = scene.snapshot()
                        accepted = scene.step_contacts(inputs["h"], plan)
                        after = scene.snapshot()
                        receipt = plan.last_accepted.get(inputs["world"])
                        rows.append(
                            dict(
                                entry=entry,
                                accepted=after,
                                result=accepted,
                                receipt=receipt,
                                status=plan.last_status,
                                scene_stats=scene.stats(),
                                plan_stats=plan.stats(),
                            )
                        )
                        if accepted != {inputs["world"]: True} or receipt is None:
                            raise RuntimeError("world rejected; complete saved entry retained")
                        if after[inputs["world"]].tick != tick + 1:
                            raise RuntimeError("world tick is not continuous")
                        if len(receipt.participants) != 2 or len(receipt.sites) != 1:
                            raise RuntimeError("both participant responses and one site required")
                        site = receipt.sites[0]
                        np.testing.assert_allclose(
                            site.first_force_n + site.second_force_n,
                            np.zeros(3),
                            atol=5e-6,
                            rtol=5e-5,
                        )
                        for key, item in receipt.participants.items():
                            if item.entry_state != entry[key.world].participants[key]:
                                raise RuntimeError("receipt lost the actual common entry")
                            if item.accepted_state != after[key.world].participants[key]:
                                raise RuntimeError("force and motion are not jointly published")
                            if item.diagnostic.equation_residual > 2e-5:
                                raise RuntimeError("physical mass equation residual exceeded")
                            if item.diagnostic.control_residual > 5e-5:
                                raise RuntimeError("bounded actuator residual exceeded")
                        if tick == 0:
                            if (
                                np.linalg.norm(
                                    receipt.participants[participants[0].key].law_wrench_fp64[1, 3:]
                                )
                                <= 0
                            ):
                                raise RuntimeError("arm contact failed to generate a COM moment")
                            if (
                                np.linalg.norm(
                                    receipt.participants[participants[1].key].law_wrench_fp64[0, 3:]
                                )
                                <= 0
                            ):
                                raise RuntimeError(
                                    "free box failed to receive the opposite COM moment"
                                )
                except BaseException as error:
                    failure = dict(type=type(error).__name__, message=str(error))
                    raise
                finally:
                    result = dict(
                        inputs=inputs,
                        mode=mode,
                        law=law,
                        requested_steps=steps,
                        completed_steps=sum(
                            row["result"].get(inputs["world"], False) for row in rows
                        ),
                        initial=initial,
                        final=scene.snapshot(),
                        rows=rows,
                        failure=failure,
                        stats=scene.stats(),
                    )
                    with gzip.open(output / "trajectory.json.gz", "wt", encoding="utf-8") as stream:
                        json.dump(encode(result), stream, allow_nan=False, separators=(",", ":"))
    finally:
        scene.close()
        final_stats = scene.stats()
        (output / "closed.json").write_text(
            json.dumps(encode(final_stats), allow_nan=False), encoding="utf-8"
        )
    if final_stats["device_owned_bytes"] != 0:
        raise RuntimeError("owned device bytes remain after close")
    return dict(
        status="PASS",
        mode=mode,
        steps=steps,
        h=inputs["h"],
        friction=friction,
        output=str(output.resolve()),
        closed_owned_bytes=0,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("interpret", "cuda"), default="cuda")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--friction", type=float, choices=(0.0, 0.25), default=0.25)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not 1 <= args.steps <= 40:
        parser.error("--steps must be within [1,40]")
    print(json.dumps(run(args.mode, args.steps, args.friction, args.output), allow_nan=False))


if __name__ == "__main__":
    main()
