"""Generate paired 2 x 2 free-pin reset banks for the state-only reach task."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from _panda_grasp_insert_scene import build_scene
from _panda_pickup_resets import initial_state


# seed, absolute world yaw (degrees), absolute fork cross offset (m), fork along offset (m)
SPLITS = {"crossed": (271130, 26., .001, -.009), "validation": (271131, 22., .0008, .007),
          "train-extra": (271132, 30., .0012, -.007)}
ARM_HALFWIDTH_RAD = .025


def measured_pose(mj, model, data, q):
    """Recover the pin displacement from the compiled sites at an actual reset."""
    data.qpos[:] = q
    mj.mj_kinematics(model, data)
    pin = model.site("part_grasp_center").id
    fork = model.site("pickup_grasp_center").id
    fork_rotation = data.site_xmat[fork].reshape(3, 3)
    offset = fork_rotation.T @ (data.site_xpos[pin] - data.site_xpos[fork])
    relative_world = data.site_xmat[pin].reshape(3, 3) @ fork_rotation.T
    yaw = math.degrees(math.atan2(relative_world[1, 0], relative_world[0, 0]))
    if not np.allclose(relative_world[:, 2], (0., 0., 1.), atol=2e-6, rtol=0):
        raise ValueError("Measured pin orientation has a tilt relative to the pickup frame")
    return dict(cross_m=float(offset[0]), along_m=float(offset[1]), fork_z_m=float(offset[2]),
                yaw_deg=yaw, world_pin_xyz_m=data.site_xpos[pin].tolist())


def generate_bank(scene, model, split, out):
    """Publish one bank and return the same saved arrays that passed the physical pose checks."""
    import mujoco as mj

    data = mj.MjData(model)
    if (model.nq, model.nv, model.nu) != (23, 22, 16):
        raise ValueError("Expected the original dual-Panda, free-pin model")
    arm_names = [name for arm in scene.arm_joint_names for name in arm]
    for adr, name in enumerate(arm_names):
        joint = model.joint(name).id
        if model.jnt_type[joint] != mj.mjtJoint.mjJNT_HINGE or model.jnt_qposadr[joint] != adr:
            raise ValueError(f"Expected Panda arm hinge {name} at qpos {adr}")

    seed, yaw, cross, along = SPLITS[split]
    rng = np.random.Generator(np.random.PCG64(seed))
    independent = split == "train-extra"
    offset_count = 12 if independent else 3
    offsets = rng.uniform(-ARM_HALFWIDTH_RAD, ARM_HALFWIDTH_RAD, size=(offset_count, 14))
    home = np.asarray(scene.initial_qpos)
    rows, positions, velocities = [], [], []
    for yaw_deg in (-yaw, yaw):
        for cross_m in (-cross, cross):
            case = dict(yaw_deg=yaw_deg, cross_m=cross_m, along_m=along)
            state = initial_state(scene, case)
            cell = len(rows) // 3
            for replicate in range(3):
                perturbation = offsets[len(rows) if independent else replicate]
                q = np.array(state.q, dtype=np.float64, copy=True)
                q[:14] += perturbation
                positions.append(q)
                velocities.append(np.array(state.v, dtype=np.float64, copy=True))
                rows.append(dict(world=len(rows), cell=cell, replicate=replicate, **case))
    qpos, qvel = np.stack(positions), np.stack(velocities)
    if qpos.shape != (12, 23) or qvel.shape != (12, 22) or np.any(qvel != 0.):
        raise ValueError("Expected twelve zero-velocity dual-Panda resets")
    if len(np.unique(qpos[:, :14], axis=0)) != offset_count or len(np.unique(qpos[:, 16:23], axis=0)) != 4:
        raise ValueError(f"Expected {offset_count} distinct arm poses and four distinct free-pin poses")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    npz = out / "initial-states.npz"
    np.savez_compressed(npz, qpos=qpos, qvel=qvel, model_sha256=scene.xml_sha256)
    with np.load(npz, allow_pickle=False) as saved:
        if str(saved["model_sha256"].item()) != scene.xml_sha256:
            raise ValueError("Reset bank model binding differs from the compiled scene")
        qpos, qvel = saved["qpos"], saved["qvel"]
    for row, q in zip(rows, qpos, strict=True):
        offset_index = row["world"] if independent else row["replicate"]
        np.testing.assert_array_equal(q[:14], home[:14] + offsets[offset_index])
        np.testing.assert_array_equal(q[14:16], home[14:16])
        # The resident bank stores FP32. Check the actual saved state after
        # that conversion, independently of the transform used to build it.
        measured = measured_pose(mj, model, data, q.astype(np.float32))
        np.testing.assert_allclose([measured[k] for k in ("cross_m", "along_m", "fork_z_m")],
                                   [row["cross_m"], row["along_m"], 0.], atol=2e-7, rtol=0.)
        np.testing.assert_allclose(measured["yaw_deg"], row["yaw_deg"], atol=2e-5, rtol=0.)
        row["measured_from_saved_fp32_qpos"] = measured
    digest = hashlib.sha256(qpos.tobytes() + qvel.tobytes()).hexdigest()
    metadata = dict(schema="dual-panda-crossed-reach-resets-v1", split=split,
                    model_sha256=scene.xml_sha256, physical_reset_sha256=digest,
                    file=npz.name, file_sha256=hashlib.sha256(npz.read_bytes()).hexdigest(),
                    seed=seed, generator="numpy.random.PCG64", cells=4, replicates_per_cell=3,
                    axes=dict(cross_m="pickup fork local X", along_m="pickup fork local Y",
                              yaw_deg="right-handed rotation about world +Z from nominal pin orientation"),
                    levels=dict(yaw_deg=[-yaw, yaw], cross_m=[-cross, cross], along_m=along),
                    arm_joint_order=arm_names, arm_uniform_halfwidth_rad=ARM_HALFWIDTH_RAD)
    metadata["independent_arm_offsets_rad" if independent else "paired_arm_offsets_rad"] = offsets.tolist()
    metadata["worlds"] = rows
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"{split}: {npz}, physical_reset_sha256={digest}", flush=True)
    return qpos, qvel, metadata


def run(split, out, urdf=None):
    scene = build_scene(urdf_path=urdf)
    _, _, metadata = generate_bank(scene, scene.compile().mj_model, split, out)
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--out", type=Path, required=True, help="new output directory for this bank")
    parser.add_argument("--urdf", type=Path)
    args = parser.parse_args()
    run(args.split, args.out, args.urdf)
