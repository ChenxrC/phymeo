"""Cold logical URDF frames bound to retained bodies in a native compilation."""

from dataclasses import dataclass

import numpy as np

from ..dynamics.rotation import quaternion_matrix


@dataclass(frozen=True)
class NativeJointFrame:
    instance: str
    source_name: str
    kind: str
    parent_body: str
    child_body: str
    # SI body-local joint anchors: xyz followed by xyzw, never COM-local.
    parent_anchor: tuple[float, ...]
    child_anchor: tuple[float, ...]
    axis_joint_local: tuple[float, ...] | None


@dataclass(frozen=True)
class NativeFrameMetadata:
    # (compiled body name, commit-reference COM in that body's local frame).
    bodies: tuple
    joints: tuple[NativeJointFrame, ...]
    contract_version: int = 1


def compile_urdf_frames(model, sources):
    """Opt-in URDF fixed/scalar frame export; no extra FK or simulation state."""
    import mujoco as mj

    joints = []
    for instance, root in sources:
        if root.tag == "mujoco":
            joints.extend(_mjcf_frames(model, instance))
            continue
        if root.tag != "robot":
            raise NotImplementedError("native source frame export requires URDF or MJCF")
        prefix = instance + "::"
        for source in root.findall("joint"):
            name = source.attrib["name"]
            kind = source.attrib["type"]
            kind = "revolute" if kind == "continuous" else kind
            if kind not in {"fixed", "revolute", "prismatic"}:
                raise NotImplementedError(f"native source frame type {kind!r} is not qualified")
            parent_name = prefix + source.find("parent").attrib["link"]
            child_name = prefix + source.find("child").attrib["link"]
            parent, child = model.body(parent_name), model.body(child_name)
            if int(model.body_parentid[child.id]) != parent.id:
                raise ValueError("native source frame requires retained parent/child bodies")
            position = np.zeros(3)
            axis = None
            count = int(model.body_jntnum[child.id])
            if kind == "fixed":
                if count:
                    raise ValueError("fixed source frame has native moving joints")
            else:
                native = model.joint(prefix + name)
                expected = (
                    mj.mjtJoint.mjJNT_SLIDE if kind == "prismatic" else mj.mjtJoint.mjJNT_HINGE
                )
                if (
                    count != 1
                    or int(model.jnt_bodyid[native.id]) != child.id
                    or int(model.jnt_type[native.id]) != int(expected)
                ):
                    raise ValueError("source scalar frame does not match the native joint")
                position = np.asarray(model.jnt_pos[native.id], dtype=float)
                vector = np.asarray(model.jnt_axis[native.id], dtype=float)
                norm = np.linalg.norm(vector)
                if not np.isfinite(norm) or norm <= 0:
                    raise ValueError("native joint axis must be finite and nonzero")
                axis = tuple(vector / norm)
            rotation = np.asarray(model.body_quat[child.id], dtype=float)[[1, 2, 3, 0]]
            parent_position = (
                model.body_pos[child.id] + quaternion_matrix(rotation[None])[0] @ position
            )
            parent_anchor = tuple(float(x) for x in (*parent_position, *rotation))
            child_anchor = tuple(float(x) for x in (*position, 0, 0, 0, 1))
            if not np.isfinite((*parent_anchor, *child_anchor)).all():
                raise ValueError("native joint anchors must be finite")
            joints.append(
                NativeJointFrame(
                    instance, name, kind, parent_name, child_name, parent_anchor, child_anchor, axis
                )
            )
    bodies = tuple(
        (model.body(i).name, tuple(float(x) for x in model.body_ipos[i]))
        for i in range(1, model.nbody)
    )
    return NativeFrameMetadata(bodies, tuple(joints))


def _mjcf_frames(model, instance, *, identities=None):
    """Native scalar anchors in compiled SI units, including a world parent."""
    import mujoco as mj

    prefix = instance + "::"
    for index in range(model.njnt):
        joint = model.joint(index)
        if identities is None and not joint.name.startswith(prefix):
            continue
        kind = {int(mj.mjtJoint.mjJNT_HINGE): "revolute",
                int(mj.mjtJoint.mjJNT_SLIDE): "prismatic"}.get(int(model.jnt_type[index]))
        child = int(model.jnt_bodyid[index])
        if identities is not None and int(model.jnt_type[index]) == int(mj.mjtJoint.mjJNT_FREE):
            continue  # Free roots are body-state bindings, not scalar joint frames.
        if kind is None or int(model.body_jntnum[child]) != 1:
            raise NotImplementedError("MJCF frames require one scalar joint per moving body")
        parent = int(model.body_parentid[child])
        position = np.asarray(model.jnt_pos[index], dtype=float)
        rotation = np.asarray(model.body_quat[child], dtype=float)[[1, 2, 3, 0]]
        matrix = quaternion_matrix(rotation[None])[0]
        anchor = model.body_pos[child] + matrix @ position
        axis = np.asarray(model.jnt_axis[index], dtype=float)
        reference = float(model.qpos0[model.jnt_qposadr[index]])
        if kind == "prismatic":
            anchor = anchor - matrix @ (axis * reference)
        else:
            correction, composed = np.empty(4), np.empty(4)
            mj.mju_axisAngle2Quat(correction, axis, -reference)
            mj.mju_mulQuat(composed, model.body_quat[child], correction)
            rotation = composed[[1, 2, 3, 0]]
        yield NativeJointFrame(instance, (identities['joint'][index] if identities is not None else joint.name)[len(prefix):], kind,
                               (identities['body'][parent] if identities is not None else model.body(parent).name) if parent else "world",
                               identities['body'][child] if identities is not None else model.body(child).name,
                               tuple(float(x) for x in (*anchor, *rotation)),
                               tuple(float(x) for x in (*position, 0, 0, 0, 1)),
                               tuple(float(x) for x in axis / np.linalg.norm(axis)))
