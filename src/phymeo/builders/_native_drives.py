"""URDF scalar drives and affine linkage lowering for the native solver profile."""

from collections import deque
from dataclasses import replace

import numpy as np

from ._mimic_xml import extract_mimic_relations


def prepare_scalar_drives(mj, spec, asset, input_type, root):
    if spec.actuators:
        raise ValueError(f"{asset.name}: scalar_pd cannot replace or duplicate source actuators")
    if root.tag not in {"robot", "mujoco"}:
        raise ValueError("scalar_pd requires URDF or MJCF")
    native = root.tag == "mujoco"
    if native and spec.compiler.degree:
        raise NotImplementedError("MJCF scalar_pd requires compiler angle=radian")
    if native and spec.keys:
        raise NotImplementedError("MJCF scalar_pd keyframes need an explicit initial-state policy")
    xml = {j.get("name"): j for j in root.findall("joint")}
    actual = {j.name: j for j in spec.joints}
    supplied = {j.name: j for j in asset.joints}
    if set(supplied) - set(actual):
        raise ValueError(
            f"{asset.name}: unknown joint inputs {sorted(set(supplied) - set(actual))}"
        )
    relations = ()
    if any(j.find("mimic") is not None for j in xml.values()):
        relations = extract_mimic_relations(tuple(xml), {n: (None, n, j) for n, j in xml.items()})
    followers = {row[1]: row for row in relations}
    for driver, follower, ratio, offset in relations:
        if driver not in actual or follower not in actual or driver == follower:
            raise ValueError("mimic requires distinct known scalar joints")
        if actual[driver].type != actual[follower].type or ratio == 0:
            raise ValueError("mimic requires same-type scalar joints and nonzero multiplier")
    resolved = {name: (name, 1.0, 0.0) for name in actual if name not in followers}
    outgoing = {name: [] for name in actual}
    for row in relations:
        outgoing[row[0]].append(row)
    queue, ordered = deque(resolved), []
    while queue:
        driver = queue.popleft()
        leader, gain, bias = resolved[driver]
        for row in outgoing[driver]:
            _, follower, ratio, offset = row
            value = (ratio * gain, ratio * bias + offset)
            with np.errstate(over="ignore", under="ignore"):
                fp32 = np.asarray((*value, ratio, offset), np.float32)
            if not np.isfinite(fp32).all() or fp32[0] == 0 or fp32[2] == 0:
                raise ValueError("mimic affine map is not representable in FP32")
            resolved[follower] = (leader, *value)
            ordered.append(row)
            queue.append(follower)
    if len(resolved) != len(actual):
        raise ValueError("mimic cycle")
    rows, diagnostics = {}, []
    defaults = dict(asset.joint_defaults)
    for name, joint in actual.items():
        if joint.type not in (mj.mjtJoint.mjJNT_HINGE, mj.mjtJoint.mjJNT_SLIDE):
            raise NotImplementedError("scalar_pd admits scalar hinge/slide joints only")
        row = supplied.get(name, input_type(name))
        limit = None if native else xml[name].find("limit")
        rating = 0.0 if limit is None else float(limit.get("effort", "0"))
        if not np.isfinite(rating) or rating < 0:
            raise ValueError("invalid asset motor rating")
        if name in followers and any(value is not None for value in (row.target_position, row.target_velocity)):
            raise ValueError("mimic follower cannot own an independent target command")
        servo = name not in followers or asset.prepare_follower_drives or any(
            value not in (None, 0) for value in (row.kp, row.kd)
        )
        if name in followers and not servo and row.effort_limit is not None:
            if row.effort_limit != rating:
                raise ValueError(
                    "mimic follower has no actuator; effort limit override unsupported"
                )
            diagnostics.append(
                f"{asset.name}::{name}: mimic effort rating {rating} restated; no actuator"
            )
        if row.lower_limit is not None or row.upper_limit is not None:
            # MjSpec stores AUTO as 2; unlike the compiled bool, it is also
            # truthy for a zero/default range that has no active constraint.
            limited = int(joint.limited) == 1 or (
                int(joint.limited) == 2 and joint.range[0] < joint.range[1]
            )
            if not limited and (row.lower_limit is None or row.upper_limit is None):
                raise ValueError("an unlimited joint requires both finite position limits")
            lower = float(joint.range[0]) if row.lower_limit is None else row.lower_limit
            upper = float(joint.range[1]) if row.upper_limit is None else row.upper_limit
            if lower >= upper:
                raise ValueError("native position limits require lower < upper")
            joint.range[:] = [lower, upper]
            joint.limited = True
        velocity_rating = (
            (
                None
                if limit is None or "velocity" not in limit.attrib
                else float(limit.attrib["velocity"])
            )
            if row.velocity_limit is None
            else row.velocity_limit
        )
        row = replace(row, velocity_limit=velocity_rating)
        if row.velocity_limit is not None:
            diagnostics.append(
                f"{asset.name}::{name}: velocity_limit={row.velocity_limit} is declared metadata; native solver does not clamp velocity"
            )
        dynamics = None if native else xml[name].find("dynamics")
        if dynamics is not None:
            extra = sorted(set(dynamics.attrib) - {"damping", "friction"})
            if extra:
                diagnostics.append(
                    f"{asset.name}::{name}: native MuJoCo 3.8 ignores nonstandard URDF dynamics attributes {','.join(extra)}"
                )
        friction = (
            joint.frictionloss if native or (dynamics is not None and "friction" in dynamics.attrib) else 1e-5
        )
        if native and np.any(np.asarray(joint.springdamper) != 0):
            raise NotImplementedError("MJCF springdamper requires an explicit parameter contract")
        if native and np.any(np.asarray(joint.damping)[1:] != 0):
            raise NotImplementedError("MJCF nonlinear damping requires an explicit parameter contract")
        damping = float(joint.damping[0])
        row = replace(
            row,
            kp=row.kp if row.kp is not None else (0.0 if name in followers else defaults.get("kp", 500.0)),
            kd=row.kd if row.kd is not None else (0.0 if name in followers else defaults.get("kd", 50.0)),
            armature=row.armature if row.armature is not None else defaults.get("armature", float(joint.armature) if native else 0.01),
            damping=row.damping if row.damping is not None else defaults.get("damping", damping),
            frictionloss=row.frictionloss
            if row.frictionloss is not None
            else defaults.get("frictionloss", friction),
        )
        joint.armature, joint.frictionloss = row.armature, row.frictionloss
        joint.damping[:] = [row.damping, 0.0, 0.0]
        if servo:
            cap = rating if row.effort_limit is None else row.effort_limit
            if not np.isfinite(cap) or cap < 0:
                raise ValueError("invalid asset motor rating")
            actuator = spec.add_actuator(name=f"__drive_{name}")
            actuator.trntype, actuator.target = mj.mjtTrn.mjTRN_JOINT, name
            actuator.gaintype, actuator.biastype = mj.mjtGain.mjGAIN_FIXED, mj.mjtBias.mjBIAS_AFFINE
            actuator.gainprm[0] = 1.0
            actuator.biasprm[:3] = [0.0, -row.kp, -row.kd]
            actuator.forcelimited = bool(cap)
            if cap:
                actuator.forcerange = [-cap, cap]
        rows[name] = row
    for driver, follower, ratio, offset in ordered:
        d, f = rows[driver], rows[follower]
        q = ratio * (actual[driver].ref if d.position is None else d.position) + offset
        v = ratio * (0.0 if d.velocity is None else d.velocity)
        if (f.position is not None and not np.isclose(f.position, q, atol=1e-12, rtol=0)) or (
            f.velocity is not None and not np.isclose(f.velocity, v, atol=1e-12, rtol=0)
        ):
            raise ValueError("conflicting initial mimic position/velocity")
        rows[follower] = replace(f, position=q, velocity=v)
        equality = spec.add_equality(name=f"__mimic_{follower}")
        equality.type = mj.mjtEq.mjEQ_JOINT
        equality.name1, equality.name2 = follower, driver
        equality.data[:5] = [
            offset + ratio * actual[driver].ref - actual[follower].ref,
            ratio,
            0.0,
            0.0,
            0.0,
        ]
    return tuple(rows.values()), resolved, tuple(diagnostics)
