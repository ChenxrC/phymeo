"""Deterministic physical domain groups and supported pickup starts from reach handoffs."""

from collections import Counter
from dataclasses import dataclass
import hashlib
from pathlib import Path
from xml.etree import ElementTree as ET

import mujoco as mj
import numpy as np

from _panda_grasp_insert_scene import PAD_NAMES, RAIL_NAMES
from panda_pickup_from_reach import HANDOFF_SHA256, Handoff, read_handoffs


CANONICAL_XML_SHA256 = "130264683cc037ed3c2b36a01d2ec2fa1a4d8e21742fa280775cdb639ad50663"
SPLITS = {
    "train": (271141, "factor-update", (0, 1, 3, 4, 6, 7, 9, 10)),
    "select": (271142, "factor-update", (2, 5, 8, 11)),
    "blind": (271143, "validation", tuple(range(12))),
}
PART_GEOMS = ("part_handle", "part_shoulder", "assembly_pin")
MASS_RANGE = (.85, 1.15)
FRICTION_RANGE = (.80, 1.20)
HALFWIDTH = dict(arm_q_rad=.006, arm_v_rad_s=.035, finger_q_m=.00025, finger_v_m_s=.0015,
                 pin_fork_cross_m=.00065, pin_fork_along_m=.0015, pin_yaw_deg=3.)
MAX_ATTEMPTS = 192


@dataclass
class DomainBatch:
    xml: str
    model: mj.MjModel
    handoffs: list[Handoff]
    manifest: dict


def _rng(*seed):
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed)))


def _xml_domain(xml, seed, group):
    if group == 0:
        return xml, 1., 1., 1.
    rng = _rng(seed, group, 0)
    mass = float(rng.uniform(*MASS_RANGE))
    pad_mu, pin_mu = map(float, rng.uniform(*FRICTION_RANGE, size=2))
    root = ET.fromstring(xml)
    default_mu = root.find("default/geom").attrib["friction"]
    for name in (*PAD_NAMES, *PART_GEOMS):
        geom = root.find(f".//geom[@name='{name}']")
        friction = geom.get("friction", default_mu).split()
        factor = pad_mu if name in PAD_NAMES else pin_mu
        friction[0] = str(float(friction[0]) * factor)
        geom.set("friction", " ".join(friction))
        if name in PART_GEOMS:
            geom.set("mass", str(float(geom.attrib["mass"]) * mass))
    return ET.tostring(root, encoding="unicode"), mass, pad_mu, pin_mu


def _candidate(model, src, fork_r, rng):
    q, v = src.qpos.astype(float).copy(), src.qvel.astype(float).copy()
    target = src.target.astype(float).copy()
    dq = np.empty(16)
    dq[:14] = rng.uniform(-HALFWIDTH["arm_q_rad"], HALFWIDTH["arm_q_rad"], 14)
    dq[14:] = rng.uniform(-HALFWIDTH["finger_q_m"], HALFWIDTH["finger_q_m"], 2)
    q[:16] += dq
    target += dq
    limits = model.jnt_range[:16].copy()
    limits[:14] += [.001, -.001]
    # The reach interface's absolute motor command can open to 35 mm per jaw.
    limits[14:, 1] = np.minimum(limits[14:, 1], .035)
    target = np.clip(target, limits[:, 0], limits[:, 1]).astype(np.float32)
    v[:14] += rng.uniform(-HALFWIDTH["arm_v_rad_s"], HALFWIDTH["arm_v_rad_s"], 14)
    v[14:16] += rng.uniform(-HALFWIDTH["finger_v_m_s"], HALFWIDTH["finger_v_m_s"], 2)
    xy = rng.uniform(-1., 1., 2) * [HALFWIDTH["pin_fork_cross_m"], HALFWIDTH["pin_fork_along_m"]]
    q[16:18] += fork_r[:2, :2] @ xy
    yaw = np.deg2rad(rng.uniform(-HALFWIDTH["pin_yaw_deg"], HALFWIDTH["pin_yaw_deg"]))
    quat = np.empty(4)
    mj.mju_mulQuat(quat, np.array([np.cos(yaw/2), 0., 0., np.sin(yaw/2)]), q[19:23])
    q[19:23] = quat / np.linalg.norm(quat)
    return Handoff(q, v.astype(np.float32), target, src.bias.copy())


class _Geometry:
    def __init__(self, model):
        self.model, self.data = model, mj.MjData(model)
        self.rails = [model.geom(n).id for n in RAIL_NAMES]
        self.shoulder = model.geom("part_shoulder").id
        self.pin = model.site("part_grasp_center").id
        self.wrist = model.site("right_grasp_center").id
        self.fork = model.site("pickup_grasp_center").id
        self.part = {model.geom(name).id for name in PART_GEOMS}
        bodies = [model.geom_bodyid[model.geom(n).id] for n in PAD_NAMES]
        self.fingers = set(np.flatnonzero(np.isin(model.geom_bodyid, bodies)))
        mj.mj_kinematics(model, self.data)
        self.fork_r = self.data.site_xmat[self.fork].reshape(3, 3).copy()

    def inspect(self, sample):
        """Check the actual FP32 device-start coordinates with separate native geometry."""
        q, v = sample.qpos.astype(np.float32), sample.qvel.astype(np.float32)
        data, model = self.data, self.model
        if not np.isfinite(q).all() or not np.isfinite(v).all():
            return "nonfinite", {}
        if np.any(q[:16] <= model.jnt_range[:16, 0]) or np.any(q[:16] >= model.jnt_range[:16, 1]):
            return "joint_range", {}
        if (np.any(np.abs(v[:14]) > np.tile([2.175]*4 + [2.61]*3, 2))
                or np.max(np.abs(v[14:16])) > .20 or np.linalg.norm(v[16:19]) > .08
                or np.linalg.norm(v[19:22]) > 2.):
            return "initial_speed", {}
        data.qpos[:], data.qvel[:] = q, v
        mj.mj_kinematics(model, data)
        mj.mj_collision(model, data)
        wrist_distance = np.linalg.norm(data.site_xpos[self.pin] - data.site_xpos[self.wrist])
        if wrist_distance > .080:
            return "outside_pickup_neighborhood", {}
        finger_gap = np.inf
        for c in data.contact[:data.ncon]:
            a, b = map(int, c.geom)
            if (a in self.fingers and b in self.part) or (b in self.fingers and a in self.part):
                finger_gap = min(finger_gap, float(c.dist))
            if c.dist < -.0005:
                return "initial_penetration", {}
        if finger_gap < 0.:
            return "already_in_finger_contact", {}
        # A shoulder no more than 0.35 mm above BOTH rails can settle under
        # gravity. The COM projection must be between the outer support edges.
        gaps = [float(mj.mj_geomDistance(model, data, self.shoulder, rail, .100, None)) for rail in self.rails]
        if min(gaps) < -.0005 or max(gaps) > .00035:
            return "shoulder_off_fork", {}
        com = self.fork_r.T @ (data.xipos[model.body("assembly_part").id] - data.site_xpos[self.fork])
        rail_xy = np.stack([self.fork_r.T @ (data.geom_xpos[i] - data.site_xpos[self.fork]) for i in self.rails])
        low = (rail_xy[:, :2] - model.geom_size[self.rails, :2]).min(axis=0)
        high = (rail_xy[:, :2] + model.geom_size[self.rails, :2]).max(axis=0)
        if np.any(com[:2] <= low) or np.any(com[:2] >= high):
            return "com_outside_support", {}
        return None, dict(shoulder_rail_gaps_m=gaps, com_fork_xy_m=com[:2].tolist(),
                          hand_pin_distance_m=float(wrist_distance), arm_peak_speed_rad_s=float(np.abs(v[:14]).max()),
                          pin_linear_speed_m_s=float(np.linalg.norm(v[16:19])), pin_angular_speed_rad_s=float(np.linalg.norm(v[19:])))


def make_batch(scene_xml, handoff_dir, split, group, worlds=32, *, batch=0):
    """Return a single shared compiled domain, randomized handoffs and their manifest.

    Reusing a split/group across batch numbers holds materials fixed. Per-world
    reset streams and source-row permutations are deterministic and batch-specific.
    """
    if split not in SPLITS or any(type(x) is not int or x < 0 for x in (batch, group, worlds)) or worlds == 0:
        raise ValueError("expected train/select/blind, nonnegative group/batch and positive world count")
    canonical = hashlib.sha256(scene_xml.encode()).hexdigest()
    if canonical != CANONICAL_XML_SHA256:
        raise ValueError("expected exact XML of the published reach-127 scene")
    seed, bank, indices = SPLITS[split]
    path = Path(handoff_dir) / f"{bank}.npz"
    xml, mass_factor, pad_factor, pin_factor = _xml_domain(scene_xml, seed, group)
    model = mj.MjModel.from_xml_string(xml)
    sources = read_handoffs(path, model, expected_split=bank)
    geometry = _Geometry(model)
    order_rng = _rng(seed, group, 1, batch)
    order = np.concatenate([order_rng.permutation(indices) for _ in range((worlds + len(indices) - 1) // len(indices))])
    samples, rows, rejected = [], [], Counter()
    digest = hashlib.sha256()
    for world, index in enumerate(order[:worlds]):
        rng = _rng(seed, group, 2, batch, world)
        failures = Counter()
        for attempt in range(1, MAX_ATTEMPTS + 1):
            sample = _candidate(model, sources[index], geometry.fork_r, rng)
            reason, measured = geometry.inspect(sample)
            if reason is None:
                break
            failures[reason] += 1
        else:
            raise ValueError(f"{split} group {group} batch {batch} world {world} source {index}: "
                             f"no supported reset in {MAX_ATTEMPTS} draws: {dict(failures)}")
        samples.append(sample)
        rejected.update(failures)
        single = hashlib.sha256()
        for arr in (sample.qpos, sample.qvel, sample.target, sample.bias):
            single.update(np.asarray(arr, dtype="<f8").tobytes())
        digest.update(single.digest())
        rows.append(dict(world=world, source_world=int(index), attempts=attempt, rejected=dict(failures),
                         reset_sha256=single.hexdigest(), **measured))
    part = model.body("assembly_part").id
    manifest = dict(schema="pickup-rl-domain-v1", split=split, split_seed=seed, group=group, batch=batch,
                    source_file=path.name, source_sha256=HANDOFF_SHA256[bank], allowed_source_worlds=list(indices),
                    source_history=("The published pickup student trained on the factor-update handoffs."
                                    if bank == "factor-update" else
                                    "The published pickup baseline was previously evaluated on the validation handoffs."),
                    canonical_xml_sha256=canonical, xml_sha256=hashlib.sha256(xml.encode()).hexdigest(),
                    physics=dict(mass_factor=mass_factor, pad_slide_friction_factor=pad_factor,
                                 pin_slide_friction_factor=pin_factor, part_mass_kg=float(model.body_mass[part]),
                                 part_inertia_kg_m2=model.body_inertia[part].tolist(),
                                 sliding_mu={n: float(model.geom(n).friction[0]) for n in (*PAD_NAMES, *PART_GEOMS)}),
                    distribution=dict(nominal_group=0, mass_uniform_factor=MASS_RANGE, independent_friction_uniform_factors=FRICTION_RANGE,
                                      reset_uniform_halfwidth=HALFWIDTH.copy(), velocity_basis="Recorded handoff plus arm/finger increments; retain pin twist.",
                                      target_basis="Shift arm and finger setpoints with their joint perturbations, then clip to controller range.",
                                      bias_basis="Retain the recorded previous-input motor bias as initial controller memory.",
                                      seed_streams="PCG64 SeedSequence: [seed,group,0] domain; [seed,group,1,batch] source order; [seed,group,2,batch,world] resets."),
                    valid_reset_count=len(samples), rejected_draws=dict(rejected), rows=rows, resets_sha256=digest.hexdigest())
    return DomainBatch(xml, model, samples, manifest)
