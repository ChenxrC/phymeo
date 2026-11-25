"""Motor-driven photography routine for the registered parallel-jaw grippers."""
import math
from hashlib import sha256
import xml.etree.ElementTree as ET

import mujoco as mj
import numpy as np


def studio_grasp_mjcf(scene):
    """Shallow V-shaped fingers and a rolling-resistance table contact for round parts."""
    if not (scene.finger_geom_names and len(scene.object_shapes) == 1
            and scene.object_shapes[0].kind in ('sphere', 'capsule')):
        return scene.mjcf
    tree = ET.fromstring(scene.mjcf)
    for side, name in zip((-1, 1), scene.finger_geom_names, strict=True):
        pad = tree.find(f".//geom[@name='{name}']")
        half_angle = side * math.radians(13.) / 2.
        pad.set('quat', f'{math.cos(half_angle):.17g} {math.sin(half_angle):.17g} 0 0')
        # Broad rubber fingers make area contact with the round stock. Enable
        # the source material's torsional and rolling coefficients on the pads.
        pad.set('size', '.038 .009 .040')
        pad.set('condim', '6')
    # The source geoms declare sliding, torsional and rolling coefficients,
    # but their default 3-D table contact only uses sliding friction. Apply
    # the existing two additional coefficients only to this exact geom pair.
    material = tree.find('./default/geom')
    slide, spin, roll = material.attrib['friction'].split()
    contacts = tree.find('./contact')
    if contacts is None:
        contacts = ET.SubElement(tree, 'contact')
    ET.SubElement(contacts, 'pair', name='studio-round-worktop',
                  geom1=scene.support_geom_name, geom2=scene.object_geom_names[0], condim='6',
                  friction=' '.join((slide, slide, spin, roll, roll)),
                  solref=material.attrib['solref'], solimp=material.attrib['solimp'])
    return ET.tostring(tree, encoding='unicode')


def _upright_extent(shape):
    if shape.kind == 'box':
        return shape.size[1], shape.size[2]
    if shape.kind == 'capsule':
        return shape.size[0], sum(shape.size)
    if shape.kind == 'sphere':
        return shape.size[0], shape.size[0]
    raise ValueError(f'unknown workpiece primitive {shape.kind}')


class _GraspController:
    def __init__(self, scene, model, data):
        if data.time != 0 or not scene.finger_geom_names:
            raise ValueError('initialize an original two-finger gripper at time zero')
        self.model = model
        self.actors = np.array([model.actuator(n).id for n in scene.actuator_names])
        joints = np.array([model.joint(n).id for n in scene.robot_joint_names])
        self.q = model.jnt_qposadr[joints]
        self.v = model.jnt_dofadr[joints]
        self.period = 10.
        self.spans = np.array((0., .13, .22, .29, .43, .56, .69, .77, .87, 1.))
        self.turn = .28
        pads = [model.geom(g).id for g in scene.finger_geom_names]
        obj_joint = model.joint('object-free').id
        obj = model.jnt_qposadr[obj_joint]
        scratch = mj.MjData(model)
        mj.mj_forward(model, scratch)
        original = data.qpos[obj:obj + 3].copy()
        original_quat = data.qpos[obj + 3:obj + 7].copy()
        home = np.zeros(len(joints))
        low, high = home.copy(), home.copy()
        z0 = float(np.mean(scratch.geom_xpos[pads, 2]))
        pad_z = float(model.geom_size[pads[0], 2])
        forward_gap = float(np.linalg.norm(scratch.geom_xpos[pads[1]] - scratch.geom_xpos[pads[0]]) / 2
                            - model.geom_size[pads[0], 1])
        extent = [_upright_extent(shape) for shape in scene.object_shapes]
        if len(scene.object_shapes) == 1 and scene.object_shapes[0].kind == 'capsule':
            # A capsule on one rounded end topples before the fingers arrive.
            # Lay its long axis across the two pad faces, along local x.
            radius = scene.object_shapes[0].size[0]
            data.qpos[obj + 2] = radius + .001
            data.qpos[obj + 3:obj + 7] = (math.sqrt(.5), 0., math.sqrt(.5), 0.)
            extent = [(radius, radius)]
        upper = max(shape.position[2] + half_z for shape, (_, half_z) in zip(scene.object_shapes, extent))
        half_width = max(abs(shape.position[1]) + half_y for shape, (half_y, _) in zip(scene.object_shapes, extent))
        # Keep pad bottoms above the physical plane and the wrist above tall
        # workpieces. The flat faces still extend below the object's equator.
        grip_z = max(pad_z + .006, data.qpos[obj + 2] + upper - .025)
        lifted_z = min(.152, grip_z + .084)
        if scene.robot_kinds == 'RRPR':
            for pose, height in ((low, grip_z), (high, lifted_z)):
                pose[:4] = (-.19, .38, height - z0, -.19)
        else:
            # Counter-rotate the adjacent pitch joints. Only their intervening
            # link contributes vertical motion, while both pad faces stay upright.
            pitch_link = float(model.body('link2').pos[0])
            for pose, height in ((low, grip_z), (high, lifted_z)):
                angle = math.asin((z0 - height) / pitch_link)
                pose[1:3] = (angle, -angle)
        scratch.qpos[self.q] = low
        mj.mj_kinematics(model, scratch)
        midpoint = scratch.geom_xpos[pads].mean(axis=0)
        data.qpos[obj:obj + 2] = midpoint[:2]
        data.qpos[self.q] = high
        mj.mj_forward(model, data)
        closed = np.clip(forward_gap - half_width + .015, 0., model.jnt_range[joints[-1], 1])
        low_closed, high_closed = low.copy(), high.copy()
        low_closed[-2:] = closed
        high_closed[-2:] = closed
        # Hold the grasp on the table before lifting, and let the table take
        # the load at the second station before opening the jaws.
        self.poses = np.array((high, low, low_closed, low_closed, high_closed, high_closed,
                               low_closed, low_closed, low, high))
        self.yaw = np.array((0., 0., 0., 0., 0., 1., 1., 1., 1., 1.))
        self.acceleration = np.zeros(model.nv)
        self.inertial_force = np.zeros(model.nv)
        self.round_stock = len(scene.object_shapes) == 1 and scene.object_shapes[0].kind in ('sphere', 'capsule')
        self.squeeze = 4.
        if self.round_stock:
            # Horizontal motor force at each 13-degree face needed to support
            # mg without squeezing the workpiece up the widening throat.
            self.squeeze = 1.05 * scene.object_mass_kg * abs(model.opt.gravity[2]) / (2. * math.tan(math.radians(13.)))
            self.period = 15.
            self.turn = .18
        self.setup = dict(original_object_xyz=original.tolist(), initial_object_xyz=data.qpos[obj:obj + 3].tolist(),
                          initial_joint_angles_or_travel=high.tolist(), low_pad_height_m=grip_z,
                          original_object_wxyz=original_quat.tolist(), initial_object_wxyz=data.qpos[obj + 3:obj + 7].tolist(),
                          studio_mjcf_sha256=sha256(studio_grasp_mjcf(scene).encode()).hexdigest(),
                          pad_quaternions_wxyz=model.geom_quat[pads].tolist(),
                          squeeze_limit_n=self.squeeze,
                          high_pad_height_m=lifted_z, closed_jaw_target_m=float(closed),
                          placement_yaw_rad=self.turn, cycle_s=self.period,
                          control='coupled mass-matrix PD arm; force-limited independent jaw motors')
        if self.round_stock:
            pair = model.pair('studio-round-worktop').id
            self.setup['worktop_contact'] = dict(geoms=[scene.support_geom_name, scene.object_geom_names[0]],
                condim=int(model.pair_dim[pair]), friction=model.pair_friction[pair].tolist(),
                solref=model.pair_solref[pair].tolist(), solimp=model.pair_solimp[pair].tolist())

    def apply(self, data, t):
        if self.round_stock:
            # Complete one handoff, then park with the jaws open above the
            # receiver while the transport and settling remain in view.
            t = min(t, self.period)
        phase = (t / self.period) % 1.
        segment = np.searchsorted(self.spans, phase, side='right') - 1
        u = (phase - self.spans[segment]) / (self.spans[segment + 1] - self.spans[segment])
        blend = u*u*u * (10. + u * (6.*u - 15.))
        target = self.poses[segment] + blend * (self.poses[segment + 1] - self.poses[segment])
        fraction = self.yaw[segment] + blend * (self.yaw[segment + 1] - self.yaw[segment])
        if int(t / self.period) % 2:
            fraction = 1. - fraction
        target[0] += self.turn * fraction
        error, speed = target - data.qpos[self.q], data.qvel[self.v]
        # Independent derivative torques overdrive coupled, low-inertia modes.
        # Apply the arm's complete current inertia to a damped acceleration
        # request, leaving both fingers compliant while in contact.
        self.acceleration[self.v[:-2]] = 1000. * error[:-2] - 60. * speed[:-2]
        mj.mj_mulM(self.model, data, self.inertial_force, self.acceleration)
        force = self.inertial_force[self.v] + data.qfrc_bias[self.v]
        squeeze = self.squeeze
        if self.round_stock and .69 <= phase < .87:
            # Ease the grip during the on-table dwell. The underside settles
            # onto the bench while both pads still share some compressive load.
            u = min((phase - .69) / .05, 1.)
            squeeze *= 1. - .65 * u*u*(3.-2.*u)
        force[-2:] += np.clip(290. * error[-2:] - 5. * speed[-2:], -4., squeeze)
        data.ctrl[self.actors] = np.clip(force, *self.model.actuator_ctrlrange[self.actors].T)


def grasp_setup(scene, model, data):
    """Initialize the workpiece under the open, raised jaws; return its motor controller."""
    return _GraspController(scene, model, data)
