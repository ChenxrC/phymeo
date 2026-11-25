"""Inspect the solver's loaded contacts, lifted workpieces and the registered PushT target."""
from dataclasses import replace
import math

import mujoco as mj
import numpy as np

from phymeo.training.manipulation_tasks import H, footprint_iou


class ContactEvidence:
    def __init__(self, scene, model, setup):
        self.scene, self.model = scene, model
        self.obj = set(model.geom(n).id for n in scene.object_geom_names)
        self.tools = [model.geom(n).id for n in scene.finger_geom_names or ('pusher',)]
        self.support = model.geom('support').id
        self.body = model.body('object').id
        self.weight = scene.object_mass_kg * 9.81
        self.force = np.zeros(6)
        self.longest_lift = self.lift_ticks = self.pushed_ticks = self.rest_ticks = 0
        self.carried = self.await_open = False
        self.table_handoffs = 0
        self.max_bottom = self.push_distance = self.max_gripped_travel = 0.
        self.peak_forces = np.zeros(len(self.tools))
        self.previous = self.lift_start = None
        self.first_loaded_s = None
        self.target = None
        if not scene.finger_geom_names:
            self.target = replace(scene.targets[0], position=tuple(setup['target_position_m']),
                                  yaw=setup['target_yaw_rad'])
        self.checkpoints = []

    def inspect(self, data, tick):
        # After mj_step these Cartesian geoms, contacts and contact forces all
        # describe the just-solved input geometry. qpos already belongs to the next step.
        vals = dict.fromkeys([self.support, *self.tools], 0.)
        for con_index, contact in enumerate(data.contact):
            a, b = contact.geom
            other = b if a in self.obj else a if b in self.obj else None
            if other not in vals or contact.efc_address < 0:
                continue
            mj.mj_contactForce(self.model, data, con_index, self.force)
            vals[other] += max(0., self.force[0])
        tool_load = np.asarray([vals[i] for i in self.tools])
        self.peak_forces = np.maximum(self.peak_forces, tool_load)
        loaded = tool_load.min() > .02
        if loaded and self.first_loaded_s is None:
            self.first_loaded_s = tick*H
        xyz = data.xpos[self.body]
        bottoms = []
        for i in self.obj:
            sizes = self.model.geom_size[i]
            z = data.geom_xmat[i, 6:9]
            kind = self.model.geom_type[i]
            if kind == mj.mjtGeom.mjGEOM_BOX:
                extent = np.abs(z) @ sizes
            elif kind == mj.mjtGeom.mjGEOM_CAPSULE:
                extent = sizes[0] + abs(z[2])*sizes[1]
            else:
                extent = sizes[0]
            bottoms.append(data.geom_xpos[i, 2] - extent)
        bottom = float(min(bottoms))
        lifted = loaded and vals[self.support] < .02 and bottom >= .03
        supported = abs(vals[self.support] - self.weight) <= .2*self.weight and tool_load.max() < .02
        self.rest_ticks = self.rest_ticks + 1 if supported else 0
        if lifted:
            if self.lift_ticks == 0:
                self.lift_start = xyz.copy()
            self.lift_ticks += 1
            self.longest_lift = max(self.longest_lift, self.lift_ticks)
            self.max_gripped_travel = max(self.max_gripped_travel, float(np.linalg.norm(xyz[:2]-self.lift_start[:2])))
            self.max_bottom = max(self.max_bottom, bottom)
            if np.linalg.norm(xyz[:2]-self.lift_start[:2]) >= .03:
                self.carried = True
            self.await_open = False
        else:
            self.lift_ticks = 0
        if self.carried and bottom < .003 and vals[self.support] > .2*self.weight:
            self.await_open = True
            self.carried = False
        elif self.carried and not loaded:
            self.carried = False
        if self.await_open and self.rest_ticks*H >= .2:
            self.table_handoffs += 1
            self.await_open = False
        if self.target is not None and loaded:
            self.pushed_ticks += 1
            if self.previous is not None:
                self.push_distance += float(np.linalg.norm(xyz[:2] - self.previous[:2]))
        self.previous = xyz.copy()
        if tick % 1000 == 0:
            self.checkpoints.append([tick, *np.round(xyz, 5).tolist(), round(bottom, 5),
                                     *np.round(tool_load, 3).tolist(), round(vals[self.support], 3)])

    def result(self, data):
        observation = dict(first_loaded_s=self.first_loaded_s, peak_tool_normal_forces_n=self.peak_forces.tolist(),
                           final_released_on_table_s=self.rest_ticks*H, samples=self.checkpoints)
        if self.target is None:
            observation.update(loaded_clearance_m=self.max_bottom, longest_continuous_loaded_lift_s=self.longest_lift*H,
                               loaded_xy_transport_m=self.max_gripped_travel,
                               carried_to_table_then_released=self.table_handoffs)
            passed = self.longest_lift*H >= .2 and self.max_gripped_travel >= .03 and self.rest_ticks*H >= .2
        else:
            xyz, quat = data.xpos[self.body], data.xquat[self.body]
            goal = self.scene.goals[self.scene.default_goal]
            rot = data.xmat[self.body].reshape(3, 3)
            angle = math.atan2(rot[1, 0], rot[0, 0])-self.target.yaw
            yaw_error = abs(math.atan2(math.sin(angle), math.cos(angle)))
            position_error = math.dist(xyz[:2], self.target.position[:2])
            iou = footprint_iou(self.scene.object_shapes, xyz, quat, self.target)
            velocity = np.zeros(6)
            mj.mj_objectVelocity(self.model, data, mj.mjtObj.mjOBJ_BODY, self.body, velocity, 0)
            tilt = math.acos(float(np.clip(rot[2, 2], -1., 1.)))
            angular, speed = float(np.linalg.norm(velocity[:3])), float(np.linalg.norm(velocity[3:]))
            observation.update(push_loaded_s=self.pushed_ticks*H, motion_under_push_m=self.push_distance,
                               final_iou=iou, final_yaw_error_rad=yaw_error, final_xy_error_m=position_error,
                               final_tilt_rad=tilt, final_linear_speed_m_s=speed, final_angular_speed_rad_s=angular)
            passed = (self.push_distance >= .03 and self.pushed_ticks*H >= .15 and self.rest_ticks*H >= .2
                      and position_error <= goal.position_tolerance_m and yaw_error <= goal.yaw_tolerance_rad
                      and iou >= goal.footprint_iou_min and tilt <= goal.tilt_tolerance_rad
                      and speed <= goal.speed_m_s and angular <= goal.angular_speed_rad_s)
        return dict(observation, passed=bool(passed))
