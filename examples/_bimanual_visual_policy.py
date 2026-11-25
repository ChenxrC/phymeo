"""Shared visual manipulation primitives. Only RGB, installed geometry and arm joints enter the policy."""

import numpy as np

from _bimanual_image_pose import GroundMark, locate, outline, template

PARAMETERS = ("closure", "lift", "translation", "visual_correction", "rotation", "release", "damping")
PRIOR = np.zeros(len(PARAMETERS))


def _original_visual_observation(rgb, cameras, planes):
    """First, centroid-based controller retained as the original policy comparison."""
    points = [[], []]
    counts = []
    for img, (eye, rotation) in zip(rgb, cameras, strict=True):
        r, g, b = np.moveaxis(img.astype(float), -1, 0)
        masks = ((r > 1.20*g + 5) & (r > 1.26*b + 5),
                 (g > 1.15*r + 4) & (b > 1.08*r + 3) & (g > .88*b))
        h, w = img.shape[:2]
        f = .5*h / np.tan(np.deg2rad(39.0))
        counts.append([])
        for i, mask in enumerate(masks):
            y, x = np.nonzero(mask)
            p = np.empty((0, 2))
            if len(x) >= 4:
                directions = np.stack(((x+.5-w/2)/f, -(y+.5-h/2)/f, -np.ones(len(x))), axis=1) @ rotation.T
                rays = directions[np.abs(directions[:, 2]) > 1e-5]
                distance = (planes[i] - eye[2]) / rays[:, 2]
                ok = (distance > 0) & (distance < 1.8)
                p = eye[:2] + distance[ok, None]*rays[ok, :2]
            counts[-1].append(len(p))
            if len(p) >= 4:
                points[i].append(p)
    observations = []
    for p in points:
        if not p:
            observations.append(None)
            continue
        center = np.mean([np.median(v, axis=0) for v in p], axis=0)
        covariance = np.mean([(v-center).T @ (v-center)/len(v) for v in p], axis=0)
        values, axes = np.linalg.eigh(covariance)
        angle = np.arctan2(axes[1, -1], axes[0, -1])
        observations.append((np.r_[center, angle, 0.], float((values[-1]-values[0])/max(values.sum(), 1e-10))))
    return observations, counts


class VisualPolicy:
    """State-feedback targets in Cartesian space, converted to incremental joint commands.

    Parameters multiply closure, clearance, travel, visual alignment, turning,
    release radius and IK damping. Progress depends on arm/read-image feedback.
    """

    def __init__(self, task, model, weights, *, variant="centroid", decision_dt=.010, servo_gains=None):
        import mujoco as mj

        self.mj, self.model, self.task = mj, model, task
        if variant not in ("centroid", "registered", "fiducial"):
            raise ValueError("unknown policy variant")
        self.variant = variant
        self.dt = decision_dt
        self.data = mj.MjData(model)
        w = np.asarray(weights, float)
        if w.shape != PRIOR.shape or not np.isfinite(w).all():
            raise ValueError("invalid visual policy weights")
        self.gains = np.exp(np.clip(w, -1.5, 1.5))
        self.joints, self.sites = [], []
        for name in ("left", "right"):
            ids = [mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, f"{name}_j{i}") for i in range(7)]
            joints = np.array([j for j in ids if j >= 0])
            self.joints.append((model.jnt_qposadr[joints], model.jnt_dofadr[joints],
                                np.where(model.jnt_limited[joints, None], model.jnt_range[joints], [-np.inf, np.inf])))
            tool, eye = (mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, f"{name}_{suffix}")
                         for suffix in ("tool", "wrist_camera"))
            if len(joints) not in (6, 7) or tool < 0 or eye < 0:
                raise ValueError(f"incomplete physical {name} arm/camera installation")
            self.sites.append((tool, eye))
        self.rest = self.reference_q = self.initial_object = None
        self.object = self.goal = None
        self.templates = (template(task, False), template(task, True))
        self.goal_fit = np.inf
        self.mark = GroundMark(task)
        self.goal_shape = self.object_shape = 0.
        self.initial_bottom = None
        self.place_xy = None
        self.up_effort = self.squeeze_effort = 0.
        self.effort_limited = False
        self.servo_gains = servo_gains
        self.fiducial = None
        if variant == "fiducial":
            from _bimanual_fiducial import FiducialControl
            self.fiducial = FiducialControl(task, decision_dt)
        self.top = self.releasing = False
        self.features = {}

    def act(self, proprio, rgb, *, wrist_force_world_n=None):
        proprio = np.asarray(proprio, float)
        rgb = np.asarray(rgb)
        if proprio.shape != (42,) or rgb.ndim != 4 or rgb.shape[0] != 2 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
            raise ValueError("policy requires 42 joint observations and two HWC uint8 RGB images")
        mj, model, data = self.mj, self.model, self.data
        q = proprio[:14].reshape(2, 7)
        active = proprio[28:].reshape(2, 7)
        for i, (qa, _, _) in enumerate(self.joints):
            data.qpos[qa] = q[i, :len(qa)]
        # The observer's payload stays at model defaults. Only robot site FK/J are used.
        mj.mj_kinematics(model, data)
        mj.mj_comPos(model, data)
        tools = np.asarray([data.site_xpos[tool].copy() for tool, _ in self.sites])
        if self.rest is None:
            self.rest, self.reference_q = tools.copy(), q.copy()
        base = self.rest.mean(axis=0)
        center = tools.mean(axis=0)
        height = max(0., center[2] - base[2])
        carry = self.task == "carry"
        cameras = [(data.site_xpos[eye], data.site_xmat[eye].reshape(3, 3)) for _, eye in self.sites]
        if self.fiducial is not None:
            result = self.fiducial.apply(self, proprio, rgb, tools, cameras, wrist_force_world_n)
            self.features = self.fiducial.features
            return result
        registered = self.variant == "registered"
        if registered:
            views = [outline(img, *cam) for img, cam in zip(rgb, cameras, strict=True)]
            counts = [[0 if v is None else v[4] for v in pair] for pair in views]
            observation = locate(self.templates[0], [v[0] for v in views], self.object, is_goal=False,
                                 iterations=8 if self.object is None else 3)
            estimate = self.mark.observe([v[1] for v in views], base[:2])
            target_observation = None if estimate is None else (estimate, self.mark.fit_error_m)
        else:
            (observation, target_observation), counts = _original_visual_observation(
                rgb, cameras, ((.094 + height) if carry else .049, .002))
        if observation is not None:
            self.object = observation[0]
            self.object_shape = observation[1]
            if self.initial_object is None:
                self.initial_object = self.object[:2].copy()
                self.initial_bottom = float(self.object[3])
        if target_observation is not None:
            estimate, error = target_observation
            # The installation marker is stationary. Prevent an occluded edge
            # in one late exposure dragging a previously well-observed target.
            if not registered or self.goal is None or error < self.goal_fit + .25:
                self.goal = estimate
                self.goal_fit = self.goal_shape = error
        close, lift, translation, visual, rotation, release, damping = self.gains
        delta = np.zeros(2)
        orientation_error = 0.
        if self.goal is not None and self.object is not None:
            delta = self.goal[:2] - self.object[:2]
            angle = self.goal[2] - self.object[2]
            if registered or min(self.goal_shape, self.object_shape) > .12:
                orientation_error = .5 * np.arctan2(np.sin(2 * angle), np.cos(2 * angle))
        object_motion = self.object[:2] - self.initial_object if self.object is not None else np.zeros(2)
        # Keep alignment errors bounded when a silhouette enters or leaves the edge of a wrist image.
        visual_offset = np.clip(object_motion - (center - base)[:2], -.024, .024) * visual
        travel = np.clip(translation * delta, -.055, .055)
        target = tools.copy()
        actual_lift = 0. if self.initial_bottom is None else float(self.object[3] - self.initial_bottom)
        stereo = registered and observation is not None and self.object_shape < 4. and all(k[0] >= 5 for k in counts)
        effort = np.zeros((2, 3))
        closure = 0.
        if carry:
            sign = np.sign(self.rest[:, 1] - base[1])
            closure = .5 * (np.ptp(self.rest[:, 1]) - np.ptp(tools[:, 1]))
            # Approach, lift and release are conditioned on observed geometry.
            # No elapsed-time stage schedule or load trajectory enters the actor.
            lifted = stereo and actual_lift > .035 if registered else height > .032
            if lifted:
                self.top = True
            travelled = not registered or (stereo and np.linalg.norm(object_motion) > .060)
            radius = .016 * np.clip(release, .65, 1.4) if registered else .026 * release
            if self.top and travelled and self.goal is not None and self.object is not None and np.linalg.norm(delta) < radius:
                if not self.releasing:
                    self.place_xy = center[:2].copy()
                self.releasing = True
            clamp = np.clip((closure - .002) / .005, 0., 1.)
            dz = .060 * lift * clamp if not self.releasing else 0.
            target[:, 2] = self.rest[:, 2] + dz
            clearance = actual_lift if registered else height
            move = np.clip((clearance - .018) / .025, 0., 1.) if not self.releasing else 0.
            if registered and not stereo:
                move = 0.
            target[:, :2] += move * travel
            target[:, :2] += .28 * visual_offset
            front = center[1] + .28 * visual_offset[1] + move * travel[1]
            if registered and self.releasing:
                target[:, 0] = self.place_xy[0] + (self.rest[:, 0] - base[0])
                front = self.place_xy[1]
            half = .5 * np.ptp(self.rest[:, 1]) - .032 * close
            opening = self.releasing and ((stereo and actual_lift < .008) if registered else height < .012)
            if opening:
                half = .5 * np.ptp(self.rest[:, 1]) + .016
            target[:, 1] = front + sign * half
            if registered:
                # Integral wrench compensation uses visible load clearance and
                # robot pad displacement. Integration stops when actuators saturate.
                if self.releasing or not stereo:
                    self.up_effort *= np.exp(-self.dt/.09)
                elif not self.effort_limited:
                    self.up_effort += self.dt * 900. * clamp * (dz - actual_lift)
                if opening or not stereo:
                    self.squeeze_effort *= np.exp(-self.dt/.06)
                elif not self.effort_limited and not self.releasing:
                    self.squeeze_effort += self.dt * 750. * clamp * max(dz - actual_lift, 0.)
                self.up_effort = float(np.clip(self.up_effort, 0., 42.))
                self.squeeze_effort = float(np.clip(self.squeeze_effort, 0., 50.))
                effort[:, 1] = -sign * self.squeeze_effort
                effort[:, 2] = self.up_effort
        else:
            # Two independently commanded tools press and turn the workpiece.
            approach = base[:2] + object_motion if self.object is not None else base[:2]
            y_relative = self.rest[:, 1] - base[1]
            push = np.clip(np.linalg.norm(delta)/.040 + abs(orientation_error), 0., 1.) if registered else 1.
            target[:, 0] = approach[0] + .019 * close * push + travel[0] - np.clip(orientation_error * rotation, -.45, .45) * y_relative
            target[:, 1] = approach[1] + y_relative + travel[1] + .2 * visual_offset[1]
            target[:, 2] = self.rest[:, 2]
            if registered:
                effort[:, 0] = 19. * np.clip(delta[0]/.035 - np.sign(y_relative)*orientation_error*1.2, -.4, 1.)
        gravity = np.zeros(model.nv)
        if registered:
            mj.mj_comVel(model, data)
            mj.mj_rne(model, data, 0, gravity)
        self.effort_limited = False
        result = np.zeros((2, 7))
        for i, ((qa, va, limits), (tool, _)) in enumerate(zip(self.joints, self.sites, strict=True)):
            n = len(qa)
            jac = np.zeros((3, model.nv))
            mj.mj_jacSite(model, data, jac, None, tool)
            j = jac[:, va]
            dx = np.clip(target[i] - tools[i], -.060, .060)
            dq = j.T @ np.linalg.solve(j @ j.T + (.025 * damping)**2 * np.eye(3), dx)
            if registered:
                # Convert robot gravity and task-space force into an equivalent
                # target offset for the actual source PD motors, retaining their limits.
                gains = self.servo_gains
                start = sum(len(joints[0]) for joints in self.joints[:i])
                kp = gains[0][start:start + n]
                tau = gravity[va] + j.T @ effort[i]
                bounds = model.actuator_ctrlrange[start:start+n]
                self.effort_limited |= bool(np.any(np.abs(tau) > .96 * bounds[:, 1]) and active[i, 0])
                dq += np.clip(tau, bounds[:, 0], bounds[:, 1]) / kp
            desired = np.clip(q[i, :n] + dq, limits[:, 0], limits[:, 1])
            result[i, :n] = np.clip((desired - self.reference_q[i, :n]) / .045, -1., 1.) * active[i, :n]
            self.reference_q[i, :n] = np.clip(self.reference_q[i, :n] + .045 * result[i, :n], limits[:, 0], limits[:, 1])
        self.features = dict(pixel_counts=counts, variant=self.variant, tool_heights=tools[:, 2].tolist(),
                             closure_m=float(closure), stereo_load_height=bool(stereo), inferred_lift_m=actual_lift,
                             commanded_up_n=self.up_effort, commanded_squeeze_n=self.squeeze_effort,
                             object_xy=None if self.object is None else self.object[:2].tolist(),
                             goal_xy=None if self.goal is None else self.goal[:2].tolist(),
                             object_bottom_z=None if self.object is None else float(self.object[3]),
                             goal_yaw=None if self.goal is None else float(self.goal[2]),
                             goal_edge_count=self.mark.fit_count, goal_edge_error_m=self.mark.fit_error_m,
                             object_fit_error_px=self.object_shape if registered and observation is not None else None,
                             image_yaw_error=orientation_error, releasing=self.releasing)
        return result.ravel()
