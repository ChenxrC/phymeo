"""Low-speed physical Push-T demonstration for the three registered tabletop arms."""
import math

import mujoco as mj
import numpy as np


def _ease(t):
    t = float(np.clip(t, 0., 1.))
    return t**3 * (10. - 15.*t + 6.*t*t)


class PushController:
    """Set a reachable start at t=0; subsequent updates write only the original motors."""

    def __init__(self, scene, model, data):
        if scene.robot_kinds not in ('RR', 'RRR', 'PPPR') or scene.goals[scene.default_goal].kind != 'pusht':
            raise ValueError('expected a registered tabletop Push-T scene')
        if data.time != 0.:
            raise ValueError('the demonstration layout is set only at initialization')
        self.model = model
        joints = [model.joint(name).id for name in scene.robot_joint_names]
        self.q = model.jnt_qposadr[joints]
        self.v = model.jnt_dofadr[joints]
        self.act = [model.actuator(name).id for name in scene.actuator_names]
        self.limits = model.jnt_range[joints].copy()
        self.slide = model.jnt_type[joints] == mj.mjtJoint.mjJNT_SLIDE
        self.dim = 3 if 'P' in scene.robot_kinds else 2
        self.tip = model.geom('pusher').id
        self.obj = model.body(scene.object_key).id
        self.object_q = model.jnt_qposadr[model.joint('object-free').id]
        self.jac = np.zeros((3, model.nv))
        self.eye = np.eye(self.dim)
        self.pullback_t = None
        self.pullback_start = None
        self.trim = None
        self.trim_done = None

        # The transverse bar's outer face is a wide, symmetric contact patch.
        # A vertical capsule pushes its center along the negative world Y axis.
        self.front = max(s.position[1] + s.size[1] for s in scene.object_shapes)
        self.cross_half_width = max(s.size[0] for s in scene.object_shapes)
        self.stem_half_width = min(s.size[0] for s in scene.object_shapes)
        self.tip_r = float(model.geom_size[self.tip, 0])
        floor = scene.support_top_m
        self.x = .38
        self.start_y, self.target_y = .04, -.04
        self.rest_z = floor + max(s.size[2] - s.position[2] for s in scene.object_shapes)
        self.start = np.array([self.x, self.start_y, self.rest_z+.001])
        data.qpos[self.object_q:self.object_q+7] = (*self.start, 1., 0., 0., 0.)
        mj.mj_forward(model, data)
        self.tip_z = floor+.056 if self.dim == 3 else float(data.geom_xpos[self.tip, 2])
        self.clear_y = self.start_y + self.front + self.tip_r + .024
        self._initialize_arm(data, np.array([self.x, self.clear_y, self.tip_z])[:self.dim])
        self.posture = data.qpos[self.q].copy()
        self.setup = dict(
            target_position_m=[self.x, self.target_y, self.rest_z], target_yaw_rad=0.,
            initial_object_position_m=self.start.tolist(), initial_object_yaw_rad=0.,
            initial_arm_qpos=self.posture.tolist(), initial_tip_position_m=data.geom_xpos[self.tip].tolist(),
            goal_origin='fixed reachable studio target, set before simulation',
            approach='center of the positive local Y crossbar face; push 80 mm along world -Y',
            controller='current pusher Jacobian with motor-space bias compensation and yaw feedback',
        )

    def _tip_jacobian(self, data):
        mj.mj_jacGeom(self.model, data, self.jac, None, self.tip)
        return self.jac[:self.dim, self.v]

    def _initialize_arm(self, data, goal):
        """Damped IK is used only to choose the initial unladen robot assembly pose."""
        for _ in range(160):
            error = goal - data.geom_xpos[self.tip, :self.dim]
            if np.linalg.norm(error) < 1.e-5:
                return
            j = self._tip_jacobian(data)
            inc = j.T @ np.linalg.solve(j @ j.T + .0001*self.eye, error)
            step = np.where(self.slide, .02, .12)
            data.qpos[self.q] = np.clip(data.qpos[self.q] + np.clip(inc, -step, step),
                                         self.limits[:, 0]+.002, self.limits[:, 1]-.002)
            mj.mj_forward(self.model, data)
        raise ValueError('could not initialize the pusher at its clear approach pose')

    def _trim_waypoints(self, tip, obj):
        # Small yaw during the first stroke can send a planar slider slightly
        # off its intended line. Push its narrow stem from the exterior after
        # routing the real capsule all the way around the transverse bar.
        direction = math.copysign(1., self.x-obj[0])
        beyond = obj[0] - direction*(self.cross_half_width + self.tip_r + .018)
        side_y = obj[1] - .003
        side_contact = obj[0] - direction*(self.stem_half_width + self.tip_r)
        destination = self.x - direction*(self.stem_half_width + self.tip_r) + direction*.005
        lifted_clear = tip[1] + .035
        points = np.array([(tip[0],tip[1]), (tip[0],lifted_clear), (beyond,lifted_clear),
                           (beyond,side_y), (side_contact-direction*.004,side_y), (destination,side_y)])
        return direction, points

    def _trim_goal(self, t, tip, obj):
        direction, points = self.trim
        age = t - self.pullback_t
        if self.trim_done is None and (age >= 11. or
                (age > 5.5 and direction*(self.x-obj[0]) <= .0002)):
            self.trim_done = (t, tip.copy())
        if self.trim_done is not None:
            done, start = self.trim_done
            return start - np.array([direction*.028*_ease((t-done)/1.25), 0.])
        times = (0., 1., 2.5, 4., 5.5, 9.5)
        for i in range(1, len(times)):
            if age < times[i]:
                s = _ease((age-times[i-1]) / (times[i]-times[i-1]))
                return points[i-1] + s*(points[i]-points[i-1])
        return points[-1]

    def apply(self, data, t):
        """Run at a held controller period of up to 4 ms, advancing only via mj_step."""
        obj = data.xpos[self.obj]
        yaw = math.atan2(data.xmat[self.obj, 3], data.xmat[self.obj, 0])
        tip = data.geom_xpos[self.tip, :self.dim]
        # Stopping is measured on the free body. Until this latch the pusher
        # follows an independently scheduled reference (never the desired outcome).
        if self.pullback_t is None and (t >= 24. or (t > 2. and obj[1] <= self.target_y + .0002)):
            self.pullback_t = t
            self.pullback_start = tip.copy()
            if self.dim == 2 and abs(obj[0]-self.x) > .001:
                self.trim = self._trim_waypoints(tip, obj)
        if self.trim is not None:
            goal = self._trim_goal(t, tip, obj)
        elif self.pullback_t is not None:
            park = self.pullback_start.copy()
            park[1] += .04
            goal = self.pullback_start + _ease((t-self.pullback_t)/2.)*(park-self.pullback_start)
        else:
            face_offset = self.front + self.tip_r
            if t <= 2.:
                y = self.clear_y + _ease(t/2.) * (self.start_y+face_offset-self.clear_y)
            else:
                progress = _ease((t-2.)/10.)
                # A bounded additional stroke counters static friction at the
                # end of the nominal trajectory until the free-body latch fires.
                extra = float(np.clip((t-12.)*.001, 0., .006))
                y = self.start_y + face_offset + progress*(self.target_y-self.start_y) - extra
            # Positive yaw requires a negative world-Z moment. Moving the
            # downward (-Y) push to the right creates this correcting moment.
            x = obj[0] + float(np.clip(.085*yaw, -.012, .012))
            goal = np.array((x, y, self.tip_z))[:self.dim]
        j = self._tip_jacobian(data)
        vel = data.qvel[self.v]
        force = 900.*(goal-tip) - 28.*(j @ vel)
        regularized = j @ j.T + 1.e-6*self.eye
        posture_effort = 1.2*(self.posture-data.qpos[self.q]) - .30*vel
        null_effort = posture_effort - j.T @ np.linalg.solve(regularized, j @ posture_effort)
        tau = j.T @ force + null_effort + data.qfrc_bias[self.v]
        bounds = self.model.actuator_ctrlrange[self.act]
        data.ctrl[self.act] = np.clip(tau, bounds[:, 0], bounds[:, 1])

    update = apply
