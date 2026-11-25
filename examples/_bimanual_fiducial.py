"""Estimate workpiece and destination poses from known paint and calibrated wrist RGB only."""

from itertools import combinations, product

import numpy as np


def _yaw(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def _skew(p):
    x, y, z = p
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def _exp(w):
    a = np.linalg.norm(w)
    s = _skew(w)
    if a < 1e-7:
        return np.eye(3) + s + .5 * s @ s
    return np.eye(3) + (np.sin(a)/a)*s + ((1-np.cos(a))/a**2)*(s@s)


def _hue(rgb):
    rgb = rgb.astype(float)
    hi, lo = rgb.max(axis=2), rgb.min(axis=2)
    r, g, b = np.moveaxis(rgb, -1, 0)
    chroma = hi-lo
    den = np.maximum(1., chroma)
    hue = np.where(hi == r, (g-b)/den, np.where(hi == g, 2+(b-r)/den, 4+(r-g)/den))
    return (60*hue) % 360., chroma, hi


def _components(mask, error, area_limit, *, limit=3):
    """Small chromatic islands; keep the ambiguity for geometric matching."""
    # Run-length union removes Python per-pixel flood fill from the control loop.
    runs, parents, previous = [], [], []
    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i
    for y in np.flatnonzero(mask.any(axis=1)):
        xs = np.flatnonzero(mask[y])
        groups = np.split(xs, np.flatnonzero(np.diff(xs) != 1)+1)
        current = []
        for seg in groups:
            left, right = int(seg[0]), int(seg[-1])
            idx = len(runs)
            parents.append(idx)
            runs.append((int(y), left, right))
            for old, a, b in previous:
                if runs[old][0] == y-1 and a <= right+1 and b >= left-1:
                    parents[root(old)] = root(idx)
            current.append((idx, left, right))
        previous = current
    blobs = {}
    for i, (y, a, b) in enumerate(runs):
        stats = blobs.setdefault(root(i), [0, 0., 0., 0., a, b, y, y])
        n = b-a+1
        stats[0] += n
        stats[1] += n*(.5*(a+b)+.5)
        stats[2] += n*(y+.5)
        stats[3] += error[y, a:b+1].sum()
        stats[4], stats[5] = min(a, stats[4]), max(b, stats[5])
        stats[6], stats[7] = min(y, stats[6]), max(y, stats[7])
    result = []
    for n, sx, sy, total_error, a, b, top, bottom in blobs.values():
        if 3 <= n <= area_limit and n / ((b-a+1)*(bottom-top+1)) > .25:
            result.append((np.array([sx/n, sy/n]), n, total_error/n))
    result.sort(key=lambda row: row[2] - min(4., np.log(row[1])))
    return result[:limit]


def detections(image, markers):
    hue, chroma, hi = _hue(image)
    nominal = np.array([m["hue_deg"] for m in markers])
    distances = np.abs((hue[:, :, None] - nominal + 180) % 360 - 180)
    nearest = distances.argmin(axis=2)
    saturated = (chroma > 28) & (chroma > .40*hi) & (hi > 42)
    # Orange payload and the green-blue goal outline are broad unlabelled surfaces.
    reject = (np.abs((hue-25+180) % 360-180) < 11) | (np.abs((hue-165+180) % 360-180) < 10)
    result = {}
    for i, marker in enumerate(markers):
        error = distances[:, :, i]
        mask = saturated & ~reject & (nearest == i) & (error <= 17.)
        if mask.any():
            blobs = _components(mask, error, .085*image.shape[0]*image.shape[1],
                                limit=8 if marker["role"] == "goal" else 3)
            if blobs:
                result[marker["id"]] = blobs
    return result


def _project(pose, key, obs):
    r, t = pose
    local, cam, candidates = obs[key]
    eye, w2c, f, principal = cam
    rel = r @ local
    p = w2c @ (rel + t-eye)
    depth = max(.006, p[2])
    xy = p[:2] / depth
    pixel = principal + f*xy
    idx = np.argmin([np.linalg.norm(c[0]-pixel) for c in candidates])
    observed = candidates[idx][0]
    jac = f/depth * np.array([[1., 0., -xy[0]], [0., 1., -xy[1]]]) @ w2c
    return pixel-observed, jac @ np.column_stack((np.eye(3), -_skew(rel))), p[2]


def _ray(cam, uv):
    eye, w2c, f, principal = cam
    ray = w2c.T @ np.r_[(uv-principal)/f, 1.]
    return eye, ray/np.linalg.norm(ray)


def _score(pose, obs):
    errors = []
    for k in obs:
        err, _, depth = _project(pose, k, obs)
        errors.append(np.linalg.norm(err) if depth > .012 else 1e3)
    return np.array(errors)


def _rank(pose, obs):
    err = _score(pose, obs)
    return float(np.minimum(err, 8.).mean() + .35*np.median(np.minimum(err, 12.)))


def _refine(pose, obs):
    r, t = pose
    for _ in range(6):
        rows = [_project((r, t), k, obs) for k in obs]
        residual, jac, depth = (np.asarray(v) for v in zip(*rows, strict=True))
        weights = np.minimum(1., 1.6 / np.maximum(np.linalg.norm(residual, axis=1), .001))
        weights[depth < .012] = 0.
        jac = jac.reshape(-1, 6)
        wj = weights.repeat(2)[:, None]*jac
        update = np.linalg.solve(jac.T@wj + np.diag([2., 2., 2., .7, .7, .7]),
                                 -wj.T @ residual.ravel())
        t = t + np.clip(update[:3], -.030, .030)
        w = update[3:]
        r = _exp(w*min(1., .24/max(np.linalg.norm(w), .001))) @ r
        if np.linalg.norm(update[:3]) < .00015 and np.linalg.norm(w) < .002:
            break
    return r, t


def _object_pose(obs, previous):
    distinct = {k[1] for k in obs}
    if not obs or (len(distinct) < 2 and previous is None):
        return None
    proposals = []
    if previous is not None:
        proposals.append(previous)
    # Triangulate labels shared by both eyes. Perfect contact with the workpiece
    # is neither assumed nor used as an alternative pose measurement.
    triangulated = []
    for marker in distinct:
        ka, kb = (0, marker), (1, marker)
        if ka not in obs or kb not in obs:
            continue
        local, a, pixels_a = obs[ka]
        _, b, pixels_b = obs[kb]
        best = None
        for uv, _, _ in pixels_a:
            ea, ra = _ray(a, uv)
            for other, _, _ in pixels_b:
                eb, rb = _ray(b, other)
                if np.linalg.norm(np.cross(ra, rb)) < .09:
                    continue
                dist = np.linalg.lstsq(np.column_stack((ra, -rb)), eb-ea, rcond=None)[0]
                if np.min(dist) < .018 or np.max(dist) > 1.4:
                    continue
                p, q = ea+dist[0]*ra, eb+dist[1]*rb
                error = np.linalg.norm(p-q)
                if best is None or error < best[0]:
                    best = error, .5*(p+q)
        if best is not None and best[0] < .009:
            triangulated.append((local, best[1]))
    if len(distinct) == 1:
        if not triangulated or previous is None:
            return None
        local, world = triangulated[0]
        pose = previous[0], world - previous[0] @ local
        error = float(np.max(_score(pose, obs)))
        return (pose, error, 2, False) if error < 2.5 else None
    for group in combinations(triangulated, 3):
        a, b = (np.asarray(v) for v in zip(*group, strict=True))
        centered = a - a.mean(axis=0)
        if np.linalg.matrix_rank(centered, tol=.002) < 2:
            continue
        u, _, vh = np.linalg.svd(centered.T @ (b-b.mean(axis=0)))
        flip = np.diag([1., 1., np.linalg.det(vh.T@u.T)])
        r = vh.T @ flip @ u.T
        proposals.append((r, b.mean(axis=0) - r@a.mean(axis=0)))
    # Multiple calibrated mono correspondences, including different labels across
    # the two wrists, can initialize a 3D pose when no single label triangulates.
    if len(distinct) >= 3 and (previous is None or not proposals or _rank(previous, obs) > 3.):
        eyes, rays, locals_ = [], [], []
        for local, cam, blobs in obs.values():
            eye, ray = _ray(cam, blobs[0][0])
            eyes.append(eye)
            rays.append(ray)
            locals_.append(local)
        rays = np.array(rays)
        projectors = np.eye(3)[None] - rays[:, :, None]*rays[:, None, :]
        normal = projectors.sum(axis=0)
        if np.linalg.cond(normal) < 1e5:
            for yaw in np.linspace(-np.pi, np.pi, 12, endpoint=False):
                r = _yaw(yaw)
                rhs = np.einsum("nij,nj->i", projectors, np.asarray(eyes)-np.asarray(locals_)@r.T)
                proposals.append((r, np.linalg.solve(normal, rhs)))
    if not proposals:
        return None
    guesses = sorted(proposals, key=lambda p: _rank(p, obs))[:3]
    best = min((_refine(p, obs) for p in guesses), key=lambda p: _rank(p, obs))
    errors = _score(best, obs)
    inliers = [k for k, error in zip(obs, errors, strict=True) if error < 3.5]
    if len(inliers) < 3 or len({k[1] for k in inliers}) < 2:
        return None
    points = np.array([obs[k][0] for k in inliers])
    full = (len({k[1] for k in inliers}) >= 3 and np.linalg.matrix_rank(points-points.mean(axis=0), tol=.002) >= 2)
    if not full:
        if previous is None or len(triangulated) < 2:
            return None
        # With only two reconstructed labels the orientation around their
        # baseline is unobservable. Retain the previous orientation explicitly.
        r = previous[0]
        t = np.median([b-r@a for a, b in triangulated], axis=0)
        best = r, t
        if np.median(_score(best, obs)) > 3.5:
            return None
    return best, float(np.median(_score(best, obs))), len(inliers), full


def _cross2(a, b):
    return a[..., 0]*b[..., 1] - a[..., 1]*b[..., 0]


def _floor_disc(cam, uv, area_px, radius, z):
    """Intersect a measured ink centroid with the calibrated tabletop, in metres."""
    eye, ray = _ray(cam, uv)
    if ray[2] >= -.07:
        return None
    distance = (z-eye[2])/ray[2]
    if not .02 < distance < 1.2:
        return None
    point = eye + distance*ray
    depth = (cam[1] @ (point-eye))[2]
    if depth < .02:
        return None
    # The centre of a projected ellipse is not the projection of the disc
    # centre. In floor coordinates the inverse image of the ellipse centre is
    # centre - r²*g/Z, where g is the horizontal camera-depth gradient and Z
    # the true centre depth (the pole of the image line at infinity).
    g = cam[1][2, :2]
    spread = radius**2 * (g @ g)
    center_depth = .5*(depth + np.sqrt(depth**2 + 4*spread))
    center_xy = point[:2] + radius**2*g/center_depth
    # Exact area of the projected circle. The fraction check still allows
    # partial visibility, including occlusion at the robot and payload edges.
    expected = np.pi*radius**2 * cam[2]**2 * abs(eye[2]-z) / (center_depth**2-spread)**1.5
    fraction = area_px / expected
    if not .23 <= fraction <= 1.85:
        return None
    return center_xy, abs(np.log(fraction))


class _FloorCluster:
    """Distinct exposures of one coloured island at a fixed table coordinate."""

    def __init__(self, xy):
        self.xy = np.array(xy)
        self.samples = {}

    def add(self, frame, eye, xy, cost):
        key = (frame, eye)
        if key in self.samples and self.samples[key][1] <= cost:
            return
        self.samples[key] = (xy, cost)
        if len(self.samples) > 48:
            del self.samples[next(iter(self.samples))]
        self.xy = np.median([p for p, _ in self.samples.values()], axis=0)

    @property
    def last(self):
        return next(reversed(self.samples))[0]

    @property
    def exposures(self):
        return len({key[0] for key in self.samples})


def _triangle_fit(local, world):
    """Recover proper planar rotation from all three colour-labelled centres."""
    a, b = local-local.mean(axis=0), world-world.mean(axis=0)
    edges = np.roll(world, -1, axis=0) - world
    expected_edges = np.roll(local, -1, axis=0) - local
    if np.max(np.abs(np.linalg.norm(edges, axis=1)-np.linalg.norm(expected_edges, axis=1))) > .011:
        return None
    # The off-axis rose label distinguishes the two sides of the baseline.
    signed_area_ratio = _cross2(edges[0], edges[1]) / _cross2(expected_edges[0], expected_edges[1])
    if not .78 <= signed_area_ratio <= 1.24:
        return None
    angle = np.arctan2(_cross2(a, b).sum(), (a*b).sum())
    # Both workcells transport forwards in +X. This broad fixture mounting
    # envelope admits lateral displacements and either sign of task yaw.
    if abs(angle) > np.deg2rad(80.):
        return None
    r = _yaw(angle)
    xy = world.mean(axis=0) - r[:2, :2] @ local.mean(axis=0)
    residual = np.linalg.norm(local @ r[:2, :2].T + xy-world, axis=1)
    if residual.max() > .008:
        return None
    return (r, np.r_[xy, .001]), float(np.sqrt(np.mean(residual**2)))


def _same_goal(a, b, metres, radians):
    return (np.linalg.norm(a[1][:2]-b[1][:2]) <= metres and
            a[0][:2, 0] @ b[0][:2, 0] >= np.cos(radians))


class FiducialTracker:
    def __init__(self, task):
        from bimanual_wrist_render import marker_layout

        self.task = task
        self.markers = tuple(marker_layout(task))
        self.ids = {m["id"]: m for m in self.markers}
        self.pose = self.goal = None
        self.goal_names = tuple(m["id"] for m in self.markers if m["role"] == "goal")
        self.goal_local = np.array([self.ids[k]["xyz"][:2] for k in self.goal_names])
        self.goal_samples = {k: [] for k in self.goal_names}
        self.goal_frame = self.goal_streak = 0
        self.goal_pending = self.goal_origin = self.goal_last_seen = None
        self.goal_quality = {}
        self.age = self.rotation_age = 1000
        self.diagnostics = {}

    def _goal(self, obs):
        self.goal_frame += 1
        frame = self.goal_frame
        if self.goal_origin is None and self.rotation_age == 0:
            self.goal_origin = self.pose[1][:2].copy()
        for name, clusters in self.goal_samples.items():
            clusters[:] = [c for c in clusters if frame-c.last <= 90]
        for (eye, name), (local, cam, blobs) in obs.items():
            clusters = self.goal_samples[name]
            for uv, area, hue_error in blobs:
                hit = _floor_disc(cam, uv, area, self.ids[name]["radius_m"], .001+local[2])
                if hit is None:
                    continue
                xy, area_error = hit
                near = min(clusters, key=lambda c: np.linalg.norm(c.xy-xy), default=None)
                if near is None or np.linalg.norm(near.xy-xy) > .008:
                    near = _FloorCluster(xy)
                    clusters.append(near)
                near.add(frame, eye, xy, area_error + hue_error/17.)

        choices = []
        for name in self.goal_names:
            clusters = self.goal_samples[name]
            clusters.sort(key=lambda c: min(12, c.exposures) - (frame-c.last)/18., reverse=True)
            del clusters[16:]
            choices.append([c for c in clusters if c.exposures >= 2][:8])
        fits = []
        for triangle in product(*choices):
            if all(c.last != frame for c in triangle):
                continue
            fit = _triangle_fit(self.goal_local, np.array([c.xy for c in triangle]))
            if fit is None:
                continue
            pose, rms = fit
            if self.goal_origin is not None:
                dx, dy = pose[1][:2]-self.goal_origin
                if not -.03 <= dx <= .42 or abs(dy) > .21:
                    continue
            support = min(c.exposures for c in triangle)
            age = max(frame-c.last for c in triangle)
            cost = rms + .0015/min(8, support) + .001*age/90.
            fits.append((cost, pose, rms, triangle))
        fits.sort(key=lambda f: f[0])
        reason = "insufficient triangle evidence"
        if fits:
            cost, pose, rms, triangle = fits[0]
            ambiguous = any(other_cost < cost+.0015 and
                            not _same_goal(pose, other_pose, .014, np.deg2rad(8.))
                            for other_cost, other_pose, _, _ in fits[1:])
            if ambiguous:
                reason = "multiple plausible triangles"
            else:
                if self.goal_pending is None or not _same_goal(self.goal_pending, pose, .010, np.deg2rad(5.)):
                    self.goal_streak = 0
                self.goal_streak += 1
                self.goal_pending = pose
                if self.goal_streak >= 3:
                    self.goal, self.goal_last_seen = pose, frame
                    reason = "confirmed"
                else:
                    reason = "awaiting repeat exposures"
        if not fits or reason == "multiple plausible triangles":
            self.goal_pending, self.goal_streak = None, 0
        self.goal_quality = dict(status=reason, coasting=self.goal is not None and self.goal_last_seen != frame,
                                 confirmed_age=None if self.goal_last_seen is None else frame-self.goal_last_seen,
                                 streak=self.goal_streak, fit_count=len(fits))
        if fits:
            self.goal_quality.update(rms_m=rms, support_frames=[c.exposures for c in triangle],
                                     label_ages=[frame-c.last for c in triangle])
        return self.goal

    def observe(self, rgb, cameras):
        h, w = rgb.shape[1:3]
        focal = .5*h/np.tan(np.deg2rad(39.))
        groups = {"object": {}, "goal": {}}
        counts = np.zeros((2, 2), int)
        labels = []
        for i, (img, (eye, rotation)) in enumerate(zip(rgb, cameras, strict=True)):
            cam = (eye, np.diag([1., -1., -1.])@rotation.T, focal, np.array([w/2, h/2]))
            detected = detections(img, self.markers)
            labels.append(sorted(detected))
            for k, blobs in detected.items():
                marker = self.ids[k]
                role = marker["role"]
                groups[role][i, k] = np.array(marker["xyz"]), cam, blobs
                counts[i, int(role == "goal")] += sum(b[1] for b in blobs)
        previous = self.pose if self.age < 9 else None
        measured = _object_pose(groups["object"], previous)
        self.age += 1
        self.rotation_age += 1
        object4 = error = None
        if measured:
            pose, error, inliers, full = measured
            r, t = pose
            if (previous is None or np.linalg.norm(t-previous[1]) < .055 or self.age >= 5) and (full or self.rotation_age <= 6):
                self.pose, self.age = pose, 0
                if full:
                    self.rotation_age = 0
                boxes = [(0., .06, .11, .05)] if self.task == "carry" else [(0., .028, .13, .028), (.106, .078, .033, .028)]
                bottom = min(t[2] + r[2, 0]*dx - np.abs(r[2]) @ [x,y,z] for dx,x,y,z in boxes)
                object4 = np.r_[t[:2], np.arctan2(r[1,0], r[0,0]), bottom]
        goal = self._goal(groups["goal"])
        goal4 = None if goal is None else np.r_[goal[1][:2], np.arctan2(goal[0][1,0], goal[0][0,0]), 0.]
        self.diagnostics = dict(visible_labels=labels, object_age=self.age, attitude_age=self.rotation_age,
                                inliers=0 if measured is None else measured[2], reprojection_px=error,
                                center_xyz=None if self.pose is None else self.pose[1].tolist(),
                                rotation=None if self.pose is None else self.pose[0].tolist(),
                                goal_label_count=sum(bool(v) for v in self.goal_samples.values()),
                                goal_triangle=self.goal_quality)
        return object4, goal4, error, counts.tolist()


def _toward(current, goal, maximum):
    delta = goal-current
    return current + delta*min(1., maximum/max(np.linalg.norm(delta), 1e-9))


class FiducialControl:
    """Bilateral operational-space impedance and force feedback, with bounded TCP motion."""

    def __init__(self, task, dt):
        self.tracker = FiducialTracker(task)
        self.task, self.dt = task, dt
        self.phase = "acquire"
        self.ref = self.last_position = self.last_yaw = self.initial_tool_rot = None
        self.speed = np.zeros(3)
        self.spin = self.grip_time = self.lift_feed = 0.
        self.start_xy = self.floor_center_z = None
        self.opening = 0.
        self.carry_trim_xy = np.zeros(2)
        self.last_action = None
        self.features = {}

    def _carry_fixture(self, model):
        # Only body-local manufactured dimensions and the nameplate mass are
        # read here. Dynamic object/goal state comes from the wrist images.
        body = model.body("payload").id
        box = model.geom("payload_0").id
        pad = next(i for i in range(model.ngeom) if model.geom(i).name in ("left_pad", "left_pad_a"))
        self.pad_radius = float(model.geom_size[pad, 0])
        self.box_half_y = float(model.geom_size[box, 1])
        self.weight = float(model.body_mass[body])*9.81
        ledges = []
        for i in np.flatnonzero(model.geom_bodyid == body):
            p, half = model.geom_pos[i], model.geom_size[i]
            if (i != box and np.allclose(model.geom_quat[i], (1.,0.,0.,0.))
                    and abs(abs(p[1])-half[1]-self.box_half_y) < .002 and p[2]-half[2] > 0.):
                ledges.append((p[1], p[2]-half[2], abs(p[1])+half[1]))
        self.lipped = bool(len(ledges) == 2 and ledges[0][0]*ledges[1][0] < 0.)
        self.seat_z = float(min(row[1] for row in ledges)-self.pad_radius) if self.lipped else 0.
        self.outer_y = float(min(row[2] for row in ledges)) if self.lipped else self.box_half_y

    def _carry(self, actor, tools, wrist, position, r, yaw, obj, goal, gains, fresh):
        grip, clearance, travel, _, turning, arrival, _ = gains
        dt, signs = self.dt, np.array([1.,-1.])
        side = _yaw(yaw)[:,1]
        forward = _yaw(yaw)[:,0]
        if not hasattr(self, "grasp_yaw"):
            self._carry_fixture(actor.model)
            self.grasp_yaw = yaw
            self.filtered_wrist = wrist.copy()
            self.missing_grip_s = self.unload_time = self.pressure = 0.
            self.grasp_tilt = r[2,:2].copy()
            self.tilt_rate = np.zeros(2)
            self.lip_seated = self.regrasp = False
        self.filtered_wrist += (1.-np.exp(-dt/.036))*(wrist-self.filtered_wrist)
        normal = np.maximum(0., signs*(self.filtered_wrist @ side))
        load = np.maximum(0., -self.filtered_wrist[:,2])
        tilt = r[2,:2]
        if fresh:
            self.tilt_rate = .76*self.tilt_rate + .24*np.clip((tilt-self.grasp_tilt)/dt,-1.8,1.8)
            self.grasp_tilt = tilt.copy()
        else:
            self.tilt_rate *= .90
        half = self.box_half_y+self.pad_radius
        local = (tools-position) @ r
        aligned = np.max(abs(local[:,0])) < .021 and np.max(abs(local[:,2]-self.seat_z)) < .020
        under_ledge = (self.lipped and np.all(abs(local[:,2]-self.seat_z) < .007)
                       and np.all(signs*local[:,1] < self.outer_y+.003))
        if under_ledge and np.min(load) > .12*self.weight:
            self.lip_seated = True
        bottom = None if obj is None else float(obj[3])
        tilt_ok = np.linalg.norm(tilt) < .16
        if self.phase == "acquire" and fresh:
            self.phase = "clamp"
        if self.phase == "clamp":
            threshold = 7. if self.lipped else 17.
            self.grip_time = self.grip_time+dt if fresh and aligned and np.min(normal)>threshold else 0.
            if self.grip_time >= .048:
                self.phase = "lift"
                self.ref = position.copy()
        if (self.phase == "lift" and fresh and goal is not None and bottom >= .040 and tilt_ok
                and np.min(load) > .20*self.weight):
            self.phase = "transport"
        if (self.phase == "transport" and fresh and goal is not None and tilt_ok
                and np.linalg.norm(goal[:2]-position[:2]) < np.clip(.018*arrival,.013,.021)
                and np.linalg.norm(position[:2]-self.start_xy) > .069):
            self.phase = "lower"
        grasp_lost = min(load) < 3. and np.max(abs(local[:,2]-self.seat_z)) > .025
        self.missing_grip_s = self.missing_grip_s+dt if fresh and grasp_lost else 0.
        if self.phase in ("lift","transport") and self.missing_grip_s > .13:
            self.phase, self.regrasp = "lower", True
            self.ref[:2] = position[:2]
        if self.phase == "lower" and fresh and bottom < .006 and abs(self.speed[2]) < .065:
            self.phase = "unload"
        if self.phase == "unload":
            supported = fresh and bottom < .009 and load.sum() < .18*self.weight and abs(self.speed[2]) < .045
            self.unload_time = self.unload_time+dt if supported else 0.
            if self.unload_time >= .048:
                self.phase = "clamp" if self.regrasp else "release"
                self.lip_seated = self.regrasp = False
                self.grip_time = self.unload_time = 0.

        holding = self.phase in ("lift","transport","lower","unload")
        # Anchor an in-progress lift in world XY. Following the expelled part
        # sideways while squeezing both wrists made early grasp loss accelerate.
        vref = np.zeros(3)
        if holding and fresh:
            previous = self.ref.copy()
            wanted_z = self.floor_center_z + np.clip(.065*clearance,.055,.078)
            lower = self.phase in ("lower","unload")
            self.ref[2] = float(_toward(self.ref[2], self.floor_center_z-.005 if lower else wanted_z,
                                      (.093 if lower else .12)*dt))
            if self.phase == "transport" and goal is not None and tilt_ok:
                self.ref[:2] = _toward(self.ref[:2], goal[:2], np.clip(.15*travel,.09,.18)*dt)
                self.ref[:2] = _toward(position[:2], self.ref[:2], .034)
            vref = (self.ref-previous)/dt
        if self.phase == "clamp" and fresh:
            self.ref = position.copy()
        correction = np.clip(self.ref-position, [-.06,-.045,-.05], [.06,.045,.06])
        angle_ref = goal[2] if goal is not None and self.phase in ("transport","lower") else self.grasp_yaw
        angle = float(np.arctan2(np.sin(angle_ref-yaw),np.cos(angle_ref-yaw)))
        # Shift the upward resultant toward the low end. Both front/rear pad
        # pairs can thereby support pitch even when elbow torque is limiting.
        offset_x = np.clip(-.068*tilt[0]-.012*self.tilt_rate[0], -.022,.022) if holding else 0.
        spread = half - .001
        target = position + offset_x*forward + self.seat_z*r[:,2] + signs[:,None]*spread*side
        if self.phase == "clamp":
            target[:,2] += signs*half*tilt[1] - (.002 if self.lipped else 0.)
        pressure = np.clip(31.*grip,26.,39.)
        if self.lipped:
            pressure = np.clip((10. if self.lip_seated else 19.)*grip, 7.,24.)
        if self.phase == "clamp" and not aligned:
            pressure = 6.
        if self.phase == "unload":
            pressure = min(pressure, 5.)
            target[:,2] -= .008
        if self.phase == "release":
            pressure = 0.
            self.opening = min(.055,self.opening+.19*dt)
            target += signs[:,None]*(self.opening+.001)*side
            target[:,2] -= .009
        if self.tracker.age > 6 and not holding:
            pressure = 0.
            target = tools.copy()
        self.pressure = float(_toward(self.pressure, pressure, (200. if pressure>self.pressure else 260.)*dt))
        drive = np.clip(self.pressure+.22*(self.pressure-normal),0.,46.)
        if self.phase == "clamp" and np.min(normal)<4.:
            drive = np.where(normal>8., np.minimum(drive,7.), drive)
        effort = -signs[:,None]*drive[:,None]*side
        feed = self.weight*.5 if holding else 0.
        if self.phase == "unload":
            feed = 0.
        self.lift_feed = float(_toward(self.lift_feed, feed, (135. if feed>self.lift_feed else 155.)*dt))
        effort[:,2] = self.lift_feed
        if holding:
            trim = np.clip(-signs*(33.*tilt[1]+4.5*self.tilt_rate[1]),-9.,9.)
            zservo = np.clip(230.*correction[2]+12.*(vref[2]-self.speed[2]),-10.,10.)
            if self.phase == "unload":
                zservo = min(0.,zservo)
                trim *= .25
            effort[:,2] += zservo + trim
            # The pitch controller shifts both pads along the box. This can
            # leave a static offset in the centre-of-mass position at the goal.
            # Integrate common-mode position error only while both wrists are
            # carrying and a current optical measurement is available.
            if self.phase in ("transport", "lower") and fresh and min(load) > .15*self.weight:
                self.carry_trim_xy = np.clip(self.carry_trim_xy + 115.*dt*correction[:2], -8., 8.)
            else:
                self.carry_trim_xy *= np.exp(-dt/.14)
            effort[:,:2] += np.clip(350.*correction[:2]+self.carry_trim_xy
                                    +20.*(vref[:2]-self.speed[:2]),-16.,16.)
            effort -= signs[:,None]*np.clip(21.*turning*angle-2.1*self.spin,-6.,6.)*forward
        self.carry_diag = dict(lipped=self.lipped, lip_seated=bool(self.lip_seated), load_n=load.tolist(),
                               tilt_sines=tilt.tolist(), fore_aft_offset_m=float(offset_x),
                               pressure_n=self.pressure, reference_xyz_m=self.ref.tolist(),
                               transport_integral_n=self.carry_trim_xy.tolist())
        return target, effort, angle

    def apply(self, actor, proprio, rgb, tools, cameras, wrist):
        mj, data, model = actor.mj, actor.data, actor.model
        if wrist is None:
            raise ValueError("fiducial policy needs the two actual wrist-force observations")
        wrist = np.asarray(wrist, float)
        if wrist.shape != (2,3) or not np.isfinite(wrist).all():
            raise ValueError("invalid wrist wrench observation")
        obj, goal, error, counts = self.tracker.observe(rgb, cameras)
        pose = self.tracker.pose
        fresh = obj is not None
        dt = self.dt
        gains = np.clip(actor.gains, .55, 1.8)
        grip_gain, clearance_gain, speed_gain, alignment, rotation_gain, arrival_gain, damping = gains
        left_right = np.array([1., -1.])
        q = proprio[:14].reshape(2,7)
        dq = proprio[14:28].reshape(2,7)
        active = proprio[28:].reshape(2,7)
        mj.mj_comVel(model, data)
        gravity = np.zeros(model.nv)
        mj.mj_rne(model, data, 0, gravity)
        if self.initial_tool_rot is None:
            self.initial_tool_rot = [data.site_xmat[s].reshape(3,3).copy() for s, _ in actor.sites]
            self.initial_q = q.copy()
        if pose is not None:
            r, position = pose
            yaw = float(np.arctan2(r[1,0], r[0,0]))
            if self.last_position is not None and fresh:
                self.speed = .65*self.speed + .35*np.clip((position-self.last_position)/dt, -.55, .55)
                da = yaw - self.last_yaw
                self.spin = .7*self.spin + .3*float(np.clip(np.arctan2(np.sin(da),np.cos(da))/dt, -3., 3.))
            if fresh:
                self.last_position, self.last_yaw = position.copy(), yaw
            if self.start_xy is None:
                self.start_xy = position[:2].copy()
                self.floor_center_z = .05 if self.task == "carry" else .028
                self.ref = position.copy()
        else:
            r, position, yaw = np.eye(3), tools.mean(axis=0), 0.
        yaxis = _yaw(yaw)[:,1]
        xaxis = _yaw(yaw)[:,0]
        compression = np.maximum(0., left_right*(wrist @ yaxis))
        target = tools.copy()
        effort = np.zeros_like(tools)
        fresh_motion = fresh and self.tracker.rotation_age < 7
        goal_error = np.zeros(2) if goal is None else goal[:2]-position[:2]
        angle = 0. if goal is None else goal[2]-yaw
        angle = np.arctan2(np.sin(angle), np.cos(angle))
        fitted_bottom = None if obj is None else float(obj[3])
        if pose is not None and self.task == "carry":
            target, effort, angle = self._carry(actor, tools, wrist, position, r, yaw, obj, goal, gains, fresh_motion)
        elif pose is not None and self.task == "push_t":
            self.phase = "push" if goal is not None else "acquire"
            # The striking points are relative to the observed rotating crossbar.
            back = position - (.028+.026)*xaxis
            target = back[None] + (.081*left_right[:,None])*yaxis
            target[:,2] = self.floor_center_z+.001
            if fresh_motion and goal is not None:
                planar_error = np.r_[goal_error, 0.]
                forward_error = planar_error @ xaxis
                vforward = self.speed @ xaxis
                turn = np.clip(rotation_gain*(88.*angle - 7.*self.spin), -32., 32.)
                moving = np.clip((forward_error-.003)/.025, 0., 1.)
                thrust = moving*(25.+4.*speed_gain) + 95.*np.clip(forward_error,0.,.055) - 17.*vforward
                # Positive yaw requires the -Y pad to push harder. Near the
                # endpoint allow one-sided differential pressure to finish the turn.
                if np.linalg.norm(goal_error)<.013 and abs(angle)<.078:
                    self.phase = "settle"
                    thrust = turn = 0.
                    target -= .012*xaxis
                drive = np.clip(thrust-left_right*turn, 0., 60.)
                effort += drive[:,None]*xaxis
                effort += np.clip(165.*(planar_error @ yaxis)-9.*(self.speed @ yaxis), -11., 11.)*yaxis
        result = np.zeros((2, 7))
        carries_load = self.task == "carry" and pose is not None and self.phase in ("lift","transport","lower","unload")
        proposals = []
        for i, ((qa, va, _), (tool, _)) in enumerate(zip(actor.joints, actor.sites, strict=True)):
            n = len(qa)
            jp, jr = np.zeros((3,model.nv)), np.zeros((3,model.nv))
            mj.mj_jacSite(model, data, jp, jr, tool)
            j, rot_j = jp[:,va], jr[:,va]
            vtip = j @ dq[i,:n]
            force = np.clip(650.*alignment*(target[i]-tools[i]) - 31.*damping*vtip + effort[i], -78., 78.)
            null = np.eye(n) - j.T@np.linalg.solve(j@j.T+.0004*np.eye(3), j)
            desired_r = (self.initial_tool_rot[i] if self.task == "carry"
                         else _yaw(np.clip(yaw,-.55,.55)) @ self.initial_tool_rot[i])
            current = data.site_xmat[tool].reshape(3,3)
            orientation = .5 * sum((np.cross(current[:,k], desired_r[:,k]) for k in range(3)), np.zeros(3))
            neutral = self.initial_q[i,:n] if self.task == "carry" else 0.
            posture = rot_j.T@(2.2*orientation-.32*(rot_j@dq[i,:n])) - 1.1*(q[i,:n]-neutral)
            start = sum(len(x[0]) for x in actor.joints[:i])
            lo, hi = model.actuator_ctrlrange[start:start+n].T
            if carries_load:
                force[2] = effort[i,2] + np.clip(110.*(target[i,2]-tools[i,2]) - 5.*vtip[2], -2.5, 2.5)
                # Reduce a limiting elbow moment through the available redundant
                # posture. Differentiate a dot (r cross F) through each real hinge.
                joint_ids = model.dof_jntid[va]
                axes, pivots = data.xaxis[joint_ids], data.xanchor[joint_ids]
                elbow = 2
                lever = tools[i]-pivots[elbow]
                elbow_load = gravity[va[elbow]] + force @ j[:,elbow]
                limit = hi[elbow] if elbow_load >= 0 else -lo[elbow]
                gradient = np.empty(n)
                for k in range(n):
                    da = np.cross(axes[k],axes[elbow]) if k < elbow else np.zeros(3)
                    dc = np.cross(axes[k],pivots[elbow]-pivots[k]) if k < elbow else np.zeros(3)
                    gradient[k] = force @ (np.cross(da,lever) + np.cross(axes[elbow],j[:,k]-dc))
                utilization = np.clip((abs(elbow_load)/limit-.72)/.28, 0., 1.)
                posture -= 3.0*utilization*np.sign(elbow_load)*gradient/limit
                base_torque = gravity[va] + null@posture - .45*damping*dq[i,:n]
                leveled = np.clip(3.7*np.cross(r[:,2], [0.,0.,1.])
                                  + .48*(self.tilt_rate[0]*yaxis-self.tilt_rate[1]*xaxis), -.78,.78)
                # Keep the minimum bilateral lateral seating force. Under elbow
                # saturation first trade some steering/levelling effort for lift.
                preload = .82*(force[:2]@yaxis[:2])*yaxis[:2]
                essential = base_torque + j[:2].T@preload
                secondary = rot_j.T@leveled + j[:2].T@(force[:2]-preload)
                slopes = j[2]
                affected = np.abs(slopes) > 1e-6
                best = None
                for fraction in (1.,.75,.50,.25):
                    horizontal = essential + fraction*secondary
                    bound1 = (lo[affected]-horizontal[affected])/slopes[affected]
                    bound2 = (hi[affected]-horizontal[affected])/slopes[affected]
                    low = max(0., np.minimum(bound1,bound2).max(initial=0.))
                    high = min(55., np.maximum(bound1,bound2).min(initial=55.))
                    if np.any((horizontal[~affected]<lo[~affected]) | (horizontal[~affected]>hi[~affected])):
                        high = -1.
                    gap = max(0., low-force[2], force[2]-high)
                    score = gap + 1.2*(1.-fraction) + (100. if low>high else 0.)
                    if best is None or score < best[0]:
                        best = score,horizontal,low,high
                _,horizontal_torque,zmin,zmax = best
                proposals.append((i, n, horizontal_torque, slopes, lo, hi, force[2], zmin, zmax))
                continue
            tau = gravity[va] + j.T@force + null@posture - .45*damping*dq[i,:n]
            result[i,:n] = active[i,:n]*np.clip(tau/np.where(tau>=0,hi,-lo),-1.,1.)
        allocation = None
        if proposals:
            requested = np.array([p[6] for p in proposals])
            bounds = np.array([p[7:] for p in proposals])
            feasible = bool(np.all(bounds[:,0] <= bounds[:,1]))
            if feasible:
                # A persistent excess lift on the already high side twists the
                # payload out of its grasp. Keep the feasible differential near
                # the visual roll-servo request, even if that reduces total lift.
                balance = requested[0]-requested[1]
                max_balanced = min(2.*bounds[1,1]+balance+4., 2.*bounds[0,1]-balance+4.)
                total = np.clip(requested.sum(), bounds[:,0].sum(),
                                max(bounds[:,0].sum(),min(bounds[:,1].sum(),max_balanced)))
                low = max(bounds[0,0], total-bounds[1,1])
                high = min(bounds[0,1], total-bounds[1,0])
                left = np.clip(.5*(total+requested[0]-requested[1]),low,high)
                assigned = (left,total-left)
            else:
                # Fixed-horizontal force cannot be realized at this posture.
                # Preserve the physical motor clipping and mark the event.
                assigned = requested
            for p, vertical in zip(proposals, assigned, strict=True):
                i,n,horizontal_torque,slopes,lo,hi = p[:6]
                tau = horizontal_torque + slopes*vertical
                result[i,:n] = active[i,:n]*np.clip(tau/np.where(tau>=0,hi,-lo),-1.,1.)
            allocation = dict(feasible_intervals=feasible, vertical_intervals_n=bounds.tolist(),
                              requested_n=requested.tolist(), assigned_n=list(map(float,assigned)))
        if self.task == "carry":
            if self.last_action is not None:
                filtered = (1.-np.exp(-dt/.009))*(result-self.last_action)
                result = self.last_action + np.clip(filtered,-12.*dt,12.*dt)
            result = active*np.clip(result,-1.,1.)
            self.last_action = result.copy()
        self.features = dict(variant="fiducial", phase=self.phase, pixel_counts=counts, **self.tracker.diagnostics,
                             object_xy=None if pose is None else position[:2].tolist(),
                             goal_xy=None if goal is None else goal[:2].tolist(), goal_yaw=None if goal is None else float(goal[2]),
                             image_yaw_error=float(angle), object_bottom_z=fitted_bottom,
                             pressure_observed_n=compression.tolist(), lift_feed_n=self.lift_feed,
                             requested_force_world_n=effort.tolist(), tool_heights=tools[:,2].tolist(),
                             wrist_force_world_n=wrist.tolist(), fresh_pose=fresh_motion,
                             carry_vertical_allocation=allocation, carry_grasp=getattr(self,"carry_diag",None))
        return result.ravel()
