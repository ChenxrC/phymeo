"""Fit the known workpiece/goal outlines to two calibrated colour silhouettes."""

import numpy as np


_ANGLES = np.arange(12) * np.pi / 6
_DIRECTIONS = np.stack((np.cos(_ANGLES), np.sin(_ANGLES)), axis=1)


def template(task, is_goal):
    boxes = [(0., .060, .110, .100)] if task == "carry" else [(0., .028, .130, .056), (.106, .078, .033, .056)]
    return np.array([(offset + sx * half_x, sy * half_y, z)
                     for offset, half_x, half_y, top in boxes for sx in (-1., 1.) for sy in (-1., 1.)
                     for z in ((.002,) if is_goal else (0., top))])


def outline(image, eye, rotation):
    r, g, b = np.moveaxis(image.astype(float), -1, 0)
    warm = (r > 1.20 * g + 5) & (r > 1.26 * b + 5)
    goal = (g > 1.15 * r + 4) & (b > 1.08 * r + 3) & (g > b + 1)
    rows, cols = image.shape[:2]
    focal = .5 * rows / np.tan(np.deg2rad(39.))
    result = []
    for is_goal, mask in enumerate((warm, goal)):
        y, x = np.nonzero(mask)
        if len(x) < 5:
            result.append(None)
            continue
        # Normalized image axes: rightward and upward. Store the extreme
        # support in 12 directions, discounting the extreme one or two pixels.
        uv = np.stack(((x + .5 - cols / 2) / focal, -(y + .5 - rows / 2) / focal), axis=1)
        support = uv @ _DIRECTIONS.T
        bound = np.quantile(support, .995, axis=0)
        edge = (x < 2) | (y < 2) | (x >= cols - 2) | (y >= rows - 2)
        clipped = np.any((support >= bound[None] - 1.0/focal) & edge[:, None], axis=0)
        floor_points = np.empty((0, 2))
        if is_goal:
            subset = uv[::max(1, len(uv)//128)]
            rays = np.column_stack((subset, -np.ones(len(subset)))) @ rotation.T
            rays = rays[rays[:, 2] < -.035]
            distance = (.001 - eye[2]) / rays[:, 2]
            valid = (distance > .005) & (distance < 1.2)
            floor_points = eye[:2] + rays[valid, :2] * distance[valid, None]
        result.append((eye.copy(), rotation.copy(), bound, clipped, len(x), focal, np.median(uv, axis=0), floor_points))
    return result


def locate(vertices, views, previous, *, is_goal, iterations):
    """Return (center x/y, yaw, bottom height) using visible RGB support alone.

    Clipping censors overshooting image boundaries. Hidden boundaries cost less
    than silhouettes that exclude observed pixels. World poses are never inputs.
    """
    views = [view for view in views if view is not None]
    if not views:
        return None
    if previous is not None:
        pose = np.asarray(previous, float).copy()
    else:
        centers = []
        for eye, rotation, _, _, _, _, uv, _ in views:
            ray = rotation @ np.r_[uv, -1.]
            if abs(ray[2]) > 1e-5:
                distance = (vertices[:, 2].max() - eye[2]) / ray[2]
                if .005 < distance < 1.8:
                    centers.append(eye[:2] + ray[:2] * distance)
        if not centers:
            return None
        guess = np.mean(centers, axis=0) - np.mean(vertices[:, :2], axis=0)
        pose = np.r_[guess, 0., 0.]
    # Work in metres/radians. Square-pixel calibration is supplied by the sensor.
    nvar = 3 if is_goal else 4
    delta = np.array([.0005, .0005, .003, .0005])
    measured = np.concatenate([view[2] * view[5] for view in views])
    censored = np.concatenate([view[3] for view in views])

    def residual(poses):
        angle = poses[:, 2:3]
        x, y, z = vertices.T
        world = np.stack((poses[:, 0:1] + np.cos(angle)*x - np.sin(angle)*y,
                          poses[:, 1:2] + np.sin(angle)*x + np.cos(angle)*y,
                          poses[:, 3:4] + z), axis=-1)
        guesses = []
        for eye, rotation, _, _, _, focal, _, _ in views:
            local = (world - eye) @ rotation
            depth = np.maximum(-local[:, :, 2:3], .01)
            guess = (local[:, :, :2] / depth) @ _DIRECTIONS.T
            guesses.append(focal * guess.max(axis=1))
        return np.concatenate(guesses, axis=1) - measured

    perturbations = np.zeros((nvar + 1, 4))
    perturbations[1:, :nvar] = np.diag(delta[:nvar])
    for _ in range(iterations):
        residuals = residual(pose[None, :] + perturbations)
        err = residuals[0]
        jac = ((residuals[1:] - err) / delta[:nvar, None]).T
        weight = np.minimum(1., 2.5 / np.maximum(np.abs(err), 1e-6))
        weight[(err > 0) & censored] = 0.
        weight[(err > 0) & ~censored] *= .45
        # A small angular regularizer avoids displacing an ill-conditioned
        # near-edge-on estimate purely to explain one antialiased fragment.
        system = jac.T @ (weight[:, None] * jac) + np.diag([5., 5., 1.2, 5.][:nvar])
        step = np.linalg.solve(system, -jac.T @ (weight * err))
        limits = np.array([.035, .035, .12, .014])[:nvar]
        pose[:nvar] += .80 * np.clip(step, -limits, limits)
        pose[2] = np.clip(pose[2], -.85, .85)
        pose[3] = 0. if is_goal else np.clip(pose[3], -.006, .18)
        if np.max(np.abs(step[:2])) < .0002 and abs(step[2]) < .004:
            break
    err = residual(pose[None])[0]
    if not np.isfinite(pose).all():
        return None
    return pose, float(np.median(np.abs(err[~(censored & (err > 0))]))) if np.any(~censored | (err <= 0)) else 100.


class GroundMark:
    """Register measured floor pixels to physical segments even when one edge is hidden."""

    def __init__(self, task):
        self.boxes = [(0., .060, .110)] if task == "carry" else [(0., .028, .130), (.106, .078, .033)]
        self.cells = {}
        self.fit_count = 0
        self.pose = None
        self.fit_error_m = None

    @staticmethod
    def _shifts(coords, edges):
        # Every observed edge may correspond to either side of a rectangle.
        # Vote for translations, retaining multiple distinct hypotheses.
        shifts = np.rint((coords[:, None] - edges).ravel()/.004).astype(int)
        low = shifts.min()
        vote = np.bincount(shifts - low)
        smooth = np.convolve(vote, [1, 2, 1], mode="full")[1:-1]
        best = []
        for index in np.argsort(-smooth, kind="stable"):
            if smooth[index] == 0 or len(best) >= 6:
                break
            if all(abs(index - chosen) > 2 for chosen in best):
                best.append(index)
        return (np.asarray(best) + low)*.004

    def observe(self, views, starting_center):
        for view in views:
            if view is None:
                continue
            for point in view[-1]:
                if np.linalg.norm(point-starting_center) < .46:
                    self.cells[tuple(np.rint(point/.0035).astype(int))] = point
        if len(self.cells) < 4:
            return self.pose
        if self.pose is not None and len(self.cells) < self.fit_count * 1.09 + 4:
            return self.pose
        self.fit_count = len(self.cells)
        cloud = np.array(list(self.cells.values()))
        cloud = cloud[np.linspace(0, len(cloud)-1, min(144, len(cloud)), dtype=int)]
        xedges = np.unique([c + s*w for c, w, _ in self.boxes for s in (-1, 1)])
        yedges = np.unique([s*h for _, _, h in self.boxes for s in (-1, 1)])
        angles = np.linspace(-np.pi/6, np.pi/6, 17)
        if self.pose is not None:
            angles = np.unique(np.r_[angles, self.pose[2] + np.array([-.025, 0., .025])])
        best = np.inf
        for yaw in angles:
            c, s = np.cos(yaw), np.sin(yaw)
            rotation = np.array([[c, -s], [s, c]])
            local = cloud @ rotation
            xs, ys = self._shifts(local[:, 0], xedges), self._shifts(local[:, 1], yedges)
            centers = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)
            x = local[None, :, 0] - centers[:, None, 0]
            y = local[None, :, 1] - centers[:, None, 1]
            dis = np.full(x.shape, np.inf)
            for cx, half_x, half_y in self.boxes:
                dx = np.abs(x - cx) - half_x
                dy = np.abs(y) - half_y
                # Euclidean distance to nearest edge, including its endpoints.
                box_dis = np.hypot(np.maximum(dx, 0.), np.maximum(dy, 0.)) + np.minimum(np.maximum(dx, dy), 0.)
                dis = np.minimum(dis, np.abs(box_dis))
            discrepancy = np.minimum(dis, .018).mean(axis=1)
            world = centers @ rotation.T
            # Weak workcell prior only breaks ambiguities between different edges:
            # the mark is ahead in the same operating lane. Neither its center
            # nor its randomized yaw is passed from the task state.
            local_center = world - starting_center
            penalty = .008*np.abs(local_center[:, 1]) + .09*np.maximum(-local_center[:, 0], 0.)
            if self.pose is not None:
                penalty += .001*np.linalg.norm(world - self.pose[:2], axis=1)
            scores = discrepancy + penalty
            index = np.argmin(scores)
            if scores[index] < best:
                best = scores[index]
                answer = np.r_[world[index], yaw, 0.]
                error = discrepancy[index]
        self.pose, self.fit_error_m = answer, float(error)
        return self.pose
