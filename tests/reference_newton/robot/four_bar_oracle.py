"""Independent one-coordinate mechanics of a 1 m by 1/2 m parallelogram."""

import numpy as np


def case(tilted=False):
    # Link order is left side, horizontal coupler, right side. Vector loads
    # in this fixture are expressed in the planar basis and rotate with it.
    return dict(
        frame=np.array([[0.6, 0, 0.8], [0, 1, 0], [-0.8, 0, 0.6]]) if tilted else np.eye(3),
        origin=np.array([0.25, -0.375, 0.5]) if tilted else np.zeros(3),
        theta=np.arctan2(0.6, 0.8) if tilted else np.arctan2(0.8, 0.6),
        speed=-0.375 if tilted else 0.25,
        drive=-0.25 if tilted else 0.375,
        gravity=np.array([0.25, -0.75, 0.0]) if tilted else np.zeros(3),
        forces=np.array([[0.0, 0, 0], [-0.5, 0.25, 0], [0, 0, 0]]) if tilted else np.zeros((3, 3)),
        moments=np.array([0.0, 0.125, 0.0625]) if tilted else np.zeros(3),
        h=1 / 256,
        steps=64,
    )


def mechanics(row, theta, speed):
    c, s = np.cos(theta), np.sin(theta)
    n, t = np.array([c, s, 0]), np.array([-s, c, 0])
    f = row["forces"] + np.array([1, 2, 1])[:, None] * row["gravity"]
    applied = (
        row["drive"]
        + row["moments"][0]
        + row["moments"][2]
        + (0.25 * (f[0] + f[2]) + 0.5 * f[1]) @ t
    )
    accel = applied / 0.75
    coupler_acc = 0.5 * (accel * t - speed**2 * n)
    # Linear and angular balance on the coupler, then angular balance of the right side about its fixed pin.
    fy = (2 * coupler_acc[1] - f[1, 1]) / 2 - row["moments"][1]
    ft = 0.5 * t @ f[2] + 2 * (row["moments"][2] - 0.125 * accel)
    fx = (c * fy - ft) / s
    first_force = row["frame"] @ np.array([fx, fy, 0])
    end_velocity = row["frame"] @ (0.5 * speed * t)
    return accel, first_force, float(first_force @ end_velocity)


def points(row, theta):
    end = 0.5 * np.array([np.cos(theta), np.sin(theta), 0])
    local = np.stack([end, end + [1.0, 0, 0]])
    return row["origin"] + local @ row["frame"].T


def centers(row, theta):
    end = 0.5 * np.array([np.cos(theta), np.sin(theta), 0])
    return np.stack([end / 2, end + [0.5, 0, 0], end / 2 + [1.0, 0, 0]])


def physical_work(row, theta):
    forces = row["forces"] + np.array([1, 2, 1])[:, None] * row["gravity"]
    rotation_work = (row["drive"] + row["moments"][0] + row["moments"][2]) * (theta - row["theta"])
    return rotation_work + float(
        np.sum(forces * (centers(row, theta) - centers(row, row["theta"])))
    )
