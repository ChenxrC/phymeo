"""Independent rigid-link kinematics and momentum from the dimensions of fixed_tool_probe.urdf."""

import numpy as np


def rot(axis, theta):
    c, s = np.cos(theta), np.sin(theta)
    if axis == 0:
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == 1:
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


PIVOT = np.array([0.25, 0.0, 0.25])
Z = np.array([0.0, 0.0, 1.0])
FORCE = np.array([0.5, -2.0, 1.25])
MOTOR = np.array([0.125, -0.25])
H = 1 / 2048
CASES = ((0.37, 0.025, 0.55, -0.08), (-0.61, -0.035, -0.4, 0.12))


def frames(q):
    A = rot(2, np.pi / 2 + q[0])
    E = A @ rot(2, np.pi / 2)
    pos_elbow = PIVOT + A @ [0.25, 0, 0]
    pos_slide = pos_elbow + E @ [0.125 + q[1], 0, 0]
    pos_collet = pos_slide + E @ [0.125, -0.0625, 0]
    C = E @ rot(2, np.pi / 2)
    return dict(
        upper=(PIVOT, A),
        elbow=(pos_elbow, E),
        slide=(pos_slide, E),
        collet=(pos_collet, C),
        probe=(pos_collet + C @ [0.0625, 0, 0.03125], C @ rot(1, np.pi / 2)),
    )


def jacobian(point, axis, moving_slide):
    w = np.column_stack((Z, np.zeros(3)))
    J = np.column_stack((np.cross(Z, point - PIVOT), axis if moving_slide else np.zeros(3)))
    return J, w


def reference(q, v):
    v = np.asarray(v)
    f = frames(q)
    members = (
        (
            "upper",
            0,
            2.0,
            [0.125, 0, 0],
            rot(0, np.pi / 2),
            [[0.0625, 0.015625, 0], [0.015625, 0.078125, 0], [0, 0, 0.09375]],
        ),
        ("elbow", 0, 1.0, [0.0625, 0.03125, 0], rot(2, np.pi / 4), np.diag([0.020, 0.025, 0.030])),
        ("slide", 1, 0.5, [0.03125, 0, 0], np.eye(3), np.diag([0.004, 0.005, 0.006])),
        ("collet", 1, 0.25, [0.03125, 0, 0], np.eye(3), np.diag([0.001, 0.00125, 0.0015])),
        (
            "probe",
            1,
            0.25,
            [0.015625, 0.0078125, 0.015625],
            rot(2, 0.38) @ rot(1, -0.25) @ rot(0, 0.45),
            [[0.0015, 0.0002, 0], [0.0002, 0.0018, 0.0001], [0, 0.0001, 0.002]],
        ),
    )
    axis = f["slide"][1][:, 0]
    omega = v[0] * Z
    momentum = np.zeros(6)
    M = np.zeros((2, 2))
    bias = np.zeros(2)
    components = [[], []]
    for name, group, m, offset, Q, tensor in members:
        origin, R = f[name]
        c = origin + R @ offset
        Iw = (R @ Q) @ tensor @ (R @ Q).T
        J, W = jacobian(c, axis, group)
        speed = J @ v
        linear = m * speed
        momentum += np.r_[linear, np.cross(c, linear) + Iw @ omega]
        M += m * J.T @ J + W.T @ Iw @ W
        # d(omega x (p-pivot) + axis*qdot_slide)/dt at fixed generalized velocity.
        ac = np.cross(omega, np.cross(omega, c - PIVOT)) + 2 * group * np.cross(omega, axis * v[1])
        bias += J.T @ (m * ac) + W.T @ np.cross(omega, Iw @ omega)
        components[group].append((m, c, Iw))
    tip = f["probe"][0] + f["probe"][1] @ [0.03125, 0, 0]
    J, W = jacobian(tip, axis, True)
    acceleration = np.linalg.solve(M, MOTOR + J.T @ FORCE - bias)
    combined = []
    for group in components:
        mass = sum(x[0] for x in group)
        com = sum(m * c for m, c, _ in group) / mass
        inertia = sum(
            tensor + m * (((p - com) @ (p - com)) * np.eye(3) - np.outer(p - com, p - com))
            for m, p, tensor in group
        )
        combined.append((mass, com, inertia))
    return dict(
        frames=f,
        combined=combined,
        tip=tip,
        tip_velocity=J @ v,
        omega=W @ v,
        jacobian=J,
        M=M,
        bias=bias,
        momentum=momentum,
        energy=0.5 * v @ M @ v,
        acceleration=acceleration,
    )
