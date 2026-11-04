"""Full-coordinate spatial dynamics and equality multipliers independent of either importer."""

import numpy as np

H = 1 / 2048
Q_ELBOW = np.array([0.2])
V_ELBOW = np.array([0.35])
Q_JAWS = np.array([0.02, 0.008])
V_JAWS = np.array([0.05, 0.03])


def kkt(mass, bias, external, motor, row):
    C = np.array(row, dtype=float)[None]
    system = np.block([[mass, -C.T], [C, np.zeros((1, 1))]])
    x = np.linalg.solve(system, np.r_[external + motor - bias, 0.0])
    return x[:-1], C[0] * x[-1]


def physical(case):
    if case == "jaws":
        q = np.r_[Q_JAWS, -Q_JAWS[1] + 0.004]
        v = np.r_[V_JAWS, -V_JAWS[1]]
        center = np.array([0.25 + 0.031 + q[2], -0.125, 0.25 + q[0]])
        point = center + [-0.003, 0.006, -0.004]
        com = center + [0, -0.003, 0.004]
        force = np.array([2.5, 0.6, -1.75])
        motor = np.array([2.0, -0.5, 0.0])
        mass = np.diag([1.5, 0.2, 0.3])
        bias = np.zeros(3)
        J = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        C = np.array([0.0, 1.0, 1.0])
        T = np.array([[1.0, 0], [0, 1], [0, -1.0]])
    else:
        q = np.array([Q_ELBOW[0], 1.5 * Q_ELBOW[0] - 0.125])
        v = np.array([V_ELBOW[0], 1.5 * V_ELBOW[0]])
        c, s = np.cos(q[0]), np.sin(q[0])
        A = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
        c, s = np.cos(q[1]), np.sin(q[1])
        B = A @ np.array([[c, 0, s], [0, 1.0, 0], [-s, 0, c]])
        p0 = np.array([0.0625, -0.125, 0.25])
        p1 = p0 + A @ [0.32, 0.04, 0.02]
        z = np.array([0.0, 0.0, 1.0])
        y = A[:, 1]
        w0 = z * v[0]
        w1 = w0 + y * v[1]

        def jac(p, aff):
            return np.column_stack(
                (np.cross(z, p - p0), np.cross(y, p - p1) if aff else np.zeros(3))
            )

        ac0 = p0 + A @ [0.16, 0.03, 0.02]
        com = p1 + B @ [0.09, -0.04, 0.06]
        point = p1 + B @ [0.22, 0.06, -0.03]
        J = jac(point, True)
        alpha = np.cross(w0, y * v[1])
        acc_joint = np.cross(w0, np.cross(w0, p1 - p0))
        mass = np.zeros((2, 2))
        bias = np.zeros(2)
        for m, p, r, spin, spin_bias, acc, inertia, aff in (
            (
                2.0,
                ac0,
                A,
                w0,
                np.zeros(3),
                np.cross(w0, np.cross(w0, ac0 - p0)),
                [0.035, 0.05, 0.06],
                False,
            ),
            (
                1.2,
                com,
                B,
                w1,
                alpha,
                acc_joint + np.cross(alpha, com - p1) + np.cross(w1, np.cross(w1, com - p1)),
                [0.012, 0.018, 0.021],
                True,
            ),
        ):
            V = jac(p, aff)
            W = np.column_stack((z, y if aff else np.zeros(3)))
            tensor = r @ np.diag(inertia) @ r.T
            mass += m * V.T @ V + W.T @ tensor @ W
            bias += V.T @ (m * acc) + W.T @ (tensor @ spin_bias + np.cross(spin, tensor @ spin))
        C = np.array([-1.5, 1.0])
        T = np.array([[1.0], [1.5]])
        force = np.array([0.75, -2.25, 1.125])
        motor = np.array([0.375, 0.0])
    external = J.T @ force
    accel, reaction = kkt(mass, bias, external, motor, C)
    return dict(
        q=q,
        v=v,
        mass=mass,
        bias=bias,
        point=point,
        com=com,
        force=force,
        torque=np.cross(point - com, force),
        J=J,
        T=T,
        external=external,
        motor=motor,
        C=C,
        accel=accel,
        reaction=reaction,
    )


def grip_command(t):
    """Smooth external targets; no object pose is used as a command."""
    knots = (
        (0.0, 0.0, 0.0),
        (0.10, 0.0, 0.0),
        (0.34, 0.0, 0.018),
        (0.43, 0.0, 0.018),
        (0.77, 0.05, 0.018),
        (0.95, 0.05, 0.018),
        (1.29, 0.0, 0.018),
        (1.36, 0.0, 0.018),
        (1.62, 0.0, 0.0),
        (1.72, 0.0, 0.0),
    )
    i = next((i for i in range(len(knots) - 1) if t < knots[i + 1][0]), len(knots) - 2)
    a, b = knots[i], knots[i + 1]
    h = b[0] - a[0]
    u = min(max((t - a[0]) / h, 0.0), 1.0)
    s = u**3 * (10 + u * (-15 + 6 * u))
    ds = 30 * u * u * (1 - u) ** 2 / h
    delta = np.array(b[1:]) - a[1:]
    return np.asarray(a[1:]) + s * delta, ds * delta
