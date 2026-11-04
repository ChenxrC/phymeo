"""Independent full-body kinematics and full-coordinate KKT for a fan and a serial transmission."""

import numpy as np

H = 1 / 2048


def kkt(mass, bias, external, motor, C):
    system = np.block([[mass, -C.T], [C, np.zeros((len(C), len(C)))]])
    answer = np.linalg.solve(system, np.r_[external + motor - bias, np.zeros(len(C))])
    return answer[: len(mass)], answer[len(mass) :]


def fan():
    names = ("lift", "screw", "idler", "meter")
    q = np.array([0.02, 0.005, -0.001, 0.0045])
    v = np.array([0.04, 0.025, -0.025, 0.0125])
    P = np.array([[1, 0], [0, 1], [0, -1], [0, 0.5]])
    C = np.array([[0, 1, 1, 0], [0, -0.5, 0, 1.0]])
    mass = np.diag([1.9, 0.2, 0.3, 0.4])
    bias = np.zeros(4)
    F = np.array([[0.0, 0.0, 0.0], [-0.7, 0, 0.0], [1.1, 0, 0.5], [-1.6, 0, -0.1]])
    W = np.zeros_like(F)
    motor = np.array([0.25, 0.4, 0.0, 0.0])
    ext = np.r_[F[:, 2].sum(), F[1:, 0]]
    point = np.array([0.25 + 0.1 + q[-1] + 0.03125, -0.125 + 0.04, 0.25 + q[0] + 0.01])
    J = np.array([[0, 0, 0, 1.0], [0, 0, 0, 0], [1.0, 0, 0, 0]])
    physical_energies = 0.5 * np.array([1.0, 0.2, 0.3, 0.4]) * (v[0] ** 2 + np.r_[0.0, v[1:] ** 2])
    return dict(
        names=names,
        P=P,
        C=C,
        q=q,
        v=v,
        mass=mass,
        bias=bias,
        forces=F,
        torques=W,
        motor=motor,
        external=ext,
        point=point,
        J=J,
        point_link="gauge",
        point_offset=[0.03125, 0, 0],
        energy=0.5 * v @ mass @ v,
        body_energy=physical_energies,
        row_names=(("screw", "idler"), ("screw", "meter")),
    )


def rot(axis, angle):
    c, s = np.cos(angle), np.sin(angle)
    return (
        np.array([[1, 0, 0], [0, c, -s], [0, s, c]]),
        np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
        np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]),
    )[axis]


def serial(u=0.18, rate=0.36):
    names = ("shaft", "gear", "nozzle_drive")
    q = np.array([u, 1.5 * u - 0.125, -0.75 * u + 0.2625])
    v = np.array([rate, 1.5 * rate, -0.75 * rate])
    P = np.array([[1.0], [1.5], [-0.75]])
    C = np.array([[-1.5, 1.0, 0], [0, 0.5, 1.0]])
    offsets = ([0.0625, -0.125, 0.25], [0.32, 0.04, 0.02], [0.22, 0.04, -0.03])
    com = ([0.16, 0.03, 0.02], [0.09, -0.04, 0.06], [0.065, 0.035, -0.025])
    tensors = ([0.035, 0.05, 0.06], [0.012, 0.018, 0.021], [0.003, 0.004, 0.005])
    mass = np.zeros((3, 3))
    bias = np.zeros(3)
    body_energy = []
    pos = np.zeros(3)
    basis = np.eye(3)
    spin = np.zeros(3)
    angular_bias = np.zeros(3)
    origin_bias = np.zeros(3)
    origins = []
    axes = []
    center = None

    def jacobian(point):
        J = np.zeros((3, 3))
        W = np.zeros_like(J)
        for j, (p, ax) in enumerate(zip(origins, axes, strict=True)):
            J[:, j] = np.cross(ax, point - p)
            W[:, j] = ax
        return J, W

    for i, (axis, m) in enumerate(zip((2, 1, 0), (2.0, 1.2, 0.6), strict=True)):
        arm = basis @ offsets[i]
        origin_bias = (
            origin_bias + np.cross(angular_bias, arm) + np.cross(spin, np.cross(spin, arm))
        )
        pos = pos + arm
        origins.append(pos.copy())
        ax = basis[:, axis]
        axes.append(ax)
        angular_bias = angular_bias + np.cross(spin, ax * v[i])
        spin = spin + ax * v[i]
        basis = basis @ rot(axis, q[i])
        lever = basis @ com[i]
        center = pos + lever
        J, W = jacobian(center)
        tensor = basis @ np.diag(tensors[i]) @ basis.T
        acceleration = (
            origin_bias + np.cross(angular_bias, lever) + np.cross(spin, np.cross(spin, lever))
        )
        mass += m * J.T @ J + W.T @ tensor @ W
        bias += J.T @ (m * acceleration) + W.T @ (
            tensor @ angular_bias + np.cross(spin, tensor @ spin)
        )
        linear = J @ v
        body_energy.append((m * linear @ linear + spin @ tensor @ spin) / 2)
    point = pos + basis @ [0.10, 0.06, -0.04]
    J, _ = jacobian(point)
    F = np.zeros((3, 3))
    W = F.copy()
    F[-1] = [1.3, -1.7, 0.8]
    W[-1] = np.cross(point - center, F[-1])
    return dict(
        names=names,
        P=P,
        C=C,
        q=q,
        v=v,
        mass=mass,
        bias=bias,
        forces=F,
        torques=W,
        motor=np.array([0.33, 0, 0]),
        external=J.T @ F[-1],
        point=point,
        J=J,
        point_link="nozzle",
        point_offset=[0.10, 0.06, -0.04],
        energy=sum(body_energy),
        body_energy=body_energy,
        row_names=(("shaft", "gear"), ("gear", "nozzle_drive")),
    )
