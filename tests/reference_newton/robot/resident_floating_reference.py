"""Independent C9 full floating reference, AUTHORING_UNEXECUTED.

Only NumPy and stdlib. No Phymeo/TYX/Newton imports, finite differences, product
FK, product M/J, or recursive product bias. Homogeneous transforms carry exact
first and second time derivatives; velocity-basis jets construct every J column.
The subsequent native Newton consumer is a separate, later-authorized driver.
"""

import copy
from itertools import product

import numpy as np


def skew(x):
    a, b, c = np.asarray(x, float)
    return np.array([[0., -c, b], [c, 0., -a], [-b, a, 0.]])


def vee(x):
    return np.array([x[2, 1]-x[1, 2], x[0, 2]-x[2, 0], x[1, 0]-x[0, 1]]) / 2


def quat_matrix(q):
    x, y, z, w = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])


def quat_product(a, b):
    av, bv = np.asarray(a[:3]), np.asarray(b[:3])
    return np.r_[a[3]*bv+b[3]*av+np.cross(av, bv), a[3]*b[3]-av@bv]


def axis_rotation(axis, angle):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    K = skew(axis)
    return np.eye(3) + np.sin(angle)*K + (1-np.cos(angle))*(K@K)


def transform(rotation, position):
    result = np.eye(4)
    result[:3, :3], result[:3, 3] = rotation, position
    return result


def multiply(a, b):
    # These are derivatives, not Taylor coefficients.
    return a[0]@b[0], a[1]@b[0]+a[0]@b[1], a[2]@b[0]+2*a[1]@b[1]+a[0]@b[2]


def jets(data, q, v):
    q, v = np.asarray(q, float), np.asarray(v, float)
    root = quat_matrix(q[3:7])
    spin = skew(v[3:6])
    first, second = spin@root, spin@spin@root
    T = transform(root, q[:3])
    D, DD = np.zeros((4, 4)), np.zeros((4, 4))
    D[:3, :3], DD[:3, :3] = first, second
    # The public root velocity is COM/world, not authored-origin velocity.
    D[:3, 3] = v[:3] - first@data["root_com"]
    DD[:3, 3] = -second@data["root_com"]
    result = [(T, D, DD)]
    children = data["children"]
    for i, parent in enumerate(children["parents"]):
        fixed = (transform(quat_matrix(children["orientations"][i]), children["offsets"][i]), np.zeros((4, 4)), np.zeros((4, 4)))
        axis = children["axes"][i] / np.linalg.norm(children["axes"][i])
        rate = v[6+i]
        local, local_d, local_dd = np.eye(4), np.zeros((4, 4)), np.zeros((4, 4))
        if children["kinds"][i] == "prismatic":
            local[:3, 3], local_d[:3, 3] = q[7+i]*axis, rate*axis
        else:
            R = axis_rotation(axis, q[7+i])
            K = skew(axis)
            local[:3, :3], local_d[:3, :3], local_dd[:3, :3] = R, rate*K@R, rate*rate*K@K@R
        result.append(multiply(multiply(result[parent+1], fixed), (local, local_d, local_dd)))
    com_local = np.vstack((data["root_com"], children["com"]))
    positions, rotations, centers, velocity, acceleration, omega, alpha = [], [], [], [], [], [], []
    for (T, D, DD), c in zip(result, com_local, strict=True):
        point = np.r_[c, 1.]
        positions.append(T[:3, 3]); rotations.append(T[:3, :3])
        centers.append((T@point)[:3]); velocity.append((D@point)[:3]); acceleration.append((DD@point)[:3])
        omega.append(vee(D[:3, :3]@T[:3, :3].T))
        alpha.append(vee(DD[:3, :3]@T[:3, :3].T + D[:3, :3]@D[:3, :3].T))
    return dict(position=np.array(positions), rotation=np.array(rotations), com_position=np.array(centers), com_velocity=np.array(velocity), com_acceleration=np.array(acceleration), angular_velocity=np.array(omega), angular_acceleration=np.array(alpha))


def mechanics(data, q, v):
    d = len(v)
    observed = jets(data, q, v)
    # Every column is an exact unit-velocity transform derivative, never FD.
    basis = [jets(data, q, column) for column in np.eye(d)]
    Jv = np.stack([item["com_velocity"] for item in basis], axis=2)
    Jw = np.stack([item["angular_velocity"] for item in basis], axis=2)
    masses = np.r_[data["root_mass"], data["children"]["mass"]]
    principal = np.vstack((data["root_inertia"], data["children"]["inertia"]))
    frames = observed["rotation"].copy()
    frames[0] = frames[0]@quat_matrix(data["root_inertia_frame"])
    inertia = np.array([R@np.diag(I)@R.T for R, I in zip(frames, principal, strict=True)])
    M, bias = np.zeros((d, d)), np.zeros(d)
    energy, momentum, angular = 0., np.zeros(3), np.zeros(3)
    for i, mass in enumerate(masses):
        spin = observed["angular_velocity"][i]
        F = mass*(observed["com_acceleration"][i]-data["gravity"])
        torque = inertia[i]@observed["angular_acceleration"][i]+np.cross(spin, inertia[i]@spin)
        M += mass*(Jv[i].T@Jv[i])+Jw[i].T@inertia[i]@Jw[i]
        bias += Jv[i].T@F+Jw[i].T@torque
        linear_momentum = mass*observed["com_velocity"][i]
        momentum += linear_momentum
        angular += np.cross(observed["com_position"][i], linear_momentum)+inertia[i]@spin
        energy += (observed["com_velocity"][i]@linear_momentum+spin@inertia[i]@spin)/2
    return dict(**observed, mass=M, bias=bias, linear_jacobian=Jv, angular_jacobian=Jw, inertia_world=inertia, energy=energy, momentum=momentum, angular_momentum=angular)


def external(data, q):
    effort, force, torque = (data[key].copy() for key in ("effort", "force", "torque"))
    if data.get("force_point") is not None:
        # A declared local application point and held world force; no product FK.
        arm = quat_matrix(q[3:7])@(data["force_point"]-data["root_com"])
        torque[0] = np.cross(arm, force[0])
    return effort, force, torque


def applied(terms, effort, force, torque):
    result = np.einsum("bci,bc->i", terms["linear_jacobian"], force)
    result += np.einsum("bci,bc->i", terms["angular_jacobian"], torque)
    result[6:] += effort
    return result


def bounded_drive(M, bias, load, q, v, drive, kp, kd, caps, h):
    n = len(kp)
    if not n:
        return np.zeros(0)
    free = np.linalg.solve(M, load-bias)[6:]
    mobility = np.linalg.inv(M)[6:, 6:]
    beta = h*h*kp+h*kd
    rhs = drive[:, 2]+kp*(drive[:, 0]-q[7:]-h*v[6:])-kd*v[6:]-beta*free
    fixed, moving = np.flatnonzero(beta == 0), np.flatnonzero(beta > 0)
    tau = np.clip(rhs, -caps, caps)
    if not len(moving):
        return tau
    if len(moving) > 4:
        raise ValueError("independent exhaustive PD reference limited to four nonzero-gain axes")
    A = mobility[np.ix_(moving, moving)]+np.diag(1/beta[moving])
    b = rhs[moving]/beta[moving]-mobility[np.ix_(moving, fixed)]@tau[fixed]
    # Exhaust every box face, solve its free block, check the full global KKT.
    for signs in product((-1, 0, 1), repeat=len(moving)):
        x = np.array(signs, float)*caps[moving]
        free_ids = np.flatnonzero(np.array(signs) == 0)
        pinned = np.flatnonzero(np.array(signs) != 0)
        if len(free_ids):
            x[free_ids] = np.linalg.solve(A[np.ix_(free_ids, free_ids)], b[free_ids]-A[np.ix_(free_ids, pinned)]@x[pinned])
        if np.any(np.abs(x) > caps[moving]+1e-11):
            continue
        gradient = A@x-b
        if any((s == 0 and abs(g) > 1e-10) or (s < 0 and g < -1e-10) or (s > 0 and g > 1e-10) for s, g in zip(signs, gradient, strict=True)):
            continue
        tau[moving] = x
        return tau
    raise AssertionError("no independently feasible PD active face")


def drift(data, q, velocity, h):
    q, velocity = np.asarray(q, float), np.asarray(velocity, float)
    rotation_vector = h*velocity[3:6]
    theta = np.linalg.norm(rotation_vector)
    inc = np.r_[np.zeros(3), 1.] if theta == 0 else np.r_[rotation_vector*np.sin(theta/2)/theta, np.cos(theta/2)]
    orientation = quat_product(inc, q[3:7]); orientation /= np.linalg.norm(orientation)
    if orientation@q[3:7] < 0:
        orientation *= -1
    origin = q[:3]+h*velocity[:3]-(quat_matrix(orientation)-quat_matrix(q[3:7]))@data["root_com"]
    return np.r_[origin, orientation, q[7:]+h*velocity[6:]]


def advance(data, q, v, drive, external_packet, h, *, kp=None, kd=None, caps=None):
    n = len(q)-7
    kp = np.zeros(n) if kp is None else np.asarray(kp)
    kd = np.zeros(n) if kd is None else np.asarray(kd)
    caps = np.full(n, 64.) if caps is None else np.asarray(caps)
    t = mechanics(data, q, v)
    load = applied(t, *external_packet)
    motor = bounded_drive(t["mass"], t["bias"], load, q, v, drive, kp, kd, caps, h)
    total = load.copy(); total[6:] += motor
    a = np.linalg.solve(t["mass"], total-t["bias"])
    next_v = np.asarray(v)+h*a
    next_q = drift(data, q, next_v, h)
    return next_q, next_v, dict(**t, external=load, drive=motor, acceleration=a)


def _children(n):
    return dict(parents=np.array([-1 if i < 2 else (i-2)//2 for i in range(n)], dtype=int), kinds=tuple("revolute" if i % 2 == 0 else "prismatic" for i in range(n)),
                offsets=np.array([[1/8, (i%3-1)/16, (i%2)/32] for i in range(n)], float).reshape(n, 3),
                orientations=np.tile([0., 0., 0., 1.], (n, 1)), axes=np.array([[0., 3/5, 4/5] if i%2 == 0 else [1/3, 2/3, 2/3] for i in range(n)]).reshape(n, 3),
                mass=np.array([1+i/8 for i in range(n)], float), com=np.tile([1/32, -1/64, 1/128], (n, 1)), inertia=np.array([[1/8+i/128, 3/16+i/128, 1/4+i/128] for i in range(n)], float).reshape(n, 3))


def case(name, *, n=None, placement="base", boost=False):
    n = (0 if name.startswith("P02") else 1) if n is None else n
    children = _children(n)
    data = dict(name=name, root_mass=2., root_com=np.zeros(3), root_inertia=np.full(3, .5), root_inertia_frame=np.array([0., 0., 0., 1.]),
                gravity=np.zeros(3), children=children, q=np.r_[np.zeros(3), [0., 0., 0., 1.], np.zeros(n)], v=np.zeros(6+n), drive=np.zeros((n, 3)), effort=np.zeros(n), force=np.zeros((n+1, 3)), torque=np.zeros((n+1, 3)), placement=placement, force_point=None)
    if name in ("P01", "P03", "P03_external", "P04"):
        children.update(parents=np.array([-1]), kinds=("revolute" if name == "P04" else "prismatic",), offsets=np.zeros((1, 3)), orientations=np.array([[0., 0., 0., 1.]]), axes=np.array([[0., 0., 1.]] if name == "P04" else [[1., 0., 0.]]), mass=np.ones(1), com=np.zeros((1, 3)), inertia=np.full((1, 3), .25))
        if name == "P01":
            data["gravity"] = np.array([0., 0., -2.]); data["q"][7] = 1/8
        elif name == "P03_external":
            data["force"][1, 0] = 3.
        else:
            data["drive"][0, 2] = 1. if name == "P04" else 3.
    elif name.startswith("P02"):
        data["root_com"] = np.array([1/8, 0., 0.]); data["root_inertia"] = np.array([1/4, 3/8, 1/2])
        if name == "P02_origin":
            data["force"][0, 1] = 2.; data["force_point"] = np.zeros(3)
        else:
            data["root_com"] *= 0; data["root_inertia"] = np.array([1/4, 1/2, 1/2])
            data["root_inertia_frame"] = np.array([0., 0., np.sin(np.pi/8), np.cos(np.pi/8)])
            data["torque"][0, 0] = 1.
    elif name in ("P04_eccentric", "P05"):
        data["root_com"] = np.array([1/16, -1/32, 1/64]); data["root_inertia"] = np.array([.5, .625, .75])
        data["root_inertia_frame"] = np.array([0., 0., .6, .8])
        data["q"][3:7] = [.5, .5, .5, .5]
        data["q"][7:] = np.array([(i%3-1)/16 for i in range(n)])
        data["v"][:6] = [1/16, -1/32, 1/64, 1/32, -1/64, 1/128]
        data["v"][6:] = np.array([(i%3-1)/64 for i in range(n)])
        data["drive"][:, 2] = np.array([(-1)**i/8 for i in range(n)])
        data["effort"][:] = np.array([(-1)**(i+1)/32 for i in range(n)])
        data["force"][-1] = [1/8, -1/16, 1/32]; data["torque"][0] = [-1/32, 1/64, 0.]
        data["gravity"] = np.array([0., 0., -1/4])
    else:
        raise ValueError(name)
    if boost:
        data["v"][:3] += [1/4, -1/8, 1/16]
    if placement != "base":
        quat = np.array([.5, .5, .5, .5]) if placement in ("cycle", "large") else np.array([0., 0., 1., 0.])
        Q = quat_matrix(quat)
        data["q"][:3] = Q@data["q"][:3]+([4096., -2048., 1024.] if placement == "large" else [1/2, -1/4, 3/4])
        data["q"][3:7] = quat_product(quat, data["q"][3:7])
        data["v"][:3], data["v"][3:6] = Q@data["v"][:3], Q@data["v"][3:6]
        data["gravity"] = Q@data["gravity"]
        data["force"], data["torque"] = data["force"]@Q.T, data["torque"]@Q.T
    return copy.deepcopy(data)