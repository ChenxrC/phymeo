"""C11 independent full-body time jets and unreduced original-row KKT.

NumPy/stdlib only. No Phymeo, TYX, Newton, finite differences or product M/J/P.
The C9 independent transform algebra is reused; A04 sums all four authored
inertial parts before forming each physical group's COM Jacobian.
"""

import copy
import importlib.util
from functools import lru_cache
from pathlib import Path

import numpy as np


@lru_cache(maxsize=1)
def base():
    path = Path(__file__).with_name("resident_floating_reference.py")
    spec = importlib.util.spec_from_file_location("c11_independent_time_jets", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _children(masses, offsets):
    n = len(masses)
    return dict(parents=np.full(n, -1, int), kinds=("prismatic",) * n,
                offsets=np.asarray(offsets, float).reshape(n, 3),
                orientations=np.tile([0., 0., 0., 1.], (n, 1)),
                axes=np.tile([1., 0., 0.], (n, 1)), mass=np.asarray(masses, float),
                com=np.zeros((n, 3)), inertia=np.tile([1/16, 5/64, 3/32], (n, 1)))


def _graph(data, p, c, rows, drivers):
    p, c, rows = np.asarray(p, float), np.asarray(c, float), np.asarray(rows, float)
    k, n = p.shape[1], p.shape[0]
    S = np.zeros((6+n, 6+k)); S[:6, :6] = np.eye(6); S[6:, 6:] = p
    C = np.zeros((len(rows), 6+n)); C[:, 6:] = rows
    E = np.zeros((6+n, k))
    for j, i in enumerate(drivers):
        E[6+i, j] = 1.
    data.update(P=p, offset=c, C=C, S=S, E=E, drivers=tuple(drivers),
                q=np.r_[np.zeros(3), [0., 0., 0., 1.], np.zeros(k)],
                v=np.zeros(6+k), drive=np.zeros((k, 3)), kp=np.zeros(k), kd=np.zeros(k),
                caps=np.full(k, 1.5), force=np.zeros((n+1, 3)), torque=np.zeros((n+1, 3)),
                effort=np.zeros(n), delay=0)


def case(name, *, variant=None, placement="base"):
    children = _children([1., 2.], [[-.25, 0, 0], [.25, 0, 0]])
    children["inertia"][1] = [3/32, 7/64, 1/8]
    data = dict(name=name, variant=variant, placement=placement,
                root_mass=2., root_com=np.zeros(3), root_inertia=np.array([.2, .3, .4]),
                root_inertia_frame=np.array([0., 0., 0., 1.]), gravity=np.zeros(3),
                children=children, parts=None, h=1/1024, fixture="jaws", contacts=())
    _graph(data, [[1.], [-1.]], [0., .004], [[1., 1.]], [0])
    if name == "A01":
        data["drive"][0, 2] = 1.
    elif name == "A02":
        data["contacts"] = (dict(kind="box_sphere", first_group=2, half=np.array([1/32, 1/16, 1/16]),
                                  sphere=np.array([1/4+.004+1/32+1/16-1/1024, 0., 0.]),
                                  radius=1/16, stiffness=2048.),)
    elif name == "A03":
        data["fixture"] = "graph"
        children = _children([1., 2., .5], [[-.25, 0, 0], [.25, 0, 0], [.5, 0, 0]])
        children["inertia"][1:] = [[3/32, 7/64, 1/8], [.02, .03, .04]]
        data["children"] = children
        chain = variant == "chain"
        _graph(data, [[1.], [-1.], [.5]], [0., .004, 0. if chain else .002],
               [[1., 1., 0.], [0., .5, 1.] if chain else [-.5, 0., 1.]], [0])
        data["v"][6] = .02
        data["force"][:, 0] = [1/8, 1/4, -3/8, 1/2]
        data["drive"][0, 2] = 3/8
    elif name == "A04":
        data.update(fixture="elbow", h=1/4096, root_com=np.array([1/32, -1/64, 0.]),
                    gravity=np.array([.5, -1., -2.]))
        children = _children([1., .75], [[1/4, 0, 1/16], [3/8, 0, 0]])
        children.update(parents=np.array([-1, 0]), kinds=("revolute", "revolute"),
                        axes=np.array([[0., 0., 1.], [0., 1., 0.]]),
                        com=np.array([[1/8, 1/32, 0.], [1/16, -1/32, 1/64]]),
                        inertia=np.array([[1/16, 5/64, 3/32], [1/32, 5/128, 3/64]]))
        data["children"] = children
        _graph(data, [[1.], [1.5]], [0., -1/8], [[-1.5, 1.]], [0])
        tool_rotation = base().axis_rotation([0, 0, 1], np.pi/2)
        data["parts"] = (
            dict(group=0, mass=2., center=data["root_com"].copy(), inertia=np.array([.2, .3, .4]), rotation=np.eye(3)),
            dict(group=1, mass=1., center=children["com"][0].copy(), inertia=children["inertia"][0].copy(), rotation=np.eye(3)),
            dict(group=2, mass=.75, center=children["com"][1].copy(), inertia=children["inertia"][1].copy(), rotation=np.eye(3)),
            dict(group=2, mass=.25, center=np.array([3/16, 1/16, -1/32])+tool_rotation@np.array([1/32, 0., 1/64]),
                 inertia=np.array([1/128, 5/512, 3/256]), rotation=tool_rotation),
        )
        data["q"][:3] = [1/8, -1/4, 3/8]
        data["v"][:6] = [1/8, -1/16, 1/32, .25, -.125, .1875]
        if variant == "p0":
            data["q"][7], data["v"][6] = -.25, -.375
        elif variant in ("p1", "p1-cycle"):
            data["q"][3:7] = [0., np.sin(.375/2), 0., np.cos(.375/2)]
            data["q"][7], data["v"][6] = .125, .25
        elif variant == "p2":
            axis = np.array([1., 2., -1.])/np.sqrt(6.)
            data["q"][3:7] = np.r_[axis*np.sin(-.5/2), np.cos(-.5/2)]
            data["q"][7], data["v"][6] = .375, .125
        else:
            raise ValueError(variant)
        data["force"][:] = [[.25, -.5, .75], [-.5, .375, -.25], [1.25, -.75, .5]]
        data["torque"][:] = [[1/32, -1/16, 3/64], [-1/32, 1/64, 1/16], [1/8, 1/16, -1/32]]
        data["drive"][0, 2], data["caps"][0] = 13/32, 4.
        if variant == "p1-cycle":
            placement = "cycle"
    elif name == "A05":
        data.update(fixture="bilateral", root_inertia=np.full(3, .5))
        children = _children([2., 1.], [[-7/64, 0, 0], [7/64, 0, 0]])
        children["inertia"][:] = .25
        data["children"] = children
        _graph(data, [[1.], [-1.]], [0., 0.], [[1., 1.]], [0])
        data["contacts"] = (dict(kind="sphere_pair", first_group=1, second_group=2,
                                  radius=1/8, stiffness=64.),)
    elif name == "lifecycle":
        data["q"][2], data["v"][2], data["gravity"][2] = 1/8-1/4096, 1/8, -8.
        data["contacts"] = (dict(kind="sphere_plane", first_group=0, radius=1/8,
                                  point=np.zeros(3), normal=np.array([0., 0., 1.]), stiffness=64.),)
    elif name == "counts":
        n, k = 16, (1 if variant == "n16-k1" else 15)
        data["fixture"] = "counts"
        children = _children(np.full(n, .5), [[(i+1)/32, 0., 0.] for i in range(n)])
        children["inertia"][:] = [1/64, 3/128, 1/32]
        data["children"] = children
        if k == 1:
            p = np.array([[(-1.)**i] for i in range(n)])
            rows = np.zeros((n-1, n))
            for i in range(1, n):
                rows[i-1, 0], rows[i-1, i] = -(-1.)**i, 1.
            c, drivers = np.zeros(n), (0,)
        else:
            p = np.vstack((np.eye(k), -np.eye(k)[0]))
            rows = np.zeros((1, n)); rows[0, 0] = rows[0, -1] = 1.
            c, drivers = np.r_[np.zeros(k), 1/256], tuple(range(k))
        _graph(data, p, c, rows, drivers)
        data["caps"][:] = 8.; data["force"][-1, 0] = 1/8
    else:
        raise ValueError(name)
    data["placement"] = placement
    if placement != "base":
        if placement != "cycle":
            raise ValueError(placement)
        quat = np.full(4, .5); R = base().quat_matrix(quat); shift = np.array([.5, -.25, .125])
        data["q"][:3] = R@data["q"][:3]+shift
        data["q"][3:7] = base().quat_product(quat, data["q"][3:7])
        data["v"][:3], data["v"][3:6] = R@data["v"][:3], R@data["v"][3:6]
        data["gravity"] = R@data["gravity"]
        data["force"], data["torque"] = data["force"]@R.T, data["torque"]@R.T
        for contact in data["contacts"]:
            for key in ("sphere", "point"):
                if key in contact:
                    contact[key] = R@contact[key]+shift
            if "normal" in contact:
                contact["normal"] = R@contact["normal"]
    return copy.deepcopy(data)


def expand(data, q, v):
    return np.r_[q[:7], data["P"]@q[7:]+data["offset"]], np.r_[v[:6], data["P"]@v[6:]]


def _part_motion(data, q, v):
    observed = base().jets(data, q, v)
    centers = np.vstack((data["root_com"], data["children"]["com"]))
    rows = []
    for part in data["parts"]:
        g = part["group"]; R = observed["rotation"][g]
        arm = R@(part["center"]-centers[g])
        spin, angular = observed["angular_velocity"][g], observed["angular_acceleration"][g]
        rows.append(dict(position=observed["com_position"][g]+arm,
                         velocity=observed["com_velocity"][g]+np.cross(spin, arm),
                         acceleration=observed["com_acceleration"][g]+np.cross(angular, arm)+np.cross(spin, np.cross(spin, arm)),
                         rotation=R@part["rotation"], omega=spin, alpha=angular))
    return observed, rows


def mechanics(data, q, v):
    if data["parts"] is None:
        return base().mechanics(data, q, v)
    observed, parts = _part_motion(data, q, v)
    basis = [_part_motion(data, q, column)[1] for column in np.eye(len(v))]
    d, B = len(v), len(data["children"]["mass"])+1
    Jv, Jw = np.zeros((B, 3, d)), np.zeros((B, 3, d))
    centers, speeds, masses = np.zeros((B, 3)), np.zeros((B, 3)), np.zeros(B)
    M, bias = np.zeros((d, d)), np.zeros(d)
    energy, momentum, angular_momentum = 0., np.zeros(3), np.zeros(3)
    for i, (part, motion) in enumerate(zip(data["parts"], parts, strict=True)):
        g, m = part["group"], part["mass"]
        jv = np.stack([column[i]["velocity"] for column in basis], axis=1)
        jw = np.stack([column[i]["omega"] for column in basis], axis=1)
        I = motion["rotation"]@np.diag(part["inertia"])@motion["rotation"].T
        M += m*jv.T@jv+jw.T@I@jw
        bias += jv.T@(m*(motion["acceleration"]-data["gravity"]))
        bias += jw.T@(I@motion["alpha"]+np.cross(motion["omega"], I@motion["omega"]))
        energy += (m*(motion["velocity"]@motion["velocity"])+motion["omega"]@I@motion["omega"])/2
        momentum += m*motion["velocity"]
        angular_momentum += np.cross(motion["position"], m*motion["velocity"])+I@motion["omega"]
        centers[g] += m*motion["position"]; speeds[g] += m*motion["velocity"]
        Jv[g] += m*jv; Jw[g] = jw; masses[g] += m
    centers /= masses[:, None]; speeds /= masses[:, None]; Jv /= masses[:, None, None]
    observed.update(com_position=centers, com_velocity=speeds)
    return dict(**observed, mass=M, bias=bias, linear_jacobian=Jv, angular_jacobian=Jw,
                energy=energy, momentum=momentum, angular_momentum=angular_momentum,
                authored_parts=parts, group_mass=masses)


def contact_wrenches(data, q, v, terms=None):
    terms = mechanics(data, q, v) if terms is None else terms
    wrench, rows = np.zeros((len(data["children"]["mass"])+1, 6)), []
    for law in data["contacts"]:
        first, second = law["first_group"], law.get("second_group")
        center = terms["position"][first]
        if law["kind"] == "sphere_pair":
            other = terms["position"][second]
            delta = other-center; distance = np.linalg.norm(delta)
            if distance == 0:
                raise ValueError("independent sphere centers have no unique normal")
            normal = -delta/distance
            p1, p2 = center-law["radius"]*normal, other+law["radius"]*normal
            gap = distance-2*law["radius"]
        elif law["kind"] == "box_sphere":
            R = terms["rotation"][first]
            local = R.T@(law["sphere"]-center)
            nearest = np.clip(local, -law["half"], law["half"])
            p1 = center+R@nearest
            delta = p1-law["sphere"]; distance = np.linalg.norm(delta)
            if distance == 0:
                raise ValueError("independent A02 domain excludes sphere center inside box")
            normal = delta/distance; p2 = law["sphere"]+law["radius"]*normal
            gap = distance-law["radius"]
        else:
            normal = law["normal"]
            distance = (center-law["point"])@normal
            p1, p2 = center-law["radius"]*normal, center-distance*normal
            gap = distance-law["radius"]
            # Original finite F09 box footprint; never an infinite-plane fallback.
            if np.linalg.norm(p2-law["point"], np.inf) > 1.:
                raise ValueError("independent lifecycle left its declared finite face")
        point = (p1+p2)/2
        magnitude = max(0., -law["stiffness"]*gap)
        force = magnitude*normal
        wrench[first, :3] += force
        wrench[first, 3:] += np.cross(point-terms["com_position"][first], force)
        if second is not None:
            wrench[second, :3] -= force
            wrench[second, 3:] += np.cross(point-terms["com_position"][second], -force)
        rows.append(dict(gap=gap, force_n=magnitude, normal=normal, first_surface=p1,
                         second_surface=p2, application=point, first_force=force,
                         second_force=-force, first_group=first, second_group=second))
    return wrench, rows


def advance(data, q, v, drive, h=None, *, wrench=None):
    h = data["h"] if h is None else h
    full_q, full_v = expand(data, np.asarray(q), np.asarray(v))
    t = mechanics(data, full_q, full_v)
    calculated, sites = contact_wrenches(data, full_q, full_v, t)
    wrench = calculated if wrench is None else np.asarray(wrench, float)
    input_load = base().applied(t, np.zeros(len(data["effort"])), data["force"], data["torque"])
    contact = base().applied(t, np.zeros(len(data["effort"])), wrench[:, :3], wrench[:, 3:])
    S, C, E = data["S"], data["C"], data["E"]
    M, bias, load = t["mass"], t["bias"], input_load+contact
    Mr, br, gr = S.T@M@S, S.T@bias, S.T@load
    motor = base().bounded_drive(Mr, br, gr, q, v, drive, data["kp"], data["kd"], data["caps"], h)
    # Solve the FULL constrained equations; never product reduced acceleration.
    system = np.block([[M, -C.T], [C, np.zeros((len(C), len(C)))]])
    solved = np.linalg.solve(system, np.r_[load+E@motor-bias, np.zeros(len(C))])
    full_a, multiplier = solved[:len(full_v)], solved[len(full_v):]
    independent = np.r_[np.arange(6), 6+np.asarray(data["drivers"], int)]
    reduced_a = full_a[independent]
    next_v = np.asarray(v)+h*reduced_a
    next_q = base().drift(data, np.asarray(q), next_v, h)
    next_full_q, next_full_v = expand(data, next_q, next_v)
    reaction = C.T@multiplier
    row_powers = multiplier*(C@full_v)
    return next_q, next_v, dict(**t, full_entry_position=full_q, full_entry_velocity=full_v,
        external=load, input_external=input_load, contact_generalized=contact,
        reduced_mass=Mr, reduced_bias=br, reduced_external=gr,
        reduced_linear_jacobian=t["linear_jacobian"]@S,
        reduced_angular_jacobian=t["angular_jacobian"]@S,
        full_acceleration=full_a, acceleration=reduced_a, drive=motor,
        constraint_effort_full=reaction, row_multipliers=multiplier, row_powers_w=row_powers,
        constraint_power_w=float(np.sum(row_powers)), root_constraint_residual=reaction[:6],
        full_applied=load+E@motor, reduced_applied=gr+np.r_[np.zeros(6), motor],
        next_full_position=next_full_q, next_full_velocity=next_full_v,
        wrench=wrench, independent_wrench=calculated, contact_sites=sites)