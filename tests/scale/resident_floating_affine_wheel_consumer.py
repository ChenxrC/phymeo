"""C11 fixed installed-wheel public consumer; AUTHORING_UNEXECUTED.

Run only after parent acceptance, from a fresh outside-checkout directory with
python -I -B. Independent NumPy transform derivatives/KKT are embedded here;
no checkout test/reference import, product private solver, or native comparator.
"""

import argparse
import copy
import gzip
import hashlib
import json
import sys
import traceback
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from importlib.metadata import distribution
from itertools import product
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import url2pathname
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import numpy as np

H = 1 / 1024
ORDER = (503, 91, 1207, 6)
ATOL, RTOL, WINDOW_ATOL = 5e-6, 5e-5, 2e-4
MAX_BYTES, MAX_GZIP_BYTES = 256 * 1024 * 1024, 64 * 1024 * 1024
BROAD_PHYSICAL = ("world_keys", "ticks", "shape_keys", "shape_revisions", "bounds",
                  "registered_pairs", "candidates")

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


def full_mechanics(data, q, v):
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
    rhs = drive[:, 2]+kp*(drive[:, 0]-q[7:]-h*v[6:])+kd*(drive[:, 1]-v[6:])-beta*free
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


def case(name, *, variant=None):
    children = _children([1., 2.], [[-.25, 0, 0], [.25, 0, 0]])
    children["inertia"][1] = [3/32, 7/64, 1/8]
    data = dict(name=name, variant=variant, placement="base",
                root_mass=2., root_com=np.zeros(3), root_inertia=np.array([.2, .3, .4]),
                root_inertia_frame=np.array([0., 0., 0., 1.]), gravity=np.zeros(3),
                children=children, parts=None, h=1/1024, fixture="jaws", contacts=())
    _graph(data, [[1.], [-1.]], [0., .004], [[1., 1.]], [0])
    if name == "A04":
        data.update(fixture="elbow", h=1/4096, root_com=np.array([1/32, -1/64, 0.]),
                    gravity=np.array([.5, -1., -2.]))
        children = _children([1., .75], [[1/4, 0, 1/16], [3/8, 0, 0]])
        children.update(parents=np.array([-1, 0]), kinds=("revolute", "revolute"),
                        axes=np.array([[0., 0., 1.], [0., 1., 0.]]),
                        com=np.array([[1/8, 1/32, 0.], [1/16, -1/32, 1/64]]),
                        inertia=np.array([[1/16, 5/64, 3/32], [1/32, 5/128, 3/64]]))
        data["children"] = children
        _graph(data, [[1.], [1.5]], [0., -1/8], [[-1.5, 1.]], [0])
        tool_rotation = axis_rotation([0, 0, 1], np.pi/2)
        data["parts"] = (
            dict(group=0, mass=2., center=data["root_com"].copy(), inertia=np.array([.2, .3, .4]), rotation=np.eye(3)),
            dict(group=1, mass=1., center=children["com"][0].copy(), inertia=children["inertia"][0].copy(), rotation=np.eye(3)),
            dict(group=2, mass=.75, center=children["com"][1].copy(), inertia=children["inertia"][1].copy(), rotation=np.eye(3)),
            dict(group=2, mass=.25, center=np.array([3/16, 1/16, -1/32])+tool_rotation@np.array([1/32, 0., 1/64]),
                 inertia=np.array([1/128, 5/512, 3/256]), rotation=tool_rotation),
        )
        data["q"][:3] = [1/8, -1/4, 3/8]
        data["v"][:6] = [1/8, -1/16, 1/32, .25, -.125, .1875]
        if variant != "p1":
            raise ValueError("the fixed installed subset uses only A04-p1")
        data["q"][3:7] = [0., np.sin(.375/2), 0., np.cos(.375/2)]
        data["q"][7], data["v"][6] = .125, .25
        data["force"][:] = [[.25, -.5, .75], [-.5, .375, -.25], [1.25, -.75, .5]]
        data["torque"][:] = [[1/32, -1/16, 3/64], [-1/32, 1/64, 1/16], [1/8, 1/16, -1/32]]
        data["drive"][0, 2], data["caps"][0] = 13/32, 4.
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
    else:
        raise ValueError(name)
    return copy.deepcopy(data)


def expand(data, q, v):
    return np.r_[q[:7], data["P"]@q[7:]+data["offset"]], np.r_[v[:6], data["P"]@v[6:]]


def _part_motion(data, q, v):
    observed = jets(data, q, v)
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
        return full_mechanics(data, q, v)
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
    input_load = applied(t, np.zeros(len(data["effort"])), data["force"], data["torque"])
    contact = applied(t, np.zeros(len(data["effort"])), wrench[:, :3], wrench[:, 3:])
    S, C, E = data["S"], data["C"], data["E"]
    M, bias, load = t["mass"], t["bias"], input_load+contact
    Mr, br, gr = S.T@M@S, S.T@bias, S.T@load
    motor = bounded_drive(Mr, br, gr, q, v, drive, data["kp"], data["kd"], data["caps"], h)
    # Solve the FULL constrained equations; never product reduced acceleration.
    system = np.block([[M, -C.T], [C, np.zeros((len(C), len(C)))]])
    solved = np.linalg.solve(system, np.r_[load+E@motor-bias, np.zeros(len(C))])
    full_a, multiplier = solved[:len(full_v)], solved[len(full_v):]
    independent = np.r_[np.arange(6), 6+np.asarray(data["drivers"], int)]
    reduced_a = full_a[independent]
    next_v = np.asarray(v)+h*reduced_a
    next_q = drift(data, np.asarray(q), next_v, h)
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

JAWS_XML = '''<?xml version="1.0"?>
<robot name="floating_mimic_jaws">
  <!-- SI. All COMs, joint origins and both +X axes are collinear.
       Root 2 kg; driver 1 kg; follower 2 kg. No hidden angular coupling. -->
  <link name="base">
    <inertial><mass value="2"/>
      <inertia ixx="0.2" iyy="0.3" izz="0.4" ixy="0" ixz="0" iyz="0"/>
    </inertial>
    <collision name="root_marker"><geometry><sphere radius="0.03125"/></geometry></collision>
  </link>
  <link name="left_jaw">
    <inertial><mass value="1"/>
      <inertia ixx="0.0625" iyy="0.078125" izz="0.09375" ixy="0" ixz="0" iyz="0"/>
    </inertial>
    <collision name="left_pad"><geometry><box size="0.0625 0.125 0.125"/></geometry></collision>
  </link>
  <joint name="driver" type="prismatic">
    <parent link="base"/><child link="left_jaw"/><origin xyz="-0.25 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-0.032" upper="0.032" effort="1.5" velocity="0.2"/>
  </joint>
  <link name="right_jaw">
    <inertial><mass value="2"/>
      <inertia ixx="0.09375" iyy="0.109375" izz="0.125" ixy="0" ixz="0" iyz="0"/>
    </inertial>
    <collision name="right_pad"><geometry><box size="0.0625 0.125 0.125"/></geometry></collision>
  </link>
  <joint name="follower" type="prismatic">
    <parent link="base"/><child link="right_jaw"/><origin xyz="0.25 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-0.012" upper="0.020" effort="0.125" velocity="0.08"/>
    <mimic joint="driver" multiplier="-1" offset="0.004"/>
  </joint>
</robot>
'''

ELBOW_XML = '''<?xml version="1.0"?>
<robot name="floating_mimic_elbow">
  <!-- SI. Root COM is not its authored origin. The two moving axes are Z
       then body-local Y; the tool is a real positive-mass fixed descendant. -->
  <link name="base">
    <inertial><origin xyz="0.03125 -0.015625 0"/><mass value="2"/>
      <inertia ixx="0.2" iyy="0.3" izz="0.4" ixy="0" ixz="0" iyz="0"/>
    </inertial>
    <collision name="root_marker"><geometry><sphere radius="0.03125"/></geometry></collision>
  </link>
  <link name="upper">
    <inertial><origin xyz="0.125 0.03125 0"/><mass value="1"/>
      <inertia ixx="0.0625" iyy="0.078125" izz="0.09375" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>
  <joint name="shaft" type="revolute">
    <parent link="base"/><child link="upper"/><origin xyz="0.25 0 0.0625"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1" upper="1" effort="4" velocity="2"/>
  </joint>
  <link name="forearm">
    <inertial><origin xyz="0.0625 -0.03125 0.015625"/><mass value="0.75"/>
      <inertia ixx="0.03125" iyy="0.0390625" izz="0.046875" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>
  <joint name="gear" type="revolute">
    <parent link="upper"/><child link="forearm"/><origin xyz="0.375 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.25" upper="1.25" effort="0.25" velocity="1.5"/>
    <mimic joint="shaft" multiplier="1.5" offset="-0.125"/>
  </joint>
  <link name="tool">
    <inertial><origin xyz="0.03125 0 0.015625"/><mass value="0.25"/>
      <inertia ixx="0.0078125" iyy="0.009765625" izz="0.01171875" ixy="0" ixz="0" iyz="0"/>
    </inertial>
    <collision name="tool_box"><geometry><box size="0.125 0.0625 0.0625"/></geometry></collision>
  </link>
  <joint name="tool_mount" type="fixed">
    <parent link="forearm"/><child link="tool"/>
    <origin xyz="0.1875 0.0625 -0.03125" rpy="0 0 1.5707963267948966"/>
  </joint>
</robot>
'''

def encode(value):
    module = sys.modules.get("phymeo")
    program_type = None if module is None else getattr(module, "DrivenRobotProgram", None)
    if program_type is not None and isinstance(value, program_type):
        return encode(value.problem_signature())
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return {field.name: encode(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): encode(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return encode(value.tolist())
    if isinstance(value, np.generic):
        return encode(value.item())
    if isinstance(value, (tuple, list)):
        return [encode(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return {"nonfinite": repr(value)}
    return value


def fingerprint(value):
    return json.dumps(encode(value), sort_keys=True, allow_nan=False, separators=(",", ":"))


def close(actual, expected, *, atol=ATOL, rtol=RTOL):
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol, equal_nan=False)


def readonly(value):
    if isinstance(value, np.ndarray):
        assert not value.flags.writeable
        try:
            value.setflags(write=True)
        except ValueError:
            pass
        else:
            raise AssertionError("public owned array can be made writable")
    elif is_dataclass(value):
        for field in fields(value):
            readonly(getattr(value, field.name))
    elif isinstance(value, Mapping):
        for item in value.values():
            readonly(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            readonly(item)


def rejected(call, label):
    try:
        call()
    except (ValueError, TypeError, RuntimeError) as error:
        return dict(label=label, type=type(error).__name__, message=str(error))
    raise AssertionError(label + " unexpectedly succeeded")



def box(low, high):
    return np.array(list(product(*zip(low, high, strict=True))), float)


def numbers(values):
    return " ".join(map(str, values))


def bilateral_xml(data):
    root = ET.Element("robot", name="c11_bilateral")
    c = data["children"]
    for i, (mass, center, inertia) in enumerate(zip(
            np.r_[data["root_mass"], c["mass"]],
            np.vstack((data["root_com"], c["com"])),
            np.vstack((data["root_inertia"], c["inertia"])), strict=True)):
        link = ET.SubElement(root, "link", name="base" if i == 0 else f"body_{i-1:02d}")
        inertial = ET.SubElement(link, "inertial")
        ET.SubElement(inertial, "origin", xyz=numbers(center))
        ET.SubElement(inertial, "mass", value=str(mass))
        ET.SubElement(inertial, "inertia", ixx=str(inertia[0]), iyy=str(inertia[1]),
                      izz=str(inertia[2]), ixy="0", ixz="0", iyz="0")
    for i, parent in enumerate(c["parents"]):
        joint = ET.SubElement(root, "joint", name=f"axis_{i:02d}", type=c["kinds"][i])
        ET.SubElement(joint, "parent", link="base" if parent < 0 else f"body_{parent:02d}")
        ET.SubElement(joint, "child", link=f"body_{i:02d}")
        ET.SubElement(joint, "origin", xyz=numbers(c["offsets"][i]))
        ET.SubElement(joint, "axis", xyz=numbers(c["axes"][i]))
        ET.SubElement(joint, "limit", lower="-1", upper="1", velocity="1", effort="8")
        if i not in data["drivers"]:
            j = int(np.flatnonzero(data["P"][i])[0])
            ET.SubElement(joint, "mimic", joint=f"axis_{data['drivers'][j]:02d}",
                          multiplier=str(data["P"][i, j]), offset=str(data["offset"][i]))
    return ET.tostring(root, encoding="unicode")


def fixed_data():
    # Original C5/C7 A03: three independent Px bodies, original CHAIN rows.
    return dict(name="fixed-A03", kind="affine", h=H, delay=0,
                parents=np.full(3, -1, int), kinds=("prismatic",)*3,
                offsets=np.array([[0., 0., 0.], [0., .5, 0.], [0., -.5, 0.]]),
                axes=np.tile([1., 0., 0.], (3, 1)), mass=np.array([2., 1., 1.]),
                com=np.zeros((3, 3)), inertia=np.full((3, 3), 1/8), gravity=np.zeros(3),
                P=np.array([[1.], [2.], [-1.]]), offset=np.array([0., 1/8, 1/8]),
                C=np.array([[-2., 1., 0.], [0., .5, 1.]]),
                rows=((0, 1, 2., 1/8), (1, 2, -.5, 3/16)),
                q=np.zeros(1), v=np.zeros(1), drive=np.array([[0., 0., 7.]]),
                kp=np.zeros(1), kd=np.zeros(1), caps=np.full(1, 16.),
                effort=np.zeros(3), force=np.zeros((3, 3)), torque=np.zeros((3, 3)),
                lower=np.full(3, -8.), upper=np.full(3, 8.), speed=np.full(3, 16.),
                rating=np.full(3, 16.))


def fixed_xml(data):
    root = ET.Element("robot", name="resident_affine_frozen")
    ET.SubElement(root, "link", name="base")
    followers = {f: (d, a, b) for d, f, a, b in data["rows"]}
    for i in range(3):
        link = ET.SubElement(root, "link", name=f"link{i}")
        inertial = ET.SubElement(link, "inertial")
        ET.SubElement(inertial, "origin", xyz=numbers(data["com"][i]))
        ET.SubElement(inertial, "mass", value=str(data["mass"][i]))
        x, y, z = data["inertia"][i]
        ET.SubElement(inertial, "inertia", ixx=str(x), iyy=str(y), izz=str(z),
                      ixy="0", ixz="0", iyz="0")
        col = ET.SubElement(link, "collision", name="material")
        ET.SubElement(col, "origin", xyz="0.25 -0.125 0", rpy="0 0 0.7853981633974483")
        ET.SubElement(ET.SubElement(col, "geometry"), "sphere", radius="0.0625")
        joint = ET.SubElement(root, "joint", name=f"joint{i}", type="prismatic")
        ET.SubElement(joint, "parent", link="base")
        ET.SubElement(joint, "child", link=f"link{i}")
        ET.SubElement(joint, "origin", xyz=numbers(data["offsets"][i]), rpy="0 0 0")
        ET.SubElement(joint, "axis", xyz="1 0 0")
        ET.SubElement(joint, "limit", lower="-8.0", upper="8.0", velocity="16.0", effort="16.0")
        if i in followers:
            d, a, b = followers[i]
            ET.SubElement(joint, "mimic", joint=f"joint{d}", multiplier=str(a), offset=str(b))
    return ET.tostring(root, encoding="unicode")


def write_asset(directory, xml):
    directory.mkdir(parents=True, exist_ok=False)
    path = directory / "declared.urdf"
    content = xml.encode("utf-8")
    path.write_bytes(content)
    return path, dict(path=str(path), text=xml, bytes=len(content),
                      sha256=hashlib.sha256(content).hexdigest())


def floating_item(ph, directory, name, variant=None):
    data = case(name, variant=variant)
    xml = ELBOW_XML if name == "A04" else JAWS_XML if name == "lifecycle" else bilateral_xml(data)
    path, source = write_asset(directory, xml)
    asset = ph.load_floating_mimic_urdf(path, gravity=data["gravity"],
                                      collapse_fixed_joints=name == "A04")
    close(asset.matrix, data["P"], atol=0, rtol=0)
    close(asset.offset_vector, data["offset"], atol=2e-12, rtol=0)
    close(asset.constraint_rows, data["C"][:, 6:], atol=0, rtol=0)
    tree = ph.FloatingAffineRobotTree(
        asset, ph.FloatingRobotState(data["q"], data["v"]), kp=data["kp"], kd=data["kd"],
        effort_limit=data["caps"], delay_steps=0)
    external = ph.FloatingRobotInputs(np.zeros(len(data["effort"])), data["force"], data["torque"])
    command = tree.inputs(data["drive"], external=external)
    return dict(kind="floating_affine", data=data, asset=asset, source=source,
                tree=tree, command=command, external=external, delay=0)


def declarations(ph, directory):
    items = {503: floating_item(ph, directory / "503", "lifecycle"),
             91: floating_item(ph, directory / "91", "A05")}
    model = ph.RobotModel(np.array([-1]), ("prismatic",), np.array([[0., 0., 3/32]]),
                         np.array([[0., 0., 0., 1.]]), np.array([[0., 0., 1.]]),
                         np.array([2.]), np.zeros((1, 3)), np.full((1, 3), 1/8), np.zeros(3))
    program = ph.DrivenRobotProgram(model, kp=np.zeros(1), kd=np.zeros(1), effort_limit=np.array([8.]))
    tree = ph.FixedRobotTree(program, program.initial_state(ph.RobotState(np.zeros(1), np.zeros(1))))
    command = program.inputs(np.zeros(1), np.zeros(1), feedforward=np.zeros(1), loads=ph.RobotInputs.zeros(1))
    items[1207] = dict(kind="ordinary", data=dict(name="ordinary-P", h=H, q=np.zeros(1),
                         v=np.zeros(1), mass=np.array([2.]), origin=np.array([[0., 0., 3/32]]),
                         axis=np.array([[0., 0., 1.]]), inertia=np.full((1, 3), 1/8),
                         com=np.zeros((1, 3)), gravity=np.zeros(3), delay=0), tree=tree, command=command)
    data = fixed_data()
    path, source = write_asset(directory / "6", fixed_xml(data))
    asset = ph.load_mimic_urdf(path, gravity=data["gravity"])
    assert asset.tree.joint_names == ("joint0", "joint1", "joint2") and asset.joint_names == ("joint0",)
    close(asset.matrix, data["P"], atol=0, rtol=0)
    close(asset.offset_vector, data["offset"], atol=0, rtol=0)
    close(asset.constraint_rows, data["C"], atol=0, rtol=0)
    tree = ph.AffineRobotTree(asset, ph.RobotState(data["q"], data["v"]),
                             kp=data["kp"], kd=data["kd"], effort_limit=data["caps"], delay_steps=0)
    command = tree.inputs(data["drive"][:, 0], data["drive"][:, 1],
                          feedforward=data["drive"][:, 2], loads=ph.RobotInputs(
                              data["effort"], data["force"], data["torque"]))
    items[6] = dict(kind="affine", data=data, asset=asset, source=source, tree=tree, command=command)
    assert tuple(items) == ORDER
    return items


def selected(item, tick):
    # This installed subset preserves the original delay0 inputs. The separate
    # core delay1 tests retain their FP32-initial-target contract unchanged.
    assert item["data"].get("delay", 0) == 0
    return item["data"]["drive"]


def contact_geometry(ph, batch, items):
    shapes, bindings, sources, laws = [], [], {}, {}
    for world, item in items.items():
        source = {}; checkpoint = batch.snapshot([world])[world]
        def add(key, group, kind, points, ids, radius=0., revision=7):
            shape = ph.CollisionShape(key, kind, ids, radius, revision=revision)
            if group is None:
                bound = ph.ResidentStaticBinding(key, ph.CollisionMotion(points))
            elif item["kind"] == "floating_affine":
                bound = ph.ResidentFloatingAffineTreeBinding(key, world, group, points,
                    topology=checkpoint.topology, graph_token=item["command"].graph_token)
            elif item["kind"] == "affine":
                bound = ph.ResidentAffineTreeBinding(key, world, group, points, topology=checkpoint.topology)
            else:
                bound = ph.ResidentTreeBinding(key, world, group, points, topology=checkpoint.topology)
            shapes.append(shape); bindings.append(bound); source[key] = (shape, group, points)
            return key
        if item["kind"] == "floating_affine":
            rule = item["data"]["contacts"][0]
            a = add(ph.CollisionKey(world, 100+rule["first_group"], 11), rule["first_group"],
                    "sphere", np.zeros((1, 3)), (31,), rule["radius"])
            if rule["kind"] == "sphere_pair":
                b = add(ph.CollisionKey(world, 100+rule["second_group"], 12), rule["second_group"],
                        "sphere", np.zeros((1, 3)), (1031,), rule["radius"])
            else:
                b = add(ph.CollisionKey(world, 901, 12), None, "hull", box([-1., -1., -.25], [1., 1., 0.]),
                        tuple(1031+7*i for i in range(8)))
        elif item["kind"] == "ordinary":
            a = add(ph.CollisionKey(world, 100, 11), 0, "hull", box([-1/8]*3, [1/8]*3), tuple(range(101, 109)), revision=0)
            b = add(ph.CollisionKey(world, 900, 1), None, "hull", box([-.5, -.5, -.25], [.5, .5, 0.]), tuple(range(201, 209)), revision=0)
        else:
            site = item["asset"].tree.sites["link2"]
            local = np.asarray(site.local_position_m)[None, :]
            assert site.link_index == 2
            a = add(ph.CollisionKey(world, 102, 11), 2, "sphere", local, (71,), 1/8)
            b = add(ph.CollisionKey(world, 900, 1), None, "hull", box([3/64-1/4, -.75, -.25], [3/64, -.25, .25]),
                    tuple(range(201, 209)), revision=3)
        laws[tuple(sorted((a, b)))] = ph.RobotContactMaterial(64, 0, 0)
        sources[world] = source
    return ph.CollisionModel(tuple(shapes)), tuple(bindings), sources, laws

def floating_audit(item, entry, accepted, geometry, diagnostic, summary, *, h=None, wrench=None):
    data = item["data"]
    h = data["h"] if h is None else h
    selected_drive = selected(item, entry.tick)
    q, v, truth = advance(data, np.asarray(entry.position), np.asarray(entry.velocity), selected_drive, h, wrench=wrench)
    n, k = data["P"].shape
    assert (len(accepted.position), len(accepted.velocity), len(accepted.full_position), len(accepted.full_velocity)) == (7+k, 6+k, 7+n, 6+n)
    assert (geometry.joint_count, geometry.body_count, geometry.velocity_count, geometry.configuration_count) == (n, n+1, n+6, n+7)
    assert accepted.tick == entry.tick+1 and accepted.control == entry.control
    assert (diagnostic.entry_tick, diagnostic.exit_tick, diagnostic.dt_s) == (entry.tick, accepted.tick, h)
    close(diagnostic.entry_position, entry.position)
    close(diagnostic.entry_velocity, entry.velocity)
    close(diagnostic.full_entry_position, entry.full_position)
    close(diagnostic.full_entry_velocity, entry.full_velocity)
    for field in ("mass", "bias", "external", "linear_jacobian", "angular_jacobian",
                  "reduced_mass", "reduced_bias", "reduced_external", "reduced_linear_jacobian",
                  "reduced_angular_jacobian", "acceleration", "full_acceleration", "drive",
                  "constraint_effort_full", "row_multipliers", "row_powers_w", "root_constraint_residual"):
        close(getattr(diagnostic, field), truth[field], atol=2e-5 if field in ("mass", "reduced_mass", "constraint_effort_full", "row_multipliers", "row_powers_w", "root_constraint_residual") else ATOL)
    close(accepted.position, q)
    close(accepted.velocity, v)
    close(accepted.full_position, truth["next_full_position"])
    close(accepted.full_velocity, truth["next_full_velocity"])
    close(accepted.applied_effort, truth["drive"])
    close(accepted.full_applied_effort, (data["E"]@truth["drive"])[6:])
    close(accepted.generalized_applied, truth["reduced_applied"])
    close(accepted.full_generalized_applied, truth["full_applied"])
    close(geometry.position, accepted.full_position)
    close(geometry.velocity, accepted.full_velocity)
    close(geometry.independent_position, accepted.position)
    close(geometry.independent_velocity, accepted.velocity)
    assert geometry.joint_names == item["asset"].joint_names
    assert geometry.full_joint_names == item["asset"].tree.joint_names
    actual_q, actual_v = np.asarray(accepted.full_position), np.asarray(accepted.full_velocity)
    end = mechanics(data, actual_q, actual_v)
    for field, expected in (("com_position_m", "com_position"), ("linear_velocity_m_s", "com_velocity"),
                            ("angular_velocity_rad_s", "angular_velocity")):
        close(getattr(geometry, field), end[expected])
    if data["parts"] is None:
        close(geometry.link_position_m, end["position"])
        close(geometry.link_rotation, end["rotation"])
    else:
        # Compare authored frames, not principal group axes with raw URDF axes.
        for group, name in enumerate(("base", "upper", "forearm")):
            site = item["asset"].tree.sites[name]
            actual_R = geometry.link_rotation[group]@quat_matrix(site.local_orientation_xyzw)
            actual_p = geometry.link_position_m[group]+geometry.link_rotation[group]@site.local_position_m
            close(actual_R, end["rotation"][group])
            close(actual_p, end["position"][group])
    close(summary["kinetic_energy_j"], end["energy"])
    assert diagnostic.control_residual <= 5e-5 and diagnostic.equation_residual <= 2e-5
    full_motor = data["E"]@np.asarray(diagnostic.drive)
    lhs = diagnostic.mass@diagnostic.full_acceleration+diagnostic.bias
    residual = lhs-diagnostic.external-full_motor
    scale = max(1., float(np.max(abs(lhs))), float(np.max(abs(diagnostic.external+full_motor))))
    close(residual, diagnostic.constraint_effort_full, atol=2e-5*scale, rtol=0)
    close(residual[:6], np.zeros(6), atol=2e-5*scale, rtol=0)
    close(data["S"].T@residual, np.zeros(k+6), atol=2e-5*scale, rtol=0)
    close(residual, data["C"].T@diagnostic.row_multipliers, atol=2e-5*scale, rtol=0)
    powers = np.asarray(diagnostic.row_multipliers)*(data["C"]@np.asarray(entry.full_velocity))
    close(diagnostic.row_powers_w, powers, atol=2e-5, rtol=0)
    close(diagnostic.constraint_power_w, np.sum(powers), atol=2e-5, rtol=0)
    power_scale = max(1., float(np.sum(abs(np.asarray(entry.full_velocity)*residual))))
    assert np.max(abs(powers), initial=0.) <= 2e-5*power_scale
    assert abs(diagnostic.constraint_power_w) <= 2e-5*power_scale
    assert np.asarray(accepted.position[3:7])@np.asarray(entry.position[3:7]) >= 0
    for value in (diagnostic.mass, diagnostic.reduced_mass, diagnostic.full_acceleration,
                  diagnostic.row_multipliers, geometry.independent_position):
        assert not value.flags.writeable
    return dict(same_entry=truth, next_q=q, next_v=v, actual_end=end, raw_full_residual=residual,
                original_row_power=powers, power_scale=power_scale)


def material_audit(receipt, diagnostic, sources):
    full_velocity = np.asarray(getattr(receipt.entry_state, "full_velocity", receipt.entry_state.velocity), dtype=float)
    wrench = np.zeros_like(receipt.law_wrench_fp64, dtype=float)
    static, total_work, total_material_work = {}, 0.0, 0.0
    for item in receipt.sites:
        witness = item.geometry.witness
        point = np.asarray(item.geometry.load_point, float)
        np.testing.assert_array_equal(item.first_force_n, -item.second_force_n)
        work = material_work = 0.0
        for side, key in zip(("first", "second"), item.pair, strict=True):
            shape, group, local = sources[key]
            weights = getattr(witness, side + "_weights")
            by_id = dict(zip(shape.vertex_ids, local, strict=True))
            material_local = sum(weight * by_id[node] for node, weight in weights)
            skin = getattr(witness, side + "_offset")
            surface = np.asarray(getattr(witness, side + "_point"), float)
            if group is None:
                rebuilt = material_local
                reference, velocity = local.mean(axis=0), np.zeros(3)
                displacement = application = np.zeros(3)
            else:
                entry, end = receipt.entry_geometry, receipt.accepted_geometry
                p0, p1 = entry.link_position_m[group], end.link_position_m[group]
                r0, r1 = entry.link_rotation[group], end.link_rotation[group]
                rebuilt = p0 + r0 @ material_local
                reference = entry.com_position_m[group]
                omega = diagnostic.angular_jacobian[group] @ full_velocity
                velocity = diagnostic.linear_jacobian[group] @ full_velocity + np.cross(omega, point - reference)
                displacement = p1 + r1 @ np.linalg.solve(r0, surface - p0) - surface
                application = p1 + r1 @ np.linalg.solve(r0, point - p0) - point
            rebuilt = rebuilt + (np.zeros(3) if skin is None else skin)
            close(rebuilt, surface, atol=1e-10, rtol=0)
            close(getattr(item, side + "_reference_point_m"), reference, atol=1e-10, rtol=0)
            close(getattr(item, side + "_velocity_m_s"), velocity)
            close(getattr(item, side + "_material_displacement_m"), displacement, atol=1e-10, rtol=0)
            close(getattr(item, side + "_application_displacement_m"), application, atol=1e-10, rtol=0)
            force = np.asarray(getattr(item, side + "_force_n"))
            work += float(force @ application)
            material_work += float(force @ displacement)
            contribution = np.r_[force, np.cross(point - reference, force)]
            if group is None:
                static.setdefault(key, np.zeros(6))
                static[key] += contribution
            else:
                wrench[group] += contribution
        close(item.application_work_j, work, atol=1e-12, rtol=1e-10)
        close(item.material_work_j, material_work, atol=1e-12, rtol=1e-10)
        total_work += work
        total_material_work += material_work
    close(receipt.law_wrench_fp64, wrench)
    np.testing.assert_array_equal(receipt.uploaded_wrench_fp32, np.asarray(wrench, np.float32))
    projected = np.einsum("bij,bi->j", diagnostic.linear_jacobian, wrench[:, :3]) + np.einsum("bij,bi->j", diagnostic.angular_jacobian, wrench[:, 3:])
    close(receipt.contact_generalized, projected)
    assert set(receipt.static_reactions) == set(static)
    for key, reaction in receipt.static_reactions.items():
        close(reaction.reference_point_m, sources[key][2].mean(axis=0), atol=1e-12, rtol=0)
        close(reaction.force_n, static.get(key, np.zeros(6))[:3])
        close(reaction.torque_nm, static.get(key, np.zeros(6))[3:])
    return dict(application_work_j=total_work, material_work_j=total_material_work)



def fixed_truth(item, q, v, *, wrench=None):
    """Independent declared Px forest, literal diagonal full mass and chain KKT."""
    data = item["data"]; q, v = np.asarray(q), np.asarray(v)
    affine = item["kind"] == "affine"
    if affine:
        P, c, C = data["P"], data["offset"], data["C"]
        full_q, full_v = P@q+c, P@v
        M = np.diag([2., 1., 1.]); J = np.zeros((3, 3, 3))
        for i in range(3):
            J[i, 0, i] = 1.
        centers = data["offsets"]+np.column_stack((full_q, np.zeros((3, 2))))
        gap = centers[2, 0]-3/64-1/8
        assert centers[2, 0] > 3/64 and abs(centers[2, 1]+.5) < .25 and abs(centers[2, 2]) < .25
        normal = max(0., -64*gap)
        independent_wrench = np.zeros((3, 6)); independent_wrench[2, 0] = normal
        motor_full, drive = np.array([7., 0., 0.]), np.array([7.])
    else:
        P, c, C = np.ones((1, 1)), np.zeros(1), np.zeros((0, 1))
        full_q, full_v, M = q, v, np.array([[2.]])
        J = np.array([[[0.], [0.], [1.]]])
        centers = np.array([[0., 0., 3/32+q[0]]])
        gap = q[0]-1/32
        normal = max(0., -64*gap)
        independent_wrench = np.array([[0., 0., normal, 0., 0., 0.]])
        motor_full, drive = np.zeros(1), np.zeros(1)
    used = independent_wrench if wrench is None else np.asarray(wrench, float)
    contact = np.einsum("bci,bc->i", J, used[:, :3])
    system = np.block([[M, -C.T], [C, np.zeros((len(C), len(C)))]])
    solved = np.linalg.solve(system, np.r_[contact+motor_full, np.zeros(len(C))])
    full_a, lam = solved[:len(full_v)], solved[len(full_v):]
    a = full_a[:1]
    next_v, next_q = v+H*a, q+H*(v+H*a)
    return next_q, next_v, dict(mass=M, bias=np.zeros(len(full_v)), external=contact,
        input_external=np.zeros(len(full_v)), contact_generalized=contact,
        linear_jacobian=J, angular_jacobian=np.zeros_like(J), reduced_mass=P.T@M@P,
        reduced_bias=np.zeros(1), reduced_external=P.T@contact,
        reduced_linear_jacobian=J@P, reduced_angular_jacobian=np.zeros_like(J@P),
        acceleration=a, full_acceleration=full_a, constraint_effort_full=C.T@lam,
        row_multipliers=lam, row_powers_w=lam*(C@full_v), constraint_power_w=float(lam@(C@full_v)),
        drive=drive, full_motor=motor_full, full_q=full_q, full_v=full_v,
        next_full_q=P@next_q+c, next_full_v=P@next_v, P=P, C=C,
        com_position=centers, com_velocity=np.einsum("bci,i->bc", J, full_v),
        energy=float(full_v@M@full_v/2), independent_wrench=independent_wrench,
        gap=gap, normal_force=normal)


def audit_patch_contract(receipt, sources, truth):
    """Independent law magnitude; stored frames only rebuild public material records."""
    sums = {}
    for site in receipt.sites:
        assert site.pair == tuple(sorted(site.pair))
        assert all(key.world == receipt.world and key in sources for key in site.pair)
        law, geometry = site.law, site.geometry
        assert (law.stiffness, law.damping, law.friction, law.slip_speed_m_s) == (64., 0., 0., .01)
        assert geometry.weight > 0
        sums[site.pair] = sums.get(site.pair, 0.)+geometry.weight
        close(np.linalg.norm(site.first_force_n), max(0., -64*geometry.witness.gap_m)*geometry.weight)
        if "contact_sites" in truth:
            expected_site = truth["contact_sites"][0]
            close(geometry.witness.gap_m, expected_site["gap"])
            close(geometry.witness.first_point, expected_site["first_surface"])
            close(geometry.witness.second_point, expected_site["second_surface"])
            close(geometry.load_point, expected_site["application"])
            close(geometry.witness.normal, -expected_site["normal"])
            close(site.first_force_n, geometry.weight*expected_site["first_force"])
        else:
            close(geometry.witness.gap_m, truth["gap"])
        for side, key in zip(("first", "second"), site.pair, strict=True):
            shape = sources[key][0]
            weights = getattr(geometry.witness, side+"_weights")
            assert weights and all(node in shape.vertex_ids and weight >= 0 for node, weight in weights)
            close(sum(weight for _, weight in weights), 1., atol=1e-12, rtol=0)
    for total in sums.values():
        close(total, 1., atol=1e-12, rtol=0)
    close(receipt.law_wrench_fp64, truth["independent_wrench"])
    # There may be no geometric candidate while the finite pair is separated.
    if not receipt.sites:
        close(truth["independent_wrench"], np.zeros_like(truth["independent_wrench"]))
    readonly(receipt)


def contact_audit(item, receipt, state, geometry, diagnostic, summary, sources):
    data, entry, accepted = item["data"], receipt.entry_state, receipt.accepted_state
    assert receipt.exit_tick == accepted.tick == entry.tick+1
    assert receipt.entry_tick == entry.tick and receipt.dt_s == H
    assert accepted == state and fingerprint(receipt.accepted_geometry) == fingerprint(geometry)
    assert accepted.control == entry.control and accepted.history == entry.history == ()
    meta = receipt.diagnostics
    if item["kind"] == "floating_affine":
        n, k = data["P"].shape
        assert meta["coordinate_kind"] == "floating_affine"
        assert tuple(meta[f] for f in ("joint_count", "body_count", "velocity_count", "position_count")) == (n, n+1, n+6, n+7)
        assert tuple(meta[f] for f in ("full_dof_count", "reduced_dof_count", "independent_dof_count")) == (n+6, k+6, k)
        assert meta["graph_token"] == item["command"].graph_token
        close(meta["projection_matrix"], data["S"], atol=0, rtol=0)
        close(meta["offset_vector"], data["offset"])
        close(meta["constraint_rows"], data["C"][:, 6:], atol=0, rtol=0)
        assert fingerprint(meta["floating_affine_diagnostic"]) == fingerprint(diagnostic)
        audit = floating_audit(item, entry, accepted, geometry, diagnostic, summary,
                               wrench=np.asarray(receipt.uploaded_wrench_fp32, float))
        truth = audit["same_entry"]
        close(receipt.selected_drive, selected(item, entry.tick), atol=0, rtol=0)
        assert len(entry.control) == 5 and entry.control[0] == item["command"].graph_token
        assert receipt.selected_drive.shape == (k, 3) and receipt.actual_drive.shape == (k,)
        assert receipt.law_wrench_fp64.shape == (n+1, 6)
        energy_end = audit["actual_end"]["energy"]
    else:
        q, v, truth = fixed_truth(item, entry.position, entry.velocity,
                                   wrench=np.asarray(receipt.uploaded_wrench_fp32, float))
        close(accepted.position, q); close(accepted.velocity, v)
        for field in ("mass", "bias", "external", "linear_jacobian", "angular_jacobian", "acceleration", "drive"):
            close(getattr(diagnostic, field), truth[field])
        close(geometry.com_position_m, truth["com_position"]+
              np.einsum("bci,i->bc", truth["linear_jacobian"], truth["next_full_q"]-truth["full_q"]))
        close(geometry.position, truth["next_full_q"])
        close(geometry.velocity, truth["next_full_v"])
        if item["kind"] == "affine":
            assert meta["coordinate_kind"] == "affine"
            assert (meta["full_dof_count"], meta["independent_dof_count"]) == (3, 1)
            assert meta["graph_token"] == item["command"].graph_token
            close(meta["projection_matrix"], data["P"], atol=0, rtol=0)
            close(meta["offset_vector"], data["offset"])
            close(meta["constraint_rows"], data["C"], atol=0, rtol=0)
            for field in ("reduced_mass", "reduced_bias", "reduced_external", "reduced_linear_jacobian",
                          "reduced_angular_jacobian", "full_acceleration", "constraint_effort_full", "row_multipliers", "row_powers_w"):
                close(getattr(diagnostic, field), truth[field], atol=2e-5)
            close(accepted.full_position, truth["next_full_q"])
            close(accepted.full_velocity, truth["next_full_v"])
            close(diagnostic.constraint_power_w, truth["constraint_power_w"], atol=2e-5)
            assert fingerprint(meta["affine_diagnostic"]) == fingerprint(diagnostic)
            close(data["C"]@np.asarray(entry.full_velocity), np.zeros(2), atol=2e-5)
            close(diagnostic.mass@diagnostic.full_acceleration-diagnostic.external-truth["full_motor"],
                  diagnostic.constraint_effort_full, atol=2e-5)
            close(data["P"].T@diagnostic.constraint_effort_full, np.zeros(1), atol=2e-5)
        energy_end = float(truth["next_full_v"]@truth["mass"]@truth["next_full_v"]/2)
        close(summary["kinetic_energy_j"], energy_end)
        audit = dict(same_entry=truth, next_q=q, next_v=v)
    assert diagnostic.control_residual <= 5e-5 and diagnostic.equation_residual <= 2e-5
    assert meta["control_residual"] == diagnostic.control_residual
    assert meta["equation_residual"] == diagnostic.equation_residual
    close(receipt.contact_generalized, truth["contact_generalized"])
    close(receipt.input_external, truth["input_external"])
    close(receipt.actual_drive, truth["drive"])
    close(receipt.actual_drive, accepted.applied_effort)
    close(diagnostic.external, receipt.input_external+receipt.contact_generalized)
    for field in ("mass", "bias", "linear_jacobian", "angular_jacobian"):
        close(meta[field], getattr(diagnostic, field))
    close(meta["total_external"], diagnostic.external)
    close(meta["acceleration"], getattr(diagnostic, "full_acceleration", diagnostic.acceleration))
    close(meta["wrench_quantization_error"], receipt.uploaded_wrench_fp32.astype(float)-receipt.law_wrench_fp64,
          atol=0, rtol=0)
    audit_patch_contract(receipt, sources, truth)
    work = material_audit(receipt, diagnostic, sources)
    for field, value in work.items():
        close(meta[field], value, atol=1e-12, rtol=1e-10)
    close(meta["entry_kinetic_energy_j"], truth["energy"])
    close(meta["accepted_kinetic_energy_j"], energy_end)
    close(meta["kinetic_energy_change_j"], energy_end-truth["energy"])
    close(meta["entry_power_times_h_j"], H*sum(np.asarray(s.first_force_n)@s.first_velocity_m_s+
          np.asarray(s.second_force_n)@s.second_velocity_m_s for s in receipt.sites), atol=1e-12, rtol=1e-10)
    return dict(**audit, material_work=work)

def broad(result):
    return {name: getattr(result, name) for name in ("topology_generation", "state_epoch", *BROAD_PHYSICAL)}


def contact_capture(batch, plan):
    return dict(state=batch.snapshot(), geometry=batch.tree_kinematics(), diagnostics=batch.tree_diagnostics(),
                energy=batch.summary(), receipts=dict(plan.last_accepted), status=dict(plan.last_status),
                broad=broad(plan.query()), batch_stats=batch.stats(), plan_stats=plan.stats())


def bounds_audit(row, sources, model):
    result = row["broad"]
    assert set(result["world_keys"]) == {key.world for key in sources}
    assert tuple(result["shape_keys"]) == tuple(shape.key for shape in model.shapes)
    wanted = {}
    for key, (shape, group, local) in sources.items():
        if group is None:
            points = local
        else:
            geometry = row["geometry"][key.world]
            points = geometry.link_position_m[group] + local @ geometry.link_rotation[group].T
        low, high = points.min(axis=0)-shape.radius, points.max(axis=0)+shape.radius
        index = result["shape_keys"].index(key)
        actual = result["bounds"][index]
        assert np.isfinite(actual).all() and np.all(actual[:3] <= low) and np.all(actual[3:] >= high)
        assert result["shape_revisions"][index] == shape.revision
        wanted[key] = (low, high)
    registered = set(result["registered_pairs"])
    assert all(a.world == b.world and a.body != b.body for a, b in registered)
    assert set(result["candidates"]) <= registered
    true_pairs = {pair for pair in registered if np.all(wanted[pair[0]][0] <= wanted[pair[1]][1])
                  and np.all(wanted[pair[1]][0] <= wanted[pair[0]][1])}
    assert true_pairs <= set(result["candidates"])
    assert true_pairs, "the declared loaded finite pairs disappeared"
    for world, tick in zip(result["world_keys"], result["ticks"], strict=True):
        assert tick == row["state"][world].tick
    readonly(result["bounds"])
    return dict(material_bounds=wanted, true_pairs=tuple(sorted(true_pairs)))



def capture(batch, plan=None):
    if plan is not None:
        return contact_capture(batch, plan)
    return dict(state=batch.snapshot(), geometry=batch.tree_kinematics(),
                diagnostics=batch.tree_diagnostics(), energy=batch.summary(),
                batch_stats=batch.stats(), receipts={}, status={})


def expected_next(item, q, v):
    if item["kind"] == "floating_affine":
        return advance(item["data"], q, v, item["data"]["drive"])
    return fixed_truth(item, q, v)


def one_step(batch, plan, items, expected, rows, sources=None, model=None, *, acceptance=None, label="forward"):
    acceptance = dict.fromkeys(items, True) if acceptance is None else dict(acceptance)
    h = next(iter(items.values()))["data"]["h"]
    assert all(item["data"]["h"] == h for item in items.values())
    before = capture(batch, plan)
    row = dict(label=label, input={w: i["command"] for w, i in items.items()},
               entry=before["state"], entry_geometry=before["geometry"], entry_energy=before["energy"],
               entry_stats=before["batch_stats"], entry_receipts=before["receipts"],
               h_s=h, expected_acceptance=acceptance, audit={})
    if plan is not None:
        row["entry_broad"] = before["broad"]
    rows.append(row)  # Preserve accepted-but-audit-failed rows without rerunning.
    try:
        flags = batch.step(h) if plan is None else batch.step_contacts(plan, h)
        row["acceptance"] = flags
        row.update(capture(batch, plan))
        assert flags == acceptance
        for world, item in items.items():
            entry, state = before["state"][world], row["state"][world]
            assert state.world == world
            assert state.control == entry.control and state.history == entry.history == ()
            if not acceptance[world]:
                assert state == entry
                assert fingerprint(row["geometry"][world]) == fingerprint(before["geometry"][world])
                assert fingerprint(row["energy"][world]) == fingerprint(before["energy"][world])
                assert row["status"][world] == "SITE_CAPACITY_EXCEEDED"
                assert world not in row["receipts"]
                continue
            assert state.tick == entry.tick+1
            q, v, truth = expected_next(item, *expected[world])
            expected[world] = (q, v)
            row.setdefault("independent", {})[world] = dict(position=q, velocity=v, full=truth)
            if plan is None:
                audit = floating_audit(item, entry, state, row["geometry"][world],
                                       row["diagnostics"][world], row["energy"][world])
            else:
                receipt = row["receipts"][world]
                assert receipt.entry_state == entry
                audit = contact_audit(item, receipt, state, row["geometry"][world], row["diagnostics"][world],
                                      row["energy"][world], sources[world])
            close(state.position, q, atol=WINDOW_ATOL)
            close(state.velocity, v, atol=WINDOW_ATOL)
            row["audit"][world] = audit
        if plan is not None:
            flat = {key: value for source in sources.values() for key, value in source.items()}
            row["bounds_audit"] = bounds_audit(row, flat, model)
        readonly((row["geometry"], row["diagnostics"], row["receipts"]))
        row["checks_complete"] = True
        return row
    except Exception as error:
        row["failure"] = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
        raise


def replay_equal(actual, original):
    # Both complete rows remain archived. Only epoch and cumulative scheduling
    # counters differ; no state, body, diagnostic, material, or work field is removed.
    for field in ("input", "entry", "entry_geometry", "entry_energy", "acceptance",
                  "state", "geometry", "diagnostics", "energy", "receipts", "status"):
        assert fingerprint(actual[field]) == fingerprint(original[field]), field
    for field in ("entry_broad", "broad"):
        for key in BROAD_PHYSICAL:
            assert fingerprint(actual[field][key]) == fingerprint(original[field][key]), (field, key)
        assert actual[field]["topology_generation"] == original[field]["topology_generation"]
        assert actual[field]["state_epoch"] > original[field]["state_epoch"]
    actual["complete_replay_equal"] = True


def save_retained(section, batch, plan):
    if batch is not None:
        try:
            section["retained_on_failure"] = capture(batch, plan)
        except Exception as error:
            section["retained_read_error"] = dict(type=type(error).__name__, message=str(error))


def close_resources(section, batch, plan):
    try:
        try:
            if plan is not None:
                plan.close(); plan.close()
                section["closed_plan_stats"] = plan.stats()
                assert plan.stats()["device_owned_bytes"] == 0
                assert plan.stats()["contact_device_owned_bytes"] == 0
        finally:
            if batch is not None:
                batch.close(); batch.close()
                section["closed_batch_stats"] = batch.stats()
                assert batch.stats()["device_owned_bytes"] == 0
                assert batch.stats()["collision_plan_live_count"] == 0
    except Exception as error:
        section["status"] = "failed"
        section["cleanup_failure"] = dict(type=type(error).__name__, message=str(error),
                                           traceback=traceback.format_exc())
        raise

def run_w4(ph, mode, output, record):
    section = dict(status="running", world_order=ORDER, h_s=H, trace=[], replay=[],
                   capacity=[], alignments=[], checks={}, inputs={}, declarations={})
    record["w4"] = section
    batch = plan = None
    try:
        items = declarations(ph, output / "w4-inputs")
        section.update(inputs={w: i["data"] for w, i in items.items()},
                       declarations={w: i["tree"] for w, i in items.items()},
                       sources={w: i.get("source") for w, i in items.items()},
                       assets={w: i.get("asset") for w, i in items.items()},
                       controls={w: i["command"] for w, i in items.items()})
        expected = {w: (i["data"]["q"].copy(), i["data"]["v"].copy()) for w, i in items.items()}
        # Independent first literal anchors supplement full same-entry KKT.
        _, _, a05 = advance(items[91]["data"], *expected[91], items[91]["data"]["drive"])
        close(a05["acceleration"][[0, 6]], [2/7, -10/7], atol=2e-13, rtol=0)
        close(a05["row_multipliers"], [-2/7], atol=2e-13, rtol=0)
        _, _, old = fixed_truth(items[6], *expected[6])
        close(old["acceleration"], [4/7], atol=2e-13, rtol=0)
        close(old["row_multipliers"], [41/14, -25/7], atol=2e-13, rtol=0)
        section["literal_first_anchors"] = dict(a05=a05, fixed_a03=old)
        batch = ph.TyxWorldBatch([(w, i["tree"]) for w, i in items.items()], mode, cache_root=output / "w4-cache")
        batch.set_controls(section["controls"])
        model, bindings, sources, laws = contact_geometry(ph, batch, items)
        section.update(collision_model=model, bindings=bindings, material_sources=sources, laws=laws)
        plan = batch.prepare_contacts(model, bindings, laws, tree_dynamic_pairs=True)
        section["initial"] = capture(batch, plan)
        saved = saved_expected = None
        retained = retained_fingerprint = None
        for index in range(40):
            row = one_step(batch, plan, items, expected, section["trace"], sources, model)
            if index == 0:
                retained = {name: row[name] for name in ("state", "geometry", "diagnostics", "receipts")}
                retained_fingerprint = fingerprint(retained)
            if index == 18:
                saved = batch.snapshot(); saved_expected = copy.deepcopy(expected)
                section["restore_checkpoint"] = saved
        loaded = [bool(np.any(row["receipts"][503].law_wrench_fp64 != 0)) for row in section["trace"]]
        first_open = next((i for i, value in enumerate(loaded) if not value), None)
        assert loaded[0] and first_open is not None and any(loaded[first_open+1:]), loaded
        section["lifecycle"] = dict(loaded=loaded, first_open_index=first_open)
        assert saved is not None and saved_expected is not None
        section["before_restore"] = capture(batch, plan)
        epoch = batch.stats()["state_epoch"]
        batch.restore(saved)
        section["after_restore"] = capture(batch, plan)
        assert batch.snapshot() == saved and batch.stats()["state_epoch"] == epoch+1
        expected = saved_expected
        for index in range(19, 40):
            row = one_step(batch, plan, items, expected, section["replay"], sources, model, label="restore-replay")
            replay_equal(row, section["trace"][index])
        for bank, limited_world in ((0, 503), (1, 91)):
            if batch.stats()["current_buffer"] != bank:
                assert len(section["alignments"]) < 2
                one_step(batch, plan, items, expected, section["alignments"], sources, model, label="bank-alignment")
            assert batch.stats()["current_buffer"] == bank
            capacities = dict.fromkeys(items, 128); capacities[limited_world] = 0
            tx = dict(bank=bank, limited_world=limited_world, capacities=capacities, limited_rows=[], retry_rows=[])
            section["capacity"].append(tx)
            tx["before"] = capture(batch, plan)
            previous = fingerprint(plan.last_accepted)
            limited = batch.prepare_contacts(model, bindings, laws, tree_dynamic_pairs=True, site_capacity=capacities)
            try:
                flags = {w: w != limited_world for w in items}
                row = one_step(batch, limited, items, expected, tx["limited_rows"], sources, model,
                               acceptance=flags, label="actual-capacity-zero")
                assert row["entry_stats"]["current_buffer"] == bank
                tx["full_plan_receipts_after_rejection"] = dict(plan.last_accepted)
                assert fingerprint(plan.last_accepted) == previous
                assert batch.snapshot()[limited_world] == tx["before"]["state"][limited_world]
            finally:
                limited.close(); limited.close()
                tx["closed_plan_stats"] = limited.stats()
                assert limited.stats()["device_owned_bytes"] == 0
                assert limited.stats()["contact_device_owned_bytes"] == 0
            retry = one_step(batch, plan, items, expected, tx["retry_rows"], sources, model, label="same-controls-full-capacity-retry")
            assert retry["receipts"][limited_world].entry_state == row["entry"][limited_world]
            tx["checks_complete"] = True
        assert retained is not None and fingerprint(retained) == retained_fingerprint
        readonly(retained)
        section["retained_first_owned"] = retained
        section["final"] = capture(batch, plan)
        section["independent_final"] = expected
        for world in items:
            assert section["final"]["state"][world].tick == 40+len(section["alignments"])+4-int(world in (503, 91))
        assert batch.stats()["mode"] == mode and batch.stats()["submitted_kernels"] > 0
        if mode == "cuda":
            assert batch.stats()["device_allocations"] > 0
        section["checks"] = dict(complete_forward_40=True, complete_restore_21=True,
            lifecycle_loaded_open_reenter=True, both_capacity_banks=True, owned_initial_unchanged=True)
        section["status"] = "passed"
    except Exception as error:
        section["status"] = "failed"
        section["failure"] = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
        save_retained(section, batch, plan)
        raise
    finally:
        close_resources(section, batch, plan)


def run_a04(ph, mode, output, record):
    section = dict(status="running", world_order=(503,), trace=[], h_s=1/4096)
    record["a04_p1"] = section
    batch = None
    try:
        item = floating_item(ph, output / "a04-input", "A04", "p1")
        section.update(input=item["data"], source=item["source"], asset=item["asset"],
                       declaration=item["tree"], command=item["command"])
        assert len(item["data"]["parts"]) == 4 and np.any(item["data"]["v"][3:6] != 0)
        batch = ph.TyxWorldBatch([(503, item["tree"])], mode, cache_root=output / "a04-cache")
        batch.set_controls({503: item["command"]})
        expected = {503: (item["data"]["q"].copy(), item["data"]["v"].copy())}
        section["initial"] = capture(batch)
        for _ in range(4):
            one_step(batch, None, {503: item}, expected, section["trace"])
        section["final"] = capture(batch)
        assert section["final"]["state"][503].tick == 4
        assert batch.stats()["submitted_kernels"] > 0 and batch.stats()["mode"] == mode
        section["independent_final"] = expected
        section["status"] = "passed"
    except Exception as error:
        section["status"] = "failed"
        section["failure"] = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
        save_retained(section, batch, None)
        raise
    finally:
        close_resources(section, batch, None)

def archive_identity(installed, expected):
    wheel = expected.resolve(strict=True)
    if wheel.suffix.lower() != ".whl":
        raise ValueError("--expected-wheel must name the actual wheel archive")
    with wheel.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    direct_text = installed.read_text("direct_url.json")
    if direct_text is None:
        raise AssertionError("local wheel installation must retain direct_url.json")
    direct = json.loads(direct_text)
    origin = urlsplit(direct["url"])
    assert origin.scheme == "file" and origin.netloc in ("", "localhost")
    assert Path(url2pathname(origin.path)).resolve(strict=True) == wheel
    archive_info = direct["archive_info"]
    recorded = archive_info.get("hashes", {}).get("sha256")
    if recorded is None and archive_info.get("hash"):
        algorithm, _, recorded = archive_info["hash"].partition("=")
        assert algorithm == "sha256"
    if recorded is not None:
        assert recorded.lower() == digest
    # Installer hashes are optional under direct-URL metadata. Exact local
    # archive binding plus every installed member's bytes is mandatory here.
    site = Path(installed.locate_file("")).resolve(strict=True)
    product_members, metadata_members = [], []
    with ZipFile(wheel) as archive:
        members = {name for name in archive.namelist() if name.startswith("phymeo/") and not name.endswith("/")}
        actual = {path.relative_to(site).as_posix() for path in (site/"phymeo").rglob("*") if path.is_file()}
        assert members and actual == members, "installed product membership differs from actual wheel"
        for name in sorted(members):
            path = (site/name).resolve(strict=True)
            assert path.is_relative_to(site/"phymeo")
            content = path.read_bytes()
            assert content == archive.read(name), "installed content differs: "+name
            product_members.append(dict(name=name, bytes=len(content), sha256=hashlib.sha256(content).hexdigest()))
        for name in archive.namelist():
            if ".dist-info/" in name and name.rsplit("/", 1)[-1] in ("METADATA", "WHEEL", "top_level.txt"):
                content = (site/name).read_bytes()
                assert content == archive.read(name), name
                metadata_members.append(dict(name=name, bytes=len(content), sha256=hashlib.sha256(content).hexdigest()))
    return dict(path=str(wheel), bytes=wheel.stat().st_size, sha256=digest, direct_url=direct,
                installed_from=direct["url"], version=installed.version, installer_recorded_sha256=recorded,
                installed_product_files_verified=len(product_members), members=product_members, metadata_members=metadata_members)


def module_provenance(site, package, wheel):
    rows = []
    with ZipFile(wheel) as archive:
        for name, module in sorted(tuple(sys.modules.items())):
            if name != "phymeo" and not name.startswith("phymeo."):
                continue
            file = getattr(module, "__file__", None)
            assert file is not None, "no installed origin for "+name
            path = Path(file).resolve(strict=True)
            assert path.is_relative_to(package), "non-wheel module: "+name
            spec = getattr(module, "__spec__", None)
            assert spec is not None and spec.origin is not None
            assert Path(spec.origin).resolve(strict=True) == path
            search_paths = tuple(str(Path(p).resolve(strict=True)) for p in getattr(module, "__path__", ()))
            assert all(Path(p).is_relative_to(package) for p in search_paths)
            member = path.relative_to(site).as_posix()
            content = path.read_bytes()
            assert archive.read(member) == content, "module differs from actual wheel: "+member
            rows.append(dict(module=name, path=str(path), origin=spec.origin, package_paths=search_paths,
                             wheel_member=member, bytes=len(content), sha256=hashlib.sha256(content).hexdigest()))
    assert rows
    return rows


def outside_checkout(path):
    return not any((parent/".git").exists() for parent in (path, *path.parents))



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("interpret", "cuda"), required=True)
    for name in ("output", "wheel-site", "expected-package-root", "expected-wheel", "forbidden-checkout"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    assert __debug__ and sys.flags.isolated and sys.flags.dont_write_bytecode, "use python -I -B"
    for path in (args.output, args.wheel_site, args.expected_package_root, args.expected_wheel, args.forbidden_checkout):
        if not path.is_absolute():
            raise ValueError("all declared paths must be absolute")
    site, package = args.wheel_site.resolve(strict=True), args.expected_package_root.resolve(strict=True)
    script, output = Path(__file__).resolve(strict=True), args.output.resolve()
    forbidden = args.forbidden_checkout.resolve(strict=True)
    assert package == (site/"phymeo").resolve(strict=True)
    assert Path.cwd().resolve() == script.parent
    assert all(not path.is_relative_to(forbidden) for path in (script, site, package, output))
    assert all(outside_checkout(path) for path in (script.parent, site, package, output))
    assert "phymeo" not in sys.modules
    # The output must not create an extra member under the installed package.
    assert not output.is_relative_to(package)
    output.mkdir(parents=True, exist_ok=False)
    content = script.read_bytes()
    record = dict(status="running", mode=args.mode, errors=[],
        protocol="c11-installed-floating-affine-v1", world_order=ORDER,
        cwd=str(Path.cwd().resolve()), interpreter=sys.executable,
        script=dict(path=str(script), bytes=len(content), sha256=hashlib.sha256(content).hexdigest(),
                    source_utf8=content.decode("utf-8")),
        expected_package_root=str(package), forbidden_checkout=str(forbidden),
        fixed_membership=dict(w4=dict(worlds=ORDER, forward_steps=40, restore_tick=19, replay_steps=21,
                                     capacity_banks=(0, 1), capacity_worlds=(503, 91), maximum_alignment_steps=2),
                              a04_p1=dict(worlds=(503,), steps=4)),
        gates=dict(h_w4_s=H, h_a04_s=1/4096, all_delay_steps=0,
                   state_atol=ATOL, state_rtol=RTOL, window_atol=WINDOW_ATOL,
                   full_mass_reaction_atol=2e-5, equation_residual=2e-5, control_residual=5e-5,
                   material_record_construction_atol_m=1e-10, material_work_atol_j=1e-12,
                   material_work_rtol=1e-10, payload_bytes_cap=MAX_BYTES, compressed_bytes_cap=MAX_GZIP_BYTES),
        units=dict(configuration="root authored-origin m, XYZW; independent joints SI",
                   velocity="root COM world m/s, world rad/s; independent joint SI/s",
                   force="N", torque="N*m about actual COM", energy="J", time="s"),
        limits="Fixed C11 subset, full floating-affine response with HOST finite mu=0 contact sampling and TYX mechanics. Not a speed test, native Newton comparison, all-device narrow phase, CCD, persistent friction, or a claim that the whole C11 matrix passed.")
    installed = None
    sys.path.insert(0, str(site))
    try:
        import phymeo as ph

        assert Path(ph.__file__).resolve(strict=True).parent == package
        installed = distribution("phymeo")
        assert Path(installed.locate_file("")).resolve(strict=True) == site
        record["wheel_before"] = archive_identity(installed, args.expected_wheel)
        record["modules_before"] = module_provenance(site, package, args.expected_wheel)
        record["versions"] = dict(phymeo=installed.version, tyx=distribution("tyx-compute").version,
                                  numpy=np.__version__, python=sys.version)
        # One fixed invocation per window. A failure is preserved; no retry or
        # replacement fixture is used to make a window appear complete.
        run_w4(ph, args.mode, output, record)
        run_a04(ph, args.mode, output, record)
        record["status"] = "passed"
    except Exception as error:
        record["status"] = "failed"
        record["errors"].append(dict(stage="import_or_fixed_windows", type=type(error).__name__,
                                     message=str(error), traceback=traceback.format_exc()))
    try:
        assert script.read_bytes() == content, "consumer source changed during this run"
        if installed is not None:
            record["wheel_after"] = archive_identity(installed, args.expected_wheel)
            assert record["wheel_after"] == record["wheel_before"]
        if "phymeo" in sys.modules:
            record["modules_after"] = module_provenance(site, package, args.expected_wheel)
    except Exception as error:
        record["status"] = "failed"
        record["errors"].append(dict(stage="final_provenance", type=type(error).__name__,
                                     message=str(error), traceback=traceback.format_exc()))
    path = output / "resident-floating-affine-wheel.trace.json.gz"
    trace = dict(path=str(path), complete=False, prefix_only=record["status"] != "passed")
    try:
        payload = json.dumps(encode(record), allow_nan=False, separators=(",", ":")).encode()
        trace.update(uncompressed_bytes=len(payload), uncompressed_sha256=hashlib.sha256(payload).hexdigest())
        if len(payload) > MAX_BYTES:
            raise ValueError("resource_stop: complete payload exceeds frozen 256 MiB; no truncated success")
        temporary = path.with_name(path.name+".tmp")
        with gzip.open(temporary, "wb") as stream:
            stream.write(payload)
        size = temporary.stat().st_size
        trace["temporary_path"] = str(temporary)
        trace["compressed_bytes"] = size
        if size > MAX_GZIP_BYTES:
            raise ValueError("resource_stop: complete gzip exceeds frozen 64 MiB; temporary evidence retained")
        temporary.replace(path)
        trace["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        # complete means the full available record (possibly a failed prefix)
        # was archived, never that the physical protocol necessarily passed.
        trace["complete"] = True
    except Exception as error:
        record["status"] = "failed"
        record["errors"].append(dict(stage="resource_stop" if str(error).startswith("resource_stop:") else "archive_io",
                                     type=type(error).__name__, message=str(error), traceback=traceback.format_exc()))
    summary = dict(status=record["status"], raw_exit=int(record["status"] != "passed"), mode=args.mode,
                   protocol=record["protocol"], trace=trace, wheel=record.get("wheel_before"),
                   verified_module_count=len(record.get("modules_after", ())), windows={},
                   errors=record["errors"], gates=record["gates"], fixed_membership=record["fixed_membership"],
                   script=record["script"] | {"source_utf8": "stored in complete trace"}, limits=record["limits"])
    for label in ("w4", "a04_p1"):
        section = record.get(label, {})
        rows = list(section.get("trace", ()))+list(section.get("replay", ()))+list(section.get("alignments", ()))
        for tx in section.get("capacity", ()):
            rows.extend(tx.get("limited_rows", ())); rows.extend(tx.get("retry_rows", ()))
        summary["windows"][label] = dict(status=section.get("status", "not_started"),
            completed_forward=sum(bool(row.get("checks_complete")) for row in section.get("trace", ())),
            completed_replay=sum(bool(row.get("complete_replay_equal")) for row in section.get("replay", ())),
            completed_capacity_banks=sum(bool(tx.get("checks_complete")) for tx in section.get("capacity", ())),
            real_alignment_steps=sum(bool(row.get("checks_complete")) for row in section.get("alignments", ())),
            attempted_batch_steps=len(rows), returned_acceptance_steps=sum("acceptance" in row for row in rows),
            accepted_world_steps=sum(sum(bool(flag) for flag in row.get("acceptance", {}).values()) for row in rows),
            rejected_world_steps=sum(sum(not flag for flag in row.get("acceptance", {}).values()) for row in rows),
            checks=section.get("checks", {}), closed_batch_stats=section.get("closed_batch_stats"),
            closed_plan_stats=section.get("closed_plan_stats"), failure=section.get("failure"),
            cleanup_failure=section.get("cleanup_failure"))
    # A final write error remains an IO failure/nonzero exit, not a simulated
    # physics rejection. No old evidence is removed on any path.
    (output/"summary.json").write_text(json.dumps(encode(summary), allow_nan=False, indent=2), encoding="utf-8")
    print(json.dumps(encode(summary), allow_nan=False))
    return summary["raw_exit"]


if __name__ == "__main__":
    raise SystemExit(main())