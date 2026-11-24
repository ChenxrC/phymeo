"""Registered six-axis FK and geometry-calibrated parallel-jaw R1 pick/place."""
from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class WorkshopLayout:
    pick_xy: tuple = (.46, .43)
    place_xy: tuple = (.49, .12)
    support_z: float = .88
    payload_size: tuple = (.078, .056, .068)
    lift_m: float = .145


LAYOUT = WorkshopLayout()
ARM = tuple(f"left_arm_joint{i}" for i in range(1, 7))
FINGERS = ("left_gripper_finger_joint1", "left_gripper_finger_joint2")
OPEN = .048
STOCK_BEVEL_M = .001
TOOL_ROTATION = np.array([[0.,0.,1.], [0.,1.,0.], [-1.,0.,0.]])
PHASES = ((.18,"approach"), (.29,"lower_to_pick"), (.37,"close"), (.47,"lift"),
          (.66,"transfer"), (.76,"lower_to_place"), (.84,"open"), (1.,"withdraw"))


@dataclass(frozen=True)
class MotionSample:
    base: object
    joints: dict
    payload: object
    phase: str
    attached: bool


def _ease(t, start, end):
    u = min(1., max(0., (t-start)/(end-start)))
    return u**3 * (10. + u*(-15. + 6.*u))


def _path(t, layout):
    pick, place = np.array(layout.pick_xy), np.array(layout.place_xy)
    xy = pick + (1.-_ease(t,0.,.18))*np.array([-.072,.038])
    xy += _ease(t,.47,.66)*(place-pick) + _ease(t,.86,1.)*np.array([-.07,.04])
    z = layout.support_z + layout.payload_size[2]/2
    z += .165*(1.-_ease(t,.18,.29)) + layout.lift_m*(_ease(t,.37,.47)-_ease(t,.66,.76))
    z += .175*_ease(t,.84,.99)
    closed = _ease(t,.29,.37) - _ease(t,.76,.84)
    phase = next(name for end,name in PHASES if t < end or end == 1.)
    return np.r_[xy,z], closed, phase


def _calibrate(model, layout):
    """Support of the actual registered fingertip triangles in the grasp band."""
    from _panda_scene import pose_matrix
    from phymeo_rendering.mesh_data import load_mesh
    from phymeo_rendering.scene import compose_pose
    meshes = {m.id:m for m in model.scene.meshes}
    surfaces = []
    for j, direction in zip(FINGERS, (1.,-1.)):
        joint = model.joints[j]
        if (joint.parent != "left_gripper_link" or joint.kind != "prismatic" or
                not np.allclose(joint.axis, [0.,direction,0.], atol=1e-8)):
            raise ValueError("unexpected R1 parallel jaw registration")
        pieces = []
        for instance in model.scene.instances:
            if instance.node_id == joint.child:
                m = load_mesh(meshes[instance.mesh_id])
                T = pose_matrix(compose_pose(joint.origin, instance.local_pose))
                points = np.asarray(m.positions) @ T[:3,:3].T + T[:3,3]
                pieces.append(points[np.asarray(m.triangles)])
        if not pieces:
            raise ValueError(f"missing R1 jaw geometry for {joint.child}")
        surfaces.append(np.concatenate(pieces))
    # Place the upper 16 mm of the stock between the distal pads. Beyond
    # this band, the short finger ends leave the fixture unobstructed.
    start = min(float(s[:,:,0].max()) for s in surfaces) - .016
    center_x = start + layout.payload_size[2]/2
    rows = []
    for j, direction, triangles in zip(FINGERS, (1.,-1.), surfaces):
        # Clip the registered CAD triangles at the stock's top surface. Both
        # the original vertices and all clipped-edge intersections can attain
        # the extremum, including where no CAD vertex lies on the slice.
        a = triangles.reshape(-1,3)
        b = np.roll(triangles, -1, axis=1).reshape(-1,3)
        above = a[:,0] >= start
        candidates = [a[above]]
        for plane in (start, start+STOCK_BEVEL_M):
            crossings = (a[:,0] >= plane) != (b[:,0] >= plane)
            a_c, b_c = a[crossings], b[crossings]
            blend = (plane - a_c[:,0]) / (b_c[:,0] - a_c[:,0])
            candidates.append(a_c + blend[:,None]*(b_c-a_c))
        clipped = np.vstack(candidates)
        # All contact points lie on the straight long sides, inside the 4 mm
        # corner chamfer. Above that straight face the top bevel reduces stock
        # half-width linearly by up to 1 mm (workshop_set._payload).
        if (not len(clipped) or np.max(np.abs(clipped[:,2])) >= layout.payload_size[0]/2-.004 or
                clipped[:,0].max() >= center_x + layout.payload_size[2]/2):
            raise ValueError("registered fingertip grasp band exceeds the stock projection")
        bevel = np.maximum(0., start+STOCK_BEVEL_M-clipped[:,0])
        support = direction * clipped[:,1] + bevel
        index = int(np.argmin(support))
        inner = float(support[index])
        contact = clipped[index]
        q = layout.payload_size[1]/2 - inner
        joint = model.joints[j]
        if not (joint.lower <= q < OPEN-.004 and OPEN <= joint.upper):
            raise ValueError(f"part width cannot be gripped by {j}: closed={q}, open={OPEN}")
        rows.append(dict(joint=j, sign=direction, inner_y_at_zero_m=float(contact[1]),
                         stock_half_width_at_contact_m=layout.payload_size[1]/2-float(bevel[index]),
                         contact_at_zero_tool_m=contact.tolist(), closed_command_m=q,
                         open_clearance_m=OPEN-q))
    return center_x, rows


def _axis_matrix(axis):
    x,y,z = axis
    return np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])


def _rotation_error(current):
    difference = TOOL_ROTATION @ current.T
    cosine = np.clip((np.trace(difference)-1.)*.5, -1.,1.)
    theta = math.acos(cosine)
    skew = np.array([difference[2,1]-difference[1,2], difference[0,2]-difference[2,0],
                     difference[1,0]-difference[0,1]])
    if theta < 1e-7:
        return .5 * skew
    if math.pi-theta < 1e-5:
        # A pi rotation has vanishing skew; use its invariant axis.
        _, vectors = np.linalg.eigh((difference + difference.T)/2)
        axis = vectors[:,-1]
        return theta * (axis if np.dot(axis,skew) >= 0. else -axis)
    return theta / (2*math.sin(theta)) * skew


class _Arm:
    def __init__(self, model, posture, base, center_x):
        from _panda_scene import pose_matrix
        base_nodes = {n.node_id:n.world_pose for n in model.pose_nodes(posture, base)}
        self.root = pose_matrix(base_nodes[model.joints[ARM[0]].parent])
        self.joints = []
        self.lower, self.upper = [], []
        for name in ARM:
            j = model.joints[name]
            axis = np.array(j.axis)
            skew = _axis_matrix(axis)
            self.joints.append((pose_matrix(j.origin), axis, skew, skew@skew))
            self.lower.append(j.lower)
            self.upper.append(j.upper)
        self.lower, self.upper = np.array(self.lower), np.array(self.upper)
        self.offset = pose_matrix(model.joints["left_gripper_joint"].origin)
        self.offset[:3,3] += self.offset[:3,0]*center_x

    def evaluate(self, angles, target):
        T = self.root.copy()
        origins, axes = [], []
        for (local, axis, skew, square), angle in zip(self.joints, angles):
            T = T @ local
            origins.append(T[:3,3].copy())
            axes.append(T[:3,:3] @ axis)
            R = np.eye(3) + math.sin(angle)*skew + (1.-math.cos(angle))*square
            T[:3,:3] = T[:3,:3] @ R
        end = T @ self.offset
        delta = np.r_[target - end[:3,3], .26*_rotation_error(end[:3,:3])]
        J = np.vstack((np.cross(axes, end[:3,3]-np.array(origins)).T, .26*np.array(axes).T))
        return delta, J

    def solve(self, target, seed):
        # Damped, box-constrained Newton steps with a monotone backtrack. The
        # branch is seeded only once; all subsequent poses start at frame n-1.
        q = np.clip(seed, self.lower, self.upper)
        damping = .001
        for _ in range(85):
            error, jacobian = self.evaluate(q, target)
            score = np.linalg.norm(error)
            if score < 1e-7:
                return q, score
            step = np.linalg.solve(jacobian.T @ jacobian + damping**2*np.eye(6), jacobian.T @ error)
            scale = min(1., .28/max(float(np.max(np.abs(step))), 1e-12))
            for factor in (1., .5, .25, .125):
                trial = np.clip(q+factor*scale*step, self.lower, self.upper)
                if np.linalg.norm(self.evaluate(trial,target)[0]) < score - 1e-13:
                    q = trial
                    damping = max(.0001, damping*.7)
                    break
            else:
                damping *= 4.
                if damping > 1.:
                    break
        return q, float(np.linalg.norm(self.evaluate(q,target)[0]))


def build_motion(model, frames, fps=30):
    """Return a MotionSample per exposure and measurements from independent URDF FK."""
    from _panda_scene import pose_matrix
    from phymeo_rendering.scene import Pose, compose_pose
    if frames < 2 or fps <= 0:
        raise ValueError("workshop motion needs two or more frames and a positive frame rate")
    layout = LAYOUT
    center_x, contacts = _calibrate(model, layout)
    base = Pose((0.,0.,model.floor_offset_m))
    posture = dict(torso_joint1=-.24, torso_joint2=.47, torso_joint3=.23, torso_joint4=0.)
    for i, value in enumerate((-.98,1.82,-1.45,-.08,.46,-.10), start=1):
        posture[f"right_arm_joint{i}"] = value
    for i in (1,2):
        posture[f"right_gripper_finger_joint{i}"] = .012
    arm = _Arm(model, posture, base, center_x)
    first_target = _path(0.,layout)[0]
    guesses = ((1.,2.2,-1.6,0.,.75,0.), (.8,2.6,-1.85,1.,1.05,-.65),
               (1.45,2.4,-1.45,-1.,.9,1.), (.2,2.25,-1.9,1.5,1.25,-1.15),
               (1.75,2.1,-1.7,-1.4,-.7,1.4))
    for guess in guesses:
        q, score = arm.solve(first_target, np.array(guess))
        if score < 1e-6:
            break
    else:
        raise ValueError(f"initial R1 workshop target is unreachable: weighted error {score}")
    # Relative payload orientation is the inverse of the target wrist. The
    # transported stock is composed from the measured FK pose every frame.
    relative = Pose((center_x,0.,0.),(0.,-math.sqrt(.5),0.,math.sqrt(.5)))
    height = layout.support_z + layout.payload_size[2]/2
    pick = Pose((*layout.pick_xy,height))
    place = Pose((*layout.place_xy,height))
    samples, errors, angular, offsets, gaps, heights = [], [], [], [], [], []
    arm_commands = []
    for index in range(frames):
        t = index/(frames-1)
        target, closure, phase = _path(t,layout)
        q, score = arm.solve(target, q)
        if score > 5e-6:
            raise ValueError(f"R1 IK frame {index} {phase}, residual {score:.6g} at {target.tolist()}")
        joints = {**posture, **dict(zip(ARM, q.tolist()))}
        for row in contacts:
            joints[row["joint"]] = OPEN + closure*(row["closed_command_m"]-OPEN)
        nodes = {n.node_id:n.world_pose for n in model.pose_nodes(joints,base)}
        measured = compose_pose(nodes["left_gripper_link"], relative)
        actual = pose_matrix(measured)
        errors.append(float(np.linalg.norm(actual[:3,3]-target)))
        angular.append(float(math.acos(np.clip((np.trace(actual[:3,:3])-1)/2,-1.,1.))))
        attached = .37 <= t < .76
        payload = measured if attached else (pick if t < .37 else place)
        heights.append(payload.position_m[2]-layout.payload_size[2]/2-layout.support_z)
        if attached:
            object_frame = pose_matrix(payload)
            for row in contacts:
                # Transform the original zero-command support point with the
                # actual finger node, then measure against the object's side.
                j = model.joints[row["joint"]]
                zero = pose_matrix(j.origin)
                local = zero[:3,:3].T @ (np.asarray(row["contact_at_zero_tool_m"])-zero[:3,3])
                finger = pose_matrix(nodes[j.child])
                world = finger[:3,:3]@local + finger[:3,3]
                point = object_frame[:3,:3].T @ (world-object_frame[:3,3])
                local_bevel = max(0.,point[2] - (layout.payload_size[2]/2-STOCK_BEVEL_M))
                gaps.append(abs(row["sign"]*point[1] - (layout.payload_size[1]/2-local_bevel)))
                offsets.append(point.tolist())
        arm_commands.append(q.copy())
        samples.append(MotionSample(base,joints,payload,phase,attached))
    if max(errors) > .0001 or max(angular) > .001 or max(gaps,default=0.) > .0001:
        raise ValueError("independent R1 link FK failed the workshop contact tolerance")
    commands = np.asarray(arm_commands)
    proof = {}
    for label, at in (("open_above_pick", .18), ("closed_at_pick", .37), ("raised", .48),
                      ("in_transit", .56), ("seated", .76), ("open_at_place", .84), ("withdrawn", 1.)):
        index = round((frames-1)*at)
        sample = samples[index]
        proof[label] = dict(frame=index, phase=sample.phase, attached=sample.attached,
                            payload_center_m=sample.payload.position_m,
                            finger_commands_m=[sample.joints[key] for key in FINGERS])
    evidence = dict(method="joint-limited URDF IK with commanded jaw contact; kinematic transport",
        frame_count=frames, fps=fps, tool_x_at_payload_center_m=center_x, finger_contacts=contacts,
        exact_payload_on_rest=dict(pick=pick.position_m, place=place.position_m),
        phase_end_fractions=PHASES, proof_frames=proof, max_fk_position_error_m=max(errors),
        max_fk_orientation_error_rad=max(angular), max_held_contact_error_m=max(gaps,default=None),
        held_contact_local_xyz_range_m=[np.min(offsets,axis=0).tolist(),np.max(offsets,axis=0).tolist()] if offsets else None,
        max_undershoot_support_m=max(0.,-min(heights)), max_payload_lift_m=max(heights),
        max_arm_step_rad=float(np.max(np.abs(np.diff(commands,axis=0)))),
        joint_extrema_rad={k:[float(commands[:,i].min()),float(commands[:,i].max())] for i,k in enumerate(ARM)})
    return samples, evidence
