"""Batched, state-only reaching for the existing dual Panda and free pin.

MuJoCo provides the actual compiled screw axes, anchors and site transforms at
construction. After reset, all measurements use the accepted device qpos. An
independent CPU audit calls native mj_kinematics on the saved states.
"""
import hashlib
import math

import numpy as np


FEATURE_COUNT = 21
DISTANCE_M = .020
ORIENTATION_RAD = math.radians(15)
BOARD_DISPLACEMENT_M = .010
BOARD_ROTATION_RAD = math.radians(5)
PART_DRIFT_M = .010
DWELL_S = .100
VELOCITY_SCALE = [2.] * 14 + [.20] * 2
SITE_NAMES = ("assembly_hole", "right_grasp_center", "part_grasp_center")


def _pose(d, site):
    result = np.eye(4)
    result[:3, :3] = d.site_xmat[site].reshape(3, 3)
    result[:3, 3] = d.site_xpos[site]
    return result


def _compiled_geometry(model):
    import mujoco as mj

    if (model.nq, model.nv, model.nu) != (23, 22, 16):
        raise ValueError("expected the two seven-axis Pandas, two driven fingers and a free pin")
    d = mj.MjData(model)
    d.qpos[:] = model.qpos0
    mj.mj_kinematics(model, d)
    sites = [model.site(n).id for n in SITE_NAMES]
    generators = np.zeros((2, 7, 4, 4))
    home = np.stack([_pose(d, s) for s in sites[:2]])
    for arm, (side, site) in enumerate(zip(("left", "right"), sites[:2], strict=True)):
        # Follow the compiled parent tree to this actual TCP. Refuse an
        # unmodelled hinge/slide in either site's ancestry.
        ancestry = []
        body = int(model.site_bodyid[site])
        while body != 0:
            start = int(model.body_jntadr[body])
            ancestry.extend(reversed(range(start, start + int(model.body_jntnum[body]))))
            body = int(model.body_parentid[body])
        joints = [model.joint(f"{side}_panda_joint{i}").id for i in range(1, 8)]
        if list(reversed(ancestry)) != joints:
            raise ValueError(f"unexpected compiled ancestor chain for {SITE_NAMES[arm]}")
        for i, joint in enumerate(joints):
            if (model.jnt_type[joint] != mj.mjtJoint.mjJNT_HINGE
                    or model.jnt_qposadr[joint] != 7*arm+i):
                raise ValueError("Panda hinge coordinate mapping differs")
            x, y, z = d.xaxis[joint]
            skew = np.array(((0.,-z,y), (z,0.,-x), (-y,x,0.)))
            generators[arm, i, :3, :3] = skew
            generators[arm, i, :3, 3] = -skew @ d.xanchor[joint]
    part = model.body("assembly_part").id
    jid = int(model.body_jntadr[part])
    if (model.body_parentid[part] != 0 or model.body_jntnum[part] != 1
            or model.jnt_type[jid] != mj.mjtJoint.mjJNT_FREE or model.jnt_qposadr[jid] != 16
            or model.site_bodyid[sites[2]] != part):
        raise ValueError("expected the measured grasp site on the world-rooted free workpiece")
    body = np.eye(4)
    body[:3, :3] = d.xmat[part].reshape(3, 3)
    body[:3, 3] = d.xpos[part]
    part_to_site = np.linalg.solve(body, _pose(d, sites[2]))
    return generators, home, np.array(model.qpos0[:14]).reshape(2,7), part_to_site


def _mm(a, b):
    # Explicit small-matrix FP32 products keep reduced-precision GEMM settings
    # in a training job from changing millimetre-scale kinematics.
    return (a.unsqueeze(-1) * b.unsqueeze(-3)).sum(-2)


def _mv(a, b):
    return (a*b.unsqueeze(-2)).sum(-1)


def _angle(torch, r):
    s = torch.stack((r[...,2,1]-r[...,1,2], r[...,0,2]-r[...,2,0],
                     r[...,1,0]-r[...,0,1]), dim=-1).square().sum(-1).sqrt()
    c = r[...,0,0]+r[...,1,1]+r[...,2,2]-1.
    return torch.atan2(s, c)


class ReachTask:
    def __init__(self, torch, model, device):
        self.torch = torch
        self.device = torch.device(device)
        g, home, qref, local = _compiled_geometry(model)
        digest = hashlib.sha256()
        for arr in (g,home,qref,local):
            digest.update(np.ascontiguousarray(arr,dtype="<f8").tobytes())
        def constant(x):
            return torch.as_tensor(x,dtype=torch.float32,device=self.device)
        self.g = constant(g)
        self.g2 = constant(g @ g)
        self.eye = constant(np.eye(4))
        self.home = constant(home)
        self.qref = constant(qref)
        self.part_site = constant(local)
        self.velocity_scale = constant(VELOCITY_SCALE)
        self.reference = None
        self.contract = dict(schema="dual-panda-free-pin-reach-v1", site_names=list(SITE_NAMES),
            fk_geometry_sha256=digest.hexdigest(), feature_count=FEATURE_COUNT,
            features=["hand-frame pin translation / .15m", "hand-frame pin rotation columns 0,1",
                      "initial-board-frame board translation / .03m", "board rotation columns 0,1",
                      "world pin displacement from reset / .03m"],
            reward="1-(distance/.15)^2-.25*(orientation/pi)^2-.5*(board_displacement/.025)^2"
                   "-.15*(board_rotation/(pi/6))^2-.4*(pin_drift/.025)^2"
                   "-.002*mean(normalized_joint_speed^2)-.001*mean(normalized_effort^2)",
            joint_velocity_scale=VELOCITY_SCALE.copy(), orientation_units="radians",
            success=dict(max_distance_m=DISTANCE_M,max_hand_pin_angle_rad=ORIENTATION_RAD,
                         max_board_displacement_m=BOARD_DISPLACEMENT_M,max_board_rotation_rad=BOARD_ROTATION_RAD,
                         max_pin_displacement_m=PART_DRIFT_M,min_continuous_dwell_s=DWELL_S),
            scope="approach the free pin while the other wrist holds the board at its reset pose")

    def poses(self, q):
        """Return world (position, rotation) in board, hand, pin order; each is batched over worlds."""
        t = self.torch
        if q.ndim != 2 or q.shape[1] != 23 or q.device != self.g.device or q.dtype != self.g.dtype:
            raise ValueError("expected accepted float32 [world,23] qpos on the observation device")
        angle = (q[:,:14].reshape(-1,2,7)-self.qref).unsqueeze(-1).unsqueeze(-1)
        transforms = self.eye + angle.sin()*self.g + (1-angle.cos())*self.g2
        chain = t.cat((transforms,self.home.expand(len(q),-1,-1,-1).unsqueeze(2)),dim=2)
        # Seven hinge transforms and the fixed terminal. Left/right, all worlds
        # and every independent pair in a tree level share each operation.
        for _ in range(3):
            chain = _mm(chain[:,:,::2],chain[:,:,1::2])
        world = chain[:,:,0]
        quat = q[:,19:23]
        w,x,y,z = (quat/quat.square().sum(-1,keepdim=True).sqrt()).unbind(-1)
        r = t.stack((1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y),
                     2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x),
                     2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)),dim=-1).reshape(-1,3,3)
        pin_p = q[:,16:19]+_mv(r,self.part_site[:3,3])
        pin_r = _mm(r,self.part_site[:3,:3])
        return (world[:,0,:3,3],world[:,0,:3,:3]), (world[:,1,:3,3],world[:,1,:3,:3]), (pin_p,pin_r)

    def reset(self, q):
        board, _, pin = self.poses(q)
        self.reference = tuple(a.detach().clone() for a in (*board,pin[0]))

    def measure(self, q):
        if self.reference is None or self.reference[0].shape[0] != q.shape[0]:
            raise ValueError("reset must bind initial poses for the same batch")
        t = self.torch
        (bp,br),(hp,hr),(pp,pr) = self.poses(q)
        bp0,br0,pp0 = self.reference
        right_pos = _mv(hr.transpose(-1,-2),pp-hp)
        right_rot = _mm(hr.transpose(-1,-2),pr)
        board_pos = _mv(br0.transpose(-1,-2),bp-bp0)
        board_rot = _mm(br0.transpose(-1,-2),br)
        part_delta = pp-pp0
        features = t.cat((right_pos/.15,right_rot[:,:,0],right_rot[:,:,1],board_pos/.03,
                          board_rot[:,:,0],board_rot[:,:,1],part_delta/.03),dim=-1)
        dist = right_pos.square().sum(-1).sqrt()
        angle = _angle(t,right_rot)
        board_dist = board_pos.square().sum(-1).sqrt()
        board_angle = _angle(t,board_rot)
        drift = part_delta.square().sum(-1).sqrt()
        good = ((dist <= DISTANCE_M) & (angle <= ORIENTATION_RAD) & (board_dist <= BOARD_DISPLACEMENT_M)
                & (board_angle <= BOARD_ROTATION_RAD) & (drift <= PART_DRIFT_M))
        return dict(features=features,distance_m=dist,orientation_rad=angle,board_displacement_m=board_dist,
                    board_tilt_rad=board_angle,part_drift_m=drift,qualified=good)

    def reward(self,q,v,controls,effort_limit,*,measurement=None):
        m = self.measure(q) if measurement is None else measurement
        return (1.-(m["distance_m"]/.15).square()-.25*(m["orientation_rad"]/math.pi).square()
                -.5*(m["board_displacement_m"]/.025).square()-.15*(m["board_tilt_rad"]/(math.pi/6)).square()
                -.4*(m["part_drift_m"]/.025).square()-.002*(v[:,:16]/self.velocity_scale).square().mean(-1)
                -.001*(controls/effort_limit).square().mean(dim=(0,2)))


def audit_numpy(model, reset_q, q, v, controls, effort_limit):
    """Independent native site observations and reward; q=[W,23] or [T,W,23]."""
    import mujoco as mj

    q,v,u,limit = [np.asarray(a,dtype=float) for a in (q,v,controls,effort_limit)]
    single = q.ndim==2
    if single:
        q,v,u = q[None],v[None],u[None]
    if q.ndim != 3 or q.shape[-1] != 23 or v.shape != (*q.shape[:2],22):
        raise ValueError("expected [time,world,native coordinate] for accepted q and v")
    steps,worlds = q.shape[:2]
    if (u.ndim != 4 or u.shape[0] != steps or u.shape[2:] != (worlds,16)
            or not u.shape[1] or limit.shape != (16,) or np.any(limit<=0)):
        raise ValueError("controls must be [time,substep,world,16], with 16 positive effort limits")
    initial = np.broadcast_to(np.asarray(reset_q,dtype=float),(worlds,23))
    if not all(np.isfinite(a).all() for a in (initial,q,v,u,limit)):
        raise ValueError("nonfinite reach audit input")
    data = mj.MjData(model)
    ids = [model.site(n).id for n in SITE_NAMES]
    def read(positions):
        data.qpos[:] = positions
        mj.mj_kinematics(model,data)
        return np.stack([_pose(data,i) for i in ids])
    ref = np.stack([read(row) for row in initial])
    poses = np.stack([read(row) for row in q.reshape(-1,23)]).reshape(steps,worlds,3,4,4)
    board,hand,pin = (poses[:,:,i] for i in range(3))
    # Use inverse homogeneous transforms, independent of the Torch products.
    hr = np.linalg.inv(hand) @ pin
    br = np.linalg.inv(ref[:,0]) @ board
    dp = pin[:,:,:3,3]-ref[:,2,:3,3]
    def angle(rot):
        anti = rot[..., :3,:3]-rot[..., :3,:3].swapaxes(-1,-2)
        sine = np.linalg.norm(anti,axis=(-2,-1))/math.sqrt(8.)
        cosine = (np.trace(rot[..., :3,:3],axis1=-2,axis2=-1)-1.)/2.
        return np.arctan2(sine,cosine)
    distance = np.linalg.norm(hr[:,:,:3,3],axis=-1)
    orientation = angle(hr)
    translation = np.linalg.norm(br[:,:,:3,3],axis=-1)
    rotation = angle(br)
    drift = np.linalg.norm(dp,axis=-1)
    features = np.concatenate((hr[:,:,:3,3]/.15,hr[:,:,:3,0],hr[:,:,:3,1],br[:,:,:3,3]/.03,
                               br[:,:,:3,0],br[:,:,:3,1],dp/.03),axis=-1)
    reward = (1.-(distance/.15)**2-.25*(orientation/np.pi)**2-.5*(translation/.025)**2
              -.15*(rotation/(np.pi/6))**2-.4*(drift/.025)**2
              -.002*((v[:,:,:16]/VELOCITY_SCALE)**2).mean(axis=-1)-.001*((u/limit)**2).mean(axis=(1,3)))
    qualified = ((distance<=DISTANCE_M) & (orientation<=ORIENTATION_RAD)
                 & (translation<=BOARD_DISPLACEMENT_M) & (rotation<=BOARD_ROTATION_RAD) & (drift<=PART_DRIFT_M))
    result = dict(features=features,distance_m=distance,orientation_rad=orientation,
                  board_displacement_m=translation,board_tilt_rad=rotation,part_drift_m=drift,
                  qualified=qualified,reward=reward)
    return {key:a[0] for key,a in result.items()} if single else result
