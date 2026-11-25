"""Render-only task ink, prismatic guides, and end-effector mounting hardware."""
from dataclasses import replace
from hashlib import sha256
from math import cos, pi, sin
from pathlib import Path

import numpy as np

from _manipulation_photo_studio import _Surface
from _panda_scene import pose_matrix
from phymeo_rendering.assets import Material
from phymeo_rendering.scene import Instance, Pose


def target_ids(scene):
    """Segmentation IDs of the two inks framing the actual target footprint."""
    return tuple(item.id for item in scene.instances if item.mesh_id.startswith('studio-target-'))


def tool_visual_ids(scene):
    """Segmentation IDs of decorative tool mounts and slide hardware only."""
    return tuple(item.id for item in scene.instances if item.mesh_id.startswith('studio-tool-'))


def _ink_surfaces(shapes):
    # Partition the plane by every rectangle edge at each offset. Classifying
    # cells against the *union* removes the shared edge of stem and crossbar.
    # Both strokes are outside the true silhouette, so their inner edge is
    # the exact boundary even when the workpiece is sitting at the goal.
    if len(shapes) != 2 or any(s.kind != 'box' for s in shapes):
        raise ValueError('PushT requires the two native rectangular components')
    boxes = np.array([(s.position[0]-s.size[0], s.position[1]-s.size[1],
                       s.position[0]+s.size[0], s.position[1]+s.size[1]) for s in shapes])
    layers = [boxes + w*np.array((-1.,-1.,1.,1.)) for w in (0.,.007,.012)]
    # Decimal CAD edges at the same physical coordinate may differ by one ULP.
    grid = np.asarray(layers).reshape(-1,4)
    xs = np.unique(grid[:,(0,2)].round(12))
    ys = np.unique(grid[:,(1,3)].round(12))
    surfaces = (_Surface(), _Surface())
    for l,r in zip(xs,xs[1:]):
        for b,t in zip(ys,ys[1:]):
            x,y = (l+r)/2,(b+t)/2
            for level,rects in enumerate(layers):
                if any(x0<x<x1 and y0<y<y1 for x0,y0,x1,y1 in rects):
                    if level:
                        surfaces[level-1].quad(((l,b,0.),(r,b,0.),(r,t,0.),(l,t,0.)), ((0.,0.,1.),)*4)
                    break
    return surfaces


def _tube(surface, a, b, radius, *, sides=16):
    """Closed machined cylinder between two points in the given body's frame."""
    a,b = np.asarray(a),np.asarray(b)
    axis = b-a
    axis = axis/np.linalg.norm(axis)
    side = np.eye(3)[np.argmin(np.abs(axis))]
    u = np.cross(axis,side)
    u /= np.linalg.norm(u)
    v = np.cross(axis,u)
    circle = [cos(2*pi*i/sides)*u + sin(2*pi*i/sides)*v for i in range(sides)]
    for i,n in enumerate(circle):
        m = circle[(i+1)%sides]
        surface.quad((a+radius*n,a+radius*m,b+radius*m,b+radius*n), (n,m,m,n))
    for end, normal, sign in ((a,-axis,-1),(b,axis,1)):
        first = len(surface.points)
        surface.points.append(end)
        surface.points.extend(end+radius*n for n in circle)
        surface.normals.extend([normal]*(1+sides))
        for i in range(sides):
            j = (i+1)%sides
            uidx, vidx = (i,j) if sign>0 else (j,i)
            surface.faces.append((first,first+1+uidx,first+1+vidx))


def _slide_hardware(model, authored, add):
    # Parent rail and child carriage are registered separately. At the neutral
    # body pose their track axes coincide; translation along the native joint
    # keeps them coaxial throughout its original limits.
    for i,kind in enumerate(authored.robot_kinds):
        if kind != 'P':
            continue
        j = model.joint(authored.robot_joint_names[i]).id
        child = int(model.jnt_bodyid[j])
        parent = int(model.body_parentid[child])
        parent_node, child_node = model.body(parent).name, model.body(child).name
        axis = np.asarray(model.jnt_axis[j],dtype=float)
        axis = axis/np.linalg.norm(axis)
        side = np.array((0.,0.,1.) if abs(axis[2])<.8 else (0.,-1.,0.))
        side = side-axis*np.dot(axis,side)
        side /= np.linalg.norm(side)
        pivot = np.asarray(model.jnt_pos[j])
        track = pivot + .022*side
        rest = float(model.qpos0[model.jnt_qposadr[j]])
        lo,hi = np.asarray(model.jnt_range[j])-rest
        if not model.jnt_limited[j] or not lo<=0<=hi or hi-lo<.018:
            raise ValueError('studio guide expects the original finite prismatic travel')
        q = model.body_quat[child]
        child_at_rest = Pose(tuple(model.body_pos[child]), (q[1],q[2],q[3],q[0]))
        fixed, moving = _Surface(),_Surface()
        _tube(fixed,track+axis*(lo-.007),track+axis*(hi+.007),.0075)
        _tube(fixed,pivot,track,.010)
        _tube(moving,track-axis*.019,track+axis*.019,.0135)
        _tube(moving,pivot,track,.0105)
        add(f'studio-tool-guide-{i}',fixed,'steel',parent_node,child_at_rest)
        add(f'studio-tool-carriage-{i}',moving,'housing',child_node)


def _gripper(model, add):
    # Work out the common rail elevation in the wrist frame. A tilted pad's
    # +Z end can sit laterally away from its finger origin and its four end
    # corners no longer have a common height in that frame.
    def native_pose(position, wxyz):
        w,x,y,z = wxyz
        return Pose(tuple(position), (x,y,z,w))

    wrist_body = int(model.geom_bodyid[model.geom('wrist').id])
    records = []
    ends = []
    for name in ('left-pad','right-pad'):
        geom = model.geom(name).id
        body = int(model.geom_bodyid[geom])
        if model.body_parentid[body] != wrist_body:
            raise ValueError('finger slide must be attached to the tool body')
        pad_pose = native_pose(model.geom_pos[geom], model.geom_quat[geom])
        finger_pose = native_pose(model.body_pos[body], model.body_quat[body])
        wrist_from_pad = pose_matrix(finger_pose) @ pose_matrix(pad_pose)
        if wrist_from_pad[2,2] <= 0.:
            raise ValueError('the native pad +Z end must face the wrist guide')
        sx,sy,sz = model.geom_size[geom]
        face = np.array(((-sx,-sy,sz),(sx,-sy,sz),(sx,sy,sz),(-sx,sy,sz)))
        ends.extend(face @ wrist_from_pad[:3,:3].T + wrist_from_pad[:3,3])
        records.append((name,model.body(body).name,pad_pose,wrist_from_pad,face))
    ends = np.asarray(ends)
    z = float(ends[:,2].max())
    half = float(np.abs(ends[:,1]).max()) + .003
    housing, bars = _Surface(),_Surface()
    housing.block((-.019,-half,z+.013),(.019,half,z+.030))
    for x in (-.012,.012):
        _tube(bars,(x,-half,z+.010),(x,half,z+.010),.0045)
    tool = model.body(wrist_body).name
    add('studio-tool-gripper-palm',housing,'housing',tool)
    add('studio-tool-finger-rails',bars,'steel',tool)
    for name,node,pad_pose,wrist_from_pad,face in records:
        shoe = _Surface()
        # Start directly on the real end face. Extend each corner along the
        # pad's own +Z axis up to a plane through the rails. This keeps the
        # whole shoe inside the pad's XY footprint and above its gripping faces.
        bottom = face.copy()
        bottom[:,:2] *= .94
        top = bottom.copy()
        normal = wrist_from_pad[2,:3]
        top[:,2] = (z+.012-wrist_from_pad[2,3]-top[:,:2]@normal[:2])/normal[2]
        shoe.quad(top,(normal,)*4)
        for i in range(4):
            j = (i+1)%4
            side = np.cross(bottom[j]-bottom[i],top[i]-bottom[i])
            side /= np.linalg.norm(side)
            shoe.quad((bottom[i],bottom[j],top[j],top[i]),(side,)*4)
        add('studio-tool-shoe-'+name,shoe,'steel',node,pad_pose)


def _push_shank(model, add):
    geom = model.geom('pusher').id
    size = model.geom_size[geom]
    center = model.geom_pos[geom]
    # The upper pole of the native Z-aligned capsule is the end of the gap.
    # The shank terminates inside that pole; the original working end and sides
    # remain the native pusher geometry.
    upper = float(center[2]+size[0]+size[1])
    node = model.body(int(model.geom_bodyid[geom])).name
    stem, collet = _Surface(),_Surface()
    x,y = map(float,center[:2])
    _tube(stem,(x,y,upper-.005),(x,y,.003),.010)
    _tube(collet,(x,y,-.022),(x,y,.008),.016)
    add('studio-tool-pusher-shank',stem,'steel',node)
    add('studio-tool-pusher-collet',collet,'housing',node)


def add_task_visuals(scene, model, authored, task, directory):
    """Augment a styled scene, using only its existing dynamic and static nodes.

    PushT ``task`` supplies target_position_m (the body center) and
    target_yaw_rad; the mark is printed on the physical support at that XY.
    """
    directory = Path(directory).resolve()
    directory.mkdir(exist_ok=True,parents=True)
    if target_ids(scene) or tool_visual_ids(scene):
        raise ValueError('task visuals already present')
    nodes = {item.node_id for item in scene.instances}
    finishes = {
        'steel': Material('studio-tool-steel',(.47,.51,.52),.70,.34),
        'housing': Material('studio-tool-housing',(.078,.134,.144),.22,.42),
        'ink': Material('studio-target-amber',(.93,.47,.045),.0,.60),
        'edge': Material('studio-target-charcoal',(.036,.052,.060),.0,.66),
    }
    meshes, materials, additions = [], {}, []
    first_id = max(i.id for i in scene.instances)+1
    def add(name, surface, finish, node, pose=Pose()):
        if node not in nodes:
            raise ValueError(f'{name} requires pre-existing bound node {node}')
        mesh = surface.mesh(directory,name)
        meshes.append(mesh)
        m = finishes[finish]
        materials[m.id] = m
        additions.append(Instance(first_id+len(additions),mesh.id,m.id,node,pose))

    _slide_hardware(model,authored,add)
    if any(g.kind=='pusht' for g in authored.goals):
        _push_shank(model,add)
        center = np.asarray(task['target_position_m'],dtype=float)
        yaw = float(task['target_yaw_rad'])
        if center.shape!=(3,) or not np.isfinite(center).all() or not np.isfinite(yaw):
            raise ValueError('PushT requires a finite target body center and yaw')
        ink, edge = _ink_surfaces(authored.object_shapes)
        support = next(i for i in scene.instances if i.mesh_id==authored.support_geom_name)
        pose = Pose((float(center[0]),float(center[1]),float(authored.support_top_m)+.00035),
                    (0.,0.,sin(yaw/2),cos(yaw/2)))
        add('studio-target-footprint',ink,'ink',support.node_id,pose)
        add('studio-target-outline',edge,'edge',support.node_id,pose)
    else:
        _gripper(model,add)
    # The mesh filenames contain a digest of their bytes; poses distinguish
    # sites with identical footprints at different targets within a model.
    declaration = repr([(Path(m.uri).name,i.local_pose,i.node_id) for m,i in zip(meshes,additions)])
    token = sha256(declaration.encode()).hexdigest()[:16]
    return replace(scene,key=scene.key+':task-visual-v1-'+token,meshes=(*scene.meshes,*meshes),
                   materials=(*scene.materials,*materials.values()),instances=(*scene.instances,*additions))
