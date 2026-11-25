"""One photographic room for eight independent manipulation worlds, in metres.

Rigid stage transforms place independent worlds on the set at their physical scale.
"""
from dataclasses import dataclass, replace
from hashlib import sha256
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from _manipulation_photo_studio import _Surface, studio_lighting
from _panda_scene import look_at, pose_matrix, write_mesh
from galaxea_r1.finish_details import _store
from phymeo_rendering.assets import Material, Mesh, TexturedMaterial
from phymeo_rendering.cameras import Camera
from phymeo_rendering.mesh_data import _local_path, load_mesh
from phymeo_rendering.scene import FrameKey, Instance, NodePose, Pose, SceneFrame, VisualScene, compose_pose


_LEFT, _RIGHT = -.30, .75
_FRONT, _BACK = -.34, .34
_FLOOR = -.70
_ROOM = 'overview-room'
_PEDESTAL = {'photo-studio-plinth', 'photo-studio-graphite', 'photo-studio-satin-alloy'}


def overview_lighting():
    """The same workshop HDR and two real rectangular emitters at room scale."""
    return replace(studio_lighting(), center_m=(0., 0., .05), scale_m=2.65)


def _mesh(directory, name, p, n, t, *, uv=None):
    with TemporaryDirectory(dir=directory) as temp:
        source = write_mesh(temp, name, p, n, t, texcoords=uv)
        path, _ = _store(directory, name, _local_path(source.uri).read_bytes(), 'gltf')
    return Mesh(name, path.as_uri())


def _pedestal(mesh, model, directory):
    # The styled graphite and alloy meshes also hold the four edges of the
    # full 2 m support. Drop only triangles on those four vertical planes.
    source = load_mesh(mesh)
    p, n, tris = map(np.asarray, (source.positions, source.normals, source.triangles))
    sx, sy = model.geom_size[model.geom('support').id, :2]
    vertices = p[tris]
    fascia = np.zeros(len(tris), bool)
    for axis, edge in ((0, sx), (1, sy)):
        for sign in (-1, 1):
            fascia |= (np.abs(vertices[:, :, axis]-sign*edge) < 1e-6).all(axis=1)
    if np.count_nonzero(fascia) != 8:
        raise ValueError(f'expected four original table edges in {mesh.id}')
    kept, indices = np.unique(tris[~fascia].ravel(), return_inverse=True)
    return _mesh(directory, mesh.id, p[kept], n[kept], indices.reshape(-1,3))


class _Assets:
    def __init__(self):
        self.meshes, self.materials, self.images = {}, {}, {}
        self.instances = []

    def add(self, mesh, material, node, local=Pose()):
        # Individual world asset directories have different URIs even when the
        # underlying glTF or PNG is identical. Fold by bytes, including images.
        digest = sha256(_local_path(mesh.uri).read_bytes()).hexdigest()
        mesh_asset = self.meshes.setdefault(digest, replace(mesh, id='mesh-'+digest[:24]))
        m = replace(material, id='finish')
        if isinstance(m, TexturedMaterial):
            def image(source):
                return None if source is None else self.images.setdefault(source.sha256, source)
            rough = m.roughness_map
            m = replace(m, base_color_map=image(m.base_color_map), normal_map=image(m.normal_map),
                        roughness_map=replace(rough, image=image(rough.image)) if rough else None)
        finish = self.materials.setdefault(m, replace(m, id=f'finish-{len(self.materials)}'))
        ident = len(self.instances)+1
        self.instances.append(Instance(ident, mesh_asset.id, finish.id, node, local))
        return ident


def _shared_tables(directory):
    corners = ((_LEFT,_FRONT,0.), (_RIGHT,_FRONT,0.), (_RIGHT,_BACK,0.), (_LEFT,_BACK,0.))
    top = _mesh(directory, 'compact-top', corners, ((0.,0.,1.),)*4, ((0,1,2),(0,2,3)),
                uv=.5+.5*np.asarray(corners)[:,:2])
    trim, frame = _Surface(), _Surface()
    # All upper faces of the underframe are below the original z=0 contact plane.
    trim.block((_LEFT,_FRONT,-.031), (_RIGHT,_BACK,-.0004))
    frame.block((_LEFT+.032,_FRONT+.038,-.092), (_RIGHT-.032,_BACK-.038,-.031))
    for x, y in product((_LEFT+.052,_RIGHT-.052), (_FRONT+.058,_BACK-.058)):
        frame.block((x-.022,y-.022,_FLOOR+.001), (x+.022,y+.022,-.092))
    frame.block((_LEFT+.052,_BACK-.074,-.54), (_RIGHT-.052,_BACK-.042,-.513))
    return top, trim.mesh(directory,'compact-edge'), frame.mesh(directory,'bench-underframe')


def _room(directory, assets, keys, placements):
    floor, plaster, storage, handles = (_Surface() for _ in range(4))
    floor.quad(((-8.,-7.,_FLOOR),(8.,-7.,_FLOOR),(8.,5.,_FLOOR),(-8.,5.,_FLOOR)), ((0.,0.,1.),)*4)
    plaster.block((-6.,2.49,_FLOOR), (6.,2.55,3.4))
    plaster.block((-4.3,-4.,_FLOOR), (-4.24,2.49,3.4))
    # A single row of real service cabinets behind the two islands.
    for col in range(6):
        left = -2.52 + .84*col
        storage.block((left,2.28,_FLOOR+.10), (left+.822,2.48,.12))
        handles.block((left+.32,2.253,-.13), (left+.51,2.28,-.111))
    for name, surface, color, metal, rough in (
        ('floor',floor,(.31,.34,.34),.0,.77), ('walls',plaster,(.67,.68,.64),.0,.86),
        ('cupboards',storage,(.43,.48,.45),.03,.62), ('cupboard-pulls',handles,(.18,.14,.09),.59,.40),
    ):
        assets.add(surface.mesh(directory,'room-'+name), Material(name,color,metal,rough), _ROOM)

    # Small etched bay identifiers carry the actual world IDs. All plates and
    # all digit strokes share two static draw instances in the room.
    plates, typeface = _Surface(), _Surface()
    segments = ((0,2,1,2),(1,1,1,2),(1,0,1,1),(0,0,1,0),(0,0,0,1),(0,1,0,2),(0,1,1,1))
    digits = ('abcdef','bc','abdeg','abcdg','bcfg','acdfg','acdefg','abc','abcdefg','abcdfg')
    for key in keys:
        corners = [compose_pose(placements[key],Pose((x,y,0.))).position_m
                   for x,y in product((_LEFT,_RIGHT),(_FRONT,_BACK))]
        left, front = min(p[0] for p in corners)+.026, min(p[1] for p in corners)+.018
        plates.block((left,front,.0004),(left+.146,front+.055,.0018))
        label = str(key)
        x = left + (.146-(len(label)*.023-.005))*.5
        for digit in label:
            for s in digits[int(digit)]:
                x0,y0,x1,y1 = segments[ord(s)-ord('a')]
                a,b = x+.018*x0, front+.009+.018*y0
                c,d = x+.018*x1, front+.009+.018*y1
                typeface.quad(((a-.0016,b-.0016,.0021),(c+.0016,b-.0016,.0021),
                               (c+.0016,d+.0016,.0021),(a-.0016,d+.0016,.0021)), ((0.,0.,1.),)*4)
            x += .023
    assets.add(plates.mesh(directory,'bay-plaques'), Material('plaques',(.086,.107,.108),.33,.50), _ROOM)
    assets.add(typeface.mesh(directory,'bay-numbers'), Material('stencil',(.80,.76,.58),.08,.56), _ROOM)


@dataclass
class EightStudio:
    scene: VisualScene
    keys: tuple
    offsets: dict
    instance_ids: dict
    source_scenes: dict
    moving_nodes: dict
    placements: dict

    def frame(self, local_frames, frame_key=None, time_s=None):
        if set(local_frames) != set(self.keys):
            raise ValueError('supply one original SceneFrame for each of the eight worlds')
        frames = [local_frames[key] for key in self.keys]
        for key, f in zip(self.keys, frames):
            if f.key.scene_key != self.source_scenes[key] or f.key.world_id != str(key):
                raise ValueError(f'foreign source frame for overview world {key}')
        first = frames[0]
        if frame_key is None:
            if any((f.key.epoch,f.key.revision,f.key.tick,f.key.phase) !=
                   (first.key.epoch,first.key.revision,first.key.tick,first.key.phase) for f in frames):
                raise ValueError('independent source clocks require an explicit overview FrameKey')
            frame_key = replace(first.key, scene_key=self.scene.key, world_id='eight-studio')
        if not isinstance(frame_key, FrameKey) or frame_key.scene_key != self.scene.key:
            raise ValueError('overview frame key must name the combined scene')
        if time_s is None:
            if any(f.time_s != first.time_s for f in frames):
                raise ValueError('independent local times require an explicit presentation time_s')
            time_s = first.time_s
        nodes = [NodePose(_ROOM, Pose())]
        for key, f in zip(self.keys, frames):
            placement = self.placements[key]
            nodes.append(NodePose(f'bay-{key}/fixed',placement))
            local = {n.node_id:n.world_pose for n in f.nodes}
            for name in self.moving_nodes[key]:
                pose = local[name]
                nodes.append(NodePose(f'bay-{key}/{name}',compose_pose(placement,pose)))
        return SceneFrame(frame_key, time_s, tuple(nodes))

    def camera(self, frame_key, width=1920, height=1080):
        if frame_key.scene_key != self.scene.key:
            raise ValueError('overview camera must use the combined frame key')
        pose = look_at((.57,-7.,5.2), (0.,0.,-.11), (0.,0.,1.))
        # Fit both rows and all front feet to the requested aspect ratio. The
        # more distant camera keeps scale similar across the two rows.
        bounds = [compose_pose(stage, Pose(xyz)).position_m for stage in self.placements.values() for xyz in
                  product((_LEFT,_RIGHT),(_FRONT,_BACK),(_FLOOR,.30))]
        transform = pose_matrix(pose)
        local = (np.asarray(bounds)-transform[:3,3]) @ transform[:3,:3]
        slope = np.max(np.abs(local[:,:2]/(-local[:,2:3])),axis=0)
        focal = float(min(.466*width/slope[0], .46*height/slope[1]))
        return Camera('eight-studio',frame_key,pose,width,height,focal,focal,width/2,height/2,.02,20.)


def build_eight_studio(scenes, node_bodies, models, directory, *, end_on=()):
    """Assemble eight styled P4 scenes in insertion order, front row first."""
    keys = tuple(scenes)
    if len(keys)!=8 or set(keys)!=set(node_bodies) or set(keys)!=set(models):
        raise ValueError('need eight styled scenes with matching native models and node bindings')
    if any(type(key) is not int or key<0 for key in keys):
        raise ValueError('expected native nonnegative integer world keys')
    directory = Path(directory).resolve()
    directory.mkdir(parents=True,exist_ok=True)
    top, trim, frame = _shared_tables(directory)
    assets = _Assets()
    offsets, ids, moving, placements = {}, {}, {}, {}
    material_trim = Material('aluminium',(.42,.46,.45),.58,.42)
    material_frame = Material('underframe',(.092,.145,.153),.27,.42)
    for index, key in enumerate(keys):
        scene, model, static = scenes[key], models[key], f'bay-{key}/fixed'
        meshes = {m.id:m for m in scene.meshes}
        finishes = {m.id:m for m in scene.materials}
        local_names = {model.geom(i).name for i in range(model.ngeom)}
        offsets[key] = ((index%4-1.5)*1.18-(_LEFT+_RIGHT)*.5, (index//4-.5)*1.19, 0.)
        if key in end_on:
            # Point the jaw opening toward the overview camera so both fingers
            # and the round workpiece between them are visible during the lift.
            center = (_LEFT+_RIGHT)*.5
            x,y,z = offsets[key]
            offsets[key] = (x+center,y+center,z)
            placements[key] = Pose(offsets[key], (0.,0.,-2**-.5,2**-.5))
        else:
            placements[key] = Pose(offsets[key])
        ids[key], moving[key] = {}, []
        included = set()
        for item in scene.instances:
            name = item.mesh_id
            if name in local_names or name in _PEDESTAL:
                included.add(name)
            elif not name.startswith(('studio-target-', 'studio-tool-')):
                continue
            mesh = meshes[name]
            if name == 'support':
                mesh = top
            elif name in ('photo-studio-graphite','photo-studio-satin-alloy'):
                mesh = _pedestal(mesh, model, directory)
            is_fixed = node_bodies[key][item.node_id] == 0
            node = static if is_fixed else f'bay-{key}/{item.node_id}'
            if not is_fixed and item.node_id not in moving[key]:
                moving[key].append(item.node_id)
            ids[key][item.id] = assets.add(mesh, finishes[item.material_id], node, item.local_pose)
        if included != local_names | _PEDESTAL:
            raise ValueError(f'world {key} is missing authored manipulation or pedestal geometry')
        assets.add(trim, material_trim, static)
        assets.add(frame, material_frame, static)
    _room(directory, assets, keys, placements)
    # Include source order, semantic layout version, original scene identities
    # and complete mesh content fingerprints in the combined registration key.
    manifest = repr(([(key,scenes[key].key,placements[key]) for key in keys],tuple(assets.meshes)))
    name = 'eight-studio-v1:'+sha256(manifest.encode()).hexdigest()[:24]
    scene = VisualScene(name, tuple(assets.meshes.values()), tuple(assets.materials.values()), tuple(assets.instances))
    return EightStudio(scene, keys, offsets, ids, {key:scenes[key].key for key in keys}, moving, placements)
