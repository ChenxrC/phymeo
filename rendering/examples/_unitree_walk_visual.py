"""Render the official G1 12-DOF walking model from its compiled visual CAD."""
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
from math import radians, tan
from numbers import Integral
from pathlib import Path

import numpy as np

from _manipulation_photo_studio import _Surface
from _panda_scene import look_at, pose_matrix
from phymeo_rendering._prepare_cache import publish_bytes
from phymeo_rendering._prepare_geometry import _mesh_bytes
from phymeo_rendering.assets import Material, Mesh, TextureMap, TexturedMaterial
from phymeo_rendering.cameras import Camera
from phymeo_rendering.scene import FrameKey, Instance, NodePose, Pose, SceneFrame, VisualScene, compose_pose


def _pose(position, wxyz):
    w, x, y, z = map(float, wxyz)
    return Pose(tuple(map(float,position)), (x,y,z,w))


def _mesh(directory, name, points, normals, indices, uv=None):
    raw = _mesh_bytes(np.asarray(points),np.asarray(normals),None if uv is None else np.asarray(uv),indices)
    digest = sha256(raw).hexdigest()
    path = directory/(name+'-'+digest+'.gltf')
    publish_bytes(path,raw)
    return Mesh(name,path.resolve().as_uri()), digest


def _cad_mesh(model, mesh_id, directory):
    """MuJoCo stores distinct *local* vertex and normal indices at each face corner."""
    name = model.mesh(mesh_id).name
    va, vn = int(model.mesh_vertadr[mesh_id]), int(model.mesh_vertnum[mesh_id])
    na, nn = int(model.mesh_normaladr[mesh_id]), int(model.mesh_normalnum[mesh_id])
    fa, fn = int(model.mesh_faceadr[mesh_id]), int(model.mesh_facenum[mesh_id])
    points = np.asarray(model.mesh_vert[va:va+vn], dtype=np.float32)
    normals = np.asarray(model.mesh_normal[na:na+nn], dtype=np.float64)
    faces = np.asarray(model.mesh_face[fa:fa+fn], dtype=np.intp)
    corner_normals = np.asarray(model.mesh_facenormal[fa:fa+fn], dtype=np.intp)
    if (not len(faces) or faces.min()<0 or faces.max()>=vn or corner_normals.min()<0
            or corner_normals.max()>=nn or not np.isfinite(points).all() or not np.isfinite(normals).all()):
        raise ValueError(f'{name}: invalid compiled visual mesh indices or attributes')
    p = points[faces].astype(np.float64)
    cross = np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0])
    area = np.linalg.norm(cross,axis=1)
    valid = area>0
    faces, corner_normals = faces[valid], corner_normals[valid]
    if not len(faces):
        raise ValueError(f'{name}: all compiled mesh triangles are degenerate')
    lengths = np.linalg.norm(normals,axis=1)
    missing = lengths[corner_normals] == 0
    normals = normals/np.where(lengths>0,lengths,1.)[:,None]
    if missing.any():
        # Only corners without a compiled shading direction use their triangle normal.
        face_ids, corners = np.nonzero(missing)
        used, replacement = np.unique(face_ids,return_inverse=True)
        corner_normals[face_ids,corners] = nn+replacement
        normals = np.concatenate((normals, (cross[valid]/area[valid,None])[used]))

    meshes, chunks = [], []
    def write(first, last):
        pairs = np.column_stack((faces[first:last].ravel(), corner_normals[first:last].ravel()))
        unique, inverse = np.unique(pairs,axis=0,return_inverse=True)
        if len(unique)>65536:
            middle = (first+last)//2
            write(first,middle)
            write(middle,last)
            return
        chunk = f'g1-cad-{mesh_id:02d}-{len(meshes):02d}'
        mesh, digest = _mesh(directory,chunk,points[unique[:,0]],normals[unique[:,1]],inverse.reshape(-1,3))
        meshes.append(mesh)
        chunks.append(dict(mesh=chunk, sha256=digest, vertices=len(unique), triangles=last-first))

    for start in range(0,len(faces),65536):
        write(start,min(len(faces),start+65536))
    return meshes, dict(name=name, mesh_id=mesh_id, source_triangles=fn, triangles=len(faces),
                        zero_area_removed=int((~valid).sum()), missing_normal_corners=int(missing.sum()), chunks=chunks)


def _floor_map(directory):
    """A two-metre repeat with one-metre joints and quarter-metre lane markings."""
    from PIL import Image
    n = 512
    y,x = np.mgrid[:n,:n]
    x, y = (x+.5)*2/n, (y+.5)*2/n
    def dist(v,period):
        return np.abs((v+period*.5)%period-period*.5)
    def ink(d, half):
        return np.clip((half+.002-d)/.004,0,1)
    seam = ink(np.minimum(dist(x,1.),dist(y,1.)),.006)
    tile = np.array([155,163,161],float)+(1-2*((np.floor(x)+np.floor(y))%2))[...,None]*2
    tile -= seam[...,None]*19
    # A painted edge and measured ticks run along the walking direction.
    stripe = ink(dist(y-.76,2.),.010)
    half_tick = np.where(dist(x,1.)<.03,.075,np.where(dist(x,.5)<.03,.052,.032))
    ticks = ink(dist(x,.25),.010)*ink(np.abs(y-(.76+half_tick)),half_tick)
    paint = np.maximum(stripe,ticks)[...,None]
    tile = tile*(1-paint)+np.array([107,128,124])*paint
    stream = BytesIO()
    Image.fromarray(np.rint(tile).astype(np.uint8)).save(stream,format='PNG')
    raw = stream.getvalue()
    digest = sha256(raw).hexdigest()
    path = directory/('g1-metre-floor-'+digest+'.png')
    publish_bytes(path,raw)
    return TextureMap(path.resolve().as_uri(),digest)


def _workshop(model, directory, first_id, node):
    import mujoco as mj
    floor_ids = np.flatnonzero((model.geom_bodyid==0) & (model.geom_type==mj.mjtGeom.mjGEOM_PLANE))
    if len(floor_ids)!=1:
        raise ValueError('expected the official scene with one static ground plane')
    floor = int(floor_ids[0])
    pose = _pose(model.geom_pos[floor],model.geom_quat[floor])
    surface = _Surface()
    corners = ((-28.,-18.,0.),(40.,-18.,0.),(40.,18.,0.),(-28.,18.,0.))
    surface.quad(corners,((0.,0.,1.),)*4)
    mesh, digest = _mesh(directory,'walk-floor',surface.points,surface.normals,surface.faces,
                         np.asarray(corners)[:,:2]/2)
    material = TexturedMaterial('walk-floor',(1.,1.,1.),0.,.77,_floor_map(directory))
    meshes, materials = [mesh], [material]
    instances = [Instance(first_id,mesh.id,material.id,node,pose)]
    evidence = [dict(mesh=mesh.id, sha256=digest, vertices=len(surface.points), triangles=len(surface.faces))]
    wall, metal, panels = _Surface(), _Surface(), _Surface()
    wall.block((-28.,5.20,0.),(40.,5.32,4.0))
    wall.block((-28.,-18.,0.),(-27.88,5.32,4.0))
    metal.block((-28.,5.15,.045),(40.,5.2,.17))
    metal.block((-28.,5.17,2.43),(40.,5.2,2.47))
    for x in range(-27,40,4):
        panels.block((x+.16,5.175,2.50),(x+3.84,5.20,3.44))
        metal.block((x+1.985,5.15,2.49),(x+2.015,5.175,3.45))
    for name, surf, color, metallic, rough in (
            ('plaster',wall,(.56,.59,.57),0.,.84), ('anodized',metal,(.17,.20,.20),.35,.51),
            ('clerestory',panels,(.32,.43,.46),.13,.39)):
        mesh,digest = _mesh(directory,'walk-'+name,surf.points,surf.normals,surf.faces)
        mat = Material('walk-'+name,color,metallic,rough)
        meshes.append(mesh)
        materials.append(mat)
        instances.append(Instance(first_id+len(instances),mesh.id,mat.id,node,pose))
        evidence.append(dict(mesh=mesh.id, sha256=digest, vertices=len(surf.points), triangles=len(surf.faces)))
    return meshes, materials, instances, evidence


@dataclass(frozen=True)
class UnitreeWalkVisual:
    scene: VisualScene
    report: dict
    body_nodes: tuple
    pelvis_node: str
    geometry: tuple
    nbody: int

    def frame_from_bodies(self, positions, quaternions_wxyz, *, tick, time_s, epoch=0, world_id=None):
        xyz, quat = np.asarray(positions), np.asarray(quaternions_wxyz)
        if xyz.shape!=(self.nbody,3) or quat.shape!=(self.nbody,4):
            raise ValueError('G1 frame requires complete model-indexed xpos and wxyz xquat arrays')
        if any(isinstance(k,bool) or not isinstance(k,Integral) for k in (tick,epoch)):
            raise TypeError('G1 tick and epoch must be integer counters')
        key = FrameKey(self.scene.key, 'unitree-g1' if world_id is None else world_id,
                       int(epoch),int(tick),int(tick),'accepted_endpoint')
        nodes = tuple(NodePose(node,_pose(xyz[body],quat[body])) for body,node in self.body_nodes)
        return SceneFrame(key,float(time_s),nodes)

    def frame(self, data, *, tick, epoch=0, world_id=None):
        return self.frame_from_bodies(data.xpos,data.xquat,tick=tick,time_s=data.time,epoch=epoch,world_id=world_id)

    def camera(self, frame, *, width=1440, height=810, view='follow', id=None):
        if frame.key.scene_key != self.scene.key:
            raise ValueError('camera requires a frame from this G1 visual scene')
        if view == 'fixed':
            eye, aim = (3.80,-5.70,1.6),(1.82,0.,.66)
            vfov = 34.
        else:
            offsets = {'follow':(1.55,-2.32,.40),'side':(.12,-2.78,.34),'front':(2.82,-.14,.33)}
            if view not in offsets:
                raise ValueError('G1 camera view must be follow, side, front or fixed')
            pos = next(n.world_pose.position_m for n in frame.nodes if n.node_id==self.pelvis_node)
            # A fixed camera elevation and pitch avoid inheriting individual
            # stance-phase bob and torso rotation from the horizontal track.
            aim = np.array([pos[0]+.06,pos[1],.70])
            eye = aim+offsets[view]
            vfov = 35.
        focal = .5*height/tan(.5*radians(vfov))
        return Camera(id if id is not None else view,frame.key,look_at(eye,aim,(0.,0.,1.)),
                      width,height,focal,focal,width/2,height/2,.025,90.)

    def audit_transforms(self, data):
        """Compare Vulkan body×geom placement to MuJoCo's independent world geom arrays."""
        worst_origin = worst_basis = 0.
        for geom, body, local in self.geometry:
            actual = pose_matrix(compose_pose(_pose(data.xpos[body],data.xquat[body]),local))
            worst_origin = max(worst_origin,float(np.linalg.norm(actual[:3,3]-data.geom_xpos[geom])))
            worst_basis = max(worst_basis,float(np.max(np.abs(actual[:3,:3]-data.geom_xmat[geom].reshape(3,3)))))
        return dict(visual_geoms=len(self.geometry), max_origin_error_m=worst_origin, max_basis_error=worst_basis)


def build_unitree_visual(model, directory, *, robot='g1'):
    """Import group-1 non-colliding render surfaces of Unitree RL Gym's G1 12-DOF model.

    geom_pos/geom_quat already include the mesh compiler's principal-axis
    centering and alignment. Compiled mesh_vert/normal are in that frame:
    applying mesh_pos/mesh_quat again would displace the actual CAD.
    """
    import mujoco as mj
    if robot.lower()!='g1':
        raise ValueError('this visual binding supports the official G1 12-DOF scene')
    directory = Path(directory)
    visual = np.flatnonzero((model.geom_group==1) & (model.geom_type==mj.mjtGeom.mjGEOM_MESH)
                            & (model.geom_contype==0) & (model.geom_conaffinity==0))
    if len(visual)!=27 or model.nu!=12:
        raise ValueError(f'expected 27 visual meshes and 12 actuators in the G1 walk model, got {len(visual)}, {model.nu}')
    nodes = {i:'g1/'+(model.body(i).name or str(i)) for i in sorted(set(map(int,model.geom_bodyid[visual]))|{0})}
    pelvis = int(model.body('pelvis').id)
    meshes, materials, instances, geom_rows, geometry = [], {}, [], [], []
    by_mesh = {}
    next_id = int(model.ngeom)+1
    for gid in map(int,visual):
        mid, body, mat = int(model.geom_dataid[gid]), int(model.geom_bodyid[gid]), int(model.geom_matid[gid])
        if mid not in by_mesh:
            by_mesh[mid] = _cad_mesh(model,mid,directory)
            meshes.extend(by_mesh[mid][0])
        rgba = tuple(map(float,model.geom_rgba[gid]))
        if mat >= 0 and rgba == (.5,.5,.5,1.):
            rgba = tuple(map(float,model.mat_rgba[mat]))
        if rgba[3]!=1. or (mat>=0 and np.any(model.mat_texid[mat]>=0)):
            raise ValueError('G1 robot visuals must be opaque untextured CAD')
        metallic, rough = (float(model.mat_metallic[mat]),float(model.mat_roughness[mat])) if mat>=0 else (0.,.48)
        material_key = mat,rgba,metallic,rough
        if material_key not in materials:
            materials[material_key] = Material(f'g1-finish-{len(materials)}',rgba[:3],metallic,rough)
        material = materials[material_key]
        local = _pose(model.geom_pos[gid],model.geom_quat[gid])
        geometry.append((gid,body,local))
        ids = []
        for part,mesh in enumerate(by_mesh[mid][0]):
            identity = next_id if part else gid+1
            next_id += int(part>0)
            instances.append(Instance(identity,mesh.id,material.id,nodes[body],local))
            ids.append(identity)
        geom_rows.append(dict(geom_id=gid, mesh_id=mid, body_id=body, body=nodes[body], instance_ids=ids,
                              material=material.id, source_rgba=list(rgba), metallic=metallic, roughness=rough,
                              local_xyz=list(local.position_m), local_xyzw=list(local.orientation_xyzw)))
    floor_meshes, floor_materials, decor, stage = _workshop(model,directory,next_id,nodes[0])
    cad = [row for _,row in by_mesh.values()]
    pieces = [p for row in cad for p in row['chunks']]+stage
    report = dict(model='Unitree RL Gym G1, 12 DOF', visual_geoms=geom_rows, meshes=cad, stage=stage,
                  rgb_policy='authored geom/material RGB; unnamed untextured plastic uses roughness 0.48',
                  stage_policy='visual-only workshop at the actual ground-plane pose; 1 m joints and 0.25 m lane ticks',
                  floor_map_sha256=floor_materials[0].base_color_map.sha256,
                  robot_triangles=sum(row['triangles'] for row in cad),
                  scene_triangles=sum(p['triangles'] for p in pieces), unique_vertices=sum(p['vertices'] for p in pieces),
                  unique_meshes=len(pieces), instances=len(instances)+len(decor),
                  instanced_triangles=sum(by_mesh[g['mesh_id']][1]['triangles'] for g in geom_rows)
                                        +sum(s['triangles'] for s in stage))
    digest = sha256(json.dumps(report,sort_keys=True,allow_nan=False).encode()).hexdigest()
    scene = VisualScene('unitree-g1-walk:'+digest[:16],tuple(meshes+floor_meshes),
                        tuple(materials.values())+tuple(floor_materials),tuple(instances+decor))
    return UnitreeWalkVisual(scene,report,tuple(nodes.items()),nodes[pelvis],tuple(geometry),int(model.nbody))
