"""Eight actual saved native models mapped to immutable rendering frames.

Uses the archive's existing SHA/order/clock checks and read-only mj_kinematics.
No mj_step, dynamics owner, renderer, or alternate graphics API is constructed.
Curved primitives are explicitly tessellated; infinite planes are finite visual
patches. Transparent target sites are excluded from this opaque-geometry test.
"""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from dataclasses import asdict


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _pose(position, rotation):
    import mujoco as mj
    import numpy as np
    from phymeo_rendering.scene import Pose
    quaternion = np.empty(4)
    mj.mju_mat2Quat(quaternion, np.asarray(rotation).reshape(9))
    w,x,y,z = quaternion
    return Pose(tuple(float(x) for x in position),(float(x),float(y),float(z),float(w)))


def _round_mesh(directory, name, kind, size):
    """12-sector analytic-profile triangulation; no source dimensions changed."""
    import numpy as np
    from _panda_scene import write_mesh
    radius, half = float(size[0]), float(size[1])
    if kind == 'sphere': half=0.
    if kind == 'cylinder':
        profile=[(0.,half),(radius,half),(radius,-half),(0.,-half)]
    else:
        profile=[(radius*math.sin(a),half+radius*math.cos(a))
                 for a in [i*math.pi/8 for i in range(5)]]
        if half: profile.append((radius,-half))
        profile += [(radius*math.sin(a),-half+radius*math.cos(a))
                    for a in [i*math.pi/8 for i in range(5,9)]]
        profile[0]=(0.,half+radius); profile[-1]=(0.,-half-radius)
    positions=[]; normals=[]; triangles=[]
    def emit(a,b,c):
        normal=np.cross(np.subtract(b,a),np.subtract(c,a))
        length=float(np.linalg.norm(normal))
        if not length>0: raise ValueError('degenerate source primitive triangulation')
        offset=len(positions)
        positions.extend((a,b,c)); normals.extend((normal/length,)*3)
        triangles.append((offset,offset+1,offset+2))
    def point(row,sector):
        r,z=row; angle=2*math.pi*(sector%12)/12
        return (r*math.cos(angle),r*math.sin(angle),z)
    for top,bottom in zip(profile,profile[1:]):
        for sector in range(12):
            a,b=point(top,sector),point(top,sector+1)
            c,d=point(bottom,sector),point(bottom,sector+1)
            if top[0]==0: emit(a,c,d)
            elif bottom[0]==0: emit(a,c,b)
            else: emit(a,c,b); emit(b,c,d)
    return write_mesh(directory,name,positions,normals,triangles)


class ArchiveWorlds:
    """Frozen archive source; reset is replay-only and never alters saved q/v."""
    physics_steps = 0

    def __init__(self, root, asset_directory):
        import mujoco as mj
        import numpy as np
        from _panda_scene import box, write_mesh, look_at
        from phymeo_rendering.assets import Material
        from phymeo_rendering.scene import DirectionalLight, Instance, VisualScene
        root=Path(root).resolve(strict=True)
        adapter=Path(__file__).resolve().parents[2]/'scripts'/'render_manipulation_gallery.py'
        spec=importlib.util.spec_from_file_location('_r4_original_archive_reader',adapter)
        module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        self.manifest,self.records,models,self.geometry,cameras=module.read_run(root)
        self.adapter_sha256=_sha(adapter)
        self.root=root; self.scenes=[]; self.fixed_cameras=[]; self.assets=[]
        self.source_digest=_sha(root/'manifest.json')
        self.state_digest=self.manifest['state_sha256']
        self.record_count=len(self.records)
        self._input_digest=hashlib.sha256(json.dumps(self.records,sort_keys=True,
            allow_nan=False,separators=(',',':')).encode()).hexdigest()
        kindmap={int(mj.mjtGeom.mjGEOM_BOX):'box',int(mj.mjtGeom.mjGEOM_PLANE):'plane',
                 int(mj.mjtGeom.mjGEOM_SPHERE):'sphere',int(mj.mjtGeom.mjGEOM_CAPSULE):'capsule',
                 int(mj.mjtGeom.mjGEOM_CYLINDER):'cylinder'}
        for world,(row,model,camera) in enumerate(zip(self.manifest['worlds'],models,cameras,strict=True)):
            directory=Path(asset_directory)/f'world-{row["world"]}'
            directory.mkdir(parents=True,exist_ok=False)
            meshes=[]; materials=[]; instances=[]; asset_rows=[]
            for geom in range(model.ngeom):
                kind=kindmap.get(int(model.geom_type[geom]))
                if kind is None: raise ValueError(f'unsupported source geom type {model.geom_type[geom]}')
                size=tuple(float(x) for x in model.geom_size[geom]); name=f'geom-{geom}'
                if float(model.geom_rgba[geom,3])!=1.:
                    raise ValueError('source geom is not opaque; no silent alpha conversion')
                if kind=='box': mesh=box(directory,name,tuple(2*x for x in size))
                elif kind=='plane':
                    h=max(float(camera.distance)*2, size[0], size[1],1.)
                    mesh=write_mesh(directory,name,((-h,-h,0),(h,-h,0),(h,h,0),(-h,h,0)),
                                    ((0,0,1),)*4,((0,1,2),(0,2,3)))
                else: mesh=_round_mesh(directory,name,kind,size)
                meshes.append(mesh)
                materials.append(Material(name,tuple(float(x) for x in model.geom_rgba[geom,:3]),0.,.65))
                instances.append(Instance(geom+1,name,name,name))
                asset_rows.append(dict(id=geom+1,source_geom=geom,source_name=mj.mj_id2name(model,mj.mjtObj.mjOBJ_GEOM,geom),
                    kind=kind,size=size,mesh_file=Path(mesh.uri.split('/')[-1]).name,
                    mesh_sha256=_sha(directory/(name+'.gltf')),
                    finite_plane_half_extent_m=h if kind=='plane' else None))
            scene=VisualScene(f'r4:{row["source_sha256"]}',tuple(meshes),tuple(materials),tuple(instances),
                (DirectionalLight((.36,-.48,-.8),(1.,1.,1.)),))
            self.scenes.append(scene); self.assets.append(asset_rows)
            target=np.asarray(camera.lookat); distance=float(camera.distance)
            poses=[]
            for azimuth in (125.,-55.):
                a=math.radians(azimuth); elevation=math.radians(32.)
                eye=target+distance*np.array((math.cos(a)*math.cos(elevation),math.sin(a)*math.cos(elevation),math.sin(elevation)))
                poses.append(look_at(tuple(eye),tuple(target),(0,0,1)))
            self.fixed_cameras.append(tuple(poses))
        if len({row['source_sha256'] for row in self.manifest['worlds']})!=8:
            raise ValueError('R4 requires eight genuinely different archived model assets')

    def frame(self, world, source_index, epoch):
        from phymeo_rendering.scene import FrameKey, SceneFrame, NodePose
        row=self.manifest['worlds'][world]; state=self.records[source_index]['worlds'][world]
        positions,rotations,_,_=self.geometry[source_index][world]
        key=FrameKey(self.scenes[world].key,f'archive:{self.state_digest}:world:{row["world"]}',
                     epoch,int(state['tick']),int(state['tick']),'accepted_endpoint')
        return SceneFrame(key,float(state['time']),tuple(NodePose(f'geom-{i}',_pose(p,r))
                    for i,(p,r) in enumerate(zip(positions,rotations,strict=True))))

    def cameras(self, world, frame):
        from phymeo_rendering.cameras import Camera
        return tuple(Camera(name,frame.key,pose,320,240,270.,270.,160.,120.,.005,20.)
                     for name,pose in zip(('front','reverse'),self.fixed_cameras[world],strict=True))

    def assert_unchanged(self):
        digest=hashlib.sha256(json.dumps(self.records,sort_keys=True,allow_nan=False,
                                       separators=(',',':')).encode()).hexdigest()
        if digest!=self._input_digest or self.physics_steps!=0:
            raise ValueError('archive input mutated or physics advanced')

    def provenance(self):
        return dict(source_directory=str(self.root),manifest_sha256=self.source_digest,
            state_sha256=self.state_digest,record_count=self.record_count,
            adapter='scripts/render_manipulation_gallery.py:read_run; mj_kinematics only',
            adapter_sha256=self.adapter_sha256,
            physics_step_evidence='no physics owner or integration is constructed; only archived q and mj_kinematics',
            worlds=self.manifest['worlds'],assets=self.assets,physics_steps=self.physics_steps,
            reset='replay epoch increment + actual source record zero; not a live physics reset',
            visuals='actual geom dimensions and FK, flat 12-sector curved tessellation; finite plane patches',
            material='source geom RGB interpreted as linear; declared opaque metallic=0 roughness=.65',
            excluded='transparent target sites are annotations, not included in opaque geometry capture')
