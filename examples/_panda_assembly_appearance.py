"""Render-only classic Panda CAD and a workcell for the two-arm assembly replay."""

from dataclasses import dataclass, replace
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import re

import mujoco as mj
import numpy as np

from bimanual_wrist_render import _box_faces, _round_faces
from phymeo_rendering._prepare_cache import publish_bytes
from phymeo_rendering._prepare_geometry import _mesh_bytes
from phymeo_rendering.assets import Material, Mesh
from phymeo_rendering.panda_visuals import panda_visual_assets, _default_asset_dir
from phymeo_rendering.scene import FrameKey, Instance, NodePose, Pose, SceneFrame, VisualScene


@dataclass
class AssemblyAppearance:
    scene: VisualScene
    provenance: dict
    robot_ids: dict
    tool_ids: dict
    _nodes: tuple

    def frame(self, data, tick, time_s):
        """Read published mj_kinematics results without setting a joint or running a step."""
        key = FrameKey(self.scene.key,"assembly",0,int(tick),int(tick),"accepted_endpoint")
        nodes = tuple(NodePose(name,Pose(tuple(map(float,data.xpos[body])),
                                         tuple(map(float,data.xquat[body,[1,2,3,0]]))))
                      for body,name in self._nodes)
        return SceneFrame(key,float(time_s),nodes)


def _surface_shells(source_scene, root, cache):
    # Use the same tested corner-angle normal treatment as the Panda video.
    path = Path(__file__).resolve().parents[1]/"rendering/examples/preview_stable_lighting/shell_normals.py"
    spec = importlib.util.spec_from_file_location("_assembly_cad_normals",path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # The verified assets can live outside a source checkout (e.g., wheel).
    # Bind just this isolated helper's manifest to the same verified archive.
    module.MANIFEST = root/"manifest.json"
    return module.scene_with_smooth_shells(source_scene,cache)


def build_appearance(model, cache_dir, *, asset_dir=None, smooth_shells=True):
    """Build one reusable PBR scene; both arms share meshes, not poses or IDs.

    Only original URDF envelope geoms named *_proxyN are suppressed. All
    apparatus geoms use the source size, attachment, position and orientation.
    robot_ids maps 'left'/'right' to stable tuples of segmentation IDs; tool_ids
    maps each source apparatus geom name to its original (geom id + 1).
    """
    root = Path(asset_dir or _default_asset_dir()).resolve()
    cache = Path(cache_dir).resolve()
    cache.mkdir(parents=True,exist_ok=True)
    meshes,mats,parts = panda_visual_assets(root)
    manifest = json.loads((root/"manifest.json").read_bytes())
    source = VisualScene("pinned-classic-panda",meshes,mats,parts)
    normal_audit = None
    if smooth_shells:
        source, normal_audit = _surface_shells(source,root,cache/"shell-normals")
        if normal_audit["skipped"]:
            raise ValueError(f"source CAD normal/material mismatch: {normal_audit['skipped']}")
    # The manufacturer's bright housings use diffuse white with no authored
    # Phong gloss. A neutral satin finish retains the black seams and inserts.
    shell_ids = {"panda_visual_"+suffix+"_material" for suffix in (
        "link0_03","link1_00","link2_00","link3_01","link3_03","link4_02","link5_00",
        "link6_00","link7_00","link7_07","hand_00","hand_01","hand_04")}
    materials = [replace(m,base_color=tuple(c*.81 for c in m.base_color),
                         metallic=.025,roughness=.46) if m.id in shell_ids else m for m in mats]
    swatches = {
        "plate": ((.102,.32,.46),.24,.53), "pin": ((.70,.51,.21),.61,.37),
        "holder": ((.25,.30,.32),.45,.50), "index": ((.82,.10,.025),.05,.37),
        "worktop": ((.37,.40,.40),.10,.53), "frame": ((.078,.128,.143),.38,.41),
        "steel": ((.43,.48,.49),.62,.31), "foot": ((.040,.055,.059),.05,.64),
        "floor": ((.33,.365,.37),0.,.75), "wall": ((.64,.69,.69),0.,.80),
        "cabinet": ((.30,.38,.39),.06,.62), "handle": ((.24,.28,.29),.28,.48),
    }
    materials.extend(Material("cell-"+name,*params) for name,params in swatches.items())
    nodes = {}
    placements = []
    robot_ids = {}
    tool_ids = {}
    shapes = {}
    primitive_meshes = []

    def node(name):
        b = mj.mj_name2id(model,mj.mjtObj.mjOBJ_BODY,name)
        if b < 0:
            raise ValueError(f"assembly MJCF lacks required CAD attachment {name}")
        nodes[b] = name
        return name

    for i,side in enumerate(("left","right")):
        ids = []
        for p in parts:
            object_id = 1000*(i+1)+p.id
            ids.append(object_id)
            placements.append(replace(p,id=object_id,node_id=node(side+"_"+p.node_id)))
        robot_ids[side] = tuple(ids)

    def shape(kind,size):
        size = tuple(map(float,size))
        key = int(kind),size
        if key not in shapes:
            if kind==mj.mjtGeom.mjGEOM_BOX:
                v,n,f = _box_faces(np.array(size))
            elif kind in (mj.mjtGeom.mjGEOM_CAPSULE,mj.mjtGeom.mjGEOM_CYLINDER,
                          mj.mjtGeom.mjGEOM_SPHERE,mj.mjtGeom.mjGEOM_ELLIPSOID):
                v,n,f = _round_faces(kind,size,sectors=96,hemi=24)
            else:
                raise ValueError(f"unknown assembly apparatus shape {kind}")
            data = _mesh_bytes(v,n,None,f)
            h = sha256(data).hexdigest()
            path = cache/(h+".gltf")
            publish_bytes(path,data)
            mesh = Mesh("cell-"+h,path.as_uri())
            shapes[key] = mesh.id
            if mesh.id not in {p.id for p in primitive_meshes}:
                primitive_meshes.append(mesh)
        return shapes[key]

    skipped = []
    finish = dict(assembly_pin="pin",assembly_pin_adapter="holder",board_carrier="holder",
                  spindle_orientation_marker="index",assembly_table_visual="worktop")
    for g in range(model.ngeom):
        geom = model.geom(g)
        name = geom.name or f"geom-{g}"
        if re.fullmatch(r"(?:left|right)_panda_\w+_proxy\d+",name):
            skipped.append(name)
            continue
        if model.geom_rgba[g,3]==0:
            continue
        if model.geom_rgba[g,3]!=1:
            raise ValueError(f"unexpected translucent assembly geom {name}")
        owner = model.body(int(model.geom_bodyid[g])).name
        material = "plate" if name.startswith("board_wall_") else finish.get(name)
        if material is None:
            mat = Material("apparatus-"+name,tuple(map(float,model.geom_rgba[g,:3])),.12,.48)
            materials.append(mat)
            material_id = mat.id
        else:
            material_id = "cell-"+material
        tool_ids[name] = g+1
        w,x,y,z = map(float,model.geom_quat[g])
        placements.append(Instance(g+1,shape(model.geom_type[g],model.geom_size[g]),material_id,
                                   node(owner),Pose(tuple(map(float,model.geom_pos[g])),(x,y,z,w))))
    if len(skipped)==0 or not {"assembly_pin", "board_wall_0", "assembly_table_visual"} <= tool_ids.keys():
        raise ValueError("expected named Panda proxy geoms, both assembly tools and the worktop")
    # All bench decoration is stationary. Nothing is mounted to the free pin,
    # its spindle marker, the four aperture walls, or the moving gripper.
    static = node("world")
    nset = 5000
    def prop(kind,size,p,material):
        nonlocal nset
        placements.append(Instance(nset,shape(kind,size),"cell-"+material,static,Pose(tuple(map(float,p)))))
        nset += 1
    def box(p,half,material):
        prop(mj.mjtGeom.mjGEOM_BOX,half,p,material)
    def cylinder(p,r,h,material):
        prop(mj.mjtGeom.mjGEOM_CYLINDER,(r,h,0.),p,material)

    d = mj.MjData(model)
    mj.mj_kinematics(model,d)
    table = model.geom("assembly_table_visual").id
    table_r = d.geom_xmat[table].reshape(3,3)
    if not np.allclose(table_r,np.eye(3),atol=1e-6):
        raise ValueError("industrial bench dressing requires the upright assembly worktop")
    cx,cy,cz = d.geom_xpos[table]
    hx,hy,hz = model.geom_size[table]
    surface = cz+hz
    bottom = cz-hz
    floor_z = bottom-.78
    box((cx,cy,bottom-.014),(hx-.007,hy-.007,.014),"frame")
    for x in (cx-hx+.07,cx+hx-.07):
        for y in (cy-hy+.07,cy+hy-.07):
            box((x,y,(bottom+floor_z)/2),(.030,.030,(bottom-floor_z)/2),"steel")
            box((x,y,floor_z+.016),(.041,.041,.016),"foot")
        box((x,cy,floor_z+.21),(.022,hy-.06,.016),"frame")
    box((cx,cy+hy-.07,floor_z+.21),(hx-.06,.022,.016),"frame")
    for side in robot_ids:
        x,y,z = d.xpos[model.body(side+"_panda_link0").id]
        stand = z-surface
        if stand>.016:
            cylinder((x,y,surface+.006),.115,.006,"foot")
            cylinder((x,y,surface+.012+(stand-.012)/2),.096,(stand-.012)/2,"steel")
            # Four exterior flange heads stay below the real base attachment.
            for a in np.arange(4)*np.pi/2+np.pi/4:
                cylinder((x+.104*np.cos(a),y+.104*np.sin(a),surface+.013),.005,.002,"frame")
    box((cx,cy,floor_z-.022),(6.,6.,.022),"floor")
    box((cx,cy+2.85,floor_z+1.8),(5.,.04,1.8),"wall")
    # Recessed wall joint and low skirting give scale without busy shelving
    # or painted markings near the aperture.
    box((cx,cy+2.80,floor_z+.059),(5.,.012,.059),"frame")
    # The rear bench sits beyond the work envelope. Repeated, low-contrast
    # cabinet fronts give perspective; no image-space shadows or painted highlights.
    for col in range(-3, 4):
        x = cx + .70*col
        box((x,cy+2.22,floor_z+.49),(.344,.24,.42),"cabinet")
        box((x,cy+1.966,floor_z+.76),(.115,.014,.009),"handle")
    box((cx,cy+2.22,floor_z+.925),(2.46,.265,.018),"steel")
    all_meshes = source.meshes + tuple(primitive_meshes)
    source_hash = sha256((root/"manifest.json").read_bytes()).hexdigest()
    key = sha256(repr((source_hash,bool(smooth_shells),tuple(placements),tuple(materials),all_meshes)).encode()).hexdigest()
    scene = VisualScene("dual-panda-cad-"+key,all_meshes,tuple(materials),tuple(placements))
    proof = dict(cad_source=manifest["source_repository"],cad_commit=manifest["source_commit"],cad_manifest_sha256=source_hash,
        license="Apache-2.0",copyright="2017 Franka Emika GmbH",license_files=manifest["licenses"],
        source_coordinate_frame=manifest["coordinate_frame"],cad_parts_per_arm=len(parts),
        cad_triangles_per_arm=manifest["totals"]["surface_triangles_instanced"],shared_cad_meshes=len(source.meshes),
        appearance="Manufacturer Panda shells with a render-only neutral satin finish; positions and topology preserved.",
        shell_normal_audit=normal_audit,
        robot_ids=robot_ids,apparatus_ids=tool_ids,hidden_envelopes=skipped,
        furniture_objects=nset-5000,scene_key=scene.key)
    return AssemblyAppearance(scene,proof,robot_ids,tool_ids,tuple(sorted(nodes.items())))
