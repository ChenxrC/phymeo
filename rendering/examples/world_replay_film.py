"""Photograph one of the R4 worlds with a reference renderer in a separate Python environment."""
import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory


def save(path, value):
    path.write_text(json.dumps(value, separators=(",", ":"), allow_nan=False), encoding="utf-8")


def pack(source_dir, directory, world):
    from _r4_archive import ArchiveWorlds

    directory.mkdir(parents=True, exist_ok=False)
    with TemporaryDirectory() as temporary:
        source = ArchiveWorlds(source_dir, Path(temporary))
        scene = source.scenes[world]
        assets = []
        for row, mesh in zip(source.assets[world], scene.meshes, strict=True):
            name = row["mesh_file"]
            copied = directory / name
            copyfile(Path(temporary) / f"world-{world}" / name, copied)
            if sha256(copied.read_bytes()).hexdigest() != row["mesh_sha256"]:
                raise ValueError("world mesh copy differs from archived asset")
            assets.append(dict(id=mesh.id, file=name, sha256=row["mesh_sha256"]))
        frames = [source.frame(world, ordinal, 0) for ordinal in range(source.record_count)]
        camera = source.cameras(world, frames[0])[0]
        record = dict(schema="phymeo-r4-presentation-v1", world=source.manifest["worlds"][world],
                      source_sha256=source.source_digest, state_sha256=source.state_digest,
                      adapter_sha256=source.adapter_sha256, scene_key=scene.key, meshes=assets,
                      materials=[asdict(m) for m in scene.materials],
                      instances=[asdict(i) for i in scene.instances], lights=[asdict(l) for l in scene.lights],
                      camera_world_pose=asdict(camera.world_pose), frames=[asdict(f) for f in frames])
        source.assert_unchanged()
        save(directory / "episode.json", record)
        print(f"world {world}: {len(frames)} real published poses; {len(assets)} meshes")


def unpack(directory):
    from phymeo_rendering.assets import Material, Mesh
    from phymeo_rendering.scene import DirectionalLight, FrameKey, Instance, NodePose, Pose, SceneFrame, VisualScene

    def pose(row):
        return Pose(tuple(row["position_m"]),tuple(row["orientation_xyzw"]))

    data = json.loads((directory / "episode.json").read_text(encoding="utf-8"))
    if data["schema"] != "phymeo-r4-presentation-v1":
        raise ValueError("unexpected presentation schema")
    meshes = []
    for row in data["meshes"]:
        path = (directory / row["file"]).resolve(strict=True)
        if path.parent != directory.resolve() or sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("presentation mesh identity mismatch")
        meshes.append(Mesh(row["id"], path.as_uri()))
    materials = tuple(Material(r["id"],tuple(r["base_color"]),r["metallic"],r["roughness"]) for r in data["materials"])
    instances = tuple(Instance(r["id"],r["mesh_id"],r["material_id"],r["node_id"],pose(r["local_pose"])) for r in data["instances"])
    lights = tuple(DirectionalLight(tuple(r["direction"]),tuple(r["illuminance"])) for r in data["lights"])
    scene = VisualScene(data["scene_key"],tuple(meshes),materials,instances,lights)
    frames = tuple(SceneFrame(FrameKey(**f["key"]),f["time_s"],
                      tuple(NodePose(n["node_id"],pose(n["world_pose"])) for n in f["nodes"])) for f in data["frames"])
    if any(f.key.scene_key != scene.key for f in frames) or any(b.time_s<=a.time_s for a,b in zip(frames,frames[1:])):
        raise ValueError("inconsistent frame clock or scene")
    return data,scene,frames,pose(data["camera_world_pose"])


def subject_camera(scene, frames, original, width, height, asset_info):
    """Fit one fixed camera to the complete swept non-floor geometry."""
    import numpy as np
    from itertools import product
    from _panda_scene import pose_matrix
    from phymeo_rendering.mesh_data import load_mesh
    from phymeo_rendering.scene import Pose

    rotation = pose_matrix(original)[:3,:3]
    bounds = {}
    for mesh in scene.meshes:
        xyz = np.asarray(load_mesh(mesh).positions)
        bounds[mesh.id] = np.asarray(list(product(*zip(xyz.min(0),xyz.max(0),strict=True))))
    full, subject = [], []
    for frame in frames:
        nodes = {n.node_id:pose_matrix(n.world_pose) for n in frame.nodes}
        for item in scene.instances:
            transform = nodes[item.node_id] @ pose_matrix(item.local_pose)
            corners = (bounds[item.mesh_id]@transform[:3,:3].T+transform[:3,3])@rotation
            full.append(corners)
            if asset_info[item.id] != "plane":
                subject.append(corners)
    if not subject:
        raise ValueError("source has no non-floor subject")
    points = np.concatenate(subject)
    entire = np.concatenate(full)
    low, high = points.min(0),points.max(0)
    centre = (low+high)/2
    distance = max(np.linalg.norm(high-low)*1.65, entire[:,2].max()-centre[2]+.1)
    eye = centre+np.array([0,0,distance])
    rel = points-eye
    projected = rel[:,:2]/(-rel[:,2,None])
    # Leave eight percent margin on each side of the tightest dimension.
    limits = np.divide([width*.42,height*.42], np.maximum(np.max(abs(projected),axis=0),1e-10))
    focal = float(limits.min())
    position = rotation@eye
    return (Pose(tuple(position), original.orientation_xyzw), focal, float(eye[2]-entire[:,2].min()+.1),
            rotation@centre, float(np.linalg.norm(high-low)), (np.ptp(projected,axis=0)*focal).tolist())


def render(directory, output, device, framing_report):
    from PIL import Image
    import numpy as np
    from _panda_scene import pose_matrix
    from phymeo_rendering.backends.sapien import SapienBackend, SapienConfig, SapienStudio
    from phymeo_rendering.cameras import Camera
    from phymeo_rendering.mesh_data import load_mesh
    from phymeo_rendering.outputs import Channel
    from phymeo_rendering.pipeline import CaptureRequest, RenderPath
    from phymeo_rendering.scene import Pose

    data,scene,frames,original = unpack(directory)
    output.mkdir(parents=True,exist_ok=False)
    images = output / "frames"
    images.mkdir()
    width,height = 960,640
    # The reference primary shader requires every vertex in front of near.
    # Move the presentation camera along its original optical axis until the
    # complete envelope, including the authored finite floor, clears the slab.
    back = pose_matrix(original)[:3,2]
    origin = np.asarray(original.position_m)
    extents = {m.id:np.max(np.linalg.norm(load_mesh(m).positions,axis=1)) for m in scene.meshes}
    points = np.array([n.world_pose.position_m for f in frames for n in f.nodes])
    radius = max(extents.values())
    advance = max(0., np.max((points-origin)@back)+radius+.1)
    # Frame moving bodies, the floor remains visible as a supporting surface.
    active = [n.world_pose.position_m for n in frames[0].nodes if any(
        dict((t.node_id,t.world_pose) for t in f.nodes)[n.node_id] != n.world_pose for f in frames[1::20])]
    if not active:
        raise ValueError("archived world has no moving bodies")
    target = np.median(np.array(active),axis=0)
    nominal_depth = float((origin-target)@back)
    if nominal_depth<=0:
        raise ValueError("source camera points away from moving bodies")
    camera_position = origin+advance*back
    focal = 270.*height/240.*(nominal_depth+advance)/nominal_depth
    far = float(np.max((camera_position-points)@back)+radius+.5)
    clip_pose = Pose(tuple(float(x) for x in camera_position),original.orientation_xyzw)
    framing = {"mode":"archive overview"}
    light_scale = max(.15,nominal_depth*.65)
    if framing_report:
        source = json.loads(framing_report.read_text(encoding="utf-8"))["source"]
        if source["manifest_sha256"] != data["source_sha256"] or source["state_sha256"] != data["state_sha256"]:
            raise ValueError("framing metadata belongs to a different archive")
        assets = source["assets"][data["world"]["world"]]
        if {a["mesh_file"]:a["mesh_sha256"] for a in assets} != {m["file"]:m["sha256"] for m in data["meshes"]}:
            raise ValueError("source geometry and framing assets disagree")
        kinds = {a["id"]:a["kind"] for a in assets}
        clip_pose, focal, far, target, extent, coverage = subject_camera(scene,frames,original,width,height,kinds)
        light_scale = max(.15,extent*1.4)
        framing = {"mode":"fit complete non-floor sweep","subject_extent_px":coverage,
                   "camera_displacement_world_m":(np.asarray(clip_pose.position_m)-origin).tolist(),
                   "archived_materials":True}
    path = RenderPath.PATH_TRACE
    studio = SapienStudio(center_m=tuple(float(x) for x in target),scale_m=light_scale)
    session = SapienBackend(SapienConfig(path=path, device=device, samples_per_pixel=256, studio=studio)).open_scene(scene)
    hashes = []
    try:
        for index,frame in enumerate(frames):
            camera = Camera("reference-film",frame.key,clip_pose,width,height,focal,focal,width/2,height/2,.005,far)
            shot = session.capture(frame,CaptureRequest(camera,path,(Channel.RGB,)))
            if shot.frame_key != frame.key or shot.time_s != frame.time_s:
                raise ValueError("rendered wrong source state")
            rgb = shot.planes[0].data
            hashes.append(sha256(rgb).hexdigest())
            Image.frombytes("RGB",(width,height),rgb).save(images/f"{index:04d}.png",compress_level=2)
            if (index+1)%40==0:
                print(f"{index+1}/{len(frames)}",flush=True)
    finally:
        session.close()
    save(output/"capture.json",dict(source_excerpt_sha256=sha256((directory/"episode.json").read_bytes()).hexdigest(),
         backend=session.diagnostics(),world=data["world"],frames=len(frames),fps=30,size=[width,height],
         source_times_s=[f.time_s for f in frames],frame_sha256=hashes,original_camera=asdict(original),
         camera_pose=asdict(clip_pose),clip=[.005,far],focal_pixels=focal,framing=framing))


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    modes=cli.add_subparsers(dest="mode",required=True)
    prep=modes.add_parser("pack")
    prep.add_argument("source",type=Path)
    prep.add_argument("destination",type=Path)
    prep.add_argument("--world",type=int,default=5,choices=range(8))
    film=modes.add_parser("render")
    film.add_argument("input",type=Path)
    film.add_argument("output",type=Path)
    film.add_argument("--device",default="cuda:0")
    film.add_argument("--fit-source-subject",type=Path,metavar="R4_SUMMARY")
    args=cli.parse_args()
    if args.mode=="pack":
        pack(args.source,args.destination,args.world)
    else:
        render(args.input,args.output,args.device,args.fit_source_subject)
