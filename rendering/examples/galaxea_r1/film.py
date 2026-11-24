"""Film a joint-limited Galaxea R1 URDF kinematic demonstration at 720p.

Run as python -m galaxea_r1.film with rendering/src and rendering/examples on
PYTHONPATH. --kinematics-only exports the same motion and cameras without Vulkan.
"""
import argparse
from dataclasses import asdict, replace
from hashlib import sha256
import json
import math
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image


SIZE = (1280, 720)
FPS = 30
SHOT_CUTS = (.45, .79)
CURVATURE = .31  # inverse metres, positive yaw towards the robot's left


def ease(t, start, end):
    x = min(1., max(0., (t-start)/(end-start)))
    return x*x*x*(10. + x*(-15. + 6.*x))


def wheel_geometry(model):
    from phymeo_rendering.scene import Pose
    nodes = {p.node_id: p.world_pose for p in model.pose_nodes({}, Pose())}
    geometry = []
    for number in (1, 2, 3):
        steer = model.joints[f"steer_motor_joint{number}"]
        wheel = model.joints[f"wheel_motor_joint{number}"]
        x, y, _ = nodes[steer.child].position_m
        # Measured from original wheel vertices in the plane normal to its axle.
        radius = model.wheel_radius_m
        if wheel.kind != "continuous" or not math.isfinite(radius) or radius <= 0.:
            raise ValueError(f"invalid R1 wheel joint or tread radius at {wheel.child}: {radius}")
        geometry.append((x, y, radius))
    return geometry


def motion(model, wheels, t):
    from phymeo_rendering.scene import Pose
    # Move on a circular arc. Each wheel has its own steering tangent and
    # travel distance. The wheels stop with the base during the arm close-up.
    distance = .89*ease(t, .015, .43) + .25*ease(t, .81, .985)
    yaw = CURVATURE*distance
    base = Pose((-.56 + math.sin(yaw)/CURVATURE, -.08 + (1.-math.cos(yaw))/CURVATURE,
                 model.floor_offset_m), (0., 0., math.sin(yaw/2), math.cos(yaw/2)))
    q = {}
    for number, (x, y, radius) in enumerate(wheels, start=1):
        forward, left = 1.-CURVATURE*y, CURVATURE*x
        q[f"steer_motor_joint{number}"] = math.atan2(left, forward)
        q[f"wheel_motor_joint{number}"] = distance*math.hypot(forward, left)/radius
    lift = ease(t, .10, .59)-ease(t, .78, .98)
    q.update(torso_joint1=-.24+.15*lift, torso_joint2=.47-.29*lift,
             torso_joint3=.23-.14*lift, torso_joint4=-.12+.29*lift)
    for side, sign, delay in (("left", 1., 0.), ("right", -1., .045)):
        reach = ease(t, .24+delay, .60+delay)-ease(t, .77+delay, .96+delay)
        roll = ease(t, .48+delay, .64+delay)-ease(t, .70+delay, .88+delay)
        # j1 and j5 have opposite physical Y axes on the two shoulders.
        # j2/j3 already have mirrored physical Z axes in the URDF.
        arm = (sign*(.99-.64*reach), 1.72+.69*reach, -1.25-.68*reach,
               sign*(.08+.21*roll), sign*(-.46+.22*reach), sign*(.10+.60*roll))
        q.update({f"{side}_arm_joint{i}": a for i, a in enumerate(arm, start=1)})
        gap = .008 + .038*(ease(t,.45+delay,.62+delay)-ease(t,.67+delay,.79+delay))
        for finger in (1, 2):
            q[f"{side}_gripper_finger_joint{finger}"] = gap
    for name, value in q.items():
        j = model.joints[name]
        if j.kind != "continuous" and not j.lower <= value <= j.upper:
            raise ValueError(f"{name} outside the imported URDF limit at t={t}: {value}")
    return base, q


def stage(model, directory):
    """A continuous floor/cyclorama and short fixed half-metre floor marks."""
    from _panda_scene import write_mesh
    from phymeo_rendering.assets import Material
    from phymeo_rendering.scene import Instance
    directory.mkdir(parents=True, exist_ok=True)
    n = 64
    p, normals, triangles = [(0.,0.,0.)], [(0.,0.,1.)], []
    rings = [(4.5+math.sin(a), 1.-math.cos(a), a) for a in np.linspace(0, math.pi/2, 9)]
    rings.append((5.5, 4., math.pi/2))
    for k, (radius, height, slope) in enumerate(rings):
        for j in range(n):
            theta = 2.*math.pi*j/n
            cs, sn = math.cos(theta), math.sin(theta)
            p.append((radius*cs, radius*sn, height))
            normals.append((-cs*math.sin(slope), -sn*math.sin(slope), math.cos(slope)))
            i = 1+k*n+j
            adjacent = 1+k*n+(j+1)%n
            if k:
                triangles.extend(((i-n, i, adjacent), (i-n, adjacent, adjacent-n)))
            else:
                triangles.append((0, i, adjacent))
    surface = write_mesh(directory,"r1-stage-cyclorama",p,normals,triangles)
    vertices, indices = [], []
    for tick in range(-5, 6):
        for side in (-1,1):
            x, y = .5*tick, side*.66
            length = .12 if tick % 2 == 0 else .065
            first = len(vertices)
            vertices.extend(((x-.008,y-length/2,.001), (x+.008,y-length/2,.001),
                             (x+.008,y+length/2,.001), (x-.008,y+length/2,.001)))
            indices.extend(((first,first+1,first+2),(first,first+2,first+3)))
    ticks = write_mesh(directory,"r1-stage-floor-marks",vertices,[(0.,0.,1.)]*len(vertices),indices)
    floor = Material("r1-stage-mineral",(.21,.265,.28),.0,.85)
    mark = Material("r1-stage-paint",(.39,.46,.45),.0,.72)
    scene = model.scene
    first_id = max(i.id for i in scene.instances)+1
    items = (Instance(first_id,surface.id,floor.id,"r1-film-stage"),
             Instance(first_id+1,ticks.id,mark.id,"r1-film-stage"))
    return replace(scene, key=scene.key+":studio-arc-v1", meshes=(*scene.meshes,surface,ticks),
                   materials=(*scene.materials,floor,mark), instances=(*scene.instances,*items))


def camera_for(frame, t):
    from _panda_scene import look_at
    from phymeo_rendering.cameras import Camera
    nodes = {p.node_id: np.asarray(p.world_pose.position_m) for p in frame.nodes}
    if t < SHOT_CUTS[0]:
        label, eye, aim, focal = "drive", (2.10,-3.48,1.63), (.05,0.,.82), 1460.
    elif t < SHOT_CUTS[1]:
        # Include the real hand motion on both sides. Follow the chassis only,
        # leaving articulation visible against a steady camera during the stop.
        offset = nodes["base_link"][:2]
        label, eye = "hands", (offset[0]+2.65,offset[1]-.62,1.54)
        aim, focal = (offset[0]+.22,offset[1]+.03,1.15), 1410.
    else:
        label, eye, aim, focal = "return", (-2.18,3.55,1.22), (.24,.14,.84), 1480.
    return Camera(label,frame.key,look_at(eye,aim,(0,0,1)),*SIZE,focal,focal,SIZE[0]/2,SIZE[1]/2,.025,16.)


def prepare(model, scene, count):
    from phymeo_rendering.scene import FrameKey, NodePose, Pose, SceneFrame
    frames, records = [], []
    wheels = wheel_geometry(model)
    for index in range(count):
        t = index/(count-1)
        base, joints = motion(model, wheels, t)
        key = FrameKey(scene.key, "galaxea-r1-authored-fk", 0, index, index, "authored")
        nodes = (*model.pose_nodes(joints, base), NodePose("r1-film-stage", Pose()))
        frame = SceneFrame(key, index/FPS, nodes)
        cam = camera_for(frame,t)
        frames.append((frame,cam))
        records.append(dict(frame=index, time_s=index/FPS, root=asdict(base), joints=joints,
                            camera=asdict(cam), link_poses={n.node_id: asdict(n.world_pose) for n in nodes[:-1]}))
    return frames, records


def write_json(path, value):
    path.write_text(json.dumps(value, separators=(",",":"), allow_nan=False)+"\n", encoding="utf-8")


def run(args):
    if args.frames < 2 or (args.still is not None and not 0 <= args.still < args.frames):
        raise ValueError("provide at least two storyboard frames and an in-range still index")
    if not args.kinematics_only:
        from preview_stable_lighting.runtime import activate
        activate(args.package)
    from .model import load_model
    args.output.mkdir(parents=True, exist_ok=True)
    trajectory = args.output/"trajectory.json"
    if trajectory.exists():
        raise FileExistsError(f"choose a new output directory: {trajectory} already exists")
    model = load_model(args.asset_dir, args.cache_dir or args.output/"model-cache")
    scene = stage(model, args.output/"stage")
    frames, records = prepare(model,scene,args.frames)
    write_json(trajectory,dict(motion="authored joint-limited URDF forward kinematics",
                               mode="FK data only" if args.kinematics_only else "GPU film input",
                               ground_offset_m=model.floor_offset_m, frame_rate=FPS, frames=records))
    if args.kinematics_only:
        return
    from preview_stable_lighting.film_aa import film_capture
    from preview_stable_lighting.runtime import StableBackend
    from preview_compact_rigid_history.state import bind_presentations
    from phymeo_rendering.outputs import Channel
    from phymeo_rendering.pipeline import CaptureRequest, RenderPath
    bindings = bind_presentations([frame for frame,_ in frames], sha256(trajectory.read_bytes()).hexdigest())
    from phymeo_rendering.environment import RadianceEnvironment
    from phymeo_rendering.studio import StudioLighting
    panorama = Path(__file__).resolve().parents[1]/"materials"/"university_workshop_1k.hdr"
    # Represent both emitters in quarter radiance units, compensating at
    # exposure. The HDR workshop's brightest texels then fit the backend.
    light_unit = .25
    environment = RadianceEnvironment(panorama.as_uri(),
        "a125ada38fc602025326f29b68df09531bb16f899f1973b03d9ac5ac98142548",
        strength=.18*light_unit, yaw_degrees=25., face_size=256)
    exposure = 2./light_unit
    studio = StudioLighting(center_m=(.03,.04,.85),scale_m=1.5,panel_size_scale=1.55,
                            panel_strength=.52*light_unit,environment_map=environment)
    owner = StableBackend(args.package, studio=studio, device_index=args.device_index,
                          validation=args.validation, exposure=exposure, display_transform="pbr-neutral").open_scene(scene)
    proc = None
    movie = args.output/"galaxea-r1.mp4"
    temporary = args.output/"galaxea-r1.partial.mp4"
    indices = range(args.frames) if args.still is None else (args.still,)
    selected = {0, round((args.frames-1)*.60), args.frames-1, args.still}
    try:
        if args.still is None:
            if movie.exists() or temporary.exists():
                raise FileExistsError("R1 movie or temporary movie already exists")
            proc = subprocess.Popen(["ffmpeg","-hide_banner","-loglevel","error","-nostdin","-n",
                "-f","rawvideo","-pixel_format","rgb24","-video_size","1280x720","-framerate",str(FPS),"-i","-",
                "-an","-vf","scale=in_range=full:out_range=limited:out_color_matrix=bt709",
                "-c:v","libx264","-threads","2","-crf","16","-pix_fmt","yuv420p","-colorspace","bt709",
                "-color_primaries","bt709","-color_trc","iec61966-2-1","-movflags","+faststart",str(temporary)],
                stdin=subprocess.PIPE)
        for i in indices:
            frame, camera = frames[i]
            request = CaptureRequest(camera, RenderPath.RASTER, (Channel.RGB,))
            shot, _ = film_capture(owner, frame, request, presentation=bindings[i], samples=args.samples,
                                   exposure=exposure, display_transform="pbr-neutral")
            if shot.camera != camera or shot.frame_key != frame.key:
                raise RuntimeError("renderer published a different R1 exposure")
            pixels = shot.planes[0].data
            if i in selected:
                Image.frombytes("RGB", SIZE, pixels).save(args.output/f"frame-{i:03}.png")
            if proc:
                proc.stdin.write(pixels)
            if i % 30 == 0:
                print(f"R1 frame {i}/{args.frames}", flush=True)
        if proc:
            proc.stdin.close()
            if proc.wait():
                raise RuntimeError("R1 H.264 encoder failed")
    finally:
        try:
            if proc and proc.poll() is None:
                proc.kill()
                proc.wait()
        finally:
            owner.close()
    device = owner.diagnostics()["device"]
    if device["validation_errors"] or device["live_resources"]:
        raise RuntimeError("Vulkan reported errors or leaked R1 resources")
    report = dict(engine="Phymeo native resident Vulkan", device=device["device_name"], film_aa=args.samples,
                  validation_enabled=args.validation, validation_errors=device["validation_errors"],
                  live_resources_after_close=device["live_resources"], model=model.diagnostics,
                  width=SIZE[0], height=SIZE[1], exposure=exposure, camera_cuts=SHOT_CUTS,
                  frame_rate=FPS, captures=len(indices), scene=scene.key, studio=asdict(studio),
                  trajectory_sha256=sha256(trajectory.read_bytes()).hexdigest())
    if args.still is None:
        stream = json.loads(subprocess.check_output(["ffprobe","-v","error","-count_frames","-select_streams","v:0",
            "-show_entries","stream=codec_name,width,height,nb_read_frames,avg_frame_rate,duration","-of","json",str(temporary)]))["streams"][0]
        if ((stream["width"],stream["height"]) != SIZE or stream["codec_name"] != "h264"
                or int(stream["nb_read_frames"]) != args.frames or stream["avg_frame_rate"] != f"{FPS}/1"):
            raise RuntimeError(f"unexpected encoded R1 stream: {stream}")
        temporary.rename(movie)
        report.update(video=movie.name, stream=stream, movie_sha256=sha256(movie.read_bytes()).hexdigest())
    write_json(args.output/"render.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-dir",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--cache-dir",type=Path)
    parser.add_argument("--package",type=Path,default=Path(__file__).resolve().parents[3]/"artifacts/rendering/panda-stable-surface/default")
    parser.add_argument("--device-index",type=int,default=1)
    parser.add_argument("--frames",type=int,default=240)
    parser.add_argument("--still",nargs="?",const=0,type=int,help="Render just this storyboard frame as a PNG")
    parser.add_argument("--samples",choices=("grid2","rotated4"),default="rotated4")
    parser.add_argument("--kinematics-only",action="store_true",help="Export the URDF trajectory and cameras without opening a GPU device")
    parser.add_argument("--validation",action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
