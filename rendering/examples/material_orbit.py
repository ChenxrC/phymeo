"""A fixed material study photographed through a moving public-session camera.

No physics is advanced. Three spheres share exactly one mesh; the wood specimen
is a separate finite UV board. Images compare appearance, not calibrated BRDF
measurements: the specimens occupy different positions in the fixed light field.
"""
import argparse
from dataclasses import asdict
from hashlib import sha256
import json
import math
from pathlib import Path
import subprocess
from time import perf_counter

import numpy as np

from _panda_scene import box, look_at, pose_matrix, write_mesh
from phymeo_rendering.assets import Material, TextureChannel, TextureMap, TexturedMaterial
from phymeo_rendering.cameras import Camera
from phymeo_rendering.mesh_data import load_mesh
from phymeo_rendering.outputs import Channel
from phymeo_rendering.pipeline import CaptureRequest, RenderPath
from phymeo_rendering.scene import FrameKey, Instance, NodePose, Pose, SceneFrame, VisualScene

ROOT = Path(__file__).resolve().parent
SOURCES = {
    'wood_table_001_diff_1k.jpg': '460dd08d240f4a1f02982415048c3c5c200f385db212da18dc7e2df68bf4d0be',
    'wood_table_001_rough_1k.jpg': 'a2ca9ae89dfe7b8e2e5e92f0aa43467845417de8a50205b5852ef0d57c7624f9',
    'university_workshop_1k.hdr': 'a125ada38fc602025326f29b68df09531bb16f899f1973b03d9ac5ac98142548',
}
CENTRES = {1: (-.42, 0., .11), 2: (-.14, 0., .11), 3: (.14, 0., .11), 4: (.45, .02, .15)}
LABELS = {1: 'Mirror steel | r = 0.045', 2: 'Rough steel | r = 0.45',
          3: 'Matte rubber | r = 0.85', 4: 'Scanned wood | color + roughness'}


def verify_sources(directory=ROOT/'materials'):
    evidence = {}
    for name, expected in SOURCES.items():
        path = Path(directory)/name
        actual = sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f'SHA256 mismatch for {name}: {actual}')
        evidence[name] = {'path': str(path.resolve()), 'sha256': actual, 'bytes': path.stat().st_size}
    return evidence


def sphere_mesh(radius=.11, rings=48, segments=96):
    """Closed outward sphere with shared poles, no zero-area cap triangles."""
    normals = [(0., 0., 1.)]
    for ring in range(1, rings):
        polar = math.pi*ring/rings
        for index in range(segments):
            azimuth = 2*math.pi*index/segments
            normals.append((math.sin(polar)*math.cos(azimuth),
                            math.sin(polar)*math.sin(azimuth), math.cos(polar)))
    bottom = len(normals)
    normals.append((0., 0., -1.))
    triangles = []
    for index in range(segments):
        nxt = (index+1) % segments
        triangles.append((0, 1+index, 1+nxt))
        for ring in range(rings-2):
            start = 1+ring*segments
            triangles.extend(((start+index, start+segments+index, start+nxt),
                              (start+nxt, start+segments+index, start+segments+nxt)))
        start = 1+(rings-2)*segments
        triangles.append((start+index, bottom, start+nxt))
    return tuple(tuple(radius*v for v in n) for n in normals), tuple(normals), tuple(triangles)


def build_scene(directory, *, materials_directory=ROOT/'materials'):
    verify_sources(materials_directory)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    sphere = write_mesh(directory, 'shared-sphere', *sphere_mesh())
    board_plain = box(directory, 'board-geometry', (.25, .06, .30))
    data = load_mesh(board_plain)
    uv = []
    for (x,y,z), (nx,ny,nz) in zip(data.positions,data.normals,strict=True):
        horizontal, vertical = (x,y) if abs(nz)>.5 else ((y,z) if abs(nx)>.5 else (x,z))
        uv.append((.5+horizontal/1.5, .5-vertical/1.5))
    board = write_mesh(directory, 'scanned-board', data.positions,data.normals,data.triangles,texcoords=uv)
    stage = box(directory, 'stage', (1.35, .66, .05))
    texture = lambda name: TextureMap((Path(materials_directory)/name).resolve().as_uri(), SOURCES[name])
    materials = (
        Material('mirror-steel', (.56,.58,.60), 1., .045),
        Material('rough-steel', (.56,.58,.60), 1., .45),
        Material('matte-rubber', (.035,.041,.045), 0., .85),
        TexturedMaterial('scanned-wood', (1.,1.,1.), 0., 1.,
                         texture('wood_table_001_diff_1k.jpg'),
                         TextureChannel(texture('wood_table_001_rough_1k.jpg'),'r')),
        Material('neutral-stage', (.14,.16,.18), 0., .65),
    )
    instances = tuple(Instance(i, sphere.id if i<4 else board.id, materials[i-1].id, f'specimen-{i}')
                      for i in (1,2,3,4)) + (Instance(5,stage.id,materials[4].id,'stage'),)
    nodes = tuple(NodePose(f'specimen-{i}',Pose(CENTRES[i])) for i in (1,2,3,4)) + (
        NodePose('stage',Pose((0.,0.,-.025))),)
    return VisualScene('authored-material-orbit-v1',(sphere,board,stage),materials,instances), nodes


def exposures(scene, nodes, *, fps=24, width=1280, height=720):
    if type(fps) is not int:
        raise TypeError('fps must be an integer')
    if fps not in (24,30):
        raise ValueError('this bounded eight-second study supports 24 or 30 fps')
    if (type(width) is not int or type(height) is not int or not 2<=width<=1920
            or not 2<=height<=1080 or width%2 or height%2):
        raise ValueError('expected positive even dimensions up to 1920x1080')
    result = []
    count = 8*fps
    for index in range(count):
        phase = index/(count-1)
        eased = phase*phase*(3-2*phase)
        angle = math.radians(-24+48*eased)
        eye = (1.65*math.sin(angle), -1.65*math.cos(angle), .73)
        frame = SceneFrame(FrameKey(scene.key,'authored-materials',0,0,index,'authored'),index/fps,nodes)
        focal = 1180*height/720
        camera = Camera('material-orbit',frame.key,look_at(eye,(0.,0.,.13),(0,0,1)),
                        width,height,focal,focal,width/2,height/2,.01,10.)
        result.append((frame,camera))
    return tuple(result)


def scout_indices(count):
    return (0,count//2,count-1)


def project(camera, point):
    transform = pose_matrix(camera.world_pose)
    local = transform[:3,:3].T@(np.asarray(point)-transform[:3,3])
    depth = -local[2]
    return (float(camera.cx_px+camera.fx_px*local[0]/depth),
            float(camera.cy_px-camera.fy_px*local[1]/depth),float(depth))


def _json(path, value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def _observe(capture):
    planes = {plane.channel: plane.data for plane in capture.planes}
    h,w = capture.camera.height,capture.camera.width
    rgb = np.frombuffer(planes[Channel.RGB],np.uint8).reshape(h,w,3)
    ids = np.frombuffer(planes[Channel.INSTANCE_ID],'<u4').reshape(h,w)
    depth = np.frombuffer(planes[Channel.FORWARD_DEPTH],'<f4').reshape(h,w)
    visible = np.frombuffer(planes[Channel.VALID],np.uint8).reshape(h,w)
    metrics = {}
    for identity in (1,2,3,4):
        mask = ids==identity
        yy,xx = np.nonzero(mask)
        if len(xx)<200 or not np.isfinite(depth[mask]).all() or not (visible[mask]==1).all():
            raise ValueError(f'specimen {identity} lacks the predeclared visible finite-depth support')
        values = rgb[mask].astype(np.float64)
        centre = project(capture.camera,CENTRES[identity])
        px,py = int(centre[0]),int(centre[1])
        if not (0<=px<w and 0<=py<h and ids[py,px]==identity):
            raise ValueError(f'specimen {identity} projected centre is occluded or outside the image')
        metrics[str(identity)] = dict(label=LABELS[identity],visible_pixels=int(len(xx)),
            bbox_xyxy=[int(xx.min()),int(yy.min()),int(xx.max()),int(yy.max())],
            projected_centre_xyz=list(centre),mean_srgb8=values.mean(axis=0).tolist(),
            p95_max_srgb8=float(np.percentile(values.max(axis=1),95)),
            fraction_any_channel_255=float(np.mean(values.max(axis=1)==255)),
            forward_depth_range_m=[float(depth[mask].min()),float(depth[mask].max())])
    return metrics, planes


def run(args):
    from PIL import Image
    from phymeo_rendering.backends.native import NativeBackend
    from phymeo_rendering.environment import RadianceEnvironment
    from phymeo_rendering.studio import StudioLighting

    args.output.mkdir(parents=True,exist_ok=False)
    report = dict(schema='phymeo-material-orbit-v1',title='材质观察·相机绕拍',status='preparing',
        script_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        physics_steps=0,source='authored fixed scene; no Panda or physical trajectory replay',
        motion='camera only; geometry, lights, material parameters and exposure fixed',
        comparison_limit='same sphere mesh, different positions in a fixed light field; not a calibrated BRDF ranking',
        fps=args.fps,encoded_duration_s=8,frame_times='i/fps for i=0..8*fps-1; last image held for one frame',
        planned_frames=3 if args.scout else 8*args.fps,accepted_captures=0,records=[],
        inputs={},exposure=8.,display_transform='pbr-neutral',
        shadow_mode=args.shadow_mode,area_light_grid=args.area_light_grid,
        scout_acceptance='each specimen >=200 visible pixels, finite depth, VALID=1, own projected centre ID',
        measurements='display sRGB8 appearance statistics; no linear radiance or universal quality claim')
    session = None
    started = perf_counter()
    try:
        report['inputs'] = verify_sources()
        scene,nodes = build_scene(args.output/'geometry')
        sequence = exposures(scene,nodes,fps=args.fps,width=args.width,height=args.height)
        report.update(scene=asdict(scene),fixed_nodes=[asdict(node) for node in nodes],
            mesh_sha256={mesh.id:load_mesh(mesh).source_sha256 for mesh in scene.meshes},
            scout_indices=list(scout_indices(len(sequence))),
            camera_schedule=[asdict(camera) for frame,camera in sequence],
            library={'path':str(args.library.resolve()),'sha256':sha256(args.library.read_bytes()).hexdigest()})
        environment = RadianceEnvironment((ROOT/'materials/university_workshop_1k.hdr').as_uri(),
            SOURCES['university_workshop_1k.hdr'],strength=.0625,yaw_degrees=25.,face_size=256)
        studio = StudioLighting(center_m=(0.,0.,.13),scale_m=.85,panel_strength=.125,
                                 environment_map=environment)
        report['studio'] = asdict(studio)
        backend = NativeBackend(library_path=args.library,shader_directory=args.shader_directory,
            device_index=args.device_index,resident=True,validation=args.validation,
            studio=studio,exposure=8.,display_transform='pbr-neutral',shadow_mode=args.shadow_mode,
            area_light_grid=args.area_light_grid)
        opened = perf_counter()
        session = backend.open_scene(scene)
        report['open_ms'] = 1000*(perf_counter()-opened)
        directory = args.output/'frames'
        directory.mkdir()
        indices = scout_indices(len(sequence)) if args.scout else range(len(sequence))
        journal = args.output/'frames.jsonl'
        with journal.open('w',encoding='utf-8',newline='\n') as stream:
            for index in indices:
                frame,camera = sequence[index]
                channels = tuple(Channel) if index in scout_indices(len(sequence)) else (Channel.RGB,)
                before = perf_counter()
                captured = session.capture(frame,CaptureRequest(camera,RenderPath.RASTER,channels))
                elapsed = 1000*(perf_counter()-before)
                report['accepted_captures'] += 1
                if captured.frame_key != frame.key or captured.time_s != frame.time_s or captured.camera != camera:
                    raise ValueError('capture changed the submitted source or camera')
                planes = {plane.channel:plane.data for plane in captured.planes}
                image_file = directory/f'{index:04d}.png'
                Image.frombytes('RGB',(args.width,args.height),planes[Channel.RGB]).save(image_file)
                row = dict(index=index,time_s=frame.time_s,frame_key=asdict(frame.key),
                    camera=asdict(camera),capture_wall_ms=elapsed,file=str(image_file.relative_to(args.output)),
                    png_sha256=sha256(image_file.read_bytes()).hexdigest(),
                    plane_sha256={channel.value:sha256(data).hexdigest() for channel,data in planes.items()})
                if len(channels)>1:
                    metrics,planes = _observe(captured)
                    row['specimens'] = metrics
                    sensor_path = args.output/f'scout-{index:04d}.npz'
                    np.savez_compressed(sensor_path,**{channel.name:np.frombuffer(data,np.uint8)
                                                       for channel,data in planes.items()})
                    row['sensor_file'] = sensor_path.name
                    row['sensor_sha256'] = sha256(sensor_path.read_bytes()).hexdigest()
                stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
                stream.flush()
                report['records'].append(row)
                print(json.dumps({'index':index,'captured':report['accepted_captures'],'capture_ms':elapsed}),flush=True)
        report['journal'] = {'file':journal.name,'sha256':sha256(journal.read_bytes()).hexdigest()}
        report['status'] = 'scout_complete' if args.scout else 'complete'
    except BaseException as error:
        report['status'] = 'failed'
        report['error'] = {'type':type(error).__name__,'message':str(error)}
        raise
    finally:
        close_error = None
        if session is not None:
            try:
                session.close()
            except Exception as error:
                close_error = error
                report['status'] = 'close_failed'
                report['close_error'] = {'type':type(error).__name__,'message':str(error)}
            try:
                report['diagnostics_after_close'] = session.diagnostics()
                device = report['diagnostics_after_close']['device']
                if not device['closed'] or device['validation_errors']:
                    raise ValueError('native teardown or validation diagnostic failed')
                telemetry = device.get('resident_telemetry')
                if telemetry is not None and (not telemetry['final_after_destroy']
                        or telemetry['device_local_bytes_current'] or telemetry['host_visible_bytes_current']
                        or telemetry['allocation_count'] != telemetry['free_count']):
                    raise ValueError('native final memory allocation ledger is not closed')
            except Exception as error:
                report['diagnostics_error'] = {'type':type(error).__name__,'message':str(error)}
                report['status'] = 'diagnostics_failed'
                close_error = close_error or error
        report['total_wall_s'] = perf_counter()-started
        _json(args.output/'summary.json',report)
        if close_error is not None:
            raise close_error
    if args.ffmpeg and not args.scout:
        movie = args.output/'material-orbit.mp4'
        command = [str(args.ffmpeg),'-hide_banner','-loglevel','error','-n','-framerate',str(args.fps),
            '-i',str(args.output/'frames/%04d.png'),'-frames:v',str(8*args.fps),
            '-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(movie)]
        try:
            subprocess.run(command,check=True)
            report['movie'] = {'file':movie.name,'sha256':sha256(movie.read_bytes()).hexdigest(),'command':command}
        except Exception as error:
            report['encoding_error'] = {'type':type(error).__name__,'message':str(error),'command':command}
            raise
        finally:
            _json(args.output/'summary.json',report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library',type=Path,required=True)
    parser.add_argument('--shader-directory',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device-index',type=int,default=1)
    parser.add_argument('--shadow-mode',choices=('software','ray_query'),default='ray_query')
    parser.add_argument('--area-light-grid',type=int,choices=(2,4,8),default=8)
    parser.add_argument('--width',type=int,default=1280)
    parser.add_argument('--height',type=int,default=720)
    parser.add_argument('--fps',type=int,choices=(24,30),default=24)
    parser.add_argument('--scout',action='store_true')
    parser.add_argument('--validation',action='store_true')
    parser.add_argument('--ffmpeg',type=Path)
    run(parser.parse_args())


if __name__ == '__main__':
    main()



