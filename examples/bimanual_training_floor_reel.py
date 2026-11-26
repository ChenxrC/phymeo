"""One continuous S4000 Vulkan view of a 16x16 measured-training replay floor.

No physics step, training or image collage is performed here. The native raster
adapter owns one real 3-D scene; only recorded joint poses and the camera move.
"""
import argparse
import ctypes as ct
from dataclasses import replace
from hashlib import sha256
import json
import math
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def digest(path):
    h = sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def smooth(t):
    t = float(np.clip(t, 0., 1.))
    return t*t*t*(10.+t*(-15.+6.*t))


def feature_front_pair(stations):
    """Put two complete demonstrations in the close shot without dropping any trials."""
    placed = list(stations)
    for column,task in ((7,'carry'),(8,'push_t')):
        candidates = [i for i,s in enumerate(placed) if s.clip.task == task and s.clip.success]
        if not candidates:
            continue
        index = min(candidates,key=lambda i:(placed[i].clip.seed,placed[i].clip.job))
        front,source = placed[column],placed[index]
        placed[index] = replace(source,clip=front.clip,phase_s=front.phase_s)
        # The move-in ends at 13 s. Bring both real episodes to their opening
        # grasp/approach as the camera settles on the boundary between tasks.
        phase = (-12.5 * front.playback_rate) % source.clip.duration
        placed[column] = replace(front,clip=source.clip,phase_s=phase)
    return placed


def camera_at(t, floor, width, height):
    center = np.mean(floor.bounds, axis=0).astype(float)
    center[2] = .35
    fov = math.radians(52.)
    direction = np.array([0., -.67, .74]);direction /= np.linalg.norm(direction)
    up_wide = np.cross(direction, [1.,0.,0.])
    corners = np.array([[x,y,z] for x in floor.bounds[:,0]
                        for y in floor.bounds[:,1] for z in floor.bounds[:,2]]) - center
    tan_v = math.tan(fov / 2.)
    framing = np.maximum(abs(corners[:,0]) / (tan_v * width / height),
                         abs(corners @ up_wide) / tan_v)
    distance = float(np.max(corners @ direction + 1.14 * framing)) + .15
    wide = center + direction*distance
    near_target = np.asarray(floor.focus, dtype=float).copy()
    near_target[2] = .18
    near = near_target + np.array([0., -1.58, 1.58])
    amount = smooth((t-4.)/9.) if t < 23. else 1.-smooth((t-23.)/7.)
    eye = wide*(1.-amount)+near*amount
    target = center*(1.-amount)+near_target*amount
    forward = target-eye;forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.array([0.,0.,1.]));right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    view = np.eye(4);view[:3,:3] = np.stack((right,up,-forward))
    view[:3,3] = -view[:3,:3]@eye
    n, f = .05, max(100., distance*3.)
    focal = 1./math.tan(fov*.5)
    projection = np.zeros((4,4))
    projection[0,0] = focal/(width/height);projection[1,1] = -focal
    projection[2,2] = f/(n-f);projection[2,3] = f*n/(n-f);projection[3,2] = -1.
    matrix = np.ascontiguousarray((projection@view).T, dtype=np.float32)
    return matrix, dict(eye_m=eye.tolist(), target_m=target.tolist(), vertical_fov_deg=52.,
                        near_m=n,far_m=f,close_fraction=amount,
                        convention='RH camera -Z, Vulkan depth0..1, negative projection Y; column-major VP')


def executable(name, override=None):
    result = str(override) if override else shutil.which(name)
    if not result or not Path(result).is_file():
        raise FileNotFoundError(f'{name} unavailable; supply its explicit path, no automatic installation')
    return result


def font(path, size):
    if path:
        return ImageFont.truetype(str(path), size)
    try:
        return ImageFont.truetype('DejaVuSans.ttf', size)
    except OSError:
        return ImageFont.load_default()


def caption(image, t, rate, face, chinese):
    result = image.copy()
    draw = ImageDraw.Draw(result, 'RGBA')
    w,h = result.size
    strip = round(h*.061)
    draw.rectangle((0,h-strip,w,h),fill=(20,31,34,240))
    fields = (('左区   双臂搬箱  ×128','右区   协同推 T  ×128','训练轨迹回放')
              if chinese else ('LEFT   CARRY  ×128','RIGHT   PUSH-T  ×128','Training replay'))
    for x,label,color in ((.026,fields[0],(79,218,198)),(.29,fields[1],(255,188,94))):
        draw.rectangle((w*x,h-strip*.70,w*x+4,h-strip*.30),fill=color)
        draw.text((w*x+12,h-strip/2),label,font=face,fill=(238,244,239),anchor='lm')
    draw.text((w*.973,h-strip/2),f'{fields[2]}   {rate:g}×   /   {t:04.1f} s',
              font=face,fill=(213,224,220),anchor='rm')
    return result


def encode(ffmpeg, ffprobe, root, fps, width, height, count):
    movie = root/'training-floor.mp4'
    command = [ffmpeg,'-hide_banner','-loglevel','error','-nostdin','-n','-framerate',str(fps),
               '-i',str(root/'frames/%06d.png'),'-frames:v',str(count),'-an','-c:v','libx264',
               '-preset','medium','-crf','17','-pix_fmt','yuv420p','-movflags','+faststart',str(movie)]
    with (root/'encode.log').open('xb') as log:
        subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT)
    probe_command = [ffprobe,'-v','error','-select_streams','v:0','-show_streams','-show_frames',
                     '-show_entries','stream=codec_name,width,height,avg_frame_rate,nb_frames:frame=best_effort_timestamp_time','-of','json',str(movie)]
    probe = json.loads(subprocess.check_output(probe_command));save(root/'ffprobe.json',probe)
    streams = probe['streams'];frames = probe['frames']
    if len(streams)!=1 or streams[0]['codec_name']!='h264' or (streams[0]['width'],streams[0]['height'])!=(width,height) or len(frames)!=count:
        raise RuntimeError('encoded video codec/extent/frame count differs')
    pts = np.array([float(row['best_effort_timestamp_time']) for row in frames])
    if not np.allclose(pts,np.arange(count)/fps,rtol=0,atol=1e-5):
        raise RuntimeError('encoded presentation timestamps differ from fixed film clock')
    decoder = subprocess.Popen([ffmpeg,'-v','error','-i',str(movie),'-f','rawvideo','-pix_fmt','rgb24','pipe:1'],stdout=subprocess.PIPE,stderr=(root/'decode.log').open('xb'))
    errors = [];decoded_hash = sha256()
    try:
        for i in range(count):
            remaining=width*height*3;parts=[]
            while remaining:
                block=decoder.stdout.read(remaining)
                if not block:raise RuntimeError(f'decoder ended at frame {i}')
                parts.append(block);remaining-=len(block)
            pixels=b''.join(parts);decoded_hash.update(pixels)
            actual=np.frombuffer(pixels,np.uint8).reshape(height,width,3)
            with Image.open(root/'frames'/f'{i:06d}.png') as image:
                expected=np.asarray(image.convert('RGB'),dtype=np.int16)
            difference=np.abs(actual.astype(np.int16)-expected)
            errors.append(dict(frame=i,mae=float(difference.mean()),max_error=int(difference.max())))
        if decoder.stdout.read(1):raise RuntimeError('decoder produced excess frames')
        if decoder.wait()!=0:raise RuntimeError('full video decode failed')
    finally:
        if decoder.poll() is None:decoder.kill();decoder.wait()
        decoder.stdout.close()
    receipt=dict(codec='h264',fps=fps,frames=count,duration_s=count/fps,width=width,height=height,
                 sha256=digest(movie),decoded_rgb_sha256=decoded_hash.hexdigest(),all_frame_errors=errors,
                 mean_mae=float(np.mean([e['mae'] for e in errors])),max_frame_mae=max(e['mae'] for e in errors),
                 validation='all frames decoded and compared with their captioned original PNG; lossy encoding errors reported, not pixel equality')
    save(root/'encoding.json',receipt)
    if receipt['max_frame_mae']>8.:
        raise RuntimeError('encoded frame mean absolute RGB8 error exceeds registered 8-code bound; original receipt retained')
    return receipt

class FloorDevice:
    """Only ABI binding; all rasterization, depth and shading happen in Vulkan."""
    def __init__(self, library, index, width, height):
        self.lib = ct.CDLL(str(library));self.handle = None
        f32=ct.POINTER(ct.c_float);u32=ct.POINTER(ct.c_uint32)
        api = {'floor_create':([ct.c_uint32]*3,ct.c_void_p),
               'floor_upload':([ct.c_void_p,f32,ct.c_uint32,u32,ct.c_uint32,u32,ct.c_uint32,u32,u32,f32,ct.c_uint32],ct.c_int),
               'floor_render':([ct.c_void_p,f32,f32,ct.POINTER(ct.c_uint8)],ct.c_int),
               'floor_info':([ct.c_void_p],ct.c_char_p),'floor_error':([],ct.c_char_p),
               'floor_destroy':([ct.c_void_p],None)}
        for name,(arguments,result) in api.items():
            function=getattr(self.lib,name);function.argtypes=arguments;function.restype=result
        self.handle=self.lib.floor_create(index,width,height)
        if not self.handle:raise RuntimeError(self.error())
        self.width,self.height=width,height
        self.output=np.empty((height,width,4),dtype=np.uint8)
        self.count=None
        try:
            info=self.info()
            if info['backend']!='vulkan-indexed-instanced' or 'S4000' not in info['device_name'].upper():
                raise RuntimeError('this reel requires the actual S4000 Vulkan adapter: '+str(info))
        except BaseException:
            self.close();raise

    def error(self):
        value=self.lib.floor_error()
        return value.decode('utf-8',errors='replace') if value else 'native floor operation failed without error text'

    def info(self):
        value=self.lib.floor_info(self.handle)
        if not value:raise RuntimeError(self.error())
        return json.loads(value)

    def upload(self,floor):
        vertices=np.ascontiguousarray(floor.vertices,dtype=np.float32)
        indices=np.ascontiguousarray(floor.indices,dtype=np.uint32)
        meshes=np.ascontiguousarray(floor.meshes,dtype=np.uint32)
        draws=np.ascontiguousarray(floor.draws,dtype=np.uint32)
        imesh=np.ascontiguousarray(floor.instance_mesh,dtype=np.uint32)
        colors=np.ascontiguousarray(floor.instance_color,dtype=np.float32)
        n=len(imesh)
        if vertices.ndim!=2 or vertices.shape[1]!=6 or meshes.shape[1:]!=(4,) or draws.shape!=(len(meshes),2) or colors.shape!=(n,4):
            raise ValueError('floor geometry ABI shape differs')
        if not np.isfinite(vertices).all() or not np.isfinite(colors).all() or not np.all(colors[:,3]==1.):
            raise ValueError('nonfinite or nonopaque floor geometry')
        if np.any(imesh>=len(meshes)) or int(draws[:,1].sum())!=n:
            raise ValueError('floor draw grouping differs')
        for i,(start,count) in enumerate(draws):
            if not np.all(imesh[int(start):int(start+count)]==i):raise ValueError('instances not grouped by mesh')
        f=lambda a:a.ctypes.data_as(ct.POINTER(ct.c_float))
        u=lambda a:a.ctypes.data_as(ct.POINTER(ct.c_uint32))
        if self.lib.floor_upload(self.handle,f(vertices),len(vertices),u(indices),indices.size,u(meshes),len(meshes),u(draws),u(imesh),f(colors),n):
            raise RuntimeError(self.error())
        self.count=n
        return {name:dict(shape=list(a.shape),dtype=a.dtype.str,sha256=sha256(a.tobytes()).hexdigest())
                for name,a in dict(vertices=vertices,indices=indices,meshes=meshes,draws=draws,instance_mesh=imesh,instance_color=colors,
                                   instance_id=np.ascontiguousarray(floor.instance_id,dtype=np.uint32)).items()}

    def render(self,matrices,viewproj):
        matrices=np.asarray(matrices)
        if matrices.shape!=(self.count,4,4) or not np.isfinite(matrices).all():raise ValueError('invalid current instance matrices')
        model=np.ascontiguousarray(matrices.transpose(0,2,1),dtype=np.float32)
        vp=np.ascontiguousarray(viewproj,dtype=np.float32)
        if not np.isfinite(model).all() or vp.shape!=(4,4) or not np.isfinite(vp).all():raise ValueError('invalid FP32 camera/model transport')
        if self.lib.floor_render(self.handle,model.ctypes.data_as(ct.POINTER(ct.c_float)),vp.ctypes.data_as(ct.POINTER(ct.c_float)),
                                 self.output.ctypes.data_as(ct.POINTER(ct.c_uint8))):raise RuntimeError(self.error())
        return self.output.copy(),sha256(model.tobytes()).hexdigest()

    def close(self):
        if self.handle:
            self.lib.floor_destroy(self.handle);self.handle=None


def run(args):
    import _bimanual_floor_motion as motion
    import _bimanual_floor_scene as geometry
    if not args.preview_only:
        ffmpeg=executable('ffmpeg',args.ffmpeg);ffprobe=executable('ffprobe',args.ffprobe)
    if args.device_index<0:
        raise ValueError('device index must be nonnegative')
    if args.width<1280 or args.height<720 or args.width%2 or args.height%2:
        raise ValueError('use an even, at least 1280x720 output extent')
    if not 0<args.playback_rate<=1. or not 1<=args.frames<=720:
        raise ValueError('playback-rate must be (0,1] and frames 1..720')
    root=args.out.resolve();root.mkdir(parents=True,exist_ok=False)
    (root/'frames').mkdir();(root/'stills').mkdir()
    report=dict(schema='s4000-bimanual-floor-reel-v1',status='preparing',rows=16,columns=16,stations=256,arms=512,
                tasks={'carry':128,'push_t':128},fps=24,planned_duration_s=30.,requested_frames=args.frames,
                preview_only=args.preview_only,source_playback_rate=args.playback_rate,
                scope='one actual 3D raster scene of archived episodes; no physics steps or simultaneous training claim',
                renderer_scope='dedicated headless Vulkan indexed-instanced raster, depth buffer and real transforms; not StudioRenderer or path tracing',
                motion_input=dict(path=str(args.clips.resolve()),sha256=digest(args.clips)),
                source_sha256={str(Path(p).resolve()):digest(p) for p in (__file__,motion.__file__,geometry.__file__)},
                library=dict(path=str(args.library.resolve()),sha256=digest(args.library)),
                shaders={p.name:digest(p) for p in args.library.parent.glob('*.spv')},
                size=[args.width,args.height],camera='single perspective, wide 0-4s; approach4-13s; close13-23s; retreat23-30s',
                completed_frames=0,device=None)
    save(root/'summary.json',report)
    owner=None;timings=[]
    try:
        clips=motion.load_clips(args.clips)
        stations=feature_front_pair(motion.build_stations(
            clips,rows=16,columns=16,playback_rate=args.playback_rate,sampled=True))
        if len(stations)!=256 or sum(s.clip.task=='carry' for s in stations)!=128 or sum(s.clip.task=='push_t' for s in stations)!=128:
            raise ValueError('floor must contain the fixed two-task station split')
        manifest = motion.floor_manifest(stations)
        if not args.preview_only:
            if manifest['unique_episodes'] != 256 or manifest['unique_by_stage'] != {'training-candidate': 256}:
                raise ValueError('the training film requires 256 distinct captured training episodes')
            if not all(s.get('completed_generation_checkpoint_sha256') for s in manifest['sources']):
                raise ValueError('a training episode has no completed optimizer generation')
        save(root/'motion-manifest.json',manifest)
        report['unique_episodes'] = manifest['unique_episodes']
        report['source_stages'] = manifest['unique_by_stage']
        report['featured_stations'] = [
            dict(row=s.row,col=s.col,task=s.clip.task,success=s.clip.success,source=s.clip.motion_id)
            for s in (stations[7],stations[8])]
        start=time.perf_counter();floor=geometry.build_floor(stations)
        report['geometry_prepare_s']=time.perf_counter()-start
        owner=FloorDevice(args.library,args.device_index,args.width,args.height)
        report['geometry']=owner.upload(floor);report['device']=owner.info()
        report['bounds_m']=np.asarray(floor.bounds).tolist();report['focus_m']=np.asarray(floor.focus).tolist()
        save(root/'summary.json',report)
        face=font(args.font, round(args.height*.026))
        indices=(0,360,719) if args.preview_only else range(args.frames)
        previous_indices=None
        with (root/'frames.jsonl').open('x',encoding='utf-8') as journal:
            for index in indices:
                t=index/24.;begin=time.perf_counter()
                matrices=floor.update(t);fk_end=time.perf_counter()
                vp,camera=camera_at(t,floor,args.width,args.height)
                before=time.perf_counter();pixels,matrix_sha=owner.render(matrices,vp)
                render_end=time.perf_counter()
                image=Image.fromarray(pixels[:,:,:3],'RGB')
                if index in (0,360,719):
                    image.save(root/'stills'/f'{index:06d}-raw.png')
                presented=caption(image,t,args.playback_rate,face,args.chinese_labels)
                path=root/'frames'/f'{index:06d}.png';presented.save(path)
                if index in (0,360,719):presented.save(root/'stills'/f'{index:06d}.png')
                source_indices=[int(s.sample_index(t)) for s in stations]
                resets=[] if previous_indices is None else [i for i,(a,b) in enumerate(zip(previous_indices,source_indices)) if b<a]
                row=dict(index=index,pts_s=t,source_sample_indices=source_indices,source_episode_wrap_stations=resets,
                         camera=camera,model_transport_sha256=matrix_sha,viewproj_transport_sha256=sha256(vp.tobytes()).hexdigest(),
                         raw_rgba_sha256=sha256(pixels.tobytes()).hexdigest(),png_sha256=digest(path),
                         fk_ms=1000*(fk_end-begin),capture_ms=1000*(render_end-before),
                         image_scope='original one-view Vulkan RGB plus caption; no spatial collage')
                journal.write(json.dumps(row,allow_nan=False,separators=(',',':'))+'\n');journal.flush()
                timings.append(row['capture_ms']);previous_indices=source_indices
                report['completed_frames']+=1
                if report['completed_frames']%24==0:
                    save(root/'summary.json',report)
                    print(json.dumps(dict(completed=report['completed_frames'],last_index=index,capture_ms=row['capture_ms'])),flush=True)
        report['device_after']=owner.info()
        report['capture_ms']={'mean':float(np.mean(timings)),'p50':float(np.percentile(timings,50)),'p95':float(np.percentile(timings,95))}
        report['journal_sha256']=digest(root/'frames.jsonl')
        report['motion_manifest_sha256']=digest(root/'motion-manifest.json')
        owner.close();owner=None
        report['close']='floor_destroy returned; native ABI does not expose post-destroy allocation counts'
        if not args.preview_only:
            report['encoding']=encode(ffmpeg,ffprobe,root,24,args.width,args.height,args.frames)
        report['status']='preview_complete' if args.preview_only else ('complete' if args.frames==720 else 'partial_duration_complete')
    except BaseException as error:
        report.update(status='failed',error={'type':type(error).__name__,'message':str(error)})
        raise
    finally:
        if owner is not None:
            try:
                report['device_at_close']=owner.info()
            except BaseException as error:
                report['close_info_error']=str(error)
            finally:owner.close()
        save(root/'summary.json',report)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clips',type=Path,required=True,help='verified two-task floor input manifest')
    parser.add_argument('--library',type=Path,required=True,help='parent-built dedicated instanced Vulkan library')
    parser.add_argument('--chinese-labels',action='store_true',help='use with a CJK font supplied via --font')
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--device-index',type=int,default=0)
    parser.add_argument('--width',type=int,default=1280)
    parser.add_argument('--height',type=int,default=720)
    parser.add_argument('--frames',type=int,default=720,help='first N frames of unchanged 30-second camera clock')
    parser.add_argument('--preview-only',action='store_true',help='same scene/camera at indices0,360,719; no encoding')
    parser.add_argument('--playback-rate',type=float,default=.25)
    parser.add_argument('--ffmpeg',type=Path)
    parser.add_argument('--ffprobe',type=Path)
    parser.add_argument('--font',type=Path)
    run(parser.parse_args())


if __name__=='__main__':
    main()
