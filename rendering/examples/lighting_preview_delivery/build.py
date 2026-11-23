"""Build an evidence-bound local comparison page from two real 300-frame videos.

No renderer is invoked. ffprobe decodes frame timestamps; ffmpeg extracts only
registered frames. Pillow preserves each ROI's original pixel dimensions.
"""
import argparse
from fractions import Fraction
from hashlib import sha256
import html
import json
import math
from pathlib import Path
import subprocess
import traceback


def digest(path):return sha256(Path(path).read_bytes()).hexdigest()
def write_json(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def pointer(document,path):
    if not isinstance(path,str) or not path.startswith('/'):raise ValueError('explicit JSON pointer required')
    value=document
    try:
        for token in path[1:].split('/'):
            token=token.replace('~1','/').replace('~0','~')
            value=value[int(token)] if isinstance(value,list) else value[token]
    except (KeyError,IndexError,TypeError,ValueError) as error:raise ValueError('missing receipt pointer: '+path) from error
    return value


def receipt_values(receipt,fields):
    result={k:pointer(receipt,fields[k]) for k in ('profile','rgb_mean_ms','status')}
    if not isinstance(result['profile'],str) or not result['profile']:raise ValueError('actual named profile required')
    if type(result['rgb_mean_ms']) not in (float,int) or not math.isfinite(result['rgb_mean_ms']) or result['rgb_mean_ms']<=0:
        raise ValueError('finite positive measured RGB mean ms required')
    if type(result['status']) not in (str,bool):raise ValueError('explicit receipt status required')
    return result


def require_same_trajectory(actual,receipt_hashes):
    if len(receipt_hashes)!=2 or not isinstance(actual,str) or len(actual)!=64 or any(x!=actual for x in receipt_hashes):
        raise ValueError('both receipts must bind the exact same actual trajectory SHA256')
    return actual


def crop_box(box,width,height):
    if not isinstance(box,(list,tuple)) or len(box)!=4 or any(type(v) is not int for v in box):raise ValueError('integer xyxy crop required')
    x0,y0,x1,y1=box
    if not (0<=x0<x1<=width and 0<=y0<y1<=height):raise ValueError('crop is empty or outside actual pixels')
    return tuple(box)


def video_contract(data):
    streams=data.get('streams',[])
    if len(streams)!=1:raise ValueError('exactly one selected video stream required')
    s=streams[0];frames=data.get('frames',[])
    if s.get('codec_name')!='h264' or Fraction(s['avg_frame_rate'])!=30 or len(frames)!=300:
        raise ValueError('require real H264 / 30 FPS / exactly 300 decoded frames')
    times=[float(f['best_effort_timestamp_time']) for f in frames]
    if any(not math.isfinite(t) or abs(t-i/30)>1e-5 for i,t in enumerate(times)):
        raise ValueError('decoded PTS differs from registered zero-origin f/30')
    if abs(float(s['duration'])-10)>1e-3:raise ValueError('video duration is not 10 seconds')
    w,h=s['width'],s['height']
    if type(w) is not int or type(h) is not int or min(w,h)<=0:raise ValueError('invalid dimensions')
    return dict(width=w,height=h,frames=300,fps=30,duration_s=float(s['duration']),pts_s=times,codec='h264')


def image_assertion(example,decoded,limit):
    from PIL import ImageChops,ImageStat
    if type(limit) not in (int,float) or not math.isfinite(limit) or limit<0:raise ValueError('explicit finite PNG-to-decode RGB8 MAE limit required')
    if example.size!=decoded.size:raise ValueError('example/decode dimensions disagree')
    difference=ImageChops.difference(example.convert('RGB'),decoded.convert('RGB'))
    mae=sum(ImageStat.Stat(difference).mean)/3
    maximum=max(high for low,high in difference.getextrema())
    if mae>limit:raise ValueError(f'example PNG differs: RGB8 MAE {mae} > {limit}; max {maximum}')
    return dict(rgb8_mae=mae,rgb8_max_error=maximum,limit=limit,passed=True)


def page(data):
    cards=[]
    for lane in data['lanes']:
        cards.append('<section><h2>'+html.escape(lane['label'])+'</h2><p>'+html.escape(lane['profile'])+
            '</p><p>RGB render mean: <b>'+html.escape(str(lane['rgb_mean_ms']))+' ms</b> · Receipt status: <b>'+html.escape(str(lane['status']))+
            '</b></p><canvas aria-label="Decoded same-frame comparison"></canvas><video muted playsinline preload="auto" controls src="'+html.escape(lane['video'],quote=True)+
            '"></video><p><a href="'+html.escape(lane['video'],quote=True)+'">Open original video / 原片</a></p></section>')
    crops=[]
    for frame in data['frames']:
        rows=[]
        for roi in frame['rois']:
            images=''.join('<figure><figcaption>'+html.escape(x['label'])+' · '+str(x['width'])+' × '+str(x['height'])+' px</figcaption><img src="'+html.escape(x['file'],quote=True)+'" alt="'+html.escape(roi['label'],quote=True)+'"></figure>' for x in roi['lanes'])
            rows.append('<h4>'+html.escape(roi['label'])+'</h4><div class="crop">'+images+'</div>')
        crops.append('<details open><summary>Frame '+str(frame['index'])+' · motion '+format(frame['source_time_s'],'.6f')+' s</summary>'+''.join(rows)+'</details>')
    payload=json.dumps(data,ensure_ascii=False).replace('<','\\u003c')
    return '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Lighting comparison · measured frames</title><style>
body{margin:0;background:#11151b;color:#e7edf3;font:16px system-ui,sans-serif}main{max-width:1500px;margin:auto;padding:24px}h1{font-size:28px}p{line-height:1.5;color:#bdc9d5}a{color:#93c7ff}.pair{display:grid;grid-template-columns:1fr 1fr;gap:20px}canvas,video{width:100%;height:auto;background:#050708}canvas{display:none}section{min-width:0}button,input{font:inherit}button{padding:8px 12px;background:#283b51;color:white;border:1px solid #527091;border-radius:6px}input[type=range]{width:min(600px,95%)}.controls{position:sticky;top:0;background:#19222ef5;padding:12px;z-index:2}.crop{display:flex;gap:20px;overflow:auto}.crop img{max-width:none;width:auto;image-rendering:auto}figure{margin:0}figcaption{margin:8px 0}details{padding:16px 0;border-top:1px solid #3a495c}#state{font-variant-numeric:tabular-nums}@media(max-width:720px){main{padding:12px}.pair{grid-template-columns:1fr}h1{font-size:23px}}</style>
<main><h1>Lighting · same-trajectory comparison / 同轨迹光照对照</h1>
<p>30 FPS playback · 10 s / 300 frames. Playback speed is separate from measured RGB render cost. 下面裁剪保留原像素尺寸；影片画面可能受有损压缩影响。Status is copied from the named receipt, not inferred from appearance.</p>
<div class="controls"><button id="back">◀ Frame</button> <button id="play">Play / Pause</button> <button id="next">Frame ▶</button><br><input id="seek" aria-label="Frame index" type="range" min="0" max="299" value="0"><span id="state">Loading…</span><br><label>Motion time / 运动时刻 <input id="motion" type="number" step="0.001"><button id="jump">Go</button></label></div>
<div class="pair">'''+''.join(cards)+'''</div><h2>Registered original-size crops / 登记原尺寸裁剪</h2>'''+''.join(crops)+'''
<p><a href="checks.json">Media, receipt and pixel checks / 核验收据</a></p></main>
<script>const data='''+payload+''';
const videos=[...document.querySelectorAll('video')],canvases=[...document.querySelectorAll('canvas')],state=document.querySelector('#state'),slider=document.querySelector('#seek');
let frame=0,playing=false,busy=false,start=0,startFrame=0;const cached=new WeakMap();
const ready=v=>v.error?Promise.reject(Error('Video unavailable: use original link')):v.readyState>=2?Promise.resolve():new Promise((ok,no)=>{v.addEventListener('loadeddata',ok,{once:true});v.addEventListener('error',()=>no(Error('Video unavailable: use original link')), {once:true});});
function decoded(v,index){if(cached.get(v)?.index===index&&Math.floor(v.currentTime*30)===index&&v.paused)return Promise.resolve(cached.get(v).verified);return new Promise((resolve,reject)=>{const target=(index+.25)/30;let done=false,callback;
const timer=setTimeout(()=>{if(!done){done=true;if(callback)v.cancelVideoFrameCallback(callback);reject(Error('Decode timeout; original videos remain available'));}},8000);
const finish=t=>{if(done)return;done=true;clearTimeout(timer);cached.set(v,{index,verified:t});resolve(t)};
if(v.requestVideoFrameCallback){callback=v.requestVideoFrameCallback((now,meta)=>finish(Math.round(meta.mediaTime*30)));}
else v.addEventListener('seeked',()=>requestAnimationFrame(()=>requestAnimationFrame(()=>finish(null))),{once:true});
v.currentTime=target;});}
async function show(index){if(busy)return;busy=true;try{await Promise.all(videos.map(ready));const actual=await Promise.all(videos.map(v=>decoded(v,index)));
if(actual.some(x=>x!==null&&x!==index))throw Error('Decoded frame mismatch '+actual+'; not displayed as matched');
videos.forEach((v,i)=>{const c=canvases[i];c.width=v.videoWidth;c.height=v.videoHeight;c.getContext('2d').drawImage(v,0,0);c.style.display='block';v.style.cssText='position:absolute;width:1px;height:1px;opacity:.01;pointer-events:none'});
frame=index;slider.value=index;state.textContent=` Frame ${index}/299 · PTS ${(index/30).toFixed(3)} s · motion ${data.source_times_s[index].toFixed(6)} s`+(actual.includes(null)?' · seeked fallback (no decoded-PTS API)':' · both decoded PTS verified');
}catch(e){playing=false;state.textContent=e.message;videos.forEach(v=>v.style.cssText='');canvases.forEach(c=>c.style.display='none')}finally{busy=false}}
function stop(){playing=false}document.querySelector('#back').onclick=()=>{stop();show(Math.max(0,frame-1))};document.querySelector('#next').onclick=()=>{stop();show(Math.min(299,frame+1))};slider.oninput=()=>{stop();show(+slider.value)};
document.querySelector('#play').onclick=()=>{playing=!playing;start=performance.now();startFrame=frame;if(playing)requestAnimationFrame(tick)};
function tick(now){if(!playing)return;const wanted=Math.min(299,startFrame+Math.floor((now-start)*30/1000));if(!busy&&wanted!==frame)show(wanted);if(wanted===299&&!busy&&frame===299)playing=false;else requestAnimationFrame(tick)}
document.querySelector('#jump').onclick=()=>{stop();const t=+document.querySelector('#motion').value;let nearest=0;data.source_times_s.forEach((x,i)=>{if(Math.abs(x-t)<Math.abs(data.source_times_s[nearest]-t))nearest=i});show(nearest)};
window.addEventListener('keydown',e=>{if(['INPUT','TEXTAREA'].includes(document.activeElement.tagName))return;if(e.key==='ArrowLeft')document.querySelector('#back').click();if(e.key==='ArrowRight')document.querySelector('#next').click();if(e.code==='Space'){e.preventDefault();document.querySelector('#play').click()}});
if(canvases.every(c=>c.getContext&&c.getContext('2d')))show(0);else state.textContent='Canvas unavailable; original video controls below';
</script></html>'''


def build(manifest_path,output,*,ffmpeg='ffmpeg',ffprobe='ffprobe'):
    from PIL import Image
    manifest_path=Path(manifest_path).resolve();base=manifest_path.parent
    specification=json.loads(manifest_path.read_text(encoding='utf-8-sig'))
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    checks=dict(schema='phymeo-lighting-preview-delivery-v1',status='running',manifest_path=str(manifest_path),
        manifest_sha256=digest(manifest_path),tool_sha256=digest(__file__),lanes=[],frames=[],commands=[])
    def resolve(value):return (base/Path(value)).resolve()
    def run(command,log):
        result=subprocess.run(command,capture_output=True,text=True)
        checks['commands'].append(dict(command=command,returncode=result.returncode,log=log))
        (output/log).write_text(result.stderr,encoding='utf-8')
        if result.returncode:raise ValueError('media command failed; preserved '+log)
        return result.stdout
    try:
        lanes=specification['lanes'];indices=specification['frames'];rois=specification['rois']
        if len(lanes)!=2 or not indices or len(set(indices))!=len(indices) or any(type(i) is not int or not 0<=i<300 for i in indices):
            raise ValueError('two lanes and unique actual frame indices in 0..299 required')
        if not rois:raise ValueError('registered ROI list required')
        trajectory=resolve(specification['trajectory_file']);trajectory_hash=digest(trajectory)
        trace=json.loads(trajectory.read_text(encoding='utf-8-sig'))
        times=pointer(trace,specification.get('source_times_pointer','/source_times_s'))
        if not isinstance(times,list) or len(times)!=300 or any(type(t) not in (int,float) or not math.isfinite(t) for t in times) or any(a>=b for a,b in zip(times,times[1:])):
            raise ValueError('300 strictly increasing actual motion times required')
        checks['trajectory']=dict(path=str(trajectory),sha256=trajectory_hash,source_times_pointer=specification.get('source_times_pointer','/source_times_s'))
        hashes=[];decoded={};page_lanes=[]
        for number,lane in enumerate(lanes):
            video=resolve(lane['video']);receipt_path=resolve(lane['receipt'])
            receipt=json.loads(receipt_path.read_text(encoding='utf-8-sig'))
            measured=receipt_values(receipt,lane['receipt_fields'])
            hashes.append(pointer(receipt,lane['receipt_fields']['trajectory_sha256']))
            actual_hash=digest(video)
            if actual_hash!=lane['video_sha256']:raise ValueError('video SHA mismatch: '+str(video))
            raw=run([str(ffprobe),'-v','error','-select_streams','v:0','-show_streams','-show_frames','-show_entries',
                'stream=codec_name,width,height,avg_frame_rate,duration:frame=best_effort_timestamp_time','-of','json',str(video)],f'lane-{number}-ffprobe.log')
            (output/f'lane-{number}-ffprobe.json').write_text(raw,encoding='utf-8')
            media=video_contract(json.loads(raw))
            record=dict(label=lane['label'],video=str(video),video_sha256=actual_hash,receipt=str(receipt_path),
                receipt_sha256=digest(receipt_path),receipt_fields=lane['receipt_fields'],measurements=measured,media=media,examples=[])
            checks['lanes'].append(record)
            page_lanes.append(dict(label=lane['label'],video=video.as_uri(),**measured))
            for index in indices:
                target=output/f'lane-{number}-frame-{index:03d}.png'
                run([str(ffmpeg),'-v','error','-i',str(video),'-vf',f'select=eq(n\,{index})','-frames:v','1','-vsync','0',str(target)],f'lane-{number}-frame-{index:03d}.log')
                with Image.open(target) as image:
                    if image.size!=(media['width'],media['height']):raise ValueError('decoded PNG dimensions disagree with actual video')
                    decoded[(number,index)]=image.convert('RGB').copy()
            if not lane.get('examples'):raise ValueError('at least one SHA-bound actual example PNG per lane required')
            for sample in lane['examples']:
                index=sample['frame'];path=resolve(sample['path'])
                if index not in indices:raise ValueError('example frame must be registered for extraction')
                if digest(path)!=sample['sha256']:raise ValueError('example PNG digest mismatch')
                with Image.open(path) as image:
                    if image.format!='PNG':raise ValueError('example is not actual PNG')
                    expected=decoded[(number,index)]
                    if 'rect' in sample:expected=expected.crop(crop_box(sample['rect'],*expected.size))
                    assertion=image_assertion(image,expected,sample['decoded_rgb8_mae_max'])
                record['examples'].append(dict(frame=index,path=str(path),sha256=digest(path),rect=sample.get('rect'),assertion=assertion))
        require_same_trajectory(trajectory_hash,hashes)
        for index in indices:
            entry=dict(index=index,source_time_s=times[index],pts_s=index/30,rois=[])
            for roi_number,roi in enumerate(rois):
                crop=dict(label=roi['label'],lanes=[])
                for number,lane in enumerate(lanes):
                    image=decoded[(number,index)]
                    box=roi['rects'][number] if 'rects' in roi else roi['rect']
                    box=crop_box(box,*image.size);cut=image.crop(box)
                    path=output/f'frame-{index:03d}-roi-{roi_number}-lane-{number}.png';cut.save(path)
                    crop['lanes'].append(dict(label=lane['label'],file=path.name,sha256=digest(path),rect=list(box),
                        width=cut.width,height=cut.height,source_png=f'lane-{number}-frame-{index:03d}.png',
                        source_png_sha256=digest(output/f'lane-{number}-frame-{index:03d}.png'),resampled=False))
                entry['rois'].append(crop)
            checks['frames'].append(entry)
        document=page(dict(lanes=page_lanes,frames=checks['frames'],source_times_s=times))
        (output/'index.html').write_text(document,encoding='utf-8')
        checks.update(status='passed',html=dict(file='index.html',sha256=digest(output/'index.html')),
            quality_scope='receipt status reproduced, not independently re-adjudicated; no claim that MP4 RGB is raw renderer output',
            browser_scope='page authored; runtime rVFC checks both decoded timestamps; browser QA is a separate execution')
    except BaseException as error:checks.update(status='failed',error=repr(error),traceback=traceback.format_exc())
    finally:write_json(output/'checks.json',checks)
    return checks


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--ffmpeg',default='ffmpeg');p.add_argument('--ffprobe',default='ffprobe')
    a=p.parse_args();result=build(a.manifest,a.output,ffmpeg=a.ffmpeg,ffprobe=a.ffprobe)
    print(json.dumps(dict(status=result['status'],output=str(a.output),error=result.get('error')),ensure_ascii=False))
    return 0 if result['status']=='passed' else 1
if __name__=='__main__':raise SystemExit(main())
