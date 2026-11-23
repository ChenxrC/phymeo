"""Actual Vulkan open-endpoint diagnostic using the exact patched query bodies.

This is a query probe, not a lighting/performance capture. The RGB codes encode
shadow boolean and closest instance identity; no physical radiance is claimed.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from overlay import digest,write_json,patch_shadow,patch_closest


def probe_source(source,optimized):
    source=Path(source)
    main=(source/'resident.frag').read_text()
    reflection=(source/'resident_reflection.glsl').read_text()
    if optimized:reflection=patch_closest(reflection)
    start=reflection.index('    rayQueryEXT query;');end=reflection.index('    uint slot=',start)
    query=reflection[start:end].replace('return false;','return 0u;')
    function='uint previewNearest() {\n    vec3 P=vec3(0.0),Ng=vec3(0.0),direction=vec3(0,0,-1);\n'+query
    function+='    return scene.instances[uint(rayQueryGetIntersectionInstanceCustomIndexEXT(query,true))].binding.y;\n}\n'
    return main[:main.index('void main() {')]+function+'''
void main() {
    forwardDepth=-(u.view*vec4(worldPosition,1.0)).z;
    instanceId=ids.x;valid=1u;
    bool blocked=occluded(vec3(0),vec3(0,0,-1),1.0);
    rgb=vec4(blocked?1.0:0.0,float(previewNearest())/255.0,0.0,1.0);
}
'''


def run(source,baseline,library,compiler,output,device_index=1):
    from phymeo_rendering.assets import Material
    from phymeo_rendering.backends.native import NativeBackend
    from phymeo_rendering.cameras import Camera
    from phymeo_rendering.outputs import Channel
    from phymeo_rendering.pipeline import CaptureRequest,RenderPath
    from phymeo_rendering.scene import VisualScene,FrameKey,SceneFrame,Pose,NodePose,Instance
    from _panda_scene import write_mesh
    from near_reflection_chart import closed
    source=Path(source).resolve();baseline=Path(baseline).resolve();output=Path(output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    bias=struct.unpack('<f',struct.pack('<f',1e-5))[0]
    fixtures=(('empty',()),('lower',( ('lower',bias,2),)),('upper',( ('upper',1.,4),)),
        ('inside',( ('inside',.5,3),)),('upper-first',( ('upper',1.,4),('inside',.5,3))),
        ('lower-first',( ('lower',bias,2),('inside',.5,3))),
        ('both-endpoints',( ('lower',bias,2),('upper',1.,4))),
        ('all',( ('upper',1.,4),('lower',bias,2),('inside',.5,3))))
    report=dict(schema='phymeo-preview-opaque-endpoints-v1',status='running',library_sha256=digest(library),
        bias_f32=bias,shadow_interval='(bias,1)',closest_interval='original: t>bias through FLT_MAX',
        boundary_order='instance input orders test endpoint+interior; hardware candidate visitation order is not controlled',
        fixtures=[dict(name=n,triangles=list(f)) for n,f in fixtures],captures=[],cleanup=[],passed=False)
    write_json(output/'registration.json',report)
    dirs={}
    try:
        for mode in ('original','opaque'):
            folder=output/mode;folder.mkdir();stage=folder/'source';stage.mkdir();binary=folder/'shaders';binary.mkdir()
            for p in source.iterdir():
                if p.is_file():shutil.copyfile(p,stage/p.name)
            for p in baseline.glob('*.spv'):shutil.copyfile(p,binary/p.name)
            if mode=='opaque':
                (stage/'resident_rayquery.glsl').write_text(patch_shadow((source/'resident_rayquery.glsl').read_text()),encoding='utf-8')
            # Query nearest is copied into the probe directly, not by executing
            # terminal shading, while occluded() is the actual include function.
            (stage/'probe.frag').write_text(probe_source(source,mode=='opaque'),encoding='utf-8')
            cmd=[str(compiler),'-V','--target-env','vulkan1.2','--glsl-version','460','-DPR_RESIDENT_RAY_QUERY=1',
                str(stage/'probe.frag'),'-o',str(binary/'resident_rayquery.frag.spv')]
            result=subprocess.run(cmd,text=True,capture_output=True)
            (folder/'compile.log').write_text(result.stdout+result.stderr,encoding='utf-8')
            if result.returncode:raise RuntimeError('endpoint probe compile failed: '+mode)
            dirs[mode]=binary
        mesh=write_mesh(output,'plane',((-.75,.75,0.),(.75,.75,0.),(.75,-.75,0.),(-.75,-.75,0.)),
            ((0.,0.,1.),)*4,((0,2,1),(0,3,2)))
        for name,triangles in fixtures:
            instances=[Instance(1,mesh.id,'white','receiver')]+[Instance(i,mesh.id,'white',n) for n,t,i in triangles]
            scene=VisualScene('endpoint-'+name,(mesh,),(Material('white',(1.,1.,1.),0.,1.),),tuple(instances))
            key=FrameKey(scene.key,'probe',0,0,0,'authored')
            frame=SceneFrame(key,0.,(NodePose('receiver',Pose((0.,0.,-2.))),)+tuple(NodePose(n,Pose((0.,0.,-t))) for n,t,i in triangles))
            camera=Camera('probe',key,Pose(),8,8,8.,8.,4.,4.,.1,4.)
            expected_shadow=any(bias<t<1. for n,t,i in triangles)
            candidates=[(2.,1)]+[(t,i) for n,t,i in triangles if t>bias]
            expected_id=min(candidates)[1];control=None
            for mode in ('original','opaque'):
                owner=NativeBackend(library_path=library,shader_directory=dirs[mode],resident=True,shadow_mode='ray_query',
                    device_index=device_index,validation=True).open_scene(scene)
                try:
                    capture=owner.capture(frame,CaptureRequest(camera,RenderPath.RASTER,tuple(Channel)))
                    planes={p.channel.name:p.data for p in capture.planes};pixel=planes['RGB'][(4*8+4)*3:(4*8+4)*3+3]
                    expected=bytes((255 if expected_shadow else 0,expected_id,0))
                    equal=control is None or all(control[n]==planes[n] for n in ('FORWARD_DEPTH','INSTANCE_ID','VALID'))
                    files={}
                    for channel,data in planes.items():
                        path=output/f'{name}-{mode}-{channel}.raw';path.write_bytes(data);files[channel]=dict(file=path.name,sha256=digest(path))
                    report['captures'].append(dict(name=name,mode=mode,actual=list(pixel),expected=list(expected),
                        passed=pixel==expected and equal,sensor_equal=equal,files=files,frame=asdict(frame),
                        shader_sha256=digest(dirs[mode]/'resident_rayquery.frag.spv')))
                    if mode=='original':control=planes
                finally:
                    owner.close();diag=owner.diagnostics();report['cleanup'].append(dict(case=name,mode=mode,passed=closed(diag),diagnostics=diag))
                write_json(output/'summary.json',report)
        report.update(status='complete',passed=all(x['passed'] for x in report['captures']+report['cleanup']))
    except BaseException as error:report.update(status='failed',error=repr(error))
    finally:write_json(output/'summary.json',report)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','baseline','library','compiler','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--device-index',type=int,default=1);a=p.parse_args()
    r=run(a.source,a.baseline,a.library,a.compiler,a.output,a.device_index)
    print(json.dumps(dict(status=r['status'],passed=r['passed'])))
    return 0 if r['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
