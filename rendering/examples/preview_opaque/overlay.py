"""Create an opt-in opaque-triangle shader overlay; never overwrite source/build.

Commands execute only when explicitly invoked. No public backend default changes.
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import shutil
import struct
import subprocess

MARKER=74
DEFAULT_FRAGMENTS=(
    'resident_rayquery_area8.frag.spv',
    'resident_rayquery_reflection_area8.frag.spv',
    'resident_rayquery_indirect_progressive_area8.frag.spv',
)


def shadow_decision(selected,distances,lower,upper):
    if selected is None:return False
    if lower<selected<upper:return True
    return any(lower<t<upper for t in distances)


def patch_shadow(source):
    source=source.replace('\r\n','\n')
    if source.count('bool occluded(')!=1 or 'gl_RayFlagsNoOpaqueEXT' not in source:
        raise ValueError('shadow source contract changed')
    original=source.replace('bool occluded(','bool previewOriginalOccluded(',1)
    return original+'''
layout(constant_id=74) const uint PR_PREVIEW_OPAQUE_ABI=1u;
bool occluded(vec3 origin,vec3 direction,float tMax) {
    if (PR_PREVIEW_OPAQUE_ABI!=1u) return previewOriginalOccluded(origin,direction,tMax);
    if (u.counts.w==0u || tMax<=u.options.y) return false;
    rayQueryEXT query;
    rayQueryInitializeEXT(query,sceneAcceleration,
        gl_RayFlagsOpaqueEXT | gl_RayFlagsTerminateOnFirstHitEXT,
        0xffu,origin,u.options.y,direction,tMax);
    while (rayQueryProceedEXT(query)) {}
    if (rayQueryGetIntersectionTypeEXT(query,true)==gl_RayQueryCommittedIntersectionNoneEXT) return false;
    float t=rayQueryGetIntersectionTEXT(query,true);
    if (t>u.options.y && t<tMax) return true;
    // First-hit is unordered: a committed upper endpoint can hide an interior
    // blocker. Retry the ENTIRE original segment, never just reject that hit.
    return previewOriginalOccluded(origin,direction,tMax);
}
'''


def original_query(source):
    source=source.replace('\r\n','\n')
    start=source.find('    rayQueryInitializeEXT(query,sceneAcceleration,gl_RayFlagsNoOpaqueEXT')
    end=source.find('    if(rayQueryGetIntersectionTypeEXT(query,true)!=',start)
    if start<0 or end<0 or source.count('rayQueryEXT query;')!=1:
        raise ValueError('closest-hit source contract changed')
    return source[start:end]


def patch_closest(source):
    source=source.replace('\r\n','\n');old=original_query(source)
    fast='''    bool previewRetry=true;
    if(PR_PREVIEW_OPAQUE_ABI==1u) {
        rayQueryInitializeEXT(query,sceneAcceleration,gl_RayFlagsOpaqueEXT,0xffu,
            P+Ng*u.options.y,u.options.y,direction,3.402823466e38);
        while(rayQueryProceedEXT(query)) {}
        if(rayQueryGetIntersectionTypeEXT(query,true)==gl_RayQueryCommittedIntersectionNoneEXT) return false;
        float previewT=rayQueryGetIntersectionTEXT(query,true);
        previewRetry=!(previewT>u.options.y && previewT<3.402823466e38);
    }
    if(previewRetry) {
'''+old+'''    }
'''
    return source.replace(old,fast,1)


def variant_defines(filename):
    m=re.fullmatch(r'resident_rayquery(?P<normal>_normal)?(?P<reflection>_reflection)?(?P<indirect>_indirect)?(?P<progressive>_progressive)?(?:_area(?P<grid>[48]))?(?:_secondary(?P<secondary>[24]))?\.frag\.spv',filename)
    if not m or (m['progressive'] and not m['indirect']):raise ValueError('unregistered fragment filename')
    args=['PR_RESIDENT_RAY_QUERY=1','PR_STUDIO_AREA_GRID='+str(m['grid'] or 2)]
    for key,macro in (('normal','NORMAL_MAPS'),('reflection','REFLECTION'),('indirect','INDIRECT'),('progressive','PROGRESSIVE')):
        if m[key]:args.append('PR_RESIDENT_'+macro+'=1')
    if m['secondary']:args.append('PR_STUDIO_SECONDARY_GRID='+m['secondary'])
    return args


def marker_map(blob):
    if len(blob)%4 or len(blob)<20:raise ValueError('invalid SPIR-V')
    w=struct.unpack('<%dI'%(len(blob)//4),blob)
    if w[0]!=0x07230203:raise ValueError('invalid SPIR-V magic')
    names={};ids={};values={};i=5
    while i<len(w):
        n=w[i]>>16;op=w[i]&65535
        if not n or i+n>len(w):raise ValueError('truncated instruction')
        a=w[i+1:i+n]
        if op==5:names[a[0]]=struct.pack('<%dI'%(len(a)-1),*a[1:]).split(b'\0')[0].decode()
        if op==71 and len(a)>=3 and a[1]==1:ids[a[0]]=a[2]
        if op==50:values[a[1]]=list(a[2:])
        i+=n
    return {str(index):dict(name=names.get(target),value=values.get(target)) for target,index in ids.items()}


def digest(path):return sha256(Path(path).read_bytes()).hexdigest()
def write_json(path,data):Path(path).write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8')


def prepare(source,baseline,output,compiler,fragments=DEFAULT_FRAGMENTS):
    source=Path(source).resolve();baseline=Path(baseline).resolve();output=Path(output).resolve()
    output.mkdir(parents=True,exist_ok=False);staged=output/'source';staged.mkdir();built=output/'shaders';built.mkdir()
    registration=dict(schema='phymeo-preview-opaque-build-v1',status='preparing',source=str(source),baseline=str(baseline),
        compiler=str(Path(compiler).resolve()),compiler_sha256=digest(compiler),source_files={},baseline_files={},compiled=[],
        semantics='opaque triangles only; original full-segment fallback at boundaries; no stable tie ordering promised',
        lighting_and_sample_budgets_changed=False,public_default_changed=False)
    write_json(output/'manifest.json',registration)
    try:
        for path in sorted(source.iterdir()):
            if path.is_file() and path.suffix in ('.glsl','.frag','.vert','.comp'):
                shutil.copyfile(path,staged/path.name);registration['source_files'][path.name]=digest(path)
        for path in sorted(baseline.glob('*.spv')):
            shutil.copyfile(path,built/path.name);registration['baseline_files'][path.name]=digest(path)
        (staged/'resident_rayquery.glsl').write_text(patch_shadow((staged/'resident_rayquery.glsl').read_text()),encoding='utf-8')
        (staged/'resident_reflection.glsl').write_text(patch_closest((staged/'resident_reflection.glsl').read_text()),encoding='utf-8')
        selected=set(fragments)
        for name in tuple(selected):
            variant_defines(name)
            # A progressive direct guide has the original PRIMARY grid and no
            # secondary/reflection/indirect flags. Compile it from the same patch.
            m=re.search(r'_area([48])',name);grid='_area'+m[1] if m else ''
            selected.add('resident_rayquery'+('_normal' if '_normal' in name else '')+grid+'.frag.spv')
        for name in sorted(selected):
            old=baseline/name
            if not old.is_file():raise ValueError('baseline artifact missing: '+name)
            args=[str(compiler),'-V','--target-env','vulkan1.2','--glsl-version','460']
            args+=['-D'+x for x in variant_defines(name)]+[str(staged/'resident.frag'),'-o',str(built/name)]
            result=subprocess.run(args,text=True,capture_output=True)
            (output/(name+'.log')).write_text(result.stdout+result.stderr,encoding='utf-8')
            if result.returncode:raise RuntimeError('compiler failed: '+name)
            before=marker_map(old.read_bytes());after=marker_map((built/name).read_bytes())
            if str(MARKER) in before:raise ValueError('preview marker already occupied')
            if any(after.get(k)!=v for k,v in before.items()):raise ValueError('original compiled marker changed: '+name)
            if after.get(str(MARKER))!=dict(name='PR_PREVIEW_OPAQUE_ABI',value=[1]):raise ValueError('preview marker missing')
            registration['compiled'].append(dict(file=name,command=args,baseline_sha256=digest(old),sha256=digest(built/name),markers=after))
        registration['output_files']={p.name:digest(p) for p in sorted(built.glob('*.spv'))}
        registration['status']='ready'
    except BaseException as error:
        registration.update(status='failed',error=repr(error));raise
    finally:write_json(output/'manifest.json',registration)
    return registration


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--baseline',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--compiler',type=Path,required=True);p.add_argument('--fragment',action='append')
    a=p.parse_args();prepare(a.source,a.baseline,a.output,a.compiler,a.fragment or DEFAULT_FRAGMENTS)

if __name__=='__main__':main()
