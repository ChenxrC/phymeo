"""Hash-bound one-pass low-rate incoming filter over an existing rigid package."""
import argparse
from copy import deepcopy
import math
from pathlib import Path
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from preview_lowres_filter._helpers import verify,digest,read,save,markers,once
HERE=Path(__file__).resolve().parent


def policy():
    return dict(schema='phymeo-preview-lowres-filter-v1',marker=100,kernel=3,stride=2,passes=1,
        weights='separable [1,2,1] product, normalized over accepted low cells',relative_depth=.003,normal_dot=.98,
        support='center and conservative intervening low-grid rectangle: traced, same actual ID/coverage and matching full anchor/depth/Ns; unsupported center original',
        changes_center_alpha=False,changes_ray_count=False,changes_primary=False,changes_sensor=False,
        resources='one 16-byte RGBA low-grid SSBO; actual original guide MRT immutable; compute-write/read barrier before original K25',
        history='original age4 and rejection unchanged, now consumes filtered current incoming',performance='unmeasured',
        registered_quality='three poses x two 65536-spp refs: same-ID8x8 left-arm block luminance residual RMS each improves >=10%, mean bias increase <=.005; other pixel/motion/speed gates separate; no retuning')


def accept(a,b,valid_a,valid_b,z,qz,normal_dot,traced):
    return (a!=0 and a==b and valid_a==1 and valid_b==1 and traced and math.isfinite(z) and math.isfinite(qz)
        and min(z,qz)>0 and abs(z-qz)<=.003*min(z,qz) and math.isfinite(normal_dot) and normal_dot>=.98)


def corridor(delta):return [(x,y) for y in range(min(0,delta[1]),max(0,delta[1])+1) for x in range(min(0,delta[0]),max(0,delta[0])+1)]


def reference_filter(center,neighbors,accepted,center_supported=True):
    if len(center)!=4 or len(neighbors)!=9 or len(accepted)!=9:raise ValueError('registered kernel required')
    if not center_supported:return center
    weights=[(2 if x==0 else 1)*(2 if y==0 else 1) for y in (-1,0,1) for x in (-1,0,1)]
    denominator=sum(w for w,good in zip(weights,accepted) if good)
    if not denominator:return center
    return tuple(sum(v[c]*w for v,w,good in zip(neighbors,weights,accepted) if good)/denominator for c in range(3))+(center[3],)


def patch_compute(text):
    if 'preview_lowres.glsl' in text:raise ValueError('already filtered')
    text=once(text,'void main() {','''#include "preview_lowres.glsl"
void main() {
    if(PR_PREVIEW_LOWRES_FILTER_ABI!=1u){atomicOr(status.invalid,128u);return;}
    if((p.mode&256u)!=0u){lowresPass();return;}''')
    return once(text,'vec4 candidate=texelFetch(incomingGuide,cell,0);',
        'vec4 candidate=texelFetch(incomingGuide,cell,0);\n                candidate.rgb=lowFiltered.values[uint(cell.y)*((p.width+3u)/4u)+uint(cell.x)].rgb;')


EXPORT='''
PR_API uint32_t pr_preview_lowres_filter_abi_version(){return 1;}
PR_API int pr_preview_lowres_filter_read(void* handle,void* output,size_t bytes){
    try{require(handle,"null lowres owner");lowres_read(*static_cast<Context*>(handle),output,bytes);return 0;}
    catch(const std::exception& e){last_error=e.what();return -1;}
}
'''


def prepare(parent,output):
    parent,manifest=verify(parent);p=manifest['contract']
    if p.get('paths_per_anchor')!=8 or p.get('tile')!=4 or p.get('rigid_history',{}).get('schema')!='phymeo-preview-compact-rigid-history-v1' or any(k in p for k in ('lowres_filter','local_diffuse','pooled_sky','irradiance_filter')):
        raise ValueError('unaltered full-distance8 rigid package required')
    output=Path(output).resolve()
    if output.is_relative_to(parent):raise ValueError('independent new directory required')
    output.mkdir(parents=True,exist_ok=False)
    for key in manifest['generated_sha256']:
        if key.startswith(('source/','runtime/')):
            target=output/key;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(parent/key,target)
    shutil.copyfile(parent/'source-manifest.json',output/'lowres-parent-source-manifest.json')
    def write(path,text):path.write_text(text,encoding='utf-8',newline='\n')
    path=output/'source/_shaders/resident_spatial.comp';write(path,patch_compute(path.read_text(encoding='utf-8')))
    shutil.copyfile(HERE/'lowres.glsl',output/'source/_shaders/preview_lowres.glsl')
    shutil.copyfile(HERE/'lowres_native.inc',output/'source/_native/preview_lowres_native.inc')
    path=output/'source/_native/resident_progressive.inc';text=path.read_text(encoding='utf-8')
    edits=[
        ('Buffer status{},readback{};','Buffer status{},readback{},low_filtered{};'),
        ('&history_debug,&rigid_motion}','&history_debug,&rigid_motion,&low_filtered}'),
        ('#include "preview_history.inc"','#include "preview_lowres_native.inc"\n#include "preview_history.inc"'),
        ('(guided&&!settling)?8191u:15u','(guided&&!settling)?139263u:15u'),
        ('require(named_shader_marker(code,bytes,92,','require(named_shader_marker(code,bytes,100,"PR_PREVIEW_LOWRES_FILTER_ABI",5),"lowres filter compute marker absent");\n    require(named_shader_marker(code,bytes,92,'),
        ('maxPerStageDescriptorStorageBuffers>=5','maxPerStageDescriptorStorageBuffers>=6'),
        ('maxDescriptorSetStorageBuffers>=5','maxDescriptorSetStorageBuffers>=6'),
        ('maxPerStageResources>=17','maxPerStageResources>=18'),
        ('VkDescriptorSetLayoutBinding,17>','VkDescriptorSetLayoutBinding,18>'),
        ('binding_count=17u','binding_count=18u'),
        ('{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,5}','{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,6}'),
        ('        std::array<VkDescriptorPoolSize,4> sizes', '''        const uint64_t low_bytes=static_cast<uint64_t>((width+3)/4)*((height+3)/4)*16;
        require(low_bytes<=c.storage_buffer_limit,"lowres buffer exceeds device range");
        c.make_buffer(state.low_filtered,low_bytes,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_SRC_BIT,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        std::array<VkDescriptorPoolSize,4> sizes'''),
        ('    vkUpdateDescriptorSets(c.device,4,writes.data(),0,nullptr);','''    vkUpdateDescriptorSets(c.device,4,writes.data(),0,nullptr);
    VkDescriptorBufferInfo low_info{state.low_filtered.handle,0,static_cast<VkDeviceSize>((state.width+3)/4)*((state.height+3)/4)*16};
    VkWriteDescriptorSet low_write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};low_write.dstSet=state.set;low_write.dstBinding=17;
    low_write.descriptorCount=1;low_write.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;low_write.pBufferInfo=&low_info;
    vkUpdateDescriptorSets(c.device,1,&low_write,0,nullptr);'''),
        ('    vkCmdPushConstants(commands,state.pipeline_layout,VK_SHADER_STAGE_COMPUTE_BIT,0,c.settling_info.enabled?24u:20u,&push);',
         '    lowres_record(c,frame,push.mode,push.exposure);\n    vkCmdPushConstants(commands,state.pipeline_layout,VK_SHADER_STAGE_COMPUTE_BIT,0,c.settling_info.enabled?24u:20u,&push);')]
    for old,new in edits:text=once(text,old,new)
    write(path,text)
    path=output/'source/_native/renderer.cpp';write(path,path.read_text(encoding='utf-8')+EXPORT)
    result=deepcopy(manifest);result['contract']['lowres_filter']=policy();result['lowres_parent']=str(parent)
    result['lowres_parent_sha256']=digest(parent/'source-manifest.json')
    result['lowres_author_sha256']={p.name:digest(p) for p in (HERE/'build_source.py',HERE/'_helpers.py',HERE/'lowres.glsl',HERE/'lowres_native.inc')}
    result['generated_sha256']={p.relative_to(output).as_posix():digest(p) for f in ('source','runtime') for p in sorted((output/f).rglob('*')) if p.is_file()}
    save(output/'source-manifest.json',result);return result


def seal(directory):
    root,manifest=verify(directory);parent,_=verify(manifest['lowres_parent'])
    if digest(parent/'source-manifest.json')!=manifest['lowres_parent_sha256']:raise ValueError('parent changed')
    files={p.name:digest(p) for p in (root/'build').glob('*.spv')};old={p.name:digest(p) for p in (parent/'build').glob('*.spv')}
    if not old or set(files)!=set(old):raise ValueError('actual shader inventory differs')
    name='resident_spatial.comp.spv';actual=markers((root/'build'/name).read_bytes());before=markers((parent/'build'/name).read_bytes())
    if actual.get('100')!=dict(name='PR_PREVIEW_LOWRES_FILTER_ABI',value=[1]):raise ValueError('actual filter marker missing')
    if any(actual.get(k)!=v for k,v in before.items()) or any(files[k]!=old[k] for k in files if k!=name):raise ValueError('original guide/ray/history shader changed')
    result=dict(schema='phymeo-preview-lowres-filter-seal-v1',manifest_sha256=digest(root/'source-manifest.json'),dll_sha256=digest(root/'build/phymeo_native.dll'),spv_sha256=files,
        unchanged_raster_ray_history_shaders=True,compute_markers=actual,measured_quality=None,measured_performance=None)
    save(root/'lowres-seal.json',result);return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--parent',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p=sub.add_parser('seal');p.add_argument('--prepared',type=Path,required=True)
    a=parser.parse_args();prepare(a.parent,a.output) if a.command=='prepare' else seal(a.prepared)
if __name__=='__main__':main()