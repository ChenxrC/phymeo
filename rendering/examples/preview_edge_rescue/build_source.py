"""Prepare a separate, source-bound cp4 + actual selective full-rate pass.

No compile/import/device work occurs during preparation. Only the exact cp4
layout is admitted. Existing source packages are never edited.
"""
from copy import deepcopy
from hashlib import sha256
import argparse
import json
from pathlib import Path
import shutil

HERE=Path(__file__).resolve().parent
SCHEMA='phymeo-preview-edge-rescue-v1'


def once(text,old,new):
    if text.count(old)!=1: raise ValueError('source anchor must occur once: '+old[:100])
    return text.replace(old,new,1)


def contract(limit=262144):
    if type(limit) is not int or not 1<=limit<=1048576:
        raise ValueError('max_invocations must be an integer in [1,1048576]')
    return dict(schema=SCHEMA,max_support=2,paths_per_selected_fragment=16,
        selection='actual accepted K25 support <= 2 and nonzero diffuse coefficient',
        max_invocations=limit,history=False,
        estimator='original full-rate 16-proposal one-hop radiance MIS, not the compact sky-ratio estimator',
        filter='selected real incoming only; 5x5 binomial; same ID/valid, relative depth .003, Ns dot .98, direct display-code difference <=2',
        counter_scope='extra traced fragment invocations times 16 proposals, not hardware ray queries; overdraw counts toward hard cap',
        budget_failure='reject publication; no truncated or silently sparse substitute',
        unchanged='primary direct/specular and local rho; original four sensor/color output semantics',
        additional_cost='one full-extent selection dispatch, one depth-EQUAL selective full raster draw, one merge dispatch; one RGBA32F attachment',
        primitive_gate=False)


def distance_policy(policy,reflection,ratio):
    local=policy.get('local_diffuse')
    if local is None:
        if 'PR_DIFFUSE_GUIDE_LIMIT' in reflection:raise ValueError('undeclared diffuse distance cap')
        return dict(sparse_radius_m=None,rescue_radius_m=None,sparse_fade=None,
            rescue='original full-distance 16-proposal radiance estimator',specular='unchanged primary mirror policy')
    expected='#ifdef PR_PREVIEW_SPARSE_GUIDE\n#define PR_DIFFUSE_GUIDE_LIMIT 0.1200\n#else\n#define PR_DIFFUSE_GUIDE_LIMIT 3.402823466e38\n#endif'
    if local.get('radius_m')!=.12 or local.get('marker')!=96 or local.get('fade')!='1-smoothstep(0,R,t)':
        raise ValueError('only the frozen 120 mm local base is admitted')
    if not reflection.startswith(expected) or reflection.count('direction,PR_DIFFUSE_GUIDE_LIMIT)')!=2:
        raise ValueError('local guide ray bound/non-guide unlimited scope differs')
    if 'PR_NEARFIELD_DIFFUSE_ABI' not in ratio or '1.0-smoothstep(0.0,PR_DIFFUSE_GUIDE_LIMIT,hit.previewDistance)' not in ratio:
        raise ValueError('bound local hit fade is absent')
    return dict(sparse_radius_m=.12,rescue_radius_m=None,sparse_fade='1-smoothstep(0,R,t)',
        sparse_miss='directions beyond 120 mm use original environment/SH visibility model',
        rescue='selected pixels replace local diffuse with original full-distance16 radiance, with real full-distance terminal hits',
        specular='primary near-field mirror mode unchanged; not clipped to the 120 mm diffuse-guide radius',
        model='generic spatial hybrid of local sparse and full-distance rescue, not a uniform full-distance integrator')

def validate_counts(d):
    keys=('selected_pixels','fragment_invocations','traced_invocations','reconstructed_rescue_pixels',
          'filtered_neighbors','missing_rescue','budget_exhausted','max_invocations','invalid')
    if any(type(d.get(k)) is not int or d[k]<0 for k in keys): raise ValueError('edge counter type')
    if d['invalid'] or d['missing_rescue'] or d['budget_exhausted']:
        raise ValueError('edge capture did not complete')
    if d['selected_pixels']!=d['reconstructed_rescue_pixels']:
        raise ValueError('selected/rescued pixel partition differs')
    if not d['selected_pixels']<=d['fragment_invocations']<=d['max_invocations']:
        raise ValueError('edge invocation bound differs')
    if d['traced_invocations']>d['fragment_invocations']:
        raise ValueError('edge trace invocation count differs')
    if d['filtered_neighbors']<d['selected_pixels']:
        raise ValueError('each rescued pixel must include its actual own sample')
    return dict(d,extra_primary_proposals=16*d['traced_invocations'])


def patch_shader(text):
    text=once(text,'#include "resident_indirect.glsl"',
        '#ifdef PR_EDGE_RESCUE\n#include "preview_edge_indirect.glsl"\n#else\n#include "resident_indirect.glsl"\n#endif')
    text=once(text,'void main() {', '#ifdef PR_EDGE_RESCUE\n#include "preview_edge_fragment.glsl"\n#endif\nvoid main() {')
    anchor='#ifdef PR_PREVIEW_SPARSE_GUIDE\n    float sparseMetal='
    text=once(text,anchor,'#ifdef PR_EDGE_RESCUE\n    edgeRescue(N,reflectionFace,reflectionFaceValid,shadingFacesView,albedo);return;\n#endif\n'+anchor)
    return text


def patch_compute(text):
    text=once(text,'writeonly uniform image2D meanLinear','uniform image2D meanLinear')
    text=once(text,'uint fallback; uint nondiffuse; uint acceptedNeighbors; uint primaryProposals;',
        'uint fallback; uint nondiffuse; uint acceptedNeighbors; uint primaryProposals;\n'
        '    uint selected,invocations,traced,merged,neighbors,missingRescue,budgetExhausted,maxInvocations;')
    text=once(text,'void main() {','#include "preview_edge_merge.glsl"\nvoid main() {')
    text=once(text,'vec4 primary=imageLoad(currentLinear,pixel);',
        'vec4 primary=imageLoad(currentLinear,pixel);\n    if(p.batch==1u){edgeMerge(pixel,extent,primary);return;}')
    text=once(text,'vec3 linear=primary.rgb;', 'vec3 linear=primary.rgb;bool selected=false;')
    text=once(text,'vec3 incoming=own.rgb;',
        'selected=accepted<=2u;\n            if(selected)atomicAdd(status.selected,1u);\n            vec3 incoming=own.rgb;')
    text=once(text,'imageStore(meanLinear,pixel,vec4(linear,1.0));',
        'imageStore(meanLinear,pixel,vec4(linear,selected?1.0:0.0));')
    return text


def edge_setup_source(original):
    # Reuse the bound pipeline's vertex layout and raster convention; replace
    # its attachments/depth policy explicitly rather than importing a renderer.
    start=original.index('        VkShaderModuleCreateInfo module')
    end=original.index('\nvoid spatial_record_guide')
    body=original[start:end]
    body=body.replace('state.guide_fragment','state.edge_fragment').replace('state.guide_pipeline','state.edge_pipeline').replace('state.guide_pass','state.edge_pass')
    body=once(body,'depth_state.depthTestEnable=depth_state.depthWriteEnable=VK_TRUE;depth_state.depthCompareOp=VK_COMPARE_OP_LESS;',
        'depth_state.depthTestEnable=VK_TRUE;depth_state.depthWriteEnable=VK_FALSE;depth_state.depthCompareOp=VK_COMPARE_OP_EQUAL;')
    body=once(body,'std::array<VkPipelineColorBlendAttachmentState,4> blends{};', 'std::array<VkPipelineColorBlendAttachmentState,1> blends{};')
    body=once(body,'blend.attachmentCount=4;', 'blend.attachmentCount=1;')
    return (HERE/'edge_setup.inc').read_text(encoding='utf-8')+body


def patch_progressive(t):
    t=once(t,'VkRenderPass guide_pass{};', 'VkRenderPass edge_pass{};VkPipeline edge_pipeline{};VkShaderModule edge_fragment{};\n    VkRenderPass guide_pass{};')
    t=once(t,'        clear_targets();\n        release(settling_pipeline',
        '        clear_targets();\n        release(edge_pipeline,[&](auto h){vkDestroyPipeline(c.device,h,nullptr);});\n'
        '        release(edge_fragment,[&](auto h){vkDestroyShaderModule(c.device,h,nullptr);});\n'
        '        release(edge_pass,[&](auto h){vkDestroyRenderPass(c.device,h,nullptr);});\n        release(settling_pipeline')
    t=once(t,'?4095u:15u','?8191u:15u')
    t=once(t,'std::array<VkDescriptorSetLayoutBinding,12>', 'std::array<VkDescriptorSetLayoutBinding,13>')
    t=once(t,'binding_count=guided?12u:4u','binding_count=guided?13u:4u')
    t=once(t,'void progressive_prepare(', 'void edge_bind(Context&,FrameResources&);\nvoid edge_record(Context&,FrameResources&);\nvoid progressive_prepare(')
    t=once(t,'VK_FORMAT_R32G32B32A32_SFLOAT,false,VK_SAMPLE_COUNT_1_BIT,VK_IMAGE_USAGE_STORAGE_BIT);',
        'VK_FORMAT_R32G32B32A32_SFLOAT,false,VK_SAMPLE_COUNT_1_BIT,VK_IMAGE_USAGE_STORAGE_BIT|VK_IMAGE_USAGE_SAMPLED_BIT);')
    t=once(t,'c.make_buffer(state.status,32,','c.make_buffer(state.status,64,')
    t=once(t,'{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,8}', '{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,9}')
    t=once(t,'VkDescriptorBufferInfo status{state.status.handle,0,32};','VkDescriptorBufferInfo status{state.status.handle,0,64};')
    t=once(t,'        vkUpdateDescriptorSets(c.device,8,extra.data(),0,nullptr);',
        '        vkUpdateDescriptorSets(c.device,8,extra.data(),0,nullptr);\n        edge_bind(c,frame);')
    t=once(t,'vkCmdFillBuffer(commands,state.status.handle,0,32,0);',
        'vkCmdFillBuffer(commands,state.status.handle,0,60,0);\n    vkCmdUpdateBuffer(commands,state.status.handle,60,4,&c.edge_limit);')
    t=once(t,'status.size=32;', 'status.size=64;')
    t=once(t,'VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,\n            0,0,nullptr,0,nullptr,8,guide_barriers.data()',
        'VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT|VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,\n            0,0,nullptr,0,nullptr,8,guide_barriers.data()')
    marker='    if(c.settling_info.enabled&&batch<15) {'
    t=once(t,marker,(HERE/'edge_record_dispatch.inc').read_text(encoding='utf-8')+'\n'+marker)
    t=once(t,'std::array<VkImageMemoryBarrier,3> before{{', 'std::array<VkImageMemoryBarrier,4> before{{\n        progressive_barrier(frame.images[5].handle,VK_IMAGE_LAYOUT_UNDEFINED,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,0,VK_ACCESS_SHADER_READ_BIT),')
    t=once(t,'0,0,nullptr,0,nullptr,3,before.data()', '0,0,nullptr,0,nullptr,4,before.data()')
    t=once(t,'VkBufferCopy copy{0,status_offset,32};','VkBufferCopy copy{0,status_offset,64};')
    return t


def patch_renderer(t):
    t=once(t,'std::array<uint32_t,8> sparse_counts{};',
        'std::array<uint32_t,8> sparse_counts{};\n    std::array<uint32_t,16> edge_counts{};uint32_t edge_limit=0;')
    t=once(t,'        device_info.pEnabledFeatures=&enabled_features;',
        '        require(!resident||available_features.fragmentStoresAndAtomics,"edge rescue requires fragment atomics");\n'
        '        enabled_features.fragmentStoresAndAtomics=resident?VK_TRUE:VK_FALSE;\n        device_info.pEnabledFeatures=&enabled_features;')
    t=once(t,'std::array<VkDescriptorSetLayoutBinding, 16>', 'std::array<VkDescriptorSetLayoutBinding, 20>')
    t=once(t,'        layout.bindingCount=binding_count; layout.pBindings=bindings.data();',
        '        if(resident)for(uint32_t i=16;i<20;++i)bindings[binding_count++]={i,i==17?VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1,VK_SHADER_STAGE_FRAGMENT_BIT,nullptr};\n'
        '        if(resident){VkPhysicalDeviceProperties ep{};vkGetPhysicalDeviceProperties(physical,&ep);const auto& l=ep.limits;\n'
        '            const uint32_t buffers=normal_info.shader_compatible?10u:9u;\n'
        '            require(l.maxPerStageDescriptorSamplers>=9&&l.maxPerStageDescriptorSampledImages>=9&&\n'
        '                l.maxDescriptorSetSamplers>=9&&l.maxDescriptorSetSampledImages>=9&&\n'
        '                l.maxPerStageDescriptorStorageBuffers>=buffers&&l.maxDescriptorSetStorageBuffers>=buffers&&\n'
        '                l.maxPerStageDescriptorUniformBuffers>=3&&l.maxDescriptorSetUniformBuffers>=3&&\n'
        '                l.maxPerStageDescriptorStorageImages>=3&&l.maxDescriptorSetStorageImages>=3&&\n'
        '                l.maxPerStageResources>=binding_count&&l.maxFragmentCombinedOutputResources>=buffers+1,\n'
        '                "edge descriptor/resource limits unavailable");}\n'
        '        layout.bindingCount=binding_count; layout.pBindings=bindings.data();')
    t=once(t,'VkFramebuffer framebuffer{},guide_framebuffer{};', 'VkFramebuffer framebuffer{},guide_framebuffer{},edge_framebuffer{};')
    t=once(t,'        release(guide_framebuffer,',
        '        release(edge_framebuffer,[&](auto h){vkDestroyFramebuffer(c.device,h,nullptr);});\n        release(guide_framebuffer,')
    t=once(t,'require(named_shader_marker(code,bytes,88,"PR_PREVIEW_COMPACT_COMPUTE_ABI",5),',
        'require(named_shader_marker(code,bytes,98,"PR_PREVIEW_EDGE_COMPUTE_ABI",5)&&named_shader_marker(code,bytes,88,"PR_PREVIEW_COMPACT_COMPUTE_ABI",5),')
    t=once(t,'c.sparse_captures,5,8,0,0,0','c.sparse_captures,5,9,0,0,0')
    return t+'\n'+(HERE/'edge_exports.inc').read_text(encoding='utf-8')


def patch_resident(t):
    t=once(t,'(c.progressive_info.shader_compatible?36:0)', '(c.progressive_info.shader_compatible?68:0)')
    t=once(t,'    const uint32_t low_width=(width+3)/4,low_height=(height+3)/4;',
        '    c.make_image(frame->images[5],width,height,VK_FORMAT_R32G32B32A32_SFLOAT,false,VK_SAMPLE_COUNT_1_BIT,VK_IMAGE_USAGE_SAMPLED_BIT);\n'
        '    const uint32_t low_width=(width+3)/4,low_height=(height+3)/4;')
    t=once(t,'{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 8}', '{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 9}')
    t=once(t,'{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,2}', '{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,5}')
    t=once(t,'#include "resident_spatial.inc"', '#include "resident_spatial.inc"\n#include "preview_edge.inc"')
    t=once(t,'        std::memcpy(c.sparse_counts.data(),static_cast<const uint8_t*>(frame.readback_mapping)+progressive_status_offset,32);',
        '        std::memcpy(c.sparse_counts.data(),static_cast<const uint8_t*>(frame.readback_mapping)+progressive_status_offset,32);\n'
        '        std::memcpy(c.edge_counts.data(),static_cast<const uint8_t*>(frame.readback_mapping)+progressive_status_offset,64);')
    t=once(t,'        require(invalid==0,"sparse shader produced invalid radiance/attributes; no publication");',
        '        std::memcpy(c.edge_counts.data(),static_cast<const uint8_t*>(frame.readback_mapping)+progressive_status_offset,64);\n'
        '        require(invalid==0,"edge/sparse shader invalid or rescue budget exceeded; no publication");')
    t=t.replace('(progressive_rgb?32:0)','(progressive_rgb?64:0)')
    t=once(t,'c.sparse_counts.fill(0);', 'c.sparse_counts.fill(0);c.edge_counts.fill(0);')
    return t


def patch_compact_read(t):
    t=once(t,'plane<5,','plane<6,')
    t=t.replace('plane==0?state.', '(plane==0||plane==5)?state.')
    t=once(t,'plane<2?16u:', '(plane<2||plane==5)?16u:')
    t=once(t,'const uint32_t image=plane==0?6u:6u+plane;', 'const uint32_t image=plane==5?5u:(plane==0?6u:6u+plane);')
    return t


def prepare(base,output,adapter_root,limit=262144):
    policy=contract(limit);base=Path(base).resolve();output=Path(output).resolve();adapter_root=Path(adapter_root).resolve()
    if output.exists():raise FileExistsError('new output required: '+str(output))
    source_manifest=base/'source-manifest.json'
    manifest=json.loads(source_manifest.read_text(encoding='utf-8'))
    c=manifest['contract']
    expected=(4,4,2,25,2,'visibility-ratio')
    if tuple(c.get(k) for k in ('tile','paths_per_anchor','area_light_grid','neighbor_candidates','reconstruction_revision','estimator'))!=expected or c.get('compact',{}).get('schema')!='phymeo-preview-compact-v1':
        raise ValueError('only the explicitly bound cp4 T4/P4/G2/K25/R2/ratio package is supported')
    for name,digest in manifest['generated_sha256'].items():
        path=base/Path(name.replace('\\','/'))
        if sha256(path.read_bytes()).hexdigest()!=digest:raise ValueError('base source changed: '+name)
    paths={n:(base/'source/_native'/n).read_text(encoding='utf-8') for n in ('renderer.cpp','resident.inc','resident_progressive.inc','resident_spatial.inc','preview_compact.inc','CMakeLists.txt')}
    shader=(base/'source/_shaders/resident.frag').read_text(encoding='utf-8')
    compute=(base/'source/_shaders/resident_spatial.comp').read_text(encoding='utf-8')
    full=(base/'runtime/phymeo_rendering/backends/_shaders/resident_indirect.glsl').read_text(encoding='utf-8')
    if 'for(uint i=0u;i<16u;++i)' not in full or 'return sum*(1.0/16.0)' not in full:raise ValueError('bound original full-rate estimator missing')
    policy['distance']=distance_policy(c,
        (base/'source/_shaders/resident_reflection.glsl').read_text(encoding='utf-8'),
        (base/'source/_shaders/preview_sparse_ratio.glsl').read_text(encoding='utf-8'))
    # Apply every strict transformation before creating any output.
    patched={'renderer.cpp':patch_renderer(paths['renderer.cpp']), 'resident.inc':patch_resident(paths['resident.inc']),
        'resident_progressive.inc':patch_progressive(paths['resident_progressive.inc']),
        'preview_compact.inc':patch_compact_read(paths['preview_compact.inc']),
        'preview_edge.inc':edge_setup_source(paths['resident_spatial.inc'])+'\n'+(HERE/'edge_runtime.inc').read_text(encoding='utf-8')}
    cmake=paths['CMakeLists.txt']
    cmake=once(cmake,'shader(resident_spatial.comp resident_spatial.comp)',
        'shader(resident_spatial.comp resident_spatial.comp)\n'
        'shader(resident_edge_rescue.frag resident.frag -DPR_RESIDENT_RAY_QUERY=1 -DPR_RESIDENT_INDIRECT=1 -DPR_RESIDENT_PROGRESSIVE=1 -DPR_PREVIEW_SPARSE_MAIN=1 -DPR_EDGE_RESCUE=1)\n'
        'shader(resident_edge_rescue_normal.frag resident.frag -DPR_RESIDENT_RAY_QUERY=1 -DPR_RESIDENT_INDIRECT=1 -DPR_RESIDENT_PROGRESSIVE=1 -DPR_PREVIEW_SPARSE_MAIN=1 -DPR_RESIDENT_NORMAL_MAPS=1 -DPR_EDGE_RESCUE=1)')
    patched['CMakeLists.txt']=cmake
    shader=patch_shader(shader);compute=patch_compute(compute)
    for directory in ('source','runtime'):shutil.copytree(base/directory,output/directory,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name,value in patched.items():(output/'source/_native'/name).write_text(value,encoding='utf-8',newline='\n')
    for name,value in {'resident.frag':shader,'resident_spatial.comp':compute,'preview_edge_indirect.glsl':full,
        'preview_edge_fragment.glsl':(HERE/'edge_fragment.glsl').read_text(encoding='utf-8'), 'preview_edge_merge.glsl':(HERE/'edge_merge.glsl').read_text(encoding='utf-8')}.items():
        (output/'source/_shaders'/name).write_text(value,encoding='utf-8',newline='\n')
    adapters={}
    for name in ('preview_sparse','preview_compact'):
        source=adapter_root/name/'runtime.py';target=output/'adapters'/name/'runtime.py'
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
        adapters[str(source)]=sha256(source.read_bytes()).hexdigest()
    manifest=deepcopy(manifest);manifest.pop('compiled_sha256',None);manifest['contract']['edge_rescue']=policy
    manifest['contract']['timing']='full primary + actual compact GI raster + K25 selection + full-extent selective16 raster + incoming filter/merge + requested copies; one owner'
    manifest['edge_source']={'base_package':str(base),'base_manifest_sha256':sha256(source_manifest.read_bytes()).hexdigest(),
        'full_estimator_sha256':sha256(full.encode()).hexdigest(),'adapter_source_sha256':adapters,
        'author_files_sha256':{p.name:sha256(p.read_bytes()).hexdigest() for p in HERE.iterdir() if p.suffix in ('.py','.glsl','.inc')}}
    manifest['generated_sha256']={p.relative_to(output).as_posix():sha256(p.read_bytes()).hexdigest()
        for folder in ('runtime','source','adapters') for p in sorted((output/folder).rglob('*')) if p.is_file()}
    (output/'source-manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return manifest


def seal(output):
    root=Path(output).resolve(strict=True);path=root/'source-manifest.json'
    manifest=json.loads(path.read_text(encoding='utf-8'))
    if manifest['contract'].get('edge_rescue',{}).get('schema')!=SCHEMA:raise ValueError('edge package required')
    if 'compiled_sha256' in manifest:raise ValueError('already sealed; never overwrite compiled evidence')
    for relative,expected in manifest['generated_sha256'].items():
        if sha256((root/Path(relative.replace('\\','/'))).read_bytes()).hexdigest()!=expected:
            raise ValueError('generated source changed before seal: '+relative)
    files=[root/'build/phymeo_native.dll',*sorted((root/'build').glob('*.spv'))]
    if not files[0].is_file() or not all((root/'build'/n).is_file() for n in ('resident_edge_rescue.frag.spv','resident_edge_rescue_normal.frag.spv','resident_spatial.comp.spv')):
        raise ValueError('complete real build required')
    manifest['compiled_sha256']={p.name:sha256(p.read_bytes()).hexdigest() for p in files}
    path.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base');parser.add_argument('--output',required=True);parser.add_argument('--seal',action='store_true')
    parser.add_argument('--adapter-root');parser.add_argument('--max-invocations',type=int,default=262144)
    a=parser.parse_args();m=seal(a.output) if a.seal else prepare(a.base,a.output,a.adapter_root,a.max_invocations)
    print(json.dumps({'output':str(Path(a.output).resolve()),'contract':m['contract']['edge_rescue']},indent=2))

if __name__=='__main__':main()