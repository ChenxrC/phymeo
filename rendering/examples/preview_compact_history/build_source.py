"""Create an exclusive GPU-history package from an immutable compact package."""
import argparse
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import shutil


HERE=Path(__file__).resolve().parent


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def prepare(parent,output):
    parent=Path(parent).resolve();output=Path(output).resolve()
    raw=(parent/"source-manifest.json").read_bytes();manifest=json.loads(raw)
    policy=deepcopy(manifest["contract"])
    if policy.get("compact",{}).get("schema")!="phymeo-preview-compact-v1" or policy["compact"].get("history")!="none":
        raise ValueError("immutable plain compact source required")
    if not isinstance(policy.get("history"),str) or not policy["history"].startswith("none") or policy.get("bounces")!=1:
        raise ValueError("history composition is explicit; unsupported source")
    for relative,expected in manifest["generated_sha256"].items():
        if digest(parent/relative)!=expected:raise ValueError("compact source changed: "+relative)
    output.mkdir(parents=True,exist_ok=False)
    shutil.copytree(parent/"source",output/"source")
    shutil.copytree(parent/"runtime",output/"runtime")

    def write(path,value):
        path.write_text(value,encoding="utf-8",newline="\n")

    def change(relative,edits):
        path=output/relative;value=path.read_text(encoding="utf-8")
        for old,new,count in edits:
            if value.count(old)!=count:
                raise ValueError(f"history anchor drift {relative}: expected {count}: {old[:90]}")
            value=value.replace(old,new)
        write(path,value)

    change("source/_native/renderer.cpp",[
        ('    std::array<uint32_t,8> sparse_counts{};', '''    std::array<uint32_t,8> sparse_counts{};
    bool history_enabled=false,history_armed=false,history_reset=true,history_last_reset=false,history_rendered=false;
    uint32_t history_sequence=0,history_last_sequence=0,history_captures=0,history_resets=0;
    uint64_t history_uniform_bytes=0,history_reads=0,history_read_bytes=0;
    std::array<uint32_t,8> history_counts{};''',1),
    ])
    change("source/_native/resident_progressive.inc",[
        ('    Buffer status{},readback{};', '''    Buffer status{},readback{};
    Buffer history_current{},history_pixels[2]{},history_uniform{},history_debug{};
    void* history_debug_mapping=nullptr;
    VkShaderModule history_shader{};
    VkPipeline history_pipeline{};
    bool history_valid=false;
    uint32_t history_read_slot=0;
    PrParams history_params{},history_pending_params{};''',1),
        ('        for(auto* buffer:{&status,&readback}) {', '''        if(history_debug_mapping){vkUnmapMemory(c.device,history_debug.memory);history_debug_mapping=nullptr;}
        history_valid=false;history_read_slot=0;
        for(auto* buffer:{&status,&readback,&history_current,&history_pixels[0],&history_pixels[1],&history_uniform,&history_debug}) {''',1),
        ('        clear_targets();\n        release(settling_pipeline', '''        clear_targets();
        release(history_pipeline,[&](auto h){vkDestroyPipeline(c.device,h,nullptr);});
        release(history_shader,[&](auto h){vkDestroyShaderModule(c.device,h,nullptr);});
        release(settling_pipeline''',1),
        ('void progressive_setup(Context& c,', '#include "preview_history.inc"\n\nvoid progressive_setup(Context& c,',1),
        ('((guided&&!settling)?4095u:15u)', '((guided&&!settling)?8191u:15u)',1),
        ('    std::array<VkDescriptorSetLayoutBinding,12> bindings{};', '''    require(named_shader_marker(code,bytes,92,"PR_PREVIEW_HISTORY_CURRENT_ABI",5),"history current-incoming shader required");
    VkPhysicalDeviceProperties device_properties{};vkGetPhysicalDeviceProperties(c.physical,&device_properties);
    require(device_properties.limits.maxPerStageDescriptorStorageBuffers>=4&&device_properties.limits.maxDescriptorSetStorageBuffers>=4&&
            device_properties.limits.maxPerStageResources>=16,"history descriptor limits unavailable");
    std::array<VkDescriptorSetLayoutBinding,16> bindings{};''',1),
        ('const uint32_t binding_count=guided?12u:4u;', 'const uint32_t binding_count=16u;',1),
        ('    for(uint32_t i=0;i<binding_count;++i)bindings[i]={i,i>=4?VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:\n        (i==3?VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:VK_DESCRIPTOR_TYPE_STORAGE_IMAGE),1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};',
         '''    for(uint32_t i=0;i<binding_count;++i)bindings[i]={i,i==15?VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
        (i>=12||i==3?VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:(i>=4?VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)),
        1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};''',1),
        ('c.make_buffer(state.status,32,','c.make_buffer(state.status,64,',1),
        ('std::array<VkDescriptorPoolSize,3> sizes{{{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,3},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1},{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,8}}};',
         'std::array<VkDescriptorPoolSize,4> sizes{{{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,3},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,4},{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,8},{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,1}}};',1),
        ('pool.poolSizeCount=c.spatial_info.enabled?3u:2u;', 'pool.poolSizeCount=4u;',1),
        ('VkDescriptorBufferInfo status{state.status.handle,0,32};', 'VkDescriptorBufferInfo status{state.status.handle,0,64};',1),
        ('vkCmdFillBuffer(commands,state.status.handle,0,32,0);', 'vkCmdFillBuffer(commands,state.status.handle,0,64,0);',1),
        ('status.size=32;', 'status.size=64;',1),
        ('    vkCmdFillBuffer(commands,state.status.handle,0,64,0);', '''    VkMemoryBarrier history_ready{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    history_ready.srcAccessMask=VK_ACCESS_HOST_WRITE_BIT|VK_ACCESS_SHADER_WRITE_BIT|VK_ACCESS_TRANSFER_READ_BIT;
    history_ready.dstAccessMask=VK_ACCESS_UNIFORM_READ_BIT|VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(commands,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT|VK_PIPELINE_STAGE_HOST_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0,1,&history_ready,0,nullptr,0,nullptr);
    vkCmdFillBuffer(commands,state.status.handle,0,64,0);''',1),
        ('    if(c.settling_info.enabled&&batch<15) {', '    history_record(c,frame);\n    if(c.settling_info.enabled&&batch<15) {',1),
        ('VkBufferCopy copy{0,status_offset,32};', 'VkBufferCopy copy{0,status_offset,64};',1),
    ])
    change("source/_native/resident.inc",[
        ('(c.progressive_info.shader_compatible?36:0)', '(c.progressive_info.shader_compatible?68:0)',1),
        ('    if(progressive_rgb)progressive_prepare(c,frame,width,height);',
         '    if(progressive_rgb){progressive_prepare(c,frame,width,height);history_prepare(c);}',1),
        ('    if(c.indirect_info.enabled) submitted.counts[1]=c.indirect_info.seed;',
         '    if(c.indirect_info.enabled) submitted.counts[1]=c.indirect_info.seed^(c.history_sequence*0x9e3779b9u);\n    if(progressive_rgb)history_upload(c,params);',1),
        ('std::memcpy(c.sparse_counts.data(),static_cast<const uint8_t*>(frame.readback_mapping)+progressive_status_offset,32);',
         '''std::memcpy(c.sparse_counts.data(),static_cast<const uint8_t*>(frame.readback_mapping)+progressive_status_offset,32);
        std::memcpy(c.history_counts.data(),static_cast<const uint8_t*>(frame.readback_mapping)+progressive_status_offset+32,32);''',1),
        ('    c.sparse_last_rgb=progressive_rgb?1u:0u;',
         '    if(progressive_rgb){c.progressive_state->history_pending_params=params;c.history_rendered=true;}else c.history_counts.fill(0);\n    c.sparse_last_rgb=progressive_rgb?1u:0u;',1),
        ('+(progressive_rgb?32:0)', '+(progressive_rgb?64:0)',2),
        ('progressive.status_readback_bytes+=32;', 'progressive.status_readback_bytes+=64;',1),
    ])
    # The first compute publishes a separate immutable incoming buffer; it does
    # not overwrite raw primary RGB or infer irradiance by dividing dark rho.
    change("source/_shaders/resident_spatial.comp",[
        ('layout(set=0,binding=0,rgba32f)', '''layout(constant_id=92) const uint PR_PREVIEW_HISTORY_CURRENT_ABI=1u;
layout(set=0,binding=12,std430) writeonly buffer CurrentIncoming {vec4 values[];} historyCurrent;
layout(set=0,binding=0,rgba32f)''',1),
        ('    vec4 primary=imageLoad(currentLinear,pixel);',
         '    vec4 primary=imageLoad(currentLinear,pixel);\n    vec4 historyValue=vec4(0.0);',1),
        ('PR_PREVIEW_COMPACT_COMPUTE_ABI!=1u||p.batch!=0u',
         'PR_PREVIEW_COMPACT_COMPUTE_ABI!=1u||PR_PREVIEW_HISTORY_CURRENT_ABI!=1u||p.batch!=0u',1),
        ('            linear+=rho*incoming;',
         '            historyValue=vec4(incoming,weightSum>0.0?1.0:0.0);\n            linear+=rho*incoming;',1),
        ('    imageStore(meanLinear,pixel,vec4(linear,1.0));',
         '    historyCurrent.values[uint(pixel.y)*p.width+uint(pixel.x)]=historyValue;\n    imageStore(meanLinear,pixel,vec4(linear,1.0));',1),
    ])
    write(output/"source/_native/preview_history.inc",(HERE/"history_native.inc").read_text(encoding="utf-8"))
    write(output/"source/_shaders/preview_history.comp",(HERE/"history.comp").read_text(encoding="utf-8"))
    change("source/_native/CMakeLists.txt",[
        ('add_custom_target(native_shaders ALL DEPENDS ${shader_outputs})',
         'shader(preview_history.comp preview_history.comp)\nadd_custom_target(native_shaders ALL DEPENDS ${shader_outputs})',1)])
    with (output/"source/_native/renderer.cpp").open("a",encoding="utf-8") as stream:
        stream.write((HERE/"history_exports.inc").read_text(encoding="utf-8"))
    policy["history"]=dict(schema="phymeo-preview-compact-history-v1",scope="static-geometry-moving-camera",
        sequence="base_seed XOR uint32(successful_RGB_sequence * 0x9e3779b9); reset sequence zero",
        eligible="current K25 reconstruction has at least one actual traced low-anchor support; SH fallback age zero",
        rejection=dict(relative_depth=.003,normal_dot=.995,roundtrip_px=.75,clamp_radius=2,clamp_min_support=3,age_cap=8),
        dynamic="any node pose changes reset all; no dynamic-Panda history benefit claimed",
        resources="80 extra device buffer bytes per full pixel plus 176B UBO; optional 32B/pixel mapped debug buffer",
        bias="nearest reprojection and neighborhood clamp/age-capped EMA are biased; reconstructed samples overlap; age is not spp",
        reset="scene/world/epoch/phase/camera-id/lens/extent/nodes/lights/exposure/signal/tick-time gaps/camera cuts/explicit",
        sensors="current primary sensors untouched",markers=[92,93])
    policy["compact"]["history"]="separate GPU reprojection after immutable current compact reconstruction"
    policy["unsupported"]=[x for x in policy.get("unsupported",[]) if x!="temporal accumulation"]+[
        "history reuse across any instance transform change", "unbounded history", "unverified spatial/pool overlays"]
    result=deepcopy(manifest)
    result.update(contract=policy,history_parent_prepared=str(parent),history_parent_manifest_sha256=sha256(raw).hexdigest(),
        history_builder_sha256=digest(Path(__file__)),
        generated_sha256={str(p.relative_to(output)):digest(p) for p in sorted(output.rglob("*")) if p.is_file()},
        state="history sources only; no executed build or device validation implied")
    write(output/"source-manifest.json",json.dumps(result,indent=2)+"\n")
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact-prepared",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();prepare(args.compact_prepared,args.output)


if __name__=="__main__":main()
