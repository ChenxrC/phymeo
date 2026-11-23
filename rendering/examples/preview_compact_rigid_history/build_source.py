"""Exclusive rigid-history variant of a hash-bound, validated static D package."""
import argparse
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import shutil

HERE=Path(__file__).resolve().parent


def digest(path):return sha256(path.read_bytes()).hexdigest()


def prepare(parent,output):
    parent=Path(parent).resolve();output=Path(output).resolve()
    raw=(parent/"source-manifest.json").read_bytes();manifest=json.loads(raw)
    policy=deepcopy(manifest["contract"])
    if policy.get("history",{}).get("schema")!="phymeo-preview-compact-history-v1" or "rigid_history" in policy:
        raise ValueError("plain static GPU history package required")
    for relative,expected in manifest["generated_sha256"].items():
        if digest(parent/relative)!=expected:raise ValueError("static history source changed: "+relative)
    output.mkdir(parents=True,exist_ok=False)
    shutil.copytree(parent/"source",output/"source");shutil.copytree(parent/"runtime",output/"runtime")
    def write(path,text):path.write_text(text,encoding="utf-8",newline="\n")
    def change(relative,edits):
        path=output/relative;text=path.read_text(encoding="utf-8")
        for old,new,count in edits:
            if text.count(old)!=count:raise ValueError(f"rigid source anchor drift: {relative}: {old[:80]}")
            text=text.replace(old,new)
        write(path,text)
    change("source/_native/renderer.cpp",[
        ('    std::array<uint32_t,8> history_counts{};', '''    std::array<uint32_t,8> history_counts{};
    std::array<uint32_t,4> rigid_counts{};
    uint32_t rigid_instance_count=0,rigid_moved_instances=0;
    uint64_t rigid_upload_bytes=0;''',1),
        ('s->height*80+176','s->height*112+176',1),
        ('s?s->height:0,8,s?', 's?s->height:0,4,s?',1),
    ])
    change("source/_native/resident_progressive.inc",[
        ('    Buffer history_current{},history_pixels[2]{},history_uniform{},history_debug{};',
         '    Buffer history_current{},history_pixels[2]{},history_uniform{},history_debug{},rigid_motion{};\n    std::vector<GpuTransform> rigid_previous,rigid_pending;',1),
        ('&history_uniform,&history_debug}', '&history_uniform,&history_debug,&rigid_motion}',1),
        ('history_valid=false;history_read_slot=0;', 'history_valid=false;history_read_slot=0;rigid_previous.clear();rigid_pending.clear();',1),
        ('maxPerStageDescriptorStorageBuffers>=4','maxPerStageDescriptorStorageBuffers>=5',1),
        ('maxDescriptorSetStorageBuffers>=4','maxDescriptorSetStorageBuffers>=5',1),
        ('maxPerStageResources>=16','maxPerStageResources>=17',1),
        ('VkDescriptorSetLayoutBinding,16>','VkDescriptorSetLayoutBinding,17>',1),
        ('binding_count=16u','binding_count=17u',1),
        ('c.make_buffer(state.status,64,','c.make_buffer(state.status,80,',1),
        ('{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,4}','{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,5}',1),
        ('status{state.status.handle,0,64}','status{state.status.handle,0,80}',1),
        ('vkCmdFillBuffer(commands,state.status.handle,0,64,0)','vkCmdFillBuffer(commands,state.status.handle,0,80,0)',1),
        ('status.size=64','status.size=80',1),
        ('VkBufferCopy copy{0,status_offset,64}','VkBufferCopy copy{0,status_offset,80}',1),
    ])
    change("source/_native/preview_history.inc",[
        ('interface.bindings==0xf1efu','interface.bindings==0x1f1efu',1),
        ('93,"PR_PREVIEW_COMPACT_HISTORY_ABI"','97,"PR_PREVIEW_RIGID_HISTORY_ABI"',1),
        ('pixels*32','pixels*48',4),
        ('void history_upload(Context& c,const PrParams& params) {',
         '#include "preview_rigid.inc"\n\nvoid history_upload(Context& c,const PrParams& params,const std::vector<GpuTransform>& transforms) {\n    rigid_upload(c,transforms);',1),
        ('uniform.controls[3]=8;', 'uniform.controls[3]=4;',1),
        ('s.history_params=s.history_pending_params;', 's.history_params=s.history_pending_params;s.rigid_previous=s.rigid_pending;',1),
        ('*(plane==0?16:32)', '*(plane==0?16:48)',1),
        ('s.height*32,VK_BUFFER_USAGE_TRANSFER_DST_BIT','s.height*48,VK_BUFFER_USAGE_TRANSFER_DST_BIT',1),
    ])
    change("source/_native/resident.inc",[
        ('shader_compatible?68:0','shader_compatible?84:0',1),
        ('history_upload(c,params);','history_upload(c,params,transforms);',1),
        ('std::memcpy(c.history_counts.data(),static_cast<const uint8_t*>(frame.readback_mapping)+progressive_status_offset+32,32);',
         'std::memcpy(c.history_counts.data(),static_cast<const uint8_t*>(frame.readback_mapping)+progressive_status_offset+32,32);\n        std::memcpy(c.rigid_counts.data(),static_cast<const uint8_t*>(frame.readback_mapping)+progressive_status_offset+64,16);',1),
        ('else c.history_counts.fill(0);','else {c.history_counts.fill(0);c.rigid_counts.fill(0);}',1),
        ('+(progressive_rgb?64:0)','+(progressive_rgb?80:0)',2),
        ('progressive.status_readback_bytes+=64','progressive.status_readback_bytes+=80',1),
    ])
    change("source/_shaders/preview_history.comp",[
        ('constant_id=93','constant_id=97',1),
        ('PR_PREVIEW_COMPACT_HISTORY_ABI','PR_PREVIEW_RIGID_HISTORY_ABI',2),
        ('uint eligible,accepted,rejectedProjection,rejectedGeometry,rejectedSupport,clamped,ageSum,resetPixels;',
         'uint eligible,accepted,rejectedProjection,rejectedGeometry,rejectedSupport,clamped,ageSum,resetPixels;\n    uint rejectedLighting,rejectedIncoming,movingAccepted,lookupMisses;',1),
        ('struct HistoryPixel {vec4 incomingAge;uvec4 geometry;};',
         'struct HistoryPixel {vec4 incomingAge;uvec4 geometry;vec4 lighting;};',1),
        ('#include "resident_display.glsl"','#include "resident_display.glsl"\n#include "preview_rigid_common.glsl"',1),
        ('        vec3 normal=normalFrom(code);', '''        vec3 normal=normalFrom(code);
        int slot=findRigid(id);
        if(slot<0){atomicAdd(status.lookupMisses,1u);atomicOr(status.invalid,2048u);return;}
        RigidPair pair=rigid.pairs[uint(slot)];''',1),
        ('            vec3 projected=projectPoint(worldPoint(vec2(pixel)+.5,z,h.view,h.intrinsics),h.previousView,h.previousIntrinsics);',
         '''            vec3 point=worldPoint(vec2(pixel)+.5,z,h.view,h.intrinsics);
            vec3 oldPoint=(pair.previousWorld*pair.local*vec4(point,1.0)).xyz;
            vec3 projected=projectPoint(oldPoint,h.previousView,h.previousIntrinsics);
            vec3 oldExpectedNormal=normalize(mat3(pair.previousWorld)*mat3(pair.local)*normal);''',1),
        ('bool geometry=old.geometry.z==id', 'bool geometry=pair.identity.z==0u&&old.geometry.z==id',1),
        ('dot(normal,normalFrom(old.geometry.y))>=.995','dot(oldExpectedNormal,normalFrom(old.geometry.y))>=.995',1),
        ('vec3 roundtrip=projectPoint(worldPoint(vec2(q)+.5,oldZ,h.previousView,h.previousIntrinsics),h.view,h.intrinsics);',
         'vec3 previousPoint=worldPoint(vec2(q)+.5,oldZ,h.previousView,h.previousIntrinsics);\n                    vec3 roundtrip=projectPoint((pair.world*pair.previousLocal*vec4(previousPoint,1.0)).xyz,h.view,h.intrinsics);',1),
        ('                    if(support<3u)atomicAdd(status.rejectedSupport,1u);',
         '''                    bool lightStable=old.lighting.a==base.a&&guideMatches(old.lighting.rgb,base.rgb);
                    bool incomingStable=incomingMatches(old.incomingAge.rgb,low,high);
                    if(!lightStable){atomicAdd(status.rejectedSupport,1u);atomicAdd(status.rejectedLighting,1u);}
                    else if(!incomingStable){atomicAdd(status.rejectedSupport,1u);atomicAdd(status.rejectedIncoming,1u);}
                    else if(support<3u)atomicAdd(status.rejectedSupport,1u);''',1),
        ('                        atomicAdd(status.accepted,1u);',
         '                        atomicAdd(status.accepted,1u);\n                        if(pair.identity.y!=0u)atomicAdd(status.movingAccepted,1u);',1),
        ('    nextHistory.pixels[index].incomingAge=vec4(filtered,age);',
         '    nextHistory.pixels[index].lighting=base;\n    nextHistory.pixels[index].incomingAge=vec4(filtered,age);',1),
    ])
    write(output/"source/_native/preview_rigid.inc",(HERE/"rigid_native.inc").read_text(encoding="utf-8"))
    write(output/"source/_shaders/preview_rigid_common.glsl",(HERE/"rigid_common.glsl").read_text(encoding="utf-8"))
    with (output/"source/_native/renderer.cpp").open("a",encoding="utf-8") as stream:
        stream.write('''
PR_API uint32_t pr_preview_rigid_history_abi_version(){return 1;}
PR_API int pr_preview_rigid_history_info(void* handle,uint32_t* out,size_t bytes){
    try{require(handle&&out&&bytes==64,"rigid history info extent");auto& c=*static_cast<Context*>(handle);
        const uint32_t values[8]={1,c.history_enabled,c.rigid_instance_count,c.rigid_moved_instances,4,c.history_captures,48,97};
        std::memset(out,0,64);std::memcpy(out,values,32);std::memcpy(out+8,c.rigid_counts.data(),16);
        std::memcpy(out+12,&c.rigid_upload_bytes,8);return 0;
    }catch(const std::exception& e){last_error=e.what();return -1;}
}
''')
    policy["history"]["schema"]="phymeo-preview-rigid-history-v1"
    policy["history"].update(scope="rigid-primary-short-history",dynamic="real per-ID primary rigid correspondence; secondary visibility is locally unobserved",
        resources="112 B per full pixel +176B camera UBO +16+272N rigid table, plus optional48B/pixel debug buffer")
    policy["history"]["rejection"]["age_cap"]=4
    policy["history"]["markers"]=[92,97]
    policy["history"]["reset"]="scene/world/epoch/phase/camera-id/lens/extent/lights/exposure/signal/presentation-source/camera-cuts/explicit; bound adjacent presentation required; per-instance motion is reprojected"
    policy["rigid_history"]=dict(schema="phymeo-preview-compact-rigid-history-v1",marker=97,
        motion="same FP32 current matrices used by raster/TLAS; previous successful publication matrices; sorted uint32 ID lookup",
        clock="bound adjacent presented records with exact source SHA/index/current frame digest/previous full key,time,digest; source ticks need not advance by one",
        confidence=dict(age_cap=4,primary_relative=.2,primary_absolute=1e-5,incoming_range_margin=.5,incoming_high_margin=.1,
                        incoming_floor=1e-5,material_rho="same encoded uint24",motion_translation_cut_m=.25,motion_rotation_cut_deg=20),
        limitations="Primary reprojection, current direct/specular changes, and current incoming neighborhood cannot prove hidden secondary visibility unchanged; bounded biased history, not physical or independent spp.")
    policy["unsupported"]=[x for x in policy.get("unsupported",[]) if x!="history reuse across any instance transform change"]+[
        "non-rigid/scaled/reflected instance transforms","unbound presented-frame continuity"]
    result=deepcopy(manifest);result.update(contract=policy,rigid_parent_prepared=str(parent),
        rigid_parent_manifest_sha256=sha256(raw).hexdigest(),rigid_builder_sha256=digest(Path(__file__)),
        generated_sha256={str(p.relative_to(output)):digest(p) for p in sorted(output.rglob("*")) if p.is_file()},
        state="rigid history source candidate; not yet a device result")
    write(output/"source-manifest.json",json.dumps(result,indent=2)+"\n")
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history-prepared",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();prepare(args.history_prepared,args.output)


if __name__=="__main__":main()
