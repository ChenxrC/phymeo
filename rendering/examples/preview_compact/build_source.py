"""Create a new compact GI build from a hash-bound sparse source package.

No compiler or project execution is started here. Input package is immutable.
"""
import argparse
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import shutil

HERE=Path(__file__).resolve().parent


def low_extent(width,height):
    if any(type(x) is not int or x<1 for x in (width,height)):
        raise ValueError("positive integer extent required")
    return (width+3)//4,(height+3)//4


def viewport_axis(size):
    if type(size) is not int or size<1:raise ValueError("positive extent required")
    return -.125,size/4


def anchor(cell,extent):
    value=tuple(4*x+2 for x in cell)
    return value if all(0<=x<n for x,n in zip(value,extent)) else None


def contract(source):
    if (source.get("schema")!="phymeo-preview-sparse-v1" or source.get("tile")!=4 or
        source.get("paths_per_anchor") not in (2,4,16) or source.get("bounces")!=1 or
        source.get("neighbor_candidates")!=25 or source.get("reconstruction_revision")!=2 or
        source.get("estimator") not in ("radiance","visibility-ratio") or
        source.get("area_light_grid") not in (2,4,8)):
        raise ValueError("compact requires tile4, K25/R2, one-hop bounded sparse source")
    value=deepcopy(source)
    value["compact"]=dict(schema="phymeo-preview-compact-v1",tile=4,history="none",
        full_attributes="primary location4: RGBA32F SH.rgb and oct10x2 shading-normal numeric integer; rho remains primary alpha",
        guide_attributes="own independent low-resolution forward-depth/ID/valid/Ns; never relabel full-rate attributes as low hits",
        viewport="(-1/8,-1/8,W/4,H/4), target ceil(W/4) x ceil(H/4)",
        random_pixel="full integer anchor 4*low_pixel+2",
        odd="out-of-bounds regular anchors are invalid, never clamped; full receivers use supported in-bounds neighbors or own SH",
        footprint="K25 receiver-local ID/depth/normal gates; low hit also checked against actual full anchor attributes",
        raster_limit="analytic sample centers match; finite hardware raster precision can differ, rejected support is reported",
        counter_partition="reconstructed + fallback + nondiffuse = full visible; traced_anchor_pixels independently counts valid low traced anchors")
    return value


def _hash(path):return sha256(path.read_bytes()).hexdigest()
def _write(path,text):path.write_text(text,encoding="utf-8",newline="\n")


def prepare(parent,output):
    parent=Path(parent).resolve();out=Path(output).resolve()
    raw=(parent/"source-manifest.json").read_bytes();manifest=json.loads(raw)
    policy=contract(manifest["contract"])
    for relative,expected in manifest["generated_sha256"].items():
        path=parent/relative
        if _hash(path)!=expected:raise ValueError("sparse source changed: "+relative)
    out.mkdir(parents=True,exist_ok=False)
    shutil.copytree(parent/"source",out/"source")
    shutil.copytree(parent/"runtime",out/"runtime")
    native=out/"source/_native";shader=out/"source/_shaders"

    def change(relative,edits):
        path=out/relative;text=path.read_text(encoding="utf-8")
        for old,new,count in edits:
            if text.count(old)!=count:
                raise ValueError(f"compact source anchor drift {relative}: expected {count}: {old[:90]}")
            text=text.replace(old,new)
        _write(path,text)

    change("source/_native/renderer.cpp",[
        ('require(sparse_shader&&progressive_info.shader_compatible&&ray_query,',
         'require(named_shader_marker(frag,frag_bytes,86,"PR_PREVIEW_COMPACT_PRIMARY_ABI"),"compact primary ABI required");\n        require(sparse_shader&&progressive_info.shader_compatible&&ray_query,',1),
        ('p.limits.maxColorAttachments >= 4 && p.limits.maxFragmentOutputAttachments >= 4',
         'p.limits.maxColorAttachments >= 5 && p.limits.maxFragmentOutputAttachments >= 5',1),
        ('std::array<VkAttachmentDescription, 5> attachments{};', 'std::array<VkAttachmentDescription, 6> attachments{};',1),
        ('std::array<VkAttachmentReference, 4> colors{};', 'std::array<VkAttachmentReference, 5> colors{};',1),
        ('for (uint32_t i=0; i<5; ++i) {', 'for (uint32_t i=0; i<6; ++i) {',1),
        ('if (i<4) colors[i]={i,VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};',
         'if (i<4) colors[i]={i,VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};\n            if(i==5)colors[4]={5,VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};',1),
        ('subpass.colorAttachmentCount=4; subpass.pColorAttachments=colors.data();',
         'subpass.colorAttachmentCount=5; subpass.pColorAttachments=colors.data();',1),
        ('pass.attachmentCount=5; pass.pAttachments=attachments.data();', 'pass.attachmentCount=6; pass.pAttachments=attachments.data();',1),
        ('std::array<VkPipelineColorBlendAttachmentState,4> blends{};', 'std::array<VkPipelineColorBlendAttachmentState,5> blends{};',1),
        ('blending.attachmentCount=4; blending.pAttachments=blends.data();', 'blending.attachmentCount=5; blending.pAttachments=blends.data();',1),
        ('std::array<Image,6> images{};', 'std::array<Image,12> images{};',1),
        ('require(named_shader_marker(guide,guide_bytes,83,',
         'require(named_shader_marker(guide,guide_bytes,87,"PR_PREVIEW_COMPACT_GUIDE_ABI"),"compact guide ABI required");\n        require(named_shader_marker(code,bytes,88,"PR_PREVIEW_COMPACT_COMPUTE_ABI",5),"compact compute ABI required");\n        require(named_shader_marker(guide,guide_bytes,83,',1),
    ])
    # Full-size location4 lives at image6; guide has its own five attachments.
    change("source/_native/resident.inc",[
        ('''    for (uint32_t i = 0; i < (c.spatial_info.enabled&&!c.settling_info.enabled?6u:5u); ++i) {
        c.make_image(frame->images[i], width, height, c.attachment_format(i), i == 4, VK_SAMPLE_COUNT_1_BIT,
            (i==0 && c.progressive_info.shader_compatible?VK_IMAGE_USAGE_STORAGE_BIT:0) | (c.spatial_info.enabled && i!=0 && i!=4?VK_IMAGE_USAGE_SAMPLED_BIT:0));
        views[i] = frame->images[i].view;
    }''','''    for(uint32_t i=0;i<6;++i) {
        const uint32_t image=i==5?6u:i;
        c.make_image(frame->images[image],width,height,c.attachment_format(i),i==4,VK_SAMPLE_COUNT_1_BIT,
            i==0?VK_IMAGE_USAGE_STORAGE_BIT:(i==4?0:VK_IMAGE_USAGE_SAMPLED_BIT));
        views[i]=frame->images[image].view;
    }
    const uint32_t low_width=(width+3)/4,low_height=(height+3)/4;
    for(uint32_t i=0;i<5;++i)
        c.make_image(frame->images[7+i],low_width,low_height,i==0?VK_FORMAT_R32G32B32A32_SFLOAT:FORMATS[i],i==4,
            VK_SAMPLE_COUNT_1_BIT,i==4?0:VK_IMAGE_USAGE_SAMPLED_BIT);''',1),
        ('framebuffer.renderPass = c.render_pass; framebuffer.attachmentCount = 5;',
         'framebuffer.renderPass = c.render_pass; framebuffer.attachmentCount = 6;',1),
        ('pass.renderArea.extent = {width, height}; pass.clearValueCount = 5;',
         'pass.renderArea.extent = {width, height}; pass.clearValueCount = 6;',1),
    ])
    # Sky changes RGB only; fifth primary output remains its clear value.
    change("source/_native/resident_environment.inc",[
        ('std::array<VkPipelineColorBlendAttachmentState,4> attachments{};',
         'std::array<VkPipelineColorBlendAttachmentState,5> attachments{};',1),
        ('blending.attachmentCount=4; blending.pAttachments=attachments.data();',
         'blending.attachmentCount=5; blending.pAttachments=attachments.data();',1),
    ])
    change("source/_native/resident_spatial.inc",[
        ('properties.limits.maxPerStageDescriptorSamplers>=4&&properties.limits.maxPerStageDescriptorSampledImages>=4',
         'properties.limits.maxPerStageDescriptorSamplers>=8&&properties.limits.maxPerStageDescriptorSampledImages>=8',1),
        ('a.loadOp=(i==0||i==4)?VK_ATTACHMENT_LOAD_OP_CLEAR:VK_ATTACHMENT_LOAD_OP_LOAD;',
         'a.loadOp=VK_ATTACHMENT_LOAD_OP_CLEAR;',1),
        ('a.initialLayout=i==0?VK_IMAGE_LAYOUT_UNDEFINED:(i==4?VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);',
         'a.initialLayout=VK_IMAGE_LAYOUT_UNDEFINED;',1),
        ('blends[0].colorWriteMask=15; // Preserve every original sensor bit.',
         'for(auto& blend:blends)blend.colorWriteMask=15; // Write independent low-resolution hit attributes.',1),
        ('    auto& state=*c.progressive_state;\n    if(!frame.guide_framebuffer){',
         '    auto& state=*c.progressive_state;\n    const uint32_t full_width=width,full_height=height;\n    width=(width+3)/4;height=(height+3)/4;\n    if(!frame.guide_framebuffer){',1),
        ('const std::array<VkImageView,5> views{c.settling_info.enabled?state.retained_guide.view:frame.images[5].view,frame.images[1].view,frame.images[2].view,frame.images[3].view,frame.images[4].view};',
         'const std::array<VkImageView,5> views{frame.images[7].view,frame.images[8].view,frame.images[9].view,frame.images[10].view,frame.images[11].view};',1),
        ('std::array<VkClearValue,5> clears{};clears[4].depthStencil.depth=1;',
         'std::array<VkClearValue,5> clears{};clears[4].depthStencil.depth=1;clears[1].color.float32[0]=std::numeric_limits<float>::infinity();',1),
        ('    vkCmdBindPipeline(frame.commands,VK_PIPELINE_BIND_POINT_GRAPHICS,state.guide_pipeline);',
         '''    vkCmdBindPipeline(frame.commands,VK_PIPELINE_BIND_POINT_GRAPHICS,state.guide_pipeline);
    VkViewport compact_viewport{-.125f,-.125f,static_cast<float>(full_width)*.25f,static_cast<float>(full_height)*.25f,0,1};
    VkRect2D compact_scissor{{0,0},{width,height}};
    vkCmdSetViewport(frame.commands,0,1,&compact_viewport);
    vkCmdSetScissor(frame.commands,0,1,&compact_scissor);''',1),
    ])
    change("source/_native/resident_progressive.inc",[
        ('((guided&&!settling)?255u:15u)', '((guided&&!settling)?4095u:15u)',1),
        ('std::array<VkDescriptorSetLayoutBinding,8> bindings{};', 'std::array<VkDescriptorSetLayoutBinding,12> bindings{};',1),
        ('const uint32_t binding_count=guided?8u:4u;', 'const uint32_t binding_count=guided?12u:4u;',1),
        ('{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,4}', '{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,8}',1),
        ('const uint32_t indices[4]={5,1,2,3};', 'const uint32_t indices[8]={7,1,2,3,6,8,9,10};',2),
        ('std::array<VkDescriptorImageInfo,4> sampled{};', 'std::array<VkDescriptorImageInfo,8> sampled{};',1),
        ('std::array<VkWriteDescriptorSet,4> extra{};', 'std::array<VkWriteDescriptorSet,8> extra{};',1),
        ('for(uint32_t i=0;i<4;++i){\n            sampled[i]', 'for(uint32_t i=0;i<8;++i){\n            sampled[i]',1),
        ('vkUpdateDescriptorSets(c.device,4,extra.data(),0,nullptr);', 'vkUpdateDescriptorSets(c.device,8,extra.data(),0,nullptr);',1),
        ('std::array<VkImageMemoryBarrier,4> guide_barriers{};', 'std::array<VkImageMemoryBarrier,8> guide_barriers{};',1),
        ('for(uint32_t i=0;i<4;++i)guide_barriers[i]', 'for(uint32_t i=0;i<8;++i)guide_barriers[i]',1),
        ('0,0,nullptr,0,nullptr,4,guide_barriers.data());', '0,0,nullptr,0,nullptr,8,guide_barriers.data());',2),
        ('void spatial_read(Context& c,void* output,size_t bytes) {',
         'void spatial_read(Context& c,void* output,size_t bytes) {\n    require(false,"compact debug reads require pr_preview_compact_read and actual low extent");',1),
    ])
    # Admit the two explicitly recorded C guide-return spellings. The later
    # one additionally carries ratio outcome flags; do not erase those bits.
    fragment=(shader/"resident.frag").read_text(encoding="utf-8")
    alternatives=(
        '    rgb=vec4(incoming,float(sparsePackNormal(N,traced)|sparseOutcomes));return;',
        '    rgb=vec4(incoming,float(sparsePackNormal(N,traced)));return;',
    )
    matched=[line for line in alternatives if fragment.count(line)==1]
    if len(matched)!=1:raise ValueError("unrecognized parent guide output declaration")
    guide_return=matched[0]
    # Fragment source has one shared main; location4 exists only on primary.
    change("source/_shaders/resident.frag",[
        ('layout(location=0) out vec4 rgb;', '''layout(location=0) out vec4 rgb;
#ifdef PR_PREVIEW_SPARSE_MAIN
layout(location=4) out vec4 compactAttributes;
layout(constant_id=86) const uint PR_PREVIEW_COMPACT_PRIMARY_ABI=1u;
#endif
#ifdef PR_PREVIEW_SPARSE_GUIDE
layout(constant_id=87) const uint PR_PREVIEW_COMPACT_GUIDE_ABI=1u;
#endif''',1),
        ('    forwardDepth = -(u.view*vec4(worldPosition, 1.0)).z;', '''#ifdef PR_PREVIEW_SPARSE_GUIDE
    ivec2 fullAnchor=ivec2(gl_FragCoord.xy)*4+ivec2(2);
    if(any(greaterThanEqual(fullAnchor,ivec2(u.clipping.xy)))) {
        rgb=vec4(0.0);forwardDepth=uintBitsToFloat(0x7f800000u);instanceId=0u;valid=0u;return;
    }
#endif
    forwardDepth = -(u.view*vec4(worldPosition, 1.0)).z;''',1),
        ('&&sparseAnchor(ivec2(gl_FragCoord.xy));',';',1),
        (guide_return,'    if(PR_PREVIEW_COMPACT_GUIDE_ABI!=1u)incoming=vec3(importanceInvalid());\n'+guide_return,1),
        ('    vec3 sparseOutputRho=(indirectActive&&reflectionFaceValid)?rho:vec3(0.0);',
         '''#ifdef PR_PREVIEW_SPARSE_MAIN
    compactAttributes=vec4(shadingFacesView?sparseFallback(N):vec3(0.0),float(sparsePackNormal(N,false)));
    if(PR_PREVIEW_COMPACT_PRIMARY_ABI!=1u)color=vec3(importanceInvalid());
#endif
    vec3 sparseOutputRho=(indirectActive&&reflectionFaceValid)?rho:vec3(0.0);''',1),
    ])
    # Both estimators retain the original full-resolution random-key coordinate.
    for relative in ("resident_indirect.glsl","preview_sparse_ratio.glsl"):
        if (shader/relative).exists():
            change("source/_shaders/"+relative,[('uvec2 pixel=uvec2(gl_FragCoord.xy);',
                '''uvec2 pixel=uvec2(gl_FragCoord.xy);
#ifdef PR_PREVIEW_SPARSE_GUIDE
    pixel=pixel*4u+uvec2(2u);
#endif''',1)])
    compute=(HERE/"compact_reconstruct.comp").read_text(encoding="utf-8")
    profile=manifest["profile_marker"]
    compute=compute.replace("PROFILE_MARKER",profile)
    _write(shader/"resident_spatial.comp",compute)
    _write(native/"preview_compact.inc",(HERE/"compact_native.inc").read_text(encoding="utf-8"))
    with (native/"resident.inc").open("a",encoding="utf-8") as stream:
        stream.write('\n#include "preview_compact.inc"\n')
    with (native/"renderer.cpp").open("a",encoding="utf-8") as stream:
        stream.write('''
PR_API uint32_t pr_preview_compact_abi_version(){return 1;}
PR_API int pr_preview_compact_info(void* handle,uint32_t* out,size_t bytes){
    try{require(handle&&out&&bytes==64,"invalid compact info");auto& c=*static_cast<Context*>(handle);
        const uint32_t w=c.resident_state?c.resident_state->width:0,h=c.resident_state?c.resident_state->height:0;
        const uint32_t value[16]={1,1,4,w,h,(w+3)/4,(h+3)/4,c.sparse_last_rgb,c.sparse_captures,c.sparse_sensor_captures,c.sparse_captures,5,8,0,0,0};
        std::memcpy(out,value,64);return 0;}catch(const std::exception& e){last_error=e.what();return -1;}
}
PR_API int pr_preview_compact_read(void* handle,uint32_t plane,void* out,size_t bytes){
    try{require(handle,"null compact context");compact_read(*static_cast<Context*>(handle),plane,out,bytes);return 0;}
    catch(const std::exception& e){last_error=e.what();return -1;}
}
''')
    new=deepcopy(manifest)
    new.update(contract=policy,parent_prepared=str(parent),parent_manifest_sha256=sha256(raw).hexdigest(),
        compact_builder_sha256=_hash(Path(__file__)),
        generated_sha256={str(p.relative_to(out)):_hash(p) for p in sorted(out.rglob("*")) if p.is_file()},
        state="compact source generated only; no build/device result implied")
    _write(out/"source-manifest.json",json.dumps(new,indent=2)+"\n")
    return new


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sparse-prepared",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    prepare(args.sparse_prepared,args.output)


if __name__=="__main__":main()
