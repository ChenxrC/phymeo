"""Add geometry-aware contour resolve to a copied lf100 source tree."""
from pathlib import Path

from preview_lowres_filter._helpers import once


HERE = Path(__file__).resolve().parent
MARKER = 104
SHADER = "preview_history.comp.spv"


def policy():
    return {
        "schema": "phymeo-preview-contour-aa-v1",
        "marker": MARKER,
        "visibility": "nearest opaque TLAS hit at four rotated subpixel positions within the camera clip planes",
        "scope": "pixels with a different instance or background in their clipped 3x3 center neighborhood",
        "shading": "per-surface center radiance; a different hit needs a local ID match and forward depth within 3 percent",
        "unsupported": "retain the original center for the entire pixel when a hit has no supported donor",
        "resolve": "four equal scene-linear HDR weights before exposure and the single display transform",
        "sensors": "original deterministic center samples",
        "history": "center-sampled history stays independent of color resolve",
    }


def history_shader(source):
    source = once(source, "#version 450", "#version 450\n#extension GL_EXT_ray_query : require")
    source = once(source, "    uint rejectedLighting,rejectedIncoming,movingAccepted,lookupMisses;",
                  "    uint rejectedLighting,rejectedIncoming,movingAccepted,lookupMisses;\n"
                  "    uint contourCandidates,contourResolved,contourUnsupported,contourRays;")
    source = once(source, "writeonly buffer Next", "buffer Next")
    source = once(source, "} h;", "    vec4 clipping; // width, height, near and far\n} h;")
    start = source.index("    if(!finite3(color))")
    publish = source[start:]
    if not publish.rstrip().endswith("}"):
        raise ValueError("history color publication changed")
    source = source[:start] + "    publishColor(pixel,color);\n}\n"
    source = once(source, "void main() {", "void publishColor(ivec2 pixel,vec3 color) {\n" +
                  # The high mode bit only selects the contour dispatch.
                  once(publish, "float(p.mode)", "float(p.mode&255u)") +
                  '\n#include "preview_contour.glsl"\nvoid main() {')
    return once(source, "    uint index=uint(pixel.y)*p.width+uint(pixel.x);",
                """    if(PR_PREVIEW_CONTOUR_AA_ABI!=1u){atomicOr(status.invalid,4096u);return;}
    if((p.mode&512u)!=0u){resolveContour(pixel,extent);return;}
    uint index=uint(pixel.y)*p.width+uint(pixel.x);""")


def history_native(source):
    source = once(source, "uint32_t controls[4];", "uint32_t controls[4]; float clipping[4];")
    source = once(source, "sizeof(HistoryUniform)==176", "sizeof(HistoryUniform)==192")
    source = once(source, "    c.upload(s.history_uniform,", "    std::memcpy(uniform.clipping,params.clipping,sizeof(uniform.clipping));\n    c.upload(s.history_uniform,")
    source = once(source, "interface.bindings==0x1f1efu", "interface.bindings==0xdf1efu")
    source = once(source, 'named_shader_marker(code,bytes,97,', 'named_shader_marker(code,bytes,104,"PR_PREVIEW_CONTOUR_AA_ABI",5)&&\n            named_shader_marker(code,bytes,97,')
    return source


BIND_CONTOUR = """    VkWriteDescriptorSetAccelerationStructureKHR acceleration{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR};
    acceleration.accelerationStructureCount=1;
    acceleration.pAccelerationStructures=&c.resident_state->acceleration->top->handle;
    VkWriteDescriptorSet contourScene{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    contourScene.pNext=&acceleration;contourScene.dstSet=state.set;contourScene.dstBinding=18;
    contourScene.descriptorType=VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;contourScene.descriptorCount=1;
    VkDescriptorBufferInfo slots{c.resident_state->instances_buffer.handle,0,VK_WHOLE_SIZE};
    VkWriteDescriptorSet contourSlots{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    contourSlots.dstSet=state.set;contourSlots.dstBinding=19;contourSlots.descriptorCount=1;
    contourSlots.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;contourSlots.pBufferInfo=&slots;
    const VkWriteDescriptorSet contourWrites[]{contourScene,contourSlots};
    vkUpdateDescriptorSets(c.device,2,contourWrites,0,nullptr);"""


def progressive_native(source):
    for old, new in (
        ("maxPerStageDescriptorStorageBuffers>=6", "maxPerStageDescriptorStorageBuffers>=7"),
        ("maxDescriptorSetStorageBuffers>=6", "maxDescriptorSetStorageBuffers>=7"),
        ("maxPerStageResources>=18", "maxPerStageResources>=20"),
        ("VkDescriptorSetLayoutBinding,18>", "VkDescriptorSetLayoutBinding,20>"),
        ("binding_count=18u", "binding_count=20u"),
        ("{i,i==15?", "{i,i==18?VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR:i==15?"),
        ("VkDescriptorPoolSize,4>", "VkDescriptorPoolSize,5>"),
        ("{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,6}", "{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,7}"),
        ("{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,1}}}", "{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,1},{VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,1}}}"),
        ("pool.poolSizeCount=4u", "pool.poolSizeCount=5u"),
        ("    VkDescriptorBufferInfo low_info", BIND_CONTOUR + "\n    VkDescriptorBufferInfo low_info"),
        ("state.status,80,", "state.status,96,"),
        ("status{state.status.handle,0,80}", "status{state.status.handle,0,96}"),
        ("state.status.handle,0,80,0", "state.status.handle,0,96,0"),
        ("status.size=80;", "status.size=96;"),
        ("copy{0,status_offset,80}", "copy{0,status_offset,96}"),
        ("    history_record(c,frame);", """    history_record(c,frame);
    // All per-surface center shading must be visible before neighboring pixels
    // can read it. The second pass only writes final HDR/display images.
    VkMemoryBarrier contourReady{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    contourReady.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT;
    contourReady.dstAccessMask=VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(commands,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0,1,&contourReady,0,nullptr,0,nullptr);
    push.mode|=512u;
    vkCmdPushConstants(commands,state.pipeline_layout,VK_SHADER_STAGE_COMPUTE_BIT,0,20,&push);
    vkCmdDispatch(commands,(state.width+7)/8,(state.height+7)/8,1);"""),
    ):
        source = once(source, old, new)
    return source


def resident_native(source):
    old = "std::memcpy(c.rigid_counts.data(),static_cast<const uint8_t*>(frame.readback_mapping)+progressive_status_offset+64,16);"
    for before, after in (
        ("c.progressive_info.shader_compatible?84:0", "c.progressive_info.shader_compatible?100:0"),
        ("VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr,\n        static_cast<uint32_t>(barriers.size())",
         "VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr,\n        static_cast<uint32_t>(barriers.size())"),
        (old, old + "\n        std::memcpy(c.contour_counts.data(),static_cast<const uint8_t*>(frame.readback_mapping)+progressive_status_offset+80,16);"),
        ("c.rigid_counts.fill(0);", "c.rigid_counts.fill(0);c.contour_counts.fill(0);"),
        ("c.telemetry.readback_bytes += output_bytes+(progressive_rgb?80:0);", "c.telemetry.readback_bytes += output_bytes+(progressive_rgb?96:0);"),
        ("c.readback_info.copied_bytes += output_bytes+(progressive_rgb?80:0);", "c.readback_info.copied_bytes += output_bytes+(progressive_rgb?96:0);"),
        ("progressive.status_readback_bytes+=80;", "progressive.status_readback_bytes+=96;"),
    ):
        source = once(source, before, after)
    return source


def ray_native(source):
    for old in ("VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |",
                "VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,"):
        source = once(source, old, old.replace("VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT",
                                              "VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT"))
    return source


def renderer_native(source):
    source = once(source, "std::array<uint32_t,4> rigid_counts{};",
                  "std::array<uint32_t,4> rigid_counts{},contour_counts{};")
    source = once(source, "s->height*112+176:0", "s->height*112+sizeof(HistoryUniform):0")
    return source + """
PR_API int pr_preview_contour_counts(void* handle,uint32_t* output,size_t bytes){
    try{
        require(handle&&output&&bytes==16,"contour counter ABI mismatch");
        const auto& c=*static_cast<Context*>(handle);
        std::memcpy(output,c.contour_counts.data(),16);return 0;
    }catch(const std::exception& error){last_error=error.what();return -1;}
}
"""


def apply(source_dir):
    """Patch only the copied, verified source tree owned by the package builder."""
    root = Path(source_dir)
    transforms = {
        "_shaders/preview_history.comp": history_shader,
        "_native/preview_history.inc": history_native,
        "_native/resident_progressive.inc": progressive_native,
        "_native/resident.inc": resident_native,
        "_native/resident_rayquery.inc": ray_native,
        "_native/renderer.cpp": renderer_native,
    }
    outputs = {root / name: transform((root / name).read_text(encoding="utf-8"))
               for name, transform in transforms.items()}
    include = root / "_shaders/preview_contour.glsl"
    if include.exists():
        raise ValueError("contour resolve already present")
    outputs[include] = (HERE / "contour.glsl").read_text(encoding="utf-8")
    for path, content in outputs.items():
        path.write_text(content, encoding="utf-8", newline="\n")
    return policy()
