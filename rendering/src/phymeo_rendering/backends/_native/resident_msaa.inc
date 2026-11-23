// Private, optional MSAA resources. The original ABI2 and 1x targets are unchanged.
constexpr VkFormat MSAA_COLOR=VK_FORMAT_R16G16B16A16_SFLOAT;
constexpr VkFormat MSAA_DEPTH=VK_FORMAT_D32_SFLOAT;

void probe_msaa(Context& c,const VkPhysicalDeviceLimits& limits) {
    auto& info=c.msaa_info;
    info.color_format=info.resolve_format=MSAA_COLOR;
    info.depth_format=MSAA_DEPTH; info.output_format=VK_FORMAT_R8G8B8A8_UNORM;
    info.standard_sample_locations=limits.standardSampleLocations;
    VkImageFormatProperties color{},depth{},resolved{};
    const auto color_usage=VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    auto a=vkGetPhysicalDeviceImageFormatProperties(c.physical,MSAA_COLOR,VK_IMAGE_TYPE_2D,
        VK_IMAGE_TILING_OPTIMAL,color_usage,0,&color);
    auto b=vkGetPhysicalDeviceImageFormatProperties(c.physical,MSAA_DEPTH,VK_IMAGE_TYPE_2D,
        VK_IMAGE_TILING_OPTIMAL,VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,0,&depth);
    auto d=vkGetPhysicalDeviceImageFormatProperties(c.physical,MSAA_COLOR,VK_IMAGE_TYPE_2D,
        VK_IMAGE_TILING_OPTIMAL,color_usage | VK_IMAGE_USAGE_SAMPLED_BIT,0,&resolved);
    if(a!=VK_SUCCESS || b!=VK_SUCCESS || d!=VK_SUCCESS || !(resolved.sampleCounts&VK_SAMPLE_COUNT_1_BIT)) return;
    info.color_sample_counts=color.sampleCounts&limits.framebufferColorSampleCounts;
    info.depth_sample_counts=depth.sampleCounts&limits.framebufferDepthSampleCounts;
    info.max_width=std::min({color.maxExtent.width,depth.maxExtent.width,resolved.maxExtent.width,
        limits.maxFramebufferWidth,limits.maxViewportDimensions[0]});
    info.max_height=std::min({color.maxExtent.height,depth.maxExtent.height,resolved.maxExtent.height,
        limits.maxFramebufferHeight,limits.maxViewportDimensions[1]});
}

struct MsaaTarget {
    Context& c;
    uint32_t width{},height{};
    Image color{},depth{},resolved{};
    VkFramebuffer color_framebuffer{},encode_framebuffer{};
    VkDescriptorPool descriptors{};
    VkDescriptorSet descriptor{};
    explicit MsaaTarget(Context& context):c(context) {}
    ~MsaaTarget() {
        release(descriptors,[&](auto h){vkDestroyDescriptorPool(c.device,h,nullptr);});
        release(color_framebuffer,[&](auto h){vkDestroyFramebuffer(c.device,h,nullptr);});
        release(encode_framebuffer,[&](auto h){vkDestroyFramebuffer(c.device,h,nullptr);});
        for(auto* image:{&color,&depth,&resolved}) {
            release(image->view,[&](auto h){vkDestroyImageView(c.device,h,nullptr);});
            release(image->handle,[&](auto h){vkDestroyImage(c.device,h,nullptr);});
            c.free_memory(image->memory,image->allocation);
        }
    }
};
struct MsaaState {
    Context& c;
    VkRenderPass color_pass{},encode_pass{};
    VkPipeline color_pipeline{},sky_pipeline{},encode_pipeline{};
    VkDescriptorSetLayout descriptor_layout{};
    VkPipelineLayout encode_layout{};
    VkShaderModule resolve_vertex{},resolve_fragment{};
    VkSampler sampler{};
    std::unique_ptr<MsaaTarget> target;
    explicit MsaaState(Context& context):c(context) {}
    ~MsaaState() {
        target.reset();
        for(auto* p:{&color_pipeline,&sky_pipeline,&encode_pipeline})
            release(*p,[&](auto h){vkDestroyPipeline(c.device,h,nullptr);});
        release(encode_layout,[&](auto h){vkDestroyPipelineLayout(c.device,h,nullptr);});
        release(descriptor_layout,[&](auto h){vkDestroyDescriptorSetLayout(c.device,h,nullptr);});
        release(sampler,[&](auto h){vkDestroySampler(c.device,h,nullptr);});
        for(auto* p:{&color_pass,&encode_pass})
            release(*p,[&](auto h){vkDestroyRenderPass(c.device,h,nullptr);});
        for(auto* p:{&resolve_vertex,&resolve_fragment})
            release(*p,[&](auto h){vkDestroyShaderModule(c.device,h,nullptr);});
    }
};

void msaa_pipeline(Context& c,VkPipeline& output,VkShaderModule vertex,VkShaderModule fragment,
                   VkRenderPass pass,VkPipelineLayout layout,uint32_t kind) {
    // kind: 0 indexed opaque geometry, 1 fullscreen sky, 2 final OETF.
    uint32_t linear=1;
    VkSpecializationMapEntry entry{60,0,sizeof(linear)};
    VkSpecializationInfo specialization{1,&entry,sizeof(linear),&linear};
    std::array<VkPipelineShaderStageCreateInfo,2> stages{};
    for(auto& stage:stages) { stage.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO; stage.pName="main"; }
    stages[0].stage=VK_SHADER_STAGE_VERTEX_BIT; stages[0].module=vertex;
    stages[1].stage=VK_SHADER_STAGE_FRAGMENT_BIT; stages[1].module=fragment;
    if(kind!=2) stages[1].pSpecializationInfo=&specialization;
    VkVertexInputBindingDescription binding{0,sizeof(PrResidentVertex),VK_VERTEX_INPUT_RATE_VERTEX};
    std::array<VkVertexInputAttributeDescription,2> attributes{{
        {0,0,VK_FORMAT_R32G32B32A32_SFLOAT,0},{1,0,VK_FORMAT_R32G32B32A32_SFLOAT,16}}};
    VkPipelineVertexInputStateCreateInfo vertices{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    if(kind==0) {
        vertices.vertexBindingDescriptionCount=1; vertices.pVertexBindingDescriptions=&binding;
        vertices.vertexAttributeDescriptionCount=2; vertices.pVertexAttributeDescriptions=attributes.data();
    }
    VkPipelineInputAssemblyStateCreateInfo assembly{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    assembly.topology=VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo viewport{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    viewport.viewportCount=1; viewport.scissorCount=1;
    VkPipelineRasterizationStateCreateInfo raster{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    raster.polygonMode=VK_POLYGON_MODE_FILL; raster.cullMode=VK_CULL_MODE_NONE;
    raster.frontFace=VK_FRONT_FACE_COUNTER_CLOCKWISE; raster.lineWidth=1;
    VkPipelineMultisampleStateCreateInfo samples{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    samples.rasterizationSamples=kind==2 ? VK_SAMPLE_COUNT_1_BIT : VK_SAMPLE_COUNT_4_BIT;
    // Ordinary MSAA coverage and per-sample depth, no hidden supersampling or
    // sample-rate-shading feature requirement. Shading rate remains per fragment.
    VkPipelineDepthStencilStateCreateInfo depth{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    depth.depthTestEnable=depth.depthWriteEnable=kind==0 ? VK_TRUE : VK_FALSE;
    depth.depthCompareOp=VK_COMPARE_OP_LESS;
    std::array<VkPipelineColorBlendAttachmentState,4> blends{};
    for(auto& blend:blends) blend.colorWriteMask=15;
    VkPipelineColorBlendStateCreateInfo blending{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    blending.attachmentCount=kind==2 ? 1u : 4u; blending.pAttachments=blends.data();
    const std::array<VkDynamicState,2> states{VK_DYNAMIC_STATE_VIEWPORT,VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dynamic.dynamicStateCount=2; dynamic.pDynamicStates=states.data();
    VkGraphicsPipelineCreateInfo pipeline{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    pipeline.stageCount=2; pipeline.pStages=stages.data(); pipeline.pVertexInputState=&vertices;
    pipeline.pInputAssemblyState=&assembly; pipeline.pViewportState=&viewport;
    pipeline.pRasterizationState=&raster; pipeline.pMultisampleState=&samples;
    pipeline.pDepthStencilState=&depth; pipeline.pColorBlendState=&blending;
    pipeline.pDynamicState=&dynamic; pipeline.layout=layout; pipeline.renderPass=pass;
    VkPipeline temporary=VK_NULL_HANDLE;
    VkResult status=vkCreateGraphicsPipelines(c.device,VK_NULL_HANDLE,1,&pipeline,nullptr,&temporary);
    if(temporary) { output=temporary; ++live_resources; }
    check(status,"resident MSAA graphics pipeline");
}

void make_msaa_passes(Context& c,MsaaState& state) {
    std::array<VkAttachmentDescription,3> attachments{};
    for(uint32_t i=0;i<3;++i) {
        auto& a=attachments[i]; a.format=i==1 ? MSAA_DEPTH : MSAA_COLOR;
        a.samples=i==2 ? VK_SAMPLE_COUNT_1_BIT : VK_SAMPLE_COUNT_4_BIT;
        a.loadOp=i==2 ? VK_ATTACHMENT_LOAD_OP_DONT_CARE : VK_ATTACHMENT_LOAD_OP_CLEAR;
        a.storeOp=i==2 ? VK_ATTACHMENT_STORE_OP_STORE : VK_ATTACHMENT_STORE_OP_DONT_CARE;
        a.stencilLoadOp=VK_ATTACHMENT_LOAD_OP_DONT_CARE; a.stencilStoreOp=VK_ATTACHMENT_STORE_OP_DONT_CARE;
        a.initialLayout=VK_IMAGE_LAYOUT_UNDEFINED;
        a.finalLayout=i==2 ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL :
            (i==1 ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL : VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    }
    // Keep the existing fragment's sensor locations explicitly unused. They are
    // neither allocated nor resolved here; only the independent center pass owns them.
    std::array<VkAttachmentReference,4> colors{},resolves{};
    for(uint32_t i=0;i<4;++i) {
        colors[i]={i==0 ? 0u : VK_ATTACHMENT_UNUSED,VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
        resolves[i]={i==0 ? 2u : VK_ATTACHMENT_UNUSED,VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    }
    VkAttachmentReference depth{1,VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
    VkSubpassDescription subpass{}; subpass.pipelineBindPoint=VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount=4; subpass.pColorAttachments=colors.data();
    subpass.pResolveAttachments=resolves.data(); subpass.pDepthStencilAttachment=&depth;
    std::array<VkSubpassDependency,2> deps{};
    deps[0].srcSubpass=VK_SUBPASS_EXTERNAL; deps[0].dstSubpass=0;
    deps[0].srcStageMask=VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    deps[0].dstStageMask=VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    deps[0].dstAccessMask=VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    deps[1].srcSubpass=0; deps[1].dstSubpass=VK_SUBPASS_EXTERNAL;
    deps[1].srcStageMask=VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    deps[1].dstStageMask=VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    deps[1].srcAccessMask=VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT; deps[1].dstAccessMask=VK_ACCESS_SHADER_READ_BIT;
    VkRenderPassCreateInfo pass{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    pass.attachmentCount=3; pass.pAttachments=attachments.data(); pass.subpassCount=1; pass.pSubpasses=&subpass;
    pass.dependencyCount=2; pass.pDependencies=deps.data();
    acquire(state.color_pass,"MSAA display-linear render pass",[&](auto* out){return vkCreateRenderPass(c.device,&pass,nullptr,out);});

    VkAttachmentDescription encoded{};
    encoded.format=VK_FORMAT_R8G8B8A8_UNORM; encoded.samples=VK_SAMPLE_COUNT_1_BIT;
    encoded.loadOp=VK_ATTACHMENT_LOAD_OP_DONT_CARE; encoded.storeOp=VK_ATTACHMENT_STORE_OP_STORE;
    encoded.stencilLoadOp=VK_ATTACHMENT_LOAD_OP_DONT_CARE; encoded.stencilStoreOp=VK_ATTACHMENT_STORE_OP_DONT_CARE;
    encoded.initialLayout=VK_IMAGE_LAYOUT_UNDEFINED; encoded.finalLayout=VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    VkAttachmentReference encoded_color{0,VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    subpass.colorAttachmentCount=1; subpass.pColorAttachments=&encoded_color;
    subpass.pResolveAttachments=nullptr; subpass.pDepthStencilAttachment=nullptr;
    deps[0].srcStageMask=VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    deps[0].dstStageMask=VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    deps[0].srcAccessMask=VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    deps[0].dstAccessMask=VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    deps[1].dstStageMask=VK_PIPELINE_STAGE_TRANSFER_BIT; deps[1].dstAccessMask=VK_ACCESS_TRANSFER_READ_BIT;
    pass.attachmentCount=1; pass.pAttachments=&encoded;
    acquire(state.encode_pass,"MSAA final sRGB pass",[&](auto* out){return vkCreateRenderPass(c.device,&pass,nullptr,out);});
}

void resident_set_msaa(Context& c,uint32_t samples,const uint32_t* vert,size_t vert_bytes,
                       const uint32_t* frag,size_t frag_bytes) {
    require(c.resident && !c.poisoned,"resident MSAA context unavailable");
    require(samples==1 || samples==4,"resident MSAA accepts only 1 or 4 samples");
    require(c.properties.captures==0 && !c.msaa_state,"resident MSAA is immutable after configuration or capture");
    if(samples==1) return;
    const auto& info=c.msaa_info;
    require(info.shader_compatible && (info.color_sample_counts&4u) && (info.depth_sample_counts&4u) &&
            info.max_width && info.max_height,"resident MSAA format/sample count or main shader unsupported");
    require(!c.environment_info.loaded || c.resident_state->environment->msaa_compatible,
            "resident sky fragment lacks named MSAA color ABI marker");
    const auto vertex=spirv_interface(vert,vert_bytes,0),fragment=spirv_interface(frag,frag_bytes,4);
    require(vertex.valid && vertex.main && vertex.locations==0 && vertex.bindings==0,
            "resident MSAA resolve vertex stage/interface mismatch");
    require(fragment.valid && fragment.main && fragment.locations==1 && fragment.bindings==1 &&
            named_shader_marker(frag,frag_bytes,61,"PR_MSAA_RESOLVE_ABI"),
            "resident MSAA resolve fragment stage/interface/ABI mismatch");
    const auto errors=c.errors.load();
    auto resource=std::make_shared<MsaaState>(c);
    make_msaa_passes(c,*resource);
    VkDescriptorSetLayoutBinding binding{0,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1,VK_SHADER_STAGE_FRAGMENT_BIT,nullptr};
    VkDescriptorSetLayoutCreateInfo descriptors{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    descriptors.bindingCount=1; descriptors.pBindings=&binding;
    acquire(resource->descriptor_layout,"MSAA resolve descriptor layout",[&](auto* out){return vkCreateDescriptorSetLayout(c.device,&descriptors,nullptr,out);});
    VkPipelineLayoutCreateInfo layout{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layout.setLayoutCount=1; layout.pSetLayouts=&resource->descriptor_layout;
    acquire(resource->encode_layout,"MSAA resolve pipeline layout",[&](auto* out){return vkCreatePipelineLayout(c.device,&layout,nullptr,out);});
    VkShaderModuleCreateInfo module{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    module.codeSize=vert_bytes; module.pCode=vert;
    acquire(resource->resolve_vertex,"MSAA resolve vertex",[&](auto* out){return vkCreateShaderModule(c.device,&module,nullptr,out);});
    module.codeSize=frag_bytes; module.pCode=frag;
    acquire(resource->resolve_fragment,"MSAA resolve fragment",[&](auto* out){return vkCreateShaderModule(c.device,&module,nullptr,out);});
    VkSamplerCreateInfo sampler{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    sampler.magFilter=sampler.minFilter=VK_FILTER_NEAREST; sampler.mipmapMode=VK_SAMPLER_MIPMAP_MODE_NEAREST;
    sampler.addressModeU=sampler.addressModeV=sampler.addressModeW=VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    acquire(resource->sampler,"MSAA resolved sampler",[&](auto* out){return vkCreateSampler(c.device,&sampler,nullptr,out);});
    msaa_pipeline(c,resource->color_pipeline,c.vertex_shader,c.fragment_shader,resource->color_pass,c.pipeline_layout,0);
    msaa_pipeline(c,resource->encode_pipeline,resource->resolve_vertex,resource->resolve_fragment,
                  resource->encode_pass,resource->encode_layout,2);
    require(c.errors.load()==errors,"Vulkan validation failed creating MSAA pipelines");
    c.msaa_state=std::move(resource);
    c.msaa_info.requested_samples=c.msaa_info.color_samples=4;
}

void msaa_prepare(Context& c,FrameResources& frame,uint32_t width,uint32_t height) {
    auto& state=*c.msaa_state;
    const auto errors=c.errors.load();
    require(width<=c.msaa_info.max_width && height<=c.msaa_info.max_height,"MSAA target exceeds format extent");
    if(c.environment_info.loaded && !state.sky_pipeline) {
        auto& environment=*c.resident_state->environment;
        require(environment.msaa_compatible,"resident sky fragment lacks MSAA ABI marker");
        msaa_pipeline(c,state.sky_pipeline,environment.sky_vertex,environment.sky_fragment,
                      state.color_pass,c.pipeline_layout,1);
    }
    require(c.errors.load()==errors,"Vulkan validation failed creating MSAA sky pipeline");
    if(state.target && state.target->width==width && state.target->height==height) return;
    auto target=std::make_unique<MsaaTarget>(c); target->width=width; target->height=height;
    c.make_image(target->color,width,height,MSAA_COLOR,false,VK_SAMPLE_COUNT_4_BIT);
    c.make_image(target->depth,width,height,MSAA_DEPTH,true,VK_SAMPLE_COUNT_4_BIT);
    c.make_image(target->resolved,width,height,MSAA_COLOR,false,VK_SAMPLE_COUNT_1_BIT,VK_IMAGE_USAGE_SAMPLED_BIT);
    std::array<VkImageView,3> views{target->color.view,target->depth.view,target->resolved.view};
    VkFramebufferCreateInfo framebuffer{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
    framebuffer.renderPass=state.color_pass; framebuffer.attachmentCount=3; framebuffer.pAttachments=views.data();
    framebuffer.width=width; framebuffer.height=height; framebuffer.layers=1;
    acquire(target->color_framebuffer,"MSAA framebuffer",[&](auto* out){return vkCreateFramebuffer(c.device,&framebuffer,nullptr,out);});
    framebuffer.renderPass=state.encode_pass; framebuffer.attachmentCount=1; framebuffer.pAttachments=&frame.images[0].view;
    acquire(target->encode_framebuffer,"MSAA encoded framebuffer",[&](auto* out){return vkCreateFramebuffer(c.device,&framebuffer,nullptr,out);});
    VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1};
    VkDescriptorPoolCreateInfo pool{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pool.maxSets=1; pool.poolSizeCount=1; pool.pPoolSizes=&size;
    acquire(target->descriptors,"MSAA image descriptors",[&](auto* out){return vkCreateDescriptorPool(c.device,&pool,nullptr,out);});
    VkDescriptorSetAllocateInfo allocate{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocate.descriptorPool=target->descriptors; allocate.descriptorSetCount=1; allocate.pSetLayouts=&state.descriptor_layout;
    check(vkAllocateDescriptorSets(c.device,&allocate,&target->descriptor),"MSAA image descriptor allocation");
    VkDescriptorImageInfo image{state.sampler,target->resolved.view,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    write.dstSet=target->descriptor; write.dstBinding=0; write.descriptorCount=1;
    write.descriptorType=VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; write.pImageInfo=&image;
    vkUpdateDescriptorSets(c.device,1,&write,0,nullptr);
    require(c.errors.load()==errors,"Vulkan validation failed creating MSAA attachments");
    state.target=std::move(target); ++c.msaa_info.target_rebuilds;
}

void msaa_record(Context& c,FrameResources& frame,const PrParams& params) {
    auto& state=*c.msaa_state; auto& target=*state.target; auto& scene=*c.resident_state;
    std::array<VkClearValue,3> clear{};
    clear[0].color.float32[3]=1; clear[1].depthStencil.depth=1;
    if(c.studio.environment[3]!=0) {
        const auto mapped=c.display_transform==1 ? neutral_clear(c.studio.environment,params.options[0]) : std::array<double,3>{};
        for(uint32_t k=0;k<3;++k) clear[0].color.float32[k]=static_cast<float>(c.display_transform==1 ?
            std::clamp(mapped[k],0.0,1.0) : std::clamp(static_cast<double>(c.studio.environment[k])*params.options[0],0.0,1.0));
    }
    VkRenderPassBeginInfo pass{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    pass.renderPass=state.color_pass; pass.framebuffer=target.color_framebuffer;
    pass.renderArea.extent={target.width,target.height}; pass.clearValueCount=3; pass.pClearValues=clear.data();
    vkCmdBeginRenderPass(frame.commands,&pass,VK_SUBPASS_CONTENTS_INLINE);
    VkViewport viewport{0,0,static_cast<float>(target.width),static_cast<float>(target.height),0,1};
    VkRect2D scissor{{0,0},{target.width,target.height}};
    vkCmdSetViewport(frame.commands,0,1,&viewport); vkCmdSetScissor(frame.commands,0,1,&scissor);
    vkCmdBindDescriptorSets(frame.commands,VK_PIPELINE_BIND_POINT_GRAPHICS,c.pipeline_layout,0,1,&scene.descriptor_set,0,nullptr);
    if(c.environment_info.loaded) {
        vkCmdBindPipeline(frame.commands,VK_PIPELINE_BIND_POINT_GRAPHICS,state.sky_pipeline);
        vkCmdDraw(frame.commands,3,1,0,0);
    }
    vkCmdBindPipeline(frame.commands,VK_PIPELINE_BIND_POINT_GRAPHICS,state.color_pipeline);
    VkDeviceSize offset=0;
    vkCmdBindVertexBuffers(frame.commands,0,1,&scene.vertices.handle,&offset);
    vkCmdBindIndexBuffer(frame.commands,scene.indices.handle,0,VK_INDEX_TYPE_UINT32);
    for(uint32_t slot=0;slot<scene.instances.size();++slot) {
        const auto& mesh=scene.meshes[scene.instances[slot].mesh_index];
        vkCmdDrawIndexed(frame.commands,mesh.index_count,1,mesh.first_index,static_cast<int32_t>(mesh.first_vertex),slot);
    }
    vkCmdEndRenderPass(frame.commands); // hardware average: display-linear RGBA16F
    pass.renderPass=state.encode_pass; pass.framebuffer=target.encode_framebuffer;
    pass.clearValueCount=0; pass.pClearValues=nullptr;
    vkCmdBeginRenderPass(frame.commands,&pass,VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(frame.commands,VK_PIPELINE_BIND_POINT_GRAPHICS,state.encode_pipeline);
    vkCmdBindDescriptorSets(frame.commands,VK_PIPELINE_BIND_POINT_GRAPHICS,state.encode_layout,0,1,&target.descriptor,0,nullptr);
    vkCmdDraw(frame.commands,3,1,0,0);
    vkCmdEndRenderPass(frame.commands);
}
