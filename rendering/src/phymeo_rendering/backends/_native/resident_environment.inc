// Explicit directional environment resources, uploaded once. No file I/O,
// resizing, convolution, shader compilation or hidden renderer fallback.
std::unique_ptr<ResidentEnvironment> upload_environment_resource(Context& c,
    const PrResidentEnvironment& configuration,const void* rgba,size_t rgba_bytes,
    VkFormat format,bool enabled) {
    auto result=std::make_unique<ResidentEnvironment>(c);
    const uint32_t side=configuration.side,levels=configuration.mip_levels;
    VkImageCreateInfo image{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    image.flags=VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
    image.imageType=VK_IMAGE_TYPE_2D; image.format=format; image.extent={side,side,1};
    image.mipLevels=levels; image.arrayLayers=6; image.samples=VK_SAMPLE_COUNT_1_BIT;
    image.tiling=VK_IMAGE_TILING_OPTIMAL; image.usage=VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    image.sharingMode=VK_SHARING_MODE_EXCLUSIVE; image.initialLayout=VK_IMAGE_LAYOUT_UNDEFINED;
    acquire(result->image.handle,"resident environment image",[&](auto* out){return vkCreateImage(c.device,&image,nullptr,out);});
    VkMemoryRequirements needed{}; vkGetImageMemoryRequirements(c.device,result->image.handle,&needed);
    VkMemoryAllocateInfo allocate{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocate.allocationSize=needed.size;
    allocate.memoryTypeIndex=c.memory_type(needed.memoryTypeBits,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    acquire(result->image.memory,"resident environment memory",[&](auto* out){return vkAllocateMemory(c.device,&allocate,nullptr,out);});
    c.account_allocation(result->image.allocation,allocate);
    check(vkBindImageMemory(c.device,result->image.handle,result->image.memory,0),"resident environment bind");
    VkImageViewCreateInfo view{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    view.image=result->image.handle; view.viewType=VK_IMAGE_VIEW_TYPE_CUBE; view.format=format;
    view.subresourceRange={VK_IMAGE_ASPECT_COLOR_BIT,0,levels,0,6};
    acquire(result->image.view,"resident environment cube view",[&](auto* out){return vkCreateImageView(c.device,&view,nullptr,out);});
    VkSamplerCreateInfo sampler{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    sampler.magFilter=sampler.minFilter=VK_FILTER_LINEAR;
    sampler.mipmapMode=VK_SAMPLER_MIPMAP_MODE_LINEAR;
    sampler.addressModeU=sampler.addressModeV=sampler.addressModeW=VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler.minLod=0; sampler.maxLod=static_cast<float>(levels-1);
    acquire(result->sampler,"resident environment sampler",[&](auto* out){return vkCreateSampler(c.device,&sampler,nullptr,out);});
    c.make_buffer(result->uniform,sizeof(PrResidentEnvironment),
        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    PrResidentEnvironment uniform=configuration;
    uniform.reserved[0]=enabled ? 1u : 0u; // private GPU enabled word; caller must pass 0
    FrameResources staging(c);
    const size_t total=rgba_bytes+sizeof(uniform);
    c.make_buffer(staging.vertices,total,VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    void* mapped=nullptr;
    check(vkMapMemory(c.device,staging.vertices.memory,0,total,0,&mapped),"resident environment staging map");
    std::memcpy(mapped,rgba,rgba_bytes);
    std::memcpy(static_cast<uint8_t*>(mapped)+rgba_bytes,&uniform,sizeof(uniform));
    vkUnmapMemory(c.device,staging.vertices.memory);
    make_commands(c,staging);
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    check(vkBeginCommandBuffer(staging.commands,&begin),"resident environment upload begin");
    VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    barrier.srcQueueFamilyIndex=barrier.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;
    barrier.image=result->image.handle; barrier.subresourceRange=view.subresourceRange;
    barrier.oldLayout=VK_IMAGE_LAYOUT_UNDEFINED; barrier.newLayout=VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.dstAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(staging.commands,VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,
        0,0,nullptr,0,nullptr,1,&barrier);
    std::vector<VkBufferImageCopy> copies(levels);
    size_t offset=0;
    for (uint32_t level=0;level<levels;++level) {
        uint32_t dim=side>>level;
        copies[level].bufferOffset=offset;
        copies[level].imageSubresource={VK_IMAGE_ASPECT_COLOR_BIT,level,0,6};
        copies[level].imageExtent={dim,dim,1};
        offset+=static_cast<size_t>(dim)*dim*6*(format==VK_FORMAT_R32G32B32A32_SFLOAT ? 16u : 4u);
    }
    require(offset==rgba_bytes,"internal environment upload layout mismatch");
    vkCmdCopyBufferToImage(staging.commands,staging.vertices.handle,result->image.handle,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,levels,copies.data());
    VkBufferCopy uniform_copy{rgba_bytes,0,sizeof(uniform)};
    vkCmdCopyBuffer(staging.commands,staging.vertices.handle,result->uniform.handle,1,&uniform_copy);
    barrier.oldLayout=VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL; barrier.newLayout=VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT; barrier.dstAccessMask=VK_ACCESS_SHADER_READ_BIT;
    VkBufferMemoryBarrier uniform_barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    uniform_barrier.srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT; uniform_barrier.dstAccessMask=VK_ACCESS_UNIFORM_READ_BIT;
    uniform_barrier.srcQueueFamilyIndex=uniform_barrier.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;
    uniform_barrier.buffer=result->uniform.handle; uniform_barrier.size=VK_WHOLE_SIZE;
    vkCmdPipelineBarrier(staging.commands,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        0,0,nullptr,1,&uniform_barrier,1,&barrier);
    check(vkEndCommandBuffer(staging.commands),"resident environment upload end");
    submit_and_wait(c,staging);
    return result;
}

void make_sky_pipeline(Context& c,ResidentEnvironment& environment,
    const uint32_t* vert,size_t vert_bytes,const uint32_t* frag,size_t frag_bytes) {
    VkShaderModuleCreateInfo module{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    module.codeSize=vert_bytes; module.pCode=vert;
    acquire(environment.sky_vertex,"resident sky vertex",[&](auto* out){return vkCreateShaderModule(c.device,&module,nullptr,out);});
    module.codeSize=frag_bytes; module.pCode=frag;
    acquire(environment.sky_fragment,"resident sky fragment",[&](auto* out){return vkCreateShaderModule(c.device,&module,nullptr,out);});
    std::array<VkPipelineShaderStageCreateInfo,2> stages{};
    for (auto& stage:stages) { stage.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO; stage.pName="main"; }
    stages[0].stage=VK_SHADER_STAGE_VERTEX_BIT; stages[0].module=environment.sky_vertex;
    stages[1].stage=VK_SHADER_STAGE_FRAGMENT_BIT; stages[1].module=environment.sky_fragment;
    VkPipelineVertexInputStateCreateInfo vertices{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo assembly{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    assembly.topology=VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo viewport{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    viewport.viewportCount=1; viewport.scissorCount=1;
    VkPipelineRasterizationStateCreateInfo raster{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    raster.polygonMode=VK_POLYGON_MODE_FILL; raster.cullMode=VK_CULL_MODE_NONE; raster.lineWidth=1;
    VkPipelineMultisampleStateCreateInfo samples{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    samples.rasterizationSamples=VK_SAMPLE_COUNT_1_BIT;
    // Both depth operations are off. The three sensor write masks are ZERO;
    // they retain render-pass clears or later geometry writes byte for byte.
    VkPipelineDepthStencilStateCreateInfo depth{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    std::array<VkPipelineColorBlendAttachmentState,4> attachments{};
    attachments[0].colorWriteMask=VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo blending{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    blending.attachmentCount=4; blending.pAttachments=attachments.data();
    const std::array<VkDynamicState,2> states{VK_DYNAMIC_STATE_VIEWPORT,VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dynamic.dynamicStateCount=2; dynamic.pDynamicStates=states.data();
    VkGraphicsPipelineCreateInfo pipeline{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    pipeline.stageCount=2; pipeline.pStages=stages.data(); pipeline.pVertexInputState=&vertices;
    pipeline.pInputAssemblyState=&assembly; pipeline.pViewportState=&viewport; pipeline.pRasterizationState=&raster;
    pipeline.pMultisampleState=&samples; pipeline.pDepthStencilState=&depth; pipeline.pColorBlendState=&blending;
    pipeline.pDynamicState=&dynamic; pipeline.layout=c.pipeline_layout; pipeline.renderPass=c.render_pass;
    VkPipeline temporary=VK_NULL_HANDLE;
    VkResult status=vkCreateGraphicsPipelines(c.device,VK_NULL_HANDLE,1,&pipeline,nullptr,&temporary);
    // Pipeline creation can return an owned handle even on partial failure.
    if (temporary) { environment.sky_pipeline=temporary; ++live_resources; }
    check(status,"resident sky pipeline");
}

void resident_set_environment(Context& c,const PrResidentEnvironment& config,const float* rgba,size_t rgba_bytes,
    const uint32_t* vert,size_t vert_bytes,const uint32_t* frag,size_t frag_bytes) {
    require(c.resident && c.resident_state && !c.poisoned,"resident geometry must precede environment upload");
    auto& info=c.environment_info;
    require(c.properties.captures==0 && info.loaded==0,"resident environment is immutable after upload or capture");
    require(info.supported,"resident RGBA32F cube filtering or independentBlend unavailable");
    require(info.shader_compatible,"resident main SPIR-V lacks directional environment interface");
    require(config.side>=2 && config.side<=info.max_side && !(config.side&(config.side-1)),
        "resident environment side must be a supported power of two");
    uint32_t levels=0;
    for (uint32_t dim=config.side;dim;dim>>=1) ++levels;
    require(config.mip_levels==levels && levels<=info.max_mip_levels,"resident environment requires complete mip chain");
    require(config.reserved[0]==0 && config.reserved[1]==0,"resident environment reserved words must be zero");
    uint64_t expected=0;
    for (uint32_t dim=config.side;dim;dim>>=1) expected+=static_cast<uint64_t>(dim)*dim*6*16;
    require(rgba && expected==rgba_bytes && expected<=info.max_rgba_bytes,"resident environment RGBA byte count/capacity mismatch");
    for (size_t i=0;i<rgba_bytes/sizeof(float);++i)
        require(std::isfinite(rgba[i]) && (i%4==3 ? rgba[i]==1 : rgba[i]>=0 && rgba[i]<=100),
            "resident environment requires finite RGB in [0,100], alpha=1");
    for (uint32_t k=0;k<4;++k) {
        double bound=0;
        for (uint32_t i=0;i<9;++i) {
            require(std::isfinite(config.diffuse_sh[i][k]) && (k!=3 || config.diffuse_sh[i][k]==0),
                "resident environment requires finite SH RGB, w=0");
            bound+=std::abs(static_cast<double>(config.diffuse_sh[i][k]))*1.1;
        }
        require(bound<std::numeric_limits<float>::max()/4.0,"resident environment SH evaluation can overflow FP32");
    }
    const auto vertex=spirv_interface(vert,vert_bytes,0),fragment=spirv_interface(frag,frag_bytes,4);
    require(vertex.valid && vertex.main && vertex.locations==0 && vertex.bindings==0,
        "resident sky vertex SPIR-V has incompatible stage/interface");
    constexpr uint64_t sky_bindings=(1ull<<1)|(1ull<<11)|(1ull<<12);
    require(fragment.valid && fragment.main && fragment.bindings==sky_bindings && fragment.locations==1,
        "resident sky fragment SPIR-V has incompatible stage/interface");
    require(bool(c.progressive_info.shader_compatible)==named_shader_marker(frag,frag_bytes,66,"PR_PROGRESSIVE_ABI"),
        "resident sky progressive radiance ABI mismatch");
    const bool display_compatible=display_shader_marker(frag,frag_bytes);
    const bool msaa_compatible=named_shader_marker(frag,frag_bytes,59,"PR_MSAA_COLOR_ABI");
    require(c.msaa_info.requested_samples!=4 || msaa_compatible,"resident sky lacks MSAA color ABI marker");
    require(c.display_transform!=1 || display_compatible,
        "resident sky fragment lacks named display-transform ABI marker" );
    // All deterministic input/ABI checks above precede any resource allocation.
    uint64_t errors_before=c.errors.load();
    try {
        auto resource=upload_environment_resource(c,config,rgba,rgba_bytes,VK_FORMAT_R32G32B32A32_SFLOAT,true);
        make_sky_pipeline(c,*resource,vert,vert_bytes,frag,frag_bytes);
        resource->display_compatible=display_compatible;
        resource->msaa_compatible=msaa_compatible;
        require(c.errors.load()==errors_before,"Vulkan validation failed during environment upload");
        c.resident_state->environment=std::move(resource);
    } catch (...) { c.poisoned=true; throw; }
    info.side=config.side; info.mip_levels=levels; info.loaded=1; info.format=VK_FORMAT_R32G32B32A32_SFLOAT;
    info.source_rgba_bytes=rgba_bytes; info.image_upload_bytes+=rgba_bytes;
    info.parameter_upload_bytes+=sizeof(config); ++info.upload_calls;
}
