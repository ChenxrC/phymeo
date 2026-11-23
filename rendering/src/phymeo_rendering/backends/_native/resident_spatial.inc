// A separate first-batch guide draw preserves the exact primary SPIR-V and
// raw radiance. Same command buffer, uniform, matrices, assets and TLAS; the
// three sensor writes are disabled. Cost includes direct-light shadow queries.
void spatial_setup(Context& c,uint32_t bounces,const uint32_t* compute,size_t compute_bytes,
                   const uint32_t* fragment,size_t fragment_bytes) {
    require(c.progressive_info.shader_compatible&&c.resident_state&&!c.progressive_info.enabled,
            "spatial requires an unconfigured progressive session");
    auto interface=spirv_interface(fragment,fragment_bytes,4);
    require(interface.valid&&interface.main&&(interface.bindings&(1ull<<13))&&
            display_shader_marker(fragment,fragment_bytes)&&
            !named_shader_marker(fragment,fragment_bytes,64,"PR_REFLECTION_ABI")&&
            !named_shader_marker(fragment,fragment_bytes,65,"PR_INDIRECT_ABI")&&
            !named_shader_marker(fragment,fragment_bytes,66,"PR_PROGRESSIVE_ABI"),
            "spatial guide must be compatible direct/environment-only hardware SPIR-V");
    require(bool(c.normal_info.shader_compatible)==named_shader_marker(fragment,fragment_bytes,62,"PR_NORMAL_MAPS_ABI"),
            "spatial guide normal-map ABI mismatch");
    VkPhysicalDeviceFeatures features{};vkGetPhysicalDeviceFeatures(c.physical,&features);
    require(features.independentBlend,"spatial guide needs independent sensor write masks");
    VkPhysicalDeviceProperties properties{};vkGetPhysicalDeviceProperties(c.physical,&properties);
    require(properties.limits.maxPerStageDescriptorSamplers>=4&&properties.limits.maxPerStageDescriptorSampledImages>=4,
            "spatial needs four compute sampled images");
    for(auto format:{VK_FORMAT_R8G8B8A8_UNORM,VK_FORMAT_R32_SFLOAT,VK_FORMAT_R32_UINT,VK_FORMAT_R8_UINT}) {
        VkFormatProperties sampled{};vkGetPhysicalDeviceFormatProperties(c.physical,format,&sampled);
        require(sampled.optimalTilingFeatures&VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT,"spatial guide/sensor sampling unavailable");
    }
    progressive_setup(c,bounces,compute,compute_bytes,true);
    auto& state=*c.progressive_state;
    try {
        std::array<VkAttachmentDescription,5> attachments{};
        std::array<VkAttachmentReference,4> colors{};
        for(uint32_t i=0;i<5;++i){
            auto& a=attachments[i];a.format=FORMATS[i];a.samples=VK_SAMPLE_COUNT_1_BIT;
            a.loadOp=(i==0||i==4)?VK_ATTACHMENT_LOAD_OP_CLEAR:VK_ATTACHMENT_LOAD_OP_LOAD;
            a.storeOp=VK_ATTACHMENT_STORE_OP_STORE;a.stencilLoadOp=VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            a.stencilStoreOp=VK_ATTACHMENT_STORE_OP_DONT_CARE;
            a.initialLayout=i==0?VK_IMAGE_LAYOUT_UNDEFINED:(i==4?VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
            a.finalLayout=i==4?VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            if(i<4)colors[i]={i,VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
        }
        VkAttachmentReference depth{4,VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
        VkSubpassDescription subpass{};subpass.pipelineBindPoint=VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount=4;subpass.pColorAttachments=colors.data();subpass.pDepthStencilAttachment=&depth;
        std::array<VkSubpassDependency,2> dependencies{};
        dependencies[0].srcSubpass=VK_SUBPASS_EXTERNAL;dependencies[0].dstSubpass=0;
        dependencies[0].srcStageMask=VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
        dependencies[0].dstStageMask=VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT|VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        dependencies[0].srcAccessMask=VK_ACCESS_MEMORY_WRITE_BIT;
        dependencies[0].dstAccessMask=VK_ACCESS_COLOR_ATTACHMENT_READ_BIT|VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT|VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        dependencies[1].srcSubpass=0;dependencies[1].dstSubpass=VK_SUBPASS_EXTERNAL;
        dependencies[1].srcStageMask=VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependencies[1].dstStageMask=VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT|VK_PIPELINE_STAGE_TRANSFER_BIT;
        dependencies[1].srcAccessMask=VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        dependencies[1].dstAccessMask=VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_TRANSFER_READ_BIT;
        VkRenderPassCreateInfo pass{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
        pass.attachmentCount=5;pass.pAttachments=attachments.data();pass.subpassCount=1;pass.pSubpasses=&subpass;
        pass.dependencyCount=2;pass.pDependencies=dependencies.data();
        acquire(state.guide_pass,"spatial guide render pass",[&](auto* out){return vkCreateRenderPass(c.device,&pass,nullptr,out);});
        VkShaderModuleCreateInfo module{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};module.codeSize=fragment_bytes;module.pCode=fragment;
        acquire(state.guide_fragment,"spatial guide fragment",[&](auto* out){return vkCreateShaderModule(c.device,&module,nullptr,out);});
        std::array<VkPipelineShaderStageCreateInfo,2> stages{};
        for(auto& stage:stages){stage.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;stage.pName="main";}
        stages[0].stage=VK_SHADER_STAGE_VERTEX_BIT;stages[0].module=c.vertex_shader;
        stages[1].stage=VK_SHADER_STAGE_FRAGMENT_BIT;stages[1].module=state.guide_fragment;
        VkVertexInputBindingDescription binding{0,sizeof(PrResidentVertex),VK_VERTEX_INPUT_RATE_VERTEX};
        const std::array<VkVertexInputAttributeDescription,2> attributes{{{0,0,VK_FORMAT_R32G32B32A32_SFLOAT,0},{1,0,VK_FORMAT_R32G32B32A32_SFLOAT,16}}};
        VkPipelineVertexInputStateCreateInfo vertices{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        vertices.vertexBindingDescriptionCount=1;vertices.pVertexBindingDescriptions=&binding;
        vertices.vertexAttributeDescriptionCount=2;vertices.pVertexAttributeDescriptions=attributes.data();
        VkPipelineInputAssemblyStateCreateInfo assembly{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};assembly.topology=VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        VkPipelineViewportStateCreateInfo viewport{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};viewport.viewportCount=viewport.scissorCount=1;
        VkPipelineRasterizationStateCreateInfo raster{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        raster.polygonMode=VK_POLYGON_MODE_FILL;raster.cullMode=VK_CULL_MODE_NONE;raster.frontFace=VK_FRONT_FACE_COUNTER_CLOCKWISE;raster.lineWidth=1;
        VkPipelineMultisampleStateCreateInfo samples{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};samples.rasterizationSamples=VK_SAMPLE_COUNT_1_BIT;
        VkPipelineDepthStencilStateCreateInfo depth_state{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        depth_state.depthTestEnable=depth_state.depthWriteEnable=VK_TRUE;depth_state.depthCompareOp=VK_COMPARE_OP_LESS;
        std::array<VkPipelineColorBlendAttachmentState,4> blends{};
        blends[0].colorWriteMask=15; // Preserve every original sensor bit.
        VkPipelineColorBlendStateCreateInfo blend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};blend.attachmentCount=4;blend.pAttachments=blends.data();
        const std::array<VkDynamicState,2> dynamics{VK_DYNAMIC_STATE_VIEWPORT,VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};dynamic.dynamicStateCount=2;dynamic.pDynamicStates=dynamics.data();
        VkGraphicsPipelineCreateInfo pipeline{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};pipeline.stageCount=2;pipeline.pStages=stages.data();
        pipeline.pVertexInputState=&vertices;pipeline.pInputAssemblyState=&assembly;pipeline.pViewportState=&viewport;
        pipeline.pRasterizationState=&raster;pipeline.pMultisampleState=&samples;pipeline.pDepthStencilState=&depth_state;
        pipeline.pColorBlendState=&blend;pipeline.pDynamicState=&dynamic;pipeline.layout=c.pipeline_layout;pipeline.renderPass=state.guide_pass;
        VkPipeline temporary{};auto status=vkCreateGraphicsPipelines(c.device,VK_NULL_HANDLE,1,&pipeline,nullptr,&temporary);
        if(temporary){state.guide_pipeline=temporary;++live_resources;}check(status,"spatial guide pipeline");
    }catch(...){c.poisoned=true;throw;}
}

void spatial_record_guide(Context& c,FrameResources& frame,uint32_t width,uint32_t height) {
    auto& state=*c.progressive_state;
    if(!frame.guide_framebuffer){
        const std::array<VkImageView,5> views{frame.images[5].view,frame.images[1].view,frame.images[2].view,frame.images[3].view,frame.images[4].view};
        VkFramebufferCreateInfo framebuffer{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
        framebuffer.renderPass=state.guide_pass;framebuffer.attachmentCount=5;framebuffer.pAttachments=views.data();
        framebuffer.width=width;framebuffer.height=height;framebuffer.layers=1;
        acquire(frame.guide_framebuffer,"spatial guide framebuffer",[&](auto* out){return vkCreateFramebuffer(c.device,&framebuffer,nullptr,out);});
    }
    std::array<VkClearValue,5> clears{};clears[4].depthStencil.depth=1;
    VkRenderPassBeginInfo pass{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};pass.renderPass=state.guide_pass;
    pass.framebuffer=frame.guide_framebuffer;pass.renderArea.extent={width,height};pass.clearValueCount=5;pass.pClearValues=clears.data();
    vkCmdBeginRenderPass(frame.commands,&pass,VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(frame.commands,VK_PIPELINE_BIND_POINT_GRAPHICS,state.guide_pipeline);
    for(uint32_t slot=0;slot<c.resident_state->instances.size();++slot){
        const auto& mesh=c.resident_state->meshes[c.resident_state->instances[slot].mesh_index];
        vkCmdDrawIndexed(frame.commands,mesh.index_count,1,mesh.first_index,static_cast<int32_t>(mesh.first_vertex),slot);
    }
    vkCmdEndRenderPass(frame.commands);frame.guide_initialized=true;
}
