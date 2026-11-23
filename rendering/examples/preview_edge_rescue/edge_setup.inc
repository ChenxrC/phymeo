// Independent selective raster pipeline, sharing actual immutable scene data.
void edge_setup(Context& c,const uint32_t* fragment,size_t fragment_bytes,uint32_t limit) {
    require(c.progressive_state&&c.sparse_shader&&!c.properties.captures&&!c.edge_limit,
            "edge setup requires new configured compact owner");
    require(limit>0&&limit<=1048576,"edge fragment budget outside hard bound");
    const auto interface=spirv_interface(fragment,fragment_bytes,4);
    require(interface.valid&&interface.main&&
            named_shader_marker(fragment,fragment_bytes,99,"PR_PREVIEW_EDGE_FRAGMENT_ABI")&&
            (interface.bindings&(15ull<<16))==(15ull<<16),"edge fragment/interface marker absent");
    require(bool(c.normal_info.shader_compatible)==named_shader_marker(fragment,fragment_bytes,62,"PR_NORMAL_MAPS_ABI"),
            "edge normal-map ABI differs");
    VkPhysicalDeviceProperties props{};vkGetPhysicalDeviceProperties(c.physical,&props);
    require(props.limits.maxPerStageDescriptorSamplers>=9&&props.limits.maxPerStageDescriptorStorageBuffers>=9,
            "edge sampled/atomic descriptor budget unavailable");
    auto& state=*c.progressive_state;c.edge_limit=limit;
    try {
        std::array<VkAttachmentDescription,2> attachments{};
        attachments[0].format=VK_FORMAT_R32G32B32A32_SFLOAT;
        attachments[1].format=FORMATS[4];
        for(auto& a:attachments){a.samples=VK_SAMPLE_COUNT_1_BIT;a.storeOp=VK_ATTACHMENT_STORE_OP_STORE;
            a.stencilLoadOp=VK_ATTACHMENT_LOAD_OP_DONT_CARE;a.stencilStoreOp=VK_ATTACHMENT_STORE_OP_DONT_CARE;}
        attachments[0].loadOp=VK_ATTACHMENT_LOAD_OP_CLEAR;
        attachments[0].initialLayout=VK_IMAGE_LAYOUT_UNDEFINED;
        attachments[0].finalLayout=VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        attachments[1].loadOp=VK_ATTACHMENT_LOAD_OP_LOAD;
        attachments[1].initialLayout=attachments[1].finalLayout=VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        VkAttachmentReference color{0,VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
        VkAttachmentReference depth{1,VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
        VkSubpassDescription subpass{};subpass.pipelineBindPoint=VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount=1;subpass.pColorAttachments=&color;subpass.pDepthStencilAttachment=&depth;
        std::array<VkSubpassDependency,2> dependencies{};
        dependencies[0].srcSubpass=VK_SUBPASS_EXTERNAL;dependencies[0].dstSubpass=0;
        dependencies[0].srcStageMask=VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
        dependencies[0].dstStageMask=VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT|VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT|VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        dependencies[0].srcAccessMask=VK_ACCESS_MEMORY_WRITE_BIT;
        dependencies[0].dstAccessMask=VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT|VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT|VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT;
        dependencies[1].srcSubpass=0;dependencies[1].dstSubpass=VK_SUBPASS_EXTERNAL;
        dependencies[1].srcStageMask=VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT|VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        dependencies[1].dstStageMask=VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        dependencies[1].srcAccessMask=VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT|VK_ACCESS_SHADER_WRITE_BIT;
        dependencies[1].dstAccessMask=VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT;
        VkRenderPassCreateInfo pass{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
        pass.attachmentCount=2;pass.pAttachments=attachments.data();pass.subpassCount=1;pass.pSubpasses=&subpass;
        pass.dependencyCount=2;pass.pDependencies=dependencies.data();
        acquire(state.edge_pass,"edge rescue render pass",[&](auto* out){return vkCreateRenderPass(c.device,&pass,nullptr,out);});