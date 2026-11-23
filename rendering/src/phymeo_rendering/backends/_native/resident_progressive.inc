// GPU-only scene-linear running mean. Targets/pipelines exist only on explicit
// progressive contexts; old contexts retain their original attachments/limits.
struct ProgressiveState {
    Context& c;
    VkDescriptorSetLayout layout{};
    VkPipelineLayout pipeline_layout{};
    VkShaderModule shader{};
    VkPipeline pipeline{};
    VkDescriptorPool pool{};
    VkDescriptorSet set{};
    VkSampler sampler{};
    VkRenderPass guide_pass{};
    VkPipeline guide_pipeline{};
    VkShaderModule guide_fragment{};
    Image mean{},display{};
    Buffer status{},readback{};
    void* mapping{};
    uint32_t width{},height{};
    bool initialized=false,armed=false,pending_reset=false;
    explicit ProgressiveState(Context& owner):c(owner) {}
    void clear_targets() noexcept {
        release(pool,[&](auto h){vkDestroyDescriptorPool(c.device,h,nullptr);});set={};
        for(auto* image:{&mean,&display}) {
            release(image->view,[&](auto h){vkDestroyImageView(c.device,h,nullptr);});
            release(image->handle,[&](auto h){vkDestroyImage(c.device,h,nullptr);});
            c.free_memory(image->memory,image->allocation);
        }
        if(mapping){vkUnmapMemory(c.device,readback.memory);mapping=nullptr;}
        for(auto* buffer:{&status,&readback}) {
            release(buffer->handle,[&](auto h){vkDestroyBuffer(c.device,h,nullptr);});
            c.free_memory(buffer->memory,buffer->allocation);
        }
        width=height=0;initialized=false;
    }
    ~ProgressiveState() {
        clear_targets();
        release(guide_pipeline,[&](auto h){vkDestroyPipeline(c.device,h,nullptr);});
        release(guide_fragment,[&](auto h){vkDestroyShaderModule(c.device,h,nullptr);});
        release(guide_pass,[&](auto h){vkDestroyRenderPass(c.device,h,nullptr);});
        release(sampler,[&](auto h){vkDestroySampler(c.device,h,nullptr);});
        release(pipeline,[&](auto h){vkDestroyPipeline(c.device,h,nullptr);});
        release(shader,[&](auto h){vkDestroyShaderModule(c.device,h,nullptr);});
        release(pipeline_layout,[&](auto h){vkDestroyPipelineLayout(c.device,h,nullptr);});
        release(layout,[&](auto h){vkDestroyDescriptorSetLayout(c.device,h,nullptr);});
    }
};

void progressive_setup(Context& c,uint32_t bounces,const uint32_t* code,size_t bytes,bool guided=false) {
    require(c.progressive_info.supported&&c.progressive_info.shader_compatible&&c.indirect_info.enabled,
            "progressive requires explicit compatible indirect hardware shader");
    require(c.resident_state&&!c.poisoned&&!c.properties.captures&&!c.progressive_info.enabled,
            "progressive setup requires uploaded uncaptured session");
    require((bounces==1||bounces==2)&&c.msaa_info.requested_samples==1,"progressive supports one/two bounces and MSAA1");
    require(!guided||c.spatial_info.shader_compatible,"spatial fragment/compute ABI mismatch");
    const auto interface=spirv_interface(code,bytes,5);
    require(interface.valid&&interface.main&&interface.bindings==(guided?255u:15u)&&
            named_shader_marker(code,bytes,guided?69:67,guided?"PR_SPATIAL_MEAN_ABI":"PR_PROGRESSIVE_MEAN_ABI",5),"progressive compute shader ABI unavailable");
    auto state=std::make_shared<ProgressiveState>(c);
    std::array<VkDescriptorSetLayoutBinding,8> bindings{};
    const uint32_t binding_count=guided?8u:4u;
    for(uint32_t i=0;i<binding_count;++i)bindings[i]={i,i>=4?VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
        (i==3?VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:VK_DESCRIPTOR_TYPE_STORAGE_IMAGE),1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
    VkDescriptorSetLayoutCreateInfo layout{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    layout.bindingCount=binding_count;layout.pBindings=bindings.data();
    acquire(state->layout,"progressive descriptor layout",[&](auto* out){return vkCreateDescriptorSetLayout(c.device,&layout,nullptr,out);});
    VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT,0,20};
    VkPipelineLayoutCreateInfo pipeline_layout{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pipeline_layout.setLayoutCount=1;pipeline_layout.pSetLayouts=&state->layout;
    pipeline_layout.pushConstantRangeCount=1;pipeline_layout.pPushConstantRanges=&range;
    acquire(state->pipeline_layout,"progressive pipeline layout",[&](auto* out){return vkCreatePipelineLayout(c.device,&pipeline_layout,nullptr,out);});
    VkShaderModuleCreateInfo module{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};module.codeSize=bytes;module.pCode=code;
    acquire(state->shader,"progressive compute shader",[&](auto* out){return vkCreateShaderModule(c.device,&module,nullptr,out);});
    VkComputePipelineCreateInfo pipeline{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipeline.stage={VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    pipeline.stage.stage=VK_SHADER_STAGE_COMPUTE_BIT;pipeline.stage.module=state->shader;pipeline.stage.pName="main";
    pipeline.layout=state->pipeline_layout;
    VkPipeline temporary{};auto status=vkCreateComputePipelines(c.device,VK_NULL_HANDLE,1,&pipeline,nullptr,&temporary);
    if(temporary){state->pipeline=temporary;++live_resources;}check(status,"progressive compute pipeline");
    if(guided) {
        VkSamplerCreateInfo sampler{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        sampler.magFilter=sampler.minFilter=VK_FILTER_NEAREST;
        sampler.mipmapMode=VK_SAMPLER_MIPMAP_MODE_NEAREST;
        sampler.addressModeU=sampler.addressModeV=sampler.addressModeW=VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        acquire(state->sampler,"spatial nearest sampler",[&](auto* out){return vkCreateSampler(c.device,&sampler,nullptr,out);});
    }
    c.progressive_state=std::move(state);
    if(guided){c.spatial_info.enabled=1;c.spatial_info.kernel_radius=2;c.spatial_info.guide_code_limit=2;}
    auto& info=c.progressive_info;info.enabled=1;info.bounces=bounces;info.seed=c.indirect_info.seed;
    info.samples_per_batch=16;info.max_batches=4096;
    c.indirect_info.max_bounces=bounces;
}

void progressive_prepare(Context& c,FrameResources& frame,uint32_t width,uint32_t height) {
    auto& state=*c.progressive_state;
    if(state.width!=width||state.height!=height) {
        state.clear_targets();
        c.make_image(state.mean,width,height,VK_FORMAT_R32G32B32A32_SFLOAT,false,VK_SAMPLE_COUNT_1_BIT,VK_IMAGE_USAGE_STORAGE_BIT);
        c.make_image(state.display,width,height,VK_FORMAT_R8G8B8A8_UNORM,false,VK_SAMPLE_COUNT_1_BIT,VK_IMAGE_USAGE_STORAGE_BIT);
        c.make_buffer(state.status,4,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT|VK_BUFFER_USAGE_TRANSFER_SRC_BIT,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        std::array<VkDescriptorPoolSize,3> sizes{{{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,3},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1},{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,4}}};
        VkDescriptorPoolCreateInfo pool{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pool.maxSets=1;pool.poolSizeCount=c.spatial_info.enabled?3u:2u;pool.pPoolSizes=sizes.data();
        acquire(state.pool,"progressive descriptor pool",[&](auto* out){return vkCreateDescriptorPool(c.device,&pool,nullptr,out);});
        VkDescriptorSetAllocateInfo allocation{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        allocation.descriptorPool=state.pool;allocation.descriptorSetCount=1;allocation.pSetLayouts=&state.layout;
        check(vkAllocateDescriptorSets(c.device,&allocation,&state.set),"progressive descriptor set");
        state.width=width;state.height=height;state.pending_reset=true;
    }
    // The raster target can also change after a sensor-only capture. Rebind its
    // current color view without discarding the independent previous RGB mean.
    std::array<VkDescriptorImageInfo,3> images{{{VK_NULL_HANDLE,frame.images[0].view,VK_IMAGE_LAYOUT_GENERAL},
        {VK_NULL_HANDLE,state.mean.view,VK_IMAGE_LAYOUT_GENERAL},{VK_NULL_HANDLE,state.display.view,VK_IMAGE_LAYOUT_GENERAL}}};
    VkDescriptorBufferInfo status{state.status.handle,0,4};
    std::array<VkWriteDescriptorSet,4> writes{};
    for(uint32_t i=0;i<4;++i){auto& w=writes[i];w.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet=state.set;w.dstBinding=i;w.descriptorCount=1;
        w.descriptorType=i==3?VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        if(i==3)w.pBufferInfo=&status;else w.pImageInfo=&images[i];}
    vkUpdateDescriptorSets(c.device,4,writes.data(),0,nullptr);
    if(c.spatial_info.enabled) {
        const uint32_t indices[4]={5,1,2,3};
        std::array<VkDescriptorImageInfo,4> sampled{};
        std::array<VkWriteDescriptorSet,4> extra{};
        for(uint32_t i=0;i<4;++i){
            sampled[i]={state.sampler,frame.images[indices[i]].view,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
            auto& w=extra[i];w.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;w.dstSet=state.set;
            w.dstBinding=4+i;w.descriptorCount=1;w.descriptorType=VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;w.pImageInfo=&sampled[i];
        }
        vkUpdateDescriptorSets(c.device,4,extra.data(),0,nullptr);
    }
}

VkImageMemoryBarrier progressive_barrier(VkImage image,VkImageLayout old_layout,VkImageLayout new_layout,
                                         VkAccessFlags source,VkAccessFlags destination) {
    VkImageMemoryBarrier value{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    value.image=image;value.oldLayout=old_layout;value.newLayout=new_layout;
    value.srcAccessMask=source;value.dstAccessMask=destination;
    value.srcQueueFamilyIndex=value.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;
    value.subresourceRange={VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1};return value;
}
void progressive_record(Context& c,FrameResources& frame,const PrParams& params,uint32_t batch,size_t status_offset) {
    auto& state=*c.progressive_state;auto commands=frame.commands;
    std::array<VkImageMemoryBarrier,3> before{{
        progressive_barrier(frame.images[0].handle,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,VK_IMAGE_LAYOUT_GENERAL,VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,VK_ACCESS_SHADER_READ_BIT),
        progressive_barrier(state.mean.handle,state.initialized?VK_IMAGE_LAYOUT_GENERAL:VK_IMAGE_LAYOUT_UNDEFINED,VK_IMAGE_LAYOUT_GENERAL,state.initialized?VK_ACCESS_MEMORY_WRITE_BIT:0,VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT),
        progressive_barrier(state.display.handle,state.initialized?VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:VK_IMAGE_LAYOUT_UNDEFINED,VK_IMAGE_LAYOUT_GENERAL,state.initialized?VK_ACCESS_TRANSFER_READ_BIT:0,VK_ACCESS_SHADER_WRITE_BIT)}};
    vkCmdPipelineBarrier(commands,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,0,nullptr,0,nullptr,3,before.data());
    std::array<VkImageMemoryBarrier,4> guide_barriers{};
    if(c.spatial_info.enabled) {
        const uint32_t indices[4]={5,1,2,3};
        for(uint32_t i=0;i<4;++i)guide_barriers[i]=progressive_barrier(frame.images[indices[i]].handle,
            (i==0&&!frame.guide_initialized)?VK_IMAGE_LAYOUT_UNDEFINED:VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,VK_ACCESS_SHADER_READ_BIT);
        frame.guide_initialized=true;
        vkCmdPipelineBarrier(commands,VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0,0,nullptr,0,nullptr,4,guide_barriers.data());
    }
    vkCmdFillBuffer(commands,state.status.handle,0,4,0);
    VkBufferMemoryBarrier status{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    status.buffer=state.status.handle;status.offset=0;status.size=4;
    status.srcQueueFamilyIndex=status.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;
    status.srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT;status.dstAccessMask=VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(commands,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,0,nullptr,1,&status,0,nullptr);
    struct Push {uint32_t width,height,batch,mode;float exposure;} push{state.width,state.height,batch,c.display_transform,params.options[0]};
    vkCmdBindPipeline(commands,VK_PIPELINE_BIND_POINT_COMPUTE,state.pipeline);
    vkCmdBindDescriptorSets(commands,VK_PIPELINE_BIND_POINT_COMPUTE,state.pipeline_layout,0,1,&state.set,0,nullptr);
    vkCmdPushConstants(commands,state.pipeline_layout,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(push),&push);
    vkCmdDispatch(commands,(state.width+7)/8,(state.height+7)/8,1);
    auto after=progressive_barrier(state.display.handle,VK_IMAGE_LAYOUT_GENERAL,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,VK_ACCESS_SHADER_WRITE_BIT,VK_ACCESS_TRANSFER_READ_BIT);
    status.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT;status.dstAccessMask=VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(commands,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,0,0,nullptr,1,&status,1,&after);
    if(c.spatial_info.enabled) {
        for(auto& barrier:guide_barriers){
            barrier.oldLayout=VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;barrier.newLayout=VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            barrier.srcAccessMask=VK_ACCESS_SHADER_READ_BIT;barrier.dstAccessMask=VK_ACCESS_TRANSFER_READ_BIT;
        }
        vkCmdPipelineBarrier(commands,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,
            0,0,nullptr,0,nullptr,4,guide_barriers.data());
    }
    VkBufferCopy copy{0,status_offset,4};vkCmdCopyBuffer(commands,state.status.handle,frame.readback.handle,1,&copy);
}

void progressive_read(Context& c,void* output,size_t bytes) {
    auto& state=*c.progressive_state;auto& info=c.progressive_info;
    require(!c.poisoned&&info.batches&&output&&bytes==static_cast<size_t>(state.width)*state.height*16,
            "progressive mean read requires completed RGB history and exact RGBA32F extent");
    try {
    auto& frame=*c.resident_state->target;
    if(!state.readback.handle) {
        c.make_buffer(state.readback,bytes,VK_BUFFER_USAGE_TRANSFER_DST_BIT,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
        check(vkMapMemory(c.device,state.readback.memory,0,VK_WHOLE_SIZE,0,&state.mapping),"progressive mean map");
    }
    check(vkResetFences(c.device,1,&frame.fence),"progressive read fence");
    check(vkResetCommandBuffer(frame.commands,0),"progressive read command");
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};begin.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    check(vkBeginCommandBuffer(frame.commands,&begin),"progressive read begin");
    auto before=progressive_barrier(state.mean.handle,VK_IMAGE_LAYOUT_GENERAL,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,VK_ACCESS_SHADER_WRITE_BIT,VK_ACCESS_TRANSFER_READ_BIT);
    vkCmdPipelineBarrier(frame.commands,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,0,0,nullptr,0,nullptr,1,&before);
    VkBufferImageCopy copy{};copy.imageSubresource={VK_IMAGE_ASPECT_COLOR_BIT,0,0,1};copy.imageExtent={state.width,state.height,1};
    vkCmdCopyImageToBuffer(frame.commands,state.mean.handle,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,state.readback.handle,1,&copy);
    auto after=progressive_barrier(state.mean.handle,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,VK_IMAGE_LAYOUT_GENERAL,VK_ACCESS_TRANSFER_READ_BIT,VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT);
    vkCmdPipelineBarrier(frame.commands,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,0,nullptr,0,nullptr,1,&after);
    VkBufferMemoryBarrier host{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};host.buffer=state.readback.handle;host.size=bytes;
    host.srcQueueFamilyIndex=host.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;host.srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT;host.dstAccessMask=VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(frame.commands,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_HOST_BIT,0,0,nullptr,1,&host,0,nullptr);
    check(vkEndCommandBuffer(frame.commands),"progressive read end");
    const auto errors=c.errors.load();submit_and_wait(c,frame);
    require(c.errors.load()==errors,"Vulkan validation failed reading progressive mean");
    std::memcpy(output,state.mapping,bytes);++info.mean_readbacks;info.mean_readback_bytes+=bytes;
    c.telemetry.readback_bytes+=bytes;
    }catch(...){c.poisoned=true;throw;}
}

// Debug readback only; never a filtering input. Valid only after a successful
// first-batch RGB capture, before any later RGB or sensor-only exposure.
void spatial_read(Context& c,void* output,size_t bytes) {
    require(c.spatial_info.enabled&&!c.poisoned&&c.spatial_info.last_capture_filtered,
            "spatial guide requires the latest capture to be a filtered RGB batch");
    auto& state=*c.progressive_state;
    require(output&&bytes==static_cast<size_t>(state.width)*state.height*4,"spatial guide byte extent mismatch");
    try {
        auto& frame=*c.resident_state->target;
        if(!state.readback.handle){
            c.make_buffer(state.readback,static_cast<size_t>(state.width)*state.height*16,VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
            check(vkMapMemory(c.device,state.readback.memory,0,VK_WHOLE_SIZE,0,&state.mapping),"spatial guide map");
        }
        check(vkResetFences(c.device,1,&frame.fence),"spatial read fence");
        check(vkResetCommandBuffer(frame.commands,0),"spatial read command");
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};begin.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        check(vkBeginCommandBuffer(frame.commands,&begin),"spatial read begin");
        VkBufferImageCopy copy{};copy.imageSubresource={VK_IMAGE_ASPECT_COLOR_BIT,0,0,1};copy.imageExtent={state.width,state.height,1};
        vkCmdCopyImageToBuffer(frame.commands,frame.images[5].handle,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,state.readback.handle,1,&copy);
        VkBufferMemoryBarrier host{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};host.buffer=state.readback.handle;host.size=bytes;
        host.srcQueueFamilyIndex=host.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;host.srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT;host.dstAccessMask=VK_ACCESS_HOST_READ_BIT;
        vkCmdPipelineBarrier(frame.commands,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_HOST_BIT,0,0,nullptr,1,&host,0,nullptr);
        check(vkEndCommandBuffer(frame.commands),"spatial read end");
        const auto errors=c.errors.load();submit_and_wait(c,frame);
        require(c.errors.load()==errors,"Vulkan validation failed reading spatial guide");
        std::memcpy(output,state.mapping,bytes);++c.spatial_info.guide_readbacks;c.spatial_info.guide_readback_bytes+=bytes;
        c.telemetry.readback_bytes+=bytes;
    }catch(...){c.poisoned=true;throw;}
}
