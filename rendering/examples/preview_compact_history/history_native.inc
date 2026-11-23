// Included after ProgressiveState. All resources share its existing teardown.
struct HistoryUniform {
    float view[16],previous_view[16],intrinsics[4],previous_intrinsics[4];
    uint32_t controls[4];
};
static_assert(sizeof(HistoryUniform)==176,"history UBO ABI");

void history_setup(Context& c,const uint32_t* code,size_t bytes) {
    require(c.progressive_state&&c.progressive_info.enabled&&!c.properties.captures&&!c.history_enabled,
            "history setup requires uncaptured compact owner");
    const auto interface=spirv_interface(code,bytes,5);
    require(interface.valid&&interface.main&&interface.bindings==0xf1efu&&
            named_shader_marker(code,bytes,93,"PR_PREVIEW_COMPACT_HISTORY_ABI",5),"history shader ABI mismatch");
    auto& state=*c.progressive_state;
    VkShaderModuleCreateInfo module{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};module.codeSize=bytes;module.pCode=code;
    acquire(state.history_shader,"compact history shader",[&](auto* out){return vkCreateShaderModule(c.device,&module,nullptr,out);});
    VkComputePipelineCreateInfo pipeline{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipeline.stage={VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    pipeline.stage.stage=VK_SHADER_STAGE_COMPUTE_BIT;pipeline.stage.module=state.history_shader;pipeline.stage.pName="main";
    pipeline.layout=state.pipeline_layout;
    VkPipeline temporary{};const auto status=vkCreateComputePipelines(c.device,VK_NULL_HANDLE,1,&pipeline,nullptr,&temporary);
    if(temporary){state.history_pipeline=temporary;++live_resources;}check(status,"compact history pipeline");
    c.history_enabled=true;
}

void history_prepare(Context& c) {
    auto& s=*c.progressive_state;
    require(c.history_enabled&&c.history_armed,"RGB history capture requires explicit metadata arm");
    const uint64_t pixels=static_cast<uint64_t>(s.width)*s.height;
    require(pixels*32<=c.storage_buffer_limit,"history storage buffer range exceeded");
    if(!s.history_current.handle) {
        const auto usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        c.make_buffer(s.history_current,pixels*16,usage,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        for(auto& buffer:s.history_pixels)c.make_buffer(buffer,pixels*32,usage,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        c.make_buffer(s.history_uniform,sizeof(HistoryUniform),VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    }
    const uint32_t next=s.history_read_slot^1u;
    const std::array<VkDescriptorBufferInfo,4> buffers{{
        {s.history_current.handle,0,pixels*16},{s.history_pixels[s.history_read_slot].handle,0,pixels*32},
        {s.history_pixels[next].handle,0,pixels*32},{s.history_uniform.handle,0,sizeof(HistoryUniform)}}};
    std::array<VkWriteDescriptorSet,4> writes{};
    for(uint32_t i=0;i<4;++i){auto& w=writes[i];w.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet=s.set;w.dstBinding=12+i;w.descriptorCount=1;
        w.descriptorType=i==3?VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w.pBufferInfo=&buffers[i];}
    vkUpdateDescriptorSets(c.device,4,writes.data(),0,nullptr);
}

void history_upload(Context& c,const PrParams& params) {
    auto& s=*c.progressive_state;HistoryUniform uniform{};
    std::memcpy(uniform.view,params.view,sizeof(uniform.view));
    std::memcpy(uniform.intrinsics,params.intrinsics,sizeof(uniform.intrinsics));
    const auto& previous=s.history_valid?s.history_params:params;
    std::memcpy(uniform.previous_view,previous.view,sizeof(uniform.previous_view));
    std::memcpy(uniform.previous_intrinsics,previous.intrinsics,sizeof(uniform.previous_intrinsics));
    uniform.controls[0]=s.width;uniform.controls[1]=s.height;
    uniform.controls[2]=s.history_valid&&!c.history_reset;uniform.controls[3]=8;
    c.upload(s.history_uniform,&uniform,sizeof(uniform));
    c.history_uniform_bytes+=sizeof(uniform);
}

void history_record(Context& c,FrameResources& frame) {
    auto& s=*c.progressive_state;
    // First pass has completed every current incoming cell before any temporal
    // neighborhood reads. Previous and next are distinct allocations.
    VkMemoryBarrier ready{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    ready.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT;
    ready.dstAccessMask=VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(frame.commands,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0,1,&ready,0,nullptr,0,nullptr);
    vkCmdBindPipeline(frame.commands,VK_PIPELINE_BIND_POINT_COMPUTE,s.history_pipeline);
    // Same descriptor set and the existing 20-byte display push constants.
    vkCmdDispatch(frame.commands,(s.width+7)/8,(s.height+7)/8,1);
}

void history_commit(Context& c) {
    auto& s=*c.progressive_state;
    require(c.history_armed&&c.history_rendered,"history transaction has no completed GPU result");
    c.history_last_reset=c.history_reset||!s.history_valid;
    if(c.history_last_reset)++c.history_resets;
    s.history_read_slot^=1u;s.history_valid=true;s.history_params=s.history_pending_params;
    ++c.history_captures;c.history_last_sequence=c.history_sequence;c.history_armed=false;c.history_rendered=false;
}

void history_read(Context& c,uint32_t plane,void* output,size_t bytes) {
    require(c.history_enabled&&c.sparse_last_rgb&&!c.poisoned&&c.progressive_state&&plane<2,
            "history read requires latest successful RGB");
    auto& s=*c.progressive_state;auto& frame=*c.resident_state->target;
    require(s.history_valid&&output&&bytes==static_cast<size_t>(s.width)*s.height*(plane==0?16:32),"history read extent mismatch");
    try {
        if(!s.history_debug.handle){
            c.make_buffer(s.history_debug,static_cast<size_t>(s.width)*s.height*32,VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
            check(vkMapMemory(c.device,s.history_debug.memory,0,VK_WHOLE_SIZE,0,&s.history_debug_mapping),"history debug map");
        }
        auto& buffer=plane==0?s.history_current:s.history_pixels[s.history_read_slot];
        check(vkResetFences(c.device,1,&frame.fence),"history read fence");
        check(vkResetCommandBuffer(frame.commands,0),"history read command");
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};begin.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        check(vkBeginCommandBuffer(frame.commands,&begin),"history read begin");
        VkBufferMemoryBarrier barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        barrier.buffer=buffer.handle;barrier.size=bytes;barrier.srcQueueFamilyIndex=barrier.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;
        barrier.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT;barrier.dstAccessMask=VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(frame.commands,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,0,0,nullptr,1,&barrier,0,nullptr);
        VkBufferCopy copy{0,0,bytes};vkCmdCopyBuffer(frame.commands,buffer.handle,s.history_debug.handle,1,&copy);
        barrier.buffer=s.history_debug.handle;barrier.srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT;barrier.dstAccessMask=VK_ACCESS_HOST_READ_BIT;
        vkCmdPipelineBarrier(frame.commands,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_HOST_BIT,0,0,nullptr,1,&barrier,0,nullptr);
        check(vkEndCommandBuffer(frame.commands),"history read end");
        const auto errors=c.errors.load();submit_and_wait(c,frame);
        require(c.errors.load()==errors,"history read validation error");
        std::memcpy(output,s.history_debug_mapping,bytes);
        ++c.history_reads;c.history_read_bytes+=bytes;c.telemetry.readback_bytes+=bytes;
    }catch(...){c.poisoned=true;throw;}
}
