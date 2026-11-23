// Included after ProgressiveState; buffer shares normal clear_targets teardown.
void lowres_record(Context& c,FrameResources& frame,uint32_t mode,float exposure) {
    auto& s=*c.progressive_state;
    struct Push {uint32_t width,height,batch,mode;float exposure;} push{s.width,s.height,0,mode|256u,exposure};
    vkCmdPushConstants(frame.commands,s.pipeline_layout,VK_SHADER_STAGE_COMPUTE_BIT,0,20,&push);
    const uint32_t w=(s.width+3)/4,h=(s.height+3)/4;
    vkCmdDispatch(frame.commands,(w+7)/8,(h+7)/8,1);
    VkMemoryBarrier ready{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    ready.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT;
    ready.dstAccessMask=VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(frame.commands,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0,1,&ready,0,nullptr,0,nullptr);
}
void lowres_read(Context& c,void* output,size_t bytes) {
    require(c.history_enabled&&c.sparse_last_rgb&&!c.poisoned&&c.progressive_state,"lowres read requires successful RGB");
    auto& s=*c.progressive_state;auto& frame=*c.resident_state->target;
    require(output&&bytes==static_cast<size_t>((s.width+3)/4)*((s.height+3)/4)*16,"lowres RGBA32F extent mismatch");
    try {
        if(!s.readback.handle){
            c.make_buffer(s.readback,static_cast<size_t>(s.width)*s.height*16,VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
            check(vkMapMemory(c.device,s.readback.memory,0,VK_WHOLE_SIZE,0,&s.mapping),"lowres debug map");
        }
        check(vkResetFences(c.device,1,&frame.fence),"lowres read fence");
        check(vkResetCommandBuffer(frame.commands,0),"lowres read command");
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};begin.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        check(vkBeginCommandBuffer(frame.commands,&begin),"lowres read begin");
        VkBufferMemoryBarrier barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        barrier.buffer=s.low_filtered.handle;barrier.size=bytes;barrier.srcQueueFamilyIndex=barrier.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;
        barrier.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT;barrier.dstAccessMask=VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(frame.commands,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,0,0,nullptr,1,&barrier,0,nullptr);
        VkBufferCopy copy{0,0,bytes};vkCmdCopyBuffer(frame.commands,s.low_filtered.handle,s.readback.handle,1,&copy);
        barrier.buffer=s.readback.handle;barrier.srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT;barrier.dstAccessMask=VK_ACCESS_HOST_READ_BIT;
        vkCmdPipelineBarrier(frame.commands,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_HOST_BIT,0,0,nullptr,1,&barrier,0,nullptr);
        check(vkEndCommandBuffer(frame.commands),"lowres read end");
        const auto errors=c.errors.load();submit_and_wait(c,frame);
        require(c.errors.load()==errors,"lowres debug Vulkan validation failed");
        std::memcpy(output,s.mapping,bytes);c.telemetry.readback_bytes+=bytes;
    }catch(...){c.poisoned=true;throw;}
}