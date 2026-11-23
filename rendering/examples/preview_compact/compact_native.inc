// Included after the resident implementation; resources remain owned by the
// existing FrameResources/ProgressiveState teardown, including partial setup.
void compact_read(Context& c,uint32_t plane,void* output,size_t bytes) {
    require(c.sparse_last_rgb&&!c.poisoned&&c.progressive_state&&c.resident_state&&plane<5,
            "compact read requires latest successful RGB exposure");
    auto& state=*c.progressive_state;auto& frame=*c.resident_state->target;
    const uint32_t width=plane==0?state.width:(state.width+3)/4;
    const uint32_t height=plane==0?state.height:(state.height+3)/4;
    const uint32_t channels_bytes=plane<2?16u:(plane==4?1u:4u);
    require(output&&bytes==static_cast<size_t>(width)*height*channels_bytes,"compact read extent differs");
    const uint32_t image=plane==0?6u:6u+plane;
    try {
        if(!state.readback.handle) {
            c.make_buffer(state.readback,static_cast<size_t>(state.width)*state.height*16,VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
            check(vkMapMemory(c.device,state.readback.memory,0,VK_WHOLE_SIZE,0,&state.mapping),"compact read map");
        }
        check(vkResetFences(c.device,1,&frame.fence),"compact read fence");
        check(vkResetCommandBuffer(frame.commands,0),"compact read command");
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};begin.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        check(vkBeginCommandBuffer(frame.commands,&begin),"compact read begin");
        VkBufferImageCopy copy{};copy.imageSubresource={VK_IMAGE_ASPECT_COLOR_BIT,0,0,1};copy.imageExtent={width,height,1};
        vkCmdCopyImageToBuffer(frame.commands,frame.images[image].handle,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,state.readback.handle,1,&copy);
        VkBufferMemoryBarrier host{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};host.buffer=state.readback.handle;host.size=bytes;
        host.srcQueueFamilyIndex=host.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;
        host.srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT;host.dstAccessMask=VK_ACCESS_HOST_READ_BIT;
        vkCmdPipelineBarrier(frame.commands,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_HOST_BIT,0,0,nullptr,1,&host,0,nullptr);
        check(vkEndCommandBuffer(frame.commands),"compact read end");
        const auto errors=c.errors.load();submit_and_wait(c,frame);
        require(c.errors.load()==errors,"Vulkan validation failed reading compact attributes");
        std::memcpy(output,state.mapping,bytes);
        ++c.spatial_info.guide_readbacks;c.spatial_info.guide_readback_bytes+=bytes;c.telemetry.readback_bytes+=bytes;
    }catch(...){c.poisoned=true;throw;}
}
