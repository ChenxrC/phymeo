    // Complete selection before any selective fragment reads it.
    auto selected_ready=progressive_barrier(state.mean.handle,VK_IMAGE_LAYOUT_GENERAL,VK_IMAGE_LAYOUT_GENERAL,
        VK_ACCESS_SHADER_WRITE_BIT,VK_ACCESS_SHADER_READ_BIT);
    status.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT;status.dstAccessMask=VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(commands,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        0,0,nullptr,1,&status,1,&selected_ready);
    edge_record(c,frame);
    // The extra pass finishes all selected incoming values before the merge.
    status.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT;status.dstAccessMask=VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT;
    auto mean_ready=progressive_barrier(state.mean.handle,VK_IMAGE_LAYOUT_GENERAL,VK_IMAGE_LAYOUT_GENERAL,
        VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT,VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT);
    auto rescue_ready=progressive_barrier(frame.images[5].handle,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,VK_ACCESS_SHADER_READ_BIT);
    auto display_ready=progressive_barrier(state.display.handle,VK_IMAGE_LAYOUT_GENERAL,VK_IMAGE_LAYOUT_GENERAL,
        VK_ACCESS_SHADER_WRITE_BIT,VK_ACCESS_SHADER_WRITE_BIT);
    const std::array<VkImageMemoryBarrier,3> edge_ready{mean_ready,rescue_ready,display_ready};
    vkCmdPipelineBarrier(commands,VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT|VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT|VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,0,nullptr,1,&status,3,edge_ready.data());
    push.batch=1u;
    vkCmdBindPipeline(commands,VK_PIPELINE_BIND_POINT_COMPUTE,state.pipeline);
    vkCmdBindDescriptorSets(commands,VK_PIPELINE_BIND_POINT_COMPUTE,state.pipeline_layout,0,1,&state.set,0,nullptr);
    vkCmdPushConstants(commands,state.pipeline_layout,VK_SHADER_STAGE_COMPUTE_BIT,0,20,&push);
    vkCmdDispatch(commands,(state.width+7)/8,(state.height+7)/8,1);
    auto rescue_readable=progressive_barrier(frame.images[5].handle,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        VK_ACCESS_SHADER_READ_BIT,VK_ACCESS_TRANSFER_READ_BIT);
    vkCmdPipelineBarrier(commands,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,0,0,nullptr,0,nullptr,1,&rescue_readable);