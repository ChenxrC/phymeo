void edge_bind(Context& c,FrameResources& frame) {
    auto& s=*c.progressive_state;
    require(s.edge_pipeline&&c.edge_limit,"explicit edge pipeline setup required before capture");
    VkDescriptorImageInfo extra{s.sampler,frame.images[5].view,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkWriteDescriptorSet compute{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    compute.dstSet=s.set;compute.dstBinding=12;compute.descriptorCount=1;
    compute.descriptorType=VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;compute.pImageInfo=&extra;
    vkUpdateDescriptorSets(c.device,1,&compute,0,nullptr);
    const std::array<VkDescriptorImageInfo,3> images{{
        {s.sampler,s.mean.view,VK_IMAGE_LAYOUT_GENERAL},
        {s.sampler,frame.images[2].view,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
        {s.sampler,frame.images[6].view,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL}}};
    VkDescriptorBufferInfo status{s.status.handle,0,64};
    std::array<VkWriteDescriptorSet,4> writes{};
    for(uint32_t i=0;i<4;++i){auto& w=writes[i];w.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet=c.resident_state->descriptor_set;w.dstBinding=16+i;w.descriptorCount=1;
        w.descriptorType=i==1?VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        if(i==1)w.pBufferInfo=&status;else w.pImageInfo=&images[i==0?0:i-1];}
    vkUpdateDescriptorSets(c.device,4,writes.data(),0,nullptr);
}
void edge_record(Context& c,FrameResources& frame) {
    auto& s=*c.progressive_state;
    if(!frame.edge_framebuffer){
        const std::array<VkImageView,2> views{frame.images[5].view,frame.images[4].view};
        VkFramebufferCreateInfo f{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
        f.renderPass=s.edge_pass;f.attachmentCount=2;f.pAttachments=views.data();
        f.width=s.width;f.height=s.height;f.layers=1;
        acquire(frame.edge_framebuffer,"edge framebuffer",[&](auto* out){return vkCreateFramebuffer(c.device,&f,nullptr,out);});
    }
    std::array<VkClearValue,2> clears{};
    VkRenderPassBeginInfo begin{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    begin.renderPass=s.edge_pass;begin.framebuffer=frame.edge_framebuffer;
    begin.renderArea.extent={s.width,s.height};begin.clearValueCount=2;begin.pClearValues=clears.data();
    vkCmdBeginRenderPass(frame.commands,&begin,VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(frame.commands,VK_PIPELINE_BIND_POINT_GRAPHICS,s.edge_pipeline);
    VkViewport viewport{0,0,static_cast<float>(s.width),static_cast<float>(s.height),0,1};
    VkRect2D scissor{{0,0},{s.width,s.height}};
    vkCmdSetViewport(frame.commands,0,1,&viewport);vkCmdSetScissor(frame.commands,0,1,&scissor);
    VkDeviceSize offset=0;auto& resident=*c.resident_state;
    vkCmdBindVertexBuffers(frame.commands,0,1,&resident.vertices.handle,&offset);
    vkCmdBindIndexBuffer(frame.commands,resident.indices.handle,0,VK_INDEX_TYPE_UINT32);
    vkCmdBindDescriptorSets(frame.commands,VK_PIPELINE_BIND_POINT_GRAPHICS,c.pipeline_layout,0,1,&resident.descriptor_set,0,nullptr);
    for(uint32_t slot=0;slot<resident.instances.size();++slot){
        const auto& mesh=resident.meshes[resident.instances[slot].mesh_index];
        vkCmdDrawIndexed(frame.commands,mesh.index_count,1,mesh.first_index,static_cast<int32_t>(mesh.first_vertex),slot);
    }
    vkCmdEndRenderPass(frame.commands);
}