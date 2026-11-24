// Shared recording and result accounting for single-camera and batched captures.
void upload_camera_params(Context& c,FrameResources& frame,const PrParams& params,
                          bool progressive_rgb,uint32_t progressive_batch) {
    const bool shared=c.shared_input;
    auto& info=c.resident_properties;
    PrParams submitted=params;
    // Reserved caller words never opt into a display change; only the checked setter does.
    submitted.options[2]=static_cast<float>(c.display_transform);
    if (c.shared_blas_compatible) submitted.options[3]=shared ? -1.0f : 0.0f;
    // Resident shaders do not use the legacy triangle-count word. The caller
    // still supplies/validates it; only explicit indirect variants read a seed.
    if(c.indirect_info.enabled) submitted.counts[1]=c.indirect_info.seed;
    if(progressive_rgb)submitted.options[3]=static_cast<float>(2*progressive_batch+c.progressive_info.bounces-1);
    c.upload(frame.uniform, &submitted, sizeof(submitted));
    info.parameter_upload_bytes += sizeof(params);
}
void record_camera(Context& c,FrameResources& frame,const PrParams& params,uint32_t channel_mask,
                   uint32_t camera_index,uint32_t progressive_batch,size_t output_bytes) {
    auto& state=*c.resident_state;
    const auto& info=c.resident_properties;
    const uint32_t instance_count=info.instance_count,width=frame.width,height=frame.height;
    const bool shared=c.shared_input,progressive_rgb=c.progressive_info.enabled && params.counts[2]!=0;
    const bool msaa_color=c.msaa_info.requested_samples==4 && params.counts[2]!=0;
    const bool center_pass=!msaa_color || (channel_mask&14u)!=0;
    const size_t progressive_status_offset=(output_bytes+3)&~size_t(3);
    if (frame.fence) check(vkResetFences(c.device,1,&frame.fence),"resident reset fence");
    check(vkResetCommandBuffer(frame.commands, 0), "resident reset command");
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    check(vkBeginCommandBuffer(frame.commands, &begin), "resident begin frame");
    if (frame.timestamps) {
        vkCmdResetQueryPool(frame.commands,frame.timestamps,0,3);
        vkCmdWriteTimestamp(frame.commands,VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,frame.timestamps,0);
    }
    if (c.ray_query && !shared) record_ray_update(c,state,frame.commands);
    const VkBuffer input_buffers[]{state.transforms.handle,frame.uniform.handle,state.tlas.handle,state.studio.handle};
    std::array<VkBufferMemoryBarrier, 4> inputs{};
    uint32_t input_count=0;
    for (uint32_t i = 0; i < inputs.size(); ++i) {
        if (shared && (i==0 || i==2)) continue; // ownership barrier covers transforms; shared fragment bypasses TLAS
        auto& barrier = inputs[input_count++];
        barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        barrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
        barrier.dstAccessMask = (i == 1 || i == 3) ? VK_ACCESS_UNIFORM_READ_BIT : VK_ACCESS_SHADER_READ_BIT;
        barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.buffer = input_buffers[i]; barrier.size = VK_WHOLE_SIZE;
    }
    vkCmdPipelineBarrier(frame.commands, VK_PIPELINE_STAGE_HOST_BIT,
        VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
        0, nullptr, input_count, inputs.data(), 0, nullptr);
    if (camera_index!=UINT32_MAX) {
        const VkDeviceSize base=(VkDeviceSize(info.instance_count)+camera_index)*sizeof(GpuTransform);
        const VkBufferCopy fields[]={
            {base+offsetof(GpuTransform,local),offsetof(PrParams,view),16*sizeof(float)},
            {base+12*sizeof(float),offsetof(PrParams,camera),3*sizeof(float)}
        };
        VkBufferMemoryBarrier barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        barrier.srcQueueFamilyIndex=barrier.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;
        barrier.buffer=frame.uniform.handle;
        barrier.size=sizeof(PrParams);
        barrier.srcAccessMask=VK_ACCESS_HOST_WRITE_BIT;
        barrier.dstAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT;
        vkCmdPipelineBarrier(frame.commands,VK_PIPELINE_STAGE_HOST_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0,0,nullptr,1,&barrier,0,nullptr);
        vkCmdCopyBuffer(frame.commands,state.transforms.handle,frame.uniform.handle,2,fields);
        barrier.srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask=VK_ACCESS_UNIFORM_READ_BIT;
        vkCmdPipelineBarrier(frame.commands,VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_VERTEX_SHADER_BIT|VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,0,0,nullptr,1,&barrier,0,nullptr);
    }
    VkDeviceSize offset=0;
    if (center_pass) {
    std::array<VkClearValue, 6> clear{};
    clear[0].color.float32[3] = 1;
    if (c.studio.environment[3] != 0 && params.counts[2] != 0) {
        const auto mapped=c.display_transform==1 ? neutral_clear(c.studio.environment,params.options[0]) : std::array<double,3>{};
        for (uint32_t k=0; k<3; ++k) {
            double linear=c.display_transform==1 ? std::clamp(mapped[k],0.0,1.0) :
                std::clamp(static_cast<double>(c.studio.environment[k])*params.options[0],0.0,1.0);
            clear[0].color.float32[k]=static_cast<float>(linear<=0.0031308 ? 12.92*linear :
                1.055*std::pow(linear,1.0/2.4)-0.055);
        }
    }
    if(c.progressive_info.shader_compatible) {
        for(uint32_t k=0;k<3;++k)clear[0].color.float32[k]=c.studio.environment[3]!=0&&params.counts[2]?c.studio.environment[k]:0.f;
    }
    clear[1].color.float32[0] = std::numeric_limits<float>::infinity();
    clear[4].depthStencil.depth = 1;
    VkRenderPassBeginInfo pass{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    pass.renderPass = c.render_pass; pass.framebuffer = frame.framebuffer;
    pass.renderArea.extent = {width, height}; pass.clearValueCount = 5; pass.pClearValues = clear.data();
    vkCmdBeginRenderPass(frame.commands, &pass, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(frame.commands, VK_PIPELINE_BIND_POINT_GRAPHICS, c.pipeline);
    VkViewport viewport{0,0,static_cast<float>(width),static_cast<float>(height),0,1};
    VkRect2D scissor{{0,0},{width,height}};
    vkCmdSetViewport(frame.commands, 0, 1, &viewport);
    vkCmdSetScissor(frame.commands, 0, 1, &scissor);
    offset = 0;
    vkCmdBindVertexBuffers(frame.commands, 0, 1, &state.vertices.handle, &offset);
    vkCmdBindIndexBuffer(frame.commands, state.indices.handle, 0, VK_INDEX_TYPE_UINT32);
    vkCmdBindDescriptorSets(frame.commands, VK_PIPELINE_BIND_POINT_GRAPHICS, c.pipeline_layout,
                           0, 1, &frame.scene_set, 0, nullptr);
    if (c.environment_info.loaded && params.counts[2] && !msaa_color) {
        vkCmdBindPipeline(frame.commands,VK_PIPELINE_BIND_POINT_GRAPHICS,state.environment->sky_pipeline);
        vkCmdDraw(frame.commands,3,1,0,0);
        vkCmdBindPipeline(frame.commands,VK_PIPELINE_BIND_POINT_GRAPHICS,c.pipeline);
    }
    for (uint32_t slot = 0; slot < instance_count; ++slot) {
        const auto& mesh = state.meshes[state.instances[slot].mesh_index];
        vkCmdDrawIndexed(frame.commands, mesh.index_count, 1, mesh.first_index,
                         static_cast<int32_t>(mesh.first_vertex), slot);
    }
    vkCmdEndRenderPass(frame.commands);
    }
    if (msaa_color) msaa_record(c,frame,params);
    if(progressive_rgb && c.spatial_info.enabled && progressive_batch==0)spatial_record_guide(c,frame,width,height);
    if(progressive_rgb)progressive_record(c,frame,params,progressive_batch,progressive_status_offset);
    if (frame.timestamps)
        vkCmdWriteTimestamp(frame.commands,VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,frame.timestamps,1);
    const size_t bytes_per_pixel[]{4,4,4,1};
    const size_t pixels = static_cast<size_t>(width)*height;
    offset = 0;
    for (uint32_t i = 0; i < 4; ++i) {
        if (!(channel_mask & (1u<<i))) continue;
        // All preceding formats are four bytes; VALID (one byte) is last.
        // Thus odd width/height and VALID-only still satisfy copy alignment.
        VkBufferImageCopy copy{};
        copy.bufferOffset = offset;
        copy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        copy.imageSubresource.layerCount = 1;
        copy.imageExtent = {width,height,1};
        vkCmdCopyImageToBuffer(frame.commands, i==0&&progressive_rgb?c.progressive_state->display.handle:frame.images[i].handle, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                              frame.readback.handle, 1, &copy);
        offset += pixels*bytes_per_pixel[i];
    }
    VkBufferMemoryBarrier host{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    host.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; host.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    host.srcQueueFamilyIndex = host.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    host.buffer = frame.readback.handle; host.size = VK_WHOLE_SIZE;
    vkCmdPipelineBarrier(frame.commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                         0, 0, nullptr, 1, &host, 0, nullptr);
    if (frame.timestamps)
        vkCmdWriteTimestamp(frame.commands,VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,frame.timestamps,2);
    check(vkEndCommandBuffer(frame.commands), "resident end frame");
}
struct CameraGpuTiming { uint64_t render{},copy{},total{},samples{}; };
CameraGpuTiming camera_gpu_timing(Context& c,FrameResources& frame,const SubmitTiming& timing) {
    CameraGpuTiming result;
    if (frame.timestamps) {
        uint64_t ticks[3]{};
        check(vkGetQueryPoolResults(c.device,frame.timestamps,0,3,sizeof(ticks),ticks,sizeof(uint64_t),
            VK_QUERY_RESULT_64_BIT),"resident timestamp read");
        uint32_t bits=c.profile.timestamp_valid_bits;
        uint64_t mask=bits==64 ? UINT64_MAX : ((1ull<<bits)-1);
        double period=c.profile.timestamp_period_ns;
        // A possible multiple counter wrap cannot be reconstructed honestly.
        // Keep the capture, omit that sample instead of reporting a false time.
        double wrap_ns=std::ldexp(period,static_cast<int>(bits));
        if (static_cast<double>(timing.submit)+timing.wait < wrap_ns) {
            double render_ns=static_cast<double>((ticks[1]-ticks[0])&mask)*period;
            double copy_ns=static_cast<double>((ticks[2]-ticks[1])&mask)*period;
            double total_ns=static_cast<double>((ticks[2]-ticks[0])&mask)*period;
            if (std::isfinite(total_ns) && total_ns < static_cast<double>(UINT64_MAX)) {
                result.render=static_cast<uint64_t>(render_ns); result.copy=static_cast<uint64_t>(copy_ns);
                result.total=static_cast<uint64_t>(total_ns); result.samples=1;
            }
        }
    }
    return result;
}
void count_camera(Context& c,const PrParams& params,bool msaa_color,bool center_pass) {
    const bool shared=c.shared_input;
    auto& state=*c.resident_state;
    auto& info=c.resident_properties;
    if (c.normal_info.loaded && params.counts[2]) ++c.normal_info.rgb_captures_with_normals;
    if (c.texture_info.loaded && params.counts[2]) ++c.texture_info.rgb_captures_with_textures;
    if (c.environment_info.loaded && params.counts[2]) {
        ++c.environment_info.rgb_captures_with_environment;
        ++c.environment_info.sky_draws;
    }
    if (c.ray_query) {
        if (!shared) {
            if (state.acceleration->built) ++c.ray_info.tlas_updates;
            else ++c.ray_info.tlas_builds;
            state.acceleration->built=true;
        }
        if (params.counts[2]) ++c.ray_info.shadow_captures;
    }
    if (msaa_color) { ++c.msaa_info.color_passes; ++c.msaa_info.resolve_passes; }
    if (center_pass) ++c.msaa_info.center_passes;
    ++c.properties.captures; ++info.captures;
}
