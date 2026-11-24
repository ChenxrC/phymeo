void resident_shared_batch(Context& c,const PrResidentBatchView* input,uint32_t count,
                           uint8_t* output,size_t output_bytes,bool device_pixels=false) {
    constexpr uint32_t max_cameras=16;
    constexpr uint64_t pixel_limit=4096ull*4096;
    struct View {
        PrResidentBatchView request{};
        size_t offset{},bytes{};
        bool msaa_color{},center{};
        FrameResources* frame{};
        CameraGpuTiming gpu{};
    };
    const auto start=HostClock::now();
    require(c.shared_input && c.resident_state && !c.poisoned && c.shared_slot.phase==SharedPhase::reading,
            "camera batch requires an acquired shared publication on a healthy resident context");
    auto& scene=*c.resident_state;
    require(!c.progressive_info.shader_compatible && !c.reflection_info.shader_compatible &&
            !c.indirect_info.shader_compatible && (!c.ray_query || scene.acceleration->built),
            "camera batch requires shared direct raster with current shadow geometry");
    require(count && count<=max_cameras && input && (output || device_pixels),"camera batch requires 1 to 16 views and an output");
    if (device_pixels) require(c.device_output.info.phase==1 && output_bytes<=c.device_output.info.capacity_bytes,
                               "device output slot unavailable or batch exceeds its capacity");
    std::array<View,max_cameras> views{};
    uint64_t pixels=0;
    size_t bytes=0,image_bytes=0;
    for (uint32_t i=0;i<count;++i) {
        auto& v=views[i];
        v.request=input[i];
        const auto& r=v.request;
        const uint64_t n=static_cast<uint64_t>(r.width)*r.height;
        require(n && n<=pixel_limit-pixels,"camera batch combined pixel budget exceeded");
        pixels+=n;
        v.offset=device_pixels ? (bytes+3)&~size_t(3) : bytes;
        v.bytes=static_cast<size_t>(n)*compact_pixel_bytes(r.channel_mask);
        require(v.offset<=output_bytes && v.bytes<=output_bytes-v.offset,"camera batch output is too short");
        validate_resident_params(c,r.params,r.width,r.height,output ? output+v.offset : nullptr,
                                 v.bytes,r.channel_mask,true,device_pixels);
        require(r.camera_index==UINT32_MAX || r.camera_index<c.shared_camera_capacity,
                "camera batch mounted slot exceeds declared capacity");
        v.msaa_color=c.msaa_info.requested_samples==4 && (r.channel_mask&1u);
        v.center=!v.msaa_color || (r.channel_mask&14u);
        require(!v.msaa_color || (r.width<=c.msaa_info.max_width && r.height<=c.msaa_info.max_height),
                "camera batch MSAA extent exceeds device limit");
        bytes=v.offset+v.bytes; image_bytes+=v.bytes;
    }
    require(bytes==output_bytes,"camera batch output byte extent mismatch");
    const auto upload_begin=HostClock::now();
    try {
        const auto errors=c.errors.load();
        scene.additional_targets.resize(count-1);
        // Retire old sizes before allocating any new images. A previous batch
        // can have a different size distribution with the same total pixel count.
        for (uint32_t i=0;i<count;++i) {
            const auto& r=views[i].request;
            auto& owner=i ? scene.additional_targets[i-1] : scene.target;
            if (owner && (owner->width!=r.width || owner->height!=r.height)) owner.reset();
        }
        if (!c.resident_properties.captures) {
            c.upload(scene.studio,&c.studio,sizeof(c.studio));
            c.resident_properties.parameter_upload_bytes+=sizeof(c.studio);
        }
        for (uint32_t i=0;i<count;++i) {
            auto& v=views[i]; const auto& r=v.request;
            auto& owner=i ? scene.additional_targets[i-1] : scene.target;
            v.frame=&resident_target(c,scene,r.width,r.height,owner,!device_pixels);
            if (!device_pixels) require(v.frame->readback_mapping && v.bytes<=v.frame->readback_capacity,
                                        "camera batch readback capacity mismatch");
            if (v.msaa_color) msaa_prepare(c,*v.frame,r.width,r.height);
            upload_camera_params(c,*v.frame,r.params,false,0);
        }
        const auto record_begin=HostClock::now();
        std::array<VkCommandBuffer,max_cameras> commands{};
        for (uint32_t i=0;i<count;++i) {
            const auto& v=views[i]; const auto& r=v.request;
            const DeviceOutputSlice slice{v.offset,i==0 && c.device_output.returned,i+1==count};
            record_camera(c,*v.frame,r.params,r.channel_mask,r.camera_index,0,v.bytes,device_pixels ? &slice : nullptr);
            commands[i]=v.frame->commands;
        }
        require(c.errors.load()==errors,"Vulkan validation failed recording camera batch");
        const auto record_ns=elapsed_ns(record_begin);
        // All attachments, descriptors, UBOs, timestamps and readback buffers
        // are per view. Only the first frame's fence tracks the whole submission.
        const auto timing=submit_and_wait(c,*views[0].frame,commands.data(),count,device_pixels ? &c.device_output : nullptr);
        const auto readback_begin=HostClock::now();
        for (uint32_t i=0;i<count;++i)
            views[i].gpu=camera_gpu_timing(c,*views[i].frame,timing);
        require(c.errors.load()==errors,"Vulkan validation failed completing camera batch");
        // The fence and every result/query succeeded. Only now publish any byte.
        auto& profile=c.profile;
        for (uint32_t i=0;i<count;++i) {
            const auto& v=views[i]; const auto& r=v.request;
            if (!device_pixels) std::memcpy(output+v.offset,v.frame->readback_mapping,v.bytes);
            count_camera(c,r.params,v.msaa_color,v.center);
            if (r.camera_index!=UINT32_MAX) c.shared_camera_d2d_bytes+=19*sizeof(float);
            profile.gpu_render_ns+=v.gpu.render; profile.gpu_copy_ns+=v.gpu.copy;
            profile.gpu_total_ns+=v.gpu.total; profile.timestamp_samples+=v.gpu.samples;
        }
        profile.captures+=count;
        profile.host_prepare_ns+=elapsed_ns(start,upload_begin);
        profile.host_upload_ns+=elapsed_ns(upload_begin,record_begin);
        profile.host_record_ns+=record_ns; profile.host_submit_ns+=timing.submit;
        profile.host_wait_ns+=timing.wait;
        if (device_pixels) {
            auto& info=c.device_output.info;
            ++info.batches; info.views+=count; info.image_bytes+=image_bytes;
        } else {
            profile.host_readback_ns+=elapsed_ns(readback_begin);
            c.telemetry.readback_bytes+=bytes; c.readback_info.copied_bytes+=bytes;
            c.readback_info.last_channel_mask=views[count-1].request.channel_mask;
        }
    } catch (...) { c.poisoned=true; throw; }
}
