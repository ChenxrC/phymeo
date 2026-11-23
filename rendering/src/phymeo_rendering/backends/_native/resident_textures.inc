// Texture-only resources and cold-start packing, included in the private
// native namespace. Existing geometry_upload_bytes keeps its five-buffer scope.
std::unique_ptr<ResidentTextures> upload_texture_resource(Context& c,uint32_t width,uint32_t height,
    uint32_t layers,const std::vector<uint8_t>& rgba,const std::vector<uint32_t>& words,
    VkFormat format=VK_FORMAT_R8G8B8A8_SRGB) {
    auto texture=std::make_unique<ResidentTextures>(c);
    VkImageCreateInfo create{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    create.imageType=VK_IMAGE_TYPE_2D; create.format=format;
    create.extent={width,height,1}; create.mipLevels=1; create.arrayLayers=layers;
    create.samples=VK_SAMPLE_COUNT_1_BIT; create.tiling=VK_IMAGE_TILING_OPTIMAL;
    create.usage=VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    create.sharingMode=VK_SHARING_MODE_EXCLUSIVE; create.initialLayout=VK_IMAGE_LAYOUT_UNDEFINED;
    acquire(texture->image.handle,"resident texture image",[&](auto* out){return vkCreateImage(c.device,&create,nullptr,out);});
    VkMemoryRequirements needed{}; vkGetImageMemoryRequirements(c.device,texture->image.handle,&needed);
    VkMemoryAllocateInfo allocate{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocate.allocationSize=needed.size;
    allocate.memoryTypeIndex=c.memory_type(needed.memoryTypeBits,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    acquire(texture->image.memory,"resident texture memory",[&](auto* out){return vkAllocateMemory(c.device,&allocate,nullptr,out);});
    c.account_allocation(texture->image.allocation,allocate);
    check(vkBindImageMemory(c.device,texture->image.handle,texture->image.memory,0),"resident bind texture memory");
    VkImageViewCreateInfo view{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    view.image=texture->image.handle; view.viewType=VK_IMAGE_VIEW_TYPE_2D_ARRAY;
    view.format=format;
    view.subresourceRange={VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,layers};
    acquire(texture->image.view,"resident texture view",[&](auto* out){return vkCreateImageView(c.device,&view,nullptr,out);});
    VkSamplerCreateInfo sampler{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    sampler.magFilter=sampler.minFilter=VK_FILTER_LINEAR;
    sampler.mipmapMode=VK_SAMPLER_MIPMAP_MODE_NEAREST;
    sampler.addressModeU=sampler.addressModeV=sampler.addressModeW=VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler.minLod=sampler.maxLod=0;
    acquire(texture->sampler,"resident texture sampler",[&](auto* out){return vkCreateSampler(c.device,&sampler,nullptr,out);});
    texture->metadata_bytes=words.size()*sizeof(uint32_t);
    c.make_buffer(texture->metadata,texture->metadata_bytes,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    // One bounded staging allocation and synchronous transfer owns both image
    // and metadata. The caller's arrays are no longer needed on return.
    FrameResources staging(c);
    const size_t total=rgba.size()+texture->metadata_bytes;
    c.make_buffer(staging.vertices,total,VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    void* mapped=nullptr;
    check(vkMapMemory(c.device,staging.vertices.memory,0,total,0,&mapped),"resident texture staging map");
    std::memcpy(mapped,rgba.data(),rgba.size());
    std::memcpy(static_cast<uint8_t*>(mapped)+rgba.size(),words.data(),texture->metadata_bytes);
    vkUnmapMemory(c.device,staging.vertices.memory);
    make_commands(c,staging);
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    check(vkBeginCommandBuffer(staging.commands,&begin),"resident texture upload begin");
    VkImageMemoryBarrier image{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    image.srcQueueFamilyIndex=image.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;
    image.image=texture->image.handle; image.subresourceRange=view.subresourceRange;
    image.oldLayout=VK_IMAGE_LAYOUT_UNDEFINED; image.newLayout=VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    image.dstAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(staging.commands,VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0,0,nullptr,0,nullptr,1,&image);
    VkBufferImageCopy copy{};
    copy.imageSubresource={VK_IMAGE_ASPECT_COLOR_BIT,0,0,layers}; copy.imageExtent={width,height,1};
    vkCmdCopyBufferToImage(staging.commands,staging.vertices.handle,texture->image.handle,
                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,1,&copy);
    VkBufferCopy metadata_copy{rgba.size(),0,texture->metadata_bytes};
    vkCmdCopyBuffer(staging.commands,staging.vertices.handle,texture->metadata.handle,1,&metadata_copy);
    image.oldLayout=VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL; image.newLayout=VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    image.srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT; image.dstAccessMask=VK_ACCESS_SHADER_READ_BIT;
    VkBufferMemoryBarrier metadata{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    metadata.srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT; metadata.dstAccessMask=VK_ACCESS_SHADER_READ_BIT;
    metadata.srcQueueFamilyIndex=metadata.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;
    metadata.buffer=texture->metadata.handle; metadata.size=VK_WHOLE_SIZE;
    vkCmdPipelineBarrier(staging.commands,VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        0,0,nullptr,1,&metadata,1,&image);
    check(vkEndCommandBuffer(staging.commands),"resident texture upload end");
    submit_and_wait(c,staging);
    return texture;
}

void resident_set_textures(Context& c,const PrResidentTexture* textures,uint32_t texture_count,
    const uint8_t* rgba,size_t rgba_bytes,const PrResidentUV* uv,uint32_t uv_count,
    const uint32_t* instance_slots,uint32_t instance_count,bool surface_maps=false) {
    require(c.resident && c.resident_state && !c.poisoned,"resident geometry must precede texture upload");
    auto& info=c.texture_info; auto& state=*c.resident_state;
    require(!surface_maps || c.surface_maps_compatible,
        "resident fragment SPIR-V lacks named surface-map ABI marker");
    require(c.properties.captures==0 && info.loaded==0,"resident textures are immutable after upload or capture");
    require(texture_count>0 && texture_count<=info.max_textures && textures,"resident texture count exceeds capacity");
    require(rgba && rgba_bytes>0 && rgba_bytes<=info.max_array_bytes,"resident source texture bytes exceed capacity");
    require(uv_count==c.resident_properties.vertex_count && (uv_count==0 || uv),"resident complete vertex UV table required");
    require(instance_count==state.instances.size() && (instance_count==0 || instance_slots),"resident texture instance table mismatch");
    uint64_t source_bytes=0;
    uint32_t width=0,height=0;
    for (uint32_t i=0; i<texture_count; ++i) {
        const auto& record=textures[i];
        require(record.width>0 && record.height>0 && record.width<=info.max_width && record.height<=info.max_height,
                "resident texture dimensions exceed capacity; no resizing is performed");
        if (surface_maps) require(record.reserved<=1 && record.first_byte==source_bytes,
            "resident surface-map flags unsupported or source ranges not tightly packed");
        else require(record.reserved==0 && record.first_byte==source_bytes,
            "resident texture source ranges must be tightly packed");
        source_bytes+=4ull*record.width*record.height;
        require(source_bytes<=rgba_bytes,"resident texture source range out of bounds");
        width=std::max(width,record.width); height=std::max(height,record.height);
    }
    require(source_bytes==rgba_bytes,"resident texture source has unused bytes");
    width+=2; height+=2;
    const uint64_t padded_bytes=4ull*width*height*texture_count;
    require(padded_bytes<=info.max_array_bytes,"resident padded texture array exceeds capacity");
    for (uint32_t texture=0;texture<texture_count;++texture) {
        const auto& record=textures[texture];
        if (record.reserved==0) {
            const size_t end=record.first_byte+static_cast<size_t>(record.width)*record.height*4;
            for (size_t i=record.first_byte+3;i<end;i+=4)
                require(rgba[i]==255,"resident base-color textures must be opaque");
        }
    }
    for (uint32_t i=0; i<uv_count; ++i)
        require(std::isfinite(uv[i].u) && std::isfinite(uv[i].v),"resident texture UV must be finite FP32");
    uint32_t bound=0;
    for (uint32_t i=0; i<instance_count; ++i) {
        require(instance_slots[i]==UINT32_MAX || instance_slots[i]<texture_count,"resident texture slot out of bounds");
        if (instance_slots[i]!=UINT32_MAX) ++bound;
    }
    require(bound>0,"resident texture upload has no bound textured instance");
    const uint64_t uv_offset=4ull+instance_count+2ull*texture_count;
    const uint64_t word_count=uv_offset+2ull*uv_count;
    require(word_count*4<=c.storage_buffer_limit,"resident texture metadata exceeds storage-buffer capacity");
    std::vector<uint32_t> words(static_cast<size_t>(word_count));
    words[0]=texture_count; words[1]=instance_count; words[2]=uv_count; words[3]=static_cast<uint32_t>(uv_offset);
    for (uint32_t i=0; i<instance_count; ++i) words[4+i]=instance_slots[i];
    for (uint32_t i=0; i<texture_count; ++i) {
        words[4+instance_count+2*i]=textures[i].width;
        words[4+instance_count+2*i+1]=textures[i].height;
    }
    for (uint32_t i=0; i<uv_count; ++i) {
        std::memcpy(&words[uv_offset+2*i],&uv[i].u,4);
        std::memcpy(&words[uv_offset+2*i+1],&uv[i].v,4);
    }
    std::vector<uint8_t> pixels(static_cast<size_t>(padded_bytes),0);
    for (uint32_t layer=0; layer<texture_count; ++layer) {
        const auto& record=textures[layer];
        // One periodic guard texel on EACH side, including wrapped corners.
        // Other padding is unreachable by the shader's source-size mapping.
        for (uint32_t y=0; y<record.height+2; ++y) for (uint32_t x=0; x<record.width+2; ++x) {
            uint32_t sx=(x+record.width-1)%record.width, sy=(y+record.height-1)%record.height;
            size_t destination=((static_cast<size_t>(layer)*height+y)*width+x)*4;
            size_t source=record.first_byte+(static_cast<size_t>(sy)*record.width+sx)*4;
            std::memcpy(pixels.data()+destination,rgba+source,4);
        }
    }
    uint64_t errors_before=c.errors.load();
    try {
        auto uploaded=upload_texture_resource(c,width,height,texture_count,pixels,words);
        require(c.errors.load()==errors_before,"Vulkan validation failed during texture upload");
        state.textures=std::move(uploaded); // no capture/descriptor can reference the default image yet
    } catch (...) { c.poisoned=true; throw; }
    info.texture_count=texture_count; info.array_width=width; info.array_height=height;
    info.array_layers=texture_count; info.uv_count=uv_count; info.bound_instances=bound;
    info.loaded=1; info.source_rgba_bytes=source_bytes; info.padded_rgba_bytes=padded_bytes;
    info.metadata_upload_bytes+=words.size()*sizeof(uint32_t); ++info.upload_calls;
    info.image_upload_bytes+=padded_bytes;
}
