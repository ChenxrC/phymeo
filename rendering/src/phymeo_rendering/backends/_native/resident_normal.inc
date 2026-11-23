// Independent linear normal maps. Nothing here rewrites geometry or color data.
void resident_set_normal_maps(Context& c,const PrResidentTexture* textures,uint32_t texture_count,
    const uint8_t* rgba,size_t rgba_bytes,const PrResidentUV* uv,uint32_t uv_count,
    const uint32_t* slots,uint32_t instance_count) {
    require(c.resident && c.resident_state && !c.poisoned,"resident geometry must precede normal maps");
    auto& info=c.normal_info; auto& state=*c.resident_state;
    require(info.shader_compatible,"resident vertex/fragment lacks named normal-map ABI markers");
    require(c.properties.captures==0 && info.loaded==0,"normal maps are immutable after upload or capture");
    require(texture_count>0 && texture_count<=info.max_textures && textures,"normal texture count exceeds capacity");
    require(rgba && rgba_bytes>0 && rgba_bytes<=info.max_array_bytes,"normal source byte capacity");
    require(uv_count==c.resident_properties.vertex_count && (uv_count==0 || uv),"normal complete UV table required");
    require(instance_count==state.instances.size() && (instance_count==0 || slots),"normal instance table mismatch");
    uint64_t source_bytes=0; uint32_t width=0,height=0,bound=0;
    for(uint32_t i=0;i<texture_count;++i) {
        const auto& record=textures[i];
        require(record.width>0 && record.height>0 && record.width<=info.max_width && record.height<=info.max_height,
                "normal texture dimension exceeds capacity; no resizing");
        require(record.reserved==0 && record.first_byte==source_bytes,"normal flags/ranges mismatch");
        source_bytes+=4ull*record.width*record.height;
        require(source_bytes<=rgba_bytes,"normal source range out of bounds");
        width=std::max(width,record.width); height=std::max(height,record.height);
    }
    require(source_bytes==rgba_bytes,"normal unused source bytes");
    for(size_t i=3;i<rgba_bytes;i+=4) require(rgba[i]==255,"normal maps must be opaque RGB data");
    for(uint32_t i=0;i<uv_count;++i)
        require(std::isfinite(uv[i].u) && std::isfinite(uv[i].v) && std::abs(uv[i].u)<=65536.f && std::abs(uv[i].v)<=65536.f,
                "normal UV must be finite FP32 within +/-65536");
    std::vector<bool> checked(state.meshes.size(),false);
    for(uint32_t i=0;i<instance_count;++i) {
        require(slots[i]==UINT32_MAX || slots[i]<texture_count,"normal slot out of bounds");
        if(slots[i]==UINT32_MAX) continue;
        ++bound; const auto mesh_index=state.instances[i].mesh_index;
        if(checked[mesh_index]) continue;
        checked[mesh_index]=true; const auto& mesh=state.meshes[mesh_index];
        for(uint32_t index=mesh.first_index;index<mesh.first_index+mesh.index_count;index+=3) {
            const auto& a=uv[mesh.first_vertex+state.normal_validation_indices[index]];
            const auto& b=uv[mesh.first_vertex+state.normal_validation_indices[index+1]];
            const auto& d=uv[mesh.first_vertex+state.normal_validation_indices[index+2]];
            const double first=(double(b.u)-a.u)*(double(d.v)-a.v), second=(double(b.v)-a.v)*(double(d.u)-a.u);
            require(std::abs(first-second)>3.814697265625e-6*(std::abs(first)+std::abs(second)),
                    "normal UV triangle singular or ill-conditioned after FP32 conversion");
        }
    }
    require(bound>0,"normal upload has no bound instance");
    width+=2; height+=2; const uint64_t padded_bytes=4ull*width*height*texture_count;
    require(padded_bytes<=info.max_array_bytes,"normal padded array capacity exceeded");
    const uint64_t uv_offset=4ull+instance_count+2ull*texture_count,word_count=uv_offset+2ull*uv_count;
    require(word_count*4<=c.storage_buffer_limit,"normal metadata storage capacity exceeded");
    std::vector<uint32_t> words(static_cast<size_t>(word_count));
    words[0]=texture_count; words[1]=instance_count; words[2]=uv_count; words[3]=static_cast<uint32_t>(uv_offset);
    for(uint32_t i=0;i<instance_count;++i) words[4+i]=slots[i];
    for(uint32_t i=0;i<texture_count;++i) {
        words[4+instance_count+2*i]=textures[i].width; words[5+instance_count+2*i]=textures[i].height;
    }
    for(uint32_t i=0;i<uv_count;++i) {
        std::memcpy(&words[uv_offset+2*i],&uv[i].u,4); std::memcpy(&words[uv_offset+2*i+1],&uv[i].v,4);
    }
    std::vector<uint8_t> pixels(static_cast<size_t>(padded_bytes));
    for(uint32_t layer=0;layer<texture_count;++layer) {
        const auto& record=textures[layer];
        for(uint32_t y=0;y<record.height+2;++y) for(uint32_t x=0;x<record.width+2;++x) {
            uint32_t sx=(x+record.width-1)%record.width,sy=(y+record.height-1)%record.height;
            const size_t destination=((static_cast<size_t>(layer)*height+y)*width+x)*4;
            const size_t source=record.first_byte+(static_cast<size_t>(sy)*record.width+sx)*4;
            std::memcpy(pixels.data()+destination,rgba+source,4);
        }
    }
    const auto errors=c.errors.load();
    try {
        auto uploaded=upload_texture_resource(c,width,height,texture_count,pixels,words,VK_FORMAT_R8G8B8A8_UNORM);
        require(c.errors.load()==errors,"Vulkan validation failed during normal-map upload");
        state.normal_maps=std::move(uploaded);
    } catch(...) { c.poisoned=true; throw; }
    info.texture_count=texture_count; info.array_width=width; info.array_height=height; info.array_layers=texture_count;
    info.uv_count=uv_count; info.bound_instances=bound; info.loaded=1;
    info.source_rgba_bytes=source_bytes; info.padded_rgba_bytes=padded_bytes;
    info.metadata_upload_bytes+=words.size()*sizeof(uint32_t); info.image_upload_bytes+=padded_bytes; ++info.upload_calls;
}
