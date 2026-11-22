// Explicit hardware shadow acceleration. Shared raster/material data are not
// Synchronization follows https://docs.vulkan.org/spec/latest/chapters/accelstructures.html
// (geometry inputs use SHADER_READ; AS/scratch use ACCELERATION_STRUCTURE_READ/WRITE).
// Geometry is not simplified or filtered. Every uploaded mesh receives a BLAS, every instance
// receives a TLAS entry. The same current rounded rigid matrices drive both.
VkDeviceAddress ray_buffer_address(Context& c,const Buffer& buffer) {
    VkBufferDeviceAddressInfo info{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO}; info.buffer=buffer.handle;
    VkDeviceAddress address=c.ray.buffer_address(c.device,&info);
    require(address!=0,"ray-query buffer device address unavailable");
    return address;
}
std::unique_ptr<AccelerationObject> make_acceleration(Context& c,VkDeviceSize bytes,VkAccelerationStructureTypeKHR type) {
    require(bytes>0,"ray-query acceleration storage size is zero");
    auto result=std::make_unique<AccelerationObject>(c);
    c.make_buffer(result->storage,bytes,VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VkAccelerationStructureCreateInfoKHR create{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
    create.buffer=result->storage.handle; create.size=bytes; create.type=type;
    acquire(result->handle,"ray-query acceleration structure",[&](auto* out){return c.ray.create(c.device,&create,nullptr,out);});
    VkAccelerationStructureDeviceAddressInfoKHR address{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR};
    address.accelerationStructure=result->handle; result->address=c.ray.address(c.device,&address);
    require(result->address!=0,"ray-query acceleration device address unavailable");
    return result;
}
VkAccelerationStructureGeometryKHR ray_triangles(Context& c,const ResidentState& state,uint32_t mesh_index) {
    const auto& mesh=state.meshes[mesh_index];
    VkAccelerationStructureGeometryKHR geometry{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
    geometry.geometryType=VK_GEOMETRY_TYPE_TRIANGLES_KHR; geometry.flags=VK_GEOMETRY_OPAQUE_BIT_KHR;
    auto& triangles=geometry.geometry.triangles;
    triangles.sType=VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
    triangles.vertexFormat=VK_FORMAT_R32G32B32_SFLOAT;
    triangles.vertexData.deviceAddress=ray_buffer_address(c,state.vertices)+static_cast<uint64_t>(mesh.first_vertex)*sizeof(PrResidentVertex);
    triangles.vertexStride=sizeof(PrResidentVertex); triangles.maxVertex=mesh.vertex_count-1;
    triangles.indexType=VK_INDEX_TYPE_UINT32;
    triangles.indexData.deviceAddress=ray_buffer_address(c,state.indices)+static_cast<uint64_t>(mesh.first_index)*sizeof(uint32_t);
    return geometry;
}
VkAccelerationStructureGeometryKHR ray_instances_geometry(VkDeviceAddress address) {
    VkAccelerationStructureGeometryKHR geometry{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
    geometry.geometryType=VK_GEOMETRY_TYPE_INSTANCES_KHR;
    geometry.geometry.instances.sType=VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    geometry.geometry.instances.arrayOfPointers=VK_FALSE;
    geometry.geometry.instances.data.deviceAddress=address;
    return geometry;
}
VkAccelerationStructureBuildGeometryInfoKHR ray_build_info(VkAccelerationStructureTypeKHR type,
    VkBuildAccelerationStructureFlagsKHR flags,const VkAccelerationStructureGeometryKHR* geometry) {
    VkAccelerationStructureBuildGeometryInfoKHR info{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
    info.type=type; info.flags=flags; info.mode=VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    info.geometryCount=1; info.pGeometries=geometry;
    return info;
}

std::unique_ptr<ResidentAcceleration> build_ray_geometry(Context& c,ResidentState& state,uint32_t primitive_count) {
    auto result=std::make_unique<ResidentAcceleration>(c);
    const uint32_t mesh_count=static_cast<uint32_t>(state.meshes.size());
    const uint32_t instance_count=static_cast<uint32_t>(state.instances.size());
    auto& telemetry=c.ray_info;
    std::vector<VkAccelerationStructureBuildSizesInfoKHR> sizes(mesh_count);
    VkDeviceSize storage_bytes=0,scratch_bytes=0;
    for (uint32_t mesh=0;mesh<mesh_count;++mesh) {
        auto geometry=ray_triangles(c,state,mesh);
        auto info=ray_build_info(VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,telemetry.build_flags,&geometry);
        uint32_t triangles=state.meshes[mesh].index_count/3;
        sizes[mesh].sType=VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
        c.ray.sizes(c.device,VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,&info,&triangles,&sizes[mesh]);
        require(sizes[mesh].accelerationStructureSize<=telemetry.max_acceleration_bytes,"BLAS capacity exceeded");
        storage_bytes+=sizes[mesh].accelerationStructureSize;
        scratch_bytes=std::max(scratch_bytes,sizes[mesh].buildScratchSize);
        require(storage_bytes<=telemetry.max_acceleration_bytes,"combined BLAS capacity exceeded");
    }
    auto top_geometry=ray_instances_geometry(0);
    auto top_info=ray_build_info(VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,telemetry.update_flags,&top_geometry);
    VkAccelerationStructureBuildSizesInfoKHR top_size{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
    c.ray.sizes(c.device,VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,&top_info,&instance_count,&top_size);
    require(top_size.accelerationStructureSize<=telemetry.max_acceleration_bytes,"TLAS capacity exceeded");
    storage_bytes+=top_size.accelerationStructureSize;
    scratch_bytes=std::max({scratch_bytes,top_size.buildScratchSize,top_size.updateScratchSize,VkDeviceSize(1)});
    require(scratch_bytes<=telemetry.max_acceleration_bytes && c.ray_scratch_alignment<=telemetry.max_acceleration_bytes,
        "ray-query scratch capacity exceeded");
    const VkDeviceSize padded_scratch=scratch_bytes+c.ray_scratch_alignment-1;
    const VkDeviceSize instance_bytes=std::max<uint64_t>(1,instance_count)*sizeof(VkAccelerationStructureInstanceKHR);
    require(storage_bytes<=telemetry.max_acceleration_bytes && padded_scratch<=telemetry.max_acceleration_bytes &&
        storage_bytes+padded_scratch+instance_bytes<=telemetry.max_acceleration_bytes,
        "ray-query AS/scratch/instance budget exceeded; no geometry reduction");
    for (uint32_t mesh=0;mesh<mesh_count;++mesh)
        result->bottom.push_back(make_acceleration(c,sizes[mesh].accelerationStructureSize,VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR));
    result->top=make_acceleration(c,top_size.accelerationStructureSize,VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR);
    c.make_buffer(result->scratch,padded_scratch,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VkDeviceAddress base=ray_buffer_address(c,result->scratch);
    result->scratch_address=(base+c.ray_scratch_alignment-1)&~static_cast<VkDeviceAddress>(c.ray_scratch_alignment-1);
    c.make_buffer(result->instances,instance_bytes,VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);
    result->instances_address=ray_buffer_address(c,result->instances);
    if (mesh_count) {
        FrameResources frame(c); make_commands(c,frame);
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; begin.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        check(vkBeginCommandBuffer(frame.commands,&begin),"ray-query BLAS begin");
        VkMemoryBarrier input{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        input.srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT; input.dstAccessMask=VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(frame.commands,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
            0,1,&input,0,nullptr,0,nullptr);
        for (uint32_t mesh=0;mesh<mesh_count;++mesh) {
            auto geometry=ray_triangles(c,state,mesh);
            auto info=ray_build_info(VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,telemetry.build_flags,&geometry);
            info.dstAccelerationStructure=result->bottom[mesh]->handle; info.scratchData.deviceAddress=result->scratch_address;
            VkAccelerationStructureBuildRangeInfoKHR range{}; range.primitiveCount=state.meshes[mesh].index_count/3;
            const VkAccelerationStructureBuildRangeInfoKHR* ranges=&range;
            c.ray.build(frame.commands,1,&info,&ranges);
            // Scratch is reused only after the previous build's reads/writes.
            VkMemoryBarrier scratch{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            scratch.srcAccessMask=VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
            scratch.dstAccessMask=VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR | VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
            vkCmdPipelineBarrier(frame.commands,VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,0,1,&scratch,0,nullptr,0,nullptr);
        }
        check(vkEndCommandBuffer(frame.commands),"ray-query BLAS end"); submit_and_wait(c,frame);
    }
    telemetry.blas_count=mesh_count; telemetry.tlas_instance_count=instance_count; telemetry.primitive_count=primitive_count;
    telemetry.acceleration_buffer_bytes=storage_bytes; telemetry.scratch_buffer_bytes=padded_scratch;
    telemetry.blas_builds=mesh_count;
    return result;
}

void upload_ray_instances(Context& c,ResidentState& state,const std::vector<GpuTransform>& transforms) {
    static_assert(sizeof(VkAccelerationStructureInstanceKHR)==64,"Vulkan instance record size");
    std::vector<VkAccelerationStructureInstanceKHR> instances(state.instances.size());
    for (size_t i=0;i<instances.size();++i) {
        auto& instance=instances[i];
        for (uint32_t row=0;row<3;++row) for (uint32_t column=0;column<4;++column)
            instance.transform.matrix[row][column]=transforms[i].world[4*column+row];
        instance.instanceCustomIndex=static_cast<uint32_t>(i); instance.mask=0xff;
        instance.instanceShaderBindingTableRecordOffset=0;
        instance.flags=VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
        instance.accelerationStructureReference=state.acceleration->bottom[state.instances[i].mesh_index]->address;
    }
    if (!instances.empty()) c.upload(state.acceleration->instances,instances.data(),instances.size()*sizeof(instances[0]));
    c.ray_info.instance_upload_bytes+=instances.size()*sizeof(VkAccelerationStructureInstanceKHR);
}
void record_ray_update(Context& c,ResidentState& state,VkCommandBuffer commands) {
    auto& acceleration=*state.acceleration;
    // Includes previous frame traversal and scratch use, plus this host write.
    VkMemoryBarrier before{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    before.srcAccessMask=VK_ACCESS_HOST_WRITE_BIT | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR | VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
    before.dstAccessMask=VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR | VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
    vkCmdPipelineBarrier(commands,VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
        VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
        0,1,&before,0,nullptr,0,nullptr);
    auto geometry=ray_instances_geometry(acceleration.instances_address);
    auto info=ray_build_info(VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,c.ray_info.update_flags,&geometry);
    info.dstAccelerationStructure=acceleration.top->handle; info.scratchData.deviceAddress=acceleration.scratch_address;
    if (acceleration.built) {
        info.mode=VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR;
        info.srcAccelerationStructure=acceleration.top->handle;
    }
    VkAccelerationStructureBuildRangeInfoKHR range{}; range.primitiveCount=static_cast<uint32_t>(state.instances.size());
    const VkAccelerationStructureBuildRangeInfoKHR* ranges=&range;
    c.ray.build(commands,1,&info,&ranges);
    VkMemoryBarrier after{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    after.srcAccessMask=VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
    after.dstAccessMask=VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
    vkCmdPipelineBarrier(commands,VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        0,1,&after,0,nullptr,0,nullptr);
}
