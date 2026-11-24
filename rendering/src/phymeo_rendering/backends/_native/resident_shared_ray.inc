// Convert only draw instances. The published transform allocation can also
// contain camera records, which are outside both compute descriptor ranges.
void prepare_shared_ray_pipeline(Context& c,ResidentState& state) {
    auto& a=*state.acceleration;
    const VkDeviceSize count=state.instances.size();
    require(count && count<=c.ray_info.max_instances && count<=(1u<<24),"invalid shared TLAS instance count");
    require(count*sizeof(VkAccelerationStructureInstanceKHR)<=c.storage_buffer_limit,
            "shared TLAS records exceed compute storage descriptor range");
    std::array<VkDescriptorSetLayoutBinding,2> bindings{};
    for (uint32_t i=0;i<bindings.size();++i) {
        bindings[i].binding=i; bindings[i].descriptorCount=1;
        bindings[i].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].stageFlags=VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo layout{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    layout.bindingCount=static_cast<uint32_t>(bindings.size()); layout.pBindings=bindings.data();
    acquire(a.pack_bindings,"shared TLAS descriptor layout",[&](auto* out){return vkCreateDescriptorSetLayout(c.device,&layout,nullptr,out);});
    VkPipelineLayoutCreateInfo pipeline_layout{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pipeline_layout.setLayoutCount=1; pipeline_layout.pSetLayouts=&a.pack_bindings;
    acquire(a.pack_layout,"shared TLAS compute layout",[&](auto* out){return vkCreatePipelineLayout(c.device,&pipeline_layout,nullptr,out);});
    VkComputePipelineCreateInfo compute{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    compute.stage.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    compute.stage.stage=VK_SHADER_STAGE_COMPUTE_BIT; compute.stage.module=c.shared_instance_shader; compute.stage.pName="main";
    compute.layout=a.pack_layout;
    acquire(a.pack_pipeline,"shared TLAS compute pipeline",[&](auto* out){
        return vkCreateComputePipelines(c.device,VK_NULL_HANDLE,1,&compute,nullptr,out);});
    VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,2};
    VkDescriptorPoolCreateInfo pool{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pool.maxSets=1; pool.poolSizeCount=1; pool.pPoolSizes=&size;
    acquire(a.pack_pool,"shared TLAS descriptor pool",[&](auto* out){return vkCreateDescriptorPool(c.device,&pool,nullptr,out);});
    VkDescriptorSetAllocateInfo alloc{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    alloc.descriptorPool=a.pack_pool; alloc.descriptorSetCount=1; alloc.pSetLayouts=&a.pack_bindings;
    check(vkAllocateDescriptorSets(c.device,&alloc,&a.pack_set),"shared TLAS descriptor set");
    const VkDescriptorBufferInfo buffers[]={{state.transforms.handle,0,count*sizeof(GpuTransform)},
        {a.instances.handle,0,count*sizeof(VkAccelerationStructureInstanceKHR)}};
    std::array<VkWriteDescriptorSet,2> writes{};
    for (uint32_t i=0;i<writes.size();++i) {
        auto& w=writes[i]; w.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; w.dstSet=a.pack_set;
        w.dstBinding=i; w.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w.descriptorCount=1; w.pBufferInfo=&buffers[i];
    }
    vkUpdateDescriptorSets(c.device,static_cast<uint32_t>(writes.size()),writes.data(),0,nullptr);
}

void record_shared_ray_update(Context& c,ResidentState& state,VkCommandBuffer cmd) {
    auto& a=*state.acceleration;
    // The previous acceleration build read these records. The new dispatch can
    // overwrite 48 bytes per entry only after that read has finished.
    VkBufferMemoryBarrier records{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    records.srcAccessMask=VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR | VK_ACCESS_TRANSFER_WRITE_BIT;
    records.dstAccessMask=VK_ACCESS_SHADER_WRITE_BIT;
    records.srcQueueFamilyIndex=records.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;
    records.buffer=a.instances.handle; records.size=VK_WHOLE_SIZE;
    vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR|VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,0,nullptr,1,&records,0,nullptr);
    vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,a.pack_pipeline);
    vkCmdBindDescriptorSets(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,a.pack_layout,0,1,&a.pack_set,0,nullptr);
    vkCmdDispatch(cmd,(static_cast<uint32_t>(state.instances.size())+63u)/64u,1,1);
    record_ray_update(c,state,cmd);
}
