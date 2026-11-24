// The output and transform publication have separate memory and semaphores.
// An outstanding CUDA image can outlive its pose publication.
void destroy_device_output(Context& c) noexcept {
    auto& slot=c.device_output;
    drop_shared_handles(slot);
    release(slot.ready,[&](auto h){vkDestroySemaphore(c.device,h,nullptr);});
    release(slot.writable,[&](auto h){vkDestroySemaphore(c.device,h,nullptr);});
    release(slot.buffer.handle,[&](auto h){vkDestroyBuffer(c.device,h,nullptr);});
    c.free_memory(slot.buffer.memory,slot.buffer.allocation);
    slot.pending_fence={}; slot.returned=false;
    slot.info.phase=0; slot.info.capacity_bytes=0;
}
void export_device_output(Context& c,uint64_t capacity,PrResidentSharedExport& output) {
    require(c.shared_input && c.resident_state && !c.poisoned,"device output requires a loaded shared resident world");
    auto& slot=c.device_output;
    require(!slot.info.phase && capacity && capacity<=4096ull*4096ull*13ull+16*3,
            "device output slot is already configured or its capacity exceeds the batch limit");
#ifdef _WIN32
    try {
        create_shared_buffer(c,slot.buffer,capacity);
        create_external_exports(c,slot,slot.buffer);
        output=describe_external(slot,slot.buffer,capacity);
        slot.info.capacity_bytes=capacity;
        slot.info.phase=1; // first Vulkan transfer claims an unused exclusive buffer
    } catch (...) { destroy_device_output(c); throw; }
#else
    (void)output;
    throw std::runtime_error("device output requires Win32 external memory");
#endif
}
void return_device_output(Context& c) {
    auto& slot=c.device_output;
    require(slot.info.phase==2,"no CUDA output lease to return");
    if (slot.pending_fence) {
        // Only a previous failure/timeout leaves a pending fence. CUDA stream
        // completion should already imply completion of the producing submission.
        const auto rc=vkGetFenceStatus(c.device,slot.pending_fence);
        if (rc==VK_NOT_READY) throw SharedTimeout("device output fence pending; CUDA lease retained");
        if (rc==VK_ERROR_DEVICE_LOST) { c.poisoned=true; slot.pending_fence={}; slot.info.phase=1; }
        check(rc,"check outstanding device output fence");
        slot.pending_fence={};
    }
    slot.returned=true;
    slot.info.phase=1;
}
bool outstanding_device_output(Context& c) {
    if (c.device_output.info.phase!=2) return false;
    last_error="return the CUDA image lease before closing the resident context";
    return true;
}
void device_output_ownership(Context& c,VkCommandBuffer cmd,bool acquire_from_cuda) {
    VkBufferMemoryBarrier barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    barrier.buffer=c.device_output.buffer.handle; barrier.size=VK_WHOLE_SIZE;
    barrier.srcQueueFamilyIndex=acquire_from_cuda ? VK_QUEUE_FAMILY_EXTERNAL : c.queue_family;
    barrier.dstQueueFamilyIndex=acquire_from_cuda ? c.queue_family : VK_QUEUE_FAMILY_EXTERNAL;
    barrier.srcAccessMask=acquire_from_cuda ? 0 : VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask=acquire_from_cuda ? VK_ACCESS_TRANSFER_WRITE_BIT : 0;
    vkCmdPipelineBarrier(cmd,acquire_from_cuda ? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT : VK_PIPELINE_STAGE_TRANSFER_BIT,
        acquire_from_cuda ? VK_PIPELINE_STAGE_TRANSFER_BIT : VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
        0,0,nullptr,1,&barrier,0,nullptr);
}
