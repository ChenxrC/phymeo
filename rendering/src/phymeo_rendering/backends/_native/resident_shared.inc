// Shared slot belongs to the resident device. The original graphics pipeline
// and descriptor binding read this allocation directly on every camera draw.
struct SharedTimeout : std::runtime_error {
    SharedTimeout() : std::runtime_error("shared transform ownership fence timed out; context retained") {}
    explicit SharedTimeout(const char* message) : std::runtime_error(message) {}
};
void drop_shared_handles(ExternalSync& s) noexcept {
#ifdef _WIN32
    for (auto& h:s.handles) release(h,[](void* v){ CloseHandle(v); });
#else
    (void)s;
#endif
}
void destroy_shared(Context& c) noexcept {
    auto& s=c.shared_slot;
    drop_shared_handles(s);
    release(s.commands,[&](auto h){ vkFreeCommandBuffers(c.device,c.command_pool,1,&h); });
    release(s.fence,[&](auto h){ vkDestroyFence(c.device,h,nullptr); });
    release(s.ready,[&](auto h){ vkDestroySemaphore(c.device,h,nullptr); });
    release(s.writable,[&](auto h){ vkDestroySemaphore(c.device,h,nullptr); });
    s={};
}
void create_shared_buffer(Context& c,Buffer& buffer,VkDeviceSize bytes) {
#ifdef _WIN32
    VkExternalMemoryBufferCreateInfo ext{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};
    ext.handleTypes=VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.pNext=&ext; bi.size=bytes;
    bi.usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT|VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    bi.sharingMode=VK_SHARING_MODE_EXCLUSIVE;
    acquire(buffer.handle,"create resident shared transform buffer",[&](auto out){return vkCreateBuffer(c.device,&bi,nullptr,out);});
    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(c.device,buffer.handle,&req);
    VkMemoryDedicatedAllocateInfo dedicated{VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO};
    dedicated.buffer=buffer.handle;
    VkExportMemoryWin32HandleInfoKHR win{VK_STRUCTURE_TYPE_EXPORT_MEMORY_WIN32_HANDLE_INFO_KHR};
    win.dwAccess=GENERIC_ALL; win.pNext=&dedicated;
    VkExportMemoryAllocateInfo exp{VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO};
    exp.handleTypes=ext.handleTypes; exp.pNext=&win;
    VkMemoryAllocateInfo alloc{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    alloc.pNext=&exp; alloc.allocationSize=req.size;
    alloc.memoryTypeIndex=c.memory_type(req.memoryTypeBits,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    acquire(buffer.memory,"allocate resident shared transforms",[&](auto out){return vkAllocateMemory(c.device,&alloc,nullptr,out);});
    c.account_allocation(buffer.allocation,alloc);
    check(vkBindBufferMemory(c.device,buffer.handle,buffer.memory,0),"bind shared transform allocation");
#else
    (void)c; (void)buffer; (void)bytes;
    throw std::runtime_error("shared transform buffer requires Win32");
#endif
}

using SharedPhase=SharedSlot::Phase;
void check_shared(Context& c,VkResult result,const char* operation) {
    if (result==VK_ERROR_DEVICE_LOST) { c.poisoned=true; c.shared_slot.phase=SharedPhase::unused; }
    check(result,operation);
}
void enqueue_shared_ownership(Context& c,bool take,bool initial=false) {
    auto& s=c.shared_slot;
    check_shared(c,vkResetFences(c.device,1,&s.fence),"reset shared ownership fence");
    check_shared(c,vkResetCommandBuffer(s.commands,0),"reset shared ownership command");
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    check_shared(c,vkBeginCommandBuffer(s.commands,&begin),"begin shared ownership command");
    VkBufferMemoryBarrier barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    barrier.buffer=c.resident_state->transforms.handle; barrier.size=VK_WHOLE_SIZE;
    barrier.srcQueueFamilyIndex=take ? VK_QUEUE_FAMILY_EXTERNAL : c.queue_family;
    barrier.dstQueueFamilyIndex=take ? c.queue_family : VK_QUEUE_FAMILY_EXTERNAL;
    // The same publication may supply a camera UBO copy, graphics transforms,
    // and (hardware mode) the compute pass that assembles TLAS instance records.
    const VkAccessFlags reads=VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_TRANSFER_READ_BIT;
    barrier.srcAccessMask=take ? 0 : (initial ? VK_ACCESS_TRANSFER_WRITE_BIT : reads);
    barrier.dstAccessMask=take ? reads : 0;
    VkPipelineStageFlags consumers=VK_PIPELINE_STAGE_VERTEX_SHADER_BIT|VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT|VK_PIPELINE_STAGE_TRANSFER_BIT;
    if (c.ray_query) consumers|=VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    if (initial) vkCmdFillBuffer(s.commands,barrier.buffer,0,VK_WHOLE_SIZE,0);
    vkCmdPipelineBarrier(s.commands,take ? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT : (initial ? VK_PIPELINE_STAGE_TRANSFER_BIT : consumers),
        take ? consumers : VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,0,0,nullptr,1,&barrier,0,nullptr);
    if (take && c.ray_query) record_shared_ray_update(c,*c.resident_state,s.commands);
    check_shared(c,vkEndCommandBuffer(s.commands),"end shared ownership command");
    VkPipelineStageFlags wait_stage=VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount=1; submit.pCommandBuffers=&s.commands;
    if (take) {
        submit.waitSemaphoreCount=1; submit.pWaitSemaphores=&s.ready; submit.pWaitDstStageMask=&wait_stage;
    } else { submit.signalSemaphoreCount=1; submit.pSignalSemaphores=&s.writable; }
    check_shared(c,vkQueueSubmit(c.queue,1,&submit,s.fence),"submit shared ownership command");
    ++c.telemetry.submissions;
    s.phase=take ? SharedPhase::acquiring : SharedPhase::releasing;
}
void finish_shared_ownership(Context& c) {
    auto& s=c.shared_slot;
    if (s.phase!=SharedPhase::acquiring && s.phase!=SharedPhase::releasing) return;
    const auto start=std::chrono::steady_clock::now();
    const auto rc=vkWaitForFences(c.device,1,&s.fence,VK_TRUE,1000000000ull);
    ++c.telemetry.fence_wait_calls;
    c.telemetry.fence_wait_wall_ns+=static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now()-start).count());
    if (rc==VK_TIMEOUT) throw SharedTimeout();
    check_shared(c,rc,"wait for shared ownership fence");
    if (s.phase==SharedPhase::acquiring && c.ray_query) {
        auto& acceleration=*c.resident_state->acceleration;
        if (acceleration.built) ++c.ray_info.tlas_updates;
        else ++c.ray_info.tlas_builds;
        acceleration.built=true;
    }
    s.phase=s.phase==SharedPhase::acquiring ? SharedPhase::reading : SharedPhase::external;
}
#ifdef _WIN32
void create_external_exports(Context& c,ExternalSync& s,const Buffer& buffer) {
    auto memory_handle=reinterpret_cast<PFN_vkGetMemoryWin32HandleKHR>(vkGetDeviceProcAddr(c.device,"vkGetMemoryWin32HandleKHR"));
    auto semaphore_handle=reinterpret_cast<PFN_vkGetSemaphoreWin32HandleKHR>(vkGetDeviceProcAddr(c.device,"vkGetSemaphoreWin32HandleKHR"));
    require(memory_handle && semaphore_handle,"Win32 external handle functions unavailable");
    VkExportSemaphoreWin32HandleInfoKHR win{VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_WIN32_HANDLE_INFO_KHR};
    win.dwAccess=GENERIC_ALL;
    VkExportSemaphoreCreateInfo ext{VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO};
    ext.pNext=&win; ext.handleTypes=VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;
    VkSemaphoreCreateInfo si{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO}; si.pNext=&ext;
    for (auto* owner:{&s.writable,&s.ready})
        acquire(*owner,"create shared buffer semaphore",[&](auto out){return vkCreateSemaphore(c.device,&si,nullptr,out);});
    VkMemoryGetWin32HandleInfoKHR mem{VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR};
    mem.memory=buffer.memory; mem.handleType=VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
    acquire(s.handles[0],"export dedicated allocation",[&](auto out){return memory_handle(c.device,&mem,out);});
    VkSemaphoreGetWin32HandleInfoKHR sem{VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR};
    sem.handleType=VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;
    sem.semaphore=s.writable;
    acquire(s.handles[1],"export writable semaphore",[&](auto out){return semaphore_handle(c.device,&sem,out);});
    sem.semaphore=s.ready;
    acquire(s.handles[2],"export ready semaphore",[&](auto out){return semaphore_handle(c.device,&sem,out);});
}
#endif
PrResidentSharedExport describe_external(const ExternalSync& s,const Buffer& buffer,uint64_t bytes) {
    return {reinterpret_cast<uintptr_t>(s.handles[0]),reinterpret_cast<uintptr_t>(s.handles[1]),
            reinterpret_cast<uintptr_t>(s.handles[2]),buffer.allocation.bytes,bytes};
}
void export_shared(Context& c,PrResidentSharedExport& output) {
#ifdef _WIN32
    require(c.shared_input && c.resident_state && !c.poisoned,"shared geometry has not been uploaded");
    auto& s=c.shared_slot;
    require(!s.exported && (s.phase==SharedPhase::unused || s.phase==SharedPhase::releasing),
            "shared slot already exported");
    auto& buffer=c.resident_state->transforms;
    if (s.phase==SharedPhase::unused) try {
        create_external_exports(c,s,buffer);
        VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        acquire(s.fence,"create shared ownership fence",[&](auto out){return vkCreateFence(c.device,&fi,nullptr,out);});
        VkCommandBufferAllocateInfo ci{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        ci.commandPool=c.command_pool; ci.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY; ci.commandBufferCount=1;
        acquire(s.commands,"allocate shared ownership command",[&](auto out){return vkAllocateCommandBuffers(c.device,&ci,out);});
        enqueue_shared_ownership(c,false,true);
    } catch (...) { destroy_shared(c); throw; }
    // Complete the initial release before publishing handles. A timeout here
    // retains the fence, allocation and handles for a retry of export or close.
    finish_shared_ownership(c);
    s.exported=true;
    output=describe_external(s,buffer,(static_cast<uint64_t>(c.resident_properties.instance_count)+
                                      c.shared_camera_capacity)*sizeof(GpuTransform));
#else
    (void)c; (void)output;
    throw std::runtime_error("resident external handles require Win32");
#endif
}
void begin_shared(Context& c) {
    auto& s=c.shared_slot;
    require(c.shared_input && s.exported && !c.poisoned,"shared slot unavailable");
    const bool retry=s.phase==SharedPhase::acquiring;
    finish_shared_ownership(c);
    if (retry) return;
    require(s.phase==SharedPhase::external,"a shared publication is already open");
    enqueue_shared_ownership(c,true);
    finish_shared_ownership(c);
}
void end_shared(Context& c) {
    auto& s=c.shared_slot;
    require(c.shared_input && s.exported,"shared slot unavailable");
    const bool retry=s.phase==SharedPhase::releasing;
    finish_shared_ownership(c);
    if (retry) return;
    require(s.phase==SharedPhase::reading,"no acquired shared publication to return");
    enqueue_shared_ownership(c,false);
    finish_shared_ownership(c);
}
template<class F> int shared_call(F action) {
    try { last_error.clear(); action(); return 0; }
    catch (const SharedTimeout& e) { last_error=e.what(); return -2; }
    catch (const std::exception& e) { last_error=e.what(); return -1; }
    catch (...) { last_error="unknown shared resident error"; return -1; }
}
int drain_shared(Context& c) {
    return shared_call([&]{
        finish_shared_ownership(c);
        if (c.shared_slot.phase==SharedPhase::reading) end_shared(c);
    });
}
