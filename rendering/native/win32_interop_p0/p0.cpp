#include "p0.h"
#include <windows.h>
#include <vulkan/vulkan.h>

#include <array>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr auto mem_type = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
constexpr auto sem_type = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;
constexpr VkBufferUsageFlags shared_usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
constexpr uint64_t drain_ns = 250000000;
std::atomic<uint64_t> live{0};
std::atomic<uint32_t> global_errors{0};
thread_local char last_error[768]{};
static_assert(sizeof(void*) == 8 && sizeof(Pv0Export) == 40 && sizeof(Pv0Result) == 16);

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void check(VkResult result, const char* label) {
    if (result != VK_SUCCESS) throw std::runtime_error(std::string(label)+" VkResult="+std::to_string(result));
}
template<class Fn> int call(Fn fn) noexcept {
    last_error[0]=0;
    try { return fn(); }
    catch (const std::exception& e) { std::snprintf(last_error,sizeof(last_error),"%s",e.what()); }
    catch (...) { std::snprintf(last_error,sizeof(last_error),"unknown P0 error"); }
    return 1;
}
template<class H,class Fn> void make(H& handle, Fn fn, const char* label) {
    H created{};
    check(fn(&created),label);
    require(created!=H{},"Vulkan returned a null resource");
    handle=created;
    ++live;
}
template<class H,class Fn> void drop(H& handle,Fn destroy) noexcept {
    if (handle) { destroy(handle); handle={}; --live; }
}
struct Buffer {
    VkBuffer buffer{};
    VkDeviceMemory memory{};
    VkDeviceSize size{};
    VkMemoryPropertyFlags properties{};
};
struct Slot {
    Buffer payload, audit, readback;
    VkSemaphore writable{}, ready{};
    VkFence fence{};
    VkCommandBuffer commands{};
    VkDescriptorSet descriptors{};
    std::array<HANDLE,3> handles{};
    bool exported=false, pending=false;
};
struct Context {
    VkInstance instance{};
    VkDebugUtilsMessengerEXT messenger{};
    PFN_vkDestroyDebugUtilsMessengerEXT destroy_messenger{};
    VkPhysicalDevice physical{};
    VkPhysicalDeviceMemoryProperties memory_types{};
    VkDevice device{};
    VkQueue queue{};
    uint32_t family=0, words=0;
    VkDeviceSize bytes=0;
    PFN_vkGetMemoryWin32HandleKHR export_memory{};
    PFN_vkGetSemaphoreWin32HandleKHR export_semaphore{};
    VkDescriptorSetLayout descriptor_layout{};
    VkPipelineLayout pipeline_layout{};
    VkShaderModule shader{};
    VkPipeline pipeline{};
    VkDescriptorPool descriptor_pool{};
    VkCommandPool command_pool{};
    VkCommandBuffer initial_commands{};
    VkFence initial_fence{};
    bool initial_submitted=false;
    std::atomic<uint32_t> validation_errors{0};
    std::array<Slot,2> slots{};
    Context() { ++live; }
    Context(const Context&)=delete;
    Context& operator=(const Context&)=delete;
    ~Context();
    Slot& slot(uint32_t index) {
        require(index<slots.size(),"slot index must be 0 or 1");
        return slots[index];
    }
};
Context& ctx(void* p) { require(p!=nullptr,"null P0 context"); return *static_cast<Context*>(p); }

void close_handles(Slot& slot) noexcept {
    for (auto& h:slot.handles) drop(h,[](HANDLE v){ CloseHandle(v); });
}
Context::~Context() {
    // Open submits only after all fallible setup has completed. The public
    // close drains fences before this destructor; device loss allows teardown.
    for (auto& s:slots) {
        close_handles(s);
        drop(s.descriptors,[](auto){}); // implicitly destroyed with descriptor_pool
        drop(s.commands,[](auto){});    // implicitly destroyed with command_pool
        drop(s.fence,[&](auto v){vkDestroyFence(device,v,nullptr);});
        drop(s.ready,[&](auto v){vkDestroySemaphore(device,v,nullptr);});
        drop(s.writable,[&](auto v){vkDestroySemaphore(device,v,nullptr);});
        for (Buffer* b:{&s.readback,&s.audit,&s.payload}) {
            drop(b->buffer,[&](auto v){vkDestroyBuffer(device,v,nullptr);});
            drop(b->memory,[&](auto v){vkFreeMemory(device,v,nullptr);});
        }
    }
    drop(initial_commands,[](auto){});
    drop(initial_fence,[&](auto v){vkDestroyFence(device,v,nullptr);});
    drop(command_pool,[&](auto v){vkDestroyCommandPool(device,v,nullptr);});
    drop(descriptor_pool,[&](auto v){vkDestroyDescriptorPool(device,v,nullptr);});
    drop(pipeline,[&](auto v){vkDestroyPipeline(device,v,nullptr);});
    drop(shader,[&](auto v){vkDestroyShaderModule(device,v,nullptr);});
    drop(pipeline_layout,[&](auto v){vkDestroyPipelineLayout(device,v,nullptr);});
    drop(descriptor_layout,[&](auto v){vkDestroyDescriptorSetLayout(device,v,nullptr);});
    drop(device,[](auto v){vkDestroyDevice(v,nullptr);});
    drop(messenger,[&](auto v){destroy_messenger(instance,v,nullptr);});
    drop(instance,[](auto v){vkDestroyInstance(v,nullptr);});
    --live;
}

VKAPI_ATTR VkBool32 VKAPI_CALL validation(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
        VkDebugUtilsMessageTypeFlagsEXT, const VkDebugUtilsMessengerCallbackDataEXT* msg, void* data) {
    if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
        ++static_cast<Context*>(data)->validation_errors;
        ++global_errors;
        std::fprintf(stderr,"P0 Vulkan validation: %s\n",msg && msg->pMessage ? msg->pMessage : "(no message)");
    }
    return VK_FALSE;
}
void init_device(Context& c,const uint8_t* uuid,bool with_validation) {
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName="Phymeo external-buffer P0"; app.apiVersion=VK_API_VERSION_1_1;
    VkInstanceCreateInfo ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO}; ci.pApplicationInfo=&app;
    const char* layer="VK_LAYER_KHRONOS_validation";
    const char* debug_ext=VK_EXT_DEBUG_UTILS_EXTENSION_NAME;
    VkDebugUtilsMessengerCreateInfoEXT debug{VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
    if (with_validation) {
        debug.messageSeverity=VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        debug.messageType=VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT|VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT|
                          VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        debug.pfnUserCallback=validation; debug.pUserData=&c;
        ci.enabledLayerCount=1; ci.ppEnabledLayerNames=&layer;
        ci.enabledExtensionCount=1; ci.ppEnabledExtensionNames=&debug_ext; ci.pNext=&debug;
    }
    make(c.instance,[&](auto p){return vkCreateInstance(&ci,nullptr,p);},"create instance");
    if (with_validation) {
        auto create=reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
            vkGetInstanceProcAddr(c.instance,"vkCreateDebugUtilsMessengerEXT"));
        c.destroy_messenger=reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
            vkGetInstanceProcAddr(c.instance,"vkDestroyDebugUtilsMessengerEXT"));
        require(create && c.destroy_messenger,"missing validation messenger entry points");
        make(c.messenger,[&](auto p){return create(c.instance,&debug,nullptr,p);},"create validation messenger");
    }
    uint32_t count=0;
    check(vkEnumeratePhysicalDevices(c.instance,&count,nullptr),"enumerate devices");
    require(count>0,"no Vulkan physical devices");
    std::vector<VkPhysicalDevice> devices(count);
    check(vkEnumeratePhysicalDevices(c.instance,&count,devices.data()),"read physical devices");
    VkPhysicalDeviceProperties properties{};
    for (auto d:devices) {
        VkPhysicalDeviceIDProperties id{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
        VkPhysicalDeviceProperties2 p{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2}; p.pNext=&id;
        vkGetPhysicalDeviceProperties2(d,&p);
        if (std::memcmp(uuid,id.deviceUUID,VK_UUID_SIZE)==0) { c.physical=d; properties=p.properties; break; }
    }
    require(c.physical!=VK_NULL_HANDLE,"CUDA UUID does not match a Vulkan physical device");
    require(properties.apiVersion>=VK_API_VERSION_1_1,"Vulkan 1.1 external capabilities required");
    require(c.bytes<=properties.limits.maxStorageBufferRange &&
            (c.words+63u)/64u<=properties.limits.maxComputeWorkGroupCount[0],
            "requested storage buffer or compute grid exceeds device limit");
    VkPhysicalDeviceExternalBufferInfo bi{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_BUFFER_INFO};
    bi.usage=shared_usage; bi.handleType=mem_type;
    VkExternalBufferProperties bp{VK_STRUCTURE_TYPE_EXTERNAL_BUFFER_PROPERTIES};
    vkGetPhysicalDeviceExternalBufferProperties(c.physical,&bi,&bp);
    const auto& mp=bp.externalMemoryProperties;
    const auto mf=VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT|VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT;
    require((mp.externalMemoryFeatures&mf)==mf && (mp.compatibleHandleTypes&mem_type),
            "opaque Win32 buffer memory is not importable and exportable");
    VkPhysicalDeviceExternalSemaphoreInfo si{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_SEMAPHORE_INFO}; si.handleType=sem_type;
    VkExternalSemaphoreProperties sp{VK_STRUCTURE_TYPE_EXTERNAL_SEMAPHORE_PROPERTIES};
    vkGetPhysicalDeviceExternalSemaphoreProperties(c.physical,&si,&sp);
    const auto sf=VK_EXTERNAL_SEMAPHORE_FEATURE_EXPORTABLE_BIT|VK_EXTERNAL_SEMAPHORE_FEATURE_IMPORTABLE_BIT;
    require((sp.externalSemaphoreFeatures&sf)==sf && (sp.compatibleHandleTypes&sem_type),
            "opaque Win32 binary semaphore is not importable and exportable");
    vkGetPhysicalDeviceMemoryProperties(c.physical,&c.memory_types);
    vkGetPhysicalDeviceQueueFamilyProperties(c.physical,&count,nullptr);
    std::vector<VkQueueFamilyProperties> families(count);
    vkGetPhysicalDeviceQueueFamilyProperties(c.physical,&count,families.data());
    for (c.family=0;c.family<count;++c.family)
        if (families[c.family].queueCount && (families[c.family].queueFlags&VK_QUEUE_COMPUTE_BIT)) break;
    require(c.family<count,"no compute-capable Vulkan queue");
    const float priority=1.f;
    VkDeviceQueueCreateInfo qi{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qi.queueFamilyIndex=c.family; qi.queueCount=1; qi.pQueuePriorities=&priority;
    const char* extensions[]={VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME,VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME};
    VkDeviceCreateInfo di{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    di.queueCreateInfoCount=1; di.pQueueCreateInfos=&qi; di.enabledExtensionCount=2; di.ppEnabledExtensionNames=extensions;
    make(c.device,[&](auto p){return vkCreateDevice(c.physical,&di,nullptr,p);},"create Win32 interop device");
    vkGetDeviceQueue(c.device,c.family,0,&c.queue);
    c.export_memory=reinterpret_cast<PFN_vkGetMemoryWin32HandleKHR>(vkGetDeviceProcAddr(c.device,"vkGetMemoryWin32HandleKHR"));
    c.export_semaphore=reinterpret_cast<PFN_vkGetSemaphoreWin32HandleKHR>(vkGetDeviceProcAddr(c.device,"vkGetSemaphoreWin32HandleKHR"));
    require(c.export_memory && c.export_semaphore,"missing Win32 memory/semaphore export entry points");
}

void create_buffer(Context& c, Buffer& b, VkDeviceSize bytes, VkBufferUsageFlags usage,
                   VkMemoryPropertyFlags required, bool external=false) {
    VkExternalMemoryBufferCreateInfo external_info{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};
    external_info.handleTypes=mem_type;
    VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    info.pNext=external ? &external_info : nullptr;
    info.size=bytes; info.usage=usage; info.sharingMode=VK_SHARING_MODE_EXCLUSIVE;
    make(b.buffer,[&](auto p){return vkCreateBuffer(c.device,&info,nullptr,p);},"create buffer");
    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(c.device,b.buffer,&req);
    uint32_t type=VK_MAX_MEMORY_TYPES;
    for (uint32_t i=0;i<c.memory_types.memoryTypeCount;++i) {
        const auto flags=c.memory_types.memoryTypes[i].propertyFlags;
        if (!(req.memoryTypeBits&(1u<<i)) || (flags&required)!=required) continue;
        type=i;
        if (!(required&VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) || (flags&VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) break;
    }
    require(type!=VK_MAX_MEMORY_TYPES,"no suitable memory type for P0 buffer");
    VkMemoryDedicatedAllocateInfo dedicated{VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO};
    dedicated.buffer=b.buffer;
    VkExportMemoryWin32HandleInfoKHR win{VK_STRUCTURE_TYPE_EXPORT_MEMORY_WIN32_HANDLE_INFO_KHR};
    win.dwAccess=GENERIC_ALL; win.pNext=&dedicated;
    VkExportMemoryAllocateInfo exp{VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO};
    exp.handleTypes=mem_type; exp.pNext=&win;
    VkMemoryAllocateInfo alloc{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    alloc.pNext=external ? static_cast<const void*>(&exp) : static_cast<const void*>(&dedicated);
    alloc.allocationSize=req.size; alloc.memoryTypeIndex=type;
    make(b.memory,[&](auto p){return vkAllocateMemory(c.device,&alloc,nullptr,p);},"allocate buffer memory");
    check(vkBindBufferMemory(c.device,b.buffer,b.memory,0),"bind buffer allocation");
    b.properties=c.memory_types.memoryTypes[type].propertyFlags; b.size=req.size;
}

void create_semaphore(Context& c,VkSemaphore& semaphore) {
    VkExportSemaphoreWin32HandleInfoKHR win{VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_WIN32_HANDLE_INFO_KHR};
    win.dwAccess=GENERIC_ALL;
    VkExportSemaphoreCreateInfo exp{VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO};
    exp.handleTypes=sem_type; exp.pNext=&win;
    VkSemaphoreCreateInfo info{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    info.pNext=&exp;
    make(semaphore,[&](auto p){return vkCreateSemaphore(c.device,&info,nullptr,p);},"create exportable semaphore");
}
void create_commands(Context& c,VkCommandBuffer& cmd,VkFence& fence) {
    VkCommandBufferAllocateInfo info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    info.commandPool=c.command_pool; info.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY; info.commandBufferCount=1;
    make(cmd,[&](auto p){return vkAllocateCommandBuffers(c.device,&info,p);},"allocate command buffer");
    VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    make(fence,[&](auto p){return vkCreateFence(c.device,&fi,nullptr,p);},"create fence");
}

void create_pipeline(Context& c,const wchar_t* path) {
    std::ifstream input(std::filesystem::path(path),std::ios::binary|std::ios::ate);
    require(input.good(),"cannot open verifier SPIR-V");
    const std::streamoff length=input.tellg();
    require(length>0 && length<=1024*1024 && length%4==0,"invalid verifier SPIR-V size");
    std::vector<uint32_t> words(static_cast<size_t>(length)/4);
    input.seekg(0);
    input.read(reinterpret_cast<char*>(words.data()),length);
    require(input.good() && words[0]==0x07230203,"invalid verifier SPIR-V contents");
    VkShaderModuleCreateInfo shader{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    shader.codeSize=static_cast<size_t>(length); shader.pCode=words.data();
    make(c.shader,[&](auto p){return vkCreateShaderModule(c.device,&shader,nullptr,p);},"create verification shader");
    std::array<VkDescriptorSetLayoutBinding,2> bindings{};
    for (uint32_t i=0;i<bindings.size();++i) {
        bindings[i].binding=i; bindings[i].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount=1; bindings[i].stageFlags=VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo layout{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    layout.bindingCount=static_cast<uint32_t>(bindings.size()); layout.pBindings=bindings.data();
    make(c.descriptor_layout,[&](auto p){return vkCreateDescriptorSetLayout(c.device,&layout,nullptr,p);},"create descriptor layout");
    VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT,0,2*sizeof(uint32_t)};
    VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pli.setLayoutCount=1; pli.pSetLayouts=&c.descriptor_layout; pli.pushConstantRangeCount=1; pli.pPushConstantRanges=&push;
    make(c.pipeline_layout,[&](auto p){return vkCreatePipelineLayout(c.device,&pli,nullptr,p);},"create pipeline layout");
    VkComputePipelineCreateInfo pipe{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipe.stage.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipe.stage.stage=VK_SHADER_STAGE_COMPUTE_BIT; pipe.stage.module=c.shader; pipe.stage.pName="main"; pipe.layout=c.pipeline_layout;
    make(c.pipeline,[&](auto p){return vkCreateComputePipelines(c.device,VK_NULL_HANDLE,1,&pipe,nullptr,p);},"create verification pipeline");
    VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,4};
    VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpi.maxSets=2; dpi.poolSizeCount=1; dpi.pPoolSizes=&size;
    make(c.descriptor_pool,[&](auto p){return vkCreateDescriptorPool(c.device,&dpi,nullptr,p);},"create descriptor pool");
    VkCommandPoolCreateInfo ci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    ci.queueFamilyIndex=c.family; ci.flags=VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    make(c.command_pool,[&](auto p){return vkCreateCommandPool(c.device,&ci,nullptr,p);},"create command pool");
}
void create_slot(Context& c,Slot& s) {
    create_buffer(c,s.payload,c.bytes,shared_usage,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,true);
    create_buffer(c,s.audit,sizeof(Pv0Result),VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                  VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    create_buffer(c,s.readback,sizeof(Pv0Result),VK_BUFFER_USAGE_TRANSFER_DST_BIT,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    create_semaphore(c,s.writable); create_semaphore(c,s.ready);
    create_commands(c,s.commands,s.fence);
    VkDescriptorSetAllocateInfo alloc{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    alloc.descriptorPool=c.descriptor_pool; alloc.descriptorSetCount=1; alloc.pSetLayouts=&c.descriptor_layout;
    make(s.descriptors,[&](auto p){return vkAllocateDescriptorSets(c.device,&alloc,p);},"allocate slot descriptor set");
    VkDescriptorBufferInfo buffers[]={{s.payload.buffer,0,c.bytes},{s.audit.buffer,0,sizeof(Pv0Result)}};
    std::array<VkWriteDescriptorSet,2> writes{};
    for (uint32_t i=0;i<writes.size();++i) {
        auto& w=writes[i]; w.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; w.dstSet=s.descriptors; w.dstBinding=i;
        w.descriptorCount=1; w.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w.pBufferInfo=&buffers[i];
    }
    vkUpdateDescriptorSets(c.device,static_cast<uint32_t>(writes.size()),writes.data(),0,nullptr);
}

void barrier(VkCommandBuffer cmd,const Buffer& b,VkDeviceSize size,
             VkAccessFlags from,VkAccessFlags to,VkPipelineStageFlags before,VkPipelineStageFlags after,
             uint32_t src=VK_QUEUE_FAMILY_IGNORED,uint32_t dst=VK_QUEUE_FAMILY_IGNORED) {
    VkBufferMemoryBarrier v{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    v.srcAccessMask=from; v.dstAccessMask=to; v.srcQueueFamilyIndex=src; v.dstQueueFamilyIndex=dst;
    v.buffer=b.buffer; v.offset=0; v.size=size;
    vkCmdPipelineBarrier(cmd,before,after,0,0,nullptr,1,&v,0,nullptr);
}
void begin_commands(VkCommandBuffer cmd) {
    VkCommandBufferBeginInfo info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    info.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    check(vkBeginCommandBuffer(cmd,&info),"begin command buffer");
}
void initial_release(Context& c) {
    create_commands(c,c.initial_commands,c.initial_fence);
    auto cmd=c.initial_commands;
    begin_commands(cmd);
    for (const auto& s:c.slots) {
        vkCmdFillBuffer(cmd,s.payload.buffer,0,c.bytes,0);
        barrier(cmd,s.payload,c.bytes,VK_ACCESS_TRANSFER_WRITE_BIT,0,VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,c.family,VK_QUEUE_FAMILY_EXTERNAL);
    }
    check(vkEndCommandBuffer(cmd),"end initial release");
    VkSemaphore signals[]={c.slots[0].writable,c.slots[1].writable};
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount=1; submit.pCommandBuffers=&cmd; submit.signalSemaphoreCount=2; submit.pSignalSemaphores=signals;
    check(vkQueueSubmit(c.queue,1,&submit,c.initial_fence),"submit initial release to CUDA");
    c.initial_submitted=true;
}
void submit_slot(Context& c,Slot& s,uint32_t marker) {
    require(s.exported,"slot must be exported before submitting");
    require(!s.pending,"slot has a pending fence");
    auto cmd=s.commands;
    check(vkResetCommandBuffer(cmd,0),"reset slot commands");
    check(vkResetFences(c.device,1,&s.fence),"reset slot fence");
    begin_commands(cmd);
    const auto shader_rw=VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT;
    barrier(cmd,s.payload,c.bytes,0,shader_rw,VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_QUEUE_FAMILY_EXTERNAL,c.family);
    vkCmdFillBuffer(cmd,s.audit.buffer,0,sizeof(Pv0Result),0);
    barrier(cmd,s.audit,sizeof(Pv0Result),VK_ACCESS_TRANSFER_WRITE_BIT,shader_rw,
            VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,c.pipeline);
    vkCmdBindDescriptorSets(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,c.pipeline_layout,0,1,&s.descriptors,0,nullptr);
    const uint32_t constants[]={c.words,marker};
    vkCmdPushConstants(cmd,c.pipeline_layout,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(constants),constants);
    vkCmdDispatch(cmd,(c.words+63)/64,1,1);
    barrier(cmd,s.payload,c.bytes,shader_rw,0,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,c.family,VK_QUEUE_FAMILY_EXTERNAL);
    barrier(cmd,s.audit,sizeof(Pv0Result),VK_ACCESS_SHADER_WRITE_BIT,VK_ACCESS_TRANSFER_READ_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT);
    VkBufferCopy copy{0,0,sizeof(Pv0Result)};
    vkCmdCopyBuffer(cmd,s.audit.buffer,s.readback.buffer,1,&copy);
    barrier(cmd,s.readback,sizeof(Pv0Result),VK_ACCESS_TRANSFER_WRITE_BIT,VK_ACCESS_HOST_READ_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_HOST_BIT);
    check(vkEndCommandBuffer(cmd),"end slot verification");
    VkPipelineStageFlags stage=VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.waitSemaphoreCount=1; submit.pWaitSemaphores=&s.ready; submit.pWaitDstStageMask=&stage;
    submit.commandBufferCount=1; submit.pCommandBuffers=&cmd; submit.signalSemaphoreCount=1; submit.pSignalSemaphores=&s.writable;
    check(vkQueueSubmit(c.queue,1,&submit,s.fence),"submit external wait, verify and return");
    s.pending=true;
}
int timeout(const char* label) {
    std::snprintf(last_error,sizeof(last_error),"%s timed out; submission and context retained",label);
    return 2;
}
} // namespace

int pv0_open(const uint8_t uuid[16],uint32_t words,uint32_t flags,const wchar_t* compute_spv,void** out) {
    return call([&] {
        require(out && !*out,"expected empty P0 output context");
        require(uuid && compute_spv && *compute_spv,"expected device UUID and shader path");
        require(words>0 && words<=1048576,"word count must be between 1 and 1048576");
        require(!(flags&~7u),"unrecognized flags");
        require(!(flags&4u),"unsupported handle type OPAQUE_FD for Win32 P0");
        auto c=std::make_unique<Context>();
        c->words=words; c->bytes=VkDeviceSize(words)*sizeof(uint32_t);
        init_device(*c,uuid,(flags&1u)!=0);
        create_pipeline(*c,compute_spv);
        create_slot(*c,c->slots[0]);
        require(!(flags&2u),"injected initialization failure after first slot");
        create_slot(*c,c->slots[1]);
        initial_release(*c);
        *out=c.release();
        return 0;
    });
}
int pv0_export(void* handle,uint32_t index,Pv0Export* out) {
    return call([&] {
        require(out,"null export output");
        *out={};
        auto& c=ctx(handle); auto& s=c.slot(index);
        require(!s.exported,"slot already exported");
        VkMemoryGetWin32HandleInfoKHR mem{VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR};
        mem.memory=s.payload.memory; mem.handleType=mem_type;
        VkSemaphoreGetWin32HandleInfoKHR sem{VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR};
        sem.handleType=sem_type;
        try {
            make(s.handles[0],[&](auto p){return c.export_memory(c.device,&mem,p);},"export memory NT handle");
            sem.semaphore=s.writable;
            make(s.handles[1],[&](auto p){return c.export_semaphore(c.device,&sem,p);},"export writable NT handle");
            sem.semaphore=s.ready;
            make(s.handles[2],[&](auto p){return c.export_semaphore(c.device,&sem,p);},"export ready NT handle");
        } catch (...) { close_handles(s); throw; }
        *out={reinterpret_cast<uintptr_t>(s.handles[0]),reinterpret_cast<uintptr_t>(s.handles[1]),
              reinterpret_cast<uintptr_t>(s.handles[2]),s.payload.size,c.bytes};
        s.exported=true;
        return 0;
    });
}
int pv0_drop_exports(void* handle,uint32_t index) {
    return call([&] {
        auto& s=ctx(handle).slot(index);
        require(s.exported,"slot has never been exported");
        close_handles(s);
        return 0;
    });
}
int pv0_submit(void* handle,uint32_t index,uint32_t marker) {
    return call([&] {
        auto& c=ctx(handle);
        submit_slot(c,c.slot(index),marker);
        return 0;
    });
}
int pv0_finish(void* handle,uint32_t index,uint64_t timeout_ns,Pv0Result* out) {
    return call([&] {
        require(out,"null verifier result");
        *out={};
        require(timeout_ns!=UINT64_MAX,"a finite timeout is required");
        auto& c=ctx(handle); auto& s=c.slot(index);
        require(s.pending,"slot has no pending submission");
        const auto wait=vkWaitForFences(c.device,1,&s.fence,VK_TRUE,timeout_ns);
        if (wait==VK_TIMEOUT) return timeout("finish");
        check(wait,"wait for slot verification");
        void* data=nullptr;
        check(vkMapMemory(c.device,s.readback.memory,0,VK_WHOLE_SIZE,0,&data),"map audit result");
        if (!(s.readback.properties&VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
            VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
            range.memory=s.readback.memory; range.size=VK_WHOLE_SIZE;
            const auto invalidate=vkInvalidateMappedMemoryRanges(c.device,1,&range);
            if (invalidate!=VK_SUCCESS) { vkUnmapMemory(c.device,s.readback.memory); check(invalidate,"invalidate audit result"); }
        }
        std::memcpy(out,data,sizeof(*out));
        vkUnmapMemory(c.device,s.readback.memory);
        s.pending=false;
        return 0;
    });
}
int pv0_close(void** handle) {
    return call([&] {
        require(handle,"null context pointer");
        if (!*handle) return 0;
        auto* c=static_cast<Context*>(*handle);
        std::array<VkFence,3> fences{};
        uint32_t count=0;
        if (c->initial_submitted) fences[count++]=c->initial_fence;
        for (const auto& s:c->slots) if (s.pending) fences[count++]=s.fence;
        auto rc=count ? vkWaitForFences(c->device,count,fences.data(),VK_TRUE,drain_ns) : VK_SUCCESS;
        if (rc==VK_TIMEOUT) return timeout("close");
        if (rc!=VK_ERROR_DEVICE_LOST) check(rc,"drain P0 context");
        delete c;
        *handle=nullptr;
        check(rc,"P0 device lost during close");
        return 0;
    });
}
const char* pv0_error() { return last_error; }
uint64_t pv0_live() { return live.load(); }
uint32_t pv0_validation_errors(void* handle) {
    return handle ? static_cast<Context*>(handle)->validation_errors.load() : global_errors.load();
}
