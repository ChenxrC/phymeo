#include "renderer.h"
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef VK_USE_PLATFORM_WIN32_KHR
#define VK_USE_PLATFORM_WIN32_KHR
#endif
#include <windows.h>
#endif
#include <vulkan/vulkan.h>
#ifdef interface
#undef interface
#endif
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <functional>
#include <unordered_set>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
thread_local std::string last_error;
// DLL-wide owned objects, including partially initialized contexts/frames.
// Queues/physical devices are borrowed; descriptor sets belong to their pool.
std::atomic<uint64_t> live_contexts{0}, live_frames{0}, live_resources{0};
void require(bool valid, const char* message) {
    if (!valid) throw std::runtime_error(message);
}
void check(VkResult result, const char* operation) {
    if (result != VK_SUCCESS)
        throw std::runtime_error(std::string(operation)+" VkResult="+std::to_string(result));
}
template<class Handle, class Create>
void acquire(Handle& owner, const char* operation, Create create) {
    Handle temporary = VK_NULL_HANDLE;
    // Single-object failure output is undefined: never transfer it to an owner.
    check(create(&temporary), operation);
    owner = temporary;
    ++live_resources;
}
template<class Handle, class Destroy>
void release(Handle& owner, Destroy destroy) noexcept {
    if (owner) {
        destroy(owner);
        owner = VK_NULL_HANDLE;
        --live_resources;
    }
}
constexpr uint32_t MAX_VERTICES = 4096u*3u;
constexpr std::array<VkFormat, 5> FORMATS = {
    VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R32_SFLOAT,
    VK_FORMAT_R32_UINT, VK_FORMAT_R8_UINT, VK_FORMAT_D32_SFLOAT};
struct Allocation {
    VkDeviceSize bytes{};
    bool local_heap{}, host_visible{};
    uint32_t type_index{};
    VkMemoryPropertyFlags property_flags{};
};
struct Buffer { VkBuffer handle{}; VkDeviceMemory memory{}; Allocation allocation{}; };
struct Image { VkImage handle{}; VkDeviceMemory memory{}; VkImageView view{}; Allocation allocation{}; };
#include "resident_spirv.inc"
#include "resident_display.inc"
#include "resident_rayquery_types.inc"
struct ResidentState;
struct MsaaState;
struct ProgressiveState;
struct Context;
struct SharedSlot {
    enum class Phase { unused, external, acquiring, reading, releasing };
    Phase phase=Phase::unused;
    VkSemaphore writable{},ready{};
    VkFence fence{};
    VkCommandBuffer commands{};
    void* handles[3]{};
    bool exported=false;
};
void destroy_shared(Context&) noexcept;
int drain_shared(Context&);
void probe_msaa(Context&,const VkPhysicalDeviceLimits&);
struct Context {
    bool resident = false;
    bool shared_input=false;
    uint32_t shared_camera_capacity=0;
    uint64_t shared_camera_d2d_bytes=0;
    bool shared_blas_compatible=false;
    SharedSlot shared_slot{};
    bool surface_maps_compatible = false;
    bool display_compatible = false;
    PrResidentMsaaInfo msaa_info{};
    PrResidentNormalInfo normal_info{};
    PrResidentReflectionInfo reflection_info{};
    PrResidentIndirectInfo indirect_info{};
    PrResidentProgressiveInfo progressive_info{};
    PrResidentSpatialInfo spatial_info{};
    PrResidentSettlingInfo settling_info{};
    std::shared_ptr<ProgressiveState> progressive_state;
    std::shared_ptr<MsaaState> msaa_state;
    uint32_t display_transform = 0;
    bool ray_query = false;
    RayDispatch ray;
    PrResidentRayQueryInfo ray_info{};
    uint32_t ray_scratch_alignment{};
    PrResidentInfo resident_properties{};
    PrResidentTelemetry telemetry{};
    PrResidentProfile profile{};
    PrResidentReadbackInfo readback_info{};
    PrResidentTextureInfo texture_info{};
    PrResidentEnvironmentInfo environment_info{};
    uint32_t storage_buffer_limit{};
    PrStudio studio{};
    std::shared_ptr<ResidentState> resident_state;
    VkInstance instance{};
    VkDebugUtilsMessengerEXT messenger{};
    VkPhysicalDevice physical{};
    VkDevice device{};
    VkQueue queue{};
    uint32_t queue_family{};
    VkCommandPool command_pool{};
    VkDescriptorSetLayout descriptor_layout{};
    VkPipelineLayout pipeline_layout{};
    VkRenderPass render_pass{};
    VkPipeline pipeline{};
    VkShaderModule vertex_shader{}, fragment_shader{}, shared_instance_shader{};
    VkPhysicalDeviceMemoryProperties memory_properties{};
    PrInfo properties{};
    std::atomic<uint64_t> errors{0}, warnings{0};
    std::mutex message_lock;
    std::string validation_message;
    bool poisoned = false;

    Context() {
        ++live_contexts; telemetry.telemetry_abi_version=1; profile.profile_abi_version=1;
        readback_info.readback_abi_version=1; readback_info.memory_type_index=UINT32_MAX;
        texture_info.texture_abi_version=1;
        normal_info.normal_abi_version=1;
        progressive_info.progressive_abi_version=1;
        spatial_info.spatial_abi_version=1;
        settling_info.settling_abi_version=1;
        indirect_info.indirect_abi_version=1;indirect_info.samples_per_pixel=16;indirect_info.max_bounces=1;
        reflection_info.reflection_abi_version=1; reflection_info.max_bounces=1;
        reflection_info.metallic_min=.99f; reflection_info.roughness_max=.10f;
        environment_info.environment_abi_version=1;
        ray_info.ray_query_abi_version=1;
        msaa_info.msaa_abi_version=1; msaa_info.requested_samples=msaa_info.color_samples=msaa_info.sensor_samples=1;
    }
    ~Context() { shutdown(); --live_contexts; }
    VkResult shutdown() noexcept {
        VkResult waited = VK_SUCCESS;
        if (device) {
            waited = vkDeviceWaitIdle(device);
            progressive_state.reset();
            destroy_shared(*this);
            resident_state.reset(); // dependent buffers/targets die before VkDevice
            msaa_state.reset(); // per-camera framebuffers die before their shared render passes
            release(pipeline, [&](auto h) { vkDestroyPipeline(device,h,nullptr); });
            release(pipeline_layout, [&](auto h) { vkDestroyPipelineLayout(device,h,nullptr); });
            release(render_pass, [&](auto h) { vkDestroyRenderPass(device,h,nullptr); });
            release(vertex_shader, [&](auto h) { vkDestroyShaderModule(device,h,nullptr); });
            release(fragment_shader, [&](auto h) { vkDestroyShaderModule(device,h,nullptr); });
            release(shared_instance_shader, [&](auto h) { vkDestroyShaderModule(device,h,nullptr); });
            release(descriptor_layout, [&](auto h) { vkDestroyDescriptorSetLayout(device,h,nullptr); });
            release(command_pool, [&](auto h) { vkDestroyCommandPool(device,h,nullptr); });
            release(device, [&](auto h) { vkDestroyDevice(h,nullptr); });
        }
        if (messenger && instance) {
            auto destroy = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
                vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT"));
            release(messenger, [&](auto h) { destroy(instance,h,nullptr); });
        }
        release(instance, [&](auto h) { vkDestroyInstance(h,nullptr); });
        return waited;
    }
    VkFormat attachment_format(size_t i) const {
        if(i==5)return VK_FORMAT_R8G8B8A8_UNORM;
        return i==0 && progressive_info.shader_compatible ? VK_FORMAT_R32G32B32A32_SFLOAT : FORMATS[i];
    }
    void info(PrInfo* output) const {
        *output = properties;
        output->validation_errors = errors.load();
        output->validation_warnings = warnings.load();
    }
    static VKAPI_ATTR VkBool32 VKAPI_CALL debug(
        VkDebugUtilsMessageSeverityFlagBitsEXT severity,
        VkDebugUtilsMessageTypeFlagsEXT,
        const VkDebugUtilsMessengerCallbackDataEXT* data, void* user) {
        auto* context = static_cast<Context*>(user);
        if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) ++context->errors;
        if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) ++context->warnings;
        const char* message = data && data->pMessage ? data->pMessage : "Vulkan validation message";
        { std::lock_guard<std::mutex> guard(context->message_lock);
          context->validation_message = message; }
        std::fprintf(stderr, "phymeo-native validation: %s\n", message);
        return VK_FALSE;
    }
    uint32_t memory_type(uint32_t bits, VkMemoryPropertyFlags wanted,
                         VkMemoryPropertyFlags preferred=0) const {
        // HOST_CACHED is a preference, never a relaxation of HOST_COHERENT.
        // A coherent fallback needs no invalidate and keeps the same contract.
        for (uint32_t pass=0; pass<(preferred ? 2u : 1u); ++pass) {
            VkMemoryPropertyFlags required=wanted | (pass==0 ? preferred : 0u);
            for (uint32_t i=0; i<memory_properties.memoryTypeCount; ++i)
                if ((bits & (1u<<i)) &&
                    (memory_properties.memoryTypes[i].propertyFlags & required)==required) return i;
        }
        throw std::runtime_error("required Vulkan memory type unavailable");
    }
    void account_allocation(Allocation& owner, const VkMemoryAllocateInfo& allocation) {
        owner.bytes = allocation.allocationSize;
        const auto& type = memory_properties.memoryTypes[allocation.memoryTypeIndex];
        owner.type_index=allocation.memoryTypeIndex; owner.property_flags=type.propertyFlags;
        owner.local_heap = (memory_properties.memoryHeaps[type.heapIndex].flags &
                            VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0;
        owner.host_visible = (type.propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0;
        ++telemetry.allocation_count;
        if (owner.local_heap) {
            telemetry.device_local_bytes_current += owner.bytes;
            telemetry.device_local_bytes_allocated += owner.bytes;
            telemetry.device_local_bytes_peak = std::max(
                telemetry.device_local_bytes_peak, telemetry.device_local_bytes_current);
        }
        if (owner.host_visible) {
            telemetry.host_visible_bytes_current += owner.bytes;
            telemetry.host_visible_bytes_allocated += owner.bytes;
            telemetry.host_visible_bytes_peak = std::max(
                telemetry.host_visible_bytes_peak, telemetry.host_visible_bytes_current);
        }
    }
    void free_memory(VkDeviceMemory& memory, Allocation& allocation) noexcept {
        if (!memory) return;
        release(memory, [&](auto h) { vkFreeMemory(device, h, nullptr); });
        if (allocation.local_heap) telemetry.device_local_bytes_current -= allocation.bytes;
        if (allocation.host_visible) telemetry.host_visible_bytes_current -= allocation.bytes;
        ++telemetry.free_count;
        allocation = {};
    }
    void make_buffer(Buffer& value, VkDeviceSize bytes, VkBufferUsageFlags usage,
                     VkMemoryPropertyFlags memory_flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     VkMemoryPropertyFlags preferred_flags=0) {
        VkBufferCreateInfo create{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        create.size = bytes; create.usage = usage; create.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        acquire(value.handle, "vkCreateBuffer", [&](auto* output) { return vkCreateBuffer(device, &create, nullptr, output); });
        VkMemoryRequirements needed{}; vkGetBufferMemoryRequirements(device, value.handle, &needed);
        VkMemoryAllocateInfo alloc{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        alloc.allocationSize = needed.size;
        alloc.memoryTypeIndex = memory_type(needed.memoryTypeBits, memory_flags, preferred_flags);
        VkMemoryAllocateFlagsInfo address_flags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
        if (usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) {
            require(ray_query,"device-address buffer requires explicit ray-query context");
            address_flags.flags=VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT; alloc.pNext=&address_flags;
        }
        acquire(value.memory, "vkAllocateMemory buffer", [&](auto* output) { return vkAllocateMemory(device, &alloc, nullptr, output); });
        account_allocation(value.allocation, alloc);
        check(vkBindBufferMemory(device, value.handle, value.memory, 0), "vkBindBufferMemory");
    }
    void upload(const Buffer& buffer, const void* source, size_t bytes) {
        void* mapped{};
        check(vkMapMemory(device, buffer.memory, 0, bytes, 0, &mapped), "vkMapMemory upload");
        std::memcpy(mapped, source, bytes);
        vkUnmapMemory(device, buffer.memory);
    }
    void make_image(Image& value, uint32_t width, uint32_t height, VkFormat format, bool depth, VkSampleCountFlagBits samples=VK_SAMPLE_COUNT_1_BIT,
                    VkImageUsageFlags extra_usage=0) {
        VkImageCreateInfo create{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        create.imageType = VK_IMAGE_TYPE_2D; create.format = format;
        create.extent = {width, height, 1}; create.mipLevels = 1; create.arrayLayers = 1;
        create.samples = samples; create.tiling = VK_IMAGE_TILING_OPTIMAL;
        create.usage = depth ? VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT :
            (VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
        create.usage |= extra_usage;
        create.sharingMode = VK_SHARING_MODE_EXCLUSIVE; create.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        acquire(value.handle, "vkCreateImage", [&](auto* output) { return vkCreateImage(device, &create, nullptr, output); });
        VkMemoryRequirements needed{}; vkGetImageMemoryRequirements(device, value.handle, &needed);
        VkMemoryAllocateInfo alloc{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        alloc.allocationSize = needed.size;
        alloc.memoryTypeIndex = memory_type(needed.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        acquire(value.memory, "vkAllocateMemory image", [&](auto* output) { return vkAllocateMemory(device, &alloc, nullptr, output); });
        account_allocation(value.allocation, alloc);
        check(vkBindImageMemory(device, value.handle, value.memory, 0), "vkBindImageMemory");
        VkImageViewCreateInfo view{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        view.image = value.handle; view.viewType = VK_IMAGE_VIEW_TYPE_2D; view.format = format;
        view.subresourceRange.aspectMask = depth ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
        view.subresourceRange.levelCount = 1; view.subresourceRange.layerCount = 1;
        acquire(value.view, "vkCreateImageView", [&](auto* output) { return vkCreateImageView(device, &view, nullptr, output); });
    }
    void init(uint32_t selected, bool validation, const uint32_t* vert, size_t vert_bytes,
              const uint32_t* frag, size_t frag_bytes, bool resident_path = false, bool hardware_shadow = false,
              const uint8_t* shared_uuid=nullptr) {
        resident = resident_path; ray_query=hardware_shadow;
        shared_input=shared_uuid!=nullptr;
        shared_blas_compatible=resident && !ray_query && named_shader_marker(frag,frag_bytes,74,"PR_SHARED_BLAS_ABI");
        if (shared_input) {
#ifdef _WIN32
            require(resident && (ray_query || shared_blas_compatible),
                    "shared software matrices require a resident shader with PR_SHARED_BLAS_ABI");
#else
            throw std::runtime_error("shared resident matrices require opaque Win32 handles");
#endif
        }
        normal_info.shader_compatible=resident && named_shader_marker(frag,frag_bytes,62,"PR_NORMAL_MAPS_ABI") &&
            named_shader_marker(vert,vert_bytes,63,"PR_NORMAL_VERTEX_ABI",0);
        if (resident) {
            const auto vs=spirv_interface(vert,vert_bytes,0),fs=spirv_interface(frag,frag_bytes,4);
            const uint64_t normal_bindings=(1ull<<14)|(1ull<<15);
            require(!((vs.bindings|fs.bindings)&normal_bindings) || normal_info.shader_compatible,
                    "normal-map vertex/fragment bundle ABI is missing or incompatible");
        }
        reflection_info.shader_compatible=resident && named_shader_marker(frag,frag_bytes,64,"PR_REFLECTION_ABI");
        require(!reflection_info.shader_compatible || ray_query,"reflection shader requires explicit hardware ray-query context");
        reflection_info.supported=reflection_info.shader_compatible && ray_query;
        indirect_info.shader_compatible=resident && named_shader_marker(frag,frag_bytes,65,"PR_INDIRECT_ABI");
        require(!indirect_info.shader_compatible || ray_query,"diffuse indirect requires explicit hardware ray-query context");
        indirect_info.supported=indirect_info.shader_compatible && ray_query;
        progressive_info.shader_compatible=resident && named_shader_marker(frag,frag_bytes,66,"PR_PROGRESSIVE_ABI");
        require(!progressive_info.shader_compatible || (ray_query&&indirect_info.shader_compatible),"progressive shader requires explicit indirect ray-query context");
        progressive_info.supported=progressive_info.shader_compatible;
        // Reserved environment info words expose optional producer/consumer
        // compatibility without changing either public structure size.
        environment_info.reserved[0]=progressive_info.shader_compatible &&
            named_shader_marker(frag,frag_bytes,72,"PR_ENVIRONMENT_CDF_ABI");
        spatial_info.shader_compatible=progressive_info.shader_compatible;
        require(!spatial_info.shader_compatible||progressive_info.shader_compatible,"spatial guide requires progressive shader");
        spatial_info.supported=spatial_info.shader_compatible;
        surface_maps_compatible=resident && surface_maps_shader_marker(frag,frag_bytes);
        display_compatible=resident && display_shader_marker(frag,frag_bytes);
        msaa_info.shader_compatible=resident && named_shader_marker(frag,frag_bytes,59,"PR_MSAA_COLOR_ABI");
        require(!ray_query || resident,"ray-query shadows require resident geometry");
        if (ray_query) {
            auto interface=spirv_interface(frag,frag_bytes,4);
            require(interface.valid && interface.main && (interface.bindings&(1ull<<13)),
                "ray-query main SPIR-V interface is unavailable; no software fallback");
        }
        require(vert && frag && vert_bytes >= 20 && frag_bytes >= 20 &&
            vert_bytes%4 == 0 && frag_bytes%4 == 0, "invalid SPIR-V input");
        require(vert[0] == 0x07230203 && frag[0] == 0x07230203, "invalid SPIR-V magic");
        VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        app.pApplicationName = "phymeo-native-offscreen";
        app.applicationVersion = 1; app.pEngineName = "phymeo_rendering";
        app.engineVersion = 1; app.apiVersion = ray_query ? VK_API_VERSION_1_2 : VK_API_VERSION_1_1;
        VkInstanceCreateInfo create{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        create.pApplicationInfo = &app;
        const char* layer = "VK_LAYER_KHRONOS_validation";
        const char* extension = VK_EXT_DEBUG_UTILS_EXTENSION_NAME;
        VkDebugUtilsMessengerCreateInfoEXT debug_info{VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
        if (validation) {
            uint32_t count{}; check(vkEnumerateInstanceLayerProperties(&count, nullptr), "enumerate layers");
            std::vector<VkLayerProperties> layers(count);
            check(vkEnumerateInstanceLayerProperties(&count, layers.data()), "enumerate layers");
            require(std::any_of(layers.begin(), layers.end(), [&](const auto& item) {
                return std::strcmp(item.layerName, layer) == 0;
            }), "requested VK_LAYER_KHRONOS_validation is not installed");
            create.enabledLayerCount = 1; create.ppEnabledLayerNames = &layer;
            create.enabledExtensionCount = 1; create.ppEnabledExtensionNames = &extension;
            debug_info.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT |
                VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT;
            debug_info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
            debug_info.pfnUserCallback = debug; debug_info.pUserData = this;
            create.pNext = &debug_info;
        }
        acquire(instance, "vkCreateInstance", [&](auto* output) { return vkCreateInstance(&create, nullptr, output); });
        if (validation) {
            auto make = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
                vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT"));
            require(make != nullptr, "debug-utils messenger unavailable");
            acquire(messenger, "vkCreateDebugUtilsMessengerEXT", [&](auto* output) { return make(instance, &debug_info, nullptr, output); });
        }
        uint32_t count{}; check(vkEnumeratePhysicalDevices(instance, &count, nullptr), "enumerate devices");
        require(selected < count, "requested Vulkan physical device index is unavailable");
        std::vector<VkPhysicalDevice> physicals(count);
        check(vkEnumeratePhysicalDevices(instance, &count, physicals.data()), "enumerate devices");
        physical = physicals[selected];
        VkPhysicalDeviceIDProperties identity{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
        VkPhysicalDeviceProperties2 description{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        description.pNext = &identity;
        VkPhysicalDeviceAccelerationStructurePropertiesKHR acceleration_limits{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR};
        if (ray_query) identity.pNext=&acceleration_limits;
        vkGetPhysicalDeviceProperties2(physical, &description);
        if (shared_input) require(std::memcmp(identity.deviceUUID,shared_uuid,VK_UUID_SIZE)==0,
                                  "resident Vulkan device UUID differs from CUDA producer UUID");
        const auto& p = description.properties;
        if (resident) probe_msaa(*this,p.limits);
        VkPhysicalDeviceFeatures available_features{};
        vkGetPhysicalDeviceFeatures(physical,&available_features);
        const std::array<const char*,3> ray_extensions{VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
            VK_KHR_RAY_QUERY_EXTENSION_NAME,VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME};
        VkPhysicalDeviceBufferDeviceAddressFeatures address_features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES};
        VkPhysicalDeviceAccelerationStructureFeaturesKHR acceleration_features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
        VkPhysicalDeviceRayQueryFeaturesKHR query_features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};
        if (ray_query) {
            require(p.apiVersion>=VK_API_VERSION_1_2,"ray-query shadows require Vulkan 1.2");
            uint32_t extension_count=0;
            check(vkEnumerateDeviceExtensionProperties(physical,nullptr,&extension_count,nullptr),"ray-query extension count");
            require(extension_count<=16384,"invalid device extension count");
            std::vector<VkExtensionProperties> extensions(extension_count);
            check(vkEnumerateDeviceExtensionProperties(physical,nullptr,&extension_count,extensions.data()),"ray-query extensions");
            for (const char* name:ray_extensions)
                require(std::any_of(extensions.begin(),extensions.end(),[&](const auto& item){return std::strcmp(item.extensionName,name)==0;}),
                    "required ray-query device extension unavailable; no software fallback");
            VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
            features.pNext=&address_features; address_features.pNext=&acceleration_features; acceleration_features.pNext=&query_features;
            vkGetPhysicalDeviceFeatures2(physical,&features);
            require(address_features.bufferDeviceAddress && acceleration_features.accelerationStructure && query_features.rayQuery,
                "required ray-query/accelerationStructure/bufferDeviceAddress feature unavailable");
            require(acceleration_limits.maxGeometryCount>=1 && acceleration_limits.maxPerStageDescriptorAccelerationStructures>=1 &&
                    acceleration_limits.maxDescriptorSetAccelerationStructures>=1,"ray-query acceleration descriptor capacity unavailable");
            VkFormatProperties vertex_format{};
            vkGetPhysicalDeviceFormatProperties(physical,VK_FORMAT_R32G32B32_SFLOAT,&vertex_format);
            require((vertex_format.bufferFeatures & VK_FORMAT_FEATURE_ACCELERATION_STRUCTURE_VERTEX_BUFFER_BIT_KHR)!=0,
                "ray-query float3 vertex format unavailable");
            ray_scratch_alignment=acceleration_limits.minAccelerationStructureScratchOffsetAlignment;
            require(ray_scratch_alignment && !(ray_scratch_alignment&(ray_scratch_alignment-1)),"ray-query scratch alignment unavailable");
            // Enable only the three required features, not optional query results.
            address_features.bufferDeviceAddressCaptureReplay=VK_FALSE; address_features.bufferDeviceAddressMultiDevice=VK_FALSE;
            acceleration_features.accelerationStructureCaptureReplay=VK_FALSE;
            acceleration_features.accelerationStructureIndirectBuild=VK_FALSE;
            acceleration_features.accelerationStructureHostCommands=VK_FALSE;
            acceleration_features.descriptorBindingAccelerationStructureUpdateAfterBind=VK_FALSE;
        }
        require(p.apiVersion >= VK_API_VERSION_1_1, "Vulkan 1.1 device is required");
        require(p.deviceType != VK_PHYSICAL_DEVICE_TYPE_CPU, "CPU Vulkan devices are not a GPU fallback");
        require(p.limits.maxColorAttachments >= 4 && p.limits.maxFragmentOutputAttachments >= 4,
                "four color attachments are required");
        require(p.limits.maxUniformBufferRange >= sizeof(PrParams), "uniform capacity unavailable");
        properties.abi_version = 2; properties.device_index = selected;
        properties.api_version = p.apiVersion; properties.driver_version = p.driverVersion;
        properties.vendor_id = p.vendorID; properties.device_id = p.deviceID;
        properties.device_type = p.deviceType;
        properties.max_width = std::min({p.limits.maxFramebufferWidth, p.limits.maxImageDimension2D,
                                         p.limits.maxViewportDimensions[0]});
        properties.max_height = std::min({p.limits.maxFramebufferHeight, p.limits.maxImageDimension2D,
                                          p.limits.maxViewportDimensions[1]});
        properties.max_vertices = std::min(MAX_VERTICES,
            static_cast<uint32_t>(p.limits.maxStorageBufferRange/sizeof(PrVertex)));
        if (resident) {
            require(p.limits.maxPerStageDescriptorStorageBuffers >= 8 &&
                    p.limits.maxDescriptorSetStorageBuffers >= 8,
                    "resident rendering requires eight fragment storage buffers");
            require(p.limits.maxPerStageDescriptorUniformBuffers >= 3 &&
                    p.limits.maxDescriptorSetUniformBuffers >= 3, "resident uniform capacity unavailable");
            require(p.limits.maxPerStageDescriptorSamplers>=2 && p.limits.maxDescriptorSetSamplers>=2,
                    "resident texture sampler capacity unavailable");
            storage_buffer_limit=p.limits.maxStorageBufferRange;
            VkFormatProperties sampled{};
            vkGetPhysicalDeviceFormatProperties(physical,VK_FORMAT_R8G8B8A8_SRGB,&sampled);
            VkFormatFeatureFlags features=VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT |
                VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT | VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
            require((sampled.optimalTilingFeatures & features)==features,"resident sRGB linear texture format unavailable");
            VkImageFormatProperties texture_limits{};
            check(vkGetPhysicalDeviceImageFormatProperties(physical,VK_FORMAT_R8G8B8A8_SRGB,VK_IMAGE_TYPE_2D,
                VK_IMAGE_TILING_OPTIMAL,VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,0,&texture_limits),
                "resident texture image properties");
            require(texture_limits.maxExtent.width>2 && texture_limits.maxExtent.height>2,
                    "resident texture guard extent unavailable");
            auto& texture=texture_info;
            texture.max_textures=std::min({64u,p.limits.maxImageArrayLayers,texture_limits.maxArrayLayers});
            texture.max_layers=texture.max_textures;
            texture.max_width=std::min({2048u,p.limits.maxImageDimension2D-2,texture_limits.maxExtent.width-2});
            texture.max_height=std::min({2048u,p.limits.maxImageDimension2D-2,texture_limits.maxExtent.height-2});
            texture.max_array_bytes=std::min<uint64_t>(128ull*1024*1024,texture_limits.maxResourceSize);
            texture.sampler_filter=VK_FILTER_LINEAR; texture.mip_levels=1; texture.format=VK_FORMAT_R8G8B8A8_SRGB;
            if (normal_info.shader_compatible) {
                require(p.limits.maxPerStageDescriptorStorageBuffers>=9 && p.limits.maxDescriptorSetStorageBuffers>=9 &&
                        p.limits.maxPerStageDescriptorSamplers>=3 && p.limits.maxDescriptorSetSamplers>=3,
                        "normal-map shader descriptor capacity unavailable");
                VkFormatProperties normal_format{}; VkImageFormatProperties normal_limits{};
                vkGetPhysicalDeviceFormatProperties(physical,VK_FORMAT_R8G8B8A8_UNORM,&normal_format);
                auto normal_result=vkGetPhysicalDeviceImageFormatProperties(physical,VK_FORMAT_R8G8B8A8_UNORM,VK_IMAGE_TYPE_2D,
                    VK_IMAGE_TILING_OPTIMAL,VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,0,&normal_limits);
                require(normal_result==VK_SUCCESS && (normal_format.optimalTilingFeatures&features)==features &&
                        normal_limits.maxExtent.width>2 && normal_limits.maxExtent.height>2,"normal-map linear UNORM format unavailable");
                normal_info.max_textures=std::min({64u,p.limits.maxImageArrayLayers,normal_limits.maxArrayLayers});
                normal_info.max_layers=normal_info.max_textures;
                normal_info.max_width=std::min({2048u,p.limits.maxImageDimension2D-2,normal_limits.maxExtent.width-2});
                normal_info.max_height=std::min({2048u,p.limits.maxImageDimension2D-2,normal_limits.maxExtent.height-2});
                normal_info.max_array_bytes=std::min<uint64_t>(128ull*1024*1024,normal_limits.maxResourceSize);
                normal_info.sampler_filter=VK_FILTER_LINEAR; normal_info.mip_levels=1; normal_info.format=VK_FORMAT_R8G8B8A8_UNORM;
            }
            auto& environment=environment_info;
            const auto fragment_interface=spirv_interface(frag,frag_bytes,4);
            environment.shader_compatible=fragment_interface.valid && fragment_interface.main &&
                (fragment_interface.bindings & ((1ull<<11)|(1ull<<12)))==((1ull<<11)|(1ull<<12));
            environment.axis_transform=1; environment.diffuse_basis=1;
            environment.cube_layers=6; environment.sh_coefficients=9;
            environment.format=VK_FORMAT_R8G8B8A8_SRGB;
            VkFormatProperties hdr_format{};
            vkGetPhysicalDeviceFormatProperties(physical,VK_FORMAT_R32G32B32A32_SFLOAT,&hdr_format);
            VkImageFormatProperties hdr_limits{};
            VkResult hdr_result=vkGetPhysicalDeviceImageFormatProperties(physical,VK_FORMAT_R32G32B32A32_SFLOAT,
                VK_IMAGE_TYPE_2D,VK_IMAGE_TILING_OPTIMAL,VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT,&hdr_limits);
            environment.supported=hdr_result==VK_SUCCESS && available_features.independentBlend &&
                (hdr_format.optimalTilingFeatures & features)==features && hdr_limits.maxArrayLayers>=6;
            if (environment.supported) {
                environment.max_side=std::min({1024u,p.limits.maxImageDimensionCube,hdr_limits.maxExtent.width,hdr_limits.maxExtent.height});
                environment.max_mip_levels=std::min(11u,hdr_limits.maxMipLevels);
                environment.max_rgba_bytes=std::min<uint64_t>(128ull*1024*1024,hdr_limits.maxResourceSize);
            }
            auto& r = resident_properties;
            r.resident_abi_version = 1;
            r.max_vertices = std::min(1048576u, p.limits.maxStorageBufferRange / 32u);
            r.max_indices = std::min(1572864u, p.limits.maxStorageBufferRange / 4u);
            r.max_meshes = std::min(1024u, p.limits.maxStorageBufferRange / 32u);
            r.max_instances = std::min(1024u, p.limits.maxStorageBufferRange / 128u);
            r.max_bvh_nodes = std::min(262144u, p.limits.maxStorageBufferRange / 48u);
            r.max_instanced_triangles = 1048576u;
            properties.max_vertices = r.max_vertices;
            if (ray_query) {
                ray_info.enabled=1; ray_info.max_meshes=r.max_meshes;
                ray_info.max_instances=static_cast<uint32_t>(std::min<uint64_t>(r.max_instances,acceleration_limits.maxInstanceCount));
                ray_info.max_triangles=static_cast<uint32_t>(std::min<uint64_t>(r.max_indices/3,acceleration_limits.maxPrimitiveCount));
                ray_info.max_acceleration_bytes=512ull*1024*1024;
                ray_info.build_flags=VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
                ray_info.update_flags=VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
                ray_info.trace_flags=6; // SPIR-V NoOpaque(2) | TerminateOnFirstHit(4)
            }
        }
        std::memcpy(properties.device_uuid, identity.deviceUUID, 16);
        telemetry.device_luid_valid = identity.deviceLUIDValid ? 1u : 0u;
        telemetry.device_node_mask = identity.deviceNodeMask;
        if (identity.deviceLUIDValid) std::memcpy(telemetry.device_luid, identity.deviceLUID, 8);
        std::memcpy(properties.device_name, p.deviceName, sizeof(properties.device_name));
        for (size_t i=0; i<FORMATS.size(); ++i) {
            VkFormatProperties format{}; vkGetPhysicalDeviceFormatProperties(physical, attachment_format(i), &format);
            VkFormatFeatureFlags needed = i == 4 ? VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT :
                (VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT | VK_FORMAT_FEATURE_TRANSFER_SRC_BIT);
            require((format.optimalTilingFeatures & needed) == needed, "required attachment format unavailable");
        }
        if(progressive_info.shader_compatible) {
            require(p.limits.maxPerStageDescriptorStorageImages>=3 && p.limits.maxDescriptorSetStorageImages>=3,
                    "progressive needs three compute storage images");
            for(auto fmt:{VK_FORMAT_R32G32B32A32_SFLOAT,VK_FORMAT_R8G8B8A8_UNORM}) {
                VkFormatProperties fp{};vkGetPhysicalDeviceFormatProperties(physical,fmt,&fp);
                const auto needed=VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT|VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT|VK_FORMAT_FEATURE_TRANSFER_SRC_BIT;
                require((fp.optimalTilingFeatures&needed)==needed,"progressive storage/color format unsupported");
            }
        }
        vkGetPhysicalDeviceMemoryProperties(physical, &memory_properties);
        vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, nullptr);
        std::vector<VkQueueFamilyProperties> queues(count);
        vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, queues.data());
        queue_family = count;
        VkQueueFlags needed_queue=VK_QUEUE_GRAPHICS_BIT;
        if (shared_input && ray_query) needed_queue|=VK_QUEUE_COMPUTE_BIT;
        for (uint32_t i=0; i<count; ++i) if (queues[i].queueCount &&
            (queues[i].queueFlags & needed_queue)==needed_queue) { queue_family=i; break; }
        require(queue_family < count, "required graphics/compute queue unavailable");
        require(!progressive_info.shader_compatible || (queues[queue_family].queueFlags&VK_QUEUE_COMPUTE_BIT),"progressive needs a graphics+compute queue");
        if (resident) {
            profile.timestamp_valid_bits = queues[queue_family].timestampValidBits;
            profile.timestamp_supported = profile.timestamp_valid_bits != 0 ? 1u : 0u;
            profile.timestamp_period_ns = profile.timestamp_supported ? p.limits.timestampPeriod : 0.0;
        }
        float priority = 1.0f;
        VkDeviceQueueCreateInfo queue_info{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        queue_info.queueFamilyIndex = queue_family; queue_info.queueCount = 1; queue_info.pQueuePriorities = &priority;
        VkDeviceCreateInfo device_info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        device_info.queueCreateInfoCount = 1; device_info.pQueueCreateInfos = &queue_info;
        VkPhysicalDeviceFeatures enabled_features{};
        // The sky pipeline disables writes on the three sensor attachments.
        // Optional support is reported false when independentBlend is absent.
        enabled_features.independentBlend=resident ? available_features.independentBlend : VK_FALSE;
        device_info.pEnabledFeatures=&enabled_features;
        std::vector<const char*> extensions;
#ifdef _WIN32
        const char* shared_extensions[]={VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME,VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME};
        if (shared_input) {
            VkPhysicalDeviceExternalBufferInfo bi{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_BUFFER_INFO};
            bi.usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT|VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
            bi.handleType=VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
            VkExternalBufferProperties bp{VK_STRUCTURE_TYPE_EXTERNAL_BUFFER_PROPERTIES};
            vkGetPhysicalDeviceExternalBufferProperties(physical,&bi,&bp);
            const auto mf=VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT | VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT;
            require((bp.externalMemoryProperties.externalMemoryFeatures&mf)==mf &&
                    (bp.externalMemoryProperties.compatibleHandleTypes&bi.handleType),"resident Win32 external SSBO unavailable");
            VkPhysicalDeviceExternalSemaphoreInfo si{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_SEMAPHORE_INFO};
            si.handleType=VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;
            VkExternalSemaphoreProperties sp{VK_STRUCTURE_TYPE_EXTERNAL_SEMAPHORE_PROPERTIES};
            vkGetPhysicalDeviceExternalSemaphoreProperties(physical,&si,&sp);
            const auto sf=VK_EXTERNAL_SEMAPHORE_FEATURE_EXPORTABLE_BIT|VK_EXTERNAL_SEMAPHORE_FEATURE_IMPORTABLE_BIT;
            require((sp.externalSemaphoreFeatures&sf)==sf && (sp.compatibleHandleTypes&si.handleType),
                    "resident Win32 external binary semaphores unavailable");
            extensions.assign(shared_extensions,shared_extensions+2);
        }
#endif
        if (ray_query) {
            device_info.pNext=&address_features;
            extensions.insert(extensions.end(),ray_extensions.begin(),ray_extensions.end());
        }
        device_info.enabledExtensionCount=static_cast<uint32_t>(extensions.size());
        device_info.ppEnabledExtensionNames=extensions.data();
        acquire(device, "vkCreateDevice", [&](auto* output) { return vkCreateDevice(physical, &device_info, nullptr, output); });
        vkGetDeviceQueue(device, queue_family, 0, &queue);
        if (ray_query) {
            ray.create=reinterpret_cast<PFN_vkCreateAccelerationStructureKHR>(vkGetDeviceProcAddr(device,"vkCreateAccelerationStructureKHR"));
            ray.destroy=reinterpret_cast<PFN_vkDestroyAccelerationStructureKHR>(vkGetDeviceProcAddr(device,"vkDestroyAccelerationStructureKHR"));
            ray.sizes=reinterpret_cast<PFN_vkGetAccelerationStructureBuildSizesKHR>(vkGetDeviceProcAddr(device,"vkGetAccelerationStructureBuildSizesKHR"));
            ray.build=reinterpret_cast<PFN_vkCmdBuildAccelerationStructuresKHR>(vkGetDeviceProcAddr(device,"vkCmdBuildAccelerationStructuresKHR"));
            ray.address=reinterpret_cast<PFN_vkGetAccelerationStructureDeviceAddressKHR>(vkGetDeviceProcAddr(device,"vkGetAccelerationStructureDeviceAddressKHR"));
            ray.buffer_address=reinterpret_cast<PFN_vkGetBufferDeviceAddress>(vkGetDeviceProcAddr(device,"vkGetBufferDeviceAddress"));
            require(ray.create && ray.destroy && ray.sizes && ray.build && ray.address && ray.buffer_address,
                "ray-query device entrypoints unavailable");
        }
        VkCommandPoolCreateInfo pool{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pool.queueFamilyIndex = queue_family; pool.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT |
            (resident ? VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT : 0u);
        acquire(command_pool, "vkCreateCommandPool", [&](auto* output) { return vkCreateCommandPool(device, &pool, nullptr, output); });
        std::array<VkDescriptorSetLayoutBinding, 16> bindings{};
        bindings[0] = {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
        bindings[1] = {1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1,
                       VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
        VkDescriptorSetLayoutCreateInfo layout{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        if (resident) {
            for (uint32_t i = 2; i < 7; ++i) {
                VkShaderStageFlags stages = VK_SHADER_STAGE_FRAGMENT_BIT;
                if (i == 2 || i == 3) stages |= VK_SHADER_STAGE_VERTEX_BIT;
                bindings[i] = {i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, stages, nullptr};
            }
        }
        if (resident) {
            bindings[7] = {7, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
            bindings[8] = {8, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
            bindings[9] = {9, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
            bindings[10] = {10, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                            VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
        }
        if (resident) {
            bindings[11]={11,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1,VK_SHADER_STAGE_FRAGMENT_BIT,nullptr};
            bindings[12]={12,VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,1,VK_SHADER_STAGE_FRAGMENT_BIT,nullptr};
        }
        if (ray_query) bindings[13]={13,VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,1,VK_SHADER_STAGE_FRAGMENT_BIT,nullptr};
        uint32_t binding_count=ray_query ? 14u : (resident ? 13u : 2u);
        if (normal_info.shader_compatible) {
            bindings[binding_count++]={14,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1,VK_SHADER_STAGE_FRAGMENT_BIT,nullptr};
            bindings[binding_count++]={15,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,nullptr};
        }
        layout.bindingCount=binding_count; layout.pBindings=bindings.data();
        acquire(descriptor_layout, "descriptor layout", [&](auto* output) { return vkCreateDescriptorSetLayout(device, &layout, nullptr, output); });
        VkPipelineLayoutCreateInfo pipeline_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pipeline_info.setLayoutCount = 1; pipeline_info.pSetLayouts = &descriptor_layout;
        acquire(pipeline_layout, "pipeline layout", [&](auto* output) { return vkCreatePipelineLayout(device, &pipeline_info, nullptr, output); });
        std::array<VkAttachmentDescription, 5> attachments{};
        std::array<VkAttachmentReference, 4> colors{};
        for (uint32_t i=0; i<5; ++i) {
            auto& value = attachments[i]; value.format=attachment_format(i); value.samples=VK_SAMPLE_COUNT_1_BIT;
            value.loadOp=VK_ATTACHMENT_LOAD_OP_CLEAR; value.storeOp=VK_ATTACHMENT_STORE_OP_STORE;
            value.stencilLoadOp=VK_ATTACHMENT_LOAD_OP_DONT_CARE; value.stencilStoreOp=VK_ATTACHMENT_STORE_OP_DONT_CARE;
            value.initialLayout=VK_IMAGE_LAYOUT_UNDEFINED;
            value.finalLayout=i==4 ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL : VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            if (i<4) colors[i]={i,VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};

        }
        VkAttachmentReference depth{4,VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
        VkSubpassDescription subpass{}; subpass.pipelineBindPoint=VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount=4; subpass.pColorAttachments=colors.data(); subpass.pDepthStencilAttachment=&depth;
        std::array<VkSubpassDependency,2> dependencies{};
        dependencies[0].srcSubpass=VK_SUBPASS_EXTERNAL; dependencies[0].dstSubpass=0;
        dependencies[0].srcStageMask=VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        dependencies[0].dstStageMask=VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        dependencies[0].dstAccessMask=VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        dependencies[1].srcSubpass=0; dependencies[1].dstSubpass=VK_SUBPASS_EXTERNAL;
        dependencies[1].srcStageMask=VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependencies[1].dstStageMask=VK_PIPELINE_STAGE_TRANSFER_BIT;
        dependencies[1].srcAccessMask=VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        dependencies[1].dstAccessMask=VK_ACCESS_TRANSFER_READ_BIT;
        VkRenderPassCreateInfo pass{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
        pass.attachmentCount=5; pass.pAttachments=attachments.data(); pass.subpassCount=1; pass.pSubpasses=&subpass;
        pass.dependencyCount=2; pass.pDependencies=dependencies.data();
        acquire(render_pass, "vkCreateRenderPass", [&](auto* output) { return vkCreateRenderPass(device,&pass,nullptr,output); });
        VkShaderModuleCreateInfo shader{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        shader.codeSize=vert_bytes; shader.pCode=vert;
        acquire(vertex_shader, "vertex shader", [&](auto* output) { return vkCreateShaderModule(device,&shader,nullptr,output); });
        shader.codeSize=frag_bytes; shader.pCode=frag;
        acquire(fragment_shader, "fragment shader", [&](auto* output) { return vkCreateShaderModule(device,&shader,nullptr,output); });
        std::array<VkPipelineShaderStageCreateInfo,2> stages{};
        for (auto& stage:stages) { stage.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO; stage.pName="main"; }
        stages[0].stage=VK_SHADER_STAGE_VERTEX_BIT; stages[0].module=vertex_shader;
        stages[1].stage=VK_SHADER_STAGE_FRAGMENT_BIT; stages[1].module=fragment_shader;
        VkVertexInputBindingDescription vertex_binding{0,
            resident ? static_cast<uint32_t>(sizeof(PrResidentVertex)) : static_cast<uint32_t>(sizeof(PrVertex)),
            VK_VERTEX_INPUT_RATE_VERTEX};
        std::array<VkVertexInputAttributeDescription,4> attributes{};
        for (uint32_t i=0;i<4;++i) attributes[i]={i,0,
            i==3?VK_FORMAT_R32G32B32A32_UINT:VK_FORMAT_R32G32B32A32_SFLOAT,i*16u};
        VkPipelineVertexInputStateCreateInfo vertex_input{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        vertex_input.vertexBindingDescriptionCount=1; vertex_input.pVertexBindingDescriptions=&vertex_binding;
        vertex_input.vertexAttributeDescriptionCount=resident ? 2u : 4u;
        vertex_input.pVertexAttributeDescriptions=attributes.data();
        VkPipelineInputAssemblyStateCreateInfo assembly{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
        assembly.topology=VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        VkPipelineViewportStateCreateInfo viewport{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
        viewport.viewportCount=1; viewport.scissorCount=1;
        VkPipelineRasterizationStateCreateInfo raster{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        raster.polygonMode=VK_POLYGON_MODE_FILL; raster.cullMode=VK_CULL_MODE_NONE;
        raster.frontFace=VK_FRONT_FACE_COUNTER_CLOCKWISE; raster.lineWidth=1.0f;
        VkPipelineMultisampleStateCreateInfo sample{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
        sample.rasterizationSamples=VK_SAMPLE_COUNT_1_BIT;
        VkPipelineDepthStencilStateCreateInfo depth_state{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        depth_state.depthTestEnable=VK_TRUE; depth_state.depthWriteEnable=VK_TRUE; depth_state.depthCompareOp=VK_COMPARE_OP_LESS;
        std::array<VkPipelineColorBlendAttachmentState,4> blends{};
        for(auto& blend:blends) blend.colorWriteMask=VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
            VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        VkPipelineColorBlendStateCreateInfo blending{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        blending.attachmentCount=4; blending.pAttachments=blends.data();
        std::array<VkDynamicState,2> dynamic_states={VK_DYNAMIC_STATE_VIEWPORT,VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
        dynamic.dynamicStateCount=2; dynamic.pDynamicStates=dynamic_states.data();
        VkGraphicsPipelineCreateInfo pipeline_create{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        pipeline_create.stageCount=2; pipeline_create.pStages=stages.data(); pipeline_create.pVertexInputState=&vertex_input;
        pipeline_create.pInputAssemblyState=&assembly; pipeline_create.pViewportState=&viewport;
        pipeline_create.pRasterizationState=&raster; pipeline_create.pMultisampleState=&sample;
        pipeline_create.pDepthStencilState=&depth_state; pipeline_create.pColorBlendState=&blending;
        pipeline_create.pDynamicState=&dynamic; pipeline_create.layout=pipeline_layout;
        pipeline_create.renderPass=render_pass;
        // Pipeline creation can return valid handles even on partial failure.
        VkPipeline temporary_pipeline = VK_NULL_HANDLE;
        VkResult pipeline_result = vkCreateGraphicsPipelines(
            device,VK_NULL_HANDLE,1,&pipeline_create,nullptr,&temporary_pipeline);
        if (temporary_pipeline) { pipeline=temporary_pipeline; ++live_resources; }
        check(pipeline_result,"graphics pipeline");
        if(errors.load()) throw std::runtime_error("Vulkan validation failed during initialization: "+validation_message);
    }
};

struct MsaaTarget;
struct FrameResources {
    Context& c;
    Buffer vertices{}, uniform{}, readback{};
    void* readback_mapping{};
    size_t readback_capacity{};
    std::array<Image,6> images{};
    VkFramebuffer framebuffer{},guide_framebuffer{};
    bool guide_initialized=false;
    VkDescriptorPool descriptors{};
    VkDescriptorSet scene_set{}; // allocated from descriptors
    uint32_t width{},height{};
    std::shared_ptr<MsaaTarget> msaa_target;
    VkQueryPool timestamps{};
    VkCommandBuffer commands{};
    VkFence fence{};
    bool submitted=false;
    explicit FrameResources(Context& context):c(context){ ++live_frames; }
    ~FrameResources() {
        if(submitted) vkQueueWaitIdle(c.queue);
        msaa_target.reset(); // encode framebuffer references images[0]
        release(fence, [&](auto owned) { vkDestroyFence(c.device,owned,nullptr); });
        release(commands, [&](auto h) { vkFreeCommandBuffers(c.device,c.command_pool,1,&h); });
        release(descriptors, [&](auto owned) { vkDestroyDescriptorPool(c.device,owned,nullptr); });
        release(timestamps, [&](auto owned) { vkDestroyQueryPool(c.device,owned,nullptr); });
        release(guide_framebuffer, [&](auto owned) { vkDestroyFramebuffer(c.device,owned,nullptr); });
        release(framebuffer, [&](auto owned) { vkDestroyFramebuffer(c.device,owned,nullptr); });
        for(auto& image:images) {
            release(image.view, [&](auto owned) { vkDestroyImageView(c.device,owned,nullptr); });
            release(image.handle, [&](auto owned) { vkDestroyImage(c.device,owned,nullptr); });
            c.free_memory(image.memory, image.allocation);
        }
        if (readback_mapping) {
            vkUnmapMemory(c.device,readback.memory); readback_mapping=nullptr;
            ++c.readback_info.unmap_calls;
        }
        for(auto* buffer:{&vertices,&uniform,&readback}) {
            release(buffer->handle, [&](auto owned) { vkDestroyBuffer(c.device,owned,nullptr); });
            c.free_memory(buffer->memory, buffer->allocation);
        }
        --live_frames;
    }
};

void render(Context& c,const PrVertex* vertices,uint32_t vertex_count,const PrParams& params,
            uint32_t width,uint32_t height,uint8_t* output,size_t output_bytes) {
    require(!c.resident,"use pr_resident_render for a resident context");
    require(!c.poisoned,"Vulkan session failed; close and open an explicit new session");
    require(width && height && width<=c.properties.max_width && height<=c.properties.max_height,
            "capture extent exceeds device capability");
    require(static_cast<uint64_t>(width)*height<=4096ull*4096ull,"native capture pixel budget exceeded");
    size_t pixels=static_cast<size_t>(width)*height;
    require(output && output_bytes==pixels*13,"output byte extent mismatch");
    require(vertex_count%3==0 && vertex_count<=c.properties.max_vertices,"native triangle budget exceeded");
    require(vertex_count==0 || vertices,"null vertex data");
    require(params.counts[0]<=4 && params.counts[1]==vertex_count/3 && params.counts[2]<=1,
            "uniform scene counts mismatch");
    require(params.clipping[0]==width && params.clipping[1]==height && params.clipping[2]>0 &&
            params.clipping[3]>params.clipping[2],"invalid clipping calibration");
    const float* finite_blocks[]={params.view,params.intrinsics,params.clipping,params.camera,
                                 &params.light_directions[0][0],&params.light_colors[0][0],params.options};
    const size_t block_sizes[]={16,4,4,4,16,16,4};
    for(size_t b=0;b<7;++b) for(size_t i=0;i<block_sizes[b];++i)
        require(std::isfinite(finite_blocks[b][i]),"uniform FP32 conversion is nonfinite");
    require(params.intrinsics[0]>0 && params.intrinsics[1]>0 && params.options[0]>=0 && params.options[1]>0,
            "invalid projection or exposure");
    for(uint32_t i=0;i<vertex_count;++i) {
        const auto& v=vertices[i];
        for(uint32_t k=0;k<4;++k) require(std::isfinite(v.position[k]) &&
            std::isfinite(v.normal_metallic[k]) && std::isfinite(v.color_roughness[k]),
            "vertex FP32 conversion is nonfinite");
        double normal_length=std::hypot(static_cast<double>(v.normal_metallic[0]),
            static_cast<double>(v.normal_metallic[1]),static_cast<double>(v.normal_metallic[2]));
        require(normal_length>=0.999 && normal_length<=1.001,"native vertex normal must be unit length");
        require(v.ids[0]>0 && v.ids[1]==i/3,"vertex identity mismatch");
    }
    uint64_t errors_before=c.errors.load();
    FrameResources f(c);
    size_t vertex_bytes=std::max<size_t>(sizeof(PrVertex),vertex_count*sizeof(PrVertex));
    c.make_buffer(f.vertices,vertex_bytes,VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    PrVertex empty{}; c.upload(f.vertices,vertex_count?static_cast<const void*>(vertices):&empty,vertex_bytes);
    c.make_buffer(f.uniform,sizeof(params),VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT); c.upload(f.uniform,&params,sizeof(params));
    c.make_buffer(f.readback,output_bytes,VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    std::array<VkImageView,5> views{};
    for(uint32_t i=0;i<5;++i) { c.make_image(f.images[i],width,height,FORMATS[i],i==4); views[i]=f.images[i].view; }
    VkFramebufferCreateInfo fb{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
    fb.renderPass=c.render_pass; fb.attachmentCount=5; fb.pAttachments=views.data(); fb.width=width; fb.height=height; fb.layers=1;
    acquire(f.framebuffer, "vkCreateFramebuffer", [&](auto* output) { return vkCreateFramebuffer(c.device,&fb,nullptr,output); });
    std::array<VkDescriptorPoolSize,2> sizes={VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1},
                                           VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,1}};
    VkDescriptorPoolCreateInfo pool{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pool.maxSets=1; pool.poolSizeCount=2; pool.pPoolSizes=sizes.data();
    acquire(f.descriptors, "descriptor pool", [&](auto* output) { return vkCreateDescriptorPool(c.device,&pool,nullptr,output); });
    VkDescriptorSetAllocateInfo alloc{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    alloc.descriptorPool=f.descriptors; alloc.descriptorSetCount=1; alloc.pSetLayouts=&c.descriptor_layout;
    VkDescriptorSet set{}; check(vkAllocateDescriptorSets(c.device,&alloc,&set),"descriptor allocation");
    std::array<VkDescriptorBufferInfo,2> buffers={VkDescriptorBufferInfo{f.vertices.handle,0,vertex_bytes},
                                               VkDescriptorBufferInfo{f.uniform.handle,0,sizeof(params)}};
    std::array<VkWriteDescriptorSet,2> writes{};
    for(uint32_t i=0;i<2;++i) {
        writes[i].sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; writes[i].dstSet=set; writes[i].dstBinding=i;
        writes[i].descriptorCount=1; writes[i].descriptorType=sizes[i].type; writes[i].pBufferInfo=&buffers[i];
    }
    vkUpdateDescriptorSets(c.device,2,writes.data(),0,nullptr);
    VkCommandBufferAllocateInfo commands{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    commands.commandPool=c.command_pool; commands.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY; commands.commandBufferCount=1;
    acquire(f.commands, "command allocation", [&](auto* output) { return vkAllocateCommandBuffers(c.device,&commands,output); });
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; begin.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    check(vkBeginCommandBuffer(f.commands,&begin),"vkBeginCommandBuffer");
    std::array<VkClearValue,5> clear{};
    clear[0].color.float32[3]=1.0f; clear[1].color.float32[0]=std::numeric_limits<float>::infinity();
    clear[4].depthStencil.depth=1.0f;
    VkRenderPassBeginInfo pass{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    pass.renderPass=c.render_pass; pass.framebuffer=f.framebuffer; pass.renderArea.extent={width,height};
    pass.clearValueCount=5; pass.pClearValues=clear.data();
    vkCmdBeginRenderPass(f.commands,&pass,VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(f.commands,VK_PIPELINE_BIND_POINT_GRAPHICS,c.pipeline);
    VkViewport viewport{0,0,static_cast<float>(width),static_cast<float>(height),0,1};
    VkRect2D scissor{{0,0},{width,height}};
    vkCmdSetViewport(f.commands,0,1,&viewport); vkCmdSetScissor(f.commands,0,1,&scissor);
    VkDeviceSize offset=0; vkCmdBindVertexBuffers(f.commands,0,1,&f.vertices.handle,&offset);
    vkCmdBindDescriptorSets(f.commands,VK_PIPELINE_BIND_POINT_GRAPHICS,c.pipeline_layout,0,1,&set,0,nullptr);
    if(vertex_count) vkCmdDraw(f.commands,vertex_count,1,0,0);
    vkCmdEndRenderPass(f.commands);
    const size_t bpp[]={4,4,4,1}; offset=0;
    for(uint32_t i=0;i<4;++i) {
        VkBufferImageCopy region{}; region.bufferOffset=offset;
        region.imageSubresource.aspectMask=VK_IMAGE_ASPECT_COLOR_BIT; region.imageSubresource.layerCount=1;
        region.imageExtent={width,height,1};
        vkCmdCopyImageToBuffer(f.commands,f.images[i].handle,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                              f.readback.handle,1,&region);
        offset+=pixels*bpp[i];
    }
    VkBufferMemoryBarrier host{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    host.srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT; host.dstAccessMask=VK_ACCESS_HOST_READ_BIT;
    host.srcQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED; host.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;
    host.buffer=f.readback.handle; host.size=VK_WHOLE_SIZE;
    vkCmdPipelineBarrier(f.commands,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_HOST_BIT,0,
                         0,nullptr,1,&host,0,nullptr);
    check(vkEndCommandBuffer(f.commands),"vkEndCommandBuffer");
    VkFenceCreateInfo fence{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    acquire(f.fence, "vkCreateFence", [&](auto* output) { return vkCreateFence(c.device,&fence,nullptr,output); });
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO}; submit.commandBufferCount=1; submit.pCommandBuffers=&f.commands;
    VkResult submitted=vkQueueSubmit(c.queue,1,&submit,f.fence);
    // DEVICE_LOST does not establish that no work is pending. The frame owner
    // waits for the queue before releasing submitted resources on either path.
    f.submitted = submitted==VK_SUCCESS || submitted==VK_ERROR_DEVICE_LOST;
    if(submitted!=VK_SUCCESS) { c.poisoned=true; check(submitted,"vkQueueSubmit"); }
    VkResult completed=vkWaitForFences(c.device,1,&f.fence,VK_TRUE,UINT64_MAX);
    if(completed!=VK_SUCCESS) { c.poisoned=true; check(completed,"vkWaitForFences"); }
    f.submitted=false;
    if(c.errors.load()!=errors_before) {
        c.poisoned=true;
        throw std::runtime_error("Vulkan validation failed during capture: "+c.validation_message);
    }
    void* mapped{}; check(vkMapMemory(c.device,f.readback.memory,0,output_bytes,0,&mapped),"vkMapMemory readback");
    std::memcpy(output,mapped,output_bytes); vkUnmapMemory(c.device,f.readback.memory);
    ++c.properties.captures;
}
#include "resident.inc"
} // namespace

uint32_t pr_abi_version() { return 2; }
const char* pr_error() { return last_error.c_str(); }
void* pr_create(uint32_t selected,uint32_t validation,const uint32_t* vert,size_t vert_bytes,
                const uint32_t* frag,size_t frag_bytes) {
    try {
        last_error.clear(); auto context=std::make_unique<Context>();
        context->init(selected,validation!=0,vert,vert_bytes,frag,frag_bytes);
        return context.release();
    } catch(const std::exception& error) { last_error=error.what(); return nullptr; }
      catch(...) { last_error="unknown native creation error"; return nullptr; }
}
int pr_info(void* handle,PrInfo* output) {
    if(!handle || !output) { last_error="null native info argument"; return -1; }
    static_cast<Context*>(handle)->info(output); return 0;
}
int pr_render(void* handle,const PrVertex* vertices,uint32_t count,const PrParams* params,
              uint32_t width,uint32_t height,uint8_t* output,size_t bytes) {
    try {
        last_error.clear(); require(handle && params,"null native capture argument");
        render(*static_cast<Context*>(handle),vertices,count,*params,width,height,output,bytes);
        return 0;
    } catch(const std::exception& error) { last_error=error.what(); return -1; }
      catch(...) { last_error="unknown native capture error"; return -1; }
}
int pr_counts(PrCounts* output) {
    if(!output) { last_error="null native counts output"; return -1; }
    *output={live_contexts.load(),live_frames.load(),live_resources.load()};
    return 0;
}
int pr_destroy(void* handle,PrInfo* final_info) {
    if(!handle) { if(final_info) *final_info={}; return 0; }
    auto* context=static_cast<Context*>(handle);
    if (context->shared_input && drain_shared(*context)!=0 && context->shared_slot.phase!=SharedPhase::unused)
        return -2; // keep the handle while a live device cannot complete/return ownership
    VkResult waited=context->shutdown();
    if(final_info) context->info(final_info);
    uint64_t errors=context->errors.load();
    delete context;
    if(waited!=VK_SUCCESS) { last_error="vkDeviceWaitIdle during close: "+std::to_string(waited); return -1; }
    if(errors) { last_error="Vulkan validation reported errors before close completed"; return -1; }
    return 0;
}


uint32_t pr_resident_abi_version() { return 1; }
uint32_t pr_resident_shared_abi_version() { return 1; }
uint32_t pr_resident_shared_modes_abi_version() { return 1; }
void* pr_resident_create_shared_ray_query(uint32_t selected,uint32_t validation,const uint32_t* vert,size_t vert_bytes,
    const uint32_t* frag,size_t frag_bytes,const uint8_t* cuda_uuid16,const uint32_t* comp,size_t comp_bytes) {
    try {
        last_error.clear();
        require(cuda_uuid16 && validation<=1,"invalid shared ray-query device UUID or validation flag");
        const auto shader=spirv_interface(comp,comp_bytes,5);
        require(shader.valid && shader.main && shader.bindings==3 &&
                named_shader_marker(comp,comp_bytes,75,"PR_SHARED_INSTANCE_PACK_ABI",5),
                "shared ray-query instance compute shader interface is incompatible");
        auto c=std::make_unique<Context>();
        c->init(selected,validation!=0,vert,vert_bytes,frag,frag_bytes,true,true,cuda_uuid16);
        require(!c->reflection_info.shader_compatible && !c->indirect_info.shader_compatible &&
                !c->progressive_info.shader_compatible,"shared ray-query mode requires direct raster lighting");
        VkShaderModuleCreateInfo module{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        module.pCode=comp; module.codeSize=comp_bytes;
        acquire(c->shared_instance_shader,"shared TLAS compute module",
            [&](auto* out){return vkCreateShaderModule(c->device,&module,nullptr,out);});
        return c.release();
    } catch (const std::exception& e) { last_error=e.what(); return nullptr; }
      catch (...) { last_error="unknown shared ray-query creation error"; return nullptr; }
}
void* pr_resident_create_shared(uint32_t selected,uint32_t validation,const uint32_t* vert,size_t vert_bytes,
                                const uint32_t* frag,size_t frag_bytes,const uint8_t* cuda_uuid16) {
    try {
        last_error.clear();
        require(cuda_uuid16,"shared creation requires CUDA device UUID");
        auto c=std::make_unique<Context>();
        c->init(selected,validation!=0,vert,vert_bytes,frag,frag_bytes,true,false,cuda_uuid16);
        return c.release();
    } catch (const std::exception& e) { last_error=e.what(); return nullptr; }
      catch (...) { last_error="unknown resident shared creation error"; return nullptr; }
}
int pr_resident_shared_export(void* context,PrResidentSharedExport* output) {
    return shared_call([&]{
        require(context && output,"null shared export argument");
        *output={}; export_shared(*static_cast<Context*>(context),*output);
    });
}
int pr_resident_shared_drop_exports(void* context) {
    return shared_call([&]{
        require(context,"null shared context");
        auto& s=static_cast<Context*>(context)->shared_slot;
        require(s.exported,"shared slot has not been exported");
        drop_shared_handles(s);
    });
}
int pr_resident_shared_begin(void* context) {
    return shared_call([&]{require(context,"null shared begin context"); begin_shared(*static_cast<Context*>(context));});
}
int pr_resident_shared_end(void* context) {
    return shared_call([&]{require(context,"null shared end context"); end_shared(*static_cast<Context*>(context));});
}
uint32_t pr_resident_shared_camera_abi_version() { return 1; }
uint32_t pr_resident_shared_batch_abi_version() { return 1; }
int pr_resident_render_shared_batch(void* context,const PrResidentBatchView* views,
                                    uint32_t count,uint8_t* output,size_t output_bytes) {
    return shared_call([&]{
        require(context,"null camera batch context");
        resident_shared_batch(*static_cast<Context*>(context),views,count,output,output_bytes);
    });
}
int pr_resident_set_shared_camera_capacity(void* context,uint32_t count) {
    return shared_call([&]{
        require(context,"null shared camera context");
        auto& c=*static_cast<Context*>(context);
        require(c.shared_input && !c.resident_state && count<=64,"invalid shared camera capacity or phase");
        c.shared_camera_capacity=count;
    });
}
uint64_t pr_resident_shared_camera_d2d_bytes(void* context) {
    return context ? static_cast<Context*>(context)->shared_camera_d2d_bytes : 0;
}
int pr_resident_render_shared_camera_planes(void* context,uint32_t camera_index,const PrParams* params,
                                           uint32_t width,uint32_t height,uint32_t mask,uint8_t* output,size_t bytes) {
    return shared_call([&]{
        require(context && params,"null mounted capture argument");
        auto& c=*static_cast<Context*>(context);
        require(camera_index<c.shared_camera_capacity,"mounted camera index exceeds the declared capacity");
        resident_render(c,nullptr,c.resident_properties.instance_count,*params,width,height,output,bytes,
                        mask,true,true,camera_index);
        c.shared_camera_d2d_bytes+=19*sizeof(float);
    });
}
int pr_resident_render_shared_planes(void* context,const PrParams* params,uint32_t width,uint32_t height,
                                     uint32_t mask,uint8_t* output,size_t bytes) {
    return shared_call([&]{
        require(context && params,"null shared capture argument");
        auto& c=*static_cast<Context*>(context);
        resident_render(c,nullptr,c.resident_properties.instance_count,*params,width,height,output,bytes,mask,true,true);
    });
}
void* pr_resident_create(uint32_t selected, uint32_t validation,
                         const uint32_t* vert, size_t vert_bytes,
                         const uint32_t* frag, size_t frag_bytes) {
    try {
        last_error.clear();
        auto context = std::make_unique<Context>();
        context->init(selected, validation != 0, vert, vert_bytes, frag, frag_bytes, true);
        return context.release();
    } catch (const std::exception& error) { last_error = error.what(); return nullptr; }
      catch (...) { last_error = "unknown resident creation error"; return nullptr; }
}
int pr_resident_upload(void* handle,
                       const PrResidentVertex* vertices, uint32_t vertex_count,
                       const uint32_t* indices, uint32_t index_count,
                       const PrResidentMesh* meshes, uint32_t mesh_count,
                       const PrResidentInstance* instances, uint32_t instance_count) {
    try {
        last_error.clear(); require(handle, "null resident upload context");
        resident_upload(*static_cast<Context*>(handle), vertices, vertex_count, indices, index_count,
                        meshes, mesh_count, instances, instance_count);
        return 0;
    } catch (const std::exception& error) { last_error = error.what(); return -1; }
      catch (...) { last_error = "unknown resident upload error"; return -1; }
}
int pr_resident_render(void* handle, const float* matrices, uint32_t instance_count,
                       const PrParams* params, uint32_t width, uint32_t height,
                       uint8_t* output, size_t bytes) {
    try {
        last_error.clear(); require(handle && params, "null resident capture argument");
        resident_render(*static_cast<Context*>(handle), matrices, instance_count, *params,
                        width, height, output, bytes);
        return 0;
    } catch (const std::exception& error) { last_error = error.what(); return -1; }
      catch (...) { last_error = "unknown resident capture error"; return -1; }
}
int pr_resident_info(void* handle, PrResidentInfo* output) {
    if (!handle || !output) { last_error = "null resident info argument"; return -1; }
    auto& context = *static_cast<Context*>(handle);
    if (!context.resident) { last_error = "context does not use resident rendering"; return -1; }
    *output = context.resident_properties;
    output->poisoned = context.poisoned ? 1u : 0u;
    return 0;
}


int pr_resident_telemetry(void* handle, PrResidentTelemetry* output) {
    if (!handle || !output) { last_error = "null resident telemetry argument"; return -1; }
    auto& context = *static_cast<Context*>(handle);
    if (!context.resident) { last_error = "context does not use resident rendering"; return -1; }
    *output = context.telemetry;
    return 0;
}
int pr_resident_destroy(void* handle, PrInfo* final_info, PrResidentTelemetry* final_telemetry) {
    if (!handle) {
        if (final_info) *final_info = {};
        if (final_telemetry) { *final_telemetry = {}; final_telemetry->telemetry_abi_version = 1; }
        return 0;
    }
    auto* context = static_cast<Context*>(handle);
    if (!context->resident) { last_error = "context does not use resident rendering"; return -1; }
    if (context->shared_input && drain_shared(*context)!=0 && context->shared_slot.phase!=SharedPhase::unused)
        return -2;
    VkResult waited = context->shutdown();
    if (final_info) context->info(final_info);
    if (final_telemetry) *final_telemetry = context->telemetry;
    uint64_t errors = context->errors.load();
    delete context;
    if (waited != VK_SUCCESS) {
        last_error = "resident vkDeviceWaitIdle during close: "+std::to_string(waited);
        return -1;
    }
    if (errors) { last_error = "Vulkan validation reported errors before resident close completed"; return -1; }
    return 0;
}

int pr_resident_set_studio(void* handle, const PrStudio* value) {
    try {
        last_error.clear(); require(handle && value,"null resident studio argument");
        auto& context=*static_cast<Context*>(handle);
        require(context.resident && !context.poisoned && context.properties.captures==0,
                "resident studio is immutable after first capture");
        auto finite4=[](const float* data) {
            for (uint32_t k=0; k<4; ++k) require(std::isfinite(data[k]),"nonfinite resident studio input");
        };
        finite4(value->environment);
        for (const auto& panel:value->panels)
            for (const float* block:{panel.position,panel.half_u,panel.half_v,panel.radiance}) finite4(block);
        require(value->environment[3]==0 || value->environment[3]==1,"resident studio enabled flag must be zero or one");
        for (uint32_t k=0; k<3; ++k) require(value->environment[k]>=0,"negative studio environment");
        if (value->environment[3]==0) {
            for (float component:value->environment) require(component==0,"disabled resident studio has nonzero inputs");
            for (const auto& panel:value->panels)
                for (const float* block:{panel.position,panel.half_u,panel.half_v,panel.radiance})
                    for (uint32_t k=0; k<4; ++k) require(block[k]==0,"disabled resident studio has nonzero inputs");
        }
        for (const auto& panel:value->panels) {
            require(panel.position[3]==0 && panel.half_u[3]==0 && panel.half_v[3]==0 && panel.radiance[3]==0,
                    "resident studio padding must be zero");
            double uu=0,vv=0,uv=0;
            for (uint32_t k=0; k<3; ++k) {
                require(panel.radiance[k]>=0,"negative studio panel radiance");
                uu+=static_cast<double>(panel.half_u[k])*panel.half_u[k];
                vv+=static_cast<double>(panel.half_v[k])*panel.half_v[k];
                uv+=static_cast<double>(panel.half_u[k])*panel.half_v[k];
            }
            if (panel.radiance[0]!=0 || panel.radiance[1]!=0 || panel.radiance[2]!=0) {
                require(uu>0 && vv>0 && std::isfinite(static_cast<float>(uu*vv)) &&
                        static_cast<float>(uu*vv)>0,"resident studio emitting rectangle is degenerate");
                require(std::abs(uv)<=1e-5*std::sqrt(uu*vv),"resident studio half axes must be orthogonal");
            }
        }
        context.studio=*value;
        return 0;
    } catch (const std::exception& e) { last_error=e.what(); return -1; }
      catch (...) { last_error="unknown resident studio error"; return -1; }
}
int pr_resident_profile(void* handle, PrResidentProfile* output) {
    if (!handle || !output) { last_error="null resident profile argument"; return -1; }
    auto& context=*static_cast<Context*>(handle);
    if (!context.resident) { last_error="context does not use resident rendering"; return -1; }
    *output=context.profile;
    return 0;
}

int pr_resident_render_planes(void* handle, const float* matrices, uint32_t instance_count,
                             const PrParams* params, uint32_t width, uint32_t height,
                             uint32_t channel_mask, uint8_t* output, size_t output_bytes) {
    try {
        last_error.clear(); require(handle && params,"null resident compact capture argument");
        resident_render(*static_cast<Context*>(handle),matrices,instance_count,*params,
                        width,height,output,output_bytes,channel_mask,true);
        return 0;
    } catch (const std::exception& e) { last_error=e.what(); return -1; }
      catch (...) { last_error="unknown resident compact capture error"; return -1; }
}
int pr_resident_readback_info(void* handle, PrResidentReadbackInfo* output) {
    if (!handle || !output) { last_error="null resident readback info argument"; return -1; }
    auto& context=*static_cast<Context*>(handle);
    if (!context.resident) { last_error="context does not use resident rendering"; return -1; }
    *output=context.readback_info;
    output->memory_type_index=UINT32_MAX; output->memory_property_flags=0;
    output->persistent_mapped=0; output->capacity_bytes=0;
    if (context.resident_state && context.resident_state->target) {
        const auto& frame=*context.resident_state->target;
        output->memory_type_index=frame.readback.allocation.type_index;
        output->memory_property_flags=frame.readback.allocation.property_flags;
        output->persistent_mapped=frame.readback_mapping ? 1u : 0u;
        output->capacity_bytes=frame.readback_capacity;
        for (const auto& extra:context.resident_state->additional_targets) if (extra) {
            output->capacity_bytes+=extra->readback_capacity;
            if (!extra->readback_mapping) output->persistent_mapped=0;
        }
    }
    return 0;
}

uint32_t pr_resident_texture_abi_version() { return 1; }
int pr_resident_set_textures(void* handle,const PrResidentTexture* textures,uint32_t texture_count,
    const uint8_t* rgba,size_t rgba_bytes,const PrResidentUV* uv,uint32_t uv_count,
    const uint32_t* instance_slots,uint32_t instance_count) {
    try {
        last_error.clear(); require(handle,"null resident texture context");
        resident_set_textures(*static_cast<Context*>(handle),textures,texture_count,rgba,rgba_bytes,
                              uv,uv_count,instance_slots,instance_count);
        return 0;
    } catch (const std::exception& e) { last_error=e.what(); return -1; }
      catch (...) { last_error="unknown resident texture upload error"; return -1; }
}
int pr_resident_texture_info(void* handle,PrResidentTextureInfo* output) {
    if (!handle || !output) { last_error="null resident texture info argument"; return -1; }
    auto& context=*static_cast<Context*>(handle);
    if (!context.resident) { last_error="context does not use resident rendering"; return -1; }
    *output=context.texture_info;
    return 0;
}

uint32_t pr_resident_environment_abi_version() { return 1; }
int pr_resident_set_environment(void* handle,const PrResidentEnvironment* environment,
    const float* rgba,size_t rgba_bytes,const uint32_t* sky_vert,size_t sky_vert_bytes,
    const uint32_t* sky_frag,size_t sky_frag_bytes) {
    try {
        require(handle && environment,"null resident environment argument");
        resident_set_environment(*static_cast<Context*>(handle),*environment,rgba,rgba_bytes,
            sky_vert,sky_vert_bytes,sky_frag,sky_frag_bytes);
        return 0;
    } catch (const std::exception& error) { last_error=error.what(); return -1; }
}
int pr_resident_environment_info(void* handle,PrResidentEnvironmentInfo* output) {
    if (!handle || !output) { last_error="null resident environment info"; return -1; }
    auto& context=*static_cast<Context*>(handle);
    if (!context.resident) { last_error="not a resident context"; return -1; }
    *output=context.environment_info;
    return 0;
}

uint32_t pr_resident_ray_query_abi_version() { return 1; }
void* pr_resident_create_ray_query(uint32_t selected,uint32_t validation,
    const uint32_t* vert,size_t vert_bytes,const uint32_t* frag,size_t frag_bytes) {
    try {
        require(validation<=1,"invalid validation flag");
        auto context=std::make_unique<Context>();
        context->init(selected,validation!=0,vert,vert_bytes,frag,frag_bytes,true,true);
        return context.release();
    } catch (const std::exception& error) { last_error=error.what(); return nullptr; }
}
int pr_resident_ray_query_info(void* handle,PrResidentRayQueryInfo* output) {
    if (!handle || !output) { last_error="null resident ray-query info"; return -1; }
    auto& context=*static_cast<Context*>(handle);
    if (!context.resident) { last_error="not a resident context"; return -1; }
    *output=context.ray_info; output->poisoned=context.poisoned ? 1u : 0u;
    return 0;
}

uint32_t pr_resident_surface_maps_abi_version() { return 1; }
int pr_resident_surface_maps_shader_compatible(void* handle) {
    if (!handle) { last_error="null resident surface-map context"; return -1; }
    const auto& context=*static_cast<Context*>(handle);
    if (!context.resident) { last_error="not a resident context"; return -1; }
    return context.surface_maps_compatible ? 1 : 0;
}
int pr_resident_set_surface_maps(void* handle,const PrResidentTexture* textures,uint32_t texture_count,
    const uint8_t* rgba,size_t rgba_bytes,const PrResidentUV* uv,uint32_t uv_count,
    const uint32_t* instance_slots,uint32_t instance_count) {
    try {
        require(handle,"null resident surface-map context");
        resident_set_textures(*static_cast<Context*>(handle),textures,texture_count,rgba,rgba_bytes,
            uv,uv_count,instance_slots,instance_count,true);
        return 0;
    } catch (const std::exception& error) { last_error=error.what(); return -1; }
}

uint32_t pr_resident_display_transform_abi_version() { return 1; }
int pr_resident_display_transform_shader_compatible(void* handle) {
    if (!handle) { last_error="null resident display context"; return -1; }
    const auto& c=*static_cast<Context*>(handle);
    if (!c.resident) { last_error="not a resident context"; return -1; }
    return c.display_compatible ? 1 : 0;
}
int pr_resident_set_display_transform(void* handle,uint32_t mode) {
    try {
        require(handle,"null resident display context");
        auto& c=*static_cast<Context*>(handle);
        require(c.resident && !c.poisoned,"resident display context unavailable");
        require(mode<=1,"unknown resident display transform");
        require(c.properties.captures==0,"resident display transform is immutable after capture");
        if (mode==1) {
            require(c.display_compatible,"resident fragment lacks named display-transform ABI marker");
            require(!c.environment_info.loaded || (c.resident_state && c.resident_state->environment &&
                c.resident_state->environment->display_compatible),
                "resident sky fragment lacks named display-transform ABI marker");
        }
        c.display_transform=mode;
        return 0;
    } catch (const std::exception& error) { last_error=error.what(); return -1; }
}

uint32_t pr_resident_msaa_abi_version() { return 1; }
int pr_resident_msaa_info(void* handle,PrResidentMsaaInfo* output) {
    if(!handle || !output) { last_error="null resident MSAA info"; return -1; }
    auto& c=*static_cast<Context*>(handle);
    if(!c.resident) { last_error="not a resident MSAA context"; return -1; }
    *output=c.msaa_info;
    return 0;
}
int pr_resident_set_msaa(void* handle,uint32_t samples,const uint32_t* vert,size_t vert_bytes,
                        const uint32_t* frag,size_t frag_bytes) {
    try {
        require(handle,"null resident MSAA context");
        resident_set_msaa(*static_cast<Context*>(handle),samples,vert,vert_bytes,frag,frag_bytes);
        return 0;
    } catch(const std::exception& error) { last_error=error.what(); return -1; }
}

uint32_t pr_resident_normal_maps_abi_version() { return 1; }
int pr_resident_normal_info(void* handle,PrResidentNormalInfo* output) {
    if(!handle || !output) { last_error="null resident normal-map info"; return -1; }
    auto& c=*static_cast<Context*>(handle);
    if(!c.resident) { last_error="normal-map info requires resident context"; return -1; }
    *output=c.normal_info; return 0;
}
int pr_resident_set_normal_maps(void* handle,const PrResidentTexture* textures,uint32_t texture_count,
    const uint8_t* rgba,size_t rgba_bytes,const PrResidentUV* uv,uint32_t uv_count,
    const uint32_t* slots,uint32_t instance_count) {
    try {
        require(handle,"null normal-map context");
        resident_set_normal_maps(*static_cast<Context*>(handle),textures,texture_count,rgba,rgba_bytes,uv,uv_count,slots,instance_count);
        return 0;
    } catch(const std::exception& error) { last_error=error.what(); return -1; }
}
uint32_t pr_resident_reflection_abi_version() { return 1; }
int pr_resident_reflection_info(void* handle,PrResidentReflectionInfo* output) {
    if(!handle || !output) { last_error="null resident reflection info"; return -1; }
    auto& c=*static_cast<Context*>(handle);
    if(!c.resident) { last_error="reflection info requires resident context"; return -1; }
    *output=c.reflection_info;return 0;
}
int pr_resident_set_reflection(void* handle,uint32_t mode) {
    try {
        require(handle,"null reflection context"); auto& c=*static_cast<Context*>(handle);
        auto& info=c.reflection_info;
        require(mode==1,"unsupported reflection mode");
        require(c.resident && c.ray_query && c.resident_state && !c.poisoned,"reflection requires loaded hardware ray-query geometry");
        require(info.supported && info.shader_compatible,"reflection shader ABI unavailable; no fallback");
        require(!info.enabled && !c.properties.captures,"reflection is immutable after enable or capture");
        uint32_t receivers=0;
        for(const auto& instance:c.resident_state->instances)
            if(instance.material[0]>=info.metallic_min && instance.color_roughness[3]<=info.roughness_max) ++receivers;
        // Zero receivers is a valid explicit capability-enabled scene.
        info.receiver_count=receivers;info.enabled=1;++info.set_calls;return 0;
    } catch(const std::exception& error) { last_error=error.what();return -1; }
}

uint32_t pr_resident_indirect_abi_version() { return 1; }
int pr_resident_indirect_info(void* handle,PrResidentIndirectInfo* output) {
    if(!handle||!output){last_error="null resident indirect info";return -1;}
    auto& c=*static_cast<Context*>(handle);
    if(!c.resident){last_error="indirect info requires resident context";return -1;}
    *output=c.indirect_info;return 0;
}
int pr_resident_set_indirect(void* handle,uint32_t mode,uint32_t seed) {
    try {
        require(handle,"null indirect context");auto& c=*static_cast<Context*>(handle);auto& info=c.indirect_info;
        require(mode==1,"unsupported indirect mode");
        require(c.resident&&c.ray_query&&c.resident_state&&!c.poisoned,"indirect requires loaded hardware ray-query scene");
        require(info.supported&&info.shader_compatible,"indirect shader ABI unavailable; no fallback");
        require(!info.enabled&&!c.properties.captures,"indirect configuration is immutable after enable/capture");
        uint32_t receivers=0;
        for(const auto& instance:c.resident_state->instances)
            if(instance.material[0]<1.f && (instance.color_roughness[0]>0.f||instance.color_roughness[1]>0.f||instance.color_roughness[2]>0.f)) ++receivers;
        info.receiver_count=receivers;info.seed=seed;info.enabled=1;++info.set_calls;return 0;
    }catch(const std::exception& error){last_error=error.what();return -1;}
}

uint32_t pr_resident_progressive_abi_version(){return 1;}
int pr_resident_progressive_info(void* handle,PrResidentProgressiveInfo* output){
    if(!handle||!output){last_error="null progressive info";return -1;}
    *output=static_cast<Context*>(handle)->progressive_info;return 0;
}
int pr_resident_set_progressive(void* handle,uint32_t bounces,const uint32_t* code,size_t bytes){
    try{require(handle,"null progressive context");progressive_setup(*static_cast<Context*>(handle),bounces,code,bytes);return 0;}
    catch(const std::exception& error){last_error=error.what();return -1;}
}
int pr_resident_begin_progressive(void* handle,uint32_t reset){
    try{
        require(handle,"null progressive context");auto& c=*static_cast<Context*>(handle);
        require(c.progressive_info.enabled&&!c.poisoned&&reset<=1,"invalid progressive begin");
        require(reset||c.progressive_info.batches<4096,"progressive maximum 4096 batches reached");
        c.progressive_state->armed=true;c.progressive_state->pending_reset=reset!=0;return 0;
    }catch(const std::exception& error){last_error=error.what();return -1;}
}
int pr_resident_reset_progressive(void* handle){
    try{
        require(handle,"null progressive context");auto& c=*static_cast<Context*>(handle);
        require(c.progressive_info.enabled&&!c.poisoned,"progressive reset requires valid enabled session");
        c.progressive_info.batches=0;c.progressive_info.last_batch_index=0;++c.progressive_info.reset_calls;
        c.progressive_state->armed=false;c.progressive_state->pending_reset=true;c.spatial_info.last_capture_filtered=0;c.settling_info.guide_valid=0;c.settling_info.last_capture_filtered=0;c.settling_info.last_filtered_fraction=0;c.settling_info.last_batch=0;c.settling_info.last_filtered_primary_samples=0;return 0;
    }catch(const std::exception& error){last_error=error.what();return -1;}
}
int pr_resident_read_progressive_mean(void* handle,void* output,size_t bytes){
    try{
        require(handle,"null progressive context");auto& c=*static_cast<Context*>(handle);
        require(c.progressive_info.enabled,"progressive mean unavailable");
        progressive_read(c,output,bytes);return 0;
    }catch(const std::exception& error){last_error=error.what();return -1;}
}

uint32_t pr_resident_spatial_abi_version(){return 1;}
int pr_resident_spatial_info(void* handle,PrResidentSpatialInfo* output){
    if(!handle||!output){last_error="null spatial info";return -1;}
    *output=static_cast<Context*>(handle)->spatial_info;return 0;
}
int pr_resident_set_spatial(void* handle,uint32_t bounces,const uint32_t* code,size_t bytes,const uint32_t* guide,size_t guide_bytes){
    try{require(handle,"null spatial context");spatial_setup(*static_cast<Context*>(handle),bounces,code,bytes,guide,guide_bytes);return 0;}
    catch(const std::exception& e){last_error=e.what();return -1;}
}
int pr_resident_read_spatial_guide(void* handle,void* output,size_t bytes){
    try{require(handle,"null spatial context");spatial_read(*static_cast<Context*>(handle),output,bytes);return 0;}
    catch(const std::exception& e){last_error=e.what();return -1;}
}
uint32_t pr_resident_settling_abi_version(){return 1;}
int pr_resident_settling_info(void* handle,PrResidentSettlingInfo* output){
    if(!handle||!output){last_error="null settling info";return -1;}
    *output=static_cast<Context*>(handle)->settling_info;return 0;
}
int pr_resident_set_settling(void* handle,uint32_t bounces,const uint32_t* mean,size_t mean_bytes,
    const uint32_t* guide,size_t guide_bytes,const uint32_t* filter,size_t filter_bytes){
    try{require(handle,"null settling context");settling_setup(*static_cast<Context*>(handle),bounces,mean,mean_bytes,guide,guide_bytes,filter,filter_bytes);return 0;}
    catch(const std::exception& e){last_error=e.what();return -1;}
}
