#include "renderer.h"
#include <vulkan/vulkan.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
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
struct Buffer { VkBuffer handle{}; VkDeviceMemory memory{}; };
struct Image { VkImage handle{}; VkDeviceMemory memory{}; VkImageView view{}; };
struct Context {
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
    VkShaderModule vertex_shader{}, fragment_shader{};
    VkPhysicalDeviceMemoryProperties memory_properties{};
    PrInfo properties{};
    std::atomic<uint64_t> errors{0}, warnings{0};
    std::mutex message_lock;
    std::string validation_message;
    bool poisoned = false;

    Context() { ++live_contexts; }
    ~Context() { shutdown(); --live_contexts; }
    VkResult shutdown() noexcept {
        VkResult waited = VK_SUCCESS;
        if (device) {
            waited = vkDeviceWaitIdle(device);
            release(pipeline, [&](auto h) { vkDestroyPipeline(device,h,nullptr); });
            release(pipeline_layout, [&](auto h) { vkDestroyPipelineLayout(device,h,nullptr); });
            release(render_pass, [&](auto h) { vkDestroyRenderPass(device,h,nullptr); });
            release(vertex_shader, [&](auto h) { vkDestroyShaderModule(device,h,nullptr); });
            release(fragment_shader, [&](auto h) { vkDestroyShaderModule(device,h,nullptr); });
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
    uint32_t memory_type(uint32_t bits, VkMemoryPropertyFlags wanted) const {
        for (uint32_t i=0; i<memory_properties.memoryTypeCount; ++i)
            if ((bits & (1u<<i)) &&
                (memory_properties.memoryTypes[i].propertyFlags & wanted) == wanted) return i;
        throw std::runtime_error("required Vulkan memory type unavailable");
    }
    void make_buffer(Buffer& value, VkDeviceSize bytes, VkBufferUsageFlags usage) {
        VkBufferCreateInfo create{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        create.size = bytes; create.usage = usage; create.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        acquire(value.handle, "vkCreateBuffer", [&](auto* output) { return vkCreateBuffer(device, &create, nullptr, output); });
        VkMemoryRequirements needed{}; vkGetBufferMemoryRequirements(device, value.handle, &needed);
        VkMemoryAllocateInfo alloc{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        alloc.allocationSize = needed.size;
        alloc.memoryTypeIndex = memory_type(needed.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        acquire(value.memory, "vkAllocateMemory buffer", [&](auto* output) { return vkAllocateMemory(device, &alloc, nullptr, output); });
        check(vkBindBufferMemory(device, value.handle, value.memory, 0), "vkBindBufferMemory");
    }
    void upload(const Buffer& buffer, const void* source, size_t bytes) {
        void* mapped{};
        check(vkMapMemory(device, buffer.memory, 0, bytes, 0, &mapped), "vkMapMemory upload");
        std::memcpy(mapped, source, bytes);
        vkUnmapMemory(device, buffer.memory);
    }
    void make_image(Image& value, uint32_t width, uint32_t height, VkFormat format, bool depth) {
        VkImageCreateInfo create{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        create.imageType = VK_IMAGE_TYPE_2D; create.format = format;
        create.extent = {width, height, 1}; create.mipLevels = 1; create.arrayLayers = 1;
        create.samples = VK_SAMPLE_COUNT_1_BIT; create.tiling = VK_IMAGE_TILING_OPTIMAL;
        create.usage = depth ? VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT :
            (VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
        create.sharingMode = VK_SHARING_MODE_EXCLUSIVE; create.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        acquire(value.handle, "vkCreateImage", [&](auto* output) { return vkCreateImage(device, &create, nullptr, output); });
        VkMemoryRequirements needed{}; vkGetImageMemoryRequirements(device, value.handle, &needed);
        VkMemoryAllocateInfo alloc{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        alloc.allocationSize = needed.size;
        alloc.memoryTypeIndex = memory_type(needed.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        acquire(value.memory, "vkAllocateMemory image", [&](auto* output) { return vkAllocateMemory(device, &alloc, nullptr, output); });
        check(vkBindImageMemory(device, value.handle, value.memory, 0), "vkBindImageMemory");
        VkImageViewCreateInfo view{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        view.image = value.handle; view.viewType = VK_IMAGE_VIEW_TYPE_2D; view.format = format;
        view.subresourceRange.aspectMask = depth ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
        view.subresourceRange.levelCount = 1; view.subresourceRange.layerCount = 1;
        acquire(value.view, "vkCreateImageView", [&](auto* output) { return vkCreateImageView(device, &view, nullptr, output); });
    }
    void init(uint32_t selected, bool validation, const uint32_t* vert, size_t vert_bytes,
              const uint32_t* frag, size_t frag_bytes) {
        require(vert && frag && vert_bytes >= 20 && frag_bytes >= 20 &&
            vert_bytes%4 == 0 && frag_bytes%4 == 0, "invalid SPIR-V input");
        require(vert[0] == 0x07230203 && frag[0] == 0x07230203, "invalid SPIR-V magic");
        VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        app.pApplicationName = "phymeo-native-offscreen";
        app.applicationVersion = 1; app.pEngineName = "phymeo_rendering";
        app.engineVersion = 1; app.apiVersion = VK_API_VERSION_1_1;
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
        vkGetPhysicalDeviceProperties2(physical, &description);
        const auto& p = description.properties;
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
        std::memcpy(properties.device_uuid, identity.deviceUUID, 16);
        std::memcpy(properties.device_name, p.deviceName, sizeof(properties.device_name));
        for (size_t i=0; i<FORMATS.size(); ++i) {
            VkFormatProperties format{}; vkGetPhysicalDeviceFormatProperties(physical, FORMATS[i], &format);
            VkFormatFeatureFlags needed = i == 4 ? VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT :
                (VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT | VK_FORMAT_FEATURE_TRANSFER_SRC_BIT);
            require((format.optimalTilingFeatures & needed) == needed, "required attachment format unavailable");
        }
        vkGetPhysicalDeviceMemoryProperties(physical, &memory_properties);
        vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, nullptr);
        std::vector<VkQueueFamilyProperties> queues(count);
        vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, queues.data());
        queue_family = count;
        for (uint32_t i=0; i<count; ++i) if (queues[i].queueCount &&
            (queues[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) { queue_family=i; break; }
        require(queue_family < count, "graphics queue unavailable");
        float priority = 1.0f;
        VkDeviceQueueCreateInfo queue_info{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        queue_info.queueFamilyIndex = queue_family; queue_info.queueCount = 1; queue_info.pQueuePriorities = &priority;
        VkDeviceCreateInfo device_info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        device_info.queueCreateInfoCount = 1; device_info.pQueueCreateInfos = &queue_info;
        acquire(device, "vkCreateDevice", [&](auto* output) { return vkCreateDevice(physical, &device_info, nullptr, output); });
        vkGetDeviceQueue(device, queue_family, 0, &queue);
        VkCommandPoolCreateInfo pool{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pool.queueFamilyIndex = queue_family; pool.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
        acquire(command_pool, "vkCreateCommandPool", [&](auto* output) { return vkCreateCommandPool(device, &pool, nullptr, output); });
        std::array<VkDescriptorSetLayoutBinding, 2> bindings{};
        bindings[0] = {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
        bindings[1] = {1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1,
                       VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
        VkDescriptorSetLayoutCreateInfo layout{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        layout.bindingCount = 2; layout.pBindings = bindings.data();
        acquire(descriptor_layout, "descriptor layout", [&](auto* output) { return vkCreateDescriptorSetLayout(device, &layout, nullptr, output); });
        VkPipelineLayoutCreateInfo pipeline_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pipeline_info.setLayoutCount = 1; pipeline_info.pSetLayouts = &descriptor_layout;
        acquire(pipeline_layout, "pipeline layout", [&](auto* output) { return vkCreatePipelineLayout(device, &pipeline_info, nullptr, output); });
        std::array<VkAttachmentDescription, 5> attachments{};
        std::array<VkAttachmentReference, 4> colors{};
        for (uint32_t i=0; i<5; ++i) {
            auto& value = attachments[i]; value.format=FORMATS[i]; value.samples=VK_SAMPLE_COUNT_1_BIT;
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
        VkVertexInputBindingDescription vertex_binding{0,sizeof(PrVertex),VK_VERTEX_INPUT_RATE_VERTEX};
        std::array<VkVertexInputAttributeDescription,4> attributes{};
        for (uint32_t i=0;i<4;++i) attributes[i]={i,0,
            i==3?VK_FORMAT_R32G32B32A32_UINT:VK_FORMAT_R32G32B32A32_SFLOAT,i*16u};
        VkPipelineVertexInputStateCreateInfo vertex_input{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        vertex_input.vertexBindingDescriptionCount=1; vertex_input.pVertexBindingDescriptions=&vertex_binding;
        vertex_input.vertexAttributeDescriptionCount=4; vertex_input.pVertexAttributeDescriptions=attributes.data();
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

struct FrameResources {
    Context& c;
    Buffer vertices{}, uniform{}, readback{};
    std::array<Image,5> images{};
    VkFramebuffer framebuffer{};
    VkDescriptorPool descriptors{};
    VkCommandBuffer commands{};
    VkFence fence{};
    bool submitted=false;
    explicit FrameResources(Context& context):c(context){ ++live_frames; }
    ~FrameResources() {
        if(submitted) vkQueueWaitIdle(c.queue);
        release(fence, [&](auto owned) { vkDestroyFence(c.device,owned,nullptr); });
        release(commands, [&](auto h) { vkFreeCommandBuffers(c.device,c.command_pool,1,&h); });
        release(descriptors, [&](auto owned) { vkDestroyDescriptorPool(c.device,owned,nullptr); });
        release(framebuffer, [&](auto owned) { vkDestroyFramebuffer(c.device,owned,nullptr); });
        for(auto& image:images) {
            release(image.view, [&](auto owned) { vkDestroyImageView(c.device,owned,nullptr); });
            release(image.handle, [&](auto owned) { vkDestroyImage(c.device,owned,nullptr); });
            release(image.memory, [&](auto owned) { vkFreeMemory(c.device,owned,nullptr); });
        }
        for(auto* buffer:{&vertices,&uniform,&readback}) {
            release(buffer->handle, [&](auto owned) { vkDestroyBuffer(c.device,owned,nullptr); });
            release(buffer->memory, [&](auto owned) { vkFreeMemory(c.device,owned,nullptr); });
        }
        --live_frames;
    }
};

void render(Context& c,const PrVertex* vertices,uint32_t vertex_count,const PrParams& params,
            uint32_t width,uint32_t height,uint8_t* output,size_t output_bytes) {
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
    VkResult waited=context->shutdown();
    if(final_info) context->info(final_info);
    uint64_t errors=context->errors.load();
    delete context;
    if(waited!=VK_SUCCESS) { last_error="vkDeviceWaitIdle during close: "+std::to_string(waited); return -1; }
    if(errors) { last_error="Vulkan validation reported errors before close completed"; return -1; }
    return 0;
}
