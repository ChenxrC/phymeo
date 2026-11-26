// Dedicated opaque indexed-instanced Vulkan floor raster. No physics or CPU image formation.
#include <vulkan/vulkan.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#define API extern "C" __declspec(dllexport)
#else
#include <dlfcn.h>
#define API extern "C" __attribute__((visibility("default")))
#endif
namespace {
thread_local std::string error_text;
void require(bool ok,const char* text){if(!ok)throw std::runtime_error(text);}
void check(VkResult result,const char* operation){if(result!=VK_SUCCESS)throw std::runtime_error(std::string(operation)+" VkResult="+std::to_string(result));}
template<class T,class F> void acquire(T& destination,F operation,const char* name){
    T temporary{};auto status=operation(&temporary);check(status,name);destination=temporary;
}
void location_anchor(){}
std::filesystem::path shader_directory(){
    if(const char* p=std::getenv("PHYMEO_FLOOR_SHADER_DIR"))return p;
#ifdef _WIN32
    HMODULE module{};require(GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS|GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
        reinterpret_cast<LPCWSTR>(&location_anchor),&module)!=0,"cannot locate native library");
    std::vector<wchar_t> path(32768);DWORD n=GetModuleFileNameW(module,path.data(),static_cast<DWORD>(path.size()));
    require(n>0&&n<path.size(),"cannot resolve native library path");return std::filesystem::path(std::wstring(path.data(),n)).parent_path();
#else
    Dl_info info{};require(dladdr(reinterpret_cast<void*>(&location_anchor),&info)&&info.dli_fname,"cannot locate native library");
    return std::filesystem::absolute(info.dli_fname).parent_path();
#endif
}
std::vector<uint32_t> spirv(const char* name){
    auto path=shader_directory()/name;std::ifstream file(path,std::ios::binary|std::ios::ate);
    require(bool(file),"missing floor SPIR-V (set PHYMEO_FLOOR_SHADER_DIR if required)");auto bytes=file.tellg();
    require(bytes>=20&&bytes<=16*1024*1024&&static_cast<size_t>(bytes)%4==0,"invalid SPIR-V byte length");
    std::vector<uint32_t> code(static_cast<size_t>(bytes)/4);file.seekg(0);file.read(reinterpret_cast<char*>(code.data()),bytes);
    require(bool(file)&&code[0]==0x07230203u,"invalid floor SPIR-V");return code;
}
struct Buffer{VkBuffer handle{};VkDeviceMemory memory{};void* mapped{};VkDeviceSize bytes{};};
struct Image{VkImage handle{};VkDeviceMemory memory{};VkImageView view{};};
struct Mesh{uint32_t first_vertex,vertex_count,first_index,index_count;};
struct Draw{uint32_t first_instance,instance_count;};
struct Floor{
    VkInstance instance{};VkPhysicalDevice physical{};VkPhysicalDeviceProperties properties{};VkPhysicalDeviceMemoryProperties memories{};
    VkDevice device{};VkQueue queue{};uint32_t queue_family{},width{},height{};VkSampleCountFlagBits samples=VK_SAMPLE_COUNT_1_BIT;
    VkCommandPool commands_pool{};VkCommandBuffer commands{};VkFence fence{};
    VkRenderPass color_pass{},shadow_pass{};VkFramebuffer color_frame{},shadow_frame{};
    VkDescriptorSetLayout descriptors{};VkDescriptorPool descriptor_pool{};VkDescriptorSet descriptor_set{};VkSampler shadow_sampler{};
    VkPipelineLayout pipeline_layout{};VkPipeline color_pipeline{},shadow_pipeline{};
    Image color,depth,resolve,shadow;Buffer vertices,indices,models,colors,readback;
    std::vector<Mesh> meshes;std::vector<Draw> draws;uint32_t instance_count{};uint64_t submissions{};
    bool loaded=false,poisoned=false,pending=false;std::string info;
    static constexpr uint32_t shadow_size=2048;
    ~Floor(){clear();}
    void clear() noexcept{
        if(device){vkDeviceWaitIdle(device);
            for(auto p:{color_pipeline,shadow_pipeline})if(p)vkDestroyPipeline(device,p,nullptr);
            if(pipeline_layout)vkDestroyPipelineLayout(device,pipeline_layout,nullptr);
            for(auto f:{color_frame,shadow_frame})if(f)vkDestroyFramebuffer(device,f,nullptr);
            for(auto p:{color_pass,shadow_pass})if(p)vkDestroyRenderPass(device,p,nullptr);
            if(descriptor_pool)vkDestroyDescriptorPool(device,descriptor_pool,nullptr);
            if(descriptors)vkDestroyDescriptorSetLayout(device,descriptors,nullptr);
            if(shadow_sampler)vkDestroySampler(device,shadow_sampler,nullptr);
            for(auto* p:{&color,&depth,&resolve,&shadow}){if(p->view)vkDestroyImageView(device,p->view,nullptr);if(p->handle)vkDestroyImage(device,p->handle,nullptr);if(p->memory)vkFreeMemory(device,p->memory,nullptr);*p={};}
            for(auto* p:{&vertices,&indices,&models,&colors,&readback}){if(p->mapped)vkUnmapMemory(device,p->memory);if(p->handle)vkDestroyBuffer(device,p->handle,nullptr);if(p->memory)vkFreeMemory(device,p->memory,nullptr);*p={};}
            if(fence)vkDestroyFence(device,fence,nullptr);if(commands_pool)vkDestroyCommandPool(device,commands_pool,nullptr);
            vkDestroyDevice(device,nullptr);device={};
        }
        if(instance){vkDestroyInstance(instance,nullptr);instance={};}
    }
    uint32_t memory_type(uint32_t bits,VkMemoryPropertyFlags flags){
        for(uint32_t i=0;i<memories.memoryTypeCount;++i)if((bits&(1u<<i))&&(memories.memoryTypes[i].propertyFlags&flags)==flags)return i;
        throw std::runtime_error("required Vulkan memory type unavailable");
    }
    void buffer(Buffer& out,VkDeviceSize bytes,VkBufferUsageFlags usage){
        require(bytes>0,"zero buffer allocation");VkBufferCreateInfo create{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};create.size=bytes;create.usage=usage;create.sharingMode=VK_SHARING_MODE_EXCLUSIVE;
        VkBuffer h{};check(vkCreateBuffer(device,&create,nullptr,&h),"create buffer");out.handle=h;out.bytes=bytes;
        VkMemoryRequirements req{};vkGetBufferMemoryRequirements(device,h,&req);VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocation.allocationSize=req.size;allocation.memoryTypeIndex=memory_type(req.memoryTypeBits,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        VkDeviceMemory m{};check(vkAllocateMemory(device,&allocation,nullptr,&m),"allocate buffer");out.memory=m;
        check(vkBindBufferMemory(device,h,m,0),"bind buffer");check(vkMapMemory(device,m,0,bytes,0,&out.mapped),"map persistent buffer");
    }
    void image(Image& out,uint32_t w,uint32_t h,VkFormat format,VkSampleCountFlagBits count,VkImageUsageFlags usage,VkImageAspectFlags aspect){
        VkImageCreateInfo create{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};create.imageType=VK_IMAGE_TYPE_2D;create.format=format;create.extent={w,h,1};
        create.mipLevels=create.arrayLayers=1;create.samples=count;create.tiling=VK_IMAGE_TILING_OPTIMAL;create.usage=usage;create.sharingMode=VK_SHARING_MODE_EXCLUSIVE;
        VkImage value{};check(vkCreateImage(device,&create,nullptr,&value),"create image");out.handle=value;
        VkMemoryRequirements req{};vkGetImageMemoryRequirements(device,value,&req);VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocation.allocationSize=req.size;allocation.memoryTypeIndex=memory_type(req.memoryTypeBits,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        VkDeviceMemory m{};check(vkAllocateMemory(device,&allocation,nullptr,&m),"allocate image");out.memory=m;check(vkBindImageMemory(device,value,m,0),"bind image");
        VkImageViewCreateInfo view{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};view.image=value;view.viewType=VK_IMAGE_VIEW_TYPE_2D;view.format=format;view.subresourceRange={aspect,0,1,0,1};
        VkImageView v{};check(vkCreateImageView(device,&view,nullptr,&v),"create image view");out.view=v;
    }
    void init(uint32_t index,uint32_t w,uint32_t h){
        require(w>0&&h>0&&w<=8192&&h<=8192,"unsupported output extent");width=w;height=h;
        VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};app.pApplicationName="Phymeo training floor";app.apiVersion=VK_API_VERSION_1_1;
        VkInstanceCreateInfo create{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};create.pApplicationInfo=&app;
        VkInstance ins{};check(vkCreateInstance(&create,nullptr,&ins),"create Vulkan instance");instance=ins;
        uint32_t count=0;check(vkEnumeratePhysicalDevices(instance,&count,nullptr),"enumerate devices");require(index<count,"Vulkan device index unavailable");
        std::vector<VkPhysicalDevice> devices(count);check(vkEnumeratePhysicalDevices(instance,&count,devices.data()),"enumerate devices");physical=devices[index];
        vkGetPhysicalDeviceProperties(physical,&properties);vkGetPhysicalDeviceMemoryProperties(physical,&memories);
        require(properties.apiVersion>=VK_API_VERSION_1_1,"Vulkan1.1 required");
        require(w<=properties.limits.maxFramebufferWidth&&h<=properties.limits.maxFramebufferHeight&&shadow_size<=properties.limits.maxImageDimension2D,"device extent limit");
        vkGetPhysicalDeviceQueueFamilyProperties(physical,&count,nullptr);std::vector<VkQueueFamilyProperties> families(count);vkGetPhysicalDeviceQueueFamilyProperties(physical,&count,families.data());
        queue_family=count;for(uint32_t i=0;i<count;++i)if(families[i].queueFlags&VK_QUEUE_GRAPHICS_BIT){queue_family=i;break;}require(queue_family<count,"graphics queue required");
        float priority=1;VkDeviceQueueCreateInfo q{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};q.queueFamilyIndex=queue_family;q.queueCount=1;q.pQueuePriorities=&priority;
        VkDeviceCreateInfo dc{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};dc.queueCreateInfoCount=1;dc.pQueueCreateInfos=&q;
        VkDevice d{};check(vkCreateDevice(physical,&dc,nullptr,&d),"create graphics device");device=d;vkGetDeviceQueue(device,queue_family,0,&queue);
        VkFormatProperties depth_properties{};vkGetPhysicalDeviceFormatProperties(physical,VK_FORMAT_D32_SFLOAT,&depth_properties);
        require((depth_properties.optimalTilingFeatures&(VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT|VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT))==(VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT|VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT),"sampled D32 shadow unavailable");
        auto supported=properties.limits.framebufferColorSampleCounts&properties.limits.framebufferDepthSampleCounts;
        VkImageFormatProperties color_caps{},depth_caps{};
        check(vkGetPhysicalDeviceImageFormatProperties(physical,VK_FORMAT_R8G8B8A8_SRGB,VK_IMAGE_TYPE_2D,VK_IMAGE_TILING_OPTIMAL,VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,0,&color_caps),"SRGB attachment support");
        check(vkGetPhysicalDeviceImageFormatProperties(physical,VK_FORMAT_D32_SFLOAT,VK_IMAGE_TYPE_2D,VK_IMAGE_TILING_OPTIMAL,VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,0,&depth_caps),"depth support");
        supported&=color_caps.sampleCounts&depth_caps.sampleCounts;if(supported&VK_SAMPLE_COUNT_4_BIT)samples=VK_SAMPLE_COUNT_4_BIT;
        VkCommandPoolCreateInfo pool{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};pool.queueFamilyIndex=queue_family;pool.flags=VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        acquire(commands_pool,[&](auto* out){return vkCreateCommandPool(device,&pool,nullptr,out);},"command pool");
        VkCommandBufferAllocateInfo ca{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};ca.commandPool=commands_pool;ca.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY;ca.commandBufferCount=1;check(vkAllocateCommandBuffers(device,&ca,&commands),"allocate commands");
        VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};fi.flags=VK_FENCE_CREATE_SIGNALED_BIT;acquire(fence,[&](auto* out){return vkCreateFence(device,&fi,nullptr,out);},"create fence");
        image(color,w,h,VK_FORMAT_R8G8B8A8_SRGB,samples,VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT|(samples==VK_SAMPLE_COUNT_1_BIT?VK_IMAGE_USAGE_TRANSFER_SRC_BIT:0),VK_IMAGE_ASPECT_COLOR_BIT);
        image(depth,w,h,VK_FORMAT_D32_SFLOAT,samples,VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,VK_IMAGE_ASPECT_DEPTH_BIT);
        if(samples!=VK_SAMPLE_COUNT_1_BIT)image(resolve,w,h,VK_FORMAT_R8G8B8A8_SRGB,VK_SAMPLE_COUNT_1_BIT,VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT|VK_IMAGE_USAGE_TRANSFER_SRC_BIT,VK_IMAGE_ASPECT_COLOR_BIT);
        image(shadow,shadow_size,shadow_size,VK_FORMAT_D32_SFLOAT,VK_SAMPLE_COUNT_1_BIT,VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT|VK_IMAGE_USAGE_SAMPLED_BIT,VK_IMAGE_ASPECT_DEPTH_BIT);
        buffer(readback,static_cast<VkDeviceSize>(w)*h*4,VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        setup_passes();setup_descriptors();color_pipeline=pipeline(false);shadow_pipeline=pipeline(true);
    }
    void setup_passes(){
        VkAttachmentDescription s{};s.format=VK_FORMAT_D32_SFLOAT;s.samples=VK_SAMPLE_COUNT_1_BIT;s.loadOp=VK_ATTACHMENT_LOAD_OP_CLEAR;s.storeOp=VK_ATTACHMENT_STORE_OP_STORE;
        s.stencilLoadOp=VK_ATTACHMENT_LOAD_OP_DONT_CARE;s.stencilStoreOp=VK_ATTACHMENT_STORE_OP_DONT_CARE;s.initialLayout=VK_IMAGE_LAYOUT_UNDEFINED;s.finalLayout=VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
        VkAttachmentReference sr{0,VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};VkSubpassDescription sp{};sp.pipelineBindPoint=VK_PIPELINE_BIND_POINT_GRAPHICS;sp.pDepthStencilAttachment=&sr;
        std::array<VkSubpassDependency,2> dependencies{};
        dependencies[0]={VK_SUBPASS_EXTERNAL,0,VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT|VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,VK_ACCESS_SHADER_READ_BIT,VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,0};
        dependencies[1]={0,VK_SUBPASS_EXTERNAL,VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT|VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,VK_ACCESS_SHADER_READ_BIT,0};
        VkRenderPassCreateInfo rp{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};rp.attachmentCount=1;rp.pAttachments=&s;rp.subpassCount=1;rp.pSubpasses=&sp;rp.dependencyCount=2;rp.pDependencies=dependencies.data();
        acquire(shadow_pass,[&](auto* out){return vkCreateRenderPass(device,&rp,nullptr,out);},"shadow render pass");
        VkFramebufferCreateInfo fb{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};fb.renderPass=shadow_pass;fb.attachmentCount=1;fb.pAttachments=&shadow.view;fb.width=fb.height=shadow_size;fb.layers=1;
        acquire(shadow_frame,[&](auto* out){return vkCreateFramebuffer(device,&fb,nullptr,out);},"shadow framebuffer");
        std::array<VkAttachmentDescription,3> a{};
        a[0].format=VK_FORMAT_R8G8B8A8_SRGB;a[0].samples=samples;a[0].loadOp=VK_ATTACHMENT_LOAD_OP_CLEAR;
        a[0].storeOp=samples==VK_SAMPLE_COUNT_1_BIT?VK_ATTACHMENT_STORE_OP_STORE:VK_ATTACHMENT_STORE_OP_DONT_CARE;
        a[0].initialLayout=VK_IMAGE_LAYOUT_UNDEFINED;a[0].finalLayout=samples==VK_SAMPLE_COUNT_1_BIT?VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        a[1]=s;a[1].samples=samples;a[1].storeOp=VK_ATTACHMENT_STORE_OP_DONT_CARE;a[1].finalLayout=VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        a[2]=a[0];a[2].samples=VK_SAMPLE_COUNT_1_BIT;a[2].loadOp=VK_ATTACHMENT_LOAD_OP_DONT_CARE;a[2].storeOp=VK_ATTACHMENT_STORE_OP_STORE;a[2].finalLayout=VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        for(auto& v:a){v.stencilLoadOp=VK_ATTACHMENT_LOAD_OP_DONT_CARE;v.stencilStoreOp=VK_ATTACHMENT_STORE_OP_DONT_CARE;}
        VkAttachmentReference cr{0,VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL},dr{1,VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL},rr{2,VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
        VkSubpassDescription cp{};cp.pipelineBindPoint=VK_PIPELINE_BIND_POINT_GRAPHICS;cp.colorAttachmentCount=1;cp.pColorAttachments=&cr;cp.pDepthStencilAttachment=&dr;if(samples!=VK_SAMPLE_COUNT_1_BIT)cp.pResolveAttachments=&rr;
        dependencies[0]={VK_SUBPASS_EXTERNAL,0,VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT|VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,0,VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT|VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,0};
        dependencies[1]={0,VK_SUBPASS_EXTERNAL,VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,VK_ACCESS_TRANSFER_READ_BIT,0};
        rp.attachmentCount=samples==VK_SAMPLE_COUNT_1_BIT?2:3;rp.pAttachments=a.data();rp.pSubpasses=&cp;
        acquire(color_pass,[&](auto* out){return vkCreateRenderPass(device,&rp,nullptr,out);},"color render pass");
        std::array<VkImageView,3> views{color.view,depth.view,resolve.view};fb.renderPass=color_pass;fb.attachmentCount=rp.attachmentCount;fb.pAttachments=views.data();fb.width=width;fb.height=height;
        acquire(color_frame,[&](auto* out){return vkCreateFramebuffer(device,&fb,nullptr,out);},"color framebuffer");
    }
    void setup_descriptors(){
        VkDescriptorSetLayoutBinding b{0,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1,VK_SHADER_STAGE_FRAGMENT_BIT,nullptr};
        VkDescriptorSetLayoutCreateInfo layout{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};layout.bindingCount=1;layout.pBindings=&b;
        acquire(descriptors,[&](auto* out){return vkCreateDescriptorSetLayout(device,&layout,nullptr,out);},"shadow descriptor layout");
        VkPushConstantRange push{VK_SHADER_STAGE_VERTEX_BIT,0,64};VkPipelineLayoutCreateInfo pl{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};pl.setLayoutCount=1;pl.pSetLayouts=&descriptors;pl.pushConstantRangeCount=1;pl.pPushConstantRanges=&push;
        acquire(pipeline_layout,[&](auto* out){return vkCreatePipelineLayout(device,&pl,nullptr,out);},"pipeline layout");
        VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1};VkDescriptorPoolCreateInfo pool{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};pool.maxSets=1;pool.poolSizeCount=1;pool.pPoolSizes=&size;
        acquire(descriptor_pool,[&](auto* out){return vkCreateDescriptorPool(device,&pool,nullptr,out);},"descriptor pool");
        VkDescriptorSetAllocateInfo allocation{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};allocation.descriptorPool=descriptor_pool;allocation.descriptorSetCount=1;allocation.pSetLayouts=&descriptors;
        check(vkAllocateDescriptorSets(device,&allocation,&descriptor_set),"shadow descriptor set");
        VkSamplerCreateInfo sampler{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};sampler.magFilter=sampler.minFilter=VK_FILTER_NEAREST;sampler.mipmapMode=VK_SAMPLER_MIPMAP_MODE_NEAREST;
        sampler.addressModeU=sampler.addressModeV=sampler.addressModeW=VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;sampler.borderColor=VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
        acquire(shadow_sampler,[&](auto* out){return vkCreateSampler(device,&sampler,nullptr,out);},"shadow sampler");
        VkDescriptorImageInfo image_info{shadow_sampler,shadow.view,VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL};
        VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};write.dstSet=descriptor_set;write.dstBinding=0;write.descriptorCount=1;write.descriptorType=VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;write.pImageInfo=&image_info;
        vkUpdateDescriptorSets(device,1,&write,0,nullptr);
    }
    VkPipeline pipeline(bool is_shadow){
        std::vector<VkShaderModule> modules;
        try{
            std::vector<VkPipelineShaderStageCreateInfo> stages;
            for(auto name:is_shadow?std::vector<const char*>{"floor_shadow.vert.spv"}:std::vector<const char*>{"floor.vert.spv","floor.frag.spv"}){
                auto code=spirv(name);VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};ci.codeSize=code.size()*4;ci.pCode=code.data();VkShaderModule module{};
                check(vkCreateShaderModule(device,&ci,nullptr,&module),"floor shader module");modules.push_back(module);
                VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};stage.stage=stages.empty()?VK_SHADER_STAGE_VERTEX_BIT:VK_SHADER_STAGE_FRAGMENT_BIT;stage.module=module;stage.pName="main";stages.push_back(stage);
            }
            std::array<VkVertexInputBindingDescription,3> bindings{{{0,24,VK_VERTEX_INPUT_RATE_VERTEX},{1,64,VK_VERTEX_INPUT_RATE_INSTANCE},{2,16,VK_VERTEX_INPUT_RATE_INSTANCE}}};
            std::array<VkVertexInputAttributeDescription,7> attributes{{{0,0,VK_FORMAT_R32G32B32_SFLOAT,0},{1,0,VK_FORMAT_R32G32B32_SFLOAT,12},
                {2,1,VK_FORMAT_R32G32B32A32_SFLOAT,0},{3,1,VK_FORMAT_R32G32B32A32_SFLOAT,16},{4,1,VK_FORMAT_R32G32B32A32_SFLOAT,32},{5,1,VK_FORMAT_R32G32B32A32_SFLOAT,48},{6,2,VK_FORMAT_R32G32B32A32_SFLOAT,0}}};
            VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};vi.vertexBindingDescriptionCount=3;vi.pVertexBindingDescriptions=bindings.data();vi.vertexAttributeDescriptionCount=7;vi.pVertexAttributeDescriptions=attributes.data();
            VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};ia.topology=VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
            VkViewport viewport{0,0,static_cast<float>(is_shadow?shadow_size:width),static_cast<float>(is_shadow?shadow_size:height),0,1};VkRect2D scissor{{0,0},{is_shadow?shadow_size:width,is_shadow?shadow_size:height}};
            VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};vp.viewportCount=vp.scissorCount=1;vp.pViewports=&viewport;vp.pScissors=&scissor;
            VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};rs.polygonMode=VK_POLYGON_MODE_FILL;rs.cullMode=VK_CULL_MODE_NONE;rs.frontFace=VK_FRONT_FACE_COUNTER_CLOCKWISE;rs.lineWidth=1;
            rs.depthBiasEnable=is_shadow;rs.depthBiasConstantFactor=is_shadow?1.25f:0;rs.depthBiasSlopeFactor=is_shadow?1.75f:0;
            VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};ms.rasterizationSamples=is_shadow?VK_SAMPLE_COUNT_1_BIT:samples;
            VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};ds.depthTestEnable=ds.depthWriteEnable=VK_TRUE;ds.depthCompareOp=VK_COMPARE_OP_LESS;
            VkPipelineColorBlendAttachmentState blend{};blend.colorWriteMask=15;VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};cb.attachmentCount=is_shadow?0:1;cb.pAttachments=&blend;
            VkGraphicsPipelineCreateInfo ci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};ci.stageCount=static_cast<uint32_t>(stages.size());ci.pStages=stages.data();ci.pVertexInputState=&vi;ci.pInputAssemblyState=&ia;ci.pViewportState=&vp;ci.pRasterizationState=&rs;ci.pMultisampleState=&ms;ci.pDepthStencilState=&ds;ci.pColorBlendState=&cb;ci.layout=pipeline_layout;ci.renderPass=is_shadow?shadow_pass:color_pass;
            VkPipeline result{};auto status=vkCreateGraphicsPipelines(device,VK_NULL_HANDLE,1,&ci,nullptr,&result);
            if(status!=VK_SUCCESS){if(result)vkDestroyPipeline(device,result,nullptr);check(status,"graphics pipeline");}
            for(auto m:modules)vkDestroyShaderModule(device,m,nullptr);return result;
        }catch(...){for(auto m:modules)vkDestroyShaderModule(device,m,nullptr);throw;}
    }
    void upload(const float* v,uint32_t nv,const uint32_t* idx,uint32_t ni,const uint32_t* ranges,uint32_t nm,const uint32_t* groups,const uint32_t* im,const float* color_data,uint32_t n){
        require(!loaded&&!poisoned,"floor upload is once-only");require(v&&idx&&ranges&&groups&&im&&color_data&&nv&&ni&&nm&&n,"empty floor upload");
        require(nv<=16000000u&&ni<=48000000u&&nm<=65536u&&n<=262144u,"bounded floor upload exceeded");
        for(size_t i=0;i<static_cast<size_t>(nv)*6;++i)require(std::isfinite(v[i]),"nonfinite vertex");
        for(uint32_t i=0;i<nv;++i){const auto* a=v+size_t(i)*6+3;float length=a[0]*a[0]+a[1]*a[1]+a[2]*a[2];require(length>.998f&&length<1.002f,"non-unit vertex normal");}
        for(size_t i=0;i<static_cast<size_t>(n)*4;++i)require(std::isfinite(color_data[i])&&color_data[i]>=0&&color_data[i]<=1,"invalid linear material RGBA");
        uint64_t vertex_end=0,index_end=0,instance_end=0;
        for(uint32_t i=0;i<nm;++i){Mesh mesh{ranges[4*i],ranges[4*i+1],ranges[4*i+2],ranges[4*i+3]};Draw draw{groups[2*i],groups[2*i+1]};
            require(mesh.first_vertex==vertex_end&&mesh.first_index==index_end&&mesh.vertex_count&&mesh.index_count%3==0,"mesh ranges must be contiguous triangle lists");
            vertex_end+=mesh.vertex_count;index_end+=mesh.index_count;require(vertex_end<=nv&&index_end<=ni,"mesh range outside upload");
            require(draw.first_instance==instance_end,"draw groups must be contiguous");instance_end+=draw.instance_count;require(instance_end<=n,"draw instance range outside upload");
            for(uint32_t j=0;j<mesh.index_count;++j)require(idx[mesh.first_index+j]<mesh.vertex_count,"mesh-local index outside mesh");
            for(uint32_t j=draw.first_instance;j<instance_end;++j)require(im[j]==i&&color_data[4*size_t(j)+3]==1.f,"instance mesh/opacity differs");
            meshes.push_back(mesh);draws.push_back(draw);
        }
        require(vertex_end==nv&&index_end==ni&&instance_end==n,"incomplete upload ranges");
        buffer(vertices,size_t(nv)*24,VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);buffer(indices,size_t(ni)*4,VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
        buffer(models,size_t(n)*64,VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);buffer(colors,size_t(n)*16,VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
        std::memcpy(vertices.mapped,v,vertices.bytes);std::memcpy(indices.mapped,idx,indices.bytes);std::memcpy(colors.mapped,color_data,colors.bytes);
        instance_count=n;loaded=true;
    }
    void draw_all(){
        std::array<VkBuffer,3> buffers{vertices.handle,models.handle,colors.handle};std::array<VkDeviceSize,3> offsets{};
        vkCmdBindVertexBuffers(commands,0,3,buffers.data(),offsets.data());vkCmdBindIndexBuffer(commands,indices.handle,0,VK_INDEX_TYPE_UINT32);
        for(size_t i=0;i<meshes.size();++i)if(draws[i].instance_count)vkCmdDrawIndexed(commands,meshes[i].index_count,draws[i].instance_count,meshes[i].first_index,static_cast<int32_t>(meshes[i].first_vertex),draws[i].first_instance);
    }
    void render(const float* matrices,const float* vp,uint8_t* output){
        require(loaded&&!poisoned&&matrices&&vp&&output,"floor render unavailable");
        for(size_t i=0;i<size_t(instance_count)*16;++i)require(std::isfinite(matrices[i]),"nonfinite model transform");for(int i=0;i<16;++i)require(std::isfinite(vp[i]),"nonfinite camera");
        check(vkWaitForFences(device,1,&fence,VK_TRUE,UINT64_MAX),"wait prior frame");pending=false;
        std::memcpy(models.mapped,matrices,models.bytes);
        check(vkResetCommandBuffer(commands,0),"reset commands");VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};begin.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;check(vkBeginCommandBuffer(commands,&begin),"begin frame");
        VkMemoryBarrier host{VK_STRUCTURE_TYPE_MEMORY_BARRIER};host.srcAccessMask=VK_ACCESS_HOST_WRITE_BIT;host.dstAccessMask=VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT|VK_ACCESS_INDEX_READ_BIT;
        vkCmdPipelineBarrier(commands,VK_PIPELINE_STAGE_HOST_BIT,VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,0,1,&host,0,nullptr,0,nullptr);
        VkClearValue shadow_clear{};shadow_clear.depthStencil={1,0};VkRenderPassBeginInfo pass{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};pass.renderPass=shadow_pass;pass.framebuffer=shadow_frame;pass.renderArea={{0,0},{shadow_size,shadow_size}};pass.clearValueCount=1;pass.pClearValues=&shadow_clear;
        vkCmdBeginRenderPass(commands,&pass,VK_SUBPASS_CONTENTS_INLINE);vkCmdBindPipeline(commands,VK_PIPELINE_BIND_POINT_GRAPHICS,shadow_pipeline);draw_all();vkCmdEndRenderPass(commands);
        std::array<VkClearValue,3> clear{};clear[0].color={{.46f,.55f,.59f,1}};clear[1].depthStencil={1,0};
        pass.renderPass=color_pass;pass.framebuffer=color_frame;pass.renderArea.extent={width,height};pass.clearValueCount=samples==VK_SAMPLE_COUNT_1_BIT?2:3;pass.pClearValues=clear.data();
        vkCmdBeginRenderPass(commands,&pass,VK_SUBPASS_CONTENTS_INLINE);vkCmdBindPipeline(commands,VK_PIPELINE_BIND_POINT_GRAPHICS,color_pipeline);
        vkCmdBindDescriptorSets(commands,VK_PIPELINE_BIND_POINT_GRAPHICS,pipeline_layout,0,1,&descriptor_set,0,nullptr);vkCmdPushConstants(commands,pipeline_layout,VK_SHADER_STAGE_VERTEX_BIT,0,64,vp);draw_all();vkCmdEndRenderPass(commands);
        VkBufferImageCopy copy{};copy.imageSubresource={VK_IMAGE_ASPECT_COLOR_BIT,0,0,1};copy.imageExtent={width,height,1};
        vkCmdCopyImageToBuffer(commands,samples==VK_SAMPLE_COUNT_1_BIT?color.handle:resolve.handle,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,readback.handle,1,&copy);
        VkBufferMemoryBarrier transfer{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};transfer.srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT;transfer.dstAccessMask=VK_ACCESS_HOST_READ_BIT;transfer.srcQueueFamilyIndex=transfer.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;transfer.buffer=readback.handle;transfer.size=readback.bytes;
        vkCmdPipelineBarrier(commands,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_HOST_BIT,0,0,nullptr,1,&transfer,0,nullptr);check(vkEndCommandBuffer(commands),"end frame");
        check(vkResetFences(device,1,&fence),"reset frame fence");VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};submit.commandBufferCount=1;submit.pCommandBuffers=&commands;
        pending=true;check(vkQueueSubmit(queue,1,&submit,fence),"submit floor frame");check(vkWaitForFences(device,1,&fence,VK_TRUE,UINT64_MAX),"wait floor frame");pending=false;++submissions;
        std::memcpy(output,readback.mapped,readback.bytes);
    }
    const char* information(){
        std::ostringstream s;s<<"{\"backend\":\"vulkan-indexed-instanced\",\"device_name\":\"";
        for(char c:std::string(properties.deviceName)){if(c=='"'||c=='\\')s<<'\\';if(static_cast<unsigned char>(c)>=32)s<<c;}
        s<<"\",\"device_type\":"<<properties.deviceType<<",\"vendor_id\":"<<properties.vendorID<<",\"api_version\":"<<properties.apiVersion
         <<",\"submissions\":"<<submissions<<",\"instance_count\":"<<instance_count<<",\"mesh_count\":"<<meshes.size()
         <<",\"color_samples\":"<<static_cast<uint32_t>(samples)<<",\"shadow_size\":"<<shadow_size<<",\"shadow_samples\":1,\"format\":\"R8G8B8A8_SRGB\",\"poisoned\":"<<(poisoned?"true":"false")
         <<",\"validation_layer\":false,\"depth\":\"D32_SFLOAT\",\"readback_bytes_per_frame\":"<<readback.bytes<<"}";info=s.str();return info.c_str();
    }
};
}
API void* floor_create(uint32_t index,uint32_t width,uint32_t height){try{error_text.clear();auto p=std::make_unique<Floor>();p->init(index,width,height);return p.release();}catch(const std::exception& e){error_text=e.what();return nullptr;}}
API int floor_upload(void* handle,const float* vertices,uint32_t nv,const uint32_t* indices,uint32_t ni,const uint32_t* meshes,uint32_t nm,const uint32_t* draws,const uint32_t* instance_mesh,const float* colors,uint32_t n){
    try{require(handle,"null floor owner");static_cast<Floor*>(handle)->upload(vertices,nv,indices,ni,meshes,nm,draws,instance_mesh,colors,n);return 0;}catch(const std::exception& e){error_text=e.what();if(handle)static_cast<Floor*>(handle)->poisoned=true;return -1;}}
API int floor_render(void* handle,const float* matrices,const float* view_projection,uint8_t* rgba){try{require(handle,"null floor owner");static_cast<Floor*>(handle)->render(matrices,view_projection,rgba);return 0;}catch(const std::exception& e){error_text=e.what();if(handle)static_cast<Floor*>(handle)->poisoned=true;return -1;}}
API const char* floor_info(void* handle){try{require(handle,"null floor owner");return static_cast<Floor*>(handle)->information();}catch(const std::exception& e){error_text=e.what();return nullptr;}}
API const char* floor_error(){return error_text.c_str();}
API void floor_destroy(void* handle){delete static_cast<Floor*>(handle);}
