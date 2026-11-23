// Included inside renderer.cpp's private namespace. The ABI2 renderer is kept
// separate; only explicitly resident contexts use these persistent resources.
struct GpuResidentMesh {
    uint32_t first_vertex, vertex_count, first_index, index_count;
    uint32_t root, node_count, reserved[2];
};
struct GpuBvhNode {
    float minimum[4], maximum[4];
    uint32_t data[4]; // first global triangle, leaf count (0 = branch), escape, 0
};
struct GpuTransform { float world[16], local[16]; };
static_assert(sizeof(GpuResidentMesh) == 32, "resident mesh shader layout");
static_assert(sizeof(GpuBvhNode) == 48, "resident BVH shader layout");
static_assert(sizeof(GpuTransform) == 128, "resident transform shader layout");

struct ResidentTextures {
    Context& c;
    Image image{};
    Buffer metadata{};
    VkSampler sampler{};
    size_t metadata_bytes{};
    explicit ResidentTextures(Context& context):c(context) {}
    ~ResidentTextures() {
        release(sampler,[&](auto h){vkDestroySampler(c.device,h,nullptr);});
        release(image.view,[&](auto h){vkDestroyImageView(c.device,h,nullptr);});
        release(image.handle,[&](auto h){vkDestroyImage(c.device,h,nullptr);});
        c.free_memory(image.memory,image.allocation);
        release(metadata.handle,[&](auto h){vkDestroyBuffer(c.device,h,nullptr);});
        c.free_memory(metadata.memory,metadata.allocation);
    }
};
struct ResidentEnvironment {
    Context& c;
    Image image{};
    Buffer uniform{};
    VkSampler sampler{};
    VkPipeline sky_pipeline{};
    VkShaderModule sky_vertex{},sky_fragment{};
    bool display_compatible=false;
    bool msaa_compatible=false;
    explicit ResidentEnvironment(Context& context):c(context) {}
    ~ResidentEnvironment() {
        release(sky_pipeline,[&](auto h){vkDestroyPipeline(c.device,h,nullptr);});
        release(sky_vertex,[&](auto h){vkDestroyShaderModule(c.device,h,nullptr);});
        release(sky_fragment,[&](auto h){vkDestroyShaderModule(c.device,h,nullptr);});
        release(sampler,[&](auto h){vkDestroySampler(c.device,h,nullptr);});
        release(image.view,[&](auto h){vkDestroyImageView(c.device,h,nullptr);});
        release(image.handle,[&](auto h){vkDestroyImage(c.device,h,nullptr);});
        c.free_memory(image.memory,image.allocation);
        release(uniform.handle,[&](auto h){vkDestroyBuffer(c.device,h,nullptr);});
        c.free_memory(uniform.memory,uniform.allocation);
    }
};
struct AccelerationObject {
    Context& c;
    Buffer storage{};
    VkAccelerationStructureKHR handle{};
    VkDeviceAddress address{};
    explicit AccelerationObject(Context& context):c(context) {}
    ~AccelerationObject() {
        release(handle,[&](auto h){c.ray.destroy(c.device,h,nullptr);});
        release(storage.handle,[&](auto h){vkDestroyBuffer(c.device,h,nullptr);});
        c.free_memory(storage.memory,storage.allocation);
    }
};
struct ResidentAcceleration {
    Context& c;
    std::vector<std::unique_ptr<AccelerationObject>> bottom;
    std::unique_ptr<AccelerationObject> top;
    Buffer scratch{},instances{};
    VkDeviceAddress scratch_address{},instances_address{};
    bool built=false;
    explicit ResidentAcceleration(Context& context):c(context) {}
    ~ResidentAcceleration() {
        top.reset(); bottom.clear();
        for (auto* buffer:{&scratch,&instances}) {
            release(buffer->handle,[&](auto h){vkDestroyBuffer(c.device,h,nullptr);});
            c.free_memory(buffer->memory,buffer->allocation);
        }
    }
};
struct ResidentState {
    Context& c;
    Buffer vertices{}, indices{}, meshes_buffer{}, instances_buffer{}, nodes{}, transforms{}, tlas{}, studio{};
    std::vector<GpuResidentMesh> meshes;
    std::vector<PrResidentInstance> instances;
    std::vector<std::array<float, 6>> bounds;
    std::unique_ptr<FrameResources> target;
    std::unique_ptr<ResidentTextures> textures;
    std::unique_ptr<ResidentTextures> normal_maps;
    std::vector<uint32_t> normal_validation_indices;
    std::unique_ptr<ResidentEnvironment> environment;
    std::unique_ptr<ResidentAcceleration> acceleration;
    VkDescriptorSet descriptor_set{};
    uint32_t width{}, height{};
    explicit ResidentState(Context& context) : c(context) {}
    ~ResidentState() {
        target.reset();
        textures.reset();
        normal_maps.reset();
        environment.reset();
        acceleration.reset();
        for (auto* buffer : {&vertices, &indices, &meshes_buffer, &instances_buffer, &nodes, &transforms, &tlas, &studio}) {
            release(buffer->handle, [&](auto h) { vkDestroyBuffer(c.device, h, nullptr); });
            c.free_memory(buffer->memory, buffer->allocation);
        }
    }
};

struct TriangleBounds {
    uint32_t source;
    std::array<float, 3> low, high, center;
};

uint32_t build_bvh(const PrResidentVertex* vertices, const uint32_t* indices,
                   const PrResidentMesh& mesh, std::vector<GpuBvhNode>& nodes,
                   std::vector<uint32_t>& reordered, uint32_t node_limit) {
    const uint32_t count = mesh.index_count / 3;
    std::vector<TriangleBounds> triangles(count);
    for (uint32_t t = 0; t < count; ++t) {
        auto& triangle = triangles[t];
        triangle.source = t;
        const auto* a = vertices[mesh.first_vertex + indices[mesh.first_index + 3*t]].position;
        const auto* b = vertices[mesh.first_vertex + indices[mesh.first_index + 3*t + 1]].position;
        const auto* d = vertices[mesh.first_vertex + indices[mesh.first_index + 3*t + 2]].position;
        double u[3], v[3];
        for (uint32_t k = 0; k < 3; ++k) {
            triangle.low[k] = std::min({a[k], b[k], d[k]});
            triangle.high[k] = std::max({a[k], b[k], d[k]});
            triangle.center[k] = static_cast<float>(
                (static_cast<double>(triangle.low[k]) + triangle.high[k]) * 0.5);
            u[k] = static_cast<double>(b[k]) - a[k];
            v[k] = static_cast<double>(d[k]) - a[k];
        }
        double area = std::hypot(u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0]);
        require(std::isfinite(area) && area > 0, "resident triangle is degenerate");
    }
    std::function<uint32_t(uint32_t,uint32_t,uint32_t)> build;
    build = [&](uint32_t begin, uint32_t end, uint32_t depth) -> uint32_t {
        require(depth <= 32 && nodes.size() < node_limit, "resident BVH capacity exceeded");
        uint32_t index = static_cast<uint32_t>(nodes.size());
        nodes.emplace_back();
        GpuBvhNode node{};
        std::array<float, 3> center_low = triangles[begin].center;
        std::array<float, 3> center_high = center_low;
        for (uint32_t k = 0; k < 3; ++k) {
            node.minimum[k] = triangles[begin].low[k];
            node.maximum[k] = triangles[begin].high[k];
        }
        for (uint32_t t = begin + 1; t < end; ++t)
            for (uint32_t k = 0; k < 3; ++k) {
                node.minimum[k] = std::min(node.minimum[k], triangles[t].low[k]);
                node.maximum[k] = std::max(node.maximum[k], triangles[t].high[k]);
                center_low[k] = std::min(center_low[k], triangles[t].center[k]);
                center_high[k] = std::max(center_high[k], triangles[t].center[k]);
            }
        if (end - begin <= 8) {
            node.data[0] = mesh.first_index / 3 + begin;
            node.data[1] = end - begin;
        } else {
            uint32_t axis = 0;
            for (uint32_t k = 1; k < 3; ++k)
                if (static_cast<double>(center_high[k]) - center_low[k] >
                    static_cast<double>(center_high[axis]) - center_low[axis]) axis = k;
            uint32_t middle = begin + (end - begin) / 2;
            std::nth_element(triangles.begin()+begin, triangles.begin()+middle, triangles.begin()+end,
                [axis](const auto& a, const auto& b) {
                    return a.center[axis] < b.center[axis] ||
                           (a.center[axis] == b.center[axis] && a.source < b.source);
                });
            build(begin, middle, depth+1);
            build(middle, end, depth+1);
        }
        node.data[2] = static_cast<uint32_t>(nodes.size());
        nodes[index] = node; // recursive vector growth must not invalidate an owner reference
        return index;
    };
    uint32_t root = build(0, count, 0);
    for (uint32_t t = 0; t < count; ++t)
        for (uint32_t k = 0; k < 3; ++k)
            reordered[mesh.first_index + 3*t + k] =
                indices[mesh.first_index + 3*triangles[t].source + k];
    return root;
}

struct StaticUpload {
    Buffer* destination;
    const void* data;
    size_t bytes;
    VkBufferUsageFlags usage;
};

using HostClock = std::chrono::steady_clock;
uint64_t elapsed_ns(HostClock::time_point begin, HostClock::time_point end = HostClock::now()) {
    return static_cast<uint64_t>(std::max<int64_t>(0,
        std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count()));
}
struct SubmitTiming { uint64_t submit, wait; };
SubmitTiming submit_and_wait(Context& c, FrameResources& frame) {
    auto submitted_at = HostClock::now();
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1; submit.pCommandBuffers = &frame.commands;
    VkResult result = vkQueueSubmit(c.queue, 1, &submit, frame.fence);
    frame.submitted = result == VK_SUCCESS || result == VK_ERROR_DEVICE_LOST;
    if (result != VK_SUCCESS) { c.poisoned = true; check(result, "resident vkQueueSubmit"); }
    uint64_t submit_ns = elapsed_ns(submitted_at);
    ++c.telemetry.submissions; // only successful vkQueueSubmit calls
    auto before = std::chrono::steady_clock::now();
    result = vkWaitForFences(c.device, 1, &frame.fence, VK_TRUE, UINT64_MAX);
    auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - before).count();
    c.telemetry.fence_wait_wall_ns += static_cast<uint64_t>(std::max<int64_t>(0, elapsed));
    ++c.telemetry.fence_wait_calls;
    if (result != VK_SUCCESS) { c.poisoned = true; check(result, "resident vkWaitForFences"); }
    frame.submitted = false;
    return {submit_ns, static_cast<uint64_t>(std::max<int64_t>(0, elapsed))};
}

void make_commands(Context& c, FrameResources& frame) {
    VkCommandBufferAllocateInfo allocate{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    allocate.commandPool = c.command_pool;
    allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocate.commandBufferCount = 1;
    acquire(frame.commands, "resident command allocation", [&](auto* out) {
        return vkAllocateCommandBuffers(c.device, &allocate, out);
    });
    VkFenceCreateInfo fence{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    acquire(frame.fence, "resident fence", [&](auto* out) {
        return vkCreateFence(c.device, &fence, nullptr, out);
    });
}

uint64_t upload_static(Context& c, const std::vector<StaticUpload>& uploads) {
    size_t total = 0;
    for (const auto& upload : uploads) {
        require(upload.bytes > 0 && upload.bytes % 4 == 0, "resident static buffer alignment");
        c.make_buffer(*upload.destination, upload.bytes,
            upload.usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        total += upload.bytes;
    }
    FrameResources staging(c);
    c.make_buffer(staging.vertices, total, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    void* mapped{};
    check(vkMapMemory(c.device, staging.vertices.memory, 0, total, 0, &mapped), "resident stage map");
    size_t offset = 0;
    for (const auto& upload : uploads) {
        std::memcpy(static_cast<uint8_t*>(mapped)+offset, upload.data, upload.bytes);
        offset += upload.bytes;
    }
    vkUnmapMemory(c.device, staging.vertices.memory);
    make_commands(c, staging);
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    check(vkBeginCommandBuffer(staging.commands, &begin), "resident stage begin");
    std::vector<VkBufferMemoryBarrier> barriers;
    offset = 0;
    for (const auto& upload : uploads) {
        VkBufferCopy copy{offset, 0, upload.bytes};
        vkCmdCopyBuffer(staging.commands, staging.vertices.handle, upload.destination->handle, 1, &copy);
        VkBufferMemoryBarrier barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        if (upload.usage & VK_BUFFER_USAGE_VERTEX_BUFFER_BIT) barrier.dstAccessMask |= VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT;
        if (upload.usage & VK_BUFFER_USAGE_INDEX_BUFFER_BIT) barrier.dstAccessMask |= VK_ACCESS_INDEX_READ_BIT;
        barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.buffer = upload.destination->handle; barrier.size = VK_WHOLE_SIZE;
        barriers.push_back(barrier);
        offset += upload.bytes;
    }
    vkCmdPipelineBarrier(staging.commands, VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_VERTEX_INPUT_BIT | VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr,
        static_cast<uint32_t>(barriers.size()), barriers.data(), 0, nullptr);
    check(vkEndCommandBuffer(staging.commands), "resident stage end");
    submit_and_wait(c, staging);
    return total;
}

#include "resident_textures.inc"
#include "resident_normal.inc"
#include "resident_environment.inc"
#include "resident_rayquery.inc"
#include "resident_msaa.inc"

void resident_upload(Context& c, const PrResidentVertex* vertices, uint32_t vertex_count,
                     const uint32_t* indices, uint32_t index_count,
                     const PrResidentMesh* meshes, uint32_t mesh_count,
                     const PrResidentInstance* instances, uint32_t instance_count) {
    require(c.resident, "resident upload requires an explicit resident context");
    require(!c.poisoned && !c.resident_state, "resident scene already loaded or context poisoned");
    auto& info = c.resident_properties;
    require(vertex_count <= info.max_vertices && index_count <= info.max_indices &&
            mesh_count <= info.max_meshes && instance_count <= info.max_instances,
            "resident scene capacity exceeded");
    require((vertex_count == 0 || vertices) && (index_count == 0 || indices) &&
            (mesh_count == 0 || meshes) && (instance_count == 0 || instances),
            "null resident static data");
    require(index_count % 3 == 0, "resident triangle index count mismatch");
    if (c.ray_query) require(mesh_count<=c.ray_info.max_meshes && instance_count<=c.ray_info.max_instances &&
        index_count/3<=c.ray_info.max_triangles,"ray-query scene capacity exceeded");
    require(mesh_count != 0 || (vertex_count == 0 && index_count == 0 && instance_count == 0),
            "empty resident scene has orphan data");
    for (uint32_t i = 0; i < vertex_count; ++i) {
        const auto& vertex = vertices[i];
        for (uint32_t k = 0; k < 4; ++k)
            require(std::isfinite(vertex.position[k]) && std::isfinite(vertex.normal[k]),
                    "resident vertex FP32 conversion is nonfinite");
        require(vertex.position[3] == 1 && vertex.normal[3] == 0, "resident homogeneous vertex layout");
        double length = std::hypot(static_cast<double>(vertex.normal[0]),
            static_cast<double>(vertex.normal[1]), static_cast<double>(vertex.normal[2]));
        require(length >= 0.999 && length <= 1.001, "resident vertex normal must be unit length");
    }
    auto state = std::make_shared<ResidentState>(c);
    uint32_t next_vertex = 0, next_index = 0;
    for (uint32_t i = 0; i < mesh_count; ++i) {
        const auto& mesh = meshes[i];
        require(mesh.first_vertex == next_vertex && mesh.first_index == next_index &&
                mesh.vertex_count >= 3 && mesh.index_count >= 3 && mesh.index_count % 3 == 0 &&
                mesh.vertex_count <= vertex_count - next_vertex &&
                mesh.index_count <= index_count - next_index, "resident mesh ranges are not tightly packed");
        for (uint32_t j = 0; j < mesh.index_count; ++j)
            require(indices[mesh.first_index+j] < mesh.vertex_count, "resident mesh-local index out of range");
        next_vertex += mesh.vertex_count;
        next_index += mesh.index_count;
    }
    require(next_vertex == vertex_count && next_index == index_count, "resident unused vertex/index tail");
    std::unordered_set<uint32_t> identifiers;
    uint64_t instanced_triangles = 0;
    for (uint32_t i = 0; i < instance_count; ++i) {
        const auto& instance = instances[i];
        require(instance.mesh_index < mesh_count && instance.instance_id > 0 &&
                identifiers.insert(instance.instance_id).second &&
                instance.reserved[0] == 0 && instance.reserved[1] == 0, "resident instance identity mismatch");
        for (uint32_t k = 0; k < 4; ++k)
            require(std::isfinite(instance.color_roughness[k]) && instance.color_roughness[k] >= 0 &&
                    instance.color_roughness[k] <= 1 && std::isfinite(instance.material[k]),
                    "resident material must be finite in its declared range");
        require(instance.material[0] >= 0 && instance.material[0] <= 1 &&
                instance.material[1] == 0 && instance.material[2] == 0 && instance.material[3] == 0,
                "resident metallic/reserved material mismatch");
        instanced_triangles += meshes[instance.mesh_index].index_count / 3;
    }
    require(instanced_triangles <= info.max_instanced_triangles, "resident instanced triangle budget exceeded");
    std::vector<GpuBvhNode> nodes;
    std::vector<uint32_t> reordered(index_count);
    for (uint32_t i = 0; i < mesh_count; ++i) {
        const auto& mesh = meshes[i];
        uint32_t root = build_bvh(vertices, indices, mesh, nodes, reordered, info.max_bvh_nodes);
        uint32_t node_count = static_cast<uint32_t>(nodes.size()) - root;
        state->meshes.push_back({mesh.first_vertex, mesh.vertex_count, mesh.first_index, mesh.index_count,
                                 root, node_count, {0, 0}});
        const auto& box = nodes[root];
        state->bounds.push_back({box.minimum[0], box.minimum[1], box.minimum[2],
                                 box.maximum[0], box.maximum[1], box.maximum[2]});
    }
    if (instance_count) state->instances.assign(instances, instances+instance_count);
    if (c.normal_info.shader_compatible) state->normal_validation_indices=reordered;
    uint64_t node_budget = 0;
    for (const auto& instance : state->instances)
        node_budget += state->meshes[instance.mesh_index].node_count;
    PrResidentVertex empty_vertex{};
    uint32_t empty_index{};
    PrResidentInstance empty_instance{};
    GpuResidentMesh empty_mesh{};
    GpuBvhNode empty_node{};
    uint64_t errors_before = c.errors.load();
    std::vector<StaticUpload> uploads{
        {&state->vertices, vertex_count ? static_cast<const void*>(vertices) : &empty_vertex,
         std::max<size_t>(1, vertex_count)*sizeof(PrResidentVertex),
         VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT},
        {&state->indices, index_count ? static_cast<const void*>(reordered.data()) : &empty_index,
         std::max<size_t>(1, index_count)*sizeof(uint32_t),
         VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT},
        {&state->meshes_buffer, mesh_count ? static_cast<const void*>(state->meshes.data()) : &empty_mesh,
         std::max<size_t>(1, mesh_count)*sizeof(GpuResidentMesh), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT},
        {&state->instances_buffer, instance_count ? static_cast<const void*>(instances) : &empty_instance,
         std::max<size_t>(1, instance_count)*sizeof(PrResidentInstance), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT},
        {&state->nodes, nodes.empty() ? static_cast<const void*>(&empty_node) : nodes.data(),
         std::max<size_t>(1, nodes.size())*sizeof(GpuBvhNode), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT}
    };
    if (c.ray_query)
        for (uint32_t i=0;i<2;++i) uploads[i].usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
    uint64_t bytes = 0;
    try {
    bytes = upload_static(c, uploads);
    c.make_buffer(state->tlas, std::max<size_t>(1, instance_count ? 2ull*instance_count-1 : 0)*sizeof(GpuBvhNode),
                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    c.make_buffer(state->studio, sizeof(PrStudio), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    std::vector<uint32_t> plain_metadata(4ull+instance_count,UINT32_MAX);
    plain_metadata[0]=0; plain_metadata[1]=instance_count; plain_metadata[2]=vertex_count;
    plain_metadata[3]=static_cast<uint32_t>(plain_metadata.size());
    state->textures=upload_texture_resource(c,1,1,1,std::vector<uint8_t>{255,255,255,255},plain_metadata);
    auto& texture_info=c.texture_info;
    texture_info.array_width=texture_info.array_height=texture_info.array_layers=1;
    texture_info.padded_rgba_bytes=4; texture_info.image_upload_bytes+=4;
    texture_info.metadata_upload_bytes+=plain_metadata.size()*sizeof(uint32_t);
    if (c.normal_info.shader_compatible) {
        state->normal_maps=upload_texture_resource(c,1,1,1,std::vector<uint8_t>{128,128,255,255},plain_metadata,VK_FORMAT_R8G8B8A8_UNORM);
        c.normal_info.array_width=c.normal_info.array_height=c.normal_info.array_layers=1;
        c.normal_info.padded_rgba_bytes=4; c.normal_info.image_upload_bytes+=4;
        c.normal_info.metadata_upload_bytes+=plain_metadata.size()*sizeof(uint32_t);
    }
    PrResidentEnvironment default_environment{};
    default_environment.side=1; default_environment.mip_levels=1;
    const std::array<uint8_t,24> black_cube{0,0,0,255,0,0,0,255,0,0,0,255,
                                          0,0,0,255,0,0,0,255,0,0,0,255};
    state->environment=upload_environment_resource(c,default_environment,black_cube.data(),
        black_cube.size(),VK_FORMAT_R8G8B8A8_SRGB,false);
    c.environment_info.image_upload_bytes+=black_cube.size();
    c.environment_info.parameter_upload_bytes+=sizeof(default_environment);
    if (c.ray_query) state->acceleration=build_ray_geometry(c,*state,index_count/3);
    c.make_buffer(state->transforms, std::max<size_t>(1, instance_count)*sizeof(GpuTransform),
                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    if (c.errors.load() != errors_before) {
        c.poisoned = true;
        throw std::runtime_error("Vulkan validation failed during resident upload: "+c.validation_message);
    }
    } catch (...) { c.poisoned = true; throw; }
    info.vertex_count = vertex_count; info.index_count = index_count;
    info.mesh_count = mesh_count; info.instance_count = instance_count;
    info.bvh_nodes = static_cast<uint32_t>(nodes.size());
    info.instanced_triangles = static_cast<uint32_t>(instanced_triangles);
    info.loaded = 1; ++info.upload_calls; info.geometry_upload_bytes += bytes;
    if (c.ray_query) c.ray_info.loaded=1;
    c.profile.tlas_node_count = !c.ray_query && instance_count ? 2*instance_count-1 : 0;
    info.shadow_node_budget = c.ray_query ? 0 : node_budget + c.profile.tlas_node_count;
    c.resident_state = std::move(state);
}

std::vector<GpuTransform> resident_transforms(const ResidentState& state, const float* matrices,
                                             uint32_t count) {
    require(count == state.instances.size() && (count == 0 || matrices),
            "resident transform count mismatch");
    std::vector<GpuTransform> output(count);
    for (uint32_t i = 0; i < count; ++i) {
        auto& transform = output[i];
        const float* source = matrices + 16ull*i;
        for (uint32_t k = 0; k < 16; ++k)
            require(std::isfinite(source[k]), "resident transform FP32 conversion is nonfinite");
        require(source[3] == 0 && source[7] == 0 && source[11] == 0 && source[15] == 1,
                "resident matrices must be affine rigid transforms");
        for (uint32_t column = 0; column < 3; ++column)
            for (uint32_t other = 0; other < 3; ++other) {
                double dot = 0;
                for (uint32_t row = 0; row < 3; ++row)
                    dot += static_cast<double>(source[4*column+row])*source[4*other+row];
                require(std::abs(dot - (column == other ? 1.0 : 0.0)) <= 1e-4,
                        "resident matrix scale/shear is unsupported");
            }
        double determinant =
            source[0]*(static_cast<double>(source[5])*source[10]-static_cast<double>(source[9])*source[6]) -
            source[4]*(static_cast<double>(source[1])*source[10]-static_cast<double>(source[9])*source[2]) +
            source[8]*(static_cast<double>(source[1])*source[6]-static_cast<double>(source[5])*source[2]);
        require(determinant > 0, "resident reflected transforms are unsupported");
        std::memcpy(transform.world, source, 64);
        // Invert the admitted FP32 matrix itself instead of assuming that its
        // rounded rotation is exactly orthogonal. Primary/shadow geometry then
        // agrees on the same transform, including the final FP32 rounding.
        double columns[3][3]{};
        for (uint32_t column = 0; column < 3; ++column)
            for (uint32_t row = 0; row < 3; ++row)
                columns[column][row] = source[4*column+row];
        for (uint32_t row = 0; row < 3; ++row) {
            const auto& a = columns[(row+1)%3];
            const auto& b = columns[(row+2)%3];
            const double cross[3]{a[1]*b[2]-a[2]*b[1],
                                  a[2]*b[0]-a[0]*b[2],
                                  a[0]*b[1]-a[1]*b[0]};
            for (uint32_t column = 0; column < 3; ++column)
                transform.local[4*column+row] = static_cast<float>(cross[column]/determinant);
        }
        transform.local[15] = 1;
        for (uint32_t row = 0; row < 3; ++row) {
            double translation = 0;
            for (uint32_t k = 0; k < 3; ++k)
                translation -= static_cast<double>(transform.local[4*k+row])*source[12+k];
            transform.local[12+row] = static_cast<float>(translation);
            require(std::isfinite(transform.local[12+row]), "resident inverse translation overflow");
        }
        const auto& bound = state.bounds[state.instances[i].mesh_index];
        for (uint32_t corner = 0; corner < 8; ++corner)
            for (uint32_t row = 0; row < 3; ++row) {
                double point = source[12+row];
                for (uint32_t column = 0; column < 3; ++column)
                    point += static_cast<double>(source[4*column+row]) *
                             bound[column+((corner & (1u<<column)) ? 3u : 0u)];
                require(std::isfinite(static_cast<float>(point)), "resident transformed bounds overflow");
            }
    }
    return output;
}



// A small balanced top-level tree over the *current* instance boxes. Leaves
// retain upload-order instance indices. No triangle is dropped or simplified.
// Boxes include both forward raster and the inverse-matrix shadow geometry.
std::vector<GpuBvhNode> resident_tlas(const ResidentState& state,
                                    const std::vector<GpuTransform>& transforms) {
    std::vector<TriangleBounds> boxes(transforms.size());
    for (uint32_t slot = 0; slot < transforms.size(); ++slot) {
        auto& box = boxes[slot]; box.source = slot;
        const auto& transform = transforms[slot];
        const auto& bounds = state.bounds[state.instances[slot].mesh_index];
        // Invert the actual rounded inverse used by the shadow shader. This
        // also encloses its represented surface when inverse(round(M)) != M.
        const float* local = transform.local;
        double columns[3][3]{};
        for (uint32_t col=0; col<3; ++col)
            for (uint32_t row=0; row<3; ++row) columns[col][row] = local[4*col+row];
        double determinant = columns[0][0]*(columns[1][1]*columns[2][2]-columns[2][1]*columns[1][2]) -
            columns[1][0]*(columns[0][1]*columns[2][2]-columns[2][1]*columns[0][2]) +
            columns[2][0]*(columns[0][1]*columns[1][2]-columns[1][1]*columns[0][2]);
        require(std::isfinite(determinant) && determinant > 0, "resident shadow inverse is singular");
        double inverse[3][3]{}, translation[3]{};
        for (uint32_t row=0; row<3; ++row) {
            const auto& a = columns[(row+1)%3]; const auto& b = columns[(row+2)%3];
            const double cross[3]{a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]};
            for (uint32_t col=0; col<3; ++col) {
                inverse[row][col] = cross[col]/determinant;
                translation[row] -= inverse[row][col]*local[12+col];
            }
        }
        for (uint32_t row=0; row<3; ++row) {
            double low = std::numeric_limits<double>::infinity(), high = -low, magnitude = 0;
            for (uint32_t corner=0; corner<8; ++corner) {
                double a = transform.world[12+row], b = translation[row];
                double scale_a = std::abs(a), scale_b = std::abs(b);
                for (uint32_t col=0; col<3; ++col) {
                    double point = bounds[col+((corner & (1u<<col)) ? 3u : 0u)];
                    double term_a = static_cast<double>(transform.world[4*col+row])*point;
                    double term_b = inverse[row][col]*point;
                    a += term_a; b += term_b;
                    scale_a += std::abs(term_a); scale_b += std::abs(term_b);
                }
                low = std::min({low,a,b}); high = std::max({high,a,b});
                magnitude = std::max({magnitude,scale_a,scale_b});
            }
            // Culling-only outward padding for finite FP32 affine evaluation.
            // It never moves vertices, changes t, or produces a shadow hit.
            double padding = 32.0*std::numeric_limits<float>::epsilon()*magnitude;
            box.low[row] = std::nextafter(static_cast<float>(low-padding), -std::numeric_limits<float>::infinity());
            box.high[row] = std::nextafter(static_cast<float>(high+padding), std::numeric_limits<float>::infinity());
            require(std::isfinite(box.low[row]) && std::isfinite(box.high[row]), "resident TLAS bounds overflow");
            box.center[row] = static_cast<float>((static_cast<double>(box.low[row])+box.high[row])*0.5);
        }
    }
    std::vector<GpuBvhNode> output;
    if (boxes.empty()) return output;
    output.reserve(2*boxes.size()-1);
    std::function<void(uint32_t,uint32_t)> build;
    build = [&](uint32_t begin, uint32_t end) {
        uint32_t index = static_cast<uint32_t>(output.size());
        output.emplace_back(); GpuBvhNode node{};
        auto center_low = boxes[begin].center, center_high = center_low;
        for (uint32_t k=0; k<3; ++k) { node.minimum[k]=boxes[begin].low[k]; node.maximum[k]=boxes[begin].high[k]; }
        for (uint32_t i=begin+1; i<end; ++i) for (uint32_t k=0; k<3; ++k) {
            node.minimum[k]=std::min(node.minimum[k],boxes[i].low[k]);
            node.maximum[k]=std::max(node.maximum[k],boxes[i].high[k]);
            center_low[k]=std::min(center_low[k],boxes[i].center[k]);
            center_high[k]=std::max(center_high[k],boxes[i].center[k]);
        }
        if (end-begin == 1) { node.data[0]=boxes[begin].source; node.data[1]=1; }
        else {
            uint32_t axis=0;
            for (uint32_t k=1; k<3; ++k)
                if (static_cast<double>(center_high[k])-center_low[k] >
                    static_cast<double>(center_high[axis])-center_low[axis]) axis=k;
            uint32_t middle=begin+(end-begin)/2;
            std::nth_element(boxes.begin()+begin,boxes.begin()+middle,boxes.begin()+end,
                [axis](const auto& a,const auto& b) { return a.center[axis]<b.center[axis] ||
                    (a.center[axis]==b.center[axis] && a.source<b.source); });
            build(begin,middle); build(middle,end);
        }
        node.data[2]=static_cast<uint32_t>(output.size()); output[index]=node;
    };
    build(0,static_cast<uint32_t>(boxes.size()));
    return output;
}

void resident_target(Context& c, ResidentState& state, uint32_t width, uint32_t height) {
    if (state.target && state.width == width && state.height == height) return;
    if (c.msaa_state) c.msaa_state->target.reset();
    const auto& info = c.resident_properties;
    uint64_t errors_before = c.errors.load();
    auto frame = std::make_unique<FrameResources>(c);
    c.make_buffer(frame->uniform, sizeof(PrParams), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    frame->readback_capacity=static_cast<size_t>(width)*height*13;
    c.make_buffer(frame->readback,frame->readback_capacity,VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                  VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
    void* readback_mapping=nullptr;
    check(vkMapMemory(c.device,frame->readback.memory,0,VK_WHOLE_SIZE,0,&readback_mapping),
          "resident persistent readback map");
    frame->readback_mapping=readback_mapping; // own a mapping only after VK_SUCCESS
    ++c.readback_info.map_calls;
    std::array<VkImageView, 5> views{};
    for (uint32_t i = 0; i < 5; ++i) {
        c.make_image(frame->images[i], width, height, FORMATS[i], i == 4);
        views[i] = frame->images[i].view;
    }
    VkFramebufferCreateInfo framebuffer{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
    framebuffer.renderPass = c.render_pass; framebuffer.attachmentCount = 5;
    framebuffer.pAttachments = views.data(); framebuffer.width = width;
    framebuffer.height = height; framebuffer.layers = 1;
    acquire(frame->framebuffer, "resident framebuffer", [&](auto* out) {
        return vkCreateFramebuffer(c.device, &framebuffer, nullptr, out);
    });
    std::array<VkDescriptorPoolSize, 4> sizes{{
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 8}, {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 3},
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,2},{VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,1}}};
    VkDescriptorPoolCreateInfo pool{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    if (c.normal_info.shader_compatible) { ++sizes[0].descriptorCount; ++sizes[2].descriptorCount; }
    pool.maxSets = 1; pool.poolSizeCount = c.ray_query ? 4u : 3u; pool.pPoolSizes = sizes.data();
    acquire(frame->descriptors, "resident descriptor pool", [&](auto* out) {
        return vkCreateDescriptorPool(c.device, &pool, nullptr, out);
    });
    VkDescriptorSetAllocateInfo allocate{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocate.descriptorPool = frame->descriptors;
    allocate.descriptorSetCount = 1; allocate.pSetLayouts = &c.descriptor_layout;
    VkDescriptorSet set{};
    check(vkAllocateDescriptorSets(c.device, &allocate, &set), "resident descriptor allocation");
    std::array<VkDescriptorBufferInfo, 13> buffers{{
        {state.vertices.handle, 0, std::max<size_t>(1, info.vertex_count)*sizeof(PrResidentVertex)},
        {frame->uniform.handle, 0, sizeof(PrParams)},
        {state.transforms.handle, 0, std::max<size_t>(1, info.instance_count)*sizeof(GpuTransform)},
        {state.instances_buffer.handle, 0, std::max<size_t>(1, info.instance_count)*sizeof(PrResidentInstance)},
        {state.nodes.handle, 0, std::max<size_t>(1, info.bvh_nodes)*sizeof(GpuBvhNode)},
        {state.indices.handle, 0, std::max<size_t>(1, info.index_count)*sizeof(uint32_t)},
        {state.meshes_buffer.handle, 0, std::max<size_t>(1, info.mesh_count)*sizeof(GpuResidentMesh)},
        {state.studio.handle, 0, sizeof(PrStudio)},
        {state.tlas.handle, 0, std::max<size_t>(1, c.profile.tlas_node_count)*sizeof(GpuBvhNode)},
        {},
        {state.textures->metadata.handle,0,state.textures->metadata_bytes},
        {},
        {state.environment->uniform.handle,0,sizeof(PrResidentEnvironment)}
    }};
    VkDescriptorImageInfo image{state.textures->sampler,state.textures->image.view,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkDescriptorImageInfo environment_image{state.environment->sampler,state.environment->image.view,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    std::array<VkWriteDescriptorSet, 16> writes{};
    VkWriteDescriptorSetAccelerationStructureKHR acceleration_write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR};
    if (c.ray_query) {
        acceleration_write.accelerationStructureCount=1;
        acceleration_write.pAccelerationStructures=&state.acceleration->top->handle;
        writes[13].sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; writes[13].pNext=&acceleration_write;
        writes[13].dstSet=set; writes[13].dstBinding=13; writes[13].descriptorCount=1;
        writes[13].descriptorType=VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    }
    for (uint32_t i = 0; i < 13; ++i) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = set; writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = (i == 1 || i == 7 || i == 12) ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        if (i==9 || i==11) {
            writes[i].descriptorType=VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writes[i].pImageInfo=i==9 ? &image : &environment_image;
        } else writes[i].pBufferInfo=&buffers[i];
    }
    uint32_t write_count=c.ray_query ? 14u : 13u;
    VkDescriptorImageInfo normal_image{}; VkDescriptorBufferInfo normal_buffer{};
    if (c.normal_info.shader_compatible) {
        normal_image={state.normal_maps->sampler,state.normal_maps->image.view,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        normal_buffer={state.normal_maps->metadata.handle,0,state.normal_maps->metadata_bytes};
        auto& image_write=writes[write_count++]; image_write.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        image_write.dstSet=set; image_write.dstBinding=14; image_write.descriptorCount=1;
        image_write.descriptorType=VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; image_write.pImageInfo=&normal_image;
        auto& metadata_write=writes[write_count++]; metadata_write.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        metadata_write.dstSet=set; metadata_write.dstBinding=15; metadata_write.descriptorCount=1;
        metadata_write.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; metadata_write.pBufferInfo=&normal_buffer;
    }
    vkUpdateDescriptorSets(c.device,write_count,writes.data(),0,nullptr);
    make_commands(c, *frame);
    if (c.profile.timestamp_supported) {
        VkQueryPoolCreateInfo query{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        query.queryType=VK_QUERY_TYPE_TIMESTAMP; query.queryCount=3;
        acquire(frame->timestamps,"resident timestamp query pool",[&](auto* out) {
            return vkCreateQueryPool(c.device,&query,nullptr,out);
        });
    }
    if (c.errors.load() != errors_before) {
        c.poisoned = true;
        throw std::runtime_error("Vulkan validation failed creating resident targets: "+c.validation_message);
    }
    state.target = std::move(frame);
    state.descriptor_set = set; state.width = width; state.height = height;
    ++c.resident_properties.target_rebuilds;
}

void validate_resident_params(const Context& c, const PrParams& params,
                              uint32_t width, uint32_t height, uint8_t* output, size_t output_bytes,
                              uint32_t channel_mask, bool compact) {
    require(width && height && width <= c.properties.max_width && height <= c.properties.max_height,
            "resident capture extent exceeds device capability");
    require(static_cast<uint64_t>(width)*height <= 4096ull*4096ull, "resident pixel budget exceeded");
    require(channel_mask!=0 && (channel_mask & ~15u)==0,"resident invalid output channel mask");
    const size_t plane_bytes[]{4,4,4,1};
    size_t bytes_per_pixel=0;
    for (uint32_t i=0; i<4; ++i) if (channel_mask & (1u<<i)) bytes_per_pixel+=plane_bytes[i];
    require(output && output_bytes==static_cast<size_t>(width)*height*bytes_per_pixel,
            "resident output byte extent mismatch");
    require(!compact || params.counts[2]==((channel_mask & 1u) ? 1u : 0u),
            "resident output mask disagrees with RGB request");
    require(params.counts[0] <= 4 && params.counts[1] == c.resident_properties.instanced_triangles &&
            params.counts[2] <= 1 && params.counts[3] == c.resident_properties.instance_count,
            "resident uniform scene counts mismatch");
    require(params.clipping[0] == width && params.clipping[1] == height &&
            params.clipping[2] > 0 && params.clipping[3] > params.clipping[2],
            "resident clipping calibration mismatch");
    const float* blocks[]{params.view, params.intrinsics, params.clipping, params.camera,
        &params.light_directions[0][0], &params.light_colors[0][0], params.options};
    const uint32_t sizes[]{16,4,4,4,16,16,4};
    for (uint32_t b = 0; b < 7; ++b)
        for (uint32_t k = 0; k < sizes[b]; ++k)
            require(std::isfinite(blocks[b][k]), "resident uniform FP32 conversion is nonfinite");
    require(params.intrinsics[0] > 0 && params.intrinsics[1] > 0 &&
            params.options[0] >= 0 && params.options[1] > 0, "resident invalid projection/exposure");
    for (uint32_t light = 0; light < params.counts[0]; ++light) {
        double length = std::hypot(static_cast<double>(params.light_directions[light][0]),
            static_cast<double>(params.light_directions[light][1]),
            static_cast<double>(params.light_directions[light][2]));
        require(length >= 0.999 && length <= 1.001, "resident light direction must be unit length");
        for (uint32_t k = 0; k < 3; ++k)
            require(params.light_colors[light][k] >= 0, "resident negative light radiance");
    }
}

void resident_render(Context& c, const float* matrices, uint32_t instance_count, const PrParams& params,
                     uint32_t width, uint32_t height, uint8_t* output, size_t output_bytes,
                     uint32_t channel_mask=15u, bool compact=false) {
    const auto prepare_begin=HostClock::now();
    require(c.resident && c.resident_state && !c.poisoned,
            "resident scene is not loaded or context failed; open an explicit new session");
    require(!c.reflection_info.shader_compatible || c.reflection_info.enabled,
            "reflection shader requires successful explicit setup before capture");
    require(!c.indirect_info.shader_compatible||c.indirect_info.enabled,
            "indirect shader requires successful explicit setup before capture");
    validate_resident_params(c, params, width, height, output, output_bytes, channel_mask, compact);
    auto& state = *c.resident_state;
    const bool msaa_color=c.msaa_info.requested_samples==4 && params.counts[2]!=0;
    const bool center_pass=!msaa_color || (channel_mask&14u)!=0;
    auto transforms = resident_transforms(state, matrices, instance_count);
    auto tlas = c.ray_query ? std::vector<GpuBvhNode>{} : resident_tlas(state, transforms);
    const auto upload_begin=HostClock::now();
    try {
    resident_target(c, state, width, height);
    auto& frame = *state.target;
    if (msaa_color) msaa_prepare(c,frame,width,height);
    require(frame.readback_mapping && output_bytes<=frame.readback_capacity,
            "resident readback target capacity mismatch");
    auto& info = c.resident_properties;
    uint64_t errors_before = c.errors.load();
    if (instance_count) {
        c.upload(state.transforms, transforms.data(), transforms.size()*sizeof(GpuTransform));
        if (!c.ray_query) c.upload(state.tlas, tlas.data(), tlas.size()*sizeof(GpuBvhNode));
    }
    if (c.ray_query) upload_ray_instances(c,state,transforms);
    // This static UBO is populated only before the first draw. Its lifetime is
    // the resident scene, independent of reusable target creation/resizing.
    if (!info.captures) {
        c.upload(state.studio, &c.studio, sizeof(c.studio));
        info.parameter_upload_bytes += sizeof(c.studio);
    }
    info.transform_upload_bytes += transforms.size()*sizeof(GpuTransform);
    info.host_transform_input_bytes += static_cast<uint64_t>(instance_count)*64;
    PrParams submitted=params;
    // Reserved caller words never opt into a display change; only the checked setter does.
    submitted.options[2]=static_cast<float>(c.display_transform);
    // Resident shaders do not use the legacy triangle-count word. The caller
    // still supplies/validates it; only explicit indirect variants read a seed.
    if(c.indirect_info.enabled) submitted.counts[1]=c.indirect_info.seed;
    c.upload(frame.uniform, &submitted, sizeof(submitted));
    info.parameter_upload_bytes += sizeof(params);
    const auto record_begin=HostClock::now();
    check(vkResetFences(c.device, 1, &frame.fence), "resident reset fence");
    check(vkResetCommandBuffer(frame.commands, 0), "resident reset command");
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    check(vkBeginCommandBuffer(frame.commands, &begin), "resident begin frame");
    if (frame.timestamps) {
        vkCmdResetQueryPool(frame.commands,frame.timestamps,0,3);
        vkCmdWriteTimestamp(frame.commands,VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,frame.timestamps,0);
    }
    if (c.ray_query) record_ray_update(c,state,frame.commands);
    const VkBuffer input_buffers[]{state.transforms.handle,frame.uniform.handle,state.tlas.handle,state.studio.handle};
    std::array<VkBufferMemoryBarrier, 4> inputs{};
    for (uint32_t i = 0; i < inputs.size(); ++i) {
        auto& barrier = inputs[i];
        barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        barrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
        barrier.dstAccessMask = (i == 1 || i == 3) ? VK_ACCESS_UNIFORM_READ_BIT : VK_ACCESS_SHADER_READ_BIT;
        barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.buffer = input_buffers[i]; barrier.size = VK_WHOLE_SIZE;
    }
    vkCmdPipelineBarrier(frame.commands, VK_PIPELINE_STAGE_HOST_BIT,
        VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
        0, nullptr, static_cast<uint32_t>(inputs.size()), inputs.data(), 0, nullptr);
    VkDeviceSize offset=0;
    if (center_pass) {
    std::array<VkClearValue, 5> clear{};
    clear[0].color.float32[3] = 1;
    if (c.studio.environment[3] != 0 && params.counts[2] != 0) {
        const auto mapped=c.display_transform==1 ? neutral_clear(c.studio.environment,params.options[0]) : std::array<double,3>{};
        for (uint32_t k=0; k<3; ++k) {
            double linear=c.display_transform==1 ? std::clamp(mapped[k],0.0,1.0) :
                std::clamp(static_cast<double>(c.studio.environment[k])*params.options[0],0.0,1.0);
            clear[0].color.float32[k]=static_cast<float>(linear<=0.0031308 ? 12.92*linear :
                1.055*std::pow(linear,1.0/2.4)-0.055);
        }
    }
    clear[1].color.float32[0] = std::numeric_limits<float>::infinity();
    clear[4].depthStencil.depth = 1;
    VkRenderPassBeginInfo pass{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    pass.renderPass = c.render_pass; pass.framebuffer = frame.framebuffer;
    pass.renderArea.extent = {width, height}; pass.clearValueCount = 5; pass.pClearValues = clear.data();
    vkCmdBeginRenderPass(frame.commands, &pass, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(frame.commands, VK_PIPELINE_BIND_POINT_GRAPHICS, c.pipeline);
    VkViewport viewport{0,0,static_cast<float>(width),static_cast<float>(height),0,1};
    VkRect2D scissor{{0,0},{width,height}};
    vkCmdSetViewport(frame.commands, 0, 1, &viewport);
    vkCmdSetScissor(frame.commands, 0, 1, &scissor);
    offset = 0;
    vkCmdBindVertexBuffers(frame.commands, 0, 1, &state.vertices.handle, &offset);
    vkCmdBindIndexBuffer(frame.commands, state.indices.handle, 0, VK_INDEX_TYPE_UINT32);
    vkCmdBindDescriptorSets(frame.commands, VK_PIPELINE_BIND_POINT_GRAPHICS, c.pipeline_layout,
                           0, 1, &state.descriptor_set, 0, nullptr);
    if (c.environment_info.loaded && params.counts[2] && !msaa_color) {
        vkCmdBindPipeline(frame.commands,VK_PIPELINE_BIND_POINT_GRAPHICS,state.environment->sky_pipeline);
        vkCmdDraw(frame.commands,3,1,0,0);
        vkCmdBindPipeline(frame.commands,VK_PIPELINE_BIND_POINT_GRAPHICS,c.pipeline);
    }
    for (uint32_t slot = 0; slot < instance_count; ++slot) {
        const auto& mesh = state.meshes[state.instances[slot].mesh_index];
        vkCmdDrawIndexed(frame.commands, mesh.index_count, 1, mesh.first_index,
                         static_cast<int32_t>(mesh.first_vertex), slot);
    }
    vkCmdEndRenderPass(frame.commands);
    }
    if (msaa_color) msaa_record(c,frame,params);
    if (frame.timestamps)
        vkCmdWriteTimestamp(frame.commands,VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,frame.timestamps,1);
    const size_t bytes_per_pixel[]{4,4,4,1};
    const size_t pixels = static_cast<size_t>(width)*height;
    offset = 0;
    for (uint32_t i = 0; i < 4; ++i) {
        if (!(channel_mask & (1u<<i))) continue;
        // All preceding formats are four bytes; VALID (one byte) is last.
        // Thus odd width/height and VALID-only still satisfy copy alignment.
        VkBufferImageCopy copy{};
        copy.bufferOffset = offset;
        copy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        copy.imageSubresource.layerCount = 1;
        copy.imageExtent = {width,height,1};
        vkCmdCopyImageToBuffer(frame.commands, frame.images[i].handle, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                              frame.readback.handle, 1, &copy);
        offset += pixels*bytes_per_pixel[i];
    }
    VkBufferMemoryBarrier host{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    host.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; host.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    host.srcQueueFamilyIndex = host.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    host.buffer = frame.readback.handle; host.size = VK_WHOLE_SIZE;
    vkCmdPipelineBarrier(frame.commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                         0, 0, nullptr, 1, &host, 0, nullptr);
    if (frame.timestamps)
        vkCmdWriteTimestamp(frame.commands,VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,frame.timestamps,2);
    check(vkEndCommandBuffer(frame.commands), "resident end frame");
    const uint64_t record_ns=elapsed_ns(record_begin);
    const auto timing=submit_and_wait(c, frame);
    const auto readback_begin=HostClock::now();
    if (c.errors.load() != errors_before) {
        c.poisoned = true;
        throw std::runtime_error("Vulkan validation failed during resident capture: "+c.validation_message);
    }
    uint64_t gpu_render=0,gpu_copy=0,gpu_total=0,timestamp_samples=0;
    if (frame.timestamps) {
        uint64_t ticks[3]{};
        check(vkGetQueryPoolResults(c.device,frame.timestamps,0,3,sizeof(ticks),ticks,sizeof(uint64_t),
            VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT),"resident timestamp read");
        uint32_t bits=c.profile.timestamp_valid_bits;
        uint64_t mask=bits==64 ? UINT64_MAX : ((1ull<<bits)-1);
        double period=c.profile.timestamp_period_ns;
        // A possible multiple counter wrap cannot be reconstructed honestly.
        // Keep the capture, omit that sample instead of reporting a false time.
        double wrap_ns=std::ldexp(period,static_cast<int>(bits));
        if (static_cast<double>(timing.submit)+timing.wait < wrap_ns) {
            double render_ns=static_cast<double>((ticks[1]-ticks[0])&mask)*period;
            double copy_ns=static_cast<double>((ticks[2]-ticks[1])&mask)*period;
            double total_ns=static_cast<double>((ticks[2]-ticks[0])&mask)*period;
            if (std::isfinite(total_ns) && total_ns < static_cast<double>(UINT64_MAX)) {
                gpu_render=static_cast<uint64_t>(render_ns); gpu_copy=static_cast<uint64_t>(copy_ns);
                gpu_total=static_cast<uint64_t>(total_ns); timestamp_samples=1;
            }
        }
    }
    // The HOST_COHERENT mapping lives with the target. Fence completion and the
    // transfer->HOST barrier make writes visible without invalidate operations.
    std::memcpy(output, frame.readback_mapping, output_bytes);
    c.telemetry.readback_bytes += output_bytes;
    c.readback_info.copied_bytes += output_bytes;
    c.readback_info.last_channel_mask=channel_mask;
    auto& profile=c.profile;
    ++profile.captures;
    profile.host_prepare_ns+=elapsed_ns(prepare_begin,upload_begin);
    profile.host_upload_ns+=elapsed_ns(upload_begin,record_begin);
    profile.host_record_ns+=record_ns; profile.host_submit_ns+=timing.submit;
    profile.host_wait_ns+=timing.wait; profile.host_readback_ns+=elapsed_ns(readback_begin);
    profile.gpu_render_ns+=gpu_render; profile.gpu_copy_ns+=gpu_copy; profile.gpu_total_ns+=gpu_total;
    profile.timestamp_samples+=timestamp_samples; profile.tlas_upload_bytes+=tlas.size()*sizeof(GpuBvhNode);
    if(c.indirect_info.enabled){++c.indirect_info.captures;if(params.counts[2])++c.indirect_info.rgb_captures;}
    if (c.reflection_info.enabled) { ++c.reflection_info.captures; if(params.counts[2]) ++c.reflection_info.rgb_captures; }
    if (c.normal_info.loaded && params.counts[2]) ++c.normal_info.rgb_captures_with_normals;
    if (c.texture_info.loaded && params.counts[2]) ++c.texture_info.rgb_captures_with_textures;
    if (c.environment_info.loaded && params.counts[2]) {
        ++c.environment_info.rgb_captures_with_environment;
        ++c.environment_info.sky_draws;
    }
    if (c.ray_query) {
        if (state.acceleration->built) ++c.ray_info.tlas_updates;
        else ++c.ray_info.tlas_builds;
        state.acceleration->built=true;
        if (params.counts[2]) ++c.ray_info.shadow_captures;
    }
    if (msaa_color) { ++c.msaa_info.color_passes; ++c.msaa_info.resolve_passes; }
    if (center_pass) ++c.msaa_info.center_passes;
    ++c.properties.captures; ++info.captures;
    } catch (...) {
        c.poisoned = true; // a failed GPU operation cannot publish or be silently retried
        throw;
    }
}
