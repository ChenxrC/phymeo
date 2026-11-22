#pragma once
#include <cstddef>
#include <cstdint>
#ifdef _WIN32
#define PR_API extern "C" __declspec(dllexport)
#else
#define PR_API extern "C" __attribute__((visibility("default")))
#endif

// All matrices use GLSL column-major order. Arrays are packed IEEE FP32/uint32.
struct PrVertex {
    float position[4];
    float normal_metallic[4];
    float color_roughness[4];
    uint32_t ids[4]; // instance, deindexed triangle, reserved, reserved
};
struct PrParams {
    float view[16];
    float intrinsics[4]; // fx fy cx cy
    float clipping[4]; // width height near far
    float camera[4];
    float light_directions[4][4]; // world direction of light travel
    float light_colors[4][4];
    uint32_t counts[4]; // lights, triangles, RGB requested, reserved
    float options[4]; // linear exposure multiplier, shadow bias metres
};
struct PrInfo {
    uint32_t abi_version, device_index, api_version, driver_version;
    uint32_t vendor_id, device_id, device_type, max_width, max_height, max_vertices;
    uint8_t device_uuid[16];
    char device_name[256];
    uint64_t validation_errors, validation_warnings, captures;
};
struct PrCounts { uint64_t live_contexts, live_frames, live_resources; };
static_assert(sizeof(PrVertex) == 64, "vertex ABI");
static_assert(sizeof(PrParams) == 272, "uniform ABI");
PR_API uint32_t pr_abi_version();
PR_API const char* pr_error();
PR_API void* pr_create(uint32_t device_index, uint32_t validation,
    const uint32_t* vert, size_t vert_bytes, const uint32_t* frag, size_t frag_bytes);
PR_API int pr_info(void* context, PrInfo* output);
PR_API int pr_render(void* context, const PrVertex* vertices, uint32_t vertex_count,
    const PrParams* params, uint32_t width, uint32_t height,
    uint8_t* output, size_t output_bytes);
PR_API int pr_counts(PrCounts* output);
// ABI 2: diagnostics are copied after actual teardown, including failure paths.
PR_API int pr_destroy(void* context, PrInfo* final_info);


// Optional resident extension v1. The original pr_abi_version() remains 2.
// Static mesh-local positions/normals and uint32 triangle indices are uploaded
// once. Scene matrices are rigid world_from_mesh transforms, column-major FP32.
struct PrResidentVertex {
    float position[4]; // xyz, 1
    float normal[4];   // unit xyz, 0
};
struct PrResidentMesh {
    uint32_t first_vertex, vertex_count, first_index, index_count;
};
struct PrResidentInstance {
    uint32_t mesh_index, instance_id, reserved[2];
    float color_roughness[4]; // linear RGB, perceptual roughness
    float material[4]; // metallic, 0, 0, 0
};
// All input pointers are borrowed only for the duration of a call. Static data
// is copied into owned device-local buffers; captures never reread host geometry.
// geometry_upload_bytes counts copied GPU buffers including BVH/reordered indices
// and empty-scene descriptor sentinels. transform_upload_bytes counts the actual
// forward+inverse GPU writes (128 per nonempty instance), while
// host_transform_input_bytes counts the caller matrices (64 per instance).
// A same-size capture reuses attachments/readback/descriptors/commands/fence;
// target_rebuilds counts successful target allocations. One live frame owner is
// therefore expected until resident close. GPU errors poison the session;
// input validation failures occur before any frame mutation.
struct PrResidentInfo {
    uint32_t resident_abi_version;
    uint32_t max_vertices, max_indices, max_meshes, max_instances;
    uint32_t max_bvh_nodes, max_instanced_triangles;
    uint32_t vertex_count, index_count, mesh_count, instance_count, bvh_nodes;
    uint32_t instanced_triangles, loaded, poisoned, reserved;
    uint64_t upload_calls, geometry_upload_bytes, transform_upload_bytes;
    uint64_t parameter_upload_bytes, captures, shadow_node_budget, target_rebuilds;
    uint64_t host_transform_input_bytes;
};
static_assert(sizeof(PrResidentVertex) == 32, "resident vertex ABI");
static_assert(sizeof(PrResidentMesh) == 16, "resident mesh ABI");
static_assert(sizeof(PrResidentInstance) == 48, "resident instance ABI");
static_assert(sizeof(PrResidentInfo) == 128, "resident info ABI");
static_assert(offsetof(PrResidentInfo, upload_calls) == 64, "resident info alignment");
PR_API uint32_t pr_resident_abi_version();
PR_API void* pr_resident_create(uint32_t device_index, uint32_t validation,
    const uint32_t* vert, size_t vert_bytes, const uint32_t* frag, size_t frag_bytes);
// Mesh ranges must be ordered and tightly packed; indices are mesh-local.
// Empty scenes use all zero counts. A successful upload cannot be replaced.
PR_API int pr_resident_upload(void* context,
    const PrResidentVertex* vertices, uint32_t vertex_count,
    const uint32_t* indices, uint32_t index_count,
    const PrResidentMesh* meshes, uint32_t mesh_count,
    const PrResidentInstance* instances, uint32_t instance_count);
// counts = {light_count, instanced_triangle_count, RGB_requested, instance_count}.
// Matrices contain 16 floats per instance, in the immutable upload order.
// C++ derives the inverse; GPU transform bytes include forward and inverse.
// Output packing is identical to pr_render; pr_info/pr_destroy/pr_counts apply.
PR_API int pr_resident_render(void* context, const float* world_from_mesh,
    uint32_t instance_count, const PrParams* params,
    uint32_t width, uint32_t height, uint8_t* output, size_t output_bytes);
PR_API int pr_resident_info(void* context, PrResidentInfo* output);


// Optional telemetry extension; it does not change resident v1 or legacy ABI2.
// Counts cover this context's explicit VkDeviceMemory allocations only, using
// the actual chosen memory heap/type. Driver-hidden allocations are excluded.
// fence_wait_wall_ns is cumulative host wall time inside vkWaitForFences,
// including the initial static upload, not GPU kernel/execution time.
struct PrResidentTelemetry {
    uint32_t telemetry_abi_version, device_luid_valid, device_node_mask, reserved;
    uint8_t device_luid[8], reserved_identity[8];
    uint64_t device_local_bytes_current, device_local_bytes_peak, device_local_bytes_allocated;
    uint64_t host_visible_bytes_current, host_visible_bytes_peak, host_visible_bytes_allocated;
    uint64_t readback_bytes, fence_wait_wall_ns, fence_wait_calls, submissions;
    uint64_t allocation_count, free_count;
};
static_assert(sizeof(PrResidentTelemetry) == 128, "resident telemetry ABI");
static_assert(offsetof(PrResidentTelemetry, device_local_bytes_current) == 32, "telemetry alignment");
PR_API int pr_resident_telemetry(void* context, PrResidentTelemetry* output);
// Optional teardown returns final counters after actual resource destruction.
// Call either this or pr_destroy exactly once for the owned context.
PR_API int pr_resident_destroy(void* context, PrInfo* final_info, PrResidentTelemetry* final_telemetry);
