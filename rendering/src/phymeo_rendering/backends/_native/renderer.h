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

// Optional static studio input (std140 binding 7). panel halfU x halfV points
// along the emitter normal; radiance is linear RGB. All padding must be zero.
struct PrStudioPanel { float position[4], half_u[4], half_v[4], radiance[4]; };
struct PrStudio { float environment[4]; PrStudioPanel panels[2]; };
static_assert(sizeof(PrStudio) == 144, "studio ABI");
PR_API int pr_resident_set_studio(void* context, const PrStudio* studio);

// Optional per-capture cumulative profile, excluding initial static upload.
// HOST upload includes target allocation on first use/resize. GPU timestamps
// separate raster+fragment work from attachment copies, not individual lights.
// Unsupported device timestamps have supported=0, samples=0, GPU times=0.
struct PrResidentProfile {
    uint32_t profile_abi_version, timestamp_supported, timestamp_valid_bits, tlas_node_count;
    double timestamp_period_ns;
    uint64_t captures, host_prepare_ns, host_upload_ns, host_record_ns;
    uint64_t host_submit_ns, host_wait_ns, host_readback_ns;
    uint64_t gpu_render_ns, gpu_copy_ns, gpu_total_ns;
    uint64_t tlas_upload_bytes, timestamp_samples, reserved;
};
static_assert(sizeof(PrResidentProfile) == 128, "resident profile ABI");
PR_API int pr_resident_profile(void* context, PrResidentProfile* output);

// Optional compact HOST readback. Bits 0..3 select RGBA8, depth F32, ID U32,
// VALID U8. Nonempty planes are tightly concatenated in that order. VALID is
// last, so every GPU copy starts on a four-byte boundary even at odd extents.
// params.counts[2] must equal (channel_mask & 1) != 0. Original render retains
// its fixed 13-byte layout and original RGB flag semantics.
PR_API int pr_resident_render_planes(void* context, const float* world_from_mesh,
    uint32_t instance_count, const PrParams* params, uint32_t width, uint32_t height,
    uint32_t channel_mask, uint8_t* output, size_t output_bytes);
struct PrResidentReadbackInfo {
    uint32_t readback_abi_version, memory_type_index, memory_property_flags;
    uint32_t persistent_mapped, last_channel_mask, reserved[3];
    uint64_t capacity_bytes, copied_bytes, map_calls, unmap_calls;
};
static_assert(sizeof(PrResidentReadbackInfo) == 64, "resident readback info ABI");
// Actual selected Vulkan memory flags, not a request or a simulated capability.
// A close-time cached query is a last-live observation, not final allocator state.
PR_API int pr_resident_readback_info(void* context, PrResidentReadbackInfo* output);

// Optional opaque base-color textures, independent of existing resident ABI1.
// Source pixels are top-row-first RGBA8 sRGB, alpha=255. Ranges are tightly
// packed; no external decoder, resizing, mip generation or hidden I/O occurs.
struct PrResidentTexture { uint32_t width, height, first_byte, reserved; };
struct PrResidentUV { float u, v; };
struct PrResidentTextureInfo {
    uint32_t texture_abi_version, max_textures, max_width, max_height, max_layers;
    uint32_t texture_count, array_width, array_height, array_layers, uv_count;
    uint32_t bound_instances, loaded, sampler_filter, mip_levels, format, reserved;
    uint64_t max_array_bytes, source_rgba_bytes, padded_rgba_bytes, metadata_upload_bytes;
    uint64_t upload_calls, image_upload_bytes, rgb_captures_with_textures, reserved64;
};
static_assert(sizeof(PrResidentTexture)==16 && sizeof(PrResidentUV)==8, "texture input ABI");
static_assert(sizeof(PrResidentTextureInfo)==128, "texture info ABI");
PR_API uint32_t pr_resident_texture_abi_version();
// Exactly once after geometry upload and before any successful capture. UVs
// follow the complete original vertex table, slots follow instance order;
// UINT32_MAX is an untextured instance. Caller arrays are copied, never retained.
PR_API int pr_resident_set_textures(void* context,
    const PrResidentTexture* textures, uint32_t texture_count,
    const uint8_t* rgba, size_t rgba_bytes, const PrResidentUV* uv, uint32_t uv_count,
    const uint32_t* instance_slots, uint32_t instance_count);
PR_API int pr_resident_texture_info(void* context, PrResidentTextureInfo* output);

// Optional directional environment extension v1, independent of ABI2/resident1.
// Level-major RGBA32F linear radiance: six Vulkan faces +X,-X,+Y,-Y,+Z,-Z
// per mip, each top row first. Full power-of-two chain to 1x1; RGB in [0,100],
// alpha=1. World ray XYZ samples cube XZY. SH remains WORLD XYZ, real standard
// orthonormal basis (no negative phase), already Lambert-convolved / pi:
// [1,y,z,x,xy,yz,3z*z-1,xz,x*x-y*y] with their normalized SH constants.
struct PrResidentEnvironment {
    uint32_t side, mip_levels, reserved[2];
    float diffuse_sh[9][4]; // signed finite RGB coefficients; w=0
};
struct PrResidentEnvironmentInfo {
    uint32_t environment_abi_version, supported, shader_compatible, max_side, max_mip_levels;
    uint32_t side, mip_levels, loaded, format, cube_layers, sh_coefficients, axis_transform, diffuse_basis;
    // Optional readback flags. Older binaries return zero for both.
    // [0] progressive fragment understands cubemap CDF, [1] active uploaded CDF.
    uint32_t reserved[3];
    uint64_t max_rgba_bytes, source_rgba_bytes, image_upload_bytes, parameter_upload_bytes;
    uint64_t upload_calls, rgb_captures_with_environment, sky_draws, reserved64;
};
static_assert(sizeof(PrResidentEnvironment)==160, "environment input ABI");
static_assert(sizeof(PrResidentEnvironmentInfo)==128, "environment info ABI");
static_assert(offsetof(PrResidentEnvironmentInfo,max_rgba_bytes)==64, "environment info alignment");
PR_API uint32_t pr_resident_environment_abi_version();
// Exactly once after geometry and before any capture. The SPIR-V main stages
// must be compatible; explicit sky vertex/fragment bytes avoid implicit file I/O.
// All caller memory is copied/consumed synchronously; no pointer survives return.
// A map replaces the uniform studio environment, retaining rectangle lights.
// Upload counters INCLUDE default black cube/UBO, independent of older counters.
PR_API int pr_resident_set_environment(void* context, const PrResidentEnvironment* environment,
    const float* rgba, size_t rgba_bytes, const uint32_t* sky_vert, size_t sky_vert_bytes,
    const uint32_t* sky_frag, size_t sky_frag_bytes);
PR_API int pr_resident_environment_info(void* context, PrResidentEnvironmentInfo* output);

// Optional ray-query shadow extension v1. Only this create enables Vulkan 1.2
// bufferDeviceAddress + KHR acceleration_structure/ray_query. No fallback.
// Geometry, cameras, shading, capture channels and older ABIs are unchanged.
struct PrResidentRayQueryInfo {
    uint32_t ray_query_abi_version, enabled, blas_count, tlas_instance_count, primitive_count;
    uint32_t loaded, poisoned, reserved, max_meshes, max_instances, max_triangles;
    uint32_t build_flags, update_flags, trace_flags, reserved2[2];
    uint64_t max_acceleration_bytes, acceleration_buffer_bytes, scratch_buffer_bytes;
    uint64_t blas_builds, tlas_builds, tlas_updates, instance_upload_bytes, shadow_captures;
};
static_assert(sizeof(PrResidentRayQueryInfo)==128, "ray-query info ABI");
// primitive_count is the sum of UNIQUE BLAS triangles; resident_info still
// reports every instanced triangle. Buffer byte counts are logical sizes, not
// VkDeviceMemory allocations (the latter remain in existing telemetry).
// shadow_captures counts successful RGB captures, not actual ray launches.
PR_API uint32_t pr_resident_ray_query_abi_version();
PR_API void* pr_resident_create_ray_query(uint32_t device_index,uint32_t validation,
    const uint32_t* vert,size_t vert_bytes,const uint32_t* frag,size_t frag_bytes);
PR_API int pr_resident_ray_query_info(void* context,PrResidentRayQueryInfo* output);

// Optional opaque surface-map extension v1. Same upload layout as textures.
// Only this entry accepts record.reserved=1: A is linear UNORM roughness,
// multiplied into the material's perceptual roughness in every lighting path.
// Flag 0 still requires A=255. RGB remains sRGB; output opacity is always one.
// The old set_textures entry continues to require flag 0 and opaque A=255.
PR_API uint32_t pr_resident_surface_maps_abi_version();
// 1=declared compatible fragment (named uint32 SpecId57 default1), 0=old SPV,
// -1=invalid/nonresident context. The setter repeats this check atomically.
PR_API int pr_resident_surface_maps_shader_compatible(void* context);
PR_API int pr_resident_set_surface_maps(void* context,
    const PrResidentTexture* textures,uint32_t texture_count,
    const uint8_t* rgba,size_t rgba_bytes,const PrResidentUV* uv,uint32_t uv_count,
    const uint32_t* instance_slots,uint32_t instance_count);

// Optional display extension v1: 0 preserves clip, 1 Khronos PBR Neutral.
// Same linear exposure -> display map -> sRGB order for main, sky and clear.
// Mode is immutable after any capture; sensors and lighting are unchanged.
// Named uint32 SpecId58/default1 must be declared by main and loaded sky SPV.
PR_API uint32_t pr_resident_display_transform_abi_version();
PR_API int pr_resident_display_transform_shader_compatible(void* context);
PR_API int pr_resident_set_display_transform(void* context,uint32_t mode);

// Optional MSAA1: 4x display-linear color, independent 1x pixel-center sensors.
// Names and sample masks are actual Vulkan observations. Counters count completed
// captures/passes; target rebuilds count allocated attachment sets, including resize.
struct PrResidentMsaaInfo {
    uint32_t msaa_abi_version, requested_samples, color_samples, sensor_samples;
    uint32_t color_format, depth_format, resolve_format, output_format;
    uint32_t color_sample_counts, depth_sample_counts, max_width, max_height;
    uint32_t shader_compatible, standard_sample_locations, reserved0, reserved1;
    uint64_t color_passes, center_passes, resolve_passes, target_rebuilds;
    uint64_t reserved64[4];
};
static_assert(sizeof(PrResidentMsaaInfo)==128, "MSAA info ABI");
PR_API uint32_t pr_resident_msaa_abi_version();
PR_API int pr_resident_msaa_info(void* context,PrResidentMsaaInfo* output);
// Once before capture. Caller shader bytes are consumed synchronously. New main
// and sky must declare MSAA ABI marker59; resolve fragment marker61. Old 1x ABI
// never calls this entry and remains usable with its original shaders.
PR_API int pr_resident_set_msaa(void* context,uint32_t samples,
    const uint32_t* resolve_vert,size_t resolve_vert_bytes,
    const uint32_t* resolve_frag,size_t resolve_frag_bytes);

// Optional linear tangent-space normal-map extension v1. Fixed strength 1,
// glTF +Y; independent from sRGB/roughness resources. Complete UV table and
// per-instance slots use the same packed record ABI as color textures.
struct PrResidentNormalInfo {
    uint32_t normal_abi_version,max_textures,max_width,max_height,max_layers;
    uint32_t texture_count,array_width,array_height,array_layers,uv_count;
    uint32_t bound_instances,loaded,sampler_filter,mip_levels,format,shader_compatible;
    uint64_t max_array_bytes,source_rgba_bytes,padded_rgba_bytes,metadata_upload_bytes;
    uint64_t upload_calls,image_upload_bytes,rgb_captures_with_normals,reserved64;
};
static_assert(sizeof(PrResidentNormalInfo)==128,"normal-map info ABI");
PR_API uint32_t pr_resident_normal_maps_abi_version();
PR_API int pr_resident_normal_info(void* context,PrResidentNormalInfo* output);
PR_API int pr_resident_set_normal_maps(void* context,
    const PrResidentTexture* textures,uint32_t texture_count,const uint8_t* rgba,size_t rgba_bytes,
    const PrResidentUV* uv,uint32_t uv_count,const uint32_t* instance_slots,uint32_t instance_count);

// Optional one ideal specular bounce. Shader marker64 and hardware TLAS are
// mandatory. Setter is one-shot after geometry, before any capture.
struct PrResidentReflectionInfo {
    uint32_t reflection_abi_version,supported,shader_compatible,enabled;
    uint32_t max_bounces,receiver_count,set_calls,reserved;
    float metallic_min,roughness_max,reserved_f[2];
    uint64_t captures,rgb_captures;
};
static_assert(sizeof(PrResidentReflectionInfo)==64,"reflection info ABI");
PR_API uint32_t pr_resident_reflection_abi_version();
PR_API int pr_resident_reflection_info(void* context,PrResidentReflectionInfo* output);
PR_API int pr_resident_set_reflection(void* context,uint32_t mode);

// Fixed sixteen stratified cosine samples for one diffuse bounce. uint32 seed
// is retained in context; only explicit indirect shaders interpret uploaded
// resident counts.y as that seed (caller triangle-count validation is unchanged).
struct PrResidentIndirectInfo {
    uint32_t indirect_abi_version,supported,shader_compatible,enabled;
    uint32_t samples_per_pixel,seed,max_bounces,set_calls,receiver_count,reserved[3];
    uint64_t captures,rgb_captures;
};
static_assert(sizeof(PrResidentIndirectInfo)==64,"indirect info ABI");
PR_API uint32_t pr_resident_indirect_abi_version();
PR_API int pr_resident_indirect_info(void* context,PrResidentIndirectInfo* output);
PR_API int pr_resident_set_indirect(void* context,uint32_t mode,uint32_t seed);

// Optional progressive ABI1. Counters describe published RGB batches, not hits.
struct PrResidentProgressiveInfo {
    uint32_t progressive_abi_version, supported, shader_compatible, enabled;
    uint32_t bounces, max_batches, batches, last_batch_index;
    uint32_t seed, samples_per_batch, width, height, reserved[4];
    uint64_t rgb_captures, reset_calls, mean_readbacks, mean_readback_bytes, status_readback_bytes, reserved64[3];
};
static_assert(sizeof(PrResidentProgressiveInfo)==128,"progressive info ABI");
PR_API uint32_t pr_resident_progressive_abi_version();
PR_API int pr_resident_set_progressive(void*,uint32_t,const uint32_t*,size_t);
PR_API int pr_resident_progressive_info(void*,PrResidentProgressiveInfo*);
PR_API int pr_resident_begin_progressive(void*,uint32_t);
PR_API int pr_resident_reset_progressive(void*);
PR_API int pr_resident_read_progressive_mean(void*,void*,size_t);

// Opt-in first-batch spatial display. Raw progressive mean is never filtered.
struct PrResidentSpatialInfo {
    uint32_t spatial_abi_version,supported,shader_compatible,enabled;
    uint32_t last_capture_filtered,kernel_radius,guide_code_limit,reserved;
    uint64_t filtered_rgb_captures,rgb_captures,guide_readbacks,guide_readback_bytes;
};
static_assert(sizeof(PrResidentSpatialInfo)==64,"spatial info ABI");
PR_API uint32_t pr_resident_spatial_abi_version();
// Replaces set_progressive; uses a separate same-exposure direct guide pass.
PR_API int pr_resident_set_spatial(void*,uint32_t,const uint32_t*,size_t,const uint32_t*,size_t);
PR_API int pr_resident_spatial_info(void*,PrResidentSpatialInfo*);
PR_API int pr_resident_read_spatial_guide(void*,void*,size_t);

// Optional settling ABI1: immutable raw mean -> bounded display-only dispatch.
struct PrResidentSettlingInfo {
    uint32_t settling_abi_version,enabled,last_batch,last_capture_filtered;
    float last_filtered_fraction;
    uint32_t last_filtered_primary_samples,guide_valid,reserved;
    uint64_t guide_draws,filtered_rgb_captures,filtered_primary_samples,filter_dispatches;
};
static_assert(sizeof(PrResidentSettlingInfo)==64,"settling info ABI");
PR_API uint32_t pr_resident_settling_abi_version();
PR_API int pr_resident_set_settling(void*,uint32_t,const uint32_t*,size_t,const uint32_t*,size_t,const uint32_t*,size_t);
PR_API int pr_resident_settling_info(void*,PrResidentSettlingInfo*);
