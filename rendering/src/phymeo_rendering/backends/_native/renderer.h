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
