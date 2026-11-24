#pragma once
#include <stdint.h>
#include <stddef.h>

#if defined(PV0_BUILD)
#define PV0_API __declspec(dllexport)
#else
#define PV0_API __declspec(dllimport)
#endif
#ifdef __cplusplus
extern "C" {
#endif

typedef struct Pv0Export {
    uint64_t memory, writable, ready, allocation_bytes, buffer_bytes;
} Pv0Export;
typedef struct Pv0Result {
    uint32_t checked, mismatches, first_input, last_input;
} Pv0Result;

/* 64-bit Windows cdecl. Serialize host calls on a context. All functions
   returning int use 0=success, 1=error, 2=timeout (owner may drain/retry).
   flags: 1=validation, 2=inject failure after creating slot zero,
   4=request an unsupported FD handle on this Win32 ABI (negative control). */
PV0_API int pv0_open(const uint8_t uuid[16], uint32_t words, uint32_t flags,
                     const wchar_t* compute_spv, void** context);
/* Import these borrowed NT handles into CUDA, then drop the originals here.
   Dedicated allocation, offset zero, import size allocation_bytes, logical
   mapping size buffer_bytes, opaque Win32 binary semaphores. */
PV0_API int pv0_export(void* context, uint32_t slot, Pv0Export* out);
PV0_API int pv0_drop_exports(void* context, uint32_t slot);
/* Initially Vulkan signals writable on both slots. CUDA consumes writable
   once, fills all uint32 words with marker, then enqueues signal(ready).
   Submit waits ready, validates and writes input ^ 0xa5a55a5a ^ index on
   the GPU, then returns ownership and signals writable. Finish returns
   GPU audit counts; an observed mismatch is reported in data. */
PV0_API int pv0_submit(void* context, uint32_t slot, uint32_t marker);
PV0_API int pv0_finish(void* context, uint32_t slot, uint64_t timeout_ns, Pv0Result* out);
/* Synchronize CUDA, free mappings and destroy all CUDA imports first.
   On a drain timeout, close preserves its owner and all native objects. */
PV0_API int pv0_close(void** context);
PV0_API const char* pv0_error(void);
PV0_API uint64_t pv0_live(void);
/* NULL reports process-wide cumulative validation errors, including cleanup. */
PV0_API uint32_t pv0_validation_errors(void* context);

#ifdef __cplusplus
}
#endif
