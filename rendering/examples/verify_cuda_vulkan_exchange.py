"""Verify Vulkan storage-buffer ownership transfers with the Windows CUDA driver.

Build rendering/native/win32_interop_p0, then pass its DLL and compute SPIR-V.
This diagnostic reads back the GPU results on purpose. It does not measure cameras.
"""
import argparse
import ctypes as ct
from contextlib import ExitStack, contextmanager
import json
from pathlib import Path
import sys


class Export(ct.Structure):
    _fields_ = [(key, ct.c_uint64) for key in
                ("memory", "writable", "ready", "allocation_bytes", "buffer_bytes")]


class Result(ct.Structure):
    _fields_ = [(key, ct.c_uint32) for key in
                ("checked", "mismatches", "first_input", "last_input")]


def bind(library):
    dll = ct.CDLL(str(library.resolve()))
    ptr, u32, u64 = ct.c_void_p, ct.c_uint32, ct.c_uint64
    signatures = {
        "open": (ct.c_int, ct.POINTER(ct.c_uint8), u32, u32, ct.c_wchar_p, ct.POINTER(ptr)),
        "export": (ct.c_int, ptr, u32, ct.POINTER(Export)),
        "drop_exports": (ct.c_int, ptr, u32),
        "submit": (ct.c_int, ptr, u32, u32),
        "finish": (ct.c_int, ptr, u32, u64, ct.POINTER(Result)),
        "close": (ct.c_int, ct.POINTER(ptr)),
        "error": (ct.c_char_p,),
        "live": (u64,),
        "validation_errors": (u32, ptr),
    }
    for name, (result, *arguments) in signatures.items():
        function = getattr(dll, "pv0_" + name)
        function.restype, function.argtypes = result, arguments
    return dll


def vk(dll, name, *args):
    rc = getattr(dll, "pv0_" + name)(*args)
    if rc:
        message = (dll.pv0_error() or b"").decode("utf-8", errors="replace")
        raise RuntimeError(f"Vulkan {name} returned {rc}: {message}")


def cu(operation, *args):
    status, *values = operation(*args)
    if int(status):
        raise RuntimeError(f"{operation.__name__}: {status}")
    return values[0] if values else None


def verify_rejections(dll, uuid, words, shader, validation):
    controls = {}
    for name, selected, extra, expected in (
        ("foreign_uuid", bytes([uuid[0] ^ 255]) + uuid[1:], 0, "UUID does not match"),
        ("unsupported_handle_injected", uuid, 4, "unsupported handle type"),
        ("partial_initialization_injected", uuid, 2, "after first slot"),
    ):
        owner = ct.c_void_p()
        before = dll.pv0_live()
        rc = dll.pv0_open((ct.c_uint8 * 16).from_buffer_copy(selected), words,
                          validation | extra, str(shader), ct.byref(owner))
        reason = (dll.pv0_error() or b"").decode("utf-8", errors="replace")
        if owner:
            vk(dll, "close", ct.byref(owner))
            raise RuntimeError(f"{name} returned a live owner")
        if rc != 1 or expected not in reason or dll.pv0_live() != before:
            raise RuntimeError(f"{name} failed its cleanup contract: {rc}, {reason}")
        controls[name] = reason
    return controls


def map_slot(stack, driver, dll, ctx, slot, words):
    info = Export()
    vk(dll, "export", ctx, slot, ct.byref(info))
    if (not all((info.memory, info.writable, info.ready)) or
            info.buffer_bytes != words * 4 or info.allocation_bytes < info.buffer_bytes):
        raise RuntimeError("invalid exported slot sizes or handles")
    desc = driver.CUDA_EXTERNAL_MEMORY_HANDLE_DESC()
    desc.type = driver.CUexternalMemoryHandleType.CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32
    desc.handle.win32.handle = info.memory
    desc.size = info.allocation_bytes
    desc.flags = 1  # CUDA_EXTERNAL_MEMORY_DEDICATED, matching the Vulkan allocation.
    memory = cu(driver.cuImportExternalMemory, desc)
    stack.callback(cu, driver.cuDestroyExternalMemory, memory)
    mapping = driver.CUDA_EXTERNAL_MEMORY_BUFFER_DESC()
    mapping.size = info.buffer_bytes
    address = cu(driver.cuExternalMemoryGetMappedBuffer, memory, mapping)
    stack.callback(cu, driver.cuMemFree, address)
    semaphores = []
    for handle in (info.writable, info.ready):
        desc = driver.CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC()
        desc.type = driver.CUexternalSemaphoreHandleType.CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32
        desc.handle.win32.handle = handle
        semaphore = cu(driver.cuImportExternalSemaphore, desc)
        stack.callback(cu, driver.cuDestroyExternalSemaphore, semaphore)
        semaphores.append(semaphore)
    vk(dll, "drop_exports", ctx, slot)
    stream = cu(driver.cuStreamCreate, driver.CUstream_flags.CU_STREAM_NON_BLOCKING)
    stack.callback(cu, driver.cuStreamDestroy, stream)
    return address, *semaphores, stream


@contextmanager
def imported_slots(library, shader, device, words, validation):
    """Hold both APIs and their imported slots until the consumer finishes."""
    from cuda.bindings import driver

    dll = bind(library)
    cu(driver.cuInit, 0)
    dev = cu(driver.cuDeviceGet, device)
    uuid = bytes(cu(driver.cuDeviceGetUuid, dev).bytes)
    shader = shader.resolve(strict=True)
    before = dll.pv0_live()
    errors_before = dll.pv0_validation_errors(None)
    with ExitStack() as resources:
        primary = cu(driver.cuDevicePrimaryCtxRetain, dev)
        resources.callback(cu, driver.cuDevicePrimaryCtxRelease, dev)
        cu(driver.cuCtxPushCurrent, primary)
        resources.callback(cu, driver.cuCtxPopCurrent)
        owner = ct.c_void_p()
        vk(dll, "open", (ct.c_uint8 * 16).from_buffer_copy(uuid), words, int(validation),
           str(shader), ct.byref(owner))
        resources.callback(vk, dll, "close", ct.byref(owner))
        slots = [map_slot(resources, driver, dll, owner, i, words) for i in range(2)]
        yield driver, dll, owner, slots, uuid
    errors = dll.pv0_validation_errors(None) - errors_before
    if dll.pv0_live() != before or errors:
        raise RuntimeError(f"interop teardown or validation error: live={dll.pv0_live()}, errors={errors}")


def run(library, shader, device, words, validation):
    records = []
    with imported_slots(library, shader, device, words, validation) as (driver, dll, owner, slots, uuid):
        controls = verify_rejections(dll, uuid, words, shader.resolve(), int(validation))
        dev = cu(driver.cuDeviceGet, device)
        name = cu(driver.cuDeviceGetName, 256, dev).split(b"\0", 1)[0].decode()
        version = cu(driver.cuDriverGetVersion)
        wait = driver.CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS()
        signal = driver.CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS()
        for generation in range(3):
            markers = []
            for i, (address, writable, ready, stream) in enumerate(slots):
                marker = 0x34560000 | (generation << 8) | i
                wrong = generation == 1 and i == 1
                if generation == 0:
                    cu(driver.cuWaitExternalSemaphoresAsync, [writable], [wait], 1, stream)
                # On reuse, the previous result check already acquired this slot.
                cu(driver.cuMemsetD32Async, address, marker, words, stream)
                cu(driver.cuSignalExternalSemaphoresAsync, [ready], [signal], 1, stream)
                vk(dll, "submit", owner, i, marker ^ int(wrong))
                if i == 0 and generation == 0:
                    if dll.pv0_submit(owner, i, marker) != 1 or b"pending fence" not in dll.pv0_error():
                        raise RuntimeError("pending Vulkan slot was not protected against reuse")
                    controls["pending_slot_reuse"] = "rejected before the original finish"
                markers.append((marker, wrong))

            # Finish in reverse order. Each slot has its own fence and generation.
            for i in (1, 0):
                address, writable, _, stream = slots[i]
                marker, wrong = markers[i]
                result = Result()
                vk(dll, "finish", owner, i, 10_000_000_000, ct.byref(result))
                expected_errors = words if wrong else 0
                if (result.checked, result.mismatches, result.first_input, result.last_input) != (
                        words, expected_errors, marker, marker):
                    raise RuntimeError(f"Vulkan compute check failed for slot {i} generation {generation}")
                cu(driver.cuWaitExternalSemaphoresAsync, [writable], [wait], 1, stream)
                cu(driver.cuStreamSynchronize, stream)
                returned = (ct.c_uint32 * words)()
                cu(driver.cuMemcpyDtoH, ct.addressof(returned), address, ct.sizeof(returned))
                if any(value != marker ^ 0xa5a55a5a ^ j for j, value in enumerate(returned)):
                    raise RuntimeError(f"CUDA observed stale or incorrect Vulkan results in slot {i}")
                records.append(dict(slot=i, generation=generation, marker=marker,
                                    vulkan_words=result.checked, injected_bad_marker=wrong,
                                    observed_mismatches=result.mismatches, cuda_verified_words=words))
    return dict(device_uuid=uuid.hex(), device_name=name,
                driver_version=version, words_per_slot=words, slots=2,
                validation_enabled=validation, validation_errors=dll.pv0_validation_errors(None),
                native_live_after=dll.pv0_live(),
                negative_controls=controls, exchanges=records,
                cuda_oracle_readback_bytes=sum(row["cuda_verified_words"] * 4 for row in records))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--shader", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--words", type=int, default=257)
    parser.add_argument("--validation", action="store_true")
    args = parser.parse_args()
    if sys.platform != "win32" or args.report.exists() or not 1 <= args.words <= 4096:
        parser.error("requires Win32, 1..4096 words and a new report path")
    report = run(args.library, args.shader, args.device, args.words, args.validation)
    with args.report.open("x", encoding="utf-8") as out:
        json.dump(report, out, indent=2)
        out.write("\n")
    print(f"{report['device_name']}: verified {len(report['exchanges'])} CUDA/Vulkan round trips")


if __name__ == "__main__":
    main()
