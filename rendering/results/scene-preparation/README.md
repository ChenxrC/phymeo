# Thirty-object room preparation and frame timing

This benchmark measures preparation, opening a native session, and delivering RGB to Python as separate operations. It uses the 30 original household assets and the room tour's 13 cameras. All captures use 1440 × 810, 4× MSAA, 16 samples per area light, ray-query shadows and the scene's HDR lighting. They use the native pinhole image. The film's additional supersampling, background compositing, resizing and video encoding run outside this benchmark.

## Measured on RTX 4060 Laptop GPU

The source assembly took 30.243 s once. Both preparations yielded the same 210 mesh and instance bindings, 524,257 triangles and 476,558 vertices. After the first preparation the derived cache contained 309 files totalling 55,618,450 bytes. The second run retained those totals.

| Derived cache at preparation | Prepare | Open session | Host capture p50 / p95 | GPU render p50 / p95 | GPU copy p50 / p95 | GPU total p50 / p95 |
|---|---:|---:|---:|---:|---:|---:|
| Empty | 112.606 s | 13.725 s | 39.60 / 48.13 ms | 32.73 / 40.99 ms | 0.36 / 0.79 ms | 33.23 / 41.48 ms |
| Populated | 34.641 s | 13.920 s | 40.67 / 51.23 ms | 33.01 / 41.27 ms | 0.47 / 0.91 ms | 33.73 / 41.86 ms |

Cache reuse saved 77.97 seconds of startup preparation in this run. Both sets of hot frames exceeded the 33.33 ms host budget for 30 FPS at the median. In the populated-cache session, the scale-and-seasonings view (frame 246) had a 41.87 ms median GPU total, followed by the breakfast dishes (frame 345) at 41.40 ms. The profiles put most GPU time in rendering, with less than 1 ms at p95 in the attachment-copy interval. Each session delivered 39 measured frames and 39 valid timestamp samples with a 1 ns period and 64 valid bits. The 13 camera RGB hashes matched throughout both sessions, and Vulkan reported zero validation errors.

The full per-exposure record for this run is at `artifacts/home-showcase-30/performance-preparation-01/benchmark.json` with SHA256 `c8fffe74ab5baf7109484d0d77cd3eb4b5d27c6f6d41efad82aed257c73a489b`. The native DLL SHA256 was `d7f9221226fcc0cbef3725599e142c6cea009c4b4c48f810e088b64fb98848cb`.

## Reproduce

Use the same Windows Python environment as the room film, with NumPy, Pillow, OpenCV and VTK installed. Run the following from the repository root. Set the first three paths for the machine under test. The output directory must be new. The benchmark creates a private derived cache inside it and leaves the source asset pack untouched.

```powershell
$assets = 'C:/Users/38347/Desktop/同元/美工素材_135个资产_20261118/资产'
$python = 'C:/Python313/python.exe'
$library = 'artifacts/physics-vulkan-interop/p1-raster-20261124/build/phymeo_native.dll'
$env:PYTHONPATH = (Resolve-Path src).Path + ';' + (Resolve-Path rendering/src).Path + ';' + (Resolve-Path rendering/examples).Path + ';C:/Users/38347/AppData/Roaming/Python/Python313/site-packages'
$env:VK_LAYER_PATH = (Resolve-Path 'artifacts/rendering/g3-table/toolchain/Validation_Layers_Windows_64').Path
$env:VTK_SMP_MAX_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
& $python -s -B -u rendering/examples/benchmark_scene_preparation.py `
  --assets $assets --asset-cache artifacts/home-showcase-30/cache --library $library `
  --device-index 1 --warmup-tours 1 --measured-tours 3 `
  --output artifacts/home-showcase-30/benchmark-fresh
```

## Reading the measurements

- The benchmark imports and assembles the full source scene once. Both preparation calls receive the same scene, calibrated cameras and protected oak geometry. “Cold” and “warm” refer to the private derived-resource cache. The operating system and graphics driver manage their own caches.
- `prepare_scene_s` measures the full public call, including its device-limit query and asset validation. Each case then opens a separate native session. `open_scene_s` covers the public open call, which includes resource decoding, pipelines and upload. First-capture target creation is part of warmup.
- Each session visits all 13 views once to warm up, then three more times for measurement. Host time surrounds the synchronous public RGB capture, including its readback. GPU render, copy and total use deltas of cumulative Vulkan timestamp counters around one capture. Missing samples are marked unavailable. Host and GPU times overlap.
- Device diagnostic queries, SHA256 checks and reporting occur outside the host frame timer. All measured RGB frames must be byte-identical to the warmup for the same pose. The two sessions must also agree on all 13 RGB hashes, the physical device and the prepared mesh and texture contents.
- `benchmark.json` in the output directory contains per-frame times, per-camera statistics, cache counts, exact source and library hashes, and the prepared resource identity. The percentiles use linear interpolation across the measured samples.
