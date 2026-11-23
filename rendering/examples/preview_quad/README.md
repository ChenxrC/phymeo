# Experimental Panda first-hop quad preview

This directory is an isolated experiment. It does not add a public renderer setting.
The base DLL and every original SPV remain untouched. Only the copied grid2,
normal-map + reflection + indirect-progressive fragment is replaced.

```powershell
python rendering/examples/preview_quad/build_overlay.py --source-shaders rendering/src/phymeo_rendering/backends/_shaders --base-shaders BASE --output NEW_Q2 --samples 2 --compiler PATH_TO_GLSLANG
python rendering/examples/preview_quad/validate.py --library DLL --base-shaders BASE --overlay NEW_Q2 --output NEW_RUN --device-index 1 --save-linear
```

Build a separate NEW_Q4 with `--samples 4`. `--no-share` binds a same-budget,
no-spatial-reuse control to a different marker. Add `--reference-batches 64`
for a real independent seed1, 1024-proposal same-pose baseline reference.
Defaults: original 1280×720 detail camera, poses59/60/149, same fixed assets,
normal map, exposure8, panel grid2, single near-field mirror, one diffuse bounce,
guided_settling. `--filter none` is the separate unfiltered diagnostic.
These few poses are not a 300-frame acceptance or a 30FPS claim.

The builder pins the baseline main/indirect normalized-text SHA and replaces one
exact invocation. Original direct, mirror, sensor writes and existing MIS helper
functions remain as authored. All other copied binaries are hash-checked.
The source and original binary build provenance must refer to the same baseline;
the manifest records both, but does not prove the original binary compilation.

2 proposals = 1 cosine + 1 HDR-CDF, 4 = 2 + 2. Each component covers its own
equal-probability strata. Both use the original balance-mixture PDF; rejected
hemisphere directions stay zero under the fixed denominator. Without an active
importance environment, all proposals use stratified cosine sampling. Only one
diffuse bounce is allowed; the shader finite-value gate rejects a two-hop request.

A live ballot excludes helpers/uncovered/ineligible fragments. Every constant
quad-source broadcast is inside a quad-uniform ballot-bit guard, before any
per-pixel compatibility test. Never fetch an inactive source then mask its value.
Compatibility: same instance and instance slot, positive finite forward depth
difference <=0.003 of the larger depth, shading-normal dot>=0.995 and geometric
face-normal dot>=0.999. Only demodulated incoming is averaged; each receiver's
original albedo/diffuse factor is applied once by the unchanged studio shader.
Spatial reuse can bias varying irradiance or secondary occlusion; this is an
experimental preview, not an unbiased per-pixel estimator or approved quality tier.
There is no primitive-ID equality gate. Do not infer a cross-triangle guarantee from the active/helper and primary-geometry checks.

Public diagnostics still say16. The hash-bound named SPIR-V markers (120/121/122)
are authoritative for this experiment's actual 2/4 primary proposal budget.
A neighbor-reused estimate is not 8/16 rays launched from the current pixel.
These are proposal budgets, not measured hardware ray-query/hit counts.

glslang -H supplies readable compiled SPIR-V. The standard-library parser checks
capabilities61/64/68, instructions339/341/365, and named uint32 specialization
defaults. It is not a complete SPIR-V validator. Optional spirv-val/dis are used
when available; --strict-validation requires independent spirv-val. Otherwise the
actual validation runner requires Vulkan validation layers during real pipeline
creation/capture. No silent device support fallback: it independently queries
fragment BASIC+BALLOT+QUAD and binds the resulting device UUID to the actual session.

Validation writes original PNGs, optional RGBA32F_LE scene-linear mean,
frame/camera/source bindings, all three sensor hashes, per-capture wall and GPU
profile deltas, raw-read overhead, legacy vs actual budgets and final allocation
telemetry. Baseline16 is noisy and cannot serve as a truth image; optional
independent reference errors are descriptive, not a new acceptance threshold.

HOST admission only (parent executor):
```powershell
python -m unittest discover -s rendering/examples/preview_quad -p test_preview_quad.py -v
```

Normative references:
- https://docs.vulkan.org/glslext/latest/glslext/khr/GL_KHR_shader_subgroup.html
- https://docs.vulkan.org/spec/latest/chapters/shaders.html
- https://raw.githubusercontent.com/KhronosGroup/SPIRV-Headers/main/include/spirv/unified1/spirv.hpp


## Bound motion and media wrapper

The frozen B script is reused without changes. Every output group is explicitly
named baseline16, quad2, quad4, local2 or local4 plus its actual pixel extent.
Only one diffuse bounce, grid2 and the original fixed scene/camera/exposure are
admitted. No overlay is inserted into the old multi-preset runner.

```powershell
python rendering/examples/preview_quad/motion.py capture --library DLL --base-shaders BASE --overlays NEW_Q2 NEW_Q4 --output NEW_MOTION --resolutions 720p --channels rgb all --rounds 2 --device-index 1 --validation-proof VALIDATED_Q2/summary.json VALIDATED_Q4/summary.json --baseline-cache OLD_B_CAPTURE
python rendering/examples/preview_quad/motion.py encode NEW_MOTION --media-output NEW_MEDIA
```

For a six-pose montage/screen, append `--quick --resolutions 720p 540p 360p` and
omit incompatible full-run caches. Otherwise each fresh group is 10 warmups +
300 completed chronological states. Round2 reverses configuration/channel order,
not state chronology. RGB/all results and same-extent geometry must match;
fresh repeated groups must match all RGB/sensor/frame/camera hashes. Full RGB
PNGs are retained in round1; `--keep-second-round` additionally keeps all round2
PNGs. Media names include `panda-quad2-720p.mp4`; quick clips end `-diagnostic`.
Encoding validates every PNG, actual dimensions, 30Hz PTS, full decode, SAR1,
BT.709 limited YUV and sRGB tags. Playback cadence is not measured throughput.
Encoding failure can use a new `--media-output` without another capture.

`--baseline-cache` only borrows an admitted, same-DLL/full-SPV/native-Python,
same-declaration/decoded-geometry and exact ordered state/camera group. Both the
historically executed script SHA and the current SHA are recorded. The only
nonidentical-script admission is an exact hash reconstruction that changes
terminal empty lines (up to8), never an arbitrary semantic-compatibility claim.
Cached RGB/receipt/PNG identities are retained. Referencing that receipt in two
rounds is one historical measurement, not two newly interleaved rounds; pooled
latency deduplicates it. No cached images are recopied or re-encoded implicitly.

`experiment.json` binds both rounds, every overlay/marker/sample budget, subgroup
UUID and validation proof. Per-frame runtime nominal16 counters remain unchanged;
actual2/4 proposal budgets are separate fields. Timings are completed synchronous
capture wall intervals (RGB or all planes) and native GPU profile deltas, plus
separate owner opening/warmup/PNG overhead. `geometry/720p/pose-0059.npz`, 0060 and
0149 contain actual ID/depth/valid from separate sensor-only captures, excluded
from motion timing. `--no-geometry` omits them but disables this scorer's geometry
binding. Quality is not inferred from latency or a playable movie.

## Shared reference and independent ROI audit

The existing `validate.py --reference-batches 64` captures seed1. Reuse its bound
reference across candidates. A second reference can be captured without repeating
any candidate or motion group:

```powershell
python rendering/examples/preview_quad/reference.py --library DLL --base-shaders BASE --overlay NEW_Q2 --output REF_SEED2 --seed 2 --device-index 1
python rendering/examples/preview_quad/score.py --motion NEW_MOTION --references VALIDATED_WITH_REF1 REF_SEED2 --variant quad2-720p --output NEW_QUAD2_SCORE.json
python -m unittest discover -s rendering/examples/preview_quad -p test_motion.py -v
```

References are actual original one-hop, 16proposals x64batches, independent seed,
same model/DLL/allbase-SPVs/camera/geometry/exposure and unfiltered final display.
A previous two-hop or cosine-only image is not accepted as a one-hop oracle.
Five masks use the frozen Q0 identities exactly: white-shell150/151/154,
fingers156/158, rubber2, left-arm112/113/115/117, wrist-shell125/142/149.
No intensity selection. Pose59/60/149 are fixed; the registered temporal pair is
59->60. Candidate and baseline errors use untouched displayed RGB8, inverted IEC
sRGB (display-linear, not raw scene radiance). Static allowance is baseline RMSE
+0.005. Two independent references must disagree by <=0.0025 RMSE in each ROI
before a quality pass is supported; this is a finite noise-resolution screen,
not a confidence interval. A single seed still yields measurements, but returns
insufficient evidence rather than silently passing quality.

Finger temporal correspondences use actual camera/rigid-instance transforms and
forward depth at both endpoints, same156/158 ID, 3x3 eroded endpoints, relative
depth<=0.003 and round-trip reprojection<=0.75pixel. Reference true brightness
change is subtracted; candidate residual P95 must be <=baseline P95, with >=16
reliable pixels. Fewer pixels are an evidence gap, not a relaxed gate. This is
one adjacent pair, not300-frame temporal acceptance. The thin-board69-pixel
reveal belongs to the separate guided-motion fixture and is not claimed here.