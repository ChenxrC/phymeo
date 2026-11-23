# Independent sparse motion delivery

This directory is a caller and saved-data auditor. It changes no renderer,
shader, prepared source or frozen B/quad script. A prepared sparse runtime is
activated before any renderer import. Exactly one prepared profile is admitted
per process. Different prepared profiles use separate processes/results.

## Capture and film

```powershell
python rendering/examples/preview_sparse_motion/run.py capture --runtime C:/Users/38347/.codex/worktrees/p65-forest-tests/phymeo/artifacts/rendering/nov24-worktrees/appearance/rendering/examples/preview_sparse/runtime.py --prepared PREPARED_T4P16G2K25 --baseline-library OLD_DLL --baseline-shaders OLD_SPV_DIRECTORY --output NEW_RESULT --device-index 1 --channels rgb all --rounds 2 --film sparse
python rendering/examples/preview_sparse_motion/run.py encode NEW_RESULT --media-output NEW_MEDIA
```

Use `--quick` for the six registered states0/59/60/149/239/299 (a montage, not a
300-state run). `--poses` appends other registered indices;59/60/149 are always
retained. Full mode is exactly original300 states in chronological order. Every
owner has10 warmups. Round1: baseline RGB, baseline all, sparse RGB, sparse all;
round2 reverses these configuration/channel jobs, never the state order.
`--rounds 1` is an explicitly shorter measurement. `--resolution 540p|360p` is a
real smaller public camera/output, not upscaled720p or a720p quality pass.

The scene, recorded states, detail camera, normal map, scanned roughness, exposure8,
PBR Neutral, direction lights, studio/HDR and near-field mirror are shared. Baseline
is actual full-rate one-hop16 proposals with `guided_settling`. Its grid equals the
prepared sparse contract. Sparse uses the public `SparseSession.capture`; it has
no temporal accumulation. It must not inherit the baseline's nominal16spp or
history assertions. The source manifest and DLL/SPV hashes bind the two profiles
separately; loaded hashes, device UUID, decoded geometry, actual partitions,
capture counts and final allocation/validation cleanup are checked.

Sparse `primary_proposals` = final visible guide anchors x configured proposals.
It excludes hidden overdraw/helper invocations and does not count actual hardware
queries; a hemisphere-rejected proposal may issue no ray. `fallback_pixels` is
reported explicitly and must not be described as traced indirect illumination.
Reconstruction revision is bound, with missing old fields interpreted as revision1.
Profile/media names include tile, paths, grid, neighbors and reconstruction revision.

Only the designated `--film sparse|baseline16` winner retains all first-round
PNGs. Other groups/rounds retain the three registered PNGs plus all300 identities,
RGB/sensor hashes, actual sparse counters and timing records. Default`--film none`
is a screening run. No large linear array dump is needed for motion. Full all-mode
first-round registered poses also retain actual ID/depth/valid NPZs. All-mode
baseline/candidate geometry is checked for every measured state. Same-mode RGB/all
and repeated fresh rounds must reproduce the same actual RGB bytes.

Outputs:
- `summary.json`: source bindings, contract, actual platform, round order, separate
  profiles, nearest-rank P50/P95/max, group means and pooled completed timings.
- `inputs.json`: full registered300 frame/camera/scene/light declarations.
- `round-1/sparse/rgb/frames/0000.png` through`0299.png` when sparse is the winner.
- `round-1/{baseline16,sparse}/all/pose-0059.npz` etc: original registered sensors.
- `NEW_MEDIA/panda-sparse-t4-p16-g2-k25-r2.mp4` for that actual profile; quick movies
  have`-diagnostic`. Exact name follows the saved contract, not a hardcoded claim.

Wall intervals begin immediately before public capture and end on its return with
CPU-owned planes. GPU timestamps/copy categories are separate, not added to CPU
wait. Owner opening,10 warmups, consumer hashing/diagnostics and after-close PNG
encoding are recorded separately. Both timing rounds execute sequentially on the
same device; these are observed completed timings, not GPU presentation or an
asynchronous30FPS guarantee. No quality acceptance is inferred from latency.

The media adapter hardlinks original PNGs into a new source view (copy fallback
across filesystems) and reuses frozen B encoding/QA: exact PNG hashes/pixels,
30Hz PTS, dimensions/count, H264/yuv420p limitedBT.709 conversion, sRGB tag, SAR1,
faststart and full decode. It does not recolor/upscale images. A new media target
can retry encoding without recapturing; original captures remain unchanged.

## Shared one-hop reference and independent scoring

Existing genuine `preview_quad/validate.py --reference-batches 64` reference
records may be reused if every model/pose/camera/DLL/SPV/light binding matches.
They must be original one-hop1024-proposal references, not old two-hop appearance
figures. Alternatively capture only a new seed of the shared original baseline:

```powershell
python rendering/examples/preview_sparse_motion/reference.py --runtime SPARSE_RUNTIME_PY --prepared PREPARED --baseline-library OLD_DLL --baseline-shaders OLD_SPV_DIRECTORY --output NEW_REF_SEED1 --seed 1
python rendering/examples/preview_sparse_motion/reference.py --runtime SPARSE_RUNTIME_PY --prepared PREPARED --baseline-library OLD_DLL --baseline-shaders OLD_SPV_DIRECTORY --output NEW_REF_SEED2 --seed 2
python rendering/examples/preview_sparse_motion/assess.py --result NEW_RESULT --references NEW_REF_SEED1 NEW_REF_SEED2 --output NEW_ASSESSMENT.json
```

Each reference uses actual unchanged full-rate one-hop,64 batches x16 proposals,
independent seed and unfiltered final display at59/60/149; no sparse image serves
as truth. Reference PNGs/sensors/raw mean hashes and profile/readback costs are
saved. `--save-linear` optionally retains original RGBA32F_LE. The same references
can serve multiple candidates without recapture after strict input/model binding.

Assessment reuses the registered five identity masks and independent rigid-depth
correspondence implementation in`preview_quad/score.py`, hash-recorded in the
result. white-shell150/151/154, fingers156/158, rubber2,
left-arm112/113/115/117, wrist-shell125/142/149; no intensity-based selection.
Untouched RGB8 is inverted with IEC sRGB to display-linear, never called raw
radiance. ROI RMSE gate is candidate<=baseline+0.005; reference disagreement must
resolve0.0025 with two independent seeds before signing quality. Single references
still produce measurements but remain insufficient evidence. Signed RGB and linear
luminance bias are retained. The reference noise screen is not a confidence interval.

Temporal59->60 uses actual camera/rigid-instance/depth data at both endpoints,
same finger ID and eroded boundaries, depth tolerance.003 and round-trip.75pixel.
It subtracts the matched reference's physical brightness change; candidate residual
P95 must not exceed baseline. Fewer than16 reliable pixels is an evidence gap.
These are three poses/one adjacent pair, not all300-frame visual acceptance. The
thin-board69-pixel right-finger reveal belongs to another fixture and is not claimed.

## HOST checks (execution owner only)

```powershell
python -m unittest discover -s rendering/examples/preview_sparse_motion -p test_*.py -v
```

Authoring does not claim execution success. The tests cover timing rank, scope,
frame/sensor/repeated-byte identity, actual sparse counters and source revision,
plus independent-reference rejection for wrong model/seed/sampling/display.
### Existing sealed references and score shortcut

The already saved `artifacts/rendering/nov24/ref1` and`ref2` have genuine
`phymeo-preview-quad-validation-v1` reference-only groups, actual seeds1/2,
poses59/60/149,64batches x16, unfiltered one-hop, grid2/exposure8/PBR Neutral.
They are accepted directly if their complete runtime/input/geometry bindings match:

```powershell
python rendering/examples/preview_sparse_motion/run.py score NEW_RESULT --references C:/Users/38347/.codex/worktrees/phymeo-rendering/phymeo/artifacts/rendering/nov24/ref1 C:/Users/38347/.codex/worktrees/phymeo-rendering/phymeo/artifacts/rendering/nov24/ref2 --output NEW_SCORE.json
```

Every measured group has its own`identity_audit`, independent mean/P95/max and
`phase_timing`. The latter uses pre-existing film-view clock windows[0,.5),
[.5,2.3),[2.3,10] and explicitly is not inferred controller phase. RGB and all
remain separate even if all-mode exceeds the budget; pooled statistics do not
replace either round. Unique frame/source identities do not imply that every
physical joint configuration differs. A complete first-round designated movie
can be encoded even if an unrelated later group failed; the media source records
that enclosing partial failure. Scoring still requires complete baseline/candidate
all-channel groups and the genuine reference/registered-image evidence.

### Optional separately labelled aesthetic film

`--camera sequence --studio-look soft-studio` uses the original film's sequence
camera (detail only during.5..2.3s), the same recorded300 states and unchanged
materials/HDR/exposure. The explicit new lighting declaration removes standalone
directional lights and doubles panel dimensions with original total panel power.
It has a distinct scene key and remains a new presentation, not original lighting
acceptance. Defaults stay`--camera detail --studio-look recorded`. The scorer
rejects aesthetic configurations; filenames add camera/look labels. This makes
no promise that the untested artistic appearance is better.
## Explicit CompactBackend delivery

Use the producer's `preview_compact/runtime.py` with the new explicit
`--candidate-backend compact`. The prepared contract must contain
`compact.schema=phymeo-preview-compact-v1`; there is no fallback to SparseBackend.
The legacy group/source slot remains `sparse` for saved-reader compatibility, but
`candidate_backend`, complete contract, actual backend diagnostics and the movie
name identify **compact**. This is not a relabelled sparse capture. In compact,
full-rate reconstructed + fallback + nondiffuse equals visible pixels; low-raster
traced anchors are counted separately. Actual guide extent must be ceil(W/4) by
ceil(H/4), with one guide draw per successful RGB capture, five primary color
attachments and eight reconstruction sampled images. Per-frame counters and both
producer adapter source hashes are retained. No incoming-guide readback is needed
for this movie; its low-resolution pixels are never reshaped as a full image.

Registered quantitative capture (the references remain the same original-model
one-hop seed1/seed2 records, subject to every existing binding check):

```powershell
python rendering/examples/preview_sparse_motion/run.py capture --candidate-backend compact --runtime C:/Users/38347/.codex/worktrees/p65-forest-tests/phymeo/artifacts/rendering/nov24-worktrees/appearance/rendering/examples/preview_compact/runtime.py --prepared PREPARED_COMPACT --baseline-library ORIGINAL_BASELINE_DLL --baseline-shaders ORIGINAL_BASELINE_SPVS --output NEW_COMPACT_DETAIL --device-index 1 --resolution 720p --camera detail --studio-look recorded --channels rgb all --rounds 2 --film sparse
python rendering/examples/preview_sparse_motion/run.py score NEW_COMPACT_DETAIL --references C:/Users/38347/.codex/worktrees/phymeo-rendering/phymeo/artifacts/rendering/nov24/ref1 C:/Users/38347/.codex/worktrees/phymeo-rendering/phymeo/artifacts/rendering/nov24/ref2 --output NEW_COMPACT_SCORE.json
python rendering/examples/preview_sparse_motion/run.py encode NEW_COMPACT_DETAIL --media-output NEW_COMPACT_MEDIA
```

For a separately labelled presentation, choose `--camera sequence`: original
wide/detail/wide camera schedule (.5 <= time < 2.3 seconds is the finger view),
all 300 actual original states, original lights/material/exposure. `--candidate-only
--rounds 1 --channels rgb --film sparse` executes only the selected candidate and
retains its complete movie, with paired_baseline=false. This is a presentation run,
not the registered detail quality run; it cannot inherit detail quality gates.
Optional `--studio-look soft-studio` is a distinct authored lighting presentation,
not a performance/quality replacement for recorded lights. The full-resolution
movie name is `panda-compact-t4-p16-g2-k25-r2-visibility-ratio.mp4` for that exact
contract; sequence/lighting suffixes are appended when selected.

The fixed old vr2 receipt with `row.profile` holding a GPU timing dictionary is
still encodable: names are validated from the actual saved contract/counters,
never from that timing dictionary, and original receipts are never rewritten.
Use a NEW media directory (and its new `-source-view`) for an encoding retry.