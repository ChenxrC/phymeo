# I3b: persistent device area observations

2026-11-17; codex/exp starting 7cd039e4. Root owns shared seams and sole serial project execution. Original agents only author/review.

## Scope and contract
Same-model multiworld batch device export, preserving the existing native seven-field ABI. Optional area banks live in the same allocation manifest and double-buffered physical owner. The accepted active index publishes state and area record together; failed trials may dirty only inactive/scratch storage.

Four extra fields: area_meta [valid,samples,positive-force,overflow] i32; area_ids [valid,pair-ordinal,global-sample-ordinal,body-a,body-b,world-moment-valid] i32; area_data [point3,normal3,k,phi,force,force-on-a3,moment-at-world-origin-on-a3] f32; area_measure_words [low32,high32] i32 encoding exact little-endian IEEE754 binary64 dA. Upload measure words with the original step inputs, not from a receipt after solving. Do not impose a new FP32 area admission condition. Zero padding, count/capacity checks, positive-force flags and phase bindings are explicit. Native entry q/v/clock/meta describe the force input phase; separate endpoint q/v/clock describe accepted output.

Public runtime.area_pressure_device_layout describes caller buffers. copy_area_pressure(outputs, expected_revision=...) checks all layouts/devices/ownership/aliasing before any write; then synchronous interpret copy or CUDA D2D, returning immutable world/pair/source catalog metadata. No HOST numerical reconstruction, state/force download, or new collision/physics step. Full material/sample string identities remain in the bound immutable HOST catalog; device ordinals index it. Caller buffers belong to caller. A copy failure returns no frame and leaves physics unchanged; partly written caller output is undefined until a successful call.

Checkpoint/reset/clone copy and validate actual area banks with original native entry. Old I3a area capsules are explicitly rejected by new layout binding, never fabricated from HOST history. Native nonarea capsules retain original compatibility. Source catalog remains historical through controls-only changes.

## Fixed verification
Five new short tests in interpret and CUDA: (1) multiworld device values versus HOST accepted receipts, exact dA bits, no HOST producer/readback; (2) initial invalid vs successful empty, old output independence; (3) checkpoint/restore and reordered/subset clone device bytes plus selective reset; (4) stale/missing/wrong dtype/shape/readonly/overlap/physical aliases reject before writes; (5) late download and capacity failures preserve accepted banks. Add focused capsule-corruption negative if needed.
Affected I3a lifecycle tests and native entry compatibility are selected because storage changes. No old long support/impact runs or certificate re-audits. Fresh guard for every import/test/lint/formatter; no install/CI/push/dev changes.

## Limits and next
No heterogeneous model assembly, GPU geometry clipping, device material-string interning, friction, FFT material law, performance claim or new training outcome. Next add a common observation consumer/heterogeneous aggregation, then measure device geometry generation as a separate unit.

## Completed verification

Six new functions ran in both modes: interpret 6 passed (6.06 s), CUDA 6 passed (6.15 s). The extra direct-stage case rejects INT_MIN counts before index arithmetic and marks unavailable extreme-world-origin FP32 moment without failing physics. World moments use a validity flag and all-zero sentinel when either FP32 product exceeds 2**125 or is nonfinite; HOST FP64 moments remain unchanged. Area lifecycle/mixed native compatibility 16 passed (30.37 s), native entry/observation lifecycle 4 passed (20.50 s). All ten Python files passed final Ruff and format check. No numerical failures or changed physics parameters; original import sorting and one invalid CLI selector receipt retained. Device output tests include preflight mapping mutation/alias refusal, exact measure words, copied-bank capsule tamper and explicit old I3a capsule rejection.
