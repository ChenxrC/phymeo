# I3c: heterogeneous area contact and unified observation

2026-11-17; codex/exp starts at 47c38bcd. Root owns shared source and the sole serial execution slot. Original agents author tests/review only.

## Contract
- Extend ModelSpec with immutable ordered area_pressure_pairs. Exact bucket identity and checkpoint binding include this declaration; native declarations retain their old binding.
- Preflight/allocate area plans with their real allocation manifest. Public batch configuration IDs bind surfaces to current leaf state. step_area_pressure requires exactly all area worlds and all declared pairs, including empty surfaces. Pack every bucket before any physics, then reuse the existing whole-batch draft/validate/commit transaction. Clear one-shot packets on every path. Plain step cannot silently omit area forces.
- prepare_device_workspace(include_area_pressure=True) also includes native accepted-entry observations. A shared frame carries padded endpoint and input state, native contact rows, and the four I3b area fields. area_declared_mask separates nonarea models from invalid initial/reset records and valid empty records.
- Area numerics are scattered from accepted device banks, never rebuilt from HOST receipts. Original FP64 dA words and moment-valid flag remain intact. Pair ordinals are scoped by model and frame catalogs; world IDs retain caller order. HOST catalogs contain identities only.
- All new observations stage in the workspace inactive bank. Failed refresh preserves the prior frame; state/control/reset/restore revision changes stale it. Existing readback and copy-owned CUDA to_torch export all fields together. No borrowed writable physics storage.
- Native and area batches may coexist. No automatic mixed-model HOST clipping is introduced: caller supplies each model's current surfaces using the public stamps. No physics, friction, tolerance or capacity changes.

## Fixed verification
Short interpret and CUDA cases cover two distinct area configurations/dimensions/capacities plus native-only worlds; unordered keys and shared buckets; per-world independent result agreement, masks/padding/exact area words; missing/stale/over-capacity preflight; last-bucket physics and observation failure; checkpoint/selective reset/restore owner replacement; same model with different declarations cannot coalesce; identity-only HOST metadata during refresh; copy-owned Torch export. Run affected native heterogeneous transaction/resource/IO seams only as justified by modified paths, not previous full trajectories.

Each project execution uses a fresh entrypoint-aware guard and original environment. Preserve first failures; diagnose before changes. No installs, CI, push or dev modifications.

## Exit and next
Commit implementation/tests after focused checks; report exact passes and remaining limitations. Next: current-geometry generation across heterogeneous scenes, then GPU finite-surface clipping with total-step cost. Area observations alone do not establish heterogeneous task training, FFT coupling or end-to-end device collision.
