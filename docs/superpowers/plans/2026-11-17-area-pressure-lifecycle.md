# I3a: area-contact HOST observations and lifecycle

Date: 2026-11-17. Branch: codex/exp. Starting HEAD: b90b0506.
Parent owns shared seams and is the sole project execution owner. Original agents only author/review without execution.

## Goal and scope

Expose the accepted area load with its input phase, independently of the current endpoint. Continue the same trajectory through checkpoint/restore/clone and isolate subset resets. Reuse the existing runtime checkpoint transaction and native entry ABI. Geometry remains HOST; no device-resident area observation, GPU clipping, friction or FFT material integration is claimed.

## Implementation

1. Keep the original immutable quadrature in each accepted area receipt, alongside the actual FP32-converted row inputs and published loads.
2. Return owned read-only HOST observations containing world, historical identity/stamp, input and output ticks/times, endpoint q/v, held control from that step, quadrature, actual rows, and force/moment sums. Moments explicitly distinguish world origin, input body origin and input COM. No kernel launch, new geometry or CPU physics step.
3. Extend runtime capsule binding with ordered area pairs/body IDs, law and response operator. Whitelist area records in the bounded codec. Validate source quadrature, actual conversion, row mapping and force/motion publication before replacement allocation. Preserve historical configuration_id; current controls may differ from the last receipt controls.
4. Add a scene capsule binding every local pressure/rigid asset coordinate, topology, pressure and material identity. Restore delegates to the existing atomic runtime transaction only after asset checks. Geometry cache clears on successful restore/reset/clone and is regenerated from the next accepted state.

## Fixed validation

New lifecycle tests run interpret then CUDA, in the single guarded root queue:
- Dual support: 30 steps, checkpoint, 50 more; bytes roundtrip/restore and identical 50-step continuation. Compare all physical values, normalize only historical configuration identity when comparing newly generated runs.
- Independent owner clone, reordered world subset, source mutation isolation.
- Observation is immutable and causes no launch; input force phase and endpoint state remain separate; controls-only update preserves history; selective reset invalidates only the reset world.
- Different asset or ordered pair binding, re-checksummed source/row/load tampering, and last readback failure reject without replacing accepted state or geometry.
- Offset-COM hinge verifies world/body/COM moments rather than testing coincident origins.

Affected original area tests run because their schema/checkpoint contract changes. Native checkpoint compatibility uses the existing complete exact-continuation function (four model cases, both modes), not the old long physics windows. Existing I2b/I2c trajectories and old certificate audits are reused, not rerun.

Keep first command failure logs, source snapshots, focused receipts and saved new checkpoints. A failed case gets a diagnosis before modification; do not alter fixture physics or loosen error thresholds to pass. Report actual results and remaining device export limitations, then commit exact scoped files. No dev merge, push, installation, remote CI or cleanup.
## Completion

Implemented and verified on 2026-11-17. Observation names are `applied_control` for the historical stored control and `current_control` for current held input. Four primary groups passed: new lifecycle interpret 6 / CUDA 6, affected area 31, exact native checkpoint compatibility 8. An additional ordered-pair binding test passed both modes (2); no already-green group was repeated. New tests also check missing native-entry rejection and phase tampering. Final Ruff and format check passed on all seven changed Python files. Root execution is complete; see the lifecycle report for use and limits.