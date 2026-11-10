# 2026-11-11 overnight implementation record

Execution deadline: 08:00 Asia/Shanghai; no new implementation scope after 06:30.
Parent alone executes projects. This is a rolling record; pending code is not an accepted capability.

## First integrated qualification, through approximately 03:00

| Capability | Actual result | Boundaries |
| --- | --- | --- |
| A1 accepted-state checkpoint/restore | 24 interpret and 23 CUDA nodes passed; exact continued public states and receipts | Tested on parent 6ba6ec40. First run exposed native mj_setState requiring writable input; two detached copies fixed that ABI issue without changing the immutable capsule. Controller memory remains explicitly composed by the caller. |
| C1 world-complete loaded pages | 17 interpret and 17 CUDA cases passed, including W127/128/129/256 and four robot layouts | One solve/publication per storage; page ordering, partial reset, last-page failure and capacity controls covered. This resident path is distinct from MuJoCo-TYX. |
| C2 device task output workspace | 13 interpret and 12 CUDA nodes passed | Device q/v to observation/reward/done and real TYX consumer. Full physical HOST audit remains. CUDA Torch/PPO is not qualified. The expected overflow warning belongs to the explicit nonfinite-output rejection case. |
| B1 expanded device patches and sources | First interpret run: 22 passed, then cycle-partial_bh failed | Generic polar ordering omitted the negative horizontal ray. Seven same-form comparators corrected in 5d827a73; precise and full requalification pending. No oracle or tolerance changed. |
| C2b device action input | Six-file authored delivery integrated; validation pending | Native three-channel copy-owned action packet, stream/CAS and lifecycle; physics kernels unchanged. C2b modifies runtime seams, so its focused suite and affected checkpoint continuation still must pass. |

The first combined style checks retained their five import-order findings and 17 format findings. Subsequent exact 21-file checks passed; 18 raw ASTs matched, all 21 scoped import bindings and non-import ASTs matched. The only parent public-API addition was the three contact-page exports. Old unrelated untracked K1 files remain untouched.

### Subsequent qualification through approximately 03:25

C2b's complete interpret selection passed 22 nodes. First CUDA selection passed 17 and then failed in construction of the deliberate overlapping-buffer fixture: the installed TYX external-buffer API requires a Python integer address. Converting the existing live CUDA pointer to `int` preserved the intended alias; the exact failed node and the complete CUDA selection then passed (22 nodes). Product code did not change for this fixture repair. The affected A1 exact-checkpoint continuation selection passed four cases on each backend after the C2b integration. This qualifies the public device-action path, including the inline free-root/29-joint/29-actuator width case, not CUDA Torch or GPU PPO. Canonical action auditing and complete physical HOST diagnostics remain.

B1's polar-order repair passed the original failed interpret node and the complete 96-node interpret selection. First CUDA execution stopped before patch fill because the clear kernel had two top-level parallel roots. The index-preserving single-root clear repair passed seven interpret late-failure/retry cases. Its CUDA target advanced to patch fill compilation, exposing a local variable named `signed`, a C++ reserved word. That separate compilation repair and the complete CUDA qualification remain pending; B2 is still locked.

The public CUDA loaded-page benchmark completed seven fresh runs: bundle W1/32/128/256 and original single-page W1/32/128. Each used 16 continuing loaded steps without reset or warmup. All final states and complete receipts matched for the three comparable world counts. Bundle steady-step totals for 15 steps were 0.2013, 5.1813, 20.9735 and 41.7330 seconds respectively; first-step compilation/construction and checking times are reported separately in the raw result. These are measured local costs of this slider case with HOST assembly/diagnostics, not a Newton speed comparison or a general robot throughput claim.

## G1 original full-window attempt

Run 06 started 01:12:58 and ended 02:37:18. Requested 131.48 s at h=.002, no reset, original model/policy/initial state and engineering gates. It stopped after 56,186 successful device and independent CPU steps, at nominal 112.372 s.

The independent CPU reference crossed the original 1 m planar reference-error gate: 1.0016082401258426 m. Device maximum was 0.996658481524757 m. Device failed_batches remained zero. Original RAW_EXIT=1 and FAIL/incomplete status are retained; this is not a complete dance acceptance. Run 19's saved-evidence audit passed: all 56,186 attempt/step pairs, full gzip stream and declared schemas/bindings were checked. Its `PASS_EVIDENCE_AUDIT_ONLY` does not change the original trajectory failure.

Source-only deployment triage found that the fixed policy observation has no absolute XY-error feedback, plus specific frame-selection and IMU-axis protocol differences requiring bounded comparison. Neither difference is an established cause. No policy, model, physics gate or motion was changed, and no complete rerun was started.

The compensated device clock is checked against the actual FP32 step value, with the unchanged 2e-6 residual gate. Nominal policy time stays unchanged. Run 03 was an earlier checker-correction interruption, retained separately; it is neither a solver failure nor a pass.

## Evidence and next queue

Canonical new evidence: `artifacts/p65-parity-20261111-01`. Commands preserve fresh source snapshots, before/after hashes, full raw logs and raw exits. G1's 37 source/config snapshots were independently bound before integrating subsequent product changes. The full gzip remains compressed and intact.

Next: finish the one saved audit; run new C2b/B1 static checks, original B1 failure then its complete new suites, C2b plus narrowly affected checkpoint continuation; only B1 acceptance unlocks B2 device-patch force consumption. Then public loaded scaling, a fresh wheel and outside-source public API checks. Final documentation distinguishes all partial domains and failures; no Newton speed claim follows from an unmeasured implementation.
