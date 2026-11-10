# MuJoCo-TYX mixed constraints and G1 validation

Status: **LOCAL_VALIDATED — T4 mixed constraints and T5 finite G1 support**, completed2026-11-11. The full prior accepted repository was pushed to [codex/dev](https://github.com/ChenxrC/phymeo/tree/codex/dev) at `73a85f8affccde5ebf21a4629845d06af6949479`; this accepted increment is synchronized to the same development branch. The scope and remaining gaps are stated below.

## Implemented scope

The [frozen plan](../superpowers/plans/2026-11-10-mujoco-tyx-mixed-constraints.md) connects native pyramidal condim1/3/4/6 contact edges, scalar soft joint limits and DOF dry friction to one physical mass matrix and one bounded box solve. All forces and motion publish atomically. The device visits all admitted pairs, compacts actual contacts/EFC rows, retains inactive-contact six-dimensional Jacobians, and explicitly rejects overflow. Limits remain G128/P1024/static slots2048, actual contacts128/EFC512; defaults64/256 remain.

Cold active-set solving uses original source iterations as the face budget, no CPU/Warp fallback or parameter repair. Small faces use QR of the physical-mass-whitened Jacobian augmented with sqrt(R); larger faces use the equivalent primal correction. The same checked physical M determines qacc, and the existing M+hD Euler solve integrates the state. This is an independent bounded algorithm, not MuJoCo's native Newton iteration. Existing T0–T3 routes remain separate.

## Actual results so far

All project execution belongs to the parent, in one serial queue. Raw evidence is `artifacts/p65-mujoco-tyx-t4-20261110-01`; all first failures remain.

- Source/ENV initial style, frontend and first numerical failures were preserved (02–12). Incremental face recovery closed the first FP32 cancellation failure; interpret ENV13 and CUDA ENV14 each passed.
- 23 short interpret stopped on native MINMU after six passes. Saved-only analysis24 showed the tiny-R force recovery cancellation; square-root small-face solve passed the exact MINMU control25.
- 26 short interpret stopped on a faulty last-step friction assertion after six passes. Same-run native and device had both separated by that endpoint.
- 27 remaining short interpret: **71 passed / 16 failed**. Failures included incorrect endpoint/couple assumptions and exception attribute access, a budget fixture that actually needed only one face, two prematurely stopped oblique responses and four solver/protocol failures.
- Saved-only analysis28 used padded width incorrectly and failed; corrected29 read the actual nonzero physical-M block. Product fixes retained original parameters and global thresholds, additionally checked bound KKT and allowed release after a completed blocked face. Targeted30: **8 passed**.
- First G1 harness lint31 passed; format32 reported two files; format33 and final lint34/format35 passed. This is not G1 physics evidence.
- Long interpret36: 4 D01 failures / 8 D02 passes. After completing the initial dry free face before releasing bound rows, targeted short41 passed all 24 selected nodes and original-initial-state D01 run42 passed all four complete one-second windows. Unaffected no-dry D02 evidence is reused from36. These are historical pre-batch results.

Those early results retain their original source bindings. The final-source matrix and acceptance are recorded below; no historical pass is relabeled as a run on the final source.


## Final-source validation and retained failures

The intermediate source `657e69a9` refined the current free face at a strict local scale; `2b4323e3` added generic capability domains and deprecated the model-named `g1_supported` alias. Short run69 on that source: **186 passed / 24 deselected, 161.08s** across interpret and CUDA. The subsequent original long run81 exposed a second work-selection issue, so69 is historical evidence, not the final-source matrix.

Earlier CUDA43/44 passed86 short nodes and12 full one-second windows; shared legacy routes52 passed10. G1 F1 source-default/motion0/motion200 passed on interpret63/CUDA47. Batch-release revision `ee1effdd` then passed h=.001/2s (48) and both signed .1mg push/2s windows (49/50), with no reset or rejected step.

The original20s run51 **failed at17.284s after8642 accepted steps**: source-budget100 was not exhausted, but a bound was prematurely released before its existing free face was sufficiently refined. No robot behavior gate failed before the rejection: minimum height .668329656m, maximum tilt .367177304rad, settled mean foot force .999319834mg, peak2.339163462mg and no non-foot ground load. Full failed state and native comparison remain in51. Saved-only analysis72 established the specific free-face/bound mismatch; it did not generate a replacement physical trajectory.

Runs79/80 replayed the exact saved device-bank input once on interpret/CUDA after the fix. Both passed with5 face solves,70 rows,27 free rows, independent KKT4.56e-8/2.74e-8, and the original native force/acceleration comparisons. These are low-level failure regressions, **not public checkpoint or continuous support evidence**. Attempts75/77 rejected before a device step because the public FP64-unit-quaternion initializer cannot exactly represent this saved FP32 quaternion; the mismatch remains an explicit interoperability follow-up, not a silently repaired state.

Run81 rejected at9.296s after4648 accepted steps, exhausting100 faces while its original behavior gates still held. Saved-only analysis85 found a real bound violation0.861673 alongside a free-face FP32 residual near3.60e-5: the strict absolute free gate could repeatedly refine the face without admitting the strongly violated bound.

Current solver `3da108c1` admits a bound only when its locally scaled violation exceeds both the original tolerance and the current maximum locally scaled free residual. This generic proportional work-selection rule closes both saved failures without changing final KKT, force-feasibility, mass-equation or budget gates. Runs86/87 replayed51's exact input on interpret/CUDA in5 faces;88/89 replayed81's input in26 faces. Risk controls92 passed26 nodes. These are failure regressions, not continuous trajectory acceptance.

Harness `5538c18f` also copies accumulated slip/work when saving a checkpoint. The older failed51/81 final summaries held shallow references for those two fields; their stream checkpoints retain the correct point-in-time values. The old defect is explicitly reported rather than normalized away. This correction does not change physics or behavior gates.

Run93 completed the fresh public20s trajectory from the original source initial state: **10,000 accepted steps, RAW0**, original assets/controller and no reset. Device minimum height .668324232m, maximum tilt .374636827rad, maximum joint RMSE .199919557rad, reference XY error .228200631m, settled mean foot load1.000459887mg, peak2.491446885mg, and no non-foot ground force. The independently advancing CPU lane also passed all original behavior gates. This is finite20s support/policy validation, not the entire131.48s motion or every-step native force equivalence.

Runs82–84 completed the predeclared final-source h/push checks because the solver changed after48–50. All four current-source windows passed, with zero rejected batches:

| Run | h / duration | Accepted device steps | Settled mean foot load / mg | Peak foot load / mg |
|---|---|---:|---:|---:|
| 93 original policy | .002 / 20s | 10000 | 1.000459887 | 2.491446885 |
| 82 half step | .001 / 2s | 2000 | .999806521 | 1.561152684 |
| 83 negative push | .002 / 2s | 1000 | .999855632 | 1.523117784 |
| 84 positive push | .002 / 2s | 1000 | .999779132 | 1.523117784 |

The independently advancing CPU lane passed the same original behavior gates in each window. Recorded work, slip and input energy-balance diagnostics remain separate from those gates; they are not an exact energy-conservation certificate. Final short run96 passed186 nodes/24 deselected in166.72s; long interpret70 passed12/198 deselected in1707.76s and CUDA71 passed12/198 deselected in179.58s. Together they cover all210 final-source nodes. Runs98/99 additionally passed the three original same-input G1 poses on each backend after the final solver change.

## G1 scope, package and evidence acceptance

The [new harness](../../examples/g1_mujoco_tyx_policy.py) and [reviewed handoff](../development/mujoco-tyx-t4-g1-harness.md) use the existing primitive G1, original nonzero frame-0 q/v, original ONNX/gains/control frequency and native iterations100. All 502 pairs and 942 geometry slots remain; both seven-capsule feet, all robot self-pairs, original excludes/limits and condim6 are preserved.

The policy observes each stream's current state in the correct inertial-local gyro frame; this corrects the old reference's cached cvel lag and is explicitly not reuse of its old trajectory results. Same-input native comparisons and the independently advancing CPU feedback trajectory are distinct from the device state. Input-step force and output-state time remain separate. Key contact/constraint events save full same-run fields without another physical step. Optional .gz output streams losslessly preserve every JSON field and avoid a redundant plain copy.

The final source is `5538c18f77c819a9f3212a5db5cd02ac7af4ddb1`. Build53 and fresh install54 succeeded; outside-source consumers55/56 passed public mixed-constraint ENV/PolicyRunner, selected reset and clone-next-step on interpret/CUDA. The wheel SHA256 is `729042d31bd150144431ee6a3bfb5cd963755847b81e9a257916efa62eb97051`. File-only parity100 verified exact complete membership and bytes for476 Git-tracked product files against both wheel and installed site, with zero issues; the wheel contains480 total entries. No package-source WIP was present.

Saved-evidence audit97 passed with16 explicitly selected successful command receipts,210 unique passed nodes, all four successful G1 trajectories and both original failures51/81. It checked14,000 accepted device-step records and2,130 full event captures in the successful windows, complete gzip streams/input-output/control bindings and original checkpoint coverage. Both historical shallow-reference defects remain explicitly classified in the failed runs; successful checkpoints require full exact equality. The audit performs no physical replay. Report SHA256: `a64fb161ed839c2f4bee2b7aae1797fade768461c5d316b709d1738c95f8f684`.

All project execution has ended. The accepted source, tests, example and documentation are the development-branch increment; local caches, raw evidence and the three unrelated unaccepted K1 drafts are not part of that commit. Full131.48s dance, mesh deployment, elliptic cones, equality/tendons, public exact restoration of an arbitrary saved FP32 quaternion, and GPU tensor training/throughput remain separate work. This is not overall Newton equivalence or whole-P6.5 completion.
