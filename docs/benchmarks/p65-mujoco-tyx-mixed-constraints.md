# MuJoCo-TYX mixed constraints and G1 validation

Status: **INTEGRATED / VALIDATING, not accepted**. This is the active continuation record for the 2026-11-10 T4/T5 batch. The full prior accepted repository was pushed to [codex/dev](https://github.com/ChenxrC/phymeo/tree/codex/dev) at `73a85f8affccde5ebf21a4629845d06af6949479`. New unaccepted work stays on `codex/p65-parallel-modules`.

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
- Long interpret matrix36 is running. Its D01 mixed branch exposed an initial dry-free-face ordering defect; correction is prepared only after the running source snapshot is released. No result is declared for the whole matrix yet.

These are per-run results with source bindings, not a single final-source union or a complete T4 acceptance.

## G1 preparation and remaining acceptance

The [new harness](../../examples/g1_mujoco_tyx_policy.py) and [reviewed handoff](../development/mujoco-tyx-t4-g1-harness.md) use the existing primitive G1, original nonzero frame-0 q/v, original ONNX/gains/control frequency and native iterations100. All 502 pairs and 942 geometry slots remain; both seven-capsule feet, all robot self-pairs, original excludes/limits and condim6 are preserved.

The policy observes each stream's current state in the correct inertial-local gyro frame; this corrects the old reference's cached cvel lag and is explicitly not reuse of its old trajectory results. Same-input native comparisons and the independently advancing CPU feedback trajectory are distinct from the device state. Input-step force and output-state time remain separate. Key contact/constraint events save full same-run fields without another physical step. Optional .gz output streams losslessly preserve every JSON field and avoid a redundant plain copy.

Pending: finish new A–E cases on interpret/CUDA; original G1 admission/F1 snapshots and sustained support; fixed push and h variants; 2-to-20s continuous policy window without reset; appropriate adjacent shared-seam checks, a new outside-source wheel verification and final evidence/architecture update. Only accepted increments will be pushed to codex/dev. Full 131.48s dance, mesh deployment, elliptic cones, equality/tendons and overall Newton equivalence are not claimed.