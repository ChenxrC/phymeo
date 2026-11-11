# ROB2a filterexact authoring report

Status: **CODE_READY_TESTS_PENDING / AUTHORING_RELEASED**. No import, collection, numerical probe, pytest, formatter/lint, compilation, installation, native run or build was executed by this author. Git diff checks and text/file-IO checks are not physical validation.

## Delivered vertical slice

The [frozen design](p65-rob2a-actuator-design.md) is implemented in the author-owned files. Source route, runtime and checkpoint shared seams are delivered as three unapplied parent-review patches, not silently edited in the child. Until those patches are integrated and tested, this branch alone does not run the new domain.

- New immutable native activation layout: mixed NONE/FILTEREXACT source order, unique native activation-address coverage, scalar JOINT identity, gear, gain/bias, disabled group and separate clamp masks. Capacity 32 actors / 32 activations is rejected **before** ids/value buffer allocation. Existing fixed T1 body/joint/options/contact checks remain authoritative.
- New device stage computes raw derivative, effective integration derivative, exact-law filtered candidate, actrange, entry/early force selection, scalar force clamp and full generalized force. Disabled raw derivative and zero integration derivative remain distinct. Small positive h/tau evaluates `1-exp(-x)` with a degree-five series below 0.01 to avoid cancellation; the first omitted term is bounded by x^6/720 before FP32 arithmetic. Larger x uses the supported TYX exp intrinsic. No HOST state update is used.
- `_tree_device.py` allocates two actual activation banks tied to the original active index. The new route uses unchanged geometry, unchanged mechanics with zero local actor count, one new activation kernel adding to original external loads, and unchanged solve. It therefore has four launches per real step; old NONE paths keep three.
- Public `MuJoCoTyxActuation` stores input/output act, raw/integration derivatives, clamped control, selected force input, raw/final scalar force, full generalized force and static source mapping. `MuJoCoTyxActivationStepReceipt` is a separate subclass; old receipt fields/type are unchanged. Arrays are bytes-backed immutable copies of the actual download, not a recomputed output oracle.
- Parent patches add early native-na admission routing, raw act bank pack/quantization/input binding, candidate act and receipt in the existing one-swap publication, stateful checkpoint v2 and exact restore. The old v1 state fields/layout are retained for na=0. Original MIRROR_FIELDS retains qfrc_actuator, and the new restore binds it to the saved actuation generalized-force channel. Existing CPU initialization/explicit full snapshot observation is not misrepresented as device physics.

The full M, bias, passive forces, body Jacobians, force projection, Euler solve and q/v/time integrator were not copied or edited. `_tree_kernels.py` has no author changes. No shared runtime, checkpoint, training environment, collision or G1 file was edited.

## Predeclared tests and numerical gates

`tests/test_mujoco_tyx_activation.py` contains **52 manually enumerated parameter nodes** (26 per backend), not a collected or passing count:

| Test family | Per backend | Evidence / purpose |
|---|---:|---|
| Native same-input and independent continuous trajectory | 8 | 32 steps each; hinge early/ordinary, h/tau > 1, small h/tau, negative-gear slide, disabled positive force range, mixed coupled R/P |
| Sparse-world continuation | 3 | W1/W2/W8, nonzero act checkpoint bytes, real successor, restore successor, clone and selected reset |
| Late failure / retry | 3 | Actual completed device work followed by download/input-act rejection, or replacement restore readback fault; accepted bytes retained |
| Unsupported preallocation | 6 | filter, integrator, free, ball, joint limit, actor/activation capacity |
| Stateless compatibility | 1 | Same old receipt, checkpoint v1, three launches |
| Device-only step | 1 | CPU mj_step/mj_forward forbidden after construction; integration_snapshot reads real accepted act and motion |
| Input / source identity | 4 | Wrong act shape, nonfinite act, corrupted authoritative bank order, different source before replacement allocation |

New activation/derivative gates: rtol **3e-5**, atol **3e-6**. Scalar and generalized force: rtol **3e-4**, atol **2e-4**. Original T1 gates reused: M (2e-4,3e-5), bias/passive/applied (3e-4,2e-4), smooth acceleration (5e-4,8e-4); continuous q absolute 5e-5, v 3e-4, time 2e-6. Parent must preserve these first-run gates and retain failures. Same-backend accepted state, clone/restore successor and source/input bindings use exact array or complete structured-byte comparisons.

The oracle compiles original XML independently with MuJoCo 3.3.7. Same-input native receives actual accepted entry q/v/act and FP32-stored control; continuous native is initialized once, then advances independently. A separate source-field exponential/force calculation does not call lowering/kernel helpers. The complete 32-step native/device/analytic records are saved in gzip, including the last collected step on an assertion failure. Restore/fault cases save their complete before/after/retry records when they complete. Negative admission cases are pytest assertions, not claimed saved physical trajectories.

Hand checks, not execution: with a=0.1, u=0.4 and h/tau=ln(2), unclipped next activation is 0.25. With unit gain and gear=-2, ordinary force gives generalized effort -0.2; actearly gives -0.5. Disabled raw act_dot can be nonzero while integration uses zero; enabled actrange still clamps stored activation, and a positive source forcerange clamps zero raw force to its lower endpoint. These checks distinguish timing, transmission sign and the native disabled-force corner.

## Shared patches and reproducibility

Parent-review directory: `artifacts/p65-rob2a-authoring` in the capacity worktree. The copies and patches use baseline `847b8ee359320d260e3d36c13cd0530cde7a28b9` and touch only the three approved shared files. Parent's newer integration_snapshot method is outside every hunk and must remain intact.

| Patch | SHA256 |
|---|---|
| `01-route.patch` | e782c94c0659385db19e0e0109db4c2b6cf630b914a7eabb4e9547f0f843b1f6 |
| `02-runtime.patch` | 71c4139ae6343e74ee9baaa2e38fac80ca73a1c6c95490f52102c222223ff0ae |
| `03-checkpoint.patch` | 702474fe2f266ca28267a5b8950f21d8908d8fa07c0138bd08325734e1b50d48 |

Commit order: d668af1b design; a8af6d64 tests-first; 0d0e282c five product files; 1cf50a3a tests identity/device-only refinement; 2762402c native-valid ball rejection; this report. Exact eight tracked delivery files and SHA/blob bindings are recorded in the ignored `handoff.json` after final commit. The older untracked `docs/development/mujoco-tyx-t1-review.md` is untouched and is not part of delivery.

Pending commands, for the parent's sole serial slot only: the exact nine changed/new Python files after applying the three shared patches need first lint/format records (five author product Python + one test + three parent seams; `_tree_kernels.py` excluded). Then run the full new file separately as `python -B -m pytest tests/test_mujoco_tyx_activation.py -q -x --tb=short -p no:cacheprovider -m 'not cuda'` and the same command with `-m cuda`, fresh absolute basetemps and the existing pinned environment. Parent may select the already identified narrow old T1/A1 compatibility nodes; do not rerun unrelated accepted sets. This report claims neither static nor functional pass.

## Remaining boundary

This is fixed-root scalar T1 only. No free/ball activation, contact-coupled activation, joint-wide force cap, tendon/site transmission, other actuator dynamics, resident four-family activation or implicit cross-tree solve is accepted. Native limits/constraints are refused, not removed from source. Invalid or nonfinite intermediate state rejects the complete attempted publication. The input law and physical tolerances remain unchanged; further qualification requires actual parent execution and recorded resource/state evidence.

## Parent source qualification (2026-11-12)

The parent integrated the authoring, all three shared patches and its existing explicit integration observation. Tested product is parent `9369d061`; later `404c62f0` only records the rejected collision experiment. The final ball-joint negative was retained when resolving a formatting overlap in the test file. The original stateless receipt and v1 checkpoint remain separate.

- `45-rob-interpret`: 26 passed, 26 other-backend nodes deselected, 8.70 s, exit 0.
- `46-rob-cuda`: 26 passed, 26 other-backend nodes deselected, 7.96 s, exit 0.
- `47-rob-stateless-compat`: 14 passed, 17.95 s, exit 0. Covers original analytic eccentric tree dynamics, disabled scalar-force clamping, all four original checkpoint domains (prismatic/fixed/floating/mixed), and late restore failure in both modes.
- First lint/format findings were import ordering and formatting; exact nine-file repair and final checks passed in commands 38–43. The first functional launch 44 used the old collision environment, which does not contain MuJoCo, and failed at import before physical construction. Command 45 uses the existing training CUDA313 environment with MuJoCo 3.3.7 and unchanged TYX pin. No installation or physics change was used to fix that runner selection.

Evidence is `artifacts/p65-parity-n4b-rob2a-col-stop-20261111-01`. Original failures and all before/after source bindings remain. Both modes exercised the actual four device stages, full force/motion/activation assertions, complete-window native and analytic comparisons, immutable observations, exact restore successors and atomic failure retry. The device-only test forbids CPU `mj_step` and `mj_forward` during physical advancement and integration observation. Separate source-only shared-seam review found no blocking issue.

Status: **SOURCE_QUALIFIED_FIXED_T1_FILTEREXACT; FRESH_INSTALLED_CONSUMER_PENDING**. This does not enable free-root, contacts, constraints or ball-joint activation and does not qualify complete G1 or arbitrary robot learning.
