# MuJoCo-TYX C2b device control input

Current status: **FOCUSED_INTERPRET_AND_CUDA_VERIFIED**, 28 nodes per backend
after the Runtime-cache reuse and cleanup repair; see the parent qualification
sections below. Public CUDA W1/8/32 action-path comparisons passed eight
continuous steps per route with exact full state, action and receipt equality.
Measured steady submission cost in this bounded case fell from about13–15ms to
0.65/0.80/0.98ms, with unchanged full copies and auditing. This is not a Newton
or total-simulation speed claim. New wheel checks remain pending.

The original authoring account and hashes below are historical. No import, collection, numerical probe,
pytest, lint, formatter, compile, build, install or native execution occurred in this lane.
The parent owns focused serial validation after A1/C2 acceptance.

## Delivery and scope

Base: frozen C2 b318af40bdd14df8cbbb84086f274b3053467fd6.
Tests-first: 5e0dc583a6923c43b9008d6d31cd28c9186aaef0.
Product and final pre-execution case additions: f812f88e128f1dee527fe00f89eb6674aff06125.
Authority: parent plan docs/superpowers/plans/2026-11-11-device-action-input.md at c6effef9.

Exactly six paths: two modified runtime/transport seams, two new transport modules, one
new test file and this report. No physics, geometry, constraint, integration, admission,
A1 capsule schema, C2 task formula, assets, policy parameters, dependencies or tolerances changed.
The parent A1 native mj_setState writable-copy fix is independent and must remain on integration.

## Public API

runtime.set_controls_from_device(ctrl, *, qfrc_applied=None, xfrc_applied=None,
expected_revision, producer_stream=None) -> MuJoCoTyxDeviceControlReceipt.

The method submits the complete runtime.world_keys ordered batch. Public shapes are
[W, nu], [W, nv], [W, nbody, 6], all native FP32. CUDA accepts live same-device TYX arrays;
interpret accepts contiguous numpy arrays and takes owned copies. qfrc follows original
native generalized coordinates; body wrench is world force then torque at body COM.
For nu=0 ctrl must be None; internal one-column padding is zero. Omitted forces are newly
zeroed, never inherited. Native actuator control/force clipping remains in the existing kernels.

Raw arrays carry no semantic row labels. A correctly shaped permutation cannot be detected
automatically: the receipt records the caller-declared runtime order, and tests explicitly
check permutation covariance. The internal packet additionally binds actual runtime/device
owner, source/archive/model SHA256, backend/device, native layout, world order and epoch.
Wrong-owner, superseded or invalidated internal packets cannot supply the step.

MuJoCoTyxDeviceControlReceipt is a frozen dataclass with binding, source_revision,
revision, epoch and immutable controls. Its controls mapping has its own dictionary so
runtime.close cannot clear a previously returned receipt. Values own immutable bytes.

## Copy, validation and publication

The runtime lock covers preflight, producer completion, copying, audit, pack and CAS.
All three inputs and producer stream are preflighted before any output writes: dtype,
dimensions, device, liveness, contiguity, overlap among channels and physical/action aliases.
The caller must finish enqueueing producer writes and must not modify/free inputs during
the call. A supplied producer stream is synchronized; None is the explicitly synchronous
default-stream contract, not automatic discovery of hidden asynchronous producers.

A fresh packet owns its own runtime/stream and native copies. CUDA copy_from_device
preserves raw bits. The single required HOST audit reads these copies before arithmetic:
uint32 exponent and mantissa reject NaN/Inf and nonzero positive/negative subnormal values.
Signed zero is accepted unchanged. No clipping, normalization or flush-to-zero repair occurs.
The new TYX pack kernel only copies qfrc into the existing padded width, zeros that padding,
and checks finite fields. ctrl and xfrc remain the exact owned native copies.
Status is checked before immutable canonical MuJoCoControl mirrors and receipt are prepared.

All allocations, transfer/readback, numeric validation, canonical mirrors, receipt, next
revision, counter dictionary and retirement list are prepared before CAS. Publication replaces
held controls and the packet together and increments the existing revision exactly once.
A pre-commit failure preserves previous held action, packet, physics, clock/tick, last
successful physics receipt and revision. Staged owners have reserved cleanup ownership.
Original failure paths and source bytes are retained; no failure was observed by executing
this authoring-only lane.

Old packets are retired after commit. Their cleanup cannot escape as a false atomic
submission rejection. Failures remain owned for retry and appear in control_cleanup_errors.
Packet cleanup synchronizes its stream and attempts all resources. Runtime close attempts
all action owners and all physical owners, retains each failed owner and supports retry.

## Step, reset, checkpoint and caller lifetime

The existing step(h) obtains the trusted packet and the original complete physics checks.
Common TyxDevice._put adds D2D copying; all five device advance implementations and every
physics kernel remain unchanged. Submitted actions stay held across accepted physical steps.
Only a new submission compares expected_revision; successful held actions do not expire
because the next physical step advances the revision.

Action submission and physics step are two transactions. If the action commits and physics
then fails, the new held action remains, but accepted physical state/clock/tick stays old.
An explicit retry uses the same packet. There is no implied combined action+step rollback.

HOST set_controls, partial reset, successful restore and close invalidate the transport cache.
Partial reset preserves untouched worlds' controls. Failed restore leaves the old packet
usable. A1 already captures canonical controls; it serializes no device pointer, stream or
pending transfer. After restore the ordinary HOST load reconstructs exact FP32 controls.
The new transport counter keys are initialized on every fresh device owner so A1's existing
counter transfer has matching keys. Policy memory, action history and policy clock remain
caller-owned; this does not provide a generic ENV/policy clone or automatic reset transition.

Caller buffers may be changed or freed after submission returns. Accepted packets are
independent copies and never alias physical banks or caller arrays.

## Counters and HOST boundary

The runtime publishes:
- device_control_submissions and device_control_refusals;
- control_copy_d2d_bytes for input-copy traffic;
- control_audit_downloaded_bytes for the exact channel audit;
- control_status_downloaded_bytes, control_initial_uploaded_bytes and control_pack_kernels.

The common device exposes device_control_d2d_bytes for loading held controls and
control_host_uploaded_bytes for the ordinary HOST load, including cache rebuild after restore.
Existing uploaded_bytes/downloaded_bytes remain their original HOST transfer accounting.
Failed staged work contributes completed copy/audit/status counts, while refusal counts remain
operational diagnostics rather than accepted simulated state.

The three-channel audit is still HOST, once per device submission. Complete physical state,
body dynamics, contacts/EFC, KKT and clock diagnostics still return to HOST and are validated.
CPU compilation and detached observers remain. C2 workspace itself does not dispatch actions;
its frozen first-slice metadata is not an end-to-end training claim. This new explicit runtime
API supplies the opt-in action direction. No CUDA Torch input, GPU PPO or speedup is claimed.

## Fixed tests, not yet collected or run

Eleven authored test functions manually expand to 42 nodes: 21 interpret and 21 CUDA.
This is enumeration from decorators, not pytest collection and not a pass count.

- Real TYX policy consumes C2 device outputs for two sparse worlds and emits all three action
  channels. Device and HOST routes receive the same FP32 controls and compare complete
  accepted public states and physics/control receipts across held steps.
- Fixed hinge/slide, anisotropic offset-COM free root plus child, and mixed contact/limit
  domains. Only the mixed case enables the joint limit, ensuring the direct T1/T2 routes
  and T4 orchestration are all represented. Mixed nu=0, clipping, absent-force clearing.
- Explicit world permutation, cross/default stream, post-return caller mutation/free,
  dead input/stream, wrong device, old revision, wrong owner/epoch and storage alias.
- Raw uint32 NaN, Inf, positive/negative subnormal rejection before arithmetic; negative-zero
  audit/receipt preservation; bad shape/dtype/noncontiguous data.
- Faults after first copy, audit or pack preserve the previous action and physical state;
  subsequent step still works. Accepted action plus failed physics preserves the new action.
- Failed restore, successful exact continuation restore, partial reset, HOST supersession,
  retained cleanup failures/all-owner retry and receipt immutability after close.

Parent commands, only after the unique serial slot is granted:
```text
ruff check --no-cache <exact five changed Python files>
ruff format --check --no-cache <exact five changed Python files>
pytest tests/test_mujoco_tyx_device_controls.py -q -x --tb=short -m "not cuda" -p no:cacheprovider --basetemp <fresh-absolute-interpret>
pytest tests/test_mujoco_tyx_device_controls.py -q -x --tb=short -m cuda -p no:cacheprovider --basetemp <fresh-absolute-cuda>
```
Do not rerun accepted complete older suites. Parent may select narrowly affected A1/C2 consumers.
Any necessary style-only repair receives its own qualified source binding before execution.

## Exact authoring manifest at f812f88e

Working SHA256 and Git blob were freshly bound by file IO/Git only. All filtered working
blobs equal their corresponding HEAD blobs.

| Path | SHA256 | Git blob |
| --- | --- | --- |
| src/phymeo/backends/_mujoco_tyx_runtime.py | 17c7097e2035e2424b79df2d8b82a0316de62ed45e513c3322566331374f4094 | 58a0c120891929565dac5565f95e930f6f18b486 |
| src/phymeo/backends/mujoco_tyx/_device.py | 7ae2589e79bf106799105515fc69282c4c53e3cb05ce084e424f61be28ed1e11 | 0d3554dee983280b456c161456affa66b4354f4f |
| src/phymeo/backends/mujoco_device_controls.py | 0da5f0178e1ec9c56dd7dede3feb9806f2a7c4bacbd339686bac2b8e08a8bf36 | 9aa5aa4fccd96a2685cc0462b3ccdee319784881 |
| src/phymeo/backends/mujoco_tyx/_control_kernels.py | 2323e42a4d86422ba98447d41c85a450e6ac1df1b83fa6e44de112b1521a41a9 | c7ce79b4c82db71d35e7c2bbe4476b485bf55085 |
| tests/test_mujoco_tyx_device_controls.py | 07f9359716180829d0c2e8c6c7f1ed6dd350be2d8f086b425415fbc1013facff | a5cedc2c9336f729bf8548602fea68b109244ef6 |

This report's own SHA/blob belongs in the final handoff manifest after its commit, avoiding
self-referential content. Author-string readback and git diff --check were the only checks;
they do not establish syntax importability, TYX frontend admission or numerical correctness.


## Pre-execution capacity coverage supplement

Parent review requested explicit coverage of the native width ceiling; no product change
was requested or made. The test file now adds one independent inline model with one
anisotropic offset-COM free root, 29 scalar hinge/slide branches and 29 motors:
nq=36, nv=35, nu=29, nbody=31 including world. All dynamic bodies have ordinary positive
mass and strictly positive inertias; root full inertia is strictly diagonally dominant
and satisfies the triangle inequalities. Each scalar DOF has positive armature/damping.
The collision-disabled primitives do not substitute geometry to bypass an actual contact
test: this case specifically exercises the admitted T2 transport width, while the
existing mixed contact/limit case remains unchanged.

Two sparse worlds begin with nonidentity unit quaternion declarations, nonzero root and
scalar velocities and scalar positions. All 29 controls, all 35 generalized-force entries
and all six COM-wrench entries on each dynamic body are nonzero FP32. A separate HOST
route receives exactly the same stored FP32 requests. Three actual consecutive steps are
authored, comparing the complete public state/control/physics receipt after every step,
plus explicit nq/nv/nu and full 35-by-35 mass-matrix publication checks.

This is source-only authoring, not a successful capacity result or a G1 rerun. No asset
download, product edit, test import, collection, numeric probe or execution was performed.
The earlier 42-node count and test-file SHA/blob above remain historical at f812f88e.
The updated manual count is 12 test functions / 44 nodes: 22 interpret and 22 CUDA,
all still UNCOLLECTED and UNEXECUTED in this lane. The supplement's two current hashes
are supplied in the final handoff rather than rewriting historical product bindings.

## Parent qualification, 2026-11-11

The earlier authoring labels and hashes above are historical. Parent source baseline
6ad1290a passed the complete interpret selection: 22 passed / 22 deselected. The first
CUDA selection passed 17 nodes, then the overlap-negative fixture failed before calling
the product: TYX `CudaArray.from_external` requires an integer, whereas `device_ptr`
is a CUDA pointer wrapper. Commit ded3662b changes only that fixture argument to
`int(device_ptr)`, retaining the same live allocation and deliberate overlap. The exact
failed CUDA node and the complete CUDA selection then passed: 22 passed / 22 deselected.
No numerical, product, tolerance or capacity repair was required.

The complete new selection includes the free-root plus 29-joint/29-actuator maximum-width
case and the original mixed contact/limit case. Four narrowly affected A1 exact-restore
continuation cases passed on each backend after this runtime integration. Exact new-file
Ruff and format checks passed; first failures and before/after source bindings remain in
`artifacts/p65-parity-20261111-01`, jobs 35--38 and 42--45. No accepted older complete suite
was rerun. Installed-wheel public validation remains a separate pending check.

This acceptance is the declared device-owned input transaction and its connection to the
existing physical step. It retains HOST canonical action auditing and full physical
diagnostic readback. It does not qualify CUDA Torch, a GPU PPO trainer, or an end-to-end
zero-readback simulation loop.

## 04:10 measured action-pack cache reuse

Parent product integration 90951628, test-only formatting c1422ccd: the packet borrows its live physical owner's TYX Runtime, while retaining five independent buffers, an independent stream, exact raw-bit auditing, status readback and single control CAS. An adjacent resource-retry defect was fixed: successful owned-stream close clears that reference, so later failed-buffer cleanup retries do not synchronize a destroyed stream. Failed stream close remains owned and retryable.

Complete changed C2b qualification: run65 interpret 28 passed/28 deselected in 65.06 s; run66 CUDA 28 passed/28 deselected in 5.78 s. Original 44 nodes remain, with 12 new lifecycle/cache nodes. Tests include actual cache-object reuse, independent live packet allocations, audit/pack failure, stream/buffer cleanup failure, reset/restore owner replacement and continued state equality. Exact-file Ruff passed; initial format check required one test file, and complete raw AST equality plus final format check passed.

The unchanged run52 benchmark was rerun as run67 on the new source. Both real TYX-policy routes passed at W1/8/32, each with eight continuing steps and no warmup/reset. Complete task/action words, public state and receipts matched the independent HOST-action route. Steady-seven device-ingress totals changed from 0.1073208/0.0913893/0.0899252 s to 0.0045356/0.0055732/0.0068556 s: approximately 0.65/0.80/0.98 ms per submission in this new measurement. First submissions remain separately measured at 25.09/11.14/11.15 ms. The original complete copy/audit/status and physical-readback counters are unchanged; all three channels still have exact HOST auditing.

This is a bounded local transport comparison, not a Newton benchmark, broad statistical throughput result, buffer-pooling implementation, zero-readback claim or GPU PPO acceptance. The physical equations and kernels did not change. Canonical evidence: artifacts/p65-parity-20261111-01 jobs60–67, action-cache-parent-review.json and action-cache-format-ast-equivalence.json. No second profile is needed after the measured improvement and actual compiled-object identity test; continue with the independent collision integration.
