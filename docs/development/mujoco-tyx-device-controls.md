# MuJoCo-TYX C2b device control input

Status: CODE_READY_TESTS_PENDING / AUTHORING_SLOT_RELEASED.
All new product and test code is UNEXECUTED. No import, collection, numerical probe,
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
