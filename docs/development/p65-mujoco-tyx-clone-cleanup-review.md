# MuJoCo-TYX clone cleanup: bounded independent review

Status: TEXT_REVIEW_COMPLETE_WITH_ONE_RESOURCE_LAYER_FINDING. No project execution, import, numerical probe, test, lint, build, or installation was performed by this reviewer. This report does not replace the parent's actual validation records.

## Reviewed change and outer ownership

Reviewed parent `12e7c0a1cefe52953f4f0c8468177c6d86c8e8e2`, formatted test at `e16f8767`, and the same current source at parent `33ec87f638c335e78d1db1846759641898add799`. Scope was clone transfer/cleanup and its four authored failure injections, followed by the exact underlying cleanup methods required to interpret that claim. No full dynamics review was repeated.

In `src/phymeo/backends/_mujoco_tyx_runtime.py:873-913`, the source physical bank is never assigned or closed. The clone is a separate owner. Native-copy failure and replacement-build failure enter clone cleanup. After replacement construction, both the old clone bank and its replacement have owners before retiring the old bank. If clone cleanup itself fails, the source retains the failed clone in `_checkpoint_retired`; the exception group preserves the original and cleanup errors. The source can still advance with its original bank.

The outer `close()` method at lines 915-942 keeps owners that throw, removes successfully closed owners, and permits a later close even after the CPU runtime is marked closed. This outer owner transition is consistent. No additional source-bank mutation or premature publication was found in the reviewed clone change.

`tests/test_mujoco_tyx_clone_cleanup.py:21-118` actually creates banks, fails the second native copy / second build / first bank close, compares exact source snapshot and checkpoint, advances source against an independent pre-fault clone, creates a successful later clone, and finally checks the recorded bank owners were closed. These are meaningful checks of the outer transfer. The replacement fault is before allocation inside the second build; it is not a mid-allocation driver fault. Constructor failure before `cloned` exists and asynchronous driver completion are not qualified by these four cases.

## Finding C1: an internal cleanup exception is consumed by ExitStack

`src/phymeo/backends/mujoco_tyx/_device.py:160-161` delegates bank close to `ExitStack.close()`. The same file registers `runtime.close` at line 22 and each actual CUDA array `free` at line 63. The tree constructor uses the same base buffer/close implementation and registers its runtime at `_tree_device.py:27`.

The actual test environment's `C:/Python313/Lib/contextlib.py:596-612` pops a callback before calling it and does not restore a callback that throws. Its close continues through the remaining callbacks, including the runtime. Therefore retaining the bank object alone does not retain a retryable failed callback. In the retirement path, a second outer bank.close can return successfully with an empty stack even though the first internal callback did not establish successful release.

The current tests inject at `test_mujoco_tyx_clone_cleanup.py:56-65`, before entering the real `close_bank()`. Consequently they cover outer close-before-cleanup failure, but not an exception inside an already-registered `free` or `runtime.close` callback. Their successful outcome must not be generalized to recovery from arbitrary internal release failures. This is a resource-ownership finding, not a physics/solver failure.

## Pinned dependency semantics

Read only the actual installed files under parent `.venv-training-cuda313/Lib/site-packages/tyx`. `tyx_compute-0.1.0a0.dist-info/direct_url.json` binds commit `99f6df7c0fd8fdeef77f97df83914da27b64b7ee`.

| Source | SHA256 | Relevant behavior |
|---|---|---|
| `cuda_array.py` | `3ee29f9067242556d4e2e079031393d0ba0fd85b8e7458fc23df101aa48b5cdf` | Lines 290-308 lock free, call native release, then mark `_freed`; lines 325-329 destructor attempts free. |
| `runtime.py` | `cebbaac59c6d3eaf253e94286a86f24e78fbc517b5eac01fcfa4a3845964662c` | Lines 40-70 permit retry after disposal failure; 307-350 retain failed closeables/artifacts and remove only successes. |
| `backends/cuda_runtime.py` | `68ead01d5c99ab4455368438dd515d5f599d2b0bb625f44a352053729589795d` | Lines 1487-1490 call native free then check status; 585-598 unload a module before clearing its handle; 2840-2852 raise on nonzero status. |

A normally completed `CudaArray.free()` is idempotent because its `_freed` flag is then true. An exception from native free occurs before that assignment. There is no public completed-release receipt that permits this layer to distinguish a failure before the native side effect from an error observed after it. Reading a private flag would not repair this ambiguity.

TYX Runtime has a real retry ledger and lifecycle gate. That does not prove every contained native operation is failure-atomic: compiled CUDA artifact close calls synchronize/unload before clearing its handle. The current generic exception does not preserve a public phase distinguishing those operations. Runtime.close does not destroy the CUDA primary context in the reviewed method; the ordering concern is retaining all dependencies until buffer cleanup has a confirmed outcome, not claiming context destruction here.

The fixed [NVIDIA CUDA 12.6.1 memory API](https://docs.nvidia.com/cuda/archive/12.6.1/cuda-driver-api/group__CUDA__MEM.html) and [module API](https://docs.nvidia.com/cuda/archive/12.6.1/cuda-driver-api/group__CUDA__MODULE.html) document that cuMemFree and cuModuleUnload can surface earlier asynchronous errors. They do not provide the release-failure atomicity needed to infer that retrying the same opaque handle is always safe. The conservative classification below is an inference from this missing guarantee, not an assertion that a particular driver error already double-freed memory.

## Proposed minimal follow-up, not yet authorized implementation

Use a small device-owned cleanup ledger instead of treating ExitStack as a retry ledger. Preserve the existing outer clone ownership fix and all physical kernels, inputs, capacities, counters and numerical gates.

1. Register buffer owners separately from their runtime owner, immediately after successful construction. Retain strong references and release in reverse buffer allocation order. Only remove a callback after normal completion. Never infer completion solely from a thrown exception or parse driver error strings.
2. Distinguish PENDING, DONE, and UNKNOWN_RELEASE. Mark a release as entered before invoking the actual callback; normal return marks DONE/removes it. Any opaque exception inside a real release marks UNKNOWN_RELEASE and retains the owner and original exception. It is not silently reclassified as pending.
3. After an unknown buffer outcome, independent pending sibling buffers may each receive their first release attempt, but do not call runtime.close while any buffer remains unresolved. A later bank.close reports the unresolved state without invoking that buffer's free again. The object must not be reused for physics.
4. Only after all buffers are confirmed DONE, attempt runtime.close. Normal completion removes the runtime owner. An opaque exception from this call is also retained as UNKNOWN_RELEASE; do not blindly rely on Runtime's internal retry capability when the actual failing native phase is unknown. A separate, explicitly qualified pre-release failure may remain PENDING, but the current TYX exception alone does not establish that classification.
5. Keep the outer retry path for exceptions known to occur before entering this ledger: for example, the existing authored bank.close wrapper that throws before calling the real method. A subsequent call then performs each PENDING action once. An outer error after the ledger has completed sees DONE state and cannot repeat release.
6. Avoid accidental destructor retries when the application drops a failed runtime: an unresolved resource needs a strong-reference quarantine owned beyond the failed runtime's ordinary lifetime, plus inspectable unresolved identities/errors. Do not call `CudaArray.__del__`, manually alter its private flags, unload/recreate the primary context, or pretend ownership was successfully cleared. Process termination is outside an in-process recovery claim; this proposal does not claim successful in-process reclamation of unknown handles.

This is a fail-safe owner-retention design, not a promise to recover every CUDA failure. A universal automatic free retry cannot be justified with the pinned public API. If persistent quarantine is not included in this batch, explicitly limit its guarantee to the failed owner's retained lifetime and leave arbitrary internal free recovery unsupported.

## Minimal predeclared checks for a separately authorized repair

| Case | Required result |
|---|---|
| All releases return normally | Reverse buffer order then runtime; repeated bank.close makes zero additional release calls. Existing successful clone/state results remain unchanged. |
| Existing outer fault before real bank.close | Source stays usable; later close enters the still-pending ledger once; no product resource was called by the failed wrapper. |
| Internal callback throws before its fake release side effect | Retain UNKNOWN_RELEASE and original error; no runtime close; a second cleanup does not call that opaque callback again. Conservatively unknown despite the test author knowing its phase. |
| Internal callback releases fake resource then throws | Same unknown retention; its release-call count remains one across repeated close, preventing a double release. |
| One failing buffer among successful siblings | Successful entries are removed and never retried, failing owner is retained, runtime remains held. No lost entries after error aggregation. |
| Runtime cleanup throws after buffers finish | Buffer release counts do not change on subsequent close; runtime owner/error remain explicit and not claimed closed successfully. |
| Failed clone cleanup reaches source.close | Source bank identity/state remains unmodified before explicit source.close; clone's unknown owner remains observable rather than vanishing into an empty ExitStack. |
| Application drops its failed runtime | If the implemented contract includes quarantine, unresolved owners stay strongly held and no implicit destructor invokes free a second time during ordinary object collection. |

Use harmless fake release callbacks/owners for uncertain-native-phase cases rather than deliberately corrupting a real CUDA context. A real successful bank lifecycle can qualify the positive adapter integration. No new physical matrix, altered contact model, dependency update, GPU reset, or repeated green trajectory is required by this review.
## Authorized authoring follow-up

The parent accepted the bounded sticky-unknown design after the review above. Tests-first checkpoint `6a475630fad7b1dbaf96721bf2b6899c635e7869` adds only `tests/test_mujoco_tyx_resource_cleanup.py`. The implementation adds only `src/phymeo/backends/mujoco_tyx/_resources.py`; this report is the authoring record. Status remains CODE_READY_TESTS_PENDING; no tests were imported, collected or run here.

The new `DeviceResources` uses `set_runtime(owner)`, `add_buffer(owner)`, and `close()`. Read-only `state`, `closed`, `failures`, `pending_buffer_count`, and `runtime_pending` describe the ledger; `quarantined_resources()` exposes a tuple of the strong retained ledgers without a clear/retry operation. `ReleaseFailure` preserves the original owner, kind and BaseException. RLock serializes registration/close, and recursive close is rejected before a callback can repeat. Success removes each owner; unknown failure retains it in the quarantine, drains only independent pending siblings, and prevents runtime close. Subsequent calls raise the original errors without re-invoking releases. No TYX private lifetime flag is read or written.

The 24 authored cases consist of nine fake-owner cases, six actual early-constructor failure cases (base/tree/floating, interpret/CUDA), and nine actual post-release late-failure cases (the three families crossed with interpret runtime, CUDA runtime, CUDA buffer). This count is an authored enumeration, not an observed collection or pass count. Actual cases use an independent inline MJCF and real public runtime. A CUDA buffer fault is deliberately raised only after normal free return; after checking that the product correctly retains its runtime, the test alone closes that runtime using its known fault provenance to avoid creating a live-device leak. The product ledger remains UNKNOWN_RELEASE and quarantined. This deliberate test cleanup does not grant a production recovery operation or erase the failure.

The parent-owned wiring is a separate ignored patch at `artifacts/p65-resource-cleanup-authoring-20261112/device-resource-wiring.patch`, SHA256 `16f7411d87263c37ccbfff75a87013e2782475812df918947b55533bbd9ad330`. It replaces ExitStack initialization/runtime registration in only `_device.py`, `_tree_device.py`, `_floating_device.py`, and the base `_buffer` registration with `add_buffer`. The old `TyxDevice.close()` remains the shared call site. Normal/constraint devices inherit these paths. No old source was modified in this author worktree. `git apply --check` verified only patch applicability; this is not Python validation. The parent's outer clone source and existing four fault tests remain unchanged.

Pending parent commands are scoped to the two new Python files plus the three applied wiring files for static checks, followed by `tests/test_mujoco_tyx_resource_cleanup.py` and the existing `tests/test_mujoco_tyx_clone_cleanup.py` under the sole serial execution owner. The first qualifies the new ledger; the existing clone group is justified by its changed cleanup dependency. No trajectory or solver revalidation is asserted by these checks. Construction failure before a device owner exists, allocator-internal failure before returning a CudaArray, process-exit destructor order, and successful recovery of unknown native release remain outside this slice.