# Paired observation memory diagnosis

Status: CODE_READY_TESTS_PENDING. Authoring and text review only; no project
execution, memory probe, import, collection, formatter or tests were run by
this author.

## Trigger and limits

The original parent run11 completed J W8/128/256 and S W8/128, then completed
S W256 full mode before integration mode failed at
`_mujoco_tyx_runtime.py`'s actual `copy.copy(original)`. The reported native
error was failure to allocate memory. Those original raw results remain
unchanged. This revision does not turn them into successful runs.

Source inspection established that native batch publication retains the old
W accepted MjData values while constructing the new W candidates. This creates
a possible near-two-batch native allocation peak. It did not establish a
native leak: public receipt/state arrays copy owned bytes, and runtime.close
clears its data/model ownership. The old benchmark also retained local
session/learner/rollout references while constructing the next mode. These
are distinct facts; neither identifies the measured OOM cause by itself.

The parent owns clone/failure cleanup work. This patch does not change shared
runtime, solver, native arena sizes, contact capacities, task model, controls,
seeds, PPO configuration or trajectory budget.

## Isolated ownership and comparison

`_run_mode` contains each factory/runtime, task wrapper, adapter, PPO session,
model state views, observations, update records and RNG tensors. Only a
JSON-safe summary and comparison hashes return to `run_pair`. The next mode
cannot acquire any of those owners through the returned value. Existing public
close/ExitStack ownership is retained; no caller close is removed. This removes
the benchmark's explicit cross-mode strong references. It does not promise
that an external allocator immediately returns all cached memory to Windows.

The unchanged schedule is one warmup and two measured updates, each 16 steps;
W128 runs integration then full, W8/W256 full then integration. Each mode has a
fresh environment and the original seed. Existing nine CUDA matrix nodes are
preserved, but this handoff requests no repeat of the five already passed cells.

Every common transition still compares complete last_step receipts, q/v,
activation, held and native three-channel controls, action, observation,
reward/done/final-observation, episode bookkeeping and accepted-world status.
Complete last_step includes the same force, acceleration and constraint data as
before. The comparison now also includes every frame's stable binding and the
environment binding. Only workspace_owner/runtime_owner opaque identity and
the deliberate environment state_observation output-contract name are removed.
Task/source/archive/model hashes, native/output layout, world order, device,
precision, action/task hashes, h and horizons remain. Unknown additional
binding fields are compared, not silently ignored. Required frame fields
must exist.

RNG comparison means Torch CPU and the current CUDA device only; it is not
a claim about Python/NumPy or arbitrary caller Generators. Full-only endpoint
geometry/contact/sensor/energy outputs remain separately saved and are not
required in integration mode.

## Optional bounded S256 observer

Only `--task S --worlds 256 --memory-diagnostic` enables this diagnostic.
Other task/size combinations reject before creating an output directory.
A fresh output directory is mandatory. No dummy MjData is allocated and no
additional CPU forward/step or physical command is issued by the observer.

The local context replaces only the `copy` name in
`phymeo.backends._mujoco_tyx_runtime` with a facade; the shared stdlib copy
module and source files remain untouched. It delegates the original copy,
returns its actual result or rethrows its actual exception, and restores the
original module binding in finally. Only public scalar properties are read:
MjData narena/nbuffer/maxuse_arena and its public model's narena/nbuffer. No
native object, bank or view is retained by the observer.

Installed MuJoCo public API evidence was read from
`mujoco/introspect/structs.py` (model and data size fields) and the shipped
`bindings_test.py::test_copy_mjdata_with_plugin` (data.model identity).
The public compilation model's narena/nbuffer are also recorded after
construction, separately labelled from the actual runtime data.model values.
This avoids inventing an API or measuring a newly created substitute object.

Samples cover construct entry/completion, the first physical-step entry/end,
each actual step's first-copy-before and last-copy-complete, a copy exception
with exact global/step ordinal, mode close attempts and the mode function's
return. The adapter-step observer is also restored in finally and invokes the
original public step exactly once. A failed first or later step remains a
failure and keeps the existing complete update-prefix evidence.

For two modes of 48 steps and W256 the actual expected step copies are 24576;
the conservative hard bound is 25152 including 2W+64 extra construction slots.
There are at most 512 process events. Per-copy native scalar minima/maxima,
actual copy counts and exception counts are recorded. Exceeding a diagnostic
bound fails explicitly; no physics budget is enlarged or data silently dropped.
Full-mode detached snapshot copies occur in a different module and are not
intercepted; their CPU work remains inside the original physical step envelope.

Windows GetProcessMemoryInfo reports PrivateUsage as process private commit
and WorkingSetSize as RSS. These are whole-process values, not exclusive
MuJoCo allocations or Torch/TYX peaks. If the OS sampling API fails, the sample
is explicitly UNKNOWN with its reason. Native maxuse_arena is the history of
the actual CPU data object; it is not GPU scratch utilization, and may not
include separate detached full-observation copies.

Immutable numbered memory-observation-NNN.json snapshots are emitted at mode
close and mode return, retaining previous snapshot names; no prior observation
file is overwritten. The full/integration protocol and summaries mark every
memory-observed timing `performance_eligible=False`. Such a run can diagnose
allocation stages and still enforce physical/learner equality, but supplies
no formal speed comparison and does not by itself prove or disprove a leak.

## Predeclared tests and next parent action

Original nine CUDA nodes are unchanged. Twelve additional HOST nodes cover:
five stable-binding changes plus missing-field rejection; common-transition
binding wiring; refusing an owner in the summary; weak-reference evidence that
a completed mode's learner is absent before the next factory; delegated copy
success/failure with facade restoration and no retained native owner; and two
invalid diagnostic task/size requests. These are authored tests, not executed
results. The real CUDA path is still required for any actual S256 conclusion.

Parent-only focused sequence, without rerunning the green cells:

    pytest tests/training/test_mujoco_tyx_observation_training.py -m "not cuda" -q -x --tb=short -p no:cacheprovider
    python examples/compare_mujoco_tyx_observation_training.py --task S --worlds 256 --memory-diagnostic --output NEW_DIAGNOSTIC_DIRECTORY

The old five passed matrix cells and old S256 failure retain their original
source bindings; they are not retroactively assigned to this source. Whether
a later uninstrumented S256 performance measurement is justified depends on
the bounded diagnostic and the parent's separate cleanup repair.

Tests-first commit: 7f60e27f. Main implementation: 463754fe. A first authored
test trailing blank line was removed in the implementation commit; final Git
whitespace validation is required at handoff. All project tests and analysis
remain UNEXECUTED by this author. AUTHORING_SLOT_RELEASED after final handoff.
