# Compact native allocation diagnostic v2

Status: CODE_READY_TESTS_PENDING. Authoring only, based on parent 20850b87.
No import, numerical probe, test, formatter, build or benchmark was run here.
The existing paired S256 protocol remains 48 actual steps in each of full and
integration observation modes, with the original policy, physical inputs,
transition/learner/optimizer/RNG equality gates. `performance_eligible` is false.

## Observation boundary

The temporary public `mujoco.MjData` factory delegates its unchanged positional
and keyword arguments to the captured real type and returns that real native
object. The captured type supplies `isinstance`; the proxy is not a subclass or
native pointer bridge. Both factory and the pre-existing runtime-only copy
facade are exactly restored in `finally`, including exceptions. The diagnostic
must run alone in one process: replacing the public module attribute is not a
thread-safe global service. Text searches of current Phymeo and installed MuJoCo
Python paths found no call-path `isinstance(..., mujoco.MjData)` or equivalent
that would consume the temporarily replaced attribute as a type.

Constructor attempts/failures and elapsed native call time are separate from
runtime-module `copy.copy` attempts/failures. This does not count arbitrary C
allocations or copies invoked through other module aliases. Original v1 JSON
files and copy fields keep their historical meaning; absent v1 constructor
fields mean UNKNOWN, never zero. New snapshots declare v2 and retain all old
cumulative copy/process/event keys. There was no old parser to rewrite.

The phase ledger separates construction, exact `runtime.step`, public full
snapshot, public integration snapshot, and other work. The original adapter
wrapper still supplies the 1..48 rollout ordinal. It is not the physical-step
interval: env action export and endpoint observation occur outside the wrapped
runtime method. Every successful real runtime step must have zero intercepted
constructors and runtime native copies. Both modes must record all 48 intervals.
Nested phase wall durations include explicit inclusive/exclusive values; native
constructor/copy wall times are subtimings, not additional totals. Diagnostic
process sampling and field inspection are overhead, so no speed claim follows.

Actual public data/model `narena`, `nbuffer`, `maxuse_arena`, stack/thread-stack,
contact and EFC high-water fields are saved when available. Required old fields
remain required; missing optional fields are explicit. Ranges describe only the
observed constructor/copy and post-snapshot scratch samples, not a census of all
native allocations or every intermediate world. Whole-process commit/RSS retain
their original scope. Neither value establishes a leak or exclusive MuJoCo cost.

Two read-only ownership samples per mode count accepted record types and compact
worlds, accepted native records and the runtime's one observer-scratch reference
(0 or 1). They do not inspect native pointers, cache IDs as live-object evidence,
retain native owners, or claim process-wide live-native counts. All saved
summaries contain only ordinary scalar/list/dict values. The final sample is
before the original close sequence, which remains unchanged.

## Parent-only pending verification

Existing HOST tests remain. Eight new fake-owner cases are authored:

- public constructor success/failure (2), unchanged arguments/real type and no owner retention;
- exact step vs full-observer interval (1), original zero-copy step and one reused scratch;
- unexpected step constructor/copy plus original step exception (3), no masked original exception;
- escaping constructor failure (1), exact factory/copy restoration;
- read-only layout scalar counts (1), no retained compact/scratch objects.

Suggested incremental selection in `tests/training/test_mujoco_tyx_observation_training.py`:
`test_memory_public_constructor_preserves_real_type_failure_and_restoration`,
`test_memory_physics_interval_excludes_full_observer_and_retains_no_runtime`,
`test_memory_step_allocation_gate_and_original_failure_restore_wrappers`,
`test_memory_constructor_failure_unwinds_both_public_proxies`,
`test_memory_layout_records_only_owned_reference_counts`.
The old `test_memory_copy_facade_delegates_and_restores_without_native_owners`
is the focused compatibility selection (2 cases); broader old suites are not
required by this diagnostic-only increment. Parent performs static checks and
these HOST cases before one fixed S256 run with `--memory-diagnostic`.
