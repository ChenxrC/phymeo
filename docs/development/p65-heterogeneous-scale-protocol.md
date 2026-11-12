# G2a heterogeneous scale protocol — 2026-11-13

Status: CODE_READY_TESTS_PENDING / UNEXECUTED. Optional after the deadline plan's P0/P1;
a source-only handoff is not a scale qualification. No product/capacity change.

## Frozen sources and six cells

Reuse the complete independent consumer
`tests/installed/check_mujoco_tyx_heterogeneous.py`, SHA256
`379a2139257398a2dec25f74a474b59c417618bbc7cfa5f15b0795cccabaf8cb`.
Its scalar `coupled, early=True`, fixed BALL `r4-pyramid`, and FREE
`mixed` XML, starts, controls and numerical gates are retained. The new
consumer requires that exact sibling file, verifies its bytes before loading,
and imports no repository test/example. An outside run copies both consumers
unchanged. The original consumer is not executed as a second run.

| Cell | Worlds | Actual source buckets | Source membership |
|---|---:|---:|---|
| w1-scalar | 1 | 1 | scalar |
| w1-ball | 1 | 1 | BALL |
| w1-free | 1 | 1 | FREE |
| w8-b3 | 8 | 3 | 3 scalar / 3 BALL / 2 FREE |
| w16-b3 | 16 | 3 | 6 scalar / 5 BALL / 5 FREE |
| w16-b8 | 16 | 8 | 8 independently compiled sources, each 2 worlds |

Sparse declaration order is the prefix of
`(41,7,305,11,203,59,997,2,88,509,17,701,109,31,809,53)`.
B3 source assignment repeats scalar/BALL/FREE; original per-source slot
0/1 starts and controls repeat alternately, rather than extrapolating slots.
B8 source family order is scalar/BALL/FREE/scalar/BALL/FREE/scalar/BALL;
first eight worlds use source slots 0, next eight use the same sources slot 1.
For B8 only the XML root `model` name changes to a fixed unique identity; the
original anonymous BALL root receives this documentary attribute.
Restoring that one name must give the original XML byte-for-byte. Each source
is independently compiled, and all XML/archive/MJB hashes and physical native
model dimensions are saved. Actual layout must expose eight distinct buckets
with two worlds each; changing only a ModelSpec key is insufficient.

## Physics and independent reference

Every cell runs all 24 steps at h=0.001, with the original nonzero positive /
negative actuation phases and original later unloading controls. No abbreviated
cell, discarded world, changed h, extended force phase, or new tolerance.
Every world retains its own native MuJoCo 3.3.7 model and continuous MjData.
Every step also creates a same-entry native evaluation from the actual
accepted q/v/act/time and the actual FP32 control channels. Complete original
mass/J/FK/COM/bias/passive/applied/smooth and constrained acceleration,
activation, transmission, contact and constraint fields are checked by the
unchanged family oracle; all q/v/act/time are checked against both native
successor and the independent continuous trajectory. This is a force-and-motion
check, not successful-return or endpoint-only counting.

Original scalar tolerances, BALL TOL/ACT/FORCE/CONTACT_FORCE and FREE TOL/state
gates are read from the pinned helper and saved in each declaration. No scaled
tolerance based on W/B or new aggregate-only error gate. Native mj_step/forward
are forbidden during actual TYX step and integration readback; reference calls
occur only in their separately timed section.

After step 24, save a public checkpoint and full reference states; execute A,
restore the same checkpoint with current CAS, execute B and require the entire
accepted integration state/receipts exactly equal. Select one world from each
present family (unique subset; W1 naturally resets its one world), reset, and
execute a real successor. Unselected complete states/receipts remain exact.
Thus each positive cell has 27 successful calls; six cells total 1161 accepted
worldsteps per backend (43 worlds × 27), excluding isolated fault tests.

## New source tests

Predeclared, not collected: 26 nodes = 12 continuous cells (6×2 modes),
12 late-fault nodes (W16/B3 and W16/B8 × last-status/last-download/late-aggregate
× 2 modes), and 2 HOST admission nodes (W17/model9).
Each fault node first takes a real successful step, then faults after the final
bucket's real preparation/download or complete aggregate construction.
Save raw attempted scratch/publication before raising. All worlds false,
accepted banks, ticks, histories, checkpoint bytes and old receipt remain.
Restore only the fault hook and take the same real next step with complete
native comparison; never replace failed physics with a mock success.
Source tests may inspect original lifecycle fault seams; the installed positive
consumer uses the public API only.
W17/model9 must reject before model preflight or runtime allocation. No extra
capacity or unsupported-model domain is admitted.

## Evidence and timing

Each cell owns a fresh absolute directory and streams complete gzip JSONL:
declaration+source hashes/layout, every attempt (input/control/native reference),
immediate physics return, complete output/receipt, boundaries, failures, close,
and summary. No list of nested histories accumulates in RAM. A failed check
preserves the successful physics return and all observations already obtained.
Each world is represented every step; the summary is not a substitute for rows.

Wall times are disjoint call measurements for native compile/source preparation,
runtime construction, action submission, physics step, complete observation,
independent reference, validation, save/serialization, and boundary/cleanup.
Outer wall time is also saved; timers are host observation, not GPU kernel timing.
Raw counters retain their names. Allocation/upload cumulative counters are
explicitly not live/peak memory. Both live_device_bytes and peak_device_bytes are
UNKNOWN without an independent actual observer. This is no comparison to Newton,
no fused-kernel claim, and no measured throughput promise.

CLI:
`python check_mujoco_tyx_heterogeneous_scale.py --mode cuda --cell w16-b8
--installed-root ABS_SITE --output FRESH_ABS`.
The same entry accepts interpret. There is no steps override.
Source command (parent serial owner only):
`pytest tests/test_mujoco_tyx_heterogeneous_scale.py -q -x --tb=short`.
Parent may execute selected cells while reporting the others UNEXECUTED.
Any real force/state/reference or transaction failure preserves evidence and
stops that qualification; this authoring grant does not authorize tuning.

## Author handoff

Three NEW files only: this protocol, the source scale test, and the installed
scale consumer. Product source, original tests, pinned reference consumer,
capacities, and public exports are unchanged. The two consumer files must both
be copied to the outside scripts directory; the wrapper verifies the dependency
SHA before any Phymeo import. Module origins are checked before construction and
after close for every loaded Phymeo module. Source tests use the same wrapper
without the installed-root assertion, retaining the same physical oracle.

Suggested parent order is the three W1 cells, W8/B3, W16/B3, W16/B8, then the
bounded scaled transaction/admission nodes. Installed runs may select a complete
cell with --cell; the 24-step main trajectory and three boundary successors have
no truncation argument. Status and timing are recorded per cell, not inferred
from another cell or from previous G1. Authoring performed only text/file IO and
Git checks; no imports, collection, numerical evaluations, static tools or tests
were executed. Parent exclusively owns any subsequent execution and style repair.
# Parent source qualification, 2026-11-13

The fixed matrix is SOURCE_VERIFIED, installation pending. Run37 passed the
six original W1 cells (12.76 s); run51 passed the remaining twenty nodes
(147.85 s): W8/B3, W16/B3, W16/B8 on both backends, all six scaled failure/retry
combinations on both backends, and W17/model9 preflight negatives. No prior
green cell was repeated and no product capacity, model input or tolerance was
changed. Each positive cell retains its complete 24-step sequence and three
real boundary successors, for 1,161 accepted worldsteps per backend across
the six source cells. Source evidence is under
`artifacts/p65-feature-until-0730-20261113-01`.

This adds original-capacity functional qualification. Live/peak device memory
is still UNKNOWN and no Newton performance comparison has been performed.
The fresh installed consumers remain a separate pending stage.

## First installed package qualification

Package113629b3, wheel SHA2569250b9e9cea641992c7a1081fe134686e57bce2b718f9d8b6a8abb6a878b9044, has530 byte-identical product members across source/wheel/site. Runs77/78 execute the complete W16/B3 cell in interpret/CUDA; runs79/80 execute W16/B8 in both. All four report PASS, each27 actual calls /432 accepted worldsteps including the three real lifecycle successors. B8 preserves eight original source identities with two worlds each. W1/W8 installed cells were not rerun; their source qualification is retained explicitly, not promoted to extra installed runs. Every loaded Phymeo module is bound to the source-external installed site.

This closes G2a original-capacity functional qualification. The memory/performance limitations above remain; no new larger-capacity or Newton speed claim.
