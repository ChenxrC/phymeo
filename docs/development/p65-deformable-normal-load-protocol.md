# D4-N fixed continuous normal-load protocol

Status: **PREDECLARED / AUTHORING_UNEXECUTED**, 2026-11-13. This is the single
cases contract for the [07:30 plan](../superpowers/plans/2026-11-13-feature-completion-until-0730.md).
No project execution or parameter probe selected these values. Existing D3
remains the product; this slice qualifies a reusable public continuous runner.

## Fixed declarations

World 7, participant19/body202/shape0; deformable body101/shape0/surface23.
Point IDs `(1001,1013,1021,1031,1049)` have positions center `(0,0,0)` then
corners `(-1,-1,0),(1,-1,0),(1,1,0),(-1,1,0)`. Facet IDs `(503,509,521,523)`
use row triplets `(0,1,2),(0,2,3),(0,3,4),(0,4,1)`, all oriented +z.
Only center is free: inverse masses `(2,0,0,0,0)`. Four ordered radial edges
`(0,1),(0,2),(0,3),(0,4)` have rest length sqrt(2), compliance1/16384; no
fixed-rim constraints are needed. Every initial node velocity and gravity is0.

One real ordinary free body has mass2, principal inertia(.25,.25,.25), COM at
body origin, identity XYZW, initial velocity `(0,0,-1/16,0,0,0)`, no children or
actuators. Sphere local point ID41 is(0,0,0), radius1/8. Capsule point IDs41/43
are `(0,0,-1/8),(0,0,1/8)` with the same radius. Its COM is higher by1/8 so
both shapes' lowest axis point has the same initial gap9/128. Sphere COM z is
25/128; capsule COM z is41/128. Both are moving bodies, not prescribed paths.

The four facets are **four separate declared normal-layer springs**, each
width1/16, stiffness128N/m, damping coefficient0.5s/m, friction0. Nothing
renormalizes their sum or claims area-pressure/refinement invariance. 4k and
L/(4k)=1/32 are only initial common-normal dimensional references, not exact
sagged-membrane equilibrium oracles. Four active slots cover all four facets;
source/query/native budgets retain original public defaults and are recorded.

| Case | Shape/material | h | Steps |
|---|---|---:|---:|
| n1_sphere | sphere, compliance1/16384 | 1/1024 | 3072 |
| n2_capsule | normal capsule, same compliance | 1/1024 | 3072 |
| n3_soft | sphere, compliance2/16384 | 1/1024 | 3072 |
| n1_fine | sphere, original compliance | 1/2048 | 6144 |

Every window is3s. COM external force is `(0,0,-16)` for entry times `[0,1)`,
`(0,0,+16)` for `[1,9/8)`, then `(0,0,-16)` for `[9/8,3)`; external torque0.
Force changes use the public control channel and its real revision. The
positive phase is active outward lifting, not natural separation with zero
external work. There is no reset, state edit or zero-velocity replay in a
positive window. D1 uses substeps1, iterations200, tolerance2e-6 and full trace.

## Frozen observations and gates

Each actual attempt/return, raw query and physical candidates is streamed before
assertions. Full identity is(surface,round,facet), not facet-only deduplication.
The independent geometry oracle enumerates all active faces of the finite
segment/triangle domain with exact rational arithmetic. It does not call any
product detector. Its continuous FP64 material/root state advances from its own
previous state, using the same declared external schedule; it never drives or
repairs the actual device state.

Original same-entry gates stay unchanged: distance1e-4 L, witness2e-4 L,
normal2e-3, layer arithmetic5e-6 absolute/5e-5 relative, material position/
velocity/lambda2e-5 absolute/relative, mass2e-5 and rigid mechanics original
5e-6/5e-5. L uses local triangle/axis/radius lengths only. Geometry-to-force
error is explicitly propagated through the unchanged law. Long independent
trajectory gates are position2e-3m, velocity1e-2m/s, orientation2e-3rad.

Support windows are `[.75,1)` and `[2.75,3)`: time-mean total rigid contact
force along +z lies in[.8,1.2]*16N; rigid COM vz RMS and free-center vz RMS
are each<=.1m/s. No relative-velocity statistic can substitute for either.
Normal support peak<=128N. Every independently checked accepted geometry has
gap>=-radius/2 and remains on the membrane's original oriented side; the
unsigned distance may not turn a side crossing into a false recontact. This is
an endpoint engineering domain, not CCD or solid containment.

The event order is initial outside/zero force, entry/positive force, the first
support window, an outside/zero-force run during the lifting/reload interval,
and positive recontact before the second support window. Each sustained event
runs at least1/128s (8 coarse or16 fine steps). Event tests use actual entry
geometry and actual force, not scheduled labels; no active rows are dropped.
No requirement moves its statistics window after seeing output.

Material impulse sums every actual pre-row gradient times delta-lambda/h.
Fixed reaction includes that material impulse and all fixed external/contact
forces. Independently check both force and COM moment, discrete momentum and
position-update angular terms, sampled damping power, layer potential/work,
external work, and finite-step energy defect. The defect is a diagnostic, not
an invented exact continuous-energy conservation gate or internal credit.

## Bounded execution / failure

Four physical nodes total15360 steps/backend,30720 across both. Parent runs the
entire n1_sphere coarse node first, then the remaining fixed nodes. Each node
has fresh output, complete logical-stream hash and actual prefix on failure;
no in-memory list retains all historical receipts. Fixed original source/model
provenance can be referenced by a stable stream identity, but actual per-step
arrays/fields are not omitted. True construction, returned-step, audited-step,
worldstep and cleanup counts are distinct.

First failure stops that node and retains it. Do not change h, damping, load,
window duration, geometry, budget or gates to turn it green. Classify an actual
implementation/API/independent-oracle mistake separately from a bounded model
that fails its support/recontact gates. No further solver mechanism is
introduced just to enlarge test count. Friction, generic facet-history/CCD,
closed solids and general deformable scene domains remain outside D4-N.
## Authoring handoff and exact invocation

Checkpoint b1ad9c05 fixes the declarations and independent reference;326e8338
adds the streaming public runner. Both are frozen for parent review/execution.
The new test wrapper declares four full TYX windows plus one short TYX
observer-failure protocol; seven HOST-only declaration/oracle/statistic controls
remain separate. These are hand-enumerated **5 common TYX +7 HOST** nodes,
not collection or pass results. The observer fault occurs after a real returned
step and is explicitly recorded as committed physics plus failed observation,
not a fake rollback test. Existing D3 physical transaction tests remain reused.

First full node, from the repository with the parent's fixed environment:

```text
<python> examples/run_deformable_normal_loads.py --case n1_sphere --mode interpret --output <fresh-absolute-directory>
```

Then the identical n1_sphere CUDA window and the remaining n2_capsule,n3_soft,
n1_fine windows on each backend, or their exact pytest wrappers. A successful
CLI window and its wrapper are the same case/runner and must not be rerun merely
to obtain another test count. Source snapshots and full raw exits bind each
actual invocation. There is no shorten-window, omit-reference or tune-tolerance
flag. To use the runner outside the source, copy both public example files
unchanged and pass `--expected-install-root <absolute-site-directory>`; every
loaded Phymeo origin remains installation-bound.

Each invocation owns one fresh gzip stream and summary. Timing distinguishes
actual step, independent reference, save work and total elapsed; these are
observations, not a Newton comparison or a native-kernel-only profile. If time
runs out, whole remaining cases are PENDING; successful partial prefixes do not
qualify continuous support. All authoring here used text/file IO/Git only;
**CODE_READY_TESTS_PENDING / AUTHORING_SLOT_RELEASED** until parent validation.
## Parent diagnosis before continuation (2026-11-13)

The first full coarse sphere run (20) failed after 57 returned steps: D3 added
pair_origins_relative_m to public query witnesses which already included that
frame. The parent fixes the response consumer to add packet.origin_m once;
raw kernel witnesses and public query production remain unchanged.

Run21 on that fix reached 149 returned steps. Its original ideal momentum
identity omitted FP32 position-update rounding. The saved-only run23 checked
the complete predictor-to-row-to-final position chain without new physics:
step148 z imbalance -6.5078076246635405e-6 N s decomposes into predictor
+8.149072527885437e-9 and ordered position -6.515956697191426e-6; reconstruction
and exact closing difference are zero. This observation alone is not a bound.
The new independent example auditor checks every stored primitive transition
and a-priori binary32 half-ULP bounds, then explicitly records the predictor,
ordered-position and reconstruction impulses in linear and angular accounting.
Raw imbalance, original n*delta/h material impulse and fixed reaction stay
separate. Neither contact state nor physical force is repaired by the audit.

All original geometry, force, same-entry state, long-reference, support-window,
peak, oriented-side and unload/recontact gates remain unchanged. Only the
incorrect exact-arithmetic accounting identity receives its independently
verified discrete arithmetic terms. The original failures remain recorded.
The coordinate regression opts into the same auditor; old short controls keep
their original default checks. Outside-source D4 use now copies the runner,
case module AND deformable_impulse_audit.py together. No physical success is
asserted until the fixed complete window actually passes.
### Functional diagnosis after the retained strict precision failure

Run46 reached 301 actual returned steps, then failed the unchanged short
same-entry velocity gate: 2.27667386e-5 m/s absolute discrepancy. This strict
qualification stays FAIL. Following the user's functional-first direction,
the runner now offers explicit --diagnose-short-state: it streams each original
short position/velocity/multiplier assertion failure, with unchanged thresholds,
actual and reference arrays, while continuing the same fixed trajectory.
It can only return DIAGNOSTIC_COMPLETED, never PASS. Every force/geometry,
stored-arithmetic, long-window drift, oriented-side, penetration, peak and
support/unload/recontact requirement remains active. No input, damping, h,
statistics window or state is altered; this diagnoses real functional behavior
without pretending the original precision qualification passed.
# 2026-11-13 actual first full window

Run50 completed all 3,072 real steps of the unchanged N1 sphere, interpret,
3-second trajectory, with three control submissions and normal close. Entry,
release and recontact events occurred. All per-step force, geometry, stored
arithmetic, long independent trajectory, oriented-side, penetration and peak
gates completed. This is not steady-support qualification: final window
statistics failed, so the run status remains FAIL / DIAGNOSTIC_ONLY.

The original windows [0.75,1) and [2.75,3) had mean support force
24.8393461 N and 5.41425136 N, against the unchanged [12.8,19.2] N requirement.
Rigid velocity RMS was 0.584440/0.477767 m/s and center RMS 0.340356/0.384283
m/s, each above the 0.1 m/s requirement. Peak normal force was 69.546494 N;
minimum gap -0.042032315 m and oriented side 0.082967685 m remained within
their original gates. Ten short material-velocity comparisons failed, maximum
2.69912155e-5 m/s; the original strict run46 remains separately FAIL.

The full trajectory therefore distinguishes a working transient load/release
pipeline from unqualified steady support. It does not justify changing the
load, statistics windows, damping, h or tolerances to rewrite this result.
The other three declared grids remain unexecuted. Follow-up steady-support
work must explicitly investigate the existing undamped distance material and
entry-sampled layer damping; a new damping model needs its own contract and
validation, rather than attributing the observed oscillation to roundoff.

Run58 additionally passed the eight declaration/statistics-negative/observer
nodes. The observer negative records the already committed physical prefix;
it does not pretend a post-return observation failure rolled physics back.

