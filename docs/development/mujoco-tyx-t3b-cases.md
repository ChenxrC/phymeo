# MuJoCo-TYX T3b finite-pair preregistration

2026-11-10. Baseline `275872beca81767bdc2ee8a1d0ca52fb5ae9c100`. **AUTHORING_ONLY / UNEXECUTED**. Parent owns all project execution. This file fixes inputs and gates before new product output. Only NEW tests/test_mujoco_tyx_finite_contacts.py and this file belong to the test author.

## Native semantics and scope

CPU MuJoCo3.3.7 is the independent oracle. The existing condim1, source options, physical-M coupled QP, native Euler damping/integration and atomic publication remain unchanged. The added families are capsule-capsule, capsule-box and box-box; native geometry, regularization and source control are compared at the actual stored input, then an independent CPU trajectory advances continuously.

Read the fixed [capsule primitive source](https://github.com/google-deepmind/mujoco/blob/3.3.7/src/engine/engine_collision_primitive.c), [capsule/box and box/box source](https://github.com/google-deepmind/mujoco/blob/3.3.7/src/engine/engine_collision_box.c) and [native collision driver](https://github.com/google-deepmind/mujoco/blob/3.3.7/src/engine/engine_collision_driver.c). Parallel capsule axes can produce two endpoint contacts, while the general closest-segment branch produces one. Capsule-box may select a second contact. Box-box has face and edge branches; its CPU contact distance convention, including the face branch half-depth, is retained exactly.

Native driver filtering and stable compaction are part of the resulting contact set. New feature_index is the final per-pair native output ordinal; it is not a persistent material address. A complete (pair, ordinal) bijection preserves all final native contacts, including any equal/near-equal positions that remain. No sorting by position, output deduplication, or clipping CPU contacts to Warp's eight-point layout is allowed.

Static slots are capsule-capsule2, capsule-box2, box-box24; total static ceiling32 remains. The box24 bound is conservative scratch/output allocation, not an expected typical contact count. One box-box pair is admitted; two need48 and reject at construction. Runtime detected and active-row capacities remain separate and are tested at exact native count and count-1.

## Fixed inputs

Units are SI and native q/v coordinates. Defaults: h=.001 s, Euler, Newton source iterations128/tolerance1e-10, dense J, condim1, solref(.02,1), solimp(.8,.9,.002,.5,2), nonnegative margin. No parameter is fitted to TYX output.

| Group | Inputs and independent checks |
|---|---|
| capsule-capsule | parallel overlapping axes with two endpoint contacts; skew axes at right angle; endpoint-only; separation and exact positive-margin detection |
| capsule-box | horizontal side over a finite top face; vertical endpoint over top; side near a finite edge; endpoint near a corner; same-input full geometry, law, J, force |
| box-box | flat face; yawed face polygon; tilted edge; general nonidentity rotation/offset COM; retain complete native manifold and native distance convention |
| free eccentric rotation | m2kg, COM(.03,-.02,.01), inertia(.08,.10,.12,.002,-.001,.003), nonzero incoming translation/local omega; 24 steps capsule-box and 24 steps box-box with world COM wrench |
| two dynamic branches | fixed common ancestor, two hinge-slide branches with opposing moving capsules; both real joint groups receive nonzero equal/opposite contact force and nonzero joint torque; 32 steps |
| full unload/reentry | each of horizontal capsule-box and box-box on a real vertical slider, m in {.25,2}kg and h in {.001,.002}s, full Cartesian 8 windows, all from original contact initial state to .512s |
| capacities | same new layout exact detected/active native counts pass; count-1 fails atomically; detected inactive row counts; box24slots plus second pair48 rejects explicitly |
| batch/protocol | new finite-pair W4 sparse IDs, selected reset then clone next real step, actual late readback failure/all-world retry, real device step with CPU/Warp/HOST/NumPy numerical fallback disabled |
| public ENV | finite box-box contact force enters observation and control; real accepted action held through decimation; selected reset and next-step comparison, without changing contact law or rerunning old T3a cases |

The full unload input uses a finite static box halfsizes(.24,.18,.05). A horizontal capsule (radius.1, halfaxis.12) or box halfsizes(.12,.08,.06) starts .001m inside its top, incoming vz=-.02m/s, armature0/damping0. The unit-gear vertical motor applies zero until .096s, then upward 3mg during [.096,.160)s, then zero to .512s. Gravity is -9.81m/s^2 throughout. No q/v reset, position prescription, force injection after integration, or hidden support is used. All eight windows must exhibit initial real support, a positive-duration contact-free/zero-force interval, then positive force on reentry; final total support is checked against weight, not every contact being positive.

## Unchanged engineering gates and independent oracle

Reuse T3a's registered gates: point/distance/body pose rtol3e-5 atol4e-5; normal atol1e-4; J rtol2e-4 atol5e-5; M rtol3e-4 atol5e-5; smooth forces rtol4e-4 atol4e-4; R/aref/Jv rtol5e-4 atol4e-4; dual H rtol7e-4 atol5e-4; force/qfrc rtol3e-3 atol3e-3; qacc/integration rtol3e-3 atol1e-2; continuous q atol2e-4, v atol3e-3, time2e-6.

KKT independently reconstructs H=J M^-1 J^T+diag(R), b=J a0-aref from native same-input fields. With g=Hf+b, residual=max_i(f_i>0?abs(g_i):max(0,-g_i)), scale=max(1,max_i(abs(b_i)+sum_j abs(H_ij*f_j))); residual/scale<=5e-5 and f>=0. Runtime's own converged/iteration/budget/stored tolerance checks also remain. For <=4 active rows the existing independent exhaustive active-set oracle is retained.

T3a serialization/native mechanics helpers may be reused. New test-only wrappers supply final native per-pair ordinals and h-dependent integration acceleration; they do not use product geometry to manufacture expected values. The new complete-ordinal matching replaces only test-side matching when calling the old validation helper, never production calculations. Same-input and continuous CPU results are both checked, so neither a CPU endpoint restart nor matching only final q can hide an incorrect trajectory.

Each full window saves every q/v/control/time/contact-count/normal-force and selected complete input/native/receipt at first, pulse boundaries, first separation, first reentry and final. Failure adds its complete evidence. Compact JSONL only; no oversized proof snapshots. Phase behavior gates: support before pulse; at least .02s with no contacts and zero force after lift; reentry after pulse and before .48s; final .02s average force in [0.7,1.3]mg, |vz|<.1m/s and height within .01m of initial support. These are behavior gates alongside unchanged per-step native engineering comparisons.

No execution result is claimed. Old T3a suites are reused by acceptance and not rerun. Friction, generic mesh/CCD, multiple free roots, equality/tendon and G1 standing remain outside T3b.
