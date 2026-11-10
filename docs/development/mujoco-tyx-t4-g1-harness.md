# G1 MuJoCo-TYX public policy harness — authoring handoff

Current parent acceptance: the original20s CUDA trajectory, h=.001/2s and both signed push/2s windows passed on the final solver, alongside210 matrix nodes, six same-input pose checks, outside-source wheel consumers and the complete saved-evidence audit. See the [current validation report](../benchmarks/p65-mujoco-tyx-mixed-constraints.md). The original asset URLs and redistribution boundary are in the [G1 source plan](../superpowers/plans/2026-11-09-training-humanoid-priority.md); external policy weights and motions are not bundled into the product.

Historical authoring status: **CODE_READY_TESTS_PENDING / UNEXECUTED**. The handoff below describes the authored snapshot; subsequent parent execution does not retroactively change that snapshot's status. Authoring slot was released after the report commit; parent validation is now complete and its execution slot is released.

## Scope and immutable inputs

New executable entry: [g1_mujoco_tyx_policy.py](../../examples/g1_mujoco_tyx_policy.py). No existing product, test, asset, or old example was modified. Authority is [the T4 mixed-constraint plan](../superpowers/plans/2026-11-10-mujoco-tyx-mixed-constraints.md), especially its G1 section, and [the fixed cases](mujoco-tyx-t4-cases.md), sections 4 and 7.

The entry selects the existing `g1_primitive_actuated.xml`, SHA256 `f7e856e057a095a9f8708d302531d40ff2519440012c24488407ae8d0f0586aa`. It does not select `scene_g1.xml` or silently simplify the latter. Original primitive G1 has nq=36, nv=35, nu=29, nbody=31 and ngeom=34; original exclusions, limits, mass, inertia, COM offsets, friction, actuator ranges, Euler/pyramidal configuration and iterations=100 stay unchanged. Public backend capabilities must report the complete 502 eligible pairs and 942 geometry slots. Runtime live capacities are contact=128 and EFC=512.

Original XML, YAML, NPZ, ONNX and external ONNX data receive start/end byte-size/SHA bindings. The compiler record retains original/derived XML, archive and MJB hashes and lengths, not an unserializable binary object in JSON. Native version is explicitly 3.3.7.

For h=.002 the original disk XML is compiled through the public compiler. For the already authorized h=.001 matrix, only XML `option.timestep` is changed **in memory** before public compilation, and the derived XML hash is recorded separately. Original disk bytes are never edited. This is a declared timestep variant, not a new asset pass.

The real primitive model has **seven** named foot capsules per ankle, fourteen total. The harness resolves the exact left/right foot1…foot7 names. Other robot shapes are not counted as foot supports. The prior case-plan “eight per foot” text was reported to the parent as a documentation discrepancy.

## Public stepping and CPU boundaries

Simulation construction and control use only `compile_mujoco` / `compile_mujoco_xml`, `MuJoCoRuntime`, `MuJoCoInitialState`, `MuJoCoControl`, `set_controls`, and `step`. Modes are explicit interpret or CUDA; there is no CPU/Warp fallback.

There is one narrowly authorized private diagnostic read: if `runtime.step` raises, `step_with_failure_capture` calls `runtime._device.download()` once before context-manager close, writes the entire rejected device scratch and counters, and re-raises. It does not advance, rerun, reset, publish, or feed a repaired state. If the download also fails, both failures are preserved. Successful runs do not use this private access.

Public `snapshot()` performs detached CPU observation. The harness explicitly labels this boundary and uses snapshot q/v/FK for policy and pose metrics. Device contact forces are read **only** from the last accepted device receipt, never from snapshot contact fields.

Three state streams remain separate:

1. TYX device trajectory advances with public step, retaining its published input q/v, stored control and input-step wrench.
2. A fresh independent native `mj_forward` at each actual stored device input supplies same-input native forces/accelerations. It never feeds state into TYX.
3. A separate native `mj_step` trajectory starts once from the same first published state, then feeds back its own q/v to its own controller/action history. No later TYX endpoint is copied into it.

The independent native Euler trajectory reads its post-step cached input-phase M/FK/contact quantities with separately saved input q/v. Fixed [MuJoCo 3.3.7 Euler source](https://github.com/google-deepmind/mujoco/blob/3.3.7/src/engine/engine_forward.c#L769-L815) uses a temporary acceleration for implicit damping and leaves the physical input `d.qacc` available. Input force time and output motion time are recorded separately. Native invalid-state/acceleration warnings and unexpected time changes stop the paired run rather than allowing an unnoticed native auto-reset.

## Original policy and initialization

The protocol follows [the original native reference example](../../examples/g1_policy_reference.py): 29 joint IDs in actuator order, 154-value observation, 29-action ONNX output, CPUExecutionProvider, original gains/action scale/action offset/default target and actuator limits. Policy updates every .02 s, exactly 10 steps at h=.002 or 20 at h=.001. Previous-action history commits only after the corresponding physical step succeeds.

Motion initialization copies original frame-0 nonzero q/v. The root linear velocity remains world root-origin velocity; angular velocity is converted from original world angular velocity to native local coordinates. The initial quaternion is normalized as in the original reference; later published q is never repaired. FK/observation quaternion normalization is a local observation copy only.

The root angular observation reads `R(model.body_iquat[root]).T @ qvel[3:6]`: native free-root qvel is body-local, whereas native `mj_objectVelocity(BODY, local=1)` reports the body inertial-frame axes. The original pelvis inertial quaternion is nonidentity and is preserved. Both new feedback streams apply the same transformation from their own current state. This explicitly corrects the old script's stale cached cvel after Euler, rather than reproducing the one-step observation lag. It is a current-state observation protocol correction; old native reference results are not imported as new execution evidence.

Push cases apply ±.1mg along world x at the pelvis COM during source-grid time [.5,.6). They do not add an artificial torque, teleport the body or clamp orientation. The requested and actual stored FP32 device packets are both saved.

## Four CLI phases and comparison contracts

| Phase | Authored behavior | Meaning of success |
|---|---|---|
| admission | Compile original/declared h variant; construct public runtime; save full source model inventory, candidate layout, capacities and published initial state | ADMITTED_NOT_PHYSICS only |
| snapshots | Independently initialize source default pose, original frame0 and frame200; perform one real device step and an independent same-input CPU step for each | Full F1 field checks, complete contact/EFC correspondence, independent KKT and endpoint checks |
| trajectory | Original motion frame0, original CPU ONNX feedback; same continuous TYX instance and independent CPU feedback stream | Fixed F2/F3/F5 engineering gates for the requested window |
| stand | Original default source pose and original default PD target/gains; separate diagnostic | Explicit stand diagnostic, never presumed stable and never substitutes for trajectory |

F1 compares complete current body pose/COM/J, M, bias/passive/applied, smooth acceleration, physical qacc, implicit-Euler acceleration, actuator and constraint forces. Contact matching uses a complete pair/geometry/position/normal bijection; no force-based selection, ordinal coercion, duplicate removal or subset comparison is used. Each matched contact retains all six wrench/J components. Native EFC identities use type, native joint/DOF/contact owner and side/edge; every native row must be matched exactly once.

Engineering thresholds are taken from the fixed case document: geometry 3e-5 relative/4e-5 absolute, normal/frame 1e-4 absolute; J 2e-4/5e-5; M 3e-4/5e-5; bias/passive/applied 4e-4/4e-4; smooth a 2e-3/4e-3; R/aref/Jv and related row parameters 5e-4/4e-4; force/generalized force/couple 3e-3/3e-3; qacc/integration 3e-3/1e-2. One-step endpoint q/v/time use 2e-4/3e-3/2e-6 absolute. No tolerance is fitted to execution output.

Independent KKT reconstructs H=J M^-1 J^T+diag(R) and b=J a0-aref from the same published physical fields, checks each lower/interior/upper residual, exact box membership, J^T force reconstruction and normalized residual<=5e-5. This is a CPU oracle in the harness, not a device fallback or face-scratch reuse. It is used for F1; the continuous policy phases retain the same-input native comparisons as saved diagnostics and use the fixed robot behavior gates rather than forcing independent feedback trajectories to match bitwise.

## Continuous windows, metrics and evidence

The standard trajectory run with duration20 uses one runtime, one controller history and one native feedback trajectory from t=0 through 2 s and onward to 20 s. The 2 s checkpoint is evaluated before continuing. No reset or history truncation occurs. Parent execution order still requires preceding F1/F2/F3 gates; the harness does not manufacture proof of earlier independent runs.

Per-step JSONL includes actual device q/v/time/tick, requested/stored control, action/target/observation, all contact six-wrenches, typed constraint force summaries, qfrc_constraint, same-input native force/qacc, independent native q/v/control/action/contact forces, source time, foot load/unload events and diagnostics. Output paths must be fresh; records are streamed and flushed. Source bindings, selected records, failures and summary are fsynced. No evidence compression/deletion is performed.

Full device receipts, including dynamics and complete raw EFC/J buffers, plus full native fields are selected at the first step, 2 s checkpoint, final requested step, first failure, either feedback stream's foot load/unload event, or any change in the complete device constraint identity tuple `(type, source_id, side_or_edge, geometry_slot, dof)`. Each full record states its capture reasons. Event-driven native expansion reads the same cached evaluation; it adds no forward or step. A first behavior failure expands the already evaluated native data without another `mj_forward` or TYX step. The rejected-candidate path additionally saves the real device scratch before close. Full F1 receipts are saved before matching, so a matcher exception cannot lose the failed physical output.

Each accepted device output must additionally bind to `tick+1` and the source output grid time within the original 2e-6 s clock gate. A mismatch records the current full receipt as the first failure.

Robot gates retain height>=.35 m, tilt<=1 rad, joint RMSE<=.5 rad, reference XY<=1 m and all-window peak foot vertical force<=8mg. Both all-window and settled mean are recorded; the mean acceptance interval [.5,1.5]mg uses the existing .25 s settling boundary. The mean window is explicit in metadata. The parent should preserve this declared boundary or adjudicate a change before execution, not silently reinterpret a resulting mean.

Foot diagnostics retain both feet separately, world force, pure contact couple, moment about each foot-body COM, contact-frame tangent speed, loaded slip integral and non-foot ground reaction. The stand diagnostic additionally checks tilt .35 rad, root XY .1 m, foot-center XY .08 m, mean [.7,1.3]mg and fraction both feet above1N>=.9. Stand foot displacement uses detached endpoint FK; policy trajectory foot-center diagnostics use input FK. This phase distinction is explicit.

Every joint's soft-limit excess and actuator saturation are recorded. Actual motor torque is checked against original ranges with 3e-5 relative+3e-5 Nm allowance. A soft limit is not falsely presented as a strict geometric bound.

Energy diagnostics use actual input M/v/COM, kinetic and gravitational energy, actuator/external/passive/viscous/constraint power and integrated work. Typed dry/limit/contact power is retained. The discrete balance is diagnostic: no omitted soft-contact potential is called conservation, and no new energy cutoff is invented.

If either independent native policy/plant or device trajectory violates a declared robot gate, the run stops and records which stream failed. It does not adjust mass, damping, friction, policy, geometry or initial pose to force agreement. Windows shorter than2s are DIAGNOSTIC_ONLY. A 20 s pass would still not mean the full131.48 s dance passed.

## Deferred parent-only execution examples

These commands are **not executed** in this lane. Parent must supply the existing environment, actual queue guard, child-src/PYTHONPATH and fresh output paths. `--cache-root` is optional and never changes the model.

```powershell
python examples/g1_mujoco_tyx_policy.py --assets <original-assets> --output <fresh-F0.jsonl> --mode interpret --phase admission
python examples/g1_mujoco_tyx_policy.py --assets <original-assets> --output <fresh-F1.jsonl> --mode interpret --phase snapshots
python examples/g1_mujoco_tyx_policy.py --assets <original-assets> --output <fresh-F2-h001.jsonl> --mode cuda --phase trajectory --h .001 --duration 2
python examples/g1_mujoco_tyx_policy.py --assets <original-assets> --output <fresh-F3-plus.jsonl> --mode cuda --phase trajectory --h .002 --duration 2 --push-sign 1
python examples/g1_mujoco_tyx_policy.py --assets <original-assets> --output <fresh-F3-minus.jsonl> --mode cuda --phase trajectory --h .002 --duration 2 --push-sign -1
python examples/g1_mujoco_tyx_policy.py --assets <original-assets> --output <fresh-F5-continuous.jsonl> --mode cuda --phase trajectory --h .002 --duration 20
```

A CUDA2s no-push gate can be the same instance's 2s checkpoint in the continuous20s command, once parent prerequisites permit it. It is not necessary to rerun a green same-source standalone prefix merely to call the checkpoint a new result.

## Authoring validation and handoff

Author-only checkpoints:

- `80b1185911795a749cce8976a99d71ac98433ee4`: complete first harness.
- `57d86b2b129f1561961a424d675aa1f0e18f40e9`: pure-text review correction of JSON source binding and explicit full model inventory.
- `8d3acd4b86f33be37ca8ad26b42f143a234af712`: parent-requested rejected-device scratch capture.
- `3a34b9e0e448eac7e58cfadf5ab4fbf223bc713f`: parent-authorized narrow review corrections: current inertial-frame gyro, clock/tick gate and same-run contact/constraint event evidence. No model, policy action mapping, frequency, physical gate or .25 s settling boundary changed.

Current source: 59,110 bytes; SHA256 `0170f0646f795160467c5b7b6ba4100b973a53d4aba3764505b466195fdc4d6d`; Git blob `ef8112233675a5fe6fad40e743885631eb9530c3`.

Author strings were compared with complete file readback. Small commits used exact paths and Git diff whitespace checks. **No import, collection, numerical calculation/probe, lint, formatter, pytest, build, install, profile, interpret or CUDA execution was performed.** The source may still contain ordinary syntax/API/style defects; parent static and runtime checks remain required. There is no actual G1 result in this report.

The example intentionally depends on the parent's T4 public typed receipts and larger candidate layout; the child branch alone has not integrated all parent shared changes. Parent is responsible for the combined source snapshot, first raw failures, static repair, independent cases and serial G1 execution. This report releases authoring only and does not grant any execution or claim T4/P6.5 completion.

## Parent storage supplement before first G1 execution

The parent added optional streaming gzip output selected by a .gz suffix. Exclusive file creation, flush/fsync and finally-close remain; the exact JSON records and all fields are unchanged, with no redundant plain copy and no deletion of old evidence. The summary sidecar remains ordinary JSON. This is storage only, not a change to the model, comparison thresholds or selected-event policy. It is unexecuted until the parent admission/trajectory jobs below supply actual results.
父首次准入：45因Python3.12现有环境缺onnxruntime而在harness import前失败，未执行G1物理；原Python3.11 humanoid环境版本为onnxruntime1.22.0/PyYAML6.0.3。57以原版本安装对应3.12 wheel到本批独立policy-deps-py312目录（--no-deps，不改原venv/项目pin）。58因快照路径数组组装错误在执行前被拒，无RAW；59修正路径后ADMITTED_NOT_PHYSICS/RAW0，所有原资产before/after绑定相同，lossless .gz记录已正常关闭。F1及持续窗口仍需真实执行。
## Parent checkpoint snapshot correction

Before the next original-initial-state long run93, the parent found that Stats.summary returned references to mutable accumulated slip and work. The earlier51/81 stream checkpoint records preserve their correct point-in-time values, but final summary.checkpoints reused updated slip/work containers. All scalar behavior gates, input/force/state trajectories and original failures are unchanged. The harness now copies exactly these two diagnostic containers; new runs require exact stream-versus-final checkpoint equality. Historical files are retained verbatim and the known two-field alias defect is explicitly reported, never normalized away as though those old summaries were complete immutable checkpoints.94/95 static checks passed. This is a saved-observation correction, not a physical state change.
