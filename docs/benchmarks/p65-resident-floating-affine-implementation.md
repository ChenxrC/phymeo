# C11 resident floating affine implementation

Status: **CODE_READY / UNEXECUTED / AUTHORING_RELEASED**. The authoritative contract is
[the C11 plan](../superpowers/plans/2026-11-08-resident-floating-affine.md), parent
`d8bbf40`. RD tests-first checkpoint `91099315bf6ee1c015327eaa7c7c68d592b73e03`
precedes this product authoring. No import, collection, numerical probe, test,
static check, formatter, build, installation or device execution is authorized
or claimed here. Parent owns all OLD hooks and all execution; C10 formal
acceptance remains the prerequisite for C11 integration/execution.

## Public declaration and owned identity

`FloatingAffineRobotTree(asset, state, *, kp, kd, effort_limit, delay_steps=0,
control_max_iterations=512)` accepts the existing `FloatingMimicRobotAsset`
and independent `FloatingRobotState`. There are n real scalar joints,
1 <= K < n <= 16 independent motors, B=n+1 bodies, Dfull=6+n and Dreduced=6+K.
The physical model is the actual full floating model; no reduced physical body
or motor for a root coordinate is created. `inputs(drive, external=None)` owns
K-by-3 target/rate/feedforward and all B COM force/torque rows. External full
joint effort must be n zeros; it is not a second motor channel.

`floating_affine_model_token` preserves the original FP64 physical model,
joint names/limits/ratings, original relation order and coefficients, and
composed P/c. Compilation rejects source mutation and keeps FP32 tables
separate from this token. Full and independent original intervals/rates and
normal-FP32-or-zero input checks are retained. Representable position/speed
limits are rounded inward before computing the joint intersection. Current
state-dependent mechanics is never computed by the compiler.

The public checkpoint has independent Q=7+K and D=6+K fields and explicit full
Q=7+n/D=6+n fields. `control` is exactly five immutable blocks:
`(graph_token, drive[K,3], effort[n] zeros, force[B,3], torque[B,3])`.
`local_origin_m` preserves the actual FP32 root bank independently of the FP64
anchor/world-origin observation. Full and reduced root values must agree;
restore validates the supplied full state and original graph rather than
projecting it back or reconstructing local coordinates from a rounded world Q.

`FloatingAffineTreeKinematics` keeps full position/velocity/body geometry and
adds `independent_position`, `independent_velocity`, `joint_names` (K), and
`full_joint_names` (n). Diagnostics retain full M/b/external/J and add their
reduced counterparts, both entry coordinates and accelerations, K actual drive,
original `row_multipliers`/`row_powers_w`, `constraint_power_w`, and six raw
`root_constraint_residual` components. All returned arrays have owned immutable
storage. They remain same-attempt diagnostics, not a permanent state history.

## Exact compiler and storage seam

`compile_floating_affine_tree_world(world) -> FloatingAffineTreeWorldSpec` has
`physical: FloatingTreeWorldSpec`, `domain='floating_affine_tree'`, kind=11+delay,
`n/k/r`, B/D/Q and reduced D/Q properties. Original physical n/B metadata is
never replaced by K. The physical spec has the same complete C11 topology,
zero ordinary full motor gains/commands/history, and actual expanded initial
n-row state. K owns its own initial drive/history. `projection` is S=diag(I6,P).

- `spec.packet(packet) -> (drive[K,3], effort[n], loads[B,6])`, all owned FP32;
  graph mismatch or nonzero external effort fails before control publication.
- `spec.checkpoint(world, root, reduced, full, reduced_applied, full_applied,
  tick, control)` creates the public floating-affine checkpoint.
- `spec.restore_row(checkpoint, world) -> (root[13], reduced[K,7], full[n,7],
  reduced_applied[6+K], full_applied[6+n], packet, tick)`.
- Ordinary validation helpers receive an explicit full-dimensional private
  checkpoint and neutral n-row motor packet, never the K packet. Public
  checkpoints retain their floating-affine type.

The NEW `FloatingAffineTreeDeviceStorage(batch, specs_by_world)` composes
one existing `FloatingTreeDeviceStorage` as `.physical`. Its `rows` order must
match the physical storage exactly. Both use the original batch runtime, global
slot/ticks, active bank, lock/epoch and resource stack. The wrapper owns K-state
and reduced-applied banks, not a separate root bank. `stage_mechanics(src)` and
`stage_candidate_geometry(dst)` delegate to the real physical tree.
`stage_solve_from_mechanics(h,src,dst,tick_src)` projects once, calls the shared
solve once with K gains and the real root COM/FK, then expands and checks original
rows. `stage_finalize_and_retain(src,dst,tick_src,tick_dst,accepted)` uses the
physical full finalize and the same accepted global slot to retain K banks.
`stage_advance` only orders these stages; it does not swap/publish.

Control lifecycle remains `validate_controls -> commit_controls ->
upload_controls -> controls_uploaded`; the last call occurs only after every
storage upload succeeded. Restore lifecycle remains `validate_restore ->
stage_restore -> check_restore -> parent swap -> commit_restore`.
`check_advance` reads small status before the sole swap; `finish_advance` only
publishes HOST validity after that swap. Explicit snapshot/FK/diagnostic reads
are counted. `borrow_geometry(active)` delegates to the actual full FK/root
bank and the existing floating bounds kernel. `close` never destroys a borrowed
runtime. The corresponding owned stage types are FloatingAffineControlStage and FloatingAffineRestoreStage.

## Required mathematics and limits

Full mechanics and any C10 contact J-transpose contribution precede exactly one
S-transpose projection. The unchanged shared `floating_solve` receives
`count,h,solve_meta,motor_gains,physical_root_com,commands,root,state,fk,applied,
tick,next_root,candidate,next_applied,mass,factor,mobility,vector,info,diag`.
For ordinary trees K=n; for this adapter K is independent motor count. Root/FK
are always the actual full tree, so the root exponential/COM-origin drift is
performed only once. NEW kernels use the final C9 floating arithmetic helper.

The full motor is E*tau on original independent driver rows, not P*tau. Full
applied load is external+E*tau; constraint force is recorded separately. Original
row recovery checks all Dfull equation rows including root6, S-transpose balance,
row position/velocity/acceleration and row/total power with the frozen 2e-5
engineering gates. PD retains its original 5e-5 residual and finite iteration
budget. A projection failure flag must survive the shared solve's status reset.
No joint/root state repair, extra force, HOST solve, changed input or tolerance
is part of the adapter.

New storage status names follow the existing affine family: codes 7/8 are
`AFFINE_CONSTRAINT_RESIDUAL`/`AFFINE_FULL_EQUATION_RESIDUAL`, code9 is
`SITE_CAPACITY_EXCEEDED`; 0..6 retain the ordinary tree meanings. Ordinary
floating code7 remains unchanged. Parent contact routing must use the separate
floating and affine facts, not classify every nonordinary tree as fixed-affine.

## Scope and pending verification

Only the following NEW product paths belong to this author: the schema/compiler,
`_resident_floating_affine_storage.py`, `_tyx_resident_floating_affine.py`,
`_resident_floating_affine_binding.py`, and this report. Parent owns all existing
batch/solver/contact/export hooks; RD owns the complete A01-A05/X/W tests and
independent reference. This checkpoint does not claim any test or timing result.
C9/C10 geometry/law/runtime are reused without changing their supported domains.
Nonlinear or cyclic constraints, cross-participant coupling, persistent friction,
CCD, all-device narrow phase, floating training gradients and performance gains
are not implemented or claimed by this work.

## Completed NEW storage, kernels and caller handoff

All four NEW Python modules are authored; execution and OLD integration remain
pending. The schema checkpoint was `9b0411f`; projection/expansion and binding
were committed as `189d3f4`. Parent text review found no blocking issue in those
committed sources, which is not numerical/static/device validation. The final
schema additionally compares the supplied full/reduced root velocity values
before FP32 conversion, so rounding cannot conceal inconsistent root records.

`FloatingAffineTreeDeviceStorage` owns `states[2]` [T,Kmax,7] and `applied[2]`
[T,6+Kmax], with `.physical` owning the only real root/full-state/FK banks.
`meta[T,4]`, independent indices, row indices and reverse traversal are int32;
all numerical device tables/scratch are FP32. `rows/worlds/count` match physical
rows exactly, independently of the original batch's global slot numbers.
`is_floating=True` and `is_affine=True` are separate properties. `nmax/bmax/dmax`
remain physical, while `kmax/reduced_dmax` describe algebraic dimensions.

`FloatingAffineControlStage(owner,drive,effort,loads,physical)` owns all three
HOST blocks and the already validated full physical control stage. `_held_drive`
is K, `_held_effort` is n zeros and `_held_loads` is B-by-6; physical has its own
neutral n-drive plus the same real instantaneous COM loads. Parent temporary
contact snapshots must preserve both sets. Uploads cannot clear dirty flags
until every original batch storage upload has completed. Neither partial upload
nor subsequent candidate/receipt failure writes the active state or epoch.

`FloatingAffineRestoreStage(owner,mask,ticks,state,applied,controls,physical,
validity,status_codes)` stages both banks through the same original global tick
mask. Full restore delegates to physical, K restore copies or restores its own
state/applied only, and complete map validation occurs before parent publication.
Full geometry checks plus K/full original-row/domain/applied consistency remain
separate checks. The wrapper's reduced retain uses only the full finalize's
actual `accepted[global_slot]` decision, including site-capacity or geometry
failure that occurs after the algebraic solve. No early success is published.

The NEW factory `make_floating_affine_kernels()` returns two handles per key:
`project`, `expand`, `retain`, `restore`, `validate_state`. There is no copied
mechanics, PD solver, root integrator, FK, energy, contact law or bounds kernel.
`projection_flags[T]` survives solve's info reset. `reaction[T,Dfull,3]` retains
raw full balance, reverse-row working balance and reconstructed original-row
force; `row_receipt[T,n-K,2]` stores lambda and entry power. `metrics[T,4]` stores
full normalized residual, total entry power, original-row relation residual and
projected residual. Public constraint effort follows C5's raw balance convention;
`root_constraint_residual` exposes those first six entries instead of replacing
them with declared zeros. Original reconstructed rows are checked against that
balance before acceptance. It is not an exact-arithmetic force certificate.

`diagnostics_from_arrays(worlds,full_arrays,reduced_arrays)` creates owned
same-attempt diagnostics without device reads or calculations of mechanics.
Both mappings must cover all storage rows with actual dtype and dimensions:

- full: linear/angular [T,Bmax,3,Dmax], mass [T,Dmax,Dmax], vector [T,Dmax,14],
  entry_pose [T,7], info int32[T,4], diag [T,3];
- reduced: linear/angular [T,Bmax,3,Drmax], mass [T,Drmax,Drmax], vector
  [T,Drmax,14], reaction [T,Dmax,3], row_receipt [T,Rmax,2], metrics [T,4].

Caller supplies only worlds with same-attempt accepted info. `_read_diagnostics`
is the explicitly counted readback wrapper; public `diagnostics` uses latest
accepted-attempt validity and returns None after restore/latest rejected attempt.
For a contact packet, parent reads physical root/full-state/full-applied/FK/twist
plus K state/reduced-applied from the specified same bank, then invokes the
compiled spec's checkpoint method; it must not call the ordinary checkpoint
method with K controls. C11 receipt fields are `coordinate_kind='floating_affine'`,
`floating_affine_diagnostic`, full/reduced/independent counts and complete S/c/C
as frozen by parent. J-transpose and S-transpose are each applied once.

`ResidentFloatingAffineTreeBinding(key,world,group_index,local_points_m,*,
topology,graph_token)` requires the original graph keyword. `from_asset` delegates
the full ordinary floating collider/fixed-tool transform once, then attaches the
graph token. `compile_floating_affine_tree_binding(shape,binding,spec)` returns
the original `FloatingBindingCompiled`; full group count, material IDs/skin and
actual model proof still pass the original compiler. Parent must dispatch this
subclass before its ordinary floating base. `borrow_geometry(active)` returns
the physical view directly, so current device FK and original bounds are shared.

New read/upload categories are prefixed `floating_affine_` (topology, initial,
control, restore, state, fk, diagnostic, status). Stage launch counters separately
count projection, shared solve, expansion, retain, restore and validation; full
mechanics/FK/energy/summary retain their physical floating counters. Explicit
snapshot may read bounded complete banks; warm bounds do not call it. Buffers
are allocated on the batch's existing resource stack and closed there; the
wrapper does not close the borrowed runtime or claim zero transfer cost.

Final author scope is exactly:

1. `src/phymeo/backends/_resident_floating_affine_tree.py`
2. `src/phymeo/backends/_resident_floating_affine_storage.py`
3. `src/phymeo/backends/_tyx_resident_floating_affine.py`
4. `src/phymeo/backends/_resident_floating_affine_binding.py`
5. `docs/benchmarks/p65-resident-floating-affine-implementation.md`

No OLD source, export, test, fixture, prior report, runtime or dependency was
modified. All A01-A05/X/W membership and thresholds remain owned by the frozen
plan/RD tests. Import/collection/lint/format/interpret/CUDA/physical windows,
Newton references, wheel and saved audits are **NOT RUN by this author**.
The code depends on the parent-only shared solve signature and C11 routing;
absence of those hooks in this isolated author tree is deliberate. Parent will
record actual outcomes after C10 acceptance. This handoff asserts source scope
and interface readiness only, not functional acceptance or performance.

## Pre-execution diagnostic readback review correction

RD's source review identified that owned dataclasses alone did not reject a
nonfinite same-attempt diagnostic or inconsistent entry/exit ticks. This was an
authoring finding, not an executed product failure. The pure adapter now checks
code zero, a nonnegative entry tick and exactly one exit tick, then checks only
each selected world's actual B/Dfull/Dreduced/original-row slices: both J/M/vector
sets, full reaction, row receipt, metrics, pose and diagnostic scalars. Padding
and unselected worlds cannot invalidate an unrelated requested record. Assembled
full/reduced world-position arrays are also checked after the FP64 anchor sum.
The adapter raises ValueError before returning any candidate mapping; it changes
no mechanics, tolerances, device arrays, publication or test inputs. These checks
and the corresponding new RD fault tests remain UNEXECUTED by this author.