# MuJoCo-TYX T4 mixed constraint rows: source handoff

Status: **CODE_READY_TESTS_PENDING / UNEXECUTED**.

Author-only source commit: `a74117574d94834f88a131a1676e9aeab5a76f21`.
Parent interface authority: `2fe2e3f8c6e200119570fb45eb41e74e89094974`, [mixed-constraints plan](../superpowers/plans/2026-11-10-mujoco-tyx-mixed-constraints.md).
This isolated branch starts from research commit `e4a543575054505be0bc4b625ce47c81454f326d`; no parent WIP was merged or modified.

## Scope and binding

Exactly one product module is new: [`_constraint_rows.py`](../../src/phymeo/backends/mujoco_tyx/_constraint_rows.py).

- Original source bytes: 20823; 457 lines.
- SHA256: `9a735ce8de01cbd7c30e12af116f002ab29494c6f5cc9a9288658bde675af785`.
- Entire file was read back and compared to the complete authored string.
- Exact staged path and Git whitespace check passed.
- No import, collection, numerical probe, native call, test, lint, formatter, compile, build, install or profile ran in this lane. The module has not yet passed the TYX frontend.
- No old product source, model, device, solver, runtime, tests, exports or NOTICE changed. Parent owns shared wiring and the only execution slot.

Fixed source semantics follow MuJoCo 3.3.7 `engine_core_constraint.c`; device organization also references MuJoCo-Warp 0.0.2. The file retains Apache-2.0 notices. [The source research](mujoco-tyx-t4-constraint-design.md) separates unchanged primitive G1 from the unsupported mesh deployment scene.

## Exact public kernel ABI

Two independently created handles share one authored assembly function, following existing T1/T2 bank handling:

```text
assemble(
  count,nv,nslot,nfriction,nlimit,contact_capacity,efc_capacity,h,impratio,
  slot_ids,geom_ids,pair_condim,pair_friction,pair_law,
  friction_ids,friction_law,limit_ids,limit_law,
  q,v,motion,body_jacobian,normal_rows,
  efc_ids,efc_data,efc_J,contact_map,contact_J,counts,status
)
assemble_other: same ABI, separate tx.kernel handle

decode(
  count,nv,contact_capacity,efc_capacity,
  slot_ids,pair_condim,pair_friction,contact_map,counts,
  efc_data,contact_wrench,status
)
```

No extra scratch argument, host callback or model-specific precomputed dynamics is required. Scalar helpers `_power` and `_impedance` are `tx.func` functions. Inner loops are plain `range`; only worlds use `tx.range`. No while loops, bit shifts or augmented assignments are used.

## Data ownership and assembly

Assembly clears all EFC storage, six-component contact Jacobian storage and maps/counts before use. Padding up to V=max(8,nv) is cleared. Unused IDs/map entries are -1; unused data/J are zero; inactive contact first-EFC=-1 and row count=0.

The implementation writes exactly the frozen ABI:

- `efc_ids[W,E,5]`: native type, source ID, side/edge, static geometry slot, DOF.
- `efc_data[W,E,14]`: position, margin, velocity, aref, R, D, impedance, lower, upper, force, gradient, active, finite-upper mask, final diagApprox.
- `efc_J[W,E,V]`.
- `contact_map[W,C,3]`: static slot, first active EFC row, active EFC count.
- `contact_J[W,C,6,V]`: contact-frame linear n/t1/t2 followed by angular n/t1/t2.
- `counts[W,3]`: detected contacts, active contacts, actual required EFC rows.

Order is source DOF friction order, source limited-joint order with lower before upper, then static contact-slot order and native edge order. Static admission must supply native-ordered unique DOF/joint descriptors. Duplicate/absent descriptor bindings fail during parameter resolution.

Dry rows use native type1, source=dof, side=-1, static slot=-1 and bounds [-loss,+loss]. Lower/upper limits use type3, source=joint, side=-1/+1 and native qpos/DOF addresses. Both sides are independently retained when dist<margin. Contacts use type5 for condim1 or type6 for pyramids; source is the current compact contact ordinal, edge0..9 and static slot preserved.

Full point Jacobians are recomputed from both bodies' current COM Jacobians and point-to-COM lever arms. All six projected rows are saved even for detected-but-inactive contacts. Angular rows use relative angular Jacobians directly; they do not acquire an extra lever arm. Numerically zero rows are not removed.

Pyramidal dimensions 3/4/6 emit 4/6/10 rows, each Jnormal plus/minus the associated friction coefficient times its physical component. Condim1 emits one normal row. Every active edge uses its own J*v and the same normal distance/inclusion margin.

## Source-matched row parameters

After ordered compaction, a common parameter pass resolves the immutable DOF/joint/pair law for every stored row. This deliberately avoids trusting old normal-only R/aref fields.

- Joint-only models may have no pairs. Their fixed MuJoCo constants are MINVAL=1e-15, MINIMP=.0001 and MAXIMP=.9999 in device FP32; no pair_law[0] access is made. Contact rows use the equivalent constants supplied in their pair law.
- Impedance endpoints are clamped separately, not sorted. Flat width or equal endpoints use the CPU average. The power1/power2 branches and midpoint equality follow CPU source.
- Standard solref applies the existing enabled REFSAFE timeconstant>=2h rule; direct negative solref is retained. Parent admission remains responsible for rejecting unsupported mixed-sign laws and unsupported option flags.
- Dry friction sets K=0. All rows use aref=-B*Jv-K*imp*(pos-margin).
- Native R is clamped before pyramid regularization. Frictional rows use the first sliding coefficient, staged impratio adjustment and one common final R for every edge, with no extra dimension factor and no extra final clamp.
- D=1/R; final published diagApprox=R*imp/(1-imp).

No impedance, R, force or state is repaired. A power denominator that underflows/nonfinitely evaluates returns an invalid impedance sentinel0, which assembly rejects with status2. Impedance and R are checked before dependent reciprocal/square-root operations; all stored row/J scalars are checked for finiteness. This is an explicit finite-precision failure boundary, not an assertion that every native parameter law is representable in FP32.

## Capacity and failure behavior

All raw geometry slots are scanned. Detected contacts and actual EFC requirements are counted completely even when their configured capacities are exceeded. Only indices below the respective capacities are written. A truncated prefix is never successful: overflow sets status1 after complete counting.

An upstream geometry status1 still permits complete row/count reconstruction, because it is a capacity error. Other upstream errors leave freshly cleared, non-publishable outputs. Data/parameter/map errors set status2. Counts exactly equal to capacity do not fail.

When detected capacity is exceeded, rows for contacts outside the retained map are counted but not written through an out-of-bounds contact_J access. Maps for retained contacts can report a required EFC interval beyond capacity; this is an attempted mapping on a failed transaction, never a valid solve input. Solver/decode and parent publication must honor status.

## Decode and force publication

Decode first clears every contact wrench. It only decodes successful status0 rows and verifies map intervals against actual EFC counts/capacity. It checks the expected edge count and active-contact count.

Normal force is the sum of native pyramid edge forces. Component k is mu[k-1]*(force_plus-force_minus); unused components stay zero. Condim1 keeps only its normal force. No force is clamped, independently recomputed or applied to motion here.

Parent's common solver owns row force/gradient and J^T*f. Decode only publishes the same solved physical contact-frame wrench. Joint forces remain in typed EFC rows. Parent's Euler stage and double-bank transaction retain sole state publication authority.

## Runtime cost and unresolved checks

Resident output storage is O(E*V+C*6*V), with no E-by-E array. The law resolution pass scans at most F/L static descriptors per matching row (current maximum35/29); this is bounded source work, not measured performance. Contact work visits all admitted static slots and computes full J for retained detected contacts.

The following remain unexecuted and belong to the parent serial validation queue:

1. TYX frontend admission, lint and formatting.
2. Independent native comparisons for all four dimensions, inactive contact J, dry friction, both limit sides, flat-width/direct/REFSAFE laws and final diagApprox.
3. Solver integration, decoded wrench consistency, physical mass coupling and actual Euler outputs.
4. Exact-capacity and overflow, zero-pair joint-only worlds, multiworld/reset/stale-buffer and failed-transaction behavior.
5. Unchanged G1 integration after generic constraints, and fresh outside-source wheel consumers.

The authoring slot is released at handoff. No test pass, physical support result or G1 success is claimed.
