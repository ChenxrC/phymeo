# D6-B continuous-load application protocol candidate

Status: **PREDECLARED / UNEXECUTED / WAITING_FOR_D6_A_INTEGRATION**, 2026-11-13.
This document grants no execution and changes no old case or runner. The
[parent D6-A contract](../superpowers/specs/2026-11-13-distance-dissipation-design.md)
must pass its kernel, public D1 and short D3 integration before the parent may
authorize this application. The original [D4 protocol and failures](p65-deformable-normal-load-protocol.md)
remain separate results. In particular, original run46 strict qualification
and run50/run69 steady-support failures are not overwritten.

## 1. One additional material declaration

Use the original five-node membrane, its four ordered radial edges, four
facet springs, rigid model, controls, all identities, source budgets and all
physical gates. The sole additional physical parameter is:

**Every radial edge has c = 4096 N s/m, in every one of the four cases.**

The original ordered stable endpoint pairs are
(1001,1013), (1001,1021), (1001,1031), (1001,1049). Declare these exact pairs and
a float64 coefficient vector (4096,4096,4096,4096) through
DistanceDissipation, then the keyword-only DeformableNodeWorld.dissipation.
The coefficient is exactly representable in FP32. No global drag, extra
damping sweeps, damping of fixed velocities, contact-parameter change,
artificial support force, velocity reset or force-dependent parameter is added.

New case identities are:

| New identity | Original declaration reused | h | Full steps |
|---|---|---:|---:|
| d6b_n1_sphere_c4096 | n1_sphere | 1/1024 | 3072 |
| d6b_n2_capsule_c4096 | n2_capsule | 1/1024 | 3072 |
| d6b_n3_soft_c4096 | n3_soft | 1/1024 | 3072 |
| d6b_n1_fine_c4096 | n1_fine | 1/2048 | 6144 |

Each is a new, explicitly damped material. These names must appear in the
declaration, source binding, summary and output directory; do not append their
data to old D4 streams. The original 15360-step/backend matrix remains the
complete application scope, with unexecuted whole cases reported PENDING.

Unchanged inputs include the center mass 0.5 kg, four fixed corners, rigid mass
2 kg and inertia (0.25,0.25,0.25), zero gravity, initial rigid velocity -ez/16,
initial lowest-axis gap 9/128 m, radius 1/8 m, original sphere/capsule points,
compliances 1/16384 or the original soft-case 2/16384, and layer
(width=1/16 m, stiffness=128 N/m, damping=0.5 s/m, mu=0) on each facet.
The layer damping has different units and remains exactly 0.5 s/m.

## 2. A priori coefficient rationale, not a fitted equilibrium oracle

Take a deliberately simple symmetric center-loading model. Corners are
(plus/minus 1, plus/minus 1, 0), the center is (0,0,-z), z>0, and each edge
has reference length L0=sqrt(2) m and stiffness k=16384 N/m. Put the full
external load L=16 N on the free center. Its four-edge restoring force is

    ell(z) = sqrt(L0^2 + z^2)
    F(z) = 4 k (ell-L0) z/ell
         = 4 k z [1-L0/ell].

For z/L0 small, F(z) = (2k/L0^2) z^3 + higher-order terms = k z^3 + ...
Thus the leading static scale is

    z0 = (L/k)^(1/3) = (1/1024)^(1/3) m, approximately 0.10 m.

The exact derivative of this idealized force, evaluated near that scale, is

    Kt(z) = 4k [1-(L0/ell)^3],
    Kt approximately 4.8e2 N/m.

If the rigid body and free center move together, their total moving mass is
M=2.5 kg. A one-coordinate critical damping scale is then

    Ccrit = 2 sqrt(M Kt), approximately 69 N s/m.

For an instantaneous purely vertical center velocity at this symmetric
geometry, four equal axial coefficients produce the vertical damping
coefficient

    Cz(z) = 4 c z^2/(L0^2+z^2), approximately 0.020 c.

Hence a per-edge scale is ccrit = Ccrit / [4z^2/(L0^2+z^2)],
approximately 3.5e3 N s/m. Choose **4096 N s/m**, the next power of two above
that estimate, once and before execution. Its nominal one-coordinate damping
ratio is about 1.2. These rounded hand estimates explain an engineering
material choice; no root-finding, simulation, parameter scan or old trajectory
amplitude was used to choose it.

This estimate is not the full system's static solution or modal calibration.
Actual contact forces are distributed through finite-facet barycentric maps,
so some reaction can reach the fixed corners without going through the center.
The normal-layer compliance permits rigid/center relative motion. Even at the
initial vertical configuration its four-spring stiffness scale is 512 N/m,
comparable to Kt, and its pressure-dependent approach damping acts on relative
contact velocity. Replacing that two-coordinate system by mass 2.5 kg is only
a coefficient-selection approximation. In particular, damping the center
strongly need not critically damp the rigid body's contact-layer mode.
The actual finite geometry and forces remain independently checked.

## 3. Null modes and the finite physical step remain visible

At z=0, each original radial edge is horizontal. A purely vertical center
velocity has zero first-order axial strain rate, so this material contributes
no instantaneous damping in that mode. Positive c does not erase that null
mode. The exact geometry-dependent coefficient Cz above grows from zero as
the membrane deflects; it is neither a universal membrane viscosity nor a
prediction of the net impulse over a complete ordered pass.

D6-A applies one implicit axial velocity operation per edge at the unpublished
elastic candidate. The free center has stored inverse mass 2; each other
endpoint is fixed. Ignoring only normal-rounding for this explanation,

    h c (wi+wj) = 8 at h=1/1024, and 4 at h=1/2048,
    vi_new = [I-alpha n n^T] vi_old,
    alpha = 8/9 (coarse), or 4/5 (fine).

The actual operation uses the stored g2=dot(n,n) in its denominator and the
full current endpoint velocities. Four edge normals generally differ; the
four rank-one operations are ordered and do not commute. The normal before a
later row and its incoming velocity are the actual current ones. Therefore
h*Cz*v is not the net material impulse and the nominal critical-damping scale
does not prove a finite-h critical response. The half-step case is an
independently required physical discretization check, not permission to shrink
h until the coarse case passes. No complete-pass repetition is added.

Each individual ideal update dissipates kinetic energy for positive c; the
accepted FP32 result still needs the original per-row signed arithmetic audit.
That local property does not prove global stability, contact resolution,
steady-window support or recontact for the coupled simulation.

## 4. The runner must observe the two actual phases

Before any application execution, its consumer must be extended under a
separate code grant; the unchanged D4 audit cannot simply be pointed at new
dissipated velocities.

- Keep the original entry-sampled contact law and rigid force/COM-moment audit.
  Both use the actual common entry state, whose node velocity now comes from
  the previous accepted damped step.
- Feed the original elastic reference and stored-primitive
  audit_deformable_impulse with the actual candidate position and
  **damping_step.reconstructed_velocity**. Preserve its predictor, complete
  ordered position trace, lambda and elastic rounding terms. Do not feed
  final dissipated velocity to that reconstruction identity.
- Independently recompute one complete ordered damping pass from this stored
  reconstructed velocity and candidate geometry for the same-entry comparison.
  Keep all 32 columns, original float64 parameters, stored FP32 inputs, requested
  impulse, actual velocity impulse, per-edge ideal work, actual kinetic change
  and rounding work. Apply the D6-A 2e-6 absolute /2e-5 relative phase gate and
  its per-edge arithmetic energy bounds; do not cancel errors across rows.
- Preserve material_impulse as the original elastic quantity. Save the new
  damping_impulse and damping_step separately. Total fixed reaction is
  -(h*F_fixed + I_elastic_fixed + I_damping_fixed), including every original
  ordered direction/delta-lambda contribution.
- Complete momentum accounting adds the new actual velocity impulse, with
  requested damping impulse and its FP32 difference separate. For angular
  momentum, split the old entry-to-elastic-candidate change from the velocity
  update at the fixed candidate positions: the latter uses candidate-position
  cross actual damping impulse. Do not move this impulse to entry positions
  or silently reuse the old final-velocity angular formula.
- The old contact work, layer potential, contact dissipation and external work
  retain their original definitions. Material damping work is a second,
  separately saved phase. It is not credit that may hide a contact or
  fixed-reaction defect. Final state and all receipts publish only after the
  existing joint candidate checks.

The continuous FP64 reference starts from its **own previous damped state**.
It computes its own finite geometry and contact force, original elastic
predictor/ordered projection/reconstruction, one independent ordered damping
pass, and its own rigid step with the unchanged external schedule. Its
geometric and material continuation never takes the real next state, real
contact force, real damping trace or real sweep count as its next input.
Same-entry audits and continuous-reference audits are distinct records.

## 5. Original application gates and evidence

The external schedule is exactly -16 ez N for [0,1), +16 ez N for [1,9/8),
then -16 ez N for [9/8,3), with zero external torque. All four trajectories
last 3 s with actual retained velocities and no reset. The positive load phase
is active outward lifting and its external work must remain recorded.

The two original windows [0.75,1) and [2.75,3) both require mean rigid normal
support in [12.8,19.2] N, rigid normal-speed RMS <=0.1 m/s and center
normal-speed RMS <=0.1 m/s. Peak force <=128 N; every original independent
geometry check retains gap >=-radius/2 and positive original oriented side.
Initial outside, entry, release and recontact each retain the original 1/128 s
duration and event ordering. No event is inferred from the commanded schedule.

All original local distance/witness/normal gates, layer/force/COM-moment gates,
elastic short-step gates and stored arithmetic checks remain. Continuous
position <=2e-3 m, velocity <=1e-2 m/s and orientation <=2e-3 rad remain.
Changing the reference to include the declared new material is necessary;
changing a tolerance, origin, mass, load, h or observation window is not.

Preserve the original qualification split. A STRICT run stops and records an
original short-state failure. If the parent predeclares functional diagnosis
using the existing diagnostic semantics, every such failure remains a recorded
strict failure; even complete physical-window success can only be
DIAGNOSTIC_COMPLETED, never STRICT PASS. This protocol does not authorize an
automatic second run after a failed first run.

Stream each attempt and returned raw receipt before assertions, complete two
phase arrays, all original query/contact fields, fixed reactions, independent
references and signed defects. Include the new c vector and edge IDs in the
immutable declaration/checkpoint binding. Keep full step/control/close counts,
all failure prefixes and logical gzip hash; avoid nested cumulative histories.
Any plot or summary must carry the original qualifier and the new case identity.

## 6. Conditional execution order and reuse limits

1. First require accepted D6-A kernel, D1 and short sphere/capsule D3 integration.
   Then freeze the application consumer and independent reference implementation
   against this document. No application run is evidence for missing module gates.
2. Run the complete coarse new N1 sphere on interpret first, with qualification
   mode fixed before launch. A real law/publication/energy/geometry error stops
   the application. A complete steady-window failure is retained as a model/
   application failure; no second coefficient is tried in this batch.
3. If the declared first gate succeeds and time permits, execute the identical
   coarse N1 CUDA case and the three remaining fixed identities on both modes.
   Their data/geometry/reference branches are retained; a successful CLI and
   its pytest wrapper are not two different cases needing reruns.
4. The normal capsule retains its original longer axis and real rotational
   inertia. Symmetry makes the same c a reusable proposal, not a sphere-only
   proof. The soft case keeps c=4096 while compliance doubles; equilibrium
   deflection, tangent stiffness, and modal damping ratio consequently change.
   Do not recalibrate it after inspecting the sphere. The fine case also keeps
   c=4096, rather than preserving an artificial h*c ratio.
5. Each unexecuted complete case stays PENDING. A failed prerequisite may lead
   the parent to reserve the half-step or other cases for a later independent
   diagnosis, but no silent reduced matrix or shortened trajectory qualifies
   the four-case application. No performance/Newton-superiority claim follows.

Only one rational c choice is made here. If this objective axial material still
does not meet the original support windows, retain that result and investigate
the remaining contact-layer/relative-motion modes under a new scope. Do not add
isotropic drag, extra rows, new damping order or a different parameter during
this application's first qualification.
