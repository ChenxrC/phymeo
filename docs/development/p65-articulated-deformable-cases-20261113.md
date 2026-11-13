# M-B2 fixed articulated/deformable cases

Status: **PREDECLARED / UNEXECUTED**. This is the cases-first checkpoint for the [implementation contract](p65-articulated-deformable-implementation-20261113.md). Only source authoring is authorized here; the parent owns all execution. No output has selected these inputs or tolerances.

## Shared inputs

Both C and D run eight continuous steps at `h=1/1024` in logical worlds 7 and 31, with anchors `(8,-4,2)` and `(-3,5,-1)`. Registration order is reversed; identities remain canonical. The original M-B1 sheet is retained: points `(0,0,0),(1,0,0),(1,1,0),(0,1,0)` plus anchor; IDs `101,113,127,139`; facets `503,509` with index triples `(0,1,2),(0,2,3)`; surface 23/body 101/shape 0. Ordered distance edges `(0,1),(1,2),(2,3),(3,0),(0,2)`, rest lengths `(1,1,1,1,sqrt(2))`, compliance `1/16384`, inverse masses `(2,0,2,2)`, zero initial node velocities/gravity. Settings retain one substep, 200 position iterations and tolerance `2e-6`.

Each declared facet uses the existing NormalLayer, width `1/16`, stiffness `128`, approach damping `.5 s/m`, friction zero. Radius is `1/8`. This is the original entry-sampled, per-facet law, not an area pressure or implicit material/contact solve. All initial contacts have normal gap `width/2`. A near row without a usable normal remains an attempt rejection.

Each world has participant `(world,19)`. Root mass is 2, COM `(1/32,-1/64,1/128)`, principal inertia `(.5,.625,.75)` and principal frame XYZW `(.5,.5,.5,.5)`. Initial root orientation is identity. Initial root COM/world velocity is `(1/128,-1/256,-1/16)`, world angular velocity `(1/128,-1/256,1/256)`. All scalar joint positions are zero. Gravity is zero.

## C: floating R-P chain, terminal capsule

Root origin is anchor plus `(.25,.125,3/16-1/32) = (.25,.125,5/32)`. Child parents `(-1,0)`, kinds `(revolute,prismatic)`, offsets `(.25,0,0),(.25,0,0)`, identity orientations, axes `(0,1,0),(0,0,1)`. Child masses `(1,.75)`, COMs `(1/32,-1/64,1/64),(-1/64,1/32,-1/64)` and principal inertias `(.25,.375,.5),(.125,.1875,.25)`.

Body IDs are `(202,211,223)`. Only group 2 has a collider: capsule centerline local endpoints `(0,0,0),(1/16,1/16,1/8)`. Its lower end projects to `(.75,.125)` with facet-503 barycentric weights `(1/4,5/8,1/8)` initially. Root and group 1 retain zero contact wrench rows while their coupled motion and all FK/J/M rows are checked. Initial child rates are `(1/32,-1/64)`.

## D: branching R/P/R, two loaded links and one unloaded link

Root origin is anchor plus `(.5,.5,5/32)`. Child parents `(-1,-1,-1)`, kinds `(revolute,prismatic,revolute)`, offsets `(.25,-.375,0),(-.25,.375,0),(.125,.125,.5)`, identity orientations, axes `(0,1,0),(0,0,1),(1,0,0)`. First two child mass/COM/inertia rows equal C; third mass `.5`, COM `(1/64,-1/64,1/32)`, inertia `(.125,.1875,.25)`.

Body IDs `(202,211,223,239)`. Group 1 sphere has local point `(0,0,0)`; group 2 capsule has local endpoints `(0,0,0),(1/16,0,1/8)`. Their lower projections are `(.75,.125)` and `(.25,.875)` on distinct facets. Root and group 3 have no collider and no contact wrench, but full source identity, FK and motion remain required. Initial child rates `(1/32,-1/64,1/64)`.

## Fixed controls and explicit external loads

All scalar gains are `kp=1/4`, `kd=1/8`, effort limit 4. Steps 0–3 use target positions `(1/32,-1/64,1/64)`, target rates `(1/64,-1/64,1/128)`, feedforward `(1/16,-1/32,1/64)`, truncated to the original number of scalar joints. Steps 4–7 negate those three columns. Explicit external joint effort is **zero**, as required by FloatingRobotCommand; nonzero joint actuation belongs to its bounded command.

The held world force at root COM is `(1/64,-1/128,0)`, with root COM torque `(0,0,1/128)`. Group 2 receives force `(0,1/64,0)` and torque `(1/128,0,0)`. Other body external rows are zero. External loads remain fixed during the command change. These are recorded separately from contact rows and included once in generalized work and the independent mass equation.

## Independent checks and fixed gates

The standalone consumer copies the already established independent homogeneous time-jet reference from `tests/reference_newton/robot/resident_floating_reference.py`: analytic first/second transform derivatives, every unit generalized-velocity column for full J, complete M/bias/COM/gyro terms, an exhaustive bounded-PD active-face solve, and an independent quaternion/COM drift. No product FK, mass, integration, collision or material helper generates expected values. The finite segment–triangle Fraction oracle and ordered FP64 material recurrence are copied from the qualified M-B1 consumer. Each continuous reference advances its own node and full tree state, without resynchronizing from published states.

Retain M-B1 mechanics absolute `5e-6`, relative `5e-5`; same-entry material `2e-5`; continuous state `2e-4`. The prior floating-tree mass gate is `2e-5`. Geometry uses original local L only: distance `1e-4 L`, witness `2e-4 L`, normal `2e-3`. All raw rows and fields are saved before assertions. Contact point velocity uses the actual nonroot group; full perbody force/COM torque, generalized J-transpose load, PD effort, mass equation, node fixed material reaction and work are independently checked. Unloaded bodies are not removed.

Source-only and installed runs share this standalone checker. Additional bounded nodes cover canonical/reversed registration, mixed old singlebody and new articulated worlds, true complete-body memory exact/one-less, bad group/body/source, forbidden same-world extra participant, zero response sites, late query/publication faults, no partial physical commit, real retry/clone, checkpoint/partial reset/restore and successors. New schemas and every no-collider body address must be present. Unknown resource release remains sticky; no unsafe second free. No solver thresholds, capacity or fixture changes are allowed after the first result; a failure remains evidence for the parent to classify.