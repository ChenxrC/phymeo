# I2c: sustained external load and coupled finite-area supports

2026-11-17, codex/exp, parent d18175c3. Root is the only project execution owner.
Original agents review/author without execution. Existing accepted suites are reused.

## Scope and fixed cases before execution

Exercise the I2b actual-state geometry bridge and I2a common TYX solve without
altering their equations. Add a runnable experiment, independent validation and
a plot of saved trajectories. Correct the dynamics-field documentation:
area smooth_acceleration uses M+hD, not M. No force or pose repair.

A vertical guided body has mass .5 kg, joint damping 40 Ns/m, initial lower
surface z=.002 m and v=0, gravity -9.81 m/s2. The guide carries transverse
forces/moments; this is not a free 6DOF balance test. Full zero-boundary pressure
tetrahedron, four cells and 400000 Pa interior value are identical to I2b.
The bottom cell has g=2e7 Pa/m, and all contacts stay in depth <.004 m.

- Single: original finite triangle, A=.00045 m2, K=9000 N/m, h=.001 s.
- Dual: same moving pressure body and two separate fixed rigid bodies.
  Left triangle x scaled .5 and translated -.035 m, right x translated +.035 m.
  Areas .000225/.00045 m2, K=(4500,9000), h=.001/.0005 s.
  Both patches remain strictly within the full pressure field's bottom cell.
- Every trajectory starts outside and runs continuously to .55 s.
  Held generalized external force: 0 N [0,.15), -3 N [.15,.35),
  +9 N [.35,.55). Switch by accepted tick using public set_controls; no state reset.
  Gravity is only in the compiled model, never added again to qfrc_applied.
- All three configurations run once in interpret and CUDA: six saved trajectories,
  4400 physical steps, plus short order/capacity tests.

## Independent checks

Each actual entry supplies only accepted q/v to a scalar oracle:
Meff=m+hD, vfree=(m*v+h*(fext-m*g))/Meff.
If depth>=0, F=max(0,Ksum*(depth-h*vfree)/(1+h*h*Ksum/Meff)); else F=0.
vnext=vfree+h*F/Meff. Check actual force <=.001 N+.001 relative and
velocity error <=3e-5 m/s, semi-implicit position <=2e-7 m.
Check effective smooth response, complete step force balance, action/reaction,
world-origin moments, original pair/source identities, and actual dA.

At plateau windows [.12,.15) and [.32,.35), required total support is
4.905/7.905 N. Mean force and mean depth errors <=3%; per-sample force <=5%,
max speed <=.002 m/s. Expected depths single .545/.878333 mm,
dual .363333/.585556 mm. Dual force shares are 1:2, tolerance2e-4 relative+2e-5N.
Across dual h, plateau force/depth means agree within3%.
Last .05 s: all forces zero, separation >1 mm; depth always <4 mm.

Track E=.5*m*v^2+.5*Ksum*max(depth,0)^2+m*g*z, external work fext*dq,
and dissipation h*D*vnext^2. Sum of positive per-step residuals
dE-fext*dq+h*D*vnext^2 <=2% of initial m*g*.002=.00981 J.
This admits entry-sampling discretization at contact birth and does not claim
exact energy conservation, nor physical damping from backward-Euler dissipation.

Two-patch same-entry counterexample: h=.001, q=-.003, v=-.02.
Solving each patch independently with the full common body mass overestimates
total force by factor7462/7381, and gives F_right/F_left=121/61 instead of2.
Require actual common solve to match the joint oracle and differ from naive sum
by >0.5%; reversing declared pair order preserves per-identity forces and motion.
Second pair overflowing total capacity3 must reject all6 rows atomically,
retaining accepted state/tick/revision and already accepted external load.

## Evidence and stop rules

Save each complete compact trajectory before checking its result. Save raw first
failure and diagnose one cause before a justified rerun; do not tune physical
parameters or gates after observing results. Serial fresh guarded lint/format,
then complete new interpret/HOST group and CUDA group. No old green replay.
Plot only the original six JSONs, not a fresh simulation.

No new claim of friction, CCD, free-body support, nonlocal FFT response,
automatic algorithm selection, device clipping, area observation ABI or
checkpoint/clone. Successful scope is continuous guided support and coupled
normal response through the existing reusable engine path.
