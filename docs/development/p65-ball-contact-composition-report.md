# BALL-C fixed-root contact composition qualification

## Parent acceptance — 2026-11-12 06:10

**ACCEPTED within the declared fixed-root BALL contact composition.** The authoring
history below remains unchanged; its TESTS_PENDING status is historical.

Parent product/test snapshot `1ce57489` composes the existing normal/pyramidal
solvers with the existing complete BALL mechanics and scalar actuator stage.
It changes admission, dispatch and capability labels; it adds no mathematical
kernel, hidden CPU physics fallback, new iteration budget or relaxed gate.

- run284: 33 HOST/interpret passed, 19 CUDA nodes deselected, 42.03 s.
- run285: 19 CUDA passed, 33 HOST/interpret nodes deselected, 16.70 s.
- run286: six affected old no-contact/admission cases passed, 2.02 s.
- run287/288: a fresh external wheel was built and installed. run289 confirms
  all 509 product files are byte-identical between source, wheel and installation.
- run291/292: the independent installed consumer passed in interpret and CUDA,
  each with 64 actual worldsteps and all Phymeo imports under the installation.
  These include continuous contact, force/motion/activation checks, clone,
  restore/A-B, selected reset and stale-CAS refusal. Device late-failure injection
  is covered by the source C6 tests, not claimed as installed fault injection.

All original C1–C6 declarations and numerical gates were fixed before execution.
The only pre-test changes were import sorting/formatting; the old blanket contact
rejections were replaced by still-unsupported condim4 negatives. First lint/format
failures (277/278) remain recorded. Task283 stopped during source snapshot creation
because two guessed kernel filenames were absent; no Python/test process started.
The corrected complete selection ran once in284/285, with no physics failure or
fixture tuning. Final static checks281/282 passed for all nine changed Python files.

Evidence: `artifacts/p65-parity-until-0720-20261112-01`, runs277–292 and
`wheel-source-staging-02.json` / `wheel-parity-02/summary.json`. The final wheel
snapshot also includes the separately qualified control-window evaluator repair;
run290 passed its 192-worldstep installed CUDA consumer.

This adds primitive condim1/pyramidal3 contacts to fixed-root BALL/HINGE/SLIDE
trees with the declared scalar NONE/FILTEREXACT actors. Limits, dry friction,
direct BALL motors, FREE+BALL, broader mesh contact and G1 trajectory completion
remain outside this acceptance. It does not complete Newton parity.

Status: CODE_READY_TESTS_PENDING / AUTHORING_SLOT_RELEASED after final report commit.
2026-11-12. No import, collection, numerical probe, pytest, formatter, lint,
build or native execution was performed by this author. Parent owns four product
routing files and every execution. This is a test/consumer delivery, not acceptance.

## Frozen source and matrix

Exactly four NEW files:
- tests/test_mujoco_tyx_ball_contacts.py
- tests/test_mujoco_tyx_ball_contacts_lifecycle.py
- tests/installed/check_ball_contacts.py
- this report

The first two were committed before the consumer implementation (b7ef1e38);
complete first implementation is c04c213d. Existing BALL, BALL-A, activation,
normal/pyramid solvers, runtime and budgets were not edited.

The standalone consumer contains the fixed XML and independent native oracle;
it imports no repository tests/examples. The two test files load that standalone
file as a plain module. Optional native dependencies are required by execution;
a skipped node is not qualification.

| Cell | Fixed source / initial state | Required evidence |
|---|---|---|
| C1 | One ball, original m1.3 eccentric anisotropic inertia; off-axis sphere radius .1, local (.08,.03,-.18); ball WXYZ=(2/3,1/3,2/3,0); world root z=.13577777777777778, or +.2 for inactive variant | All three normal J columns nonzero, contact moment, complete M/J, loaded and separately inactive windows |
| C2 | Hinge-ball-slide native nq6/nv5; original three m1.3 inertia tensors and COM offsets; root/middle/tip z=.4/-.2/-.1; hinge z-axis, slide z-axis, finite box halfsize(.8,.8,.05), top z0; sphere offset(.06,0,0), radius.1; scalar qpos references .07 | Initial normal gap 2^-16 and slide speed -1; first step entry has no contact; second actual step entry has positive normal force; complete coupled M with offdiagonal terms and qa!=dof |
| C3 | Same H/B/S family and finite support, condim3; gap -.001, slide speed -.02; signed hinge and three-axis ball rates; mixed NONE/FILTEREXACT; early/late/disabled variants | 24 uninterrupted steps, reversal and command/load removal; native scalar force/filter/clamp, all normal/tangent rows and force/motion |
| C4 | Two sibling ball roots and one x-slide descendant each; nq10/nv8; left masses1, right1.7, distinct positive anisotropic inertias; root x=±.095, sphere offsets(0,.06,-.1), radius.1, common nonidentity Rx(.25) | Both branches genuinely move; both contact COM wrenches; complete native J and contact-coupled response |
| C5 | C3 early/late W4=(7,41,103,509) | checkpoint/A/restore/B, clone, stale CAS refusal, selected original reset, exact unselected public state plus serialized raw q/v/act/clock/mirror, two real successors |
| C6 | C1/C3 final-world status, complete readback, real prepared restore refusal; C3 input-act mismatch | No partial publication; whole checkpoint bytes/revision retained; real retry equals clone |

All continuous windows use h=.001,24 steps,W2=(7,41), both interpret and CUDA.
Original BALL-A actor declarations preserve negative/shared gears, gains, affine
bias, control/activation/force clamps and disabled ordering. Commands are +.8 for
0..7, -.75 for8..15, zero for16..23; the second world gets +.015 on nonzero
commands. Full generalized forces are signed .03*(dof+1), and the last body's
world-COM wrench is (.3,-.2,.4,.01,-.03,.02)*(1+.2*slot); both external channels
become zero at16. Zero command is not treated as zero stored activation or force.

Only body transforms and initial kinematics were newly declared for contact.
C2/C3 preserve the original HINGE_BALL_SLIDE masses, inertial tensors, COM offsets,
joint damping .12, armature .04, scalar ref .07/springref -.03/stiffness .2,
native joint order, and original numerical gates.

Hand-enumerated test identities, not a collection result:
- 7 continuous declarations ×2 backends =14.
- 14 constructor domain/capacity negatives, HOST construction =14.
- exact P16 sphere-plane and static S32 four box-plane slots, each with one real
  complete native-compared step ×2 backends =4.
- C5 two variants ×2 =4.
- Seven C6 fault declarations ×2 =14.
- wrong checkpoint model/actearly binding ×2 =2.
Total52. The installed consumer independently exercises C3-early W2:
24 continuous batches plus eight lifecycle batches including the clone call,
64 accepted worldsteps if all succeed. These counts are intended coverage,
not executed pass counts.

## C2 birth premises fixed before execution

At the initial state, hinge and ball rotations are about world z. The ball
quaternion is Rz(.25), and the sphere is centered at height .1+g0 where
g0=2^-16>0. The finite box top is exactly z0. Its horizontal halfsize .8
contains the sphere projection with a substantial margin; no infinite-plane
substitution is made. Both initial world states have slide speed -1 and ball
local angular rates(.001,-.001,.001); hinge speed .01.

The native physical mass includes .04 armature on every one of the five
coordinates. Therefore M+hD is bounded below by .04 I; this is a spectral
lower bound from the authored positive armature, not a sampled matrix estimate.

A conservative first-entry generalized RHS bound is <20 in Euclidean norm:

1. World-z hinge gravity torque is zero. The two ball descendants' horizontal
   COM lever lengths are .05 each, so their total gravity torque norm is
   at most2*1.3*9.81*.05=1.2753. The slide gravity load is1.3*9.81=12.753.
   The complete gravity generalized norm is less than14.
2. Scalar elastic plus complete viscous terms have norm below.3:
   scalar displacement from spring reference is at most .21/.10002,
   stiffness .2, slide viscous magnitude .12, all rotational rates below .012.
3. Four stateless actors produce less than2 generalized norm, including the
   direct negative gear, shared hinge scatter, bounded slide control .4,
   affine bias using native scalar q and gear-scaled velocity, and the
   second-world .015 command increment. No force/activation channel is omitted.
4. The full authored qfrc has norm below.25. The full world COM wrench has
   force norm below.65 and couple norm below.05; all five COM Jacobian column
   levers are below.5, the slide column has unit norm. Their combined generalized
   load plus qfrc is less than2.
5. Velocity bias is below1: every angular speed is below.012, all relevant
   offsets below.3, masses total3.9, and the sole unit prismatic speed generates
   transport terms bounded by2*m*omega*v. Centripetal and rotational inertia
   terms are smaller. This bound includes moving-joint transport and both
   descendant inertias; it does not assume bias zero.

Thus norm(a)<500. Euler's new slide velocity is at most -.5 and its displacement
at most -.0005 in the first1ms. The ball increment angle is below .000502;
the ball-to-sphere center lever after the scalar update is below .13. Its
rotation-induced vertical movement is below .000066. Allowing the complete
axis-direction and initial FP32 storage effects conservatively below .00015
still gives endpoint gap <2^-16-.0005+.00015<0. Hinge z rotation cannot change
height. Positive initial gap and this strict negative endpoint have a wide
margin relative to FP32 storage; they are authoring premises, not a native
execution result or a promise about 24-step acceptance.

The executable tests deliberately distinguish first **entry** (empty contact
list) and the second actual **entry** (active contact with positive force).
They do not infer birth merely from a final visual pose, add hidden steps,
reposition the object, or adjust the gap after outputs.

C1's rational quaternion gives R^T ez=(-8/9,4/9,-1/9).
With r=(.08,.03,-.18), r cross R^T ez has all three nonzero components
(.076666..., .168888..., .062222...), independently establishing the full
three-column ball lever. Its center height is .098 and lower gap -.002;
the separate inactive declaration raises the body .2.
C3's world-z initial rotations preserve its lower gap -.001 and tangent
incoming rates are nonzero. C4 shares the same initial orientation on both
branches, so their sphere center separation is .19 and overlap is .01;
neither partner is prescribed/static. These premises do not replace the
native oracle or actual load checks.

## Independent observations and original gates

Each step records full public entry, requested and stored controls, complete
same-entry native fields and an independently advancing native trajectory,
then the full accepted public/full and integration snapshots and complete raw
device receipt. CPU forward/step/step1/step2 are forbidden only around the
actual TYX physics call. Native oracle calls occur outside that guard.

Native expected fields include body pose/COM, every COM Jacobian, full physical
M, passive/bias/applied forces, smooth acceleration, full qacc, Euler M+hD
acceleration, native contact frames/positions/distances/J/wrenches, complete
EFC J/R/aref/force/type/owner, scalar forces and activation derivatives.
The independent scalar filter/control/clamp formulas are copied as source
from the accepted BALL-A standalone consumer, not imported from product math.
Contact matching uses geometry IDs, position and normal before any force check.
Both body force/torque receipts are reconstructed at their actual COMs;
complete J-transpose generalized force must equal the published constraint force.

BALL mechanics gates remain pose2e-5/3e-5,J2e-4/5e-5,M2e-4/3e-5,
bias/passive/applied3e-4/2e-4,smooth5e-4/8e-4.
New contact-force gate is3e-3/3e-3, qacc/Euler3e-3/1e-2, native continuous
q2e-4,v3e-3,time2e-6. EFC R/aref5e-4/4e-4.
Activation3e-5/3e-6 and scalar force3e-4/2e-4 remain BALL-A gates.
The original source iterations128 and max(source tolerance,2e-5) KKT test stay.
No contact force budget, inertia or source tolerance was increased after output.

The installed consumer uses only public compile/runtime/control/checkpoint APIs,
except reading the documented versioned public checkpoint bytes to compare
unselected raw array rows. It does not use private device banks. Every loaded
phymeo module must originate under --expected-install-root; script and cwd must
be outside the checkout. Gzip JSONL retains dtype/shape/raw bytes, all attempts,
outputs, native data, failure metadata and lifecycle snapshots. Primary failures
and finalization failures are preserved, with all finalization stages attempted.

## Execution handoff and boundaries

Parent should run the two new test files serially, first interpretation and then
CUDA under its qualified environment, retaining all first failures. Parent may
format/lint only after this source snapshot; author has performed only text
review, Git diff checks and exact file hashing. The installed consumer CLI is:

    python check_ball_contacts.py --mode interpret --expected-install-root ABS_SITE --output ABS_FRESH
    python check_ball_contacts.py --mode cuda --expected-install-root ABS_SITE --output ABS_FRESH

No ball motors, ball limits, free+ball, condim4/6, equality, mesh contact, training,
G1 completion, general Newton parity or performance advantage is asserted.
If a declared state cannot pass, preserve that fixed input and classify the
actual failure; do not shift geometry, retry with a different initial velocity,
extend24steps or relax numerical gates. Two nonproductive diagnosis rounds stop
this composition with its minimal reproducer.
