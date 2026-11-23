# Secondary-only motion: retained indirect-light control

Private consumer of the already verified **full-distance rigid4** package. It changes no shader, sampler, guide, history rule or runtime. It rejects a finite-near `local_diffuse` package. This experiment is authored, not yet executed; the first complete result may establish measurable old indirect signal, no registered old signal, or insufficient secondary signal.

## Frozen geometry and phases

96x96, focal 160px, eye (0,0,1), optical axis -Z, near .01/far5. White roughness1/metal0 receiver spans x/y ±.35 at z0. The fixed red wall is x=.50, y ±.50, z=.02...90. Black roughness1 shutter is x=.40, y ±.60, z=.01...95; only its node moves to y+2 at presentation8. Primary visible receiver, material, geometry, light, exposure and camera never move. Directional light travels (+1,0,-1)/sqrt2 with RGB illuminance3; environment and panels are absent. The shutter changes secondary visibility and red-wall illumination, while receiver direct illumination is unchanged.

Independent host ray/finite-rectangle intersection checks all 9216 pixel centers in both states. Actual device full-frame ID/VALID/depth must remain byte-identical, all ID1/VALID1/depth1. Actual history lighting guide (current primary direct/specular and rho) must remain byte-identical across every reference, motion and cold capture. Thus direct-light shadow changes cannot be mislabeled as retained incoming. Source scene/frames/cameras and every presentation/predecessor are saved and SHA-bound.

Frames0..7 establish closed-state history; frames8..15 hold the opened state. The first four open frames are the primary question, with the remaining four reported for decay. Two independent owner modes (`reuse_history=False/True`) have exactly the same base seeds0/1, successful capture sequence and frames. They must publish identical *current* incoming in every frame. Age≤4 is measured from the actual 48-byte GPU history records, not inferred from tick count.

For each seed and first four open frames, a new history owner starts at that exact frame with the recorded **effective seed**. Its current incoming, total raw-linear and RGB must equal the corresponding current-mode frame; ROI age must1. This rules out substituting a differently sampled cold image for the real changed phase.

## Signal, reference and decision

Fixed ROI [32,32,64,64), never picked from observed brightness. Every ROI pixel must have actual current traced support. The reference comprises closed/open captures at 32 independent base seeds100..131 per phase; report the means, seed standard error and two disjoint 16-seed-group differences. These are strong finite Monte Carlo averages of the **same compact estimator**, not unbiased path tracing or a correction for its spatial reconstruction bias. Do not call ROI pixels independent observations.

The open-minus-closed red excess `R-(G+B)/2` must be at least .01 scene-linear and exceed five times combined reference seed-mean standard error. Otherwise the result is `insufficient_secondary_signal` regardless of any visible small difference.

Let d=referenceClosed-referenceOpen. For a mean color x, its signed old-phase fraction is dot(x-referenceOpen,d)/dot(d,d). The paired additional old-phase fraction compares history against its **same-seed current**, so ordinary current-estimator error is reported separately. Claim `measured_stale_indirect` only if that paired mean is at least .10, exceeds three paired-seed standard errors, is positive in both motion seeds, and the history mean itself retains a positive old fraction above max(.10,3 reference-projection SE). Two motion seeds have limited statistical power; this is a registered diagnostic rule, not a universal confidence theorem. Negative and insufficient outcomes are preserved. No parameter fitting, ROI reselection or threshold relaxation.

All incoming/total measurements are genuine GPU FP32 scene-linear readbacks. RGB8 is additionally saved for viewing; no inverse-sRGB approximation substitutes for raw measurements. The summary reports each open frame separately, including current/history error against finite reference, signed residual, uncertainty and age. It does not call ordinary Monte Carlo noise a ghost, nor does a negative result prove arbitrary secondary-motion safety.

## Exact serial-owner entry

Only the execution owner runs these commands. Put the parent checkout's current `rendering/examples` in `--examples`; the driver activates the selected sealed private runtime before importing renderer modules. Output must be new.

```powershell
& $python -B -m unittest discover -s "$authorExamples/preview_secondary_motion" -p test_contract.py -v
& $python -B "$authorExamples/preview_secondary_motion/study.py" --prepared "$parent/artifacts/rendering/nov24/rigid4" --examples "$parent/rendering/examples" --device-index 1 --output "$parent/artifacts/rendering/nov24/secondary-motion-01"
```

Total budget:64 static-reference captures,64 paired continuous captures,8 cold phase captures = **136 actual 96x96 RGB+three-sensor captures**. No warmups masquerade as samples, no physics stepping and no adaptive rerun. Capture time excludes debug readbacks/PNG/zlib. The result owns PNG plus losslessly compressed original planes/current/history/total with uncompressed and compressed SHA; reports retain actual diagnostics, source/DLL/SPV identity, full inputs, continuous bindings and real allocation/validation teardown. Failure preserves prior files and exits nonzero; a valid insufficient/negative result exits0 with its honest outcome. There is no default installation or public quality promise.