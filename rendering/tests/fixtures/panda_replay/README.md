# Panda archive replay fixture

Two complete records copied, without numerical reserialization, from the real
`rotated-grip.json` archive named in `manifest.json`. The checked-in trace digest
is also pinned in `test_panda_replay_source.py`. No new physical step was run to
create this excerpt. The archived model XML is byte-identical to the source hash;
LICENSE, NOTICE and SOURCES.md accompany it. This XML supplies kinematic frames,
not visual meshes: visual assets and their licensed source are a separate input.

- Record 479, t = 0.30 s: pregrasp, each measured finger coordinate 0.038 m,
  zero recorded pad load. The recorded cube rests on the table.
- Record 2159, t = 1.35 s: holding after translation, bilateral recorded normal
  loads approximately 5.42/5.44 N, zero recorded table load. Arm coordinates and
  cube position both differ materially from the first frame.
- Cube full size: (0.046, 0.058, 0.040) m; half size: (0.023, 0.029, 0.020) m.
- Table center: (0.3443905665929414, 2.1423834928313568e-16,
  0.39963205230283927) m; full size: (0.179, 0.110, 0.016) m;
  top Z = 0.4076320523028393 m. This is the archived payload support, not an
  invented floor or a separately measured robot mounting bench.
- Panda fixed-base pose is identity, from the experiment's `load_panda()` default.
  No extra base translation or visual-mesh scaling is applied.

`q`, `cube`, `xyzw`, and `t` are post-step values. The recorded loads and
`pad_points_before_m` were evaluated at the step input (`q_before`). They are
preserved as source evidence, not relabelled as forces at the visual endpoint.
The original archive contains no link matrices: replay link poses are derived
from its exact q and the bound URDF. The unit test checks input FK against the
archive's independently saved pad-point witnesses. Visual offsets belong in
Instance.local_pose, and must not be multiplied into these link poses twice.
The two measured finger coordinates remain independent despite the URDF mimic
command declaration.

FrameKey world/epoch/revision are a documented archive replay namespace, not
recovered live physics identities: `archive:<archive SHA>`, epoch 0, revision
source index, tick source index + 1. Missing live world/CAS identities are None.
The archive digest binds external original evidence; it is not a signature.

Reproduce using PowerShell 7, with the original archive and model directory:

```powershell
./extract.ps1 -Archive <rotated-grip.json> -ModelDirectory <src/phymeo/builders/panda_assets>
```

This only parses text, selects original JSON tokens, computes digests, and copies
licensed model files. Compare trace and record digests with the manifest; never
replace the source by a newly simulated trajectory or infer a missing world ID.
