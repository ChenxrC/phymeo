# N1 CUDA Torch policy cases and authoring handoff

Date: 2026-11-11. Status: CODE_READY_TESTS_PENDING / UNEXECUTED.

Authority: [GPU training plan](../superpowers/plans/2026-11-11-mujoco-tyx-gpu-training.md), parent baseline 95212cff. This author changed only the new example, focused test and this case table. No imports, collection, numerical probes, tests, lint, formatter, build, installation or CUDA operation were executed by the author. The tests-first commit is eaaba58c. Original backends, physical tests, CPU profile, pins and budgets are unchanged.

The parent reported an environment qualification blocker: only 2,006,888,448 B free on C:, with no other disk; the official torch 2.14.1+cu126 cp311 Windows wheel alone is 2,602,753,280 B compressed and about 4.192 GB unpacked (at least 6.794 GB combined). The known wheel SHA256 is 4cff57ddd28c8bdf8bf2fc473f149c1b5f0063b1a6bd57593288e0bf18459adb. This is parent-supplied environment evidence, not an installation performed here. No Torch/CUDA version is claimed qualified by this handoff. N1 inference and N2 PPO remain unverified until the parent resolves qualification and runs the frozen matrix.

## Frozen positive matrix

| Case | Worlds / sparse keys | h / steps | Source and native layout |
| --- | --- | --- | --- |
| hinge-slide | W1, W8, W32; key[i]=5+36i | .002 s / 8 | Existing `test_mujoco_tyx_device_controls.py` fixed hinge+slide material/geometry/actuator; nq=2,nv=2,nu=1,nbody=3 |
| support-slide | W8; key[i]=5+36i | .001 s / 8 | Existing `test_mujoco_tyx_finite_contacts.py::_support_xml("box", mass=2,h=.001)` geometry/material/motor; nq=1,nv=1,nu=1,nbody=2 |

Each row/world choice runs once on the default Torch producer stream and once on a newly owned nondefault Torch stream: eight positive tests, each eight continuous steps. Both trajectories begin from the same explicit initial data and advance independently throughout. There is no per-step state copying, reset, CPU physical stepping, PPO, policy training or hidden backend fallback.

The literal XMLs are embedded independently in the new example; no test helper is imported. The test binds their UTF-8/LF text SHA256 before runtime construction:

- hinge-slide: `8596be527027c85f5c6f4f17d64ababf5820a75771943da71e18509347cb16e7` (980 characters).
- support-slide: `bf4eb00b8d0eb00ac26be3f5fa31c2972fcde895ce5290bc5bdde60d45d11e92` (699 characters).

Compiled MJB/model and archive SHA256 are recorded by the future actual compilation. They cannot be claimed precomputed without executing the compiler. XML formatting is canonical to this example; the material and geometry numbers are reused from the original fixtures, not claimed byte-identical XML text.

### Model, initial state and control details

Hinge-slide retains two mass-1.2 bodies, COM (.02,-.01,.03), full inertia (.04,.05,.06,.001,-.001,.001), first Y hinge damping .03/armature .02, second X slide damping .03/stiffness .4/springref .04. The general actuator fixed gain 1.2, affine bias (0,-.2,-.04), ctrl bounds ±.05 and force bounds ±.08 remain. Gravity is (0,0,-9.81), Euler, 128 source iterations, tolerance 1e-10. First world q=(0,.03), v=(.015,.02); other worlds q=(0,.04), v=(.015,.08), matching the original fixed case's first/other slot rule.

Support-slide uses a truly finite static box halfsize (.24,.18,.05) and moving box halfsize (.12,.08,.06), body origin (0,0,.109), mass 2, centered full inertia (.08,.10,.12,.002,-.001,.003). Its vertical slide has armature/damping 0, original motor, condim1, solref (.02,1), solimp (.8,.9,.002,.5,2), gravity/Euler/128/tolerance 1e-10. Native q=0 is the nonzero world-height configuration; v=-.02 is the original real entry velocity. Native `plane size` is not falsely used as a finite collision boundary. This N1 control-driven eight-step segment is distinct from the already accepted long unload/reentry trajectory and does not replace or rerun it.

Task features are all native q then all v. Feature i has scale 1+i/8, bias .05, target 0, weight 1, bounds [-2,2]. Observation, quadratic reward and termination are checked against the explicit formula from detached accepted snapshots; snapshots do not feed policy inference. The inherited task comparison gate is rtol 2e-5/atol 2e-6, with exact termination mask.

The real CUDA `torch.nn.Sequential` is Linear(nq+nv,8), Tanh, Linear(8,nu+nv+6), Tanh. Every linear weight at flattened index k is ((k mod 5)-2)*.01; bias k=.01+.001k. Initialization is independent of physical outputs and subsequent weights are held fixed. No random output-dependent seed search or training is involved. Parameter/source hashes and actual devices are saved.

Native actions are separate contiguous FP32 allocations: ctrl=.01+.02*policy[:nu]; qfrc=.005+.01*policy[nu:nu+nv]; the last body's world COM wrench component i=.001*(i+1)+.01*policy[-6+i], all other bodies zero. These three channels are nonzero on the driven body and use original native ordering, force followed by torque. This is an explicit task action map, not a general actuator model or solver change.

## Comparison, public API and saved evidence

The example uses only public `MuJoCoRuntime`, `fresh_native_model`, `MuJoCoInitialState`, task workspace, `to_torch(copy=True)`, `tyx.torch.from_torch/stream_from_torch`, device submission, HOST control submission, step, capabilities, counters, snapshot and checkpoint APIs. There is no access to private physical banks. The helper checks CUDA/TYX runtime kind, three channel shapes, contiguous FP32 tensors and producer-device agreement before borrowing views; the runtime retains authoritative device/liveness/alias/raw-bit/CAS checks.

The control tensors and producer stream remain owned until `set_controls_from_device` returns. TYX borrowed views/stream wrappers are then closed; original Torch allocations and stream are not destroyed. The runtime packet owns its copies. Task outputs are independent Torch-owned copies, not borrowed physical scratch. The original complete HOST action audit and physical diagnostics are retained.

The independent reference uses the same TYX CUDA physical implementation through its public HOST-control entry and gets exactly the same CUDA forward's FP32 native action values. Every accepted step independently asserts the exact world-key/true result, tick, stored-step clock and complete recursive state/receipt equality, including all controls, contact fields, force/torque, M/J/dynamics and constraint records. Every world must move, and every support world must report actual positive device contact load during the segment. No tolerance is substituted for the complete same-input route equality.

Existing native-comparison reference gates remain documented, not relaxed: Q_ATOL=2e-4, V_ATOL=3e-3, FORCE_TOL rtol=3e-3/atol=3e-3. This N1 bridge does not rerun the complete native physics qualification or use those tolerances to disguise a route mismatch.

Each positive case creates one new JSONL without overwrite: exact XML/model/archive/script binding, initial states, task spec, complete fixed network parameters/device, actual dependency origins, each input state, exported task tensors, raw policy output, all native action bytes, actual device-control receipt, both full published states and complete step receipts, and final counters. Arrays retain dtype, shape, complete hex bytes and SHA256. Attempt records precede physics. Failure records identify action/physical/reference publication stages and retain the last complete public state; no falsely rolled-back success transition is emitted. Owner cleanup must succeed before the final PASS record is written.

Diagnostic policy/observation/action HOST copies used by this qualification are separately recorded; they are not used to produce the policy observation or device input. Runtime counters retain the required HOST raw-bit action audit, status readbacks and physical diagnostic traffic. No zero-HOST, zero-copy, GPU-only orchestration or throughput advantage is claimed.

## Frozen negative and ownership cases

There are 23 predeclared pytest nodes by manual decorator count; none has been collected or executed:

| Group | Nodes | Required observation |
| --- | ---: | --- |
| Four world/model choices × two streams | 8 | Real CUDA MLP forward, eight accepted steps each, independent complete route equality |
| CPU / FP64 / wrong shape / noncontiguous / NaN / Inf / positive subnormal / negative subnormal | 8 | Failed submission leaves complete public checkpoint bytes and revision unchanged; old accepted controls still drive an actual accepted step |
| ctrl and qfrc share storage | 1 | Both shapes valid (nu=nv=1), overlap rejected before commit |
| Closed TYX borrowed view / closed borrowed stream | 2 | Runtime refusal before publication; original Torch allocation and stream remain usable |
| Stale revision and task frame | 1 | Old input/frame rejected, then fresh actual accepted physics and export |
| Workspace close and caller action release/mutation | 1 | Exported Torch observation survives workspace close; copied action receipt and two actual continuous steps match the independent HOST route |
| Negative-zero word | 1 | Exact 0x80000000 retained in native action receipt; next step actually accepted |
| Real wrong GPU when a second device exists | 1 | Device mismatch rejected before commit; on a single GPU explicitly UNTESTED_CROSS_DEVICE, never counted as interop pass |

NaN/Inf/subnormal negative controls write exact int32 words into actual CUDA tensor storage, avoiding a floating-point arithmetic conversion before the original raw-bit gate. No fake second GPU, synthetic pass flag or dependency skip is used for the positive matrix. CUDA Torch and TYX imports are inside actual functions/tests, not mandatory at test collection; missing CUDA support fails the N1 execution gate rather than falling back.

## Parent-only execution commands, not executed here

After the parent qualifies the isolated CUDA profile and owns the sole execution slot:

```text
python -m pytest tests/training/test_mujoco_tyx_cuda_policy.py --require-tyx -q -x --tb=short -p no:cacheprovider --basetemp <fresh-absolute-path>
python examples/mujoco_tyx_cuda_policy.py --case hinge-slide --worlds 8 --stream nondefault --output <new-absolute-jsonl>
python examples/mujoco_tyx_cuda_policy.py --case support-slide --worlds 8 --stream nondefault --output <new-absolute-jsonl>
```

The CLI examples are alternative standalone entrypoints, not authorization to duplicate already-green test trajectories. The parent chooses the minimal nonduplicating validation and outside-source packaging checks. A focused lint/format result is not CUDA inference qualification. No N2 PPO execution is authorized by this handoff.
