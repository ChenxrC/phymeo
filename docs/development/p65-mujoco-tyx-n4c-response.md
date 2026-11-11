# N4c fixed saved-policy response diagnosis

Status: CODE_READY_TESTS_PENDING / UNEXECUTED / AUTHORING_SLOT_RELEASED. Parent is the only project execution owner.
Base is the accepted a18305cf plan, normally merged into the existing support
worktree without reset or lost work. Deadline is2026-11-12 07:20+08; this slice
contains no new training and does not change the N4a learning gates.

## Frozen interface and ownership

Only four NEW paths:

- src/phymeo/training/policy_response_diagnostics.py: bounded saved-model
  extraction, fixed native observation grid, actual TYX feature generation,
  pinned RSL deterministic inference and descriptive response decomposition.
- examples/diagnose_mujoco_tyx_policy_response.py: all nine fixed identities,
  original three summary anchors, fresh atomic output and failure evidence.
- tests/training/test_policy_response_diagnostics.py: independent synthetic
  evidence/response arithmetic and actual CUDA feature/MLP tests.
- This report.

The module's callable entry points are read_policy_pair(directory,binding,
expected_task,expected_seed), fixed_grid(protocol,entry), evaluate_pair(pair,
device='cuda:0'), and response_components(raw_action,mapped_ctrl,grid).
The CLI takes --evidence-root, --output and --device. It predeclares all nine
identities before opening a stream; it never selects a seed or grid by output.
No old source, test, task configuration, export or dependency is modified.

## Provenance and precision

Original initial_policy contains full actor and critic state_dict. Final
final_policy_and_adam contains both plus optimizer and next_iteration. The
original parameter hash covers sorted actor+critic tensor names/dtypes/shapes/
bytes; it is checked before loading either actor. Initial state is not inferred
from a seed. Normalizer state is included in strict state_dict loading. The
actual frozen MLP[32,32]/tanh has obs_normalization=False and Identity; unknown
network/distribution classes and absent/extra/layout-changed tensors are refused.
No class path from evidence is imported dynamically, and no pickle is accepted.

The N4b bounded data decoder/stream reader/hash are reused without editing them.
Every original stream is read sequentially once for this new extraction and
bound by its full compressed/logical hash/byte count. Only initial/final models
and the first held-out request/transition remain resident. This is not a repeat
of the N4b cost/gradient analysis. Complete summary/publication/manifest and
original dependency/config/source bindings must agree. All nine extractions
complete before the first inference is published.

Original held-out tick1 requested_action saved input_frame metadata, not a
standalone input observation tensor. Use its first actual receipt input_q/v,
prove the original held-out templates cast to FP32 match those stored q/v bits,
and pass them with original FP32 scale/bias to the unchanged TYX task kernel.
The resulting observations are newly generated from saved native state, not
claimed to be original saved observation bytes. The original requested action
is retained as a reference, not an equality gate or a sampled training action.

The low-level feature entry is _task_kernels.evaluate on completely owned arrays
and an owned TYX Runtime/stream. It does not construct a physical runtime,
MuJoCo model/data, environment, optimizer or workspace, and cannot step physics.
Features are explicitly downloaded and uploaded to Torch for this diagnostic;
there is no zero-HOST-transfer or training-throughput claim. The current feature
source hash must match the originally bound repository source. Device status,
finite observations and full resource cleanup are checked. HOST formula values
are descriptive only and cannot replace the real feature kernel.

## Fixed inputs and outputs

For each original seed, both hash-bound checkpoints use exactly this ordered
native grid:8 original held-out starts; q=target/v=0; each q_j=target_j +/-
q_scale_j/4 with all other errors/v zero; each v_j=+/-v_scale_j/4 with q=target.
Thus J has17, S13 and R21 points per checkpoint. Original centered/scaled task
observation is reused, not normalized a second time. Counterfactual grid points
are policy diagnostics, not validated new physical trajectories.

Each result saves all native FP32 q/v, labels, FP32 scale/bias and actual device
features, raw deterministic action and original scale*tanh(action) mapped ctrl.
For each input coordinate and all output joints, retain paired even response,
odd response, even-minus-target offset, and finite-difference secant per native
input unit. Zero denominators are rejected when input FP32 points collapse;
there is no hidden epsilon, new gain threshold or grid expansion.

The actual pinned RSL MLP/Gaussian distribution implementation is used with
stochastic_output=False and eval mode. No PPO code is copied or reimplemented.
Python, NumPy, Torch CPU and all CUDA RNG states are restored in finally, even
if model loading, feature launch or inference fails. This preserves random
state, not CUDA initialization, allocator/cache state or execution timing.
Strict dependency versions and every loaded model tensor layout/hash are saved in the result.
All old scalar learning outcomes remain unchanged.

J's paired restoring direction is descriptive for its uncoupled direct-force
coordinates. R's coupled response does not impose a per-joint sign criterion.
S may legitimately have nonzero equilibrium ctrl. A completed report means a
complete bound response experiment; it does not upgrade J/R learning quality.
If the fixed experiment does not distinguish the offset/feedback hypothesis,
stop it; do not enlarge the grid, choose new seeds or add training iterations.
Any subsequent single learning ablation needs a separate preregistration using
all three original seeds and the original128-update budget and gates.

## Predeclared tests

HOST tests cover the three exact grid sizes/order/centering, original-template
FP32 binding, collapsed/nonfinite/invalid metadata, known even/odd/cross response,
complete model/first-input extraction, whole-policy hashes, wrong identity,
missing/duplicate model or first-input records, stream truncation, and source/
class/dependency rejection. Synthetic fixtures are independent minimal stream
fixtures for the extractor, not fabricated successful physics trajectories.

CUDA tests use the actual pinned MLP and actual unchanged TYX feature kernel:
full initial/final state loading and finite complete output, all global RNGs
unchanged after success, strict missing/extra/wrong-layout tensor rejection, and
injected feature/inference failure followed by a successful clean retry. CUDA
availability is mandatory for this slice; optional dependency absence may skip
ordinary HOST collection but cannot count as N4c qualification. Exact manual
node count and executed parent results will be appended at handoff/acceptance.

## Authoring handoff

Tests/case contract were committed first at57d449bc; implementation/CLI at
73dd299d. Manual node count is31:25 HOST and6 actual CUDA. No node was collected
or run by this author; no import, lint/formatter, numerical probe, policy call,
original gzip read, build or install was performed. Git whitespace checks passed.
All source remains within the four NEW paths and all existing N4a/N4b files are
unchanged. The original parent source was merged normally; no reset/force/push.

The parent can run the following in its one serial slot using the qualified
Python313 CUDA training profile (the dedicated CUDA qualification must not
count skips or CPU fallback):

    pytest tests/training/test_policy_response_diagnostics.py -m "not cuda" -q -x --tb=short -p no:cacheprovider
    pytest tests/training/test_policy_response_diagnostics.py -m cuda -q -x --tb=short -p no:cacheprovider
    python examples/diagnose_mujoco_tyx_policy_response.py --evidence-root C:/Users/38347/.codex/worktrees/p65-parallel-integration/phymeo/artifacts/p65-parity-n4a-rob1d-col-reducer-20261111-01 --output NEW_OUTPUT_DIRECTORY --device cuda:0

The one CLI command uses all original nine identities and refuses an existing
output directory. All streams are fully checked before inference begins; only
then can a complete summary.json be published. Ordinary failure emits failure.json
with phase/error and no successful summary. Output arrays retain dtype/shape,
exact base64 bytes and SHA256 plus readable values. Current upstream MLP/MLP
layers/Gaussian/Identity implementation paths and hashes are also recorded;
these are current installed-source observations, not invented historical
installed-source hashes. Dependency versions must exactly equal the saved ones.

Resource/RNG boundary: the feature stage owns Runtime, stream and every device
array, synchronizes before retirement, and closes all registered resources even
on exceptions. It never borrows a physical bank. Torch receives an explicit
copied feature array; this diagnostic does not test device training throughput.
RNG restoration attempts every global owner even if another restoration fails;
any restoration error prevents successful publication. No arbitrary caller
Generator state or CUDA allocator/cache rollback is claimed.

There is one unchanged TYX feature launch per successfully evaluated pair and
two deterministic actor forwards. The finite-difference outputs are secants on
a fixed grid, not analytic derivatives or stability certificates. The eight
held-out requested actions are retained as references; possible batched GEMM
rounding differences are recorded without inventing a new equality tolerance.
S nonzero equilibrium action is legal. J/R's existing failed learning gates
remain failed regardless of successful extraction/inference.

### Frozen working bytes before parent execution

| Path | SHA256 | Git blob |
|---|---|---|
| src/phymeo/training/policy_response_diagnostics.py |304265457b18f772132a21e8f5e4960e34ac6e921a56ad230195005f6f298fbb|3a15651093c88b7877cf6144e8faf7e6db2b0cde|
| examples/diagnose_mujoco_tyx_policy_response.py |939a257e462ff79e7ef584b7701fd920d0cad93794826ef20e9f4da4edc297e7|da0a3b26e5227a6efbfdaca86371e039c455403b|
| tests/training/test_policy_response_diagnostics.py |e9b92a31c823d6732b0c1054ea5d8aa3195bf2fcbfba72129821e20fc13d17c8|6e6d18e6bf31be7ded3aaf1a6ca0cddf9271cd41|

AUTHORING_SLOT_RELEASED. Parent review/execution results must be appended with
their actual tested source; no success result is claimed in this handoff.

## Pre-execution computation-flag correction

The parent text review found that preserving only RNG did not reproduce the
original task computation configuration when a caller enabled TF32. This was
corrected before any N4c project execution. Tests-first commit 9b8afe28 and
implementation a6847fb8 modify only the existing N4c NEW test/module.

Original `examples/mujoco_tyx_learning_tasks.py::seed_all` assigns
`torch.backends.cuda.matmul.allow_tf32=False` and
`torch.backends.cudnn.allow_tf32=False`. The response context now performs these
same two assignments before constructing/loading the upstream models or running
features/inference. Each result records the actual two flags and the public
`torch.get_float32_matmul_precision()` value observed inside that context.

The installed Torch 2.12.0 source documents `highest` as the default/equivalent
of matmul TF32 disabled; `high` and `medium` both permit TF32. No separate
precision enum is imposed for inference and no new numerical acceptance limit
is added. Because restoring only the old bool can collapse a caller's `medium`
setting to `high`, the context also captures/restores the exact public precision
enum. It attempts all RNG/flag restorations in finally, verifies restored flag
values, and refuses successful publication on a restoration error. No arbitrary
other backend settings, concurrent caller mutation or allocator rollback is
claimed; evaluation still requires exclusive synchronous caller ownership.

The existing successful CUDA test now starts with caller precision `medium`
and cuDNN TF32 enabled, checks both actual actor calls see the original disabled
flags, checks result metadata and exact caller restoration. Existing injected
feature/inference failures start from `high`/cuDNN enabled and
`medium`/cuDNN disabled respectively; both check original flags inside the
failing call, restored flags/RNG afterward, and an unmodified successful retry.
The manual count remains 25 HOST + 6 CUDA = 31; all remain UNEXECUTED by this
author. Git diff whitespace checks only were performed. The earlier working
byte table is historical; current module/test byte bindings follow below.

| Path | SHA256 | Git blob |
|---|---|---|
| src/phymeo/training/policy_response_diagnostics.py | 4ddfa84fb12c3657336d607ef09f1263e8d79687fad50b4736217c913bd3cf86 | 94837d2c397a75c2bc2c3b933ceb00c6757832fe |
| tests/training/test_policy_response_diagnostics.py | 6c90e95d2e374c16eac6ca3b14b74accbf3f2b17163aec72ca78b6636787a9ae | 1f9654b537779eab89370497248aea05fee2ae21 |
| examples/diagnose_mujoco_tyx_policy_response.py | 939a257e462ff79e7ef584b7701fd920d0cad93794826ef20e9f4da4edc297e7 | da0a3b26e5227a6efbfdaca86371e039c455403b |

CODE_READY_TESTS_PENDING; AUTHORING_SLOT_RELEASED. Parent remains sole execution owner.
