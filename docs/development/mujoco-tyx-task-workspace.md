# C2 first slice — owned MuJoCo-TYX device task workspace

Status: **CODE_READY_TESTS_PENDING / AUTHORING_ONLY / UNEXECUTED**.
This is the bounded first slice of C2 in
[the 2026-11-11 plan](../superpowers/plans/2026-11-11-newton-parity-until-0800.md),
following the independently frozen [A1 checkpoint](mujoco-tyx-checkpoint.md).
The author performed no project import, collection, numerical probe, test,
lint, formatting, build, installation or native execution. The parent retains
the sole execution slot and acceptance authority.

## What the callable increment does

`runtime.prepare_task_workspace(spec)` creates separate TYX output storage and
a separate TYX runtime on the physical runtime's backend/device. A refresh reads
the **current accepted native q/v device bank** under the source runtime lock
and evaluates one task kernel into a candidate output bank.

The output schema is explicit:

- Features `x` are `q_indices` followed by `v_indices`, preserving original
  native coordinate order, units and free-root conventions. Duplicate indices
  are allowed. There is no implicit FK, quaternion repair, projected gravity,
  reference motion lookup or model-specific controller.
- Observation: `scale * x + bias`.
- Reward: `-sum(weight * (x-target)^2)`, with nonnegative weights.
- Terminated: any raw native feature lies outside its inclusive lower/upper
  bounds. Bounds apply to `x`, not the scaled observation.

The reward is a declared **task objective**, not physical energy or an energy
acceptance gate. Termination is a task condition, not a collision verdict.
Zero-weight terms are skipped, so irrelevant large differences do not cause
`0 * infinity`. Any nonfinite feature, observation or accumulated reward
rejects the candidate task frame. There is no output clamp or tolerance repair.

The immutable spec owns FP32 parameter bytes and hashes its ordered native
indices and exact stored values. It admits 1..256 features and at most 256
worlds. Native indices are checked against the original model before a task
runtime/output allocation. Defaults are unit scale, zero bias/target/weight,
and finite FP32 extreme bounds. The actual stored parameters are public
read-only arrays.

The refresh path has **no snapshot(), CPU forward, CPU q/v staging or policy
observation HOST copy**. In CUDA mode it reads only a small per-world i32 status
vector to validate task publication. Existing physics still performs its
original complete HOST state/dynamics/J/EFC/KKT validation and publication;
this change does not reduce that readback. HOST action packing and original/
stored control receipt binding remain unchanged.

## Public calls

```python
from phymeo.backends.mujoco_task_workspace import MuJoCoTyxTaskSpec

spec = MuJoCoTyxTaskSpec(
    q_indices=(0,), v_indices=(0,),
    scale=(1.0, 1.0), bias=(0.0, 0.0),
    target=(0.0, 0.0), weight=(1.0, 0.1),
    lower=(-10.0, -20.0), upper=(10.0, 20.0),
)
workspace = runtime.prepare_task_workspace(spec)
frame = workspace.refresh(expected_revision=runtime.checkpoint_revision)

# Caller-owned outputs: observation f32[W, O], reward f32[W], terminated i32[W].
workspace.copy_to(frame, observation, reward, terminated, stream=consumer_stream)

# Explicit diagnostic HOST copies, not used by refresh or copy_to:
host_observation, host_reward, host_terminated = workspace.readback(frame)

# Optional copy-only CUDA Torch export; explicitly rejects CPU-only Torch.
observation_tensor, reward_tensor, terminated_tensor = workspace.to_torch(frame)
workspace.close()
```

For interpret mode, caller output arrays are writable contiguous NumPy arrays
and no device stream is accepted. CUDA outputs are same-device TYX CudaArray
objects, including compatible separately owned public interop views. A CUDA
consumer stream is optional. All three destinations, dtype, shape, liveness,
device and mutual overlap are preflighted **before any output is written**.
Output aliases of physical device arrays or workspace arrays are rejected.

`copy_to` synchronizes its explicit consumer stream before returning; the
default-stream path synchronizes the task runtime. It publishes independent
caller buffers, never a writable view into physical or workspace storage.
A later device-copy error can leave caller destinations partially written;
it does not change physics or the already-published task frame. There is no
claim of a three-array external-memory transaction across a device fault.

`to_torch` uses TYX's public `to_torch(copy=True, stream=producer)` and waits
for the Torch consumer stream to finish the D2D copy before returning. Tensors
are Torch-owned and can outlive the workspace. No writable physical bank is
exported. CPU-only Torch raises explicitly; there is no HOST staging fallback.

## Owner, revision, stream and failure rules

A frame contains immutable source/spec/layout/owner/world/backend/device
binding, an output epoch, A1 source CAS revision, and accepted per-world
tick/time metadata. Source binding uses XML/archive/native-MJB digests.
The native and output dimensions are explicit. Frame identity must be the
workspace's current published frame, so a copied/forged dataclass or a frame
from another workspace is rejected.

Any successful source control update, physical step, reset or checkpoint
restore changes A1's monotonic revision and invalidates old frames. Rewinding
to the same tick with A1 restore does not revive an old frame. The workspace
can refresh against the new accepted device owner because its output runtime
and allocations have an independent lifetime.

The source seam keeps the parent runtime lock throughout producer synchronization,
task evaluation, status validation and publication. Task output readers take
the same source/workspace lock order. A successful refresh increments the
workspace epoch and publishes one of two output banks. A failed refresh at the
same source revision preserves the prior successful frame. If source physics
has already advanced, the old frame remains stale even when refresh fails.

`MuJoCoTyxTaskFailure` declares `physics_accepted=True`,
`physics_mutated=False`, and `workspace_published=False`: the source state
was already committed and this task call cannot roll it back. These flags do
not claim that refresh itself performs a physical step. A higher-level Env
must account for a post-physics task failure explicitly.

The task runtime uses a dedicated CUDA producer stream. Source physics is
synchronized before the task reads its bank. The task stream completes before
status validation/publication, and copy/export consumers complete before
returning. This first version deliberately uses synchronous boundaries and
makes no throughput or overlap claim. Workspace close releases only its own
resources. Closing the physical runtime invalidates future workspace access
but does not make the workspace own the physics runtime.

Counters separately report task kernels, source synchronizations, CUDA status
D2H bytes, explicit diagnostic output D2H bytes, output D2D bytes, publication
and failure counts. Interpret mode has no device-transfer bytes. These counters
must not be confused with the preserved full physical diagnostic readback.

## Why this is not full GPU training or Env support

The existing ResidentVectorEnv and RslRlVecEnv remain HOST callback/tensor
bridges. PolicyRunner state and action decimation retain their previous owner.
This slice does not implement action D2D submission, reward accumulation,
episode bookkeeping, automatic reset, final_observation, truncated/horizon
semantics, recurrent-policy recovery, GPU PPO, or a replacement of those APIs.
It supplies callable device task outputs from which an opt-in device task
adapter can be built later.

Source inspection of the pinned TYX package confirms public
`tyx.torch.from_torch`, `to_torch`, `launch` and stream/context APIs.
The inspected parent training environment's `torch/version.py` reports
`2.15.0+cpu` with `cuda=None`. No installation was authorized or attempted.
A real TYX CUDA consumer is the first positive interop test. Positive CUDA
Torch tensor export remains an environment-dependent follow-up; the authored
tests verify explicit CPU-only refusal without claiming GPU Torch execution.

## Predeclared validation

`tests/test_mujoco_tyx_task_workspace.py` has **25 manually enumerated nodes**:
13 non-CUDA and 12 CUDA. Nothing has been collected or executed.

The positive matrix uses fixed hinge/slide, floating hinge/slide, and floating
primitive-contact models, two sparse worlds, original nonzero q/v, real held
controls and accepted physical steps. It independently computes expected
features/objective/bounds from the accepted observations before forbidding
HOST observation in refresh. It checks all outputs and uses a separate actual
TYX kernel to consume caller-owned device outputs.

Additional checks cover:

- snapshot(), mj_forward, physical _get and physical/output to_numpy are forbidden
  during refresh; only per-world status D2H is permitted.
- Same/default and distinct consumer streams, independent output copies, caller
  mutation leaving workspace and accepted physics unchanged.
- All-destination validation before writes: shape/dtype/liveness, closed stream,
  device mismatch metadata, physical/workspace alias, caller output overlap.
  The device mismatch injection is an admission check, not a multi-GPU run.
- Foreign and superseded frame identities, stale revision, control/step/reset/
  A1 restore, source close and workspace lifetime.
- A real task kernel followed by authored candidate-output/status failure:
  previous frame survives only at the same source revision; source physics
  remains exactly unchanged by the failed refresh.
- Real nonfinite FP32 task output rejection, inclusive raw-coordinate bounds,
  immutable spec ownership/hash, negative weights and invalid native indices.
- Explicit CPU-only Torch refusal, with no output D2D call or CPU fallback.

Task numeric comparisons use independently authored FP32 engineering gates
(observation 2e-6 relative/absolute; objective 2e-5 relative and 2e-6 absolute);
termination is exact. These are new task-output tests and do not change any
physical solver, contact or energy tolerances. Accepted q/v/time/tick are
compared exactly before and after task operations.

Parent-only suggested commands using the existing approved Python and fresh
absolute basetemp paths, after normal serial guard/environment/static work:

```text
python -m pytest tests/test_mujoco_tyx_task_workspace.py -m "not cuda" -q -x --tb=short -p no:cacheprovider --basetemp=<fresh-absolute-path>
python -m pytest tests/test_mujoco_tyx_task_workspace.py -m cuda -q -x --tb=short -p no:cacheprovider --basetemp=<different-fresh-absolute-path>
```

## Author delivery

Base A1 delivery remains `045c83dc85d4ad34de96606bf99382c7921f8cea`.
C2 product: `496a05e4`.
C2 tests: `920d624ee51751270c63c0127113cffe3406f9df`.

| File | Working SHA256 | Git blob |
|---|---|---|
| `src/phymeo/backends/_mujoco_tyx_runtime.py` | `860b67fdcdac289f2386a1e600426b070941fe59a1a3825845b0cd2b8df761b1` | `07b14552d8a8ef3f5de4fd3d425fc919778a653f` |
| `src/phymeo/backends/mujoco_task_workspace.py` | `11f7f55b6b216582f98037f1904b02162d428ac04f26cb825df40d4fa3a5de2f` | `b8b2f6cbec1fe66865e5d011686edeba45c7c726` |
| `src/phymeo/backends/mujoco_tyx/_task_kernels.py` | `de4a1807f735c9fbba06bca71b7e68914dfd232134988deea6408713a26f90c9` | `33df7c6593b5bc2ccd1a3cf59585f93fb5f179be` |
| `tests/test_mujoco_tyx_task_workspace.py` | `7ad1bafed6cad733b177c657d98747932374ed53e740b4decea7d327191beb75` | `4918f1a70101d771e29c33aa7820d86cd7bfb35d` |

Product/test author strings were read back exactly; explicit staged scope and
git diff --check passed. Runtime source changes are only public preparation and
the bounded locked accepted-source seam. Existing solver/device physics,
model admission, assets, pins, ENV/RSL files and accepted A1 capsule code were
not changed. No execution result or first-slice acceptance is claimed.
