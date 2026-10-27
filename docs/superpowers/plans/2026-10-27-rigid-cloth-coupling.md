# 刚柔双向交互 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 交付一个有双向反作用、可重复运行和独立验收的布片/动态球场景，并完成 tyx FP32 对照。

**Architecture:** 所有权明确的混合状态和原子 step；先 NumPy FP64 联合 XPBD，再按明确顺序实现 tyx。几何、响应、调度、实验分离。

**Tech Stack:** Python >=3.11，NumPy，现有固定版本 tyx，pytest；标准库生成离线 HTML/JSON。

**Spec:** [设计和验收合同](../specs/2026-10-27-rigid-cloth-coupling-design.md)。

## Global Constraints

- 分支 `codex/rigid-cloth-coupling`，基线 `b4cbc6b`；不混入未验收的 tyx 刚体草稿。
- SI、Y-up、xyzw；NumPy FP32/FP64，tyx FP32；核心只有 NumPy 必选依赖。
- 相反作用只能施加一次；固定支撑不是封闭系统，不假报总动量守恒。
- 不复制 Newton 源码；只调用 tyx 公共 API。
- 每个外层 step 原子提交，失败恢复双方全部字段；所有数值与形状错误明确可观察。
- 本轮无截止时刻；全部验收才完成，不能以增加测试数量代替场景通过。
- 小步提交/推送，不合并 main、不发布包。现有大纲记录保留为上一轮历史。

## Review Focus

- 多粒子共享一球：串行/并行写冲突、残差滞后、相反作用重复计入（M2/M3/M6）。
- 接触从分离到穿透或从活动到释放：乘子稳定身份和每轮重检测（M2/M3）。
- 初态穿透、极端质量比、中心重合：确定的退化处理，无虚假有限/收敛（M1/M2）。
- 后半段子步失败：恢复整帧双方状态和报告，无残留设备缓存（M3/M6）。
- 固定点、外重力与单向对照：不把外界交换或非物理实验当守恒证据（M4/M7）。

## Task 1: M1 粒子/球几何

**Files:** Create `src/phymeo/geometry/particle_sphere.py`, `tests/test_particle_sphere.py`;
modify public exports only when the module is verified.

**Interfaces:**
- Consumes: `(Np,3)` / `(Nb,3)` same-dtype positions and existing `SphereShapes`.
- Produces: `particle_sphere_geometry(particle_position, rigid_position, spheres) -> ParticleSphereGeometry`;
  readonly `particle_indices`, `body_indices`, `normal`, `gap`, `degenerate` arrays for all pairs.

- [x] 写内/外/切触解析间隙、粒子优先行序、中心 +X、空集合、FP32 提升、非法输入和无别名测试；31 项先因缺少模块失败。
- [x] 实现 FP64 几何与稳定范数，几何不修改输入、不响应接触；有限输入计算溢出报 SolverFailure。另补不精确次正规距离假零间隙回归，先失败再修复；精确次正规轴向距离保留。
- [x] `python -m pytest tests/test_particle_sphere.py -q`：32 passed；真实 CUDA 全套 518 passed，Python 3.11/原生 pip 核心各 463 passed / 55 deselected；ruff 与文档检查通过。
- [x] 更新进度，提交 `feat(geometry): evaluate particle sphere contact geometry` 并推送。

## Task 2: M2 接触投影与独立单对验证

**Files:** Create `src/phymeo/constraints/particle_sphere.py`, `tests/test_coupled_contact.py`.

**Interfaces:**
- Consumes M1 geometry plus model inverse masses and stable per-pair multipliers.
- Produces `project_particle_spheres(particle_position, rigid_position, particle_inverse_mass,
  rigid_inverse_mass, spheres, multipliers) -> None`, in-place on explicitly owned candidate arrays;
  `particle_sphere_residual(...) -> float` with the spec's complementarity equation.
- Produces `project_particle_sphere_velocities(..., impulses) -> None` and separately checked velocity residual.
  The velocity function owns a persistent per-solve nonnegative normal impulse vector so release is possible.

- [ ] 先写单点/球不同质量的解析修正、法向碰撞共同速度、固定体/固定点、无碰撞、中心退化测试并观察失败。
- [ ] 测试两点共享一球、多轮新增/释放接触、固定-固定穿透拒绝、近零逆质量、非有限中间量。
- [ ] 实现位置投影、投影互补残差及速度互补求解；分别累计位置有效冲量和速度冲量，禁止重复作用。
- [ ] 验证无外力单对线动量 <=1e-10、无初始穿透碰撞动能增量 <=1e-10、法向零力矩；解析误差 <=1e-12。
- [ ] 跑新测试与核心回归，记录边界和推导，提交并推送。

## Task 3: M3 混合状态与原子联合步进

**Files:** Create `src/phymeo/core/coupled.py`, `src/phymeo/solvers/coupled_xpbd.py`,
`tests/test_coupled_solver.py`; use existing distance projection, prediction and rotation modules.

**Interfaces:**
- `CoupledState(particles: ParticleState, rigid: RigidBodyState)` owns normalized copies.
- `CoupledInputs(particles: StepInputs, rigid: RigidBodyInputs)` owns normalized copies.
- `CoupledStepResult(state: CoupledState, report: CoupledStepReport)`; report includes accepted,
  completed_substeps, iterations, distance_residual, contact_residual, velocity_residual, reason.
- `CoupledXPBDSolver(particle_model, rigid_model, constraints, spheres, *, substeps=4,
  iterations=200, tolerance=1e-7, reference_length=1.0)`;
  `step(state: CoupledState, inputs: CoupledInputs, dt: float) -> CoupledStepResult`.

- [ ] 测试拥有副本、错精度/类型/数量、空族、固定对象、无接触与原积分器一致；观察失败。
- [ ] 测试距离/接触交替时两者同时达标；多子步运动、旋转、无重复重力/冲量。
- [ ] 实现统一预测、每子步乘子初始化、距离/接触迭代、速度重建与法向投影。
- [ ] 在第 2 个子步注入预算失败和数值失败，断言整个外层输入逐元素恢复且不别名。
- [ ] 全核心回归和针对极端质量/穿透/共享体的审查；提交并推送。

## Task 4: M4 固定基准、双向对照与证据输出

**Files:** Create `examples/rigid_cloth.py`, `tests/test_rigid_cloth_scene.py`,
`src/phymeo/builders/rigid_cloth.py`, `docs/benchmarks/rigid-cloth.md`.

**Interfaces:**
- `rigid_cloth_scene(dtype=np.float64)` returns the models, state, inputs, distance constraints and spheres.
- `run_scene(*, frames=240, dt=1/120, substeps=4, mode="numpy", reaction=True)`
  is an experiment helper returning arrays and independently computed metrics, not a physical solver API.
- CLI `python examples/rigid_cloth.py --output artifacts/rigid-cloth` writes configuration,
  environment/commit metadata, NPZ trajectory and JSON metrics; rejected steps cause nonzero exit.

- [ ] 固定 spec 场景，先写接触发生、双方响应、固定点、每步接受、顶点穿透 <=2e-6 的失败测试。
- [ ] 实现单向球自由轨迹对照；球轨迹差 >=1e-3 m，布片也有接触响应，不调参制造假成功。
- [ ] 独立重复两次逐元素比较；补 h/2 与两种质量的实验，记录收敛和精度限制。
- [ ] 冻结参数和阈值，保存可再生成报告；新增核心 CI 的短 smoke；全回归后提交推送。

## Task 5: M5 tyx 刚体预测验收

**Files:** Review prototype `codex/tyx-rigid-wip` without importing its unverified branch wholesale;
integrate only corrected `src/phymeo/backends/tyx_rigid.py`, `_tyx_rigid_kernels.py`,
`tests/test_tyx_rigid.py` and necessary exports.

**Interfaces:** `TyxRigidSemiImplicitSolver` matches the existing CPU free-rigid solver contract.

- [ ] 先复现 torque/inertia 下溢后被大 dt 放大的差异；有效有限响应须正确或明确拒绝，不能假零成功。
- [ ] 审查 FP32 转换、四元数、陀螺项、空体、资源构造/释放失败与多步轨迹。
- [ ] interpreter、真实 C++、真实 CUDA 分别运行；旋转/角速度与 NumPy 参考对照。
- [ ] 修复后独立审阅、当前源码 wheel 验证、全回归；提交推送才计入交付能力。

## Task 6: M6 同场景 tyx 联合路径

**Files:** Create `src/phymeo/backends/tyx_coupled.py`, `_tyx_coupled_kernels.py`,
`tests/test_tyx_coupled.py`; reuse public tyx XPBD kernels where their scheduling is compatible.

**Interfaces:** `TyxCoupledXPBDSolver` has M3's bound model/settings and step/result contract;
supports `mode="interpret"|"cpu_cpp"|"cuda"`, FP32 only, explicit resource close.

- [ ] 先写单对解析、多个共享体、约束竞争、接触释放、子步回滚、设备异常和空族测试。
- [ ] 实现真实设备距离/接触/预测/速度更新；共享球采用确定的串行设备接触轮，避免竞态。
- [ ] 提供同着色/接触顺序的 NumPy FP32 对照，避免拿不同迭代次序当精度缺陷。
- [ ] 运行三模式；真实 CUDA 240 帧与 NumPy FP32 最大位置差 <=2e-3 m，另报 FP64 差与完整步耗时。
- [ ] 独立审阅资源/数值/调度，跑全套与包验证，提交推送。

## Task 7: M7 可视回放、交付与停止

**Files:** Create `scripts/render_rigid_cloth.py`, `tests/test_rigid_cloth_artifacts.py`;
modify README, `scripts/smoke_installed.py`,
`.github/workflows/ci.yml`, docs environment and progress.

**Interfaces:** offline renderer consumes M4 configuration/NPZ/JSON; embeds trajectory and draws it
with local HTML canvas, time scrub/pause and two comparison panels. No external service required.

- [ ] 输出来自真实轨迹的回放；检查帧/点/球数、数值有限、JSON 可解析、指标与实验一致。
- [ ] 提供 uv 与原生 pip 两套复现命令，明确 CUDA 依赖和顶点接触局限。
- [ ] 验证 Python 3.11/3.12 核心、真实 CUDA 全套、ruff、docs、当前源码 sdist/wheel 和源码外安装 smoke。
- [ ] 检查远程 Actions；未能访问则记录为未核验，不能假报绿色；GPU CI 缺凭据不阻断已有真机证据交付。
- [ ] 对整个分支进行新上下文审阅，处理实质问题；按 spec 逐项对照验收、提交推送并核对 SHA。
- [ ] 全部目标能力和证据完成后标记 goal complete，暂停续跑；交付可看结果、复现命令与限制。

## 执行台账

- M1 已完成；下一项 M2 接触投影。M2–M7 尚未完成。基线核心测试 431 passed / 55 deselected。
- 新 goal 已激活；原聊天续跑已更新到新目标，每 30 分钟可继续，不新建重复任务。
- 继续使用本聊天的干净专用 checkout 和新分支，保留前一交付分支；不额外创建会改变环境路径的 worktree。
- 具体测试和决策追加到 `docs/progress.md`；详细临时执行台账保存在本计划独立的 `.superpowers` 目录。
