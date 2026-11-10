# mujoco-tyx：首批独立设备迁移

**CODE_READY_TESTS_PENDING / AUTHORING_ONLY，2026-11-10。** 本文对应作者源码检查点 `1a45fb87`，以及下列新增文件。作者没有 import、collection、pytest、数值 probe、lint、formatter、编译或安装；实际 interpret/CUDA 由父唯一串行执行槽验证后另记。首批是可验证的有限算法域，**不是 G1 或完整 MuJoCo 已迁移**，没有性能对标结论。

用户新选择：Phymeo MuJoCo 的默认 CUDA implementation 由父共享 dispatch 切向 `tyx`；只有显式 `implementation='warp'` 才进入原 MuJoCo-Warp。CPU native 模式继续独立存在。未知或未迁移的源模型返回明确能力问题，不降级到 Warp 或 CPU。

## 已写代码和复用矩阵

| 层 | 复用/迁移 | 新代码位置与实际边界 |
|---|---|---|
| 源声明 | M2 的原 `MjSpec.compile`、拥有的 MJB 和 `fresh_native_model()` | [runtime](../../src/phymeo/backends/_mujoco_tyx_runtime.py) 直接消费原 native 模型；不借 M2 的 primitive asset 子集替代模型，不丢 world geometry。 |
| 所有权/协议 | 原 MuJoCoRuntime 的 native qpos/qvel、Control、初始化和显式观察协议 | 独立 owner、稀疏 world、held control、请求值与实际 FP32 `stored_control`、tick/time、reset/clone/close、完整 bool mask。共享 dispatch、exports、pyproject 归父所有。 |
| 常量结构解码 | CPU MuJoCo 编译产品中的完整 M、body COM Jacobian、bias 和 source 参数，只在构造时读取 | [_model.py](../../src/phymeo/backends/mujoco_tyx/_model.py)：全 prismatic 结构使这些量与 q 无关，保完整 dense 跨关节/祖先耦合及 armature；没有把每个 joint 当独立质量。 |
| 接触几何 | MuJoCo-Warp 0.0.2 的 plane/sphere 距离、接触中点 | [_kernels.py](../../src/phymeo/backends/mujoco_tyx/_kernels.py)：实际 TYX 算子在当前设备 q 上生成 site；保 margin 与 gap 的生成/激活差别，零力 site 也保留和计容量。 |
| 软法向方程 | Warp `constraint._update_efc_row` 的 solref/solimp、阻抗、D 与 aref | 同一 MuJoCo condim=1 软约束方程，单行凸 QP 闭式求解；没有换成原 Phymeo 的硬接触投影或自定义力—间隙弹簧。 |
| 参数准入 | priority、solmix、正参数/直接参数两种 solref | 修正上游 Warp 0.0.2 的一处差异：priority 不等时按 CPU MuJoCo 3.3.7 直接复制高 priority 的 solref/solimp；仅同 priority 才 mix/min。原差异与修改留在 NOTICE。 |
| 积分 | Warp Euler 的顺序和隐式关节阻尼；局部角速度 quaternion 积分 | 实际 TYX 求 `v += h (M+hD)^-1 M qacc` 后推进 q。浮根保 native WXYZ 和 body-local angular rate，真实 world torque 用 R.T 映射；不写入参考轨迹。 |
| 设备 | 既有 TYX 公共 Runtime/cuda_array/copy_from/to_numpy/use_runtime/sync 资源协议 | [_device.py](../../src/phymeo/backends/mujoco_tyx/_device.py)：自己的持久双状态 bank、kernel 和资源生命周期。既不 import Warp，也不包装 Warp launch。 |

## 首域与显式拒绝

`MuJoCoTyxRuntime(compilation, *, world_keys=(0,), initial_states=None, mode='cuda', implementation=None, contact_capacity=64, constraint_capacity=256, cache_root=None)`；`mode='interpret'` 为同算子的核对模式。类继承公共 MuJoCoRuntime，供父 `__new__` 直接分派。控制与 snapshot 保持 native 坐标协议，`step(h)` 只接源模型 timestep，不隐含子步。

- 关节树：1–8 个全 prismatic 自由度，可带固定结构、非正交轴、分支共享质量、偏心质量和初始 ref。支持常量完整 M、scalar armature/damping/stiffness、无激活状态的 direct/affine actuator、gear、原控制/actuator force clamp 及 disabled actuator group。
- 浮根：单个无子树、COM 位于 body 原点、物理惯量各轴相同的刚体；包括 native 6D 速度、任意合法初始姿态、真实外力和 world body torque。球形碰撞几何必须居中，因而本首域接触不产生被遗漏的偏心接触力矩。
- 接触：原过滤规则下至多一个可能的 static plane/sphere pair，可能接触的合成 `condim` 必须为 1。没有可能接触 pair 的模型不因默认 `condim=3` 被误拒。使用源 margin/gap、priority/solmix、solref/solimp 和 inverse-weight approximation；接触时按完整 M 的单行法向 QP 解算。
- 限位：只支持 `strict-interior-reject-on-exit`。每个初态和实际设备入步/候选 q 都必须严格处于 `range + margin` 内域；触及 native limit 的作用域时整批不发布，不能称 joint-stop 已移植。对本首域线性 prismatic 坐标，Euler 的步内直线段也留在该区间。
- 数值选择：保留 source solver/iterations 元数据，同时明确数值算法为 `single-normal-direct-qp`，不冒称完整原 Newton/CG/PGS 实现。`iterations=0`、非 Euler、noslip、非默认 enable/disable flags 等不在首域。源迭代器配置的进一步细分准入仍应经过父 review 与验证。
- 尚不支持：revolute/ball 或混合浮根树、偏心/各向异性自由刚体、多个可能 contact pair、任意凸体组合、摩擦与切向 cone、equality/tendon/flex/plugin/mocap/userdata、activation actuator、joint-wide force clamp、原关节软限位作用域和环境流体。构造时返回 `MuJoCoTyxUnsupported.issues`，不会静默忽略。

能力字段 `native_model=False`、明确的 `domain`、`g1_supported=False` 与 `implementation='tyx'` 是范围声明；默认选 TYX 不表示所有 MuJoCo 模型都已受支持。

## 力、状态和失败时刻

每次 device step 从原 bank 读取实际 q/v，完成施力、常量 M 求解、软约束、积分和状态检查后只写 candidate bank。所有 world 的状态、site/constraint 容量、readback 和 receipt 都准备成功后，才一次交换 active bank。容量超限、非有限数、限位域或晚读取失败都抛完整全 False 的 `PhysicsStepFailure`，原发布状态和 tick 保留；不把“GPU 做过未接受候选工作”当作物理接受。

`last_step` 记录真实 TYX 输入态 q/v、原请求控制和 FP32 存储控制、实际 qacc 与阻尼积分 acceleration、actuator/generalized constraint force、site/双边法向反作用、源/实际 solver 和 input/output time。Euler 接触读取的明确时刻是 `input_time`，phase 是 `tyx-euler-input-constraint-evaluation`。`snapshot()` 的端点 body/geometry/contact 观察在 detached CPU `mj_forward` 上重算，并明确标 `endpoint-cpu-forward-recomputed`；它不改原发布数据或下一步，并非上一步 TYX 力/冲量。step 内没有 CPU `mj_step`、CPU `mj_forward` 或 Warp。

模型和控制先检查 FP32 正常表示域；未使用参数调谐、质量缩放或参考动作修状态。这里接受 FP32 工程误差，后续验证应分别记录模型域、坐标精度、物理行为与性能，不用旧不同接触律的通过标签替代。

## 已作者的独立案例

[test_mujoco_tyx_runtime.py](../../tests/test_mujoco_tyx_runtime.py) 对每个实际物理案例分别参数化 interpret/CUDA。它们尚未由作者收集或运行：

- 稀疏双 world 无接触 slider：native gear、ctrl clamp、armature、implicit damping；无可能接触对时保默认 condim=3。
- 0.05 kg、4 kg prismatic 与 0.5 kg free-body：同一原 native source/controls 连续承载、卸载、分离、再进入；逐步比较 q/v、时间和真实 normal force，保存命令/状态/载荷轨迹。
- 斜 plane 与非正交两轴树、8DOF 分支/共享祖先的非对角完整 M，以及自由根真实 world body torque / native quaternion rate。
- 正/直接 solref 混合、两种 priority 方向与同 priority direct-min：独立核 native contact 参数和力，而非用 port 本身生成 oracle。
- source model 修改不污染、clone/selected reset 隔离、只读 snapshot、零力 site 的容量预算、晚 readback 失败不发布、FP32 控制溢出整批不改 held control、有限限位退出拒绝。
- 禁止 CPU mj_step / step 内 CPU forward / Warp import 的负控，以及未迁移 joint/equality/tendon/friction/solver 配置的能力错误。

最终通过数字、真实 CUDA 类型、首失败和源码外 wheel 仍由父保存；本文不给未运行结果。

## 后续扩域

1. 先移植一般旋转/浮根树的 kinematics、COM velocity、RNE/CRB、完整质量分解与施力映射，用多姿态 M/J/短步和偏心惯量 oracle 验证，再谈 Panda/G1。不能用初始化一次 M 支持旋转树。
2. 然后扩多接触生成、完整 constraint/solver 与摩擦 cone、原状态/容量事务。沿原源参数和同律多案例比较；单法向闭式解不是这些模块的替代品。
3. equality/tendon/activation/flex 等单列能力节点，原 CPU 和显式 Warp 仍可选择。性能比较在功能域和真实硬件相同后独立进行。

## 许可

[LICENSE](../../src/phymeo/backends/mujoco_tyx/LICENSE)、[NOTICE](../../src/phymeo/backends/mujoco_tyx/NOTICE) 和 [AUTHORS](../../src/phymeo/backends/mujoco_tyx/AUTHORS) 随包保留 Apache-2.0 与 Newton Developers 归属、修改说明。AUTHORS 只规范了末尾空白，姓名内容保留。父负责 wheel package-data，不能只保网页链接。

主要来源：[MuJoCo-Warp v0.0.2](https://github.com/google-deepmind/mujoco_warp/tree/v0.0.2/mujoco_warp/_src)、[CPU MuJoCo 3.3.7 contact 参数实现](https://github.com/google-deepmind/mujoco/blob/3.3.7/src/engine/engine_collision_driver.c)、[MuJoCo solver 参数](https://mujoco.readthedocs.io/en/3.3.7/modeling.html#solver-parameters)。
