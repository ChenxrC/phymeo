# MuJoCo-Warp → TYX 的机器人与 G1 扩展计划

当前状态：**T0/T1/T2/T3a/T3b_LOCAL_VALIDATED，T4_NEXT**，2026-11-10。T0见[统一入口验收](../benchmarks/p65-unified-scene-compilation.md)，T1见[旋转树实际结果](../benchmarks/p65-mujoco-tyx-rotational-tree.md)，T2见[原生自由根树实际结果](../benchmarks/p65-mujoco-tyx-floating-tree.md)，T3a见[共同法向接触实际结果](../benchmarks/p65-mujoco-tyx-coupled-normal.md)。T3b见[有限配对与连续重入实际结果](../benchmarks/p65-mujoco-tyx-finite-contacts.md)；T4–T6仍待实施和各自验收，父保持唯一项目执行owner。下文来源研究原由独立作者在零项目执行下完成，新的验收不倒灌为当时已运行。主计划是[统一场景编译](../superpowers/plans/2026-11-10-unified-scene-compilation.md)。

## 1. 目标与真实起点

目标是把**MuJoCo 模型语义的设备求解**逐层移植到 TYX。CPU `MjSpec.compile` 可以继续作为一次性模型编译，不能在每个物理步调用 CPU `mj_step`、HOST 动力学或 Warp 求解，再把结果包装成 TYX。TYX 的解释模式与真实 CUDA launch 应分别留证；显式 Warp 模式保留自己的名称和精度声明。

已完成的T0是1–8DOF纯平移树、独立自由球及单个plane–sphere法向接触；T1进一步完成固定根hinge/slide混合树，T2完成原生自由根与0–29 scalar joints。T3a新增plane–sphere/capsule/box与sphere–sphere/capsule/box六类配对、完整J与共享M的多个condim1法向共同求解，公共策略环境和真实CUDA已验。T3b已补齐capsule–capsule/box、box–box及八条1秒连续卸载重入，原.512s胶囊恢复门失败保持；**摩擦、G1持续支撑和通用MuJoCo仍未覆盖**。本计划不把“换默认backend”本身当功能扩展，也不以一个舞蹈例子代替模块矩阵。

已有可复用资产：

- [MuJoCo 编译记录](../../src/phymeo/builders/mujoco_compiler.py)：独立 native MJB 快照、原 ID/地址、q/v 映射及明确 lowering issues。新 TYX 后端应消费原编译物字段；不能先经过会拒绝/丢失某些 native 语义的资产子集再默认声称同源完整模型。
- [现有 TYX 浮根核](../../src/phymeo/backends/_tyx_resident_floating_kernels.py)：`floating_geometry / floating_motion_energy / floating_mechanics / floating_solve / floating_finalize`，已有完整根–关节 M/J、armature/damping 与真实负载路径。它可作为算子实现、分层 oracle 和发布框架的参考；其 root COM 世界线/角速度与 XYZW 四元数约定均须同 native qpos/qvel 显式核对，不能直接按索引接线。
- [浮根数据所有权](../../src/phymeo/backends/_resident_floating_storage.py)、[接触投影](../../src/phymeo/backends/_tyx_resident_floating_contact.py)、[设备凸体查询](../../src/phymeo/backends/_tyx_convex_query_kernels.py)：可复用机制，不自动继承 MuJoCo 的 contact margin、impedance、constraint cone 或 solver 语义。
- [ResidentVectorEnv](../../src/phymeo/training/resident_env.py) 与 [PolicyRunner](../../src/phymeo/control/policy_runner.py)：已经具备拒步/复位/动作保持协议；当前 NumPy 观察接口有 HOST 往返，不是零拷贝 GPU 训练。

## 2. 固定研究版本与移植方式

本次实际阅读的是父 artifact 中的 **mujoco-warp 0.0.2**：`artifacts/p65-unified-scene-20261110-01/optional-deps/mujoco_warp/_src`，配套 MuJoCo 3.3.7。不是依据当前 main 的功能表反推旧版本能力；不改现有 pins。

0.0.2 的实际调用骨架为 `forward.py::step → forward → fwd_position / fwd_velocity / fwd_actuation / fwd_acceleration → solver.solve → integrator`。`smooth.py` 有 `kinematics / com_pos / crb / factor_m / com_vel / rne / transmission / solve_m`；`constraint.py::make_constraint` 生成 `efc` 行；`solver.py` 有 CG/Newton、摩擦状态与线搜索。不是一个“总碰撞判断函数”或一层 wrapper。

移植分类：**A 可按原公式/字段直接翻译**，仍需 TYX 编码和验证；**B 需替换 Warp 执行/算子**；**C 可借 Phymeo 机制，但物理/坐标语义须另证一致**。这里的“直接”不表示复制 Python 文件就可运行。

| 层 | 固定上游位置与关键数据 | A：可直接移植的内容 | B：必须 TYX 重写的部分 | C：Phymeo 可复用与差异 |
|---|---|---|---|---|
| 编译与 Model/Data | `types.py::Model/Data/Option`、`io.py::put_model/put_data`；body/jnt/dof/geom 地址、树遍历与 qM 布局 | 枚举、静态地址表、原字段含义、能力拒绝清单 | 从原 compiled model 打包 TYX SoA；world/slot 与 shape/constraint 索引、前后缓冲、有限 FP32 入场与溢出状态 | M2 快照及来源绑定、resident banks/epoch；不能把 native 可变 MjModel 当 immutable，也不能复用单姿态 M 作为一般旋转树模型 |
| 数学与空间算子 | `math.py`、`support.py::jac/apply_ft`；WXYZ、本地角速度、空间惯量/力 | 四元数乘积/指数积分、空间运动与力变换、lever arm、交叉积的原公式 | 原 `wp.vec/mat/spatial_vector`、`wp.func/kernel` 改为 TYX 数组/函数；验证归约次序及 FTZ | 现有 TYX quaternion/J/数值封装；Phymeo 浮根是 root-origin q + root-COM world linear/world angular v，native 是 root-origin translation/world linear + local angular，必须显式 T(q) 与加速度运输项 |
| FK 与平滑动力学 | `smooth.py::kinematics/com_pos/crb/com_vel/rne/factor_m/solve_m` | body/joint 树递推、复合刚体惯量、RNEA bias、M/armature、重力/外力投影 | 每层并行或有界每 world 递推；稀疏树遍历、dense LDL/Cholesky；不能照搬 Warp tile/launch_tiled | 现有 full M/J/COM/wrench 与 dense solve 可作首个正确版本；它是另一种装配方式，须对 native M/b/a/J 和能量逐层比较 |
| 关节与驱动 | `passive.py::passive`、`smooth.py::transmission`、`forward.py::fwd_actuation` | hinge/slide/free/ball 的运动子空间，armature、viscous damping、unit motor/gear、ctrl/force bounds | 激活状态、传动 moment、非线性 gain/bias、joint/tendon passive 等按各能力独立 kernel | Phymeo a/D 已有，但 bounded PD 是另一驱动模型；不能把 native motor ctrl 当 q-target，也不能把 frictionloss 当 viscous damping |
| 碰撞发现与接触描述 | `collision_driver.py`、`collision_primitive.py`、`collision_convex.py/GJK`；geom flags、pair/exclude、contact frame | primitive 公式、候选/过滤规则、距离/位置/法向、混合材料与 contact 参数 | 排序/scan、原子计数、紧凑化、多接触生成、GJK/EPA 与共享内存数据组织 | TYX 宽相、凸查询可用于算法或 oracle；MuJoCo contype/conaffinity 是 OR 规则，现 Phymeo AND 规则不可静默替代；查询 witness 不等于已生成/施力 contact manifold |
| 约束与摩擦 | `constraint.py::_update_efc_row/make_constraint`、`types.py::Constraint`；J、D、aref、state、force | solref/solimp、margin/gap、参考加速度、regularization、normal/limit/friction/equality 行公式 | compact row 编码、Jv/Jᵀf、完整双侧 J、joint limits、frictionloss、pyramidal/elliptic cone 与 condim 分支 | 现 HOST 速度摩擦力律和 material 功审计不是 native `efc` 模型；不能把旧法向罚力加上去就宣称原 MuJoCo 摩擦 |
| 约束求解器 | `solver.py::solve/_solve`、gradient、Hessian、CG/Newton、line search、warm start | 原目标函数、活跃状态、梯度/Hessian、收敛与 force 解码 | Warp tiles/block Cholesky/分段归约改写；先 bounded dense，再稀疏；批内收敛 mask 与确定终止 | 可借现有正定/有限检查和事务；旧 XPBD、PGS 或 penalty solver 不等价。若只实现 CG，必须拒绝/显式更改来源选项，不能忽略 `opt.solver=NEWTON` |
| 积分与时刻 | `forward.py::euler/implicit/rungekutta4/_advance`、`derivative.py` | 原生 qpos 对应的速度/激活更新；Euler 的 implicit damping、free/ball quaternion 更新 | TYX 原地/前后 buffer、速度导数与 factor solve、RK 阶段；按 integrator 独立 admission | 原 Phymeo 半隐式/drive mobility 可作参考；不得重复加 D 或混用 physical M 与 step mobility，不能把 end-point forward 力乘 h 当整步冲量 |
| 观察、batch 与训练 | `sensor.py`、`io.py::get_data_into`、`support.py::get_state/set_state` | 命名 state、contact frame force、原 timestep/控制时刻、传感器字段的真实来源 | TYX reduction、batched reset/clone、设备张量 view/stream/epoch、sensor/ray 等算子 | 可复用现有 Env/PolicyRunner/accepted mask；CPU 观察/ONNX 或 NumPy policy 仍需准确标注，真实 CUDA 求解不等于全 GPU 训练 |

**算子策略：** 第一轮允许有界 dense O(n³) 和一 world 一线程/组，不把性能优化作为正确性前提。Warp `launch_tiled/tile_cholesky/segmented sort/CUDA conditional graph` 不能仅改 import；先使用已存在且实际验证过的 TYX ndarray、loop、kernel、host 调度组成同一运算，再按 profiling 的真实瓶颈替换。源码看到的接口不当成尚未运行的 TYX 功能已可用。

## 3. 到 G1 的优先批次与依赖

| 批次 | 难度/前置 | 交付的通用能力 | 独立多案例门 | 此时仍不能声称 |
|---|---|---|---|---|
| T0 首域收口 | 中；原 support 作者进行中 | 真 TYX kernel、严格静态能力报告、snapshot/control/step/reset，纯平移 tree 与自由球/单 normal | 1/2/8 DOF 串/分支共享质量；自由球有初始角速度/真实转动惯量；分离/入层/承载；unsupported 与晚故障 | 旋转树/G1/一般接触/全 MuJoCo |
| T1 有限旋转 tree | 高；T0 的 Model/Data/协议 | fixed-root hinge+slide，每步 FK、full M、RNEA b、a/J、unit motor+a/D；先不接触 | 1 DOF 偏心摆；2 DOF 双摆非零 qdot；R/P 混合；双分支共祖；7关节非主轴 COM/局部 anchor/ref；多 q/v 与外部 off-center wrench | free-root、足地支撑、多接触 |
| T2 free-root 树与容量 | 高；T1 | native free-root translation/local omega、所有身体惯量/静态 welded body 的身份，0/7/16/29 scalar joints；根–关节完整耦合 | 自由非球惯量陀螺/无外力动量；偏心根COM；根扭矩引起关节运动及反作用；跨分支加载；最后关节/最后 body；W1/W4 reset/clone | 仅因为能跑29关节就说G1可站立 |
| T3 多 normal 接触与有限形状 | 高；T1/T2 与 device FK | plane–round → finite box/capsule/sphere 成对矩阵；共同生成多个 normal `efc` 行；双侧真实运动与力矩 | 两脚多点/双物体互碰/边缘撤离/背面/偏心触点；转动路径下一步候选；碰撞过滤/身份/同 body 排除；容量恰满和溢出 | 摩擦支持、native全部mesh、CCD |
| T4 native 摩擦与约束求解 | 很高；T3 + dense M solve | source 选定的 Newton 或 CG；MuJoCo solref/solimp/aref/D；condim=3 tangential cone；joint limits，必要 frictionloss | 受载静止、侧向推力临界滑移、正负切速、离开重入、双侧互易 wrench、friction=0负控；同接触配置多质量/多 h；warm/cold start；固定预算收敛/失败状态 | condim4/6、equality/tendon 等未实现语义 |
| T5 G1 支撑组合留出 | 高；T2–T4 | 原生29关节/自由根、实际原几何与驱动参数、双脚持续支撑；随后原策略公共链 | 先无接触三个姿态动力学；小扰动站立、偏载、横向 push、抬脚卸载重入；再原动作短窗与较长窗；保存全部原失败与完整接触/力矩/控制时刻 | 一个动作通过代表通用人形或整体 Newton 对齐 |
| T6 通用形状/约束与规模 | 高；稳定模块后拆分 | 凸 mesh/高度场、ball/equality/tendon/复杂传动、condim4/6；W32/更大规模、设备张量训练 | 每新增语义独立正/负控；异构批、随机 reset、长期力和约束统计；多任务 PPO保存重载；延迟/吞吐AB | 此前批次自动涵盖这些功能 |

T1、碰撞候选/normal 参数、独立 oracle 作者可以隔离并行；FK/J/M 的共享布局只有一个 owner，T3/T4 的共同 solve/commit 不分给多个作者同时改。执行仍父单一串行槽。不是再次派出已结束的旧 builder/reviewer，也不在首域作者交接前插入相同 kernel 实现。

**29 关节不是 29 维 solve。** G1 native 应核 `nq=36 / nv=35`（一个 free root + 29 scalar joints），并逐个读取实际编译的 `nbody/ngeom` 与所有 welded/static members，不预假 body 数等于关节数+1。现有 Phymeo 普通浮根 32 scalar joints 容量可作有界实现参考，但不能据此宣称 native 35维矩阵、原生所有身体/形状索引和 constraint 容量已通过。

G1 的优先依赖是 T1→T2→T3/T4→T5，不先追全部 mesh、tendon 或 renderer。G1 编译资产若实际带非零 `frictionloss`、非 unit gear、特殊 actuator/limit，则这些成为 T5 的前置 admission；不能静默置零后仍叫同一来源。可以声明单独的 primitive/zero-dry-friction variant，但必须与原资产区分。

## 4. 测试库与成功判据

每个能力先建小案例矩阵，再实现差异模块，最后接组合留出。独立 oracle 使用原 CPU MuJoCo 编译/`mj_forward/mj_fullM/mj_jac/mj_step`；Warp 是同家族 GPU 交叉比较，不能代替独立解析的重力摆、球惯量、平移动量、单法向解等固定 oracles。Phymeo 既有不同接触律的通过记录不倒灌。

每批至少检查这些层：

1. **装配：** 源 MJB/版本、所有 ID/name/address、qpos0/ref/limits、body COM/principal inertia、a/D、gear、geom mask/pair/exclude；unsupported 必须在运行前报告，不能截断或仅忽略 metadata。
2. **同一入步：** native q/v/act/ctrl/qfrc/xfrc、nonzero 真实入步速度；FK、COM、J、M、b、smooth acceleration 与 qfrc 的独立比较。加速度坐标转换含运输项，不只做 T 的矩阵乘。
3. **接触与 solve：** 候选 membership、所有有效 contact/efc、frame/摩擦系数/solref/solimp、双侧 J 与 `Jᵀf`、力/力矩、solver残差/停止原因和 iteration。多站点力不是目标力或 geometry witness。
4. **物理轨迹：** 同源 h、质量、几何、驱动、接触参数短窗比较；能量/线性与角动量或外部冲量余额、接触累计冲量/载荷/滑移；h细化及正负/卸载/重入。长窗按工程行为门，不要求混沌轨迹逐位重合，不以精确无穿透证明无限阻塞实现。
5. **时刻与发布：** 入口取力/控制和 `[t,t+h]` 实际步、solver阶段 force、端点重算 force 分开标记；native qpos/qvel 与 Phymeo公共坐标不可混写。force、impulse、目标力三个字段分开。
6. **批与失败：** W1/W4/稀疏 keys、不同 controls；只 reset 一 world 后其他世界**下一步**仍与 clone 对照，不止 snapshot q/v 相等；保留 warmstart/act/eq/mocap（若支持）。已发布 state与tick、持有动作不能被失败候选改变。
7. **实际设备与数值：** 禁止 CPU mj_step、HOST mechanics、Warp step 的负控；TYX CUDA launch有真实生产者证据。有限 FP64→FP32 的溢出输入、NaN、奇异/不定 solve、contact/efc恰满与溢出、晚解码故障都必须明确拒绝且不发布。不能先截断后 count=capacity 就说成功。

数值容差应在每批实施前按单位/规模和 FP32 精度登记，与同律 CPU oracle比较；不沿用 CPU FP64 逐位门，也不运行后追改质量、h、friction 或阈值。优先关闭真实来源/坐标/漏项问题，之后才处理可接受的工程误差。只有 T5 完成真实持续支撑，才把“G1 支撑”改为支持；无接触或单次冲量不算。

训练链继续使用现有 Env/PolicyRunner，先短窗多场景环境、再真实接触学习与全新世界 checkpoint 重载。GPU physics + CPU policy 可以是明确产品模式；只有公开设备 tensor 所有权、stream/event、epoch/borrow、reset mask 和 accepted publication 都通过，才声称设备训练数据面。不要为了“全GPU”先另造一个无法复用现有协议的环境。

## 5. 首域设计复核与高风险边界

已向首域作者指出并收到源码范围确认：

- 纯平移树即使 M 不随 q 变，也有祖先共享质量和分支耦合；必须使用完整常量 M 及 `(M+hD)`，不能逐关节质量除法。一次性 native `mj_fullM(qpos0)` 用于该**已证明常量**模型的编译预处理可接受；一旦放开 revolute，必须每步重算，不允许继续冻结 qpos0 M。
- 自由球首域限制真实 COM 与球心一致、isotropic inertia，仍保留原生 local omega/world torque 映射及 quaternion 积分；若偏心、非球惯量或附着树，需新完整方程或明确拒绝。不能按球的几何半径临时重造与原惯量不同的 body。
- 单法向接触须具有真实 `condim=1` 的可作用 pair，来源其它 friction/condim 语义未实现就拒绝。无可能接触的模型不因默认 geom condim=3误拒；反过来不能把可能的 friction contact 当 normal-only。
- 一条独立 normal 的解析解不能串行施给多点后称共同 solve；多接触会经 `JM⁻¹Jᵀ` 耦合，尤其两脚、偏心点和多 body。
- 原生 contact law 参数与入步速度要保留。当前 HOST 速度摩擦材料-law 是可选另一产品模式，不得作为本移植的隐式替代。

这些是实施边界和验收要求，不是已经运行的新失败。首域作者已确认完整 prismatic dense M、真实 free-sphere惯量/坐标和 possible-pair admission；本研究没有发现需要重复实现其首域的证据。

## 6. Apache 来源与代码接纳

实际安装的 0.0.2 `METADATA` 声明 `Apache-2.0`，分发含 `licenses/LICENSE` 与 `licenses/AUTHORS`；所读 `types.py/smooth.py/solver.py` 等头部保留 **Copyright 2025 The Newton Developers**，AUTHORS列 Google LLC/NVIDIA Corporation。上游 [LICENSE](https://github.com/google-deepmind/mujoco_warp/blob/main/LICENSE) 与[项目主页](https://github.com/google-deepmind/mujoco_warp)公开该许可；算法语义另对照 [MuJoCo 3.3.7 computation](https://mujoco.readthedocs.io/en/3.3.7/computation/index.html) 和[原生模型/坐标API](https://mujoco.readthedocs.io/en/3.3.7/APIreference/APItypes.html)。

移植代码应保留适用原头部与归属、标出 TYX 修改，随分发携带 Apache LICENSE；若所复制来源分发有 NOTICE，保留适用归属。记录原文件、固定版本/提交（取得时）、源码 hash、移植文件和函数映射。新增 Phymeo 重写部分明确自己的归属，不能把改 import后的派生文件标成完全原创。本文未复制算法实现、未改任何依赖，也不把上游 main 今天的功能表作为 0.0.2 已验功能清单。

建议每批新增一个小的 port manifest，和该批 source/config、实际 backend、oracles、首次失败及最终结果一起归档。仅修改文档无需导入上游或重跑其旧组。

## 7. 如何避免再次陷入无效循环

每一批的结束物必须是一个**新的可调用能力**、一个实质失败的闭环，或一项有对照的测量收益；不是文档页数、测试条数或“更严谨”的证明长度。两轮没有这些收益就停止该支线，回到缺失字段/算子/语义的最小反例。

- M/J/b 不对：查字段、body frame、qpos0 ref、root速度和惯量，不调接触参数掩盖。
- 多接触不稳：先看全 J/M、efc参数、cone状态与solver停止，禁止把载荷拆到多个独立物理step。
- 精度误差已在登记的工程门内：继续连接模块，长窗只验证行为，不把“精确端点相触”变成重写全部流程的前提。
- G1支撑不过但小模块已过：保留失败并找第一个错误时刻/量，分别判断几何、动力学、求解还是策略域；不调质量/关节阻尼/初态去追一个舞蹈结果。
- 低吞吐：在模块正确和真实受载多 world过后测热点，再选 dense→sparse、批并行、减少回读或排序优化；没有profile数据不先造大型缓存或更换后端。

T0/T1/T2及T3a/T3b九类配对、共同法向求解和连续重入已有实际限域验收，不重复实施或重跑已绿组。下一批 **T4先冻结native摩擦cone/共同求解的源码语义与多案例矩阵，再开发condim3、joint limits及实际G1需要的frictionloss**；保持现有几何、动力学与事务接口。每批仍先研究固定上游并预登记案例。T3a用明确命名的dual active-set解同一frictionless QP，不把它当原生Newton/CG迭代；T4必须重新冻结摩擦solver语义。G1完整移植依赖T3–T5组合，多法向接触通过不等于G1支撑通过。
