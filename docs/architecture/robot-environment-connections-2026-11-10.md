# 机器人—环境—策略：当前连接架构

2026-11-10批次实际连接：公共接口HOST72与interpret/CUDA各12通过；G1公共HOST20秒、CUDA受载2秒均过全部原工程门；原HOST入口完成官方全131.48秒无跌倒，但峰值力/参考XY两门未过。CPU PPO三任务（含安装包真实接触/两质量组）、checkpoint重载、W256混合运动、wheel调用和实际HOST轨迹回放均已运行。实线表示核实的调用，不代表所有后端/工程门或性能都通过。

[离线架构图](robot-environment-connections-2026-11-10.html) · [批次计划](../superpowers/plans/2026-11-10-0800-robot-environment-connections.md) · [本批前已验收H1](../benchmarks/p65-h1-actuator-capacity-support.md)

## 可调用边界与真实所有权

| 层 | 源码入口 | 做什么 | 不做什么 / 当前边界 |
|---|---|---|---|
| 资产与物理声明 | `builders/floating_mjcf.py`、`dynamics/floating_robot.py` | 导入原关节/惯量/几何；不可变a/D；真实自由根 | MJCF仍为明确子集；不从动作参考生成发布姿态 |
| 任务映射 | `examples/g1_connected_policy.py::G1Task`、`examples/g1_resident_policy.py` | 当前实际观测→154维协议；原动作缩放/限幅力矩→公共command | 案例层适配；HOST20s/CUDA2s分别验收，不借原HOST131秒标签 |
| 策略调度 | `control/policy_runner.py::PolicyRunner` | due mask、动作保持、每世界整数节拍；一次调用一个真实物理步 | 不拥有物理，不提供RNN或完整物理checkpoint；已通过专项 |
| 任务环境 | `training/resident_env.py::ResidentVectorEnv` | controls/observe/evaluate、reward/done、final observation、选择复位 | NumPy/HOST边界；新受载stepper与拒步mask已通过专项 |
| HOST场景适配 | `training/scene_backend.py::SceneBatchBackend` | 一个逻辑world拥有一个原RobotScene/MultiRobotScene；factory复位 | CPU逐scene循环，不能称设备批处理；不覆盖裸ContactScene |
| TYX受载适配 | `training/resident_stepper.py::ResidentContactStepper` | 借用原batch/plan，调用原step_contacts | 不复制solver、不替换owner、不关闭借用资源；已通过两backend专项 |
| 真实物理推进 | `runtime/multi_robot_scene.py`、`backends/tyx_world_batch.py` | 接触载荷、完整M/J、驱动、候选与共同发布 | HOST和TYX能力域不同；原速度摩擦已混合接通，几何/力律仍在HOST，静摩擦历史未接 |
| 观测与传感 | `sensors/`、`experiments/humanoid_support.py` | 接受q/v、载荷、IMU、力矩、深度/法线/ID、输入时刻 | 力采样在步入口，姿态在步末；不能混作同一时刻量 |
| 学习器连接 | `training/rsl_rl.py`、`examples/train_scene_ppo.py`、`examples/train_contact_ppo.py` | 可选TensorDict接口、原RSL-RL PPO、保存/重载 | 三HOST任务含真实接触实训通过；HOST拷贝，非GPU或G1从头训练 |
| 回放/审计 | `examples/export_g1_policy_replay.py`与离线Three.js | 从保存的实际q做FK，可见真实受力与失败 | 全131.48秒已导出并浏览检查；不重放参考NPZ，不生成新物理步 |

上述源码路径均相对 `src/phymeo/`，`examples/` 除外。当前公共运行示例是 `examples/g1_phymeo_support.py`；新的统一连接示例是 `examples/g1_connected_policy.py`，状态分别验收。

## 一次真实推进

1. 调度器按当前接受观测和due mask调用无内部副作用的策略。未到控制时刻的world保留上一次接受动作。
2. 任务将动作转换成实际command。G1使用原PD公式、原限额，控制目标只生成力矩，不覆盖q/v。
3. 环境显式选择普通步、原受载stepper或HOST场景适配。
4. 原runtime负责接触采样、求解、候选检查和接受发布；新工具层不冒充solver。
5. 只有所有目标world的完整接受map合法且为True，才产生reward/done/episode transition。已完成world先保留final observation，再按原接口复位。
6. 接受结果回到策略调度和传感/记录。G1当前50Hz策略、名义2kHz物理，每40个实际接受子步更新策略。

PPO是另一条上层闭环：RSL-RL自己的rollout runner → `RslRlVecEnv` → 同一个 `ResidentVectorEnv` → 原后端。它直接消费环境step/reset/观测，不再套一层PolicyRunner。当前action/observation经CPU/NumPy复制；PPO更新属于Torch，物理状态及发布仍属于scene/batch。

## 拒步与复位

- 明确完整allFalse：没有环境transition，不推进策略/episode时钟，可重试。
- 部分True或未知发布：记录实际mask，要求全部恢复/复位，不声称已接受物理被回滚。
- 物理已接受但观察/奖励失败：保留后置失败与实际已推进状态，禁止继续用旧观察训练。
- HOST选择复位通过factory重建原初态，不直接写q/v；新对象全部准备完成后才替换被选world。未选world保持。
- 策略快照仅记录调度诊断。完整learner/normalizer/物理状态恢复必须另有绑定，不能凭一个动作数组宣称可精确resume。

## 与整体功能全景的关系

这条公共链连接场景与机器人、碰撞响应、GPU/多世界、训练、传感与求解编排。G1是一个实际留出，独立受载机械任务与拒步/复位仍必须通过。

原速度摩擦混合桥已过HOST63、interpret/CUDA各12，并通过新wheel源码外受载调用；[限域结果](../benchmarks/p65-resident-friction-connections.md)分别列出两个连续运动机制。设备站点力核、材料锚点静摩擦历史、跨独立participant联合设备求解、无HOST往返策略张量、受载W256/W1024及吞吐，以及软布/体软体/杆缆/MPM物理族仍需分别推进。W256已过的只是两运动族、8步公共连接。

## 本批证据

完整结果见[本批验收报告](../benchmarks/p65-robot-environment-connections-20261110.md)。`artifacts/p65-connections-20261110-0800-01` 保存原命令、源码绑定、所有首失败、真实状态/受力和安装调用。G1长动作只在原HOST入口运行；统一工具入口验20秒，不借用其长窗标签。Torch2.15.0 CPU、TensorDict0.15.5、RSL-RL5.5.2为本次实际版本。

[CUDA2秒组合](../benchmarks/p65-g1-resident-policy.md)和[安装包真实接触PPO](../benchmarks/p65-contact-ppo-connections.md)补上两条端到端留出。当前长动作播放器只验了HOST日志；CUDA短窗日志完整保存但尚未接该播放器。单输入HOST profile及原生产者拷贝/同步计数用于下一批定位，不能把“有CUDA”当作已具吞吐优势。

## 全引擎定位：已实现模块与连接边界

上图聚焦本批补的机器人闭环。它不是所有物理族已经共用一套设备求解器的声明。现有模块按源码可以放进以下结构；本批不重跑其历史验收，具体适用域沿原报告保留。

| 功能层 | 当前已实现入口 | 与本批连接的关系 | 仍未接通 |
|---|---|---|---|
| 场景与机器人 | `builders/urdf.py`、`floating_mjcf.py`、`mimic_urdf.py`、`usd.py`；`runtime/contact_scene.py` | 原公共场景经SceneBatchBackend进入相同任务/策略接口 | 资产schema仍为子集；更广闭环、多自由度关节和批量IK不能由G1推断 |
| 碰撞与响应 | `geometry/collision_pipeline.py`；有限形状/网格/BVH、独立查询、原场景双侧载荷 | G1使用真实球/胶囊–有限盒地面和原力律；N2借用原resident contact plan | 查询覆盖与受载响应覆盖不同；设备完整patch、持续换特征摩擦及更广CCD分别验收 |
| GPU与多世界 | `backends/tyx_world_batch.py`与resident树/浮根绑定 | ResidentContactStepper选择原受载发布；HOST场景适配另走逐scene循环 | W256运动留出已过；HOST窄相/NumPy读回仍在链上，非受载W256或吞吐通过 |
| 控制、任务与训练 | `control/policy_runner.py`、`training/resident_env.py`、`training/rsl_rl.py` | 真观测→策略→原command→真实物理→reward/final-observation/复位 | 三HOST任务含真实接触已完成PPO与重载；G1用原ONNX，PPO与物理反传不同 |
| 传感与交互 | `sensors/depth_camera.py`、`robot_imu.py`、`contact_wrench.py`；已有在线Python控制与离线Three.js | 读取接受态和注明输入时刻的载荷；本批回放直接消费日志 | RGB/材质/光照、批量传感和通用编辑器未因回放而完成 |
| 跨域编排 | `runtime/scene_stepper.py`、`coupling/domain_graph.py`与点端口 | 有场景事务和指定多域共同推进；新env只借用后端，不另造物理事务 | 任意域自动选后端、跨独立participant设备联合求解仍需专门组合层 |
| 布与体软体 | `solvers/cloth.py`、`compliant_cloth_contact.py`、`tetrahedral.py`、`materials/stvk.py` | 各自已有程序、材料和交互接口；不混入本批G1动力学 | 密集持续折叠/静摩擦、目标软布放台、软块抓捏与训练接口仍分域推进 |
| 杆缆、MPM/流体 | 当前刚性连杆、粒子和弹性固体不能代替这些本构族 | 不画成已经连接的训练物理后端 | 柔性截面弯扭、粒子–网格与塑性/流动需另立物理族 |

数据所有权仍分三层：静态模型/资产声明由builder校验；真实状态和物理接受发布由原scene/batch/SceneStepper持有；任务、策略和训练层只持有输入、观测副本与episode时钟。一个策略动作数组不能当作物理checkpoint，一个查询法向也不能当作已经施加的力。

## 身份分层：应用对象不能替代物理刚体

| 身份 | 含义与规则 |
|---|---|
| 环境world key | 向量环境中的稳定世界身份，不是数组行号。HOST适配内部scene可各有自己的world0；它们仍由不同owner隔离。 |
| 应用robot ID | HOST场景中整台关节机器人的对象身份，用于command/observation路由。 |
| CollisionKey(world, body, shape) | resident碰撞里的原物理刚体与形状身份；一台30body机器人必须保留各body身份，不能全标成同一robot ID。 |
| tree group index / model token | 将形状绑定到原物理根或连杆；body与group在同一world里须对应无歧义。from_asset保留原几何、principal frame与模型来源。 |
| state/topology epoch | 接受版本及注册来源；reset/restore使旧query失效。它不是某world的物理tick，不能作为策略时钟代替品。 |

N8的首次注册错误正是混淆了整台robot ID与逐body ID，原guard在0物理步时拒绝。新示例保留原资产body/shape ID，地面另用不冲突ID；没有删除来源检查。这条规则同时适用于以后URDF、多机器人或分页注册，不是G1专用碰撞例外。

[较大功能族全景与原验收入口](../development/phymeo-newton-panorama-2026-11-07.md)保留更广范围；本批最终状态以本页实际执行证据为准。这里的源码路径相对`src/phymeo/`。

规模边界：CollisionModel单模型最多256个shape。W256每world一物体加独立地面需512个声明，当前单contact plan尚不能容纳；不通过增cap或隐藏分页把该缺口称作通过。原W32接口和纯运动world数与该shape域是不同维度。[分页单次发布调研](../development/p65-resident-contact-scale-research.md)和[设备摩擦/张量接口调研](../development/p65-device-policy-contact-research.md)只画后续路线，不冒充实现。

## 新增并行分支：MuJoCo-TYX

本页上方的原HOST/resident链和验收保持独立。新分支为CPU MuJoCo编译记录 → 默认CUDA MuJoCo-TYX → 同一公共controls/step/snapshot/reset协议；可以继续接ResidentVectorEnv/PolicyRunner。设备内完成primitive窄相、当前Jacobian、dry/limit/pyramid行装配、共同质量矩阵求解和Euler，整批共同发布力与运动。详[统一编译与T4管线](unified-scene-compilation.md)。

组合消费者为examples/g1_mujoco_tyx_policy.py，保留原G1策略/输入并分别推进TYX与CPU反馈轨迹。同入步native力学字段在F1严格比较，长窗保存同入步诊断并使用固定支撑/姿态/载荷/跟踪工程门。最新窗口状态见[T4/G1报告](../benchmarks/p65-mujoco-tyx-mixed-constraints.md)。它尚未复用本页旧Three.js长动作播放器，也不把原resident CUDA2秒或HOST全舞蹈当作自己的结果。

当前性能边界：CPU编译与准入、CPU ONNX示例、NumPy控制/快照和完整验证回读仍在链上；新几何/求解已在TYX设备内，不等于无HOST往返训练。下一步可在已验证的共同物理接口上独立推进批量world、选择性观测、设备策略张量与吞吐测量。
