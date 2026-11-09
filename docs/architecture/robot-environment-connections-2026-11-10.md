# 机器人—环境—策略：当前连接架构

2026-11-10批次实际连接：公共接口HOST72与interpret/CUDA各12通过；公共入口G1完整20秒通过全部原工程门；原HOST入口完成官方全131.48秒无跌倒，但峰值力/参考XY两门未过。CPU PPO两任务及checkpoint重载、W256混合运动、源码外wheel受载调用和实际轨迹回放均已运行。下图的实线表示这次核实的调用，不代表所有后端或所有工程门都通过。

[离线架构图](robot-environment-connections-2026-11-10.html) · [批次计划](../superpowers/plans/2026-11-10-0800-robot-environment-connections.md) · [本批前已验收H1](../benchmarks/p65-h1-actuator-capacity-support.md)

## 可调用边界与真实所有权

| 层 | 源码入口 | 做什么 | 不做什么 / 当前边界 |
|---|---|---|---|
| 资产与物理声明 | `builders/floating_mjcf.py`、`dynamics/floating_robot.py` | 导入原关节/惯量/几何；不可变a/D；真实自由根 | MJCF仍为明确子集；不从动作参考生成发布姿态 |
| 任务映射 | `examples/g1_connected_policy.py::G1Task` | 当前实际观测→154维协议；原动作缩放/限幅力矩→公共command | G1协议保留在案例层；公共连接已验20秒，不借独立入口的131.48秒标签 |
| 策略调度 | `control/policy_runner.py::PolicyRunner` | due mask、动作保持、每世界整数节拍；一次调用一个真实物理步 | 不拥有物理，不提供RNN或完整物理checkpoint；已通过专项 |
| 任务环境 | `training/resident_env.py::ResidentVectorEnv` | controls/observe/evaluate、reward/done、final observation、选择复位 | NumPy/HOST边界；新受载stepper与拒步mask已通过专项 |
| HOST场景适配 | `training/scene_backend.py::SceneBatchBackend` | 一个逻辑world拥有一个原RobotScene/MultiRobotScene；factory复位 | CPU逐scene循环，不能称设备批处理；不覆盖裸ContactScene |
| TYX受载适配 | `training/resident_stepper.py::ResidentContactStepper` | 借用原batch/plan，调用原step_contacts | 不复制solver、不替换owner、不关闭借用资源；已通过两backend专项 |
| 真实物理推进 | `runtime/multi_robot_scene.py`、`backends/tyx_world_batch.py` | 接触载荷、完整M/J、驱动、候选与共同发布 | HOST和TYX能力域不同；原速度摩擦已混合接通，几何/力律仍在HOST，静摩擦历史未接 |
| 观测与传感 | `sensors/`、`experiments/humanoid_support.py` | 接受q/v、载荷、IMU、力矩、深度/法线/ID、输入时刻 | 力采样在步入口，姿态在步末；不能混作同一时刻量 |
| 学习器连接 | `training/rsl_rl.py`、`examples/train_scene_ppo.py` | 可选TensorDict接口、原RSL-RL PPO、保存/重载 | 两个HOST任务实训通过；HOST拷贝，非GPU训练；不宣称G1从头学习 |
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

## 全引擎定位：已实现模块与连接边界

上图聚焦本批补的机器人闭环。它不是所有物理族已经共用一套设备求解器的声明。现有模块按源码可以放进以下结构；本批不重跑其历史验收，具体适用域沿原报告保留。

| 功能层 | 当前已实现入口 | 与本批连接的关系 | 仍未接通 |
|---|---|---|---|
| 场景与机器人 | `builders/urdf.py`、`floating_mjcf.py`、`mimic_urdf.py`、`usd.py`；`runtime/contact_scene.py` | 原公共场景经SceneBatchBackend进入相同任务/策略接口 | 资产schema仍为子集；更广闭环、多自由度关节和批量IK不能由G1推断 |
| 碰撞与响应 | `geometry/collision_pipeline.py`；有限形状/网格/BVH、独立查询、原场景双侧载荷 | G1使用真实球/胶囊–有限盒地面和原力律；N2借用原resident contact plan | 查询覆盖与受载响应覆盖不同；设备完整patch、持续换特征摩擦及更广CCD分别验收 |
| GPU与多世界 | `backends/tyx_world_batch.py`与resident树/浮根绑定 | ResidentContactStepper选择原受载发布；HOST场景适配另走逐scene循环 | W256运动留出已过；HOST窄相/NumPy读回仍在链上，非受载W256或吞吐通过 |
| 控制、任务与训练 | `control/policy_runner.py`、`training/resident_env.py`、`training/rsl_rl.py` | 真观测→策略→原command→真实物理→reward/final-observation/复位 | 两HOST任务已完成PPO与重载；G1用原ONNX，PPO学习与物理反传是不同能力 |
| 传感与交互 | `sensors/depth_camera.py`、`robot_imu.py`、`contact_wrench.py`；已有在线Python控制与离线Three.js | 读取接受态和注明输入时刻的载荷；本批回放直接消费日志 | RGB/材质/光照、批量传感和通用编辑器未因回放而完成 |
| 跨域编排 | `runtime/scene_stepper.py`、`coupling/domain_graph.py`与点端口 | 有场景事务和指定多域共同推进；新env只借用后端，不另造物理事务 | 任意域自动选后端、跨独立participant设备联合求解仍需专门组合层 |
| 布与体软体 | `solvers/cloth.py`、`compliant_cloth_contact.py`、`tetrahedral.py`、`materials/stvk.py` | 各自已有程序、材料和交互接口；不混入本批G1动力学 | 密集持续折叠/静摩擦、目标软布放台、软块抓捏与训练接口仍分域推进 |
| 杆缆、MPM/流体 | 当前刚性连杆、粒子和弹性固体不能代替这些本构族 | 不画成已经连接的训练物理后端 | 柔性截面弯扭、粒子–网格与塑性/流动需另立物理族 |

数据所有权仍分三层：静态模型/资产声明由builder校验；真实状态和物理接受发布由原scene/batch/SceneStepper持有；任务、策略和训练层只持有输入、观测副本与episode时钟。一个策略动作数组不能当作物理checkpoint，一个查询法向也不能当作已经施加的力。

[较大功能族全景与原验收入口](../development/phymeo-newton-panorama-2026-11-07.md)保留更广范围；本批最终状态以本页实际执行证据为准。这里的源码路径相对`src/phymeo/`。

规模边界：CollisionModel单模型最多256个shape。W256每world一物体加独立地面需512个声明，当前单contact plan尚不能容纳；不通过增cap或隐藏分页把该缺口称作通过。原W32接口和纯运动world数与该shape域是不同维度。[分页单次发布调研](../development/p65-resident-contact-scale-research.md)和[设备摩擦/张量接口调研](../development/p65-device-policy-contact-research.md)只画后续路线，不冒充实现。
