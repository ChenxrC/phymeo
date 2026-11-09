# 机器人—环境—策略：当前连接架构

执行中草稿，2026-11-10到08:00批次。橙色作者代码尚待父串行验证，不能当作全部已验收。最终结果将以本批收据更新。

[离线架构图](robot-environment-connections-2026-11-10.html) · [批次计划](../superpowers/plans/2026-11-10-0800-robot-environment-connections.md) · [本批前已验收H1](../benchmarks/p65-h1-actuator-capacity-support.md)

## 可调用边界与真实所有权

| 层 | 源码入口 | 做什么 | 不做什么 / 当前边界 |
|---|---|---|---|
| 资产与物理声明 | `builders/floating_mjcf.py`、`dynamics/floating_robot.py` | 导入原关节/惯量/几何；不可变a/D；真实自由根 | MJCF仍为明确子集；不从动作参考生成发布姿态 |
| 任务映射 | `examples/g1_connected_policy.py::G1Task` | 当前实际观测→154维协议；原动作缩放/限幅力矩→公共command | G1协议保留在案例层，不写入通用物理核心；新连接待验 |
| 策略调度 | `control/policy_runner.py::PolicyRunner` | due mask、动作保持、每世界整数节拍；一次调用一个真实物理步 | 不拥有物理，不提供RNN或完整物理checkpoint；新连接待验 |
| 任务环境 | `training/resident_env.py::ResidentVectorEnv` | controls/observe/evaluate、reward/done、final observation、选择复位 | NumPy/HOST边界；新受载stepper与拒步mask待验 |
| HOST场景适配 | `training/scene_backend.py::SceneBatchBackend` | 一个逻辑world拥有一个原RobotScene/MultiRobotScene；factory复位 | CPU逐scene循环，不能称设备批处理；不覆盖裸ContactScene |
| TYX受载适配 | `training/resident_stepper.py::ResidentContactStepper` | 借用原batch/plan，调用原step_contacts | 不复制solver、不替换owner、不关闭借用资源；新连接待验 |
| 真实物理推进 | `runtime/multi_robot_scene.py`、`backends/tyx_world_batch.py` | 接触载荷、完整M/J、驱动、候选与共同发布 | HOST和TYX能力域不同；设备受载几何仍有HOST路径，持续摩擦受限 |
| 观测与传感 | `sensors/`、`experiments/humanoid_support.py` | 接受q/v、载荷、IMU、力矩、深度/法线/ID、输入时刻 | 力采样在步入口，姿态在步末；不能混作同一时刻量 |
| 学习器连接 | 候选 `training/rsl_rl.py` | 可选TensorDict接口、外部PPO | 当前尚未安装/验证；不宣称零拷贝或G1从头学习 |
| 回放/审计 | 原JSONL与候选G1离线Three.js导出器 | 从保存的实际q做FK，可见真实受力与失败 | 不重放参考NPZ作为实际运动，不生成新物理步 |

上述源码路径均相对 `src/phymeo/`，`examples/` 除外。当前公共运行示例是 `examples/g1_phymeo_support.py`；新的统一连接示例是 `examples/g1_connected_policy.py`，状态分别验收。

## 一次真实推进

1. 调度器按当前接受观测和due mask调用无内部副作用的策略。未到控制时刻的world保留上一次接受动作。
2. 任务将动作转换成实际command。G1使用原PD公式、原限额，控制目标只生成力矩，不覆盖q/v。
3. 环境显式选择普通步、原受载stepper或HOST场景适配。
4. 原runtime负责接触采样、求解、候选检查和接受发布；新工具层不冒充solver。
5. 只有所有目标world的完整接受map合法且为True，才产生reward/done/episode transition。已完成world先保留final observation，再按原接口复位。
6. 接受结果回到策略调度和传感/记录。G1当前50Hz策略、名义2kHz物理，每40个实际接受子步更新策略。

## 拒步与复位

- 明确完整allFalse：没有环境transition，不推进策略/episode时钟，可重试。
- 部分True或未知发布：记录实际mask，要求全部恢复/复位，不声称已接受物理被回滚。
- 物理已接受但观察/奖励失败：保留后置失败与实际已推进状态，禁止继续用旧观察训练。
- HOST选择复位通过factory重建原初态，不直接写q/v；新对象全部准备完成后才替换被选world。未选world保持。
- 策略快照仅记录调度诊断。完整learner/normalizer/物理状态恢复必须另有绑定，不能凭一个动作数组宣称可精确resume。

## 与整体功能全景的关系

这条公共链连接场景与机器人、碰撞响应、GPU/多世界、训练、传感与求解编排。G1是一个实际留出，独立受载机械任务与拒步/复位仍必须通过。

尚未完成的设备持续摩擦、跨独立participant联合设备求解、零拷贝张量、W256/W1024性能，以及软布/体软体/杆缆/MPM物理族，不因这条HOST链成功而被标为完成。

## 本批证据

`artifacts/p65-connections-20261110-0800-01` 保存长舞蹈与后续命令/源绑定/首失败；最终检查点、实际通过组、安装调用和图的视觉检查待回填。G1长动作当前只在原HOST物理入口运行，统一工具入口不借用其通过标签。

## 全引擎定位：已实现模块与连接边界

上图聚焦本批补的机器人闭环。它不是所有物理族已经共用一套设备求解器的声明。现有模块按源码可以放进以下结构；本批不重跑其历史验收，具体适用域沿原报告保留。

| 功能层 | 当前已实现入口 | 与本批连接的关系 | 仍未接通 |
|---|---|---|---|
| 场景与机器人 | `builders/urdf.py`、`floating_mjcf.py`、`mimic_urdf.py`、`usd.py`；`runtime/contact_scene.py` | 原公共场景经SceneBatchBackend进入相同任务/策略接口 | 资产schema仍为子集；更广闭环、多自由度关节和批量IK不能由G1推断 |
| 碰撞与响应 | `geometry/collision_pipeline.py`；有限形状/网格/BVH、独立查询、原场景双侧载荷 | G1使用真实球/胶囊–有限盒地面和原力律；N2借用原resident contact plan | 查询覆盖与受载响应覆盖不同；设备完整patch、持续换特征摩擦及更广CCD分别验收 |
| GPU与多世界 | `backends/tyx_world_batch.py`与resident树/浮根绑定 | ResidentContactStepper选择原受载发布；HOST场景适配另走逐scene循环 | HOST窄相、NumPy读回仍在链上；不是零拷贝，也不是W256/1024吞吐已通过 |
| 控制、任务与训练 | `control/policy_runner.py`、`training/resident_env.py`与本批适配 | 真观测→策略→原command→真实物理→reward/final-observation/复位 | RSL-RL列为可选连接；PPO学习与可微多步反传是两项不同能力 |
| 传感与交互 | `sensors/depth_camera.py`、`robot_imu.py`、`contact_wrench.py`；已有在线Python控制与离线Three.js | 读取接受态和注明输入时刻的载荷；本批回放直接消费日志 | RGB/材质/光照、批量传感和通用编辑器未因回放而完成 |
| 跨域编排 | `runtime/scene_stepper.py`、`coupling/domain_graph.py`与点端口 | 有场景事务和指定多域共同推进；新env只借用后端，不另造物理事务 | 任意域自动选后端、跨独立participant设备联合求解仍需专门组合层 |
| 布与体软体 | `solvers/cloth.py`、`compliant_cloth_contact.py`、`tetrahedral.py`、`materials/stvk.py` | 各自已有程序、材料和交互接口；不混入本批G1动力学 | 密集持续折叠/静摩擦、目标软布放台、软块抓捏与训练接口仍分域推进 |
| 杆缆、MPM/流体 | 当前刚性连杆、粒子和弹性固体不能代替这些本构族 | 不画成已经连接的训练物理后端 | 柔性截面弯扭、粒子–网格与塑性/流动需另立物理族 |

数据所有权仍分三层：静态模型/资产声明由builder校验；真实状态和物理接受发布由原scene/batch/SceneStepper持有；任务、策略和训练层只持有输入、观测副本与episode时钟。一个策略动作数组不能当作物理checkpoint，一个查询法向也不能当作已经施加的力。

[较大功能族全景与原验收入口](../development/phymeo-newton-panorama-2026-11-07.md)保留更广范围；本批最终状态以本页实际执行证据为准。这里的源码路径相对`src/phymeo/`。