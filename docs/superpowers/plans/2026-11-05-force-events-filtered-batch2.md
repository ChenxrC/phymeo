# Force Control, Contact Events and Filtered Rows — Batch 2 Implementation Plan

> **For agentic workers:** Use superpowers:subagent-driven-development / executing-plans task by task. Existing user-authorized agents and isolated worktrees take precedence over creating replacement workers.

**Goal:** 在既有通用场景上补受限获取控制、统一载荷事件观察、保持全部公开结果的过滤行复用。

**Architecture:** A2只新增控制器，发送RobotCommand而不写物理状态；B2用不可变多分量事件适配两类场景，父维护一次提交接缝；C2独占普通detect的静态过滤行复用。三作者隔离编码，父独占所有项目执行、公共导出与集成。

**Tech Stack:** existing Python/NumPy/HOST articulation and pinned TYX interpret/CUDA. No dependency or native pin changes.

**Spec:** [上一批实测与已经选定的后续节点](../../benchmarks/p65-scene-contact-scale-batch1.md)；[总体功能路线](2026-11-05-after-night-newton-parity.md)。用户本次继续执行该路线，没有新的截止时间或周期任务。

## Global Constraints

- 起点69bd12d；父及三作者均实时确认分支与WIP，原作者树只在clean时FF，不reset、不覆盖未交接工作。
- 新证据根artifacts/p65-postnight-batch2-20261105-01。所有import/pytest/数值/离线对照/lint/format/build/install/profile严格父串行；作者只源文本、file IO、小范围Git。
- 先tests-only RED，父实际确认缺能力/机制，再实现。必要失败保留原输出；未改语义的已绿集合不重复。
- 旧mimic突加2mm预载的RAW1、裸门及输入保留。新的渐闭协议独立预先冻结，不回写旧通过状态。
- 不改质量/物理误差门/原C1输入和输出来制造通过或加速。只读观测不推进生命周期，拒步不发布半步。
- 原A1限HOST单固定根机器人/单world；新控制器不因此提供CCD、自碰撞、柔体参与或GPU关节动力学。
- 不push/远程CI/main merge/发布，不新建agent或周期任务，不清理证据/junction，不重复旧whole suites。

## Review Focus

1. 多个接触点同shape pair时，完整不同法向、双侧材料与摩擦力矩不能平均掉。
2. 载荷取样时间与接受态时间必须区分，记录/控制器不能消费未来或重复采样。
3. 获取状态只能由实际测量和预定持续判据判定；空夹、单边接触、超力和失持不能伪报held。
4. 静态过滤记录可以复用；实际code5分离仍要通过旧的几何合法性检查，被过滤形状的非法运动也不能漏验。
5. 性能保全局slot和全部字段。完整载荷/状态/history逐步对照，不以少输出冒优化。

## Preflight / ownership

| 边界 | 检查与裁定 |
| --- | --- |
| A2 ↔ B2 | A只消费现RobotSceneObservation发命令，独立新control模块；B加事件观察，父改robot_scene提交点。A不依赖B才能运行 |
| B2 ↔ C2 | B1 ContactHistory保持兼容；普通scene的新事件视图只读适配已提交history，原C1公开帧不删改。C测量绑定实际产品HEAD |
| C2 ↔ parent | C独占collision_pipeline过滤/slot部分；父不改其算法。原核、pair表、分类和容量门保持 |
| A2 自洽 | 不把控制成功等同碰撞/物理功能；两个拓扑正控与零摩擦/超载负控预配 |
| B2 自洽 | 明确材料law种类，不转换不同力律；多component保真，稳定pair身份不冒静摩擦/恢复求解 |
| C2 自洽 | 性能RED针对旧重复分派，语义用旧原结果及独立负控；不只测新字段存在 |

Ruling: 保留原三个作者与工作树、并行编码/串行运行，是当前用户长期授权；不额外引入技能建议的新助手。已有隔离与环境直接复用，不重装或重复baseline。A/B源合同由下面作者交接细化，范围超出此边界先给父文本发现，不先运行。

## A2 — 受限驱动力的通用获取控制（高）

Files: NEW src/phymeo/control/gripper_acquisition.py、tests/test_gripper_acquisition.py、tests/test_gripper_acquisition_windows.py、examples/robot_force_acquisition.py、docs/benchmarks/p65-gripper-acquisition.md。禁止改现robot_scene/contact_scene/root exports。

API首切片为 GripperAcquisition.propose(observation, dt_s, posture_target, intent)，只返回RobotCommand与不可变判定；配置独立joint_indices/open_q/closed_q/速度与驱动力上限、payload_key与两组pad、力目标及持续门。contact_ready只表示双侧力和关节速度符合持续门；实际holding必须另有lift净空、台面卸载与相对滑移证据。重复同step和拒步重试不推进控制状态。

- [ ] 核现bounded PD、真实q/v/载荷及本机Newton控制入口，冻结通用闭合映射、SI单位、限幅与状态机。
- [ ] 写实际能力缺失RED；非法测量/时序、空夹、单边接触、失持和饱和均有负控。
- [ ] 纯控制器产生RobotCommand；不修改任何物体/关节状态，不对象名称/世界坐标/特定时刻特判。
- [ ] 预先冻结Panda/mimic、球/箱的新渐闭获取/保持/释放窗口；不同协议独立记账，旧突加预载仍失败。
- [ ] 父实际执行控制专项与物理窗口，记录所有q/v、接触和状态转换；失败只按证据修机制。
- [ ] 实际受限正负控报告与公共用法交接，不把未做到的从台面获取/完整放置记为完成。

## B2 — 两类场景的统一事件首切片（高）

Files: NEW src/phymeo/runtime/contact_events.py、tests/test_contact_events.py、tests/test_robot_contact_events.py、docs/benchmarks/p65-unified-contact-events.md；父改src/phymeo/runtime/contact_scene.py、src/phymeo/runtime/robot_scene.py与公共导出。原contact_history数据语义保留。

两scene .contact_events 返回 SceneContactEvents；每pair SceneContactRecord沿用ContactIdentity，保物理owner与components元组。ContactComponent含完整原PairLoad、每site材料、material_kind和不可变原law；不平均normal/gap、不声明跨步component身份。普通 contact_events_from_history 是纯只读视图；机器人 RobotContactEventAdapter.stage 在父一次commit前构造完整新snapshot，坐标采用入步link/COM姿态。

- [ ] 冻结 .contact_events 的不可变snapshot与组件字段，完整原PairLoad、law类型、所有site/双侧address/施力点/力/力矩。
- [ ] 普通与机器人两个正常入口缺失RED，真实载荷不由假mock代替。
- [ ] 普通history只读适配，机器人新事件在现一次commit前stage；不改变物理积分。
- [ ] 多点/非共线法向、纯扭转、退出重入、局部ID复用、查询幂等及拒步重试验证。
- [ ] HOST及已有普通场景CUDA适配；机器人仍HOST，不能静默fallback。
- [ ] 父集成同一public接口并验证源码外用法。

## C2 — 完整静态过滤记录复用（中）

Files: 修改src/phymeo/geometry/collision_pipeline.py；新增tests/collision/test_filtered_pair_reuse.py、scripts/compare_scene_scale_frames.py、docs/benchmarks/p65-filtered-pair-reuse.md。不修改原C1驱动/inputs/oracle。

- [ ] 真实RED量到重复静态行进入原窄相，而非仅缺新属性。
- [ ] workspace绑定模型时冻结code1–4记录；code0/5用原有global pair_slots窄相，再按完整顺序组装。
- [ ] 保路径验证、真实backend分类、capacity、code5非法几何和失败时旧publication；覆盖mask/excluded优先级。
- [ ] HOST/interpret/CUDA语义验证和两个模型同key互不污染。
- [ ] 新源码固定九档矩阵一次，逐步完整对照上一批18个已存窗口；仅精确已有timing字段可排除。
- [ ] 基于实际相同输出墙钟报告收益或无收益，不声称Newton整体速度。

## Parent integration

- [ ] 小范围交叉review，所有实质发现有聚焦回归，不重跑无变化全集。
- [ ] 父合入作者独立提交、完成共享接缝与导出，所有旧WIP保留。
- [ ] 干净归档构建与源码外wheel使用新接口；仅已有uv离线target安装，不修改主环境。
- [ ] 更新本计划、结果报告与continuation；记录剩余差距，不据局部通过宣布Newton全面对齐。

## Ledger

- 三作者已FF基线69bd12d，A/B路径与接口已冻结；父执行槽唯一。
- C2首tests-only 65a35e3：c2-filter-red真实1failed/1.06s，完整21行物理正控后发现code1–4仍送原窄相，原输出已保。只允许针对该机制实施。
- 父先单独集成C2并测九档，与上一批已存完整帧比较，然后再接B2共享源，避免混淆C2性能收益。
