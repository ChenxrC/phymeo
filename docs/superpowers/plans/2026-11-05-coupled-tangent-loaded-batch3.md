# 耦合切向状态与受载成本第三批实施计划

> **For agentic workers:** 使用 superpowers:subagent-driven-development，沿用三个原作者，父独占项目执行队列。

**Goal:** 补一个可被刚体、机器人与粒子材料适配器共同使用的耦合切向求解核及事务状态，并在同进程解释受载窄相成本。

**Architecture:** 法向力和几何仍由已有模块提供；新切向核消费全接触响应矩阵，返回力与弹性、塑性和功的收据。材料身份由独立显式站点登记管理，观察事件不充当求解状态。当前场景默认定律不变，自动接入完整机器人和非线性材料路径另行验收。

**Tech Stack:** 原 NumPy HOST、现有局部摩擦盘求解；原 TYX CUDA 规模场景作诊断，无依赖或 pin 变更。

**Spec:** 本文“设计与边界”及[第二批的下一批判据](../../benchmarks/p65-force-events-filtered-batch2.md#收尾和后续)；总体[功能路线](2026-11-05-after-night-newton-parity.md)保持。

## 已核实的起点和设计与边界

父源 ff6efeb；三个原工作树先核 clean 后 fast-forward 到该源。当前 RobotContactMaterial 使用速度正则摩擦，零切向速度时无静载牵引；已有 MaterialStickSlipStepper 与材料 seam 复用 _material_stick_law 的隐式弹性返回映射，但各自限制单站点或特定几何。不能把这些限制直接套到共享机器人和多刚体接触。

本机 Newton METADATA 为 1.7.0。其 VBD 的 rigid_contact_history 文档区分 latest 的数值乘子 warm-start 与 sticky 的材料锚点重放；只有确实对应前一求解 contact buffer/generation 才恢复。源码在独立环境 newton/_src/solvers/vbd/solver_vbd.py:544。官方[当前 VBD 源码](https://github.com/newton-physics/newton/blob/main/newton/_src/solvers/vbd/solver_vbd.py)用于查证分层；实现以本机 pinned 源为准，不把远端 main 当固定版本。该调研没有新增 Newton 物理执行结果。

选择：
- 继续增加夹力不能补上通用静摩擦机制，也会掩盖原100g失败，排除。
- 每站点独立弹簧忽略共享自由度的交叉响应，排除。
- 本批采用**全响应矩阵的有限弹性 Coulomb 接触**，先取相同静/动系数保持凸性；不改旧材料的两系数接口或默认行为。刚柔/机器人适配必须提供真实质量、雅可比与双侧材料位移，不能拿事件行序号当长期身份。

A3 的数学问题：A = diag(1/k_t) + h²W，最小化 fᵀAf/2 + (xi_old + d_free)ᵀf，每个二维站点满足 ||f_i|| <= mu_i N_i。W 保留全部站点交叉块，来自同一入步状态的 J M^-1 Jᵀ，可半正定；正切向柔量使 A 正定。需验完整系统最优性和容量，不用局部绿代替耦合收敛。输出预测实际材料增量、elastic/plastic、功、储能、塑性耗散、隐式耗散和求解残差。

B3 只消费调用方有依据的稳定站点标识、完整双侧材料地址与明确材料坐标系。相同空间点、最近邻或数组位置不自动产生材料连续性。未知换面/重参数化必须重置或拒绝并记录释放能量。状态采用不可变 proposal 与明确 commit；拒步、重复查询、旧 proposal 不会消耗历史。

一般转动后真实材料位移不等于 hJv。本批核可处理瞬时非对角响应，长窗位移/能量成功声明仅限试片确实满足的运动域；不自动宣布一般机器人持续抓取已解决。生产 scene 的法向/驱动自由预测、全切向联解、实际共同推进、材料位移复核和原子发布将作为后继适配节点。

## Global Constraints

- 所有 import、pytest、数值驱动、离线分析、lint、formatter、build、install 只由父串行执行，每命令 fresh guard。
- 作者纯文本读写与小 Git 可并行；现有共享接缝由父维护；不新建 agent。
- 不调整原质量、法向律、夹力、容差或旧失败窗；原100g mimic RAW1 保留。
- 不重跑同源已绿组、不重做旧归档、不恢复自动化、不改 pin/CI/main/发布。
- 明确 HOST 切向求解与 TYX 几何/法向/积分的实际边界，不宣称新 GPU 切向核。
- 每个新增物理窗口预声明输入和预算；不满足则保留最早失败，不循环试参。
- 新证据根 artifacts/p65-postnight-batch3-20261105-01，旧第二批目录只读。

## Review Focus

- 共享机器人/刚体或重复站点产生的非对角响应：不能丢 cross blocks。
- 材料点相同世界位置但不同 owner/frame、顺序改变或换面：不能串用弹簧。
- 弹簧卸载允许正功，但功、储能和两类耗散必须共同核验。
- 无法收敛、容量耗尽、假转动矩阵和拒步：不发布部分状态。
- 性能报告保完整公开输出及计数；分开时间阶段的快照不当因果证据。

## A3：可复用耦合切向核（高）

原机器人作者 p65-robot-substep。
NEW src/phymeo/solvers/coupled_tangent.py、tests/test_coupled_tangent.py、
tests/test_coupled_tangent_motion.py、docs/benchmarks/p65-coupled-tangent.md。

- [ ] 先提交测试；父运行一条最小真实缺失 RED。
- [ ] 实现有界 full-W 求解、输入所有权及完整结果收据；保留旧 scalar law。
- [ ] 验单站点独立解析解、非对角/冗余站点、排列/坐标变换、零摩擦/负载减少、上限失败。
- [ ] 两种真实质量拓扑的多步受载/起滑/卸载，核实际速度、位移、动量与能量；额外机器人雅可比响应留出。
- [ ] 父串行测试、针对性修复、静态核验；原另一作者交叉审阅后小提交/集成。

## B3：显式材料站点状态（中高）

原刚柔作者 p65-world-scale。NEW src/phymeo/solvers/tangent_history.py、tests/test_tangent_history.py、docs/benchmarks/p65-tangent-history.md。TangentHistory.initial_state / prepare(state, samples, sample_time_s, dt_s) / commit(state, trial, updates)均保输入不可变；StationKey(world, token)显式声明，MaterialAddress保存owner/shape/topology_revision/feature/weights/local_point_m，StationSample另给proper material frame、normal和实际法向力。commit只产生候选，调用方与真实物理状态一起发布；此模块自身不作物理验收。

- [ ] 固定身份、地址/frame、episode、transport 与 reset 规则；先提交测试，父实际 RED。
- [ ] 纯 prepare/commit 事务及有界状态，完整丢弃能量、过期/重复输入与对调隔离。
- [ ] 从真实有限几何和原事件构造材料样本，验证顺序、刚体共转、材料点、退出重入与不支持换面。
- [ ] 与 A3 做一个父持有的组合用法：失败不推进，同输入重试与有效历史加载。
- [ ] 父静态/专项验证，原另一作者交叉审阅后集成。

## C3：同进程受载诊断（中）

原刚刚/规模作者 p65-tyx-rays。
NEW scripts/profile_scene_loaded_narrow.py、tests/scale/test_loaded_narrow_protocol.py、
docs/benchmarks/p65-loaded-narrow-diagnostic.md。

- [ ] 固定 loaded/birth × W1/W32，每档16帧；同进程两个独立 scene 同输入，逐帧 AB/BA 交替。
- [ ] A仅在诊断上下文把完整配对分派送回保留的 _detect_pairs；B用当前过滤复用。此A是同源 retained full-dispatch，不冒旧产品二进制。
- [ ] 保全 step/state/geometry/history 全字段，仅列明的墙钟字段可不同；原非墙钟计数保留。
- [ ] 另外独立 tracing 窗记录 HullShape 重建、面/边最近特征、材料见证/距离界、HOST路径证明；互斥计时和未分类残项明确。
- [ ] 父执行固定四档并报告真实数据；无证据不预先改产品/许诺加速；不重跑旧九档。
- [ ] 恢复所有诊断 monkeypatch（含异常）；全输出不一致直接失败。

## 父集成与判停

- [ ] 审查三个独立交付，与原测试/作者结论核对；只新增必要组合测试。
- [ ] 明确公共导出；从干净源构建并安装到源码外独立 site，执行新公共用法。
- [ ] 本批报告/continuation只写实际结果、边界和下一个可执行节点；工作树 clean，小提交，不 push。

若两次同因失败而无新机制证据，停止该路线并保留最小反例。不能用不停追加合同/证明代替可运行模块；不能把本批低层核通过称 RobotScene 已启用静摩擦或原 mimic 已修复。
