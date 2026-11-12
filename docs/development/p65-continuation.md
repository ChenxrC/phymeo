# Phymeo 当前续作入口

更新：2026-11-12，**三主线 Batch A 已完成源码与安装资格**：S1/S2 多参与者场景设备查询、D1/M2-P 持久真实变形节点、R1/首 R2 直接 BALL FILTEREXACT 与原接触组合。最终产品 `d2a18bee`，520 成员同源，六个源码外解释/CUDA消费者通过。整个 P6.5 / Newton 功能对齐仍未完成，无新截止。旧 pending/deadline 不覆盖本页。

## 当前入口

1. [八方向差异全景](phymeo-newton-panorama-2026-11-12.md)：已有能力、限定范围、缺口和优先级。
2. [Batch A 实际结果](p65-three-track-batch-a-20261112.md)：三项实现、源码/安装证据、首失败、资源与下一片。
3. [三主线计划](../superpowers/plans/2026-11-12-scene-deformable-scale-roadmap.md)：依赖与验收；[Batch A](../superpowers/plans/2026-11-12-three-track-batch-a.md)已关闭。
4. [M2 原设计](p65-m2-producer-bank-design.md)与[实际节点报告](p65-deformable-bank-report.md)：M2-P 已实现，M2-Q 查询及响应仍待做。
5. [统一编译架构](../architecture/unified-scene-compilation.md)：CPU MuJoCo 编译、CPU/TYX/显式 Warp 与 resident 路径。
6. [07:20 架构快照](../architecture/phymeo-newton-parity-2026-11-12.html)：历史图，未绘入近期新增模块；当前以本页和全景为准。
7. [上一批 BALL-MC/M1-A](p65-parity-motor-asset-20261112.md)、[更早装配批](p65-parity-m1-assembly-20261112.md)、[历史入口](p65-continuation-history-through-20261112-0035.md)：按需要定位，不重读 heartbeat 或重复旧 hash 审计。

## 最新已交付

- **S1/S2**：公共 ResidentSceneBatch.prepare_boundary_query 连接逻辑 world/participant 与真实 accepted 设备 FK，处理不同 anchor、原资产/形状/面身份和完整 revision/tick；四类树、多参与者过滤、资源与失败事务已验。源码解释/HOST41、CUDA39，旧场景兼容4；安装每后端6逻辑步/12参与者步/4queries/64rows/3stale拒绝。仅查询，不是新载荷或 CCD；S3 高层编译接线及第二类独立安装场景待补。
- **D1/M2-P**：公共 TyxDeformableBatch 持久双 bank，真实节点积分/ordered 距离投影/速度重建，力输入、borrow、原子发布、checkpoint/partial reset；普通步不回填 HOST 当前位置。解释44、CUDA33；安装每后端16batch steps/32worldsteps，独立参考及恢复后继通过。M2-Q、接触力/摩擦和布弯曲不在本批。
- **直接 BALL FILTEREXACT**：原 JOINT 广义力映射接执行器动态，与 fixed 球树原 normal/pyramidal 接触共同求解；同次力与运动、完整 EFC/双方 COM 力矩和事务。70新源码节点、4旧机器人兼容；安装每后端160worldsteps。无 FREE+BALL、BALL affine bias、limits/dry 或任意组合。
- **统一安装包**：产品 d2a18bee 的520成员 source/wheel/installed 字节相同；六消费者首次运行均通过。完整收尾绑定143安装证据文件，仍保留原 source/wheel 及所有首失败。没有新增性能结论。

## 既有资格复用

- BALL-MC、M1-A 的 d362799f / 516成员、BALL-M1/ParticipantScene/M1 的 2f5f19d6 / 515成员按原报告复用，不倒灌为新包全量回归。旧报告的限制只描述其当时版本，直接 BALL activation 和场景查询连接现已有本批独立资格。
- ROB2b、BALL1/A/C、COL1a/2/3及旧 resident 载荷范围保持。HOST当前快照查询、M1设备来源、MuJoCo-TYX求解器是不同接口/资格，不能任意相乘。
- N5状态随机 reset、N6 D=1..8/GRU、N7 J8/D2/GRU和S128/S256/D4共同状态/更新通过；不等于新学习成功。紧凑 HOST 镜像消除普通步 native 构造/copy，不代表零 HOST 或 Newton 性能胜出。
- N8 J/R/S共768真实转移、0新Adam更新和297条保存审计保持；未证明弱学习原因，不自动改clip/lr。

## 后续从这里开始

下一批 **Batch B**，按[三主线路线](../superpowers/plans/2026-11-12-scene-deformable-scale-roadmap.md)继续：

1. **S3 统一公共场景**：补编译声明到既有 device query 的自动装配和公共流程；增加工具/自由物体的第二类独立安装场景。复用已验 S1/S2，不重做 adapter。
2. **D2/M2-Q 真实变形查询**：借用 D1 当次 accepted 设备顶点，更新 bounds/BVH，接原有限球/胶囊查询，验证面实际变形、稳定身份与旧 epoch 拒绝。随后 D3 才做合法接触法则、双方力和共同发布；D4/D5 做持续载荷及组合安装。
3. **R2 FREE+BALL 与异构基础**：核实际树/状态/DOF/activation/constraint 接缝，冻结第一格多案例和资源后实现；异构批只组合已验能力。G1功能、G2规模先于G3公平性能比较，性能超越不是强制条件。

原三位作者可在无交叉依赖的隔离文件内并行编码，父独占共享协议/接缝；全部项目执行串行。当前 Batch A 作者和执行均结束，无新执行 owner；下一批启动时登记新的精确范围和唯一执行 owner，不复活旧运行组。

训练 N8 诊断保留为第三线需要时的支撑，不抢占前三主线或触发无假设重训练。允许事先声明且有物理意义的有限穿透、漂移和数值误差，以稳定多案例流程、真实载荷及力/运动一致为验收；不恢复精确相触/无限证明的旧障碍。同一假设两次诊断未缩小问题就保存最小反例并调整假设。

## 保留的失败和缺口

J/R原学习门、G1原46.29秒参考XY门及完整131.48秒动作、R03两项小净加速度门仍未通过。B-point touch-SE 法向与曲面见证不相容，候选隔离未合入。M2-Q及后续响应、通用CCD/持续载荷/摩擦历史、更广机器人组合、批量传感、跨域、密集布/体软体、杆缆与MPM/流体仍见全景；局部资格不替代这些验收。

## 所有权与证据

父目录 C:/Users/38347/.codex/worktrees/p65-parallel-integration/phymeo，分支 codex/p65-parallel-modules，本地未 push。新证据 artifacts/p65-three-track-a-20261112-01，installed-close.json 绑定520产品及六消费者；installed-wheel-01有143个物理文件/8,196,524 bytes，保完整压缩表示。

原三个作者、源码/安装执行均释放；所有 import/collection/数值/pytest/离线分析/lint/formatter/build/install/profile 在登记单一 owner 与 fresh entrypoint-aware guard 后执行。上一批证据和所有首失败不重审、不恢复旧 session、不全解压。

保留三个原 K1 未跟踪文件、用户 Torch、当前 venv/pins、原作者稿和旧证据 Junction。无 remote CI、push/mainmerge、发包、环境清理或 P7/P8 扩域。本批关闭不等于整个阶段 complete。
