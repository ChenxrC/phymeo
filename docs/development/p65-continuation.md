# Phymeo 当前续作入口

更新：2026-11-12 12:05。当前正执行用户授权的[下一批](../superpowers/plans/2026-11-12-parity-m1-assembly.md)：BALL-M1、设备 M1、资产级多参与者场景，无新截止。07:20批次已收束；**整个P6.5/Newton功能对齐仍未完成**。本页只保存当前状态，旧阶段记录在[历史入口](p65-continuation-history-through-20261112-0035.md)，历史pending/deadline不覆盖本页。

## 当前入口

1. [可视化连接架构](../architecture/phymeo-newton-parity-2026-11-12.html)：两条物理路径、独立查询、训练链路及未连接部分。
2. [八方向差异全景](phymeo-newton-panorama-2026-11-12.md)：已有、限域、缺口和优先级。
3. [本轮实际结果](p65-parity-until-0720-20261112.md)：实际执行、首失败、证据编号。
4. [下一批行动计划](../superpowers/plans/2026-11-12-newton-parity-next.md)：三条无交叉作者线、文件责任、验收及停止条件。
5. [统一编译架构](../architecture/unified-scene-compilation.md)：CPU MuJoCo编译、CPU/TYX/显式Warp与resident路径。
6. [本轮预登记计划](../superpowers/plans/2026-11-12-newton-parity-until-0720.md)：已关闭批次的范围与变更历史。

不要先通读历史heartbeat或重做旧hash审计；仅在当前任务需要具体已验范围时定位原报告。

## 本轮交付，可直接复用

- ROB2b：自由根/接触或混合约束与合法scalar NONE/FILTEREXACT组合，源码和安装消费者通过。
- BALL1/A/C：固定BALL/HINGE/SLIDE完整动力学、合法scalar执行器和primitive condim1/3接触已连接。BALL-C新52节点、受影响6项、最终包两后端各64worldsteps通过；不含直接球motor、FREE+BALL、limits/dry。
- COL1a闭环原Y法向/资源回归；COL2有限segment–triangle距离和COL3公共boundary query两后端及安装通过。当前来源仍是HOST快照，不能叫真实device mesh source、CCD或接触力。
- N5/N6：可复现随机状态reset、D=1..8控制窗口、完整子步/失败前缀、真实GRU续训。N7 J8/D2/GRU、S128/D4、S256/D4的full/integration固定更新与共同状态通过；不是学习门。
- 紧凑HOST镜像已消除普通步native构造/copy，保原资源预算；没有Newton等输出性能优势结论。
- 最终产品快照 **1ce57489**：509产品source/wheel/installed字节相同；新版评价器安装CUDA192worldsteps、BALL-C解释/CUDA各64通过。首包七消费者按原快照复用。其后只有诊断example/tests和文档，无产品source增量。
- N8一次有界诊断：J/R/S各32步W8，768真实转移、0新Adam更新；297条同次保存审计通过。J/R actor范数裁剪活跃，S未给区分原因；不能据此断言裁剪导致弱学习。

## 下一批从这里开始

- 机器人：按[十组案例](p65-robot-next-compositions-20261112.md)补通用JOINT moment表，先直接scalar/BALL NONE电机；更广activation/contact组合分步资格。
- 碰撞：按[M1公共API父裁定](p65-m1-api-freeze-candidate.md)和[17组案例](p65-m1-case-library-20261112.md)，S0首个tests/schema提交固定buffer布局/字节预算，然后开发device gather→bounds→BVH→pack。父补batch runtime lease关闭保护。**M1尚未实现**。
- 场景：显式多participant声明连接已验ROB1公共接口，不复制积分器或私有bank。
- 训练诊断后继：只用[同次保存的梯度与Adam历史](p65-credit-diagnostic-next-decision.md)比较虚拟更新；尚未执行，不能自动改clip/lr或再跑128更新。与产品开发分开排唯一串行槽。

## 保留的失败和能力缺口

J/R原学习门、G1原46.29秒参考XY门及完整131.48秒动作、R03两项小净加速度门仍未通过。B-point touch-SE选定法向与曲面见证不相容，候选隔离未合入。M2还缺真实deformable accepted/candidate vertex bank。通用CCD/持续载荷/摩擦历史、更广机器人组合、传感、跨域、密集布/体软体、杆缆与MPM/流体见全景；不以单个Panda/G1/递布案例替代模块资格。

## 所有权与证据

父：`C:/Users/38347/.codex/worktrees/p65-parallel-integration/phymeo`，分支`codex/p65-parallel-modules`，本轮提交均在本地，未push。父拥有全部既有共享接缝；原三个隔离作者已在新分支开始本批纯编码；父单一串行执行。当前证据根为 artifacts/p65-parity-m1-assembly-20261112-01。仍沿用原作者，独立worktree可并行编码，所有import/collection/数值/pytest/离线分析/lint/formatter/build/install/profile严格串行，由父fresh entrypoint-aware guard后运行。

证据根：`artifacts/p65-parity-until-0720-20261112-01`。两份安装归档在`installed-wheel-01/02`，N8原记录在`301-n8-fixed-credit-cuda`，独立保存审计在`302-n8-saved-audit`；最终关闭收据为`batch-close-0720.json`。旧同源绿套/审计不重跑，不恢复已关闭session，不全量解压旧证据。

保留三个原K1未跟踪文件、用户Torch、当前venv/pins、旧证据Junction和所有首失败。无remote CI、push/mainmerge、发包或P7/P8扩域。本批关闭不等于整个阶段complete。
