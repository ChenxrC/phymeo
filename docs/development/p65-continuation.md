# Phymeo 当前续作入口

更新：2026-11-12 06:10。当前批次按用户要求执行到 **北京时间 07:20**；本页只保存当前状态，旧阶段记录已移至[历史入口](p65-continuation-history-through-20261112-0035.md)。历史 pending / deadline 不覆盖本页，不重新派发已验收任务。

## 先读这四个入口

1. [当前八方向差异全景](phymeo-newton-panorama-2026-11-12.md)：已具备、限域、未完成与后续优先级。
2. [本轮实际结果](p65-parity-until-0720-20261112.md)：真实运行、失败与证据编号。
3. [到07:20的计划](../superpowers/plans/2026-11-12-newton-parity-until-0720.md)：预登记范围、停止规则。
4. [统一编译架构](../architecture/unified-scene-compilation.md)：CPU MuJoCo编译、CPU/TYX/显式Warp与resident路径的区别。

不要先通读历史heartbeat或重做旧hash审计；仅在当前任务需要具体已验范围时定位原报告。

## 当前已完成

- ROB2b：自由根/接触或混合约束与合法scalar NONE/FILTEREXACT组合，源码及首包安装通过。
- BALL1/A/C：固定BALL/HINGE/SLIDE完整动力学、合法scalar执行器、primitive condim1/3接触已连接。BALL-C新52节点、受影响6项、第二包两后端各64worldsteps通过；不含直接球motor、FREE+BALL、limits/dry。
- COL1a修复原Y法向/资源回归；COL2有限segment–triangle距离与COL3公共boundary query两后端及首包安装通过。查询来源目前仍是HOST快照，不能叫真实device mesh source、CCD或接触力。
- N5/N6：可复现随机状态reset、D=1..8控制窗口与完整子步/失败前缀、真实GRU续训。N7 J8/D2/GRU、S128/D4、S256/D4各full/integration固定更新与共同状态通过，不能代替学习门。
- 紧凑HOST镜像已消除普通步native构造/copy，原资源预算保持；没有Newton等输出性能优势结论。
- 最终第二包产品快照 **1ce57489**：509产品source/wheel/installed逐字节相同；新版评价器安装CUDA192worldsteps、BALL-C解释/CUDA各64通过。首包七消费者资格按原快照复用。

## 明确保留的缺口

- J/R原预算学习门仍失败；J偏置居中假设已停止。N4e控制可达不等于PPO学会，不加预算或种子追绿。
- G1原46.29秒参考XY门及完整131.48秒动作未通过；R03两项小净加速度门仍开放。
- B-point原touch-SE选定法向与曲面见证不相容，候选未合入；不悄悄接C-point载荷。
- [M1/M2设备来源合同](p65-device-mesh-source-next-contract.md)：M1刚性设备来源尚未实现；M2先缺真实deformable accepted/candidate vertex bank。两者不合称“mesh已支持”。
- 更广机器人组合、传感、跨域编排、密集布/体软体、杆缆、MPM/流体仍见全景；不以Panda/G1/递布单案例替代通用案例库。

## 执行与所有权

父仓库：`C:/Users/38347/.codex/worktrees/p65-parallel-integration/phymeo`，分支`codex/p65-parallel-modules`。父拥有既有共享接缝；原三个隔离作者只按明确文件范围编码/研究。全部项目执行严格串行，由父fresh entrypoint-aware guard后运行；不杀外来执行、不重启旧session、不重跑同源已绿组。当前物理测试/安装已结束，父继续收尾与下一批任务交接，尚未到07:20批次关闭点。

本轮证据：`artifacts/p65-parity-until-0720-20261112-01`。首/第二包完整输出分别在`installed-wheel-01`、`installed-wheel-02`；原外部安装目录保留，不重新全量解压旧压缩证据。所有首失败与未通过假设保留。

不得覆盖三个原K1未跟踪文件、用户Torch、当前venv/pins或旧证据Junction。无remote CI、push/mainmerge、发包；不扩P7/P8。当前截止是本批截止，**整个P6.5/Newton对齐未完成**。
