# Phymeo 当前续作入口

更新：2026-11-12 13:28。[BALL-M1、设备M1、资产级多参与者装配批次](p65-parity-m1-assembly-20261112.md)已完成源码与安装资格，没有新截止。**整个P6.5/Newton功能对齐仍未完成**。本页只保留当前状态；旧阶段记录在[历史入口](p65-continuation-history-through-20261112-0035.md)，历史pending/deadline不覆盖本页。

## 当前入口

1. [八方向差异全景](phymeo-newton-panorama-2026-11-12.md)：当前已有、限域、缺口和优先级。
2. [本批实际结果](p65-parity-m1-assembly-20261112.md)：三条源码资格、六个安装消费者、首失败和证据。
3. [本批执行计划](../superpowers/plans/2026-11-12-parity-m1-assembly.md)：已关闭，原作者范围与共享接缝责任保留。
4. [统一编译架构](../architecture/unified-scene-compilation.md)：CPU MuJoCo编译、CPU/TYX/显式Warp与resident路径。
5. [07:20架构快照](../architecture/phymeo-newton-parity-2026-11-12.html)：历史图，尚未绘入本批直接BALL电机/资产装配/设备M1；当前能力以本页和全景为准。
6. [07:20批次记录](p65-parity-until-0720-20261112.md)、[原后继设计](../superpowers/plans/2026-11-12-newton-parity-next.md)：可复用历史，不重做已经完成的开发。

不先通读历史heartbeat或重做旧hash审计；需要具体已验范围时定位相应报告。

## 已交付，可直接复用

- **BALL-M1新增**：统一native JOINT moment映射连接固定BALL/HINGE/SLIDE直接电机、同次广义力/运动和恢复；47新源码节点、9兼容节点，解释/CUDA安装各56worldsteps通过。直接BALL目标限JOINT/NONE/FIXED/BIAS_NONE、na0、无接触；不是FREE+BALL或BALL activation/contact电机。
- **资产级场景新增**：ParticipantSceneDeclaration/compile_participant_scene显式连接四类resident树、完整身份/材料对/控制与world恢复。HOST19+两后端各19，安装各24worldsteps通过。不把MuJoCo-TYX BALL导入resident标量树。
- **设备M1新增**：四类显式完整拓扑resident accepted FK→设备gather/bounds/BVH/pack→有限边界query。4HOST+67interpret+68CUDA、20兼容节点；安装各6worldsteps/3queries/12rows通过。保留HOST读回/校验，不是M2、CCD、实体内外或接触力。
- **本批统一包**：产品2f5f19d6，515产品source/wheel/installed字节相同；六个源码外消费者通过。消费者563e4798仅修FP32设备力到FP64公开力的精确提升检查，未改产品或误差门。
- **既有机器人/碰撞**：ROB2b、BALL1/A/C、COL1a/2/3按原报告复用。旧COL3仍是HOST当前快照输入；新M1是另一真实设备来源，不改变旧接口语义。BALL-C的scalar activation/contact资格不自动扩大到直接BALL motor。
- **既有训练/规模**：N5随机状态reset、N6 D=1..8控制窗口与GRU续接；N7 J8/D2/GRU、S128/D4、S256/D4实际更新及共同状态资格通过，不是新学习门。紧凑HOST镜像消除了普通步native构造/copy，不代表零HOST或Newton性能胜出。
- **既有诊断**：N8 J/R/S共768真实转移、0新Adam更新，297条同次保存审计通过；J/R actor范数裁剪活跃，S未给区分原因，不能据此宣称因果或自动改超参。

旧产品1ce57489/509成员与首包结果保持原版本资格，不倒灌到本批515成员；新包只执行明确新增/受影响范围。

## 后续从这里开始

1. 机器人：沿[十组组合合同](p65-robot-next-compositions-20261112.md)选择直接BALL电机与接触的下一有限组合，保原力/运动共同发布、独立native对照和多模型测试。更广activation、FREE+BALL与limits/dry分开准入，不能一次删除限制。
2. 场景/碰撞接缝：明确资产from_asset缺少的完整物理拓扑映射，连接新场景装配与M1的真实设备来源。当前两个模块分别通过，不能称已经组合通过。
3. M2：按[设备来源合同](p65-device-mesh-source-next-contract.md)先冻结真实deformable accepted/candidate vertex bank、producer提交与lease；随后复用M1查询。不能以HOST手写两帧伪装真实变形来源。
4. 训练诊断：只沿[原保存梯度/Adam历史](p65-credit-diagnostic-next-decision.md)比较虚拟更新；尚未执行，不重新采样或自动改clip/lr/追加128更新。继续独立排串行槽。

按多案例→缺失通用模块→受影响验证→源码外消费者推进；不只追加单个shape pair或单个Panda/G1案例，不重复同源绿套。

## 保留的失败和能力缺口

J/R原学习门、G1原46.29秒参考XY门及完整131.48秒动作、R03两项小净加速度门仍未通过。B-point touch-SE选定法向与曲面见证不相容，候选隔离未合入。M2、通用CCD/持续载荷/摩擦历史、更广机器人组合、批量传感、跨域、密集布/体软体、杆缆与MPM/流体见全景。局部模块通过不替代这些验收。

## 所有权与证据

父：`C:/Users/38347/.codex/worktrees/p65-parallel-integration/phymeo`，分支`codex/p65-parallel-modules`，本批本地提交未push。三名原作者已释放，当前无项目执行owner或运行中的批次。后续仍用隔离worktree作者、父唯一共享接缝owner；所有import/collection/数值/pytest/离线分析/lint/formatter/build/install/profile由唯一串行owner在fresh entrypoint-aware guard后执行。

本批证据：`artifacts/p65-parity-m1-assembly-20261112-01`；新包与消费者归档`installed-wheel-01`共72文件，`installed-close.json`绑定515产品/六消费者/首失败。旧07:20证据仍在`artifacts/p65-parity-until-0720-20261112-01`，不重审、不恢复旧session、不全量解压。

保留三个原K1未跟踪文件、用户Torch、当前venv/pins、旧证据Junction和所有首失败。无remote CI、push/mainmerge、发包或P7/P8扩域。本批关闭不等于整个阶段complete。
