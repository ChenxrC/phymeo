# N4b诊断、积分状态观测、ROB2a与凸体候选发布

2026-11-11，起点847b8ee3。沿[持续计划](../superpowers/plans/2026-11-11-newton-parity-continuous.md)继续，无新截止。下面是本批执行前合同，未代表通过。

| 节点 | 难度 | 交付与实际验收 | 停止边界 |
|---|---|---|---|
| N4b已存学习诊断 | 中高 | 三任务九固定身份的只读完整证据诊断入口；独立合成正负测试；全部九份实际分析 | 不重新训练/改原门；未保存梯度等字段明确UNKNOWN；最终给可检验原因和后继协议 |
| 显式积分状态观测 | 中高 | 新immutable积分状态类型；TYX公共integration_snapshot；VecEnv显式full/integration选择和checkpoint绑定；同输入实际状态/奖励/收据/部分reset、失败与恢复对照 | 默认full不变；不删物理step检查；integration不冒端点几何/传感/能量重算；保留full reference；不直接声称性能提升 |
| ROB2a首纵向片 | 高 | 官方来源对照与状态/执行器接口、多案例；优先第一阶activation到真实控制/接受/恢复的最小完整链路 | 先裁定确切文件范围；球关节/闭环/tendon等独立后继；不能只加未连接辅助类 |
| COL同候选停止与发布 | 中高 | 当前q/本次支持下界/同一simplex权重共同决定分离接受；新一致性和重复支持/预算测试；原六入口及影响矩阵 | 不增资源/改0.002法向门/恢复旧strictFW；一次有界假设，四Y未获实际通过前仍失败 |

原support/capacity/actuator三作者分别负责N4b、ROB2a、COL，父负责共享runtime/VecEnv与集成。作者可源码/测试编写和官方研究；全部import、pytest、数值/离线分析、静态、build/install/profile由父唯一串行执行，逐命令fresh guard。隔离树正常merge基线，不reset，保留全部旧WIP。新通用功能应有源码外安装资格；只文档或分析不重复构建。

积分状态观测只公开已接受q/v/act/ctrl/time/tick、外载、held control与原step receipt。它没有body/geom/endpoint-contact/energy/sensor字段，不能将缺失解释为零。原step中完整动力学/接触诊断照常运行。训练模式须进入稳定binding，跨模式checkpoint拒绝；全观测仍可显式调用。固定验证J/S/R同输入短连续路径和partial reset，选W8及W128的有界计时，正式测量不混入编译/预热，完整结果如实保存。

本批不重训已通过S、不重跑旧完整N4a/旧审计、不重启G1长窗、不清理环境/Junction、不远程CI/push/mainmerge/发包。两个R03和四Y原失败保留；不把当前片称完整Newton或P6.5。
