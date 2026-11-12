# 2026-11-13 功能补齐批次交付

本页收口[到07:30的计划](../superpowers/plans/2026-11-13-feature-completion-until-0730.md)。功能、受影响兼容、安装及归档在截止前完成；没有为占满剩余时间启动新的大域。当前已验产品为 `23757ae6a0d2a5ed3c6b15b0ec762bfd571e3b75`，后续为测试、文档与结果归档提交。整个P6.5及Newton功能对齐未完成。

## 本批实际交付

| 方向 | 增量与验收结果 | 仍未覆盖 |
| --- | --- | --- |
| 公共场景到刚柔响应 | D5公共声明连接真实节点、来源核验、有限球/胶囊查询、双向载荷与共同发布；M-B1进一步支持同world多个独立浮体。新源码28节点、容量4节点及旧兼容各8通过；安装interpret/CUDA各68worldsteps、132participantsteps | 每participant仍ordinary floating单body。S8实际4world/8participants，另验mixed单/多body；不是关节机器人刚柔、全摩擦或CCD |
| 机器人接触与执行器组合 | R3-N法向、R3-P pyramidal、R3-A NONE/FILTEREXACT JOINT接入FREE+BALL/R/P原力学。R3-A117新源码＋6兼容通过，第三包每后端352worldsteps通过 | 标量limits/DOF dry的BALL树组合、BALL角限位、更多接触维数/形状；不是G1完整舞蹈通过 |
| 异构功能规模 | G2a原8模型/16world域：W1/W8/W16源码及W16/B3、W16/B8安装资格。独立world的控制、力、运动和边界操作完整保留 | 无跨桶碰撞、W32及更大异构资格；没有同模型同输出Newton性能结论 |
| 恢复失败的资源归属 | G2-R0保留验证主错与候选cleanup错、强持有未完成关闭的候选、保留旧发布。178预期RED→185源码12GREEN→193安装12GREEN，真实retry/后继/关闭完成 | 不重试未知native free；不等于完整allocation manifest或生命周期资源预算 |
| 材料耗散 | D6-A公开可选轴向速度耗散，D1/D3/D5接线及独立算术/反力/事务资格；两后端安装各44worldsteps取得功能资格 | 原严格弹性精度失败单列；轴向耗散不是所有模态衰减，不能推导持续稳定支撑 |

具体输入、原数值门、失败及限定范围见[功能验收矩阵](p65-feature-acceptance-matrix-20261113.md)，不要把上表计数相加成总体完成比例。各包只验证声明的消费者；没有重跑全部历史功能。

## 本批没有通过的应用

原D4无阻尼N1的interpret/CUDA均完整推进3秒、3072步，但两个预定稳态窗均失败。新的D6-B `c=4096 N s/m`应用也完整推进同一粗网格N1：

- 两窗支撑力均值21.4613 /19.3665 N，目标12.8–19.2 N。
- 刚体法向速度RMS约0.3044 /0.2136 m/s，节点中心约0.1652 /0.1175 m/s，门为0.1 m/s。
- 入层、卸载、离层、再接触及逐步几何/载荷/算术/连续长参考完成，但这些不能抵消稳态失败；另7项原严格短速度门失败独立保存。

因此它是实际持续功能缺口，不能仅用“允许浮点误差”解释。按预登记停止该系数的CUDA及其余格，不扫描阻尼、不挪窗口、不改质量或阈值。保存对照图已完成；[后继短实验](p65-sustained-support-next-experiments-20261113.md)将材料模态、接触共同/相对运动、步内离散分开检验，当前只是待冻结实验，尚无新根因结论。

## 包与交接入口

- 第四包534产品成员在source/wheel/installed完整同字节，parity为BYTE_IDENTICAL且issues为空。wheel SHA256：`859825776bfd14467aa6e4f6ff2ecbaa6430a3a947f7029d16352eed9f06e3dd`。
- 本包M-B1、S8/M3、R0消费者在源码外实际运行，解释/CUDA或对应完整双端矩阵通过；清理结果完整。原D5/R3/G2/D6资格按其各自版本复用。
- canonical证据为 `artifacts/p65-feature-until-0730-20261113-01/installed-closed-archive-04/wheel-04`，33文件；前三包在同级`installed-closed-archive-01`。原件与完整压缩轨迹保留，不重解压或重复归档。
- [当前架构HTML](p65-functional-architecture-20261113.html)、[架构说明](p65-feature-architecture-draft-20261113.md)、[八方向全景](phymeo-newton-panorama-2026-11-12.md)、[M-B1公共用法](p65-multibody-deformable-usage-20261113.md)、[续作入口](p65-continuation.md)。HTML可离线打开，已做浏览器布局检查。

## 下一批按依赖拆分

| 优先节点 / 难度 | 交付与测试标准 | 分工与停止边界 |
| --- | --- | --- |
| M-B2 / 高 | 非根sphere/capsule连杆与节点片共同响应；复用原FloatingRobotTree完整J/M，验根与关节反应、逐body力矩/功、连续多案例及晚失败/恢复/安装 | [现有设计](p65-multibody-deformable-next-design.md)的M-B1已关闭；只冻结M-B2。父持有共享来源/事务，原作者隔离实现和独立参考。若需新关节积分器或跨runtime共享bank，先停并明确接口 |
| BALL树标量约束组合 / 中高 | fixed/FREE BALL+R/P的标量软限位、native DOF dry及activation/contact混合；完整EFC type/identity/J/R/aref/bounds/force和独立连续参考 | [源码与checker调查](p65-ball-constraint-next-design.md)已完成。复用既有行核；不能套只懂contact type6的R3-A checker。遇新行数学/solver/ABI需求即拆模块，不顺带开发BALL角限位 |
| G2b资源预算 / 中高 | 原8/16域先补真实allocation描述、alias、候选/退休/UNKNOWN、边界重叠预留；恰好够/少1字节、真实失败后继；再另冻W32 | [资源设计](p65-heterogeneous-resource-next-design.md)中的R0已关闭。原作者可做不交叉文件，父接公开预算与聚合；登记payload不叫完整显存峰值。性能比较排在功能资格之后 |
| 持续支撑短实验 / 小至中 | 固定短试样区分轴向模态、共同/相对运动和接触离散；独立预测与反例能否互相证伪 | 先冻结四组最小试验，不重新跑旧3秒格或试第二个c。只有新证据支持时才选择材料/接触模块 |

前三线可以并行编码，所有项目import、测试、数值、静态工具、构建和安装仍由父唯一串行执行。新合同须先列公共接口、多案例、负控、原数值门及文件owner；不重复旧已绿组。相同原因两次诊断仍未缩小，或约60分钟无新证据，固定最小反例并转独立节点。

本轮未调用远程CI、push/mainmerge或发包，未更改pin、环境或证据Junction；三份无关未跟踪草稿保留。全部项目执行槽和原作者任务已释放，没有新增周期任务。接手时从当前续作入口与Git/WIP开始，不复活旧heartbeat或历史运行session。
