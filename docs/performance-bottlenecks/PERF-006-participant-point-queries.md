# PERF-006：组合接触中重复端口快照与几何/质量查询

状态：真实阶段时间已测，调用结构已确认；逐函数热点和内存占比未测。
证据基线 `77e9532`，详见[测量与原始 JSON](measurements/p5-participants.md)。

## 现象与位置

双节点域共享一个关节的解析微基准需要 10 次位置 sweep，完整 step 中位数约 49 ms，
位置阶段约 42.8 ms。单对只需一次 sweep、完整 step 约 5.7–6.2 ms。
该对比同时改变接触数与收敛次数，不能当作纯域数量缩放，也不能称为 tyx 性能问题。

最终完整活动接触测量补充见[P5全场景记录](measurements/p5-full.md)：S2/S3/S4各300个完整step均通过独立审计。
S4完整step批次中位数约1.115–1.161s，每子步位置阶段中位数515.631ms；每子步12次位置sweep。
这加强了位置迭代阶段占主要时间的证据，但没有替代逐函数profiler，仍不把复制/几何/质量分别排定占比。

- [NodeRobotConstraints](../../src/phymeo/constraints/node_robot.py)：每次 sweep/metrics 重新读取几何和端口，活动接触再次求质量响应；逐事件留存前态/增量。
- [参与者](../../src/phymeo/coupling/participant.py)：返回有所有权的端口、把世界点映射回连杆，位置载荷再次计算当前 J/M。
- [RobotPointPort](../../src/phymeo/coupling/robot_port.py)：构建模型/状态副本，查询 FK/J，应用载荷求解质量矩阵。
- [TetrahedralPointPort](../../src/phymeo/coupling/tetrahedral_port.py)：节点端口仍验证整个材料域；完整 S3/S4 成本需后续场景观测。

已确认代码存在重复计算；尚未使用 profiler 给复制、指纹、几何、线性代数或 SVD 排名。
本轮不优化，也不移除验证或诊断来给出更快的可交付数字。

## 后续 agent 指引

先冻结源码和场景，记录域/关节/节点/配对数、实际活动接触与 sweep 数，
分别 profiling 端口构造、M/J、材料域、完整诊断复制。
再评估一个候选版本内的只读几何/质量缓存，或批量端口响应；
配置变化必须失效，重复节点/共享祖先仍须合并与交叉响应。
优化后重跑独立解析解、偏心载荷、所有域失败回滚、完整数值轨迹与实际旧 CUDA 回归。
不得把物理状态移到不可回滚的缓存，不得直接将逐对更新并行化而改变结果顺序。
