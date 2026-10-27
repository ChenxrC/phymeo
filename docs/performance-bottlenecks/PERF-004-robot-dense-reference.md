# PERF-004：机器人稠密参考的重复计算

- 状态：阶段时间已测，重复调用结构已确认；细项归因与优化未开展。
- 里程碑：P3，2026-10-27；提交 `1b0d1c7`。
- 环境、配置、原始记录及复现：[P3 性能](measurements/p3-robot.md)。

## 问题及代码位置

8 自由度无约束完整 step 中位数 8.154–8.335 ms，预测阶段中位数 6.489 ms；
这是一份小规模正确性参考的观测，不是相对其他引擎的性能差距。

- [RobotWorkspace.predict](../../src/phymeo/solvers/robot.py)：获取 applied/bias 后再调用 forward_dynamics，再求质量矩阵记录残差；部分几何/载荷重复。
- [机器人动力学](../../src/phymeo/dynamics/robot_dynamics.py)：mass、bias、COM Jacobian 各自评估 FK；质量矩阵稠密组装与求解。
- [RobotPointPort](../../src/phymeo/coupling/robot_port.py)：位置、速度、response 每次由拥有的快照重算；该端口尚未做单独吞吐测量。

## 证据和后续指引

调用结构可由代码确认，但没有 profiler 将预测时间拆成 FK、重复副本、矩阵、Python 循环和门禁百分比。
先在同一冻结状态对完整 step 做调用计数/阶段 profiler，并对 1/2/8 自由度及 P5 活动接触数扩展复测。
之后可比较子步内只读几何/质量缓存、共享动力学评估，以及独立 CRBA/ABA 路线；本轮不先实现这些优化。

缓存必须跟随 q 和绑定模型更新，接触每次改变 q 后不得复用旧质量/雅可比；隐藏持久状态不得绕过全局回滚。
任何优化需重跑解析和 MuJoCo 对照、非零零位旋转/移动轴、共享祖先反作用、失败恢复和完整 P5 轨迹。
这属于 phymeo 参考与集成开销，不是 tyx 实现问题，不登记为 tyx 性能缺陷。
