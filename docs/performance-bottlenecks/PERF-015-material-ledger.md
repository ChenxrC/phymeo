# PERF-015 材料原式账本的主机认证成本

状态：[900次串行观测](measurements/p65-material-ledger.md)完成，原生求解与账本构造/认证阶段已分别计时；细函数占比尚未profile，本轮不优化。

观测事实：`validation/material_path.py::MaterialPathLedger.require`与`_material_path_math.py`在host用Fraction、平方根包络、逐边原式累计预算和物理报告核对native候选；支持owned snapshots。计量包含这些门禁，没有关闭认证来冒称可交付速度。NumPy与tyx本来使用不同容差/迭代数，不能把端到端差异归为设备kernel缺陷。

待验证热点：大量Fraction对象和sqrt包络、逐边/逐节点Python循环、矩阵快照与报告复制。下一步在代表网格/接触重放中分项profile，分清求解器/传输/账本/事务成本，再评估精确算术缓存或批量验证；必须保留大坐标、累计误差、固定点例外、精确请求与设备存储真实性门禁，不能用tol放宽换速度。

当前是Phymeo集成与独立认证成本，没有新tyx实现缺陷证据，未测Warp。首次调用缓存不是冷启动；峰值内存和C++长期未测，TYX-003仍开放。
