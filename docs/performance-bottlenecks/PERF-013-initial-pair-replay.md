# PERF-013 步初机器人—节点双向重放

状态：最终aef9b257源码的[串行观测](measurements/p65-initial-pair.md)已完成，NumPy/interpret/CUDA各300次全部接受；调用结构已确认，分项占比未定位，本轮不优化。

`solvers/initial_robot_node_contact.py::_loads/_evaluate/_audit`按节点和机器人共享祖先合并相反载荷，再完整重放native程序。`_particle_replay_ledger.py`对所选自由粒子域每个分量精确核对真实输入、动力学、位移与动能；机器人沿用精确M0账本及仿射范围检查。接触迭代包含host几何/有理数与native后端，不是纯GPU求解。修复后的`_contact_replay_audit.py`还包含精确投影及有理平方根包络；当前每次2次replay，细项成本没有单独profile。

待证实热点：同一node domain较大而少数接触时全域精确检查的代价；多个接触/共享祖先导致的完整重放次数；粒子与机器人同步、状态/history复制和workspace关闭。需先用代表规模profile区分这些成本，再讨论缓存或批处理，不能删门禁或忽略交叉响应。

与PERF-012机器人固定平面负载不同，端到端差值不能直接当作tyx实现损耗或加速比。未做Warp测量。Windows C++长期能力继续由[TYX-003](../../tyx_flaw/TYX-003-windows-cpp-library-lifetime.md)单独跟踪，有限短测不关闭该问题。

未来改变求解器/积分或添加材料账本时，必须重测单转子真实Lz、初态共点/末穿透拒绝、节点原式、冻结总能量及完整事务，不能仅重跑吞吐计时。
