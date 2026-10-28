# PERF-012 机器人动态平面的最终构型映射与精确审计

状态：调用结构已知、[900次串行完整小负载](measurements/p65-robot-plane.md)通过；无profiler细项占比，不宣称已定位最主要瓶颈。本阶段不优化。

`solvers/robot_plane_contact.py::_geometry/_evaluate/_audit`重复查询最终FK/J、将全部世界冲量映射至广义effort并重放完整native程序。`_robot_replay_ledger.py::FrozenRobotLedger`精确LDL、动力学/实际输入/运动学/能量审计，`audit_affine_geometry`对COM和接触点重复独立仿射重建。SceneLoadReplay及SceneStepper继续拥有完整快照/history、workspace关闭与回滚。

本次一个机器人移动关节、一个固定平面、两次重放的小负载，NumPy/interpret/CUDA各300次全接受，actual tyx都有真实执行计数。它与PERF-011的一机器人加一节点负载不同，不能直接取差值当优化或退化。原始时间、replay/launch/传输计数及源码见测量JSON。

待证实假设：重复host几何、有理数和完整native重放可能随接触数/深度/迭代数放大；分配、编译缓存、传输与本地计算占比未知。下一步先做代表网格/机器人规模的profile，再考虑跨候选复用不可变系数、批量映射或精确预算替代。必须保留同域交叉响应、最终J映射、材料速度、能量、仿射边界和历史事务，不得缓存旧构型或删除审计换速度。

这不是tyx实现缺陷证据，也没有等价Warp测量。Windows C++长期阻塞仍由[TYX-003](../../tyx_flaw/TYX-003-windows-cpp-library-lifetime.md)跟踪：本阶段短功能通过不关闭依赖问题，原长期规模待修复后复测。
