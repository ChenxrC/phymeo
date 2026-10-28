# PERF-011 动态平面外层的完整重放与数值审计

状态：已确认调用结构；端到端串行小负载计量见[测量报告](measurements/p65-plane.md)。尚无profiler细项归因，不宣称已定位占比或开展优化。

## 观测与代码位置

`solvers/particle_plane_contact.py::_evaluate`在每次接触候选变化时从完整原快照重跑base程序；`_integrals`按节点聚合，`_input_velocity_error`和`_prediction_roundoff`对原始/实际输入和精度预算做有理数审计；`_audit`复核最终几何、材料速度、接触功及摩擦。一次外层step可有多次完整SceneLoadReplay，而不止一遍kernel。

`runtime/load_replay.py`和`runtime/scene_stepper.py`继续进行完整输入/state/history快照、事务检查、workspace创建/关闭及结果所有权处理。actual tyx还经现有participant、节点与机器人kernel、传输和主机验证。每次完整step的launch/replay计数和原始时间样本保存于测量JSON。

这是一关节＋一节点＋一平面的微型固定输入负载，不代表网格、机器人接触或双臂吞吐。冷缓存JIT、纯kernel、分项主机时间和峰值内存尚未单独测量，不能把端到端差值当作tyx实现损耗或Warp差距。

## 待验证假设与后续方向

- 完整base重放可能主导接触次数/外层迭代增长后的成本；先测节点数、活动接触数、材料刚度与重放次数的关联，再决定是否需要专用Jacobian/批量响应。
- 有理数审计、每节点匹配contact行、重复快照可能成为host成本；用profiler分离之后再考虑预索引、受控误差界或复用。不能删除实际输入/候选物理审计来取得加速。
- 短小tyx工作负载可能更多受启动/传输/资源生命周期影响；需公开API独立复现并剥离Phymeo开销后，才进入tyx_flaw。本轮未做等价Warp测量。

正确性风险：重用可变状态会污染历史；漏加同节点载荷或重复冲量；缓存旧构型M/J；把浮点同源残差当作精确账本。任何优化必须保持正常/低mu失败对照、末子步/关闭失败回滚、尺度反例、实际后端执行及全部原容差。

## 已知运行阻塞

[TYX-003](../../tyx_flaw/TYX-003-windows-cpp-library-lifetime.md)仍使Windows C++长期运行未闭环。此次4项实际C++短功能通过只覆盖有限次数；没有新依赖修复，不重复原长期压力失败。必须在tyx修复后按原规模重测，不能作为普通延后性能优化忽略。
