# PERF-014 材料投影路径记录

状态：修复后源码dc245f7的[串行观测](measurements/p65-material-path.md)1800次全部成功；调用结构已确认，细项占比未定位，不做优化。

观测事实：`TyxNodeMath.project_distances`的记录分支使用traced kernel，状态与乘子保持bitwise一致；每sweep新增记录缓冲上传和输出下载，launch数不变。`path_diagnostics`创建owned readonly快照；参与者/SceneStepper的外层诊断复制可能带来额外开销。本微型workspace测试包括材料诊断复制，但不包含完整SceneStepper外层，也没有单独测量复制成本。

待证实热点：多边、多sweep记录量线性增长；逐轮分配与传输；重复诊断快照复制。`trace_bytes`只统计帧有效载荷，不代表峰值内存。测量顺序固定off→on，跨backend原精度/迭代数不同，不能把端到端差异归因为tyx kernel性能缺陷。

后续定位：在代表性材料网格与完整重放中分项profile分配/复制/传输/kernel，分离第一次编译、缓存热态和长期资源。只有完成记录完整性、开关状态逐位一致、零分母哨兵、固定点转换与事务检查后，才评估批量缓冲或压缩；不得删记录或改物理容差换取速度。未测Warp，不声明差距；TYX-003另行跟踪，旧依赖无修复不重跑C++长压力。
