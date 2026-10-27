# P6.5 多域点图小负载观测

2026-10-28，环境沿用[基础测量](p65-foundation.md)：Windows11、Python3.12.14、NumPy2.5.4、固定tyx SHA、RTX4060 Laptop/驱动566.07。图路由主机FP64；CUDA底层实际FP32算术/FP64状态。源码检查点`c89fa81`加导出/测量器变化，原始JSON含参与代码逐文件哈希，计时前后核对一致。

执行`python scripts/benchmark_domain_point_graph.py --mode numpy --output artifacts/graph-numpy.json`后再执行同命令`--mode cuda`。无其他数值任务并行。每例3个独立单节点域，逆质量1/2/4，2条边共享第三域，每调用完整响应＋全部候选＋解析断言；不可变快照重复查询，不推进时间。各1首调用、2预热、3×20计时调用，共120计时调用全部接受、失败0。

| 模式 | 三批中位 ms | 三批p95 ms |
| --- | --- | --- |
| NumPy | .458,.430,.438 | .513,.511,.536 |
| tyx CUDA | 8.016,7.577,7.235 | 10.444,12.611,8.405 |

原始记录：[NumPy](p65-graph-numpy.json)、[CUDA](p65-graph-cuda.json)。每个CUDA域的完成launch在查询期间增长；计时包含主机结果返回及独立断言。构造/预测在计时外，首调用不称冷JIT。没有分离路由、复制、backend或验证占比，没有测字节、峰值内存、完整step、Warp或计算加速比。

两机器人共享祖先＋布域的功能验证另在三种实际tyx模式通过，不能把本表的单节点性能当成两机器人负载。后续在真实接触图中记录几何/关节/接触数与一次全图迭代成本，再对[PERF-008](../PERF-008-surface-foundations.md)作归因；当前不优化，也没有定位新的tyx依赖缺陷。
