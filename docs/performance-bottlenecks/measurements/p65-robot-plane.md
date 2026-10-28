# 机器人固定平面：串行端到端观测

产品a4b1ad4，114文件源码SHA256 `230f2470e71b72a2098afb7a83e10db2170bbc4b05067ce757fa79434e751b14`。三组使用同一源码、脚本和配置；源码未变，采样时文档未提交所以git_dirty=true。全部数值测试/审阅探针退出后按NumPy→interpret→CUDA串行运行。

| 模式 | 中位数 ms | p95 ms | 首次调用 ms | 接受数 | replay / completed launches |
| --- | ---: | ---: | ---: | ---: | --- |
| numpy | 11.845 | 13.099 | 35.552 | 300 | [2] / [0] |
| interpret | 59.600 | 65.182 | 243.495 | 300 | [2] / [40] |
| cuda | 183.799 | 204.645 | 521.097 | 300 | [2] / [40] |

负载：一个2kg移动关节、一个固定平面，h=.01、tol2e-5、gravity9.81、mu0。每次从同一原状态求完整step，独立检查q/v0与冲量.1962；不是连续3s轨迹。计时包括workspace、native重放、精确动力学/仿射几何/能量审计、事务关闭和断言，不含一次性外层工厂构造及最终关闭。首次调用使用已有缓存，不是冷进程/冷JIT。2次预热后3×100样本，每模式303次含首次/预热均成功。

三模式都2次重放、1次位置迭代。actual tyx具有非零completed_launches；CUDA通过结果读取和物理断言确认完成后计时。固定微负载下的端到端耗时不能代表大网格加速比，也不是纯kernel性能。

原始证据：[NumPy](p65-robot-plane-numpy.json)、[interpret](p65-robot-plane-interpret.json)、[CUDA](p65-robot-plane-cuda.json)。包括每次执行计数、样本、源码/脚本哈希和失败字段。

未测：分项profiler归因、峰值主机/设备内存、冷缓存、旋转轨迹吞吐、机器人—布双向场景、CCD、Warp。Windows C++长期TYX-003未修复，故不重复压力失败，也不填写C++分位数。候选热点见[PERF-012](../PERF-012-robot-plane-replay.md)，本轮无优化。
