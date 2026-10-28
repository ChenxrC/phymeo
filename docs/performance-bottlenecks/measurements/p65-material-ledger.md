# P6.5 材料原式认证串行观测

源码 `07b23d6`，122文件hash `80831f04b59587c1805b141c26a73a659f4b5b0a399f0f5f954f4c7371de683d`；tyx `99f6df7c`。CPU i9-13980HX、RTX4060 Laptop、驱动566.07，Windows11/Python3.12.14/NumPy2.5.4。默认线程/亲和性，未调优；不属于独占硬件测试。

所有本任务测试和review探针退出后，NumPy→interpret→CUDA串行，每模式3次预热＋3×100次计量。每次由同一未固定3节点/2软边初态重新创建workspace、求解/诊断、构造账本并认证、检查状态未被改变和重复一致性、关闭资源。h=.1、零外力、compliance(.001,.002)。不是连续300步轨迹，也不是接触outer。

| 模式 | 接受 | 端到端中位 / p95 ms | 原生阶段中位 ms | 账本阶段中位 ms | sweep / 完成launch |
| --- | --- | --- | --- | --- | --- |
| numpy | 300 | 7.924 / 8.790 | 0.601 | 7.173 | [7] / [0] |
| interpret | 300 | 18.367 / 20.565 | 13.676 | 4.436 | [4] / [8] |
| cuda | 300 | 70.687 / 80.984 | 65.840 | 4.303 | [4] / [8] |

NumPy原容差1e-8、设备2e-5，分别保留各自原生迭代数；跨后端不是等精度加速比。原生阶段包含workspace构造、预测、迭代、速度重建和诊断；账本阶段包含owned构造与require。总时间另含一致性检查/关闭，因此不能用两个中位数相加替代总中位数。设备诊断/输出下载完成后计时，不是仅异步发射耗时。

首次调用使用已有缓存，不称冷JIT；源对象构建不在计时内。没有分项测kernel/传输/分配/所有权复制，没有峰值内存/大网格/长期轨迹/Warp或C++压力数据。完整CPU与CUDA套件并行执行的秒数仅是验证记录，不是性能结果。

原始数据：[NumPy](../../benchmarks/p65-material-ledger-numpy.json)、[interpret](../../benchmarks/p65-material-ledger-interpret.json)、[CUDA](../../benchmarks/p65-material-ledger-cuda.json)。[工程记录](../../benchmarks/p65-material-ledger-engineering.json)包含环境、源码、首次调用和计数。[PERF-015](../PERF-015-material-ledger.md)记录后续定位方向，未进行优化。

复现：`python scripts/benchmark_material_ledger.py --mode numpy --samples 100 --repeats 3 --output <新的JSON路径>`，顺序改为interpret/cuda；先结束本任务其他数值负载，输出路径必须不存在。
