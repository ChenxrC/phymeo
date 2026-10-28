# 材料投影路径串行观测

源码`dc245f74e2a2877c1137f46dabb4638367eb5c2a`，120文件hash`09a3bd9819cb75fdf5bf463adea9397c452cf07365039707e11a5e057c91d624`。Windows11/Python3.12.14/NumPy2.5.4；tyx0.1.0a0固定99f6df7c，实际RTX4060Laptop CUDA路径。未锁CPU亲和性，沿用进程默认线程设置。

数值测试、打包与审阅探针均退出后，按NumPy→interpret→CUDA串行执行；各模式先关闭后开启记录，没有随机交错。每配置首调用和2次预热独立记录，再3×100次计量，共1800次全部成功。每次从同一三个动态节点/两条非线性共享边快照重算，包含workspace建立、预测、材料求解、诊断所有权复制、bitwise一致性检查和关闭；factory构造不计。原始每次耗时/计数保存在[NumPy](../../benchmarks/p65-material-path-numpy.json)、[interpret](../../benchmarks/p65-material-path-interpret.json)、[CUDA](../../benchmarks/p65-material-path-cuda.json)。

| 模式 | 记录 | 中位ms | P95ms | 成功计量 |
| --- | --- | ---: | ---: | ---: |
| numpy | 关 | 0.366 | 0.393 | 300 |
| numpy | 开 | 0.567 | 0.613 | 300 |
| interpret | 关 | 12.652 | 13.744 | 300 |
| interpret | 开 | 13.450 | 14.788 | 300 |
| cuda | 关 | 36.682 | 45.107 | 300 |
| cuda | 开 | 40.786 | 51.628 | 300 |

NumPy原tol1e-8，7sweep；两个tyx模式原tol2e-5，4sweep，因此**跨后端不是等精度性能比**。同一模式开关两组的最终position/velocity/lambda/迭代次数持续逐位一致；tyx均8个completed_launches。CUDA关闭42upload/13download，开启46/17：每材料sweep多上传一个记录缓冲、多下载一个记录输出。记录本身的主机帧有效载荷NumPy2016B、tyx1152B，不包括复制/对象/设备临时量，不是峰值内存。

本数据是微型材料算子端到端观察，不是连续时域、机器人布料联合求解、夹持或kernel计时。组间顺序/调度差异未隔离，不能把中位差全归到某一行代码。未测Warp，不形成tyx实现缺陷或GPU加速比结论。首调用未清编译缓存，不称冷启动。Windows C++长期受TYX-003约束，本轮只做4项有界数值短测，不做无修复的长期重试。

候选成本与排查方向见[PERF-014](../PERF-014-material-path.md)，本轮不优化。
