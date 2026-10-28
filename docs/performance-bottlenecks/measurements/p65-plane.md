# P6.5 动态平面外层串行小负载观测

产品`c6390f4`，计量head`c021885`，111个源码文件hash `2df1ec407cae54a8b24d6fccac92aaf5a6725526c1b5dacb159f41fda8a10e1c`。所有数值测试进程退出后依次执行NumPy、interpret、CUDA；各首调用1次、预热2次、3×100次测量。900次计量全部接受，含首调用/预热共909次成功。Windows C++长期仍受TYX-003阻塞，本轮不重复已知失败压力，不能填成功分位数。

环境：Windows11、Python3.12.14、NumPy2.5.4、tyx0.1.0a0 pin99f6df7c0fd8fdeef77f97df83914da27b64b7ee；i9-13980HX、RTX4060 Laptop、driver566.07。主机映射/残差FP64，有理账本/范围审计；tyx原计算FP32。单进程顺序调用，无批量环境，不人为更改系统线程设置。

负载：一个自由转动关节q=.2/v=.3和一个节点，节点受法向20N压力与切向重力9.81；静止x=0平面，mu=.6，h=.01、外层tol2e-5。每次从同一初态完整step，自动求p=(.2,.0981,0)，节点末q/v为0；机器人自由推进到q=.203/v=.3。不是接触机器人，也不是完整时域递布。包含工作区、输入/快照、原程序重放、完整验证、关闭和数值断言；不含一次性模型/外层factory构造及最终factory.close。CUDA结果同步回传后停止计时。

| 后端 | 完整step中位数 ms | p95 ms | 首调用 s | 成功计量数 |
| --- | --- | --- | --- | --- |
| NumPy | 8.3477 | 9.3153 | .0338 | 300 |
| interpret | 96.9710 | 104.5975 | .2783 | 300 |
| CUDA | 214.2119 | 236.8049 | .5015 | 300 |
| Windows C++ | 未测本轮长期分布 | 未测 | 未测 | 长期门禁开放 |

每个调用均1次外层position_sweep、2次完整重放。interpret/CUDA各44个completed_launches；CUDA每次338次upload/88次download（为现有公开计数，未计字节）。正常报告只累加成功返回的重放，不能推断失败尝试的执行总数。最大归一化metric NumPy3.49e-16、tyx1.88e-7，后者包含保守prediction_roundoff预算，不能将其称为测得的纯物理误差。

原始样本：[NumPy](p65-plane-numpy.json)、[interpret](p65-plane-interpret.json)、[CUDA](p65-plane-cuda.json)。复现：`python scripts/benchmark_particle_plane_contact.py --mode <mode> --samples 100 --repeats 3 --output <new-json>`。每份记录保留逐次时间、执行计数、源码/脚本hash及失败保存逻辑。git_dirty来自报告/文档，三模式产品源码hash一致。

首调用有既有磁盘编译缓存，不是独立冷启动。没有测峰值内存、纯kernel、主机各阶段占比或Warp；不能把微型重放端到端差异归为tyx吞吐缺陷。已知依赖生命周期证据继续指向[TYX-003](../../../tyx_flaw/TYX-003-windows-cpp-library-lifetime.md)；本轮没有新依赖缺陷证据。后续定位方向见[PERF-011](../PERF-011-plane-replay.md)，不通过删除精确审计来优化。
