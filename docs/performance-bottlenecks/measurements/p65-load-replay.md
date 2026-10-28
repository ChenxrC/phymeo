# P6.5 固定区间载荷重放：串行观测

产品3f23378，源码109文件SHA256`bbc798c78fedc222192fdd557a72000d02f52f0caa0d13b3f5daf4a11bcbff0c`。i9-13980HX（24核32线程），Windows11、Python3.12.14/NumPy2.5.4；tyx0.1.0a0@99f6df7c，MSVC14.50.35717；RTX4060 Laptop、driver566.07。OMP/OPENBLAS/MKL线程环境变量均未设置，未单独锁定库线程数。缓存为可写`artifacts/tyx-cache`，TEMP/TMP为`artifacts/tmp`。

全数值测试进程退出后四模式串行尝试。每模式1次首调用、2预热，再3组×100次；NumPy、interpret、CUDA共900次计量全部接受。C++长循环失败，独立重开进程后另行完成CUDA；不把C++跳过或失败写成通过。

负载：1个质量2kg的移动关节＋1个逆质量1的节点、无接触/材料约束，h=.01s，原q=.2、v=.3；区间关节积分.04N·s、节点积分(.1,0,0)N·s。每次从同一初始状态重算，核验机器人v=.32/q=.2032、节点v=.1/q=.001和最终原始快照未推进。包含输入构造、完整原program各阶段、工作区创建/关闭、复制/校验和数值断言；不含一次性的SceneLoadReplay/模型构造及最终外层close。CUDA各kernel同步并将结果回传主机后完成计时。

| 后端 | 中位数 ms | p95 ms | 首调用 s | 成功计量数 |
| --- | --- | --- | --- | --- |
| numpy | 2.6934 | 3.0861 | 0.0254 | 300 |
| interpret | 29.2907 | 34.5142 | 0.2625 | 300 |
| cuda | 106.0339 | 130.0459 | 0.5275 | 300 |
| cpu_cpp | 未得完整分布 | 未得完整分布 | 未保留 | 失败前计数未保存 |

每次tyx interpret/CUDA重放报告22个completed_launches，含首调用/预热各303次均检查；无需猜测设备是否真正执行。首调用包含已有磁盘缓存等环境状态，不是独立冷进程/冷缓存指标。provenance git_dirty来自报告/诊断文件，三份源码哈希完全一致。

原始每次样本：[NumPy](p65-load-replay-numpy.json)、[interpret](p65-load-replay-interpret.json)、[CUDA](p65-load-replay-cuda.json)。复现`python scripts/benchmark_load_replay.py --mode <mode> --samples 100 --repeats 3 --output <new-json>`。

C++抛出`CpuCppBackendError: stage load_verified / WinError1114`；原测量器异常时未保存部分样本，不能制造C++成功计数或分位数。后续独立tyx公开API控制已确认Windows Runtime关闭后DLL持续留存；完整证据与修复建议在[TYX-003](../../../tyx_flaw/TYX-003-windows-cpp-library-lifetime.md)。这是持续运行能力缺陷，不能归为可忽略速度优化。原C++2项功能测试通过仍有效，但不证明300次长期运行。

热点定位与后续方向见[PERF-010](../PERF-010-load-replay.md)。未测完整接触迭代/CCD/双臂、峰值内存或纯kernel占比；未做优化或Warp对照。小负载端到端差异混有输入/快照/主机校验/分配/传输/启动，不能直接当作tyx计算吞吐。
