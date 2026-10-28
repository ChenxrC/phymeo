# P6.5 厚度材料点：串行观测

源码`10f8cdd`；源码树哈希`eac7e6ccc0c2052fcf54c59adeed74237c3b5b1a5fe8fa4962301bed4557edc6`。Python3.12.14/NumPy2.5.4，tyx0.1.0a0@99f6df7c0fd8fdeef77f97df83914da27b64b7ee，Windows11，RTX4060 Laptop，driver566.07。

全测试进程结束后独占串行运行，各模式两个小负载：3节点、1或16个共享同面的固定偏置附件。每个构造/响应/候选操作先1次首调用、2次预热，再3组×100次热态调用；合计7200次计量全部成功。首调用不等于冷进程或冷缓存。结果含必要断言，候选项含整域实际动能审计，响应返回主机后计时。没有完整step、CCD、动态夹持或设备内存采样。

16附件结果（合并3组样本，中位数 / p95，ms）：

| 底层后端 | host映射构造 | 响应＋断言 | 候选＋审计＋断言 |
| --- | --- | --- | --- |
| NumPy | 1.0645 / 1.2382 | .1144 / .1466 | .2270 / .2792 |
| tyx interpret | 1.1537 / 1.3584 | .6194 / .7772 | 1.1829 / 1.4178 |
| tyx C++ | 1.0733 / 1.2344 | .6040 / .7171 | 1.1817 / 1.3959 |
| tyx CUDA | 1.0839 / 1.2737 | .9109 / 1.0960 | 1.7393 / 2.2405 |

每个tyx模式、每个负载的completed_launches增长909，涵盖首调用和预热；无缺失执行。1附件NumPy host构造中位数.1154ms，16附件为1.0645ms，符合目前逐面检查的结构性成本，但未做profiler占比归因。如此小的3节点负载受封装、复制、host验算和同步影响，不用于推断tyx吞吐或与Warp的差距。

原始每次样本、环境、源码/脚本哈希和执行计数：[NumPy](p65-offset-numpy.json)、[interpret](p65-offset-interpret.json)、[C++](p65-offset-cpu_cpp.json)、[CUDA](p65-offset-cuda.json)。后几个provenance的git_dirty来自新增测量文件；四份源码哈希完全相同，没有测量时改代码。

复现：`python scripts/benchmark_offset_port.py --mode numpy|interpret|cpu_cpp|cuda --samples 100 --repeats 3 --output <json>`。四种模式逐次运行，先关闭其他测试进程；保留缓存状态说明。热点和下一步见[PERF-009](../PERF-009-thickness-map.md)。
