# P6.5 表面端口与静态几何的小负载观测

2026-10-28；只测已实现的H1a/H1b基础，不是双臂场景、完整step或加速比。测量前所有测试进程均已结束，四模式依次串行运行。没有进行性能优化。

## 环境与复现

Windows11 26200、Intel64 Family6 Model183、RTX4060 Laptop 8188MiB/驱动566.07；Python3.12.14、NumPy2.5.4；tyx固定`99f6df7c0fd8fdeef77f97df83914da27b64b7ee`。新增映射/几何均主机FP64；tyx底层实际FP32算术/FP64状态。没有设置额外线程变量，线程数/调度与峰值内存未测。

源码为`f41c943`检查点加导出/测量器未提交变化；每份JSON保存Git脏状态和101产品文件＋测量器逐文件SHA256，结束时再次核对一致。后续源码变更不能据此覆盖原始结果。所有模式使用工作区TEMP/TMP及TYX_CACHE_DIR；缓存未清空，首调用不称冷JIT，NumPy首调用也包括断言首次加载等开销。

PowerShell中设置TEMP/TMP=`artifacts/tmp`、TYX_CACHE_DIR=`artifacts/tyx-cache`后，依次执行：

```sh
python scripts/benchmark_surface_foundation.py --mode numpy --output artifacts/p65-numpy.json
python scripts/benchmark_surface_foundation.py --mode interpret --output artifacts/p65-interpret.json
python scripts/benchmark_surface_foundation.py --mode cpu_cpp --output artifacts/p65-cpp.json
python scripts/benchmark_surface_foundation.py --mode cuda --output artifacts/p65-cuda.json
```

工作负载为3节点含1固定节点、2个重心点/1对，及2关节杆/2重心点/1对；每调用执行相对响应＋一次候选＋独立解析断言，查询完成主机返回后停止计时。机器人静止q=0、重力0、固定冲量.1Ns，每次从不可变快照求解，不推进时间。各1首调用＋2预热＋3×20计时调用，8个配置共480计时调用全部通过，失败0；几何另60调用，每调用各1次点/三角和偏斜边查询与解析断言。

## 结果

| 模式 | 节点三批中位范围 ms | 节点p95范围 ms | 机器人三批中位范围 ms | 机器人p95范围 ms |
| --- | --- | --- | --- | --- |
| NumPy | .253–.339 | .285–.441 | 1.252–1.280 | 1.507–2.149 |
| tyx interpret | 1.323–1.376 | 1.471–1.547 | 66.975–70.239 | 69.646–74.675 |
| tyx C++ | 3.016–3.148 | 3.192–3.375 | 16.497–17.494 | 17.314–18.576 |
| tyx CUDA | 2.907–3.278 | 3.414–10.290 | 18.576–19.315 | 24.213–28.110 |

主机两项几何组合的三批中位.249–.258ms。完整样本、setup/predict/首调用、公开执行前后计数：[NumPy](p65-foundation-numpy.json)、[interpret](p65-foundation-interpret.json)、[C++](p65-foundation-cpu_cpp.json)、[CUDA](p65-foundation-cuda.json)。设备模式查询期间完成launch计数实际增长，不以predict或导入充当证据。

## 解释与未测项

三角/边归一化、凸域检查、重心映射、候选验证和断言包含在端到端时间。机器人保留完整质量/Jacobian共享响应，反复调用底层会重复计算与复制。上述是phymeo集成延迟，没有分离tyx内部kernel、编译、分配、主机映射或验证各自成本，不能据此断言tyx实现低效。两端精度也不同，未计算CPU/GPU加速比；Warp未安装、未测。

传输字节、峰值内存、真正冷进程/JIT、完整场景吞吐、几何规模扩展均未测。下一步先以代表网格/接触数量重测，并在保持响应交叉项、同域反作用和候选验证的前提下分别剖析；详见[PERF-008](../PERF-008-surface-foundations.md)。未发现新增独立tyx实现缺陷，不增造`tyx_flaw`条目。
