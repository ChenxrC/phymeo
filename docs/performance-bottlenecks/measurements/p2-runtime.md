# P1/P2 修复版串行性能观测

全部测量使用干净 `b946c1924389efcf52f0a57ad6e85e34317cff21`，包源码哈希 `dd1dcd12f14ee1ccc11839dbc3cc5a54516e38f2d4d858109257ce46a2177669`，先结束测试和完整轨迹，再依次测合同、NumPy、CUDA；没有并发测试/实验。Python 3.12.14、NumPy 2.5.4、Windows 11；硬件信息及固定 tyx 来源完整保存在 [NumPy JSON](p2-numpy.json)和[CUDA JSON](p2-cuda.json)。GPU 为 RTX 4060 Laptop，驱动 566.07。后台操作系统负载未隔离。

## 口径

一个粒子与一个动态球的闭合接触对，质量分别为 1 kg 和 2 kg，零重力，dt=0.1 s、4 子步、每阶段预算 200。每次 step 都从同一个闭合接触初态开始，不是连续推进的 300 帧。NumPy FP64/容差 1e-7，tyx FP32/容差 1e-6；每步独立检查动量、质量比例加权位置及接触间隙，阈值 2e-6。每模式首次调用后预热 2 次，再做 3 次重复，每次 100 个完整 step，全部接受、失败数 0。实际迭代数和质量最大值在 JSON 中逐批保留。

完整 step 包括事务/身份门禁、主机校验和子步 workspace 的同步释放。运行器最终 close 位于所有样本之后；阶段计时不是 GPU event 计时。新建编译缓存目录，但进程/工具链/系统缓存可能已有热状态；构造/首次 step 不等同于已隔离的 JIT 成本。

| 模式 | 模型构造 ms | 运行器构造 ms | 首次 step ms |
| --- | --- | --- | --- |
| numpy | 0.308300 | 0.441000 | 11.724200 |
| cuda | 0.374500 | 190.765800 | 1391.641500 |

| 模式 | 重复 | 接受数 | 中位数 ms | p95 ms | 最大动量误差 kg·m/s | 最大间隙误差 m |
| --- | --- | --- | --- | --- | --- | --- |
| numpy | 1 | 100 | 6.151550 | 6.614175 | 6.66134e-16 | 2.22045e-16 |
| numpy | 2 | 100 | 6.128200 | 6.649725 | 6.66134e-16 | 2.22045e-16 |
| numpy | 3 | 100 | 6.098000 | 7.151835 | 6.66134e-16 | 2.22045e-16 |
| cuda | 1 | 100 | 21.636650 | 29.964220 | 9.53674e-07 | 0 |
| cuda | 2 | 100 | 21.431100 | 31.100555 | 9.53674e-07 | 0 |
| cuda | 3 | 100 | 22.676250 | 28.809315 | 9.53674e-07 | 0 |

由 300 次样本的总 step 时间计算，观测吞吐为 NumPy 164.228 step/s、CUDA 43.356 step/s。它只适用于上述微小负载和各自精度/容差，不能外推大规模性能或作为同语义加速比。

## 阶段及公开计数器

下表为三个重复的总时间除以 300，使用均值便于阶段相加；p95 和各批阶段中位数见 JSON。

| 阶段 | NumPy 平均 ms/step | CUDA 平均 ms/step |
| --- | --- | --- |
| setup | 0.870656 | 0.882472 |
| predict | 1.079563 | 12.156322 |
| position | 1.311053 | 2.259533 |
| prepare_velocity | 0.382633 | 2.812184 |
| velocity | 0.656696 | 1.760219 |
| validation | 0.618272 | 0.776132 |
| verification | 0.002120 | 0.002407 |
| diagnostics | 0.385682 | 0.383311 |
| close | 0.001382 | 0.961169 |
| outer_remainder | 0.781053 | 1.070987 |

`outer_remainder` 包含所有权副本、模型/配置指纹、外层报告和适配，不能叫作纯复制时间。`predict` 同时包含预测器、设备数组准备和相关读回；不能把该阶段全部归因于 tyx 内核。

| 公开计数器 | 300 次 step 合计 | 每 step |
| --- | --- | --- |
| cuda_kernel_launches | 8400 | 28 |
| graph_builds | 0 | 0 |
| graph_replays | 0 | 0 |
| graph_updates | 0 | 0 |
| cuda_h2d_transfers | 43200 | 144 |
| cuda_h2d_bytes | 432000 | 1440 |
| cuda_d2h_transfers | 26400 | 88 |
| cuda_d2h_bytes | 254400 | 848 |
| cuda_d2d_transfers | 0 | 0 |
| cuda_d2d_bytes | 0 | 0 |
| cuda_memsets | 0 | 0 |
| cuda_memset_bytes | 0 | 0 |
| arg_pack_build_count | 8400 | 28 |
| prepared_launch_count | 8400 | 28 |

计数来自公开 `tyx.runtime_stats()` 的样本区间差分。launch/传输/参数打包次数描述这个 phymeo 调用方式，不能单凭次数判定依赖实现缺陷。0 只表示对应公开计数器未增加，不代表零开销或某种未测能力不存在。独立同步次数、分配/释放次数、峰值主机/设备内存、纯 kernel/传输时间均未测；不将其填为 0。

## 修复版合同快照

49 粒子与 49 个显式历史值，FP64，2×100 预热、3×100 采样；检查历史相等和位置无别名。仅合同快照/验证，不含物理求解或 GPU。计时器开销未扣除，见[原始 JSON](p1-contracts-reviewed.json)。早期 [P1-T1 记录](p1-contracts.md)保留作历史，不冒充最终源码证据。

| 重复 | 样本 | 中位数 ms | p95 ms |
| --- | --- | --- | --- |
| 1 | 100 | 0.055400 | 0.062210 |
| 2 | 100 | 0.058500 | 0.068405 |
| 3 | 100 | 0.049650 | 0.060430 |

## 已知热点和后续

- [PERF-001](../PERF-001-host-sync.md)：主机门禁与反复传输在新路径继续存在；已经量到阶段与计数，但同步/残差/kernel 的独立成本仍需 profiler。
- [PERF-002](../PERF-002-serial-contact.md)：共享体顺序接触尚未做规模扫描，这个单对结果不能证明其扩展性。
- [PERF-003](../PERF-003-buffer-lifetime.md)：每子步分配、快照和身份检查保留；当前没有分配/峰值内存的精确归因。
- 点端口单算子吞吐、跨规模曲线、回滚性能、独立 JIT 和峰值内存仍未测。先以新增机器人/软体真实负载补足，再决定优化排序。

不进行性能优化，不声明 GPU/NumPy 加速比或 Warp 差距。FP64 能力缺口独立记录于 [TYX-001](../../../tyx_flaw/TYX-001-fp64-compute.md)，本测量未新增已定位的 tyx 性能缺陷。

```sh
python scripts/measure_scene_contracts.py --output artifacts/p1p2-contracts-reviewed.json
python scripts/measure_scene_runtime.py --mode numpy --output artifacts/p1p2-measure-numpy-reviewed.json
python scripts/measure_scene_runtime.py --mode cuda --output artifacts/p1p2-measure-cuda-reviewed.json
```
