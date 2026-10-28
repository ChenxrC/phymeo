# TYX-003：Windows C++ Runtime关闭后动态库持续留存

- 状态：tyx独立复现DLL留存，Phymeo长循环出现DLL初始化失败；未修复。
- 类别：资源生命周期 / 持续运行能力。
- 优先级：高。阻塞当前Windows C++三组×100次重放验收；NumPy、interpret、CUDA独立观测已完成，不能把C++短测通过外推到长场景。
- 首次发现：2026-10-28，`scripts/benchmark_load_replay.py --mode cpu_cpp --samples 100 --repeats 3`。

## 问题与合理预期

固定源码3f23378的完整C++重放长循环在`robot_project_loads`加载时抛`CpuCppBackendError`，stage=`load_verified`，底层Windows错误1114（DLL初始化例程失败）。输出的“artifact cache failed”不能直接理解为磁盘缓存损坏。原计时脚本在异常时未保存部分样本，因此成功计量数未知，不能写成300成功或补造分位数。

显式关闭Runtime后，下一次新Runtime不应无限累积旧动态库直至无法加载。独立脚本只用tyx公开API和Windows只读模块枚举，没有导入Phymeo、没有改写依赖或调用私有卸载函数。**Warp未测。**

## 环境与最小复现

Phymeo产品`3f2337850e130e56172e957a4696ceb93cb738ed`，源码109文件SHA256`bbc798c78fedc222192fdd557a72000d02f52f0caa0d13b3f5daf4a11bcbff0c`；tyx-compute0.1.0a0固定`99f6df7c0fd8fdeef77f97df83914da27b64b7ee`；Python3.12.14、NumPy2.5.4、Windows11 10.0.26200；i9-13980HX，24核32线程；MSVC14.50.35717。C++ FP32四元素翻倍kernel，无GPU计算。机器另有RTX4060 Laptop/driver566.07，与本问题归因无关。

[最小脚本](repros/windows_cpp_lifecycle.py)：在独立新进程分别运行shared和new。指定可写缓存与TEMP，避免与此前默认缓存不可写问题混淆。

```powershell
$env:TEMP=(Join-Path (Get-Location) 'artifacts/tmp'); $env:TMP=$env:TEMP
python tyx_flaw/repros/windows_cpp_lifecycle.py --lifetime shared --cycles 64 --cache-root artifacts/tyx-lifecycle-cache --output shared.json
python tyx_flaw/repros/windows_cpp_lifecycle.py --lifetime new --cycles 64 --cache-root artifacts/tyx-lifecycle-cache --output new.json
python tyx_flaw/repros/windows_cpp_lifecycle.py --lifetime new --cycles 1200 --cache-root artifacts/tyx-lifecycle-cache --output long.json
```

## 证据及归因强度

| 独立条件 | 正确调用/已关闭Runtime | 关闭＋GC后仍加载的tyx DLL | JIT观测 |
| --- | --- | --- | --- |
| shared64 | 64 / 1 | 1 | cold_compile1，process_hit63 |
| new64 | 64 / 64 | 64 | artifact_hit64 |
| new1200 | 1200 / 1200 | 1200 | cold_compile1，artifact_hit1199 |

原始文件：[shared64](evidence/TYX-003-shared-64.json)、[new64](evidence/TYX-003-new-64.json)、[new1200](evidence/TYX-003-new-1200.json)。模块数每16轮检查，new持续线性增加；每次输出数值都正确，所有Runtime显式close并断言closed，GC后仍留存。1200次独立单kernel未触发1114，这一事实不能省略。

已确认：依赖Windows库生命周期留存，不是Phymeo持有物理状态或漏调close；可写缓存能artifact_hit，磁盘缓存不可写不是本次原因。源码也明确在Windows跳过卸载。尚未确认：1114具体触及哪项Windows/CRT资源、最小触发规模及不同kernel/工具链的影响。DLL留存与实际集成资源失败有一致方向，但不将未做的TLS/堆/loader诊断写成已证明原因。

这些是资源诊断，不是正式热态速度基线；峰值内存、纯kernel成本、跨平台影响和Warp对照均未测。

## 代码位置

- tyx固定提交：`backends/cpu_cpp.py::_load_verified_cpu_cpp_library`每次建立新的`tyx-cpu-cpp-load-*`目录再加载同一缓存artifact；`CompiledKernel.close`调用`_close_shared_library`，后者在`os.name == "nt"`直接返回（当前安装源码909—912行）。
- tyx：`runtime.py::RuntimeState.close`会遍历kernel_cache并close，因此不是完全没有关闭路径。
- Phymeo：`backends/tyx_participants.py::_RobotWorkspace/_NodeWorkspace`每个试算拥有独立引擎；`TyxMath.close`调用runtime.close；`SceneLoadReplay.evaluate`通过原SceneStepper创建/关闭工作区，放大了这个依赖生命周期问题。

## 给tyx agent的排查与修复建议

先用上述三组在固定提交复现，保留同Runtime对照与可写artifact缓存。检查Windows不卸载是否有并发调用、ctypes函数引用或安全生命周期方面的设计理由；不能简单在Phymeo里私调FreeLibrary或清掉安全门禁。候选修复包括有正确引用计数与并发关闭保护的Windows模块释放，或受控的进程级不可变artifact模块复用；须明确关闭后不可调用和异常清理语义。

进一步最小化1114：同进程逐步增加库数量，比较代表robot kernel、同名重复artifact、简单kernel，分别采集模块/句柄/内存/loader异常。不要一开始以关掉缓存或反复重启子进程掩盖持续运行问题。Phymeo当前不改依赖pin、不引入隐藏共享物理状态，也不把C++错误退回NumPy。

## 修复验收

独立new1200结束后模块数量应有明确有界生命周期，不再随Runtime数量线性增长；并发执行/关闭、重复close、编译/加载失败与fork等适用平台边界需由tyx测试。回到同一Phymeo载荷、同样300次计量和原数值门禁；成功后补完整C++分布及闭环记录。必要时在真实P6.5长期集成中扩大压力。未修复之前本条不能标关闭。

## 集成重复复现补充

[原始诊断](evidence/TYX-003-phymeo-replay.json)在独立新进程中重建同一单关节＋单节点负载，226次完整重放成功，第227次抛同类DLL初始化错误；最后仍加载2036个tyx库。每16次检查呈每重放增加9库（第224次为2016库）。这是带模块计数的诊断，不是补做正式C++性能分布；原计时失败前成功数仍未知。

[可复用集成脚本](repros/phymeo_cpp_replay_lifecycle.py)保留相同计算与计数路径，并将本地临时诊断整理成CLI；`--calls 1`已smoke。设置同上TEMP/TMP及`TYX_CACHE_DIR=artifacts/tyx-cache`后运行`python tyx_flaw/repros/phymeo_cpp_replay_lifecycle.py --calls 303 --output replay.json`。该脚本有Phymeo依赖，不能冒称独立tyx最小复现；前面的单kernel控制才用于排除Phymeo持有资源因素。
