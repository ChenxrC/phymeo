# PERF-003：子步缓冲区分配与状态复制

- 状态：生命周期、阶段与公开传输计数已确认，分配/复制各自占比未测；2026-10-27，原定位 `67371f9`，新运行器测量 `b946c19`。

## 现象与代码

- [tyx_coupled.py](../../src/phymeo/backends/tyx_coupled.py)：`_substep` 的资源作用域（约 200 行）每子步分配并释放数组。
- [tyx.py](../../src/phymeo/backends/tyx.py)：`_cuda_step`（111 行），预测输入上传、输出分配和回读。
- [tyx_rigid.py](../../src/phymeo/backends/tyx_rigid.py)：设备预测缓冲与结果回读（约 90 行）。
- [coupled.py](../../src/phymeo/core/coupled.py)：`CoupledState` 复制状态，支持独立候选与原子恢复。

现有设计明确保证所有权和资源释放。数据常驻设备、持久工作区或跨求解器零拷贝尚未作为联合路径的正式能力交付。未测得分配/复制占整体时间的比例。

P1/P2 新位置：[tyx_coupled_program.py](../../src/phymeo/backends/tyx_coupled_program.py) 的 `_DeviceWorkspace._buffer/predict/close` 每子步管理设备数组与 ExitStack；[registry.py](../../src/phymeo/runtime/registry.py) 的 `_copy_fields/snapshot` 拥有状态副本；[scene_stepper.py](../../src/phymeo/runtime/scene_stepper.py) 的 `_substep/step` 实施事务与身份门禁。[修复版测量](measurements/p2-runtime.md)给出快照、setup、predict、validation、close 和外层剩余时间，但这些阶段混有其他工作。分配计数和峰值内存仍未测，不能从传输次数推算分配次数。

## 定位与后续方向

记录每帧分配/释放次数、上传/下载字节、主机副本数量、峰值内存；分开启用/关闭可选子步快照的诊断结果，但两者均保留必需事务与物理门禁。

后续可评估持久 workspace、设备状态句柄、双缓冲、缓存失效规则。风险包括旧拓扑/参数缓存被复用、失败留下脏状态、释放后引用、跨 stream 生命周期错误。未来变更必须覆盖模型/dt/外力改变、部分分配失败、close 后调用与重复运行。

本轮允许实现统一工作区的所有权接口；没有性能必要性证据前，不把该接口实现成复杂缓存或图捕获系统。
