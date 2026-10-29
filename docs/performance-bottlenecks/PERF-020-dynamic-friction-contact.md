# PERF-020：两阶段动态摩擦重放与失败观测边界

状态：计量脚本最小自检通过，正式性能未执行，未优化。源码6df5b14（134文件hash见[工程记录](../benchmarks/p65-dynamic-friction-engineering.json)）。不能将实现者测试耗时或每模式两个smoke样本当正式性能结论。

## 已确认的结构与未测部分

新程序的法向和切向每次更新都进行完整原生重放，候选包含实际材料/机器人和末速度账本、精确Fraction/sqrt原式与物理预算审计。此结构意味着多次执行和host工作；其耗时占比尚未隔离，不能归因于tyx实现，更没有Warp差距证据。

正式计划是NumPy→interpret→CUDA串行各3×100，静支撑μ2和材料斜撞μ.5各150计时、各3warmup。每次固定初态，计完整step/审计/断言/workspaceclose，排除factory/finalclose；不是连续900步。不同容差/迭代不能直接作为后端速度比。当前没有正式raw，不列中位/P95或编造加速比。

## 失败帧观测限制

`src/phymeo/runtime/scene_stepper.py::_substep`在失败时清空candidate和diagnostics。成功帧能报告所有实际trial（包括未选best），并以read-only participant.engine.stats独立核对；失败report不能给出已执行部分的launch合计，故障测试只证明原生调用/close/回滚，不等于失败性能已测。

测试中曾额外调用native.diagnostics读取计数，该调用自身执行机器人kernel而扰动计数；已改为无副作用engine.stats探针。此为观察器问题，不是生产计数缺陷。本阶段保留旧失败API；后续可设计独立的失败执行统计快照，要求在close前后保留真实已完成/未完成计数、不影响事务和数值、不把失败候选当成功。该扩展尚未设计/实现。

## 代码位置与后续方向

- `src/phymeo/solvers/_dynamic_friction_workspace.py`：联合法切迭代、两个阶段重放与所有实际trial统计；未来先分离输入构造/native执行/材料与接触审计成本。
- `src/phymeo/solvers/_dynamic_friction_law.py`：精确实际world分解、容量、sqrt包围与最邻近可行FP64表示；记录算术成本，当前不做优化。
- `scripts/benchmark_dynamic_friction_contact.py`：固定解析负载、bitwise重复和真实执行统计，正式运行必须等全部测试/reviewer退出。
- `src/phymeo/runtime/scene_stepper.py::_substep`：上述失败诊断清空边界；若后续开发失败统计，先定义不可影响native执行的采样合同和故障对照。

未测峰值内存、冷JIT、完整递布轨迹或C++长期。TYX-003/004仍保持原问题记录，不把本项host集成成本新增成依赖缺陷。
