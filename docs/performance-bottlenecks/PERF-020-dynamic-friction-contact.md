# PERF-020：两阶段动态摩擦重放与失败观测边界

状态：正式NumPy/interpret各300计时已完成并独立核对，CUDA正式计量仍运行，未优化。源码6df5b14（134文件hash见[工程记录](../benchmarks/p65-dynamic-friction-engineering.json)）。不能将实现者测试耗时或每模式两个smoke样本当正式性能结论。

## 已确认的结构与未测部分

新程序的法向和切向每次更新都进行完整原生重放，候选包含实际材料/机器人和末速度账本、精确Fraction/sqrt原式与物理预算审计。此结构意味着多次执行和host工作；其耗时占比尚未隔离，不能归因于tyx实现，更没有Warp差距证据。

正式计划是NumPy→interpret→CUDA串行各3×100，静支撑μ2和材料斜撞μ.5各150计时、各3warmup。每次固定初态，计完整step/审计/断言/workspaceclose，排除factory/finalclose；不是连续900步。不同容差/迭代不能直接作为后端速度比。当前已有两模式正式raw，见下表；CUDA未结束，不能声称完整900成功或跨后端速度比。

## 失败帧观测限制

`src/phymeo/runtime/scene_stepper.py::_substep`在失败时清空candidate和diagnostics。成功帧能报告所有实际trial（包括未选best），并以read-only participant.engine.stats独立核对；失败report不能给出已执行部分的launch合计，故障测试只证明原生调用/close/回滚，不等于失败性能已测。

测试中曾额外调用native.diagnostics读取计数，该调用自身执行机器人kernel而扰动计数；已改为无副作用engine.stats探针。此为观察器问题，不是生产计数缺陷。本阶段保留旧失败API；后续可设计独立的失败执行统计快照，要求在close前后保留真实已完成/未完成计数、不影响事务和数值、不把失败候选当成功。该扩展尚未设计/实现。

## 代码位置与后续方向

- `src/phymeo/solvers/_dynamic_friction_workspace.py`：联合法切迭代、两个阶段重放与所有实际trial统计；未来先分离输入构造/native执行/材料与接触审计成本。
- `src/phymeo/solvers/_dynamic_friction_law.py`：精确实际world分解、容量、sqrt包围与最邻近可行FP64表示；记录算术成本，当前不做优化。
- `scripts/benchmark_dynamic_friction_contact.py`：固定解析负载、bitwise重复和真实执行统计，正式运行必须等全部测试/reviewer退出。
- `src/phymeo/runtime/scene_stepper.py::_substep`：上述失败诊断清空边界；若后续开发失败统计，先定义不可影响native执行的采样合同和故障对照。

未测峰值内存、冷JIT、完整递布轨迹或C++长期。TYX-003/004仍保持原问题记录，不把本项host集成成本新增成依赖缺陷。

## 最终验收中的慢测试观察（非正式性能）

同一冻结快照的第一套core311已1945通过/362未选，pytest总665.54s。自带durations记录显示新`test_two_second_window_and_halving_error[2.0]`静摩擦T2/两步长/重建重复测试耗时187.58s；这是该测试整体执行与断言成本，尚未拆分物理步、重放、native与host审计。其余慢项包括旧12帧联合场景约28–35s。原始20项记录保存在工程JSON和本地core311.txt；不将它们当正式基准样本或tyx缺陷，不为归因另重跑整套。第二CPU随后也1945通过/362未选657.52s，工程JSON保留其独立durations；完整CUDA随后2307通过6196.34s；旧高/低μ步长测试分别1206.55/1062.46s，旧normal生命周期1052.50s，新μ2细步T.5窗口595.01s。这些仍是同次验收的测试级时长，不能作kernel归因。


## 正式观测部分结果（CUDA待完成）

只读汇总器核对NumPy/interpret各300计时、每case150及各3warmup，独立重算中位/P95，source/script/frozen摘要保持不变。共600计时/612总调用通过；完整900仍待CUDA结束，不重复测量。

| 模式 | 静支撑中位 / P95 ms | 材料斜撞中位 / P95 ms |
| --- | --- | --- |
| NumPy | 916.547 / 954.414 | 397.765 / 418.050 |
| interpret | 4520.627 / 11657.570 | 848.657 / 2364.152 |

静支撑每次held110/terminal1；完整step包含重复重放和审计。interpret尾部分布较宽，尚未隔离原因，不能仅凭总耗时归因tyx。NumPy记录HEAD784adf0且clean；interpret首784adf0/末afcb5d1，期间只有分层验收文档提交与前序raw；源hashd18e557c、脚本c9426b7a、冻结摘要ea6e07全未变。raw待CUDA齐备后一并提交。

后续按[分层验收](../development/p65-validation-policy.md)在性能基线、优化比较、集成里程碑或最终交付采集正式大样本，普通小切片不再例行900采样；本次已启动冻结流水线完整保留。
