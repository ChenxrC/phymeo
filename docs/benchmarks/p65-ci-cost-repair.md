# P6.5：保留 30 分钟 CI 的精确算术优化

用户选择保持 CPU job 30 分钟限时，另行优化耗时。完整八格矩阵、测试、示例、required、fail-fast 和依赖均未改。原5150c3e的CI40950637127因超时cancelled，Documentation40950637280success；历史失败保留，最新交付仍等待优化版本的准确CI。

产品0065200、实现报告d2cacff仅修改厚度模块的 `_prism_exact.py`，增加独立数学测试。纯有理数与已规范化同域运算免去重复根式分类及零项计算；保留Fraction精确值、非法输入/跨域/零除、规范形、符号/排序/hash、向外界和float位模式。没有geometry/solver/cache/physics/旧测试改动。设计与边界见[优化合同](../superpowers/specs/2026-10-30-p65-ci-cost-repair-design.md)和[计划](../superpowers/plans/2026-10-30-p65-ci-cost-repair.md)。

34项数学控制初次通过，2项冗余运算成本控制真实RED，修复后36通过，新增c8域控制后37通过。冻结旧模块的119次构造和6682次运算差分全等。相同A单步三个独立旧进程为12.29–12.59s，三个新进程为5.89–5.95s，该负载中位观察降低52.1%；不是P95、跨后端或完整CI速度结论。六次完整输入、state/history、53held+1terminal和CCD相同。父直接遍历原始报告，唯一差异为既有SceneSubstepReport.timings墙钟字段；R5只在观察器排除这个精确类型字段，原始报告及计时完整保留。

| 优化后同一源码的验证 | 结果 | pytest 时间 |
| --- | --- | --- |
| 全部受影响host：数学、几何、映射、接触和事务五文件 | 96 passed，exit0 | 818.12s |
| 实际interpret全部厚度测试 | 9 passed，exit0，无skip/fallback | 328.43s |
| 实际CUDA全部厚度测试 | 9 passed，exit0，无skip/fallback | 787.11s |

原规定A/C窗口、步长减半、独立重建及12次后段故障/retry均保留。上述同次运行的124/50/50共224条实际记录独立审计通过，完整state/history重建逐位相同、原tol及力/映射/能量/角动量/CCD与设备FP32检查通过。没有为了生成报告重跑物理。

父核对308源码/测试/scripts及93证据文件零差异，并核对20份原始成功退出码；未重新发起整阶段fresh review。新源码build和源码外fresh native-pip完整installed smoke均exit0，wheel153源码逐字节相同。wheelSHA256 a2b1bd35d0e99946fde91837eed819936ea414c2852a6866ed8bed497c5182c6。新308摘要9bbdeeeba487184c3a95fd96b0d20ae928ce1554c550beb2c5a0d598b79c745d；153source摘要5f01a176f8ef29604da0437799eea49829cf971657334a1095360eebaac189bc。旧CPP52launch仅保留原33ec9ea版本，不冒称优化后新增CPP验收，TYX003长期问题不变。

[机器可读证据](p65-ci-cost-repair.json)记录精确命令、摘要、观察限制和包装结果。最终是否满足30分钟必须由新准确SHA的完整CI确认。现有coordinate-prismatic支持域及R4 diagonal候选可达性限制保持；一般机器人/完整厚度摩擦及后续P6.5阶段尚待。
