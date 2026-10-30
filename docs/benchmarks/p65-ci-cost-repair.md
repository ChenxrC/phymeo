# P6.5：保留 30 分钟 CI 的精确算术优化

## 当前第二轮结果

第二轮纯区间成本修复已完成本地验收：产品 `7e68154a286bff65a7cdd727b63ffd58e676f88c`，最终报告 `c72cdc30b798f40d23fdc5cdf519f0c5aadd4c84`。最终受影响 host 195 项（840.64s）、actual interpret 19 项（351.09s）、CUDA 19 项（975.66s）全部 exit0，含完整 finite/thick 消费者及原窗口/失败/事务；同次224条厚度实际记录审计通过。父独立核对309源码及126证据hash无差异，原30分钟workflow字节不变。

父已完成新源码构建与源码外 fresh nativepip wheel 验证，两step exit0；153源码逐字节相同，wheel SHA256 `ebb49fac5399b0b60898df4321aa47b7a408a684939fc69d0f8c7e8be2085210`。当前待专用分支推送和新准确head完整CI/Documentation；旧a478828的7格超时、1格成功保留历史，不能替代新版本结果。本次未重跑C++，原52launch证据仅属33ec9ea。

用户随后明确同意“先并行完成模块代码，最后测试因资源原因串行”。已在隔离工作树启动弯曲B1、自碰撞基础S1及有限夹爪G1代码开发，合同提交 `1ccab186`、父G1未验证代码检查点 `88631ae` 位于 `codex/p65-parallel-modules`，不进入本交付head。各线只编写代码与测试，尚未运行验证；公共材料/载荷/事务由父集成，所有检查和数值执行排入唯一串行队列。完整自碰撞、弯曲材料账本/native及真实承重/低摩擦滑落仍须后续验收。此人工决定替代历史“不提前开始下一生产代码”的排期限制，未放宽任何验收。

本轮仅共享 `_patch_interval.py` 与新增测试变化。最终快路仅用于精确builtin二元tuple且端点均为标准Fraction，其余类型/shape仍走原四角分支；相同convertedq只在构造器内复用不可变端点。14474 helper、9 legacy shape、22完整curve差分相等。最终三旧观察复用、随后三新观察，非交错；单Windows CPU A负载中位成本低16.11%，完整report/state/history/input/CCD仅按R5排除精确类型timings字段后相同，不代表CI/P95/跨后端速度。成本源到最终源仅一空CRLF，AST一致且版本单独记录。

## 第一轮结果与历史诊断

用户选择保持 CPU job 30 分钟限时，另行优化耗时。完整八格矩阵、测试、示例、required、fail-fast 和依赖均未改。原5150c3e的CI40950637127因超时cancelled，Documentation40950637280success；历史失败保留。a478828 的准确 CI40991186564 已取消：1 格 CPU 测试和示例通过（Ubuntu/py3.12/pip，1357s），其余7格明确30m0s超时，required失败；Documentation40991186517成功。完整交付仍开放。

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

## 第二轮诊断与范围

优化版本 `a478828aa6ab8c59dc310cd4639e6f7a55704bec` 已推送；准确 CI `40991186564` 于 2026-10-30 02:51:45Z 结束为 cancelled，Documentation `40991186517` success。8 格 CPU 中 Ubuntu/Python3.12/pip 的测试与示例全部 success，job 共 1357s；其余 7 格均由公开完整 commit 页面确认超过 30m0s，测试 cancelled、示例 skipped，required failure。完整矩阵未通过，不能据此推断未完成测试的物理断言结果。用户已明确保留 30 分钟；90 分钟方案不再考虑。第一轮 0065200 的本地等价优化、host96/实际两后端各9及新 wheel 证据保持原版本。现从已完成 optimized-A.prof 继续诊断，父冻结下一轮有理区间重复计算优化；不重跑该已结束 CI，不提前进入下一生产合同。

同一已完成的 optimized-A 剖析中，patch_geometry 累计5.8747s；其共享 PatchCurve._prismatic 924次/4.5104s，区间mul166536次/3.9234s（累计时间相互重叠，不能相加）。下一步只消除单点有理区间乘法和相同端点FK的重复计算，扩大到旧finite与新thick直接消费者，详见现有合同R6与计划Task3。没有重新采集基线、没有新功能/预算变动。逐测试远程耗时当前不可得：公开日志片段返回404，API后续查询实际触发403限流；诊断只使用已取得job/step及公开超时注释。
