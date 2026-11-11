# 显式积分状态观测

2026-11-11。本片在既有完整物理检查之后提供 `MuJoCoTyxRuntime.integration_snapshot(keys=None)`，返回独立 `MuJoCoTyxIntegrationState`。保存accepted q/v/act/ctrl、time/tick、外载、held control和完整原step receipt；数组immutable。没有endpoint geometry/contact/sensor/energy字段，缺失不是零。`runtime.snapshot()`继续执行并标记原CPU endpoint forward，语义不变。

训练入口 `MuJoCoTyxVectorEnv(..., observation_mode="integration")` 显式选择该路径；默认仍是full。模式只读并进入训练binding；full原checkpoint binding保持兼容，跨模式恢复被拒绝。物理step的完整动力学/接触检查与共同发布不变，仍有HOST工作；这不是零拷贝或纯设备训练。

实际CUDA专项由父唯一串行完成：J/S/R各66步×W8，同输入完整观测与积分观测逐步比对q/v/控制、原完整力/接触收据、奖励、终止/timeout及partial reset。没有endpoint forward的断言区分合法reset初始化forward。历史快照不可变、wrong-world拒绝、恢复状态、跨模式checkpoint拒绝、同模式恢复和观察晚失败后的真实物理接受/needs_recovery均验。两个真实4步PPO更新用于合法update边界checkpoint，不用于学习效果结论。旧默认full两原任务连续回归通过。

新6节点按05的J1通过、07的S/R2通过、08剩余3通过联合覆盖；09原默认2通过。产品源码期间未变，后两次修正只在测试：05误将checkpoint按key排序后的mapping当有序物理状态；07错误在尚未完成update时保存checkpoint。首失败保留。改正仅忽略mapping键顺序，全部字段/数组字节/shape/dtype及序列顺序仍核；保存检查改为真正完成update后再保存。最终Ruff/format均通过。

固定已接受S状态、交替顺序6次观测的本机耗时（含HOST状态复制，未包含任何新物理step、PPO或证据编码）：

| Worlds | 原full ms | integration ms |
|---|---:|---:|
| 8 | 4.90–5.36 | 0.169–0.193 |
| 128 | 77.91–91.00 | 2.63–3.68 |

这是显式少做端点重算后的局部操作对比，非同输出工作量的求解器加速，也不是Newton性能比较或训练总吞吐改进证明。默认full reference和完整step检查都保留；后续需单独测声明此观测模式的端到端训练。三条完整配对gzip、两timing JSON及原失败/raw/source在 `artifacts/p65-parity-n4b-rob2a-col-stop-20261111-01`，新包资格待本批统一构建。
