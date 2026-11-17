# 面积接触：持久设备记录与批量读取

2026-11-17，codex/exp，实现提交 `45dec87a06d1be622f40ff98ce7874809bf4c70a`。dev 未改，未 push。

本批把面积载荷从“只有 HOST 历史回执”推进到“与物理状态一起保留的设备记录”。GPU 消费者可以取得独立的批量副本，不必先把数值读回 HOST 再上传。接触面生成仍在 HOST，本批没有改变接触求解方程或完成全设备碰撞管线。

## 实际连接

本步 HOST 几何输入（原先已有） → TYX 共同求解 → 设备 kernel 将输入点/法向、实际 type 7 行力及面积位写入 inactive bank → 原有验证 → 与 q/v 同一次 active bank 发布 → 调用者自有缓冲区同步复制。

准备失败只污染候选或 scratch；已经接受的状态与面积记录不变。设备记录由真实求解输入和行生成，不从 HOST 数值回执重建。HOST 回执只用于交叉检查，以及提供字符串形式的材料/source/sample 身份目录。

## 公共接口

`runtime.area_pressure_device_layout` 返回只读的字段→(shape,dtype) 映射。

`runtime.copy_area_pressure(outputs, expected_revision=...)` 复制完整同模型多世界 batch。interpret 使用调用者 NumPy 数组，CUDA 使用同设备的 TYX CudaArray。返回值绑定 revision、world order、pair names 和历史身份目录；没有隐式世界重排。

| 字段 | 含义 |
| --- | --- |
| entry_q / entry_v / entry_clock / entry_meta | 力对应的输入状态、原始输入/输出 clock words、valid 与 tick |
| endpoint_q / endpoint_v / endpoint_clock | 同一已接受步的运动输出，不能与输入相位混用 |
| area_meta | valid、样本数、正法向力样本数、overflow |
| area_ids | valid、pair ordinal、本条记录内的 global sample ordinal、body A/B、world moment valid |
| area_data | point3、normal3、k、phi、法向力、A方世界力3、A方共同世界原点矩3，FP32 |
| area_measure_words | 每点 dA 的 little-endian IEEE754 binary64 低/高两个 i32 word，原始位完整保留 |

B 方世界力与共同原点矩取相反数。sample ordinal 只在本条记录及其 catalog 中有效，不是跨帧接触跟踪 ID。字符串材料/source/sample 身份在不可变 HOST catalog 中，数值输出从设备直接复制。

世界原点矩可因极端坐标超出 FP32 可用域。该字段必须连同 `area_ids[...,5]` 使用：两乘积非有限或超过 `2**125` 时，mask=0、三列为零 sentinel，不能解释为真实零矩；这不会使物理步失败。I3a 的 HOST FP64 力矩仍保留。常规案例比较通过，极值负控验证了显式不可用路径。

复制先固定目标对象，完整检查 dtype/shape/device/存活/可写/连续性，拒绝目标互相重叠及别名到当前或保留待清理的物理/控制存储。preflight 失败不写目标；传输中途失败不改变物理，但调用者目标可能部分写入，必须等成功返回后消费。该首版为同步接口，不提供异步 lease/外部 stream 合同。

## 验证结果

| 组 | 实际结果 | pytest 用时 |
| --- | --- | --- |
| 新设备观察 interpret | 6 passed | 6.06 s |
| 新设备观察 CUDA | 6 passed | 6.15 s |
| I3a 生命周期与面积/原生混合接触 | 16 passed | 30.37 s |
| 原生 entry/checkpoint/观察生命周期 | 4 passed | 20.50 s |

**12 新节点＋20 受影响/兼容节点通过**；无 skip。新组按 mode 分两次执行，各次 deselect 另一模式，合并覆盖全部声明节点。最终十个 Python 文件 Ruff 与 format check 通过，测试前后源码哈希一致。所有项目执行串行；原两个 agent 仅 author/review。

新测试实际覆盖：

- 两环境完整批量输出与同次 HOST 接受回执比较；点/法向/k/phi/force 和 dA 位精确匹配。复制期间禁用 HOST 数值回执、状态快照、设备读回及 CPU MuJoCo 计算入口，CUDA `to_numpy` 也被禁用，仍成功。
- 初始 invalid 与成功空面 valid=1/count=0 区分；调用者写坏副本、后续物理步或 reset 均不影响另一方存储。
- checkpoint roundtrip、恢复、乱序/子集 clone 及继续后设备字节一致；篡改面积设备 bank 被拒。
- stale revision、缺项、额外项、错误 dtype/shape、不可写、错设备、重叠、物理别名被拒；变化的 Mapping.items() 不能绕过已完成的预检。
- 真实步的晚 download 故障和容量不足不发布候选数据，之后合法重试成功。
- 直接 stage 的极端世界原点矩只标不可用；INT32_MIN count 在减法/索引前拒绝，没有越界读行。

证据：[本批原始日志、guard、命令与源码绑定](../../artifacts/area-pressure-device-observation-20261117-01)。首次 import 排序/unused import 与一次 Ruff CLI selector 错误均保留，随后修正；一次自动审批超时发生在进程创建前，按工具允许重试一次后执行成功。没有产品测试失败、参数调优、旧长期轨迹重跑或远程 CI。

## 生命周期与资源

四个 area bank 字段只在声明 area pair 的路径分配，沿原 manifest/预算/清理 owner 管理。reset/restore/clone 复制并核对实际 bank 字节；不从 HOST 历史补造设备记录。原七个 native entry 字段的布局与非 area 旧存档兼容路径保持。

**旧 I3a 面积存档显式拒绝新布局**，需要用旧实现读取；本批没有伪造迁移。新的 scene 存档继续绑定原压力/刚面资产，控制更新不改历史 catalog。

按实际布局，新增持久设备 payload 为每世界 `192*constraint_capacity + 32` 字节（capacity 使用至少1个的存储上界），包括双 area bank 与原始面积输入位；cap=64 时为12,320字节。这是布局核算，不是测速或总运行内存。复制按完整容量进行，成本随世界数与容量增长；没有宣称性能超越其他引擎。

## 最小使用

```python
# runtime 是已经构造的 CUDA 面积接触运行时。
import numpy as np
import tyx as tx

outputs = {
    name: tx.CudaArray(np.zeros(shape, dtype), device=device_ordinal)
    for name, (shape, dtype) in runtime.area_pressure_device_layout.items()
}
try:
    frame = runtime.copy_area_pressure(
        outputs, expected_revision=runtime.checkpoint_revision
    )
    # 成功返回后，设备消费者使用 outputs；身份由 frame["catalogs"] 解释。
    # 面积原始位可以在消费者侧 reinterpret 为 float64，不能当 i32 数值相加。
finally:
    for array in outputs.values():
        array.free()
```

## 后续顺序

1. 把面积设备记录接到统一观察消费入口，再扩展异构 model buckets 的汇总、mask 和世界排序。此次只有同模型多世界复制，尚未完成异构面积训练接入。
2. 单独设备化有限接触面生成，对照现有 HOST 裁剪结果，并计入输入上传、生成、求解、观察的整步成本。
3. 再推进分布摩擦、材料模型选择与后续刚柔扩展。FFT 非局部材料、GPU 裁剪和新训练结果均不在本批完成范围。

继续遵循[检测＋动力学架构](contact-model-architecture-20261117.md)，上一批[HOST 观察与生命周期](area-pressure-lifecycle-report-20261117.md)仍作为语义参考。
