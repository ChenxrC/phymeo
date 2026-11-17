# 异构模型的面积接触与统一设备观察

2026-11-17，codex/exp；实现提交 f01b5505a15b8401074718240fb82b3b10c4e66c。dev 未改，未 push。

本批把“同模型的面积接触设备记录”连接到现有异构批次：不同自由度、不同容量、不同接触模型的环境共同提交一步，再在同一观察帧读取端点状态、入步状态、原生接触和面积载荷。既有 readback 是显式诊断读回；to_torch 产生调用者独有的 CUDA 副本。数值不从 HOST 回执重建。

## 实现与实际能力

MuJoCoTyxModelSpec 新增 area_pressure_pairs=()。声明参与预编译、精确分桶、实际设备分配清单、运行时构造及 checkpoint 绑定。同一 XML 的不同面积接触声明不会误合并。非面积模型保留原 checkpoint 绑定字段。

batch.area_pressure_configuration_id(world) 总是从当前物理 owner 取得 stamp，reset/restore 更换 owner 后也适用。batch.step_area_pressure(h, surfaces, expected_revision=...) 要求 surfaces 恰好覆盖全部面积模型的 world；每个 world 按声明次序供应所有 pair 的 surface，包括空面。原生模型无需在 surfaces 中放占位。

所有 surface 先 pack 检查，随后沿现有 prepare-all / validate-all / commit-all 路径推进。任何一个 bucket 失败，全部环境维持原发布。一次性 packet 在成功或失败后都清除。普通 batch.step(h) 不会静默漏掉面积接触。

这些 surface 仍由调用者依据当前状态生成。本批没有建立异构 Scene 自动裁剪封装，也没有改变既有面积法向接触律、阻尼限制、精度阈值或求解容量。

## 同一个观察帧

调用 batch.prepare_device_workspace(include_area_pressure=True) 开启面积观察，也开启原生入步记录与入步 body 观察。端点 body 观察仍由 include_bodies 独立控制。

| 字段 | 意义 |
| --- | --- |
| q/v/act/clock 与原有效掩码 | 各模型接受后的端点状态，统一按最大宽度填充 |
| entry_q/v/clock/meta 等 | 产生本步力的输入状态与时刻，不能当成端点 |
| area_declared_mask [world,1] | 该 world 是否声明面积接触模型 |
| area_meta [world,4] | 记录有效、样本数、正法向力数、overflow |
| area_ids [world,capacity,6] | 有效、pair序号、记录内sample序号、body A/B、世界原点矩有效 |
| area_data [world,capacity,15] | 点、法向、局部k/phi、法向力、A方力、A方共同世界原点矩 |
| area_measure_words [world,capacity,2] | 原始FP64 dA的两个i32位字，无FP32面积转换 |

capacity 取面积模型的 constraint capacity 最大值，原生模型更大的容量不会扩大面积字段。无面积模型时 capacity=1，全零且 declared mask=0。声明面积但初始化/重置尚无记录时 meta.valid=0；成功求解空面时 meta.valid=1、样本数0。因此三种情况有明确区别。

binding.area_pair_names 按 model key 保存声明。frame.area_catalogs 按 world key 保存 identity/stamp/sample_ids/source_ids，只含身份元数据。pair序号与sample序号必须在本模型、本帧目录内解释，不能视为全局或跨帧稳定接触ID。

新 scatter 读取当前 accepted 设备银行，验证计数、填充和精确pair/body对应，写工作区候选银行；全成功后才发布新 frame。失败刷新保留旧 frame，物理修订变化则使旧 frame 失效。设备数值与 HOST 身份目录由同一个锁内刷新绑定。原始 dA 位及 I3b 的 moment-valid 语义不变：无效矩的零 sentinel 不是“真实零力矩”。

原生接触力作用于 geom2；面积力作用于 compliant A，B取反。未将两种身份伪装成相同 geom ID。既有 ManipulationSuccessReader 依赖几何体角色，因此对面积声明明确拒绝，直到补上 body-pair 成功判定适配；不能用关闭 include_area_pressure 绕过。统一数值观察已经可供 CUDA 消费者使用，但本批不声称原训练 Env 已支持面积任务。

## 接口使用顺序

```python
spec = MuJoCoTyxModelSpec(
    "model-a", compilation,
    constraint_capacity=64,
    area_pressure_pairs=(("compliant", "rigid"),),
)
# 用 models/worlds 建立 batch，按其当前 configuration_id 生成 surfaces。
accepted = batch.step_area_pressure(
    batch.timestep, surfaces,
    expected_revision=batch.checkpoint_revision,
)
with batch.prepare_device_workspace(include_area_pressure=True) as workspace:
    frame = workspace.refresh(expected_revision=batch.checkpoint_revision)
    tensors = workspace.to_torch(frame)  # CUDA，copy-owned
    identities = frame.area_catalogs
```

surfaces 是调用者提供的真实几何结果，上例不提供虚构几何生成器。interpret 使用 workspace.readback(frame)。to_torch 副本后续不会因 refresh/reset 被覆盖，调用者修改副本也不会改写物理状态。

## 验证

固定夹具有四个非连续乱序 world：两个共享 hinge+slider bucket（nq/nv=2/2、容量8），一个自由根面积模型（7/6、容量16），一个原生球/平面模型（1/1、容量32）。短步结果与分别运行的独立leaf逐字段比较，未使用旧长轨迹作为新增结果。

- 新 interpret：6 passed，10.10秒。
- 新 CUDA：7 passed，39.17秒，含真实 CUDA Torch 副本。
- 受影响兼容：16 passed，75.64秒，覆盖原生8世界IO、零宽字段、末bucket三种故障、重入拒绝和旧reader准入。
- 六个Python文件 Ruff / format-check 全通过；测试前后SHA一致。

新测试包括成功空面/无声明/初始无记录、精确dA位、原生与面积混合、乱序身份、双方力/矩、最后bucket失败及重试、缺失/过期/超容量输入、checkpoint与选择reset/restore、更换owner后的读取、相同XML不同声明隔离、禁HOST数值源刷新、刷新失败保前frame，以及有效body编号但错误pair身份的拒绝。原 review finding 已由设备静态body表精确比对修复。

只有首次Ruff import排序失败并修正，未出现数值/产品测试失败，未调整物理参数或门槛。原始命令、guard、exit与源码快照保存在 [本批证据](../../artifacts/area-pressure-heterogeneous-observation-20261117-01)。13新＋16兼容共29节点全部通过，无skip。

## 下一步及边界

下一步先复用当前单场景面积几何资产，让异构 batch 从各自当前状态生成新的接触面，再整批提交，验证连续运动、选择reset及晚失败。之后将有限面裁剪设备化，并比较包含几何、同步、求解和导出的整步成本。

目前仍有HOST几何生成和求解诊断，bucket顺序调度，并非融合成一个GPU kernel；无新增规模或性能优越结论。尚未新增面积摩擦、FFT非局部材料、多模型面积训练成功判定或变形体面积响应。
