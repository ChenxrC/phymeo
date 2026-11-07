# F2 浮根仿射 mimic：T2 standalone 候选

2026-11-07，**产品候选 UNEXECUTED**。原隔离树从 `24506a0dd4d30931fc866fbc17b49bed424894aa` 继续，未 merge 父。作者只做文本、文件 IO 和 Git；没有 import、collection、数值 probe、测试、lint/format、build 或设备执行。

父在 T1 导入 `bf3cadb` 后已执行 104 首 RED：旧浮根 loader 对 mimic 的明确拒绝及旧固定根 mimic 对同一正质量资产的 P/c 识别先通过，随后准确失败在新 loader 缺失，1 failed / 0.80 s / RAW1。此原失败保留，作者未复跑。下述实现与全部后续案例仍待父集成验证。

## 五个新增文件与接缝

- `dynamics/_mimic_coordinates.py`：唯一通用纯图编译器 `compile_mimic_coordinates(names, types, lower, upper, velocity, effort, relations_tuple)`。frozen `MimicCoordinates` 返回 `indices / forward / independent_indices / matrix / offset_vector / constraint_rows / joint_names / lower / upper / velocity_limit / effort_limit`。矩阵用独立 bytes 存储；原行次序和拓扑处理次序分别保留。有限有界标量 metadata，含同类型、非零倍率、环、重复从动、FP64 不可表示映射/边界和空联合区间检查。
- `builders/_mimic_xml.py`：`extract_mimic_relations(order, incoming)` 使用旧严格 XML helpers，验证整组标签后才从本次已读树中移除，返回非空 raw tuple；图合法性只交坐标编译器，不另维护图实现。
- `dynamics/floating_mimic.py`：`FloatingMimicRobotAsset(tree, relations_tuple)` 包装真实完整浮根资产，`FloatingMimicStep` 返回独立状态、完整状态和原类型的 force receipt。包括 `configuration / full_state / reduce_state / collision_motion / step`；static collision motion委托完整 tree，interval 继续明确拒绝。
- `builders/floating_mimic_urdf.py`：`load_floating_mimic_urdf(path, *, gravity=..., collapse_fixed_joints=False)` 只调用一次 `_read_tree`，提取 mimic 后调用父提供的 `_floating_from_tree(name, root, links, order, incoming, gravity, path, collapse_fixed_joints=...)`。不复制浮根或固定组装配，不生成临时 URDF。
- 本报告。

父独占的旧接缝未在本树修改：`joint_mimic.py` 改委托唯一坐标编译器；`mimic_urdf.py` 改委托 XML 提取器；`floating_urdf.py` 提供共享 `_floating_from_tree`。原 `MimicRelation / MimicRowForce / MimicLoads` 类型和路径不迁移。本候选 loader 在这些父接缝未导入的原树上不可直接使用，不将此依赖误报为已验功能。

## 状态、真实力与候选推进

完整 `tree` 保留全部物理 body、COM、惯量、固定工具和材料 owner。包装资产的 scalar `joint_names / lower / upper / velocity_limit / effort_limit` 仅对应独立轴；`matrix` 是 P，`velocity_projection` 是 bytes-backed `S=diag(I6,P)`。独立状态 q/v 长度为 7+k/6+k，完整状态为 7+n/6+n。根 q 是 authored origin/XYZW，根 v 是该根物体的世界 COM 速度与世界角速度，不是全系统 COM。

`step(state, *, effort, dt_s, external=None)` 的 effort 只有独立电机 k 项；完整 external 使用 `FloatingRobotInputs`，按 n+1 个真实 body 输入 COM 世界力和矩，n 项 external.effort 必须全零。输入和外载先复制、验证；独立电机只写到原独立物理轴。

在同一完整入态调用原 `floating_dynamics_terms` 取得 M、bias 和原 COM 载荷投影。该函数也计算它自己的完整加速度诊断，本候选不使用该无约束加速度推进。实际候选只求约束方程 `SᵀMS`，展开已约束的末速度，然后调用原 `_drift_floating_state` **一次**。验证完整漂移仍满足关系/限位后，用同一常数映射提供一致的 full state；没有先做无约束物理步再修补从动坐标。

回执中 `acceleration_full / actuator_effort_full / contact_effort_full / constraint_effort_full` **全部为 6+n**，前六项分别是根 COM 线分量和根角分量，后面才是物理 scalar 轴。旧 `MimicLoads` 的仅 joint 文档不应用于此 `FloatingMimicStep`：这里已在新类型和 step 文档明确完整维数。`contact_effort_full` 沿原字段名包含除重力外的外部非电机载荷，不限定为碰撞力；重力在 bias。

完整余额 `M*a+b−motor−external` 独立核根六维，再按逆拓扑恢复每条原关系乘子，核全部 Gᵀlambda 余额和 Sᵀreaction。总约束功及每行功均按真实入步完整速度记录并审核。`row_forces` 保留中间从动兼驱动节点的各原行贡献。所有向量回执独立不可写；无内部接受态或计数器，拒绝不会改变输入，合法同输入重试不消费任何历史。

## 冻结验证与未完成域

原四个 T1 测试/资产/案例册的字节保持不变；F01 的 1/14、5/14、−4/7、共轴条件和16步，F01-G 扇出/多级原行 KKT/联合限位/失败重试，F02 原四部件/三姿态/两档独立 bias 差分/完整鞍点/材料锚点/协变输入和误差门均未调整。正式实例数由父后续 collection 记录。[固定案例与 oracle](../development/p65-floating-mimic-casebook.md)

此 T2 不包含 scene/PD、接触响应新律、浮根 pads、CCD 响应、CUDA 求解或性能结论。T3 的共同参与者事务与 F03 有限受载窗口还未授权/实现。父后续只运行新增完整单文件与共享抽取确实影响的旧图/普通浮根兼容节点；没有要求重跑 F1 或历史已绿矩阵。

原 100 g mimic 持稳在 1.048 s 滑移 3.005689 mm，仍保留超过 3 mm 门的原 RAW1。此候选不能把浮根约束和无摩擦窗口等同于持续静摩擦或抓取成功。[原失败范围](p65-force-events-filtered-batch2.md)
