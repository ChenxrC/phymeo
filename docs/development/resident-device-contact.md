# Resident 设备接触的公开使用边界

2026-11-11：单页 `prepare_contacts` 的新设备 patch 消费已完成 interpret/CUDA 各67项功能验证及每后端47份同次保存审计。本轮安装包验证尚待执行。分页 `prepare_contact_bundle` 的设备模式已整合，专项验证尚待执行，不能把单页结果外推给它。实际状态以[本轮记录](p65-parity-2026-11-11-progress.md)为准。

## 调用与结果

下列片段承接已经构建的 `TyxWorldBatch`、`CollisionModel`、原材料坐标 bindings 和精确覆盖 allowed pairs 的 `RobotContactMaterial` 映射。它展示公共接线，不是完整资产构造例程。

```python
with batch.prepare_contacts(
    collision_model,
    bindings,
    materials_by_pair,
    narrow_phase="device_patch",
    device_patch_site_capacity=1024,
    site_capacity=128,
    sampling="discrete_entry",
) as contacts:
    accepted_by_world = batch.step_contacts(contacts, h)
    for world, accepted in accepted_by_world.items():
        if accepted and world in contacts.last_accepted:
            receipt = contacts.last_accepted[world]
            entry = receipt.entry_state
            state = receipt.accepted_state
            forces = receipt.law_wrench_fp64
            geometry = receipt.diagnostics["device_patch"]
    query_for_this_entry = contacts.last_device_query
```

默认 `narrow_phase="host_patch"` 保留原 HOST 几何路径。设备模式从同一已接受入步状态做 TYX broad phase 与凸体 patch，不调用 HOST patch 搜索来补完结果。随后 HOST 验证表示转换、按材料点和完整 Jacobian 装配力及 COM 力矩；TYX 推进原完整动力学。两种模式使用同一材料力律和物理发布事务，当前并非全设备力装配。

`last_result` 是 broad-phase 结果，`last_device_query` 是同次设备窄相结果；力和运动必须从 `last_accepted[world]` 的同一收据读取。收据内 `DevicePatchConsumption` 保留原 query、逐 pair 有效 samples、每站点差分及完整 patch 的一阶矩差分。query 绑定的是**入步** epoch、bank、tick 与形状版本，步后是历史证据，不能再当终点的当前查询使用。部分 world 被拒时，其旧收据可以继续保留；务必先检查本次 accepted mask/`last_status` 和收据 tick。

## 形状、所有者和容量

当前资格覆盖 S/C/OBB/H 的十类无序凸体配对，以及普通、affine、floating、floating-affine 的合法 accepted-source 布局。这里 OBB 是 box 资格，不意味着任意凹体、mesh 或柔性面可以走同一路径。材料点转换仍核原 ID、半径、地址、完整候选归属与 source；不按输出修补错源。

| 参数/结果 | 语义 |
|---|---|
| `candidate_capacity` | 保留原 broad-phase 候选容量，与站点数是不同资源。 |
| `device_patch_site_capacity` | 单 query 的完整设备 patch 站点容量；默认4096，显式整数1–4096。仅设备模式可指定。 |
| `site_capacity` | 原每 world 的受力站点上限，可按 world 映射；包括零力站点。不是设备 query 总容量。 |
| UNKNOWN / RESOURCE / ABSENT、错源或转换失败 | 当前整次尝试异常，保留原状态、查询与力收据；不切 HOST、不截断点。 |
| 原逐 world 站点容量拒绝 | 只拒该 world，其余合法 world 可在同一次事务中接受；拒绝 world 保留旧状态与旧收据。 |

设备浮点表示到有效材料点的转换不是免费或无损假设。逐点误差和完整 patch 的原始/有效一阶矩都必须通过已有局部尺度门，完整差分保存在收据；相同合力不自动证明力矩正确。当前 `discrete_entry` 也不是 CCD，快速穿越需要单独的连续碰撞路径。

## 分页接口（已编码，待专项验证）

`pages` 是已构建的 `ResidentContactPageSpec` 序列；每个 world 完整属于一页。公开接线如下：

```python
with batch.prepare_contact_bundle(
    pages,
    narrow_phase="device_patch",
    device_patch_site_capacity=1024,
    site_capacity=128,
) as contacts:
    accepted_by_world = batch.step_contacts(contacts, h)
    page_queries = contacts.last_device_queries
    aggregate = contacts.last_device_patch
```

设备站点上限按每个非空页分别计；`site_capacity` 仍按 world 计。各页完成源绑定和表示转换后，共享同一次力装配、求解与发布。最后一页的 query 或转换失败不得发布前面页的状态或收据。真正空页的窄相结果为 `None`；有 shape 而无 allowed pair 的页仍保留完整源绑定和合格的零行 query。无页或全空页走一次原自由推进。

`last_device_queries` 按声明页序排列；聚合保留各页原 query、raw/source、有效材料点和转换差分，不伪造统一的页 registration。它与单页结果一样属于入步历史。当前最高已声明分页资源边界不变，不能用此入口绕过每页容量或把一个 world 拆到多页。

## 验证范围与后继

本轮多案例包含独立形状参考、部分面、反序/旋转、四类树真实入步速度和驱动/外载、连续三步、滑动两方向、同输入 HOST 对照、末期失败与重试。一阶矩负控使用相同原始局部预算，不为特定位置扩大容差。详[消费合同](../superpowers/plans/2026-11-11-resident-device-patch-consumption.md)。

[B3分页合同](../superpowers/plans/2026-11-11-paged-device-contact.md)把多个 world-complete 页的全部转换合并到一次力装配/推进/发布；每页保持原容量。完成实际多页矩阵、同次证据和源码外安装包之前，仍记为待验。

MuJoCo-TYX 是另一条物理路径：CPU MuJoCo 编译模型、TYX 原生 primitive/约束与混合求解。这里的 resident patch API 不修改其模型或求解器，也不自动扩大它的形状范围。两条路径与 GPU 训练的关系见[当前全景](phymeo-newton-panorama-2026-11-11.md)。
