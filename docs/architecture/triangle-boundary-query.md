# 查询球／胶囊到有限三角面的边界距离

`CollisionWorkspace.boundary_distances()` 从公共注册和当前坐标自动生成候选，再调用 TYX BVH 与有限 segment–triangle 距离核。它回答的是“声明邻近带内有哪些原始面，以及这些有限面到球心／胶囊轴的距离”，不回答实体内外、连续碰撞或接触力。

本批状态（2026-11-12）：父任务的 COL3 源码资格已通过 75 个新节点和 11 个旧发布邻居；产品检查点为 `1c796893`。新 wheel 的源码外消费者仍待父任务执行。下列公共类型 re-export 随本批由父集成；本文示例未单独执行，不能视为安装包验证结果。

## 最小公共调用

所有 API 从 `phymeo` 获取，调用者不必导入内部模块。`interpret` 与 `cuda` 使用同一接口；这里显式选择 CUDA。

```python
import numpy as np
import phymeo as ph

surface_key = ph.CollisionKey(world=7, body=101, shape=0)
sphere_key = ph.CollisionKey(world=7, body=202, shape=0)
model = ph.CollisionModel((
    ph.CollisionShape(surface_key, "triangle", vertex_ids=(11, 37, 91)),
    ph.CollisionShape(sphere_key, "sphere", vertex_ids=(11,), radius=0.125),
))
vertices = np.array(((0, 0, 0), (2, 0, 0), (0, 2, 0)), dtype=np.float64)
motion = {
    surface_key: ph.CollisionMotion(vertices),
    sphere_key: ph.CollisionMotion(np.array(((0.5, 0.5, 0.25),), dtype=np.float64)),
}
limits = ph.BoundaryDistanceLimits(
    groups=1, candidates=1, vertices=3, faces=1, segments=1,
)
margin = 0.2

with ph.CollisionWorkspace(model, backend="cuda") as workspace:
    first = workspace.boundary_distances(motion, margin=margin, limits=limits)
    workspace.require_boundary_distances(first, motion, margin=margin)
    assert workspace.last_boundary_distances is first
    assert workspace.last_result is None  # 独立于普通 detect 的发布。

    for group in first.groups:
        frame = np.asarray(group.packet.inputs["origin_m"])
        for row in group.packet.rows:
            print(group.surface, group.round_keys[row.segment_id], row.status)
            if row.status == "RESOLVED":
                axis_world = frame + np.asarray(row.axis_point_relative_m)
                surface_world = frame + np.asarray(row.triangle_point_relative_m)
                print(row.distance, row.gap, axis_world, surface_world, row.normal)

    # 新坐标形成一个新的快照；不是只更换标签。
    deformed = vertices.copy()
    deformed[2, 2] += 0.125
    next_motion = dict(motion)
    next_motion[surface_key] = ph.CollisionMotion(deformed)
    try:
        workspace.require_boundary_distances(first, next_motion, margin=margin)
    except ValueError:
        pass  # 完整输入绑定不同，旧证据不能替新快照作证。
    else:
        raise AssertionError("changed coordinates must invalidate old evidence")

    with workspace.transaction():
        second = workspace.boundary_distances(next_motion, margin=margin, limits=limits)
        assert workspace.last_boundary_distances is first
        # second 此时是未提交结果；require_boundary_distances 不接受它为 current。
    assert workspace.last_boundary_distances is second
    workspace.require_boundary_distances(second, next_motion, margin=margin)

    # 仅收紧容量；异常保留上次已提交结果，不截断候选。
    try:
        workspace.boundary_distances(
            next_motion, margin=margin,
            limits=ph.BoundaryDistanceLimits(
                groups=1, candidates=0, vertices=3, faces=1, segments=1,
            ),
        )
    except ph.BoundaryDistanceCapacityError as error:
        print(error.resource, error.required, error.limit)
        assert workspace.last_boundary_distances is second
```

胶囊注册使用 `kind="capsule"`、两个原始 `vertex_ids` 和正半径；对应 `CollisionMotion.start` 为两个有限轴端点。球使用一个中心，内部转换为零长轴。顶点 ID 可在不同 shape/world 中重复，完整 `CollisionKey` 才是所属对象。mesh 的 `triangles` 使用原始材料顶点 ID；保留原 facet 表及其绑定，不把 mesh 当成凸包。

## 邻近带、行与来源的含义

`margin` 必须有限且非负。候选盒包住整条有限轴，再向外扩张 `radius + margin`；实际设备 BVH 返回原 facet 身份。盒子允许保守的额外候选，所以有候选不等于发生碰撞。空带状态 `no_boundary_candidate` 只表示这个查询带内没有面候选，既不保证带外全局最近面，也不证明封闭 mesh 中的球位于实体外。

| 记录 | 可使用的信息 |
| --- | --- |
| `result.pairs` | 完整 shape 配对；保留跨世界、同刚体、mask、excluded 等过滤原因，以及同类别未适用状态 |
| `result.groups` | 每个 surface 的原 round key、材料顶点 ID、facet 材料三元组、完整候选映射、同次 BVH 原始 bitset/计数 |
| `group.packet.rows` | 原 facet/segment 地址、有限轴参数、重心权重、两侧见证、距离、可用方向和明确状态 |
| `group.packet.inputs/raw/uploaded/conversion_deltas` | 完整源输入、真实设备结果、上传值与转换／发布误差证据 |
| `result.source_binding/source` | 整个模型、motion 材料顺序坐标、revision、半径、过滤和 margin 的绑定；不是仅信 query 序号 |

`distance` 是有限轴到有限三角面的非负距离；`gap = distance - radius`。这个 gap 不是封闭实体的有符号穿透深度。见证点以该 packet 的 `origin_m` 为参考；`normal` 若存在，指向从轴见证到面见证的方向。零距离、`CONDITIONING` 或 `DEGENERATE` 可能没有法向，调用方必须保留该状态，不能补造响应方向。`axis_parameter`、barycentric 和 facet ID 是本快照的材料地址，不是跨时间追踪接触的承诺。

原工程门沿用距离 `1e-4 L`、见证 `2e-4 L` 和法向 `2e-3`，其中 L 来自该 pair 原始局部边／轴／半径长度。大世界平移不能放宽误差门。这里的距离输出不是严格认证区间；原始数据和误差记录应与结果一同保存。

## 快照、容量与事务

本片只接受完整的当前有限快照：sphere/capsule/triangle/mesh，surface skin 必须为零。非恒定 `end`、turn、独立 body orientation 及未支持的注册类型会明确拒绝。坐标打包来自 HOST，不是 resident device-vertex 的零往返入口；内核计算和 BVH 候选使用指定 TYX 后端。

`BoundaryDistanceLimits` 只能收紧上限，且字段必须是真正的 builtin int，不能用 bool：总共最多 32 个有效 surface group、8192 个候选；每 group 最多 512 个原顶点、128 个面、64 个 round axis。零容量合法。超限的 `BoundaryDistanceCapacityError` 给出实际 `resource/required/limit`，所有候选计数完成后才进入窄相，不以截断制造成功。

没有外层事务时，完整查询成功才更新 `last_boundary_distances`。外层 `transaction()` 内的多个查询按同一 owner 的暂存结果递增序号，只在外层成功时发布最终结果；异常中止保留旧公开结果和旧 BVH 布局。已执行的设备工作与失败诊断不会被抹掉。未提交结果、其他 workspace 的 owner、被替代的旧结果以及与当前输入不匹配的证据都会被 `require_boundary_distances` 拒绝。

## 下一层连接

普通 `detect`、`last_result` 和既有物理消费语义保持独立。后续要分别补 resident mesh 顶点来源与 epoch 绑定、完整变形/运动路径和 CCD、持续材料身份，再冻结把距离／见证送入力学的表示与载荷合同。不能用本接口的一个成功 distance row 直接宣称实体碰撞、安全时间步或可施加的接触力。

详细交付与边界见[COL3 源码报告](../development/p65-triangle-distance-dispatch-report.md)和[源码外消费者协议](../development/p65-triangle-dispatch-installed-report.md)。