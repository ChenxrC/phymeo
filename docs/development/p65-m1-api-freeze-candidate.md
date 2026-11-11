# M1 S0 公共 API 冻结候选

**AUTHORING_ONLY / 待父裁定**，2026-11-12；只读父 38f5b0cc。承接[原合同](p65-device-mesh-source-next-contract.md)与[17组案例](p65-m1-case-library-20261112.md)，不改 limits、几何门或矩阵。下列新名字均未实现。

## 公共面与实际可复用类型

    batch.prepare_boundary_query(model: CollisionModel, bindings: Sequence,
                                 *, limits: BoundaryDistanceLimits,
                                 memory_limit_bytes: int) -> ResidentBoundaryQueryPlan
    plan.query(*, margin: float = 0.0) -> ResidentBoundaryDistances
    plan.require_current(result) -> ResidentBoundaryDistances
    plan.last_result -> ResidentBoundaryDistances | None
    plan.last_attempt -> ResidentBoundaryQueryAttempt | None
    plan.stats() -> immutable mapping
    plan.close() -> None

mode/runtime/cache 继承 batch，不另建 runtime；limits/memory_limit_bytes 必填（memory 为正 builtin int），plan 内固定（M12 exact/one-less 使用不同 plan）。margin 非 bool、有限非负。query 持原 batch RLock/_device，仅 accepted，不收 caller Motion/裸 pointer/candidate token。require_current 只认本 plan 最新已发布包，另核 owner/registration/topology/epoch/active/ticks/source_binding；只读 query 不推进物理 epoch。成功 cleanup 后一次 swap，失败保旧 last_result，last_attempt/counters 记录真实失败前缀。

四个现有公开 binding 的 **direct constructor** 与 static 可原样复用：

    ResidentStaticBinding(key, CollisionMotion(start))
    ResidentTreeBinding(key, world, group_index, local_points_m, *, topology)
    ResidentAffineTreeBinding(key, world, group_index, local_points_m, *, topology)
    ResidentFloatingTreeBinding(key, world, group_index, local_points_m, *, topology)
    ResidentFloatingAffineTreeBinding(key, world, group_index, local_points_m,
                                      *, topology, graph_token)

现 [tree](../../src/phymeo/backends/_resident_tree_binding.py):44、[affine](../../src/phymeo/backends/_resident_affine_binding.py):10、[floating](../../src/phymeo/backends/_resident_floating_binding.py):26、[floating-affine](../../src/phymeo/backends/_resident_floating_affine_binding.py):17 的声明可装任意有限 local 点；其旧 compile/from_asset 另有 convex 限制。**M1 新 boundary 编译器不调用旧 compiler 放行 mesh，也不改旧四文件**。首片只收 explicit topology 非 None；from_asset mesh lowering 暂无承诺。static 限 start 单姿态，end/turn/body_orientation 均 None。

## 注册与借用的精确校验

每 model key 恰一 binding；key.world=world 属于本 batch。按 accepted.bindings[slots[world]].kind：5/6、7/8、9/10、11/12 对四种 **exact binding type**，不得用继承关系混同 affine/ordinary。核 source.owner/topology_generation/state_epoch/active、rows/world/specs，注册 row 必须等于 source.rows[world]；固定根 group<physical.n、浮根 group<physical.b。direct topology 等于实际 spec.topology；floating-affine graph_token 另与 owning affine spec 一致。shape 原 IDs/triangles/revision/radius/local 点数及顺序全绑；triangle 是否刚性由安装声明确定，不能套现 CollisionShape.rigid=False 误拒或改全局属性。

现 [ResidentCollisionPlan](../../src/phymeo/backends/_resident_collision_plan.py):430–440 要 (world,body)↔真实 group 双向唯一，static 不别名动态 body。父 8b0cbe71 已纠正案例第2节：同 follower 的 sphere=(world,202,0)、capsule 为 **(world,202,1)**，不是 body203；几何/运动/门/节点不变。body 是应用 ID，不直接作 FK index。

[batch._borrow_collision_state](../../src/phymeo/backends/tyx_world_batch.py):353–400 可直接复用 accepted scope。[TreeGeometryView](../../src/phymeo/backends/_resident_tree_storage.py):39/527 借 meta/parents/kinds/anchor/state/fk；floating view 另有 root。两 affine storage 的 borrow_geometry 都返回 physical.borrow_geometry，所以不能从 reduced s 重建 pose。新私有 BoundarySourceLease 保活 token、runtime/device identity 与完整 source stamps，scope 结束失效；不直接调用依赖 convex plan 私有字段的 borrow_convex_state，只复用其身份校验原则。

## 新不可变 packet（不冒 HOST source schema）

| 类型 | 冻结候选字段 |
| --- | --- |
| ResidentBoundaryDistances | schema="phymeo-resident-boundary-distances-v1"；semantics="threshold-limited-boundary-distance"；source_kind="resident-rigid-device-v1"；owner/registration opaque identity；sequence/backend/topology_generation/state_epoch/active；world_keys、ticks:int32[W]；margin/limits/source_binding/source；pairs/groups/required_candidates/counters |
| pairs / groups | 复用 BoundaryDistancePair / BoundaryDistanceGroup 全字段与状态；group.packet 用下面新来源类型。完整 facet/round IDs、bounds/BVHraw/candidates/计数不删 |
| ResidentBoundaryNarrowPacket | schema="phymeo-resident-triangle-distance-v1"；source_binding、surface:CollisionKey、revision；facet_ids:int64[F]、segment_ids:int64[S]、candidates:int64[P,2]；origin_m:float64[3]、pair_origins_relative_m:float32[P,3]；packed/raw、rows:tuple[TriangleDistanceRow,...] |
| packed / raw | packed 是同次设备 geometry:float32[P,5,3]、radii/scales:float32[P]、owners:int32[P,2]，不叫 HOST uploaded geometry。raw 沿 COL2 candidates[P,18,11]/valid[P,18]/summary[P,16]/meta[P,7]；rows 复用原全字段/ties/status/undefined-normal |
| source 固定键 | registration/model/bindings/shape_keys/shape_revisions/point_ids/facet_ids/facet_vertices/world_layouts/shape_rows/shape_groups/static_flags/world_anchors/authored_constants/uploaded_constants/conversion_deltas/gathered_points/gathered_axes/stamps。offset/ID→row 表放 bindings。authored/delta 仅注册常量，不能伪造 FP64 理想当前 FK |
| ResidentBoundaryQueryAttempt | sequence；status="COMPLETED"|"FAILED"|"UNKNOWN_RELEASE"；source stamps（未取得则 None）、counters、required capacities、completed_groups、raw_prefix、error(type/message)。COMPLETED 只指调用/发布完成，未决距离行不是通过 |

所有数组 bytes-backed readonly、嵌套 mappings/tuples 递归 immutable，无 callable device handle。opaque identity 内存按 is 验，保存用同记录 ID，不用 Python hash 冒跨运行认证。source_binding 绑定注册+同次 accepted stamps，不能替代 owner 校验。group origin 为实际 producer 公共 anchor；pair frame 由设备三角首点生成，HOST只读取/核绑定，不能搜索或修点。

## 内部接缝及关闭保护

[TreeBuffers.hits](../../src/phymeo/backends/_tyx_collision_bvh.py):22 的 shape=False 会上传 HOST bounds，shape=True 只借旧 engine._bounds；两者都不是现成 M1 入口。建议新增 hits_device(tree, *, leaf_bounds, query_bounds, query_count, params, source_lease)，constructor 加 device_inputs=False，True 不分配旧 upload leaf/query 缓冲。device输入只收新 plan 同 runtime 注册句柄，params 是整数拓扑。新 query 内 ledger-aware engine adapter 提供原 _scratch/_spatial_upload/_spatial_output/counts 接口；不实例化旧 ExitStack-owned TyxCollisionBatch，不需改 tyx_collision.py。旧 hits 默认路径原样。

[COL2 adapter](../../src/phymeo/backends/_device_triangle_distance.py) 的 _prepare 在 HOST 打包五点/尺度，M1 禁调用；仅抽取原核 launch 与 raw/row 校验供设备 pack 后共用，保旧 HOST TriangleDistancePacket/schema。原 _tyx_triangle_distance.py 不改。

**关闭是必要共享 delta，当前未具备。** TyxWorldBatch.close:904 先置 closed，finally 无条件父 ExitStack.close 并把 device_owned_bytes=0；仅 no-op lease 不能阻止提前关 runtime。父最小新增 _boundary_plans 与强引用 registry，以及 lock 内 _register_boundary_runtime_lease(lease)、_release_boundary_runtime_lease(token)、_quarantine_boundary_runtime_lease(token, ledger) 三个私有 hook。每个 persistent/query DeviceResources 在分配首 buffer 前注册 lease；它强持 batch/runtime/token，正常 ledger runtime-close 回调只注销、不关 runtime。任何 buffer free 抛异常，ledger 的全局 quarantine 保 lease→batch/runtime，父 registry 也保 ledger；不减失败 buffer live bytes。

accepted bank 的 _Lease.active 是**查询 scope**，runtime lease 是**数组生命周期**，两者不能混用。query finally 使 bank loan 到期，但 UNKNOWN runtime lease 必须继续存活。batch.close 在进入现有 close 主体/父 ExitStack 前先查 sticky unknown，正常时尝试关闭全部新 boundary plans（独立 siblings 仍清理），再核 **registry 全部已正常注销**；unknown 或未决 lease 均阻止父栈关闭，保全部父资源/计数，标 unavailable 并抛原错误。UNKNOWN 后的 query/step/close 只拒绝、不重试 free；不依赖最后一条 Python局部引用。仅全部已注销时才走原普通 close；不能用清父 siblings 触发其 runtime callback。M15/M16 验此新共享保护，旧 COL2 自有 runtime 测试不能代替。

## 精确范围与父裁定项

**NEW**（src/phymeo/backends/）：_resident_boundary_source.py、_resident_boundary_query.py、_tyx_boundary_source.py；NEW tests/reference_newton/rr/resident_boundary_source_reference.py、tests/collision/test_resident_boundary_query.py、test_resident_boundary_query_protocol.py；源资格后 NEW tests/installed/check_resident_boundary_query.py。**MOD**：COL 仅 _tyx_collision_bvh.py、_device_triangle_distance.py；父仅 tyx_world_batch.py、phymeo/__init__.py。四 binding/storage、_resident_convex_source.py、旧凸体/triangle数学核与资源 ledger不改。

待父一次裁定三点：(1) 上述 direct-topology 准入与新 packet/attempt 名字段；(2) M1 按 COL3 的显式 world/body/mask/excluded 过滤，**不继承**旧 convex 的 tree_dynamic_pairs 默认拒绝/隐式 joint邻接排除，需要排除的 link 用 model.excluded；M02 sibling 不受此差异影响；(3) 作者 tests-first 补精确 buffer 维度、anchor/bounds 内部表示及 memory 字节式，不能从运行峰值倒填。源码不能确定尚未实现的 buffer ABI；这三点确认后可发 S0/S1 grant。本稿无实现或项目执行。
