# 下一片合同：真实设备来源的有限 mesh 边界查询

状态：**RESEARCH_ONLY / 待父冻结，不授权实现或执行**。2026-11-12。只读父 `3e37c1af` 对应源码、已验 COL2/COL3 与固定 Newton 1.7.0 源码；父在进行的训练/安装工作未改。目标是替换当前查询的 HOST 当前坐标上传，保留有限边界距离语义，不扩大为实心碰撞、CCD 或载荷。

## 1. 最小可交付片与真正前置条件

建议分成两个具有同一来源协议的节点：

- **M1：刚性设备来源，先开发/验收。** sphere/capsule 对 triangle/mesh，来自同一个 TyxWorldBatch 的四种已接受物理树布局及明确静态资产。树的完整 FK、局部材料点、世界 anchor 在设备生成当前顶点；facet bounds、BVH 候选和逐 pair 几何也在设备生成。提供独立公共 query，不改旧 `CollisionWorkspace.detect/boundary_distances` 或任何受力路径。
- **M2：变形来源，单独冻结后开发。** 接同一查询 kernel，但必须先有真实物理所有者提供的、带 accepted/candidate 身份与生命周期的设备顶点 bank。当前 inspected cloth/node 路径没有这种持久 accepted loan；把 HOST ParticleState 上传后称“device source”不满足 M2。M1 不等待新软体状态架构，M2 未完成必须明确保持 unsupported。

两者不能以同一个“mesh 已支持”标签合并。M1 不依赖失败的 B-point 曲面触点表示，COL2 距离核不需要新曲面 patch。

## 2. 当前源码可直接复用什么

| 已读接口（父源码位置） | 可复用能力 | 尚缺的接缝 |
| --- | --- | --- |
| `backends/tyx_world_batch.py:353` `_borrow_collision_state` | 持原 batch lock/runtime；借 accepted active bank、ticks、owner/topology_generation/state_epoch 与四树视图 | 不能借 standby 候选或在 context 外保存指针；新公共入口由父接 |
| `_resident_tree_storage.py:39,527` 的 `TreeGeometryView/borrow_geometry`；另三 storage 的同名方法 | 普通/affine/floating/floating-affine 完整物理 FK、anchor、真实 body/row 布局。affine 已是展开物理 FK，无需 HOST 解 reduced 坐标 | 没有 mesh vertex/facet 注册；不可把 affine 坐标直接当 pose |
| `_resident_convex_source.py:99,161` 的 `_qualified_layouts/borrow_convex_state` | owner、world、source layout、group、epoch/active 交叉验证及 loan 到期语义，可作为协议依据 | 是 convex plan 专用字段，不可直接把 mesh 塞成 hull；应新建 boundary 注册，保持旧 convex 路径不变 |
| `_resident_tree_binding.py:16,45,93` 等四 binding | 原局部材料点、真实 body 安装与完整 topology/from_asset 来源 | 当前 `_FINITE_SHAPES` 只列 sphere/capsule/hull/cylinder/cone/ellipsoid。mesh/triangle 需要显式新绑定，不更改原 convex 准入含义 |
| `_tyx_convex_query_kernels.py:120` `gather_convex_pose` | 设备 FK 到 anchor-relative 当前点的运算模式已验 | 新 boundary gather 不必调用 GJK/query，也不能通过旧 query 的 HOST 输出取顶点 |
| `_tyx_collision_bvh.py:22` `TreeBuffers.hits`；`_tyx_spatial_kernels.py:45,69` | 原 refit 与 stackless traversal、完整 raw bitset/end marker 可复用 | `shape=False` 仍上传 HOST leaves/queries；缺接受设备 leaf/query buffers 的受控入口。现 `shape=True` 借的是 shape broadphase bounds，不能冒充 triangle leaf bounds |
| `_device_triangle_distance.py:254,318,576`；`_tyx_triangle_distance.py:7` | COL2 已验的 18 候选、距离/见证/status、真实资源 ledger 可复用 | `_prepare` 现在在 HOST 为每 pair 打包五点/局部尺度，再上传；需要设备 pack 与独立 source packet 构造接口，不可继续调用 HOST pack 后换标签 |
| `geometry/_triangle_distance_dispatch.py:278` | COL3 的完整过滤表、原 facet 映射、总容量先检查、aggregate publication 语义可复用 | 当前来源是完整 HOST motion binding，不是 resident epoch；保持原 API，新增结果来源类型 |
| `tyx_node_math.py:36`、`tyx_composite_cloth.py:1,66`、`tyx_coupled.py:199` | 已有真实 TYX 节点数学、耦合暂存数组和失败门 | node/composite 是 HOST FP64 存储及调用结果；coupled 当前在局部 scope 上传/释放设备位置。没有可跨调用借出的 accepted deformable vertex bank，不能偷借局部临时数组 |

M1 可以直接借 FK/anchor/ticks，局部资产/拓扑只在注册时上传；**当前 mesh 顶点并不是已有可直接借的单一数组**。新增 gather 后才有本次查询的设备顶点 scratch。M2 则还缺其物理 producer 的 bank/loan/提交协议。

## 3. 拟冻结的最小公共 ABI

建议 `batch.prepare_boundary_query(model, bindings, *, limits, memory_limit_bytes)` 返回独立 `ResidentBoundaryQueryPlan`，其 `query(*, margin=0.0)`、`last_result`、`last_attempt`、`require_current(result)`、`close()` 与旧 query 分开。名称由父最终冻结；不是给旧 query 添加含混的 `device=True`。

M1 binding 明确 `CollisionKey(world, body, shape)`、真实物理 group、layout、完整 model/topology token、局部 vertex IDs/coordinates、原 facet IDs/有向 material triples、radius、filter/revision。static 只接受固定注册 geometry，不把逐步 HOST 更新的 motion 冒成设备物理。模型只准 sphere/capsule/triangle/mesh；其他 shape、mesh–mesh、round–round 先明确拒绝或列 non-applicable，不隐式走 HOST fallback。

内部 `BoundarySourceLease` 仅由已注册物理 owner 生成，包含 owner/registration/topology_generation/state_epoch/active、world→layout/row、每世界 ticks、source kind/frame、runtime/device identity、只读 handles 与 lease token。用户不能通过手写 epoch 或裸 device pointer 取得资格。完整查询持 producer 锁和 runtime scope；返回包不含仍可调用的借用指针。

M2 后继同 ABI 另用 `accepted_vertex_bank` 来源，必须保 point-ID→bank-row 映射、完整坐标表示/anchor、顶点数、device generation、producer 的 accepted/candidate 标志与 readiness。若与刚体源来自不同 runtime/context，先解决可审的共同借用/同步协议；不能把跨 runtime 指针直接混用，未有合格方案时拒绝。新 source owner 不能以一次任意 HOST 上传替代真实物理连续步。

## 4. 设备数据通路与不可隐藏的 HOST 工作

1. HOST 只做一次不可变注册、过滤/ID映射、BVH 拓扑构建、容量/类型准入。每 query 在同一 accepted loan 内发出设备 gather，生成完整材料顺序的当前 mesh 顶点及球中心/胶囊轴；完整 stamps 核每个 source 的 layout/row/world/tick。
2. 设备生成三角 leaf AABB 和 axis AABB（扩 radius+margin），按原有限非负邻近带语义 outward 包围。复用 refit/traversal 算法，新增 device-buffer 入口，不下载顶点计算 boxes 再上传。原 bounds 精度门保持；非有限/无法可靠包围时明确拒绝。
3. 所有原过滤 eligible pair 的真实 BVH raw bitsets/计数先完成，算出所需总 candidate，再检查每组和全 query 容量。首片允许将**候选整数 bitsets/计数**读回、在 HOST 只作稳定 ID 编排，再上传整表；此处是明确控制往返，不叫全设备 compaction/零往返。不得据几何值或输出状态挑选子集。
4. 设备按完整 candidate 表打包 COL2 五点/半径/尺度/owner，随后调用原 `triangle_distances` kernel。公共 anchor 与 pair 首顶点先相对化，再做局部几何；不能先把局部 offset 加到巨大绝对 FP32 world 坐标后再减 origin。L 仍只取本 pair 原局部边长/轴长/半径的 max，不加世界位置、shape 间距或 scene extent。
5. 允许 kernel 完成后读回完整 source audit、同次 packed geometry/转换证据和 raw output，HOST 仅验绑定/重构/有限字段，不搜索最近点、计算 BVH bounds、修改 bary/normal 或填补未决行。计真实 H2D/D2H、同步、launch、owned/scratch bytes；不能因源在设备就声称没有读回。
6. 两见证保原 facet material triple + bary、原轴 endpoint IDs + t、原 radius 与 pair frame；共享边/重合 facet 保多行，不以坐标去重改变材料身份。0/邻方向带保 `normal=None`，CONDITIONING/DEGENERATE 不删。保持 distance `1e-4 L`、witness `2e-4 L`、normal `2e-3` 的既有工程门。

结果语义继续是 `threshold-limited-boundary-distance`；明确 `source_kind=resident-rigid-device-v1`，未来 M2 使用不同 source kind。空带仍非 closed-solid separated，未承诺带外全局最近面。原 FP64 静态注册→实际 FP32 常量转换 delta 和设备当前 geometry 都保留；当前 FK 本就是已验 FP32 状态，不能伪造一个 HOST “理想当前坐标”作来源。

## 5. 身份、容量、原子发布与资源

- 注册保全 world/body/shape、shape revision、原 vertex/facet IDs、storage-order map、filters 和 topology token。同 local ID 跨 world 合法，但不得跨 world 配对；同刚体/group、mask、excluded 的原规则完整保留。任何 world 缺源、重复映射、错 owner/layout/active 先拒绝。
- 复用 COL3 硬域：每 surface 512 vertices/128 faces/64 round axes，全 query 32 groups/8192 candidates；typed limits 只能收紧。整 query 的几何 bounds/计数完成才准 narrow，exact capacity 通过、少一报告真实 required，不截断。scratch bytes 另单独受 memory_limit 约束。
- 公共 query 只读 accepted bank；步进/reset/restore 成功后旧包不再 current。M1 不提供查询 standby 物理候选的公共口。未来接步内 consumer 时，需独立 `candidate-token` 和联合 commit hook；失败候选绝不能以新 accepted epoch 发布。
- 先准备全 immutable result/source/raw/映射与 pending BVH layouts，所有 groups/readback/重构/清理成功后再一次 swap `last_result`。晚失败保旧结果/布局和真正 attempt prefix，不回滚已发生的 kernel 计数；查询成功不会推进物理 state_epoch。同 source 同输入重试不混旧 stamps。
- 借用 bank/runtime 不归 plan free/close。新 scratch 复用已验 `DeviceResources` + no-op runtime lease，正常释放完成才减 live bytes。UNKNOWN_RELEASE 必须强保留 scratch owner 与实际 runtime，后继 query/close sticky 拒绝且不盲 free；父必须在共享 runtime owner 的 close 前加入此未知借用隔离检查，不能由 plan 单方面宣称 runtime 可关闭。独立 siblings 仍清理并保存原异常。
- 借用到期后只允许使用不可变保存包；不得把 raw handle 藏在 result 中延长可调用期。关闭/失败证据既不污染 accepted 物理 bank，也不被当作新 query 成功。

## 6. 精确文件责任建议

| Owner | 下一 M1 允许范围（待父冻结） |
| --- | --- |
| COL source | NEW `backends/_resident_boundary_source.py`（专用 rigid/static 注册、lease/source proof）；NEW `backends/_resident_boundary_query.py`（plan、typed source packet、容量与发布）；NEW `backends/_tyx_boundary_source.py`（gather、leaf/axis bounds、pair pack） |
| COL source / 父审核 | MOD `backends/_tyx_collision_bvh.py` 增 device-buffer 受控入口；MOD `backends/_device_triangle_distance.py` 仅抽取共享运行/结果校验接缝，保原 HOST API 和资源语义。原 `_tyx_triangle_distance.py`、凸体 query/patch 数学核保持已验字节 |
| 父共享 owner | `backends/tyx_world_batch.py` 公共 prepare/close 与 loan-owner 保护；`phymeo/__init__.py` 公共类型导出。若复用原四 binding 的 asset 安装证明，准入必须分离 boundary 域，不能扩大旧 convex domain；需要改哪一 binding 先逐文件申报 |
| 原测试作者 | NEW `tests/reference_newton/rr/resident_boundary_source_reference.py`；NEW `tests/collision/test_resident_boundary_query.py`、`tests/collision/test_resident_boundary_query_protocol.py`；功能资格后 NEW installed 公共消费者 |
| M2 后继 owner | 真实 deformable producer 的 accepted/candidate device bank 与 lease adapter；其文件表须先查具体 producer 再冻结，当前不授 node/composite/coupled 改写 |

同一批最多先做 M1；不得同时重写软体 resident 状态、引入 CCD 或接旧 force consumer。不改既有 COL3 public guide/入口，避免原 HOST 快照调用语义回归。

## 7. 预登记多案例与停止条件

M1 编码前冻结 literals/节点清单，使用原门，不根据产品输出改输入：

- **Rigid/source 矩阵**：四物理布局 × sphere/capsule，对真实移动/转动的 rigid mesh，各连续 3 个非零速度入步；mesh 与 round 至少一组双方运动。另有 static mesh，轴退化为 sphere 必须保来源差异。
- **Finite/topology 矩阵**：front/back、face/edge/vertex、共享边/co-nearest、凹槽内壁、真正远 facet 剪枝；闭 mesh 内球正 boundary distance 与空小带不能标 separated。保持完整 facet 行而非单最近行。
- **World/frame 矩阵**：两个稀疏 world、重复 local IDs、shape/vertex/facet 存储排列、共同大平移、非恒等姿态；独立已知几何和真实入步 FK/native 参考，不能用新 gather 作自己的 oracle。
- **完整负控**：source owner/epoch/active/layout/registration/point mapping 错误、expired loan、未提交 candidate、accepted reset/restore 后旧包、全部 filters、exact/one-less 各容量、末组 raw/stamp/packet 构造失败、关闭前后副作用 UNKNOWN_RELEASE。拒步不改上一公开包；可重试普通失败同输入通过，未知释放不盲重试。
- **M2 必须另验**：sphere/capsule 对真正物理推进产生的非刚性 mesh，至少一条边长改变而非整体刚体变换；stable facet/point IDs、共享边/凹体、producer 失败不换 accepted bank、同步/跨 owner 拒绝。只有上传两个手写快照不算 M2 成功。

成功需要两后端、四来源/多案例、同次完整数据链、独立几何门、实际传输计数及源码外公共调用。原 COL2 全部已绿几何矩阵不重跑；只跑新来源与共享接缝受影响范围。任何“设备路径”调用 HOST 当前坐标 pack/closest/BVH boxes，或没有真实 deformable accepted owner，都按功能缺口失败而非换标签放行。一个有证据的通用修复仍未缩小障碍则保反例/停止该片，不调容量/精度/geometry 追绿。

## 8. 固定官方参照与后继边界

本地 Newton 1.7.0 根为 `.venv-newton-1.7.0/Lib/site-packages/newton/_src/`，以下接口本轮只读：

- `sim/state.py:111,120` 明确 `State.particle_q/body_q`；`sim/collide.py:2269` 的 `CollisionPipeline.collide(state, contacts)` 以实际当前 State 为入口，并说明 soft self-contact BVH refit 到 `state.particle_q`。
- `geometry/tri_mesh_collision.py:371,399` 的 `TriMeshCollisionDetector(..., vertex_positions=...)`；`:888–910` 的 `refit(new_pos)/refit_triangles/refit_edges` 从设备 vertex positions 与原 tri/edge indices 发 kernel 并 refit BVH。这是 source→bounds→candidate 的参考，不意味着 Phymeo 已有相同 source owner。
- `geometry/narrow_phase.py:1554,1673` 的 mesh triangle overlap 与后继 contact kernel 分离原 triangle 身份和消费；本片只复用这种分层原则，不复制其 SDF、contact reduction 或接触律。
- 官方固定 [collision 概念](https://newton-physics.github.io/newton/1.7.0/concepts/collisions.html) 是功能参照；没有执行 Newton 或新增性能比较。

完成 M1/M2 后再分别冻结：连续输入/时间区间与保守候选的 CCD；原 material J/velocity/完整 M 与双方力矩的载荷消费者；跨帧 feature/摩擦历史。SDF/hydroelastic、mesh–mesh 及变形自碰撞仍是不同模块。不能让一个当前边界 query 在名称上承担这些未实现的义务。
