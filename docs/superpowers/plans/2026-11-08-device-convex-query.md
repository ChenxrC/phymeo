# K0 通用设备凸体查询：开发与验收计划

**FROZEN_AUTHORING / TESTS_FIRST，2026-11-08。** 持续任务无08:00限制。C11已[正式接受](../../benchmarks/p65-resident-floating-affine.md)并释放所有执行。父以真实源码、[101身份案例清单](../../development/p65-device-convex-query-case-manifest.md)和[API提案](../../development/p65-device-convex-query-api-proposal.md)全文裁定本片；提案保留历史PROPOSED状态，本计划是当前authoring合同。原robot独立文本审阅未发现阻断；预算内收敛仍未执行。

## 1. 一次补通整个形状族

K0交付accepted current pose上的S/C/OBB/H十种无序组合距离、相交状态和双侧有限材料见证。SS/SC/CC共用有限圆族解析；SB/SH/CB/CH用core GJK后加真实skin；BB/BH/HH共用有界GJK/EPA。只有经真实盒验证的H可特化OBB，不能以AABB替代一般H。以普通fixed-root tree/forest与static来源首验，同一support/query入口供后续affine/floating来源扩域。

K0不是首触时间、全patch或新物理律；K1补完整点/线/面接触集合、归一权重与材料chart，K2再薄接原共同受载步。已接受的HOST几何/law/J/M/控制/history不改。设备不支持、未知和资源耗尽明示，不内建HOST fallback，也不把候选tuple上传包装成零往返设备链路。

原20个Q01非零例必须真实device-resolved，不以HOST completion补通过：解析6、core+skin8、polytope6。后者实际收敛是本次待测工作；前者负gap=-.01时core距凸体仍.04，未触发深包含拒绝。没有为某case/坐标/时间硬编码算法。多法向与深包含允许明确未决，不能任意造受力法向。

## 2. 公开入口、数据与有界工程域

采用API提案§2–7的语义/数值上限及wire version=1，不复制第二model/bindings注册：
`batch.prepare_convex_query(broad, *, pairs=None, pair_capacity=None, gjk_iterations=64, epa_iterations=192, epa_vertex_capacity=64, epa_face_capacity=128, memory_limit_bytes=67108864)`；
`batch.convex_query(plan)`、`plan.query()`同路；`plan.require_current(result)`、`plan.last_result`、`plan.stats`、`plan.close()`。

除broad外均keyword-only。plan类为ConvexQueryPlan；容量异常ConvexQueryCapacityExceeded含required/capacity/per_world_required。参数非bool整数，GJK1..64/EPA1..192/vertices4..64/faces4..128/pairs0..32640；内存正整数。构造前按真实dtype/shape计算本plan全部owned bytes，借用不重复计。半构造失败关闭已分配集合；不得关闭batch runtime。

公开记录使用frozen dataclass、tuple/immutable mapping、bytes-backed不可写数组：
- ConvexQueryWitness：key、vertex_ids（仅活跃原int IDs）、weights、offset_m、surface_relative_m、valid。point/offset均相对该row.origin_m的米制，ID不经FP32；skin_offset是世界轴向量，不能把body-local方向误当世界方向。
- ConvexQueryRow：pair（canonical两key）、status/relation/reason/phase（提案枚举对应的固定大写字符串）、gap_m/gap_lower_m/gap_upper_m/scale_m、origin_m（FP64三维）、normal、gap_valid/normal_valid/witness_valid、first/second、gjk_used/epa_used/support_calls/vertices_used/faces_used。无效数值为固定零且valid=False，不以占位值证明分离。first/second与pair严格对应。
- ConvexQueryResult：owner/registration、topology_generation/state_epoch/active、world_keys/ticks、shape_keys/shape_revisions、pair_source、requested_pairs/request_row/request_reversed、rows、raw、source、stats、attempt_tag。canonical rows不因反序请求换掉；反序映射让调用者交换两侧、反normal、保gap，不冒连续材料身份。pairs=None时requested_pairs即同次canonical candidates，映射恒等。
- raw含完整实际meta/gap/normal/surface/material_ids/barycentric/skin_offset数组，不裁剪valid行或padding；source含完整原声明/真实上传字、原local/IDs/skin和量化差、同次accepted pose/stamp和参考origin，均可独立归档。raw枚举/16列顺序/四米制gap列/valid bits严格沿提案§3、§5。
- 同次source pose采用每shape [S,12] FP32：relative frame origin3+row-major R9；source_stamp [S,3] int32：valid/tick/world_slot。动态从当前accepted tree FK采集，静态用已验证同world固定声明，identity R与相对点明确记录。完整注册源保留，不借下次snapshot补旧epoch。重复body pose可冗余存储但复制字节如实计，不冒unique-body压缩。

plan.stats为累计immutable mapping，至少有allocations/live_bytes/peak_owned_bytes/static_uploads/dynamic_uploads/uploaded_bytes/downloads/downloaded_bytes/launches/completed_launches/synchronizations/attempts/queries/failed_queries/pose_gather_launches/query_launches/pair_uploads；成功result.stats含同一次attempt delta及原broadphase delta，logical_pairs/padded_pairs独立，失败成本仍累加在plan。零live不抹累计分配；原broad与NEW复制/launch分别列，不双计或漏计。

FP32 raw门沿案例§8：gap1e-4 L、表面/closure2e-4 L、normal2e-3/unit2e-4、权重非负且和误差64u。L=max(1m,真实diamA,真实diamB,相对origin距离)，不含共同世界平移；参考以独立原输入与实际stored输入分别计算，产品自报L不能放宽测试门。内部稳定缩放可用安全尺度，发布米制及实际L一致；不是严格包围证明。旧ContactSite 1e-10/1e-12不变，K0 raw不直接驱动law。

原Q06大origin若表示已丢失须明确SOURCE_PRECISION/unsupported，不凭local偷偷补回与声明world不同的成功。相对表示仍成立时先在FP64声明端rebase，再由真实FK/gather供query，绝不大world FP32形成后相减。真实skin不是bounds outward padding。raw NaN/非法ID/未知enum/stale stamp是整次失败；非收敛是完整非成功row，不能当clear。

## 3. 父独占的来源与事务接缝

父新增薄适配 `src/phymeo/backends/_resident_convex_source.py`，OLD仅 `tyx_world_batch.py`、`_resident_collision_plan.py`、`src/phymeo/__init__.py`；不改旧support/bridge/solver/kernel数学。产品作者不直接猜旧私有bank布局。NEW入口构造名固定ConvexQueryPlan(batch,broad,*,同public参数)，batch-facing helper固定convex_query_for(batch,plan)。普通树使用type(binding) is ResidentTreeBinding与实际ordinary source域，不能因affine继承关系误接。

`broad._borrow_convex_state()`为contextmanager，持原batch锁及原 `_borrow_collision_state` 的一次device/runtime作用域，yield只读ResidentConvexSourceView：
- accepted：原AcceptedCollisionState，tree为现TreeGeometryView（meta/parents/kinds/anchor/state/fk/rows/specs）。
- model、bindings：原CollisionModel及按model.shapes顺序的原绑定。父在broad注册保留完整不可变来源，不从包围盒反推shape。
- owner、registration、world_keys、shape_keys、shape_revisions、allowed_pairs：真实同world/物理域，再经过samebody/mask/excluded过滤；从原_space.defaults==5取允许表，不能直接用原_registered_pairs冒全部允许。
- `.stage_broadphase()`在这一accepted view内调用旧_read_bounds并构造ResidentCollisionResult/pending；空model同原语义。不调用先发布的broad.query，不嵌套第二次borrow拼接epoch。
- shape_rows/groups/offsets：原ordinary tree/static映射的owned只读HOST数据；动态row/group来自原绑定与accepted.tree.rows。registry中static=-1、ordinary动态行明确；不接受其它来源误用旧布局。

NEW编译和query只在该context内取值。原tree.anchor是实际stored固定参考；静态点先在HOST FP64声明端减同world参考，再上传；动态用actual FK+原group-local点在设备gather。不得HOST FK或snapshot→upload。query成功后，pairs=None才在所有raw/owned记录已构造完成时调用旧broad._publish_query(result,pending)，再以不可失败赋值发布query last_result；显式pairs不改变broad.last_result。任何失败都不改原已发布layout/结果、state/epoch/tick/history/held。

父broad持 `_convex_dependents` 集合，NEW构造完全成功后登记、关闭时移除；broad/batch关闭通过原ExitStack保证一项清理失败仍关闭其它。独立query close不关broad或runtime。batch.prepare与batch.convex_query做精确owner/type门，不能借同键同坐标绕过。

显式pairs只绕过AABB剔除；不能绕过过滤。pairs=None先旧真实HOST候选读回/上传，pair_source='host_broadphase'。原broad candidate_capacity仍可能先拒绝，按原异常保留；测试须给broad足额，仅缩query pair_capacity才能验证新的required=2异常。后继真正device candidate handle再独立验，不在本片偷偷改变旧统计。

## 4. 文件所有权及tests-first

原robot只NEW四路径：
1. tests/collision/test_device_convex_query.py
2. tests/scale/test_resident_convex_query.py
3. tests/reference_newton/rr/device_convex_query_reference.py
4. docs/benchmarks/p65-device-convex-query-tests.md

reference为独立NumPy/stdlib声明/有限公式/材料检查，不import新产品、旧HOST query或新kernel产expected。现阶段不运行Newton；原生driver后置另授。测试用真实公共FixedRobotTree/CollisionModel/bindings；不在module import时引用缺失新API做假RED。先原SS/CH sep/overlap四身份作者checkpoint，独立几何前提在函数内先核，再查新public入口。显式参数ids，完整encoder事先处理精确opaque owner token不可重建标记，不序列化假身份。

原RD在上述checkpoint交接后只NEW四路径：
1. src/phymeo/backends/_device_convex_query.py
2. src/phymeo/backends/_tyx_convex_support.py
3. src/phymeo/backends/_tyx_convex_query_kernels.py
4. docs/benchmarks/p65-device-convex-query-implementation.md

统一bilateral support、固定维simplex和有界EPA；round核心/skin语义与每次支持预算保留。最多32pair tile复用scratch，禁止Python逐world造第二solver。NEW内辅助小函数布局可协调，公开字段/wire/父来源接口变更先报父具体冲突。

两树fresh确认NEW路径均不存在、HEAD/branch/clean后作者；不merge父、不改既有source/tests/report、不新建助手。所有import/collection/numerics/probe/static/native/build/install/profile均禁止，AUTHORING_ONLY不持执行槽。小逻辑commit、最终精确NEW清单/clean/源码读审/UNEXECUTED交接。父等tests-first源码审阅后导入首测试并单槽有效RED，再导入已审产品/OLD薄接缝。原RR额度错误不重启或替换。

## 5. 固定案例与保存

冻结原案例清单全部输入/101纯query、16声明、Q08的16协议身份37query尝试+1半构造失败，每backend逻辑134节点；其中W32 sparse/dense两个后置节点，核心132。作者若因合法pytest参数化需要不同函数划分，身份/调用量不能变；发现原声明自矛盾报父，不能暗调参数/减少组合。

另加8个公开API preflight节点（0 query）：bool GJK、GJK0、EPA193、vertex3、face129、pair_capacity负、memory0、重复正反显式pair。给全部原合法其它参数，不用错误owner等先失败掩盖；其余非法owner/source与filter沿Q08/域声明，不无界加矩阵。总142节点/backend，核心140、W32后置2。构造清理仍独立原1身份。

每逻辑身份一完整gzip，包括原声明、stored源、raw全数组/dtype/bytes、实际status/计费、全部owned记录、独立expected/actual/门、每attempt及失败原字、retry/恢复/清理。原101即101文件；16声明保存输入/实际分类；17协议（含构造失败）各一文件，8额外参数各一文件：142固定文件/backend，不按运行输出挑组。异常未产生device raw时明确None和未执行阶段，不伪造全零结果。故障在真实返回后call-through原validator，不能直接throw替代raw检查。

原zero_axis不新增构造拒绝，解析退化有限段按参数1/2给device解；本片无自动HOST completion，因此不得以普通UNKNOWN填该正控；source非法在broad/shape原门先拒绝也保原输入、不能当device pass。深包含/并列法向保距离与normal状态的区别。Q03原ID整数与非轴姿态保持，tie不比任意顶点。Q06带外符号与20首族必须通过，带内标CONTACT_BAND/未知，不能错报clear。

## 6. 父唯一执行队列与验收层次

- [x] C11正式接受和释放；案例/API提案全文审阅、父来源接缝裁定。
- [ ] 原robot先4身份tests-first checkpoint，原RD随后并行产品作者；父真实RED只一次。
- [ ] final交叉源码review/NEW精确导入；首valid Ruff/format原错误留证，必要最小修复后静态。
- [ ] 原4身份最小interpret→CUDA，随后非零余16→有限/协变→近零/域→协议/额外参数；已绿同源节点不重跑。
- [ ] 核心通过后W32 sparse/dense，每端两节点；实际memory/launch/copy报告，不猜收益。
- [ ] 同一几何的pinned Newton独立原生短参照、源码外新wheel、只读同次保存审计后置另授具体NEW文件/固定输出，不能仅hash替代实际调用。
- [ ] 更新正式范围与全景；K1完整patch，K2真实消费者随后各自冻结，未知缺口不抹去。

每次执行fresh entrypoint-aware guard，父唯一owner，真实foreign任务自然结束。保原所有首失败、无remote CI/no pin/no旧绿色组重跑。受影响adjacent旧入口只选少数能证旧默认未改的节点，冻结具体清单后一次执行，不整套C6–C11重来。runtime异常和geometry UNKNOWN分别记录。

同一非零族连续两次无缩小复现/新能力，或60min只扩写数学而无可执行入口，应停止该分支扩域，记录最小失败并处理接口/算法；不能无限加迭代/改gap/改质量/放宽门。精度按既定工程合同，不追求精确接触恒等。没有速度提升也可验收功能；性能后置必须同硬件、同完整输出、同输入/容差，不拿query对完整patch或kernel对端到端算倍率。

## 7. source记录精确键名（作者对齐补充）

result.source 为不可变 mapping，schema='resident-convex-source-v1'，以下必填且包含完整注册源；不因当次无候选删去源表：
- model：原 CollisionModel；bindings：按 model.shapes 顺序的完整原不可变绑定（含已有拓扑、from_asset来源证明）。不另造几何声明替换原输入。
- world_keys：model涉及的有序world tuple；world_origins_m：FP64[W,3]，由各world实际 spec.anchor 的FP32字扩宽；shape_world_rows：int32[S]，索引此world_keys。
- shape_keys、shape_revisions、shape_kinds：有序tuple；shape_rows、shape_groups：int32[S]，原tree row/group或静态-1；vertex_offsets：int32[S+1]；vertex_ids：int32[V]原ID。
- declared_local_points_m：FP64[V,3]，动态为原binding group-local点，静态为原motion.start世界点；upload_frame：长度S的tuple，分别'body_local'或'world_anchor_relative'。
- uploaded_local_points_m：FP32[V,3]，动态为原local最近FP32，静态先FP64减本world真实参考再舍入；point_quantization_error_m：FP64[V,3]，上传字扩宽减对应同frame的FP64值。
- declared_radius_m：FP64[S]；uploaded_radius_m：FP32[S]真实skin；radius_quantization_error_m：FP64[S]同样actual-minus-declared，不用bounds padding。
- pose：实际同次device gather FP32[S,12]，前三为参考origin相对位置、后九R row-major；静态为0和I，其已rebase点由uploaded_local_points_m给出。stamp：实际int32[S,3] valid/tick/global_world_slot，最后一列是accepted.slots的真实slot，不混shape_world_rows。
- allowed_pairs：同world、全部原filter之后的canonical key pairs。资源/上传表若另存buffer ledger可追加键，不能重解释以上字段。

result.raw 的固定七个主键为meta/gap/normal/surface/material_ids/barycentric/skin_offset；pose/stamp完整实际字只存source中，避免两份可分歧来源。测试用这些明确键逐字段保存，未知附加ledger也保留，不通过删字段规避比较。所有浮点数组完整保dtype/shape/bytes；公开表面/offset和raw仍沿§2，不从source重新搜索生成raw。
