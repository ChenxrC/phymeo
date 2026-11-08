# K0 设备凸体查询：待冻结 API 与数据合同

**API_PROPOSED_UNEXECUTED / NOT_FROZEN / AUTHORING_ONLY。** 仅本 NEW 文档；未 import、数值探测、测试、静态检查、原生执行或测量。child 基线 `d10c122bf85b87ccfbe400bc355f4e59f764676b`；父源码读取点 `8f43eb4fa31cc96b7940ed19dbced5d1f4a8bd33`。不提前宣布 C11 接受，不改默认入口、任何物理律或旧构造门。本文供父直接裁定接口，授权实施仍需另发。

## 1. 最小交付与现有代码的分工

K0 是 **accepted current pose → 有限形状距离/相交状态 → 双侧材料见证**，无物理推进、首触时间、完整 patch 或 force。沿 [原候选清单](p65-device-convex-query-case-manifest.md) 的十族、101 个纯 query 身份、16 个域声明以及 Q08 协议；不重列或更改这些输入。K1 才生成全部 point/segment/polygon sites 和原归一权重；K2 才进入原共同接触步。

| 当前源码（父读取点） | 可复用与不可混称 |
|---|---|
| `geometry/_convex_support.py:43–132,228–333` | 已有 bilateral support vertex、simplex barycentric、GJK/EPA、双侧材料地址/offset、gap bracket 和预算失败；HOST 使用 FP64/`lstsq`/动态 polytope，不能直接改装饰器冒 TYX 核。其 `_MAX_GJK=64/_MAX_EPA=192/_TOL_M=2e-7` 不是本片 FP32 误差合同。 |
| `_convex_support.py:336–399`；`_round_sweep.py:112–190` | 原 sphere–hull 有限投影、round 轴参数和退化轴解析语义值得保留。不能为了统一算法把低维点/线强送 EPA，或把 sphere skin 的若干支撑点平均后当精确球面。 |
| `_convex_hull.py:35–114`；`_convex_contact.py:166–206,214–272` | 原完整 hull 预检、真实 face/edge/chart、support 平面及有限最近点为 HOST 差异参照；`hull_intervals` 是设备投影提议，最终裁决仍 HOST。不是已有整族设备窄相。 |
| `contact_patch_adapter.py:33–118,147–305` | 旧 `ContactSite` 权重/单位法向/共同点门、完整 patch 及深包含/多法向拒绝不动；K0 输出不伪装此类型，不把一个 witness 当一站载荷。 |
| `backends/_resident_collision_plan.py:37–60,180–238,477–563,655–844` | 已有 accepted owner/topology/epoch/bank/tick、注册/过滤、设备 bounds、stage→publish；可作为 K0 唯一 runtime/锁/来源 owner。`registered_pairs` 是物理域，真正允许 pair 还须 mask/excluded/same-body 过滤。 |
| `tyx_collision.py:159–195,200–229`；`_tyx_collision_bvh.py:22–57` | 当前 bounds 和 BVH bitset 读回后在 HOST 形成 candidates；没有公开可借用的 device compact pair 句柄。不能把这条 tuple 再上传称零往返。 |
| `_tyx_resident_tree_bounds.py:16–135` | 原 FK/local points 可供设备几何 gather，但 outward radius+padding 只包围。窄相必须用原 shape radius，绝不把 bounds skin/运动包围余量当物理厚度。 |
| `_resident_contact_bridge.py:332–416,439–499` | 当前 HOST 完整 patch→原 law→COM wrench、入口 anchors→真实末态位移/功已有单一实现。K0 不另写 law、J/M 或求解器。 |

首验收来源域建议限原 Q08 所用 `ResidentTreeBinding` 普通 fixed-root tree/forest 与 `ResidentStaticBinding`；十种形状组合均须成立。其余已有 slider/rotor/affine/floating 来源不得仅因宽相可用就声称 K0 已验：以后只增加同一 pose 借用描述的 adapter 与独立资格，不复制 query/solver。Q01/Q02/Q03/Q06 的原 stationary material points 可绑定零速、零推进的普通组与 static，形状/ID/gap/姿态不变；这是输入来源适配，不新增物理步或改 101 身份。

## 2. 建议冻结的公开接口

```python
broad = batch.prepare_collision(model, bindings,
    candidate_capacity=..., tree_dynamic_pairs=...)  # 原 public API
plan = batch.prepare_convex_query(
    broad, pairs=None, pair_capacity=None,
    gjk_iterations=64, epa_iterations=192,
    epa_vertex_capacity=64, epa_face_capacity=128,
    memory_limit_bytes=64 * 1024 * 1024,
)
result = batch.convex_query(plan)  # owner gate 在任何 device 动作前
# plan.query() 为同一路径的便利入口
plan.require_current(result)
plan.close()
```

- `broad` 之后均为 keyword-only；预算均为非 bool 整数，GJK `1..64`、EPA `1..192`、EPA vertices `4..64`、faces `4..128`。内存预算为正整数 bytes，分配前按真实 dtype/shape 核全部 NEW owned buffers，超限返回所需字节并拒绝构造；不计借用资源为新分配。
- `pairs=None`：查询这一刻真实 broad candidates，不凭 narrow phase 增删宽相域。保存 `pair_source='host_broadphase'` 和实际 candidate read/upload；内部 stage 不能先调用 `broad.query()` 把半套结果发布。后续借 bitset 的实现仍用此公开接口，另标 `device_broadphase`。
- 显式 `pairs=tuple[(CollisionKey, CollisionKey), ...]`：冻结要查询的有向请求，**只绕过 AABB 距离剔除**，不绕过 owner/world/物理域、same-body/mask/excluded/直接父子限制。Q01 正 gap 与 Q02 近失依此取得距离，不能因不是 broad candidate 被静默省略。输入同一个无序 pair 重复（含正反两次）拒绝；顺序/身份只影响请求映射，不影响物理。
- 输出 `rows` 按 canonical key 排序；显式请求另保 `requested_pairs`、精确 `request_row` 与 `request_reversed`。反向读取交换两侧地址/点/offset并反 normal，gap 不变。不得按浮点位置或 support slot 推连续身份。
- `pair_capacity=None` 取通过过滤的同 world 物理域 pair 数；显式容量是非 bool 整数 `0..32640`。实际必需数超过容量整次抛 `ConvexQueryCapacityExceeded(required, capacity, per_world_required)`，不截断成功、不改 `last_result`。迭代/几何 scratch 限额是每 pair 状态，见 §5。
- K0 初版不内建自动 HOST fallback。`UNKNOWN/UNSUPPORTED/RESOURCE_LIMIT` 是完整可读 query 结果，调用方可显式用结果中同次owned entry几何另调已有 HOST 几何（不能重新读新epoch）；合并归档时记录 `host_completed` 与其独立结果，不能改写 `raw_device` 或计为 device-resolved。这样不为首片再造一条 HOST dispatcher。
- `require_current` 验同一 batch/plan 注册、模型与材料 revision、epoch/bank/tick；restore 即使同 tick 也使旧结果 stale。重复 query 不消耗控制历史、episode、tick 或 state epoch。结果为 owned immutable 数据，查询结果仍可离线读，但 stale 时不可供后续消费。
- 空显式请求/空模型合法，返回零 rows；无相关 world 时不作设备操作。`pairs=None` 的零候选仍需真实执行发现候选的 bounds/BVH，不能记成零 launch。query 无单独 HOST 物理 runtime。
- `broad.close()`、`batch.close()` 都必须即时关闭依赖 plan；独立 `plan.close()` 不关闭借用的 broad。多资源清理一项抛错仍关闭其余，live bytes 归零、累计 allocation 保留。半构造失败按实际已分配集合清理。

**相对原草案的必要收窄**：不用再接收第二份 `model/bindings/site_capacity`；借现成 broad 绑定以免双重来源，K0 一 pair 一 witness 不设 patch site cap。显式 pairs 补足原 20 个非零族里正分离查询；static/static、跨物理 participant 等不因这个选项扩域。没有新的源码实施授权。

## 3. 同源输入与有界布局

所有整数标识都以 int32/owned Python integer 保留，绝不经 FP32。用户 key 与设备 row 分表，原非连续大 vertex IDs 不用于数组下标。编译先对**全部注册几何**做原合法性验证，包括远处/被过滤坏 hull；只缓存成功且绑定完整 local/material source 的拓扑。rounded H 标 unsupported，不清 radius；合法同坐标不同 ID 的零轴 C 保留。

| 层 | 建议字段/实际 dtype | 所有权与使用 |
|---|---|---|
| static registry | `keys[S,3] i32`，`shape_meta[S,8] i32`（kind、world row、vertex begin/count、pose row、真实 body group、revision、flags），`vertex_ids[V] i32`，`local_points[V,3] f32`，`radius[S] f32` | 原 FP64 声明/完整绑定 token 同时 owned 保留；S≤256、V≤8192、每 H≤64。radius 是真实 skin 的最近 FP32 表示，量化差另存，不用 outward bounds 参数。 |
| scoped pose source | current active bank 的实际 FK origin/R 与 tick 数组；HOST 持每 world FP64 公共 origin 和 source proof | 父只给受锁借用视图。新 gather 不做 HOST FK、不推进、不从 `.snapshot()` 再上传；实际 stored R 不 QR、不吸附，forward 使用原矩阵。 |
| pair table | `pairs[P,3] i32=(world_row,shape_a,shape_b)`，canonical slot；显式请求映射独立 | 显式表一次上传；broad tuple 过渡路径每次真实上传。不同 world 绝不生成全局交叉 pair。 |
| numeric result | `meta[P,16] i32`，`gap[P,4] f32`（estimate/lower/upper/L），`normal[P,3] f32`，`surface[P,2,3] f32` | meta 的固定列与单位见下；所有点/长度在 pair-local 米制结果中，origin/L 单列。reserved 列固定值，读回也验。 |
| material result | `material_ids[P,2,4] i32`、`barycentric[P,2,4] f32`、`skin_offset[P,2,3] f32` | 每侧活跃项计数写入 meta；kernel 从已绑定 slot 表写原 ID，HOST 逐项核属于该侧 shape。内部 slot 不冒身份。simplex 至多4项、EPA foot至多3项；重复 owner 项合并保权重，未使用槽有固定哨兵。它不是 K1 的全部 feature/patch。 |
| scratch | simplex≤4；EPA 每 pair V/F 有显式上限；建议最多32 pair一tile复用 scratch | 空间 O(P outputs + tile×EPA capacity)，不是 P×全局最大 patch。tile 只按 canonical pair 行分页，不 Python 逐 world 造独立求解；launch 次数如实计。 |

建议 wire version=1：`meta[:,0:16]` 依次是 status/relation/reason/phase/gjk_used/epa_used/support_calls/vertices_used/faces_used/valid_bits/countA/countB/attempt_tag/pair_row/reserved0/reserved1，末两列必须0；phase为0 NONE、1 ANALYTIC_ROUND、2 GJK_CORE、3 GJK_POLYTOPE、4 EPA；valid bits为1 gap、2 normal、4 witness，不接受其它bit。`gap` 列值是已还原米单位的 estimate/lower/upper/L；surface/offset不含公共世界origin，normal无单位。内部缩放坐标不能与米制列混存。`meta` 的 valid bits 必须明确哪些有限数字有意义；无效 gap/normal/点采用固定零占位并标无效，不能把 NaN 当协议哨兵。raw 非有限字本身必须保留在失败收据，不能修成零后过门。任意布局修订须由父在tests-first之前另冻结版本，不能依 kernel 分支另解释同列。

使用“共同平移 + 等比尺度”的 query 坐标，不把近正交 R 的转置当精确逆。对 hull/axis，用实际 forward 变换后的 core 做 support；sphere/capsule skin 始终是 world 欧氏半径，不随 R 的舍入变成椭球。支持方向的内部任意初始化不等于有唯一物理法向。

大公共 origin 必须先在来源仍可表示时做 FP64 静态 rebasing，再读真正相对 FK；不能先形成大 world FP32 点再相减。查询记录同时保存 declared FP64、实际上传字、pair origin/L和输入量化差。原 Q06 `loss` 的 world 表面若已不可表示，标 `SOURCE_PRECISION`/无 world witness；不能凭保留 local 点悄造一个与已丢 world source 不同的成功记录。`rebase` 若当前借用来源仍不能保同一几何，明确 unsupported，原身份不改。

## 4. 算法复用与十族统一 dispatch

建议 **一个 bilateral support 核合同 + 一个 bounded query dispatcher**，而不是十套 scene solver。support 返回 `(minkowski_point, first_core, second_core, first_slot, second_slot)`；每次规范化/feature 选择都有实际 finite 检查，计 support_calls。相同投影 ties 可按原 material ID 确定内部顺序，但公开验收只要求真实最近集合，不能借 tie 顺序生成持久摩擦 identity。

| 分支 | 十族覆盖 | 建议首实现及保留边界 |
|---|---|---|
| 有限 round 解析 | SS/SC/CC | 共用点/有限段 metric projection，保区间 clamp、平行区间与零轴语义；非重合 core 时 `g=d-rA-rB`，沿真实 core normal 加 ±r n。重合 core 可知 overlap，但 normal 仍 UNKNOWN。不新增零轴构造拒绝。 |
| core distance + skin | SB/SH/CB/CH | H/OBB 用完整原顶点 support，圆方只给点/有限轴；GJK 求 core 距离后才减圆半径，避免曲面支撑点 barycenter 落在球内。外部最近面/边/端材料 slot 保留；core 已入 hull时返回 containment/normal 未决或显式 HOST 完成，不借 K0 放开旧 deep-patch guard。 |
| polytope distance/penetration | BB/BH/HH | GJK separated simplex + EPA overlap，沿现 `_convex_support` 的双侧 owner/foot/bracket 设计做有界数组版本。OBB 可仅作经过真实盒验证的 support 加速；默认完整8点扫描已经正确，H 永不变 AABB。 |

现 HOST `lstsq`、Fraction 和动态 polytope 不在 TYX 内调用；simplex 1/2/3/4个活跃点用固定维数公式/秩检查。EPA 无足够满维 seed、horizon不闭合、重复support停滞或超 cap时显式返回理由；不扩到圆柱/锥/椭球等旧 `_KINDS` 的其它族。候选 default GJK64/EPA192沿现预算量级，但每 plan 可声明较小预算（Q08 budget1）；每个循环/seed/horizon遍历皆 bounded，原形状 caps仍独立。

每个最终 surface witness必须满足有限集合与原 barycentric/skin重建，不能仅有 Minkowski distance。对 HH EPA已有 overlap bracket但精确见证未收敛，允许 `relation=OVERLAP`、`status=UNKNOWN/RESOURCE_LIMIT`、`witness_valid=False`；禁止把 bracket中点补成成功见证。

**Newton 1.7 对照不是“可直接复用的 EPA”**：本机 `.venv-newton-1.7.0/Lib/site-packages/newton-1.7.0.dist-info/METADATA:3` 为1.7.0。`newton/_src/geometry/collision_convex.py:142–224` 是共享 support 的 MPR→GJK，`:166–212` 有 inflate及回退修点；本片不复制这些参数改变真实 gap。`simplex_solver.py:327–376` 的 cutoff可返回非最近估计，K0距离任务必须关闭该 cutoff，不能以 separated boolean作为最近见证通过。`support_function.py:195–217,221–307` 是实际有限mesh/box/capsule（局部+Z），其 box deadband/任意零方向也不等于本项目材料特征或唯一normal。`mpr.py:298–384` 的预算出口不提供本片完整资源状态，不能把其 false直接译成 separated。

优先复用本项目已有 bilateral GJK/EPA结构和 round解析，Newton只作独立原生几何参照；不引入第二Warp runtime到TYX产品。`multicontact.py:651–667,700–730` 的最多4个裁剪点及额外点、`narrow_phase.py:1245–1274` 的query/manifold分队列仅供K1结构参考；既不提供原稳定material IDs，也不保证本项目面积/线归一权重。

## 5. 状态、工程门和失败语义

建议 wire 将几何关系与执行状态分开，避免已知 overlap/无唯一 normal被抹成 separated：

| 字段 | 建议精确编码 |
|---|---|
| `status` | 0 UNWRITTEN（读到即整次错误）；1 RESOLVED；2 UNKNOWN；3 UNSUPPORTED；4 RESOURCE_LIMIT。 |
| `relation` | 0 UNDETERMINED；1 SEPARATED；2 OVERLAP；3 CONTACT_BAND。后者是工程零附近，不声称严格相触。 |
| `reason` | 0 OK；1 NEAR_ZERO；2 NORMAL_NONUNIQUE；3 CORE_CONTAINMENT；4 GJK_LIMIT；5 EPA_LIMIT；6 DEGENERATE_SIMPLEX；7 EPA_CAPACITY；8 UNSUPPORTED_SHAPE；9 SOURCE_PRECISION；10 STALLED。读取其它数值拒绝。 |
| `phase` / validity | ANALYTIC_ROUND / GJK_CORE / GJK_POLYTOPE / EPA；gap/normal/witness的独立bit，及 `raw_device` 保真。UNRESOLVED不能靠非零占位normal变RESOLVED。 |

旧非法构造/错 world/owner/丢点/重复ID等 preflight继续抛明确异常；设备异常、未知enum、未写stamp、NaN/Inf、非法material ID/slot/weights、错epoch/bank/tick和晚HOST分配失败为**整次 query失败**，不发布新 result/layout，也不触动原state/history/held/diagnostics/last_result。预算耗尽则是已完成查询的显式非成功行，可发布完整报告但不能被碰撞消费者当clear。原 device recovery按既有 runtime规则处理，同输入重试不得缓存失败。

采用原清单 §8 的候选 raw门，不增新物理门：`L=max(1 m,diamA,diamB,|originB-originA|)`；gap `1e-4 L`，表面/closure `2e-4 L`，唯一normal误差 `2e-3`、unit误差 `2e-4`，权重非负且和误差≤`64*2**-23`。FP32算法 bracket/这些余量都是**工程估计而非严格区间证明**；带内不能报无余量 strict separated。记录原FP64真值和实际stored输入真值两套差异，不能只验量化后错误模型。输入/资源/身份/stamp是精确门。

结果 public类型建议 `ConvexQueryResult/ConvexQueryRow/ConvexQueryWitness`，不是旧 `CollisionWitness` 或 `ContactSite` 的隐式替代。包括实际 owner/registration、source/model/shape revision、epoch/bank/world ticks、请求映射、全部 raw数组、状态/理由/预算、双侧key+ID+weights+offset+surface、构造路径。完整公共结果另保 selected唯一body的真实entry origin/R原字与FP64共同origin、static声明、immutable原local/ID/shape表；这份pose读回按实际bytes单列，不用后来的state补旧输入。HOST仅把实际raw位和原ID解码成 owned记录，不悄做新几何搜索、重归一后冒raw。若以后需要旧严格记录，必须单列 `host_representation`：用同源 FP64及已选feature按旧门构造；发生重新feature/最近点搜索就是 `host_completed`。旧1e-10/1e-12门仍保持。

同一 query锁内：检查owner与计划→借accepted bank→stage候选及几何→全部device工作/同步→验证完整raw→构造所有owned记录→最后一次publish查询layout/last_result。没有state swap或控制推进；`require_current`仅验证，不耗序号。调用期间锁被释放后取得的新epoch不能拼进旧结果。每次尝试的私有attempt_tag递增（失败亦递增，非物理history），溢出拒绝；输出先清成UNWRITTEN再写当前tag，不仅靠tick检测旧scratch。

## 6. 对原案例身份的接口映射

| 原清单 | 此API实际验证 | 不替代的后继 |
|---|---|---|
| Q01 30，Q02 9，Q03 50，Q06-band12，共101 | 显式pairs路径，完整正/负gap、有限surface/材料、反向请求映射、实际stored非轴姿态/ID重排；Q01非零20必须真实device-resolved。 | touching允许明确未决；并列集合只核合法性，不逼任意顶点逐位相同。 |
| Q06-domain16 | 编译全源预检与query状态分开保存；远invalid、深包含、零轴、v64/v65、large origin原输入保持。 | 不借新query扩大旧patch受载域或削点救容量。 |
| Q08 W1/W4/W32、filter、epoch/容量/预算/故障/close | `pairs=None`真实当前宽相；原37 query尝试+1真实advance/1restore及单独半构造失败；跨owner用 `other_batch.convex_query(plan)`真实公开边界。 | 无40步接触窗口，无C9/C10旧矩阵复跑；W32后置。 |

independent reference沿原手推支持面、有限段/真实H斜面与顶点；新核或产品 M/J 不能产期望。旧HOST完整query可差分，但非唯一oracle。原生可用 pinned Newton相同 S/C/真实convex输入、GJK cutoff关，保默认MPR/inflate实际路径与normal语义差异；不同law/manifold不比较受力或性能比。本轮没有额外 query身份；若新增API-only非法参数组合，应单独父冻结计数，不塞入原101或把旧失效当通过。

最小tests-first切片仍是原 SS/CH正负四身份（原清单身份的分段执行，不新增/重跑），先旧有限几何前提后新公开API缺口。再非零20→有限/变换→边界/事务→W32。首片不要求K1全patch先完成；同一正域反复UNKNOWN或需要HOST completion才能过时，保原输入/raw与理由，整族K0仍未完成，不调几何或扩大预算无限追证。

## 7. 数据往返、计数与可比任务

原路径调用事实：stamp `int32[S,2]` D2H为8S bytes，bounds `f32[S,6]` 为24S；BVH每页params `i32[Q,3]` 上传12Q、原output整个 `i32[Q,ceil(S/30)+3]` 读回，尾页/布局上传按真实nbytes记录。这些是源码布局，不是本次实测。

K0新增账分别报告 static vertex/ID/radius/拓扑上传，显式pair一次或HOST候选每次上传，pose gather、query/分页launch、完整meta/value/material读回和HOST记录构造；记录每个实际buffer的dtype/shape/nbytes、logical rows/padding、同步、completed launch、累计alloc与live bytes。source为device FK仍不等于整个候选链驻留。计时须分cold和warm raw query/同完整public result；IO/审计排除，显式sync与copy等待墙钟不可重复加。

第一版直接候选借用不是必须阻塞K0的工作：先交实际device距离及来源可见的 `host_broadphase`。后继若父拆 `TreeBuffers.hits` 的launch/readout并给同epoch scoped bitset/compact handle，新增 `device_broadphase`执行路径须证明无需HOST decode→upload，完整同任务结果和计数一致后才声称省往返。不得先把旧tuple包装成device handle。

不设猜测速度目标，不自动切默认，不拿本query与旧完整patch/force的不同输出算倍率。真正性能比较需另授同完整K0输出的未插桩配对；没有端到端收益可接受有明确边界的设备功能，不宣称Newton全求解提速。

## 8. 建议实施文件与父接缝

| owner | 建议allowlist / 必要职责 |
|---|---|
| NEW产品作者 | `src/phymeo/backends/_device_convex_query.py`：schema/静态编译/plan资源与stage-publication；`_tyx_convex_support.py`：统一bilateral support与有界simplex；`_tyx_convex_query_kernels.py`：解析round、GJK/EPA及同源gather/tile。不按每pair族或每robot类型新建solver。 |
| 独立NEW测试作者 | `tests/collision/test_device_convex_query.py`：原101+16域；`tests/scale/test_resident_convex_query.py`：原Q08；`tests/reference_newton/rr/device_convex_query_reference.py`：独立公式/原生字段对照（不能import新产品算期望）；一份NEW验收report。具体职责/是否拆native由父tests-first时冻结。 |
| parent OLD最小首接缝 | `tyx_world_batch.py` 两薄public方法/owner gate/stats；`_resident_collision_plan.py` 注册依赖close、accepted source借用与stage结果可共同publish；顶层 `phymeo/__init__.py` 导出。旧 `.query()`/default行为保持，不改kernel数学。 |
| parent 后继候选直达 | 仅真正实施时拆 `tyx_collision.py` / `_tyx_collision_bvh.py` 的launch/readback，原完整读回调用保持兼容；不是首片借口删除旧公开records。 |
| 保持原owner且本片不改 | `contact_patch_adapter.py`、`_convex_support.py`、`_convex_contact.py`、`_resident_contact_bridge.py` 与全部动力学/控制/历史。它们用于差异参照和后继薄消费，不被NEW并行替换。 |

父 source借用最小内部合同应包含 `view`、只读shape→真实world/group/source row映射、原local/material/skin、same-world允许pair表、原pending layouts；在同锁context内有效，不能让NEW依赖多个私有bank布局自行猜。首支持ordinary tree由现 accepted geometry 中的 stored FK与原binding证明提供，其它来源明确拒绝至另验。若不能小幅从现`_stage_query`分出这个借用边界，先报告具体冲突，不复制整份ResidentCollisionPlan或另建runtime。

K1以后从同一support/feature协议取完整有限接触集合，恢复原线/面归一权重、一阶矩及双侧material chart；未知normal/deep域保旧拒绝。K2只能在K1必要族合格后，将完整patch送原 `sample_patch_loads` 和原 staged `Jᵀ→full/reduced solve→finish→单发布`，保原真实速度、COM矩、anchors/功及资源事务。K0没有资格用近似单点提前替换这些消费层。

**交接状态：API_PROPOSED_UNEXECUTED / AUTHORING_RELEASED。** 当前只提议冻结上述接口、布局、工程门和phase边界；没有新增产品、测试结果、性能数或实施许可。