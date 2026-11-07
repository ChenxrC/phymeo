# 设备凸体窄相后继：整族查询、材料斑块与原受载律

**NOT_FROZEN / PROPOSAL_UNEXECUTED**。只读父 `f9ab296e2a3dd7cfdd9aa6b45922c4c5afceb92f` 与本机 Newton 1.7.0 源码；无项目执行、测试或性能数据。本提案不改变 C9→C10 优先级，不授实施，不提前放开浮根接触。

## 1. 当前能力与真正接缝

| 实际源码 | 后继不能省掉的工作 |
|---|---|
| `backends/_resident_contact_bridge.py:384–416` 调 HOST `contact_patch_sample`，再 `sample_patch_loads`，逐侧写共同点 COM wrench；`:439–499` 运输入口材料点/作用点，保存两类实际功。 | 设备几何输出必须接回同一 material/site/law 合同，不复制受载求解器或改法则。 |
| `geometry/contact_patch_adapter.py:283–305` 支持 sphere、有限 capsule、未圆角 full-dimensional hull；OBB 是真 hull 特化。`:149–280` 分 point/segment/polygon 构造，圆轴深包含/并列多法向明确拒绝。 | 支撑函数全族共用；特征集合与 patch 维数仍须正确，不能以无限轴/AABB/单个 GJK 点代替完整有限形状。 |
| 同文件 `:58–137` 要求双侧材料点、gap/normal、共同 midpoint、正权重且总和1；`:140–157` 用面积扇形分配保持非对称多边形的一阶矩。 | manifold 减点必须保持该物理离散化，不能换为等权“最多4点”而仍声称同输出。 |
| `backends/tyx_collision.py:188–258,348–400` 是设备 bounds/BVH 分类与有限时间/SAT interval 提议；`geometry/_convex_contact.py:181–206` 明确最终区间/材料判断仍 HOST。 | 已有设备候选/投影不是整族 device narrow phase，更不是设备 patch 或受载响应。 |
| `_resident_collision_plan.py:630–742` 下载 stamp/bounds，并从 flags 构造 HOST candidates；`_tyx_collision_bvh.py:22–57` 下载 bitset，`_collision_spatial.py:79–109` 在 HOST 解码。 | 尚缺带 owner/topology/epoch/bank/tick 的设备候选借用句柄；不能把 HOST tuple 重传隐藏成零往返。 |

`constraints/collision_shapes.py:12–14` 的原边界为256 shapes、8192 material vertices、每 hull64 vertices；采用当前限制，不靠无限 scratch 扩域。无效 hull 必须在编译/更新时拒绝，即使后续 AABB 分离，亦不能失去原 invalid-geometry 语义。

## 2. 一个整族协议，分层推进

建议一个 opt-in `ResidentConvexQueryPlan`，由现有 batch/runtime/锁拥有；注册原 shape/binding，不再创建按机器人/形状组合分裂的 solver。暂拟 `batch.prepare_convex_query(model, bindings, *, pair_capacity, site_capacity, fallback='host')`，返回 plan；`plan.query()` 只读同一 accepted state，接触步使用内部 staging。公开名称和容量由父另授时冻结，旧入口/default 完全保持。

- **静态编译**：一次验证/缓存完整 convex 拓扑、真面/边、stable material vertex IDs、局部顶点和 skin radius。OBB 只有经过原真实盒验证才特化；H不包成盒。设备存有界索引和数值，原可能非连续大ID由 owned 表还原，不能直接窄化ID。shape/local geometry/revision 变化必须重建，失败不缓存。
- **入步绑定**：从 accepted device pose/FK 及现有候选位集取同源视图；static pose也同注册。固定表仅含原允许的同world pair域，不新增跨world全配对分配。保原同body/mask/excluded/父子过滤及 capacity。不得用预测 pose 或另做 step 生成候选。
- **距离/相交层**：统一 support map 提供 S中心+radius、C有限轴+radius、OBB/H完整极点，同时返回两侧 material indices。GJK距离、带有限预算的 penetration 求解及解析 round 特化只输出 gap/normal/双侧见证/feature证据；区分 `separated / witness / unresolved / unsupported / overflow / invalid`。边界不确定或预算耗尽不能标 separated。
- **斑块层**：以已确认法向提取真实支持 feature，统一 point/segment/polygon 相交与回投；S退化点、C轴端/轴段、OBB/H面环共用 clipping/地址路径。保所有必要极端点、两侧 surface points、原 material chart 权重与 round skin offset，再依原 `_sites` 作归一 quadrature。单点距离见证不自动证明 point patch；parallel capsule 线段与 hull 面区必须升级。
- **原受载层**：先继续 HOST `sample_patch_loads`、共同点速度与 COM torque 聚合，再接 C6–C10 已接受的原 Jᵀ/full solve/单发布。未来迁移 law 是另一节点，须保 stiffness/damping/friction 的 whole-patch 语义；本片不切换 law、不提供新的持续摩擦或 CCD。

输出 packet 必须有完整 pair/key/source token、entry tick、几何状态、iteration/budget、dimension、site offsets/count、两侧 material IDs/weights/offset、surface/common point、normal/gap、quadrature，以及 `device_resolved/host_completed` 路径标志。失败和 HOST completion 数逐world可见；不能用一个“GPU=true”覆盖混合结果。

FP32原始结果不能直接 cast 后通过旧 `ContactSite` 的1e−10几何格式门和1e−12权重和门。建议保原 public 门：设备提供已选 feature/参数，HOST仅用同源 FP64声明与实际stored pose作地址还原、单位法向表示和记录构造；不搜索新最近面或悄换 feature。raw几何工程误差与 public构造一致性分别验收，任何需要重新几何搜索的行明记 HOST completion。新raw误差门在tests-first冻结，不据结果调；近零未决走显式fallback/拒绝。采用相对原点与统一尺度控制消减，但不QR/吸附实际stored矩阵，不假定近刚体矩阵的转置就是精确逆。

## 3. 首片与后继的可交付切分

**K0：整族设备距离与双侧有限见证。** 十种无序 S/C/OBB/H组合全部走同一 packet/support 路径；独立数值oracle验证距离/相交、有限轴端及材料重建。旧 HOST 完整patch/law仍消费所有需要 patch 的行，完整受载输出保持原合同。K0是设备距离窄相，不宣称完整device manifold、节省全部HOST几何或零读回；若只返回 boolean 不予验收。

**K1：共用 device patch。** 依同feature协议加入 round point/line 与 convex polygon clipping/材料地址/原归一权重；从静态近表面唯一法向域开始，不悄加深包含。point/line/area 三种结果都通过实际力矩/功与完整输出对照，才允许该行跳过 HOST patch 搜索。编译/记录/法则仍HOST要分别标明。若K0工程稳定性不足，先交其真实有限域和反例，不边修边扩大支持。

**K2：接触消费者与可选读回。** 接入原 staged contact transaction；有完整记录与少量选择读回两种任务时各自计时，不能比较不同输出。只有后续法则/装配与受载响应也真正device且接受，才能称更完整设备接触链；当前混合首片不满足该称谓。

候选 NEW 文件：`backends/_device_convex_query.py`（布局/所有权/状态协议）、`backends/_tyx_convex_query_kernels.py`（共用 support/query）、K1另加 `_tyx_convex_patch_kernels.py`；`tests/collision/test_device_convex_query.py`、`test_device_convex_patch.py` 与一个有限计量工具。父 OLD hooks：`tyx_collision.py` / `_tyx_collision_bvh.py` 只借用原位集、`_resident_collision_plan.py` 绑定epoch与候选、`_resident_contact_bridge.py` 可消费已完成patch并保原law、原 batch负责最后发布。`contact_patch_adapter.py` 作为完整HOST参照，不由作者平行改写。实际allowlist另行冻结，不现在实施。

## 4. 多案例与独立 oracle 草案

所有测试从公开注册/accepted pose进入；以下是预声明候选输入，不是本轮结果。复用原 `tests/collision/convex_family_cases.py` 仅作输入声明，独立期望不得调用新kernel生成。

| 组 | 输入与必核预期 |
|---|---|
| Q01 整族10组合 | 原sphere r=.05、capsule轴x∈[−.5,.5]/r=.05、OBB半边(.6,.1,.1)、非盒五边形柱H；平行支持面 gap=.02/0/−.01。法向+z、gap原值，含S为point，含C而无S为segment，余为area；K0必须明确patch尚未完成，K1查完整sites。零gap允许声明未决，不得漏成正分离或凭任意法向受力。 |
| Q02 有限特征 | 原capsule–box沿x移1.2的近失：真正最近点是有限轴端，gap>0；另cap侧/端入触、平行CC全重合区间、斜交轴。最近段参数手算并夹在[0,1]；线区两端、ties均保留，禁止无限线假命中。 |
| Q03 材料与协变 | 全10组合换序、axis-cycle+平移(2,−3,.25)、顶点/注册顺序重排及非连续ID。同物理surface重建、normal反号、gap/面积质心不变；不同world同局部ID不串联。地址只是本次材料见证，不宣传自动history station identity。 |
| Q04 非对称H面积 | 梯形环(0,0),(2,0),(1,1),(0,1)，z=±.1，另一体移z=.19；area=3/2，质心(7/9,4/9)，gap−.01。k100、零速度总法向1N；加边中点后的总力和质心/矩不变。强制检出“最多4点等权”或按site再乘一次权重。 |
| Q05 原法则受载 | 原point/2端/4角patch，gap−.01、k100，总N均1，不随site数增加。另d4、mu.25、v1=(.5,0,.25)、v2=0，原共同点合力(−.5,0,−2)、总功率−.75W；共同速度平移不改相对力。双方非零角速时逐site功率等于双方COM wrench功率，纯力偶不能被平均normal/point抹去。 |
| Q06 几何失败 | 重合球心、深round轴包含、多个最小法向、退化/坏H、rounded H、零轴、极大公共原点导致不可表示表面；沿原明确拒绝/未决。64顶点合法H与65拒绝、非有限/坏device读回/缺完整material ID；远处invalid H仍不能绕过预检。 |
| Q07 真物理消费者 | 在已接受普通/affine resident案例各选一个有限H静态压载及同树双dynamic，保持原h/k/质量/初态；只换opt-in几何路径，逐步完整state/forces/材料/功/反力对照。C10正式接受后再加root/末body浮根，不能用proposal替代资格。必须包括进入→承载→卸载、对方真实运动和多接触力偶。 |
| Q08 规模/故障 | W1/W4/W32，非连续键，共址独立world；每world S/C/H三shape，稀疏与密集分档，总量在原256/8192内。零候选、单候选、多site、pair/site/迭代容量耗尽、stale epoch/owner、候选写后晚故障、最后记录分配失败、close/重建、同请求retry。保存一组40步及tick19后21步完整重放，不重跑旧全矩阵。 |

K0距离 oracle 用解析圆/有限段、独立支持平面与现有冻结H手算；K1用手算polygon面积/质心、双侧 barycentric 重建和独立原law代数。HOST完整patch是差异对照而非唯一真值。设备工程误差、未决比例与实际fallback均报告，不追逐无限精确相触证明，也不以“误差小”掩盖不同形状/材料owner。

## 5. 有限容量与事务

两个阶段先计count，再作有界prefix/fill；pair/site/feature/iteration 各有明确上限、实际count、required capacity及状态。不要隐式分配 `pair_capacity × 全局site_capacity`，不让原子计数超限静默截断接触。每world site overflow沿原接触拒步合同，global candidate/device故障沿原整次未发布合同；无新半成功规则。

owned 拓扑表、两bank pose借用、输出scratch分离。一次query必须绑定同一owner/topology/epoch/active/tick；真正step的query、law、候选audit到最后publish不可串代。所有读取/验证/记录分配先于swap；失败不替换last accepted records，不消耗history/控制delay；重试重新计算，不缓存失败。分配中途、plan.close与batch.close都释放全部已分配资源，计数区分累计分配与当前live bytes。

## 6. 往返与计量：源码调用，不冒充实测

当前 resident BVH路径可逐调用核账：设 S为shape数，Q为实际最大query buffer行数。

- stamp 为int32[S,2]，一次D2H为8S bytes；bounds为FP32[S,6]，一次24S bytes（`_resident_collision_plan.py:705–717`）。
- shape BVH output实际读整个 `int32[Q,ceil(S/30)+3]`，不是仅命中行；H2D params为 `int32[Q,3]`（12Q bytes），layout变更另传nodes/order。来源 `_tyx_collision_bvh.py:9–57`；位集随后HOST展开flags/tuple。分区/分页时按每次实际array.nbytes累加，不能只报逻辑P×2。
- 原 `hull_intervals` 每次先传 `(2×64+1,3)` FP32 cloud（1548 bytes），每256方向页传完整axes（3072 bytes）、同步一次并读完整256×2输出（2048 bytes），尾页亦如此（`tyx_collision.py:353–389`）；它仍是HOST准备点/轴、HOST最终裁决的提议链，不是本提案的新设备斑块。
- 当前 ordinary contact每group的entry packet 5次read、entry mechanics4次、candidate packet5次、candidate mechanics4次；原状态/诊断完整输出仍会发生。另有status、stamp、BVH、affine读回。每次wrench上传24TB bytes、reject上传4T bytes（T/B按实际padding）；新kernel不能把这些原完整任务成本算成已消除。
- K0若暂用HOST candidate tuple上传，应显式列 pair buffer、pose/拓扑/输出的实际nbytes与次数。要称device candidates直达，必须借原bitset/compact索引同epoch句柄，稳态不得经HOST解码后再上传；旧完整公开query仍需要的D2H继续计。

未来计量分别报告cold编译/拓扑上传、暖query、完整step+同样public records；记录query/manifold/law HOST构造、H2D/D2H、显式sync、真实completed launches、padding bytes、fallback及live scratch。显式sync与隐式copy等待的墙钟采用排他计时，不能重复相加。读写gzip/审计不入物理计时；一方不能只回一个bool而另一方回全patch。

先 W1 与 W32 各一稀一密短正确性窗；输出/身份/容量完全合格后，再三独立AB/BA/AB未插桩同任务测量，保W1退化。没有收益也可接受明确设备查询能力，但不切默认、不报性能提速，更不拿Newton不同manifold/law的速度作同工作量比例。

## 7. Newton 1.7 的可借鉴层次与差异

本机 `.venv-newton-1.7.0/Lib/site-packages/newton-1.7.0.dist-info/METADATA:3` 锁为1.7.0；下面均为该目录的原源码，未运行：

- `newton/_src/geometry/collision_convex.py:27–34,142–165,214–260` 共用support与相对坐标，MPR overlap / GJK separated查询和之后的manifold是不同阶段；`ConvexQueryResult` 的两点/normal/distance不是材料面全斑块。`:37–48` 明说GJK cutoff余量是经验FP32 allowance而非证明，可借其明确界限，不能照搬inflate改变本项目物理gap。
- `narrow_phase.py:1245` 提供split query/manifold路径；`:970–974` 先计数且overflow保持可见、`:2130–2254`核各buffer，值得复用分层/容量思路。Phymeo失败要进入既有事务拒绝，不只打印warning后继续受载。
- `multicontact.py:651–667,700–730,937` 通过support扰动/polygon clipping选择至多4点及必要deepest。`contact_reduction.py:27–44,71–80` 的mesh reducer用normal bins、极端位置与depth、最多255代表点，目标是有限稳定接触集；两者都不是Phymeo完整面片的一阶面积积分权重合同。
- Newton输出的contact center/local points与原材料body绑定可供几何对照，但不是本项目稳定vertex-ID/chart与skin offset。必须独立保双侧地址；不能把contact index/原子输出顺序当持续站点身份。

后续原生参照只比较同几何的距离/normal/有界多点分布，先注明数量/权重/force-law差异。设备距离、完整device patch、HOST或device law、实际受载响应和端到端速度分别签收；本提案不把其中一层通过外推到其它层。
