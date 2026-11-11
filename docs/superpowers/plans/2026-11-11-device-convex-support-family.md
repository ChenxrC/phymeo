# COL1a：统一设备凸支持族与十八类新增查询

**状态：接口/案例冻结草案；未实施、未执行。** 基线 `95212cff12cd6437566fdce80b93ddc7c4b64fe7`。本页只授权下一可编码片的范围设计，实际编码/唯一串行验证由父派发。服从[持续总计划](2026-11-11-newton-parity-continuous.md)。

## 1. 目标、复用与边界

七形状为 S=sphere、C=capsule、B=box、H=hull、Y=cylinder、N=cone、E=ellipsoid。B 仍是八顶点 hull 的能力标签，不新增 `CollisionShape.kind="box"`。由注册顺序的 `i<=j` 自动产生28个唯一无序组合，而非逐 pair 增加 solver。

- 旧十类 SS/SC/SB/SH/CC/CB/CH/BB/BH/HH 按[现设备查询、patch与受载证据](../../benchmarks/p65-parity-20261111-connections.md)复用；只对实际修改影响的旧节点做有限回归。
- **首片只新增十八类 device query**：SY/SN/SE/CY/CN/CE/BY/BN/BE/HY/HN/HE/YY/YN/YE/NN/NE/EE。新曲面 patch/load 留后继，不能用支持采样点冒充完整曲面接触片。
- [既有曲面报告](../../benchmarks/p65-convex-support-pairs.md)已接受有限 HOST support-map 查询；其中椭球–椭球/球/三角面/材料点还含真实载荷。逐条保留其 HOST 几何＋TYX物理范围，不能写成曲面从未实现或从未受载。
- 此处是 resident 路径；MuJoCo-TYX 原生 primitive/约束另有模型与证据，本片不改，也不互相外推资格。训练 N1/N2 独立推进；ROB1联解不是几何查询前提，共享绑定文件由父独占。

能力记录仍用 `(unordered_pair, operation, owner_mode, backend)`。static/manifold/response/translation_ccd/rotation_ccd 分开，HOST/interpret/CUDA分开。[registry](../../../src/phymeo/geometry/collision_capabilities.py)是事实表，不是运行dispatch；缺行不等于没代码。父归并旧报告与新B1/B2/B3事实，新行只有执行后才由 implemented_unverified 升为 verified。

## 2. 已核源码与实现选择

| 已读位置 | 本片约束 |
|---|---|
| `constraints/curved_solids.py`；`geometry/_convex_support.py:_Body.support/material` | Y/N尺寸(r,h)，E尺寸(a,b,c)；单中心原material ID＋offset；已有解析支持公式，不造凸包代理 |
| `geometry/_curved_solids.py:_rotation` | 动态有效frame=accepted body R × asset R；static asset R就是world R |
| `backends/_device_convex_query.py:_compile_source/_layouts/_read_raw` | 原0/1/2 tags为S/C/H，3为unsupported；原wire最多四material槽、完整source/epoch绑定 |
| `backends/_tyx_convex_query_kernels.py` | GJK、EPA seed、EPA expansion三处同语义；重复按Minkowski坐标判定，不能按曲面唯一中心ID删不同support点 |
| `_resident_tree_binding.py`、`_resident_floating_binding.py` | 仅放行curved tag会导致零radius单中心AABB漏检，必须先改绑定准入与保守边界 |
| `backends/_tyx_convex_support.py` | 当前TYX纯函数只接受单return表达式；循环/分支扫描留kernel，不改编译器或pin |

固定参照为父树 `.venv-newton-1.7.0/Lib/site-packages/newton`：`_src/geometry/support_function.py:221,309–400` 的通用支持函数及三曲面分支；425–440的lean版本明确删去这些族，不能误用。其 `collision_convex.py:142–147` 以同一support接MPR/GJK。本片复用公式/统一接口思想，不复制inflate/contact-reduction阈值；Newton的barrel-cylinder扩展不在当前Y域。tie只比较支持值/合法集合，不强求Newton任意选点相同。

本机 `C:/Users/38347/Documents/ChatGPT/phymeo/.venv/Lib/site-packages/tyx/frontend.py:155,323` 支持条件表达式、要求纯函数单表达式。以上均是纯文本依据，不是新运行或性能结果。

## 3. 冻结API与数据合同

**公共入口不变：** `CurvedSolid`＋`CollisionShape`＋四类binding＋`batch.prepare_convex_query(...)`＋`plan.query()`。保原pairs/pair_capacity、GJK/EPA预算、memory_limit_bytes/site_capacity参数；不增加solver选项或隐式HOST搜索。查询借同一接受FK bank，affine广义坐标不当作body pose。

**shape/source：** tag4=Y、5=N、6=E，保3为unsupported。原一个中心ID与radius=0不变。四树broad的 `geometry[0]` 对Y/N用向外包围的r+h，对E用max(a,b,c)；只是保守外包半径，不写入shape.radius/narrow skin，不生成代理顶点。local/path/depth原逻辑保留。父处理 `from_asset` 时须合成folded/welded site完整orientation；若现接口尚不能保真，明确拒绝该asset映射，不能仅移动中心就放行。

**frame：** 固定asset R可HOST一次验证/上传；动态有效R必须由device用本次accepted body R合成，static用asset R。不每步HOST FK。维度、orientation、原key/revision/group/world/layout、lease/bank/epoch及原材料完整绑定。

**source证据：** 保 `resident-convex-source-v1` 原必需字段；含新族时添加 `support_family_schema="resident-convex-support-family-v1"` 以及按shape排列的：
`declared_dimensions_m/uploaded_dimensions_m/dimension_quantization_error_m`，
`declared_orientation_xyzw`，
`declared_asset_rotation/uploaded_asset_rotation/asset_rotation_quantization_error`，
`effective_shape_rotation`。
Y/N第三dimension为0，旧族dims为0、asset R为I；effective R为本次device值，全部不可变。旧-only source不用新增字段。上溢、下溢、非零上传丢失、非有限和错源仍走invalid或原SOURCE_PRECISION，不能判分离。

**统一support：** NEW `_tyx_convex_support_family.py` 提供三个单表达式 `tx.func`：
`curved_support_x/y/z(kind,dx,dy,dz,dim0,dim1,dim2)->f32`。
E为 `dims²*d/||dims*d||`；Y为半径r的径向点＋h sign(dz)，径向零取轴上合法点；N比较apex与base circle支持值，tie固定规则。调用kernel处理零方向。三处GJK/EPA全部使用一致的局部支持→有效R→world-relative转换。

任一侧新族时使用full-surface support：S/C含skin、H扫完整顶点、Y/N/E用解析公式，结尾不再加radius。旧十族core+skin路径保持。新阶段标签GJK_SUPPORT只追加，不重编号旧_PHASE；EPA仍原标签。预算GJK≤64、EPA≤192、EPA vertices≤64/faces≤128、pair≤32640、site≤4096及原shape/vertex门不提高，新增上传/scratch实际计费。

**材料/wire：** 原wire尺寸不变，scratch保实际support点；最终曲面为单中心ID、weight1、offset=实际见证−中心。混合侧按原ID/bary聚合实际支持坐标，先聚合坐标再合ID，不能塌回中心。独立表面/包含与见证闭合门同时检查；重构正确本身不证明点在真实边界。

**patch隔离：** `fill_convex_patch` 遇tag4..6明确UNSUPPORTED、零sites、完整性flags=false；几何可以RESOLVED。必要受载候选因此拒步，保全部旧发布，无HOST fallback。原十类patch和默认HOST曲面能力不改。

## 4. 独占文件范围

| Owner | 精确路径（相对仓库） |
|---|---|
| 碰撞lane，NEW | `src/phymeo/backends/_tyx_convex_support_family.py` |
| 碰撞lane，MOD | `src/phymeo/backends/_tyx_convex_query_kernels.py`、`_device_convex_query.py`；`_resident_convex_source.py`仅必要source补充；`_tyx_convex_patch_kernels.py`仅新族unsupported guard |
| 碰撞lane，NEW独立reference | `tests/reference_newton/rr/device_convex_family_reference.py` |
| 碰撞lane，NEW tests | `tests/collision/test_device_convex_support_family.py`、`tests/collision/test_device_convex_family_sources.py` |
| 父，共享接缝 | `src/phymeo/backends/_resident_tree_binding.py`、`_resident_floating_binding.py`准入/broad/frame；`src/phymeo/geometry/collision_capabilities.py`和能力文档事实合并；`tyx_world_batch.py`仅真需公开转发时 |

affine/floating-affine现委托基础binding，先复用，具体缺口另报父，不自行扩文件。旧三份K1 WIP不读入测试、不修改；训练、MuJoCo-TYX、ROB1 solver、现受力bridge不在本片范围。

## 5. 固定多案例，不按产品输出造oracle

以下是预登记身份，未collection。先提交reference/tests，再实现。独立库不得导入product/TYX/Newton producer作期望。

新literal（米）：S r=1/8；C轴x∈[-1/4,1/4]、r=1/16；B半边长(3/8,1/4,1/8)；H为z=±1/8五边柱，xy环(-1/4,-1/8),(1/4,-1/8),(3/8,0),(1/8,1/4),(-1/4,1/8)。Y(r,h)=(1/8,1/4)，N=(3/16,1/4)，E=(1/4,3/16,1/8)。全原ID/owner保留。

| 身份 | 输入与独立验收 |
|---|---|
| G01，18×3=54 | 轴为z，按上下支持高度设置轴向布置参数s=+1/32、0、-1/64。s>0的共同支持面给出精确gap=s；s<0仅为placement overlap，不直接充当EPA最小平移signed_distance。正负例仍须RESOLVED/正确符号；touch沿原CONTACT_BAND/明确unknown，不能误判SEPARATED |
| G02，18 | G01正gap整体以q=(1/2,1/2,1/2,1/2)循环旋转、平移(4,-2,1)，同时反向请求；gap不变、法向反号、双方材料与requested mapping完整 |
| G03，YE/NE/EH | n=(2,3,6)/7，gap=+1/32；中心差固定为独立解析sA(n)−sB(-n)+gap*n。支持平面证明分离最近距离，不采样拟合曲面 |
| G04，每新族 | 方向表为±三轴、(1,±1,0)、(0,1,±1)、(1,0,±1)、(2,3,6)，identity及非identity asset R；核支持值、集合成员、方向缩放与tie |
| G05，负控 | 非法维度/quaternion、上传精度丢失、零方向、深同心/nonunique、GJK/EPA预算或容量不足；明确invalid/unknown/resource，不伪造分离 |
| S01，3族×4布局 | 每身份四sparse world、不同非零q/v/asset R/中心偏置，3次真实连续物理更新后query；独立FK/几何核accepted bank，不能只拿query自报pose自证 |
| S02，broad | Y/N中心不重叠而实体侧面相触；细长E转动后接触；static/dynamic和dynamic/dynamic，防止一中心AABB漏检，保全allowed-pair集合 |
| S03，source/事务 | body R×asset R、共同大平移；错owner/world/group/layout/revision、stale lease/epoch、reset/clone下一真查询、最后行坏readback、资源恰满/少一、零pair与尾项；拒步原发布不变、同输入修复重试 |
| S04，边界/兼容 | 曲面距离＋site_capacity得到UNSUPPORTED patch；必要受载不HOST fallback/不部分发布；旧默认HOST及同页旧族不变 |
| S05，受影响旧节点 | 从原reference选每旧族一个非零gap，及实际受改影响的CB/CH cycle_reverse_ids、partial face、clear/晚失败；保原literal/gates，不重跑无关旧绿全套 |

G01非零例是必须通过的有限域，不能整批unknown算成功。负例须独立证明真实interior overlap及正确符号；首次tests提交前逐族列明depth/witness oracle：采用独立解析、边界/支持判据或语义匹配的外部参照，实际核对穿透深度与见证，不能只核负号。轴向重叠量不普遍等于最小平移穿透深度，不能把-1/64或其绝对值当作通用depth期望，也不能只拿product自报normal/witness自证。G05深穿入只核诚实状态，不宣称任意深穿入已支持。手工身份数不冒充实际pytest节点数。

## 6. 固定工程门与执行顺序

L为独立输入的max(1m、两体真实直径界、中心间距)，不含绝对世界坐标；Y/N直径界2(r+h)，E为2max(a,b,c)，旧族原定义。期望不能从result.scale反推。

沿原device门：gap误差≤1e-4 L，相对上述独立距离/depth oracle，绝不相对负例轴向布置参数s；surface/材料重构/闭合≤2e-4 L；normal L2≤2e-3、norm误差≤2e-4；scale误差≤1e-4 L；bary sum误差≤64×2^-23及非负/完整ID。G04支持值误差≤1e-4 L||d||、集合/表面距离≤2e-4 L。多解点/法向核合法集合，不硬套唯一点。量化差分完整保存；原patch权重/一阶矩门不动。

父唯一串行执行：共享binding接缝→全文review/source绑定→必要静态→短support/source→新矩阵interpret→真实CUDA。只针对失败节点修复后跑新源受影响组，不调fixture、gap或预算。Newton固定1.7.0作为外部参照，cutoff/tie/contact-reduction差异单列；首oracle仍为独立解析/literal，不以输出点数相同替代同一数学对象。

每后端一次compact保存全部预定identity的input、声明/上传几何/量化、source/epoch、raw、typed材料、资源/状态和oracle；空结果也保存。完整成员/hash/shape/raw-typed审计不重跑producer、不重算巨大证明包。源码外公共wheel smoke固定Y–H/N–C/E–S、两sparse world、非identity来源三次连续查询及curve patch拒绝；构建/安装由父串行一次。完成后只升级对应query能力行。

## 7. 后继门和停止规则

**COL1b（高）**先冻曲面支持集合：E唯一点、Y侧母线/端圆盘、N apex/母线/base disk，以及混合交集。须完整片或明确可标定近似及面积/权重/力矩误差；几个support样本不是完整圆盘。原HOST有限椭球接触逐条复用。

**COL1c（高）**按b通过子域接同入步速度、完整J/M、双方COM力矩、一次发布。至少倾斜E承载/滑移卸载、Y跨有限边、N apex→侧面换特征、双动态反作用；质量/步长/连续门另在编码前冻结。ROB1未接时只能宣称同树双body/静态环境，不冒跨机器人联解。

本片不含mesh/非凸、变形、CCD、摩擦历史、SDF/hydroelastic、完整曲面压力或MuJoCo-TYX扩域，也不承诺快于Newton。COL2/3独立后继。

失败先归source/算法/精度退化/资源/oracle语义/工具链并保存最小证据。连续两次针对性尝试未缩小同一障碍，停该子域由父决定拆片，独立训练/机器人继续；不得通过改质量、形状、阈值追绿。几何通过即收COL1a，不能为尚未授权的曲面压力或连续精确证明无限追加理论。
