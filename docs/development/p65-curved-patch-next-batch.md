# COL1b/1c：曲面 patch 与载荷的下一批最小增量

状态：**研究/规划草稿，未授权实施；无新增执行结果**。本文由现有源码和固定
Newton 1.7.0 本地源码得出。当前 COL 停滞修复的完整受影响矩阵由父执行中，
这里不提前宣布 COL1a 已接受，也不修改其 kernel/test/source。

## 1. 先收什么

建议接下来约五小时只做两层有依赖的小交付：

1. **B-point（中高）**：统一“至少一侧为单点支撑集”的曲面设备 patch。
   首片准入由形状性质选择 S/E，不写逐 pair 算法。
   新增九族 **SY、SN、SE、CE、BE、HE、YE、NE、EE**；单点、权重1、
   原始双方材料/offset、相同 pair frame 和同次 source 证据。
2. **C-point（高，必须等 B-point 通过）**：把这个子域接入现有 resident
   完整 J/M、同入步速度、双侧 COM 力矩和一次发布；四布局短窗及一个
   解析双动态卸载窗。复用力律和动力学，不另写求解器。

剩余九个新曲面族 **CY、CN、BY、BN、HY、HN、YY、YN、NN** 继续明确
UNSUPPORTED patch，直至母线/圆盘及混合交集的后继合同通过。这不否认它们
已经实现的距离查询。原十个 S/C/B/H 族保持已有行为和证据。

S 表示球，E 表示正半轴椭球，Y/N 表示有限圆柱/圆锥，B 是 OBB 资格，
H 是一般已准入凸包。B-point 针对选定合法接触法向下的零维接触离散；
不声称完整连续压力分布、Hertz 接触面积、静摩擦历史或全局唯一退让方向。

## 2. 本地研究依据与实际缺口

- [COL1a 合同§7](../superpowers/plans/2026-11-11-device-convex-support-family.md)
  已把距离、完整曲面 patch、真实载荷分开；本草稿是其中 b/c 的首片，
  不重开 GJK/EPA 收敛研究。
- [已有 HOST 曲面结果](../benchmarks/p65-convex-support-pairs.md)有 Y/N/E
  支持查询，以及 E–E/E–S/E–布/点的真实受载和卸载。其精细几何在 HOST；
  本批补设备来源与 resident 消费，不能将它们重新说成零实现。
- [设备 patch 核](../../src/phymeo/backends/_tyx_convex_patch_kernels.py):102
  当前对 tag4..6 全部 UNSUPPORTED。它的有限多边形路径不是曲面圆盘算法。
- [设备 query](../../src/phymeo/backends/_device_convex_query.py):898,1268
  已拥有 typed patch 和同次原始输出，但 patch 调用目前未传完整曲面
  dimensions/effective R/同候选 witness；不得在 HOST 再搜索补点。
- [载荷 frame](../../src/phymeo/backends/_resident_contact_bridge.py):71,110
  只准入 S/C/H。
- [表示桥](../../src/phymeo/backends/_resident_device_patch_bridge.py):150,233,305
  有三处明确旧假设：尺度只计顶点/radius；origin 必须等 world origin；
  offset 固定为 radius*n。曲面恰有非球 offset、dimensions 和 pair anchor，
  因此“打开曲面 patch 状态”不能直接升级为载荷通过。
- [原力律](../../src/phymeo/solvers/contact_patch_loads.py):10–38 已按真实
  共点速度计算牵引，并只乘一次站点权重。这部分直接复用。
- [单页/分页接受范围](resident-device-contact.md)有完整共同发布、四布局、
  raw/effective 差分与一阶矩；不是全设备装配，也不是 CCD。

固定上游阅读根：
`<parent>/.venv-newton-1.7.0/Lib/site-packages/newton/_src/geometry/`。

| 上游源码 | 读到的机制 | 本批采用/不采用 |
| --- | --- | --- |
| `narrow_phase.py:1477–1482` | 任一侧 sphere/ellipsoid 设 skip_multi_contact | 支持按单点支撑性质组织首片，非按每个组合单写算法 |
| `collision_convex.py:51–119` | 距离查询与 manifold/write 分离；单点用双 witness 中点 | 保持 query→patch→law 分层；不复制阈值 |
| `support_function.py:309–400` | E 解析支持；Y 端盖/侧面；N apex/base 选择 | 复用已移植支持语义，独立核真实边界和支撑值 |
| `multicontact.py:851–925` | 五个扰动支持方向、投影裁剪、最多四 manifold 点 | 这是有界近似 manifold；不能叫精确圆盘覆盖或完整面积压力 |
| `multicontact.py:927+` | 可靠性不足时还保留 deepest 点 | 不直接移植成无条件成功；本片未知仍拒 |

上游接触点数不是 Phymeo 压力积分权重的 oracle。本草稿不运行 Newton、
不比较未测性能，也不要求复刻其任意 tie 点顺序。

## 3. B-point 的可开发合同

保持现有公共 query/site_capacity API。建议新增内部
`_tyx_convex_point_patch.py`，接现有固定 patch wire；不要继续往大型旧
有限 patch 核里按九个 pair 堆分支。

输入必须来自**同次**设备 distance row、原 source dims/effective R、pair
anchor、原材料 IDs/bary/offset 和已准入顶点。对任一 S/E 的新曲面对：

- distance row 必须 RESOLVED，法向/双方 witness 有效；未决、容量失败或
  已声明的退化状态不得出点。需要父明确的同心/深包含/多法向边界仍拒绝，
  不新增任意选法向规则。选定法向不等于唯一全局 MTD 的证明。
- 核对该 S/E 的解析单点支撑、双方真实表面/支持面残差与双 witness 闭合。
  另一侧可为凸包面、胶囊轴、圆柱圆盘或圆锥面；它的 witness 必须落在
  完整有限支撑集上，不能因为有一个 GJK 点就跳过该检查。
- 这些检查只使用有界解析式、原有限顶点/bary 和已有支持函数；不另跑
  GJK/EPA、不 HOST nearest search、不把一圈采样点变成圆盘。
- 合格时 dimension=0、site_count=1、weight=1，common 为双方中点，
  完整保留双方材料 ID。每个 raw offset 和 surface 都使用本行 pair frame。
  原 first moment 门仍逐 patch 核，即使只有一个点。
- 不合格返回原未决/不支持或资源状态，不发布部分 sites。不伪造
  device_resolved 或 skip_host_patch_search。
- 工程门原样：gap 1e-4 L；surface/重构/闭合与 first moment 2e-4 L；
  normal L2 2e-3、单位长度 2e-4；bary sum 64u32/非负。
  source/epoch/owner 精确绑定不变。检查开销和新增读回如实计费。

“严格凸一侧使选定支撑交集至多是点”仅给零维离散的依据。它不能独自
证明查询法向准确，也不能把深包含中的任意方向变成可验收载荷。首次 tests
提交必须同时写正控和这些准入负控；如果无额外猜测便无法明确分域，停在
设备几何接缝，先不开放 C-point。

### 固定多案例入口

| 组 | 预登记身份/输入 | 主要独立标准 |
| --- | --- | --- |
| P01 | 九族各 sep/touch/overlap，共27身份；复用 COL1a 已有独立 literal | 双侧真实边界/支撑、原 gap/normal 门、dimension0、单权重、完整 ID，不将 placement overlap 当深度 |
| P02 | 九族各一个现有 cycle/反序/资产旋转变体，保原 literal | 原有换序/材料双射、pair anchor、不依赖世界原点大小 |
| P03 | E–B 四种 accepted-source 布局，每个三次真实连续步后查询 | 非零 entry v/drive/外载；动态 body R×asset R、epoch、最后关节/浮根真实来源 |
| P04 | Y侧母线–H、Y端圆盘–B、N底盘–Y、NN母线等无 S/E 子域 | 明确 UNSUPPORTED/零sites；不拿采样点假装完整 patch |
| P05 | 两个同页合法点：capacity2恰满、capacity1失败；再含零力/分离点 | 无截断，原完整站点资源语义与失败保留 |
| P06 | 错 source/epoch/dims/R/pair anchor、坏材料 offset/一阶矩、末次读回失败 | 原 state/query/receipt 原子性；无 HOST fallback；修正输入后同 plan 重试 |

实现前列准确 pytest 节点清单；上述身份不得根据首轮输出删减。正控仍用
独立解析方程/有限材料 oracle；同输入 HOST 比较只能作为额外对照。

## 4. C-point 的最小真实载荷合同

沿用 `narrow_phase="device_patch"` 与原 `RobotContactMaterial`。
必须同时补下列接缝；只对 B-point 资格放行，旧 HOST 几何入口不暗中扩域：

1. ContactShapeFrame/registration 允许有资产曲面描述的合法来源。
   动态有效旋转仍为 accepted body R×asset R；static body orientation 的
   COL1a 限制保留，不能把非 identity asset R 丢掉。
2. source adapter 比对 dimensions、shape R 和 pair anchor。局部尺度计真实
   几何尺寸，不乘绝对世界坐标。
3. 曲面 offset 不用 radius*n 替代。按完整声明 source 重建 material center
   与 offset，独立验证真实表面/支持残差，保全部 raw→effective 差分。
   禁止 HOST 搜索/投影“修好”坏 raw 点。最终严格 ContactSite 可按原规则
   归一 normal/weights、重算共点/gap；所有变化仍受原工程门约束。
4. 复用一次采样权重、真实共点速度、完整 J/M、双侧反力和各自 COM 力矩。
   独立参考先由原输入 M/J/真实 v 验力学；不能仅拿输出的 force/qacc 互证。
5. 继续原 staged query→conversion→load→solve→receipt→单一 commit 事务。
   末页/末 world/最后 force/receipt 失败不发布前面结果；failed work 仍计费。

最小正负控不以 G1、递布或一对形状代替通用验证：

- **C01 四布局各三步**：倾斜 E–B、E–H 或 E–E 的既有 B2 布局参考，
  非零 entry v、drive、external；每一步独立 M/J/force/q/v/wrench，
  保原 5e-6 absolute / 5e-5 relative（mass 2e-5）力学门，几何门分开核。
- **C02 质量/步长**：普通轴向 E–B，m={0.5,2} kg，
  h={1/1024,1/512} s 的四输入；原 k=64 N/m、入速1/64 m/s，
  非零驱动与外载保持；位移/速度用独立离散递推，不能 reset 制造事件。
- **C03 一个完整卸载窗**：同树两个独立平移支路上的 E–E，
  m={0.5,1} kg、初始正 gap=1/64 m、相对入速0.5 m/s、k=64 N/m、
  damping=friction=0，无外载；h=1/256 s、80步。独立两质量离散振子递推
  固定入层、峰值和最终卸载行为，同时核双侧冲量与 COM 力矩。
  这是新 resident 消费路径；不重跑旧 CollisionLoadSystem 已绿窗口冒新结果。
- **C04 最小事务/滑移**：正负切速各一小窗、selected reset 后真实下一步、
  最后转换失败/最后收据失败及一次重试、恰满/少一容量。正负力及力矩
  必须基于同一共点速度，设备路径禁止调用 HOST contact_patch_sample。

C01/C02 可作为首个载荷接受边界；若串行时间不足，C03/C04 保留待办而不能
称完整持续曲面载荷已验。跨独立机器人/跨 storage 联解由机器人共享层负责，
本片只使用已接受的同树双动态组或静态环境。

## 5. 文件责任与并行边界

| 所有者 | B-point | C-point（条件） |
| --- | --- | --- |
| 碰撞作者 | NEW `backends/_tyx_convex_point_patch.py`；MOD `_device_convex_query.py` 调用/typed表示；NEW独立 reference 与 geometry/source tests | MOD `_resident_device_patch_bridge.py` 的有限表示适配；NEW curved adapter/physical tests |
| 父共享 owner | 审 raw ABI/源与容量接线；已有 GJK、有限patch核不改 | `_resident_contact_bridge.py` 的 frame 准入；必要 `_resident_contact_plan.py`/bundle资格门。公共参数若不变则不改 `tyx_world_batch.py` |
| 原 reviewer | tests-first 独立 surface/material/oracle、后源码review | 同输入 J/M/力矩与原子性、安装包 smoke |
| 训练/机器人作者 | 可继续不相交源码 | 不并行编辑上述共享 owner 文件 |

建议新 tests 按 geometry/source/consumer 三文件分开；独立 reference 仅含固定
输入和解析式。完整具体路径/ownership 由父在新编码 grant 中冻结。
MuJoCo-TYX 的 shape准入、原生 collision/约束求解、训练 VecEnv 不属于此片，
本结果不会自动扩大该第二运行路径的碰撞域。

## 6. 五小时内的取舍与停止规则

建议预算：约30分钟冻结点域/原始→有效规则和测试；90分钟 B-point纯编码；
45分钟父review/串行两后端；60分钟条件 C01/C02；最后45分钟验证、saved
membership及源码外 smoke。只是顺序和工作上限估计；机器人/训练占用同一
执行槽时不并行抢跑，不保证把五小时编成同时开跑的工时。

硬依赖：当前 COL1a 候选完整矩阵接受 → B-point 代码/完整多案例接受 →
C-point 表示审阅 → 真实受载。B-point未过时 C-point只能写合同，不能接力；
不能用“query通过”替代 patch 或 load 接受。

成功至少有：九族点域真实 interpret/CUDA、原始/有效材料与矩门、来源负控、
旧十族只跑被接缝修改影响的最小回归。载荷另按 C01/C02/C03/C04 逐层记事实。
每段保首失败、完整小证据与源码hash，不追加证明包或同源全绿重跑。

若30分钟内无法明确同心/包含/法向歧义准入，先裁定点模型，不写隐式 fallback。
若首次实际失败属于数学/表示合同或一次修复未缩小障碍，停止该子域，交最小
反例和已通过范围；不扩大128面/64顶点、调物性/尺寸/门、复制逐pair特判，
也不挤占其它独立方向。母线、端圆盘、混合曲面面积、CCD和持续摩擦身份是
下一独立节点；明确近似 manifold 模型后才能实施，不以本片点接触代表完成。
