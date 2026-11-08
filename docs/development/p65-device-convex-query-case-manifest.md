# K0 设备凸体距离与有限见证：候选案例清单

**CASE_MANIFEST_PROPOSED_UNEXECUTED / API_NOT_FROZEN。** 单文档纯作者，未 import、生成数值、执行 query/test/static/native 或测量。基线 child `be4f1a052a13c2d3227fc4a4ab8599c044e64a42`；源码依据为父读取时 `2966aa9f24708180ed9c25316a7dad6be0139984`。本清单供父冻结 tests-first，不能替代当前 C10/C11 验收，也不授 K0 产品实施。

## 1. 边界与源码身份

K0 只交付当前 accepted pose 下的距离/相交状态、法向存在性、双侧有限表面见证及其真实材料地址。K1 才负责完整 point/segment/polygon patch、面积/线权重与所有 sites；K2 才接触步消费。K0 成功不允许跳过旧 HOST 完整 patch/law，不要求 K1 先完成；本轮案例均不施力、不改变材料律，不把 query 次数称物理子步。

| 只读来源 | 本清单采用的事实 |
|---|---|
| [原提案](p65-device-convex-narrowphase-next-proposal.md) §3 K0、§4 Q01–Q03/Q06/Q08 | 整族一个有界协议，raw 与公开记录分层，未决不得报分离。 |
| `tests/collision/convex_family_cases.py:9–44`；`tests/collision/test_contact_patch_adapter.py:32–80` | 十种无序组合、原尺寸/顶点次序/ID、三 gap、+z 法向、有限 C–box 端点近失。原 HOST patch 是差异对照，非唯一 oracle。 |
| `src/phymeo/constraints/collision_shapes.py:12–25,95–104,140–155`；`src/phymeo/geometry/_convex_hull.py:80–114` | 256 shapes/8192 总点/64 hull 点，ID 为非负 i32；重复 ID、重复 hull 坐标、非满维和容量有不同原边界，不能统称 separated。 |
| `src/phymeo/geometry/_round_sweep.py:112–155`；`src/phymeo/geometry/contact_patch_adapter.py:283–305` | 零长轴按参数 1/2 处理；两个不同材料 ID 同坐标并非统一非法。核心重合的唯一法向、深 round-axis containment、多法向 patch 另有限制。 |

本机 pinned 根为父 `.venv-newton-1.7.0/Lib/site-packages/newton/`。`_src/geometry/collision_convex.py:27–48,142–260` 将 query 两点/normal/gap、MPR/GJK 与 manifold 分层；`:37–48` 明说 cutoff 余量是经验 FP32 allowance。`support_function.py:195–217,221–307` 区分真实 mesh/box 支撑及有限 capsule（Newton 局部 +Z、half-height=.5、radius=.05，必须旋转到本例 +X）。`multicontact.py:651–667,700–730` 的至多 4 个裁剪点及必要额外点不是本项目整面面积积分。`narrow_phase.py:967–975,1245–1264` 保计数溢出并分 query/manifold 队列。原生后续只能作同几何参照，不拿其 boolean、任意支撑顶点或点数当本项目完整材料 patch。

## 2. 精确共同声明与方向约定

长度 m；第一体局部原点为零，第二体只平移 `(0,0,eA+eB+g)`。原世界 `w=17`，键 `A=(17,1,0)`、`B=(17,2,0)`，ID 为 `100+i`、`200+i`，点次序完全沿原函数；radius 不改。每个输入还保存原 FP64 坐标、实际上传 FP32/相对原点、真实 stored pose/binding、source revision。

| 形状代号 | 原完整声明（不得换成包围盒/无限线） | e / 点数 |
|---|---|---|
| S | 中心 `(0,0,0)`，r=.05 | .05 / 1 |
| C | 有向材料轴端 `(-.5,0,0),(.5,0,0)`，r=.05 | .05 / 2 |
| B（OBB） | `product((-.6,.6),(-.1,.1),(-.1,.1))`，注册仍是 hull，r=0；只有原真实盒检查通过才特化 | .1 / 8 |
| H | 环 `(-.5,-.1),(.5,-.1),(.6,0),(.3,.1),(-.5,.1)`，每环点依次 z=-.1,+.1，r=0 | .1 / 10 |

此文 normal `n` 从声明第一体指向第二体；`g=(pB-pA)·n`，正分离、负浅穿入。候选 API 若 canonical 排序，先按实际两个 owner 还原此方向再比较；仅注册顺序颠倒不能误要求 canonical 法向反号。K0 原输出和方向转换过程都保存，不能只存整理后的值。

独立支撑平面证明：A 最高点高度 eA，B 最低点 eA+g，投影交集非空；正 gap 的欧氏最近距离恰 g，浅负 gap 的最小平移轴为 +z、深度 -g。见证必须位于各自真实表面；仅投影 gap 正确而点在无限延长面上不通过。S/C 材料 core 加 `+r*n/-r*n` skin；H/B 必须用原 vertex IDs 的非负凸组合重建。对并列最近点不强求某个任意顶点或唯一 barycentric 分解。

## 3. Q01：十族首片（30 个独立纯 query）

候选节点 `test_k0_family_gap_witness`，身份 `K0-Q01-{SS,SC,SB,SH,CC,CB,CH,BB,BH,HH}-{sep,touch,overlap}`；g 分别 `.02,0,-.01`。每身份 1 次 query、0 物理步，独立归档。以下集合描述用于验点是否真实，不要求 K0 枚举整个集合。

| pair | 最近见证 xy 集合 / 独立材料限制 | K1 后继维数，不能冒作本次完成 |
|---|---|---|
| SS | 两中心投影 `(0,0)`，双方唯一 core ID | point |
| SC | `(0,0)`，C 轴参数 1/2 | point |
| SB | `(0,0)`，B 下表面原四顶点内 | point |
| SH | `(0,0)`，H 下表面原五顶点内 | point |
| CC | `(x,0), x∈[-.5,.5]`，两轴对应同 x | segment |
| CB | 同上，B 下表面内 | segment |
| CH | 同上，H 下表面内 | segment |
| BB | `[-.6,.6]×[-.1,.1]` | area |
| BH | 原 H 五边形环内部（恰是两投影交集） | area |
| HH | 原 H 五边形环内部 | area |

正/负非零 20 行须真正 device-resolved 且通过 raw/材料检查，才能称整族 K0 首片通过；不能用 20 次 HOST completion 充数。touch 的 10 行允许显式 unresolved 或同一相触状态，禁止不带不确定性地声称严格分离。没有完整 patch 的记录必须明示该层尚未完成，不能把一个见证变成一站受载。

## 4. Q02：有限端、边、面留出（9 个纯 query）

候选节点 `test_k0_finite_features`；以下第一/第二顺序明确，未指定 ID 时仍用 §2 两组 ID，形状按需使用原 C/B/H，S 半径 .05。每行 1 次 query、0 步；全部公式由有限段投影、点到线/平面或端点距离得到。

| ID | 精确输入 | 独立预期 |
|---|---|---|
| `end_box` | 原 CB、g=-.01，仅 C 加 `(1.2,0,0)` | core A=(.7,0,0)，B=(.6,0,.04)；d=√29/50，g=d-.05>0，n=(-5,0,2)/√29；C 参数0/ID100。无限轴会假命中。 |
| `cap_side` | C 在原点，S 中心 `(0,0,.09)` | d=.09，g=-.01，n=+z；C 参数1/2。 |
| `cap_end` | C 在原点，S 中心 `(.59,0,0)` | d=.09，g=-.01，n=+x；C 参数1。 |
| `cap_oblique` | C 在原点，S 中心 `(.575,0,.1)` | 最近 core=(.5,0,0)，d=.125，g=.025，n=(3/5,0,4/5)。 |
| `cc_parallel` | 原 C；另一 C 端 `(-.25,0,.09),(.75,0,.09)` | g=-.01，n=+z，合法最近 x 区间[-.25,.5]；返回点在该集合，不要求 K1 两端归一权重。 |
| `cc_skew` | 原 C；另一 C 端 `(0,-.5,.09),(0,.5,.09)` | 两参数均1/2，g=-.01，n=+z；不是端点采样距离。 |
| `cc_end_end` | 原 C；另一 C 端 `(.6,.1,0),(.6,1.1,0)` | 参数(1,0)，d=√2/10，g=d-.1>0，n=(1,1,0)/√2。 |
| `h_slant` | H 在原点，S 中心 `(.63,-.13,0)` | H 真斜面 x-y=.6，脚点(.55,-.05,0)，d=.08√2，g=d-.05>0，n=(1,-1,0)/√2；H 的 AABB 反会给 `.03√2-.05<0`，必须检出代理错误。 |
| `h_vertex` | H 在原点，S 中心 `(.68,0,.16)` | H 真实顶点(.6,0,.1)，d=.1，g=.05，n=(4/5,0,3/5)；表面/法向锥均核。 |

通用独立有限段 oracle 是 `t=clamp((x-a)·(b-a)/|b-a|²,0,1)`；两段需检查内部驻点及四个端点投影候选。面/边/端参数和 active bound 原样保存，不从新设备结果反算期望。

## 5. Q03：顺序、协变与身份（50 个纯 query）

候选节点 `test_k0_frame_and_identity`；对 Q01 的十种 g=-.01 原输入，各新增五个独立身份，均1次 query、0步：

- `reverse`：对调请求两侧，依实际返回 owner 处理方向；双侧点/地址交换、方向随逻辑 first/second 反转，gap 不变。
- `cycle`：`R(x,y,z)=(z,x,y)`，平移 `(2,-3,.25)`，正常顺序；n=(1,0,0)，p'=Rp+t。
- `tilt`：`R=((3/5,0,4/5),(0,1,0),(-4/5,0,3/5))`，同平移；n=(4/5,0,3/5)。这是实际非轴对齐 OBB/有限轴，不仅坐标置换。
- `permute`：每侧 `(vertex_id,point)` 配对整体逆序，shape 注册顺序逆序；原物理键与材料 ID 保留。绝不只重排点不重排 ID。
- `cycle_reverse_ids`：cycle+请求反序+注册逆序，物理 body/shape 键改为 `(w,71,29)/(w,13,7)`；原第 i 材料点重标为 `2**31-1-101*b-7*i`（b∈{1,2}），显式映射回原地址。不得 FP32 化或按 row slot 当 ID。

唯一最近点比较实际 Rp+t；并列集只核 gap/normal、真实集合成员、各侧材料重建和 support 条件，不强求 tie 所选点/权重与原行完全一致。确定输入的结果按完整 pair/key 对齐，不能排序浮点坐标来推断材料连续性。反复 query 不创建 episode，也不据这些 ID 继承摩擦弹簧。

## 6. Q06：近零与有限失败域

`test_k0_near_zero` 取 SS/CH/HH，各 g=`±2**-10, ±2**-20`，共12身份，各1次 query；原 g=0 复用 Q01 身份，不另重跑。外两值远离 §8 余量带，应给正确符号；内两值可 unresolved/HOST completion，不能用 round-to-zero 后的结果掩盖原 gap、输入字或错误 separated。

`test_k0_domain_classification` 的下表共16个固定声明尝试，0步；预检拒绝也保存原输入/异常，但不伪计 device query。词语是建议分类，不冻结 enum/API。

| ID / variants | 输入 | 允许结果与禁止结果 |
|---|---|---|
| `coincident_ss` | 两 S 中心同(0,0,0) | overlap 已知；唯一 normal 不存在，unresolved/unsupported-normal，不能任意 +x 受力或 separated。 |
| `deep_center` | S 中心0，原 B 中心0 | 包含，最小退出深度 .15，±y/±z 并列；保旧 patch 拒绝，K0 状态可有定义。 |
| `deep_offset` | S 中心(0,0,.02)，原 B 中心0 | 包含且最近退出 +z/.13；旧 round-axis-inside patch 仍 unsupported，不能因唯一距离方向就放开原受载域。 |
| `identical_hh` | 原 H 完全重合 | overlap/多法向未决；不得伪造唯一法向/分离。 |
| `zero_axis` | C 两点均(0,0,0)，ID100/101不同；S 中心(0,0,.12) | 球等价 g=.02、n=+z；core参数1/2可重建。device或明确HOST completion，不能新增构造拒绝。 |
| `flat_h_far` | H 四点(±.5,±.1,0)，另一 S 中心(10,0,0) | 原满维预检失败，即使AABB远离；不跳过为分离。 |
| `duplicate_xyz_far` | 原 H 最后坐标换成首坐标、ID仍不同；S在(10,0,0) | 原 convex_duplicate_coordinates，不因宽相拒绝隐藏。 |
| `bad_address` / `repeat_id,missing_point` | 原 H 分别重复一个材料ID；或少一个对应pose行 | 原注册/完整source绑定拒绝；不得默补或静默删除。 |
| `nonfinite` / `nan,inf` | 原 H 一个坐标分别 NaN/+Inf | 原有限输入门拒绝；不归为 separated。 |
| `rounded_h` | 原 H radius=.01，与S分离 | 注册可能合法，现 K0 声明族及原patch不支持；显式 unsupported，不能偷偷清零radius。 |
| `h_capacity` / `v64,v65` | 半径.5的32边形柱 `( .5cos(2πj/32),.5sin(2πj/32),±.1 )`，j=0..31；S中心(0,0,.17)。v65仅再加不同ID的面内点(0,0,.1) | v64满维且g=.02；v65是原 convex_vertex_capacity，不是求解近似失败。原65点不得先简化成64救过门。 |
| `origin` / `rebase,loss` | rebase：Q01-SH(g=.02)整体加 `(2**20,-2**20,2**20)`；loss：原B已声明local点、origin=(2**60,0,0)而world FP64展开丢x半边 | rebase若同源相对坐标可表示则核原gap，否则明确HOST completion/precision unsupported；loss必须暴露source不可表示/实际展开非满维。若接口只获world点，不能凭空“恢复”已丢local，必须保输入表示边界。 |

## 7. Q08：小规模与失败记录，先不放40步窗口

本 K0 不复制原提案 Q08 的接触40+21步：那属于 K2 真实消费者。K0 先做静态重复query与一次 accepted-state 换代，足以区分 device 窄相/材料记录/借用状态问题。

规模声明复用已接受的 ordinary fixed-root 接口，新增以下尚未执行的三P森林输入：parents=(-1,-1,-1)，轴全+z，m各1 kg，COM0，I各diag(1/16)，初v0，重力/drive/PD0，三独立组0/1/2对应S/C/H。q 即下面z中心，非邻接动态对显式启用原同tree域；不新增跨owner求解。world序号j的真实键 `17+37*j`，物理body=11/29/71、shape=3/5/7，材料ID分别 `100+i/200+i/300+i`，跨world故意复用局部ID。

| 身份 / 数量 | 原输入和预期 | 调用量 |
|---|---|---|
| `K0-Q08-w{1,4,32}-sparse`（3） | q=(0,.5,1)；SC/SH/CH gaps=.4/.85/.35；严格宽相应零候选 | 各同一accepted状态query2次，0步。 |
| `K0-Q08-w{1,4,32}-dense`（3） | q=(0,.09,.23)；SC/SH/CH gaps=-.01/.08/-.01；每world实际候选SC/CH两对 | 各query2次，0步；P域=3W、返回2W，不分配跨world全局配对。W32=96shape/416材料点。 |
| `filters-{samebody,mask,excluded}`（3） | W4 dense原几何，仅world54的SC被原过滤规则移除，其他world不变；精确声明见下文 | 每身份query1次，0步，实际7对；invalid-source仍按原预检，不用过滤逃过invalid。 |
| `epoch_bank`（1） | W1 dense，仅S初v_z=1/32，h=1/1024；一次无载advance后S位移1/32768，SC gap=-.01-1/32768，CH不变 | query初态→advance1→旧借用拒绝→新query→restore初态→旧借用再拒绝→新query；共5 query尝试、1真实步/1restore。 |
| `pair_capacity`（1） | W1 dense实际2候选；先合法capacity2，capacity1拒绝/显式overflow且required=2，足额新计划retry | 共3 query尝试，0步；所有原状态/已返回owned记录保留，无截断成功。 |
| `budget_one`（1） | Q03-tilt-HH、g=-.01，拟查询预算1 | 1 query；只有真正完成且独立检查过才resolved，否则必须unresolved/budget；不能假定某未来算法恰需几轮，也不能凭预算耗尽标分离。 |
| `fault-{raw_nan,raw_bad_id,late_record}`（3） | W1 dense，真实device返回后分别破坏一条raw有限值/材料ID；或原记录构造后注入一次分配失败 | 每身份先合法query、一次拒绝、去注入同请求retry，共3次；保原/破坏数组与原字，不手工异常代替验证。 |
| `owner_close_rebuild`（1） | 两份同键同坐标独立batch；plan跨owner调用、关闭plan/owner后调用、原声明新建计划 | 先合法1、跨owner拒绝1、close后拒绝1、fresh成功1，共4尝试；未使用closed资源，live bytes归0，累计分配另存。 |

三个过滤身份都只改world54：samebody将C键改为(54,11,5)、绑定group0并把原C局部轴整体加z=.09，保持实际world几何，原group1保留但不带该shape；mask用S(group=1,mask=1)、C(2,2)、H(3,3)；excluded仅列该world原SC canonical pair。三者均只有CH留下，其他三个world各SC/CH两对；不能用同body标签伪装两个不同物理group。

规模共12次query；过滤3；换代5；容量3；低budget1；故障9；owner/close4，共37次query尝试。公开API/故障接缝尚未冻结，父需将这些语义映射到实际可调用边界，不能测试伪造private owner来代替跨实例入口。分配中途清理另作1次构造失败尝试，保资源/异常，无query/物理步；未发明新runtime。W32排在W1/W4后，设备资源不足时保该档失败并停止该档，不调小其它原输入冒完成。

## 8. 待父裁定的 FP32 raw 工程门

以下是**测试前候选值，不是已测误差或严格证明**。独立真值首先来自原FP64声明；另用相同公式在“实际stored/上传输入”上给第二个预期，分开列输入量化误差和算法误差，不能只对量化后的错误模型报通过。共同世界平移不进入局部尺度。

令 `u=2**-23`，`L=max(1 m, diam(A), diam(B), |originB-originA|)`，L按真实有限shape而非world绝对坐标。建议有效raw域先限定本表尺寸和相对距离，不把此门无界外推。

| 项 | 候选工程门 / 分类 |
|---|---|
| gap | `abs(g_raw-g_oracle) <= 1e-4*L`，不加不同单位的任意rtol；输入量化/算法两项均单列。 |
| 表面点/材料重建 | 唯一见证或到独立合法最近集合距离 `<=2e-4*L`；两侧surface重建残差同门。并列点不与任意另一点作坐标等值。 |
| 法向 | 唯一非退化方向时 `norm(n_raw-n_oracle)<=2e-3`，`abs(norm(n_raw)-1)<=2e-4`；无唯一方向只核状态，不任意造单位normal。 |
| closure / 权重 | `norm((pB-pA)-g_raw*n_raw)<=2e-4*L`；raw权重finite、非负、和误差<=64u，ID/owner/索引必须精确，不能以浮点容差救错ID。 |
| 符号 | `abs(g_oracle)>2e-4*L` 才要求raw明确正/负；带内保原raw及uncertainty，允许unresolved/HOST completion，不批准无余量的strict separated。 |
| 资源/身份 | 全部数目、owner/world/source/revision/epoch/bank/tick、used<=budget、容量与错误类别精确；无浮点放宽。 |

依据：pinned `simplex_solver.py:337–363` 原默认相对duality-gap/绝对距离阈值1e-4，`mpr.py:195–196` 为1e-5；`simplex_solver.py:41–43` cutoff另用64u，且 `collision_convex.py:40–43` 明确它非证明。这里只借其数量级设工程候选，不抄MPR inflate、deadband或改变本项目物理gap。若父选择别值，须在首RED/测量前另冻结，不能从后续失败倒调。

旧公开 `ContactSite` 的1e-10几何构造、1e-12权重和门保持；raw不能简单cast后冒通过。HOST按同一已选feature、原FP64 source及actual stored pose构造记录，原输入不QR、不吸附、不重新选最近面。需要新搜索则记host_completed；无法按原门构造则unresolved。K0近似gap不直接驱动旧law，因此此raw门不是新的力/稳定性容差。

## 9. 保存、最小交付与止损

建议每个identity一份完整JSON/gzip，固定文件stem即上述ID；所有before/after/source和原数组字节保留，不按成功挑样。Q01/Q02/Q03/Q06-band分别30/9/50/12，共101个纯query身份；Q06-domain为16个声明尝试；Q08为16个协议身份（37 query尝试）加1个半构造失败。数字是单套逻辑身份/每backend的作者清单，尚未collection/执行；父冻结时须逐项确认，不从运行产物倒推membership。

每记录至少含：声明及生成公式/显式ID映射；完整source/blob与pinned源码SHA；world/body/shape和完整有限顶点/skin；same-owner的accepted epoch/bank/tick、实际pose、candidate原位集或确实HOST重传的表；raw dtype/shape/全字节、status/iterations/budget/capacity/required；两个material地址/weights/offset与surface/gap/normal；独立formula/actual/expected/门/逐项结果；device_resolved/host_completed/unresolved原因；完整公共结果、异常和同请求retry；所有H2D/D2H实际array.nbytes/次数、显式sync、真实launch、累计allocation/live bytes。padding与逻辑行分开，不拿HOST dataclass算设备buffer。

按“SS/CH非零4首query→Q01非零20→有限/变换→边界/事务→W32”推进；首4是Q01固定身份的执行分段，不多算或重跑。HOST oracle与interpret先核构造；真CUDA才可验device路径。K0不必完成K1面积、归一权重、总力/矩/功或40步接触窗口；也不得据K0通过声称这些已成立。

| 困难 | 明确止损/后继 |
|---|---|
| 同一有限正域反复预算失败 | 保存最小原输入/raw/used budget；不无限加迭代、调gap或扩物理门。可交部分有名支持域，但整族K0仍未完成。 |
| 只有HOST completion通过 | 可交协议/诊断，不能签device窄相功能或收益；先解决该原shape/feature，不加K1掩盖。 |
| 并列/深包含/近零无唯一方向 | 记录距离/overlap已知与normal未决的区别，沿原明确边界；不强造载荷法向。 |
| raw工程门通过、旧材料构造失败 | 保两层结果；不降低legacy门、不用重新搜索却标device完成，转明确HOST completion或拒绝。 |
| 成本不合适 | 先保实际W1/W4功能与资源数据，W32后置；不重复旧分页/profile矩阵，不在本清单承诺速度。未来同完整输出AB/BA/AB另授。 |

父负责原collision/batch/candidate借用接缝及API冻结；后继作者只在另授NEW query/support/test范围实现。此提交只有本文件，**CASE_MANIFEST_PROPOSED_UNEXECUTED / AUTHORING_RELEASED**。
