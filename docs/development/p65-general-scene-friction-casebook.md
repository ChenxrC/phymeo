# 通用场景持续切向接触：案例册

2026-11-07，DOCS_RESEARCH_ONLY，全部新增案例 **UNEXECUTED**。研究源为父产品 `fdfa586`；本文件在独立树 `c8551e3` 上新增，不合并父树、不改产品或测试。本册冻结下一次 tests-first 的候选输入和工程门，最终实施范围由父计划裁定；下文的预期不是新实测结果。

## 1. 当前事实与首次可调用能力

`MultiRobotScene.enable_coupled_tangent()` 目前在 `src/phymeo/runtime/multi_robot_scene.py:302–307` 明确拒绝。既有单机器人桥 `runtime/_scene_coupled_tangent.py:54–60` 只接受普通 prismatic；该桥的 robot 点 Jacobian 在 `:173–185` 只有线速度项，符合旧域，不能移除 guard 后声称支持 R、mimic 或浮根。新 F2 的浮根 affine mimic 场景已经有真实 full 接触/约束积分，但尚未接入此持续切向模式。

首次能力拟沿现有公开方式调用：先 `ContactSceneBuilder` 声明两个独立 P 工具和一个自由箱，再 `scene.enable_coupled_tangent(registry, stiffness_n_per_m=200)`，最后同一 `scene.step(...)` 同时接受真实法向、全部跨 owner 切向响应、物理状态和历史。不是事件观察器自动识别静摩擦，也不让用户手工逐站提交多个物理步。第一个 RED 应在真实法向正控后触发上述公开拒绝。

已有证据只引用：`docs/benchmarks/p65-scene-tangent-batch4.md:19–25` 的 24 站点压载和旧纯 P 双 pad；不重跑同源旧整组。原普通 ContactScene 的 `NormalLayer` 和机器人 `RobotContactMaterial` 保留不同法则/开距约定；本册的新机器人案例统一采用后者。

## 2. 公共输入、记录与工程门

除 F07 明列的原值外：SI 单位，HOST FP64，重力零，初速零，机器人 `kp=kd=0`，驱动通过真实 feedforward，未列 pair 显式排除。所有正质量 link 的 COM 在其局部原点；未专列时惯量为 `diag(.01,.015,.02) kg·m²`。P 关节范围 ±.1 m、速度上限 8 m/s、驱动力帽 8 N。固定支撑无规定运动。机器人法则为 `RobotContactMaterial(2048, 0, .5, .01)`：真实穿深 `δ=1/1024 m` 对应整 patch 法向 2 N；各 site 由原几何/原权重分配，不给每个 site 再乘或再发一个 2 N。

碰撞key通过 scene.shape_key/已命名collider实际取得，不把物理body ID冒作重映射后的CollisionKey.body。显式登记每个实际站点的稳定 `StationKey(world, token)`、双方 shape/owner、局部材料点、局部切轴及有限 chart。token 按事先声明的采样点命名，不按当前 row 排序。盒面 chart 预声明完整实际面 vertex IDs；稀疏 barycentric 支撑变化不自动等于换面，也不证明任意连续面。正载荷 site 缺漏、歧义或超界必须拒绝。真实 `site.force_n` 是库仑半径来源。

| 检查 | 冻结门与含义 |
|---|---|
| 首步独立代数 | 位置/速度绝对误差 ≤2e−11 m 或 m/s；力/完整广义力 ≤1e−9 N 或 Nm；使用下列独立质量/J 构造，不以产品 M/b 输出互证 |
| 实际运动/功 | 保留原 `motion_tolerance_m=1e−6`、`energy_tolerance_j=1e−8`、`max_frame_turn_rad=.25`；材料和原作用点位移均核，不增门以容纳旋转误差 |
| 每步平衡 | 同一入步载荷推动真实半隐式状态；线动量余额 ≤1e−9 N·s；固定根反力矩另记。转动例的离散角动量误差不冒充精确守恒 |
| 力律 | 每站 `|f_t|≤μN+1e−9 N`；零 μ 无切力/残余储能；实际功、储能差、塑性损失、隐式耗散和释放能按原 receipt 独立相加，逐站与总量均保留 |
| h/h2 工程门 | 指定转动窗口末位移差 ≤1e−5 m、角度差 ≤1e−5 rad、速度差 ≤1e−4 m/s、角速差 ≤1e−4 rad/s，累计接触冲量/角冲量差各 ≤1e−5 N·s / N·m·s；是候选接受门，未达即报告，不改输入 |
| 事务/归属 | 拒步前后 state/history/events/geometry/time/last observation 全部原样；同输入重试与从同 base 的干净单步一致；非连续 world/body ID 与注册反序不改变物理值 |

这里的功审计是切向本构/离散运动一致性，不等于整个场景总能量守恒；有限穿深的力矩、关节约束、重力和法向阻尼分别入账。128 个 active/曾加载 station token 的原容量及原迭代上限保留，不无限追加 tombstone 或静默驱逐旧历史。

## 3. F01：两个普通 P 工具，共同推动自由压载箱

- world=23；自由箱 body=91、质量 1 kg、半尺寸 (.1,.1,.05) m、中心 (0,0,0)，惯量按均匀箱 `m/3·diag(b²+c²,a²+c²,a²+b²)`。
- 两个非 Panda 固定根单 P 机构 body=17/42，各 moving link 质量 2 kg；P 轴 +X。局部球形工具半径 `r=1/16 m`，零位球心分别 `(0,0,−(.05+r−δ))`、`(0,0,.05+r−δ)`。采用现有资产 collider 声明方式，不假称 URDF 有未支持的 capsule 标签。
- 两工具分别与箱启用上述 law，工具间排除。每对一个真实球–有限盒面站点；锚定工具局部球面 ±rZ，箱侧显式上/下面 chart；两原作用点保留。两个法向各 2 N，箱净法向和初始合矩为零。
- h=1/1024 s，128 步，箱外力前 64 步 +.5 X N、后 64 步 −.5 X N；两驱动为零。不把末态写成静态精确平衡。低摩擦留出 μ=0；超载留出外力固定 +3 X N、同 128 步。

独立首步：令 f 为每个工具作用于箱的 X 切力。两站共同作用点距箱 COM 的 Z 杠杆为 `ℓ=.05−δ/2`，`Iyy=1/240 kg·m²`；完整 X 切向子块是 `W=[[3/2+a,1−a],[1−a,3/2+a]] kg⁻¹`，`a=ℓ²/Iyy`。不能丢掉自由箱的转动交叉项；对称载荷恰好激发零净矩子空间。在未达帽时 `f=−kt·h²F/(1+(5/2)kt h²)`；箱加速度 `F+2f`、每工具加速度 `−f/2`。总线动量增量为 `hF`；对称点的箱合矩为零，工具约束承担偏心反矩；由全体动量/外载核算支撑反力，不把仅记录直接根接触的 fixed_root_contact_wrench 当总支撑反力。μ=0 时箱有解析 `x_n=F h² n(n+1)/2`（恒外力段），工具 X 不动。

超载的长期共同加速所需箱切力为 `4F/5`；F=3 时为 2.4 N，大于两站合帽 2 N。这说明不能无限黏着，不预报某个未经测量的首滑时刻；该有限窗应记录是否已滑、储能和相对位移，不能靠稳态代数伪造窗口结果。

最小先跑 1 步/8 步，再完整三个 128 步窗口。另做循环轴变换 `(x,y,z)→(z,x,y)` 加平移 (.3,−.2,.4) m 和注册反序；位姿、力/矩、world惯量、重力和外力按同一变换协变；局部材料锚点及body-local质量/惯量本体不改。

## 4. F02：有限杆–非盒凸台，两个材料站点

单 P 杆 body=17，轴 +X，质量 2 kg；局部 capsule 轴端 `(0,±1/8,0)` m、半径 1/32 m，零位中心 z=`1/32−δ`。静态凸台 body=91 的底面 z=−.1 m、半宽 (.3,.2)，顶面 z=0、半宽 (.25,.15)；八顶点、六个凸四边形外表面采用预声明合法三角化，明确不是盒代理。通过已有 asset/collider 包装声明 capsule，台由 `add_convex` 声明。

有限杆身在顶面内部，几何 patch 为整段，两端各 1/2 法向权重；总 N=2 N，两站 N=1 N。站点锚定杆局部端点下方半径偏置，台侧完整顶面 chart；kt=200/站。h=1/1024，64 步，驱动 +.5 N；超载 +4 N/μ=.5 与零 μ 各同窗。K=400 N/m，首步总切力 `F_t=−K h² τ/(2+K h²)`，未饱和时杆加速度 `(τ+F_t)/2`，两端切力相等。持续正控要求两个真实站点都保留、无虚构一点近似；如果实际 adapter 给出不同合法离散，先报协议差异，不从结果反算新 kt。

边界负控：将杆中心 y 置为 .2 m，使接触区被台边截断；原端点登记不再完整，正载荷未匹配时必须拒步并保原态。另一完全移出 y=.4 m 应是零接触、一次释放，不能把“边界裁切拒绝”混成“分离”。该批不自动搬运端点 token。

## 5. F03：fixed affine mimic 与共享祖先的完整交叉响应

两棵非 Panda 树：A 有两个 +X 平行 P 子体，质量 1/2 kg，`q₂=2q₁`、偏置 0，工具为 r=1/16 的两个球，中心 y=±1/8、z=`r−δ`；B 是质量 3 kg 的 +X 单 P 台面，半尺寸 (.3,.3,.05)，中心 z=−.05。A 的根固定；两球只与 B 接触，其他碰撞排除，两个实际站点共享 B 的祖先响应。A 独立驱动零，B 驱动 +1 N；全状态零，h=1/1024、64 步。

独立约束投影 `P=(1,2)^T`，`H_A=Pᵀ diag(1,2)P=9 kg`，B 为 3 kg。以 B 相对 A 两球的 +X 位移为站点坐标，

`W=[[4/9,5/9],[5/9,7/9]] kg⁻¹`，`d_free=(h²/3)(1,1)`。

首步解 `(I/kt+h²W) f=−d_free`，未饱和时 `a_B=(1+f₁+f₂)/3`、`ü_A=−(f₁+2f₂)/9`。所有接触仍 full physical 维数；原 mimic 行反力由独立鞍点方程核，`Pᵀ constraint=0`，约束行功合为零，不把 full 力直接当 independent 驱动力。法向和偏心矩由固定根实际承担并记录。

留出为 A 注册反序/B 非连续 ID；同样数值复制另一 world 的独立 scene，确保不同世界的 history 不继承。低 effort cap .25 N 的 B 保真实限幅；feedforward=None 与显式 0 按旧 PD 语义另做短节点，不重写控制器。完整矩阵首步有决定力，不以整棵树的产品 M/b 当独立 oracle。

## 6. F04：ordinary floating / affine floating mimic 的接触反作用

采用已冻结 `tests/fixtures/urdf/floating_mimic_jaws.urdf` 的 coaxial root2/child1/2 kg 物理树、工具和惯量。正控 A 用现有 affine `q₂=−q₁+.004`；留出 B 使用同一物理声明但不含 mimic 的普通浮根资产，不能拿 wrapper 的 reduced 接触维数替代完整树。

只在 right_jaw 上下两有限面与同一普通 +X P 夹持台接触：台 moving mass3 kg，带两个对称 box collider，半尺寸 (.1,.1,.02) m；中心 x=.254、z=±(1/16+.02−δ) m，right_pad 原半厚为 1/16 m。普通浮根留出的初始 full joints 明确为 (0,.004)，与 affine 零独立坐标的初始几何相同。每面真实总 N=2 N。台驱动零；浮根 A 独立驱动 +.1 N，B 完整 joint effort=(0,.1) N；所有重力/外载零。每个面对称登记实际面积采样点，kt=200/站；沿现 generic robot–robot patch 的矩形四顶点/每顶点1/4权重，共八站点。必须先核原法向和材料地址；若公开离散不同须先报协议差异，不能通过增减站点调刚度。

A 的 X 子空间 `H=[[5,−1],[-1,3]]`，right_jaw 点行 `j=(1,−1)`，对称合力作用于该 body COM 方向的 mobility 为 `3/7 kg⁻¹`；加台的 `1/3`，对称平移合力子空间的相对 mobility 为 `16/21`。每个偏心 site 的完整 W 仍须包含实际角 J/全惯量及交叉项，不能将全矩阵填成这个标量。驱动 .1 N 时 right_jaw 自由加速度为 `−1/35 m/s²`。普通 B 的 full X 质量矩阵为 `[[5,1,2],[1,1,0],[2,0,2]]`，right_jaw 行 `(1,0,1)`，对应对称合力子空间的 mobility `1/2`；与台合为 `5/6`，自由 right 加速度 `.05 m/s²`。两种系统分别验收，不把不同驱动/约束说成相同动力学。

h=1/1024、64 步各一窗；先 1 步核矩阵和双侧冲量，再长窗。面上对称法向/切向避免刻意引入未验证大转动；仍记录完整 root COM/角速、各 joint、所有 site 和作用矩。mimic 的完整根六维约束残差应为零；内驱动和内接触不能凭空改变整个自由系统的总线动量。若实际 patch 稀疏支撑改变，用声明 chart 校验而不是更新 token。

## 7. F05：转动工具与真实材料运动门

固定根单 R 机构，轴 +Z、moving mass1 kg、COM 在轴上、惯量 `diag(.025,.025,.02) kg·m²`。有限 box 半尺寸 (.125,.0625,.03125) m，轴原点/COM z=`.03125−δ`；固定台半尺寸 (.5,.5,.05)，中心 z=−.05。R 范围 ±.25 rad、速度帽 1 rad/s、力矩帽 .05 Nm；请求恒 .01 Nm，零初速，原 law、kt 不变。选 box 下表面材料 chart 和局部面积站点；台 chart 只允许其有限面内漂移。

先 1 步：以已声明局部站点 r_i、原共同作用点和 `J_i=Z×r_i`，`Kθ=Σkt|J_i|²`，未滑时线性离散预测 `Δθ=h²τ/(Izz+h²Kθ)`、`f_i=−kt J_i Δθ`。端点真 FK 圆弧必须再通过原 material/application 位移与功门；不能用此线性式覆盖实际末姿态。随后 T=1/32 s，h=1/4096 与 h/2，共128/256步，按 §2 比较完整角冲量和末态。

这是法向恒定、材料切轴实际旋转的第一正域，**不是任意变法向滚动已解决**。共同转动留出把固定台换成第二个同尺寸/同质量/同惯量的 R box，中心 z=−.03125、轴同为+Z；两个 q=0、角速均 .2 rad/s、驱动均0，跑8个原h。该零历史留出应无相对材料滑移、无切力/虚假弹性功；不声称预载剪切历史在任意折角处自动守恒。

负控：改轴为 +Y、初速 5 rad/s、h=1/128，仅一个候选步；该负控速度帽明确为8 rad/s，位置范围仍±.25 rad，先核输入合法，再验实际motion/work拒绝，不能让限速检查代替运动反例。另设支持点实际进入不同有限面的边界。允许在材料配对或实际运动门明确拒绝；必须留下具体原因和所有旧发布数据，不删除 guard。若恒法向小转动也过不了原门，保留其失败，把首批产品范围停在平移，不调门或缩小每个案例直到全绿。

## 8. F06：卸载/重入、显式换面边界、拒步重试

两 P 串联工具：第一轴 +X、link mass1 kg；第二轴 +Z、link mass1 kg，带 box 半尺寸 (.25,.25,.05)，零位中心 z=−.05。固定球 r=1/8、中心 `(0,0,1/8−δ)`。h=1/64；Z 驱动三步依次 `(−254,+512,−254) N`，本例 Z 力帽显式为 600 N，范围 ±.1 m/速度帽8不变。X 驱动三步 `(.125,0,0) N`，其等效质量2 kg。法向指向工具 −Z，首/第三入步 N=2 N；Z 真实半隐式结果 `q_Z=(-1/16,0,0)`、`v_Z=(-4,4,0)`，第二入步分离；不复位或直接写状态。

锚定固定球底部材料点，移动 box 顶面为显式 chart。切向步1储能、步2释放、步3 generation 重入；释放能只记一次且新弹簧不继承旧预载。X 与 Z 质量/运动解耦；用法向标量递推和切向独立质量2 kg 验证。此为有限 3 步生命周期，不冒充长期持续接触。

失败注入分别置于：真实测力后 prepare、末态 motion gate、owned observation、events stage。每种失败只重试同一输入一次；前后包括 participant/full mimic state、rigid state、history/generation、events、time、geometry 和 receipt。另测站点缺漏/重复/错误 world/owner、128 个已有 token 后第129个、非法 full-contact 维数及读取返回数组后外部篡改。

换面先冻结保守合同：有限面裁切/另一面正载荷不能用旧 token 静默承接；未证明同 chart 的连续性时拒步，原储能不丢失。之后若授权显式 rebind，应由用户提供新材料地址，并在同一物理提交记 `ended(reason)` 和释放能、冷启动新 generation；不是本首切片的自动最近站点匹配。

## 9. F07：原 100 g mimic 箱失败回归，不替代上述多案例

原输入直接引用 `examples/robot_force_acquisition.py:74–160,268–373` 和 `tests/reference_newton/robot/mimic_jaws.urdf`：40 mm 箱质量 .1 kg、另一个 .03 kg 球、原有限桌、原 full 几何；pad 各四点 ±5 mm；robot–box 法则 1200 N/m、1.2 Ns/m、μ=.65、滑速正则 .04 m/s；桌/其他 pair 各自原法则保留。kp=(2700,6000)、kd=(95,35)、帽=(40,8)，原闭合/20 mm提升/释放控制和 h=.001 s 不动。

新候选仅显式启用选定 pad pair 的持续切向、kt=200 N/m/站；不改原质量、真实 N、驱动、控制时钟或门。原成功门仍是双侧 ≥1 N、离台 ≥10 mm、台反力 <.1mg、相对位移 ≤2 mm 连续100 ms；原 >3 mm失持和实际释放门保留。零摩擦、1 kg超载为原独立负控，不能只展示100g。

历史事实：`docs/benchmarks/p65-force-events-filtered-batch2.md:24–28` 和 `p65-gripper-acquisition.md:47` 记录100g案例1807步/1.807s、获得夹持但未 held、真实释放，7/8 RAW1；1.048s相对滑移3.005689 mm、离台3.499783 mm、桌反力0。速度正则法则会产生持载滑速是可解释机制，**不是该失败已被隔离证明的唯一原因**。新模式不预写通过；若仍失败，保存失持/关节/材料/反力证据，停止调 kt/夹持阈值，不重放同一轨迹寻优。

原预算上界由例程的 `ceil((1.5+.6+.1+1.2+.1)/h)+3` 给出；先做三种输入各一个原规则窗口，事件到达可提前结束。不能把新旧不同摩擦律结果当同输出性能比较。

## 10. 执行成本与交付顺序（均待授权）

先 F01 首步/8步和 F03/F04 各首步：预估小于20秒只是调度预算，未测。再 F01/F02/F03/F04 的限定64/128步窗、F05的384步与短负控、F06事务。已有 F2 T3 四窗2560步用17.83秒仅是此前不同法则/站点的参考，不能外推本核耗时。父在首个64步真实收据后决定排长窗；每个命令>2分钟先报真实成本，不新增隐藏重复或扩大取样。

最后才跑 F07 三个原控制窗口；先报首窗实耗/预计余量，任何一个机制性失败都保留。新增例程一次保存每步完整状态/原法向载荷/切向前后载荷/材料地址/末态位移/历史/receipt；无损压缩 sidecar，索引标源 SHA、输入、步数、law、HOST 责任。禁止用摘要代替完整验收输出；无性能排名或 Newton 速度比。本册所有数字门和输入须在父授权 tests-first 时转成固定测试，再执行真实 RED/GREEN。
## 11. T1 authoring 交接：公开接口与完整记录（UNEXECUTED）

2026-11-07。父已将 tests-first `4a81571` 导入 `3079f39`：`151-friction-first-red` 为 **1 failed / .55 s / RAW1**。既有两边各2 N、共同作用点、零净力矩和查询不推进的前提全部通过，唯一失败为 `MultiRobotScene.enable_coupled_tangent` 的原明确拒绝。此 RED 不重跑。以下增量仅文本/Git；尚未执行、收集、格式化或作数值探针，未写产品。F05仍是工程候选域而非已验成功。

### 11.1 声明和公开调用

`examples/general_scene_friction.py` 只建模、声明原输入和保存完整读数，不含 step 循环或状态覆写。返回 `FrictionCase(scene, registry, actions, world, parameters)`，可用 context manager 关闭；公共调用不增加另一套记录类型：

```python
case = make_two_paddles(directory)
case.scene.enable_coupled_tangent(case.registry, stiffness_n_per_m=200)
external = case.scene.inputs()
external.rigid.force[case.scene.rigid_rows[case.world, BOX], 0] = .5
reading = case.scene.step(1 / 1024, actions=case.actions, external=external)
```

- `make_two_paddles(directory, *, world=23, friction=.5, reverse=False, transformed=False, selected_tools=None, box_velocity_x=0., separated=False)`：原五个128步窗口。混合律短留出仅选LOWER、箱初速X=1/64，未选UPPER保原速度律−1 N；empty留出两个初始几何间隙+.01 m，仍由+.5 N实际推动箱。
- `make_rod(directory, *, friction=.5, center_y=0., drive=.5, world=23)`：原有限杆三种64步窗及y=.2/.4边界。
- `make_fixed_mimic_pair(directory, *, reverse=False, world=23, partner_cap=8., gravity=(0,0,0))`：原三个64步窗；额外PD语义短留出明定重力(+1,0,0) m/s²，A请求feedforward=None，B分别None/显式0。A/B重力补偿驱动独立为−5/−3 N；B显式0时自由加速度+1，接触首步仍用原2×2 W，`(I/200+h²W)f=−h²(1,1)`。这仅冻结原§5的None/0区分，不改原窗口重力/驱动。
- `make_floating_clamp(directory, *, fixture_dir, mimic=True, reverse=False, world=23)`：原3个64步窗；fixture_dir指仓库既有URDF目录，仿射/普通资产保持实际同初始几何。
- `make_rotating_tool(directory, *, common=False, tilted=False, world=23)`：原128/256/8步和合法快速倾倒拒绝；没有为通过而调小实际角速度或放宽运动门。
- `make_withdrawal(directory, *, world=23)`、`withdrawal_actions(index)`：原三步预先给定电机力，不重置任何状态。

`capture_frame(case, entry, reading)` 保存入步全状态、真实返回reading、当前几何、history、receipt、contact_events。`write_trace(path, *, case, frames, failure=None, actual_dt_s=None)` 用exclusive新文件+无损gzip保存上述完整公共字段、registry/actions和参数；半步窗显式传实际dt，不把工厂声明h误写为h/2。events的公开generation tombstone也保存；只有私有owner/seal排除，完整数组保dtype/shape/全部元素。失败保已接受完整前缀与实际错误；物性/初态由同源声明与父源码快照共同绑定。示例不私自执行Git或生成哈希来冒完整输出。

### 11.2 已写新专项及容量边界

两个测试文件的所有节点都以真实公共场景取法向，完整3N×3N W通过包裹原 `_coupled_tangent_step._validate_world_mobility` 只读截取；未扩公开receipt，不用产品M/J作为预期。浮根独立M由已声明三质量/COM/惯量的动能组装，角J由真实共同点独立构造；F05实际位移由末角的sin/cos圆弧独立重建。

| 文件/节点组 | 冻结验证 |
|---|---|
| `test_multi_robot_tangent.py::test_a01...a08` | 有效RED、完整跨ownerW/首步、128步低μ/超载/反向、注册序/刚体协变、selected+legacy、empty、四处拒步重试、跨world/owned读数 |
| 同文件 `test_b01...b03` | 有限杆两site权重一次、三64步窗、有限裁切/分离、缺漏/错锚点 |
| 同文件 `test_c01...c02` | fixed mimic full/independent维数、W交叉项、原行反力/功、三个64步窗与帽 |
| 同文件 `test_d01...` | 真实2N→0→2N、弹簧释放一次、generation重入 |
| 同文件 `test_e01...e03` | 同锚点不同token歧义、未注册shape、full2维错作independent1维时拒绝与重试、None/0和重力补偿 |
| `test_floating_rotating_tangent.py::test_a01...a03` | ordinary/affine浮根首步全W/根反作用、三64步窗、同pair多site组件合并/逆序后物理与身份不丢 |
| 同文件 `test_b01...b04` | 真角J/首步、h/h2实际弧位移/冲量/功、共同转动零相对功、快速倾倒拒步保全旧发布 |

当前静态registry最多128个声明，不能以重新给同一scene注册或私改store伪造第129个动态载荷。本批测试129个独立声明在构造时拒绝；“128已用token再加载第129个”的底层封顶语义复用B3既有证据，未列成本批新增绿例。跨world/不存在shape的registry应在enable拒绝，不得因永远采不到pair而静默变成成功空观测。输入维数失败注入只替换当次测得的full向量，不改实际normal/几何，随后恢复原采样并核同base重试。

完整窗口合计预声明 **1611物理步**（F01 640、F02 192、F03 192、F04 192、F05 392、F06 3），另有短首步/故障/重试节点；这不是已跑步数或时间估计。父先真实首步和短节点后取得成本，再安排完整窗。每个共享module fixture仅推进一次并供相关断言复用；不得重复执行已绿同源旧物理整组。

### 11.3 给父的旧接缝合同提案（待产品grant裁定）

公开仍为 `enable_coupled_tangent(registry, *, stiffness_n_per_m, motion_tolerance_m=1e-6, energy_tolerance_j=1e-8, max_frame_turn_rad=.25)`、`.tangent_history`、`.tangent_receipt`、原`.step`。不改`SceneContactEvents`/`RobotSceneObservation`/`LoadSite` schema，不把配置切换写进物理载荷。只允许在首步前配置一次；getters/sample/geometry不创建或提交episode。

NEW产品候选仍仅 `_scene_tangent_ports.py`（readonly owner响应/姿态映射）和 `_multi_robot_tangent.py`（纯候选编排及site散回），本次均未写。建议后者使用以下最小父hook签名；父裁定后再落源码：

```python
configure(scene, registry, *, stiffness_n_per_m,
          motion_tolerance_m=1e-6, energy_tolerance_j=1e-8,
          max_frame_turn_rad=.25)
prepare(scene, prepared, measured, loads, details, external, h)
# -> attempt.measured / attempt.loads / attempt.details
attempt.finish(candidate_body, end_geometry_by_address)
# -> (candidate_history, candidate_receipt), still unpublished
```

`prepared` 的key是实际 `(world, registered_robot_body)`。Multi直接使用现有一次`participant.prepare`结果；普通/fixed mimic条目已有`entry/physical/geometry/linear/angular/full_mass/full_bias/mass/bias/effort`；floating条目的`full_mass/full_bias`是6+n，独立projection从真实asset取得，prepared.effort只有k个motor，不能把它当6+k或full驱动。其已冻结`external`是完整body COM力/矩，仍单独计一次。新helper不得另调用participant.prepare/integrate。

`measured`原值是各owner的`AssembledContacts`；`loads`为同入步聚合`SurfaceLoads`（包括direct robot–robot，不能只取measured[addr].loads）；`details`仍逐原site完整返回。输出保`joint_effort_full`/每body COM wrench/直接rootcontact wrench原义。固定root材料有姿态而响应为0；一个浮根mechanism只占一个full响应块，不给root/各child新建动力学owner。freebody由scene._rigid_model和scene._body的world惯量/重力/原外载处理。

单`RobotScene`父可将已算好的 `arm/physical/geo/linear/angular/mass/bias/M/b/effort` 包成同一既有`RobotPreparation`的只读描述，映射到`{scene._addr: ...}`；不重新解析PD、不构建第二participant。finish的几何为实际候选full FK/COM姿态（fixed mimic须用full_q，floating须保root authored frame），rigid为本次原积分器候选。不能传旧geo或候选末站点最近重采样。

父 `MultiRobotScene.step` 的接点是原 `_sample(terms)` 后、各原integrate前替换三份当次load；finish放候选几何形成后、`_owned(reading)`和events最终成功发布前。候选history/receipt与原7项状态同一tuple assign。Single对应原robot_prepare/finish调用位置；默认分支完全保留。任何中途输入、kernel、实际motion、owned、events失败均不改accepted state/history/time。父可改内部签名以合既有结构，但必须保本表的维数、一次力/一次推进和公共合同；禁止删新断言迁就旧桥。
