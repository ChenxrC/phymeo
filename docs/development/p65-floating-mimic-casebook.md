# F2 浮根仿射 mimic：tests-first 案例册

状态：**UNEXECUTED**。本提交只有本页、一个测试文件及两份 URDF；作者未运行 Python、导入、collection、测试、数值探针、格式或构建。工作树基线为 `1c39eee4ac95c8320cd4c3b2251a33c5e357aa8b`，未 merge 父后续提交。这里固定输入和判据，不宣称新产品已存在。F1 固定工具文件保持原样。

## 本片接口和有限域

拟公开 `load_floating_mimic_urdf(path, *, gravity=(0,0,-9.81), collapse_fixed_joints=False)`，返回包装完整 `tree: FloatingRobotAsset` 的新资产。`tree` 的 joint/body 顺序始终是完整物理树；包装的 `joint_names`、位置/速度/驱动上限及命令按独立轴。资产提供 `matrix`、`offset_vector`、`constraint_rows`、`independent_indices`、`full_state`、`reduce_state` 和 `step`。

- 独立状态为 `q[7+k] / v[6+k]`，完整状态为 `q[7+n] / v[6+n]`。q 的前三项是原始根原点；v 的前三项是**根物体 COM** 世界速度，其后三项是世界角速度。不是全系统 COM。
- 关节满足 `q_full=P u+c`，速度映射 `S=diag(I6,P)`。完整 M/bias/J 必须来自真实全部质量，再作约束投影。保留原根 COM 漂移和四元数一次推进规则；不得先无约束推进再覆盖从动坐标。
- `asset.step(state, *, effort, dt_s, external=None)` 返回 `state/full_state/loads`。effort 只含独立电机；external 是覆盖所有物理 body 的 `FloatingRobotInputs`，其完整 joint effort 必须全零，COM 世界力和矩原样保留。
- loads 的 `acceleration_full / actuator_effort_full / contact_effort_full / constraint_effort_full` 均为 **6+n**。根前三项是 N 或 m/s²，后三项是 N·m 或 rad/s²；后续移动/旋转关节分别用 N/m/s² 或 N·m/rad/s²。`contact_effort_full` 沿原命名记录除重力外全部非电机载荷，standalone 的外部 COM 力/矩也在其中，不能把它误写成只含碰撞。
- `constraint_effort_full[:6]` 必须为零，并另核完整力平衡的根残差；`row_forces` 保留原始行的乘子、两轴贡献和功，总约束功另报。驱动只作用独立物理轴，不能把 P 倍后的驱动当作给从动轴另装电机。
- 后续 scene 的 `velocity_count` 仍为完整 `6+n`，接触和材料 owner 使用完整 body/J；独立命令维数另存。本提交没有 scene 适配或事务实现。

原图机制不复制：后续 NEW `_mimic_coordinates.py` 的纯 `compile_mimic_coordinates(names, types, lower, upper, velocity, effort, relations_tuple)` 接原始 tuple，产生不可变映射、图顺序和联合限位。旧 `MimicRelation` 类保留；父负责让旧 `MimicRobotAsset` 委托及全部 shared hooks。本轮没有实现该 helper、loader 或 participant。

## F01：自由根、不等质量双移动轴

资产 `tests/fixtures/urdf/floating_mimic_jaws.urdf`，所有量 SI。根质量 2 kg，I=(1/5,3/10,2/5) kg·m²；左件 1 kg，I=(1/16,5/64,3/32)；右件 2 kg，I=(3/32,7/64,1/8)。各 COM 均在各自原点。两个关节原点分别为根坐标 (−1/4,0,0)、(+1/4,0,0)，轴均 +X；所有 COM/轴/原点共 X 线，因此本解析不会遗漏根转动耦合。

`q_f=−q_d+0.004 m`；驱动区间 ±0.032 m、速度 0.2 m/s、电机 1.5 N；从动区间 [−0.012,0.020] m、速度 0.08 m/s、原 effort 元数据 0.125 N。联合独立区间 [−0.016,0.016] m、速度 0.08 m/s；从动 effort 不是新增电机限幅。

冻结初态：根原点零、单位姿态、独立 q=0、全速度零、重力零、外部载荷零；独立电机 +1 N；h=1/1024 s。X 平动的独立惯量由原三质量直接给出 `[[5,-1],[-1,3]] kg`，因此根 a_x=1/14、独立 a=5/14、从动 a=−5/14 m/s²。完整原约束行 `[1,1]` 的乘子为 −4/7 N；原两关节约束力均 −4/7 N，根六维约束反力为零。

首节点 `test_a01_public_floating_mimic_preserves_mass_and_root_reaction` 先确认旧浮根 loader 明确拒绝 mimic，并由旧固定根 mimic loader 识别同一真实资产的 P/c，再请求新公开 loader。缺口只可能出现在新能力处；不会在 collection 导入一个不存在的模块。

持续留出为 16 个真实半隐式步，h 不变；n 步速度系数 `n*h`、位置系数 `n(n+1)h²/2`。独立检查三物体实际线动量为零，而非只查投影广义动量。另将全资产根姿态设为 q=(1/2,1/2,1/2,1/2)，同一原 +X 轴映射世界 +Y，输入和解析相应协变。这里没有接触、摩擦或持稳结论。

## F01-G：扇出、多级、联合边界与失败

测试只在临时目录从上述固定原输入生成 gauge：第三个质量 1/2 kg 的 +X 移动体，根关节原点 (1/2,0,0)，COM 为零，I=(0.02,0.03,0.04)，范围 [−0.003,0.007] m、速度 0.03 m/s。不是修改原 fixture。

- 扇出：gauge=0.5*driver+0.002；P=(1,−1,0.5)，c=(0,0.004,0.002)，联合区间 [−0.01,0.01] m。
- 多级：gauge=−0.5*follower+0.002；同 P，c=(0,0.004,0)，联合区间 [−0.006,0.014] m。
- 两者速度上限均 0.06 m/s，独立电机上限仍 1.5 N。初态独立 q=0、速度 0.02 m/s；独立电机 3/8 N；根/左/右/gauge 的世界 X 力依次为 (1/8,1/4,−3/8,1/2) N，矩零，h=1/1024 s。

测试从四个原质量独立装配完整 X 惯量与两条**原始约束行**，解未约化鞍点系统 `[[M,-Cᵀ],[C,0]]`，核每行乘子、完整反力、根反力零和总/逐行约束功，不用产品投影矩阵生成这些预期。负例包括后代位置/速度边界、错误完整位置/速度/维数、图环、零倍率、未知轴、异类轴、空交集、重复 mimic 标签。

standalone 拒绝测试包括电机 1.501 N、external 偷带从动 effort、错误完整 COM 行数；位置 q=0.015999 m、速度0.079 m/s 的末步越界也应拒绝。原输入不变，恢复合法输入后与 fresh 资产同输入重试逐数组一致；不把该纯候选检查称为完整 scene 多参与者回滚。

## F02：非平行转轴、偏置联动与真实固定工具

资产 `floating_mimic_elbow.urdf` 的四个原始质量如下，惯量均在各自 COM 的 authored 坐标轴下给出：

| 原部件 | kg | 局部 COM m | 对角惯量 kg·m² |
|---|---:|---|---|
| base | 2 | (1/32,−1/64,0) | (1/5,3/10,2/5) |
| upper | 1 | (1/8,1/32,0) | (1/16,5/64,3/32) |
| forearm | 3/4 | (1/16,−1/32,1/64) | (1/32,5/128,3/64) |
| tool | 1/4 | (1/32,0,1/64) | (1/128,5/512,3/256) |

shaft 原点为 base 的 (1/4,0,1/16)，轴 +Z；gear 原点为 upper 的 (3/8,0,0)，轴 +Y。`q_gear=1.5*q_shaft−1/8 rad`。工具固定在 forearm 的 (3/16,1/16,−1/32)，额外局部 Z 转 π/2；其碰撞盒为 (1/8,1/16,1/16) m。必须显式 `collapse_fixed_joints=True`；完整模型为根加两子体，工具仍保原 material frame 转换。

shaft 范围 [−1,1] rad，速度2 rad/s，电机4 N·m；gear 范围 [−1.25,1.25] rad，速度1.5 rad/s，原 effort 0.25 N·m。联合独立范围 [−3/4,11/12] rad、速度1 rad/s。

三姿态都取根原点 (1/8,−1/4,3/8) m，根 COM 世界速度 (1/8,−1/16,1/32) m/s，世界角速度 (1/4,−1/8,3/16) rad/s：

| 根 axis-angle | 独立 q rad | 独立速度 rad/s |
|---|---:|---:|
| 单位姿态 | −1/4 | −3/8 |
| 绕 +Y，3/8 rad | 1/8 | 1/4 |
| 绕归一化 (1,2,−1)，−1/2 rad | 3/8 | 1/8 |

g=(1/2,−1,−2) m/s²；独立 effort=13/32 N·m；h=1/4096 s。root/upper/(forearm+tool) 的 COM 力分别为 (1/4,−1/2,3/4)、(−1/2,3/8,−1/4)、(5/4,−3/4,1/2) N；COM 矩分别为 (1/32,−1/16,3/64)、(−1/32,1/64,1/16)、(1/8,1/16,−1/32) N·m。external effort 全零。

独立 oracle 直接从上述四部件的旋转、原点和 COM 求 J，逐部件求 `m JᵀJ + Wᵀ I_world W`、实际动能和材料点 J；不读取产品质量/惯量/主轴数组建立 expected。bias 使用固定世界根 COM 速度、固定世界 omega 和固定全关节速度的几何曲线：COM 二阶导数为五点中心差分，角加速度由两实际转轴的导数解析计算，再加 gyro 与重力。预固定差分时间 δ=1/512 与1/1024 s，二者 bias 最大逐分量差门为 2e−7（线项 N，角/转轴项 N·m），产品 M 绝对门2e−11、bias门2e−7；完整鞍点加速度门5e−6（各自 m/s² 或 rad/s²），状态速度门为该值乘 h，q门2e−11（m或无量纲四元数/rad）。这些门未经过作者数值试调。

M/b 与原未约化鞍点解独立核对；再用预期末速度进行一次根 COM/四元数漂移。另核完整力平衡的根残差2e−10、真实工具角点(1/16,1/32,1/32) m的 world 点与 J。全局轴循环+平移 (1/2,−1/4,1/8) m留出同时旋转根速度、重力、外部力/矩；原行约束反力和标量运动须保持。单步功率不冒充长窗能量守恒。

## F03 候选窗口：待父另行授权，尚未编写执行测试

采用 F01 jaws，无重力、外力、PD或主动 drive；独立初态 q/v=0，根初速度 (0,1/4,0) m/s，其余速度零。world=23、机器人body=17、自由球body=91；自由球质量1 kg、半径1/16 m，初速度/角速度零，其球心固定为 (0.254, 1/16+1/16+1/1024, 0) m：从 right_pad 的有限 +Y 面外 gap=1/1024 m 入射。沿既有真实有限表面 law，k=2048 N/m、d=0、mu=0，T=1/4 s，h=1/2048 与1/4096 s；不得按结果改路径或材料。

分阶段仅期待验证：层外→首次受载→峰值→被动反弹离开；逐步保 root/全部关节/自由球状态、两侧完整力与 COM 矩、入步作用点与 material 地址、原行反力/功及约束残差。首触/峰值/离开必须从真实记录确定。若该候选未离开、超联合边界或遇原未支持域，保存实际失败，不声称通过。规划的 h/h2 工程门沿 F1 同类短窗：最终 COM 位移差≤0.5 mm、平移速度差≤5e−3 m/s、角速度/关节速率差≤1e−2（rad/s或m/s）、累计球冲量差≤5e−3 N·s；不能据此推导一般收敛定理或不碰撞保证。具体公开 scene assertions 与失败注入等下一 grant 再冻结。

scene 后续事务应覆盖：一个参与者限位拒步与晚期事件 staging 失败时，所有机器人、自由体、geometry、载荷、时间及 episode 共同不变；合法重试与 fresh scene 一致。当前测试没有把 contact observer 当约束/摩擦求解器。

## 交接与保留边界

父待执行首 RED：`tests/test_floating_mimic.py::test_a01_public_floating_mimic_preserves_mass_and_root_reaction`。后续单文件覆盖 F01/F01-G/F02，参数化实际数量以父 collection 为准。所有状态均 UNEXECUTED，未标 RED/GREEN。没有增加产品源码、共享导出、GPU、pad适配或新求解器；不重跑旧组。

原100 g mimic持稳在1.048 s滑移3.005689 mm、超过3 mm门且净空仅3.499783 mm的 RAW1继续保留。此片补浮根约束和真实反作用，不改摩擦/持稳门，不将 F03 无摩擦被动窗口代替该原失败。[原报告](../benchmarks/p65-force-events-filtered-batch2.md)

## T3 tests-first 冻结（2026-11-07，AUTHORING_ONLY）

此追加接续上面的 F03 候选，未修改原 F01/F02/F03 输入或验收门。父已接受 T2：110 新23项、111受影响旧37项及112/113静态检查通过；这些是父的既有结果，本次未重跑。本 T3 新测试与例程均 **UNEXECUTED**，不得将编写完成标成物理通过。产品 participant 尚未编写，先由父记录公开 builder 的有效 RED。

首节点：`tests/test_floating_mimic_scene.py::test_a01_public_builder_accepts_reduced_controls_and_full_physical_owners`。先调用已验独立资产的 +1 N 单步，核原行乘子 −4/7 N，再通过 `ContactSceneBuilder.add_floating_urdf(loader=load_floating_mimic_urdf)` 登记同一资产；期待当前缺口在 wrapper 注册而非缺失模块。公开观察固定为 q/v 8/7、full_q/full_v 9/8、contact_effort 7、full 接触/加速度/施加载荷8、真实 body COM wrench 3×6。shape/material 仍属于真实 full tree，根不伪装固定支座。

### 补充正控：实际自由根 PD 与普通机器人互碰

PD 四个实例复用 F01 的全部物理参数，初始 q/v=0，g=0，kp=100 N/m、kd=2 N·s/m、target=0.01 m、h=1/128 s。独立电机配置上限1.5 N；命令分别为 `(cap, root COM +X force, feedforward)=(1.5,0,0)、(0.25,0,0)、(1.5,1,None)、(1.5,1,0)`。external 完整三个 COM 行，只有根 +X 力 F 非零，external.effort 全零。真实根 Schur 得 `M_eff=14/5 kg, b_eff=−F/5 N`；令 `a=h(100h+2)/M_eff`，`ff=b_eff` 仅用于 None，显式0保持0，预期 `tau=clip((1+ff+a*b_eff)/(1+a), ±cap)`。根/driver/follower 的 +X 加速度分别 `(3F+tau)/14、(F+5tau)/14、−(F+5tau)/14 m/s²`；原约束行乘子 `(2F−4tau)/7 N`。根完整外载、resolved 和 requested command 都独立保留，不能把接触当默认前馈，也不能把 target 直接写进状态。

普通机器人留出声明另一非 mimic 固定根树：一个 +X prismatic、移动质量1 kg、COM 在关节局部原点，I=(0.01,0.015,0.02) kg·m²，球半径1/16 m；轴原点 `x=1/4+0.004+1/32+1/16−1/1024 m,y=z=0`，q/v=0，关节范围±0.01 m、速度1 m/s、effort4 N。world23/body42，与 jaws body17 配同一 k2048/d0/mu0，h=1/1024 s，两侧驱动均0。球与 right_pad 的有限 +X 面穿深1/1024 m，真实法向采样应为2 N。独立投影载荷 `(root,driver)=(-2,+2) N` 和质量矩阵 `[[5,-1],[-1,3]] kg` 给根 −2/7、driver +4/7、follower −4/7 m/s²，普通轴 +2 m/s²，原行乘子 +2/7 N；各 COM 矩为0，总 +X 线动量为0。两登记顺序均覆盖；每条 material 地址按实际所属 full link 原点重建世界表面点，与共同作用点分开，不借组件序号推材料身份。

### 原 F03 的完整物理保存与有限窗口门

`examples/floating_mimic_contact.py` 只用原 F03 公开声明和 `scene.step` 推进：T=1/4 s，h=1/2048 与1/4096 s；另分别使用 h 跑反序登记和整体轴循环 `R(x,y,z)=(z,x,y)`、平移 `(1/2,−1/4,3/4) m`，同时变换初始速度与位置。没有重置、预设之后姿态、路径/物性/门的再调整。四窗合计计划2560个真实子步，数量不是已执行事实。

每个接受步保存完整 public reading、独立/full 初末状态、完整 SurfaceLoads 与原材料/作用点、各 body COM 力矩、mimic 原行反力/功、事件时间、当前 geometry poses，以及同一入步的 full M/b 与平衡残差。initial 保存完整模型/geometry；静态模型不逐帧复制。另保存自由球累计真实力冲量、声明质量/惯量下的总线/角动量与动能。M/b 是产品收据的逐步余额诊断，不是独立动力学 oracle；独立解析正控在上述 PD/2 N 节点。载荷/功/材料属于 `load_time_s`，状态与 geometry 属于 `time_s`；共同点力矩抵消不等于有限步总角动量精确守恒，实际角动量残差保留而不改名。

窗口要求从层外到真实受载再卸载、末态无正法向力、root/关节确有响应；各步核双侧合力/共同点矩及实际入步功率、约束功与根约束残差。原步长门保持：位移差≤0.5 mm、平移速度差≤5e−3 m/s、角速度/关节速率差≤1e−2（rad/s或m/s）、累计球冲量差≤5e−3 N·s；另本例整体线动量残差上限5e−4 N·s，单步 full 力余额绝对门1e−8（N或N·m），功率门1e−9 W。未从运行结果反调门。若超联合限位/支持域或未离开，文件保真实异常及全部已接受前缀，qualified=false，不把缺失终态作通过。

pytest module fixture 把四份完整结果分别无损写入其明确 basetemp 下 `floating-mimic-full-windows*/{h,half,reverse,transform}.json.gz`，包含失败前缀，不按结果删帧。CLI 要求新的绝对 `--out`，可用 `.json.gz`，固定输入由 `--dt`、`--transformed`、`--reverse-registration` 选择；`--fixture-dir` 只定位随交付的两份既有 URDF 目录。收据带 fixture/driver SHA256，若 standalone 失败退出2。父执行时同时绑定 source 和 wheel 路径，作者本批不执行。

### 事务、所有权与待执行清单

普通轴初态 `(q,v)=(0.0099999 m,0.001 m/s)`，无驱动将于 h=1/1024 s 超 +0.01 m 联合边界；同时给浮根 COM +1 N、远处分离自由球 +Y 7 N，确保存在其他真实候选。失败时全部状态、geometry、reading、time、events 不变；同态普通轴命令 −4 N 的合法重试与 fresh scene 比全量。另一例先接受真实2 N 接触，再注入晚期 events.stage 错误，核共同回滚及合法重试一致。输入数组后改、返回的可写载荷数组后改均不能改引擎旧读数；sample/observe_scene 不消耗 episode。

预期专项为首公开入口1、PD4、普通机器人互碰2、共同回滚/晚期回滚/所有权各1、完整窗口/步长/反序协变各1；手工计13实例，实际 collection 由父报告。首 RED 后才进入已授权的 NEW participant 实现阶段。仍不增加 CUDA、CCD、静摩擦或抓取持稳结论；原100 g 持稳 RAW1与其门原样保留。
