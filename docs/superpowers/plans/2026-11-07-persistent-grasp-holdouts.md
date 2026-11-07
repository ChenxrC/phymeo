# 持续抓持独立留出：原输入、显式材料站点与系统验收

2026-11-07。**PLAN_ONLY / UNEXECUTED：本次只读源码、原收据并新增本文件，不授权自动实施。** 父已签收通用场景持续切向产品 `c595c76` 的 F01–F06、材料桥、兼容及源码外使用；本次只读父 HEAD 为 `d2eeba55ff1936be5abb1ac66be967618b8b3227`，文档子树基线 `7f70454`。这些既有成绩不重跑，也不据此关闭原 F07。下一授权按本计划先 tests-first、父串行 RED，再决定声明工具/例程接缝及一次完整候选窗口。

## 1. 具体目标与原失败

把**已有** `RobotScene.enable_coupled_tangent`、`MaterialStationRegistry` 和全跨 owner 的切向求解接入原获取例程。原例程已经通过 `ContactSceneBuilder` 创建真实场景，并非另一个尚未接通的积分器；无须再复制 controller、mimic 投影或求解器。新增可复用能力是从用户已经声明的 `PadFace` 产生稳定的材料站点登记，并完整记录这一次新摩擦模式下的获取→保持/失持→释放。

原四条独立实测均保留，来源为 `artifacts/p65-postnight-batch2-20261105-01/a2-*-window/{stdout.log,final.json}` 及对应 `a2-*-result.json`。本计划只读摘要/源，不重新求解或程序化重算旧轨迹。

| 原案例 | 真实原结果 | 必须保留的判读 |
|---|---|---|
| `mimic-box`，100 g | 1807 接受步，约1.807 s；取得双侧力、未 held、已真实释放；7/8、RAW1 | 首失败 `relative_slip`：1.0479999999999954 s，滑移0.0030056893591103844 m，离台0.003499782732429041 m，台力三分量全0；不是执行异常，也不是离台10 mm/保持100 ms成功。 |
| `mimic-zero-friction` | 1681步；失持、释放；8/8、RAW0 | .923 s 滑移3.012614 mm，台向上力.980305 N；负控通过不是抓取通过。 |
| `mimic-overload` | 1789步；失持、释放；8/8、RAW0 | 1.033 s 滑移3.010834 mm，台向上力7.625533 N；不能只用 holding=false 掩盖获取/释放失败。 |
| `panda-sphere` | 3008步，约1.880 s；获取、held、释放；8/8、RAW0 | 新模式还必须单独通过；Panda 球可能有几何包持，不能替代竖直夹箱的零摩擦/超载留出。 |

原100 g失败发生在[原运行协议](../../../examples/robot_force_acquisition.py) `run_case:261` 的 lifting 阶段、`slip > .003` 分支（当前约311–326行）；`contact_ready` 与 `held` 的定义始终分开。[原报告](../../benchmarks/p65-gripper-acquisition.md)与[第二批总览](../../benchmarks/p65-force-events-filtered-batch2.md)保留首次RAW1。旧 A1 突加2 mm预压失败是另一件事，本计划不修改或重标它。

## 2. 输入冻结：四个候选，强夹不加力

直接调用原 `examples/robot_force_acquisition.py:35 make_case(case, urdf=None)`；本计划不复制一份“等价”模型。原 `tests/reference_newton/robot/mimic_jaws.urdf` 和包内经典 Panda 资产原字节绑定。所有初速度为0、原 g=9.81 m/s²、真实有限桌和第二自由物体及全部结构 pair 保留。唯一物理模式变化是**只对两侧目标 pad–目标物配对**显式启用切向弹簧，预先固定 `kt=200 N/m/站`；原正常接触、阻尼、μ、其他 pair 和驱动均不变。

| 新验收编号 / 原 case | 输入与独立预期 |
|---|---|
| G01 强夹持载 / `mimic-box` | 原40 mm立方体，m=.1 kg，COM=(.25,−.125,.25)m，均匀箱惯量；q=(0,0)，h=.001 s。目标各侧2 N、μ=.65，理想切向容量2.6 N大于mg=.981 N；这是可持载正控，仍须真实达到原 held 门。|
| G02 零摩擦 / `mimic-zero-friction` | 与G01全部相同，仅原μ=0。正N时新切力、弹性储能和累计塑性滑移为0；竖直指面法向不提供竖直承重，必须不 held，并仍完成原获取、释放和真实台面支撑。|
| G03 超载 / `mimic-overload` | 与G01同几何/控制/h，原m=1 kg，均匀箱惯量按原builder生成，μ=.65。独立闭合驱动帽8 N，在静态对称且无其他竖直支撑时每侧N≤4 N，切向容量≤5.2 N<9.81 N；不把这当所有瞬态N的硬上限。最终按实际载荷/台力/运动核未 held。|
| G04 不同机构与曲面 / `panda-sphere` | 原7R+2P经典Panda，arm=π(0,−.25,0,−.75,0,.5,.25)，两指初q=.03 m、目标.01 m；r=.020 m、m=.04096 kg球，COM为原初始两指面站点中点，h=1/1600 s。原受力获取与实际IK提升20 mm、持稳、释放均须通过；单独记录真实转动/球面材料运动。|

原 M 机构：carriage=1 kg，left=.2 kg、COM偏移(0,.004,.006)m，right=.3 kg、COM偏移(0,−.003,.004)m；rubber全尺寸(.006,.024,.018)m；升降+Z、两指+X，idler=−screw+.004。原完整惯量/限位以URDF为准，不把独立2维当完整3关节。M 第二球r=.015 m、.03 kg、COM=(.33,−.125,.245)m；台半尺寸(.12,.12,.01)m、COM=(.25,−.125,.22)m。

原 P 第二箱全尺寸(.030,.024,.028)m、.06 kg、主球COM偏移(+.10,0,−.006)m；桌半尺寸(.14,.08,.02)m，中心为主球COM−(0,0,.04)m。P原球惯量和全部link质量不替换。

两例 pad 都是各侧四点、面内偏移 `product((−.005,.005), repeat=2)`。目标 pair 原律 `RobotContactMaterial(1200,1.2,μ,.04)`；物体–桌2200/3/.45/.04；其他结构6000/16/.35/.04。每个 site 使用真实 `site.force_n`，不再次乘fraction，也不向八站点各发2 N。

原控制不变：目标2±.25 N/侧，ready要求各侧≥1 N和夹指速度≤.002原坐标单位/s持续.020 s；失载持续.010 s；获取超时1.5 s。M闭合目标.018 m、速度.020 m/s、帽(40,8)N，kp=(2700,6000)、kd=(95,35)。P指速度.025 m/s、帽各4 N，arm/finger kp=500/4500、kd=30/18，其余额定帽保持。提升20 mm、.600 s五次平滑posture目标；holding .100 s；release超时1.200 s。`None`与显式0 feedforward沿原请求/解析语义，禁止改增益或按结果重选夹力。

## 3. 可复用的最小声明工具

建议新增纯构造函数（最终名字/导出由父冻结）：

```python
material_stations_from_pads(
    scene, *, robot, payload_key, pads,
    station_keys, counter_features, chart_ids,
) -> MaterialStationRegistry
```

- `robot=(world, physical_body)`；`pads`就是构建scene时使用的原不可变 `PadFace.named(...)`。`station_keys`按作者的 `(link_name, collision_name)` 映射到与**声明offsets**一同冻结的四个 `StationKey`；例如左101–104、右201–204。可同时重排offset/key声明，不能用采样row、世界坐标hash或当前最近点生成token。
- `counter_features`与`chart_ids`亦由用户按语义pad声明；每项chart为pad侧/物体侧两个显式ID，不根据当前接触结果“猜”连续性。返回只读registry，无历史推进、无profile、无积分。限定本片实际使用的普通固定根/mimic可动cuboid pad，与真实sphere/cuboid对象；其他组合明确拒绝。
- 重用 `runtime/robot_payload_scene.py:105 _pad_points` 的同一local点/固定折叠变换；mimic用原完整 `asset.tree`。通过公开语义 `scene.shape_key` 和真实 `vertex_ids` 映射，不能将authored link-local误当model-local。切向轴由pad剩余局部轴中的第一轴经同一frame转换得到，仍由registry投影normal；平行退化照常拒绝。
- 原 M 左pad为+X面，pad IDs=(4,5,6,7)，物体对面为−X IDs=(0,1,2,3)；右pad相反。原 P axis=1、side=−1/+1，pad IDs分别(0,1,4,5)/(2,3,6,7)，球中心材料feature=(0,)，真实skin offset与局部球面点仍保存。不能把整个box八顶点宣称一个平面chart。
- 在t=0、尚无目标接触时一次构造八登记并调用已有 `scene.enable_coupled_tangent(registry, stiffness_n_per_m=200)`。`configure`只允许开始前一次，故不在 `contact_ready` 后才装history。未出生的登记合法；原采样某站点N>0时由真实固定锚点匹配出生，N=0/absent按既有事务结束。

源码依据：`MaterialStationRegistration`/`Registry.bind`见 `runtime/material_stations.py:111,390`；每个selected正N site必须恰一声明，match tolerance原1e−8 m。`_multi_robot_tangent.py:28,333`已有single/multi共享准备；`material_stations.py:323`用入步真实双侧local锚点/作用点随末姿态移动；`solvers/tangent_history.py:322,370`保存same-chart连续性、generation和释放储能，不把closest漂移作为滑动。`RobotContactEventAdapter`仍只观察同次入步载荷。

**文件建议，待父下一grant：** NEW `src/phymeo/runtime/pad_material_stations.py`、`tests/test_persistent_grasp_stations.py`、`tests/test_persistent_grasp_holdouts.py`、`examples/persistent_grasp_holdouts.py`、`docs/benchmarks/p65-persistent-grasp-holdouts.md`。不新增物理kernel、controller或第二套事件schema。

**父旧文件的最小接缝：** root导出上述函数；原 `examples/robot_force_acquisition.py:35` 的info额外返回原 `pads` 声明（不重建或改b.add_urdf输入）；`run_case:261`新增可选 `on_accepted` 只读观察钩子，接受步后传递原scene/before/after/proposal/stage。默认None原输出/时序保持。新driver复用 `make_case/run_case`、原八门和原异常路径，不复制状态机。观察钩子不参与物理发布；其IO失败应单列已接受步的归档错误，不能伪称该物理步回滚。若可直接复用等效已有公开观察钩子，优先复用。

## 4. 真正开放的边界与最小反例

1. **出生不是当前主要数学缺口。** 八个pad点是装配时的真实固定材料点，inactive→正N可提前登记。先验证原open场景无目标N、原命令真实闭合后每个正载site准确绑定；不能从第一次力行序建立永久ID。
2. **payload在同一已声明面内漂移**可保同chart历史，但每步弹性增量必须来自`measure_motion`的入步材料点真实末位移，而非相邻closest点差。原M支持集只准对应±X面；P球只准原sphere和真实owner。两侧作用点仍沿旧pad路径的共同 `p`，不能为通过功门把它移到skin。
3. **正在受载的跨面/边角迁移未实现。** 当前registry不支持在t>0重新配置；support离开声明face、同token匹配两个正N site或出现未登记正N site都应拒步。不得加大match tolerance、扩大为整box、删guard，或每步换token冷启弹簧掩盖不连续。若完整原窗首次在此拒绝：保存最后接受state、原下一proposal、真实入步site/local/world点、normal/weights、已声明feature、history/receipt和错误；下一授权仅以这些真实输入写绑定最小复现。迁移模型另立节点，明确旧储能处置和真实功，当前原窗标未完成。
4. **Panda实际非线性转动仍过原门。** 保持材料/作用点预测及彼此差1e−6 m、能量残差1e−8 J、单步frame转角.25 rad。sphere skin与共同作用点有不同local offset，真实角运动可能使两种位移超门；须报告具体值/步时，不能减dt、改IK目标或放宽门使本窗通过。已有F05通过不等于任意Panda步必通过。
5. **材料数和故障**保持128站点/历史token上限；本片显式8，不动态增容逃避限额。错误owner/world/重复key/错误face、stage分配或finish故障均保持scene状态、history、events、receipt、clock和controller待提交命令；同一已接受base的同proposal重试一致。观察不耗episode。

两个优先短反例：原M第一个真实有载入步，将一侧counter face故意声明为另一侧（只改独立负控声明，不改原正例），必须在bind拒绝且未推进；同样入步注入一次stage/finish异常，重复 `controller.propose` 与同命令重试须无第二个目标增量。这两项专查新应用连接，不重跑F06整套。

## 5. RED与短验收顺序

T1 tests-first先固定 `tests/test_persistent_grasp_stations.py` 的以下节点，产品未授权前交父：

- `test_named_pad_registry_predeclares_real_open_mimic_sites`：原make_case、真实asset/finite几何先通过，核8个原offset在真实指面、目标pair初始无预压；再调用拟公开声明工具。缺口必须落在这个已命名公开行为，不能用拼错import/缺依赖作RED。
- `test_pad_declarations_survive_mimic_projection_and_panda_fixed_frames`：两个真实资产，非连续token、声明重排/不同world不污染；用实际owner frame重建全部anchor，物体侧原feature不扩大；root/child错误映射与重复声明拒绝。
- `test_first_loaded_pad_sites_and_retry_publish_once`：沿原controller/原dt得到真实首次正N，完整事件所有site一一绑定；与原入步采样核normal及未选力未改，完整full/independent广义力和COM力矩一致；仅新连接的失败/重试正负控。

允许先交首节点checkpoint让父有效RED，然后author小声明工具；不必等完整长窗脚本。父先短节点与真正受影响的调用兼容，任何坐标/metadata错误只修通用桥并保存首错，不修改原几何/阈值。201/208发现的OBB face-chart丢失已经由父修复并通过，不重新运行旧RED；这不预先证明原 sampled-pad fallback 在所有旋转姿态都无材料误差。

## 6. 一次完整窗口：八门不变，新增独立账

四例各从原t=0完整运行一次新候选；使用原 `run_case` 接受步、阶段和事件提前停止规则。G01/G04须八门均真；G02/G03的预期holding=false仅满足一个门，其他七门仍须真。旧CLI/旧pytest窗口不再重复取基线，不用新旧不同摩擦律的时间做性能比。

- **物理held原门：** 真离台≥10 mm、table向上支撑<.1mg、双侧≥1 N、保持窗口相对COM变化≤2 mm持续100 ms。自contact_ready累计相对滑移>3 mm仍立刻记原物理失败并走原release。contact_ready沿原2±.25 N目标带控制，硬判据仍为各侧≥1 N及速度≤.002持续20 ms，不另添力带判据，也不等于held。
- **原release门：** 原open目标误差≤.1 mm、速度≤.002 m/s、双侧载荷≤1e−8 N连续20 ms；两自由体真实台支撑、目标物release请求后的台力采样时间均保留。不得在request瞬间清history；仅后续真实入步N消失时结束站点、释放旧E一次，之后无残余切力。实际重新入载若自然发生，原token的新generation必须冷启动，不能复用旧弹簧。
- **反力/推进：** 每个正N站点两侧force相反，同一原作用点对应双方COM矩；记录完整 `contact_effort_full`、独立 `contact_effort`、`drive_effort`、link/root wrench和mimic原行反力/功。按真实Q转置检查投影，无把mimic两个指头当两个独立电机。固定根、重力和桌是外力，不能宣称全场总动量恒定。
- **独立自由体账：** 两自由体逐步 `m(v1−v0)=h(F_contact+mg)`，沿原误差<3e−12 N·s、motor帽超额≤1e−9；载荷属于load_time_s，姿态属于time_s。台/非目标体/非pad结构不得被选择器隐藏。
- **新切向账：** 每站 `|ft|≤μN`、`ft=−kt e`（当前候选返回的弹性段）、E=.5kt|e|²、实际作用点work、塑性增量/耗散及释放E逐项可复算；原motion/energy门不变。完整history/receipt与entry/末态共同发布，功账不冒称整个场景严格能量守恒。
- **独立量级参照而非达标替身：** 理想G01八站各N=.5 N且纯竖直同位移时，8kt=1600 N/m，平衡弹性位移 `.981/1600=.000613125 m`；每站承重.122625 N小于μN=.325 N。此只说明原2 N协议静态容量可行，不用它覆写轨迹或要求真实系统精确到该值。G02无切向承重，G03静态容量不足，各由原实际系统门裁定。

## 7. 执行、记录与止损

拟NEW CLI（待父授权实现）：

```text
python examples/persistent_grasp_holdouts.py --case mimic-box --out-dir <fresh-absolute-dir>
python examples/persistent_grasp_holdouts.py --case mimic-zero-friction --out-dir <fresh-absolute-dir>
python examples/persistent_grasp_holdouts.py --case mimic-overload --out-dir <fresh-absolute-dir>
python examples/persistent_grasp_holdouts.py --case panda-sphere --out-dir <fresh-absolute-dir>
```

先G01，后两负控，最后G04；若父为串行调度调整顺序，输入/门不随顺序变化。单条完整新窗只执行一次，相关pytest长窗不再重复。原步数预算 `ceil((1.5+.6+.1+1.2+.1)/h)+3` 原样保留，实际整数预算直接由原代码计算，不另行取整或替换为手写常量；不得因切向法则更慢或未 held 而加时。法则改变后转折时刻可能变化，不硬套旧1807/3008步。

旧收据仅给调度参照：M从22:29:31.636至22:30:18.509，P从22:22:16.586至22:23:03.886，均约47 s，属于旧滑速律和当时源码。本新full-W/逐站history成本未测，不能外推性能。在新同一进程得到首个正载后的64接受步时报告实际用时/站点数，再继续原窗，不以重启重复前缀代替续算。controller没有公开序列化恢复合同，单独恢复q/v不等于恢复控制时钟/材料历史；资源中止必须标incomplete，不伪装可恢复完整窗口。

输出新目录、绑定源HEAD/全部相关SHA、原URDF/模型声明/控制量/所选keys/charts/kt/gates。每步gzip完整保存原request/resolved命令、proposal、entry/accepted state、full loads/normal与切向力、material addresses、history/receipt/events、两时刻、台力/净空/滑移及原八门；不能仅保存摘要hash或关键帧。完整geometry保存同锁对应姿态，流式无损gzip，单例预声明256 MiB压缩上限；容量不足在命令自然边界保完整前缀并失败，不删除旧证据/按结果抽帧。源码外验收仅在新声明/应用实际通过后由父同一新wheel执行，不冒称这份计划已验证。

**止损：** 首个未知材料迁移、真正运动/功拒绝、结构冲突或物理持稳失败均保原原因；不扫kt、μ、力目标、时间、步长或容差。原物理失持继续原release以保卸载观察；真正执行/事务错误按原协议终止，不能退回旧摩擦偷偷续算。G01若仍失败，交付稳定声明工具/可重复最短材料负例及其它独立正/负控结果，原F07仍开放；Panda成功不足以替代它。后续只有父明确新grant才能研究迁移或更广的非线性运动响应。

## 父裁定：先验声明测试库（2026-11-07）

父已全文读本110行计划及原make_case/run_case、_pad_points、registry与frictionless实现，正常导入921ac62。允许原RD在clean 4de207a隔离树继续 **TEST_AUTHORING_ONLY**。只NEW `tests/test_persistent_grasp_stations.py` 与 `docs/benchmarks/p65-persistent-grasp-holdouts.md`；先交首个正常公共行为节点的小checkpoint，完整短矩阵随后同一任务补齐。不得写产品、driver、长窗测试或修改任何旧文件；禁止全部项目执行/数值probe/static/format/CI。

冻结公开名称 `phymeo.material_stations_from_pads`，按§3关键字签名：robot=(world,body)，pads原PadFace序列；station_keys/counter_features/chart_ids均以原(link_name,collision_name)为键，分别为与声明offset同序的StationKey元组、payload完整面vertex IDs、(pad_chart_id,payload_chart_id)。只声明并返回owned MaterialStationRegistry，不调用enable、不采样当前contact、不推进时钟/history。原make_case尚未返回pads时，RED测试可用原公开PadFace和已冻结原offset显式重建同一声明作为测试输入；不能因读取未来info字段提前失败。父以后只给原info补同一pads tuple。

首个实际RED先成功构造原mimic-box scene/asset、核初态与8个合法face锚点，再调用上述正常公共行为；缺失函数为预期边界。短矩阵保两个真实资产、mimic完整物理投影/Panda fixed frame、声明重排并同步重排keys、非连续token、错误owner/world/face/重复key、真实首次正N绑定与唯一新连接的rollback/retry。不要重跑既过F06长套或G01–G04旧窗口。首次有载短测试在原dt/原controller下只前进到所需节点，明确不同于完整持稳通过。

每个真正正载样本必须保存真实first/second的owner、local material/application、normal、weights与完整history/receipt；不可把plan静态容量当实际持稳证据。父审阅/串行RED后才授新声明工具与应用driver，原四完整候选窗依然全部UNEXECUTED。若声明接口发现更广能力缺口，给真实最小输入，不调原参数。
