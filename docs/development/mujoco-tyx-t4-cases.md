# MuJoCo-TYX T4 多案例与 G1 持续支撑预登记

2026-11-10；基线 `73a85f8affccde5ebf21a4629845d06af6949479`。本文件状态为 **RESEARCH_CASES_READY / API_PENDING / UNEXECUTED**。作者仅负责本 NEW 文档，父负责共享接口与唯一串行执行。以下六个节点是能力验收顺序，不是按一个舞蹈结果倒推参数；未经 API 冻结不编写或执行新测试。

## 1. 来源与本批边界

固定 CPU MuJoCo 3.3.7 的 [约束与求解说明](https://mujoco.readthedocs.io/en/3.3.7/computation/index.html#constraint-model)、[原生约束装配](https://github.com/google-deepmind/mujoco/blob/3.3.7/src/engine/engine_core_constraint.c)、[原生求解器](https://github.com/google-deepmind/mujoco/blob/3.3.7/src/engine/engine_solver.c)是研究依据；实现输出不作期望。复用 [T3b 验收](../benchmarks/p65-mujoco-tyx-finite-contacts.md)与[扩展计划](mujoco-tyx-expansion-plan.md)，不重跑旧绿组。

父已选本批研究 **pyramidal condim=1/3/4/6、scalar joint limits、joint frictionloss 的共同有界 box-QP**。condim 1/3/4/6 的接触边变量数分别为 1/4/6/10；接触数量与 efc 行容量必须分开。数值算法若为新命名的 box-QP active-set，应明示其算法/预算，不冒充 CPU Newton 的逐次迭代。elliptic、explicit pair、mesh、ball、equality/tendon不混入首域。

Native dry friction 是独立有界约束，不是 viscous damping。标量限位两侧分别建行；margin足够大时同一关节两侧可以同时存在。原生类型次序、DOF/J/下上界、R/aref与物理力均须绑定实际行。pyramidal摩擦是软接触模型，不预设硬库仑的严格静止或每步最大耗散。

Native 会把 condim>1 的 authored 零摩擦系数限制到 `mjMINMU`；因此真正零切向力负控用condim1，另测 authored0与原生最小μ的一致性。implicit geom三系数分别为滑动、扭转、滚动；两个切轴和两个滚动轴各自复制。首域不宣称独立的五轴异向输入；仍测三个物理族不同系数、所有方向正负与非平凡接触frame。

## 2. 六节点与最小必须范围

| 节点 | 难度 | 必须交付与停止条件 |
|---|---|---|
| A 单接触摩擦 | 高 | condim1/3/4/6完整装配/解码与独立同入步；μ0、正负速度、阈值附近、滑/扭/滚方向；失败先定位frame/J/参数/解码 |
| B 共同质量与多接触 | 高 | 完整M、两足/偏心/两个活动branch、双方force+couple；含独立逐点求解反例，不能把多个force分别推进 |
| C 关节约束 | 中高 | hinge/slide lower/upper、frictionloss与黏性阻尼区别、边界切换、同一DOF同时参与多类约束 |
| D 持续混合轨迹 | 高 | 摩擦+limit+dry-friction共同solve；有限表面卸载重入；两质量×两h，真实入步速度不清零 |
| E 公共协议 | 中 | capacity恰满/超限、budget耗尽、全批拒绝/晚故障、selected reset/clone下一步、ENV控制保持、禁止CPU/Warp/HOST fallback |
| F G1组合留出 | 高 | 原资产admission/布局→短窗支撑→已声明扰动→原策略2秒/20秒；依赖A–E，不替代通用矩阵 |

A–E是本批最小通用完成范围。F若被资产字段/容量阻塞，要记录具体缺项并保留独立进度；不能删几何、exclude、condim或干摩擦来假称原资产支持。测试节点数以后按冻结参数化清单报告；本文件的表行数不是pytest collection或完成比例。

## 3. 固定公共输入与案例族

默认单位SI；hinge rad/Nm，slide m/N，接触切向N，扭转/滚动Nm。基础h=.001s、m=1kg、gravity=(0,0,-9.81)、Euler、source Newton iterations128/tolerance1e-10、dense Jacobian、pyramidal；接触solref=(.02,1)、solimp=(.8,.9,.002,.5,2)，非负margin。基础 sphere r=.1、初始中心z=.099，box半尺寸(.12,.08,.06)/初始z=.059；有限平台半尺寸(.24,.18,.05)时同步加其顶面高度。旋转正控用完整正定非主轴惯量与COM(.03,-.02,.01)。除明确行语义正控外armature=.01、D=.04；解析纯平移正控显式a=D=0，不修改其他case。

| ID | 最小原始输入与独立判断 |
|---|---|
| A01 | condim1，初始vx=±.4；实际切向力/接触couple为0，横向运动仍保留 |
| A02 | condim3 authored friction=(0,0,0)，核实际native混合后μ及小摩擦；不要求其等于condim1 |
| A03 | condim3 friction=(.6,.02,.003)，切速分别沿frame两个切轴±.4；无横向外驱时反向摩擦，全部力按native比较 |
| A04 | 3轴纯平移sphere，v=0，固定外力Fx=±.25mg与±1.2mg；分别记录接近粘着/持续滑移，禁止用“必须精确静止”断言软模型 |
| A05 | 同A04的独立阈值测试：从该固定原生模型的H与b(F)=b0+F*b1解析枚举相邻active-set，求首个边变量归零的正阈值Fc；**预先固定**0.9Fc/Fc/1.1Fc。构造输入只用原生模型/解析公式，早于TYX输出，不搜索产品结果。核状态转变与连续力，不用μmg替换软模型阈值 |
| A06 | condim4，初始局部ωn=±2rad/s，其余角速0；核扭转couple的符号、单位、到双方COM的完整世界力矩 |
| A07 | condim6，分别ωt1/ωt2=±2rad/s；同时保原(.6,.02,.003)，核两滚动couple，不能当成切向力或漏掉接触纯力偶 |
| A08 | condim6，自由偏心body，v=(.3,-.15,-.04)，localω=(1,-.8,.6)，非identity WXYZ；另有斜面frame正控，全部六维wrench与CPU对照 |
| B01 | 同一free root两分离接触区域，偏心外载使分配不对称；保持所有零力接触，不能要求每站均承载 |
| B02 | 同祖两hinge-slide活动branches互碰，双方实际关节力矩及root/祖先反作用；全J包含所有相关DOF |
| B03 | 同一support的多个切向行，分别求每点与共同H求解产生不同答案的固定解析小例；产品必须匹配共同解与qacc |
| B04 | finite capsule-box/box-box非主轴接触，frame和contact顺序允许T3b原几何容差内完整双向匹配；不按force选择对应、不丢重复点 |
| C01 | hinge/slide各lower/upper：range=(-.2,.2)、margin=.01，q分别±.195（行已激活但几何未越界）、±.205（软越界）、0（无行）；v为趋近/离开±.3，核J符号和恢复力 |
| C02 | 同一hinge range=(-.005,.005)、margin=.01、q=0；native两限位行同在，完整保留而非只取最近侧 |
| C03 | hinge/slide dry frictionloss=.2，v=0、外广义力分别±.1/±.3；另v=±.4/外力0；核独立盒约束及阈值，不能用-f*v |
| C04 | C03显式D=.04与dry=.2共存；两项都计且不重复；free-root原生6DOF干摩擦按真实source admission单列正控或明确拒绝，不能静默丢根行 |
| D01 | 2R/P耦合树：支撑contact condim3/6、接近关节上限、非零dry，同一input/control共同解；卸去驱动后真实离开limit与重新接触 |
| D02 | 有限平台三平移slider上的capsule/box，保vx=.15、vz=-.02；先支撑、上推3mg持续[.096,.160)s后解除，连续推进至1.0s，不reset；横向实际有限运动与摩擦一起核 |
| D03 | D01一固定混合fixture与D02两形状均m={.25,2}kg、h={.001,.002}s完整Cartesian；惯量按m比例、geometry/μ/solref/compliance不变，不把mass-only变体改成相似缩放 |

A06/A07正负方向均必测；A08再以两质量×两h核完整六维耦合，不能只给condim6一个姿态。D02使用T3b已公开的一秒窗口，不沿用原.512秒“已经全过”的错误标签；新增摩擦使这是新的物理矩阵。

## 4. 三层oracle与固定工程阈值

1. **同入步原生**：CPU3.3.7从实际FP32发布的q/v/控制构造自己的state，只规范化native自身quaternion副本；读取FK/COM/J/M/b/passive/applied/a0、全部contact与efc(type/id/owner/address/J/pos/margin/frictionloss/R/aref/force)，独立读完整contact-frame wrench。Native原迭代统计与TYX自己的统计分列，不能要求相同迭代轨迹。
2. **独立持续CPU**：从一次共同初态开始连续走每个物理步，不逐步将TYX端点写回CPU。控制有两个明确模式：相同预登记开环输入做solver比较；策略组合则各自反馈各自真实state。全程保存q/v/控制/力与事件，完整payload只取固定首步、边界、关键相变、末步和首次失败。
3. **独立解析**：单normal解析；≤6行box-QP完整active-set枚举（lower/interior/upper按行合法集合）；Jᵀλ、双侧force/couple和功率恒等式；纯dry单DOF的λ=clip(-b/H,[-f,f])。高维用CPU与独立KKT，不再构造昂贵精确证明。

沿用T3的同入步门：geometry point/body pose rtol3e-5/atol4e-5m，normal atol1e-4；J 2e-4/5e-5；M 3e-4/5e-5；bias/passive/applied 4e-4/4e-4；a0 2e-3/4e-3；R/aref/Jv 5e-4/4e-4；dual H 7e-4/5e-4；force、广义力和新couple分量 3e-3/3e-3（按各自N/Nm），qacc/integration 3e-3/1e-2。同输入积分用physical M与M+hD各自正确位置；不重复加D。

基础短窗≤.1s保持原q abs2e-4、v abs3e-3、time2e-6；D长窗逐步同入步仍过上述门，同时持续CPU最终q abs2e-3、v abs2e-2、全窗累积contact impulse相对2%+绝对2e-3Ns、角冲量相对2%+绝对2e-3Nms，分离/重入事件差≤max(2h,.005s)。这些是**首次执行前新增长摩擦窗工程门**，不是改写旧T3通过或失败。

独立box-KKT设g=Hλ+b，lower行违反max(0,-g)，upper行违反max(0,g)，interior为|g|；归一尺度max(1,max_i(|b_i|+sum_j|H_ij λ_j|))，残差≤5e-5。原生最小摩擦系数使用实际source值；不拿未解码edge变量当物理切向力。全部λ满足其盒界；接触解码满足对应pyramid，不能只验证圆锥的较弱界。迭代耗尽必须失败，不能靠吞掉行过门。

D02的行为门：前.096s有真实支撑；连续无接触/零力区间≥.02s；解除上推后于.7s前重入；最后.05s平均竖向力∈[.7,1.3]mg、|vz|<.1m/s、相对原支撑高度差<.01m。横向距离/速度由同律CPU与预登记摩擦诊断检查，不假定被摩擦严格钉住。摩擦功率负号只作为无外驱纯滑/纯转动解析case的断言；一般受驱软接触报告完整功率，不套硬接触定理。

## 5. 容量、原子发布与ENV

E至少包含：每种condim的contact count与expanded efc分别恰满/超一；同一批几世界分别有dry/limit/contact、inactive接触仍占contact容量；单world overflow不得让其他world先发布；source小迭代预算确实未满足KKT时全批失败；晚下载失败前后q/v/act/ctrl、tick/time、force、solver/warmstart（若存在）全部原样，重试与未失败clone下一真实步相等；selected reset只清选中world的全部约束历史。

禁止数值fallback的负控覆盖CPU mj_step/forward、HOST mechanics、Warp和NumPy linalg，在真实device step中触发；oracle在负控外计算。NaN/overflow/不支持cone/condim/source选项须准确拒绝，不能改选项后声称兼容。父冻结动态compact ABI后增加候选遍历完整性、stable身份、恰满/overflow、反向输入顺序与不同world mask；不按可见输出推测候选完整。

公共ENV用新的混合约束fixture，observation确实消费已发布摩擦/limit力，动作保持跨decimation；实际受力后selected reset与clone下一步对照。这是NumPy策略接口，不能据此宣称全GPU policy/training。

## 6. 实际G1资产与支撑入口

只读本机原资产目录 `artifacts/p65-g1-research-20261109-01/`：`sources.json` 固定官方Unitree commit `1425b15f73bd4095f0df53709d7c389c3eb9e790`；`deploy.yaml` 为50Hz/29关节，ONNX输入154/输出29，动作6574帧。实际入口 [g1_policy_reference.py](../../examples/g1_policy_reference.py) 使用native free-root world-linear/local-angular速度与原frame0 q/v；[g1_phymeo_support.py](../../examples/g1_phymeo_support.py)和[旧连接验收](../benchmarks/p65-robot-environment-connections-20261110.md)保留原工程门与真实失败。旧HOST/TYX resident接触律不等于MuJoCo-TYX。

| 原资产 | 本次文本核实 | 对本批的含义 |
|---|---|---|
| scene_g1.xml | default D=.05/a=.01/frictionloss=.2，原物理mesh/cylinder与足球 | 完整模型尚不能由primitive-only T4冒称迁移；不静默去mesh或dry |
| g1_primitive_actuated.xml | collision priority1/condim6；原33个球/胶囊；左右脚各7个foot_capsule；floor摩擦(1,.005,.0001)；4个body exclude；29个scalar joints有原限位，显式a=.01/D=.05/dry0 | T5选用这一已存在明确变体，保原condim/质量/几何/驱动/初态；新支持exclude及完整potential-pair/efc容量是前置 |
| 原策略参照 | 原生primitive-actuated独立20秒报告failure=null；老HOST全131.48秒无跌倒，但峰力14.91mg与XY1.275m两旧门失败 | 可复用来源/协议证据，不倒灌新MuJoCo-TYX结果，也不再为这份计划重跑旧参考 |

现T3静态P16/C32不足以直接代表G1完整候选；父已要求通用动态compact研究，保全部候选/排除/身份，不只挑脚或删自碰。F0必须输出真实编译nq/nv/nu/nbody/ngeom、所有joint/type/limits/a/D/dry、所有geom/condim/mask、4排除与完整layout/capacity说明；本文件没有运行编译或推测native实际接触峰值。

## 7. G1按门推进，避免长窗空转

| F阶段 | 固定任务 | 成功门/后续动作 |
|---|---|---|
| F0来源与admission | 同一primitive-actuated XML/MJB、完整29motor/35DOF；缺项先明确；不要重跑旧无接触全套 | 所有字段完整进入新backend，预分配/动态compact没有截断；明确CPU编译与TYX步进边界 |
| F1同入步留出 | 原默认姿态、motion frame0、frame200；原非零速度/控制，接触和joint行全部比较 | A–E原工程门，含足地6D wrench/根–关节耦合；不依靠修改初态消掉异常 |
| F2原策略2秒支撑 | 原frame0 q/v、同ONNX/YAML/原限幅；物理h=.001与.002、策略.02s整数decimation；先interpret短专项，再CUDA完整2秒 | 全程无拒步/NaN，root高≥.35m/倾角≤1rad，足地实际平均竖向力[.5,1.5]mg、峰值≤8mg、最大joint RMSE≤.5rad/参考XY≤1m；范围内继续，不追逐位轨迹 |
| F3外扰留出 | 同初态/原策略；t∈[.5,.6)s root COM世界外力±.1mg沿x，随后自然解除；2秒 | 不跌倒、完整load/torque/控制时刻正确；同源CPU也走该扰动。若native自身失败，分类为策略/plant边界，不调参数追绿 |
| F4静态支撑工具 | 原默认目标PD作为独立diagnostic，原gain/姿态不变；原先曾失败，不能假定静止必稳定 | 如成功才用stand门：tilt≤.35rad、rootXY≤.1m、footXY≤.08m、平均载荷[.7,1.3]mg、双脚>1N比例≥.9。失败不阻止已过通用模块，也不重跑长前缀 |
| F5短舞蹈到20秒 | F2/F3通过后，同一原始轨迹前20秒，所有状态由物理产生；自然离地/重入按native实际行为核 | 原政策门、足地/非足接触分开、控制/限位/滑移诊断齐；完整131.48秒另列后续独立批次，不把20秒叫全动作 |

F2/F3 h矩阵是新MuJoCo-TYX的工程验证，不借旧HOST h=.0005通过；若步长本身是原因，保原失败，用同源CPU与首差异判定后单独登记h=.0005，不静默延长窗口或调质量。F2是反馈支撑组合，不声称默认PD静态平衡成功。

每步保存实际joint限位超越量（soft limit非严格硬边界）、力矩限幅、contact/limit/dry合载、源/设备时间。力矩限幅按FP32工程门3e-5相对+3e-5Nm，不能套旧HOST1e-9。载荷记录包括支持/滑移时间、每只脚的切向速度与累计滑移、接触扭矩，不能把原33个shape全部称足地。

能量诊断按实际M/速度记录动能、重力势、驱动功、阻尼功、contact/dry广义功与积分余额；接触力时刻t和运动t+h明确分开。受驱软接触不要求总能量单调，不把未建模contact潜能省略后声称守恒。短窗核同律CPU的功率/能量趋势，长窗报告非物理爆增或原控制/力矩限额违反的首时刻；未冻结的新能量数值门不能作为无限证明前置。

## 8. 实施前交接与防止循环

父冻结的最小API应可提供：全部原生约束行owner/type/id/上下界/参数、contact-frame完整6D wrench、双侧world force/COM torque、原始edge λ与解码关系、同次H/b/J/solver停止原因、实际candidate/contact/efc容量和accepted mask。新增字段不能靠测试反复调用产品来补计数。

所有正式输入、h/m/窗口和工程门在首次生产执行前固定。若原生CPU本身反驳某“必须静止/必须承载”的测试假设，保原RAW失败并修正错误oracle分类，不改产品或同律参数遮掩。T3b已示范接触序号不是跨精度物理身份；摩擦比较继续保完整几何双向匹配、重复点数量及原阈值，绝不按force挑配。

两轮没有新增可调用能力、失败闭环或可归因测量时，停止扩大文档/长窗，提取最小同入步差异：geometry→frame/J→参数R/aref→边解码→共同solve→积分→策略。已绿同源组不重跑。API冻结后另行授权NEW tests编码；当前没有任何project import、collection、numerical probe、pytest、lint/format、build或native执行结果。


## 9. T4 API 冻结后的纯测试交接

当前状态覆盖开头研究时的 API_PENDING：**CODE_READY_TESTS_PENDING / UNCOLLECTED / UNEXECUTED**。父已冻结 [mixed constraints 计划](../superpowers/plans/2026-11-10-mujoco-tyx-mixed-constraints.md) 2fe2e3f8 和逐 EFC typed receipt；父唯一项目执行 owner。作者源码 checkpoint 54e1fceba2d930c62c9f21547fc65ead11364cd1 只新增两个测试文件，不修改产品或旧绿组。

| 文件 | 纯源码规模与 SHA256 |
|---|---|
| [test_mujoco_tyx_constraints.py](../../tests/test_mujoco_tyx_constraints.py) | 839 行 / 21 个 test 定义；d7026ef2ecedbe7fd46f86a8cf8b332878f2a81067f81819f47ca8048c1c5938 |
| [test_mujoco_tyx_constraints_protocol.py](../../tests/test_mujoco_tyx_constraints_protocol.py) | 455 行 / 15 个 test 定义；d8e7e8548d751314557bcf058a7c369b89ba07d72d3aa286102e7aa3d728b1cd |

按源码 decorators **手工展开**为主文件164、protocol42、合计206个预计节点：97 interpret、97 CUDA、12个不带mode的独立解析/准入节点。没有 collection，这不是实际节点结果或通过数。每个backend保留12个完整1秒窗口（D01四个h/m、D02两形状八个h/m），没有把原1秒缩回短窗。公共ENV和G1由父独立消费者负责，不重复实现。

源码落实：

- 新native oracle读取全部原始efc type/id/J/pos/margin/R/D/KBIP/diagApprox/aref/force；接触按geometry/normal完整双向匹配，保重复点 multiplicity，不按force选对应。再映射compact contact source_id核全行，不要求跨精度序号相同。
- 独立≤6行box-QP枚举lower/free/upper完整faces，另有4个固定手算参照；全部实际行另重构小H/b和原5e-5 KKT，验证完整Jᵀf及physical M质量方程。physical qacc与M+hD积分加速度分别核对；不读取solver face scratch作为期望。
- 全1/3/4/6、signed滑/扭/滚、native MINMU、source-only解析阈值、偏心/斜面/双活动body、双侧limit和6个free-root dry行保留。追加父指定的五个 **单步** law分支：default、direct solref=(-1000,-30)、width0 solimp=(.2,.8,0,.5,2)、decreasing solimp=(.9,.2,.002,.5,2)、REFSAFE solref=(.0001,1)。没有新增长窗或变更原容差。
- D01/D02实际、同入步native与持续native均先按双方body ID保存完整COM wrench。支撑/冲量行为门明确选择D01的tip、D02的root；capsule-box的moving body可能是geometry1，绝不固定把geometry2平台反力叫支撑力。所有body的完整wrench仍保存。未指定目标body的诊断和只是各自COM wrench分量之和，不冒充一个共同参考点的总空间wrench。
- D01显式检查初始limit激活、随后退出和contact分离/重入，并对持续CPU事件时间；D02保原支撑、无接触区间、重入时限、末窗载荷/速度/高度门。完整q/v/control/力/count每步保存；首末、脉冲边界、接触数或EFC(type/source/side)变化时保存完整receipt/oracle。失败增加已执行attempt的readback scratch，不重跑producer或solver补数据。
- Protocol保每种condim恰满/超一、inactive完整6D J、稀疏W4/reset/clone/输入顺序、预算耗尽、晚下载故障和禁止CPU/Warp/HOST/linalg fallback。新有限input_q/input_v单world篡改要求全批拒绝、旧发布原样、恢复真实download后retry等于clone。
- 通用compact正控保40个有限marker及最后一个真实接触、native body exclude，不裁到旧32槽；另61个geom构成60个condim6接触/600 EFC，静态布局合法但实际行数超过512时必须全批拒绝。不同world中无接触者也不得抢先发布。hard requested cap及potential pair超域另核显式拒绝。
- unused contact_map两身份列为-1、计数列0；inactive contact的first=-1/count0。这个字段勘误不改变物理模型或阈值。G1表的左右foot capsule已按原资产核实各7个，33个总shape及原资产SHA不变。

本轮只有文本/文件IO/Git：两新文件与作者串逐字回读一致；精确两文件staged scope、git diff --cached --check成功，源码commit后clean。**未运行import、collection、算术probe、pytest、lint、formatter、compile/build、install、native或CI。** 首次静态/项目执行及任何实际缺陷判断交父串行队列。任何首次native对照反驳某行为假设都应保存原失败并分类，不能倒改输入或放宽阈值追绿。


## 10. 首次执行后行为 oracle 勘误（26/27 原失败保留）

本节仅记录父串行执行的既有证据与窄测试修正；作者没有执行新的项目命令。原证据位于父 artifacts/p65-mujoco-tyx-t4-20261110-01 的 26-short-interpret-qr、27-short-interpret-remaining；保存日志/JSONL均未修改。run26报告6 passed后signed slide失败；run27报告71 passed、16 failed。这里不把尚未执行的修订称为通过，也不处理父负责的2个oblique force差异和4个solver拒绝。

- A03原末步必须摩擦功率<0假设错误。原24步保存中tick0–14两端都有contact，tick15–23 TYX和持续CPU同时离面；第一步Fx=15.3053789139 N与input vx=-.40000000596 m/s反向，末速vx=-.25362700224 m/s。修订只读同一24步完整saved actual receipt，以各步input_v/J/切向wrench计算h加权摩擦功，要求真实非零摩擦、全窗功<0及原末速下降；原fixture、24步与全部逐步native门不变。
- A04的±1.2mg原64步两端在tick29–41、50–57、62–63出现合法离面，其他段接触/重入一致。将“每步count必须1及末步normal>0”改为初始接触、整窗真实支撑和摩擦；count仍只能0/1，原高载末速方向/幅度及全部原生逐步检查保持。不是强行延长接触或改正常物理输入。
- A07 axis1±2并未离面：原12步两端均有contact，但native和TYX的纯couple分量全为0；实际切向力及lever×force的COM矩承担阻转。axis0原生末步有约±.0002506 Nm纯couple、torsion约±.2149 Nm，原非零门仍有效。修订读取同一完整窗，计算接触点切向功加纯couple角功，不再次加lever矩重复计功；要求实际COM torque和总摩擦耗散。对native整窗达到原1e-5门的couple分量，actual也必须达到原门；完整6D wrench、10 EFC/J/R/aref、运动和所有逐步force容差不改，不凭“旋转”要求native没有的纯力偶。
- 5个准入负控原来访问不存在的exception.code，实际公开结构为MuJoCoTyxUnsupported.issues中的MuJoCoTyxIssue(code,detail)。改核非空issues及每项code/detail，原拒绝类型和所有非法source保持。
- 原预算负控两collinear normal的保存H=[[1.1111111111,1],[1,1.249219238]]、b=(-34.81,-10.0570679012)，无界解虽然第二项负，但非负最优为(31.329,0)。当前命名算法每次只释放最大违反的一行，因此一次face就能正确完成；原DID NOT RAISE是错误预算假设。预算负控单独改为原3slide球与两个正交plane，center=(.099,0,.099)，h和law不变。J沿x/z、M=I，H对角且两b负、两最优normal力正；一次只释放一行后另一独立KKT仍违反，故source iterations=1必须不足。测试同时核native source与capabilities的budget1、真实attempt status7/diag budget1及整批原子拒绝，不mock、不增加预算；该新负控输入须由父首次验证，不能追认旧预算例失败。

父首次修订验证：41-revised-short-interpret 实际24 passed / 184 deselected / RAW_EXIT0（53.52s），涵盖上述三行为函数全部参数、五准入负控、预算负控、两mixed同入步、晚下载回滚及公共ENV。静态37/38原失败保留，39只format，40lint0。此结果不替代D01连续窗或CUDA验收。
