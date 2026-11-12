# 材料耗散后继研究：2026-11-13

状态：**RESEARCH_ONLY / NOT_IMPLEMENTED / UNEXECUTED**。只研究、源码文本、
手推和 Git；不修改当前物理或测试，不授新执行。本文提出独立材料功能，
不是修饰 D4 原轨迹的验收结果。父需另行冻结一个方案、接口与测试后才能实现。

## 1. 事实、证据和推断分开

父报告 run50 的固定 N1 完成 3s / 3072 步，入层、卸载、离层、再接触事件
均发生，长状态参考、几何和防贯穿门通过；但两预定支撑窗平均力为
24.839 / 5.414 N（目标16N），刚体法向 RMS 为 .584 / .478m/s，
中心 RMS 为 .340 / .384m/s（门 .1）。因此**持续稳态支撑资格失败**，
不是没有任何功能。run46 第301返回步的原短速度门失败也单独保留。
本文未重新读取/运行整条轨迹，数字来自父同次实际结果。

从当前源码能确定：

- [DistanceConstraints](../../src/phymeo/constraints/distance.py:13)只保存
  indices/rest_length/compliance；没有显式材料阻尼参数。
- [project_node_distances](../../src/phymeo/backends/_tyx_node_kernels.py:58)
  和[原 traced 行](../../src/phymeo/backends/_tyx_distance_trace.py:12)
  使用 C+alpha_tilde*lambda，没有速度/应变率耗散项。
- [D1 _stage_step](../../src/phymeo/backends/tyx_deformable_batch.py:767)
  先预测、ordered projection，再
  [reconstruct_nodes](../../src/phymeo/backends/_tyx_node_kernels.py:24)重建速度。
  有积分/有限迭代/FP32的数值耗散，不等于已标定的材料粘性。
- [NormalLayer](../../src/phymeo/backends/_tyx_normal_layer.py:7)已经有接触耗散。
  在未被截断的分支，法向力是 k*compression*(1-d*rate)，d 的单位 s/m；
  等效速度项系数 k*compression*d 随压缩量变化，离层时为零。
  它只观察接触点相对法向速度，不能代替所有内部材料应变率的耗散。
- [D3共同候选/发布](../../src/phymeo/backends/tyx_deformable_contact.py:605)
  已有完整节点材料 impulse、固定反力和双 owner 发布；新耗散应加入这里的
  候选及账本，不得提交后再改 accepted velocity 或伪造能量 credit。

**推断而非定因**：缺少显式内部耗散是值得补齐的通用功能，也可能使某些
振型衰减慢；但仅这一条失败不能证明某个粘性模型一定正确或一定能在
0.75s前达到稳态。几何非线性、接触分离、不同固有振型、施加载荷与积分
也会影响窗口均值。不得据两段均值直接反推“所需阻尼”。

一个必须保留的解析反例：在理想对称竖直切片，角点为(±1,±1,0)，
中心为(0,0,z)，单边长度 l=sqrt(2+z²)。若每径向边采用轴向粘性 c，
则中心的等效竖直阻尼是 **4c*z²/(2+z²)**；在 z=0趋零。
弹性竖直回复力也由几何非线性产生，并非四根竖直弹簧。
这是该受限切片的手推，不是实际轨迹拟合。它直接否证“任意正 c 都能
消除平面附近的全部振荡”的说法。若目标需要弯曲/剪切耗散，应有对应材料
自由度和应变，不用全局速度缩放偷偷补齐。

## 2. Primary 依据及适用边界

[XPBD 原论文 §5、式19–26](https://mmacklin.com/xpbd.pdf)从约束应变率
的 Rayleigh 耗散势出发，给出与 compliance 联合的耗散更新；也明确区分
积分本身的耗散与另行建模的约束阻尼。下面 A 复用这个建模入口，
不是引用论文保证当前任意有限迭代、接触或 FP32 都无条件稳定。

[Newton v1.7.0 的 solve_springs](https://github.com/newton-physics/newton/blob/v1.7.0/newton/_src/solvers/xpbd/kernels.py#L320)
分别输入 spring_stiffness 与 spring_damping，使用 kd/(ke*dt) 项和相对轴向
速度；本机固定环境同名文件的 solve_springs:320–385 已纯读核对。
这里参考参数/更新结构，不把不同速度更新时间和并行松弛语义直接照搬。

[MuJoCo 3.3.7 XML 文档](https://mujoco.readthedocs.io/en/3.3.7/XMLreference.html#body-joint)
把 joint damping 作为速度线性被动力，并说明 Euler 的隐式处理；
[flex elasticity damping](https://mujoco.readthedocs.io/en/3.3.7/XMLreference.html#flex-elasticity)
则以时间量纲系数缩放刚度。二者说明参数单位与积分位置必须绑定，
**不是**宣称 MuJoCo 的 joint/flex 实现可直接替代 Phymeo 距离材料。
未采用博客/论坛作为算法依据，也未以最新 main 代替固定版本。

## 3. 共用的本构、客观性和反力约定

每条声明边 e=(i,j)，非退化 d=x_i-x_j，n=d/|d|，
C=|d|-l0，s=n·(v_i-v_j)。材料粘性系数 **c_e≥0，单位 N s/m**，
耗散势 R_e=.5*c_e*s²，内部力为
f_i=-c_e*s*n、f_j=-f_i；功率为 -c_e*s²≤0。
它与 NormalLayer 的 d[s/m] 不是同一个参数，禁止共用无单位 kd。

这是本文设计的共同本构定义：

- 加共同平移速度不改变 s；加瞬时刚体自转 Ω×x 时，
  n·(Ω×d)=0；共同坐标旋转使力协变。不能采用 -c*v_world 或
  v*=常数因子冒称客观的内部耗散。
- 成对中心力保总线动量、关于同一施力时刻坐标的角动量。
  固定点速度为零时，固定支座功率为零，但其**反力 impulse 不能丢**。
  第一片不支持运动固定点；以后支持时需显式计 prescribed motion 的外功。
- 零长/方向条件不足的非零粘性边必须有显式拒绝状态，不能沿用旧
  弹性零长 world-axis fallback 声称旋转客观。c=0的旧路径保持原语义。
- 本构的瞬时客观性与离散积分误差分开。特别是有限转角的端点弦速度
  未必等于瞬时刚体角速度；任何方案都不能只凭“用了相对速度”就宣布
  任意离散旋转零耗散。测试要同时保刚体速度零模与跨步旋转误差。

## 4. 两个可选实现；先选一个，不能叠加追绿

| | A：XPBD 行内 Kelvin–Voigt | B：候选上的隐式边粘性子步 |
| --- | --- | --- |
| 最小新增能力 | 每条原 distance 行有独立材料粘性；弹性/粘性共同影响位置和重建速度 | 一个可复用的速度耗散算子，显式作用于未发布材料候选 |
| 参数 | compliance a[m/N] 与 c[Ns/m]，不能将法向接触 d传入 | 独立 c[Ns/m]；不靠 compliance 推导或修正粘性 |
| 时间积分 | 联立约束残差，最接近现有 position→velocity 合同 | 固定本次候选几何，按原 stable edge 顺序作一次隐式子流组合 |
| 主要成本 | 改实际行、traced行、停止残差与参考；有限迭代的力/能量拆分更难 | 新小核/候选 velocity/impulse 记录；必须公开新的 operator-split 速度语义 |
| 不能宣称 | 只复制 Newton 一行就得到相同解或无条件非线性稳定 | 与全网络联合 backward Euler 完全等价、顺序无关或二阶积分 |

### A：行内方案的精确候选

记 a_tilde=a/h²，gamma=a*c/h，A_e=G_e M^-1 G_e^T，
G_e=(n,-n)。论文形式的 ordered 行为：

    delta_lambda =
      -(C + a_tilde*lambda + gamma*G_e*(x_candidate-x_entry))
      / ((1+gamma)*A_e + a_tilde)

lambda 在本步累积，使用当前行方向更新两节点。c=0回到原行。
必须明确 x_entry 是整步初态，而不是本次 sweep 的起点；否则阻尼随迭代
重复计时。原 residual 也必须改为同一耗散残差，不能弹性 residual 已小就停。

有限转动时 G_e*(x_candidate-x_entry) 是弦近似，可能出现纯刚体转动的
离散耗散；不能隐藏。首片先用精确线性振子/旋转速度场反例界定离散误差，
若不满足工程客观性门则不选此片，不再临时更换率定义。
a=0时此参数化 gamma=0，不能承诺硬约束的独立粘性；它须另有域定义。

实际 material impulse 仍是**所有 ordered 行真实方向×delta_lambda/h**，
不能用最终 lambda×最后方向替代。若只记录合并 lambda，必须叫总材料
impulse；不能把终点 c*s*h当成“已实际施加的 damping impulse”。
弹性力、耗散功率和积分缺陷需分别记录，不能由一次末态 λ 伪拆分。
这些是采用 A 前的真实共享接缝成本。

### B：建议先实现的独立最小片（明确为一阶分裂）

在 D1 原 elastic candidate 的 x*、v* 上，以同一 x* 冻结每条边 n。
每条边只执行一次 h 长度的隐式轴向耗散子流：

    A = w_i+w_j
    s_before = n·(v_i-v_j)
    j = -h*c*s_before/(1+h*c*A)
    v_i += w_i*j*n
    v_j -= w_j*j*n
    s_after = s_before/(1+h*c*A)

j 是 N s。固定点只累积反向支座 impulse，不更改 v。
一个完整 ordered pass 是多个边耗散算子的 Lie 分裂；
**不能再重复200遍同一个 h** 来假装收敛，那等于改变物理耗散时间。
不隐含网络全局解；若后继要联立解，应另立总 impulse / residual 协议，
不能偷偷把本片改成任意 sweep 次数。

在这一子步固定几何、静止支座与精确算术下，手推单边有：

    K_after-K_before
      = j*s_before + .5*A*j²
      = -h*c*s_after² - .5*A*j² <= 0

完整 pass 可逐边求和。h>0、c>=0时分母至少1，独立耗散子步不受显式
h*c*A<2的稳定性限制；这不是整个非线性系统无条件稳定。该性质只针对
新增耗散子步，不证明整步弹性+接触+
外载能量单调。必须保存实际 FP32 前后 v、j、方向、反力与浮点余量；
不把上述理想恒等当实测，也不以总能量“差不多”覆盖符号错误。
此算子在固定 x* 上严格保瞬时刚体平移/转动速度零模；整体积分的转动误差
仍由主积分器和先前候选产生，分别检查。

选择 B 的理由是独立可证伪、无需重写原 nonlinear elastic 行、不会把新
c混入接触 d。**代价必须公开**：发布 v_new 不再仅由旧位置重建公式决定；
receipt 要同时保 v_reconstructed 和 v_dissipated，旧审计不能默认为相等。
这是新离散材料功能，不是事后滤波修复状态。若父要求绝不改变该速度合同，
则选 A 并承担其完整行/残差工作；本文不自动授编码。

## 5. 设备数据、共同发布和最小文件责任

提议新增不可变 `DistanceDissipation`（暂定名）：
original edge IDs/order + c_e[Ns/m]，逐 world显式绑定原 model/source/revision。
默认 absent/全零走原路径；不能按 body/world/案例名分支。
不把 c静默塞入 StepInputs 外力，也不使用 HOST 当前坐标计算粘性后回填。

B 的初片接线：

1. D1 standby 内完成原材料 projection/reconstruction，保原全部 trace；
   新 TYX 核从同一 standby x/v计算冻结方向、冲量和耗散后速度。
2. 新 candidate receipt 保每条边 pre/post rate、j、固定端 impulse、
   kinetic before/after 与实际 arithmetic 输入输出；总材料 impulse 分成
   elastic ordered impulse 与 damping impulse，旧原始字段不能含糊改名。
3. D3 prepare_publication把两项与外载/接触合并进固定反力：
   R_fixed=-(h*F_fixed + I_elastic + I_damping)；完整动量/力矩审计使用各自
   实际施加时刻。只在节点与刚体全部候选、审计和不可变receipt完成后一起swap。
   拒步丢 standby，不影响旧 accepted velocity/history/query。
4. 正常 query继续借用接受态；next step的接触率读新已接受速度。
   当前接触 entry-sampled law保持原值，不事后重算力使其适配新速度。
5. 每边 c及新 trace、每节点 candidate scratch 的 dtype/shape/峰值字节式
   在编码前冻结；readback/upload/launch据实计费。保原memory限额，
   exact/one-less/late allocation/UNKNOWN_RELEASE全测，不能借重复释放补预算。

推荐独立 NEW参数/核/测试/consumer；父拥有
tyx_deformable_batch.py、_tyx_deformable_buffers.py、
tyx_deformable_contact.py、builders/deformable_scene_lowering.py、公共exports
和checkpoint/source binding接缝。A还必须修改两个原distance核及残差，
B保持它们数学不变。具体 allowlist、预算与 schema要在选择后另冻结，
本研究不修改这些文件，也不承诺无需任何共享改动。

## 6. tests-first 最小多案例库与可证伪门

以下是**待父冻结的独立模块矩阵**，没有执行或据输出调参。
先做1–8，再做9–12；原 D4 输入/窗口/门不改，仍保 FAIL。

| 组 | 固定可复用输入/独立参考 | 必须能否证什么 |
| --- | --- | --- |
| 1 零参数兼容 | absent 与 c=0；原 D1/D3小已验fixture，同入步 | 旧数值状态/力学receipt字段不因新开关改变，新增参数身份仍如实绑定；默认无新阻尼力、隐式额外步或隐藏耗散 |
| 2 单边两自由质量 | x=(-.5,0,0),(.5,0,0)，m_i=1kg、m_j=2kg，v=(.25,0,0),(-.125,0,0)，c=2Ns/m；正反相对速度，h=1/128、1/256 | B 与上面的独立闭式解；两侧相反 impulse、COM不漂、轴向衰减符号正确。不是仅看“动得慢” |
| 3 固定支座振子 | rest1m，m1kg，compliance1/64m/N，初伸长1/8m，v0=0，c=0和16Ns/m（由线性临界阻尼量纲选定），两h、T1s | 独立2×2离散递推及连续振子解；固定反力与动量一致；不能把固定点反力清零，不能用改质量当阻尼 |
| 4 两边非共线链 | 3个不同质量节点、2条不共线边，所有节点活动，非零一般速度 | 完整网络 impulse/力矩、每个真实子流能量；B按声明order对照，不能冒称与联合BE或任意重排完全相同 |
| 5 刚体零模 | 至少4个非共面点；v_i=V+Ω×(x_i-origin)，非零V/Ω；共同旋转/平移 | 粘性增量近零；全局速度缩放会失败。坐标变换时固定边/模型一起变，不能把真实支座运动误作零模 |
| 6 纯应变协变 | 对组2/4施统一正交变换、共同translation/velocity boost及storage permutation，stable edge顺序不变 | impulse随坐标旋转，能量/应变率不变；不靠world坐标或storage行号选法则 |
| 7 平膜零切向投影 | 原五点膜仅构造一个无contact材料snapshot：z=0、中心纯竖直v；另独立z=1/8快照 | 前者轴向粘性应为零，后者核4c*z²/(2+z²)；不能为了N1稳态把此合法零模改成全局drag |
| 8 参数/退化负控 | c<0/NaN、长度/edge IDs错、foreign model、非零c零长边、全fixed静止边、未支持运动支座 | 执行前或明确conditioning拒绝；完整边不丢、未支持不能当成功零耗散 |
| 9 h与质量变化 | 同组2/3参数不变；两h及不同质量，独立质量缩放公式 | 衰减按物理 c与h改变而非按iteration计时；细步对连续参考的一阶收敛诊断，不要求所有非线性误差每步单调 |
| 10 多world与事务 | sparse worlds/重复localIDs、不同c；末world真核后/receipt前失败；reset/restore后真实step | c/state/source不串world；两owner无部分发布、旧query/receipt保留，retry是实际推进 |
| 11 原D3短耦合 | 球和斜胶囊两种原有限接触；显式新材料参数，非零entry速度 | 接触法则仍按原入步速度；节点粘性+材料+接触与刚体反力全账本闭合，不能靠接触力重算修补 |
| 12 新材料应用窗 | 新身份N1-damped，几何/h/质量/外载/窗口保持旧N1；c仅取组3的独立材料标定约定，首跑前登记 | 对新材料功能另验支承/离层/再接触。即使本窗仍失败，也不否定已过的耗散模块；不得改回旧N1结果或扫描c追绿 |

建议执行前固定的工程判据（不是已有验证）：

- 单边/短网络与独立 FP64同离散oracle的 v/impulse 使用原短状态量级，
  初拟 atol2e-6、rtol2e-5；线动量和固定时刻角动量残差分别保原
  FP32 impulse审计方式及单位，禁止随绝对平移/整体boost放宽。
- B每个子流 ΔK≤其实际FP32操作的预声明工程误差，且
  ΔK+h*c*s_after²+.5*A*j² 的残差独立核；需给原始有符号值。
  能量门不能只比较最终总K或被巨大共同平移K掩盖。
- 零模直接核每边rate/impulse，不只核总量抵消；参数零控不含粘性效果。
- A若被选，先固定含阻尼同一residual和完整λ/方向证据，再定有限迭代门；
  不把论文的收敛方程当已有有限步oracle，也不复用B的精确子流能量恒等。
- 模块一次首失败只据具体字段定位；若真实违反客观性/功率符号/原子发布，
  先修实现，不调c。若应用稳态仍失败，独立报告模型范围/模态原因；
  不连续添加阻尼机制或改变接触率定义直到N1变绿。

## 7. 建议下一裁定

优先选择 **B的独立隐式轴向耗散子步**作为最小通用功能，
以单边解析、刚体零模、固定反力、网络能量和事务完成模块验收。
它有清楚的离散/顺序边界，需要公开新速度phase，不能悄悄插入旧契约。
A保留为更紧密的粘弹性方案，适用于父选择保持position-based整体组织的情形，
但需要同时解决残差、率定义和实际力分解。

两者都不是“增加阻尼即可解决N1”的承诺。当前已证明的是原无显式材料粘性
模型能完整运行并发生功能事件，**没有满足原两个稳态窗**。旧首失败、
run50功能证据与严格/诊断标签完整保留；只有另获授权、完成新方案tests-first
及两后端/包外资格后，才能在功能表新增“材料耗散”。
