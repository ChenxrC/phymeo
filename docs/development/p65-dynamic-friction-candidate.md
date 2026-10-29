# P6.5 下一摩擦切片：解析候选与必须保留的反例

状态：2026-10-29，数学准备，**尚未冻结生产合同、未派实现、未运行新native摩擦程序**。上一自动法向产品4ff19ea本地验收结束，准确29165ca CI因job时限取消，aa8ec42仅修CPU工作流时限；先闭环准确远程结果，不重做旧物理审阅或全量。

用户已要求框架级可复用摩擦与接触；本页只确定需要验证的物理语义，不是递布demo或完成声明。只读数学agent `dynamic_friction_law_probe` 已结束，无文件修改或native测试；主agent另写[独立Fraction计算](../../scripts/probe_dynamic_friction_law.py)，[原始有理结果](../benchmarks/p65-dynamic-friction-algebra.json)的静支撑、材料斜撞和两个交叉接触候选检查全部通过。脚本不导入phymeo、不求自动接触，只代入明确的候选冲量检查方程；不能计入生产或后端测试数量。

## 当前真实缺口与三条路线

现有 DynamicRobotNormalContactProgram 分held位置载荷和terminal速度冲量，真实原生推进、末阶段配置保持，支持固定朝向无限平面、全prismatic机器人与全动态distance材料。旧法仅μ=0，已有 TerminalImpulseProgram/账本可执行任意点向量，但不等于新摩擦定律已经成立。

1. **仅terminal摩擦**：若容量取μη，静支撑η=0时完全没有摩擦；即便取μ(p+η)，只能消末速而不消本步切向漂移，不足以验收持续承重。
2. **A，分阶段Coulomb法（当前优先验证）**：初态几何闭合行held共同求p_n和p_t，容量μp_n、速度用真实pre；初态开放行held切向为0。末闭合集合共同求η_n/η_t：初态闭合行切向容量μη_n，新建立行容量μ(p_n+η_n)，速度用真实post。每阶段共同保留同域/机器人/非平行面的全部交叉响应。
3. **B，统一总冲量末速法**：闭合行切向全部放held、开放行切向放terminal，以末post速度求总Coulomb，并反复重算两阶段。这能改变A的末态方向反例，但可能使held摩擦在真实声明路径上顺滑动做正功。它需要另一个明确的物理合同，不能仅以全步能量下降或“更隐式”直接放行。

首版候选μ_static=μ_dynamic，任何路线都不得只移除旧μ=0或全prismatic门禁。A有阶段被动性论证，B需要放弃/替换该语义。最终设计必须写清选择及限制，不能同时宣传两种不等价的定律。

## A的适用性与证明前提

零数值误差、radius=0、全prismatic和原生速度重建对应端点位移时，held法向互补给旧闭合行p_n*u_n_minus=0，新建立行等于-p_n*g0/h<=0；held切向Coulomb给p_t·u_t_minus<=0。terminal的完整质量响应W保留全部行交叉项，ΔK=z·u_plus-zᵀWz/2<=0。容差下必须再用原有精确冲量、配对力矩、材料路径、全步能量和角动量门槛核对，不能把上述零误差推导当任何原生base均可接受的证明。

分阶段容量不重复消费：闭合行μp_n+μη_n，开放行held切向0、末阶段μ(p_n+η_n)。但总切向冲量一般不满足单一末态Coulomb律，也不构成整个离散步的单一最大耗散解。不得隐去这一点。

初态分类用几何区间，不用history.active代替。初态开放g0>0时在节点中心/机器人平面投影施held切向会产生g0*n×p_t力偶；非零半径也产生r*n×p_t力偶，当前无自旋节点不能自行接收。因此下一候选摩擦行先显式限radius=0。厚度/旋转/有限面后续另补真实材料点与角响应，不把零厚度候选当真实夹爪完成。

## 独立场景一：支撑与低摩擦滑动

h=1/4，节点m=1、机器人M=2且只沿Y平移；初态共点静止，n=+Y。节点g=-Y、原外力+X，机器人原effort=+1N。μ=2时精确held p_n=1/4、p_t=-X/4，η=0，全部状态保持零。terminal-only即使把末vx消为0，每步x仍漂移1/16m，这是拒绝该路线的实际精度问题。

相同输入改μ=1/4，held切向=-X/16，末vx=3/16，x=3/64，η=0，必须滑动。后续生产验收要扩到T=2、h=1/16和1/32：低μ参考v_x=3T/4、x=3(T²+T*h)/8，高μ静止；独立离散递推及连续参照分开，不能用提高摩擦或pin让低μ对照成功。该粗步位移误差不替代最终递布精度门槛。

## 独立场景二：材料链的新接触斜撞

h=1/4，机器人M=2仅沿Y，初q=V=0；两节点m=1，A=(0,1/8)、B=(0,9/8)，速度均(1,-1)，g=0；距离rest=1、compliance=1/16，n=Y，μ=1/2。

held p_A=3/7、切向0；A1=(1/4,-3/56)，B1=(1/4,51/56)，q1=-3/56。真实材料投影不能当free粒子。terminal η_A=1/3、t_A=-8X/21；末v_A=(13/21,-8/21)、v_B=(1,-6/7)、V=-8/21；K+U=1135/882，Y动量=-2。X反力由基座承担，不能宣称全系统世界动量守恒。后续须在NumPy和实际interpret/CUDA逐一执行同一原式参照，预设候选也不能冒称自动求解通过。

## 非平行多面反例：A与B必须明确区分

h=1、全部质量1、无外力，两独立平移平面：n1=(0,1)、μ1=1/2、g10=0；n2=(-3/5,4/5)、μ2=0、g20=28/5。节点x0=0、v0=(21/10,-9/5)，平面沿各自法向速度(1,6)，第二平面初锚点=-(28/5)n2。

A候选held p1=p2=1、t1=(-1/2,0)，节点pre v=(1,0)，机器人pre速度(0,5)，末gap均0；terminal η1=0、η2=14/5，节点post v=(-17/25,56/25)。terminal ΔK=-196/25，阶段律成立，但总切向冲量与末滑速点积=17/50>0。不能将这个A的合法分阶段结果宣称为总末Coulomb结果。

B在同一初态可给t=201/550、p1=241/275、p2=72/55、同一η2，末地面切速0且切向总容量内；然而pre切速42/25，held摩擦功=4221/6875>0。全步能量仍下降，说明仅总能量门禁发现不了该局部语义变化。不是B数学无解，而是它需要不同的声明与接受条件。

## 生产设计前的下一步

- 主agent先核对aa8ec42准确CI，关闭上一法向交付；不重开旧review或全量。
- 用现有公开native replay和TerminalImpulseProgram运行两独立场景的**预设**向量候选，核对材料/机器人pre及post、反作用一次、配置保持、paired moment和后端执行；这仍只是合同前置，不是自动摩擦完成。
- 冻结A或明确改选后的接口、单位、支持域和数值门槛。建议独立DynamicRobotFrictionContactProgram与owned系数映射，复用collider几何、terminal执行和原式账本；旧normal行为不改，不能案例脚本复制算法。
- 接口须公开两阶段法/切冲量、事件与regimes，history只作显式事件/诊断，不当隐藏施加载荷。分离所有冲量清零，基准初态几何不确定拒绝；跨行法切耦合必须完整native重算。
- 明确正/低μ、材料、共享/非平行多面、变换顺序、输入非法/退化、预算失败、已有best后的native/trace/terminal/close失败和后段全帧history恢复retry=fresh测例；一个实现owner、阶段一次fresh review和必要工程证据。

新的spec/plan尚未写，不派生产实现。有限摩擦切片不完成H1/H2或双臂；有限特征与FK CCD、弯曲/自碰撞、真实承重、控制/交接和同源Three.js仍在同一持续任务范围。

## 原生执行核对（2026-10-29）

[scripts/probe_dynamic_friction_native.py](../../scripts/probe_dynamic_friction_native.py)使用现有公开SceneLoadReplay/TerminalImpulseProgram，依次在NumPy、actual interpret、actual CUDA核对各6例，共18例成功。每例先完整held replay，再以实际末配置投影作用点完整重算并施加给定terminal向量；不是求解摩擦冲量。预设值来自上面的独立分数解，不调用任何新自动摩擦程序。

三种模式均覆盖：高μ静支撑、低μ滑动、terminal-only漂移反例、材料斜撞、非平行多面候选A与B。真实native状态匹配解析位置/速度，末阶段前后位置bitwise一致，第二次pre状态/history与选中held一致，原输入不变；MaterialPathLedger/FrozenRobotLedger只验pre，TerminalImpulseLedger验post。真实tyx模式/FP32提升/launch均检查。A的总切向冲量点乘最终滑速仍为17/50>0，B的held切向功仍为4221/6875>0，验证不能混称两种定律。

原始记录：[NumPy](../benchmarks/p65-dynamic-friction-native-numpy.json)、[interpret](../benchmarks/p65-dynamic-friction-native-interpret.json)、[CUDA](../benchmarks/p65-dynamic-friction-native-cuda.json)。每例两次重放，记录的execution仅最终一次；不冒称全尝试汇总，不作为性能或长轨迹。全部原生操作在未改产品131文件hash ac777319d0b8fbc03adab92286656a5572df05bca403e913912500908e991060上执行，脚本与source前后hash一致。首次NumPy探针与准确CI查询独立并行；随后interpret/CUDA串行，无第二本地数值进程。尚无新的自动摩擦求解器、生产合同或最终code review；原法向程序和μ0门禁保持原样。
