# Newton 功能对齐：执行到 2026-11-12 07:20

**本批已于2026-11-12 07:20收束**。全部实际执行和原三个作者交接结束；[最终结果](../../development/p65-parity-until-0720-20261112.md)、[更新全景](../../development/phymeo-newton-panorama-2026-11-12.md)与[下一批计划](2026-11-12-newton-parity-next.md)为当前入口。以下计划和调整按历史保留，旧pending不触发重跑；整个P6.5/Newton对齐未完成。

用户授权：上一批完成后按真实差异和失败制定计划并直接执行。本轮北京时间 00:35 接续，截止 07:20；旧“无截止”只指历史批次。功能优先，性能为次要、同条件可解释的实测。不给总体完成百分比。

## 已完成的起点

基线 `b9acf7be`。见[上一批实际结果](../../development/p65-parity-batch-n4b-rob2a-col-stop.md)与[全景](../../development/phymeo-newton-panorama-2026-11-11.md)。

- N4b 完整九份既有轨迹诊断成功：S 保持学习通过，J/R 仍弱；都有真实参数更新、未动作饱和，不能把“相似方向动作”直接当成因果。
- ROB2a 固定根 NONE/FILTEREXACT 执行器、力/运动共同发布通过；解释/CUDA各26项、相邻14项，新wheel两路径各40 world-steps。尚未与自由根、接触、混合约束组合。
- 显式 integration observation 已通过 source/installed 对照；保留原物理审计与默认 full，缺少端点几何等字段明确，不是静默关闭检查。
- COL 同候选停止实验触发9个旧资源失败，未合入。四Y法向门仍开放。不能继续凭猜测扩大阈值。

## 并行节点与交付

| 节点 / 难度 | 负责人 | 执行内容 | 成功或停止条件 |
|---|---|---|---|
| N4c / 中高 | 原 source agent | 读取固定9组初始/最终策略；固定对称输入网格，核目标偏置、奇偶响应及跨关节响应，使用原模型/normalizer/hash；先基础测试，再唯一串行实际分析 | 解释每种子响应且绑定真实策略；不训练、不改原学习门。若无法区分假设，不扩大网格，转缺失优化诊断 |
| ROB2b / 高 | 原 audit agent | 从实际 T1/T2/T3/T4 路由研究，复用已有完整M/J与统一actuation stage，把一阶执行器连接自由根与接触/约束；多案例先冻结，根共享接口交父 | 正常/反gear/混合NONE+FILTEREXACT、接触/限位/摩擦、actearly/饱和、恢复/晚失败；力/act/q/v同发布且独立native对照；不复制新求解器 |
| COL1a repair / 高 | 原 tests agent | 先对固定9个回归采集实际simplex/支持/重复/退出信息，与接受基线对照；结合已有Newton/MuJoCo代码研究；原因具体后只尝试一个通用修复 | 原6目标+9回归+受影响矩阵，保持原预算/门；不按形状/ID/坐标分支。一次修复未缩小问题即保留失败，转独立任务 |
| GPU observation / 中 | 父 | 将明确的integration模式放入实际完整PPO短更新链，与full同输入对照；固定J/S/R与W8/128/256，保存物理/奖励/策略输入/更新身份和端到端时间 | 共同字段与训练输入/更新一致；清楚区分输出契约、含证据与不含证据计时；不把局部快照快称Newton加速 |
| 收敛集成 / 中高 | 父 | 分支审阅、必要共享接口、新包源码外验证、全景与架构更新 | 新代码/安装包/模块来源一致，失败不掩盖；只验新和受影响能力，不重跑已绿同源 |

N4c 的固定网格：原8个held-out初始观测；q=target/v=0；各q分量为target±qscale/4且v=0；各v分量为±vscale/4且q=target。初始和最终policy/critic全hash绑定；保持调用者RNG。保存raw action及映射ctrl。J的恢复方向是诊断，R的耦合响应仅描述，S非零平衡控制合法。禁止新增成功门倒灌N4a。

只有新证据能区分原因时，才预登记一次后继学习消融；保留原三种子和128更新预算，不搜索参数/加预算追绿。截止前预算不足则只交可验证诊断与下一合同。

GPU短链默认每格2次更新、每更新16步；full/integration使用相同初态、seed、任务、算法与物理步长。先W8资格，再128/256计时；出现不一致立即定位，不扩大规模。两次更新只证明连接和资源，绝不声称学会。完整观测多出的端点字段不参与策略输入；结果不得叫相同输出成本比较。配置细节在执行前补齐到本批协议。

## 时间安排

- 00:35–01:30：原三支源码/研究与案例冻结；父新串行队列与GPU短链。
- 01:30–03:30：独立编码、固定N4c/COL证据；机器人共享接口按交付接入。
- 03:30–06:20：串行解释/CUDA与受影响回归；只有诊断明确且时间足够才做一次新训练消融。
- 06:20 起不接长实验；06:30–07:00新stage安装与消费者，07:00–07:20完成实际全景、剩余差异与交接。依据实际耗时调整顺序，不为赶点伪称通过。
- 任何任务预计无法在剩余窗口安全结束则不启动；已合法运行的有界任务不随意终止。

## 资源、所有权与避免循环

原三个隔离worktree并行纯代码；全部import/collection/pytest/数值/离线分析/lint/formatter/build/install/profile由父唯一串行执行，逐命令fresh入口guard，保留首失败/raw/source。其他真实执行自然结束，不kill。沿用现有环境/依赖/pin，不清理活跃环境或证据Junction，不使用远程CI、不push/mainmerge/发包。

COL失败实验分支保留，不能直接合入；只在干净原worktree创建接受基线上的新分支或提交明确诊断增量。父是共享runtime/checkpoint等唯一owner。三个既有K1未跟踪WIP不动。

完成一项应有公共可调用能力、多案例证据和明确边界。连续两次针对性尝试无新信息、只改阈值或增加迭代、重复同源绿套、以单示例代替矩阵，都视为循环：保存最小失败并转向独立模块。本轮不重复G1长前缀；原46.29s参考XY失败、完整131.48s未通过仍开放。

## 实际记录

GPU执行前补齐：每个模式独立clone同一未推进基线；固定原任务首seed（14011/14111/14211）、原MLP与PPO配置，每模式1次16步warmup后2次16步正式更新。W8/256先full、W128先integration；两模式开始前重置同一随机数。保存完整原record和learner/RNG，对照视图仅排除不同owner标识及full独有端点字段，全部共同状态/原step receipt/训练输入/loss/参数/Adam/RNG必须完全相同。基础构造与warmup不计正式吞吐，完整update计算/物理审计和loop至trace关闭分别报告；初始观测单独保存/hash。首3格W8资格成功后才执行其余6格，不重复已过格。

待执行结果只写实际结果；计划勾选不是验收。证据目录将使用独立 `artifacts/p65-parity-until-0720-20261112-01`。07:20给用户本轮实际交付、失败/未完成、当前八方向差异及下一优先项。

00:52 GPU首轮W8三格及J-W128通过；J-W256在clone第二批native MjData时内存不足、尚未推进。修正工具生命周期：两模式各自按相同固定factory顺序创建单套世界，先关闭factory未使用的task workspace，再借用该runtime创建显式模式；一模式完全close后再开另一模式。不改物理/容量/输出契约。原四格结果保留为旧生命周期证据，不混入最终低峰值内存计时；由于benchmark资源生命周期已变，本次修复后完整九格作为最终资格，之后不重跑同源绿格。


## 01:35 后继消融预登记

run57固定九份N4c已完成，J的目标偏置明显且自身正负q扰动未换号，仍有交叉/非恢复响应，不能断言唯一原因。仅批准J原三种子一次“初始目标mean居中”消融：只改变actor最终均值层初始bias，原其余参数/critic/std/Adam/RNG、任务、128×32×8预算和持出门不变，之后自由训练；保存前后完整绑定并同时并列旧新绝对cost，弱或矛盾则停止。原source agent先独立协议/tests/实现，父shared runner仅可选initializer hook，默认不变。不是S/R参数调优，也不倒灌原J学习通过。

## 01:55 资源释放审阅增量

独立审阅确认：原 clone 修复只保证外层 bank.close 调用前失败的 owner 重试；实际 ExitStack 会先弹出 callback，底层 free/Runtime.close 自身失败不能由第二次空栈返回冒称清理完成。固定 TYX 的释放失败也没有公开的失败原子保证，不能盲目重试 CUDA 释放。

原 audit agent 新增一个有限资源 ledger 与故障案例，父接三个既有 device 构造/分配接缝。正常路径按 buffer→runtime 顺序关闭；成功项移除，实际释放内异常保留为 UNKNOWN_RELEASE，独立 siblings 继续关闭但不销毁依赖 runtime；后续 close 明确 sticky 失败、不重复未知释放。工作进程内强引用隔离 owner，明确不承诺进程退出析构或自动恢复。外层调用前故障保留已有重试语义。只改生命周期，不改数值、pin、控制或物理门。

测试先覆盖前/后副作用异常、顺序、重复 close、构造失败；实际两后端只在已成功释放后注入晚错误以验证接缝，不制造无法释放的真实设备资源。新源码外组合消费者覆盖正常资源关闭；必要旧 clone 受影响验证有新修改依据。该节点不把 S256 的历史内存失败因果归于 clone，也不扩成全项目资源重写。

## 02:10 COL1a 收片后推进 B-point

COL1a retained-candidate 修复已实际通过两后端各10项新例、12项原目标/发布回归和282项受影响矩阵；原门、预算和九个原回归输入保持，父合入 aa839fba。原一次实验失败仍保留；本次不称严格全局 GJK 证明。

根据[曲面后继研究](../../development/p65-curved-patch-next-batch.md)，原碰撞作者先做 B-point：任一侧 S/E 的九个新增曲面族统一零维 device patch，完整材料/offset/pair anchor、同次 source 和固定工程门。只验证并传递同一候选的合法 witness，不 HOST 搜索/投影；旧十族有限 patch 和已验 GJK/EPA 不改，剩余九族母线/圆盘 patch 仍 unsupported。这是已解析法向下的点接触离散，不承诺全局唯一 MTD、连续面积压力。旧未决状态不能出点；多解/包含若已解析，仍须全部独立支持/边界/闭合检查，不能新加按输出挑选的阈值或声称唯一法向。

先固定九族三态27身份、九协变、四来源三步和容量/错源/晚失败，再实现 NEW point-patch 核与 query 接线。旧三个 K1 WIP 不动。C-point 的真实载荷必须等几何资格和表示桥审阅后另接；时间不足则明列待办，不能将点 patch 通过称受力通过。该支线与 N4d/资源账本纯编码并行，执行继续父唯一串行。

## 02:43 实证后的调整

N4d原三种子已完整结束，中位ratio0.993057、一个变差，学习门仍失败；停止居中因素，禁止扩大预算/搜索。后继只做N4e一次J原8个held-out起点、256步的固定两段解析控制可达性诊断，使用原公开动作/成本/边界，固定128/128切换，不可行则直接保留失败；它不替代learner验收，也不外推R。原source作者只增example/tests和协议，父唯一执行。

B-point首轮touch-SE暴露原已接受法向误差与同法向singleton点位门不相容，11pass/1fail后按表示停止规则隔离，不合入、不C-point、不改原门。剩余几何与CUDA未执行。完整载荷合同仅保留研究稿；后续要另定共同候选或明确近似消费合同。

GPU内存改进转为完整compact HOST mirror+单native观察scratch，已通过新29、受影响55、训练续接9项。原source研究证明固定MuJoCo3.3.7无公开Python目标copy API，因此不借私有指针/pin更换；原CPU求解/TYX物理核不改。父下一次固定S256仅测新表示下的native构造/copy与进程commit/RSS、原两mode完整48步/三PPO更新一致性；不重跑九格或旧轨迹。随后仍按06:20长实验截止与07:20交接推进。

## 04:00 N6 控制与物理频率接线

已完成的 BALL1、N5、COL2 新结果进入集成与安装包资格，不重复旧同源绿套。下一独立节点 N6 根据原 Newton/IsaacLab/RSL 源码研究，在现有设备 VecEnv 上增加显式 control-window 规格；父拥有 Env/RSL 接缝，原 source 作者拥有 NEW 规格、测试、例子，原 audit 作者只读复核。执行仍父唯一串行。

冻结范围：D 为 1..8 的整数，默认未启用保持旧 Transition 类型和 checkpoint binding。opt-in 一次映射/提交原三类 native action，保持它推进 D 次原 runtime.step(h)，逐次保留真实完整 public state/last-step、time/tick/revision。原始双精度时钟仍由 N3 公共 checkpoint 在更新边界保存，不额外每个子步制造 capsule。h 是物理步长，policy_dt=D*h；horizon、reward、done、gamma、lambda 都按 policy transition，奖励只用原 endpoint 值，不隐含乘 dt 或子步累加；终止在窗口末判断一次，重置一次。不承诺中途终止锁存或逐世界冻结。

新记录保留 attempted/completed、每步 acceptance、真实诊断或 missing-diagnostics、首次失败位置。runtime.step 成功返回即计完成，随后 snapshot/export 失败不能抹去真实状态。BaseException 也记失败前缀；接受后失败 poison，不声称窗口原子回滚。N5 reset 已消费索引在 reset 后 export 失败时保留。Env 与 RSL publication 使用短 busy guard 阻止同线程重入；N3 整个 training_boundary 不能被标 busy，因为正常更新需要在其内部调用 Env.step。

恢复语义不扩展：失败的 N3 session 自身永久 poisoned；Env 全重置不清除 session poison。需要丢弃失败 session，并由健康新 session/owner 在合法空更新边界装载有效 checkpoint，不能声称原 poisoned session 直接恢复。

预登记验证：J 四世界 D=1/2/4 各8窗口、S 四世界 D=4 共8窗口；独立 held-action HOST/public-runtime 对照、逐步完整接触诊断、endpoint任务公式及 N5 部分重置。J 原 GRU D=2、8步 rollout 的实际更新/save/A/restore/B，保完整 Adam/RNG/hidden/物理与 reset stream；跨 D checkpoint 必须在物理 CAS 前拒绝。第二次 physics 失败、第二次诊断失败、末端 export/reset-export 失败、同线程 step/reset/capture/restore 重入均有真实故障例。新功能不更改原学习预算、fixture、精度门，也不声称 PPO 已学会。原 06:20/07:20 截止和安装包计划保持。

## 04:23 两个已调研的组合节点

N6 新40项与受影响41项已完成（原过时reset观察计数首失败保留）。根据[球树驱动调研](../../development/p65-ball-actuation-next.md)与[三角面dispatch调研](../../development/p65-triangle-dispatch-next.md)，剩余开发窗口执行以下两项；06:20后不再扩域，优先把已过内容放入新包。

**BALL-A / 中高**：保fixed BALL/hinge/slide nv<=8/nq<=10，最多32个scalar JOINT执行器、32 activation。NONE或FILTEREXACT只能指向真实hinge/slide；球通过完整M/J/b被动耦合运动。父只改_model/_tree_model/_actuator_model/_mujoco_tyx_runtime四既有文件的结构dispatch/准入/能力名，复用原三/四stage kernel、bank和checkpoint。原audit作者在原隔离树先写NEW测试及报告。冻结研究中的四组多案例、24连续步反向/卸载、qa!=dof、mixed/disabled/clamps/actearly、稀疏world恢复/clone/reset/晚失败。完整native与独立scalar期望、原BALL/ROB2a数值门保持。真实ball-target motor、free/contact/limit/dry/spring/JOINTINPARENT仍拒绝；遇需要数学kernel/ABI变化的反例停在原边界。

**COL3 / 高**：原COL作者只NEW geometry/_triangle_distance_dispatch.py、NEW tests/collision/test_triangle_distance_dispatch.py、NEW报告；父只接CollisionWorkspace的boundary_distances(motion, *, margin=0.0, limits=None)、last_boundary_distances、require_boundary_distances及close。复用现注册/过滤和同次实际BVH；独立拥有COL2 runtime，不借旧engine的native pointer。返回typed aggregate与全部候选/原facet映射、完整同次raw packets/bitsets；旧detect/last_result/force语义不改。

COL3仅interpret/CUDA当前快照，全部模型/原material顺序motion/半径/filter/revision完整绑定；拒非恒定end、turn、非零surface skin与模型中本切片不支持的shape类型。注册sphere/capsule/triangle/mesh；同族或过滤对明确列出状态，不伪造跨族结果。margin为有限非负工程邻近带：axis AABB向外扩radius+margin，所有能够落带的facet必须成为候选；空带不证明闭mesh内部物体已分离，也不保证带外全局最近面。每surface最多512vertices/128faces/64segments，整个query最多32groups/8192候选，limits只能收紧；超限报告真实所需数，不截断。保留每coincident/shared-edge原facet和CONDITIONING/DEGENERATE/undefined-normal，不以HOST closest fallback补结果。

公共publication与pending BVH layout沿workspace.transaction一起提交，最后组失败保旧aggregate；已进行的私有kernel计数/失败诊断不抹掉。require_boundary_distances按完整当前输入再绑定，不仅信query_id。独立query资源UNKNOWN_RELEASE沿原ledger保留、不盲retry；正常close尝试独立siblings，失败不能冒已释放。

COL3冻结三组：sphere/capsule对有限面face/edge/vertex/outside/front/back；concave/shared-edge/co-nearest与真实pruned facets及closedmesh内球的正boundary-distance语义；两个稀疏world/重用localID/存储排列/变形快照/filter、exact/one-less容量、末组晚readback与外层transaction拒绝。只有新公共调用与受影响接口资格通过才claim集成，不重跑COL2全绿矩阵。resident mesh设备顶点源、实体CCD、curved point和真实载荷消费仍后继。


## 03:18 后继功能节点

已关闭S256表示诊断与N4e固定控制可达性，结果见实际报告。保持J学习门失败，停止无新证据的训练调参。

- BALL1（高，原audit作者）：固定根ball+hinge/slide，nv<=8/nq<=10、无执行器/接触/限制/干摩擦/球spring；native-joint与DOF分别映射，SO(3)右乘积分与完整M/J/bias。多结构、各方向力矩、恢复/失败、原scalar接缝回归。父负责路由与共享ABI。
- COL2（高，原tests作者）：新的设备finite segment–triangle边界距离，sphere为零长segment、capsule为有限segment+radius。全部候选/参数/重心/两见证/未决状态绑定；独立精确oracle和多形状姿态、共面/平行/退化/事务案例。距离并非solid内外或响应；旧B-point隔离保留。
- N5（中高，原source作者）：先调研原Newton随机重置和本项目事务/checkpoint接口，冻结按stable world和reset index的独立采样；父拥有runtime/env共享接缝。仅状态随机化，不暗加模型质量/摩擦随机化。

BALL1目标05:30交接，COL2目标06:00；06:20后不启新长验证。任一实现无法保原兼容或独立验收即明确保留未完成，不把已编写称通过。

## 04:55 N7：已连接训练功能的批次规模资格

BALL-A 与 COL3 的新源码矩阵及受影响接口已通过。剩余时间优先验证 N5 随机重置、N6 控制窗口和真实上游 RSL 更新在同一批次运行；不继续 J/R 学习调参，不改变原物理门或模型。原 source 作者只新增 `examples/mujoco_tyx_window_scale.py`、`tests/training/test_mujoco_tyx_window_scale.py`、`docs/development/p65-window-scale-report.md`，父保全部共享接口及唯一串行执行。

冻结三格：J/W8/D2/GRU 作为首格；S/W128/D4/MLP；S/W256/D4/MLP。每格 full 与 integration 两模式依次独立构造/关闭，复用原 J/S 模型、步长、任务/动作尺度和原 N5 q/v 采样界。stable world keys 为 7+13*i，anchors 使用原8模板按 i%8 映射，seed=6511、namespace="N7"；horizons 按 3/5/7/11 循环。每模式固定1次8-policy-step warmup及2次8-step测量更新，使用原 N3 learner/optimizer 配置；J recurrent 只选择既有 GRU，禁止调整隐藏宽度/学习率/奖励或增加预算。两模式开始前恢复同一 CPU/CUDA/owned RNG。它证明更新连接和规模，不是学习有效性实验。

每个真实控制窗口保存请求、逐物理步完整接受状态与原收据、真实reset索引和episode、endpoint任务与RSL输出；actual attempts/accepted worldsteps不按reset后的tick猜。完整 learner/Adam/normalizer/RNG及GRU hidden按允许的owner标识差异比较，full独有观察字段单独列明，所有共同物理/动作/任务/更新须相同。HOST sampling和端点观察仍明示，CPU step不代跑物理。S独立检查发布接触力与原动量公式；不是重新求接触律的独立oracle。

先完成新协议测试和J8实际两模式资格，才执行S128/256；任何共同状态差异/资源拒绝立即停止扩大，保真实失败，不降world数/放宽门追绿。顺序构造限制native内存峰值；记录native构造/copy、设备/物理调用、process commit/RSS及端到端时间，计时分update与完整证据关闭。仅本机同配置/不同输出模式比较，没有Newton同输出性能优势声明。06:20仍停止新增长任务；若未完成，保已过格及明确未启动格，不影响已验产品的wheel资格。


## 05:35 BALL-C 后继组合：按源码合同做一次有界开发

已全文审阅 [BALL-C 源码合同](../../development/p65-ball-next-composition-contract.md)。父独立确认现有 normal/constraint wrappers 已传完整球关节 DOF 表和 quaternion integrator；本片只接固定根 BALL 树与现有 contact 路由，原 kernel/solver/预算不改。原 source 作者在隔离树先 author NEW tests/test_mujoco_tyx_ball_contacts.py、tests/test_mujoco_tyx_ball_contacts_lifecycle.py、tests/installed/check_ball_contacts.py 和 docs/development/p65-ball-contact-composition-report.md；父独占合同四个既有产品文件。C1–C6 的 XML、输入、24步时序和原误差门必须在首次执行前登记，不调 fixture 追绿。nv<=8/nq<=10、scalar NONE/FILTEREXACT、condim1/pyramidal3、最多16 pairs/32 slots；FREE、直接球电机、limits/dry、condim4/6仍拒绝，完整结构检查不能被路由绕过。

本授权先为纯编码。目标06:00前交接，父全文审阅后才在唯一串行槽执行。若证据或剩余时间不足，保隔离 authoring，不把未验代码带入最终 wheel；06:20不启动新长验证和07:20交接边界保持。N7正在跑的source不改，accounting新发现先完成诊断与受影响修复。

## 06:20 收尾期间的有界训练诊断作者任务

新功能与最终wheel已全部通过，产品快照1ce57489冻结，不再扩物理模型域。
根据[原记录与实际上游源码研究](../../development/p65-jr-learning-next-diagnosis-20261112.md)，旧N4记录缺价值/GAE和preclip梯度；继续学习调参没有足够证据。只授权原source作者在隔离树tests-first新增
`examples/diagnose_mujoco_tyx_credit.py`、`tests/training/test_mujoco_tyx_credit_diagnostic.py`、`docs/development/p65-credit-diagnostic-report.md`。
不改任何产品、上游RSL、原案例/预算/报告或已冻结wheel；其余原agents仍只文档。

原协议的J14011/R14211/S14111原final权重、W8/D1/32步、原task seed新诊断流、完整actual upstream backward、public optimizer pre-hook在第一次Adam前停止均固定。
总量768实际world-transitions、0完成optimizer步骤；不是新学习实验，不加warmup/rollout/重试或改normalizer。
用现有bounded decoder读取所需原保存策略并绑定来源；不得猜缺失历史RNG。观察原upstream方法的真实数据，不替换PPO/梯度或私有physics bank。
Synthetic tests覆盖schema/归一/GAE、完整batch permutation与重复歧义、异常hook/RNG清理、意外mutation和固定A–E分类。

目标06:45代码交接，06:50仍未交接则只保authoring待后批。父全文审阅、静态/合成资格后才可另授一次短实际诊断；预计只原三格，不启长训练。
若07:05前无法启动，或一次实际诊断遇到物理/绑定失败，保完整前缀并停止，不挤占07:20交接。
这项仅补缺失可观测量，不修改已验产品域；06:20不启新长实验的规则保持。已通过最终包无需为此example重建。

### 06:39 诊断执行决定

作者5970a4cc与父独立审阅已结束。父00e13f03仅修import/格式（299保非import AST与scope内import集合相同）；293/294首静态失败保留，297/298最终静态通过，300全部30 HOST合成节点通过/6.82s。当前选择**CLI一次**运行301，CUDA测试节点不另跑。父为唯一owner，每次命令fresh guard；24个选定source/config/upstream文件保存前后绑定。

301仍严格原J14011/R14211/S14111、32步W8/D1、最多768真实world-transitions、Adam128→128和0新更新。若失败，保原请求、已接受物理前缀、梯度/清理证据；本批不再启动第二采样流。返回诊断分类不代表学习成功。最终wheel1ce57489及所有产品文件保持不变。
