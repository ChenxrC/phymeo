# C10：Newton 1.7 原生 FREE 树接触参照

状态：CODE_READY / UNEXECUTED / AUTHORING_RELEASED。此文预声明输入、口径及验收门，未运行 Python/import、数值、静态检查或设备程序。权威合同为父 bdc5bd2 的 docs/superpowers/plans/2026-11-08-resident-floating-contacts.md 末节。现仅新增本报告与 tests/reference_newton/robot/resident_floating_contacts_native.py；其他 C9/C10 文件不变。

## 固定案例与独立真值

每个 cpu 或 cuda:0 调用必须完成两例各 40 个真实 Featherstone 子步，h=1/1024 s。初始速度、重力、关节输入和所有外部 wrench 为零；不存在 matched-law 力回填或 Phymeo 状态输入。

| 案例 | 固定物理输入 | 首步独立锚及逐步独立递推 |
|---|---|---|
| F01 | 显式 FREE 根，m=2 kg，COM=0，I=.5I kg·m²；r=1/8 m 球，原点 z=3/32 m；固定有限箱中心 (0,0,-1/8)，半尺寸 (1,1,1/8) m | gap=-1/32 m，Fz=2 N，az=1 m/s²。N=64 max(0,1/8-z)，v+=hN/2，z+=hv_new。 |
| F03 | 一个完整 FREE articulation；根 m=2/I=.5I，两 Px 兄弟 m=(2,1)/I=.25I，COM=0，偏置 (-7/64,+7/64)m，两球 r=1/8 m | full g=[root6=0,-2,+2]，root-x/两 P 质量块 [[5,2,1],[2,2,0],[1,0,1]]；a=(0,-1,+2)，总动量0。N=64 max(0,1/32+qL-qR)，两 P 加速度 (-N/2,+N)，各用半隐式更新；根保持。 |

两侧 shape 声明相同 ke=64 N/m，kd=kf=ka=mu=mu_rolling=mu_torsional=0；density=0、作者 margin/gap=0、collision_group=1、has_shape_collision=True、angular_damping=0。lock_inertia=True 保留独立质量/惯量。直接父子过滤开启；F03 是同一 articulation 的兄弟接触，不取消整个 self-filter。原生 active margin 含有效半径，必须保留，不断言它全零。

同入步比较 atol=5e-6、rtol=5e-5；完整 M 比较 atol=2e-5；40 步独立轨迹 atol=2e-4，rtol 仍 5e-5。零输入/ID/类型/计数用精确相等。门在首次执行前固定，失败不调参。独立 J 来自 vCOM_i=vCOM_root+omega×r_i+axis*qdot_i，M 为逐体质量和 COM 惯量贡献；不拿原生 M/J 输出生成期望，也不把一维质量块冒成完整 M。

## 原生程序和观测合同

正常公开 ModelBuilder 创建真实 FREE 根和 P 子体，单一 add_articulation；CollisionPipeline.collide(entry,contacts) 后将真实 contacts 传给 SolverFeatherstone.step。两例每步先清外载，joint_f 维持零。独立法向公式只作观测/验收，不施加回模型。

Newton 的 FREE public qd 为父锚基下 COM 线速度及角速度；本两例父锚为恒等，因此显式基变换为 I。body_qd 是世界 COM 线速度+世界角速度。完整 public M/J、两侧入/末态 pose/twist、joint_qdd 及内部 qd 都保存；内部加速度不能无条件冒成一般 COM/world 加速度。

Featherstone 在 RNEA 中原位改写 body_f_ext：原始 raw 是负的、关于实际 body_solve_origin 的 wrench。保存原数组及实际入步 body_q_com/solve_origin，按 F=-raw[:3]、tauCOM=-raw[3:]-(COM-solve_origin)×F 独立解码；FREE 原点是入步根 COM，不能套固定根的 world0。另从每条原生 row 的完整 points/normal/margins/material 重建双方力和 COM 力矩，交叉核解码结果。固定箱没有动态 body buffer，其反力明确标为 row-derived 固定参考点 wrench。

rigid_contact_force 在该 pin 尚未填；BODY_PARENT_F 是递归净关节 wrench，均不得称本参照的接触传感器结果。raw row 保留这些实际字段和 unavailable=None，不虚构原生观测。

## 保存与失败约定

自包含 driver 提供 --device cpu|cuda|cuda:0 --output ABS_DIR。固定输出 F01/F03 两个完整 .json.gz、summary、driver 与 pinned source 原字节快照；每例未压缩上限128 MiB。保存每个 active 原生 row、原 model/material/filter/shape type、全部原输入、entry/exit 状态、native scratch/解码、独立真值和检查、动量/KE/弹簧能量。检查失败继续该例 40 步及另一例，最终 exit 非零；真实 API/设备或无效状态异常保原失败前缀，不冒完成。全记录是原生自身轨迹，不依赖 Phymeo/TYX/repo tests。

本节点只比较纯法向、单站点、无转动的弹簧接触动力学。Newton 归一 Euler 朝向推进与 Phymeo exp-map、一般多 patch/阻尼/摩擦不据此宣称等价。无性能收益声明。

## 已读本机源码依据

环境为父 .venv-newton-1.7.0/Lib/site-packages/newton/；不安装或探针。完整运行时需固定 Newton1.7.0/Warp1.17.0，并保存实际版本、路径、device 与下列源码：

- _src/sim/builder.py:6021 FREE 创建；:5681/:7937 直接父子过滤；:15083 同体/静态体过滤，兄弟可接触；:803 起材质单位。
- _src/sim/articulation.py:1264 public COM/world J；:1687 public M。
- _src/solvers/semi_implicit/kernels_contact.py:414 材质平均与正 override；:461 A→B normal；:468 实际皮肤点；:517 显式弹簧；:552 双方相反力。
- _src/solvers/featherstone/solver_featherstone.py:557 入步 FK、:717 实际原生 contact 调用；kernels.py:1042 FREE 速度转换、:1397 FREE solve origin、:1498 post-RNEA wrench 原位写回。
- _src/sim/contacts.py:297 有效半径 margin、:303 未使用 force buffer；_src/geometry/narrow_phase.py:772/:869 sphere-sphere/sphere-box 原生窄相。

完整 CLI、模型、原生推进、读取、原始保存与检查已纯文本落盘；未运行任何 import/数值/静态/设备程序。运行、静态与实际结果均待父唯一串行槽，不能把本次文本审查标成通过。

## 最终作者交接

首 checkpoint 41fbc34 仅声明/模型/解析。后续补齐两例全窗口和完整观测；相同两文件范围，无 OLD 修改、无父 merge、无任何项目执行。

创建前源码复核修正一项作者过滤前提：FREE→world 的 collision_filter_parent 必须按 pin 默认 False。builder.py:7937 也遍历 body=-1 的 child 关系，若显式 True，后添加的固定箱会被误过滤。真实 root→Px 子体仍显式 True，ShapeConfig.collision_filter_parent/has_shape_collision 仍 True；兄弟动态球允许相撞。最终完整过滤表通过 shape_collision_filter_pairs_array() 保存，并核目标 pair 不在其中。本修正发生在首执行前，不称原生运行失败、不改物理/材料/门。

每例保存 F01-native.json.gz / F03-native.json.gz。summary.json 绑定原压缩与未压缩字节数、压缩 SHA256、原运行参数、两例 completed/checks/首错及原执行异常；source_snapshot 保 driver 和 13 个 pinned Newton 源原字节，摘要保存运行前后来源 SHA。单例未压缩上限128 MiB，恰两例，无按结果选择子集。无覆盖旧证据：输出目录必须是新的绝对目录。

每步 trace 明确 native_step_submitted、native_step_completed、checks_complete；只在实际状态与本次 scratch 读回后计物理步，CUDA提交返回本身不当完成。门失败继续实际40步和另一例；真正 API/设备/非有限状态异常保原prefix，仍尝试另一例，最终exit1。保存器/观察失败不称物理回滚。JSON 非有限原数值以带 nonfinite 标签保存，不改为0。

运行入口仅供父后续唯一串行调度：
- 原 .venv-newton-1.7.0/Scripts/python.exe + 此脚本 --device cpu --output NEW_ABSOLUTE_DIRECTORY
- 同解释器 + 此脚本 --device cuda:0 --output ANOTHER_NEW_ABSOLUTE_DIRECTORY

每个调用预声明80真实子步（F01/F03各40），当前实际完成数为0。全M/J均为公开原生读回与独立身体求和对照；解码接触wrench和两surface重建分开保存。离散动能门为平移半隐式的 ΔK=h*g·v_next−(Δv)ᵀM(Δv)/2，实际弹簧+动能变化只报告，不额外要求虚假的精确能量守恒。静态固定箱参考点取实际 shape_transform 中心；两例纯法向零COM力矩是该输入的独立门，不外推偏心/摩擦力学。
