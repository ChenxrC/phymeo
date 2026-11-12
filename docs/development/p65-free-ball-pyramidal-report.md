# FREE + BALL pyramidal 接触组合：R3-P 作者交付

状态：CODE_READY_TESTS_PENDING / UNEXECUTED。本报告记录执行前固定合同；
作者没有运行 import、collection、数值、pytest、lint、formatter、build 或安装。
父拥有唯一项目执行槽。原 R3-N 已通过的结果不转记为本次 P 通过。

## 1. 范围及接线责任

产品仅修改 [_floating_model.py](../../src/phymeo/backends/mujoco_tyx/_floating_model.py)
既有 FREE+BALL 接触准入：

- 先经过原完整 native FREE/BALL/R/P、惯性、actor、被动力与资源结构验证；
- 原 possible-pair predicate 与 mixed priority/condim 规则得到的实际维数只准入1或3；
- 只有实际存在 mixed condim3 时要求 pyramidal cone；
- 全部 mixed1 仍允许原 normal 路由，不能因默认某个 geom authored condim3
  或 opt.cone=elliptic 误选 mixed solver；
- 保留 nv<=35、nq<=45、body<=128、actor<=32、geom<=128、
  possible pair<=16、raw candidate slot<=32；
- 不放行 activation、joint limits、dry friction、BALL spring、ball affine bias、
  FREE-target/JOINTINPARENT transmission、multiple FREE、explicit pair/exclude、
  新几何、condim4/6 或 elliptic condim3。

父负责 _model.py 在完整 floating validation 后选择原 compile_constraint_model，
以及 runtime 的精确能力标签。这两条不属于作者修改。冻结标签为
free-ball-joint-motor-pyramidal-contact-v1，contact_dimensions=(1,3)；
joint_limits_supported=False、dry_friction_supported=False。纯 mixed1 保原 normal。

没有新几何核、求解核、CSR或积分器。复用原 full current M/J/b、
physical-M 混合非负 QP、原 Euler M+hD 与 native_joint_layout 四元数积分。
维持 source iterations 1..512、原 stored solver tolerance/KKT gate。
不以 native MuJoCo Newton 的迭代过程描述设备 active-set 求解器。

## 2. 执行前固定输入

两族完整复用 R3-N 的 feet / siblings XML、初态、质量、惯性、驱动和窗口，
唯一物理声明变化是 default geom condim=1 改为3；原 friction=.4 .01 .001
保持。测试通过 AST 检查 source/motor/initial/control 函数和 ROOT_INERTIA、
HEADER，再核两族完整 XML 仅有此差异。此检查不导入旧测试作为物理 oracle。

| 族 | 原 native布局 | 固定实际窗口 | 必需现象 |
| --- | --- | --- | --- |
| feet | nq12/nv10/nu2，FREE/BALL/SLIDE，两球与平面 | W2，keys(7,41)，h=.001，24步 | 原2个初始受载接触、真实FREE运动、normal及非零tangential响应 |
| siblings | nq15/nv12/nu3，FREE与两BALL支链 | 同上 | 原1个初始受载接触、两BALL运动、双方非零COM矩、共同FREE广义接触合力/矩为零 |

真实初始时间.125、非零root线/角速度与子关节速度保持。
第0–7步原正驱动，第8–15步原反向驱动，第16–23步所有ctrl/qfrc/xfrc卸载为零。
卸载控制不等于要求所有接触立即离开；不修改原几何或延长窗口追求某种事件。

原 R3-N 数值门保持：q绝对1e-4、v绝对1e-3、时间2e-6；
接触力 rtol/atol=3e-3/3e-3，qacc及积分加速度3e-3/1e-2；
原完整动力学、motor及几何各门见 consumer 常量，不按输出调整。
mixed 行和独立 physical-M KKT 使用原 fixed BALL-MC/T4 参考口径，
不把只含一条normal的代理当作四边模型。

## 3. 完整独立参考和保存

[check_free_ball_pyramidal.py](../../tests/installed/check_free_ball_pyramidal.py)
是独立源码外消费者。直接复制已验 N 消费者的通用记录/来源/生命周期代码，
本文件自包含；执行时不需要旧 sibling 文件或任何 repo tests/examples。

每实际步：

1. 先取得真实 stored入步q/v/act/time/三通道，保存完整输入。
2. 原生3.3.7在该实际入步上独立 mj_forward，得到完整 M、每body COM J、
   bias/passive/applied、所有接触/frame、四条EFC、独立motor CSR/功率及same-entry后继。
3. 另一条 native MjData 从原真实 stored初态连续演化，保存完整连续后继；
   不把设备后继回填到此连续参考。
4. TYX实际step期间同时禁止 CPU mj_step/mj_step1/mj_step2/mj_forward。
5. step返回立即保存返回值/计数，再读取并保存 full与integration快照；
   首断言之前已保存该次实际payload和native参考。失败保原错误及可用前缀。

比较以几何位置/法向和pair做完整唯一bijection，禁止按force排序。
随后按实际native contact ownership重排全部EFC：

- complete6×nv contact J、所有body M/J/b、qacc/integration acceleration；
- EFC type/id/edge/slot/dof、position/margin/velocity/R/D/impedance/diagApprox、
  aref/force以及raw14列、5列身份与尾部；
- 每active condim3恰4边，edge J=Jn±mu*Jt，非负力无upper bound；
- 四边力独立decode为完整接触wrench，再映射双方COM force/torque与generalized force；
- 接触点速度与双方COM wrench功率等价，motor CSR/力/速度/功率独立；
- 从完整物理M重建 H=J M^-1 J^T+diagR、梯度及原5e-5独立审计门；
  产品自身 stored KKT门仍为原值，source预算不变；
- 全q/v/act的same-entry及独立连续参考，FREE及每BALL单位四元数门，
  以及实际Euler入步v+h_fp32*qacc_integration；
- sibling接触完整6行J的FREE六列为零，而非误把mixed J按normal向量切片。

这包括有摩擦的力与运动共同发布，不是只扩准入或只对比几何。

## 4. 预登记测试节点

[test_mujoco_tyx_free_ball_pyramidal.py](../../tests/test_mujoco_tyx_free_ball_pyramidal.py)：
- 2族×2backend连续24步；
- 17个unsupported-before-device负控；
- 6个直接floating flags不能绕过准入；
- 2个原静态capacity声明；
- 2backend×root/child×zero/nonunit初态拒绝；
- 4个priority-mixed-dimension/cone路由；
- 2backend×pair/slot exact capacity，至少一个真实step完整参考；
- 2族原输入完整绑定。

[test_mujoco_tyx_free_ball_pyramidal_transactions.py](../../tests/test_mujoco_tyx_free_ball_pyramidal_transactions.py)：
- 2族×2backend checkpoint/clone/partial-reset及真实后继；
- 2族×2backend×status/readback/final-world motor publication/late restore四种晚失败；
- 每backend完整bad-control/foreign-checkpoint拒绝及真实后继。

手工预登记总数69：31个HOST结构/输入节点，19个interpret节点，19个CUDA节点。
未进行collection取数。父可按最终实际收集结果报告差异，不能把手工数称实际pass。

晚失败先调用真实device download/native publication或verify bank，然后注入；
要求全部world accepted状态、last_step身份、checkpoint完整bytes及revision不变。
解除注入后，与未中断clone逐项比较真实force/motion后继。
不是把驱动未知释放或CUDA硬错误假称为可自动重试。

静态exact-capacity节点检验16pairs或32rawslots，同时新增实际step消费。
该构造的接触可不活跃；不冒充运行时active contact/efc cap边界。
容量overflow保持原原始准入码，不扩T4更大的静态cap进入FREE+BALL。

## 5. 包外消费者及父执行入口

CLI：

~~~text
python check_free_ball_pyramidal.py --mode interpret|cuda --expected-install-root ABS_SITE --output ABS_FRESH_DIR
~~~

脚本、cwd均须位于源码外，且全部实际已导入phymeo模块必须位于指定安装根，
保存每个模块路径/SHA。没有伪装CUDA或CPUfallback。

每backend：
- feet24批×W2=48worldsteps；
- siblings24批×W2=48worldsteps；
- siblings真实生命周期8批×W2=16worldsteps：
  3warm、A、clone-A、restore-B、partialreset后2后继；
- 合计56实际batch calls /112worldsteps。

完整gzip日志、summary、portable checkpoint、primary/finalization errors
独立保留；close或来源记录失败不能替换原物理异常。
这只是预登记计数，未执行、不宣称安装PASS。

父验证路径为本批两个NEW tests，之后源码外consumer两backend。
原R3-N中“pyramid全部拒绝”已不再是本批合同，父需窄改为仍不支持的dim；
作者没有改任何旧测试、旧installed消费者或共享路由。
本批不要求重跑已绿完整N或其它旧套。

## 6. 提交和限制

作者分支 codex/p65-free-ball-pyramidal-20261113，自父bb3c95bd，原D5/架构分支保留。
tests-first 62004f0a；有限准入82b7b83e；输入绑定45ee0bb5；
独立consumer2efac12f。以上均纯author，父执行前可全文审核。

仅以下五路径有作者改动：原_floating_model.py和本报告、两NEW测试、
一NEW安装消费者。未改模型/参数/求解器/pin，也未产生性能或Newton比较结论。
仍不支持FREE+BALL activation、关节限位/干摩擦、condim4/6/elliptic、mesh或跨体系耦合；
本次脚接触24ms不构成人形长期足地支撑、训练或舞蹈资格。

## Parent source qualification (2026-11-13)

SOURCE_VERIFIED_LIMITED; installed qualification for this pyramidal increment is still pending.
Parent route performs the complete FREE+BALL structural validation first, then sends any actual mixed condim3 pair to the existing constraint compiler; otherwise the original normal/no-contact path remains. Capability family is explicit.

The first run86 returned a real step but the new consumer read normal_geometry_launches, a capability absent from the constraint route. The consumer now checks the source-confirmed seven launches for these plane/sphere and sphere/sphere declarations. Run87 then passed37 and exposed a HOST test premise: changing only the ground priority left the toe-to-toe potential pair at inherited condim3. That route correctly remained pyramidal. The new priority-only fixture now declares both toes' intended dimensions, with the higher-priority toe3 branch unchanged. Original continuous source/physical parameters and numerical gates were not altered.

Run88: both complete new test files, **69 passed in74.43s**, RAW_EXIT0; no skips/deselections. All31 HOST and19 per actual backend were exercised.
Run89: four affected original routing controls (normal feet and no-contact mixed, each interpret/CUDA), **4 passed in16.53s**, RAW_EXIT0. No whole old suite rerun.
Ruff82/90 passed; first format83/91 failed, formatting84/92 and final85/93 passed. Run94 independently confirms the post-test formatted test AST equals its tested run88 snapshot. Product source was not changed after run88.

Evidence: artifacts/p65-feature-until-0730-20261113-01, exact command/environment/before-after source snapshots and first failure logs. This is bounded FREE+BALL stateless joint-motor primitive condim1/3 support. The continuous positives use condim3; mixed1/3 is currently compile-route qualification only. No claim of activation, joint limits/dry, long human support/dance, G1 closure or performance.
