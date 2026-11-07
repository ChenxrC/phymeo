# C10 普通浮根常驻接触：开发与多案例验收

状态：**AUTHORING_FROZEN / INTEGRATION_AND_EXECUTION_PENDING_C9**，2026-11-08，持续无截止。父基线 f9ab296；C9 解释 run460 正在执行。完整 C9 接受仍是共享接缝集成和所有 C10 项目执行的前置条件。

本计划采用已全文审阅的 [C10 提案](../../development/p65-resident-floating-contact-next-proposal.md)（原提交750dfba、文件SHA256 B157EA687FCC3CBF8233B2D4595FD2069AF3FA1C94BD9A34218D035E855C8197）。其 F01–F10 固定输入、首步锚、工程域与负控纳入本合同。为利用父串行验证期间的空档，调整纯作者次序：原 agents 可先在隔离树编写 NEW 文件；父验收 C9 前不得把 C10 接入旧代码或运行。不是提前接受 C9，也不是全设备窄相。

## 1. 公共行为与不交叉所有权

复用 TyxWorldBatch.prepare_contacts / step_contacts 和 ResidentContactReceipt。新增允许已有 ResidentFloatingTreeBinding / from_asset；sampling 仍 discrete_entry，有限 S/C/未圆角 H、mu=0。True 的 tree_dynamic_pairs 仅同一个完整浮根树、不同且非直接相邻 body；默认False与现有显式过滤保持。不存在第二套浮根接触求解器。

父唯一 OLD：
- src/phymeo/backends/_resident_contact_plan.py：明确种类、三维数、浮根bank packet与同次receipt。
- src/phymeo/backends/_resident_contact_bridge.py：有界 physical body index≤16 / count≤17，实际spec仍逐世界校验；旧fixed/affine不会因此多出body。
- src/phymeo/backends/_resident_floating_storage.py：仅在需要时提取已有诊断构造的纯helper，避免读取已发布active伪装candidate。
- src/phymeo/backends/tyx_world_batch.py 与 _resident_collision_plan.py 原则不改；若出现具体薄接缝缺口，父先记录原因。不得改旧积分器、全M/PD方程、旧_numeric、材料律、shape算法或root导出以掩盖未支持域。
- 本计划、续作入口和最终接受文档均归父。

原robot NEW产品（先完成其现有C9 saved-audit作者交接再followup）：
1. src/phymeo/backends/_resident_floating_contact.py
2. src/phymeo/backends/_tyx_resident_floating_contact.py
3. docs/benchmarks/p65-resident-floating-contact-implementation.md

原RD NEW测试/独立参考（先完成其当前设备窄相提案再followup）：
1. tests/scale/test_resident_floating_contacts.py
2. tests/scale/test_resident_floating_contact_transactions.py
3. tests/reference_newton/robot/resident_floating_contact_reference.py
4. docs/benchmarks/p65-resident-floating-contact-tests.md

两作者只能修改各自这些 NEW 文件，检查不存在后开始；不merge父、不改OLD、exports或旧tests/reports、不加助手，不执行import/collection/数值/probe/pytest/static/format/compile/build/install/native。只text/fileIO/smallGit。原RR不可用，不替换或重试。依赖源码从父只读；新的私有浮根算术以C9最终接受版本为准。

## 2. B / D / N 及私有适配入口

B为物理刚体数，D为full广义速度数，N为电机数。ordinary fixed=(n,n,n)，fixed affine=(n_full,n_full,k)，ordinary floating=(n+1,n+6,n)；后者Q=n+7。affine仍在原solve以P归约，root六维没有伪电机。

临时 wrench[T,Bmax,6]、projected[T,Dmax]；完整 Jv/Jw[T,Bmax,3,Dmax]、M[T,Dmax,Dmax]。浮根n0使用真实B1/D6/N0；n16末body16使用B17/D22/N16，非padding捷径。metadata仍是C9实际slot/n/delay/budget，不能把meta第二列n当body或D。

冻结私有协作接口：
- _tyx_resident_floating_contact.make_floating_contact_kernels() 返回 project 的两个bank kernel。project(count, meta, linear, angular, wrench, vector, projected) 的参数顺序与旧contact_project一致，但按B=n+1、D=n+6完整求和；将结果仅加到vector[...,1]一次。数值算术只复用C9最终floating-only helper，保非finite传播。没有solve/integrator/额外external force channel。
- _resident_floating_contact.read_floating_contact_packet(plan, storage, worlds, bank) 返回按world的 dict(state,geometry,frames,energy)，与旧plan packet同形。从指定bank的roots/states/applied/FK/twist/energy及同bank ticks真实读回；计数经plan._read_contact，sync经原batch。checkpoint用原spec.checkpoint和真实held包；geometry为原FloatingTreeKinematics；frames为原ContactShapeFrame。只查询注册world/binding，静态mean reference和零速度沿旧协议。
- _resident_floating_contact.floating_contact_diagnostics(storage, worlds, arrays) 返回world→原FloatingTreeDiagnostic。arrays明确含同次linear/angular/mass/vector/entry_pose/info/diag，全部已由父在发布前读回；不调用active diagnostics、snapshot、HOST FK或求解。构造语义与C9 storage当前诊断逐字段一致。若父提取共享pure constructor，这个helper可直接委托，最终不得存在两种物理解释。
- NEW模块不持第二个runtime、不保存跨attempt全局缓存、不swap/commit/修改held。kernel lazy创建遵循当前runtime；其他分配归父原ExitStack，NEW packet helper不分配设备内存。

父group增加明确kind及body_max/velocity_max；ordinary/affine原字段可兼容保留，禁止用“不是ordinary即affine”。每world读取spec的实际B/D/N，未绑定浮根继续原stage_advance。旧采样、力律、事务、receipt共用，不按kind复制整条plan。

## 3. 同一步的真实输入、力和运动

1. 原锁固定epoch/src/controls，current-device-bounds query；有材料的全部合法pair沿原过滤。原stage_mechanics(src)一次得到M/b/J/输入外载。
2. 原HOST bridge以真实COM twist与共同load point速度采样。全部sites（含零力）先计容量；双方COM wrench进入各自body行。shape与机器人类型无关。
3. 上传wrench并用完整J转置投影一次；外部effort、bodyload、gravity、真实motor各只算一次。原完整floating solve/PD和旋转积分、candidate FK/twist/energy照用。
4. 原reject核可复用 code7；capacity拒绝保root/joints/applied/FK/twist/energy/tick/history，其他world正常完成。错误设备/读回/receipt构造导致整次不swap，合法retry完整重写候选。
5. 全部owned records及校验先于一次发布。原finish_contact_loads只运输入口材料/共同作用点；不得在candidate重采力。大anchor local_origin_m直接保存；无法表示的微小world几何仍明确拒绝。

receipt维数：input_external/contact_generalized D；selected_drive N×3、actual_drive N；law/uploaded wrench B×6。diagnostics保原mass、J、bias、total_external、acceleration、power/work/energy；新增 coordinate_kind='floating'、joint_count/body_count/velocity_count/position_count 以及同次 floating_diagnostic。general equation residual等于该原诊断字段。root广义力是动量交换，不叫固定支座反力。

## 4. 固定多案例与独立oracle

采用提案F01–F10，h=1/1024；原same-entry atol5e-6/rtol5e-5、40步atol2e-4、方程2e-5/原PD域不变。原失败/初始输入保留，不调质量/夹力/步长/阈值取绿。

- F01 root-only球/有限箱2N；F02 child/root受力与真实内部motor各支；F03同树兄弟双动态。
- F04转动工具、非零入步速度和damping；F05偏心COM及非对角主惯量的双接触力偶；F06真URDF fixed工具一次折叠。
- F07完整S/C/H（包括capsule端/侧/miss、CC、非盒HH），不是只球–球。复用C8固定几何输入/独立首载锚，不重跑C8套。
- F08非轴复杂分支n9/n16与body16/B17/D22；新n3双分支取提案明确的C8 D02几何，不能把它误记成已验C9 P05案例。大N使用零PD+独立full solve；非零box-KKT控制留出限参考已支持的N≤4。
- F09同一初态连续受载→分离零力→重入，不reset或按时间切force。
- F10 W1/W4/W32真混合及每档40步+19→21完整restore/replay；旧ordinary/affine/contact/scalar/未绑定floating混合要有代表例。

独立参考可复用原resident_floating_reference的time jets/full inertia，不导入Phy/TYX求M/b/J或solve。F01–F06/F08/F09有限sphere–plane/球–球的几何/法向律由固定输入独立构造并全窗推进；material/load功与KE分别报告。F07的2步几何/响应留出分开核：首几何载荷锚独立，设备same-entry动力学用实际上传wrench作已知输入再独立求full response，不能以这项循环声称几何正确。N≤4的非零PD由独立box-KKT。所有测试预配payload身份，异常保存已接受前缀/原输入/retained，不为保存计数重复物理。

最小首有效RED：旧HOST patch独立验证F01完整有限箱与2N，再经公开prepare_contacts触当前浮根拒域；查询新能力只能在函数内，不能module-import缺少新API造成假RED。首作者checkpoint先交此节点/独立数据，再写全矩阵。C9不重跑，C10首执行仍等其正式接受。

负控：非法body17；index16对B16；旧fixed/affine不存在body16；未排除直接父子、跨模型、world/owner/topology/source/law/mu/finite错误。两bank容量0/足额重试、project上传后、candidate读回、receipt构造latefault、restore/stale/close半构造；全部原状态和资源。受影响旧邻域只挑有限具体节点，禁止整套C6–C9重复。

## 5. 集成与执行门

A. 隔离纯作者可现在并行；依赖C9 API按当前文本，变更由父明确告知。不改当前run460快照内任何文件。
B. C9完整interpret/CUDA、独立native与saved/wheel收尾后，父核正式接受，再导入C10 NEW交付和首RED；父接OLD与最小正例。
C. 新矩阵一次串行interpret/CUDA；失败定位原请求，修复只补受影响节点及尚未完成部分，绿色同源不重复。相邻回归依据实际OLD差异。
D. 原生Newton只另冻F01/F03同FREE系统mu0/d0：真实contact和body force时态需先查本机1.7，不拟合材料。该原生driver/源码外consumer之后再授NEW范围；当前作者不扩文件。
E. 同次完整保存审计、源码绑定、源码外新wheel，交限域接受再启浮根affine/跨participant/设备窄相后继。全部执行只有父唯一串行slot与fresh入口进程guard。

本片成功意味着普通浮根设备运动与有限HOST接触反馈闭环；不代表affine浮根、持续摩擦、CCD、全GPU窄相或整个Newton功能对齐。超过两轮仅增加理论/摘要而没有公开可调用能力或具体失败闭环时，回到已声明的最小公共链路，不扩证明清单。

## C9 历史拒域测试在 C10 接入时迁移

父接入 C10 后，旧 tests/scale/test_resident_floating_collision.py 中 test_p07_floating_contacts_stay_explicitly_unsupported 的“所有浮根接触拒绝”前提失效；其空材料映射还可能让失败来自不相关检查。因此将父 OLD 白名单仅增加该文件的这一节点：在 C9 正式接受后，迁移为明确不支持的连续采样/CCD 请求，验证拒绝前后完整状态和资源不变；合法浮根接触由新 F01–F10 验证。迁移前不修改 run460/455 的源或固定覆盖；原 C9 已接受证据、外部 wheel 消费者和 saved 审计按当次源码保留，不复跑它们或把历史拒域当 C10 当前功能限制。C10 的源码外消费者将另行按新能力验证。

## 源码外消费者纯作者补充（2026-11-08）

原RD完成 d221065 四NEW测试交接后，允许在其原隔离树只新增：
- tests/scale/resident_floating_contact_wheel_consumer.py
- docs/benchmarks/p65-resident-floating-contact-wheel-validation.md

状态仍 AUTHORING_ONLY_UNEXECUTED。不修改已交接四文件，不merge父、不新增助手；
不运行import/collection/数值/pytest/static/build/install。全部C10执行仍等C9正式接受。
原robot同期只读本机Newton1.7的真实接触源码，消息交F01/F03原生对照建议；
未授native driver编码或执行，不能将外部匹配力当原生接触通过。

消费者沿C9公开wheel规范，独立复制到checkout外，python -I -B，原CLI
--mode / --output / --wheel-site / --expected-package-root / --expected-wheel /
--forbidden-checkout，完整installed成员/hash及loaded origins/owned-resource核验。
不导入repo tests或私有backend，不调用私有求解或HOST补状态。

固定W4顺序(503,91,1207,6)：F01 root-only球/有限箱、F02 motor原child接触、
C9外部消费者原ordinary P、原FreeSphere。后两world不绑定接触仍参与同一次推进。
前两world按F01/F02原模型/初态/载荷/控制/材料，h=1/1024，40真实步；
tick19完整检查点恢复后21步全部input/entry/state/FK/diagnostic/receipt/
material/candidate按原物理字段完整相同，epoch/计数另外核单调。
同一次真实入口力、广义力、root与joint响应/原工程门独立核；
独立轨迹只从原初态和自身状态推进，不读设备M/J/force作唯一期望。

另外F03原同FREE树兄弟P子体双球mu0原输入40步，动态opt-in；
完整两侧力/COM力矩、全树M/J耦合和总动量，独立原球-球法向律逐步推进。
独立公式可复制原参考的低级数学为本消费者自包含代码，但不得import产品求M/J。
F01/F02/F03首载literal、same-entry 5e-6/5e-5、M/方程2e-5、
原PD5e-5及40步2e-4不变，不能修改数据或阈值取绿。

W4两个真实bank各做浮根contact容量0拒步、邻world正常推进、足额合法重试；
保存原完整retained和最后accepted，不reset物理窗伪装连续。错误owner、
关闭/重复close、owned回执不变及资源0保留。公开unsupported sampling使用完整
合法材料对，不能靠空law错误伪通过。一次失败保存完整前缀和原输入，退出非零。
完整payload/summary/压缩bytes和SHA有界，作者先写固定身份与计数而非运行后挑选。
此补充只把已冻结能力放入真实安装环境验证，不新增物理域或原生接触结论。