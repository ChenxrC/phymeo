# P6.5 两阶段动态摩擦实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 交付可复用、自动求两阶段法切冲量的零厚度机器人无限平面/距离材料程序，独立验证静摩擦承重、滑动和新接触。

**Architecture:** 复用原SceneLoadReplay、TerminalImpulseProgram与原式账本；新程序拥有显式μ映射、阶段摩擦law和history。原normal程序及其门禁保持。内部模块按绑定/工作区/精确法则划分，同一产品阶段只做一次fresh最终code review。

**Tech Stack:** Python3.11+、NumPy、锁定tyx实际interpret/CUDA；host FP64/Fraction审计，tyx设备FP32输出提升。uv/pip两路线。

**Spec:** [完整冻结合同](../specs/2026-10-29-p65-dynamic-friction-design.md)。本计划不能放宽其定律/物理/事务/支持域。

## Global Constraints

- 用户持续授权P6.5及专用分支codex/p6-5-handoff，不另建goal，不合main/发布包/扩P7P8。
- all-prismatic含旁支，radius_m严格0，原RobotPlaneCollider.friction仍只接受0；新的friction_by_collider精确键集合且值builtin finite float>=0、owned、签名包含。
- law=dynamic-robot-friction-split-v1；初态closed held容量μp，open held tangent=0；terminal旧closed容量μη、新建立容量μ(p+eta)，原式在各自真实pre/post速度上验。
- 实际world矢量输入双方各一次。保持全部native精度/材料/机器人/末速度/原输入功/按域L1角动量/成对力矩门槛；不删FreeParticleLedger或旧law，不host修状态。
- 每outer子步两阶段，inner仅1，basehistory原样演化仅提交一次；所有尝试计数，后段真实失败即全帧拒绝。代码缺口先报告主agent，不扩大编辑范围。
- 数值进程唯一owner，父/子不并发跑物理或性能；性能待全部数值/reviewer退出后串行。pytest basetemp全新独立且在artifacts内，不能复用被递归清理路径。
- 本阶段只一次fresh最终review；主agent按用户要求统一整阶段审阅，覆盖源码/测试/计量。实现者不再派任何子agent或reviewer。
- NUM005、TYX003、TYX004保持；无依赖修复不重复CPP长压力/改pin/私调FreeLibrary/回退/重启掩盖。未测Warp不称差距。

## Review Focus

- 已选held候选后调用末验收时，不能使用post滑速重验held摩擦；两面A/B独立反例专门区分，Task1覆盖。
- μp只能消费一次，初态几何classified与历史active不能混用；冷closed、open→enter、exit→reenter与脏history，Task1覆盖。
- 实际world映射、非单位n的精确切向/normal分解及一ULP FP32伪提升不能被latent投影或展示数掩盖；三维旋转/微扰负控制，Task1覆盖。
- 多面/共享祖先/同域材料反作用不能各自独立解后相加；独立合质量响应、材料链另一端、单原输入与terminal配对力矩，Task1覆盖。
- 不收敛与native错误不是可选候选；已有best后trace/close失败和后子步全帧恢复，Task1覆盖。CI时限修复不降低物理门槛。

## Task 1: 新生产程序、独立物理与事务

**Files:** Create src/phymeo/solvers/dynamic_robot_friction_contact.py，最多两个私有模块_dynamic_friction_law.py/_dynamic_friction_workspace.py；Create tests/test_dynamic_robot_friction_contact.py、test_dynamic_robot_friction_physics.py、test_dynamic_robot_friction_transaction.py、test_dynamic_robot_friction_tyx.py。Modify src/phymeo/__init__.py、scripts/smoke_installed.py（只增加新公共API的有限解析smoke）。报告写本plan SDD task-1-report.md。不改旧solver/kernel/participants/terminal或旧tests期望。

**Interfaces:** Consume现有SceneLoadReplay、TerminalImpulseProgram/PointVelocityImpulses、MaterialPathLedger、FrozenRobotLedger、MaterialContactBalance、_dynamic_normal_geometry。Produce spec签名的DynamicRobotFrictionContactProgram、initial_history、features/history/problem_signature和dynamic_robot_friction_contact诊断。旧几何collider不承载新μ；新owned映射是唯一摩擦参数源。若复用旧workspace继承，必须显式防止held审计误用post；不能为复用而改旧模块。

- [x] 缺API/schema/owned RED：明确μ键集合、builtinfloat非负/NaN/bool、collider key/radius0、allprismatic旁支/全动态材料/trace、empty selected node domain、history冲突。实现新绑定、签名和稳定rows。源数据修改不影响owned模型；不新增案例专用分支。
- [x] 静支撑/低μ解析RED后实现held联合法切求解和精确原式：spec h1/4,m1,M2，p1/4，μ2→t=-X/4全静；μ1/4→t=-X/16,vx3/16,x3/64。native terminal-only x1/16/vx0应不能作为高μ生产接受；至少一个真实负控制不只assert相同实现公式。
- [x] 未固定材料斜撞RED后实现terminal联合法切求解：p3/7、t0、eta1/3、tau=-8X/21及spec全部分数位置/速度、K+U1135/882、Py=-2，B保持open。验证给定系数由新程序自动解出，不调用研究脚本填冲量。全pre/path/metrics/history与selected held一致，actualterminal只施一次。
- [x] 多面A/B和实际world审核RED：A自动p1=p2=1,t1=-.5X,eta2=14/5以及post(-17/25,56/25)，允许总t·u+17/50；注入B的held正功4221/6875须拒绝。normal+切向交叉响应、实际world cone/最大耗散sqrt包围、方向/1ULP/负容量等控制；独立聚合terminal质量响应和全world成对力矩，不用观测dv自证。
- [x] 稳定生命周期/history与共享耦合：enter→persist→exit→inactive→reenter、冷closed不依赖旧active；分离phase_world和initial_closed严格0；旧history only record不warmstart。共享祖先双节点、同节点斜面、未固定材料三角、三维滑动、变换及域/面/节点顺序，含正常/失败/退化对照。原式预算用actualworld，原非接触输入功不能混入接触输入，按域角动量不得相消。
- [x] 连续/步长独立参照：NumPy T2静/滑两μ、h1/16与1/32，v=3T/4,x=3(T²+Th)/8及连续误差减半；actualinterpret/CUDA为T.5、两h同参照，不混称设备T2。至少两outer子步，模型/输入不变，多次重新建程序重复结果；不是整体递布精度证明。
- [x] 事务及故障RED：第2/末子步材料trace、terminal、ledger预算或close失败全帧state/input/basehistory/自身history恢复，retry=fresh；已有best后失败不吞，半构造资源释放，所有actualtrials正确计数。终阶段pre故意微改、错x0末作用点（非纯Y切向运动控制）、能力借名/FP64nextafter要有正常对照。
- [x] 单次相关回归与真实后端：运行新四文件及旧normal/initial/free/material/load/terminal/SceneStepper受影响集合；actualinterpret/CUDA包含新两解析、两面、生命周期和上述短窗并检查真实mode/launch/FP32；一个boundedCPP留Task2统一。旧normal程序测试及期望不改；新μ全0安全例与旧normal状态原tol内一致。ruff/diff与source smoke，真实命令/RED/GREEN/未执行/版本写报告，自查后提交本任务，释放唯一数值槽。不要跑旧完整CUDA2216或900性能、不要自行review。

## Task 2: 主agent一次review、工程与交付

**Files:** 新scripts/benchmark_dynamic_friction_contact.py、docs/benchmarks/p65-dynamic-friction-{acceptance,engineering,review}、PERF020及raw；更新progress/continuation/framework双线。主agent不改实现者正在持有的文件。

**Interfaces:** Consume Task1公共API与诊断、全部报告；produce准确版本审阅/测试/wheel/CI/性能证据。benchmark固定静支撑与材料斜撞两小case，各mode3x100交替，各case150+3warmup；不把研究预设向量当自动摩擦性能。

- [x] 接Task1报告、核对diff和物理范围，准备原生自动基准最小smoke；全部实现数值退出后派本阶段唯一fresh最终review覆盖Task1+benchmark。Important先独立RED后修，原门槛不变；不重复task/branch review，不重开旧normal审阅。
- [x] 冻结最终src/tests/scripts快照，串行最终完整core311/nativepip/CUDA、Ubuntu新核心、actualinterpret受影响集和一个boundedCPP短例。仅按真实变化重复必要证据，不把前版数量混成最终全量；每个套件新basetemp。旧CUDA长低μ已通过，只有新产品改变才由一次最终全集承担回归，不额外跑旧专项。
- [x] 质量ruff/docs/publictyxboundary/compile、uvpip/nativepip/锁导出，构建wheel并全部源码比对，源码外新nativepip安装smoke。只修改文档或计量不造成物理证据失效；真实失败先分类，未执行不能算通过。
- [ ] 所有本地数值/reviewer退出后，NumPy→interpret→CUDA各3x100正式串行，记录各case真实counts/tol/源码脚本hash/native执行与上传下载。初态每次重置非连续轨迹，整step含审计/断言/close排除factory/finalclose；不同精度/迭代不称速度比，只记录不优化。
- [ ] 更新框架摩擦有限支持与综合案例未接入双线，分开legacy/perf/tyx问题，提交push专用分支核对远程SHA及准确CI。30分钟CPU timeout已实际aa8ec42验收成功；若新阶段触及时限另凭证据判断，不skip测试。完成此切片后继续有限特征/FK保守CCD，不把完整H1/P6.5勾完成。

本次完整数值与质量/源码外wheel已全部exit0，见工程JSON；锁导出沿用同未变锁文件此前核对，不冒称pipeline重做。正式NumPy/interpret完成、CUDA仍运行，最后两交付框未勾。后续切片执行用户最新[分层验收规则](../../development/p65-validation-policy.md)，不照抄本计划的每切片全量模板。
