# P6.5 自动法向接触建立与分离 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development。用户要求冻结框架合同/独立测例后交子 agent 实现。每个独立阶段仅一次 fresh 最终 code review；本计划三个任务属于同一阶段，不另加重复 task/branch 审阅。

**Goal:** 交付真实原生两阶段法向接触，自动从分离进入、持续、分离，同时保留材料/机器人原式与显式history和全局事务。
**Architecture:** live native terminal impulse包装先构成可复用输入边界；新程序对全部配置node-plane行分别求held p与末冲量η，每次原生重算从同一子步初态；阶段账本与物理预算共同接受。
**Tech Stack:** Python3.11+/NumPy/Fraction；实际tyx interpret/CUDA现有kernels，FP32 compute/FP64 storage；不新增依赖。
**Spec:** [冻结设计](../specs/2026-10-29-p65-dynamic-normal-design.md)。

## Global Constraints

- 全机器人prismatic、选中粒子全动态、distance material或None、无限平面、固定允许侧、显式半径、μ=0、恢复0。拒绝revolute旁支/fixed节点/非零μ/缺真实trace/缺native capability。
- 新程序独立命名；旧 initial/free/material law与ledgers默认语义、SceneStepper时间语义、tyx pin/kernel数学不改。允许旧私有balance新增default=None末冲量项。
- 外层支持多子步，每子步完整两阶段；内层一个native substep。不要新增全帧最后子步计数器。
- 原始force与末impulse单位分别N、Ns，p/h与-J0.T p/h各一次；η只改速度。不把p+η塞回原force ledger。原式和物理预算分开。
- 所有candidate从原输入完整重放；第二阶段捕获真实pre状态、关闭前apply一次、关闭后只消费owned证据。不可host覆盖、零dt、关闭后调用或第二次物理推进。
- 同域所有点请求一次聚合，实际请求精度/转换也审计。history不是热启动，不带残留载荷；确定性pre-state与path不一致拒绝。
- 物理测试固定NumPy tol1e-8/tyx2e-5、outer128/128与inner128、relaxation1、正常energy/angular预算各1e-6 SI；发生合理不一致先报告证据，不事后放宽。
- 独立fresh最终review仅Task3派一次；数值/性能串行；不重做旧几何900观测、旧review或未修复CPP长期。无fake animation/pin/weld。
- 总H1/P6.5仍开放；有限法向生命周期不是摩擦/FK CCD/真实夹持，最终需继续用户已授权的完整流程。

## Review Focus

- native末冲量发生在base velocity未收敛、重复metrics或close之后，造成时间/载荷重复或资源越界。
- stage1与post ledger混用，微小FP64非FP32输出篡改、超大相消batch丢请求、未请求域/history被偷偷改写。
- 选中域材料通路或共享祖先响应遗漏；只按自由点求p却未重新执行材料。
- 闭合集合舍入分类、极小/零法向、半径和侧、历史active不符合几何、分离仍有η/拉力。
- 第N子步才触发terminal/trace/close错误后部分提交；事件/输入/history与重试不一致。

### Task 1: 原生末速度点冲量边界与独立账本

**Files:** Create src/phymeo/runtime/terminal_impulse.py、src/phymeo/validation/terminal_impulse.py、tests/test_terminal_impulse.py、tests/test_terminal_impulse_tyx.py。Modify src/phymeo/runtime/program.py（可选能力协议）、src/phymeo/solvers/articulated_coupling.py（无contact时能力及显式转发）、src/phymeo/__init__.py公共导出。不得改其他产品/现有测试/基准；不提前实现Task2。
**Interfaces:** 按spec提供 PointVelocityImpulses、TerminalImpulseProgram、TerminalPointImpulseWorkspace；新增 TerminalImpulseLedger(scene, before, requests, settings, context=BackendContext(), *, registry=None, history_fields=())，require(after)返回owned只读报告。Ledger选择requested particles/全prismatic robot，其他域只验未改变；history_fields显式用于完整copy/validation，默认空。返回max_residual、by_domain速度单位化方程误差、输入映射误差、配置保持与compute_precision；不负责接触互补或全步能量。
**Ownership:** Task1 child独占上述文件；完成后Task2依赖此API，不改Task1实现除已证正确性缺口并获主agent裁定。

- [x] API断言RED，测试PointVelocityImpulses所有权、非法shape/type/NaN/Inf/indices/local points、空请求、重复索引、source数组突变不影响signature；最小实现immutable校验。
- [x] 真实NumPy自由节点+prismatic机器人给定held p后的terminalη：捕获pre、配置bitwise不变、速度按解析改变；wrapper base完整阶段次数与末调用一次，重复metrics不重复apply。先RED后实现关闭前阶段包装。非收敛base/未prepare调用/未知domain/部分域失败/close正常与失败对照。
- [x] ledger独立RED：篡改q/x、重复η、漏掉robot反作用、未请求域或history变化、请求聚合丢小量、非SPD/非支持模型、unknownprecision；精确速度残差归一规则按spec新增节。保留原请求快照，不能只检查已舍入batch。
- [x] actualinterpret/CUDA两个预设解析例（free与chain，使用实验脚本的独立Fraction参考）及FP64 nextafter篡改输出必须拒绝；真实compute dtype/launch增加、原MaterialPathLedger只接受pre、要求显式tyx marker。六例实验不是新生产测试，实际运行新tests。
- [x] 透明显式history wrapper透传、两子步后段terminal故障与close失败state/history/input恢复+retry=fresh；empty requests不改变base数值/迭代输出。材料路径diagnostics不得在末阶段重算/假造。
- [x] 相关Articulated/SceneStepper/LoadReplay/participant定向回归一次；ruff/diffcheck，自查范围，提交产品测试。记录真实RED/GREEN/退出码和已知限制，不跑完整旧CUDA/900性能/独立review。报告路径写入本plan SDD目录，告知主agent最终SHA。

### Task 2: 自动两阶段法向接触、生命周期和物理验收

**Files:** Create src/phymeo/constraints/robot_plane_collider.py、src/phymeo/solvers/dynamic_robot_normal_contact.py，以及必要时最多两个私有辅助模块 _dynamic_normal_geometry.py、_dynamic_normal_balance.py；Create tests/test_dynamic_robot_normal_contact.py、tests/test_dynamic_robot_normal_physics.py、tests/test_dynamic_robot_normal_transaction.py、tests/test_dynamic_robot_normal_tyx.py。Modify src/phymeo/solvers/_material_contact_balance.py默认保持的terminal项、src/phymeo/__init__.py、scripts/smoke_installed.py。不得改Task1文件和其他旧算法；实际缺口先报告主agent。
**Interfaces:** Consume Task1 terminal API/ledger；produce spec RobotPlaneCollider/DynamicRobotNormalContactProgram和initial_history()、稳定row诊断。base.native能力必须真实调用，不type-check一个特定demo。selected所有节点自动生成rows，owned材料/预算和history规则严格按spec。

- [x] 缺API RED与scope/schema/owned tests先行；实现collider与程序绑定、签名、自动候选和base history合并/隔离。空节点域、反向顺序、键冲突、radius/side/μ/revolute旁支/fixed/trace缺失正常和失败对照。
- [x] 自由探头解析RED：h1/4,g0=1/8,v=-1,m1,M2，自动求p1/3与η1/3；检查中间状态、末状态、P=-1、K1/6和全部候选反作用一次。实现held原生重放与分阶段独立审计；预算不足和只位置律负控制必须拒绝。
- [x] 两节点材料链解析RED（spec所有分数值不改）：p3/7/η1/3，B响应证明同域材料；完整pre native/path重复一致性、末账和原输入功/角动量末矩。旧balance默认输出回归保持；故意按x0记η、有效输入混入外功、FP64展示舍入代替精确预算的负控制要RED。
- [x] 同一free程序连续三阶段：enter→无载persist→+1N节点拉离exit，再一步inactive且冲量history严格0；另更远接近但未闭合不施η。拒绝脏history、闭合区间不确定、深穿透，不允许epsilon推开或host修位置。
- [x] 两非递布独立物理窗与共享响应：spec明确T2support两个h、自由碰撞三个h解析终点，及新增材料链T=.5三个h独立implicit-Euler/连续弹簧参照（NumPy三h与tyx最细h，误差界见spec末尾）；三角材料、共享robot祖先、同节点两个不平行平面，行/域顺序和可表示共同变换；独立质量系统/动量oracle。数值预算失败不吞掉，不能只复述实现当oracle。
- [x] 至少两outer子步的trace/terminal/close后段失败，真实basehistory递增且每子步只提交一次，state/input/history全帧恢复retry=fresh；normal对照确保注入触发。终阶段即使已有best candidate也不得吞native/trace/close错误。
- [x] actualinterpret/CUDA解析+生命周期短集成并断言真实launch/两个阶段输入；有界CPP一个短例由Task3统一执行，不重复压力。源码smoke新增可重用free collision与解析输出；运行相关新/旧initial/free/material/LoadReplay/terminal tests，ruff/diff，自查提交并交主agent，不派review、不正式计量。

### Task 3: 一次独立审阅、工程证据与交付

**Files:** 新stage acceptance/engineering/review、PERF019及serial benchmark；更新progress/continuation/framework两条线和原NUM005边界。主agent负责，不并发改Task1/2文件。
**Interfaces:** Consume全生产API和两任务报告；输出最终版本真实测试/build/perf/远程证据。

- [ ] 核对两个实现报告和接口/物理范围；建立基准脚本后，整阶段产品+tests+script一次fresh最终review。Important先独立RED后修；只重复受影响验收，不派第二次review。
- [ ] 最终完整CPU两环境、最终CUDA全量（逐套串行）、Ubuntu新核心、interpret相关、有界CPP一个短例；每suite全新受核对basetemp。实际失败准确记录、不把挂起/中止算通过。
- [ ] ruff/docs/边界/compile/uvpip/锁导出与最终源码外wheel源码一致、新nativepip安装smoke。源码或测试改变只重验受影响证据，不伪造全量版本。
- [ ] 数值及review探针均退出后NumPy→interpret→CUDA分别3x100串行正式观测，取free establish与链建立小负载（每样本明确一个case，至少覆盖两case计量口径），记录held/terminal replay数、实际launch/上传下载、耗时及源/script hashes。迭代精度不同不作速度比；未测Warp不声称差距，不优化。
- [ ] 更新框架线有限法向生命周期验收与案例线未接入；保持摩擦/有限特征/FK CCD/弯曲/自碰撞/夹持/双臂/Three.js后续。记录遗留/host瓶颈/tyx区别；commit/push专用分支核对远程SHA/准确CI，不能将P6.5整体勾完成。

计划自查：spec所有API/阶段原式/所有权/后端/接触生成/history/物理与事务对应Task1/2，工程对应Task3。Task1 native机制独立可测，Task2求解器独立可拒收；单次review合并整阶段，避免重复审阅。主agent采用用户已授权的持续执行与技术裁定，不等待逐项许可。
