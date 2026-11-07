# 通用场景持续切向摩擦

## 当前验收状态：HOST 已验证，源码外 wheel 待验收

父于 2026-11-07 接入原 Single/Multi 场景的一次 prepare、真实全 M/b/J 和共同发布。F01–F06 共36项（分组执行）、新增材料记录回归7项、受影响兼容23项通过；18个完整物理窗口共1611步及一次快速倾斜拒绝记录已做同次保存审计。原输入、步长、力律与工程门未调整。本节覆盖下方历史 authoring 状态。

实际覆盖双机器人与自由体、有限杆两站点、固定 affine mimic、普通/affine 浮根、真实旋转工具、共同转动、卸载重入和四处失败回滚。固定根与浮根材料身份严格区分。根 marker 的新增 schema 短测为零载身份验证，不冒称根碰撞窗口。

验证记录在 [本次串行档案](../../artifacts/p65-generic-collision-20261107-01/)：

- 201 首组前6项通过，F05首步在材料面检查失败；206 根schema6项，207 非旋转20项，209材料回归7项及F05首步1项，210旋转3项，211共享兼容23项通过。各未改绿组未重复。
- 208 独立 RED 从原12站点保存了另一层面的正权重（约 (10^{-19})）。几何采样已有正确面内 chart；旧记录桥忽略它后从 rounded world point 反算三线性权重。cf97f8a 保留原 link/table 权重并映射原 vertex IDs，然后执行旧 key 换序；无 metadata 的旧圆体 fallback 及显式权重优先保持。没有扩大面或降低门。
- 197/198 和216/217最终静态检查通过；199/218限定AST核对记录纯排版/import顺序及一个未用局部名变更，未将这些变更当物理重跑。
- [219 保存审计](../../artifacts/p65-generic-collision-20261107-01/219-friction-saved-audit/summary.json)按预定19条文件成员核完整payload：18窗1611步，无缺帧；另一个拒绝文件保留原before与原因，回滚after由实际测试断言。没有重新求解或运行producer。
- 保存的最大材料运动差 (6.0581\times10^{-13}) m、作用点运动差 (6.0581\times10^{-13}) m、两者差 (2.7106\times10^{-20}) m；切向弹簧能量残差最大 (1.2956\times10^{-17}) J，逐站点最大 (6.4921\times10^{-18}) J。原门分别为1e-6 m和1e-8 J。这是该切向事务的能量账，不是整个引擎能量守恒或跨引擎性能结论。

**范围仍有限：** 显式材料站点、原支持的有限chart、HOST物理与原真实运动审核；未覆盖自动跨面持续身份、任意旋转步、GPU切向物理、软体历史，亦未复验/关闭F07原100g mimic持稳失败。独立wheel consumer已author并静态通过，等待本阶段新wheel实际安装使用；不能现在标包装完成。

## 历史：T2 authoring（保留原交接）


2026-11-07。**UNEXECUTED**：本增量为新适配模块及获准的root身份短测试，父将维护所有旧场景hook并统一执行。T1 e8cca784 已冻结F01–F06输入/门/完整记录；首个151公开RED已证明原两2 N法向前提后旧Multi guard拒绝，未重复运行。未声称F05转动门或F07原100 g失败已通过。

新增 `_scene_tangent_ports.py`：直接消费原prepared full M/b/COM J，按I、P或diag(I6,P)形成每个真实owner的一块响应；浮根驱动只有k motor，根6维补零，完整外部COM力/矩和gravity/bias各入账一次。freebody从其当下world惯量及原gyro/重力构造同口径速度预测。所有站点使用完整cross-W；固定root材料有真实姿态/零响应，浮根本体归入原机制，不建立另一积分器。

新增 `_multi_robot_tangent.py`：复用原registry、TangentHistory、prepare/finish核。当前entry events仅纯staging测量；移除selected旧切力→原normal+未选全部力预测→一次full-W切力→逐原site散回full generalized/COM/root载荷。多components和多site以当次完整材料/作用点记录定位，索引仅为本次回填，不作为持续身份。原normal、权重、势能、材料地址与未选力保留。原actual-material/application/work/KKT门和128上限不变；超出当前核支持的材料组合照常拒绝，不换律。

## 父旧hook精确接口

- `configure(scene, registry, *, stiffness_n_per_m, motion_tolerance_m=1e-6, energy_tolerance_j=1e-8, max_frame_turn_rad=.25)`：在原锁内、首步前一次性配置。校验真实shape、world、exclusion、原RobotContactMaterial；成功才写`_tangent/_tangent_state/_tangent_receipt`。不影响普通rigid ContactScene旧桥。
- `prepare(scene, prepared, measured, loads, details, external, h)`：原`_sample`后、原integrate前调用；返回`attempt.measured/loads/details`。Multi直接用原prepared映射；Single用父包装的一次已有`RobotPreparation`，key是scene._addr。没有第二prepare、PD、积分或live mutation。
- `attempt.finish(candidate_body, end_geometry_by_address)`：末态几何来自父纯`participant.kinematics(candidate)`，single fixed mimic必须先full_state后FK。返回`(candidate_history, candidate_receipt)`，仍不发布；parent在owned observation及最终events成功后与原state/time一并assign。失败或同base重试无消费；later base拒绝stale attempt。
- `owner_poses`区分固定根`robot_root/robot_root/-1`与真实浮根`robot_root/robot_model_link/0`。父已同意在旧两材料验证器只增加后者严格组合；本模块不伪装root为child、不改F04原夹具。schema接受不等于已验证root碰撞物理窗口。

所有新模块仅HOST适配，未改原物理求解、采样、CCD、default或Newton参数。有限chart正N未匹配拒绝；只真实卸载结束episode，重入冷启动；不自动最近点重认或换面。返回完整公共schema不扩展，不把事件账本称摩擦求解器。父执行结果与必要修复后再补本报告；当前没有新增测试/性能数值。

根身份补充专项仅用F04实际资产的root_marker和对侧原顶点构造**零载材料schema样本**：ordinary/affine两个root0正控、frame/index错配与bool索引拒绝，分别经过OwnerPose和TangentHistory.prepare；不积分、不造root法向/受力。它验证父两处校验接缝，不冒称已经跑完整root碰撞动力学。所有新节点仍UNEXECUTED；F04原窗口未改。
