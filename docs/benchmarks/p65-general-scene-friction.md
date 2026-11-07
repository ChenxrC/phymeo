# 通用场景持续切向：T2 authoring

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
