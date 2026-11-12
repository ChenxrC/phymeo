# 下一批：直接球电机接触与资产设备边界来源

状态：2026-11-12 14:30 开始，用户授权“继续下一批”，无新截止。父基线b66504f3；原三K1未跟踪文件保留。上批三个有限模块已完成源码/安装资格，不重新实施或重跑同源绿集。

## 本批交付与责任

| 线 | 原作者 / 新分支 | 交付 / 依赖 |
| --- | --- | --- |
| BALL-MC | mtyx_t2_audit / codex/p65-ball-mc-20261112 | 原fixed BALL JOINT/NONE电机与既有condim1/pyramidal3接触共同求解；新多案例、事务、安装消费者。先S0接口和tests-first；子改_tree_model有限准入，父改normal/constraint wrapper及runtime capability等既有接缝 |
| M1资产来源 | mtyx_t2_tests / codex/p65-boundary-asset-20261112 | 先研究四类from_asset原FP64 model/shape provenance与完整物理拓扑的对应，再冻结最小适配器和多案例。禁止删除来源检查或HOST当前坐标重上传冒设备输入 |
| M2首片设计 | mtyx_t2_source / codex/p65-m2-producer-design-20261112 | 只读现有真实producer与固定Newton1.7源码，落accepted/candidate deformable bank、epoch/lease/提交/失败合同与案例；本批不声称M2实现或通过 |
| 父 | codex/p65-parallel-modules | 既有共享接缝、独立全文review、精确执行grant、唯一串行测试与统一构建/安装 |

原worktree分别p65-h1-capacity/support/actuator。创建新分支从精确父HEAD开始，保留历史和未跟踪文件，不reset/覆盖；不新增助手。全部作者只text/fileIO/Git纯编码，不import/collection/数值/probe/pytest/lint/formatter/build/install/profile。

## 已确认的接缝，不凭能力标签推断完成

BALL normal/constraint wrapper已经通过TyxTreeDevice分配新moment表和实际scalar-force字段，且base.download可返回它们，但mechanics调用仍把transmission flag固定为0并传旧占位表；仅删准入将丢掉后两球DOF的电机力。父必须接入已验统一表，保floating ABI和旧scalar路径；复用原contact solve和同一发布。父capability也须区别direct motor与scalar contact。checkpoint已有typed transmission绑定，是否需要额外改动须由实际组合负控决定。

M1现compile_registration强制direct topology且拒_source_shape。四类from_asset当前走原model/shape token而topology=None；更重要的是多participant ResidentSceneBatch使用另一ResidentBatchTransaction bank，不等同TyxWorldBatch。先明确两级差异，不能把单asset资格宣称完整多participant接入。来源适配必须保原shape IDs/全facet/半径/revision/过滤、实际物理model/graph/group和accepted epoch。

## 成功与停止

- BALL-MC：固定树nv8/nq10/nu32和原pair/slot/资源/精度门。na0，直接BALL只JOINT/NONE/FIXED/BIAS_NONE；无FREE+BALL、activation、limit/dry/spring/equality扩域。多primitive、多运动体、normal/friction、球+scalar、共享actor、真实非零入步速度、反向/卸载、两个稀疏world、24步、晚失败/恢复。每步同入步native、独立连续native、原gear/功率与双方力矩并列；不复制CPU状态回TYX。
- 资产M1：在冻结后按四家族、静态/动态双来源、完整shape/facet、相同local IDs跨world、换source/model/graph/revision负控、真实连续步/恢复/旧结果拒绝、容量/清理失败验证。不是新的几何核，也不自动接入接触载荷。
- 每片原输入/门/容量保持。首错先定位实现/参考/API/保存层，完整记录；两次针对同一问题没有缩小障碍即保最小反例，暂停该片继续独立工作，不无假设迭代或改门追绿。
- 作者交接后父先文本核对，只跑新矩阵与明确受影响旧入口。所有项目执行严格串行，fresh entrypoint-aware guard，真实外来执行自然结束，不kill。
- 两实现线完成后构建一次新wheel；完整source/wheel/installed成员同源，从源码外跑新公共消费者。若一线实质阻塞，按实际范围独立交付，不以未完片冒整批通过。
- 不新增训练/G1长窗、不做Newton性能胜出声明、不清理环境/旧证据Junction、不remoteCI/push/mainmerge/发包，不扩P7/P8。

证据根：artifacts/p65-parity-motor-asset-20261112-01。该计划和M2调研不算功能通过。后续API/病例冻结与实际结果在本文件追加，当前源码结果仍以前批[报告](../../development/p65-parity-m1-assembly-20261112.md)为准。
