# 下一批：直接球电机接触与资产设备边界来源

状态：本批已完成源码与安装资格；M2仅调研设计。2026-11-12 14:30开始，用户授权“继续下一批”，无新截止。父基线b66504f3；原三K1未跟踪文件保留。上批三个有限模块已完成源码/安装资格，不重新实施或重跑同源绿集。

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

证据根：artifacts/p65-parity-motor-asset-20261112-01。计划和M2调研本身不算功能通过；本批实际资格以[本批报告](../../development/p65-parity-motor-asset-20261112.md)及下方关闭记录为准。下面14:48段保留当时S0状态，不覆盖后来执行结果。

## 14:48 S0裁定与父接线

BALL-MC原作者S0确定六个固定连续案例MC1–MC6：sphere-plane normal/friction、mixed H/B/S对finite box、同球共享actor、双球运动分支互碰、disabled+正force clamp。W={7,41}、h=.001、24步、三段正/反/卸载，原独立对照和门不变；两solver路径含checkpoint/clone/reset、status/readback/末world发布/restore负控。手工51节点尚未collection，不计通过。父两wrapper接同tree五参数、runtime能力区分joint-motor；旧floating ABI保持，001/002四个父文件静态检查通过，尚无新物理结果。

资产源码确认URDF/MJCF原本保存完整OBJ mesh、原顶点与facets，阻碍只是两个from_asset factory有限convex准入。父只在factory接受原triangle/mesh，不修改全局_FINITE_SHAPES或旧convex compiler。NEW admission adapter签名：admit_boundary_asset(shape,binding,*,physical_spec,owning_spec)，返回完整group/body-local FP64 points与immutable provenance。注册对原完整shape（仅key可重映射）、原model、affine graph与真实physical group逐项核；不删facet、不hull改mesh。direct/static原语义保持，asset provenance加入完整source绑定和HOST数组计数。

该片是四类资产→TyxWorldBatch M1，不是ParticipantScene多bank→M1：后者的逻辑participant/内部机械world/lease映射必须独立接通。程序triangle与URDF原OBJ mesh来源分别记录，不伪称URDF triangle导入。父共享范围另含_resident_tree_binding.py、_resident_floating_binding.py、_resident_boundary_source.py；子只NEWadapter/tests/报告。M2本批仅研究实际producer，并未实施。

## 本批关闭：源码和安装实际结果

**BALL-MC与M1-A已完成；M2只完成调研设计。** 父产品最终d362799f，作者/审阅均释放，本批项目执行结束。原三个K1未跟踪文件保留；没有push/remoteCI、环境清理或新训练。

| 交付 | 源码实际结果 | 源码外安装 |
| --- | --- | --- |
| BALL-MC | 015完整51通过、016兼容6通过；没有物理失败或门放宽 | 043/044各112worldsteps、两后端首次PASS |
| M1-A | 023 HOST2；interpret按原访问顺序13+5+19=37、CUDA37；旧direct/static解释8/CUDA8 | 045/046各6worldsteps、3queries、24rows、2stale拒绝，两后端首次PASS |
| M2 | 19738ca0真实producer调研、M2-P/M2-Q分片与案例 | 未实现、未执行，不计功能资格 |
| 统一包 | 040 build、041 install、042完整516产品parity均RAW0 | 四消费者strict installed origins，61文件完整压缩归档 |

资产两次源码首错分别为独立fixed-capsule材点ID顺序、floating-affine前置拒绝文本；产品/输入/门不改。036 compatibility basetemp目录编号错误只在setup失败，037修目录后通过。各首静态失败和原日志保留，详情见[本批结果](../../development/p65-parity-motor-asset-20261112.md)。

下一批优先明确ParticipantScene逻辑participant到真实多bank M1来源的lease/epoch映射，以及按已完成调研冻结M2-P数组预算并实现持久节点状态。这两支可隔离编码，父共享接缝和验证串行。原BALL-MC、M1-A、已过源码/安装矩阵及本批证据审计不重复派做。

## 后续路线更新

2026-11-12用户将后续明确为三条主线：统一场景到设备碰撞 → 真实变形状态/查询/接触响应/连续载荷 → 机器人组合与异构GPU规模及公平性能比较。当前行动以[三主线计划](2026-11-12-scene-deformable-scale-roadmap.md)为准；本文件的本批关闭与证据保持，旧局部后续建议不覆盖新顺序。