# M-B2：非根连杆与变形表面实施合同

状态：**VALIDATED_ACCEPTED_SLOT_RELEASED**。用户在上一批收口后于2026-11-13再次要求继续；本批从父9c7433d7开始，没有继承已结束的07:30截止或旧执行session。按[已有研究](p65-multibody-deformable-next-design.md)补最高优先级的共同响应接线，现已完成限定源码及安装资格，详见[父验收报告](p65-articulated-deformable-report-20261113.md#parent-acceptance)。原M-B1完整结果复用。BALL标量约束组合与G2b预算按各自提案随后排队，不在同一共享文件内混做。以下保留原实施合同；不授予重跑已绿组或新域执行。

## 功能域与不变部分

- 每logical world一个原D1节点模型，加一个ordinary `FloatingRobotTree` 关节participant；允许原R/P非根连杆上的sphere/capsule，完整root–joint J/M、控制、偏心COM及真实入步速度。不是MuJoCo FREE+BALL路径。
- 已验M-B1多独立单body world仍支持，可与新单关节树world混在同batch；含多body participant的world只准一个participant。多个关节机器人同world另验，不通过本次删除所有guards放开。
- 总participants≤8、worlds≤8，每world总physical bodies≤33；原nodes/facets/sites128、query/solver预算、几何和数值门不增。完整body数量在Runtime分配前计入response预算。
- 原sphere/capsule–triangle查询、NormalLayer mu0、entry-sampled split、D1弹性及可选D6方程、resident J/M/积分与事务内核均不改。条件未知的near row仍拒绝，零载荷body仍推进。无新RR响应、摩擦、CCD、持续稳态或性能资格。

## 共享接口冻结

`DeformableResponseLayout`保原world/participant顺序及字段，新增：`body_addresses`为canonical `(participant_key, group_index, logical_body_id)` tuple；`body_slots[(key,group)]`为global slot；`participant_body_counts[key]`为完整body数；`articulated`标记任一participant多body。原`participant_slots[key]`现在指该participant首body，旧单body数值完全不变。`body_offsets`覆盖每world全部physical bodies。`multibody`在多participant或多body时为真，以复用已验通用transpose核；单body旧核/旧receipt不变。

source adapter仍验证所有participant，包括无collider/无contact者；spec.n+1、原topology/model/graph/anchor、真实native row/slot/owner/bank和world tick完整绑定。moving metadata的group来自原binding，必须与scene.body_owners一致；不得固定0，也不得使用HOST FK替代实际accepted FK。扩展source记录覆盖完整body域；旧M-B1记录字段/含义保持。

新articulated binding schema为 `tyx-deformable-articulated-contact-v1`，step schema为 `tyx-deformable-articulated-contact-step-v1`。新域world沿用`rigid_participants`映射，每项保持完整`rigid_wrench`/delivery/uploaded数组，形状为(body_count,6)；保首`body_slot`并另列`body_slots`和`body_ids`。site明列participant/body_group/body_slot。各点速、COM lever、advected material point和有限步work按真实group取值。整条树外载功仍由完整generalized external dot generalized velocity给出，不把每body视作独立自由体。

每participant交完整(body_count,6)载荷，未接触body交零行；原设备J-transpose投影、全coupled M及控制只执行一次。所有world、node和树候选及全部证据先完整构造，最后共同发布；任何晚失败保原accepted/history/last receipts。checkpoint与partial reset仍按完整world，UNKNOWN释放不重试。

response额外分配原式保持：`116*max(1,site_capacity)+12*N+24*B+4*W+8*(W+1)`，B改为全部physical bodies；原单体分支仍用4*(W+1)。实际数组nbytes必须等于预检，borrowed完整FK/tick按原唯一owner收费。不是完整进程/显存峰值。

## 作者和文件归属

父为唯一共享接缝和项目执行owner；只沿用当前两位可用原agent，现有隔离worktree从干净父9c7433d7建立新命名branch，保留旧branch、不reset。

| owner | 允许修改 |
| --- | --- |
| 原mtyx_t2_audit | `_deformable_response_layout.py`、`_deformable_response_source.py`；只静态地址与完整实际source接线，原来源检查不得弱化 |
| 原mtyx_t2_tests | NEW `tests/installed/check_deformable_articulated.py`、NEW `tests/collision/test_deformable_articulated_contact.py`；先提交固定cases/独立oracle/tests，再完善安装consumer；NEW简短cases合同与报告 |
| 父/root | `tyx_deformable_contact.py`、`builders/deformable_scene_lowering.py`及本批计划/文档；必要受影响旧测试仅修改精确原unsupported例为明确新positive，保其余原负控 |

不改原resident事务/力学核、MuJoCo路由、D6模型或原M-B1消费者。若需上述域外改动，先报具体源码缺口并冻结增量；不为方便造dummy participant、私有状态补写或调用两个独立step。

## 测试先行与执行

测试作者在首次项目执行前提交唯一固定输入：C为非零root/关节入速、非零驱动的R–P链，末端capsule接触；D为非对称分支R/P，两不同links同时受载，另body零接触。两稀疏world、不同非零anchor、输入倒序；每族至少8连续步与真实控制变化，独立参考从自己初态推进。独立几何/材料可复用既有测试低级算术；整树M/J/bias/FK/积分须来自独立原生或已存在独立参考，不调用产品helper自证。

完整记录查询/body身份、所有site、每body raw/delivered wrench、全M/J/bias/控制/投影广义力、节点材料/固定反力及双方实际后继。沿用M-B1力学abs5e-6/rel5e-5、几何distance1e-4 L/witness2e-4 L/normal2e-3、材料同入步2e-5、连续状态2e-4；若原树机械参考适用已有更具体门，先列其出处，不执行后改门。

负控覆盖错误group/body/source、同world混合tree+另一participant的前置拒绝、真实完整body预算与少1字节、末world/完整publication准备后故障、未接触body零载荷、stale query、原owner restore/partial reset后真实后继。与同声明独立owner比较，但不伪称跨owner checkpoint可恢复。

父将使用当前既有venv，先fresh entrypoint-aware guard与source/config快照，再新文件及受影响产品静态检查、完整新解释/CUDA矩阵、必要旧M-B1兼容，最后fresh wheel/source parity和源码外公共consumer。禁止预collection/数值probe挑参数，所有import、pytest、lint/format、build/install/profile严格串行。作者只文本/Git，不执行项目。

同一原因两次诊断仍无缩小或约60分钟无新证据，保最小反例转独立节点；不改质量/h/层参数/门追绿。通过后更新当前全景与续作；失败不能借旧M-B1绿宣称新关节域完成。无远程CI/push/mainmerge/环境清理，保留三份无关WIP和全部证据Junction。
