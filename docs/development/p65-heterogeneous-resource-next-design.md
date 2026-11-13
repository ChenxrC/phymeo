# G2b：异构规模扩展前的生命周期资源合同提案

当前后继：**G2b-A已完成限定源码/安装验收，G2b-B仍待冻结**，详见[报告](p65-g2ba-resource-gallery-report-20261113.md)。当前32models/worlds及六solver-bank数组预算按新合同生效；HOST/native/context等完整资源仍未交付。下文保留2026-11-13最初PROPOSAL_NOT_GRANTED调研时点及原8/16域，不作为当前实现状态。

建议先关闭可独立验证的 **restore 候选清理归属**，再建立真实 buffer 的生命周期账目与边界操作预留；完成后才选择 W32。无需先重跑 G2a，也不以新预算文档代替功能交付。

## 1. 已有能力和本次缺口

[G2a协议](p65-heterogeneous-scale-protocol.md)已记录原容量六格的源码资格；W16/B3、W16/B8各有两后端源码外完整27调用/432 worldsteps，W1/W8没有额外安装运行。原完整动力学、接触/执行器、连续独立 MuJoCo 与生命周期门继续有效。本提案不重登记旧矩阵或冒充新的执行结果。

[G1设计](p65-heterogeneous-batch-design.md)的 exact-source 分桶、全桶 prepare/validate/commit 已实现。[总路线](../superpowers/plans/2026-11-12-scene-deformable-scale-roadmap.md)要求资源资格先于扩容。[N7报告](p65-window-scale-report.md)的同构 S256/完整输出与 integration 输出不是异构 W256，也不是相同输出的 Newton 性能比较。

当前明确边界：

- `mujoco_heterogeneous.py:30,154–173`：8 declared models / 16 worlds，先检查数量，再 native 编译预检；按完整 source bytes/caps 合桶，而非 nq/nv。
- 原 leaf `_world_keys` 只校验非负 i32 身份、重复和非空，不能据此声称任意 W 合法。`mujoco_checkpoint.py:22–24` 另有128 MiB、256 worlds、每数组1,048,576项；aggregate checkpoint 的 `MAX_BUCKETS=8`、最终128 MiB也独立约束。
- 每 leaf 仍保原 native 模型资格。固定树矩阵宽度8；floating当前 nv<=35、nq<=45、native bodies<=128；normal 128 geoms/16 pairs/32 slots；一般mixed约束128 geoms/1024 pairs/2048 raw slots、实际contact<=128/EFC<=512。FREE+BALL的组合域还保自身16 pairs/32 slots，不能因进入同一个约束核继承一般T4全部容量。
- 当前计数 `device_allocations/static_uploaded_bytes/uploaded_bytes/downloaded_bytes` 是累计工作。facade `_observe_owner` 仅差分原 runtime 公共计数；其 `counter_scope` 已明确 boundary constructor 可能更换内部owner。没有逐owner live/peak GPU内存合同。

## 2. 资源从哪里来、何时共存

以下引用是本轮读取的实际源码位置，不是只凭模型尺寸估算。

| 路径 | 当前实物与寿命 | 必须计入的边界 |
| --- | --- | --- |
| `_device.py:11–80`；`_tree_device.py:12–113`；`_floating_device.py` constructor | 每bucket独立TYX Runtime，静态表、两份q/v/双分量clock、scratch、controls/bodyforces；stateful另两份act | accepted/candidate两bank已在同一个device owner内，不可遗漏备用bank，也不可再把完整owner误乘2 |
| `_normal_device.py:17–99` | 原base资源加完整geometry/raw rows/J、response及两块 slot×slot矩阵等 | 以实际nslot和width计，不以本步activecontacts计；原空padding/alias规则保持 |
| `_constraint_device.py:13–107` | 原base加raw slots、max(1,contactcap)/max(1,EFCcap)缓冲、全J、数块nv×nv矩阵 | 没有EFC²大矩阵；不能据此漏掉raw slot和full-J开销。logical cap0仍可能有1行物理padding |
| `_actuator_device.py:28–41`及tree/floating CSR分配 | act双bank、derivatives、actor fields；启用时的CSR与transmission fields | 未启用alias不是新分配；不同实际buffer不能按相同内容去重 |
| `_mujoco_tyx_runtime.py:436–534`；`_mujoco_tyx_mirror.py:93–163` | compact HOST integration mirrors；初态逐world短暂native MjData；full observer懒建一个detached MjData/leaf；pack和immutable复制 | CPU native arena不是每步W个长期MjData，但constructor/observer仍真实分配。CPU state与mirror中的共享view按真实owner去重 |
| facade `step:470–543` | 所有bucket先prepare，完整candidate mirrors/typed receipts/states累积，末桶成功后一次发布 | 旧publication与全桶新draft同时存活；失败后的原始证据另收费。普通step不clone，但不等于无HOST临时内存 |
| leaf `clone:1313–1358` | clone constructor先建初始device，再建复制状态的replacement，成功后关闭初始device | 暂时同时有源owner、clone初始owner、clone替代owner；最终公开counter不能反推该峰值 |
| facade `_boundary:585–668` | 按bucket依序clone、原checkpoint restore，再可选partial reset；先保留所有旧bucket和已准备的新bucket | 每次leaf restore/reset仍会建新device。直到全aggregate通过才交换；未选bucket不重建 |
| `mujoco_checkpoint.capture:295–345`与aggregate `_capture:167–190` | accepted bank真实readback、cpu_state/mirror/receipt复制、leaf JSON/base64，aggregate再次base64/JSON | 128 MiB是最终capsule限额，不是进程峰值。所有leaf、encoded字符串、JSON bytes可以共存；解码也有重叠 |
| `_resources.py:99–186`；facade `_close_owners:327–349` | buffer先于Runtime关闭；真正callback异常进入sticky UNKNOWN并强引用quarantine；上层保retired owners | 成功close才释放对应账目；UNKNOWN不能以“尝试过”当0，也不可重复native free |

令 `B_i` 为一个完整leaf device的**登记payload**（包含两bank、static、scratch、padding；不包含驱动/JIT/runtime隐藏分配），`D=ΣB_i`。无清理故障、按当前顺序准备selected bucket时，候选构造/restore/reset的登记payload上界应至少覆盖：

`max(max_i[D + Σ(previous prepared B_j) + 2 B_i], D + Σ(selected B_j))`，i遍历本次selected buckets。

再加已经保留的旧/UNKNOWN owner及其它真实登记owner。此式来自现有逐桶控制流，不是实测显存峰值；一个bucket source+两候选的三份重叠已经包含在D与2B_i里。若阶段未来改成并发准备，必须重新推导，不能沿用此界。

## 3. 最小独立修复候选：restore 清理失败的上层归属

`mujoco_checkpoint.py:499–513` 当前控制流：build replacement → verify bank → detached native observers；任一步抛 `BaseException` 后，若replacement已返回，则调用 `replacement.close()`，仅 `except Exception: pass`，随后重抛原失败。该分支没有加入 `runtime._checkpoint_retired`，也未把cleanup error附于primary。

必须区分两种情形：

1. **已经进入真实 `DeviceResources.close` 的 opaque callback 失败**：底层 `_unknown` 已把ledger及原owner强引用到 `_QUARANTINE`；buffer失败不close其依赖Runtime，后续只报原UNKNOWN不重试。这里不是“底层资源无人持有”的新泄漏。缺的是上层可见性/预算归属，不能重复造一个free重试方案。
2. **在真正ledger.close之前抛错**：例如bank.close的外层故障、尚未调用任何free。当前restore分支吞掉Exception并离开，未明确保留replacement给父owner继续close；底层未进入UNKNOWN quarantine。因此不能证明本分支具备可重试关闭的完整归属。若cleanup抛的是非Exception的BaseException，它还可能替换primary，现分支也未聚合。

建议独立窄修：候选失败时完整保留primary；只要候选close未正常返回，就把replacement加入原runtime退休清单并聚合原cleanup异常。原checkpoint/accepted banks/receipt/revision不提交。后续public close仍调用owner.close：底层OPEN可在外层故障撤销后完成；底层UNKNOWN只再次报告sticky结果，不能再次执行free。postcommit旧owner的失败仍是清理诊断，不得报告物理回滚。**本提案不修改这一分支。**

已有覆盖的准确边界：

- `tests/test_mujoco_tyx_resource_cleanup.py:55–161` 覆盖opaque callback before/after side-effect、strong quarantine、以及**测试自身仍持有ledger**的outer-before-close恢复；229/285两个actual测试覆盖constructor早失败与真实释放成功后再抛错。它们不覆盖上述restore局部replacement被丢弃的分支。
- `tests/test_mujoco_tyx_clone_cleanup.py:22–123` 覆盖clone copy/build/retire/cleanup_retry和source持有失败clone；不是restore替代覆盖。
- `tests/test_mujoco_tyx_checkpoint.py:312–333` 的late readback失败没有同时让replacement.close失败；468–548覆盖postcommit旧owner退休与公共close，而不是precommit候选清理失败。

因此这一条先成为独立生命周期修复及小回归，无须等预算系统或W32，也不需重跑G2a完整物理矩阵。父随后以276eb933单独冻结 [G2-R0 tests-first合同](p65-restore-cleanup-ownership-20261113.md)，现已完成178预期RED、185源码12GREEN、第四包193源码外12GREEN；见[正式报告](p65-restore-cleanup-report-20261113.md)。上文是修复前的源码发现，不再代表当前缺陷仍开放。该单独验收不扩大本预算提案的PROPOSAL_NOT_GRANTED状态。

## 4. 随后的可实现预算与公开口径

建议新增纯静态 allocation manifest + lifecycle reservation，不从累计计数反算峰值：

- 每个实际allocation列 `name/shape/dtype/nbytes/category/owner_id/bucket/source-binding`，由同一shape描述驱动预检和allocation，避免一份公式与实现漂移。实际 `_buffer` 注册时核shape/dtype/nbytes；所有alias只引用已登记owner，按实际allocation identity收费。先用Python整数检查溢出/非法维度，再创建np.zeros或CUDA数组。
- 原容量下提供可选 `resource_budget` 与 `resource_snapshot()`；建议budget仅命名 `device_payload_limit_bytes`、`host_known_payload_limit_bytes`、`checkpoint_limit_bytes`。缺省保旧接口；用于后续扩容的profile须显式给值，但**本提案不冻结任何新数值上限**。
- snapshot schema建议 `mujoco-tyx-resource-lifecycle-v1`：source/cap/layout绑定、owner generation、operation/phase、resident bytes、pending reserved bytes、retired bytes、unknown-release retained bytes、成功分配/释放事件、登记payload高水位、完整性与未知项。均为immutable数据，不公开可关闭的子owner。
- constructor在所有bucket GPU分配前预留完整manifest；step在准备全bucket draft前预留已声明HOST数组payload；边界操作在任何replacement构造前预留完整重叠上界。超限拒绝在发布前，不能释放旧accepted bank来腾地方后假装事务仍原子。
- reservation不是一次真实分配，不与allocated bytes双计；allocate成功转registered，释放成功扣减；unknown保守占用不扣减。下一操作若无法在剩余额度内完成应在开始前拒绝，不吞失败后继续多次堆积。
- 对checkpoint先声明raw数组/编码输出的保守payload上界，旧128 MiB/256world/数组项门保持。不能只验证最终字符串长度而忽略同时存在的leaf/outer/decoded数据；保存文件磁盘大小另列，不当HOST活内存。

**可保证**：纳入manifest的TYX数组payload、显式HOST NumPy/bytes payload、已知生命周期重叠和强保留对象对应登记字节；CPU interpret这些数组收费到HOST，不伪称device显存。

**暂不能保证**：TYX Runtime/CUDA上下文、编译器/JIT、驱动allocator/cache、Python对象/JSON解释器开销、第三方native隐藏临时空间、调用者长期持有历史receipt。`total_device_live_bytes/total_device_peak_bytes`和完整进程峰值仍允许UNKNOWN；登记payload高水位不能用这两个名字。

原`MemoryObservation`（`examples/compare_mujoco_tyx_observation_training.py:210–265`）记录公开native `nbuffer/narena/maxuse_*`与进程RSS/commit，可复用作**观测证据**；不能由进程采样或nvidia总显存分配到每个owner，也不能把采样未捕获峰值当硬上界。全局free/total只作运行环境信息。无需私有指针、Torch allocator旁路或新依赖。

## 5. 最小多案例与实施顺序（预算/扩容未获授权）

| 节点 | 固定验证家族 | 成功/停止条件 |
| --- | --- | --- |
| G2-R0，已单独源码/安装验收 | 新候选真实建立后verify/末world observer失败；候选close成功、外层before-call失败、cleanup非Exception；原真实UNKNOWN资格复用，不重新注入 | primary+cleanup完整保留，原accepted/state/control/clock/receipt/CAS精确不变；OPEN可恢复关闭，UNKNOWN无二次free；原same-input真实后继通过。复用已有结果 |
| G2b-1，原8/16容量内manifest与reservation | 原三已验模型族；零logical contact/EFCpadding；activation/CSR on/off；同bytes合桶/不同caps分桶；constructor、普通step、partial reset、whole restore及最后owner清理失败 | 独立shape表核实际alloc/free事件，不拿产品manifest自证；budget恰好够/少1字节在分配前拒绝；step无新device allocation；跨桶待提交HOST数据完整。只增资源行为和必要短后继，不重复原G2a24步矩阵 |
| G2b-2，另行冻结W32 | 先B3功能，再B8同W；保原三族非零输入/完整same-entry+continuous oracle、不同接触密度、晚桶失败与真实恢复 | 只有上两节点通过且所有登记payload phase纳入预算、隐藏项明确UNKNOWN后才授权；models仍8，W33明确拒绝，原per-leafcaps/数值/输出不改。不顺延到W128/256或声称单kernel |

G2-R0不以资源计数改变物理；G2b-1不优化clone分配顺序。若优化后可少建一次device，应另列行为相同的后继，不能同时改变峰值模型和事务语义而失去归因。

预算少1字节案例用预声明descriptor之和，不根据OOM临时选择阈值。真实OOM/同步/释放失败保留原始异常、已有accepted状态与完整attemptprefix；两轮不能定位则停止该格，不抬cap或减输出追绿。新安装消费者只核新增资源API/失败后真实后继与一个冻结扩容格；轮次、完整输出和性能边界预先登记。

## 6. 精确文件归属建议

父独占共享：`mujoco_checkpoint.py`（G2-R0）、`_mujoco_tyx_runtime.py`（constructor/clone/reset/restore/draft reservation）、公共导出；当前caller与checkpoint schema不能由两个作者同时改。

一个资源作者：NEW `mujoco_tyx/_allocation_manifest.py`、NEW `mujoco_resource_budget.py`；MOD `mujoco_tyx/_resources.py`（元数据/事件，不改UNKNOWN释放规则）、`_device.py/_tree_device.py/_floating_device.py/_normal_device.py/_constraint_device.py/_actuator_device.py`（统一shape/登记接入）。这些名称是建议allowlist，冻结前须逐项确认；无kernel改动。

一个facade作者：MOD `mujoco_heterogeneous.py`、`mujoco_heterogeneous_checkpoint.py` 的预留/aggregate公开记录；与父明确传递单一budget owner。若接入叶子的device-control packet，仅在原facade实际开放该入口后另列；当前HOST set_controls不借新预算任务扩成C2设备动作API。

独立测试作者：NEW `tests/test_mujoco_tyx_restore_cleanup.py`、`tests/test_mujoco_tyx_resource_budget.py`、`tests/test_mujoco_tyx_heterogeneous_resources.py`及一个installed资源消费者。测试不通过读取私有device状态来替代公共物理后继；注入点可沿既有内部故障接缝。原G2a文件/physics kernels/solver参数/模型域不改。

执行仍归父唯一串行槽。此提案与正在推进的多刚体—变形响应可独立排队；先补资源归属可明确关闭一个真实工程缺口，完整预算随后再决定规模档，不阻塞已验机器人组合，也不宣称Newton性能优越。

## Allocation route delta after M-B2, 2026-11-13 (text only)

The original source author inspected the current six device constructors while
BALL-S tests were authored. No allocation probe, numerical execution, resource
fix or new acceptance was performed; G2b remains PROPOSAL_NOT_GRANTED.

The leaf descriptor belongs after actual route selection and before
`MuJoCoTyxRuntime._build_device` packs HOST state. Derive it from the complete
plan, world count, body count, requested capacities and activation/CSR flags.
All six constructors ultimately use `_device._buffer`; a check only inside
that method is too late to prevent the preceding NumPy staging allocations.
Reserve against the descriptor first, then create each declared source/zero
array lazily. Existing plan/packed HOST arrays and new CUDA copies have distinct
owners and must be described separately.

| Actual allocation family | Descriptor detail that must remain exact |
| --- | --- |
| `_device.py` | Real plan table shapes, two q/v/clock banks, controls/applied/body-force/scratch. Equal bank contents still mean two allocations. |
| `_tree_device.py` | Mechanical width8; q bank follows packed nq (BALL can be10); static vectors use their actual shape. |
| `_floating_device.py` | width=max(8,nv); q and v dimensions remain independent. |
| `_normal_device.py` | Base constructor shares this owner; four static tables, full raw slots and both S-by-S buffers. extended_work is allocated even when its flag is disabled. |
| `_constraint_device.py` | Existing integer/float shape dictionaries are a direct descriptor source. Retain max(1,logical) padding for geoms/slots/contact/EFC capacities. Base scratch remains resident even when a later public field supersedes it. |
| `_actuator_device.py` | Appends two activation banks, two static tables, two derivatives and actor fields to the same owner, not another whole base allocation. |

Disabled CSR fields alias parents/gravity/body_force; enabled CSR creates four
new buffers. Ownership identity, not equality of array contents, decides aliases.
Constructor `static_uploaded_bytes` includes scratch and is not a static-table
memory inventory. Tree HOST body-position/body-COM views share a downloaded motion
owner; smooth-acceleration may alias qacc. Constraint download first obtains base
arrays, then replaces acceleration/constraint fields, so peak HOST staging cannot
be inferred from the final returned dictionary alone.

Interpret `_buffer` uses a HOST copy without CUDA resource registration, and
current close does not delete its array attributes. A surviving owner reference
therefore prevents claiming that HOST live payload dropped to zero. CUDA release
failure transfers ownership to failures/quarantine after removing the pending
entry: pending_buffer_count=0 does not mean UNKNOWN payload was released.
Carry byte metadata through those transitions; deduct only a successful release.
Driver/context/JIT and incomplete HOST temporaries remain explicitly UNKNOWN.

Suggested first development split: one resource author owns the shared
descriptor/budget primitive, all six leaf constructors and _resources; parent
owns runtime pack/download/clone/reset/checkpoint reservation boundaries and
public schema; a separate facade author owns cross-bucket reservations using the
same leaf descriptors. First qualify original8-model/16-world capacities and
short lifecycle cases. Do not start W32 or call the partial ledger total live/
peak memory until the named HOST overlap paths are covered.
