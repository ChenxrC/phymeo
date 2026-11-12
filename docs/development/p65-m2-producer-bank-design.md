# M2：可变形设备顶点生产者与查询借用设计

状态：SOURCE_RESEARCH_ONLY / IMPLEMENTATION_NOT_GRANTED。基线为 b66504f3c82c868fddf3830bb0f9b2260bd3fd7b；本稿只读源码、手推和文档，不包含执行或性能结论。当前 M1 刚性边界查询已经通过原限域验收，本稿不重开该批次。

## 1. 首片结论

现有 TYX 节点、复合布和耦合程序具备真实物理更新核，但没有可供另一个长期查询对象借用的 accepted 顶点设备状态。缺口首先是状态所有权和发布事务，不是再写一个独立三角形距离例子。

建议分为两个相邻、可单独验收的小片：

1. **M2-P：持久距离约束节点生产者。** 复用原节点积分、完整有序距离投影和速度重建核，新增 accepted/candidate 双缓冲 owner。真实弹性边长改变来自原物理核，不能把 HOST 算出的第二帧重新上传叫作设备生产者。
2. **M2-Q：消费该 owner 的 accepted 表面。** 在同一 runtime 内接入 M1 已有三角面 bounds/BVH/距离核，首片只对静态球/胶囊做端点查询。每次 query 从当次 accepted 顶点 bank 生成面数据；不经 HOST 顶点快照回填。

M2-P 单独通过只能声称“真实可变形设备状态与事务”；M2-Q 通过后才能声称“真实可变形状态自动供边界查询”。两者均不是接触力、CCD、软体自碰撞响应或通用刚柔联合求解。复合布、StVK 体材料、R7 表示及 coupled owner 暂不迁移。

## 2. 当前实际状态寿命

| 实际源码接缝 | 已有能力与寿命 | 对 M2 的结论 |
| --- | --- | --- |
| [TyxMath.call](../../src/phymeo/backends/tyx_math.py)，第 76 行起 | 每次调用拥有 HOST 参数副本；CUDA 数组在 ExitStack 内创建、同步、读回，并在返回时释放 | 不能借用“上一次 call 的输出指针” |
| [TyxNodeMath](../../src/phymeo/backends/tyx_node_math.py)：predict / project_distances / reconstruct | 调用已有 TYX 核后返回 HOST 数组；_restore_fixed 还保留原固定节点的 FP64 坐标 | 可复用核与验收定义，不能直接延长返回数组的设备寿命 |
| [\_NodeWorkspace](../../src/phymeo/backends/tyx_participants.py)，第 220 行起 | predicted / candidate 是 HOST ParticleState；完整有序 sweep、lambda 和指标真实存在 | 不等于 resident accepted bank |
| [TyxCompositeClothProgram](../../src/phymeo/backends/tyx_composite_cloth.py)、[CompositeWorkspace](../../src/phymeo/solvers/composite_cloth.py) | 复合材料次序、路径账本与能量检查保留；各块经 math.call 读回 | 不能把节点首片的资格外推到完整复合布 |
| [RepresentedMath](../../src/phymeo/backends/tyx_represented_material.py) | R7 使用不可变 FP64 reference 加实际 FP32 displacement，调用仍经 call | 不是 FP32 绝对位置 bank；后续必须带完整表示 codec |
| [TyxCoupledXPBDSolver.\_substep](../../src/phymeo/backends/tyx_coupled.py) 与 [\_DeviceWorkspace](../../src/phymeo/backends/tyx_coupled_program.py) | pp/pv 等设备数组属于一次 substep / workspace；最后返回 HOST CoupledState，释放临时数组 | 不能跨 workspace.close 借用；particle/rigid runtime 也不能被假定为同一 owner |
| [SubstepWorkspace](../../src/phymeo/runtime/program.py)、[SceneStepper.\_substep](../../src/phymeo/runtime/scene_stepper.py) | workspace 是临时候选，result/验证后 finally close，外部再发布状态 | 不应在旧 close 中偷偷 promote bank 或改变旧事务语义 |
| [TyxWorldBatch](../../src/phymeo/backends/tyx_world_batch.py) | 已有持久双 bank、epoch、四树 FK 和 runtime lease；当前类型域不含 ParticleModel / 布 | 可参考生命周期，不能把节点伪装成 fixed tree |

原 ordered 节点核为 [integrate_particles](../../src/phymeo/backends/_tyx_kernels.py)、[project_node_distances / reconstruct_nodes](../../src/phymeo/backends/_tyx_node_kernels.py)。原 [distance_residual](../../src/phymeo/constraints/distance.py) 检查 constitutive residual；有限 compliance 时残余几何伸长本来就合法。

[TyxXPBDSolver](../../src/phymeo/backends/tyx_xpbd.py) 的 colored 顺序和速度路径不是上述 ordered participant 的可随意替代品。首片选定 ordered 路径，不在实现时混用两条法则。

## 3. 固定 Newton 1.7.0 的实际参照

本轮核对父 checkout 的固定安装源码，而非推测最新 Newton：

- 安装根：C:/Users/38347/.codex/worktrees/p65-parallel-integration/phymeo/.venv-newton-1.7.0/Lib/site-packages/newton；dist-info/METADATA 标记 1.7.0。这里没有据包版本推断未核实的 Git commit。
- _src/sim/model.py:1742 Model.state() 以 wp.clone 创建 State.particle_q / particle_qd；_src/sim/state.py:111 定义这些长期数组。模型 tri_indices 是与动态顶点分开的固定拓扑。
- _src/geometry/tri_mesh_collision.py:371 TriMeshCollisionDetector 接受 vertex_positions；:888 refit(new_pos) 使用当前设备顶点，重算 triangle / edge AABB，再 refit BVH。
- _src/sim/collide.py:2201 refit_soft_self_contact_bvh(new_pos)，以及 :2269 collide(state, contacts) 直接消费 State。:2780 的可选 full-surface edge/face passes 同样消费 state.particle_q，在已有 particle pass 后追加记录。
- _src/solvers/xpbd/solver_xpbd.py:301 _apply_particle_deltas 在 nongrad 模式可能交替写 state_in / state_out 的数组，:404 step 最后给出输出；requires_grad 路径另有分配行为。

可借鉴的是“固定拓扑 + 实际设备 State + 每步 refit”的连接方式。上述 Newton 代码并不构成 Phymeo 所要求的 accepted 不可变、晚失败回滚、epoch 借用和 UNKNOWN 清理合同；不能直接复制其数组交替写法并声称保住旧 accepted 状态。也不把 Newton 的完整 soft collision 功能列为本首片已覆盖。

## 4. 建议冻结的生产者域与 API

下列名称是待实施 API，不是现有导出。

- DeformableNodeWorld(key, model, initial_state, constraints, surfaces)：稳定 world key；原 ParticleModel 质量/外力语义；原 DistanceConstraints；显式面拓扑。
- DeformableSurface(key, point_ids, facet_ids, oriented_points)：每个三角面以三个稳定 point ID 定义，而非外部临时 row index；绑定所属 world、模型和拓扑 digest。
- TyxDeformableBatch(worlds, mode, settings, memory_limit_bytes, limits)：拥有一个 runtime、固定模型常量、两个物理 bank 和全部暂存数组。
- step(inputs_by_world, h, expected_revision=...)：首片完整 batch 一次事务；不引入 per-world 部分接受。
- snapshot()、reset(keys, expected_revision=...)、checkpoint()/restore(..., expected_revision=...)、close()：快照是 owned HOST 证据；恢复内容包括 q/v、tick/time、精确模型/布局绑定。revision 单调，不能用恢复旧 tick 来复活旧借用。

首片只支持 interpret / CUDA、固定拓扑、节点位移自由度、固定非负 inverse mass、外力/重力及有限非负 distance compliance。初态明确为 native FP32 world 坐标和速度，必须 finite、normal-or-zero；固定节点坐标不能静默从旧 FP64 值舍入。首片不宣称大平移精度、R7 reference/displacement 或旧全部 FP64 节点域等价。

完整状态包括两个 bank 的 x/v、每 world tick/time，以及操作修订号。lambda 在每一步按原 workspace 规则从零开始，不冒用上一帧 warmstart；predictor、alpha 和 sweep 临时数据不属于已接受历史。输入力是当前请求，不增设隐含 held-control 语义。

拓扑注册应拒绝重复 world/surface/point/facet ID、未知点、退化或重复三角面、跨 world 面、非法 dtype/shape、容量溢出与不支持的表示。定向拓扑与存储排列分别绑定；合法 row permutation 可有等价几何，但 native layout digest 不应假装相同。面是碰撞材料身份，距离边是物理约束身份，二者不能混为一张表。

初始资源上限建议限定 W<=8、每 world N<=512、E<=1536、每表面 F<=128；这是待冻结的有界首片，不是已验证容量。M2-Q 继续遵守 M1 更严格的分组、轴、候选和字节上限，不因生产者可装更多节点而扩大查询合同。

## 5. accepted / candidate 借用与失败原子性

内部 source loan 应包含以下完整绑定：

owner identity、runtime/device、world order、model/topology/layout digest、storage generation、source revision、active bank、每 world tick/time、point ID→native row 映射、facet ID→稳定三点映射、native representation 和只读 x/v device handles。

这些是 owner 在锁内创建的不可伪造借用，不是用户提供一个指针加 epoch。结果只保存 owned 数据与身份摘要，不保存可在作用域外再次解引用的指针。

1. accepted loan 在 consumer 完成全部设备工作和所需同步前保持；其间拒绝 step/reset/restore/close 及同线程重入。查询不能在释放 loan 后再异步读取 bank。
2. candidate loan 只能由当前内部 StepAttemptToken 派生，绑定 entry revision、唯一 attempt 和 standby bank；不得作为 public accepted query 的输入。
3. 第一片公开 query 只读 accepted。若将来要把候选碰撞作为物理接受条件，必须在共同事务内另冻 before-commit hook；不能先独立发布 physics 再假装 query 失败回滚它。
4. reset/restore 必须保留未选 world 的全部状态、原始时间与记录；所有请求先验证，准备完全部候选才发布。恢复到旧物理 tick 后仍增加 revision，旧查询和旧 loan 保持失效。
5. source loan 与 runtime/resource lease 分开。query plan 关闭前 runtime 不得释放；只有确定 CLOSED 才结束 lease。PENDING 保留原 cleanup owner，进入 close-only；UNKNOWN 强保留并隔离，不能自动再次调用状态未知的 free。

参考当前 [TyxWorldBatch.\_borrow_boundary_collision_state](../../src/phymeo/backends/tyx_world_batch.py)、runtime lease 与 close-pending 接缝，以及 [ResidentBoundaryQueryPlan](../../src/phymeo/backends/_resident_boundary_query.py)。复用的是合同和可抽取的纯 owner/lease 部件，不复制一份稍弱的 ExitStack 清理语义。

建议 step 的固定顺序：

- 在任何写入前校验 source/CAS、全部 world 输入、h、设置与完整资源预算。
- 从 accepted 只读预测到 pred_x/pred_v；设备复制 pred_x 到 standby_x；原完整 ordered distance sweeps 只修改 standby_x/lambda，原 reconstruct_nodes 写 standby_v。
- 每次规定的残差读回只作独立验收；绝不把 HOST 修改后的 candidate 写回当作结果。alpha 的两次原 FP32 乘 1/h 边界也必须保留。
- 最后一个 world 的状态、native status、残差、全部 owned 收据和必要清理/同步均成功后，一次切换 active、tick/time、revision 和 last_step。
- CAS 前任一失败保留全部旧 publication；脏 standby 不可读作成功。同步/释放结果 UNKNOWN 时不能假定安全重试。
- 独立 query 晚失败保留该 query 原 last_result；若 physics 已成功，那次 physics 仍是成功，不制造跨操作回滚。

这与临时 SceneStepper workspace 是不同的 opt-in owner；不能修改旧 workspace.close 来实现发布。checkpoint 格式也不能直接借旧 rigid 或 MuJoCo capsule 冒充 node 布局。

## 6. M1 查询连接：已有部分与缺口

已有 [compile_registration / check_accepted](../../src/phymeo/backends/_resident_boundary_source.py) 专门检查现有 batch/four-tree source；目前不能接收 DeformableNodeWorld。新 API 不应绕开这些检查，或伪造 body/FK identity。

M2-Q 最小改动是增加明确的 deformable source adapter：

- 注册时把稳定三点映射解析为 native gather indices；拓扑只上传一次。
- 每次 accepted loan 内，从实际 x bank gather 三角形，生成当前 bounds，复用既有 device BVH/refit、候选 pack 和 sphere/capsule–triangle 距离核。
- 首片静态球/胶囊常量由同一 plan/runtime 拥有，不跨借另一 rigid batch 的裸数组。未来 moving rigid + deformable 必须有共同 owner 或明确双 owner 协议。
- 每个结果保留 material facet ID、point IDs、source revision/bank/ticks、候选/raw/最终行和完整范围计费；不能把形变后的面按坐标重新命名。
- 保持 M1 对 feature、重复面、touch/unknown、容量和查询 failure 的原合同；不是 force manifold、penetration energy 或实体内外分类。
- HOST snapshot 可供独立测试参考，不能成为这条 query 的设备输入。

因此首片无需修改距离求解数学；必须新增的是生产者、真实顶点 source 编译及贷款接缝。generic old query 接口若必须抽象，应由父共享 owner 做窄改，保原 rigid registration 默认分支和已验测试。

## 7. 内存与执行成本

先冻结缓冲表，再授权代码；不能只报 x/v 两个数组而漏掉 predictor、lambda、读回和失败隔离。

首片至少拥有：双 bank x/v；pred_x/pred_v；inverse mass、当前 force/gravity；edge indices/rest/compliance、原 alpha 中间量、lambda；world tick/time/status；稳定点面映射；完整 HOST 验收副本和收据。若采用原逐行 trace，另计 E×18 个 FP32，不得在预算外悄悄分配。

可用于审阅的设备下限是双 bank x/v 的 48N bytes，加 predictor 24N bytes。它不是完整总预算。实现前按实际 dtype/shape、最小物理 padding、临时 overlap、失败后仍持有资源逐项求和，所有 allocation 前核峰值；UNKNOWN owner 的 bytes 仍计 live。M2-Q 的 BVH、候选、raw row、packet 和 HOST constant/cache 继续单独完整计费，源 bank 作为借用字节列出，不伪装成本为零。

首片允许每个 world 顺序 launch、规定的 HOST 残差/证据读回。只有物理 x/v 不在每步重复 HOST→device 上传；不声称零 HOST、全 GPU force、训练吞吐或优于 Newton。统计实际分配、上传/下载 bytes、launch/sync 和 live/peak；对不可观察部分标明边界。任何性能测量待功能资格后另冻结。

## 8. 固定多案例与测试分层

以下为实施前建议案例，不是运行结果；最终 fixture 在代码前冻结，不按输出调 compliance、步数或门。

**D1：解析有限 compliance 三节点三角面。** x=((0,0,0),(1,0,0),(0,1,0))；point IDs=(11,37,91)，facet ID=501 的一个定向面；单位质量，g=F=0；v0=(-1/16,0,0)，v1=(1/16,0,0)，v2=0；唯一物理边 0–1，rest=1，h=1/128，compliance=h²。原预测后边长 1+h/8，alpha=1，一次 ordered row 的理想 lambda=-h/24，发布边长 1+h/24。这是可直接手推的弹性伸长，不是刚性 FK。FP32 实际值按原节点数值门验证，不把理想有理数当作实际字节。

**D2：有共享边的四节点弹性片。** square 四点 ((0,0,0),(1,0,0),(1,1,0),(0,1,0))，point IDs=(101,113,127,139)，facet IDs=(503,509) 对应 (0,1,2)/(0,2,3)；四边加对角 0–2，rest 为原初态长度并经过原 as_compute 边界；有限 compliance=h²，上边两个节点有 +x 的 1/16 初速，下边初速零，g=F=0，h=1/128。完整连续 12 步；检查真实边长变化、每个面的当前顶点、完整 lambda/速度/残差和稳定共享点身份。独立原 ordered HOST 参考保留全部约束与顺序，不仅查 endpoint 像素或刚体变换。

**D3：稀疏 W2、固定节点与事务。** keys=(7,31)，分别使用 D1 与 D2 的注册，第二片节点 0 以 inverse mass=0 固定、其它节点单位质量；显式不同外力，合法 point row permutation 和面存储反序。正常两世界共同发布；最后 world 验收故障时两个 accepted bank、tick/time、last_step 均不发布。部分 reset/恢复后至少两步实际后继，未选 world 字节保持，旧 loan/source 失效。

M2-Q 在上述真实轨迹上查询静态球和有限胶囊，各有近面/远面、共享边附近、touch/分离，以及多面 BVH 真 refit。至少一例首次/后续间的真实面 bounds 与边长均改变；禁止只平移或旋转两张 HOST 快照充当 positive case。查询只报告原距离域，不把其中一例解释成碰撞响应。

测试分层：

- HOST：注册身份/布局、资源 exact cap 与 cap−1、旧 source/CAS、无效 pinned 表示、关闭状态机；合成只检事务逻辑，不冒设备资格。
- interpret / CUDA：完整 D1–D3；复用原节点 predict/reconstruct/serial multiplier 数值门和设置，不提高容差追绿；独立 CPU oracle 仅在测试侧使用。
- 生命周期：首/末 kernel、末 world readback、owned 收据构造、query 晚失败、reset/restore、lease 内 mutation、close 前/中失败。保 primary 与 cleanup 错误，未知释放不自动 retry。
- 数据路径：记录真实 allocation/upload；连续 step 的 x/v 输入只能来自 accepted device bank。只允许初始化、显式 reset/restore 上传节点状态；不能禁止合法 force 上传或谎称无读回。
- 源码外：独立小 W2 consumer，真实形变、reset/restore 后继；到 M2-Q 再增加 public accepted query。验证安装来源，不导入仓库测试 helper。

D1 只证明其对应的简单材料响应；D2/D3 和 M2-Q 不能省成 D1 的同一复制品。原 composite/coupled/R7 绿组不重跑，也不被新首片结果重新命名为已迁移。

## 9. 文件与执行切分

建议 M2-P 产品新增：

- src/phymeo/backends/tyx_deformable_batch.py：公共 owner、模型/状态/请求绑定、事务和生命周期。
- src/phymeo/backends/_tyx_deformable_buffers.py：有界两个物理 bank、暂存和精确计费。
- 如原 public device copy/status 操作不足，新增 _tyx_deformable_state.py，仅放 copy/clear/status；不得复制积分器或距离投影。
- 新专用 source/lifecycle tests、一个源码外 consumer 和实际结果报告。

M2-Q 后续由父冻结 _resident_boundary_source.py / _resident_boundary_query.py 的新增 source 分支，必要时新增 deformable adapter；原四树与资产来源 lane 保持独立。公共导出和旧共享 source 仍归父。新 producer 本身不改 tyx_math.call、SceneStepper、composite、coupled 或当前正在开发的 BALL/M1 asset 文件。

开工条件：父接受状态表示、完整缓冲表、上述固定案例和 lease/CAS 接口；唯一串行执行 owner 再跑新矩阵。M2-P 若无法复用原 ordered 核而必须改物理语义，或不能保住晚失败的 accepted bank，应先给最小反例并缩小接口，不开第二套近似积分器。两轮诊断仍不能缩小同一问题时交父裁定，不追加测试预算掩盖生命周期缺口。

完成标准是“真实非刚性设备状态 → 有身份和寿命的 accepted source → 已有通用距离查询”的实际连通与多案例验收；本研究文档本身不满足其中任何执行验收。
