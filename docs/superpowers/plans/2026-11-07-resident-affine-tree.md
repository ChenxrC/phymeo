# C5 固定根 affine-mimic 树设备驻留：待审实施计划

状态：**AUTHOR_ONLY / PROPOSAL / UNEXECUTED**。2026-11-07，仅新增本计划；不授权实现，不报告 C5 测试通过或性能收益。作者基线 `5de33ea35001953e64122b8fff6e5c39a814db1f`，依据只读父当前 C4 源码及父 `884d474` 验收通报。C4 已由父报告 interpret/CUDA 各 101、共享 37、Newton 24 窗 960 步完成；源码外 wheel 消费者仍待父同阶段执行，本文不将其记为已通过。

## 1. 可复用事实与最小切片

本仓实际 HOST 类型是 `MimicRobotAsset`，并非另一个名为 `MimicRobotModel` 的类型。`tree.model` 保留全部物理关节/连杆，`matrix/offset_vector` 描述 `q_full=P s+c`。`load_mimic_urdf` 已支持同类型、非零倍率、无环、每从动轴唯一驱动的多行图，包括 fan-out 和链；无需重做 URDF 解析或把已有功能记为缺失。

| 已读源码位置 | 本片复用/限制 |
| --- | --- |
| `src/phymeo/dynamics/_mimic_coordinates.py:48` | 图排序、原始约束行 C、P/c 合成、位置交集、速度界、独立电机额定值；拒循环/重名/多驱动/异类型/零倍率/空行程。该编译器只处理静态数据。 |
| `src/phymeo/dynamics/joint_mimic.py:149,187,198,203,254` | full state、J P、Pᵀ M P/Pᵀ b、按反拓扑还原每行反力、真实 HOST 半隐式步；独立电机只施于 independent_indices。 |
| `src/phymeo/runtime/_robot_participant.py:138,184` | 现 HOST 场景分 full loads 与 independent drives，逐物理行受力后投影；不能将它的每步 HOST 准备上传冒设备驻留。 |
| `src/phymeo/backends/_resident_tree_world.py:67,155,222,239` | C4 ordinary world、FP64 原模型 token、FP32 编译表、N≤16、完整命令；现明确不接 mimic。 |
| `src/phymeo/backends/_tyx_resident_tree_kernels.py:13,93,272,536,592` | geometry/mechanics/solve/energy/finalize 已分开；全物理 FK/J/M/b 与末态动能可复用，当前 solve/state/history 假定所有 N 行独立，不能直接套用。 |
| `src/phymeo/backends/_resident_tree_storage.py:89,278,336,355,413,505` | 原 runtime/双 bank、同步前后检查、restore、借用 accepted FK；需将独立坐标一起纳入事务。 |
| `src/phymeo/backends/_resident_tree_binding.py:84,177` | 原 authored shape→真实 physical group 的 site 变换、材料/source/token、同世界 owner 规则；目前 from_asset 明确拒 mimic，不能只传 asset.tree 绕过图身份。 |

推荐首片：**同一原 TyxWorldBatch 中 N≤16 的真实固定根 R/P 物理树，1≤K<N 个独立坐标，至少一行既有合法 affine mimic**。每 world 可有不同 N/K/parent/图，保分枝、固定工具聚合的质量/COM/主轴和原碰撞材料。关系只连同种 R 或 P，倍率允许正负非单位，偏置可非零；首片不扩到任意 dense 线性约束、零倍率锁轴或时变 P。原普通树/旧 world 行为不变。

这不是一条 loader 薄扩展：还缺 **K 维接受状态与队列、设备完整投影、K 维耦合 PD、N/K 同步候选/反力/restore**。最小可运行里程碑先交两平移轴 N=2/K=1 的同 runtime 真动态与 full FK，随后同实现扩到矩阵中的 R、分枝、链与 N9/N16；不得以先导代替最终范围，也不创建第二 runtime。

## 2. 建议公共 API 与单位

以下为父待冻结的签名，不是已存在入口：

```python
world = ph.AffineRobotTree(
    asset, independent_state,  # MimicRobotAsset, RobotState with K entries
    kp=kp, kd=kd, effort_limit=cap,
    delay_steps=0, control_max_iterations=512,
)
action = world.inputs(
    target, target_velocity=None, feedforward=None, loads=None,
)  # owned AffineTreeInputs: drive[K,3] + full RobotInputs[N]
batch = ph.TyxWorldBatch({world_id: world, other_id: ordinary_world}, mode=mode)
batch.set_controls({world_id: action})
batch.step(h)
checkpoint = batch.snapshot((world_id,))[world_id]
fk = batch.tree_kinematics((world_id,))[world_id]
diag = batch.tree_diagnostics((world_id,))[world_id]
binding = ph.ResidentAffineTreeBinding.from_asset(
    asset, source_key, key=scene_key, world=world_id,
)
plan = batch.prepare_collision(collision_model, bindings)  # pairs/exclusions in model
```

- 独立顺序是 `asset.joint_names/independent_indices`，物理顺序是 `asset.tree.joint_names`；两者都随读回公开。R 的 q/v 用 rad、rad/s，P 用 m、m/s；电机 effort 分别 N·m/N，COM force/torque 始终世界 N/N·m。
- `kp/kd/cap` 为 K 长；cap 非负且不超过独立电机源额定值。`feedforward=None` 明确取零，沿 C4/DrivenRobotProgram 输入语义，不暗加 HOST 场景的重力补偿。`loads=None` 为 N 行全零；其 effort 是物理非电机广义载荷，force/torque 为各物理活动组 COM 外力/矩，不含本次电机或 gravity。禁止用 K 行外载冒 N 行，禁止从动电机目标。
- drive packet 仅 target/rate/ff；延迟 0/1 个**已接受的物理步**。初始历史为初始 s、零 rate/ff；外载始终当前步。完整 held packet、队列及 last applied 在 checkpoint 中保留。未接受步不吃队列；新动作不直接改 state。
- 复用原位置交集/速度界作**输入及接受域**，扩展后的 full q 也须在原关节界和 C4 `abs(q)≤4096` 工程域内。越界整 world 拒步，不夹回边界、不产生限位冲量，不称物理 joint stop。完整状态/命令先验证，FP32 编译后再核有限性、非零 gain 和可行域，舍入导致空域须具名拒绝。
- `tree_kinematics` 的共同 `position/velocity` 仍为 N 物理行，另有 `independent_position/independent_velocity` 与各自 joint_names；不是把 K 个 q 配给 N 个 link。FK 来自真实末态 device bank。summary 只给全体当前动能，不用 entry Mr 估末态 KE。

## 3. 同步物理路径与反力

记 E 为把 K 个电机放到 N 行 independent_indices 的选择矩阵，`Pᵀ E=I`；C 每行在 follower 为 1、driver 为 −a。静态编译保存原始声明、P/c/C/拓扑顺序与 FP64 token，设备离散索引必须 int32，数值表 FP32。图恒定，所以没有 Pdot 项。

每步从同一 accepted `s−/u−` 与其持久 full FK 出发：

1. 在设备保持 `q−=P s−+c, v−=P u−`；用 C4 原 full 几何/解析偏置机制算 **N×N M、N b、全部 full J**。不能删 follower 质量，不能以 K 行假模型替代真实树。
2. 以同一 q− 形成 `e_full = loads.effort + Σ(Jvᵀ F + Jwᵀ T)`，然后 `Mr=Pᵀ M P, br=Pᵀ b, er=Pᵀ e_full, Jr=J P`，保完整交叉块。gravity 已在 b，不重复施加。
3. 在 K 维复用 C4 原投影坐标式 bounded PD 与 Cholesky；`u+=u−+h Mr⁻¹(tau+er−br)`，`tau=clip(kp*(target−s−h*u+)+kd*(rate−u+)+ff,±cap)`。未驱动分支/另一独立坐标须有耦合响应。不是逐点标量解，也不把 C4 方法改称 active-set。
4. `s+=s−+h*u+`，只一次 drift；设备扩展 `q+=P s++c,v+=P u+`，重建全部 N 行 FK/末态 KE。不得先独立积分 N 行、再硬设 follower。原全物理载荷、独立/完整电机、投影载荷各保存一次。
5. 以入步 full 数据还原 `r=M(P a)+b−E tau−e_full=Cᵀ lambda`。复用 HOST 反拓扑扣除法求各原始 row lambda，保 follower/driver 两份贡献；链中兼具 follower/driver 的轴不得丢掉其中一行。检查完整平衡与 `C(P a)=0`、`C v−=0`；实际 constraint power 为 `r·v−`，每行另记 `lambda*(v_f−a_row*v_d)`。传动可对固定基座传力，不一般宣称全树线动量守恒。
6. 只有 reduced/full/FK/history/energy/reaction/status 全部可接受才设置 world accepted/tick，父统一一次 swap/epoch。reaction 与诊断是 latest accepted attempt 的入步读数；初始化、restore、最新数值拒步为 None。driver/sync/readback 异常不发布既有状态、epoch 或半条诊断。

保 C4 完整原始 clipping 门 `abs(tau_i−expected_i)≤5e−5*max(1,abs(tau_i),abs(expected_i))`，质量方程原归一门 `2e−5`，原 512 default / ≤4096 sweeps 上限。宽 inactive cap 或相消前大项不能放宽小实际 effort。full 平衡工程门建议 `2e−5*max(1,||M P a||inf,||b||inf,||E tau||inf,||e_full||inf,||r||inf)`；另用精确小反力 oracle 约束小项，避免只靠大范数掩盖错行。任一非有限/非 SPD/未收敛/域外都拒绝，无 HOST rescue、特征值截断或隐藏重试。

## 4. 存储与父接缝

推荐复用 **原 batch tree physical banks + 新 affine coordinate banks**。N 是物理行数，K 是独立数，R=N−K 是已支持图的行数；每 world metadata 必须分别保存，不能重用一个 `n`。

- 物理 `[A,Nmax,7]` 双 bank 仍保存 full q/v、full applied motor `E tau`、full generalized `E tau+e_full`；此类 world 的 full 历史三列为零，真正队列只在 K bank。full FK `[A,Nmax,18]`、full M/J/b、anchor 与 source group 复用 C4 数据路径。
- 新 reduced `[A,Kmax,7]` 双 bank：s/u/last tau/last generalized(`tau+er`)/delay triplet。held 分为 `[A,Kmax,3]` drive 与 `[A,Nmax,7]` full loads。不得在 N 行填 K 个非零目标后让 C4 solver 将 followers 当独立轴。
- 新 P `[A,Nmax,Kmax]`、c `[A,Nmax]` 与 C/反拓扑 metadata 只静态上传。另备 K 方阵/因子/向量、R 行反力 scratch。full 力学主开销沿 C4，新增量为投影和 reduced solve；不预称更快。N≤16 下最朴素投影也有固定有限预算，可按每 full 行仅一个非零独立祖先列做稀疏累加，但结果必须保所有 cross blocks。
- 建议父将 C4 storage 的资源/geometry/mechanics/energy 暴露成私有复用薄层，例如 `TreeDeviceStorage(..., coordinate_mode="external")` 加 `stage_mechanics(src)`、`stage_geometry_energy(dst)`；默认 ordinary 路径原样；静态物理表也复用 C4 编译片段，不构造虚假 DrivenRobotProgram 冒约束求解。NEW `AffineTreeDeviceStorage` 组合该层，拥有 K bank/投影/共同 finalize，不调用 ordinary `tree_solve`，不让普通 finalize 提前接受 N bank。具体私有签名在 tests-first 后由父冻结，不能以“调用原 stage_advance 后再修 q”代替。
- 推荐新 world kind 7/8 表示 affine delay0/1（须由父核现枚举再定）。旧 12 列 header 仍占位；旧 sphere/slider/rotor/ordinary passes 明确跳过，绝不落入旧 else 分支。所有 arrays/双方向 handles 由原 runtime 资源栈拥有。
- controls 先在锁内校验所有混合 world，再共同发布 held 候选；全部 dirty bank 上传后才清 dirty。partial upload 异常不启动物理、不清待传状态。
- snapshot 同锁读取 N/K 两接受 bank、同 tick、完整 held/历史/last applied，输出 `AffineTreeCheckpoint`；restore 验 world/kind、原 FP64 完整模型+关系+gains/domain token、N/K、full↔reduced 一致及全域，再共同 stage 两 bank/FK。任一 late-check 失败不 swap、不改 held；same-tick restore 也使旧 collision result stale。
- full FK 借用仍附 owner/generation/epoch/active/world slot。`ResidentAffineTreeBinding.from_asset` 核 mimic 图 token **及**完整物理 model/source shape；其物理点变换复用 `asset.tree` 的 folded sites，owner 为 N 行 physical group。collision 编译可委托 C4 full geometry，但不得丢图身份。bound 从 accepted full FK 直接写原 device buffer，不另上传 q/FK/envelope。
- 计数保 C4 原意义并新增 affine projection/solve/expand/commit launch、affine state/control/status bytes；借用 runtime 不 close。空 affine 子集零分配；mixed 数值拒步按 world 保原态，全设备失败无任何 bank/epoch 发布。旧 prepared scalar reader 对 affine 明确拒绝，不静默截 K 或 N。

父拥有旧 `tyx_world_batch.py`、`_world_step_layout.py`、`_tyx_world_kernels.py`、`_world_kinematics.py`、`_world_readback_plan.py`、`_resident_collision_plan.py`、`_resident_tree_world.py`、`_resident_tree_storage.py`、`_resident_tree_binding.py` 与 root exports 的必要复用接缝。C4 原 kernel 的物理公式不重写；若确需提取通用 solve 原语，也由父最小改动并保持 ordinary 正控。旧 HOST mimic/scene/接触律不改。

## 5. 建议新增文件与阶段

未来 allowlist 待父裁定，本次只创建当前计划。

| 新文件 | 责任 |
| --- | --- |
| `src/phymeo/backends/_resident_affine_tree.py` | public world/owned inputs、静态图与 full spec 编译、typed checkpoint/kinematics/diagnostic |
| `src/phymeo/backends/_resident_affine_storage.py` | N/K banks、父同锁/同 swap staging、restore/readback/borrow，不拥有 runtime |
| `src/phymeo/backends/_tyx_resident_affine_kernels.py` | full→reduced 投影、真实 K solve、full 展开、每行反力、联合状态 gate |
| `src/phymeo/backends/_resident_affine_binding.py` | 图+full asset proof；委托 C4 原材料/FK bounds，不复制 bounds 核 |
| `tests/scale/test_resident_affine_step.py` | 下列 A01–A07 力学/驱动/事务 |
| `tests/scale/test_resident_affine_collision.py` | A05/A07/A08 full materials/mixed/warm |
| `tests/reference_newton/robot/resident_affine_reference.py` | pinned 原生 direct/fan-out 参考与链式拒绝/显式变换记录 |
| `tests/installed/check_resident_affine_tree.py` | 源码外公开 wheel 消费者 |
| `docs/benchmarks/p65-resident-affine-tree.md` | 实际失败、环境/阶段/原始路径/限域，初始 UNEXECUTED |

先完整冻结案例、公共 API 和保存 schema；首有效 RED 在已成功创建 HOST MimicRobotAsset 与 reduced RobotState 后调用正常 `ph.AffineRobotTree`/原 batch，不能用顶层缺 module 导致 collection 失败。父真实 RED 后才授产品。先导两 P 同步 physical/independent 接受态→完整 A01–A08→native 参照→outside-wheel；不自动宣告“设备机械臂接触场景”。

## 6. 预声明独立案例矩阵

默认单位 SI，g=0，KP/KD=0，v=0，h=1/1024；不特别说明的独立 motor cap=16、各轴位置界[-8,8]、速度界16。这些是先行设计数值，尚未运行；父冻结后不得为过门改质量/倍率/预算/净空。所有原始 URDF、完整惯量/轴/pose、P/c/C 和输入都要保存，oracle 不调用产品 dynamics/FK 生成。

| ID | 固定输入与独立预期 | 主要判据 |
| --- | --- | --- |
| A01 两 P 非单位/非零偏置 | 平行 x 滑轨各固定根，质量(2,1)，COM=0，I=diag(1/8,1/8,1/8)，轴原点分别(0,0,0)/(0,1/2,0)。q1=2s+1/8；s=u=0，motor=3。P=(1,2)ᵀ，Mr=6，a_s=1/2，a_full=(1/2,1)，lambda=1，r=(-2,1)。另 motor0、follower COM Fx=3：er=6，a_s=1，r=(2,-1)、lambda=-1。 | 真首步 u=h*a、s=h²*a；全外载一次，两个方向虚功为零。40步恒第一输入，s_k=(1/2)h²*k(k+1)/2、u_k=k*h/2。|
| A02 fan-out 与负倍率 | 三个平行 x 根滑轨，质量(2,1,4)，COM=0，同 A01 惯量，三个原点 y=0,1/2,-1/2。q1=2s+1/8，q2=-s/2-1/16；motor7。Mr=7，a_s=1，a_full=(1,2,-1/2)，lambda=(2,-2)，r=(-5,2,-2)。 | 原始两行反力非零；joint/source 元素重排按名字还原相同；各实物质量都入动能。 |
| A03 真链式图 | 同 A02 轴/原点，质量改为(2,1,1)。q1=2q0+1/8；q2=-q1/2+3/16，故 P=(1,2,-1)ᵀ，c=(0,1/8,1/8)。motor7。Mr=7，a_full=(1,2,-1)，lambda=(5/2,-1)，r=(-5,2,-1)。 | 中间轴同时接收两行贡献；与直接连接q2=-q0+1/8的等运动图状态相同，原行lambda不同，不能按合成图丢原反力。 |
| A04 真 2R M/b 与偏置 | 复用 C4 T01 两杆完整物理参数；q2=-2s+π/2，s=0,u=1/2，motor1。q_full=(0,π/2)，v_full=(1/2,-1)。Mr=5/3，br=-1/4，a_s=3/4，a_full=(3/4,-3/2)，lambda=-1/8，r=(-1/4,-1/8)。一般式 Mr=5/3-cos(q2)，br=-sin(q2)*u²。 | 40步逐步独立更新公式，非零速度偏置不冻结；actual KE由两个杆COM/角速度逐体求和。共同轴循环/平移与非零s留出另保完整M/J。 |
| A05 R/P 分枝及 fixed tool | C4 T02 根R+两P，P2=2P1+1/8；独立(r,s)=(0,0)，full q=(0,0,1/8)。Mr=[[161/64,-1],[-1,5]]。非电机full effort=(0,1,0)、g0：a_ind=(64/741,161/741)。再同初态g=(0,-2,0)、同effort，加第二P COM F=(3,0,0)：br=(9/4,2)，er=(-3,7)，a_ind=(-1360/741,469/741)。另 C4 T03 fixed tool 附在 A04 两R末杆，改关系q2=-2s，q/v0：tool贡献ΔMr=13/128，总Mr=295/384，motor295/384→a_s=1。 | 两独立列交叉响应、当前COM投影/重力不双加。固定工具三原部件逐体独立oracle，聚合非对角惯量/原material到group只变换一次。 |
| A06 PD/限幅/历史 | A01 reduced Mr=6，h=1/1024，kp64,kd8,target1/8,rate0,ff0：a=8,d=129/16，非饱和 tau=8/(1+129/98304)；cap1时tau=1。delay1首步原hold、第二步用前target；follower外载首步即响应。宽帽子例源电机额定和cap均10^6、kp=kd=0、ff=1/128；再kp1024,kd0,s=1/2,target0,ff=512+1/128：实际小驱动门必须按原clipped式。另A05两列PD用固定1-sweep不足与512正常预算对照。 | 保 C4 原门，不以cap归一；full candidate域外不吃历史。未收敛用预声明确实有交叉且需多迭代的输入，在T1手推其首sweep残差后冻结，不依赖碰运气的失败断言。 |
| A07 图/维度/事务 | cycle/重复follower/未知名/R↔P/zero ratio/非有限/空交集/FP32 gain塌零或溢出、N17构造前拒；N9与N16包含真实物理行且K不同，不能只测2R。A01的s范围[-1,1]、follower范围[0,1]时交集[-1/16,7/16]；在边界向外候选拒绝。 | N9/N16解析由独立平行P质量块Σ m_i P_i²及非零offset材料求；加非平凡R/P树留出沿A05。两active方向均测partial upload、late restore、read_status异常、坏world与好world并存；同tick restore stale、checkpoint图不同拒、读回owned不可篡改、close所有callback/empty。 |
| A08 真驻留几何/多world | W1/4/32包含A01/A03/A04/A05、旧FreeSphere/Slider/oneR/ordinary tree；同body ID异world、同group多shape。每40步当前 full FK 的偏心球、有限capsule两端、局部旋转box/hull与固定对方做bounds/candidates。正负s、多周q在4096内、斜安装、共同平移、near-touch和有限端外miss。 | 原材料IDs/source与物理group一一核验；独立FP64 full链oracle，用实际stored s和静态P/c复算，不拿device FK反写真值。warm禁snapshot/envelope上传/newalloc，必要status和metadata如实计数。exact pairs只对超出工程padding的gap断言，保守额外候选单列；不宣称CCD/接触响应。 |

A04 的 C4 原杆参数：m1=m2=1，COM=(1/2,0,0)，I=(1/24,1/24,1/12)，parents=(-1,0)，第二关节offset=(1,0,0)，轴z。完整独立式 `M=[[5/3+cos q2,1/3+cos q2/2],[1/3+cos q2/2,1/3]]`、`b=(-sin(q2)*(2v1v2+v2²)/2,sin(q2)*v1²/2)`。A05 分枝/工具的原物理参数与逐部件公式直接复用已冻结 `docs/development/p65-resident-tree-casebook.md` T02/T03，测试需拷入独立常量，不 import C4 测试或产品 oracle。

A07 的容量输入固定为 N9/K3 与 N16/K4：全部物理轴为固定根 x 滑轨，origin=(0,i/8,0)，m_i=1+i/8，COM=0，I=diag(1/8,1/8,1/8)。前 K 行独立；i≥K 的 follower 引用 driver=i%K，倍率=(-1)^i*(1+(i%3)/2)，偏置=i/64。初 s_j=(j+1)/64、u_j=(j-1)/32，恒 motor_j=1+j/8。独立 Mr 对角为各组 Σ m_i*P_ij²，br=0，逐步式直接手推；另用 A05 world 验跨独立列的非对角动力学，二者不能相互替代。

A08 多周子例单独声明源关节位置界[-4096,4096]，取 A04 的 s=±2π、±256π（full q仍在4096域内），不沿用默认[-8,8]后绕开校验。最大域邻近及域外用stored FP32输入在T1明确冻结；有限大共同平移/近触/miss的原尺寸与工程padding一起声明后再执行。

除明确解析值，还用全坐标独立 KKT `[M,-Cᵀ; C,0] [a_full;lambda]=[E tau+e_full-b;0]` 检受力；P投影只作为第二种核法。位置偏置由 `C q` 的原行 b 校核，不能对非零offset误要求 Cq=0。数值门先沿 C4 实际 FP32 力学/几何工程预算，T1须列每一独立oracle的明确绝对/相对门与单位，再执行；不由首跑误差临时扩大。N16资源/编译成本只测量如实报，不假设缩维后更快。

## 7. Newton 1.7 参照与 wheel 门

主依据是本机已固定安装，不新增依赖：父 `.venv-newton-1.7.0/Lib/site-packages/newton`（Newton1.7.0/Warp1.17）；既有 `tests/reference_newton/robot/mimic_native.py`、`mimic_graph_native.py`、`resident_tree_reference.py` 可借 API/保存模式，不重跑旧组。

- `_src/utils/import_urdf.py:628,914` 读取 multiplier/offset 后调用 `builder.set_joint_mimic`；未知目标可 warning 后跳过，因此新参考必须断言真实 metadata，不能把无mimic导入当通过。
- `_src/sim/builder.py:6520` 的 `set_joint_mimic` 接受 PRISMATIC/REVOLUTE/D6匹配维度、同world/同articulation；**明确拒绝 mimic 链**。`_src/solvers/featherstone/solver_featherstone.py:102,175,953,1005,1051` 支持 joint-owned reduction，旧 sparse equality不是替代路径。
- `_src/solvers/featherstone/kernels.py:67,131,165` 真正将完整 H/forces投影、展开加速度；`_src/sim/joint_mimic.py:63,130` 更新坐标，非只可视FK。fan-out A01/A02/A04可按原关系独立 step，关节armature/damping/limit/drive的匹配设置逐项保存，真实COM wrench用原body_f线在前/角在后通道，control.joint_f只放独立电机。
- A03原链的真实import拒绝单独记录，不算动力学通过。可另明确生成**合成等价direct fan-out** `q2=-q0+1/8` 后跑相同运动/惯性输入；该路径是变换后参考，不能写“Newton原生链通过”，也不直接等同原行lambda。原图每行反力由完整 KKT/解析为主，若从native全轴balance重建须注明派生观测。
- CPU/CUDA至少覆盖direct、fan-out、2R偏置速度与分枝当前COM力四组，各自40步；不灌Phymeo中间状态。该版Featherstone还明确不支持joint_effort_limit/joint_velocity_limit，不能拿原生推进代替A07的能力域校验。Native PD默认法与本片implicit delay不同，首对照只用预声明motor/full external输入；若需同驱动比较，另记录控制协议，不凭相似命令声称逐步同律。完整raw保存q/qdot/body_q/body_qd/body_f/control、mass/labels/coeffs/版本/device/source，初始/入末时间清楚，native失败保留。

产品完成后父真实新wheel以正常公开 `ph.*` 在源码外、`-I -B`、新target `--no-deps --no-index` 安装运行独立消费者。混合batch至少旧自由体+ordinary2R+affine fan-out(N9物理行)+chain40步，完整full/reduced状态、原材料bounds/candidates、same-tick restore/retry、图token错配拒绝；所有已加载phymeo模块必须来自该wheel并核源码字节。不 import repo test/私有backend来绕公开API，不把本计划或C4旧wheel当C5包装证据。

## 8. 明确不含与后续

首片是**固定根被动理想机械联动的设备动力学与当前姿态宽相**。不含 floating 6D root/姿态、闭环几何约束/时变P、物理限位接触、机械间隙/柔性传动/效率、从动主动电机、设备通用接触力或抓取场景。现 HOST 浮树/mimic/接触入口继续存在，此片不改变它们。

后续 floating-affine 需要6+K速度、root COM偏心、惯性frame、世界角速率及完整root–joint交叉块，同事务接入；不能将root拆为独立刚体。后续设备接触需要真实same-step normal/material samples、全参与方 mobility和Jᵀ、摩擦/历史与实际位移共同发布，不由bounds候选自动获得。闭环需独立约束Jacobian/反力/秩/漂移处理，也不能拿无环mimic图冒闭环求解。

本次交付仅当前一份计划，小commit/clean后交父审定；全部未来测试/源码/计时仍 **UNEXECUTED**，没有自动开始下一阶段的授权。
