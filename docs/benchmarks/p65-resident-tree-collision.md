# C4：完整固定根 R/P 树的接受设备状态与有限形状宽相

状态：**CODE_READY_TESTS_PENDING / UNEXECUTED**。本报告记录纯文本作者实现及已冻结的接口，不代表物理、设备、Newton、静态或 wheel 验收。父已保存第136项有效 RED：真实 T01 模型、program、命令和输入记录先成立，随后正常公共入口缺少 `FixedRobotTree`，1 failed / 0.63 s / RAW1。不是模型或数值失败；不重复该 RED。

## 范围及文件

本片沿现有 `TyxWorldBatch` 的 runtime、资源栈、锁、slots、active、ticks 与 epoch。一个 world 是完整的固定根普通 R/P 树，真实活动行数 1..16；允许支链、多个根活动行、非主轴、一般固定安装、COM 偏心及按主轴表达的真实惯量。已由普通 URDF 折叠为活动组的固定工具使用其原 authored sites，不删除工具质量或更改材料归属。每 world 可有不同 n/parents/模型和控制。

五个 NEW backend 文件的职责：

| 文件 | 职责 |
| --- | --- |
| `_resident_tree_world.py` | 拥有的原 FP64 模型身份、FP32 静态表、完整命令、typed checkpoint/kinematics/diagnostic |
| `_resident_tree_storage.py` | 父 runtime 下的持久双 bank 和 scratch；stage、读回、短期借用；不发布 active/epoch |
| `_tyx_resident_tree_kernels.py` | 设备 FK/J/full M/解析 bias/COM 外载、耦合限幅驱动、一步漂移、末态 FK/动能、拒步与 restore |
| `_resident_tree_binding.py` | 真实活动组与原材料/source 校验；普通/折叠资产的一次静态局部变换 |
| `_tyx_resident_tree_bounds.py` | 同接受态 FK 的有限材料范围，直接写原 device bounds，再交原 BVH |

父拥有的旧接缝、公共导出与新事务测试另计；作者没有改旧源码或另造 runtime。原已冻结的两 scale 专项与一个独立 Newton 参考脚本保持输入、oracle、门和容量。当前没有新实测 pass 数或性能收益。

## 公共输入与单位

正常入口为 `FixedRobotTree(program, state, control_max_iterations=512)`，随后原 batch 的 set_controls、step、snapshot、restore/reset、prepare_collision。program 为普通 `DrivenRobotProgram`，state 保完整 SceneState。首域只支持 0/1 个物理子步延迟，实际 R/P 坐标均要求 stored |q| <= 4096；创建、restore 和候选统一检查。N17、stops、mimic、浮根、未知控制域/尺寸、不正常有限 FP32 输入明确拒绝，不使用 HOST 补解。

每行完整 command 为 `[target,rate,feedforward,generalized,Fx,Fy,Fz,Tx,Ty,Tz]`。前三列可延迟，当前广义外载和世界 COM 力/矩当步投影一次；重力进入 bias 一次。R 的 q/v/effort 为 rad、rad/s、N m；P 为 m、m/s、N；COM 外载为 N 与 N m。没有一个虚构的统一 q_m。summary 的树项只有当前 q+/v+ 的全树 `kinetic_energy_j`。

snapshot 保全 n 行位置、速度、实际电机 `applied_effort`、未减 bias 的 `generalized_applied`、完整 held n×10 及 delay1 的 n×3 队列。`tree_kinematics(keys)` 提供末态 link/COM 世界位置、rotation、实际 q/v、joint_types 与 tick。返回值拥有不可变数组/tuple，不能反写内部状态。

`tree_diagnostics(keys)` 是最近成功 attempt 的**入步** J/M/b、即时外载投影、实际驱动、加速度、残差与 entry/exit tick，不是永久 accepted 历史。初始化、对应 restore、最新数值拒步或设备异常后的相关树为 None；已返回的 owned 记录不受后续覆盖影响。HOST 输入校验失败且尚未启动 attempt 不使旧诊断失效。诊断及状态读回真实计数，warm 区间不调用。

## 完整力学与有限求解

设备 FK 在同一固定 anchor 的相对世界方向坐标中递推；世界平移最后只加一次。每个真实组 COM 的全 J 包含全部祖先，mass 为逐组 `m Jv.T Jv + Jw.T I_world Jw` 之和，保跨关节和跨支链块。解析速度偏置递推包含转动轴导数、P 关节 Coriolis、COM 切/向心加速度及 `omega × I omega`；没有数值差分。惯性、重力和当前姿态外载全部来自同一接受 q/v。

完整 M 做 Cholesky，再解各列获得完整逆质量响应。没有对角替代、正则位移、特征值裁剪或 HOST rescue。限幅隐式 PD 使用全耦合投影坐标更新；d=0 的已知电机力也进入所有交叉响应。所有真实 effort 再按原 clipping 式检查，保每轴 `5e-5*max(1,|tau|,|expected|)` 的量纲门与归一质量方程 `2e-5`。宽 cap 不放宽小实际力；大项相消使用原式而非相消前 a 或 d*u 归一。

迭代上限为原 1..4096/default512；内部继续细化至实际 effort 尺度 1e-6，资源末尾仍只依原5e-5/2e-5门接受，不新增更宽门。无 PD 的已知电机行只需一次全方程检查，diagnostic iterations 可为0。任何非有限、中间不正常算术、非 SPD、未收敛或候选坐标越域均有状态码并整 world 拒绝。

一次求解后仅漂移一次 q+=q+h*v+，重新计算 q+ 的 FK 和逐体动能，再由父统一发布。候选失败同时复制旧 n×7 状态、FK、能量、tick；不只恢复 q 而消费新控制历史。

## 父接缝精确合同

`compile_tree_world` 返回不可变 TreeWorldSpec；原完整 FP64 `model_token` 与数值 FP32 表分开。parents/kinds/meta/几何 rows、groups、offsets、ticks 为实际 int32。静态 HOST 转换和原模型声明不充当当前状态 FK。

TreeDeviceStorage 通过父 `_buffer/_launch/_device/_read/_upload/_sync` 工作。主要维度：meta[T,4] 为 world_slot,n,delay,budget；static[T,Nmax,16] 为 offset3/axis3/COM3/I3/mass/kp/kd/cap；zero[T,Nmax,9]、gravity/anchor[T,3]。状态双 bank[T,Nmax,7]，FK 双 bank[T,Nmax,18]，动能双 bank[T]；command[T,Nmax,10]。J 为[T,Nmax,3,Nmax]，M/L/inverse 为[T,Nmax,Nmax]，其余 vector/spatial/info/flags 全部持久预分配。

1. controls 先 `validate_controls` 构造拥有的候选，再和旧 bank 同锁提交。`upload_controls()` 不清 dirty；父确认所有 bank 上传成功后调用 `controls_uploaded()`。不能部分上传成功就启动物理步。
2. `stage_advance(h,src,dst,tick_src,tick_dst,accepted)` 仅写候选。父 sync/read_status 后、swap 前调用 `check_advance(status)`：读小 info、校验 status 一致性、预构造发布记录；该读回异常仍无 swap。`finish_advance(status)` 只能在 swap 后调用，纯 HOST 赋值、无设备 IO。
3. `validate_restore` 构造完整 restore stage；`stage_restore` 写同一 dst、复制未选树，重算候选 FK/能量。父所有 bank stage/sync 后 `check_restore()`，全部通过才共同 swap，最后 `commit_restore(stage)` 更新 held 和诊断有效性。旧类型 restore 也要保持树同一 active 方向。
4. `borrow_geometry(active)` 只在父同锁内借用已接受 state/FK，带 owner/generation/epoch/rows/specs。不持有跨 epoch 候选姿态、不关闭借来的 runtime。
5. geometry view 封装 `write_bounds(launch,kernels,*,count,rows,groups,offsets,points,geometry,ticks,bounds,stamp)`；factory 为 `make_tree_bounds_kernels()`，按 active 选择两个独立 handle。父无需复制内部长 kernel tuple。

`TreeBindingCompiled` 精确字段是 `group_index,zero_motion,points_m,geometry`。prepared tree rows[S]/groups[S]/offsets[S+1] 为 int32，points[V,3] 为原 group-local FP32 材料，geometry[S,4] 为向外半径、局部 reach、固定路径 reach、祖先深度。zero_motion 仅是静态 q=0 的 BVH 安装初始化；每次实际 query 不用它替代当前设备 pose。父保 key.body↔实际 group 的双向唯一性，以及同 world 动态 pair 的显式 exclusion。

所有资源归原父资源栈；storage.close 只使 helper 失效，不 free 借用 handle 或 close runtime。查询失败不发布新 result，物理接受态不回退。旧资源清理仍逐个尝试 callback；父另写两 bank partial-upload 和 restore late-check 注入用例，作者不重复改旧 hooks。

## 有限几何与精度边界

首域为 sphere、有限 capsule、hull（包括 box 的原全部角点），每组可多 shape；固定 root 使用原 static binding。直接构造 `ResidentTreeBinding(key,world,group_index,local_points_m,*,topology)` 是显式安装；`from_asset` 才证明原模型/shape/source/实际 group。folded 原 authored 材料经 site 的局部变换一次到动力学组，ID/revision/radius/内容均交叉核，绝不把 authored link 序号当组序号。

bounds 包全原材料点及真实 sphere/capsule 皮肤。按真实 accepted state 添加该路径实际 P 位移，并采用冻结预算：`64 eps32 max(1,|anchor|inf,L) + 128(d+1)eps32 L + 16(d+1)2^-20 L + 32(d+1)tiny32`。非负预算及端点向外 FP32，FTZ 项保底；读取 normal-math FK，不改物理积分。它是待真实 interpret/CUDA 留出验证的工程 envelope，不是所有实数 sin/cos 的形式认证。允许保守多候选，不能用现场误差反调 padding 或移动 miss fixture。

current accepted pose 宽相不等于 CCD、TOI、窄相材料接触或反力反馈。自碰撞/动态–动态、浮根/mimic/stops、任意N、完整多机器人设备场景和自动微分均不在首片。旧场景/材料默认行为不受这六个新增文件替代。

## 验证入口与尚未完成事项

完整固定矩阵见 [案例册](../development/p65-resident-tree-casebook.md) 和 [实施计划](../superpowers/plans/2026-11-07-resident-tree-broad-phase.md)：T01 2R、T02 R+双P、T03 固定工具/原材料、T04 斜轴与真实 COM 外载、T05 限幅延迟/宽帽相消、T06 N1/9/16与失败、T07 各有限形状/姿态/近触、T08 W1/4/32与40步及 warm 无snapshot/动态envelope上传/新分配。

父唯一串行执行两 scale 专项及共享事务专项、interpret/CUDA、必要相邻兼容、原 Newton CPU/CUDA 自有状态推进、静态与源码外 wheel。参考脚本 `tests/reference_newton/robot/resident_tree_reference.py` 不 import Phymeo，采用独立齐次变换导数/逐部件惯性，Newton保 fixed joints；它不是 HOST 当前 FK 上传路径。所有未来 pass、原失败、修复、准确 bytes/launch/allocation 和设备版本由父实际收据补充。

当前作者只做源码文本/Git，未执行 import、collection、pytest、数值、lint、format、compile、native 或 build。六 NEW 产品/报告与此前测试库不能单独作为“实际支持”结论；阶段完成需完整真实验收后另行标记。
