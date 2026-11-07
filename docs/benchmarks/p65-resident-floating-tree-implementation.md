# C9 普通浮根树设备驻留：实现交接

状态：**CODE_READY_TESTS_PENDING / UNEXECUTED**。本报告对应纯作者交接，未执行 import、collection、数值脚本、pytest、lint、format、编译、构建或设备程序。权威边界为 `docs/superpowers/plans/2026-11-08-resident-floating-tree.md`，包括父 `f1d22c6` 的大坐标完整检查点裁定。解释器、CUDA、独立 time-jet/Newton、保存审计及 wheel 验收均由父串行执行，本报告不预写通过结果或性能收益。

父已报告作业 434 的有效首 RED：现有 HOST 浮根 P03 的真实 7×7 质量矩阵和内部 3 N 导致根加速度 −1.5、相对关节加速度 +4.5 的前置成立，随后准确停在缺少公开 `ph.FloatingRobotTree`。此处仅记录父提供的结果，作者未复跑。

## 精确新文件与所有权

| 新文件 | 内容 |
| --- | --- |
| `src/phymeo/backends/_resident_floating_tree.py` | 公开声明、输入、完整检查点、运动学/诊断读回及静态编译 |
| `src/phymeo/backends/_resident_floating_storage.py` | 同 runtime 的根/关节/广义载荷/FK/twist/能量双 bank、阶段调用与显式读回 |
| `src/phymeo/backends/_tyx_resident_floating_kernels.py` | 实际浮根 FK、完整 M/b/J、有限 PD、根与子链一次漂移、接受/恢复核 |
| `src/phymeo/backends/_resident_floating_binding.py` | 原材料到真实 body 的显式绑定与普通/固定工具资产来源验证 |
| `src/phymeo/backends/_tyx_resident_floating_bounds.py` | 从当前接受态设备 FK 生成有限形状外扩 AABB |
| 本报告 | 接口、事务、来源与待验边界 |

父拥有所有旧文件的 batch、布局、公开导出、碰撞分派与接触排除接缝。公开类型由父导出到 `phymeo` 和原 `tyx_world_batch` 模块，原空 `backends/__init__.py` 保持不变。原 RD 作者负责独立专项及 time-jet 真值；本交接不修改其测试或参考脚本。

## 公开数据与单位

`FloatingRobotTree(model, state, *, kp, kd, effort_limit, delay_steps=0, control_max_iterations=512)` 接受真实 `FloatingRobotModel/FloatingRobotState`。`spec.inputs(drive, external=None)` 返回拥有输入的 `FloatingTreeInputs`，其中 `drive[n,3]` 为 target/rate/feedforward；`external` 是原 `FloatingRobotInputs`，包含 n 个即时关节 effort 和 B 个世界 COM force/torque。默认外载为零；所有数组深拥有，原模型、状态和调用者输入后续修改不改变已编译声明。

计数独立：n=0..16 个子关节，B=n+1 个真实物理体，D=n+6 个速度坐标，Q=n+7 个配置坐标，电机数仅 n。kind 9/10 分别表示延迟 0/1；没有根部虚拟电机。根仅刚体是 n=0 的真实子类，仍保留 B=1、D=6、Q=7 和空 `(0,3)` 驱动。

配置依次为 authored 根世界原点 xyz、body→world XYZW、子关节 q。速度依次为根刚体自身 COM 世界线速度、世界角速度、子关节速度；不是整棵树总 COM，也不是根原点速度。子 R/P 的配置/速度单位分别为 rad/m 与 rad/s/m/s。力矩均绕各真实 COM；根惯量主轴 frame 与根 body frame 分开保存，原始 FP64 model token 不由舍入后的设备表替代。

`FloatingTreeCheckpoint` 保存 world/topology/tick、完整 Q/D、实际 n 电机 effort、完整 D 广义 applied、延迟队列与四块 held control。四块为 drive[n,3]、effort[n]、force[B,3]、torque[B,3]；队列只延迟 drive 三列，不延迟即时外载。`generalized_applied` 含即时外载投影和实际电机载荷、不减 bias。

`FloatingTreeKinematics` 返回实际接受态配置/速度、B 个 link 原点/朝向/COM、COM 线速度和世界角速度，并明确四类 count。世界位置以实际设备 FP32 局部值扩为 FP64 后加原 FP64 anchor；q/v/R 保留实际存储值含义。`summary()` 仅给所有 B 个真实体的 `kinetic_energy_j`。

`FloatingTreeDiagnostic` 记录最近成功 attempt 的入态、入/出 tick、h、完整 B×3×D J、D×D M、D 维 b/外载/加速度、n 维实际 drive，以及迭代数、PD 和质量方程残差。它不是历史状态库：初始化、restore、最近数值拒步返回 None；设备异常不能把半成 scratch 当作新的成功诊断。诊断 entry_position 是世界观测，原完整入步 checkpoint 另存可逆 local。

## 真正耦合的设备步骤

1. 从接受根姿态、COM/world twist 和子关节 q/v 建立所有真实 body 的 FK 与完整 Jacobian。根线速度列为世界平移；根角速度列绕根 COM；所有子链祖先列保留，没有拆成独立单铰链。
2. 对每体旋转主惯量，累计全 D 质量矩阵。以实际零广义加速度递推向心、Coriolis、角偏置及 `I*alpha + omega×I*omega`，重力只在 b 中一次出现。全部 COM 外力/力矩通过同次 J 投影，即时关节 effort 仅加在其真实子关节行。
3. 对全 M 做 Cholesky/逆质量求解。n 个电机的有限投影坐标迭代使用全逆矩阵的 joint/joint 块，等价于根六维自由响应消元后的 Schur mobility；根不参与电机 cap 或目标循环。有限预算保持 1..4096，默认 512。完整 original clipped PD 残差门为 `5e-5*max(1,abs(actual),abs(expected))`；全 D 质量方程门为原 `2e-5`。非有限、非 SPD 或未收敛明确拒绝，不作 HOST rescue。
4. 用同次最终速度只漂移一次。根姿态左乘世界 `exp(h*omega)`，归一化后显式以新旧 quaternion 内积选择同半球。根原点按 COM 偏心校正 `origin+ = origin + h*vCOM+ - (R+ - R)*root_com`；子 q 同时更新。
5. 生成候选 FK/twist/全体动能，统一检查并逐 world 接受或整行保留。父 batch 最后只进行一次 active swap/epoch 发布。

角域保留原裁定：root local xyz、子 q 的绝对值不超过 4096；输入/状态需 normal FP32 或零，初态使用原 FP64 单位四元数检查，存储/恢复 quaternion 范数平方误差不超过 1e-5。每步最终旋转向量范数不得超过 π；超域拒绝，不改变 h 或将状态裁剪回域。

父在执行前的文本审阅指出 π 边界 FP32 cosine 可略负，作者在 `882433d` 加入 normalize 后同半球反号。它不改变物理旋转、原角域或输入；尚未执行，RD 的相邻留出由父安排。

## 大世界锚点与可逆恢复

允许 anchor=(2**60,0,0)、root local=(1/1024,0,0) 时，FP64 的 world=anchor+local 仍可能等于 anchor。此为明确表示反例，不能以 world Q 相减重建实际 root bank，也不以缩小 anchor 域掩盖。

检查点末项 `local_origin_m` 是保存实际 root bank FP32 三项的不可变 tuple。restore 先验证其 normal FP32/零和 ±4096，再要求 `position[:3]` 精确等于原 FP64 anchor 加该 local 的 FP64 展开，最后从 local 恢复。世界 Q 单字段是含加法舍入的观测；完整 checkpoint 才承担可逆状态承诺。合法编辑检查点也必须同步这两种表示，二者不一致直接拒绝。

## 父共享层的精确调用合同

`compile_floating_tree_world(spec)` 返回 `FloatingTreeWorldSpec`，含原 FP64 `model_token/topology/anchor`、int32 parents/kinds 和分开拥有的 FP32 静态物理表。初始 root[13]、joint[max(1,n),7]、drive[n,3]、loads[B,6]、effort[n]、generalized[D] 明确分离。

`FloatingTreeDeviceStorage(batch, specs_by_world)` 使用原 batch `_slots`；`meta[T,4]` 是 global slot/n/delay/budget。父不需展开长 kernel tuple：

- 控制：`validate_controls(mapping)` → stage，全部 storage 先验通过后 `commit_controls(stage)`；所有 dirty bank 的 `upload_controls()` 均成功后才 `controls_uploaded()` 清标记。
- 步进：`stage_advance(h,src,dst,tick_src,tick_dst,accepted)`；父 sync/read_status 后、swap 前 `check_advance(status)` 读有界 info；swap 后 `finish_advance(status)` 只作 HOST 结果发布。
- 恢复：`validate_restore(mapping)` → stage；`stage_restore(stage,src,dst,tick_src,tick_dst)` 后全部 storage `check_restore()` 成功，才共同 swap/`commit_restore(stage)`。stage 不修改 active 或 epoch。
- 公开读回：`snapshot(keys,active,ticks)`、`kinematics(keys,active,ticks)`、`diagnostics(keys)`、`summary(keys,active)`，均明确读回并计数。旧标量 reader 不隐式解释新布局。
- 同锁借用：`borrow_geometry(active)` 返回 owner/topology_generation/state_epoch/active、rows/specs 和实际设备 handles。`view.write_bounds(launch,kernels,*,count,rows,groups,offsets,points,geometry,ticks,bounds,stamp)` 封装全部根/关节/FK/anchor 参数。

存储保留 root、joint、generalized applied、FK、COM/world twist、energy 两套 bank。每次拒步复制所有对应 source 行和 tick；数值拒绝不得只退 q 而发布新 history/drive/energy。上传、sync、状态读回或晚检查异常没有新 active/epoch。临时 `_pending_info` 仅在成功发布后转为最新诊断有效性。

所有实际缓冲由原 `batch._buffer` 与原资源栈持有；storage.close 只标记已关闭，不关借用 runtime、不重复释放。父需将 floating 纳入所有统一 storage 循环和资源关闭，并在未绑定浮根的 contact 混合步明确保存 `_held_drive/_held_effort/_held_loads`，不能按 affine `.physical` 结构误判。

计数使用 `floating_{topology,initial,control,restore,state,fk,summary,diagnostic,status}_{uploaded_bytes,downloaded_bytes,host_read_bytes}`；kernel launches 分 dynamics/fk/energy/restore/commit。显式 state/fk/summary/diagnostic/status reads 单列。解释模式真实 HOST 读字节仍记录，CUDA 模式实际下载单列。父宽相另计 `floating_bounds_kernel_launches`。暖态不调用这些状态读回、不上传动态 envelope、不分配新的设备缓冲；必要有界状态 mask/info 读回不虚报为零。

## 原材料几何与接受态宽相

`ResidentFloatingTreeBinding(key,world,group_index,local_points_m,*,topology)` 是显式 body-local 安装声明，允许真实 body 0..16，并在编译时按 spec.b 再限域。它不继承旧固定树 group<16 的类型。`from_asset(asset,source_key,*,key,world)` 额外核完整注册 shape、原 IDs/skin/revision/model token；普通根 owner −1 映为 body0，子 i 映为 body i+1。固定工具资产按原 authored site 固定变换转换一次，保原 shape 材料及真实 owner。

`compile_floating_tree_binding(shape,binding,spec)` 返回 `group_index/zero_motion/points_m/geometry[4]`；四项为 outward skin、local reach、path reach、depth。zero_motion 仅注册时的原模型 identity-root/all-child-zero 静态声明，不代替当前 HOST FK。工厂名为 `make_floating_bounds_kernels()`。

每次查询由实际接受态设备 FK 变换原有限 sphere/capsule/hull 点，含真实根平移/朝向、子链滑移及多形状 body 映射。按原工程外扩包括局部尺度、祖先长度、P 行程、root local 位移、旋转层数及 anchor FP32 舍入。允许保守多候选；不声称所有实数形式区间证明。current-pose 候选不是 CCD，也不是窄相接触或载荷响应。

父旧碰撞路由需保持同 world owner/key/body 双向映射、接受 epoch 与双 buffer、未执行 candidate 不可借用、空场景与关闭行为。新绑定默认沿旧显式动态 pair 排除；允许的同模型非相邻 body 动态候选不自动启用接触求解。

## 待验与能力边界

RD 独立矩阵应覆盖自由落体、偏心根惯量/COM、根与关节反冲、分支 R/P、非主轴姿态与排列、n=0/9/16、PD delay/cap/预算拒绝、完整 restore/held/失败与 W1/W4/W32 原材料几何。独立 time-jet 参考不从此核生成真值；全部执行与数据绑定由父负责，作者不删减冻结案例或门。

本片只实现 ordinary floating tree 的机制/持续状态/current-pose 宽相。浮根 affine、闭环、多个独立参与者同 world、浮根接触反馈、CCD、传感器新通道、训练/可微及性能收益均不在本片接受声明内。`prepare_contacts` 对绑定浮根明确拒绝；混合 batch 中没有绑定接触的浮根 world 应仍按自身真实动力学推进。完成父专项、独立参照、保存审计和新 wheel 前，不将本源码交接标为物理验收通过。
