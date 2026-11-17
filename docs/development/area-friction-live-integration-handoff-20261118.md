# 面积摩擦设备装配与 live 集成切口（2026-11-18）

状态：纯源码核查与手推；未运行项目、测试或数值探针。不是已实现的 F3。
权威父树本次读回 HEAD：`73acc4e8270c349a838dd6ad1cf7e0e038be28dc`；I3u/I3v 的既有结果归父报告，不在此重验。
下列源码路径均相对仓库根目录；行号对应本次父树。旧 type7、资产 v1、原 normal-only 默认路径保持独立。

## 1. 最小有增量价值的下一步

先新增独立设备 pack：当前同入步 resident 缓冲 → I3u 七个输入缓冲 → 原 `solve_batch`，末尾才读回诊断。
它验证真实生产布局、双方点 J、阻尼与外力装配；不调用 I3v 历史 receipt 桥，不再次下载再上传 M/J。
这一节点不写 q/v、不施力、不积分、不发布物理状态；只完成设备装配接缝，不能称 live friction 已接入。
若测试只上传保存的原设备字节，应明确是实际布局的历史诊断；不能声称测得生产整步省时。
真正 F3 必须再完成第 4–6 节的积分、law/观察版本和原子发布，不能用 pack 的通过替代。

## 2. 同入步源与精确布局

切口：`src/phymeo/backends/mujoco_tyx/_constraint_device.py:97` 的 `mechanics[old]` 后，`:135` 的 actuator 更新后，旧 contacts/assemble/solve 前。
`old=device.active`；该阶段 q/v、M、COM/J、bias/passive/applied 同属当前入步，尚未发布候选端点。
设 W≤32、nv≤6、P=max(8,nv)、B≤128；Q=max(8,nq)，C=max(1,device.capacities[1]) 为面积存储容量。

| resident 字段 | 实际 shape / 含义 |
| --- | --- |
| `mass` | `[W,P,P]`，物理 M，含原模型 armature；不使用已加 hD 的 response_mass |
| `jacobian` | `[W,B,6,P]`；0:3 世界 COM 线速度 J，3:6 世界角速度 J |
| `motion` | `[W,B,7,3]`；slot 0 body origin，slot 5 世界 COM |
| `v[old]` | `[W,P]`，原 native DOF 顺序 |
| `bias/passive/applied` | 各 `[W,P]`；applied 已含实际驱动、qfrc 与 body force 投影 |
| 静态 `vectors` | `[9,Q]`；第 1 行为已 RN32 的真实 dof_damping |
| `area_input` | `[W,C,8]`，point3、normal3、k、phi；已按旧入口 RN32 |
| `area_pair_index/area_sizes` | `[W,C]` / `[W]`；全样本序号与实际数量 |
| `area_body_ids` | `[pair_count,2]`；精确 A/B body ID，不按 welded component 扩张 |

布局来源：`mujoco_tyx/_allocation_manifest.py:147,154,265`；点 J 原实现：`mujoco_tyx/_area_pressure_kernels.py:49`。
每列 `Jpoint = Jcom_linear + Jangular_column × (point - COM)`；两侧分别保留，不能只传法向标量 J。
静态 pair-law 表另给 `[pair_count,2] = (mu,vs)`，必须由有序身份声明绑定，不能靠 body ID 猜材料。

## 3. 输出 I3u 输入及检查

复用 `backends/_tyx_area_friction_kernels.py:7 solve_batch`；不要在生产阶段调用 `TyxAreaFrictionWorkspace.solve_many`。
后者在 `backends/tyx_area_friction.py:319,375,404` 拥有独立 runtime、HOST pack/upload/download，不是 resident 接口。
七输出：`sizes[W,4]=(nv,ns,24,12)`，`mass[W,6,6]`，`inputs[W,3,6]`，`steps[W]`，`samples[W,128,10]`，`ja/jb[W,128,3,6]`。
`inputs[:,0]=D; inputs[:,1]=v0; inputs[:,2]=applied+passive-bias+D*v0`；不得纳入旧 constraint force。
`samples=(point3,n3,k,phi,mu,vs)`；h 用生产实际 FP32 h；normal 与 point 保持原上传 FP32，不偷偷再归一化。
这是 actual-packed-point 接缝，与 I3v 原 FP64 source-quadrature 点的装配不是逐字节同一问题。
入口先核 shape、dtype、所有资源界；设备先核源 status、ns、pair ID、A/B ID，再间接寻址；全有效字段 finite、k>0、mu≥0、vs>0。
不得靠 I3u 求解器替代全部准入：其原公开 wrapper 承担的 D 非负、M 对称、法向与输入完整性检查需在新入口有等效归属。
I3u 内核先清自己的 meta；pack 必须保留独立 upstream status，失败世界不得被清成成功，也不得消费未初始化输出。
所有 padding 显式清零；零样本仍是完整自由运动问题，不把 ns=0 当失败。禁止 count/ns 静默截断。
I3u 的 14 缓冲 W32 合计 1,770,624 B，其中 7 输入 761,216 B；另加静态 law、pack status 和未来 accepted banks，不能冒称此即生产总内存。
8 个 vector scratch 槽与 sample 上限无关；trace 固定 290（initial+最多288 trials+final）。

## 4. 真正 F3 的唯一积分决策

旧 `mujoco_tyx/_floating_kernels.py:575–616` 先以 M 求物理 qacc，再以 M+hD 求增量，随后按 native joint 地址积分一次。
I3u 方程与该离散化相容：`(M+hD)(v*-v0)=h(applied+passive-bias+Q(v*))`。
最小复用候选是将实际总 Q 写入 `mixed_constraint`，将 `applied+Q` 交旧 solve 一次；不能把已补 Dv0 的 solver f 再交旧 solve。
但旧 solve 会重新计算 FP32 vnext；必须在实际发布 vnext 上重核最终 Fn/Ft、残差、摩擦功与上界，不能只核 I3u 的 v*。
若此复核失败，停止局部接线：应提取共享 native q 积分入口直接消费 v*，保留独立物理 qacc 与 integration_acc，而非复制 FREE 四元数积分。
不得运行旧法向约束 solve 后再叠摩擦 kick；也不得对已经正常推进的 I3v receipt 再施一次解。
这是结构性验收门，不预先承诺两个 FP32 路径相等。

## 5. 显式 law、准入与观察版本

首期建议 area-only：每桶 W≤32、nv1..6、样本≤128；原三点 quadrature 实际最多126点，空面合法。
保留 scalar/FREE、原 actuation/passive damping；拒 BALL/activation、equality/tendon、limit/dry、任何未排除的 native pair；不改旧 native 混合求解语义。
新增 opt-in law `final_normal_smooth_v1` 及有序 pair mu/vs；同模型不同 law/参数不能进入同一 bucket。
修改入口为 `mujoco_heterogeneous.py:49 ModelSpec/:132 _signature/:146 _model_preflight` 与 leaf 静态 plan/manifest；默认不新增旧 binding 字段。
`mujoco_checkpoint.py:235 _model_binding` / `mujoco_area_pressure_state.py:26 area_binding` 必须绑定 law、参数、布局版本与 precision。
资产 bundle v1 在 `area_pressure_assets.py:165,393` 明示 friction=none；不能静默把它当新摩擦材料声明。需独立新 law envelope/版本，旧资产字节含义不变。
当前 `area_data[W,C,15]` 与原 receipt 验证假定 forceA=n*Fn；`_entry_storage.py:198` 从 type7 efc 行构造它，不能直接冒用来表达 Ft。
新 law-aware accepted record 必须保存 Fn、Ft、实际相对速度、mu/vs、full force/world moment、sample/source/pair 身份、入出步 clock/tick；dA 仍只作测度，不能再乘一次力。
旧七 native entry 字段不伪造 area geom/contact；独立 law 数值 solve 不冒原 type7 KKT 标量行。新 checkpoint 必须验证新 force/actual-v 一致性。
若尚未做新观察/恢复协议，显式拒绝该 law 的 copy/reader/checkpoint/restore/clone；不能让旧 normal-only consumer 静默少算切向载荷。

## 6. 原子发布与资源边界

生产 allocation manifest 一次预留新 scratch/status/双 accepted bank；复用 owner runtime/ledger，禁止求解过程中临时创建另一个 runtime。
`_mujoco_tyx_runtime.py:962 _prepare_step_locked` 负责 advance→inactive entry stage→download/验证→draft；新 force 候选沿同一 inactive bank，失败不翻 active。
`:1199 _commit_step_draft_locked` 仅在全部验证后同时翻物理/entry bank，交换 receipt/tick/revision；不能在多桶 swap 间加入可失败回调。
`mujoco_heterogeneous.py:661 _step_locked` 先准备所有桶，`:685 _before_commit` 后才于 `:716` 共同提交；晚桶失败必须保全部旧 state/force/history。
`mujoco_heterogeneous_area_scene.py:161–165` 在 batch 成功后才发布 last_geometry；沿用该边界。attempt scratch/计数可变化，accepted frame 不可变化。
reset、restore、clone 必须复制/清理新 law banks，并重绑 owner/catalog；新旧 capsule 不可互相补造摩擦历史。

## 7. 五个可否证集成测试（未来执行，非本次结果）

1. 非零 D/v0/actuation/外力的同入步：独立 native mj_forward M/J/bias 与手算自由速度，核仅一次 +Dv0、mu=0 极限及实际最终速度；错误双阻尼公式须明显失败。
2. 旋转 FREE 偏置 COM 与另一个两侧各3DOF案例：独立 mj_jac 点 J、双侧作用反作用/虚功、两方向滑速；q 只积分一次，world-origin 与 COM 矩分别检查。
3. 正入步压力但最终分离：Fn_final=0 必须 Ft=0；另低滑速支撑核最终 normal-dependent smooth 力与非正切向功，不能用 Lagged 或独立后 kick 通过。
4. >8 独立点及126点上界、空面、W1/批量/反序：实际 force/Q 同源对照；129点、错误 body/pair、同 MJB 不同 mu 与晚字段非法在发布前拒绝，不能截断。
5. 最后世界数值失败/最后桶提交前故障：旧 q/v/clock、accepted forces、workspace frame、receipt、scene geometry 均保留；随后 retry；新 law checkpoint roundtrip/选择 reset 或明确生命周期拒绝不可混淆。

停止条件：上游状态不能可靠传递、actual-v 力律不成立、没有 law-aware 观察/恢复边界、资源未进统一 ledger 任一存在时，不宣称生产 F3 完成。
