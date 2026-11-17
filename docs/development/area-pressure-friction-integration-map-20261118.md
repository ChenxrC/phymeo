# 面积摩擦从独立参考进入生产路径的接缝图

生产源码按父 `codex/exp` 的 `c56b92c072e56dc3da9c31729e7faedb0e10ec5f` 核读；至 `5784012499e57d9dd86404def1c336f665532c65`，`src/phymeo` 无增量。
I3r 已在 `57840124` 获独立参考资格：首次 5 CUDA＋1 HOST mock 共 6 passed / 4.22s / RAW_EXIT=0，见[原日志](../../artifacts/area-pressure-tyx-coupled-friction-20261118-01/05-first-full.stdout.log)与[完整摘要](../../artifacts/area-pressure-tyx-coupled-friction-20261118-01/validation-summary.json)。
六个 W1、反序 W6 同字节、W32 分片、非对角主元交换及故障边界已验；制造例角速度误差约 3.76225e-5、质量残差约 2.27663e-6 满足固定 FP32 门，但没有通过 HOST 1e-10 严格门。
这是独立共享 DOF 方程/FP32 workspace 的资格，没有 q 积分、自动面积场景或生产 type7 摩擦接入；不能据此宣称生产场景已支持摩擦。
以下“当前”均指该源码基准；“建议”是待另行设计、冻结并验证的范围，不授予执行许可，不设置新物理参数或通过门。

## 源码接缝表

| 接缝 | 当前源码锚点（仓库相对路径:行） | 可复用与必须补齐的部分 |
| --- | --- | --- |
| 当前几何/相位 | `src/phymeo/backends/mujoco_area_pressure_scene.py:70,151,188,250` | 已接受 device q/v 做 FK，变换 owned body-local 资产，重新裁剪，再绑定当前 tick/time/configuration。复用此链；不能缓存跨步接触点或只传点而省略资产/压力/法向/身份。 |
| 原生状态/动力学 | `src/phymeo/backends/mujoco_tyx/_constraint_device.py:96`；`src/phymeo/backends/mujoco_tyx/_tree_kernels.py:348`；`src/phymeo/backends/mujoco_tyx/_floating_kernels.py:392` | 已生成完整 M、COM Jacobian、bias/passive/applied。M 含原 body 耦合及 armature；不能改用各点独立有效质量或仅 body 标量质量。 |
| 控制与外载 | `src/phymeo/backends/_mujoco_tyx_runtime.py:783,972`；`src/phymeo/backends/mujoco_tyx/_floating_kernels.py:492` | 复用原 ctrl、qfrc_applied、COM 世界 wrench 投影与 actuator；bias 已含重力/运动偏置，passive 已含阻尼和已支持的弹簧。参考器的 f 不能直接等于用户外载。 |
| M 与 M+hD | `src/phymeo/backends/mujoco_tyx/_area_pressure_kernels.py:29`；`src/phymeo/backends/mujoco_tyx/_constraint_device.py:215,247` | 面积路径把 A=M+hD 交给约束响应；当前 smooth_acceleration 是 A 的加速度响应，不能把它当最终速度或物理 M 的 qacc。 |
| 两侧完整点 J | `src/phymeo/backends/mujoco_tyx/_area_pressure_kernels.py:40`；`src/phymeo/backends/mujoco_area_pressure.py:165` | 已从两侧 COM 的线/角 Jacobian 及点力臂计算 JA/JB，再只保留 nᵀ(JA−JB)。摩擦需在同一设备阶段保留/复用完整 3×nv 的 G；现 normal_jacobian 无法恢复切向信息。 |
| 材料与样本 | `src/phymeo/backends/mujoco_area_pressure.py:87,121` | 当前 packet 每点仅 point/normal/k/phi，其中 k=dA·gradient、phi=−pressure/gradient；dA 原 FP64 位另存。需显式 owned 的 mu/vs 和法则身份；不能从面积或原生 geom friction 猜材料。 |
| type7/T4 | `src/phymeo/backends/mujoco_tyx/_area_pressure_kernels.py:71`；`src/phymeo/backends/mujoco_tyx/_constraint_solver.py:1,138,191` | type7 是标量单边法向行，原 T4 是有界 box/active-face 求解。不能追加 Ft 后继续把它冒充该标量行的解，也不能对新的非对称 Newton 矩阵复用此 SPD 分解。 |
| 原生准入 | `src/phymeo/backends/mujoco_area_pressure.py:27,48,56`；`src/phymeo/backends/mujoco_tyx/_constraint_model.py:15,85` | 当前按精确 body pair 排除其 native contacts；activation/BALL 不准入。D>0 且存在剩余 native possible pair/limit/dry friction 即拒；D=0 的现纯法向路径可共解。新律不能悄悄扩大这些条件。 |
| q/时钟积分 | `src/phymeo/backends/mujoco_tyx/_constraint_device.py:296`；`src/phymeo/backends/mujoco_tyx/_tree_kernels.py:496`；`src/phymeo/backends/mujoco_tyx/_floating_kernels.py:540` | 原积分器分别输出物理 qacc 与 integration_acc，按 native joint 地址积分 scalar/FREE/BALL 及补偿时钟。复用坐标约定，不能在 HOST 写 q，或把 FREE 的局部角速度当世界角速度。 |
| 单叶/全桶发布 | `src/phymeo/backends/_mujoco_tyx_runtime.py:962,1183,1199`；`src/phymeo/backends/mujoco_heterogeneous.py:626,659,716` | 所有 packet 准入、所有候选/receipt 验证后才统一 active bank/revision/tick。新 Newton、记录和观察验证必须进入同一 prepare；末 world 失败不得先提交其他 world。 |
| 持久记录/观察 | `src/phymeo/backends/mujoco_tyx/_area_entry_kernels.py:1`；`src/phymeo/backends/mujoco_area_pressure_device.py:10,27`；`src/phymeo/backends/area_pressure_readback.py:18,295` | 当前 v1 强制 F_A=Fn·n，15 列数据及 receipt 双向校验均依赖此式。新总力/切向力必须有显式版本与验证；不能直接覆写 F_A 后继续声称 v1 相容。 |
| 消费与恢复 | `src/phymeo/control/body_contact_wrench.py:254`；`src/phymeo/training/area_pressure_success.py:147`；`src/phymeo/backends/mujoco_area_pressure_state.py:26,50` | body wrench 的反号与参考点平移可复用；通用 reader 的方向验证、成功 reader 的 Fn/垂直总力分工、raw/typed rows 与动力学一致性都须扩展。旧历史不能补造切向力。 |
| 资产/胶囊绑定 | `src/phymeo/backends/area_pressure_assets.py:32,164,392`；`src/phymeo/backends/mujoco_checkpoint.py:235`；`src/phymeo/backends/mujoco_entry_checkpoint.py:21` | 当前 bundle law 固定 `frozen-surface-elastic-backward-euler-v1` 且 `friction=none`。新参数须与原 source/model 一起参与分桶、资产 hash、law/schema、原始银行布局与 checkpoint 配对；不改写原 source SHA，旧胶囊明确保持旧律或拒绝。 |
| 资源所有者 | `src/phymeo/backends/mujoco_tyx/_allocation_manifest.py:11,34`；`src/phymeo/backends/mujoco_tyx/_resources.py:95,119,159` | 持久 Newton scratch/输出加入现 manifest、预算和 ledger。复用 runtime/分配器/UNKNOWN_RELEASE 次序；不能把独立实验 workspace 变成每步另建的第二执行栈。 |

## 最容易接错的动力学等式

记当前入步速度为 v0，D 为原非负对角关节阻尼，b=`applied + passive − bias`。
源码的 `passive` 已包含 `−D v0`；令其余非阻尼广义载荷 f0=b+Dv0，则：

`A=M+hD; vfree=v0+h A^-1 b = A^-1(Mv0+h f0)`。

因此 I3r 风格输入 f 应对应 f0；直接传 b 会重复阻尼，直接传 applied 会漏 bias、重力和其他 passive。
生产实现更适合直接消费同阶段的 M、D、b/vfree，不绕 HOST receipt 下载再上传。
输入/输出完整力学核验必须同时记录 f0、b、实际 applied/passive/bias，而非只保存一个含糊的 external_force。

每个世界点 p 的两侧列 Jacobian 为 `Jp[:,d]=Jcom_linear[:,d]+Jcom_angular[:,d]×(p−com)`。
使用同入步 p/n/COM/JA/JB，G=JA−JB；摩擦方向由 `Gv` 的切向分量决定，两侧可动时不能假设 JB=0。
最终 Fn 与 Ft 必须使用同一个已解 v；保留 dFn/dv 的交叉项，因此完整 Newton Jacobian 一般非对称。
本候选是 final-normal smooth 点摩擦；它不提供精确静摩擦 stick 分支，也不添加独立滚动/扭转 couple。

当前末段积分器用 `total_applied=applied+Qcontact` 再求速度。
若保留此入口，必须验证重新得到的实际 v 与摩擦解一致、同一 Q 只加入一次；不能先写 v 再重复施加冲量。
若改为直接积分已解 v，应在共享 native 积分层抽取可复用路径，不能复制一套 FREE/quaternion 更新逻辑。
物理 qacc 应仍对应 `M^-1(b+Qcontact)`，integration_acc 对应 `(vnext−v0)/h`，两者不可混标。

## 最小生产接入边界（建议，尚未准入）

第一批仅显式 opt-in 新律、现有完整压力体/有限刚面、支持的无 activation/BALL native 构型。
仅 area constraints；同一物理世界有任何剩余 native possible contact、limit 或 dry-friction row 就先明确拒绝，D=0 也不例外。
旧 `friction=none` 与 native-only 世界继续原分支；异构批次可混合不同世界，但必须保持原全桶提交。
先明确生产样本/DOF 容量：现 I0 每个 triangle 固定 3 个积分点，常见有限面资产的多个 triangle 可超过 I3r 的 8 samples；这是真实接入的首要容量障碍，不能裁面、删点或重归一权重绕过。
I3r 的 8 samples/6 DOF 不能截断现 cap、不能隐式替换原 T4 的 512-row 上限；若要扩大，应另做独立容量设计与验收。
若首批沿参考的小容量，应作为新律公开准入限制，超限在任何物理提交前拒绝；不得回退另一摩擦律。

将通过验收的可复用数值内核提升到共享 backend 层；生产代码不应 import `experiments` 或每步调用其 HOST 驱动器。
HOST 当前几何仍可复用；新增 scratch 及 k/phi/mu/vs 的设备准备归原 leaf 所有，带明确法则/参数绑定。
原正交投影、最终法向 capacity、FP32 原输出与独立残差证书必须分别保留，不能 HOST polish 后覆盖实际解。
仅收到若干点不足以接通：还缺当前完整 J/M/载荷、law 身份、native 排除策略、积分一致性和事务生命周期。

## 最小真实案例与反例矩阵（参数/窗口/门须另行冻结）

| 案例 | 必须开放的实际运动/载荷 | 关键区分与反例 |
| --- | --- | --- |
| 平移滑动 | 平面内与法向均由实际 DOF 运动，有限支撑持续按当前 q 生成 | 验证同一步 Fn、Ft、GᵀF、非正切向功率及平滑微滑；反例是仅事后施加 `−mu·Fn` 或先独立求法向。 |
| 卸载分离 | 同资产真实向外运动，接触区由有载到空，不硬写坐标 | 最终 Fn=0 时 Ft=0；保留入步仍有弹性压力而最终卸载的区间，揭示 Lagged 载荷/旧帧摩擦残留。 |
| 偏心旋转 | 已支持的 hinge/平移组合，非零接触力臂，完整真实 J/M | 分别验证 p×F、body-origin/COM 参考点、虚功和实际角速度；反例是只用线性 J 或把力矩归到错误参考点。 |
| 双方动体 | 两个独立可动 body，法向和切向共享质量响应 | 两侧使用各自当前 JA/JB，接触反力/共同原点反矩闭合；在相应无外载方向验证系统动量，揭示 JB=0、逐 pair 独立解及重复载荷。 |
| 旧路径兼容 | native-only、显式 friction=none，以及它们与新律的不同 world 同批 | 原配置不变，核原银行/receipt 语义；新律与同 world native row 的未支持组合须构造拒绝，而非过滤掉第三方接触。 |

这些测试必须由真实 scene 自动几何→设备求解→native 积分→公共持久观察产生，不能拿独立参考的 supplied points 当连续物理证据。
窗口以接触启闭和区分反例所需为限；不把现有已验长支撑窗口重新跑一遍当摩擦验收。
每个案例保存入步/出步 q/v、实际载荷分解、完整面/目录、mu/vs、法切力/矩、真实 J、残差、原设备记录与公开消费结果。
同批补末几何/pack/Newton/下载/receipt 校验失败、选择 reset、成对 checkpoint/restore 与同模式继续；step 候选失败保持旧银行和历史观察，不伪回滚此前独立接受的 controls 更新。
`mu=0` 检查新律回到纯法向方程；旧 friction=none 的精确兼容与新算法的数值等价须分开报告。

## 留到后续的一般混合约束

native pyramidal 摩擦、soft limits、dry friction 与新面积摩擦共用自由度，需要同一最终速度下的联合方程/互补条件与独立证书。
顺序调用“新面积 Newton→旧 T4”或反过来都会改变另一方已使用的 v/Fn/Ft，不能当通用解。
不得移除现 D/native guard，或将 final-normal smooth 偷换为 Lagged 来获得原 SPD 结构。
一般 native/area 共解、activation/BALL 扩域、更多 DOF/样本、全设备几何、精确静摩擦及滚动/扭转律应分别立项。
新观察版本需同步覆盖 checkpoint raw/typed/dynamics 一致性、目录摘要、Torch/任务 reader 和 body wrench；旧 `friction=none` 胶囊不自动升级。
本图谱仅给出源码事实与候选接入边界；I3r 已获独立设备参考资格，仍不替代上述真实场景与生命周期验收。

## 2026-11-18后续状态入口

以上表格保留原c56b92c0/57840124基准，不倒灌后续结果。现已新增[I3u 128点backend](area-pressure-backend-friction-report-20261118.md)、[I3v真实历史入步适配](area-pressure-entry-friction-report-20261118.md)、[I3w resident设备装配](area-pressure-resident-friction-pack-report-20261118.md)及源码外安装包验证。生产Scene摩擦仍待接通；下一工作以[live交接](area-friction-live-integration-handoff-20261118.md)和[B1–B3计划](../superpowers/plans/2026-11-18-area-friction-production-integration.md)为准，不按本表旧“8点”容量重复开发。
