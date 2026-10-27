# P6 tyx 新动力学全链路设计

状态：详细实施规格已完成一次独立 builder check，未发现重要缺项；承接用户已批准的集成大纲 P6 与“P6→P6.5、28日08:00”授权。
主机工程基线为 P3–P5 专用分支；P6 应在其完成验收后另开 codex/p6-tyx-chain。
本文件的计算分工、容差和实现路径是实施者技术选择，不写成用户指定算法。

## 目标与边界

将已验证固定基座机器人、布料、受限 StVK 四面体与节点/连杆球联合求解迁移到固定版本
tyx 99f6df7c0fd8fdeef77f97df83914da27b64b7ee。统一 SceneStepper 继续拥有子步、候选与整帧事务。
机器人 FK/J/M/bias/广义载荷/质量求解、材料内力、积分、布料投影、点响应与接触更新必须
实际由 tyx kernel 执行。三个模式 interpret、cpu_cpp、cuda 显式选择，错误不得回退到 NumPy。
完整 S2/S3/S4 各 200 帧、两子步，实际 CUDA 执行并独立审计、重复、同参数 FP64 对照。
不在此阶段增加摩擦、面碰撞、双机器人或优化；它们属于已授权的随后 P6.5。
截止为 2026-10-28 00:00 UTC；未满足门槛就保存未完成，不通过更换案例或放宽门槛冒充完成。

## 方案比较与选择

1. 只移植积分、M/J/内力留主机：不能满足全链路目标，排除。
2. 一次重写设备常驻全场景和所有状态格式：并非跑通所需，会混入优化和大范围合同变化，暂缓。
3. **选择：公开 tyx kernel＋有所有权的显式同步适配器。** 主机管理阶段、串行次序、迭代门槛、
   数据记录与独立验证；核心数值操作在所选后端执行。承认传输/分配和小 kernel 调度成本。
   保留原 FP64 CPU 实现作为物理参考，不复制外部引擎源码。

## 数据、精度与事务

P3/P4 RobotModel、RobotState、RobotInputs、TetrahedralModel 当前严格 FP64。为避免破坏已冻结合同，
本阶段继续以这些 FP64 容器保存主机快照，**设备计算明确为 FP32**：边界先检查再转 FP32，
输出只无损扩宽为 FP64 容器，不使用主机 FP64 修正动态求解结果。
BackendContext 的 dtype 仍表示存储精度，backend 为 tyx:MODE；program 签名及诊断必须显式包含
storage_dtype=float64、compute_dtype=float32、host_validation=float64、ordered_serial，公开文档必须同样说明。
模型静态预处理（参考体积/逆矩阵、质量、零位四元数矩阵）可由现有 FP64 构建器一次生成再量化，
不能把随候选 q/x 改变的 M/J/F/力作为“预处理”留主机。
固定点按入口原位原样复制回主机快照保持精确事务不变量；设备使用量化的同一入口位置。
动态状态、输入与必要物理参数若 FP32 溢出/非零下溢、维度越界均明确拒绝；转换不得静默丢弃非零力。
不能表达的极端材料/病态质量矩阵明确失败，不用伪逆、正则化或 clamp 改物理模型。

每个 kernel 通过公开 tx.Runtime/use_runtime 执行；CUDA 使用真实 cuda_array，资源逐个注册到 ExitStack。
任何分配、kernel、同步、下载、材料域、迭代或 close 失败，全帧返回入口所有域/history；程序异常继续暴露。
workspaces 不保留可影响下一帧的隐藏物理状态。缓存限编译产物，关闭幂等。launch/upload/download 计数可审查。

## 算法与组件

### 机器人计算引擎

TyxRobotMath(model, mode, cache_root=None) 拥有 runtime 与静态模型转换，不调用 CPU 机器人求解函数。
geometry(q) 返回世界位置/旋转/轴/COM；jacobians(q, indices, local_points) 返回线性和角 J。
mass(q)、bias(q,v)、generalized(q, effort, force, torque)、solve(q, load)、acceleration(q,v,inputs)
均由 kernel 计算。质量仍采用 COM Jacobian 二次型，bias 用世界惯性载荷；拓扑有序遍历，
稠密 Cholesky 前后替代，正对角/有限性/对称性与归一化残差门禁；任意共享祖先响应由整体 J M^-1 J^T 得出。
首版可串行一个逻辑 lane 处理一棵树，不声称并行效率。点位置、速度、世界/局部变换、合并载荷、
配置更新和瞬时冲量更新也通过 tyx 数值原语；瞬时冲量严格不推进 q。

### 节点与材料

TyxNodeMath 提供 FP32 预测、速度重建、节点响应/载荷更新和有序 XPBD sweep。
布料保持原约束顺序；不改成着色 Jacobi 使比较对象不同。每轮内累计 lambda，子步初始清零。
StVK kernel 计算 F、Green strain、能量、Piola 和每单元节点力，确定顺序装配共享节点；保留固定节点内力。
负 lambda 使用已修复的体积/偏差分解，避免纯体积应变被消去。模型域仍 det(F)>0.2、sigma∈[0.6,1.4]。
主机独立重算形变域和能量/力误差作为验证，不给求解器提供替代力。
重建使用 v_pred+(x-x_pred)/h，只计一次配置载荷；固定点速度零。

### 联合求解

TyxRobotParticipant/TyxClothParticipant/TyxTetrahedralParticipant 实现既有 ParticipantWorkspace 合同。
TyxArticulatedCouplingProgram 消费既有 SceneModel/attachments/contacts；单独局部 registry，
不改变 CPU program 接受的后端/精度，不改 SceneStepper。
TyxNodeRobotConstraints 保留逐行投影与事件结构；球几何、相对法向速度、有效响应、lambda 更新与向量载荷由 tyx 执行。
主机仅按已读标量决定下一次 sweep、分支与拒绝，保存前态/后态差以独立审计。
正负载荷来自同一量化向量；generalized load 由设备 J^T 计算，世界反作用只记一次。
独立审计不得调用被测计算引擎推导期望值。

## 冻结验证门槛

下列是实施前的 P6 FP32 门槛，旧 P5 FP64 门槛不变。失败先定位，不按结果扩大阈值。

| 层级 | 门槛/证据 |
| --- | --- |
| 单/两连杆、混合分支解析及 FP64 随机对照 | FK/J atol=2e-5 rtol=2e-5；M/bias/加速度 atol=2e-4 rtol=2e-4；质量方程归一化残差≤2e-5 |
| 材料力/能量 | 单元、24单元、共享节点与负泊松比，atol=2e-4 rtol=5e-4；固定支撑力不能丢失 |
| 点响应/解析接触 | 重复点、共享祖先、偏心力、零/非零响应；q/v绝对误差≤2e-5，同点成对载荷尺度误差≤5e-8 |
| S2/S3/S4 | 200帧、h=1/4000、两子步，位置/速度各200sweep、tolerance=2e-6；其余模型/初态/输入不变 |
| 独立每子步 | 穿透≤5e-6m、闭合速度≤5e-5m/s、关节动量平衡误差≤5e-5kg·m/s、成对载荷误差≤5e-8尺度；固定点位置/速度严格不变/零，材料域不放宽 |
| CUDA vs FP64 同设置 | 全程关节q差≤5e-4m、节点位置差≤1e-3m、速度差≤2e-2m/s；两节点域均实际接触≥5子步 |
| 确定性与实际执行 | 同CUDA再运行完整三场景，所有物理数组逐元素一致；诊断包含真实非零launch与执行模式，不把interpret算GPU |
| 小场景三模式 | 三模式都执行含真实接触的解析一对及布/软体短场景，不能只编译空kernel |

对照是相同物理模型与同 P6 迭代容差的 FP64 参考。单独保存新配置/审计函数/加载器，
不覆盖 P5 记录，不借用原严格loader绕过其精度/场景身份合同。FP32计算元数据与输入/源码哈希进入记录。
伪造错场景/错模式/缺少设备执行证据/重新hash后的载荷污染必须被消费端拒绝。

## 工程与性能交付

测试先 RED→GREEN；完整 CPU 回归、实际 CUDA 全回归、公开依赖边界、uv/pip/打包保持可用。
按三模式分模块验证，GPU缺失报告未执行。P6整分支一次独立审阅，Important修复按TDD。
冻结源后串行测量：构造/首调用、2次预热、3×20完整step，保留每个质量门槛与launch/传输计数；
完整200帧耗时另记。只做观测，不要求快于CPU，不因计时剔除验证。
确认的依赖缺口写tyx_flaw，包含最小复现/版本/证据等级；自身集成慢写performance-bottlenecks。
无等价Warp实测不得声称差距。保存完整可复现命令和来源，推送专用分支/核对SHA及CI，不合并主分支。
P6验收后才进入P6.5详细实现，案例仍需真实夹持与Three.js回放。
