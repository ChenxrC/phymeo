# MuJoCo-TYX 多接触共同法向求解：T3a 本机验收

2026-11-10，产品/测试快照 `a9a39dcb5228218bf4d65998fabcdabd15183b1f`。新后端已经能在固定根和原生自由根树上生成多个有限形状接触，通过完整共享惯量共同求解法向力，并把力、双侧力矩和运动一起发布。interpret、真实 CUDA、公共 Env/PolicyRunner 和新 wheel 源码外使用均已验证。默认 CUDA 仍选择 MuJoCo-TYX，CPU 负责模型编译和明确的观察；没有在设备 step 内调用 CPU/Warp 求解。

## 能力和物理语义

本批支持 plane–sphere、plane–capsule、plane–box、sphere–sphere、sphere–capsule、sphere–box 六类。包括有限盒子的面、边、角和内部最近面，胶囊单端/双端，盒子多角、同树双活动分支和原生碰撞过滤。范围为固定根1–8个 scalar DOF，或一个 free root 加0–29个 scalar joints，nq≤36、nv≤35。没有扩成同世界多个独立 free root。

所有可能接触的混合 condim 必须是1，混合 margin 非负。保留原 solref/solimp、margin/gap、原 body_invweight0 和当前完整两体点 Jacobian。每世界最多128个 geom、16对可能碰撞、32个静态 feature 槽；plane–box 每对预留8个角身份、按 CPU 规则最多产生4个 contact。运行 contact_capacity 计全部检测接触，constraint_capacity 计 active 行，包括零力行；恰满接受，溢出整批拒绝，不截断。

五阶段设备路径为当前 FK → 完整平滑动力学 → 碰撞与接触行 → 共同法向 QP → Euler 积分。令 `a0=M^-1 rhs`，解 `min(f>=0) .5 f^T A f+b^T f`，其中 `A=J M^-1 J^T+diag(R)`、`b=J a0-aref`。约束力为完整 `J^T f`。接触 QP 使用物理 M；Euler 隐式阻尼另使用 M+hD。双侧 world wrench 在同一接触点、相对各自 COM 计算。

求解器明确称 `normal-dual-active-set-v1`：冷启动零力、有限 active-face 线性解预算，不声称复现原生 Newton/CG/PGS 迭代或 warmstart。source iterations 映射为1–512次 face solve；实际 FP32 归一 KKT 门为 max(source tolerance,2e-5)，完整定义在[执行前合同](../superpowers/plans/2026-11-10-mujoco-tyx-coupled-normal.md)和[独立案例](../development/mujoco-tyx-t3-cases.md)。预算耗尽、坏主元、非有限或晚回读失败都不发布部分世界。

`last_step.contacts` 提供 geom/body 身份、feature、世界点/frame、完整 J、原接触参数、实际 normal force 和双方 COM wrench。`raw_contact_fields` 保留全部静态槽、设备行、J、dual matrix 及 solver diagnostics。检测到但未激活的接触，以及激活但最终零力的行均保留；同次入步力与端点 CPU 观察分开标记，不把力叫整步冲量。

## 实际验证

测试独立从原 XML 创建 MuJoCo3.3.7 模型。每个设备入步另作 native geometry/J/M/R/aref/force/qacc 对照；小于等于4行时再用测试侧穷举 active sets 的独立 QP oracle。连续 CPU 从相同存储初态、相同控制向前推进，未将 CPU 状态复制回 TYX。工程容差在首次执行前登记，未调整模型、h、质量或误差门追结果。

| 原始任务 | 结果 | 范围 |
|---|---|---|
| 06-environment | exit0 | Python3.12.14、MuJoCo3.3.7、TYX0.1.0a0、NumPy2.5.4，实际模块来源 |
| 10-interpret | 44 passed、1 failed、33 deselected / 68.67s | 全部专项通过；父 ENV 的错误正力预期失败，见下文 |
| 12-env-interpret | 1 passed、1 deselected / 5.11s | 修正原生力预期后的完整公共 ENV/reset/clone |
| 13-cuda | 33 passed、45 deselected / 44.28s | 全部真实 CUDA 专项及公共 ENV |
| 14 / 15 | exit0 | 最终精确10个 Python 文件 Ruff 和 format-check |
| 16-adjacent | 9 passed / 3.00s | T0 slider/斜平面、T1偏心摆、T2偏心自由根与摩擦准入 |
| 17 / 18 | exit0 | 新 wheel 构建与源码外无依赖安装 |
| 19 / 20 | 各exit0 | 新安装包 interpret/CUDA 的共同接触策略环境 |

最终覆盖为45个 interpret 节点、33个 CUDA 节点；interpret 按上述两组联合覆盖，没有重跑已绿的44项。无 skip；deselected 只来自后端分组。各后端六条主轨迹共288步，另有12个 primitive 用例、解析双接触反例、容量、过滤和事务测试。双接触解析反例区分共同解 `beta/(2+r)` 与错误的逐点独立解 `beta/(1+r)`。

主轨迹包括偏心自由盒子四角承载、自由根双脚、运动兄弟分支、有限盒边撤离，以及35维系统最后关节/身体的实际接触。W4稀疏世界有不同接触数；selected reset 后比较未重置世界与 clone 的真实下一步，被重置世界与 fresh runtime 比较。检测容量和 active 容量分别检验恰满/溢出；一轮 face-solve 预算的双法向负控拒绝整批；真实设备回读后注入的失败可原请求重试。禁止 CPU/Warp/HOST/NumPy solve 时仍完成设备步。

公共 ENV 模型为偏心 free root 加两个关节、两个原接触点，稀疏 worlds4/31，控制每3步更新并读取上次接受的接触力。12步连续 CPU 运动及接触力对照后，selected reset 和 clone 第13步一致。每个 wheel 消费者执行26个主 runtime world-steps加1个 clone world-step，核所有 Phymeo 模块来自新安装目录。

## 首次失败和修复

- 01 import：TYX frontend 不支持 `1 << j`；改成 j=0/1/2 对应 bit_weight=1/2/4，保持原角身份及几何规则。06复核通过。
- 02 / 03：一个测试 import 排序及8个文件格式；04 / 05修复，最终14 / 15通过。
- 09首个测试：FP32静态转换助手返回一元素数组，NumPy2.5不能用 float(array)；改为 `.item()`，保实际 FP32 KKT 门。
- 10 / 11：父 ENV 错误假设“两只脚必须同时正力”。原倾斜初态下 native MuJoCo 的12步也由一行零力、另一行承载；11逐步比对两行实际力全部通过后仍触发旧断言。修正为保留两行、验证真实承载和零力行，并保留逐步 native force 对照。原模型、轨迹长度和误差门不变，12完成余下 reset/clone。原失败日志保留，没有把10记为完整成功。

## 证据与边界

[本批证据](../../artifacts/p65-mujoco-tyx-t3-20261110-01)保留20个实际命令的 guard、源码快照、日志及 raw exit。60个同次 JSONL 覆盖40个固定轨迹身份，主轨迹保全部 q/v/control 和预定首/中/末完整 receipt；未为报告重跑物理或求解器。

[最终文件核对](../../artifacts/p65-mujoco-tyx-t3-20261110-01/final-audit.json)核128项当前源码绑定、保存身份/行数、470个 wheel 产品/资源成员与安装目录，0 mismatch；fresh执行匹配0。wheel SHA256 `09332a2534d93e79a35c28d75fefeec7c6b8f24d97a4f86aea35365ff359d693`，Apache LICENSE/NOTICE/AUTHORS随包。原作者全部释放，父唯一执行槽结束；K1三个旧WIP未动，无CI/push/pin变更。

这是 T3a 多法向接触能力，不是全部 T3 或 G1 站立。capsule–capsule、capsule–box、box–box，有限面完整卸载重入矩阵、摩擦/关节停止、mesh/CCD、equality/tendon、多个自由根及大规模吞吐仍未覆盖。当前有界 dense solve 和 NumPy观察往返不宣称性能超越Newton或全设备训练。下一批优先 T3b 补剩余 primitive 配对和连续重入，再依 T4 接 native 摩擦与限制，最后 T5 G1组合验证。
