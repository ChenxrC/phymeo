# Phymeo / Newton 功能差异全景（2026-11-11）

更新时间：北京时间 04:40，本轮仍在执行，08:00 收口。对照固定为 Newton 1.7.0，不把它的不同求解器能力合成一套默认保证，也不声称 Phymeo 已逐项超过它。训练与规模是当前最高主线，场景/机器人、碰撞和 GPU/多场景仍是前三个基础方向。

[本轮实际记录](p65-parity-2026-11-11-progress.md) · [长程计划](../superpowers/plans/2026-11-11-newton-parity-until-0800.md) · [架构全景图](../architecture/phymeo-newton-parity-2026-11-11.html)

## 本轮已连接的能力

| 连接 | 当前证据 | 仍然存在的边界 |
|---|---|---|
| 物理状态 → 中途检查点 → 原样继续 | 公共 checkpoint/restore；固定根/自由根、完整接受状态与控制、CAS 和失败回滚；恢复后下一步状态与收据一致 | 上层策略/RNN/随机数状态须明确组合；不是完整训练作业快照 |
| 四类机器人 → 真实受载多世界分页 | W127/128/129/256，world 不拆页，所有载荷汇总，每 storage 一次求解、全批共同发布；两后端专项已过 | 原分页的接触几何和载荷装配仍在 HOST；设备 patch 分页组合另验 |
| 已接受设备状态 → 观测/奖励/终止 | 公共 MuJoCo-TYX task workspace，真实 TYX consumer，reset/restore 后旧数据失效 | 固定显式 q/v 映射和二次任务；未实现通用 CUDA Torch/PPO、随机化或全部传感器 |
| 设备策略输出 → 三类控制 → 物理推进 | 公共设备 action packet；ctrl、qfrc_applied、xfrc_applied，完整字节审计、流同步、CAS；29 执行器宽度已测 | 当前仍保留 HOST 控制审计和完整物理诊断读回，不是零往返环 |
| 设备凸体 patch → 原材料点 → 真实载荷 | 新十类无序 S/C/OBB/H 组合资格、四类树来源；单页消费解释/CUDA 各67项，完整力/COM力矩、逐世界拒步、异常回滚 | 此 resident 路径与 MuJoCo-TYX 分开；是设备几何加 HOST 力装配，尚非广泛 mesh/变形窄相或 CCD |
| G1 策略 → 正确 IMU-site 角速度 | 新版16项语义测试；原条件2秒各1000个 CUDA/独立CPU步骤与保存审计均过 | 新版完整131.48秒正在运行；旧版在112.372秒触发独立CPU的1米参考XY门，保留失败 |

本轮 action-pack 复用物理 Runtime 的编译缓存，固定 W1/8/32 测量中每次稳态提交从约 13–15 ms 降至约 0.65/0.80/0.98 ms。完整数据拷贝和审计计数保持；这是本机小案例的局部接口测量，不是对 Newton 的总吞吐优势，也不是统计充分的性能结论。

本轮新源码外 wheel 与公共入口检查仍待最后源冻结，不能提前将源码测试等同安装包验收。旧已验安装包仅覆盖其原版本。

## 两条物理执行路径

### Phymeo resident 路径

资产/程序模型 → 普通、affine、floating、floating-affine 树 → TYX FK/AABB/BVH → 默认 HOST patch 或限定设备 S/C/OBB/H patch → 原材料点/完整 Jacobian/力与 COM 力矩 → TYX 完整动力学 → 唯一接受状态与收据发布。

已增加 world-complete 受载分页；目前默认分页仍走 HOST patch。下一连接是把单页已验设备 patch 事务组合到分页，不拆世界、不增加每 storage 求解次数、不允许末页错误留下前页成功结果。保留工程精度与未知/资源失败语义。

### MuJoCo-TYX 路径

URDF/MJCF/程序场景 → CPU MuJoCo 编译 MjModel → 明确的模型/状态映射 → 真正 TYX 的固定根或单自由根树动力学 → 设备 primitive 接触、limits、dry friction 和 pyramidal condim 1/3/4/6 → 有界共同求解 → 公共运行环境、策略与设备工作区。

CPU 编译不是 CPU 代跑 CUDA 物理。默认 CUDA 使用 MuJoCo-TYX；MuJoCo-Warp 是显式选择的独立实现。当前设备物理仍附有 HOST 观察/诊断，HOST ONNX 与完整 GPU 训练是不同边界。这条路径的原生 primitive 能力不会自动赋予 resident 路径，反之亦然。

## 八个方向，哪些大模块仍缺

| 方向 | 已有通用基础 | 确实尚缺或覆盖不足 | 后续可验节点 |
|---|---|---|---|
| 01 场景与机器人 | URDF/MJCF/USD、统一编译接口、PD/IK、普通/浮根/affine、工具/根惯量、双方反力、中途恢复 | 更广关节/闭环/传动，复杂执行器，多个独立 participant 在同 world 的通用设备联合求解；完整抓取释放工作流 | 多机器人/工具/关节组合库，独立动力学与反作用对照，公共安装包操作 |
| 02 碰撞检测与响应 | RR/RD/DD、有限面边与多形状 patch、部分凹体/CCD、转动逐面设备候选；限定设备凸体消费 | 更广 mesh/凹体/变形组合的设备窄相与连续检测；跨特征材料身份、持续静摩擦/滚动历史 | 形状×表示×运动×所有者矩阵；每层独立几何和真实受载验收 |
| 03 GPU 与多场景 | 四类树、W256运动和真实受载分页，设备几何、MuJoCo-TYX 混合约束、公共设备数据口 | 几何与受载分页的完整组合、HOST装配/诊断开销、异构批次/更大约束规模、跨 participant 联解 | 先完成分页设备接触；再固定物理和输出的吞吐/传输/峰值内存测试，按热点优化 |
| 04 训练与可微（最高主线） | 公共向量ENV、接受mask/final observation/reset、CPU PPO三任务、原ONNX、设备 task/action 端口 | 实际 CUDA Torch/PPO 或其它GPU策略接入；完整训练恢复/RNN/随机化；多步梯度和参数辨识 | 设备真实策略消费者→公共env完整rollout→训练/保存/恢复；可微另设梯度对照，不能拿CPU PPO代替 |
| 05 传感与交互 | 深度/法线/ID、TYX射线、IMU基础、真实轨迹Three.js；G1示例明确传感器坐标绑定 | RGB/材质/光照、更广相机、批量传感和编辑生态；示例 gyro 修正不是通用传感子系统完工 | 同一真实场景的多传感器、时序/坐标协议及交互输入 |
| 06 跨域求解编排 | 事务、点端口、域图与指定刚–布–软体组合 | 通用按区域选择后端、界面力/约束、耦合调度与共同失败恢复 | 两个异构后端的完整界面载荷、能量与回滚案例 |
| 07 布与体软体 | 拉伸/剪切/弯曲、自接触、StVK四面体、双向载荷、递布基础 | 密集折叠、持续静摩擦/换特征、软块抓捏；目标软布放台仍有留出 | 多网格/刚度/初态的支撑、抓捏、折叠、放台；递布必要但不充分 |
| 08 杆缆、MPM/流体 | 刚性连杆、粒子、弹性体基础 | 完整弯扭杆缆；粒子–网格转移、塑性/流动、流固边界 | 分别建立扭转绕柱、颗粒堆积/倒砂、溃坝与双向边界库 |

这里的“已有”指已声明的能力域，不等于所有组合都能工作。Newton 同样按求解器与配置区分，例如其可微、布自碰撞和多物理域能力不能统一归给所有后端。

## 后续顺序与停止规则

1. 完成本轮设备 patch 消费的保存/安装包验收，组合到 world-complete 分页，保持原单次发布和双侧力学。
2. 在公共设备 task/action 入口上接实际 GPU 训练消费者；先核环境能力，再做完整短训练、选择 reset、保存/恢复。单个数据接口或TYX toy consumer不算PPO训练完工。
3. 扩展同world独立机器人/工具的设备共同求解与更多约束；保持独立多案例库，不围绕单个G1动作定义通用成功。
4. 用同模型、同初态、同精度门、同输出级别测吞吐。当前完整 HOST 诊断与原始记录成本需分开报告；若要提供生产快速模式，必须明示检查频率和契约变化，不能悄悄去掉检查获取加速。
5. 依次补传感/交互、通用跨域、布与体软体、杆缆/MPM/流体；每个物理族单独研究和立项。

一个节点应产出可调用公共能力、独立正负案例、实际执行证据和安装包验证。连续两轮仅增加理论/统计却没有修复明确反例或新增可验证能力，就收缩该支线。已绿同源不重跑；不调质量/初态/阈值追某一条轨迹。不用模块项数、参数组合数或测试数计算整体完成百分比。

## 固定参照与历史证据

[11月7日全景及历史验收](phymeo-newton-panorama-2026-11-07.md)保留旧能力演进；当前状态以本页和本轮实际记录为准。

Newton 1.7 官方入口：[求解器矩阵](https://newton-physics.github.io/newton/1.7.0/solvers/index.html)、[碰撞](https://newton-physics.github.io/newton/1.7.0/concepts/collisions.html)、[多世界](https://newton-physics.github.io/newton/1.7.0/concepts/worlds.html)、[传感器](https://newton-physics.github.io/newton/1.7.0/concepts/sensors.html)、[耦合](https://newton-physics.github.io/newton/1.7.0/concepts/coupling.html)、[变形与杆](https://newton-physics.github.io/newton/1.7.0/concepts/deformable_objects.html)。以上是架构/功能参照，不冒充当前机器上的逐项 Newton 性能测量。
