# Phymeo / Newton 功能差异全景（2026-11-11）

2026-11-11更新：用户已取消08:00限制，改为按节点验收持续推进。上一批功能/保存/安装包验收已完成，核对基线d6a5c383、产品/测试e3d15707；本次新增路线仍是计划，未新增实现或测试通过。对照固定为 Newton 1.7.0，不把它的不同求解器能力合成一套默认保证。训练与规模是当前最高主线，场景/机器人、碰撞和GPU/多场景仍是前三个基础方向。

[上一批实际记录](p65-parity-2026-11-11-progress.md) · [持续推进计划](../superpowers/plans/2026-11-11-newton-parity-continuous.md) · [架构与后续路线图](../architecture/phymeo-newton-parity-2026-11-11.html)

## 本轮已连接的能力

| 连接 | 当前证据 | 仍然存在的边界 |
|---|---|---|
| 物理状态 → 中途检查点 → 原样继续 | 公共 checkpoint/restore；固定根/自由根、完整接受状态与控制、CAS 和失败回滚；恢复后下一步状态与收据一致 | 上层策略/RNN/随机数状态须明确组合；不是完整训练作业快照 |
| 四类机器人 → 真实受载多世界分页 | W127/128/129/256，world 不拆页，所有载荷汇总，每 storage 一次求解、全批共同发布；两后端专项已过 | 默认仍走HOST patch；显式设备patch分页已另行验收。材料点转换/载荷装配与诊断仍有HOST工作 |
| 已接受设备状态 → 观测/奖励/终止 | 公共 MuJoCo-TYX task workspace，真实 TYX consumer，reset/restore 后旧数据失效 | 固定显式 q/v 映射和二次任务；未实现通用 CUDA Torch/PPO、随机化或全部传感器 |
| 设备策略输出 → 三类控制 → 物理推进 | 公共设备 action packet；ctrl、qfrc_applied、xfrc_applied，完整字节审计、流同步、CAS；29 执行器宽度已测 | 当前仍保留 HOST 控制审计和完整物理诊断读回，不是零往返环 |
| 设备凸体 patch → 原材料点 → 真实载荷 | 新十类无序 S/C/OBB/H 组合资格、四类树来源；单页各67项/47份保存；分页另各29项/28份保存，完整力/COM力矩、逐世界拒步、异常回滚与新wheel均通过 | 此 resident 路径与 MuJoCo-TYX 分开；是设备几何加 HOST 力装配，尚非广泛 mesh/变形窄相或 CCD |
| G1 策略 → 正确 IMU-site 角速度 | 新版16项语义测试；原条件2秒各1000个 CUDA/独立CPU步骤与保存审计均过 | 新版于46.29秒触发设备1米参考XY门，独立CPU同时0.99978米，无设备拒步；未完成131.48秒。旧版112.372秒失败另保留 |

本轮 action-pack 复用物理 Runtime 的编译缓存，固定 W1/8/32 测量中每次稳态提交从约 13–15 ms 降至约 0.65/0.80/0.98 ms。完整数据拷贝和审计计数保持；这是本机小案例的局部接口测量，不是对 Newton 的总吞吐优势，也不是统计充分的性能结论。

本轮新wheel的483个产品文件与源码、安装目录逐字节一致；五类公共能力的interpret/CUDA十组源码外调用全部通过。两次环境预检首错和最小修复保留，见[正式验收](../benchmarks/p65-parity-20261111-connections.md)。W256是隔离slider/support世界的限定受载案例，不是256台G1或GPU PPO吞吐。

G1仍未完成原131.48秒动作。保存诊断中两条独立轨迹根部XY差最大17.30mm、末步3.29mm，而双方参考误差接近1m；同输入峰值已定位到接触检出集合不同：native额外检出约-33nm间隙的右足胶囊点；设备拒绝距离未保存，未证明其导致长期漂移。准确训练run配置尚未与公开ONNX绑定，不凭默认配置调参。见[事实、来源与下一定位](g1-drift-diagnosis-20261111.md)。

## 两条物理执行路径

### Phymeo resident 路径

资产/程序模型 → 普通、affine、floating、floating-affine 树 → TYX FK/AABB/BVH → 默认 HOST patch 或限定设备 S/C/OBB/H patch → 原材料点/完整 Jacobian/力与 COM 力矩 → TYX 完整动力学 → 唯一接受状态与收据发布。

world-complete受载分页已连接默认HOST patch及显式device patch，后者实际通过W129/W256连续受载和末页失败矩阵。不拆世界，每storage一次求解，各页有效后共同发布；保留工程精度与未知/资源失败语义。设备站点上限按页计，原world站点上限独立；更复杂异构规模仍需另验。

### MuJoCo-TYX 路径

URDF/MJCF/程序场景 → CPU MuJoCo 编译 MjModel → 明确的模型/状态映射 → 真正 TYX 的固定根或单自由根树动力学 → 设备 primitive 接触、limits、dry friction 和 pyramidal condim 1/3/4/6 → 有界共同求解 → 公共运行环境、策略与设备工作区。

CPU 编译不是 CPU 代跑 CUDA 物理。默认 CUDA 使用 MuJoCo-TYX；MuJoCo-Warp 是显式选择的独立实现。当前设备物理仍附有 HOST 观察/诊断，HOST ONNX 与完整 GPU 训练是不同边界。这条路径的原生 primitive 能力不会自动赋予 resident 路径，反之亦然。

## 八个方向，哪些大模块仍缺

| 方向 | 已有通用基础 | 确实尚缺或覆盖不足 | 后续可验节点 |
|---|---|---|---|
| 01 场景与机器人 | URDF/MJCF/USD、统一编译接口、PD/IK、普通/浮根/affine、工具/根惯量、双方反力、中途恢复 | 更广关节/闭环/传动，复杂执行器，同world独立participant设备联解；完整抓取释放 | ROB1双自由体/机械臂推盒/双臂工具；ROB2球关节/执行器及夹爪、Panda工作流 |
| 02 碰撞检测与响应 | RR/RD/DD、有限面边、多形状patch、部分凹体/CCD；设备S/C/OBB/H消费；圆柱/圆锥/椭球有限HOST support-map | 后三形状的设备完整消费；广泛mesh/变形窄相/CCD；跨特征摩擦历史；SDF/hydroelastic管线 | COL1七形状28类矩阵，旧十类复用；COL2设备mesh/变形；COL3首触；其后历史/SDF独立节点 |
| 03 GPU 与多场景 | 四类树、W256运动和受载分页，设备几何、MuJoCo-TYX混合约束、公共设备数据口 | HOST转换/装配/诊断开销、异构批次/约束规模、跨participant联解；真实机器人训练规模尚缺 | N4 W1/8/32/64/128/256实际链路资格和端到端指标；256当前只是task接口上限，不能倒灌resident结果 |
| 04 训练与可微（最高主线） | 公共向量ENV、接受mask/final observation/reset、CPU PPO三任务、ONNX、设备task/action与物理checkpoint | 真实CUDA Torch/PPO；完整训练恢复/RNN/随机化；多步梯度/辨识 | N1真实GPU策略→N2两任务PPO→N3训练恢复→N4三类任务/三种子学习；物理可微另立梯度对照 |
| 05 传感与交互 | 深度/法线/ID、TYX射线、IMU基础、真实轨迹Three.js；G1示例明确传感器坐标绑定 | RGB/材质/光照、更广相机、批量传感和编辑生态；示例 gyro 修正不是通用传感子系统完工 | 同一真实场景的多传感器、时序/坐标协议及交互输入 |
| 06 跨域求解编排 | 事务、点端口、域图与指定刚–布–软体组合 | 通用按区域选择后端、界面力/约束、耦合调度与共同失败恢复 | 两个异构后端的完整界面载荷、能量与回滚案例 |
| 07 布与体软体 | 拉伸/剪切/弯曲、自接触、StVK四面体、双向载荷、递布基础 | 密集折叠、持续静摩擦/换特征、软块抓捏；目标软布放台仍有留出 | 多网格/刚度/初态的支撑、抓捏、折叠、放台；递布必要但不充分 |
| 08 杆缆、MPM/流体 | 刚性连杆、粒子、弹性体基础 | 完整弯扭杆缆；粒子–网格转移、塑性/流动、流固边界 | 分别建立扭转绕柱、颗粒堆积/倒砂、溃坝与双向边界库 |

这里的“已有”指已声明的能力域，不等于所有组合都能工作。Newton 同样按求解器与配置区分，例如其可微、布自碰撞和多物理域能力不能统一归给所有后端。

## 后续顺序与停止规则

| 下一批次 | 主线与并行工作 | 成功判断 |
|---|---|---|
| 1 | N1独立CUDA Torch实际策略闭环；ROB1/COL1冻结接口、多案例和独占文件 | 真GPU策略驱动两类连续物理；另外两分支得到可直接开发的边界 |
| 2 | N2 device VecEnv/两任务PPO；ROB1与COL1隔离编码 | 实际优化器更新；多参与者双方受力；通用形状矩阵分层资格 |
| 3 | N3训练恢复；ROB2关节/执行器；COL2再COL3 | 恢复后真实继续一致；多案例关节控制；设备mesh/变形与首触边界 |
| 4及以后 | N4多任务学习/规模；后续机器人、碰撞和其它物理族 | 预定种子/预算下可重复改善；相同条件的资源与吞吐报告 |

批次允许独立分支先行，具体硬依赖见[总计划](../superpowers/plans/2026-11-11-newton-parity-continuous.md)。三个原agent分别负责训练、机器人、碰撞；父负责共享接口与唯一串行执行。N1可进入下一批，其他新分支先完成短接口/案例冻结；这里没有宣称已启动编码。

G1保持兼容侧线：来源绑定与保存输入的接触边界诊断，不自动重跑完整动作，也不阻塞训练。接受预先声明的工程精度，不为零margin位级一致无限推导。性能按同模型、初态、精度和输出比较；HOST诊断成本单列，生产快速模式的检查契约须明示。

之后依次补传感/交互、通用跨域、布与体软体、杆缆/MPM/流体；物理可微与训练接口不是同一能力。每个物理族单独研究立项，不能只复用某个示例的通过。

一个节点应产出可调用公共能力、独立正负案例、实际执行证据和安装包验证。连续两次针对性尝试未缩小同一障碍，保存最小反例、拆分或暂停该节点，独立支线继续。研究须落到可开发接口与案例，不能无限追加理论/统计。已绿同源不重跑；不调质量/初态/阈值追某一条轨迹。不用清单数、参数组合数或测试数计算整体百分比。

## 固定参照与历史证据

[11月7日全景及历史验收](phymeo-newton-panorama-2026-11-07.md)保留旧能力演进；当前状态以本页和本轮实际记录为准。

Newton 1.7 官方入口：[求解器矩阵](https://newton-physics.github.io/newton/1.7.0/solvers/index.html)、[碰撞](https://newton-physics.github.io/newton/1.7.0/concepts/collisions.html)、[多世界](https://newton-physics.github.io/newton/1.7.0/concepts/worlds.html)、[传感器](https://newton-physics.github.io/newton/1.7.0/concepts/sensors.html)、[耦合](https://newton-physics.github.io/newton/1.7.0/concepts/coupling.html)、[变形与杆](https://newton-physics.github.io/newton/1.7.0/concepts/deformable_objects.html)。以上是架构/功能参照，不冒充当前机器上的逐项 Newton 性能测量。
