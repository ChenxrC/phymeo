# Phymeo / Newton 功能差异全景（2026-11-11）

2026-11-12当前增量：N4b九份完整保存诊断已完成，J/R弱收益原因未定；固定根NONE/FILTEREXACT执行器与act/力/运动共同发布、新显式integration observation及499产品新包安装已通过。COL同候选停止实验新增9个资源回归，未合入。当前执行到07:20，见[本轮计划](../superpowers/plans/2026-11-12-newton-parity-until-0720.md)与[实际批次](p65-parity-batch-n4b-rob2a-col-stop.md)。以下旧N4a数量为历史，不覆盖本段。
2026-11-11最新：N4a九次固定预算训练完成，持续支撑S通过学习门，中位成本下降26.4%；J/R下降5.6%/8.2%，未达原门。三任务W1–W256共18格规模通过；ROB1d新wheel源码外interpret/CUDA均通过；通用仿射重建修复两个HN失败，四个Y法向失败保留。训练与规模仍为最高主线，无固定截止。见[N4a](p65-mujoco-tyx-n4a-report.md)、[ROB1d](p65-resident-rob1d-installed-report.md)、[COL修复](p65-convex-affine-reconstruction-report.md)、[本批结果与后续](p65-parity-batch-n4a-rob1d-col-reducer.md)。

[上一批实际记录](p65-parity-2026-11-11-progress.md) · [持续推进计划](../superpowers/plans/2026-11-11-newton-parity-continuous.md) · [上一连接批架构快照](../architecture/phymeo-newton-parity-2026-11-11.html)

## 本轮已连接的能力

| 连接 | 当前证据 | 仍然存在的边界 |
|---|---|---|
| 物理状态 → 组合训练检查点 → 原样继续 | N3实际组合物理/控制、episode、PPO/Adam、normalizer、RNG及GRU hidden；恢复后下一次真实更新一致，晚失败语义已验 | 同设备/配置、完整update边界；不是任意时刻、任意learner或跨平台快照 |
| 四类机器人 → 真实受载多世界分页 | W127/128/129/256，world 不拆页，所有载荷汇总，每 storage 一次求解、全批共同发布；两后端专项已过 | 默认仍走HOST patch；显式设备patch分页已另行验收。材料点转换/载荷装配与诊断仍有HOST工作 |
| 已接受设备状态 → VecEnv → CUDA PPO | N3完整MLP/GRU恢复；N4a三任务三种子共294912训练转移，持续支撑S学得可重复改善；三任务18格W1–W256规模实测 | J/R未达原学习门；随机化、更复杂真实机器人训练、多步梯度仍缺 |
| 同world独立参与者 → 共同载荷 → 一次发布 | ROB1c混合四storage与完整world事务；ROB1d新安装包两后端分别13次调用/21个logical-world steps，含机械臂推盒、恢复后继、局部reset、错owner拒绝 | <=8参与者、<=33物理body/world；HOST候选/patch，非跨树隐式QP；双臂R03两个精度门仍失败 |
| 设备策略输出 → 三类控制 → 物理推进 | 公共设备 action packet；ctrl、qfrc_applied、xfrc_applied，完整字节审计、流同步、CAS；29 执行器宽度已测 | 当前仍保留 HOST 控制审计和完整物理诊断读回，不是零往返环 |
| 设备凸体 patch → 原材料点 → 真实载荷 | 新十类无序 S/C/OBB/H 组合资格、四类树来源；单页各67项/47份保存；分页另各29项/28份保存，完整力/COM力矩、逐世界拒步、异常回滚与新wheel均通过 | 此 resident 路径与 MuJoCo-TYX 分开；是设备几何加 HOST 力装配，尚非广泛 mesh/变形窄相或 CCD |
| G1 策略 → 正确 IMU-site 角速度 | 新版16项语义测试；原条件2秒各1000个 CUDA/独立CPU步骤与保存审计均过 | 新版于46.29秒触发设备1米参考XY门，独立CPU同时0.99978米，无设备拒步；未完成131.48秒。旧版112.372秒失败另保留 |

此前连接批的 action-pack 复用物理 Runtime 的编译缓存，固定 W1/8/32 测量中每次稳态提交从约 13–15 ms 降至约 0.65/0.80/0.98 ms。完整数据拷贝和审计计数保持；这是本机小案例的局部接口测量，不是对 Newton 的总吞吐优势，也不是统计充分的性能结论。

最新wheel的494个产品文件与源码和安装目录逐字节一致；ROB1d两后端与N4公共HOST评价入口均源码外通过。首次启动器环境路径失败及修正保留。此前483成员/十组消费者只复用其原资格，不冒新包全部回归。N4a现在有真实GPU PPO规模测量：含完整trace关闭的W256吞吐约J693/S219/R560 transitions/s，均低于W128；只是三个已声明的小任务、单次有界测量，不是256台G1，也没有Newton同协议性能基线。

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
| 01 场景与机器人 | URDF/MJCF/USD、统一编译接口、PD/IK、四类树、双方反力；混合来源同world响应/恢复和ROB1d新安装资格 | 双臂R03小净加速度精度；更广关节/闭环/传动、复杂执行器和完整抓取释放 | R03有界PD算术复现；ROB2固定根一阶执行器已验；下一接自由根/接触约束，球关节与Panda工作流另验 |
| 02 碰撞检测与响应 | RR/RD/DD、有限面边、多形状patch、部分凹体/CCD；设备S/C/OBB/H消费；三曲面support、内切球界、迭代诊断；通用仿射重建已修复2HN | 4个圆柱Y法向失败；曲面device patch/load、广泛mesh/变形窄相/CCD、跨特征历史、SDF/hydroelastic | 同候选停止/见证发布假设；通过后COL1b/c；COL2设备mesh/变形，COL3首触 |
| 03 GPU 与多场景 | 四类树、W256受载分页、设备几何、MuJoCo-TYX混合约束、公共设备口；N4a三任务18格真实PPO规模和完整trace计时 | 物理推进/状态快照/诊断开销、异构批次/约束规模；256复杂机器人训练尚缺 | 按已有profile拆热点，明确快速模式与reference对照；同精度/输出Newton比较另立基线 |
| 04 训练与可微（最高主线） | CPU PPO三任务、ONNX、device VecEnv、上游RSL CUDA PPO；N3同配置续训；N4a持续支撑S三种子改善通过 | J/R固定预算弱收益、随机化、任意时刻/跨配置恢复；多步物理梯度/辨识 | N4b已诊断；N4c固定已存策略响应，再依证据冻结单次后继；物理可微另立梯度对照 |
| 05 传感与交互 | 深度/法线/ID、TYX射线、IMU基础、真实轨迹Three.js；G1示例明确传感器坐标绑定 | RGB/材质/光照、更广相机、批量传感和编辑生态；示例 gyro 修正不是通用传感子系统完工 | 同一真实场景的多传感器、时序/坐标协议及交互输入 |
| 06 跨域求解编排 | 事务、点端口、域图与指定刚–布–软体组合 | 通用按区域选择后端、界面力/约束、耦合调度与共同失败恢复 | 两个异构后端的完整界面载荷、能量与回滚案例 |
| 07 布与体软体 | 拉伸/剪切/弯曲、自接触、StVK四面体、双向载荷、递布基础 | 密集折叠、持续静摩擦/换特征、软块抓捏；目标软布放台仍有留出 | 多网格/刚度/初态的支撑、抓捏、折叠、放台；递布必要但不充分 |
| 08 杆缆、MPM/流体 | 刚性连杆、粒子、弹性体基础 | 完整弯扭杆缆；粒子–网格转移、塑性/流动、流固边界 | 分别建立扭转绕柱、颗粒堆积/倒砂、溃坝与双向边界库 |

这里的“已有”指已声明的能力域，不等于所有组合都能工作。Newton 同样按求解器与配置区分，例如其可微、布自碰撞和多物理域能力不能统一归给所有后端。

## 后续顺序与停止规则

| 下一批次 | 主线与并行工作 | 成功判断 |
|---|---|---|
| 本批已执行 | N4a持续支撑学习门通过、J/R未达；18格规模完成；ROB1d安装通过；COL两HN修复 | 学习/规模分别报告；四Y与R03精度失败仍保留 |
| 下一批 | J/R已存训练行为与优化证据诊断；物理/快照热点；ROB2a接口与案例；COL同候选停止 | 可复用开发边界和预登记多案例；不挑种子/改原门追绿，不把距离收敛当法向收敛 |
| 独立后继 | R03有界PD算术复现；COL1a通过后才进COL1b/c | 不为微小精度缺口阻塞训练；曲面patch/载荷分别验收 |
| 后继扩域 | ROB2关节/执行器；COL2再COL3 | 多案例关节控制；设备mesh/变形与首触边界 |
| 4及以后 | N4多任务学习/规模；后续机器人、碰撞和其它物理族 | 预定种子/预算下可重复改善；相同条件的资源与吞吐报告 |

批次允许独立分支先行，具体硬依赖见[总计划](../superpowers/plans/2026-11-11-newton-parity-continuous.md)。三个原agent分别负责训练、机器人、碰撞，当前作者任务均已交接；父负责共享接口与唯一串行执行。[ROB1合同](../superpowers/plans/2026-11-11-resident-multi-participant.md)的混合来源限域已通过，采用共同入步载荷、各机械块一次求解和完整world一次发布，不是跨树隐式KKT；R03精度缺口单列。[COL1a合同](../superpowers/plans/2026-11-11-device-convex-support-family.md)仍未完整验收，patch/受力另验。这些缺口不阻塞N4训练主线。

G1保持兼容侧线：来源绑定与保存输入的接触边界诊断，不自动重跑完整动作，也不阻塞训练。接受预先声明的工程精度，不为零margin位级一致无限推导。性能按同模型、初态、精度和输出比较；HOST诊断成本单列，生产快速模式的检查契约须明示。

之后依次补传感/交互、通用跨域、布与体软体、杆缆/MPM/流体；物理可微与训练接口不是同一能力。每个物理族单独研究立项，不能只复用某个示例的通过。

一个节点应产出可调用公共能力、独立正负案例、实际执行证据和安装包验证。连续两次针对性尝试未缩小同一障碍，保存最小反例、拆分或暂停该节点，独立支线继续。研究须落到可开发接口与案例，不能无限追加理论/统计。已绿同源不重跑；不调质量/初态/阈值追某一条轨迹。不用清单数、参数组合数或测试数计算整体百分比。

## 固定参照与历史证据

[11月7日全景及历史验收](phymeo-newton-panorama-2026-11-07.md)保留旧能力演进；当前状态以本页和本轮实际记录为准。

Newton 1.7 官方入口：[求解器矩阵](https://newton-physics.github.io/newton/1.7.0/solvers/index.html)、[碰撞](https://newton-physics.github.io/newton/1.7.0/concepts/collisions.html)、[多世界](https://newton-physics.github.io/newton/1.7.0/concepts/worlds.html)、[传感器](https://newton-physics.github.io/newton/1.7.0/concepts/sensors.html)、[耦合](https://newton-physics.github.io/newton/1.7.0/concepts/coupling.html)、[变形与杆](https://newton-physics.github.io/newton/1.7.0/concepts/deformable_objects.html)。以上是架构/功能参照，不冒充当前机器上的逐项 Newton 性能测量。
