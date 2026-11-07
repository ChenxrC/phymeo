# Phymeo / Newton 功能差异全景（更新至2026-11-08 C9接受）

[打开离线全景图](phymeo-newton-panorama-2026-11-07.html) · [C9浮根设备树计划](../superpowers/plans/2026-11-08-resident-floating-tree.md)

**当前最大差距是设备链路的通用组合、持续工作流和后续物理族。** 已完成能力不能继续列为缺项：HOST浮根mimic、fixed工具/根惯量、通用非球双侧响应、转动逐面设备候选、固定根普通/affine设备树及同树双动态接触都已分别验收。

本页合并原多次追加的过时状态；历史原文在Git中保留。Phy新增C9普通浮根设备运动与宽相正式接受，产品4927ef9、同次保存审计495；C10浮根接触进入父集成，尚未验收。Newton固定为本机已验证的1.7.0，不称最新版本。不按测试数/清单数给整体完成百分比，也没有总体性能领先结论。

| 优先级 / 模块 | Phymeo已验范围 | 确实剩余的差距 | Newton1.7参照 | 后续节点 |
|---|---|---|---|---|
| 01 场景与机器人 | URDF / MJCF / USD、PD / IK；HOST 普通/浮根/mimic共同场景、fixed工具折叠及非对角根惯量；通用非球双方反力。 | **已有，继续扩域**：更广关节与闭环、浮根pads、自动材料站点迁移；抓取释放的工程精度边界仍留出。 | 统一模型和资产；具体关节、执行器能力依求解器。 | C10接入真实root/child接触载荷和共同发布。 |
| 02 碰撞检测与响应 | RR / RD / DD、S/C/B/H整族patch；有限面边、凹闭体、部分CCD；转动mesh–round采样已走TYX逐面BVH。 | **已有，覆盖不完整**：更广连续运动、变形/凹体组合；全设备窄相、稳定跨特征材料身份和密集持续摩擦。 | GPU凸体GJK/MPR、网格BVH/SDF；推测接触不等于首次碰撞时间。 | 维持形状×表示×运动×所有者矩阵；扩域逐层验收。 |
| 03 GPU与多场景 | TYX普通/affine固定根树及普通浮根完整运动（0–16关节），接受态FK→AABB→BVH；固定根接触反馈与多世界原子发布。 | **普通浮根运动已通**：浮根接触/affine、同world独立participants联合设备求解、设备窄相与吞吐仍缺。 | Warp模型/状态与多世界；异构域和性能依后端与配置。 | C10浮根接触→浮根affine；再测同条件端到端性能。 |
| 04 可微仿真与训练 | 解析Jacobian、TYX底层、多世界状态/选择复位。 | **缺端到端工作流**：公开rollout反传、参数辨识与梯度验收；动作/观测/终止/随机化的向量环境。 | 部分后端基础可微；训练生态可接Isaac Lab。 | 无接触梯度闭环＋W1/W32向量环境。 |
| 05 传感与交互 | 米制深度/法线/ID、TYX射线、IMU、在线Python控制、视觉闭环与Three.js真实回放。 | **基础可用**：RGB/材质/光照、更广相机模型；场景实时编辑、批量传感与生态接口。 | 接触/IMU/图像传感、多个Viewer与镜头模型。 | 同一真实场景的传感器和交互接口。 |
| 06 跨域求解编排 | 事务、点端口、域图与指定刚–布–软体共同推进。 | **缺通用组合层**：同模型按区域选后端、公共耦合调度、界面力/约束及失败处理。 | Proxy / ADMM等实验性耦合，有接口边界。 | 两个后端共享模型，真实界面力与共同回滚。 |
| 07 布与体软体 | 布拉伸/剪切/弯曲、自接触与双向载荷；StVK四面体；参考布递交、单爪软布。 | **已有，稳健性待补**：可变形持续静摩擦、密集折叠/换特征；目标软布两初态仍未放台，软块抓捏覆盖不足。 | cloth / tet依后端；VBD有自碰撞，XPBD布无自碰撞。 | 多网格/刚度的持载、折叠、软块抓捏。 |
| 08 杆缆、MPM / 流体 | 刚性连杆、自由粒子与弹性体基础。 | **缺完整物理族**：杆截面取向、弯扭本构与接触；MPM粒子–网格、塑性/流动及双向边界。 | rod / cable、ImplicitMPM等独立求解器路线。 | 分别立项：扭转绕柱、倒砂/溃坝。 |

## 目前这条链路走到哪里

~~~text
固定根普通 / affine 接受态
          ↓ TYX FK
     TYX AABB → 设备 BVH
          ↓ 候选读回
 HOST 有限几何 / patch / 法向律
          ↓ 双侧 COM wrench
 TYX 全 N 力学 / K 投影 / 有界 PD
          ↓
    唯一接受发布 / 拒步保留
~~~

这条链路已覆盖同一固定根完整tree/forest中的非相邻双动态group、原静态pair和多world事务。当前仍mu=0、离散入步采样，几何与法向律在HOST。**HOST共同场景已有持续摩擦和浮根，不等于驻留链路已支持它们。**

C8联合覆盖每backend74项、112原保存payload、Newton160实际原生步与新wheel两backend各251接受+1拒绝world-step。CPU原后置力读数误标已用同一80步解码补件闭环，原失败保留。参照范围是两个具体动态原生案例，不能外推Newton全功能或性能。

## 当前开发与后继次序

1. **C10普通浮根接触反馈**：C9真实root COM/完整M/b/J、内部反冲、有界PD、设备FK及宽相已验收。现在沿同一入口装配root/child完整载荷，验证有限静态及同树双动态接触、两bank事务与源码外调用。
2. **浮根affine/mimic**：后继方案已调研，待C10正式接受后冻结完整/归约自由度、原约束反力与驱动，复用既有求解和事务。提案不等于实施或通过。
3. **剩余碰撞组合和设备瓶颈**：持续材料换接/更广CCD与设备窄相分别验收；性能采用同物理、同输出、同容差、同硬件对照。
4. 依用户八大顺序推进训练、传感、通用编排与后续物理族。抓取释放的小量能量门失败留证，不阻塞独立功能；目标软布两初态仍须完成真实放台。

## 如何判断进展或停滞

- 每节点交可调用公共入口，至少多个独立物理case与负控/事务；实现、实际运行、保存证据和源码外安装包分别记录。
- 已绿同源不重复；先保原失败，明确输入/表示/观察还是实现缺陷，再做最小修复。调质量、改案例或仅换观察标签不能冒物理改进。
- 若连续两轮只增加理论/统计而无新的公开能力、独立反例闭环或可验修复，应停止该细枝，回到接口/数值模型和用户功能目标裁定。
- 工程误差按既定量纲容差；不追求Newton对应后端也不承诺的精确接触/能量恒等。未知/不支持明示，Panda递布必要但不充分。

## 证据与边界

- [C8正式接受](../benchmarks/p65-resident-tree-dynamic-contacts.md)
- [C9正式限域接受](../benchmarks/p65-resident-floating-tree.md) · [原计划](../superpowers/plans/2026-11-08-resident-floating-tree.md)
- [C10开发与多案例验收计划](../superpowers/plans/2026-11-08-resident-floating-contacts.md)
- [浮根mimic HOST场景](../benchmarks/p65-floating-mimic-scene.md)
- [浮根惯量与工具](../benchmarks/p65-floating-root-inertia-frame.md)
- [通用共同场景摩擦](../benchmarks/p65-general-scene-friction.md)
- [抓取工程精度边界](../benchmarks/p65-tangent-application-reference.md)
- [持续实施入口](p65-continuation.md)

Newton参照沿已调研固定1.7文档与本机源码：[求解器矩阵](https://newton-physics.github.io/newton/1.7.0/solvers/index.html)、[碰撞](https://newton-physics.github.io/newton/1.7.0/concepts/collisions.html)、[多世界](https://newton-physics.github.io/newton/1.7.0/concepts/worlds.html)、[传感器](https://newton-physics.github.io/newton/1.7.0/concepts/sensors.html)、[耦合](https://newton-physics.github.io/newton/1.7.0/concepts/coupling.html)、[可变形/杆](https://newton-physics.github.io/newton/1.7.0/concepts/deformable_objects.html)。Newton也按后端/配置分域；不照搬其依赖。测试工具开源继续列后续优化。持续目标ACTIVE、无08:00限制、无周期任务。
