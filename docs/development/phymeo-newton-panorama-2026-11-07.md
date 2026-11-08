# Phymeo / Newton 功能差异全景（更新至2026-11-08保存点）

[打开离线全景图](phymeo-newton-panorama-2026-11-07.html) · [C11浮根affine接受](../benchmarks/p65-resident-floating-affine.md) · [K0保存点与恢复入口](../superpowers/plans/2026-11-08-device-convex-query.md#11-用户清除长期目标后的保存点2026-11-08-1124)

**当前任务状态：已保存暂停。** 长期目标已清除，原作者均释放。本次仅更新全景说明，不恢复实现、测试或周期任务。证据基线为父仓库 `08dc557`；最近正式功能验收为C11。

**当前最大差距是设备链路的通用组合、持续工作流和后续物理族。** 已完成能力不能继续列为缺项：HOST浮根mimic、fixed工具/根惯量、通用非球双侧响应、转动逐面设备候选、固定根普通/affine设备树及同树双动态接触都已分别验收。

本页合并过时状态，历史保留在Git。C10普通浮根接触已正式限域接受；541保存审计带明确URDF来源限定，不能称全部输入有当次raw快照。C11浮根affine已完成核心、原生、源码外及同次保存验收；K0设备凸体查询已有隔离工作树中的产品与测试草稿，尚未完成最终审阅、编译、集成和首个设备查询正控。Newton固定为本机已验证的1.7.0，不称最新版本。不按测试数/清单数给整体完成百分比，也没有总体性能领先结论。

## 本次新增进展应如何计入

| 保存点 | 已经发生 | 在功能全景中的归类 |
|---|---|---|
| 共享来源与公开薄接口 | 父已保存同次设备状态借用、注册来源、资源关闭与查询入口；interpret/CUDA相邻兼容各2项通过。 | 接入准备有验证；不等于新查询已可用。 |
| K0产品草稿 | RD `21ac993` 保存schema、support、GJK/EPA设备核与报告，工作树clean。最终草稿未导入父。 | 代码已保存、未验证；不能列为GPU窄相已支持。 |
| K0测试与独立参考 | robot `c3f9de5` 保存142个预声明身份；核心140，W32后置2。有效RED在缺公开入口处失败，原记录保留。 | 测试库草稿；142是待验案例数，不是通过数或完成率。 |
| 当前待解决 | 统计记录结构对齐、EPA等深面判定、TYX编译与分组正控；最终oracle/source/protocol审阅。 | 恢复后的首个工作节点，不继续扩案例掩盖未通入口。 |

## 八大模块差异

| 优先级 / 模块 | Phymeo已验范围 | 确实剩余的差距 | Newton1.7参照 | 后续节点 |
|---|---|---|---|---|
| 01 场景与机器人 | URDF / MJCF / USD、PD / IK；HOST 普通/浮根/mimic共同场景、fixed工具折叠及非对角根惯量；通用非球双方反力。 | **已有，继续扩域**：更广关节与闭环、浮根pads、自动材料站点迁移；抓取释放的工程精度边界仍留出。 | 统一模型和资产；具体关节、执行器能力依求解器。 | 普通/affine固定根与浮根共同发布已通；继续补更广机器人工作流。 |
| 02 碰撞检测与响应 | RR / RD / DD、S/C/B/H整族patch；有限面边、凹闭体、部分CCD；转动mesh–round采样已走TYX逐面BVH。 | **已有，覆盖不完整**：更广连续运动、变形/凹体组合；全设备窄相、稳定跨特征材料身份和密集持续摩擦。 | GPU凸体GJK/MPR、网格BVH/SDF；推测接触不等于首次碰撞时间。 | 维持形状×表示×运动×所有者矩阵；扩域逐层验收。 |
| 03 GPU与多场景 | TYX普通/affine固定根与浮根完整树的运动、宽相、有限接触反馈、原图反力；多世界原子发布。 | **浮根affine接触已通**：同world独立participants联合设备求解、设备窄相与吞吐仍缺。 | Warp模型/状态与多世界；异构域和性能依后端与配置。 | K0设备距离与双侧见证（草稿）→K1设备完整patch→K2真实受载消费；再测同条件性能。 |
| 04 可微仿真与训练 | 解析Jacobian、TYX底层、多世界状态/选择复位。 | **缺端到端工作流**：公开rollout反传、参数辨识与梯度验收；动作/观测/终止/随机化的向量环境。 | 部分后端基础可微；训练生态可接Isaac Lab。 | 无接触梯度闭环＋W1/W32向量环境。 |
| 05 传感与交互 | 米制深度/法线/ID、TYX射线、IMU、在线Python控制、视觉闭环与Three.js真实回放。 | **基础可用**：RGB/材质/光照、更广相机模型；场景实时编辑、批量传感与生态接口。 | 接触/IMU/图像传感、多个Viewer与镜头模型。 | 同一真实场景的传感器和交互接口。 |
| 06 跨域求解编排 | 事务、点端口、域图与指定刚–布–软体共同推进。 | **缺通用组合层**：同模型按区域选后端、公共耦合调度、界面力/约束及失败处理。 | Proxy / ADMM等实验性耦合，有接口边界。 | 两个后端共享模型，真实界面力与共同回滚。 |
| 07 布与体软体 | 布拉伸/剪切/弯曲、自接触与双向载荷；StVK四面体；参考布递交、单爪软布。 | **已有，稳健性待补**：可变形持续静摩擦、密集折叠/换特征；目标软布两初态仍未放台，软块抓捏覆盖不足。 | cloth / tet依后端；VBD有自碰撞，XPBD布无自碰撞。 | 多网格/刚度的持载、折叠、软块抓捏。 |
| 08 杆缆、MPM / 流体 | 刚性连杆、自由粒子与弹性体基础。 | **缺完整物理族**：杆截面取向、弯扭本构与接触；MPM粒子–网格、塑性/流动及双向边界。 | rod / cable、ImplicitMPM等独立求解器路线。 | 分别立项：扭转绕柱、倒砂/溃坝。 |

## 目前这条链路走到哪里

~~~text
普通 / affine 固定根与浮根接受态
          ↓ TYX FK
     TYX AABB → 设备 BVH
          ↓ 候选读回
 HOST 有限几何 / patch / 法向律
          ↓ 双侧 COM wrench
 TYX 全 N 力学 / K 投影 / 有界 PD
          ↓
    唯一接受发布 / 拒步保留
~~~

这条链路已覆盖固定根完整tree/forest与普通/affine浮根中的非相邻双动态group、root/child静态载荷和多world事务。当前仍mu=0、离散入步采样，几何与法向律在HOST。**HOST共同场景已有持续摩擦；驻留链路浮根affine现已验收，持续摩擦仍需后续验证。**

C10每backend76唯一节点、133完整保存gzip（含原失败前缀）、Newton160原生步和新wheel每backend302接受/2拒绝world-step完成。F06与source_revision的URDF缺少原当次字节快照，以原Git身份和完整保存语义作限定追溯，原缺口与失败保留。不能外推Newton全功能或性能。

## 恢复后的推进次序（尚未启动）

1. **C10已完成限定验收**：真实root/child完整载荷、有限静态及同树双动态接触、两bank事务与源码外调用已验证；来源限定见正式报告，不再重跑已绿组。
2. **C11浮根affine/mimic已接受**：两backend各52核心节点、相邻6及W32；原生每端9步、新wheel每端266接受/2预期拒绝world-step完成。587保存审计核102完整输出与3原失败产物，435产品成员字节同源；不重复已绿组。
3. **K0整族设备查询先闭环**：[案例清单](p65-device-convex-query-case-manifest.md)覆盖十种S/C/OBB/H组合、有限特征/协变/近零/包含/资源。先按[保存点](../superpowers/plans/2026-11-08-device-convex-query.md)核最终增量与接口，再按4首正控→其余16非零→有限特征/协变→边界/事务→W32推进解释与CUDA验证，最后做原生参照、源码外包和保存审计。通过后才进入K1设备完整patch、K2真实受载消费；当前草稿不计功能通过。
4. 前三项继续补同world独立participants的设备共同求解、持续材料换接与更广CCD；吞吐以同物理、同输出、同容差、同硬件对照，不将K0单点查询与完整patch求解直接比速。
5. 依用户八大顺序推进训练、传感、通用编排与后续物理族。抓取释放的小量能量门失败留证，不阻塞独立功能；目标软布两初态仍须完成真实放台。

## 如何判断进展或停滞

- 每节点交可调用公共入口，至少多个独立物理case与负控/事务；实现、实际运行、保存证据和源码外安装包分别记录。
- 已绿同源不重复；先保原失败，明确输入/表示/观察还是实现缺陷，再做最小修复。调质量、改案例或仅换观察标签不能冒物理改进。
- 若连续两轮只增加理论/统计而无新的公开能力、独立反例闭环或可验修复，应停止该细枝，回到接口/数值模型和用户功能目标裁定。
- 工程误差按既定量纲容差；不追求Newton对应后端也不承诺的精确接触/能量恒等。未知/不支持明示，Panda递布必要但不充分。

## 证据与边界

- [C8正式接受](../benchmarks/p65-resident-tree-dynamic-contacts.md)
- [C9正式限域接受](../benchmarks/p65-resident-floating-tree.md) · [原计划](../superpowers/plans/2026-11-08-resident-floating-tree.md)
- [C10正式接受与来源限定](../benchmarks/p65-resident-floating-contacts.md) · [C11正式接受](../benchmarks/p65-resident-floating-affine.md) · [C11开发计划](../superpowers/plans/2026-11-08-resident-floating-affine.md)
- [浮根mimic HOST场景](../benchmarks/p65-floating-mimic-scene.md)
- [浮根惯量与工具](../benchmarks/p65-floating-root-inertia-frame.md)
- [通用共同场景摩擦](../benchmarks/p65-general-scene-friction.md)
- [抓取工程精度边界](../benchmarks/p65-tangent-application-reference.md)
- [K0共享接缝兼容记录](../../artifacts/p65-generic-collision-20261107-01/k0-parent-adjacent-01.json) · [执行前142身份清单](../../artifacts/p65-generic-collision-20261107-01/k0-predeclared-execution-membership-v1.json)
- [当前保存与续作入口](p65-continuation.md)

Newton参照沿已调研固定1.7文档与本机源码：[求解器矩阵](https://newton-physics.github.io/newton/1.7.0/solvers/index.html)、[碰撞](https://newton-physics.github.io/newton/1.7.0/concepts/collisions.html)、[多世界](https://newton-physics.github.io/newton/1.7.0/concepts/worlds.html)、[传感器](https://newton-physics.github.io/newton/1.7.0/concepts/sensors.html)、[耦合](https://newton-physics.github.io/newton/1.7.0/concepts/coupling.html)、[可变形/杆](https://newton-physics.github.io/newton/1.7.0/concepts/deformable_objects.html)。Newton也按后端/配置分域；不照搬其依赖。测试工具开源继续列后续优化。2026-11-08重新核对上述固定版的求解器矩阵、碰撞和多世界官方页面；保持后端/配置限定，不把页面所述所有能力冒成本机逐项实测。当前没有活动长期目标或周期任务。
