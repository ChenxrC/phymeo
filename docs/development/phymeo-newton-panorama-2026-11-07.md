# Phymeo / Newton 功能差异全景（2026-11-10：机器人连接与训练闭环）

[当前已实现连接图](../architecture/robot-environment-connections-2026-11-10.html) · [本批实际结果](../benchmarks/p65-robot-environment-connections-20261110.md) · [11月9日历史全景图](phymeo-newton-panorama-2026-11-07.html) · [C11浮根affine接受](../benchmarks/p65-resident-floating-affine.md)

**11月10日实质变化：** 原HOST G1完成官方全131.48秒无跌倒/拒步，原峰值力与参考XY两工程门仍未过；新公共工具入口另验20秒，全部原工程门通过。受载Env/逐世界拒步/PolicyRunner已通；可选CPU RSL-RL两真实物理任务学习和checkpoint重载已通；W256混合运动已通，受载W256仍被512>256shape注册限制挡住。33body旧门已修，速度摩擦混合桥HOST63及两设备各12通过，见[限域结果](../benchmarks/p65-resident-friction-connections.md)。不能再把动作/观测/终止/复位或PPO接口列为完全未开发；设备数据面和规模仍有实质缺口。

**当前状态：K0默认设备查询已限域验收并提交 b3f4cf3。** 最终两端各142项通过、284完整记录留存，新wheel两端各4接受、440产品成员同源。用户已将大规模训练与G1舞蹈提升为K0后的最高优先级；见[新主线计划](../superpowers/plans/2026-11-09-training-humanoid-priority.md)。

**当前最大差距是设备链路的通用组合、持续工作流和后续物理族。** 已完成能力不能继续列为缺项：HOST浮根mimic、fixed工具/根惯量、通用非球双侧响应、转动逐面设备候选、固定根普通/affine设备树及同树双动态接触都已分别验收。

本页以实际验收区分能力。C10保留原URDF来源限定，C11完整浮根affine已接受，K0现已限域接受。驻留受载链的完整几何与速度摩擦律仍在HOST；K0查询不代表完整设备接触求解。Newton固定参照1.7.0，不给总体完成百分比或总体性能领先结论。

## K0最新验收与优先级

[正式验收](../benchmarks/p65-device-convex-query-acceptance.md)：默认距离/见证通过，K1可选patch作者内容未验收，K2受载消费未冻结。十touch每端9解出contact-band、1法向不唯一UNKNOWN。旧缺失明细的11月8日记录保留在原专项计划§12，本轮补足见§13。

**下一主线：完整G1设备受载组合 → 站点力设备化/公开策略张量端口 → 分页单次发布受载W256 → W256/W1024吞吐实测。** CPU舞蹈/PPO与公共环境基础已经完成，复用其证据。训练列原序号04但当前优先级已提升；其余大模块清单仍作为覆盖台账。

## 八大模块差异

| 优先级 / 模块 | Phymeo已验范围 | 确实剩余的差距 | Newton1.7参照 | 后续节点 |
|---|---|---|---|---|
| 01 场景与机器人 | URDF / MJCF / USD、PD / IK；HOST 普通/浮根/mimic共同场景、fixed工具折叠及非对角根惯量；通用非球双方反力。 | **已有，继续扩域**：更广关节与闭环、浮根pads、自动材料站点迁移；抓取释放的工程精度边界仍留出。 | 统一模型和资产；具体关节、执行器能力依求解器。 | 普通/affine固定根与浮根共同发布已通；继续补更广机器人工作流。 |
| 02 碰撞检测与响应 | RR / RD / DD、S/C/B/H整族patch；有限面边、凹闭体、部分CCD；转动mesh–round采样已走TYX逐面BVH。 | **已有，覆盖不完整**：更广连续运动、变形/凹体组合；全设备窄相、稳定跨特征材料身份和密集持续摩擦。 | GPU凸体GJK/MPR、网格BVH/SDF；推测接触不等于首次碰撞时间。 | 维持形状×表示×运动×所有者矩阵；扩域逐层验收。 |
| 03 GPU与多场景 | TYX普通/affine固定根与浮根完整树、原图反力/共同发布；32joint容量、29/32joint最后body受载；普通fixed/floating速度摩擦混合桥；W256混合运动。K0当前位姿查询每端142身份已验。 | **通用组合和数据面未完**：同world独立participants联合设备求解、设备完整patch/力核、受载W256分页、无HOST往返策略张量及吞吐。K1原作者WIP未验，不覆盖K2。 | Warp模型/状态与多世界；异构域和性能依后端与配置。 | 保完整输出/原预算，先单次发布分页与公开张量契约，再规模性能。 |
| 04 可微仿真与训练 | 解析Jacobian/TYX底层；真实受载向量环境、完整接受mask、final observation/选择reset、动作保持/时钟；可选RSL-RL和两HOST PPO任务/重载已过。G1原ONNX反馈全动作HOST完成。 | **最高优先主线已有闭环**：缺设备端到端训练数据面、物理域随机化/批量任务、RNN与精确中途恢复；公开rollout反传、参数辨识与梯度验收仍缺。CPU PPO不等于可微物理。 | 部分后端基础可微；训练生态可接Isaac Lab。 | 设备状态/策略张量与受载规模；可微路线另设独立多步梯度任务。 |
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
 HOST 有限几何 / patch / 速度摩擦律
          ↓ 双侧 COM wrench
 TYX 全 N 力学 / K 投影 / 有界 PD
          ↓
    唯一接受发布 / 拒步保留
~~~

这条链路的μ0历史验收覆盖固定根tree/forest与普通/affine浮根的非相邻双动态group、root/child静态载荷和多world事务。11月10日追加普通fixed/floating正μ速度摩擦两机制留出；affine和同树双动态的正μ新窗口尚未验。仍为离散入步采样，几何/力律在HOST。K0设备距离查询不替换完整HOST接触斑。材料锚点静摩擦/滚动历史与速度摩擦是不同能力。

C10每backend76唯一节点、133完整保存gzip（含原失败前缀）、Newton160原生步和新wheel每backend302接受/2拒绝world-step完成。F06与source_revision的URDF缺少原当次字节快照，以原Git身份和完整保存语义作限定追溯，原缺口与失败保留。不能外推Newton全功能或性能。

## 随后的次序

1. **C10已完成限定验收**：真实root/child完整载荷、有限静态及同树双动态接触、两bank事务与源码外调用已验证；来源限定见正式报告，不再重跑已绿组。
2. **C11浮根affine/mimic已接受**：两backend各52核心节点、相邻6及W32；原生每端9步、新wheel每端266接受/2预期拒绝world-step完成。587保存审计核102完整输出与3原失败产物，435产品成员字节同源；不重复已绿组。
3. **K0默认查询已验收**：b3f4cf3；两端各142通过，284份完整明细保留，源码外wheel与440产品成员同源。见[验收报告](../benchmarks/p65-device-convex-query-acceptance.md)。
4. **训练与人形提升为最高优先主线**：先T0向量接口与H0 G1资产/策略；再29关节设备容量、必要足地接触、真实舞蹈和PPO闭环。K1已有可选作者代码但未验收，按人形与通用场景实际依赖推进。详见[新计划](../superpowers/plans/2026-11-09-training-humanoid-priority.md)。
5. 前三项继续补同world独立participants的设备共同求解、持续材料换接与更广CCD；吞吐以同物理、同输出、同容差、同硬件对照。
6. 训练闭环与规模优先，传感、通用编排和其它物理族随后；原抓取释放与软布失败继续保留，不能以人形一个案例替代多案例验收。

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
- [K0 Newton短参照](../../artifacts/p65-generic-collision-20261107-01/k0-closeout/newton-short.json) · [安装包解释器记录](../../artifacts/p65-generic-collision-20261107-01/k0-closeout/wheel-interpret.json.gz) · [安装包CUDA记录](../../artifacts/p65-generic-collision-20261107-01/k0-closeout/wheel-cuda.json.gz) · [只读审计](../../artifacts/p65-generic-collision-20261107-01/k0-closeout/audit.json)
- [K1 冻结范围](../superpowers/plans/2026-11-08-device-convex-patch.md)
- [当前保存与续作入口](p65-continuation.md)

Newton参照沿已调研固定1.7文档与本机源码：[求解器矩阵](https://newton-physics.github.io/newton/1.7.0/solvers/index.html)、[碰撞](https://newton-physics.github.io/newton/1.7.0/concepts/collisions.html)、[多世界](https://newton-physics.github.io/newton/1.7.0/concepts/worlds.html)、[传感器](https://newton-physics.github.io/newton/1.7.0/concepts/sensors.html)、[耦合](https://newton-physics.github.io/newton/1.7.0/concepts/coupling.html)、[可变形/杆](https://newton-physics.github.io/newton/1.7.0/concepts/deformable_objects.html)。Newton也按后端/配置分域；不照搬其依赖。测试工具开源继续列后续优化。2026-11-08重新核对上述固定版的求解器矩阵、碰撞和多世界官方页面；保持后端/配置限定，不把页面所述所有能力冒成本机逐项实测。当前没有活动长期目标或周期任务。
