# Newton 功能对齐：07:20 之后的下一批

状态：计划已具体化，尚未执行本页的新产品开发。当前产品基线为 **1ce57489** 的已验最终包；本页不把研究/案例库当作功能通过。总入口为[当前全景](../../development/phymeo-newton-panorama-2026-11-12.md)与[当前续作](../../development/p65-continuation.md)。

用户顺序保持：场景与机器人 → 碰撞检测 → GPU/多场景；训练保持已提升的优先级，但缺乏新证据时停止盲目优化。以下以通用模块和多案例验收推进，不以单个Panda/G1/递布案例替代模块资格。

## 第一批可并行作者工作

| Lane / 难度 | 可交付模块 | 复用 / 需补齐 | 验收与停止 |
| --- | --- | --- | --- |
| 原机器人作者 / 中高 | **BALL-M1：通用 native JOINT moment 表**，首先资格scalar与BALL直接NONE电机 | 复用完整M/J/b、四元数积分、runtime事务；补完整1列/3列传动、统一scalar-force→广义力scatter与actual设备收据 | [十组案例与原门](../../development/p65-robot-next-compositions-20261112.md)。同入步原生oracle、独立连续轨迹、力/功率、反向卸载、失败恢复和安装调用。新域不解禁FREE/contact/activation；旧BALL-A/C仍保原资格 |
| 原碰撞作者 / 高 | **M1：刚性设备来源 → 有限mesh边界查询** | 复用四resident树的accepted FK/anchor/lease和COL2核；补device gather、bounds、BVH设备buffer入口、candidate pack | [来源合同](../../development/p65-device-mesh-source-next-contract.md)。四来源/真实物理连续步、双侧移动、有限面与共享facet、world/容量/晚失败。当前坐标下载到HOST再pack即不满足目标；M2/CCD/载荷另列 |
| 原训练作者释放后 / 中高 | **显式多participant场景装配**，连接SceneDeclaration层与已验ROB1公共API | 复用R01/R02/R04与四storage；补logical-world内participant身份、资产/控制/collider所有权lowering。不得复制积分器/私有bank | 先冻结显式新声明版本和错误语义；双刚体、臂推盒、四storage混合、W2、恢复/partial reset/晚失败。R03旧失败保持，不靠装配改名关闭 |
| 父 / 串行 | 共享接缝、独立审阅、执行与最终包 | 原native compiler/kernel/wrapper和resident runtime/BVH共享接缝由父独占；各author只改冻结范围 | 任一lane tests-ready后父排唯一执行槽。所有静态/import/数值/pytest/build/install/profile串行；作者继续纯编码其他无依赖片 |

同一时刻最多原三名作者；不另建助手。若训练诊断尚未交接，第三lane先收束诊断，再接场景装配，不能让两个任务同时占有同一旧接缝。

## 训练与规模：先补证据，再选择一个改动

本批[有界信用分配诊断](../../development/p65-jr-learning-next-diagnosis-20261112.md)只针对J14011/R14211/S14111原final，各32步W8和一次upstream backward，第一次Adam前停止。它最多768世界转移、0更新；不重跑128次学习，不修改目标/门/种子集合，不把N7连接当学会。

诊断若完成，严格按其A–E顺序决定下一工作：绑定失败先修观察；无mean credit查实际路径；critic主导只另登记一次校准研究；clip活跃先量化而不直接调阈值；无区分原因就停止这个采样，不追加rollout。S保原成功任务对照，R未证可达，不能套用J的控制结论。

规模优化从原N7实际分项成本选一个开销节点。必须先固定模型、精度、输出、reset/观察协议再计时；现有full/integration和旧进程commit观察不等于Newton等输出性能优势。不得通过少保存字段、缩小arena、去掉检查或降低world数制造“优化”。

## 后继依赖

1. BALL-M1通用moment表通过后，沿同一表资格FILTEREXACT，再组合已验BALL-C接触。FREE+BALL另需多宽度地址/祖先/bias/双四元数积分ABI，不能只加路由。
2. M1刚体来源通过后，M2先建设真实deformable accepted/candidate设备顶点bank和lease；其后再接同源query。连续碰撞、material J/速度/完整M、接触力与摩擦历史均独立验收。
3. 机器人装配与上述后端能力分别标域；native编译成功不等于asset lowering/任意runtime/完整轨迹都支持。
4. 按后续用户顺序补传感/交互、通用跨域、布/体软体、杆缆及MPM/流体，不因这批机器人扩展勾掉它们。

## 每个节点的交付包

- 短合同：公共API、能力边界、现有源码依据、固定多案例与独立oracle。
- tests-first源码及小提交，保完整首失败，不改fixture或门追绿。
- 两后端真实执行、受影响兼容与同次完整状态/力/运动/失败收据；源码外公共消费者。
- 当前全景和一页续作更新；已通过范围复用，不全量重跑旧绿套。

**成功**：通用API在不同模型/来源/world/连续状态复用，实际力与运动及失败语义有独立证据。
**无效循环**：连续两次有针对性尝试没有缩小障碍，或靠放宽阈值、增加预算、挑输入掩盖反例；保最小反例、停止该片，推进独立lane。
整个P6.5/Newton功能对齐仍未完成；无remote CI、push/mainmerge、发包或P7/P8扩域。
