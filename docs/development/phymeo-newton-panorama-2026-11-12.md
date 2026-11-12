# Phymeo / Newton 功能差异全景（2026-11-13更新）

本页更新至**2026-11-13 D5-A / R3-N/P / G2a / D6-A有限资格**。当前产品ec0e2997，第二包532产品成员同源，指定D5、R3-P和D6消费者已经结束；首包113629b3的D5、R3-N和G2a资格保留。[当前功能矩阵](p65-feature-acceptance-matrix-20261113.md)分别记录源码、后端、指定安装、完整应用和精度结果；不把历史通过当新包全量回归。

**D4原无阻尼N1两后端3秒稳态仍FAIL；D6-A包外是FUNCTIONAL_PASS并保留strict_elastic_precision=FAIL_RECORDED。** 新c4096持续载荷应用D6-B的run143已完成N1 interpret原3072步/3秒，仍为DIAGNOSTIC_ONLY / FAIL；停止该系数CUDA和其余案例，不扫参数。没有从耗散算子或短窗通过推导持续支撑。最新[架构图](p65-feature-architecture-draft-20261113.md)和[统一编译架构](../architecture/unified-scene-compilation.md)区分三条runtime；旧07:20 HTML与Batch A/B/C报告仍是各自版本的历史证据。

比较对象固定为 **Newton 1.7.0** 官方文档及本地已固定源码，不声称调查了后续最新版本。这里按公共能力、适用范围和可复用案例判断差距，不按测试数或清单数计算总体完成百分比。

## 总体判断

Phymeo已连接模型编译、受限机器人动力学、设备接触与约束、批量环境、真实CUDA PPO和恢复；公共声明已能把真实变形节点与单ordinary运动球/胶囊装配到共同响应，并可声明轴向材料耗散。当前主要差距是**更广组合及完整应用**：多body/关节刚柔、持续稳定支撑、摩擦换面/CCD、FREE+BALL的其它执行器/约束、复杂机器人训练与批量传感；可微物理、通用跨域、密集软体和杆缆/MPM/流体仍有独立模块缺口。

此前已解决四个圆柱Y查询法向失败，增加BALL1/A/C、COL2/COL3和紧凑native镜像，随后连接JOINT moment表、显式多参与者装配和真实accepted设备FK。上一批将直接球电机接入原normal/pyramidal接触，并让四类原资产来源进入设备边界查询；本批进一步连通逻辑多参与者来源、持久真实变形状态和direct BALL执行器动态。随机重置、D步窗口与PPO的原N7组合资格保持。本批又完成S3公共编译接线、D2真实变形查询和FREE＋BALL基础格。Batch C连接D3法向载荷与原模型异构分桶后，D5补公共刚柔装配，R3-N/P补FREE+BALL法向/摩擦接触，G2a完成原8模型16world域的实际规模格，D6-A补可选轴向材料耗散。曲面触点表示、J/R学习门、G1人形长轨迹及持续稳态载荷/更广组合缺口仍开放。

## 求解与查询路径分别看

| 路径 | 已连接的主要流程 | 当前边界 |
| --- | --- | --- |
| Phymeo resident | 程序/资产模型 → ordinary、affine、floating、floating-affine 树 → TYX FK/宽相/设备凸体查询 → 默认 HOST patch 或限定 S/C/OBB/H 设备 patch → 原材料点、完整 J/M、双方力与 COM 力矩 → 共同发布 | 已验受载分页和混合参与者；材料点转换、载荷装配与诊断仍有 HOST 工作。曲面查询的新能力不会自动成为曲面 patch 或载荷能力 |
| MuJoCo-TYX | 统一模型入口 → CPU MuJoCo 3.3.7 编译模型 → 明确的数据映射 → TYX 树动力学、primitive 接触和混合约束 → Runtime/VecEnv → CUDA RSL PPO | CPU 编译不代替设备物理步。CUDA 默认 MuJoCo-TYX，显式 MuJoCo-Warp 是另一实现；各关节/执行器/接触组合有独立准入，不能把固定球树和自由根接触域任意拼接 |
| 公共 CollisionWorkspace 边界查询 | 完整 HOST 当前快照 → 原过滤/注册和 TYX BVH 候选 → COL2 有限线段–三角距离 → typed aggregate 与事务发布 | 新 COL3 是独立查询接口；不是 resident 设备顶点借用、实心内外判定、CCD 或接触力通道。此公共接口也不会自动扩展 MuJoCo-TYX primitive 求解器 |
| resident M1 / 场景边界查询 | 四类direct/from_asset刚性真实accepted设备FK、逻辑world/participant及anchor转换 → gather/bounds/BVH → 有限边界核 → typed发布 | S3公共自动装配已通；该通用查询本身仍不是载荷、CCD或实体内外判定。D3/D5另有有限刚柔响应，不能反向授予M1任意响应 |
| D1/M2-P与D6-A | 持久设备节点integrate → ordered elastic → reconstruct → 可选一次ordered轴向耗散 → 验证/共同发布 → loan/reset/restore | 普通步无HOST当前坐标回填；None/零保原路径。耗散是显式参数的一阶split，有轴向零模，不是所有振型衰减或全布/体软体迁移 |
| D5公共装配→D3共同响应 | ParticipantSceneDeclaration＋ParticipantDeformable → 唯一共同owner → 同入步节点/运动球胶囊查询与NormalLayer → 节点材料/可选D6、刚体候选 → 双方力/COM矩/固定反力和运动共同发布 | 每world一个节点模型＋ordinary单body浮体，mu0、entry-sampled；载荷和诊断有HOST往返。多刚体/关节刚柔、持续稳态、摩擦/历史和CCD仍缺 |
| G1/G2a异构原生world | 完整原生来源/cap身份 → 每桶原MuJoCo-TYX runtime → 全world控制/力/运动 → 聚合step/reset/restore | 上限8模型16world；W1/W8/W16源矩阵、指定W16B3/B8安装已过。独立world、非单异构kernel或跨桶碰撞；不包含D5节点模型，也无Newton性能结论 |

## 历史增量、当前资格与保留失败

| 节点 | 当前实际状态 | 不应据此推导的能力 |
| --- | --- | --- |
| COL1a 通用凸体查询 | 同候选发布加完整固定点停滞出口，两后端各新 10、目标/候选 12、受影响矩阵 282 项通过；原四 Y 法向及九资源回归闭环。原门和容量未放宽 | 不是 GJK/EPA 全输入完备性证明，也没有完成新曲面接触 patch |
| B-point 曲面零维触点 | **FAILED / 停止当前候选**。解释首轮 11 通过、touch-SE 失败；法向满足查询门，但该法向的椭球单点支撑与旧见证不满足原位置门。其余未访问节点和 CUDA 未运行，候选未合入 | 不能把 query RESOLVED 改称 point/完整 patch 成功；C-point 载荷仍未实施 |
| COL2 有限边界核 | 源码 93 节点：4 HOST 资源、44 interpret、45 CUDA 通过。完整候选、两见证、参数、来源转换和资源失败有资格；首包两后端各 2 次 launch/8 rows 通过 | 距离是有限 segment–triangle 边界距离；球为零轴、胶囊为有限轴加半径。没有实体穿透符号、连续碰撞或载荷 |
| COL3 公共分派 | 源码 75 新项、11 旧发布兼容项通过；公共类型导出 HOST 1 项通过。支持 sphere/capsule 对 triangle/mesh 当前快照、自动 BVH 候选、完整原 facet 映射与整体发布 | HOST 快照上传、邻近带有限查询；空候选不证明闭 mesh 内部物体分离，未提供带外全局最近面。首包两后端各 4 次 aggregate 查询通过，仍不扩展实体/载荷语义 |
| BALL1 / BALL-A | 固定根 BALL/HINGE/SLIDE 完整动力学及合法标量执行器组合已通过 interpret/CUDA。BALL-A 最多 32 nu / 32 na，保 nq/nv 分离、完整耦合与恢复；首包 BALL1 两后端各 40 worldsteps、BALL-A 各 58 通过 | BALL-A 的执行器只驱动真实 hinge/slide，球被动耦合运动；**不是直接球电机**。BALL-A原验收不含接触，后继BALL-C已单独验收；仍不含自由根、限位/干摩擦等任意组合 |
| ROB2b | 单自由根/原法向或混合约束与 NONE/FILTEREXACT 标量执行器复用组合，解释 22、CUDA 22、兼容 48 通过；首包两后端各 40 worldsteps 通过 | 不是任意 transmission、闭环或完整人形动作资格；与固定球树分别受限 |
| N5 / N6 | N5 按 stable world/reset index 的随机初态、恢复已过源码；N6 D=1..8 控制窗口、完整物理子步、endpoint 任务/部分 reset、GRU 续接及跨 D 拒绝已过源码；首包 N5 CUDA 32、N6 CUDA 192 worldsteps 通过 | 当前 N5 是 q/v 状态随机化，不是质量/摩擦/资产随机化；N6 有真实失败前缀，物理已推进后的任务失败不称整窗回滚 |
| S256 紧凑镜像 | full/integration 各 48 个实际步与三次 PPO 更新通过，共同物理、learner、Adam、RNG 逐字节相同。合计 96 步 native 构造/copy 为 0，原 arena 预算不变 | 不是全流水线零 HOST 工作，也不是 256 台 G1。进程 commit 在匹配观察阶段由旧约 4.886 GB 降至 1.268 GB；**不是峰值内存或 Newton 对比** |
| N7 组合规模 | run265 J/W8/D2/GRU、266 S/W128/D4、267 S/W256/D4 的 full/integration 两模式均通过；每模式 24 policy windows，分别 384/12,288/24,576 个实际物理 worldsteps，共同物理/动作/reset/Adam/RNG（及 GRU hidden）一致 | 这是更新与规模连接资格，不是 J 学会、S 新学习实验或超越 Newton。后续评价器收紧跨窗口 action/input/output 绑定，275 HOST 32 与 276 J8 实际 pair 通过；原 S 大格不冒作新评价器下重跑 |
| 首包安装 | run235 的 509 个产品成员在 source/wheel/installed 逐字节相同；七消费者所有声明后端均 RAW_EXIT=0、strict origins 通过，无 source tests/examples 导入或 CPU 物理代跑 | 首包绑定 `85caed4e`。run240 只暴露消费者 FP32/FP64 编码错误，修复后保存原失败；后续评价器增量与新消费者须新包另验 |
| BALL-C 固定球树接触 | 源码52新节点与6受影响入口通过；最终包解释/CUDA各64worldsteps通过。完整M/J、双方COM力矩、标量驱动/activation、力与运动共同发布、连续接触与生命周期 | 限fixed BALL/HINGE/SLIDE、scalar NONE/FILTEREXACT、primitive condim1/pyramidal3、nv8/nq10；无直接球motor、FREE+BALL、limits/dry。安装stale-CAS与源码设备晚失败资格分开 |
| BALL-M1 直接球电机 | 统一native JOINT CSR映射、同次实际scalar/广义力与运动、checkpoint恢复；47新源码节点+9兼容节点；安装解释/CUDA各56worldsteps | fixed BALL/HINGE/SLIDE、直接BALL只JOINT/NONE/FIXED/BIAS_NONE、na0；无直接球电机的activation/contact、FREE+BALL、limits/dry |
| Participant scene | 显式资产/参与者/控制/完整材料对表连接原四类resident树；HOST19+interpret19+CUDA19；安装各24worldsteps | 有HOST离散接触采样；不是新求解器、MuJoCo-TYX BALL导入、跨树隐式KKT或R03通过 |
| M1 resident boundary | 实际accepted设备来源与runtime lease已实现；4HOST+67interpret+68CUDA独立节点、20兼容节点；安装各6worldsteps/3queries/12rows | 限刚性完整拓扑direct constructor。不是零HOST、M2、CCD、实体内外、带外全局最近面或接触力 |
| BALL-MC 直接球电机接触 | 源码51新节点+6兼容通过；统一包解释/CUDA各112worldsteps。原normal/pyramidal接触与真实JOINT motor的力、广义力、运动共同发布 | 限fixed BALL/HINGE/SLIDE、JOINT/NONE/FIXED/BIAS_NONE、na0、nq10/nv8/nu32；无direct BALL activation、FREE+BALL、limits/dry或任意组合 |
| M1-A 原资产设备来源 | 四类from_asset完整model/shape/graph/group/原材点进入M1；2HOST+37interpret+37CUDA、旧direct/static各8兼容；安装各6worldsteps/3queries/24rows/2stale拒绝 | 该批只验TyxWorldBatch原资产接线，后继S1–S3另验ParticipantScene；本批不扩展闭实体、CCD、接触力或变形 |
| S1/S2 场景设备查询 | 解释/HOST41、CUDA39与旧场景4兼容；安装每后端6logical/12participant steps、4queries/64rows/3stale拒绝；完整逻辑身份/同tick、共同框架、真实device FK及borrow | 当前一个真实batch/runtime；S3更高层编译声明及第二类安装场景现已另行通过。仍是有限查询，无新载荷/CCD |
| direct BALL FILTEREXACT | 70新源码节点、4旧机器人兼容；安装每后端160worldsteps。activation/JOINT映射/原normal或pyramidal接触、同次力与运动联合发布 | fixed树nq10/nv8/nu32/na32；BALL BIAS_NONE；无FREE+BALL、BALL affine bias、limits/dry或任意组合 |
| D1/M2-P 真实producer | 解释44、CUDA33；安装每后端16batch/32worldsteps。accepted/candidate bank、借用同步、实际形变和完整事务/资源已实现 | M2-Q已有本批后继资格；没有接触力、布弯曲、可移植checkpoint、最大容量或吞吐资格。详见[节点报告](p65-deformable-bank-report.md) |
| S3 公共编译查询 | 源码解释/HOST27、CUDA20；统一包每后端8逻辑/16参与者步、5queries/60rows，第二类固定网格工具＋自由物体场景通过 | 关闭当前公共设备查询连接目标；仍限同runtime、当前有限边界，无新接触力或CCD |
| D2/M2-Q 真变形查询 | 源码解释31、CUDA31；安装每后端32worldsteps/19queries；真实accepted顶点更新bounds/BVH，完整身份、恢复与失败发布 | 每后端102 RESOLVED＋12 CONDITIONING，斜平行未解析行不能施力。该批只验静态round同runtime；运动双owner响应后由D3/D5另验，持续稳态仍未完成 |
| R2 FREE＋BALL 基础 | 新源码64、受影响兼容16；安装每后端112worldsteps。单FREE根、BALL/R/P后代、完整耦合/四元数、NONE JOINT motor与恢复 | nv35/nq45/nu32；该基础批不含contact，R3-N/P后继另验；activation/limits/dry、BALL spring/affine bias、多FREE根仍未放行，无性能结论 |
| D3法向响应首片 | 解释27、CUDA28通过；统一包每后端8共同步/13worldsteps/8queries。压片/偏心/胶囊、固定反力、真实入层离层、全部晚失败及恢复 | W≤8、单浮体/世界、入步采样离散弹簧、mu0；不等于连续支撑、摩擦/换面、CCD或面积压力；资源往返完整保留 |
| G1异构功能首片 | 新源码42通过；统一包每后端27calls/162worldsteps。三种不同原支持域模型分桶，控制/完整力运动/恢复共同接受 | 原8模型16world上限，该首片正控3模型6world；G2a后继另验容量格，原不支持组合仍拒绝且无性能结论；不代表G1人形舞蹈完成 |
| D5-A 公共刚柔声明 | 新26＋受影响5节点/后端；首包81/74、第二包132/133均PASS，各38calls/57worldsteps/6queries | 球资产/程序胶囊两类短窗与完整生命周期；每world单ordinary浮体，不是任意关节机器人或持续稳态支撑 |
| R3-N FREE+BALL法向 | 源码55＋兼容7，首包75/76各112worldsteps；完整M/J、NONE JOINT、condim1力运动与恢复 | 原native坐标/容量保留；不代表activation、limits/dry或人形长时动作 |
| R3-P FREE+BALL摩擦 | 源码69＋兼容4，第二包130/131各112worldsteps；primitive pyramidal condim3及原CSR/共同约束核 | 连续正控为condim3，mixed1/3只有编译路由资格；不扩activation、limits/dry、condim4/6、elliptic3或新shape |
| G2a 原容量功能规模 | 完整W1/W8/W16源矩阵；首包77–80的W16B3/W16B8双端PASS，各27calls/432worldsteps | 上限仍8模型16world、同h；W1/W8未另做安装消费者。没有live/peak或公平Newton性能结论 |
| D6-A 轴向材料耗散 | 原elastic/reconstruct后一次候选内耗散；源码功能门与D1/D3/D5接线通过；包外128/129各22calls/44worldsteps，FUNCTIONAL_PASS | 原strict_elastic_precision=FAIL_RECORDED，旧弹性重构最大速度差约1.253865e-4 m/s；不覆盖D4稳态FAIL，不是全网络隐式/全局drag |
| D4原无阻尼 / D6-B新应用 | D4 run50解释/run69 CUDA各完成3072步/3秒，两稳态窗均FAIL；D6-B c4096 run143同样完成N1解释3072步/3秒，DIAGNOSTIC_ONLY / FAIL | D6-B逐步law/phase/算术/长参考通过，但两窗支撑/速度RMS不合格，停止该系数CUDA/余格；事件发生和算子通过不等于稳态承载 |
| J/R 学习与 N4e | 原 J/R 固定预算学习门仍失败。J 偏置居中三种子完成但未过门，已停止该因素；N4e 独立控制器在固定窗口可达 | 控制可达不是 PPO 学习成功；不追加种子、预算或改原门制造改善 |
| N8 信用分配诊断 | 原final J14011/R14211/S14111各32步W8，实际768转移/0更新及297条保存审计通过。J/R actor范数裁剪活跃、S未给区分原因；三者ratio clipping均0 | 只是本次固定采样的证据。裁剪因子不等于带原Adam历史的参数步缩放，不证明弱学习原因，不授权自动改clip/lr或新增训练 |
| G1 / R03 | G1 原 46.29 s 参考 XY 门及完整 131.48 s 动作未通过；R03 两项小净加速度门仍开放 | 已有短窗、驱动/传感修正和两侧响应不能替代这些原验收 |

表中早期BALL-M1/MC、M1-A、D1/D2、R2及G1分桶行的限制只描述当时版本：后继direct BALL activation、场景设备来源、D3/D5响应、FREE+BALL接触和G2a规模已有独立资格，不能继续列为整体未实现。表中测试数只标证据范围。D6专门带耗散D5声明已有源码121/122各2通过；普通D5安装132/133与D6直接D1/D3安装128/129分别授予资格，不冒作专门带耗散D5包外矩阵。历史TESTS_PENDING不覆盖父后续结果。

## 八个方向的大颗粒度差异

| 用户方向 | Phymeo 已有基础与本轮变化 | 与固定 Newton 能力版图相比仍需补齐的模块 |
| --- | --- | --- |
| **01 场景与机器人** | URDF/MJCF/USD/程序场景、统一编译、PD/IK、四类resident树及新增显式多参与者装配；MuJoCo-TYX自由根标量树、BALL1/A/C、ROB2b、直接BALL JOINT电机/FILTEREXACT及原接触；新公共scene设备查询、D5单浮体刚柔及R3-N/P | 更广关节/闭环/传动、FREE+BALL的activation/limits/dry/condim4/6组合；S3/D5公共接线及R3-N/P已通，继续补多body/关节刚柔、多机器人/工具抓取释放库。单个Panda/G1不替代组合矩阵 |
| **02 碰撞检测与响应** | 既有RR/RD/DD有限域、S/C/OBB/H设备patch与载荷、七形状support-map查询；COL2/COL3、M1与S1–S3真实设备来源；D1/D2真实变形查询、D5公共运动球/胶囊双向法向响应和D6可选材料耗散 | 斜平行胶囊仍有CONDITIONING。持续稳态支撑（原D4与新D6-B N1均FAIL）、摩擦换面历史、曲面patch、通用CCD响应及更广多刚体/变形组合；预计算SDF、SDF接触缩减和hydroelastic仍独立缺失 |
| **03 GPU 与多场景** | TYX批量动力学/约束、W129/W256受载分页、设备action、真实GPU PPO；紧凑镜像、N7 S128/S256两模式；accepted设备FK/持久节点、G2a原cap内W1/8/16源矩阵和W16B3/B8安装资格 | 原cap之外的扩容与资源资格、更广异构资产/约束和变形耦合规模、HOST往返优化；live/peak资源与吞吐基线。尚无Newton同模型/精度/输出性能比较 |
| **04 可微仿真与训练工具** | 实际 CUDA RSL PPO、MLP/GRU、组合 checkpoint、S 持续支撑学习成功、N5 随机状态重置、N6 控制频率窗口与 N7 组合规模资格 | J/R 仍未达学习门；更广机器人训练、模型/环境随机化、可复用任务库。多步物理梯度、辨识与梯度对照仍是独立能力，不能由 PPO 能运行推导已可微 |
| **05 传感与交互** | 深度/法线/ID、TYX 射线、IMU 基础、真实轨迹 Three.js、G1 传感坐标绑定 | 批量、多传感器时序/坐标协议，更广 RGB/材质/光照和交互编辑工具。单个 gyro 修正不代表完整传感生态 |
| **06 跨域求解编排** | 事务、点端口、域图与指定刚–布–软体耦合；同 world 多参与者共同发布 | 通用区域/后端选择、异构界面力/约束、耦合迭代、能量与共同失败恢复。resident 多树一次载荷组合不等于跨树隐式 KKT，也不等于通用多求解器耦合 |
| **07 布与体软体** | 拉伸/剪切/弯曲、自接触、StVK 四面体、双向载荷和递布基础；新增D1持久节点、D2查询、D5公共D3响应、D6轴向材料耗散 | 持续稳态应用、多body/关节刚柔、摩擦/换特征、密集折叠、软块抓捏、广泛材料/网格和软布放台。需完整多案例资格，原有限几何/响应范围不扩大为全部柔性碰撞 |
| **08 杆缆、MPM/流体** | 刚性连杆、粒子与弹性体基础可复用 | 完整弯扭杆缆、粒子–网格转移、塑性/流动、流固边界仍需分别立项；连杆不是柔杆，粒子不是 MPM/流体求解器 |

Newton 本身也按 solver/backend 配置提供能力；上表不把所有 Newton 模块描述为可以任意一起使用。Phymeo 的旧 HOST 曲面几何和部分载荷资格仍可复用，但不自动升级为新 device 支持或完整多场景接口。

## 后续三条主线：从剩余依赖开始

沿用户既定三主线推进，[原路线图](../superpowers/plans/2026-11-12-scene-deformable-scale-roadmap.md)中的D1/D2/D3、S1–S3、D5-A、R3-N/P和G2a已有当前限定资格，不再重复安排为从零实施。[当前功能矩阵](p65-feature-acceptance-matrix-20261113.md)与[本轮计划](../superpowers/plans/2026-11-13-feature-completion-until-0730.md)记录本次结果及执行队列。

| 优先方向 / 难度 | 已接通的公共节点 | 下一可交付节点与验收依赖 |
| --- | --- | --- |
| **1 / 高：统一场景到设备碰撞/响应** | S1–S3资产/控制/来源到当前查询，D5-A到单浮体＋节点的共同响应与生命周期，两类包外场景已过 | 多body/关节资产刚柔、更广多机器人/工具/物体组合；需明确完整载荷、稳定身份和共同失败发布，不能只放开构造guard或顺序调用两个step |
| **2 / 高：真实变形到持续载荷** | D1持久bank→D2查询→D3响应→D5声明；D6-A轴向材料算子功能已资格且原strict失败保留 | D4无阻尼与D6-B c4096 N1稳态均FAIL；该系数停止扩跑，不凭更大阻尼或扫描参数追绿。后续须根据真实轨迹另立可辨别合同；摩擦/换面历史、CCD和更广材料/网格独立登记，轴向零模/未解析行/HOST往返仍明示 |
| **3 / 中高至高：机器人组合与GPU规模** | FREE+BALL/NONE JOINT condim1及pyramidal3、原8模型16world异构源矩阵与指定W16安装已过 | activation、limits/dry等尚缺组合先独立多案例；异构W32/128/256须新增完整资源/生命周期资格，随后才做同硬件、同模型/输入/质量/完整输出的Newton1.7.0基线与热点优化 |

D6-B run143已结束，3072 attempted/returned/checked步、3 controls、close1完整；本次文档只引用父闭合结果，不另算轨迹。两稳态窗支撑均值21.46127079/19.36648514 N，刚体速度RMS .30439415/.21362712、中心RMS .16520125/.11754240 m/s，均未满足原支撑范围/0.1 m/s门。逐步law、耗散phase、算术与长参考通过，另7项旧弹性短精度记录（最大约3.56759e-5 m/s）仍保留；两类失败不互相掩盖。后续源码可以按不交叉owner并行编写，但项目import/测试/数值/build/install仍保持唯一串行。性能超越不是功能验收的替代条件；不可比模型格、live/peak未知项与保留失败必须单列。

[原机器人组合研究](p65-robot-next-compositions-20261112.md)和[真实producer设计](p65-m2-producer-bank-design.md)保留可复用依据，不要求重写原核。N8 saved梯度/Adam诊断只作有明确假设的训练后继；不自动新增训练或挤掉三主线。传感、可微物理、广泛软体/杆缆/MPM仍在八方向缺口表，曲面touch-SE表示失败也继续保留。

## 如何识别成功与无效循环

- **成功**：公共能力能在不同形状、来源、world 和连续步上复用；独立 oracle、原数值门、原子发布/容量负控和安装调用均有实际证据。学习成功额外需要原持出门，不只训练脚本退出成功。
- **已推进但未完成**：能明确缩小缺口。例如 COL2/COL3 完成当前边界查询，但尚无载荷；N4e证明控制可达，但学习仍失败。单列边界即可继续独立模块。
- **应停止当前假设**：同一最小反例在一次有界通用修复后引入新回归，或两次针对尝试未缩小障碍；保存原输入、真实中间量和失败，回到合同/表示或另一个独立模块。不能不断改 fixture、门、容量、预算来使案例消失。
- **避免重复劳动**：同源已绿范围按原证据复用；算法/来源发生实质变化才跑受影响回归。先诊断实际数据再选修复，不重新做无限精确证明或无假设训练。

## 当前关闭与待补结果

| 收尾项 | 当前状态 | 证据边界 |
| --- | --- | --- |
| 首包 `85caed4e` + 七消费者 | **PASSED** | run235 509 成员 parity；236/237 ROB2b、238/239 BALL1、245/246 BALL-A、247/248 COL2、249/250 COL3、251 N5、252 N6 全部 RAW0/strict origins。240 消费者编码首错保留 |
| N7 三格 | **PASSED，限定组合与更新资格** | 265/266/267，两 mode/格，每 mode 24 policy windows；J8 384、S128 12,288、S256 24,576 worldsteps。共同完整状态/更新一致，不称学习或 Newton 性能胜出 |
| 新评价器 `4118f2ec` | **源码及最终包 PASSED** | 275 HOST32、276 J8实际pair；第二包290 CUDA192worldsteps。完整action/input/output跨窗口绑定，不倒灌首包 |
| BALL-C | **SOURCE_AND_INSTALLED_PASSED** | 284/285共52新节点，286兼容6；291/292源码外各64worldsteps。范围见[BALL-C报告](p65-ball-contact-composition-report.md) |
| BALL-M1 / Participant scene / M1 | **SOURCE_AND_INSTALLED_QUALIFIED** | 515成员同源；061–066六消费者通过。新增72文件安装归档及installed-close.json；详细源码矩阵、修复与首失败见本批报告 |
| BALL-MC / M1-A | **SOURCE_AND_INSTALLED_QUALIFIED** | d362799f的516成员parity；043–046四消费者通过。61文件安装归档与installed-close.json保留完整压缩表示 |
| Batch A：S1/S2、D1、BALL FILTEREXACT | **SOURCE_AND_INSTALLED_QUALIFIED** | d2a18bee的520成员parity；043–048六消费者通过。143文件安装归档及installed-close.json，完整解释/CUDA原返回和参考保留 |
| Batch B：S3、D2/M2-Q、FREE＋BALL基础 | **SOURCE_AND_INSTALLED_QUALIFIED** | 0321c5a4的522成员parity；39/40、41/42、45/46六消费者通过。147文件安装归档及installed-close.json保留完整返回和首失败 |
| Batch C：D3与G1异构分桶 | **SOURCE_AND_INSTALLED_QUALIFIED** | 3534f742的528成员parity；38–41四消费者RAW0。18文件完整保存归档；D3每后端13worldsteps，异构每后端162worldsteps |
| D5-A / R3-N / G2a首包 | **SOURCE_AND_INSTALLED_QUALIFIED** | 113629b3，530成员同源；D5 81/74、R3-N75/76、G2a77–80指定消费者通过。范围见当前功能矩阵 |
| R3-P / D5第二包 | **SOURCE_AND_INSTALLED_QUALIFIED** | ec0e2997，532成员同源；R3-P130/131、D5普通兼容132/133均PASS，不冒作全部历史功能回归 |
| D6-A第二包 | **FUNCTIONAL_PASS / STRICT_FAIL_RECORDED** | 128/129各44worldsteps；耗散相位/事务与连续工程门通过，旧弹性重构严格速度失败保留。带耗散D5专门源码格与普通D5安装分开 |
| 连续载荷完整应用 | **D4 FAIL；D6-B N1 DIAGNOSTIC_ONLY / FAIL** | D4无阻尼双后端各3秒稳态未合格；run143的c4096新材料也完成N1解释3秒但两稳态门失败，停止该系数CUDA/余格，不改写D4或D6-A算子资格 |
| 架构与整体状态 | **公共连接已更新，P6.5仍未完成** | 当前Markdown图区分resident刚柔、MuJoCo-TYX和异构聚合；07:20 HTML为历史。G1、J/R、R03、B-point及其它模块仍开放 |

## 固定依据与入口

- [当前功能矩阵](p65-feature-acceptance-matrix-20261113.md)、[当前架构](p65-feature-architecture-draft-20261113.md)、[D5](p65-deformable-scene-report.md)、[R3-N](p65-free-ball-normal-report.md)、[R3-P](p65-free-ball-pyramidal-report.md)、[G2a](p65-heterogeneous-scale-protocol.md)、[D6-A](p65-distance-dissipation-report.md)、[D6-B预登记](p65-damped-normal-load-application-20261113.md)。产品/结果依据父b1e95c70功能矩阵及e25df478报告，D6-B状态据父run143已闭结果追加。
- [历史Batch C实际结果](p65-three-track-batch-c-20261112.md)、[本批计划](../superpowers/plans/2026-11-12-three-track-batch-c.md)、[Batch B](p65-three-track-batch-b-20261112.md)、[Batch A](p65-three-track-batch-a-20261112.md)、[上一批保存证据](p65-parity-motor-asset-20261112.md)、[上一批](p65-parity-m1-assembly-20261112.md)、[旧07:20记录](p65-parity-until-0720-20261112.md)；父执行事实优先于旧作者TESTS_PENDING标签。
- [公共三角边界查询](../architecture/triangle-boundary-query.md)、[COL2 报告](p65-device-triangle-distance-report.md)、[COL3 报告](p65-triangle-distance-dispatch-report.md)。
- [BALL-C 后继合同](p65-ball-next-composition-contract.md)、[M1/M2 设备来源合同](p65-device-mesh-source-next-contract.md)、[N7 组合规模](p65-window-scale-report.md)。
- [ROB2b 组合](p65-rob2b-actuator-composition-report.md)、[BALL1](p65-native-ball-joint-report.md)、[BALL-A](p65-ball-scalar-actuation-report.md)。
- [N4a 学习](p65-mujoco-tyx-n4a-report.md)、[N4d 固定消融](p65-n4d-initialization-protocol.md)、[native 紧凑镜像](p65-mujoco-tyx-native-mirror-report.md)、[观察内存诊断](p65-compact-mirror-memory-diagnostic.md)、[G1 保存诊断](g1-drift-diagnosis-20261111.md)。
- Newton 1.7.0 官方固定参照：[solver 矩阵](https://newton-physics.github.io/newton/1.7.0/solvers/index.html)、[collision](https://newton-physics.github.io/newton/1.7.0/concepts/collisions.html)、[worlds](https://newton-physics.github.io/newton/1.7.0/concepts/worlds.html)、[sensors](https://newton-physics.github.io/newton/1.7.0/concepts/sensors.html)、[coupling](https://newton-physics.github.io/newton/1.7.0/concepts/coupling.html)、[deformable objects](https://newton-physics.github.io/newton/1.7.0/concepts/deformable_objects.html)。仅作功能/架构参照，没有新增 Newton 性能实测。

本页只依据既有源码/安装结果和固定Newton1.7.0参照，没有新增Newton调查、项目执行、性能实测或未执行的通过声明。run143已闭且稳态FAIL保留；后继仍由父按独立合同持唯一串行槽，不从图表推导未执行的组合资格。
