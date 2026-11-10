# G1完整动作：已保存漂移诊断与下一定位

2026-11-11。原v2 run93仍为 **FAIL**：目标131.48s，在46.29s/23145接受步因设备参考XY误差超过原1m门停止。无设备拒步或reset。run106的保存审计及run123的离线诊断成功，只说明证据和诊断完成，不把完整动作改为成功。

原v1在112.372s触发独立CPU参考XY门的失败另存。新版使用正确的当前IMU-site角速度；更早停止不证明该语义修正错误，也不允许用回滚、改门或初态修补追结果。本夜不再跑一遍完整动作。

## 两类比较应分开

1. **独立闭环轨迹**：CUDA与CPU从原初态各自反馈、各自演进。直接比较同一输出步的根部XY位置。
2. **同输入物理诊断**：以设备的同一入步q/v和已保存控制创建fresh native MjData并mj_forward，与该设备步的物理收据比较。它不沿用独立CPU的闭环历史或warmstart，不是第三条连续轨迹。

run123对原完整压缩流只读一次，复用run106协议审计；处理23145 attempts/steps和6159完整捕获。原压缩与逻辑SHA均吻合，生成完整精简前缀、29关节饱和分段和小结，没有运行策略或物理。

## D1：直接位置差与参考跟踪误差

| 名义时间 | 两条轨迹根部XY距离 | 设备参考误差 | 独立CPU参考误差 |
|---|---:|---:|---:|
| 2s | 0.00107mm | 0.05902m | 0.05902m |
| 10s | 0.01380mm | 0.04342m | 0.04341m |
| 20s | 6.396mm | 0.15471m | 0.16046m |
| 30s | 8.138mm | 0.02982m | 0.03782m |
| 40s | 5.124mm | 0.60741m | 0.60363m |
| 46.29s | 3.291mm | 1.000277m | 0.999777m |

全保存前缀根部XY直接距离最大17.3036mm，首现输出tick13577；末步差向量为(-0.00071231,-0.00321277)m。该次共同跟踪偏差远大于两后端位置差，**提示优先核共同部署/模型/策略配置，但不能证明TYX完全无关或策略有错**。

参考帧仅21847/23145步相同；直接位置差不依赖参考帧，两条参考误差曲线却不能默认每步对同一帧。用于解释的参考向量是按固定运动文件和源码公式重建，明确标 `reconstructed_source_algebra`，不是从日志恢复原alignment字节。此次重建的范数与已存范数逐步相同，不改变这一来源分类。

6159完整捕获中的同输入最大分量差为：

- 物理qacc：796.3317649，输出tick17855。
- qfrc_constraint：111.1804938，同一捕获。

这是含平移与转动自由度的原始最大分量，不统一写成m/s²或N，也不是全23145步的最大值。tick为输出步标签，力/加速度是该步入端评估；不把单峰直接归因为最后1m漂移。设备另存integration_acceleration，不能将它与native物理qacc混比。源码口径复核未发现输入/输出时点错比。

## D3：限幅与真实足底力矩

| 通道 | 设备饱和物理步 | 独立CPU饱和物理步 | 设备最长连续段 |
|---|---:|---:|---:|
| 右踝pitch | 848 | 885 | 130步 |
| 右踝roll | 1 | 0 | 1步 |
| 腰roll | 50 | 50 | 50步 |
| 腰pitch | 113 | 100 | 40步 |

这些是每个物理步的PD请求被原actuator ctrl range裁切的次数，不是策略推理次数、求解拒绝或已证明电机能力不足。四通道的同保存raw在±50下的请求差仅作离线代数；没有模拟改限幅后的运动，不能据它决定增大力矩上限。

足地condim6确实产生扭转/滚动力矩：设备左右足接触坐标扭转绝对峰值约0.4122/0.4534N·m，滚动力矩范数峰值约0.000351/0.000322N·m。保存诊断分别统计detected、active、正法向力、原已承载足四组；保原总足世界Fz>1N的loaded含义。样本力矩时间积分不是机械功或总刚体力矩，非零本身也不证明它造成漂移。

## 部署来源：已知与未知

当前公开ONNX及同伴deploy/运动文件有固定hash，但没有找到能将该权重绑定到准确训练run的env.yaml、agent.yaml、训练checkpoint与依赖清单。既有ONNX custom metadata为空；不能把上游当前默认配置当作它的原训练设置。

一项具体历史差异：同一公开权重保留，而同伴29DOF部署文件曾将waist_yaw的动作scale从0.35改成0.55；提交没有解释此变更原因。[官方改动](https://github.com/unitreerobotics/unitree_rl_mjlab/commit/d5ba5b2d21fc778a506804dc7505a8ecbca3c088)

当前模型的统一armature/damping、左右踝不对称限幅、condim6和h=.002/Euler，与固定训练源码默认候选存在差异。但该ONNX准确训练配置仍unknown；这是一组可检验候选，不能一次性照抄默认值或声称“恢复原训练模型”。详本机 `g1-policy-provenance-followup.md` 与 `g1-drift-source-review.md`。

## D4：同输入峰值已定位到接触检出集合

run126于06:09:39结束RAW0，只读取原gzip至输出tick17855，保留原source/initial/attempt/step四行；完整流hash/CRC复用123，不重做全流审计或物理。固定脚本v2的新增内容只澄清来源：设备请求/存储控制逐字绑定；native没有保存三项原始控制数组，其控制由固定harness传参链绑定。

本步设备检测出3个接触、30条EFC；same-input native有4个接触、40条EFC。差异先出现在检测集合，不能按“同一个约束问题、仅求解器结果不同”解释。

| 保存事实 | 结果 |
|---|---|
| native独有接触 | geom(0,22)、body(0,13)、condim6 |
| 几何22 | right_foot6_collision，胶囊，right_ankle_roll_link；半径0.01m、半长0.0835m |
| native距离/门 | -3.2631712111944045e-8m（约-33nm），include_margin=0，active=true |
| native该点法向力 | 109.281806N；同组完整Jᵀf最大分量109.281806，与全约束力差111.180494同量级 |
| 设备候选是否缺枚举 | 没有。静态pair21的41/42两槽明确对应胶囊两端，保存candidate与J均为零 |
| 共同3点的几何差 | normal完全同；position最大约4.0e-8m，J最大约9.3e-8（各自单位按字段） |
| 动力学输入差 | applied和qfrc_actuator完全同；M最大分量差2.56e-6，bias约8.67e-6 |

冻结run93源码给出具体层级：source-14-_normal_geometry.py先清零槽，枚举全部注册pair，再对plane/capsule两端计算距离；第254行只有`distance <= margin`才写present，第329行之后才判active。source-4-_constraint_rows.py只消费present，source-3-_constraint_publication.py保留原候选槽。因此该pair的差异发生在**窄相检出margin门**，不是漏注册、动态宽相遗漏、active筛选或QP删行。

已接受步与该分支的eligible=1控制流支持“设备两端距离均大于零”的推断，但拒绝候选的实际距离没有保存。不能编造设备距离，也不能进一步指定是FK、局部变换、胶囊端点或半径运算导致符号差异。native约-33nm的结果说明这里靠近零margin的精度敏感边界；这不等于证明全部长期跟踪漂移由它造成。

设备normalized KKT约4.05e-7，只描述自己的30行问题。两侧完整行Jᵀf可分别重构；多点/切向基不完全相同的行不强制一一对齐。质量方程中applied已含执行器，未重复加力。该局部差异是具体工程边界，不把它升级成“全引擎必须精确相触”的新前提。

## 后继：两个小问题，不阻塞GPU训练

1. 接触边界资格：下一批在预声明小案例中保留**被拒候选的距离/理由**，用plane–sphere/capsule/box的正/零/负间隙、非零margin/gap及接近/分离速度核对检测与激活语义。复用这一已保存输入作回归，但不能只为G1单点调epsilon；精度差与持续支撑稳定性分开验收。先界定工程接触策略，再决定是否需要产品修复，避免无休止追逐FP32与FP64的零点逐字相同。
2. 部署兼容：先取得来源支持并固定一个改动、同输入小案例与停止条件；缺原训练run时明确叫兼容变体。不能一口气改动作scale、armature、限幅、condim或h后重跑舞蹈并宣称定位完成。

本夜此诊断到此收口，不重跑全窗、不改模型/门/内核。下一主线仍为[GPU训练N1–N3](../superpowers/plans/2026-11-11-mujoco-tyx-gpu-training.md)，其环境/两任务训练连接无需等待G1全程成功。

## 本机证据与代码入口

根目录 `artifacts/p65-parity-20261111-01`：93原失败；106失败前缀保存审计；123离线诊断及 `123-g1-drift-analysis-output/report.json`。123脚本SHA256 `e7ea0d33df374b1f9841e4a98fffb15aa2929d361232ad1661ef833de3d1c028`。原gzip SHA256 `574bff1d33dc16d1390bcf698b1763f9caa233fb866edb77704b5a7f7e57c12f`，保持压缩，不全量解压落盘。

[harness](../../examples/g1_mujoco_tyx_policy.py)中same-input fresh forward、独立CPU输入/输出缓存、观测/PD分别保留独立字段。[runtime](../../src/phymeo/backends/_mujoco_tyx_runtime.py)分别发布物理与积分加速度。所有新统计均来自保存证据；完整动作成功仍未实现。

run126：`126-g1-peak-localization-output/report.json`，SHA256 `5818b01c18b1967f4f7cfb9d59891879854a9aa424c2b30ef708cbf806eeee05`；脚本 `analyze_saved_step_v2.py` SHA256 `9d45bd20b56e2caca1f3b204a0d76bf8f98611b1605dfdcbd4b90e7f1532b0cd`。原始四行与前后文件stat保留，无重算完整gzip hash。执行owner已于06:12释放，matching execution=0。
