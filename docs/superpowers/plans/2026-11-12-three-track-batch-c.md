# Batch C：真实刚柔响应与异构机器人功能

状态：**SOURCE_AND_INSTALLED_QUALIFIED（限定D3/G1首片）**。父基线 `5d3b4c50`，2026-11-12；统一安装收尾2026-11-13，产品 `3534f742`。
按[三主线路线](2026-11-12-scene-deformable-scale-roadmap.md)继续；Batch B已验范围保留，不重跑同源旧组。

## 本批交付和边界

| 节点 | 难度 | 可调用交付 | 本批不包含 |
| --- | --- | --- | --- |
| D3 首片 | 高 | 真实设备变形节点与真实6DOF运动球/胶囊共用入步来源，法向力、材料内力、双方力/COM力矩及运动共同提交；有固定点反力和完整失败恢复 | 摩擦、通用CCD、隐式接触收敛、密集网格连续支撑、多关节刚柔组合 |
| G1 首片 | 中高 | 不同已验机器人模型按模型分桶，公共world身份、控制、完整力/运动收据、step/reset/restore跨桶共同提交 | 任意关节/接触组合、单kernel异构、训练收益、GPU规模或超越Newton的性能结论 |
| 父共享接口 | 高 | 原节点和MuJoCo-TYX求解器的prepare/validate/commit接缝；受管共享runtime、刚柔共同恢复及公共导出 | 重写积分器、改变原已验数值法则或跳过原能力拒绝 |

## 本地源码调研结论

- 原 `TyxDeformableBatch.step` 已使用accepted/standby双bank；原ordered投影、速度重建、残差和trace可原样复用，只拆出未发布候选。
- `ResidentBatchTransaction` 已从同一entry生成完整J/M/速度，再由adapter给双方COM载荷；所有候选、诊断、收据成功后才切bank。扩展这一事务，不能顺序调用两次public step再回滚。
- 原 `normal_layer_rows` 是已使用的可标定法向层；复用kernel而非自建另一法则。每声明facet是一根离散法向弹簧，不声称面积压力或任意细分不变。
- 固定Newton 1.7的XPBD粒子接触核将同一contact反作用累加到粒子和刚体COM，但其约束法则不同，只借鉴映射/组织，不将Newton轨迹当本层逐字oracle。
- 固定Newton 1.7 `solver_mujoco.py` 的world一致性检查要求相同body/joint/shape等结构；builder中的碰撞模板复用不是异构动力学。G1明确是Phymeo同模型桶调度，不误称复现任意异构单核。

## D3实施合同

第一域为W≤8，每world一个原D1模型和一个ordinary floating participant、body_count=1，允许偏置COM/非主惯性。节点原容量与距离核限制不变；运动球/胶囊通过真实resident device FK，节点通过accepted device x共同gather，不HOST回填当前几何。

共同owner创建resident scene，再以受管runtime lease创建节点owner。query的双来源明确保两个owner、epoch、bank与tick；runtime相同不代表物理状态相同。资源关闭按query→nodes→scene，UNKNOWN释放保留依赖，不重复free。

步序：同时锁定两owner→完成entry query并退出只读借用→进入节点候选事务→在原resident sample中按同entry位置/速度计算层力→原节点材料积分和原刚体动力学分别写standby→验证全部world、双方实际力/力矩和完整收据→最后直接字段共同发布。任一near候选CONDITIONING/无有效法向，或者任一world候选失败，整个attempt不发布。

首片为明确的entry-sampled离散法则、mu=0。允许真实有界穿透/漂移；不恢复精确相触前置，不将此显式法则称隐式耦合。原生力核结果可下载后经原节点force/刚体wrench通道上传；计入所有往返，不称全GPU响应。查询结果属于内部attempt，不提前对外发布成功响应。

固定点材料反力使用每个ordered row的真实方向和delta-lambda历史，不能以最终方向乘最终lambda替代。固定节点contact/external输入仍保留，质量为零的加速度不能把反力证据抹掉。

测试作者预登记P压片/E偏心固定点/C运动斜胶囊三类，两h和质量变体；原r=1/8、gap=1/32、width=1/16、k=128 N/m、d=1/2，位置/完整oracle见独立cases文档。测试包括真实非共面变形、双方力/矩、材料反力、真实非零入步速度、late failure、部分reset/restore后继、未解析负控、资源exact/cap−1、关闭/重入。

## G1实施合同

公共模型spec绑定编译记录与contact/constraint容量；world spec绑定稳定world和model key、初态。最多8模型/16world、同h。按完整源模型/资产字节及容量分桶，不能仅按DOF或形状相似合并；不同world/模型的控制维度分别校验。

普通step持全部桶锁，先准备全部候选和aggregate再验证、提交；不每步clone。set_controls同样先完整验证，不逐桶半接受。reset/restore可在边界clone受影响桶，先用完整原checkpoint恢复双分量clock和所有未选world历史，再做原操作；全部替代桶成功才交换。清理错误与已接受物理分开保留owner，不能伪称回滚。

至少三种不同已验域模型（含activation/contact一格及FREE＋BALL基础格），非零控制/外载、相同模型不同world、存储反序与别名合桶、连续独立CPU参考、完整原同入步参考、末桶故障/恢复后继。禁止通过删除FREE＋BALL contact/activation等guard扩大支持。

## 所有权与验证顺序

- 原 `mtyx_t2_source`：D3三个NEW产品文件和报告。
- 原 `mtyx_t2_tests`：D3 cases、两个NEW测试文件及独立安装consumer。
- 原 `mtyx_t2_audit`：G1 facade/checkpoint、设计、两个测试文件、安装consumer和报告。
- 父：所有既有共享源码、公共导出、集成、证据与唯一执行槽。原3份未跟踪草稿不动。

作者仅隔离编码/文本调研，禁止任何项目import/collection/数值/pytest/lint/formatter/build/install/profile。父合入后按变更范围串行：保首次static结果→修复→新文件完整interpret/CUDA→实际共享受影响兼容→统一wheel→两个独立源码外consumer两后端→收尾提交。每条执行前fresh entrypoint-aware guard；真实外来执行自然结束，不kill。所有首失败保留；同源完整绿组不重跑。

本批关闭要求公共能力和多案例真实通过，而非文档或测试数量。若同一假设两轮不能缩小问题，保存最小反例并调整接口/分片；不通过调fixture/预算追绿。D4/D5、G2/G3及原G1舞蹈/JR学习等未过项仍独立保留。

## 实际关闭

[父执行结果](../../development/p65-three-track-batch-c-20261112.md)记录完整范围、首失败和修复。G1新双文件42项通过；D3解释27、CUDA28通过；受影响节点/runtime/resident共享接缝兼容通过。产品528成员source/wheel/installed逐字节相同，两个独立安装消费者各解释/CUDA均通过；G1每后端27calls/162worldsteps，D3每后端8共同步/13worldsteps/8queries。四份完整压缩轨迹及安装证据已核对归档，原始失败保留。

本合同未放宽数值门、质量、容量或原求解法则。D3仍为入步采样法向层，G1仍为受限同模型桶编排；后继D4/D5、G2/G3照原路线推进。
