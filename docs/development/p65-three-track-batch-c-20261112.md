# Batch C：刚柔共同响应与异构模型分桶

状态：**SOURCE_AND_INSTALLED_QUALIFIED（本页限定范围）**。源码验证于2026-11-12完成，统一安装与保存核对于2026-11-13完成。产品快照 `3534f742fefe16f50e90d15dd5d0fbf70a057d6b`；不代表整个P6.5或Newton功能对齐完成。

## 可调用交付

- `TyxDeformableContactBatch`：真实设备节点与真实运动球/胶囊共享入步状态，复用原法向接触层、ordered节点材料与刚体动力学。双方力、COM力矩、固定点完整材料反力和运动在同一事务发布。失败保留原状态、历史和epoch；reset/restore后有真实后继。
- `MuJoCoTyxHeterogeneousBatch`：按完整源模型/资产/容量分桶；不同DOF与控制维度的world共享公共身份和调用入口。控制、物理步、reset/restore跨桶共同接受，晚桶失败不半发布；边界清理错误与已提交物理分开记录。
- 父共享接口：原D1和MuJoCo-TYX双bank拆出候选prepare/validate/commit，受管runtime lease支持刚柔双来源。保留原求解法则和准入限制。

D3是明确的入步采样、零摩擦离散法向层，首域W≤8、每world一个节点模型及一个ordinary单body浮动刚体。每facet一根离散弹簧，不声称面积压力/任意细分不变；载荷读回再上传原通道已计费，不称全GPU或隐式耦合。

异构分桶首域最多8模型/16world、同h；实际正控为3种已支持模型和6world。标量FILTEREXACT、fixed BALL＋原pyramidal接触/FILTEREXACT、FREE＋BALL/NONE无接触分别保原准入；不是任意组合、异构单kernel或规模性能资格。本页“G1分桶节点”不是G1人形完整舞蹈验收。

## 实际验证

所有项目执行由父/root持唯一串行槽，fresh entrypoint-aware guard，原raw日志/退出值与source快照保留。证据根：`artifacts/p65-three-track-c-20261112-01`。

| 组 | 实际结果 | 范围 |
| --- | --- | --- |
| 04/05 原节点共享接缝 | interpret 44；CUDA33＋11 HOST-only deselected，RAW0 | 原D1双文件；未扩大为旧全量回归 |
| 06 原runtime候选接缝 | 6通过，RAW0 | actual controls与FREE/BALL受影响兼容，两模式 |
| 07/08 原resident事务 | interpret7、CUDA7，RAW0 | 完整原受影响事务文件 |
| 17 异构新双文件 | 42通过/57.81s，RAW0 | 两模式、完整力/运动参考、分桶身份、原子失败与恢复 |
| 32 D3解释 | 27通过、1 CUDA-only deselected/38.58s，RAW0 | 压片、偏心固定点、运动斜胶囊、h/质量变体、入层/离层、外载功、事务和资源 |
| 33 D3 CUDA | 28通过/35.20s，RAW0 | 完整同矩阵及真实free-then-throw UNKNOWN_RELEASE |
| 35–37 统一包 | build/install/parity RAW0 | 528产品成员在source/wheel/installed逐字节相同 |
| 38/39 异构包外消费者 | interpret/CUDA均PASS、RAW0 | 每后端27真实calls/162worldsteps；115完整保存events，含恢复和局部reset后继 |
| 40/41 D3包外消费者 | interpret/CUDA均PASS、RAW0 | 每后端8共同steps/13worldsteps/8queries；1restore、1partial-reset、1stale拒绝，54完整records |
| 42 保存核对 | RAW0 | 四消费者strict installed origins、原完整gzip字节/长度/hash、18文件无损归档 |

安装使用全新源码外目录和新target，仅安装本wheel、no-deps/no-index/no-compile；没有重新安装依赖或修改现有环境。四消费者未导入源码tests/examples或由CPU物理代跑。安装消费者覆盖新增公共能力，不冒称新包全量回归。

归档 `artifacts/p65-three-track-c-20261112-01/installed-wheel-01` 共18文件、11,669,729 bytes；`installed-close.json` SHA256 `4cec0531fc6e69ebee9c6280964c5935284ace1d8b88269ce29bfac847595b41`。保完整压缩流，不删除原件、不重新序列化。四流逻辑字节分别48,521,705、48,516,667、1,965,533、1,966,077；它们是各自同次证据，不要求interpret/CUDA轨迹逐字相同。

## 真实首失败与修复

1. 首次共享/G1/D3 Ruff和format结果保留。G1仅import排序/排版；5文件中4个raw AST相同，全部nonimport tree及scoped import bindings相同。D3除排版/import外，删除测试未用numpy绑定、未用局部row改为_row；6文件中4个raw AST相同，全部qualified检查相同。安装consumer纯format前后raw AST/import序列相同。
2. run25是实际产品问题：节点保留输入world次序(31,7)，resident按(7,31)规范化，构造跨来源身份不一致。修复统一以稳定world ID规范化两侧，保持每world原节点、材料与物理参数。
3. run26是清理负控的错误消息匹配问题：产品保留了未确定释放的owner并报告聚合异常，测试正则没有覆盖该公开消息。仅修正消息匹配，UNKNOWN_RELEASE、保留owner、release次数等原断言未变。
4. 最终完整D3双后端与安装验证均一次通过。首失败没有删除；未调物理fixture、质量、阈值、容量或独立oracle追绿。

## 后继

[三主线路线](../superpowers/plans/2026-11-12-scene-deformable-scale-roadmap.md)仍按功能优先：D4先完成连续法向支撑/卸载/再接触，随后摩擦/换面历史及D5公共场景组合；G2先在原容量内完成W1/8/16异构质量与资源记录，再讨论有依据的扩容及G3公平基线。D3短窗的入层/离层不自动关闭D4。

[差异全景](phymeo-newton-panorama-2026-11-12.md)保留FREE＋BALL接触/activation/limits/dry、通用CCD/曲面patch、广泛变形与多关节组合、训练/传感/跨域及软体等缺口。原G1人形46.29s/131.48s、J/R学习门、R03两门与B-point touch-SE仍未通过，无性能超越或整体完成声明。
