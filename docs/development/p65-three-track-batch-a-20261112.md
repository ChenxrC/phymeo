# 三主线 Batch A：实际交付与验证

状态：**SOURCE_AND_INSTALLED_QUALIFIED**，2026-11-12。产品提交 `d2a18bee2758b628fabdefa399edb9d67058a4d0`；此后收尾只改文档。范围来自[Batch A](../superpowers/plans/2026-11-12-three-track-batch-a.md)及[三主线路线](../superpowers/plans/2026-11-12-scene-deformable-scale-roadmap.md)，不是整个 P6.5 / Newton 对齐完成。

## 用户获得的能力

| 主线 | 新的公共行为 | 仍未覆盖 |
| --- | --- | --- |
| S1/S2：统一场景设备查询 | `ResidentSceneBatch.prepare_boundary_query` 将逻辑 world、participant 与真实 accepted 设备状态连接；在有效借用内转换不同 anchor、执行有限表面查询，发布原 shape/facet 身份、完整参与者 revision/tick。覆盖四类树、工具/自由物体、稀疏 world、过滤及恢复 | 当前借用一个真实 TyxWorldBatch 中各参与者的机械 world，未承诺任意跨 runtime/device bank。不是新接触响应、CCD 或闭实体内外判定；S3 更高层编译声明自动接线和第二类独立安装场景仍待补 |
| D1/M2-P：持久真实变形状态 | `TyxDeformableBatch`、`DeformableNodeWorld`、`DeformableSurface` 持有 accepted/candidate 两套设备状态，复用原积分、ordered 距离投影与速度重建。普通步只上传力，无 HOST 当前坐标回填；完整多 world 原子发布、借用、reset/restore 和资源限制 | 只关闭节点生产者。M2-Q 查询、接触力、持续载荷、布弯曲和广泛体材料尚未接上；checkpoint 是 owner 绑定的内存对象，不是可移植文件格式 |
| R1/首 R2：直接球电机动态 | fixed BALL/HINGE/SLIDE 的 JOINT/FILTEREXACT 驱动可与原 normal/pyramidal 接触共同运行；同一次实际调用联合发布 activation、传动、广义力和运动，恢复后能真实续步 | nq≤10、nv≤8、nu/na≤32 的声明域；BALL BIAS_NONE。无 FREE+BALL、BALL affine bias、球限位/干摩擦、任意传动或完整人形/训练资格 |

多参与者查询失败只回滚该次 query publication，不倒滚已成功的物理步。未来 D3 的接触响应必须进入共同候选事务，不能将 accepted 状态后的独立查询冒充同一步物理耦合。

## 开发和源码验证

原三位作者在隔离树并行完成代码；父负责共享协议、runtime/receipt、借用同步及整合。所有 import、静态检查、pytest、构建、安装与消费者均由父唯一串行槽执行，每项启动前保存 fresh entrypoint-aware guard。

| 范围 | 解释 / HOST | CUDA | 确实受影响的旧入口 |
| --- | --- | --- | --- |
| 场景设备查询 | run014：41 passed，含 2 个 HOST 释放用例 | run015：39 passed，2 HOST-only deselected | run023/024：ordinary 与双 ordinary 场景，两后端共 4 节点 |
| BALL FILTEREXACT | run021：41 passed，含 12 个一次性准入负控；29 CUDA deselected | run022：29 passed，41 其它节点 deselected | run023：旧 stateless BALL、scalar FILTEREXACT，两后端共 4 节点 |
| 持久变形节点 | run036：44 passed | run033：33 passed，11 已覆盖 HOST 准入/预算用例 deselected | 原节点核直接复用；新测试逐步对照原设备核和独立 FP64 参考 |

以上最终选择均 RAW_EXIT=0；227 个新源码节点与 8 个受影响兼容节点只表示本批范围，不是总体完成比例。没有机械重跑所有历史模块。

保留的首失败与修复：

- run013：新设备 frame-translation 核中的 Python docstring 被 TYX lowering 拒绝；改为注释，原运算不变。最终解释/CUDA均通过。
- run030：测试错误要求无外载弹性边在第 12 帧仍有超过 1e-6 的残余应变；原独立递推允许松弛和过零。改为在原完整 12 步轨迹上验证最大边长变化超过同一 1e-6，并保首帧 h/24、逐步原核/FP64 对照。物理输入和阈值未改。
- run032：NEW 状态辅助核的多个 root parallel loops 触发 CUDA `mixed_root_effect`。合并为一个有界循环，原材料核及其算术不变。最终两后端覆盖该修复；run031 的早期解释通过仅为历史。
- 初始 import/format 失败完整保留。消费者审阅发现的 EFC-empty 漏检和返回后证据保存顺序已在首次消费者执行前修复；lease 退出在解锁前同步，异常时保留双错误并进入 close-only 状态。

run036 记录的 4463.03 秒只是该次命令 elapsed，未作为性能测量解释或用于与 Newton 比较。

## 一次统一 wheel 与六个安装消费者

从最终产品 HEAD 构建一个 wheel，安装到独立 Temp site。源码、wheel、installed 的 **520 个产品成员逐字节一致**；完整 RECORD/成员核对见原 parity receipt，不重复全量审计。消费者在源码目录外启动，使用固定安装根，检查全部加载的 Phymeo module origins；不导入源码 tests/examples，也不由 CPU 代跑 TYX 物理。

| 消费者 | 解释 / CUDA 运行 | 每后端实际内容 |
| --- | --- | --- |
| 直接 BALL 动态 | run043 / run044，PASS / RAW0 | 160 worldsteps；无接触、normal、pyramidal 各 24 步两 world，另有 8 次两 world 生命周期调用；同入步和连续 MuJoCo 独立参考、力/运动与双方 COM 力矩、reset/restore 后继 |
| 多参与者设备查询 | run045 / run046，PASS / RAW0 | 6 logical worldsteps、12 participant steps、4 queries、64 rows、3 stale 拒绝；两个稀疏 world、不同 anchor 的 ordinary 与 floating-affine 原资产/工具，完整查询及恢复 |
| 持久变形节点 | run047 / run048，PASS / RAW0 | 16 batch steps、32 worldsteps；12 步连续真实形变、checkpoint 两分支复现与 partial reset 后续步；0 CPU projection calls，逐步独立参考 |

六次首次安装消费者均成功，没有安装消费者修复/重跑。保留全部实际返回、独立参考、原输入、完整状态与失败前缀协议；原 gzip 按存储字节归档，不删字段、不重编码或全解压。

- wheel：`phymeo-0.1.0a0-py3-none-any.whl`，1,667,402 bytes，SHA256 `e39632eb88e4cafd9b1ebcdb5ed8916b380ce030d44e5a36f9ae029dd68baa30`。
- 证据根：`artifacts/p65-three-track-a-20261112-01`。源码命令、首次错误、快照、guard 和原始退出码原样保留。
- 安装归档：`installed-wheel-01`，143 个物理文件、8,196,524 bytes；只包含本批 wheel/scripts/results，源外原件保留。
- `installed-close.json` SHA256：`1346e1082524c170ff17f7fdfc07b345ee6ef6da95e33e6ef314eee46561c28c`，绑定六个 summary/bootstrap、完整存储清单与520成员 parity。
- `wheel-parity/summary.json` SHA256：`ca9ea81d60136d882da2d18d9505541cdaf0bc840e35413715b09b4f521f945c`。

## 资源和验收边界

D1 声明上限为 W8、N512、E1536、每 world 总 F128、200 迭代、substeps=1；有实际 buffer 账和 cap/cap−1 负控，但本批没有资格声称最大容量、吞吐或峰值 RSS。source loan 进入/退出均同步；不确定退出会 poison owner，不能继续正常使用未确认的设备状态。场景查询增加的共同框架转换也按数组/设备容量计费，HOST 诊断与读回仍存在，不称零 HOST。

细节见[场景报告](p65-resident-scene-boundary-report.md)、[节点报告](p65-deformable-bank-report.md)、[球电机报告](p65-ball-activation-report.md)。旧 G1 完整动作、J/R 学习、R03 与曲面 touch-SE 的原失败保持；本批没有新 Newton 性能比较、远程 CI、push/mainmerge 或发布。

## 下一批：从已运行基础继续连接

1. **S3**：收拢编译场景的自动接线及公共流程，增加与本批不同的工具/自由物体安装场景；保现有 scene query 接口与事务。
2. **D2/M2-Q**：在 D1 有效借用内用真实变形顶点更新 bounds/BVH，接原有限球/胶囊查询；验证真实面变化、稳定材点和旧 epoch 拒绝。D3 双向力与共同发布随后推进。
3. **R2**：先核 FREE+BALL 的树状态、DOF/actuator/constraint 组合与资源，冻结有限多案例，再实现第一格；异构小批只装已通过组合。功能矩阵建立后才作公平规模/性能比较。

本批作者与执行槽在最终收尾后释放。继续沿原三位作者隔离编码、父共享接缝单 owner 与唯一串行执行；不以本批局部通过宣布三条整线或整个阶段完成。
