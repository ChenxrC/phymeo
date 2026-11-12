# 三主线 Batch B：实际交付与下一步

状态：**SOURCE_AND_INSTALLED_QUALIFIED（有限范围）**，2026-11-12。
产品快照 `0321c5a42abc1de1cb9b3f520e51a18e00a9960c`，后续只修安装消费者与文档，产品字节未变。
统一 wheel 的 522 个产品成员与源码、安装目录逐字节相同；六个源码外消费者全部通过。
没有新增 Newton 性能比较、CI 或整个 P6.5 完成结论。

## 本批可调用的能力

| 主线 | 新交付 | 实际资格与限制 |
| --- | --- | --- |
| S3 统一场景 | 编译后的 ParticipantScene 公共查询准备、生命周期及能力错误；自动复用原材料身份、owner 和设备来源 | 统一声明→真实 step→查询→reset/restore 后继→关闭。独立安装场景为带固定网格工具的 mimic 机械结构＋自由物体，补足 Batch A 的另一类场景。当前是同一 runtime 的有限边界查询，不是新接触力或 CCD |
| D2 / M2-Q | `TyxDeformableBatch.prepare_boundary_query`，公开 `DeformableBoundaryModel/Surface`；真实 accepted 顶点 gather、bounds/BVH、原球/胶囊有限距离与保守状态发布 | 三点片、共享边弹性方片、固定节点、多 world、反序、恢复、资源/晚失败。普通步不回填 HOST 当前位置。静态球/胶囊与变形来源共享 runtime；没有运动刚体双 owner 响应 |
| R2 FREE＋BALL | 单 FREE 根＋BALL/HINGE/SLIDE 后代，完整原生地址、四元数、耦合 M/J/bias、NONE JOINT motor CSR、typed receipt 与 checkpoint | 多轴/共享驱动、分支/嵌套、非零入步速度和正反卸载；nv≤35、nq≤45、nu≤32。无 FREE＋BALL contact/activation/limits/dry、BALL spring/affine bias、多个 FREE 根。容量正控不是规模或吞吐资格 |

共享事务改动保留完整状态：D1 loan 退出同步成功后才发布 query；query 失败不回滚已成功物理步。资源计数区分 owned native、borrowed bank 与真实 HOST 读回，不把估计数称实测。

## 执行证据

所有项目执行由父唯一串行槽完成，各命令前有 fresh entrypoint-aware guard。作者只编码/只读审阅。证据根 `artifacts/p65-three-track-b-20261112-01`。

| 验证 | 实际结果 |
| --- | --- |
| S3 源码 | `05` 解释/HOST 27 passed / 45.36 s；`06` CUDA 20 passed、7 HOST deselected / 89.75 s |
| D2 完整两文件 | `29` interpret 31 passed / 7.56 s；`30` CUDA 31 passed / 8.10 s |
| R2 完整两文件 | `22` 64 passed / 68.32 s，包含解释/CUDA与准入负控 |
| 受影响旧 ABI | `31` 16 passed / 69.82 s；无 BALL 浮根、kind3 normal/constraint、原 activation/stateless 和固定 BALL schema，保两个不支持组合负控 |
| 静态 | 首次 import/style 问题保留，修复后原 25 文件检查通过；新增测试与消费者末次检查 `32/33/44` 通过。没有调整物理门 |
| 统一构建/安装 | `34/35/36` RAW_EXIT0；522 产品成员 BYTE_IDENTICAL、0 issues。消费者只能加载源码外安装目录 |
| S3 安装 | `45/46`，每后端 8 logical worldsteps / 16 participant steps、5 queries / 60 rows、3 stale 拒绝、1容量拒绝及恢复后真实后继 |
| D2 安装 | `39/40`，每后端 16 batch steps / 32 worldsteps、19 queries；102 RESOLVED＋12 CONDITIONING 行；0 HOST projection 调用 |
| R2 安装 | `41/42`，每后端两结构共112 worldsteps；每结构27 main calls＋1 clone call，独立连续 CPU 和同入步参考、真实 restore/reset 后继 |

`47-close-installed` 收尾核对六个同次结果/消费者/来源/原始退出码；gzip只流式核原始 bytecount/SHA，无落盘解压或重跑物理。归档 `installed-wheel-01` 保147文件、8,109,911 bytes，含首次失败。`installed-close.json` SHA256：`46abb3de72d3b1563e7e6ca0cdbdb42ccea64939291cf8158bd032dfc5534188`。

`source-close.json`另绑定05/06/22/29/30/31六组源码执行：同次before/after清单、实际保存源文件及当前对应source/test一致，没有后续源码漂移。此收尾只做文件核对，没有重跑已通过组。

## 首次失败及处理

1. D2 `20`：新测试误把旧核斜向平行的 CONDITIONING 当错误。产品未改；仅原方片胶囊允许该状态，初态明确要求它，全部独立距离、间隙、材料见证、候选和误差门仍保留。未解析行没有法向，不能用于施力；三点片球/胶囊和方片球仍必须 RESOLVED。
2. D2 `27`：固定节点方片只度量第一条几乎静止的边。改为检查全部原材料边，原 `>1e-6` 门、12步、输入/力/质量均不变。
3. S3 安装 `37`：在物理零步时误把 authored-link 原始 OBJ 顶点与已施加固定工具变换的点比较。分别核原始顶点、独立固定位姿、group-local 和 world oracle；不改 URDF/OBJ/运动/产品。
4. 初始 Ruff import/format 问题保原输出；本地变量 `field` 遮蔽改名及限定 style 证明已保存。consumer 补强完整成员集合、安装来源、reset 选中状态及错误保留，未放宽物理判据。

## 下一批

- **D3 优先**：基于已验设备变形查询，明确接触层/法向/材料地址和合法法则，把节点内力、接触力与运动刚体放在同一候选状态和共同提交中。先覆盖压片、偏心载荷、运动胶囊推动片；未解析距离行须明确拒绝或由通用解析补足，不能用于虚构法向。
- **G1 可独立并行**：只组合已验机器人域，登记异构小批的分桶、状态/DOF/控制/力、reset/restore与末 world 故障；随后G2容量梯度。FREE＋BALL新接触/activation组合另立明确格，不删除guard来假装支持。
- S3当前公共查询目标已闭环；后续维护该入口并为D5变形响应装配提供接缝，不重写一套场景或另跑原已绿组。
- D4连续支撑/摩擦/卸载、D5安装组合、G2/G3同输出规模与Newton基线仍未完成。既有G1长动作、J/R学习、R03原失败保持。

详见[三主线路线](../superpowers/plans/2026-11-12-scene-deformable-scale-roadmap.md)、[差异全景](phymeo-newton-panorama-2026-11-12.md)及[S3](p65-participant-scene-boundary-report.md)、[D2](p65-deformable-boundary-report.md)、[R2](p65-free-ball-composition-report.md)报告。原三份未跟踪草稿、环境/pins、所有证据 Junction 原样保留；本批未 push。
