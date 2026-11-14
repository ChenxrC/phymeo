# M2-B1：公共 GPU 端点 body 观察

日期：2026-11-14。产品提交：`f1c3d9a0246fd4caf0211bdabdac25218fe9d769`。
[实施计划](../superpowers/plans/2026-11-14-language-conditioned-gpu-manipulation.md)；
[公共用法](p65-heterogeneous-manipulation-io-usage-20261114.md)。

## 交付

`batch.prepare_device_workspace(include_bodies=True)` 在同一个公共 frame 中增加：
`body_position [W,B,3]`、`body_rotation [W,B,3,3]`、
`body_linear_velocity [W,B,3]`、`body_angular_velocity [W,B,3]`。
均为 FP32，`body_mask` 为 int32；有效 body 包含 world/fixed body，padding 全零。
位置/线速度参考 body origin，旋转矩阵将 body 轴转为世界轴，线/角速度均为世界轴分量。
原生 FREE qvel 中的局部角速度经现有 FK 转换；没有将局部角速度直接标为世界角速度。

全部输出从 accepted q/v 重算，与 q/v/act/clock 共同发布为 accepted integration endpoint。
观察器借用只读静态模型数据和 accepted bank，在自己拥有的 scratch 内调用现有 tree/floating geometry kernel；
仅新增 padding/搬运及逐分量 finite 检查，不另写求解器或 FK 算法。
默认 `include_bodies=False` 保持原 native-state 输出，不分配 body scratch。
T0 scalar/FREE 复用提取后的静态 frame/address lowering，不重复 solver 准入。

所有桶成功后才换 frame；最后桶或非有限 body 输出失败保留旧 frame 和物理状态。
reset/restore 后重新解析当前 owner/device，不缓存已退休物理缓冲。
Torch 输出继续是独立 D2D 副本；工作区数组进入既有 alias/close/UNKNOWN_RELEASE 生命周期。
工作区分配与读回计数单列，未将其纳入 solver-bank 总预算，也不声称完整进程显存预算。

## 实际验证

父 /root 是唯一串行项目执行 owner，每条命令有 fresh entrypoint-aware guard。
现有 Python 3.13.3、MuJoCo 3.3.7、NumPy 2.4.6、pytest 9.1.1；
TYX 0.2.0a1 来自已验 site313，Phymeo 来自当前父 src。
未安装依赖、改 pin、执行新 wheel、远程 CI 或全仓库套件。

| 范围 | 结果 |
| --- | --- |
| 新 B1 解释路径 | run09：6 passed / 40.28s |
| 新 B1 CUDA | run10：8 passed / 32.77s |
| 受影响兼容 | run11：58 passed / 193.56s；原 M2-A 完整38项、偏心单摆2项、FREE+BALL九类24步18项 |

14 个新节点覆盖：

- 六类构型、稀疏 world IDs、前后排列、W8 与独立 W1；包含 scalar、fixed R/P、fixed BALL、FREE+BALL、非首地址多 FREE。
- 原生 FK 位姿及独立 body-origin Jacobian 乘 qvel，显式检查偏心 body 的 origin/COM 差异，以及 FREE 局部/世界角速度差异。
- 原输入阶段 dynamics 与 accepted endpoint 确实不同；查询禁用 CPU dynamics/FK、NumPy solve/inv、HOST snapshot。
- 查询前后每个物理 allocation manifest buffer 的完整字节保持，包括 scratch；最后桶失败、非有限旋转输出与 retry。
- 同桶部分 reset、checkpoint/restore、下一步逐字节重放；有状态执行器 wrapper。
- CUDA Torch 独立所有权、旧 frame 失效，以及八份原 M0 场景各两步的 body 原生对照。

位姿沿用原浮根门 rtol=3e-5 / atol=4e-5，速度沿用已有 Jacobian 组件门 rtol=2e-4 / atol=5e-5，均在执行前写入。
MuJoCo 独立 oracle 使用 [mj_jacBody](https://mujoco.readthedocs.io/en/3.3.7/APIreference/APIfunctions.html#mj-jacbody)，并与 `mj_jacBodyCom` 区分参考点。
八场景两步只检验公共接口，不增加抓取/放置/PushT 成功声明。

## 保留记录

原作者单测试提交 `e06365f1` 导入为 `c7d7042c`；父补 activation/nonfinite 两个后端节点。
run03 首 source format-check、run05 首 test import-order、run06 首 test format-check 失败原件保留；
run04、07、08 仅修复样式。14 新功能节点首次执行全部通过，无数值门/fixture 调参或产品功能修复。
run12/13 最终7文件 lint/format 均通过。合计72个不同功能/兼容节点通过。
同次证据收据核对7个交付源与5次命令的before/after快照全部相同；32个新保存文件共4,888,280字节原样归档，无解压改写。
`final-evidence.json` SHA256：`ef416eee2dcca67ef5bde8c30f34e19c6657516270106a7b2d3649255d09ab7a`；执行槽已释放。
证据位于 `artifacts/mujoco-tyx-body-observation-20261114-01`：
原命令、环境来源、source-before/after、stdout/stderr、raw exit、保存结果和 final evidence receipt。
现有旧测试资格仅按受影响范围复验，未重跑旧完整验收或重审旧归档。

## 下一连接

B1 是端点观察；`contact_tokens=False` 保持。现有 contact rows/motion 是可被后续失败 prepare 覆盖的临时区，
且 reset/restore 不保证保留过去入步数据，不能直接作为公共接触观察。

M2-B2 先在叶子 device 增加 accepted-entry 持久双 bank：
在 prepare 内 D2D 保存本步真实入步位姿/速度、contact 身份/点/坐标轴/实际 wrench 与 input/output tick、补偿时钟，
随全桶物理事务一起接受。无接触成功步骤是有效空集合，与 reset 后无记录区分。
失败不换已接受记录；selected reset 只清所选 row，checkpoint 保存/恢复完整记录。
workspace 只 scatter 已接受记录；不能用 HOST receipt 重建再上传。

T0/T1 固定 rows、T3 normal_rows、T4 contact_map/contact_wrench 需分别核对并统一，
尤其不得把 T4 contact index 当 slot index，或把 limit/dry 约束行当接触。
双侧力与力矩须明确世界轴及参考点；SuccessFrame 使用同一 entry 内的状态与载荷。
B2 生命周期和相位闭合后才做 B3 Env 的 goal/episode/reset/final observation/checkpoint。

M2 整体、T1 真实操作可控性、L1 预训练语言编码、L2 共享策略与 T2 学习结果仍未完成。
本批只是源码资格；没有新安装包、性能胜出、零 HOST 往返、融合异构 kernel 或训练成功声明。
