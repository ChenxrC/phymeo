# N1 策略连接准备与并行合同冻结

2026-11-11。**STATIC_VERIFIED_RUNTIME_PENDING / PREINSTALL_BLOCKED_DISK**。本批交付代码、静态检查和两条独立开发合同；没有新增 CUDA 物理或策略运行验收。上一批物理/安装包证据继续复用，不重跑。当前新代码源 `872473aa0da8740154daff5830e680ed119fea45`，旧 `src/` 物理代码未改。

## 实际交付

| 方向 | 已完成 | 尚未完成 |
|---|---|---|
| N1 GPU 策略 | [公共示例](../../examples/mujoco_tyx_cuda_policy.py)、[focused 测试](../../tests/training/test_mujoco_tyx_cuda_policy.py)、[固定案例表](../development/mujoco-tyx-cuda-policy-n1-cases.md)；候选 CUDA Torch profile | 新环境安装、真实 CUDA Torch/TYX 互操作和全部新案例运行；源码外调用 |
| ROB1 | [多参与者合同](../superpowers/plans/2026-11-11-resident-multi-participant.md)父审阅冻结；身份、完整 J/载荷、事务、六组案例与独占文件 | ROB1a layout/schema 及后续共同响应实现 |
| COL1a | [十八类设备查询合同](../superpowers/plans/2026-11-11-device-convex-support-family.md)父审阅冻结；七族矩阵、通用支持函数及 GJK/EPA 复用 | 新族设备查询实现、interpret/CUDA 矩阵；曲面 patch/受力为后继 |

N1 作者提交 `eaaba58c`→`1ca1e790`→`8e6ab93f`→`eaedf4b2`，父分别导入 `f8d1c40d`→`e573ff26`→`0d3c80f5`→`90d3de5f`；父 `872473aa` 仅整理两个 Python 文件格式。ROB1 作者 `230c0258` 导入 `dd29546b`；COL1a 作者 `dd3bbd4a` 与 oracle 修正 `7625c874` 导入 `759e2656`、`4470ecca`。三个原作者均已交接释放，没有新助手或并行项目执行。

N1 固定两个模型：hinge-slide W1/W8/W32、有限支撑 slide W8，均覆盖默认/非默认 CUDA stream，连续八个真实物理步。MLP 输出三类原生控制，采用同一次 FP32 输出喂给独立连续的 TYX CUDA 对照；逐步比较完整状态、收据与控制，保留所有 HOST 诊断/复制计数。CPU 对照入口只负责控制提交，不是 CPU 物理 oracle。预登记 23 个测试身份是手工数量，未 collection；其中真实第二 GPU 负控在单卡机器上不能获得资格。

ROB1 复用原柔顺接触模型：同一入步状态生成跨 participant 载荷，各机械块完整求解一次，全 world 一次发布。首域 W1/W2/W4、1–8 participant/world；不是跨树未知接触力的隐式 KKT。COL1a 首片仅查询，新曲面 patch 返回明确 UNSUPPORTED；不能把几何通过升级成受力资格。浅交负例的轴向布置重叠量不等于一般最小穿透深度，逐族独立 oracle 在 tests-first 中固定。

## 父唯一串行静态验证

原始文件：`artifacts/p65-cuda-policy-n1-20261111-01`。每次实际 Ruff 调用有精确两路径、fresh 进程门、源前后快照及 raw exit。

| 任务 | 实际结果 |
|---|---|
| 01 first Ruff | RAW_EXIT 0 |
| 02 format 预检 | 等待外部 Python 时 CIM 报内存不足；格式程序未启动，无 RAW_EXIT |
| 03 first format-check | RAW_EXIT 1，两个文件需要格式整理 |
| 04 format repair | RAW_EXIT 0，仅两文件格式整理 |
| 05 final Ruff | RAW_EXIT 0 |
| 06 final format-check | RAW_EXIT 0，两个文件已格式化 |

02 原始情况保存在 `02-first-format-check/guard-failure.json`，没有改写成产品测试失败；任务本地 guard 改为仅请求所需四个进程属性，03 等外部执行自然结束后启动。未 kill 任何外部进程。格式 diff 全文审阅为排版；未声称执行过独立 AST 比较。

最终工作字节 SHA256：示例 `66e588fa7f4b0ec6f9934efa2ad743e8694923c87f84e3e4b83e2fd0e81a8c44`；测试 `8b8a946b0e99ef6297f050a2aab0b04d523993b8cca5c11b5d55886ec1bbd765`。05/06 后快照与最终文件相同。`static-and-release-summary.json` 保存原命令、时刻及最终绑定；10:43 fresh matching=0，父执行槽释放。没有项目 import、pytest、CUDA Torch、build 或安装结果。

## 环境阻塞与下一步

[候选环境预检](../development/mujoco-tyx-cuda-profile-qualification.md)选官方 Torch 2.14.1+cu126、CPython 3.11 Windows wheel，保留原 TYX pin、CPU 环境及全部验收证据。仅读取 HTTP HEAD 和 ZIP 中央目录元数据，未下载 wheel 内容。

wheel 压缩大小 2,602,753,280 bytes，展开 4,191,672,496 bytes；同时留存的下界 **6,794,425,776 bytes**，尚不含依赖/临时文件。最初 C 盘约 2.01 GB；10:43 变为 **6,195,863,552 bytes**，后台变化未归因，本批未删除文件。仍低于该下界，故未启动安装。建议预留 8–10 GB 后再按完整依赖解析核峰值，建议值不是实测安装占用。

后续可直接并行 ROB1a 与 COL1a 的隔离纯编码，父接共享服务/绑定并串行验证；不依赖 N1 安装完成。N1 待足够空间后恢复 fresh 预检→独立环境→互操作→固定案例→源码外调用。N1 通过前不启动 N2 PPO，也不以增加迭代或换 CPU 路径绕过资格门。整个 P6.5/对 Newton 功能补齐均未完成。
