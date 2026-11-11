# Python 3.13 CUDA 环境复用与 P65 清理

2026-11-11。用户授权检查已有 Torch，并清理不再需要的 P65 调试环境。现有 Torch 可执行 CUDA 运算，予以保留；原 CPU 训练、G1、Newton 对照及物理运行环境均保留。

## 已执行的清理

只删除下列五个明确退役的虚拟环境。删除前核对实际绝对路径、无 reparse/junction、无 Git 跟踪文件、无活跃消费者，保存 pyvenv.cfg 和所有已安装 distribution 的 METADATA/WHEEL/direct_url。源码、测试结果、验收证据、最终构建、依赖 pin 均未删除。

| 相对位置 | 删除的逻辑字节数 |
| --- | ---: |
| 原交付仓库 `.venv-docs` | 114,350,484 |
| 原交付仓库 `.venv-material-ledger-nativepip` | 114,407,633 |
| 原交付仓库 `.venv-wheel` | 44,716,946 |
| 原交付仓库 `.venv-mujoco` | 83,505,430 |
| 父仓库 `artifacts/p65-connections-20261110-0800-01/63-device-policy-venv/venv` | 132,725,255 |

原交付仓库为 `C:/Users/38347/Documents/ChatGPT/phymeo`；父仓库为 `C:/Users/38347/.codex/worktrees/p65-parallel-integration/phymeo`。合计 19,166 文件、489,705,748 逻辑字节。清理事务前后 C: 可用空间从 23,671,390,208 增至 24,152,932,352 字节，观察差值约 **0.48 GB**。全盘背景活动使空间继续变化，不把此前约 19 GB 的背景增长归功于本次清理。

保留原交付仓库 `.venv`、`.venv-py311`，父仓库 `.venv-humanoid`、`.venv-newton-1.7.0`、`.venv-training`。旧 Temp 证据 Junction 和 canonical archives 不动。证据中的 `original-environment/pyvenv.cfg` 是历史环境记录，不当作可删虚拟环境。

## 已有 Torch 的实际资格

使用 `C:/Python313/python.exe` 3.13.3，以 `-s` 禁止自动加载用户 site，再显式加入 `C:/Users/38347/AppData/Roaming/Python/Python313/site-packages`。其中 Torch **2.12.0+cu126** 已在 RTX 4060 Laptop GPU 上完成非默认 stream 的 4×4 矩阵乘法及结果核对，run01 RAW_EXIT=0。没有加载用户目录内的 editable `.pth`。

该结果证明已有 Torch 可以复用，尚不等于 RSL 优化器或 Phymeo 物理闭环通过。用户 site 的 MuJoCo/RSL/TensorDict 版本不同于当前合同，TYX 的 editable 来源也没有原 99f6 pin 的绑定，因此不直接把整个用户 site 当成项目运行环境。

## 新的隔离 profile

父仓库 `.venv-training-cuda313` 使用 Python 3.13.3。自身 site-packages 优先；通过一个仅含路径的 `.pth` 借用原用户 site 的 Torch 和依赖，不执行被借目录的其它 `.pth`，不复制约数 GB 的 Torch。

自身安装 NumPy 2.4.6、MuJoCo 3.3.7、cuda-python/cuda-bindings 12.9.10、cuda-pathfinder 1.8.4、nvidia-cuda-runtime-cu12 12.9.79、RSL 5.5.2、TensorDict 0.15.5、pytest 9.1.1。run02 建环境和 run03 安装均 RAW_EXIT=0，原用户目录的软件未卸载或升级。TYX 从已验物理环境复制纯 Python 包及 provenance，逐文件字节匹配，保持 `99f6df7c0fd8fdeef77f97df83914da27b64b7ee`；CUDA bindings 使用新 cp313 wheel。

这是本机复用 profile，不是自包含可移植环境；原用户 Torch 目录是显式依赖，不能在后续清理中删除。项目运行需 `PYTHONNOUSERSITE=1`、`PYTHONDONTWRITEBYTECODE=1`、显式父 `src`、单线程与 pytest no-autoload。后续须先通过完整 import/来源/CUDA driver/NVRTC 门，再运行 N1 固定连续物理矩阵。此处不提前宣布 N1 或 PPO 通过。

原始命令、环境 metadata、清理事务、源包复制清单及首次失败保存在父 `artifacts/p65-env-refresh-20261111-01`。run04 的 PowerShell 文件数组拼接错误发生在 Python 启动前，原记录保留；run05 修正任务描述，产品源码和测试未因此修改。
