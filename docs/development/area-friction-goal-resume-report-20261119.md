# 语言条件面积摩擦 PPO 的断点续训

两个单铰链拨板在 CUDA 上接受不同的中文推移指令，共用策略网络、价值网络和 Adam。每个世界每次更新采样 12 个物理步，步长 4 ms。首批 24 条样本完成一次更新后，训练器写出一个约 1.58 MB 的文件。新 Python 进程读取该文件，再用 24 条新样本完成第 2 次更新。[实验场景与指令](area-friction-goal-language-report-20261119.md)

## 如何判断恢复成功

| 对照 | 核验内容 | 本机结果 |
| --- | --- | --- |
| 进程退出后重新启动，与另一个进程连续训练两次相比 | 独立解析两个最终文件，分别比较策略、价值网络、Adam 各字段，六个策略输入字段的摘要以及 Torch CPU/CUDA 随机状态 | 全部相同。两个进程组的第 2 次价值损失均为 0.0008108167676255107。 |
| 关闭首个环境，用不同的初始化种子重建同构环境和训练器 | 将恢复后的下一批与原训练器连续运行的 24 条样本逐行比较，包括原始动作、对数概率、价值、原始和入账奖励，及步前、终端、重置后的全部策略输入 | 所有张量以零容差通过。第 2 次更新后两张网络的全部参数、13 组 Adam 状态和各项损失也一致。 |
| 连续窗口中的超时 | 在全局第 14、20、21 步分别核对第一个、第二个、第一个世界；读取重置前的指令和终端估值，再核对重置后的观察与入账回报 | 三次均与连续运行一致；更新后的最后两个原始动作为 −0.14052871、−0.05263095。 |
| 中断和拒绝路径 | 首次更新前保存、半个 rollout 保存、覆盖已有文件、损坏文件、缺失 Adam 一阶矩、过期目标版本、中文文件用于英文环境 | 拒绝操作。损坏文件、Adam 不完整与过期版本的测试还逐项核对了目标版本、学习器和 Torch RNG。 |
| 物理恢复后的晚失败 | 在安装随机状态时注入异常 | 错误带有物理已恢复标志。残留学习器拒绝采样，进程原有的 Torch CPU/CUDA 随机状态恢复。 |

观察的六组字段为物理状态、面积接触特征、动作掩码、文本指令、所有可见目标区及其有效掩码。其中接触特征含真实承载的非零样本。本批度量的是训练连续性，沿用上一批的任务奖励与成功判据。实测硬件为 RTX 4060 Laptop GPU，Torch 2.12.0+cu126，RSL-RL 5.5.2。跨设备、跨依赖版本和未完成 rollout 的续采仍在本次验收范围之外。

## 检查点保存哪些状态

文件包含原类型的环境检查点，恢复物理状态、目标、episode、接触读取位置和语言配置。学习器部分记录策略与价值网络、Adam 参数组与矩、更新次数、模块模式以及 Python、NumPy、Torch CPU 和本次 CUDA 设备的随机状态。文件绑定 PPO 配置、设备、运行库和重置前终端估值的记账规则。恢复时重新打包当前观察，校验它与保存帧的六个字段一致，再切换学习器和随机状态。

保存与恢复均在完整更新边界工作。文件按显式大小和 SHA-256 校验，以完整临时文件在同卷发布到新的目标名。当前实现复用已有的有界张量编码、随机状态和 Adam 校验。调用方独占环境、学习器及进程随机源；恢复需要与保存时兼容的构型、语言模型和设备配置。[训练器接口](../../src/phymeo/training/rsl_manipulation.py) · [磁盘格式](../../src/phymeo/training/manipulation_ppo_checkpoint.py)

## 复现

在仓库根目录运行。下面的两条 Python 命令各启动一个独立进程，使用本机已验证的语言模型。每个检查点输出文件名需尚未被使用。

```powershell
$env:PYTHONPATH = (Resolve-Path src).Path
$model = (Resolve-Path artifacts/models/paraphrase-multilingual-MiniLM-L12-v2/e8f8c211226b894fcb81acc59f3b34ba3efd5f42).Path
$python = ".\.venv-training-cuda313\Scripts\python.exe"
& $python -m experiments.area_pressure.train_goal_language $model --save-to .\paddle-update-1.ppo
& $python -m experiments.area_pressure.train_goal_language $model --resume-from .\paddle-update-1.ppo --save-to .\paddle-update-2.ppo
```

一次连续完成两次更新可用 `--updates 2`。本机运行记录位于 `artifacts/area-friction-goal-resume-20261119-01`。其中 03 至 05 是三个实际 CLI 运行，11 为独立进程结果比对，10 为 30 项原有检查点与 PPO 兼容回归。新增的两个 CUDA 测试覆盖完整续训与恢复失败边界。[示例入口](../../experiments/area_pressure/train_goal_language.py) · [逐步对照与故障断言](../../tests/test_manipulation_ppo_checkpoint.py)
