# 同一拨板根据文本目标参与 CUDA 共享训练

同一台单铰链拨板复制为两个世界。两边的机构、方块、重力、初始状态、动作通道、接触模型和可见目标表相同。第一个世界收到“把方块推到近处的蓝区”，第二个收到“把方块推到远处的橙区”。目标中心沿推移方向分别在 8 mm 和 18 mm。每行都看见两个区域，目标轮廓采用 PushT 评价器实际使用的方块半尺寸 20 × 20 mm。上一批核验的是不同工具共享更新。[双模型训练对照](area-friction-mixed-tools-ppo-report-20261119.md)

## 从指令到更新

建环境时，冻结的多语言 MiniLM 在 CPU 上将每条指令、改写和区域名称转换成 384 维单位向量，按文本去重后缓存在训练设备。每个候选区还提供世界坐标、朝向的正余弦及平面半尺寸。目标区来自场景声明。策略看到所有候选及有效位，不接收评价器选中的索引。形状不同的候选表按批次填充，打包时屏蔽无效槽。状态、物体侧接触力、动作掩码、文本与候选区共同进入 actor 和 critic。物理推进和 PPO 更新均在 CUDA 上执行。[编码与候选区](../../src/phymeo/training/goal_language.py) · [策略输入](../../src/phymeo/training/rsl_manipulation.py)

固定随机种子 720，两个世界各采样 12 步，每步 4 ms。中文实验得到 24 条已接受样本，81,155 个可训练参数共享一个 Adam 优化器，完成一次更新，梯度有限。这 24 条样本的原始任务奖励均为零。实验验证了语言条件能进入训练，尚无按指令完成推移的成功率证据。

| 对照 | 实际观察 |
| --- | --- |
| 固定初态，仅互换文本 | 近区和远区的初始动作均值分别为 −0.069297、−0.068873；交换文本后，两行的策略输出与价值估计也互换。 |
| 固定文本，仅移动可见区域 | 将一个世界的两个候选中心沿 x 移动 10 mm，该行的动作均值与估值均改变。这是对网络输入的干预。 |
| 两个世界先后超时 | 第 7、10 步分别使用原指令下的终端估值 0.197016、0.153802 计算超时回报；重置后改写指令的估值分别为 0.157034、0.149456。 |
| 打乱后更新 | 按实际采集的原始动作逐行还原 24 条训练输入，逐字段匹配状态、接触、指令、候选和掩码；两个网络的语言列与位置列权重最大改变量均约 0.0003。 |
| 不等长候选表 | 英文目录一行含两个区域，另一行只含一个，原顺序与有效位通过独立比对。另在无效候选槽注入 NaN，网络收到的对应特征全为零。 |

同一中文指令与对应英文译文的余弦相似度为 0.7544，与一条无关英文天气句的相似度为 −0.0136。此项只检查一个翻译对和一个负例。中文版单世界换目标、其余世界保持原指令、环境检查点恢复也已通过。实际任务条件包含两个有部分重叠的目标轮廓，当前对照没有测量策略对陌生语言、陌生机构或互斥目标区域的泛化能力。[可执行断言](../../tests/test_manipulation_goal_language.py)

## 本地复现

使用固定版本 `e8f8c211226b894fcb81acc59f3b34ba3efd5f42` 的 `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`。目录需要 `tokenizer.json` 和 `onnx/model_quint8_avx2.onnx`，共 127,535,388 字节。加载时逐文件检查上游 SHA-256。模型文件保存在本机忽略目录，代码可从任一含这两份文件的目录离线运行。本批使用已有的 ONNX Runtime 1.26.0 和 tokenizers 0.22.0。[固定版本的模型文件](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2/tree/e8f8c211226b894fcb81acc59f3b34ba3efd5f42)

在仓库根目录执行：

```powershell
$env:PYTHONPATH = (Resolve-Path src).Path
$env:PHYMEO_MINILM_DIR = (Resolve-Path artifacts/models/paraphrase-multilingual-MiniLM-L12-v2/e8f8c211226b894fcb81acc59f3b34ba3efd5f42).Path
.\.venv-training-cuda313\Scripts\python.exe -m experiments.area_pressure.train_goal_language $env:PHYMEO_MINILM_DIR --language zh
```

`--language en` 选择英文指令与区域名称。独立英文入口也运行成功，24 条样本完成一次 Adam 更新，价值损失 0.000929。[可运行的双目标案例](../../experiments/area_pressure/train_goal_language.py)

## 空接触回归与验证

原双爪训练回归曾在接触力复核处失败。诊断记录捕获了一个具体状态：世界 41 的首步三个接触对各有零个积分点，设备广义力为八个零，主机复核的累加值却全为 NaN。失败后重新计算各个空缩并返回零，数值库内部成因尚未确定。新回归在卸载到空接触的路径捕获到原复核确实对空样本执行缩并。修复后，零积分点的接触对对广义力和独立接触律各贡献零；所有接触身份、时间戳、数组和原生关节力仍经过原检查。误差门槛保持原值。[接触收据复核](../../src/phymeo/backends/mujoco_area_friction_state.py) · [卸载回归](../../tests/test_mujoco_area_friction_live.py)

最终串行联合回归 40 项通过（62.69 秒），覆盖卸载、分离时仍有原生关节力、旧双爪 PPO、异构工具、环境检查点以及中文语言条件 CUDA 训练。英文不等长候选表和翻译对照另有单独验证。原始结果保存在 `artifacts/area-friction-goal-language-20261119-01`，其中 `14-compat`、`15-compat-diagnostic`、`16-empty-red` 保存修复前记录，`19-empty-regressions` 保存最终联合测试，`20-english-demo` 保存独立入口结果。

下一批在一次 PPO 更新结束时保存环境、文本配置、两张网络、优化器及随机状态。在新会话恢复后，与连续运行逐项比较下一批采样的动作、超时估值和下一次参数更新，再扩展到八种构型。[共享训练计划](../superpowers/plans/2026-11-14-language-conditioned-gpu-manipulation.md)
