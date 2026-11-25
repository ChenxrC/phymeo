# 用 S4000 训练带双腕相机的异构机械臂

这组实验分别训练重物搬运和 PushT。每台机械臂有六个或七个受限力矩驱动的转动关节。每个场景共用一个自由刚体、一张工作台与一套动力学求解器。策略从左右腕 RGB 和自身关节观测产生动作。位置、接触力、目标误差由独立评分器读取。[预登记目标和成功条件](s4000-bimanual-wrist-rl-plan.md)

| 任务 | 已验证的策略包 | 两种构型实际轨迹的 36 秒录像 |
| --- | --- | --- |
| 6 kg 侧缘箱，抬起后运输和放置 | [搬运 CEM](../../artifacts/mtt-bimanual-20261126/carry-policy.zip) · [带视觉神经网络的搬运模型](../../artifacts/mtt-bimanual-20261126/carry-neural-policy.zip) | [搬运 CEM 回放](../../artifacts/mtt-bimanual-20261126/carry-studio.mp4) |
| 9 kg T 件，平移并对准目标朝向 | [PushT CEM](../../artifacts/mtt-bimanual-20261126/push-t-policy.zip) · [视觉神经网络实验包](../../artifacts/mtt-bimanual-20261126/push-t-neural-policy.zip) | [PushT CEM 回放](../../artifacts/mtt-bimanual-20261126/push-t-studio.mp4) |

在已装好 Phymeo、MuJoCo、Tyx MUSA 与 Vulkan 渲染库的 S4000 环境中，将任一 CEM 包解压，在解压目录运行以下命令。将 `PHYMEO_RENDER_LIBRARY` 指向该机实际构建的 `libphymeo_native.so`。模型包内附精确版本的控制、观测、场景生成源码和训练检查点。新回合的 XML、真实状态轨迹和结果写入 `my-replay`。

```bash
python bimanual_policy_replay.py --policy . --source-dir ./sources --seed 55001 55002 \
    --out ./my-replay --library "$PHYMEO_RENDER_LIBRARY" --device-index 0
```

## 在设备上运行

安装与渲染分支相同版本的 Phymeo、MuJoCo 3.3.7、固定版 Tyx 和 Phymeo Rendering。指定 S4000 对应的 Vulkan ICD，并将包含着色器资源的本机渲染库路径传给 `--library`。训练画面为左右各 96 × 96 的 RGB。录像从真实状态轨迹再次渲染，可使用更高分辨率和更多面积灯采样。[MUSA 安装与动力学验证](mujoco-tyx-musa-slider.md)

以下是首代实验的 Linux 命令。每个任务使用独立输出目录。每次 CEM 更新在预先选定的训练种子上按完整回合回报排序候选策略，再更新控制参数的采样分布。候选世界共享同一个物理模型，在 S4000 上成批积分；腕部画面也由 S4000 生成。图像姿态估计、动作计算和这个小规模参数分布的更新在 CPU 完成。

```bash
common=(--mode musa --library /path/to/libphymeo_native.so --device-index 0
        --cache-dir ./training-cache --image-size 96 --action-repeat 8 --horizon 200
        --policy-variant registered --population 4 --generations 1 --train-seeds 2 --batch 4)
for task in carry push_t; do
    python examples/bimanual_musa_train.py "${common[@]}" --task "$task" --stage train --out "./runs/$task"
done
```

每个训练目录包含 `checkpoint.json`（采样分布、种子、候选回报、更新历史），训练与选模完成后写入 `frozen-policy.json`。恢复同次中断的训练使用原参数并加 `--resume`。运行 `--stage evaluate` 会重新读取冻结权重和源码绑定，输出各留出场景的 JSON、模型 XML 和记录物体及关节真实状态的 NPZ。目标完成率必须从冻结评估取得。录像中的慢动作与真实仿真时间分别标注。

## 第一轮实测

S4000 上两个独立关节树通过真实接触向自由载荷传力的回归通过。两项完整场景各用 12 步进行起始状态核验，S4000 与独立 CPU MuJoCo 的最大广义位置分量差均小于 `6e-9`，速度分量差均小于 `3e-7`。两腕观察实际使用同一张 S4000 的 Vulkan 着色器阴影和 4× MSAA。动力学每个场景完成 108 次内核启动。

在开发种子 701，每项任务用 1,600 个 2 ms 物理步测试首个未训练控制器。搬运时双手同时接触 2.232 s，峰值向上合力 46.39 N，物重 58.86 N，未成功起抬。PushT 最终位置误差 9.17 mm，偏航误差 19.99°，超过 6° 的验收门槛。仅看位置会把这次推 T 误判为成功。

## 9 kg 协同 PushT

冻结后的双臂策略在预登记的四个留出场景全部完成平移与转向。条件相同的未训练先验完成 1/4，遮黑两腕、只启用左臂、只启用右臂各完成 0/4。将交付的策略 ZIP 解压到新目录，用自带的回放入口在种子 55001、55002 上重新运行，两次均成功。

在另一项配对实验中，固定种子 52001 的机器人构型、工件与初态，只将目标从 −15° 旋到 +15°。冻结策略两次均成功，最终工件偏航相差 20.70°。两腕都停在首帧时两次均失败，最终偏航只相差 1.82°。遮住任意一只腕相机时仍能完成两种转向。这组对照表明持续更新的腕部图像对该策略有实际作用，单个视角也能提供足够信息。

独立的单臂强推实验在种子 701 上用相同的 9 kg 工件和原电机限矩。单臂饱和推送时，左右臂分别使工件移动最多 152 mm 和 162 mm，两次均未完成指定终态。若要检验工业推杆的额定持续推力，需另外给出可测量的推力上限和持续直推对照。

策略包内包含冻结权重、训练检查点、对应版本的 Python 源码和四组留出轨迹。录像展示种子 43001 和 43002，各用一个完整回合。总长 36 秒，1080 帧，H.264，1352 × 828。影片保留仿真时间与各片段的慢放倍率。[策略包](../../artifacts/mtt-bimanual-20261126/push-t-policy.zip) · [完整回放](../../artifacts/mtt-bimanual-20261126/push-t.mp4) · [留出结果](../../artifacts/mtt-bimanual-20261126/push-holdout-v1.json) · [配对视觉对照](../../artifacts/mtt-bimanual-20261126/push-target-and-wrist-audit.json)

[工作台版 PushT 录像](../../artifacts/mtt-bimanual-20261126/push-t-studio.mp4) 使用相同的两段轨迹。总览增加了底座、工作台、柜体与金属材质。左右腕画面和仿真时钟沿用上面的回放。

补做的额定推力实验把每个直线推杆的电机上限设为 28 N。在同一 S4000 初态与 9 kg 载荷下，左臂或右臂单独持续直推的最大位移均为 1.06 mm，双推杆移动了 138.09 mm，同时承载 2.1995 s。双推杆的逐步台面反力有过零值，严格的全程台面支撑检查失败。这项试验衡量额定工具的直推能力。上面的腕部视觉策略使用普通工具，还需要同时控制平移和转向。[额定推力实测](../../artifacts/mtt-bimanual-20261126/rated-musa02-summary.json)

## 6 kg 双臂搬运

记录中需区分两种箱体。原始平侧箱使用两只夹具从侧面夹紧。另一种箱体保留 12 × 22 × 10 cm 的主体和 6 kg 的总质量，增加左右各 32 mm 的侧缘，双腕可从缘下共同托举。两种配置都使用有限矩的关节电机和可自由运动的载荷。在侧缘箱实验中，控制器从两路腕部 RGB 识别箱体及目标标记，从腕部反力判断是否已经承载及是否可以松手。

开发种子 701 的侧缘箱在 2.544 s 内完成整段动作。最高离桌 70.51 mm，双腕同时承载的离台时间为 1.838 s，放下后中心距目标 1.90 mm，桌面承担 58.86 N，两腕接触力都降至零，终态稳定 0.10 s。运输阶段用两腕共同的平移积分补偿俯仰调整带来的持续偏差。这条轨迹验证控制与物理流程，策略成绩另取冻结后留出结果。[开发回合](../../artifacts/mtt-bimanual-20261126/carry-tote-smoke10.json)

搬运训练在训练种子 1001–1004 上完成两代 CEM，每代六组候选。按两个选模场景的平均回报选中训练后参数分布的均值。冻结模型在四个留出场景全部成功，未训练的阻抗控制先验同样完成四次。本组成功率支持该控制结构能完成任务。七个控制增益的学习确有更新，这四次留出不能证明学习使搬运成功率进一步提高。双腕遮黑、单用左臂、单用右臂各为 0/4。[训练检查点](../../artifacts/mtt-bimanual-20261126/carry-training-v10.json) · [冻结留出评估](../../artifacts/mtt-bimanual-20261126/carry-holdout-v10.json) · [含精确源码的策略包](../../artifacts/mtt-bimanual-20261126/carry-policy.zip)

| 留出种子 | 左、右轴数 | 真实仿真时长 | 最大离台高度 | 放置中心误差 |
| --- | --- | ---: | ---: | ---: |
| 43001 | 6、7 | 2.544 s | 70.53 mm | 3.65 mm |
| 43002 | 7、6 | 2.560 s | 70.46 mm | 1.76 mm |
| 43003 | 6、7 | 2.576 s | 71.45 mm | 0.31 mm |
| 43004 | 7、6 | 2.544 s | 70.98 mm | 1.79 mm |

所有终态的工作台反力均约为 58.86 N，手爪接触力均为零。以下参数可以复现侧缘箱训练配置；重放已发布权重请使用压缩包内的加载器及冻结源码。

```bash
python examples/bimanual_musa_train.py --task carry --stage train --mode musa \
    --policy-variant fiducial --tool-profile twin-pad --load-profile side-lip --stance bent-elbow \
    --action-repeat 8 --horizon 400 --population 6 --generations 2 --train-seeds 2 --batch 6 \
    --image-size 96 --device-index 0 --library /path/to/libphymeo_native.so --out ./runs/carry
```

[36 秒工作台版搬运回放](../../artifacts/mtt-bimanual-20261126/carry-studio.mp4) 同屏显示外部镜头和两个腕部镜头，每次真实仿真约 2.55 秒。高清腕视角按记录的姿态重渲，策略决策使用的分辨率为 96 × 96。[原镜头和轨迹时间表](../../artifacts/mtt-bimanual-20261126/carry-film.json)

另用两组开发构型，对左右臂各跑 6.4 秒单臂跟踪搬运。四次均接触到工件，停放的另一只手全程零接触。单臂峰值向上承载为 46.55–57.21 N，最大离桌仅 0.13–0.18 mm。另四次强制饱和电机的激进试验均未达标，不过被撞开的箱体碰到了停放的另一只手，不能把这四次计入独立单臂能力证明。[单臂实测与原始轨迹数据](../../artifacts/mtt-bimanual-20261126/carry-single-strong10.json)

将搬运交付包解压到新目录，用其中的模型和源码在 S4000 上重新运行种子 55001、55002。两个新回合均完成托起、运输、落台和卸载。[从交付包重载的回执](../../artifacts/mtt-bimanual-20261126/carry-reload10.json)

另固定种子 52001 的工件和两台机械臂，只将放置区向左、向右各移动 40 mm，使两个合格区域互不重叠。双腕实时观察时两次都放到目标，箱体落点相距 **81.44 mm**，与要求的 80 mm 一致到 1.44 mm。保持首帧、遮住左腕、遮住右腕、遮住双腕这四组对照都未完成任一目标，每组两个落点的横向差均为零。这一结论对应侧缘箱、当前镜头布局和本次配对实验。[两目标、五种视觉条件的原始结果](../../artifacts/mtt-bimanual-20261126/carry-target-and-wrist-audit.json)

## 双腕图像神经策略

在冻结的七参数策略上叠加一个小网络。它从实时双腕 RGB 提取空间分块的颜色和色差，拼接关节位置、速度和有效轴掩码，经固定的八维非线性编码器产生四项动作修正。每项任务分别用三代成对的正负扰动对比整回合回报，训练 32 个输出权重。经验来自 S4000 的 MUSA 物理步和 Vulkan 双腕曝光，扰动、权重更新和推理在 CPU。六个训练种子与两个开发种子均独立于下表的八个留出种子 63001–63008。

| 冻结后使用的输入与权重 | 6 kg 侧缘箱 | 9 kg PushT |
| --- | ---: | ---: |
| 原 CEM 控制器 | 8/8 | 5/8 |
| CEM 加已训练视觉网络 | 8/8 | 4/8 |
| CEM 加相同网络的初始权重 | 8/8 | 8/8 |
| 整个策略只能看到首帧 | 0/8 | 0/8 |
| 整个策略的腕图均遮黑 | 0/8 | 0/8 |
| 仅网络的腕图遮黑，原视觉控制器仍能看到 | 8/8 | 4/8 |

搬运网络在这八次共改变了 1,264 个决策，PushT 网络改变了 1,204 个。搬运能从训练后的模型完成任务，成功率与原 CEM 相同。PushT 的这次网络训练降低了留出成功数。两项的遮挡对照均支持整套策略依赖实时视觉。单独遮黑神经分支后的成功数保持不变，尚无该分支独立提供必需视觉信息的证据。[搬运新留出](../../artifacts/mtt-bimanual-20261126/carry-spsa-holdout.json) · [PushT 新留出](../../artifacts/mtt-bimanual-20261126/push-spsa-holdout.json)

在四个另留的校准种子 64001–64004 上，把训练后网络的输出依次缩放至 0、0.25、0.5、1 倍，按成功数、平均回报、较小幅度的顺序选定，再冻结并打开 66011–66018 八个新种子。搬运选择 0 倍，在新组完成 8/8。PushT 选择 0.5 倍（校准 3/4），在新组完成 5/8；同组原 CEM 为 6/8，同幅度的初始化网络为 7/8。减小残差仍未带来可复现的胜率优势。[搬运选择与新留出](../../artifacts/mtt-bimanual-20261126/carry-selected-holdout.json) · [PushT 选择与新留出](../../artifacts/mtt-bimanual-20261126/push-selected-holdout.json)

## 交换两条机械臂后还能否完成同一个目标

原训练按种子奇偶指定左右轴数，PushT 的目标转向也依赖奇偶。为分开这两个因素，在 70001–70008 八组新工件和目标上，各自跑一次左六右七和一次左七右六，打乱执行顺序。每对都核对工件初态、目标、工具与腕镜的初始空间位姿。七轴模型实际多一段有质量的连杆和一个限矩 8 Nm 的驱动铰链。

| 原冻结 CEM | 左六右七 | 左七右六 | 合计 |
| --- | ---: | ---: | ---: |
| 6 kg 侧缘箱 | 8/8 | 8/8 | **16/16** |
| 9 kg PushT，目标 −15° | 2/4 | 4/4 | 6/8 |
| 9 kg PushT，目标 +15° | 4/4 | 2/4 | 6/8 |

PushT 原策略在完整交叉中为 **12/16**。两个构型都能朝两个方向完成任务，仍有四次没有达到规定的终态。每个压缩包均包含 16 个实际回合、源场景、轨迹以及逐步实测记录。[搬运构型汇总](../../artifacts/mtt-bimanual-20261126/carry-crossed-layouts.json) · [搬运完整记录](../../artifacts/mtt-bimanual-20261126/carry-crossed-evidence.tgz) · [PushT 构型汇总](../../artifacts/mtt-bimanual-20261126/push-crossed-layouts.json) · [PushT 完整记录](../../artifacts/mtt-bimanual-20261126/push-crossed-evidence.tgz)

对两个成功搬运回合再问一个静力问题：在实际离台且两腕均有载荷的每个决策姿态，给定腕边与箱侧缘的接触几何、摩擦和原电机限矩，能否平衡 58.86 N 重力，同时保持水平合力与重心力矩为零？两回合分别有 112、113 个合格姿态，双臂在全部 225 个都能静托。在其中 83、81 个姿态，任何一只腕单独都不能静托，两个回合合计 **164/225**。另外 61 个姿态中至少有一侧具备理想静力平衡解。该结果描述已记录的姿态与力矩约束；实际单臂动态尝试另有四次有效对照。[43001 离台姿态](../../artifacts/mtt-bimanual-20261126/carry-capacity-43001.json) · [43002 离台姿态](../../artifacts/mtt-bimanual-20261126/carry-capacity-43002.json)
