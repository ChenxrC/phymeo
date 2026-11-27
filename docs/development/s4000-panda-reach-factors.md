# 双 Panda 接近时的首次碰撞与交叉测试

本批沿用 [上轮发布的状态策略](s4000-panda-guided-reach.md)，考察右指何时首次碰到自由工件，以及偏航和跨轨偏置分别如何影响接近。所有策略仍通过原电机驱动两臂与双指。每次运行 128 次决策、1,024 个物理子步，共 2.048 秒。

**交付结果**　S4000 上的第 127 号候选在五组共 60 个评估起点全部完成驻留，59 个保持到终点，全部轨迹的销漂移均在 10 mm 内。唯一的严格数值审计超差是一项板位移特征。候选单独发布，默认仍为第 126 号策略。[候选权重](../../examples/checkpoints/panda-reach-factors-s4000/candidate.pt) · [配对结果和误差明细](../../examples/checkpoints/panda-reach-factors-s4000/results.json)

## 实验设计

| 用途 | 偏航，绕世界 Z | 跨轨偏置，叉架 X | 沿轨偏置，叉架 Y | 初态 |
| --- | --- | --- | --- | --- |
| 诊断上轮冻结策略 | −26°、+26° | −1、+1 mm | −9 mm | 四格共享相同三组关节扰动，种子 271130 |
| 追加示范训练 | −30°、+30° | −1.2、+1.2 mm | −7 mm | 每格三个独立关节扰动，种子 271132；另保留原 12 个训练世界 |
| 独立验证 | −22°、+22° | −0.8、+0.8 mm | +7 mm | 四格共享另外三组关节扰动，种子 271131 |

每个评估库内部可以在固定偏置和相同手臂起点下翻转偏航，也可以在固定偏航和相同手臂起点下翻转跨轨偏置。两库之间有多项变化，不能从库间差值估计沿轨偏置的单独效应。生成器从保存的 FP32 关节状态重新算出工件相对叉架的位姿，核验四格定义。[初态生成器](../../examples/panda_reach_crossed_resets.py)

先从上轮三个留出库保存的子步位置重建首次指–件几何接触。旧轨迹每 2 ms 保存位置、每 16 ms 保存一次速度。接触若在速度样本之间，就分别记录相邻样本的实测速度投影和由位置差分得到的估计。新轨迹每 2 ms 同时保留位置与速度。速度符号取工件相对手指在指向工件的接触法线上的投影，负值表示靠近。这项几何复核不提供求解器接触力或冲量。[事件诊断程序](../../examples/panda_reach_contact_events.py)

仅在 24 个训练起点上选择模仿学员。依次比较全程销漂移超过 10 mm 的世界数、五项连续达标 0.1 秒的世界数、终点同时达标数和平均终点距离；完全相同则保留上一版权重。学员胜出才进行一次真实 PPO 更新，并在诊断库和独立验证库与同一父策略逐世界配对。两库都要求销漂移超限次数不增加、驻留与终点达标数不下降，并至少有一项改善，才进入旧留出库回归；通过回归前保留原发布权重。即使学员未入选，也独立测量父策略在第二个库上的表现。

## 首次接触

在上轮新增留出集的两个大漂移世界，事件重建结果如下。开度为每根手指相对闭合位置的位移，35 mm 为该实验的全张开目标。首次接触都是右侧指垫与销柄。

| 世界 | 首次接触 | 手偏离销轴 | 手在销中心上方 | 两指开度 | 估计法向靠近速度 | 销首次漂移超过 10 mm |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| 9 | 0.516 s | 38.4 mm | 22.4 mm | 34.93 / 34.91 mm | 0.139 m/s | 1.244 s |
| 10 | 0.610 s | 33.0 mm | 22.4 mm | 34.92 / 34.92 mm | 0.133 m/s | 1.098 s |

两次接触时的姿态误差都不到 1°，销离重置位置不足 0.001 mm。这能定位手指在销的侧上方提前接近；从第一次几何接触到销明显偏移之间还有多步运动，无法把最终位移全部归到第一次碰触。第三个 +26° 失败起点没有重建出指–件接触，手最近距离为 74.4 mm。

## 同一手臂起点下的诊断

| 偏航 | 跨轨 | 连续达标 | 终点达标 | 有指–件接触 | 销漂移超过 10 mm |
| --- | ---: | ---: | ---: | ---: | ---: |
| −26° | −1 mm | 3/3 | 2/3 | 0/3 | 0/3 |
| −26° | +1 mm | 3/3 | 2/3 | 0/3 | 0/3 |
| +26° | −1 mm | 1/3 | 0/3 | 2/3 | 1/3 |
| +26° | +1 mm | 1/3 | 0/3 | 0/3 | 0/3 |

在这三个配对的关节起点上，翻转偏航重复降低了驻留表现。跨轨偏置对本组驻留计数没有影响，但接触次数有变化。每格只有三次，结论限于这些起点。

## 扩展示范后的模型选择

三行都是同一批 24 个训练起点的实际电机回放。控制权完全交给当前网络，教师仅在采样阶段提供用于之后拟合的标签。

| 网络 | 连续达标 | 终点达标 | 销曾漂移 >10 mm | 平均终点距离 |
| --- | ---: | ---: | ---: | ---: |
| 上轮已发布的网络 | 19/24 | 13/24 | 2/24 | 80.6 mm |
| 前两轮示范拟合后的学员 | 24/24 | 24/24 | 0/24 | 10.7 mm |
| 再拟合一轮后的学员 | 8/24 | 0/24 | 7/24 | 186.5 mm |

按预设的物理指标，第二行进入 PPO 更新。S4000 阶段日志从 12:54:29 到 13:06:44，示范、学员拟合和训练集回放共用 12 分 15 秒。随后第 127 号 PPO 更新阶段用时 3 分 13 秒，计入数据收集和物理探针。该阶段沿用父检查点的 12 个训练世界、奖励和电机模型。独立评估加载更新后的权重，直接从 98 维状态输出 16 路电机动作。

## 两组交叉初态的配对结果

教师退出后，每个起点分别运行第 126 号父策略和第 127 号候选。驻留要求右手距销中心 ≤20 mm、相对朝向误差 ≤15°、板位移 ≤10 mm、板倾角 ≤5°、销漂移 ≤10 mm，五项连续达标 0.1 秒。终点另看 2.048 秒时这五项是否仍同时达标。漂移统计检查全部物理子步。

| 初态，每组 12 个 | 驻留，父 → 候选 | 终点，父 → 候选 | 销曾漂移 >10 mm，父 → 候选 | 平均终点距离，父 → 候选 |
| --- | ---: | ---: | ---: | ---: |
| ±26°，诊断库 | 8 → 12 | 4 → 11 | 1 → 0 | 108.7 → 11.8 mm |
| ±22°，独立验证库 | 12 → 12 | 2 → 12 | 0 → 0 | 40.5 → 8.9 mm |

两组关电机的对照都为 0/12 驻留、0/12 终点达标。候选在这 24 个起点上均完成驻留，其中 23 个保持到终点。唯一未保持到终点的是 +26°、跨轨 −1 mm 格中的一个起点。按每 2 ms 保存的几何状态复核，24 条候选轨迹都没有检出指–件接触，销的最大漂移小于 0.065 mm。本阶段在抓取前结束。

## 原 36 个起点的回归

两组交叉评估通过预定条件后，冻结权重，继续在原有三组逐起点配对测试。每组单独要求驻留和终点通过数均不下降，漂移超限数不增加。

| 各组 12 个起点 | 驻留，父 → 候选 | 终点，父 → 候选 | 销曾漂移 >10 mm，父 → 候选 |
| --- | ---: | ---: | ---: |
| 上轮新增留出集（fresh） | 9 → 12 | 8 → 12 | 2 → 0 |
| 首次留出集（blind） | 11 → 12 | 7 → 12 | 1 → 0 |
| 原始回归集（regression） | 7 → 12 | 5 → 12 | 1 → 0 |

三组均通过行为回归。合计五组，父策略为 47/60 驻留、26/60 终点达标、5/60 销漂移超限；候选为 60/60、59/60、0/60，全程最大销漂移为 0.0643 mm。关电机的对照为 0/60 驻留。60 个起点中有 36 个是复用的历史留出起点，另有 12 个用于交叉诊断、12 个用于独立验证；这些数量描述本次实测覆盖。诊断脚本的球面法线方向、速度采样间隔和四格初态配对另有 4 项测试通过。

下一批可从同一策略的实际驻留末态继续，先验证闭指与抬离叉托。保存两侧指垫分别接触销柄的时刻、法向载荷、销与手的相对滑动，以及托架反力何时卸到零；双指稳定承载后再判定抓取成功。关电机、只碰到一侧指垫、销仍由托架支撑分别作为对照。这样能直接回答画面里“看似没夹住，载荷从哪来”。

## 检查点的独立数值复核

S4000 完成了一次 PPO 更新。用同一份观测和探索噪声，更新前后的实际电机动作最大差为 0.00755。策略、价值网络、轨迹、优势估计及 Adam 的一、二阶矩均在 MUSA 设备，Adam 步数计数在 CPU。随后用同一批保存的关节位置，经原生 MuJoCo 的双精度正运动学独立复算策略所见的相对位姿。

32,256 个被核对的位姿特征值里，有一项超出原门槛。世界 4、从零编号的决策 70，板在其初始坐标系中的 X 位移相差 **0.700 µm**，该值的容差是 **0.603 µm**。在同一点用 CPU Torch 单精度计算，数值介于设备记录和原生双精度参考之间。这个比较能定位误差发生在哪个量，尚不足以归因到某个乘法或某个驱动。

独立复核中，12,288 个物理子步的五项达标判定全部一致。1,536 个决策的距离、姿态、板位移、板倾角、销漂移和奖励均在原容差内。记录的 FP32 回报可精确复算，优势和更新前后动作也通过各自的原检验。原检查点综合审计仍以非零状态退出。[误差定位程序](../../examples/panda_reach_feature_error.py)

## 直接使用候选权重

在已配置 Phymeo、MuJoCo-TYX 和 MUSA PyTorch 的 S4000 环境中，从仓库根目录运行下列命令即可比较两版权重，无需重新训练。三条运动轨迹分别使用父策略、候选策略和关闭电机的候选策略，起点完全相同。

```bash
export PYTHONPATH="$PWD/src:$PWD/examples${PYTHONPATH:+:$PYTHONPATH}"
candidate_dir=examples/checkpoints/panda-reach-factors-s4000
guided_dir=examples/checkpoints/panda-reach-guided-s4000
out=artifacts/replay-panda-factors
python examples/panda_state_ppo.py --stage evaluate --task reach --worlds 12 --steps 128 \
  --action-repeat 8 --epochs 4 --minibatch 192 --seed 91027 --learning-rate .0003 --bias lagged \
  --checkpoint "$candidate_dir/candidate.pt" --comparison-checkpoint "$guided_dir/policy.pt" \
  --reset-npz "$candidate_dir/validation-12.npz" --eval-seed 99029 --eval-steps 128 --out "$out"
python examples/panda_reach_trace_diagnosis.py --trace "$out/heldout-paired-physics.npz" \
  --scene "$out/scene.xml" --out "$out/metrics.json"
```

## 复现训练

沿用上轮 S4000 环境，从仓库根目录运行。生成器要求输出目录尚未存在。

```bash
export PYTHONPATH="$PWD/src:$PWD/examples${PYTHONPATH:+:$PYTHONPATH}"
work=artifacts/reproduce-panda-factors
base=examples/checkpoints/panda-reach-s4000
guided=examples/checkpoints/panda-reach-guided-s4000
for split in crossed validation; do
  python examples/panda_reach_crossed_resets.py --split "$split" --out "$work/bank-$split"
done
common=(--task reach --worlds 12 --steps 128 --action-repeat 8 --epochs 4
        --minibatch 192 --seed 91027 --learning-rate .0003 --bias lagged)
python examples/panda_state_ppo.py --stage evaluate "${common[@]}" --checkpoint "$guided/policy.pt" \
  --comparison-checkpoint "$base/policy.pt" --reset-npz "$work/bank-crossed/initial-states.npz" \
  --eval-seed 99029 --eval-steps 128 --out "$work/factor"
python examples/panda_reach_contact_events.py --trace "$work/factor/heldout-paired-physics.npz" \
  --scene "$work/factor/scene.xml" --out "$work/events-factor.json"
python examples/panda_reach_distill.py --checkpoint "$guided/policy.pt" --reset-npz "$base/train-12.npz" \
  --teacher-path axial-recovery --minimum-teacher-worlds 12 --include-crossed-training \
  --select-physical-seed --out "$work/distill"
python examples/panda_state_ppo.py --stage train "${common[@]}" --checkpoint "$guided/policy.pt" \
  --actor-warm-start "$work/distill/actor-seed.pt" --warm-actor-lr-factor .01 \
  --reset-npz "$base/train-12.npz" --updates 1 --eval-steps 16 --out "$work/ppo"
python examples/panda_state_ppo.py --stage evaluate "${common[@]}" --checkpoint "$work/ppo/update-00127.pt" \
  --comparison-checkpoint "$guided/policy.pt" --reset-npz "$work/bank-validation/initial-states.npz" \
  --eval-seed 99029 --eval-steps 128 --out "$work/validation"
```

数值复核可直接使用随仓库发布的候选和同目录训练轨迹。第一条输出逐项误差，第二条在该检查点的板位移特征处返回非零，保留原接受门槛。

```bash
candidate=examples/checkpoints/panda-reach-factors-s4000/candidate.pt
python examples/panda_reach_feature_error.py --checkpoint "$candidate" --out "$work/feature-error.json"
python examples/panda_state_ppo.py --stage audit --checkpoint "$candidate" --out "$work/audit.json"
```
