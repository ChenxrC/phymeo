# 双 Panda 装配与成功率引导采样

本实验用两台七轴 Panda 操作一块开孔板和一件自由工件，在摩尔线程 MTT S4000 上运行 Phymeo MuJoCo-TYX 动力学。目标是检验毫米级装配和 SGS 训练的数据链路。实验使用两项自建工序，从已装夹或已啮合状态开始。

## 任务与通过条件

| 工序 | 几何和动力学 | 终局测量 |
|---|---|---|
| 插销 | 左臂持 12 mm 厚板，四面实体孔壁围出 10 × 10 mm 的方孔。右臂通过顺应装夹带动半径 4 mm 的销。起点包含孔外、入口前及浅插三种状态。 | 自由销尖进入 8–12 mm，两个横向坐标相对孔中心各在 ±0.9 mm 内，轴线夹角小于等于 2°。 |
| 已啮合旋拧 | 同一自由工件和孔壁，附加单头螺纹的被动导向，导程 1.5 mm/转。角度来自工件相对板的姿态。轴向弹性、阻尼及旋转摩擦的反力同时作用于工件和板。 | 终角落在 1.20 ± 0.13 rad，相对本回合起点实际向前至少 0.07 mm，轴向与实测旋转的螺距误差小于等于 0.15 mm，终端转速小于等于 0.22 rad/s，克服旋转阻力的功大于 0.003 J。横向偏差和倾斜沿用插销门槛。 |

两项工序均要求终端相对线速小于等于 8 mm/s，最后 25 个采样时刻连续满足判据，每步 2 ms。全过程检查已报告的最深孔壁接触和最大装夹位置偏差，分别以 0.7 mm 和 2.5 mm 为上限。完整场景有 14 个主动关节、1 件独立的 0.12 kg 自由刚体和 20 个广义速度坐标。右手驱动座通过六维顺应力传递载荷；可视化使用原始状态重建，运行期间仅写入力和力矩。

本次使用已固定开度的手指，左手板夹具刚接，碰撞范围是装配件与孔壁。自由件稳定夹持、接触起牙、错牙、真实牙面和最终扭矩锁紧各需额外模型与案例。真实操作工位的机械臂互撞和桌面防碰也需要另行检验。

## SGS 与对照

AlphaNIST 的复位采样公式给成功率约为 50% 的状态更高的采样权重。本例对 12 个预先生成的训练初态分别保存最近 100 次二值结果，按 `0.05 + sqrt(p*(1-p))` 归一化后抽样。每代先抽 3 个初态，让同一组 6 个候选在这 3 个初态上比较，然后用全部候选的终局成败更新状态统计。每项任务训练 4 代，共 72 个候选回合。各训练条件使用相同的起始权重和候选随机种子。固定均匀采样对照只更换复位分布。

两项工序分别训练一份小策略。插销使用横向对中与轴向进给两个增益，旋拧额外使用转角增益；两份检查点都采用三个参数的格式。控制器读取仿真中的精确位姿，用双臂逆运动学生成关节目标，以关节位置和速度反馈输出限矩电机控制。SGS 决定从哪里开始练习，交叉熵方法（CEM）用终局二值结果更新增益。原 AlphaNIST 在 20 项任务上训练共享策略，任务规模和策略表达能力均与这个试验不同。[AlphaNIST 方法](https://arxiv.org/html/2610.13354v1#S4.SS2) · [SGS 作者说明](https://sgs-rl.github.io/)

## 2026-11-26 S4000 实测

| 测试集 | 保持初始姿势 | 训练前 | SGS 训练后 | 相同预算的均匀采样训练后 |
|---|---:|---:|---:|---:|
| 插销，6 个固定留出初态 | 0/6 | 4/6 | 4/6 | 4/6 |
| 已啮合旋拧，6 个固定留出初态 | 0/6 | 2/6 | 4/6 | 4/6 |
| 已啮合旋拧，12 个新增初态 | 未测 | 4/12 | 8/12 | 8/12 |

SGS 的初始采样概率是每个状态 1/12。第一代结束后，两项任务的最高采样概率均约为 0.487，说明后续计算确实按实测成败重新分配。在这一次优化器种子下，均匀采样与 SGS 的测试通过数相同。两份 SGS 检查点另由独立进程重新载入，在种子 290317 的六个初态上各通过 4/6；每项评估记录了 S4000 的 5,250 次完成的内核调用。

首版旋拧在 2 ms 时间步下显式施加平滑动摩擦，终端角速度出现约 1.53 rad/s 的振荡，留出组 0/6。修正使用完整广义质量矩阵计算相对转动的等效逆惯量，在预测末速上隐式求同一个阻力矩，以实测相邻转角累计阻力做功。原导程、材料参数及成功阈值相同。36 项几何、坐标导数、优化器和螺纹力学测试通过。

剩余失败集中在较难起点。两条未通过的插销轨迹终深为 6.89 和 7.39 mm，横向偏差约 1 mm。两条未通过的旋拧轨迹终角均低于下限 1.07 rad，过程中孔壁穿入分别达到 1.07 与 0.91 mm，超过原定上限。下一阶段应让策略根据运行中的偏差与载荷修正路径，再引入初次啮合与扭矩锁紧测试。

[中文实测回放](../../artifacts/s4000-panda-assembly-sgs-20261126/dual-panda-assembly-zh.mp4) · [两套场景与冻结策略](../../artifacts/s4000-panda-assembly-sgs-20261126/panda-assembly-checkpoints.zip) · [插销结果](../../artifacts/s4000-panda-assembly-sgs-20261126/pin-result.json) · [旋拧结果](../../artifacts/s4000-panda-assembly-sgs-20261126/screw-result.json)

## 重现

使用已配置 MuJoCo、NumPy、Phymeo 和 TYX MUSA 运行时的工作树。从仓库根目录执行下列命令，每次传入新的输出目录。每项训练都会输出 XML、冻结策略、独立测试的实测轨迹及结果。CPU 可通过显式的 `--mode cpu` 运行。

```bash
python examples/panda_assembly_sgs_train.py --task pin --mode musa --out runs/pin
python examples/panda_assembly_sgs_train.py --task screw_proxy --mode musa --audit-seed 241173 --out runs/screw
python examples/panda_assembly_sgs_train.py --task pin --mode musa --sampler uniform --out runs/pin-uniform
python examples/panda_assembly_sgs_train.py --task screw_proxy --mode musa --sampler uniform --audit-seed 241173 --out runs/screw-uniform
python examples/panda_assembly_sgs_evaluate.py --policy runs/screw/policy.json --mode musa --seed 290317 --count 6 --out runs/frozen-screw
python -m pytest -q tests/test_panda_assembly_sgs_scene.py tests/test_panda_thread_guide.py
```

回放使用 S4000 的 Vulkan 渲染库。每章左侧是两臂全景，右侧是同一个时刻的装配局部。近景报出的进给和角度来自实测自由件与板的相对状态。

```bash
python examples/panda_assembly_replay.py --backend vulkan --library build/floor/libphymeo_floor.so \
  --episode runs/pin/scene.xml runs/pin/holdout-1.npz \
  --episode runs/screw/scene.xml runs/screw/holdout-1.npz \
  --playback-rate 0.15 --out runs/movie
```
