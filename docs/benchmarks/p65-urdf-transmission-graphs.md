# 多行 URDF 机械联动

普通机械联动装载器 `load_mimic_urdf` 可以读取多条标量约束。一个驱动可带多个从动轴；从动轴也可继续驱动其他轴。每条原始约束都是同种关节类型、有限非零倍率的 `q_f = a q_d + b`。每个从动轴恰有一个直接驱动。没有上游的轴依照物理树顺序构成独立命令向量。缺失关节、多个上游、环、不匹配的关节类型及空的可动区间均具名拒绝。

装载时按机械图而非 XML 声明次序传播每条物理轴相对于源驱动的倍率和位移。位置限和速度限从所有后代逆映射到相应源轴后取交集。动力学沿用完整原树的每个连杆、质量、惯量与受力，独立质量为 `Pᵀ M P`。原始约束矩阵 `asset.constraint_rows` 与原关节列对应，`asset.matrix` 表示完整关节到独立命令之间的速度映射。

```python
from phymeo.builders.mimic_urdf import load_mimic_urdf

gearing = load_mimic_urdf(path)
step = gearing.step(independent_state, effort=motor_efforts, external=full_body_loads, dt_s=h)
for row in step.loads.row_forces:
    print(row.driver, row.follower, row.driver_effort, row.follower_effort, row.constraint_power_W)
```

每条约束的乘子是其对从动轴的作用力；同一条约束对驱动轴的作用力为 `−a λ`。一个既是从动又是下游驱动的中间轴会收到多行的贡献，因此它的总反力通常不同于其上游一行的乘子。求解先由原树的实际加速度与载荷得到全部物理轴的平衡余量，再从机械图叶片反推每个原始乘子；以 `Cᵀ λ` 重建并校验整树广义反力。单条关系仍可使用原有的 `multiplier_force` 标量；多条时该字段为 `None`，可按 `row_forces` 中的驱动和从动名称读取。

分叉试片是三个同向滑轨材料件和一个独立升降台。丝杠带一支反向夹指（−1，+4 mm）和一支正向量规（+0.5，+2 mm），量规行程令丝杠间隔缩为 −4 至 +16 mm，速度上限 80 mm/s。串联试片的三个物理铰轴不平行。末轴从中继轴取 −0.5 倍并加 0.2 rad，中继从根轴取 +1.5 倍并减 0.125 rad；最终轴对根的系数为 −0.75、偏置 +0.2625 rad。末轴和中继轴的限位共同约束根轴。[分叉量规资产](../../tests/reference_newton/robot/mimic_fan.urdf) · [三级齿轮资产](../../tests/reference_newton/robot/mimic_chain.urdf) · [全坐标解析](../../tests/reference_newton/robot/mimic_graph_oracle.py)


## 真实受载和原生边界

三级齿轮的两行原始乘子在根轴 0.18 rad 时为 (+0.0721424, +0.0122597) N·m，根轴 −0.16 rad 时为 (−0.0612143, +0.0198636) N·m。第一姿态的中间轴还收到下游驱动端的 +0.0061299 N·m，总反力与第一行乘子可清楚区分。两姿态的总动能分别为 0.0511326 J、0.0209185 J。每个物理刚体的平动与转动能量、每个原关节的共轭动量能量项 `v_i (M v)_i / 2` 和每条原约束的零功分别对照了五维全坐标鞍点解及独立空间运动学。后一项能量分摊包含耦合惯性，不代表把相连的轴分成独立刚体。

分叉量规的有限探球对一个 125 g 自由球首步施加 2 N，随后 24 个 0.5 ms 物理步均检出实际接触。自由球沿 X 移动 0.915225 mm。每步从原材料对的实际力独立组装四个物理轴的广义载荷，两个原始乘子与六维鞍点解的最大差为 2.23×10⁻¹⁶ N。物体冲量、全轴加速度和两条反力都受到逐步断言。[实际自由球与逐行反力试片](../../tests/test_mimic_graph.py)

Newton 1.7.0 Featherstone 在 CPU 和真实 CUDA 上均原生登记了以 `screw` 为驱动的两条关系。冻结外载下，全坐标加速度为 (0.3421021, −3.6666679, +3.6666679, −1.8333340) m/s²。两条反力从实际状态增量和原生质量矩阵重建，分别为 +4.25×10⁻⁷ N 和 +0.8666664 N，独立解是约零与 +0.8666667 N。该原生输入同时检验了应为空载的一行与承载的一行。原生链式参考在两个进程内均由解析器拒绝，原错误为 `Reference joint 2 is already a mimic joint`；没有原生三级链的进步数值。Phy 的三级链以全坐标解为参照。[Newton 分叉步骤及链式实际导入](../../tests/reference_newton/robot/mimic_graph_native.py)

新增 HOST 9 项，单行兼容与非法输入 12 项（旧的“所有链均非法”示例换为真实双轴环），两个原生设备作业和最终 Ruff 检查均通过。原有限单行程序对第一份多行资产的真实拒绝保存在证据目录，红测使用的原始资产仅 robot 名称标签与最终版本不同，原始字节另附。[原始收据与两端数值](../../artifacts/p65-night-mimic-graphs-20261105-01/)
