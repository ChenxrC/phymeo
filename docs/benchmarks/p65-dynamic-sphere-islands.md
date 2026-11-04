# 同一世界内的多动态球接触与承重

`TyxSphereIslandBatch` 支持同一世界内 1–8 颗有限质量、有限半径的球相互碰撞，并可登记一个固定水平面。物体 ID 允许在其他世界重用。实测中，1 kg 和 3 kg 球斜向正碰时交换了 **2.25 N·s** 冲量；1、2、1 kg 的三球承重链在地面、中层、顶层分别传递 **32、24、8 N**。Phymeo interpret、真实 TYX CUDA、Newton XPBD CPU 和 CUDA 都通过了这两个试片和独立空程控制的预设功能门。[公共入口与验收](../../src/phymeo/backends/tyx_sphere_island.py) · [两引擎实际状态和原生接触读数](../../scripts/observe_sphere_islands.py)

```python
from phymeo import FreeSphere
from phymeo.backends.tyx_sphere_island import SphereIslandWorld, TyxSphereIslandBatch

world = SphereIslandWorld((
    (4, FreeSphere(1, 0.3125, (0, 0, 0), (1.2, 1.6, 0), (0, 0, 0))),
    (81, FreeSphere(3, 0.3125, (0.375, 0.5, 0), gravity=(0, 0, 0))),
), restitution=0.5)
with TyxSphereIslandBatch([(17, world)], "cuda") as batch:
    first = batch.step(1 / 128)[17]
    assert first.accepted
    contact = first.contacts[0]
    print(contact.world, contact.first, contact.second, contact.normal_impulse_n_s)
```

## 几何、求解与发布

每批最多 128 个世界。每个世界按局部 ID 建立全部无向球对，以及每个球与所选地面的槽位。球数为 B 时，球对槽数为 B(B−1)/2，有地面另加 B。每次调用重新计算活跃接触。HOST 装入完整现态和可选的单步外力；设备预测各球运动，处理每个本地对的几何，按逆质量校正位置，并对真实校正后法向执行非负的耦合冲量迭代。不同世界可并行，每个世界内部按固定顺序处理共享物体的行。几何、位置、冲量和速度均在所选 TYX 后端求出。[设备约束核](../../src/phymeo/backends/_tyx_sphere_island_kernels.py) · [工作区与同步](../../src/phymeo/backends/_sphere_island_device.py)

位置校正针对自由预测姿态的局部支撑面。其余物体的位移可使另一行进入接触。HOST 在最终候选坐标上独立扫描所有真实表面，检查球面间隙、单位法向、互补残差、逐球动量以及接触对动能的影响。位置与速度容差分别为 10 μm 与 20 μm/s。已接受行分别给出初始间隙、自由预测间隙、实际末态间隙、A→B 法向及两侧材料表面点。对动态球 A/B，法向冲量分别为 −Jn 和 +Jn。固定面为第一侧，ID 为 `None`。默认 96 次扫描，可选 1–128 次。所有接触无摩擦，法向经过球心；只有超过本步两倍外部法向相对速度变化的迎面速度才触发指定的恢复系数。

容量不足会返回真实所需行数并保留那个世界的旧状态；同心无唯一法向，或相对自由线段提示中途穿越而末态接触不能覆盖过程时，返回具体局部对、最近线段分数和间隙，接触数标为未知。迭代未达容差、位移后变成分离的接触图、超出被动能量预算的多体恢复解也会明确拒绝。一例三球以 (2, 0, 0.5) m/s 并列运动，两个局部恢复目标会令系统额外获得 0.25 J，原状态因此保留。该步为有限离散求解，未给出连续首触时间。设备晚读回故障使全批保持上次发布，下次调用重新建工作区。局部 `reset([world_id])` 只恢复选定世界。非零间隙与正冲量分别保留其数值，`contacts` 中几何闭合行可有零载荷。[容量、潜在穿越、材料、能量与恢复试片](../../tests/scale/test_sphere_island.py)

## 三世界实际观测

每个引擎分别推进 16 步，步长 1/128 s。撞击沿 (3,4,0)/5，两球初始接触，轻球以 2 m/s 入射，e=0.5、g=0。另一世界的球半径都是 0.25 m，质量从下到上为 1、2、1 kg，g_z=−8 m/s²。第三个世界使用重复局部 ID，两颗相隔 7 m 的球有不同切向速度并在重力下自由飞行。Newton 1.7.0 使用同样的有限半径、质量和实心球惯量、独立世界及静面，SAP 与真正的 XPBD 刚体接触，64 次迭代、0.8 刚体松弛、关闭接触计数加权，另请求 `CONTACT_F`。全部引擎在每步后读回球心和质心速度。[四端功能后验](../../scripts/check_sphere_island_observations.py)

| 从真实出入速度重建，CUDA | Phymeo | Newton 1.7.0 |
| --- | ---: | ---: |
| 轻球传出的法向冲量 N·s | 2.250000072 | 2.250000089 |
| 重球收到的法向冲量 N·s | 2.250000089 | 2.250000143 |
| 实际恢复系数 | 0.500000027 | 0.500000045 |
| 两球动能损失 J | 1.125000022 | 1.124999978 |
| 初始到任一观察帧的最大线动量差 N·s | 2.98×10⁻⁸ | 8.94×10⁻⁸ |
| 三球堆叠最大穿入 m | 5.96×10⁻⁸ | 5.22×10⁻⁷ |
| 堆叠末八步最大竖直速度绝对值 m/s | 7.45×10⁻⁹ | 0 |

碰撞后轻球和重球的法向速度分别为 −0.25 和 +0.75 m/s。在 Phymeo 的接受姿态中两球实际表面间隙为 5.77 nm；现有公共 `ph.detect` 对相同末态独立复核了材料点和法向。堆叠按实际动量变化扣除重力，再从顶部向下累加，各层末八步读数均为 **8、24、32 N**。Phymeo 的原生约束冲量给出同样三层载荷；Newton 的原生接触力读到 **8.00156、24.00221、32.00195 N**。首个弹性碰撞中，Newton 原生接触力乘步长在重球方向上为 **1.499991 N·s**，与重球的净速度变化不同。两类观测在收据中独立保存。

两个 CUDA 轨迹在堆叠构型的最大分量差为 0.954 μm 和 0.1465 mm/s；碰撞构型为 0.0596 μm 和 2.98×10⁻⁸ m/s，自由世界逐点相等。本批没有性能排序。实际 TYX CUDA 三世界共 7 颗球、10 个同世界可能对，16 步记录 48 次核提交和 16 次显式同步。固定拓扑上传 320 B，逐步现态/外力/容量共上传 4,224 B，下载候选状态、行及审核数据 9,856 B。

父树 `artifacts/p65-dynamic-sphere-islands-20261105-01` 保存全部原始物理状态、接触力、退出码和文件绑定。`g4-old-api-red` 为固定旧源码缺少公共入口的原失败，`g4-sphere-interpret-first` 保留 TYX 前端最初拒绝增强赋值的记录。最终七项专项在 interpret / 真实 CUDA 均通过。四端观测位于 `g4-observe-{phymeo,newton}-{cpu,cuda}/report.json`；独立逐世界比较为 `g4-native-audit/comparison.json`。复现使用各自解释器运行 `python scripts/observe_sphere_islands.py --engine <phymeo 或 newton> --device <cpu 或 cuda:0> --cache-root <缓存目录> --out <新路径>`。
