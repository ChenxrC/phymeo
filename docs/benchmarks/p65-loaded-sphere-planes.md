# 有限刚体球与固定平面的持载和反弹

`ph.SpherePlaneWorld` 可在已有 `TyxWorldBatch` 中登记一个有限质量、有限半径的球和本世界的固定水平面。世界坐标向上为 +Z，无摩擦，接触法向穿过质心，切向速度不变。原自由球与滑轨继续使用各自的原运动律。下例的第二行检查到静止球已把本步重力传给固定面。

```python
import phymeo as ph

s = ph.SpherePlaneWorld(mass=2, radius=0.125, position=(0, 0, 0.125),
                        gravity=(0, 0, -8), plane_height=0, restitution=0.5)
with ph.TyxWorldBatch([(7, s)], "cuda") as batch:
    assert batch.step(1 / 128)[7]
    assert batch.snapshot()[7].normal_impulse_n_s == 0.125
```

未提交的自由预测抵达或越过中心支撑高度时，设备将球心恢复至可表示的支撑侧。若法向自由速度向内，恢复系数作用于入射速率大于两倍本步外部法向速度变化的接触；慢速接触停止。设备保存本步向上的冲量 `m(v⁺−v_free)`，与外加力分列。出射中的初始穿入只修正位置。设备以同一事务发布位置、速度、步号和真实冲量，不能表达的局部数值让该世界保留上次检查点。固定面吸收等反冲量。这里发布的是离散末步响应。原始平面若落在 FP32 网格之间，球心支撑阈值向外取整。[注册与检查点](../../src/phymeo/backends/_world_step_layout.py) · [真实 TYX 法向响应](../../src/phymeo/backends/_tyx_sphere_plane_kernel.py) · [受载、开离与回滚试片](../../tests/scale/test_resident_sphere_plane.py)

## 先验资格

固定 m=2 kg、r=1/8 m、h=1/128 s；有重力的世界为 g_z=−8 m/s²。四个混合世界覆盖从接触开始的平稳支撑、从接触以 −1 m/s 入射的半恢复反弹、从间隙 3/128 m 和 v_z=−1/2 m/s 开始的落地，以及整窗分离的上抛。另有无重力单次反弹和 32 颗有不同切向速度的持续受载球。每个世界具有自己的静止平面和非连续世界键。Newton 1.7.0 用真实形状、SAP 宽相与窄相、4 次 XPBD 迭代、刚体接触松弛 1、无接触计数加权、零摩擦与每形状 1/64 m 的候选间距。混合/单次弹性案例开启原生恢复求解。资格过程另请求了 `CONTACT_F`。[冻结输入和门槛](../../tests/scale/sphere_plane_cases.py) · [两端原生装配](../../tests/scale/loaded_sphere_engines.py)

每步按相同世界键排序，读取 XYZ 和质心速度，计算几何间隙以及从实际出入速度扣除重力得到的净法向冲量。预设的支撑标准为间隙误差 ≤20 μm、速度 ≤0.2 mm/s、末 8 步平均载荷误差 ≤1%；首次进入真实力层允许与离散解析预测相差一步，反弹系数误差 ≤0.03。严格同输出另逐比特比较全部 8 列、每步、每世界的实际轨迹。Newton 原生力和 Phymeo 设备冲量单独保留。

| 资格案例 | Phymeo interpret / Newton CPU 和 Phymeo CUDA / Newton CUDA 的结论 |
| --- | --- |
| 持续支撑 W32 | 两端每步每球输出相同。首步冲量为 0.125 N·s，末八步平均支撑为 16 N；资格还分别读取了设备冲量和 Newton 原生接触力。 |
| 无重力单次反弹 | 八列逐比特相同，实际恢复系数 0.5。 |
| 有重力 W4 混合 | 四世界功能门均通过。下落球第 5 步受力，支撑、落地和净空三世界全 32 步逐比特相同。反弹球有 14 个出步的完整观测不同。 |

有重力反弹球首步的两端净冲量均为 3.1875 N·s，Phymeo 的设备冲量除以步长为 408 N。Newton `CONTACT_F` 的 Z 分量为 272 N，乘以步长为 2.125 N·s；这两个量在该次含弹性恢复的出步中不同。第 17 步发生第二次接近，两个球心恰在 z=r 处。Phymeo 的闭端点规则给 v_z=+0.234375 m/s、J=1.40625 N·s；Newton 此时仍为 −0.46875 m/s、J=0。Newton 的 `mark_restitution_contacts` 以严格负分离标记恢复接触。混合窗最大位置、速度和冲量差分别为 2.31934 mm、0.703125 m/s、1.59375 N·s；该窗只作功能比较。本机核源码为 `newton/_src/solvers/xpbd/restitution_kernels.py`。[Newton 官方 XPBD 参数说明](https://newton-physics.github.io/newton/latest/api/_generated/newton.solvers.SolverXPBD.html) · [逐世界原始分歧和资格审计](../../scripts/check_loaded_sphere_planes.py)

## W32 持续受载的同输出 CUDA 计时

在同一 RTX 4060 Laptop 上，三个独立进程配对均通过逐比特门。每世界有一对已登记的有限刚体球和固定平面，每一步均真实传递法向载荷；32 球合计支撑 **512 N**。Phymeo 的专用受限求解直接执行已绑定平面对的自由预测及法向投影。Newton 的原生求解调用 SAP 碰撞管线产生约束，再执行通用 XPBD。这个计时覆盖该已知配对、始终支撑、无摩擦、有不同切向线速度的任务。

| 配对，先运行端 | Phymeo 16 步 p50 / p95 ms | Newton 16 步 p50 / p95 ms | Phymeo / Newton 核验世界步每秒 | Phymeo / Newton 吞吐 |
| --- | ---: | ---: | ---: | ---: |
| 1，Phymeo | 3.876 / 3.942 | 26.820 / 31.102 | 132541 / 18524 | 7.155 |
| 2，Newton | 3.851 / 4.250 | 27.300 / 32.351 | 131910 / 18409 | 7.165 |
| 3，Phymeo | 3.727 / 3.931 | 24.926 / 30.096 | 136330 / 19876 | 6.859 |

每进程首次一步后连续做 3 个 16 步预热窗，再测 10 个 16 步窗。每条路径每轮在正式窗口内核验 5,120 个世界步。数字包含求解、显式同步、实际状态下载、共同应用的八列输出与逐步功能门；原生额外载荷通道的读回在独立资格窗中完成。三个新进程的首次物理步分别为 Phymeo 28.31 / 37.11 / 26.53 ms、Newton 98.23 / 74.64 / 82.20 ms，各自复用资格阶段已填充的磁盘编译缓存。p95 取十个有序窗的最大值。

每个正式 16 步窗中，Phymeo 实际执行 32 次 CUDA 物理核、16 次显式工作区同步，状态与步号下载 26,624 B、接受标记 2,048 B，稳态上传为零。Newton 调用 16 次碰撞管线和 16 次物理求解，显式同步 16 次，读取 32 个姿态/速度数组共 26,624 B。每轮另在窗外跟踪一个实际 Newton 支撑步，Warp 记录 26 次核活动，包含真实刚体接触投影。六份完整 NPY 均为 210 × 32 × 8 个 FP32，并与各自资格窗的前 33 帧逐比特相同，后验还核查了三个配对的起跑顺序和守卫边界。每个世界每个出步的实际间隙为 0，净冲量为 0.125 N·s。[测量驱动](../../scripts/benchmark_loaded_sphere_planes.py) · [资格与全程核验](../../scripts/check_loaded_sphere_planes.py)

父树 `artifacts/p65-loaded-sphere-planes-20261105-01` 保存各阶段源码摘要、守卫、完整状态和实际退出码。`n3-qualify-{phymeo,newton}-{cpu,cuda}-1` 和 `n3-four-lane-audit/compare.json` 包含原生力、世界/形状归属和逐世界第一处不符；`n3-measure-{phymeo,newton}-cuda-{1,2,3}` 与 `n3-formal-audit/comparison.json` 是同输出支撑窗口。冻结旧版本 `b8cba9b` 的公开入口缺失首红保存在 `n3-public-ball-red`。最终由专用设备模块执行后，四项实际 CUDA 和四项 interpret 均通过，含单独自由球仍只启动原核的控制。

复现先分别在 Phymeo 与 Newton 环境运行 `python scripts/benchmark_loaded_sphere_planes.py --engine <phymeo 或 newton> --device <cpu 或 cuda:0> --phase qualify --cache-root <缓存目录> --out <新目录>/report.json`。将四份带守卫的报告传给 `scripts/check_loaded_sphere_planes.py --reports ... --out <资格.json>`。取得严格 W32 资格后，新进程使用 `--phase measure --device cuda:0 --eligibility <资格.json>`，并为每轮提供新输出目录。
