# 折面接触站的自动选择、释放与重接

一个 1 kg 自由顶点从层外靠近带折角的有限布片。在固定外载下连续推进 96 个 `1/32 s` 物理步，顶点在 **0.063445 s** 入层，**0.490942 s** 离层，**2.158458 s** 用新的剪切历史重接，**2.587913 s** 再次离层。68 个完整步没有接触，节点接触冲量逐项为零。每次事件都用实际初末节点坐标插值，并从三个当时的支撑节点独立重算法向间隙，等于层宽 0.0625 m。[逐步位置、速度、冲量和事件](../../artifacts/p65-continuous-folded-sites-20261105-01/pytest-program-host/test_fixed_external_load_disco0/continuous-folded-sites.json)

## 上层如何调用

`FoldedContactProgram(model, geometry, bindings, *, layer, static_friction, tangent_stiffness_n_per_m, material=None, layer_operator=None)` 接受布程序的 `ParticleModel`、已登记的 `CollisionModel`、`ParticleAttachment` 行绑定和可选的 `ClothMaterial`。用 `initial_state(particles)` 建立含节点状态和站点历史的检查点，随后反复调用 `step(state, inputs, dt)`。距离或弯曲弹性与所有站点的反力在同一组节点上相加，产生一次全体粒子速度/位移更新。只需几何时可调用 `FoldedSiteSelector.select(start, end)`。[布程序](../../src/phymeo/solvers/folded_contact_program.py) · [几何选择](../../src/phymeo/geometry/folded_contact_sites.py)

候选来自同一材料体两张真实三角形共有的边。显式登记粒子时，以登记的粒子为探针。纯三角形注册允许所有材料顶点充当探针。注册所属世界、双向掩码、排除对及同体一环邻居决定哪些配对可用。对扫过的有限三角形先用包围盒和相对运动上界排除，再在当前折面上求材料法向间隙达到层宽的时刻。每段端点分别返回沿材料投影的面键、顶点和权重、法向间隙，以及对两张有限三角形各自求得的欧氏最近特征、点、权重和距离。同一组注册与材料标识保持不动，只把两探针的初态平移 4 m，两个接触便从材料体 10 切到材料体 11。[候选与容量测试](../../tests/collision/test_folded_site_selection.py) · [物理和拓扑留出](../../tests/collision/test_folded_contact_program.py)

## 实际接触与结算

单探针从法向间隙 84 mm 处以 −0.32 m/s 靠近。取左翼单位法向 `n=(-1/16,0,1)/|n|`、横向单位切向 `t=(1,0,1/16)/|t|`，顶点所受恒力为 `−0.4n+0.02t N`。每片布的共享边端点各 96 kg、两翼外节点各 64 kg，全部自由，重力为零。法向刚度 64 N/m，切向刚度 16 N/m，静、动摩擦系数为 0.5、0.25。完整受载步使用入步法向压力。只受载一部分的步在该区间中点取一次压力，只按该区间的路径积分冲量。这个近似具有步长依赖性。[输入和独立法向间隙断言](../../tests/collision/test_folded_contact_program.py)

第一次进入时左翼节点 31、17、7 的材料权重为 0.388081、0.388081、0.223838。此时两张有限三角形到探针的欧氏距离分别为 229.061 和 62.500 mm。本步压力 0.336755 N 对应受载区间中点。第一次释放时记录剩余弹性能量 22.54 µJ，第二次释放为 131.87 µJ。两次重建均从零弹性变形开始。逐步检验每一行实际动量变化等于外力冲量加上接触反力冲量，并核对每段切向功、弹簧储能、塑性耗散和隐式离散耗散。[96 步原始记录](../../artifacts/p65-continuous-folded-sites-20261105-01/pytest-program-host/test_fixed_external_load_disco0/continuous-folded-sites.json)

在双探针留出中，四个合格候选里有两个同时压在同一片布的两翼。共享边上还有一根真实拉伸弹簧，向两端分别施加 +1 和 −1 N 的 y 向力。HOST 法向力为 **2.000000、1.312000 N**。用独立三维有限差分求每个材料地址的节点虚功导数，重建两侧节点冲量，再与全部粒子的实际动量变化比较。六次联合迭代后的最大力残差为 1.37×10⁻¹² N 以下。容量不足或粒子求解拒步时可以从原检查点重试并重放相同轨迹。[HOST 双站和结构力原始记录](../../artifacts/p65-continuous-folded-sites-20261105-01/pytest-program-host/test_two_probes_share_live_mat0/coupled-fold-sites.json)

真实 CUDA 用一个核计算两个法向材料行，用一个核推进全部十个自由粒子。法向力为 2.0、1.312000036 N，最终节点力残差 4.78×10⁻⁷ N 以下。法向层工作区实际上传 6,656 字节、下载 2,560 字节。候选搜索、阈值根、几何、切向返回映射、弹性和非线性迭代在 HOST。96 步轨迹在 HOST 完成，设备资格覆盖上述双站物理步。[实际 CUDA](../../artifacts/p65-continuous-folded-sites-20261105-01/pytest-program-cuda/test_two_normal_rows_and_one_g0/joint-fold-device.json) · [TYX 解释执行](../../artifacts/p65-continuous-folded-sites-20261105-01/pytest-program-interpret/test_two_normal_rows_and_one_g0/joint-fold-device.json)

## 步长与角动量边界

对相同前 0.7 s 外载窗口，保留节点初态、质量和材料。y 方向是共享边轴向，末列采用速度先行更新所用的入步力臂。两档的节点内力矩沿实际漂移积分均小于 1.2×10⁻¹⁷ N·m·s。[同一初态与外载的细分记录](../../artifacts/p65-continuous-folded-sites-20261105-01/pytest-program-host/test_same_externally_driven_wi0/fold-sites-time-resolution.json)

| 子步 | 入层 / 离层 (s) | 探针接触冲量 z (N·s) | 实际 ΔLᵧ (N·m·s) | 入步外载角冲量 y (N·m·s) | 入步内部角冲量 y (N·m·s) |
|---|---:|---:|---:|---:|---:|
| 14 × 0.05 s | 0.063048 / 0.491565 | 0.834822 | −0.06057860 | −0.06068551 | +0.00010691 |
| 28 × 0.025 s | 0.063628 / 0.490707 | 0.858767 | −0.06060040 | −0.06065708 | +0.00005668 |

原有轻质量快速越脊样例的单步离散内部角冲量为 **+0.01564638 N·m·s**，沿 +y。新增候选选择继续使用速度先行的节点更新，其实际角动量仍与沿漂移积分的力矩有差异。这里的两档时间步只能说明这组输入的敏感性。[原始折脊及子步对照](p65-folded-material-seam.md)

支持域是有限、同向绕序的双三角小折角、穿过共享边中点的横截面和一维横向剪切。每步允许边平移、翼法向随自由节点变化，边轴方向须稳定。靠近无配对的三角形、单探针对重叠折面有同时归属、无法区分的掠碰或迭代不收敛都会显式拒绝。这个入口可由现有布上层直接调用，任意折叠网格的多面接触和二维面内摩擦仍待扩展。[有限域检查](../../src/phymeo/geometry/folded_material_seam.py)

## 新入口的正、负折角留出

在同一次注册里放入两张完整折面和一个探针，使两个候选中仅一处受载。相同 2 N 压力下，折脊在归一化步时 0.5047292464 换面，左右材料地址在节点 17、31 各占一半，材料法向间隙 24.345785 mm。两张三角形的欧氏最近特征都为共边，距离 24.418651 mm。节点实际内部角冲量沿 +y 为 0.0156463844 N·m·s。反向折谷经空间轴循环后在 0.5049941135 换面，材料法向间隙 39.897246 mm，欧氏最近点分别在两张不同面内部，实际内部角冲量沿 +z 为 0.0152280663 N·m·s。两种符号各只建立一个受载历史，未受载的另一整片逐节点为零接触冲量。[折脊实测](../../artifacts/p65-continuous-folded-sites-20261105-01/pytest-cross-host/test_geometry_selected_site_tr0/selected-fold-crossing.json) · [折谷实测](../../artifacts/p65-continuous-folded-sites-20261105-01/pytest-cross-host/test_geometry_selected_site_tr1/selected-fold-crossing.json)
