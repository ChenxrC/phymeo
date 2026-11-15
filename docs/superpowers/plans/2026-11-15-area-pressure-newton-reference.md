# V03 双软球：Newton 外部静态参考批次

日期：2026-11-15。分支 codex/exp；入口 f2d18485。状态：**预登记，未执行**。
上批：[V00–V02 报告](../../development/area-pressure-static-reference-report-20261115.md)。

## 本批问题

在相同连续压力场下，Newton 1.7 的未缩减接触面是否与独立 NumPy 曲面积分一致？
只执行碰撞/压力接触面前端，不调用 solver.step，不把静态牵引称为动力学输出。
本批不改 Phymeo 产品接触、训练或 TYX；T-smoke 仍是独立后续任务。

Drake 官方安装表不含 Windows；本机有 WSL Ubuntu24.04，但未发现已登记 Drake 环境。本批先复用现有 Newton 环境，不安装新的参考环境、不以工具准备阻塞已有可执行对照。
本地 Newton 1.7.0 / Warp 1.17.0：`.venv-newton-1.7.0`。wheel 无 VCS metadata，绑定本地关键源码 SHA，不声称整个 wheel 与网上 tag 完全同源。
Newton 此 hydro 路由要求双方体 SDF，刚平面 V02 暂不纳入；不能用极大 kh 替代刚平面。

## 固定案例与指标（首次执行前）

- 两球半径均 0.02 m；A 中心 (0,0,0)，B 中心 (0,0,0.0396) m，总重叠 0.0004 m。
- 基准 EA=100000、EB=300000 Pa；交换格只交换两材料模量。
- 各自连续压力 p=E(1-r/R)，Newton kh=E/R=5e6 / 1.5e7 Pa/m。显式 margin=gap=0。
- 请求空间分辨率 0.002 / 0.001 / 0.0005 m；三档基准与三档交换，共六格。另加零重叠、最细档一格，共七格。
- Newton：cuda:0、sdf_texture_format=float32、reduce_contacts=False、pre_prune_contacts=False、output_contact_surface=True、mc_edge_clamp_min=0、rigid_contact_max=16384。其余表示默认值实际保存；记录实际 grid、voxel size、AABB，不将请求步长冒充实际网格间距。
- 返回完整未缩减 face 数据，包括真实面积、压力、原始编码法向/位置/shape-pair/fingerprint、完整世界曲面顶点、原计数与 overflow。有效 raw ID 从1开始；只读版本绑定的内部数据，不修改外部库。
- 原始数据的 local→world 与双方力符号按该版本源码确定，不能凭图像调符号。整面面积与实际承压 fraction 后 dA 分开；不能从渲染三角面积替代真实 dA。

独立 NumPy 参考按 XY 三角积分点逐点二分等压曲面，64次为硬上限；解析坡度给真实曲面 dA 和法向，保存投影面积。二分只在明确浅交轴对齐球域。
独立测试以公共压力参数化：rA=RA(1-p/EA)、rB=RB(1-p/EB)、z=(d²+rA²-rB²)/(2d)、s=rA²-z²；固定64点 Gauss 积分给 Fz 和面积，不共享产品 root 或网格。

原预登记门保持：
1. 最细档 Newton 对独立连续场参考的面积/轴向合力误差≤5%，压力加权中心误差≤0.2 mm；曲面中心不称平面COP。
2. 三档全部报告；最后两档面积/合力变化≤5%。粗档不是最细门。
3. NumPy轴点 z=.0197m、p=1500Pa（交换后z=.0199m）；位置误差≤1e-12m、压力≤1e-7Pa；每点等压残差≤1e-7Pa。
4. 所有结果有限、双方同点反向牵引和共同原点力矩相消；零重叠无正面积/正载荷；任何 overflow 或静默截断拒绝。
5. 交换材料保持轴向力大小与面积，压力面关于中面镜像；最细误差采用同一5%/0.2mm工程门。
6. 每种实现先证明自己声明的表示，再区分连续场离散误差；不同网格一致不是“相同离散输入严格等价”。

## 作者、执行与停止

原 scene agent 只新增 two_sphere_reference.py / test_area_pressure_two_sphere.py。
原 training-review agent 只新增 newton_reference.py / test_area_pressure_newton_export.py。
父负责固定比较runner、解释性viewer、文档与唯一串行执行；不新增agent或分支，原WIP全部保留。
纯作者可并行；项目import/pytest/数值/Warp JIT/静态检查严格串行、每命令fresh guard。

先新HOST → 独立NumPy保存 → Newton CUDA一次完整七格 → 只读保存比较和可视化。
出现失败先保留原输出；最多一次有依据修正，禁止增加第四/第五档或改载荷/模量/阈值追绿。
参考工具/API不兼容则明确阻塞该项，继续独立可完成部分；不将工具导入成功当物理通过。

查看器以“两球侧视 + 界面放大 + 压力热图 + 对照数值”讲清楚案例，固定空间比例、固定压力色标，明确预设静态姿态。纵向放大必须标倍数，压力中心、合力箭头与双方身份要可辨。视频只消费保存数据，不重新求解。

## 一手资料与本地绑定

- [Drake 安装支持](https://drake.mit.edu/installation.html)
- [Drake 压力参数说明](https://drake.mit.edu/doxygen_cxx/group__hydro__params.html)
- [Newton v1.7.0 collision 源文档](https://github.com/newton-physics/newton/blob/v1.7.0/docs/concepts/collisions.rst)
- [Newton hydro 前端](https://github.com/newton-physics/newton/blob/v1.7.0/newton/_src/geometry/sdf_hydroelastic.py)

本地 sdf_hydroelastic.py SHA256：59317898aec932dd5c8ca38f1ad4d82310d39e6a66bacad891ae7c535a7c3661。
实际软件版本、导入路径、设备和其余关键源码哈希在首次执行环境记录中保存。
