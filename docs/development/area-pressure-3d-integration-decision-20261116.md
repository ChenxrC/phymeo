# 面积压力三维接口与求解接线裁定

2026-11-16，codex/exp。对应[冻结批次合同](../superpowers/plans/2026-11-16-area-pressure-3d-port.md)。本文件先记录设计，实际结果在末节追加；本段不是运行通过声明。

## 采用范围

采用刚性三角表面 B 与连续 PL 体压力 A 作为首个三维前端候选；当前 I0 接收已裁好的三角压力面，并给出真正的面积积分及双侧广义力端口。产品级三维裁剪、体网格有效性、跨单元连续性和边界去重属于下一批 I1。

V05/V06 两种独立运动参考已经通过其固定门，允许推进接口。V03 Newton 双软球表示的面积/收敛失败原样保留；这不能作为新三维表示已经通过的证据。FFT 作为小变形半空间标定参考保留；Fourier 几何暂不移植。

首域要求 g=grad(pA)·n>0，其中 n 指向 A、相对分离速度为 n·(vA-vB)。g<=0 的整面/整包明确拒绝，不跳过。这个条件不天然覆盖所有 PL 压力场，例如最近边距离压力的某些边缘区域；后续前端须暴露不支持域。法向冻结近似也不声称捕捉所有切向压力输运。

## 本次新增数据如何连接

已裁三角压力面 → 三点二次精确求积 → 真 dA、world point、各面 normal、pressure、g → full shared-DOF Jrow → 等效 k/phi 与双侧 force pullback。

三点使线性压力的合力和一阶力矩都保留；单质心虽然可积分合力，通常不能积分力矩。曲面不同三角可有不同法向；净力为零而力偶非零的情况也保留。所有结果保存 body/material、source cell/rigid face、world/step/time/configuration 与 sample 顺序。

source 标签是调用者提供的绑定元数据，不是碰撞证书或真实状态哈希。当前函数不发布状态，不输出“求解成功”或“accepted step”。旧 ContactPatchSample 权重仍和为1，旧 patch 刚度保持 N/m 语义；它们没有被冒充为面积。

## 已核验的 MuJoCo-TYX 真实接缝

| 接缝 | 当前语义 | 对后续影响 |
| --- | --- | --- |
| [_constraint_rows.py](../../src/phymeo/backends/mujoco_tyx/_constraint_rows.py) | 共同接触点 frame·(JB−JA)，原 solref/solimp 给 aref、R、D=1/R | 新 A/B/normal 必须显式转换；dA*g 不能直接写进原 solref/solimp |
| [_constraint_solver.py](../../src/phymeo/backends/mujoco_tyx/_constraint_solver.py) | M a=tau_smooth+J^T f，残差 J a+R f−aref | 解未知 f 是 N，不是冲量；现有 solver 不收 h |
| [_constraint_device.py](../../src/phymeo/backends/mujoco_tyx/_constraint_device.py) | 先约束求解，再复用 M+h*Cjoint 积分 | physical_acc 与 integration_acc 可能不同；不能只验报告加速度 |
| [_mujoco_tyx_runtime.py](../../src/phymeo/backends/_mujoco_tyx_runtime.py) | 入步 q/v 几何，候选 advance/check，最后交换 physics/entry bank | 压力、力、状态必须沿同一事务发布 |
| [_model.py](../../src/phymeo/backends/mujoco_tyx/_model.py) | legacy / normal / T4 多路选择 | pair 模型归属须在 route selection 前确定，不能只在 T4 局部去重 |

首期 I0 映射 Jrow=n^T(JA−JB)，QA=JA^T(fn)，QB=JB^T(−fn)，使用完整共同自由度顺序。共有自由度上的总广义力未必为零；共同 world 原点的双方 wrench 相反。完整 coupling 来自共同质量算子，I0 未计算它。

## I2 可检验的初始代数，不冒充现有运行时等价

对冻结几何、零耗散/零摩擦、J 固定且 a 就是实际用于速度更新的加速度：

vn_next=vn_entry+h*J*a
f=max(0, -k*(phi0+h*vn_next))。

因此单边约束可写为：

r=J*a+R*f-aref >= 0，f>=0，f*r=0，
R=1/(h*h*k)，aref=-phi0/(h*h)-vn_entry/h。

这是推导的一个具体接线候选，说明为什么 h 必须进入新的行装配。R 单位是逆质量，f 单位 N；k=dA*g，phi0=-p/g。它不是把原 MuJoCo stiffness 参数替换成 k。

该推导目前有清楚的条件：旧后端在存在关节阻尼时使用 M+h*Cjoint 更新速度，而约束解使用 physical M。若直接用 physical_acc 代入上式，所得 f 不一定满足实际更新后的压力律。因此 I2 必须：
1. 先以零关节阻尼、双自由体/单关节固定模型验证候选方程及独立标量解。
2. 明确实际速度响应算子，处理含关节阻尼的共同约束，独立比较实际 v_next 与求解残差；不能删阻尼或修改已有 native 语义作为永久方案。
3. 再推导耗散与摩擦，保留法向/切向耦合和完整共同解。不得把旧 V05 的 Hunt–Crossley 力律直接标成现有 native 行等价。
4. 同 pair 只选 native 或 area-pressure 一路，原 collision filters 与其他 pair 保持；容量不足明确失败。
5. 只有实际 joint/free-body 运动、整步力/矩/能量及事务恢复通过后才接受 I2。

论文中的离散模型提供局部物理依据，以上与当前后端的对应及限制是本项目源码分析和代数推导。[速度级压力场论文](https://arxiv.org/html/2110.04157v2)

## 下一批具体入口

I1 从“一枚有正体积的四面体内仿射压力场，与有限刚性三角相交”开始，输出本端口，不先上 GPU。解析横截面、斜截面、擦边空集、跨相邻 cell 的去重/连续压力、多个非共面刚面，以及 g<=0 明确拒绝组成固定矩阵。先解决真实三维表面生成，再用同一接口做两体/关节共同解，最后设备化；不再重做旧 V00–V06 图像追绿。

若测试暴露数据模型遗漏，只修具体接口问题；若 g 准入不足以覆盖目标，在 I1 报告并选择更合适的压力律/离散模型，不无限添加特判。
