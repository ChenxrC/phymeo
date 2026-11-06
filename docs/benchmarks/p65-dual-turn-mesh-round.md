# 闭网格与圆体双方独立转轴：冻结测试合同

初始测试交付状态：**TEST_AUTHORING_ONLY / UNEXECUTED**，提交 `2c596cc` 只含三份新测试/fixture及本文；当时未创建产品模块。父从 clean `b63bb895796899355d769911e06486d4ccb90ce6` 计划基线串行运行。收到下节有效 RED 后，作者按 **IMPLEMENTATION_AUTHORING_ONLY** 编写新增模块；作者仍未执行 import、collection、数值 probe、pytest、lint/format 或设备程序，未修改父共享分派。

公开有效 RED 节点：

```text
tests/collision/test_dual_turn_mesh_round.py::test_b01_public_two_axes_after_independent_static_overlap
```

它先独立确认两端净空和真实轴向变化，调用现有静态 `ph.detect` 在 t=3/8 得到 overlap，核全双侧材料，再调用原双方转动路径并要求 hit。预期旧实现在 candidate=True/status=unsupported 处失败；新模块缺失或环境失败不算有效 RED。

## 八个执行前固定输入

全部长度 SI 米、总角弧度，除特别写出外两枢轴均0、r=1/8。盒用既有 test-only `box` 的8个字典序顶点和12个朝外三角；U复用已固定七格外边界构造，独立有向边配对和正体积检查。本片不以该构造名称分派产品。

| 输入 | 固定几何、运动及独立目标 |
|---|---|
| B01 | 盒 `[-2,2]²×[7/8,9/8]`；有限轴0→Y。mesh绕X转−2A，round绕Z转A，A=实际存储的 `math.pi/2`。两轴垂直不交换，轴方向实际改变。mesh-local tip=`(-sin a,cos(2a)cos a,sin(2a)cos a)`，a=At；首次高度3/4等价 `2s(1-s²)=3/4` 的第一根s=1/2，所以首触为π_real/(6A)，不是精确1/3。 |
| B02 | B01的圆体改为中心Y的偏心球，其中心真实绕Z移动。首根相同；球材料只有ID41，不伪造胶囊轴地址。 |
| B03 | B01的mesh枢轴终点为 `(1/8,-1/16,3/32)`，round终点为 `(1/8,-1/16,7/64)`。相对多出z方向1/64。首根解 `sin(2At)cos(At)+(t/64)cos(2At)=3/4`，预定区间[5/16,1/3]，在此前段导数有独立严格正下界。 |
| B04 | 盒x±1/2、y±2、z[1,5/4]绕Z转1rad；轴 `(-3,1,0)→(3,1,0)` 绕X转3rad。杆身首触asin(7/8)/3，两个端球整段净空≥√10−√93/4−1/8>1/2。首次轴上每一点有相同高度，投影在有限旋转矩形内的完整区间由独立线性裁剪给出，不能只验证一个幸运点。此例胶囊轴仍沿X，真实朝向变化由B01/B03负责。 |
| B05 | B01盒底升为1、顶为5/4。全程皮肤净空≥7/8−4/(3√3)>0，来自独立三次函数最大值，不用离散采样排除。 |
| B06 | U实体为 `{|x|∈[1,2],y∈[-2,2]}∪{x∈[-1,1],y∈[-2,-1]}`，z±1/4；轴±(1/4,0,0)，mesh绕X0.9、round绕Z1.3。U排除单位球，完整轴在半径1/4球内；全程皮肤净空≥5/8。凸包替代将产生假接触。 |
| B07 | 闭盒±2，轴±(1/2,0,0)，mesh绕X0.9、round绕Z1.3。整段实体包含，首触0；边界皮肤净距≥11/8，不能从表面净空误判实体分离。 |
| B08 | B01盒z改[5/8,7/8]，mesh角−A、round角+A。高度为sin(a)cos(a)，最大恰1/2；孤立相切时刻π_real/(4A)。只接受有证据hit或unresolved，不接受clear。 |

## 独立真值与验收

`dual_turn_mesh_round_cases.py` 不导入 producer 或生产旋转/距离内核。π使用独立50位十进制有理区间；B03、B04分别在预定区间以有理 Taylor 展开和导数余项包围作二分，参考区间宽≤2^-42。B03另验证在[0,1/3]高度严格递增，B04利用sin在[3/4,3/2]严格递增。最早时刻不是从 producer 输出选择的。

实际采样有限距离复用旧 test-only 枚举器：两端点–三角面/边、有限轴–三条有限边、有限轴–平面穿越，实际浮点样本坐标转有理数。球用重复端点处理。真实姿态由普通Rodrigues独立求值；该采样检查仅复查实际见证，不能作为全路径净空证明。

HOST测试要求首触括区覆盖独立参考、宽≤2^-27，witness_time在括区内且不早于独立第一根；对应实际采样有占用，双侧所有面ID、重心权重、球/胶囊offset、有限轴端参数都能在同一时刻重建。材料误差门2e-9m，独立采样距离门2e-12m，不放宽产品原1e-6m世界/时间精度预算。另有反序、顶点存储重排、镜像、整体刚体变换、移动枢轴协变、重复局部ID及跨world；两侧角容量、样本容量、非法闭体、候选容量、事务中止与重试均显式。

TYX专项固定四个共址world：B01/B02/B03/B06，复用局部body/shape编号。interpret与真实CUDA必须逐字段匹配HOST完整pairs/bounds，保实际launch、CUDA读回/同步、容量拒绝后恢复及设备错误关闭工作区且旧发布不变。四个同world结果应为hit/hit/hit/separated，24条跨world过滤行完整保留。

计划产品接口仍为 `DualTurnMeshRoundQuery(prepared_mesh,round_shape,mesh_path,round_path,mesh_first)`，继承原 `MeshRoundQuery` 五字段合同。父仅改旧 mesh–round 路由；原单侧路径继续复用。新窄相必须保完整相对路径速度/两次旋转与存储误差包围，未知/容量不足不得伪装分离。测试按现有32rad、512样本与4096时间单元上限冻结，不新增无限线/凸包近似。

实现后的全部正确性/设备结果当前待父实际执行，未记录通过数。范围是**HOST保守连续窄相＋TYX真实宽相**，不宣称全GPU CCD、接触响应或摩擦流形。Newton当前姿态的原生离散/speculative接触不作为首触真值；本批不安排重复Newton采样。

## 有效 RED 与实现候选

父已保存 `artifacts/p65-top-three-batch6-20261106-01/01-b-public-red`：source `2c596cc` 的 before/after 全部相同，独立首根、两端净空、实际轴向改变、旧静态 overlap 和双侧材料重建先通过，只有新双方转动公共路径返回 unsupported 而期望 hit。**1 failed / 1.45 s / RAW_EXIT 1**，不是模块缺失。原始[stdout](../../artifacts/p65-top-three-batch6-20261106-01/01-b-public-red/stdout.log)及[退出码](../../artifacts/p65-top-three-batch6-20261106-01/01-b-public-red/RAW_EXIT.txt)保留。

新增 `geometry/_dual_turn_mesh_round.py` 的候选 API：

```text
DualTurnMeshRoundQuery(prepared_mesh, round_shape, mesh_path, round_path, mesh_first)
.result(moving) -> (status, reason, endpoint_witness, sweep, DualTurnMeshRoundTrace)
```

它继承已验 `RotatingMeshRoundQuery`（也是 `MeshRoundQuery`），直接复用其有限点/段到三角特征 `_minimum`、距离调用和完整 `result` 搜索/首态实体/占用样本门。没有复制宽相、碰撞响应或修改原单侧转动实现。新增的 `_sample_axis` 和 `_pose` 只负责完整双转弧、保守误差、实体关系的可判定性和材料映射。近轴穿越只复用不依赖共轴的严格跨面符号判据，不继承旧的轴向守恒或共面特例。

设网格和圆体的初始枢轴为 p_m、p_r，平移为 v_m、v_r；d=p_r−p_m，v=v_r−v_m，a_i 为圆体材料端点相对 p_r 的向量。在固定网格材料系中：

`x_i(t)=p_m+R_m(t)^T [d+t*v+R_r(t)*a_i]`。

两次 `rotate_enclosures` 分别返回有理坐标提案及欧氏误差。真实外旋转是等距变换，因此相对轴点误差取 **inner_error+outer_error**，而不是漏掉内层误差。随后转换成静态材料求解所需 FP64 时另累计实际存储差；`best` 也按存储点重算。该误差用于半径阈值、轴穿面/绕数、圆体包住网格等判定，不确定则 unresolved。

相对完整路径速度上界固定为：

`S=|v| + |theta_r|*max_i|a_i_perp_r| + |theta_m|*[max(|d|,|d+v|)+max_i|a_i|]`。

所有长度用精确有理数及向上平方根界；有限轴上任意参数点是两端凸组合。S的单位为米/归一化时间，它服务完整间隙排除和首触相对空间括区，不是含公共平移的绝对世界点速度。两个原始角各保持32rad上限；512时间轴样本和原4096时间单元、2^-27首触宽度及1e-6m原预算不改。未知不作为净空，孤立擦触不要求用放宽半径取得hit。

世界材料映射在同一 witness_time 做两条独立重建：网格原面权重经网格正向旋转/枢轴映射；球心或有限胶囊两端经圆体自己的正向转弧映射。圆体的世界 radial offset 与法向用网格正向旋转搬运，再核对两条重建的一致性、原始权重舍入、旋转列误差和实际浮点表示。完整轴参数区间的两端及其对应面权重同样重建。并列集合来自有误差包围的存储相对姿态，不冒称理想超越姿态上的精确全并列或一般MTD。

新 trace 保留 start/end/first_contact，追加 `relative_samples`、`relative_speed_upper_m`、`observed_relative_error_upper_m`、`observed_world_error_upper_m`；公开上界全部向上舍入。最后一项记录各已发布世界材料点/重建检查的最大误差界，首触见证属于实际占用样本，不冒称直接在数学首根求得。

父共享 hook 约定：未倒圆的 mesh 与 sphere/capsule 双方均为非零 `RigidTurn` 时，code0/5继续 `prepared_mesh` 的真实闭体合法性检查，仅code0构造本query；沿用原 `_pair` 的 `isinstance(MeshRoundQuery)` 五字段发布，过滤/容量/设备失败事务不变。原单侧、零角/平移路由保持旧实现。父另行迁移已存在的双转动历史 unsupported 断言，原输入不变，作者未修改旧测试。

当前候选 **UNEXECUTED**，三份冻结测试、八输入、oracle和全部容差未修改。父建议先完整 `tests/collision/test_dual_turn_mesh_round.py`，然后 `test_tyx_dual_turn_mesh_round.py` 的interpret与真实CUDA，再按实际受影响范围验旧路由；失败原始输出保留，不以改fixture追绿。

## 父串行最终验收与源码外 wheel

本节记录最终状态；前文的 TEST_AUTHORING_ONLY / UNEXECUTED 保留为 tests-first 和作者候选交付时的历史。作者没有运行项目命令。父在同一批次的唯一串行执行队列完成下列验收；原八输入、独立 oracle、容差和资源上限均未为取得通过而修改。

收据根为 `artifacts/p65-top-three-batch6-20261106-01/`。`01-b-public-red` 的原始失败继续保留：静态正控、独立首根与材料校验先成立，旧双方转动公共路径才在 unsupported != hit 处失败（1 failed / 1.45 s / RAW_EXIT 1）。没有以模块缺失代替该有效 RED。

| 父实际作业 | 结果与限定 |
| --- | --- |
| [04-b-host-first](../../artifacts/p65-top-three-batch6-20261106-01/04-b-host-first/stdout.log) | 首次完整 HOST：18 passed / 18.17 s / RAW_EXIT 0。 |
| 05-b-ruff-first、06-b-format-first | 首次静态检查分别为 HOST test 的 I001 导入顺序、四个 NEW Python 文件需格式化；原失败收据保留。 |
| 07-b-import-fix、08-b-format | 父仅修导入别名顺序和格式，没有几何、输入或判据修复。 |
| 09-b-ruff-final、10-b-format-final、[11-b-style-ast](../../artifacts/p65-top-three-batch6-20261106-01/11-b-style-ast/stdout.log) | 最终 Ruff/format 均 RAW_EXIT 0；六个受核文件中五个原始 AST 相同，HOST test 仅 import alias order 改变。复用 04 的物理验收，没有为样式重跑 HOST。 |
| [12-b-interpret](../../artifacts/p65-top-three-batch6-20261106-01/12-b-interpret/stdout.log) | 实际 interpret 专项：1 passed / 10.52 s / RAW_EXIT 0。 |
| [13-b-cuda](../../artifacts/p65-top-three-batch6-20261106-01/13-b-cuda/stdout.log) | 实际 CUDA 专项：1 passed / 11.14 s / RAW_EXIT 0，检查真实宽相提交、读回/同步及完整 HOST 结果对照。 |
| [14-b-dispatch-compat](../../artifacts/p65-top-three-batch6-20261106-01/14-b-dispatch-compat/stdout.log) | 四项受影响旧路由兼容：4 passed / 2.79 s / RAW_EXIT 0。旧 two_turns 输入保留，按 t=1/2 的独立贯穿判据迁移能力预期；不是删除历史控制。 |

A 原作者另对 B 实现作独立文本交叉复核，未报告本片范围内的阻断；该复核不计为新的数值验证。

父集成产品快照为 `3e6dff494d898ad19a3a3695d1de0c368b511805`。[47-integrated-wheel-build](../../artifacts/p65-top-three-batch6-20261106-01/47-integrated-wheel-build/stdout.log) 实际构建成功，RAW_EXIT 0。wheel 为 `phymeo-0.1.0a0-py3-none-any.whl`，大小 **1,115,791 B**，父保存的 SHA-256 为：

```text
5cd405e970bd1ef5709769d7b439f143613be69c23c1b9cc58c87c11b775aa09
```

[48-outside-wheel-integration](../../artifacts/p65-top-three-batch6-20261106-01/48-outside-wheel-integration/stdout.log) 在源码外安装目录运行，RAW_EXIT 0；所有已加载 Phymeo 模块都位于 wheel 安装目录。B 检查使用 B01 两个不交换旋转轴，核最早首触括区和同一采样时刻的双侧材料重建；同次集成还完成 A 的原登记反序 40 步实际直接接触例程。这里是打包后的有限集成留出，不声称从 wheel 重跑八输入矩阵或全部 CUDA 窄相。

最终交付范围保持为**两个独立 RigidTurn 的闭网格–球/有限胶囊 HOST 保守连续窄相，以及真实 TYX 宽相**。完整路径速度与旋转/存储误差界、首态实体关系、实际占用采样、材料寻址和 unknown/容量失败语义均保留；孤立擦触仍可明确 unresolved。没有新增接触响应、摩擦流形或设备窄相，也不把 Newton 的离散/speculative 接触称为连续首触真值。本次最终报告只追加验收证据，不改变已测产品与测试源码。
