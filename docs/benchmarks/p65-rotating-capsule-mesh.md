# 胶囊自身转弧与有限闭网格

初始交付时，本批 B 首片的源码和独立用例已编写，作者 **UNEXECUTED**。作者没有运行 import、pytest、数值 probe、静态检查或设备程序；全部验证由父唯一串行槽执行。本文不预先宣称测试通过或性能收益。

## 范围与公共接缝

新增 `geometry/_rotating_capsule_mesh.py`，入口为 `RotatingCapsuleMeshQuery(prepared_mesh, capsule, mesh_path, capsule_path, mesh_first).result(moving)`。它继承原 `MeshRoundQuery`，返回原五字段 `(status, reason, endpoint_witness, sweep, mesh_round_trace)`。父在普通 `ph.detect` 的 mesh–round 分支接入：只有胶囊自身非零单轴转动、网格无非零转动、网格无倒圆才使用新查询。静态/平移旧路径、双方转动的显式能力边界、过滤与容量规则保留；code 0/5 仍验证真实闭网格，只有 code 0 执行窄相。

范围包含任意已验证的单一闭合三角网格、有限胶囊轴段、线性移动枢轴、独立网格平移。胶囊端帽与整个杆身均参与距离，不使用无限直线或闭网格凸包。`RigidTurn.angle` 按实际存储的有符号弧度使用，不将浮点 π 替换成理想半圈。

查询返回首个真实实体接触的有界时间区间和区间内的实际占用样本；初始包含、重叠或相切均先在 t=0 分类。`RotatingCapsuleMeshTrace` 保留 start/end/first_contact，另存 axis_samples、axis_speed_upper_m、observed_axis_error_upper_m。速度单位为米/归一化时间；误差单位为米，公开上界向上取整。首触 witness 的所有面 ID、重心、有限轴参数、球皮 offset 均绑定同一次 witness_time。gap 是该采样材料特征的有向量；边界净距和实体包含分别记录，不宣称一般 MTD、接触力、冲量响应或稳定摩擦流形。

## 全路径界与有限资源

在网格初始材料坐标系中，两端轨迹为

`p_i(t) = pivot_start + t*(pivot_travel - mesh_travel) + R(axis, angle*t)*(p_i(0)-pivot_start)`。

已有有理 Rodrigues 包围给出端点的距离误差；有限轴段上的任意点是两端的凸组合。全段速度上界取相对枢轴平移速度，加 `abs(angle)` 乘两端最大的垂轴臂长。因此有限轴段到真实三角边界的距离具有同一 Lipschitz 上界。复用 `first_bounded_contact` 对每个时间单元给出分离下界，正例须有占用样本。拒绝不确定单元，不能用有限采样推全段净空。

原闭体只由工作区验证一次。每次轴距离按真实面 AABB 下界排序，复用有限段–三角精确特征枚举；所有并列最近面及各自材料参数由原 `MeshRoundQuery` 生成。这里的并列集合是有误差包围的存储姿态上的精确最近集合，不保证理想超越姿态上所有数值不可分辨的并列关系。

本片限 `abs(angle) <= 32` rad、最多 512 个不同时间轴样本，沿用时间搜索 4096 单元上限、首触括区宽度至多 `2^-27`。首触时间括区乘速度及世界/材料重建误差要求不超过 `1e-6 m`；原静态材料/世界分辨率检查仍在。近轴穿越若不能由严格跨面符号证明、半径阈值落入旋转误差带、孤立擦触、容量耗尽等明确返回 unresolved。特别是共面或近棱轴穿越不自动判可解。双方独立转轴仍 unsupported。上述有限边界不影响已能证明的分离和占用路径。

TYX interpret/CUDA 负责原真实 swept broad phase；本片有限特征、实体判定和根隔离均为 HOST，不称 GPU 窄相。原不同 world、原候选上限、工作区事务与设备错误传播均由公共管线处理。

## 先冻结的独立输入

`tests/collision/rotating_capsule_mesh_cases.py` 不导入新 producer 或旧最近特征函数。它独立枚举有限段端点–面/边、轴段–三条有限边和轴段–面穿越，静态距离使用实际浮点坐标的精确有理数；独立 Rodrigues 使用普通数学函数。单独测试闭边界有向边配对、正体积和有限线段的解析控制。

| 类别 | 执行前固定输入与独立目标 |
|---|---|
| B01 杆身首触 | 盒 `[1,2]×[-1/8,1/8]×[-1/4,1/4]`，轴 `(0,3/2,-1)→(0,3/2,1)`，r=1/8，枢轴0、+z、角−π；`t=acos(1/6)/abs(stored_angle)`，x=√35/4，杆身全参数区间[3/8,5/8]。两端球整段至少5/8m净空。t=15/32 先用旧静态公共入口确认真实相交。此例轴仍沿z，不冒作朝向变化。 |
| B02 径向净空 | 同盒，轴轨道半径9/4；全程皮肤净距 `(17−√257)/8>0`，由盒最远径向角点和圆周径差独立证明。 |
| B03 真凹槽 | 七个格单元合并边界的闭U，z±1/4，实体最小xy半径1，轴绕半径1/4全圈、r1/8，皮肤净空≥5/8。不能用外凸包替代。 |
| B04 真姿态变化 | 轴0→(1,0,0)，r1/8，绕z转3rad；有限壁盒 `[-2,2]×[1,5/4]×[-1/4,1/4]`，端帽首触 asin(7/8)/3。另有3/5–4/5斜轴、独立枢轴漂移及共同平移，首根来自预定[1/4,1/2]内的独立有限特征二分，宽≤2^-40。数值参考与产品括区比较单列2e-12归一化时间误差，不放宽产品括区。 |
| B05 初态与失败 | 严格包含、正好相切后开离/闭入、孤立角点擦触；角容量、采样容量、开网格及双方转轴分别分类。擦触只能给可证命中或未决。 |
| B06 留出与恢复 | 有限短轴位于z=[5/8,7/8]使无限直线成为假阳性；反序、顶点存储重排、镜像/刚体变换；相同局部ID跨world、容量不足、事务中止重试、设备失败旧发布保留。 |

HOST 专项核首触、实际采样占用、起末实体关系与所有双侧材料。TYX 专项固定三world：B01、B02、斜轴B04；逐字段比 HOST 的完整 pairs/bounds，要求真实 completed launches、CUDA 读回/同步，禁静默 HOST 宽相。

## 验证台账与下一入口

父已保存 `03-b-capsule-mesh-red`：B01 先通过旧静态 boundary_overlap 与材料重建，再在完整转弧得到 candidate=True/status=unsupported；**1 failed，1.12 s，原 exit 1**。这是有效旧入口缺口，不是缺模块导入失败。测试先于新模块冻结：B01 `f49ba46`，B02–06 与设备组 `03c7f93`。

初始交付时，其余测试/设备结果均 **UNEXECUTED**；后续父实际结果见下一节。父采用新共享 hook 和作者 checkpoint，验证入口为：

```text
python -m pytest tests/collision/test_rotating_capsule_mesh.py -q -x -p no:cacheprovider --basetemp <fresh-absolute>
python -m pytest tests/collision/test_tyx_rotating_capsule_mesh.py -q -x -p no:cacheprovider --basetemp <fresh-absolute> <既有interpret/CUDA选择>
```

不得从作者的文本手推提前写通过数。若出现失败，保留首次输出并从真实几何/路径原因诊断，不改冻结输入追绿。Newton 本批不新增执行；其原生 mesh–convex 离散姿态与 speculative contact 不是连续首触 oracle。

## 父串行验证补记（2026-11-06）

父已完成本片的实际公共入口验收。以下结果来自父唯一串行槽，作者仅只读收据并更新本文，没有新增项目执行、物理重跑或改动四个 Python 文件。收据的 canonical 根为父集成树 `artifacts/p65-top-three-batch5-20261106-01/`；各目录保有 `start.json`、源快照、`stdout.log`、`stderr.log`、`RAW_EXIT.txt` 和 `final.json`。

| 阶段 | 父实际结果 | 原始收据 |
|---|---|---|
| 首次有效 RED | 1 failed / 1.12 s / RAW_EXIT 1；先通过静态材料正控，再在转弧得到 unsupported | [03-b-capsule-mesh-red](../../artifacts/p65-top-three-batch5-20261106-01/03-b-capsule-mesh-red/stdout.log) |
| 首次完整 HOST | 19 passed / 15.72 s / RAW_EXIT 0 | [12-b-first-host](../../artifacts/p65-top-three-batch5-20261106-01/12-b-first-host/stdout.log) |
| TYX interpret | 1 passed / 4.91 s / RAW_EXIT 0 | [13-b-interpret](../../artifacts/p65-top-three-batch5-20261106-01/13-b-interpret/stdout.log) |
| 真实 TYX CUDA | 1 passed / 5.46 s / RAW_EXIT 0 | [14-b-cuda](../../artifacts/p65-top-three-batch5-20261106-01/14-b-cuda/stdout.log) |
| 最终 Ruff | RAW_EXIT 0 | [19-abc-lint-final](../../artifacts/p65-top-three-batch5-20261106-01/19-abc-lint-final/stdout.log) |
| 最终格式检查 | RAW_EXIT 0 | [20-abc-format-final](../../artifacts/p65-top-three-batch5-20261106-01/20-abc-format-final/stdout.log) |
| 仅样式变化资格 | 非 import 语法树及作用域内 import 绑定全部相同，RAW_EXIT 0 | [21-qualified-style](../../artifacts/p65-top-three-batch5-20261106-01/21-qualified-style/stdout.log) |

物理三组绑定的 child HEAD 为 `58a5bf511f56f61f1aa1b01e82be4d1750189db4`，已合父薄 hook `bcc2b15`。首轮静态检查的 import 排序（15）与格式（16）差异原收据保留；父仅在17/18修样式，19/20/21确认完成，提交 `7a6fa0e`。没有数值、输入、oracle 或容差修复，也未为样式重跑物理。父正常合入 `49b2117`，B 实际范围恰为本文五个新文件及父独占的薄分派。

本次通过包括两端分离的杆身首触、真轴变化与斜轴/移动枢轴留出、独立有限段 oracle、凹U槽净空、初始包含和开离/闭入、有限端点与无限线区别、镜像/换序、重复局部ID、容量/事务重试及设备失败发布保留。设备专项将三世界完整 `pairs`/`bounds` 与 HOST 对照，并核对实际完成的设备调用。A 作者完成本片独立文本交叉审阅，无阻断发现。

能力划分保持：**HOST 执行有限特征窄相、实体判定和时间包围，TYX 执行真实扫掠宽相**。纯擦触可显式 unresolved；一般双方独立转轴仍 excluded/unsupported；本片未接触力、冲量响应或摩擦历史。没有 Newton 连续首触对照，没有性能排名或全 GPU CCD 声明。首次 RED 未被覆盖，作者原 UNEXECUTED 记录保留为交付时状态；上述父实测为本批最终验证状态。