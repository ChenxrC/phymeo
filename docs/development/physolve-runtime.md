# PhySolve 原生运行时实验

**资格状态：未通过完整机器人验收，不可用于消费者部署。**
Panda 的 B1g 固定输入与官方参考复测见[初始加速度诊断](#b1g-panda-初始加速度诊断)。
后续 B1h 的独立几何审计与显式配置见[Panda连接处规则](../architecture/native-assembly.md#b1h-panda-显式连接处规则)；不改原始无规则失败。
后续 B1c 已增加显式连接处排除配置和 snapshot 的 masked reset 保持，见
[装配合同](../architecture/native-assembly.md#b1c-显式排除与观测保持)；以下无规则
机器人历史失败仍保留，不自动推广到完整机器人、C ABI 或 Graph 资格。
原始机器人原生对照存在超过既定 1e-6 门限的数值差异。B3 修复两处 TYX 初始化
标记缺陷后，接触点几何回归通过、原 FANUC CUDA 719 在完整矩阵中不再出现，
但 FANUC step 后速度全零，与 CPU 参照不符。B4 的同载荷 Warp 和官方求解函数
对照也复现该停滞；固定输入的 Panda 求解存在顺序敏感性。尚无满足冻结载荷与原
门限的修复，不能将失败豁免为通过。未缩小验收范围或放宽门限；详见配套 P2PE
`docs/phymeo-integration.md` 与实验 evidence。本入口保留用于定位，不能据可调用
step/forward/reset 认定整条物理链路正确。

此分支提供 `PhySolvePayload` 与 `PhySolveRuntime`，将 `compile_mujoco` 的完整
MuJoCo 编译快照送入独立 PhySolve Host。Newton、Warp、MuJoCo-Warp 不参与导入或
执行；`physolve_tyx` 保留 MuJoCo-Warp 算法来源告知，其外围内核通过 TYX 编译。
六件求解 CUBIN 和 Host 二进制由显式路径及 SHA-256 指定，不随 PHYMEO 自动下载。

## 依赖和调用

使用 `requirements/physolve-runtime.txt` 专用清单，不混用原 MuJoCo extras。
限定 Windows x64、标准 CPython 3.13.13、MuJoCo 3.8.0、TYX 0.1.0a0、
PhySolve TYX frontend 0.0.0a6 和本批验证的 sm_120 载荷。
`PhySolvePayload(native_path, cubin_directory, native_sha256, cubin_sha256)`
接收一件 Host 和六件 CUBIN 的冻结散列；这不是正式 SDK 安装 manifest 的替代品。

```python
from phymeo import PhySolveRuntime
from phymeo.builders.mujoco_compiler import compile_mujoco

record = compile_mujoco(robot_path, require_asset=False)
with PhySolveRuntime(record, payload=payload, nworld=8) as runtime:
    runtime.step(qfrc_applied=forces)  # float32 可表示的 [8, nv]，N/N·m
    state = runtime.snapshot()       # 显式同步的独立只读 host 快照
    runtime.forward()               # 更新派生结果，不推进 qpos/qvel/time
    runtime.reset(mask)              # bool[8]；不传则全量恢复
```

URDF 不自动生成 actuator：`ctrl` 使用原生 `[N, nu]`，外力使用 `[N, nv]`。
坐标保留 MuJoCo 原生 q/DOF 顺序，自由根四元数 WXYZ。P2PE 的 XYZW、public name、
joint force 和 reaction wrench 契约需要下一阶段显式适配，不能直接暴露原生数组冒充完成。
`xpos/xquat/qfrc_*` 是 solver 最近一次求值结果；step 后不追加 CPU forward。

## 所有权与失败

每个实例独占模型、两个持久 data bank、两个 Host handle 和一条 TYX CUDA stream。
每次 step/forward/reset 从已发布 bank 复制积分状态到候选 bank，验证完成才发布。
只复制 time、qpos/qvel、act、warmstart、控制/外力、equality 和 mocap 积分字段；
派生量由 Host 计算。循环遍历固定字段，不逐 world 调度。
模型只在准备时转换；CPU MuJoCo forward 仅用于初始 data 准备。

当前 API 是同步 direct 验证接口：有明确的 stream completion、容量和有限值 host 检查，
快照包含 D2H，host 控制写入包含 H2D；reset 使用持久 device selector 和完整积分备份，
完整初态和未选观测限制见[原生装配](../architecture/native-assembly.md)。
复制工作随 N×状态宽度增长。这些成本公开保留，不能作为 P2PE 高频 executor 或性能结果。
后续应与 P2PE execution epoch gate 和 device 检查接合，避免第二套事务与同步。
关闭失败保留未释放资源可重试；进入关闭状态后拒绝再次推进。

当前限定 Euler、Newton 约束算法、dense Jacobian、同构 world；不接受插件、flex、
userdata 或原生 sensor。`graph_conditional=True` 选择内部约束循环：未处于捕获时，
Host 读取 device 条件并同步 stream 后继续循环；实际捕获时才登记条件 Graph。
因此此 direct 验证入口不能称为已经执行或资格化整步 CUDA Graph。
外部 stream、异步/D2D 接口、任意状态写入、传感器、整步 Graph 与 P2PE C ABI 尚未接入。
`require_asset=False` 保留完整原生碰撞/接触数据，不意味着 resident articulation lowering
已支持所有 geom、equality 或 actuator。

## 正式测试入口

`tests/test_physolve_models.py` 比较 Panda/FANUC 全部可枚举原生数组、标量及 option，
并验证 q/v 映射。MJB 不保留 MjSpec signature，测试明确验证其为 0；其他字段不放宽。
`tests/test_physolve_runtime.py` 使用实际 Host/CUDA，覆盖 N=1/8/64、自由落体、外力、
机器人单 bank 原生对照、forward/reset、接触与失败发布/清理重试。

设置 `PHYMEO_PHYSOLVE_SOURCES` 为显式来源 JSON 路径。它包含 `native_host`、`cubins`、
`robot_assets.fanuc`；同级 `evidence/payload-baseline.json` 包含 `native_host.sha256`
及六件 `cubins` 的 name/sha256。无配置时 GPU 用例明确 skip，不算通过。
测试输出和逐字段报告保留在实验根 evidence；与旧 Newton 转换存在的差异必须在 P2PE
场景编译适配阶段处理，不能据相同 nq/nv 认定原始 URDF 与旧转换模型物理等价。

最终安装环境复测使用 fail-fast：43 通过、1 失败、13 项未继续执行；失败为 Panda
首次 forward 的 qacc 对照，最大差 296.59686，容差仍为 atol=1e-6、rtol=0。
已安装 TYX wheel 的相关七模块 101/101 通过，但不足以覆盖下游碰撞/约束组合。
Panda 指尖接触点另有约 0.413 m 几何越界证据，保持失败，不作为舍入误差豁免。

后续 B3 安装环境矩阵：50 通过、9 失败，无 skip。Panda/FANUC N=1/8/64
均实际执行，Panda 的独立 step/forward/masked/full reset 三项通过，CPU/CUDA
指尖接触点对照通过；六项原生对照与三项 FANUC 运动检查仍失败。
FANUC 的同配置 CPU 参照确实运动，不能删除 motion 检查让静止结果通过。

B4 未修改生产代码，只做阶段归因与固定输入重放。Panda 两 bank 的质量矩阵、
无约束力/加速度完全一致；接触 geom/dist/pos/frame 与约束 J/D/aref/pos 按内容
排序后也一致，原始顺序不同。固定约束输入交给冻结求解 CUBIN，保持顺序重复仍有
约 6.10e-5 的 qacc 差，反转同类型约束行产生约 5.7604 的差；官方 Warp 求解函数
也复现。排序不能据此被当作完整修复，更不能放宽 1e-6 原生对照门限。

FANUC 的四个初始自接触形成 16 条约束，总代价约 3.28e10。实测原生和官方求解
均在首轮 alpha=0、前后 float32 代价相同的情况下停止。NumPy 重建目标函数时，
从零加速度到无约束加速度的 float64 改进约 471.96，float32 代价 ULP 为 2048，
改进被舍入为零。该重建不是 CUDA 归约的逐位仿真，需与实际 solver context 证据
共同解读。禁用接触或 warmstart 的运动恢复仅为诊断，不进入产品默认行为。

证据为实验根 `evidence/b4-*`，详细矩阵和后续边界以配套 P2PE 集成记录为准。
本批没有新增运行时结构优化，也未重新执行完整 B3 验收矩阵；C ABI、整步 Graph、
GUI、安装包与性能资格仍未推进。六件 CUBIN 和 Host 保持冻结。

## FANUC 碰撞模型审计（B5）

后续设计见[机器人碰撞规则补齐方案](robot-collision-rules.md)：复用 PHYMEO 现有
形状身份和显式排除机制，分别资格化 resident 与完整 native 路线。
C0 已实现来源映射和排除计划合同，尚未将计划施加到运行期候选碰撞表；执行过滤属于 C1。

本轮独立使用用户原始 URDF/STL 和 `fanuc.zip`，不采用 Newton 导出模型或过滤
配置作为标准。归档与 16 件文件散列一致。底座和第一节的 collision STL 在 URDF
初始坐标下本身相交：竖直重叠约 0.06700005 m，三个点的封闭网格绕数与独立
射线奇偶检验均确认位于两者内部。这不是仅由凸包填充产生的假象。相应 visual STL
在 z 方向有约 0.00098456 m 间隙；visual 只能作为资产表达的参照，不是制造商 CAD 认证。

原生 MuJoCo、PHYMEO 编译模型和 MJB 恢复模型的 592 个可枚举字段一致（MJB
signature 的已记录例外不变），两个 mesh 的世界顶点与独立 URDF/STL 坐标误差
小于既有 1e-6。PHYMEO 未引入额外偏移或穿透；`require_asset=False` 使用完整原生
模型，未借用 resident articulation-only 碰撞实现。

第一轴全范围取 9 点，均保留四个约 -0.067 m 的接触；接触法向雅可比最大绝对值
约 3.44e-15，没有可用关节速度消除该法向穿透。因此将这对代理重叠当作应被推开的
关节连接处接触约束，不适合作为正常机器人动作场景。资产未说明是否刻意依赖相邻
过滤，不能据此宣称原 STL 作者的意图或全部机械几何已经验收。

MuJoCo 原生 URDF 导入把固定根融合到 world，PHYMEO 保持该语义。即使诊断性关闭
fusestatic、保留 `base_link` body，它仍属于 world weld（weldid=0），接触保持不变。
当前父子自动过滤不适用于 world weld，所以仅保留 body 名称不足以解决问题。需要明确的
robot link/geom 级碰撞关系；不能用 world 与 link_1 的 body-wide exclude，否则
可能同时屏蔽地面/障碍物。该规则尚未进入生产导入器。

正式 pytest 新增 `tests/test_fanuc_collision_model.py`，四项 CPU 几何/导入/过滤
审计和既有两项模型逐字段测试全部通过。定向 GPU 诊断仅用 mask 移除底座—第一节
这一对，断言其他 geom pair 的 mask 关系保持不变；复用原测试的 0.03 N/N·m、
三步、forward 和 masked/full reset 断言，N=1/8/64 三项均通过。未放宽容差、
未禁用 warmstart/全部接触、未更改 CUBIN。CPU 外部平面负控制确认 link_1 与
world 障碍物的接触仍存在；该外部平面没有单独做 GPU 物理资格。

同批保留的原始 FANUC 配置三项 GPU 用例仍失败于速度全零，整体为 3 pass/3 fail，
24 项未选；不能把定向诊断当作原始配置或完整机器人资格通过。B4 对数值敏感性的
观察仍保留，但“先修改求解内核”的优先级判断由本轮纠正：先确定模型/场景碰撞
契约，不以旧 Newton 配置为真值。Panda、C ABI、Graph、GUI 和完整 release 未复测。
证据在实验根 `evidence/b5-*`；本轮只增加测试和记录，生产默认行为未变。

追加定向 GPU 复测 3/3 通过，逐 world 检查均运动，同构 qpos/qvel 在原 1e-6
门限内一致；原有 forward 和 reset 精确比较保留。运动恢复不等于原始模型全部
动力学参数或机器人资格合格。六件 CUBIN 和 Host 的 B5 散列复核均与基线一致。

全库 `scripts/check_docs.py` 检出 479 条缺失目标，当前输出与本批修改前 HEAD
内容的检查逐行相同，均为已有 artifact/页面缺失；该门禁仍记失败。本轮不补造历史
制品或改动无关页面。修改文档中的路径/事实、测试源和全部 diff 已另行复核。

## B1g Panda 初始加速度诊断

本批保持原始 Panda 零位、碰撞配置、六个 CUBIN、Host 和测试绝对容差
`1e-6, rtol=0`。生产代码未修改，数值阻断未关闭。候选安装在独立 `env-b1g`，
依赖文件与 B1f 的 7090 件文件逐件一致，没有 Newton、Warp、MuJoCo-Warp。
下列探针只用于定位，不替代正式 pytest，也不构成 C ABI 或 Graph 资格。

### 输入与所有权排查

冷准备的 CPU 数据和 device 初值一致；第二 bank 的创建和 reset 准备没有改写
第一 bank 的已发布观测。完整 Data/contact/efc 共 120 个数组或标量快照中，
同一次 forward 后立即再次 solve 的差异只出现在求解输出。`ne/nf=0`、`nl=1`、
`nefc=41`、`nacon=10`，有效 contact 与约束行映射自洽。只检查 J/D/aref 或只跑
一组固定输入不够：最初八次 solve 逐位一致，后续同输入探针仍可复现输出差异。

初始 direct 的条件求解循环读取紧凑条件并同步绑定流，没有捕获整步 Graph。
未发现以增加双 bank 镜像、重绑、全局同步或重新初始化全部工作区解决问题的依据。
只读中间快照会改变调度时序，其结果单独保存，未替代未插桩的正式测试结果。

### 代价精度与停止条件

零位模型含 `panda_link0`（融合进 world）与 `panda_link1` 的代理形状接触，
穿透约 0.0118–0.0600 m。有效约束中有 8 行 Jacobian 严格为零，贡献约
`6.9691e9` 的加速度无关代价；总初始约束代价约 `1.3938e10`，float32 间距为
1024。这是当前模型的实际输入，不据此推定资产作者意图或自动添加过滤规则。

冻结求解核以 float32 原子累加代价，并按
`(prev_cost-cost)/(meaninertia*nv) < tolerance` **或**归一化梯度小于容差停止。
本模型CPU solver tolerance原值为 `1e-8`。两套GPU前端会将其提升至至少 `1e-6`，
B1j另存实际GPU opt并核实此转换；这与测试的绝对误差门槛是不同概念。
公开原生 context 的固定输入探针实际得到：

| 量 | 原生 PhySolve context | 官方 MuJoCo-Warp context |
| --- | --- | --- |
| 结束时代价 | 约 `1.3938153e10` | 约 `1.3938154e10` |
| `prev_cost-cost` | 0 或 -1024 | 0 或 -1024 |
| 归一化梯度 | 约 0.11109 | 约 4.80971 |
| 梯度是否达到实际GPU容差 `1e-6` | 否 | 否 |

因此这些样本确实由代价差条件停止，不能称为达到梯度收敛；原生 context 的八次
qacc 最大变化 `7.6293945e-6`，已超过原测试门槛。上述探针复用了现有求解入口，
没有更改容差、迭代上限或 CUBIN。首次直接使用可选 legacy context 入口时出现
misaligned address；该入口仍直接消费旧 `__ctype__` 描述符契约，后续仅在探针中
显式投影冻结描述符后成功取值。兼容缺口及其完整因果仍待独立确认；此可选入口
未修复、未资格化，不能归为原始 qacc 失败原因。

官方参考在独立历史环境使用同一 MJB、dense Jacobian、N=1、nconmax=128、
njmax=256、MuJoCo 3.8.0 / MuJoCo-Warp 3.8.0.3 / Warp 1.13.0；包文件与安装
RECORD 散列一致。六次完整 forward 的 qacc 最大范围 `194.34528`，迭代数为
2/4/5；固定约束的八次 solve 最大范围 `1.9073486e-6`。官方路径也复现了该
模型的顺序/精度敏感性，不代表两条实现输出已数值一致，也不豁免候选失败。

### 修复边界与验收

当前证据确认了模型大常数代价与 float32 停止判据的数值问题，尚未逐核记录所有
差异的首次发生点。没有证明可在保持模型、算法、容差及六 CUBIN 不变时，通过
PHYMEO owner 或 Host 互操作调整保证原门槛。改变归约精度/停止计算属于求解数值
实现变更；用 Host 重算并覆盖代价同样不是普通绑定修复。本批不实施此类替代。

后续先独立审查 Panda 连接处碰撞关系；若需排除，应由显式模型规则经 PHYMEO
公共转换表达，保留无规则失败及外部接触负控制。若必须保留完全相同的原始约束，
则需单独评审求解数值实现的范围变更。不能关闭接触、改初态、重复求解挑结果、
强制迭代或放宽容差来关闭本项。正式 N=1/8/64 结果、独立复审与制品身份见
配套 P2PE Stage 批次记录及实验根 `evidence/b1g-summary.json`。

## B1h 显式规则后的剩余数值差异

本节只分析显式底座连接规则配置；原始模型的B1g结论保持独立。
正式N=1/8/64结果见[模型与规则记录](../architecture/native-assembly.md#b1h-panda-显式连接处规则)。

N=64固定约束实验在同一个既有原生实例直接调用8次 `mw_phycore_solve`，不执行
完整forward、不切换bank、不重新生成接触。每次采集的qacc及其他结果完全相同，
并逐位确认qpos/qvel/warmstart、控制/外力、质量矩阵及分解、无约束加速度/力、
约束J/D/aref等输入没有变化。证据为 `b1h-fixed-solver.json` 与对应NPZ。
因此不能把正式矩阵差异直接归因于固定输入下的求解核不确定性，也不能由这8次
观察承诺所有输入下求解确定。

该快照64个世界各有25条约束；J/D/aref内容按行排序后完全相同，原行序仅51个
世界与world0相同。其余13个世界均出现qacc差6.1035156e-5，其他已采集积分输入、
控制、质量矩阵及无约束加速度/力逐位相同。这是约束行序与差异的关联证据，
不是已经证明某个特定kernel错误；需结合完整forward前后边界继续定位。

完整forward的control回调采样在新进程完成：两个bank及随后4次forward的最终
qacc完全一致，没有复现正式无采样矩阵的数值失败。全局contact槽位/world排列、
efc.id及contact.efc_address发生变化，但按world/geom/距离/位置/frame/dim组成的
接触记录排序后内容一致；每world的J/D/aref/type联合行序也完全一致。创建第二bank
及reset准备均未改变第一bank已采集输出。证据为 `b1h-trace.json`、
`b1h-trace-analysis.json` 及原始NPZ。

该回调包含同步host读取，会改变调度；它只能证明这次采样下接触槽位重排没有
改变物理内容，不能覆盖正式失败或排除未采样路径的行序变化。回调位于
fwd_velocity之后、actuation/acceleration/solve之前，其中旧qacc_smooth、qLD等
尚未刷新，不能把重复回调与冷回调的这些差值称为本轮求解输入的首个错误。

下一步应以完整约束输入冻结和一致的排列/逆排列对照验证行序的因果作用，再判断
是否存在可行修复及其所属层；尚无依据改动P2PE状态owner、TYX数组复制或
加入生产同步。不能增加采样同步、重复求解挑结果或更改容差来使正式矩阵通过。

## B1i 约束行排列的受控实验

在独立安装 `env-b1i` 中冻结显式连接规则的 Panda native 输入，分别使用N=1/8/64。
本实验只覆盖dense Jacobian、pyramidal cone、每接触4条约束的当前模型；不是新的
生产排序策略。保留equality/friction/limit前缀，只将接触的四行整块逆序，随后恢复
原序。所有行字段一起移动，`contact.efc_address` 的每个有效lane同步更新；
`efc.id`仍指向原接触槽，接触几何、摩擦、solver参数和积分输入保持不变。

`efc.Ma`属于DOF空间，不随行排列；dense布局的sparse索引为空。每次安装后读回
检查原始/排列数组及地址，逆排列必须逐位恢复。保存全部行输入、接触字段、
排列/逆排列、模型哈希与原生输出，以便独立复核，不只记录一个最大误差。

同一原生实例执行“原序两次→逆序两次→恢复原序”。同序重复及恢复是负对照；
它们不一致时不得把原序/逆序差异全部归因于排列。负对照结果先保存再硬失败。
这是诊断用的同步host拷贝实验，不加入热路径、不改变现有容差或替代原正式矩阵。

源码审查未发现可启用的确定EFC行序入口：接触约束仍用原子加法预留行块；SAP
排序只作用于几何候选，sensor排序只处理传感器匹配ID，均不保证原子预留顺序。
因此不能把已有排序名称当作已具备求解输入确定性，也不预定新增生产排序。

### 已完成的排列与规范排序诊断

三档规模均为每world 25条约束、6个接触。所有原序重复、逆序重复和恢复原序
负对照逐位相同；固定输入及接触字段检查通过。仅逆序即出现以下差异，证据为
实验根 `evidence/b1i-order.json` 和各N的完整输入/输出NPZ：

| N | qacc最大差 | qfrc_constraint最大差 | 逆映射后的efc.force最大差 |
| --- | --- | --- | --- |
| 1 | 3.0517578e-5 | 3.8146973e-6 | 6.1035156e-5 |
| 8 | 6.1035156e-5 | 2.3283064e-9 | 6.1035156e-5 |
| 64 | 3.0517578e-5 | 3.8146973e-6 | 6.1035156e-5 |

这证明这些固定输入对行排列敏感，不再只是关联；不证明全部正式失败均由行序
造成，也不把排列后的结果当作原始物理资格通过。

随后在诊断回调中按geom pair和本次接触生成序号 `geomcollisionid` 规范排列行，
并以相同host读写但保持原序的前、后两臂作对照，复用未修改的正式
`tests/test_panda_collision_runtime.py`。三臂各为N=1通过、N=8/64失败，共3通过、
6失败、无跳过。规范排序臂初态对照通过，但N=64 forward的qacc最大差仍为
0.0285050869，不能宣称充分修复。该序号不是公共实体身份，不据本次唯一键检查
承诺跨任意forward的几何对应关系。完整阶段结果由JUnit和日志保留。

这些回调包含同步host传输及逐world诊断循环，未进入生产；前后臂仍失败表明
不能仅靠增加这类读写消除门禁。三臂存在进程/模块/缓存历史差异，不作性能
或安装资格对照。首版驱动复用禁包import guard导致第二臂preflight自触发，
没有实际导入禁包；保留该错误，修正驱动后补跑其余两臂，没有覆盖首臂结果。

设备端暂存与TYX公共分段排序原型仅完成语法检查，未运行、未加入仓库生产代码。
鉴于仅接触行的host规范排序仍失败，本批不将该原型包装成候选修复，也不为尚未验证的
方案增加常驻缓冲、复制和kernel调用。六CUBIN、Host、模型初态和原容差均不变。

### 同输入双实例与首次分歧定位

在初态和推进后的两个固定快照上，将枚举到的114个Python可见Data/contact/efc
数组分别逐位安装到两个原生实例，交替各求解3次；qacc、qfrc_constraint、
solver_niter和efc.force均逐位相同。证据为 `evidence/b1i-twins.json` 及输入/输出
NPZ。这不包含Host私有工作区和sidecar，也不证明所有时刻的确定性；只说明这两个
快照未表现出仅因bank身份而产生的差异。推进快照具有前述诊断的私有状态历史。

随后在原N=64正式用例的control回调及native step/forward返回处采集114个数组，
保留接触行规范排序。原用例仍失败，日志为 `evidence/b1i-chain.log`。初态与第一步
物理结果一致；第二步进入control时，qpos、qvel、warm-start、qM、bias、passive、
ctrl和外加力逐位一致。world 3/10/42/60的两条标量限位约束顺序交换，原生joint
ID为3和7；对齐这些行后，所有有效行的type/J/pos/margin/D/vel/aref/frictionloss/
Jqvel逐位一致。该步输出在world 10/60首次出现qacc差异，最大6.1035156e-5。

该定位说明只规范接触行遗漏了限位行，并未证明全部剩余误差都由限位行引起。
control边界尚未刷新qacc/qacc_smooth等派生值，其跨bank差异不能作为错误证据。
采样包含显式同步及host拷贝，不是生产性能或资格结果；完整数组与离线比对保留在
`evidence/b1i-chain-*` 和 `evidence/b1i-limits-analysis.json`。

### 接触与标量限位共同排序的诊断结果

后续host诊断限定 `ne=nf=0`、前缀均为type=3的标量关节limit、各world的limit ID
唯一；仅将该前缀按joint ID排列，接触仍按前述键整块排列。没有泛化到equality、
friction、tendon limit或任意cone/sparse布局。原正式Panda测试N=1/8/64得到
3通过、0失败、0跳过，75次回调，覆盖initial、六步推进、forward、状态不变、
masked/full reset和精确replay。日志/JUnit为 `evidence/b1i-limits-host.log` 和
`evidence/b1i-limits-canonical.xml`；容差及断言未修改。

此结果只属于带host插桩的显式连接规则Panda模型。它支持继续研究完整约束行序，
不能关闭B1h未插桩失败，不能宣称设备执行、固定Graph、原生重建或生产性能通过。
没有将Python控制回调、逐world排序、同步快照或全行数组复制写入生产。

候选设备实现必须先明确各类别的稳定key和块语义，再在冷准备阶段建立按bank拥有
的暂存与索引；保持contact槽位引用及所有地址一致，未知布局明确拒绝。需要验证
无host中转、无新增全局同步、固定Graph、错误gate、绑定失效与资源释放，并量化
排序、gather/scatter、复制及额外常驻容量。不能将当前Panda特例当作通用合同，
也不能为消除同实现回归而偷偷改变公开solver默认行为。

补充因果对照冻结N=64第二步reference输出的114个可见数组，仅逆序标量limit
前缀，接触、地址及尾部保持不变。原序两次、逆序两次、恢复原序的负对照均逐位
一致；逆序相对原序的qacc最大差6.1035156e-5、qfrc_constraint最大差4.1961670e-5，
solver_niter最大差1，逆映射efc.force最大差1.0681152e-4。它证明该固定输入上的
限位排列敏感性，但使用的是步后快照再次solve，不冒充原第二步pre-solve的精确
重建。完整输入、排列、输出及模型来源散列见 `evidence/b1i-limit-order*`。

## B1j 官方与原生同条件对照

正式pytest入口为 `tests/test_native_comparison.py`，显式设置
`PHYMEO_COMPARISON_ROUTE=warp|physolve|runtime`、`PHYMEO_COMPARISON_CASE`、
`PHYMEO_COMPARISON_OUTPUT`；后两路线还需要 `PHYMEO_PHYSOLVE_SOURCES`。
未选择此可选矩阵时跳过，显式选择后缺依赖、缺证据或未知路线均硬失败。
离线跨路线检查使用 `tests/test_native_comparison_results.py`，证据目录通过
`PHYMEO_COMPARISON_RESULTS` 指定；属于同一正式pytest体系，不替代既有机器人门禁。

在实验根先冻结B1h显式九对排除的Panda MJB、完整11项integration初态及N=1/8/64
动作数组。原始无规则模型的B1g失败继续单独保留。两套独立安装分别固定官方
MuJoCo-Warp 3.8.0.3/Warp 1.13.0和现有无三包PhySolve载荷，CPU MuJoCo 3.8.0、
NumPy 2.2.6、CPython 3.13.13相同；不导入Newton、不修改六CUBIN或生产排序。

三条路线分别是官方原生单Data、PhySolve原生单Data、PHYMEO Runtime双bank。
每档三次独立实例，固定Euler、dense、pyramidal、2ms、六步、nconmax=128、
njmax=256；均使用各自唯一owned canonical stream。`graph_conditional=True`
只选择求解循环实现，不表示捕获整步Graph；本批无整步捕获和control回调。
每次实例保存全部实际GPU opt，离线比较必须逐字段相等且不能含未处理类型。
CPU模型参数、转换后GPU参数和验收容差分别记录，不能只由MJB相同推定完全同条件。

各路线在相同观测点完成绑定stream并保存OBS、integration、原始contact/EFC与
必要派生诊断数组。首轮和replay均每步使用相同采样节奏；有限差异保留到最终硬
失败，shape/dtype、非有限值、容量和执行错误立即失败。原1e-6绝对容差与精确
qpos/qvel/time replay要求不变，保留每world实际运动及12ms时间推进负控制。

两条原生路线的masked/full reset由测试工具显式恢复初始GPU integration快照，
执行forward并保留未选OBS；含host读写和同步，不是官方masked-reset产品能力。
Runtime路线使用生产reset，选中世界完整11字段精确恢复、未选OBS逐位不变分别
检查。raw contact/EFC仍是全量forward结果，不冒充未选contact发布合同。
Runtime构建两bank会执行两次初始化GPU forward，原生路线每实例一次；Runtime
还具有自身完成/检查。这些组织差异如实保留，不能把跨路线差异直接归为求解核。

本矩阵不测吞吐，诊断快照会影响调度；同机另有RC7测试任务，未干预其进程。
同实现的跨实例差异与跨实现差异分别判定，官方也失败不关闭候选失败，差异范围
重叠也不证明同一根因。没有验证的阶段或能力继续保持未通过/未执行状态。

### 重复性及跨实现结果

27项实际GPU opt在三条路线、三档规模和三次重复间全部相等；包含转换后的
float32 tolerance约1e-6、ls_tolerance约0.01及graph_conditional。各原生DOF按
对应SI单位比较，保留原绝对门槛，不根据候选误差重新选择容差。

三条路线的正式重复性用例均为N=1通过、N=8/64失败，共3通过、6失败、无跳过。
以下是同一路线新实例相对其第一次实例的qacc最大绝对差，不是跨实现误差：

| 路线 | N | 初态 | 六步中的最大差 | forward |
| --- | --- | --- | --- | --- |
| 官方MuJoCo-Warp | 8 | 0 | 9.15527e-5 | 4.57764e-4 |
| 原生PhySolve | 8 | 6.10352e-5 | 6.10352e-5 | 1.52588e-5 |
| PHYMEO Runtime | 8 | 6.10352e-5 | 6.10352e-5 | 2.28882e-5 |
| 官方MuJoCo-Warp | 64 | 6.10352e-5 | 5.67627e-3 | 1.83868e-3 |
| 原生PhySolve | 64 | 1.22070e-4 | 7.32422e-4 | 1.84631e-3 |
| PHYMEO Runtime | 64 | 1.52588e-5 | 6.71387e-4 | 1.83105e-3 |

N=1各路线跨重复的全部OBS均精确相同。三条路线均通过所检查的forward积分状态
保持、selected/full完整integration恢复和未选OBS精确保留；N=8/64仍有派生值
超门槛及精确replay失败。原生两路线的这些reset结果只资格化本诊断恢复工具。

离线跨路线正式用例为1通过、5失败、无跳过：官方与原生PhySolve的三档全部失败；
原生PhySolve与Runtime的N=1全部OBS精确一致，N=8/64失败。主要qacc差为：

| 比较 | N | 初态 | 六步中的最大差 | forward |
| --- | --- | --- | --- | --- |
| 官方—原生PhySolve | 1 | 6.10352e-5 | 5.02014e-3 | 6.10352e-4 |
| 官方—原生PhySolve | 8 | 1.22070e-4 | 4.94385e-3 | 5.95093e-4 |
| 官方—原生PhySolve | 64 | 1.83105e-4 | 5.92041e-3 | 2.51770e-3 |
| 原生PhySolve—Runtime | 8 | 0 | 6.10352e-5 | 2.28882e-5 |
| 原生PhySolve—Runtime | 64 | 6.10352e-5 | 7.17163e-4 | 1.84631e-3 |

共同限制已有直接证据：三条路线N=64的r1相对r0初态，语义匹配的contact及
J/D/aref等内容相同、qM和warm-start等已采输入相同，但原始行序和qacc不同。
这不是完全相同的有序输入，也不证明固定输入求解核非确定。

另有跨实现残余：N=1各自稳定，但官方与PhySolve之间已有qM约7.45e-9、qLD约
1.49e-8、qacc_smooth约1.91e-6，以及contact frame/J的微小差异；该样本语义
行序相同，不能把所有差异归为重复性波动。它继续阻断原跨实现门槛，尚不能仅凭
这些幅度判定为迁移代码缺陷或无害舍入。Runtime的N=1与原生PhySolve完全相同，
且未发现本矩阵中的状态/控制恢复或未选OBS合同破坏；不扩展为全部owner正确性证明。

### 冷态输入定位与原门禁复测

另起N=1进程，在put_model/put_data之后、原生prepare/首次GPU forward之前保存
全部可见GPU数组：432项逐位相同，shape/dtype/stride、标量、实际选项及block_dim
元数据均相同。CPU的780项数值数组/标量仅两个未使用的稀疏efc_AR行索引工作区
不同；本模型采用dense，efc_AR及colind为空。这些未初始化工作区不作为物理差异。

首次GPU forward后，ximat最大差8.94070e-8、geom_xmat为2.98023e-8、cinert为
3.72529e-9，crb/qM为7.45058e-9，qLD为1.49012e-8；qacc_smooth为1.90735e-6，
qacc为6.10352e-5。由此排除已采上传值/布局不一致，收窄到GPU执行实现的派生
计算差异。完整forward快照不定位时间上最先不同的算子，不证明所有qacc差都由
ximat导致，也没有覆盖Host私有描述符或冻结CUBIN编译选项。不能据此判定TYX
编译器缺陷、认定无害舍入，或把当前codedev源码当作冻结二进制的完整来源证明。
原始输入、布局和结果见 `evidence/b1j-cold-*`。

原有未插桩 `test_panda_collision_model.py` / `test_panda_collision_runtime.py`
最后复跑为4通过、2失败、无跳过：三个模型审计及N=1通过，N=8/64仍在原数值和
精确replay门槛失败。原文件未改。对照新增的采样节奏不替代这一门禁。

两套环境pip check及载荷身份检查通过。首次官方环境预检曾把工作区同名目录识别
为namespace；两次冷态探针分别缺少Warp初始化、误用Stream完成方法，均在首次
物理forward前中止。测试工具修正后重跑成功，前置失败保留而不记为solver失败。

下一步先做N=1旋转矩阵/惯量变换的最小算子及实际二进制来源对照，区分浮点执行
与实现错误；约束行序问题独立保持。没有依据增加生产同步、改变owner、降低
容差或直接投入排序。B1/B2仍部分完成，未完成C ABI、整步Graph、传感器、GUI、
完整release或性能验收；本批没有生产开销或性能改善声明。

完整diff自审及独立代理只读复审完成，证据索引为实验根
`evidence/b1j-summary.json`，意见与采用情况为 `evidence/b1j-review.md`。
针对新增测试的Ruff检查通过；PHYMEO文档检查仍为479条历史缺失目标，输出散列
与原基线一致，保持失败。提交仅含测试及本主题记录，不推送、不合并。

## B1k 物理影响评估

本批限定为独立物理参照和两秒机器人轨迹，不修改生产排序、旧数值容差或六件
CUBIN。复用隔离安装 `env-b1j` / `env-b1j-warp`，入口逐件核对安装RECORD；
使用独立B1k缓存与证据，原环境不安装/升级。正式测试为
`tests/test_physical_impact.py` 和 `tests/test_physical_impact_results.py`，显式
设置 `PHYMEO_IMPACT_ROUTE`、`PHYMEO_IMPACT_CASES`、`PHYMEO_IMPACT_OUTPUT` 及
`PHYMEO_PHYSOLVE_SOURCES`；未选择才跳过，显式请求缺依赖硬失败。

所有模型、完整初态、控制数组和判据在首条候选结果前冻结至
`evidence/b1k-cases/protocol.json`。三GPU路线N=1/8/64顺序执行CUDA direct；
CPU仅隔离计算Panda参考，不推进GPU实例。独立物理参照沿用既有测试的有量纲门槛：

| 案例 | 参照及范围 |
| --- | --- |
| 自由落体 | 2kg球、-Z重力9.81、0.2s，dt=1/0.5ms；半隐式Euler位置3e-5m、速度2e-5m/s；连续解位置误差减半比0.4～0.6 |
| 恒力滑块 | 2kg、无重力/约束/阻尼，100步×2ms，逐world不同力；独立离散q/v递推1e-6 |
| 重力补偿关节 | 2kg、COM偏心0.3m、力矩-5.886N·m，0.5s；角度1e-4rad、速度0.001rad/s |
| 球落地支撑 | 2kg、直径0.2m，4s；末0.5s速度0.001m/s、穿透0.002m、平均支持力mg±1%，横向力范数亦不超过1%mg |

支撑力为该自由体的约束广义力平移分量；不是P2PE contact/SensorView接口资格。
每world单独检查，保存逐步速度和力，不以均值隐藏振荡。时间逐步递增和局部
float32 ULP检查用于识别漏步/重置；另记录累计时间误差。

Panda stress保留B1j零初态、九对显式排除、Euler和world-scaled恒力，延长至
1000步×2ms。独立motion案例使用明确的限位内初态、小幅往返目标、固定CPU bias
前馈及host PD，不替代stress，不修改模型/接触；同策略闭环实际动作会随路线状态
不同，不能称相同动作输入。CPU控制输入也转换为float32，CPU物理仍为float64，
CPU容差1e-8与GPU前端约1e-6的差异保留，因此CPU只是隔离参照。

原跨路线q/qvel/qacc的1e-6门槛单列。额外1mrad/1mm位置及0.01rad/s或m/s速度
仅为预先登记的影响尺度，按转动/平动DOF区分，不是任务验收容差。末端使用保存的
步后qpos离线CPU FK，取 `panda::panda_hand_tcp` body原点的world位姿；不额外
调用运行期forward，不混用求解时刻contact与步后位姿。有限值、容量、CPU warning、
自动重置和清理/reset失败均明确失败，保留阶段与异常记录。

上述trajectory测试完成只代表有限时段执行及恢复检查完成；不能据此宣称抓取、
训练、长时稳定性、C ABI、Graph或完整机器人资格。生产模型/owner/资源组织不变，
诊断host控制、快照和逐world CPU参考成本不作为性能结论。

用户在本批评估期间明确调整优先级：不再以追清与官方的逐值差异为接入主线任务，
不继续增加算子/编译/生产排序排查。完成已启动评估并保留数值失败记录，随后推进
模型转换、状态/动作/reset、数据通道和生命周期。求解器自身验证优先复用codedev
已有且与实际载荷身份匹配的证据；本批未审计该仓完整测试证据，不能宣称其已被
本批重新资格化。进一步数值根因定位不作为推进独立接口工作的前置条件，原门槛
和最终资格事实不因优先级调整而改写。

### 结果与工作方向

三GPU路线各21项通过、无跳过：合计45项独立物理参照及18项两秒轨迹执行/恢复
检查。另有6条CPU机器人参照完成1000步，无warning或自动重置。离线42项中，
实际选项21项和自由落体收敛9项通过；原1e-6轨迹对照3项通过、9项失败。通过的
三项为motion的原生PhySolve—Runtime；该路线所有q/qvel/qacc及动作逐位一致。
stress三档的两组比较及motion官方—PhySolve仍超过严格数值门槛，没有改成通过。

45项参照中的最坏值：落体位置4.323e-7m、速度3.564e-6m/s，步长减半误差比约
0.49952；恒力递推2.772e-7；重力补偿角度2.986e-7rad；稳态支撑速度6.645e-7m/s、
穿透0.0003672m、平均力误差0.0001519N，均满足事先冻结的既有物理判据。

| 两秒轨迹比较 | N | 最大arm角差(rad) | 最大TCP位置差(m) |
| --- | --- | --- | --- |
| stress 官方—PhySolve | 1 | 3.15905e-5 | 3.74344e-6 |
| stress 官方—PhySolve | 8 | 9.47714e-5 | 1.31172e-5 |
| stress 官方—PhySolve | 64 | 5.74350e-4 | 1.13458e-4 |
| stress PhySolve—Runtime | 1 | 3.81470e-6 | 9.12729e-7 |
| stress PhySolve—Runtime | 8 | 5.14984e-5 | 7.41235e-6 |
| stress PhySolve—Runtime | 64 | 5.25355e-4 | 1.02291e-4 |
| motion 官方—PhySolve | 1/8/64 | 1.60933e-6 | 1.31471e-6 |
| motion PhySolve—Runtime | 1/8/64 | 0 | 0 |

stress N=64官方—PhySolve角差由首0.5s最大1.064e-5rad增长至末态5.744e-4rad；
不能称为长期有界或无害舍入。CPU及GPU共同存在约42.65rad/s瞬态、0.265rad
软限位超出及0.0284m采样穿透。motion的共同跟踪误差约0.184rad，且接触仅每10步
采样；两案例都不构成抓取/训练成功或长期稳定性资格。

本批没有观察到基础物理失效，现有差异不支持继续把底层算子/排序排查放在接入
主线之前。保留严格数值诊断结果，按用户决定停止追加追因，推进实际C ABI接入。
[MuJoCo 3.8官方说明](https://mujoco.readthedocs.io/en/3.8.0/mjwarp/index.html)
已明确GPU原子操作可能导致运行间顺序/数值差异，float32也与CPU默认float64有别；
这些已知边界不应被误当作所有接入工作必须先消除的缺陷。

全部原始轨迹、模型、控制、脚本及身份散列归档于 `evidence/b1k-summary.json`。
无生产修改，无新环境安装；仅复用两套隔离安装。已完成自审及独立只读复审，
P2PE文档/政策18项、Ruff及diff门禁通过；PHYMEO文档479条历史缺失目标保持失败，
日志散列与原基线一致。本节点不推送、不合并。

## B3b：公开存储与候选执行

接口和职责以[引擎集成合同](../architecture/engine-integration.md#字段借用与候选执行-v1)
为准。本批只改 PHYMEO 的适配和公共借用描述，原生 Host、TYX 和六个 CUBIN 不变。
同步 step/forward/reset 保持入口，改为复用候选事务；不改生产排序和数值容差。

既有 env-b1j 从工作分支源码运行，进程同时检查三包未安装并禁止三包导入；不声称
已更新安装 wheel。`test_physolve_execution.py` 的 13 项真实 CUDA 测试覆盖 N=1/8/64、
四子步统一发布、借用权限/stale、丢弃、异常锁存及准备/清理失败恢复；与既有 masked
观测、工作区测试合计 28 通过。9 项历史 installed B1c 对照因未配置参考而跳过，
不记为通过，没有为此新建环境或复制历史大制品。证据见实验根 `evidence/b3b-*`。
原同步入口定向回归另有 11 项通过：落体/forward/reset、命令力原子性、接触响应、
清理重试、驱动控制和双实例隔离。没有重跑机器人严格跨实现数值调查。

执行结构中 begin 只复制一次 integration 字段；计数实测四子步完成仅校验读回 4 字节；
局部 reset 的 forward 与未选观测保持共用一次完成读回。测试禁止准备后调用数组
工厂，禁止新执行路径 `.numpy()` 全量校验；不把这些计数写作完整 CUDA 分配或性能资格。
原生 direct 内部条件同步、CPU 冷初始化和显式 host snapshot 成本仍存在。

独立代理两轮只读复审，已修复准备失败 pending 残留及校验预热失败 owner 保留，
没有剩余阻断。审查者未运行 GPU。Ruff 与 diff 检查通过；全仓文档仍有 479 条历史
缺失目标，日志与 B0 相同；公开 TYX 导入检查仍拒绝已有 mujoco_task_workspace.py
的 tyx.torch 导入，本批未修改该模块，不借机扩展整理范围。

该节点没有授予 P2PE 物理 C ABI、resident、Graph、外部 stream、传感器、安装或 GUI
资格。后续先接公开字段转换、candidate owner 与错误门禁，再完成实际 C ABI 数据闭环。

## B3f：端点 body 观测

公开合同与派生资源成本见[集成合同](../architecture/engine-integration.md#步后-body-观测-v1)。
只复用已有 TYX 运动学计算，不修改 Host、求解 CUBIN、排序或既有物理容差。
原生执行套件 20 项通过：原 13 项保持，新增自由落体 N=1/8/64、偏心 COM 速度、
多 branch/固定子 body、NaN/执行失败不发布与 discard 恢复、prepare/close 双失败
owner 保留。四子步只增加一次端点运动学；forward/reset 不重复 FK，命令不做 FK。
对应 P2PE 实际 DLL 资格单独记录于其 `docs/phymeo-cabi.md`，跨仓完整来源、提交和
哈希见实验根 `evidence/b3f-summary.json`。复用 env-b1j 源码入口，不声称安装升级。
全仓文档检查仍有历史缺失目标，日志与 B3b 的 SHA256 相同；本批不扩展无关修复。

## B3g：接触测量与受控发布

新增可选接触观测合同 v1，见[公共合同](../architecture/engine-integration.md#接触测量-v1)。
它复用锁定 physolve_tyx.contact_force，保持求解 Host/六 CUBIN、生产排序和现有容差。
设备端提取最后子步测量，命令不重新求解，masked reset 合并未选旧记录；错误归入
现有固定完成 gate。具体采样时刻按 world 保持，不与 body 积分端点混淆。
首轮专项 4 项通过，后续增加有效 EFC 行、空接触/stale、联合容量与多接触记录测试，
并由独立代理复审；逐项结果与配套实际 DLL 验证在实验根 `evidence/b3g-summary.json`。
继续复用 env-b1j 源码验证，不宣称 wheel 安装或 Graph/传感器资格；旧数值失败保留。

实际执行套件先有 26 项通过；审查收尾修正采样捕获/初始化 owner 后，接触专项
8 项全部通过（含新增 6D 斜面 wrench 转换/摩擦方向及 prepare/close 双失败），
其余 20 项未重复运行。全仓文档仍有 479 条历史缺失链接，日志散列与 B3f 完全相同。

## B3i：候选设备观测

公开 observation_bank 与只读 execution_error 合同支持 P2PE 在物理完成前排入
受门禁保护的设备 gather；语义见[候选设备观测](../architecture/engine-integration.md#候选设备观测执行-v1)。
只调整 PHYMEO 公开借用边界，求解 Host、六 CUBIN、TYX、生产排序和容差均不变。
实际 CUDA 定向测试 4 项通过：N=1/8/64 的 bank/权限/stale/延迟发布，以及错误
锁存、discard 清 gate 和恢复；日志为配套实验根 evidence/b3i-native.log。
继续复用 env-b1j 源码验证，不声明独立安装、Graph 或 GUI 已通过。

## B3j：设备选择更新

enqueue_update 可借用 bool[nworld] 设备 mask，执行一次 forward 后保留未选
世界的旧派生观测和接触记录。调用方仍负责正确选择写入；公共合同见
[选择世界更新](../architecture/engine-integration.md#选择世界更新-v1)。
原生 CUDA 9 项通过，包含 N=1/8/64 的选中 q 修改、未选 q/v/力/body/接触采样保持，
以及既有延迟发布、reset 和紧凑校验回归。接触按 world 身份比较，不把 GPU 行顺序
作为身份；不改生产排序或容差。首轮测试误用 step 参数已修正，原始日志保留。
Host、TYX、六 CUBIN 不变；没有新环境/SDK 副本，证据为配套 evidence/b3j-native-final.log。

## B3l：设备 reset

新增公开设备 mask 入口，保留到事务结束；不经 NumPy 或 host 转换。复用已有原生
reset/forward 和未选观测保持，没有修改 Host 或六 CUBIN。合同见
[设备 reset v1](../architecture/engine-integration.md#设备-reset-v1)。
原生 CUDA 9 项通过，覆盖 N=1/8/64 全选/空选/局部 reset、接触及力/body 保持，
并回归 selected update 和 host masked reset。热阶段禁止 host assign、numpy 和
tx.empty；记录为配套实验根 evidence/b3l-native.log。实际 C ABI 由 P2PE 配套验证，
不以本测试替代 Graph、安装、GUI 或完整发行资格。

## B3m：自由根写入前置合同

FREE 根的公共世界 origin pose / COM twist 通过 prepared native binding 转换为
PhySolve qpos/qvel。一次冷准备、一次候选更新与原生 forward；不改变 CUBIN。
具体字段、四元数检查、保留分量和成本以
[FREE body update v1](../architecture/engine-integration.md#free-body-update-v1) 为准。

正式入口仍在 test_physolve_execution.py：12 项 CUDA 选择集通过，其中新增
6 项覆盖 N=1/8/64、两个非零 COM 自由体及受约束 child、分量保持、step/reset、
非法输入原子拒绝/丢弃恢复、空绑定及准备/关闭失败保活；另六项回归设备 reset
和 selected update。增强选择测试三项通过，未选 body/未写分量含 NaN 不被消费；
补充一项负例通过，有限 twist 经 COM 转换溢出、真实 native storage/input alias
在修改候选前拒绝。CPU MuJoCo 矩阵和 mj_objectVelocity 仅作隔离测试参考。
当前 kernel 容差在首次比较前固定，未因候选失败改变。
最终 N=1/8/64 再验三项通过，execution_costs 的新增准备缓冲容量分别为
147/896/6888 字节（两个 FREE body）；此数不是整个 runtime 峰值显存或吞吐。

首次三失败均发生于测试末尾误用同步 step 的 steps 参数，改为既有
begin/enqueue_step(steps=4)/complete/accept；前面的坐标/速度检查已通过，原日志保留。
独立只读复审无剩余生产阻断。未执行完整库、Graph、P2PE body C ABI 或 GUI；
证据为配套实验根 evidence/b3m-{free-final,free-selected,free-negative}.log。

## B3o：逐 body 组件与输入错误分类

新增 free_body_component v1，保持原 free_body_update v1 调用可用；各 body 可用
不同组件，额外 selected worlds 与 body mask 合并，使混合字段只执行一次 forward。
PhySolveInputError 仅表示独立输入转换失败，未发布候选可丢弃；其他错误仍保留
执行失败分类。公开语义集中在[FREE body update](../architecture/engine-integration.md#free-body-update-v1)。
原生 CUDA free-body 9 项通过，含 N=1/8/64 逐 body 组件、非法 flags/原子恢复、
非零 COM、未选分量保持及既有负例。Host、TYX、六 CUBIN 与容差不变；没有新增
原生工作区容量，未扩充排序或微小数值诊断。配套证据为 evidence/b3o-native.log；
实际 DLL 由 P2PE 另行验证，不表示独立安装、Graph 或 GUI 资格。

## B4b：外部 stream 原生绑定

stream binding v1 的定义见[集成合同](../architecture/engine-integration.md#stream-binding-v1)。
实际无 Newton/Warp/MuJoCo-Warp 进程 7 项通过，含 N=1/8/64 控制/step/forward/
masked reset、旧/候选真实 CUDA 捕获拒绝、双 bank 第二次准备失败恢复、退休失败、
回滚失败与 close 重试。另三规模原生计数复测通过；每次实际换流固定两个 bank
rebind，同指针不增加。原生 Host 与六 CUBIN 不变，不追加物理排序或数值差异调查。
TYX 新增 validate_context，实际候选测试使用隔离仓源码，未改旧安装环境；
现有 direct 条件读回同步保持。证据：配套根 evidence/b4b-native-{final,counters}.log。
完整库、Graph 推进、安装和 GUI 不由本批继承资格。

## B5a：固定四子步 Graph

step_graph v1 定义见[固定 Graph 合同](../architecture/engine-integration.md#fixed-step-graph-v1)。
仅改 PHYMEO 运行组织，复用 TYX native stream capture 与既有 Host，六 CUBIN 不变。
主案例 4×10 ms；两 bank 原生捕获不推进，无预热 rollout 或全状态恢复副本。
正式测试 test_physolve_graph 验证简单控制/接触 N=1/8/64、body/contact 观测、
局部 reset、GC、换流、捕获失败、关闭重试、错误不发布和显存采样。

初轮综合运行 9 通过、2 失败：直接注入 raw NaN 超出借用输入的有限值前提，
触发 CUDA 719 并连带阻断后续采样；保留 evidence/b5a-native-final.log，不宣称
原生 CUBIN 已修复或支持此恢复。改用紧凑错误位故障注入验证发布门禁，并由
P2PE 正式输入测试验证 NaN 拒绝。两项后续均通过，见 b5a-native-gates-memory.log。
8 次交替流捕获/重放的 free memory 均为 7240417280 字节（共享 GPU 观测）；
此规模未观察到持续增长，不等同长期/机器人峰值显存或完整 Graph 资格。

## B6a：接触参考点前置合同

为附着体 COM 力矩提供与接触力同一求解时刻的公共借用位置；合同和成本集中在
[Contact reference v1](../architecture/engine-integration.md#contact-reference-v1)。
不混用步后姿态，不增加 forward，也不修改 TYX/codedev/Host/六 CUBIN。
正式入口为 test_physolve_execution 的 contact_reference 两项（含三规模参数化）：
偏心 COM 的 CPU 几何参照、direct/Graph 四步、命令保持、masked reset、非有限
派生输出拒绝与 discard 恢复。CPU 只执行 mj_kinematics，不作跨求解器轨迹验收。
新几何比较容差固定 1e-6 m，时间 1e-7 s，不改变已有测试容差。
最终执行结果与审查统一保存配套实验根 evidence/b6a-summary.json；SensorView
筛选、聚合、采样和安装 GUI 并未因本合同而完成。

## B6c：求解时刻 body 测量前置接口

公共字段、所有权、成本和可用范围集中于
[Solve-time body measurements v1](../architecture/engine-integration.md#solve-time-body-measurements-v1)。
配套 codedev a7 源码提供预分配 RNE 调用层，复用锁定算子；六 CUBIN 与 Host 不变。
不新增求解算法、不做跨求解器排序/微小数值差异调查。默认不开启测量时成本不变。

正式 pytest 入口 `tests/test_physolve_measurements.py` 最终 10 项通过（70.96 s），
覆盖 N=1/8/64 解析铰链与自由落体、direct/固定四步 Graph、求解时间、局部 reset、
command 保持、非有限观测拒绝和恢复、外部 stream/GC/换流、准备与清理失败保活。
保持现有容差；独立解析参照使用既定 1e-6，时刻 1e-7 s。保留 TYX AST 弃用告警。
初次最终入口遗漏 `phymeo/` 路径而未收集用例，修正入口后上述全套通过，原日志保留。

本批是配套源码验证，未覆盖旧环境中的已安装 a6；不在原环境安装替换。
独立复审关闭 current-device/冻结布局门禁发现；未运行新 GUI、完整 release 或性能。
接触支撑/非零外力矩、site frame 和 P2PE IMU/F-T SensorView 属于后续接入验收，
不能据本批 body 测量测试宣称完整 F/T 能力。证据见配套根 `evidence/b6c-*`。

## B6d：公开 site 投影与消费方 IMU/F-T

本批补充 [Selected body-site projection v1](../architecture/engine-integration.md#selected-body-site-projection-v1)，
通过公开 BodySite、编译绑定和预分配投影让消费者借用求解时刻测量。
不重建 Newton 模型，不插入 native sensor，不增加物理推进、forward 或 RNE。
codedev/TYX、六 CUBIN 与 Host 均不修改。

正式 `tests/test_physolve_sites.py` 最终 6 项通过（39.01 s）：N=1/8/64 的偏心
旋转 site、非零外加力矩、Graph 四子步、局部 reset、command-only 保持、
接触支撑比力和自由根 reaction、来源失配、准备/清理失败保活。
热投影禁止 tx.empty/Array.numpy 的检查通过。接触参照是同次物理测量的受力平衡，
未追加官方 MuJoCo-Warp 对照或轨迹排序调查。

新增测试首轮遗漏 site 参考点变换中的动态力矩项、误把 contact_world 的
`[capacity,1]` 当作一维；修正解析公式和索引后通过，无生产算法或容差修改。
更早的投影 device 字符串检查已修复，原始失败日志全部保留。
既有 TYX AST 弃用警告保留。消费者 P2PE 的实际 DLL/host/D2D、错误与生命周期
验证在其 `docs/phymeo-sensors.md` 记录，不能替代独立安装、GUI 或完整传感器资格。
来源、载荷、最终提交及审查见配套实验根 `evidence/b6d-summary.json`。

## B6e：预准备设备射线前置接口（部分验证）

公开几何、编译绑定、借用与成本见
[Prepared selected ray queries v1](../architecture/engine-integration.md#prepared-selected-ray-queries-v1)。
复用已有解析求交与 TYX Mesh/BVH；没有新物理推进、forward、逐帧几何上传或快照。
TYX 冷构造先注册嵌套资源 owner 再提交异步工作，异常后由 runtime 保留并重试关闭。

原射线路径 21 项通过（14.19 s），新增 test_physolve_rays 9 通过、1 失败
（271.44 s）；N=1/8/64、解析几何、FANUC mesh、固定 bank Graph、错误门禁和
嵌套失败保活通过。Panda 原始命中 shape ID 与 CPU mj_ray 不同，未确认原因，
保留原断言及失败，不授予其完整射线资格，也不归因为求解器缺陷。标准对象分配
调整后嵌套构造/关闭 2 项复测通过（18.86 s）。TYX 配套 71 项回归通过（16.92 s）。

独立复审明确 Graph 只捕获固定 bank；bank 发布切换后须由消费方选对应 Graph
或重新捕获。当前逐 ray 遍历形状，mesh 内部用 BVH，不声称场景级加速结构。
原生测试不是 Camera/LiDAR C ABI、安装或 GUI 验收。文档检查 479 个既有缺失目标
未变；公共导入检查的 mujoco_task_workspace 既有违规未变；既有 AST 弃用告警保留。
不改六 CUBIN、Host、现有容差或生产排序；不开展跨求解器微差调查。
证据、来源与提交统一记录配套实验根 `evidence/b6e-summary.json`。

## B6f：持久射线容量与消费方接入

prepared_ray_capacity_rows 明示 sample 可使用容量首维大于 selected rows 的输入；
仅计算选中前缀，输出仍是 compact row/ray 顺序。消费者可复用持久 buffer，不需
在热路径重建 view；max_rays 与 int32 容量门禁保持。N=1/8/64 的正式原生 endpoint、
step/reset 测试改用完整容量输入，3 项通过（26.60 s）。旧的精确 rows 输入仍兼容。

P2PE 用一个最大容量 query 为所有 Camera/LiDAR batch 顺序 sample→gather，避免
复制几何/BVH；PHYMEO 仍拥有 query 生命周期，P2PE 只拥有挂载和字段转换 scratch。
本批不改求解器、六 CUBIN、Host 或容差；Panda 原始 ray ID 对照失败仍保留。
配套 C ABI 结果及最终来源统一见实验根 `evidence/b6f-summary.json`，不替代安装资格。

## B6h：公共 visual 快照

新增独立冷显示快照，来源/尺寸/所有权见
[Cold visual scene v1](../architecture/engine-integration.md#cold-visual-scene-v1)。
原生编译与 visual 合同 29 项通过，含真实 Panda/FANUC 所有原生 ndarray 不变、
多形状、相对 mesh、源文件移除、旋转有限平面、UV 独立索引及纹理显式拒绝。
Panda 本夹具无 visual，明确发布 collision_geometry fallback；FANUC 保留真实 visual。
未变更求解器、六 CUBIN、Host、容差或生产排序。初轮测试路径及断言错误保留。
消费方实际 DLL 结果及各仓提交统一见配套实验根 `evidence/b6h-summary.json`；
这些源码验证不代表完整纹理、resident、安装或真实 GUI 资格。
