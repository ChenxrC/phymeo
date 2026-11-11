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
