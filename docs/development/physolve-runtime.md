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
本模型 solver tolerance 原值为 `1e-8`。公开原生 context 的固定输入探针实际得到：

| 量 | 原生 PhySolve context | 官方 MuJoCo-Warp context |
| --- | --- | --- |
| 结束时代价 | 约 `1.3938153e10` | 约 `1.3938154e10` |
| `prev_cost-cost` | 0 或 -1024 | 0 或 -1024 |
| 归一化梯度 | 约 0.11109 | 约 4.80971 |
| 梯度是否达到 `1e-8` | 否 | 否 |

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
