# 外部接口层与多求解器集成

本页是 `feature/wh/phymeo-engine-integration` 的集成设计约束，不宣布新接口已经实现。
现有公共入口和支持范围见 [场景编译](unified-scene-compilation.md)，原生实验缺口见
[PhySolve 运行时](../development/physolve-runtime.md)。P2PE 的 ABI/schema/handle 不进入本仓库。

## 承接路线

首个公共声明到原生输入的实现切片见 [原生场景编译](native-scene-lowering.md)，
包含绑定版本、坐标、控制/接触律及未接入的运行期边界。

PHYMEO 使用已有物理描述、资产、装配和可组合算子。外部接口层把 authored 场景转成
这些公共表达；专用 solver 适配再转换为其输入。不能创建 Newton ModelBuilder/Model/State
兼容别名，不能为使旧消费代码可运行而维护另一套权威模型或状态。

现有 `SceneDeclaration` 面向独立 world 的单 articulation 和有限静态环境；这不是
任意多参与者场景已经支持的证明。扩展应落到已有装配职责，保持可解释的能力报告。
缺少的组合明确拒绝，不能拆成不相互作用的独立仿真伪装支持。

| 边界 | PHYMEO 的职责 | 保留的限制 |
| --- | --- | --- |
| 公共编译 | 自有表达、初态、控制、来源身份、能力和诊断 | 原生文件入口也须保留完整环境和身份 |
| 专用模型转换 | 物理参数、坐标、控制及结果转换 | 不要求所有 solver 使用 MjModel |
| 字段绑定 | 单位/frame/参考点/时刻、存储借用及转换 | 不把同步 host snapshot 当作设备接口 |
| 执行 | 准备、提交、完成、接受结果、失败状态 | 明确部分推进，不能发布错误帧 |
| 生命周期 | 真实物理实例及可写资源的 owner | close 失败保留资源，外部 stream 只借用 |
| 测量 | 接触、frame、物理传感器真值及质量 | 产品相机、噪声和训练编排不进入此边界 |

## PhySolve 转换

CPU MuJoCo 3.8.0 仅作为本实验 solver profile 的模型编译、初始化和参考依赖，
不成为 PHYMEO 核心必选依赖。运行候选不安装或导入 Newton、Warp、MuJoCo-Warp。
`physolve_tyx` 的算法来源与运行依赖分别记录。现有六 CUBIN 保持字节不变。

碰撞排除由公共身份表达，PHYMEO 适配转换到已有原生输入，codedev 负责专属候选表及
派生数据。先验证现有原生表达是否足够；不先增加平行过滤机制。不配置新增规则时行为
不变。来源映射的 native geom 下标只在本次编译内有效，不能作为公共持久身份。

不同接触律没有名称相同即等价的保证。resident 的材料/响应与 PhySolve 原生约束律
分别声明；不得同时施加两次响应。关节命令力、驱动实际输出、约束广义力与 wrench
分别绑定，不能因为某个数组存在就声明结果为 measured。

## 执行与演进

目前 PhySolveRuntime 是同步验证入口；包含 H2D 控制写入、D2H 校验/快照和候选 bank
复制，不能直接包装成高频 executor。需要延迟完成、受控发布及明确错误归并入口。
resident 公共入口的部分 world 发布行为同样必须补齐共同发布/恢复边界后才授予外部
事务契约资格。不能由 P2PE 每步整批 host 快照补偿底层合同缺失。

每个缓存/备份/bank 登记 owner、容量、用途、更新和失效条件。编译时冻结索引、缓冲
和绑定；热循环不重复 FK/forward/接触响应，不逐 world 重新编译或跨层调度。
Graph 捕获资源完整保活，同一 stream 包装复用；捕获失败不 fallback direct。

公共集成合同、包版本、源提交、TYX 互操作、原生 ABI 和载荷身份分别管理。
合同版本化实现尚待后续批次；不能从此文档推断任意版本组合已兼容。字段语义、采样
时刻和存储有效期变化要显式迁移；升级使用实际旧包和新候选跑同一测试套件。

## 分批验收

B1 先实现公共表达到原生输入的最小转换，并用 resident 小样证明其不依赖 PhySolve
格式；覆盖固定/自由根、偏心 COM、多 shape、初态/控制、来源和 world 映射。
B2 接通原生 step/forward/reset，关闭机器人历史失败，逐层记录分配/传输/同步成本。
随后与 P2PE 验证 host、D2D、固定 Graph、测量及独立安装，不将源码测试等同消费者资格。
resident 本轮限定受支持简单模型的 C ABI direct/host、动作、状态、接触及局部 reset；
其 HOST 窄相/诊断成本单列，完整 Graph/传感器/GUI 不是该 profile 的本轮要求。

同实现沿用原容差；新案例先冻结 SI/frame/初态/精度/步长及物理标准，再比较候选。
自由落体、支撑、COM/惯量、驱动与接触方向要有独立物理参照。Panda/FANUC 历史失败
保持未通过，不能通过放宽门限、改初态、关闭接触或改求解 CUBIN 关闭。

## 本分支 B0 实测

2026-11-11，新独立 CPython 3.13.13 环境从本分支源码构建 wheel 安装，MuJoCo 3.8.0。
禁止三包导入进程运行 `test_mujoco_compiler.py`：53 通过；原生 runtime 小模型选择集
10 通过、14 个机器人相关项未选。本批未重验机器人、不授予其资格。安装包来源和
实际 Host、六 CUBIN 哈希保存在配套实验根 sources/evidence，原始载荷未修改。
全仓文档检查仍失败：479 个缺失本地目标与旧检查点逐行相同，没有新增链接错误。
此次只增文档，未改变运行接口、数值或碰撞行为；完整 CUDA/C ABI/Graph/GUI 资格待后续。
