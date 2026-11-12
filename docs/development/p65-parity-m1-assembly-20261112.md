# BALL-M1、设备边界与多参与者装配：本批实际结果

2026-11-12 13:28。用户授权“继续下一批”，本批没有新截止。原三名作者在隔离 worktree 并行编码，父负责共享接缝、独立文本审阅和唯一串行执行。[执行计划](../superpowers/plans/2026-11-12-parity-m1-assembly.md)的三条功能线现为 **SOURCE_AND_INSTALLED_QUALIFIED**。这不是整个 P6.5 或 Newton 全功能完成。

## 已交付的通用模块

| 模块 | 公共能力与复用路径 | 本批源码资格 | 源码外解释 / CUDA，各后端 |
| --- | --- | --- | --- |
| BALL-M1 | native JOINT CSR moment 表把 scalar 与三自由度 BALL 电机共同映射到完整广义力；同次设备力、输入与运动发布，支持 checkpoint/恢复 | 47 个新节点；9 个受影响兼容节点 | 27 次主运行 + 1 次 clone，56 worldsteps；原三 kernel/步 |
| Participant scene | 显式资产、参与者、初态、控制、body/collider 身份及完整材料对表，装配原四类 resident 树，复用共同 world 发布与恢复 | HOST19、interpret19、CUDA19；57 次执行 | 12 次物理 batch step、24 worldsteps；attempted/returned/checked 均12 |
| M1 resident boundary | 已接受的真实设备 FK → gather → bounds → BVH → pair pack → 原有限 segment–triangle 核 → typed 完整结果 | 4 HOST + 67 interpret + 68 CUDA 独立节点；20 个旧接口兼容节点 | 3 次物理 batch step、6 worldsteps、3 次完整查询/12 rows、2 次旧结果拒绝 |

计数仅描述证据范围，不是整体完成率。源码失败后的成功前缀和受影响节点按实际版本组合，不冒称重复跑了一遍全绿大套。

## 实现中发现并关闭的问题

- M1 的 PENDING 清理不能被下一次 query 覆盖原 ledger；现保持关闭专用状态和强所有权，UNKNOWN 不重试不透明释放。
- 浮根大平移必须保留原 FP64 anchor；静态与动态几何使用同一公共原点，不能先把原点压到 FP32。
- HOST 注册和 BVH 缓存数组按对象身份去重计数；设备容量公式与原上限保持。
- 场景构造与清理双失败保留原错误、清理错误及清理状态未决、显式强保留的 owner；它用于诊断，不能盲目重试 UNKNOWN。安装消费者在断言之前保存原调用和完整返回；M1 隔离后的诊断读取失败也不能盖掉原查询异常。

这些是通用所有权、坐标和发布问题；没有通过调整质量、测试输入、几何门或预算解决。

## 真实执行与首失败

证据根为 `artifacts/p65-parity-m1-assembly-20261112-01`。完整命令、环境、fresh entrypoint-aware guard、源前后哈希、stdout/stderr、RAW_EXIT 与原失败均保留。

- BALL：010 的39个主节点先通过，随后 NEW 事务保存器不能 JSON 编码原 checkpoint bytes；修复保存编码，013 的8个事务通过。014 的9个兼容节点通过。原物理与完整 checkpoint 不变，见[详细报告](p65-ball-joint-motor-report.md)。
- 场景：023/024/025 分别19通过，无物理修复；显式材料、四家族、两 world、恢复、reset、晚失败与身份覆盖见[场景报告](p65-participant-scene-report.md)。
- M1：032 HOST4通过；040 interpret23、043 interpret41成功前缀，046 四个受影响/未访问节点及048两anchor节点通过；034 CUDA27成功前缀，049余下41通过。独立计数为139节点，不相加重复检查。047旧 COL2/COL3 精确受影响20节点通过。首错依次是 NEW oracle 的 tuple 行索引、解释模式传输计数口径、故障注入时实际已分配8字节的峰值断言、大平移下仿射重建 oracle。后者改为 `tri[0] + bary @ (tri-tri[0])`，保留原8u权重和2e-4L见证门；不归一化权重、不读产品 frame。完整原因及独立审阅见[M1报告](p65-resident-boundary-report.md)。
- 首静态失败和修复都留存。034第一次 wrapper 查找尚未创建的 job JSON 在项目执行前失败，不能算一次 CUDA 测试。
- 安装053首错：消费者要求 FP32 transmission force 与原 FP64 step force 的编码完全一致。`563e4798` 只改消费者：显式断言两者各自 dtype，把 FP32 精确提升后做逐位比较。没有向下舍入、物理容差变化或产品修改；059/060静态检查通过。

## 一次统一构建与源码外验证

产品快照 `2f5f19d6781950581eae2e741f329409f59886bb`；最终消费者快照 `563e479840611f0482f7922d563da30866426760` 只改上述检查。050构建、051安装、052完整成员比较均 RAW_EXIT0。515个产品文件在 source/wheel/installed 逐字节相同，零差异。wheel 为1,646,653字节，SHA256 `364a1067c19b7c779c9684ec8d100f9d075c2b89224f112fa08fb8c3993535b3`。之后没有产品改动或第二次构建。

061/062 BALL、063/064场景、065/066 M1 的 interpret/CUDA 六个消费者全部 RAW_EXIT0、PASS。consumer/cwd/output 都在源码目录之外；bootstrap分别检查312/304/301个已加载 Phymeo 模块全部来自新安装目录，无 source tests/examples 导入。CPU MuJoCo 只作 BALL 的独立对照，不代替 TYX 物理步；场景消费者记录 CPU integrator calls=0。

六条完整 gzip 轨迹分别保存64/64、42/42、17/17个事件；不删字段或规范化。首次安装失败另存8个事件。新 wheel、原/修复消费者、bootstrap、完整结果和原失败共72个文件已逐字节复制到本批 `installed-wheel-01`，源与目标 SHA/size 全同，没有全量解压。`installed-close.json` SHA256为 `69fc6c1de3427b2e53c4c7ea697225073e4ac80daa6b2a7f55110c94cdc631ff`。这次只做新证据归档，没有重审旧证据。

## 仍需补齐，不能由本批推导

1. **机器人组合**：直接 BALL motor 当前限固定树、JOINT/NONE/FIXED/BIAS_NONE、na=0、nv8/nq10/nu32；BALL activation/contact motor、FREE+BALL、limits/dry/更广传动仍独立准入。旧 BALL-A/C 标量路径与 ROB2b 复用原验收。
2. **场景装配**：复用 resident 的 ordinary/affine/floating/floating-affine 公共接口；不导入 MuJoCo-TYX BALL 到 resident 标量树，不是跨树隐式 KKT 或通用软体装配，R03仍开放。
3. **碰撞与规模**：M1只接显式完整物理拓扑的刚性 accepted FK。仍有 HOST 读回、校验和整数候选控制，不是零HOST全GPU管线；from_asset mesh lowering、M2真实变形顶点 bank、实体内部、带外全局最近面、CCD、patch/接触力和摩擦历史尚未接通。
4. **其他方向**：J/R学习、G1原XY及完整动作、B-point表示失败没有被本批关闭。无新训练、Newton性能比较、远程CI、环境清理、push/mainmerge或发布。

下一片优先扩展直接球电机的接触组合与设备来源的资产装配接缝；M2先冻结真实 producer bank/提交所有权，再复用 M1 查询。继续按多案例→缺失通用模块→受影响验证→安装消费者推进，不能仅增加单个形状对或重复旧已绿组。当前全景与入口见[差异全景](phymeo-newton-panorama-2026-11-12.md)、[续作入口](p65-continuation.md)。
