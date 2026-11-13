# BALL-S 标量限位、干摩擦与接触组合验收

状态：**SOURCE_AND_INSTALLED_VERIFIED / VALIDATED_ACCEPTED_SLOT_RELEASED**。
2026-11-13，本批把已有 HINGE/SLIDE 软限位及 native DOF 干摩擦接入
fixed BALL/R/P、单 FREE 根 BALL/R/P 的完整约束路径，可与 NONE/FILTEREXACT
JOINT 执行器及原 primitive condim1/pyramidal3 接触共同求解和发布。
没有另写求解器或改变数值核、源 iterations、容量、输入或容差。

[公共用法](p65-ball-scalar-constraints-usage-20261113.md)、
[冻结案例](p65-ball-scalar-constraints-cases-20261113.md)、
[实施合同](p65-ball-scalar-constraints-implementation-20261113.md)。
本批已结束；整个 P6.5 / Newton 对齐未结束。

## 实现及恢复缺口

三个编译入口先检查完整结构、原 native 地址和所有 dry 参数，再根据
scalar/dry/contact 行选择现有 T4；无接触但有关节约束的模型不再提前落到无约束分支。
公共能力报告区分 fixed/free、joint-only/mixed-contact、FILTEREXACT，保留
native 行来源、F/L/S 和保守 EFC 上界。静态上界只是诊断，实际活动行决定容量拒绝。
无新增行的旧模型保留原标签。真实 BALL 角限位仍拒绝。

首次 interpret 事务 run15 在 19 项通过后发现真实问题：外层 checksum 合法、
但 receipt 中 qfrc_constraint 被单独改动的检查点仍可 restore。
原格式是可信应用数据的一致性检查，不是认证机制；问题在于 receipt 与已接受
mirror 的发布通道没有完全绑定。修复 `c5d74303` 在替换设备分配之前逐字核对
qacc/qfrc_actuator/qfrc_constraint 的 FP64 dtype、完整 nv shape 和 bytes，
不会重新计算物理、修复状态或要求 held control 等于上一拍 control。
同一负控增加三个通道的篡改覆盖，保留原输入、拒绝后全批不变和实际后继要求。
原失败和完整返回保留，没有抹成首次全绿。

原源码交接 e19b1689/57fcdbe3、测试交接 f71e1056 仅是未执行作者版本。
父产品/测试整理 a6536615 后，run12–14 验过未变的准入/连续物理/原 XML；
随后只修复 checkpoint 和对应负控。run19、run16、兼容及安装使用修复版本。
最终已安装快照 **d7105a9ddde55ce33de8a1f003605073b5a129ea**。
不把早期结果倒灌到作者版本，也不为仅恢复逻辑变化重跑未变物理窗口。

## 实际验证

| 命令 | 结果 | 范围 |
| --- | --- | --- |
| 01–05 产品静态 | 最终通过 | 精确五产品文件；首 format 两文件排版，修复后仅复核受影响文件 |
| 06–11 测试静态 | 最终通过 | 精确十三测试/consumer 文件；首 Ruff 50 E701、59 E702、10 I001 和首 format 保留；仅导入顺序/排版 |
| 12 HOST | 22 passed / 1.89 s | 准入、明确拒绝、小型独立 QP oracle |
| 13 interpret 连续 | 12 passed / 77.45 s | 十二预登记窗口、两 world、完整同入步和独立连续原生参考 |
| 14 interpret 原 XML | 18 passed / 9.62 s | 精确旧 scalar/dry 拒绝身份的真实后继；不修改原 source helper/XML |
| 15 interpret 首事务 | 19 passed、1 failed / 45.70 s | 检查点发布力绑定缺口；原始失败保留 |
| 17/18 修复静态 | 通过 | checkpoint 及对应事务文件 |
| 19 interpret 事务 | 25 passed / 54.45 s | 恢复/clone/reset、末 world 失败、容量恰满/少一、真实 reduced-face budget 失败及原预算恢复 |
| 16 CUDA 完整新矩阵 | 55 passed / 48.71 s | 十二窗口、十八原 XML、二十五事务；HOST 不重复 |
| 20 受影响旧恢复 | 8 passed / 17.41 s | prismatic/fixed-tree/floating-tree/floating-mixed × 双后端，换 control、CPU 观察、恢复真实后继与 reset |
| 21/22/23 新包 | 全部 RAW_EXIT=0 | build/install、534 产品成员 source/wheel/installed 同字节，issues=0 |
| 24/25 包外消费者 | 两后端 PASSED、RAW_EXIT=0 | 每后端 400 窗口 worldsteps＋48 生命周期 worldsteps |

新增唯一源码矩阵为 **22 HOST＋55 interpret＋55 CUDA＝132**；旧恢复兼容另 8 项。
run15 已成功部分是故障前证据，不重复计入唯一节点总数。所有项目命令由父串行执行，
各次 fresh guard、原 stdout/stderr、RAW_EXIT、输入选择及 source snapshots 完整保留。
没有 skip、缩短预定窗口或调整阈值获取通过。

十二窗口覆盖限位上下侧真实释放、精确/重叠 margin、FREE+BALL 激活、dry 正负滑移/
内部静摩擦/零行，以及接触＋limit＋dry 同时求解、反转与卸载。检查完整 M/J/bias、
原生 EFC 身份/side/J/R/aref/上下界/力、完整质量方程和 KKT、双方 COM wrench、
解析 activation/transmission 与同次运动。CPU 参考独立连续推进，不用设备结果重置它。

包外两端各 **224 batch steps / 448 worldsteps**，每条日志 1181 完整事件，
包含十二窗口及三种 checkpoint/clone/partial-reset 后继；所有 owner 关闭成功，errors=[]。
两个 bootstrap 各确认 332 个 Phymeo 模块均从新 site 加载。私有故障注入、容量负控、
checksum 篡改和十八原 XML 是源码资格，不冒充包外再次执行。

## 产品与证据

- wheel：1,728,522 bytes，SHA256 `567a8e98bf93be01d38687b8e7ea3fbfc485bfc36c73181ef18582563c806c37`。
- 独立消费者 SHA256：`7ed5543f2b051e2d41068a1645291d745d9199dbb98c8d85b308a77290101602`。
- interpret 日志：153,468,667 logical bytes / 9,871,403 gzip bytes；CUDA：153,471,141 / 9,909,323。
- canonical：`artifacts/p65-ball-scalar-constraints-20261113-01`，全部 25 个命令已关闭。
  `installed-close.json` 绑定摘要、原始命令和 15 个包外文件的完整字节归档；
  `installed-closed-archive` 保留 wheel、脚本、checkpoints、summary、bootstrap 和完整压缩轨迹。
  原文件保留，没有重新落盘全部解压数据或重跑 producer。
- `release.json`：2026-11-13 11:50:45 +08，fresh matching project execution=0，父槽释放。

这不是 native FP64、G1 长舞、长期稳定支撑、BALL 角限位、新形状/摩擦维度、
BALL 刚柔耦合或异构扩容资格。fixed nv8/nq10、FREE nv35/nq45 及原容量保持。
旧 D4/D6-B 支撑失败不因本模块通过而关闭。下一独立模块是
[G2b 生命周期资源预算](p65-heterogeneous-resource-next-design.md)，之后才扩大异构规模；
当前没有新增性能优于 Newton 的结论。
