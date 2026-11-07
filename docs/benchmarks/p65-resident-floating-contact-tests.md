# C10 普通浮根常驻接触：测试与独立参考作者交接

状态：**UNEXECUTED / AUTHORING_RELEASED**。本文件不宣称 C9 或 C10 验收通过。全部项目执行、静态检查和共享源码修改由父任务负责；本线只作文本、文件 IO 与小 Git 提交。依据父 `5013f70` 的 `2026-11-08-resident-floating-contacts.md`，不改旧输入、求解器、接触律或误差门。

## 文件与来源

本片恰四个 NEW：

- `tests/scale/test_resident_floating_contacts.py`
- `tests/scale/test_resident_floating_contact_transactions.py`
- `tests/reference_newton/robot/resident_floating_contact_reference.py`
- 本报告。

首节点 checkpoint 为 `03b57c0`；`35e98a0` 只把读数名称改为实际公共 `link_position_m`，未改输入或数值门。父已纯文本审阅 `3254174` 的 306 行独立参考及 `ec9560d` 的 F01–F09 增量，尚无本片执行结果。此后增量只有 F08 新的 `variant='both'` 独立实例、接受输出先保存后断言、完整 NEW 事务文件与本报告。原 `(9,None)`、`(16,None)`、`(16,'body15')` 三个实例完整保留。 主要代码 checkpoint `890b358`：参考仅追加3行root并载分支；功能测试增量20行；NEW事务测试629行。

独立参考仅 NumPy/stdlib，按相邻文件路径装载已接受的 `resident_floating_reference.py`：齐次变换一/二阶 time jets 给出每个真实 COM 的完整 J/M/b，原独立 box-face KKT 给出小 N 的有界 PD，一次原半隐式位置/四元数漂移。没有导入 Phymeo、TYX、Newton、产品 M/J/solve 或有限差分。实际延迟声明保存在每份 owned input 的 `delay` 字段；`first_independent_zero_delay` 明确只是零延迟首锚，每个实际物理步另按真实选中控制作独立参考。测试的产品构造器只声明模型，不能给参考制造期望。

## 首个有效 RED 与公共合同

固定首节点：

`tests/scale/test_resident_floating_contacts.py::test_f01_public_root_contact_keeps_full_six_coordinate_response`

先以旧 HOST 有限球–箱 patch 和原 law 验证 −1/32 m、2 N、完整 literal 6×6 质量矩阵与 a_z=1 m/s²，随后函数内部真实创建设备浮根、读取 FK，再调用 `batch.prepare_contacts(...)`。能力缺口应是当前浮根 contact binding 拒域，不是 module import 失败。本节点原输入/阈值不变。

全部新正域复用 `prepare_contacts/step_contacts`、`ResidentFloatingTreeBinding`、原 `ResidentContactReceipt`；同树双 dynamic 显式 `tree_dynamic_pairs=True`。B/N/D 分别为真实刚体数/电机数/full 速度数：n0 为 B1/D6，n16 为 B17/D22，body16 非 padding。完整原 law wrench(B,6)、实际上传 FP32 wrench、input_external/contact_generalized(D)、selected_drive(N,3)、actual_drive(N)、同次 nested FloatingTreeDiagnostic 均逐项核对。

## 冻结案例

统一 h=1/1024 s；same-entry atol=5e−6、rtol=5e−5；40 步状态 atol=2e−4；完整方程/质量 2e−5、原 PD residual 5e−5。1e−10 仅原已声明表面/作用点的记录重建，1e−12 只用于可独立精算的 HOST 首锚和同一 FP64 账，不拿它验设备动量。

| 节点 | 固定输入与独立首锚 | 窗口/留出 |
|---|---|---|
| F01 | root m2、I=.5、球 r1/8、z3/32，对有限箱；Fz2、az1 | 公共首步；完整 root 六维 |
| F02 | root2+Px child1；child +2N / 再内部 motor3 / root +2N | 三个各40步；(a_root,qdd)=(0,2)/(-1.5,6.5)/(1,-1) |
| F03 | root2、两个 Px sibling m2/1、间距7/32；两球 r1/8 | 各40步；(0,-1,+2)，反序、轴循环+平移、共同 boost；净线动量/总质量按原速度门 |
| F04 | Rz child 上 y1/4 偏心球；root ωz1/16、joint v−1/32、motor1、d4 | 40步真实弧；首 N65/32，ax65/96、α_root−2、qdd127/32 |
| F05 | root COM x1/8 的2N；另双偏心球组成1N·m力偶，root principal frame旋45° | 三个40步；ay1/αz−.5，或α=(3,1,0)，另整体轴循环 |
| F06 | 原 `floating_fixed_composite.urdf` 真 fixed折叠工具 | 40步；M3、COM x1/12、I=(3/10,53/120,13/24)；首 Fy3、τz.5、ay1/αz12/13；原 cube support 随真实姿态切换 |
| F07 | C8 原 H/H、capsule侧/端/miss、CC、非盒五边柱HH | 每项2步；独立首 gap/normal/N、权重和面积质心矩；后续响应只以实际 uploaded wrench 为已知输入，**不是独立整窗几何** |
| F08 | 原 P05 n9/n16 非轴分支、偏心 COM、非零初速/外载/主轴；另原 C8 D02 的 n3 R/P 分支 | 每项40步；body15/body16单载荷保留；新增 root0+body16 同时各2N，完整B17/D22；n3共享祖先接触广义力=(0六维,0,-2,0) |
| F09 | F01改 z=r−1/4096、vz1/8、g−8 | 40步自然载荷→分离→重入；首 N1/64、az−1023/128；无复位或时刻换力 |
| F10 | n0/1/9/16重复单元，world keys非连续；drive delay1 | 四份 W1、W4、W32，各40步+tick19后21步完整恢复重放；W32反注册序 |

F10 另有旧普通 fixed/affine 接触 + scalar + 未绑定 floating 的40+21步共同提交，用既有独立 oracle 核原世界；浮根仍走原 standalone 动力学，不假设 `.physical`。小 N PD 保留原 C9 P06 声明：kp=(16,8,4)、kd=(2,1,.5)、target=1/8、rate=−1/16、ff=(1,−1,.5)，cap=1/8或8、delay0/1；外 effort=(1,−1,1)/32、末 COM 力=(1/8,−1/16,1/32)、root矩=(−1/32,1/64,0)，各4步。大 N 保持零 PD。

## 事务与边界

局部容量分别在两个 bank、root-only 与 n16 末 body 上取0：该世界完整 root/joint/applied/FK/twist/energy/tick/history不动，邻 world 正常接受，再用原容量重试。实际上传、candidate读回和 receipt构造后抛异常，整次不发布；恢复同 tick 的 epoch 变化使旧 attempt 失效。第二次真实 scratch allocation 后抛错、全局 candidate cap0、跨 batch plan、close两次、关闭后使用、完整 rebuild+restore 均保留真实原状态/资源。

所有 owner/source/world/topology、缺/额外材料、mu>0、错误 law、CCD/sampling、非 builtin-bool flag、未过滤直接父子均为明确负控；mask/exclusion/same-body 先过滤，不借登记对绕过实际父图。非法 body17、B16 的 body16、B1 的 body1拒绝，旧 fixed/affine不因此多出 body16。from_asset 改 revision 拒绝，正常 F06 source与material IDs完整保留。

额外 representability 负控固定 root anchor=(2**60,0,0)、local material=(1/8,0,0)，对有限静态球先取得真实 broad candidate，再要求 contact packet 因不可逆的 world展开明确 ValueError、无接受态发布。它不是local中心0。没有QR、吸附、改点或放宽 `_CONSTRUCTION_M`。

材料与作用点各按真实入步 stored R/p 的逆解、真实末态 R+/p+ 运输；两侧 F·Δx、共同点 COM 力矩与入口功率分别重算，不能把两种功抹成同义量，也不要求 penalty/半隐式模型虚构能量守恒。已接受完整记录在独立断言之前入列表，异常保存已接受前缀与 retained 状态。

## 预声明保存集合与运行入口

按作者文本计数为功能25实例、事务51实例，每backend76；**尚未 collection，最终以父实际collection核对**。预期每backend61个gzip：功能25、事务36。所有目录取 pytest 对节点名的前30字符前缀+ordinal；参数顺序就是各 `parametrize` literal 顺序，不从运行结果选样。固定文件名如下：

| source节点族 | 子目录/文件 | 每backend数量 |
|---|---|---:|
| F01 public | `f01-public-floating-contact.json.gz` | 1 |
| F02/F03/F04/F05/F06/F08/F09 | `floating-contact-window.json.gz` | 18 |
| F07 六形状 | `floating-shape-response.json.gz` | 6 |
| F10 W1/W4/W32 | `w1-503/`, `w1-91/`, `w1-1207/`, `w1-6/`, `w4/`, `w32/`各 `floating-contact-batch.json.gz` | 6 |
| F10 容量 | `floating-contact-capacity.json.gz` | 4 |
| F10 PD/delay | `floating-contact-window.json.gz` | 4 |
| F10 三 latefault×两bank | `floating-contact-late-fault.json.gz` | 6 |
| F10 same-tick restore | `floating-contact-stale.json.gz` | 2 |
| F10 scratch/globalcap | `floating-contact-resource.json.gz` | 1 |
| F10 ownership/close两模型 | `floating-contact-close.json.gz` | 2 |
| F10 旧类型混合 | `floating-contact-old-mixed.json.gz` | 1 |
| F10 过滤域六项 | `floating-contact-domain.json.gz` | 6 |
| F10 真实candidate的shape/group/world错配 | `floating-contact-candidate-source.json.gz` | 3 |
| F10 大anchor | `floating-contact-source-representability.json.gz` | 1 |

配置负控（11种输入、4种flag、3种body、2种旧类型边界）不生成虚构轨迹，父 raw pytest/source receipt 是其执行证据。完整物理gzip包括声明模型/shape/bindings/law/控制、所有实际 receipt/state/geometry、独立矩阵与两种功；W1/W4/W32前四个同ID世界逐字段相同，额外W32世界各自独立审计，tick19后21步比较全部state/geometry/receipt/summary，不删除世界身份或诊断列。无性能排名或吞吐承诺。

父唯一执行槽后可用原3.12环境，PYTHONPATH指父当前src、fresh cache/basetemp：

```text
python -m pytest -q tests/scale/test_resident_floating_contacts.py::test_f01_public_root_contact_keeps_full_six_coordinate_response --tyx-mode=interpret --basetemp=<fresh-absolute>
python -m pytest -q tests/scale/test_resident_floating_contacts.py tests/scale/test_resident_floating_contact_transactions.py --tyx-mode=interpret --basetemp=<fresh-absolute>
python -m pytest -q tests/scale/test_resident_floating_contacts.py tests/scale/test_resident_floating_contact_transactions.py --tyx-mode=cuda --basetemp=<different-fresh-absolute>
```

如首节点此前已在同源通过，完整组按该唯一精确identity排除；不重做 C6–C9 整套。独立 reference 不是 Newton native 对照，wheel/native后续授权。本线未执行任何上述命令、import/数值/static/format/build，最终源码须经父 review、实际 interpret/CUDA 与同次保存审核。能力域只 ordinary floating、有限 HOST mu0离散入步窄相+真实设备 full-tree响应；affine浮根、跨独立参与方、持续摩擦、CCD、全设备窄相仍未支持。

## 父执行观察：F07 保存精度口径（2026-11-08）

506解释完整剩余矩阵已通过16项，随后CC在已接受首步的gap断言失败：原0.09 m child offset保存为FP32，与十进制-0.01 m的理想gap相差3.57627868e-9 m，原断言门1e-10 m。原输入/产品/力与运动工程门不变。父仅对CC/HH增加两套独立支持函数锚：原FP64声明仍按1e-10核literal；实际入口由原offset显式FP32转换和原local/radius计算expected gap，同样1e-10核保存witness。完整first_geometry_anchor记录两口径，非从设备witness反求expected。510原失败CC与尚未运行HH各通过，RAW0；506已绿16项不重跑，后续继续F08/F09/F10，最终CUDA仍为首完整剩余矩阵。


## 首次延迟历史参照修正（父，2026-11-08）

原511解释组的F08五项与F09均通过，随后F10在W1的N9首步因测试把延迟历史统一预期为零而失败（RAW1保留）。原C9从创建起就把首个延迟target设为已存FP32初始关节角，非零初态应为±1/16；rate/ff为零。父只修独立测试_selected、delay显式断言和同次保存reader，以原声明q的FP32表示构造期望，不从实际selected_drive反推。原产品、模型、控制、工程门和完整保存均未改。517 Ruff与518 format-check均0；全部25功能节点已有分组解释覆盖，接下来仅执行尚未完整通过的51个事务节点，原W矩阵未完成前缀保留，不重跑F01–F09。
