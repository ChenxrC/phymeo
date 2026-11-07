# C8 同一完整驻留树内双动态接触：作者交接

状态：**TEST_AUTHORING / UNEXECUTED**。本作者只新增两个测试文件和本报告；没有执行 import、collection、数值、pytest、静态、构建或 native 程序，没有合并父源码。基线为原作者树 `5dd903f`；权威输入与能力域是父 `5d85e42` 的 [C8 计划](../superpowers/plans/2026-11-08-resident-tree-dynamic-contact-feedback.md)。所有 OLD 产品、执行、计量及最终接受均由父负责。

## 真实首节点与后续范围

首 checkpoint `6f88ffa` 的父394遇到测试数组 dtype 前提错误：`np.full` 的整数增益不符合原 DrivenRobotProgram 输入合同，尚未碰能力门。`e1c49d` 仅将 kp/kd/cap 数组声明为 float，数值不变。父395随后有效 RED：完整双P模型、真实设备 FK、HOST sphere patch 的 gap −1/32、归一权重与双方 ±2 N 全成立，准确停于 `tree_dynamic_pairs` 缺口。父反馈产品 `13c3fad` 下首节点401 interpret 1 passed / 1.28 s、402 CUDA 1 passed / 3.11 s。原394日志保留，不能称有效功能RED；这些是父执行结果，不是作者运行。

本文其余新增矩阵尚未执行，不预写通过、速度或完整C8接受。父已验首节点应按固定身份从后续同源完整组排除，不重复C6/C7旧组。

## 文件与调用合同

- `tests/scale/test_resident_tree_dynamic_contacts.py`：公开 D01 首节点、独立全模型参考、有限表面/动力学与真实材料记录。
- `tests/scale/test_resident_tree_dynamic_contact_transactions.py`：D08 世界与双bank事务、原过滤/相邻边界和不可变来源。
- `docs/benchmarks/p65-resident-tree-dynamic-contacts.md`：本交接。

测试调用 `prepare_collision(..., tree_dynamic_pairs=True)` / `prepare_contacts(..., tree_dynamic_pairs=True)` 与原 `step_contacts(plan,h)`；没有新求解器、替代运行时或测试内写入物理状态。默认/显式 False 保原双动态拒绝；True 必须 builtin bool。same-body、mask、excluded 先于直接父子门；过滤后pair不允许law。`registered_pairs` 仍是原物理域登记，候选/材料覆盖才是空间过滤后的集合，测试不会将两者混同。

所有几何binding属于同一world的同一实际 full ordinary/affine模型；多个真实root可以属于一个森林。主静态law为 k=64 N/m、d=0、mu=0，h=1/1024 s。并未支持同world多独立参与者、浮根驻留、切向摩擦、CCD或全设备窄相。

## 独立案例及判断

| 族 | 声明与独立期望 | 新覆盖 |
| --- | --- | --- |
| D01 | 两P根质量2/1、轴+x、球r=1/8、中心距7/32；g=(−2,+2)、M=diag(2,1)、a=(−1,2) | 公开入口、两侧一次加载、40步与自由方向净动量 |
| D02 | Rz共享祖先和Px/Py孩子，质量1/2/1；原 M 的第一行 `(753/1024,−1/2,7/32)`，g=(0,−2,0)，a=(−8/9,−11/9,7/36) | 全cross M/b/J、40步、Rz(.37)及轴循环/平移、反序登记 |
| D03 | 双Rz森林、球心局部y=1/4、Izz=1/4与1/2 | COM矩(+1/2,−1/2)、a=(2,−1)、40步真实弧运输与功 |
| D04 | q=(s,−s/2+1/8)，第二安装3/32 | reduced a=−4/3、full a=(−4/3,2/3)、原行lambda=−4/3；40步 |
| D04Z | q=(s,s+1/8)，静止与s_dot=1/8各40步 | full正载±2、K接触0、lambda=−2与约束功0，不以状态不动冒无接触 |
| D05/flatten | P=(1,2,−1)，m=(2,1,1)、原偏置与原两行保留 | a_s=−4/7；chain lambda=(−3/7,−10/7)，flatten=(−8/7,−10/7)；各40步、另两步同运动/异图恢复拒绝 |
| D06 | 两有限盒、sphere–capsule轴部/有限端/近失、原CC线、10顶点非盒HH面 | 正域总2 N，CC/HH首16/25 N，miss0；各两步；归一权重只乘一次、双方材料ID/skin/作用点 |
| D07 | 共同/相向/分离速度1/8且d=4；两对z=±1/8球 | 首总力2/3/1 N；双pair总4 N/COM矩相消；默认/类型/过滤/来源/相邻负控 |
| D08 | keys503/91/1207/6，D01、D04、原静态受载P、未绑定free sphere | W1与逆序W4各40步；tick19恢复后21步，完整state/geometry/receipt逐字段相同，不要求stats相同 |

主输入/惯量/阈值均按冻结计划。短PD留出显式使用 kp16、kd2、cap1/4、target1/16、feedforward1，D01和D04各delay0/1两步；独立标量隐式PD公式处理各自 reduced M，cap只夹motor，真实接触不被cap裁成1/4。共同模型输入与原持有/延迟队列完整保存。

独立力学复用已经接受的 `tests/reference_newton/robot/resident_affine_reference.py` 的齐次变换导数/逐部件惯性装配和full-coordinate KKT（没有Phymeo导入）；新测试只给新的固定声明与几何闭式间隙，禁止从产品 M/J/solve 输出制造预期。ordinary直接完整RobotModel；affine由该参考的URDF文本声明和公开loader创建，随后按真实资产名称重排；explicit binding保完整checkpoint topology。源树无需新增fixture或复制参考。原child未包含父C5参考，依父已tracked该路径运行；不为此合并父或放宽来源。

每步一方面从实际entry坐标独立算力/M/b/J/约束和预期末态，另一方面独立推进整40步参考以核累计误差。双方净世界wrench为零；固定基座可能有约束反力，不要求一般固定树总线动量守恒。作用点功、材料点功、入步功率乘h与KE变化分列，不设未经授权的守恒等式。现有C7 `_audit_sites` 仅复用完整材料记录重建，产品J另与独立参考核对。

工程门维持原 atol5e−6/rtol5e−5、40步atol2e−4、M2e−5、PD5e−5；1e−10只验证stored p/R及原材料声明下的记录构造，不称动力学无限精确。

## 事务与保存

- 两active bank分别限制ordinary/affine的四点box patch到容量1：整world两动态group及N/K/FK/energy/held/history/tick保持，邻world继续；足帽原请求随后推进。
- 同一plan从分离到精确接触，零法向力几何site仍计帽；容量失败保上次owned receipt。另同world动态2 N与静态1 N并存：总g=(−1,2)、静态反力(−1,0,0)，cap1拒绝、cap2仅推进一次。
- 实际第二scratch分配后失败、global candidate cap0、实际上传/candidate读取/receipt构造完成后异常（两bank），全batch无发布；新plan原请求与未故障独立batch的全state/geometry/receipt完全同值。
- 同tick restore确切使epoch变化并拒绝旧attempt；跨owner、错误full图、关闭重建和所有双侧owned arrays不可写。不能用伪造接受标志代替真实执行。
- 每条物理window保存完整gzip：原声明、control/model/binding/law、逐步实际receipt/full diagnostics/双方材料及力/功、独立参考、retained状态和实际计数。长window在finally保存错误与最后接受态；短配置负控只有测试收据，不伪装物理轨迹。

作者手工展开为功能28和事务46实例/后端；这不是collection结果，最终以父实际收集与执行身份为准。源码测试不启动benchmark，不报性能排名。本片未运行任何Python或native命令。

## 父执行交接

父按既有guard和项目解释器在集成源绑定下运行两个NEW测试文件，分别interpret/CUDA，排除已过同一D01首节点；父决定受影响的旧相邻组、独立Newton、完整保存审计及wheel。作者只交上述NEW3，所有产品路径保持父所有权。

最终状态：**AUTHORING_RELEASED / 后续矩阵UNEXECUTED**。首RED及两backend首正控的真实范围如第一节；其余没有验收结论。
