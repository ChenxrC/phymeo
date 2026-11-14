# M2-A 与 M0：公共异构 GPU IO 和操作场景基线

日期：2026-11-14。M2-A 产品提交 `2de488b945b3e1eaeb95aa91e6fdff4e9086b2ad`。
[实施计划](../superpowers/plans/2026-11-14-language-conditioned-gpu-manipulation.md)；
[公共用法](p65-heterogeneous-manipulation-io-usage-20261114.md)。

## 本批解决的问题

原异构 facade 的控制入口要求 HOST mapping，训练调用者没有统一的设备状态/掩码，也不能一次原子地提交异构设备动作。
现在由公共 workspace 导出 accepted native q/v/act/clock 和掩码，保留各模型原生地址。
Torch 获得独立 D2D 副本；新输入通过同一个已有 packet 审计和共同发布机制进入全部桶。
独立状态 reset 复用原 clone-before-publication/checkpoint 边界；工作区跨 owner 替换继续读取当前 accepted bank。

实现没有另写物理求解器、数值积分器或内存分配器。
新形状驱动 gather/scatter 只搬运有效列；模型常量不可变、外力省略时归零、全零宽字段为 false-mask sentinel。
资源关闭复用两级 DeviceResources；UNKNOWN_RELEASE 保留数组依赖与 runtime，不重试未知 native free。
工作区自有 allocation/读回/D2D 计数与原 solver-bank budget 分开，不声称完整 GPU 内存预算。

## M0 场景与目标

原作者 `2bd67e55/dc0d202a` 分别正常导入为 `25fc0ed8/1c74b870`，父排版提交 `3768bff6`。
[场景与参考判据](../../src/phymeo/training/manipulation_tasks.py)提供8份程序机器人 MJCF：
RRPR、4R、5R、6R、7R双指，以及2R、3R、PPPR推杆。
物体是独立完整FREE刚体；球、盒、胶囊、阶梯复合盒与三种T形块。
仅机器人关节有执行器，物体不接收策略外力；robot/self/object/support碰撞开启。
目标区域是可见非碰撞site；左右/中间目标、中文/英文及改写文本单独保存。

HOST成功判据绑定完整Scene物理元数据、goal/world/episode及连续tick/time。
新history须从tick0的支撑面初态开始；Scene构造复制并校验所有序列与标量。
抓取必须真实抬高、双指载荷、离开支撑和连续保持；
放置必须先抬起与搬运，再释放、真实支撑、稳定落在目标；
PushT使用实际多box平面并集IoU、朝向和支撑/速度门，不能用中心重合代替。
合成帧测试只验证这些判断逻辑，不构成真实抓放成功。

## 实际源码验证

全部项目执行由父 /root 独占串行槽，每条命令前 fresh entrypoint-aware guard。
现有 Python3.13.3、MuJoCo3.3.7、NumPy2.4.6、pytest9.1.1、TYX0.2.0a1。
Phymeo 来自父 src；TYX来自已验证的
`C:/Users/38347/AppData/Local/Temp/tyx-main-integration-wheel-20261114-02/site313`。
没有重装依赖、修改pin、执行新wheel或远程CI。

| 范围 | 实际通过 |
| --- | ---: |
| M2-A解释：W8四族/排列、W1逐字节、零宽、晚失败、partial reset/checkpoint、非法输入 | 18个不同节点 |
| M2-A CUDA：上述设备路径、真实Torch stream/独立所有权、UNKNOWN_RELEASE | 20 |
| 原 device_controls + heterogeneous_transactions 完整兼容 | 80 |
| M0：8模型native静态编译/初始零接触/interpret准入，声明与合成成功判据 | 24 |

M2新增节点通过后没有重跑同源已绿组：解释路径按失败点修复fixture并继续余下节点。
run06=8pass1fail，09=6pass1fail，10=4pass；11CUDA=20pass/34.51s；
12原兼容=80pass/227.01s；17M0=24pass/3.52s。
完整原命令、source-before/after、stdout/stderr、exit及保存文件位于
`artifacts/mujoco-tyx-heterogeneous-io-20261114-01` 与每条command声明的唯一basetemp。
这是聚焦源码资格，不是全仓库或新安装包资格。

## 首次失败与修复

- run02/03：新文件import顺序与排版首错；04/05修复。
- run06：测试把已发布FP32 quaternion直接作为严格FP64新初态；保持原单位四元数入口合同，测试改用原声明初始姿态。
- run07/08：checkpoint规范化映射顺序导致测试误判状态差异。比较器现按键比较，保留全部字段、dtype、shape与每个数组字节；不删除诊断字段。
- run09：bool world ID由原严格validator抛TypeError，测试此前只期待ValueError。修正异常类型断言，原状态不变门保留。
- run13/14与18/19：M0及后来修改的测试import/format首错；已修复。24/25最终8文件lint/format通过。
- run22：样式审计脚本把一个from-import拆成两行误判成绑定变化。v2逐alias保留type/module/level/name/asname，且非import AST完全比较；run23三文件通过，v1失败原件保留。
- 最初一个run02尝试被自动审批服务的额度限制挡在执行前；用户继续后授权重试成功，不记作产品失败或额外lint执行。
- 作者文本审查发现的dtype keyword-only、T0激活字段、closed storage、UNKNOWN_RELEASE等在正式功能验证前修复。最终独立只读复核无阻断项。

## 尚未完成的连接

M2-A只覆盖 native state transport。M2整体、M0可达性与T1三任务控制资格均未完成。

下一批M2-B应复用现有lowering/kinematics，以单独工作区从accepted端点q/v计算body世界位姿/速度。
不能把原步进scratch误标端点，也不能让观察改变physics bank。
接触token另保留input phase、input tick/time和实际geom/body归属，力/矩与该时相成对。
SuccessFrame使用同一入步位姿、速度和接触力，FREE角速度必须从local转换到world。
之后在现有VecEnv上接目标/episode/reset/checkpoint、final observation、terminated/truncated及masked action，
再接真实预训练语言编码器和共享PPO；不另建私有桶访问训练脚本。

8场景接触推进不是抓放/PushT控制成功；还须可达性、执行器裕量及独立控制器T1。
L1实际语义模型、L2一个共享策略/优化器、T2按任务和构型分列的学习成功率仍未执行。
本批不声明零HOST往返、融合异构kernel、跨world接触、大规模训练或Newton性能胜出。

## 八模型真实 CUDA 集成结果

作者单文件 `62899114` 正常导入为 `c66064c9`，最终排版/测试快照 `7fed0cc4`。
[八场景 GPU 集成测试](../../tests/test_manipulation_tasks_gpu.py) run29首次完整通过：
**1 passed / 60.85s，8个模型、8个world、每个32步，共256个batch环境步**；
另有8个独立CUDA W1对照各32步。不是512个batch环境步。

实际Torch CUDA输出经过公共mask/设备控制入口；有效执行器请求0.025，padding为NaN且未被使用，物体外力全零。
每步q/v/act/ctrl/time/tick与独立W1逐字节相同；全部workspace Torch tensor与显式诊断副本一致。
模型初态物体离支撑面1mm，8个物体均通过真实input-phase contact receipt证明向上支撑力和下落运动。
运行期间CPU MuJoCo dynamics与NumPy solve/inv等禁止调用。
这32ms窗口验证装配、动作路由和接触连接；没有抓起、搬运、放稳、PushT或学习策略成功声明。

完整33项history（初态加32步）包含batch receipts、控制、frame、workspace字节和W1状态。
原gzip 4,236,610 bytes保存于run29声明的basetemp，并按原压缩字节归档到
`artifacts/mujoco-tyx-heterogeneous-io-20261114-01/saved-payloads/29-eight-scene-cuda/`。
未解压重写、删字段或重跑仿真生成证据。

本批合计 **143个不同验证节点**：M2-A38、原兼容80、M0 24、八模型CUDA1。
最终单测试lint/format run30/31通过；原8文件24/25已通过，没有重复执行同源绿组。
文件审计 `final-source-evidence.json` 绑定9个源码/测试Git blob与工作字节、31条原始执行收据、
21份完整压缩payload（共5,027,207 bytes），copy/source mismatch=0，执行guard=0。
SHA256：`6154494ff8b7e212794738ba941254c36a28f891a4cc0b7dcbfe00d9d2598c80`。
首次file-audit因若干不使用tmp_path的用例未创建basetemp而停止；v2显式记录存在性并复用已复制同字节文件，
没有把未创建临时目录当作测试失败、补造payload或重新执行。原脚本保留。
