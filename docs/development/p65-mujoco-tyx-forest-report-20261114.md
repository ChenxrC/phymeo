# M1：机械臂与独立自由物体的共享 GPU 动力学

日期：2026-11-14。状态：**M1_SOURCE_VALIDATED**。
产品提交：`0c21731028c23c4ef5e22ef0192661efd0ce4a97`。
本批是[语言条件异构操作计划](../superpowers/plans/2026-11-14-language-conditioned-gpu-manipulation.md)的物理底座；
没有完成八任务场景、训练接口、语言模型下载或策略训练，也没有新安装包资格。

## 解决的问题

此前 floating lowering 要求唯一、排在 joint0 的 FREE；其他关节必须属于该根的后代。
因此固定机械臂与独立六自由度物体不能组成同一 MuJoCo-TYX world。
仅删除准入检查会继续把物体的状态误读为 q[0:7]/v[0:6]，并写错惯量祖先关系与积分结果。

现在每个原生关节统一保存 kind、qpos address、DOF address、body id：
FREE=3、BALL=2、HINGE=1、SLIDE=0。
各 FREE 仍须是 world-child body 的唯一关节；其他关节可属于独立固定树或自由树。
每个根分别计算位姿、世界平移速度、本地角速度及惯量；完整关节地址用于 quaternion 积分。
独立树的无约束质量矩阵交叉块为零，接触通过双方相对 Jacobian 在同一个约束求解中耦合。

## 复用与删除

- [静态 lowering](../../src/phymeo/backends/mujoco_tyx/_floating_model.py)：逐 joint 建立完整 DOF 参数、body ranges、ancestor spans；首个标量关节可有执行器，FREE 仍不能作为该 JOINT motor 的目标。
- [共享 kernels](../../src/phymeo/backends/mujoco_tyx/_floating_kernels.py)：每个 FREE 使用自身原生地址；合并 FREE/BALL quaternion 积分；删除固定首六维及从 joint1 开始的专用积分分支。
- floating、normal、constraint 三个 device 继续调用同一套 kinematics/mechanics/integration。
  质量矩阵、bias、passive、已有接触 law/求解器、allocation manifest 和原子发布机制直接复用。
- [runtime](../../src/phymeo/backends/_mujoco_tyx_runtime.py)的 BALL capability 按真实 joint type 判断；
  新增 articulation layout、原生 tree root/free root ids 和原有容量信息。
  多树模型不通过过时的 g1_supported 别名声称机器人应用资格。
- 未增加 forest 专用 solver、物体锁转、焊接、轨迹回填、物体策略外力、CPU 步进回退或另一个数组/allocator。
  物理语义留在 Phymeo，TYX 没有修改。
- 六个产品文件相对批次前净减少40行；新增代码主要是多案例测试。
  native_joint_layout=True 保留为已有 CSR/actuation/checkpoint 的内部 schema 元数据，不再选择两套 kernel。

## 实际验证

运行环境：Python3.13.3、MuJoCo3.3.7、NumPy2.4.6、pytest9.1.1、TYX0.2.0a1。
Phymeo 明确来自当前 src；TYX 明确来自之前已验证的安装目录
`C:/Users/38347/AppData/Local/Temp/tyx-main-integration-wheel-20261114-02/site313`。
实际路径和版本在 run01 及各 pytest stdout 中记录，没有改环境或安装依赖。
全部项目执行由父 /root 独占串行槽，每命令前保存 entrypoint-aware guard。

新增28个测试节点全部通过：

| 场景/合同 | interpret | CUDA |
| --- | ---: | ---: |
| RP、PRP、9标量关节+FREE、固定树+两个FREE；完整 native FK/J/M/bias 与12步连续运动 | 4 | 4 |
| 机器人/物体 XML 排序互换，按原生名称比较状态、力学与执行器力 | 2 | 2 |
| RP/PRP 与独立盒子的 condim1/3 偏心接触，双侧力/转动响应和12步连续参考 | 4 | 4 |
| 物理步内禁止 CPU MuJoCo、NumPy solver、Warp 回退 | 1 | 1 |
| 异构 sparse-world 独立对照、partial reset；最后桶实际 readback 失败后整体回滚/retry | — | 2 |
| 两独立 FREE 的 condim3 接触和双方角反作用，8步连续参考 | 1 | 1 |
| 固定首关节 limit+dry+FILTEREXACT，完整参考、checkpoint 重放、非首位 quaternion reset | 1 | 1 |

新增用例为[forest](../../tests/test_mujoco_tyx_forest.py)与[composition](../../tests/test_mujoco_tyx_forest_composition.py)。
两个不同 R/P 构型及未用于设计的9关节/双FREE组合均有真实执行。
物体保留完整6DOF、非零初始线/角速度、偏心COM和非球形惯量；只向机器人执行器提交控制。
CPU MuJoCo 是独立同入步及连续轨迹 oracle，其结果从未覆盖产品状态。

受影响兼容测试95个不同节点通过，合计123个不同节点：
原 floating 文件33项、FREE+BALL文件51项、BALL组合结构拒绝9项及原 multiple-free 的精确后继2项。
两个原 multiple-free XML 各自转为 interpret/CUDA 两个正例；其中一组2项计入 FREE+BALL 的51项，另一组2项单列。
没有声称全仓库套件通过。

执行记录（解释路径分段接续，没有为取得一个全绿摘要重跑已通过的同组）：

| Run | 实际结果 |
| --- | --- |
| 05 | 新forest解释：6通过，偏心力矩用例前提错误1失败 |
| 06 | 修正用例后的接触/禁止回退：5通过 |
| 07 | 新forest CUDA：13通过，21.76s |
| 08 | 原floating解释及后续队列：17通过，旧准入断言1失败 |
| 09 | 精确批次前/当前编译器准入对照通过，无device/step |
| 11 | 余下解释：13通过，旧FREE+BALL准入断言1失败 |
| 12 | 第二项精确批次前/当前准入对照通过，无device/step |
| 13 | 剩余解释、负控与新组合：36通过，19.61s |
| 14 | 原floating/FREE+BALL、精确后继和新组合CUDA：35通过，34.49s |
| 15/16 | 全部11个变更Python文件 lint / format-check 通过 |

05+06+07覆盖24个新forest节点；13/14另覆盖4个新composition节点。
08中两个已通过的准入负控在修正同函数后的11重复，合计时只计一次。
所有上述物理/功能运行使用同一份六文件产品字节；期间只有测试用例/旧断言修正。
各原始失败、原命令、前后SHA、原始stdout/stderr/exit与逐步状态保留在
`artifacts/mujoco-tyx-forest-20261114-01`及命令声明的独立basetemp。
run03的4文件排版首错、04/10格式化和一个Python执行前的PowerShell参数绑定错误也如实记录。

## 失败的根因与修正

1. 首个 sphere-box fixture 的法线穿过物体原点，不能要求 FREE 的广义角力矩非零。
   当时所有独立 native 比较已通过。仅将 tip 的 y 从 .04 改为 .065，保持在盒面范围内，
   得到明确的 .025 偏心臂；原力矩/响应门和所有数值容差保持。
2. 原 floating 与 FREE+BALL 两个 contact negative 仍用公共 runtime 检验过去的“无接触组件”边界。
   对 `2fcaaaaa` 的原 compiler 与当前 compiler 做隔离源码装载对照：
   两版公共路线均准入，裸 floating compiler 未给接触 permission 时均拒绝。
   两断言移回裸组件边界；没有关闭公共接触、放宽负控、修改物理预算或删除其余拒绝项。
3. 普通数值容差没有调整；没有通过改变质量、时间步、迭代预算或裁掉自由度取得通过。

## 适用范围与下一批

- 仍沿用原边界：最多35个广义速度、45个位置坐标、128个body；几何、contact/constraint容量和不支持项按原路径检查。
  此容量不是任意机器人、任意几何或任意约束组合的资格。
- 新验证覆盖至少一个FREE的固定/自由混合森林；独立的无FREE fixed-tree 编译器容量仍为原8DOF，不借本批扩充。
- 全部CUDA结果来自实际GPU物理步；当前异构facade依然按模型分桶，存在HOST控制/诊断往返。
  不声称异构融合kernel、无读回或性能超过Newton。
- 本批没有新 wheel 构建/安装消费者、remote CI、main merge或push；原安装版本不会因源码通过自动升级。
- M0只有本批最小原生场景fixture，不是8个抓/放/推应用已经准备完毕。
  下一批进入M2：公共异构设备观察/动作mask/选择性状态reset/checkpoint，
  同时完成8份任务场景和精确物理成功谓词。
  语言编码与共享PPO随后接入；它们不能由本批28个物理测试替代。
