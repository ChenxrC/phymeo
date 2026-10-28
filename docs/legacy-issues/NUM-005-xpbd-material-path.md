# NUM-005：XPBD材料路径与整步角动量的区别

状态：独立NumPy/actual interpret/CUDA复现；材料共同接触账本的前置问题，尚未解决。它不否定已验收的自由节点合同，但不能在机器人—布联合求解时绕过。

## 问题与根因

每条边的配置修正由当次方向和增量乘子决定。方向随迭代改变时，sum(delta_lambda * n)一般不等于lambda_final * n_final。旧诊断只给最终distance_multipliers，不足以独立重建实际材料冲量。现已增加opt-in真实逐边路径，尚未构成材料/接触认证。原position/velocity约束残差合格并不等于该替代式成立。

此外，现有半隐式预测、位置投影和速度重建不保证孤立材料的整步角动量严格守恒。x0=(+-1,0,0)、v0=(0,+-1,0)、等质量1、硬边rest2、零外力的一次步进，预测端点为(+-1,+-h,0)，投影长度恢复为2，重建速度给Lz1=2/sqrt(1+h²)。初值Lz0=2；h=.1/.05/.025时分别为1.9900743804/1.9975046778/1.9993752928。此例一次步长损失收敛，不能外推完整长程误差已合格。

## 复现与证据

运行`python scripts/probe_xpbd_material_balance.py --mode numpy --output <新JSON路径>`，mode可换interpret/cuda。源码aef9b257，118文件hash da5b54842e3256ef20b6eb5d27bfdcded77543f656330603f2ce7975d5989123；tyx仍为99f6df7c。

[NumPy证据](../benchmarks/p65-material-candidate-numpy.json)、[interpret证据](../benchmarks/p65-material-candidate-interpret.json)、[CUDA证据](../benchmarks/p65-material-candidate-cuda.json)。三模式各四例：三步长硬边和一次未固定共享双边材料。后者h=.1、compliance(.001,.002)，末梯度反推冲量的最大分量误差分别.0750138051/.0750109502/.0750109502kg·m/s（各节点质量1）；NumPy本构残差2.9e-9，tyx1.18e-5，均小于各自原容差。总线动量近舍入保持，Lz从-.71变为约-.53756。动能.52→3.188并非凭空增能：弹性能37.48165→.16239，总机械能下降。

这是主机/设备共同的离散数学与诊断能力问题，不是tyx实现性能缺陷；不写入tyx_flaw。未测Warp、完整轨迹、长期累积或新材料求解性能。

## 代码与后续方向

- `constraints/distance.py::project_distances`、`solvers/cloth.py::position_sweep/prepare_velocity/diagnostics`。
- `_tyx_node_kernels.py::project_node_distances`、`tyx_node_math.py::project_distances`、`tyx_participants.py::_NodeWorkspace`。
- `solvers/_particle_replay_ledger.py`目前正确拒绝材料域；不能直接删此限制。

先实现[实际材料路径记录设计](../superpowers/specs/2026-10-28-p65-material-path-design.md)：native写每条边前后状态、方向与delta_lambda，保留输入转换/舍入差；不能在host反算后冒充设备记录。随后独立验证实际预测、增量原式、速度重建、材料能量与外功，并明确角动量离散误差与步长门槛。若选新守恒积分器或速度修正，需显式新算法合同与机器人/接触时序，不能悄悄覆盖native结果。记录API已在0bf0619实现，NumPy与actual tyx记录每条边的操作前后/方向/dlambda；独立单步测试匹配增量式，原角动量损失仍存在。尚未实现生产共同账本、长时误差门禁或联合材料接触；不能关闭本问题。

相关原始研究：[XPBD](https://matthias-research.github.io/pages/publications/XPBD.pdf)、[Global Momentum Preservation](https://sealab.cs.utah.edu/Papers/Dahl-2019-GMP/Dahl-2019-GMP.pdf)。本地公式与代码独立推导；不宣称已采用论文修正算法。

## 2026-10-28 更新：账本已实现，物理问题保持开放

[MaterialPathLedger](../benchmarks/p65-material-ledger-acceptance.md)现独立认证native材料原式并核算实际路径冲量、K/U/端点位移功和线角动量，含FP32转换/固定host恢复。记录不足的工程缺口已推进；原XPBD角动量损失没有被修正，不能关闭NUM-005。下一outer仍须独立能量/离散角动量预算与步长细化，禁止仅凭equations-v1通过接受接触。没有修改积分器或用host候选覆盖设备。

材料接触前置实验进一步确认：压缩软边接触后K可高于free而K+U降低，纯K-free门槛不能沿用；新的含U/原外功与角动量预算已写入[材料接触设计](../superpowers/specs/2026-10-28-p65-material-contact-design.md)。这仍未修正积分器，也未交付新的生产接触程序。

## 材料接触阶段补充

0657c98引入显式每子步物理预算：旋转硬边在固定.001Nms下h=.1拒绝、h=.025接受；节点材料原式通过不再自动代表接触接受。该约束不修复原积分器的角动量损失，完整轨迹/网格收敛仍需后续验证。本条保持开放。见[阶段验收](../benchmarks/p65-material-contact-acceptance.md)。
