# P6.5 材料接触候选：能量与持续几何门槛

这是正式程序之前的有界实验，不是已交付接触API。原生产源码仍07b23d6/122文件hash80831f04b59587c1805b141c26a73a659f4b5b0a399f0f5f954f4c7371de683d。实验通过真实SceneLoadReplay、MaterialPathLedger和FrozenRobotLedger；没有host替代native状态，没有修改原容差。

## 纯动能门槛不适用于释放弹性能的材料

两节点质量各1，y=(0,.8)、rest1、compliance=.01，零初速/重力；底节点与质量2的Y移动机器人共点。法向p施节点，-p施机器人。A=2+c/h²，b=(1-d)/(hA)，u=(3/2-1/A)*p-b，所以闭合冲量p=b/(3/2-1/A)。h=.1时p=4/7，机器人/底节点vy=-2/7、上节点vy=6/7。

最终K=24/49≈.489796，高于无接触free的4/9≈.444444；但K+U由2降为6/7≈.857143。故不得照搬自由节点的K<=K_free条件。h=.05/.025的独立解析也通过。三模式各3例共9例，实际tyx每个候选25个completed launch；这是候选计数，不是整次outer的总launch，也不是性能基准。

源码脚本[probe_material_contact_candidate.py](../../scripts/probe_material_contact_candidate.py)，原始：[NumPy](../benchmarks/p65-material-contact-candidate-numpy.json)、[interpret](../benchmarks/p65-material-contact-candidate-interpret.json)、[CUDA](../benchmarks/p65-material-contact-candidate-cuda.json)。固定free inverse response只做GS预条件，每次重算材料，材料耦合并未被free方程替代。

## 单步速度通过不等于持续支撑通过

冻结T2、h=1/16与1/32，gY=-1、c=1/256、初始材料长255/256、rest1、robot质量2/effort2，解析静置接触冲量为2h。速度-only停止在NumPy两窗口分别接受1/5步后末穿透拒绝，interpret/CUDA两窗口均接受1步后拒绝。不能把这些结果写作2秒通过。

这个负例有意缺少联合几何停止判据，不代表计划采用的完整判据失败。相同参数/算法/原容差、仅把末gap加入收敛检查后，NumPy/interpret/CUDA的32/64步共六个2秒窗口全部完成。后续生产程序还须独立能源/角动量预算、摩擦、多点、失败事务；这批normal-only实验不能代替它们。

[持续脚本](../../scripts/probe_material_contact_persistence.py)支持`--stopping-rule velocity`（负例）或`geometry`（正例）。原始负例：[NumPy](../benchmarks/p65-material-persistence-numpy.json)、[interpret](../benchmarks/p65-material-persistence-interpret.json)、[CUDA](../benchmarks/p65-material-persistence-cuda.json)；联合几何正例：[NumPy](../benchmarks/p65-material-persistence-geometry-numpy.json)、[interpret](../benchmarks/p65-material-persistence-geometry-interpret.json)、[CUDA](../benchmarks/p65-material-persistence-geometry-cuda.json)。首次脚本运行在JSON输出时遇到np.int64序列化错误，转为原生int后重新运行；未把该次当成功证据。

这是Phymeo集成/离散合同问题，不是新tyx缺陷；未测Warp或性能。本实验不改已验收自由节点程序，不关闭NUM-005，也不支持自动新接触、CCD或真实夹爪。
