# 驻留树的速度摩擦连接：实际结果

**已完成本机限域验证，2026-11-10，产品`37aa1c40`。** 现有HOST有限patch及RobotContactMaterial速度摩擦律，已接到原设备COM-wrench→完整J→完整M→候选/共同发布路径。没有另写摩擦定律、状态修补、设备窄相或静摩擦历史。

## 解决的实际阻塞

旧采样桥与plan分别拒绝正μ。新HOST例先核原patch/law符合独立字面向量，再真实RED于旧admission，收据44保留。父仅解除这两个过时子域限制，保留原材料类型/构造验证及全部资源/owner/功/方程检查；公开prepare_contacts描述同步更新。

旧“正μ必须拒绝”预期改成真实错误law类型和反向canonical材料pair key负控，不删wrong-owner、topology、group、容量与分离检查。所有原μ0路径的力律计算未改。此前33body一致性修复仍保持，形状单模型上限仍256。

## 实际执行

| 验证 | 真实结果 | 范围 |
|---|---|---|
| 首admission RED | RAW1，原HOST patch/law正确后被μ0 guard拒绝 | 保存原输入与首失败，不当物理门失败 |
| 完整新旧HOST桥 | **63 passed /1.39s** | 其中10项新速度摩擦实例，保旧桥相邻覆盖 |
| interpret | **12 passed /3.30s** | 两真实运动case各8连续接受步+10个admission配置负控 |
| CUDA | **12 passed /9.37s** | 同上，真实设备执行，无HOST动力学替代 |
| 新wheel源码外HOST | **32受载world-steps** | 两独立scene原力—运动递推、选择复位隔离，基础导入不加载Torch |
| 新wheel源码外interpret/CUDA | **各8受载接受步** | 偏心浮体完整独立断言、真实site cap0拒步保态、错误owner拒绝 |
| wheel字节核 | **448产品成员，0mismatch** | 新包与当前src逐bytes同，核心依赖仅NumPy |

原firstlint通过；firstformat仅两个新测试需要排版。修复后七个受影响Python文件的**raw AST全部相同**，final lint/format通过。没有为通过修改质量、材料系数、h、误差门、pin或cap。

## 独立力学与时刻

HOST四点box patch使用k64N/m、c4Ns/m、μ1/2、v_s=1/8m/s、深度1/32m、v_n=-1/8m/s。Fn=5/2N；切速0、v_s/2、2v_s的总切力依次0、-5/8、-5/4N。检查权重恰一次、双方同作用点总矩相消、动态双方的相对速度、开离无粘附和分离无力。

真实设备例使用h=1/1024s、非零原入步速度：

- 自由浮体的偏置球：mass2kg、I=.5、初始力(-5/4,0,5/2)N，真实COM力矩(0,-125/256,0)Nm。独立6×6M/J、逐步Fn/Ft、根平移/转动和四元数漂移均与实际结果核对。
- 固定转动关节的偏置球：I=1/4、初始ω=-1/2rad/s；原弧线速度产生力(3/32,3,0)N、τ_z=3075/2048Nm。每步从实际入步q/v独立构造旋转、接触点速度与半隐式运动。

两例都核原entry load来源，候选只传输同一材料点与共同作用点，分别记录实际材料功、作用点功和入步功率×h；三者不相互冒充，也未声称任意有限h下总能量单调。真实容量拒步保留全部原q/v/tick，不生成伪接受。

## 证据和剩余范围

`artifacts/p65-connections-20261110-0800-01/44–53`保留RED、源/命令、排版前后、HOST与两backend结果；`resident-friction-outcome.json`汇总实际结果与前后源绑定。`54–59`保存新wheel/安装、仓库外consumer、原完整gzip和448成员字节核。每个设备例保存完整8步的原请求、entry/accepted、receipt、独立期望和资源统计。

本片认证普通fixed/floating树的两个独立运动机制，以及HOST packet的双方力/矩。**未认证**G1设备窗口、affine或同树双动态的正μ受载窗口、卸载/重入、材料锚点静摩擦/滚动历史、设备站点力核、设备窄相、无HOST往返训练张量或受载W256。G1设备短留出另按N8核适用域和执行，不能从本片测试数直接推断。

后续先用完整G1设备短窗检验模型/高body/摩擦的组合，再按[调研](../development/p65-device-policy-contact-research.md)把站点力核和公开张量契约逐项设备化。原HOST全动作证据复用，不重跑已绿长窗。
