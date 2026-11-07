# 通用凸接触、转动逐面候选与设备接受态宽相

状态：三条功能切片已分别通过专项验证；当前进行 N4 独立参考及源码外 wheel 集成。没有固定结束时间，没有远程 CI、依赖变更或 push。本报告不代表整个引擎已与 Newton 功能对齐。

## A：同一接触表示接入真实所有者

球 S、有限胶囊 C、盒 B、一般凸包 H 的十种无序组合，经过共同材料见证、归一化点/线/面斑块和共同作用点载荷，进入已有真实 COM/J 映射。固定–固定、固定–浮根、浮根–浮根的 30 组实际加速，偏心力矩、注册反序、真实速度和失败重试均通过。另有机器人 S/C/B/H 与自由体 S/B/H 的24组；一般H保留完整原顶点，不以盒代替。

原支持的机器人–自由体及自由体–自由体配对继续使用其原 quadrature、pad 和公开 feature。混合场景按形状配对选择新路线，添加完全排除的凸体不会改变旧摩擦力矩。新一般凸体与 sampled pad 的组合仍明确不支持。法向歧义、轴深入一般凸体、rounded hull 等超出当前域的输入拒绝，不声称完整任意穿入解算或 CCD。

- A核心：53项斑块/载荷测试已通过，源 e26a4f1；复用既有验证。
- K3最终：58-a-k3-complete，73项新矩阵/事务/混合分派及25项原直接机器人接触，共98 passed / 6.27 s / RAW_EXIT=0。
- 59-a-public-windows：公开 examples/convex_contact_family.py，CC、CB、非盒HH各在 h=1/256 与1/512连续推进0.5s，共1152步。真实初速度 .08/- .12 m/s、质量2/3kg、刚度100N/m，完成分离→入层→压缩→卸载→分离，没有逐步回填状态。
- 三组细化的末态位移差均约1.43736e-6m；总冲量相对差约1.54171e-5，低于预声明1e-3m/5%工程门。平移滑轨角度恒零；不能用该窗口宣称转动路径细化也已完成。离散弹簧独立oracle、总动量及共同点虚功误差均小于1e-9。
- 原RD只读交叉审阅已释放。最后的完整资产 pad 名称歧义先校验再过滤，真实55 RED 1fail/1pass后修复。

失败记录完整保留：36一般H仍被编译成box是有效RED；38测试读取不存在的point_world_m字段；39原.25s窗口未到实际半周期卸载，改为事先按同力律/质量推导的.5s、相对速度.2m/s新窗口，未称原窗口通过；43旧OBB内部见证为有效RED。46/47的excluded-H测试首错是参数名slip_speed（真实名slip_speed_m_s），不算该物理问题的执行RED；46 rounded hull为有效RED。48修复后28通过。49/50首次style失败，51/52修正，57最终Ruff通过；没有重跑旧B/C套件补数字。

## B：转动 mesh–round 的设备逐面候选

5d3bbce 已接受 HOST39 / interpret5 / CUDA5。现有单转及双独立转轴/移动枢轴的真实时间采样使用TYX面级BVH；最终见证沿同一selector，完整包含/最远支持和时间搜索保留。五world输出留存299查询/598真实refit+traverse核。CUDA上传324048B、下载53960B、同步299次。窄相和时间控制仍在HOST，未声称更快或全设备CCD。详 p65-rotating-mesh-candidates.md。

## C1：当前接受球状态直接供设备宽相

78ec685 已接受最终 interpret36 / CUDA36。FreeSphere的真实接受缓冲直接生成设备AABB并喂shape BVH，暖态无动态snapshot、envelope上传或新增分配。W1/W4/W32、双缓冲、同tick restore、独立registration、拒步、异常/close均验证。必要status/bounds/candidates读回仍存在。这是当前姿态dynamic/static候选，不是接触响应、滑轨或通用机器人GPU驻留。详 p65-resident-collision.md。

证据统一位于 artifacts/p65-generic-collision-20261107-01；保存所有首错、命令、源码前后快照和原始退出码。C2当前仅原作者设计提案，须在N4完成后冻结实现；Newton留出和源码外wheel完成后追加实际结果。
