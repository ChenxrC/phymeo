# 场景耦合切向与不变几何复用：第四批

基线 9e1693110880f23245d8d2e780d37316f1d7d21b。用户已授权继续实施。功能优先，按多案例验收；本批不重跑第三批已绿同源矩阵，不重开旧周期任务。

## 设计与边界

在现有 ContactScene / RobotScene 上增加显式可选的耦合切向模式。默认 smooth friction 行为保留。保持当前法向采样的 friction 参数、求积、实际作用点与完整几何检查；仅在采样之后用法向力加新切向力替换选中站点的旧切向力。相同差量必须同时进入双方物理载荷和公开观测。

从同一入口组装完整自由速度，包括外载、当前法向、重力、陀螺项、机器人 PD 与 bias。所有站点使用完整交叉响应 W。只真正推进一次。以入口材料地址和作用点分别重建真实末态位移，核对 motion、实际功与弹簧能量，再共同发布物理、切向历史、geometry、loads、events。失败不消耗历史或接受步。

站点显式登记 StationKey、shape pair、材料 chart、局部锚点和切向轴，唯一对应当次正法向载荷；不从 world coordinate hash、row order 或上一步 contact_events 推断身份。当前事件由同态真实 sample 纯 stage。最近点变化不是材料滑移。

首切片 HOST FP64 刚体及普通纯移动关节；有内投影粒子、mimic 未映射等明确拒绝。刚体转动必须经过实际 exp-map 位移/功门，不能用 hJv 冒充真实材料位移。默认工程门预先声明，失败保留，不为单个案例调参数。

## 独立分工

- A 原 robot agent：NEW runtime/_coupled_tangent_step.py、tests/test_coupled_tangent_step.py 与报告。公共 A3/B3 的世界坐标适配、全 W、空集释放、真实 motion/work gate、纯 commit。
- B 原 RD agent：NEW runtime/material_stations.py、tests/test_material_stations.py 与报告。显式登记、当前载荷唯一绑定、同一入口材料/作用点实际末态位移。
- C 原 RR agent：不变 HullShape 边界与 chart 复用。先给可精确绑定的最小缓存接口与负控；旧几何入口由父补窄 hook。
- 父：旧 scene 共享接缝、新真实场景适配、公共导出、完整实际推进与回滚测试、串行执行和源码外 wheel。A/B/C 只隔离树 author/Git，项目执行只父持槽。

## 执行与成功标准

- [ ] 三线先 tests author；父保留首次有效失败，再实现，所有项目命令串行 fresh guard。
- [ ] A/B 独立专项：完整交叉响应、两类真实位移、能量、copy/binding、歧义/遗漏、排序、退出重入、retry/stale。
- [ ] 场景：有限面真实压载下静持/滑移/卸载；至少两种质量或外载，非人为指定 N。
- [ ] 机器人：真实 URDF 移动关节多站点共享祖先，PD/bias/free motion 和双向反力；不能只核独立代数。
- [ ] 偏心转动的实际运动留出与失败回滚；旧默认模式兼容专项。
- [ ] 不变几何：缓存命中/失效/隔离、有界容量、完整输出一致；新的窄范围实际计量，不重跑旧大矩阵。
- [ ] 必要专项、静态、独立审阅、源码外 wheel 示例；更新 continuation 和诚实支持范围。

## 防止无效循环

若同一接缝两轮没有新增能力或可复现的最小失败，停止扩展该线，记录真实缺口。只对已定位问题修代码；不加长证明、不把另一独立模块 green 当场景通过、不回避实际旋转误差。性能结果仅据同输入完整输出的实测；不声明 Newton 全面对齐或整体加速。
