# P6.5 人工续作入口

状态（2026-11-01）　用户已选定可标定接触层研究路线，已授权的独立算术验证已完成。07:00 限时续作已删除。N1 完成，当前 N2，普通质量联合物理尚未冻结。

- **交付物**　A 取布、B 接稳、A 松开、B 放台并撤离。同次仿真驱动可暂停、逐帧查看接触力的 Three.js 回放。[原始验收](../superpowers/plans/2026-10-27-dual-arm-cloth-handoff-milestone.md)
- **可复用基线**　有限夹爪、独立 B2/R7 材料、限域的自接触响应与几何已集成。严格二进端点的倾斜相触限制及旧 R7 范围反例按原版本保留。[框架范围](../superpowers/plans/2026-10-30-p65-material-self-gripper-integration.md) · [端点反例](p65-b2-sc2-first-touch.md)
- **新实证**　独立固定十向量在解释器及 CUDA 各 10/10，实际中间字和返回字一致，CUDA 启动一次。次正规、假零、上溢各有原始记录。没有联合回放或完整材料行的通过结论。[逐字收据](../benchmarks/p65-fp32-reduction-verification.json)
- **当前研究**　保留原几何半厚度，以单独 δ、面积刚度 k 和阻尼 d 描述有限接触层。零间隙有有限初储能。SC-L1 设计试片五个留出点和六个一维对照均通过。支撑载荷全窗至少为相对外载的 93.71%，冲击硬核余量大于 0.5760 mm。[候选定律与标定](p65-contact-layer-calibration.md) · [材料结果](../benchmarks/p65-contact-layer-screen-results.json)
- **步内实证**　SC-L1 的两种工况、三质量、三步长共 336 步满足峰值、冲量、时间矩、硬核间隙和功的数值判据。0.5 kg 冲击的卸载尾段有 8 个原时间步，发布的速度与间隙无法同时由非负压力产生，其中 7 步冲量恰为零。这排除了给当前端点单换插值的修复。[完整数据和反例](../benchmarks/p65-contact-layer-intrastep-results.json) · [逐步观测](../benchmarks/p65-contact-layer-intrastep-steps.csv)
- **下一动作**　只先判别 0.5 kg、1/64 s 的最后一步。共同发布可实现的压力冲量、时间矩和可续步状态，把算术发布误差独立记账，核下一步实际读回后的速度与间隙。若须改变权威状态或接受条件，先记录具体代价。单步通过才扩至其余七个反例及内部路径。[节点与停止规则](../superpowers/plans/2026-10-31-p65-delivery-replan.md)
- **执行边界**　独立算术及本次一维筛选均已结束，不重复同源已绿组。N3/N4、远程 CI 均未启动。
