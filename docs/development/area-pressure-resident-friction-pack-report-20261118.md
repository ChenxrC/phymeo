# I3w：设备当前布局直接装配共同摩擦输入

2026-11-18，codex/exp，源与测试提交 de7c31e3。首次且唯一完整新组在源码外安装包执行：**3 passed in 2.61s / RAW_EXIT=0**。没有先在父源码运行再重复安装包测试。

## 补上的连接

[设备入口](../../src/phymeo/backends/_tyx_area_friction_entry.py)把同入步 resident M、v、COM、机体 Jacobian、阻尼、bias/passive/applied 和完整面积样本，直接装配到已验收的 I3u 七个输入缓冲。双方点 Jacobian 均由 COM 线性/角速度 Jacobian 和实际 FP32 点构造；非阻尼输入为 applied+passive-bias+Dv0。输出接原共同摩擦 kernel，中间不读回数值，也不调用 HOST 问题打包。

这次输入来自 [I3v](area-pressure-entry-friction-report-20261118.md) 保存的真实设备字节，经独立安装包重新上传。因此验证的是实际生产布局的设备接缝，**尚非 live Scene 步内接入，也没有测得生产整步提速**。既有 Scene、type7、积分器、v1 观察和 friction=none 资产均未修改。

接口由调用者提供同 runtime 的缓冲；metadata 检查 shape/dtype/backend、生命状态和输出别名，不能替代生产 owner/phase 身份绑定。源 status 先于任何 payload/间接地址读取；pack 状态单独保存，不被下游求解器的状态清零掩盖。

## 首次实际验证

[完整测试](../../tests/test_tyx_area_friction_entry_pack.py)包含一个 HOST metadata 节点和两个 CUDA 节点。

- 真实 FREE 入步，分别 mu=.5、mu=0、空面反事实；另一个明确标为 synthetic-layout-only 的非零 JB 案例，检验双方点 J，不能称新物理场景。
- 原真实 D40 滑块空面入步。实际速度 .0185185168 m/s，正确离散参考 .0185185180；错误重复阻尼值 .0170370365，明确区分。
- 两条不同 nv/批大小管线；pack→solve 之间禁止 to_numpy 和 HOST _prepare。完整输入、最终法向/切向力、残差和非对称 Jacobian 由独立公式核验。
- 13 个设备错误覆盖上游状态、负数量、129点、pair/body 越界、法向、mu、微滑速度、刚度、有限源装配溢出、NaN、phi 与 h。所有失败世界的七输出清零、nv/ns=0，下游无数值迭代；上游失败码保留。
- 同一 resident/workspace 随后一次干净 retry 通过。七个 metadata 错误在发 kernel 前拒绝。累计16次 pack 与16次 solve，均属于本次三节点，不是重跑历史物理轨迹。

四 FREE 布局问题独立方程残差最大 1.071e-7；D40 空面为 9.559e-10。全部通过既定 FP32 门，严格 HOST 1e-10 门均 false。此处不宣称 Scene 的跨桶原子发布或 checkpoint 已通过；workspace.last_results 在私有 kernel 验证中保持空。

## 安装包、范围与证据

首版边界：1..6 DOF、1..32 world、完整样本≤128、机体≤128、有序 pair≤16；现三点三角积分最多126点。允许零样本自由运动；拒绝静默截断。M 精确对称/正对角的入口检查之外，正定性仍由共同求解器检查。

构建使用已固定 base HEAD 100082fb 加唯一新源文件字节，随后 de7c31e3 提交相同字节。578 Python + 7 资产、源/构建暂存/wheel/安装目录逐文件相同，RECORD 完整。外部 cwd + Python -I；315 个已加载 phymeo module 全来自安装目录。

- wheel：1,911,197 B，SHA256 95bcf0e92709977e5a09059f5e2e5811bc58f48b847b49f3536dbf01b262afb1。
- 新 module：SHA256 d976586cd0b330b7aa7778167913d32125f1c6d2e928393b0a7bd620c6b44655。
- 原完整真实 fixture：72,010 B，SHA256 50583f6e9f6875a77e09c3c3fa645588aa30c33b7b7d4b6113b1d1731bccd1b6。
- 两份完整保存结果：entry-pack-pipelines.json.gz / entry-pack-errors.json.gz；摘要与全部 raw receipts 位于 artifacts/area-pressure-resident-friction-pack-20261118-01。
- 初 Ruff 一项 I001 与初 format-check 未过，style 修复前后两文件完整 AST 相同；最终静态通过。所有首错保留，没有数值修复。
- 执行前后源码不变；原始外部 consumer、payload、run metadata 和 wheel 已归档，archive-receipt.json 记录完整字节绑定。

当前值可直接复用；不重跑 producer、旧窗口或同源已绿组。下一步是 [F3 实际积分与 F4 新载荷发布](area-friction-live-integration-handoff-20261118.md)，而不是再增加一套相似的打包参考。
