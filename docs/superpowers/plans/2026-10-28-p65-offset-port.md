# P6.5 面偏置端口实施计划

Spec: [冻结合同](../specs/2026-10-28-p65-offset-port-design.md)。继续专用分支codex/p6-5-handoff；用户要求PPT后继续P6.5，未给新截止。此计划只是下一可验收切片，完整里程碑继续。

## Global constraints

先测试后实现；不能放宽旧接触共点门槛代替正确表面速度/力矩。不得修改旧P6积分/旧接触求解器。禁止把host映射伪称CUDA。不得合并主分支或发布。push专用分支已授权。Windows执行ledger与测试输出使用本计划自己的ignored目录。

## Task 1: 冻结测试并实现固定构型端口

Interfaces: TriangleOffsetPointPort(base,triangles,weights,offsets)，PointPort的positions/velocities/response/apply_impulses/snapshot/validate_candidate/candidate_audit；查询triangles/weights/offsets/jacobians返回副本。

Files: src/phymeo/coupling/offset_port.py；src/phymeo/__init__.py；tests/test_offset_port.py。

1. 写独立解析、旋转/虚功/差分、共享响应/所有权/拒绝测试。
2. `.venv/Scripts/python.exe -m pytest -q tests/test_offset_port.py`，Expected: 缺少offset_port模块导致失败（RED）。
3. 实现冻结公式、归一化、数值拒绝门禁、加载合并和实际候选审计。
4. 同测试Expected:全通过；旧surface/graph/contact回归通过；ruff通过。
5. 提交Task1，记录RED/GREEN及实际范围。不能把下一Task的真实设备验收提前称通过。

## Task 2: 真实后端、工程观测和阶段交付

Interfaces: 消费Task1完全相同PointPort；TyxClothParticipant.point_port持有设备执行，Node候选验证必须来自实际结果。

Files: tests/test_tyx_offset_port.py；scripts/benchmark_offset_port.py；docs/benchmarks/p65-offset-*；遗留/瓶颈/tyx_flaw仅按实际证据记录。

1. 写真实节点端口解析载荷/力矩/能量、共享作用点测试；interpret/C++/CUDA均实测，不把跳过算通过。
2. 全核心suite、全CUDA suite、ruff、文档、本机uv与pip现有环境、wheel源码外smoke。Expected:全通过或如实记录阻塞并修复正确性问题。
3. 测试完成后串行性能观测，分清端口host映射与base执行；不做优化。
4. 一次fresh-context最终代码审阅，范围73bcda6之后新工作，不重复审阅已验收基础。严重/重要问题一次TDD修复，自行复验，不开展逐Task审阅。
5. 更新docs/progress和continuation；推送专用分支，核对远程SHA与准确提交CI。Expected:一致且CI成功。完整P6.5尚需动态接触账本/CCD/弯曲/单爪/双臂/Three.js，heartbeat继续，不标complete。
