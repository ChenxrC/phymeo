# P6.5 H1a 表面点端口 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans。主实施者按测试先行执行；P6.5本轮已实现分支最终一次独立审阅，不为每个小任务派reviewer。用户已授权继续，不重复询问实施方法。

**Goal:** 面/边重心作用点与同域成对载荷保留共享响应，成为真实布面接触的可复用基础。

**Architecture:** 既有不可变PointPort负责真实动力学；新增主机FP64映射B及成对关联D，完整响应分别为B R B.T和D R D.T。没有时间推进或状态族分支。

**Tech Stack:** Python3.11/3.12、NumPy、既有tyx固定SHA三模式；uv/pip。

**Spec:** `docs/superpowers/specs/2026-10-28-p65-surface-ports-design.md`。

## Global Constraints

- 从已验收P6 6dd5372进入`codex/p6-5-handoff`；不改旧P6数学/调度/阈值，不合并主分支、不发布包。
- 截止2026-10-28 00:00 UTC；此切片不能代表完整P6.5完成。
- 新映射主机FP64；底层实际后端保持显式，tyx计算FP32/存储FP64，不伪称新增GPU映射。
- 权重非负、每行和1（atol1e-12），不归一化；错误输入拒绝；返回owned FP64；数值不可表示拒绝，无回退。
- 实际tyx所有命令设置工作区TEMP/TMP、TYX_CACHE_DIR；只观察性能，依赖问题与集成成本分开。

## Review Focus

- 同一底层节点/连杆在多个映射行重复出现：必须保留交叉响应，由Task1/2解析测试覆盖。
- 两个成对端点实际相同或重复对相互抵消：必须一个底层候选而非覆盖，由Task2覆盖。
- 有限FP64权重及冲量乘积/累加溢出：必须SolverFailure、状态不变，由Task1覆盖。
- 空映射、非法dtype、调用者后来修改weights/pairs：明确shape与所有权，由Task1/2覆盖。
- 名称为tyx但实际未执行：Task3检查真实三模式底层计数及解析结果，不依赖假后端。

## Task 1：加权世界点端口

**Files:** Create `src/phymeo/coupling/surface_port.py`、`tests/test_surface_port.py`。
**Interfaces:** `WeightedPointPort(base, weights)`；domain/dtype/weights（拷贝）；positions/velocities/response(impulses)/apply_impulses(impulses)/snapshot/validate_candidate(state)。底层PointPort由调用者提供，n=base.positions().shape[0]。

- [x] RED：`test_barycentric_positions_and_full_response`，三节点inverse_mass=[1,2,0]、B=[[.5,.5,0],[0,.25,.75]]，手工B diag(w) B.T非对角=.25，配置不变。
- [x] RED：独立逐节点累加、重复底层索引、功率对偶1e-12、响应对称非负、输入/weights/返回状态隔离；空m=0/n=0、错误dtype/shape、NaN/负权重/和错误与底层失败；可表示输入但累加溢出→SolverFailure。
- [x] Run `.venv/Scripts/python.exe -m pytest tests/test_surface_port.py -q`；Expected：缺少实现导致失败。
- [x] 实现上述映射，检查输入和底层数组边界；只通过既有端口计算响应/候选，不重建质量模型。Expected：所有上述测试通过。
- [x] 相关旧PointPort回归、ruff通过后小提交。

## Task 2：同域成对响应与反作用

**Files:** Modify `surface_port.py`、`tests/test_surface_port.py`。
**Interfaces:** `PointPairOperator(port,pairs)`；domain/dtype/pairs（拷贝）；relative_positions/relative_velocities/response(impulses)/apply_impulses(impulses)/snapshot/validate_candidate(state)。它不实现世界PointPort的positions方法。

- [x] RED：`test_pair_shared_node_cross_terms_and_single_candidate`，手工D B diag(w) B.T D.T；实际两个点共享节点与重复底层索引，检查反作用和总动量。
- [x] RED：同点对零响应、反向重复对抵消、重复对累加、空对、非整数/负/越界索引、输入和pair隔离；真实两杆解析J M^-1 J.T，候选q精确不变。
- [x] Run Task1同一测试文件；Expected：Task1保持通过、Task2因缺实现失败。
- [x] 实现先合并D.T*p再一次调用底层，不分别推进快照。Expected：全文件与旧机器人/粒子端口回归通过。
- [x] ruff/边界检查，小提交。

## Task 3：实际tyx、导出和交接

**Files:** Create `tests/test_tyx_surface_port.py`；Modify `src/phymeo/__init__.py`、README/进度/问题台账；Create本切片观测报告。
**Interfaces:** 公开导出两类；复用`TyxClothParticipant`、`TyxRobotParticipant`工作区的`point_port`，不引入新tyx私有导入。

- [x] RED：公开导出缺失；新增实际节点/机器人加权与成对解析测试，atol=rtol2e-5、完成launch>0、快照不变，所有资源finally关闭。
- [x] 三模式运行 `pytest tests/test_tyx_surface_port.py -q --require-tyx --tyx-mode <mode>`；Expected：各真实后端通过。CPU依赖隔离与相关旧端口/参与者回归保持通过。
- [x] 记录构造/查询小负载观测、成功/失败、主机映射/底层执行口径；无等价Warp不作比较，不优化。
- [ ] 完成本轮切片必要全回归/打包/源码外wheel/CI与一次阶段末整分支审阅；未完成项保留，不提前称P6.5通过。小提交推送并核对远程SHA。

## 自查

Task1输出PointPort给Task2，Task2仍委托同一base snapshot；Task3只验证/公开上述接口。两类都不承担接触检测、非线性位置load撤回、摩擦或完整时间步，因此不依赖尚未冻结的综合场景候选参数。完整H0/H1几何/H2–H5仍需继续，不能以此计划完成替代总goal。
