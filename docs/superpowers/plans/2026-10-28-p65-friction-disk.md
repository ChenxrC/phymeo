# P6.5 H2a 切向摩擦局部算子 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans。主实施者按测试先行继续用户已授权任务；阶段末一次整分支独立审阅。

**Goal:** 可独立验收的各向异性黏着/滑动冲量，不冒充完整夹布接触求解器。

**Architecture:** FP64固定2×2正定有效响应，静摩擦无约束解＋动摩擦圆盘KKT单调根；输出纯值对象，无时间/状态/历史副作用。

**Tech Stack:** Python3.11/3.12、NumPy，无新增依赖。

**Spec:** `docs/superpowers/specs/2026-10-28-p65-friction-disk-design.md`。

## Global Constraints

- 分支`codex/p6-5-handoff`，截止2026-10-28 00:00UTC。只冻结局部摩擦，未决全局接触/CCD/控制不得默认通过。
- FP64；对称阈值1e-12*maxabs(K)，特征值比>1e-12；默认tol1e-10/96轮，tol上限1e-3，量纲/残差下限按规格。
- μd<=μs、Jn>=0；Jn=0或滑动半径0返回零冲量但不假称速度为0；奇异/不收敛/溢出明确SolverFailure。

## Review Focus

- 非对角K时径向裁切错误：Task1已知KKT解反例。
- μs>μd切换和零半径：Task1边界与单点圆盘测试。
- 局部返回有解但功为正：Task1/2独立能量不增检查。
- 病态响应/极端输入/一轮预算：Task1数值失败，不返回部分冲量。
- 物理端口实际响应与局部K不同：Task2实际端口应用与动量/能量对照。

## Task1：局部圆盘解

**Files:** Create `src/phymeo/constraints/friction.py`、`tests/test_friction_disk.py`。

- [x] 先写静态、开接触、动摩擦已知非对角KKT解、坐标旋转、随机正定样本、非法/病态/预算/溢出与所有权测试；运行确认缺实现RED。
- [x] 实现纯值`FrictionImpulseResult`与`solve_friction_disk`，包括独立返回验收；全文件GREEN及ruff后小提交。

## Task2：物理端口组合与公开

**Files:** Modify上述测试、`src/phymeo/__init__.py`、README/进度/性能记录；按实际后端需要补`tests/test_tyx_surface_port.py`。

- [x] 导出缺失RED；真实节点/共享关节端口形成切向响应后施加总冲量，独立核对速度、动量、耗散和构型不变；实际设备测试不以导入代替。
- [x] 公共导出和解析/旧回归GREEN，记录固定小负载延迟/迭代/失败与范围；不优化。
- [ ] 与H1a/H1b一起完成本轮末整分支审阅、必要回归、打包/源码外smoke、准确提交CI和远程SHA；未执行项保留。

## 自查

只有Task2使用Task1输出的Ns冲量；全局法向/摩擦图和时间积分仍未冻结，不能把局部解直接复用成位置load或重复计入lambda/h。局部SPD矩阵的所有权和坐标基由调用方负责，换基后必须同步变换v和t。
