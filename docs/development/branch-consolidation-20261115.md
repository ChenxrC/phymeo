# 分支整理：dev 基线与 exp 实验线

日期：2026-11-15。用户授权：已合入的其他分支可删除，并从 dev 创建实验分支；随后明确同意使用现有 GitHub 认证修改默认分支。

## 完成状态

- GitHub 默认分支已从 `codex/architecture-v0` 改为 **`codex/dev`**，API 回读与远程 HEAD 均已确认。
- 稳定基线：`codex/dev` = `76a240d8499a1b6d117e47650015e85a73239b26`，本次保持不变。
- 从该基线创建、发布 **`codex/exp`**；独立面积压力实验只在 exp 推进。
- 删除 **131 个本地分支**（48 个直接祖先、83 个完整补丁等价）、**6 个远程分支**。
- 仍保留 12 个存在独有补丁的本地分支，以及一个未合并远程 WIP。不能将它们说成“全都已合并”。

本地初始 144 个，新增 exp 后删除 131 个，剩 14 个：dev、exp 与下表 12 个。
远程剩 dev、exp、tyx-rigid-wip 三个；因此是“两条当前开发线 + 保留的未合并工作”，并非无条件只剩两条 ref。

## 保留清单

以下数字是相对固定 dev 截点的 `git cherry` 独有补丁数，不是功能缺口数，也不代表这些旧分支一定应该再合并。

| 本地分支 | HEAD 前缀 | + 补丁 |
| --- | --- | --- |
| codex/h1-device-capacity | d84c0cc6 | 9 |
| codex/h1-foot-support | 226a6206 | 7 |
| codex/mujoco-tyx-t4-tests | 7aa3b0d6 | 2 |
| codex/p65-col-candidate-publication-20261111 | 5c9444e2 | 6 |
| codex/p65-contact-lifecycle | 21ac9932 | 13 |
| codex/p65-d6-tests-20261113 | 61e38c6e | 1 |
| codex/p65-deformable-query-b-20261112 | c587fb9b | 2 |
| codex/p65-device-patch-20261111 | a8cba78e | 2 |
| codex/p65-robot-checkpoint-20261111 | fdf064eb | 1 |
| codex/p65-robot-substep | c3f9de5f | 12 |
| codex/p65-scene-public-b-20261112 | 92844384 | 1 |
| codex/tyx-rigid-wip | f4771d0b | 1 |

远程 `codex/tyx-rigid-wip` 为 `10f3fa6815d6b06a4f79ef4064611f03b97cef0a`，不同于本地 f4771d0b，保留双方。
后续若要只保留 dev/exp，须先逐项判断这些独有补丁是应合并、已被新实现替代，还是用户确认放弃；本批没有强行合入或删除。

删除的远程分支：architecture-v0、p1-p2-runtime、p3-p5-dynamics、p6-5-handoff、p6-tyx-chain、rigid-cloth-coupling（均在 codex/ 下）。

## 保护与恢复

审计目录：`artifacts/branch-consolidation-20261115-01`。

1. 固定 dev/head 审核 ancestry；95 个非祖先分支用固定 hash 做 patch-equivalence，83 个全部等价、12 个有独有补丁、0 审计错误。
2. 删除前创建并验证 `merged-branches.bundle`，含首批 130 个分支引用，17,025,234 bytes。
   SHA256：`05aeec3ce802da6b903f8d6cc73d089b7216fdb86bd3400d077ca9eb5d0b9d8b`。
   旧默认分支在其后删除，其提交 b4cbc6b71ae0b780012061e5a8fb4fde91bd8269 仍是 dev 祖先；另有独立删除收据。
3. 本地删除绑定精确旧 SHA；远程删除绑定旧 SHA lease，避免删掉并发新提交。
4. 九个原工作树切换/分离前后状态一致；主工作树转 dev，父集成工作树转 exp。没有删除 worktree 目录、源码、venv、验收证据或未跟踪文件。
5. 父原三份 device-convex-patch WIP 保留；旧子树的 review 文档和用户 PNG 保留。
6. 初次默认分支 API 请求被自动审批拦截，明确获用户授权后才成功执行；凭据未显示或落盘。

运行明细：inventory、patch-equivalence、delete-plan、bundle 验证、worktree-transitions、local-deleted、remote-deleted、default-branch-change 与 old-default-deleted 收据。
历史分支已删除不表示其原验收范围扩大；不重跑旧测试或旧审计。
