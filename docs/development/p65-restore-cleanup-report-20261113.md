# G2-R0：restore 失败候选清理所有权报告

状态：**SOURCE_AND_INSTALLED_QUALIFIED**（父安装更新见末节）。本片只修 MuJoCo-TYX restore
提交前失败的资源所有权与异常可见性，不扩世界容量、模型域或动力学。

## 修复与绑定

- 冻结合同：[restore cleanup ownership](p65-restore-cleanup-ownership-20261113.md)。
- tests-first：`76c78ca733b2068daeada06c822505cf0b1d3bfa`；
  父样式整理 `5a43ea319a08d214c1a5725b4deaecfb18a99ee3`。
- 产品修复：`6672ea9f41115dc5fbc18987f616ace01d10c81c`，仅
  `mujoco_checkpoint.restore` 的失败候选清理分支，6 行新增、3 行删除。
- GREEN 执行 HEAD：`23757ae6a0d2a5ed3c6b15b0ec762bfd571e3b75`。
  两次执行的测试文件 SHA256 均为
  `B8B3138588286CEF3F0CBB70B4A850C59D7F7A087AE0AC8AE89356D79F969EE3`；
  GREEN checkpoint 源码 SHA256 为
  `D3954E8BDF6D79B559C3B2E900167C85C388095C23A3B877329C9BE3AD9D8CAC`。
  这些值引用原 final.json，没有重新哈希或重新运行。

候选已构建但完整 bank 验证或 detached native observer 失败时，原实现吞掉
`replacement.close()` 的普通 Exception。新分支捕获全部 BaseException：
close 成功仍传播原 primary；close 失败先将确切 replacement 强保留到
`runtime._checkpoint_retired`，再聚合原 primary 与 cleanup。所有 accepted
bank、state/control/receipt、tick/clock 与 revision 发布赋值仍在该分支之后。

## 实际 RED / GREEN

| 已保存执行 | 实际结果 | 解释 |
| --- | --- | --- |
| 178 old-source RED | RAW_EXIT 1；1 failed、4 passed；5.23 s | `preledger_exception-verified_bank-interpret` 收到原 MemoryError，未收到包含 cleanup 的异常组，正是预登记缺口；按 `-x` 停止 |
| 185 full GREEN | RAW_EXIT 0；12 passed；5.86 s | 同一测试、原物理输入与门完整通过，interpret 6、CUDA 6 |

原证据保留在父
`artifacts/p65-feature-until-0730-20261113-01/` 的
`178-r0-old-source-red/` 与 `185-r0-full-green/`。
各目录的 stdout.log、final.json、source snapshots 和逐节点
`basetemp/.../restore-cleanup.json.gz` 为证据入口。
RED 未覆盖的组合不凭推断称已执行；全部 12 组合由 GREEN 实际覆盖。

矩阵固定为两个失败位置 × 三种 cleanup × 两 backend：

1. 实际候选构建及完整 bank 验证返回后失败，或第二个 world 的原 native observer
   已实际返回后失败。
2. cleanup 成功、调用 ledger 前的 RuntimeError、调用 ledger 前的
   KeyboardInterrupt。
3. 原 scalar FILTEREXACT XML、worlds `(7, 91)`、`h=.001`、原控制
   `(.6, -.2)`。真实一步后保存 capsule，再推进一步建立不同的 accepted 入口。

GREEN 每节点保存完整 packed candidate、mirror、异常链与 owner/ledger 记录，并检查
失败后完整 snapshot/integration state、checkpoint bytes、五个 publication
容器 identity、active bank 和 revision 不变。随后真实 restore 重试，并与此前
独立 clone 连续推进两步，核完整状态、held control、activation 与 last-step receipt
精确一致；每次物理调用均独立断言所有 world 实际接受。

## RED 的安全兜底与 GREEN 的实际释放

报告阶段仅抽读 RED 节点 ca4、GREEN ca4 与 ca11 的末态元数据，未重解压全证据：

- RED ca4：runtime 未保留 candidate；测试本地强引用仍在，finally 移除注入后
  安全清理仍为 OPEN 的 candidate。`fallback_cleanup=["failed-candidate"]`，
  三个被记录 bank 最终 CLOSED，各真实释放一次。
- GREEN ca4（interpret 普通异常）与 ca11（CUDA BaseException）：失败 candidate
  明确保留；真实 restore 重试成功，原 runtime.close() 清空退休 owner。
  `fallback_cleanup=[]`，四个被记录 bank 均 CLOSED，各真实释放一次。
- 完整 GREEN 测试在 finally 兜底之前就断言 runtime 已完成释放。
  计数以 ledger OPEN→CLOSED 转换定义，未将重复的幂等 close 调用计为重复释放。

本测试没有注入底层真实 UNKNOWN_RELEASE，也没有授权重试未知 native free。
可重试的是明确发生于调用 ledger 之前、未释放任何 native 资源的 Python 故障；
原 sticky UNKNOWN 隔离语义保持。

## 限制与后续

本片已取得失败路径的源码和安装功能资格，不是吞吐、峰值内存或 W32 资格。
正常 restore 的物理结果与 checkpoint 格式未更改。父已完成下节所列的第四包
源码外验证。报告作者只读取原日志/必要元数据并写文档，所有 RED/GREEN
项目执行均由父唯一串行槽完成。

## 父安装资格更新（2026-11-13）

**SOURCE_AND_INSTALLED_QUALIFIED**。第四包产品23757ae6、534产品成员同字节。193源码外仅运行本片的新12项，RAW_EXIT0，12passed5.67s；所有加载的phymeo模块均位于独立安装site。测试及两个固定helper复制到包外scripts；没有导入仓库tests路径或执行旧helper测试。

安装资格覆盖相同两个晚失败位置、三种清理情况、两个backend及实际恢复重试/连续后继。原178 RED保留；不改变UNKNOWN语义，不从本次owner修复声称完整生命周期预算或W32已完成。完整12份压缩记录及bootstrap已保存在第四包canonical归档，见[M-B1父安装记录](p65-multibody-deformable-report-20261113.md)。
