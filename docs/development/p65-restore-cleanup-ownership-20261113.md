# G2-R0：restore 失败候选的清理所有权

状态：FROZEN_TESTS_FIRST，2026-11-13 05:10。下一异构扩容的资源研究发现具体共享接缝缺口：mujoco_checkpoint.restore 在候选构造成功、_verify_bank 或 native observer 失败时，replacement.close() 的 Exception 被忽略；与既有 reset/clone 路径不同。底层 DeviceResources 真正 UNKNOWN_RELEASE 已有全局隔离，不能声称它必然丢失；但若 close 在调用 ledger 前抛出，现分支没有显式保留 replacement owner，也隐藏了清理错误。

本小片只修失败候选的所有权与异常可见性，不授予W32或live/peak内存资格，不改力学/参数/设备内核/正常restore结果。与正在编码的M-B1文件不交叉。

父只可修改 src/phymeo/backends/mujoco_checkpoint.py 的该失败分支：catch BaseException as primary；候选close失败时先在原 runtime._checkpoint_retired 保留该确切owner，再抛 BaseExceptionGroup(primary,cleanup)，原q/v/act/force/receipt/clock/revision/activebank不发布。close成功则原primary传播。复用原close()及底层sticky UNKNOWN，不自动重试真实UNKNOWN/free。

原source agent只NEW tests/test_mujoco_tyx_restore_cleanup.py。测试先行，复用原clone-cleanup的固定scalar FILTEREXACT双world模型/真实一步入口；两个失败位置（完整bank验证后、末world observer后）×cleanup成功/调用ledger前可重试异常/清理BaseException；interpret/CUDA真实候选bank，保完整异常/原state/checkpointbytes/owner引用/close计数，修复后真实restore重试及后继对独立clone，最终每bank只实际释放一次。纯文本/Git，无执行。真实ledgerUNKNOWN本次不重新注入，原cleanup核资格复用，不能把此可重试前置失败说成UNKNOWN可重试。

父唯一串行执行：先完整新测试在旧分支出现预期RED，定位后窄修，再完整两模式通过；若失败不是该预期先诊断而不改oracle。新产品最终wheel中包含修复，安装恢复/真实后继覆盖使用公共消费者，源码外不注入私有故障。06:15新功能冻结/07:30收口不变。
