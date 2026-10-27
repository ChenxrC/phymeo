# P1/P2 独立整分支审阅

一次独立新上下文审阅覆盖 `bf6c41a2305dd35c19f33052c2d1bcefd8cbec7a..b26a9977844f3d4c3618829e1ba42a7eb55058cc` 的全部 P1/P2 实现，使用 gpt-6-astra。审阅范围包括注册状态、点端口、阶段运行器、NumPy/tyx 迁移、故障注入、公开入口和交付工具。审阅时完整性能和最终实验报告仍在生成；它们没有被当成已完成证据。

结果：Critical 0、Important 2、Minor 0、Declined to judge 0。实施者按实际影响接受两个 Important，在同一修复轮中先复现再修复；没有发起第二次审阅。

## 两项发现与验证

| 发现 | 用户可见影响与修复 | 回归证据 |
| --- | --- | --- |
| 派生状态/输入的额外字段被规范化时丢弃 | 原实现只复制注册字段，扩展的材料内部量可能消失而步骤仍被接受。入口及候选现在拒绝未注册 dataclass、动态实例属性、slots 字段；完整扩展应注册新状态族。只重写方法的子类仍兼容 | `test_unregistered_extension_fields_cannot_be_silently_discarded` 的 9 个组合及 `test_third_party_registered_family_rejects_further_unregistered_fields` 均先失败后通过 |
| 约束行重排绕过物理签名，复用过期设备着色 | 多重集签名正确表达物理相等，但不能保证缓存对应当前行布局。另绑定有序、带 dtype 的模型/约束指纹，tyx 同时绑定实际着色表；改变后在创建工作区前拒绝，重新构造程序可运行 | `test_reordering_bound_constraints_is_rejected_before_workspace_creation` 两种 CPU 顺序及 `test_reordered_device_constraints_cannot_reuse_stale_color_cache` 实际 CUDA，3 项均先失败后通过 |

第二项已证明旧颜色内出现共享顶点，存在并发写冲突风险；未把未执行的 GPU 数据竞争实验写成已观察到的随机错误。修复针对失效缓存的入口条件。

13 个新增测试首先全部失败。首轮较广验证为 74 通过、1 个既有错误信息匹配失败；保留原 `physical problem` 诊断语义后，新回归和该兼容性检查共 14 项通过。随后完整验证：

- Python 3.12.14 / NumPy 2.5.4 / 实际 CUDA：831 passed，278.30 s。
- Python 3.11 核心：707 passed、124 deselected，199.55 s。
- 原生 pip Python 3.12 核心：707 passed、124 deselected，210.91 s。
- ruff、文档链接、公开 tyx 导入边界（62 文件）通过；uv/pip 锁导出不变、依赖检查通过。
- sdist/wheel 构建通过；62 个包内 Python 文件与源码完全相同，源码外全新原生 pip 环境安装及新运行器 smoke 通过。

最终验证包源码 SHA256：`dd1dcd12f14ee1ccc11839dbc3cc5a54516e38f2d4d858109257ce46a2177669`。上述测试并发执行，耗时只作验证记录，不作为性能基线。中途因诊断文本调整而停止的三次运行不计作通过。原始本地日志为 `artifacts/tmp/p1p2-review-red.txt`、`p1p2-review-focused-green.txt`、`p1p2-final-{cuda,py311,pip312}-full.txt` 和 `p1p2-final-wheel.txt`。

## 实施决策及判断错误的代价

以下按计划台账中全部 `Ruling:` 的原顺序记录，没有省略或延期的 Minor。

1. 沿用已有专用 checkout，新建功能分支，以保留已经验证的工具链。代价：目录不能承载并行写入；若隔离判断错误，可能混入其他变更，因此全程检查 Git 状态。
2. 因既有 Bash 沙箱启动限制，以原生 PowerShell/补丁维护计划台账与任务 brief，保留等价的逐计划目录。代价：失去脚本自动记录的保障，实施者必须逐项核对测试与提交。
3. 物理历史明确放入 `SceneState` 数值字段，程序不保存隐藏的可变物理状态。代价：依赖外部隐藏状态的插件暂不支持，必须显式适配后接入。
4. 用分阶段工作区提取旧算法，由通用运行器组织时间、收敛和接受。代价：提取可能改变算术次序，必须用同次序逐数组迁移比较证明一致。
5. 在实现冻结、必要代码测试通过后开始唯一一次审阅，同时继续生成最终证据。代价：审阅发现修复后，相关完整轨迹和性能须重新生成；本轮采用修复后源码重跑，不以旧记录冒充最终结果。

未合并主分支或发布包。未来机器人/四面体、三角面碰撞、CCD、摩擦、自碰撞、真实 AI 训练和性能优化仍按后续大纲另行实施；本次审阅通过不扩大 P1/P2 能力范围。
