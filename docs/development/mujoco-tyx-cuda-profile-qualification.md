# N1 CUDA profile：环境资格与当前阻塞

2026-11-11。状态：PREINSTALL_BLOCKED_DISK；未下载 wheel、未安装环境、未执行 CUDA Torch 或 N1 正负案例。这里只记录真实预检，不把版本/驱动的静态相容判断当作运行资格。

## 固定候选

- 平台：Windows x86-64、现有 CPython 3.11.15。GPU：RTX 4060 Laptop，8188 MiB；驱动566.07。
- 原环境只有 Torch 2.15.0+cpu；不覆盖它。官方cu126索引没有对应2.15.0候选，选已发布2.14.1+cu126、cp311 Windows wheel。
- 原 TensorDict 0.15.5 的 metadata 要求 torch>=2.11.0a0；RSL 5.5.2 要求 torch>=2.6.0。候选满足声明版本约束，但 native ABI/真实GPU互操作仍待新环境验证。
- [候选依赖清单](../../requirements/training-cuda.txt)固定直接wheel URL和官方SHA256；不冒称全传递依赖lock或已验profile。TYX仍为原99f6df7 pin，未来另核新环境的ABI、native扩展及模块来源。

官方候选压缩大小为 **2,602,753,280 bytes**，SHA256 `4cff57ddd28c8bdf8bf2fc473f149c1b5f0063b1a6bd57593288e0bf18459adb`。当时C盘仅 **2,006,888,448 bytes** 可用，连下载完整wheel都不足，尚未计解压、依赖和运行证据。机器未发现另一文件系统盘；常规用户pip/uv缓存及原项目.uv-cache均不存在，未清理其它目录或现有环境。

原始预检位于 `artifacts/p65-cuda-policy-n1-20261111-01/environment-preflight.json` 与 `wheel-preflight.json`。首次读取原py311环境的TYX metadata路径不存在；随后从原主.venv实际direct_url确认pin。首次试选2.15.0+cu126在官方索引不存在，未启动下载；第二候选HEAD请求200。以上不是产品测试失败。

随后仅用两次有界HTTP Range读取官方wheel的ZIP尾部及中央目录，共1,238,142 bytes，未下载包内容。12,245个条目的展开长度合计4,191,672,496 bytes；**下载包加展开文件的空间下界为6,794,425,776 bytes**，还不含其它依赖/临时文件。记录在`wheel-directory-metadata.json`；完整wheel尚未下载，其内容SHA未验证，表中SHA只是官方索引给出的绑定。恢复安装建议先使可用空间达到8–10 GB，再按实际解析依赖核对峰值，不能把建议值当作已验证安装占用。

## 解除条件与执行顺序

先为独立环境腾出足够工作空间或提供有空间的安装位置。恢复时重新检查可用容量，并从wheel目录信息/依赖解析核实完整安装峰值与留存空间；不能仅释放压缩包与当前空闲的差额就冒险安装。不要删除源码、验收证据、既有venv或pin，不跟随旧Temp Junction清理归档。

父继续唯一串行owner：fresh入口检查→独立venv/profile与完整依赖来源记录→import/实际CUDA tensor与TYX互操作资格→N1既定两模型八组连续案例及负控→需要的源码外公共调用。失败保留原日志；不CPU fallback，不更换物理fixture/预算，也不在N1通过前启动N2 PPO。空间阻塞期间可完成N1代码/静态检查、ROB1/COL1接口冻结，不声明N1已验收。

依据：[PyTorch官方cu126索引](https://download.pytorch.org/whl/cu126/torch/)、[CUDA兼容说明](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)、[N1实施合同](../superpowers/plans/2026-11-11-mujoco-tyx-gpu-training.md)。CUDA12系列的驱动兼容只是选型依据；特定PTX/库功能可能有额外要求，真实资格仍以执行结果为准。
