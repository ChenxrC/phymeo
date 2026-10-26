# 环境构建与 CI 部署计划

日期：2026-10-27。当前已有文档、CPU 和 wheel 检查工作流与基础包/锁文件；尚未取得远程 Actions 结果或配置分支保护。安装基础包已在本机 uv/pip 验证，tyx 解释器/C++ CPU/本机 CUDA 也已运行；远程 GPU runner 尚未配置。证据见[进度记录](../progress.md)。

## 1. 支持层级

| 环境 | 默认工具 | 依赖与验证 | 首期状态 |
| --- | --- | --- | --- |
| 文档 | Python 3.11+，uv 或 venv/pip | 标准库检查，无第三方依赖 | 本批交付 |
| CPU 参考 | Python 3.11/3.12，NumPy | 解析/合同/XPBD 测试 | 首版必需 |
| tyx CPU | 固定 tyx SHA + 可选 LLVM 或 C++ 工具链 | 公共 API、解释/编译执行 | 独立集成门槛 |
| CUDA | NVIDIA 驱动、匹配的 CUDA/tyx 环境 | 真机执行、驻留、数值对照 | 有 runner 后启用 |
| PyTorch/AI | 平台匹配的 PyTorch wheel | 指针、流、梯度、模型有效域 | 后续 |

首期支持 Windows/Linux CPU；Python 3.13 是扩展矩阵，依赖验证后加入。macOS CPU、HIP、多 GPU 不从上游能力直接继承为本项目承诺。

## 2. 现在可执行：文档环境

uv 维护版本初始固定 `0.11.2`，Python 使用 3.11 或 3.12；CI 当前选择 3.11。工具升级单独提交。

若尚未安装 uv，可先用已有 Python 执行 `python -m pip install --user uv==0.11.2`，再执行 `uv python install 3.11`。pip-only 用户可用 python.org/系统包管理器安装 Python 3.11；下面的 `py -3.11` 要求该解释器确实已安装。

### uv（跨平台）

```sh
uv run --no-project --python 3.11 scripts/check_docs.py
```

只运行标准库脚本，不安装 tyx、PyTorch 或 CUDA。也可使用已安装的 Python 路径代替版本选择。

### pip/venv（Windows PowerShell）

```powershell
py -3.11 -m venv .venv-docs
.\.venv-docs\Scripts\python.exe -m pip --version
.\.venv-docs\Scripts\python.exe scripts/check_docs.py
```

### pip/venv（Linux）

```sh
python3.11 -m venv .venv-docs
.venv-docs/bin/python -m pip --version
.venv-docs/bin/python scripts/check_docs.py
```

无包需安装，故不执行空的 pip install。Linux 若缺 venv 组件，由宿主环境维护者安装对应 Python venv 软件包。

## 3. 包骨架后：同一依赖真源、两种安装方式

以 `pyproject.toml` 声明公开依赖，`uv.lock` 保存解析结果；从它导出 `requirements/dev.lock` 给原生 pip。不要手写第二套版本范围。

已交付 `pyproject.toml`、`uv.lock`、`requirements/dev.lock` 和独立的 `requirements/tyx.txt`。CPU 核心仅依赖 NumPy；dev 组包含 pytest、构建与静态检查工具。tyx 私有 Git 依赖单独安装，避免无权限的贡献者连核心测试都无法运行。

### uv 开发路径

```sh
uv sync --locked --group dev
uv run --no-sync python -m pytest
uv run --no-sync python scripts/check_docs.py
uv run --no-sync python -m build
```

### pip 路径（在已激活的新虚拟环境中）

```sh
python -m pip install --require-hashes -r requirements/dev.lock
python -m pip install --no-deps --no-build-isolation -e .
python -m pip check
python -m pytest
python scripts/check_docs.py
```

构建后端也须包含在导出的 dev 环境中，才使用 `--no-build-isolation`。该路径验证原生 pip，不用 `uv pip` 代替 pip 兼容性测试。

维护者更新依赖：

```sh
uv lock
uv export --locked --group dev --no-emit-project --format requirements-txt --output-file requirements/dev.lock
uv export --locked --no-dev --no-emit-project --format requirements-txt --output-file requirements/runtime.lock
```

CI 用同样命令重新导出 dev/runtime 文件并执行 `git diff --exit-code`，发现漂移即失败。runtime.lock 只包含安装 wheel 所需的公开依赖。项目自身以 editable 或 wheel 安装，不放入带 hash 的第三方清单。私有 VCS 依赖不能假装具备 pip artifact hash 校验：使用下一节的固定 SHA 通道；正式部署改用带 SHA256 的构建 wheel。

## 4. tyx 安装、认证与版本

初始研究基线固定：`99f6df7c0fd8fdeef77f97df83914da27b64b7ee`。更新必须提交配对测试证据。

`requirements/tyx.txt` 固定内容：

```text
tyx-compute @ git+ssh://git@github.com/ChenxrC/tyx.git@99f6df7c0fd8fdeef77f97df83914da27b64b7ee
```

已有 Git/SSH 读取权限后，在 CPU 环境上叠加安装：

```sh
# uv 管理的环境
uv pip install --python .venv --no-deps -r requirements/tyx.txt
uv run --no-sync python -m pytest -m tyx --require-tyx --tyx-mode=interpret

# 已激活的原生 pip 环境
python -m pip install --no-deps -r requirements/tyx.txt
python -m pytest -m tyx --require-tyx --tyx-mode=interpret
```

`--no-deps` 仅因该固定 tyx 版本的核心运行时依赖 NumPy，且前面已安装核心环境；LLVM/CUDA extras 需要另外维护已锁定 profile，不能套用此命令声称已装好 GPU。安装后运行 `uv pip check --python .venv` 或原生 `python -m pip check`（uv 创建的环境不默认内置 pip）。uv 之后若执行精确 sync 会移除未声明的 tyx，故集成步骤使用 `uv run --no-sync`；重新 sync 后重新安装 overlay。

本地通过 SSH agent/系统 Git 认证，不把私钥复制到仓库。CI 默认 `GITHUB_TOKEN` 不能假定能读取另一个私有仓库。受信任 tyx job 使用限定仓库的只读 deploy key 或短期 GitHub App token，host key 从 GitHub 官方发布的密钥材料固定，而不是无校验信任网络扫描结果。fork PR 仅运行无私密依赖的 CPU 参考测试。

tyx 最终交付建议由受信任上游流水线构建 wheel，记录源码 SHA、wheel SHA256、Python ABI 和工具链，再上传私有制品存储；phymeo 消费该制品。不要把 SSH Git URL 固化到面向公众发布的包依赖中。

### 已验证的可选模式与 CUDA profile

`TyxSemiImplicitSolver(mode="interpret" | "cpu_cpp" | "cpu_llvm" | "cuda")` 支持显式选择，不使用自动静默回退；用 `with` 或 `close()` 结束其独立 runtime。当前已实测 interpret、cpu_cpp 和 cuda，cpu_llvm 尚未安装/验证。公开 API 边界由 `python scripts/check_public_tyx_imports.py` 检查。

`TyxXPBDSolver(constraints, mode=..., plane=...)` 使用相同模式与生命周期，执行 FP32 着色距离约束、平面投影及速度重建；其测试同样属于 `-m tyx`，不会因编译/设备异常自动 skip。着色器本身是纯 CPU 公共依赖测试。可选 `cuda_graph=True` 将一轮颜色/平面 kernel 记录为显式 CUDA 图，保留逐迭代 host 验收；其他模式下请求图直接报配置错误。LLVM 仍未验证。

已有 CUDA 驱动、NVRTC 工具链及上述 tyx 基础安装后：

```sh
# uv 管理环境；原生 pip 使用下一行替代此行
uv pip install --python .venv --no-deps --require-hashes -r requirements/tyx-cuda.lock
# python -m pip install --no-deps --require-hashes -r requirements/tyx-cuda.lock
uv run --no-sync python scripts/check_tyx.py --mode=cuda --require-device
uv run --no-sync python -m pytest -m tyx --tyx-mode=cuda --require-tyx
uv run --no-sync python examples/tyx_cloth.py --mode cuda --cuda-graph
uv run --no-sync python examples/compare_cuda_graph.py --steps 30
```

原生 pip 环境直接使用 `python` 运行上述探针、测试和示例。探针必须执行非空 kernel 并核对数值；缺依赖、缺设备、编译失败都返回非零。`--require-tyx` 把缺包的 skip 变为失败，所选后端执行失败不会被跳过。CPU C++ 将 mode 换为 cpu_cpp；不加 require-device。

图专用测试同时标记 `tyx` 与 `cuda`。选择 interpret/CPU 时它们明确 skip；选择 `--tyx-mode cuda --require-tyx` 时必须真实执行，不能因图构建/重放错误跳过。相同场景的图/直接启动比较交替执行顺序，首步单列，其余报告均值、中位数和 p95；计时包括分配、传输、捕获、主机验收与释放，不只统计 GPU kernel 时间。

`requirements/tyx-cuda.in` 使用固定 tyx 所声明的依赖范围，`tyx-cuda.lock` 为单独带 hash 的解析结果。更新命令见 lock 顶部。此次本机组合：cuda-python/bindings 12.9.10、cuda-pathfinder 1.8.4、CUDA runtime headers 12.9.79，系统 Toolkit/NVRTC 12.6，驱动 566.07，RTX 4060 Laptop；Python 3.12.14。这条验证不代表其他驱动/OS 组合已通过。CUDA 当前每步主机/设备传输，未做设备常驻优化。

## 5. GitHub Actions 分阶段启用

### C0：文档检查（本批实际配置）

`.github/workflows/docs.yml`：push/PR；Windows + Ubuntu，uv + pip/venv 两种入口；Python 3.11；运行本地链接、SVG 结构和离线 HTML 检查。只读 contents 权限、固定 Action commit、超时、同分支取消旧运行。没有 tyx secret，没有 GPU 需求。

### C1：CPU 与包（工作流已配置，远程结果待核验）

`.github/workflows/ci.yml` 实际配置如下；不需要 tyx 私有凭证：

| job | 矩阵/触发 | 验收 |
| --- | --- | --- |
| quality | 每个 push/PR 或手动运行 | 文档链接、ruff、编译检查、公开导入边界、dev/runtime 导出一致 |
| cpu | Ubuntu/Windows × Python 3.11/3.12 × uv/pip | 数学、合同、失败路径；不能空收集 |
| wheel | Ubuntu/Windows × Python 3.11 | build sdist/wheel；新原生 pip 环境、源码目录外安装和求解 smoke |
| required | quality、cpu、wheel 汇总 | 任一失败/取消/跳过都失败 |

CPU 套件使用 `-m "not tyx"`，明确只验证公开依赖路径。私有 tyx-cpu 集成 job 仍待受信任凭证配置；本机已经分别运行 interpret/cpu_cpp/cuda，不能把本机证据写作云端结果。`scripts/verify_wheel.py` 会创建全新环境，只安装 runtime.lock 与构建 wheel，执行 `pip check` 和 `python -I` 求解 smoke，确认导入来自 site-packages 且没有导入 tyx/torch/warp。

不要用 `paths` 过滤必需的总体检查导致 PR 一直 pending；可以让 job 内部决定范围。分支保护由维护者在首次成功运行后设置，不能将“已提交 YAML”写成“已配置保护”。

### C2：GPU 真机（runner 就绪后）

- 独立 `.github/workflows/gpu.yml`，标签 `[self-hosted, linux, x64, phymeo-gpu]`，单卡 NVIDIA 起步。
- 初期仅允许受保护默认分支上的人工 `workflow_dispatch`，固定待测 SHA；维护者确认后才能执行。不要让任意 PR 在持久 GPU 机执行。
- 使用一次性 runner/隔离工作区，限制服务账号，不安装仓库写入或发布凭证。
- 开始检查 `nvidia-smi`、实际 kernel 执行及 FP32 对照，报告 driver/runtime/compiler/GPU。
- 设置 require-device；缺 GPU、全部 skip 或零收集都失败；JUnit 中保留 skipped 原因。
- correctness 先于性能；GPU runtime/tyx/Torch profile 单独锁定，先验证兼容后 pin 镜像 digest。
- 缓存键含 OS/架构/Python/tyx SHA/后端/工具链；冷编译和热态分别测量，不能跨不兼容 GPU 复用产物。

### C3：发布候选（用户另行授权后）

保护 tag +受控 environment；复用已验证 wheel，干净环境安装、示例与制品哈希核对，再生成 changelog/证据。PyPI 若启用用 trusted publishing；当前任务不创建公开发布或自动部署。架构图通过 Git 文件交付，暂不启用 GitHub Pages。

## 6. 环境建设清单与负责人

1. A：落地 Python/uv/pip 与基础 package；B：在另一 OS 按文档独立安装。
2. A：建立 CPU CI 与导出检查；B：添加解析/约束测试并检查失败确实能阻断。
3. A：配置只读 tyx 认证和固定 SHA；B：验证 package 边界与公开接口。
4. A：准备 GPU runner/镜像；B：维护小规模正确性场景。硬件未就绪时停留在 C1，不捏造完成。
5. A+B：GPU 正确后冻结基准和容差；实际维护者设置 required checks 与 environment。
6. 用户授权发布后，才接入 C3。

## 7. 官方参考

- [uv GitHub Actions](https://docs.astral.sh/uv/guides/integration/github/)
- [uv Git 认证](https://docs.astral.sh/uv/concepts/authentication/git/)
- [uv lock 导出](https://docs.astral.sh/uv/concepts/projects/export/)
- [pip hash 校验](https://pip.pypa.io/en/stable/topics/secure-installs/)
- [GitHub Actions 安全边界](https://docs.github.com/en/actions/reference/security/secure-use)

Action 固定 SHA 已于本次任务通过官方仓库 tag 查询核实；它们是选择的稳定版本，而不是“最新版本”声明。
