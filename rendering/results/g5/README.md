# G5 · 源码外 wheel 复现

本目录定义交付与执行口径。**当前只完成 tests-first RED 和作者实现；wheel 构建、隔离安装和设备 F0 尚待串行实测。** 不把源码树内的既有 R0 结果计为此项通过。

## 交付边界

`phymeo_rendering-0.1.0a0-py3-none-any.whl` 是平台无关的 Python、资产与构建源码包，不携带 DLL、SO、PYD、SPV 或本机 SDK。基础安装不依赖 SAPIEN、NumPy或物理包；参考额外依赖仍为 `sapien==3.0.3`、`numpy==1.26.4`。

setuptools 直接将现有 `rendering/assets/panda` 映射至安装包的 `phymeo_rendering/_panda_assets`，仓库不维护重复资产。包含原 manifest、57 个 glTF、10 个无损 DAE 源归档、xacro、package.xml、LICENSE、NOTICE、来源 README。其原始 SHA 不变。`panda_visual_assets()` 安装后无需传路径，完整核验后返回稳定文件 URI；源码树入口与显式 `asset_dir` 均保留。这里只承诺正常解包安装的 wheel，不承诺 zipimport 临时资源生命周期。

原生构建输入包括 CMakeLists.txt、C++、头文件、`.inc` 和完整 shader 目录（含 `.glsl` 公用片段）。外部构建仍需 CMake ≥3.20、C++17 编译器、Vulkan headers/loader link library、glslang 和有效的 GPU 驱动。F0 打开 Khronos validation，另需可发现的 `VK_LAYER_KHRONOS_validation`。包不会下载这些工具或把缺失设备替换成软件后端。

## 固定顺序

1. 在独立打包环境安装 `build` 与项目声明范围内的 setuptools，再执行 `python -m build --wheel rendering`。离线构建可从预备 wheelhouse 安装这些工具，使用 `--no-isolation`；明确记录版本。
2. 运行 `rendering/tests/test_g5_packaging.py`，设置 `PHYMEO_G5_WHEEL` 为本次准确 wheel。审计 RECORD、基础依赖、平台标记、每一份当前 Python/native/shader 源及全部资产字节；缺件、有效 RECORD 下篡改资产和伪装二进制均拒绝。
3. 调用下面驱动。`--work-dir` 必须是源码外尚不存在的目录。驱动复制并 SHA 绑定独立 F0 夹具及脚本，创建全新 venv，使用 `-I`、空 PYTHONPATH 与 `--no-index` 安装。基础探测先证明默认 Panda 来自 installed package，并且无 SAPIEN/NumPy 安装或导入。
4. 驱动对安装包内的 `_native` 配置和编译；显式加载此新产物完成 native F0。参考依赖只能来自本地 wheelhouse，然后分别执行 SAPIEN raster 与 PT。一个设备路径失败不阻止记录另一路，最终报告仍为 failed。

```powershell
python rendering/tools/g5_reproduce.py `
  --source-root C:/candidate/phymeo `
  --wheel C:/candidate/dist/phymeo_rendering-0.1.0a0-py3-none-any.whl `
  --wheelhouse C:/offline/wheels `
  --work-dir C:/independent/NEW-G5-run `
  --native-device-index 1 --sapien-device cuda:0 `
  --validation-layer-path C:/sdk/validation `
  --cmake-arg=-DVULKAN_INCLUDE_DIR=C:/sdk/include `
  --cmake-arg=-DVULKAN_LIBRARY=C:/sdk/lib/vulkan-1.lib `
  --cmake-arg=-DGLSLANG_VALIDATOR=C:/sdk/bin/glslang.exe
```

设备索引只是当前验收机的示例，换机必须显式确认。`--base-only` 仅做安装与默认资产证明；`--run-path native-raster` / `sapien-raster` / `sapien-pt` 可重复以选择具体设备路径，均不冒充全项验收。驱动默认三条路径全部执行。

离线 wheelhouse 需要 SAPIEN 的完整传递依赖；仅复制现有 site-packages 不能证明重新安装。当前已知旧环境版本见 G3 实际安装 log；新运行保存 wheelhouse 每个 `.whl` 的 SHA、全部命令/退出码、实际 pip freeze 和独立源码外路径。预备依赖可联网下载公开官方 wheel，但被验的安装本身无网络 fallback。

## 证据与限制

外部 `report.json` 保存候选 wheel/RECORD 审计、文件绑定、全新环境、各阶段命令和日志，以及各路径独立 status。`captures/base.json` 记录真实 import/asset 路径；各设备 JSON 记录 30 种公共路径×非空通道组合的真实 `supports` 查询，以及实际执行的 F0 路径/通道、设备与生命周期 diagnostics。`supports=True` 不等于该组合已经拍过：实拍为 native raster 四通道、SAPIEN raster 四通道、SAPIEN PT RGB。原 F0 的受光变化、几何探针、背景、错帧和不支持路径拒绝、不可推进只读快照门原样复用。

原始 RGB/深度/ID/VALID 和 PNG、两光照 metrics、所有文件 SHA 均留在外部目录。没有物理推进。此处不新增原生光追、跨平台设备保证、任意随机种子或所有 Panda 后端容量保证。

## 已有 RED 证据

固定 tests-first 提交 `03b203cf`，从 Git archive 独立提取：10 方法，3 个 `.inc/.glsl` 包配置子例失败、20 个缺新入口/映射的子例错误、1 个未提供实际 wheel 的节点跳过，退出码 1。原日志在作者工作树 `artifacts/rendering/g5-red/01/`。后续 GREEN 与实际 wheel 结果需另加，不覆盖此证据。

## Windows CPython 3.11 的离线准备清单

[参考环境版本清单](reference-windows-cp311.txt) 来自既有 G3 环境的真实 `.dist-info`，不把缓存目录当作原 wheel 文件。[独立构建工具清单](build-requirements.txt) 将 setuptools 固定在当前 build-system 的 `>=77,<83` 内；参考运行环境里的 84.2.1 与构建环境分离。

只读检查发现：本机 `C:/Users/38347/AppData/Local/uv/cache/wheels-v6/pypi` 未找到 SAPIEN 的完整原 wheel 集；G3 环境包含展开后的 SDK 和一个 NumPy wheel，但不构成完整离线安装源。因此优先复用已保存的原 wheel；缺件时从公共 PyPI 固定版本补齐，不从 site-packages 重新打包冒充官方发行。下面是待实际执行的准备命令；下载成功前不宣称版本可用或 SHA 已验证：

```powershell
C:/Python311/python.exe -m pip download --index-url https://pypi.org/simple `
  --only-binary=:all: --no-deps --dest C:/independent/g5-wheelhouse-build `
  -r rendering/results/g5/build-requirements.txt
C:/Python311/python.exe -m pip download --index-url https://pypi.org/simple `
  --only-binary=:all: --no-deps --dest C:/independent/g5-wheelhouse-reference `
  -r rendering/results/g5/reference-windows-cp311.txt
```

下载后清单必须保存每个包的官方版本 JSON 地址 `https://pypi.org/pypi/NAME/VERSION/json`、选定 wheel 文件名、官方 SHA256、实际 SHA256和字节数。任一固定版本/平台 wheel 缺失应报告阻塞；不得改版本后仍声称复用了旧环境。下载阶段可联网，之后构建和安装均只用本地文件。

构建应从最终集成**精确提交**的 `git archive` 提取到新的构建目录，避免旧 `build/lib` 或 egg-info 残留；在另一个空 venv 以 `--no-index --find-links <build-wheelhouse>` 安装构建清单，然后 `python -m build --wheel --no-isolation --outdir <dist> <snapshot>/rendering`。运行时 `--source-root` 指这份相同快照。报告同时保存 Git 提交、archive SHA、wheel SHA及外部探测 driver SHA。任何后续源码变化形成新 wheel/hash，不能沿用这份 F0 结果。

已存在且可复用的 native 工具位于父集成树 `artifacts/rendering/g3-table/toolchain/`：

- headers：`Vulkan-Headers-vulkan-sdk-1.4.363.0/include`；
- loader import library：`vulkan-1.lib`（加载器运行时由系统 Vulkan 驱动提供）；
- shader compiler：`glslang-16.6.0-windows-x86_64-release/bin/glslang.exe`；
- validation manifest/DLL：`Validation_Layers_Windows_64/`。

沿用 VS 2022 C++ 编译器和 CMake，实际版本/缓存必须留档。外部构建只引用上述 SDK，C++/.inc/CMake/shader 输入必须来自**安装后的 wheel**。当前机器 native adapter index=1、SAPIEN `cuda:0`；每次实际诊断仍保存设备名，不能把这组索引当作跨机器保证。
