# G5 · 源码外 wheel 复现

**G5 已通过。** 2026 年 11 月 23 日，候选 `5fdf4ec37896153c93a5337a44663d581bd1b286` 在 Windows、CPython 3.11.9、RTX 4060 Laptop 上完成源码外离线安装。基础安装直接从 site-packages 核验 57 份 Panda 网格和 59 个实例，随后从安装包携带的 C++ 与 GLSL 编译出本机 DLL 和 12 个 SPIR-V。共执行 23 条外部命令，均返回 0。[已验 wheel](phymeo_rendering-0.1.0a0-py3-none-any.whl) · [30 项 HOST 原始结果](built-wheel-host.log) · [安装与设备完整记录](installed-report.json)

| 用同一张标定桌检验安装结果 | 实际输出 |
| --- | --- |
| 自研 Vulkan 光栅 | [打开标定桌画面](native-raster.png) |
| SAPIEN 参考光栅 | [打开标定桌画面](sapien-raster.png) |
| SAPIEN 参考路径追踪 | [打开标定桌画面](sapien-pt.png) |

16 块粗糙度标定片上，将已知粗糙度填入材质与从贴图读取的两次曝光相比，最大 RGB8 通道差依次为 1、0、0。设备比较在编码视频前进行。[实测摘要](summary.json) · [503 份原始图像、日志与设备记录](evidence.zip) · [证据文件清单](evidence-files.json)

另外 12 个阶段均通过，涵盖已安装模块的 HOST 检查、三条路径的 UV 与粗糙度贴图、自研与参考的环境光及 Neutral 显示，以及自研硬件阴影查询。完整报告给出各阶段的设备、输入、输出文件摘要与退出码。wheel 大小为 8,134,733 字节，SHA-256 为 `0ba6e6c65515148c1c5357b6613747e8c53d3d552c998efd5569dbcec44faf43`。

此前候选 `82ca3a0b` 在首次 wheel 检查中因 Panda 资产目录的 Git 控制文件报错。修复后的检查器只允许资产根目录两份明确列出的 Git 控制文件在 wheel 中缺席，继续核验资产清单及其列出的 72 份资源，共 73 份文件。下方保留开发阶段的命令、故障与测试记录。

## 交付边界

`phymeo_rendering-0.1.0a0-py3-none-any.whl` 是平台无关的 Python、资产与构建源码包，不携带 DLL、SO、PYD、SPV 或本机 SDK。基础安装不依赖 SAPIEN、NumPy或物理包；参考 extra 仍为 `sapien==3.0.3`、`numpy==1.26.4`；独立 `textures` extra 为 `Pillow==12.2.0`，只在显式加载贴图时使用。

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
  --feature-smoke --native-device-index 1 --sapien-device cuda:0 `
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

## 纹理 extra 与源码外冒烟计划

使用厂家纯色 Panda 或基础合同无需 Pillow。选择 PNG/JPEG base-color 贴图时安装 `phymeo-rendering[textures]`；参考贴图路径安装 `phymeo-rendering[sapien,textures]`。这个依赖没有挪入基础包，也没有让原生纹理依赖 SAPIEN。`TextureMap`/`TexturedMaterial` 声明和 `TriangleMesh.texcoords` 校验本身不导入 Pillow，只有 `load_texture()` 明确解码时需要它。

官方 Pillow 12.2.0 cp311-win_amd64 wheel已下载至现有 `wheelhouse-official/reference/`，7,082,626 bytes，SHA256 `6e6b2a0c538fc200b38ff9eb6628228b77908c319a005815f2dde585a0664b60`。官方元数据地址为 `https://pypi.org/pypi/pillow/12.2.0/json`；精确 files.pythonhosted.org URL和实际hash在准备证据 `g5-prepare/05-pillow-official/manifest.json`。此步骤只有下载和哈希校验，没有安装。旧准备清单的20个原wheel与两次离线闭包验证仍留存；新增Pillow另列，不能把原14包参考证明改写成已经验证15包安装。

最终发布前，在**最终集成提交生成的新 wheel**上执行下面计划。它补充原F0，不用源码内的影片代替：

1. 在完全干净的源外 venv 仅安装基础wheel，核无Pillow/SAPIEN/NumPy，导入合同与默认Panda成功。先保留这份基础证明。
2. 从本地 wheelhouse 对**同一准确wheel路径**安装 `[textures]`，使用 `--no-index`；若需要参考同时选择 `[sapien,textures]`。记录Pillow真实import路径/版本、完整pip freeze、官方wheel hash；模块必须来自这份源外venv，不能挂系统Pillow或源码PYTHONPATH。
3. 复制并SHA固定当前已通过的四色PNG、相应FLOAT VEC2 glTF及独立CPU期望，不复制产品实现。通过installed `load_texture/load_mesh`核文件hash、顶行优先RGB8/UV、alpha拒绝、错hash拒绝和缺UV拒绝。
4. 用公开相机分别对 installed native resident 与 SAPIEN raster/PT拍同一四色夹具。包含同贴图两个不同linear factor、负/超1 UV重复、双线性中点；因子检查使用预注册的线性读数/独立像素期望。保存所有原图和每条路径的实际配置、设备、纹理SHA、上传字节、真实通道和拒绝项。SAPIEN两条路径都须核因子，不能只看纹理显示出来。
5. 保留错hash/缺UV/不透明度违规时无正常capture、close后的会话资源诊断；最后再拍原无纹理F0，避免把贴图扩展带来的回归漏掉。

这些包外纹理步骤当前**待执行**。已有源内共享58 HOST和reference因子4 HOST绿色，只证明该源码合同及CPU准备，不代表安装包、实际GPU采样方向或公开影片已通过。生成真实wheel前不得填入设备通过结果。

## IBL extra 与同一源外环境的准备

方向 Radiance 环境准备选择 phymeo-rendering[ibl]，唯一依赖固定为 numpy==1.26.4，与 reference 使用相同版本。基础 wheel 仍无依赖；原生环境准备不安装 SAPIEN，PNG/JPEG 纹理解码另选 [textures]。NumPy 的官方 wheel 已在现有离线 reference wheelhouse，不新增下载。

最终安装先保留无 NumPy、SAPIEN、Pillow 的 base 证明，再在**同一个新建源外 env**从确切 wheel 路径用 --no-index --find-links 安装 [ibl,textures]。共享环境 HOST、相同 SHA 的真实图片解码及六面输出需在此环境中执行并记录实际包路径、版本和文件 SHA；环境声明、颜色方向和材质都沿源内已固定输入，不临时换图。真实 native 与参考 PT 环境拍摄留在整套设备 smoke，不能以 HOST 转换成功代替实际图像验收。之后可在此环境增加 [sapien]；它不会要求另一个 NumPy 版本。

当前仅完成 extra 元数据与独立安装范围声明。源码外安装、图片准备和设备验证仍待最终候选 wheel 冻结后执行。

## 可复用源码外 feature 驱动

在前面的 g5_reproduce.py 命令添加 --feature-smoke，即启用同一新建 env 的如下序列；不添加时保留原 F0 路径。此开关不能与 --base-only 同用，可以和 --run-path 组合限定明确执行的设备路径。

1. 基础安装与默认 Panda 证明先执行，确保没有 NumPy、Pillow 或 SAPIEN。
2. 从同一个准确 wheel 安装 [ibl,textures]，仅使用已保存的本地 wheelhouse。执行原 test_environment.py 的六项 HOST；从原 texture_calibration.py 生成相同16块 UV 板，实际加载PNG和FLOAT VEC2 UV；从原 reference_hdri_probe.py 生成同一SHA的RGBE并准备六面。此阶段明确拒绝已装SAPIEN，不能以参考依赖掩盖原生准备依赖。
3. 从 installed package 中的 C++、.inc、完整 shader（包含 sky.vert、sky.frag 和 .glsl）构建原生库；执行原 F0、sealed UV图和 native_ibl_calibration.py。各项记录实际DLL SHA和原门槛结果。
4. 同一 env 再从相同 wheel 安装 [sapien]，执行参考 raster/PT F0、两条参考 UV 图和原 reference_hdri_probe.py。不同用例的相机与采样协议保持原样；每条结果单独记录，任一失败不会被其它通过项覆盖。

新 g5_feature_smoke.py 将七份必要的独立脚本/测试逐字节复制至源外 feature-fixture，manifest绑定全部SHA；运行前后均核验。它不复制产品实现，所有 phymeo_rendering 模块必须位于该 env 的安装目录。原接受脚本通过同一个 Python -I 启动，只有这份已绑定的 examples 目录进入sys.path。Windows USERPROFILE、AppData、系统PATH和Vulkan layer路径保留；TEMP、TMP、TMPDIR使用此新运行自己的temporary目录，避免破坏系统加载器或依赖原checkout的临时文件。

每个用例保留原输出与原协议，额外生成 *.installed.json，记录before/after包路径、实际第三方版本/路径、输入manifest、DLL、结果和全部输出SHA。父 report.json 的 features 字段区分 passed、failed、not_run；未构建/未满足前置项不能计为通过。固定fixture明确使用参考cuda:0；请求其它设备时明确拒绝该UV协议，不偷偷改变封存脚本。

本轮新驱动与五项 HOST 负控仍为文本准备，未安装、构建或执行最终 wheel。G5 已完成的旧元数据 HOST 为12通过、1个未提供实际wheel的节点跳过；HDRI参考源内独立首次设备校准通过不等于源码外交付已通过。

## 扩展验收：粗糙度贴图、参考显示变换、硬件阴影

未传 `--feature-smoke` 的基本安装/F0 配置保持原行为。传入后，三个默认路径会要求全部 12 个 feature 阶段：HOST、三端 UV、原生/参考 IBL、三端 roughness、参考 Neutral、原生 ray-query、原生 Neutral。所有设备输入来自同一个外部新环境，产品只从被验 wheel 导入；不会复制产品 Python 源作 oracle。

- `roughness-native`、`roughness-sapien-raster`、`roughness-sapien-pt`：逐字复制并 SHA 绑定候选中的 `texture_calibration.py`，仅传既有 `--role roughness`。每端为同几何的贴图与解析材质各一次曝光，16 块、每块 49 像素；保留最大 RGB8 差 ≤3、解析色差跨度 ≥40 和 raster 三种传感附件逐字相同的原门。原 roughness GREEN 字节是线性数据，不能按 sRGB 解码。另一个原 `--role color` 的 16 块 base-color 测试继续执行。
- `neutral-sapien`：复制原 `reference_neutral_calibration.py` 及已有 HDR/sphere 辅助脚本，实际 raster/PT 各拍 clip 和 pbr-neutral，共四曝光。曝光仍为 8、PT 64 spp/4 bounce/无降噪；同路径实际线性读回须完全相同，raster depth/ID/VALID 完全相同，固定前景至少 64 像素从全白剪裁恢复。保存真实线性 raw/RGB/PNG 和闭会话诊断。本项只计 SAPIEN 的四曝光；原生 Neutral 由下面的独立安装后用例签收。
- `ray-query-native`：复用原 `native_resident_smoke.moving_shadow` 及两份原 glTF，唯一材料路径变化是把 URI 定位到复制后、SHA 相同的外部文件。通过实际公开 `NativeBackend(resident=True, shadow_mode='ray_query')` 跑遮挡/向两侧移开的三曝光；保留原 receiver 1.125 m、blocker 0.875 m、误差 <2e-6 m、ID/VALID、亮度/上传/生命周期门。保存每曝光四通道原字节、RGB PNG 和真实 `resident_ray_query_info.enabled=1`。这是硬件阴影查询，主可见性仍是 raster，不是 native path tracing；不支持的设备直接失败，无软件 fallback。

- `neutral-native`：逐字复制 `test_native_display_device.py` 两个已接受的公开测试，以及提供独立 Decimal 参考与原 quad 资产的 `test_reference_neutral.py`、`test_native_ray_query_device.py`。在相同安装环境、新 DLL 和全部 12 个 SPIR-V 上实际执行两方法：常量 HDR 背景与独立 GGX 灰面，各含 software／ray_query、灯面 2／4／8、clip／Neutral，共 24 个 7×3 曝光。保持原背景 ≤1 RGB8、灰面中心 ≤2 RGB8、三个传感通道逐字相同的门；同时要求两场景各档的中心从全白剪裁恢复为低于 255。不得 skip、不得只给旧报告。记录真实 public API、每个选中 shader 的新 build SHA、每曝光四通道 raw、诊断和完整测试日志。未复制产品源码或调用其显示变换作 oracle；不重复 177 色样本或旧 shader 兼容套。此前 `cb7dcb27` 的源码内结果是前置，不能计作安装后通过。
新增独立输入只允许 `reference_neutral_calibration.py`、`native_resident_smoke.py` 和原 table/block glTF；另加上述三份原生 Neutral 测试，加上原七份文件构成固定有界集合，每个文件单独 SHA。选择清单也写入 manifest。对应新协议未选时，其额外文件无需存在，便于旧候选明确收窄验收。

只有尚不进入该候选的功能才应显式略过，例如 `--feature-smoke --skip-feature neutral-sapien:not-in-this-release`。每项必须给非空理由，HOST 不可略过。缺硬件、异常或失败不自动转为略过。略过项以及未选设备路径均记 `not_verified`；若其余请求项通过，顶层 `status=partial`（CLI 退出 0 只表示所选流程结束，不代表完整 G5 通过）。完整默认验收只能为 `passed` 或 `failed`，本机已验证功能默认不略过。

当前扩展 tests-first 固定为 `68f7d7fd`，真实 11 HOST 方法 RED：2 failures、11 errors（包含子例），0.132 秒。日志保存于 G5 作者树 `artifacts/rendering/g5-prepare/10-extended-red/`。本节描述待验收的执行合同；没有把既有源码内设备结果算作 wheel 外通过。最终构建必须使用包含这些独立脚本的精确候选提交，实际每项安装后结果和新 DLL/shader SHA 另存。
本轮扩展 HOST GREEN：11 项 feature + 13 项 packaging，共 24 方法，23 通过、1 个真实 wheel opt-in 明确 skip，3.058 秒，退出 0。固定原日志位于 `artifacts/rendering/g5-prepare/11-extended-green/`。该结果验证调度、SHA 绑定、结果门和包配置；尚未构建/安装最终 wheel，不是源码外设备验收结果。进程 guard 保留真实 idle MSBuild nodeReuse 记录（父进程已结束、无编译子进程、短间隔 CPU/I/O 不变），没有将它伪报为机器进程为零。

原生 Neutral 安装闭环 tests-first 为 `3c29fcb7`。新增四方法真实 RED：7 个失败、3 个错误（含子例），0.054 秒，退出 1；缺功能与缺证据确实被捕获。日志为 `artifacts/rendering/g5-prepare/12-native-neutral-red/`。最终 installed 用例仍等待 G5 精确候选与串行验收。

原生 Neutral 驱动补齐后，完整 `test_g5_features` 的 15 个 HOST 方法实际全部通过，0.208 秒、退出 0、无 skip，日志为 `artifacts/rendering/g5-prepare/13-native-neutral-green/`。这只验证安装后协议的调度、文件／新 shader 绑定和证据负控；24 次实际安装后公开曝光仍由最终 G5 运行产生。
