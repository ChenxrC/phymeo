# Opaque triangle query preview（仅实验，不改变公开默认）

仅自动接受硬件三角形命中，光照、BRDF/MIS、面积积分格点、随机序列、曝光、分辨率与每批16条主路径均不改。既有原始 build/DLL/源文件不写；生成新 source/shaders 副本。

- Shadow：`Opaque | TerminateOnFirstHit`。严格落在原 `(bias,tMax)` 内立即返回 true；任何端点提交命中均对**原整段**重跑原 NoOpaque。端点优先提交并不证明段内没有其他遮挡。
- Closest：Opaque **不**使用 first-hit；取得全程最近命中，严格内部直接使用原 instance/primitive/barycentric。端点回到原完整 NoOpaque 查询。原 closest 只显式排除下界，保留其 FLT_MAX 上界语义。
- 只支持当前全不透明三角形。未来透明、alpha-test、程序化 AABB 不在本实验合同；禁止把本实验直接推广为这些几何的正式路径。
- 等距/共面异材质 tie 不新加稳定顺序保证；配对捕获若有颜色差异必须报告，不把传感一致当颜色等价。零增速或回退成本高只说明该场景无收益。

## 父串行执行顺序

在隔离树根 `C:/Users/38347/.codex/worktrees/render-secondary-budget/phymeo`，采用匹配正式 build 的 Python 与 `rendering/src`，将 `rendering/examples` 加入 PYTHONPATH。以下变量由执行者绑定实际已验证路径；原预算源优先选择生成 `$baseline` 的原 shader source，不使用未知 WIP。

```powershell
$source = '<matching source>/rendering/src/phymeo_rendering/backends/_shaders'
$baseline = '<matching original complete SPV directory>'
$library = '<matching original DLL>'
$compiler = '<verified glslang.exe>'
python -m unittest discover -s rendering/examples/preview_opaque -p test_overlay.py -v
python rendering/examples/preview_opaque/overlay.py --source $source --baseline $baseline --compiler $compiler --output artifacts/rendering/preview-opaque/build-01
python rendering/examples/preview_opaque/endpoint_probe.py --source $source --baseline $baseline --library $library --compiler $compiler --output artifacts/rendering/preview-opaque/endpoints-01
python rendering/examples/preview_opaque/paired.py --library $library --overlay artifacts/rendering/preview-opaque/build-01 --output artifacts/rendering/preview-opaque/paired-01
```

`overlay.py`默认只重编 grid8 的 direct、reflection、indirect_progressive 主片元（direct亦为guide）。其余所有SPV逐字复制。生成 manifest 核所有原 SpecId 名称/默认值不变、新 SpecId74=1、编译日志、实际源/原产物/新产物 SHA。不重编 DLL，不建立新设备ABI。`paired.py`核磁盘所有SPV与真实公开session DLL/fragment SHA，不允许请求没有实际patch的片元。

若确需叠加已有且独立验过的 secondary2/4 源，给 overlay 重复 `--fragment <精确主片元名>`，三个主模式均需提供；原 direct guide 自动加入，paired 传相同 `--secondary 2` 或4。源不支持某格点应编译失败，绝不自动降档。

## 证据与预算

- HOST合同4方法，当前仅author，尚未执行。
- Endpoint实际16 captures，8场景×原/新；包含空段、恰下端、恰上端、内部，以及两个端点分别先列且有内部遮挡。query-probe的RGB仅编码bool/最近instance，**不是光照颜色**。真实遍历顺序不可控，另HOST规则反证“上端优先后直接false”的错误捷径。
- 配对112 captures：direct/mirror各1批，room与薄挡左右位置各4批，seed0/1、顺序A/B/B/A。4批原始mean都是64 spp，重复顺序不是新独立seed。严格报告RGB、rawmean、guide、3传感字节，逐项差异、实际profile GPU interval/host wall、close内存计数。三项progressive场景的稳态2..4批单列成本，不称独立ray traversal计时，也不因场景相等就推断通用等价。
- 所有首错、编译日志、原始readback保留，输出目录必须新建。只在端点与配对等价实际通过后讨论加速；当前尚无实测收益结论。

## C 稀疏单跳包 + opaque（新增包装，C主生成器不动）

`compose_sparse.py` 只接受已生成/已编译且明确SHA的C prepared目录。先逐文件验证原 `source-manifest.json.generated_sha256`，在**新目录**复制这些文件；只patch两份source GLSL。C采样/重建/profile/私有runtime/DLL完全保留。编译C的6个真实main/guide片元，所有原SpecId逐项不变并加74；其余SPV字节不动，保存原manifest文件、原/新hash及编译日志。源码copy的成本不计入capture速度。

```powershell
# C examples 当前owner（只读使用，不改该目录）：
$cExamples='C:/Users/38347/.codex/worktrees/p65-forest-tests/phymeo/artifacts/rendering/nov24-worktrees/appearance/rendering/examples'
$sceneExamples='C:/Users/38347/.codex/worktrees/phymeo-rendering/phymeo/rendering/examples'
# 由父绑定真实已验证prepared/编译器，禁止覆盖原包。
$original='<actual C prepared>'
$opaque='<NEW C+opaque prepared>'
$manifestSha=(Get-FileHash -Algorithm SHA256 -LiteralPath "$original/source-manifest.json").Hash.ToLowerInvariant()
python -m unittest discover -s rendering/examples/preview_opaque -p test_sparse_overlay.py -v
python rendering/examples/preview_opaque/compose_sparse.py prepare --prepared $original --output $opaque --expected-manifest-sha256 $manifestSha --compiler '<glslang.exe>'
python rendering/examples/preview_opaque/compose_sparse.py pair --original $original --opaque $opaque --c-examples $cExamples --scene-examples $sceneExamples --output '<NEW paired capture>' --device-index 1
```

配对wrapper严格串行四个独立Python子进程 A/B/B/A，每进程只activate一个prepared后才import renderer；不可把两个私有Python runtime混在同进程。场景复用同一Panda实际normal/material/HDR与两灯，固定59/60/149、720p、seed0/1。每进程16captures（每seed2warmup+3×RGB/all），共64；每个真实RGB计时后另取同入态四通道、重建线性值和incoming guide。比较RGB/三传感/scene-linear reconstruction/guide/原实际采样计数逐字，保scene/frame/光源/材质/配置和实际DLL/主片元SHA。

成本只取本次RGB同步完成墙钟与对应GPU render区间。guide/调试mean/PNG/压缩位于计时外；不删真实主pass/guide/reconstruction或变更提案数。两轮比值单列，没有提升就保无提升；此前96² opaque收益不写进C性能结论。C的`primary_proposals`仍只指可见锚点提案，不冒硬件查询总数或全图spp。

原端点probe已覆盖的是同一exact query body；C组合额外要求真实配对输出不变。不能把query wrapper推广到透明/alpha-test/AABB，也不保证异材质共面tie顺序。本新增3 HOST尚未执行，C组合尚未编译/实拍。

### Windows 清单别名修复与只复制恢复

新生成包的 `generated_sha256` 统一 `/`，拒绝重复路径别名（含大小写）、越界及 JSON 重复键。读取旧合法 Windows 清单不改其原字节。首次已编译但验证失败的包保留原样，可明确恢复到新目录：

```powershell
python rendering/examples/preview_opaque/compose_sparse.py recover --failed '<first failed prepared>' --original $original --output '<NEW recovered prepared>' --expected-manifest-sha256 $manifestSha
```

恢复仅接受两份指定 shader 的已知 before 反斜线键 / after 正斜线键冲突；其余任何冲突拒绝。重新核原 C 清单、源/运行时、DLL、全 SPV、六个原 profile 与 opaque marker 和编译日志，再逐文件复制并核 SHA。不编译、不导入渲染器、不运行设备。新 `recovery.json` 给逐项复制清单；旧清单/旧 overlay 收据另存原字节。compiled 字节一律不变，只有新权威清单路径规范化；旧失败包不写入。恢复后配对仍需真实执行，恢复成功不是设备等价通过。
