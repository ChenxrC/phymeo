# TYX-002：Python 连续比较尚不能 lowering

- 状态：能力缺口已独立复现；不是已承诺功能的回归。
- 类别：语法/编译；优先级低，有等价写法，不阻塞 P6。
- 首次发现：2026-10-28，P6 材料数值边界实现；phymeo 基点 `56519c4`。
- tyx：`99f6df7c0fd8fdeef77f97df83914da27b64b7ee`，tyx-compute 0.1.0a0。

## 现象与预期依据

Python 常见的 `0 < x < 1` 在 interpret、cpu_cpp、cuda 入口均抛出
`LoweringError: chained comparisons are unsupported`，未进入后端执行。
改写为两个比较加 `and` 后，三个模式均给出正确边界分类。
需求依据是上层算法中常见的区间/材料域检查，不声称 tyx 已承诺完整 Python 语法。
Warp 未测，没有性能差距结论。

## 独立复现与环境

[复现器](repros/chained_comparison.py)仅导入公开 tyx API、NumPy 和标准库，不依赖 phymeo。
[原始记录](evidence/TYX-002-chained-comparison.json)含版本来源、5 个输入、三模式两种语法的全部结果。
Python 3.12.14、NumPy 2.5.4、Windows 11 build 26200；实际 CUDA 使用 RTX 4060 Laptop / 驱动566.07。
CPU/CUDA 设备编译器沿用固定 tyx 的默认发现配置，本探针没有独立采集编译器版本；lowering 拒绝先于编译器。

PowerShell 从仓库根目录运行，输出文件须不存在：

```powershell
$env:TEMP = Join-Path (Get-Location) 'artifacts/tmp'
$env:TMP = $env:TEMP
python tyx_flaw/repros/chained_comparison.py --output artifacts/tyx-chain-recheck.json
```

每种语法每模式执行一次；这是能力测试，无预热、分位数或速度结论。
CUDA 显式上传、调用、同步、下载，资源逐个注册后释放。
首轮未设置可写 TEMP 时，C++展开写法另遇本机沙箱临时目录权限错误；
设置工作区 TEMP 后通过。该环境问题不归因 tyx，首轮原始记录保存在
`artifacts/tmp/TYX-002-initial-default-temp.json`（本地忽略文件）。

## 定位及给 tyx agent 的建议

- tyx `lowering.py::_KernelLowerer.lower_expr`，固定版641行对多操作数 `ast.Compare` 直接拒绝。
- phymeo `_tyx_node_kernels.py::_bounded_material_product` 曾使用连续比较；已改成显式 `and`，未改依赖。
- 证据等级：独立最小复现，已定位共享 lowering 入口。后端是否能直接消费扩展 IR 尚未验证。

建议先决定该语法是否纳入公共子集，再降为短路逻辑；保留 Python 中间表达式只求值一次的语义。
只机械复制中间表达式可能改变有状态或昂贵表达式的行为，不能仅凭当前纯数组读取探针判定通用实现正确。
修复验收应覆盖等号边界、负值、非有限值策略、三项以上连续比较、短路以及单次求值，并保留现有 `and` 回归。
