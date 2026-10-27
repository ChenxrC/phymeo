# TYX-001：缺少内置 FP64 计算路径

- 状态：能力缺口已独立复现；非已承诺支持功能的回归。
- 类别：计算类型/编译能力；优先级：后续高精度求解路线，当前 P1/P2 FP32 验收不阻塞。
- 发现日期：2026-10-27；tyx 固定提交 `99f6df7c0fd8fdeef77f97df83914da27b64b7ee`。

## 问题、影响与预期

tyx 公共 API 没有内置 `f64` 类型。通过公开 `ScalarType` 构造描述符后，最小加法在 interpret、cpu_cpp 和 cuda 三条路径均被同一个 lowering 类型门禁拒绝。CUDA FP64 存储/回读本身通过，不能据此认为 FP64 kernel 运算也可用。

phymeo 当前用 NumPy FP64 作为数学参考、tyx FP32 作为设备实现。这是未来精度需求的缺口：若需要同精度的 CPU/设备对照、长时域或高条件数求解，当前无法直接选择 tyx FP64 算术。尚未证明特定 phymeo 数值问题能单靠 FP64 解决，也没有测量 FP64 速度或 Warp；不将这一项写成 Warp 性能差距。

## 环境与独立复现

- Python 3.12.14，NumPy 2.5.4，tyx-compute 0.1.0a0，Windows 11 build 26200。
- 本机 NVIDIA GeForce RTX 4060 Laptop GPU，驱动 566.07；代码无需导入 phymeo。
- [复现脚本](repros/fp64_compute.py)只使用公开 tyx API；自定义描述符是诊断探针，不表示 tyx 承诺支持任意 `ScalarType`。
- [原始 JSON](evidence/TYX-001-fp64-compute.json)保存固定 tyx 来源、输入、期望与每条执行路径的原始错误。

从仓库根目录、已安装固定 tyx 的环境执行：

```sh
python tyx_flaw/repros/fp64_compute.py --output artifacts/tyx-fp64-recheck.json
```

输入为 `16777217.0`，双精度精确期望 `33554434.0`，可用于以后检测悄悄降到 FP32 的实现。当前结果：

| 项目 | 结果 |
| --- | --- |
| `hasattr(tx, "f64")` | false |
| CUDA FP64 数组往返 | 逐元素一致 |
| interpret / cpu_cpp / cuda 加法 | 均在 lowering 阶段拒绝，尚未到后端编译/执行 |

错误核心信息：`numeric operation requires i32/f32 (f16/bf16 storage promotes to f32), got f64 and f64`。

## 已确认定位与后续建议

固定提交安装包中的位置：

- `tyx/types.py`：268–272 行的内置标量集合没有 f64；`ndarray()` 接收标量描述符，不代表算术支持该描述符。
- `tyx/lowering.py::_numeric_result_type`：1947 行起，数值集合限定为 i32/bf16/f16/f32；这就是本次最早、已复现的拒绝位置。
- `tyx/__init__.py`：公开类型导出没有 f64。
- `tyx/cuda_array.py::CudaArray`：独立验证了 float64 存储可往返；不要把修复范围限定为新增数组 dtype。
- phymeo 调用边界：`src/phymeo/backends/tyx.py`、`tyx_rigid.py` 与 `tyx_coupled.py` 目前主动要求 FP32；CPU/设备精度语义见 [NUM-002](../docs/legacy-issues/NUM-002-fp32-reference-semantics.md)。

给 tyx agent：先确定 FP64 是否进入正式类型合同；若纳入，沿类型提升/常量转换 → IR → 各后端算术及内建函数 → 数组/向量接口与自动微分检查完整链路。去掉 lowering 门禁本身不是完整修复。后端尚有哪些限制没有在本次探针中执行到，需继续排查，不作已定位结论。

## 修复验收

正式内置 f64 后，用该内置类型替换探针描述符，解释器、C++、CUDA 均得到精确结果；覆盖混合类型转换、极值、数学函数及 FP32 无回归。再增加 phymeo FP64 设备小场景，保持独立物理验证与当前 FP32 路径。性能须另用同精度、同质量门禁的稳态协议测量。
