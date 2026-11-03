# 分页多世界碰撞查询

提交 fa71365 的 PagedWorldCollisionWorkspace 可以在一次 detect 调用中查询总形状数超过 256 的世界集合。每个世界完整放在同一页，按注册顺序依次装入。每页调用一次原 WorldCollisionWorkspace，CUDA 页内沿用真实 TYX 批归约与分类。全部页返回后，调用方才看到本次接受的世界结果。[实现](../../src/phymeo/geometry/paged_collision.py) · [事务与材料核验](../../tests/scale/test_paged_collision.py)

## 页、预算与返回值

CollisionPageLimits 默认将每页限制在 256 个形状、8,192 个运动顶点和 32,640 个世界内诊断对。完整结果的配对行预算独立，默认也是 32,640 行。只要输入前缀已经越过总行预算就停止读取，报告该前缀实际需要的行数。单个世界超过页限时，CollisionPageCapacityError 提供世界、资源、所需值及预算，在分配配对表之前拒绝。默认最多驻留两个工作区，设为一个即可流式调度。布局还可受基础宽相字节预算约束，超过总驻留预算时先关闭最久未使用的页。

有配对的页按六个真实 CUDA 数组计其基础宽相内存，包括形状偏移、标签、逐顶点包络、形状界、三列配对索引和分类标记。每个成功页提交后读取数组的实际 nbytes，并记录当前值和观测峰值。窄相按形状申请的临时存储、CUDA 上下文和保留的公开世界诊断不计入这六个数组。空页以及只有单形状无配对的页可在 HOST 生成所需边界，设备提交为零。

```python
from phymeo.geometry.paged_collision import CollisionPageLimits, PagedWorldCollisionWorkspace

limits = CollisionPageLimits(resident_pages=1, resident_broad_bytes=200_000)
with PagedWorldCollisionWorkspace(world_models, "cuda", limits=limits) as batch:
    results = batch.detect(motion_by_world, policy_by_world)
    pages = batch.layout
    counters = batch.stats()
```

| 单次调用 | 页内形状数 | 页内局部配对槽 | 两页的基础 GPU 字节 | 实际候选 / 命中 |
| --- | --- | --- | --- | --- |
| W4 × 65 稀疏球 | 195 / 65 | 6,240 / 2,080 | 114,664 / 38,224 | 0 / 0 |
| W4 × 72 面布加球 | 219 / 73 | 7,884 / 2,628 | 153,160 / 51,056 | 每帧 16 / 12 |
| W128 × 四球 | 256 / 256 | 384 / 384 | 25,604 / 25,604 | 基线 128 / 128，留出 127 / 127 |

前两组超过原单工作区的总形状上限，分别返回 8,320 和 10,512 行。W128 共返回 768 行。移动第二页中一个世界的第二个球，只消除该世界的接触。其余重叠球的间隙仍为 −1/16 m，两侧 x 见证为 1/8 和 1/16 m。测试另将两个空世界插入注册序列，核对完整顺序与局部 ID。W4 布组的三个原帧均从实际有限面和仿射材料重建接触。[G4 原资产与先前上限](p65-g4-world-geometry.md)

一个目标世界容量不足时，返回其实际所需候选数，保留该世界已接受的结果，其他页中的世界仍可接受。独立试验在第二页 CUDA 读回时注入故障。第一、二页的四份已接受观测全部保留，在失败处理后驻留页数为零，重试各重建一页。对照已存完整结果，只有指定负控世界发生改变。

## 冷暖查询成本

三种方式的每世界完整诊断哈希、候选、命中及材料观测全部相同，HOST 与 CUDA 也逐帧相同。人工方式以同一页布局建立两个单独的 WorldCollisionWorkspace，并按顺序调用后合并返回。每种方式新建工作区，执行一个冷帧，再测三个完整序列。下表为序列时长中位数，单位 ms。稀疏球序列为一帧，布面为三帧，W128 为基线和负控两次静态查询。单后端各以一个进程，按人工两页、驻留两页、流式一页的次序测量。物理子步数均为零。[复现入口](../../scripts/benchmark_paged_worlds.py)

| 场景 | HOST 人工 / 两页驻留 / 单页流式 | CUDA 人工 / 两页驻留 / 单页流式 |
| --- | ---: | ---: |
| W4 稀疏球 | 77.590 / 74.252 / 81.025 | 39.219 / 36.509 / 78.246 |
| W4 布面 | 311.221 / 316.518 / 353.390 | 292.987 / 316.074 / 457.562 |
| W128 四球 | 86.843 / 77.871 / 82.958 | 69.121 / 69.947 / 144.358 |

W4 布面在 CUDA 上的注册耗时分别为 5.403、0.136、0.135 ms，冷首帧分别为 224.207、141.330、138.744 ms。人工方式在注册阶段创建叶工作区，分页入口在首次查询时创建。两页驻留在三个暖帧中记录 33 次底层核、1,266,720 B H2D、190,176 B D2H，实际基础宽相观测峰值为 204,216 B。流式一页的三帧同为 33 次核、190,176 B D2H，H2D 升至 1,669,704 B，观测峰值降至 153,160 B。从冷帧到第三个暖序列，两页驻留建页 2 次、逐出 0 次，单页流式建页 20 次、逐出 19 次。

本机证据为 `artifacts/p65-night-paged-world-collision-20261104-01/`。成本报告在 `paged-cost-host/report.json` 和 `paged-cost-cuda/report.json`，真实设备故障前后记录在 `paged-cuda/pytest-temp/` 内的 `page-fault-retry.json`。专项共通过 7 项 HOST/准备或零提交检查、6 项真实 CUDA 检查。此批量测用于比较相同输出下的驻留成本。

```powershell
python scripts/benchmark_paged_worlds.py --backend cuda --samples 3 --cache-root artifacts/paged-cache --out artifacts/paged-cost.json
```

## 只消费接触与诊断的分页调用

给 `PagedWorldCollisionWorkspace` 指定 `result_format="contacts"`，每个世界返回完整的接触见证和需要处理的诊断。用 `detect(..., limits_by_world={world: ContactBufferLimits(...)})` 可以分别限定单个世界的接触数与诊断数。注册时的页大小与候选对预算仍然有效；原完整行输出保留为默认模式。[紧凑输出口径](p65-compact-world-contacts.md)

同批放入空世界、地形与球、穿面球、两张变形布面以及三个彼此分离的球，令一时刻只驻留一页。HOST 和真实 CUDA 均走四页，逐出三次，保留了 RR、RD、DD 三个接触，最后一页跳过三个已分离行。CUDA 完成九次启动。RD 的三元材料地址、DD 在归一化 1/4 时刻的四个边节点及双方权重与独立单世界完全匹配。将 DD 的接触容量降为零只拒绝该世界，明确报告需要一条记录。另在最后一页注入异常，整个批次继续保留前一已接受发布，重试可逐项复现。[混合分页用例](../../tests/scale/test_paged_compact_contacts.py)
