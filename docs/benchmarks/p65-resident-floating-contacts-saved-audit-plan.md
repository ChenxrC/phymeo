# C10 保存审计：固定身份与纯作者计划

状态：**AUTHORING_ONLY / UNEXECUTED**。这是离线 observer，不运行生产者、物理、几何、M/J、独立参考或任何项目 import。父唯一串行执行，C9/C10 物理接受与本审计是独立状态。

依据父 `d75cdaf4be0ce751f6fd27b39b4d4982f4460689` 的 C10 合同末节；作者测试固定于 RD `d7bed1556fe630f5ace96bc7c251f4cd4e056fae`。原生 driver 使用本树 `5c959c8868e215e8eb6208173fa45882b9a658f0`，外部 wheel consumer 使用同一 RD 提交。未读取任何运行中的输出，也未按输出选身份。

## 预声明 membership

两个专项每 backend **76 device 节点、61 个 gzip**；独立 HOST 节点 **0**。F01 的 HOST patch/2 N/M 前置在该设备节点内，不能另加一次 HOST 通过。脚本 `membership()` 枚举完整原测试名、参数、pytest ID 和相对保存名。

| 测试族 | device 节点 | gzip | 固定内容 |
|---|---:|---:|---|
| F01 | 1 | 1 | root-only、HOST 前置、一次真实步 |
| F02 | 3 | 3 | child/motor/root，各 40 步 |
| F03 | 4 | 4 | 原序/反序/cycle/boost，各 40 步 |
| F04 | 1 | 1 | 非零角速、阻尼、真实弧，40 步 |
| F05 | 3 | 3 | COM 偏心、principal frame、cycle，40 步 |
| F06 | 1 | 1 | 实际 folded URDF 工具，40 步 |
| F07 | 6 | 6 | hull/cap side/end/miss/CC/非盒 HH，各 2 步 |
| F08 | 5 | 5 | N9/N16/body15/root+body16/共享祖先，40 步 |
| F09 | 1 | 1 | 连续载荷、分离、重入，40 步 |
| F10 W 矩阵 | 1 | 6 | W1×4/W4/W32，各 40+21 重放 |
| 容量两 bank | 4 | 4 | warm0/1 × world503/6 |
| delay/cap | 4 | 4 | delay0/1 × cap1/8或8，各4步 |
| 真实 late fault | 6 | 6 | upload/candidate/receipt × warm0/1 |
| restore stale | 2 | 2 | 两 bank、真实 candidate 后 restore |
| allocation/global cap | 1 | 1 | 实际 scratch 关闭和合法重试 |
| owned/foreign/close | 2 | 2 | F01/F03、重建完整一致 |
| old mixed | 1 | 1 | ordinary/affine/scalar/未绑定 floating，40+21 |
| pair filters | 6 | 6 | default/same-body/mask/excluded/parent 两类 |
| invalid graph/material/source | 11 | 0 | 原测试断言，不伪造记录 |
| builtin bool | 4 | 0 | int/NumPy bool/str/None |
| invalid floating body | 3 | 0 | (N,body)=(16,17)/(15,16)/(0,1) |
| old body bound | 2 | 0 | ordinary/affine，不虚构 body16 |
| huge-anchor source | 1 | 1 | 原输入、真实 broad、拒绝与 retained |
| candidate source | 3 | 3 | shape/group/world 原 packet、变化、重试 |

W4 显式 world 顺序 `(503,91,1207,6)` 分别 F01/F02-motor/F08-N9/F08-N16；W32 另 28 个 world 为 `10001+97*i`，输入按四例循环，仍保存每个 world 全行。反向注册不删除 owner/world 字段。所有步长原 `1/1024 s`；不重算物理。F07 只把独立首几何锚与“已上传 wrench 的独立响应”分别核，不能冒全窗几何 oracle。

## 审计输入与边界

CLI：`python -I -B SCRIPT --manifest ABS --output ABS_NEW_DIR`。manifest 固定 schema `c10-contact-saved-audit-v1`，声明 required_modes、required_external 和有序 runs。每 run 必须给原目录/config/head/raw_exit/完整 source_sha256；pytest 给 authored selected/completed/first failed 节点及 payload_roots、basetemp，首失败存在的 payload 子集另列。不同源码的通过范围独立列示；必要重复 completed 节点须明确此前 run 与修改原因，unique coverage 只计一次。输出只能为全新目录。

原 start/final、command、stdout/stderr、RAW_EXIT、before/after/source snapshots 全部绑定。所有源码 bytes（包括二进制 wheel）哈希保留，二进制不当 UTF-8 解码。payload_root 绑定原 CLI；外部 --output 可以在 runner 日志目录外，不能复制后改路径。枚举保存路径只检查固定 membership 的遗漏/多余，不决定应有集合。

逐 payload 检查完整原输入/模型/命令、N/B/D/Q、held/history、local_origin/world position、同次 diagnostics/M/J/full body wrench、actual geometry/sites/material/forces/work、真实 tick 和 capacity retain/retry。只比较已有独立 oracle 与实际字段，不求新动力学/接触。40+21 原字段完整相等；epoch/counters 独立单调，不删除物理字段作“规范化”。

保存缺口明确：20 个负控无 gzip；若 late fault、容量或 owned 测试只保存 before/retry、原断言中间 retained/不可写行为未落 JSON，本审计不能重新证明该瞬间，绑定同源 pytest 结果并注明。F07 没有独立滚动几何真值；巨大 anchor 负控没有正常物理轨迹。失败节点可能在保存前失败，保日志及真实前缀，不补 null 数据当完成。

Native 每端 F01/F03 各40，完整 raw contacts/shape/material/filter、FREE state/COM velocity、body_solve_origin/raw RNEA/decoded wrench、full M/J、独立轨迹、checks 与 pin source 均保存；零 matched 外力。失配和 observer/IO 失败保持原状态，不冒 native 通过。Wheel 两模式分别 W4 40+21、两个真实 bank capacity/retry、最多2 alignment 步，以及 F03 40；逐行完整物理记录、安装成员/loaded origins/关闭资源。F03 不额外声称有 replay/capacity。

资源上限：每 gzip 64 MiB 压缩、每 payload 256 MiB 解压、全部累计 2 GiB、8192 文件、JSON 深度96；metadata 单件16 MiB。只在内存有界读解压，不落解压副本，不删除证据。原文件 SHA/压缩与原 bytes 均记录。未执行 observer 不报告 pass；最终返回 audit coverage 与原生产者状态，不能替代物理验收。
