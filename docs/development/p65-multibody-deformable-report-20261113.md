# M-B1：多独立刚体与变形表面的共同响应

状态：SOURCE_AND_INSTALLED_QUALIFIED（父安装与容量更新见末节）。2026-11-13；父唯一串行执行，本报告仅读取已结束 stdout/final，不重跑或重审保存载荷。

[实施合同](p65-multibody-deformable-implementation-20261113.md)限定的公共装配、逐体接触载荷与共同发布已通过两后端源码验证。安装消费者已交接，等待父 wheel04 的源码外验证；本报告不将源码运行当作安装通过。

## 功能与边界

同一 logical world 的多个 ordinary floating 单 body participant 共同作用于一个原 D1 变形节点模型。sphere/capsule 的原稳定 shape/facet/point 地址，经真实设备 FK 和节点来源产生 entry-sampled NormalLayer 法向力；barycentric 节点载荷与每个 participant 的 COM wrench 进入原材料与刚体 J/M 推进，最后一次共同发布。没有新增求解器或改变层定律。

总 participants <=8、worlds<=8，每 world 至少一个 participant；原节点、面、site<=128 和内存限额保持。静态边界实际覆盖 8 participants 准入、9 拒绝；本批受载连续案例最大为 2 worlds / 4 participants，不把容量声明称为最大规模物理验证。

真正 multibody receipt 使用 `tyx-deformable-multibody-contact-step-v1`，完整列出各 participant 的 body slot、原/投递 wrench、外载入步功率及共同 candidate。零接触 participant 仍推进，多个 collider 可累加到同一真实 body。旧单体 schema/核/字段保持。

不包括刚体互碰、关节树、切向摩擦、CCD、长期稳态支撑、任意机器人或性能超过 Newton；M-B2 另列。既有 D4/D6-B 稳态 FAIL 不因本批短窗通过而改变。

## 固定案例与独立观测

所有新物理输入与工程门在运行前登记。原四点片、两面、五条 Distance 行、固定节点、`h=1/1024`、width=1/16、k=128、接近阻尼0.5 s/m保持。worlds=7/31 使用不同非零 anchor；原局部 ID 重用不串 world。

- A：mass2/mass1 两个球以不同真实入步速度加载同一片的不同位置，4 步；第二身份给第一 body 增加不同材料地址的第二 collider。逐体非零力、偏心 COM 力矩和实际运动均检查。
- B：两 world、sphere + 斜 capsule、16 个无 reset 连续步。world7 sphere 与 world31 capsule 分别真实入层再离层；另一个 body 保持加载或零接触而继续运动。mass1 capsule 的 +32 N 与 mass2 sphere 的 +64 N 同为32 m/s²；初稿 +64 N capsule 仅在执行前手推发现事件前提不成立后修正，从未运行，运行后未调输入。
- E：world/participant/collider 注册顺序置换、完整 source 与 body 槽、容量、晚拒绝、真实 retry、checkpoint/restore、control 后 query 失效、整 world partial reset 后实际后继。

消费者自带原始几何字面量、有限 segment–triangle Fraction 候选 oracle、独立 FP64 ordered 材料递推及逐体连续刚体参考，不导入仓库 tests/examples 作为安装依赖。每次实际 step 返回先保存完整 receipt，再做独立断言。观测包括：全部 site、法向/间隙/材料见证、法则力、节点散射、逐体反力/COM矩、完整 M/J/bias/加速度、真实 q/v、外载功率、逐行材料冲量和固定反力、同次功/势能/耗散，以及明确施力时刻的离散角动量关系。

沿原 D5 门：力学 abs5e-6/rel5e-5；几何 distance1e-4 L、witness2e-4 L、normal2e-3；同入步材料状态2e-5；独立连续状态2e-4。未修改原 gate 或容量。

## 28 个唯一源码节点

以下 TYX 参数身份均分别运行 interpret/CUDA，共24节点；4个纯 HOST 身份只在解释作业中运行一次。

| 分组 | 每后端节点 | 实际资格 |
|---|---:|---|
| A 两 body / 同 body 两 collider | 2 | 连续力、COM矩、双方运动和独立参考 |
| B sphere/capsule 分时入层离层 | 1 | 双稀疏 world、真实速度、零接触 body 后继 |
| world/participant/shape 顺序置换 | 1 | 按稳定 key 比较完整状态/历史 |
| checkpoint/query/control/partial reset | 1 | 实际恢复后继、未选 world 保持 |
| late query / 完整 publication / 错 body slot | 3 | 双域 accepted 状态、banks、ticks、history 和旧 receipt 不变；恢复真实调用后 retry 对独立 owner |
| active site 恰满/少一 | 1 | 4 接受、3 原子拒绝 |
| query + response native budget | 1 | 完整原字节式 exact/one-less，不冒 RSS 峰值 |
| 实际新 transpose 核 | 1 | 合法多 site 累加、跨 world/越界 body slot status0、全零清零 |
| 原 A 场景全部 laws=None | 1 | 两真实步；每 body wrench0、完整 source/tick 和节点后继 |
| HOST layout | 1（仅一次） | 顺序、只读 offsets、alias/重复、8/9 容量 |
| HOST installation negatives | 3（仅一次） | wrong-world / wrong-body / missing-source 在 response allocation 前拒绝 |

晚失败注入是上述真实 query 返回后、全部 candidate/publication 准备后、或真实转置的错误 body 地址；没有声称分别向每个 participant 的独立求解器注入故障。laws=None 验的是零响应 sites，完整实际 query 仍保存，不能称为 empty query。

## 已结束执行与源码版本

证据根：[p65-feature-until-0730-20261113-01](../../artifacts/p65-feature-until-0730-20261113-01)。各作业的原 stdout、RAW_EXIT、final 及 source snapshots 保留。

| 作业 | 实际结果 | 范围 |
|---|---|---|
| 186-mb1-first-interpret | 16 passed /32.81 s /RAW_EXIT0 | 12 TYX interpret +4 HOST |
| 187-mb1-cuda | 12 passed /27.25 s /RAW_EXIT0 | 全部12 CUDA身份；不重复4 HOST |
| 171-mb1-singlebody-compat-interpret | 8 passed /16.72 s /RAW_EXIT0 | 旧单体公共装配/事务/资源/HOST禁用/D6保留 |
| 172-mb1-singlebody-compat-cuda | 8 passed /15.31 s /RAW_EXIT0 | 同一必要旧接缝 CUDA |

186/187 的 tested HEAD 为 `23757ae6a0d2a5ed3c6b15b0ec762bfd571e3b75`。产品集成提交为 `c2f03f15`（layout/kernel）、`ba2b9872`（全部 participant 的来源/tick）、`0cbb9e27`（response/compiler）。原五 product source 在新两组执行的 final 中保留各自 before/after 绑定；本报告未重新运行 hash 审计。

原静态失败不抹去：173 Ruff0；174 combined tests format1（2 files）；175排版0；176 Ruff0/177 format0。179 combined consumer Ruff1（1项 import排序）；180 format1（1 file）；181 import修复0、182 format0、183 Ruff0、184 format0。这些是在功能执行前的排版/import修复；不称首轮所有静态都绿，不倒灌到原草稿。

作者 tests-first `1b52f49f`，边界补充 `2422ee4f`，独立 consumer `5778c6fb`；cleanup `7e0574f0` 导入父 `47958c30` 后仅 style 至 tested `23757ae6`。cleanup 保存 primary + close 双失败、每 owner 实际 close 状态、强持有未知释放 owner，不修改公共 `__exit__`、不重复不安全 free。

## 安装与后继

[独立安装消费者](../../tests/installed/check_deformable_multibody.py) 预登记每后端 A4步 + B双world16步 + 双world生命周期4步，共44 logical worldsteps /88 participant steps；完整来源、raw query/receipt、模块安装根与关闭收据均为包外必验项。本段为作者交接时的预登记；父已完成的安装与容量结果见末节。

本片支持多独立单体与柔性表面的共同法向响应。后继关节树参与者、刚体互碰或持续支撑须各自建立多案例与真实执行证据，不从本报告推定完成。

## 父正式安装与容量接受（2026-11-13）

状态更新：**SOURCE_AND_INSTALLED_QUALIFIED**。第四包产品 `23757ae6a0d2a5ed3c6b15b0ec762bfd571e3b75`；190核534产品成员 source/wheel/installed同字节，issues空。wheel1726331B，SHA256 `859825776bfd14467aa6e4f6ff2ecbaa6430a3a947f7029d16352eed9f06e3dd`。

191 interpret /192 CUDA均RAW_EXIT0、PASS；每后端实际44 worldsteps /88 participant steps，全部owner关闭成功、无retained cleanup failure。每条完整trace166记录；原字节SHA分别545d86df08cf4eaa77bbb0848a095e13e8b938fd624bb1e2a9803981b3d37eed、da2d5c7f871afb99c0b6ff0652368393f407c010e79baaf0a26e129d28f82e93。模块来源全部在独立安装site。

容量后继只增加测试，产品未改：S8为4world/8participants，M3为2world且分别1/2participants。只重标原A的world/安装身份，原几何、质量、初态、控制、h及所有门保持；199 interpret2passed8.47s、200 CUDA2passed6.69s。201/202源码外各实际24worldsteps/44participantsteps、PASS、全部关闭成功。S8授予最大8participants的这组四步物理资格，不是8world全占用、更多模型或吞吐资格。M3验证同一multibody布局中单体world也使用完整逐participant映射。

本片及容量后继合计每后端安装68worldsteps/132participantsteps，四份独立运行的范围仍分列。所有实际返回、完整trace、原静态失败、源码绑定及安装provenance保留。新canonical归档为 `artifacts/p65-feature-until-0730-20261113-01/installed-closed-archive-04/wheel-04`，33文件4904941B；原件保留，归档无解压/物理重跑。归档receipt SHA256 `8295f7d5c8d0748fceea247740410946b92969789d7b64dbb15588da22232f30`。
