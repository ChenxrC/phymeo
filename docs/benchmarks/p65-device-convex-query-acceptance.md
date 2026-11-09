# K0 通用设备凸体查询：本机限域验收（2026-11-09）

状态：**ACCEPTED_K0_DISTANCE_WITNESS**。默认 `site_capacity=None` 的当前位姿距离、相交状态与双侧有限材料见证已完成最终源码验证。首域为 ordinary fixed-root tree/forest 与 static 来源，球/胶囊/OBB/凸包十种组合；无 HOST 几何补算。K1 完整接触集合、K2 受载消费、affine/floating 查询来源、CCD 与持续摩擦仍未验收。

## 本次收尾

- 唯一项目执行 owner 为父 `/root`，每条命令经 entrypoint-aware 串行 guard。
- 原最终修正后的完整两文件：interpret **142 passed / 52.32 s**，CUDA **142 passed / 213.71 s**，均 RAW_EXIT=0，无跳过。
- 此次使用工作区 artifacts 内绝对 basetemp，保留 **284 份完整 gzip**。覆盖执行前预声明的每端 142 身份，含 W32、预检、故障、恢复与资源关闭。
- 十种 touch 每端实际为 **9 RESOLVED/CONTACT_BAND/NEAR_ZERO，1 UNKNOWN/CONTACT_BAND/NORMAL_NONUNIQUE**；允许带内未决，不声称全部唯一法向，也不将 UNKNOWN 当分离。
- 原 8 次 pinned Newton 1.7.0 短参照复用，范围只为四个首批非零正控。原 margin/cutoff 与误差限定仍见 [原执行记录](../superpowers/plans/2026-11-08-device-convex-query.md#12-父执行范围2026-11-08未验收)。
- 最终源码新 wheel 构建成功，源码外 `python -I -B` 实际 interpret/CUDA 各 **4 次接受、0 次拒绝**。这 8 次是查询正控，不是受载物理步。
- `tests/scale/k0_final_saved_audit.py` 只读核 **284 份完整记录**、精确身份、保存源码、全部数组字节长度、执行前后/当前源码，以及 **440 个 source/wheel/installed 产品成员**一致；RAW_EXIT=0。没有重新运行 producer 或 verifier。

证据目录：`artifacts/p65-k0-final-20261109-01/`；主收据 `retained-audit.json`。旧缺失 gzip 的历史记录没有补写或倒灌，本轮是最终集成源码的必要留档验收。

## 修复与原失败

首 Ruff 有 4 处 import 顺序和参考模块 2 个歧义变量名；已最小修复并格式化，最终 9 文件 Ruff/format 均通过。第一测试启动误用未安装 TYX 的 `.venv-py311`，在 fixture 初始化报错、无查询；原 RAW1 保留。改用此前已验的 `.venv` 后两端完整通过。一次自动审批连接失败没有执行命令，随后同一审批路径重试成功。

源码外解压脚本首次在 PowerShell 参数 `Select-Object -Single` 处、解压前失败，纠正后使用原空目录。wheel-interpret runner 在源码外查询 Git HEAD 报错，产品调用成功且完整源码 hash 已核；wheel-cuda 显式传入 source_head。原收据保留，未为元数据重跑查询。

## 当前范围和后继

距离成功不等于完整接触斑。当前候选仍有 HOST broadphase 读回与 pair 上传，并如实计费。W32 功能/资源已验，不作端到端加速声明。来源丢精度、深包含、多法向及资源耗尽保留明确分类。

工作区已有可选 `site_capacity`、patch 核和 K1 测试的作者内容；本次默认查询不启用它们。它们的源码存在和 wheel 字节一致不能当作 K1 验收。

用户最新优先级见 [大规模训练与 G1 舞蹈接入计划](../superpowers/plans/2026-11-09-training-humanoid-priority.md)。K0 本次收束后，优先补训练向量接口、策略驱动与人形所需的关节/接触能力。
