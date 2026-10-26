"""Generate offline, editable architecture assets; PNG is captured in a browser."""

from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "architecture"


def box(x, y, width, height, title, lines, color="#34d399", tag="", future=False):
    dash = ' stroke-dasharray="6 4"' if future else ""
    text = [
        f'<g data-box="{escape(title)}">',
        f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="6" fill="#0f172a"/>',
        f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="6" fill="{color}" fill-opacity="0.065" stroke="{color}" stroke-width="1.5"{dash}/>',
        f'<text x="{x+20}" y="{y+29}" class="name">{escape(title)}</text>',
    ]
    for index, line in enumerate(lines):
        text.append(f'<text x="{x+20}" y="{y+53+index*19}" class="sub">{escape(line)}</text>')
    if tag:
        text.append(f'<text x="{x+width-18}" y="{y+27}" text-anchor="end" class="tag" fill="{color}">{escape(tag)}</text>')
    text.append("</g>")
    return "\n".join(text)


def generate():
    svg = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1160 1160" role="img" aria-labelledby="diagram-title diagram-description">
<title id="diagram-title">phymeo 物理与学习协同架构及两人模块责任</title>
<desc id="diagram-description">场景进入物理契约，步进协调数学与学习求解，验收后提交状态，失败恢复快照。tyx 负责计算，PyTorch 负责学习。A 负责基础与刚体，B 负责柔体与学习。所有框均为目标设计，实际实现看进度。</desc>
<defs>
  <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse"><path d="M 40 0 L 0 0 0 40" fill="none" stroke="#1e293b" stroke-width="0.5"/></pattern>
  <marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8Z" fill="#64748b"/></marker>
  <marker id="retry" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8Z" fill="#fb7185"/></marker>
  <style>text{font-family:"JetBrains Mono","Microsoft YaHei","Noto Sans CJK SC",sans-serif}.name{fill:#f1f5f9;font-size:17px;font-weight:600}.sub{fill:#a9b8ce;font-size:12px}.tag{font-size:11px;font-weight:700}.label{fill:#94a3b8;font-size:11px}.edge{fill:none;stroke:#64748b;stroke-width:1.7;marker-end:url(#arrow)}</style>
</defs>
<rect width="1160" height="1160" fill="#020617"/>
<rect width="1160" height="1160" fill="url(#grid)"/>
<!-- Directed data flow; render behind opaque components. -->
<path class="edge" d="M580 105V130H220V160"/>
<path class="edge" d="M580 105V160"/>
<path class="edge" d="M580 130H940V160"/>
<path class="edge" d="M580 240V315"/>
<path class="edge" d="M220 240V280H160V315"/>
<path class="edge" d="M940 240V280H1000V315"/>
<path class="edge" d="M260 355H300"/>
<path class="edge" d="M900 355H860"/>
<path class="edge" d="M580 395V445H210V500"/>
<path class="edge" d="M580 445V500"/>
<path class="edge" d="M580 445H950V500"/>
<path class="edge" d="M210 620V680"/>
<path class="edge" d="M580 620V680"/>
<path class="edge" d="M950 620V680"/>
<path class="edge" d="M580 750V780H220V810"/>
<path class="edge" d="M380 850H420"/>
<path class="edge" d="M740 850H780"/>
<path d="M60 715H24V286H480V315" fill="none" stroke="#fb7185" stroke-width="1.5" stroke-dasharray="6 5" marker-end="url(#retry)"/>
<!-- Architecture boundaries. -->
<rect x="44" y="142" width="1072" height="116" rx="12" fill="none" stroke="#334155" stroke-dasharray="5 4"/>
<rect x="44" y="474" width="1072" height="164" rx="12" fill="none" stroke="#334155" stroke-dasharray="5 4"/>
<text x="60" y="464" class="label">可组合求解模块 / 同一物理状态与作用量合同</text>
<text x="620" y="276" class="label">编译后的数据、能力与状态快照</text>
<text x="48" y="277" class="tag" fill="#fb7185">拒绝 → 恢复快照 → 数学重算</text>
<text x="620" y="770" class="tag" fill="#34d399">仅接受的候选进入下一步</text>
'''
    svg += box(60, 35, 1040, 70, "场景与输入", ["机器人 / 刚体 / 布料 / 柔性体 · 几何、材料、控制与边界条件"], "#22d3ee", "A + B")
    svg += box(60, 160, 320, 80, "物理模型", ["SceneSpec → CompiledScene"], "#22d3ee", "A")
    svg += box(420, 160, 320, 80, "状态与快照", ["运动 + 材料历史 + 学习隐状态"], "#a78bfa", "A")
    svg += box(780, 160, 320, 80, "算子与能力契约", ["单位 / 读写 / 梯度 / 支持范围"], "#22d3ee", "A·B 审")
    svg += box(60, 315, 200, 80, "碰撞与接触", ["几何、流形、响应所有权"], "#34d399", "A")
    svg += box(300, 315, 560, 80, "步进与耦合调度", ["对象所有权 · 子步 · 作用力交换 · 有界重试"], "#fb923c", "A·B 审")
    svg += box(900, 315, 200, 80, "执行计划", ["能力匹配 / 缓冲区预算"], "#fb923c", "A")
    svg += box(60, 500, 300, 120, "数学 / 刚体与机器人", ["首批：积分、平面/球体基线", "后续：旋转、关节、广义坐标", "解析参考与数值误差验收"], "#34d399", "A")
    svg += box(430, 500, 300, 120, "数学 / 布料与柔性体", ["首批：XPBD 距离、链与布片", "后续：弯曲、本构、隐式 FEM", "双向耦合端口与材料测试"], "#34d399", "B")
    svg += box(800, 500, 300, 120, "学习模型 / AI 求解", ["优先：warm start / 材料响应", "后续：残差修正、局部代理", "训练域声明与数学回退"], "#a78bfa", "B·后续", True)
    svg += box(60, 680, 1040, 70, "耦合残差与候选验收", ["有限性 / 约束误差 / 穿透 / 材料适用域 · 数学回退仍可能失败，必须报告"], "#fb7185", "A + B")
    svg += box(60, 810, 320, 80, "提交状态", ["唯一写入者 · 不修改输入状态"], "#34d399", "A")
    svg += box(420, 810, 320, 80, "观测与验证轨迹", ["配置 / 种子 / 误差 / 两仓 SHA"], "#a78bfa", "A + B")
    svg += box(780, 810, 320, 80, "离线训练与评估", ["独立数据划分 · 不在线静默更新"], "#a78bfa", "B·后续", True)
    svg += '<text x="60" y="943" class="label">计算依赖边界 / 下方是基础设施，不是额外的物理时间步</text>'
    svg += box(60, 960, 650, 100, "tyx / 通用计算底座", ["公共 kernel、数组、CPU/CUDA、自动微分、稀疏线代、流与图", "phymeo → tyx 单向依赖；GPU 与梯度按真实执行能力验收"], "#fbbf24", "A 适配")
    svg += box(770, 960, 330, 100, "PyTorch / 可选学习运行时", ["网络、训练与模型分发", "tyx.torch：共享存储、流与梯度边界"], "#94a3b8", "B")
    svg += '<path class="edge" d="M710 1010H770"/><path class="edge" d="M770 1030H710"/>'
    svg += '''<text x="60" y="1103" class="label">实线框：首期或基础模块方向　虚线框：后续学习功能　A / B：开发责任，不代表部署进程</text>
<text x="60" y="1127" class="label">本图为目标设计。已实现范围、CPU/GPU 验证与剩余任务以 docs/progress.md 为准。</text>
</svg>'''
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "phymeo.svg").write_text(svg + "\n", encoding="utf-8")
    html = '''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>phymeo · Physics × Learning Architecture</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#020617;color:#f1f5f9;font-family:"JetBrains Mono","Microsoft YaHei",sans-serif;padding:48px 32px}.container{max-width:1240px;margin:auto}.eyebrow{color:#22d3ee;font-size:12px;letter-spacing:3px}.header{display:flex;justify-content:space-between;align-items:flex-start;gap:24px;margin:18px 0 30px}h1{font-size:48px;letter-spacing:-2px;line-height:1.05;margin:0 0 16px}h1 span{color:#64748b;font-size:30px;letter-spacing:-1px}.subtitle{color:#a9b8ce;font-size:14px;line-height:1.9;margin:0;max-width:830px}.badge{border:1px solid #334155;border-radius:6px;padding:10px 14px;white-space:nowrap;color:#cbd5e1;font-size:11px;line-height:1.8}.diagram-container{border:1px solid #1e293b;border-radius:16px;padding:16px;background:#0b1222;overflow-x:auto}svg{display:block;width:100%;min-width:920px}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-top:22px}.card{padding:22px;border:1px solid #1e293b;border-radius:12px;background:#0b1222}.card h2{font-size:15px;margin:0 0 12px}.card p{font-size:12px;color:#a9b8ce;line-height:1.9;margin:0}.card:nth-child(1) h2{color:#34d399}.card:nth-child(2) h2{color:#a78bfa}.card:nth-child(3) h2{color:#fbbf24}.footer{font-size:11px;line-height:1.9;color:#94a3b8;margin-top:22px;display:flex;justify-content:space-between;gap:20px}a{color:#7dd3fc;text-decoration:none}@media(max-width:760px){body{padding:24px 16px}h1{font-size:36px}h1 span{display:block;font-size:21px;margin-top:8px}.header{display:block}.badge{display:inline-block;margin-top:18px}.cards{grid-template-columns:1fr}.footer{display:block}.diagram-container{padding:8px}}@media print{body{padding:10px;-webkit-print-color-adjust:exact;print-color-adjust:exact}.container{max-width:none}svg{min-width:0}.cards{break-inside:avoid}.diagram-container{padding:0}}
</style></head><body><main class="container">
<div class="eyebrow">PHY MEO / SYSTEM DESIGN 01</div>
<header class="header"><div><h1>phymeo <span>Physics × Learning</span></h1><p class="subtitle">以物理契约连接数学求解与学习模型。先完成刚体与柔体基线，再扩展双向耦合和 AI 加速。<br>独立实现 · tyx 计算底座 · 两人并行开发 · 可验证的状态提交</p></div><div class="badge">ARCHITECTURE v0<br>2026.10.27 · 设计基线</div></header>
<section class="diagram-container" aria-label="架构图，可横向滚动">''' + svg + '''</section>
<section class="cards" aria-label="架构约束">
<article class="card"><h2>A / 基础与刚体</h2><p>拥有 core、tyx adapter、刚体与接触基础、步进和 CI。B 审阅共享协议，并以柔体场景验收接口。</p></article>
<article class="card"><h2>B / 柔体与学习</h2><p>拥有 XPBD、布料/材料、柔体端口与学习 provider。AI 首先预测初值，保持数学终止容差不变。</p></article>
<article class="card"><h2>质量门槛 / 每个里程碑</h2><p>解析参考 → CPU 合同 → 实际 GPU 对照。候选失败从原状态重算；不把 skip 当通过，不把图中模块当已实现。</p></article>
</section><footer class="footer"><span>来源：Newton 架构/论文调研 · tyx 99f6df7c · 原创模块组合与接口设计<br>模板样式改编自 Cocoon AI / MIT；许可见同目录 template-LICENSE.txt。</span><span><a href="overview.md">架构说明</a> · <a href="../development/team-and-tests.md">分工与测试</a><br><a href="../development/environment-and-ci.md">环境与 CI</a> · <a href="../progress.md">实际进度</a></span></footer>
</main></body></html>'''
    (OUT / "phymeo.html").write_text(html + "\n", encoding="utf-8")
    print("Generated docs/architecture/phymeo.html and phymeo.svg")


if __name__ == "__main__":
    generate()
