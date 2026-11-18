"""Offline B3 evidence viewer; no Phymeo import, solver, or trajectory repair.

Only the two saved B3 schemas are consumed. Failed attempts remain separate.
Rendering can decimate real samples/frames to meet the 1 MiB HTML budget; all
load summaries use every saved sample. The original gzip remains the evidence.
"""

import argparse
import gzip
import hashlib
import itertools
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

MAX_HTML_BYTES = 1024 * 1024
MAX_JSON_BYTES = 512 * 1024 * 1024
WINDOW_SCHEMA = "mujoco-area-friction-window-v1"
CONTINUOUS_SCHEMA = "mujoco-area-friction-continuous-test-v1"
LABELS = {"center-drop": "冲击", "offset-support": "偏心旋转", "dual-patch": "双接触区"}
EVENTS = {
    "impact": "接触建立",
    "force_unload": "载荷卸除",
    "geometric_separation": "几何分离",
    "recontact": "再次接触",
    "dual_to_single": "双区转单区",
    "any_empty": "空接触面",
}


def _decode(value):
    """Decode complete saved _plain arrays, including scalars and boolean fields."""
    if isinstance(value, list):
        return [_decode(item) for item in value]
    if not isinstance(value, dict):
        return value
    if set(value) == {"dtype", "shape", "hex"}:
        dtype = np.dtype(value["dtype"])
        shape = value["shape"]
        if dtype.hasobject or dtype.kind not in "biufc" or not isinstance(shape, list):
            raise ValueError("unsupported evidence array dtype/shape")
        if len(shape) > 8 or any(type(n) is not int or n < 0 for n in shape):
            raise ValueError("invalid evidence array extent")
        size = math.prod(shape) * dtype.itemsize
        if size > MAX_JSON_BYTES or len(value["hex"]) != 2 * size:
            raise ValueError("evidence array byte count mismatch")
        return np.frombuffer(bytes.fromhex(value["hex"]), dtype=dtype).reshape(shape)
    return {key: _decode(item) for key, item in value.items()}


def _read(path):
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rb") as stream:
        raw = stream.read(MAX_JSON_BYTES + 1)
    if len(raw) > MAX_JSON_BYTES:
        raise ValueError(f"evidence JSON exceeds 512 MiB: {path}")
    return _decode(json.loads(raw))


def _array(value, shape=None):
    data = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(data)) or (shape is not None and data.shape != shape):
        raise ValueError("nonfinite or malformed plotted evidence")
    return data


def _small(value):
    """Display-only seven-significant-digit values; no acceptance decisions here."""
    if isinstance(value, np.ndarray):
        return _small(value.tolist())
    if isinstance(value, (list, tuple)):
        return [_small(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _small(item) for key, item in value.items()}
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(value):
            raise ValueError("nonfinite display value")
        return float(f"{value:.7g}")
    if isinstance(value, np.integer):
        return int(value)
    return value


def _rotation(q):
    w, x, y, z = _array(q, (4,))
    if abs(w * w + x * x + y * y + z * z - 1) > 1e-4:
        raise ValueError("saved FREE quaternion is not unit length")
    # Native WXYZ, body to world. Do not normalize or integrate the saved pose.
    return np.array(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        )
    )


def _wire(mesh, pressure):
    vertices = _array(mesh["world_vertices"])
    if pressure:
        faces = Counter(
            tuple(sorted(face))
            for cell in mesh["cells"]
            for face in itertools.combinations(map(int, cell), 3)
        )
        boundary = [face for face, count in faces.items() if count == 1]
    else:
        boundary = [tuple(map(int, face)) for face in mesh["faces"]]
    edges = sorted(
        {tuple(sorted(edge)) for face in boundary for edge in itertools.combinations(face, 2)}
    )
    used = sorted({index for edge in edges for index in edge})
    if not used or used[-1] >= len(vertices) or used[0] < 0:
        raise ValueError("invalid saved asset boundary")
    remap = {old: new for new, old in enumerate(used)}
    return vertices[used], [[remap[a], remap[b]] for a, b in edges]


def _items(payload):
    if payload["schema"] == WINDOW_SCHEMA:
        key = str(payload["world"]["world"])
        yield key, payload["window"], payload["world"]["model_key"], payload["initial"]
    else:
        specs = {str(world["world"]): world for world in payload["worlds"]}
        for index, world in enumerate(payload["world_order"]):
            key = str(world)
            yield (
                key,
                "sliding" if index == 0 else "unloading",
                specs[key]["model_key"],
                payload["initial"][key],
            )


def _record(payload, record, key):
    if payload["schema"] == WINDOW_SCHEMA:
        if not record.get("state_accepted") or "after" not in record:
            return None
        return record["before"], record["after"], record.get("geometry", ()), record.get("arrays")
    after = record.get("after", record.get("publication", {}).get("states", {}))
    if key not in after:
        return None
    return record["before"][key], after[key], record.get("geometry", {}).get(key, ()), None


def _frame(record, unpacked, assets, initial_q):
    before, after, geometry, arrays = unpacked
    receipt = after["last_step"]
    rows = receipt["raw_contact_fields"]["area_friction"]
    if len(rows) != len(assets):
        raise ValueError("saved asset/pair count mismatch")
    positions = _array(receipt["dynamics"]["body_position"])
    rotations = _array(receipt["dynamics"]["body_rotation"])
    q = _array(after["q"])
    free = q.shape == (7,)
    bodies, definitions, seen = [], [], set()
    samples, regions = [], []
    total = np.zeros(3)
    fn_sum = ft_norm_sum = 0.0
    for pair_index, (row, asset) in enumerate(zip(rows, assets, strict=True)):
        if row["identity"] != asset["identity"]:
            raise ValueError("saved asset identity mismatch")
        for side, kind in ((0, "pressure"), (1, "rigid")):
            body = int(row["body_ids"][side])
            if body in seen:
                continue
            seen.add(body)
            local, edges = _wire(asset[f"{kind}_local"], kind == "pressure")
            entry = local @ rotations[body].T + positions[body]
            endpoint = None
            if arrays is not None:
                # These window payloads contain exactly one world per workspace.
                endpoint = (
                    local @ _array(arrays["body_rotation"])[0, body].T
                    + _array(arrays["body_position"])[0, body]
                )
            elif free and side == 0:
                endpoint = local @ _rotation(q[3:]).T + q[:3]
            bodies.append([entry.tolist(), None if endpoint is None else endpoint.tolist()])
            definitions.append(
                {
                    "name": asset["identity"]["compliant_body" if side == 0 else "rigid_body"],
                    "kind": kind,
                    "edges": edges,
                }
            )
        points = _array(row["points"])
        count = len(points)
        normals = _array(row["normals"], (count, 3))
        fn = _array(row["normal_force"], (count,))
        ft = _array(row["tangential_force"], (count, 3))
        if np.any(fn < 0):
            raise ValueError("negative saved normal force")
        # Display the actually published total, not a newly evaluated force law.
        total += np.sum(_array(row["force_on_a"], (count, 3)), axis=0)
        fn_sum += float(np.sum(fn))
        ft_norm_sum += float(np.sum(np.linalg.norm(ft, axis=1)))
        for point, normal, normal_force, tangent in zip(points, normals, fn, ft, strict=True):
            samples.append([*point, *(normal * normal_force), *tangent, pair_index])
        if pair_index < len(geometry):
            regions.extend(
                _array(triangle["vertices"], (3, 3)).tolist()
                for triangle in geometry[pair_index]["surface"]["triangles"]
            )
    axes = 3 if free else 2
    displacement = (q[:axes] - initial_q[:axes]) * 1000
    return _small(
        {
            "step": int(record["step"]),
            "ti": float(before["time"]),
            "to": float(after["time"]),
            "q": q,
            "b": bodies,
            "s": samples,
            "r": regions,
            "n": len(samples),
            "nr": len(regions),
            "F": total,
            "Fn": fn_sum,
            "Ft": ft_norm_sum,
            "dq": displacement,
            "observation": bool(record.get("observation_complete", False)),
        }
    ), definitions


def _case(payload, path, root, key, name, model, initial, summary):
    free = len(initial["q"]) == 7
    label = LABELS.get(name, "滑动卸载 · " + ("切向滑动" if name == "sliding" else "上离卸载"))
    frames, definitions = [], []
    for record in payload["records"]:
        unpacked = _record(payload, record, key)
        if unpacked is not None:
            frame, current_definitions = _frame(
                record, unpacked, payload["assets"][model], _array(initial["q"])
            )
            if definitions and definitions != current_definitions:
                raise ValueError("body mesh topology changed inside saved case")
            definitions = current_definitions
            frames.append(frame)
    requested = int(payload.get("declared_steps", payload.get("steps_requested", 0)))
    accepted = int(payload["accepted_steps"])
    if accepted != len(frames):
        raise ValueError(f"accepted prefix is missing a state: {path}, world {key}")
    coverage = {EVENTS[k]: str(v) for k, v in payload.get("coverage", {}).items() if k in EVENTS}
    if not coverage:
        # Observation only: no interpolation, force-derived invented geometry, or Goal claim.
        counts = [frame["n"] for frame in frames]
        coverage = {
            "接触面": "OBSERVED" if any(counts) else "NOT_OBSERVED",
            "空接触面": "OBSERVED" if any(n == 0 for n in counts) else "NOT_OBSERVED",
        }
    errors = payload.get("errors", [])
    if payload.get("error"):
        errors = [
            *errors,
            {"type": payload.get("error_type", "Error"), "message": payload["error"]},
        ]
    status = str(payload["status"])
    if summary:
        for field in ("status", "accepted_steps"):
            if field in summary and summary[field] != payload[field]:
                raise ValueError(f"summary/payload mismatch: {path}: {field}")
    return {
        "label": label,
        "name": name,
        "world": key,
        "source": path.relative_to(root).as_posix(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "status": status,
        "accepted": accepted,
        "requested": requested,
        "free": free,
        "motion": "FREE 六自由度" if free else "导轨四自由度；导轨承担未放开的反作用",
        "coverage": coverage,
        "errors": errors,
        "definitions": definitions,
        "frames": frames,
        "trend": [
            [f["to"], float(np.linalg.norm(f["F"])), f["Fn"], f["Ft"], *f["dq"]] for f in frames
        ],
        "phase": "实线：入步几何/接触点；虚线：实际接受终点。载荷来自本步已接受 receipt。",
        "summary": None
        if not summary
        else path.with_name(path.name[:-8] + "-summary.json").relative_to(root).as_posix(),
    }


def collect(root):
    """Locate actual B3 evidence recursively; never replace an earlier failed attempt."""
    cases, summaries = [], {}
    for path in sorted(root.rglob("*summary*.json")):
        value = _read(path)
        if isinstance(value, dict) and "window" in value and "accepted_steps" in value:
            summaries[path.parent, value["window"]] = value
    for path in sorted(root.rglob("*.json.gz")):
        # Restrict filenames before decoding large unrelated physical evidence.
        if path.name not in {
            "center-drop.json.gz",
            "offset-support.json.gz",
            "dual-patch.json.gz",
            "live-continuous-slip-release.json.gz",
        }:
            continue
        payload = _read(path)
        if payload.get("schema") not in (WINDOW_SCHEMA, CONTINUOUS_SCHEMA):
            raise ValueError(f"unexpected B3 schema: {path}")
        summary = (
            summaries.get((path.parent, payload["window"]))
            if payload["schema"] == WINDOW_SCHEMA
            else None
        )
        for key, name, model, initial in _items(payload):
            cases.append(_case(payload, path, root, key, name, model, initial, summary))
    if not cases:
        raise ValueError("no saved B3 window/continuous payloads found")
    return cases


def _subset(count, limit):
    if count <= limit:
        return list(range(count))
    return sorted({round(index * (count - 1) / (limit - 1)) for index in range(limit)})


def _bounded(cases):
    # Only visualization frames/sample arrows are reduced, never source data or totals.
    for limit in (160, 80, 40, 20, 10, 2):
        output = []
        for case in cases:
            current = {key: value for key, value in case.items() if key != "frames"}
            current["frames"] = []
            for index in _subset(len(case["frames"]), limit):
                frame = dict(case["frames"][index])
                frame["s"] = [frame["s"][i] for i in _subset(len(frame["s"]), 12)]
                frame["r"] = [frame["r"][i] for i in _subset(len(frame["r"]), 24)]
                current["frames"].append(frame)
            output.append(current)
        data = json.dumps(
            {"cases": output}, ensure_ascii=False, allow_nan=False, separators=(",", ":")
        )
        data = (
            data.replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
        )
        html = HTML.replace("__EVIDENCE__", data)
        if len(html.encode("utf-8")) <= MAX_HTML_BYTES:
            return html
    raise ValueError(
        "all cases cannot fit in 1 MiB without hiding evidence; narrow --evidence-root"
    )


HTML = r"""<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>面积摩擦 · 真实轨迹</title>
<style>
:root{color-scheme:dark;--bg:#101823;--panel:#182434;--text:#e8edf4;--muted:#9eb0c4;--blue:#59b5ff;--orange:#ffad58;--green:#58d5ac}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px system-ui,"Microsoft YaHei",sans-serif}main{max-width:1320px;margin:auto;padding:26px}h1{font-size:25px;margin:0 0 8px;font-weight:650}p{line-height:1.7;margin:8px 0;color:var(--muted)}.bar,.cards,.grid{display:grid;gap:12px}.bar{grid-template-columns:1fr auto;align-items:center;margin:22px 0 12px}.cards{grid-template-columns:repeat(4,1fr);margin-bottom:14px}.card,.panel{background:var(--panel);border:1px solid #2c3d50;border-radius:12px;padding:14px}.card b{display:block;font-size:20px;margin-top:7px}.card span,.caption{color:var(--muted);font-size:12px}.grid{grid-template-columns:1fr 1fr}.panel h2{font-size:14px;margin:0 0 8px}.panel canvas{display:block;width:100%;height:285px}.panel.chart canvas{height:155px}.controls{display:flex;align-items:center;gap:12px;margin:14px 0}input[type=range]{width:100%;accent-color:var(--blue)}button,select{background:#22354b;color:var(--text);border:1px solid #415870;border-radius:7px;padding:10px;font:inherit}select{width:100%}button{cursor:pointer;white-space:nowrap}.legend{display:flex;gap:18px;flex-wrap:wrap;margin:12px 0}.blue{color:var(--blue)}.orange{color:var(--orange)}.green{color:var(--green)}.bad{color:#ff9397!important}.good{color:var(--green)}.pill{display:inline-block;border:1px solid #46566c;border-radius:5px;padding:3px 7px;margin:3px;font-size:12px}details{margin-top:16px;border-top:1px solid #304055;padding-top:12px}pre{white-space:pre-wrap;word-break:break-word;color:var(--muted);font:12px/1.7 ui-monospace,monospace}#empty{display:none;border:1px solid #b55560;color:#ffb2b5;padding:15px;margin:12px 0}label{white-space:nowrap;color:var(--muted)}@media(max-width:800px){main{padding:16px}.grid{grid-template-columns:1fr}.cards{grid-template-columns:1fr 1fr}.bar{grid-template-columns:1fr}.panel canvas{height:245px}}
</style>
<main><h1>面积摩擦 · 真实轨迹</h1><p>同次保存证据的离线回放。没有补运动、插值轨迹或新的物理求解。失败尝试独立保留。</p>
<div class="bar"><select id="case" aria-label="选择真实案例"></select><label><input id="zoom" type="checkbox"> 接触区域放大</label></div>
<div class="cards"><div class="card"><span>窗口状态 / accepted · requested</span><b id="status"></b></div><div class="card"><span>实际接受终点时间</span><b id="time"></b></div><div class="card"><span>作用于压力体的合力</span><b id="force"></b></div><div class="card"><span>运动约束</span><b id="motion" style="font-size:15px"></b></div></div>
<div id="empty">此尝试没有可播放的已接受帧；未提交候选不会画成真实运动。</div>
<div class="grid"><section class="panel"><h2>侧视 · X–Z / mm</h2><canvas id="side"></canvas></section><section class="panel"><h2>顶部 · X–Y / mm</h2><canvas id="top"></canvas></section></div>
<div class="legend"><span class="blue">↗ 蓝色：法向载荷 Fn·n</span><span class="orange">↗ 橙色：摩擦力 Ft</span><span class="green">绿色：真实裁剪接触区</span><span>实线入步 / 虚线接受终点</span></div>
<div class="controls"><button id="play">播放</button><input id="seek" type="range" min="0" value="0" aria-label="真实帧"><span id="counter"></span></div>
<p id="phase"></p><p id="sampling"></p><div class="grid"><section class="panel chart"><h2>载荷趋势 · N（全部已接受帧）</h2><canvas id="loads"></canvas><span class="caption">白：合力模　蓝：ΣFn　橙：Σ|Ft|；三者含义不同</span></section><section class="panel chart"><h2>实际 q 平移分量相对初态 · mm</h2><canvas id="travel"></canvas><span id="axes" class="caption"></span></section></div>
<div id="events"></div><details open><summary>状态、来源与范围</summary><pre id="provenance"></pre></details>
<p>帧、点箭头和裁剪面片可下采样，仅用于显示；载荷趋势由全部保存样本求和。PASS 沿用原完整样本验收，FAIL 前缀不冒充验收通过，本页不重新认证。几何显示值压缩至七位有效数字，原 gzip 保留完整精度。导轨案例不等价于自由体平衡；正则化滑动摩擦不等于静摩擦保证。</p></main>
<script id="evidence" type="application/json">__EVIDENCE__</script>
<script>
'use strict';
const cases=JSON.parse(document.getElementById('evidence').textContent).cases;
const $=id=>document.getElementById(id), colors=['#59b5ff','#ffad58','#58d5ac'];
let ci=cases.findLastIndex(c=>c.status==="PASS"&&c.name==="center-drop");
if(ci<0)ci=Math.max(0,cases.findLastIndex(c=>c.status==="PASS"));
let fi=0,playing=false,last=0;
const norm=v=>Math.hypot(...v), fmt=(x,n=3)=>Number(x).toFixed(n);
cases.forEach((c,i)=>{let o=document.createElement('option');o.value=i;o.textContent=`${c.label} · 世界 ${c.world} · ${c.status} · ${c.source}`;$('case').append(o)});
$('case').value=String(ci);
function setup(id){let el=$(id),d=window.devicePixelRatio||1,w=el.clientWidth,h=el.clientHeight;el.width=Math.round(w*d);el.height=Math.round(h*d);let ctx=el.getContext('2d');ctx.scale(d,d);return [ctx,w,h]}
function arrow(ctx,a,b,color){let dx=b[0]-a[0],dy=b[1]-a[1],m=Math.hypot(dx,dy);if(m<.7)return;ctx.strokeStyle=color;ctx.fillStyle=color;ctx.lineWidth=1.5;ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...b);ctx.stroke();let k=Math.min(5,m*.5),u=dx/m,v=dy/m;ctx.beginPath();ctx.moveTo(...b);ctx.lineTo(b[0]-k*u+k*.5*v,b[1]-k*v-k*.5*u);ctx.lineTo(b[0]-k*u-k*.5*v,b[1]-k*v+k*.5*u);ctx.fill()}
function view(id,axes,c,f){let [ctx,w,h]=setup(id);ctx.fillStyle='#101b29';ctx.fillRect(0,0,w,h);if(!f)return;let pts=[];for(const a of c.frames){for(const b of a.b)pts.push(...b[0],...(b[1]||[]));if($('zoom').checked){pts=[];for(const r of c.frames)pts.push(...r.s.map(s=>s.slice(0,3)),...r.r.flat());break}}if(!pts.length)pts=f.b.flatMap(b=>b[0]);if(!pts.length)return;let lows=axes.map(a=>Math.min(...pts.map(p=>p[a]))),highs=axes.map(a=>Math.max(...pts.map(p=>p[a])));let span=Math.max(highs[0]-lows[0],(highs[1]-lows[1])*(w-70)/(h-60),.004)*1.25,scale=(w-70)/span;let center=axes.map((_,i)=>(lows[i]+highs[i])/2),P=p=>[w/2+(p[axes[0]]-center[0])*scale,h/2-(p[axes[1]]-center[1])*scale];ctx.strokeStyle='#293c50';ctx.lineWidth=1;ctx.fillStyle='#92a9c0';ctx.font='11px system-ui';for(let i=0;i<=4;i++){let x=35+i*(w-70)/4;ctx.beginPath();ctx.moveTo(x,25);ctx.lineTo(x,h-25);ctx.stroke();ctx.fillText(fmt((center[0]+(x-w/2)/scale)*1000,1),x-14,h-8)}ctx.fillText('mm',w-28,h-8);ctx.fillText(`${axes[1]===2?'Z':'Y'}↑  X→`,8,16);
for(let i=0;i<f.b.length;i++){let body=f.b[i],def=c.definitions[i];for(let phase=0;phase<2;phase++){if(!body[phase])continue;ctx.setLineDash(phase?[4,4]:[]);ctx.strokeStyle=def.kind==='pressure'?(phase?'#dbe7f788':'#cad8e8'):'#758ba7';ctx.lineWidth=phase?1:1.3;for(let [a,b] of def.edges){ctx.beginPath();ctx.moveTo(...P(body[phase][a]));ctx.lineTo(...P(body[phase][b]));ctx.stroke()}}}ctx.setLineDash([]);
ctx.fillStyle='#58d5ac44';ctx.strokeStyle='#58d5ac';for(let tri of f.r){ctx.beginPath();tri.forEach((p,i)=>i?ctx.lineTo(...P(p)):ctx.moveTo(...P(p)));ctx.closePath();ctx.fill();ctx.stroke()}
let maxForce=Math.max(.001,...c.frames.flatMap(a=>a.s.map(s=>Math.max(norm(s.slice(3,6)),norm(s.slice(6,9))))));let gain=30/maxForce;for(let s of f.s){let p=P(s);ctx.fillStyle='#e7f0fa';ctx.beginPath();ctx.arc(...p,2,0,7);ctx.fill();for(let [start,color] of [[3,colors[0]],[6,colors[1]]])arrow(ctx,p,[p[0]+s[start+axes[0]]*gain,p[1]-s[start+axes[1]]*gain],color)}ctx.fillStyle='#9eb0c4';ctx.fillText(`箭头：30 px = ${maxForce.toPrecision(3)} N（两视图相同）`,10,h-26)}
function chart(id,c,f,travel){let [ctx,w,h]=setup(id),rows=c.trend;ctx.fillStyle='#101b29';ctx.fillRect(0,0,w,h);if(!rows.length)return;let columns=travel?(c.free?[4,5,6]:[4,5]):[1,2,3],maxT=Math.max(rows.at(-1)[0],.001),values=rows.flatMap(r=>columns.map(k=>r[k]));let lo=Math.min(0,...values),hi=Math.max(0,...values);if(hi===lo)hi=lo+1;let margin=(hi-lo)*.1;lo-=margin;hi+=margin;const X=t=>38+t/maxT*(w-52),Y=v=>h-25-(v-lo)/(hi-lo)*(h-42);ctx.strokeStyle='#354a60';ctx.fillStyle='#9eb0c4';ctx.font='11px system-ui';for(let j=0;j<3;j++){let val=lo+(hi-lo)*j/2,y=Y(val);ctx.beginPath();ctx.moveTo(38,y);ctx.lineTo(w-14,y);ctx.stroke();ctx.fillText(val.toPrecision(3),1,y+3)}columns.forEach((col,i)=>{ctx.strokeStyle=travel?colors[i]:['#edf2f8',...colors][i];ctx.lineWidth=1.5;ctx.beginPath();rows.forEach((r,j)=>j?ctx.lineTo(X(r[0]),Y(r[col])):ctx.moveTo(X(r[0]),Y(r[col])));ctx.stroke()});if(f){ctx.strokeStyle='#ffffff77';ctx.beginPath();ctx.moveTo(X(f.to),10);ctx.lineTo(X(f.to),h-23);ctx.stroke()}ctx.fillText('0 s',38,h-7);ctx.fillText(fmt(maxT,3)+' s',w-65,h-7)}
function draw(){let c=cases[ci],f=c.frames[fi];$('status').textContent=`${c.status} · ${c.accepted}/${c.requested}`;$('status').className=c.status==='PASS'?'good':'bad';$('motion').textContent=c.motion;$('empty').style.display=f?'none':'block';$('time').textContent=f?fmt(f.to,6)+' s':'—';$('force').textContent=f?fmt(norm(f.F))+' N':'—';$('phase').textContent=c.phase+(f?` 输入 ${fmt(f.ti,6)} s → 输出 ${fmt(f.to,6)} s；F = [${f.F.map(x=>fmt(x)).join(', ')}] N。`:'');$('counter').textContent=f?`${fi+1}/${c.frames.length}`:'0/0';$('sampling').textContent=`显示 ${c.frames.length}/${c.accepted} 个真实帧；当前箭头 ${f?f.s.length:0}/${f?f.n:0} 个实际样本；裁剪面片 ${f?f.r.length:0}/${f?f.nr:0}。播放为慢放，逐帧跳转，无运动插值。`;$('axes').textContent=c.free?'蓝 Δx / 橙 Δy / 绿 Δz，均来自实际 accepted q':'蓝 pressure-x / 橙 pressure-vertical，均来自实际 accepted q';$('seek').max=Math.max(0,c.frames.length-1);$('seek').value=fi;$('play').disabled=!f;view('side',[0,2],c,f);view('top',[0,1],c,f);chart('loads',c,f,false);chart('travel',c,f,true);$('events').replaceChildren();for(let [k,v] of Object.entries(c.coverage)){let e=document.createElement('span');e.className='pill';e.textContent=k+' · '+v;$('events').append(e)}$('provenance').textContent=`来源：${c.source}\nSHA256：${c.sha256}\n世界：${c.world}；约束：${c.motion}\nsummary：${c.summary||'状态直接取原 payload'}\n原始状态：${c.status}\n错误：${c.errors.length?c.errors.map(e=>e.type+': '+e.message).join('\n'):'无保存错误'}\n未提交的候选不进入画面；NOT_OBSERVED 只表示本窗口未观察到，不补做事件。`}
$('case').onchange=()=>{ci=+$('case').value;fi=0;playing=false;$('play').textContent='播放';draw()};$('seek').oninput=()=>{fi=+$('seek').value;draw()};$('zoom').onchange=draw;$('play').onclick=()=>{playing=!playing;if(playing&&fi===cases[ci].frames.length-1)fi=0;$('play').textContent=playing?'暂停':'播放';draw()};window.addEventListener('resize',draw);
function loop(t){if(playing&&t-last>90){last=t;if(fi+1<cases[ci].frames.length)fi++;else{playing=false;$('play').textContent='播放'}draw()}requestAnimationFrame(loop)}draw();requestAnimationFrame(loop);
</script></html>"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, output = args.evidence_root, args.output
    if not root.is_absolute() or not root.is_dir():
        raise ValueError("--evidence-root must be an existing absolute directory")
    if not output.is_absolute() or output.suffix.lower() != ".html" or output.exists():
        raise ValueError("--output must be a fresh absolute .html file")
    if not output.parent.is_dir():
        raise ValueError("output parent must already exist")
    cases = collect(root.resolve())
    content = _bounded(cases)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(content)
    print(
        json.dumps(
            {
                "output": str(output),
                "bytes": len(content.encode("utf-8")),
                "cases": len(cases),
                "source_files": len({c["source"] for c in cases}),
                "physics_executed": False,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
