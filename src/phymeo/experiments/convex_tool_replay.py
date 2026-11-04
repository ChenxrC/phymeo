"""Self-contained replay of recorded convex-tray states and material force samples."""

import json
from pathlib import Path


def write_tool_replay(trace, path):
    if trace.get("schema") != "phymeo-convex-tool-transfer-v1" or not trace.get("frames"):
        raise ValueError("expected a convex tool transfer trajectory")
    payload = json.dumps(trace, allow_nan=False, separators=(",", ":")).replace("<", "\\u003c")
    html = _HTML.replace("__TRACE_JSON__", payload)
    Path(path).write_text(html, encoding="utf8")


_HTML = """<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>有限凸托板与自由软布 · 物理轨迹</title>
<style>
:root{font-family:system-ui,'Microsoft YaHei',sans-serif;color:#23373b;background:#f4f4ee}*{box-sizing:border-box}body{margin:0}main{max-width:1140px;margin:2.4rem auto;padding:0 1.6rem}header{display:flex;justify-content:space-between;gap:1rem;align-items:flex-end}h1{font-weight:640;font-size:clamp(25px,4vw,44px);line-height:1.1;margin:.25rem 0}small,.dim{color:#4d6666}#badge{white-space:nowrap;background:#e1ebe2;padding:.5rem .8rem;border-radius:20px}section{border:1px solid #c6d2c9;background:#fafaf6;border-radius:12px;padding:.8rem;margin-top:1rem}.views{display:grid;grid-template-columns:2fr 1fr;gap:.75rem}canvas{width:100%;height:355px;display:block;background:#f7f8f1;border-radius:7px}.view{min-width:0}.view p{margin:0 0 .45rem;color:#506661;font-size:.82rem}.controls{display:flex;align-items:center;gap:.8rem;flex-wrap:wrap}button,select{background:#215e59;color:white;border:0;border-radius:6px;padding:.5rem .85rem;font:inherit}select{background:#e3ece6;color:#173d38}input{flex:1;min-width:160px;accent-color:#267267}#numbers{font-variant-numeric:tabular-nums;margin:.7rem 0 0}.legend{font-size:.8rem;color:#4c6260;line-height:1.6}.teal{color:#15786b}.amber{color:#a35617}.red{color:#b24939}@media(max-width:700px){.views{grid-template-columns:1fr}canvas{height:215px}header{align-items:flex-start;flex-direction:column}main{margin:1rem auto}}
</style><main><header><div><small>PHYMEO / MATERIAL CONTACT</small><h1>凸托板牵引与软布释放</h1><span id="info" class="dim"></span></div><span id="badge"></span></header>
<section class="views"><div class="view"><p>侧面 · 水平位置 x 与高度 z（m）</p><canvas id="side"></canvas></div><div class="view"><p>俯视 · 水平位置 x 与横向 y（m）</p><canvas id="top"></canvas></div></section>
<section><div class="controls"><button id="play">播放</button><input id="frame" type="range" min="0" value="0" step="1" aria-label="物理步"><select id="speed" aria-label="播放速度"><option value=".25">0.25×</option><option value="1" selected>1×</option><option value="3">3×</option></select></div><p id="numbers"></p><div id="marks" class="controls" aria-label="测得的关键步"></div><div class="legend"><b class="teal">绿</b> · 布与材料节点　<b class="amber">橙</b> · 连杆上的实体托板　<b class="red">红</b> · 当步实际中点材料站（侧图箭头为布所受力）　虚线 · 已经过的布质心轨迹</div></section></main>
<script type="application/json" id="trace">__TRACE_JSON__</script><script>
"use strict"
const trace=JSON.parse(document.getElementById("trace").textContent), frames=trace.frames, model=trace.model
const slider=document.getElementById("frame"), play=document.getElementById("play"), speed=document.getElementById("speed")
slider.max=frames.length-1
const labels={initial:"初态",lift:"托举",drag:"横拖",withdraw:"撤出"}
document.getElementById("info").textContent=model.fabric+" · "+model.footprint+" · 摩擦系数 "+model.friction+" · "+trace.dt_s.toFixed(6)+" s / 步"
document.getElementById("badge").textContent=trace.qualified?"通过牵引与释放":(trace.status==="completed"?"对照已完成":"物理窗口未完成")
function bounds(axis){let lo=Infinity,hi=-Infinity; for(const f of frames)for(const p of f.nodes_m.concat(f.tool_vertices_m,f.robot_links_m)){lo=Math.min(lo,p[axis]);hi=Math.max(hi,p[axis])}return[lo,hi]}
const ranges=[bounds(0),bounds(1),bounds(2)]
function plot(canvas,k,now){const dpr=window.devicePixelRatio||1,w=canvas.clientWidth,h=canvas.clientHeight
canvas.width=w*dpr;canvas.height=h*dpr;const c=canvas.getContext("2d");c.scale(dpr,dpr)
const m=25,x=ranges[0],y=ranges[k],scale=Math.min((w-2*m)/Math.max(.1,x[1]-x[0]),(h-2*m)/Math.max(.075,y[1]-y[0]))
function to(p){return[w/2+(p[0]-(x[0]+x[1])/2)*scale,h/2-(p[k]-(y[0]+y[1])/2)*scale]}
function line(points,color,width,fill){c.beginPath();points.forEach((p,i)=>{const v=to(p);i?c.lineTo(...v):c.moveTo(...v)});if(fill){c.closePath();c.fillStyle=fill;c.fill()}c.strokeStyle=color;c.lineWidth=width;c.stroke()}
c.strokeStyle="#dce5dc";c.fillStyle="#688078";c.font="10px system-ui"
for(let z=Math.floor(y[0]*10)/10;z<=y[1]+.06;z+=.05){let p=[0,0,0];p[k]=z;let screen=to(p)[1];c.beginPath();c.moveTo(0,screen);c.lineTo(w,screen);c.stroke();c.fillText(z.toFixed(2),3,screen-2)}
const f=frames[now];c.setLineDash([3,4]);line(frames.slice(0,now+1).map(r=>r.com_m),"#819795",1);c.setLineDash([])
for(const face of model.tool_faces)line(face.map(i=>f.tool_vertices_m[i]),"#9d682d",1.3,"#e6c58c60")
line([model.base_m,...f.robot_links_m],"#78745d",3)
for(const face of model.faces)line(face.map(i=>f.nodes_m[i]),"#25766b",2,"#89c9bda0")
for(const p of f.nodes_m){const a=to(p);c.fillStyle="#11675c";c.beginPath();c.arc(...a,2.6,0,7);c.fill()}
for(const row of f.contacts)for(const s of row.sites){let a=to(s.cloth);c.fillStyle="#bf493d";c.beginPath();c.arc(...a,2.8,0,7);c.fill();if(k===2){let tip=s.cloth.map((u,j)=>u-.055*s.force_on_tool_n[j]);line([s.cloth,tip],"#b84e3f",1.6)}}
}
function show(n){slider.value=n;const f=frames[n],free=trace.events.find(e=>e.kind==="independent_flight")
const phase=free&&n>free.start_step?"独立落下":labels[f.phase]
document.getElementById("numbers").textContent="步 "+n+" / "+(frames.length-1)+"　"+f.time_s.toFixed(4)+" s　"+phase+"　布质心 x "+f.com_m[0].toFixed(4)+" m · z "+f.com_m[2].toFixed(4)+" m　布承载 "+f.cloth_contact_n[2].toFixed(4)+" N　材料站 "+f.material_sites
plot(document.getElementById("side"),2,n);plot(document.getElementById("top"),1,n)}
const eventNames={lifted:"抬起",withdraw:"开始撤出",independent_flight:"已验自由落下"};for(const e of trace.events){const b=document.createElement("button");b.textContent=eventNames[e.kind]+" · "+e.step;b.dataset.step=e.step;b.onclick=()=>{pause();show(e.step)};document.getElementById("marks").appendChild(b)}
let playing=false,start=0,origin=0,animation=0
function tick(){if(!playing)return;let n=Math.min(frames.length-1,origin+Math.floor(Math.max(0,performance.now()-start)*.001*Number(speed.value)/trace.dt_s));show(n);if(n===frames.length-1){playing=false;play.textContent="播放"}else animation=setTimeout(tick,16)}
function pause(){playing=false;clearTimeout(animation);play.textContent="播放"}
play.onclick=()=>{if(playing){pause();return}origin=Number(slider.value)===frames.length-1?0:Number(slider.value);start=performance.now();playing=true;play.textContent="暂停";tick()}
slider.oninput=()=>{pause();show(Number(slider.value))};speed.onchange=pause;window.addEventListener("resize",()=>show(Number(slider.value)));show(0)
</script></html>"""
