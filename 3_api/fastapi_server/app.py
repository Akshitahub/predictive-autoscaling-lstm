"""
LSTM Auto-Scaling API
Minor project: ML-Based Predictive Load Balancing and Auto-Scaling in Cloud Environments

Input: request counts for the last 3 hours (36 values, one per 5 minutes)
Output: LSTM forecast for the next 5 minutes, number of VMs needed
(forecast + safety margin, divided by VM capacity) and scale up / down / hold

Endpoints:
  GET  /          -> interactive demo page
  GET  /docs      -> Swagger UI (auto-generated)
  GET  /health    -> health check
  GET  /sample    -> a real 1-hour window from the NASA test data
  POST /predict   -> forecast + scaling decision
"""
import math
import time
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from lstm_numpy import LSTMForecaster

model = LSTMForecaster("model_weights.json")
START = time.time()

app = FastAPI(
    title="LSTM Auto-Scaling API",
    description="Predicts next-interval web traffic with an LSTM and recommends how many VMs to run.",
    version="1.0",
)

# Real 3-hour window from the NASA test set (24 Jul 1995, 10:15-13:10); the actual next value (13:15) was 434
SAMPLE = [240, 270, 306, 257, 261, 325, 300, 177, 290, 361, 347, 258,
          390, 367, 348, 227, 197, 295, 379, 407, 440, 425, 454, 398,
          356, 329, 531, 447, 421, 430, 471, 409, 389, 333, 398, 421]


class PredictRequest(BaseModel):
    recent_requests: List[float] = Field(
        ..., description="Last 36 five-minute request counts (3 hours), oldest first", examples=[SAMPLE])
    vm_capacity: int = Field(100, gt=0, description="Requests one VM can serve per 5 minutes")
    safety_margin: float = Field(0.40, ge=0, le=2, description="Extra capacity on top of the forecast (0.4 = 40%)")
    current_vms: Optional[int] = Field(None, ge=0, description="VMs running now (optional, gives a scale action)")


class PredictResponse(BaseModel):
    forecast_requests: float
    recommended_vms: int
    action: str
    capacity_with_recommended_vms: int
    expected_utilization_pct: float
    model: str
    inference_ms: float


@app.get("/health")
def health():
    return {"status": "ok", "model": model.description, "parameters": model.n_params, "window": model.window,
            "interval_minutes": model.interval_minutes, "uptime_s": round(time.time() - START)}


@app.get("/sample")
def sample():
    return {"recent_requests": SAMPLE, "note": "NASA KSC HTTP traffic, 24 Jul 1995 10:15-13:10 (actual next value at 13:15: 434)"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if len(req.recent_requests) < model.window:
        raise HTTPException(422, f"recent_requests needs at least {model.window} values (got {len(req.recent_requests)})")
    if any(v < 0 for v in req.recent_requests):
        raise HTTPException(422, "request counts cannot be negative")

    t0 = time.perf_counter()
    forecast = model.predict(req.recent_requests)
    ms = (time.perf_counter() - t0) * 1000

    vms = max(1, math.ceil(forecast * (1 + req.safety_margin) / req.vm_capacity))
    if req.current_vms is None:
        action = "set"
    elif vms > req.current_vms:
        action = f"scale_up_by_{vms - req.current_vms}"
    elif vms < req.current_vms:
        action = f"scale_down_by_{req.current_vms - vms}"
    else:
        action = "hold"

    cap = vms * req.vm_capacity
    return PredictResponse(
        forecast_requests=round(forecast, 1),
        recommended_vms=vms,
        action=action,
        capacity_with_recommended_vms=cap,
        expected_utilization_pct=round(100 * forecast / cap, 1),
        model=f"{model.description} (trained on NASA HTTP logs, July 1995)",
        inference_ms=round(ms, 2),
    )


@app.get("/", response_class=HTMLResponse)
def home():
    return HOME_HTML


HOME_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>LSTM Auto-Scaling API</title>
<style>
 :root{--bg:#0f1220;--card:#181c2e;--line:#2a3050;--text:#e8eaf3;--muted:#9aa0bd;--accent:#6c8cff;--ok:#3ecf8e;--warn:#ffb547}
 *{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:16px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
 main{max-width:860px;margin:0 auto;padding:32px 16px}
 h1{margin:0 0 4px;font-size:28px}p.sub{margin:0 0 24px;color:var(--muted)}
 .card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:20px;margin-bottom:16px}
 label{display:block;font-size:13px;color:var(--muted);margin-bottom:6px}
 .grid{display:grid;grid-template-columns:repeat(6,1fr);gap:8px}
 @media(max-width:600px){.grid{grid-template-columns:repeat(4,1fr)}}
 input{width:100%;background:#0f1220;border:1px solid var(--line);color:var(--text);border-radius:8px;padding:8px;font-size:15px}
 .row{display:flex;gap:12px;flex-wrap:wrap;margin-top:14px}.row>div{flex:1;min-width:140px}
 button{background:var(--accent);color:#fff;border:0;border-radius:10px;padding:11px 18px;font-size:15px;font-weight:600;cursor:pointer}
 button.ghost{background:transparent;border:1px solid var(--line);color:var(--text)}
 .btns{display:flex;gap:10px;margin-top:16px;flex-wrap:wrap}
 .stats{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}@media(max-width:600px){.stats{grid-template-columns:1fr}}
 .stat{background:#0f1220;border:1px solid var(--line);border-radius:12px;padding:14px}
 .stat b{display:block;font-size:30px}.stat span{color:var(--muted);font-size:13px}
 .action{margin-top:12px;font-weight:600}.up{color:var(--warn)}.down{color:var(--ok)}
 svg{width:100%;height:170px;display:block}
 a{color:var(--accent)}.foot{color:var(--muted);font-size:13px}
</style></head><body><main>
<h1>LSTM Auto-Scaling API</h1>
<p class="sub">Enter the request counts for the last 3 hours (36 values, one for every 5 minutes). The LSTM model predicts the next 5 minutes and suggests how many VMs are needed.</p>
<div class="card">
 <label>Requests per 5 minutes, oldest to newest</label>
 <div class="grid" id="inputs"></div>
 <div class="row">
  <div><label>VM capacity (req / 5 min)</label><input id="cap" type="number" value="100" min="1"></div>
  <div><label>Safety margin</label><input id="margin" type="number" value="0.4" step="0.05" min="0" max="2"></div>
  <div><label>VMs running now (optional)</label><input id="cur" type="number" placeholder="e.g. 3" min="0"></div>
 </div>
 <div class="btns"><button onclick="run()">Predict</button><button class="ghost" onclick="loadSample()">Load NASA sample</button>
 <button class="ghost" onclick="spike()">Real traffic surge (27 Jul)</button></div>
</div>
<div class="card" id="out" style="display:none">
 <div class="stats">
  <div class="stat"><span>Forecast (next 5 min)</span><b id="fc">-</b></div>
  <div class="stat"><span>Recommended VMs</span><b id="vms">-</b></div>
  <div class="stat"><span>Expected utilisation</span><b id="util">-</b></div>
 </div>
 <div class="action" id="act"></div>
 <svg id="chart" viewBox="0 0 600 170" preserveAspectRatio="none"></svg>
 <div class="foot" id="meta"></div>
</div>
<p class="foot">API docs: <a href="docs">/docs</a> | Health: <a href="health">/health</a> | Model: LSTM trained on NASA KSC HTTP logs (July 1995).</p>
</main>
<script>
const S=[240,270,306,257,261,325,300,177,290,361,347,258,390,367,348,227,197,295,379,407,440,425,454,398,356,329,531,447,421,430,471,409,389,333,398,421];
const box=document.getElementById('inputs');
for(let i=0;i<S.length;i++){const e=document.createElement('input');e.type='number';e.min='0';e.value=S[i];box.appendChild(e)}
const vals=()=>[...box.querySelectorAll('input')].map(e=>parseFloat(e.value)||0);
function setVals(a){[...box.querySelectorAll('input')].forEach((e,i)=>e.value=a[i])}
function loadSample(){setVals(S);run()}
function spike(){setVals([360,246,337,308,392,362,342,379,293,322,315,379,295,399,309,259,349,398,324,354,398,454,273,286,256,317,369,482,424,427,351,376,423,507,584,395]);run()}
async function run(){
 const body={recent_requests:vals(),vm_capacity:+document.getElementById('cap').value||100,
   safety_margin:+document.getElementById('margin').value};
 const c=document.getElementById('cur').value; if(c!=='')body.current_vms=+c;
 const r=await fetch('predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
 const d=await r.json(); if(!r.ok){alert(JSON.stringify(d.detail));return}
 document.getElementById('out').style.display='block';
 fc.textContent=d.forecast_requests; vms.textContent=d.recommended_vms; util.textContent=d.expected_utilization_pct+'%';
 const a=document.getElementById('act'); a.className='action '+(d.action.startsWith('scale_up')?'up':d.action.startsWith('scale_down')?'down':'');
 a.textContent='Action: '+d.action.replaceAll('_',' ');
 meta.textContent='Capacity with '+d.recommended_vms+' VMs: '+d.capacity_with_recommended_vms+' req, inference '+d.inference_ms+' ms';
 draw(body.recent_requests,d.forecast_requests,d.capacity_with_recommended_vms);
}
function draw(h,f,cap){
 const all=[...h,f,cap],mx=Math.max(...all)*1.1||1,W=600,H=160,n=h.length+1,x=i=>20+i*(W-40)/(n-1),y=v=>H-v/mx*(H-10);
 let p=h.map((v,i)=>(i?'L':'M')+x(i)+','+y(v)).join('');
 chart.innerHTML='<line x1="0" x2="600" y1="'+y(cap)+'" y2="'+y(cap)+'" stroke="#3ecf8e" stroke-dasharray="6 5"/>'+
 '<text x="596" y="'+(y(cap)-5)+'" fill="#3ecf8e" font-size="12" text-anchor="end">VM capacity</text>'+
 '<path d="'+p+'" fill="none" stroke="#e8eaf3" stroke-width="2"/>'+
 '<line x1="'+x(n-2)+'" y1="'+y(h[n-2])+'" x2="'+x(n-1)+'" y2="'+y(f)+'" stroke="#6c8cff" stroke-width="2" stroke-dasharray="4 4"/>'+
 '<circle cx="'+x(n-1)+'" cy="'+y(f)+'" r="5" fill="#6c8cff"/>'+
 '<text x="'+(x(n-1)-8)+'" y="'+(y(f)-10)+'" fill="#6c8cff" font-size="12" text-anchor="end">forecast</text>';
}
run();
</script></body></html>"""
