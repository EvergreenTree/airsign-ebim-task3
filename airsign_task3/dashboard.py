from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from .runtime import RuntimeStore


CAMERAS = ("overview", "head", "left_wrist", "right_wrist")


class Feedback(BaseModel):
    realism: int = Field(ge=1, le=5)
    note: str = Field(max_length=2000)
    timestamp: float | None = None


def _dashboard_html() -> str:
    return r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>AirSign · EBiM Task 3</title>
  <style>
    :root { --bg:#07100e; --panel:#0d1916; --line:#24433a; --text:#e7f3ee;
      --muted:#94aba3; --lime:#a7f542; --cyan:#57e3d5; --amber:#ffc857; --red:#ff6b6b; }
    * { box-sizing:border-box; }
    body { margin:0; background:radial-gradient(circle at 70% -20%,#193a32 0,#07100e 42%);
      color:var(--text); font:14px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace; }
    header { display:flex; align-items:end; justify-content:space-between; padding:22px 26px 14px;
      border-bottom:1px solid var(--line); }
    h1 { margin:0; font:700 20px/1.1 system-ui,sans-serif; letter-spacing:.02em; }
    .kicker { color:var(--lime); text-transform:uppercase; font-size:11px; letter-spacing:.18em; }
    .live { display:flex; gap:8px; align-items:center; color:var(--muted); }
    .dot { width:8px;height:8px;border-radius:50%;background:var(--amber);box-shadow:0 0 12px var(--amber); }
    .dot.ok { background:var(--lime);box-shadow:0 0 12px var(--lime); }
    main { display:grid; grid-template-columns:minmax(0,1.75fr) minmax(310px,.75fr); gap:14px; padding:14px; }
    .panel { background:color-mix(in srgb,var(--panel) 92%,transparent); border:1px solid var(--line);
      border-radius:12px; overflow:hidden; box-shadow:0 12px 40px #0006; }
    .panel-head { display:flex; justify-content:space-between; padding:10px 12px; border-bottom:1px solid var(--line);
      color:var(--muted); text-transform:uppercase; letter-spacing:.08em; font-size:11px; }
    .overview { aspect-ratio:16/9; width:100%; object-fit:cover; display:block; background:#020504; }
    .camera-grid { display:grid; grid-template-columns:repeat(3,1fr); border-top:1px solid var(--line); }
    .camera-grid div { border-right:1px solid var(--line); position:relative; }
    .camera-grid div:last-child { border:0; }
    .camera-grid img { width:100%; aspect-ratio:16/10; object-fit:cover; display:block; background:#020504; }
    .tag { position:absolute; left:8px; bottom:6px; color:white; font-size:10px; text-shadow:0 1px 4px #000; }
    aside { display:grid; gap:14px; align-content:start; }
    .state { padding:14px; }
    .score { font:700 46px/1 system-ui,sans-serif; color:var(--lime); letter-spacing:-.04em; }
    .score small { font-size:16px; color:var(--muted); letter-spacing:0; }
    .stage { margin:12px 0; padding:10px; border:1px solid var(--line); border-radius:8px; }
    .stage b { display:block; font:650 16px system-ui,sans-serif; }
    dl { display:grid; grid-template-columns:1fr auto; gap:7px 12px; margin:12px 0 0; }
    dt { color:var(--muted); } dd { margin:0; }
    .controls { display:grid; grid-template-columns:repeat(2,1fr); gap:8px; padding:12px; }
    button { border:1px solid var(--line); background:#12241f; color:var(--text); border-radius:8px;
      padding:10px; font:inherit; cursor:pointer; }
    button:hover { border-color:var(--cyan); }
    button.primary { background:var(--lime); color:#10200b; border-color:var(--lime); font-weight:700; }
    button.danger { color:var(--red); }
    .feedback { padding:12px; }
    textarea { width:100%; min-height:70px; resize:vertical; border:1px solid var(--line); border-radius:8px;
      background:#08120f; color:var(--text); padding:9px; font:inherit; }
    .rating { display:flex; gap:6px; margin:8px 0; }
    .rating button.active { background:var(--cyan); color:#05201d; }
    .notice { color:var(--muted); font-size:11px; margin-top:8px; }
    @media(max-width:900px){ main{grid-template-columns:1fr}.camera-grid{grid-template-columns:1fr} }
  </style>
</head>
<body>
  <header><div><div class="kicker">AirSign / Physical policy</div><h1>EBiM Task 3 Control Room</h1></div>
    <div class="live"><span id="dot" class="dot"></span><span id="connection">connecting</span></div></header>
  <main>
    <section class="panel"><div class="panel-head"><span>Overview camera</span><span id="simTime">t = 0.00s</span></div>
      <img class="overview" data-stream="/stream/overview" alt="Task 3 overview camera" />
      <div class="camera-grid">
        <div><img data-stream="/stream/head" alt="head camera"/><span class="tag">HEAD SAFETY</span></div>
        <div><img data-stream="/stream/left_wrist" alt="grasp detail camera"/><span class="tag">GRASP DETAIL</span></div>
        <div><img data-stream="/stream/right_wrist" alt="right wrist camera"/><span class="tag">RIGHT WRIST</span></div>
      </div>
    </section>
    <aside>
      <section class="panel state"><div class="score"><span id="score">0.00</span><small>/16</small></div>
        <div class="stage"><span id="lifecycle">READY</span><b id="stage">TABLE SETUP</b><span id="substate">IDLE</span></div>
        <dl><dt>Real-time factor</dt><dd id="rtf">0.00×</dd><dt>Recovery ratio</dt><dd id="recovery">0.0%</dd>
          <dt>Base XY</dt><dd id="base">0.00, 0.00</dd>
          <dt>Head force</dt><dd id="force">0.0 N</dd><dt>Peak head force</dt><dd id="peak">0.0 N</dd>
          <dt>Watchdog stops</dt><dd id="watchdog">0</dd></dl>
        <p id="message" class="notice">Initializing Isaac Sim</p></section>
      <section class="panel"><div class="panel-head"><span>Episode control</span><span>seed <b id="seed">—</b></span></div>
        <div class="controls"><button class="primary" onclick="control('start')">Start</button><button onclick="control('pause')">Pause</button>
          <button onclick="control('resume')">Resume</button><button class="danger" onclick="control('reset')">Reset episode</button></div></section>
      <section class="panel feedback"><div class="panel-head"><span>Realism feedback</span><span id="feedbackState"></span></div>
        <div class="rating" id="rating"></div><textarea id="note" placeholder="What looks physically wrong or right? Include the object or motion."></textarea>
        <button style="width:100%;margin-top:8px" onclick="sendFeedback()">Timestamp & submit feedback</button>
        <div class="notice">Feedback is logged with the current simulator state. It does not pause development unless you press Pause.</div></section>
    </aside>
  </main>
<script>
let realism=3;
const rating=document.getElementById('rating');
for(let n=1;n<=5;n++){const b=document.createElement('button');b.textContent=n;b.onclick=()=>{realism=n;drawRating()};b.id=`r${n}`;rating.appendChild(b)}
function drawRating(){for(let n=1;n<=5;n++)document.getElementById(`r${n}`).classList.toggle('active',n===realism)} drawRating();
async function control(action){let response=await fetch(`/api/control/${action}`,{method:'POST'});let body=await response.json();if(!response.ok)alert(body.detail||'Control failed')}
async function sendFeedback(){let note=document.getElementById('note').value.trim();let response=await fetch('/api/feedback',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({realism,note,timestamp:Date.now()/1000})});
 document.getElementById('feedbackState').textContent=response.ok?'saved':'error';if(response.ok)document.getElementById('note').value='';}
function show(s){document.getElementById('score').textContent=Number(s.score).toFixed(2);document.getElementById('lifecycle').textContent=s.lifecycle;
 document.getElementById('stage').textContent=s.stage.replaceAll('_',' ');document.getElementById('substate').textContent=s.substate;document.getElementById('rtf').textContent=Number(s.real_time_factor).toFixed(2)+'×';
 document.getElementById('recovery').textContent=(100*Number(s.recovery_ratio)).toFixed(1)+'%';document.getElementById('force').textContent=Number(s.safety.current_head_force_n).toFixed(1)+' N';
 document.getElementById('base').textContent=Number(s.robot_position[0]).toFixed(2)+', '+Number(s.robot_position[1]).toFixed(2);
 document.getElementById('peak').textContent=Number(s.safety.peak_head_force_n).toFixed(1)+' N';document.getElementById('watchdog').textContent=s.safety.watchdog_interventions;
 document.getElementById('message').textContent=s.message;document.getElementById('seed').textContent=s.seed;document.getElementById('simTime').textContent='t = '+Number(s.simulated_seconds).toFixed(2)+'s';}
let ws;
function connect(){ws=new WebSocket(`ws://${location.host}/ws/telemetry`);ws.onopen=()=>{document.getElementById('dot').classList.add('ok');document.getElementById('connection').textContent='live'};
 ws.onmessage=e=>show(JSON.parse(e.data));ws.onclose=()=>{document.getElementById('dot').classList.remove('ok');document.getElementById('connection').textContent='reconnecting';setTimeout(connect,1000)}} connect();
// Start the never-ending MJPEG responses only after the document load event.
// This keeps browser navigation/reload from waiting forever on image streams.
window.addEventListener('load',()=>setTimeout(()=>{
  document.querySelectorAll('img[data-stream]').forEach(img=>{img.src=img.dataset.stream});
},0));
</script></body></html>"""


def create_app(store: RuntimeStore) -> FastAPI:
    app = FastAPI(title="AirSign EBiM Task 3", version="0.1.0")

    @app.get("/", response_class=HTMLResponse)
    def dashboard() -> str:
        return _dashboard_html()

    @app.get("/api/state")
    def state() -> dict[str, Any]:
        return store.state()

    @app.post("/api/control/{command}")
    def control(command: str) -> dict[str, str]:
        if command not in {"start", "pause", "resume", "reset"}:
            raise HTTPException(status_code=404, detail="Unknown command")
        if command == "reset":
            # A full process restart is the only reset that reliably restores
            # dynamic beans and every unregistered room rigid body. It also
            # prevents reset logic from becoming an object-teleport backdoor.
            store.event("control", command=command)
            store.request_reset()
        else:
            store.queue_command(command)
        return {"accepted": command}

    @app.post("/api/feedback")
    def feedback(payload: Feedback, request: Request) -> dict[str, bool]:
        record = payload.model_dump()
        record["received_at"] = time.time()
        record["remote"] = request.client.host if request.client else None
        record["state"] = store.state()
        path = store.run_dir / "realism-feedback.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        store.event("realism_feedback", realism=payload.realism, note=payload.note)
        return {"stored": True}

    def multipart(camera: str):
        while not store.stop_requested:
            frame = store.get_frame(camera)
            if frame is not None:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            time.sleep(0.08)

    @app.get("/stream/{camera}")
    def stream(camera: str) -> StreamingResponse:
        if camera not in CAMERAS:
            raise HTTPException(status_code=404, detail="Unknown camera")
        return StreamingResponse(multipart(camera), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.websocket("/ws/telemetry")
    async def telemetry(websocket: WebSocket) -> None:
        await websocket.accept()
        try:
            while not store.stop_requested:
                await websocket.send_json(store.state())
                await asyncio.sleep(0.1)
        except WebSocketDisconnect:
            return

    return app


def start_dashboard(store: RuntimeStore, port: int) -> threading.Thread:
    app = create_app(store)
    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=port,
        log_level="info",
        access_log=False,
        loop="asyncio",
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, name="task3-dashboard", daemon=True)
    thread.start()
    return thread
