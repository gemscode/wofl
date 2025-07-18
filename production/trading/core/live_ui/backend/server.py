#!/usr/bin/env python3
import asyncio, json, os, redis.asyncio as redis
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# ---------- CONFIG ----------
TICKER      = os.getenv("TICKER", "GERN").upper()
REDIS_HOST  = os.getenv("REDIS_HOST", "localhost")
REDIS_PWD   = os.getenv("REDIS_PWD", "")          # read from .redis_passwd if desired
ANALYSIS_STREAM = f"trading_analysis_{TICKER}"
GROUP       = "ui_clients"
# ---------- APP -------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    r = redis.Redis(host=REDIS_HOST, password=REDIS_PWD, decode_responses=True)
    # create consumer group once
    try:
        await r.xgroup_create(ANALYSIS_STREAM, GROUP, id="0", mkstream=True)
    except redis.ResponseError as e:
        if "BUSYGROUP" not in str(e): raise
    consumer = f"ui_{id(ws)}"
    try:
        while True:
            # block up to 1 s waiting for next message
            msgs = await r.xreadgroup(GROUP, consumer, {ANALYSIS_STREAM: ">"}, count=1, block=1000)
            if msgs:
                for _, entries in msgs:
                    id_, data = entries[0]
                    await ws.send_text(json.dumps(data))
                    await r.xack(ANALYSIS_STREAM, GROUP, id_)
    except (WebSocketDisconnect, asyncio.CancelledError):
        pass
    finally:
        await r.close()

# ---------- DEV STATIC SERVER (optional) ----------
@app.get("/")
async def root():
    with open("../frontend/index.html") as f:
        return HTMLResponse(f.read())

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)

