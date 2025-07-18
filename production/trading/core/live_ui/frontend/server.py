#!/usr/bin/env python3

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
import redis
import json
import asyncio
from datetime import datetime
from pathlib import Path

# --- Configuration ---
REDIS_HOST = "trader.wolfx0.com"
REDIS_PORT = 6379
REDIS_DB = 0

# --- FastAPI App Setup ---
app = FastAPI(title="Trading Dashboard Backend", version="1.0")
STATIC_DIR = Path(__file__).parent / "frontend"

class ConnectionManager:
    """Manages active WebSocket connections."""
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception:
                self.disconnect(connection)

manager = ConnectionManager()

def get_redis_client():
    """Establishes a connection to Redis using the password file."""
    try:
        passwd_file = Path(__file__).parent / ".redis_passwd"
        if not passwd_file.exists():
            print("ERROR: .redis_passwd file not found.")
            return None
        
        with open(passwd_file, "r") as f:
            redis_password = f.read().strip()

        client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            password=redis_password,
            decode_responses=True
        )
        client.ping()
        print(f"Successfully connected to Redis at {REDIS_HOST}")
        return client
    except Exception as e:
        print(f"FATAL: Could not connect to Redis. Error: {e}")
        return None

async def redis_stream_listener(symbol: str):
    """Monitors a Redis stream for a specific symbol and broadcasts messages."""
    redis_client = get_redis_client()
    if not redis_client:
        return

    stream_name = f"trading_analysis_{symbol.upper()}"
    group_name = "dashboard_consumers"
    consumer_name = f"dashboard_{int(datetime.now().timestamp())}"

    try:
        redis_client.xgroup_create(stream_name, group_name, id='$', mkstream=True)
    except redis.ResponseError as e:
        if "BUSYGROUP" not in str(e):
            print(f"Could not create consumer group: {e}")

    print(f"Listening to Redis stream: {stream_name}")
    while True:
        try:
            messages = redis_client.xreadgroup(
                group_name,
                consumer_name,
                {stream_name: '>'},
                count=1,
                block=1000
            )
            if messages:
                for _, stream_messages in messages:
                    for message_id, data in stream_messages:
                        await manager.broadcast(json.dumps(data))
                        redis_client.xack(stream_name, group_name, message_id)
            await asyncio.sleep(0.1) # Prevent tight loop
        except Exception as e:
            print(f"Error while listening to Redis stream: {e}")
            await asyncio.sleep(5) # Wait before retrying

# --- API Endpoints ---

@app.websocket("/ws/{symbol}")
async def websocket_endpoint(websocket: WebSocket, symbol: str):
    """WebSocket endpoint to stream data for a given symbol."""
    await manager.connect(websocket)
    # Start a new listener task for the requested symbol
    listener_task = asyncio.create_task(redis_stream_listener(symbol))
    try:
        while True:
            # Keep the connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        listener_task.cancel() # Stop the listener when client disconnects

@app.get("/")
async def get_index():
    """Serves the main HTML file."""
    return FileResponse("index.html")

@app.get("/{filepath:path}")
async def get_static_files(filepath: str):
    """Serves static files (CSS, JS)."""
    file_path = STATIC_DIR / filepath
    if file_path.exists():
        return FileResponse(file_path)
    return {"error": "File not found"}, 404

if __name__ == "__main__":
    print("Starting Trading Dashboard Server...")
    print("Access the UI at http://127.0.0.1:8001")
    uvicorn.run("dashboard_server:app", host="0.0.0.0", port=8002, reload=True)


