#!/usr/bin/env python3

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
import redis
import json
import asyncio
from datetime import datetime
import os
from pathlib import Path

app = FastAPI(title="Trading Dashboard", version="1.0.0")

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                pass

manager = ConnectionManager()

class TradingDashboard:
    def __init__(self):
        self.redis_client = self._connect_redis()
        
    def _connect_redis(self):
        try:
            base_dir = Path(__file__).parent
            passwd_file = base_dir / ".redis_passwd"
            
            if passwd_file.exists():
                with open(passwd_file, "r") as f:
                    redis_password = f.read().strip()
            else:
                redis_password = None
            
            client = redis.Redis(
                host="trader.wolfx0.com",
                port=6379,
                db=0,
                password=redis_password,
                decode_responses=True
            )
            client.ping()
            return client
        except Exception as e:
            print(f"Redis connection failed: {e}")
            return None

    async def monitor_analysis_stream(self, symbol: str):
        if not self.redis_client:
            return
        
        stream_name = f"trading_analysis_{symbol}"
        group_name = "dashboard_consumers"
        consumer_name = f"dashboard_{int(datetime.now().timestamp())}"
        
        try:
            self.redis_client.xgroup_create(stream_name, group_name, id='$', mkstream=True)
        except redis.ResponseError:
            pass
        
        while True:
            try:
                messages = self.redis_client.xreadgroup(
                    group_name,
                    consumer_name,
                    {stream_name: '>'},
                    count=1,
                    block=1000
                )
                
                if messages:
                    for stream, stream_messages in messages:
                        for message in stream_messages:
                            message_id, data = message
                            
                            if data.get('analysis_type') == 'TRADING_RECOMMENDATION':
                                analysis_data = {
                                    'type': 'analysis',
                                    'symbol': data['symbol'],
                                    'timestamp': data['timestamp'],
                                    'action': data['recommended_action'],
                                    'confidence': float(data['confidence_score']),
                                    'price': float(data['current_price']),
                                    'signal_type': data['signal_type'],
                                    'guidance': data['guidance']
                                }
                                
                                await manager.broadcast(json.dumps(analysis_data))
                            
                            self.redis_client.xack(stream_name, group_name, message_id)
                
                await asyncio.sleep(0.1)
                
            except Exception as e:
                print(f"Error monitoring stream: {e}")
                await asyncio.sleep(5)

dashboard = TradingDashboard()

@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Trading Dashboard</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background: #1a1a1a; color: #fff; }
            .container { max-width: 1200px; margin: 0 auto; }
            .header { text-align: center; margin-bottom: 30px; }
            .analysis-card { 
                background: #2d2d2d; 
                border-radius: 8px; 
                padding: 20px; 
                margin: 10px 0; 
                border-left: 4px solid #007bff; 
            }
            .price { font-size: 24px; font-weight: bold; color: #00ff00; }
            .action { font-size: 18px; font-weight: bold; }
            .buy { color: #00ff00; }
            .sell { color: #ff4444; }
            .hold { color: #ffaa00; }
            .confidence { font-size: 14px; color: #ccc; }
            .timestamp { font-size: 12px; color: #888; }
            .status { padding: 10px; background: #333; border-radius: 4px; margin-bottom: 20px; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Live Trading Dashboard</h1>
                <div id="status" class="status">Connecting...</div>
            </div>
            <div id="analysis-container"></div>
        </div>

        <script>
            const ws = new WebSocket("ws://localhost:8000/ws/GERN");
            const statusDiv = document.getElementById('status');
            const container = document.getElementById('analysis-container');

            ws.onopen = function(event) {
                statusDiv.innerHTML = "Connected - Waiting for analysis...";
                statusDiv.style.background = "#004400";
            };

            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                
                if (data.type === 'analysis') {
                    const analysisCard = document.createElement('div');
                    analysisCard.className = 'analysis-card';
                    
                    const actionClass = data.action.toLowerCase();
                    const timestamp = new Date(data.timestamp).toLocaleTimeString();
                    
                    analysisCard.innerHTML = `
                        <div class="timestamp">${timestamp}</div>
                        <div class="price">$${parseFloat(data.price).toFixed(4)}</div>
                        <div class="action ${actionClass}">${data.action}</div>
                        <div class="confidence">Confidence: ${(data.confidence * 100).toFixed(1)}%</div>
                        <div style="margin-top: 10px;">
                            <strong>Signal:</strong> ${data.signal_type}<br>
                            <strong>Guidance:</strong> ${data.guidance}
                        </div>
                    `;
                    
                    container.insertBefore(analysisCard, container.firstChild);
                    
                    if (container.children.length > 10) {
                        container.removeChild(container.lastChild);
                    }
                    
                    statusDiv.innerHTML = `Last update: ${timestamp}`;
                }
            };

            ws.onclose = function(event) {
                statusDiv.innerHTML = "Connection closed";
                statusDiv.style.background = "#440000";
            };

            ws.onerror = function(error) {
                statusDiv.innerHTML = "Connection error";
                statusDiv.style.background = "#440000";
            };
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.websocket("/ws/{symbol}")
async def websocket_endpoint(websocket: WebSocket, symbol: str):
    await manager.connect(websocket)
    
    monitor_task = asyncio.create_task(dashboard.monitor_analysis_stream(symbol))
    
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        monitor_task.cancel()

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    uvicorn.run(
        "dashboard_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

