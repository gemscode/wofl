import uvicorn
import random
import json
import time
from fastapi import FastAPI
from sse_starlette.sse import EventSourceResponse
import redis
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STREAM_NAME = "trading_analysis_GERN"
REDIS_HOST = "trader.wolfx0.com"
REDIS_PORT = 6379

def _read_redis_password():
    """Read Redis password from .redis_passwd file."""
    try:
        with open('.redis_passwd', 'r') as f:
            return f.read().strip() or None
    except FileNotFoundError:
        return None

def parse_nested_json(data, key):
    """Safely parses a string field into a JSON object."""
    if key in data and isinstance(data[key], str):
        try:
            return json.loads(data[key])
        except (json.JSONDecodeError, TypeError):
            return {}
    elif key in data and isinstance(data[key], dict):
        return data[key]
    return {}

def analysis_stream_generator():
    """A stable, synchronous generator that listens to Redis."""
    redis_password = _read_redis_password()
    redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0, password=redis_password, decode_responses=True)
    group_name = "analysis_web_subscribers"
    consumer_name = f"sse_server_consumer_{random.randint(1000, 9999)}"

    try:
        redis_client.xgroup_create(STREAM_NAME, group_name, id='$', mkstream=True)
    except redis.ResponseError:
        pass  # Group already exists

    print("SSE Server is listening for analysis messages...")
    while True:
        try:
            messages = redis_client.xreadgroup(group_name, consumer_name, {STREAM_NAME: '>'}, count=1, block=5000)
            if messages:
                for _, stream_messages in messages:
                    for message_id, data in stream_messages:
                        # Clean up data
                        data['prediction_breakdown'] = parse_nested_json(data, 'prediction_breakdown')
                        data['sell_entry_levels'] = parse_nested_json(data, 'sell_entry_levels')
                        data['profit_targets'] = parse_nested_json(data, 'profit_targets')
                        
                        
                        yield json.dumps(data)
                        redis_client.xack(STREAM_NAME, group_name, message_id)
            else:
                yield ":"
        except Exception as e:
            print(f"Error in SSE loop: {e}. Retrying...")
            time.sleep(5)

@app.get("/sse/analysis")
async def sse_analysis():
    return EventSourceResponse(analysis_stream_generator())

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)

