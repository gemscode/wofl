import uvicorn
import random
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# --- Dynamic Data Simulation ---
header_value = 87.14

@app.get("/api/data")
async def get_data():
    """
    API endpoint to provide data for the chart.
    The 'buy' value is now a range [start, end].
    """
    global header_value
    price_change = random.uniform(-0.5, 0.5)
    header_value += price_change
    if header_value < 70 or header_value > 100:
        header_value -= price_change * 2

    return {
        "header": {
            "holdings": 100,
            "value": round(header_value, 2),
            "balance": 20000,
            "budget": 25000,
        },
        "chart": {
            "trend": "up" if price_change >= 0 else "down",
            "values": {
                "buy": [0.88, 1.38],
                "sell": 1.5,
                "current": 1.35,
                "historical": 1.45,
                "stop_sell": 1.3,
            }
        }
    }

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

