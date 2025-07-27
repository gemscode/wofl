import uvicorn
from fastapi import FastAPI, Request, Body
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from ib_insync import *
import nest_asyncio
import logging
import time
import asyncio
import math  # Added for NaN checking

nest_asyncio.apply()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.mount("/static", StaticFiles(directory="static", html=True), name="static")
templates = Jinja2Templates(directory="templates")

ib = IB()

@app.on_event("startup")
def startup_event():
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            ib.connect('127.0.0.1', 4002, clientId=2, timeout=120)
            ib.reqMarketDataType(3)  # Delayed data mode
            logger.info("IB connection successful on attempt {}".format(attempt + 1))
            return
        except Exception as e:
            logger.error("Connection attempt {} failed: {}".format(attempt + 1, e))
            time.sleep(5)
    logger.warning("All IB connection attempts failed. App starting without IB.")

@app.on_event("shutdown")
def shutdown_event():
    ib.disconnect()
    logger.info("IB connection disconnected.")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/live-data/{symbol}")
async def get_live_data(symbol: str):
    try:
        contract = Stock(symbol, 'SMART', 'USD')
        bars = await ib.reqHistoricalDataAsync(
            contract, endDateTime='', durationStr='2 D', barSizeSetting='1 min',
            whatToShow='TRADES', useRTH=True, formatDate=2
        )
        ticker = ib.reqMktData(contract, '', False, False)
        await asyncio.sleep(1)  # Wait for data

        chart_data = []
        for bar in bars:
            chart_data.append({
                'time': int(bar.date.timestamp()),
                'open': float(bar.open),
                'high': float(bar.high),
                'low': float(bar.low),
                'close': float(bar.close),
                'volume': float(bar.volume)
            })

        market_data = {
            'bid': float(ticker.bid) if not math.isnan(ticker.bid) else 0.0,
            'ask': float(ticker.ask) if not math.isnan(ticker.ask) else 0.0,
            'last': float(ticker.last) if not math.isnan(ticker.last) else 0.0,
            'close': float(ticker.close) if not math.isnan(ticker.close) else 0.0,
            'volume': int(ticker.volume) if not math.isnan(ticker.volume) else 0,
            'high': float(ticker.high) if not math.isnan(ticker.high) else 0.0,
            'low': float(ticker.low) if not math.isnan(ticker.low) else 0.0
        }

        response = {
            'symbol': symbol,
            'current_price': market_data['last'] if market_data['last'] > 0 else market_data['close'],
            'chart_data': chart_data,
            'market_data': market_data
        }

        logger.info("Live data fetched successfully")
        return JSONResponse(response)

    except Exception as e:
        logger.error(f"Error fetching live data: {e}")
        return JSONResponse({'error': str(e)}, status_code=500)

@app.post("/place_order")
async def place_order(data: dict = Body(...)):
    symbol = data.get('symbol', 'GERN')
    quantity = data.get('quantity', 100)
    action = data.get('action', 'BUY')
    contract = Stock(symbol, 'SMART', 'USD')
    order = MarketOrder(action, quantity)
    trade = ib.placeOrder(contract, order)
    return {"status": trade.orderStatus.status}

@app.get("/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

