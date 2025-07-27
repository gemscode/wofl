import uvicorn
from fastapi import FastAPI, Request, Body
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from ib_insync import *
import nest_asyncio
import logging
import asyncio

nest_asyncio.apply()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.mount("/static", StaticFiles(directory="static", html=True), name="static")
templates = Jinja2Templates(directory="templates")

ib = IB()

@app.on_event("startup")
async def startup_event():
    for attempt in range(3):
        try:
            ib.connect('127.0.0.1', 4002, clientId=2)
            ib.reqMarketDataType(3)  # delayed data
            logger.info("Connected to IB Gateway with clientId=2")
            return
        except Exception as e:
            logger.error(f"Failed to connect to IB Gateway: {e}")
            await asyncio.sleep(5)
    logger.error("Could not connect to IB Gateway after 3 attempts")

@app.on_event("shutdown")
async def shutdown_event():
    if ib.isConnected():
        ib.disconnect()
        logger.info("Disconnected from IB Gateway")

@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/live-data/{symbol}")
async def live_data(symbol: str):
    try:
        contract = Stock(symbol, 'SMART', 'USD')
        bars = await ib.reqHistoricalDataAsync(
            contract,
            endDateTime='',
            durationStr='2 D',
            barSizeSetting='1 min',
            whatToShow='TRADES',
            useRTH=True,
            formatDate=1
        )

        bars_list = []
        for bar in bars:
            bar_dict = {
                'time': int(bar.date.timestamp()),
                'open': bar.open,
                'high': bar.high,
                'low': bar.low,
                'close': bar.close,
                'volume': bar.volume
            }
            bars_list.append(bar_dict)

        logger.info(f"Received {len(bars_list)} bars from IB Gateway for {symbol}")
        logger.info(f"Bars data: {bars_list}")

        ticker = ib.reqMktData(contract, '', False, False)
        await asyncio.sleep(2)

        market_data = {
            'bid': ticker.bid if ticker.bid else 0,
            'ask': ticker.ask if ticker.ask else 0,
            'last': ticker.last if ticker.last else 0,
            'close': ticker.close if ticker.close else 0,
            'volume': ticker.volume if ticker.volume else 0,
            'high': ticker.high if ticker.high else 0,
            'low': ticker.low if ticker.low else 0
        }

        logger.info(f"Market data: {market_data}")

        response = {
            'symbol': symbol,
            'current_price': market_data['last'] if market_data['last'] > 0 else market_data['close'],
            'chart_data': bars_list,
            'market_data': market_data
        }

        return JSONResponse(response)

    except Exception as e:
        logger.error(f"Error fetching live data for {symbol}: {e}")
        return JSONResponse({'error': str(e)}, status_code=500)

@app.post("/api/place-order")
async def place_order(data: dict = Body(...)):
    try:
        symbol = data.get('symbol', 'GERN')
        quantity = data.get('quantity', 100)
        action = data.get('action', 'BUY')
        contract = Stock(symbol, 'SMART', 'USD')
        order = MarketOrder(action, quantity)
        trade = ib.placeOrder(contract, order)
        return {'status': trade.orderStatus.status}
    except Exception as e:
        logger.error(f"Error placing order: {e}")
        return JSONResponse({'error': str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

