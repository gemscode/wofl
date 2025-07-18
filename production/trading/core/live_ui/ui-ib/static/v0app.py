document.addEventListener('DOMContentLoaded', () => {
    console.log('=== TRADING APP INITIALIZING ===');
    
    // DOM Elements
    const holdingsEl = document.getElementById('holdings');
    const valueEl = document.getElementById('value');
    const balanceEl = document.getElementById('balance');
    const budgetEl = document.getElementById('budget');
    const analysisDisplayEl = document.getElementById('analysis-display');
    const chartContainer = document.getElementById('chart-container');
    const debugInfoEl = document.getElementById('debug-info');
    const currentPriceEl = document.getElementById('current-price');
    const priceChangeEl = document.getElementById('price-change');
    const bidAskEl = document.getElementById('bid-ask');
    const volumeEl = document.getElementById('volume');
    const highLowEl = document.getElementById('high-low');

    // Button elements
    const simulatorBtn = document.getElementById('simulator-btn');
    const liveBtn = document.getElementById('live-btn');

    // Trading symbol
    const SYMBOL = 'GERN';
    
    // Debug logging
    let debugLog = [];
    function addDebugLog(message) {
        const timestamp = new Date().toLocaleTimeString();
        debugLog.push(`[${timestamp}] ${message}`);
        if (debugLog.length > 10) debugLog.shift();
        if (debugInfoEl) {
            debugInfoEl.textContent = debugLog.join('\n');
        }
        console.log(`DEBUG: ${message}`);
    }

    // Helper functions
    function safeString(value) {
        return value ? String(value) : 'N/A';
    }

    function safeFloat(value) {
        const num = parseFloat(value);
        return isNaN(num) ? 0 : num;
    }

    function safePercent(value) {
        const num = parseFloat(value);
        return isNaN(num) ? '0.00%' : (num * 100).toFixed(2) + '%';
    }

    function formatTimestamp(timestamp) {
        if (!timestamp) return 'N/A';
        return new Date(timestamp).toLocaleString();
    }

    // Market hours checking
    function isMarketOpen() {
        const now = new Date();
        const day = now.getDay();
        const hours = now.getHours() - 2;  // Adjust for MDT to ET
        const minutes = now.getMinutes();
        
        return day >= 1 && day <= 5 && 
               ((hours === 9 && minutes >= 30) || (hours > 9 && hours < 16) || (hours === 16 && minutes === 0));
    }

    // Update button states
    function updateButtons() {
        if (simulatorBtn && liveBtn) {
            if (isMarketOpen()) {
                liveBtn.disabled = false;
            } else {
                liveBtn.disabled = true;
            }
        }
    }

    // Button event listeners
    if (simulatorBtn) {
        simulatorBtn.addEventListener('click', () => {
            simulatorBtn.classList.add('active');
            if (liveBtn) liveBtn.classList.remove('active');
            addDebugLog('Switched to Simulator mode');
        });
    }

    if (liveBtn) {
        liveBtn.addEventListener('click', () => {
            if (!liveBtn.disabled) {
                liveBtn.classList.add('active');
                if (simulatorBtn) simulatorBtn.classList.remove('active');
                addDebugLog('Switched to Live mode');
                window.location.href = '/live-app';
            }
        });
    }

    // Check if library loaded
    if (typeof LightweightCharts === 'undefined') {
        addDebugLog('ERROR: LightweightCharts library failed to load');
        if (analysisDisplayEl) {
            analysisDisplayEl.textContent = 'Chart library failed to load. Please check console for errors.';
        }
        return;
    }

    addDebugLog('LightweightCharts library loaded successfully');

    // Initialize chart
    const chart = LightweightCharts.createChart(chartContainer, {
        width: chartContainer.clientWidth,
        height: chartContainer.clientHeight,
        layout: { 
            background: { type: 'solid', color: '#131722' }, 
            textColor: '#d1d4dc',
            fontSize: 12
        },
        grid: { 
            vertLines: { color: '#2B2B43' }, 
            horzLines: { color: '#2B2B43' } 
        },
        crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
        priceScale: { borderColor: '#485c7b' },
        timeScale: { 
            borderColor: '#485c7b', 
            timeVisible: true, 
            secondsVisible: false 
        },
        handleScroll: {
            vertTouchDrag: true,
            horzTouchDrag: true,
        },
        handleScale: {
            axisPressedMouseMove: true,
            pinch: true,
        }
    });

    // Chart series
    const candleSeries = chart.addCandlestickSeries({
        upColor: '#26a69a',
        downColor: '#ef5350',
        borderVisible: false,
        wickUpColor: '#26a69a',
        wickDownColor: '#ef5350'
    });

    // High/Low lines
    const highLine = chart.addLineSeries({
        color: '#ff6b6b',
        lineWidth: 1,
        lineStyle: LightweightCharts.LineStyle.Dashed,
        priceLineVisible: false,
        lastValueVisible: false
    });

    const lowLine = chart.addLineSeries({
        color: '#4ecdc4',
        lineWidth: 1,
        lineStyle: LightweightCharts.LineStyle.Dashed,
        priceLineVisible: false,
        lastValueVisible: false
    });

    // Buy/Sell/Hold signal markers
    let signalMarkers = [];

    addDebugLog('Chart initialized with candlestick series');

    // Load live data from IB Gateway (ASYNC FUNCTION)
    async function loadLiveData() {
        try {
            addDebugLog(`Fetching live data for ${SYMBOL}`);
            const response = await fetch(`/api/live-data/${SYMBOL}`);
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const data = await response.json();
            addDebugLog(`Received ${data.chart_data?.length || 0} data points`);
            
            // Update chart with historical data
            if (data.chart_data && data.chart_data.length > 0) {
                candleSeries.setData(data.chart_data);
                addDebugLog('Chart updated with live data');
            }
            
            // Update market info in header
            updateMarketInfo(data);
            
            return data;
        } catch (error) {
            addDebugLog(`Error loading live data: ${error.message}`);
            console.error('Error loading live data:', error);
            return null;
        }
    }

    // Update market information display
    function updateMarketInfo(data) {
        if (!data?.market_data) return;
        
        const market = data.market_data;
        
        // Update current price
        if (currentPriceEl) {
            currentPriceEl.textContent = `$${safeFloat(market.last || market.close).toFixed(2)}`;
        }
        
        // Update price change
        if (priceChangeEl && market.close && market.last) {
            const change = market.last - market.close;
            const changePercent = market.close > 0 ? (change / market.close) * 100 : 0;
            const sign = change >= 0 ? '+' : '';
            
            priceChangeEl.textContent = `${sign}${safeFloat(change).toFixed(2)} (${sign}${changePercent.toFixed(2)}%)`;
            priceChangeEl.className = `price-change ${change >= 0 ? 'positive' : 'negative'}`;
        }
        
        // Update bid/ask
        if (bidAskEl) {
            bidAskEl.textContent = `Bid: $${safeFloat(market.bid).toFixed(2)} Ask: $${safeFloat(market.ask).toFixed(2)}`;
        }
        
        // Update volume
        if (volumeEl) {
            volumeEl.textContent = `Vol: ${market.volume?.toLocaleString() || 0}`;
        }
        
        // Update high/low
        if (highLowEl) {
            highLowEl.textContent = `H: $${safeFloat(market.high).toFixed(2)} L: $${safeFloat(market.low).toFixed(2)}`;
        }
        
        addDebugLog(`Market info updated - Price: $${safeFloat(market.last || market.close).toFixed(2)}`);
    }

    // Add buy/sell/hold signal markers
    function addSignalMarker(action, confidence, price, candleTime) {
        let marker = {
            time: candleTime,
            text: '',
            color: '',
            position: '',
            shape: ''
        };
        
        const formattedText = `${safePercent(confidence)} | $${safeFloat(price).toFixed(2)}`;

        switch (action) {
            case 'SELL':
                marker.shape = 'arrowDown';
                marker.color = '#ef5350';
                marker.position = 'aboveBar';
                marker.text = `SELL ↓\n${formattedText}`;
                break;
            case 'BUY':
                marker.shape = 'arrowUp';
                marker.color = '#26a69a';
                marker.position = 'belowBar';
                marker.text = `BUY ↑\n${formattedText}`;
                break;
            case 'HOLD':
                marker.shape = 'circle';
                marker.color = '#2196f3';
                marker.position = 'inBar';
                marker.text = `HOLD -o-\n${formattedText}`;
                break;
            default:
                return;
        }

        signalMarkers.push(marker);
        
        // Keep only last 5 markers
        if (signalMarkers.length > 5) {
            signalMarkers = signalMarkers.slice(-5);
        }
        
        candleSeries.setMarkers(signalMarkers);
        addDebugLog(`Added ${action} signal at time ${candleTime}`);
    }

    // Update high/low horizontal lines
    function updateHighLowLines(high, low) {
        const currentTime = Math.floor(Date.now() / 1000);
        const futureTime = currentTime + 3600 * 24;
        
        highLine.setData([
            { time: currentTime, value: high },
            { time: futureTime, value: high }
        ]);
        
        lowLine.setData([
            { time: currentTime, value: low },
            { time: futureTime, value: low }
        ]);
        
        addDebugLog(`Updated high/low lines: H $${high.toFixed(2)} L $${low.toFixed(2)}`);
    }

    // Connect to Redis analysis stream
    const eventSource = new EventSource('http://localhost:8001/sse/analysis');
    
    eventSource.onopen = () => {
        addDebugLog('SSE connection opened');
    };
    
    eventSource.onmessage = (event) => {
        if (event.data === ':') return;
        
        try {
            const data = JSON.parse(event.data);
            addDebugLog('Received analysis data');
            
            // Update analysis display
            formatAnalysisDisplay(data);
            
            // Add candlestick from analysis data
            if (data.current_price && data.high_price && data.low_price && data.timestamp) {
                const price = safeFloat(data.current_price);
                const high = safeFloat(data.high_price);
                const low = safeFloat(data.low_price);
                const dt = (typeof data.timestamp === 'number')
                    ? data.timestamp
                    : Math.floor(new Date(data.timestamp).getTime()/1000);

                const candle = {
                    time: dt,
                    open: price,
                    high: high,
                    low: low,
                    close: price
                };
                candleSeries.update(candle);
                addDebugLog('Added candlestick from analysis');
            }

            // Add signal marker
            if (data.recommended_action && data.confidence_score && data.current_price && data.timestamp) {
                const dt = (typeof data.timestamp === 'number')
                    ? data.timestamp
                    : Math.floor(new Date(data.timestamp).getTime()/1000);
                addSignalMarker(
                    data.recommended_action,
                    data.confidence_score,
                    data.current_price,
                    dt
                );
            }

            // Update high/low lines
            if (data.high_price && data.low_price) {
                updateHighLowLines(safeFloat(data.high_price), safeFloat(data.low_price));
            }

        } catch (error) {
            addDebugLog(`Parse error: ${error.message}`);
        }
    };
    
    eventSource.onerror = (error) => {
        addDebugLog('SSE error');
    };

    // Format analysis display
    function formatAnalysisDisplay(data) {
        const time = formatTimestamp(data.timestamp);
        const highChange = data.high_distance_pct ? `(${data.high_distance_pct}%)` : '';
        const lowChange = data.low_distance_pct ? `(${data.low_distance_pct}%)` : '';
        
        const formatted = `====================================================
LIVE TRADING ANALYSIS
====================================================
Symbol: ${safeString(data.symbol)}
Time: ${time}
Current Price: $${safeFloat(data.current_price).toFixed(2)}
High (30-min): $${safeFloat(data.high_price).toFixed(2)} ${highChange}
Low (30-min): $${safeFloat(data.low_price).toFixed(2)} ${lowChange}

Recommended Action: ${safeString(data.recommended_action)}
Confidence: ${safeString(data.confidence_level)} (${safePercent(data.confidence_score)})
Signal Type: ${safeString(data.signal_type)}
Guidance: ${safeString(data.guidance)}
MONITOR ONLY

Prediction Breakdown:
• HOLD: ${safePercent(data.hold_prob)}
• BUY: ${safePercent(data.buy_prob)}
• SELL: ${safePercent(data.sell_prob)}

Analysis Details:
• Data Points: ${safeString(data.data_points_used)} (${safeString(data.analysis_window)})
====================================================`;
        
        if (analysisDisplayEl) {
            analysisDisplayEl.textContent = formatted;
        }
        addDebugLog('Analysis updated');
    }

    // Handle window resize
    window.addEventListener('resize', () => {
        chart.resize(chartContainer.clientWidth, chartContainer.clientHeight);
    });

    // Initialize function (ASYNC)
    async function initialize() {
        addDebugLog('Initializing...');
        
        try {
            await loadLiveData();
            setInterval(async () => {
                try {
                    await loadLiveData();
                } catch (error) {
                    addDebugLog(`Periodic data refresh error: ${error.message}`);
                }
            }, 30000);
        } catch (error) {
            addDebugLog(`Initialization error: ${error.message}`);
        }
        
        addDebugLog('Initialized');
    }

    // Initialize buttons and start app
    updateButtons();
    setInterval(updateButtons, 60000);
    
    // Start the app (using IIFE to handle async)
    (async () => {
        try {
            await initialize();
        } catch (error) {
            console.error('App initialization failed:', error);
            addDebugLog(`App initialization failed: ${error.message}`);
        }
    })();
});

