document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const analysisDisplayEl = document.getElementById('analysis-display');
    const chartContainer = document.getElementById('chart-container');
    const debugInfoEl = document.getElementById('debug-info');
    const SYMBOL = 'GERN';

    // Debug log setup
    let debugLog = [];
    function addDebugLog(message) {
        const t = new Date().toLocaleTimeString();
        debugLog.push(`[${t}] ${message}`);
        if (debugLog.length > 12) debugLog.shift();
        if (debugInfoEl) debugInfoEl.textContent = debugLog.join('\n');
    }

    // Helper functions
    function safeString(value) { return value ? String(value) : 'N/A'; }
    function safeFloat(value) { const n = parseFloat(value); return isNaN(n) ? 0 : n; }
    function safePercent(value) { const n = parseFloat(value); return isNaN(n) ? '0.0%' : (n * 100).toFixed(1) + '%'; }
    function formatTimestamp(ts) { if (!ts) return 'N/A'; return new Date(ts).toLocaleString(); }

    // Lightweight Charts setup
    if (typeof LightweightCharts === 'undefined') {
        if (analysisDisplayEl) analysisDisplayEl.textContent = 'Chart library failed to load. Please check console for errors.';
        return;
    }

    const chart = LightweightCharts.createChart(chartContainer, {
        width: chartContainer.clientWidth,
        height: chartContainer.clientHeight,
        layout: { background: {type:'solid', color:'#131722'}, textColor:'#d1d4dc', fontSize:12 },
        grid: { vertLines:{color:'#2B2B43'}, horzLines:{color:'#2B2B43'} },
        crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
        priceScale: { borderColor:'#485c7b' },
        timeScale: { borderColor:'#485c7b', timeVisible:true, secondsVisible:false }
    });

    const candleSeries = chart.addCandlestickSeries({
        upColor:'#26a69a', downColor:'#ef5350', borderVisible:false,
        wickUpColor:'#26a69a', wickDownColor:'#ef5350'
    });

    const highLine = chart.addLineSeries({ color:'#ff6b6b', lineWidth:1, lineStyle:LightweightCharts.LineStyle.Dotted });
    const lowLine  = chart.addLineSeries({ color:'#4ecdc4', lineWidth:1, lineStyle:LightweightCharts.LineStyle.Dotted });

    // Dynamic chart state
    let chartCandles = [];
    let signalMarkers = [];

    // Load historical IB candles initially to seed the chart
    async function seedIBData() {
        try {
            const rsp = await fetch(`/api/live-data/${SYMBOL}`);
            if (!rsp.ok) throw new Error(`HTTP ${rsp.status}`);
            const data = await rsp.json();
            if (data.chart_data && data.chart_data.length) {
                chartCandles = data.chart_data;
                candleSeries.setData(chartCandles); // Seed chart
                addDebugLog('Chart seeded with IB data');
            }
        } catch (e) {
            addDebugLog('Failed to seed IB data: ' + e.message);
        }
    }

    // Signal display logic
    function addSignalMarker(signalType, percent, price) {
        const lastCandle = chartCandles.length > 0 ? chartCandles[chartCandles.length-1] : null;
        const currentTime = lastCandle ? lastCandle.time : Math.floor(Date.now()/1000);

        let marker = { time: currentTime, text: '', color: '', position:'', shape:'' };

        if (signalType === 'SELL') {
            marker.shape = 'arrowDown';
            marker.color = '#ef5350';
            marker.position = 'aboveBar';
            marker.text = `SELL ↓ ${safePercent(percent)} | $${safeFloat(price).toFixed(2)}`;
        } else if (signalType === 'BUY') {
            marker.shape = 'arrowUp';
            marker.color = '#26a69a';
            marker.position = 'belowBar';
            marker.text = `BUY ↑ ${safePercent(percent)} | $${safeFloat(price).toFixed(2)}`;
        } else if (signalType === 'HOLD') {
            marker.shape = 'circle';
            marker.color = '#2196f3';
            marker.position = 'inBar';
            marker.text = `HOLD -o- ${safePercent(percent)} | $${safeFloat(price).toFixed(2)}`;
        } else {
            return;
        }

        signalMarkers.push(marker);
        // Clean up: only last 5 signals
        if (signalMarkers.length > 5) signalMarkers = signalMarkers.slice(-5);
        candleSeries.setMarkers(signalMarkers);
        addDebugLog(`Signal shown: ${marker.text}`);
    }

    // Update high/low lines
    function updateHighLowLines(high, low, time) {
        const right = (chartCandles.length > 0) ? Math.max(time, chartCandles[chartCandles.length-1].time+3600) : time+3600;
        highLine.setData([{ time, value: safeFloat(high) }, { time: right, value: safeFloat(high) }]);
        lowLine.setData([{ time, value: safeFloat(low) }, { time: right, value: safeFloat(low) }]);
        addDebugLog(`High/low lines: $${high} | $${low}`);
    }

    // Format analysis display (no change)
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
        analysisDisplayEl.textContent = formatted;
    }

    // SSE analysis consumer
    const eventSource = new EventSource('http://localhost:8001/sse/analysis');
    eventSource.onmessage = (event) => {
        if (event.data === ':') return;

        try {
            const data = JSON.parse(event.data);

            // 1. Add candlestick (if analysis provides price information)
            if (data.current_price && data.high_price && data.low_price && data.timestamp) {
                const price = safeFloat(data.current_price);
                const high = safeFloat(data.high_price);
                const low = safeFloat(data.low_price);
                const dt = (typeof data.timestamp === 'number')
                    ? data.timestamp
                    : Math.floor(new Date(data.timestamp).getTime()/1000);

                const candle = {
                    time: dt,
                    open: price,  // Best effort; you can provide open/close/volume if available
                    high: high,
                    low: low,
                    close: price
                };
                chartCandles.push(candle);
                candleSeries.update(candle);  // This makes the chart dynamic as new analysis arrives

                // 2. Update high/low lines at this bar
                updateHighLowLines(high, low, dt);
            }

            // 3. Show signal marker if present
            if (data.recommended_action && data.confidence_score && data.current_price) {
                addSignalMarker(
                    data.recommended_action,
                    data.confidence_score,
                    data.current_price
                );
            }

            // 4. Always update the analysis display
            formatAnalysisDisplay(data);

        } catch (error) {
            addDebugLog(`Parse error: ${error.message}`);
        }
    };

    eventSource.onerror = (error) => {
        addDebugLog('SSE error');
    };

    // Responsive
    window.addEventListener('resize', () => {
        chart.resize(chartContainer.clientWidth, chartContainer.clientHeight);
    });

    // Main init: Seed IB data, then wait for analysis to drive updates
    seedIBData();
});

