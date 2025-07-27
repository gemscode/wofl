document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const holdingsEl = document.getElementById('holdings');
    const valueEl = document.getElementById('value');
    const balanceEl = document.getElementById('balance');
    const budgetEl = document.getElementById('budget');
    const trendIndicatorEl = document.getElementById('trend-indicator');
    const trendArrowEl = document.getElementById('trend-arrow');
    const analysisDisplayEl = document.getElementById('analysis-display');
    const ctx = document.getElementById('volumeChart').getContext('2d');
    
    Chart.register(ChartDataLabels);

    const chartConfig = { type: 'bar', data: { labels: ['Buy', 'Sell', 'Current', 'High', 'Low', 'Stop Sell'], datasets: [{ label: 'Volume Data', data: [], backgroundColor: ['#00FFFF', '#00FF00', '#FFA500', '#D3D3D3', '#A9A9A9', '#FFFF00'], borderColor: 'rgba(0,0,0,0.5)', borderWidth: 2, borderDash: (context) => { const label = context.chart.data.labels[context.dataIndex]; return ['Buy', 'Current', 'High', 'Low'].includes(label) ? [6, 6] : []; }, barThickness: 20, datalabels: { color: 'black', font: { weight: 'bold' }, formatter: (value, context) => { if (context.chart.data.labels[context.dataIndex] === 'Buy') { return Array.isArray(value) ? `$${value[0]}` : ''; } return ''; }, anchor: 'start', align: 'left' } }] }, options: { indexAxis: 'y', responsive: true, maintainAspectRatio: false, scales: { x: { display: false, beginAtZero: true, max: 2.0 }, y: { display: true, grid: { display: false }, ticks: { color: 'black', font: { weight: 'bold', size: 14 } }, } }, plugins: { legend: { display: false }, tooltip: { enabled: false }, annotation: { annotations: { buyLine: createHorizontalLine('Buy'), currentLine: createHorizontalLine('Current'), highLine: createHorizontalLine('High'), lowLine: createHorizontalLine('Low'), buyPrice: createStaticPriceLabel('Buy'), sellPrice: createStaticPriceLabel('Sell'), currentPrice: createStaticPriceLabel('Current'), highPrice: createStaticPriceLabel('High'), lowPrice: createStaticPriceLabel('Low'), stopSellPrice: createStaticPriceLabel('Stop Sell'), } } } } };
    function createHorizontalLine(yValue) { return { type: 'line', yMin: yValue, yMax: yValue, borderWidth: 2, borderDash: [4, 4], borderColor: 'rgba(0, 0, 0, 0.4)', xMin: 0, xMax: 1.7 }; }
    function createStaticPriceLabel(yValue) { return { type: 'label', yValue: yValue, xValue: 1.85, font: { weight: 'bold', size: 14 }, color: 'black', }; }
    const volumeChart = new Chart(ctx, chartConfig);

    // --- This is the fix: State object to hold the last known good values ---
    let chartState = {
        buyProb: 0, sellProb: 0, currentRatio: 0,
        highPrice: 0, lowPrice: 0, stopSellPrice: 0,
        buyLabel: '', sellLabel: '', currentLabel: '',
        highLabel: '', lowLabel: '', stopSellLabel: ''
    };

    const analysisEventSource = new EventSource("http://127.0.0.1:8001/sse/analysis");
    analysisEventSource.onerror = function() { analysisDisplayEl.textContent = "Connection to analysis server lost. Reconnecting..."; };

    analysisEventSource.onmessage = function(event) {
        if (event.data.trim() === '' || event.data.startsWith(':')) return;
        try {
            const analysisData = JSON.parse(event.data);
            updateUI(analysisData);
        } catch (e) {
            analysisDisplayEl.textContent = `Error processing analysis data...\n\n${e}`;
            console.error("SSE Error:", e, "Raw data:", event.data);
        }
    };

    function updateUI(data) {
        analysisDisplayEl.textContent = formatAnalysis(data);

        // Update Header
        const price = parseFloat(data.current_price || chartState.currentPrice || 0);
        holdingsEl.textContent = "100";
        valueEl.textContent = `$${(100 * price).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
        balanceEl.textContent = "$20,000";
        budgetEl.textContent = "$25,000";
        
        // Update state object with new values, keeping old ones if new are missing
        chartState.buyProb = parseFloat(data.buy_prob || chartState.buyProb);
        chartState.sellProb = parseFloat(data.sell_prob || chartState.sellProb);
        chartState.highPrice = parseFloat(data.high_price || chartState.highPrice);
        chartState.lowPrice = parseFloat(data.low_price || chartState.lowPrice);
        
        // Calculate ratios from the stable state object
        if (chartState.highPrice > chartState.lowPrice) {
            chartState.currentRatio = (price - chartState.lowPrice) / (chartState.highPrice - chartState.lowPrice);
        }
        
        // Update chart data from the persistent state
        volumeChart.data.datasets[0].data = [
            chartState.buyProb,
            chartState.sellProb,
            chartState.currentRatio,
            1.0, // High is always 1.0
            0.0, // Low is always 0.0
            0    // Stop Sell is price-based, not a ratio bar for this view
        ];
        
        // Update price labels
        const annotations = volumeChart.options.plugins.annotation.annotations;
        annotations.currentPrice.content = `$${price.toFixed(4)}`;
        if (chartState.highPrice) annotations.highPrice.content = `$${chartState.highPrice.toFixed(4)}`;
        if (chartState.lowPrice) annotations.lowPrice.content = `$${chartState.lowPrice.toFixed(4)}`;
        
        volumeChart.update('none');

        // Update trend indicator
        if (volumeChart.getDatasetMeta(0).data.length > 1) {
            trendIndicatorEl.style.display = 'flex';
            const buyBarY = volumeChart.getDatasetMeta(0).data[0].y;
            const sellBarY = volumeChart.getDatasetMeta(0).data[1].y;
            const midPointY = (buyBarY + sellBarY) / 2;
            const chartLeft = volumeChart.chartArea.left;
            trendIndicatorEl.style.left = `${chartLeft - 75 + 25 + 15}px`; 
            trendIndicatorEl.style.top = `${midPointY + 10}px`;
            const isUp = chartState.buyProb > chartState.sellProb;
            trendArrowEl.textContent = isUp ? '▲' : '▼';
            trendArrowEl.style.color = isUp ? '#00FFFF' : '#00FF00';
        }
    }

    function formatAnalysis(data) {
        // This function formats the text for the bottom panel and remains the same.
        const safeFloat = (val, dec = 4) => { const f = parseFloat(val); return isNaN(f) ? '--' : f.toFixed(dec); };
        const safePercent = (val, sign = false) => { const f = parseFloat(val); const prefix = sign && f > 0 ? '+' : ''; return isNaN(f) ? '--' : `${prefix}${(f * 100).toFixed(1)}%`; };
        const time = data.timestamp ? new Date(data.timestamp).toLocaleTimeString() : '--';
        const action = data.recommended_action || 'N/A';
        const confidence = data.confidence_level || 'N/A';
        const confidenceScore = safePercent(data.confidence_score);
        const signal = data.signal_type || 'N/A';
        const guidance = data.guidance || 'N/A';
        const actionable = data.actionable === 'true' ? '**ACTIONABLE SIGNAL**' : 'MONITOR ONLY';

        return `======================================================================
🤖 LIVE TRADING ANALYSIS
======================================================================
📊 Symbol: ${data.symbol || 'N/A'}
⏰ Time: ${time}
💰 Current Price: $${safeFloat(data.current_price)}
📈 High (30-min): $${safeFloat(data.high_price)} (${safePercent(data.high_distance_pct, true)})
📉 Low (30-min): $${safeFloat(data.low_price)} (${safePercent(data.low_distance_pct, true)})

🎯 Recommended Action: ${action}
📈 Confidence: ${confidence} (${confidenceScore})
🔍 Signal Type: ${signal}
💡 Guidance: ${guidance}
⚠️ ${actionable}

📊 Prediction Breakdown:
   • HOLD: ${safePercent(data.hold_prob)}
   • BUY:  ${safePercent(data.buy_prob)}
   • SELL: ${safePercent(data.sell_prob)}

📈 Analysis Details:
   • Data Points: ${data.data_points_used || 'N/A'} (${data.analysis_window || 'N/A'})
   • Near High: ${data.near_high || 'N/A'}
   • Near Low: ${data.near_low || 'N/A'}
   • High Confidence: ${data.high_confidence || 'N/A'}
   • Agent ID: ${data.agent_id || 'N/A'}
   • Model Version: ${data.model_version || 'N/A'}
======================================================================
`.trim();
    }
});
