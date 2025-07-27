document.addEventListener('DOMContentLoaded', function() {
    const WEBSOCKET_URL = "ws://localhost:8000/ws/";
    let socket;
    let currentSymbol = 'GERN';

    const elements = {
        tickerSelect: document.getElementById('ticker-select'),
        analysisContent: document.getElementById('analysis-content'),
        analysisStatus: document.getElementById('analysis-status'),
        priceMarker1: document.getElementById('price-marker-1'),
        priceMarker2: document.getElementById('price-marker-2'),
        priceMarker3: document.getElementById('price-marker-3'),
        priceMarker4: document.getElementById('price-marker-4'),
        trendArrow: document.getElementById('trend-arrow'),
    };

    function connectWebSocket(symbol) {
        if (socket) {
            socket.close();
        }

        console.log(`Connecting to WebSocket for ${symbol}...`);
        socket = new WebSocket(`${WEBSOCKET_URL}${symbol}`);

        socket.onopen = () => {
            console.log(`WebSocket connection opened for ${symbol}`);
            elements.analysisStatus.textContent = `Connected to ${symbol} stream...`;
        };

        socket.onmessage = (event) => {
            const data = JSON.parse(event.data);
            updateDashboard(data);
        };

        socket.onclose = () => {
            console.log('WebSocket connection closed');
            elements.analysisStatus.textContent = 'Connection closed. Retrying...';
            // Optional: implement retry logic
        };

        socket.onerror = (error) => {
            console.error('WebSocket error:', error);
            elements.analysisStatus.textContent = 'Connection error.';
        };
    }

    function updateDashboard(data) {
        // Mock-up of price markers based on image logic
        elements.priceMarker1.textContent = `$${parseFloat(data.price * 1.2).toFixed(1)}`;
        elements.priceMarker3.textContent = `$${parseFloat(data.price * 1.1).toFixed(1)}`;
        elements.priceMarker4.textContent = `$${parseFloat(data.price * 0.9).toFixed(1)}`;
        elements.priceMarker2.textContent = `$${parseFloat(data.price * 0.6).toFixed(1)}`;

        // Update trend arrow (example logic)
        const trend = (data.price > (data.previous_price || data.price)) ? '▲' : '▼';
        const trendColor = (trend === '▲') ? '#4CAF50' : '#d9534f';
        elements.trendArrow.textContent = trend;
        elements.trendArrow.style.color = trendColor;

        // Populate analysis panel
        const analysisHtml = `
Symbol:             ${data.symbol}
Time:               ${new Date(data.timestamp).toLocaleTimeString()}
Current Price:      $${parseFloat(data.price).toFixed(4)}
Recommended Action:   ${data.action}
Confidence:         ${data.confidence_level} (${(data.confidence * 100).toFixed(1)}%)
Signal Type:        ${data.signal_type}
Guidance:           ${data.guidance}

Prediction Breakdown:
HOLD: ${(data.hold_prob * 100).toFixed(1)}%
BUY:  ${(data.buy_prob * 100).toFixed(1)}%
SELL: ${(data.sell_prob * 100).toFixed(1)}%
        `.trim();

        elements.analysisContent.textContent = analysisHtml;
    }

    elements.tickerSelect.addEventListener('change', (event) => {
        currentSymbol = event.target.value;
        connectWebSocket(currentSymbol);
    });

    // Initial connection
    connectWebSocket(currentSymbol);
});

