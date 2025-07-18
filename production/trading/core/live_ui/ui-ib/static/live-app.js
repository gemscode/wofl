document.addEventListener('DOMContentLoaded', () => {
    const livePriceEl = document.getElementById('live-price');
    const liveChangeEl = document.getElementById('live-change');
    const chartContainer = document.getElementById('live-chart-container');
    const SYMBOL = 'GERN';

    // Initialize live chart
    const chart = LightweightCharts.createChart(chartContainer, {
        width: chartContainer.clientWidth,
        height: chartContainer.clientHeight,
        layout: { background: {type:'solid', color:'#131722'}, textColor:'#d1d4dc' },
        grid: { vertLines:{color:'#2B2B43'}, horzLines:{color:'#2B2B43'} },
        timeScale: { timeVisible:true, secondsVisible:true }
    });

    const lineSeries = chart.addLineSeries({
        color: '#2196f3',
        lineWidth: 2
    });

    let liveData = [];
    let lastPrice = 0;

    // Fetch live data
    async function fetchLiveData() {
        try {
            const response = await fetch(`/api/live-feed/${SYMBOL}`);
            const data = await response.json();
            
            if (data.price) {
                const point = {
                    time: data.timestamp,
                    value: data.price
                };
                
                liveData.push(point);
                if (liveData.length > 100) liveData.shift(); // Keep last 100 points
                
                lineSeries.setData(liveData);
                
                // Update price display
                livePriceEl.textContent = `$${data.price.toFixed(2)}`;
                
                if (lastPrice > 0) {
                    const change = data.price - lastPrice;
                    const changePercent = (change / lastPrice) * 100;
                    const sign = change >= 0 ? '+' : '';
                    
                    liveChangeEl.textContent = `${sign}${change.toFixed(2)} (${sign}${changePercent.toFixed(2)}%)`;
                    liveChangeEl.className = `price-change ${change >= 0 ? 'positive' : 'negative'}`;
                }
                
                lastPrice = data.price;
            }
        } catch (error) {
            console.error('Error fetching live data:', error);
        }
    }

    // Start live data feed
    fetchLiveData();
    setInterval(fetchLiveData, 1000); // Update every second

    // Responsive
    window.addEventListener('resize', () => {
        chart.resize(chartContainer.clientWidth, chartContainer.clientHeight);
    });
});

