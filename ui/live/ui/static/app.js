document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const holdingsEl = document.getElementById('holdings');
    const valueEl = document.getElementById('value');
    const balanceEl = document.getElementById('balance');
    const budgetEl = document.getElementById('budget');
    // Trend indicator HTML elements
    const trendIndicatorEl = document.getElementById('trend-indicator');
    const trendArrowEl = document.getElementById('trend-arrow');

    const ctx = document.getElementById('volumeChart').getContext('2d');
    
    Chart.register(ChartDataLabels);

    const chartConfig = {
        type: 'bar',
        data: {
            labels: ['Buy', 'Sell', 'Current', 'Historical', 'Stop Sell'],
            datasets: [
                {
                    label: 'Volume Data',
                    data: [],
                    backgroundColor: ['#00FFFF', '#00FF00', '#FFA500', '#D3D3D3', '#FFFF00'],
                    borderColor: 'rgba(0,0,0,0.5)',
                    borderWidth: 2,
                    borderDash: (context) => {
                        const label = context.chart.data.labels[context.dataIndex];
                        return ['Buy', 'Current', 'Historical'].includes(label) ? [6, 6] : [];
                    },
                    barThickness: 20,
                    datalabels: {
                        color: 'black',
                        font: { weight: 'bold' },
                        formatter: (value, context) => {
                            if (context.chart.data.labels[context.dataIndex] === 'Buy') {
                                return Array.isArray(value) ? `$${value[0]}` : '';
                            }
                            return '';
                        },
                        anchor: 'start',
                        align: 'left'
                    }
                }
            ]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: { display: false, beginAtZero: true, max: 2.0 },
                y: {
                    display: true,
                    grid: { display: false },
                    ticks: { color: 'black', font: { weight: 'bold', size: 14 } },
                }
            },
            plugins: {
                legend: { display: false },
                tooltip: { enabled: false },
                annotation: {
                    annotations: {
                        buyLine: createHorizontalLine('Buy'),
                        currentLine: createHorizontalLine('Current'),
                        historicalLine: createHorizontalLine('Historical'),
                        buyPrice: createStaticPriceLabel('Buy'),
                        sellPrice: createStaticPriceLabel('Sell'),
                        currentPrice: createStaticPriceLabel('Current'),
                        historicalPrice: createStaticPriceLabel('Historical'),
                        stopSellPrice: createStaticPriceLabel('Stop Sell'),
                    }
                }
            }
        }
    };
    
    function createHorizontalLine(yValue) {
        return {
            type: 'line',
            yMin: yValue,
            yMax: yValue,
            borderWidth: 2,
            borderDash: [4, 4],
            borderColor: 'rgba(0, 0, 0, 0.4)',
            xMin: 0,
            xMax: 1.7
        };
    }
    
    function createStaticPriceLabel(yValue) {
        return {
            type: 'label',
            yValue: yValue,
            xValue: 1.85,
            font: { weight: 'bold', size: 14 },
            color: 'black',
        };
    }

    const volumeChart = new Chart(ctx, chartConfig);

    async function fetchDataAndUpdate() {
        try {
            const response = await fetch('/api/data');
            const data = await response.json();

            holdingsEl.textContent = data.header.holdings;
            valueEl.textContent = `$${data.header.value.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
            balanceEl.textContent = `$${data.header.balance.toLocaleString()}`;
            budgetEl.textContent = `$${data.header.budget.toLocaleString()}`;

            const chartData = data.chart;
            const values = chartData.values;
            
            volumeChart.data.datasets[0].data = [
                values.buy, values.sell, values.current, values.historical, values.stop_sell
            ];
            
            const annotations = volumeChart.options.plugins.annotation.annotations;
            annotations.buyPrice.content = `$${values.buy[1]}`;
            annotations.sellPrice.content = `$${values.sell}`;
            annotations.currentPrice.content = `$${values.current}`;
            annotations.historicalPrice.content = `$${values.historical}`;
            annotations.stopSellPrice.content = `$${values.stop_sell}`;

            volumeChart.update('none');

            if (volumeChart.getDatasetMeta(0).data.length > 1) {
                trendIndicatorEl.style.display = 'flex';
                const buyBarY = volumeChart.getDatasetMeta(0).data[0].y;
                const sellBarY = volumeChart.getDatasetMeta(0).data[1].y;
                const midPointY = (buyBarY + sellBarY) / 2;
                
                const chartLeft = volumeChart.chartArea.left;
                
                // 1. This is the fix: Adjusting top and left positions with the requested offset.
                trendIndicatorEl.style.top = `${midPointY + 10}px`; // Add 10px to move down
                trendIndicatorEl.style.left = `${chartLeft - 75 + 25}px`; // Add 25px to move right
                
                trendArrowEl.textContent = chartData.trend === 'up' ? '▲' : '▼';
                trendArrowEl.style.color = '#FF4500';
            }

        } catch (error) {
            console.error('Error fetching data:', error);
        }
    }
    
    setInterval(fetchDataAndUpdate, 2000);
    setTimeout(fetchDataAndUpdate, 100);
});

