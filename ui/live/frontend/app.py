// ---------- charts ----------
const volCtx   = document.getElementById('volumeChart');
const priceCtx = document.getElementById('priceChart');

const volumeChart = new Chart(volCtx, {
  type:'bar',
  data:{labels:[],datasets:[{label:'Volume',data:[], backgroundColor:'#007bff'}]},
  options:{responsive:true, plugins:{legend:{display:false}}}
});
const priceChart = new Chart(priceCtx, {
  type:'bar',
  data:{labels:['Current','High','Low'],datasets:[{label:'Price',data:[0,0,0],
        backgroundColor:['#00d95a','#ff4757','#ffa502']}]},
  options:{responsive:true, plugins:{legend:{display:false}}, scales:{y:{beginAtZero:false}}}
});

// ---------- WebSocket ----------
const ws = new WebSocket(`ws://${location.host.replace(/:\d+$/,'')}:8000/ws`);
ws.onopen = () => console.log('🔗 WebSocket connected');
ws.onmessage = ev => updateDashboard(JSON.parse(ev.data));
ws.onclose = () => console.log('❌ WebSocket closed');

// ---------- UI update ----------
function updateDashboard(d){
  document.getElementById('symbol').textContent = d.symbol;

  /* ----- volume chart: shift to keep last 20 bars ----- */
  const lbl = new Date().toLocaleTimeString();
  const vol = parseFloat(d.volume || 0);           // optional field
  volumeChart.data.labels.push(lbl);
  volumeChart.data.datasets[0].data.push(vol);
  if(volumeChart.data.labels.length>20){
      volumeChart.data.labels.shift();
      volumeChart.data.datasets[0].data.shift();
  }
  volumeChart.update();

  /* ----- price bar ----- */
  priceChart.data.datasets[0].data = [
      parseFloat(d.current_price),
      parseFloat(d.high_price),
      parseFloat(d.low_price)
  ];
  priceChart.update();

  /* ----- textual analysis ----- */
  const txt = `
Time: ${new Date(d.timestamp).toLocaleTimeString()}
Action: ${d.action}   Confidence: ${(parseFloat(d.confidence)*100).toFixed(1)}%
Probabilities  H:${(d.hold_prob*100).toFixed(1)}  B:${(d.buy_prob*100).toFixed(1)}  S:${(d.sell_prob*100).toFixed(1)}
Current: $${parseFloat(d.current_price).toFixed(4)}
High:    $${parseFloat(d.high_price).toFixed(4)}
Low:     $${parseFloat(d.low_price).toFixed(4)}
`;
  document.getElementById('analysisText').textContent = txt;
}

