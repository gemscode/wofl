// static/app.js

document.addEventListener("DOMContentLoaded", () => {
  const stats = document.getElementById("account-stats");
  const highLabel = document.getElementById("current-high-label");
  const currentPriceBar = document.getElementById("current-price-bar");
  const lastSell = document.getElementById("last-sell-marker");
  const lastBuy = document.getElementById("last-buy-marker");
  const analysis = document.getElementById("analysis-content");
  const status = document.getElementById("status-text");
  const counter = document.getElementById("analysis-counter");

  let analysisCount = 0;

  function generateMockData() {
    const currentPrice = (Math.random() * 2 + 1).toFixed(4);
    const high = (parseFloat(currentPrice) + 0.05).toFixed(4);
    const sell = (parseFloat(currentPrice) + 0.1).toFixed(2);
    const buy = (parseFloat(currentPrice) - 0.1).toFixed(2);

    highLabel.textContent = `$${high}`;
    currentPriceBar.style.height = `${(currentPrice / 2.5) * 100}%`;

    lastSell.style.top = "20%";
    lastSell.querySelector("span").textContent = `$${sell}`;

    lastBuy.style.top = "70%";
    lastBuy.querySelector("span").textContent = `$${buy}`;

    stats.innerHTML = `
      <span>Holdings: 100</span> |
      <span>Value: $5000</span> |
      <span>Balance: $20000</span> |
      <span>Budget: $25000</span>
    `;

    analysis.innerHTML = `
      Symbol: GERN<br>
      Time: ${new Date().toLocaleTimeString()}<br>
      Current Price: $${currentPrice}<br>
      High (30-min): $${high} (-1.1%)<br>
      Low (30-min): ${(parseFloat(currentPrice) - 0.005).toFixed(4)} (+0.4%)<br><br>
      Recommended Action: HOLD<br>
      Confidence: LOW (46.1%)<br>
      Signal Type: MONITOR<br>
      Guidance: Wait for clearer directional signal<br><br>
      Prediction Breakdown:<br>
      HOLD: 98.5%<br>
      BUY: 0.0%<br>
      SELL: 1.5%<br><br>
      Analysis Details:<br>
      Data Points: 30 (30_minutes)<br>
      Agent ID: agent_GERN_1752081698
    `;

    status.textContent = "Analysis complete.";
    counter.textContent = `Analysis #${++analysisCount} received`;
  }

  generateMockData();
  setInterval(generateMockData, 3000);
});

