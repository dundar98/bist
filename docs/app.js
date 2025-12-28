
// BIST100 AI Dashboard Logic

document.addEventListener('DOMContentLoaded', () => {
    fetchData();
    
    document.getElementById('refresh-btn').addEventListener('click', fetchData);
});

async function fetchData() {
    try {
        const response = await fetch('dashboard_data.json');
        if (!response.ok) throw new Error('Veri yüklenemedi');
        
        const data = await response.json();
        updateDashboard(data);
    } catch (error) {
        console.error('Error:', error);
        // On local, fallback to dummy data if json missing
        updateDashboard(getDummyData());
    }
}

function updateDashboard(data) {
    // 1. Update Header
    document.getElementById('last-update').textContent = `Son Güncelleme: ${data.scan_date}`;
    
    // 2. Update Stats
    animateValue('total-scanned', data.total_scanned);
    animateValue('buy-count', data.buy_count);
    animateValue('sell-count', data.sell_count);
    document.getElementById('market-volatility').textContent = data.market_volatility || 'Normal';
    
    // 3. Populate Table
    const tbody = document.querySelector('#signals-table tbody');
    tbody.innerHTML = '';
    
    data.buy_signals.slice(0, 10).forEach(signal => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td style="font-weight: 600;">${signal.symbol}</td>
            <td>${signal.current_price.toFixed(2)} ₺</td>
            <td class="signal-buy"><i class="fa-solid fa-arrow-up"></i> AL</td>
            <td>
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <div style="flex: 1; height: 6px; background: rgba(255,255,255,0.1); border-radius: 3px;">
                        <div style="width: ${signal.probability * 100}%; height: 100%; background: var(--accent-green); border-radius: 3px;"></div>
                    </div>
                    ${(signal.probability * 100).toFixed(0)}%
                </div>
            </td>
            <td>${signal.rsi.toFixed(0)}</td>
            <td><span class="badge">Aktif</span></td>
        `;
        tbody.appendChild(row);
    });
    
    // 4. Update Chart
    updateChart(data);
    
    // 5. Populate Details
    const detailsContainer = document.getElementById('signal-details');
    detailsContainer.innerHTML = '';
    
    [...data.buy_signals, ...data.sell_signals].forEach(signal => {
        const div = document.createElement('div');
        div.className = `detail-item ${signal.signal.toLowerCase()}`;
        div.innerHTML = `
            <div class="detail-header">
                <span class="detail-symbol">${signal.symbol}</span>
                <span class="${signal.signal === 'BUY' ? 'signal-buy' : 'signal-sell'}">${signal.signal}</span>
            </div>
            <div class="detail-reason">${signal.reason}</div>
            <div style="margin-top: 0.5rem; font-size: 0.8rem; color: var(--text-secondary);">
                RSI: ${signal.rsi.toFixed(0)} | Vol: ${signal.volatility.toFixed(4)}
            </div>
        `;
        detailsContainer.appendChild(div);
    });
}

function updateChart(data) {
    const ctx = document.getElementById('marketChart').getContext('2d');
    
    // Calculate simple distribution
    const buy = data.buy_count;
    const sell = data.sell_count;
    const hold = data.hold_count;
    
    if (window.marketChartInstance) {
        window.marketChartInstance.destroy();
    }
    
    window.marketChartInstance = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['AL', 'SAT', 'BEKLE'],
            datasets: [{
                data: [buy, sell, hold],
                backgroundColor: [
                    'rgba(16, 185, 129, 0.8)',
                    'rgba(239, 68, 68, 0.8)',
                    'rgba(148, 163, 184, 0.2)'
                ],
                borderColor: 'transparent',
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: { color: '#94a3b8' }
                }
            }
        }
    });
}

function animateValue(id, value) {
    const obj = document.getElementById(id);
    obj.textContent = value;
}

function getDummyData() {
    return {
        "scan_date": "2025-01-01 (Demo)",
        "total_scanned": 100,
        "buy_count": 5,
        "sell_count": 3,
        "hold_count": 92,
        "market_volatility": "Düşük",
        "buy_signals": [
            {
                "symbol": "THYAO",
                "current_price": 285.50,
                "signal": "BUY",
                "probability": 0.85,
                "rsi": 45,
                "volatility": 0.012,
                "reason": "Yapay zeka modeli güçlü yükseliş öngörüyor"
            },
            {
                "symbol": "AKBNK",
                "current_price": 42.10,
                "signal": "BUY",
                "probability": 0.76,
                "rsi": 52,
                "volatility": 0.015,
                "reason": "Trend takibi al veriyor"
            }
        ],
        "sell_signals": [],
        "hold_signals": []
    };
}
