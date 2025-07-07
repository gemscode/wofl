#!/usr/bin/env python3
"""
Main script to run the live trading system
"""

from ensemble_integration import LiveTradingSystem

def main():
    # 1. Initialize the system
    live_system = LiveTradingSystem("GERN")
    ensemble = live_system.initialize_ensemble()
    
    # 2. Example market data
    market_data = {
        'price': 1.45,
        'sma_5': 1.44,
        'sma_10': 1.43,
        'sma_20': 1.42,
        'sma_50': 1.40,
        'sma_200': 1.35,
        'rsi_14': 65,
        'volume': 15000,
        'avg_volume_20': 12000,
        'yearly_high': 1.60,
        'yearly_low': 1.20,
        'intraday_high': 1.46,
        'intraday_low': 1.43,
        'price_1h_ago': 1.44,
        'price_4h_ago': 1.43
    }
    
    # 3. Make trading decisions
    decision = live_system.make_live_trading_decision(market_data)
    print(f"Trading Decision: {decision}")
    
    # 4. Update performance after trades (simulate)
    actual_return = 0.015  # 1.5% return
    live_system.update_performance(actual_return)
    
    # 5. Get system status
    status = live_system.get_system_status()
    print(f"System Status: {status}")

if __name__ == "__main__":
    main()

