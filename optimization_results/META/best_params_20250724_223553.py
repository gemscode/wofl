# Optimized Parameters for META
# Generated on 2025-07-24 22:35:53.267447
# Target: 30% annual return
# Optimization method: genetic

from dataclasses import dataclass

@dataclass
class OptimizedTradingConfig:
    # Optimized parameters
    window_minutes: int = 10
    profit_target: float = 0.0178
    stop_loss: float = 0.0050
    confidence_threshold: float = 0.0586
    base_position_size: float = 0.1155
    max_position_size: float = 0.1500
    label_threshold_up: float = 0.0015
    label_threshold_down: float = -0.0066
    
    # Fixed parameters (from original config)
    annual_target_return: float = 0.30
    max_daily_drawdown: float = 0.02
    lookback_periods: int = 100
    forecast_horizon: int = 20
    lstm_hidden_size: int = 128
    lstm_num_layers: int = 3
    dropout: float = 0.2

# Usage:
# config = OptimizedTradingConfig()
# trading_system = EnhancedAdvancedTradingSystem(symbol, config)
