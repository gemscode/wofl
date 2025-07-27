# Optimized Parameters for META
# Generated on 2025-07-24 22:30:16.137584
# Target: 30% annual return
# Optimization method: genetic

from dataclasses import dataclass

@dataclass
class OptimizedTradingConfig:
    # Optimized parameters
    window_minutes: int = 15
    profit_target: float = 0.0235
    stop_loss: float = 0.0225
    confidence_threshold: float = 0.2171
    position_size: float = 0.0990
    label_threshold_up: float = 0.0057
    label_threshold_down: float = -0.0043
    
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
# trading_system = AdvancedTradingSystem(symbol, config)
