#!/usr/bin/env python3

"""
Optimized Configuration for META - 31.2% Annual Projection
Generated from parameter optimization achieving 1.83% test return
Generated on: 2025-07-26 20:47:10
"""

from dataclasses import dataclass

@dataclass
class OptimizedTradingConfig:
    # EXACT OPTIMIZED PARAMETERS FROM OPTIMIZER RESULTS
    window_minutes: int = 14
    confidence_threshold: float = 0.315285
    base_position_size: float = 0.130882
    max_position_size: float = 0.200000
    profit_target: float = 0.031287
    stop_loss: float = 0.016866
    
    # CRITICAL: Custom label thresholds - MUST BE IMPLEMENTED
    label_threshold_up: float = -0.023788
    label_threshold_down: float = -0.001840
    
    # Performance metrics from optimization
    optimized_return: float = 0.018349
    annual_projection: float = 0.312399
    optimization_date: str = "2025-07-26T20:47:10.454341"
    
    # Standard parameters (can be overridden)
    annual_target_return: float = 0.35
    max_daily_drawdown: float = 0.02
    lookback_periods: int = 100
    forecast_horizon: int = 20
    lstm_hidden_size: int = 128
    lstm_num_layers: int = 3
    dropout: float = 0.2
    data_split_exclude_days: int = 30
    min_volume_ratio: float = 1.2
    max_correlation_threshold: float = 0.7
    end_of_day_close_minutes: int = 15

def get_optimized_config():
    """Factory function to get optimized configuration"""
    return OptimizedTradingConfig()

# Convenience function for backwards compatibility
def load_config_for_symbol(symbol: str):
    """Load optimized config for any symbol"""
    return get_optimized_config()

# Export the optimized parameters as a dictionary for easy access
OPTIMIZED_PARAMS = {
    'window_minutes': 14,
    'confidence_threshold': 0.315285,
    'base_position_size': 0.130882,
    'max_position_size': 0.200000,
    'profit_target': 0.031287,
    'stop_loss': 0.016866,
    'label_threshold_up': -0.023788,
    'label_threshold_down': -0.001840,
}
