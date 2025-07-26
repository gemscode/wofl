#!/usr/bin/env python3
"""
Optimized Configuration for META - 55.4% Annual Projection
Generated from parameter optimization achieving 2.98% test return
"""

from dataclasses import dataclass

@dataclass
class OptimizedTradingConfig:
    # EXACT OPTIMIZED PARAMETERS FROM YOUR RESULTS
    window_minutes: int = 13
    confidence_threshold: float = 0.1981
    base_position_size: float = 0.1148
    max_position_size: float = 0.2000
    profit_target: float = 0.0253
    stop_loss: float = 0.0040
    
    # Critical: Custom label thresholds (NEED IMPLEMENTATION)
    label_threshold_up: float = -0.0252    # NEGATIVE - contrarian strategy!
    label_threshold_down: float = -0.0016
    
    # Standard parameters
    annual_target_return: float = 0.35
    max_daily_drawdown: float = 0.02
    lookback_periods: int = 100
    forecast_horizon: int = 20
    lstm_hidden_size: int = 128
    lstm_num_layers: int = 3
    dropout: float = 0.2

# Usage function
def get_optimized_config():
    return OptimizedTradingConfig()

