#!/usr/bin/env python3
"""
WolfXE Activity Enforcer
Enhanced activity enforcement with self-attention mechanism
"""

import torch
import torch.nn as nn
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class ActivityEnforcerSelfAttention(nn.Module):
    """Self-attention network for activity enforcement"""
    
    def __init__(self, input_dim=25, hidden_dim=128, num_heads=8):
        super(ActivityEnforcerSelfAttention, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Input processing layers
        self.market_encoder = nn.Linear(15, hidden_dim // 2)
        self.activity_encoder = nn.Linear(10, hidden_dim // 2)
        
        # Self-attention mechanism
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.1
        )
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # Activity pressure calculation
        self.activity_pressure = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Trading urgency levels
        self.urgency_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 3),  # LOW, MEDIUM, HIGH
            nn.Softmax(dim=-1)
        )
        
        # Override signal generator
        self.override_generator = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 3),  # HOLD, BUY, SELL
            nn.Softmax(dim=-1)
        )
    
    def forward(self, market_features, activity_features):
        """Forward pass through activity enforcer"""
        # Encode features
        market_encoded = self.market_encoder(market_features)
        activity_encoded = self.activity_encoder(activity_features)
        
        # Combine features
        combined = torch.cat([market_encoded, activity_encoded], dim=-1)
        combined = self.layer_norm1(combined)
        
        # Self-attention
        attn_output, attention_weights = self.self_attention(combined, combined, combined)
        combined = combined + attn_output  # Residual connection
        
        # Feed-forward
        ff_output = self.feed_forward(combined)
        combined = self.layer_norm2(combined + ff_output)
        
        # Global pooling
        pooled = torch.mean(combined, dim=1)
        
        # Generate outputs
        pressure = self.activity_pressure(pooled)
        urgency = self.urgency_classifier(pooled)
        override_signal = self.override_generator(pooled)
        
        return pressure, urgency, override_signal, attention_weights

class WolfXEActivityEnforcer:
    """Enhanced activity enforcer with self-attention"""
    
    def __init__(self, stock_symbol, min_trades_per_week=3, max_days_without_trade=6):
        self.stock_symbol = stock_symbol.upper()
        self.min_trades_per_week = min_trades_per_week
        self.max_days_without_trade = max_days_without_trade
        
        # Self-attention model
        self.attention_model = ActivityEnforcerSelfAttention()
        self.optimizer = torch.optim.AdamW(self.attention_model.parameters(), lr=0.001)
        self.criterion_mse = nn.MSELoss()
        self.criterion_ce = nn.CrossEntropyLoss()
        
        # Activity tracking
        self.activity_metrics = {
            'days_since_last_trade': 0,
            'trades_this_week': 0,
            'week_start': None,
            'total_trades': 0,
            'forced_trades': 0,
            'activity_score': 1.0,
            'performance_score': 0.5
        }
        
        # Learning history
        self.training_history = []
        self.is_trained = False
        
        print(f"WolfXE Activity Enforcer initialized for {self.stock_symbol}")
        print(f"Min trades/week: {min_trades_per_week}, Max days without trade: {max_days_without_trade}")
    
    def extract_market_features(self, data):
        """Extract market condition features"""
        if len(data) < 20:
            return np.zeros(15, dtype=np.float32)
        
        close_prices = data['Close'].values
        volumes = data['Volume'].values
        features = []
        
        # Price momentum (5 features)
        for period in [3, 5, 10, 20, 50]:
            if len(close_prices) >= period:
                momentum = (close_prices[-1] - close_prices[-period]) / close_prices[-period]
                features.append(momentum)
            else:
                features.append(0.0)
        
        # Volatility (3 features)
        for window in [5, 10, 20]:
            if len(close_prices) >= window:
                returns = np.diff(close_prices[-window:]) / close_prices[-window:-1]
                volatility = np.std(returns)
                features.append(volatility)
            else:
                features.append(0.0)
        
        # Volume analysis (3 features)
        for period in [5, 10, 20]:
            if len(volumes) >= period:
                avg_volume = np.mean(volumes[-period:])
                volume_ratio = volumes[-1] / avg_volume if avg_volume > 0 else 1.0
                features.append(min(volume_ratio, 3.0))
            else:
                features.append(1.0)
        
        # Technical indicators (4 features)
        if len(close_prices) >= 20:
            sma_20 = np.mean(close_prices[-20:])
            sma_5 = np.mean(close_prices[-5:])
            features.extend([
                (close_prices[-1] - sma_20) / sma_20,
                (sma_5 - sma_20) / sma_20,
                (np.max(close_prices[-10:]) - np.min(close_prices[-10:])) / close_prices[-1],
                np.mean(np.diff(close_prices[-5:]) / close_prices[-5:-1])
            ])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])
        
        return np.array(features[:15], dtype=np.float32)
    
    def extract_activity_features(self):
        """Extract activity and performance features"""
        features = [
            # Activity pressure indicators
            self.activity_metrics['days_since_last_trade'] / self.max_days_without_trade,
            max(0, (self.min_trades_per_week - self.activity_metrics['trades_this_week']) / self.min_trades_per_week),
            self.activity_metrics['activity_score'],
            self.activity_metrics['performance_score'],
            
            # Historical patterns
            self.activity_metrics['forced_trades'] / max(1, self.activity_metrics['total_trades']),
            min(self.activity_metrics['total_trades'] / 50.0, 1.0),
            
            # Time factors
            self.get_day_of_week_factor(),
            self.get_time_urgency_factor(),
            
            # Performance factors
            self.get_recent_performance_factor(),
            
            # Urgency flag
            1.0 if self.should_force_trade() else 0.0
        ]
        
        return np.array(features[:10], dtype=np.float32)
    
    def get_day_of_week_factor(self):
        """Get day of week trading factor"""
        day = datetime.now().weekday()
        # Higher activity mid-week
        factors = [0.7, 0.9, 1.0, 0.9, 0.8]  # Mon-Fri
        return factors[min(day, 4)]
    
    def get_time_urgency_factor(self):
        """Calculate time-based urgency"""
        current_day = datetime.now().weekday()
        if current_day >= 3:  # Thursday/Friday
            return (current_day - 2) / 3.0
        return 0.0
    
    def get_recent_performance_factor(self):
        """Get recent performance factor"""
        if len(self.training_history) < 5:
            return 0.5
        
        recent_performance = [h.get('performance', 0) for h in self.training_history[-5:]]
        avg_performance = np.mean(recent_performance)
        return max(0.0, min(1.0, (avg_performance + 1.0) / 2.0))
    
    def should_force_trade(self):
        """CORRECTED: Less aggressive force trade logic"""
        # Only force if REALLY necessary
        if self.activity_metrics['days_since_last_trade'] >= 5:  # Increased back up
            return True
        
        # Don't force trades just for weekly targets in simulation
        return False
    
    def calculate_activity_pressure(self):
        """CORRECTED: Less pressure for better trades"""
        pressure = 0.0
        
        # Much less aggressive pressure
        days_pressure = self.activity_metrics['days_since_last_trade'] / 7.0
        pressure += min(days_pressure, 0.5) * 0.3  # Reduced weight
        
        # Reduced weekly pressure
        trade_deficit = max(0, 2 - self.activity_metrics['trades_this_week'])  # Target 2/week
        weekly_pressure = trade_deficit / 2.0
        pressure += weekly_pressure * 0.2  # Reduced weight
        
        return min(pressure, 0.6)  # Cap at 60%
    
    def enforce_activity(self, market_data, base_decisions, current_position):
        """Main enforcement function with corrected logic"""
        # Extract features
        market_features = self.extract_market_features(market_data)
        activity_features = self.extract_activity_features()
        
        # Calculate current pressure
        current_pressure = self.calculate_activity_pressure()
        
        # Use rule-based enforcement for now (more reliable)
        return self.rule_based_enforcement(base_decisions, current_pressure, current_position)
    
    def rule_based_enforcement(self, base_decisions, pressure, current_position):
        """CORRECTED: More conservative rule-based enforcement"""
        consensus = base_decisions.get('consensus', 'HOLD')
        
        # Much more conservative pressure thresholds
        if pressure > 0.7 or self.should_force_trade():  # High threshold
            decision = 'BUY' if current_position == 0 else 'SELL'
            return {
                'decision': decision,
                'should_override': True,
                'activity_pressure': pressure,
                'urgency_level': 'HIGH',
                'reasoning': f"Conservative FORCED {decision} - Pressure: {pressure:.2f}",
                'force_trade_triggered': True
            }
        
        elif pressure > 0.5 and consensus == 'HOLD':  # Medium threshold
            decision = 'BUY' if current_position == 0 else 'SELL'
            return {
                'decision': decision,
                'should_override': True,
                'activity_pressure': pressure,
                'urgency_level': 'MEDIUM',
                'reasoning': f"Conservative activity nudge - Pressure: {pressure:.2f}",
                'force_trade_triggered': False
            }
        
        return {
            'decision': consensus,
            'should_override': False,
            'activity_pressure': pressure,
            'urgency_level': 'LOW',
            'reasoning': "No enforcement needed",
            'force_trade_triggered': False
        }
    
    def update_activity_metrics(self, trade_executed, current_date, performance=0.0):
        """Update activity tracking"""
        # Update week tracking
        if self.activity_metrics['week_start'] is None:
            self.activity_metrics['week_start'] = current_date
        elif (current_date - self.activity_metrics['week_start']).days >= 7:
            self.activity_metrics['week_start'] = current_date
            self.activity_metrics['trades_this_week'] = 0
        
        if trade_executed:
            self.activity_metrics['days_since_last_trade'] = 0
            self.activity_metrics['trades_this_week'] += 1
            self.activity_metrics['total_trades'] += 1
            
            if self.should_force_trade():
                self.activity_metrics['forced_trades'] += 1
            
            # Update performance
            self.activity_metrics['performance_score'] = max(0.0, min(1.0,
                self.activity_metrics['performance_score'] * 0.9 + performance * 0.1))
            
            # Add to training history
            self.training_history.append({
                'date': current_date,
                'performance': performance,
                'trade_type': 'executed'
            })
        else:
            self.activity_metrics['days_since_last_trade'] += 1
        
        # Update activity score
        self.update_activity_score()
    
    def update_activity_score(self):
        """Update overall activity score"""
        if self.activity_metrics['total_trades'] > 0:
            forced_ratio = self.activity_metrics['forced_trades'] / self.activity_metrics['total_trades']
            frequency_score = 1.0 - forced_ratio
        else:
            frequency_score = 0.5
        
        recent_activity = max(0, 1.0 - (self.activity_metrics['days_since_last_trade'] / 7.0))
        weekly_compliance = min(1.0, self.activity_metrics['trades_this_week'] / self.min_trades_per_week)
        
        self.activity_metrics['activity_score'] = (
            frequency_score * 0.4 +
            recent_activity * 0.3 +
            weekly_compliance * 0.3
        )

