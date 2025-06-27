#!/usr/bin/env python3
"""
WolfX Enhanced Trading System (WolfXE)
Integrates Activity Enforcer with Candlestick Self-Attention
"""

import sys
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime, timedelta, date
import yfinance as yf
import json
from pathlib import Path
from stable_baselines3 import PPO
from collections import deque
import warnings
warnings.filterwarnings('ignore')

# Import agent classes
from rw_wolfxe.agents.pure_intraday_agent import PureIntradayAgent
from rw_wolfxe.agents.intraday_trend_agent import IntradayTrendAgent
from rw_wolfxe.agents.candlestick_agent import CandlestickAgent

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
        self.training_history = deque(maxlen=500)
        self.is_trained = False
        
        print(f"⚡ WolfXE Activity Enforcer initialized for {self.stock_symbol}")
        print(f"📊 Min trades/week: {min_trades_per_week}, Max days without trade: {max_days_without_trade}")

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
        
        recent_performance = [h['performance'] for h in list(self.training_history)[-5:]]
        avg_performance = np.mean(recent_performance)
        return max(0.0, min(1.0, (avg_performance + 1.0) / 2.0))

    def should_force_trade(self):
        """Determine if trade should be forced"""
        # Force if too many days without activity
        if self.activity_metrics['days_since_last_trade'] >= self.max_days_without_trade:
            return True
        
        # Force if below weekly minimum near end of week
        current_day = datetime.now().weekday()
        if (current_day >= 3 and 
            self.activity_metrics['trades_this_week'] < self.min_trades_per_week):
            return True
        
        return False

    def calculate_activity_pressure(self):
        """Calculate current activity pressure"""
        pressure = 0.0
        
        # Days without trade pressure
        days_pressure = self.activity_metrics['days_since_last_trade'] / self.max_days_without_trade
        pressure += min(days_pressure, 1.0) * 0.5
        
        # Weekly deficit pressure
        trade_deficit = max(0, self.min_trades_per_week - self.activity_metrics['trades_this_week'])
        weekly_pressure = trade_deficit / self.min_trades_per_week
        pressure += weekly_pressure * 0.3
        
        # Time urgency
        pressure += self.get_time_urgency_factor() * 0.2
        
        return min(pressure, 1.0)

    def train_activity_enforcer(self, historical_data, epochs=50):
        """Train the activity enforcement model"""
        print("⚡ Training WolfXE Activity Enforcer...")
        
        training_samples = []
        
        # Generate training scenarios
        for i in range(30, len(historical_data) - 5):
            window_data = historical_data.iloc[i-30:i]
            market_features = self.extract_market_features(window_data)
            
            # Simulate different activity scenarios
            for days_without in range(0, 8):
                for trades_week in range(0, 6):
                    # Set scenario
                    self.activity_metrics['days_since_last_trade'] = days_without
                    self.activity_metrics['trades_this_week'] = trades_week
                    
                    activity_features = self.extract_activity_features()
                    
                    # Calculate targets
                    target_pressure = self.calculate_activity_pressure()
                    
                    # Urgency level
                    if target_pressure > 0.7:
                        urgency_label = 2  # HIGH
                    elif target_pressure > 0.4:
                        urgency_label = 1  # MEDIUM
                    else:
                        urgency_label = 0  # LOW
                    
                    # Override signal based on future price movement
                    current_price = historical_data.iloc[i]['Close']
                    future_price = historical_data.iloc[i+3]['Close']  # 3 periods ahead
                    
                    if self.should_force_trade():
                        if future_price > current_price * 1.005:  # 0.5% threshold
                            override_label = 1  # BUY
                        elif future_price < current_price * 0.995:
                            override_label = 2  # SELL
                        else:
                            override_label = 0  # HOLD
                    else:
                        override_label = 0  # HOLD
                    
                    training_samples.append({
                        'market_features': market_features,
                        'activity_features': activity_features,
                        'target_pressure': target_pressure,
                        'urgency_label': urgency_label,
                        'override_label': override_label
                    })
        
        if len(training_samples) < 100:
            print("⚠️ Insufficient training data")
            return False
        
        # Prepare tensors
        market_X = torch.FloatTensor([s['market_features'] for s in training_samples]).unsqueeze(1)
        activity_X = torch.FloatTensor([s['activity_features'] for s in training_samples]).unsqueeze(1)
        pressure_y = torch.FloatTensor([s['target_pressure'] for s in training_samples])
        urgency_y = torch.LongTensor([s['urgency_label'] for s in training_samples])
        override_y = torch.LongTensor([s['override_label'] for s in training_samples])
        
        # Training loop
        self.attention_model.train()
        best_loss = float('inf')
        
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            
            pressure_pred, urgency_pred, override_pred, attention_weights = self.attention_model(
                market_X, activity_X
            )
            
            # Calculate losses
            pressure_loss = self.criterion_mse(pressure_pred.squeeze(), pressure_y)
            urgency_loss = self.criterion_ce(urgency_pred, urgency_y)
            override_loss = self.criterion_ce(override_pred, override_y)
            
            # Combined loss
            total_loss = pressure_loss + urgency_loss + override_loss
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.attention_model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Total Loss = {total_loss.item():.4f}")
                
                if total_loss.item() < best_loss:
                    best_loss = total_loss.item()
        
        self.is_trained = True
        print("✅ WolfXE Activity Enforcer training completed!")
        return True

    def enforce_activity(self, market_data, base_decisions, current_position):
        """FIXED: Main enforcement function with better logic"""
        # Extract features
        market_features = self.extract_market_features(market_data)
        activity_features = self.extract_activity_features()
        
        # Calculate current pressure
        current_pressure = self.calculate_activity_pressure()
        
        # FIXED: More aggressive enforcement logic
        if not self.is_trained:
            return self.rule_based_enforcement(base_decisions, current_pressure, current_position)
        
        # Use trained model
        market_tensor = torch.FloatTensor(market_features).unsqueeze(0).unsqueeze(0)
        activity_tensor = torch.FloatTensor(activity_features).unsqueeze(0).unsqueeze(0)
        
        self.attention_model.eval()
        with torch.no_grad():
            pressure_pred, urgency_pred, override_pred, attention_weights = self.attention_model(
                market_tensor, activity_tensor
            )
            
            pressure_score = float(pressure_pred[0].item())
            urgency_levels = ['LOW', 'MEDIUM', 'HIGH']
            urgency = urgency_levels[torch.argmax(urgency_pred[0]).item()]
            
            signal_names = ['HOLD', 'BUY', 'SELL']
            override_signal = signal_names[torch.argmax(override_pred[0]).item()]
        
        # FIXED: More aggressive decision logic
        consensus = base_decisions.get('consensus', 'HOLD')
        should_override = False
        final_decision = consensus
        reasoning = "Base consensus maintained"
        
        # FIXED: Lower thresholds for forcing trades
        if pressure_score > 0.6 or self.should_force_trade():  # Lowered from 0.8
            should_override = True
            # Force trade based on position
            if current_position == 0:
                final_decision = 'BUY'
            else:
                final_decision = 'SELL'
            reasoning = f"FORCED TRADE - Pressure: {pressure_score:.2f}, Urgency: {urgency}"
        
        # FIXED: Medium pressure override (lowered threshold)
        elif pressure_score > 0.4 and consensus == 'HOLD':  # Lowered from 0.6
            should_override = True
            if current_position == 0:
                final_decision = 'BUY'
            else:
                final_decision = 'SELL'
            reasoning = f"Activity nudge - Pressure: {pressure_score:.2f}"
        
        # FIXED: Position-specific enforcement (lowered thresholds)
        elif current_position == 0 and pressure_score > 0.3:  # Lowered from 0.5
            should_override = True
            final_decision = 'BUY'
            reasoning = f"No position enforcement - Pressure: {pressure_score:.2f}"
        elif current_position > 0 and pressure_score > 0.5:  # Lowered from 0.7
            should_override = True
            final_decision = 'SELL'
            reasoning = f"Position exit enforcement - Pressure: {pressure_score:.2f}"
        
        return {
            'decision': final_decision,
            'should_override': should_override,
            'activity_pressure': pressure_score,
            'urgency_level': urgency,
            'reasoning': reasoning,
            'force_trade_triggered': self.should_force_trade(),
            'attention_weights': attention_weights.numpy() if attention_weights is not None else None
        }

    def rule_based_enforcement(self, base_decisions, pressure, current_position):
        """FIXED: More aggressive fallback rule-based enforcement"""
        consensus = base_decisions.get('consensus', 'HOLD')
        
        # FIXED: Lower pressure thresholds
        if pressure > 0.6 or self.should_force_trade():  # Lowered from 0.8
            decision = 'BUY' if current_position == 0 else 'SELL'
            return {
                'decision': decision,
                'should_override': True,
                'activity_pressure': pressure,
                'urgency_level': 'HIGH',
                'reasoning': f"Rule-based FORCED {decision} - Pressure: {pressure:.2f}",
                'force_trade_triggered': True
            }
        
        elif pressure > 0.3 and consensus == 'HOLD':  # Lowered from 0.5
            decision = 'BUY' if current_position == 0 else 'SELL'
            return {
                'decision': decision,
                'should_override': True,
                'activity_pressure': pressure,
                'urgency_level': 'MEDIUM',
                'reasoning': f"Rule-based activity nudge - Pressure: {pressure:.2f}",
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

    def _enforce_activity(self, market_data, base_decisions, current_position):
        """Main enforcement function"""
        # Extract features
        market_features = self.extract_market_features(market_data)
        activity_features = self.extract_activity_features()
        
        # Calculate current pressure
        current_pressure = self.calculate_activity_pressure()
        
        if not self.is_trained:
            return self.rule_based_enforcement(base_decisions, current_pressure, current_position)
        
        # Use trained model
        market_tensor = torch.FloatTensor(market_features).unsqueeze(0).unsqueeze(0)
        activity_tensor = torch.FloatTensor(activity_features).unsqueeze(0).unsqueeze(0)
        
        self.attention_model.eval()
        with torch.no_grad():
            pressure_pred, urgency_pred, override_pred, attention_weights = self.attention_model(
                market_tensor, activity_tensor
            )
            
            pressure_score = float(pressure_pred[0].item())
            urgency_levels = ['LOW', 'MEDIUM', 'HIGH']
            urgency = urgency_levels[torch.argmax(urgency_pred[0]).item()]
            
            signal_names = ['HOLD', 'BUY', 'SELL']
            override_signal = signal_names[torch.argmax(override_pred[0]).item()]
        
        # Decision logic
        consensus = base_decisions.get('consensus', 'HOLD')
        should_override = False
        final_decision = consensus
        reasoning = "Base consensus maintained"
        
        # High pressure override
        if pressure_score > 0.8 or self.should_force_trade():
            should_override = True
            final_decision = override_signal
            reasoning = f"FORCED TRADE - Pressure: {pressure_score:.2f}, Urgency: {urgency}"
        
        # Medium pressure nudge
        elif pressure_score > 0.6 and consensus == 'HOLD':
            if override_signal != 'HOLD':
                should_override = True
                final_decision = override_signal
                reasoning = f"Activity nudge - Pressure: {pressure_score:.2f}"
        
        # Position-specific enforcement
        elif current_position == 0 and pressure_score > 0.5:
            if override_signal == 'BUY':
                should_override = True
                final_decision = 'BUY'
                reasoning = f"No position enforcement - Pressure: {pressure_score:.2f}"
        elif current_position > 0 and pressure_score > 0.7:
            if override_signal == 'SELL':
                should_override = True
                final_decision = 'SELL'
                reasoning = f"Position exit enforcement - Pressure: {pressure_score:.2f}"
        
        return {
            'decision': final_decision,
            'should_override': should_override,
            'activity_pressure': pressure_score,
            'urgency_level': urgency,
            'reasoning': reasoning,
            'force_trade_triggered': self.should_force_trade(),
            'attention_weights': attention_weights.numpy() if attention_weights is not None else None
        }

    def _rule_based_enforcement(self, base_decisions, pressure, current_position):
        """Fallback rule-based enforcement"""
        consensus = base_decisions.get('consensus', 'HOLD')
        
        if pressure > 0.8 or self.should_force_trade():
            decision = 'BUY' if current_position == 0 else 'SELL'
            return {
                'decision': decision,
                'should_override': True,
                'activity_pressure': pressure,
                'urgency_level': 'HIGH',
                'reasoning': f"Rule-based FORCED {decision} - Pressure: {pressure:.2f}",
                'force_trade_triggered': True
            }
        
        elif pressure > 0.5 and consensus == 'HOLD':
            decision = 'BUY' if current_position == 0 else 'SELL'
            return {
                'decision': decision,
                'should_override': True,
                'activity_pressure': pressure,
                'urgency_level': 'MEDIUM',
                'reasoning': f"Rule-based activity nudge - Pressure: {pressure:.2f}",
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

class WolfXEnhancedTradingSystem:
    """WolfX Enhanced with Activity Enforcer and Candlestick Attention"""
    
    def __init__(self, stock_symbol, initial_balance=10000):
        self.stock_symbol = stock_symbol.upper()
        self.initial_balance = initial_balance
        
        # Portfolio management
        self.portfolio = {
            'cash': initial_balance,
            'shares': 0,
            'position_entry_price': None,
            'position_entry_time': None,
            'days_held': 0,
            'trades': []
        }
        
        # Enhanced constraints
        self.constraints = {
            'min_hold_days': 1,
            'max_hold_days': 12,
            'stop_loss_pct': 0.04,
            'take_profit_pct': 0.08,
            'max_position_size': 0.8,
            'min_trade_amount': 300
        }
        
        # Agents
        self.agent_a1 = None
        self.agent_a2 = None
        self.candlestick_agent = CandlestickAgent(stock_symbol)
        self.activity_enforcer = WolfXEActivityEnforcer(
            stock_symbol, 
            min_trades_per_week=3,
            max_days_without_trade=6
        )
        
        # Tracking
        self.daily_reports = []
        self.last_trade_date = None
        
        print(f"🐺⚡ WolfX Enhanced Trading System initialized for {self.stock_symbol}")

    def setup_models_directory(self):
        """Setup model directories"""
        self.models_dir = Path(f"models/{self.stock_symbol}")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.agent_a1_path = self.models_dir / f"agent_a1_{self.stock_symbol}"
        self.agent_a2_path = self.models_dir / f"agent_a2_{self.stock_symbol}"

    def load_or_train_all_agents(self, data):
        """Load or train all agents including enhanced components"""
        self.setup_models_directory()
        
        # Load/train base agents
        base_agents_ready = self.load_or_train_base_agents(data)
        
        # Train candlestick agent with self-attention
        print("🕯️ Training candlestick self-attention agent...")
        candlestick_trained = self.candlestick_agent.train_attention_model(data, epochs=40)
        
        # Train activity enforcer
        print("⚡ Training activity enforcer...")
        enforcer_trained = self.activity_enforcer.train_activity_enforcer(data, epochs=50)
        
        return base_agents_ready and candlestick_trained and enforcer_trained

    def load_or_train_base_agents(self, data):
        """Load or train base A1 and A2 agents"""
        try:
            if (Path(str(self.agent_a1_path) + '.zip').exists() and 
                Path(str(self.agent_a2_path) + '.zip').exists()):
                
                self.agent_a1 = PPO.load(str(self.agent_a1_path))
                self.agent_a2 = PPO.load(str(self.agent_a2_path))
                print("✅ Loaded existing base agents")
                return True
        except:
            pass
        
        print("🎯 Training base agents...")
        
        # Prepare training data
        daily_data = data.resample('1D').agg({
            'Open': 'first', 'High': 'max', 'Low': 'min', 
            'Close': 'last', 'Volume': 'sum'
        }).dropna()
        
        if len(daily_data) < 30 or len(data) < 100:
            print("❌ Insufficient data for training")
            return False
        
        # Train agents
        env_a1 = IntradayTrendAgent(daily_data, self.initial_balance)
        self.agent_a1 = PPO("MlpPolicy", env_a1, verbose=0)
        self.agent_a1.learn(total_timesteps=25000)
        self.agent_a1.save(str(self.agent_a1_path))
        
        env_a2 = PureIntradayAgent(data, self.initial_balance)
        self.agent_a2 = PPO("MlpPolicy", env_a2, verbose=0)
        self.agent_a2.learn(total_timesteps=30000)
        self.agent_a2.save(str(self.agent_a2_path))
        
        print("✅ Base agents training completed!")
        return True

    def get_agent_decision(self, agent, agent_type, current_data):
        """Get decision from base agents"""
        try:
            if agent_type == 'daily':
                obs = np.random.normal(0, 0.4, 30).astype(np.float32)
            else:
                obs = np.random.normal(0, 0.4, 35).astype(np.float32)
            
            action, _ = agent.predict(obs, deterministic=True)
            actions = ['HOLD', 'BUY', 'SELL']
            return actions[action]
        except:
            return np.random.choice(['HOLD', 'BUY', 'SELL'], p=[0.6, 0.2, 0.2])

    def make_enhanced_decision(self, data, current_bar):
        """Make enhanced decision using all components"""
        # Get base agent decisions
        a1_decision = self.get_agent_decision(self.agent_a1, 'daily', current_bar)
        a2_decision = self.get_agent_decision(self.agent_a2, 'intraday', current_bar)
        
        # Get candlestick analysis with self-attention
        candlestick_window = data.tail(15)
        candlestick_result = self.candlestick_agent.analyze_patterns(candlestick_window)
        
        # Create base consensus
        decisions = [a1_decision, a2_decision, candlestick_result['signal']]
        weights = [0.3, 0.3, 0.4]  # Give more weight to candlestick patterns
        
        decision_scores = {'HOLD': 0, 'BUY': 0, 'SELL': 0}
        for decision, weight in zip(decisions, weights):
            decision_scores[decision] += weight
        
        # Add candlestick confidence weighting
        if candlestick_result['confidence'] > 0.6:
            decision_scores[candlestick_result['signal']] += candlestick_result['confidence'] * 0.3
        
        base_consensus = max(decision_scores, key=decision_scores.get)
        
        # Prepare base decisions for activity enforcer
        base_decisions = {
            'a1': a1_decision,
            'a2': a2_decision,
            'candlestick': candlestick_result['signal'],
            'consensus': base_consensus
        }
        
        # Apply activity enforcement
        enforcement_result = self.activity_enforcer.enforce_activity(
            data, base_decisions, self.portfolio['shares']
        )
        
        return {
            'decision': enforcement_result['decision'],
            'should_override': enforcement_result['should_override'],
            'reasoning': enforcement_result['reasoning'],
            'activity_pressure': enforcement_result['activity_pressure'],
            'urgency_level': enforcement_result['urgency_level'],
            'force_trade': enforcement_result['force_trade_triggered'],
            'candlestick_result': candlestick_result,
            'base_decisions': base_decisions
        }

    def apply_enhanced_constraints(self, decision, current_price, current_date):
        """Apply enhanced trading constraints"""
        # Position holding constraints
        if self.portfolio['shares'] > 0:
            # Minimum hold period
            if self.portfolio['days_held'] < self.constraints['min_hold_days'] and decision == 'SELL':
                return 'HOLD'
            
            # Maximum hold period
            if self.portfolio['days_held'] >= self.constraints['max_hold_days']:
                return 'SELL'
            
            # Risk management
            if self.portfolio['position_entry_price']:
                # Stop loss
                if current_price <= self.portfolio['position_entry_price'] * (1 - self.constraints['stop_loss_pct']):
                    return 'SELL'
                
                # Take profit
                if current_price >= self.portfolio['position_entry_price'] * (1 + self.constraints['take_profit_pct']):
                    return 'SELL'
        
        # Buy constraints
        if decision == 'BUY':
            if self.portfolio['cash'] < self.constraints['min_trade_amount']:
                return 'HOLD'
            
            if self.portfolio['shares'] > 0:  # Already have position
                return 'HOLD'
        
        return decision

    def execute_enhanced_trade(self, decision, current_data, reasoning=""):
        """Execute trade with enhanced tracking"""
        current_price = current_data['Close']
        current_time = current_data.name
        current_date = current_time.date()
        
        if decision == 'BUY' and self.portfolio['shares'] == 0:
            # Enhanced position sizing
            available_cash = self.portfolio['cash'] * 0.75
            shares_to_buy = int(available_cash // current_price)
            
            if shares_to_buy > 0:
                cost = shares_to_buy * current_price
                
                self.portfolio['shares'] = shares_to_buy
                self.portfolio['cash'] -= cost
                self.portfolio['position_entry_price'] = current_price
                self.portfolio['position_entry_time'] = current_date
                self.portfolio['days_held'] = 0
                
                # Update tracking
                self.last_trade_date = current_date
                
                trade = {
                    'timestamp': current_time,
                    'type': 'BUY',
                    'shares': shares_to_buy,
                    'price': current_price,
                    'value': cost,
                    'reasoning': reasoning
                }
                self.portfolio['trades'].append(trade)
                
                # Update activity enforcer
                self.activity_enforcer.update_activity_metrics(True, current_date, 0.0)
                
                print(f"🐺⚡ WOLFXE BUY: {shares_to_buy} shares at ${current_price:.2f}")
                print(f"   📝 {reasoning}")
                
        elif decision == 'SELL' and self.portfolio['shares'] > 0:
            shares_to_sell = self.portfolio['shares']
            proceeds = shares_to_sell * current_price
            
            # Calculate P&L and performance
            entry_value = shares_to_sell * self.portfolio['position_entry_price']
            pnl = proceeds - entry_value
            pnl_pct = (pnl / entry_value) * 100
            performance = pnl / self.initial_balance  # Normalized performance
            
            self.portfolio['cash'] += proceeds
            self.portfolio['shares'] = 0
            
            # Update tracking
            self.last_trade_date = current_date
            
            trade = {
                'timestamp': current_time,
                'type': 'SELL',
                'shares': shares_to_sell,
                'price': current_price,
                'value': proceeds,
                'days_held': self.portfolio['days_held'],
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'reasoning': reasoning
            }
            self.portfolio['trades'].append(trade)
            
            # Update activity enforcer with performance
            self.activity_enforcer.update_activity_metrics(True, current_date, performance)
            
            print(f"🐺⚡ WOLFXE SELL: {shares_to_sell} shares at ${current_price:.2f}")
            print(f"   📊 Held {self.portfolio['days_held']} days | P&L: ${pnl:+.2f} ({pnl_pct:+.1f}%)")
            print(f"   📝 {reasoning}")
            
            # Reset position
            self.portfolio['position_entry_price'] = None
            self.portfolio['position_entry_time'] = None
            self.portfolio['days_held'] = 0

    def update_position_tracking(self, current_date):
        """Update position tracking"""
        if (self.portfolio['shares'] > 0 and self.portfolio['position_entry_time'] and
            current_date > self.portfolio['position_entry_time']):
            self.portfolio['days_held'] = (current_date - self.portfolio['position_entry_time']).days
        
        # Update activity enforcer for non-trading days
        if not self.last_trade_date or current_date > self.last_trade_date:
            self.activity_enforcer.update_activity_metrics(False, current_date)

    def run_wolfxe_simulation(self, data):
        """Run WolfX Enhanced simulation"""
        print(f"🐺⚡ Starting WolfX Enhanced simulation...")
        
        # Load/train all agents
        if not self.load_or_train_all_agents(data):
            return
        
        # Group by trading days
        simulation_days = data.groupby(data.index.date)
        total_days = len(simulation_days)
        
        day_counter = 0
        for date, day_data in simulation_days:
            day_counter += 1
            
            # Update tracking
            self.update_position_tracking(date)
            
            print(f"\n📅 Day {day_counter}/{total_days} - {date} | "
                  f"Shares: {self.portfolio['shares']} | Days held: {self.portfolio['days_held']} | "
                  f"Activity pressure: {self.activity_enforcer.calculate_activity_pressure():.2f}")
            
            day_start_value = (self.portfolio['cash'] + 
                             self.portfolio['shares'] * day_data['Close'].iloc[0])
            
            # Get historical data for analysis
            historical_data = data[data.index.date <= date]
            
            # Enhanced decision making with multiple decision points
            decision_points = [
                day_data.iloc[len(day_data)//4],    # Morning
                day_data.iloc[len(day_data)//2],    # Midday
                day_data.iloc[3*len(day_data)//4],  # Afternoon
                day_data.iloc[-1]                   # Close
            ]
            
            traded_today = False
            
            for i, bar_data in enumerate(decision_points):
                if traded_today:
                    break
                
                # Make enhanced decision
                decision_result = self.make_enhanced_decision(historical_data, bar_data)
                
                # Apply constraints
                final_decision = self.apply_enhanced_constraints(
                    decision_result['decision'], bar_data['Close'], date
                )
                
                print(f"🧠 Decision {i+1}: A1({decision_result['base_decisions']['a1']}) | "
                      f"A2({decision_result['base_decisions']['a2']}) | "
                      f"🕯️ Candlestick({decision_result['candlestick_result']['signal']}) | "
                      f"⚡ Enforcer({decision_result['decision']}) | "
                      f"Final: {final_decision}")
                
                if decision_result['should_override']:
                    print(f"   ⚡ ACTIVITY OVERRIDE: {decision_result['reasoning']}")
                
                if decision_result['force_trade']:
                    print(f"   🚨 FORCE TRADE TRIGGERED")
                
                # Execute trade
                if final_decision != 'HOLD':
                    self.execute_enhanced_trade(final_decision, bar_data, decision_result['reasoning'])
                    traded_today = True
            
            # End of day reporting
            end_price = day_data['Close'].iloc[-1]
            day_end_value = self.portfolio['cash'] + self.portfolio['shares'] * end_price
            daily_pnl = day_end_value - day_start_value
            
            daily_report = {
                'date': date,
                'start_value': day_start_value,
                'end_value': day_end_value,
                'daily_pnl': daily_pnl,
                'total_pnl': day_end_value - self.initial_balance,
                'price': end_price,
                'shares': self.portfolio['shares'],
                'days_held': self.portfolio['days_held'],
                'traded_today': traded_today,
                'activity_pressure': self.activity_enforcer.calculate_activity_pressure()
            }
            self.daily_reports.append(daily_report)
            
            print(f"💰 EOD: ${day_end_value:.2f} | Daily P&L: ${daily_pnl:+.2f} | "
                  f"Total P&L: ${daily_report['total_pnl']:+.2f}")
        
        self.generate_wolfxe_report()

    def generate_wolfxe_report(self):
        """Generate WolfX Enhanced performance report"""
        print(f"\n🐺⚡ WOLFX ENHANCED TRADING REPORT - {self.stock_symbol}")
        print("=" * 60)
        
        if not self.daily_reports:
            print("No trading data available")
            return
        
        final_value = self.daily_reports[-1]['end_value']
        total_return = ((final_value - self.initial_balance) / self.initial_balance) * 100
        
        print(f"Initial Balance: ${self.initial_balance:,.2f}")
        print(f"Final Value: ${final_value:,.2f}")
        print(f"Total Return: {total_return:+.2f}%")
        print(f"Total Trades: {len(self.portfolio['trades'])}")
        
        # Activity enforcement analysis
        enforcer_metrics = self.activity_enforcer.activity_metrics
        print(f"\n⚡ ACTIVITY ENFORCEMENT ANALYSIS:")
        print(f"Total Trades: {enforcer_metrics['total_trades']}")
        print(f"Forced Trades: {enforcer_metrics['forced_trades']}")
        print(f"Forced Trade Ratio: {enforcer_metrics['forced_trades']/max(1, enforcer_metrics['total_trades']):.1%}")
        print(f"Activity Score: {enforcer_metrics['activity_score']:.2f}")
        print(f"Performance Score: {enforcer_metrics['performance_score']:.2f}")
        
        # Trading frequency
        trading_days = sum(1 for d in self.daily_reports if d['traded_today'])
        print(f"Trading Days: {trading_days}/{len(self.daily_reports)} ({trading_days/len(self.daily_reports)*100:.1f}%)")
        
        # Trading analysis
        completed_trades = [t for t in self.portfolio['trades'] if t['type'] == 'SELL']
        if completed_trades:
            winning_trades = [t for t in completed_trades if t['pnl'] > 0]
            win_rate = len(winning_trades) / len(completed_trades)
            avg_hold_days = np.mean([t['days_held'] for t in completed_trades])
            
            print(f"\n📈 TRADING ANALYSIS:")
            print(f"Win Rate: {win_rate:.1%}")
            print(f"Average Hold Period: {avg_hold_days:.1f} days")
            print(f"Profitable Trades: {len(winning_trades)}/{len(completed_trades)}")
            
            if winning_trades:
                avg_win = np.mean([t['pnl'] for t in winning_trades])
                print(f"Average Win: ${avg_win:+.2f}")
            
            losing_trades = [t for t in completed_trades if t['pnl'] < 0]
            if losing_trades:
                avg_loss = np.mean([t['pnl'] for t in losing_trades])
                print(f"Average Loss: ${avg_loss:+.2f}")
                
                # Profit factor
                total_wins = sum(t['pnl'] for t in winning_trades)
                total_losses = abs(sum(t['pnl'] for t in losing_trades))
                profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
                print(f"Profit Factor: {profit_factor:.2f}")
        
        # Performance metrics
        daily_pnls = [d['daily_pnl'] for d in self.daily_reports]
        winning_days = len([p for p in daily_pnls if p > 0])
        
        print(f"\n📊 DAILY PERFORMANCE:")
        print(f"Winning Days: {winning_days}/{len(daily_pnls)} ({winning_days/len(daily_pnls)*100:.1f}%)")
        print(f"Best Day: ${max(daily_pnls):+.2f}")
        print(f"Worst Day: ${min(daily_pnls):+.2f}")
        print(f"Average Daily P&L: ${np.mean(daily_pnls):+.2f}")
        
        # Activity pressure analysis
        avg_pressure = np.mean([d['activity_pressure'] for d in self.daily_reports])
        print(f"Average Activity Pressure: {avg_pressure:.2f}")
        
        self.save_wolfxe_results()

    def save_wolfxe_results(self):
        """Save WolfX Enhanced results"""
        results_dir = Path(f"wolfxe_results_{self.stock_symbol}")
        results_dir.mkdir(exist_ok=True)
        
        # Save daily reports
        if self.daily_reports:
            pd.DataFrame(self.daily_reports).to_csv(results_dir / "daily_reports.csv", index=False)
        
        # Save trades
        if self.portfolio['trades']:
            pd.DataFrame(self.portfolio['trades']).to_csv(results_dir / "trades.csv", index=False)

        activity_metrics_serializable = {}
        for key, value in self.activity_enforcer.activity_metrics.items():
            if isinstance(value, (datetime, date)):
                activity_metrics_serializable[key] = value.isoformat()
            else:
                activity_metrics_serializable[key] = value
        
        # Save activity metrics
        with open(results_dir / "activity_metrics.json", 'w') as f:
            json.dump(activity_metrics_serializable, f, indent=2, default=str)
        
        print(f"📁 WolfXE results saved to: {results_dir}")

def main():
    """Main function for WolfX Enhanced trading system"""
    parser = argparse.ArgumentParser(description='WolfX Enhanced Trading System')
    parser.add_argument('symbol', help='Stock symbol (e.g., AAPL, NVDA, TSLA)')
    parser.add_argument('--balance', type=float, default=10000, help='Initial balance')
    parser.add_argument('--days', type=int, default=60, help='Days of data to fetch')
    parser.add_argument('--min-trades', type=int, default=3, help='Minimum trades per week')
    parser.add_argument('--max-days', type=int, default=6, help='Maximum days without trade')
    
    args = parser.parse_args()
    
    try:
        # Initialize WolfX Enhanced system
        wolfxe_system = WolfXEnhancedTradingSystem(args.symbol, args.balance)
        
        # Configure activity enforcer
        wolfxe_system.activity_enforcer.min_trades_per_week = args.min_trades
        wolfxe_system.activity_enforcer.max_days_without_trade = args.max_days
        
        # Fetch market data
        ticker = yf.Ticker(args.symbol)
        data = ticker.history(period=f"{args.days}d", interval="5m")
        
        if data.empty:
            print(f"❌ No data available for {args.symbol}")
            return
        
        print(f"📊 Fetched {len(data)} bars of 5-minute data")
        
        # Run WolfX Enhanced simulation
        wolfxe_system.run_wolfxe_simulation(data)
        
    except Exception as e:
        print(f"❌ WolfXE simulation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

