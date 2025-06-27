#!/usr/bin/env python3
"""
Enhanced Candlestick Pattern Agent
Detects and analyzes candlestick patterns with self-attention
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime, timedelta
import json
from pathlib import Path
from collections import deque
import warnings
warnings.filterwarnings('ignore')

class CandlestickPatternDetector:
    """Advanced candlestick pattern detection using Ta-Lib style analysis"""
    
    def __init__(self):
        self.pattern_definitions = {
            'doji': {'body_ratio_max': 0.1, 'signal': 'reversal', 'strength_base': 0.7},
            'hammer': {'body_ratio_max': 0.3, 'lower_shadow_min': 2.0, 'signal': 'bullish_reversal', 'strength_base': 0.8},
            'hanging_man': {'body_ratio_max': 0.3, 'lower_shadow_min': 2.0, 'signal': 'bearish_reversal', 'strength_base': 0.8},
            'shooting_star': {'body_ratio_max': 0.3, 'upper_shadow_min': 2.0, 'signal': 'bearish_reversal', 'strength_base': 0.8},
            'inverted_hammer': {'body_ratio_max': 0.3, 'upper_shadow_min': 2.0, 'signal': 'bullish_reversal', 'strength_base': 0.7},
            'engulfing_bullish': {'signal': 'bullish_reversal', 'strength_base': 0.9},
            'engulfing_bearish': {'signal': 'bearish_reversal', 'strength_base': 0.9},
            'morning_star': {'signal': 'bullish_reversal', 'strength_base': 0.85},
            'evening_star': {'signal': 'bearish_reversal', 'strength_base': 0.85}
        }
    
    def calculate_candlestick_metrics(self, ohlc_row):
        """Calculate comprehensive candlestick metrics"""
        open_price = ohlc_row['Open']
        high_price = ohlc_row['High']
        low_price = ohlc_row['Low']
        close_price = ohlc_row['Close']
        
        # Basic measurements
        body_size = abs(close_price - open_price)
        total_range = high_price - low_price
        upper_shadow = high_price - max(open_price, close_price)
        lower_shadow = min(open_price, close_price) - low_price
        
        # Avoid division by zero
        if total_range == 0:
            return {
                'body_ratio': 0, 'upper_shadow_ratio': 0, 'lower_shadow_ratio': 0,
                'is_bullish': close_price >= open_price, 'body_size': 0,
                'upper_shadow': 0, 'lower_shadow': 0, 'total_range': 0
            }
        
        # Ratios
        body_ratio = body_size / total_range
        upper_shadow_ratio = upper_shadow / body_size if body_size > 0 else 0
        lower_shadow_ratio = lower_shadow / body_size if body_size > 0 else 0
        
        return {
            'body_ratio': body_ratio,
            'upper_shadow_ratio': upper_shadow_ratio,
            'lower_shadow_ratio': lower_shadow_ratio,
            'is_bullish': close_price > open_price,
            'body_size': body_size,
            'upper_shadow': upper_shadow,
            'lower_shadow': lower_shadow,
            'total_range': total_range
        }
    
    def detect_single_patterns(self, ohlc_data):
        """Detect single candlestick patterns"""
        if len(ohlc_data) < 1:
            return []
        
        current = ohlc_data.iloc[-1]
        metrics = self.calculate_candlestick_metrics(current)
        detected = []
        
        # Doji pattern
        if metrics['body_ratio'] <= self.pattern_definitions['doji']['body_ratio_max']:
            strength = (1.0 - metrics['body_ratio']) * self.pattern_definitions['doji']['strength_base']
            detected.append({
                'pattern': 'doji',
                'signal': 'reversal',
                'strength': strength,
                'type': 'single',
                'confidence': min(strength, 1.0)
            })
        
        # Hammer pattern
        if (metrics['body_ratio'] <= self.pattern_definitions['hammer']['body_ratio_max'] and
            metrics['lower_shadow_ratio'] >= self.pattern_definitions['hammer']['lower_shadow_min'] and
            metrics['upper_shadow'] < metrics['body_size']):
            strength = min(metrics['lower_shadow_ratio'] / 3.0, 1.0) * self.pattern_definitions['hammer']['strength_base']
            detected.append({
                'pattern': 'hammer',
                'signal': 'bullish_reversal',
                'strength': strength,
                'type': 'single',
                'confidence': strength
            })
        
        # Shooting star pattern
        if (metrics['body_ratio'] <= self.pattern_definitions['shooting_star']['body_ratio_max'] and
            metrics['upper_shadow_ratio'] >= self.pattern_definitions['shooting_star']['upper_shadow_min'] and
            metrics['lower_shadow'] < metrics['body_size']):
            strength = min(metrics['upper_shadow_ratio'] / 3.0, 1.0) * self.pattern_definitions['shooting_star']['strength_base']
            detected.append({
                'pattern': 'shooting_star',
                'signal': 'bearish_reversal',
                'strength': strength,
                'type': 'single',
                'confidence': strength
            })
        
        return detected
    
    def detect_multi_patterns(self, ohlc_data):
        """Detect multi-candlestick patterns"""
        detected = []
        
        # Engulfing patterns (2 candles)
        if len(ohlc_data) >= 2:
            prev = ohlc_data.iloc[-2]
            curr = ohlc_data.iloc[-1]
            
            prev_metrics = self.calculate_candlestick_metrics(prev)
            curr_metrics = self.calculate_candlestick_metrics(curr)
            
            # Bullish engulfing
            if (not prev_metrics['is_bullish'] and curr_metrics['is_bullish'] and
                curr['Open'] < prev['Close'] and curr['Close'] > prev['Open']):
                strength = min(curr_metrics['body_size'] / prev_metrics['body_size'], 2.0) / 2.0
                strength *= self.pattern_definitions['engulfing_bullish']['strength_base']
                detected.append({
                    'pattern': 'bullish_engulfing',
                    'signal': 'bullish_reversal',
                    'strength': strength,
                    'type': 'multi',
                    'confidence': strength
                })
            
            # Bearish engulfing
            if (prev_metrics['is_bullish'] and not curr_metrics['is_bullish'] and
                curr['Open'] > prev['Close'] and curr['Close'] < prev['Open']):
                strength = min(curr_metrics['body_size'] / prev_metrics['body_size'], 2.0) / 2.0
                strength *= self.pattern_definitions['engulfing_bearish']['strength_base']
                detected.append({
                    'pattern': 'bearish_engulfing',
                    'signal': 'bearish_reversal',
                    'strength': strength,
                    'type': 'multi',
                    'confidence': strength
                })
        
        # Morning/Evening Star patterns (3 candles)
        if len(ohlc_data) >= 3:
            first = ohlc_data.iloc[-3]
            middle = ohlc_data.iloc[-2]
            last = ohlc_data.iloc[-1]
            
            first_metrics = self.calculate_candlestick_metrics(first)
            middle_metrics = self.calculate_candlestick_metrics(middle)
            last_metrics = self.calculate_candlestick_metrics(last)
            
            # Morning Star
            if (not first_metrics['is_bullish'] and
                middle_metrics['body_size'] < first_metrics['body_size'] * 0.3 and
                last_metrics['is_bullish'] and
                last['Close'] > (first['Open'] + first['Close']) / 2):
                strength = self.pattern_definitions['morning_star']['strength_base']
                detected.append({
                    'pattern': 'morning_star',
                    'signal': 'bullish_reversal',
                    'strength': strength,
                    'type': 'multi',
                    'confidence': strength
                })
            
            # Evening Star
            if (first_metrics['is_bullish'] and
                middle_metrics['body_size'] < first_metrics['body_size'] * 0.3 and
                not last_metrics['is_bullish'] and
                last['Close'] < (first['Open'] + first['Close']) / 2):
                strength = self.pattern_definitions['evening_star']['strength_base']
                detected.append({
                    'pattern': 'evening_star',
                    'signal': 'bearish_reversal',
                    'strength': strength,
                    'type': 'multi',
                    'confidence': strength
                })
        
        return detected

class CandlestickSelfAttention(nn.Module):
    """Self-attention network for candlestick pattern analysis"""
    
    def __init__(self, input_dim=25, hidden_dim=128, num_heads=8):
        super(CandlestickSelfAttention, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Input processing
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        
        # Multi-head self-attention
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.1
        )
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        
        # Pattern importance weighting
        self.pattern_weights = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Final decision layers
        self.decision_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 3),  # HOLD, BUY, SELL
            nn.Softmax(dim=-1)
        )
        
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """Forward pass through the attention network"""
        # Input projection
        x = self.input_projection(x)
        x = self.layer_norm1(x)
        
        # Self-attention with residual connection
        attn_output, attention_weights = self.self_attention(x, x, x)
        x = x + attn_output
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.layer_norm2(x + ff_output)
        
        # Global pooling with attention weights
        pattern_weights = self.pattern_weights(x)
        weighted_features = x * pattern_weights
        pooled_features = torch.mean(weighted_features, dim=1)
        
        # Generate decisions and confidence
        decision_probs = self.decision_head(pooled_features)
        confidence = self.confidence_head(pooled_features)
        
        return decision_probs, confidence, attention_weights, pattern_weights

class CandlestickAgent:
    """Main candlestick pattern analysis agent with self-attention"""
    
    def __init__(self, stock_symbol):
        self.stock_symbol = stock_symbol.upper()
        self.pattern_detector = CandlestickPatternDetector()
        self.attention_model = CandlestickSelfAttention()
        self.optimizer = torch.optim.AdamW(self.attention_model.parameters(), lr=0.001, weight_decay=0.01)
        self.criterion = nn.CrossEntropyLoss()
        
        # Performance tracking
        self.performance_metrics = {
            'total_predictions': 0,
            'correct_predictions': 0,
            'accuracy': 0.0,
            'pattern_performance': {}
        }
        
        self.is_trained = False
        print(f"🕯️ Enhanced Candlestick Agent initialized for {self.stock_symbol}")

    def extract_comprehensive_features(self, ohlc_data):
        """Extract comprehensive candlestick features"""
        if len(ohlc_data) < 3:
            return np.zeros(25, dtype=np.float32)
        
        features = []
        
        # Extract features from last 3 candles
        for i in range(min(3, len(ohlc_data))):
            candle = ohlc_data.iloc[-(i+1)]
            metrics = self.pattern_detector.calculate_candlestick_metrics(candle)
            
            # Add 5 features per candle
            features.extend([
                metrics['body_ratio'],
                metrics['upper_shadow_ratio'],
                metrics['lower_shadow_ratio'],
                1.0 if metrics['is_bullish'] else 0.0,
                metrics['total_range'] / candle['Close'] if candle['Close'] > 0 else 0.0
            ])
        
        # Ensure we have 15 features from candles
        while len(features) < 15:
            features.extend([0.0, 0.0, 0.0, 0.0, 0.0])
        
        # Pattern detection features (10 features)
        single_patterns = self.pattern_detector.detect_single_patterns(ohlc_data)
        multi_patterns = self.pattern_detector.detect_multi_patterns(ohlc_data)
        all_patterns = single_patterns + multi_patterns
        
        pattern_features = [0.0] * 10
        pattern_names = ['doji', 'hammer', 'shooting_star', 'inverted_hammer', 'hanging_man',
                        'bullish_engulfing', 'bearish_engulfing', 'morning_star', 'evening_star', 'other']
        
        for pattern in all_patterns:
            pattern_name = pattern['pattern']
            if pattern_name in pattern_names:
                idx = pattern_names.index(pattern_name)
                pattern_features[idx] = pattern['strength']
            else:
                pattern_features[-1] = max(pattern_features[-1], pattern['strength'])  # Other patterns
        
        features.extend(pattern_features)
        
        return np.array(features[:25], dtype=np.float32)

    def train_attention_model(self, historical_data, epochs=50):
        """Train the self-attention model on historical data"""
        print("🕯️ Training candlestick self-attention model...")
        
        training_samples = []
        
        # Generate training data
        for i in range(10, len(historical_data) - 1):
            window_data = historical_data.iloc[i-10:i]
            features = self.extract_comprehensive_features(window_data)
            
            # Calculate future return for labeling
            current_price = historical_data.iloc[i]['Close']
            future_price = historical_data.iloc[i+1]['Close']
            return_pct = (future_price - current_price) / current_price
            
            # Label based on return thresholds
            if return_pct > 0.015:  # 1.5% threshold
                label = 1  # BUY
            elif return_pct < -0.015:
                label = 2  # SELL
            else:
                label = 0  # HOLD
            
            training_samples.append({
                'features': features,
                'label': label,
                'return': return_pct
            })
        
        if len(training_samples) < 30:
            print("⚠️ Insufficient training data for candlestick model")
            return False
        
        # Prepare training tensors
        X = torch.FloatTensor([s['features'] for s in training_samples]).unsqueeze(1)  # [batch, 1, features]
        y = torch.LongTensor([s['label'] for s in training_samples])
        
        # Training loop
        self.attention_model.train()
        best_loss = float('inf')
        
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            
            decision_probs, confidence, attention_weights, pattern_weights = self.attention_model(X)
            loss = self.criterion(decision_probs, y)
            
            # Add confidence regularization
            confidence_loss = torch.mean(torch.abs(confidence - 0.5))  # Encourage diverse confidence
            total_loss = loss + 0.1 * confidence_loss
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.attention_model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            if epoch % 10 == 0:
                accuracy = (torch.argmax(decision_probs, dim=1) == y).float().mean()
                print(f"Epoch {epoch}: Loss = {loss.item():.4f}, Accuracy = {accuracy.item():.3f}")
                
                if loss.item() < best_loss:
                    best_loss = loss.item()
        
        self.is_trained = True
        print("✅ Candlestick attention model training completed!")
        return True

    def analyze_patterns(self, ohlc_data):
        """Analyze candlestick patterns and generate trading signals"""
        # Extract features
        features = self.extract_comprehensive_features(ohlc_data)
        
        # Detect patterns
        single_patterns = self.pattern_detector.detect_single_patterns(ohlc_data)
        multi_patterns = self.pattern_detector.detect_multi_patterns(ohlc_data)
        all_patterns = single_patterns + multi_patterns
        
        if not self.is_trained:
            return self.rule_based_analysis(all_patterns)
        
        # Use attention model
        features_tensor = torch.FloatTensor(features).unsqueeze(0).unsqueeze(0)  # [1, 1, 25]
        
        self.attention_model.eval()
        with torch.no_grad():
            decision_probs, confidence, attention_weights, pattern_weights = self.attention_model(features_tensor)
            
            signal_names = ['HOLD', 'BUY', 'SELL']
            predicted_signal = signal_names[torch.argmax(decision_probs[0]).item()]
            confidence_score = float(confidence[0].item())
        
        return {
            'signal': predicted_signal,
            'confidence': confidence_score,
            'patterns_detected': all_patterns,
            'pattern_count': len(all_patterns),
            'reasoning': self.generate_reasoning(all_patterns),
            'attention_weights': attention_weights.squeeze().numpy() if attention_weights is not None else None,
            'pattern_importance': pattern_weights.squeeze().numpy() if pattern_weights is not None else None
        }

    def rule_based_analysis(self, patterns):
        """Fallback rule-based analysis"""
        if not patterns:
            return {
                'signal': 'HOLD',
                'confidence': 0.2,
                'patterns_detected': [],
                'pattern_count': 0,
                'reasoning': 'No significant patterns detected'
            }
        
        # Aggregate signals
        bullish_strength = sum(p['strength'] for p in patterns if 'bullish' in p['signal'])
        bearish_strength = sum(p['strength'] for p in patterns if 'bearish' in p['signal'])
        
        # Determine signal
        if bullish_strength > bearish_strength and bullish_strength > 0.6:
            signal = 'BUY'
            confidence = min(bullish_strength, 1.0)
        elif bearish_strength > bullish_strength and bearish_strength > 0.6:
            signal = 'SELL'
            confidence = min(bearish_strength, 1.0)
        else:
            signal = 'HOLD'
            confidence = 0.3
        
        return {
            'signal': signal,
            'confidence': confidence,
            'patterns_detected': patterns,
            'pattern_count': len(patterns),
            'reasoning': self.generate_reasoning(patterns)
        }

    def generate_reasoning(self, patterns):
        """Generate human-readable reasoning"""
        if not patterns:
            return "No candlestick patterns detected"
        
        pattern_descriptions = []
        for pattern in patterns:
            desc = f"{pattern['pattern']} (strength: {pattern['strength']:.2f})"
            pattern_descriptions.append(desc)
        
        return f"Patterns: {', '.join(pattern_descriptions)}"

    def save_model(self, filepath=None):
        """Save the trained attention model"""
        if filepath is None:
            filepath = f"candlestick_attention_{self.stock_symbol}.pth"
        
        torch.save({
            'model_state_dict': self.attention_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'performance_metrics': self.performance_metrics,
            'is_trained': self.is_trained
        }, filepath)
        
        print(f"💾 Candlestick model saved to {filepath}")

    def load_model(self, filepath=None):
        """Load a trained attention model"""
        if filepath is None:
            filepath = f"candlestick_attention_{self.stock_symbol}.pth"
        
        if not Path(filepath).exists():
            return False
        
        try:
            checkpoint = torch.load(filepath)
            self.attention_model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.performance_metrics = checkpoint['performance_metrics']
            self.is_trained = checkpoint['is_trained']
            
            print(f"✅ Candlestick model loaded from {filepath}")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False

def main():
    """Test the enhanced candlestick agent"""
    import yfinance as yf
    
    # Test with sample data
    ticker = yf.Ticker("AAPL")
    data = ticker.history(period="60d", interval="5m")
    
    if data.empty:
        print("No data available")
        return
    
    # Initialize and train agent
    agent = CandlestickAgent("AAPL")
    
    # Train the attention model
    agent.train_attention_model(data)
    
    # Test pattern analysis
    recent_data = data.tail(20)
    result = agent.analyze_patterns(recent_data)
    
    print(f"\n🕯️ Candlestick Analysis Result:")
    print(f"Signal: {result['signal']}")
    print(f"Confidence: {result['confidence']:.2f}")
    print(f"Patterns: {result['pattern_count']}")
    print(f"Reasoning: {result['reasoning']}")

if __name__ == "__main__":
    main()

