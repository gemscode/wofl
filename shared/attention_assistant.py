#!/usr/bin/env python3
"""
Self-Attention Decision Assistant
Trains once, then assists Agent A1 and A2 with weighted decision making
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime, timedelta
import json
from pathlib import Path
import yfinance as yf
from stable_baselines3 import PPO
from collections import deque
import warnings
warnings.filterwarnings('ignore')

class SelfAttentionAssistant(nn.Module):
    """Self-attention neural network for decision weighting"""
    
    def __init__(self, input_dim=10, hidden_dim=64, num_heads=4):
        super(SelfAttentionAssistant, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Feature processing layers
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        
        # Decision weighting network
        self.decision_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 3),  # 3 outputs: weight_a1, weight_a2, confidence
            nn.Softmax(dim=-1)
        )
        
        # Final decision layer
        self.final_decision = nn.Sequential(
            nn.Linear(hidden_dim + 3, 32),  # hidden + decision weights
            nn.ReLU(),
            nn.Linear(32, 3),  # HOLD, BUY, SELL
            nn.Softmax(dim=-1)
        )

    def forward(self, market_features, agent_decisions):
        """
        Forward pass
        market_features: [batch_size, seq_len, input_dim]
        agent_decisions: [batch_size, 2] - A1 and A2 decisions (0=HOLD, 1=BUY, 2=SELL)
        """
        # Project input features
        x = self.input_projection(market_features)
        x = self.layer_norm1(x)
        
        # Self-attention
        attn_output, attention_weights = self.attention(x, x, x)
        x = x + attn_output  # Residual connection
        x = self.layer_norm2(x)
        
        # Global pooling (take mean across sequence)
        x_pooled = torch.mean(x, dim=1)
        
        # Generate decision weights
        decision_weights = self.decision_network(x_pooled)
        
        # Combine with pooled features for final decision
        combined = torch.cat([x_pooled, decision_weights], dim=-1)
        final_decision = self.final_decision(combined)
        
        return final_decision, decision_weights, attention_weights

class TradingDecisionAssistant:
    """Main assistant that coordinates with A1 and A2 agents"""
    
    def __init__(self, stock_symbol, lookback_window=20):
        self.stock_symbol = stock_symbol.upper()
        self.lookback_window = lookback_window
        
        # Initialize self-attention model
        self.attention_model = SelfAttentionAssistant()
        self.optimizer = optim.Adam(self.attention_model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
        
        # Performance tracking for self-tuning
        self.decision_history = deque(maxlen=1000)
        self.performance_metrics = {
            'correct_predictions': 0,
            'total_predictions': 0,
            'accuracy': 0.0,
            'agent_weights': {'a1': 0.5, 'a2': 0.5},
            'confidence_threshold': 0.6
        }
        
        # Training data collection
        self.training_data = []
        self.is_trained = False
        
        print(f"🧠 Self-Attention Assistant initialized for {self.stock_symbol}")

    def collect_training_data(self, historical_data, days=30):
        """Collect training data from historical performance"""
        print(f"📊 Collecting training data for {days} days...")
        
        # Simulate historical decisions and outcomes
        training_samples = []
        
        for i in range(self.lookback_window, len(historical_data) - 1):
            # Get market features for current window
            window_data = historical_data.iloc[i-self.lookback_window:i]
            market_features = self.extract_market_features(window_data)
            
            # Get next day's price movement (ground truth)
            current_price = historical_data.iloc[i]['Close']
            next_price = historical_data.iloc[i+1]['Close']
            price_change = (next_price - current_price) / current_price
            
            # Convert to decision label
            if price_change > 0.01:  # 1% threshold
                true_decision = 2  # SELL (should have sold before drop)
            elif price_change < -0.01:
                true_decision = 1  # BUY (should have bought before rise)
            else:
                true_decision = 0  # HOLD
            
            # Simulate agent decisions (random for training)
            agent_a1_decision = np.random.randint(0, 3)
            agent_a2_decision = np.random.randint(0, 3)
            
            training_samples.append({
                'market_features': market_features,
                'agent_decisions': [agent_a1_decision, agent_a2_decision],
                'true_decision': true_decision,
                'price_change': price_change
            })
        
        self.training_data = training_samples
        print(f"✅ Collected {len(training_samples)} training samples")

    def extract_market_features(self, window_data):
        """Extract relevant market features for attention model"""
        features = []
        
        # Price features
        close_prices = window_data['Close'].values
        returns = np.diff(close_prices) / close_prices[:-1]
        features.extend([
            np.mean(returns),           # Average return
            np.std(returns),            # Volatility
            returns[-1],                # Last return
            (close_prices[-1] - close_prices[0]) / close_prices[0]  # Total return
        ])
        
        # Volume features
        volumes = window_data['Volume'].values
        features.extend([
            np.mean(volumes),           # Average volume
            volumes[-1] / np.mean(volumes) if np.mean(volumes) > 0 else 1.0  # Volume ratio
        ])
        
        # Technical features
        sma_5 = np.mean(close_prices[-5:]) if len(close_prices) >= 5 else close_prices[-1]
        sma_10 = np.mean(close_prices[-10:]) if len(close_prices) >= 10 else close_prices[-1]
        
        features.extend([
            (close_prices[-1] - sma_5) / sma_5,    # Distance from SMA5
            (close_prices[-1] - sma_10) / sma_10,  # Distance from SMA10
            (sma_5 - sma_10) / sma_10,             # SMA crossover signal
            (np.max(close_prices) - np.min(close_prices)) / np.min(close_prices)  # Range
        ])
        
        return np.array(features, dtype=np.float32)

    def train_attention_model(self, epochs=100):
        """Train the self-attention model"""
        if not self.training_data:
            print("❌ No training data available")
            return False
        
        print(f"🎯 Training attention model for {epochs} epochs...")
        
        # Prepare training data
        X_features = []
        X_decisions = []
        y_true = []
        
        for sample in self.training_data:
            # Reshape market features for sequence input
            features = sample['market_features'].reshape(1, -1)  # [1, feature_dim]
            X_features.append(features)
            X_decisions.append(sample['agent_decisions'])
            y_true.append(sample['true_decision'])
        
        # Convert to tensors
        X_features = torch.FloatTensor(np.array(X_features))  # [batch, 1, feature_dim]
        X_decisions = torch.LongTensor(np.array(X_decisions))  # [batch, 2]
        y_true = torch.LongTensor(np.array(y_true))  # [batch]
        
        # Training loop
        self.attention_model.train()
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            
            # Forward pass
            predictions, weights, attention = self.attention_model(X_features, X_decisions)
            
            # Calculate loss
            loss = self.criterion(predictions, y_true)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            if epoch % 20 == 0:
                accuracy = (torch.argmax(predictions, dim=1) == y_true).float().mean()
                print(f"Epoch {epoch}: Loss = {loss.item():.4f}, Accuracy = {accuracy.item():.3f}")
        
        self.is_trained = True
        print("✅ Attention model training completed!")
        return True

    def assist_decision(self, market_data, agent_a1_decision, agent_a2_decision):
        """Main function: assist A1 and A2 with weighted decision"""
        if not self.is_trained:
            print("⚠️ Attention model not trained, using simple consensus")
            return self.simple_consensus(agent_a1_decision, agent_a2_decision)
        
        # Extract current market features
        market_features = self.extract_market_features(market_data)
        
        # Convert agent decisions to numerical
        decision_map = {'HOLD': 0, 'BUY': 1, 'SELL': 2}
        a1_num = decision_map.get(agent_a1_decision, 0)
        a2_num = decision_map.get(agent_a2_decision, 0)
        
        # Prepare input tensors
        features_tensor = torch.FloatTensor(market_features).unsqueeze(0).unsqueeze(0)  # [1, 1, features]
        decisions_tensor = torch.LongTensor([[a1_num, a2_num]])  # [1, 2]
        
        # Get attention model prediction
        self.attention_model.eval()
        with torch.no_grad():
            prediction, weights, attention = self.attention_model(features_tensor, decisions_tensor)
            
            # Extract results
            final_probs = prediction[0].numpy()
            agent_weights = weights[0].numpy()  # [weight_a1, weight_a2, confidence]
            
            # Get final decision
            final_decision_idx = np.argmax(final_probs)
            decision_names = ['HOLD', 'BUY', 'SELL']
            final_decision = decision_names[final_decision_idx]
            
            # Calculate confidence
            confidence = float(agent_weights[2])  # Third element is confidence
            
            # Update performance tracking
            self.performance_metrics['agent_weights']['a1'] = float(agent_weights[0])
            self.performance_metrics['agent_weights']['a2'] = float(agent_weights[1])
        
        # Log decision reasoning
        print(f"🧠 Attention Assistant:")
        print(f"   A1: {agent_a1_decision} (weight: {agent_weights[0]:.2f})")
        print(f"   A2: {agent_a2_decision} (weight: {agent_weights[1]:.2f})")
        print(f"   Final: {final_decision} (confidence: {confidence:.2f})")
        
        return {
            'decision': final_decision,
            'confidence': confidence,
            'agent_weights': {
                'a1': float(agent_weights[0]),
                'a2': float(agent_weights[1])
            },
            'reasoning': f"Weighted consensus: A1({agent_weights[0]:.2f}) + A2({agent_weights[1]:.2f})"
        }

    def simple_consensus(self, agent_a1_decision, agent_a2_decision):
        """Fallback simple consensus when model isn't trained"""
        if agent_a1_decision == agent_a2_decision:
            return {
                'decision': agent_a1_decision,
                'confidence': 0.8,
                'agent_weights': {'a1': 0.5, 'a2': 0.5},
                'reasoning': 'Simple consensus - agents agree'
            }
        else:
            return {
                'decision': 'HOLD',
                'confidence': 0.3,
                'agent_weights': {'a1': 0.5, 'a2': 0.5},
                'reasoning': 'Simple consensus - agents disagree, defaulting to HOLD'
            }

    def update_performance(self, predicted_decision, actual_outcome):
        """Update performance metrics for self-tuning"""
        is_correct = self.evaluate_prediction(predicted_decision, actual_outcome)
        
        self.performance_metrics['total_predictions'] += 1
        if is_correct:
            self.performance_metrics['correct_predictions'] += 1
        
        self.performance_metrics['accuracy'] = (
            self.performance_metrics['correct_predictions'] / 
            self.performance_metrics['total_predictions']
        )
        
        # Store for analysis
        self.decision_history.append({
            'timestamp': datetime.now(),
            'prediction': predicted_decision,
            'actual': actual_outcome,
            'correct': is_correct,
            'accuracy': self.performance_metrics['accuracy']
        })

    def evaluate_prediction(self, prediction, actual_outcome):
        """Evaluate if prediction was correct"""
        if prediction == 'BUY' and actual_outcome == 'UP':
            return True
        elif prediction == 'SELL' and actual_outcome == 'DOWN':
            return True
        elif prediction == 'HOLD' and actual_outcome == 'FLAT':
            return True
        return False

    def save_model(self, filepath=None):
        """Save trained attention model"""
        if filepath is None:
            filepath = f"attention_assistant_{self.stock_symbol}.pth"
        
        torch.save({
            'model_state_dict': self.attention_model.state_dict(),
            'performance_metrics': self.performance_metrics,
            'is_trained': self.is_trained
        }, filepath)
        
        print(f"💾 Attention model saved to {filepath}")

    def load_model(self, filepath=None):
        """Load trained attention model"""
        if filepath is None:
            filepath = f"attention_assistant_{self.stock_symbol}.pth"
        
        if not Path(filepath).exists():
            print(f"❌ Model file {filepath} not found")
            return False
        
        checkpoint = torch.load(filepath)
        self.attention_model.load_state_dict(checkpoint['model_state_dict'])
        self.performance_metrics = checkpoint['performance_metrics']
        self.is_trained = checkpoint['is_trained']
        
        print(f"✅ Attention model loaded from {filepath}")
        return True

def main():
    """Main function to train the attention assistant"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Self-Attention Decision Assistant')
    parser.add_argument('symbol', help='Stock symbol (e.g., AAPL, NVDA)')
    parser.add_argument('--train', action='store_true', help='Train the attention model')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    
    args = parser.parse_args()
    
    # Initialize assistant
    assistant = TradingDecisionAssistant(args.symbol)
    
    if args.train:
        # Fetch historical data for training
        ticker = yf.Ticker(args.symbol)
        historical_data = ticker.history(period="6mo", interval="1d")
        
        if historical_data.empty:
            print(f"❌ No data available for {args.symbol}")
            return
        
        # Collect training data and train
        assistant.collect_training_data(historical_data)
        assistant.train_attention_model(epochs=args.epochs)
        assistant.save_model()
        
    else:
        # Try to load existing model
        if assistant.load_model():
            print(f"🧠 Attention assistant ready for {args.symbol}")
        else:
            print(f"❌ No trained model found. Run with --train first")

if __name__ == "__main__":
    main()

