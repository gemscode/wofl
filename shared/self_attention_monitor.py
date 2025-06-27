#!/usr/bin/env python3
"""
Self-Attention Trading Monitor
Self-tunes based on prediction accuracy without executing trades
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
from pathlib import Path
import yfinance as yf
from collections import deque
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttentionMonitor:
    """Self-tuning attention mechanism for trading prediction monitoring"""
    
    def __init__(self, stock_symbol, lookback_window=100, attention_heads=4):
        self.stock_symbol = stock_symbol.upper()
        self.lookback_window = lookback_window
        self.attention_heads = attention_heads
        
        # Prediction tracking
        self.predictions_history = deque(maxlen=1000)
        self.accuracy_metrics = {
            'correct_predictions': 0,
            'total_predictions': 0,
            'accuracy_rate': 0.0,
            'confidence_scores': deque(maxlen=100),
            'prediction_errors': deque(maxlen=100)
        }
        
        # Self-tuning parameters
        self.tuning_params = {
            'learning_rate': 0.001,
            'attention_weight': 0.5,
            'confidence_threshold': 0.7,
            'retraining_threshold': 0.6,  # Retrain if accuracy drops below 60%
            'adaptation_rate': 0.1
        }
        
        # Attention mechanism
        self.attention_weights = np.ones(self.attention_heads) / self.attention_heads
        self.feature_importance = np.ones(30) / 30  # For 30 features
        
        # Model performance tracking
        self.performance_history = []
        self.last_retrain_time = datetime.now()
        self.retrain_interval = timedelta(hours=6)  # Minimum 6 hours between retrains
        
        print(f"🧠 Self-Attention Monitor initialized for {self.stock_symbol}")
        print(f"📊 Tracking accuracy with {self.attention_heads} attention heads")

    def multi_head_attention(self, features):
        """Multi-head self-attention mechanism for feature importance"""
        batch_size, seq_len, feature_dim = features.shape
        
        # Split features into attention heads
        head_dim = feature_dim // self.attention_heads
        attention_outputs = []
        
        for head in range(self.attention_heads):
            start_idx = head * head_dim
            end_idx = (head + 1) * head_dim
            head_features = features[:, :, start_idx:end_idx]
            
            # Compute attention scores
            attention_scores = np.matmul(head_features, head_features.transpose(0, 2, 1))
            attention_scores = attention_scores / np.sqrt(head_dim)
            
            # Apply softmax
            attention_weights = self.softmax(attention_scores)
            
            # Apply attention to features
            attended_features = np.matmul(attention_weights, head_features)
            attention_outputs.append(attended_features)
        
        # Concatenate attention heads
        combined_attention = np.concatenate(attention_outputs, axis=-1)
        
        # Update feature importance based on attention
        self.update_feature_importance(combined_attention)
        
        return combined_attention

    def softmax(self, x):
        """Stable softmax implementation"""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def update_feature_importance(self, attended_features):
        """Update feature importance based on attention patterns"""
        # Calculate average attention across batch and sequence
        importance_scores = np.mean(np.abs(attended_features), axis=(0, 1))
        
        # Smooth update of feature importance
        alpha = self.tuning_params['adaptation_rate']
        self.feature_importance = (1 - alpha) * self.feature_importance + alpha * importance_scores
        
        # Normalize
        self.feature_importance = self.feature_importance / np.sum(self.feature_importance)

    def make_prediction_with_attention(self, market_data, agent_a1, agent_a2):
        """Make prediction using attention-weighted features"""
        # Get base observations
        obs_a1 = self.get_market_observation(market_data, 'daily')
        obs_a2 = self.get_market_observation(market_data, 'intraday')
        
        # Apply attention weighting to features
        weighted_obs_a1 = obs_a1 * self.feature_importance[:len(obs_a1)]
        weighted_obs_a2 = obs_a2 * self.feature_importance[:len(obs_a2)]
        
        # Get predictions from agents
        try:
            action_a1, confidence_a1 = agent_a1.predict(weighted_obs_a1, deterministic=False)
            action_a2, confidence_a2 = agent_a2.predict(weighted_obs_a2, deterministic=False)
            
            actions = ['HOLD', 'BUY', 'SELL']
            pred_a1 = actions[action_a1]
            pred_a2 = actions[action_a2]
            
            # Attention-weighted consensus
            weight_a1 = self.attention_weights[0]
            weight_a2 = self.attention_weights[1]
            
            # Calculate confidence score
            confidence_score = (weight_a1 * np.max(confidence_a1) + 
                              weight_a2 * np.max(confidence_a2))
            
            # Final prediction based on weighted consensus
            if pred_a1 == pred_a2:
                final_prediction = pred_a1
                confidence_score *= 1.2  # Boost confidence for agreement
            else:
                # Use attention weights to resolve conflict
                if weight_a1 > weight_a2:
                    final_prediction = pred_a1
                else:
                    final_prediction = pred_a2
                confidence_score *= 0.8  # Reduce confidence for disagreement
            
            return {
                'prediction': final_prediction,
                'confidence': min(confidence_score, 1.0),
                'agent_a1': pred_a1,
                'agent_a2': pred_a2,
                'attention_weights': self.attention_weights.copy(),
                'feature_importance': self.feature_importance.copy()
            }
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return {
                'prediction': 'HOLD',
                'confidence': 0.0,
                'agent_a1': 'HOLD',
                'agent_a2': 'HOLD',
                'attention_weights': self.attention_weights.copy(),
                'feature_importance': self.feature_importance.copy()
            }

    def evaluate_prediction_accuracy(self, prediction_data, actual_outcome):
        """Evaluate prediction accuracy and update metrics"""
        prediction = prediction_data['prediction']
        confidence = prediction_data['confidence']
        
        # Determine if prediction was correct
        is_correct = self.is_prediction_correct(prediction, actual_outcome)
        
        # Update accuracy metrics
        self.accuracy_metrics['total_predictions'] += 1
        if is_correct:
            self.accuracy_metrics['correct_predictions'] += 1
        
        # Calculate current accuracy rate
        self.accuracy_metrics['accuracy_rate'] = (
            self.accuracy_metrics['correct_predictions'] / 
            self.accuracy_metrics['total_predictions']
        )
        
        # Track confidence and errors
        self.accuracy_metrics['confidence_scores'].append(confidence)
        prediction_error = 0.0 if is_correct else (1.0 - confidence)
        self.accuracy_metrics['prediction_errors'].append(prediction_error)
        
        # Store prediction history
        prediction_record = {
            'timestamp': datetime.now(),
            'prediction': prediction,
            'confidence': confidence,
            'actual_outcome': actual_outcome,
            'is_correct': is_correct,
            'accuracy_rate': self.accuracy_metrics['accuracy_rate']
        }
        self.predictions_history.append(prediction_record)
        
        # Check if self-tuning is needed
        self.check_and_trigger_self_tuning()
        
        return is_correct

    def is_prediction_correct(self, prediction, actual_outcome):
        """Determine if prediction matches actual market outcome"""
        # actual_outcome should be price movement: 'UP', 'DOWN', 'FLAT'
        if prediction == 'BUY' and actual_outcome == 'UP':
            return True
        elif prediction == 'SELL' and actual_outcome == 'DOWN':
            return True
        elif prediction == 'HOLD' and actual_outcome == 'FLAT':
            return True
        else:
            return False

    def check_and_trigger_self_tuning(self):
        """Check if self-tuning is needed based on performance metrics"""
        current_accuracy = self.accuracy_metrics['accuracy_rate']
        time_since_retrain = datetime.now() - self.last_retrain_time
        
        # Trigger conditions
        accuracy_trigger = current_accuracy < self.tuning_params['retraining_threshold']
        time_trigger = time_since_retrain > self.retrain_interval
        confidence_trigger = (len(self.accuracy_metrics['confidence_scores']) > 10 and
                             np.mean(list(self.accuracy_metrics['confidence_scores'])) < 0.5)
        
        if (accuracy_trigger or confidence_trigger) and time_trigger:
            print(f"🔄 Triggering self-tuning: Accuracy={current_accuracy:.3f}, "
                  f"Avg Confidence={np.mean(list(self.accuracy_metrics['confidence_scores'])):.3f}")
            self.perform_self_tuning()

    def perform_self_tuning(self):
        """Perform self-tuning of attention weights and parameters"""
        print("🧠 Performing self-attention tuning...")
        
        # Analyze recent prediction patterns
        recent_predictions = list(self.predictions_history)[-50:]  # Last 50 predictions
        
        if len(recent_predictions) < 10:
            print("⚠️ Insufficient data for tuning")
            return
        
        # Calculate performance by prediction type
        performance_by_type = {'BUY': [], 'SELL': [], 'HOLD': []}
        for pred in recent_predictions:
            pred_type = pred['prediction']
            performance_by_type[pred_type].append(pred['is_correct'])
        
        # Update attention weights based on performance
        for i, pred_type in enumerate(['BUY', 'SELL', 'HOLD']):
            if performance_by_type[pred_type]:
                accuracy = np.mean(performance_by_type[pred_type])
                # Adjust attention weights based on performance
                if i < len(self.attention_weights):
                    adjustment = (accuracy - 0.5) * self.tuning_params['adaptation_rate']
                    self.attention_weights[i] = max(0.1, min(0.9, 
                                                   self.attention_weights[i] + adjustment))
        
        # Normalize attention weights
        self.attention_weights = self.attention_weights / np.sum(self.attention_weights)
        
        # Adjust confidence threshold based on recent performance
        recent_errors = list(self.accuracy_metrics['prediction_errors'])[-20:]
        if recent_errors:
            avg_error = np.mean(recent_errors)
            if avg_error > 0.3:  # High error rate
                self.tuning_params['confidence_threshold'] += 0.05
            elif avg_error < 0.1:  # Low error rate
                self.tuning_params['confidence_threshold'] -= 0.02
        
        # Clamp confidence threshold
        self.tuning_params['confidence_threshold'] = max(0.5, min(0.9, 
                                                        self.tuning_params['confidence_threshold']))
        
        # Update last retrain time
        self.last_retrain_time = datetime.now()
        
        print(f"✅ Self-tuning completed:")
        print(f"   Attention weights: {self.attention_weights}")
        print(f"   Confidence threshold: {self.tuning_params['confidence_threshold']:.3f}")
        print(f"   Current accuracy: {self.accuracy_metrics['accuracy_rate']:.3f}")

    def get_market_observation(self, market_data, agent_type):
        """Generate market observations (simplified)"""
        # This would be the same as in your trading system
        # Return appropriate observation vector
        if agent_type == 'daily':
            return np.random.random(30).astype(np.float32)  # Placeholder
        else:
            return np.random.random(35).astype(np.float32)  # Placeholder

    def monitor_and_predict(self, agent_a1, agent_a2, duration_hours=24):
        """Main monitoring loop without executing trades"""
        print(f"🔍 Starting {duration_hours}h monitoring session for {self.stock_symbol}")
        
        start_time = datetime.now()
        end_time = start_time + timedelta(hours=duration_hours)
        
        monitoring_results = []
        
        while datetime.now() < end_time:
            try:
                # Fetch current market data
                ticker = yf.Ticker(self.stock_symbol)
                current_data = ticker.history(period="1d", interval="5m").tail(1)
                
                if current_data.empty:
                    time.sleep(60)  # Wait 1 minute and retry
                    continue
                
                current_price = current_data['Close'].iloc[0]
                
                # Make prediction with attention
                prediction_data = self.make_prediction_with_attention(
                    current_data.iloc[0], agent_a1, agent_a2
                )
                
                # Wait for actual outcome (next price movement)
                time.sleep(300)  # Wait 5 minutes
                
                # Get updated data to evaluate prediction
                updated_data = ticker.history(period="1d", interval="5m").tail(1)
                if not updated_data.empty:
                    new_price = updated_data['Close'].iloc[0]
                    price_change = (new_price - current_price) / current_price
                    
                    # Determine actual outcome
                    if price_change > 0.002:  # 0.2% threshold
                        actual_outcome = 'UP'
                    elif price_change < -0.002:
                        actual_outcome = 'DOWN'
                    else:
                        actual_outcome = 'FLAT'
                    
                    # Evaluate prediction
                    is_correct = self.evaluate_prediction_accuracy(prediction_data, actual_outcome)
                    
                    # Log results
                    result = {
                        'timestamp': datetime.now(),
                        'price': current_price,
                        'prediction': prediction_data['prediction'],
                        'confidence': prediction_data['confidence'],
                        'actual_outcome': actual_outcome,
                        'is_correct': is_correct,
                        'accuracy_rate': self.accuracy_metrics['accuracy_rate']
                    }
                    monitoring_results.append(result)
                    
                    print(f"📊 {datetime.now().strftime('%H:%M:%S')} | "
                          f"Pred: {prediction_data['prediction']} ({prediction_data['confidence']:.2f}) | "
                          f"Actual: {actual_outcome} | "
                          f"✅ {is_correct} | "
                          f"Accuracy: {self.accuracy_metrics['accuracy_rate']:.3f}")
                
            except Exception as e:
                print(f"❌ Monitoring error: {e}")
                time.sleep(60)
        
        # Generate monitoring report
        self.generate_monitoring_report(monitoring_results)
        
        return monitoring_results

    def generate_monitoring_report(self, results):
        """Generate comprehensive monitoring report"""
        print(f"\n📊 SELF-ATTENTION MONITORING REPORT - {self.stock_symbol}")
        print("=" * 60)
        
        if not results:
            print("No monitoring data available")
            return
        
        total_predictions = len(results)
        correct_predictions = sum(1 for r in results if r['is_correct'])
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        
        print(f"Total Predictions: {total_predictions}")
        print(f"Correct Predictions: {correct_predictions}")
        print(f"Overall Accuracy: {accuracy:.3f}")
        print(f"Final Accuracy Rate: {self.accuracy_metrics['accuracy_rate']:.3f}")
        
        # Prediction type breakdown
        pred_types = {}
        for result in results:
            pred_type = result['prediction']
            if pred_type not in pred_types:
                pred_types[pred_type] = {'total': 0, 'correct': 0}
            pred_types[pred_type]['total'] += 1
            if result['is_correct']:
                pred_types[pred_type]['correct'] += 1
        
        print(f"\n📈 PREDICTION BREAKDOWN:")
        for pred_type, stats in pred_types.items():
            type_accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
            print(f"{pred_type}: {stats['correct']}/{stats['total']} ({type_accuracy:.3f})")
        
        # Attention analysis
        print(f"\n🧠 ATTENTION ANALYSIS:")
        print(f"Current attention weights: {self.attention_weights}")
        print(f"Confidence threshold: {self.tuning_params['confidence_threshold']:.3f}")
        print(f"Feature importance (top 5): {np.argsort(self.feature_importance)[-5:]}")
        
        # Save results
        self.save_monitoring_results(results)

    def save_monitoring_results(self, results):
        """Save monitoring results and model state"""
        results_dir = Path(f"self_attention_results_{self.stock_symbol}")
        results_dir.mkdir(exist_ok=True)
        
        # Save monitoring results
        df = pd.DataFrame(results)
        df.to_csv(results_dir / "monitoring_results.csv", index=False)
        
        # Save model state
        model_state = {
            'attention_weights': self.attention_weights.tolist(),
            'feature_importance': self.feature_importance.tolist(),
            'tuning_params': self.tuning_params,
            'accuracy_metrics': {
                'accuracy_rate': self.accuracy_metrics['accuracy_rate'],
                'total_predictions': self.accuracy_metrics['total_predictions'],
                'correct_predictions': self.accuracy_metrics['correct_predictions']
            },
            'last_update': datetime.now().isoformat()
        }
        
        with open(results_dir / "model_state.json", 'w') as f:
            json.dump(model_state, f, indent=2)
        
        print(f"📁 Results saved to: {results_dir}")

def main():
    """Main function for self-attention monitoring"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Self-Attention Trading Monitor')
    parser.add_argument('symbol', help='Stock symbol to monitor')
    parser.add_argument('--hours', type=int, default=6, help='Monitoring duration in hours')
    parser.add_argument('--heads', type=int, default=4, help='Number of attention heads')
    
    args = parser.parse_args()
    
    # Initialize monitor
    monitor = SelfAttentionMonitor(args.symbol, attention_heads=args.heads)
    
    # Load pre-trained agents (you would load your actual trained agents here)
    print("Loading pre-trained agents...")
    # agent_a1 = PPO.load(f"models/{args.symbol}/agents/agent_a1_{args.symbol}")
    # agent_a2 = PPO.load(f"models/{args.symbol}/agents/agent_a2_{args.symbol}")
    
    # For demo purposes, create dummy agents
    agent_a1 = None  # Replace with actual agent
    agent_a2 = None  # Replace with actual agent
    
    # Start monitoring
    # results = monitor.monitor_and_predict(agent_a1, agent_a2, args.hours)
    
    print(f"🧠 Self-Attention Monitor ready for {args.symbol}")
    print("This would run continuous monitoring and self-tuning without executing trades")

if __name__ == "__main__":
    main()

