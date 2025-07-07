#!/usr/bin/env python3

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import os
import sys
from datetime import datetime
import pickle
import json

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'optimization'))

from data_manager import FixedDataManager
from pattern_explorer import PatternExplorer

# Define the same model classes as in training
class EnhancedEntryModel(nn.Module):
    def __init__(self, input_features=8, sequence_length=15, cnn_filters=32, cnn_kernel=2, lstm_hidden=64, lstm_layers=1, dropout=0.2, num_classes=3):
        super(EnhancedEntryModel, self).__init__()
        self.conv1 = nn.Conv1d(input_features, cnn_filters, kernel_size=cnn_kernel, padding=1)
        self.conv2 = nn.Conv1d(cnn_filters, cnn_filters*2, kernel_size=cnn_kernel, padding=1)
        self.conv3 = nn.Conv1d(cnn_filters*2, cnn_filters, kernel_size=cnn_kernel, padding=1)
        self.bn1 = nn.BatchNorm1d(cnn_filters)
        self.bn2 = nn.BatchNorm1d(cnn_filters*2)
        self.bn3 = nn.BatchNorm1d(cnn_filters)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.bilstm = nn.LSTM(input_size=cnn_filters, hidden_size=lstm_hidden, num_layers=lstm_layers, batch_first=True, bidirectional=True, dropout=dropout if lstm_layers > 1 else 0)
        self.fc1 = nn.Linear(lstm_hidden * 2, lstm_hidden)
        self.fc2 = nn.Linear(lstm_hidden, lstm_hidden // 2)
        self.fc3 = nn.Linear(lstm_hidden // 2, num_classes)
        self.confidence_head = nn.Linear(lstm_hidden * 2, 1)
    
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = x.transpose(1, 2)
        lstm_out, _ = self.bilstm(x)
        features = lstm_out[:, -1, :]
        predictions = self.fc1(features)
        predictions = self.relu(predictions)
        predictions = self.dropout(predictions)
        predictions = self.fc2(predictions)
        predictions = self.relu(predictions)
        predictions = self.dropout(predictions)
        predictions = self.fc3(predictions)
        confidence = torch.sigmoid(self.confidence_head(features))
        return predictions, confidence

class EntryExitDataPreprocessor:
    def __init__(self, sequence_length=15, prediction_horizon=1):
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.scalers = {}
    
    def prepare_market_features(self, data):
        features_df = pd.DataFrame()
        features_df['returns_1'] = data['close'].pct_change(1)
        features_df['returns_3'] = data['close'].pct_change(3)
        features_df['returns_5'] = data['close'].pct_change(5)
        features_df['volume_ratio'] = data['volume'] / data['volume'].rolling(10).mean()
        features_df['volume_momentum'] = data['volume'].pct_change(1)
        features_df['price_position'] = (data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8)
        features_df['volatility'] = data['close'].rolling(5).std() / data['close'].rolling(5).mean()
        features_df['rsi_fast'] = self._calculate_rsi(data['close'], 5)
        features_df = features_df.bfill().ffill()
        return features_df
    
    def _calculate_rsi(self, series, period=5):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-8)
        return 100 - (100 / (1 + rs)) / 100.0
    
    def create_entry_labels(self, data):
        future_returns = data['close'].shift(-self.prediction_horizon) / data['close'] - 1
        long_threshold = 0.0015
        short_threshold = -0.0015
        labels = np.zeros(len(future_returns))
        labels[future_returns > long_threshold] = 1
        labels[future_returns < short_threshold] = 2
        return labels
    
    def fit_transform_entry(self, data):
        features = self.prepare_market_features(data)
        for column in features.columns:
            scaler = MinMaxScaler(feature_range=(-1, 1))
            features[column] = scaler.fit_transform(features[column].values.reshape(-1, 1)).flatten()
            self.scalers[column] = scaler
        
        labels = self.create_entry_labels(data)
        sequences = []
        sequence_labels = []
        for i in range(self.sequence_length, len(features) - self.prediction_horizon):
            seq = features.iloc[i-self.sequence_length:i].values
            label = labels[i]
            sequences.append(seq)
            sequence_labels.append(label)
        return np.array(sequences), np.array(sequence_labels)

def get_device():
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("✅ Using Mac M-series GPU (MPS)")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print("✅ Using NVIDIA GPU (CUDA)")
        return torch.device("cuda")
    else:
        print("⚠️ Using CPU only")
        return torch.device("cpu")

def calculate_accuracy(predictions, true_labels):
    """Calculate accuracy for multi-class predictions"""
    correct = np.sum(predictions == true_labels)
    total = len(predictions)
    return correct / total

def calculate_signal_performance(predictions, confidences, true_labels, confidence_threshold=0.5):
    """Calculate performance metrics for trading signals"""
    # Filter high confidence predictions
    high_conf_mask = confidences > confidence_threshold
    high_conf_predictions = predictions[high_conf_mask]
    high_conf_labels = true_labels[high_conf_mask]
    
    if len(high_conf_predictions) == 0:
        return {"message": "No high confidence predictions"}
    
    # Calculate accuracy for high confidence predictions
    accuracy = calculate_accuracy(high_conf_predictions, high_conf_labels)
    
    # Count signal types - CONVERT TO PYTHON INTS
    buy_signals = int(np.sum(high_conf_predictions == 1))
    sell_signals = int(np.sum(high_conf_predictions == 2))
    hold_signals = int(np.sum(high_conf_predictions == 0))
    
    return {
        "total_high_conf_signals": int(len(high_conf_predictions)),
        "accuracy": float(accuracy),
        "buy_signals": buy_signals,
        "sell_signals": sell_signals,
        "hold_signals": hold_signals,
        "signal_distribution": [int(x) for x in np.bincount(high_conf_predictions)]  # Convert to list of ints
    }

def run_backtest_simulation(data, predictions, confidences, initial_cash=25000):
    """Simple backtest simulation on test data"""
    cash = initial_cash
    position = None
    trades = []
    portfolio_values = []
    
    confidence_threshold = 0.5
    sequence_length = 15
    
    # Align data with predictions (skip first sequence_length points)
    test_data = data.iloc[sequence_length:sequence_length + len(predictions)].copy()
    
    for i, (idx, row) in enumerate(test_data.iterrows()):
        if i >= len(predictions):
            break
            
        current_price = row['close_raw']
        prediction = predictions[i]
        confidence = confidences[i]
        
        # Close existing position after 10 periods or if opposite signal
        if position is not None:
            time_held = i - position['entry_index']
            should_close = False
            
            if time_held >= 10:  # Max holding period
                should_close = True
            elif position['type'] == 'LONG' and prediction == 2 and confidence > confidence_threshold:
                should_close = True
            elif position['type'] == 'SHORT' and prediction == 1 and confidence > confidence_threshold:
                should_close = True
            
            if should_close:
                if position['type'] == 'LONG':
                    cash += position['shares'] * current_price
                    pnl = (current_price - position['entry_price']) / position['entry_price']
                else:  # SHORT
                    profit = position['shares'] * (position['entry_price'] - current_price)
                    cash += profit
                    pnl = (position['entry_price'] - current_price) / position['entry_price']
                
                trades.append({
                    'type': position['type'],
                    'entry_price': float(position['entry_price']),
                    'exit_price': float(current_price),
                    'pnl': float(pnl),
                    'hold_time': int(time_held)
                })
                position = None
        
        # Open new position if high confidence signal and no current position
        if position is None and confidence > confidence_threshold:
            max_position_value = 10000
            
            if prediction == 1:  # BUY signal
                shares = int(min(cash, max_position_value) // current_price)
                if shares > 0:
                    cash -= shares * current_price
                    position = {
                        'type': 'LONG',
                        'shares': shares,
                        'entry_price': current_price,
                        'entry_index': i
                    }
            elif prediction == 2:  # SELL signal
                shares = int(max_position_value // current_price)
                if shares > 0:
                    position = {
                        'type': 'SHORT',
                        'shares': shares,
                        'entry_price': current_price,
                        'entry_index': i
                    }
        
        # Calculate portfolio value
        total_value = cash
        if position is not None:
            if position['type'] == 'LONG':
                total_value += position['shares'] * current_price
            else:  # SHORT
                unrealized_profit = position['shares'] * (position['entry_price'] - current_price)
                total_value += unrealized_profit
        
        portfolio_values.append(float(total_value))
    
    # Close final position if exists
    if position is not None and len(test_data) > 0:
        final_price = test_data.iloc[-1]['close_raw']
        if position['type'] == 'LONG':
            cash += position['shares'] * final_price
            pnl = (final_price - position['entry_price']) / position['entry_price']
        else:
            profit = position['shares'] * (position['entry_price'] - final_price)
            cash += profit
            pnl = (position['entry_price'] - final_price) / position['entry_price']
        
        trades.append({
            'type': position['type'],
            'entry_price': float(position['entry_price']),
            'exit_price': float(final_price),
            'pnl': float(pnl),
            'hold_time': len(portfolio_values) - position['entry_index']
        })
    
    final_return = (portfolio_values[-1] - initial_cash) / initial_cash if portfolio_values else 0
    
    return {
        'final_return': float(final_return),
        'total_trades': len(trades),
        'trades': trades,
        'portfolio_values': portfolio_values
    }

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Test trained entry model on 10 days of unseen data')
    parser.add_argument('--symbol', type=str, required=True, help='Symbol to test')
    parser.add_argument('--source', type=str, default='alpha', choices=['alpha', 'yahoo'])
    parser.add_argument('--test_days', type=int, default=10, help='Number of days for testing')
    args = parser.parse_args()

    device = get_device()
    
    print(f"🧪 Testing trained model for {args.symbol}")
    print(f"📊 Loading {args.test_days} days of test data...")

    # Load data manager and get available days
    dm = FixedDataManager(f"S_{args.symbol.upper()}_{args.source.upper()}")
    available_days = dm.available_days

    # Determine test days: 10 days before the last 20 days used for training
    train_days_limit = 20
    test_days_limit = args.test_days
    
    if len(available_days) < train_days_limit + test_days_limit:
        raise ValueError(f"Not enough data days available. Need at least {train_days_limit + test_days_limit} days, but only have {len(available_days)} days.")

    # Get test days (10 days before the training period)
    test_days = available_days[-(train_days_limit + test_days_limit):-train_days_limit]
    train_days = available_days[-train_days_limit:]
    
    print(f"🗓️ Training period: {train_days[0]} to {train_days[-1]}")
    print(f"🗓️ Testing period: {test_days[0]} to {test_days[-1]}")
    print(f"📈 Test days: {test_days}")

    # Load test data
    test_data = dm.load_multiple_days(test_days)
    if test_data.empty:
        raise ValueError("No test data loaded.")

    print(f"✅ Loaded {len(test_data)} test data points")

    # Preprocess test data
    explorer = PatternExplorer(test_data)
    test_data = explorer.preprocess_data()
    test_data['close_raw'] = test_data['close'].copy()

    # Prepare sequences and labels for testing
    preprocessor = EntryExitDataPreprocessor()
    sequences, true_labels = preprocessor.fit_transform_entry(test_data)

    print(f"🔍 Test data label distribution: {np.bincount(true_labels.astype(int))}")

    # Create data loader
    dataset = torch.utils.data.TensorDataset(torch.FloatTensor(sequences))
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

    # Load the trained model
    checkpoint_dir = "checkpoints"
    model_path = os.path.join(checkpoint_dir, f"entry_model_{args.symbol}.pt")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print(f"📂 Loading model from: {model_path}")

    # Initialize and load model
    model = EnhancedEntryModel(input_features=sequences.shape[2], sequence_length=sequences.shape[1])
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    print("🔮 Generating predictions on test data...")

    # Generate predictions
    predictions = []
    confidences = []
    with torch.no_grad():
        for batch in dataloader:
            data_batch = batch[0].to(device)
            pred, conf = model(data_batch)
            _, predicted = torch.max(pred, 1)
            predictions.extend(predicted.cpu().numpy())
            confidences.extend(conf.cpu().numpy().flatten())

    predictions = np.array(predictions)
    confidences = np.array(confidences)

    # Calculate performance metrics
    print("\n" + "="*50)
    print("📊 TEST RESULTS")
    print("="*50)

    overall_accuracy = calculate_accuracy(predictions, true_labels)
    print(f"🎯 Overall Accuracy: {overall_accuracy:.2%}")

    print(f"📈 Prediction distribution: {np.bincount(predictions)}")
    print(f"📊 True label distribution: {np.bincount(true_labels.astype(int))}")
    print(f"🔍 Confidence stats: min={np.min(confidences):.3f}, max={np.max(confidences):.3f}, mean={np.mean(confidences):.3f}")

    # High confidence signal analysis
    signal_performance = calculate_signal_performance(predictions, confidences, true_labels)
    print(f"\n🚀 High Confidence Signal Performance:")
    for key, value in signal_performance.items():
        print(f"   {key}: {value}")

    # Run backtest simulation
    print(f"\n💰 Running backtest simulation...")
    backtest_results = run_backtest_simulation(test_data, predictions, confidences)
    
    print(f"📈 Backtest Results:")
    print(f"   Final Return: {backtest_results['final_return']:.2%}")
    print(f"   Total Trades: {backtest_results['total_trades']}")
    
    if backtest_results['trades']:
        winning_trades = [t for t in backtest_results['trades'] if t['pnl'] > 0]
        losing_trades = [t for t in backtest_results['trades'] if t['pnl'] <= 0]
        
        print(f"   Winning Trades: {len(winning_trades)}")
        print(f"   Losing Trades: {len(losing_trades)}")
        
        if len(winning_trades) > 0:
            avg_win = np.mean([t['pnl'] for t in winning_trades])
            print(f"   Average Win: {avg_win:.2%}")
        
        if len(losing_trades) > 0:
            avg_loss = np.mean([t['pnl'] for t in losing_trades])
            print(f"   Average Loss: {avg_loss:.2%}")

    # Save test results - ENSURE ALL VALUES ARE JSON SERIALIZABLE
    results = {
        'symbol': args.symbol,
        'test_period': f"{test_days[0]} to {test_days[-1]}",
        'overall_accuracy': float(overall_accuracy),
        'signal_performance': signal_performance,  # Already converted to JSON-safe types
        'backtest_results': {
            'final_return': float(backtest_results['final_return']),
            'total_trades': int(backtest_results['total_trades'])
        },
        'prediction_distribution': [int(x) for x in np.bincount(predictions)],  # Convert to list of ints
        'confidence_stats': {
            'min': float(np.min(confidences)),
            'max': float(np.max(confidences)),
            'mean': float(np.mean(confidences))
        },
        'timestamp': datetime.now().isoformat()
    }

    # Save results to file
    results_filename = f"test_results_{args.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Test results saved to: {results_filename}")
    
    # Performance assessment
    print(f"\n🎯 PERFORMANCE ASSESSMENT:")
    if overall_accuracy > 0.4:  # Better than random for 3-class problem
        print("✅ Model shows promising accuracy on unseen data")
    else:
        print("⚠️ Model accuracy is low - consider retraining")
    
    if backtest_results['final_return'] > 0.05:  # 5% return
        print("✅ Model shows positive trading performance")
    elif backtest_results['final_return'] > 0:
        print("⚠️ Model shows modest positive performance")
    else:
        print("❌ Model shows negative trading performance - retraining recommended")

if __name__ == '__main__':
    main()

