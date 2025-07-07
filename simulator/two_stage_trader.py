#!/usr/bin/env python3

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import pyswarms as pso
from datetime import datetime
import json
import argparse
import sys
import os
import pickle

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'optimization'))

from data_manager import FixedDataManager
from pattern_explorer import PatternExplorer

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

class TradingDataset(Dataset):
    def __init__(self, sequences, labels, weights=None):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels)
        self.weights = torch.FloatTensor(weights) if weights is not None else None
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        if self.weights is not None:
            return self.sequences[idx], self.labels[idx], self.weights[idx]
        return self.sequences[idx], self.labels[idx]

class ExitDataset(Dataset):
    def __init__(self, sequences, labels, position_features):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels)
        self.position_features = torch.FloatTensor(position_features)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx], self.position_features[idx]

class EnhancedEntryModel(nn.Module):
    def __init__(self, input_features=8, sequence_length=15, 
                 cnn_filters=32, cnn_kernel=2, lstm_hidden=64, 
                 lstm_layers=1, dropout=0.2, num_classes=3):
        super(EnhancedEntryModel, self).__init__()
        
        self.conv1 = nn.Conv1d(input_features, cnn_filters, kernel_size=cnn_kernel, padding=1)
        self.conv2 = nn.Conv1d(cnn_filters, cnn_filters*2, kernel_size=cnn_kernel, padding=1)
        self.conv3 = nn.Conv1d(cnn_filters*2, cnn_filters, kernel_size=cnn_kernel, padding=1)
        
        self.bn1 = nn.BatchNorm1d(cnn_filters)
        self.bn2 = nn.BatchNorm1d(cnn_filters*2)
        self.bn3 = nn.BatchNorm1d(cnn_filters)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        self.bilstm = nn.LSTM(
            input_size=cnn_filters,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0
        )
        
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
        
        lstm_out, (hidden, cell) = self.bilstm(x)
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

class ExitTimingModel(nn.Module):
    def __init__(self, input_features=8, sequence_length=15, position_features=5,
                 cnn_filters=32, lstm_hidden=64, dropout=0.2):
        super(ExitTimingModel, self).__init__()
        
        self.market_conv1 = nn.Conv1d(input_features, cnn_filters, kernel_size=2, padding=1)
        self.market_conv2 = nn.Conv1d(cnn_filters, cnn_filters, kernel_size=2, padding=1)
        
        self.market_lstm = nn.LSTM(
            input_size=cnn_filters,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        self.position_fc = nn.Linear(position_features, lstm_hidden)
        
        self.combined_fc1 = nn.Linear(lstm_hidden * 2 + lstm_hidden, lstm_hidden)
        self.combined_fc2 = nn.Linear(lstm_hidden, lstm_hidden // 2)
        self.exit_classifier = nn.Linear(lstm_hidden // 2, 2)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, market_data, position_features):
        market_data = market_data.transpose(1, 2)
        
        x = self.market_conv1(market_data)
        x = self.relu(x)
        x = self.market_conv2(x)
        x = self.relu(x)
        
        x = x.transpose(1, 2)
        lstm_out, _ = self.market_lstm(x)
        market_features = lstm_out[:, -1, :]
        
        pos_features = self.position_fc(position_features)
        pos_features = self.relu(pos_features)
        
        combined = torch.cat([market_features, pos_features], dim=1)
        
        x = self.combined_fc1(combined)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.combined_fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        exit_signal = self.exit_classifier(x)
        
        return exit_signal

class TradingSignalRecorder:
    def __init__(self):
        self.signals = []
        self.positions = []
        
    def record_entry_signal(self, timestamp, signal_type, confidence, price, market_state):
        signal = {
            'timestamp': timestamp,
            'type': signal_type,
            'confidence': confidence,
            'entry_price': price,
            'market_state': market_state,
            'status': 'open'
        }
        self.signals.append(signal)
        return len(self.signals) - 1
    
    def record_exit_signal(self, signal_id, exit_price, exit_reason):
        if signal_id < len(self.signals):
            self.signals[signal_id]['exit_price'] = exit_price
            self.signals[signal_id]['exit_reason'] = exit_reason
            self.signals[signal_id]['status'] = 'closed'
            self.signals[signal_id]['pnl'] = self._calculate_pnl(signal_id)
    
    def _calculate_pnl(self, signal_id):
        signal = self.signals[signal_id]
        entry_price = signal['entry_price']
        exit_price = signal['exit_price']
        
        if signal['type'] == 'LONG':
            return (exit_price - entry_price) / entry_price
        else:
            return (entry_price - exit_price) / entry_price
    
    def get_open_positions(self):
        return [i for i, s in enumerate(self.signals) if s['status'] == 'open']
    
    def save_signals(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.signals, f)
    
    def load_signals(self, filename):
        with open(filename, 'rb') as f:
            self.signals = pickle.load(f)

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
        
        print(f"🔍 Label distribution: HOLD={np.sum(labels==0)}, BUY={np.sum(labels==1)}, SELL={np.sum(labels==2)}")
        
        return labels
    
    def create_exit_training_data(self, data, signals):
        exit_sequences = []
        exit_labels = []
        position_features = []
        
        features = self.prepare_market_features(data)
        
        print(f"🔍 Processing {len(signals)} signals for exit training")
        
        for signal_idx, signal in enumerate(signals):
            if signal['status'] == 'closed':
                try:
                    entry_timestamp = signal['timestamp']
                    exit_timestamp = signal.get('exit_timestamp', entry_timestamp)
                    
                    entry_idx = data.index.get_loc(entry_timestamp)
                    
                    max_hold_time = min(60, len(data) - entry_idx - 10)
                    
                    for time_offset in range(5, max_hold_time, 2):
                        current_idx = entry_idx + time_offset
                        
                        if current_idx + self.sequence_length < len(features):
                            seq = features.iloc[current_idx:current_idx+self.sequence_length].values
                            
                            current_price = data['close'].iloc[current_idx]
                            unrealized_pnl = self._calculate_unrealized_pnl(
                                signal['entry_price'], current_price, signal['type']
                            )
                            
                            volatility = data['close'].iloc[max(0, current_idx-10):current_idx].std()
                            
                            pos_features = np.array([
                                unrealized_pnl,
                                time_offset / 60.0,
                                signal['confidence'],
                                volatility / current_price if current_price > 0 else 0.01,
                                1.0 if signal['type'] == 'LONG' else -1.0
                            ])
                            
                            should_exit = 1 if abs(unrealized_pnl) > 0.002 or time_offset > 30 else 0
                            
                            exit_sequences.append(seq)
                            exit_labels.append(should_exit)
                            position_features.append(pos_features)
                            
                except Exception as e:
                    print(f"🔍 Error processing signal {signal_idx}: {e}")
                    continue
        
        print(f"🔍 Generated {len(exit_sequences)} exit training samples")
        
        if len(exit_sequences) == 0:
            return np.array([]), np.array([]), np.array([])
        
        return np.array(exit_sequences), np.array(exit_labels), np.array(position_features)
    
    def _calculate_unrealized_pnl(self, entry_price, current_price, position_type):
        if position_type == 'LONG':
            return (current_price - entry_price) / entry_price
        else:
            return (entry_price - current_price) / entry_price
    
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

class TwoStageTrader:
    def __init__(self, symbol, device=None):
        self.symbol = symbol
        self.device = device or get_device()
        self.entry_model = None
        self.exit_model = None
        self.preprocessor = EntryExitDataPreprocessor()
        self.signal_recorder = TradingSignalRecorder()
        self.entry_params = None
        self.exit_params = None
        
    def load_and_prepare_data(self, source='alpha', days_limit=20):
        print(f"📊 Loading data for {self.symbol}...")
        
        dm = FixedDataManager(f"S_{self.symbol.upper()}_{source.upper()}")
        days = dm.available_days[-days_limit:]
        data = dm.load_multiple_days(days)
        
        if data.empty:
            raise ValueError(f"No data available for {self.symbol}")
        
        print(f"✅ Loaded {len(data)} data points from {len(days)} days")
        
        explorer = PatternExplorer(data)
        data = explorer.preprocess_data()
        
        data['close_raw'] = data['close'].copy()
        
        return data
    
    def train_entry_model(self, data, epochs=30):
        print("🎯 Training entry signal model...")
        
        sequences, labels = self.preprocessor.fit_transform_entry(data)
        
        train_split = 0.8
        split_idx = int(len(sequences) * train_split)
        
        train_sequences = sequences[:split_idx]
        train_labels = labels[:split_idx]
        val_sequences = sequences[split_idx:]
        val_labels = labels[split_idx:]
        
        train_dataset = TradingDataset(train_sequences, train_labels)
        val_dataset = TradingDataset(val_sequences, val_labels)
        
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        
        self.entry_model = EnhancedEntryModel(
            input_features=sequences.shape[2],
            sequence_length=sequences.shape[1]
        ).to(self.device)
        
        class_weights = torch.tensor([0.2, 0.4, 0.4]).float().to(self.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(self.entry_model.parameters(), lr=0.001)
        
        for epoch in range(epochs):
            self.entry_model.train()
            train_loss = 0
            
            for data_batch, target in train_loader:
                data_batch, target = data_batch.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                predictions, confidence = self.entry_model(data_batch)
                loss = criterion(predictions, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f"Entry model epoch {epoch+1}/{epochs}: Loss: {train_loss/len(train_loader):.4f}")
        
        print("✅ Entry model training complete!")
    
    def generate_entry_signals(self, data):
        print("📡 Generating entry signals...")
        
        sequences, _ = self.preprocessor.fit_transform_entry(data)
        
        dataset = TradingDataset(sequences, np.zeros(len(sequences)))
        dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
        
        predictions = []
        confidences = []
        
        self.entry_model.eval()
        with torch.no_grad():
            for data_batch, _ in dataloader:
                data_batch = data_batch.to(self.device)
                pred, conf = self.entry_model(data_batch)
                
                probs = torch.softmax(pred, dim=1)
                _, predicted = torch.max(pred, 1)
                
                predictions.extend(predicted.cpu().numpy())
                confidences.extend(conf.cpu().numpy().flatten())
        
        print(f"🔍 Prediction distribution: {np.bincount(predictions)}")
        print(f"🔍 Confidence stats: min={np.min(confidences):.3f}, max={np.max(confidences):.3f}, mean={np.mean(confidences):.3f}")
        
        confidence_threshold = 0.5
        high_conf_count = np.sum(np.array(confidences) > confidence_threshold)
        print(f"🔍 High confidence predictions (>{confidence_threshold}): {high_conf_count}")
        
        start_idx = self.preprocessor.sequence_length
        end_idx = start_idx + len(predictions)
        signal_data = data.iloc[start_idx:end_idx].copy()
        
        signal_count = 0
        for i, (idx, row) in enumerate(signal_data.iterrows()):
            if i >= len(predictions):
                break
                
            prediction = predictions[i]
            confidence = confidences[i]
            
            if confidence > confidence_threshold:
                if prediction == 1:
                    signal_id = self.signal_recorder.record_entry_signal(
                        timestamp=idx,
                        signal_type='LONG',
                        confidence=confidence,
                        price=row['close_raw'],
                        market_state={
                            'rsi': row.get('rsi_14', 50),
                            'volume_ratio': 1.0,
                            'volatility': row['close_raw'] * 0.01
                        }
                    )
                    signal_count += 1
                elif prediction == 2:
                    signal_id = self.signal_recorder.record_entry_signal(
                        timestamp=idx,
                        signal_type='SHORT',
                        confidence=confidence,
                        price=row['close_raw'],
                        market_state={
                            'rsi': row.get('rsi_14', 50),
                            'volume_ratio': 1.0,
                            'volatility': row['close_raw'] * 0.01
                        }
                    )
                    signal_count += 1
        
        print(f"✅ Generated {signal_count} entry signals")
    
    def simulate_exits_for_training(self, data):
        print("🎯 Simulating exits for training data...")
        
        exit_count = 0
        for signal_id, signal in enumerate(self.signal_recorder.signals):
            if signal['status'] == 'open':
                entry_time = signal['timestamp']
                entry_price = signal['entry_price']
                signal_type = signal['type']
                
                future_data = data[data.index > entry_time].head(30)
                
                best_exit_price = entry_price
                best_exit_time = entry_time
                
                for idx, row in future_data.iterrows():
                    current_price = row['close_raw']
                    
                    if signal_type == 'LONG':
                        pnl = (current_price - entry_price) / entry_price
                        if pnl > 0.003 or pnl < -0.002:
                            best_exit_price = current_price
                            best_exit_time = idx
                            break
                    else:
                        pnl = (entry_price - current_price) / entry_price
                        if pnl > 0.003 or pnl < -0.002:
                            best_exit_price = current_price
                            best_exit_time = idx
                            break
                
                self.signal_recorder.signals[signal_id]['exit_timestamp'] = best_exit_time
                self.signal_recorder.record_exit_signal(
                    signal_id, best_exit_price, 'simulated'
                )
                exit_count += 1
        
        print(f"✅ Simulated exits for {exit_count} signals")
    
    def train_exit_model(self, data, epochs=30):
        print("🎯 Training exit timing model...")
        
        exit_sequences, exit_labels, position_features = self.preprocessor.create_exit_training_data(
            data, self.signal_recorder.signals
        )
        
        if len(exit_sequences) == 0:
            print("❌ No exit training data available")
            return
        
        print(f"🔍 Exit training data: {len(exit_sequences)} samples")
        
        train_split = 0.8
        split_idx = int(len(exit_sequences) * train_split)
        
        train_sequences = exit_sequences[:split_idx]
        train_labels = exit_labels[:split_idx]
        train_pos_features = position_features[:split_idx]
        
        val_sequences = exit_sequences[split_idx:]
        val_labels = exit_labels[split_idx:]
        val_pos_features = position_features[split_idx:]
        
        train_dataset = ExitDataset(train_sequences, train_labels, train_pos_features)
        val_dataset = ExitDataset(val_sequences, val_labels, val_pos_features)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        self.exit_model = ExitTimingModel(
            input_features=exit_sequences.shape[2],
            sequence_length=exit_sequences.shape[1],
            position_features=position_features.shape[1]
        ).to(self.device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.exit_model.parameters(), lr=0.001)
        
        for epoch in range(epochs):
            self.exit_model.train()
            train_loss = 0
            
            for market_data, target, pos_features in train_loader:
                market_data = market_data.to(self.device)
                target = target.to(self.device)
                pos_features = pos_features.to(self.device)
                
                optimizer.zero_grad()
                predictions = self.exit_model(market_data, pos_features)
                loss = criterion(predictions, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f"Exit model epoch {epoch+1}/{epochs}: Loss: {train_loss/len(train_loader):.4f}")
        
        print("✅ Exit model training complete!")
    
    def integrated_backtest(self, data, initial_cash=25000):
        print("📈 Running integrated entry-exit backtest...")
        
        cash = initial_cash
        positions = {}
        trades = 0
        portfolio_values = []
        
        sequences, _ = self.preprocessor.fit_transform_entry(data)
        
        dataset = TradingDataset(sequences, np.zeros(len(sequences)))
        dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
        
        entry_predictions = []
        entry_confidences = []
        
        self.entry_model.eval()
        with torch.no_grad():
            for data_batch, _ in dataloader:
                data_batch = data_batch.to(self.device)
                pred, conf = self.entry_model(data_batch)
                
                _, predicted = torch.max(pred, 1)
                
                entry_predictions.extend(predicted.cpu().numpy())
                entry_confidences.extend(conf.cpu().numpy().flatten())
        
        start_idx = self.preprocessor.sequence_length
        end_idx = start_idx + len(entry_predictions)
        backtest_data = data.iloc[start_idx:end_idx].copy()
        
        MAX_POSITION_VALUE = 10000
        
        for i, (idx, row) in enumerate(backtest_data.iterrows()):
            if i >= len(entry_predictions):
                break
                
            raw_price = float(row['close_raw'])
            
            if not np.isfinite(raw_price) or raw_price <= 0:
                continue
                
            prediction = entry_predictions[i]
            confidence = np.clip(entry_confidences[i], 0, 1)
            
            if confidence > 0.5:
                if prediction == 1 and 'LONG' not in positions:
                    max_shares = min(int(cash // raw_price), int(MAX_POSITION_VALUE // raw_price))
                    shares = int(max_shares * confidence * 0.4)
                    if shares > 0 and shares * raw_price <= cash:
                        cash -= shares * raw_price
                        positions['LONG'] = {
                            'shares': shares,
                            'entry_price': raw_price,
                            'entry_time': i,
                            'confidence': confidence
                        }
                        trades += 1
                
                elif prediction == 2 and 'SHORT' not in positions:
                    max_shares = min(int(cash // raw_price), int(MAX_POSITION_VALUE // raw_price))
                    shares = int(max_shares * confidence * 0.4)
                    if shares > 0:
                        positions['SHORT'] = {
                            'shares': shares,
                            'entry_price': raw_price,
                            'entry_time': i,
                            'confidence': confidence
                        }
                        trades += 1
            
            if len(positions) > 0:
                for pos_type in list(positions.keys()):
                    position = positions[pos_type]
                    time_held = i - position['entry_time']
                    
                    if time_held > 20:
                        if pos_type == 'LONG':
                            cash += position['shares'] * raw_price
                        else:
                            profit = position['shares'] * (position['entry_price'] - raw_price)
                            cash += profit
                        
                        del positions[pos_type]
                        trades += 1
            
            if i == len(backtest_data) - 1:
                for pos_type in list(positions.keys()):
                    position = positions[pos_type]
                    if pos_type == 'LONG':
                        cash += position['shares'] * raw_price
                    else:
                        profit = position['shares'] * (position['entry_price'] - raw_price)
                        cash += profit
                    del positions[pos_type]
            
            total_value = cash
            for pos_type, position in positions.items():
                if pos_type == 'LONG':
                    total_value += position['shares'] * raw_price
                else:
                    unrealized_profit = position['shares'] * (position['entry_price'] - raw_price)
                    total_value += unrealized_profit
            
            assert np.isfinite(total_value), f"Portfolio value overflow at step {i}"
            portfolio_values.append(total_value)
        
        final_return = (portfolio_values[-1] - initial_cash) / initial_cash if portfolio_values else 0
        
        results = {
            'final_return': final_return,
            'total_trades': trades,
            'portfolio_values': portfolio_values,
            'final_positions': len(positions)
        }
        
        print(f"✅ Integrated backtest complete:")
        print(f"   Final return: {final_return:.2%}")
        print(f"   Total trades: {trades}")
        print(f"   Open positions: {len(positions)}")
        print(f"   Final portfolio value: ${portfolio_values[-1]:,.2f}")
        
        return results

def main():
    parser = argparse.ArgumentParser(description='Two-Stage Entry-Exit Trading System')
    parser.add_argument('--symbol', type=str, required=True, help='Symbol to trade')
    parser.add_argument('--source', type=str, default='alpha', choices=['alpha', 'yahoo'])
    parser.add_argument('--epochs', type=int, default=30, help='Training epochs per model')
    parser.add_argument('--days', type=int, default=20, help='Days of data')
    parser.add_argument('--mode', type=str, default='full', 
                       choices=['entry', 'exit', 'full'], help='Training mode')
    
    args = parser.parse_args()
    
    trader = TwoStageTrader(args.symbol)
    
    try:
        data = trader.load_and_prepare_data(args.source, args.days)
        
        if args.mode in ['entry', 'full']:
            trader.train_entry_model(data, args.epochs)
            trader.generate_entry_signals(data)
            trader.simulate_exits_for_training(data)
            
            trader.signal_recorder.save_signals(f'signals_{args.symbol}.pkl')
        
        if args.mode in ['exit', 'full']:
            if args.mode == 'exit':
                trader.signal_recorder.load_signals(f'signals_{args.symbol}.pkl')
            
            trader.train_exit_model(data, args.epochs)
        
        if args.mode == 'full':
            results = trader.integrated_backtest(data)
            
            results_data = {
                'symbol': args.symbol,
                'strategy_type': 'Two_Stage_Entry_Exit',
                'final_return': float(results['final_return']),
                'total_trades': int(results['total_trades']),
                'final_positions': int(results['final_positions']),
                'timestamp': datetime.now().isoformat()
            }
            
            filename = f"two_stage_results_{args.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump(results_data, f, indent=2, default=str)
            
            print(f"💾 Results saved to {filename}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()

