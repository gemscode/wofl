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

class BinaryTradingDataset(Dataset):
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

class BinaryCNNBiLSTMModel(nn.Module):
    def __init__(self, input_features=8, sequence_length=15, 
                 cnn_filters=32, cnn_kernel=2, lstm_hidden=64, 
                 lstm_layers=1, dropout=0.2):
        super(BinaryCNNBiLSTMModel, self).__init__()
        
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
        self.fc3 = nn.Linear(lstm_hidden // 2, 2)
        
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
        x = lstm_out[:, -1, :]
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        
        return x

class BinaryDataPreprocessor:
    def __init__(self, sequence_length=15, prediction_horizon=1):
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.scalers = {}
        
    def prepare_features(self, data):
        features_df = pd.DataFrame()
        
        features_df['returns_1'] = data['close'].pct_change(1)
        features_df['returns_3'] = data['close'].pct_change(3)
        features_df['returns_5'] = data['close'].pct_change(5)
        features_df['returns_10'] = data['close'].pct_change(10)
        
        features_df['volume_ratio'] = data['volume'] / data['volume'].rolling(20).mean()
        features_df['volume_momentum'] = data['volume'].pct_change(1)
        
        features_df['price_position'] = (data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8)
        features_df['volatility'] = data['close'].rolling(10).std() / data['close'].rolling(10).mean()
        
        features_df['rsi_14'] = data['rsi_14'] / 100.0 if 'rsi_14' in data.columns else self._calculate_rsi(data['close'], 14)
        features_df['sma_ratio'] = data['close'] / data['close'].rolling(20).mean()
        
        features_df['bb_position'] = (data['close'] - data['close'].rolling(20).mean()) / (2 * data['close'].rolling(20).std())
        features_df['momentum_score'] = (features_df['returns_1'] + features_df['returns_3'] + features_df['returns_5']) / 3
        
        features_df = features_df.bfill().ffill()
        return features_df
    
    def _calculate_rsi(self, series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-8)
        return 100 - (100 / (1 + rs)) / 100.0
    
    def create_binary_labels(self, data):
        future_returns = data['close'].shift(-self.prediction_horizon) / data['close'] - 1
        
        threshold = 0.002
        
        labels = np.zeros(len(future_returns))
        labels[future_returns > threshold] = 1
        labels[future_returns <= threshold] = 0
        
        return labels
    
    def create_sequences(self, features, labels):
        sequences = []
        sequence_labels = []
        
        for i in range(self.sequence_length, len(features) - self.prediction_horizon):
            seq = features.iloc[i-self.sequence_length:i].values
            label = labels[i]
            
            sequences.append(seq)
            sequence_labels.append(label)
        
        return np.array(sequences), np.array(sequence_labels)
    
    def fit_transform(self, data):
        features = self.prepare_features(data)
        
        for column in features.columns:
            scaler = MinMaxScaler(feature_range=(-1, 1))
            features[column] = scaler.fit_transform(features[column].values.reshape(-1, 1)).flatten()
            self.scalers[column] = scaler
        
        labels = self.create_binary_labels(data)
        sequences, sequence_labels = self.create_sequences(features, labels)
        
        return sequences, sequence_labels
    
    def transform(self, data):
        features = self.prepare_features(data)
        
        for column in features.columns:
            if column in self.scalers:
                features[column] = self.scalers[column].transform(features[column].values.reshape(-1, 1)).flatten()
        
        labels = self.create_binary_labels(data)
        sequences, sequence_labels = self.create_sequences(features, labels)
        
        return sequences, sequence_labels

class BinaryPSOOptimizer:
    def __init__(self, train_data, val_data, device):
        self.train_data = train_data
        self.val_data = val_data
        self.device = device
        
        self.bounds = {
            'cnn_filters': (16, 64),
            'cnn_kernel': (2, 3),
            'lstm_hidden': (32, 128),
            'lstm_layers': (1, 2),
            'dropout': (0.1, 0.4),
            'learning_rate': (5e-4, 3e-3),
            'batch_size': (32, 128)
        }
        
    def decode_params(self, particle):
        params = {}
        params['cnn_filters'] = int(particle[0])
        params['cnn_kernel'] = int(particle[1])
        params['lstm_hidden'] = int(particle[2])
        params['lstm_layers'] = int(particle[3])
        params['dropout'] = particle[4]
        params['learning_rate'] = particle[5]
        params['batch_size'] = int(particle[6])
        return params
    
    def objective_function(self, particles):
        scores = []
        
        for particle in particles:
            try:
                params = self.decode_params(particle)
                score = self.evaluate_model(params)
                scores.append(-score)
            except Exception as e:
                print(f"Error evaluating particle: {e}")
                scores.append(10.0)
        
        return np.array(scores)
    
    def evaluate_model(self, params):
        input_features = self.train_data.dataset.sequences.shape[2]
        sequence_length = self.train_data.dataset.sequences.shape[1]
        
        model = BinaryCNNBiLSTMModel(
            input_features=input_features,
            sequence_length=sequence_length,
            cnn_filters=params['cnn_filters'],
            cnn_kernel=params['cnn_kernel'],
            lstm_hidden=params['lstm_hidden'],
            lstm_layers=params['lstm_layers'],
            dropout=params['dropout']
        ).to(self.device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
        
        model.train()
        for epoch in range(5):
            for batch_idx, batch in enumerate(self.train_data):
                if batch_idx > 8:
                    break
                
                data, target = batch
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
        
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in self.val_data:
                data, target = batch
                data, target = data.to(self.device), target.to(self.device)
                outputs = model(data)
                _, predicted = torch.max(outputs, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        accuracy = correct / max(total, 1)
        return accuracy
    
    def optimize(self, n_particles=8, n_iterations=6):
        print("🔧 Starting binary PSO optimization...")
        
        min_bounds = [
            self.bounds['cnn_filters'][0], self.bounds['cnn_kernel'][0], 
            self.bounds['lstm_hidden'][0], self.bounds['lstm_layers'][0],
            self.bounds['dropout'][0], self.bounds['learning_rate'][0],
            self.bounds['batch_size'][0]
        ]
        
        max_bounds = [
            self.bounds['cnn_filters'][1], self.bounds['cnn_kernel'][1],
            self.bounds['lstm_hidden'][1], self.bounds['lstm_layers'][1],
            self.bounds['dropout'][1], self.bounds['learning_rate'][1],
            self.bounds['batch_size'][1]
        ]
        
        bounds = (min_bounds, max_bounds)
        options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
        
        optimizer = pso.single.GlobalBestPSO(
            n_particles=n_particles, 
            dimensions=len(min_bounds), 
            options=options, 
            bounds=bounds
        )
        
        best_cost, best_pos = optimizer.optimize(
            self.objective_function, 
            iters=n_iterations,
            verbose=True
        )
        
        best_params = self.decode_params(best_pos)
        
        print(f"✅ Binary PSO optimization complete!")
        print(f"   Best accuracy: {-best_cost:.4f}")
        print(f"   Best parameters: {best_params}")
        
        return best_params, -best_cost

class BinaryTradingStrategy:
    def __init__(self, symbol, device=None):
        self.symbol = symbol
        self.device = device or get_device()
        self.model = None
        self.preprocessor = BinaryDataPreprocessor()
        self.best_params = None
        
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
        
        return data
    
    def prepare_datasets(self, data, train_split=0.8, batch_size=64):
        print("🔄 Preparing binary datasets...")
        
        sequences, labels = self.preprocessor.fit_transform(data)
        
        print(f"✅ Created {len(sequences)} sequences")
        print(f"   Sequence shape: {sequences.shape}")
        
        unique, counts = np.unique(labels, return_counts=True)
        label_dist = dict(zip(unique, counts))
        print(f"   Binary label distribution: {label_dist}")
        
        split_idx = int(len(sequences) * train_split)
        
        train_sequences = sequences[:split_idx]
        train_labels = labels[:split_idx]
        
        val_sequences = sequences[split_idx:]
        val_labels = labels[split_idx:]
        
        train_dataset = BinaryTradingDataset(train_sequences, train_labels)
        val_dataset = BinaryTradingDataset(val_sequences, val_labels)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader
    
    def optimize_hyperparameters(self, train_loader, val_loader, n_particles=8, n_iterations=6):
        print("🎯 Optimizing binary hyperparameters...")
        
        pso_optimizer = BinaryPSOOptimizer(train_loader, val_loader, self.device)
        best_params, best_score = pso_optimizer.optimize(n_particles, n_iterations)
        
        self.best_params = best_params
        return best_params, best_score
    
    def train_model(self, train_loader, val_loader, params=None, epochs=40):
        if params is None:
            params = self.best_params or {
                'cnn_filters': 32, 'cnn_kernel': 2, 'lstm_hidden': 64,
                'lstm_layers': 1, 'dropout': 0.2, 'learning_rate': 0.001
            }
        
        print(f"🚀 Training binary CNN-BiLSTM model...")
        print(f"   Parameters: {params}")
        
        input_features = train_loader.dataset.sequences.shape[2]
        sequence_length = train_loader.dataset.sequences.shape[1]
        
        self.model = BinaryCNNBiLSTMModel(
            input_features=input_features,
            sequence_length=sequence_length,
            cnn_filters=params['cnn_filters'],
            cnn_kernel=params['cnn_kernel'],
            lstm_hidden=params['lstm_hidden'],
            lstm_layers=params['lstm_layers'],
            dropout=params['dropout']
        ).to(self.device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=params['learning_rate'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.7)
        
        best_accuracy = 0
        patience_counter = 0
        patience = 8
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            
            for batch in train_loader:
                data, target = batch
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()
            
            if (epoch + 1) % 5 == 0:
                self.model.eval()
                correct = 0
                total = 0
                buy_correct = 0
                sell_correct = 0
                buy_total = 0
                sell_total = 0
                
                with torch.no_grad():
                    for batch in val_loader:
                        data, target = batch
                        data, target = data.to(self.device), target.to(self.device)
                        outputs = self.model(data)
                        probs = torch.softmax(outputs, dim=1)
                        _, predicted = torch.max(outputs, 1)
                        
                        total += target.size(0)
                        correct += (predicted == target).sum().item()
                        
                        buy_mask = predicted == 1
                        sell_mask = predicted == 0
                        
                        if buy_mask.sum() > 0:
                            buy_correct += (target[buy_mask] == 1).sum().item()
                            buy_total += buy_mask.sum().item()
                        
                        if sell_mask.sum() > 0:
                            sell_correct += (target[sell_mask] == 0).sum().item()
                            sell_total += sell_mask.sum().item()
                
                accuracy = correct / max(total, 1)
                buy_precision = buy_correct / max(buy_total, 1)
                sell_precision = sell_correct / max(sell_total, 1)
                
                avg_train_loss = train_loss / len(train_loader)
                scheduler.step(avg_train_loss)
                
                print(f"Epoch {epoch+1}/{epochs}: "
                      f"Loss: {avg_train_loss:.4f}, "
                      f"Accuracy: {accuracy:.3f}, "
                      f"Buy Precision: {buy_precision:.3f}, "
                      f"Sell Precision: {sell_precision:.3f}")
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        print(f"✅ Binary training complete! Best accuracy: {best_accuracy:.4f}")
    
    def predict(self, data):
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        sequences, _ = self.preprocessor.transform(data)
        
        if len(sequences) == 0:
            return [], []
        
        dataset = BinaryTradingDataset(sequences, np.zeros(len(sequences)))
        dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
        
        predictions = []
        probabilities = []
        
        self.model.eval()
        with torch.no_grad():
            for data_batch, _ in dataloader:
                data_batch = data_batch.to(self.device)
                outputs = self.model(data_batch)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                predictions.extend(predicted.cpu().numpy())
                probabilities.extend(probs.cpu().numpy())
        
        return predictions, probabilities
    
    def binary_backtest(self, data, initial_cash=25000):
        print("📈 Running binary backtest...")
        
        predictions, probabilities = self.predict(data)
        
        if len(predictions) == 0:
            return {'final_return': 0.0, 'total_trades': 0}
        
        start_idx = self.preprocessor.sequence_length
        end_idx = start_idx + len(predictions)
        backtest_data = data.iloc[start_idx:end_idx].copy()
        
        cash = initial_cash
        holdings = 0
        position = None
        trades = 0
        portfolio_values = []
        trade_log = []
        transaction_cost = 0.01
        
        for i, (idx, row) in enumerate(backtest_data.iterrows()):
            if i >= len(predictions):
                break
                
            price = row['close']
            prediction = predictions[i]
            confidence = np.max(probabilities[i])
            
            if confidence > 0.6:
                if prediction == 1 and position != 'LONG':
                    if position == 'SHORT':
                        cash += holdings * price - holdings * transaction_cost
                        trades += 1
                        trade_log.append({'action': 'COVER', 'price': price, 'confidence': confidence})
                    
                    shares = int(cash // price)
                    if shares > 0:
                        cash -= shares * price + shares * transaction_cost
                        holdings = shares
                        position = 'LONG'
                        trades += 1
                        trade_log.append({'action': 'BUY', 'price': price, 'confidence': confidence})
                
                elif prediction == 0 and position != 'SHORT':
                    if position == 'LONG':
                        cash += holdings * price - holdings * transaction_cost
                        trades += 1
                        trade_log.append({'action': 'SELL', 'price': price, 'confidence': confidence})
                    
                    shares = int(cash // price)
                    if shares > 0:
                        cash += shares * price - shares * transaction_cost
                        holdings = -shares
                        position = 'SHORT'
                        trades += 1
                        trade_log.append({'action': 'SHORT', 'price': price, 'confidence': confidence})
            
            if position == 'LONG':
                portfolio_value = cash + holdings * price
            elif position == 'SHORT':
                portfolio_value = cash - holdings * price
            else:
                portfolio_value = cash
                
            portfolio_values.append(portfolio_value)
        
        final_return = (portfolio_values[-1] - initial_cash) / initial_cash if portfolio_values else 0
        
        peak = initial_cash
        max_drawdown = 0
        for value in portfolio_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        results = {
            'final_return': final_return,
            'total_trades': trades,
            'max_drawdown': max_drawdown,
            'portfolio_values': portfolio_values,
            'trade_log': trade_log,
            'avg_confidence': np.mean([np.max(p) for p in probabilities]),
            'trades_per_day': trades / 20
        }
        
        print(f"✅ Binary backtest complete:")
        print(f"   Final return: {final_return:.2%}")
        print(f"   Total trades: {trades}")
        print(f"   Trades per day: {results['trades_per_day']:.1f}")
        print(f"   Max drawdown: {max_drawdown:.2%}")
        print(f"   Average confidence: {results['avg_confidence']:.3f}")
        
        return results

def main():
    parser = argparse.ArgumentParser(description='Binary CNN-BiLSTM Trading Strategy')
    parser.add_argument('--symbol', type=str, required=True, help='Symbol to trade')
    parser.add_argument('--source', type=str, default='alpha', choices=['alpha', 'yahoo'])
    parser.add_argument('--epochs', type=int, default=40, help='Training epochs')
    parser.add_argument('--optimize', action='store_true', help='Run PSO optimization')
    parser.add_argument('--days', type=int, default=20, help='Days of data')
    parser.add_argument('--pso_particles', type=int, default=8, help='PSO particles')
    parser.add_argument('--pso_iters', type=int, default=6, help='PSO iterations')
    
    args = parser.parse_args()
    
    strategy = BinaryTradingStrategy(args.symbol)
    
    try:
        data = strategy.load_and_prepare_data(args.source, args.days)
        train_loader, val_loader = strategy.prepare_datasets(data)
        
        if args.optimize:
            best_params, best_score = strategy.optimize_hyperparameters(
                train_loader, val_loader, args.pso_particles, args.pso_iters
            )
            print(f"🎯 Best binary accuracy: {best_score:.4f}")
        
        strategy.train_model(train_loader, val_loader, epochs=args.epochs)
        
        results = strategy.binary_backtest(data)
        
        results_data = {
            'symbol': args.symbol,
            'strategy_type': 'Binary_CNN_BiLSTM',
            'final_return': float(results['final_return']),
            'total_trades': int(results['total_trades']),
            'trades_per_day': float(results['trades_per_day']),
            'max_drawdown': float(results['max_drawdown']),
            'avg_confidence': float(results['avg_confidence']),
            'best_params': strategy.best_params,
            'timestamp': datetime.now().isoformat()
        }
        
        filename = f"binary_cnn_bilstm_results_{args.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        print(f"💾 Binary results saved to {filename}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
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

class BinaryTradingDataset(Dataset):
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

class BinaryCNNBiLSTMModel(nn.Module):
    def __init__(self, input_features=8, sequence_length=15, 
                 cnn_filters=32, cnn_kernel=2, lstm_hidden=64, 
                 lstm_layers=1, dropout=0.2):
        super(BinaryCNNBiLSTMModel, self).__init__()
        
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
        self.fc3 = nn.Linear(lstm_hidden // 2, 2)
        
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
        x = lstm_out[:, -1, :]
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        
        return x

class BinaryDataPreprocessor:
    def __init__(self, sequence_length=15, prediction_horizon=1):
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.scalers = {}
        
    def prepare_features(self, data):
        features_df = pd.DataFrame()
        
        features_df['returns_1'] = data['close'].pct_change(1)
        features_df['returns_3'] = data['close'].pct_change(3)
        features_df['returns_5'] = data['close'].pct_change(5)
        features_df['returns_10'] = data['close'].pct_change(10)
        
        features_df['volume_ratio'] = data['volume'] / data['volume'].rolling(20).mean()
        features_df['volume_momentum'] = data['volume'].pct_change(1)
        
        features_df['price_position'] = (data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8)
        features_df['volatility'] = data['close'].rolling(10).std() / data['close'].rolling(10).mean()
        
        features_df['rsi_14'] = data['rsi_14'] / 100.0 if 'rsi_14' in data.columns else self._calculate_rsi(data['close'], 14)
        features_df['sma_ratio'] = data['close'] / data['close'].rolling(20).mean()
        
        features_df['bb_position'] = (data['close'] - data['close'].rolling(20).mean()) / (2 * data['close'].rolling(20).std())
        features_df['momentum_score'] = (features_df['returns_1'] + features_df['returns_3'] + features_df['returns_5']) / 3
        
        features_df = features_df.bfill().ffill()
        return features_df
    
    def _calculate_rsi(self, series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-8)
        return 100 - (100 / (1 + rs)) / 100.0
    
    def create_binary_labels(self, data):
        future_returns = data['close'].shift(-self.prediction_horizon) / data['close'] - 1
        
        threshold = 0.002
        
        labels = np.zeros(len(future_returns))
        labels[future_returns > threshold] = 1
        labels[future_returns <= threshold] = 0
        
        return labels
    
    def create_sequences(self, features, labels):
        sequences = []
        sequence_labels = []
        
        for i in range(self.sequence_length, len(features) - self.prediction_horizon):
            seq = features.iloc[i-self.sequence_length:i].values
            label = labels[i]
            
            sequences.append(seq)
            sequence_labels.append(label)
        
        return np.array(sequences), np.array(sequence_labels)
    
    def fit_transform(self, data):
        features = self.prepare_features(data)
        
        for column in features.columns:
            scaler = MinMaxScaler(feature_range=(-1, 1))
            features[column] = scaler.fit_transform(features[column].values.reshape(-1, 1)).flatten()
            self.scalers[column] = scaler
        
        labels = self.create_binary_labels(data)
        sequences, sequence_labels = self.create_sequences(features, labels)
        
        return sequences, sequence_labels
    
    def transform(self, data):
        features = self.prepare_features(data)
        
        for column in features.columns:
            if column in self.scalers:
                features[column] = self.scalers[column].transform(features[column].values.reshape(-1, 1)).flatten()
        
        labels = self.create_binary_labels(data)
        sequences, sequence_labels = self.create_sequences(features, labels)
        
        return sequences, sequence_labels

class BinaryPSOOptimizer:
    def __init__(self, train_data, val_data, device):
        self.train_data = train_data
        self.val_data = val_data
        self.device = device
        
        self.bounds = {
            'cnn_filters': (16, 64),
            'cnn_kernel': (2, 3),
            'lstm_hidden': (32, 128),
            'lstm_layers': (1, 2),
            'dropout': (0.1, 0.4),
            'learning_rate': (5e-4, 3e-3),
            'batch_size': (32, 128)
        }
        
    def decode_params(self, particle):
        params = {}
        params['cnn_filters'] = int(particle[0])
        params['cnn_kernel'] = int(particle[1])
        params['lstm_hidden'] = int(particle[2])
        params['lstm_layers'] = int(particle[3])
        params['dropout'] = particle[4]
        params['learning_rate'] = particle[5]
        params['batch_size'] = int(particle[6])
        return params
    
    def objective_function(self, particles):
        scores = []
        
        for particle in particles:
            try:
                params = self.decode_params(particle)
                score = self.evaluate_model(params)
                scores.append(-score)
            except Exception as e:
                print(f"Error evaluating particle: {e}")
                scores.append(10.0)
        
        return np.array(scores)
    
    def evaluate_model(self, params):
        input_features = self.train_data.dataset.sequences.shape[2]
        sequence_length = self.train_data.dataset.sequences.shape[1]
        
        model = BinaryCNNBiLSTMModel(
            input_features=input_features,
            sequence_length=sequence_length,
            cnn_filters=params['cnn_filters'],
            cnn_kernel=params['cnn_kernel'],
            lstm_hidden=params['lstm_hidden'],
            lstm_layers=params['lstm_layers'],
            dropout=params['dropout']
        ).to(self.device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
        
        model.train()
        for epoch in range(5):
            for batch_idx, batch in enumerate(self.train_data):
                if batch_idx > 8:
                    break
                
                data, target = batch
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
        
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in self.val_data:
                data, target = batch
                data, target = data.to(self.device), target.to(self.device)
                outputs = model(data)
                _, predicted = torch.max(outputs, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        accuracy = correct / max(total, 1)
        return accuracy
    
    def optimize(self, n_particles=8, n_iterations=6):
        print("🔧 Starting binary PSO optimization...")
        
        min_bounds = [
            self.bounds['cnn_filters'][0], self.bounds['cnn_kernel'][0], 
            self.bounds['lstm_hidden'][0], self.bounds['lstm_layers'][0],
            self.bounds['dropout'][0], self.bounds['learning_rate'][0],
            self.bounds['batch_size'][0]
        ]
        
        max_bounds = [
            self.bounds['cnn_filters'][1], self.bounds['cnn_kernel'][1],
            self.bounds['lstm_hidden'][1], self.bounds['lstm_layers'][1],
            self.bounds['dropout'][1], self.bounds['learning_rate'][1],
            self.bounds['batch_size'][1]
        ]
        
        bounds = (min_bounds, max_bounds)
        options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
        
        optimizer = pso.single.GlobalBestPSO(
            n_particles=n_particles, 
            dimensions=len(min_bounds), 
            options=options, 
            bounds=bounds
        )
        
        best_cost, best_pos = optimizer.optimize(
            self.objective_function, 
            iters=n_iterations,
            verbose=True
        )
        
        best_params = self.decode_params(best_pos)
        
        print(f"✅ Binary PSO optimization complete!")
        print(f"   Best accuracy: {-best_cost:.4f}")
        print(f"   Best parameters: {best_params}")
        
        return best_params, -best_cost

class BinaryTradingStrategy:
    def __init__(self, symbol, device=None):
        self.symbol = symbol
        self.device = device or get_device()
        self.model = None
        self.preprocessor = BinaryDataPreprocessor()
        self.best_params = None
        
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
        
        return data
    
    def prepare_datasets(self, data, train_split=0.8, batch_size=64):
        print("🔄 Preparing binary datasets...")
        
        sequences, labels = self.preprocessor.fit_transform(data)
        
        print(f"✅ Created {len(sequences)} sequences")
        print(f"   Sequence shape: {sequences.shape}")
        
        unique, counts = np.unique(labels, return_counts=True)
        label_dist = dict(zip(unique, counts))
        print(f"   Binary label distribution: {label_dist}")
        
        split_idx = int(len(sequences) * train_split)
        
        train_sequences = sequences[:split_idx]
        train_labels = labels[:split_idx]
        
        val_sequences = sequences[split_idx:]
        val_labels = labels[split_idx:]
        
        train_dataset = BinaryTradingDataset(train_sequences, train_labels)
        val_dataset = BinaryTradingDataset(val_sequences, val_labels)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader
    
    def optimize_hyperparameters(self, train_loader, val_loader, n_particles=8, n_iterations=6):
        print("🎯 Optimizing binary hyperparameters...")
        
        pso_optimizer = BinaryPSOOptimizer(train_loader, val_loader, self.device)
        best_params, best_score = pso_optimizer.optimize(n_particles, n_iterations)
        
        self.best_params = best_params
        return best_params, best_score
    
    def train_model(self, train_loader, val_loader, params=None, epochs=40):
        if params is None:
            params = self.best_params or {
                'cnn_filters': 32, 'cnn_kernel': 2, 'lstm_hidden': 64,
                'lstm_layers': 1, 'dropout': 0.2, 'learning_rate': 0.001
            }
        
        print(f"🚀 Training binary CNN-BiLSTM model...")
        print(f"   Parameters: {params}")
        
        input_features = train_loader.dataset.sequences.shape[2]
        sequence_length = train_loader.dataset.sequences.shape[1]
        
        self.model = BinaryCNNBiLSTMModel(
            input_features=input_features,
            sequence_length=sequence_length,
            cnn_filters=params['cnn_filters'],
            cnn_kernel=params['cnn_kernel'],
            lstm_hidden=params['lstm_hidden'],
            lstm_layers=params['lstm_layers'],
            dropout=params['dropout']
        ).to(self.device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=params['learning_rate'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.7)
        
        best_accuracy = 0
        patience_counter = 0
        patience = 8
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            
            for batch in train_loader:
                data, target = batch
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()
            
            if (epoch + 1) % 5 == 0:
                self.model.eval()
                correct = 0
                total = 0
                buy_correct = 0
                sell_correct = 0
                buy_total = 0
                sell_total = 0
                
                with torch.no_grad():
                    for batch in val_loader:
                        data, target = batch
                        data, target = data.to(self.device), target.to(self.device)
                        outputs = self.model(data)
                        probs = torch.softmax(outputs, dim=1)
                        _, predicted = torch.max(outputs, 1)
                        
                        total += target.size(0)
                        correct += (predicted == target).sum().item()
                        
                        buy_mask = predicted == 1
                        sell_mask = predicted == 0
                        
                        if buy_mask.sum() > 0:
                            buy_correct += (target[buy_mask] == 1).sum().item()
                            buy_total += buy_mask.sum().item()
                        
                        if sell_mask.sum() > 0:
                            sell_correct += (target[sell_mask] == 0).sum().item()
                            sell_total += sell_mask.sum().item()
                
                accuracy = correct / max(total, 1)
                buy_precision = buy_correct / max(buy_total, 1)
                sell_precision = sell_correct / max(sell_total, 1)
                
                avg_train_loss = train_loss / len(train_loader)
                scheduler.step(avg_train_loss)
                
                print(f"Epoch {epoch+1}/{epochs}: "
                      f"Loss: {avg_train_loss:.4f}, "
                      f"Accuracy: {accuracy:.3f}, "
                      f"Buy Precision: {buy_precision:.3f}, "
                      f"Sell Precision: {sell_precision:.3f}")
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        print(f"✅ Binary training complete! Best accuracy: {best_accuracy:.4f}")
    
    def predict(self, data):
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        sequences, _ = self.preprocessor.transform(data)
        
        if len(sequences) == 0:
            return [], []
        
        dataset = BinaryTradingDataset(sequences, np.zeros(len(sequences)))
        dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
        
        predictions = []
        probabilities = []
        
        self.model.eval()
        with torch.no_grad():
            for data_batch, _ in dataloader:
                data_batch = data_batch.to(self.device)
                outputs = self.model(data_batch)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                predictions.extend(predicted.cpu().numpy())
                probabilities.extend(probs.cpu().numpy())
        
        return predictions, probabilities
    
    def binary_backtest(self, data, initial_cash=25000):
        print("📈 Running binary backtest...")
        
        predictions, probabilities = self.predict(data)
        
        if len(predictions) == 0:
            return {'final_return': 0.0, 'total_trades': 0}
        
        start_idx = self.preprocessor.sequence_length
        end_idx = start_idx + len(predictions)
        backtest_data = data.iloc[start_idx:end_idx].copy()
        
        cash = initial_cash
        holdings = 0
        position = None
        trades = 0
        portfolio_values = []
        trade_log = []
        transaction_cost = 0.01
        
        for i, (idx, row) in enumerate(backtest_data.iterrows()):
            if i >= len(predictions):
                break
                
            price = row['close']
            prediction = predictions[i]
            confidence = np.max(probabilities[i])
            
            if confidence > 0.6:
                if prediction == 1 and position != 'LONG':
                    if position == 'SHORT':
                        cash += holdings * price - holdings * transaction_cost
                        trades += 1
                        trade_log.append({'action': 'COVER', 'price': price, 'confidence': confidence})
                    
                    shares = int(cash // price)
                    if shares > 0:
                        cash -= shares * price + shares * transaction_cost
                        holdings = shares
                        position = 'LONG'
                        trades += 1
                        trade_log.append({'action': 'BUY', 'price': price, 'confidence': confidence})
                
                elif prediction == 0 and position != 'SHORT':
                    if position == 'LONG':
                        cash += holdings * price - holdings * transaction_cost
                        trades += 1
                        trade_log.append({'action': 'SELL', 'price': price, 'confidence': confidence})
                    
                    shares = int(cash // price)
                    if shares > 0:
                        cash += shares * price - shares * transaction_cost
                        holdings = -shares
                        position = 'SHORT'
                        trades += 1
                        trade_log.append({'action': 'SHORT', 'price': price, 'confidence': confidence})
            
            if position == 'LONG':
                portfolio_value = cash + holdings * price
            elif position == 'SHORT':
                portfolio_value = cash - holdings * price
            else:
                portfolio_value = cash
                
            portfolio_values.append(portfolio_value)
        
        final_return = (portfolio_values[-1] - initial_cash) / initial_cash if portfolio_values else 0
        
        peak = initial_cash
        max_drawdown = 0
        for value in portfolio_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        results = {
            'final_return': final_return,
            'total_trades': trades,
            'max_drawdown': max_drawdown,
            'portfolio_values': portfolio_values,
            'trade_log': trade_log,
            'avg_confidence': np.mean([np.max(p) for p in probabilities]),
            'trades_per_day': trades / 20
        }
        
        print(f"✅ Binary backtest complete:")
        print(f"   Final return: {final_return:.2%}")
        print(f"   Total trades: {trades}")
        print(f"   Trades per day: {results['trades_per_day']:.1f}")
        print(f"   Max drawdown: {max_drawdown:.2%}")
        print(f"   Average confidence: {results['avg_confidence']:.3f}")
        
        return results

def main():
    parser = argparse.ArgumentParser(description='Binary CNN-BiLSTM Trading Strategy')
    parser.add_argument('--symbol', type=str, required=True, help='Symbol to trade')
    parser.add_argument('--source', type=str, default='alpha', choices=['alpha', 'yahoo'])
    parser.add_argument('--epochs', type=int, default=40, help='Training epochs')
    parser.add_argument('--optimize', action='store_true', help='Run PSO optimization')
    parser.add_argument('--days', type=int, default=20, help='Days of data')
    parser.add_argument('--pso_particles', type=int, default=8, help='PSO particles')
    parser.add_argument('--pso_iters', type=int, default=6, help='PSO iterations')
    
    args = parser.parse_args()
    
    strategy = BinaryTradingStrategy(args.symbol)
    
    try:
        data = strategy.load_and_prepare_data(args.source, args.days)
        train_loader, val_loader = strategy.prepare_datasets(data)
        
        if args.optimize:
            best_params, best_score = strategy.optimize_hyperparameters(
                train_loader, val_loader, args.pso_particles, args.pso_iters
            )
            print(f"🎯 Best binary accuracy: {best_score:.4f}")
        
        strategy.train_model(train_loader, val_loader, epochs=args.epochs)
        
        results = strategy.binary_backtest(data)
        
        results_data = {
            'symbol': args.symbol,
            'strategy_type': 'Binary_CNN_BiLSTM',
            'final_return': float(results['final_return']),
            'total_trades': int(results['total_trades']),
            'trades_per_day': float(results['trades_per_day']),
            'max_drawdown': float(results['max_drawdown']),
            'avg_confidence': float(results['avg_confidence']),
            'best_params': strategy.best_params,
            'timestamp': datetime.now().isoformat()
        }
        
        filename = f"binary_cnn_bilstm_results_{args.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        print(f"💾 Binary results saved to {filename}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()

