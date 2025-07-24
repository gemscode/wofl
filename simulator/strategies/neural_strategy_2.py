# File: strategies/neural_strategy.py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from common.backtester import UniversalBacktester

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.3):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, 3) # BUY, SELL, HOLD

    def forward(self, x):
        h_lstm, _ = self.lstm(x)
        return self.fc(h_lstm[:, -1, :])

class FocalLoss(nn.Module):
    # This loss function helps with class imbalance
    def __init__(self, alpha=None, gamma=2.0):
        super(FocalLoss, self).__init__(); self.alpha = alpha; self.gamma = gamma
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((self.alpha[targets] if self.alpha is not None else 1) * (1 - pt)**self.gamma * ce_loss).mean()
        return focal_loss

class NeuralStrategy:
    def __init__(self, symbol, device, window_size=5):
        self.symbol, self.device, self.window_size = symbol, device, window_size
        self.scaler = StandardScaler()
        self.model = None

    def _add_features(self, data):
        df = data.copy()
        df['SMA_10'] = df['close'].rolling(window=10).mean()
        df['SMA_30'] = df['close'].rolling(window=30).mean()
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        df['RSI'] = 100 - (100 / (1 + (gain / (loss + 1e-9))))
        # LOGIC FIX: Properly one-hot encode the HMM regime state feature
        if 'regime_state' in df.columns:
            df = pd.get_dummies(df, columns=['regime_state'], prefix='regime', dummy_na=False)
        df = df.bfill().ffill()
        print("[FEATURES] Added TA and HMM features.")
        return df

    def _prepare_data(self, data):
        data_with_features = self._add_features(data)
        data_with_features['labels'] = self.get_triple_barrier_labels(data, upper_pct=0.005, lower_pct=-0.005, hold_duration=3)
        data_with_features = data_with_features.dropna(subset=['labels'])
        # Dynamically get feature columns after one-hot encoding
        feature_cols = sorted([col for col in data_with_features.columns if col.startswith(('SMA', 'RSI', 'regime_'))])
        X, y = data_with_features[feature_cols].values, data_with_features['labels'].values
        print(f"[LABELS] {pd.Series(y).value_counts().to_dict()}")
        X_scaled = self.scaler.fit_transform(X)
        X_seq, y_seq, ts_seq = [], [], []
        for i in range(len(X_scaled) - self.window_size):
            X_seq.append(X_scaled[i:i+self.window_size])
            y_seq.append(y[i+self.window_size])
            ts_seq.append(data_with_features.index[i+self.window_size])
        return np.array(X_seq), np.array(y_seq), ts_seq, feature_cols

    def get_triple_barrier_labels(self, data, upper_pct, lower_pct, hold_duration):
        prices = data['close']
        labels = pd.Series(2, index=prices.index, dtype=int)
        for i in range(len(prices) - hold_duration):
            entry_price, future_prices = prices.iloc[i], prices.iloc[i+1:i+1+hold_duration]
            if not future_prices.empty:
                if any(future_prices >= entry_price * (1 + upper_pct)): labels.iloc[i] = 0 # BUY
                elif any(future_prices <= entry_price * (1 + lower_pct)): labels.iloc[i] = 1 # SELL
        return labels

    def train_model(self, X_train, y_train, epochs):
        self.model = LSTMModel(input_dim=X_train.shape[2]).to(self.device)
        counts = np.bincount(y_train, minlength=3); weights = 1. / (counts + 1e-6)
        class_weights = torch.tensor(weights / weights.sum(), dtype=torch.float).to(self.device)
        criterion, optimizer = FocalLoss(alpha=class_weights, gamma=2.5), torch.optim.AdamW(self.model.parameters(), lr=0.001)
        loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float), torch.tensor(y_train, dtype=torch.long)), batch_size=64, shuffle=True)
        for epoch in range(epochs):
            for X_batch, y_batch in loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad(); outputs = self.model(X_batch); loss = criterion(outputs, y_batch); loss.backward(); optimizer.step()
        print("[SUCCESS] Entry model training complete!")

    def predict_signal(self, X_live, confidence_threshold=0.6): # Relaxed threshold
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(torch.tensor(X_live, dtype=torch.float).to(self.device))
            probs = nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)
            signals = predicted.cpu().numpy()
            signals[confidence.cpu().numpy() < confidence_threshold] = 2 # Default to HOLD
            return signals

    def run_comprehensive_analysis(self, data, epochs):
        """API FIX: This now correctly orchestrates the entire neural workflow."""
        resampled_data = data.resample(f'{self.window_size}min').agg({
            'open':'first', 'high':'max', 'low':'min', 'close':'last', 'volume':'sum', 'regime_state':'last'
        }).dropna()
        X, y, timestamps, _ = self._prepare_data(resampled_data)
        if len(X) < 50: return {'test': {'error': "Not enough data to train/test."}}
        
        train_end, val_end = int(len(X) * 0.7), int(len(X) * 0.85)
        X_train, y_train = X[:train_end], y[:train_end]
        X_test, test_timestamps = X[val_end:], timestamps[val_end:]
        
        self.train_model(X_train, y_train, epochs=epochs)
        predictions = self.predict_signal(X_test)
        
        signals_df = pd.DataFrame({'timestamp': test_timestamps, 'signal': pd.Series(predictions).map({0:'BUY', 1:'SELL', 2:'HOLD'})})
        trade_signals = signals_df[signals_df['signal'] != 'HOLD'].to_dict('records')
        print(f"Generated {len(trade_signals)} trade signals for backtesting.")

        backtester = UniversalBacktester(data, initial_cash=25000)
        return {'test': backtester.run(trade_signals)}

