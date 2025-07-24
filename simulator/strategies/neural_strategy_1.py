# File: strategies/neural_strategy.py

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from common.backtester import UniversalBacktester

# This remains the same as it's a solid labeling strategy
def get_triple_barrier_labels(prices, upper_pct=0.015, lower_pct=0.015, time_limit_bars=24):
    labels = pd.Series(0, index=prices.index, dtype=int)
    for i in range(len(prices) - time_limit_bars):
        entry_price = prices.iloc[i]
        window = prices.iloc[i+1 : i+1+time_limit_bars]
        hit_upper = window[window >= entry_price * (1 + upper_pct)]
        hit_lower = window[window <= entry_price * (1 - lower_pct)]
        if not hit_upper.empty and (hit_lower.empty or hit_upper.index[0] <= hit_lower.index[0]):
            labels.iloc[i] = 1 # BUY
        elif not hit_lower.empty:
            labels.iloc[i] = 2 # SELL
    return labels

class EntryExitDataPreprocessor:
    def __init__(self, seq_len=20): # Longer sequence for more context
        self.seq_len = seq_len
        self.scalers = {}

    def prepare_market_features(self, data, hmm_regimes=None, ts_signals=None):
        f = pd.DataFrame(index=data.index)
        
        # Core Features
        f['returns'] = data['close'].pct_change()
        f['volatility_5'] = f['returns'].rolling(window=5).std()
        f['volatility_20'] = f['returns'].rolling(window=20).std()
        
        # Advanced Feature Set
        # 1. RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        f['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-9)))

        # 2. Bollinger Bands (as requested in memory)
        ma_20 = data['close'].rolling(window=20).mean()
        std_20 = data['close'].rolling(window=20).std()
        f['bb_upper'] = ma_20 + (std_20 * 2)
        f['bb_lower'] = ma_20 - (std_20 * 2)
        f['bb_width'] = (f['bb_upper'] - f['bb_lower']) / ma_20 # Normalized width
        f['bb_pct'] = (data['close'] - f['bb_lower']) / (f['bb_upper'] - f['bb_lower']) # Position within bands

        # 3. Average True Range (ATR) - Key volatility measure
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        f['atr'] = tr.rolling(window=14).mean()
        
        # Integrate HMM regimes
        if hmm_regimes:
            hmm_df = pd.DataFrame(hmm_regimes).set_index(pd.to_datetime(pd.DataFrame(hmm_regimes)['timestamp'])).sort_index()
            # One-hot encode the regime state
            for regime in hmm_df['regime_state'].unique():
                f[f'hmm_regime_{regime}'] = (hmm_df['regime_state'] == regime).astype(int)
            f = f.fillna(0)
        
        return f.bfill().ffill().astype(np.float32)

    def fit_transform_entry(self, data, hmm_regimes=None, ts_signals=None):
        features = self.prepare_market_features(data, hmm_regimes, ts_signals)
        labels = get_triple_barrier_labels(data['close'])
        
        common_index = features.index.intersection(labels.index)
        features = features.loc[common_index]
        labels = labels.loc[common_index]
        
        for col in features.columns:
            scaler = MinMaxScaler(feature_range=(-1, 1))
            features[col] = scaler.fit_transform(features[col].values.reshape(-1, 1)).flatten()
            
        seqs = [features.iloc[i-self.seq_len:i].values for i in range(self.seq_len, len(features))]
        final_labels = labels.iloc[self.seq_len:].values
        
        return np.array(seqs), final_labels

# The rest of the Neural Strategy file remains the same, as the core model
# architecture and training loop are solid. The changes above are what matter.
# ... (FocalLoss, TradingDataset, EnhancedEntryModel, NeuralStrategy classes)
class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2): super().__init__(); self.gamma, self.weight = gamma, weight
    def forward(self, i, t): ce = nn.functional.cross_entropy(i, t, weight=self.weight, reduction='none'); pt = torch.exp(-ce); return ((1 - pt)**self.gamma * ce).mean()

class TradingDataset(Dataset):
    def __init__(self, seq, lbl): self.sequences, self.labels = torch.FloatTensor(seq), torch.LongTensor(lbl)
    def __len__(self): return len(self.sequences)
    def __getitem__(self, idx): return self.sequences[idx], self.labels[idx]

class EnhancedEntryModel(nn.Module):
    def __init__(self, input_features):
        super().__init__()
        self.conv1 = nn.Conv1d(input_features, 32, 2, padding=1); self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU(); self.dropout = nn.Dropout(0.5)
        self.bilstm = nn.LSTM(32, 64, 1, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(128, 3); self.confidence_head = nn.Linear(128, 1)
    def forward(self, x):
        x = self.dropout(self.relu(self.bn1(self.conv1(x.transpose(1, 2))))).transpose(1, 2)
        features = self.bilstm(x)[0][:, -1, :]
        return self.fc(features), torch.sigmoid(self.confidence_head(features))

class NeuralStrategy:
    def __init__(self, symbol, device, window_size=5):
        self.symbol = symbol
        self.device = device
        self.entry_model = None
        self.preprocessor = EntryExitDataPreprocessor()
        self.window_size = max(3, min(window_size, 8))

    def split_data_temporal(self, data, train_ratio=0.7, val_ratio=0.15, resample=False):
        if resample:
             data = data.resample(f'{self.window_size}min').agg({
                'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum', 'close_raw': 'last'
             }).dropna()
        train_end = int(len(data) * train_ratio)
        val_end = train_end + int(len(data) * val_ratio)
        return data.iloc[:train_end], data.iloc[train_end:val_end], data.iloc[val_end:]

    def train_entry_model(self, data, epochs, hmm_regimes=None, ts_signals=None):
        print("[TRAINING] Training entry model with Triple-Barrier Labels and advanced features...")
        seq, lbl = self.preprocessor.fit_transform_entry(data, hmm_regimes, ts_signals)
        
        if len(np.unique(lbl)) < 2:
            print("[WARNING] Insufficient label variety for training."); return
        
        counts = np.bincount(lbl.astype(int), minlength=3)
        print(f"[LABELS] HOLD={counts[0]}, BUY={counts[1]}, SELL={counts[2]}")

        weights = np.array([1.0 / (c + 1e-6) for c in counts])
        weights = torch.tensor(weights / weights.sum() * 3.0, dtype=torch.float).to(self.device)
        
        self.entry_model = EnhancedEntryModel(seq.shape[2]).to(self.device)
        criterion = FocalLoss(weight=weights)
        optimizer = optim.Adam(self.entry_model.parameters(), lr=0.0005, weight_decay=1e-5)
        
        for epoch in range(epochs):
            self.entry_model.train()
            for data_b, target_b in DataLoader(TradingDataset(seq, lbl), 64, shuffle=True):
                optimizer.zero_grad()
                pred, _ = self.entry_model(data_b.to(self.device))
                loss = criterion(pred, target_b.to(self.device))
                loss.backward()
                optimizer.step()
            if (epoch + 1) % 10 == 0: print(f"Epoch {epoch+1}/{epochs}: Loss: {loss.item():.4f}")
        print("[SUCCESS] Entry model training complete!")
        
    def generate_signals(self, data, hmm_regimes=None):
        seq, _ = self.preprocessor.fit_transform_entry(data, hmm_regimes)
        if len(seq) == 0: return []
        self.entry_model.eval()
        preds, confs = [], []
        with torch.no_grad():
            for data_b, _ in DataLoader(TradingDataset(seq, np.zeros(len(seq))), 64, shuffle=False):
                p, c = self.entry_model(data_b.to(self.device))
                preds.extend(torch.max(p, 1)[1].cpu().numpy())
                confs.extend(c.cpu().numpy().flatten())
        
        print(f"[RESULTS] Predictions: HOLD={np.sum(np.equal(preds, 0))}, BUY={np.sum(np.equal(preds, 1))}, SELL={np.sum(np.equal(preds, 2))}")
        conf_thresh = np.percentile(confs, 90) if len(confs) > 0 else 0.6
        
        signals = []
        
        # Create a regime lookup for the data period
        if hmm_regimes:
            hmm_df = pd.DataFrame(hmm_regimes).set_index(pd.to_datetime(pd.DataFrame(hmm_regimes)['timestamp'])).sort_index()
            regime_lookup = hmm_df['regime_state'].reindex(data.index, method='ffill')
        
        start_idx = self.preprocessor.seq_len
        for i, (idx, row) in enumerate(data.iloc[start_idx:start_idx+len(preds)].iterrows()):
            if confs[i] > conf_thresh:
                # REGIME-AWARE LOGIC
                current_regime = regime_lookup.loc[idx] if hmm_regimes else -1 # Default to no-trade regime
                
                # This is a placeholder for your regime logic.
                # Example: Let's assume regime 0 is uptrend, 1 is downtrend.
                if current_regime == 0 and preds[i] == 1: # Only BUY in uptrend
                    signals.append({'timestamp': idx.isoformat(), 'signal': 'BUY'})
                elif current_regime == 1 and preds[i] == 2: # Only SELL in downtrend
                    signals.append({'timestamp': idx.isoformat(), 'signal': 'SELL'})
                # No signals for other regimes to avoid chop
                
        return signals

    def run_comprehensive_analysis(self, data, epochs, hmm_signals, ts_signals):
        # We need the regime states from HMM for the NN
        hmm_regimes = hmm_signals if hmm_signals and 'regime_state' in pd.DataFrame(hmm_signals).columns else None

        train_data, val_data, test_data = self.split_data_temporal(data, resample=True)
        self.train_entry_model(train_data, epochs, hmm_regimes, ts_signals)
        print("\n--- Backtest Results ---")
        
        results = {}
        for name, d in [('train', train_data), ('val', val_data), ('test', test_data)]:
            signals = self.generate_signals(d, hmm_regimes)
            backtester = UniversalBacktester(d, 25000)
            backtest_res = backtester.run(signals)
            results[name] = backtest_res
            print(f"\n{name.title()} Backtest Results:")
            for k, v in backtest_res.items():
                print(f"  {k.replace('_', ' ').title()}: {v:,.2f}" if isinstance(v, (float, np.floating)) else f"  {k.replace('_', ' ').title()}: {v}")
                
        return results

