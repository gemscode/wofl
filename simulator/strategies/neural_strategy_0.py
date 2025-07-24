import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from common.backtester import UniversalBacktester

class MovementAnalyzer:
    def __init__(self, data, window_size=5):
        # This resampling now correctly includes 'close_raw'
        self.data = data.resample(f'{window_size}min').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 
            'volume': 'sum', 'close_raw': 'last'
        }).dropna()
        self.optimal_threshold = None
    
    def analyze_price_movements(self):
        if self.optimal_threshold: return self.optimal_threshold
        returns = self.data['close'].pct_change(1).dropna().abs()
        candidates = [returns.rolling(30).std().mean() * 1.5, returns.quantile(0.95)]
        best_score, self.optimal_threshold = -1, candidates[0]
        for t in candidates:
            score = 1 - abs(((returns > t).sum() / len(returns)) - 0.05)
            if score > best_score: best_score, self.optimal_threshold = score, t
        print(f"[ANALYSIS] Optimal Threshold: {self.optimal_threshold:.5f}")
        return self.optimal_threshold
        
    def get_recommended_thresholds(self):
        return self.analyze_price_movements(), -self.analyze_price_movements()

class EntryExitDataPreprocessor:
    def __init__(self, seq_len=15):
        self.seq_len = seq_len
        self.scalers = {}
        self.adaptive_thresholds = None
        
    def set_adaptive_thresholds(self, long, short):
        self.adaptive_thresholds = long, short
        
    def prepare_market_features(self, data, hmm_signals=None, ts_signals=None):
        f = pd.DataFrame(index=data.index)
        f['returns_1'] = data['close'].pct_change(1)
        f['volatility'] = f['returns_1'].rolling(5).std()
        f['rsi'] = 100 - (100 / (1 + (d := data['close'].diff()).where(d > 0, 0).rolling(5).mean() / (-d.where(d < 0, 0).rolling(5).mean() + 1e-9)))
        f['trend'] = np.sign(f['returns_1'].rolling(5).mean())
        
        # Integrate HMM signals as features
        if hmm_signals:
            hmm_df = pd.DataFrame(hmm_signals)
            hmm_df['timestamp'] = pd.to_datetime(hmm_df['timestamp'])
            # FIX: Sort index before reindexing to prevent monotonic error
            hmm_df = hmm_df.set_index('timestamp').sort_index()
            f['hmm_signal'] = hmm_df['signal'].map({'BUY': 1, 'SELL': -1}).reindex(f.index, method='ffill').fillna(0)
        else:
            f['hmm_signal'] = 0

        if ts_signals:
            ts_df = pd.DataFrame(ts_signals)
            ts_df['timestamp'] = pd.to_datetime(ts_df['timestamp'])
            # FIX: Sort index before reindexing
            ts_df = ts_df.set_index('timestamp').sort_index()
            f['ts_signal'] = ts_df['signal'].map({'BUY': 1, 'SELL': -1}).reindex(f.index, method='ffill').fillna(0)
        else:
            f['ts_signal'] = 0
            
        return f.bfill().ffill()
    
    def create_entry_labels(self, data):
        fut_ret = data['close'].shift(-1) / data['close'] - 1
        l, s = self.adaptive_thresholds or (0.0015, -0.0015)
        lbl = np.zeros(len(fut_ret)); lbl[fut_ret > l] = 1; lbl[fut_ret < s] = 2
        return lbl
        
    def fit_transform_entry(self, data, hmm_signals=None, ts_signals=None):
        features = self.prepare_market_features(data, hmm_signals, ts_signals)
        labels = self.create_entry_labels(data)
        for col in features.columns:
            scaler = MinMaxScaler(feature_range=(-1, 1));
            features[col] = scaler.fit_transform(features[col].values.reshape(-1, 1)).flatten()
        seqs = [features.iloc[i-self.seq_len:i].values for i in range(self.seq_len, len(features))]
        return np.array(seqs), labels[self.seq_len:]

# Other classes (FocalLoss, TradingDataset, EnhancedEntryModel) are unchanged
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
             # This resampling now correctly includes 'close_raw'
             data = data.resample(f'{self.window_size}min').agg({
                'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 
                'volume': 'sum', 'close_raw': 'last'
             }).dropna()
        train_end = int(len(data) * train_ratio)
        val_end = train_end + int(len(data) * val_ratio)
        return data.iloc[:train_end], data.iloc[train_end:val_end], data.iloc[val_end:]
        
    def train_entry_model(self, data, epochs=30, hmm_signals=None, ts_signals=None):
        print("[TRAINING] Training entry model...")
        seq, lbl = self.preprocessor.fit_transform_entry(data, hmm_signals, ts_signals)
        if len(np.unique(lbl)) < 2: print("[WARNING] Insufficient label variety."); return
        counts = np.bincount(lbl.astype(int), minlength=3)
        weights = np.array([1.0 / (c + 1e-6) for c in counts])
        weights = weights / weights.sum() * 3.0
        weights = torch.tensor(weights, dtype=torch.float).to(self.device)
        self.entry_model = EnhancedEntryModel(seq.shape[2]).to(self.device)
        criterion, optimizer = FocalLoss(weight=weights), optim.Adam(self.entry_model.parameters(), lr=0.001, weight_decay=1e-4)
        for epoch in range(epochs):
            self.entry_model.train()
            for data_b, target_b in DataLoader(TradingDataset(seq, lbl), 64, shuffle=True):
                optimizer.zero_grad(); pred, _ = self.entry_model(data_b.to(self.device))
                loss = criterion(pred, target_b.to(self.device)); loss.backward(); optimizer.step()
            if (epoch + 1) % 10 == 0: print(f"Epoch {epoch+1}/{epochs}: Loss: {loss.item():.4f}")
        print("[SUCCESS] Entry model training complete!")
        
    def generate_signals(self, data):
        seq, _ = self.preprocessor.fit_transform_entry(data)
        if len(seq) == 0: return []
        self.entry_model.eval()
        preds, confs = [], []
        with torch.no_grad():
            for data_b, _ in DataLoader(TradingDataset(seq, np.zeros(len(seq))), 64, shuffle=False):
                p, c = self.entry_model(data_b.to(self.device)); preds.extend(torch.max(p, 1)[1].cpu().numpy()); confs.extend(c.cpu().numpy().flatten())
        signals = []
        # Correctly access seq_len from the preprocessor
        start_idx = self.preprocessor.seq_len
        for i, (idx, row) in enumerate(data.iloc[start_idx:start_idx+len(preds)].iterrows()):
            if preds[i] in [1, 2]: # Simplified for now
                signals.append({'timestamp': idx.isoformat(), 'signal': 'BUY' if preds[i] == 1 else 'SELL'})
        return signals

    def run_comprehensive_analysis(self, data, epochs, hmm_signals, ts_signals):
        # The main data is passed in, resampling happens inside split_data_temporal
        train_data, val_data, test_data = self.split_data_temporal(data, resample=True)
        self.train_entry_model(train_data, epochs, hmm_signals, ts_signals)
        
        print("\n--- Backtest Results ---")
        train_results = UniversalBacktester(train_data, 25000).run(self.generate_signals(train_data))
        val_results = UniversalBacktester(val_data, 25000).run(self.generate_signals(val_data))
        test_results = UniversalBacktester(test_data, 25000).run(self.generate_signals(test_data))
        
        print("\nValidation Backtest Results:")
        print(f"Final Balance: {val_results.get('final_balance', 0):.2f}, Trades: {val_results.get('total_trades', 0)}")
        
        return {'train': train_results, 'val': val_results, 'test': test_results}

