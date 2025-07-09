import pandas as pd
import numpy as np

class PatternExplorer:
    def __init__(self, data, lookback=30):
        self.data = data.copy()
        self.lookback = lookback
        self.patterns = []

    def preprocess_data(self):
        print("Preprocessing data (adding indicators)...")
        self.data['sma_5'] = self.data['close'].rolling(5).mean()
        self.data['sma_10'] = self.data['close'].rolling(10).mean()
        self.data['sma_20'] = self.data['close'].rolling(20).mean()
        self.data['rsi_14'] = self.calc_rsi(self.data['close'], 14)
        self.data['candlestick'] = self.data.apply(self.recognize_candlestick_pattern, axis=1)
        print("✅ Preprocessing complete")
        return self.data

    def calc_rsi(self, series, period):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def recognize_candlestick_pattern(self, row):
        if row['close'] > row['open'] and (row['close'] - row['open']) > (row['high'] - row['low']) * 0.7:
            return 'bullish_marubozu'
        elif row['close'] < row['open'] and (row['open'] - row['close']) > (row['high'] - row['low']) * 0.7:
            return 'bearish_marubozu'
        return 'none'

    def discover_patterns(self):
        print("Discovering patterns in exploration phase...")
        pattern_db = []
        for i in range(self.lookback, len(self.data)):
            window = self.data.iloc[i-self.lookback:i]
            trend = (window['sma_10'].iloc[-1] - window['sma_10'].iloc[0]) / window['sma_10'].iloc[0] if window['sma_10'].iloc[0] != 0 else 0
            candle_counts = window['candlestick'].value_counts().to_dict()
            rsi = window['rsi_14'].iloc[-1]
            if i+5 < len(self.data):
                outcome = (self.data['close'].iloc[i+5] - self.data['close'].iloc[i]) / self.data['close'].iloc[i]
            else:
                outcome = 0
            pattern_db.append({
                'index': i,
                'trend': trend,
                'candle_counts': candle_counts,
                'rsi': rsi,
                'outcome': outcome
            })
        self.patterns = pattern_db
        print(f"✅ Discovered {len(pattern_db)} patterns")
        return pattern_db

    def analyze_patterns(self):
        print("Analyzing pattern effectiveness...")
        df = pd.DataFrame(self.patterns)
        df['success'] = df['outcome'] > 0.02
        print(f"Pattern analysis: {df['success'].sum()} successful out of {len(df)} patterns")
        return df

    def generate_guidance(self, current_window):
        if len(current_window) < self.lookback:
            return [0, 0, 0]
        trend = (current_window['sma_10'].iloc[-1] - current_window['sma_10'].iloc[0]) / current_window['sma_10'].iloc[0] if current_window['sma_10'].iloc[0] != 0 else 0
        rsi = current_window['rsi_14'].iloc[-1]
        min_dist = float('inf')
        best_outcome = 0
        for p in self.patterns:
            dist = np.sqrt((p['trend'] - trend)**2 + (p['rsi'] - rsi)**2)
            if dist < min_dist:
                min_dist = dist
                best_outcome = p['outcome']
        return [min_dist, best_outcome, 1 if best_outcome > 0.02 else 0]

