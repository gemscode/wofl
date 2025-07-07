import numpy as np
import pandas as pd

class DirectionalAnalyzer:
    """Analyzes market direction to find profitable bias"""
    
    def __init__(self, data):
        self.data = data
        
    def analyze_market_regime(self):
        """Determine if market is trending up, down, or sideways"""
        
        # Calculate various timeframe trends
        self.data['trend_5'] = self.data['close'].rolling(5).mean()
        self.data['trend_20'] = self.data['close'].rolling(20).mean()
        self.data['trend_50'] = self.data['close'].rolling(50).mean()
        
        # Calculate directional bias
        recent_data = self.data.tail(1000)  # Last 1000 points
        
        # Short-term bias
        short_bias = (recent_data['close'].iloc[-1] - recent_data['close'].iloc[-100]) / recent_data['close'].iloc[-100]
        
        # Medium-term bias  
        medium_bias = (recent_data['close'].iloc[-1] - recent_data['close'].iloc[-500]) / recent_data['close'].iloc[-500]
        
        # Long-term bias
        long_bias = (recent_data['close'].iloc[-1] - recent_data['close'].iloc[0]) / recent_data['close'].iloc[0]
        
        print(f"📊 DIRECTIONAL ANALYSIS:")
        print(f"   Short-term bias (100 periods): {short_bias:+.2%}")
        print(f"   Medium-term bias (500 periods): {medium_bias:+.2%}")
        print(f"   Long-term bias (1000 periods): {long_bias:+.2%}")
        
        # Determine primary direction
        avg_bias = (short_bias + medium_bias + long_bias) / 3
        
        if avg_bias > 0.02:
            direction = "BULLISH"
            strategy_bias = "LONG_ONLY"
        elif avg_bias < -0.02:
            direction = "BEARISH" 
            strategy_bias = "SHORT_ONLY"
        else:
            direction = "SIDEWAYS"
            strategy_bias = "MEAN_REVERSION"
            
        print(f"   🎯 Primary Direction: {direction}")
        print(f"   📈 Recommended Strategy: {strategy_bias}")
        
        return {
            'direction': direction,
            'strategy_bias': strategy_bias,
            'short_bias': short_bias,
            'medium_bias': medium_bias,
            'long_bias': long_bias,
            'avg_bias': avg_bias
        }
    
    def find_profitable_patterns(self):
        """Find specific patterns that are profitable"""
        
        profitable_setups = []
        
        for i in range(50, len(self.data) - 10):
            # Look for various setups
            current = self.data.iloc[i]
            future = self.data.iloc[i+10]  # 10 periods ahead
            
            # Calculate future return
            future_return = (future['close'] - current['close']) / current['close']
            
            # Pattern 1: RSI oversold + uptrend
            if (current['rsi_14'] < 30 and 
                current['close'] > current['sma_20'] and
                abs(future_return) > 0.01):  # Significant move
                
                profitable_setups.append({
                    'pattern': 'RSI_OVERSOLD_UPTREND',
                    'entry_condition': 'rsi < 30 AND price > sma_20',
                    'future_return': future_return,
                    'direction': 'LONG' if future_return > 0 else 'SHORT'
                })
            
            # Pattern 2: RSI overbought + downtrend  
            if (current['rsi_14'] > 70 and
                current['close'] < current['sma_20'] and
                abs(future_return) > 0.01):
                
                profitable_setups.append({
                    'pattern': 'RSI_OVERBOUGHT_DOWNTREND', 
                    'entry_condition': 'rsi > 70 AND price < sma_20',
                    'future_return': future_return,
                    'direction': 'SHORT' if future_return < 0 else 'LONG'
                })
        
        # Analyze profitable patterns
        if profitable_setups:
            df_setups = pd.DataFrame(profitable_setups)
            
            print(f"\n💰 PROFITABLE PATTERNS FOUND:")
            for pattern in df_setups['pattern'].unique():
                pattern_data = df_setups[df_setups['pattern'] == pattern]
                avg_return = pattern_data['future_return'].mean()
                win_rate = len(pattern_data[pattern_data['future_return'] > 0]) / len(pattern_data)
                
                print(f"   {pattern}:")
                print(f"     Average Return: {avg_return:+.2%}")
                print(f"     Win Rate: {win_rate:.1%}")
                print(f"     Occurrences: {len(pattern_data)}")
        
        return profitable_setups

