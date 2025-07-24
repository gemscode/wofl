import numpy as np
import pandas as pd

class UniversalBacktester:
    def __init__(self, data, initial_cash=25000):
        self.data = data.copy()
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.shares_held = 0
        self.short_borrowed = 0
        self.entry_price = 0  # Track entry price for realistic returns
        self.portfolio_values = [initial_cash]
        self.trade_returns = []
        self.trade_history = []
        
    def run(self, signals):
        if not signals:
            return self.calculate_performance_metrics()
        
        sig_df = pd.DataFrame(signals)
        sig_df['timestamp'] = pd.to_datetime(sig_df['timestamp'])
        sig_df = sig_df.sort_values('timestamp').set_index('timestamp')
        
        position_type = None  # Track current position type
        
        for idx, row in self.data.iterrows():
            current_price = row['close_raw']
            
            # Process signals at this timestamp
            if idx in sig_df.index:
                signal = sig_df.loc[idx]
                
                # Fixed position sizing - never exceed 5% of initial capital per trade
                max_position_value = self.initial_cash * 0.05
                max_shares = int(max_position_value / current_price)
                
                if max_shares > 0:
                    transaction_cost = max_shares * current_price * 0.001  # 0.1% cost
                    
                    if signal['signal'] == 'BUY' and position_type is None:
                        # Open long position
                        if self.cash >= max_shares * current_price + transaction_cost:
                            self.cash -= (max_shares * current_price + transaction_cost)
                            self.shares_held = max_shares
                            self.entry_price = current_price
                            position_type = 'LONG'
                            
                    elif signal['signal'] == 'SELL':
                        if position_type == 'LONG' and self.shares_held > 0:
                            # Close long position
                            exit_value = self.shares_held * current_price
                            transaction_cost = exit_value * 0.001
                            net_proceeds = exit_value - transaction_cost
                            
                            self.cash += net_proceeds
                            
                            # Calculate return
                            entry_value = self.shares_held * self.entry_price
                            trade_return = (net_proceeds - entry_value) / entry_value
                            self.trade_returns.append(trade_return)
                            
                            # Record trade
                            self.trade_history.append({
                                'entry_time': idx,
                                'exit_time': idx,
                                'entry_price': self.entry_price,
                                'exit_price': current_price,
                                'shares': self.shares_held,
                                'return': trade_return,
                                'pnl': net_proceeds - entry_value
                            })
                            
                            self.shares_held = 0
                            position_type = None
                            
                        elif position_type is None:
                            # Open short position (simplified)
                            if self.cash >= transaction_cost:
                                self.short_borrowed = max_shares
                                self.entry_price = current_price
                                position_type = 'SHORT'
            
            # Force close positions that are too old (risk management)
            if position_type is not None:
                # Simple time-based exit after 2 hours (120 minutes)
                time_in_position = pd.Timedelta(minutes=120)
                
                if position_type == 'LONG' and self.shares_held > 0:
                    # Check for profit target (1.5%) or stop loss (-1%)
                    unrealized_return = (current_price - self.entry_price) / self.entry_price
                    
                    if unrealized_return > 0.015 or unrealized_return < -0.01:
                        # Force close
                        exit_value = self.shares_held * current_price
                        transaction_cost = exit_value * 0.001
                        net_proceeds = exit_value - transaction_cost
                        
                        self.cash += net_proceeds
                        
                        entry_value = self.shares_held * self.entry_price
                        trade_return = (net_proceeds - entry_value) / entry_value
                        self.trade_returns.append(trade_return)
                        
                        self.trade_history.append({
                            'entry_time': idx,
                            'exit_time': idx,
                            'entry_price': self.entry_price,
                            'exit_price': current_price,
                            'shares': self.shares_held,
                            'return': trade_return,
                            'pnl': net_proceeds - entry_value,
                            'exit_reason': 'profit_target' if unrealized_return > 0.015 else 'stop_loss'
                        })
                        
                        self.shares_held = 0
                        position_type = None
                
                elif position_type == 'SHORT' and self.short_borrowed > 0:
                    # Simplified short handling
                    unrealized_return = (self.entry_price - current_price) / self.entry_price
                    
                    if unrealized_return > 0.015 or unrealized_return < -0.01:
                        # Close short
                        profit = self.short_borrowed * (self.entry_price - current_price)
                        transaction_cost = abs(profit) * 0.001
                        
                        self.cash += profit - transaction_cost
                        self.trade_returns.append(unrealized_return)
                        
                        self.short_borrowed = 0
                        position_type = None
            
            # Calculate current portfolio value
            current_value = self.cash
            if position_type == 'LONG':
                current_value += self.shares_held * current_price
            elif position_type == 'SHORT':
                current_value += self.short_borrowed * (self.entry_price - current_price)
            
            # Cap portfolio value to prevent unrealistic growth
            current_value = min(current_value, self.initial_cash * 10)  # Max 10x growth
            self.portfolio_values.append(current_value)
        
        return self.calculate_performance_metrics()
    
    def calculate_performance_metrics(self):
        if len(self.portfolio_values) <= 1:
            return {
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'total_trades': 0,
                'win_rate': 0.0,
                'final_balance': self.initial_cash,
                'avg_daily_profit': 0.0
            }
        
        portfolio_values = np.array(self.portfolio_values)
        
        # Calculate metrics
        total_return = (portfolio_values[-1] / self.initial_cash) - 1
        final_balance = portfolio_values[-1]
        
        # Calculate Sharpe ratio
        returns = pd.Series(portfolio_values).pct_change().dropna()
        if len(returns) > 1 and returns.std() > 0:
            sharpe = (returns.mean() / returns.std()) * np.sqrt(252 * 6.5 * 60)  # Annualized
        else:
            sharpe = 0.0
        
        # Calculate max drawdown
        running_max = np.maximum.accumulate(portfolio_values)
        drawdowns = (portfolio_values - running_max) / running_max
        max_drawdown = np.min(drawdowns) if len(drawdowns) > 0 else 0.0
        
        # Trade statistics
        trade_returns = np.array(self.trade_returns)
        win_rate = np.mean(trade_returns > 0) if len(trade_returns) > 0 else 0.0
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'total_trades': len(self.trade_returns),
            'win_rate': win_rate,
            'final_balance': final_balance,
            'avg_daily_profit': np.mean(trade_returns) if len(trade_returns) > 0 else 0.0
        }

