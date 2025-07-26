import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from shared.data_manager import DataManager

@dataclass
class Trade:
    """Represents a single trade."""
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    shares: int
    side: str  # 'long' or 'short'
    entry_signal_confidence: float
    exit_signal_confidence: str  # exit reason
    pnl: float
    pnl_pct: float
    hold_duration_minutes: int

@dataclass
class BacktestConfig:
    """Backtesting configuration."""
    initial_balance: float = 25000.0
    position_size_pct: float = 0.95
    transaction_cost_pct: float = 0.001
    min_hold_minutes: int = 15
    max_hold_minutes: int = 120
    stop_loss_pct: float = 0.015
    take_profit_pct: float = 0.025
    confidence_threshold: float = 0.8
    max_trades_per_day: int = 3

class TradingBacktester:
    """Comprehensive backtesting engine for trading models."""
    
    def __init__(self, symbol: str, config: BacktestConfig = None, verbose: bool = False):
        self.symbol = symbol.upper()
        self.config = config or BacktestConfig()
        self.verbose = verbose
        self.data_manager = DataManager()
        self.logger = self._setup_logging()
        
        # Trading state
        self.balance = self.config.initial_balance
        self.initial_balance = self.config.initial_balance
        self.position = None
        self.trades: List[Trade] = []
        self.daily_trades = {}
        
        # Performance tracking
        self.balance_history = []

    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the backtester."""
        logger = logging.getLogger(f"Backtester_{self.symbol}")
        logger.setLevel(logging.ERROR if not self.verbose else logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger

    def _calculate_position_size(self, price: float) -> int:
        """Calculate position size based on available capital."""
        available_capital = self.balance * self.config.position_size_pct
        shares = int(available_capital / price)
        return max(1, shares)

    def _calculate_transaction_cost(self, price: float, shares: int) -> float:
        """Calculate transaction costs."""
        trade_value = price * shares
        return trade_value * self.config.transaction_cost_pct

    def _can_trade_today(self, current_time: datetime) -> bool:
        """Check if we can make more trades today."""
        date_key = current_time.date()
        trades_today = self.daily_trades.get(date_key, 0)
        return trades_today < self.config.max_trades_per_day

    def _improved_entry_signal(self, data: pd.DataFrame, idx: int) -> Tuple[bool, float]:
        """Improved entry signal with multiple confirmations."""
        if idx < 50:
            return False, 0.0
            
        current_time = data.index[idx]
        if not self._can_trade_today(current_time):
            return False, 0.0
            
        # Multiple timeframe moving averages
        short_ma = data['close'].iloc[idx-10:idx].mean()
        medium_ma = data['close'].iloc[idx-20:idx].mean()
        long_ma = data['close'].iloc[idx-50:idx].mean()
        
        # RSI for momentum
        current_rsi = self._calculate_rsi(data['close'].iloc[idx-20:idx+1])
        
        # Volume confirmation
        avg_volume = data['volume'].iloc[idx-20:idx].mean()
        current_volume = data['volume'].iloc[idx]
        
        # Price momentum
        price_momentum = (data['close'].iloc[idx] - data['close'].iloc[idx-5]) / data['close'].iloc[idx-5]
        
        signal_strength = 0.0
        
        # Trend alignment (25% weight)
        if short_ma > medium_ma > long_ma:
            signal_strength += 0.25
            
        # RSI oversold but not extreme (20% weight)
        if 25 < current_rsi < 40:
            signal_strength += 0.20
            
        # Volume confirmation (25% weight)
        if current_volume > avg_volume * 1.3:
            signal_strength += 0.25
            
        # Positive momentum (30% weight)
        if price_momentum > 0.002:
            signal_strength += 0.30
            
        return signal_strength > self.config.confidence_threshold, signal_strength

    def _exit_signal(self, entry_price: float, current_price: float, 
                    hold_minutes: int, data: pd.DataFrame, idx: int) -> Tuple[bool, str]:
        """Improved exit signal with better risk management."""
        pnl_pct = (current_price - entry_price) / entry_price
        
        # Stop loss
        if pnl_pct <= -self.config.stop_loss_pct:
            return True, "stop_loss"
            
        # Take profit
        if pnl_pct >= self.config.take_profit_pct:
            return True, "take_profit"
            
        # Maximum hold time
        if hold_minutes >= self.config.max_hold_minutes:
            return True, "max_hold"
            
        # Momentum reversal (only check after minimum hold)
        if hold_minutes >= self.config.min_hold_minutes and idx >= 10:
            recent_momentum = (data['close'].iloc[idx] - data['close'].iloc[idx-5]) / data['close'].iloc[idx-5]
            if recent_momentum < -0.003:
                return True, "momentum_exit"
                
        return False, "hold"

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI for the most recent value."""
        if len(prices) < period + 1:
            return 50.0
            
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain.iloc[-1] / loss.iloc[-1] if loss.iloc[-1] != 0 else 0
        rsi = 100 - (100 / (1 + rs)) if rs != 0 else 50
        return rsi

    def run_backtest(self, start_date: str = None, end_date: str = None) -> Dict:
        """Run the backtest on historical data."""
        print(f"🚀 Starting backtest for {self.symbol}")
        
        # Get historical data
        if start_date and end_date:
            data = self.data_manager.get_market_data(self.symbol, '1min', start_date, end_date)
        else:
            end_dt = datetime.now()
            start_dt = end_dt - timedelta(days=30)
            data = self.data_manager.get_market_data(
                self.symbol, '1min', 
                start_dt.strftime('%Y-%m-%d'), 
                end_dt.strftime('%Y-%m-%d')
            )
        
        if data.empty:
            raise ValueError(f"No data available for {self.symbol}")
            
        print(f"📊 Backtesting {len(data)} data points from {data.index.min().strftime('%Y-%m-%d')} to {data.index.max().strftime('%Y-%m-%d')}")
        
        # Reset state
        self.balance = self.initial_balance
        self.position = None
        self.trades = []
        self.balance_history = []
        self.daily_trades = {}
        
        # Progress tracking
        total_points = len(data)
        last_progress = 0
        
        # Main backtesting loop
        for idx in range(len(data)):
            current_time = data.index[idx]
            current_price = data['close'].iloc[idx]
            
            # Progress indicator
            progress = int((idx / total_points) * 100)
            if progress >= last_progress + 20:
                print(f"⏳ Progress: {progress}%")
                last_progress = progress
            
            self.balance_history.append({
                'timestamp': current_time,
                'balance': self.balance,
                'price': current_price
            })
            
            if self.position is None:
                # Look for entry signals
                should_enter, confidence = self._improved_entry_signal(data, idx)
                
                if should_enter:
                    shares = self._calculate_position_size(current_price)
                    transaction_cost = self._calculate_transaction_cost(current_price, shares)
                    
                    if self.balance >= (current_price * shares + transaction_cost):
                        # Enter position
                        self.position = {
                            'entry_time': current_time,
                            'entry_price': current_price,
                            'shares': shares,
                            'entry_confidence': confidence
                        }
                        
                        self.balance -= (current_price * shares + transaction_cost)
                        
                        # Track daily trades
                        date_key = current_time.date()
                        self.daily_trades[date_key] = self.daily_trades.get(date_key, 0) + 1
                        
                        if self.verbose:
                            print(f"📈 ENTER: {shares} shares at ${current_price:.2f}")
                        
            else:
                # Check for exit signals
                hold_minutes = int((current_time - self.position['entry_time']).total_seconds() / 60)
                
                if hold_minutes >= self.config.min_hold_minutes:
                    should_exit, exit_reason = self._exit_signal(
                        self.position['entry_price'], current_price, hold_minutes, data, idx
                    )
                    
                    if should_exit:
                        # Exit position
                        shares = self.position['shares']
                        entry_price = self.position['entry_price']
                        transaction_cost = self._calculate_transaction_cost(current_price, shares)
                        
                        # Calculate P&L
                        gross_pnl = (current_price - entry_price) * shares
                        net_pnl = gross_pnl - transaction_cost
                        pnl_pct = (current_price - entry_price) / entry_price
                        
                        # Update balance
                        self.balance += (current_price * shares - transaction_cost)
                        
                        # Record trade
                        trade = Trade(
                            entry_time=self.position['entry_time'],
                            exit_time=current_time,
                            entry_price=entry_price,
                            exit_price=current_price,
                            shares=shares,
                            side='long',
                            entry_signal_confidence=self.position['entry_confidence'],
                            exit_signal_confidence=exit_reason,
                            pnl=net_pnl,
                            pnl_pct=pnl_pct,
                            hold_duration_minutes=hold_minutes
                        )
                        
                        self.trades.append(trade)
                        self.position = None
                        
                        if self.verbose:
                            print(f"📉 EXIT: {shares} shares at ${current_price:.2f} | P&L: ${net_pnl:.2f} ({pnl_pct:.2%}) | {exit_reason}")
        
        # Close any remaining position
        if self.position is not None:
            final_price = data['close'].iloc[-1]
            shares = self.position['shares']
            entry_price = self.position['entry_price']
            transaction_cost = self._calculate_transaction_cost(final_price, shares)
            
            gross_pnl = (final_price - entry_price) * shares
            net_pnl = gross_pnl - transaction_cost
            pnl_pct = (final_price - entry_price) / entry_price
            
            self.balance += (final_price * shares - transaction_cost)
            
            hold_minutes = int((data.index[-1] - self.position['entry_time']).total_seconds() / 60)
            
            trade = Trade(
                entry_time=self.position['entry_time'],
                exit_time=data.index[-1],
                entry_price=entry_price,
                exit_price=final_price,
                shares=shares,
                side='long',
                entry_signal_confidence=self.position['entry_confidence'],
                exit_signal_confidence="end_of_data",
                pnl=net_pnl,
                pnl_pct=pnl_pct,
                hold_duration_minutes=hold_minutes
            )
            
            self.trades.append(trade)
            self.position = None
        
        print("✅ Backtest completed")
        return self._calculate_performance_metrics()

    def _calculate_performance_metrics(self) -> Dict:
        """Calculate comprehensive performance metrics."""
        if not self.trades:
            return {
                'initial_balance': self.initial_balance,
                'final_balance': self.balance,
                'total_return': 0.0,
                'total_return_pct': 0.0,
                'num_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'avg_trade_pnl': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0,
                'max_drawdown': 0.0,
                'sharpe_ratio': 0.0,
                'avg_hold_time': 0.0,
                'exit_reasons': {},
                'best_trade': None,
                'worst_trade': None
            }
        
        # Basic metrics
        total_pnl = sum(trade.pnl for trade in self.trades)
        total_return_pct = (self.balance - self.initial_balance) / self.initial_balance
        
        # Trade statistics
        winning_trades = [t for t in self.trades if t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl <= 0]
        
        win_rate = len(winning_trades) / len(self.trades) if self.trades else 0
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
        avg_trade_pnl = total_pnl / len(self.trades)
        
        # Drawdown calculation
        running_max = self.initial_balance
        max_drawdown = 0
        
        for record in self.balance_history:
            running_max = max(running_max, record['balance'])
            drawdown = (running_max - record['balance']) / running_max
            max_drawdown = max(max_drawdown, drawdown)
        
        # Risk metrics
        returns = [t.pnl_pct for t in self.trades]
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if returns and np.std(returns) > 0 else 0
        
        # Exit reason analysis
        exit_reasons = {}
        for trade in self.trades:
            reason = trade.exit_signal_confidence
            exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
        
        return {
            'initial_balance': self.initial_balance,
            'final_balance': self.balance,
            'total_return': total_pnl,
            'total_return_pct': total_return_pct,
            'num_trades': len(self.trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'avg_trade_pnl': avg_trade_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': abs(avg_win / avg_loss) if avg_loss != 0 else 0,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'avg_hold_time': np.mean([t.hold_duration_minutes for t in self.trades]),
            'exit_reasons': exit_reasons,
            'best_trade': max(self.trades, key=lambda x: x.pnl) if self.trades else None,
            'worst_trade': min(self.trades, key=lambda x: x.pnl) if self.trades else None,
            'trades': self.trades
        }

    def print_summary(self, metrics: Dict):
        """Print concise backtesting summary."""
        print(f"\n{'='*60}")
        print(f"📈 BACKTEST RESULTS FOR {self.symbol}")
        print(f"{'='*60}")
        
        if metrics.get('num_trades', 0) == 0:
            print(f"\n⚠️  NO TRADES EXECUTED")
            print(f"  Initial Balance:     ${self.initial_balance:,.2f}")
            print(f"  Final Balance:       ${self.balance:,.2f}")
            print(f"  Total Return:        $0.00")
            print(f"  Total Return %:      0.00%")
            print(f"\n📊 Possible reasons:")
            print(f"  • Signal confidence threshold too high ({self.config.confidence_threshold})")
            print(f"  • No strong technical patterns found")
            print(f"  • Market conditions not suitable for strategy")
            return
        
        print(f"\n💰 PERFORMANCE")
        print(f"  Initial Balance:     ${metrics['initial_balance']:,.2f}")
        print(f"  Final Balance:       ${metrics['final_balance']:,.2f}")
        print(f"  Total Return:        ${metrics['total_return']:,.2f}")
        print(f"  Total Return %:      {metrics['total_return_pct']:+.2%}")
        print(f"  Max Drawdown:        {metrics['max_drawdown']:.2%}")
        
        print(f"\n📊 TRADING")
        print(f"  Total Trades:        {metrics['num_trades']}")
        print(f"  Win Rate:            {metrics['win_rate']:.1%}")
        print(f"  Avg Trade P&L:       ${metrics['avg_trade_pnl']:+.2f}")
        print(f"  Avg Win:             ${metrics['avg_win']:+.2f}")
        print(f"  Avg Loss:            ${metrics['avg_loss']:+.2f}")
        print(f"  Profit Factor:       {metrics['profit_factor']:.2f}")
        print(f"  Avg Hold Time:       {metrics['avg_hold_time']:.0f} minutes")
        
        # Exit reasons
        if metrics['exit_reasons']:
            print(f"\n🚪 EXIT REASONS")
            for reason, count in metrics['exit_reasons'].items():
                pct = count / metrics['num_trades'] * 100
                print(f"  {reason.replace('_', ' ').title():15} {count:3d} ({pct:.1f}%)")
        
        # Best/worst trades
        if metrics['best_trade']:
            best = metrics['best_trade']
            worst = metrics['worst_trade']
            print(f"\n🏆 BEST/WORST")
            print(f"  Best Trade:          ${best.pnl:+.2f} ({best.pnl_pct:+.2%})")
            print(f"  Worst Trade:         ${worst.pnl:+.2f} ({worst.pnl_pct:+.2%})")

def main():
    """Run backtesting."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run backtesting for trading models')
    parser.add_argument('--symbol', required=True, help='Stock symbol (e.g., META)')
    parser.add_argument('--days', type=int, default=30, help='Number of days to backtest')
    parser.add_argument('--balance', type=float, default=25000, help='Initial balance')
    parser.add_argument('--verbose', action='store_true', help='Show detailed trade logs')
    
    args = parser.parse_args()
    
    # Configuration with improved parameters
    config = BacktestConfig(
        initial_balance=args.balance,
        position_size_pct=0.95,
        confidence_threshold=0.8,
        min_hold_minutes=15,
        max_hold_minutes=120,
        max_trades_per_day=3
    )
    
    # Run backtest
    backtester = TradingBacktester(args.symbol, config, verbose=args.verbose)
    
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=args.days)
        
        print(f"📅 Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        print(f"💰 Initial Balance: ${config.initial_balance:,.2f}")
        
        metrics = backtester.run_backtest(
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
        
        # Print concise results
        backtester.print_summary(metrics)
        
    except Exception as e:
        print(f"❌ Backtesting failed: {e}")
        raise

if __name__ == "__main__":
    main()

