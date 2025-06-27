#!/usr/bin/env python3
"""
Trading Profile Manager
Creates and manages trading profiles for the multi-agent trading system
"""

import json
import argparse
import os
from datetime import datetime, timedelta
from pathlib import Path
import yfinance as yf

class TradingProfile:
    def __init__(self, budget_weekly=10000, profile_name="default"):
        self.profile_name = profile_name
        self.budget_weekly = budget_weekly
        self.budget_remaining = budget_weekly
        self.holdings = {}  # key: ticker, value: dict with qty, price, date
        self.trade_history = []
        self.created_date = datetime.now().isoformat()
        self.last_updated = datetime.now().isoformat()
        self.week_start = self._get_week_start()

    def _get_week_start(self):
        """Get the start of the current trading week (Monday)"""
        today = datetime.now()
        days_since_monday = today.weekday()
        week_start = today - timedelta(days=days_since_monday)
        return week_start.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()

    def add_holding(self, ticker, quantity, price_per_share, trade_type="BUY"):
        """Add or update holdings for a ticker"""
        ticker = ticker.upper()
        trade_value = quantity * price_per_share
        
        if trade_type == "BUY":
            if trade_value > self.budget_remaining:
                raise ValueError(f"Insufficient budget. Need ${trade_value:.2f}, have ${self.budget_remaining:.2f}")
            
            if ticker in self.holdings:
                # Update existing holding (average price calculation)
                current_qty = self.holdings[ticker]['quantity']
                current_price = self.holdings[ticker]['price_per_share']
                total_value = (current_qty * current_price) + trade_value
                new_qty = current_qty + quantity
                new_avg_price = total_value / new_qty
                
                self.holdings[ticker] = {
                    'quantity': new_qty,
                    'price_per_share': new_avg_price,
                    'last_trade_date': datetime.now().isoformat(),
                    'total_invested': total_value
                }
            else:
                # New holding
                self.holdings[ticker] = {
                    'quantity': quantity,
                    'price_per_share': price_per_share,
                    'last_trade_date': datetime.now().isoformat(),
                    'total_invested': trade_value
                }
            
            self.budget_remaining -= trade_value
            
        elif trade_type == "SELL":
            if ticker not in self.holdings:
                raise ValueError(f"Cannot sell {ticker}: not in holdings")
            
            if quantity > self.holdings[ticker]['quantity']:
                raise ValueError(f"Cannot sell {quantity} shares of {ticker}: only have {self.holdings[ticker]['quantity']}")
            
            # Update holdings after sale
            if quantity == self.holdings[ticker]['quantity']:
                # Selling all shares
                proceeds = quantity * price_per_share
                del self.holdings[ticker]
            else:
                # Partial sale
                proceeds = quantity * price_per_share
                self.holdings[ticker]['quantity'] -= quantity
                # Reduce total invested proportionally
                reduction_ratio = quantity / (self.holdings[ticker]['quantity'] + quantity)
                self.holdings[ticker]['total_invested'] *= (1 - reduction_ratio)
                self.holdings[ticker]['last_trade_date'] = datetime.now().isoformat()
            
            self.budget_remaining += proceeds
        
        # Record trade
        self.trade_history.append({
            'ticker': ticker,
            'type': trade_type,
            'quantity': quantity,
            'price_per_share': price_per_share,
            'total_value': trade_value,
            'timestamp': datetime.now().isoformat(),
            'budget_remaining': self.budget_remaining
        })
        
        self.last_updated = datetime.now().isoformat()

    def get_holding_info(self, ticker):
        """Get detailed information about a specific holding"""
        ticker = ticker.upper()
        if ticker not in self.holdings:
            return None
        
        holding = self.holdings[ticker]
        
        # Get current price
        try:
            current_price = self.get_current_price(ticker)
            current_value = holding['quantity'] * current_price
            unrealized_pnl = current_value - holding['total_invested']
            unrealized_pnl_pct = (unrealized_pnl / holding['total_invested']) * 100
        except:
            current_price = None
            current_value = None
            unrealized_pnl = None
            unrealized_pnl_pct = None
        
        return {
            'ticker': ticker,
            'quantity': holding['quantity'],
            'avg_price': holding['price_per_share'],
            'total_invested': holding['total_invested'],
            'current_price': current_price,
            'current_value': current_value,
            'unrealized_pnl': unrealized_pnl,
            'unrealized_pnl_pct': unrealized_pnl_pct,
            'last_trade_date': holding['last_trade_date']
        }

    def get_current_price(self, ticker):
        """Get current market price for a ticker"""
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period="1d", interval="1m").tail(1)
            return float(data['Close'].iloc[0])
        except:
            return None

    def can_buy(self, ticker, quantity, price_per_share):
        """Check if we can afford to buy the specified quantity"""
        required_budget = quantity * price_per_share
        return required_budget <= self.budget_remaining

    def can_sell(self, ticker, quantity):
        """Check if we have enough shares to sell"""
        ticker = ticker.upper()
        if ticker not in self.holdings:
            return False
        return quantity <= self.holdings[ticker]['quantity']

    def get_portfolio_summary(self):
        """Get a summary of the entire portfolio"""
        total_invested = sum(holding['total_invested'] for holding in self.holdings.values())
        total_current_value = 0
        
        holdings_summary = []
        for ticker, holding in self.holdings.items():
            holding_info = self.get_holding_info(ticker)
            holdings_summary.append(holding_info)
            if holding_info['current_value']:
                total_current_value += holding_info['current_value']
        
        total_unrealized_pnl = total_current_value - total_invested if total_current_value > 0 else 0
        
        return {
            'profile_name': self.profile_name,
            'budget_weekly': self.budget_weekly,
            'budget_remaining': self.budget_remaining,
            'budget_used': self.budget_weekly - self.budget_remaining,
            'total_invested': total_invested,
            'total_current_value': total_current_value,
            'total_unrealized_pnl': total_unrealized_pnl,
            'holdings_count': len(self.holdings),
            'holdings': holdings_summary,
            'total_trades': len(self.trade_history),
            'week_start': self.week_start,
            'last_updated': self.last_updated
        }

    def reset_weekly_budget(self):
        """Reset the weekly budget (call at start of new week)"""
        self.budget_remaining = self.budget_weekly
        self.week_start = self._get_week_start()
        self.last_updated = datetime.now().isoformat()

    def to_dict(self):
        """Convert profile to dictionary for saving"""
        return {
            'profile_name': self.profile_name,
            'budget_weekly': self.budget_weekly,
            'budget_remaining': self.budget_remaining,
            'holdings': self.holdings,
            'trade_history': self.trade_history,
            'created_date': self.created_date,
            'last_updated': self.last_updated,
            'week_start': self.week_start
        }

    def save_to_file(self, filepath=None):
        """Save profile to JSON file"""
        if filepath is None:
            filepath = f"trading_profile_{self.profile_name}.json"
        
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        
        print(f"✅ Trading profile saved to {filepath}")

    @staticmethod
    def load_from_file(filepath):
        """Load profile from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        profile = TradingProfile(
            budget_weekly=data['budget_weekly'],
            profile_name=data.get('profile_name', 'default')
        )
        profile.budget_remaining = data['budget_remaining']
        profile.holdings = data['holdings']
        profile.trade_history = data.get('trade_history', [])
        profile.created_date = data.get('created_date', datetime.now().isoformat())
        profile.last_updated = data['last_updated']
        profile.week_start = data.get('week_start', profile._get_week_start())
        
        return profile

class TradingProfileManager:
    """Manager class for handling multiple trading profiles"""
    
    def __init__(self, profiles_dir="profiles"):
        self.profiles_dir = Path(profiles_dir)
        self.profiles_dir.mkdir(exist_ok=True)

    def create_profile(self, name, budget_weekly):
        """Create a new trading profile"""
        profile = TradingProfile(budget_weekly=budget_weekly, profile_name=name)
        filepath = self.profiles_dir / f"{name}_profile.json"
        profile.save_to_file(filepath)
        return profile

    def load_profile(self, name):
        """Load an existing trading profile"""
        filepath = self.profiles_dir / f"{name}_profile.json"
        if not filepath.exists():
            raise FileNotFoundError(f"Profile '{name}' not found")
        return TradingProfile.load_from_file(filepath)

    def list_profiles(self):
        """List all available profiles"""
        profiles = []
        for file in self.profiles_dir.glob("*_profile.json"):
            profile_name = file.stem.replace("_profile", "")
            profiles.append(profile_name)
        return profiles

    def delete_profile(self, name):
        """Delete a trading profile"""
        filepath = self.profiles_dir / f"{name}_profile.json"
        if filepath.exists():
            filepath.unlink()
            print(f"✅ Profile '{name}' deleted")
        else:
            print(f"❌ Profile '{name}' not found")

def main():
    """Main CLI interface for trading profile management"""
    parser = argparse.ArgumentParser(description='Trading Profile Manager')
    parser.add_argument('action', choices=['create', 'load', 'list', 'buy', 'sell', 'status', 'reset'], 
                       help='Action to perform')
    parser.add_argument('--name', '-n', default='default', help='Profile name')
    parser.add_argument('--budget', '-b', type=float, default=10000, help='Weekly budget')
    parser.add_argument('--ticker', '-t', help='Stock ticker symbol')
    parser.add_argument('--quantity', '-q', type=int, help='Number of shares')
    parser.add_argument('--price', '-p', type=float, help='Price per share')
    
    args = parser.parse_args()
    
    manager = TradingProfileManager()
    
    if args.action == 'create':
        print(f"📊 Creating new trading profile: {args.name}")
        profile = manager.create_profile(args.name, args.budget)
        print(f"✅ Profile created with weekly budget: ${args.budget:,.2f}")
        
    elif args.action == 'load':
        try:
            profile = manager.load_profile(args.name)
            summary = profile.get_portfolio_summary()
            print(f"📊 Loaded profile: {args.name}")
            print(f"💰 Budget remaining: ${summary['budget_remaining']:,.2f}")
            print(f"📈 Holdings: {summary['holdings_count']} positions")
        except FileNotFoundError as e:
            print(f"❌ {e}")
            
    elif args.action == 'list':
        profiles = manager.list_profiles()
        print("📋 Available profiles:")
        for profile_name in profiles:
            print(f"  - {profile_name}")
            
    elif args.action == 'buy':
        if not all([args.ticker, args.quantity, args.price]):
            print("❌ Buy requires --ticker, --quantity, and --price")
            return
        
        try:
            profile = manager.load_profile(args.name)
            if profile.can_buy(args.ticker, args.quantity, args.price):
                profile.add_holding(args.ticker, args.quantity, args.price, "BUY")
                profile.save_to_file(manager.profiles_dir / f"{args.name}_profile.json")
                print(f"✅ Bought {args.quantity} shares of {args.ticker} at ${args.price:.2f}")
                print(f"💰 Budget remaining: ${profile.budget_remaining:,.2f}")
            else:
                required = args.quantity * args.price
                print(f"❌ Insufficient budget. Need ${required:,.2f}, have ${profile.budget_remaining:,.2f}")
        except Exception as e:
            print(f"❌ Error: {e}")
            
    elif args.action == 'sell':
        if not all([args.ticker, args.quantity, args.price]):
            print("❌ Sell requires --ticker, --quantity, and --price")
            return
        
        try:
            profile = manager.load_profile(args.name)
            if profile.can_sell(args.ticker, args.quantity):
                profile.add_holding(args.ticker, args.quantity, args.price, "SELL")
                profile.save_to_file(manager.profiles_dir / f"{args.name}_profile.json")
                print(f"✅ Sold {args.quantity} shares of {args.ticker} at ${args.price:.2f}")
                print(f"💰 Budget remaining: ${profile.budget_remaining:,.2f}")
            else:
                holding = profile.holdings.get(args.ticker.upper())
                available = holding['quantity'] if holding else 0
                print(f"❌ Cannot sell {args.quantity} shares. Available: {available}")
        except Exception as e:
            print(f"❌ Error: {e}")
            
    elif args.action == 'status':
        try:
            profile = manager.load_profile(args.name)
            summary = profile.get_portfolio_summary()
            
            print(f"\n📊 Trading Profile: {summary['profile_name']}")
            print(f"💰 Weekly Budget: ${summary['budget_weekly']:,.2f}")
            print(f"💵 Budget Remaining: ${summary['budget_remaining']:,.2f}")
            print(f"📈 Total Invested: ${summary['total_invested']:,.2f}")
            
            if summary['total_current_value'] > 0:
                print(f"💎 Current Value: ${summary['total_current_value']:,.2f}")
                print(f"📊 Unrealized P&L: ${summary['total_unrealized_pnl']:,.2f}")
            
            print(f"\n🏢 Holdings ({summary['holdings_count']} positions):")
            for holding in summary['holdings']:
                if holding:
                    pnl_str = f" (P&L: ${holding['unrealized_pnl']:,.2f})" if holding['unrealized_pnl'] else ""
                    print(f"  {holding['ticker']}: {holding['quantity']} shares @ ${holding['avg_price']:.2f}{pnl_str}")
            
            print(f"\n📅 Week Start: {summary['week_start'][:10]}")
            print(f"🔄 Last Updated: {summary['last_updated'][:19]}")
            
        except FileNotFoundError as e:
            print(f"❌ {e}")
            
    elif args.action == 'reset':
        try:
            profile = manager.load_profile(args.name)
            profile.reset_weekly_budget()
            profile.save_to_file(manager.profiles_dir / f"{args.name}_profile.json")
            print(f"✅ Weekly budget reset for profile: {args.name}")
            print(f"💰 New budget: ${profile.budget_weekly:,.2f}")
        except FileNotFoundError as e:
            print(f"❌ {e}")

if __name__ == "__main__":
    main()

