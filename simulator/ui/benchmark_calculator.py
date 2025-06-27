#!/usr/bin/env python3
"""
Benchmark Calculator
Calculates buy-and-hold and other benchmark comparisons
"""

class BenchmarkCalculator:
    """Calculates various trading benchmarks"""
    
    def __init__(self, data_manager):
        self.data_manager = data_manager
    
    def calculate_buy_and_hold(self, initial_budget, available_days):
        """Calculate buy-and-hold performance for comparison"""
        try:
            # Get first and last day prices from test period
            first_day = available_days[len(available_days) // 2 - 2]  # Same as test start
            last_day = available_days[len(available_days) // 2 + 2]   # Same as test end
            
            first_day_data = self.data_manager.load_day_data(first_day)
            last_day_data = self.data_manager.load_day_data(last_day)
            
            if not first_day_data or not last_day_data:
                return None
            
            start_price = first_day_data[0]['price']
            end_price = last_day_data[-1]['price']
            
            # Calculate shares that could be bought with initial budget
            shares = int((initial_budget * 0.9) // start_price)
            buy_hold_value = shares * end_price + (initial_budget - shares * start_price)
            buy_hold_return = ((buy_hold_value - initial_budget) / initial_budget) * 100
            
            return {
                'start_price': start_price,
                'end_price': end_price,
                'shares': shares,
                'final_value': buy_hold_value,
                'return_pct': buy_hold_return
            }
        except Exception as e:
            print(f"[ERROR] Failed to calculate buy-and-hold benchmark: {e}")
            return None
