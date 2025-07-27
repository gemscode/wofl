#!/usr/bin/env python3

import warnings, random, time, json, argparse, sys, asyncio
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, Tuple, List
import numpy as np
import pandas as pd

"""
Enhanced Parameter Optimizer for Trading System with Interactive Brokers Support
Saves optimized parameters for any symbol that can be loaded by xetrader_trainer
"""

warnings.filterwarnings("ignore", message="Model is not converging")
warnings.filterwarnings("ignore", category=RuntimeWarning)

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from xetrader_trainer import (
    EnhancedAdvancedTradingSystem as TradingSystem,
    TradingConfig,
)

@dataclass
class OptimConf:
    target_annual: float = 0.35
    min_trades: int = 5
    max_dd: float = 0.20
    max_iter: int = 200
    n_jobs: int = 1
    data_source: str = 'alpha'
    ib_bar_size: str = '1 min'
    ib_host: str = '127.0.0.1'
    ib_port: int = 4002

LABELS = [(0.0015, -0.0015), (0.0020, -0.0015), (0.0025, -0.0020),
          (0.0030, -0.0025), (0.0035, -0.0030), (0.0040, -0.0035)]
CONF_T = [0.08, 0.12, 0.16, 0.20, 0.25, 0.30]
POS_SIZE = [0.04, 0.06, 0.08, 0.10, 0.12, 0.15]
PROF_T = [0.008, 0.012, 0.016, 0.020, 0.024, 0.028]
STOP_L = [0.006, 0.010, 0.014, 0.018, 0.022]
WINDOW = [10, 15, 20]

IB_BAR_SIZES = ['30 secs', '1 min', '2 mins', '5 mins']

class ResultsTracker:
    def __init__(self):
        self.all_results = []
        self.best_return = -999
        self.best_return_params = None
        self.best_return_bt = None
    
    def add_result(self, params, bt_results, score):
        result = {
            'params': params.copy(),
            'backtest': bt_results.copy(),
            'score': score,
            'timestamp': datetime.now().isoformat()
        }
        
        self.all_results.append(result)
        
        current_return = bt_results.get('total_return', -999)
        if current_return > self.best_return:
            self.best_return = current_return
            self.best_return_params = params.copy()
            self.best_return_bt = bt_results.copy()
    
    def get_best_by_return(self):
        return {
            'params': self.best_return_params,
            'backtest': self.best_return_bt,
            'return': self.best_return
        }
    
    def get_top_by_score(self, n=5):
        sorted_results = sorted(self.all_results, key=lambda x: x['score'], reverse=True)
        return sorted_results[:n]

def enhanced_score(bt: Dict, conf: OptimConf) -> float:
    trades = bt.get("total_trades", 0)
    if trades < conf.min_trades:
        return -1000
    
    total_return = bt.get("total_return", 0)
    dd = abs(bt.get("max_drawdown", 0))
    if dd > conf.max_dd:
        return -500
    
    test_days = 17
    daily_return = total_return / max(test_days, 1)
    annual_return = (1 + daily_return) ** 252 - 1
    
    base_score = annual_return * 1000
    trade_frequency_bonus = min(trades / 20, 1) * 100
    win_rate_bonus = (bt.get("win_rate", 0) - 40) * 2
    sharpe_bonus = min(max(bt.get("sharpe_ratio", 0), 0), 3) * 50
    dd_penalty = dd * 300
    
    return base_score + trade_frequency_bonus + win_rate_bonus + sharpe_bonus - dd_penalty

def label_fn_factory(up: float, dn: float):
    def _maker(df: pd.DataFrame):
        try:
            horizons = [1, 2, 3, 5]
            signals = []
            
            for h in horizons:
                if h < len(df):
                    future_returns = df["close"].shift(-h) / df["close"] - 1
                    signal = pd.Series(1, index=df.index)
                    signal[future_returns > up] = 2
                    signal[future_returns < dn] = 0
                    signals.append(signal)
            
            if signals:
                signal_matrix = pd.concat(signals, axis=1)
                final_signal = pd.Series(1, index=df.index)
                final_signal[(signal_matrix == 2).any(axis=1)] = 2
                final_signal[(signal_matrix == 0).any(axis=1)] = 0
                return final_signal.astype(int)
            else:
                return pd.Series(1, index=df.index).astype(int)
        
        except Exception as e:
            print(f"Label creation error: {e}")
            return pd.Series(1, index=df.index).astype(int)
    
    return _maker

def save_optimized_config(symbol: str, params: Dict, data_source: str = 'alpha'):
    test_return = params.get('best_return', 0)
    annual_proj = ((1 + test_return/17)**252 - 1) if test_return > -1 else -1
    
    ib_config_section = ""
    if data_source == 'ib':
        ib_bar_size = params.get('ib_bar_size', '1 min')
        ib_config_section = f"""
    data_source: str = 'ib'
    ib_bar_size: str = '{ib_bar_size}'
    ib_host: str = '127.0.0.1'
    ib_port: int = 4002
    ib_client_id: int = 3"""
    
    config_content = f'''#!/usr/bin/env python3

"""
Optimized Configuration for {symbol} - {annual_proj*100:.1f}% Annual Projection
Generated from {data_source.upper()} parameter optimization achieving {test_return:.2%} test return
Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""

from dataclasses import dataclass

@dataclass
class OptimizedTradingConfig:
    window_minutes: int = {params['win']}
    confidence_threshold: float = {params['conf']:.6f}
    base_position_size: float = {params['ps']:.6f}
    max_position_size: float = {min(params['ps']*2, 0.20):.6f}
    profit_target: float = {params['pt']:.6f}
    stop_loss: float = {params['sl']:.6f}
    
    label_threshold_up: float = {params['up']:.6f}
    label_threshold_down: float = {params['dn']:.6f}{ib_config_section}
    
    optimized_return: float = {test_return:.6f}
    annual_projection: float = {annual_proj:.6f}
    optimization_date: str = "{datetime.now().isoformat()}"
    data_source_optimized: str = "{data_source}"
    
    annual_target_return: float = 0.35
    max_daily_drawdown: float = 0.02
    lookback_periods: int = 100
    forecast_horizon: int = 20
    lstm_hidden_size: int = 128
    lstm_num_layers: int = 3
    dropout: float = 0.2
    data_split_exclude_days: int = 30
    min_volume_ratio: float = 1.2
    max_correlation_threshold: float = 0.7
    end_of_day_close_minutes: int = 15

def get_optimized_config():
    return OptimizedTradingConfig()

def load_config_for_symbol(symbol: str):
    return get_optimized_config()

OPTIMIZED_PARAMS = {{
    'window_minutes': {params['win']},
    'confidence_threshold': {params['conf']:.6f},
    'base_position_size': {params['ps']:.6f},
    'max_position_size': {min(params['ps']*2, 0.20):.6f},
    'profit_target': {params['pt']:.6f},
    'stop_loss': {params['sl']:.6f},
    'label_threshold_up': {params['up']:.6f},
    'label_threshold_down': {params['dn']:.6f},
    'data_source': '{data_source}',{f"""
    'ib_bar_size': '{params.get('ib_bar_size', '1 min')}',""" if data_source == 'ib' else ""}
}}

'''
    
    config_file = PROJECT_ROOT / f"optimized_config_{symbol.lower()}.py"
    with open(config_file, 'w') as f:
        f.write(config_content)
    
    json_file = PROJECT_ROOT / f"optimized_config_{symbol.lower()}.json"
    json_data = {
        'symbol': symbol,
        'optimization_date': datetime.now().isoformat(),
        'test_return': test_return,
        'annual_projection': annual_proj,
        'data_source': data_source,
        'parameters': {
            'window_minutes': params['win'],
            'confidence_threshold': params['conf'],
            'base_position_size': params['ps'],
            'max_position_size': min(params['ps']*2, 0.20),
            'profit_target': params['pt'],
            'stop_loss': params['sl'],
            'label_threshold_up': params['up'],
            'label_threshold_down': params['dn'],
            'data_source': data_source,
        }
    }
    
    if data_source == 'ib':
        json_data['parameters']['ib_bar_size'] = params.get('ib_bar_size', '1 min')
    
    with open(json_file, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"Optimized configuration saved to: {config_file}")
    print(f"JSON configuration saved to: {json_file}")

class EnhancedGAOptimizer:
    def __init__(self, symbol: str, conf: OptimConf):
        self.sym = symbol.upper()
        self.conf = conf
        self.results_tracker = ResultsTracker()
        self.best_score = -np.inf
        self.best_score_params = {}
        self.best_score_bt = {}
    
    async def eval(self, params: Dict) -> float:
        try:
            params["win"] = int(round(params["win"]))
            params["conf"] = max(0.05, min(0.50, params["conf"]))
            params["ps"] = max(0.02, min(0.20, params["ps"]))
            params["pt"] = max(0.005, min(0.050, params["pt"]))
            params["sl"] = max(0.004, min(0.030, params["sl"]))
            
            if self.conf.data_source == 'ib':
                params["ib_bar_size"] = params.get("ib_bar_size", self.conf.ib_bar_size)
            
            cfg = TradingConfig(
                window_minutes=params["win"],
                confidence_threshold=params["conf"],
                base_position_size=params["ps"],
                max_position_size=min(params["ps"] * 2, 0.20),
                profit_target=params["pt"],
                stop_loss=params["sl"],
                data_source=self.conf.data_source,
                ib_bar_size=params.get("ib_bar_size", "1 min"),
                ib_host=self.conf.ib_host,
                ib_port=self.conf.ib_port
            )
            
            ts = TradingSystem(self.sym, cfg)
            gen = ts.ensemble_generator
            fname = "_create_enhanced_labels"
            orig = getattr(gen, fname)
            
            setattr(gen, fname, label_fn_factory(params["up"], params["dn"]))
            
            try:
                res = await ts.train_system(total_days=90)
                bt = res["backtest_results"]
                
                if "backtest_period" not in bt:
                    bt["backtest_period"] = {"days": 17}
                
                sc = enhanced_score(bt, self.conf)
                self.results_tracker.add_result(params, bt, sc)
                
                if sc > self.best_score:
                    self.best_score = sc
                    self.best_score_params = params.copy()
                    self.best_score_bt = bt.copy()
                    
                    annual_proj = (1 + bt["total_return"]/17)**252 - 1
                    print(f"NEW BEST SCORE: {sc:.1f} | Return: {bt['total_return']:.2%} | "
                          f"Annual: {annual_proj:.1%} | Trades: {bt['total_trades']} | "
                          f"Source: {self.conf.data_source.upper()}")
                
                return sc
            
            finally:
                setattr(gen, fname, orig)
        
        except Exception as e:
            print(f"Eval error: {e}")
            return -1000
    
    async def run(self):
        from deap import base, creator, tools, algorithms
        
        if hasattr(creator, "Fitness"):
            del creator.Fitness
            del creator.Individual
        
        creator.create("Fitness", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.Fitness)
        
        toolbox = base.Toolbox()
        
        toolbox.register("up", lambda: random.uniform(0.0015, 0.0045))
        toolbox.register("dn", lambda: random.uniform(-0.0045, -0.0015))
        toolbox.register("conf", lambda: random.uniform(0.08, 0.35))
        toolbox.register("ps", lambda: random.uniform(0.04, 0.15))
        toolbox.register("pt", lambda: random.choice(PROF_T))
        toolbox.register("sl", lambda: random.choice(STOP_L))
        toolbox.register("win", lambda: random.choice(WINDOW))
        
        if self.conf.data_source == 'ib':
            toolbox.register("ib_bar_size", lambda: random.choice(IB_BAR_SIZES))
            toolbox.register("individual", tools.initCycle, creator.Individual,
                           (toolbox.up, toolbox.dn, toolbox.conf,
                            toolbox.ps, toolbox.pt, toolbox.sl, toolbox.win, toolbox.ib_bar_size), 1)
            
            def ib_crossover(ind1, ind2):
                for i in range(len(ind1) - 1):
                    if random.random() < 0.5:
                        alpha = 0.1
                        gamma = (1 + 2 * alpha) * random.random() - alpha
                        x1, x2 = ind1[i], ind2[i]
                        ind1[i] = (1. - gamma) * x1 + gamma * x2
                        ind2[i] = gamma * x1 + (1. - gamma) * x2
                
                if random.random() < 0.5:
                    ind1[-1], ind2[-1] = ind2[-1], ind1[-1]
                
                return ind1, ind2
            
            def ib_mutate(individual):
                for i in range(len(individual) - 1):
                    if random.random() < 0.4:
                        individual[i] += random.gauss(0, 0.05)
                
                if random.random() < 0.2:
                    individual[-1] = random.choice(IB_BAR_SIZES)
                
                return individual,
            
            toolbox.register("mate", ib_crossover)
            toolbox.register("mutate", ib_mutate)
            
        else:
            toolbox.register("individual", tools.initCycle, creator.Individual,
                           (toolbox.up, toolbox.dn, toolbox.conf,
                            toolbox.ps, toolbox.pt, toolbox.sl, toolbox.win), 1)
            toolbox.register("mate", tools.cxBlend, alpha=0.1)
            toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.05, indpb=0.4)
        
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        
        async def deap_eval(ind):
            if self.conf.data_source == 'ib':
                p = dict(up=ind[0], dn=ind[1], conf=ind[2],
                        ps=ind[3], pt=ind[4], sl=ind[5], win=int(ind[6]),
                        ib_bar_size=ind[7])
            else:
                p = dict(up=ind[0], dn=ind[1], conf=ind[2],
                        ps=ind[3], pt=ind[4], sl=ind[5], win=int(ind[6]))
            return await self.eval(p),
        
        async def async_evaluate_population(population):
            results = []
            for ind in population:
                fitness = await deap_eval(ind)
                ind.fitness.values = fitness
                results.append(ind)
            return results
        
        toolbox.register("select", tools.selTournament, tournsize=3)
        
        pop_size = 40
        generations = max(self.conf.max_iter // pop_size, 10)
        
        print(f"Running Enhanced GA with {self.conf.data_source.upper()} data:")
        print(f"    {pop_size} individuals × {generations} generations = {pop_size * generations} evaluations")
        if self.conf.data_source == 'ib':
            print(f"    IB bar size will be optimized among: {IB_BAR_SIZES}")
        
        pop = toolbox.population(n=pop_size)
        
        print("Evaluating initial population...")
        await async_evaluate_population(pop)
        
        for gen in range(generations):
            print(f"Generation {gen+1}/{generations}")
            
            offspring = toolbox.select(pop, len(pop))
            offspring = list(map(toolbox.clone, offspring))
            
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < 0.7:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
            
            for mutant in offspring:
                if random.random() < 0.3:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values
            
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            if invalid_ind:
                await async_evaluate_population(invalid_ind)
            
            pop[:] = offspring
        
        print("Optimization completed!")
    
    def print_enhanced_summary(self):
        print(f"\n{'='*80}")
        print(f"ENHANCED OPTIMIZATION COMPLETE FOR {self.sym}")
        print(f"Data Source: {self.conf.data_source.upper()}")
        print(f"{'='*80}")
        
        total_evals = len(self.results_tracker.all_results)
        print(f"Total evaluations: {total_evals}")
        print(f"Target annual return: {self.conf.target_annual:.0%}")
        
        best_return_result = self.results_tracker.get_best_by_return()
        if best_return_result['params']:
            print(f"\n{'='*60}")
            print(f"HIGHEST RETURN CONFIGURATION (PRIMARY FOCUS)")
            print(f"{'='*60}")
            
            bt = best_return_result['backtest']
            params = best_return_result['params']
            return_val = best_return_result['return']
            annual_proj = (1 + return_val/17)**252 - 1 if return_val > -1 else -1
            
            print(f"\nPERFORMANCE METRICS:")
            print(f"    Test Return: {return_val:8.2%}")
            print(f"    Annual Projection: {annual_proj:8.1%}")
            print(f"    Total Trades: {bt.get('total_trades', 0):8d}")
            print(f"    Win Rate: {bt.get('win_rate', 0):8.1f}%")
            print(f"    Sharpe Ratio: {bt.get('sharpe_ratio', 0):8.2f}")
            print(f"    Max Drawdown: {bt.get('max_drawdown', 0):8.2%}")
            
            print(f"\nCOMPLETE PARAMETER SET:")
            print(f"    Data source: {self.conf.data_source.upper()}")
            if self.conf.data_source == 'ib':
                print(f"    IB bar size: {params.get('ib_bar_size', 'N/A')}")
            print(f"    Label up thresh: {params['up']:8.4f}")
            print(f"    Label down thresh: {params['dn']:8.4f}")
            print(f"    Confidence thresh: {params['conf']:8.4f}")
            print(f"    Position size: {params['ps']:8.4f}")
            print(f"    Profit target: {params['pt']:8.4f}")
            print(f"    Stop loss: {params['sl']:8.4f}")
            print(f"    Window minutes: {params['win']:8d}")
            
            params['best_return'] = return_val
            save_optimized_config(self.sym, params, self.conf.data_source)
            
            print(f"\n{'='*60}")
            print(f"COPY-PASTE READY CONFIGURATION")
            print(f"{'='*60}")
            print("config = TradingConfig(")
            print(f"    window_minutes={params['win']},")
            print(f"    confidence_threshold={params['conf']:.4f},")
            print(f"    base_position_size={params['ps']:.4f},")
            print(f"    max_position_size={min(params['ps']*2, 0.20):.4f},")
            print(f"    profit_target={params['pt']:.4f},")
            print(f"    stop_loss={params['sl']:.4f},")
            print(f"    label_threshold_up={params['up']:.4f},")
            print(f"    label_threshold_down={params['dn']:.4f},")
            print(f"    data_source='{self.conf.data_source}',")
            if self.conf.data_source == 'ib':
                print(f"    ib_bar_size='{params.get('ib_bar_size', '1 min')}'")
            print(")")
            
            print(f"\nTEST COMMAND:")
            if self.conf.data_source == 'ib':
                print(f"python xetrader_trainer.py --symbol {self.sym} \\")
                print(f"    --window {params['win']} --train --ib \\")
                print(f"    --bar-size \"{params.get('ib_bar_size', '1 min')}\"")
            else:
                print(f"python xetrader_trainer.py --symbol {self.sym} \\")
                print(f"    --window {params['win']} --train")
            
            if annual_proj >= self.conf.target_annual:
                print(f"\nTARGET ACHIEVED! {annual_proj:.1%} annual projection with {self.conf.data_source.upper()} data")
            elif annual_proj >= 0.20:
                print(f"\nSTRONG PERFORMANCE: {annual_proj:.1%} annual projection with {self.conf.data_source.upper()} data")
            elif annual_proj >= 0.10:
                print(f"\nGOOD PROGRESS: {annual_proj:.1%} annual projection with {self.conf.data_source.upper()} data")
            elif annual_proj > 0:
                print(f"\nPOSITIVE: {annual_proj:.1%} annual projection with {self.conf.data_source.upper()} data")
            else:
                print(f"\nNEEDS IMPROVEMENT: {annual_proj:.1%} annual projection with {self.conf.data_source.upper()} data")
            
            print(f"\n{'='*60}")
            print(f"TOP 5 RESULTS BY SCORE")
            print(f"{'='*60}")
            top_results = self.results_tracker.get_top_by_score(5)
            
            headers = f"{'Rank':<4} {'Return':<8} {'Annual':<8} {'Trades':<7} {'Win%':<6} {'Score':<8}"
            if self.conf.data_source == 'ib':
                headers += f" {'BarSize':<8}"
            print(headers)
            print("-" * len(headers))
            
            for i, result in enumerate(top_results, 1):
                bt = result['backtest']
                ret = bt.get('total_return', 0)
                annual = (1 + ret/17)**252 - 1 if ret > -1 else -1
                
                line = f"{i:<4} {ret:>7.2%} {annual:>7.1%} {bt.get('total_trades',0):>7d} " \
                      f"{bt.get('win_rate',0):>5.1f} {result['score']:>8.1f}"
                
                if self.conf.data_source == 'ib':
                    bar_size = result['params'].get('ib_bar_size', 'N/A')
                    line += f" {bar_size:>8}"
                
                print(line)
        
        print(f"\n{'='*80}")

async def main():
    parser = argparse.ArgumentParser(description='Enhanced Parameter Optimizer with IB Support')
    parser.add_argument("--symbol", required=True, help='Stock symbol')
    parser.add_argument("--target-return", type=float, default=0.35, help='Target annual return')
    parser.add_argument("--max-iterations", type=int, default=200, help='Maximum iterations')
    parser.add_argument("--quick-test", action="store_true", help='Quick test with fewer iterations')
    
    parser.add_argument("--ib", action="store_true", help='Use Interactive Brokers data')
    parser.add_argument("--alpha", action="store_true", help='Use Alpha Vantage data (default)')
    parser.add_argument("--ib-host", default='127.0.0.1', help='IB Gateway host')
    parser.add_argument("--ib-port", type=int, default=4002, help='IB Gateway port')
    parser.add_argument("--ib-bar-size", default='1 min', help='Default IB bar size for optimization')
    
    args = parser.parse_args()
    
    if args.ib:
        data_source = 'ib'
    elif args.alpha:
        data_source = 'alpha'
    else:
        data_source = 'alpha'
    
    conf = OptimConf(
        target_annual=args.target_return,
        max_iter=args.max_iterations if not args.quick_test else 40,
        data_source=data_source,
        ib_bar_size=args.ib_bar_size,
        ib_host=args.ib_host,
        ib_port=args.ib_port
    )
    
    optimizer = EnhancedGAOptimizer(args.symbol, conf)
    
    print(f"Enhanced Parameter Optimizer for {args.symbol}")
    print(f"Data source: {data_source.upper()}")
    if data_source == 'ib':
        print(f"IB Gateway: {args.ib_host}:{args.ib_port}")
        print(f"Default bar size: {args.ib_bar_size}")
        print(f"Bar sizes to optimize: {IB_BAR_SIZES}")
    print(f"Target: {args.target_return:.0%} annual return")
    print(f"Max iterations: {conf.max_iter}")
    
    t0 = time.time()
    await optimizer.run()
    print(f"\nOptimization completed in {(time.time()-t0)/60:.1f} minutes")
    
    optimizer.print_enhanced_summary()

if __name__ == "__main__":
    asyncio.run(main())

